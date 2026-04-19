"""Frame novelty clustering using HSV histograms.

Given a list of FrameData objects with local image paths, computes a
novelty-ranked ordering so downstream code can spend cloud-VLM budget
on semantically distinct frames first.

Pure CPU, no CUDA, no new dependencies — reuses appearance_signature.
"""
from __future__ import annotations
import logging
from typing import Callable, Optional

import cv2
import numpy as np

from backend.services.appearance_signature import (
    compute_appearance_signature, appearance_distance,
)

logger = logging.getLogger(__name__)


def _load_frame_signature(path: str) -> Optional[np.ndarray]:
    if not path:
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    return compute_appearance_signature(img)


def cluster_by_novelty(
    frames: list,
    distance_threshold: float = 0.35,
) -> list[list[int]]:
    """Greedy nearest-neighbor clustering.

    Returns a list of clusters, each a list of frame indices (into the
    input ``frames`` list) ordered by original timestamp. Frames that
    fail to load are placed in their own singleton cluster so downstream
    budget allocation still visits them.
    """
    sigs: list[Optional[np.ndarray]] = []
    for fr in frames:
        sigs.append(_load_frame_signature(getattr(fr, "path", "")))
    clusters: list[list[int]] = []
    cluster_centroids: list[np.ndarray] = []
    for i, sig in enumerate(sigs):
        if sig is None:
            clusters.append([i])
            # Zero-centroid sentinel so downstream distance checks skip this cluster.
            cluster_centroids.append(np.zeros_like(sigs[0]) if sigs and sigs[0] is not None else np.zeros(512, dtype=np.float32))
            continue
        best_c, best_d = -1, float("inf")
        for c_idx, centroid in enumerate(cluster_centroids):
            if centroid is None or centroid.sum() == 0:
                continue
            d = appearance_distance(sig, centroid)
            if d < best_d:
                best_d, best_c = d, c_idx
        if best_c >= 0 and best_d <= distance_threshold:
            clusters[best_c].append(i)
            n = len(clusters[best_c])
            cluster_centroids[best_c] = (cluster_centroids[best_c] * (n - 1) + sig) / n
        else:
            clusters.append([i])
            cluster_centroids.append(sig.copy())
    return clusters


def select_representatives(
    clusters: list[list[int]],
    importance_fn: Optional[Callable[[int], float]] = None,
) -> list[int]:
    """Return one representative frame index per cluster.

    Without importance_fn, picks the temporal median of each cluster
    (middle frame, tends to be the most stable). With importance_fn,
    picks the highest-scoring frame in each cluster.
    """
    reps = []
    for cluster in clusters:
        if not cluster:
            continue
        if importance_fn is None:
            reps.append(cluster[len(cluster) // 2])
        else:
            reps.append(max(cluster, key=importance_fn))
    return sorted(reps)


def rank_frames_by_novelty(
    frames: list,
    *,
    scene_cuts: Optional[list[float]] = None,
    face_conf_timeline: Optional[list[tuple[float, float]]] = None,
    audio_energy_timeline: Optional[list[tuple[float, float]]] = None,
    distance_threshold: float = 0.35,
) -> list[int]:
    """Return frame indices sorted by descending 'worth VLM budget' score.

    Score combines:
    - Cluster novelty (1.0 for representatives, 0.3 for members)
    - Scene-cut proximity (+0.4 within 1.5s of a cut)
    - Face-confidence peak (+0.3 when face_conf > 0.7)
    - Audio energy peak (+0.2 at local maxima)

    Callers take the top K indices as their "spend real VLM budget here"
    list.
    """
    import bisect
    if not frames:
        return []
    clusters = cluster_by_novelty(frames, distance_threshold=distance_threshold)
    reps = set(select_representatives(clusters))
    timestamps = [float(getattr(fr, "timestamp", 0) or 0) for fr in frames]
    scores = [0.0] * len(frames)
    for i in range(len(frames)):
        scores[i] = 1.0 if i in reps else 0.3
    if scene_cuts:
        cuts = sorted(float(t) for t in scene_cuts)
        for i, ts in enumerate(timestamps):
            idx = bisect.bisect_left(cuts, ts)
            nearest = float("inf")
            if idx < len(cuts):
                nearest = min(nearest, abs(cuts[idx] - ts))
            if idx > 0:
                nearest = min(nearest, abs(cuts[idx - 1] - ts))
            if nearest <= 1.5:
                scores[i] += 0.4
    if face_conf_timeline:
        fc_sorted = sorted(face_conf_timeline)
        fc_ts = [t for t, _ in fc_sorted]
        for i, ts in enumerate(timestamps):
            idx = bisect.bisect_left(fc_ts, ts)
            if idx < len(fc_sorted) and abs(fc_ts[idx] - ts) <= 1.0:
                if fc_sorted[idx][1] > 0.7:
                    scores[i] += 0.3
    if audio_energy_timeline:
        ae_sorted = sorted(audio_energy_timeline)
        ae_ts = [t for t, _ in ae_sorted]
        for i, ts in enumerate(timestamps):
            idx = bisect.bisect_left(ae_ts, ts)
            if idx < len(ae_sorted) and abs(ae_ts[idx] - ts) <= 1.0:
                if ae_sorted[idx][1] > 0.6:
                    scores[i] += 0.2
    return sorted(range(len(frames)), key=lambda i: (-scores[i], timestamps[i]))
