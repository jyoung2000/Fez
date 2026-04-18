"""Subject track fusion: unifies face detection, face mesh, and saliency
into a single SubjectTrack abstraction with cross-shot identity.

Pipeline:
  1. Ingest face tracks from FaceRegistry as source="face_confirmed"
  2. Ingest saliency regions, cluster by spatial proximity + temporal continuity
  3. Run promotion gate on each saliency cluster
  4. Promoted clusters become source="face_like_promoted" SubjectTracks
  5. Compute appearance signatures for all tracks
  6. Resolve cross-shot identity via signature matching
"""

import logging
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np

from backend.services.focus_model import SubjectTrack
from backend.services.appearance_signature import (
    compute_appearance_signature,
    appearance_distance,
    APPEARANCE_MATCH_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Promotion gate thresholds
MIN_PERSISTENCE_FRAMES = 5
MIN_ASPECT_RATIO = 1.0           # h/w -- at least square
MAX_ASPECT_RATIO = 1.8           # at most 1.8x taller than wide
MIN_AREA_RATIO = 0.005           # 0.5% of frame
MAX_AREA_RATIO = 0.15            # 15% of frame

# Base confidence for promoted tracks (before mesh boost)
PROMOTED_BASE_CONFIDENCE = 0.60
# Confidence for face-confirmed tracks
CONFIRMED_BASE_CONFIDENCE = 0.95


def build_subject_tracks(
    face_registry,
    dense_faces: list,
    saliency_regions: list,
    frame_paths: list,
    source_width: int,
    source_height: int,
    shot_cuts: list = None,
    job_id: str = "",
) -> list:
    """Build unified SubjectTracks from all available signals.

    Args:
        face_registry: Existing FaceRegistry (untouched). Source for confirmed face tracks.
        dense_faces: list[FrameFaces] from face_detector -- gives per-frame bboxes with slot_id
        saliency_regions: list[SaliencyRegion] from saliency_tracker
        frame_paths: list[(timestamp, path)] for cropping during signature + mesh validation
        source_width, source_height: actual source dimensions
        shot_cuts: list[float] shot boundary timestamps for cross-shot matching
        job_id: logging correlation

    Returns:
        list[SubjectTrack]
    """
    _log = lambda msg, *a: logger.info("[%s] SubjectFusion: " + msg, job_id, *a)

    tracks = []
    next_track_id = 0

    # Build a timestamp -> frame_path lookup for crop extraction
    frame_lookup = {round(t, 3): p for t, p in frame_paths} if frame_paths else {}

    # -- Phase 1: Ingest face-confirmed tracks from FaceRegistry --
    if face_registry and hasattr(face_registry, 'slots') and dense_faces:
        by_slot = defaultdict(list)
        for df in dense_faces:
            for f in df.faces:
                if not getattr(f, 'is_human', True):
                    continue
                sid = getattr(f, 'identity_id', -1)
                if sid < 0:
                    continue
                fx = getattr(f, 'nose_x', getattr(f, 'x', 50))
                fy = getattr(f, 'nose_y', getattr(f, 'y', 50))
                fw = getattr(f, 'width', 10)
                fh = getattr(f, 'height', fw * 1.3)
                by_slot[sid].append((df.timestamp, float(fx), float(fy),
                                     float(fw), float(fh)))

        for slot_id, bboxes in by_slot.items():
            if len(bboxes) < 2:
                continue
            bboxes.sort(key=lambda b: b[0])
            tracks.append(SubjectTrack(
                track_id=next_track_id,
                source="face_confirmed",
                confidence=CONFIRMED_BASE_CONFIDENCE,
                face_slot_id=int(slot_id),
                bbox_trajectory=bboxes,
            ))
            next_track_id += 1

    _log("ingested %d face-confirmed tracks", len([t for t in tracks if t.source == "face_confirmed"]))

    # -- Phase 2: Cluster saliency regions into candidate tracks --
    saliency_clusters = _cluster_saliency_regions(saliency_regions or [], shot_cuts or [])
    _log("clustered saliency into %d candidate tracks", len(saliency_clusters))

    # -- Phase 3: Run promotion gate on each cluster --
    promoted_count = 0
    for cluster in saliency_clusters:
        track = _promote_cluster(
            cluster, frame_lookup, source_width, source_height,
            next_track_id, job_id,
        )
        if track is not None:
            tracks.append(track)
            next_track_id += 1
            promoted_count += 1

    _log("promoted %d saliency clusters to face_like tracks", promoted_count)

    # -- Phase 4: Compute appearance signatures --
    _compute_all_signatures(tracks, frame_lookup, source_width, source_height)

    # -- Phase 5: Resolve cross-shot identity --
    _resolve_persistent_identity(tracks)

    # Summary
    n_confirmed = sum(1 for t in tracks if t.source == "face_confirmed")
    n_promoted = sum(1 for t in tracks if t.source == "face_like_promoted")
    n_persistent = len(set(t.persistent_id for t in tracks if t.persistent_id >= 0))
    _log("FINAL: %d tracks (%d confirmed, %d promoted), %d persistent identities",
         len(tracks), n_confirmed, n_promoted, n_persistent)

    return tracks


def _cluster_saliency_regions(saliency_regions: list, shot_cuts: list) -> list:
    """Group saliency regions by spatial proximity + temporal continuity.

    Returns list[list[SaliencyRegion]] -- each inner list is one candidate cluster.
    Clusters cannot span shot boundaries.
    """
    if not saliency_regions:
        return []

    sorted_regions = sorted(saliency_regions, key=lambda r: r.timestamp)

    clusters = []
    active_clusters = []  # list[list[SaliencyRegion]]

    for region in sorted_regions:
        # Close active clusters if we've crossed a shot boundary
        for cut in shot_cuts:
            if any(c[-1].timestamp < cut <= region.timestamp for c in active_clusters if c):
                clusters.extend(active_clusters)
                active_clusters = []
                break

        # Try to extend an existing active cluster via spatial IoU
        matched = False
        for cluster in active_clusters:
            last = cluster[-1]
            # Temporal gap check -- allow up to 0.5s gap
            if region.timestamp - last.timestamp > 0.5:
                continue
            # Spatial IoU check
            if _region_iou(last, region) > 0.3:
                cluster.append(region)
                matched = True
                break

        if not matched:
            active_clusters.append([region])

    clusters.extend(active_clusters)
    return clusters


def _region_iou(r1, r2) -> float:
    """IoU between two SaliencyRegion bboxes (in % coordinates)."""
    l1, r1x = r1.x - r1.w / 2, r1.x + r1.w / 2
    t1, b1 = r1.y - r1.h / 2, r1.y + r1.h / 2
    l2, r2x = r2.x - r2.w / 2, r2.x + r2.w / 2
    t2, b2 = r2.y - r2.h / 2, r2.y + r2.h / 2

    inter_l = max(l1, l2)
    inter_r = min(r1x, r2x)
    inter_t = max(t1, t2)
    inter_b = min(b1, b2)
    if inter_r <= inter_l or inter_b <= inter_t:
        return 0.0
    inter_area = (inter_r - inter_l) * (inter_b - inter_t)
    area1 = (r1x - l1) * (b1 - t1)
    area2 = (r2x - l2) * (b2 - t2)
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


def _ema_smooth_trajectory(
    trajectory: list,
    alpha: float = 0.3,
) -> list:
    """Apply EMA smoothing to a bbox trajectory.

    Smooths (x, y, w, h) independently with exponential moving average.
    Timestamp values are preserved unchanged.

    Args:
        trajectory: list[(t, x, y, w, h)]
        alpha: EMA smoothing factor. Lower = smoother. 0.3 gives good
            jitter suppression without excessive lag.

    Returns:
        Smoothed trajectory in the same format.
    """
    if len(trajectory) <= 1:
        return list(trajectory)

    smoothed = [trajectory[0]]
    sx, sy, sw, sh = trajectory[0][1], trajectory[0][2], trajectory[0][3], trajectory[0][4]

    for i in range(1, len(trajectory)):
        t, x, y, w, h = trajectory[i]
        sx = alpha * x + (1 - alpha) * sx
        sy = alpha * y + (1 - alpha) * sy
        sw = alpha * w + (1 - alpha) * sw
        sh = alpha * h + (1 - alpha) * sh
        smoothed.append((t, float(sx), float(sy), float(sw), float(sh)))

    return smoothed


def _promote_cluster(
    cluster: list,
    frame_lookup: dict,
    source_width: int,
    source_height: int,
    track_id: int,
    job_id: str,
) -> Optional[SubjectTrack]:
    """Run the 4-test promotion gate on a saliency cluster.

    Returns a SubjectTrack if promoted, None otherwise.
    """
    if len(cluster) < MIN_PERSISTENCE_FRAMES:
        return None

    # Test 1: Persistence (already checked above)
    persistence = len(cluster)

    # Test 2: Aspect ratio (use cluster median)
    aspects = [r.h / r.w if r.w > 0 else 0 for r in cluster]
    median_aspect = float(np.median(aspects))
    if not (MIN_ASPECT_RATIO <= median_aspect <= MAX_ASPECT_RATIO):
        logger.debug("[%s] cluster rejected: aspect %.2f outside [%.1f, %.1f]",
                     job_id, median_aspect, MIN_ASPECT_RATIO, MAX_ASPECT_RATIO)
        return None

    # Test 3: Size (median area ratio)
    areas = [(r.w * r.h) / 10000.0 for r in cluster]  # % * % / 10000 = fraction
    median_area = float(np.median(areas))
    if not (MIN_AREA_RATIO <= median_area <= MAX_AREA_RATIO):
        logger.debug("[%s] cluster rejected: area %.4f outside [%.4f, %.4f]",
                     job_id, median_area, MIN_AREA_RATIO, MAX_AREA_RATIO)
        return None

    # Test 4: Face mesh validation (soft -- modulates confidence only)
    mesh_passed = False
    confidence = PROMOTED_BASE_CONFIDENCE
    try:
        from backend.services.face_mesh_validator import get_validator
        validator = get_validator()
        if validator.available:
            best_region = max(cluster, key=lambda r: r.saliency_score)
            crop = _extract_crop(best_region, frame_lookup, source_width, source_height)
            if crop is not None:
                found, boost = validator.validate(crop)
                if found:
                    mesh_passed = True
                    confidence += boost
    except Exception as e:
        logger.debug("[%s] mesh validation failed (non-fatal): %s", job_id, e)

    # Build trajectory from all cluster regions — apply EMA smoothing
    raw_trajectory = [(r.timestamp, r.x, r.y, r.w, r.h) for r in cluster]
    trajectory = _ema_smooth_trajectory(raw_trajectory, alpha=0.3)

    track = SubjectTrack(
        track_id=track_id,
        source="face_like_promoted",
        confidence=min(confidence, 1.0),
        face_slot_id=None,
        bbox_trajectory=trajectory,
        raw_trajectory=raw_trajectory,
        gate_persistence_frames=persistence,
        gate_aspect_ratio=median_aspect,
        gate_area_ratio=median_area,
        gate_face_mesh_passed=mesh_passed,
    )

    logger.info("[%s] PROMOTED saliency cluster -> track %d: persist=%d aspect=%.2f "
                "area=%.4f mesh=%s conf=%.2f",
                job_id, track_id, persistence, median_aspect, median_area,
                mesh_passed, confidence)
    return track


def _extract_crop(region, frame_lookup, source_width, source_height) -> Optional[np.ndarray]:
    """Extract the bounding box region from a source frame."""
    frame_path = frame_lookup.get(round(region.timestamp, 3))
    if frame_path is None:
        # Try nearest timestamp
        for t, p in frame_lookup.items():
            if abs(t - region.timestamp) < 0.1:
                frame_path = p
                break
    if frame_path is None:
        return None

    img = cv2.imread(str(frame_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    cx = int(region.x / 100 * w)
    cy = int(region.y / 100 * h)
    cw = int(region.w / 100 * w)
    ch = int(region.h / 100 * h)

    x1 = max(0, cx - cw // 2)
    y1 = max(0, cy - ch // 2)
    x2 = min(w, x1 + cw)
    y2 = min(h, y1 + ch)

    if x2 <= x1 or y2 <= y1:
        return None

    return img[y1:y2, x1:x2]


def _compute_all_signatures(tracks: list, frame_lookup: dict,
                            source_width: int, source_height: int) -> None:
    """Compute appearance signatures for all tracks using their median frame."""
    for track in tracks:
        if not track.bbox_trajectory:
            continue
        mid = len(track.bbox_trajectory) // 2
        t, x, y, w, h = track.bbox_trajectory[mid]

        class _R:
            pass
        r = _R()
        r.timestamp = t
        r.x = x
        r.y = y
        r.w = w
        r.h = h

        crop = _extract_crop(r, frame_lookup, source_width, source_height)
        if crop is not None:
            track.appearance_signature = compute_appearance_signature(crop)


def _resolve_persistent_identity(tracks: list) -> None:
    """Assign persistent_id across tracks using appearance signature matching.

    Tracks with matching signatures (Bhattacharyya < threshold) share a persistent_id.
    """
    next_pid = 0
    for i, track in enumerate(tracks):
        if track.persistent_id >= 0:
            continue
        if track.appearance_signature is None:
            track.persistent_id = next_pid
            next_pid += 1
            continue

        matched_pid = None
        best_dist = APPEARANCE_MATCH_THRESHOLD
        for other in tracks[:i]:
            if other.persistent_id < 0 or other.appearance_signature is None:
                continue
            dist = appearance_distance(track.appearance_signature,
                                       other.appearance_signature)
            if dist < best_dist:
                best_dist = dist
                matched_pid = other.persistent_id

        if matched_pid is not None:
            track.persistent_id = matched_pid
        else:
            track.persistent_id = next_pid
            next_pid += 1
