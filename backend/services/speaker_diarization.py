"""Whisper-diarization fusion helper for active speaker detection.

Provides a second independent vote for active-speaker attribution by
clustering speech into speaker turns from audio alone, then fusing
with the lip-motion timeline from ``active_speaker.py``. The two
signals are independent (one sees faces, the other sees waveforms)
so agreement is strong evidence and disagreement reliably flags
crosstalk frames.

Three tiers:
  1. pyannote.audio (preferred) — SOTA, needs HF token.
  2. MFCC + k-means (dependency-light fallback).
  3. No-op — returns empty list; fuser passes lip events through.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────── Feature flags ────────────────────

USE_DIARIZATION = os.environ.get(
    "CLIPAI_USE_DIARIZATION", "1",
).lower() in ("1", "true", "yes", "on")

DIARIZATION_BACKEND = os.environ.get(
    "CLIPAI_DIARIZATION_BACKEND", "auto",
).lower()

DIARIZATION_DEVICE = os.environ.get(
    "CLIPAI_DIARIZATION_DEVICE", "auto",
).lower()

MIN_VRAM_MB_FOR_CUDA = 600


# ──────────────────── Tuning constants ────────────────────

AGREE_CONFIDENCE_BOOST = 0.25
AGREE_CONFIDENCE_CAP = 0.95
DISAGREE_CONFIDENCE_CAP = 0.5
LIP_NOISE_FLOOR = 0.40
DIARIZATION_OVERRIDE_CONFIDENCE = 0.60
CLUSTER_TO_SLOT_THRESHOLD = 0.60


# ──────────────────── Dataclasses ────────────────────

@dataclass
class DiarizationSegment:
    start: float
    end: float
    cluster_id: int
    confidence: float = 0.7


@dataclass
class FusionResult:
    events: list = field(default_factory=list)
    cluster_to_slot: dict = field(default_factory=dict)
    agree_count: int = 0
    disagree_count: int = 0
    override_count: int = 0


# ──────────────────── Backend probing ────────────────────

def _cuda_free_mb() -> Optional[int]:
    """Return free VRAM in MB on the default CUDA device, or None when
    CUDA is unavailable. Silently swallows any torch import error so
    the probe stays sandbox-safe."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free_bytes, _ = torch.cuda.mem_get_info()
        return int(free_bytes / (1024 * 1024))
    except Exception:
        return None


def _resolve_device() -> str:
    """Pick the diarization device based on ``CLIPAI_DIARIZATION_DEVICE``
    + the live free-VRAM probe. Always returns ``"cpu"`` when the
    explicit override is ``"cpu"`` or when ``"auto"`` can't find
    enough free VRAM (``MIN_VRAM_MB_FOR_CUDA``)."""
    if DIARIZATION_DEVICE == "cpu":
        return "cpu"
    if DIARIZATION_DEVICE == "cuda":
        return "cuda"
    free_mb = _cuda_free_mb()
    if free_mb is not None and free_mb >= MIN_VRAM_MB_FOR_CUDA:
        return "cuda"
    return "cpu"


def _pyannote_available() -> bool:
    """True when the ``pyannote.audio`` package imports AND a HF token
    is reachable (env var or on-disk cache). Either condition
    missing → no pyannote tier."""
    try:
        import pyannote.audio  # noqa: F401
    except Exception:
        return False
    return bool(
        os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.path.exists(os.path.expanduser("~/.cache/huggingface/token"))
    )


def _select_backend() -> str:
    """Pick one of ``"pyannote"`` / ``"mfcc"`` / ``"off"`` per the
    current env flags + import probes. ``USE_DIARIZATION=0`` and the
    explicit ``off`` backend override everything else."""
    if not USE_DIARIZATION or DIARIZATION_BACKEND == "off":
        return "off"
    if DIARIZATION_BACKEND == "pyannote":
        return "pyannote" if _pyannote_available() else "off"
    if DIARIZATION_BACKEND == "mfcc":
        return "mfcc"
    if _pyannote_available():
        return "pyannote"
    try:
        import numpy  # noqa: F401
        return "mfcc"
    except Exception:
        return "off"


# ──────────────────── Pyannote tier ────────────────────

def _diarize_pyannote(
    audio_path: str, num_speakers: Optional[int] = None,
) -> list:
    """Run pyannote speaker-diarization-3.1 against ``audio_path``.

    Returns ``[]`` on any failure — caller falls through to the MFCC
    tier. ``num_speakers`` here is a NOISY LOWER BOUND from the face
    registry, not ground truth: face detection misses off-screen
    speakers, profile-only shots, dark scenes, and anime
    side-characters. We therefore convert it into a ``min_speakers``
    / ``max_speakers`` bracket instead of pinning pyannote to the
    exact count — the old ``num_speakers=`` kwarg forced pyannote
    into exactly that many clusters and was the dominant cause of
    the "every speaker is Speaker 1" bug when the registry only
    saw one face.
    """
    try:
        from pyannote.audio import Pipeline
        import torch

        token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=token,
        )
        device = _resolve_device()
        if device == "cuda":
            pipeline.to(torch.device("cuda"))
            logger.info(
                "[Diarization] pyannote loaded on CUDA (%.0f MB free)",
                _cuda_free_mb() or 0,
            )
        else:
            logger.info("[Diarization] pyannote loaded on CPU")

        kwargs = _pyannote_bound_kwargs(num_speakers)
        diarization = pipeline(audio_path, **kwargs)
        segments = []
        speaker_to_cluster: dict = {}
        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            if speaker_label not in speaker_to_cluster:
                speaker_to_cluster[speaker_label] = len(speaker_to_cluster)
            segments.append(DiarizationSegment(
                start=float(turn.start),
                end=float(turn.end),
                cluster_id=speaker_to_cluster[speaker_label],
                confidence=0.85,
            ))
        logger.info(
            "[Diarization] pyannote produced %d segments, %d clusters (bounds=%s)",
            len(segments), len(speaker_to_cluster), kwargs,
        )
        return segments
    except Exception as e:
        logger.warning("[Diarization] pyannote failed (%s) — falling back", e)
        return []


def _pyannote_bound_kwargs(num_speakers: Optional[int]) -> dict:
    """Translate a noisy ``num_speakers`` hint into a bracket.

    * ``num_speakers >= 2`` → ``min_speakers = max(1, N - 1)``,
      ``max_speakers = N + 2``. Wide enough that a missed face
      slot doesn't clamp the output.
    * ``num_speakers == 1`` → ``min_speakers = 1``,
      ``max_speakers = 3`` so a single detected face never forces
      pyannote into a monologue.
    * ``None`` / ``<= 0`` → return ``{}`` and let pyannote
      auto-detect.
    * Never return a ``num_speakers=`` key — that's the clamp we're
      trying to remove.
    """
    if num_speakers is None or num_speakers < 1:
        return {}
    if num_speakers == 1:
        return {"min_speakers": 1, "max_speakers": 3}
    return {
        "min_speakers": max(1, num_speakers - 1),
        "max_speakers": num_speakers + 2,
    }


# ──────────────────── MFCC + k-means fallback tier ────────────────────

def _extract_mfcc_for_intervals(
    audio_path, intervals, sample_rate=16000, n_mfcc=13,
):
    """Extract per-interval MFCC means+variances as a feature matrix.

    Returns ``(features, kept_intervals)`` where ``features`` is an
    (N, 2*n_mfcc) numpy array. On any failure (missing librosa, bad
    audio, too-short intervals) returns ``(None, [])`` so the caller
    can fall through to the no-op tier.
    """
    try:
        import numpy as np
        import librosa
    except Exception as e:
        logger.info("[Diarization] MFCC tier skipped (no librosa: %s)", e)
        return None, []
    try:
        y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    except Exception as e:
        logger.info("[Diarization] MFCC audio load failed (%s)", e)
        return None, []

    rows = []
    kept = []
    for (start, end) in intervals:
        if end - start < 0.3:
            continue
        i0, i1 = int(start * sample_rate), int(end * sample_rate)
        chunk = y[i0:i1]
        if len(chunk) < sample_rate // 10:
            continue
        try:
            mfcc = librosa.feature.mfcc(y=chunk, sr=sample_rate, n_mfcc=n_mfcc)
            feat = np.concatenate([mfcc.mean(axis=1), mfcc.var(axis=1)])
            rows.append(feat)
            kept.append((float(start), float(end)))
        except Exception:
            continue

    if not rows:
        return None, []
    return np.vstack(rows), kept


def _kmeans_assign(features, num_clusters: int):
    """Assign per-row cluster labels via sklearn KMeans (preferred) or
    a pure-numpy Lloyd's-algorithm fallback. Returns
    ``(labels, silhouette)`` where ``silhouette`` is the sklearn
    silhouette coefficient (or a numpy approximation).
    """
    try:
        import numpy as np
    except Exception:
        return None, 0.0

    n = features.shape[0]
    k = min(num_clusters, n)
    if k <= 1:
        return np.zeros(n, dtype=int), 0.0

    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(features)
        sil = float(silhouette_score(features, labels)) if k > 1 else 0.0
        return labels, sil
    except Exception:
        pass

    # Pure-numpy Lloyd's fallback (k-means++ init, 20 Lloyd iterations)
    rng = np.random.default_rng(42)
    centers = [features[rng.integers(n)]]
    for _ in range(k - 1):
        d = np.min(
            np.stack([np.sum((features - c) ** 2, axis=1) for c in centers]),
            axis=0,
        )
        probs = d / max(d.sum(), 1e-9)
        centers.append(features[rng.choice(n, p=probs)])
    centers = np.stack(centers)

    labels = np.zeros(n, dtype=int)
    for _ in range(20):
        dists = np.stack([np.sum((features - c) ** 2, axis=1) for c in centers])
        new_labels = np.argmin(dists, axis=0)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for ci in range(k):
            members = features[labels == ci]
            if len(members) > 0:
                centers[ci] = members.mean(axis=0)

    if k > 1:
        intra, inter = 0.0, 0.0
        for i in range(n):
            own = labels[i]
            own_mask = labels == own
            other_mask = ~own_mask
            if own_mask.sum() > 1:
                intra += float(np.mean(np.linalg.norm(
                    features[own_mask] - features[i], axis=1)))
            if other_mask.sum() > 0:
                inter += float(np.mean(np.linalg.norm(
                    features[other_mask] - features[i], axis=1)))
        intra /= n
        inter /= n
        sil = (inter - intra) / max(inter, intra, 1e-9)
    else:
        sil = 0.0
    return labels, float(sil)


def _diarize_mfcc(
    audio_path: str, num_speakers: Optional[int] = None,
) -> list:
    """MFCC + k-means fallback tier.

    Requires numpy + librosa. When ``num_speakers`` is absent we
    sweep k=2..6 and keep the k with the highest silhouette score.
    If the VAD probe shows a single continuous voiced region
    covering > 90% of the clip we treat it as a monologue and keep
    k=1; otherwise the floor is k=2 so a two-speaker clip whose
    face detection missed one speaker still ends up with two
    clusters.
    """
    try:
        from backend.services.active_speaker import build_vad_presence
    except Exception:
        logger.info("[Diarization] MFCC tier: build_vad_presence unavailable")
        return []

    intervals = build_vad_presence(audio_path)
    if not intervals:
        return []
    features, kept = _extract_mfcc_for_intervals(audio_path, intervals)
    if features is None or not kept:
        return []

    # Monologue probe: one continuous voiced region covering most of
    # the clip → leave it at k=1. Otherwise floor at 2 so a missed
    # face slot doesn't produce a single-cluster diarization.
    clip_start = kept[0][0]
    clip_end = kept[-1][1]
    voiced_total = sum(e - s for s, e in kept)
    clip_span = max(clip_end - clip_start, 1e-6)
    is_monologue = (
        len(kept) == 1
        or (len(kept) <= 2 and voiced_total / clip_span > 0.9)
    )

    # Derive the k to try. Respect a caller-provided bound of 2+ as
    # a LOWER bound (face registry saw N faces so there are at
    # least N-1 speakers). Never use it as the EXACT cluster count.
    if num_speakers is not None and num_speakers >= 2:
        lower = max(2, num_speakers - 1)
        upper = num_speakers + 2
    elif is_monologue:
        lower, upper = 1, 1
    else:
        lower, upper = 2, 6

    # Need at least ``upper`` voiced intervals to run ``upper``
    # clusters meaningfully.
    upper = min(upper, len(kept))
    lower = min(lower, upper)

    if lower == upper:
        labels, sil = _kmeans_assign(features, lower)
        chosen_k = lower
    else:
        best = (None, -2.0, lower)  # (labels, sil, k)
        for k in range(lower, upper + 1):
            labels_k, sil_k = _kmeans_assign(features, k)
            if labels_k is None:
                continue
            if sil_k > best[1]:
                best = (labels_k, sil_k, k)
        labels, sil, chosen_k = best
        if labels is None:
            # Fell off the sweep — default to k=lower as a safe
            # floor so we still return something useful.
            labels, sil = _kmeans_assign(features, lower)
            chosen_k = lower
    if labels is None:
        return []

    conf = max(0.3, min(0.75, 0.5 + sil * 0.5))
    segments = [
        DiarizationSegment(
            start=s, end=e, cluster_id=int(labels[i]), confidence=conf,
        )
        for i, (s, e) in enumerate(kept)
    ]
    logger.info(
        "[Diarization] MFCC produced %d segments, %d clusters (swept %d..%d), sil=%.2f",
        len(segments), chosen_k, lower, upper, sil,
    )
    return segments


# ──────────────────── Public diarization entry point ────────────────────

def diarize_audio(
    audio_path: str, num_speakers: Optional[int] = None,
) -> list:
    """Produce a ``list[DiarizationSegment]`` for ``audio_path`` using
    the best available backend. Returns ``[]`` when the file is
    missing, diarization is disabled, or every tier falls through.
    """
    if not audio_path or not os.path.exists(audio_path):
        return []

    backend = _select_backend()
    logger.info("[Diarization] backend=%s audio=%s", backend, audio_path)

    if backend == "off":
        return []
    if backend == "pyannote":
        segs = _diarize_pyannote(audio_path, num_speakers=num_speakers)
        if segs:
            return segs
        logger.info("[Diarization] pyannote returned empty; trying MFCC")
        backend = "mfcc"
    if backend == "mfcc":
        return _diarize_mfcc(audio_path, num_speakers=num_speakers)
    return []


# ──────────────────── Cluster → slot mapping ────────────────────

def map_clusters_to_slots(
    diarization_segments: list, lip_events: list,
) -> dict:
    """For each diarization cluster, pick the face slot whose lip
    events overlap it the most. Only return the mapping when the
    best-matching slot covers ≥ ``CLUSTER_TO_SLOT_THRESHOLD`` of the
    cluster's total time; otherwise map to ``-1`` (unknown)."""
    if not diarization_segments or not lip_events:
        return {}

    cluster_totals: dict = {}
    cluster_slot_overlap: dict = {}

    for diar in diarization_segments:
        dur = max(0.0, diar.end - diar.start)
        if dur <= 0:
            continue
        cluster_totals[diar.cluster_id] = (
            cluster_totals.get(diar.cluster_id, 0.0) + dur
        )
        per_slot = cluster_slot_overlap.setdefault(diar.cluster_id, {})
        for ev in lip_events:
            if ev.slot_id < 0:
                continue
            lo = max(diar.start, ev.start)
            hi = min(diar.end, ev.end)
            overlap = max(0.0, hi - lo)
            if overlap > 0:
                per_slot[ev.slot_id] = per_slot.get(ev.slot_id, 0.0) + overlap

    mapping: dict = {}
    for cid, per_slot in cluster_slot_overlap.items():
        total = cluster_totals.get(cid, 0.0)
        if total <= 0 or not per_slot:
            mapping[cid] = -1
            continue
        best_slot, best_overlap = max(per_slot.items(), key=lambda kv: kv[1])
        share = best_overlap / total
        mapping[cid] = best_slot if share >= CLUSTER_TO_SLOT_THRESHOLD else -1

    for cid in cluster_totals:
        mapping.setdefault(cid, -1)

    logger.info("[Diarization] cluster→slot mapping: %s", mapping)
    return mapping


def _slot_active_in_diarization(
    diarization_segments, cluster_to_slot, t_start, t_end,
):
    """Within ``[t_start, t_end)``, find the mapped slot with the
    largest diarization-time overlap and return
    ``(slot_id, max_confidence)``. Returns ``(-1, 0.0)`` when no
    mapped cluster overlaps the window.
    """
    if not diarization_segments or t_end <= t_start:
        return (-1, 0.0)

    slot_time: dict = {}
    slot_conf: dict = {}
    for diar in diarization_segments:
        slot = cluster_to_slot.get(diar.cluster_id, -1)
        if slot < 0:
            continue
        lo = max(diar.start, t_start)
        hi = min(diar.end, t_end)
        overlap = max(0.0, hi - lo)
        if overlap > 0:
            slot_time[slot] = slot_time.get(slot, 0.0) + overlap
            slot_conf[slot] = max(slot_conf.get(slot, 0.0), diar.confidence)

    if not slot_time:
        return (-1, 0.0)
    best_slot = max(slot_time, key=slot_time.get)
    return (best_slot, slot_conf[best_slot])


# ──────────────────── Public fusion entry point ────────────────────

def fuse_lip_and_diarization(
    lip_events: list, diarization_segments: list,
) -> FusionResult:
    """Merge the lip-motion ``SpeakerEvent`` timeline with the
    diarization timeline. Three per-event branches:

      * **agree** — lip slot == diar slot → confidence +=
        ``AGREE_CONFIDENCE_BOOST`` (capped at
        ``AGREE_CONFIDENCE_CAP``); counted in ``agree_count``.
      * **override** — lip slot != diar slot AND lip confidence is
        below ``LIP_NOISE_FLOOR`` AND diar confidence ≥ 0.6 →
        replace the slot_id with the diar vote; counted in
        ``override_count``.
      * **disagree** — lip slot != diar slot but lip is confident
        enough to keep its vote → confidence is clamped to
        ``DISAGREE_CONFIDENCE_CAP`` so downstream estimators can see
        the uncertainty; counted in ``disagree_count``.

    Events whose window has no mapped cluster are passed through
    unchanged. Empty diarization or empty lip events are both
    treated as pass-through.
    """
    result = FusionResult(events=list(lip_events))

    if not diarization_segments:
        return result
    if not lip_events:
        return result

    cluster_to_slot = map_clusters_to_slots(diarization_segments, lip_events)
    result.cluster_to_slot = cluster_to_slot

    if not cluster_to_slot or all(v < 0 for v in cluster_to_slot.values()):
        return result

    from backend.services.active_speaker import SpeakerEvent

    fused: list = []
    for ev in lip_events:
        diar_slot, diar_conf = _slot_active_in_diarization(
            diarization_segments, cluster_to_slot, ev.start, ev.end,
        )
        if diar_slot < 0:
            fused.append(ev)
            continue
        if ev.slot_id == diar_slot:
            fused.append(SpeakerEvent(
                start=ev.start, end=ev.end,
                slot_id=ev.slot_id,
                confidence=min(
                    AGREE_CONFIDENCE_CAP,
                    ev.confidence + AGREE_CONFIDENCE_BOOST,
                ),
                on_screen=getattr(ev, "on_screen", True),
            ))
            result.agree_count += 1
            continue
        if ev.confidence < LIP_NOISE_FLOOR and diar_conf >= 0.6:
            fused.append(SpeakerEvent(
                start=ev.start, end=ev.end,
                slot_id=diar_slot,
                confidence=DIARIZATION_OVERRIDE_CONFIDENCE,
                on_screen=getattr(ev, "on_screen", True),
            ))
            result.override_count += 1
            continue
        fused.append(SpeakerEvent(
            start=ev.start, end=ev.end,
            slot_id=ev.slot_id,
            confidence=min(ev.confidence, DISAGREE_CONFIDENCE_CAP),
            on_screen=getattr(ev, "on_screen", True),
        ))
        result.disagree_count += 1

    result.events = fused
    logger.info(
        "[Diarization] fusion: agree=%d disagree=%d override=%d (of %d events)",
        result.agree_count, result.disagree_count, result.override_count,
        len(lip_events),
    )
    return result
