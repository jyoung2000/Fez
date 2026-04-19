"""Unified per-frame intent track (Fix 3.3).

Purpose
-------
The codebase had five different "subject position" representations that
disagreed (nose_x percent, saliency centroid fraction, SceneDescription
subject_x, subject_track forward-fill, FaceFrame2D fraction). The 2-D
solver consumed a blend that silently fell through to (50, 50) center
when no source was present — pinning the camera to center on every bad
frame and producing chin / head / off-center clips when the subject
reappeared.

This module builds a single authoritative per-frame intent track
``list[IntentSample]`` from:

  1. ``active_speaker`` — SpeakerEvent covering ``t`` (highest priority)
  2. dense face detections closest to ``t``
  3. saliency peak within window
  4. YOLO object detection within window
  5. Kalman-style hold of last real sample with decaying confidence
     (tagged ``source="kalman_predict"``) — NEVER a raw center default

The track NEVER emits ``(0.5, 0.5, conf <= 0.1, source="none")``. When
all real sources fail for > 0.5 s we hold the previous real sample and
tag the source so downstream consumers (critic, adapter widen-on-
uncertainty) can detect it.

Sampling at 10 Hz (matching the 2-D solver's frame grid) instead of
the legacy 2 Hz (which under-sampled the solver and produced aliased
motion).

See :mod:`backend.services.subject_track` for the legacy sparse-hold
implementation this replaces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class IntentSample:
    """One per-frame subject-intent sample.

    Coordinates are source-frame fractions in ``[0, 1]``. Legacy
    percent (0..100) convention is normalized at the source-read
    boundary inside this module.
    """
    t: float
    x: float
    y: float
    conf: float
    source: str         # "active_speaker" | "face" | "saliency" | "object" | "kalman_predict"
    motion_class: str   # "static" | "pan" | "dynamic" | "chaotic"
    shot_idx: int       # which shot this sample belongs to


# ── Helpers ───────────────────────────────────────────────────────


def _shot_idx_at(shot_boundaries: list, t: float) -> int:
    """Return the 0-based shot index covering ``t``. Shots are
    [0, b0], [b0, b1], ..., [bN-1, inf]."""
    idx = 0
    for b in shot_boundaries:
        if t < b:
            return idx
        idx += 1
    return idx


def _motion_for_shot(shot_idx: int, profiles: list) -> str:
    """Return motion_class for a shot_idx, 'static' if unknown."""
    if not profiles:
        return "static"
    if 0 <= shot_idx < len(profiles):
        return profiles[shot_idx].motion_class
    return "static"


def _active_speaker_xy_at(events: list, t: float, dense_faces: list) -> Optional[tuple]:
    """Return (x_frac, y_frac, conf) for the active-speaker face at t,
    or None if no event covers t or no matching face exists."""
    if not events:
        return None
    for ev in events:
        if not (ev.start <= t <= ev.end):
            continue
        slot = getattr(ev, "slot_id", -1)
        if slot < 0:
            continue
        # Find the dense_faces frame closest to t that contains this slot.
        best = None
        best_dt = 0.2
        for df in dense_faces or []:
            dt = abs(getattr(df, "timestamp", 0.0) - t)
            if dt > best_dt:
                continue
            for f in df.faces or []:
                if getattr(f, "identity_id", -1) == slot:
                    best = f
                    best_dt = dt
        if best is not None:
            return (
                float(best.nose_x) / 100.0,
                float(best.nose_y) / 100.0,
                0.95,
            )
    return None


def _face_xy_at(dense_faces: list, t: float, window: float = 0.15) -> Optional[tuple]:
    """Return (x_frac, y_frac, conf) from the nearest dense face frame,
    picking the largest face in that frame."""
    best_frame = None
    best_dt = window
    for df in dense_faces or []:
        if not df.faces:
            continue
        dt = abs(getattr(df, "timestamp", 0.0) - t)
        if dt <= best_dt:
            best_frame = df
            best_dt = dt
    if best_frame is None:
        return None
    faces = [f for f in best_frame.faces if getattr(f, "is_human", True)]
    if not faces:
        return None
    primary = max(faces, key=lambda f: f.width * f.height)
    return (
        float(primary.nose_x) / 100.0,
        float(primary.nose_y) / 100.0,
        0.75,
    )


def _saliency_xy_at(regions: list, t: float, window: float = 0.5) -> Optional[tuple]:
    """Return (x_frac, y_frac, conf) from the closest saliency region."""
    best = None
    best_dt = window
    for r in regions or []:
        dt = abs(getattr(r, "timestamp", 0.0) - t)
        if dt <= best_dt:
            best = r
            best_dt = dt
    if best is None:
        return None
    return (
        float(best.x) / 100.0,
        float(best.y) / 100.0,
        0.5,
    )


def _object_xy_at(objects: list, t: float, window: float = 0.5) -> Optional[tuple]:
    """Return (x_frac, y_frac, conf) from the closest object detection."""
    best = None
    best_dt = window
    for o in objects or []:
        dt = abs(getattr(o, "timestamp", 0.0) - t)
        if dt <= best_dt:
            best = o
            best_dt = dt
    if best is None:
        return None
    return (
        float(getattr(best, "x", 50.0)) / 100.0,
        float(getattr(best, "y", 50.0)) / 100.0,
        0.45,
    )


# ── Public API ────────────────────────────────────────────────────


def build_intent_track(
    *,
    duration_sec: float,
    shot_boundaries: list,
    shot_profiles: list,
    dense_faces: list,
    active_speaker_events: list,
    saliency_regions: Optional[list] = None,
    object_detections: Optional[list] = None,
    sample_hz: float = 10.0,
) -> list[IntentSample]:
    """Build a dense per-frame intent track.

    Never emits (0.5, 0.5, conf<=0.1, source='none'). When all real
    sources fail for a window longer than 0.5 s, the last real sample
    is held at its ORIGINAL (x, y) with decaying confidence and the
    sample is tagged ``source='kalman_predict'``.

    Across shot boundaries: the previous shot's predicted hold is
    NOT carried into the next shot. Each shot starts fresh — leading
    no-source frames are backfilled from the first real sample within
    the shot (not the previous shot's tail).

    Args:
        duration_sec: clip duration. Sampling goes 0..duration_sec
            in ``1/sample_hz`` steps.
        shot_boundaries: cut timestamps (N cuts → N+1 shots).
        shot_profiles: list[ShotProfile] from classify_shots (optional
            but strongly recommended — drives motion_class per sample).
        dense_faces: list[FrameFaces].
        active_speaker_events: list[SpeakerEvent].
        saliency_regions: optional list[SaliencyRegion].
        object_detections: optional list[ObjectDetection-like] with
            ``.x``, ``.y``, ``.timestamp`` attributes (0..100 percent).
        sample_hz: output sample rate (default 10 Hz — matches the
            2-D solver's frame grid).

    Returns:
        list[IntentSample] at sample_hz, length ~= duration * hz.
    """
    if duration_sec <= 0:
        return []

    step = 1.0 / max(0.5, float(sample_hz))

    # Group by shot so cross-shot samples can't be inferred from the
    # previous shot's tail.
    cuts = sorted(shot_boundaries or [])
    shots_spans: list[tuple[float, float, int]] = []
    prev_end = 0.0
    for idx, cut in enumerate(cuts):
        shots_spans.append((prev_end, cut, idx))
        prev_end = cut
    shots_spans.append((prev_end, float(duration_sec), len(cuts)))

    samples: list[IntentSample] = []
    for i, (s, e, shot_idx) in enumerate(shots_spans):
        if e - s < 1e-3:
            continue
        motion_class = _motion_for_shot(shot_idx, shot_profiles)
        # Shots are [start, end) for all but the last, which is
        # [start, end]. Prevents the last sample of shot N and the
        # first sample of shot N+1 from duplicating at the boundary.
        inclusive_end = (i == len(shots_spans) - 1)
        shot_samples = _build_shot_samples(
            s, e, shot_idx, motion_class, step,
            dense_faces=dense_faces,
            active_speaker_events=active_speaker_events,
            saliency_regions=saliency_regions,
            object_detections=object_detections,
            inclusive_end=inclusive_end,
        )
        samples.extend(shot_samples)

    return samples


def _build_shot_samples(
    start: float, end: float, shot_idx: int, motion_class: str, step: float,
    *,
    dense_faces: list,
    active_speaker_events: list,
    saliency_regions: Optional[list],
    object_detections: Optional[list],
    inclusive_end: bool = True,
) -> list[IntentSample]:
    """Build samples for one shot. Applies the source cascade +
    last-real hold + first-real backfill + 3-tap EMA smoothing."""
    # 1. Raw cascade: for each t in [start, end], pick the best real
    # source; None if no real source in range.
    grid: list[tuple[float, Optional[tuple[float, float, float, str]]]] = []
    t = start
    # Non-final shots stop strictly before end so the next shot's
    # first sample lives at t=end (not duplicated).
    end_thresh = end + 1e-6 if inclusive_end else end - 1e-6
    while t <= end_thresh:
        chosen: Optional[tuple] = None
        xy = _active_speaker_xy_at(active_speaker_events, t, dense_faces)
        if xy is not None:
            chosen = (*xy, "active_speaker")
        if chosen is None:
            xy = _face_xy_at(dense_faces, t)
            if xy is not None:
                chosen = (*xy, "face")
        if chosen is None and saliency_regions:
            xy = _saliency_xy_at(saliency_regions, t)
            if xy is not None:
                chosen = (*xy, "saliency")
        if chosen is None and object_detections:
            xy = _object_xy_at(object_detections, t)
            if xy is not None:
                chosen = (*xy, "object")
        grid.append((t, chosen))
        t += step

    # 2. Leading backfill: if the shot starts with real=None, use the
    # first real sample found within the shot to fill leading samples.
    first_real_idx = None
    for i, (_, s) in enumerate(grid):
        if s is not None:
            first_real_idx = i
            break
    if first_real_idx is None:
        # No real sources anywhere in this shot. Emit zero samples.
        # Downstream consumers treat the shot as "unknown intent" and
        # widen the crop (fix §4).
        return []
    # Backfill.
    for i in range(0, first_real_idx):
        grid[i] = (grid[i][0], grid[first_real_idx][1])

    # 3. Forward hold for internal gaps: keep the last real sample's
    # (x, y) but decay conf and tag source="kalman_predict".
    out: list[IntentSample] = []
    last_real: Optional[tuple[float, float, float, str]] = None
    hold_decay = 0.90
    for i, (tt, sample) in enumerate(grid):
        if sample is not None:
            last_real = sample
            x, y, conf, source = sample
        else:
            # Hold with decaying conf.
            assert last_real is not None  # ensured by backfill above
            x, y, prev_conf, _ = last_real
            conf = max(0.05, prev_conf * hold_decay)
            source = "kalman_predict"
            # Update last_real to reflect decayed conf so subsequent
            # holds continue to decay.
            last_real = (x, y, conf, source)
        out.append(IntentSample(
            t=tt, x=x, y=y, conf=conf,
            source=source, motion_class=motion_class, shot_idx=shot_idx,
        ))

    # 4. 3-tap EMA smoothing on x, y with α driven by motion_class.
    # Dynamic shots snap more readily (smaller α = snappier? no — EMA
    # α closer to 1 means more current-weight, so larger α is snappier).
    alpha = 0.45 if motion_class in ("static", "pan") else 0.65
    if len(out) >= 3:
        out = _ema_smooth(out, alpha=alpha, motion_class=motion_class)

    return out


def _ema_smooth(
    samples: list[IntentSample],
    *,
    alpha: float,
    motion_class: str,
) -> list[IntentSample]:
    """Apply 3-tap EMA to x, y in the sample list, preserving all
    other fields. Snap threshold widens with motion_class."""
    # Snap threshold: if distance between consecutive samples exceeds
    # this, reset the EMA so a genuine intent jump isn't smoothed to
    # a laggy drift. Dynamic shots are snappier.
    snap_thresh = 0.15 if motion_class == "dynamic" else 0.25
    out = list(samples)
    sx = samples[0].x
    sy = samples[0].y
    for i in range(1, len(samples)):
        cur = samples[i]
        dx = cur.x - sx
        dy = cur.y - sy
        dist = (dx * dx + dy * dy) ** 0.5
        if dist >= snap_thresh:
            # Snap — skip EMA for this sample.
            sx, sy = cur.x, cur.y
        else:
            sx = alpha * cur.x + (1.0 - alpha) * sx
            sy = alpha * cur.y + (1.0 - alpha) * sy
        out[i] = IntentSample(
            t=cur.t, x=sx, y=sy, conf=cur.conf,
            source=cur.source, motion_class=cur.motion_class,
            shot_idx=cur.shot_idx,
        )
    return out


def intent_at(track: list[IntentSample], t: float) -> Optional[IntentSample]:
    """Return the sample closest to t, or None if track is empty."""
    if not track:
        return None
    best = track[0]
    best_dt = abs(track[0].t - t)
    for s in track:
        dt = abs(s.t - t)
        if dt < best_dt:
            best = s
            best_dt = dt
    return best
