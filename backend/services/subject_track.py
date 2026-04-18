"""Dense per-frame subject-position track.

Purpose
-------
The legacy reframe pipeline builds the 9:16 crop window from sparse VLM
``SceneDescription.subject_x`` values. On clips with **few scenes** or
**fast camera action** between scenes, the portrait window is held at
the last keyframe — and when the VLM defaulted that keyframe to 50, the
subject is *off-screen* in the crop for a noticeable chunk of the clip.
The user hit this on an Attack on Titan reframe: the character's face
sits at x≈70 in the 16:9 source but the 9:16 window stayed at center.

This module builds a **dense** ``(t, x, y, conf, source)`` track from the
signals the pipeline has already computed:

1. **Dense face detections** (~2 Hz by default)  — highest confidence,
   pixel-accurate. Picks the active speaker when available, else the
   largest / primary face.
2. **Saliency regions** from ``saliency_tracker`` — content-agnostic
   attention peaks, works well for animation / illustrated content
   where face detectors often fail.
3. **YOLO object detections** — catches characters / vehicles / sports
   balls the VLM missed.
4. **Nearest scene ``subject_x``** — VLM hint, last line of defense.
5. **Forward-fill hold** — never emit ``None``; the previous sample is
   better than a center default.

The resulting track is saved on the job and consumed by the frontend
preview **and** the export, so the 9:16 / 1:1 reframe follows the
subject on **every** frame instead of drifting between sparse VLM
keyframes.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Lookup windows (seconds) for each signal. Dense faces run at 2 Hz so
# 0.6 s covers ±1 dense sample. Saliency / objects are sparser, so we
# allow a larger window before falling through to the next source.
_FACE_WINDOW_S = 0.6
_SAL_WINDOW_S = 2.5
_OBJ_WINDOW_S = 2.5
_SCENE_WINDOW_S = 4.0


def _clamp_pct(v: Optional[float]) -> Optional[float]:
    """Coerce any numeric input to a 0..100 float, rejecting ``None``/NaN."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return max(0.0, min(100.0, f))


def _nearest(sorted_arr: list, t: float, *, window_s: float):
    """Binary-search the closest entry by ``timestamp`` within ``window_s``."""
    if not sorted_arr:
        return None
    # Small arrays — linear scan beats bisect overhead.
    best = None
    best_d = window_s
    for item in sorted_arr:
        it = getattr(item, "timestamp", None)
        if it is None:
            continue
        d = abs(it - t)
        if d <= best_d:
            best_d = d
            best = item
    return best


def _face_xy(frame_faces) -> Optional[tuple[float, float, float]]:
    """Return ``(x_pct, y_pct, conf)`` for the most relevant face."""
    if frame_faces is None:
        return None
    faces = getattr(frame_faces, "faces", None) or []
    if not faces:
        return None

    # Prefer the actively-speaking face on multi-speaker frames. Lip
    # aperture > 0.03 is the same threshold the pipeline uses for
    # active-speaker routing elsewhere.
    speaking = [f for f in faces if getattr(f, "lip_aperture", 0.0) > 0.03]
    if speaking:
        face = max(speaking, key=lambda f: getattr(f, "lip_aperture", 0.0))
        conf_boost = 0.1
    else:
        # Fall back to the primary face, then the widest (closest) face.
        primary_idx = getattr(frame_faces, "primary_face_idx", -1)
        if 0 <= primary_idx < len(faces):
            face = faces[primary_idx]
            conf_boost = 0.05
        else:
            face = max(faces, key=lambda f: getattr(f, "width", 0.0))
            conf_boost = 0.0

    x = _clamp_pct(getattr(face, "nose_x", None) or getattr(face, "x_center", None))
    y = _clamp_pct(getattr(face, "nose_y", None) or getattr(face, "y_center", None))
    if x is None:
        return None
    conf = float(getattr(face, "confidence", 0.8)) + conf_boost
    return (x, y if y is not None else 50.0, min(1.0, conf))


def _saliency_xy(region) -> Optional[tuple[float, float, float]]:
    if region is None:
        return None
    x = _clamp_pct(
        getattr(region, "x_center_pct", None) or getattr(region, "x_center", None)
    )
    y = _clamp_pct(
        getattr(region, "y_center_pct", None) or getattr(region, "y_center", None)
    )
    if x is None:
        return None
    # Saliency confidence is weaker than face — cap at 0.55 so a face
    # anywhere near the same moment wins the cascade.
    conf = min(0.55, float(getattr(region, "confidence", 0.5) or 0.5))
    return (x, y if y is not None else 50.0, conf)


def _object_xy(det) -> Optional[tuple[float, float, float]]:
    if det is None:
        return None
    x = _clamp_pct(
        getattr(det, "x_center_pct", None) or getattr(det, "primary_object_x", None)
    )
    y = _clamp_pct(
        getattr(det, "y_center_pct", None) or getattr(det, "primary_object_y", None)
    )
    if x is None:
        return None
    conf = min(0.5, float(getattr(det, "confidence", 0.45) or 0.45))
    return (x, y if y is not None else 50.0, conf)


def _scene_xy(scene) -> Optional[tuple[float, float, float]]:
    if scene is None:
        return None
    # Prefer active speaker x when the pipeline filled it in.
    x = _clamp_pct(getattr(scene, "active_speaker_x", None) or getattr(scene, "subject_x", None))
    if x is None:
        return None
    # VLM is weakest of all sources — cap confidence so any face/saliency
    # hit within the face window replaces it.
    conf = 0.3 if 47 <= x <= 53 else 0.4
    return (x, 50.0, conf)


def build_subject_track(
    *,
    duration_s: float,
    dense_face_results: list,
    saliency_regions: list,
    object_detections: list,
    scenes: list,
    sample_rate_hz: float = 2.0,
) -> list[dict[str, Any]]:
    """Build a dense subject-position track spanning ``[0, duration_s]``.

    Samples at ``sample_rate_hz`` (default 2 Hz to match the dense face
    tracker) and for each sample picks the highest-confidence source
    from the face → saliency → object → scene cascade, forward-filling
    any remaining gaps. Returns a list of dicts the frontend and export
    can consume directly.
    """
    if duration_s <= 0:
        return []

    # Pre-sort once so ``_nearest`` walks the smallest relevant window.
    face_sorted = sorted(
        (f for f in (dense_face_results or []) if getattr(f, "faces", None)),
        key=lambda f: getattr(f, "timestamp", 0.0),
    )
    sal_sorted = sorted(
        (r for r in (saliency_regions or []) if r is not None),
        key=lambda r: getattr(r, "timestamp", 0.0),
    )
    obj_sorted = sorted(
        (d for d in (object_detections or []) if d is not None),
        key=lambda d: getattr(d, "timestamp", 0.0),
    )
    scene_sorted = sorted(
        (s for s in (scenes or []) if s is not None),
        key=lambda s: getattr(s, "timestamp", 0.0),
    )

    step = 1.0 / max(0.5, float(sample_rate_hz))
    track: list[dict[str, Any]] = []
    last: Optional[dict[str, Any]] = None

    t = 0.0
    while t <= duration_s + 1e-3:
        chosen: Optional[tuple[float, float, float, str]] = None  # (x, y, conf, source)

        # 1. Dense face within ±_FACE_WINDOW_S — strongest signal.
        f = _nearest(face_sorted, t, window_s=_FACE_WINDOW_S)
        if f is not None:
            xy = _face_xy(f)
            if xy is not None:
                chosen = (*xy, "face")

        # 2. Saliency region within ±_SAL_WINDOW_S.
        if chosen is None:
            r = _nearest(sal_sorted, t, window_s=_SAL_WINDOW_S)
            if r is not None:
                xy = _saliency_xy(r)
                if xy is not None:
                    chosen = (*xy, "saliency")

        # 3. YOLO object within ±_OBJ_WINDOW_S.
        if chosen is None:
            d = _nearest(obj_sorted, t, window_s=_OBJ_WINDOW_S)
            if d is not None:
                xy = _object_xy(d)
                if xy is not None:
                    chosen = (*xy, "object")

        # 4. Nearest VLM scene within ±_SCENE_WINDOW_S.
        if chosen is None:
            s = _nearest(scene_sorted, t, window_s=_SCENE_WINDOW_S)
            if s is not None:
                xy = _scene_xy(s)
                if xy is not None:
                    chosen = (*xy, "scene")

        # 5. Hold the previous value (never emit a raw center default).
        if chosen is None and last is not None:
            chosen = (last["x"], last["y"], max(0.1, last["conf"] * 0.9), "hold")

        # 6. Absolute last resort — center, flagged low confidence.
        if chosen is None:
            chosen = (50.0, 50.0, 0.1, "none")

        x, y, conf, source = chosen
        point = {
            "t": round(t, 3),
            "x": round(x, 2),
            "y": round(y, 2),
            "conf": round(conf, 3),
            "source": source,
        }
        track.append(point)
        last = point
        t += step

    # Backward-fill the leading "none"/"hold" runs: if the *first* real
    # signal appears at t=4.2s and before that we emitted hold/none at
    # 50, the viewer sees a half-second of empty 9:16. Replace those
    # leading center points with the first real sample.
    first_real_idx = next(
        (i for i, p in enumerate(track) if p["source"] in {"face", "saliency", "object", "scene"}),
        None,
    )
    if first_real_idx and first_real_idx > 0:
        first = track[first_real_idx]
        for i in range(first_real_idx):
            track[i]["x"] = first["x"]
            track[i]["y"] = first["y"]
            track[i]["source"] = "hold_back"
            track[i]["conf"] = max(0.15, first["conf"] * 0.6)

    # Light temporal smoothing so per-frame jitter (especially on
    # saliency) doesn't manifest as a wobbly camera. Uses a 3-tap
    # causal EMA with α tuned so hard scene cuts (large deltas) are
    # still preserved — anything over 20 percentage points of horizontal
    # travel between adjacent samples snaps instead of smoothing.
    if len(track) >= 3:
        alpha = 0.45
        for i in range(1, len(track)):
            prev = track[i - 1]
            cur = track[i]
            # Snap at cuts: big jumps are treated as cuts, not pans.
            if abs(cur["x"] - prev["x"]) > 20.0:
                continue
            cur["x"] = round(alpha * cur["x"] + (1 - alpha) * prev["x"], 2)
            cur["y"] = round(alpha * cur["y"] + (1 - alpha) * prev["y"], 2)

    # Logging summary — helps diagnose "why is the reframe still bad"
    # without having to dump the whole track.
    source_counts: dict[str, int] = {}
    for p in track:
        source_counts[p["source"]] = source_counts.get(p["source"], 0) + 1
    logger.info(
        "[subject_track] %d samples @ %.1f Hz (%.1fs), sources=%s",
        len(track), sample_rate_hz, duration_s, source_counts,
    )
    return track
