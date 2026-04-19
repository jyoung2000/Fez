"""2-D camera path solver with cinematographic composition.

Phase 2 of the human-like reframing overhaul. Accepts per-frame face
bboxes + optional gaze yaw + optional subject velocity, and returns
``(t, x, y, zoom)`` trajectories satisfying:

    * Headroom rule: the top of the face bbox sits between
      ``config.headroom_min`` and ``config.headroom_max`` of the output
      frame. Dynamic bounds scale with detected framing type.
    * Eye-line rule: the eye mid-point is biased toward the top third
      of the OUTPUT frame (``config.eye_line_y``, default 0.35).
    * Chin clip avoidance: ``face_bottom_y <= 1.0 - chin_margin``.
    * Lead-room: when ``gaze_yaw`` is non-zero, the subject is placed
      off-center in the direction they're facing.
    * Smooth camera motion: L1 smoothness penalties on velocity /
      acceleration / jerk, per-axis (see
      ``solve_autoflip_lp_2d`` in ``_autoflip_lp.py``).

This module operates purely in normalized source-frame coordinates —
all inputs and outputs are in [0, 1]. The FFmpeg pixel conversion
happens at the ``RenderPlan`` boundary as before.

Architecture note: the legacy 1-D solver (``l1_camera_path.py``)
remains the default. The 2-D solver here is called when
``config.human_reframe_enabled`` is true. Both live side-by-side so the
AutoFlip-parity fixtures stay bit-identical under the old flag.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from backend.services._autoflip_lp import solve_autoflip_lp_2d
from backend.services.reframe_config import ReframeConfig, get_default_config

logger = logging.getLogger(__name__)


# ── Inputs ─────────────────────────────────────────────────────────


@dataclass
class FaceFrame2D:
    """One face detection in one frame, normalized to source-frame."""

    t: float
    nose_x: float
    nose_y: float
    width: float
    height: float
    eye_y: Optional[float] = None        # explicit eye-line if available
    yaw: Optional[float] = None          # -1..1 (left..right), continuous
    is_active_speaker: bool = False
    slot_id: int = -1


# ── Framing classification from face size ─────────────────────────


def _infer_framing_category(face_height: float) -> str:
    """Classify framing from face-height-as-fraction-of-source.

    Roughly calibrated: a face spanning >= 35% of source height is a
    close-up; 20-35% medium shot; < 20% wide. The category drives the
    headroom bounds — close-ups tolerate less headroom than wide shots.
    """
    if face_height >= 0.35:
        return "CU"
    if face_height >= 0.20:
        return "MS"
    return "WS"


def _headroom_bounds_for_category(category: str, config: ReframeConfig) -> tuple[float, float]:
    """Return ``(min, max)`` headroom fractions per category.

    A close-up can have the face top at 8% of output; a wide shot at
    15-20%. Values linearly interpolate the config min/max.
    """
    hmin, hmax = config.headroom_min, config.headroom_max
    if category == "CU":
        return (hmin + 0.03, hmin + 0.08)
    if category == "MS":
        return (hmin + 0.05, hmax)
    return (hmin, hmax + 0.05)    # WS: more forgiving


# ── Y-axis target derivation ──────────────────────────────────────


def _target_cy_for_face(
    face: FaceFrame2D,
    *,
    crop_h_frac: float,
    config: ReframeConfig,
) -> float:
    """Return the source-frame y coordinate the crop should center on
    so that the face's eye-line lands at ``config.eye_line_y`` of the
    OUTPUT frame.

    Derivation:
      Let c_y be the crop center in source-frame fractions. The crop
      spans ``[c_y - crop_h/2, c_y + crop_h/2]`` in source coords. A
      point at ``src_y`` lies at ``(src_y - (c_y - crop_h/2)) / crop_h``
      in output coords. Setting that to ``eye_line_y`` and solving:
          c_y = src_y - (eye_line_y - 0.5) * crop_h
    """
    eye_y_src = face.eye_y if face.eye_y is not None else (
        face.nose_y - face.height * 0.25
    )
    shift = (config.eye_line_y - 0.5) * crop_h_frac
    return eye_y_src - shift


# ── X-axis target derivation (lead-room) ──────────────────────────


def _target_cx_for_face(
    face: FaceFrame2D,
    *,
    crop_w_frac: float,
    config: ReframeConfig,
) -> float:
    """Return the x crop center including gaze-based lead-room shift.

    A face with ``yaw = +0.8`` (looking hard right) is placed on the
    **left** third of the output frame so there is space to their right.
    Convention: ``shift = -sign(yaw) * lead_room_gain * crop_w_frac``.
    """
    if face.yaw is None or abs(face.yaw) < 0.1:
        return face.nose_x
    sign = 1.0 if face.yaw > 0 else -1.0
    shift = -sign * config.lead_room_gain * crop_w_frac * min(abs(face.yaw), 1.0)
    return face.nose_x + shift


# ── Y-axis bounds from headroom + chin rules ──────────────────────


def _first_face_bounds_y(
    faces_by_frame: list[Optional[FaceFrame2D]],
    *,
    crop_h_frac: float,
    config: ReframeConfig,
) -> Optional[tuple[float, float, float]]:
    """Fix 3.5: pre-pass scan for the first non-None face, compute the
    same (target_cy, lo_cy, hi_cy) triple we'd compute inline, and use
    it to initialize prev_* so leading no-face frames don't pin to 0.5.
    """
    for face in faces_by_frame:
        if face is None:
            continue
        category = _infer_framing_category(face.height)
        h_min, h_max = _headroom_bounds_for_category(category, config)
        face_top_src = face.nose_y - face.height * 0.5
        face_bot_src = face.nose_y + face.height * 0.5
        cy_at_hmin = face_top_src - (h_min - 0.5) * crop_h_frac
        cy_at_hmax = face_top_src - (h_max - 0.5) * crop_h_frac
        range_lo = min(cy_at_hmin, cy_at_hmax)
        range_hi = max(cy_at_hmin, cy_at_hmax)
        chin_margin = 0.02
        chin_upper_cy = face_bot_src + chin_margin - crop_h_frac / 2.0
        range_hi = min(range_hi, 1.0 - crop_h_frac / 2.0)
        range_lo = max(range_lo, chin_upper_cy)
        range_lo = max(range_lo, crop_h_frac / 2.0)
        if range_lo > range_hi:
            range_lo = range_hi = 0.5 * (range_lo + range_hi)
        target = _target_cy_for_face(face, crop_h_frac=crop_h_frac, config=config)
        target = max(range_lo, min(range_hi, target))
        return target, range_lo, range_hi
    return None


def _build_y_bounds(
    faces_by_frame: list[Optional[FaceFrame2D]],
    *,
    crop_h_frac: float,
    config: ReframeConfig,
) -> tuple[list[float], list[float], list[float]]:
    """Return ``(target_cy, lo_cy, hi_cy)`` lists, length ``n_frames``.

    Fix 3.5: when a frame has no face, hold the last-known-good bounds
    (prev_target/lo/hi). The initial prev_* values come from a pre-pass
    that finds the first real face — not from a hardcoded 0.5 center —
    so leading no-face frames don't pin to frame center and produce
    chin/head clips when the face appears.
    """
    targets: list[float] = []
    lo: list[float] = []
    hi: list[float] = []

    init = _first_face_bounds_y(
        faces_by_frame, crop_h_frac=crop_h_frac, config=config,
    )
    if init is not None:
        prev_target, prev_lo, prev_hi = init
    else:
        prev_target = 0.5
        prev_lo = crop_h_frac / 2.0
        prev_hi = 1.0 - crop_h_frac / 2.0

    for face in faces_by_frame:
        if face is None:
            targets.append(prev_target)
            lo.append(prev_lo)
            hi.append(prev_hi)
            continue

        category = _infer_framing_category(face.height)
        h_min, h_max = _headroom_bounds_for_category(category, config)

        face_top_src = face.nose_y - face.height * 0.5
        face_bot_src = face.nose_y + face.height * 0.5

        # For face_top to land at output fraction ``h`` of the crop:
        #   h * crop_h = face_top_src - (c_y - crop_h/2)
        #   c_y = face_top_src - (h - 0.5) * crop_h
        cy_at_hmin = face_top_src - (h_min - 0.5) * crop_h_frac
        cy_at_hmax = face_top_src - (h_max - 0.5) * crop_h_frac
        # h_min < h_max, so cy_at_hmin > cy_at_hmax: the allowed range
        # for c_y is [cy_at_hmax, cy_at_hmin].
        range_lo = min(cy_at_hmin, cy_at_hmax)
        range_hi = max(cy_at_hmin, cy_at_hmax)

        # Chin clip: face_bot_src must sit within the crop's lower edge,
        # i.e. face_bot_src <= c_y + crop_h/2 - chin_margin.
        chin_margin = 0.02
        chin_upper_cy = face_bot_src + chin_margin - crop_h_frac / 2.0
        range_hi = min(range_hi, 1.0 - crop_h_frac / 2.0)
        range_lo = max(range_lo, chin_upper_cy)
        range_lo = max(range_lo, crop_h_frac / 2.0)

        if range_lo > range_hi:
            range_lo = range_hi = 0.5 * (range_lo + range_hi)

        target = _target_cy_for_face(face, crop_h_frac=crop_h_frac, config=config)
        target = max(range_lo, min(range_hi, target))

        targets.append(target)
        lo.append(range_lo)
        hi.append(range_hi)
        prev_target, prev_lo, prev_hi = target, range_lo, range_hi

    return targets, lo, hi


# ── X-axis bounds from face visibility + lead-room ────────────────


def _first_face_bounds_x(
    faces_by_frame: list[Optional[FaceFrame2D]],
    *,
    crop_w_frac: float,
    config: ReframeConfig,
) -> Optional[tuple[float, float, float]]:
    """Fix 3.5: pre-pass for x-axis — same idea as y-axis. Returns the
    first real face's (target_cx, lo_cx, hi_cx) triple."""
    for face in faces_by_frame:
        if face is None:
            continue
        face_l = face.nose_x - face.width * 0.5 - 0.01
        face_r = face.nose_x + face.width * 0.5 + 0.01
        range_lo = max(crop_w_frac / 2.0, face_r - crop_w_frac / 2.0)
        range_hi = min(1.0 - crop_w_frac / 2.0, face_l + crop_w_frac / 2.0)
        if range_lo > range_hi:
            range_lo = range_hi = 0.5 * (range_lo + range_hi)
        target = _target_cx_for_face(face, crop_w_frac=crop_w_frac, config=config)
        target = max(range_lo, min(range_hi, target))
        return target, range_lo, range_hi
    return None


def _build_x_bounds(
    faces_by_frame: list[Optional[FaceFrame2D]],
    *,
    crop_w_frac: float,
    config: ReframeConfig,
) -> tuple[list[float], list[float], list[float]]:
    """Return ``(target_cx, lo_cx, hi_cx)`` lists, length ``n_frames``.

    Per-frame bounds enforce face containment: the face bbox (expanded
    by a small safety padding) must be fully inside the crop.

    Fix 3.5: prev_* is initialized from the first real face via a
    pre-pass so leading no-face frames don't snap to 0.5.
    """
    targets: list[float] = []
    lo: list[float] = []
    hi: list[float] = []

    init = _first_face_bounds_x(
        faces_by_frame, crop_w_frac=crop_w_frac, config=config,
    )
    if init is not None:
        prev_t, prev_lo, prev_hi = init
    else:
        prev_t = 0.5
        prev_lo = crop_w_frac / 2.0
        prev_hi = 1.0 - crop_w_frac / 2.0

    for face in faces_by_frame:
        if face is None:
            targets.append(prev_t)
            lo.append(prev_lo)
            hi.append(prev_hi)
            continue

        face_l = face.nose_x - face.width * 0.5 - 0.01
        face_r = face.nose_x + face.width * 0.5 + 0.01

        # Face containment: face_l >= c_x - crop_w/2, face_r <= c_x + crop_w/2
        # => c_x in [face_r - crop_w/2, face_l + crop_w/2]
        range_lo = max(crop_w_frac / 2.0, face_r - crop_w_frac / 2.0)
        range_hi = min(1.0 - crop_w_frac / 2.0, face_l + crop_w_frac / 2.0)
        if range_lo > range_hi:
            range_lo = range_hi = 0.5 * (range_lo + range_hi)

        target = _target_cx_for_face(face, crop_w_frac=crop_w_frac, config=config)
        target = max(range_lo, min(range_hi, target))

        targets.append(target)
        lo.append(range_lo)
        hi.append(range_hi)
        prev_t, prev_lo, prev_hi = target, range_lo, range_hi

    return targets, lo, hi


# ── Public API ────────────────────────────────────────────────────


@dataclass
class CameraPath2D:
    """Solved 2-D camera path for a segment."""

    timestamps: list[float]
    cx: list[float]
    cy: list[float]
    crop_w_frac: float
    crop_h_frac: float
    # Fix 3.5: windows where we had no face for longer than
    # ``kalman_prediction_ms * 3`` (~1s). The adapter widens the crop
    # to WIDE_MASTER there so stale y-centers don't clip a re-appearing
    # face above or below the crop edge.
    y_uncertain_windows: list[tuple[float, float]] = None

    def __post_init__(self):
        if self.y_uncertain_windows is None:
            self.y_uncertain_windows = []


def solve_2d_camera_path(
    faces_by_frame: list[Optional[FaceFrame2D]],
    *,
    timestamps: list[float],
    source_w: int,
    source_h: int,
    target_w: int = 1080,
    target_h: int = 1920,
    config: Optional[ReframeConfig] = None,
) -> CameraPath2D:
    """Solve a 2-D camera path for a segment of ``faces_by_frame``.

    ``timestamps`` and ``faces_by_frame`` must be the same length. Entries
    in ``faces_by_frame`` may be ``None`` for frames without a detection
    — bounds are held constant from the previous frame.

    Returns a :class:`CameraPath2D`.
    """
    if config is None:
        config = get_default_config()
    n = len(timestamps)
    if n == 0:
        return CameraPath2D(timestamps=[], cx=[], cy=[],
                            crop_w_frac=0.0, crop_h_frac=0.0)

    # Output aspect 9:16 on 16:9 source → crop_w = source_h * 9/16 in px.
    # In source-frame normalized fractions:
    src_aspect = source_w / max(source_h, 1)
    tgt_aspect = target_w / max(target_h, 1)
    if tgt_aspect < src_aspect:
        # crop is narrower than source horizontally
        crop_w_frac = tgt_aspect / src_aspect
        crop_h_frac = 1.0
    else:
        crop_w_frac = 1.0
        crop_h_frac = src_aspect / tgt_aspect

    # Sanitize: ensure valid crop window at least 20% of frame.
    crop_w_frac = max(0.2, min(1.0, crop_w_frac))
    crop_h_frac = max(0.2, min(1.0, crop_h_frac))

    tx, lox, hix = _build_x_bounds(faces_by_frame,
                                   crop_w_frac=crop_w_frac, config=config)
    ty, loy, hiy = _build_y_bounds(faces_by_frame,
                                   crop_h_frac=crop_h_frac, config=config)

    # Fix 3.5: record no-face runs longer than kalman_prediction_ms*3
    # as y-uncertain windows. The adapter uses these to emit
    # WIDE_MASTER instead of a tight crop while the subject is lost.
    uncertain_thresh_sec = max(
        0.5, getattr(config, "kalman_prediction_ms", 350) / 1000.0 * 3.0
    )
    y_uncertain: list[tuple[float, float]] = []
    run_start: Optional[float] = None
    run_start_idx = 0
    for i, face in enumerate(faces_by_frame):
        if face is None:
            if run_start is None:
                run_start = timestamps[i]
                run_start_idx = i
        else:
            if run_start is not None:
                run_end = timestamps[i]
                if (run_end - run_start) >= uncertain_thresh_sec:
                    y_uncertain.append((run_start, run_end))
                run_start = None
    if run_start is not None:
        run_end = timestamps[-1]
        if (run_end - run_start) >= uncertain_thresh_sec:
            y_uncertain.append((run_start, run_end))

    # LP wants pixel-ish scalars. We solve in normalized [0, 1] space
    # scaled by 1000 so the cost weights don't collapse into rounding
    # error. Solutions are unscaled back at the end.
    scale = 1000.0

    try:
        cam_x_s, cam_y_s = solve_autoflip_lp_2d(
            [v * scale for v in tx],
            [v * scale for v in ty],
            [v * scale for v in lox],
            [v * scale for v in hix],
            [v * scale for v in loy],
            [v * scale for v in hiy],
            lam1=config.lp_lambda_data,
            lam2_x=config.lp_lambda_v,
            lam2_y=config.lp_lambda_v_y,
            lam3_x=config.lp_lambda_a,
            lam3_y=config.lp_lambda_a_y,
            lam4_x=config.lp_lambda_j,
            lam4_y=config.lp_lambda_j_y,
        )
        cam_x = [v / scale for v in cam_x_s]
        cam_y = [v / scale for v in cam_y_s]
    except Exception as e:
        logger.warning("2-D LP solve failed (%s); falling back to targets", e)
        cam_x = list(tx)
        cam_y = list(ty)

    return CameraPath2D(
        timestamps=list(timestamps),
        cx=cam_x,
        cy=cam_y,
        crop_w_frac=crop_w_frac,
        crop_h_frac=crop_h_frac,
        y_uncertain_windows=y_uncertain,
    )


# ── Helpers for converting face_registry / dense faces to FaceFrame2D

def faces_from_dense(
    dense_faces: list,
    *,
    active_speaker_events: list | None = None,
    slot_preference: int | None = None,
) -> list[Optional[FaceFrame2D]]:
    """Convert a ``list[FrameFaces]`` into a per-frame list of primary
    faces, ready for ``solve_2d_camera_path``. Picks the active speaker
    when one is available in the frame, else the largest face.

    Fix 3.6: events with ``slot_id == -1`` (detected speaker whose face
    hasn't been mapped to a registry slot yet) no longer fall through
    to "largest face". Instead:

      1. Pick the face with the highest ``lip_aperture`` when any is
         >= 0.05 — that's the one actually speaking in this frame.
      2. If no face has ``lip_aperture >= 0.05`` and the event carries
         an ``audio_peak_x`` hint (0..1), pick the face nearest that x.
      3. Only then fall back to the largest face.

    ``nose_x/y/width/height`` on ``FrameFaces.faces`` entries are in
    percent (0-100) per existing convention in the codebase; this helper
    normalizes to 0-1.
    """
    def _event_at(t: float):
        """Return the active SpeakerEvent (any slot) at time t, or None."""
        if not active_speaker_events:
            return None
        for ev in active_speaker_events:
            if ev.start <= t <= ev.end:
                return ev
        return None

    def _pick_for_unresolved(faces: list, ev) -> object:
        """Fix 3.6: unresolved-slot speaker → lip/audio → largest fallback."""
        best_lip_face = None
        best_lip = 0.0
        for f in faces:
            lip = float(getattr(f, "lip_aperture", 0.0) or 0.0)
            if lip > best_lip:
                best_lip = lip
                best_lip_face = f
        if best_lip_face is not None and best_lip >= 0.05:
            return best_lip_face
        audio_peak_x = getattr(ev, "audio_peak_x", None)
        if audio_peak_x is not None:
            try:
                target = float(audio_peak_x) * 100.0  # nose_x is 0-100
                return min(faces, key=lambda f: abs(f.nose_x - target))
            except (TypeError, ValueError):
                pass
        return max(faces, key=lambda f: f.width * f.height)

    out: list[Optional[FaceFrame2D]] = []
    for ff in dense_faces:
        event = _event_at(ff.timestamp)
        active_slot = getattr(event, "slot_id", -1) if event is not None else -1
        faces = [f for f in ff.faces if getattr(f, "is_human", True)]
        if not faces:
            out.append(None)
            continue
        # Prefer active speaker slot, else slot_preference, else largest.
        chosen = None
        if active_slot >= 0:
            for f in faces:
                if getattr(f, "identity_id", -1) == active_slot:
                    chosen = f
                    break
        # Fix 3.6: a speaker event exists but its slot is unresolved
        # (slot_id == -1). Don't silently fall through to largest — use
        # lip-aperture / audio-peak hints.
        if chosen is None and event is not None and active_slot < 0:
            chosen = _pick_for_unresolved(faces, event)
        if chosen is None and slot_preference is not None:
            for f in faces:
                if getattr(f, "identity_id", -1) == slot_preference:
                    chosen = f
                    break
        if chosen is None:
            chosen = max(faces, key=lambda f: f.width * f.height)

        yaw = getattr(chosen, "yaw", None)
        # Legacy values are in degrees (~±45). Normalize to [-1, 1] when
        # a large value appears.
        if yaw is not None:
            try:
                yv = float(yaw)
                yaw = yv / 45.0 if abs(yv) > 1.5 else yv
            except (TypeError, ValueError):
                yaw = None

        out.append(FaceFrame2D(
            t=ff.timestamp,
            nose_x=chosen.nose_x / 100.0,
            nose_y=chosen.nose_y / 100.0,
            width=chosen.width / 100.0,
            height=chosen.height / 100.0,
            eye_y=None,
            yaw=yaw if isinstance(yaw, (int, float)) else None,
            is_active_speaker=(getattr(chosen, "identity_id", -1) == active_slot),
            slot_id=getattr(chosen, "identity_id", -1),
        ))
    return out
