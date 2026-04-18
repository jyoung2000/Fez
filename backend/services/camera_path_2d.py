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


def _build_y_bounds(
    faces_by_frame: list[Optional[FaceFrame2D]],
    *,
    crop_h_frac: float,
    config: ReframeConfig,
) -> tuple[list[float], list[float], list[float]]:
    """Return ``(target_cy, lo_cy, hi_cy)`` lists, length ``n_frames``.

    When a frame has no face detection we fall back to the previous
    frame's bounds (held constant), or the middle of the frame when no
    history exists. This mirrors the x-axis behavior in
    ``required_regions.build_required_regions``.
    """
    targets: list[float] = []
    lo: list[float] = []
    hi: list[float] = []

    fallback_t = 0.5
    fallback_lo = crop_h_frac / 2.0
    fallback_hi = 1.0 - crop_h_frac / 2.0

    prev_target, prev_lo, prev_hi = fallback_t, fallback_lo, fallback_hi

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


def _build_x_bounds(
    faces_by_frame: list[Optional[FaceFrame2D]],
    *,
    crop_w_frac: float,
    config: ReframeConfig,
) -> tuple[list[float], list[float], list[float]]:
    """Return ``(target_cx, lo_cx, hi_cx)`` lists, length ``n_frames``.

    Per-frame bounds enforce face containment: the face bbox (expanded
    by a small safety padding) must be fully inside the crop.
    """
    targets: list[float] = []
    lo: list[float] = []
    hi: list[float] = []

    fallback = 0.5
    fallback_lo = crop_w_frac / 2.0
    fallback_hi = 1.0 - crop_w_frac / 2.0

    prev_t, prev_lo, prev_hi = fallback, fallback_lo, fallback_hi

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

    ``nose_x/y/width/height`` on ``FrameFaces.faces`` entries are in
    percent (0-100) per existing convention in the codebase; this helper
    normalizes to 0-1.
    """
    def _active_slot_at(t: float) -> int:
        if not active_speaker_events:
            return -1
        for ev in active_speaker_events:
            if ev.start <= t <= ev.end and getattr(ev, "slot_id", -1) >= 0:
                return ev.slot_id
        return -1

    out: list[Optional[FaceFrame2D]] = []
    for ff in dense_faces:
        active_slot = _active_slot_at(ff.timestamp)
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
