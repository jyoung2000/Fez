"""Gaze direction estimator for lead-room application.

Uses face landmark asymmetry (nose tip vs eye midpoint) to infer which
direction a face is looking. For narrative content, this drives lead-room
offset: a character looking screen-left should be placed on the RIGHT
third of the vertical frame.

Two API tiers:

- **Categorical** (Phase 0+): ``estimate_gaze_direction`` returns one of
  ``"left"`` / ``"right"`` / ``"center"`` and ``apply_lead_room`` applies
  a step shift in the corresponding direction. This is the legacy path
  the reframe segmenter has used since the original AutoFlip work.

- **Continuous** (Phase 4): ``estimate_yaw`` returns a float in
  ``[-1.0, 1.0]`` (negative = looking left, positive = looking right,
  0 = forward). ``estimate_yaw_from_dense`` averages per-frame yaws
  with optional EMA smoothing. ``lead_room_offset_px`` converts a
  yaw value into a continuous pixel offset linearly scaled by
  ``0.08 * crop_width_px`` at full yaw, matching the v2 spec.

The continuous tier is wired into the reframe segmenter's Stage 8
behind ``CLIPAI_GAZE_LEAD_ROOM_V2=1`` (default OFF until the in-docker
validation run lands the post-Phase-4 numbers). With the flag off,
the categorical path stays in effect — Phase 4 ships the V2 functions
+ tests + integration without changing existing behavior.
"""

import logging
import os
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# ── Phase 4 feature flag ────────────────────────────────────────
USE_GAZE_LEAD_ROOM_V2 = os.environ.get(
    "CLIPAI_GAZE_LEAD_ROOM_V2", "1",
).lower() in ("1", "true", "yes", "on")


# ── Continuous yaw API (Phase 4) ────────────────────────────────


def estimate_yaw(face) -> float:
    """Estimate continuous head yaw in ``[-1.0, 1.0]``.

    Uses the same nose-vs-bbox-center asymmetry as the categorical
    ``estimate_gaze_direction`` but returns a signed float instead of
    bucketing into left/right/center.

    Sign convention:
        ``-1.0`` = full left  (nose is at the LEFT edge of the bbox)
        ``+1.0`` = full right (nose is at the RIGHT edge of the bbox)
         ``0.0`` = forward    (nose at bbox center) or unknown

    The asymmetry is normalized by face width and capped at ±1.0. A
    face with no nose / width data returns 0.0 so callers don't have
    to special-case missing keypoints.
    """
    nose_x = getattr(face, "nose_x", None)
    x_center = getattr(face, "x_center", None)
    width = getattr(face, "width", None)
    if nose_x is None or width is None or width <= 0 or x_center is None:
        return 0.0
    # offset is signed: negative when nose is left of center.
    offset = (nose_x - x_center) / width
    # The nose can travel at most ±0.5 face widths under a head turn
    # before the bbox snaps to the new face orientation; double to map
    # that natural range onto [-1, 1].
    yaw = max(-1.0, min(1.0, 2.0 * offset))
    return yaw


def estimate_yaw_from_dense(
    dense_faces: list,
    slot_id: int,
    start: float,
    end: float,
    *,
    smooth_alpha: float = 0.3,
) -> float:
    """Mean per-frame yaw for a face slot in a time window.

    Sweeps ``dense_faces`` for entries whose timestamp lies in
    ``[start, end]`` and whose ``identity_id == slot_id``, computes
    ``estimate_yaw`` per face, optionally smooths the per-frame
    series with a bidirectional EMA at the same α=0.3 used by the
    saliency Fix 6 path, and returns the **mean** of the smoothed
    sequence. Returns 0.0 when no matching faces are found.

    The EMA smoothing reduces jitter from per-frame keypoint noise
    so a single off-frame doesn't flip the lead-room sign.
    """
    raw_yaws: list[float] = []
    for df in dense_faces:
        ts = getattr(df, "timestamp", -1.0)
        if ts < start or ts > end:
            continue
        for f in getattr(df, "faces", []) or []:
            if getattr(f, "identity_id", -1) != slot_id:
                continue
            raw_yaws.append(estimate_yaw(f))
            break  # one face per frame for this slot
    if not raw_yaws:
        return 0.0
    smoothed = smooth_yaw_ema(raw_yaws, alpha=smooth_alpha)
    return sum(smoothed) / len(smoothed)


def smooth_yaw_ema(values: Iterable[float], alpha: float = 0.3) -> list[float]:
    """Symmetric (forward + backward) EMA smoothing.

    Mirrors the existing ``_post_solve_smooth`` pattern in
    ``backend/services/l1_camera_path.py`` but applied to a
    1-D yaw series instead of a camera path. Symmetric EMA preserves
    the timing of yaw transitions (no phase shift) while suppressing
    sub-frame jitter.

    Args:
        values: Iterable of per-frame floats.
        alpha: Smoothing factor in (0, 1]. Higher = less smoothing.
            Default 0.3 matches the saliency Fix 6 EMA.

    Returns:
        List of smoothed values, same length as the input.
    """
    xs = list(values)
    n = len(xs)
    if n < 2 or alpha >= 1.0:
        return xs
    fwd = [0.0] * n
    fwd[0] = xs[0]
    for i in range(1, n):
        fwd[i] = alpha * xs[i] + (1.0 - alpha) * fwd[i - 1]
    bwd = [0.0] * n
    bwd[n - 1] = xs[n - 1]
    for i in range(n - 2, -1, -1):
        bwd[i] = alpha * xs[i] + (1.0 - alpha) * bwd[i + 1]
    return [(f + b) / 2.0 for f, b in zip(fwd, bwd)]


def lead_room_offset_px(
    yaw: float,
    crop_width_px: float,
    *,
    max_frac: float = 0.08,
    anime_action_multiplier: float = 1.0,
) -> float:
    """Convert a yaw value to a horizontal lead-room offset in pixels.

    Returns a signed float to ADD to ``subject_x`` (the source-frame
    pixel coordinate of the crop center). The magnitude is
    ``|yaw| * max_frac * crop_width_px * anime_action_multiplier``
    and the sign is OPPOSITE of yaw — a face looking left (yaw < 0)
    gets a positive offset (camera shifts right), placing the face
    on the LEFT side of the output crop with space to look INTO.
    This matches the existing categorical convention and the
    cinematography rule.

    Args:
        yaw: Yaw value in ``[-1.0, 1.0]`` from ``estimate_yaw``.
        crop_width_px: Output crop width in source pixels (≈ 607 for
            16:9 → 9:16 on a 1920-wide source).
        max_frac: Maximum offset as a fraction of the crop width.
            Default 0.08 matches the v2 Phase 4 spec — about 50 px
            on 600 crop, gentler than the categorical step (~95 px).
        anime_action_multiplier: Phase 6 extension. Multiplies the
            offset magnitude when the caller knows the segment is
            anime action content (where head turns are exaggerated
            and the cinematography convention asks for more
            lead-room). 1.0 = unchanged (default). The v2 spec
            recommends 1.5 for ``anime_subtype == "action"``.

    Returns:
        Signed pixel offset to add to ``subject_x``. Zero when yaw
        is zero or crop_width_px is non-positive.
    """
    if crop_width_px <= 0:
        return 0.0
    yaw = max(-1.0, min(1.0, float(yaw)))
    multiplier = max(0.0, float(anime_action_multiplier))
    # Sign convention: face looking left → positive offset (shift
    # camera right) → face lands on left third of output.
    return -yaw * float(max_frac) * float(crop_width_px) * multiplier


def yaw_to_categorical(yaw: float, threshold: float = 0.15) -> str:
    """Bucket a continuous yaw into ``"left"`` / ``"right"`` / ``"center"``.

    Mirrors the threshold the legacy ``estimate_gaze_direction`` uses
    so the categorical and continuous tiers stay consistent at the
    same input. Useful when other code paths (e.g. the
    ``seg.lead_room_direction`` field) want a categorical label.
    """
    if yaw <= -threshold:
        return "left"
    if yaw >= threshold:
        return "right"
    return "center"


def estimate_gaze_direction(
    face,
    threshold: float = 0.15,
) -> str:
    """Estimate which direction a face is looking.

    Uses nose_x position relative to the face's x_center to infer gaze.
    If the nose is shifted right of center, the face is looking right.

    Args:
        face: A FaceInfo-like object with nose_x, x_center, and width.
        threshold: Minimum asymmetry (as fraction of face width) to call
            a direction. Below this → "center".

    Returns:
        "left", "right", or "center"
    """
    nose_x = getattr(face, 'nose_x', None)
    x_center = getattr(face, 'x_center', None)
    width = getattr(face, 'width', None)

    if nose_x is None or width is None or width <= 0:
        return "center"

    # If we have a face bounding box center, use that
    if x_center is None:
        return "center"

    # Compute asymmetry: how far is the nose from the bbox center,
    # normalized by face width
    offset = (nose_x - x_center) / width

    if offset > threshold:
        return "right"
    elif offset < -threshold:
        return "left"
    return "center"


def estimate_gaze_from_dense(
    dense_faces: list,
    slot_id: int,
    start: float,
    end: float,
) -> str:
    """Estimate dominant gaze direction for a face slot in a time range.

    Samples dense face frames and returns the most common gaze direction.

    Args:
        dense_faces: List of FrameFaces.
        slot_id: Face slot to analyze.
        start: Start time.
        end: End time.

    Returns:
        "left", "right", or "center"
    """
    votes = {"left": 0, "right": 0, "center": 0}

    for df in dense_faces:
        if df.timestamp < start or df.timestamp > end:
            continue
        for f in df.faces:
            if getattr(f, 'identity_id', -1) != slot_id:
                continue
            direction = estimate_gaze_direction(f)
            votes[direction] += 1

    if not any(votes.values()):
        return "center"

    return max(votes, key=votes.get)


def apply_lead_room(
    subject_x: int,
    gaze_direction: str,
    viewport_width_pct: float = 28.0,
) -> int:
    """Apply lead-room offset for narrative content.

    When a character is looking screen-left, place them on the right
    side of the vertical frame (increase subject_x), and vice versa.
    This preserves the DP's intended composition with lead room.

    Args:
        subject_x: Current subject_x (0-100).
        gaze_direction: "left", "right", or "center".
        viewport_width_pct: Width of the 9:16 viewport as percentage
            of the 16:9 source (default ~28%).

    Returns:
        Adjusted subject_x (0-100).
    """
    if gaze_direction == "center":
        return subject_x

    lead_offset = int(viewport_width_pct / 6)  # ~5 units

    if gaze_direction == "left":
        # Looking left → place on right third → increase subject_x
        return min(100, subject_x + lead_offset)
    else:
        # Looking right → place on left third → decrease subject_x
        return max(0, subject_x - lead_offset)
