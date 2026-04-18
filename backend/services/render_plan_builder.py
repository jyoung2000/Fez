"""Build a RenderPlan from ReframeSegmenter output.

Single source of truth: both the FFmpeg filter builder and the frontend
Canvas renderer consume the same RenderPlan JSON.
"""

import logging
from typing import List, Optional, Tuple

from backend.services.render_plan import (
    MotionKeypoint,
    Rect,
    RenderOp,
    RenderOpKind,
    RenderPlan,
)

logger = logging.getLogger(__name__)

# Map segmenter layout/strategy strings to RenderOpKind
_STRATEGY_TO_KIND = {
    "stationary": RenderOpKind.CROP,
    "tracking": RenderOpKind.TRACKING_CROP,
    "panning": RenderOpKind.TRACKING_CROP,
    "wide_master": RenderOpKind.WIDE_MASTER,
    "blur_fill": RenderOpKind.BLUR_FILL,
    "split_screen": RenderOpKind.SPLIT_SCREEN,
    "stacked_gameplay": RenderOpKind.STACKED_GAMEPLAY,
    "grid": RenderOpKind.GRID_2X2,
}

# Aspect ratio string -> float (width / height)
_ASPECT_RATIOS = {
    "9:16": 9 / 16,
    "1:1": 1.0,
    "16:9": 16 / 9,
    "4:5": 4 / 5,
}

# Maximum motion_path keypoints per op (decimate if segmenter gives more)
_MAX_MOTION_KEYPOINTS = 20


def build_render_plan(
    segments,
    source_width: int,
    source_height: int,
    source_fps: float,
    target_aspect: str = "9:16",
    target_height_px: int = 1920,
    clip_range: Optional[Tuple[float, float]] = None,
) -> RenderPlan:
    """Build a RenderPlan from ReframeSegmenter segments.

    Args:
        segments: List of ReframeSegment from build_reframe_segments().
        source_width: Original video width in pixels.
        source_height: Original video height in pixels.
        source_fps: Source video frame rate.
        target_aspect: Output aspect ratio string ("9:16", "1:1", "16:9").
        target_height_px: Output height in pixels.
        clip_range: Optional (start_sec, end_sec) for clip exports.
            When provided, segments are trimmed and rebased to start at 0.0.

    Returns:
        A validated RenderPlan.

    Raises:
        ValueError: If the plan fails validation.
    """
    # Step 1: Compute target dimensions
    aspect_ratio = _ASPECT_RATIOS.get(target_aspect, 9 / 16)
    target_width = int(round(target_height_px * aspect_ratio))
    target_width = target_width - (target_width % 2)  # ensure even

    # Step 2: Handle clip range (trim + rebase)
    source_offset_sec = 0.0
    working_segments = list(segments)

    if clip_range is not None:
        clip_start, clip_end = clip_range
        source_offset_sec = clip_start
        working_segments = _trim_segments_to_clip(working_segments, clip_start, clip_end)

    if not working_segments:
        # No segments — create a single CROP at center
        total_dur = clip_range[1] - clip_range[0] if clip_range else 0.0
        crop_rect = _compute_crop_rect(50, 40, source_width, source_height, aspect_ratio)
        plan = RenderPlan(
            source_width=source_width,
            source_height=source_height,
            target_width=target_width,
            target_height=target_height_px,
            total_duration_sec=total_dur,
            fps=source_fps,
            ops=[RenderOp(
                kind=RenderOpKind.CROP,
                start_sec=0.0,
                end_sec=total_dur,
                primary_rect=crop_rect,
                strategy_label="fallback_center",
            )],
            source_offset_sec=source_offset_sec,
        )
        _validate_and_raise(plan)
        return plan

    # Step 3: Map each segment to a RenderOp
    ops = []
    for seg in working_segments:
        op = _segment_to_op(seg, source_width, source_height, aspect_ratio, source_fps)
        ops.append(op)

    # Step 4: Enforce contiguity
    ops = _enforce_contiguity(ops, source_width, source_height, aspect_ratio)

    # Step 5: Set ease_in_ms from segment data
    # Already set in _segment_to_op, but cross-check shot boundary transitions
    # (first op always has ease_in_ms=0)
    if ops:
        ops[0].ease_in_ms = 0

    # Compute total duration from last op
    total_dur = ops[-1].end_sec if ops else 0.0

    plan = RenderPlan(
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height_px,
        total_duration_sec=total_dur,
        fps=source_fps,
        ops=ops,
        source_offset_sec=source_offset_sec,
    )

    _validate_and_raise(plan)
    return plan


def _trim_segments_to_clip(segments, clip_start: float, clip_end: float):
    """Filter and trim segments to a clip window, rebase times to 0.0."""
    result = []
    for seg in segments:
        seg_start = getattr(seg, "start", 0.0)
        seg_end = getattr(seg, "end", 0.0)

        # Skip non-overlapping segments
        if seg_end <= clip_start or seg_start >= clip_end:
            continue

        # Clone segment attributes for trimming
        trimmed_start = max(seg_start, clip_start)
        trimmed_end = min(seg_end, clip_end)

        # Rebase to clip-relative time (0-based)
        rebased_start = trimmed_start - clip_start
        rebased_end = trimmed_end - clip_start

        # Create a lightweight wrapper that preserves all attributes but
        # overrides start/end with rebased values
        result.append(_RebasedSegment(seg, rebased_start, rebased_end))

    return result


class _RebasedSegment:
    """Wrapper that presents a segment with rebased start/end times."""

    def __init__(self, original, start: float, end: float):
        self._original = original
        self.start = start
        self.end = end

    def __getattr__(self, name):
        if name in ("start", "end", "_original"):
            raise AttributeError(name)
        return getattr(self._original, name)


def _segment_to_op(seg, source_w: int, source_h: int, aspect_ratio: float, fps: float) -> RenderOp:
    """Convert a single ReframeSegment to a RenderOp."""
    strategy = getattr(seg, "strategy", "stationary") or "stationary"
    layout = getattr(seg, "layout", "single") or "single"
    kind = _STRATEGY_TO_KIND.get(strategy)

    # Fallback: infer kind from layout if strategy doesn't map
    if kind is None:
        _layout_to_kind = {
            "single": RenderOpKind.CROP,
            "split": RenderOpKind.SPLIT_SCREEN,
            "triple": RenderOpKind.GRID_2X2,
            "grid": RenderOpKind.GRID_2X2,
            "wide_master": RenderOpKind.WIDE_MASTER,
            "blur_fill": RenderOpKind.BLUR_FILL,
            "stacked_gameplay": RenderOpKind.STACKED_GAMEPLAY,
        }
        kind = _layout_to_kind.get(layout, RenderOpKind.CROP)

    subject_x = getattr(seg, "subject_x", 50)
    subject_y = getattr(seg, "subject_y", 40)
    ease_in_ms = getattr(seg, "ease_in_ms", 0)
    content_type = getattr(seg, "content_type", "unknown") or "unknown"
    reason = getattr(seg, "reason", "") or ""
    motion_data = getattr(seg, "motion_path", None)
    # Phase 5 (gaming): per-segment layout mode populated by
    # ``gaming_layout_chooser.choose_gaming_layout``.
    gaming_layout_mode = getattr(seg, "gaming_layout_mode", None)
    speaker_slot = getattr(seg, "active_slot", None)

    if kind == RenderOpKind.CROP:
        primary_rect = _compute_crop_rect(subject_x, subject_y, source_w, source_h, aspect_ratio)
        return RenderOp(
            kind=kind,
            start_sec=seg.start,
            end_sec=seg.end,
            primary_rect=primary_rect,
            ease_in_ms=ease_in_ms,
            strategy_label=f"{strategy}_{reason}",
            content_type=content_type,
            gaming_layout_mode=gaming_layout_mode,
            speaker_slot=speaker_slot,
        )

    elif kind == RenderOpKind.TRACKING_CROP:
        primary_rect = _compute_crop_rect(subject_x, subject_y, source_w, source_h, aspect_ratio)
        motion_path = _build_motion_path(
            motion_data, seg.start, seg.end, source_w, source_h, aspect_ratio, fps,
        )
        # If no motion data, fall back to a static CROP
        if len(motion_path) < 2:
            return RenderOp(
                kind=RenderOpKind.CROP,
                start_sec=seg.start,
                end_sec=seg.end,
                primary_rect=primary_rect,
                ease_in_ms=ease_in_ms,
                strategy_label=f"{strategy}_{reason}_static_fallback",
                content_type=content_type,
                gaming_layout_mode=gaming_layout_mode,
                speaker_slot=speaker_slot,
            )
        return RenderOp(
            kind=kind,
            start_sec=seg.start,
            end_sec=seg.end,
            primary_rect=primary_rect,
            motion_path=motion_path,
            ease_in_ms=ease_in_ms,
            strategy_label=f"{strategy}_{reason}",
            content_type=content_type,
            gaming_layout_mode=gaming_layout_mode,
            speaker_slot=speaker_slot,
        )

    elif kind == RenderOpKind.WIDE_MASTER:
        # Full source frame — renderer will letterbox
        return RenderOp(
            kind=kind,
            start_sec=seg.start,
            end_sec=seg.end,
            primary_rect=Rect(x=0.0, y=0.0, w=1.0, h=1.0),
            ease_in_ms=ease_in_ms,
            strategy_label=f"wide_master_{reason}",
            content_type=content_type,
            speaker_slot=speaker_slot,
        )

    elif kind == RenderOpKind.BLUR_FILL:
        # Full source frame — renderer will center + blur background
        return RenderOp(
            kind=kind,
            start_sec=seg.start,
            end_sec=seg.end,
            primary_rect=Rect(x=0.0, y=0.0, w=1.0, h=1.0),
            ease_in_ms=ease_in_ms,
            strategy_label=f"blur_fill_{reason}",
            content_type=content_type,
            speaker_slot=speaker_slot,
        )

    elif kind == RenderOpKind.SPLIT_SCREEN:
        primary_rect, secondary_rect = _compute_split_rects(
            seg, source_w, source_h, aspect_ratio, ratio_top=0.5, ratio_bottom=0.5,
        )
        return RenderOp(
            kind=kind,
            start_sec=seg.start,
            end_sec=seg.end,
            primary_rect=primary_rect,
            secondary_rect=secondary_rect,
            ease_in_ms=ease_in_ms,
            strategy_label=f"split_screen_{reason}",
            content_type=content_type,
            speaker_slot=speaker_slot,
        )

    elif kind == RenderOpKind.STACKED_GAMEPLAY:
        primary_rect, secondary_rect = _compute_stacked_gameplay_rects(
            seg, source_w, source_h, aspect_ratio,
        )
        return RenderOp(
            kind=kind,
            start_sec=seg.start,
            end_sec=seg.end,
            primary_rect=primary_rect,
            secondary_rect=secondary_rect,
            ease_in_ms=ease_in_ms,
            strategy_label=f"stacked_gameplay_{reason}",
            content_type=content_type,
            speaker_slot=speaker_slot,
        )

    elif kind == RenderOpKind.GRID_2X2:
        rects = _compute_grid_rects(seg, source_w, source_h, aspect_ratio)
        return RenderOp(
            kind=kind,
            start_sec=seg.start,
            end_sec=seg.end,
            primary_rect=rects[0],
            secondary_rect=rects[1],
            tertiary_rect=rects[2],
            quaternary_rect=rects[3],
            ease_in_ms=ease_in_ms,
            strategy_label=f"grid_2x2_{reason}",
            content_type=content_type,
            speaker_slot=speaker_slot,
        )

    # Should never reach here
    primary_rect = _compute_crop_rect(subject_x, subject_y, source_w, source_h, aspect_ratio)
    return RenderOp(
        kind=RenderOpKind.CROP,
        start_sec=seg.start,
        end_sec=seg.end,
        primary_rect=primary_rect,
        ease_in_ms=ease_in_ms,
        strategy_label=f"fallback_{strategy}",
        content_type=content_type,
        speaker_slot=speaker_slot,
    )


def _compute_crop_rect(
    subject_x,
    subject_y,
    source_w: int,
    source_h: int,
    target_aspect: float,
) -> Rect:
    """Compute a normalized crop Rect for a 9:16 (or other) window.

    subject_x/subject_y can be:
      - Pixel values (float > 100 or float when source dims known)
      - Legacy 0-100 scale (int)
    Auto-detects based on value range and type.

    For vertical crops (target narrower than source):
      - Crop width determined by target aspect and source height
      - Full source height used
      - Horizontal position centered on subject_x with edge clamping
    For horizontal crops (target wider than source):
      - Full source width used
      - Crop height determined by target aspect and source width
    """
    src_aspect = source_w / source_h if source_h > 0 else 1.0

    # Normalize subject_x/y to 0.0-1.0 range
    # If subject_x is in pixel space (> 100 or float), convert to normalized
    sx = float(subject_x)
    sy = float(subject_y)
    if sx > 100.0 or (isinstance(subject_x, float) and source_w > 0 and sx > 1.0):
        # Pixel space → normalized
        center_x = sx / source_w
    else:
        # Legacy 0-100 space → normalized
        center_x = sx / 100.0

    if sy > 100.0 or (isinstance(subject_y, float) and source_h > 0 and sy > 1.0):
        center_y = sy / source_h
    else:
        center_y = sy / 100.0

    if target_aspect < src_aspect:
        # Vertical crop: narrower than source (e.g., 9:16 from 16:9)
        crop_w_norm = (target_aspect * source_h) / source_w
        crop_h_norm = 1.0

        x = center_x - crop_w_norm / 2.0
        # Clamp to [0, 1 - crop_w_norm]
        x = max(0.0, min(x, 1.0 - crop_w_norm))
        y = 0.0
    else:
        # Horizontal crop or same aspect: full width, crop height
        crop_w_norm = 1.0
        crop_h_norm = (source_w / target_aspect) / source_h
        crop_h_norm = min(1.0, crop_h_norm)

        x = 0.0
        y = center_y - crop_h_norm / 2.0
        y = max(0.0, min(y, 1.0 - crop_h_norm))

    return Rect(x=x, y=y, w=crop_w_norm, h=crop_h_norm)


def _build_motion_path(
    motion_data,
    seg_start: float,
    seg_end: float,
    source_w: int,
    source_h: int,
    aspect_ratio: float,
    fps: float,
) -> List[MotionKeypoint]:
    """Build motion_path keypoints from segment motion data.

    motion_data is expected to be [(t, x, y), ...] from the segmenter,
    where t is absolute time, x/y are 0-100 subject positions.
    """
    if not motion_data:
        return []

    keypoints = []
    for point in motion_data:
        if len(point) < 3:
            continue
        t_abs, mx, my = point[0], point[1], point[2]
        # Convert to segment-relative time
        t_rel = t_abs - seg_start
        if t_rel < -0.01 or t_rel > (seg_end - seg_start) + 0.01:
            continue
        t_rel = max(0.0, min(t_rel, seg_end - seg_start))
        rect = _compute_crop_rect(int(mx), int(my), source_w, source_h, aspect_ratio)
        keypoints.append(MotionKeypoint(t=t_rel, rect=rect))

    # Decimate if too many keypoints
    if len(keypoints) > _MAX_MOTION_KEYPOINTS:
        step = len(keypoints) / _MAX_MOTION_KEYPOINTS
        decimated = []
        for i in range(_MAX_MOTION_KEYPOINTS):
            idx = int(i * step)
            decimated.append(keypoints[idx])
        # Always include the last keypoint
        if decimated[-1].t != keypoints[-1].t:
            decimated[-1] = keypoints[-1]
        keypoints = decimated

    return keypoints


def _compute_split_rects(
    seg,
    source_w: int,
    source_h: int,
    aspect_ratio: float,
    ratio_top: float = 0.5,
    ratio_bottom: float = 0.5,
) -> Tuple[Rect, Rect]:
    """Compute top and bottom crop rects for SPLIT_SCREEN.

    Each half gets a crop from the source that fills its portion of the output.
    The top and bottom halves of the target have aspect ratio = target_w / (target_h * ratio).
    """
    # For the top half, the sub-aspect is target_aspect / ratio_top
    # This means the source crop should be wider to fill the half-height slot
    half_aspect = aspect_ratio / ratio_top  # e.g., (9/16) / 0.5 = 9/8

    subject_x = getattr(seg, "subject_x", 50)

    # Use face slot positions if available for split layout
    # Top speaker: try to use the segment's subject_x
    # Bottom speaker: offset or use a default
    # In practice, the segmenter marks split when 2 speakers overlap.
    # We use the segment's subject_x for the primary and mirror for secondary.
    primary_rect = _compute_crop_rect(subject_x, 40, source_w, source_h, half_aspect)

    # Secondary: use opposite side. If subject is left (<50), put secondary right.
    secondary_x = 100 - subject_x if subject_x != 50 else 50
    secondary_rect = _compute_crop_rect(secondary_x, 40, source_w, source_h, half_aspect)

    return primary_rect, secondary_rect


def _compute_stacked_gameplay_rects(
    seg,
    source_w: int,
    source_h: int,
    aspect_ratio: float,
) -> Tuple[Rect, Rect]:
    """Compute gameplay (top 60%) and facecam (bottom 40%) rects.

    Primary: action-centered gameplay crop from the main source.
    Secondary: detected facecam region (or a default corner).
    """
    # Gameplay: top 60% of output -> crop aspect = target_aspect / 0.6
    gameplay_aspect = aspect_ratio / 0.6
    subject_x = getattr(seg, "subject_x", 50)
    primary_rect = _compute_crop_rect(subject_x, 40, source_w, source_h, gameplay_aspect)

    # Facecam: try to use hard_constraints or persistent region info
    # Default: bottom-right corner of source, typical facecam location
    hard_constraints = getattr(seg, "hard_constraints", None)
    if hard_constraints and len(hard_constraints) > 0:
        # Use the first constraint as facecam region (already normalized or in pixels)
        hc = hard_constraints[0]
        if len(hc) >= 4:
            # If values > 1, assume pixel coordinates
            hx, hy, hw, hh = hc[0], hc[1], hc[2], hc[3]
            if max(hx, hy, hw, hh) > 1.0:
                hx, hy = hx / source_w, hy / source_h
                hw, hh = hw / source_w, hh / source_h
            secondary_rect = Rect(x=hx, y=hy, w=hw, h=hh)
        else:
            secondary_rect = Rect(x=0.65, y=0.65, w=0.30, h=0.30)
    else:
        # Default facecam position: bottom-right quarter
        secondary_rect = Rect(x=0.65, y=0.65, w=0.30, h=0.30)

    return primary_rect, secondary_rect


def _compute_grid_rects(
    seg,
    source_w: int,
    source_h: int,
    aspect_ratio: float,
) -> List[Rect]:
    """Compute 4 crop rects for GRID_2X2 layout.

    Each tile fills 1/4 of the output (2x2 grid), so each tile's source
    aspect ratio is target_aspect * 2 (twice as wide per tile height).
    """
    tile_aspect = aspect_ratio * 2  # each tile is half-width, half-height of output

    # Spread tiles across the source frame
    # Use face positions if available, otherwise space evenly
    positions = [25, 75, 25, 75]  # default x positions for 4 tiles
    y_positions = [30, 30, 60, 60]

    rects = []
    for i in range(4):
        rect = _compute_crop_rect(positions[i], y_positions[i], source_w, source_h, tile_aspect)
        rects.append(rect)

    return rects


def _enforce_contiguity(
    ops: List[RenderOp],
    source_w: int,
    source_h: int,
    aspect_ratio: float,
) -> List[RenderOp]:
    """Ensure ops are contiguous with no gaps or overlaps.

    - Gaps < 50ms: snap the next op's start to the previous op's end.
    - Gaps >= 50ms: insert a bridging CROP op holding the last frame.
    - Overlaps: snap the next op's start to the previous op's end.
    """
    if not ops:
        return ops

    result = [ops[0]]

    for i in range(1, len(ops)):
        prev = result[-1]
        curr = ops[i]

        gap = curr.start_sec - prev.end_sec
        if abs(gap) < 0.001:
            # Close enough — no adjustment needed
            result.append(curr)
        elif gap < 0:
            # Overlap — snap current start to previous end
            curr.start_sec = prev.end_sec
            if curr.start_sec < curr.end_sec:
                result.append(curr)
        elif gap < 0.05:
            # Small gap — snap
            curr.start_sec = prev.end_sec
            result.append(curr)
        else:
            # Large gap — insert bridging op
            bridge = RenderOp(
                kind=RenderOpKind.CROP,
                start_sec=prev.end_sec,
                end_sec=curr.start_sec,
                primary_rect=prev.primary_rect,
                ease_in_ms=0,
                strategy_label="bridge_hold",
            )
            result.append(bridge)
            result.append(curr)

    return result


def _validate_and_raise(plan: RenderPlan):
    """Validate the plan and raise ValueError on any violation."""
    violations = plan.validate()
    if violations:
        msg = "RenderPlan validation failed:\n" + "\n".join(f"  - {v}" for v in violations)
        logger.error(msg)
        raise ValueError(msg)
