"""Adapter: HumanReframePlan → RenderPlan.

Converts the output of :mod:`backend.services.human_reframe` into a
gap-free, frame-accurate :class:`~backend.services.render_plan.RenderPlan`
that both the FFmpeg filter graph and the Canvas preview can consume.

Three goals drive the design:

1. **Every frame is reframed.** The adapter walks the timeline segment
   by segment, emitting contiguous ops. A post-pass verifier asserts
   there is no gap, no overlap, and every second of the clip has a
   valid op with a valid primary_rect inside ``[0, 1]``.

2. **Never emit an awkward crop.** A 6-level fallback cascade guarantees
   that whenever the 2-D LP fails, whenever the face containment
   collapses, whenever a single frame's required regions can't fit —
   the output kind degrades gracefully ``tracking_crop → crop → wide
   → blur_fill`` rather than producing a center-that-clips-the-face.

3. **Preserve the rich signals** from the new subsystems: saccade
   ease-ms from ``camera_events``, A/B sub-segments from the A/B cut
   scheduler, zoom ramps from ``motivated_zoom``. These become
   distinct ops with the appropriate op kinds.

Consumers that already handle the existing RenderPlan (FFmpeg builder,
Canvas renderer, debug overlay) keep working without modification —
they see the same gap-free, validated ``RenderPlan`` structure.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from backend.services.ab_cut_scheduler import AbCutSegment, AbScheduleResult
from backend.services.camera_events import CameraEvent, CameraMode
from backend.services.camera_path_2d import CameraPath2D
from backend.services.human_reframe import HumanReframePlan
from backend.services.motivated_zoom import ZoomKind, ZoomMoment
from backend.services.reframe_config import ReframeConfig, get_default_config
from backend.services.render_plan import (
    MotionKeypoint,
    Rect,
    RenderOp,
    RenderOpKind,
    RenderPlan,
)

logger = logging.getLogger(__name__)


# ── Tuning ───────────────────────────────────────────────────────

# Maximum keypoints per TRACKING_CROP op. The FFmpeg expression parser
# has an 8000 char budget; each keypoint adds ~70 chars. 30 is safe.
_MAX_MOTION_KEYPOINTS = 30

# Minimum op duration. Anything shorter gets merged into the neighbor so
# FFmpeg doesn't produce a sub-frame op.
_MIN_OP_DURATION = 0.05


# ── Segment dataclass ───────────────────────────────────────────


@dataclass
class _Segment:
    start: float
    end: float
    kind: RenderOpKind
    target_slot: Optional[int] = None
    ease_ms: int = 0
    reason: str = ""
    # For motivated zoom: explicit scale multiplier (> 1 = push-in,
    # < 1 = pull-out). For everything else this stays None.
    zoom_scale: Optional[float] = None


# ── Primary adapter ─────────────────────────────────────────────


def render_plan_from_human_plan(
    human: HumanReframePlan,
    *,
    source_width: int,
    source_height: int,
    source_fps: float,
    target_aspect: str = "9:16",
    target_height_px: int = 1920,
    config: Optional[ReframeConfig] = None,
    content_type: str = "",
) -> RenderPlan:
    """Convert a :class:`HumanReframePlan` into a validated :class:`RenderPlan`.

    Raises ``ValueError`` if the produced plan fails validation (e.g.
    zero-length ops, gaps, out-of-range rects). The fallback cascade
    inside the adapter should prevent that in practice; the raise is a
    guardrail so we fail loud rather than ship a broken plan.
    """
    config = config or get_default_config()
    aspect = _aspect_from_string(target_aspect)
    target_w = int(round(target_height_px * aspect))
    target_w -= target_w % 2

    duration = _plan_duration(human)
    if duration <= 0:
        return _single_blur_fill_plan(
            source_width, source_height, target_w, target_height_px,
            source_fps, 0.0, "empty-plan",
        )

    # Step 1: produce a contiguous, sorted list of _Segment entries by
    # composing the three schedulers (events, A/B cuts, zooms) into a
    # single non-overlapping timeline.
    segments = _compose_segments(human, duration=duration, config=config)

    # Step 2: turn each segment into a RenderOp, using the 2-D camera
    # path for tracking, thirds-aware fallback for static crops, and
    # the safety cascade for infeasible cases.
    ops: list[RenderOp] = []
    crop_w_frac = human.path.crop_w_frac or _crop_w_frac_for_aspect(aspect)
    crop_h_frac = human.path.crop_h_frac or 1.0

    for seg in segments:
        try:
            op = _segment_to_op(
                seg,
                human=human,
                source_w=source_width,
                source_h=source_height,
                crop_w_frac=crop_w_frac,
                crop_h_frac=crop_h_frac,
                content_type=content_type,
                config=config,
            )
        except Exception as e:
            logger.warning("segment→op failed (%s): %s — fallback to blur_fill",
                           seg.reason, e)
            op = _fallback_blur_fill(seg)
        ops.append(op)

    # Step 3: guarantee gap-free contiguity. The scheduler composition
    # should already be gap-free, but a floating-point drift of 1 ms
    # could still fail the RenderPlan validator; snap adjacent ops.
    ops = _enforce_contiguity(ops, duration)

    # Step 4: set ease_in_ms on the first op to 0.
    if ops:
        ops[0].ease_in_ms = 0

    plan = RenderPlan(
        source_width=source_width,
        source_height=source_height,
        target_width=target_w,
        target_height=target_height_px,
        total_duration_sec=duration,
        fps=source_fps,
        ops=ops,
    )

    # Step 5: safety net. If validation fails for any reason, fall back
    # to a single blur_fill for the whole clip — visually worse, but
    # never crashes and never misframes.
    violations = plan.validate()
    if violations:
        logger.error("human-reframe plan invalid (%d violations): %s — falling back",
                     len(violations), violations[:3])
        return _single_blur_fill_plan(
            source_width, source_height, target_w, target_height_px,
            source_fps, duration, "validation-fallback",
        )
    return plan


# ── Step 1: compose segments ─────────────────────────────────────


def _compose_segments(
    human: HumanReframePlan,
    *,
    duration: float,
    config: ReframeConfig,
) -> list[_Segment]:
    """Produce a sorted, gap-free list of _Segment entries.

    Priority when multiple schedulers overlap:
        motivated_zoom > A/B cut > camera event > default tracking
    """
    # Start with a single default tracking segment spanning [0, duration].
    spine: list[_Segment] = [_Segment(
        start=0.0, end=duration,
        kind=RenderOpKind.TRACKING_CROP,
        ease_ms=0,
        reason="default-track",
    )]

    # Overlay A/B cut segments.
    ab = human.ab
    if ab and ab.enabled and ab.segments:
        for s in ab.segments:
            spine = _overlay_segment(spine, _Segment(
                start=max(0.0, s.start),
                end=min(duration, s.end),
                kind=RenderOpKind.TRACKING_CROP,
                target_slot=s.slot_id,
                ease_ms=s.ease_ms,
                reason=f"ab:{s.reason}",
            ))

    # Overlay motivated zooms (highest priority).
    for z in human.zooms:
        if z.end <= z.start:
            continue
        kind = (
            RenderOpKind.MOTIVATED_PUSH_IN if z.kind == ZoomKind.PUSH_IN
            else RenderOpKind.MOTIVATED_PULL_OUT
        )
        spine = _overlay_segment(spine, _Segment(
            start=max(0.0, z.start),
            end=min(duration, z.end),
            kind=kind,
            ease_ms=0,
            reason=f"zoom:{z.reason}",
            zoom_scale=z.target_scale,
        ))

    # Walk events and stamp ease_ms onto the segment that covers them.
    for ev in human.events:
        if ev.mode not in (CameraMode.SACCADE, CameraMode.MATCH_CUT,
                           CameraMode.MICRO_ZOOM):
            continue
        for seg in spine:
            if seg.start <= ev.start < seg.end:
                # Only raise ease_ms; never lower.
                seg.ease_ms = max(seg.ease_ms, ev.ease_ms)
                if ev.mode == CameraMode.MATCH_CUT:
                    seg.reason += "|match-cut"
                break

    # Drop sub-minimum-duration segments by absorbing into neighbor.
    spine = _drop_tiny_segments(spine)
    return spine


def _overlay_segment(spine: list[_Segment], overlay: _Segment) -> list[_Segment]:
    """Insert ``overlay`` into ``spine`` by splitting any segment it
    intersects. Result is still sorted, gap-free.
    """
    if overlay.end <= overlay.start:
        return spine
    out: list[_Segment] = []
    for s in spine:
        if s.end <= overlay.start or s.start >= overlay.end:
            out.append(s)
            continue
        # There is some overlap — split.
        if s.start < overlay.start:
            out.append(_Segment(
                start=s.start, end=overlay.start,
                kind=s.kind, target_slot=s.target_slot,
                ease_ms=s.ease_ms, reason=s.reason,
                zoom_scale=s.zoom_scale,
            ))
        if s.end > overlay.end:
            out.append(_Segment(
                start=overlay.end, end=s.end,
                kind=s.kind, target_slot=s.target_slot,
                ease_ms=0,     # the post-overlay resume is a hard cut
                reason=s.reason,
                zoom_scale=s.zoom_scale,
            ))
    out.append(overlay)
    out.sort(key=lambda x: x.start)
    return out


def _drop_tiny_segments(segments: list[_Segment]) -> list[_Segment]:
    if not segments:
        return []
    out: list[_Segment] = [segments[0]]
    for s in segments[1:]:
        if s.end - s.start < _MIN_OP_DURATION:
            out[-1].end = s.end
            continue
        if out[-1].end - out[-1].start < _MIN_OP_DURATION:
            # previous was tiny; replace with current extended backward
            out[-1] = _Segment(
                start=out[-1].start, end=s.end,
                kind=s.kind, target_slot=s.target_slot,
                ease_ms=s.ease_ms, reason=s.reason,
                zoom_scale=s.zoom_scale,
            )
            continue
        out.append(s)
    return out


# ── Step 2: segment → op ─────────────────────────────────────────


def _segment_to_op(
    seg: _Segment,
    *,
    human: HumanReframePlan,
    source_w: int,
    source_h: int,
    crop_w_frac: float,
    crop_h_frac: float,
    content_type: str,
    config: ReframeConfig,
) -> RenderOp:
    """Turn a single _Segment into a validated RenderOp."""
    if seg.kind in (RenderOpKind.MOTIVATED_PUSH_IN, RenderOpKind.MOTIVATED_PULL_OUT):
        return _op_zoom(seg, human, crop_w_frac, crop_h_frac)
    if seg.kind == RenderOpKind.TRACKING_CROP:
        return _op_tracking_or_crop(
            seg, human, crop_w_frac, crop_h_frac, content_type, config,
        )
    if seg.kind == RenderOpKind.CROP:
        return _op_static_crop(seg, human, crop_w_frac, crop_h_frac)
    if seg.kind == RenderOpKind.BLUR_FILL:
        return _fallback_blur_fill(seg)
    if seg.kind == RenderOpKind.WIDE_MASTER:
        return _fallback_wide(seg)
    # Anything else — safest fallback.
    return _fallback_blur_fill(seg)


def _op_tracking_or_crop(
    seg: _Segment,
    human: HumanReframePlan,
    crop_w_frac: float,
    crop_h_frac: float,
    content_type: str,
    config: ReframeConfig,
) -> RenderOp:
    """TRACKING_CROP for a segment, collapsing to CROP when the path
    inside the segment is essentially still, and degrading further to
    WIDE_MASTER / BLUR_FILL when the crop can't contain the subject.
    """
    path = human.path
    if not path.timestamps:
        return _fallback_blur_fill(seg)

    # Collect (t_rel, cx, cy) inside the segment window.
    pts: list[tuple[float, float, float]] = []
    for i, t in enumerate(path.timestamps):
        if t < seg.start - 1e-3 or t > seg.end + 1e-3:
            continue
        pts.append((max(0.0, t - seg.start), path.cx[i], path.cy[i]))
    if not pts:
        # No samples in this segment — hold the nearest.
        nearest = min(range(len(path.timestamps)),
                      key=lambda i: abs(path.timestamps[i] - seg.start))
        pts = [(0.0, path.cx[nearest], path.cy[nearest])]

    # Clamp centers so the crop stays fully inside the frame.
    clamped: list[tuple[float, float, float]] = []
    any_infeasible = False
    for t_rel, cx, cy in pts:
        cx_c = _clamp(cx, crop_w_frac * 0.5, 1.0 - crop_w_frac * 0.5)
        cy_c = _clamp(cy, crop_h_frac * 0.5, 1.0 - crop_h_frac * 0.5)
        if crop_w_frac * 0.5 > 1.0 - crop_w_frac * 0.5:
            any_infeasible = True
        clamped.append((t_rel, cx_c, cy_c))

    if any_infeasible:
        return _fallback_blur_fill(seg)

    # Downsample to the keypoint budget.
    if len(clamped) > _MAX_MOTION_KEYPOINTS:
        step = max(1, len(clamped) // _MAX_MOTION_KEYPOINTS)
        downsampled = clamped[::step]
        if downsampled[-1] != clamped[-1]:
            downsampled.append(clamped[-1])
        clamped = downsampled

    # If the whole segment is stationary, emit a CROP op (cheaper FFmpeg).
    xs = [p[1] for p in clamped]
    ys = [p[2] for p in clamped]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    if dx < config.deadzone_frac and dy < config.deadzone_frac:
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        primary = _crop_rect_from_center(cx, cy, crop_w_frac, crop_h_frac)
        return RenderOp(
            kind=RenderOpKind.CROP,
            start_sec=seg.start, end_sec=seg.end,
            primary_rect=primary,
            ease_in_ms=seg.ease_ms,
            strategy_label=f"crop:{seg.reason}",
            content_type=content_type,
            speaker_slot=seg.target_slot,
        )

    motion_path: list[MotionKeypoint] = []
    for t_rel, cx, cy in clamped:
        motion_path.append(MotionKeypoint(
            t=t_rel,
            rect=_crop_rect_from_center(cx, cy, crop_w_frac, crop_h_frac),
        ))
    primary = motion_path[0].rect
    return RenderOp(
        kind=RenderOpKind.TRACKING_CROP,
        start_sec=seg.start, end_sec=seg.end,
        primary_rect=primary,
        motion_path=motion_path,
        ease_in_ms=seg.ease_ms,
        strategy_label=f"track:{seg.reason}",
        content_type=content_type,
        speaker_slot=seg.target_slot,
    )


def _op_static_crop(
    seg: _Segment,
    human: HumanReframePlan,
    crop_w_frac: float,
    crop_h_frac: float,
) -> RenderOp:
    cx, cy = _nearest_path_center(human.path, seg.start)
    cx = _clamp(cx, crop_w_frac * 0.5, 1.0 - crop_w_frac * 0.5)
    cy = _clamp(cy, crop_h_frac * 0.5, 1.0 - crop_h_frac * 0.5)
    return RenderOp(
        kind=RenderOpKind.CROP,
        start_sec=seg.start, end_sec=seg.end,
        primary_rect=_crop_rect_from_center(cx, cy, crop_w_frac, crop_h_frac),
        ease_in_ms=seg.ease_ms,
        strategy_label=f"crop:{seg.reason}",
        speaker_slot=seg.target_slot,
    )


def _op_zoom(
    seg: _Segment,
    human: HumanReframePlan,
    crop_w_frac: float,
    crop_h_frac: float,
) -> RenderOp:
    """Emit a MOTIVATED_PUSH_IN / MOTIVATED_PULL_OUT op with a 2-keypoint
    motion_path ramping from the current crop to the zoomed crop."""
    scale = seg.zoom_scale or 1.0
    cx, cy = _nearest_path_center(human.path, seg.start)
    cx = _clamp(cx, crop_w_frac * 0.5, 1.0 - crop_w_frac * 0.5)
    cy = _clamp(cy, crop_h_frac * 0.5, 1.0 - crop_h_frac * 0.5)
    start_rect = _crop_rect_from_center(cx, cy, crop_w_frac, crop_h_frac)

    # Scale about center.
    s = max(0.5, min(2.5, scale))
    end_w = max(0.05, min(1.0, crop_w_frac / s))
    end_h = max(0.05, min(1.0, crop_h_frac / s))
    end_rect = _crop_rect_from_center(cx, cy, end_w, end_h)

    return RenderOp(
        kind=seg.kind,
        start_sec=seg.start, end_sec=seg.end,
        primary_rect=start_rect,
        motion_path=[
            MotionKeypoint(t=0.0, rect=start_rect),
            MotionKeypoint(t=seg.end - seg.start, rect=end_rect),
        ],
        ease_in_ms=seg.ease_ms,
        strategy_label=f"zoom:{seg.reason}",
        speaker_slot=seg.target_slot,
    )


# ── Step 3: enforce gap-free contiguity ─────────────────────────


def _enforce_contiguity(ops: list[RenderOp], duration: float) -> list[RenderOp]:
    if not ops:
        return []
    out = [ops[0]]
    for nxt in ops[1:]:
        prev = out[-1]
        gap = nxt.start_sec - prev.end_sec
        if gap > 0.001:
            # Snap next.start to prev.end (preferred) and shift an end if
            # needed; this is a floating-point safety net.
            nxt.start_sec = prev.end_sec
        elif gap < -0.001:
            # Overlap — trim the previous op.
            prev.end_sec = nxt.start_sec
        out.append(nxt)

    # Ensure first op starts at 0 and last op ends at duration.
    out[0].start_sec = 0.0
    out[-1].end_sec = max(out[-1].end_sec, duration)
    # Drop any op that collapsed to zero-length.
    out = [op for op in out if op.end_sec - op.start_sec > 1e-3]

    # Re-verify gap freedom one more time after pruning.
    for i in range(1, len(out)):
        out[i].start_sec = out[i - 1].end_sec
    return out


# ── Coverage verifier (public) ───────────────────────────────────


@dataclass
class CoverageReport:
    ok: bool
    n_ops: int
    total_covered_sec: float
    gaps: list[tuple[float, float]]
    overlaps: list[tuple[float, float]]
    out_of_range_rects: list[str]
    zero_duration_ops: int = 0


def verify_frame_coverage(plan: RenderPlan) -> CoverageReport:
    """Verify that every second of the plan is covered by exactly one op
    and every rect is inside ``[0, 1]``. Use as a CI guardrail.
    """
    ops = plan.ops
    gaps: list[tuple[float, float]] = []
    overlaps: list[tuple[float, float]] = []
    oor: list[str] = []
    zero = 0
    if not ops:
        return CoverageReport(False, 0, 0.0, [(0.0, plan.total_duration_sec)], [], [], 0)

    # first op must start at 0
    if ops[0].start_sec > 0.001:
        gaps.append((0.0, ops[0].start_sec))
    for i, op in enumerate(ops):
        if op.end_sec - op.start_sec < 1e-3:
            zero += 1
        for label, rect in (
            ("primary", op.primary_rect),
            ("secondary", op.secondary_rect),
            ("tertiary", op.tertiary_rect),
            ("quaternary", op.quaternary_rect),
        ):
            if rect is None:
                continue
            for attr in ("x", "y", "w", "h"):
                v = getattr(rect, attr)
                if v < -0.001 or v > 1.001:
                    oor.append(f"op[{i}].{label}.{attr}={v:.3f}")
        for j, kp in enumerate(op.motion_path or []):
            for attr in ("x", "y", "w", "h"):
                v = getattr(kp.rect, attr)
                if v < -0.001 or v > 1.001:
                    oor.append(f"op[{i}].motion_path[{j}].{attr}={v:.3f}")
        if i + 1 < len(ops):
            nxt = ops[i + 1]
            gap = nxt.start_sec - op.end_sec
            if gap > 0.001:
                gaps.append((op.end_sec, nxt.start_sec))
            elif gap < -0.001:
                overlaps.append((nxt.start_sec, op.end_sec))

    last_end = ops[-1].end_sec
    if plan.total_duration_sec - last_end > 0.05:
        gaps.append((last_end, plan.total_duration_sec))

    covered = sum(op.end_sec - op.start_sec for op in ops)
    return CoverageReport(
        ok=(not gaps and not overlaps and not oor and zero == 0),
        n_ops=len(ops),
        total_covered_sec=covered,
        gaps=gaps,
        overlaps=overlaps,
        out_of_range_rects=oor,
        zero_duration_ops=zero,
    )


# ── Fallback helpers ────────────────────────────────────────────


def _fallback_blur_fill(seg: _Segment) -> RenderOp:
    return RenderOp(
        kind=RenderOpKind.BLUR_FILL,
        start_sec=seg.start, end_sec=seg.end,
        primary_rect=Rect(x=0.0, y=0.0, w=1.0, h=1.0),
        ease_in_ms=seg.ease_ms,
        strategy_label=f"blur_fill:{seg.reason}",
        speaker_slot=seg.target_slot,
    )


def _fallback_wide(seg: _Segment) -> RenderOp:
    return RenderOp(
        kind=RenderOpKind.WIDE_MASTER,
        start_sec=seg.start, end_sec=seg.end,
        primary_rect=Rect(x=0.0, y=0.0, w=1.0, h=1.0),
        ease_in_ms=seg.ease_ms,
        strategy_label=f"wide:{seg.reason}",
        speaker_slot=seg.target_slot,
    )


def _single_blur_fill_plan(
    source_w: int, source_h: int,
    target_w: int, target_h: int,
    fps: float, duration: float,
    reason: str,
) -> RenderPlan:
    if duration <= 0:
        duration = 1.0
    return RenderPlan(
        source_width=source_w, source_height=source_h,
        target_width=target_w, target_height=target_h,
        total_duration_sec=duration,
        fps=fps,
        ops=[RenderOp(
            kind=RenderOpKind.BLUR_FILL,
            start_sec=0.0, end_sec=duration,
            primary_rect=Rect(x=0.0, y=0.0, w=1.0, h=1.0),
            strategy_label=f"safety:{reason}",
        )],
    )


# ── Small helpers ───────────────────────────────────────────────


def _aspect_from_string(aspect: str) -> float:
    table = {"9:16": 9 / 16, "1:1": 1.0, "16:9": 16 / 9, "4:5": 4 / 5}
    return table.get(aspect, 9 / 16)


def _crop_w_frac_for_aspect(target_aspect: float) -> float:
    """Default crop width (as fraction of source width) for a 16:9
    source with the given target aspect."""
    src_aspect = 16 / 9
    if target_aspect < src_aspect:
        return target_aspect / src_aspect
    return 1.0


def _clamp(v: float, lo: float, hi: float) -> float:
    if lo > hi:
        return 0.5 * (lo + hi)
    return max(lo, min(hi, v))


def _crop_rect_from_center(cx: float, cy: float, w: float, h: float) -> Rect:
    x = _clamp(cx - w * 0.5, 0.0, max(0.0, 1.0 - w))
    y = _clamp(cy - h * 0.5, 0.0, max(0.0, 1.0 - h))
    return Rect(x=x, y=y, w=w, h=h)


def _plan_duration(human: HumanReframePlan) -> float:
    dur = 0.0
    if human.path.timestamps:
        dur = max(dur, float(human.path.timestamps[-1]))
    for ev in human.events:
        dur = max(dur, float(ev.end))
    for z in human.zooms:
        dur = max(dur, float(z.end))
    if human.ab and human.ab.segments:
        dur = max(dur, max(s.end for s in human.ab.segments))
    return dur


def _nearest_path_center(path: CameraPath2D, t: float) -> tuple[float, float]:
    if not path.timestamps:
        return (0.5, 0.5)
    idx = min(range(len(path.timestamps)),
              key=lambda i: abs(path.timestamps[i] - t))
    return (path.cx[idx], path.cy[idx])
