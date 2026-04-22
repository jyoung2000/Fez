"""Blueprint v2 Phase 3 — ``splice_segment`` pure helper.

Replaces the ops inside ``[start, end]`` of a ``RenderPlan`` with ops
from a ``segment`` plan whose timeline starts at 0. Keeps callers
from having to pull the full ``human_render_plan_adapter`` graph
(camera_path_2d, camera_events, numpy …) for a plan-level edit.

Also contains the tiny ``_enforce_contiguity`` helper used to snap
floating-point drift at op boundaries so the spliced plan passes
coverage verification.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from backend.services.render_plan import RenderOp, RenderPlan

logger = logging.getLogger(__name__)


def _enforce_contiguity(ops: list[RenderOp], total: float) -> list[RenderOp]:
    """Snap adjacent ops so the plan is gap-free + ends at ``total``.

    Floating-point drift of a few µs would otherwise fail the render
    plan validator on re-splice. This only adjusts op boundaries; it
    never drops or re-orders ops.
    """
    if not ops:
        return ops
    out = [ops[0]]
    for op in ops[1:]:
        prev = out[-1]
        if abs(op.start_sec - prev.end_sec) < 0.01 and op.start_sec != prev.end_sec:
            op = replace(op, start_sec=prev.end_sec)
        out.append(op)
    # Snap the final op to cover the full duration.
    if abs(out[-1].end_sec - total) < 0.05 and out[-1].end_sec != total:
        out[-1] = replace(out[-1], end_sec=total)
    return out


def splice_segment(
    original: RenderPlan,
    segment: RenderPlan,
    *,
    start: float,
    end: float,
) -> RenderPlan:
    """Replace ops inside ``[start, end]`` of ``original`` with ops
    from ``segment`` (whose timeline is assumed to start at 0).

    Steps:
      1. Keep every op that fully precedes ``start``.
      2. Trim any op that straddles ``start`` so it ends at ``start``.
      3. Shift every op in ``segment`` by ``+start`` and clamp the
         first / last ops to ``[start, end]`` so they cover the
         window exactly (scales the inner timeline if it doesn't
         match the window).
      4. Trim any op that straddles ``end`` so it begins at ``end``.
      5. Keep every op that fully follows ``end``.

    If the resulting plan fails ``RenderPlan.validate()`` returns
    ``original`` unchanged so the caller never ships a broken plan.
    Never raises.
    """
    if not original.ops:
        return original
    total = float(original.total_duration_sec)
    start = max(0.0, float(start))
    end = max(start, min(total, float(end)))
    if end - start <= 1e-3:
        return original

    new_ops: list[RenderOp] = []

    # (1) + (2): ops up to and across ``start``.
    for op in original.ops:
        if op.end_sec <= start + 1e-6:
            new_ops.append(op)
            continue
        if op.start_sec < start:
            trimmed = replace(op, end_sec=float(start))
            if trimmed.end_sec - trimmed.start_sec > 1e-3:
                new_ops.append(trimmed)

    # (3): segment ops, shifted to land inside ``[start, end]``.
    seg_end_origin = 0.0
    for op in segment.ops:
        seg_end_origin = max(seg_end_origin, float(op.end_sec))
    if seg_end_origin <= 1e-6:
        return original

    inner_dur = float(end) - float(start)
    scale = (
        inner_dur / seg_end_origin
        if abs(seg_end_origin - inner_dur) > 1e-3
        else 1.0
    )

    seg_ops: list[RenderOp] = []
    for op in segment.ops:
        shifted_start = float(op.start_sec) * scale + start
        shifted_end = float(op.end_sec) * scale + start
        shifted_start = max(start, min(end, shifted_start))
        shifted_end = max(start, min(end, shifted_end))
        if shifted_end - shifted_start <= 1e-3:
            continue
        seg_ops.append(replace(
            op, start_sec=shifted_start, end_sec=shifted_end,
        ))

    # Force the segment to cover ``[start, end]`` exactly.
    if seg_ops:
        seg_ops[0] = replace(seg_ops[0], start_sec=start)
        seg_ops[-1] = replace(seg_ops[-1], end_sec=end)
    new_ops.extend(seg_ops)

    # (4) + (5): ops after ``end``.
    for op in original.ops:
        if op.start_sec >= end - 1e-6:
            new_ops.append(op)
            continue
        if op.end_sec > end and op.start_sec < end:
            trimmed = replace(op, start_sec=float(end))
            if trimmed.end_sec - trimmed.start_sec > 1e-3:
                new_ops.append(trimmed)

    new_ops.sort(key=lambda o: o.start_sec)
    new_ops = _enforce_contiguity(new_ops, total)

    spliced = RenderPlan(
        source_width=original.source_width,
        source_height=original.source_height,
        target_width=original.target_width,
        target_height=original.target_height,
        total_duration_sec=total,
        fps=original.fps,
        ops=new_ops,
        source_offset_sec=getattr(original, "source_offset_sec", 0.0),
    )

    if spliced.validate():
        logger.info(
            "splice_segment produced invalid plan (window=[%.2f, %.2f]) "
            "— reverting to original",
            start, end,
        )
        return original
    return spliced
