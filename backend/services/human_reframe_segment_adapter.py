"""Adapter: ``HumanReframePlan`` → ``list[ReframeSegment]``.

Phase A wiring shim. The downstream pipeline consumes a list of
:class:`~backend.services.reframe_segmenter.ReframeSegment` (which
feeds :func:`~backend.services.render_plan_builder.build_render_plan`
and the ``SceneDescription`` hoist in ``pipeline.py``), not the
:class:`~backend.services.render_plan.RenderPlan` that
:func:`~backend.services.human_render_plan_adapter.render_plan_from_human_plan`
emits. This module bridges the two so the SOTA human-reframe path
can slot into the pipeline at the same place the legacy
``build_autoflip_segments`` / ``build_reframe_segments`` results do.

The shim goes *through* the render-plan adapter on purpose:

1. ``render_plan_from_human_plan`` already owns the contiguity /
   coverage / fallback-cascade logic and is exercised by the full
   adapter test suite. Piggy-backing on it means the new pipeline
   branch gets all of that battle-testing for free.

2. Every ``RenderOp`` carries a fully-resolved normalized
   ``primary_rect`` that we can project back into the pixel-space
   ``subject_x`` / ``subject_y`` fields ``ReframeSegment`` expects.

3. Motion paths, ease-in, speaker slot, and op kind survive the
   round-trip so the per-segment ``strategy`` / ``layout`` /
   ``reason`` fields downstream remain meaningful.
"""

from __future__ import annotations

import logging
from typing import Optional

from backend.services.human_reframe import HumanReframePlan
from backend.services.human_render_plan_adapter import render_plan_from_human_plan
from backend.services.reframe_config import ReframeConfig, get_default_config
from backend.services.reframe_segmenter import ReframeSegment
from backend.services.render_plan import MotionKeypoint, RenderOp, RenderOpKind, RenderPlan

logger = logging.getLogger(__name__)


# ── Op-kind → (strategy, layout) mapping ────────────────────────────


_KIND_STRATEGY_LAYOUT: dict[RenderOpKind, tuple[str, str]] = {
    RenderOpKind.CROP: ("stationary", "single"),
    RenderOpKind.TRACKING_CROP: ("tracking", "single"),
    RenderOpKind.WIDE_MASTER: ("wide", "wide_master"),
    RenderOpKind.BLUR_FILL: ("stationary", "blur_fill"),
    RenderOpKind.SPLIT_SCREEN: ("stationary", "split"),
    RenderOpKind.STACKED_GAMEPLAY: ("stationary", "stacked_gameplay"),
    RenderOpKind.GRID_2X2: ("stationary", "grid"),
    RenderOpKind.MOTIVATED_PUSH_IN: ("push_in", "single"),
    RenderOpKind.MOTIVATED_PULL_OUT: ("pull_out", "single"),
    RenderOpKind.KEN_BURNS: ("tracking", "single"),
    RenderOpKind.OUTPAINT_FILL: ("stationary", "blur_fill"),
}

# Fallback / degraded op kinds default to a conservative confidence
# so the downstream clip-scorer doesn't over-weight them.
_DEGRADED_KINDS = frozenset({
    RenderOpKind.BLUR_FILL,
    RenderOpKind.WIDE_MASTER,
    RenderOpKind.OUTPAINT_FILL,
})


def _strategy_layout_for(kind: RenderOpKind) -> tuple[str, str]:
    return _KIND_STRATEGY_LAYOUT.get(kind, ("stationary", "blur_fill"))


def _rect_center_px(rect, source_w: int, source_h: int) -> tuple[float, float]:
    """Return ``(cx_px, cy_px)`` for a normalized Rect."""
    cx = (float(rect.x) + float(rect.w) * 0.5) * source_w
    cy = (float(rect.y) + float(rect.h) * 0.5) * source_h
    return (cx, cy)


def _motion_path_px(
    motion_path: list[MotionKeypoint],
    seg_start: float,
    source_w: int,
    source_h: int,
) -> Optional[list[tuple[float, float, float]]]:
    """Project a keypoint list to pixel-space ``(t_absolute, x_px, y_px)``.

    ``RenderOp.motion_path`` uses ``t`` relative to the op's
    ``start_sec``; ``ReframeSegment.motion_path`` is consumed by
    downstream code that treats it as an opaque list of
    ``(t, x, y)`` tuples. Returning absolute timestamps matches the
    convention ``build_reframe_segments`` already uses for panning
    segments (see ``reframe_segmenter.py:_tracking_motion_path``).
    """
    if not motion_path:
        return None
    out: list[tuple[float, float, float]] = []
    for kp in motion_path:
        cx_px, cy_px = _rect_center_px(kp.rect, source_w, source_h)
        out.append((seg_start + float(kp.t), cx_px, cy_px))
    return out or None


def _reason_from_op(op: RenderOp) -> str:
    """Extract a short reason string from the op's ``strategy_label``.

    The render-plan adapter emits labels like ``track:ab:speaker-0``,
    ``crop:default-track``, ``zoom:emotional-peak``. We keep the
    colon-separated tail because it carries the scheduler's reason
    (which ``ReframeSegment.reason`` downstream logs as-is).
    """
    label = (op.strategy_label or "").strip()
    if not label:
        return "human_reframe"
    if ":" in label:
        return label.split(":", 1)[1] or label
    return label


def _lead_room_direction_from_op(op: RenderOp) -> Optional[str]:
    """Infer a coarse lead-room direction from the op's motion_path.

    When the motion path walks monotonically left→right or right→left
    we label the segment so the Canvas preview / debug overlay can
    surface it. ``None`` when the path is flat or absent.
    """
    if not op.motion_path or len(op.motion_path) < 2:
        return None
    xs = [float(kp.rect.x) + float(kp.rect.w) * 0.5 for kp in op.motion_path]
    dx = xs[-1] - xs[0]
    if abs(dx) < 0.01:
        return None
    return "right" if dx > 0 else "left"


# ── Public adapter ──────────────────────────────────────────────────


def reframe_segments_from_human_plan(
    human: HumanReframePlan,
    *,
    source_width: int,
    source_height: int,
    source_fps: float,
    content_type: str = "",
    target_aspect: str = "9:16",
    target_height_px: int = 1920,
    config: Optional[ReframeConfig] = None,
) -> tuple[list[ReframeSegment], RenderPlan]:
    """Convert a :class:`HumanReframePlan` into the segment list that
    the rest of ``pipeline.py`` expects.

    Also returns the underlying :class:`RenderPlan` so callers that
    want to skip ``build_render_plan`` (a second conversion through
    the legacy segmenter format) can attach it to the job directly.

    The returned segments are contiguous with half-open ``[start, end)``
    semantics — ``build_reframe_segments`` uses the same convention —
    and the underlying ``RenderPlan`` has been coverage-verified by
    ``render_plan_from_human_plan``'s built-in validator.
    """
    config = config or get_default_config()
    plan = render_plan_from_human_plan(
        human,
        source_width=source_width,
        source_height=source_height,
        source_fps=source_fps,
        target_aspect=target_aspect,
        target_height_px=target_height_px,
        config=config,
        content_type=content_type,
    )
    segments = _plan_ops_to_segments(
        plan.ops,
        source_width=source_width,
        source_height=source_height,
        content_type=content_type,
    )
    return segments, plan


def _plan_ops_to_segments(
    ops: list[RenderOp],
    *,
    source_width: int,
    source_height: int,
    content_type: str,
) -> list[ReframeSegment]:
    out: list[ReframeSegment] = []
    for op in ops:
        strategy, layout = _strategy_layout_for(op.kind)
        cx_px, cy_px = _rect_center_px(
            op.primary_rect, source_width, source_height,
        )
        motion_path = _motion_path_px(
            op.motion_path or [], op.start_sec, source_width, source_height,
        )
        confidence = 0.45 if op.kind in _DEGRADED_KINDS else 0.9
        op_content_type = op.content_type or content_type or "unknown"
        out.append(ReframeSegment(
            start=float(op.start_sec),
            end=float(op.end_sec),
            subject_x=float(cx_px),
            subject_y=float(cy_px),
            layout=layout,
            active_slot=op.speaker_slot,
            confidence=confidence,
            reason=_reason_from_op(op),
            ease_in_ms=int(op.ease_in_ms or 0),
            strategy=strategy,
            content_type=op_content_type,
            lead_room_direction=_lead_room_direction_from_op(op),
            motion_path=motion_path,
            subject_source="human_reframe",
            gaming_layout_mode=op.gaming_layout_mode,
        ))
    return out
