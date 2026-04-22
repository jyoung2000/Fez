"""Blueprint v2 Phase 3 — ``splice_segment`` coverage + invariant tests."""

import pytest

from backend.services.render_plan import (
    MotionKeypoint,
    Rect,
    RenderOp,
    RenderOpKind,
    RenderPlan,
)
from backend.services.render_plan_splice import splice_segment


def _rect() -> Rect:
    return Rect(x=0.2, y=0.0, w=0.56, h=1.0)


def _mk_op(
    start: float, end: float,
    kind: RenderOpKind = RenderOpKind.CROP,
) -> RenderOp:
    return RenderOp(
        kind=kind,
        start_sec=start,
        end_sec=end,
        primary_rect=_rect(),
    )


def _mk_plan(ops: list, total: float, fps: float = 30.0) -> RenderPlan:
    return RenderPlan(
        source_width=1920, source_height=1080,
        target_width=1080, target_height=1920,
        total_duration_sec=total,
        fps=fps,
        ops=ops,
    )


def _has_gap_or_overlap(plan: RenderPlan) -> tuple[bool, list]:
    """Cheap local coverage verifier — avoids pulling numpy via
    ``human_render_plan_adapter.verify_frame_coverage``."""
    violations = plan.validate()
    return (len(violations) > 0, violations)


# ── Core splice paths ────────────────────────────────────────────


def test_splice_middle_replaces_window_and_preserves_coverage():
    """Splice [3, 6] into a 0-10 plan that had a single op."""
    original = _mk_plan([_mk_op(0.0, 10.0)], total=10.0)
    seg_op = _mk_op(0.0, 3.0, RenderOpKind.TRACKING_CROP)
    seg_op.motion_path = [
        MotionKeypoint(t=0.0, rect=_rect()),
        MotionKeypoint(t=3.0, rect=_rect()),
    ]
    segment = _mk_plan([seg_op], total=3.0)

    spliced = splice_segment(original, segment, start=3.0, end=6.0)
    assert spliced is not None
    assert spliced.total_duration_sec == pytest.approx(10.0)
    bad, violations = _has_gap_or_overlap(spliced)
    assert not bad, f"splice produced violations: {violations}"
    # The spliced plan's middle op is the tracking op.
    kinds = [op.kind for op in spliced.ops]
    assert RenderOpKind.TRACKING_CROP in kinds


def test_splice_leading_edge_preserves_tail():
    """Splice [0, 2] into a 0-10 plan that had two ops."""
    original = _mk_plan(
        [_mk_op(0.0, 5.0), _mk_op(5.0, 10.0)], total=10.0,
    )
    segment = _mk_plan([_mk_op(0.0, 2.0)], total=2.0)
    spliced = splice_segment(original, segment, start=0.0, end=2.0)
    bad, violations = _has_gap_or_overlap(spliced)
    assert not bad, violations
    # The tail op from the original is preserved.
    assert spliced.ops[-1].end_sec == pytest.approx(10.0)


def test_splice_trailing_edge_preserves_head():
    original = _mk_plan(
        [_mk_op(0.0, 5.0), _mk_op(5.0, 10.0)], total=10.0,
    )
    segment = _mk_plan([_mk_op(0.0, 2.0)], total=2.0)
    spliced = splice_segment(original, segment, start=8.0, end=10.0)
    bad, violations = _has_gap_or_overlap(spliced)
    assert not bad, violations
    # The head op is preserved intact.
    assert spliced.ops[0].start_sec == 0.0


def test_splice_zero_width_window_returns_original():
    original = _mk_plan([_mk_op(0.0, 10.0)], total=10.0)
    segment = _mk_plan([_mk_op(0.0, 1.0)], total=1.0)
    spliced = splice_segment(original, segment, start=5.0, end=5.0)
    assert spliced is original


def test_splice_empty_segment_returns_original():
    original = _mk_plan([_mk_op(0.0, 10.0)], total=10.0)
    empty = _mk_plan([], total=3.0)
    spliced = splice_segment(original, empty, start=3.0, end=6.0)
    # An empty segment would leave a gap — helper must revert to original.
    assert spliced is original


def test_splice_preserves_op_count_growth_bounded():
    """A splice should not explode the op count unboundedly."""
    original = _mk_plan([_mk_op(0.0, 10.0)], total=10.0)
    segment_ops = [_mk_op(float(i), float(i + 1)) for i in range(3)]
    segment = _mk_plan(segment_ops, total=3.0)
    spliced = splice_segment(original, segment, start=3.0, end=6.0)
    # 1 prefix op [0, 3] + 3 segment ops [3, 6] + 1 suffix op [6, 10] = 5.
    assert len(spliced.ops) == 5


def test_splice_scales_segment_duration_to_window():
    """Segment with inner duration != window duration gets scaled so
    start/end land on the window boundary."""
    original = _mk_plan([_mk_op(0.0, 10.0)], total=10.0)
    # Segment's inner timeline is 0-5, but we splice into a 2s window.
    segment = _mk_plan([_mk_op(0.0, 5.0)], total=5.0)
    spliced = splice_segment(original, segment, start=3.0, end=5.0)
    bad, violations = _has_gap_or_overlap(spliced)
    assert not bad, violations
    # The window boundary is exact.
    mid = [op for op in spliced.ops if op.start_sec == pytest.approx(3.0)]
    assert mid, "expected a spliced op starting at 3.0"
    assert mid[0].end_sec == pytest.approx(5.0)
