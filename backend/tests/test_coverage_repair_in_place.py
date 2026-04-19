"""Fix 3.7: coverage is repaired in place, never silently falls back
to the legacy plan on a fixable issue.
"""
from backend.services.human_render_plan_adapter import (
    fill_gaps_with_blur,
    try_repair_coverage,
    verify_frame_coverage,
)
from backend.services.render_plan import (
    MotionKeypoint,
    Rect,
    RenderOp,
    RenderOpKind,
    RenderPlan,
)


def _op(start, end, x=0.2, y=0.0, w=0.56, h=1.0,
        kind=RenderOpKind.CROP, label="test"):
    return RenderOp(
        kind=kind,
        start_sec=start, end_sec=end,
        primary_rect=Rect(x=x, y=y, w=w, h=h),
        strategy_label=label,
    )


def test_fills_leading_gap():
    """Plan starts at t=0.3 — leading gap should be repaired."""
    plan = RenderPlan(
        source_width=1920, source_height=1080,
        target_width=1080, target_height=1920,
        total_duration_sec=5.0, fps=30.0,
        ops=[_op(0.3, 5.0)],
    )
    report = verify_frame_coverage(plan)
    assert not report.ok
    repaired = try_repair_coverage(plan, report, duration_sec=5.0)
    assert repaired is not None
    new_report = verify_frame_coverage(repaired)
    assert new_report.ok
    assert repaired.ops[0].start_sec == 0.0


def test_fills_mid_gap():
    """Plan has a 0.3s gap between two ops — repaired by extending
    neighbors to meet in the middle."""
    plan = RenderPlan(
        source_width=1920, source_height=1080,
        target_width=1080, target_height=1920,
        total_duration_sec=5.0, fps=30.0,
        ops=[_op(0.0, 2.0), _op(2.3, 5.0)],
    )
    report = verify_frame_coverage(plan)
    assert not report.ok
    repaired = try_repair_coverage(plan, report, duration_sec=5.0)
    new_report = verify_frame_coverage(repaired)
    assert new_report.ok


def test_absorbs_zero_duration_op():
    plan = RenderPlan(
        source_width=1920, source_height=1080,
        target_width=1080, target_height=1920,
        total_duration_sec=5.0, fps=30.0,
        ops=[_op(0.0, 2.0), _op(2.0, 2.0), _op(2.0, 5.0)],
    )
    report = verify_frame_coverage(plan)
    assert not report.ok
    repaired = try_repair_coverage(plan, report, duration_sec=5.0)
    new_report = verify_frame_coverage(repaired)
    assert new_report.ok
    # Zero-duration op absorbed.
    assert len(repaired.ops) == 2


def test_trims_overlap():
    plan = RenderPlan(
        source_width=1920, source_height=1080,
        target_width=1080, target_height=1920,
        total_duration_sec=5.0, fps=30.0,
        ops=[_op(0.0, 3.0), _op(2.5, 5.0)],
    )
    report = verify_frame_coverage(plan)
    assert not report.ok
    repaired = try_repair_coverage(plan, report, duration_sec=5.0)
    new_report = verify_frame_coverage(repaired)
    assert new_report.ok
    # Op 0 trimmed to the later op's start.
    assert abs(repaired.ops[0].end_sec - 2.5) < 1e-6


def test_clamps_out_of_range_rect():
    plan = RenderPlan(
        source_width=1920, source_height=1080,
        target_width=1080, target_height=1920,
        total_duration_sec=5.0, fps=30.0,
        ops=[_op(0.0, 5.0, x=1.02, w=0.56)],  # x=1.02 → out of range
    )
    report = verify_frame_coverage(plan)
    assert not report.ok
    repaired = try_repair_coverage(plan, report, duration_sec=5.0)
    new_report = verify_frame_coverage(repaired)
    assert new_report.ok
    # Strategy label annotated with '|clamped'.
    assert "|clamped" in (repaired.ops[0].strategy_label or "")


def test_spec_acceptance_combined():
    """Spec acceptance: a plan with a 0.3s gap and one 2% OOR rect
    is repaired in place without reverting to legacy."""
    plan = RenderPlan(
        source_width=1920, source_height=1080,
        target_width=1080, target_height=1920,
        total_duration_sec=5.0, fps=30.0,
        ops=[
            _op(0.0, 2.0),
            _op(2.3, 5.0, x=1.02),  # both gap AND OOR
        ],
    )
    report = verify_frame_coverage(plan)
    assert not report.ok
    assert report.gaps
    assert report.out_of_range_rects

    repaired = try_repair_coverage(plan, report, duration_sec=5.0)
    assert repaired is not None
    new_report = verify_frame_coverage(repaired)
    assert new_report.ok, f"repair failed: {new_report}"


def test_fill_gaps_with_blur_last_resort():
    """fill_gaps_with_blur splices BLUR_FILL ops into any remaining
    gap so coverage is always complete."""
    # A plan with a stubborn gap that try_repair_coverage might skip.
    plan = RenderPlan(
        source_width=1920, source_height=1080,
        target_width=1080, target_height=1920,
        total_duration_sec=5.0, fps=30.0,
        ops=[_op(0.0, 2.0), _op(3.5, 5.0)],
    )
    filled = fill_gaps_with_blur(plan, duration_sec=5.0)
    report = verify_frame_coverage(filled)
    assert report.ok
    # A blur_fill op was added.
    assert any(
        op.kind == RenderOpKind.BLUR_FILL
        and "coverage-repair" in (op.strategy_label or "")
        for op in filled.ops
    )
