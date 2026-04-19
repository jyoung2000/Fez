"""Fix 3.8: local reframe critic detects chin-clips / head-clips /
off-center / jitter windows and auto-repairs them.
"""
from dataclasses import dataclass, field

from backend.services.reframe_config import get_default_config
from backend.services.reframe_critic import (
    apply_window_fix,
    attempt_window_fix,
    auto_repair_plan,
    score_plan,
)
from backend.services.render_plan import (
    MotionKeypoint,
    Rect,
    RenderOp,
    RenderOpKind,
    RenderPlan,
)


@dataclass
class _Face:
    nose_x: float
    nose_y: float = 50.0
    width: float = 10.0
    height: float = 12.0
    identity_id: int = 0
    is_human: bool = True


@dataclass
class _FrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)


def _plan_with_op(op: RenderOp, duration: float = 2.0) -> RenderPlan:
    return RenderPlan(
        source_width=1920, source_height=1080,
        target_width=1080, target_height=1920,
        total_duration_sec=duration, fps=30.0,
        ops=[op],
    )


class TestChinClipDetection:
    def test_detects_chin_clip(self):
        """Face bottom extends below crop bottom → score = 0 on the window."""
        # Face at y=0.85, h=0.20 → face_bot = 0.95.
        # Crop y=0.0, h=0.9 → crop_bot = 0.9. face_bot (0.95) > 0.9 - 0.02.
        dense = [
            _FrameFaces(timestamp=t, faces=[
                _Face(nose_x=50.0, nose_y=85.0, width=10.0, height=20.0),
            ])
            for t in [0.0, 0.25, 0.5, 0.75, 1.0]
        ]
        plan = _plan_with_op(RenderOp(
            kind=RenderOpKind.CROP,
            start_sec=0.0, end_sec=1.0,
            primary_rect=Rect(x=0.22, y=0.0, w=0.56, h=0.90),
            strategy_label="test",
        ), duration=1.0)
        scores = score_plan(plan, dense_faces=dense)
        assert scores
        assert scores[0].score == 0
        assert "chin_clip" in scores[0].reasons

    def test_repairs_chin_clip(self):
        dense = [
            _FrameFaces(timestamp=t, faces=[
                _Face(nose_x=50.0, nose_y=85.0, width=10.0, height=20.0),
            ])
            for t in [0.0, 0.25, 0.5, 0.75, 1.0]
        ]
        plan = _plan_with_op(RenderOp(
            kind=RenderOpKind.CROP,
            start_sec=0.0, end_sec=1.0,
            primary_rect=Rect(x=0.22, y=0.0, w=0.56, h=0.90),
            strategy_label="test",
        ), duration=1.0)
        config = get_default_config()
        scores = score_plan(plan, dense_faces=dense, config=config)
        fix = attempt_window_fix(
            scores[0], plan, dense_faces=dense, config=config,
        )
        assert fix is not None
        assert "chin_clip" in fix.reason
        repaired = apply_window_fix(plan, fix)
        new_scores = score_plan(repaired, dense_faces=dense, config=config)
        # After shift, chin-clip should be gone on the window.
        in_window = [s for s in new_scores if s.t_start < 1.0]
        assert in_window
        for s in in_window:
            assert "chin_clip" not in s.reasons, s


class TestJitterDetection:
    """Fixture puts the face at good headroom (face_top ≈ 0.18,
    crop_top=0.10, crop_h=0.80 → face_top_in_crop ≈ 0.10) so the
    only trigger should be jitter."""

    def _make_jittery_plan_and_faces(self):
        path = []
        for i, t in enumerate([i * 0.1 for i in range(11)]):
            cx = 0.5 + (0.08 if i % 2 == 0 else -0.08)
            path.append(MotionKeypoint(
                t=t, rect=Rect(x=cx - 0.28, y=0.10, w=0.56, h=0.80),
            ))
        op = RenderOp(
            kind=RenderOpKind.TRACKING_CROP,
            start_sec=0.0, end_sec=1.0,
            primary_rect=path[0].rect,
            motion_path=path,
            strategy_label="test",
        )
        plan = _plan_with_op(op, duration=1.0)
        dense = [
            _FrameFaces(timestamp=t,
                        faces=[_Face(nose_x=50.0, nose_y=24.0, height=12.0)])
            for t in [i * 0.1 for i in range(11)]
        ]
        return plan, dense

    def test_detects_jittery_motion_path(self):
        plan, dense = self._make_jittery_plan_and_faces()
        scores = score_plan(plan, dense_faces=dense)
        assert scores
        # At least one window flags jitter.
        assert any("jitter_x" in s.reasons for s in scores)

    def test_auto_repair_reduces_jitter(self):
        plan, dense = self._make_jittery_plan_and_faces()
        # Jitter alone only deducts up to 1.0 from a 10 score — it sits
        # above the default 6.0 threshold. Use a tighter threshold so
        # the test exercises the repair path on a jitter-only window.
        config = get_default_config().override(critic_threshold=9.5)
        repaired, fixes, final_scores = auto_repair_plan(
            plan, dense_faces=dense, config=config,
        )
        assert any("jitter" in f.reason for f in fixes)
        window_scores = [s for s in final_scores if s.t_start < 1.0]
        assert window_scores
        for s in window_scores:
            assert "jitter_x" not in s.reasons


class TestSubjectMissingFallback:
    def test_subject_missing_falls_back_to_wide(self):
        plan = _plan_with_op(RenderOp(
            kind=RenderOpKind.CROP,
            start_sec=0.0, end_sec=2.0,
            primary_rect=Rect(x=0.22, y=0.1, w=0.56, h=0.80),
            strategy_label="test",
        ), duration=2.0)
        # No dense faces at all → subject_missing.
        dense = []
        repaired, fixes, final = auto_repair_plan(plan, dense_faces=dense)
        assert any(f.reason == "subject_missing_wide" for f in fixes)
        # The repaired op is WIDE_MASTER.
        covering = [op for op in repaired.ops
                    if op.start_sec < 1.0 and op.end_sec > 0.0]
        assert any(op.kind == RenderOpKind.WIDE_MASTER for op in covering)


class TestOscillationGuard:
    def test_window_not_fixed_twice(self):
        """Budget-respecting: a window already fixed in this pass
        doesn't get fixed again even if still low-score."""
        dense = [
            _FrameFaces(timestamp=t, faces=[
                _Face(nose_x=50.0, nose_y=85.0, width=10.0, height=20.0),
            ])
            for t in [0.0, 0.25, 0.5, 0.75, 1.0]
        ]
        plan = _plan_with_op(RenderOp(
            kind=RenderOpKind.CROP,
            start_sec=0.0, end_sec=1.0,
            primary_rect=Rect(x=0.22, y=0.0, w=0.56, h=0.90),
            strategy_label="test",
        ), duration=1.0)
        repaired, fixes, _ = auto_repair_plan(plan, dense_faces=dense)
        # At most one fix per window.
        keys = set((round(f.t_start, 3), round(f.t_end, 3)) for f in fixes)
        assert len(keys) == len(fixes)
