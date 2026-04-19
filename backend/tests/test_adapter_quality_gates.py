"""Section 4: adapter quality gates.

  * 4.2 — widen on y_uncertain_windows (emit WIDE_MASTER instead of
    a tight crop when the 2-D path flagged a long no-face window).
  * 4.3 — _drop_tiny_segments respects target_slot continuity (a
    0.25s speaker-change is preserved, not absorbed).
"""
from backend.services.camera_path_2d import CameraPath2D
from backend.services.human_reframe import HumanReframePlan
from backend.services.human_render_plan_adapter import (
    _drop_tiny_segments,
    _op_tracking_or_crop,
    _Segment,
)
from backend.services.reframe_config import get_default_config
from backend.services.render_plan import RenderOpKind


class _StubPlan:
    """Minimal HumanReframePlan stand-in for _op_tracking_or_crop."""
    def __init__(self, path):
        self.path = path
        self.events = []
        self.zooms = []
        self.ab = None
        self.genre = None
        self.kalman = None
        self.notes = []
        self.shot_profiles = []


def _make_path(uncertain_windows):
    return CameraPath2D(
        timestamps=[0.0, 0.5, 1.0, 1.5, 2.0],
        cx=[0.5, 0.5, 0.5, 0.5, 0.5],
        cy=[0.5, 0.5, 0.5, 0.5, 0.5],
        crop_w_frac=0.56,
        crop_h_frac=1.0,
        y_uncertain_windows=uncertain_windows,
    )


class TestWidenOnUncertain:
    def test_tracking_segment_overlapping_uncertain_window_becomes_wide(self):
        path = _make_path(uncertain_windows=[(0.2, 1.8)])
        human = _StubPlan(path)
        seg = _Segment(
            start=0.5, end=1.5, kind=RenderOpKind.TRACKING_CROP,
            target_slot=0, reason="test",
        )
        op = _op_tracking_or_crop(
            seg, human, 0.56, 1.0, "talking_head", get_default_config(),
        )
        assert op.kind == RenderOpKind.WIDE_MASTER
        assert "uncertain" in (op.strategy_label or "")

    def test_tracking_segment_outside_uncertain_is_unaffected(self):
        path = _make_path(uncertain_windows=[(3.0, 4.0)])
        human = _StubPlan(path)
        seg = _Segment(
            start=0.5, end=1.5, kind=RenderOpKind.TRACKING_CROP,
            target_slot=0, reason="test",
        )
        op = _op_tracking_or_crop(
            seg, human, 0.56, 1.0, "talking_head", get_default_config(),
        )
        assert op.kind != RenderOpKind.WIDE_MASTER

    def test_no_uncertain_windows_does_not_widen(self):
        path = _make_path(uncertain_windows=[])
        human = _StubPlan(path)
        seg = _Segment(
            start=0.5, end=1.5, kind=RenderOpKind.TRACKING_CROP,
            target_slot=0, reason="test",
        )
        op = _op_tracking_or_crop(
            seg, human, 0.56, 1.0, "talking_head", get_default_config(),
        )
        assert op.kind != RenderOpKind.WIDE_MASTER


class TestTinySegmentSlotContinuity:
    def test_tiny_same_slot_absorbed(self):
        segs = [
            _Segment(start=0.0, end=1.0, kind=RenderOpKind.CROP, target_slot=0),
            _Segment(start=1.0, end=1.02, kind=RenderOpKind.CROP, target_slot=0),
            _Segment(start=1.02, end=2.0, kind=RenderOpKind.CROP, target_slot=0),
        ]
        out = _drop_tiny_segments(segs)
        # Same slot → tiny absorbed into previous.
        assert len(out) == 2
        assert out[0].end == 1.02  # absorbed

    def test_tiny_different_slot_preserved(self):
        """A tiny mid-segment cameo with a unique slot (different from
        both neighbors) must survive — it's an intentional speaker
        cameo that the pre-3.4 absorb-all logic silently dropped."""
        segs = [
            _Segment(start=0.0, end=1.0, kind=RenderOpKind.CROP, target_slot=0),
            _Segment(start=1.0, end=1.02, kind=RenderOpKind.CROP, target_slot=1),
            _Segment(start=1.02, end=2.0, kind=RenderOpKind.CROP, target_slot=0),
        ]
        out = _drop_tiny_segments(segs)
        # All three preserved: speaker 0 → cameo slot 1 → speaker 0.
        assert len(out) == 3
        assert out[1].target_slot == 1

    def test_tiny_none_slot_with_none_is_absorbed(self):
        """Both slots None → safe to absorb (no speaker distinction)."""
        segs = [
            _Segment(start=0.0, end=1.0, kind=RenderOpKind.CROP, target_slot=None),
            _Segment(start=1.0, end=1.02, kind=RenderOpKind.CROP, target_slot=None),
        ]
        out = _drop_tiny_segments(segs)
        assert len(out) == 1
