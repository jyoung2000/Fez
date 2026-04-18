"""Tests for the RenderPlan builder.

Covers segment-to-op mapping, clip trimming, contiguity enforcement,
and validation.
"""

import pytest

from backend.services.render_plan import (
    Rect,
    RenderOp,
    RenderOpKind,
    RenderPlan,
)
from backend.services.render_plan_builder import build_render_plan


# ── Lightweight segment stub ──

class _Seg:
    def __init__(self, start, end, subject_x=50, subject_y=40,
                 layout="single", strategy="stationary", reason="hold",
                 ease_in_ms=0, content_type="unknown", motion_path=None,
                 hard_constraints=None, active_slot=None, confidence=1.0,
                 lead_room_direction=None):
        self.start = start
        self.end = end
        self.subject_x = subject_x
        self.subject_y = subject_y
        self.layout = layout
        self.strategy = strategy
        self.reason = reason
        self.ease_in_ms = ease_in_ms
        self.content_type = content_type
        self.motion_path = motion_path
        self.hard_constraints = hard_constraints
        self.active_slot = active_slot
        self.confidence = confidence
        self.lead_room_direction = lead_room_direction


class TestBuildRenderPlan:
    """Basic build_render_plan tests."""

    def test_single_stationary_segment(self):
        segments = [_Seg(0, 10, subject_x=50)]
        plan = build_render_plan(
            segments, source_width=1920, source_height=1080,
            source_fps=30.0, target_aspect="9:16",
        )
        assert len(plan.ops) == 1
        assert plan.ops[0].kind == RenderOpKind.CROP
        assert plan.ops[0].start_sec == 0.0
        assert plan.ops[0].end_sec == 10.0
        assert plan.total_duration_sec == 10.0
        assert plan.target_width == 1080
        assert plan.target_height == 1920

    def test_multiple_segments_contiguous(self):
        segments = [
            _Seg(0, 5, subject_x=30),
            _Seg(5, 10, subject_x=70),
            _Seg(10, 15, subject_x=50),
        ]
        plan = build_render_plan(
            segments, source_width=1920, source_height=1080,
            source_fps=30.0,
        )
        assert len(plan.ops) == 3
        # Check contiguity
        for i in range(len(plan.ops) - 1):
            assert abs(plan.ops[i].end_sec - plan.ops[i + 1].start_sec) < 0.001

    def test_wide_master_segment(self):
        segments = [_Seg(0, 10, layout="wide_master", strategy="wide_master")]
        plan = build_render_plan(
            segments, source_width=1920, source_height=1080,
            source_fps=30.0,
        )
        assert plan.ops[0].kind == RenderOpKind.WIDE_MASTER
        assert plan.ops[0].primary_rect == Rect(x=0.0, y=0.0, w=1.0, h=1.0)

    def test_blur_fill_segment(self):
        segments = [_Seg(0, 10, layout="blur_fill", strategy="blur_fill")]
        plan = build_render_plan(
            segments, source_width=1920, source_height=1080,
            source_fps=30.0,
        )
        assert plan.ops[0].kind == RenderOpKind.BLUR_FILL

    def test_split_screen_segment(self):
        segments = [_Seg(0, 10, layout="split", strategy="split_screen", subject_x=30)]
        plan = build_render_plan(
            segments, source_width=1920, source_height=1080,
            source_fps=30.0,
        )
        assert plan.ops[0].kind == RenderOpKind.SPLIT_SCREEN
        assert plan.ops[0].secondary_rect is not None

    def test_grid_2x2_segment(self):
        segments = [_Seg(0, 10, layout="grid", strategy="grid")]
        plan = build_render_plan(
            segments, source_width=1920, source_height=1080,
            source_fps=30.0,
        )
        assert plan.ops[0].kind == RenderOpKind.GRID_2X2
        assert plan.ops[0].secondary_rect is not None
        assert plan.ops[0].tertiary_rect is not None
        assert plan.ops[0].quaternary_rect is not None

    def test_stacked_gameplay_segment(self):
        segments = [_Seg(0, 10, layout="stacked_gameplay", strategy="stacked_gameplay")]
        plan = build_render_plan(
            segments, source_width=1920, source_height=1080,
            source_fps=30.0,
        )
        assert plan.ops[0].kind == RenderOpKind.STACKED_GAMEPLAY
        assert plan.ops[0].secondary_rect is not None

    def test_ease_in_preserved(self):
        segments = [
            _Seg(0, 5, ease_in_ms=0),
            _Seg(5, 10, ease_in_ms=500),
        ]
        plan = build_render_plan(
            segments, source_width=1920, source_height=1080,
            source_fps=30.0,
        )
        assert plan.ops[0].ease_in_ms == 0
        assert plan.ops[1].ease_in_ms == 500

    def test_target_aspect_1_1(self):
        segments = [_Seg(0, 10)]
        plan = build_render_plan(
            segments, source_width=1920, source_height=1080,
            source_fps=30.0, target_aspect="1:1",
        )
        assert plan.target_width == plan.target_height

    def test_empty_segments_fallback(self):
        plan = build_render_plan(
            segments=[], source_width=1920, source_height=1080,
            source_fps=30.0, clip_range=(0, 10),
        )
        assert len(plan.ops) == 1
        assert plan.ops[0].kind == RenderOpKind.CROP


class TestClipTrimming:
    """Tests for clip_range trimming and rebasing."""

    def test_clip_range_trims_and_rebases(self):
        segments = [
            _Seg(25, 35, subject_x=30),
            _Seg(35, 55, subject_x=50),
            _Seg(55, 70, subject_x=70),
        ]
        plan = build_render_plan(
            segments, source_width=1920, source_height=1080,
            source_fps=30.0, clip_range=(30.0, 60.0),
        )
        # Timeline should be 30s long, starting at 0
        assert abs(plan.total_duration_sec - 30.0) < 0.1
        assert plan.ops[0].start_sec == 0.0
        assert plan.source_offset_sec == 30.0

        # First segment (25-35) should be trimmed to (30-35) -> rebased to (0-5)
        assert abs(plan.ops[0].end_sec - 5.0) < 0.1

    def test_clip_range_excludes_non_overlapping(self):
        segments = [
            _Seg(0, 10, subject_x=30),
            _Seg(10, 20, subject_x=50),
            _Seg(20, 30, subject_x=70),
        ]
        plan = build_render_plan(
            segments, source_width=1920, source_height=1080,
            source_fps=30.0, clip_range=(10.0, 20.0),
        )
        # Only the middle segment overlaps
        assert len(plan.ops) == 1
        assert plan.ops[0].start_sec == 0.0
        assert abs(plan.ops[0].end_sec - 10.0) < 0.1

    def test_clip_range_with_source_offset(self):
        segments = [_Seg(100, 110)]
        plan = build_render_plan(
            segments, source_width=1920, source_height=1080,
            source_fps=30.0, clip_range=(100, 110),
        )
        assert plan.source_offset_sec == 100.0
        assert plan.ops[0].start_sec == 0.0


class TestContiguity:
    """Tests for contiguity enforcement."""

    def test_small_gap_snapped(self):
        segments = [
            _Seg(0, 5),
            _Seg(5.02, 10),  # 20ms gap
        ]
        plan = build_render_plan(
            segments, source_width=1920, source_height=1080,
            source_fps=30.0,
        )
        # Gap should be snapped
        assert abs(plan.ops[0].end_sec - plan.ops[1].start_sec) < 0.001

    def test_large_gap_bridged(self):
        segments = [
            _Seg(0, 5),
            _Seg(6, 10),  # 1s gap
        ]
        plan = build_render_plan(
            segments, source_width=1920, source_height=1080,
            source_fps=30.0,
        )
        # Should have 3 ops: original, bridge, original
        assert len(plan.ops) == 3
        assert plan.ops[1].strategy_label == "bridge_hold"

    def test_overlap_snapped(self):
        segments = [
            _Seg(0, 5.5),
            _Seg(5.0, 10),  # 0.5s overlap
        ]
        plan = build_render_plan(
            segments, source_width=1920, source_height=1080,
            source_fps=30.0,
        )
        assert abs(plan.ops[0].end_sec - plan.ops[1].start_sec) < 0.001


class TestValidation:
    """Tests for RenderPlan.validate()."""

    def test_valid_plan_passes(self):
        plan = RenderPlan(
            source_width=1920, source_height=1080,
            target_width=1080, target_height=1920,
            total_duration_sec=10.0, fps=30.0,
            ops=[RenderOp(
                kind=RenderOpKind.CROP,
                start_sec=0.0, end_sec=10.0,
                primary_rect=Rect(0.2, 0.0, 0.5, 1.0),
            )],
        )
        assert plan.validate() == []

    def test_empty_plan_fails(self):
        plan = RenderPlan(
            source_width=1920, source_height=1080,
            target_width=1080, target_height=1920,
            total_duration_sec=10.0, fps=30.0, ops=[],
        )
        violations = plan.validate()
        assert len(violations) > 0
        assert "no ops" in violations[0]

    def test_gap_detected(self):
        plan = RenderPlan(
            source_width=1920, source_height=1080,
            target_width=1080, target_height=1920,
            total_duration_sec=10.0, fps=30.0,
            ops=[
                RenderOp(kind=RenderOpKind.CROP, start_sec=0.0, end_sec=4.0,
                         primary_rect=Rect(0.2, 0.0, 0.5, 1.0)),
                RenderOp(kind=RenderOpKind.CROP, start_sec=6.0, end_sec=10.0,
                         primary_rect=Rect(0.2, 0.0, 0.5, 1.0)),
            ],
        )
        violations = plan.validate()
        assert any("Gap" in v for v in violations)

    def test_out_of_range_rect_detected(self):
        plan = RenderPlan(
            source_width=1920, source_height=1080,
            target_width=1080, target_height=1920,
            total_duration_sec=10.0, fps=30.0,
            ops=[RenderOp(
                kind=RenderOpKind.CROP,
                start_sec=0.0, end_sec=10.0,
                primary_rect=Rect(1.5, 0.0, 0.5, 1.0),  # x=1.5 out of range
            )],
        )
        violations = plan.validate()
        assert any("outside" in v for v in violations)

    def test_split_screen_requires_secondary(self):
        plan = RenderPlan(
            source_width=1920, source_height=1080,
            target_width=1080, target_height=1920,
            total_duration_sec=10.0, fps=30.0,
            ops=[RenderOp(
                kind=RenderOpKind.SPLIT_SCREEN,
                start_sec=0.0, end_sec=10.0,
                primary_rect=Rect(0.0, 0.0, 0.5, 1.0),
                # Missing secondary_rect
            )],
        )
        violations = plan.validate()
        assert any("secondary_rect" in v for v in violations)

    def test_first_op_must_start_at_zero(self):
        plan = RenderPlan(
            source_width=1920, source_height=1080,
            target_width=1080, target_height=1920,
            total_duration_sec=10.0, fps=30.0,
            ops=[RenderOp(
                kind=RenderOpKind.CROP,
                start_sec=1.0, end_sec=10.0,
                primary_rect=Rect(0.2, 0.0, 0.5, 1.0),
            )],
        )
        violations = plan.validate()
        assert any("0.0" in v for v in violations)

    def test_last_op_must_end_at_duration(self):
        plan = RenderPlan(
            source_width=1920, source_height=1080,
            target_width=1080, target_height=1920,
            total_duration_sec=10.0, fps=30.0,
            ops=[RenderOp(
                kind=RenderOpKind.CROP,
                start_sec=0.0, end_sec=8.0,
                primary_rect=Rect(0.2, 0.0, 0.5, 1.0),
            )],
        )
        violations = plan.validate()
        assert any("10.0" in v or "total_duration" in v for v in violations)


class TestSerialization:
    """Tests for RenderPlan JSON serialization."""

    def test_to_json_round_trip(self):
        import json
        plan = RenderPlan(
            source_width=1920, source_height=1080,
            target_width=1080, target_height=1920,
            total_duration_sec=10.0, fps=30.0,
            ops=[RenderOp(
                kind=RenderOpKind.CROP,
                start_sec=0.0, end_sec=10.0,
                primary_rect=Rect(0.2, 0.0, 0.5, 1.0),
                strategy_label="test",
            )],
        )
        json_str = plan.to_json()
        data = json.loads(json_str)
        assert data["source_width"] == 1920
        assert data["ops"][0]["kind"] == "crop"
        assert data["ops"][0]["primary_rect"]["x"] == 0.2

    def test_to_dict(self):
        plan = RenderPlan(
            source_width=1920, source_height=1080,
            target_width=1080, target_height=1920,
            total_duration_sec=5.0, fps=30.0,
            ops=[RenderOp(
                kind=RenderOpKind.WIDE_MASTER,
                start_sec=0.0, end_sec=5.0,
                primary_rect=Rect(0.0, 0.0, 1.0, 1.0),
            )],
        )
        d = plan.to_dict()
        assert isinstance(d, dict)
        assert d["target_width"] == 1080
