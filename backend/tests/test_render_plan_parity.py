"""Parity tests for the RenderPlan system.

Tests that:
1. RenderPlan invariants hold for all segment configurations.
2. Clip trimming correctly rebases times and sets source_offset_sec.
3. Any valid segmenter output produces a contiguous, valid plan.
4. The FFmpeg filter builder produces parseable filter graphs.
5. Builder + validator round-trip works for all op kinds.
"""

import pytest

from backend.services.render_plan import (
    MotionKeypoint,
    Rect,
    RenderOp,
    RenderOpKind,
    RenderPlan,
)
from backend.services.render_plan_builder import build_render_plan
from backend.services.ffmpeg_filter_builder import (
    _build_filter_graph,
    build_ffmpeg_command,
    cleanup_filter_script,
)


# ── Segment stub ──

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


class TestInvariants:
    """Verify validate() catches violations correctly."""

    def test_valid_plan_passes(self):
        segments = [
            _Seg(0, 5, subject_x=30),
            _Seg(5, 10, subject_x=70, ease_in_ms=500),
            _Seg(10, 15, subject_x=50),
        ]
        plan = build_render_plan(
            segments, source_width=1920, source_height=1080, source_fps=30.0,
        )
        assert plan.validate() == []

    def test_contiguity_invariant(self):
        """Any valid segment list produces a contiguous plan."""
        segments = [
            _Seg(0, 3, strategy="stationary", subject_x=25),
            _Seg(3, 8, strategy="wide_master", layout="wide_master"),
            _Seg(8, 12, strategy="blur_fill", layout="blur_fill"),
            _Seg(12, 20, strategy="stationary", subject_x=75),
        ]
        plan = build_render_plan(
            segments, source_width=1920, source_height=1080, source_fps=30.0,
        )
        violations = plan.validate()
        assert violations == [], f"Unexpected violations: {violations}"
        # Explicit contiguity check
        for i in range(len(plan.ops) - 1):
            gap = abs(plan.ops[i].end_sec - plan.ops[i + 1].start_sec)
            assert gap < 0.001, f"Gap at op {i}: {gap}"

    def test_gap_mutation_detected(self):
        """Mutating a plan to introduce a gap causes validate() to fail."""
        segments = [_Seg(0, 5), _Seg(5, 10)]
        plan = build_render_plan(
            segments, source_width=1920, source_height=1080, source_fps=30.0,
        )
        assert plan.validate() == []

        # Mutate: introduce a gap
        plan.ops[1].start_sec = 6.0
        violations = plan.validate()
        assert any("Gap" in v for v in violations)

    def test_overlap_mutation_detected(self):
        """Overlapping ops detected by contiguity check."""
        plan = RenderPlan(
            source_width=1920, source_height=1080,
            target_width=1080, target_height=1920,
            total_duration_sec=10.0, fps=30.0,
            ops=[
                RenderOp(kind=RenderOpKind.CROP, start_sec=0.0, end_sec=6.0,
                         primary_rect=Rect(0.2, 0.0, 0.5, 1.0)),
                RenderOp(kind=RenderOpKind.CROP, start_sec=5.0, end_sec=10.0,
                         primary_rect=Rect(0.3, 0.0, 0.5, 1.0)),
            ],
        )
        violations = plan.validate()
        assert any("Gap" in v for v in violations)

    def test_out_of_range_rect_mutation(self):
        segments = [_Seg(0, 10)]
        plan = build_render_plan(
            segments, source_width=1920, source_height=1080, source_fps=30.0,
        )
        # Mutate: set rect out of range
        plan.ops[0].primary_rect.x = 2.0
        violations = plan.validate()
        assert any("outside" in v for v in violations)

    def test_missing_secondary_rect_for_split(self):
        """SPLIT_SCREEN without secondary_rect fails validation."""
        plan = RenderPlan(
            source_width=1920, source_height=1080,
            target_width=1080, target_height=1920,
            total_duration_sec=10.0, fps=30.0,
            ops=[RenderOp(
                kind=RenderOpKind.SPLIT_SCREEN,
                start_sec=0.0, end_sec=10.0,
                primary_rect=Rect(0.0, 0.0, 0.5, 1.0),
                secondary_rect=None,
            )],
        )
        violations = plan.validate()
        assert any("secondary_rect" in v for v in violations)

    def test_tracking_crop_without_motion_path_fails(self):
        plan = RenderPlan(
            source_width=1920, source_height=1080,
            target_width=1080, target_height=1920,
            total_duration_sec=10.0, fps=30.0,
            ops=[RenderOp(
                kind=RenderOpKind.TRACKING_CROP,
                start_sec=0.0, end_sec=10.0,
                primary_rect=Rect(0.2, 0.0, 0.5, 1.0),
                motion_path=[],
            )],
        )
        violations = plan.validate()
        assert any("motion_path" in v for v in violations)


class TestClipTrimming:
    """Verify clip_range trimming and rebasing."""

    def test_standard_clip_trim(self):
        """Segments spanning (25-35, 35-55, 55-70) trimmed to clip (30, 60)."""
        segments = [
            _Seg(25, 35, subject_x=30),
            _Seg(35, 55, subject_x=50),
            _Seg(55, 70, subject_x=70),
        ]
        plan = build_render_plan(
            segments, source_width=1920, source_height=1080,
            source_fps=30.0, clip_range=(30.0, 60.0),
        )
        # Timeline starts at 0, ends at 30
        assert plan.ops[0].start_sec == 0.0
        assert abs(plan.total_duration_sec - 30.0) < 0.1
        assert plan.source_offset_sec == 30.0

        # Verify contiguity
        assert plan.validate() == []

        # First segment trimmed from 25-35 to 30-35, rebased to 0-5
        assert abs(plan.ops[0].end_sec - 5.0) < 0.1

    def test_clip_fully_within_single_segment(self):
        segments = [_Seg(0, 100, subject_x=50)]
        plan = build_render_plan(
            segments, source_width=1920, source_height=1080,
            source_fps=30.0, clip_range=(20.0, 40.0),
        )
        assert len(plan.ops) == 1
        assert plan.ops[0].start_sec == 0.0
        assert abs(plan.ops[0].end_sec - 20.0) < 0.1
        assert plan.source_offset_sec == 20.0

    def test_no_overlapping_segments(self):
        """Clip range outside all segments produces fallback center crop."""
        segments = [_Seg(0, 10), _Seg(10, 20)]
        plan = build_render_plan(
            segments, source_width=1920, source_height=1080,
            source_fps=30.0, clip_range=(50.0, 60.0),
        )
        assert len(plan.ops) == 1
        assert plan.ops[0].strategy_label == "fallback_center"


class TestAllOpKindsPipeline:
    """Every op kind: segment → plan → filter graph → no crash."""

    @pytest.mark.parametrize("strategy,layout,extra_kwargs", [
        ("stationary", "single", {}),
        ("wide_master", "wide_master", {}),
        ("blur_fill", "blur_fill", {}),
        ("split_screen", "split", {"subject_x": 30}),
        ("stacked_gameplay", "stacked_gameplay", {}),
        ("grid", "grid", {}),
        ("tracking", "single", {
            "motion_path": [(0, 50, 40), (5, 30, 40), (10, 70, 40)],
        }),
        ("panning", "single", {
            "motion_path": [(0, 30, 40), (10, 70, 40)],
        }),
    ])
    def test_op_kind_full_pipeline(self, strategy, layout, extra_kwargs):
        seg = _Seg(0, 10, strategy=strategy, layout=layout, **extra_kwargs)
        plan = build_render_plan(
            [seg], source_width=1920, source_height=1080, source_fps=30.0,
        )
        assert plan.validate() == [], f"Validation failed for {strategy}"

        # Build FFmpeg filter graph — should not crash
        graph = _build_filter_graph(plan)
        assert "[outv]" in graph, f"Missing [outv] for {strategy}"

    @pytest.mark.parametrize("strategy,layout,extra_kwargs", [
        ("stationary", "single", {}),
        ("wide_master", "wide_master", {}),
        ("blur_fill", "blur_fill", {}),
        ("split_screen", "split", {"subject_x": 30}),
        ("stacked_gameplay", "stacked_gameplay", {}),
        ("grid", "grid", {}),
    ])
    def test_op_kind_json_serializable(self, strategy, layout, extra_kwargs):
        import json
        seg = _Seg(0, 5, strategy=strategy, layout=layout, **extra_kwargs)
        plan = build_render_plan(
            [seg], source_width=1920, source_height=1080, source_fps=30.0,
        )
        json_str = plan.to_json()
        data = json.loads(json_str)
        assert len(data["ops"]) >= 1


class TestMixedStrategyTimeline:
    """Test a realistic timeline with mixed strategies."""

    def test_narrative_timeline(self):
        """Simulate a narrative video: dialogue -> action -> dialogue."""
        segments = [
            _Seg(0, 8, subject_x=30, strategy="stationary", reason="speaker_turn"),
            _Seg(8, 15, strategy="wide_master", layout="wide_master", reason="action_sequence"),
            _Seg(15, 22, subject_x=70, strategy="stationary", reason="speaker_turn", ease_in_ms=500),
            _Seg(22, 30, strategy="blur_fill", layout="blur_fill", reason="wide_fallback"),
        ]
        plan = build_render_plan(
            segments, source_width=1920, source_height=1080, source_fps=30.0,
        )
        assert plan.validate() == []
        assert len(plan.ops) == 4
        assert plan.ops[0].kind == RenderOpKind.CROP
        assert plan.ops[1].kind == RenderOpKind.WIDE_MASTER
        assert plan.ops[2].kind == RenderOpKind.CROP
        assert plan.ops[3].kind == RenderOpKind.BLUR_FILL

    def test_podcast_timeline(self):
        """Simulate a podcast: alternating speakers with split overlap."""
        segments = [
            _Seg(0, 10, subject_x=25, strategy="stationary", reason="speaker_turn"),
            _Seg(10, 18, subject_x=75, strategy="stationary", reason="speaker_turn", ease_in_ms=400),
            _Seg(18, 25, strategy="split_screen", layout="split", reason="split_overlap", subject_x=25),
            _Seg(25, 35, subject_x=25, strategy="stationary", reason="speaker_turn"),
        ]
        plan = build_render_plan(
            segments, source_width=1920, source_height=1080, source_fps=30.0,
        )
        assert plan.validate() == []
        assert plan.ops[2].kind == RenderOpKind.SPLIT_SCREEN

    def test_gaming_timeline(self):
        """Simulate a gaming video: stacked gameplay throughout."""
        segments = [
            _Seg(0, 30, strategy="stacked_gameplay", layout="stacked_gameplay"),
        ]
        plan = build_render_plan(
            segments, source_width=1920, source_height=1080, source_fps=30.0,
        )
        assert plan.validate() == []
        assert plan.ops[0].kind == RenderOpKind.STACKED_GAMEPLAY


class TestFFmpegCommandStructure:
    """Verify the complete FFmpeg command has correct structure."""

    def test_full_video_command(self):
        segments = [
            _Seg(0, 10, subject_x=30),
            _Seg(10, 20, strategy="wide_master", layout="wide_master"),
        ]
        plan = build_render_plan(
            segments, source_width=1920, source_height=1080, source_fps=30.0,
        )
        cmd, script_path = build_ffmpeg_command(
            plan, "/data/video.mp4", "/data/output.mp4", use_gpu=False,
        )
        try:
            assert "-i" in cmd
            assert "/data/video.mp4" in cmd
            assert "-filter_complex_script" in cmd
            assert "-map" in cmd
            assert "[outv]" in cmd
            assert "-y" in cmd
        finally:
            cleanup_filter_script(script_path)

    def test_clip_command_has_seek(self):
        segments = [_Seg(50, 60)]
        plan = build_render_plan(
            segments, source_width=1920, source_height=1080,
            source_fps=30.0, clip_range=(50.0, 60.0),
        )
        cmd, script_path = build_ffmpeg_command(
            plan, "/data/video.mp4", "/data/clip.mp4", use_gpu=False,
        )
        try:
            # -ss before -i for fast seek
            ss_idx = cmd.index("-ss")
            i_idx = cmd.index("-i")
            assert ss_idx < i_idx, "-ss should come before -i for fast seek"
        finally:
            cleanup_filter_script(script_path)


class TestRectToPixels:
    """Test Rect.to_pixels() parity contract."""

    def test_full_frame_rect(self):
        r = Rect(0.0, 0.0, 1.0, 1.0)
        px, py, pw, ph = r.to_pixels(1920, 1080)
        assert pw == 1920
        assert ph == 1080
        assert px == 0
        assert py == 0

    def test_centered_9_16_crop(self):
        # 9:16 at 1080p from 1920x1080
        w_norm = (9/16 * 1080) / 1920  # ≈ 0.3164
        r = Rect(x=0.5 - w_norm/2, y=0.0, w=w_norm, h=1.0)
        px, py, pw, ph = r.to_pixels(1920, 1080)
        assert pw % 2 == 0
        assert ph % 2 == 0
        assert px >= 0
        assert px + pw <= 1920

    def test_even_dimensions_enforced(self):
        r = Rect(0.0, 0.0, 0.333, 0.777)
        px, py, pw, ph = r.to_pixels(1920, 1080)
        assert pw % 2 == 0
        assert ph % 2 == 0

    def test_clamped_to_bounds(self):
        r = Rect(0.9, 0.0, 0.5, 1.0)
        px, py, pw, ph = r.to_pixels(1920, 1080)
        assert px + pw <= 1920
        assert py + ph <= 1080
