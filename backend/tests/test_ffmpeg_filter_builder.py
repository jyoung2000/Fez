"""Tests for the FFmpeg filter builder.

Covers filter generation for each RenderOpKind, transition handling,
filter script writing, and command structure.
"""

import os
import pytest

from backend.services.render_plan import (
    MotionKeypoint,
    Rect,
    RenderOp,
    RenderOpKind,
    RenderPlan,
)
from backend.services.ffmpeg_filter_builder import (
    BLUR_BRIGHTNESS,
    BLUR_SIGMA,
    build_ffmpeg_command,
    cleanup_filter_script,
    _build_filter_graph,
)


def _make_plan(ops, duration=None):
    """Helper to create a RenderPlan with defaults."""
    if duration is None:
        duration = ops[-1].end_sec if ops else 0.0
    return RenderPlan(
        source_width=1920, source_height=1080,
        target_width=1080, target_height=1920,
        total_duration_sec=duration,
        fps=30.0, ops=ops,
    )


class TestCropFilter:
    def test_single_crop_generates_valid_filter(self):
        plan = _make_plan([RenderOp(
            kind=RenderOpKind.CROP,
            start_sec=0.0, end_sec=10.0,
            primary_rect=Rect(0.2, 0.0, 0.5625, 1.0),
        )])
        graph = _build_filter_graph(plan)
        assert "crop=" in graph
        assert "scale=1080:1920" in graph
        assert "[outv]" in graph
        assert "trim=start=0.000:end=10.000" in graph

    def test_crop_dimensions_are_even(self):
        plan = _make_plan([RenderOp(
            kind=RenderOpKind.CROP,
            start_sec=0.0, end_sec=5.0,
            primary_rect=Rect(0.1, 0.0, 0.5, 1.0),
        )])
        graph = _build_filter_graph(plan)
        # Parse crop=W:H:X:Y
        import re
        m = re.search(r"crop=(\d+):(\d+):(\d+):(\d+)", graph)
        assert m, f"No crop found in: {graph}"
        w, h = int(m.group(1)), int(m.group(2))
        assert w % 2 == 0, f"Crop width {w} is not even"
        assert h % 2 == 0, f"Crop height {h} is not even"


class TestWideMasterFilter:
    def test_wide_master_uses_blurred_fill(self):
        """Phase 4: WIDE_MASTER renders a blurred-fill background,
        NOT black-bar letterboxing. Any ``pad=`` in the graph would
        be a regression."""
        plan = _make_plan([RenderOp(
            kind=RenderOpKind.WIDE_MASTER,
            start_sec=0.0, end_sec=5.0,
            primary_rect=Rect(0, 0, 1, 1),
        )])
        graph = _build_filter_graph(plan)
        # Blurred-fill chain hallmarks
        assert "gblur=" in graph
        assert "overlay=" in graph
        assert "split=2" in graph
        # No black-bar pad anywhere
        assert "pad=" not in graph
        assert ":black" not in graph


class TestBlurFillFilter:
    def test_blur_fill_has_gblur_and_overlay(self):
        plan = _make_plan([RenderOp(
            kind=RenderOpKind.BLUR_FILL,
            start_sec=0.0, end_sec=5.0,
            primary_rect=Rect(0, 0, 1, 1),
        )])
        graph = _build_filter_graph(plan)
        assert f"gblur=sigma={BLUR_SIGMA}" in graph
        assert f"eq=brightness={BLUR_BRIGHTNESS}" in graph
        assert "overlay=" in graph
        assert "split=2" in graph


class TestSplitScreenFilter:
    def test_split_screen_has_vstack(self):
        plan = _make_plan([RenderOp(
            kind=RenderOpKind.SPLIT_SCREEN,
            start_sec=0.0, end_sec=5.0,
            primary_rect=Rect(0.1, 0.0, 0.5, 1.0),
            secondary_rect=Rect(0.5, 0.0, 0.5, 1.0),
        )])
        graph = _build_filter_graph(plan)
        assert "vstack" in graph
        assert "split=2" in graph


class TestStackedGameplayFilter:
    def test_stacked_gameplay_has_vstack(self):
        plan = _make_plan([RenderOp(
            kind=RenderOpKind.STACKED_GAMEPLAY,
            start_sec=0.0, end_sec=5.0,
            primary_rect=Rect(0.0, 0.0, 0.8, 0.6),
            secondary_rect=Rect(0.65, 0.65, 0.3, 0.3),
        )])
        graph = _build_filter_graph(plan)
        assert "vstack" in graph


class TestGrid2x2Filter:
    def test_grid_has_xstack(self):
        plan = _make_plan([RenderOp(
            kind=RenderOpKind.GRID_2X2,
            start_sec=0.0, end_sec=5.0,
            primary_rect=Rect(0.0, 0.0, 0.5, 0.5),
            secondary_rect=Rect(0.5, 0.0, 0.5, 0.5),
            tertiary_rect=Rect(0.0, 0.5, 0.5, 0.5),
            quaternary_rect=Rect(0.5, 0.5, 0.5, 0.5),
        )])
        graph = _build_filter_graph(plan)
        assert "xstack" in graph
        assert "split=4" in graph


class TestTrackingCropFilter:
    def test_tracking_crop_has_piecewise_expression(self):
        plan = _make_plan([RenderOp(
            kind=RenderOpKind.TRACKING_CROP,
            start_sec=0.0, end_sec=10.0,
            primary_rect=Rect(0.2, 0.0, 0.5, 1.0),
            motion_path=[
                MotionKeypoint(t=0.0, rect=Rect(0.2, 0.0, 0.5, 1.0)),
                MotionKeypoint(t=5.0, rect=Rect(0.4, 0.0, 0.5, 1.0)),
                MotionKeypoint(t=10.0, rect=Rect(0.3, 0.0, 0.5, 1.0)),
            ],
        )])
        graph = _build_filter_graph(plan)
        assert "crop=" in graph
        assert "clip(" in graph  # clamped expression
        assert "if(lt(t" in graph  # piecewise conditional


class TestTransitions:
    def test_xfade_applied_for_nonzero_ease(self):
        plan = _make_plan([
            RenderOp(
                kind=RenderOpKind.CROP,
                start_sec=0.0, end_sec=5.0,
                primary_rect=Rect(0.2, 0.0, 0.5, 1.0),
            ),
            RenderOp(
                kind=RenderOpKind.CROP,
                start_sec=5.0, end_sec=10.0,
                primary_rect=Rect(0.5, 0.0, 0.5, 1.0),
                ease_in_ms=500,
            ),
        ])
        graph = _build_filter_graph(plan)
        assert "xfade" in graph
        assert "transition=fade" in graph

    def test_no_xfade_for_zero_ease(self):
        plan = _make_plan([
            RenderOp(
                kind=RenderOpKind.CROP,
                start_sec=0.0, end_sec=5.0,
                primary_rect=Rect(0.2, 0.0, 0.5, 1.0),
            ),
            RenderOp(
                kind=RenderOpKind.CROP,
                start_sec=5.0, end_sec=10.0,
                primary_rect=Rect(0.5, 0.0, 0.5, 1.0),
                ease_in_ms=0,
            ),
        ])
        graph = _build_filter_graph(plan)
        assert "xfade" not in graph
        assert "concat=n=2" in graph


class TestMultiOpConcat:
    def test_three_ops_produce_concat_n3(self):
        plan = _make_plan([
            RenderOp(kind=RenderOpKind.CROP, start_sec=0.0, end_sec=3.0,
                     primary_rect=Rect(0.2, 0.0, 0.5, 1.0)),
            RenderOp(kind=RenderOpKind.WIDE_MASTER, start_sec=3.0, end_sec=6.0,
                     primary_rect=Rect(0, 0, 1, 1)),
            RenderOp(kind=RenderOpKind.CROP, start_sec=6.0, end_sec=10.0,
                     primary_rect=Rect(0.3, 0.0, 0.5, 1.0)),
        ])
        graph = _build_filter_graph(plan)
        assert "concat=n=3:v=1:a=0" in graph


class TestBuildFFmpegCommand:
    def test_command_has_filter_complex_script(self):
        plan = _make_plan([RenderOp(
            kind=RenderOpKind.CROP,
            start_sec=0.0, end_sec=5.0,
            primary_rect=Rect(0.2, 0.0, 0.5, 1.0),
        )])
        cmd, script_path = build_ffmpeg_command(
            plan, "/tmp/input.mp4", "/tmp/output.mp4", use_gpu=False,
        )
        try:
            assert "-filter_complex_script" in cmd
            assert script_path in cmd
            assert os.path.exists(script_path)
        finally:
            cleanup_filter_script(script_path)

    def test_clip_export_uses_seek(self):
        plan = _make_plan([RenderOp(
            kind=RenderOpKind.CROP,
            start_sec=0.0, end_sec=10.0,
            primary_rect=Rect(0.2, 0.0, 0.5, 1.0),
        )])
        plan.source_offset_sec = 30.0
        plan.total_duration_sec = 10.0

        cmd, script_path = build_ffmpeg_command(
            plan, "/tmp/input.mp4", "/tmp/output.mp4", use_gpu=False,
        )
        try:
            assert "-ss" in cmd
            assert "30.000" in cmd
            assert "-t" in cmd
        finally:
            cleanup_filter_script(script_path)

    def test_gpu_encoder_flags(self):
        plan = _make_plan([RenderOp(
            kind=RenderOpKind.CROP,
            start_sec=0.0, end_sec=5.0,
            primary_rect=Rect(0.2, 0.0, 0.5, 1.0),
        )])
        cmd, script_path = build_ffmpeg_command(
            plan, "/tmp/input.mp4", "/tmp/output.mp4", use_gpu=True,
        )
        try:
            assert "h264_nvenc" in cmd
        finally:
            cleanup_filter_script(script_path)

    def test_cpu_encoder_flags(self):
        plan = _make_plan([RenderOp(
            kind=RenderOpKind.CROP,
            start_sec=0.0, end_sec=5.0,
            primary_rect=Rect(0.2, 0.0, 0.5, 1.0),
        )])
        cmd, script_path = build_ffmpeg_command(
            plan, "/tmp/input.mp4", "/tmp/output.mp4", use_gpu=False,
        )
        try:
            assert "libx264" in cmd
        finally:
            cleanup_filter_script(script_path)

    def test_filter_script_cleanup(self):
        plan = _make_plan([RenderOp(
            kind=RenderOpKind.CROP,
            start_sec=0.0, end_sec=5.0,
            primary_rect=Rect(0.2, 0.0, 0.5, 1.0),
        )])
        _, script_path = build_ffmpeg_command(
            plan, "/tmp/input.mp4", "/tmp/output.mp4",
        )
        assert os.path.exists(script_path)
        cleanup_filter_script(script_path)
        assert not os.path.exists(script_path)


class TestLargeFilterGraph:
    """Verify filter_complex_script handles large numbers of ops."""

    def test_100_ops_generate_valid_graph(self):
        ops = []
        for i in range(100):
            ops.append(RenderOp(
                kind=RenderOpKind.CROP,
                start_sec=i * 1.0,
                end_sec=(i + 1) * 1.0,
                primary_rect=Rect(0.2 + (i % 5) * 0.1, 0.0, 0.5, 1.0),
            ))
        plan = _make_plan(ops, duration=100.0)
        graph = _build_filter_graph(plan)
        assert "concat=n=100:v=1:a=0" in graph
        assert "[outv]" in graph
        # Verify no duplicate labels
        import re
        labels = re.findall(r'\[v\d+\]', graph)
        label_set = set(labels)
        # Each label appears twice (output + concat input), so unique count = 100
        assert len(label_set) == 100
