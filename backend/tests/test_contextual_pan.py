"""Phase 4 — tests for contextual Ken-Burns pan and the no-black-bars
contract for WIDE_MASTER / BLUR_FILL.

Spec
----

8 tests:

1. test_pan_left_to_right                  — EWS, two saliency peaks
2. test_pan_speed_limit                    — 3s × 1920 → cap 864 px
3. test_pan_no_black_bars                  — pan x within bounds
4. test_short_shot_partial_pan             — 1.5 s shot
5. test_blur_fill_no_black                 — BLUR_FILL render is bar-free
6. test_wide_master_no_black               — WIDE_MASTER render is bar-free
7. test_crop_qa_catches_black              — synthetic black-bar frame
8. test_contextual_pan_render_plan         — RenderOp shape

The renderer-level tests (5/6/7) are skipped automatically when the
sandbox does not have ffmpeg/ffprobe on the PATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import pytest

from backend.services.crop_qa import (
    BLACK_BAR_PIXEL_THRESHOLD,
    CropQaReport,
    validate_no_black_bars,
)
from backend.services.ffmpeg_filter_builder import _build_filter_graph
from backend.services.render_plan import (
    MotionKeypoint,
    Rect,
    RenderOp,
    RenderOpKind,
    RenderPlan,
)
from backend.services.render_plan_builder import (
    PAN_SPEED_MAX_FRAC_PER_SEC,
    _compute_contextual_pan_rects,
    build_render_plan,
)
from backend.services.shot_reframe_advisor import (
    ReframeStrategy,
    SaliencyPeak,
    ShotAnalysis,
    recommend_strategy,
)


_FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")
_skip_ffmpeg = pytest.mark.skipif(
    not _FFMPEG, reason="ffmpeg/ffprobe not available in test sandbox",
)


# ── helpers ────────────────────────────────────────────────────────


class _Seg:
    """Minimal ReframeSegment-like for build_render_plan."""

    def __init__(self, start, end, **kw):
        self.start = float(start)
        self.end = float(end)
        self.subject_x = kw.get("subject_x", 50)
        self.subject_y = kw.get("subject_y", 40)
        self.layout = kw.get("layout", "single")
        self.strategy = kw.get("strategy", "stationary")
        self.reason = kw.get("reason", "")
        self.ease_in_ms = kw.get("ease_in_ms", 0)
        self.content_type = kw.get("content_type", "unknown")
        self.motion_path = kw.get("motion_path", None)
        self.pan_start = kw.get("pan_start", None)
        self.pan_end = kw.get("pan_end", None)


@pytest.fixture(scope="module")
def synthetic_video():
    """A 3-second, 320x240 solid-color mp4 generated once per test run.

    Skipped when ffmpeg is not available. The video is a uniform red
    field — so its renderer outputs should NEVER contain a black band
    on any side.
    """
    if not _FFMPEG:
        pytest.skip("ffmpeg not available")

    cache = os.path.join(tempfile.gettempdir(), "phase4_synth_320x240_red.mp4")
    if not os.path.exists(cache):
        # 320x240 solid red, 3 seconds at 24 fps. Use lavfi color source.
        cmd = [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "color=c=red:s=320x240:d=3:r=24",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
            cache,
        ]
        subprocess.run(cmd, check=True, timeout=60.0)
    return cache


def _render_with_filter(src: str, dst: str, filter_graph: str, dur: float = 3.0):
    """Run ffmpeg with the supplied filter_complex, write ``dst`` mp4."""
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-i", src,
        "-filter_complex", filter_graph,
        "-map", "[outv]",
        "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        dst,
    ]
    subprocess.run(cmd, check=True, timeout=120.0)


# ── 1. Pan left to right ──────────────────────────────────────────


def test_pan_left_to_right():
    """EWS shot with saliency on left then right → advisor returns
    CONTEXTUAL_PAN; the segmenter should produce a left-to-right
    pan via the resulting motion_path."""
    shot = ShotAnalysis(
        shot_idx=0,
        shot_type="EWS",
        content_type="narrative",
        shot_duration_sec=4.0,
        source_width=1920,
        source_height=1080,
        face_samples=[],
        saliency_peaks=[
            SaliencyPeak(timestamp=0.5, x_center=0.15, y_center=0.5, weight=2.0),
            SaliencyPeak(timestamp=2.5, x_center=0.85, y_center=0.5, weight=1.0),
        ],
    )
    advice = recommend_strategy(shot)
    assert advice.strategy == ReframeStrategy.CONTEXTUAL_PAN
    # Primary bbox = highest saliency = the LEFT one.
    assert advice.primary_subject_bbox is not None
    assert advice.primary_subject_bbox[0] < 0.5

    # Build a CONTEXTUAL_PAN op via the helper: start 0.0, end max.
    primary, mp = _compute_contextual_pan_rects(
        pan_start_frac=0.0, pan_end_frac=1.0,
        seg_start=0.0, seg_end=4.0,
        source_w=1920, source_h=1080,
        target_aspect=9 / 16,
    )
    assert mp[0].rect.x < mp[-1].rect.x  # L→R


# ── 2. Pan speed limit ────────────────────────────────────────────


def test_pan_speed_limit():
    """3-second shot, 1920px source → max travel = 0.15 × 1920 × 3
    = 864 px. Builder must clamp to that."""
    primary, mp = _compute_contextual_pan_rects(
        pan_start_frac=0.0,
        pan_end_frac=1.0,         # would request a full-width travel
        seg_start=0.0, seg_end=3.0,
        source_w=1920, source_h=1080,
        target_aspect=9 / 16,
    )
    travel_px = (mp[-1].rect.x - mp[0].rect.x) * 1920
    expected_cap = PAN_SPEED_MAX_FRAC_PER_SEC * 1920 * 3.0  # 864
    assert travel_px <= expected_cap + 1.0
    # And the cap is precisely 864 px in this configuration.
    assert abs(expected_cap - 864.0) < 0.01


# ── 3. Pan never produces black bars ──────────────────────────────


def test_pan_no_black_bars():
    """For any seg start/end, the FFmpeg crop x is clamped to
    [0, source_w - crop_w]; the filter string contains the clip()
    call and never appears as ``pad=`` (which would produce black)."""
    op = RenderOp(
        kind=RenderOpKind.CONTEXTUAL_PAN,
        start_sec=0.0, end_sec=4.0,
        primary_rect=Rect(x=0.0, y=0.0, w=0.3375, h=1.0),
        motion_path=[
            MotionKeypoint(t=0.0, rect=Rect(x=0.0, y=0.0, w=0.3375, h=1.0)),
            MotionKeypoint(t=4.0, rect=Rect(x=0.6625, y=0.0, w=0.3375, h=1.0)),
        ],
    )
    plan = RenderPlan(
        source_width=1920, source_height=1080,
        target_width=1080, target_height=1920,
        total_duration_sec=4.0, fps=30.0, ops=[op],
    )
    graph = _build_filter_graph(plan)
    assert "clip(" in graph
    assert "pad=" not in graph
    # Crop x is clamped to source_w - crop_w (0.3375*1920 = 648; max_x = 1272)
    assert "1272" in graph or "1273" in graph


# ── 4. Short shot → only as much as speed allows ──────────────────


def test_short_shot_partial_pan():
    """1.5 s shot: max travel = 0.15 × 1920 × 1.5 = 432 px."""
    _, mp = _compute_contextual_pan_rects(
        pan_start_frac=0.0, pan_end_frac=1.0,
        seg_start=0.0, seg_end=1.5,
        source_w=1920, source_h=1080,
        target_aspect=9 / 16,
    )
    travel_px = (mp[-1].rect.x - mp[0].rect.x) * 1920
    assert travel_px <= 432.0 + 1.0
    assert travel_px > 0.0


# ── 5/6. Renderer no-black-bar checks ─────────────────────────────


@_skip_ffmpeg
def test_blur_fill_no_black(synthetic_video, tmp_path):
    """A BLUR_FILL render of a solid red source must not have black
    anywhere on any edge."""
    op = RenderOp(
        kind=RenderOpKind.BLUR_FILL,
        start_sec=0.0, end_sec=3.0,
        primary_rect=Rect(0.0, 0.0, 1.0, 1.0),
    )
    plan = RenderPlan(
        source_width=320, source_height=240,
        target_width=1080, target_height=1920,
        total_duration_sec=3.0, fps=24.0, ops=[op],
    )
    graph = _build_filter_graph(plan)
    out = str(tmp_path / "blur_fill.mp4")
    _render_with_filter(synthetic_video, out, graph, dur=3.0)
    report = validate_no_black_bars(out, fps=1.0)
    assert report.passed, (
        f"BLUR_FILL produced black borders: frames={report.frames_with_black} "
        f"sampled={report.total_frames_sampled} note={report.note}"
    )


@_skip_ffmpeg
def test_wide_master_no_black(synthetic_video, tmp_path):
    """WIDE_MASTER must also render with a blurred background, NEVER
    black bars."""
    op = RenderOp(
        kind=RenderOpKind.WIDE_MASTER,
        start_sec=0.0, end_sec=3.0,
        primary_rect=Rect(0.0, 0.0, 1.0, 1.0),
    )
    plan = RenderPlan(
        source_width=320, source_height=240,
        target_width=1080, target_height=1920,
        total_duration_sec=3.0, fps=24.0, ops=[op],
    )
    graph = _build_filter_graph(plan)
    # Sanity: the new graph contains gblur (no pad).
    assert "gblur" in graph
    assert "pad=" not in graph

    out = str(tmp_path / "wide_master.mp4")
    _render_with_filter(synthetic_video, out, graph, dur=3.0)
    report = validate_no_black_bars(out, fps=1.0)
    assert report.passed, (
        f"WIDE_MASTER produced black borders: frames={report.frames_with_black} "
        f"sampled={report.total_frames_sampled} note={report.note}"
    )


# ── 7. crop_qa catches an injected black-bar frame ────────────────


@_skip_ffmpeg
def test_crop_qa_catches_black(tmp_path):
    """Inject a frame with a clearly black border → QA flags it."""
    # 9:16 black-bar style: source 16:9 scaled to 1080:-1 then padded
    # with black to 1080x1920. Mirrors the OLD WIDE_MASTER chain.
    src = os.path.join(tempfile.gettempdir(), "phase4_red_for_qa.mp4")
    if not os.path.exists(src):
        subprocess.run([
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "color=c=red:s=320x240:d=3:r=24",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
            src,
        ], check=True, timeout=60.0)

    out = str(tmp_path / "blackbars.mp4")
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-i", src,
        "-vf", "scale=1080:-1,pad=1080:1920:0:(oh-ih)/2:black",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
        out,
    ]
    subprocess.run(cmd, check=True, timeout=60.0)

    report = validate_no_black_bars(out, fps=1.0)
    assert isinstance(report, CropQaReport)
    assert not report.passed, (
        f"QA failed to flag obvious black bars: report={report}"
    )
    assert report.total_frames_sampled > 0
    assert len(report.frames_with_black) >= 1


# ── 8. CONTEXTUAL_PAN render-plan shape ───────────────────────────


def test_contextual_pan_render_plan():
    """A segment with strategy=contextual_pan + pan_start/pan_end
    should produce a CONTEXTUAL_PAN RenderOp with the correct
    primary_rect and motion_path."""
    seg = _Seg(
        0.0, 4.0,
        layout="single", strategy="contextual_pan",
        pan_start=0.0, pan_end=0.5,
    )
    plan = build_render_plan(
        [seg],
        source_width=1920, source_height=1080,
        source_fps=30.0,
        target_aspect="9:16",
        target_height_px=1920,
    )
    assert len(plan.ops) == 1
    op = plan.ops[0]
    assert op.kind == RenderOpKind.CONTEXTUAL_PAN
    # Crop dimensions are constant across the two keypoints
    assert len(op.motion_path) == 2
    assert abs(op.motion_path[0].rect.w - op.motion_path[-1].rect.w) < 1e-6
    assert abs(op.motion_path[0].rect.h - op.motion_path[-1].rect.h) < 1e-6
    # primary_rect = first keypoint
    assert abs(op.primary_rect.x - op.motion_path[0].rect.x) < 1e-6
    # Pan goes left → right
    assert op.motion_path[-1].rect.x > op.motion_path[0].rect.x
    # Travel is clamped to the speed cap (PAN_SPEED_MAX × duration).
    travel_frac = op.motion_path[-1].rect.x - op.motion_path[0].rect.x
    assert travel_frac <= PAN_SPEED_MAX_FRAC_PER_SEC * 4.0 + 1e-6
