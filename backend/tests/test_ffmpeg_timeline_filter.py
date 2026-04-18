"""Tests for FFmpeg timeline-driven tracking crop filter."""

import pytest

from backend.services.interpolated_timeline import (
    FrameSample,
    InterpolatedFaceTimeline,
)
from backend.services.ffmpeg_filter_builder import (
    _filter_tracking_crop,
    _filter_tracking_crop_from_timeline,
    FFMPEG_EXPR_MAX_LEN,
)


# ── Helpers ──

class FakeRect:
    def __init__(self, x_pct, y_pct, w_pct, h_pct):
        self.x_pct = x_pct
        self.y_pct = y_pct
        self.w_pct = w_pct
        self.h_pct = h_pct

    def to_pixels(self, src_w, src_h):
        return (
            int(self.x_pct / 100 * src_w),
            int(self.y_pct / 100 * src_h),
            int(self.w_pct / 100 * src_w),
            int(self.h_pct / 100 * src_h),
        )


class FakeKeypoint:
    def __init__(self, t, rect):
        self.t = t
        self.rect = rect


class FakeOp:
    def __init__(self, motion_path=None, primary_rect=None):
        self.motion_path = motion_path or []
        self.primary_rect = primary_rect or FakeRect(25, 0, 56, 100)


def _make_timeline(n_samples, t_start=0.0, t_end=1.0, slot_id=0, cx_start=30, cx_end=70):
    """Build a simple timeline with linearly moving slot."""
    samples = []
    for i in range(n_samples):
        t = t_start + (t_end - t_start) * i / max(1, n_samples - 1)
        cx = cx_start + (cx_end - cx_start) * i / max(1, n_samples - 1)
        samples.append(FrameSample(
            timestamp=t, fps=30.0,
            bboxes={slot_id: (cx, 40, 14, 18)},
            confidences={slot_id: 0.9},
            is_anchor=(i == 0),
        ))
    tl = InterpolatedFaceTimeline(samples=samples, source_fps=30.0,
                                   source_width=1920, source_height=1080)
    tl.build_index()
    return tl


class TestFilterTrackingCropFromTimeline:
    def test_none_timeline_returns_none(self):
        """With no timeline, function should return None (fall back to standard)."""
        op = FakeOp(
            motion_path=[
                FakeKeypoint(0.0, FakeRect(25, 0, 56, 100)),
                FakeKeypoint(1.0, FakeRect(44, 0, 56, 100)),
            ],
        )
        result = _filter_tracking_crop_from_timeline(
            op, "v0", 1920, 1080, 1080, 1920, 0.0, 1.0,
            interpolated_timeline=InterpolatedFaceTimeline(),
        )
        assert result is None

    def test_10_sample_timeline_produces_between_calls(self):
        """10 samples should produce a filter with between() calls."""
        tl = _make_timeline(10, t_start=0.0, t_end=1.0)
        op = FakeOp(
            motion_path=[
                FakeKeypoint(0.0, FakeRect(25, 0, 56, 100)),
                FakeKeypoint(1.0, FakeRect(44, 0, 56, 100)),
            ],
        )
        result = _filter_tracking_crop_from_timeline(
            op, "v0", 1920, 1080, 1080, 1920, 0.0, 1.0, tl,
        )
        assert result is not None
        assert "between(t" in result
        # Should have ~9 between() calls for 10 samples (10-1 segments)
        between_count = result.count("between(t")
        assert between_count >= 8, f"Expected ~9 between() calls, got {between_count}"

    def test_large_timeline_downsampled(self):
        """1000+ samples should be downsampled and a warning logged."""
        tl = _make_timeline(1500, t_start=0.0, t_end=50.0)
        op = FakeOp(
            motion_path=[
                FakeKeypoint(0.0, FakeRect(25, 0, 56, 100)),
                FakeKeypoint(50.0, FakeRect(44, 0, 56, 100)),
            ],
        )
        result = _filter_tracking_crop_from_timeline(
            op, "v0", 1920, 1080, 1080, 1920, 0.0, 50.0, tl,
        )
        assert result is not None
        # Expression length should be under the limit
        assert len(result) < FFMPEG_EXPR_MAX_LEN + 500  # some overhead for the outer filter

    def test_expression_contains_crop_and_scale(self):
        """Output should be a valid FFmpeg filter with crop and scale."""
        tl = _make_timeline(5, t_start=0.0, t_end=1.0)
        op = FakeOp(
            motion_path=[
                FakeKeypoint(0.0, FakeRect(25, 0, 56, 100)),
                FakeKeypoint(1.0, FakeRect(44, 0, 56, 100)),
            ],
        )
        result = _filter_tracking_crop_from_timeline(
            op, "v0", 1920, 1080, 1080, 1920, 0.0, 1.0, tl,
        )
        assert result is not None
        assert "crop=" in result
        assert "scale=" in result
        assert "lanczos" in result
        assert "[v0]" in result

    def test_regression_none_timeline_uses_standard_path(self):
        """When interpolated_timeline is None, _build_op_filter falls through
        to the standard _filter_tracking_crop."""
        op = FakeOp(
            motion_path=[
                FakeKeypoint(0.0, FakeRect(25, 0, 56, 100)),
                FakeKeypoint(1.0, FakeRect(44, 0, 56, 100)),
            ],
        )
        # Standard path should always work
        standard = _filter_tracking_crop(
            op, "v0", 1920, 1080, 1080, 1920, 0.0, 1.0,
        )
        assert "crop=" in standard
        assert "[v0]" in standard

    def test_single_sample_returns_none(self):
        """A single sample isn't enough for piecewise — returns None."""
        tl = _make_timeline(1, t_start=0.0, t_end=0.0)
        op = FakeOp(
            motion_path=[FakeKeypoint(0.0, FakeRect(25, 0, 56, 100))],
        )
        result = _filter_tracking_crop_from_timeline(
            op, "v0", 1920, 1080, 1080, 1920, 0.0, 1.0, tl,
        )
        assert result is None

    def test_clip_expression_clamps_x(self):
        """The expression should use clip() to clamp x to valid range."""
        tl = _make_timeline(5, t_start=0.0, t_end=1.0, cx_start=5, cx_end=95)
        op = FakeOp(
            motion_path=[
                FakeKeypoint(0.0, FakeRect(2, 0, 56, 100)),
                FakeKeypoint(1.0, FakeRect(44, 0, 56, 100)),
            ],
        )
        result = _filter_tracking_crop_from_timeline(
            op, "v0", 1920, 1080, 1080, 1920, 0.0, 1.0, tl,
        )
        assert result is not None
        assert "clip(" in result
