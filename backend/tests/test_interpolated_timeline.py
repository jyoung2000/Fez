"""Tests for InterpolatedFaceTimeline and SlotTracker primitives."""

import pytest
import numpy as np

from backend.services.interpolated_timeline import (
    FrameSample,
    InterpolatedFaceTimeline,
)
from backend.services.tracker_wrapper import SlotTracker


class TestInterpolatedFaceTimeline:
    def test_empty_timeline_at_returns_none(self):
        tl = InterpolatedFaceTimeline()
        assert tl.at(1.0) is None

    def test_at_exact_timestamp(self):
        samples = [
            FrameSample(timestamp=i / 30.0, fps=30.0, bboxes={0: (50, 40, 10, 14)})
            for i in range(30)
        ]
        tl = InterpolatedFaceTimeline(samples=samples, source_fps=30.0)
        tl.build_index()
        result = tl.at(0.5)
        assert result is not None
        assert abs(result.timestamp - 0.5) < 0.04  # nearest to 0.5s

    def test_at_nearest_fallback(self):
        samples = [
            FrameSample(timestamp=0.0, fps=30.0),
            FrameSample(timestamp=1.0, fps=30.0),
        ]
        tl = InterpolatedFaceTimeline(samples=samples, source_fps=30.0)
        tl.build_index()
        result = tl.at(0.8)
        assert result is not None
        assert result.timestamp == 1.0  # nearest to 0.8

    def test_slot_positions_in_range(self):
        samples = []
        for i in range(30):
            t = i / 30.0
            bboxes = {0: (50 + i, 40, 10, 14)} if i < 20 else {}
            samples.append(FrameSample(timestamp=t, fps=30.0, bboxes=bboxes,
                                       confidences={0: 0.9} if bboxes else {}))
        tl = InterpolatedFaceTimeline(samples=samples)
        positions = tl.slot_positions_in_range(0, 0.0, 1.0)
        assert len(positions) == 20  # slot 0 present in first 20 frames
        for t, cx, cy, w, h, conf in positions:
            assert 0.0 <= t <= 1.0
            assert conf == 0.9

    def test_slot_positions_in_range_empty_for_missing_slot(self):
        samples = [FrameSample(timestamp=0.0, fps=30.0, bboxes={0: (50, 40, 10, 14)})]
        tl = InterpolatedFaceTimeline(samples=samples)
        positions = tl.slot_positions_in_range(99, 0.0, 1.0)
        assert positions == []

    def test_to_dict_summary(self):
        samples = [
            FrameSample(timestamp=0.0, fps=30.0, is_anchor=True),
            FrameSample(timestamp=0.033, fps=30.0, is_anchor=False),
            FrameSample(timestamp=0.066, fps=30.0, is_anchor=False, had_reset=True),
            FrameSample(timestamp=0.1, fps=30.0, is_anchor=True),
        ]
        tl = InterpolatedFaceTimeline(samples=samples, source_fps=30.0)
        summary = tl.to_dict_summary()
        assert summary["n_samples"] == 4
        assert summary["n_anchors"] == 2
        assert summary["n_resets"] == 1
        assert summary["source_fps"] == 30.0
        assert summary["duration"] == pytest.approx(0.1)

    def test_build_index_enables_fast_lookup(self):
        samples = [FrameSample(timestamp=i / 30.0, fps=30.0) for i in range(900)]
        tl = InterpolatedFaceTimeline(samples=samples)
        tl.build_index()
        assert len(tl._index) == 900
        # Exact lookup should work for t=0.5 = 15/30
        result = tl.at(0.5)
        assert result is not None
        assert abs(result.timestamp - 0.5) < 0.04


class TestSlotTracker:
    def test_init_on_valid_region(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a white rectangle to track
        frame[100:200, 200:300] = 255
        tracker = SlotTracker(slot_id=0, backend="KCF")
        ok = tracker.init(frame, (200, 100, 100, 100))
        assert ok is True
        assert tracker._initialized is True

    def test_init_rejects_tiny_bbox(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracker = SlotTracker(slot_id=0, backend="KCF")
        ok = tracker.init(frame, (200, 100, 5, 5))
        assert ok is False
        assert tracker._initialized is False

    def test_update_returns_bbox_after_init(self):
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame1[100:200, 200:300] = 255
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2[100:200, 205:305] = 255  # moved 5px right

        tracker = SlotTracker(slot_id=0, backend="KCF")
        tracker.init(frame1, (200, 100, 100, 100))
        result = tracker.update(frame2)
        assert result is not None
        x, y, w, h = result
        assert w > 0 and h > 0

    def test_update_without_init_returns_none(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracker = SlotTracker(slot_id=0, backend="KCF")
        result = tracker.update(frame)
        assert result is None

    def test_reset_reinitializes(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[100:200, 200:300] = 255
        tracker = SlotTracker(slot_id=0, backend="KCF")
        tracker.init(frame, (200, 100, 100, 100))
        assert tracker._initialized is True
        tracker.reset(frame, (250, 150, 100, 100))
        assert tracker._initialized is True

    def test_mil_fallback(self):
        """MIL backend should always be available."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[100:200, 200:300] = 255
        tracker = SlotTracker(slot_id=0, backend="MIL")
        ok = tracker.init(frame, (200, 100, 100, 100))
        assert ok is True
