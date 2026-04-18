"""Tests for the confidence floor hard gate enforcement."""

import logging
import pytest
from unittest.mock import MagicMock

from backend.services.confidence_audit import (
    enforce_confidence_floor,
    CONFIDENCE_FLOOR_FOR_CROP,
)


def _make_segment(strategy="stationary", confidence=0.85, layout="single",
                  subject_x=30, active_slot=0, start=0.0, end=1.0):
    seg = MagicMock()
    seg.strategy = strategy
    seg.confidence = confidence
    seg.layout = layout
    seg.subject_x = subject_x
    seg.active_slot = active_slot
    seg.start = start
    seg.end = end
    seg.fallback_reason = None
    return seg


class TestConfidenceFloorEnforcement:
    def test_floor_is_050(self):
        """Confidence floor is 0.50 — faces detected at 0.55 should crop."""
        assert CONFIDENCE_FLOOR_FOR_CROP == 0.50

    def test_high_confidence_unchanged(self):
        """85% confidence stationary segment stays unchanged."""
        seg = _make_segment(strategy="stationary", confidence=0.85)
        result = enforce_confidence_floor([seg], job_id="test")
        assert result[0].strategy == "stationary"
        assert result[0].layout == "single"
        assert result[0].subject_x == 30

    def test_medium_confidence_unchanged(self):
        """55% confidence stationary segment stays unchanged (above 0.50 floor)."""
        seg = _make_segment(strategy="stationary", confidence=0.55)
        result = enforce_confidence_floor([seg], job_id="test")
        assert result[0].strategy == "stationary"

    def test_blur_fill_already_fallback_unchanged(self):
        """blur_fill at 30% confidence stays unchanged (already a fallback)."""
        seg = _make_segment(strategy="blur_fill", confidence=0.30, layout="blur_fill")
        result = enforce_confidence_floor([seg], job_id="test")
        assert result[0].strategy == "blur_fill"
        assert result[0].fallback_reason is None  # not re-tagged

    def test_tracking_low_confidence_downgraded(self):
        """Tracking strategy at 0.40 confidence is downgraded (below 0.50)."""
        seg = _make_segment(strategy="tracking", confidence=0.40)
        result = enforce_confidence_floor([seg], job_id="test")
        assert result[0].strategy == "blur_fill"

    def test_panning_below_floor_downgraded(self):
        """Panning strategy at 0.45 confidence is downgraded."""
        seg = _make_segment(strategy="panning", confidence=0.45)
        result = enforce_confidence_floor([seg], job_id="test")
        assert result[0].strategy == "blur_fill"

    def test_multiple_violations_all_downgraded(self, caplog):
        """Multiple violations in one list → all downgraded, summary log emitted."""
        segs = [
            _make_segment(strategy="stationary", confidence=0.24, start=0.0, end=1.0),
            _make_segment(strategy="tracking", confidence=0.30, start=1.0, end=2.0),
            _make_segment(strategy="stationary", confidence=0.90, start=2.0, end=3.0),
        ]
        with caplog.at_level(logging.WARNING):
            result = enforce_confidence_floor(segs, job_id="test")

        assert result[0].strategy == "blur_fill"
        assert result[1].strategy == "blur_fill"
        assert result[2].strategy == "stationary"  # high confidence, untouched
        assert "downgraded 2 segments" in caplog.text

    def test_exactly_at_floor_unchanged(self):
        """Segment at exactly 0.50 confidence is NOT downgraded."""
        seg = _make_segment(strategy="stationary", confidence=0.50)
        result = enforce_confidence_floor([seg], job_id="test")
        assert result[0].strategy == "stationary"

    def test_24_percent_face_downgraded(self):
        """24% face confidence → blur_fill (well below 0.50 floor)."""
        seg = _make_segment(strategy="stationary", confidence=0.24)
        result = enforce_confidence_floor([seg], job_id="test")
        assert result[0].strategy == "blur_fill"
        assert result[0].subject_x == 50

    def test_empty_segment_list(self):
        """Empty list returns empty list with no errors."""
        result = enforce_confidence_floor([], job_id="test")
        assert result == []
