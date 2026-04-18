"""Tests for the deterministic two-pass _smooth_layout_votes in layout_engine.py."""

import time

from backend.services.layout_engine import _smooth_layout_votes


class TestSmoothLayoutVotes:
    def test_empty_input(self):
        assert _smooth_layout_votes([]) == []

    def test_single_segment(self):
        votes = [(0.0, "SINGLE")]
        result = _smooth_layout_votes(votes, min_duration=2.0)
        assert len(result) == 1
        assert result[0][2] == "SINGLE"

    def test_three_alternating_short_segments(self):
        """[A 0.5s, B 0.3s, A 0.4s] with min_duration=2.0 -> single merged."""
        votes = [
            (0.0, "A"),
            (0.5, "B"),
            (0.8, "A"),
        ]
        result = _smooth_layout_votes(votes, min_duration=2.0)
        # All segments are < 2.0s, so they merge into one
        assert len(result) == 1

    def test_adversarial_100_alternating(self):
        """100 segments of alternating modes each 0.1s long.

        Must complete in <10ms and not hang.
        """
        votes = [(i * 0.1, "A" if i % 2 == 0 else "B") for i in range(100)]
        start = time.perf_counter()
        result = _smooth_layout_votes(votes, min_duration=2.0)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.01, f"Smoothing took {elapsed:.3f}s — must be <10ms"
        # Should produce a small number of segments (1-3), not 100
        assert len(result) <= 5

    def test_adversarial_1000_alternating_no_hang(self):
        """1000 alternating segments — worst case for the old iterative loop.

        The old code would hit max_passes=50 and still not converge.
        The new code must complete in O(n) time.
        """
        votes = [(i * 0.05, "X" if i % 2 == 0 else "Y") for i in range(1000)]
        start = time.perf_counter()
        result = _smooth_layout_votes(votes, min_duration=2.0)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1, f"Smoothing took {elapsed:.3f}s — must be <100ms for 1000 segments"
        assert len(result) <= 5

    def test_long_segments_unchanged(self):
        """Segments already longer than min_duration stay unchanged."""
        # Each mode runs for multiple votes so the raw segments are long enough
        votes = (
            [(t * 0.5, "A") for t in range(0, 10)]    # 0.0-5.0 = A (5s)
            + [(t * 0.5, "B") for t in range(10, 20)]  # 5.0-10.0 = B (5s)
            + [(t * 0.5, "A") for t in range(20, 30)]  # 10.0-15.0 = A (5s)
        )
        result = _smooth_layout_votes(votes, min_duration=2.0)
        assert len(result) == 3
        assert result[0][2] == "A"
        assert result[1][2] == "B"
        assert result[2][2] == "A"

    def test_short_between_long_absorbed(self):
        """A short segment between two long ones is absorbed."""
        votes = [
            (0.0, "A"),
            # 0.0-5.0 = A (5s, long)
            (5.0, "B"),
            # 5.0-5.3 = B (0.3s, short)
            (5.3, "A"),
            # 5.3-10.0 = A (4.7s, long)
        ]
        result = _smooth_layout_votes(votes, min_duration=2.0)
        # The short B should be absorbed, leaving just A or A+A merged
        modes = [r[2] for r in result]
        assert "B" not in modes or len(result) <= 2
