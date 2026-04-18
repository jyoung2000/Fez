"""Tests for the minimum-dwell collapse in active_speaker.py.

A short 0.3s blip from speaker B in the middle of a long speaker A run
should be collapsed away (the listener laughing/yawning shouldn't hijack
the speaker timeline).
"""

from backend.services.active_speaker import (
    SpeakerEvent,
    _collapse_short_runs,
    MIN_DWELL_SECONDS,
)


def _mk(start, end, slot, conf=0.6):
    return SpeakerEvent(start=start, end=end, slot_id=slot, confidence=conf)


class TestMinDwellCollapse:
    def test_short_B_blip_inside_long_A_is_collapsed(self):
        """A single 0.3s B in the middle of a long A run is collapsed."""
        events = [
            _mk(0.0, 2.0, 0, conf=0.7),   # A, long
            _mk(2.0, 2.3, 1, conf=0.5),   # B, 0.3s (below MIN_DWELL)
            _mk(2.3, 5.0, 0, conf=0.7),   # A, long
        ]
        collapsed, n = _collapse_short_runs(events)
        # The short B run should have been removed entirely.
        assert n >= 1
        slots = [ev.slot_id for ev in collapsed]
        assert 1 not in slots, f"slot 1 should be collapsed, got {slots}"
        # Result should span the whole timeline end-to-end as slot A.
        assert collapsed[0].start == 0.0
        assert collapsed[-1].end == 5.0

    def test_alternating_10_segments_short_B_collapsed(self):
        """Synthetic 10-event timeline with one short B inside long A.

        Segments: A(1.0) B(1.0) A(1.0) B(1.0) A(1.0) - then a tiny
        0.3s B inside a long A run that follows.
        """
        events = [
            _mk(0.0, 1.0, 0, conf=0.7),
            _mk(1.0, 2.0, 1, conf=0.7),
            _mk(2.0, 3.0, 0, conf=0.7),
            _mk(3.0, 4.0, 1, conf=0.7),
            _mk(4.0, 10.0, 0, conf=0.8),       # long A
            _mk(10.0, 10.3, 1, conf=0.5),      # short B blip (0.3s)
            _mk(10.3, 16.0, 0, conf=0.8),      # long A continues
            _mk(16.0, 17.0, 1, conf=0.7),
            _mk(17.0, 18.0, 0, conf=0.7),
            _mk(18.0, 19.0, 1, conf=0.7),
        ]
        collapsed, n = _collapse_short_runs(events)
        assert n >= 1, "Expected at least one short run collapse"
        # The 0.3s B between the two long A runs should be gone — no B
        # event should start at 10.0.
        for ev in collapsed:
            assert not (abs(ev.start - 10.0) < 1e-6 and ev.slot_id == 1), (
                f"Short B blip at 10.0 should have been collapsed: {collapsed}"
            )
        # Verify MIN_DWELL is what we assume (documentation / regression).
        assert MIN_DWELL_SECONDS >= 0.5

    def test_long_runs_survive(self):
        """Runs longer than MIN_DWELL must be preserved even when
        confidence is similar across neighbors."""
        events = [
            _mk(0.0, 2.0, 0, conf=0.6),
            _mk(2.0, 4.0, 1, conf=0.6),   # 2s B — well above dwell
            _mk(4.0, 6.0, 0, conf=0.6),
        ]
        collapsed, n = _collapse_short_runs(events)
        assert n == 0
        assert len(collapsed) == 3
        assert [ev.slot_id for ev in collapsed] == [0, 1, 0]
