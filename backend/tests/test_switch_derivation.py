"""Tests for margin-based subject switch derivation."""

import pytest

from backend.services.intent_tracker import (
    IntentSignal,
    SubjectSwitch,
    smooth_intent_timeline,
    derive_switches,
)


def _make_smoothed_signals(timeline, ema_alpha=0.4):
    """Build raw IntentSignals and smooth them.

    timeline: list of (timestamp, {subject_id: raw_confidence})
    """
    raw = [IntentSignal(timestamp=t, candidates=dict(c)) for t, c in timeline]
    return smooth_intent_timeline(raw, ema_alpha=ema_alpha)


class TestDeriveSwitches:
    def test_single_subject_throughout(self):
        """Single subject → one initial switch, no others."""
        timeline = [(i * 0.5, {0: 0.8}) for i in range(10)]
        smoothed = _make_smoothed_signals(timeline)
        switches = derive_switches(smoothed, switch_margin=0.15)
        assert len(switches) == 1
        assert switches[0].from_id is None
        assert switches[0].to_id == 0
        assert switches[0].reason == "initial"

    def test_clean_speaker_turn(self):
        """A→B clean turn produces exactly one non-initial switch."""
        timeline = []
        # A dominant for 5 frames
        for i in range(5):
            timeline.append((i * 0.5, {0: 0.9, 1: 0.05}))
        # B dominant for 5 frames
        for i in range(5):
            timeline.append((2.5 + i * 0.5, {0: 0.05, 1: 0.9}))

        smoothed = _make_smoothed_signals(timeline, ema_alpha=0.4)
        switches = derive_switches(smoothed, switch_margin=0.15, min_switch_confidence=0.35)

        non_initial = [s for s in switches if s.from_id is not None]
        assert len(non_initial) == 1
        assert non_initial[0].from_id == 0
        assert non_initial[0].to_id == 1

    def test_ambiguous_overlap_no_switch(self):
        """Both candidates at ~0.5 → no switch after initial."""
        timeline = [(i * 0.5, {0: 0.5, 1: 0.5}) for i in range(10)]
        smoothed = _make_smoothed_signals(timeline)
        switches = derive_switches(smoothed, switch_margin=0.15)
        non_initial = [s for s in switches if s.from_id is not None]
        assert len(non_initial) == 0

    def test_brief_interruption_suppressed(self):
        """B spikes for 1 frame → EMA suppresses, no switch."""
        timeline = []
        for i in range(5):
            timeline.append((i * 0.5, {0: 0.8, 1: 0.1}))
        # Single spike
        timeline.append((2.5, {0: 0.1, 1: 0.9}))
        # Back to A
        for i in range(5):
            timeline.append((3.0 + i * 0.5, {0: 0.8, 1: 0.1}))

        smoothed = _make_smoothed_signals(timeline, ema_alpha=0.4)
        switches = derive_switches(smoothed, switch_margin=0.15)
        non_initial = [s for s in switches if s.from_id is not None]
        assert len(non_initial) == 0, f"Expected 0 switches from brief spike, got {len(non_initial)}"

    def test_sustained_takeover_fires(self):
        """B sustained for 4+ frames at +0.2 over A → switch fires."""
        timeline = []
        for i in range(4):
            timeline.append((i * 0.5, {0: 0.8, 1: 0.1}))
        for i in range(6):
            timeline.append((2.0 + i * 0.5, {0: 0.2, 1: 0.9}))

        smoothed = _make_smoothed_signals(timeline, ema_alpha=0.4)
        switches = derive_switches(smoothed, switch_margin=0.15, min_switch_confidence=0.35)
        non_initial = [s for s in switches if s.from_id is not None]
        assert len(non_initial) >= 1
        assert non_initial[0].to_id == 1

    def test_a_drops_b_emerges(self):
        """A drops to 0, B emerges → switch fires when B passes min_switch_confidence."""
        timeline = []
        for i in range(4):
            timeline.append((i * 0.5, {0: 0.8}))
        for i in range(6):
            timeline.append((2.0 + i * 0.5, {1: 0.7}))

        smoothed = _make_smoothed_signals(timeline, ema_alpha=0.4)
        switches = derive_switches(smoothed, switch_margin=0.15, min_switch_confidence=0.35)
        non_initial = [s for s in switches if s.from_id is not None]
        assert len(non_initial) >= 1
        assert non_initial[0].to_id == 1

    def test_empty_smoothed_signals(self):
        """Empty input → empty output."""
        switches = derive_switches([], switch_margin=0.15)
        assert switches == []

    def test_below_min_confidence_no_initial(self):
        """All candidates below min_switch_confidence → no initial switch."""
        timeline = [(i * 0.5, {0: 0.1}) for i in range(5)]
        smoothed = _make_smoothed_signals(timeline, ema_alpha=0.4)
        switches = derive_switches(smoothed, switch_margin=0.15, min_switch_confidence=0.35)
        assert len(switches) == 0

    def test_chosen_id_set_on_signals(self):
        """derive_switches should set chosen_id on each signal after adoption."""
        timeline = [(i * 0.5, {0: 0.8}) for i in range(10)]
        smoothed = _make_smoothed_signals(timeline, ema_alpha=0.4)
        derive_switches(smoothed, switch_margin=0.15, min_switch_confidence=0.35)
        # After enough frames, smoothed value exceeds threshold and chosen_id is set
        adopted = [s for s in smoothed if s.chosen_id == 0]
        assert len(adopted) > 0, "Expected at least some frames with chosen_id=0"

    def test_switch_logs_correct_margin(self):
        """The margin in the switch should equal conf_to - conf_from."""
        timeline = []
        for i in range(4):
            timeline.append((i * 0.5, {0: 0.8, 1: 0.1}))
        for i in range(6):
            timeline.append((2.0 + i * 0.5, {0: 0.1, 1: 0.9}))

        smoothed = _make_smoothed_signals(timeline, ema_alpha=0.4)
        switches = derive_switches(smoothed, switch_margin=0.15)
        for sw in switches:
            if sw.from_id is not None:
                assert sw.margin == pytest.approx(sw.conf_to - sw.conf_from, abs=0.001)
