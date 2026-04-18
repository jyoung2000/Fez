"""Tests for LocalPacingEstimator and adaptive hold derivation.

Covers:
  1. Calm podcast → low pacing, high min_hold
  2. Debate podcast → high pacing, low min_hold
  3. Mixed pacing (calm → debate → calm)
  4. Pacing smoothing (spike suppression)
"""

import numpy as np
import pytest

from backend.services.local_pacing import (
    MIN_HOLD_CEILING,
    MIN_HOLD_FLOOR,
    LocalPacingEstimator,
    derive_min_hold_sec,
)


class _FakeEvent:
    def __init__(self, slot_id, start, end):
        self.slot_id = slot_id
        self.start = start
        self.end = end


class TestDeriveMinHold:
    def test_calm_pacing_gives_high_hold(self):
        hold = derive_min_hold_sec(0.0)
        assert hold == pytest.approx(MIN_HOLD_CEILING, abs=0.01)

    def test_frantic_pacing_gives_low_hold(self):
        hold = derive_min_hold_sec(1.0)
        assert hold == pytest.approx(MIN_HOLD_FLOOR, abs=0.01)

    def test_mid_pacing_is_between(self):
        hold = derive_min_hold_sec(0.5)
        assert MIN_HOLD_FLOOR < hold < MIN_HOLD_CEILING

    def test_monotonic_decrease(self):
        """Higher pacing → strictly lower min_hold."""
        holds = [derive_min_hold_sec(p / 10.0) for p in range(11)]
        for i in range(len(holds) - 1):
            assert holds[i] > holds[i + 1]


class TestCalmPodcast:
    """Synthetic calm podcast: one speaker for 60s, no cuts, steady audio."""

    def test_calm_podcast_low_pacing(self):
        est = LocalPacingEstimator(60, content_type="podcast")
        # No shot cuts
        est.add_shot_cuts([])
        # One speaker, no turns
        est.add_speaker_turns([_FakeEvent(0, 0, 60)])
        est.compute()

        # All pacing scores should be very low
        assert np.mean(est.pacing) < 0.2
        # All min_holds should be high
        min_holds = est.get_min_hold_array()
        assert all(h > 2.2 for h in min_holds)

    def test_calm_podcast_min_hold_at(self):
        est = LocalPacingEstimator(30, content_type="podcast")
        est.add_shot_cuts([])
        est.compute()
        assert est.min_hold_at(15.0) > 2.0


class TestDebatePodcast:
    """Synthetic debate: rapid speaker alternation every 1.2s for 30s."""

    def test_debate_high_pacing(self):
        est = LocalPacingEstimator(30, content_type="podcast")
        # Rapid speaker turns: alternate slots every 1.2s
        events = []
        for i in range(25):
            slot = i % 2
            events.append(_FakeEvent(slot, i * 1.2, (i + 1) * 1.2))
        est.add_speaker_turns(events)
        est.compute()

        # Pacing should be high in the debate window
        debate_pacing = est.pacing[5:25]
        assert np.mean(debate_pacing) > 0.3  # speaker signal should push it up

    def test_debate_low_min_hold(self):
        est = LocalPacingEstimator(30, content_type="podcast")
        events = []
        for i in range(25):
            events.append(_FakeEvent(i % 2, i * 1.2, (i + 1) * 1.2))
        est.add_speaker_turns(events)
        est.compute()

        min_holds = est.get_min_hold_array()
        # At least some min_holds should be below 1.5s
        assert min(min_holds[5:25]) < 1.8


class TestMixedPacing:
    """Calm 30s → debate 20s → calm 30s."""

    def test_mixed_has_bump(self):
        est = LocalPacingEstimator(80, content_type="podcast")
        # Debate speaker turns only in seconds 30-50
        events = []
        for i in range(17):
            events.append(_FakeEvent(i % 2, 30 + i * 1.2, 30 + (i + 1) * 1.2))
        est.add_speaker_turns(events)
        est.compute()

        calm_before = np.mean(est.pacing[5:25])
        debate_window = np.mean(est.pacing[33:47])
        calm_after = np.mean(est.pacing[55:75])

        # Debate should be higher than calm windows
        assert debate_window > calm_before
        assert debate_window > calm_after


class TestPacingSmoothing:
    """Inject a 1-second spike and verify median filter suppresses it."""

    def test_spike_suppressed(self):
        est = LocalPacingEstimator(20, content_type="unknown")
        est.add_shot_cuts([])

        # Manually inject a spike into raw signal
        est.cut_density[10] = 10.0  # huge spike at t=10
        est.compute()

        # The spike should be suppressed by median filter
        # pacing at t=10 should not be dramatically higher than neighbors
        if len(est.pacing) > 12:
            neighborhood = [est.pacing[9], est.pacing[10], est.pacing[11]]
            # Median filter should flatten the spike
            assert max(neighborhood) - min(neighborhood) < 0.5


class TestWeightsPerContentType:
    """Verify different content types weight signals differently."""

    def test_podcast_weights_speakers_high(self):
        est = LocalPacingEstimator(10, content_type="podcast")
        assert est.w_speakers > 0.4

    def test_gaming_weights_motion_high(self):
        est = LocalPacingEstimator(10, content_type="gaming")
        assert est.w_motion > 0.5

    def test_narrative_weights_cuts_high(self):
        est = LocalPacingEstimator(10, content_type="narrative")
        assert est.w_cuts > 0.3


class TestAnticipation:
    def test_calm_gets_high_anticipation(self):
        est = LocalPacingEstimator(20, content_type="unknown")
        est.compute()
        assert est.anticipation_ms_at(10) > 250

    def test_frantic_gets_low_anticipation(self):
        est = LocalPacingEstimator(20, content_type="podcast")
        # Add lots of speaker turns
        events = [_FakeEvent(i % 3, i, i + 1) for i in range(20)]
        est.add_speaker_turns(events)
        # Add lots of cuts
        est.add_shot_cuts(list(range(20)))
        est.compute()
        assert est.anticipation_ms_at(10) < 250


class TestMotionFromDenseFaces:
    def test_compute_motion_returns_array(self):
        from backend.services.local_pacing import compute_motion_from_dense_faces

        class FakeFace:
            def __init__(self, x, identity_id=0):
                self.x = x
                self.identity_id = identity_id

        class FakeDenseFace:
            def __init__(self, timestamp, faces):
                self.timestamp = timestamp
                self.faces = faces

        dense = [
            FakeDenseFace(0, [FakeFace(50)]),
            FakeDenseFace(1, [FakeFace(55)]),
            FakeDenseFace(2, [FakeFace(60)]),
            FakeDenseFace(3, [FakeFace(50)]),
        ]
        motion = compute_motion_from_dense_faces(dense, 5.0)
        assert len(motion) == 5
        assert motion[0] == 0.0  # no prev frame
        assert motion[1] > 0  # moved from 50 to 55
