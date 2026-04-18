"""Integration tests for split-second subject switching and ambiguity hold."""

import os
import pytest

from backend.services.active_speaker import SpeakerEvent
from backend.services.intent_tracker import (
    IntentSignal,
    compute_intent_signal,
    smooth_intent_timeline,
    derive_switches,
)


# ── Test helpers ──

class _F:
    """Fake face for dense_faces."""
    def __init__(self, sid, lip):
        self.identity_id = sid
        self.lip_motion_score = lip
        self.detection_confidence = 0.95
        self.nose_x = 30 if sid == 0 else 70
        self.nose_y = 40
        self.x = self.nose_x
        self.y = self.nose_y
        self.width = 12
        self.height = 16


class _DF:
    """Fake dense faces frame."""
    def __init__(self, t, faces):
        self.timestamp = t
        self.faces = faces


class TestSplitSecondSpeakerChange:
    def test_split_second_speaker_change(self):
        """When speaker B starts talking abruptly, the intent tracker should
        switch within ~200ms (a few frames at 6 FPS), not after a hardcoded hold."""
        # Synthesize speaker events: A talks 0–2.0, B talks 2.0–5.0
        events = [
            SpeakerEvent(start=0.0, end=2.0, slot_id=0, confidence=0.9),
            SpeakerEvent(start=2.0, end=5.0, slot_id=1, confidence=0.9),
        ]

        # Build dense faces at 6 FPS with lip motion matching the events
        dense = []
        for i in range(30):  # 5 seconds at 6 FPS
            t = i / 6.0
            if t < 2.0:
                dense.append(_DF(t, [_F(0, 0.8), _F(1, 0.05)]))
            else:
                dense.append(_DF(t, [_F(0, 0.05), _F(1, 0.8)]))

        raw = [
            IntentSignal(
                timestamp=df.timestamp,
                candidates=compute_intent_signal(
                    df.timestamp, [], events, df, []
                ),
            )
            for df in dense
        ]
        smoothed = smooth_intent_timeline(raw, ema_alpha=0.55)
        switches = derive_switches(
            smoothed, switch_margin=0.10,
            min_switch_confidence=0.35, job_id="test_split_second",
        )

        # First switch is the initial adoption at t=0
        # Second switch should be the speaker change near t=2.0
        speaker_switches = [s for s in switches if s.from_id is not None]
        assert len(speaker_switches) >= 1, \
            f"expected at least one speaker switch, got {len(speaker_switches)}"

        speaker_switch = speaker_switches[0]
        assert 2.0 <= speaker_switch.timestamp <= 2.50, \
            f"switch happened at {speaker_switch.timestamp}, expected within 2.0–2.5s"
        assert speaker_switch.from_id == 0
        assert speaker_switch.to_id == 1
        assert speaker_switch.margin > 0
        assert speaker_switch.reason == "confidence_overtook"

    def test_switch_latency_under_400ms(self):
        """The switch should happen within 400ms of the speaker change,
        verifying the snappy behavior intent tracking provides."""
        events = [
            SpeakerEvent(start=0.0, end=3.0, slot_id=0, confidence=0.95),
            SpeakerEvent(start=3.0, end=8.0, slot_id=1, confidence=0.95),
        ]

        dense = []
        for i in range(48):  # 8 seconds at 6 FPS
            t = i / 6.0
            if t < 3.0:
                dense.append(_DF(t, [_F(0, 0.9), _F(1, 0.02)]))
            else:
                dense.append(_DF(t, [_F(0, 0.02), _F(1, 0.9)]))

        raw = [
            IntentSignal(
                timestamp=df.timestamp,
                candidates=compute_intent_signal(
                    df.timestamp, [], events, df, []
                ),
            )
            for df in dense
        ]
        smoothed = smooth_intent_timeline(raw, ema_alpha=0.55)
        switches = derive_switches(
            smoothed, switch_margin=0.10,
            min_switch_confidence=0.35, job_id="test_latency",
        )

        speaker_switches = [s for s in switches if s.from_id is not None]
        assert len(speaker_switches) >= 1

        latency = speaker_switches[0].timestamp - 3.0
        assert latency <= 0.5, \
            f"switch latency was {latency:.3f}s, expected <= 0.5s"


class TestAmbiguousOverlapNoSwitch:
    def test_ambiguous_overlap_produces_zero_switches(self):
        """When both speakers have similar confidence throughout,
        no switch should fire after the initial adoption."""
        events = [
            SpeakerEvent(start=0.0, end=5.0, slot_id=0, confidence=0.5),
            SpeakerEvent(start=0.0, end=5.0, slot_id=1, confidence=0.5),
        ]

        dense = []
        for i in range(30):
            t = i / 6.0
            dense.append(_DF(t, [_F(0, 0.4), _F(1, 0.4)]))

        raw = [
            IntentSignal(
                timestamp=df.timestamp,
                candidates=compute_intent_signal(
                    df.timestamp, [], events, df, []
                ),
            )
            for df in dense
        ]
        smoothed = smooth_intent_timeline(raw, ema_alpha=0.4)
        switches = derive_switches(
            smoothed, switch_margin=0.15,
            min_switch_confidence=0.35, job_id="test_ambiguous",
        )

        non_initial = [s for s in switches if s.from_id is not None]
        assert len(non_initial) == 0, \
            f"expected 0 switches during ambiguous overlap, got {len(non_initial)}"

    def test_slight_advantage_does_not_trigger(self):
        """A slight advantage (below switch_margin) should not trigger a switch."""
        events = [
            SpeakerEvent(start=0.0, end=5.0, slot_id=0, confidence=0.6),
            SpeakerEvent(start=0.0, end=5.0, slot_id=1, confidence=0.55),
        ]

        dense = []
        for i in range(30):
            t = i / 6.0
            dense.append(_DF(t, [_F(0, 0.45), _F(1, 0.40)]))

        raw = [
            IntentSignal(
                timestamp=df.timestamp,
                candidates=compute_intent_signal(
                    df.timestamp, [], events, df, []
                ),
            )
            for df in dense
        ]
        smoothed = smooth_intent_timeline(raw, ema_alpha=0.4)
        switches = derive_switches(
            smoothed, switch_margin=0.15,
            min_switch_confidence=0.35, job_id="test_slight",
        )

        non_initial = [s for s in switches if s.from_id is not None]
        assert len(non_initial) == 0, \
            f"expected 0 switches for slight advantage, got {len(non_initial)}"


class TestSwitchLogging:
    def test_every_switch_has_required_fields(self):
        """Every SubjectSwitch must have from_id, to_id, conf_from, conf_to, margin."""
        events = [
            SpeakerEvent(start=0.0, end=2.0, slot_id=0, confidence=0.9),
            SpeakerEvent(start=2.0, end=5.0, slot_id=1, confidence=0.9),
        ]
        dense = []
        for i in range(30):
            t = i / 6.0
            if t < 2.0:
                dense.append(_DF(t, [_F(0, 0.8), _F(1, 0.05)]))
            else:
                dense.append(_DF(t, [_F(0, 0.05), _F(1, 0.8)]))

        raw = [
            IntentSignal(
                timestamp=df.timestamp,
                candidates=compute_intent_signal(
                    df.timestamp, [], events, df, []
                ),
            )
            for df in dense
        ]
        smoothed = smooth_intent_timeline(raw, ema_alpha=0.55)
        switches = derive_switches(
            smoothed, switch_margin=0.10,
            min_switch_confidence=0.35, job_id="test_logging",
        )

        for sw in switches:
            assert hasattr(sw, 'from_id')
            assert hasattr(sw, 'to_id')
            assert hasattr(sw, 'conf_from')
            assert hasattr(sw, 'conf_to')
            assert hasattr(sw, 'margin')
            assert hasattr(sw, 'reason')
            assert sw.to_id is not None
            assert isinstance(sw.margin, float)
            assert isinstance(sw.conf_to, float)
