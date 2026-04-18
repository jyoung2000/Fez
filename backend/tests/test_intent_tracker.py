"""Tests for per-frame intent confidence computation and EMA smoothing."""

import pytest
from dataclasses import dataclass
from typing import Optional

from backend.services.intent_tracker import (
    IntentSignal,
    compute_intent_signal,
    smooth_intent_timeline,
)


# ── Helpers ──

@dataclass
class FakeSpeakerEvent:
    start: float
    end: float
    slot_id: int
    confidence: float = 0.9


@dataclass
class FakeFace:
    identity_id: int
    lip_motion_score: float
    detection_confidence: float = 0.95
    nose_x: float = 50
    nose_y: float = 40
    x: float = 50
    y: float = 40
    width: float = 12
    height: float = 16


@dataclass
class FakeDenseFaces:
    timestamp: float
    faces: list


@dataclass
class FakeSubjectTrack:
    face_slot_id: Optional[int]
    persistent_id: int
    confidence: float
    _bbox_times: list  # [(t, bbox)]

    def bbox_at(self, t):
        for bt, bbox in self._bbox_times:
            if abs(bt - t) < 0.5:
                return bbox
        return None


@dataclass
class FakeSaliencyRegion:
    saliency_score: float = 0.8


# ── Tests: compute_intent_signal ──

class TestComputeIntentSignal:
    def test_empty_inputs(self):
        result = compute_intent_signal(1.0, [], [], None, [])
        assert result == {}

    def test_single_active_speaker(self):
        events = [FakeSpeakerEvent(start=0.0, end=5.0, slot_id=0, confidence=0.9)]
        result = compute_intent_signal(2.0, [], events, None, [])
        assert 0 in result
        assert result[0] == pytest.approx(0.9 * 0.6)

    def test_lip_motion_contributes(self):
        faces = FakeDenseFaces(2.0, [FakeFace(identity_id=1, lip_motion_score=0.8)])
        result = compute_intent_signal(2.0, [], [], faces, [])
        assert 1 in result
        assert result[1] == pytest.approx(0.8 * 0.5 * 0.95)

    def test_speaker_and_lip_combine(self):
        events = [FakeSpeakerEvent(start=0.0, end=5.0, slot_id=0, confidence=0.9)]
        faces = FakeDenseFaces(2.0, [
            FakeFace(identity_id=0, lip_motion_score=0.7),
        ])
        result = compute_intent_signal(2.0, [], events, faces, [])
        expected = 0.9 * 0.6 + 0.7 * 0.5 * 0.95
        assert result[0] == pytest.approx(min(1.0, expected))

    def test_subject_track_presence(self):
        track = FakeSubjectTrack(
            face_slot_id=2, persistent_id=0, confidence=0.8,
            _bbox_times=[(1.0, (30, 30, 10, 15))]
        )
        result = compute_intent_signal(1.0, [track], [], None, [])
        assert 2 in result
        assert result[2] == pytest.approx(0.8 * 0.2)

    def test_saliency_fallback_only_when_no_faces(self):
        saliency = [FakeSaliencyRegion(saliency_score=0.8)]
        result = compute_intent_signal(1.0, [], [], None, saliency)
        assert len(result) == 1
        sid = list(result.keys())[0]
        assert sid < 0  # negative namespace for saliency
        assert result[sid] == pytest.approx(0.8 * 0.4)

    def test_saliency_suppressed_when_faces_present(self):
        events = [FakeSpeakerEvent(start=0.0, end=5.0, slot_id=0)]
        saliency = [FakeSaliencyRegion(saliency_score=0.9)]
        result = compute_intent_signal(2.0, [], events, None, saliency)
        # Saliency should NOT appear because speaker signal exists
        for sid in result:
            assert sid >= 0

    def test_confidence_capped_at_one(self):
        events = [FakeSpeakerEvent(start=0.0, end=5.0, slot_id=0, confidence=1.0)]
        faces = FakeDenseFaces(2.0, [
            FakeFace(identity_id=0, lip_motion_score=1.0, detection_confidence=1.0),
        ])
        result = compute_intent_signal(2.0, [], events, faces, [])
        assert result[0] <= 1.0

    def test_speaker_event_outside_range_ignored(self):
        events = [FakeSpeakerEvent(start=3.0, end=5.0, slot_id=0)]
        result = compute_intent_signal(1.0, [], events, None, [])
        assert result == {}

    def test_face_with_negative_identity_ignored(self):
        faces = FakeDenseFaces(2.0, [FakeFace(identity_id=-1, lip_motion_score=0.9)])
        result = compute_intent_signal(2.0, [], [], faces, [])
        assert result == {}


# ── Tests: smooth_intent_timeline ──

class TestSmoothIntentTimeline:
    def test_empty_input(self):
        assert smooth_intent_timeline([]) == []

    def test_single_frame(self):
        sig = IntentSignal(timestamp=0.0, candidates={0: 0.8})
        result = smooth_intent_timeline([sig], ema_alpha=0.4)
        assert len(result) == 1
        assert 0 in result[0].smoothed
        assert result[0].smoothed[0] == pytest.approx(0.8 * 0.4)

    def test_ema_suppresses_single_frame_spike(self):
        """A single noisy frame should not flip the chosen subject."""
        signals = []
        # 5 frames of slot 0 dominant
        for i in range(5):
            signals.append(IntentSignal(timestamp=i * 0.5, candidates={0: 0.8, 1: 0.1}))
        # 1 frame where slot 1 spikes
        signals.append(IntentSignal(timestamp=2.5, candidates={0: 0.2, 1: 0.9}))
        # Back to slot 0
        signals.append(IntentSignal(timestamp=3.0, candidates={0: 0.8, 1: 0.1}))

        result = smooth_intent_timeline(signals, ema_alpha=0.4)
        # After the spike frame, slot 0 should still have higher smoothed value
        spike_frame = result[5]
        assert spike_frame.smoothed[0] > spike_frame.smoothed[1], \
            f"slot 0 ({spike_frame.smoothed[0]:.3f}) should beat slot 1 ({spike_frame.smoothed[1]:.3f}) after single spike"

    def test_sustained_takeover(self):
        """3+ consecutive frames with new subject should overtake."""
        signals = []
        # 3 frames of slot 0
        for i in range(3):
            signals.append(IntentSignal(timestamp=i * 0.5, candidates={0: 0.8, 1: 0.1}))
        # 4 frames of slot 1 dominant
        for i in range(4):
            signals.append(IntentSignal(timestamp=1.5 + i * 0.5, candidates={0: 0.1, 1: 0.8}))

        result = smooth_intent_timeline(signals, ema_alpha=0.4)
        last = result[-1]
        assert last.smoothed[1] > last.smoothed[0], \
            f"slot 1 ({last.smoothed[1]:.3f}) should overtake slot 0 ({last.smoothed[0]:.3f}) after sustained frames"

    def test_decay_absent_subjects(self):
        """Subjects not present in a frame should decay."""
        signals = [
            IntentSignal(timestamp=0.0, candidates={0: 0.8}),
            IntentSignal(timestamp=0.5, candidates={1: 0.8}),
            IntentSignal(timestamp=1.0, candidates={1: 0.8}),
        ]
        result = smooth_intent_timeline(signals, ema_alpha=0.4)
        # Slot 0 should decay in frames 1 and 2
        assert result[2].smoothed.get(0, 0) < result[0].smoothed[0]

    def test_near_zero_entries_dropped(self):
        """Entries below 0.05 threshold should be dropped."""
        signals = [
            IntentSignal(timestamp=0.0, candidates={0: 0.1}),
        ]
        # With alpha=0.4, first frame: 0.0 * 0.6 + 0.1 * 0.4 = 0.04 < 0.05
        result = smooth_intent_timeline(signals, ema_alpha=0.4)
        assert 0 not in result[0].smoothed
