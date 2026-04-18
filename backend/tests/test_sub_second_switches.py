"""Phase 1 — Failing test: sub-second speaker switches must survive.

A human editor would cut on a genuine 200-400ms speaker change backed by
high lip-motion confidence.  The current pipeline deletes every sub-400ms
segment via MIN_HOLD_SECONDS=0.4 and the Stage 4 look-ahead hysteresis.

This test synthesises 5 speaker switches in 2 seconds (400ms apart), each
with high confidence, and asserts that ALL 5 switches appear in the output.
"""

from dataclasses import dataclass, field
from typing import Optional

import pytest

from backend.services.reframe_segmenter import build_reframe_segments


# ── Lightweight stubs (same pattern as test_reframe_segmenter.py) ──

@dataclass
class _FaceSlot:
    slot_id: int
    x_center: float
    x_min: float = 0.0
    x_max: float = 100.0
    frame_count: int = 100
    avg_width: float = 10.0
    avg_height: float = 12.0


@dataclass
class _FaceRegistry:
    slots: list[_FaceSlot] = field(default_factory=list)
    total_frames: int = 100
    frames_with_faces: int = 100

    @property
    def multi_speaker(self) -> bool:
        return len(self.slots) >= 2

    def slot_by_id(self, slot_id: int):
        for s in self.slots:
            if s.slot_id == slot_id:
                return s
        return None

    def nearest_slot(self, x: float):
        if not self.slots:
            return None
        return min(self.slots, key=lambda s: abs(s.x_center - x))


@dataclass
class _SpeakerEvent:
    start: float
    end: float
    slot_id: int
    confidence: float


@dataclass
class _FaceInfo:
    identity_id: int
    nose_x: float
    nose_y: float = 50.0
    width: float = 10.0
    height: float = 12.0
    lip_aperture: float = 0.05
    is_speaking: bool = True
    confidence: float = 0.95


@dataclass
class _FrameFaces:
    timestamp: float
    frame_path: str = ""
    faces: list[_FaceInfo] = field(default_factory=list)
    primary_face_idx: int = -1


@dataclass
class _TranscriptSeg:
    start: float
    end: float
    text: str = ""
    speaker: str = "Speaker 1"
    confidence: Optional[float] = 0.95
    words: Optional[list] = None
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None


def _make_dense(slot_id, x, start, end, step=0.1):
    """Dense face frames at high temporal resolution."""
    frames = []
    t = start
    while t < end:
        frames.append(_FrameFaces(
            timestamp=round(t, 4),
            faces=[_FaceInfo(identity_id=slot_id, nose_x=x)],
        ))
        t += step
    return frames


class TestSubSecondSwitchesSurvive:
    """5 speaker switches in 2s (400ms each) with high confidence must all survive."""

    def test_five_rapid_switches_preserved(self):
        registry = _FaceRegistry(slots=[
            _FaceSlot(slot_id=0, x_center=25.0),
            _FaceSlot(slot_id=1, x_center=75.0),
        ])
        speaker_to_slot = {"Speaker 1": 0, "Speaker 2": 1}

        # 5 switches: A(0-0.4) B(0.4-0.8) A(0.8-1.2) B(1.2-1.6) A(1.6-2.0)
        transcript = []
        as_events = []
        dense = []
        switches = [
            (0.0,  0.4, "Speaker 1", 0, 25.0),
            (0.4,  0.8, "Speaker 2", 1, 75.0),
            (0.8,  1.2, "Speaker 1", 0, 25.0),
            (1.2,  1.6, "Speaker 2", 1, 75.0),
            (1.6,  2.0, "Speaker 1", 0, 25.0),
        ]

        for start, end, spk, slot, x in switches:
            transcript.append(_TranscriptSeg(
                start=start, end=end, speaker=spk, confidence=0.95,
            ))
            as_events.append(_SpeakerEvent(
                start=start, end=end, slot_id=slot, confidence=0.95,
            ))
            dense.extend(_make_dense(slot, x, start, end, step=0.05))

        segments = build_reframe_segments(
            shot_cuts=[],
            face_registry=registry,
            active_speaker_events=as_events,
            dense_faces=dense,
            transcript_segments=transcript,
            speaker_to_slot=speaker_to_slot,
            video_duration=2.0,
        )

        # All 5 switch segments must survive — no merging of high-confidence short segments
        switch_count = 0
        seen_slots = []
        for seg in segments:
            seen_slots.append(seg.active_slot)
            if len(seen_slots) >= 2 and seen_slots[-1] != seen_slots[-2]:
                switch_count += 1

        assert switch_count >= 4, (
            f"Expected at least 4 speaker switches (5 segments alternating), "
            f"but got {switch_count} switches in {len(segments)} segments. "
            f"Slots: {seen_slots}"
        )

    def test_200ms_high_confidence_switch_preserved(self):
        """A single 200ms speaker B between two long A segments must survive
        when confidence is high."""
        registry = _FaceRegistry(slots=[
            _FaceSlot(slot_id=0, x_center=25.0),
            _FaceSlot(slot_id=1, x_center=75.0),
        ])
        speaker_to_slot = {"Speaker 1": 0, "Speaker 2": 1}

        transcript = [
            _TranscriptSeg(start=0, end=5, speaker="Speaker 1", confidence=0.95),
            _TranscriptSeg(start=5, end=5.2, speaker="Speaker 2", confidence=0.95),
            _TranscriptSeg(start=5.2, end=10, speaker="Speaker 1", confidence=0.95),
        ]
        as_events = [
            _SpeakerEvent(start=0, end=5, slot_id=0, confidence=0.95),
            _SpeakerEvent(start=5, end=5.2, slot_id=1, confidence=0.95),
            _SpeakerEvent(start=5.2, end=10, slot_id=0, confidence=0.95),
        ]
        dense = (
            _make_dense(0, 25, 0, 5)
            + _make_dense(1, 75, 5, 5.2, step=0.05)
            + _make_dense(0, 25, 5.2, 10)
        )

        segments = build_reframe_segments(
            shot_cuts=[],
            face_registry=registry,
            active_speaker_events=as_events,
            dense_faces=dense,
            transcript_segments=transcript,
            speaker_to_slot=speaker_to_slot,
            video_duration=10.0,
        )

        # The 200ms B segment must survive because confidence is high
        slot_sequence = [seg.active_slot for seg in segments]
        assert 1 in slot_sequence, (
            f"High-confidence 200ms speaker B was deleted. "
            f"Slots: {slot_sequence}"
        )
