"""Phase 1 — Failing test: zero crop segment overlap.

Half-open interval semantics [start, end) must hold for every segment pair.
No two segments may overlap, and the timeline must be gap-free and cover
[0, video_duration] exactly.

Tests run against every fixture pattern from the existing test suite.
"""

from dataclasses import dataclass, field
from typing import Optional

import pytest

from backend.services.reframe_segmenter import build_reframe_segments


# ── Lightweight stubs ──

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
    lip_aperture: float = 0.0
    is_speaking: bool = False
    confidence: float = 0.9


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
    confidence: Optional[float] = 0.9
    words: Optional[list] = None
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None


def _make_dense(slot_id, x, start, end, step=0.5):
    frames = []
    t = start
    while t < end:
        frames.append(_FrameFaces(
            timestamp=round(t, 4),
            faces=[_FaceInfo(identity_id=slot_id, nose_x=x)],
        ))
        t += step
    return frames


def _assert_no_overlap(segments, video_duration: float, label: str = ""):
    """Assert half-open interval semantics and full coverage."""
    assert len(segments) >= 1, f"{label}: expected at least 1 segment"

    # Sort by start time
    sorted_segs = sorted(segments, key=lambda s: s.start)

    # Check first segment starts at 0
    assert sorted_segs[0].start == pytest.approx(0.0, abs=1e-6), (
        f"{label}: first segment starts at {sorted_segs[0].start}, expected 0.0"
    )

    # Check last segment ends at video_duration
    assert sorted_segs[-1].end == pytest.approx(video_duration, abs=1e-6), (
        f"{label}: last segment ends at {sorted_segs[-1].end}, "
        f"expected {video_duration}"
    )

    # Check contiguity: each seg[i].end == seg[i+1].start exactly
    for i in range(len(sorted_segs) - 1):
        a = sorted_segs[i]
        b = sorted_segs[i + 1]

        # No overlap
        assert a.end <= b.start + 1e-9, (
            f"{label}: overlap between segment {i} (end={a.end}) "
            f"and segment {i+1} (start={b.start})"
        )

        # No gap (contiguous)
        assert a.end == pytest.approx(b.start, abs=1e-9), (
            f"{label}: gap between segment {i} (end={a.end}) "
            f"and segment {i+1} (start={b.start})"
        )


class TestNoCropOverlap:
    """Segment intervals must be contiguous and non-overlapping for all fixtures."""

    def test_single_speaker_no_overlap(self):
        registry = _FaceRegistry(slots=[_FaceSlot(slot_id=0, x_center=56.0)])
        dense = _make_dense(0, 56, 0, 30, step=0.5)
        transcript = [_TranscriptSeg(start=0, end=30, speaker="Speaker 1")]
        speaker_to_slot = {"Speaker 1": 0}

        segments = build_reframe_segments(
            shot_cuts=[],
            face_registry=registry,
            active_speaker_events=[],
            dense_faces=dense,
            transcript_segments=transcript,
            speaker_to_slot=speaker_to_slot,
            video_duration=30.0,
        )
        _assert_no_overlap(segments, 30.0, "single_speaker")

    def test_two_speakers_alternating_no_overlap(self):
        registry = _FaceRegistry(slots=[
            _FaceSlot(slot_id=0, x_center=30.0),
            _FaceSlot(slot_id=1, x_center=70.0),
        ])
        speaker_to_slot = {"Speaker 1": 0, "Speaker 2": 1}

        transcript = []
        as_events = []
        dense = []
        for i in range(10):
            t_start = i * 3.0
            t_end = (i + 1) * 3.0
            spk = "Speaker 1" if i % 2 == 0 else "Speaker 2"
            slot = 0 if i % 2 == 0 else 1
            x = 30 if slot == 0 else 70
            transcript.append(_TranscriptSeg(start=t_start, end=t_end, speaker=spk))
            as_events.append(_SpeakerEvent(start=t_start, end=t_end, slot_id=slot, confidence=0.9))
            dense.extend(_make_dense(slot, x, t_start, t_end))

        segments = build_reframe_segments(
            shot_cuts=[],
            face_registry=registry,
            active_speaker_events=as_events,
            dense_faces=dense,
            transcript_segments=transcript,
            speaker_to_slot=speaker_to_slot,
            video_duration=30.0,
        )
        _assert_no_overlap(segments, 30.0, "two_speakers_alternating")

    def test_shot_cuts_with_speaker_changes_no_overlap(self):
        registry = _FaceRegistry(slots=[
            _FaceSlot(slot_id=0, x_center=30.0),
            _FaceSlot(slot_id=1, x_center=70.0),
        ])
        speaker_to_slot = {"Speaker 1": 0, "Speaker 2": 1}

        transcript = [
            _TranscriptSeg(start=0, end=5, speaker="Speaker 1"),
            _TranscriptSeg(start=5.05, end=15, speaker="Speaker 2"),
        ]
        as_events = [
            _SpeakerEvent(start=0, end=5, slot_id=0, confidence=0.9),
            _SpeakerEvent(start=5.05, end=15, slot_id=1, confidence=0.9),
        ]
        dense = _make_dense(0, 30, 0, 5) + _make_dense(1, 70, 5, 15)

        segments = build_reframe_segments(
            shot_cuts=[5.0],
            face_registry=registry,
            active_speaker_events=as_events,
            dense_faces=dense,
            transcript_segments=transcript,
            speaker_to_slot=speaker_to_slot,
            video_duration=15.0,
        )
        _assert_no_overlap(segments, 15.0, "shot_cuts_with_speaker_changes")

    def test_rapid_switches_with_anticipation_no_overlap(self):
        """Anticipation offsets on speaker turns must not create overlaps."""
        registry = _FaceRegistry(slots=[
            _FaceSlot(slot_id=0, x_center=25.0),
            _FaceSlot(slot_id=1, x_center=75.0),
        ])
        speaker_to_slot = {"Speaker 1": 0, "Speaker 2": 1}

        transcript = []
        as_events = []
        dense = []
        # 6 rapid speaker switches every 1 second
        for i in range(6):
            t_start = i * 1.0
            t_end = (i + 1) * 1.0
            spk = "Speaker 1" if i % 2 == 0 else "Speaker 2"
            slot = 0 if i % 2 == 0 else 1
            x = 25 if slot == 0 else 75
            transcript.append(_TranscriptSeg(start=t_start, end=t_end, speaker=spk, confidence=0.95))
            as_events.append(_SpeakerEvent(start=t_start, end=t_end, slot_id=slot, confidence=0.95))
            dense.extend(_make_dense(slot, x, t_start, t_end, step=0.1))

        segments = build_reframe_segments(
            shot_cuts=[],
            face_registry=registry,
            active_speaker_events=as_events,
            dense_faces=dense,
            transcript_segments=transcript,
            speaker_to_slot=speaker_to_slot,
            video_duration=6.0,
        )
        _assert_no_overlap(segments, 6.0, "rapid_switches_with_anticipation")

    def test_many_shot_cuts_no_overlap(self):
        """A clip with many shot cuts must produce contiguous segments."""
        registry = _FaceRegistry(slots=[_FaceSlot(slot_id=0, x_center=50.0)])
        dense = _make_dense(0, 50, 0, 20, step=0.5)
        transcript = [_TranscriptSeg(start=0, end=20, speaker="Speaker 1")]
        speaker_to_slot = {"Speaker 1": 0}

        # Shot cuts every 2 seconds
        shot_cuts = [float(i) for i in range(2, 20, 2)]

        segments = build_reframe_segments(
            shot_cuts=shot_cuts,
            face_registry=registry,
            active_speaker_events=[],
            dense_faces=dense,
            transcript_segments=transcript,
            speaker_to_slot=speaker_to_slot,
            video_duration=20.0,
        )
        _assert_no_overlap(segments, 20.0, "many_shot_cuts")
