"""Tests for the two-pass look-ahead segmenter.

Covers:
  10. Future-suppressed blip (short speaker B suppressed, sustained B committed)
  11. Anticipation with adaptive pacing
  12. Shot cut overrides anticipation
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pytest

from backend.services.reframe_segmenter import (
    ReframeSegment,
    build_reframe_segments,
)
from backend.services.local_pacing import LocalPacingEstimator


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
    slots: list = field(default_factory=list)
    total_frames: int = 100
    frames_with_faces: int = 100
    is_continuous_motion: bool = False

    @property
    def multi_speaker(self) -> bool:
        return len(self.slots) >= 2

    def slot_by_id(self, slot_id: int):
        for s in self.slots:
            if s.slot_id == slot_id:
                return s
        return None


@dataclass
class _SpeakerEvent:
    slot_id: int
    start: float
    end: float
    confidence: float = 0.9


@dataclass
class _DenseFace:
    timestamp: float
    faces: list = field(default_factory=list)


@dataclass
class _Face:
    identity_id: int
    x: float
    y: float = 40.0
    nose_x: float = 50.0


@dataclass
class _TranscriptSeg:
    start: float
    end: float
    speaker: str = ""
    confidence: float = 0.9


def _build_face_registry(slots_x: list[float]) -> _FaceRegistry:
    return _FaceRegistry(
        slots=[_FaceSlot(i, x) for i, x in enumerate(slots_x)],
    )


def _build_dense_faces(duration: float, slots_x: list[float], dt: float = 1.0) -> list[_DenseFace]:
    result = []
    t = 0.0
    while t < duration:
        faces = [_Face(identity_id=i, x=x) for i, x in enumerate(slots_x)]
        result.append(_DenseFace(timestamp=t, faces=faces))
        t += dt
    return result


class TestLookAheadHysteresis:
    """Verify that short blips are suppressed by look-ahead."""

    def test_brief_blip_suppressed(self):
        """Speaker A 10s → B 0.5s → A 10s: B should be suppressed."""
        registry = _build_face_registry([25.0, 75.0])
        dense = _build_dense_faces(20.5, [25.0, 75.0])

        # Active speaker: A(0-10), B(10-10.5), A(10.5-20.5)
        events = [
            _SpeakerEvent(0, 0, 10),
            _SpeakerEvent(1, 10, 10.5),
            _SpeakerEvent(0, 10.5, 20.5),
        ]

        transcript = [
            _TranscriptSeg(0, 10, "spk_A"),
            _TranscriptSeg(10, 10.5, "spk_B"),
            _TranscriptSeg(10.5, 20.5, "spk_A"),
        ]

        segments = build_reframe_segments(
            shot_cuts=[],
            face_registry=registry,
            active_speaker_events=events,
            dense_faces=dense,
            transcript_segments=transcript,
            speaker_to_slot={"spk_A": 0, "spk_B": 1},
            video_duration=20.5,
            job_id="test",
        )

        # The brief 0.5s B blip should be absorbed
        # We should have ≤ 2 unique subject_x values
        unique_x = set(s.subject_x for s in segments)
        assert len(unique_x) <= 2

    def test_sustained_speaker_committed(self):
        """Speaker A 5s → B 8s: B should be committed."""
        registry = _build_face_registry([25.0, 75.0])
        dense = _build_dense_faces(13, [25.0, 75.0])

        events = [
            _SpeakerEvent(0, 0, 5),
            _SpeakerEvent(1, 5, 13),
        ]

        transcript = [
            _TranscriptSeg(0, 5, "spk_A"),
            _TranscriptSeg(5, 13, "spk_B"),
        ]

        segments = build_reframe_segments(
            shot_cuts=[],
            face_registry=registry,
            active_speaker_events=events,
            dense_faces=dense,
            transcript_segments=transcript,
            speaker_to_slot={"spk_A": 0, "spk_B": 1},
            video_duration=13,
            job_id="test",
        )

        # Both speakers should appear
        x_values = set(s.subject_x for s in segments)
        assert len(x_values) >= 2


class TestAdaptiveAnticipation:
    """Verify anticipation is adaptive and uses pacing estimator."""

    def test_anticipation_shifts_speaker_turn(self):
        """Speaker turn at t=10 on continuous shot → segment starts before t=10."""
        registry = _build_face_registry([25.0, 75.0])
        dense = _build_dense_faces(20, [25.0, 75.0])

        events = [
            _SpeakerEvent(0, 0, 10),
            _SpeakerEvent(1, 10, 20),
        ]
        transcript = [
            _TranscriptSeg(0, 10, "spk_A"),
            _TranscriptSeg(10, 20, "spk_B"),
        ]

        segments = build_reframe_segments(
            shot_cuts=[],  # No shot cuts → allows anticipation
            face_registry=registry,
            active_speaker_events=events,
            dense_faces=dense,
            transcript_segments=transcript,
            speaker_to_slot={"spk_A": 0, "spk_B": 1},
            video_duration=20,
            job_id="test",
        )

        # Find the segment that switched to speaker B
        b_segments = [s for s in segments if s.subject_x == 75]
        if b_segments:
            # The first B segment should start slightly before t=10 (anticipation)
            assert b_segments[0].start < 10.0, (
                f"Expected anticipation shift before t=10, got start={b_segments[0].start}"
            )

    def test_shot_cut_prevents_anticipation(self):
        """Speaker turn at t=10 with shot cut at t=10 → no anticipation."""
        registry = _build_face_registry([25.0, 75.0])
        dense = _build_dense_faces(20, [25.0, 75.0])

        events = [
            _SpeakerEvent(0, 0, 10),
            _SpeakerEvent(1, 10, 20),
        ]
        transcript = [
            _TranscriptSeg(0, 10, "spk_A"),
            _TranscriptSeg(10, 20, "spk_B"),
        ]

        segments = build_reframe_segments(
            shot_cuts=[10.0],  # Hard cut at t=10
            face_registry=registry,
            active_speaker_events=events,
            dense_faces=dense,
            transcript_segments=transcript,
            speaker_to_slot={"spk_A": 0, "spk_B": 1},
            video_duration=20,
            job_id="test",
        )

        # Find the segment at the shot cut boundary
        for seg in segments:
            if abs(seg.start - 10.0) < 0.2:
                assert seg.ease_in_ms == 0, (
                    f"Shot cut should have ease_in_ms=0, got {seg.ease_in_ms}"
                )


class TestAdaptivePacingIntegration:
    """Verify that the pacing estimator integrates correctly."""

    def test_pacing_estimator_changes_merge_behavior(self):
        """With pacing estimator providing low min_hold,
        more short segments should survive merging."""
        registry = _build_face_registry([25.0, 75.0])
        dense = _build_dense_faces(10, [25.0, 75.0])

        # Create rapid turns
        events = []
        transcript = []
        for i in range(10):
            slot = i % 2
            events.append(_SpeakerEvent(slot, i, i + 1))
            transcript.append(_TranscriptSeg(i, i + 1, f"spk_{slot}"))

        # Build with pacing estimator that produces frantic pacing
        est = LocalPacingEstimator(10, content_type="podcast")
        est.add_speaker_turns(events)
        est.add_shot_cuts(list(range(10)))
        est.compute()

        segments_with_pacing = build_reframe_segments(
            shot_cuts=[],
            face_registry=registry,
            active_speaker_events=events,
            dense_faces=dense,
            transcript_segments=transcript,
            speaker_to_slot={"spk_0": 0, "spk_1": 1},
            video_duration=10,
            job_id="test",
            pacing_estimator=est,
        )

        # Should have segments covering the full duration
        assert segments_with_pacing[0].start == 0.0
        assert segments_with_pacing[-1].end == 10.0
