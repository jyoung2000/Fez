"""Phase 1 — Red test: diarization should not be capped by face registry slot count.

Bug D: The face-aware diarization code in transcription.py cannot produce
more speakers than the face registry has slots. If the face registry says 4
(due to Bugs A/B/C), audio diarization is forced to match even when Whisper
+ pyannote heard 6 distinct voices.

This test verifies that assign_speakers_with_face_data can produce more
unique speaker labels than face_registry.slots when the audio data clearly
indicates more speakers.
"""

from dataclasses import dataclass, field
from typing import Optional

import pytest

from backend.services.transcription import assign_speakers_with_face_data


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
class _FaceInfo:
    identity_id: int
    nose_x: float
    nose_y: float = 50.0
    x_center: float = 50.0
    y_center: float = 50.0
    width: float = 10.0
    height: float = 12.0
    lip_aperture: float = 0.0
    is_speaking: bool = False
    confidence: float = 0.9
    identity_embedding: Optional[list] = None
    is_human: bool = True
    y_bottom: float = 0.0
    pose_confidence: float = 0.0


@dataclass
class _FrameFaces:
    timestamp: float
    frame_path: str = ""
    faces: list = field(default_factory=list)
    primary_face_idx: int = -1


class TestDiarizationNotCappedByRegistry:
    """Audio diarization should be able to report more speakers than face slots."""

    def test_six_audio_speakers_with_four_face_slots(self):
        """Face registry has 4 slots but audio clearly has 6 speakers.

        The face registry found only 4 slots (due to Bugs A/B/C), but the
        audio timing indicates 6 distinct speakers (each speaking at
        non-overlapping times with different lip aperture patterns).

        After reconciliation, the transcript should have >=5 unique speaker
        labels, not be capped at 4.
        """
        # Face registry with 4 slots
        registry = _FaceRegistry(slots=[
            _FaceSlot(slot_id=0, x_center=15.0),
            _FaceSlot(slot_id=1, x_center=35.0),
            _FaceSlot(slot_id=2, x_center=65.0),
            _FaceSlot(slot_id=3, x_center=85.0),
        ])

        # Dense face data: 4 faces visible in frames.
        # All faces have lip_aperture=0 (not speaking) — only the active
        # speaker events or audio heuristics determine who speaks.
        face_results = []
        for i in range(60):
            t = i * 1.0
            faces = [
                _FaceInfo(identity_id=0, nose_x=15, lip_aperture=0.0),
                _FaceInfo(identity_id=1, nose_x=35, lip_aperture=0.0),
                _FaceInfo(identity_id=2, nose_x=65, lip_aperture=0.0),
                _FaceInfo(identity_id=3, nose_x=85, lip_aperture=0.0),
            ]
            face_results.append(_FrameFaces(timestamp=t, faces=faces))

        # Raw transcript segments with 6 distinct audio speakers.
        # Speakers 5 and 6 are off-camera (no face slot).
        # Large gaps (>= 1.2s) between segments signal speaker changes.
        # No face has lip_aperture > 0, so the function relies on audio
        # heuristics (gap detection) and falls through to virtual speakers.
        raw_segments = []
        segments_data = [
            # (start, end, speaker_label)
            (0.0, 5.0, "Speaker 1"),     # no lip data -> slot -1, but first seg
            (6.5, 10.0, "Speaker 2"),    # gap=1.5s, no lip -> virtual
            (11.5, 16.0, "Speaker 3"),   # gap=1.5s, no lip -> virtual
            (17.5, 22.0, "Speaker 4"),   # gap=1.5s, no lip -> virtual
            (24.0, 28.0, "Speaker 5"),   # gap=2.0s, no lip -> virtual
            (30.0, 34.0, "Speaker 6"),   # gap=2.0s, no lip -> virtual
            (36.0, 40.0, "Speaker 1"),   # gap=2.0s, back
            (42.0, 46.0, "Speaker 3"),   # gap=2.0s
            (48.0, 52.0, "Speaker 5"),   # gap=2.0s
            (54.0, 58.0, "Speaker 2"),   # gap=2.0s
        ]
        for start, end, spk in segments_data:
            raw_segments.append({
                "start": start,
                "end": end,
                "text": f"Hello from {spk}",
                "words": None,
                "confidence": 0.9,
                "avg_logprob": -0.3,
                "no_speech_prob": 0.1,
            })

        result = assign_speakers_with_face_data(
            raw_segments=raw_segments,
            face_results=face_results,
            face_registry=registry,
        )

        # Count unique speakers in the result
        unique_speakers = set()
        for seg in result:
            spk = seg.speaker if hasattr(seg, 'speaker') else seg.get('speaker', '')
            if spk:
                unique_speakers.add(spk)

        # Currently the code caps at face_registry.slots count (4).
        # It should allow >= 5 unique speakers when audio indicates more.
        # Note: exact count depends on the audio-assignment heuristics,
        # but it must exceed the face slot count.
        assert len(unique_speakers) >= 5, (
            f"Expected >= 5 unique speakers (6 audio speakers, 4 face slots), "
            f"got {len(unique_speakers)}: {sorted(unique_speakers)}. "
            f"Audio diarization was likely capped by face registry slot count."
        )
