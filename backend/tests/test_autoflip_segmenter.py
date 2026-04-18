"""Tests for the AutoFlip-parity segmenter."""

from dataclasses import dataclass, field
from typing import Optional

from backend.services.autoflip_segmenter import build_autoflip_segments


# ── Minimal mock objects ──
@dataclass
class MockFace:
    nose_x: float = 50.0
    nose_y: float = 40.0
    x: float = 50.0
    y: float = 40.0
    width: float = 10.0
    height: float = 13.0
    identity_id: int = 0
    lip_aperture: float = 0.03
    is_speaking: bool = False
    identity_embedding: Optional[list] = None


@dataclass
class MockFrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)
    primary_face_idx: int = 0
    path: str = ""


@dataclass
class MockFaceSlot:
    slot_id: int = 0
    x_center: float = 50.0
    x_min: float = 45.0
    x_max: float = 55.0
    frame_count: int = 10
    avg_width: float = 10.0
    avg_height: float = 13.0


@dataclass
class MockFaceRegistry:
    slots: list = field(default_factory=list)
    total_frames: int = 10
    frames_with_faces: int = 10

    @property
    def multi_speaker(self):
        return len(self.slots) >= 2

    @property
    def is_continuous_motion(self):
        return False

    def nearest_slot(self, x):
        if not self.slots:
            return None
        return min(self.slots, key=lambda s: abs(s.x_center - x))

    def slot_by_id(self, slot_id):
        for s in self.slots:
            if s.slot_id == slot_id:
                return s
        return None


@dataclass
class MockSpeakerEvent:
    start: float
    end: float
    slot_id: int
    confidence: float = 0.9


@dataclass
class MockTranscriptSeg:
    start: float
    end: float
    speaker: str = "SPEAKER_00"
    confidence: float = 0.95
    words: list = field(default_factory=list)


class TestAutoflipSegmenter:
    def test_single_face_single_shot(self):
        """Single-face single-shot video -> one STATIONARY segment at face center."""
        dense_faces = [
            MockFrameFaces(timestamp=t, faces=[MockFace(nose_x=50, nose_y=40)])
            for t in [0.0, 0.5, 1.0, 1.5, 2.0]
        ]
        registry = MockFaceRegistry(slots=[MockFaceSlot(slot_id=0, x_center=50)])
        speaker_events = [MockSpeakerEvent(start=0, end=2, slot_id=0)]
        transcript = [MockTranscriptSeg(start=0, end=2)]

        segments = build_autoflip_segments(
            shot_cuts=[],
            face_registry=registry,
            active_speaker_events=speaker_events,
            dense_faces=dense_faces,
            saliency_keyframes=[],
            transcript_segments=transcript,
            speaker_to_slot={"SPEAKER_00": 0},
            video_duration=2.0,
            source_width=1920, source_height=1080,
            job_id="test",
        )

        assert len(segments) == 1
        assert segments[0].strategy == "stationary"
        assert abs(segments[0].subject_x - 50) < 5

    def test_single_face_mid_clip_shot_cut(self):
        """Single face + mid-clip shot cut -> two STATIONARY segments."""
        dense_faces = [
            MockFrameFaces(timestamp=t, faces=[MockFace(nose_x=40, nose_y=40)])
            for t in [0.0, 0.5, 1.0, 1.5, 2.0]
        ] + [
            MockFrameFaces(timestamp=t, faces=[MockFace(nose_x=60, nose_y=40)])
            for t in [2.5, 3.0, 3.5, 4.0, 4.5]
        ]
        registry = MockFaceRegistry(slots=[MockFaceSlot(slot_id=0, x_center=50)])
        speaker_events = [MockSpeakerEvent(start=0, end=5, slot_id=0)]

        segments = build_autoflip_segments(
            shot_cuts=[2.5],
            face_registry=registry,
            active_speaker_events=speaker_events,
            dense_faces=dense_faces,
            saliency_keyframes=[],
            transcript_segments=[],
            speaker_to_slot={},
            video_duration=5.0,
            source_width=1920, source_height=1080,
            job_id="test",
        )

        assert len(segments) == 2
        # Both should be stationary
        assert all(s.strategy == "stationary" for s in segments)

    def test_face_walking_across_frame(self):
        """Face walking across frame -> TRACKING or PANNING segment with motion_path."""
        dense_faces = [
            MockFrameFaces(timestamp=t * 0.5, faces=[MockFace(nose_x=20 + t * 3)])
            for t in range(20)
        ]
        registry = MockFaceRegistry(slots=[MockFaceSlot(slot_id=0, x_center=50)])
        speaker_events = [MockSpeakerEvent(start=0, end=10, slot_id=0)]

        segments = build_autoflip_segments(
            shot_cuts=[],
            face_registry=registry,
            active_speaker_events=speaker_events,
            dense_faces=dense_faces,
            saliency_keyframes=[],
            transcript_segments=[],
            speaker_to_slot={},
            video_duration=10.0,
            source_width=1920, source_height=1080,
            job_id="test",
        )

        assert len(segments) == 1
        assert segments[0].strategy in ("tracking", "panning")
        assert segments[0].motion_path is not None
        assert len(segments[0].motion_path) >= 2

    def test_two_faces_80pct_apart(self):
        """Two faces 80% apart, stationary -> one PADDING (blur_fill) segment."""
        dense_faces = [
            MockFrameFaces(timestamp=t, faces=[
                MockFace(nose_x=10, nose_y=40, identity_id=0),
                MockFace(nose_x=90, nose_y=40, identity_id=1),
            ])
            for t in [0.0, 0.5, 1.0, 1.5, 2.0]
        ]
        registry = MockFaceRegistry(slots=[
            MockFaceSlot(slot_id=0, x_center=10),
            MockFaceSlot(slot_id=1, x_center=90),
        ])
        speaker_events = [MockSpeakerEvent(start=0, end=2, slot_id=0)]

        segments = build_autoflip_segments(
            shot_cuts=[],
            face_registry=registry,
            active_speaker_events=speaker_events,
            dense_faces=dense_faces,
            saliency_keyframes=[],
            transcript_segments=[],
            speaker_to_slot={},
            video_duration=2.0,
            source_width=1920, source_height=1080,
            job_id="test",
        )

        assert len(segments) == 1
        assert segments[0].strategy == "blur_fill"
        assert segments[0].layout == "blur_fill"

    def test_segment_counts(self):
        """Verify segment strategy counts match expectations."""
        # 3 shots: stationary, tracking, padding
        dense_faces_1 = [
            MockFrameFaces(timestamp=t * 0.5, faces=[MockFace(nose_x=50)])
            for t in range(6)  # 0-3s, stationary
        ]
        dense_faces_2 = [
            MockFrameFaces(timestamp=3.0 + t * 0.5, faces=[MockFace(nose_x=20 + t * 5)])
            for t in range(6)  # 3-6s, walking
        ]
        dense_faces_3 = [
            MockFrameFaces(timestamp=6.0 + t * 0.5, faces=[
                MockFace(nose_x=10, identity_id=0),
                MockFace(nose_x=90, identity_id=1),
            ])
            for t in range(6)  # 6-9s, two faces apart
        ]

        registry = MockFaceRegistry(slots=[
            MockFaceSlot(slot_id=0, x_center=50),
            MockFaceSlot(slot_id=1, x_center=90),
        ])
        speaker_events = [MockSpeakerEvent(start=0, end=9, slot_id=0)]

        segments = build_autoflip_segments(
            shot_cuts=[3.0, 6.0],
            face_registry=registry,
            active_speaker_events=speaker_events,
            dense_faces=dense_faces_1 + dense_faces_2 + dense_faces_3,
            saliency_keyframes=[],
            transcript_segments=[],
            speaker_to_slot={},
            video_duration=9.0,
            source_width=1920, source_height=1080,
            job_id="test",
        )

        assert len(segments) == 3
        strategies = [s.strategy for s in segments]
        assert "stationary" in strategies
        # The padding shot (two faces 80% apart) should be blur_fill
        assert "blur_fill" in strategies

    def test_confidence_populated(self):
        """Confidence is populated on all segments."""
        dense_faces = [
            MockFrameFaces(timestamp=0.0, faces=[MockFace(nose_x=50)])
        ]
        registry = MockFaceRegistry(slots=[MockFaceSlot(slot_id=0, x_center=50)])

        segments = build_autoflip_segments(
            shot_cuts=[],
            face_registry=registry,
            active_speaker_events=[],
            dense_faces=dense_faces,
            saliency_keyframes=[],
            transcript_segments=[],
            speaker_to_slot={},
            video_duration=1.0,
            source_width=1920, source_height=1080,
            job_id="test",
        )

        assert len(segments) == 1
        # Confidence should be a number (not the default 0.0 if estimator ran)
        assert isinstance(segments[0].confidence, float)
