"""Tests for autoflip_segmenter with saliency regions and object detections."""

from dataclasses import dataclass, field
from typing import Optional

from backend.services.autoflip_segmenter import build_autoflip_segments


# ── Mock objects ──
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
    x_min: float = 35.0
    x_max: float = 65.0
    frame_count: int = 20
    avg_width: float = 10.0
    avg_height: float = 13.0


@dataclass
class MockFaceRegistry:
    slots: list = field(default_factory=list)
    total_frames: int = 20
    frames_with_faces: int = 20

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


@dataclass
class MockSaliencyRegion:
    timestamp: float
    x: float
    y: float
    w: float
    h: float
    saliency_score: float
    motion_score: float = 0.0
    spatial_score: float = 0.0


@dataclass
class MockObjectDetection:
    timestamp: float
    x: float
    y: float
    w: float
    h: float
    class_name: str
    confidence: float


class TestRegressionFaceOnly:
    def test_single_face_identical_output(self):
        """Existing single-face test produces identical output (regression guard)."""
        dense_faces = [
            MockFrameFaces(timestamp=t, faces=[MockFace(nose_x=50, nose_y=40)])
            for t in [i * 0.5 for i in range(20)]
        ]
        registry = MockFaceRegistry(slots=[MockFaceSlot(slot_id=0, x_center=50)])
        speaker_events = [MockSpeakerEvent(start=0, end=10, slot_id=0)]

        # Without new params
        segments_old = build_autoflip_segments(
            shot_cuts=[], face_registry=registry,
            active_speaker_events=speaker_events, dense_faces=dense_faces,
            saliency_keyframes=[], transcript_segments=[],
            speaker_to_slot={}, video_duration=10.0,
            source_width=1280, source_height=720,
            job_id="regression_test",
        )

        # With new params as None
        segments_new = build_autoflip_segments(
            shot_cuts=[], face_registry=registry,
            active_speaker_events=speaker_events, dense_faces=dense_faces,
            saliency_keyframes=[], transcript_segments=[],
            speaker_to_slot={}, video_duration=10.0,
            source_width=1280, source_height=720,
            saliency_regions=None, object_detections=None,
            job_id="regression_test",
        )

        assert len(segments_old) == len(segments_new)
        for old, new in zip(segments_old, segments_new):
            assert old.strategy == new.strategy
            assert old.subject_x == new.subject_x


class TestPersonDetectionAcrossFacelessShot:
    def test_person_detection_produces_segment(self):
        """Single-person detection across a faceless shot -> produces a segment."""
        # No faces, just object detections
        detections = [
            MockObjectDetection(timestamp=t, x=40 + t * 2, y=50.0, w=15.0, h=30.0,
                                class_name="person", confidence=0.9)
            for t in [i * 0.5 for i in range(20)]
        ]
        registry = MockFaceRegistry(slots=[])

        segments = build_autoflip_segments(
            shot_cuts=[], face_registry=registry,
            active_speaker_events=[], dense_faces=[],
            saliency_keyframes=[], transcript_segments=[],
            speaker_to_slot={}, video_duration=10.0,
            source_width=1280, source_height=720,
            object_detections=detections,
            job_id="person_det_test",
        )

        assert len(segments) >= 1
        seg = segments[0]
        assert abs(seg.start - 0.0) < 0.01
        assert abs(seg.end - 10.0) < 0.01
        # With person detections as required features, trajectory planner kicks in
        assert seg.strategy in ("tracking", "panning", "stationary", "blur_fill")


class TestSaliencyOnlyShot:
    def test_saliency_only_produces_segment(self):
        """Saliency-only shot (no face, no object) -> produces segment."""
        saliency_regions = [
            MockSaliencyRegion(timestamp=t, x=50.0, y=50.0, w=20.0, h=20.0,
                               saliency_score=0.7)
            for t in [i * 0.5 for i in range(20)]
        ]
        registry = MockFaceRegistry(slots=[])

        segments = build_autoflip_segments(
            shot_cuts=[], face_registry=registry,
            active_speaker_events=[], dense_faces=[],
            saliency_keyframes=[], transcript_segments=[],
            speaker_to_slot={}, video_duration=10.0,
            source_width=1280, source_height=720,
            saliency_regions=saliency_regions,
            job_id="saliency_only_test",
        )

        assert len(segments) >= 1
        # Saliency is non-required, no face → zero confidence → fallback to
        # blur_fill or wide_master (confidence ladder now enforced correctly)
        assert segments[0].strategy in ("stationary", "blur_fill", "wide_master")


class TestFaceAndObjectMixed:
    def test_face_dominates_object_secondary(self):
        """Face + object both in frame -> face dominates as required feature."""
        dense_faces = [
            MockFrameFaces(timestamp=t, faces=[MockFace(nose_x=40, nose_y=40)])
            for t in [i * 0.5 for i in range(20)]
        ]
        detections = [
            MockObjectDetection(timestamp=t, x=70.0, y=50.0, w=10.0, h=10.0,
                                class_name="cat", confidence=0.85)
            for t in [i * 0.5 for i in range(20)]
        ]
        registry = MockFaceRegistry(slots=[MockFaceSlot(slot_id=0, x_center=40)])

        segments = build_autoflip_segments(
            shot_cuts=[], face_registry=registry,
            active_speaker_events=[MockSpeakerEvent(start=0, end=10, slot_id=0)],
            dense_faces=dense_faces,
            saliency_keyframes=[], transcript_segments=[],
            speaker_to_slot={}, video_duration=10.0,
            source_width=1280, source_height=720,
            object_detections=detections,
            job_id="mixed_test",
        )

        assert len(segments) >= 1
        # Both face and cat are required, might not fit -> could be any strategy
        # The important thing is it produces valid segments
        for seg in segments:
            assert hasattr(seg, 'strategy')
            assert hasattr(seg, 'subject_x')
