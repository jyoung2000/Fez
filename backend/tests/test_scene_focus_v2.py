"""Tests for scene_focus.py v2 with saliency regions and object detections."""

from dataclasses import dataclass, field

from backend.services.scene_focus import aggregate_scene_focus
from backend.services.focus_model import FeatureKind


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
    lip_aperture: float = 0.0
    is_speaking: bool = False


@dataclass
class MockFrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)
    primary_face_idx: int = 0
    path: str = ""


@dataclass
class MockFaceRegistry:
    slots: list = field(default_factory=list)
    total_frames: int = 10
    frames_with_faces: int = 10


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


class TestFaceOnlyRegression:
    def test_face_only_identical_output(self):
        """Face-only input (no saliency, no objects) -> byte-identical output."""
        dense_faces = [
            MockFrameFaces(timestamp=t, faces=[MockFace(nose_x=50, nose_y=40, width=10, height=13)])
            for t in [0.0, 0.5, 1.0, 1.5, 2.0]
        ]
        registry = MockFaceRegistry()

        # Without new params
        result_old = aggregate_scene_focus(
            shot_start=0, shot_end=2.5, dense_faces=dense_faces,
            saliency_keyframes=[], persistent_regions=None,
            face_registry=registry, source_width=1920, source_height=1080,
        )

        # With new params explicitly None
        result_new = aggregate_scene_focus(
            shot_start=0, shot_end=2.5, dense_faces=dense_faces,
            saliency_keyframes=[], persistent_regions=None,
            face_registry=registry, source_width=1920, source_height=1080,
            saliency_regions=None, object_detections=None,
        )

        assert len(result_old.required) == len(result_new.required)
        assert len(result_old.optional) == len(result_new.optional)
        assert result_old.fits_target_aspect == result_new.fits_target_aspect
        assert result_old.optimal_crop_center == result_new.optimal_crop_center
        assert result_old.min_bounding_rect == result_new.min_bounding_rect


class TestSaliencyRegionIntegration:
    def test_saliency_region_appears_in_optional(self):
        """One saliency region -> appears in optional, required unchanged."""
        saliency_regions = [
            MockSaliencyRegion(timestamp=1.0, x=60.0, y=50.0, w=15.0, h=20.0, saliency_score=0.8),
        ]
        result = aggregate_scene_focus(
            shot_start=0, shot_end=2, dense_faces=[],
            saliency_keyframes=[], persistent_regions=None,
            face_registry=MockFaceRegistry(), source_width=1920, source_height=1080,
            saliency_regions=saliency_regions,
        )
        assert len(result.required) == 0
        assert len(result.optional) == 1
        assert result.optional[0].kind == FeatureKind.SALIENCY
        assert result.optional[0].must_be_in_frame is False
        assert abs(result.optional[0].weight - 0.8) < 0.01


class TestObjectDetectionIntegration:
    def test_person_detection_in_required(self):
        """One 'person' detection -> appears in required with weight 1.0 * confidence."""
        detections = [
            MockObjectDetection(timestamp=1.0, x=50.0, y=40.0, w=15.0, h=30.0,
                                class_name="person", confidence=0.9),
        ]
        result = aggregate_scene_focus(
            shot_start=0, shot_end=2, dense_faces=[],
            saliency_keyframes=[], persistent_regions=None,
            face_registry=MockFaceRegistry(), source_width=1920, source_height=1080,
            object_detections=detections,
        )
        assert len(result.required) == 1
        assert result.required[0].kind == FeatureKind.OBJECT
        assert result.required[0].must_be_in_frame is True
        assert abs(result.required[0].weight - 0.9) < 0.01  # 1.0 * 0.9

    def test_stationary_car_in_optional(self):
        """One stationary 'car' -> appears in optional (not moving)."""
        detections = [
            MockObjectDetection(timestamp=1.0, x=50.0, y=50.0, w=20.0, h=15.0,
                                class_name="car", confidence=0.85),
        ]
        result = aggregate_scene_focus(
            shot_start=0, shot_end=2, dense_faces=[],
            saliency_keyframes=[], persistent_regions=None,
            face_registry=MockFaceRegistry(), source_width=1920, source_height=1080,
            object_detections=detections,
        )
        # Single detection -> not moving -> car has if_moving -> optional
        assert len(result.required) == 0
        assert len(result.optional) == 1
        assert result.optional[0].must_be_in_frame is False

    def test_moving_car_in_required(self):
        """One moving 'car' (>5% x travel) -> appears in required."""
        detections = [
            MockObjectDetection(timestamp=0.5, x=30.0, y=50.0, w=20.0, h=15.0,
                                class_name="car", confidence=0.85),
            MockObjectDetection(timestamp=1.0, x=45.0, y=50.0, w=20.0, h=15.0,
                                class_name="car", confidence=0.85),
        ]
        result = aggregate_scene_focus(
            shot_start=0, shot_end=2, dense_faces=[],
            saliency_keyframes=[], persistent_regions=None,
            face_registry=MockFaceRegistry(), source_width=1920, source_height=1080,
            object_detections=detections,
        )
        # x travel = 45 - 30 = 15 > 5 -> moving -> required
        assert len(result.required) == 2
        assert all(r.must_be_in_frame for r in result.required)


class TestFaceAndSaliencyMixed:
    def test_face_dominates_saliency_optional(self):
        """Face + saliency -> face dominates required; saliency is optional only."""
        dense_faces = [
            MockFrameFaces(timestamp=1.0, faces=[MockFace(nose_x=50, nose_y=40, width=10, height=13)]),
        ]
        saliency_regions = [
            MockSaliencyRegion(timestamp=1.0, x=70.0, y=50.0, w=15.0, h=20.0, saliency_score=0.9),
        ]
        result = aggregate_scene_focus(
            shot_start=0, shot_end=2, dense_faces=dense_faces,
            saliency_keyframes=[], persistent_regions=None,
            face_registry=MockFaceRegistry(), source_width=1920, source_height=1080,
            saliency_regions=saliency_regions,
        )
        # Face is required, saliency is optional
        assert len(result.required) == 1
        assert result.required[0].kind == FeatureKind.FACE
        assert len(result.optional) == 1
        assert result.optional[0].kind == FeatureKind.SALIENCY


class TestNonFHDSource:
    def test_1280x720_uses_passed_dims(self):
        """Non-FHD source (1280x720) -> aspect math uses passed dims."""
        dense_faces = [
            MockFrameFaces(timestamp=0.0, faces=[MockFace(nose_x=50, nose_y=40, width=10, height=13)]),
        ]
        result = aggregate_scene_focus(
            shot_start=0, shot_end=1, dense_faces=dense_faces,
            saliency_keyframes=[], persistent_regions=None,
            face_registry=MockFaceRegistry(), source_width=1280, source_height=720,
            saliency_regions=[], object_detections=[],
        )
        # 1280x720 has same 16:9 aspect as 1920x1080 -> same crop_width_pct
        assert result.fits_target_aspect is True
