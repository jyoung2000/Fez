"""Tests for aggregate_scene_focus."""

from dataclasses import dataclass, field
from typing import Optional

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

    def nearest_slot(self, x):
        if not self.slots:
            return None
        return min(self.slots, key=lambda s: abs(s.x_center - x))

    def slot_by_id(self, slot_id):
        for s in self.slots:
            if s.slot_id == slot_id:
                return s
        return None


class TestAggregateFocus:
    def test_one_face_centered(self):
        """One face centered -> bounding rect tight on the face, fits, crop center = face center."""
        dense_faces = [
            MockFrameFaces(timestamp=t, faces=[MockFace(nose_x=50, nose_y=40, width=10, height=13)])
            for t in [0.0, 0.5, 1.0, 1.5, 2.0]
        ]
        registry = MockFaceRegistry(slots=[MockFaceSlot(slot_id=0, x_center=50)])
        result = aggregate_scene_focus(
            shot_start=0, shot_end=2.5, dense_faces=dense_faces,
            saliency_keyframes=[], persistent_regions=None,
            face_registry=registry, source_width=1920, source_height=1080,
        )
        assert result.fits_target_aspect is True
        assert len(result.required) == 5
        # Bounding rect should be tight on the face
        cx, cy, w, h = result.min_bounding_rect
        assert abs(cx - 50) < 1
        assert abs(cy - 40) < 1
        # Crop center should match face center
        assert abs(result.optimal_crop_center[0] - 50) < 2

    def test_two_faces_80pct_apart(self):
        """Two faces 80% apart horizontally -> doesn't fit."""
        dense_faces = [
            MockFrameFaces(timestamp=0.0, faces=[
                MockFace(nose_x=10, nose_y=40, width=10, height=13, identity_id=0),
                MockFace(nose_x=90, nose_y=40, width=10, height=13, identity_id=1),
            ]),
        ]
        registry = MockFaceRegistry(slots=[
            MockFaceSlot(slot_id=0, x_center=10),
            MockFaceSlot(slot_id=1, x_center=90),
        ])
        result = aggregate_scene_focus(
            shot_start=0, shot_end=1, dense_faces=dense_faces,
            saliency_keyframes=[], persistent_regions=None,
            face_registry=registry, source_width=1920, source_height=1080,
        )
        # For 1920x1080 -> 9:16, crop_width_pct = (9/16)/(16/9)*100 ~ 31.6%
        # Bounding rect width = 90+5 - (10-5) = 90
        # 90 > 31.6, so doesn't fit
        assert result.fits_target_aspect is False
        # Crop center is centroid
        assert abs(result.optimal_crop_center[0] - 50) < 1

    def test_no_required_features(self):
        """No required features -> fits, center (50, 50)."""
        result = aggregate_scene_focus(
            shot_start=0, shot_end=5, dense_faces=[],
            saliency_keyframes=[], persistent_regions=None,
            face_registry=MockFaceRegistry(), source_width=1920, source_height=1080,
        )
        assert result.fits_target_aspect is True
        assert result.optimal_crop_center == (50, 50)
        assert result.min_bounding_rect == (50, 50, 0, 0)

    def test_saliency_only(self):
        """Saliency but no faces -> optional populated, required empty, still fits."""
        saliency = [(0.5, 60.0, 0.8), (1.0, 65.0, 0.7)]
        result = aggregate_scene_focus(
            shot_start=0, shot_end=2, dense_faces=[],
            saliency_keyframes=saliency, persistent_regions=None,
            face_registry=MockFaceRegistry(), source_width=1920, source_height=1080,
        )
        assert len(result.required) == 0
        assert len(result.optional) == 2
        assert result.fits_target_aspect is True
        assert result.optional[0].kind == FeatureKind.SALIENCY

    def test_non_fhd_source(self):
        """Non-FHD source (1280x720) -> aspect math uses actual dims."""
        dense_faces = [
            MockFrameFaces(timestamp=0.0, faces=[MockFace(nose_x=50, nose_y=40, width=10, height=13)]),
        ]
        registry = MockFaceRegistry(slots=[MockFaceSlot(slot_id=0, x_center=50)])
        result = aggregate_scene_focus(
            shot_start=0, shot_end=1, dense_faces=dense_faces,
            saliency_keyframes=[], persistent_regions=None,
            face_registry=registry, source_width=1280, source_height=720,
        )
        # 1280x720 has same 16:9 aspect as 1920x1080
        # crop_width_pct = (9/16)/(16/9)*100 = 31.6%
        assert result.fits_target_aspect is True

    def test_per_frame_targets(self):
        """Per-frame targets are populated at each dense_faces timestamp."""
        dense_faces = [
            MockFrameFaces(timestamp=0.0, faces=[MockFace(nose_x=30, nose_y=40)]),
            MockFrameFaces(timestamp=1.0, faces=[MockFace(nose_x=50, nose_y=40)]),
            MockFrameFaces(timestamp=2.0, faces=[MockFace(nose_x=70, nose_y=40)]),
        ]
        registry = MockFaceRegistry(slots=[MockFaceSlot(slot_id=0, x_center=50)])
        result = aggregate_scene_focus(
            shot_start=0, shot_end=3, dense_faces=dense_faces,
            saliency_keyframes=[], persistent_regions=None,
            face_registry=registry, source_width=1920, source_height=1080,
        )
        assert len(result.per_frame_target) == 3
        # Targets follow the face x positions
        assert abs(result.per_frame_target[0][1] - 30) < 1
        assert abs(result.per_frame_target[1][1] - 50) < 1
        assert abs(result.per_frame_target[2][1] - 70) < 1
