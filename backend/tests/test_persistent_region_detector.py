"""Tests for the persistent region detector."""
from dataclasses import dataclass, field

import pytest

from backend.services.persistent_region_detector import detect_persistent_regions


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
    total_frames: int = 200
    frames_with_faces: int = 200


@dataclass
class _FaceInfo:
    identity_id: int
    nose_x: float
    nose_y: float = 50.0
    width: float = 10.0
    height: float = 12.0


@dataclass
class _FrameFaces:
    timestamp: float
    faces: list[_FaceInfo] = field(default_factory=list)


class TestPersistentRegionDetector:
    def test_detects_facecam_in_bottom_right(self):
        """Small static face in bottom-right corner → facecam detected."""
        registry = _FaceRegistry(
            slots=[
                _FaceSlot(0, 50, 45, 55, frame_count=180, avg_width=15.0),  # main face
                _FaceSlot(1, 85, 83, 87, frame_count=160, avg_width=8.0),   # small facecam
            ],
            total_frames=200,
        )
        dense = [
            _FrameFaces(t, [
                _FaceInfo(0, 50, 50),
                _FaceInfo(1, 85, 80),  # bottom-right
            ])
            for t in range(0, 200)
        ]

        result = detect_persistent_regions(
            dense_faces=dense,
            face_registry=registry,
            video_duration=100.0,
        )
        assert result.has_facecam
        assert result.facecam_region is not None
        assert result.facecam_region.corner == "bottom_right"

    def test_no_facecam_when_face_is_large(self):
        """Large face is not classified as facecam."""
        registry = _FaceRegistry(
            slots=[_FaceSlot(0, 50, 45, 55, frame_count=180, avg_width=25.0)],
            total_frames=200,
        )
        dense = [_FrameFaces(t, [_FaceInfo(0, 50, 50, width=25.0)]) for t in range(200)]

        result = detect_persistent_regions(dense_faces=dense, face_registry=registry)
        assert not result.has_facecam

    def test_no_facecam_when_face_moves(self):
        """Face that moves too much is not a facecam."""
        registry = _FaceRegistry(
            slots=[_FaceSlot(0, 50, 20, 80, frame_count=180, avg_width=8.0)],
            total_frames=200,
        )
        dense = [_FrameFaces(t, [_FaceInfo(0, 50, 50)]) for t in range(200)]

        result = detect_persistent_regions(dense_faces=dense, face_registry=registry)
        assert not result.has_facecam  # x_range too wide (60)

    def test_empty_input_returns_empty(self):
        """No dense faces → no regions."""
        result = detect_persistent_regions(dense_faces=[], face_registry=None)
        assert not result.has_facecam
        assert not result.has_hud
        assert len(result.regions) == 0
