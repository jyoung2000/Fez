"""Tests for the content classifier."""
from dataclasses import dataclass, field

import pytest

from backend.services.content_classifier import classify_content, ContentProfile


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

    @property
    def multi_speaker(self):
        return len(self.slots) >= 2

    @property
    def is_continuous_motion(self):
        return False


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


@dataclass
class _Scene:
    description: str = ""
    timestamp: float = 0.0


class TestContentClassifier:
    def test_high_cut_rate_classifies_narrative(self):
        """Many shot cuts + narrative scene descriptions → narrative."""
        shot_cuts = [i * 3.0 for i in range(80)]  # 80 cuts in 240s = 20/min
        registry = _FaceRegistry(slots=[
            _FaceSlot(0, 30, 10, 60, frame_count=150),  # wide range (characters move)
            _FaceSlot(1, 70, 50, 90, frame_count=150),
        ])
        dense = [_FrameFaces(t, [_FaceInfo(0, 30), _FaceInfo(1, 70)])
                 for t in range(0, 240)]
        scenes = [
            _Scene("character close-up showing emotion during dramatic dialogue"),
            _Scene("wide shot of the scene with cinematic composition"),
            _Scene("reaction shot of the character looking surprised"),
        ]

        result = classify_content(
            shot_cuts=shot_cuts,
            face_registry=registry,
            dense_faces=dense,
            scenes=scenes,
            video_duration=240.0,
        )
        assert result.content_type == "narrative"
        assert result.confidence > 0.3

    def test_low_cut_rate_static_faces_classifies_podcast(self):
        """No cuts + 2 static face slots → podcast."""
        registry = _FaceRegistry(slots=[
            _FaceSlot(0, 25, 23, 27, frame_count=180),
            _FaceSlot(1, 75, 73, 77, frame_count=180),
        ])
        dense = [_FrameFaces(t, [_FaceInfo(0, 25), _FaceInfo(1, 75)])
                 for t in range(0, 600)]

        result = classify_content(
            shot_cuts=[],
            face_registry=registry,
            dense_faces=dense,
            scenes=[],
            video_duration=600.0,
        )
        assert result.content_type == "podcast"
        assert result.confidence > 0.3

    def test_gaming_keywords_in_scenes(self):
        """Gaming keywords in scene descriptions → gaming."""
        scenes = [
            _Scene("gameplay HUD visible with health bar and ammo counter"),
            _Scene("crosshair on enemy, kill confirmed"),
            _Scene("minimap shows team positions, game score displayed"),
        ]

        result = classify_content(
            shot_cuts=[],
            face_registry=_FaceRegistry(),
            dense_faces=[],
            scenes=scenes,
            video_duration=120.0,
        )
        assert result.content_type == "gaming"

    def test_user_override_takes_priority(self):
        """User-specified content_type overrides all signals."""
        result = classify_content(
            shot_cuts=[i * 3.0 for i in range(80)],  # would be narrative
            face_registry=_FaceRegistry(),
            dense_faces=[],
            scenes=[],
            video_duration=240.0,
            metadata={"content_type": "podcast"},
        )
        assert result.content_type == "podcast"
        assert result.confidence == 1.0

    def test_letterbox_source_classifies_narrative(self):
        """Wide aspect ratio (cinematic 2.35:1) → narrative."""
        result = classify_content(
            shot_cuts=[5.0, 10.0],
            face_registry=_FaceRegistry(),
            dense_faces=[],
            scenes=[],
            video_duration=120.0,
            metadata={"resolution": "2560x1080"},  # 2.37:1
        )
        assert result.content_type == "narrative"

    def test_no_faces_high_cut_gaming_with_keywords(self):
        """No faces + gaming keywords → gaming even with cuts."""
        scenes = [_Scene("gameplay HUD ammo health")]
        dense = [_FrameFaces(t, []) for t in range(0, 120)]

        result = classify_content(
            shot_cuts=[],
            face_registry=_FaceRegistry(total_frames=120, frames_with_faces=0),
            dense_faces=dense,
            scenes=scenes,
            video_duration=120.0,
        )
        assert result.content_type == "gaming"
