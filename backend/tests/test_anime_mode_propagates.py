"""Test that AnimeMode detected by the face verifier propagates all
the way to classify_clip() returning ClipContentType.ANIMATION_DIALOGUE.

Regression for the K S01E12 run where `_verify_faces_in_results` logged
`[AnimeMode] skipping human verifier` but the classifier still emitted
`content_type=unknown` and the whole dialogue-mode tuning stayed off.
"""

from dataclasses import dataclass, field

from backend.services import face_detector
from backend.services.content_classifier import (
    ClipContentType,
    ContentProfile,
    classify_clip,
    classify_content,
)
from backend.services.content_type_config import ContentType


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
    faces: list = field(default_factory=list)


def test_anime_mode_global_promotes_unknown_profile():
    """When face_detector.ANIME_MODE_DETECTED is True and the heuristic
    classifier emits UNKNOWN, classify_content must promote it to
    NARRATIVE and set is_animated=True, and classify_clip must then
    return ANIMATION_DIALOGUE.
    """
    face_detector.ANIME_MODE_DETECTED = True
    try:
        # Minimal signals that wouldn't otherwise reach a high-confidence
        # vote — mimics the K S01E12 case where scene descriptions had
        # 0 anime keywords.
        registry = _FaceRegistry(slots=[
            _FaceSlot(0, 30), _FaceSlot(1, 70),
        ])
        dense = [_FrameFaces(float(t), [_FaceInfo(0, 30)]) for t in range(20)]

        profile = classify_content(
            shot_cuts=[],
            face_registry=registry,
            dense_faces=dense,
            scenes=[],
            video_duration=60.0,
        )

        assert profile.is_animated is True, (
            f"Expected is_animated=True, got signals={profile.signals}"
        )
        assert profile.signals.get("anime_mode_from_detector") is True
        # classify_clip must then land on ANIMATION_DIALOGUE.
        clip_type = classify_clip(content_profile=profile)
        assert clip_type == ClipContentType.ANIMATION_DIALOGUE
    finally:
        face_detector.ANIME_MODE_DETECTED = False


def test_anime_mode_false_keeps_narrative_generic():
    """Sanity guard: when ANIME_MODE_DETECTED is False and nothing else
    indicates animation, a NARRATIVE profile should NOT turn into
    ANIMATION_DIALOGUE.
    """
    face_detector.ANIME_MODE_DETECTED = False
    profile = ContentProfile(
        content_type=ContentType.NARRATIVE.value,
        confidence=0.6,
        is_animated=False,
        is_cinematic_dialogue=False,
    )
    clip_type = classify_clip(content_profile=profile)
    assert clip_type == ClipContentType.GENERIC
