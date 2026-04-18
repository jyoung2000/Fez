"""Fix 3 — multi_speaker_panel detection and routing.

Mocks a 3-seat panel face_registry + dense_faces and asserts that
classify_content marks is_multi_speaker_panel=True and classify_clip
returns ClipContentType.MULTI_SPEAKER_PANEL.
"""

from dataclasses import dataclass, field

from backend.services.content_classifier import (
    ClipContentType,
    ContentProfile,
    classify_clip,
    classify_content,
)
from backend.services.face_registry import FaceRegistry, FaceSlot


@dataclass
class _FakeFace:
    nose_x: float = 50.0
    nose_y: float = 50.0
    width: float = 10.0
    height: float = 10.0
    identity_id: int = -1
    is_human: bool = True


@dataclass
class _FakeDense:
    timestamp: float
    faces: list = field(default_factory=list)


def _build_panel_registry():
    return FaceRegistry(
        slots=[
            FaceSlot(slot_id=0, x_center=23.0, x_min=19.0, x_max=27.0,
                     frame_count=80, avg_width=10.0, avg_height=12.0),
            FaceSlot(slot_id=1, x_center=56.0, x_min=52.0, x_max=60.0,
                     frame_count=80, avg_width=10.0, avg_height=12.0),
            FaceSlot(slot_id=2, x_center=80.0, x_min=76.0, x_max=84.0,
                     frame_count=80, avg_width=10.0, avg_height=12.0),
        ],
        total_frames=240,
        frames_with_faces=240,
    )


def _build_panel_dense_faces(n=60):
    # Each dense frame shows all 3 speakers side by side so
    # multi_face_frame_pct = 1.0.
    out = []
    for i in range(n):
        out.append(_FakeDense(
            timestamp=i * 0.5,
            faces=[
                _FakeFace(nose_x=23.0, identity_id=0),
                _FakeFace(nose_x=56.0, identity_id=1),
                _FakeFace(nose_x=80.0, identity_id=2),
            ],
        ))
    return out


def test_classify_content_flags_is_multi_speaker_panel():
    registry = _build_panel_registry()
    dense = _build_panel_dense_faces(n=60)
    profile = classify_content(
        shot_cuts=[i * 2.0 for i in range(10)],  # 5 cuts/min → podcast-ish
        face_registry=registry,
        dense_faces=dense,
        scenes=[],
        video_duration=120.0,
        metadata={},
        job_id="panel-test",
        transcript_segments=None,
    )
    assert profile.is_multi_speaker_panel is True
    assert profile.signals.get("multi_face_frame_pct", 0) >= 0.25
    assert "multi_speaker_panel" in profile.signals


def test_classify_clip_routes_to_multi_speaker_panel():
    prof = ContentProfile()
    prof.content_type = "vlog"  # the old buggy output
    prof.is_multi_speaker_panel = True
    result = classify_clip(content_profile=prof)
    assert result == ClipContentType.MULTI_SPEAKER_PANEL


def test_multi_speaker_panel_beats_animation_dialogue():
    # Edge case: anime panel. is_animated=True AND is_multi_speaker_panel=True.
    # Panel wins — seated anime characters still need panel framing,
    # not the AnimeMode dialogue treatment.
    prof = ContentProfile()
    prof.content_type = "narrative"
    prof.is_animated = True
    prof.is_multi_speaker_panel = True
    result = classify_clip(content_profile=prof)
    assert result == ClipContentType.MULTI_SPEAKER_PANEL


def test_classify_clip_preserves_single_speaker_routing():
    # Regression: when is_multi_speaker_panel=False, classify_clip
    # still returns the base type.
    prof = ContentProfile()
    prof.content_type = "podcast"
    prof.is_multi_speaker_panel = False
    result = classify_clip(content_profile=prof)
    assert result == ClipContentType.TALKING_HEAD
