"""Fix 4: content classifier anime-text gate + geometry-over-text
tiebreaker + panel detection loosened to avg_range < 20.
"""

from dataclasses import dataclass, field
from typing import Optional

from backend.services.content_classifier import (
    ClipContentType,
    classify_clip,
    classify_content,
)
from backend.services.face_registry import FaceRegistry, FaceSlot


@dataclass
class _Face:
    nose_x: float = 50.0
    nose_y: float = 50.0
    width: float = 10.0
    height: float = 12.0
    identity_id: int = 0
    is_human: bool = True


@dataclass
class _FF:
    timestamp: float
    faces: list = field(default_factory=list)


@dataclass
class _Scene:
    description: str = ""


def _panel_registry_3seat():
    return FaceRegistry(
        slots=[
            FaceSlot(slot_id=0, x_center=23.0, x_min=20.0, x_max=28.0,
                     frame_count=80, avg_width=10.0, avg_height=12.0),
            FaceSlot(slot_id=1, x_center=56.0, x_min=52.0, x_max=62.0,
                     frame_count=80, avg_width=10.0, avg_height=12.0),
            FaceSlot(slot_id=2, x_center=80.0, x_min=75.0, x_max=87.0,
                     frame_count=80, avg_width=10.0, avg_height=12.0),
        ],
        total_frames=240,
        frames_with_faces=240,
    )


def _panel_dense_faces():
    out = []
    for i in range(60):
        out.append(_FF(
            timestamp=i * 0.5,
            faces=[
                _Face(nose_x=23.0, identity_id=0),
                _Face(nose_x=56.0, identity_id=1),
                _Face(nose_x=80.0, identity_id=2),
            ],
        ))
    return out


def test_weak_anime_text_does_not_beat_panel_geometry():
    # Mimics the Verzuz bug: 2 anime keyword scene hits pushed the
    # classifier to a three-way tie. With Fix 4 the weak text signal
    # only gives +1.0 (not +3.0) AND the panel detection bumps podcast
    # by +4.0, so multi_speaker_panel wins decisively.
    scenes = [
        _Scene(description="animated intro with characters talking"),
        _Scene(description="japanese subtitles overlay"),
    ]
    # Fill remaining scenes with benign narrative content.
    for _ in range(58):
        scenes.append(_Scene(description="three people talking on couch"))

    profile = classify_content(
        shot_cuts=[i * 3.0 for i in range(20)],
        face_registry=_panel_registry_3seat(),
        dense_faces=_panel_dense_faces(),
        scenes=scenes,
        video_duration=180.0,
        metadata={},
        job_id="anime-tie-test",
        transcript_segments=None,
    )
    # Should land on podcast (the geometry-backed type) via panel bump.
    assert profile.content_type == "podcast", (
        f"got {profile.content_type} (scores and signals: {profile.signals})"
    )
    assert profile.is_multi_speaker_panel is True
    # Confidence should be comfortably above 0.25.
    assert profile.confidence >= 0.3


def test_strong_anime_text_still_wins_on_fewfaces():
    # When anime_hits >= 5 AND avg_faces < 1.5 the strong bump
    # (+3.0) still applies. Use a dense_faces stream with 1 face
    # per frame so avg_faces stays low.
    scenes = []
    for _ in range(30):
        scenes.append(_Scene(description="anime animation animated cartoon manga"))

    dense = []
    for i in range(30):
        dense.append(_FF(timestamp=i * 0.5, faces=[_Face(nose_x=50.0)]))

    profile = classify_content(
        shot_cuts=[i * 3.0 for i in range(10)],
        face_registry=FaceRegistry(
            slots=[FaceSlot(slot_id=0, x_center=50.0, x_min=45.0, x_max=55.0,
                            frame_count=30, avg_width=10, avg_height=12)],
            total_frames=30, frames_with_faces=30,
        ),
        dense_faces=dense,
        scenes=scenes,
        video_duration=60.0,
        metadata={},
        job_id="anime-strong-test",
        transcript_segments=None,
    )
    # avg_faces = 1.0 < 1.5, anime_hits = 5 → strong bump; classifier
    # should prefer anime.
    assert profile.signals.get("scene_desc_anime_strong") is True
    assert profile.content_type == "anime"


def test_geometry_over_text_tiebreaker_resolves_anime_vs_panel():
    # Set up a scenario where anime text bumped high (hits >= 2 but
    # still only weak +1.0), podcast/vlog are competitive. The
    # tiebreaker should defer to the geometry-backed podcast.
    scenes = [_Scene(description="animated characters dialogue") for _ in range(5)]
    scenes += [_Scene(description="three people on couch talking") for _ in range(55)]

    profile = classify_content(
        shot_cuts=[i * 3.0 for i in range(20)],
        face_registry=_panel_registry_3seat(),
        dense_faces=_panel_dense_faces(),
        scenes=scenes,
        video_duration=180.0,
        metadata={},
        job_id="tiebreak-test",
        transcript_segments=None,
    )
    # With the panel bump + tiebreaker, this should land on podcast
    # (not anime, not vlog).
    assert profile.content_type == "podcast"


def test_panel_detection_allows_avg_range_up_to_20():
    # Slots with avg_range 18 (just under the new 20 threshold)
    # should still fire multi_speaker_panel. The old threshold of 15
    # would have rejected this.
    registry = FaceRegistry(
        slots=[
            FaceSlot(slot_id=0, x_center=25.0, x_min=15.0, x_max=33.0,   # range 18
                     frame_count=80, avg_width=10.0, avg_height=12.0),
            FaceSlot(slot_id=1, x_center=55.0, x_min=46.0, x_max=64.0,   # range 18
                     frame_count=80, avg_width=10.0, avg_height=12.0),
            FaceSlot(slot_id=2, x_center=80.0, x_min=71.0, x_max=89.0,   # range 18
                     frame_count=80, avg_width=10.0, avg_height=12.0),
        ],
        total_frames=240,
        frames_with_faces=240,
    )
    profile = classify_content(
        shot_cuts=[i * 3.0 for i in range(20)],
        face_registry=registry,
        dense_faces=_panel_dense_faces(),
        scenes=[_Scene(description="three speakers on stage") for _ in range(20)],
        video_duration=180.0,
        metadata={},
        job_id="loose-panel-test",
        transcript_segments=None,
    )
    assert profile.is_multi_speaker_panel is True
    assert profile.signals.get("panel_avg_slot_range") == 18.0
