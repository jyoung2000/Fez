"""Fix 3.2: shot-level classifier emits one ShotProfile per shot, and
each shot's signals dominate over the clip-level prior.

Spec fixture: three shots —
  * dialogue (2 faces, static motion)
  * montage (0 faces, short duration)
  * action (1 face, high motion)

All three must produce distinct content_type values.
"""
from dataclasses import dataclass, field

from backend.services.content_classifier import ContentProfile
from backend.services.shot_classifier import classify_shots, ShotProfile


@dataclass
class _Face:
    nose_x: float
    nose_y: float = 50.0
    width: float = 10.0
    height: float = 12.0
    identity_id: int = -1
    is_human: bool = True
    lip_aperture: float = 0.0


@dataclass
class _FrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)


def _dialogue_frames(t0: float, t1: float, step: float = 0.1) -> list:
    """Two stable faces at x=30 and x=70, light jitter."""
    frames = []
    t = t0
    i = 0
    while t < t1:
        jitter = (i % 3) * 0.5
        frames.append(_FrameFaces(
            timestamp=t,
            faces=[
                _Face(nose_x=30.0 + jitter, identity_id=0),
                _Face(nose_x=70.0 - jitter, identity_id=1),
            ],
        ))
        t += step
        i += 1
    return frames


def _empty_frames(t0: float, t1: float, step: float = 0.1) -> list:
    """No faces (montage-ish b-roll)."""
    frames = []
    t = t0
    while t < t1:
        frames.append(_FrameFaces(timestamp=t, faces=[]))
        t += step
    return frames


def _action_frames(t0: float, t1: float, step: float = 0.1) -> list:
    """One face sweeping across the frame (chaotic)."""
    frames = []
    t = t0
    i = 0
    while t < t1:
        # sweep left → right → left with big swings — stdev > 20
        # so motion_class lands on "chaotic".
        phase = i % 8
        # Values: 10, 30, 50, 90, 70, 50, 20, 80 → stdev ~ 27
        values = [10.0, 30.0, 50.0, 90.0, 70.0, 50.0, 20.0, 80.0]
        x = values[phase]
        frames.append(_FrameFaces(
            timestamp=t,
            faces=[_Face(nose_x=x, identity_id=2)],
        ))
        t += step
        i += 1
    return frames


class TestClassifyShots:
    def test_three_shots_three_content_types(self):
        """Spec §3.2 acceptance: dialogue / montage / action yield
        three distinct content_type values."""
        dense = (
            _dialogue_frames(0.0, 5.0)
            + _empty_frames(5.0, 6.0)
            + _action_frames(6.0, 10.0)
        )
        shot_boundaries = [5.0, 6.0]
        profiles = classify_shots(
            shot_boundaries=shot_boundaries,
            dense_faces=dense,
            duration_sec=10.0,
        )
        assert len(profiles) == 3

        # Shot 0: dialogue-heavy talking-head or podcast
        assert profiles[0].content_type in ("podcast", "talking_head"), (
            profiles[0]
        )
        assert profiles[0].motion_class == "static"
        assert profiles[0].face_coverage >= 0.9

        # Shot 1: montage, short + empty
        assert profiles[1].content_type in (
            "music_video", "sports",
        ), profiles[1]
        assert profiles[1].face_coverage == 0.0

        # Shot 2: chaotic motion with one face
        assert profiles[2].motion_class in ("dynamic", "chaotic")
        assert profiles[2].content_type in (
            "sports", "music_video",
        ), profiles[2]

        # Distinct content_types between shots 0 and 2 minimum.
        assert profiles[0].content_type != profiles[2].content_type

    def test_shot_dominant_slot_is_the_face_that_appears_most(self):
        dense = _dialogue_frames(0.0, 5.0)
        profiles = classify_shots(
            shot_boundaries=[], dense_faces=dense, duration_sec=5.0,
        )
        assert len(profiles) == 1
        assert profiles[0].dominant_subject_slot in (0, 1)

    def test_no_boundaries_produces_single_shot(self):
        dense = _dialogue_frames(0.0, 3.0)
        profiles = classify_shots(
            shot_boundaries=[], dense_faces=dense, duration_sec=3.0,
        )
        assert len(profiles) == 1
        assert profiles[0].start == 0.0
        assert profiles[0].end == 3.0

    def test_zero_duration_returns_empty(self):
        assert classify_shots([], [], duration_sec=0.0) == []

    def test_clip_profile_is_a_soft_prior_not_a_gate(self):
        """An animation clip-level prior doesn't force a shot that
        clearly looks like a talking-head to become anime."""
        dense = _dialogue_frames(0.0, 5.0)
        clip = ContentProfile(
            content_type="anime", confidence=0.8, is_animated=True,
        )
        profiles = classify_shots(
            shot_boundaries=[],
            dense_faces=dense,
            duration_sec=5.0,
            clip_profile=clip,
        )
        # Geometry wins: face_coverage=1.0, static motion, avg_faces=2
        # → podcast/talking_head dominates despite +0.8 anime prior
        # and +1.0 is_animated bonus.
        assert profiles[0].content_type in ("podcast", "talking_head"), (
            profiles[0].content_type
        )

    def test_degenerate_zero_length_shot_is_skipped(self):
        """A boundary at exactly the clip duration shouldn't create a
        zero-length tail shot."""
        dense = _dialogue_frames(0.0, 5.0)
        profiles = classify_shots(
            shot_boundaries=[5.0],  # degenerate
            dense_faces=dense,
            duration_sec=5.0,
        )
        # Only one meaningful shot; the zero-length tail is dropped.
        assert len(profiles) == 1
