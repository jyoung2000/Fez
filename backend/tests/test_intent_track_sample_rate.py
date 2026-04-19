"""Fix 3.3: intent track samples at 10 Hz, not 2 Hz."""
from dataclasses import dataclass, field

from backend.services.intent_track import build_intent_track


@dataclass
class _Face:
    nose_x: float
    nose_y: float = 50.0
    width: float = 10.0
    height: float = 12.0
    identity_id: int = -1
    is_human: bool = True


@dataclass
class _FrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)


def test_10_hz_default_on_10s_clip_is_about_100_samples():
    """Spec: len(track) on a 10-second clip should be 100 ± 2."""
    dense = [
        _FrameFaces(timestamp=t * 0.5,
                    faces=[_Face(nose_x=50.0)])
        for t in range(21)  # 0.0, 0.5, ..., 10.0
    ]
    track = build_intent_track(
        duration_sec=10.0,
        shot_boundaries=[],
        shot_profiles=[],
        dense_faces=dense,
        active_speaker_events=[],
    )
    assert 98 <= len(track) <= 102, (
        f"expected ~100 samples at 10 Hz, got {len(track)}"
    )


def test_custom_sample_hz():
    """Override sample_hz → sample count scales accordingly."""
    dense = [
        _FrameFaces(timestamp=t,
                    faces=[_Face(nose_x=50.0)])
        for t in [0.0, 1.0, 2.0, 3.0, 4.0]
    ]
    track = build_intent_track(
        duration_sec=4.0,
        shot_boundaries=[],
        shot_profiles=[],
        dense_faces=dense,
        active_speaker_events=[],
        sample_hz=5.0,
    )
    # 4s * 5Hz = 20 samples + possibly an endpoint inclusive sample.
    assert 19 <= len(track) <= 22
