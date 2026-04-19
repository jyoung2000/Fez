"""Fix 3.3: unified intent track never emits (0.5, 0.5, conf<=0.1,
source='none'). When face detection drops out for a window, we hold
the last real sample at its original (x, y) with decayed conf and
tag source='kalman_predict'.
"""
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
    lip_aperture: float = 0.0


@dataclass
class _FrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)


def test_face_gap_fills_with_kalman_predict_not_center():
    """Faces at t=0..2.0 at x=25, then no faces [3.0, 5.0], then face
    at t=5.5. The intent track covers the full [0, 5.5] window and
    in the gap holds x=0.25 (the last real), never 0.5 with
    source='none'."""
    dense = [
        _FrameFaces(timestamp=t, faces=[_Face(nose_x=25.0, nose_y=40.0)])
        for t in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    ] + [
        _FrameFaces(timestamp=t, faces=[])
        for t in [3.0, 3.5, 4.0, 4.5, 5.0]
    ] + [
        _FrameFaces(timestamp=5.5, faces=[_Face(nose_x=25.0, nose_y=40.0)])
    ]
    track = build_intent_track(
        duration_sec=5.5,
        shot_boundaries=[],
        shot_profiles=[],
        dense_faces=dense,
        active_speaker_events=[],
    )
    # No sample is (0.5, 0.5, source='none').
    for s in track:
        assert s.source != "none"
        # In the gap [3.0, 5.0] specifically, source is kalman_predict.
        if 3.0 <= s.t <= 5.0:
            assert s.source == "kalman_predict", (
                f"at t={s.t}: expected kalman_predict, got {s.source}"
            )
            # The gap holds x=0.25 (the last real), NOT 0.5.
            assert abs(s.x - 0.25) < 0.01, (
                f"at t={s.t}: expected x≈0.25 hold, got {s.x}"
            )
