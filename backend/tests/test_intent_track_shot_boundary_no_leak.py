"""Fix 3.3: intent track does not leak samples across shot cuts.

Spec: on a 2-shot clip where shot 1 has the subject at x=0.25 and
shot 2 has no detections until t = shot2_start + 2s, the first 2s of
shot 2 do NOT hold at x=0.25 — they backward-fill from the first
real detection in shot 2.
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


@dataclass
class _FrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)


def test_shot_boundary_does_not_leak_previous_x():
    """Shot 1 [0.0, 3.0]: face at x=25. Cut at 3.0.
    Shot 2 [3.0, 7.0]: no face until t=5.0 where face at x=75.

    Samples in [3.0, 5.0] must backward-fill from the x=75 detection,
    NOT hold at x=0.25 from shot 1.
    """
    dense = [
        # Shot 1 faces at x=25
        _FrameFaces(timestamp=t, faces=[_Face(nose_x=25.0)])
        for t in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    ] + [
        # Shot 2 faces start at t=5.0 at x=75
        _FrameFaces(timestamp=t, faces=[_Face(nose_x=75.0)])
        for t in [5.0, 5.5, 6.0, 6.5]
    ]
    track = build_intent_track(
        duration_sec=7.0,
        shot_boundaries=[3.0],
        shot_profiles=[],
        dense_faces=dense,
        active_speaker_events=[],
    )
    # Pick samples in [3.0, 5.0] — the "unresolved start of shot 2" window.
    in_gap = [s for s in track if 3.0 < s.t < 4.9]
    assert in_gap, "expected samples in the shot-2 backfill window"
    for s in in_gap:
        # Backfill from shot 2's first real detection (x=0.75).
        # Must NOT be 0.25 (the previous shot's hold value).
        assert s.shot_idx == 1
        assert abs(s.x - 0.75) < 0.01, (
            f"at t={s.t}: expected backfill from x≈0.75, got {s.x}"
        )
