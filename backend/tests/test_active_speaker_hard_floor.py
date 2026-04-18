"""Test that the active speaker's RequiredRegion weight is hard-floored
to at least 1.5x the maximum passive-face weight in the same frame.

Bug 5 from the Bungo Stray Dogs run: the L1 solver's data term picked
a large non-speaking face over a smaller speaking face because the
weight gap was too narrow. This regression test pins the hard floor
invariant so the solver physically cannot pick the wrong face.
"""

from dataclasses import dataclass, field

from backend.services.active_speaker import SpeakerEvent
from backend.services.required_regions import build_required_regions


@dataclass
class _Face:
    nose_x: float
    nose_y: float
    width: float
    height: float
    identity_id: int = 0
    lip_aperture: float = 0.0
    is_human: bool = True
    is_speaking: bool = False


@dataclass
class _FrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)


def test_active_speaker_weight_is_hard_floored():
    """Slot A is the active speaker with a small bbox (width=4).
    Slot B is passive with a much larger bbox (width=12). Without the
    hard floor, A has weight=1.4 and B has weight=0.45, so the ratio
    is 3.11 — already fine. But if B were in a 'no active speaker in
    frame' branch it would have weight=0.8 and the 1.4/0.8 ratio=1.75
    is too small for the L1 solver under area weighting. The hard
    floor `weight >= 1.5 * max_other_face_weight` must guarantee the
    invariant regardless of branch.
    """
    # Two faces in the same frame, slot 0 is active speaker.
    ff = _FrameFaces(timestamp=1.0, faces=[
        _Face(nose_x=30, nose_y=50, width=4, height=5, identity_id=0),
        _Face(nose_x=70, nose_y=50, width=12, height=15, identity_id=1),
    ])
    events = [SpeakerEvent(start=0.0, end=10.0, slot_id=0, confidence=0.9)]

    regs_per_frame = build_required_regions(
        frame_faces=[ff],
        active_speaker_events=events,
    )
    face_regions = [r for r in regs_per_frame[0] if r.source == "face"]
    active = [r for r in face_regions if r.is_active_speaker]
    passive = [r for r in face_regions if not r.is_active_speaker]

    assert len(active) == 1
    assert len(passive) == 1

    max_passive = max(r.weight for r in passive)
    active_weight = active[0].weight

    assert active_weight >= 1.5 * max_passive, (
        f"Hard floor violated: active_weight={active_weight:.3f}, "
        f"max_passive={max_passive:.3f}, ratio={active_weight / max_passive:.2f}"
    )


def test_hard_floor_fires_even_when_default_weights_would_be_too_close():
    """Synthesize a case where the base passive weight is 0.8 (no other
    speaker detected in this branch would not apply here, so passive
    becomes 0.45 — test the floor anyway by inspecting the ratio).
    """
    ff = _FrameFaces(timestamp=1.0, faces=[
        _Face(nose_x=30, nose_y=50, width=10, height=12, identity_id=0),
        _Face(nose_x=50, nose_y=50, width=10, height=12, identity_id=1),
        _Face(nose_x=70, nose_y=50, width=10, height=12, identity_id=2),
    ])
    events = [SpeakerEvent(start=0.0, end=10.0, slot_id=1, confidence=0.9)]

    regs_per_frame = build_required_regions(
        frame_faces=[ff],
        active_speaker_events=events,
    )
    face_regs = [r for r in regs_per_frame[0] if r.source == "face"]
    active = [r for r in face_regs if r.is_active_speaker]
    passive = [r for r in face_regs if not r.is_active_speaker]
    assert len(active) == 1
    assert len(passive) == 2

    max_passive_w = max(r.weight for r in passive)
    assert active[0].weight >= 1.5 * max_passive_w


def test_hard_floor_noop_when_no_passive_in_frame():
    """Single-face frame: no floor adjustment needed, active weight
    stays at its base value 1.4."""
    ff = _FrameFaces(timestamp=1.0, faces=[
        _Face(nose_x=50, nose_y=50, width=10, height=12, identity_id=0),
    ])
    events = [SpeakerEvent(start=0.0, end=10.0, slot_id=0, confidence=0.9)]

    regs_per_frame = build_required_regions(
        frame_faces=[ff],
        active_speaker_events=events,
    )
    face_regs = [r for r in regs_per_frame[0] if r.source == "face"]
    assert len(face_regs) == 1
    assert face_regs[0].is_active_speaker is True
    assert abs(face_regs[0].weight - 1.4) < 1e-6
