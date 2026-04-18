"""Test that lead-room bias shifts the active-speaker cx in the direction
they're looking (negative of yaw sign), only when |yaw| > 15°, and only
for the active speaker.
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
    yaw: float = 0.0


@dataclass
class _FrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)


def _find_face_region(regions):
    for r in regions:
        if r.source == "face":
            return r
    return None


def test_positive_yaw_shifts_cx_left():
    """Active speaker at cx=0.5 with yaw=+30° should shift cx by -0.05
    (looking screen-left by convention → room on the left)."""
    ff = _FrameFaces(timestamp=1.0, faces=[
        _Face(nose_x=50, nose_y=50, width=20, height=25, identity_id=0, yaw=30.0),
    ])
    events = [SpeakerEvent(start=0.0, end=10.0, slot_id=0, confidence=0.9)]
    regs_per_frame = build_required_regions(
        frame_faces=[ff],
        active_speaker_events=events,
    )
    face_region = _find_face_region(regs_per_frame[0])
    assert face_region is not None
    assert face_region.is_active_speaker is True
    assert abs(face_region.cx - 0.45) < 1e-6, f"cx={face_region.cx} expected 0.45"


def test_negative_yaw_shifts_cx_right():
    """yaw=-30° → shift +0.05 (screen-right)."""
    ff = _FrameFaces(timestamp=1.0, faces=[
        _Face(nose_x=50, nose_y=50, width=20, height=25, identity_id=0, yaw=-30.0),
    ])
    events = [SpeakerEvent(start=0.0, end=10.0, slot_id=0, confidence=0.9)]
    regs_per_frame = build_required_regions(
        frame_faces=[ff],
        active_speaker_events=events,
    )
    face_region = _find_face_region(regs_per_frame[0])
    assert abs(face_region.cx - 0.55) < 1e-6


def test_small_yaw_does_not_shift():
    """|yaw|=10° < 15° threshold → no shift."""
    ff = _FrameFaces(timestamp=1.0, faces=[
        _Face(nose_x=50, nose_y=50, width=20, height=25, identity_id=0, yaw=10.0),
    ])
    events = [SpeakerEvent(start=0.0, end=10.0, slot_id=0, confidence=0.9)]
    regs_per_frame = build_required_regions(
        frame_faces=[ff],
        active_speaker_events=events,
    )
    face_region = _find_face_region(regs_per_frame[0])
    assert abs(face_region.cx - 0.5) < 1e-6


def test_passive_face_not_shifted():
    """Lead-room bias only applies to the active speaker, not passive faces."""
    ff = _FrameFaces(timestamp=1.0, faces=[
        _Face(nose_x=30, nose_y=50, width=15, height=20, identity_id=0, yaw=0.0),
        # Passive face with a big yaw — must NOT be shifted.
        _Face(nose_x=70, nose_y=50, width=15, height=20, identity_id=1, yaw=40.0),
    ])
    events = [SpeakerEvent(start=0.0, end=10.0, slot_id=0, confidence=0.9)]
    regs_per_frame = build_required_regions(
        frame_faces=[ff],
        active_speaker_events=events,
    )
    face_regions = [r for r in regs_per_frame[0] if r.source == "face"]
    passive = [r for r in face_regions if not r.is_active_speaker]
    assert len(passive) == 1
    assert abs(passive[0].cx - 0.7) < 1e-6, (
        f"Passive face cx should be unchanged at 0.7, got {passive[0].cx}"
    )


def test_missing_yaw_attribute_does_not_crash():
    """Faces without a `yaw` attr must be handled silently."""

    class _FaceNoYaw:
        nose_x = 50.0
        nose_y = 50.0
        width = 20.0
        height = 25.0
        identity_id = 0
        lip_aperture = 0.0
        is_human = True
        is_speaking = False

    ff = _FrameFaces(timestamp=1.0, faces=[_FaceNoYaw()])
    events = [SpeakerEvent(start=0.0, end=10.0, slot_id=0, confidence=0.9)]
    regs_per_frame = build_required_regions(
        frame_faces=[ff],
        active_speaker_events=events,
    )
    face_region = _find_face_region(regs_per_frame[0])
    assert face_region is not None
    assert abs(face_region.cx - 0.5) < 1e-6
