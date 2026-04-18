"""Test that SpeakerEvents are marked on_screen=False when a shot cut
lands inside a transcript segment and the new shot has no active lips
(classic reaction / over-the-shoulder framing).
"""

from dataclasses import dataclass, field

from backend.services.active_speaker import (
    SpeakerEvent,
    _apply_shot_reverse_tolerance,
)


@dataclass
class _Face:
    nose_x: float = 50.0
    nose_y: float = 50.0
    width: float = 10.0
    height: float = 12.0
    identity_id: int = 0
    lip_aperture: float = 0.0
    is_human: bool = True
    is_speaking: bool = False


@dataclass
class _FrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)


@dataclass
class _Segment:
    start: float
    end: float


def test_offscreen_speaker_persists_across_shot_cut():
    """Shot 1 has speaker (slot 0) talking with open lips. Shot 2 starts
    at t=2.0 showing the listener (slot 1) with closed lips, while the
    transcript segment runs 0.0 → 4.0. The SpeakerEvent should keep
    slot_id=0 and become on_screen=False after the cut.
    """
    # Before the cut: slot 0 has open lips (is speaking)
    face_results = [
        _FrameFaces(timestamp=0.5, faces=[
            _Face(identity_id=0, lip_aperture=0.08, nose_x=30),
        ]),
        _FrameFaces(timestamp=1.0, faces=[
            _Face(identity_id=0, lip_aperture=0.09, nose_x=30),
        ]),
        _FrameFaces(timestamp=1.5, faces=[
            _Face(identity_id=0, lip_aperture=0.06, nose_x=30),
        ]),
        # After the cut: only slot 1 visible, lips closed (reaction shot)
        _FrameFaces(timestamp=2.5, faces=[
            _Face(identity_id=1, lip_aperture=0.005, nose_x=70),
        ]),
        _FrameFaces(timestamp=3.0, faces=[
            _Face(identity_id=1, lip_aperture=0.005, nose_x=70),
        ]),
        _FrameFaces(timestamp=3.5, faces=[
            _Face(identity_id=1, lip_aperture=0.010, nose_x=70),
        ]),
    ]
    transcript = [_Segment(start=0.0, end=4.0)]
    events = [SpeakerEvent(start=0.0, end=4.0, slot_id=0, confidence=0.8)]
    shot_cuts = [2.0]

    out, offscreen = _apply_shot_reverse_tolerance(
        events, transcript, face_results, shot_cuts,
    )

    assert offscreen == 1
    assert len(out) == 1
    ev = out[0]
    # Speaker slot should still be 0 (the actual speaker off-screen)
    assert ev.slot_id == 0
    # Speaker should be marked off-screen now
    assert ev.on_screen is False


def test_onscreen_speaker_when_post_cut_shot_still_has_talking_lips():
    """Control case: if the second shot still has a face with open lips,
    the speaker must remain on_screen=True (normal within-shot tracking).
    """
    face_results = [
        _FrameFaces(timestamp=0.5, faces=[
            _Face(identity_id=0, lip_aperture=0.08, nose_x=30),
        ]),
        _FrameFaces(timestamp=2.5, faces=[
            _Face(identity_id=0, lip_aperture=0.08, nose_x=30),
        ]),
    ]
    transcript = [_Segment(start=0.0, end=4.0)]
    events = [SpeakerEvent(start=0.0, end=4.0, slot_id=0, confidence=0.8)]
    shot_cuts = [2.0]

    out, offscreen = _apply_shot_reverse_tolerance(
        events, transcript, face_results, shot_cuts,
    )
    assert offscreen == 0
    assert out[0].on_screen is True
