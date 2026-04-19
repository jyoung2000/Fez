"""Fix 3.6: unassigned-slot speaker events (slot_id == -1) no longer
silently fall through to "largest face". ``faces_from_dense`` picks
the face with highest lip-aperture, then audio_peak_x, then largest.
"""
from dataclasses import dataclass, field

from backend.services.camera_path_2d import faces_from_dense


@dataclass
class _Face:
    nose_x: float
    nose_y: float = 50.0
    width: float = 10.0
    height: float = 12.0
    identity_id: int = -1
    is_human: bool = True
    lip_aperture: float = 0.0
    yaw: float = 0.0


@dataclass
class _FrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)


@dataclass
class _SpeakerEvent:
    start: float
    end: float
    slot_id: int
    audio_peak_x: float = None  # optional


def _make_dense_two_faces():
    return [
        _FrameFaces(
            timestamp=t,
            faces=[
                _Face(nose_x=30.0, width=12.0, height=12.0, lip_aperture=0.0),
                _Face(nose_x=70.0, width=10.0, height=11.0, lip_aperture=0.0),
            ],
        )
        for t in [0.0, 1.0, 2.0, 3.0]
    ]


class TestUnassignedSlotRouting:
    def test_audio_peak_x_selects_nearest_face(self):
        """Spec scenario: two faces at x=0.3 and x=0.7. Speaker event
        slot_id=-1 with audio_peak_x=0.7. The face at x=0.7 must be
        chosen — not the largest (which is at x=0.3).
        """
        dense = _make_dense_two_faces()
        events = [_SpeakerEvent(start=0, end=4, slot_id=-1, audio_peak_x=0.7)]
        out = faces_from_dense(dense, active_speaker_events=events)
        # All 4 frames should pick the x=0.7 face.
        for ff2 in out:
            assert ff2 is not None
            # nose_x normalized to 0..1; x=0.7 face → 0.7
            assert abs(ff2.nose_x - 0.70) < 0.01, (
                f"expected face at x=0.70, got {ff2.nose_x}"
            )

    def test_lip_aperture_wins_over_audio_hint(self):
        """When a face has lip_aperture >= 0.05, it wins even if
        audio_peak_x points elsewhere. Lip motion is per-frame ground
        truth; audio_peak_x is a coarse hint."""
        dense = [
            _FrameFaces(
                timestamp=0.0,
                faces=[
                    # Large face with closed lips at x=30
                    _Face(nose_x=30.0, width=14.0, height=14.0,
                          lip_aperture=0.0),
                    # Smaller face at x=70 actively speaking
                    _Face(nose_x=70.0, width=10.0, height=11.0,
                          lip_aperture=0.12),
                ],
            ),
        ]
        events = [_SpeakerEvent(start=0, end=1, slot_id=-1,
                                audio_peak_x=0.3)]  # wrong hint
        out = faces_from_dense(dense, active_speaker_events=events)
        assert out[0].nose_x == 0.70  # lip-motion face wins

    def test_fallback_to_largest_when_no_lip_and_no_audio(self):
        """No lip motion, no audio_peak_x → largest-face fallback."""
        dense = _make_dense_two_faces()  # lip_aperture=0, no audio hint
        events = [_SpeakerEvent(start=0, end=4, slot_id=-1)]
        out = faces_from_dense(dense, active_speaker_events=events)
        # Largest is at x=30 (width=12 vs 10) → 0.30
        for ff2 in out:
            assert ff2.nose_x == 0.30

    def test_resolved_slot_still_takes_priority_over_unresolved_path(self):
        """Baseline: a resolved slot_id uses identity_id matching the
        usual way. Fix 3.6 only kicks in for slot_id == -1."""
        dense = [
            _FrameFaces(
                timestamp=0.0,
                faces=[
                    _Face(nose_x=30.0, width=10.0, height=10.0,
                          identity_id=0, lip_aperture=0.0),
                    _Face(nose_x=70.0, width=10.0, height=10.0,
                          identity_id=1, lip_aperture=0.2),
                ],
            ),
        ]
        # Resolved slot=0. Even though slot 1 has big lip motion, slot 0 wins.
        events = [_SpeakerEvent(start=0, end=1, slot_id=0)]
        out = faces_from_dense(dense, active_speaker_events=events)
        assert out[0].nose_x == 0.30

    def test_no_speaker_event_still_uses_largest_face(self):
        """No events at all → largest face (pre-3.6 behavior preserved)."""
        dense = _make_dense_two_faces()
        out = faces_from_dense(dense, active_speaker_events=None)
        for ff2 in out:
            # Face at x=30 (width=12) is largest.
            assert ff2.nose_x == 0.30
