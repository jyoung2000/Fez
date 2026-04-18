"""Fix 4 — active_speaker confident-vote fast path.

Verifies the new confident-vote acceptance logic by constructing a
minimal `build_speaker_to_slot_map` call with a face_registry + dense
face stream + transcript segments that encode an unambiguous speaker
-to-slot mapping. Two speakers with >>2x margin over runner-up should
BOTH be assigned — not one dropped to NONE because the other claimed
the same slot first.
"""

from dataclasses import dataclass, field

from backend.services.active_speaker import map_speakers_to_face_slots
from backend.services.face_registry import FaceRegistry, FaceSlot


@dataclass
class _FakeFace:
    nose_x: float = 50.0
    nose_y: float = 50.0
    width: float = 10.0
    height: float = 12.0
    identity_id: int = -1
    lip_aperture: float = 0.0
    is_human: bool = True


@dataclass
class _FakeDense:
    timestamp: float
    faces: list = field(default_factory=list)


@dataclass
class _FakeSeg:
    start: float
    end: float
    speaker: str
    text: str = ""


def _panel_registry():
    return FaceRegistry(
        slots=[
            FaceSlot(slot_id=0, x_center=23.0, x_min=19.0, x_max=27.0,
                     frame_count=80, avg_width=10.0, avg_height=12.0),
            FaceSlot(slot_id=1, x_center=56.0, x_min=52.0, x_max=60.0,
                     frame_count=80, avg_width=10.0, avg_height=12.0),
        ],
        total_frames=160,
        frames_with_faces=160,
    )


def test_confident_vote_does_not_drop_speaker_to_none():
    # Speaker A talks for t=0..10 while a face at x=23 shows high lip
    # motion. Speaker B ALSO talks for t=0..10 with the same face at
    # x=23 showing high lip motion — pathological construction where
    # both speakers' top vote is the same slot.
    #
    # With the pre-Fix-4 greedy logic: first speaker claims slot 0,
    # second gets NONE because their only vote was for slot 0.
    # With Fix 4: both speakers have a confident top vote (>3s,
    # >2x runner-up) so both get assigned to slot 0.
    registry = _panel_registry()

    dense = []
    # Frames 0..20 at 0.5s intervals — both faces always visible,
    # with the LEFT face (slot 0) having high lip aperture to
    # attract both speakers' votes.
    for i in range(21):
        dense.append(_FakeDense(
            timestamp=i * 0.5,
            faces=[
                _FakeFace(nose_x=23.0, lip_aperture=0.8, identity_id=0),
                _FakeFace(nose_x=56.0, lip_aperture=0.02, identity_id=1),
            ],
        ))

    transcript = [
        _FakeSeg(start=0.0, end=5.0, speaker="SPEAKER_00"),
        _FakeSeg(start=5.0, end=10.0, speaker="SPEAKER_01"),
    ]

    mapping = map_speakers_to_face_slots(
        transcript, registry, dense,
    )

    # Both speakers assigned (not NONE).
    assert "SPEAKER_00" in mapping, f"speaker 00 dropped: {mapping}"
    assert "SPEAKER_01" in mapping, f"speaker 01 dropped: {mapping}"
    # Both should land on slot 0 given the lip-aperture signal —
    # the spec explicitly allows shared slots on confident votes.
    assert mapping["SPEAKER_00"] == 0
    assert mapping["SPEAKER_01"] == 0
