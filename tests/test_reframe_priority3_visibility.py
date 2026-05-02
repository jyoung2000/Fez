"""Regression tests for the visibility gate on Priority 3
(``_dense_face_dominant_slot``) in the reframe segmenter.

The segmenter previously crowned whichever slot had the most aggregate
frames in an interval, even when the dominant slot had zero frames in
the *current* time window. The Tank vs Tyrese Verzuz panel run anchored
to the center seat for ~30% of the timeline because two adjacent faces
overlapped into that x-bin during banter, even when the actual talker
sat at a different slot. The visibility gate (mirroring Priority 1/2)
forces fall-through when the dominant slot isn't visible in the
interval.
"""
from dataclasses import dataclass
from typing import Optional

from backend.services.reframe_segmenter import (
    _dense_face_dominant_slot,
    _resolve_slot_for_interval,
)
from backend.services.face_registry import FaceRegistry, FaceSlot


@dataclass
class _Face:
    identity_id: int
    nose_x: float = 50.0
    nose_y: float = 40.0
    width: float = 15.0
    height: float = 20.0
    is_human: bool = True
    yaw: Optional[float] = None
    lip_aperture: float = 0.0


@dataclass
class _FrameFaces:
    timestamp: float
    faces: list


@dataclass
class _Ev:
    start: float
    end: float
    slot_id: int
    on_screen: bool = True


def _make_dense_faces_for_slots(slot_timeline: dict[int, list[float]]) -> list:
    """Build _FrameFaces from a {slot_id: [timestamps]} mapping."""
    by_ts: dict[float, list[_Face]] = {}
    for sid, times in slot_timeline.items():
        for t in times:
            by_ts.setdefault(t, []).append(_Face(identity_id=sid))
    return [
        _FrameFaces(timestamp=ts, faces=faces)
        for ts, faces in sorted(by_ts.items())
    ]


def _registry_with_slots(slot_ids: list[int]) -> FaceRegistry:
    return FaceRegistry(slots=[
        FaceSlot(
            slot_id=sid, x_center=20.0 + sid * 15.0,
            x_min=15.0 + sid * 15.0, x_max=25.0 + sid * 15.0,
            frame_count=100, avg_width=12.0, avg_height=18.0,
        )
        for sid in slot_ids
    ])


def test_priority3_falls_through_when_dominant_slot_invisible():
    """Slot 2 dominates aggregate (frames at t<10) but is invisible at
    [10, 12]; segmenter should fall through to Priority 4 / wide instead
    of returning slot 2.
    """
    dense = _make_dense_faces_for_slots({
        2: [t / 10.0 for t in range(0, 100)],          # 0.0..9.9
        4: [10.0, 10.5, 11.0, 11.5, 12.0],
    })
    registry = _registry_with_slots([2, 4])

    slot, _conf, _layout, _source = _resolve_slot_for_interval(
        start=10.0, end=12.0,
        transcript_segments=[], speaker_to_slot={},
        active_speaker_events=[],
        dense_faces=dense,
        face_registry=registry,
    )
    assert slot != 2, (
        "Priority 3 must NOT return slot 2 when slot 2 has zero face frames "
        "in [10, 12]; got slot=%r" % (slot,)
    )


def test_priority3_returns_dominant_when_visible():
    """Sanity: when the dominant slot is visible and not contested, Priority
    3 still wins."""
    dense = _make_dense_faces_for_slots({
        2: [10.0 + i * 0.1 for i in range(10)],
    })
    registry = _registry_with_slots([2])
    slot, _conf, _layout, source = _resolve_slot_for_interval(
        start=10.0, end=11.0,
        transcript_segments=[], speaker_to_slot={},
        active_speaker_events=[],
        dense_faces=dense, face_registry=registry,
    )
    assert slot == 2
    assert source == "dense_face_dominant"


def test_dense_face_dominant_slot_accepts_active_speaker_events():
    """The new ``active_speaker_events`` keyword should be accepted without
    altering the answer in the simple no-contest case.
    """
    dense = _make_dense_faces_for_slots({
        2: [10.0 + i * 0.1 for i in range(10)],
    })
    registry = _registry_with_slots([2])
    # Same slot wins with or without the active-speaker hint.
    base = _dense_face_dominant_slot(10.0, 11.0, dense, registry)
    with_events = _dense_face_dominant_slot(
        10.0, 11.0, dense, registry,
        active_speaker_events=[_Ev(start=10.0, end=11.0, slot_id=2)],
    )
    assert base == 2 == with_events
