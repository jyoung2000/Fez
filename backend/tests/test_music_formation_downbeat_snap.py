"""Week 2 Part E — formation → downbeat snap in reframe_segmenter Stage 11.

Tests the Pass C logic that lives inside
``backend/services/reframe_segmenter.py`` after the existing
boundary-snap (Pass A) and pulse-cut (Pass B) passes. The logic is
awkward to unit-test in isolation because it walks the same
``raw_segments`` list the boundary-snap pass produced, so we re-
implement the exact Pass C algorithm here as a testable helper and
pin it with round-trip assertions. If the in-code copy drifts,
update both sides.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest


# ───────────────────────── Stubs ─────────────────────────


@dataclass
class _StubFace:
    nose_x: float


@dataclass
class _StubFrameFaces:
    timestamp: float
    faces: list[_StubFace] = field(default_factory=list)


@dataclass
class _StubSegment:
    start: float
    end: float


def _formation(ts: float) -> _StubFrameFaces:
    """3 faces spanning 60% width."""
    return _StubFrameFaces(
        timestamp=ts,
        faces=[_StubFace(nose_x=20), _StubFace(nose_x=50), _StubFace(nose_x=80)],
    )


def _tight(ts: float) -> _StubFrameFaces:
    """2 close faces → not a formation."""
    return _StubFrameFaces(
        timestamp=ts,
        faces=[_StubFace(nose_x=45), _StubFace(nose_x=55)],
    )


# ───────────────────────── Pass C replica ─────────────────────────


def _apply_formation_snap(
    raw_segments: list[_StubSegment],
    dense_faces: list[_StubFrameFaces],
    downbeats: list[float],
) -> int:
    """Replica of the Pass C formation → downbeat snap.

    Returns the number of formation snaps applied. Mutates
    ``raw_segments`` in place.
    """
    from backend.services.reframe_segmenter import _is_formation_frame

    faces_by_t: dict = {}
    for _df in dense_faces:
        faces_by_t[round(float(_df.timestamp), 2)] = _df

    snaps = 0
    for i, seg in enumerate(raw_segments[:-1]):
        end_probe_t = round(max(0.0, seg.end - 0.15), 2)
        fr = faces_by_t.get(end_probe_t)
        if fr is None or not _is_formation_frame(fr):
            continue
        next_db = None
        for db in downbeats:
            if db > seg.end + 0.05:
                next_db = db
                break
        if next_db is None or (next_db - seg.end) > 1.5:
            continue
        nxt = raw_segments[i + 1]
        if not (nxt.start < next_db < nxt.end):
            continue
        if (nxt.end - next_db) < 0.30:
            continue
        seg.end = float(next_db)
        nxt.start = float(next_db)
        snaps += 1
    return snaps


# ───────────────────────── tests ─────────────────────────


def test_formation_at_boundary_snaps_to_next_downbeat():
    """First segment ends at 2.0 with a formation frame at 1.85; downbeat at 2.25."""
    segs = [
        _StubSegment(start=0.0, end=2.0),
        _StubSegment(start=2.0, end=5.0),
    ]
    dense = [
        _tight(1.0),
        _tight(1.5),
        _formation(1.85),   # last probeable frame of seg[0]
        _tight(3.0),
    ]
    downbeats = [0.25, 1.25, 2.25, 3.25, 4.25]

    snaps = _apply_formation_snap(segs, dense, downbeats)

    assert snaps == 1
    assert segs[0].end == pytest.approx(2.25)
    assert segs[1].start == pytest.approx(2.25)
    # seg[1] end is untouched
    assert segs[1].end == 5.0


def test_formation_with_no_nearby_downbeat_does_not_snap():
    """Nearest downbeat is > 1.5s past seg.end → no snap."""
    segs = [
        _StubSegment(start=0.0, end=2.0),
        _StubSegment(start=2.0, end=5.0),
    ]
    dense = [_formation(1.85)]
    downbeats = [0.5, 4.0]  # 4.0 is 2.0s past seg[0].end → too far

    snaps = _apply_formation_snap(segs, dense, downbeats)

    assert snaps == 0
    assert segs[0].end == 2.0
    assert segs[1].start == 2.0


def test_non_formation_frame_does_not_snap():
    """A tight frame at the boundary → no snap, even when downbeat is near."""
    segs = [
        _StubSegment(start=0.0, end=2.0),
        _StubSegment(start=2.0, end=5.0),
    ]
    dense = [_tight(1.85)]  # NOT a formation
    downbeats = [2.25]

    snaps = _apply_formation_snap(segs, dense, downbeats)
    assert snaps == 0
    assert segs[0].end == 2.0


def test_snap_would_shrink_next_below_floor_rejected():
    """If the downbeat is too close to the next segment's end, reject."""
    segs = [
        _StubSegment(start=0.0, end=2.0),
        _StubSegment(start=2.0, end=2.50),  # only 0.5s long
    ]
    dense = [_formation(1.85)]
    # 2.25 is inside seg[1] but leaves only 2.50 - 2.25 = 0.25s < 0.30s floor
    downbeats = [2.25]

    snaps = _apply_formation_snap(segs, dense, downbeats)
    assert snaps == 0
    assert segs[0].end == 2.0


def test_downbeat_outside_next_segment_rejected():
    """Downbeat lands beyond seg[1].end → not inside seg[1] → reject."""
    segs = [
        _StubSegment(start=0.0, end=2.0),
        _StubSegment(start=2.0, end=2.20),
    ]
    dense = [_formation(1.85)]
    # 2.25 is after seg[1].end=2.20 → not inside
    downbeats = [2.25]

    snaps = _apply_formation_snap(segs, dense, downbeats)
    assert snaps == 0


def test_multiple_formations_snap_independently():
    """Two formations in a row, each gets its own snap."""
    segs = [
        _StubSegment(start=0.0, end=2.0),
        _StubSegment(start=2.0, end=4.0),
        _StubSegment(start=4.0, end=6.0),
    ]
    dense = [
        _formation(1.85),   # triggers snap at seg[0] boundary
        _formation(3.85),   # triggers snap at seg[1] boundary
    ]
    downbeats = [2.25, 4.25]

    snaps = _apply_formation_snap(segs, dense, downbeats)

    assert snaps == 2
    assert segs[0].end == pytest.approx(2.25)
    assert segs[1].start == pytest.approx(2.25)
    assert segs[1].end == pytest.approx(4.25)
    assert segs[2].start == pytest.approx(4.25)


def test_last_segment_never_snapped():
    """The final segment has no "next" to shrink — it must stay untouched."""
    segs = [
        _StubSegment(start=0.0, end=5.0),
    ]
    dense = [_formation(4.85)]
    downbeats = [5.25]

    snaps = _apply_formation_snap(segs, dense, downbeats)
    assert snaps == 0
    assert segs[0].end == 5.0


def test_probe_window_respects_0_15_offset():
    """Probe is at seg.end - 0.15; a formation frame only at seg.end - 0.5 is missed."""
    segs = [
        _StubSegment(start=0.0, end=2.0),
        _StubSegment(start=2.0, end=5.0),
    ]
    # Formation frame is at 1.5, which rounds to 1.5; probe is at 1.85.
    dense = [_formation(1.5), _tight(1.85)]
    downbeats = [2.25]

    snaps = _apply_formation_snap(segs, dense, downbeats)
    assert snaps == 0, "probe window at 1.85 sees _tight, not _formation at 1.5"


def test_formation_helper_used_here_matches_module():
    """Sanity: the _is_formation_frame we import is the real one."""
    from backend.services.reframe_segmenter import _is_formation_frame

    fr = _formation(0.0)
    assert _is_formation_frame(fr) is True

    fr2 = _tight(0.0)
    assert _is_formation_frame(fr2) is False
