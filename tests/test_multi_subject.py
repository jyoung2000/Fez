"""Tests for multi-subject reframe behavior.

Verifies that the reframe segmenter handles multi-face scenes correctly:
- Never averages distant face positions (the "empty couch" bug)
- Uses active speaker as tiebreaker for wide-spread faces
- Falls back to split screen when both speakers are active
- Only uses centroid when all faces fit in a single crop
"""

import sys
import os
from dataclasses import dataclass, field

# Ensure backend is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.services.face_registry import FaceSlot, FaceRegistry
from backend.services.reframe_segmenter import (
    build_reframe_segments,
    _resolve_slot_for_interval,
    WIDE_MASTER_X,
)


# ── Test fixtures ──

@dataclass
class FakeSpeakerEvent:
    start: float
    end: float
    slot_id: int
    confidence: float = 0.9


@dataclass
class FakeFaceInfo:
    x_center: float
    y_center: float = 50.0
    width: float = 8.0
    height: float = 10.0
    nose_x: float = 0.0
    nose_y: float = 50.0
    confidence: float = 0.95
    lip_aperture: float = 0.0
    identity_embedding: list = None
    identity_id: int = -1
    is_speaking: bool = False
    y_bottom: float = 60.0
    x: float = 0.0

    def __post_init__(self):
        if self.nose_x == 0.0:
            self.nose_x = self.x_center
        if self.x == 0.0:
            self.x = self.x_center


@dataclass
class FakeFrameFaces:
    timestamp: float
    frame_path: str = ""
    faces: list = field(default_factory=list)
    primary_face_idx: int = 0


@dataclass
class FakeTranscriptSegment:
    start: float
    end: float
    text: str = ""
    speaker: str = ""
    confidence: float = 1.0


def make_registry_2_speakers(x_left=22.0, x_right=78.0):
    """Two speakers wide apart (like Tank vs Tyrese couch interview)."""
    return FaceRegistry(
        slots=[
            FaceSlot(slot_id=0, x_center=x_left, x_min=x_left - 3, x_max=x_left + 3,
                     frame_count=50, avg_width=8.0, avg_height=10.0),
            FaceSlot(slot_id=1, x_center=x_right, x_min=x_right - 3, x_max=x_right + 3,
                     frame_count=52, avg_width=8.0, avg_height=10.0),
        ],
        total_frames=60,
        frames_with_faces=55,
    )


def make_registry_2_close(x_left=42.0, x_right=58.0):
    """Two speakers close together (both fit in one 9:16 crop)."""
    return FaceRegistry(
        slots=[
            FaceSlot(slot_id=0, x_center=x_left, x_min=x_left - 2, x_max=x_left + 2,
                     frame_count=50, avg_width=8.0, avg_height=10.0),
            FaceSlot(slot_id=1, x_center=x_right, x_min=x_right - 2, x_max=x_right + 2,
                     frame_count=50, avg_width=8.0, avg_height=10.0),
        ],
        total_frames=60,
        frames_with_faces=55,
    )


def make_registry_3_speakers():
    """Three speakers spread across the frame."""
    return FaceRegistry(
        slots=[
            FaceSlot(slot_id=0, x_center=15.0, x_min=12.0, x_max=18.0,
                     frame_count=40, avg_width=7.0, avg_height=9.0),
            FaceSlot(slot_id=1, x_center=50.0, x_min=47.0, x_max=53.0,
                     frame_count=45, avg_width=7.0, avg_height=9.0),
            FaceSlot(slot_id=2, x_center=85.0, x_min=82.0, x_max=88.0,
                     frame_count=42, avg_width=7.0, avg_height=9.0),
        ],
        total_frames=60,
        frames_with_faces=55,
    )


def make_dense_faces_2speakers(start=0.0, end=30.0, x_left=22.0, x_right=78.0, step=1.0):
    """Dense face frames with two speakers visible in every frame."""
    frames = []
    t = start
    while t <= end:
        frames.append(FakeFrameFaces(
            timestamp=t,
            faces=[
                FakeFaceInfo(x_center=x_left, identity_id=0, nose_x=x_left, x=x_left),
                FakeFaceInfo(x_center=x_right, identity_id=1, nose_x=x_right, x=x_right),
            ],
        ))
        t += step
    return frames


def make_dense_faces_3speakers(start=0.0, end=30.0, step=1.0):
    """Dense face frames with three speakers."""
    frames = []
    t = start
    while t <= end:
        frames.append(FakeFrameFaces(
            timestamp=t,
            faces=[
                FakeFaceInfo(x_center=15.0, identity_id=0, nose_x=15.0, x=15.0),
                FakeFaceInfo(x_center=50.0, identity_id=1, nose_x=50.0, x=50.0),
                FakeFaceInfo(x_center=85.0, identity_id=2, nose_x=85.0, x=85.0),
            ],
        ))
        t += step
    return frames


# ── Test 1: Two faces wide-spread, single active speaker → CROP on active speaker ──

def test_two_faces_wide_single_active_speaker():
    """When one speaker is active and faces are far apart, crop on the active one."""
    registry = make_registry_2_speakers()
    dense = make_dense_faces_2speakers()
    # Speaker 0 (left, x=22) is active for the entire segment
    speaker_events = [FakeSpeakerEvent(start=0.0, end=30.0, slot_id=0)]
    transcript = [FakeTranscriptSegment(start=0.0, end=30.0, speaker="Speaker A")]
    speaker_to_slot = {"Speaker A": 0}

    segments = build_reframe_segments(
        shot_cuts=[],
        face_registry=registry,
        active_speaker_events=speaker_events,
        dense_faces=dense,
        transcript_segments=transcript,
        speaker_to_slot=speaker_to_slot,
        video_duration=30.0,
    )

    assert len(segments) >= 1
    seg = segments[0]
    # Must be on the active speaker (slot 0, x~22), NOT centroid (x~50)
    assert abs(seg.subject_x - 22) < 10, f"Expected ~22, got {seg.subject_x}"
    assert seg.subject_source != "face_registry_centroid", f"Bug: centroid path used"
    assert seg.subject_source != "hardcoded_center", f"Bug: hardcoded center fallback"
    assert seg.layout == "single"


# ── Test 2: Two faces wide-spread, both speaking → SPLIT_SCREEN ──

def test_two_faces_wide_both_speaking():
    """When both speakers are active simultaneously and far apart, split screen."""
    registry = make_registry_2_speakers()
    dense = make_dense_faces_2speakers()
    # Both speakers active simultaneously
    speaker_events = [
        FakeSpeakerEvent(start=0.0, end=30.0, slot_id=0),
        FakeSpeakerEvent(start=0.0, end=30.0, slot_id=1),
    ]
    transcript = []
    speaker_to_slot = {}

    segments = build_reframe_segments(
        shot_cuts=[],
        face_registry=registry,
        active_speaker_events=speaker_events,
        dense_faces=dense,
        transcript_segments=transcript,
        speaker_to_slot=speaker_to_slot,
        video_duration=30.0,
    )

    assert len(segments) >= 1
    for seg in segments:
        # Should never land on x=50 (empty couch) with distant faces
        if seg.layout == "single":
            assert abs(seg.subject_x - 50) > 10 or seg.subject_source != "face_registry_centroid", \
                f"Bug: segment at {seg.start:.1f}s has centroid x={seg.subject_x}"


# ── Test 3: Two faces wide-spread, neither speaking → picks most visible or blur fill ──

def test_two_faces_wide_neither_speaking():
    """When neither speaks, use most recently active speaker or dominant face."""
    registry = make_registry_2_speakers()
    dense = make_dense_faces_2speakers()
    speaker_events = []  # No active speaker data
    transcript = []
    speaker_to_slot = {}

    segments = build_reframe_segments(
        shot_cuts=[],
        face_registry=registry,
        active_speaker_events=speaker_events,
        dense_faces=dense,
        transcript_segments=transcript,
        speaker_to_slot=speaker_to_slot,
        video_duration=30.0,
    )

    assert len(segments) >= 1
    for seg in segments:
        # Must NOT be centroid at x=50 between the two speakers
        if seg.layout == "single":
            assert abs(seg.subject_x - 22) < 10 or abs(seg.subject_x - 78) < 10, \
                f"Bug: subject_x={seg.subject_x} is between speakers (expected near 22 or 78)"


# ── Test 4: Two faces close together → centroid CROP (legitimate case) ──

def test_two_faces_close_centroid_crop():
    """When both faces fit in one crop, centroid is the correct answer."""
    registry = make_registry_2_close()
    dense = make_dense_faces_2speakers(x_left=42.0, x_right=58.0)
    speaker_events = [FakeSpeakerEvent(start=0.0, end=30.0, slot_id=0)]
    transcript = [FakeTranscriptSegment(start=0.0, end=30.0, speaker="Speaker A")]
    speaker_to_slot = {"Speaker A": 0}

    segments = build_reframe_segments(
        shot_cuts=[],
        face_registry=registry,
        active_speaker_events=speaker_events,
        dense_faces=dense,
        transcript_segments=transcript,
        speaker_to_slot=speaker_to_slot,
        video_duration=30.0,
    )

    assert len(segments) >= 1
    seg = segments[0]
    # Both faces fit in one crop, so subject_x should be near one or between them
    # (either the active speaker slot or centroid is fine since they fit)
    assert 35 <= seg.subject_x <= 65, f"Expected x near 42-58 range, got {seg.subject_x}"


# ── Test 5: Three faces wide-spread, one active → CROP on active speaker ──

def test_three_faces_wide_single_active():
    """Three faces, one active: crop on the active one."""
    registry = make_registry_3_speakers()
    dense = make_dense_faces_3speakers()
    # Speaker 2 (right, x=85) is active
    speaker_events = [FakeSpeakerEvent(start=0.0, end=30.0, slot_id=2)]
    transcript = [FakeTranscriptSegment(start=0.0, end=30.0, speaker="Speaker C")]
    speaker_to_slot = {"Speaker C": 2}

    segments = build_reframe_segments(
        shot_cuts=[],
        face_registry=registry,
        active_speaker_events=speaker_events,
        dense_faces=dense,
        transcript_segments=transcript,
        speaker_to_slot=speaker_to_slot,
        video_duration=30.0,
    )

    assert len(segments) >= 1
    seg = segments[0]
    assert abs(seg.subject_x - 85) < 10, f"Expected ~85 (active speaker), got {seg.subject_x}"


# ── Test 6: Three faces wide-spread, all speaking → GRID or WIDE ──

def test_three_faces_wide_all_speaking():
    """Three faces all active: should use grid or wide, not centroid."""
    registry = make_registry_3_speakers()
    dense = make_dense_faces_3speakers()
    speaker_events = [
        FakeSpeakerEvent(start=0.0, end=30.0, slot_id=0),
        FakeSpeakerEvent(start=0.0, end=30.0, slot_id=1),
        FakeSpeakerEvent(start=0.0, end=30.0, slot_id=2),
    ]
    transcript = []
    speaker_to_slot = {}

    segments = build_reframe_segments(
        shot_cuts=[],
        face_registry=registry,
        active_speaker_events=speaker_events,
        dense_faces=dense,
        transcript_segments=transcript,
        speaker_to_slot=speaker_to_slot,
        video_duration=30.0,
    )

    assert len(segments) >= 1
    for seg in segments:
        # With 3 speakers all active, should not be a simple single crop at centroid
        # unless the layout explicitly handles it
        if seg.layout == "single" and seg.subject_source == "face_registry_centroid":
            assert False, f"Bug: centroid averaging with 3 wide-spread faces at t={seg.start:.1f}s"


# ── Test 7: Tank vs Tyrese regression ──

def test_tank_vs_tyrese_regression():
    """Regression test for the Tank vs Tyrese couch interview at t=15.85s.

    Two faces at x~22% and x~78%, clearly detected. The crop must NOT
    land on x=50 (the empty couch between them).
    """
    registry = make_registry_2_speakers(x_left=22.0, x_right=78.0)
    dense = make_dense_faces_2speakers(start=14.0, end=18.0, step=0.5)

    # Simulate: speaker 0 is active around t=15.85
    speaker_events = [
        FakeSpeakerEvent(start=14.0, end=16.0, slot_id=0),
        FakeSpeakerEvent(start=16.0, end=18.0, slot_id=1),
    ]
    transcript = [
        FakeTranscriptSegment(start=14.0, end=16.0, speaker="Tank", confidence=0.9),
        FakeTranscriptSegment(start=16.0, end=18.0, speaker="Tyrese", confidence=0.9),
    ]
    speaker_to_slot = {"Tank": 0, "Tyrese": 1}

    segments = build_reframe_segments(
        shot_cuts=[],
        face_registry=registry,
        active_speaker_events=speaker_events,
        dense_faces=dense,
        transcript_segments=transcript,
        speaker_to_slot=speaker_to_slot,
        video_duration=18.0,
    )

    # Find the segment covering t=15.85
    target_seg = None
    for seg in segments:
        if seg.start <= 15.85 <= seg.end:
            target_seg = seg
            break

    assert target_seg is not None, "No segment covers t=15.85"

    # Critical assertions:
    assert target_seg.subject_source != "face_registry_centroid", \
        f"Bug: centroid path used at t=15.85 (subject_source={target_seg.subject_source})"

    # subject_x must be near Tank (22) or Tyrese (78), NOT between them
    near_tank = abs(target_seg.subject_x - 22) < 10
    near_tyrese = abs(target_seg.subject_x - 78) < 10
    is_split = target_seg.layout in ("split", "grid", "wide_master", "blur_fill")
    assert near_tank or near_tyrese or is_split, \
        f"Bug: subject_x={target_seg.subject_x} is in the empty space between speakers"

    # Strategy should be CROP on active speaker or SPLIT, never plain stationary at center
    if target_seg.layout == "single":
        assert target_seg.subject_x != 50, \
            "Bug: single layout at x=50 with faces at 22 and 78"


# ── Test: _resolve_slot_for_interval multi-face spread ──

def test_resolve_slot_multiface_spread():
    """Direct test of _resolve_slot_for_interval with multi-face spread."""
    registry = make_registry_2_speakers()
    dense = make_dense_faces_2speakers(start=0.0, end=5.0, step=0.5)
    speaker_events = [FakeSpeakerEvent(start=0.0, end=5.0, slot_id=1)]

    slot, conf, layout, source = _resolve_slot_for_interval(
        start=0.0, end=5.0,
        transcript_segments=[], speaker_to_slot={},
        active_speaker_events=speaker_events,
        dense_faces=dense,
        face_registry=registry,
    )

    # Active speaker (slot 1) should win, not centroid
    assert slot == 1, f"Expected slot 1 (active speaker), got {slot}"
    assert layout == "single", f"Expected single layout, got {layout}"
    assert source == "active_speaker_slot", f"Expected active_speaker_slot, got {source}"


def test_resolve_slot_multiface_no_speaker():
    """When no active speaker and faces are wide, should pick dominant or split."""
    registry = make_registry_2_speakers()
    dense = make_dense_faces_2speakers(start=0.0, end=5.0, step=0.5)

    slot, conf, layout, source = _resolve_slot_for_interval(
        start=0.0, end=5.0,
        transcript_segments=[], speaker_to_slot={},
        active_speaker_events=[],
        dense_faces=dense,
        face_registry=registry,
    )

    # Should pick one of the two slots, not land at x=50
    if layout == "single":
        assert slot in (0, 1), f"Expected slot 0 or 1, got {slot}"
        assert source != "hardcoded_center", f"Bug: hardcoded center used"
    # split/blur_fill is also acceptable


if __name__ == "__main__":
    import traceback
    tests = [
        test_two_faces_wide_single_active_speaker,
        test_two_faces_wide_both_speaking,
        test_two_faces_wide_neither_speaking,
        test_two_faces_close_centroid_crop,
        test_three_faces_wide_single_active,
        test_three_faces_wide_all_speaking,
        test_tank_vs_tyrese_regression,
        test_resolve_slot_multiface_spread,
        test_resolve_slot_multiface_no_speaker,
    ]
    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            print(f"  PASS  {test_fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {test_fn.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")
    if failed > 0:
        sys.exit(1)
