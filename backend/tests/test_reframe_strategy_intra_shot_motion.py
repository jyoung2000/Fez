"""Phase 5 — force `tracking` strategy on intra-shot face motion.

The reframe segmenter currently lets the L1 solver classify a shot as
``stationary`` whenever the solved camera path's range stays under
``STATIONARY_THRESHOLD * source_width`` (≈ 8% of frame width). On
animated content the L1 solver sometimes hits the stationary band
even when the face panned 10–20% across the frame, because the
solver's velocity penalty + slot-center fallback signal smooth the
target into a near-constant shape. The result: faces drift off-
center within close-up shots in the 9:16 preview.

Phase 5 introduces ``should_force_tracking_for_motion``: a
purely-functional helper that scans dense_face nose_x within a
shot's time range and returns True when the intra-shot motion
exceeds 6 % of frame width AND the shot is at least 1.0 seconds
long. The reframe_segmenter calls this in the L1 result loop and
overrides ``stationary`` → ``tracking`` accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.services.reframe_segmenter import (
    intra_shot_face_motion_pct,
    should_force_tracking_for_motion,
)


@dataclass
class _Face:
    identity_id: int
    nose_x: float


@dataclass
class _FrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)


def _build_dense(slot_id: int, xs: list[float], step: float = 0.5) -> list:
    return [
        _FrameFaces(timestamp=i * step, faces=[_Face(slot_id, float(x))])
        for i, x in enumerate(xs)
    ]


def test_intra_shot_face_motion_pct_returns_zero_for_constant_face():
    dense = _build_dense(0, [50.0] * 6)
    motion = intra_shot_face_motion_pct(dense, slot_id=0, start=0.0, end=3.0)
    assert motion == 0.0


def test_intra_shot_face_motion_pct_measures_panning_range():
    dense = _build_dense(0, [50.0, 53.0, 56.0, 59.0, 62.0])  # 5 samples
    motion = intra_shot_face_motion_pct(dense, slot_id=0, start=0.0, end=3.0)
    assert abs(motion - 12.0) < 1e-9


def test_stationary_face_does_not_force_tracking():
    """A 3 s shot with no face motion should remain `stationary`."""
    dense = _build_dense(0, [50.0] * 6)
    forced = should_force_tracking_for_motion(
        dense, slot_id=0, start=0.0, end=3.0,
    )
    assert forced is False


def test_panning_face_forces_tracking():
    """A 3 s shot with the face panning 50 → 62 (12 % motion) should
    flip the override on, exceeding the 6 % threshold."""
    dense = _build_dense(0, [50.0, 53.0, 56.0, 59.0, 62.0, 62.0])
    forced = should_force_tracking_for_motion(
        dense, slot_id=0, start=0.0, end=3.0,
    )
    assert forced is True


def test_short_shot_below_min_seconds_never_forced():
    """Shots under the ``min_shot_seconds`` floor are never overridden,
    even when the face moves across the entire frame."""
    dense = _build_dense(0, [10.0, 90.0], step=0.4)  # 2 samples, 0.8s span
    forced = should_force_tracking_for_motion(
        dense, slot_id=0, start=0.0, end=0.8,
    )
    assert forced is False


def test_just_below_motion_threshold_stays_stationary():
    """Motion of exactly 6 % of frame width is BELOW the strict-greater
    threshold and should NOT trigger the override."""
    dense = _build_dense(0, [50.0, 56.0, 56.0, 56.0])
    forced = should_force_tracking_for_motion(
        dense, slot_id=0, start=0.0, end=2.0,
    )
    assert forced is False


def test_slot_id_filtering_isolates_active_speaker():
    """When the dense stream has multiple slots in the same window,
    the helper must only consider the requested ``slot_id``."""
    dense = []
    # Slot 0 is panning 50 → 70 (would force tracking)
    # Slot 1 is stationary at 30 (would not)
    for i, x in enumerate([50, 55, 60, 65, 70]):
        dense.append(_FrameFaces(
            timestamp=i * 0.5,
            faces=[_Face(0, float(x)), _Face(1, 30.0)],
        ))
    assert should_force_tracking_for_motion(
        dense, slot_id=0, start=0.0, end=3.0,
    ) is True
    assert should_force_tracking_for_motion(
        dense, slot_id=1, start=0.0, end=3.0,
    ) is False


def test_two_segments_a_static_b_panning():
    """Spec-required scenario: (a) face stationary at x=50 across 3s
    → stationary; (b) face panning 50→62 across 3s → tracking."""
    seg_a = _build_dense(0, [50.0] * 6)  # stationary
    seg_b = _build_dense(
        0, [50.0, 52.0, 54.0, 56.0, 58.0, 60.0, 62.0],  # panning 50→62
    )
    assert should_force_tracking_for_motion(
        seg_a, slot_id=0, start=0.0, end=3.0,
    ) is False
    assert should_force_tracking_for_motion(
        seg_b, slot_id=0, start=0.0, end=3.0,
    ) is True
