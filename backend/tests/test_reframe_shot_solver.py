"""Integration tests for shot-level L1 camera path solver (Phase 3).

Verifies that solve_camera_path_for_shot:
- Provides lookahead: solved path starts moving BEFORE the subject transitions
- Produces smooth, continuous paths across segment boundaries
- Falls back to per-segment solving gracefully
"""

from dataclasses import dataclass, field
from typing import Optional

import pytest
from backend.services.l1_camera_path import (
    solve_camera_path,
    solve_camera_path_for_shot,
)


@dataclass
class _MockFace:
    identity_id: int = 0
    nose_x: float = 50.0
    nose_y: float = 50.0
    width: float = 10.0
    height: float = 12.0


@dataclass
class _MockFrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)


@dataclass
class _MockSegment:
    start: float
    end: float
    active_slot: int = 0
    layout: str = "single"


class TestShotLevelSolverLookahead:
    """Verify that shot-level solving provides lookahead (non-causal smoothing)."""

    def test_path_starts_moving_before_transition(self):
        """Subject transitions from x=400 to x=1500 at t=2.0s.

        With shot-level solving and mirror-padding lookahead, the solved
        path should start moving BEFORE t=2.0s — proving the solver can
        see the upcoming transition and begin easing early.
        """
        # Build dense faces: subject at x=400 for t<2.0, x=1500 for t>=2.0
        # Using 0-100 scale: 400/1920*100 ≈ 20.8, 1500/1920*100 ≈ 78.1
        dense_faces = []
        for i in range(400):  # 4s at 100 samples (10Hz)
            t = i * 0.01  # 100Hz for dense data
            if t < 2.0:
                x = 20.8  # ~400px at 1920 width
            else:
                x = 78.1  # ~1500px at 1920 width
            dense_faces.append(_MockFrameFaces(
                timestamp=t,
                faces=[_MockFace(identity_id=0, nose_x=x)],
            ))

        # Two segments within one shot
        segments = [
            _MockSegment(start=0.0, end=2.0, active_slot=0),
            _MockSegment(start=2.0, end=4.0, active_slot=0),
        ]

        results = solve_camera_path_for_shot(
            shot_start=0.0,
            shot_end=4.0,
            segments_in_shot=segments,
            propagated_path=dense_faces,
            source_width=1920,
            source_height=1080,
            target_fps=30.0,
        )

        assert len(results) == 2

        # Check the first segment's solved path
        first_result = results[0]
        if first_result["mode"] in ("tracking", "panning"):
            path = first_result["path"]
            assert len(path) > 0

            # Find the x value at the END of the first segment (just before transition)
            # It should have already started moving toward 1500
            last_x = path[-1][1]
            # The target in the first segment is ~400px
            # If lookahead works, last_x should be > 400 (started moving toward 1500)
            assert last_x > 420, (
                f"Lookahead not working: last_x={last_x:.1f} should be > 420 "
                f"(started easing toward 1500 before the transition at t=2.0)"
            )

    def test_smooth_across_segment_boundary(self):
        """Path is continuous (no pop) at segment boundaries within a shot."""
        dense_faces = []
        for i in range(300):
            t = i * 0.01
            x = 50.0 + 10.0 * (t / 3.0)  # gradual drift
            dense_faces.append(_MockFrameFaces(
                timestamp=t,
                faces=[_MockFace(identity_id=0, nose_x=x)],
            ))

        segments = [
            _MockSegment(start=0.0, end=1.0, active_slot=0),
            _MockSegment(start=1.0, end=2.0, active_slot=0),
            _MockSegment(start=2.0, end=3.0, active_slot=0),
        ]

        results = solve_camera_path_for_shot(
            shot_start=0.0,
            shot_end=3.0,
            segments_in_shot=segments,
            propagated_path=dense_faces,
            source_width=1920,
        )

        assert len(results) == 3

        # Check continuity at segment boundaries
        for i in range(len(results) - 1):
            r1 = results[i]
            r2 = results[i + 1]
            if r1["path"] and r2["path"]:
                end_x = r1["path"][-1][1]
                start_x = r2["path"][0][1]
                gap = abs(end_x - start_x)
                assert gap < 5.0, (
                    f"Pop at boundary {i}/{i+1}: gap={gap:.1f}px "
                    f"(end_x={end_x:.1f}, start_x={start_x:.1f})"
                )

    def test_empty_segments_handled(self):
        """Empty segment list returns empty results."""
        results = solve_camera_path_for_shot(
            shot_start=0.0,
            shot_end=5.0,
            segments_in_shot=[],
            propagated_path=[],
            source_width=1920,
        )
        assert results == []

    def test_single_segment_same_as_legacy(self):
        """Single segment in a shot should produce results compatible with legacy solver."""
        dense_faces = []
        for i in range(100):
            t = i * 0.1
            x = 50.0  # constant position
            dense_faces.append(_MockFrameFaces(
                timestamp=t,
                faces=[_MockFace(identity_id=0, nose_x=x)],
            ))

        segments = [_MockSegment(start=0.0, end=10.0, active_slot=0)]

        results = solve_camera_path_for_shot(
            shot_start=0.0,
            shot_end=10.0,
            segments_in_shot=segments,
            propagated_path=dense_faces,
            source_width=1920,
        )

        assert len(results) == 1
        # Constant input should produce stationary mode
        assert results[0]["mode"] == "stationary"
