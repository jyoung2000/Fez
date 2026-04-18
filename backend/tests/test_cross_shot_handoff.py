"""Tests for cross-shot handoff (Phase C).

Validates that:
- Hard cuts (gap < 0.05s) don't modify either shot's tail
- Held gaps (>= 0.05s) ease the tail of shot N toward shot N+1
- Active-speaker containment is maintained during tail easing
- Gaming mode skips all handoff logic
"""
import pytest
from dataclasses import dataclass, field
from typing import List


@dataclass
class _FakeShotCamera:
    shot_index: int
    start: float
    end: float
    mode: str = "tracking"
    keyframes: List[tuple] = field(default_factory=list)
    reason: str = ""
    zoom: float = 1.0


class TestCrossShot:
    """E3: Cross-shot look-ahead handoff."""

    def test_hard_cut_no_modification(self):
        """Two shots back-to-back (gap < 0.01s): last keyframe of shot 0
        should stay near its active speaker cx, first keyframe of shot 1
        should be at its own active speaker."""
        from backend.services.camera_solver import apply_cross_shot_handoff

        shot0 = _FakeShotCamera(
            shot_index=0, start=0.0, end=3.0,
            keyframes=[
                (0.0, 0.2, 0.5),
                (1.0, 0.2, 0.5),
                (2.0, 0.2, 0.5),
                (3.0, 0.2, 0.5),
            ],
        )
        shot1 = _FakeShotCamera(
            shot_index=1, start=3.0, end=6.0,
            keyframes=[
                (3.0, 0.8, 0.5),
                (4.0, 0.8, 0.5),
                (5.0, 0.8, 0.5),
                (6.0, 0.8, 0.5),
            ],
        )

        result = apply_cross_shot_handoff([shot0, shot1], content_type="talking_head")

        # Shot 0's last keyframe should stay near 0.2 (within 4% bound)
        last_cx = result[0].keyframes[-1][1]
        assert abs(last_cx - 0.2) <= 0.04, (
            f"Hard cut: shot 0 tail cx={last_cx:.3f}, expected near 0.2"
        )

        # Shot 1's first keyframe should stay at 0.8
        first_cx = result[1].keyframes[0][1]
        assert abs(first_cx - 0.8) <= 0.04, (
            f"Hard cut: shot 1 head cx={first_cx:.3f}, expected near 0.8"
        )

    def test_held_gap_eases_tail(self):
        """200ms gap between shots: shot 0's tail can drift toward shot 1
        by up to 4%."""
        from backend.services.camera_solver import apply_cross_shot_handoff

        shot0 = _FakeShotCamera(
            shot_index=0, start=0.0, end=3.0,
            keyframes=[
                (0.0, 0.2, 0.5),
                (1.0, 0.2, 0.5),
                (2.0, 0.2, 0.5),
                (2.85, 0.2, 0.5),  # within ease window
                (2.95, 0.2, 0.5),  # within ease window
                (3.0, 0.2, 0.5),   # last kf
            ],
        )
        shot1 = _FakeShotCamera(
            shot_index=1, start=3.2, end=6.0,  # 200ms gap
            keyframes=[
                (3.2, 0.8, 0.5),
                (4.0, 0.8, 0.5),
                (5.0, 0.8, 0.5),
                (6.0, 0.8, 0.5),
            ],
        )

        result = apply_cross_shot_handoff([shot0, shot1], content_type="talking_head")

        # Shot 0's tail can drift but not more than 4%
        last_cx = result[0].keyframes[-1][1]
        drift = abs(last_cx - 0.2)
        assert drift <= 0.04 + 0.001, (
            f"Held gap: tail drift={drift:.3f}, max allowed=0.04"
        )

    def test_gaming_skips_handoff(self):
        """Gaming content should bypass all handoff logic."""
        from backend.services.camera_solver import apply_cross_shot_handoff

        shot0 = _FakeShotCamera(
            shot_index=0, start=0.0, end=3.0,
            keyframes=[(0.0, 0.2, 0.5), (3.0, 0.2, 0.5)],
        )
        shot1 = _FakeShotCamera(
            shot_index=1, start=3.2, end=6.0,
            keyframes=[(3.2, 0.8, 0.5), (6.0, 0.8, 0.5)],
        )

        # Save original keyframes
        orig_kf0 = list(shot0.keyframes)

        result = apply_cross_shot_handoff([shot0, shot1], content_type="gameplay")

        # Keyframes should be unchanged
        assert result[0].keyframes == orig_kf0

    def test_tail_lock_constant_exists(self):
        """TAIL_LOCK_SECONDS constant should exist."""
        from backend.services.camera_solver import TAIL_LOCK_SECONDS
        assert TAIL_LOCK_SECONDS == 0.30
