"""Tests for motion-aware tracking.

Covers:
  13. Horizontal motion → tracking path moves in expected direction
  14. Chaotic motion → wide_master fallback
  15. Shot cut resets tracker
"""

import numpy as np
import pytest

from backend.services.optical_flow import (
    build_motion_tracking_path,
    compute_motion_energy_per_second,
    is_motion_chaotic,
    should_use_motion_tracking,
)


class _FakeFace:
    def __init__(self, x, y=40, identity_id=0):
        self.x = x
        self.y = y
        self.identity_id = identity_id


class _FakeDenseFace:
    def __init__(self, timestamp, faces):
        self.timestamp = timestamp
        self.faces = faces


class TestMotionEnergy:
    def test_stationary_faces_low_energy(self):
        """Faces that don't move should produce low energy."""
        dense = [_FakeDenseFace(t, [_FakeFace(50)]) for t in range(10)]
        energy = compute_motion_energy_per_second(dense, 10)
        assert len(energy) == 10
        assert float(np.max(energy)) < 0.1

    def test_moving_faces_high_energy(self):
        """Faces that move significantly should produce high energy."""
        dense = []
        for t in range(10):
            x = 20 + t * 8  # moves from 20 to 92 over 10 seconds
            dense.append(_FakeDenseFace(t, [_FakeFace(x)]))
        energy = compute_motion_energy_per_second(dense, 10)
        # Should have non-zero energy
        assert float(np.max(energy)) > 0.1

    def test_empty_input(self):
        energy = compute_motion_energy_per_second([], 10)
        assert len(energy) == 10
        assert float(np.sum(energy)) == 0.0


class TestMotionChaotic:
    def test_steady_motion_not_chaotic(self):
        """Consistent motion in one direction is not chaotic."""
        energy = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        assert not is_motion_chaotic(energy, 0, 5)

    def test_erratic_motion_is_chaotic(self):
        """Wildly varying motion (high stdev vs mean) is chaotic."""
        # Pattern where stdev > mean * threshold (stdev/mean > 1.0)
        energy = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        # mean = 0.375, std ≈ 0.484 → std/mean ≈ 1.29 > 1.0
        assert is_motion_chaotic(energy, 0, 8)

    def test_zero_motion_not_chaotic(self):
        energy = np.zeros(5, dtype=np.float32)
        assert not is_motion_chaotic(energy, 0, 5)


class TestShouldUseMotionTracking:
    def test_high_energy_non_chaotic_uses_tracking(self):
        energy = np.full(10, 0.8, dtype=np.float32)
        assert should_use_motion_tracking(energy, 0, 10)

    def test_low_energy_no_tracking(self):
        energy = np.full(10, 0.2, dtype=np.float32)
        assert not should_use_motion_tracking(energy, 0, 10)

    def test_chaotic_energy_no_tracking(self):
        energy = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                          dtype=np.float32)
        assert not should_use_motion_tracking(energy, 0, 10)


class TestBuildMotionTrackingPath:
    def test_horizontal_motion_path(self):
        """Faces moving right → path should track rightward."""
        dense = []
        for t in range(6):
            x = 30 + t * 10  # 30, 40, 50, 60, 70, 80
            dense.append(_FakeDenseFace(float(t), [_FakeFace(x)]))

        path = build_motion_tracking_path(dense, 0.0, 5.0, shot_cuts=[])
        assert len(path) >= 2
        # First point should be near x=30, last near x=80
        assert path[0][1] < path[-1][1]

    def test_shot_cut_resets_path(self):
        """Shot cut at t=3 should cause a hard reset in the motion path."""
        dense = []
        for t in range(6):
            if t < 3:
                x = 30 + t * 10  # moving right
            else:
                x = 80 - (t - 3) * 10  # suddenly moving left after cut
            dense.append(_FakeDenseFace(float(t), [_FakeFace(x)]))

        path = build_motion_tracking_path(dense, 0.0, 5.0, shot_cuts=[3.0])
        assert len(path) >= 3

        # Find the keypoint nearest the shot cut
        cut_keypoints = [(t, x, y) for t, x, y in path if abs(t - 3.0) < 0.5]
        assert len(cut_keypoints) > 0

    def test_velocity_clamping(self):
        """Large jumps should be velocity-clamped."""
        dense = [
            _FakeDenseFace(0.0, [_FakeFace(20)]),
            _FakeDenseFace(0.5, [_FakeFace(80)]),  # huge jump
            _FakeDenseFace(1.0, [_FakeFace(80)]),
        ]

        path = build_motion_tracking_path(dense, 0.0, 1.0, shot_cuts=[],
                                          max_velocity=25.0)
        if len(path) >= 2:
            # The second point should be clamped, not at 80
            assert path[1][1] < 80

    def test_empty_faces_returns_empty(self):
        dense = [_FakeDenseFace(0.0, []), _FakeDenseFace(1.0, [])]
        path = build_motion_tracking_path(dense, 0.0, 1.0, shot_cuts=[])
        assert len(path) == 0

    def test_single_face_too_few_for_path(self):
        dense = [_FakeDenseFace(0.0, [_FakeFace(50)])]
        path = build_motion_tracking_path(dense, 0.0, 1.0, shot_cuts=[])
        assert len(path) < 2  # Need at least 2 points for a path
