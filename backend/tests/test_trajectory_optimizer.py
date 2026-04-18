"""Tests for the L1 trajectory optimizer."""

import time
import logging

from backend.services.trajectory_optimizer import optimize_camera_path


class TestTrajectoryOptimizer:
    def test_flat_targets(self):
        """Flat targets -> flat output."""
        targets = [(t * 0.5, 50.0) for t in range(20)]
        result = optimize_camera_path(targets, [], [])
        xs = [x for _, x in result]
        assert all(abs(x - 50.0) < 1.0 for x in xs)

    def test_noisy_targets_smoothed(self):
        """Noisy targets around a constant -> smoothed to near-constant."""
        import random
        random.seed(123)
        targets = [(t * 0.5, 50.0 + random.uniform(-5, 5)) for t in range(20)]
        result = optimize_camera_path(targets, [], [], lambda_smooth=0.5)
        xs = [x for _, x in result]
        # Output should be smoother than input (lower standard deviation)
        input_xs = [x for _, x in targets]
        input_std = (sum((x - 50) ** 2 for x in input_xs) / len(input_xs)) ** 0.5
        output_std = (sum((x - 50) ** 2 for x in xs) / len(xs)) ** 0.5
        assert output_std <= input_std * 1.1  # At least as smooth (with tolerance)

    def test_ramp_targets(self):
        """Ramp targets -> smooth ramp output."""
        targets = [(t * 0.5, 20 + t * 3.0) for t in range(20)]
        result = optimize_camera_path(targets, [], [])
        xs = [x for _, x in result]
        # Should be roughly monotonically increasing
        increases = sum(1 for i in range(1, len(xs)) if xs[i] >= xs[i - 1] - 0.5)
        assert increases >= len(xs) * 0.8

    def test_shot_boundary_discontinuity(self):
        """Targets crossing a shot boundary -> discontinuity preserved."""
        targets = [
            (0.0, 30.0), (0.5, 31.0), (1.0, 32.0),
            # Shot boundary at t=1.5
            (2.0, 70.0), (2.5, 71.0), (3.0, 72.0),
        ]
        result = optimize_camera_path(targets, [], [1.5])
        # Should have a jump at the boundary
        xs_before = [x for t, x in result if t < 1.5]
        xs_after = [x for t, x in result if t >= 1.5]
        assert xs_before and xs_after
        assert abs(xs_before[-1] - xs_after[0]) > 20  # Discontinuity preserved

    def test_infeasible_constraints_returns_targets(self, caplog):
        """Infeasible constraints -> returns targets unchanged, logs warning."""
        targets = [(0.0, 50.0), (0.5, 50.0), (1.0, 50.0)]
        # Infeasible: min_x > max_x
        constraints = [(0.0, 80.0, 20.0)]
        with caplog.at_level(logging.WARNING):
            result = optimize_camera_path(targets, constraints, [])
        assert len(result) == 3
        # Should have logged a warning about infeasibility
        assert any("infeasible" in r.message.lower() for r in caplog.records)

    def test_60s_clip_performance(self):
        """60s clip @ 2 FPS -> completes in <2s."""
        n = 120  # 60s * 2 FPS
        targets = [(t * 0.5, 30 + t * 0.3) for t in range(n)]
        constraints = [(t * 0.5, 10, 90) for t in range(n)]
        start = time.perf_counter()
        result = optimize_camera_path(targets, constraints, [])
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"Optimizer took {elapsed:.2f}s — must be <2s"
        assert len(result) == n

    def test_empty_input(self):
        """Empty targets -> empty output."""
        assert optimize_camera_path([], [], []) == []

    def test_single_target(self):
        """Single target -> single output."""
        result = optimize_camera_path([(0.0, 50.0)], [], [])
        assert len(result) == 1
        assert result[0] == (0.0, 50.0)
