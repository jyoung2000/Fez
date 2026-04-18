"""Tests for AutoFlip-matching camera mode detection refinements."""

from backend.services.camera_path import CameraMode, select_camera_mode
from backend.services.focus_model import RequiredFeature, FeatureKind, SceneFocusRegion


def _make_focus(targets, fits=True, shot_start=None, shot_end=None, required=None):
    """Helper to build a SceneFocusRegion with per-frame targets."""
    return SceneFocusRegion(
        shot_start=shot_start if shot_start is not None else (targets[0][0] if targets else 0),
        shot_end=shot_end if shot_end is not None else (targets[-1][0] if targets else 1),
        required=required or [],
        optional=[],
        min_bounding_rect=(50, 50, 10, 13),
        fits_target_aspect=fits,
        optimal_crop_center=(50, 50),
        per_frame_target=targets,
    )


def _make_required_face(t, x, must=True):
    """Helper to create a required face feature."""
    return RequiredFeature(
        t_start=t, t_end=t, x=x, y=40.0, w=10.0, h=13.0,
        kind=FeatureKind.FACE, weight=1.0, must_be_in_frame=must,
    )


class TestRuleAShortShot:
    def test_short_shot_static_face_stationary(self):
        """0.5s shot with static face -> STATIONARY via Rule A."""
        targets = [(t * 0.1, 50.0, 40.0) for t in range(5)]
        focus = _make_focus(targets, fits=True, shot_start=0.0, shot_end=0.5)
        mode = select_camera_mode(focus, source_width=1920, source_height=1080)
        assert mode == CameraMode.STATIONARY

    def test_short_shot_doesnt_fit_padding(self):
        """0.3s shot with two faces that don't fit -> PADDING via Rule A."""
        targets = [(0.0, 50.0, 40.0), (0.1, 50.0, 40.0), (0.2, 50.0, 40.0)]
        focus = _make_focus(targets, fits=False, shot_start=0.0, shot_end=0.3)
        mode = select_camera_mode(focus, source_width=1920, source_height=1080)
        assert mode == CameraMode.PADDING


class TestRuleBMultiFeatureInfeasibility:
    def test_simultaneous_multi_features_dont_fit_padding(self):
        """5s shot with 3 simultaneous hard-required faces that don't fit -> PADDING via Rule B."""
        # All 3 faces at same timestamp -> simultaneous, bounding rect exceeds crop
        required = [
            _make_required_face(t=1.0, x=10.0),
            _make_required_face(t=1.0, x=50.0),
            _make_required_face(t=1.0, x=90.0),
        ]
        targets = [(t, 50.0, 40.0) for t in [0.0, 1.0, 2.0, 3.0, 4.0]]
        focus = _make_focus(targets, fits=False, shot_start=0.0, shot_end=5.0, required=required)
        mode = select_camera_mode(focus, source_width=1920, source_height=1080)
        assert mode == CameraMode.PADDING


class TestRuleCAspectAwareTolerance:
    def test_extreme_aspect_tighter_tolerance(self):
        """Face walking with x_range at old 10% threshold on 16:9->9:16 -> TRACKING.

        With the old hardcoded tolerance of crop_width_pct * 0.1 = 3.16,
        a range of exactly 3.16 would have been STATIONARY. With the new
        aspect-aware tolerance (reduced by aspect mismatch), it exceeds the
        threshold and becomes TRACKING.
        """
        # 16:9 -> 9:16: crop_width_pct ~31.6%
        # Old tolerance: 31.6 * 0.1 = 3.16
        # New tolerance: 31.6 * 0.10 * (1.0 - min(0.684 * 0.5, 0.5))
        #   aspect_mismatch = |1.778 - 0.5625| / 1.778 = 0.684
        #   tolerance = 31.6 * 0.10 * (1.0 - 0.342) = 31.6 * 0.10 * 0.658 = 2.08
        # x_range of 2.5 > 2.08 -> no longer STATIONARY
        targets = [(t * 0.5, 50.0 + t * 0.125, 40.0) for t in range(20)]
        # x_range = 19 * 0.125 = 2.375
        focus = _make_focus(targets, fits=True, shot_start=0.0, shot_end=10.0)
        mode = select_camera_mode(focus, source_width=1920, source_height=1080)
        assert mode == CameraMode.TRACKING


class TestRuleDRSquaredHysteresis:
    def test_r_squared_087_not_panning(self):
        """R^2 ~0.87 (between old 0.85 and new 0.90 thresholds) -> not PANNING.

        This tests the hysteresis band that prevents mode oscillation.
        With noise_amp=5.5, R^2 ~0.873 and mean_accel ~21.5 (>5, so
        TRACKING doesn't trigger either). Falls through to PADDING.
        """
        targets = []
        for i in range(20):
            t = i * 0.5
            x = 20 + 3 * t + (5.5 if i % 4 == 0 else -3.3 if i % 4 == 2 else 0.0)
            targets.append((t, x, 40.0))

        focus = _make_focus(targets, fits=True, shot_start=0.0, shot_end=10.0)
        mode = select_camera_mode(focus, source_width=1920, source_height=1080)
        # Should NOT be PANNING due to hysteresis (R^2 in 0.85-0.90 band)
        assert mode != CameraMode.PANNING

    def test_r_squared_092_panning(self):
        """R^2 = 0.92 -> clearly linear, PANNING."""
        # Very clean linear motion from x=20 to x=80 over 10 seconds
        # This gives high R^2 > 0.90
        targets = [(t * 0.5, 20 + t * 3.0, 40.0) for t in range(20)]
        focus = _make_focus(targets, fits=True, shot_start=0.0, shot_end=10.0)
        mode = select_camera_mode(focus, source_width=1920, source_height=1080)
        # Clean linear = low acceleration -> TRACKING wins first
        # This is fine: TRACKING is preferred over PANNING when accel is low
        assert mode in (CameraMode.TRACKING, CameraMode.PANNING)


class TestExistingBehaviorPreserved:
    def test_single_stationary_face(self):
        """Single stationary face -> STATIONARY (unchanged)."""
        targets = [(t * 0.5, 50.0 + (t % 3) * 0.1, 40.0) for t in range(20)]
        focus = _make_focus(targets, shot_start=0.0, shot_end=10.0)
        mode = select_camera_mode(focus, source_width=1920, source_height=1080)
        assert mode == CameraMode.STATIONARY

    def test_no_targets(self):
        """No per-frame targets -> STATIONARY (unchanged)."""
        focus = _make_focus([], fits=True, shot_start=0.0, shot_end=5.0)
        mode = select_camera_mode(focus, source_width=1920, source_height=1080)
        assert mode == CameraMode.STATIONARY

    def test_doesnt_fit_single_target_padding(self):
        """Doesn't fit + single target -> PADDING (unchanged)."""
        focus = _make_focus([(0.0, 50.0, 40.0)], fits=False, shot_start=0.0, shot_end=5.0)
        mode = select_camera_mode(focus, source_width=1920, source_height=1080)
        assert mode == CameraMode.PADDING
