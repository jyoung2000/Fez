"""Tests for hard constraint enforcement in the L1 camera path solver.

Verifies that must_be_in_frame=True features produce either a path that
keeps them inside the crop at every frame, or an 'infeasible' mode signal
that tells the layout engine to fall back to PADDED (SPLIT/PIP).
"""

import pytest
from backend.services.l1_camera_path import (
    solve_camera_path,
    compute_hard_bounds,
    assert_required_in_frame,
    _tv_denoise_1d,
)
from backend.services.focus_model import RequiredFeature, FeatureKind


def _make_face(t, x, w=10, identity=0):
    """Helper: build a RequiredFeature representing a face."""
    return RequiredFeature(
        t_start=t, t_end=t,
        x=x, y=50.0, w=w, h=13.0,
        kind=FeatureKind.FACE,
        weight=1.0,
        must_be_in_frame=True,
        identity=identity,
    )


def _make_hud(t, x, w=8):
    """Helper: build a RequiredFeature representing a HUD element."""
    return RequiredFeature(
        t_start=t, t_end=t,
        x=x, y=50.0, w=w, h=8.0,
        kind=FeatureKind.HUD,
        weight=0.9,
        must_be_in_frame=True,
        excluded_from_centroid=True,
    )


class TestTwoFaces60PercentApart:
    """Two faces 60% of source width apart.

    With a 9:16 crop on 1920x1080, the crop is ~607px wide = 31.6% of frame.
    Two faces 60% apart (at x=20% and x=80%) cannot both fit in one crop.
    The solver must return 'infeasible' (triggering PADDED fallback),
    never silently drop one face.
    """

    def test_returns_infeasible_when_faces_too_far(self):
        # Face A at x=20%, face B at x=80% — 60% apart
        features = []
        positions = []
        for t in range(20):
            ts = float(t) * 0.5
            features.append(_make_face(ts, 20.0, w=10, identity=0))
            features.append(_make_face(ts, 80.0, w=10, identity=1))
            # Target = midpoint between faces (would be ideal)
            positions.append((ts, 960.0))  # center of 1920

        result = solve_camera_path(
            face_positions=positions,
            source_width=1920,
            hard_features=features,
            source_height=1080,
            crop_aspect=9 / 16,
        )

        assert result["mode"] == "infeasible", (
            f"Expected infeasible for 60%-apart faces, got mode={result['mode']}"
        )

    def test_infeasible_has_frame_info(self):
        features = [
            _make_face(0.0, 15.0, w=10, identity=0),
            _make_face(0.0, 85.0, w=10, identity=1),
        ]
        positions = [(0.0, 960.0), (0.5, 960.0)]

        result = solve_camera_path(
            face_positions=positions,
            source_width=1920,
            hard_features=features,
            source_height=1080,
        )
        assert result["mode"] == "infeasible"
        assert len(result["infeasible_frames"]) > 0


class TestFaceAndHudOppositeSides:
    """One face + one HUD region on opposite sides.

    The HUD is excluded_from_centroid=True but still must_be_in_frame=True.
    If both fit in the crop width, the solver must keep both. If not, infeasible.
    """

    def test_face_and_hud_both_fit(self):
        # Face at x=40%, HUD at x=55% — only 15% apart, fits easily in 31.6% crop
        features = []
        positions = []
        for t in range(10):
            ts = float(t) * 0.5
            features.append(_make_face(ts, 40.0, w=8, identity=0))
            features.append(_make_hud(ts, 55.0, w=6))
            positions.append((ts, 40.0 / 100.0 * 1920))

        result = solve_camera_path(
            face_positions=positions,
            source_width=1920,
            hard_features=features,
            source_height=1080,
        )

        assert result["mode"] != "infeasible", (
            "Face and HUD 15% apart should fit in 31.6% crop"
        )

        # Verify both are actually in frame
        if result["path"]:
            violations = assert_required_in_frame(
                result["path"], features, 1920, 9 / 16, 1080,
            )
            assert len(violations) == 0, f"Violations: {violations}"

    def test_face_and_hud_too_far(self):
        # Face at x=10%, HUD at x=80% — too far apart for 31.6% crop
        features = []
        positions = []
        for t in range(10):
            ts = float(t) * 0.5
            features.append(_make_face(ts, 10.0, w=8, identity=0))
            features.append(_make_hud(ts, 80.0, w=6))
            positions.append((ts, 10.0 / 100.0 * 1920))

        result = solve_camera_path(
            face_positions=positions,
            source_width=1920,
            hard_features=features,
            source_height=1080,
        )

        assert result["mode"] == "infeasible"


class TestCrossingFacePlusStaticFace:
    """Face A crosses the frame while face B stays static.

    If at any point the two faces are wider than the crop, the solver
    must return infeasible rather than silently dropping one.
    """

    def test_crossing_produces_infeasible_when_too_wide(self):
        features = []
        positions = []
        for i in range(20):
            ts = float(i) * 0.5
            # Face A walks from x=10% to x=90%
            ax = 10.0 + (80.0 * i / 19.0)
            # Face B stays at x=50%
            bx = 50.0
            features.append(_make_face(ts, ax, w=8, identity=0))
            features.append(_make_face(ts, bx, w=8, identity=1))
            # Target: midpoint
            mid_px = ((ax + bx) / 2.0) / 100.0 * 1920
            positions.append((ts, mid_px))

        result = solve_camera_path(
            face_positions=positions,
            source_width=1920,
            hard_features=features,
            source_height=1080,
        )

        # At some point face A is >60% from face B → infeasible
        assert result["mode"] == "infeasible", (
            "Crossing face should produce infeasible when separation exceeds crop width"
        )


class TestStationaryFaceRegression:
    """Existing stationary-face test: a single face that barely moves.

    TV denoise with clipping at bounds far from target should be a no-op.
    The solver must still produce a stationary path.
    """

    def test_single_stationary_face_still_stationary(self):
        features = []
        positions = []
        for i in range(20):
            ts = float(i) * 0.5
            # Face at x=50% with tiny jitter (±1%)
            jitter = 0.5 if i % 2 == 0 else -0.5
            fx = 50.0 + jitter
            features.append(_make_face(ts, fx, w=10, identity=0))
            positions.append((ts, fx / 100.0 * 1920))

        result = solve_camera_path(
            face_positions=positions,
            source_width=1920,
            hard_features=features,
            source_height=1080,
        )

        assert result["mode"] == "stationary", (
            f"Single near-stationary face should produce stationary, got {result['mode']}"
        )

    def test_backward_compat_no_features(self):
        """Without hard_features, solver behaves identically to before."""
        positions = [(float(i) * 0.5, 960.0 + (i % 2)) for i in range(20)]

        result = solve_camera_path(
            face_positions=positions,
            source_width=1920,
        )

        assert result["mode"] == "stationary"
        assert result["infeasible_frames"] == []


class TestComputeHardBounds:
    """Unit tests for the compute_hard_bounds helper."""

    def test_single_face_produces_valid_bounds(self):
        features = [_make_face(0.0, 50.0, w=10)]
        lo, hi, inf = compute_hard_bounds(
            features, [0.0], 1920, 9 / 16, 1080,
        )
        assert not inf[0]
        assert lo[0] is not None
        assert hi[0] is not None
        assert lo[0] <= hi[0]

    def test_no_features_produces_none_bounds(self):
        lo, hi, inf = compute_hard_bounds([], [0.0], 1920)
        assert lo[0] is None
        assert hi[0] is None
        assert not inf[0]

    def test_wide_feature_marks_infeasible(self):
        # Feature covering 50% of frame — wider than ~31.6% crop
        wide_face = RequiredFeature(
            t_start=0.0, t_end=0.0,
            x=50.0, y=50.0, w=50.0, h=13.0,
            kind=FeatureKind.FACE, weight=1.0, must_be_in_frame=True,
        )
        lo, hi, inf = compute_hard_bounds(
            [wide_face], [0.0], 1920, 9 / 16, 1080,
        )
        assert inf[0], "A feature 50% wide should be infeasible in a 31.6% crop"


class TestClippedTVDenoise:
    """Verify that _tv_denoise_1d respects bounds when provided."""

    def test_bounds_are_respected(self):
        signal = [100.0, 500.0, 900.0, 500.0, 100.0]
        lo = [200.0, 200.0, 200.0, 200.0, 200.0]
        hi = [800.0, 800.0, 800.0, 800.0, 800.0]

        result = _tv_denoise_1d(signal, lam=10.0, n_iter=100,
                                lo_bounds=lo, hi_bounds=hi)

        for i, v in enumerate(result):
            assert v >= lo[i] - 0.01, f"result[{i}]={v} < lo={lo[i]}"
            assert v <= hi[i] + 0.01, f"result[{i}]={v} > hi={hi[i]}"

    def test_no_bounds_matches_original(self):
        signal = [100.0, 200.0, 300.0]
        result_no_bounds = _tv_denoise_1d(signal, lam=5.0)
        result_none_bounds = _tv_denoise_1d(signal, lam=5.0,
                                             lo_bounds=None, hi_bounds=None)
        for a, b in zip(result_no_bounds, result_none_bounds):
            assert abs(a - b) < 0.01
