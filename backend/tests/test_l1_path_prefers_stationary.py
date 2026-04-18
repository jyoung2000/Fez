"""v4 Change 3 — L1 path solver holds still through tiny anchor jitter.

Anchor stream that wobbles by ≤0.01 around 0.5 for a 6-frame shot.
The L1 LP solver's velocity penalty should produce a (near-)constant
path at ~0.5 instead of chasing the noise.
"""

from dataclasses import dataclass

from backend.services.camera_solver import (
    CameraMode,
    _l1_track_keyframes,
    L1_HELD_STILL_TOL,
)


def test_l1_path_holds_through_jitter():
    # Anchor stream around 0.5 with ≤0.01 jitter. Bounds give plenty of
    # room (well-centered, narrow regions). The LP should converge to a
    # constant path with high held-still fraction.
    per_frame_bounds = [
        {"t": 0.0, "left": 0.49, "right": 0.51, "center": 0.50, "cy": 0.5},
        {"t": 0.1, "left": 0.50, "right": 0.52, "center": 0.51, "cy": 0.5},
        {"t": 0.2, "left": 0.48, "right": 0.50, "center": 0.49, "cy": 0.5},
        {"t": 0.3, "left": 0.49, "right": 0.51, "center": 0.50, "cy": 0.5},
        {"t": 0.4, "left": 0.49, "right": 0.51, "center": 0.50, "cy": 0.5},
        {"t": 0.5, "left": 0.49, "right": 0.51, "center": 0.50, "cy": 0.5},
    ]
    crop_half_width = (9.0 / 16.0) / (16.0 / 9.0) / 2.0  # ≈ 0.158

    keyframes, status, held_frac = _l1_track_keyframes(
        per_frame_bounds, crop_half_width, job_id="test", shot_index=0,
    )

    assert status == "success"
    assert len(keyframes) == 6

    cxs = [kf[1] for kf in keyframes]
    # Total range across the path should be small — the LP solver flattens
    # tiny target jitter rather than chasing it.
    cx_range = max(cxs) - min(cxs)
    assert cx_range < 0.05, (
        f"expected near-constant path, got range={cx_range:.4f}, cxs={cxs}"
    )
    # And the average distance from 0.5 should be tiny.
    mean_dev = sum(abs(c - 0.5) for c in cxs) / len(cxs)
    assert mean_dev < 0.05
    # The held-still fraction should be very high (most transitions
    # below the L1_HELD_STILL_TOL threshold).
    assert held_frac >= 0.4, (
        f"expected held_frac >= 0.4 for jitter-only stream, got {held_frac:.2f}"
    )
