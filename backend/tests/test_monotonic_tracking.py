"""v4 Change 2 — monotonic trajectory routes to TRACKING (L1 path).

Required regions whose centers march from cx=0.2 → 0.5 over 4 seconds.
Union width is wide enough to gate STATIONARY out, but the trajectory
is monotonic so the solver should pick TRACKING and emit per-frame
keyframes solved by the L1 LP.
"""

from dataclasses import dataclass

from backend.services.camera_solver import (
    CameraMode,
    is_monotonic,
    solve_shot,
)


@dataclass
class _Region:
    timestamp: float
    cx: float
    cy: float
    half_width: float
    half_height: float
    tier: str = "required"
    source: str = "face"


@dataclass
class _Shot:
    index: int
    start: float
    end: float


def test_is_monotonic_strict_increasing():
    timeline = [(0.0, 0.2), (1.0, 0.3), (2.0, 0.4), (3.0, 0.5)]
    assert is_monotonic(timeline, tolerance=0.05)


def test_is_monotonic_strict_decreasing():
    timeline = [(0.0, 0.7), (1.0, 0.5), (2.0, 0.3), (3.0, 0.1)]
    assert is_monotonic(timeline, tolerance=0.05)


def test_is_monotonic_rejects_zigzag():
    timeline = [(0.0, 0.3), (1.0, 0.7), (2.0, 0.3), (3.0, 0.7)]
    assert not is_monotonic(timeline, tolerance=0.05)


def test_monotonic_trajectory_picks_tracking():
    # cx=0.2, 0.3, 0.4, 0.5, monotonic, span 0.30 (slightly under crop
    # width 0.316 actually — let's widen with bbox half-width to push
    # union out of stationary range).
    regs_per_frame = [
        [_Region(timestamp=0.0, cx=0.20, cy=0.5, half_width=0.06, half_height=0.04)],
        [_Region(timestamp=1.0, cx=0.30, cy=0.5, half_width=0.06, half_height=0.04)],
        [_Region(timestamp=2.0, cx=0.40, cy=0.5, half_width=0.06, half_height=0.04)],
        [_Region(timestamp=3.0, cx=0.50, cy=0.5, half_width=0.06, half_height=0.04)],
    ]
    shot = _Shot(index=0, start=0.0, end=3.0)

    result = solve_shot(
        shot, regs_per_frame,
        source_aspect=16 / 9,
    )

    # Union spans 0.14..0.56, width 0.42 → wider than crop 0.316.
    # Monotonic → TRACKING (with L1 keyframes).
    assert result.mode == CameraMode.TRACKING, (
        f"expected TRACKING, got {result.mode.value} ({result.reason})"
    )
    assert len(result.keyframes) == 4
    # Path should be roughly monotonic across the keyframes (the L1
    # solver may flatten holds but cannot reverse direction here).
    cxs = [kf[1] for kf in result.keyframes]
    assert cxs[-1] >= cxs[0] - 1e-6, (
        f"expected non-decreasing path, got {cxs}"
    )
