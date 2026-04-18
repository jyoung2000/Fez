"""v4 Change 2 — wide unions trigger STATIONARY_ZOOMED instead of PADDED.

Three required regions spanning cx=0.30..0.70 (union width 0.40, well
above the 9:16-on-16:9 crop width of ~0.316). Trajectory is non-monotonic
(zig-zag) so TRACKING is rejected. Expect STATIONARY_ZOOMED with
zoom=1.3 (the smallest discrete step that absorbs the union).
"""

from dataclasses import dataclass

from backend.services.camera_solver import CameraMode, solve_shot


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


def test_wide_union_triggers_stationary_zoomed():
    # Zig-zag trajectory whose union spans 0.38 in normalized space —
    # too wide for STATIONARY (crop ≈ 0.316) and too wide for zoom 1.1
    # or 1.2, but fits zoom 1.3 (0.316 × 1.3 × 0.95 ≈ 0.39).
    # Non-monotonic so TRACKING is gated out, falling into the zoom test.
    regs_per_frame = [
        [_Region(timestamp=0.0, cx=0.31, cy=0.5, half_width=0.0, half_height=0.04)],
        [_Region(timestamp=0.25, cx=0.69, cy=0.5, half_width=0.0, half_height=0.04)],
        [_Region(timestamp=0.5, cx=0.31, cy=0.5, half_width=0.0, half_height=0.04)],
        [_Region(timestamp=0.75, cx=0.69, cy=0.5, half_width=0.0, half_height=0.04)],
        [_Region(timestamp=1.0, cx=0.31, cy=0.5, half_width=0.0, half_height=0.04)],
    ]
    shot = _Shot(index=0, start=0.0, end=1.0)

    result = solve_shot(
        shot, regs_per_frame,
        source_aspect=16 / 9,
    )

    assert result.mode == CameraMode.STATIONARY_ZOOMED, (
        f"expected STATIONARY_ZOOMED, got {result.mode.value} ({result.reason})"
    )
    # Smallest discrete zoom that absorbs union width 0.38 with safety
    # margin: 0.316 × 1.3 × 0.95 ≈ 0.39 — so zoom=1.3 is the answer.
    assert result.zoom == 1.3, f"expected zoom=1.3, got {result.zoom}"
    assert len(result.keyframes) == 2
    # cx around the union center 0.5
    for _, cx, _ in result.keyframes:
        assert abs(cx - 0.5) < 0.05
