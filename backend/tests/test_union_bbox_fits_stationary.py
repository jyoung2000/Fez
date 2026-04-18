"""v4 Change 2 — union-bbox shot mode selector picks STATIONARY when fit.

Three required regions at cx=0.4, 0.5, 0.6 with half_width=0.02 each.
Union spans 0.38..0.62 → width 0.24, well under the 9:16-on-16:9 crop
width of ~0.316. Expect STATIONARY at union center (0.5).
"""

from dataclasses import dataclass

from backend.services.camera_solver import (
    CameraMode,
    compute_union_bbox,
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


def _frame(t, regions):
    return regions  # build_required_regions returns list[list[Region]]


def test_union_bbox_picks_stationary():
    regs_per_frame = [
        [_Region(timestamp=0.0, cx=0.4, cy=0.5, half_width=0.02, half_height=0.04)],
        [_Region(timestamp=0.5, cx=0.5, cy=0.5, half_width=0.02, half_height=0.04)],
        [_Region(timestamp=1.0, cx=0.6, cy=0.5, half_width=0.02, half_height=0.04)],
    ]
    shot = _Shot(index=0, start=0.0, end=1.0)

    result = solve_shot(
        shot, regs_per_frame,
        source_aspect=16 / 9,  # crop_w_needed ≈ 0.316
    )

    assert result.mode == CameraMode.STATIONARY
    assert len(result.keyframes) == 2  # start + end
    # cx ≈ 0.5 (the union center)
    for _, cx, _ in result.keyframes:
        assert abs(cx - 0.5) < 0.02
    assert result.zoom == 1.0


def test_compute_union_bbox_helper():
    regions = [
        _Region(timestamp=0, cx=0.4, cy=0.5, half_width=0.02, half_height=0.04),
        _Region(timestamp=0, cx=0.5, cy=0.5, half_width=0.02, half_height=0.04),
        _Region(timestamp=0, cx=0.6, cy=0.5, half_width=0.02, half_height=0.04),
    ]
    union = compute_union_bbox(regions)
    assert abs(union.left - 0.38) < 1e-6
    assert abs(union.right - 0.62) < 1e-6
    assert abs(union.cx - 0.5) < 1e-6
    assert abs(union.width - 0.24) < 1e-6
    assert union.n_regions == 3
