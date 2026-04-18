"""Phase 4 — L1 solver center-distance penalty and glance clamp.

Two small helpers live next to the existing solver:

  * :func:`apply_glance_deviation_clamp` — post-solve
    hard-clamp of the horizontal pan range.
  * :func:`center_distance_cost` — scalar cost for a path
    under the gaming mode's asymmetric center penalty.

These tests verify that (a) the clamp is a no-op inside the
allowed deviation band, (b) it projects off-limits samples onto
the nearest boundary, and (c) a centered candidate path has
strictly lower center-distance cost than an off-center one for
the same weight.
"""

from __future__ import annotations

import pytest

from backend.services.l1_camera_path import (
    apply_glance_deviation_clamp,
    center_distance_cost,
)


SOURCE_W = 1920
CROP_W = SOURCE_W * 9 / 16 / (16 / 9)  # 9:16 from 16:9 source
# Simpler: compute 9:16 crop width from a 1080p source height.
CROP_W = 1080 * 9 / 16  # 607.5


def test_clamp_noop_within_band():
    path = [(0.0, 960.0), (1.0, 1000.0), (2.0, 920.0)]
    clamped = apply_glance_deviation_clamp(
        path,
        source_width=SOURCE_W,
        crop_width=CROP_W,
        max_glance_deviation_pct=0.65,
    )
    assert clamped == [(0.0, 960.0), (1.0, 1000.0), (2.0, 920.0)]


def test_clamp_projects_out_of_band_samples():
    # max_pan = (1920 - 607.5) / 2 = 656.25
    # max_dev = 0.65 * 656.25 = 426.5625
    # band = [960 - 426.5625, 960 + 426.5625] = [533.4375, 1386.5625]
    max_pan = (SOURCE_W - CROP_W) / 2
    max_dev = 0.65 * max_pan
    lo = 960.0 - max_dev
    hi = 960.0 + max_dev

    path = [
        (0.0, 100.0),   # far below lo
        (1.0, 800.0),   # inside band
        (2.0, 1500.0),  # above hi
    ]
    clamped = apply_glance_deviation_clamp(
        path,
        source_width=SOURCE_W,
        crop_width=CROP_W,
        max_glance_deviation_pct=0.65,
    )
    assert abs(clamped[0][1] - lo) < 1e-9
    assert abs(clamped[1][1] - 800.0) < 1e-9
    assert abs(clamped[2][1] - hi) < 1e-9


def test_clamp_respects_timestamps():
    path = [(0.5, 100.0), (1.0, 2000.0)]
    clamped = apply_glance_deviation_clamp(
        path,
        source_width=SOURCE_W,
        crop_width=CROP_W,
    )
    assert [t for t, _ in clamped] == [0.5, 1.0]


def test_clamp_is_a_no_op_when_crop_covers_source():
    path = [(0.0, 960.0), (1.0, 300.0)]
    clamped = apply_glance_deviation_clamp(
        path,
        source_width=SOURCE_W,
        crop_width=SOURCE_W,
    )
    assert clamped == path


def test_clamp_validates_pct_range():
    with pytest.raises(ValueError):
        apply_glance_deviation_clamp(
            [(0.0, 960.0)],
            source_width=SOURCE_W,
            crop_width=CROP_W,
            max_glance_deviation_pct=1.5,
        )


def test_center_distance_cost_centered_beats_offcenter():
    """Two candidate paths with identical length — centered one
    has strictly smaller center-distance cost."""
    centered = [(i * 0.033, 960.0) for i in range(30)]
    offcenter = [(i * 0.033, 1300.0) for i in range(30)]
    c_cost = center_distance_cost(
        centered, source_width=SOURCE_W, crop_width=CROP_W,
    )
    o_cost = center_distance_cost(
        offcenter, source_width=SOURCE_W, crop_width=CROP_W,
    )
    assert c_cost == 0.0
    assert o_cost > 0.0
    # The relationship must hold regardless of lambda_center,
    # so sanity-check with a higher tunable too.
    c_hi = center_distance_cost(
        centered, source_width=SOURCE_W, crop_width=CROP_W,
        lambda_center=1.0,
    )
    o_hi = center_distance_cost(
        offcenter, source_width=SOURCE_W, crop_width=CROP_W,
        lambda_center=1.0,
    )
    assert c_hi == 0.0
    assert o_hi > o_cost  # higher lambda = higher cost on same offset


def test_center_distance_cost_empty_path_is_zero():
    assert center_distance_cost(
        [], source_width=SOURCE_W, crop_width=CROP_W,
    ) == 0.0


def test_center_distance_cost_full_width_crop_is_zero():
    path = [(0.0, 200.0), (1.0, 1800.0)]
    c = center_distance_cost(
        path, source_width=SOURCE_W, crop_width=SOURCE_W,
    )
    assert c == 0.0


def test_clamp_composes_with_center_penalty():
    """A clamped path has a bounded cost — clamp + cost together
    are what the pipeline uses as a poor-man's hard cap."""
    long_path = [(i * 0.033, 960.0 + 2000.0 * (i % 2)) for i in range(10)]
    clamped = apply_glance_deviation_clamp(
        long_path,
        source_width=SOURCE_W,
        crop_width=CROP_W,
        max_glance_deviation_pct=0.65,
    )
    max_pan = (SOURCE_W - CROP_W) / 2
    max_dev = 0.65 * max_pan
    for _t, x in clamped:
        assert abs(x - 960.0) <= max_dev + 1e-6
    # And the cost on the clamped path is bounded by
    # (0.4 * 0.65^2) per frame at max deviation.
    max_cost_per_frame = 0.4 * 0.65 * 0.65
    c = center_distance_cost(
        clamped, source_width=SOURCE_W, crop_width=CROP_W,
        lambda_center=0.4,
    )
    assert c <= max_cost_per_frame * len(clamped) + 1e-6
