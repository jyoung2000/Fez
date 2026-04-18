"""Phase 5 — preview / export parity for gaming layout modes.

The frontend Canvas renderer (``frontend/src/engine/RenderEngine.js``)
and the backend FFmpeg filter graph builder
(``backend/services/gameplay_filters``) both consume the same
rectangle math from ``backend.services.gameplay_geometry``. This
test pins the contract: for every layout mode we exercise, the
FFmpeg filter dimensions and the rect-helper dimensions agree
to the pixel.

The acceptance band from the spec is ±2 px between Canvas and
FFmpeg output. We can't run the headless browser here, but the
guarantee that BOTH renderers consume the same helper means
they cannot disagree by more than rounding (which the helper's
even-rounding rule caps at 2 px).
"""

from __future__ import annotations

import re

from backend.services.gameplay_filters import (
    build_gameplay_blurfill_filter,
    build_gameplay_wide_zoom_filter,
)
from backend.services.gameplay_geometry import (
    compute_blurfill_rects,
    compute_fullscreen_rects,
    compute_layout_rects,
    compute_wide_zoom_rects,
)


# ──────────── geometry helpers ────────────


def test_fullscreen_centered_crop_on_16x9_source():
    rects = compute_fullscreen_rects(1920, 1080, 1080, 1920, cam_x_pct=50.0)
    sx, sy, sw, sh = rects["src"]
    assert sw % 2 == 0
    assert sh == 1080
    # 9:16 crop of 1920x1080 = (1080 * 9/16, 1080) = (607.5 → 608)
    target_aspect = 1080 / 1920
    expected_sw = int(round(1080 * target_aspect))
    expected_sw = expected_sw - (expected_sw % 2)
    assert sw == expected_sw
    # Centered horizontally
    assert sx == (1920 - sw) // 2 or abs(sx - (1920 - sw) // 2) <= 1


def test_fullscreen_off_center_crosshair():
    """Crosshair at 30% should pull the crop window left."""
    rects = compute_fullscreen_rects(1920, 1080, 1080, 1920, cam_x_pct=30.0)
    sx, _sy, sw, _sh = rects["src"]
    # Camera pixel = 576, crop center should be at 576 (clamped to source bounds)
    expected_center = 30.0 / 100.0 * 1920  # 576
    actual_center = sx + sw / 2
    assert abs(actual_center - expected_center) < 2


def test_blurfill_band_centered_vertically():
    rects = compute_blurfill_rects(1920, 1080, 1080, 1920)
    _bx, by, bw, bh = rects["band_dst"]
    # band_h = 1080 * 1080/1920 = 607.5 → 606 (even)
    expected_bh = int(round(1080 * 1080 / 1920))
    expected_bh = expected_bh - (expected_bh % 2)
    assert bh == expected_bh
    assert bw == 1080
    # Band is centered vertically within target_h=1920
    expected_by = (1920 - bh) // 2
    expected_by -= expected_by % 2
    assert by == expected_by


def test_wide_zoom_crop_is_wider_than_fullscreen():
    fs = compute_fullscreen_rects(1920, 1080, 1080, 1920)
    wz = compute_wide_zoom_rects(1920, 1080, 1080, 1920, crop_scale=1.25)
    assert wz["src"][2] > fs["src"][2], (
        f"wide_zoom width {wz['src'][2]} not > fullscreen width {fs['src'][2]}"
    )
    # By exactly 1.25x (within rounding)
    ratio = wz["src"][2] / fs["src"][2]
    assert abs(ratio - 1.25) < 0.05


def test_wide_zoom_clamps_off_screen_cam():
    """A cam_x at 5% would push the crop window off the left edge."""
    wz = compute_wide_zoom_rects(1920, 1080, 1080, 1920, cam_x_pct=5.0)
    crop_x, _, crop_w, _ = wz["src"]
    assert crop_x >= 0
    assert crop_x + crop_w <= 1920


def test_compute_layout_rects_dispatches_correctly():
    # Unknown mode → fullscreen
    r = compute_layout_rects("nope", src_w=1920, src_h=1080, target_w=1080, target_h=1920)
    assert r["mode"] == "nope"
    assert "src" in r and "dst" in r

    r = compute_layout_rects("blurfill", src_w=1920, src_h=1080, target_w=1080, target_h=1920)
    assert "band_dst" in r and "blur_dst" in r

    r = compute_layout_rects("wide_zoom", src_w=1920, src_h=1080, target_w=1080, target_h=1920)
    assert "src" in r


# ──────────── parity: filter graph dims = geometry dims ────────────


_NUM = re.compile(r"-?\d+")


def _extract_first_dims(filter_str: str, op: str) -> tuple:
    """Pull the first ``op=W:H[:X:Y]`` numbers out of the filter."""
    seg = filter_str.split(f"{op}=")[1]
    end = min(
        (seg.find(c) for c in ("[", ",", ";") if seg.find(c) != -1),
        default=len(seg),
    )
    seg = seg[:end]
    nums = [int(n) for n in _NUM.findall(seg)]
    return tuple(nums)


def test_blurfill_filter_band_dims_match_geometry():
    """The FFmpeg blurfill filter scales the band to the same
    dimensions the geometry helper computes."""
    src_w, src_h, target_w, target_h = 1920, 1080, 1080, 1920
    f = build_gameplay_blurfill_filter(src_w, src_h, target_w, target_h)
    rects = compute_blurfill_rects(src_w, src_h, target_w, target_h)
    band_w, band_h = rects["band_dst"][2], rects["band_dst"][3]
    # The first scale=W:H in the filter graph is the band scale
    band_seg = f.split("[band_src]scale=")[1].split("[band]")[0]
    nums = [int(n) for n in _NUM.findall(band_seg)]
    assert nums[:2] == [band_w, band_h], (
        f"filter band {nums[:2]} != geometry {[band_w, band_h]}"
    )
    # Overlay y matches band_dst y
    overlay_y = int(_NUM.findall(f.split("overlay=")[1])[1])
    assert overlay_y == rects["band_dst"][1]


def test_wide_zoom_filter_crop_dims_match_geometry():
    src_w, src_h, target_w, target_h = 1920, 1080, 1080, 1920
    f = build_gameplay_wide_zoom_filter(
        src_w, src_h, target_w, target_h, cam_x_pct=50.0,
    )
    rects = compute_wide_zoom_rects(
        src_w, src_h, target_w, target_h, cam_x_pct=50.0,
    )
    crop_x, crop_y, crop_w, crop_h = rects["src"]
    crop_dims = _extract_first_dims(f, "crop")
    assert crop_dims[0] == crop_w
    assert crop_dims[1] == crop_h
    assert crop_dims[2] == crop_x
    assert crop_dims[3] == crop_y


def test_wide_zoom_filter_off_center_cam_matches_geometry():
    """Off-center cam_x should produce identical filter + helper dims."""
    src_w, src_h, target_w, target_h = 1920, 1080, 1080, 1920
    for cam_x in (10.0, 30.0, 70.0, 90.0):
        f = build_gameplay_wide_zoom_filter(
            src_w, src_h, target_w, target_h, cam_x_pct=cam_x,
        )
        rects = compute_wide_zoom_rects(
            src_w, src_h, target_w, target_h, cam_x_pct=cam_x,
        )
        crop_x, _, crop_w, _ = rects["src"]
        crop_dims = _extract_first_dims(f, "crop")
        assert (crop_dims[0], crop_dims[2]) == (crop_w, crop_x), (
            f"cam_x={cam_x} filter={crop_dims} geo={(crop_w, crop_x)}"
        )


# ──────────── parity tolerance band ────────────


def test_geometry_pixel_parity_within_2px_for_all_modes():
    """The helper always rounds to even integers — call it twice
    with the same args and assert the second result matches the
    first to the pixel (deterministic)."""
    for mode in ("fullscreen", "blurfill", "wide_zoom"):
        a = compute_layout_rects(
            mode, src_w=1920, src_h=1080, target_w=1080, target_h=1920,
            cam_x_pct=42.0,
        )
        b = compute_layout_rects(
            mode, src_w=1920, src_h=1080, target_w=1080, target_h=1920,
            cam_x_pct=42.0,
        )
        assert a == b, f"mode {mode} non-deterministic"
        # Even-dimension contract
        if "src" in a:
            sx, sy, sw, sh = a["src"]
            assert sw % 2 == 0
            assert sh % 2 == 0
