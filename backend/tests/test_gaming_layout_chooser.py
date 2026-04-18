"""Phase 4 — gaming layout chooser + blurfill / wide-zoom filter tests.

The chooser is a pure decision tree gated on:

  - Scoreboard / tab events → blurfill.
  - Chaos events / high motion → wide_zoom.
  - Per-genre defaults with duration / motion overrides.

These tests pin the decision boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest


from backend.services.gaming_event_detector import GamingEvent
from backend.services.gaming_layout_chooser import (
    VALID_LAYOUTS,
    assign_gaming_layouts,
    choose_gaming_layout,
    events_in_range,
    has_event_kind,
    normalize_genre,
)


@dataclass
class _Profile:
    gameplay_subtype: str = "fps"
    content_type: str = "gameplay_fps"


@dataclass
class _Seg:
    start: float
    end: float
    gaming_layout_mode: object = None


def test_normalize_genre_from_subtype():
    assert normalize_genre(_Profile(gameplay_subtype="moba")) == "moba"
    assert normalize_genre(_Profile(gameplay_subtype="FPS ")) == "fps"


def test_normalize_genre_from_content_type():
    p = _Profile(gameplay_subtype="", content_type="gameplay_tps")
    assert normalize_genre(p) == "tps"


def test_normalize_genre_default_when_unknown():
    assert normalize_genre(None) == "fps"
    p = _Profile(gameplay_subtype="", content_type="vlog")
    assert normalize_genre(p) == "fps"


def test_has_event_kind_detects_match():
    e1 = GamingEvent(timestamp=1.0, duration=0.8, kind="kill")
    e2 = GamingEvent(timestamp=2.0, duration=0.8, kind="ult")
    assert has_event_kind([e1, e2], "ult")
    assert has_event_kind([e1, e2], "scoreboard", "kill")
    assert not has_event_kind([e1, e2], "rotation")
    assert not has_event_kind([], "kill")


def test_events_in_range_filters_by_overlap():
    events = [
        GamingEvent(timestamp=0.5, duration=0.8, kind="kill"),
        GamingEvent(timestamp=5.0, duration=0.8, kind="kill"),
        GamingEvent(timestamp=10.0, duration=0.8, kind="kill"),
    ]
    in_seg = events_in_range(events, 4.0, 6.0)
    assert len(in_seg) == 1
    assert in_seg[0].timestamp == 5.0


def test_chooser_scoreboard_forces_blurfill():
    events = [GamingEvent(timestamp=2.0, duration=1.0, kind="scoreboard")]
    mode = choose_gaming_layout(
        seg_start=1.0, seg_end=4.0,
        events_in_seg=events, motion_in_seg=5.0,
        profile=_Profile(gameplay_subtype="fps"),
    )
    assert mode == "blurfill"


def test_chooser_chaos_forces_wide_zoom():
    events = [GamingEvent(timestamp=2.0, duration=1.0, kind="chaos")]
    mode = choose_gaming_layout(
        seg_start=1.0, seg_end=4.0,
        events_in_seg=events, motion_in_seg=5.0,
        profile=_Profile(gameplay_subtype="fps"),
    )
    assert mode == "wide_zoom"


def test_chooser_high_motion_forces_wide_zoom():
    mode = choose_gaming_layout(
        seg_start=1.0, seg_end=3.0,
        events_in_seg=[], motion_in_seg=50.0,
        profile=_Profile(gameplay_subtype="fps"),
    )
    assert mode == "wide_zoom"


def test_chooser_moba_long_segment_blurfill():
    """A 5 s MOBA segment with no events → blurfill (lane fight)."""
    mode = choose_gaming_layout(
        seg_start=10.0, seg_end=15.0,
        events_in_seg=[], motion_in_seg=10.0,
        profile=_Profile(gameplay_subtype="moba"),
    )
    assert mode == "blurfill"


def test_chooser_moba_short_segment_composite():
    """A 2 s MOBA segment → composite (HUD + small lane crop)."""
    mode = choose_gaming_layout(
        seg_start=10.0, seg_end=12.0,
        events_in_seg=[], motion_in_seg=10.0,
        profile=_Profile(gameplay_subtype="moba"),
    )
    assert mode == "composite"


def test_chooser_racing_high_motion_fullscreen():
    mode = choose_gaming_layout(
        seg_start=0.0, seg_end=3.0,
        events_in_seg=[], motion_in_seg=25.0,
        profile=_Profile(gameplay_subtype="racing"),
    )
    assert mode == "fullscreen"


def test_chooser_racing_low_motion_composite():
    mode = choose_gaming_layout(
        seg_start=0.0, seg_end=3.0,
        events_in_seg=[], motion_in_seg=5.0,
        profile=_Profile(gameplay_subtype="racing"),
    )
    assert mode == "composite"


def test_chooser_tps_default_fullscreen():
    mode = choose_gaming_layout(
        seg_start=0.0, seg_end=3.0,
        events_in_seg=[], motion_in_seg=5.0,
        profile=_Profile(gameplay_subtype="tps"),
    )
    assert mode == "fullscreen"


def test_chooser_fps_default_fullscreen():
    mode = choose_gaming_layout(
        seg_start=0.0, seg_end=3.0,
        events_in_seg=[], motion_in_seg=5.0,
        profile=_Profile(gameplay_subtype="fps"),
    )
    assert mode == "fullscreen"


def test_chooser_returns_valid_layout_always():
    """Any combination of inputs must return one of the valid modes."""
    for genre in ("fps", "moba", "tps", "racing", "sandbox", "stream"):
        mode = choose_gaming_layout(
            seg_start=0.0, seg_end=2.0,
            events_in_seg=[], motion_in_seg=10.0,
            profile=_Profile(gameplay_subtype=genre),
        )
        assert mode in VALID_LAYOUTS, f"genre {genre} → {mode}"


def test_assign_gaming_layouts_in_place():
    """``assign_gaming_layouts`` populates each segment's mode."""
    segs = [
        _Seg(start=0.0, end=2.0),
        _Seg(start=2.0, end=8.0),  # long → blurfill for moba
        _Seg(start=8.0, end=10.0, gaming_layout_mode="composite"),  # already set
    ]
    n = assign_gaming_layouts(
        segs,
        events=[],
        motion_by_segment={id(segs[0]): 5.0, id(segs[1]): 5.0},
        profile=_Profile(gameplay_subtype="moba"),
    )
    assert n == 2  # third one was preserved
    assert segs[0].gaming_layout_mode == "composite"  # short MOBA
    assert segs[1].gaming_layout_mode == "blurfill"   # long MOBA
    assert segs[2].gaming_layout_mode == "composite"  # untouched


# ── Filter graph smoke tests ──


def test_blurfill_filter_outputs_v_label():
    from backend.services.gameplay_filters import (
        build_gameplay_blurfill_filter,
    )
    f = build_gameplay_blurfill_filter(1920, 1080, 1080, 1920)
    assert f.endswith("[v]")
    assert "boxblur" in f
    assert "split=2" in f
    assert "overlay" in f


def test_wide_zoom_filter_outputs_v_label():
    from backend.services.gameplay_filters import (
        build_gameplay_wide_zoom_filter,
    )
    f = build_gameplay_wide_zoom_filter(1920, 1080, 1080, 1920)
    assert f.endswith("[v]")
    assert "crop=" in f
    assert "scale=" in f


def test_wide_zoom_filter_clamps_off_screen_cam():
    """Cam_x at 5% should still produce a valid filter (clamped)."""
    from backend.services.gameplay_filters import (
        build_gameplay_wide_zoom_filter,
    )
    f = build_gameplay_wide_zoom_filter(
        1920, 1080, 1080, 1920, cam_x_pct=5.0,
    )
    # The crop_x must be ≥ 0 in the output
    assert "crop=" in f
    # Pull the crop_x out and verify it's non-negative
    crop_part = f.split("crop=")[1].split(",")[0]
    parts = crop_part.split(":")
    assert int(parts[2]) >= 0


def test_blurfill_filter_handles_unusual_aspect():
    from backend.services.gameplay_filters import (
        build_gameplay_blurfill_filter,
    )
    # 21:9 ultrawide source
    f = build_gameplay_blurfill_filter(2560, 1080, 1080, 1920)
    assert f.endswith("[v]")
    # Band height must be even
    band_part = [p for p in f.split(";") if "scale=1080:" in p][0]
    band_h = int(band_part.split("scale=1080:")[1].split("[")[0])
    assert band_h % 2 == 0

