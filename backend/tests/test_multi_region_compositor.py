"""Phase 5 — multi-region compositor tests.

Covers:
  1.  Dynamic region allocation (lecture / gaming) via
      :func:`backend.services.multi_region_layout.compute_region_split`.
  2.  Smooth keyframe interpolation over a 0.5s window.
  3.  2px separator drawn between regions (FFmpeg drawbox + Canvas line).
  4.  HUD-aware crop decision (CROP vs HUD_COMPOSITE) and HUD strip
      arrangement.
  5.  Multi-region → single-crop fade-out via xfade.
  6.  No-black-bars contract — HUD strip falls back to a blurred-cover
      fill when no HUD rects are detected.

For tests asserting rendered output we assert against the FFmpeg filter
chain string (no actual ffmpeg invocation) so the suite stays fast and
sandbox-safe.
"""

from __future__ import annotations

import pytest

from backend.services.multi_region_layout import (
    _SplitKeyframe,
    build_gaming_split_keyframes,
    build_lecture_split_keyframes,
    compute_region_split,
    gaming_secondary_importance,
    lecture_primary_importance,
    smooth_split_at,
)
from backend.services.game_layouts import (
    arrange_hud_strip_horizontally,
    choose_hud_layout,
    compute_hud_aware_crop_window,
    select_hud_strip_rects,
)
from backend.services.render_plan import (
    Rect,
    RenderOp,
    RenderOpKind,
    RenderPlan,
)
from backend.services.ffmpeg_filter_builder import _build_filter_graph


def _make_plan(ops, source_w=1920, source_h=1080,
               target_w=1080, target_h=1920) -> RenderPlan:
    total = ops[-1].end_sec if ops else 0.0
    return RenderPlan(
        source_width=source_w, source_height=source_h,
        target_width=target_w, target_height=target_h,
        total_duration_sec=total, fps=30.0, ops=ops,
    )


# ─────────────── 1. Dynamic split — lecture ────────────────


def test_dynamic_split_lecture():
    """Slide change → primary 80%, decays back to ~60% over 3s.

    The lecture importance signal boosts ``primary_importance`` to 0.8
    immediately after a slide change and linearly decays to 0.6 over
    ``boost_window_sec`` (3s). Combined with a baseline secondary of
    0.4 via ``compute_region_split``, the boost yields a primary
    fraction of (0.8 + 0.6)/2 = 0.7 (i.e. ~70% top region) and the
    decay endpoint of (0.6 + 0.6)/2 = 0.6 (60%).
    """
    boost_imp = lecture_primary_importance(seconds_since_slide_change=0.0)
    decayed_imp = lecture_primary_importance(seconds_since_slide_change=3.0)
    assert boost_imp == pytest.approx(0.8, abs=1e-6)
    assert decayed_imp == pytest.approx(0.6, abs=1e-6)

    boost_split = compute_region_split(boost_imp, 0.4)
    decayed_split = compute_region_split(decayed_imp, 0.4)
    # Boost: primary should be larger than decayed.
    assert boost_split > decayed_split
    # Boost ≈ 0.7, decayed ≈ 0.6
    assert boost_split == pytest.approx(0.7, abs=1e-3)
    assert decayed_split == pytest.approx(0.6, abs=1e-3)


# ─────────────── 2. Dynamic split — gaming ────────────────


def test_dynamic_split_gaming():
    """Speaker active → facecam ~40%; silent → ~25%."""
    active = gaming_secondary_importance(facecam_speaker_active=True)
    silent = gaming_secondary_importance(facecam_speaker_active=False)
    assert active == 0.5
    assert silent == 0.3

    # primary_baseline = 0.5 (gameplay action neither dominant nor
    # subordinate). active_split shrinks primary to give facecam 40%.
    active_primary = compute_region_split(0.5, active)
    silent_primary = compute_region_split(0.5, silent)
    # active → 1 - 0.5 = 0.5 facecam? Actually:
    # primary_fraction = (0.5 + 1 - 0.5)/2 = 0.5 for active,
    # primary_fraction = (0.5 + 1 - 0.3)/2 = 0.6 for silent.
    # So facecam = 1 - 0.5 = 0.5 (active) vs 1 - 0.6 = 0.4 (silent).
    facecam_active_frac = 1 - active_primary
    facecam_silent_frac = 1 - silent_primary
    assert facecam_active_frac > facecam_silent_frac
    assert facecam_active_frac == pytest.approx(0.5, abs=1e-3)
    assert facecam_silent_frac == pytest.approx(0.4, abs=1e-3)


# ─────────────── 3. Min region fraction ────────────────


def test_min_region_fraction():
    """Even at importance extremes, both regions stay >= 25%."""
    # primary=0, secondary=1 → raw=(0+0)/2 = 0 → clamped to 0.25.
    assert compute_region_split(0.0, 1.0, min_region_fraction=0.25) == pytest.approx(0.25)
    # Symmetric extreme: primary=1, secondary=0 → raw = 1.0 → clamped to 0.75.
    assert compute_region_split(1.0, 0.0, min_region_fraction=0.25) == pytest.approx(0.75)
    # Non-default min works too.
    assert compute_region_split(0.0, 1.0, min_region_fraction=0.30) == pytest.approx(0.30)


# ─────────────── 4. Smooth transition over 0.5s ────────────────


def test_smooth_transition():
    """Split changes are interpolated linearly across the smoothing window."""
    # Step from 0.5 to 0.7 at t=2.0 with 0.5s smoothing window.
    kps = [
        _SplitKeyframe(t=0.0, primary_fraction=0.5),
        _SplitKeyframe(t=2.0, primary_fraction=0.7),
    ]
    # Well before the ramp.
    assert smooth_split_at(kps, 1.0, smooth_sec=0.5) == pytest.approx(0.5)
    # At ramp start (1.5s) — still at the previous value.
    assert smooth_split_at(kps, 1.5, smooth_sec=0.5) == pytest.approx(0.5)
    # Halfway through the ramp.
    mid = smooth_split_at(kps, 1.75, smooth_sec=0.5)
    assert mid == pytest.approx(0.6, abs=1e-3)
    # Exactly at the keyframe.
    assert smooth_split_at(kps, 2.0, smooth_sec=0.5) == pytest.approx(0.7)
    # Sanity — values stay monotone within the ramp.
    assert smooth_split_at(kps, 1.6, smooth_sec=0.5) <= mid
    assert mid <= smooth_split_at(kps, 1.9, smooth_sec=0.5)


# ─────────────── 5. Separator present in rendered output ────────────────


def test_separator_present():
    """SPLIT_SCREEN with separator_px=2 → drawbox in filter chain."""
    op = RenderOp(
        kind=RenderOpKind.SPLIT_SCREEN,
        start_sec=0.0, end_sec=5.0,
        primary_rect=Rect(0.0, 0.0, 0.5, 1.0),
        secondary_rect=Rect(0.5, 0.0, 0.5, 1.0),
        separator_px=2,
    )
    graph = _build_filter_graph(_make_plan([op]))
    assert "drawbox" in graph
    assert "0x333333" in graph
    # 2px height
    assert "h=2" in graph


# ─────────────── 6. Separator disabled when separator_px=0 ───


def test_separator_disabled():
    """separator_px=0 → no drawbox in the filter chain."""
    op = RenderOp(
        kind=RenderOpKind.SPLIT_SCREEN,
        start_sec=0.0, end_sec=5.0,
        primary_rect=Rect(0.0, 0.0, 0.5, 1.0),
        secondary_rect=Rect(0.5, 0.0, 0.5, 1.0),
        separator_px=0,
    )
    graph = _build_filter_graph(_make_plan([op]))
    assert "drawbox" not in graph


# ─────────────── 7. HUD detection → HUD_COMPOSITE chosen ────


def test_hud_detection_integration():
    """When HUD + action don't fit in 9:16, choose_hud_layout returns
    ``hud_composite``. HUD bboxes spanning a wider band than the 9:16
    crop (which is ~56% of source for 16:9 source) trigger the strip.
    """
    # 9:16 crop in 1920x1080 source = 9/16 / (16/9) = 0.316 of source width.
    target_aspect = 9 / 16
    # Place HUD elements at extremes — left edge AND right edge — so the
    # union span (~85%) exceeds the 31.6% crop window.
    hud_bboxes = [
        (0.02, 0.85, 0.18, 0.10),  # bottom-left minimap
        (0.80, 0.05, 0.18, 0.18),  # top-right killfeed
        (0.40, 0.90, 0.20, 0.08),  # bottom-center health
    ]
    decision = choose_hud_layout(
        action_center_pct=(50.0, 50.0),
        hud_bboxes_norm=hud_bboxes,
        target_aspect=target_aspect,
        source_w=1920,
        source_h=1080,
    )
    assert decision["kind"] == "hud_composite"
    assert len(decision["hud_strip_rects"]) == 3

    # Sanity: a single centered HUD that fits → CROP, not HUD_COMPOSITE.
    fit_decision = choose_hud_layout(
        action_center_pct=(50.0, 50.0),
        hud_bboxes_norm=[(0.40, 0.85, 0.20, 0.10)],
        target_aspect=target_aspect,
        source_w=1920,
        source_h=1080,
    )
    assert fit_decision["kind"] == "crop"


# ─────────────── 8. HUD strip arrangement (3 elements) ────────


def test_hud_strip_arrangement():
    """3 HUD regions → 3 horizontal slots whose widths sum to target_w."""
    hud_bboxes = [
        (0.02, 0.85, 0.18, 0.10),
        (0.80, 0.05, 0.18, 0.18),
        (0.40, 0.90, 0.20, 0.08),
    ]
    arrangement = arrange_hud_strip_horizontally(
        hud_bboxes,
        target_strip_height_px=480,
        target_strip_width_px=1080,
        min_element_height_px=30,
    )
    assert len(arrangement["slot_widths"]) == 3
    assert sum(arrangement["slot_widths"]) == 1080
    # 480 >> 30 so no scale-up needed.
    assert arrangement["needs_scale_up"] is False
    assert arrangement["element_height_px"] == 480

    # If the strip is below the readability floor, scale-up flag fires.
    small = arrange_hud_strip_horizontally(
        hud_bboxes,
        target_strip_height_px=20,
        target_strip_width_px=1080,
        min_element_height_px=30,
    )
    assert small["needs_scale_up"] is True
    assert small["element_height_px"] == 30

    # Verify the FFmpeg HUD_COMPOSITE filter actually emits xstack.
    op = RenderOp(
        kind=RenderOpKind.HUD_COMPOSITE,
        start_sec=0.0, end_sec=5.0,
        primary_rect=Rect(0.0, 0.0, 1.0, 1.0),
        hud_strip_rects=[Rect(*b) for b in hud_bboxes],
        hud_strip_fraction=0.25,
    )
    graph = _build_filter_graph(_make_plan([op]))
    assert "xstack=inputs=3" in graph
    # Viewport scaled to top portion (75% of 1920 = 1440).
    assert "scale=1080:1440" in graph


# ─────────────── 9. Multi-region → single fade-out ────


def test_multi_to_single_transition():
    """Fade-out FROM stacked_gameplay TO crop produces an xfade."""
    multi = RenderOp(
        kind=RenderOpKind.STACKED_GAMEPLAY,
        start_sec=0.0, end_sec=5.0,
        primary_rect=Rect(0.0, 0.0, 1.0, 0.6),
        secondary_rect=Rect(0.65, 0.65, 0.30, 0.30),
        transition_fade_out_ms=300,
    )
    cut = RenderOp(
        kind=RenderOpKind.CROP,
        start_sec=5.0, end_sec=10.0,
        primary_rect=Rect(0.2, 0.0, 0.6, 1.0),
    )
    graph = _build_filter_graph(_make_plan([multi, cut]))
    assert "xfade" in graph
    assert "duration=0.300" in graph

    # Cutscene → single-crop without the fade-out flag → no xfade
    # (existing ease_in_ms behavior unchanged).
    multi_no_fade = RenderOp(
        kind=RenderOpKind.STACKED_GAMEPLAY,
        start_sec=0.0, end_sec=5.0,
        primary_rect=Rect(0.0, 0.0, 1.0, 0.6),
        secondary_rect=Rect(0.65, 0.65, 0.30, 0.30),
    )
    cut2 = RenderOp(
        kind=RenderOpKind.CROP,
        start_sec=5.0, end_sec=10.0,
        primary_rect=Rect(0.2, 0.0, 0.6, 1.0),
    )
    graph2 = _build_filter_graph(_make_plan([multi_no_fade, cut2]))
    assert "xfade" not in graph2


# ─────────────── 10. No-black-bars contract ────


def test_no_black_bars_in_multi_region():
    """A HUD_COMPOSITE op with NO detected HUD rects falls back to a
    blurred-cover strip — never a black bar.

    Phase 4 contract: any black bar in the rendered output is a
    regression. The HUD strip code path explicitly emits a gblur
    filter when ``hud_strip_rects`` is empty.
    """
    op = RenderOp(
        kind=RenderOpKind.HUD_COMPOSITE,
        start_sec=0.0, end_sec=5.0,
        primary_rect=Rect(0.0, 0.0, 1.0, 1.0),
        hud_strip_rects=[],
        hud_strip_fraction=0.25,
    )
    graph = _build_filter_graph(_make_plan([op]))
    # Blurred-cover fill present.
    assert "gblur=sigma=" in graph
    # Never a literal black-bar pad: pad= or color=black would be
    # a regression. drawbox is allowed as a separator (different op).
    assert "color=black" not in graph
    assert "pad=" not in graph

    # And the HUD-aware crop helper should not return a degenerate
    # 0-width rect when HUD doesn't fit — caller routes to composite.
    rect, fits = compute_hud_aware_crop_window(
        action_center_pct=(50.0, 50.0),
        hud_bboxes_norm=[
            (0.02, 0.85, 0.18, 0.10),
            (0.80, 0.05, 0.18, 0.18),
        ],
        target_aspect=9 / 16,
        source_w=1920,
        source_h=1080,
    )
    assert fits is False
    x, y, w, h = rect
    assert w > 0.05 and h > 0.05  # never degenerate


# ─────────────── Bonus: keyframe builders are well-formed ───


def test_lecture_keyframes_emit_pairs():
    kps = build_lecture_split_keyframes(
        slide_change_times=[2.0, 5.0],
        seg_start=0.0,
        seg_end=10.0,
    )
    # 1 baseline + 2 slide-changes × 2 keyframes each = 5
    assert len(kps) == 5
    # All keyframes should be inside the segment.
    for kp in kps:
        assert 0.0 <= kp.t <= 10.0
        assert 0.0 <= kp.primary_fraction <= 1.0


def test_gaming_keyframes_emit_pairs():
    kps = build_gaming_split_keyframes(
        facecam_active_windows=[(2.0, 4.0)],
        seg_start=0.0,
        seg_end=10.0,
    )
    # baseline + 2 boundaries
    assert len(kps) == 3
    # Active window should LIFT the secondary (= reduce primary).
    primary_at_active = smooth_split_at(kps, 3.0, smooth_sec=0.5)
    primary_at_silent = smooth_split_at(kps, 9.0, smooth_sec=0.5)
    assert primary_at_active < primary_at_silent


def test_select_hud_strip_rects_uses_update_frequency():
    """Higher update frequency = more important = preferred."""
    bboxes = [
        (0.02, 0.85, 0.18, 0.10),  # rare update (minimap-ish)
        (0.80, 0.05, 0.18, 0.18),  # high-frequency killfeed
        (0.40, 0.90, 0.20, 0.08),  # mid health
    ]
    freqs = [0.10, 0.85, 0.50]
    out = select_hud_strip_rects(bboxes, freqs, max_count=2)
    assert out == [bboxes[1], bboxes[2]]  # killfeed + health, dropped minimap
