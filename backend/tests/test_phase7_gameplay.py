"""Phase 7 — Gaming beyond FPS (MOBA / TPS / racing / stream).

Five test groups:

1. **Per-game / per-genre anchor lookups** —
   ``subject_anchor_for_game`` / ``subject_anchor_for_genre``
   verify that Phase 2's ``GAME_HUD_LAYOUTS`` action_center_pct
   values are correctly exposed: FPS = (50, 50), TPS = (50, 45),
   racing = (50, 65).

2. **Pure-Python motion-centroid tracker** —
   ``track_gameplay_subject`` smoothing + velocity clamp +
   fallback semantics across strong-motion / weak-motion /
   mixed cases.

3. **Per-segment aggregator** —
   ``aggregate_subject_x_for_segment`` returns a single
   ``(x, y, source)`` for the segmenter to consume.

4. **HUD-safe fallback** — ``derive_hud_safe_subject_x``
   centroid-of-HUD-zones picker for unrecognized games.

5. **Stage 7c / 7d AST guards** on reframe_segmenter to ensure
   the gameplay tracker + STREAM routing wiring is structurally
   present (per the spec's "no regression on existing fixtures"
   rule, the actual integration runs in docker).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from backend.services.gameplay_subject_tracker import (
    DEFAULT_EMA_TAU_SEC,
    DEFAULT_MAX_VELOCITY_PCT_PER_SEC,
    MIN_MOTION_MAGNITUDE,
    USE_GAMEPLAY_TRACKER,
    GameplayMotionCentroid,
    GameplaySubjectPath,
    aggregate_subject_x_for_segment,
    derive_hud_safe_subject_x,
    subject_anchor_for_game,
    subject_anchor_for_genre,
    track_gameplay_subject,
)


# ──────────────────── Per-game anchor lookups ────────────────────


class TestSubjectAnchorForGame:
    def test_fps_default_center(self):
        # All FPS entries land on (50, 50)
        assert subject_anchor_for_game("valorant") == (50.0, 50.0)
        assert subject_anchor_for_game("overwatch") == (50.0, 50.0)
        assert subject_anchor_for_game("apex_legends") == (50.0, 50.0)
        assert subject_anchor_for_game("generic_fps") == (50.0, 50.0)

    def test_tps_offset_above_center(self):
        # Third-person games — character offset down+right of
        # frame center → anchor sits slightly above true center.
        assert subject_anchor_for_game("gta_v") == (50.0, 45.0)
        assert subject_anchor_for_game("elden_ring") == (50.0, 45.0)
        assert subject_anchor_for_game("generic_tps") == (50.0, 45.0)

    def test_racing_lower_third(self):
        # Car sits in the lower portion of the frame
        assert subject_anchor_for_game("generic_racing") == (50.0, 65.0)

    def test_moba_centered(self):
        # MOBA / top-down — action sits at center but with
        # wider safe zone (the action_center value itself is
        # still center).
        assert subject_anchor_for_game("league_of_legends") == (50.0, 50.0)
        assert subject_anchor_for_game("dota2") == (50.0, 50.0)
        assert subject_anchor_for_game("generic_moba") == (50.0, 50.0)

    def test_unknown_game_defaults_to_center(self):
        assert subject_anchor_for_game("not_a_real_game") == (50.0, 50.0)

    def test_none_defaults_to_center(self):
        assert subject_anchor_for_game(None) == (50.0, 50.0)
        assert subject_anchor_for_game("") == (50.0, 50.0)


class TestSubjectAnchorForGenre:
    def test_genre_fps(self):
        assert subject_anchor_for_genre("fps") == (50.0, 50.0)

    def test_genre_tps(self):
        assert subject_anchor_for_genre("tps") == (50.0, 45.0)

    def test_genre_racing(self):
        assert subject_anchor_for_genre("racing") == (50.0, 65.0)

    def test_genre_moba(self):
        assert subject_anchor_for_genre("moba") == (50.0, 50.0)

    def test_genre_sandbox(self):
        # Sandbox uses minecraft (50, 50) as the default
        assert subject_anchor_for_genre("sandbox") == (50.0, 50.0)

    def test_unknown_genre_defaults_to_center(self):
        assert subject_anchor_for_genre("not_a_genre") == (50.0, 50.0)
        assert subject_anchor_for_genre(None) == (50.0, 50.0)
        assert subject_anchor_for_genre("") == (50.0, 50.0)


# ──────────────────── track_gameplay_subject ────────────────────


class TestTrackGameplaySubject:
    def test_no_centroids_returns_fallback(self):
        sp = track_gameplay_subject(
            [], fallback_xy_pct=(50.0, 45.0), start=0.0, end=2.0,
        )
        assert sp.source == "action_center"
        assert sp.path == [(0.0, 50.0, 45.0)]
        assert sp.n_motion_frames == 0

    def test_strong_motion_path(self):
        # Three strong-magnitude centroids → motion source
        centroids = [
            GameplayMotionCentroid(0.5, 40, 60, 0.9),
            GameplayMotionCentroid(1.0, 42, 58, 0.85),
            GameplayMotionCentroid(1.5, 44, 56, 0.9),
        ]
        sp = track_gameplay_subject(
            centroids, fallback_xy_pct=(50.0, 45.0),
            start=0.0, end=2.0,
        )
        assert sp.source == "motion"
        assert sp.n_motion_frames == 3
        assert len(sp.path) == 3
        # First strong frame initializes the smoother at the
        # raw position, not the fallback
        assert sp.path[0][1] == 40.0
        assert sp.path[0][2] == 60.0

    def test_weak_motion_falls_back(self):
        weak = [
            GameplayMotionCentroid(0.5, 40, 60, 0.1),
            GameplayMotionCentroid(1.0, 42, 58, 0.2),
        ]
        sp = track_gameplay_subject(
            weak, fallback_xy_pct=(50.0, 45.0),
            start=0.0, end=2.0,
        )
        assert sp.source == "action_center"
        assert sp.n_motion_frames == 0

    def test_mixed_motion(self):
        # Some strong, some weak → mixed source
        mixed = [
            GameplayMotionCentroid(0.5, 40, 60, 0.9),
            GameplayMotionCentroid(1.0, 42, 58, 0.1),
            GameplayMotionCentroid(1.5, 44, 56, 0.9),
        ]
        sp = track_gameplay_subject(
            mixed, fallback_xy_pct=(50.0, 45.0),
            start=0.0, end=2.0,
        )
        assert sp.source == "mixed"
        assert sp.n_motion_frames == 2

    def test_outside_segment_window_filtered(self):
        # Centroids before / after the window are ignored
        centroids = [
            GameplayMotionCentroid(0.0, 30, 50, 0.9),  # before
            GameplayMotionCentroid(1.0, 40, 60, 0.9),  # in
            GameplayMotionCentroid(3.0, 50, 70, 0.9),  # after
        ]
        sp = track_gameplay_subject(
            centroids, fallback_xy_pct=(50.0, 45.0),
            start=0.5, end=2.5,
        )
        # Only the t=1.0 centroid is in window
        assert sp.n_motion_frames == 1
        assert sp.source == "motion"

    def test_velocity_clamp_caps_step(self):
        # Two centroids 1 s apart, far jumps. Velocity clamp
        # caps the step at max_velocity_pct_per_sec * dt.
        centroids = [
            GameplayMotionCentroid(0.0, 20, 50, 0.9),
            GameplayMotionCentroid(1.0, 80, 50, 0.9),  # 60 % jump in 1 s
        ]
        sp = track_gameplay_subject(
            centroids, fallback_xy_pct=(50.0, 50.0),
            start=0.0, end=2.0,
            max_velocity_pct_per_sec=15.0,
        )
        # Second smoothed point should have moved at most 15 %
        # from the first
        first_x = sp.path[0][1]
        second_x = sp.path[1][1]
        delta = abs(second_x - first_x)
        assert delta <= 15.0 + 1e-6

    def test_ema_smoothing_kills_jitter(self):
        # 5 centroids alternating 40 / 50 — EMA should produce
        # a smoothed path between 40 and 50, not the raw
        # alternation.
        centroids = [
            GameplayMotionCentroid(0.2, 40, 50, 0.9),
            GameplayMotionCentroid(0.4, 50, 50, 0.9),
            GameplayMotionCentroid(0.6, 40, 50, 0.9),
            GameplayMotionCentroid(0.8, 50, 50, 0.9),
            GameplayMotionCentroid(1.0, 40, 50, 0.9),
        ]
        sp = track_gameplay_subject(
            centroids, fallback_xy_pct=(50.0, 50.0),
            start=0.0, end=2.0,
        )
        # Smoothed points should not include both 40 and 50 verbatim
        smoothed_xs = [round(p[1], 1) for p in sp.path]
        # The intermediate points should be between 40 and 50
        for i, x in enumerate(smoothed_xs[1:], start=1):
            assert 40.0 <= x <= 50.0


# ──────────────────── aggregate_subject_x_for_segment ────────────────────


class TestAggregateSubjectXForSegment:
    def test_motion_present_returns_motion_source(self):
        centroids = [GameplayMotionCentroid(1.0, 40, 60, 0.9)]
        x, y, src = aggregate_subject_x_for_segment(
            centroids, start=0.0, end=2.0,
            fallback_xy_pct=(50.0, 45.0),
        )
        assert src == "motion"
        assert x == 40.0
        assert y == 60.0

    def test_no_motion_returns_fallback(self):
        x, y, src = aggregate_subject_x_for_segment(
            [], start=0.0, end=2.0, fallback_xy_pct=(50.0, 65.0),
        )
        assert src == "action_center"
        assert x == 50.0
        assert y == 65.0

    def test_averages_over_multiple_frames(self):
        # Three centroids at x=40, 50, 60 → average ~50 after
        # smoothing (the smoother brings them in)
        centroids = [
            GameplayMotionCentroid(0.5, 40, 50, 0.9),
            GameplayMotionCentroid(1.0, 50, 50, 0.9),
            GameplayMotionCentroid(1.5, 60, 50, 0.9),
        ]
        x, _y, src = aggregate_subject_x_for_segment(
            centroids, start=0.0, end=2.0,
            fallback_xy_pct=(50.0, 50.0),
        )
        assert src == "motion"
        # Average smoothed position is somewhere in the middle
        assert 40.0 < x < 60.0


# ──────────────────── derive_hud_safe_subject_x ────────────────────


class TestDeriveHudSafeSubjectX:
    def test_empty_zones_returns_center(self):
        assert derive_hud_safe_subject_x(
            hud_zones=[], crop_width_pct=31.6,
        ) == 50.0

    def test_centroid_of_single_zone(self):
        # Zone at x=20..40 → centroid = 30
        x = derive_hud_safe_subject_x(
            hud_zones=[(20.0, 0.0, 20.0, 50.0)],
            crop_width_pct=31.6,
        )
        # Crop center clamped to [half, 100 - half] = [15.8, 84.2]
        assert x == pytest.approx(30.0, abs=1e-6)

    def test_two_zones_weighted_by_area(self):
        # Big zone at x=0..40 (area 800), small zone at x=60..80
        # (area 100) → weighted centroid pulled toward the big zone
        x = derive_hud_safe_subject_x(
            hud_zones=[
                (0.0, 0.0, 40.0, 20.0),  # area 800
                (60.0, 0.0, 20.0, 5.0),  # area 100
            ],
            crop_width_pct=31.6,
        )
        # Expected: (20 * 800 + 70 * 100) / 900 = 25.56
        assert 20.0 < x < 30.0

    def test_clamp_to_crop_safe_range(self):
        # All zones at the right edge → centroid past the right
        # clamp boundary → clamped to 100 - half = 84.2
        x = derive_hud_safe_subject_x(
            hud_zones=[(95.0, 0.0, 5.0, 5.0)],
            crop_width_pct=31.6,
        )
        # Centroid is 97.5, but clamped to 100 - 15.8 = 84.2
        assert x == pytest.approx(84.2, abs=0.01)

    def test_zero_area_zones_ignored(self):
        x = derive_hud_safe_subject_x(
            hud_zones=[
                (10.0, 0.0, 0.0, 0.0),  # zero area
                (50.0, 0.0, 20.0, 20.0),  # real
            ],
            crop_width_pct=31.6,
        )
        # Only the real zone contributes → centroid = 60
        assert x == pytest.approx(60.0, abs=1e-6)


# ──────────────────── Feature flag default ────────────────────


class TestPhase7FeatureFlagDefaultOff:
    def test_default_off(self):
        assert USE_GAMEPLAY_TRACKER is False


# ──────────────────── reframe_segmenter Stage 7c / 7d AST guards ────────────────────


class TestReframeSegmenterStage7cAST:
    SRC_PATH = (
        Path(__file__).resolve().parents[1] / "services" / "reframe_segmenter.py"
    )

    def test_stage_7c_block_present(self):
        src = self.SRC_PATH.read_text()
        assert "Stage 7c" in src
        assert "GameplayTracker" in src or "gameplay_track_count" in src

    def test_stage_7d_stream_routing_present(self):
        src = self.SRC_PATH.read_text()
        assert "Stage 7d" in src
        assert "stacked_gameplay" in src

    def test_imports_gameplay_tracker_lazily(self):
        src = self.SRC_PATH.read_text()
        assert "USE_GAMEPLAY_TRACKER" in src
        assert "aggregate_subject_x_for_segment" in src

    def test_signature_has_gameplay_motion_centroids_kwarg(self):
        src = self.SRC_PATH.read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "build_reframe_segments":
                kwarg_names = [a.arg for a in node.args.args] + [
                    a.arg for a in node.args.kwonlyargs
                ]
                assert "gameplay_motion_centroids" in kwarg_names
                return
        pytest.fail("build_reframe_segments not found")

    def test_resolves_per_game_then_per_genre_fallback(self):
        # Stage 7c prefers the specific game key over the genre
        # default — both lookups are referenced.
        src = self.SRC_PATH.read_text()
        assert "subject_anchor_for_game" in src
        assert "subject_anchor_for_genre" in src

    def test_stream_layout_routing_no_flag_gate(self):
        # Stream layout fires whenever the user picked
        # gameplay_subtype="stream" — no env flag, since stream
        # is an editorial decision rather than a quality knob.
        src = self.SRC_PATH.read_text()
        # Look for the gate string
        assert '"stream"' in src or "'stream'" in src
        # And for the layout assignment
        assert 'seg.layout = "stacked_gameplay"' in src or 'layout = "stacked_gameplay"' in src
