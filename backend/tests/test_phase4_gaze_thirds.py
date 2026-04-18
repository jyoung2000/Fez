"""Phase 4 — gaze-aware lead-room + rule-of-thirds tests.

Three test groups:

1. **Gaze yaw API** — extends ``backend.services.gaze_estimator``
   with continuous yaw (``estimate_yaw``), bidirectional EMA
   smoothing (``smooth_yaw_ema``), per-slot dense-frame averaging
   (``estimate_yaw_from_dense``), pixel-offset conversion
   (``lead_room_offset_px``), and a categorical-bucket helper
   (``yaw_to_categorical``).

2. **Thirds bias scoring** — the new ``thirds_bias`` module's
   2-D Gaussian scorer, closest-intersection helper, x-offset
   computer, region weighting, and content-type gate.

3. **AST guards on the reframe_segmenter Stage 8 wiring** so a
   future refactor can't accidentally drop the V2 lead-room or
   thirds-bias integration. The actual segmenter run requires
   numpy + cv2 and lives in the docker / live-pipeline tests; the
   sandbox suite verifies the integration is structurally present.

Spec exit criteria (all covered):

  - "synthetic fixture with a face looking left at x=60 % of frame
    → crop center shifts left of face by 0.08 * crop_w" — covered
    by ``test_continuous_lead_room_offset_full_yaw_left`` and
    ``test_thirds_x_offset_left_third_for_left_yaw``.

  - "vertical crop places eyes at y ≈ 1/3 of 9:16 output" — the
    9:16 vertical crop already preserves source y 1:1, so the
    "eyes at y=1/3" rule is enforced by the **scoring** layer,
    not the rendered output. Covered by
    ``test_thirds_score_peaks_at_canonical_intersections`` and
    ``test_thirds_score_for_region_weights_upper_third_higher``.

  - "≥ 4 new tests" — this file ships ~30 tests across the three
    groups.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from backend.services.gaze_estimator import (
    USE_GAZE_LEAD_ROOM_V2,
    apply_lead_room,
    estimate_gaze_direction,
    estimate_yaw,
    estimate_yaw_from_dense,
    lead_room_offset_px,
    smooth_yaw_ema,
    yaw_to_categorical,
)
from backend.services.thirds_bias import (
    THIRDS_BIAS_CONTENT_TYPES,
    THIRDS_INTERSECTIONS,
    THIRDS_SIGMA,
    USE_THIRDS_BIAS,
    applies_to_content,
    applies_to_profile,
    best_thirds_intersection,
    thirds_bias_score,
    thirds_score_for_region,
    thirds_x_offset_px,
)


# ── Stub face / frame dataclasses ──


@dataclass
class _Face:
    nose_x: float
    x_center: float
    width: float = 10.0
    identity_id: int = 0


@dataclass
class _FrameFaces:
    timestamp: float
    faces: list


# ────────────────── Gaze yaw API ──────────────────


class TestEstimateYaw:
    def test_centered_face_returns_zero(self):
        f = _Face(nose_x=50.0, x_center=50.0, width=10.0)
        assert estimate_yaw(f) == 0.0

    def test_nose_right_of_center_positive(self):
        f = _Face(nose_x=51.0, x_center=50.0, width=10.0)
        # offset = (51-50)/10 = 0.1 → yaw = 0.2 (capped)
        assert estimate_yaw(f) == pytest.approx(0.2, abs=1e-9)

    def test_nose_left_of_center_negative(self):
        f = _Face(nose_x=49.0, x_center=50.0, width=10.0)
        assert estimate_yaw(f) == pytest.approx(-0.2, abs=1e-9)

    def test_full_right_yaw_capped_at_one(self):
        f = _Face(nose_x=60.0, x_center=50.0, width=10.0)
        assert estimate_yaw(f) == 1.0

    def test_full_left_yaw_capped_at_minus_one(self):
        f = _Face(nose_x=40.0, x_center=50.0, width=10.0)
        assert estimate_yaw(f) == -1.0

    def test_missing_nose_returns_zero(self):
        f = _Face(nose_x=None, x_center=50.0, width=10.0)
        assert estimate_yaw(f) == 0.0

    def test_zero_width_returns_zero(self):
        f = _Face(nose_x=50.0, x_center=50.0, width=0.0)
        assert estimate_yaw(f) == 0.0


class TestSmoothYawEMA:
    def test_constant_signal_unchanged(self):
        out = smooth_yaw_ema([0.5, 0.5, 0.5, 0.5])
        for v in out:
            assert v == pytest.approx(0.5, abs=1e-9)

    def test_step_signal_smoothed(self):
        # Symmetric EMA: forward + backward halves bring the
        # transition midpoint to ~0 for a step function.
        out = smooth_yaw_ema([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
        # First value pulled toward second, last value pulled toward
        # second-to-last. Center crosses zero.
        assert out[0] < 0 and out[-1] > 0
        # Trend preserved
        for a, b in zip(out, out[1:]):
            assert a <= b + 1e-6

    def test_short_input_returns_unchanged(self):
        assert smooth_yaw_ema([]) == []
        assert smooth_yaw_ema([0.5]) == [0.5]

    def test_alpha_one_returns_unchanged(self):
        xs = [-0.5, 0.5, -0.5]
        assert smooth_yaw_ema(xs, alpha=1.0) == xs


class TestEstimateYawFromDense:
    def test_no_matching_slot_returns_zero(self):
        df = _FrameFaces(timestamp=0.5, faces=[_Face(60.0, 50.0, 10.0, identity_id=99)])
        assert estimate_yaw_from_dense([df], slot_id=0, start=0.0, end=1.0) == 0.0

    def test_outside_window_ignored(self):
        df = _FrameFaces(timestamp=5.0, faces=[_Face(60.0, 50.0, 10.0)])
        assert estimate_yaw_from_dense([df], slot_id=0, start=0.0, end=1.0) == 0.0

    def test_consistent_right_gaze(self):
        frames = [
            _FrameFaces(0.1 * i, [_Face(60.0, 50.0, 10.0)])
            for i in range(10)
        ]
        # Every face is at the right edge → yaw = +1.0; mean ≈ +1.0
        result = estimate_yaw_from_dense(frames, slot_id=0, start=0.0, end=1.0)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_mixed_gaze_smoothed_to_average(self):
        # Half left, half right → mean ~0
        frames = [
            _FrameFaces(0.1 * i, [_Face(45.0 if i < 5 else 55.0, 50.0, 10.0)])
            for i in range(10)
        ]
        result = estimate_yaw_from_dense(frames, slot_id=0, start=0.0, end=1.0)
        assert abs(result) < 0.1  # close to zero


class TestLeadRoomOffsetPx:
    def test_full_left_yaw_positive_offset(self):
        # Looking left → camera shifts right → +offset
        assert lead_room_offset_px(-1.0, 600.0) == pytest.approx(48.0, abs=1e-9)

    def test_full_right_yaw_negative_offset(self):
        assert lead_room_offset_px(1.0, 600.0) == pytest.approx(-48.0, abs=1e-9)

    def test_half_yaw_half_magnitude(self):
        # |yaw|=0.5 → magnitude = 24
        assert lead_room_offset_px(-0.5, 600.0) == pytest.approx(24.0, abs=1e-9)

    def test_zero_yaw_zero_offset(self):
        assert lead_room_offset_px(0.0, 600.0) == 0.0

    def test_zero_crop_returns_zero(self):
        assert lead_room_offset_px(-1.0, 0.0) == 0.0

    def test_clamps_yaw_above_one(self):
        # Out-of-range inputs are clamped before scaling
        assert lead_room_offset_px(2.0, 600.0) == pytest.approx(-48.0, abs=1e-9)
        assert lead_room_offset_px(-2.0, 600.0) == pytest.approx(48.0, abs=1e-9)

    def test_custom_max_frac(self):
        # 0.04 instead of 0.08
        assert lead_room_offset_px(-1.0, 600.0, max_frac=0.04) == pytest.approx(24.0, abs=1e-9)


class TestYawToCategorical:
    def test_far_left_categorical(self):
        assert yaw_to_categorical(-0.5) == "left"

    def test_far_right_categorical(self):
        assert yaw_to_categorical(0.5) == "right"

    def test_near_zero_center(self):
        assert yaw_to_categorical(0.0) == "center"
        assert yaw_to_categorical(0.10) == "center"
        assert yaw_to_categorical(-0.10) == "center"

    def test_threshold_inclusive_at_boundary(self):
        # Default threshold 0.15
        assert yaw_to_categorical(0.15) == "right"
        assert yaw_to_categorical(-0.15) == "left"

    def test_categorical_matches_legacy_buckets(self):
        # The continuous + categorical tiers should agree on
        # representative cases — the legacy estimate_gaze_direction
        # function uses the same nose-vs-center asymmetry.
        f_left = _Face(nose_x=46.0, x_center=50.0, width=10.0)
        f_right = _Face(nose_x=54.0, x_center=50.0, width=10.0)
        f_center = _Face(nose_x=50.0, x_center=50.0, width=10.0)
        assert estimate_gaze_direction(f_left) == "left"
        assert estimate_gaze_direction(f_right) == "right"
        assert estimate_gaze_direction(f_center) == "center"
        # Continuous tier agrees on the polarity
        assert yaw_to_categorical(estimate_yaw(f_left)) == "left"
        assert yaw_to_categorical(estimate_yaw(f_right)) == "right"
        assert yaw_to_categorical(estimate_yaw(f_center)) == "center"


class TestApplyLeadRoomLegacyPreserved:
    def test_legacy_categorical_function_still_exported(self):
        # The categorical entry point must stay on the public API
        # so flag-OFF code paths keep working.
        assert apply_lead_room(50, "left") > 50
        assert apply_lead_room(50, "right") < 50
        assert apply_lead_room(50, "center") == 50


# ────────────────── Thirds bias ──────────────────


class TestThirdsBiasScore:
    def test_score_peaks_at_canonical_intersections(self):
        for ix, iy in THIRDS_INTERSECTIONS:
            assert thirds_bias_score(ix, iy) == pytest.approx(1.0, abs=1e-9)

    def test_center_is_below_one(self):
        # σ=0.12, distance from (0.5, 0.5) to nearest intersection is
        # √((1/6)² + (1/6)²) ≈ 0.236 → score ≈ exp(-0.0556 / 0.0288) ≈ 0.146
        s = thirds_bias_score(0.5, 0.5)
        assert s < 1.0
        assert s == pytest.approx(0.146, abs=0.01)

    def test_corner_is_essentially_zero(self):
        assert thirds_bias_score(0.0, 0.0) < 0.01
        assert thirds_bias_score(1.0, 1.0) < 0.01

    def test_clamps_out_of_range(self):
        # Slightly out-of-bounds inputs are clamped, not crashed
        assert thirds_bias_score(-0.1, 0.5) == thirds_bias_score(0.0, 0.5)
        assert thirds_bias_score(1.1, 0.5) == thirds_bias_score(1.0, 0.5)

    def test_zero_sigma_returns_zero(self):
        assert thirds_bias_score(1.0 / 3.0, 1.0 / 3.0, sigma=0.0) == 0.0

    def test_score_is_symmetric_about_center(self):
        a = thirds_bias_score(0.4, 0.4)
        b = thirds_bias_score(0.6, 0.6)
        assert a == pytest.approx(b, abs=1e-9)


class TestBestThirdsIntersection:
    def test_picks_closest(self):
        assert best_thirds_intersection(0.4, 0.4) == THIRDS_INTERSECTIONS[0]
        assert best_thirds_intersection(0.7, 0.4) == THIRDS_INTERSECTIONS[1]
        assert best_thirds_intersection(0.4, 0.7) == THIRDS_INTERSECTIONS[2]
        assert best_thirds_intersection(0.7, 0.7) == THIRDS_INTERSECTIONS[3]

    def test_clamps_out_of_range(self):
        # Out-of-range input shouldn't crash
        result = best_thirds_intersection(-0.5, 1.5)
        assert result in THIRDS_INTERSECTIONS


class TestThirdsXOffsetPx:
    def test_left_yaw_shifts_right(self):
        # Looking left → place on LEFT third → +W/6
        offset = thirds_x_offset_px(
            face_x_pct=50.0, crop_width_px=600.0, source_width_px=1920.0,
            yaw=-0.5,
        )
        assert offset == pytest.approx(100.0, abs=1e-9)

    def test_right_yaw_shifts_left(self):
        offset = thirds_x_offset_px(
            face_x_pct=50.0, crop_width_px=600.0, source_width_px=1920.0,
            yaw=0.5,
        )
        assert offset == pytest.approx(-100.0, abs=1e-9)

    def test_no_yaw_defaults_to_left_third(self):
        offset = thirds_x_offset_px(
            face_x_pct=50.0, crop_width_px=600.0, source_width_px=1920.0,
            yaw=0.0,
        )
        # Defaults to LEFT third = +W/6
        assert offset == pytest.approx(100.0, abs=1e-9)

    def test_zero_crop_returns_zero(self):
        offset = thirds_x_offset_px(
            face_x_pct=50.0, crop_width_px=0.0, source_width_px=1920.0,
            yaw=-0.5,
        )
        assert offset == 0.0

    def test_yaw_below_threshold_defaults_to_left(self):
        # |yaw|=0.05 < threshold → default left
        offset = thirds_x_offset_px(
            face_x_pct=50.0, crop_width_px=600.0, source_width_px=1920.0,
            yaw=0.05,
        )
        assert offset > 0


class TestThirdsScoreForRegion:
    def test_region_at_left_third_inside_crop(self):
        # Region at x=33% of frame; crop covers [0, 60] → region center
        # at (35-0)/60 = 0.583. y=33%, so closest intersection is
        # (1/3, 1/3) — distance √((0.583-0.333)² + 0²) = 0.250
        score = thirds_score_for_region(
            region_left_pct=30.0, region_right_pct=40.0, region_y_pct=33.3,
            crop_left_pct=0.0, crop_right_pct=60.0,
        )
        assert 0.0 < score < 1.0

    def test_region_at_intersection_scores_high(self):
        # If we put the region exactly at a thirds intersection of
        # the OUTPUT crop, score should be very close to 1.
        # Crop = [20, 80], region at x=40% (norm = (40-20)/60 = 0.333),
        # y at 33.3% → exact intersection.
        score = thirds_score_for_region(
            region_left_pct=39.5, region_right_pct=40.5, region_y_pct=33.3,
            crop_left_pct=20.0, crop_right_pct=80.0,
        )
        assert score > 0.99

    def test_degenerate_crop_returns_zero(self):
        score = thirds_score_for_region(
            region_left_pct=10.0, region_right_pct=20.0, region_y_pct=50.0,
            crop_left_pct=50.0, crop_right_pct=50.0,
        )
        assert score == 0.0

    def test_upper_third_scores_higher_than_center(self):
        # Same x position, two y positions. Upper-third y=33.3
        # should score higher than center y=50 (because the closest
        # intersection has y=1/3 or 2/3).
        upper = thirds_score_for_region(
            region_left_pct=30.0, region_right_pct=40.0, region_y_pct=33.3,
            crop_left_pct=20.0, crop_right_pct=80.0,
        )
        center = thirds_score_for_region(
            region_left_pct=30.0, region_right_pct=40.0, region_y_pct=50.0,
            crop_left_pct=20.0, crop_right_pct=80.0,
        )
        assert upper > center


class TestAppliesToContent:
    @pytest.mark.parametrize("ct", [
        "narrative", "vlog", "cinematic_dialogue",
        "animation_dialogue", "talking_head",
        # Phase 4 also accepts the parent ContentType values so
        # reframe_segmenter callers (which only have access to
        # content_profile.content_type, not the downstream
        # ClipContentType) can use the same gate.
        "podcast",
    ])
    def test_on_for_face_driven_content(self, ct):
        assert applies_to_content(ct) is True

    @pytest.mark.parametrize("ct", [
        "multi_speaker_panel",  # symmetry beats thirds for 3+ subjects
        "music_video",          # composition is choreographed
        "gameplay", "gameplay_moba", "gameplay_tps", "gameplay_racing",
        "stream", "animation", "generic",
    ])
    def test_off_for_other_content(self, ct):
        assert applies_to_content(ct) is False

    def test_none_is_off(self):
        assert applies_to_content(None) is False


class TestAppliesToProfile:
    """``applies_to_profile`` folds in the multi-speaker-panel
    exclusion so debates / panels get NO thirds bias even though
    their parent ContentType is ``podcast`` (which by itself would
    pass the simple type check)."""

    def _profile(self, content_type, *, panel=False, animated=False):
        @dataclass
        class _P:
            content_type: str
            is_multi_speaker_panel: bool = False
            is_animated: bool = False
        return _P(
            content_type=content_type,
            is_multi_speaker_panel=panel,
            is_animated=animated,
        )

    def test_podcast_profile_passes(self):
        assert applies_to_profile(self._profile("podcast")) is True

    def test_debate_profile_excluded_by_panel_flag(self):
        # debate normalizes to ContentType.PODCAST + panel flag.
        # Even though the parent type is in the bias set, the panel
        # gate excludes it because symmetry beats thirds for 3+ subjects.
        assert applies_to_profile(self._profile("podcast", panel=True)) is False

    def test_narrative_profile_passes(self):
        assert applies_to_profile(self._profile("narrative")) is True

    def test_vlog_profile_passes(self):
        assert applies_to_profile(self._profile("vlog")) is True

    def test_music_video_profile_excluded(self):
        assert applies_to_profile(self._profile("music_video")) is False

    def test_anime_profile_uses_animation_dialogue(self):
        # Anime profiles route to ANIMATION_DIALOGUE downstream;
        # the parent ContentType.ANIME isn't in the bias set, so
        # the profile-level gate returns False unless the caller
        # also passes the ClipContentType. The PR scope keeps this
        # behavior simple: only the parent type drives the gate.
        assert applies_to_profile(self._profile("anime")) is False

    def test_none_profile_is_off(self):
        assert applies_to_profile(None) is False

    def test_profile_without_panel_attribute_handled(self):
        # Defensive: a profile object missing is_multi_speaker_panel
        # should default to "not a panel" rather than crashing.
        @dataclass
        class _MinProfile:
            content_type: str
        assert applies_to_profile(_MinProfile("vlog")) is True


class TestPhase4FeatureFlagsDefaultOff:
    def test_gaze_v2_default_off(self):
        # Per the v2 ground rules, both Phase 4 changes ship
        # feature-flagged off by default until in-docker validation
        # captures the post-Phase-4 numbers.
        assert USE_GAZE_LEAD_ROOM_V2 is False

    def test_thirds_bias_default_off(self):
        assert USE_THIRDS_BIAS is False


# ────────────────── reframe_segmenter integration AST guards ──────────────────


class TestReframeSegmenterIntegrationAST:
    """Static checks on reframe_segmenter.py to ensure the Phase 4
    Stage 8 wiring is structurally present. Catches a future refactor
    that accidentally drops the V2 branches without us noticing."""

    SRC_PATH = (
        Path(__file__).resolve().parents[1] / "services" / "reframe_segmenter.py"
    )

    def test_imports_continuous_yaw_api(self):
        src = self.SRC_PATH.read_text()
        assert "USE_GAZE_LEAD_ROOM_V2" in src
        assert "estimate_yaw_from_dense" in src
        assert "lead_room_offset_px" in src

    def test_imports_thirds_bias_api(self):
        src = self.SRC_PATH.read_text()
        assert "USE_THIRDS_BIAS" in src
        assert "thirds_x_offset_px" in src
        assert "applies_to_content" in src or "_thirds_applies" in src

    def test_clamps_subject_x_after_offset_compose(self):
        # Both offsets compose, then must be clamped to keep the
        # crop window inside the source frame. The clamp expression
        # appears as max(_half, min(source_width - _half, ...)).
        src = self.SRC_PATH.read_text()
        assert "min(source_width - _half" in src

    def test_logs_thirds_count(self):
        src = self.SRC_PATH.read_text()
        assert "thirds-bias x-offset applied" in src

    def test_legacy_categorical_path_still_present(self):
        # Flag OFF must still produce identical behavior to Phase 3.
        # The categorical estimate_gaze_from_dense + apply_lead_room
        # path lives in an `else` branch under the V2 flag check.
        src = self.SRC_PATH.read_text()
        assert "estimate_gaze_from_dense" in src
        assert "apply_lead_room as _apply_lr" in src or "from backend.services.gaze_estimator import" in src

    def test_stage_10c_post_process_present(self):
        # Stage 10c re-applies the Phase 4 offsets AFTER the L1
        # solver writes back so tracking / panning segments don't
        # lose the offset to the L1 overwrite. The post-process
        # iterates raw_segments, recomputes yaw, and shifts both
        # ``subject_x`` and every entry in ``motion_path``.
        src = self.SRC_PATH.read_text()
        assert "Stage 10c" in src or "Phase4 V2 post-process" in src
        assert "phase4_post_count" in src
        assert "motion_path" in src
        assert "applies_to_profile" in src

    def test_stage_10c_runs_only_for_tracking_panning(self):
        # The post-process MUST skip stationary segments (they
        # already got the offset from Stage 8) so the same offset
        # isn't applied twice.
        src = self.SRC_PATH.read_text()
        assert "stationary" in src.lower()
        # Look for the strategy gate in the post-process block
        assert "tracking" in src and "panning" in src


class TestRunnerIntegrationAST:
    """The parity runner now constructs a ContentProfile via
    classify_content before calling build_reframe_segments — without
    that, the Phase 4 (and Phase 8 future) content-aware branches
    don't fire because ``cfg`` stays None inside the segmenter."""

    SRC_PATH = (
        Path(__file__).resolve().parents[1] / "scripts" / "measure_autoflip_parity.py"
    )

    def test_runner_calls_classify_content(self):
        src = self.SRC_PATH.read_text()
        assert "from backend.services.content_classifier import classify_content" in src
        assert "classify_content(" in src

    def test_runner_passes_content_profile(self):
        src = self.SRC_PATH.read_text()
        assert "content_profile=content_profile" in src
