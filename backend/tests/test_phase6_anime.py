"""Phase 6 — Animation-aware pipeline tests.

The "right moment" pipeline for anime / cartoons spans four
modules:

  - ``backend.services.anime_shot_detector`` — histogram +
    edge-density shot detector tuned for flat-color anime that
    PySceneDetect's ContentDetector misses.

  - ``backend.services.anime_face_detector`` — lbpcascade-backed
    anime face detector with graceful fallback when OpenCV /
    the cascade XML isn't available.

  - ``backend.services.anime_anchor`` — per-frame anime saliency
    scorer that combines face, motion, contrast, and saturation
    signals into a single anchor decision (the "right moment"
    picker).

  - Reframe segmenter Stage 7b — applies the anime anchor
    override BEFORE the lead-room block so the L1 solver sees
    the saliency-driven subject_x.

Plus three Phase 4 / Phase 6 cross-cuts:

  - ``gaze_estimator.lead_room_offset_px`` gained an
    ``anime_action_multiplier`` kwarg (1.5 for action anime).

  - ``thirds_bias.applies_to_profile`` gained anime profile
    handling (animated dialogue / slice-of-life get the bias,
    action explicitly opts out).

  - The reframe segmenter passes the multiplier through both
    Stage 8 and Stage 10c when the profile is anime + action.

Per the v2 ground rules, all three Phase 6 feature flags
(``CLIPAI_ANIME_SHOT_DETECTOR``, ``CLIPAI_ANIME_FACE_DETECTOR``,
``CLIPAI_ANIME_ANCHOR``) default OFF until in-docker validation
lands the post-Phase-6 numbers.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import pytest


# ──────────────────── Anime shot detector ────────────────────


class TestAnimeShotDetectorHelpers:
    """Pure-Python helpers from anime_shot_detector. The OpenCV-
    backed entry points are exercised separately when the imaging
    stack is available."""

    def test_histogram_correlation_identical(self):
        from backend.services.anime_shot_detector import histogram_correlation
        assert histogram_correlation([0.5, 0.5], [0.5, 0.5]) == 1.0

    def test_histogram_correlation_orthogonal(self):
        from backend.services.anime_shot_detector import histogram_correlation
        assert histogram_correlation([1.0, 0.0], [0.0, 1.0]) == -1.0

    def test_histogram_correlation_mismatched_length(self):
        from backend.services.anime_shot_detector import histogram_correlation
        assert histogram_correlation([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0

    def test_histogram_correlation_empty(self):
        from backend.services.anime_shot_detector import histogram_correlation
        assert histogram_correlation([], []) == 1.0

    def test_score_pair_metrics_basic(self):
        from backend.services.anime_shot_detector import score_pair_metrics
        hist_pairs = [
            ([0.5, 0.5], [0.5, 0.5]),
            ([0.5, 0.5], [0.0, 1.0]),
        ]
        edges = [0.1, 0.1, 0.5]
        ts = [0.0, 0.5, 1.0]
        metrics = score_pair_metrics(hist_pairs, edges, ts)
        assert len(metrics) == 2
        # First pair has identical histograms → correlation 1.0
        assert metrics[0].hist_corr == pytest.approx(1.0, abs=1e-9)
        # Second pair has different histograms → correlation 0
        assert metrics[1].hist_corr == pytest.approx(0.0, abs=1e-9)
        # Edge delta is 0 on the first pair (matches mean) and
        # large on the second (0.5 vs rolling mean 0.1)
        assert metrics[0].edge_delta == 0.0
        assert metrics[1].edge_delta > 1.0

    def test_score_pair_metrics_validation(self):
        from backend.services.anime_shot_detector import score_pair_metrics
        with pytest.raises(ValueError):
            score_pair_metrics([([0], [0])], [0.1], [0.0])

    def test_cut_indices_filter_below_threshold(self):
        from backend.services.anime_shot_detector import (
            PairMetrics, cut_indices_from_metrics,
        )
        metrics = [
            PairMetrics(index=1, timestamp=0.5, hist_corr=0.9,
                        edge_density=0.1, edge_delta=0.05),
            PairMetrics(index=2, timestamp=1.0, hist_corr=0.3,  # cut!
                        edge_density=0.2, edge_delta=0.1),
            PairMetrics(index=3, timestamp=1.5, hist_corr=0.85,
                        edge_density=0.2, edge_delta=0.05),
        ]
        cuts = cut_indices_from_metrics(metrics)
        assert cuts == [2]

    def test_cut_indices_min_gap_coalesces(self):
        # Two cuts within min_cut_gap_sec → only the first survives
        from backend.services.anime_shot_detector import (
            PairMetrics, cut_indices_from_metrics,
        )
        metrics = [
            PairMetrics(index=1, timestamp=0.5, hist_corr=0.2,
                        edge_density=0.1, edge_delta=0.0),
            PairMetrics(index=2, timestamp=0.55, hist_corr=0.2,
                        edge_density=0.1, edge_delta=0.0),
            PairMetrics(index=3, timestamp=2.0, hist_corr=0.2,
                        edge_density=0.1, edge_delta=0.0),
        ]
        cuts = cut_indices_from_metrics(metrics, min_cut_gap_sec=0.30)
        assert cuts == [1, 3]

    def test_cut_indices_edge_delta_only(self):
        # High edge delta alone (no histogram drop) should still
        # flag a cut — anime scenes with same palette but different
        # composition.
        from backend.services.anime_shot_detector import (
            PairMetrics, cut_indices_from_metrics,
        )
        metrics = [
            PairMetrics(index=1, timestamp=1.0, hist_corr=0.85,
                        edge_density=0.5, edge_delta=0.45),
        ]
        assert cut_indices_from_metrics(metrics) == [1]

    def test_anime_shot_detector_skipped_when_video_missing(self, tmp_path):
        from backend.services.anime_shot_detector import detect_anime_shots
        result = detect_anime_shots(str(tmp_path / "missing.mp4"))
        assert not result.has_data or result.skipped_reason
        # Either the cv2 import failed or the cv2 open failed
        assert result.cut_times == []

    def test_anime_shot_detector_flag_default_off(self):
        from backend.services.anime_shot_detector import USE_ANIME_SHOT_DETECTOR
        assert USE_ANIME_SHOT_DETECTOR is False


# ──────────────────── Anime face detector ────────────────────


class TestAnimeFaceDetectorHelpers:
    def test_score_density_empty(self):
        from backend.services.anime_face_detector import score_anime_face_density
        assert score_anime_face_density([]) == 0.0

    def test_score_density_single_face(self):
        from backend.services.anime_face_detector import (
            AnimeFaceDetection, score_anime_face_density,
        )
        single = [AnimeFaceDetection(50, 50, 20, 25, 0.9)]
        score = score_anime_face_density(single)
        assert 0 < score <= 1.0

    def test_score_density_more_faces_scores_higher(self):
        from backend.services.anime_face_detector import (
            AnimeFaceDetection, score_anime_face_density,
        )
        single = [AnimeFaceDetection(50, 50, 20, 25, 0.9)]
        triple = [
            AnimeFaceDetection(30, 50, 15, 20, 0.85),
            AnimeFaceDetection(50, 50, 15, 20, 0.85),
            AnimeFaceDetection(70, 50, 15, 20, 0.85),
        ]
        assert score_anime_face_density(triple) > score_anime_face_density(single)

    def test_best_face_picks_highest_confidence_area(self):
        from backend.services.anime_face_detector import (
            AnimeFaceDetection, best_face_in_frame,
        )
        # First face has higher conf but smaller area; second has
        # lower conf but much bigger area → second wins on
        # confidence × area.
        faces = [
            AnimeFaceDetection(20, 50, 10, 10, 0.9),   # conf*area = 90
            AnimeFaceDetection(70, 50, 30, 30, 0.7),   # conf*area = 630
        ]
        best = best_face_in_frame(faces)
        assert best.x_center == 70

    def test_best_face_empty_returns_none(self):
        from backend.services.anime_face_detector import best_face_in_frame
        assert best_face_in_frame([]) is None

    def test_to_face_info_marks_non_human(self):
        from backend.services.anime_face_detector import (
            AnimeFaceDetection, to_face_info,
        )
        d = AnimeFaceDetection(40, 50, 15, 20, 0.9)
        # Lazy import will pull in face_detector — accept either the
        # FaceInfo conversion or a clean AttributeError if
        # face_detector itself can't import (sandbox without cv2).
        try:
            fi = to_face_info(d, identity_id=2)
        except (ImportError, ModuleNotFoundError):
            pytest.skip("face_detector deps missing in this environment")
        assert fi.is_human is False
        assert fi.identity_id == 2
        assert fi.x_center == 40.0
        # nose_x defaults to bbox center for cascade detections
        assert fi.nose_x == 40.0

    def test_anime_face_detector_skipped_when_frame_missing(self, tmp_path):
        from backend.services.anime_face_detector import detect_anime_faces
        result = detect_anime_faces(str(tmp_path / "missing.png"))
        assert result.has_faces is False
        assert result.skipped_reason  # non-empty

    def test_anime_face_detector_flag_default_off(self):
        from backend.services.anime_face_detector import USE_ANIME_FACE_DETECTOR
        assert USE_ANIME_FACE_DETECTOR is False


# ──────────────────── Anime anchor scorer ────────────────────


class TestAnimeAnchorScoring:
    def test_face_dominant_anchor(self):
        from backend.services.anime_anchor import (
            AnimeFrameFeatures, score_anime_frame,
        )
        feats = AnimeFrameFeatures(
            timestamp=1.0,
            face_x_pct=30, face_y_pct=40, face_score=0.9,
            motion_energy=0.1, contrast=0.1, saturation=0.1,
            motion_x_pct=70,
        )
        anchor = score_anime_frame(feats)
        assert anchor.source == "face"
        assert anchor.x_pct == 30.0
        assert anchor.y_pct == 40.0
        assert anchor.score > 0.4

    def test_motion_dominant_anchor_for_action(self):
        # Action sub-type weights motion higher; with face_score=0
        # and motion=0.95 the motion centroid should win.
        from backend.services.anime_anchor import (
            AnimeFrameFeatures, score_anime_frame,
        )
        feats = AnimeFrameFeatures(
            timestamp=2.0,
            face_x_pct=50, face_score=0.0,
            motion_energy=0.95, motion_x_pct=70,
            contrast=0.3, saturation=0.2,
        )
        anchor = score_anime_frame(feats, anime_subtype="action")
        assert anchor.source == "motion"
        assert anchor.x_pct == 70.0

    def test_dialogue_subtype_weights_face_higher(self):
        # Same features, dialogue subtype: face still dominates
        # (it's the dialogue weight) with a stronger face score.
        from backend.services.anime_anchor import (
            AnimeFrameFeatures, score_anime_frame,
        )
        feats = AnimeFrameFeatures(
            timestamp=3.0,
            face_x_pct=30, face_score=0.6,
            motion_energy=0.4, motion_x_pct=70,
        )
        action = score_anime_frame(feats, anime_subtype="action")
        dialogue = score_anime_frame(feats, anime_subtype="dialogue")
        assert dialogue.score > action.score
        assert dialogue.source == "face"

    def test_all_zero_falls_back_to_center(self):
        from backend.services.anime_anchor import (
            AnimeFrameFeatures, score_anime_frame,
        )
        anchor = score_anime_frame(AnimeFrameFeatures(timestamp=1.0))
        assert anchor.source == "fallback"
        assert anchor.x_pct == 50.0
        assert anchor.score == 0.0

    def test_aggregate_picks_best_in_window(self):
        from backend.services.anime_anchor import (
            AnimeFrameFeatures, score_anime_sequence,
            best_anchor_in_window,
        )
        seq = [
            AnimeFrameFeatures(timestamp=0.5, face_score=0.2, face_x_pct=40),
            AnimeFrameFeatures(timestamp=1.0, face_score=0.9, face_x_pct=70),
            AnimeFrameFeatures(timestamp=1.5, face_score=0.1, face_x_pct=20),
        ]
        anchors = score_anime_sequence(seq)
        best = best_anchor_in_window(anchors, start=0.0, end=2.0)
        assert best.x_pct == 70.0

    def test_aggregate_below_min_score_falls_back(self):
        from backend.services.anime_anchor import (
            AnimeFrameFeatures, score_anime_sequence,
            aggregate_anchors_to_segment_x,
        )
        # Single weak frame → below the 0.30 default floor
        seq = [AnimeFrameFeatures(timestamp=1.0, face_score=0.1, face_x_pct=70)]
        anchors = score_anime_sequence(seq)
        x, src = aggregate_anchors_to_segment_x(
            anchors, start=0.0, end=2.0, min_score=0.30,
        )
        assert src == "fallback"
        assert x == 50.0

    def test_aggregate_above_floor_overrides(self):
        from backend.services.anime_anchor import (
            AnimeFrameFeatures, score_anime_sequence,
            aggregate_anchors_to_segment_x,
        )
        seq = [AnimeFrameFeatures(timestamp=1.0, face_score=0.9, face_x_pct=70)]
        anchors = score_anime_sequence(seq)
        x, src = aggregate_anchors_to_segment_x(
            anchors, start=0.0, end=2.0, min_score=0.30,
        )
        assert src == "face"
        assert x == 70.0

    def test_motion_energy_helper(self):
        from backend.services.anime_anchor import (
            motion_energy_from_intensity_means,
        )
        energies = motion_energy_from_intensity_means([100, 100, 100, 130, 100])
        assert energies[0] == 0.0
        assert energies[3] == 1.0  # 30 / 30 norm → capped at 1.0

    def test_normalize_helpers(self):
        from backend.services.anime_anchor import (
            normalize_intensity_stdev, normalize_saturation_mean,
        )
        assert normalize_intensity_stdev(0) == 0.0
        assert normalize_intensity_stdev(64) == 1.0
        assert normalize_intensity_stdev(128) == 1.0  # clamped
        assert normalize_saturation_mean(0) == 0.0
        assert normalize_saturation_mean(255) == 1.0

    def test_anime_anchor_flag_default_off(self):
        from backend.services.anime_anchor import USE_ANIME_ANCHOR
        assert USE_ANIME_ANCHOR is False


# ──────────────────── Phase 4 → Phase 6 lead-room multiplier ────────────────────


class TestAnimeActionLeadRoomMultiplier:
    def test_default_multiplier_unchanged(self):
        from backend.services.gaze_estimator import lead_room_offset_px
        assert lead_room_offset_px(-1.0, 600.0) == pytest.approx(48.0, abs=1e-9)

    def test_action_multiplier_15x(self):
        from backend.services.gaze_estimator import lead_room_offset_px
        # 1.5x multiplier — full yaw → 48 * 1.5 = 72
        assert lead_room_offset_px(
            -1.0, 600.0, anime_action_multiplier=1.5,
        ) == pytest.approx(72.0, abs=1e-9)

    def test_zero_multiplier(self):
        from backend.services.gaze_estimator import lead_room_offset_px
        assert lead_room_offset_px(
            -1.0, 600.0, anime_action_multiplier=0.0,
        ) == 0.0

    def test_negative_multiplier_clamped_to_zero(self):
        from backend.services.gaze_estimator import lead_room_offset_px
        # Negative multipliers don't make sense — we treat them
        # as zero rather than flipping the sign.
        assert lead_room_offset_px(
            -1.0, 600.0, anime_action_multiplier=-1.0,
        ) == 0.0

    def test_zero_yaw_unaffected_by_multiplier(self):
        from backend.services.gaze_estimator import lead_room_offset_px
        assert lead_room_offset_px(
            0.0, 600.0, anime_action_multiplier=10.0,
        ) == 0.0


# ──────────────────── applies_to_profile anime extension ────────────────────


class TestAppliesToProfileAnime:
    def _profile(self, **kwargs):
        @dataclass
        class _P:
            content_type: str = "unknown"
            is_multi_speaker_panel: bool = False
            is_animated: bool = False
            anime_subtype: str = ""
        return _P(**kwargs)

    def test_anime_dialogue_passes(self):
        from backend.services.thirds_bias import applies_to_profile
        p = self._profile(content_type="anime", is_animated=True, anime_subtype="dialogue")
        assert applies_to_profile(p) is True

    def test_anime_slice_of_life_passes(self):
        from backend.services.thirds_bias import applies_to_profile
        p = self._profile(content_type="anime", is_animated=True, anime_subtype="slice_of_life")
        assert applies_to_profile(p) is True

    def test_anime_action_excluded(self):
        # Action anime opts out — choreographed compositions don't
        # benefit from thirds bias.
        from backend.services.thirds_bias import applies_to_profile
        p = self._profile(content_type="anime", is_animated=True, anime_subtype="action")
        assert applies_to_profile(p) is False

    def test_anime_no_subtype_passes(self):
        # is_animated=True with no subtype → defaults to dialogue
        # routing, gets the bias.
        from backend.services.thirds_bias import applies_to_profile
        p = self._profile(content_type="anime", is_animated=True)
        assert applies_to_profile(p) is True

    def test_non_anime_unaffected(self):
        from backend.services.thirds_bias import applies_to_profile
        # vlog still passes
        p = self._profile(content_type="vlog")
        assert applies_to_profile(p) is True
        # debate (panel flag) still excluded
        p = self._profile(content_type="podcast", is_multi_speaker_panel=True)
        assert applies_to_profile(p) is False


# ──────────────────── reframe_segmenter Stage 7b AST guards ────────────────────


class TestReframeSegmenterStage7bAST:
    """Static checks on reframe_segmenter.py Stage 7b wiring +
    Stage 8 / 10c anime multiplier propagation."""

    SRC_PATH = (
        Path(__file__).resolve().parents[1] / "services" / "reframe_segmenter.py"
    )

    def test_stage_7b_block_present(self):
        src = self.SRC_PATH.read_text()
        assert "Stage 7b" in src
        assert "AnimeAnchor" in src or "anime_anchor" in src

    def test_imports_anime_anchor_lazily(self):
        src = self.SRC_PATH.read_text()
        assert "USE_ANIME_ANCHOR" in src
        assert "aggregate_anchors_to_segment_x" in src

    def test_signature_has_anime_anchors_kwarg(self):
        src = self.SRC_PATH.read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "build_reframe_segments":
                kwarg_names = [a.arg for a in node.args.args] + [
                    a.arg for a in node.args.kwonlyargs
                ]
                assert "anime_anchors" in kwarg_names
                return
        pytest.fail("build_reframe_segments not found")

    def test_anime_action_multiplier_in_stage_8(self):
        src = self.SRC_PATH.read_text()
        # Stage 8 reads the subtype and computes the multiplier
        assert "_anime_action_mult" in src
        assert "anime_action_multiplier=" in src

    def test_anime_action_multiplier_in_stage_10c(self):
        # Stage 10c also threads the multiplier through the
        # tracking / panning post-process so anime action lead-room
        # survives the L1 solver overwrite.
        src = self.SRC_PATH.read_text()
        assert "_anime_mult_post" in src

    def test_anime_anchor_count_log(self):
        src = self.SRC_PATH.read_text()
        assert "AnimeAnchor:" in src
        assert "anime_anchor_count" in src


# ──────────────────── Phase 6 feature flag defaults ────────────────────


class TestPhase6FeatureFlagsDefaultOff:
    def test_anime_shot_detector_default_off(self):
        from backend.services.anime_shot_detector import USE_ANIME_SHOT_DETECTOR
        assert USE_ANIME_SHOT_DETECTOR is False

    def test_anime_face_detector_default_off(self):
        from backend.services.anime_face_detector import USE_ANIME_FACE_DETECTOR
        assert USE_ANIME_FACE_DETECTOR is False

    def test_anime_anchor_default_off(self):
        from backend.services.anime_anchor import USE_ANIME_ANCHOR
        assert USE_ANIME_ANCHOR is False
