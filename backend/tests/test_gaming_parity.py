"""Tests for gaming regression guard (E5).

Verifies that all changes in Phases A-D do NOT alter the RenderPlan
output for any gameplay content type. The _is_gaming_mode gate must
ensure every new code path falls through to legacy behavior for:
  - GAMEPLAY
  - GAMEPLAY_MOBA
  - GAMEPLAY_TPS
  - GAMEPLAY_RACING

Strategy: instead of byte-identical baseline comparison (which requires
pre-captured JSON fixtures), we validate the invariants directly:
  - _is_gaming_mode returns True for all gameplay types
  - No face_containment regions are emitted for gaming
  - Saliency fusion weights are legacy (0.3/0.5/0.2) for gaming
  - Camera solver safety margin is 0.95 for gaming
  - Cross-shot handoff is skipped for gaming
  - Saliency hysteresis is skipped for gaming
  - Saliency min score is 0.25 for gaming
"""
import pytest
import numpy as np


GAMEPLAY_TYPES = ["gameplay", "gameplay_moba", "gameplay_tps", "gameplay_racing"]


class TestGamingParity:
    """E5: Gaming regression guard — no behavior change for gameplay."""

    def test_is_gaming_mode_true_for_all_gameplay(self):
        """_is_gaming_mode returns True for all gameplay content types."""
        from backend.services.required_regions import _is_gaming_mode

        for ct in GAMEPLAY_TYPES:
            assert _is_gaming_mode(ct) is True, f"Expected True for {ct}"

    def test_is_gaming_mode_false_for_non_gameplay(self):
        """_is_gaming_mode returns False for non-gameplay content types."""
        from backend.services.required_regions import _is_gaming_mode

        non_gaming = [
            "talking_head", "cinematic_dialogue", "multi_speaker_panel",
            "animation", "animation_dialogue", "music_video", "stream",
            "sports", "sports_basketball", "sports_racing", "generic",
            None,
        ]
        for ct in non_gaming:
            assert _is_gaming_mode(ct) is False, f"Expected False for {ct}"

    def test_no_containment_regions_for_gaming(self):
        """Gaming content must NOT emit face_containment regions."""
        from backend.services.required_regions import build_required_regions
        from dataclasses import dataclass

        @dataclass
        class FakeFace:
            nose_x: float = 50.0
            nose_y: float = 50.0
            width: float = 16.0
            height: float = 20.0
            is_human: bool = True
            identity_id: int = 0
            yaw: float = 0.0
            lip_aperture: float = 0.0

        @dataclass
        class FakeFF:
            timestamp: float = 1.0
            faces: list = None
            def __post_init__(self):
                if self.faces is None:
                    self.faces = [FakeFace()]

        @dataclass
        class FakeEvent:
            start: float = 0.0
            end: float = 5.0
            slot_id: int = 0
            on_screen: bool = True

        for ct in GAMEPLAY_TYPES:
            regions = build_required_regions(
                [FakeFF()],
                active_speaker_events=[FakeEvent()],
                content_type=ct,
            )
            for frame_regs in regions:
                containment = [r for r in frame_regs if r.source == "face_containment"]
                assert len(containment) == 0, (
                    f"Gaming type {ct} should not emit face_containment, got {len(containment)}"
                )

    def test_fusion_weights_legacy_for_gaming(self):
        """Gaming content types should get legacy saliency weights."""
        from backend.services.saliency_tracker import _get_fusion_weights

        for ct in GAMEPLAY_TYPES:
            s, t, c = _get_fusion_weights(ct)
            assert (s, t, c) == (0.3, 0.5, 0.2), (
                f"Gaming type {ct}: expected (0.3, 0.5, 0.2), got ({s}, {t}, {c})"
            )

    def test_safety_margin_unchanged_for_gaming(self):
        """Camera solver safety margin should be 0.95 for all gameplay types."""
        from backend.services.camera_solver import _stationary_zoom_safety

        for ct in GAMEPLAY_TYPES:
            safety = _stationary_zoom_safety(ct)
            assert safety == 0.95, (
                f"Gaming type {ct}: expected safety=0.95, got {safety}"
            )

    def test_handoff_skipped_for_gaming(self):
        """apply_cross_shot_handoff should be a no-op for gaming content."""
        from backend.services.camera_solver import apply_cross_shot_handoff
        from dataclasses import dataclass, field

        @dataclass
        class FakeShotCam:
            shot_index: int = 0
            start: float = 0.0
            end: float = 3.0
            mode: str = "tracking"
            keyframes: list = field(default_factory=list)
            reason: str = ""
            zoom: float = 1.0

        for ct in GAMEPLAY_TYPES:
            shot0 = FakeShotCam(
                shot_index=0, start=0.0, end=3.0,
                keyframes=[(0.0, 0.2, 0.5), (3.0, 0.2, 0.5)],
            )
            shot1 = FakeShotCam(
                shot_index=1, start=3.2, end=6.0,
                keyframes=[(3.2, 0.8, 0.5), (6.0, 0.8, 0.5)],
            )
            orig_kf0 = list(shot0.keyframes)
            result = apply_cross_shot_handoff([shot0, shot1], content_type=ct)
            assert result[0].keyframes == orig_kf0, (
                f"Gaming type {ct}: handoff should not modify keyframes"
            )

    def test_saliency_no_hysteresis_for_gaming(self):
        """Gaming content should bypass saliency temporal hysteresis."""
        from backend.services.saliency_tracker import compute_spatiotemporal_saliency

        h, w = 120, 213
        curr = np.random.randint(0, 255, (h, w), dtype=np.uint8)
        prev_combined = np.ones((h, w), dtype=np.float32) * 0.5

        for ct in GAMEPLAY_TYPES:
            with_prev = compute_spatiotemporal_saliency(
                curr, None, prev_combined=prev_combined, content_type=ct,
            )
            without_prev = compute_spatiotemporal_saliency(
                curr, None, content_type=ct,
            )
            np.testing.assert_array_almost_equal(
                with_prev, without_prev, decimal=5,
                err_msg=f"Gaming type {ct}: hysteresis should be bypassed",
            )

    def test_saliency_min_score_legacy_for_gaming(self):
        """Saliency min score should be 0.25 for gameplay (not 0.35)."""
        from backend.services.attention_anchor import _saliency_min_score_for

        for ct in GAMEPLAY_TYPES:
            score = _saliency_min_score_for(ct)
            assert score == 0.25, (
                f"Gaming type {ct}: expected min_score=0.25, got {score}"
            )

    def test_solver_params_unchanged_for_gameplay(self):
        """The gameplay SolverParams must not be altered."""
        from backend.services.camera_solver import get_params_for_content_type
        from backend.services.content_classifier import ClipContentType

        params = get_params_for_content_type(ClipContentType.GAMEPLAY)
        assert params.smoothing_alpha == 0.6
        assert params.stationary_slack == 0.1
        assert params.prefer_stationary is True
        assert params.shot_threshold == 40.0
