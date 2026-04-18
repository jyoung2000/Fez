"""Phase 3 acceptance tests — confidence-weighted VLM / face fusion.

The fusion rule (see backend/services/vlm_fusion.py) is:

  face_conf >= 0.80 → face_high_conf (trust face detector)
  0.40 <= face_conf < 0.80 → blended (linear ramp on face_conf)
  face_conf < 0.40, vlm_conf >= 0.30 → vlm_only
  otherwise → fallback_center

These tests lock the rule in so future confidence thresholds can't
silently drift without updating the test table.
"""

import os

import pytest

from backend.models import SceneDescription
from backend.services.vlm_fusion import (
    VLM_FUSION_DEFAULT,
    apply_fusion_to_scene,
    fuse_vlm_and_face,
    fusion_enabled,
    fusion_mode,
)


def _mk_scene() -> SceneDescription:
    return SceneDescription(
        timestamp=1.0, description="x", importance_score=5,
        thumbnail_path="/tmp/t.jpg",
    )


# ── Rule table ─────────────────────────────────────────────────────


def test_fusion_high_face_conf_uses_face():
    """face_conf=0.9 overrides any VLM contribution."""
    r = fuse_vlm_and_face(face_x=30.0, face_conf=0.9, vlm_x=70.0, vlm_conf=0.6)
    assert r.subject_x == 30.0
    assert r.fusion_source == "face_high_conf"
    assert r.subject_confidence == 0.9


def test_fusion_face_at_exact_high_threshold_uses_face():
    """face_conf == 0.80 is inclusive — still face_high_conf."""
    r = fuse_vlm_and_face(face_x=40.0, face_conf=0.80, vlm_x=60.0, vlm_conf=0.9)
    assert r.fusion_source == "face_high_conf"
    assert r.subject_x == 40.0


def test_fusion_uncertain_face_blends():
    """face_conf=0.6, vlm_conf=0.7, face_x=30, vlm_x=70.

    w_face = (0.6-0.4)/0.4 = 0.5
    w_vlm  = 0.7 * (1-0.5) = 0.35
    total  = 0.85
    blend  = (30*0.5 + 70*0.35) / 0.85 = (15 + 24.5) / 0.85 ≈ 46.47
    """
    r = fuse_vlm_and_face(face_x=30.0, face_conf=0.6, vlm_x=70.0, vlm_conf=0.7)
    assert r.fusion_source == "blended"
    assert 46.0 <= r.subject_x <= 47.0
    assert r.subject_confidence == pytest.approx(max(0.6, 0.7))


def test_fusion_blended_face_at_lower_threshold():
    """face_conf == 0.40 is the lower blend threshold; w_face = 0."""
    r = fuse_vlm_and_face(face_x=20.0, face_conf=0.40, vlm_x=80.0, vlm_conf=0.9)
    # w_face = 0, w_vlm = 0.9 * 1.0 = 0.9 → blend = 80.
    assert r.fusion_source == "blended"
    assert r.subject_x == pytest.approx(80.0, abs=0.1)


def test_fusion_blended_zero_vlm_conf_falls_back_to_face():
    """Uncertain face + zero VLM confidence → still use face."""
    r = fuse_vlm_and_face(face_x=25.0, face_conf=0.40, vlm_x=75.0, vlm_conf=0.0)
    # w_face=0, w_vlm=0 → total=0 → degenerate case falls to face.
    assert r.fusion_source == "face_high_conf"
    assert r.subject_x == 25.0


def test_fusion_no_face_vlm_takes_over():
    """face_conf < 0.40 and vlm_conf >= 0.30 → vlm_only."""
    r = fuse_vlm_and_face(face_x=0.0, face_conf=0.0, vlm_x=65.0, vlm_conf=0.7)
    assert r.fusion_source == "vlm_only"
    assert r.subject_x == 65.0
    assert r.subject_confidence == 0.7


def test_fusion_low_face_and_low_vlm_falls_back_center():
    """Nothing reliable → 50, confidence 0, fallback_center."""
    r = fuse_vlm_and_face(face_x=12.0, face_conf=0.1, vlm_x=80.0, vlm_conf=0.2)
    assert r.fusion_source == "fallback_center"
    assert r.subject_x == 50.0
    assert r.subject_confidence == 0.0


def test_fusion_vlm_just_below_minimum():
    """vlm_conf = 0.29 → still fallback_center (threshold is 0.30)."""
    r = fuse_vlm_and_face(face_x=None, face_conf=0.0, vlm_x=70.0, vlm_conf=0.29)
    assert r.fusion_source == "fallback_center"


# ── Input sanitation ───────────────────────────────────────────────


def test_fusion_none_face_x_treated_as_zero():
    r = fuse_vlm_and_face(face_x=None, face_conf=0.9, vlm_x=60.0, vlm_conf=0.6)
    # face_conf forced to 0, so this falls through to vlm_only.
    assert r.fusion_source == "vlm_only"
    assert r.subject_x == 60.0


def test_fusion_none_vlm_x_treated_as_center():
    r = fuse_vlm_and_face(face_x=45.0, face_conf=0.5, vlm_x=None, vlm_conf=0.9)
    # vlm_conf forced to 0 → w_vlm=0 → blend collapses to face_x,
    # but face_conf is still in the blend range (0.40-0.80) so the
    # branch is "blended" with the numerical result equal to face_x.
    assert r.fusion_source == "blended"
    assert r.subject_x == pytest.approx(45.0, abs=0.1)


def test_fusion_confidence_clamped_to_unit_interval():
    r = fuse_vlm_and_face(face_x=30.0, face_conf=5.0, vlm_x=70.0, vlm_conf=-1.0)
    # face_conf clamped to 1.0 → face_high_conf branch.
    assert r.fusion_source == "face_high_conf"
    assert r.subject_x == 30.0
    assert r.subject_confidence == 1.0


# ── apply_fusion_to_scene ──────────────────────────────────────────


def test_apply_fusion_writes_all_scene_fields():
    scene = _mk_scene()
    apply_fusion_to_scene(scene, face_x=30.0, face_conf=0.9, vlm_x=70.0, vlm_conf=0.6)
    assert scene.subject_x == 30
    assert scene.precise_x == pytest.approx(30.0)
    assert scene.subject_confidence == 0.9
    assert scene.vlm_confidence == 0.6
    assert scene.face_confidence == 0.9
    assert scene.fusion_source == "face_high_conf"


def test_apply_fusion_blended_writes_float_precise_x():
    scene = _mk_scene()
    apply_fusion_to_scene(scene, face_x=30.0, face_conf=0.6, vlm_x=70.0, vlm_conf=0.7)
    assert 46 <= scene.subject_x <= 47
    assert 46.0 <= scene.precise_x <= 47.0
    assert scene.fusion_source == "blended"


# ── Env flag default and toggle ────────────────────────────────────


def test_fusion_default_mode_is_hard(monkeypatch):
    """Default env var is 'hard' → legacy behavior, NOT enabled."""
    monkeypatch.delenv("CLIPAI_VLM_FUSION", raising=False)
    assert fusion_mode() == "hard"
    assert fusion_enabled() is False
    assert VLM_FUSION_DEFAULT == "hard"


def test_fusion_weighted_mode_enables(monkeypatch):
    monkeypatch.setenv("CLIPAI_VLM_FUSION", "weighted")
    assert fusion_mode() == "weighted"
    assert fusion_enabled() is True


def test_fusion_mode_case_insensitive(monkeypatch):
    monkeypatch.setenv("CLIPAI_VLM_FUSION", "WEIGHTED")
    assert fusion_enabled() is True


def test_fusion_disabled_preserves_legacy_behavior(monkeypatch):
    """Running without the env flag is a no-op at the module level:
    the existing pipeline override path at pipeline.py:2842 is
    unchanged when CLIPAI_VLM_FUSION is unset."""
    monkeypatch.delenv("CLIPAI_VLM_FUSION", raising=False)
    assert not fusion_enabled()
    # Module is importable without touching pipeline state.
    import backend.services.vlm_fusion as vf  # noqa: F401
