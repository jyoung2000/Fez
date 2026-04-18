"""Phase 5 — GamingReframeConfig dataclass + env-driven overrides."""

from __future__ import annotations

import importlib

import pytest

from backend.services import gaming_reframe_config as cfg_mod


def test_defaults_match_spec():
    cfg = cfg_mod.GamingReframeConfig()
    assert cfg.enabled is False
    assert cfg.center_anchor_weight == 0.4
    assert cfg.gaming_action_scale == 1.0
    assert cfg.hud_glance_peak_weight == 1.2
    assert cfg.hud_glance_decay_tau_s == 1.0
    assert cfg.hud_glance_hold_s == 0.3
    assert cfg.pixel_delta_threshold == 8.0
    assert cfg.scheduled_glance_interval_s == 10.0
    assert cfg.lambda_center == 0.4
    assert cfg.max_glance_deviation_pct == 0.65
    assert cfg.motion_stabilization_threshold_percent == 0.65
    assert cfg.hud_detection_score_threshold == 0.55
    assert cfg.hud_detection_sample_count == 24


def test_validate_catches_bad_values():
    cfg = cfg_mod.GamingReframeConfig(
        max_glance_deviation_pct=1.3,
        hud_min_area_pct=0.2,
        hud_max_area_pct=0.1,
        hud_glance_decay_tau_s=0.0,
        hud_detection_sample_count=2,
        center_anchor_weight=-0.1,
    )
    warnings = cfg.validate()
    joined = " | ".join(warnings)
    assert "max_glance_deviation_pct" in joined
    assert "hud_min_area_pct" in joined
    assert "hud_glance_decay_tau_s" in joined
    assert "hud_detection_sample_count" in joined
    assert "center_anchor_weight" in joined


def test_from_env_reads_tunables(monkeypatch):
    monkeypatch.setenv("GAMING_CENTER_ANCHOR_WEIGHT", "0.7")
    monkeypatch.setenv("GAMING_LAMBDA_CENTER", "0.9")
    monkeypatch.setenv("GAMING_HUD_DETECTION_SAMPLE_COUNT", "48")
    cfg = cfg_mod.GamingReframeConfig.from_env()
    assert cfg.center_anchor_weight == 0.7
    assert cfg.lambda_center == 0.9
    assert cfg.hud_detection_sample_count == 48


def test_from_env_ignores_garbage(monkeypatch):
    monkeypatch.setenv("GAMING_LAMBDA_CENTER", "not-a-float")
    monkeypatch.setenv("GAMING_HUD_DETECTION_SAMPLE_COUNT", "NaN_int")
    cfg = cfg_mod.GamingReframeConfig.from_env()
    # Both fall back to defaults.
    assert cfg.lambda_center == 0.4
    assert cfg.hud_detection_sample_count == 24


def test_load_requires_autoflip_mode(caplog, monkeypatch):
    monkeypatch.setenv("USE_GAMING_REFRAME", "1")
    # Reload module so USE_GAMING_REFRAME re-reads env.
    importlib.reload(cfg_mod)
    with caplog.at_level("WARNING"):
        cfg = cfg_mod.load_gaming_reframe_config(autoflip_enabled=False)
    assert cfg.enabled is False
    assert any(
        "requires the AutoFlip L1 solver" in rec.message
        for rec in caplog.records
    )


def test_load_passes_when_both_flags_on(monkeypatch):
    monkeypatch.setenv("USE_GAMING_REFRAME", "1")
    importlib.reload(cfg_mod)
    cfg = cfg_mod.load_gaming_reframe_config(autoflip_enabled=True)
    assert cfg.enabled is True
