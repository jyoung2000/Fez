"""Blueprint v2 Phase 3 Task 3.6 — virality down-weight by reframe confidence."""

import pytest

from backend.services.clip_confidence import (
    CLIP_CONFIDENCE_VIRALITY_MULTIPLIER,
    _confidence_multiplier,
    apply_clip_confidence,
)


def test_multiplier_defaults():
    assert CLIP_CONFIDENCE_VIRALITY_MULTIPLIER["high"] == 1.0
    assert CLIP_CONFIDENCE_VIRALITY_MULTIPLIER["medium"] == 0.95
    assert CLIP_CONFIDENCE_VIRALITY_MULTIPLIER["low"] == 0.6


def test_apply_clip_confidence_low_reduces_40_percent():
    # Blueprint: low-confidence clip virality is scaled by 0.6.
    assert apply_clip_confidence(100, "low") == 60


def test_apply_clip_confidence_medium_reduces_slightly():
    assert apply_clip_confidence(100, "medium") == 95


def test_apply_clip_confidence_high_is_unchanged():
    assert apply_clip_confidence(87, "high") == 87


def test_apply_clip_confidence_unknown_is_unchanged():
    assert apply_clip_confidence(87, "unknown") == 87
    assert apply_clip_confidence(87, None) == 87


def test_apply_clip_confidence_clamps_to_range():
    assert apply_clip_confidence(0, "low") == 1  # floor at 1
    assert apply_clip_confidence(200, "high") == 100  # ceiling at 100


def test_env_override_wins_over_defaults(monkeypatch):
    # Env vars let QA sweep multipliers without a code change.
    monkeypatch.setenv("CLIPAI_VIRALITY_CONFIDENCE_LOW", "0.3")
    assert _confidence_multiplier("low") == pytest.approx(0.3)
    assert apply_clip_confidence(100, "low") == 30


def test_env_override_invalid_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("CLIPAI_VIRALITY_CONFIDENCE_LOW", "not-a-number")
    # Falls back to 0.6 default.
    assert _confidence_multiplier("low") == pytest.approx(0.6)


def test_env_override_is_clamped(monkeypatch):
    # Someone setting 2.0 shouldn't push virality above the ceiling.
    monkeypatch.setenv("CLIPAI_VIRALITY_CONFIDENCE_HIGH", "2.0")
    assert _confidence_multiplier("high") == pytest.approx(1.0)
