"""Tests for content_type_config including intent tracking tuning."""

import pytest

from backend.services.content_type_config import (
    ContentType,
    CONTENT_TYPE_CONFIG,
    TuningConfig,
    get_config,
    get_tuning_from_profile,
)


class TestIntentTuningFields:
    """Every content type must have intent tracking fields accessible via TuningConfig."""

    INTENT_FIELDS = [
        "intent_ema_alpha",
        "intent_switch_margin",
        "intent_min_switch_confidence",
        "intent_min_hold_fallback",
    ]

    @pytest.mark.parametrize("ct", list(ContentType))
    def test_tuning_has_intent_fields(self, ct):
        cfg = CONTENT_TYPE_CONFIG[ct]
        tuning = TuningConfig(cfg)
        for field in self.INTENT_FIELDS:
            val = getattr(tuning, field)
            assert isinstance(val, (int, float)), f"{ct.value}.{field} should be numeric, got {type(val)}"
            assert val > 0, f"{ct.value}.{field} should be positive"

    @pytest.mark.parametrize("ct", list(ContentType))
    def test_switch_margin_within_bounds(self, ct):
        """switch_margin must not exceed 0.30 for any content type."""
        tuning = TuningConfig(CONTENT_TYPE_CONFIG[ct])
        assert tuning.intent_switch_margin <= 0.30, \
            f"{ct.value} switch_margin={tuning.intent_switch_margin} exceeds 0.30"

    @pytest.mark.parametrize("ct", list(ContentType))
    def test_ema_alpha_in_range(self, ct):
        tuning = TuningConfig(CONTENT_TYPE_CONFIG[ct])
        assert 0.0 < tuning.intent_ema_alpha <= 1.0, \
            f"{ct.value} ema_alpha={tuning.intent_ema_alpha} out of (0, 1]"


class TestContentTypeSpecificOverrides:
    def test_podcast_slower_than_default(self):
        tuning = TuningConfig(CONTENT_TYPE_CONFIG[ContentType.PODCAST])
        assert tuning.intent_ema_alpha < 0.4  # slower
        assert tuning.intent_switch_margin > 0.15  # more resistant

    def test_anime_snappier_than_default(self):
        tuning = TuningConfig(CONTENT_TYPE_CONFIG[ContentType.ANIME])
        assert tuning.intent_ema_alpha > 0.4  # snappier
        assert tuning.intent_switch_margin < 0.15  # less resistant

    def test_gaming_most_resistant(self):
        tuning = TuningConfig(CONTENT_TYPE_CONFIG[ContentType.GAMING])
        assert tuning.intent_switch_margin == 0.30  # highest margin

    def test_sports_snappy(self):
        tuning = TuningConfig(CONTENT_TYPE_CONFIG[ContentType.SPORTS])
        assert tuning.intent_ema_alpha >= 0.50
        assert tuning.intent_switch_margin <= 0.12


class TestGetTuningFromProfile:
    def test_none_profile_returns_unknown_defaults(self):
        tuning = get_tuning_from_profile(None)
        assert tuning.intent_ema_alpha == 0.4  # unknown default

    def test_string_content_type(self):
        tuning = get_tuning_from_profile("podcast")
        assert tuning.intent_ema_alpha == 0.25

    def test_profile_object(self):
        class FakeProfile:
            content_type = "anime"
        tuning = get_tuning_from_profile(FakeProfile())
        assert tuning.intent_ema_alpha == 0.55

    def test_invalid_type_falls_back_to_unknown(self):
        tuning = get_tuning_from_profile("nonexistent_type")
        assert tuning.intent_ema_alpha == 0.4


class TestTuningConfigAccess:
    def test_non_intent_fields_accessible(self):
        tuning = TuningConfig(CONTENT_TYPE_CONFIG[ContentType.NARRATIVE])
        assert tuning.apply_lead_room is True

    def test_missing_field_raises(self):
        tuning = TuningConfig({})
        with pytest.raises(AttributeError):
            _ = tuning.nonexistent_field
