"""Phase 2 acceptance tests — content-type-aware vision model routing.

Exercises ``select_vision_model_for_content`` (the routing helper)
and ``OpenRouterProvider.apply_vision_model_override`` (which mutates
the provider in place so the next analyze_frames call lands on the
right model).

The override routes ANIME and GAMEPLAY content to Qwen3-VL regardless
of preset; everything else falls through to the preset default.
"""

import logging
from unittest.mock import patch

from backend.services.providers.openrouter_provider import (
    PRESETS,
    select_vision_model_for_content,
)

QWEN3_VL = "qwen/qwen3-vl-235b-a22b-instruct"


def test_presets_all_four_tiers_present():
    """Sanity: free / efficient / balanced / premium are all defined."""
    for tier in ("free", "efficient", "balanced", "premium"):
        assert tier in PRESETS
        assert "vision" in PRESETS[tier]
        assert "vision_fallbacks" in PRESETS[tier]


def test_presets_balanced_default_is_gemini3_flash():
    """Balanced preset default vision is Gemini 3 Flash preview."""
    assert PRESETS["balanced"]["vision"] == "google/gemini-3-flash-preview"
    # Fallback chain must include Gemini 2.5 as a safety net.
    assert any("gemini-2.5" in m for m in PRESETS["balanced"]["vision_fallbacks"])


def test_presets_premium_default_is_gemini3_pro():
    assert PRESETS["premium"]["vision"] == "google/gemini-3-pro-preview"
    assert "google/gemini-2.5-pro" in PRESETS["premium"]["vision_fallbacks"]


def test_presets_efficient_default_is_qwen3_vl():
    """Efficient tier's direct grounding-native default is Qwen3-VL."""
    assert PRESETS["efficient"]["vision"] == QWEN3_VL


def test_presets_free_fallbacks_include_qwen3_vl_free():
    """Free tier has Qwen3-VL :free in the fallback chain."""
    assert any(
        "qwen3-vl" in m and ":free" in m
        for m in PRESETS["free"]["vision_fallbacks"]
    )


def test_routing_anime_picks_qwen3vl_in_premium():
    model = select_vision_model_for_content("anime", PRESETS["premium"]["vision"])
    assert model == QWEN3_VL


def test_routing_animation_picks_qwen3vl_in_balanced():
    model = select_vision_model_for_content("animation", PRESETS["balanced"]["vision"])
    assert model == QWEN3_VL


def test_routing_cartoon_picks_qwen3vl():
    model = select_vision_model_for_content("cartoon", PRESETS["premium"]["vision"])
    assert model == QWEN3_VL


def test_routing_gameplay_picks_qwen3vl_in_balanced():
    model = select_vision_model_for_content("gameplay", PRESETS["balanced"]["vision"])
    assert model == QWEN3_VL


def test_routing_gameplay_fps_picks_qwen3vl():
    model = select_vision_model_for_content("gameplay_fps", PRESETS["premium"]["vision"])
    assert model == QWEN3_VL


def test_routing_talking_head_uses_preset_default():
    model = select_vision_model_for_content("talking_head", PRESETS["premium"]["vision"])
    assert model == PRESETS["premium"]["vision"]
    assert model != QWEN3_VL


def test_routing_music_video_uses_preset_default():
    # Live-action; Gemini wins here so override should not fire.
    model = select_vision_model_for_content("music_video", PRESETS["premium"]["vision"])
    assert model == PRESETS["premium"]["vision"]


def test_routing_narrative_uses_preset_default():
    model = select_vision_model_for_content("cinematic_dialogue", PRESETS["balanced"]["vision"])
    assert model == PRESETS["balanced"]["vision"]


def test_routing_none_uses_preset_default():
    """No content type → preset default; never guess."""
    model = select_vision_model_for_content(None, PRESETS["premium"]["vision"])
    assert model == PRESETS["premium"]["vision"]


def test_routing_unknown_content_type_uses_preset_default():
    model = select_vision_model_for_content("random_nonsense", PRESETS["balanced"]["vision"])
    assert model == PRESETS["balanced"]["vision"]


def test_routing_enum_like_object():
    """Objects with .value attribute (ClipContentType enum) are accepted."""
    class FakeEnum:
        value = "anime"
    model = select_vision_model_for_content(FakeEnum(), PRESETS["premium"]["vision"])
    assert model == QWEN3_VL


def test_routing_case_insensitive():
    model = select_vision_model_for_content("ANIME", PRESETS["premium"]["vision"])
    assert model == QWEN3_VL


def test_apply_vision_model_override_swaps_and_restores():
    """The provider-level override mutates _vision_model and adds the
    original preset default to the fallback chain so a routed model
    that 401s falls through to the original."""
    # Patch out external network / config loading so we can instantiate
    # the provider in isolation.
    with patch.object(
        __import__("backend.services.providers.openrouter_provider", fromlist=["_load_model_capabilities"]),
        "_load_model_capabilities",
        return_value={},
    ), patch("backend.services.providers.openrouter_provider.AsyncOpenAI"), \
         patch("backend.services.providers.openrouter_provider.settings") as mock_settings:
        mock_settings.OPENROUTER_API_KEY = "fake"
        mock_settings.OPENROUTER_PRESET = "premium"
        mock_settings.OPENROUTER_VISION_MODEL = ""
        mock_settings.OPENROUTER_TEXT_MODEL = ""
        mock_settings.OPENROUTER_SUMMARY_MODEL = ""
        from backend.services.providers.openrouter_provider import OpenRouterProvider

        provider = OpenRouterProvider()
        # Baseline: preset-resolved default.
        assert provider._vision_model == PRESETS["premium"]["vision"]
        base_before = provider._base_vision_model
        assert base_before == PRESETS["premium"]["vision"]

        # Fire the override for ANIME — swaps to Qwen3-VL, prepends
        # the original preset default to the fallbacks.
        returned = provider.apply_vision_model_override("anime")
        assert returned == QWEN3_VL
        assert provider._vision_model == QWEN3_VL
        assert base_before in provider._vision_fallbacks

        # Calling again with an override-target type is idempotent.
        provider.apply_vision_model_override("anime")
        assert provider._vision_model == QWEN3_VL
        # No-op for a non-override content type — returns the current
        # _vision_model (which is Qwen3-VL because we routed earlier),
        # but the override mechanism sets _base_vision_model as the
        # fallback so safety is preserved via the fallbacks list.
        returned_th = provider.apply_vision_model_override("talking_head")
        # Talking head resolves against the *base* preset default, not
        # the currently-overridden model — that's the intended reset
        # point. Assert the resolver logic is correct:
        assert returned_th == base_before


def test_apply_vision_model_override_logs_routing_decision(caplog):
    """Routing decisions are logged so the job report shows them."""
    with patch.object(
        __import__("backend.services.providers.openrouter_provider", fromlist=["_load_model_capabilities"]),
        "_load_model_capabilities",
        return_value={},
    ), patch("backend.services.providers.openrouter_provider.AsyncOpenAI"), \
         patch("backend.services.providers.openrouter_provider.settings") as mock_settings:
        mock_settings.OPENROUTER_API_KEY = "fake"
        mock_settings.OPENROUTER_PRESET = "balanced"
        mock_settings.OPENROUTER_VISION_MODEL = ""
        mock_settings.OPENROUTER_TEXT_MODEL = ""
        mock_settings.OPENROUTER_SUMMARY_MODEL = ""
        from backend.services.providers.openrouter_provider import OpenRouterProvider

        provider = OpenRouterProvider()
        with caplog.at_level(
            logging.INFO,
            logger="backend.services.providers.openrouter_provider",
        ):
            provider.apply_vision_model_override("gameplay")
        assert any("Using vision model for gameplay" in rec.message for rec in caplog.records)
