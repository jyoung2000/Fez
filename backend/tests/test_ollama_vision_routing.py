"""Phase 2 parity — Ollama content-type vision routing.

Covers Change 3 of the VLM-upgrade wiring prompt: the Ollama provider
must expose the same content-type → vision-model mapping as OpenRouter
so anime / gameplay jobs automatically route to a better-suited model
when the user has set a content_type_override.
"""

import pytest


def test_select_ollama_vision_returns_default_for_none():
    from backend.services.providers.ollama_provider import (
        select_ollama_vision_model_for_content,
    )
    assert select_ollama_vision_model_for_content(None, "moondream:1.8b") == "moondream:1.8b"


def test_select_ollama_vision_returns_default_for_unknown():
    from backend.services.providers.ollama_provider import (
        select_ollama_vision_model_for_content,
    )
    assert select_ollama_vision_model_for_content("podcast", "moondream:1.8b") == "moondream:1.8b"


@pytest.mark.parametrize("content_type,expected_substr", [
    ("anime", "minicpm"),
    ("animation", "minicpm"),
    ("cartoon", "minicpm"),
    ("gameplay", "qwen2.5vl"),
    ("gameplay_fps", "qwen2.5vl"),
])
def test_select_ollama_vision_routes_known_types(content_type, expected_substr):
    from backend.services.providers.ollama_provider import (
        select_ollama_vision_model_for_content,
    )
    result = select_ollama_vision_model_for_content(content_type, "moondream:1.8b")
    assert expected_substr in result.lower()


def test_select_ollama_vision_case_insensitive():
    from backend.services.providers.ollama_provider import (
        select_ollama_vision_model_for_content,
    )
    assert "minicpm" in select_ollama_vision_model_for_content(
        "ANIME", "moondream:1.8b",
    ).lower()


def test_apply_vision_model_override_idempotent():
    """Calling the override twice with the same content type must not
    churn the model. Reverting with ``None`` must restore the base."""
    from backend.services.providers.ollama_provider import OllamaProvider

    p = OllamaProvider()
    initial = p._vision_model
    r1 = p.apply_vision_model_override("anime")
    r2 = p.apply_vision_model_override("anime")
    assert r1 == r2
    # And reverting to no-override goes back to the base.
    p.apply_vision_model_override(None)
    assert p._vision_model == initial


def test_apply_vision_model_override_swaps_for_anime():
    from backend.services.providers.ollama_provider import OllamaProvider

    p = OllamaProvider()
    initial = p._vision_model
    p.apply_vision_model_override("anime")
    assert p._vision_model != initial
    assert "minicpm" in p._vision_model.lower()
