"""Regression: /api/providers/models/available must honor the per-user
AI_FALLBACK_CHAIN when deciding whether to enumerate Ollama models.

Before the fix, this endpoint read from the install-wide ``Settings``
singleton (which defaults to ``"openrouter,gemini,groq"`` — no ollama),
so users who toggled Ollama on got empty dropdowns even though
``/api/providers/status`` correctly reported ``_active.ollama_enabled: true``.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.app.auth.deps import get_current_user
    from backend.app.auth.models import Role, User
    from backend.routers.settings import router as settings_router

    _HAS_FASTAPI = True
except Exception:  # pragma: no cover — missing deps in test env
    _HAS_FASTAPI = False


_FAKE_OLLAMA_TAGS = {
    "models": [
        {
            "name": "moondream:1.8b",
            "size": 1_000_000_000,
            "details": {"parameter_size": "1.8B", "quantization_level": "Q4_0"},
        },
        {
            "name": "qwen2.5:3b-instruct",
            "size": 1_800_000_000,
            "details": {"parameter_size": "3B", "quantization_level": "Q4_0"},
        },
    ]
}


def _make_user(uid: str = "user_a") -> "User":
    return User(
        id=uid,
        username=uid,
        password_hash="x",
        salt="x",
        role=Role.USER,
        created_at="2026-01-01T00:00:00",
    )


def _httpx_client_returning(payload: dict):
    """Return a replacement for ``httpx.AsyncClient`` whose async-context
    instances respond to ``.get()`` with ``payload``."""

    class _Resp:
        status_code = 200

        def json(self):
            return payload

        def raise_for_status(self):
            pass

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            return _Resp()

    return _FakeAsyncClient


def _build_client(user_settings: dict, user: "User", ollama_tags: dict):
    """Build a TestClient with ``get_current_user`` and the two external
    side-effects (``read_user_settings`` and ``httpx.AsyncClient``) mocked."""
    app = FastAPI()
    # ``settings_router`` is already defined with ``prefix="/api"``, so
    # include it without an additional prefix.
    app.include_router(settings_router)
    app.dependency_overrides[get_current_user] = lambda: user

    patches = [
        patch(
            "backend.routers.settings.read_user_settings",
            AsyncMock(return_value=user_settings),
        ),
        patch(
            "backend.routers.settings.httpx.AsyncClient",
            _httpx_client_returning(ollama_tags),
        ),
        # Avoid calling out to openrouter.ai during tests even when a key
        # is unset (defensive — the endpoint already skips when no key).
        patch(
            "backend.routers.settings._fetch_openrouter_models",
            AsyncMock(return_value=None),
        ),
    ]
    for p in patches:
        p.start()

    client = TestClient(app)
    client._patches = patches  # type: ignore[attr-defined]
    return client


def _teardown(client):
    for p in getattr(client, "_patches", []):
        p.stop()


@pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi/httpx not available")
def test_available_models_includes_ollama_for_user_with_ollama_in_chain():
    """User A has ollama in their per-user chain → dropdowns include ollama entries,
    even though the install-wide chain does not contain ollama."""
    user = _make_user("user_a")
    user_settings = {
        "AI_FALLBACK_CHAIN": "ollama,openrouter",
        "OLLAMA_VISION_MODEL": "moondream:1.8b",
        "OLLAMA_TEXT_MODEL": "qwen2.5:3b-instruct",
    }

    client = _build_client(user_settings, user, _FAKE_OLLAMA_TAGS)
    try:
        # Install-wide chain intentionally does NOT contain ollama — matches prod default.
        with patch("backend.routers.settings.settings.AI_FALLBACK_CHAIN", "openrouter,gemini,groq"):
            resp = client.get("/api/providers/models/available")

        assert resp.status_code == 200
        data = resp.json()

        vision_providers = {m["provider"] for m in data["vision"]}
        text_providers = {m["provider"] for m in data["text"]}

        assert "ollama" in vision_providers, (
            "Vision dropdown is missing Ollama models despite Ollama being in the "
            "user's fallback chain — regression of the global-vs-per-user chain bug."
        )
        assert "ollama" in text_providers, (
            "Text dropdown is missing Ollama models despite Ollama being in the "
            "user's fallback chain — regression of the global-vs-per-user chain bug."
        )

        vision_ids = {m["id"] for m in data["vision"]}
        text_ids = {m["id"] for m in data["text"]}
        # moondream has vision; qwen2.5 is text-only.
        assert "ollama/moondream:1.8b" in vision_ids
        assert "ollama/qwen2.5:3b-instruct" in text_ids
    finally:
        _teardown(client)


@pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi/httpx not available")
def test_available_models_excludes_ollama_for_user_without_ollama_in_chain():
    """User B has no ollama in their per-user chain → dropdowns have zero ollama
    entries, even if the install-wide chain somehow contained it. This pins the
    direction of the fix — per-user wins over global."""
    user = _make_user("user_b")
    user_settings = {
        "AI_FALLBACK_CHAIN": "openrouter",
    }

    client = _build_client(user_settings, user, _FAKE_OLLAMA_TAGS)
    try:
        # Even if the global chain is misconfigured to include ollama, per-user wins.
        with patch("backend.routers.settings.settings.AI_FALLBACK_CHAIN", "ollama,openrouter"):
            resp = client.get("/api/providers/models/available")

        assert resp.status_code == 200
        data = resp.json()

        vision_providers = {m["provider"] for m in data["vision"]}
        text_providers = {m["provider"] for m in data["text"]}
        assert "ollama" not in vision_providers
        assert "ollama" not in text_providers
    finally:
        _teardown(client)


@pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi/httpx not available")
def test_available_models_shows_users_ollama_pick_even_before_pull_completes():
    """User picked llava:7b but /api/tags hasn't returned it yet (pull in progress) —
    the 'pulling...' placeholder must reflect THEIR pick, not the install default."""
    user = _make_user("user_c")
    user_settings = {
        "AI_FALLBACK_CHAIN": "ollama",
        "OLLAMA_VISION_MODEL": "llava:7b",       # user's pick
        "OLLAMA_TEXT_MODEL": "llama3.1:8b",      # user's pick
    }

    # /api/tags returns an empty model list (pull not done yet).
    client = _build_client(user_settings, user, {"models": []})
    try:
        resp = client.get("/api/providers/models/available")
        assert resp.status_code == 200
        data = resp.json()

        vision_ids = {m["id"] for m in data["vision"]}
        text_ids = {m["id"] for m in data["text"]}

        assert "ollama/llava:7b" in vision_ids, (
            "User's in-progress Ollama vision pick missing from dropdown — the "
            "placeholder-while-pulling fallback should read from `us`, not `settings`."
        )
        assert "ollama/llama3.1:8b" in text_ids, (
            "User's in-progress Ollama text pick missing from dropdown — the "
            "placeholder-while-pulling fallback should read from `us`, not `settings`."
        )
    finally:
        _teardown(client)
