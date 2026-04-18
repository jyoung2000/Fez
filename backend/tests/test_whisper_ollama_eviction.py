"""v2 Phase 11 — Fix 6: Whisper subprocess launcher evicts Ollama VRAM.

Ollama's idle CUDA context (~1.6 GB) is enough to evict Whisper from a
4 GB 1650, forcing the ctranslate2 CPU int8 fallback which is 10-30x
slower. Before launching the subprocess, the launcher should query
``/api/ps``, post ``keep_alive=0`` to each loaded model, and sleep
briefly so the driver can reclaim VRAM.
"""

import asyncio
from unittest import mock

import pytest


@pytest.mark.asyncio
async def test_evict_ollama_for_whisper_unloads_loaded_models(monkeypatch):
    from backend.services import transcription

    posts: list[dict] = []

    class _FakeResp:
        def __init__(self, payload, status=200):
            self._payload = payload
            self.status_code = status

        def json(self):
            return self._payload

    class _FakeAsyncClient:
        def __init__(self, timeout=None):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url):
            assert url.endswith("/api/ps")
            return _FakeResp({
                "models": [
                    {"name": "qwen3-vl:8b"},
                    {"name": "llama3.1:8b"},
                ],
            })

        async def post(self, url, json=None):
            assert url.endswith("/api/generate")
            assert json["keep_alive"] == 0
            posts.append(json)
            return _FakeResp({}, status=200)

    fake_httpx = mock.MagicMock()
    fake_httpx.AsyncClient = _FakeAsyncClient
    monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)

    # Patch asyncio.sleep so the test doesn't actually block 2 s.
    async def _nosleep(_):
        return None
    monkeypatch.setattr(asyncio, "sleep", _nosleep)

    await transcription._evict_ollama_for_whisper()

    assert len(posts) == 2, f"expected two eviction POSTs, got {posts}"
    model_names = {p["model"] for p in posts}
    assert model_names == {"qwen3-vl:8b", "llama3.1:8b"}


@pytest.mark.asyncio
async def test_evict_ollama_for_whisper_no_models_loaded(monkeypatch):
    from backend.services import transcription

    class _FakeResp:
        status_code = 200

        def json(self):
            return {"models": []}

    class _FakeAsyncClient:
        def __init__(self, timeout=None):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def get(self, url):
            return _FakeResp()

        async def post(self, url, json=None):
            raise AssertionError("should not POST when no models are loaded")

    fake_httpx = mock.MagicMock()
    fake_httpx.AsyncClient = _FakeAsyncClient
    monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)

    # Should complete silently without raising.
    await transcription._evict_ollama_for_whisper()


@pytest.mark.asyncio
async def test_evict_ollama_for_whisper_ollama_unreachable(monkeypatch):
    from backend.services import transcription

    class _FakeAsyncClient:
        def __init__(self, timeout=None):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def get(self, url):
            raise RuntimeError("connection refused")

        async def post(self, url, json=None):
            raise AssertionError("should not POST when /api/ps fails")

    fake_httpx = mock.MagicMock()
    fake_httpx.AsyncClient = _FakeAsyncClient
    monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)

    # Best-effort: unreachable Ollama must not raise.
    await transcription._evict_ollama_for_whisper()
