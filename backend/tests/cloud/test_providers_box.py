"""Unit tests for the Box provider.

Proves that the Box-specific quirks are handled:

- Folder id 0 is used when the caller passes ``None``.
- Refresh-token rotation: the new refresh token from the response is
  returned in the bundle (not silently discarded).
- Search filters to the supported video file extensions.
- 401 on ``users/me`` surfaces as a ``CloudProviderError`` with
  ``status_code=401`` so the service layer can refresh.
"""
from __future__ import annotations

import asyncio
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from backend.app.cloud.providers import box as box_mod
from backend.app.cloud.retry import CloudProviderError

# See test_providers_google_drive for the reason we stash this.
_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _factory(handler):
    def _make(*args, **kwargs):
        kwargs.pop("timeout", None)
        kwargs.pop("transport", None)
        return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler), **kwargs)

    return _make


@pytest.fixture
def provider(monkeypatch):
    from backend import config

    monkeypatch.setattr(config.settings, "BOX_CLIENT_ID", "cid", raising=False)
    monkeypatch.setattr(config.settings, "BOX_CLIENT_SECRET", "csec", raising=False)
    monkeypatch.setattr(
        config.settings,
        "BOX_REDIRECT_URI",
        "http://example.test/cb",
        raising=False,
    )
    return box_mod.BoxProvider()


def test_authorize_url(provider):
    url = provider.build_authorize_url("s1")
    assert "client_id=cid" in url
    assert "state=s1" in url
    assert "response_type=code" in url


def test_refresh_returns_new_refresh_token(provider, monkeypatch):
    """Box rotates refresh tokens — make sure we pass the new one through."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "access_token": "AT2",
                "refresh_token": "RT_NEW",
                "expires_in": 3600,
                "token_type": "bearer",
            },
        )

    monkeypatch.setattr(box_mod.httpx, "AsyncClient", _factory(handler))
    bundle = asyncio.get_event_loop().run_until_complete(provider.refresh("RT_OLD"))
    assert bundle.access_token == "AT2"
    assert bundle.refresh_token == "RT_NEW"
    assert bundle.refresh_token != "RT_OLD"


def test_list_folder_defaults_to_root(provider, monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        return httpx.Response(
            200,
            json={
                "total_count": 2,
                "offset": 0,
                "limit": 100,
                "entries": [
                    {
                        "id": "10",
                        "type": "folder",
                        "name": "Clips",
                    },
                    {
                        "id": "11",
                        "type": "file",
                        "name": "talk.mp4",
                        "extension": "mp4",
                        "size": 5000,
                        "modified_at": "2024-01-01T00:00:00Z",
                    },
                ],
            },
        )

    monkeypatch.setattr(box_mod.httpx, "AsyncClient", _factory(handler))
    page = asyncio.get_event_loop().run_until_complete(
        provider.list_folder("AT", None, None)
    )
    assert captured["path"] == "/2.0/folders/0/items"
    assert len(page.folders) == 1
    assert len(page.files) == 1
    assert page.files[0].mime_type == "video/mp4"


def test_list_folder_skips_non_video_files(provider, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "total_count": 2,
                "offset": 0,
                "limit": 100,
                "entries": [
                    {
                        "id": "20",
                        "type": "file",
                        "name": "doc.pdf",
                        "extension": "pdf",
                        "size": 1000,
                    },
                    {
                        "id": "21",
                        "type": "file",
                        "name": "clip.mov",
                        "extension": "mov",
                        "size": 2000,
                    },
                ],
            },
        )

    monkeypatch.setattr(box_mod.httpx, "AsyncClient", _factory(handler))
    page = asyncio.get_event_loop().run_until_complete(
        provider.list_folder("AT", "0", None)
    )
    assert [f.id for f in page.files] == ["21"]


def test_users_me_401_raises(provider, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    monkeypatch.setattr(box_mod.httpx, "AsyncClient", _factory(handler))
    with pytest.raises(CloudProviderError) as info:
        asyncio.get_event_loop().run_until_complete(provider.get_account_info("AT"))
    assert info.value.status_code == 401
