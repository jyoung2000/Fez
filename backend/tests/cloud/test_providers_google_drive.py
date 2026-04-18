"""Unit tests for the Google Drive provider.

We replace the module-level ``httpx.AsyncClient`` with a factory that
returns a client backed by ``httpx.MockTransport``, so no network is
required. The tests cover:

- OAuth code exchange returns a TokenBundle with expires_at set.
- Folder listing parses nextPageToken and filters non-video MIMEs.
- Search builds the correct ``q`` parameter.
- 401 on the ``get_file_metadata`` path surfaces as a
  ``CloudProviderError(status_code=401)`` so the service layer can
  trigger a refresh.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Callable

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from backend.app.cloud.providers import google_drive as gd
from backend.app.cloud.retry import CloudProviderError

# Stash a reference to the real AsyncClient so the factory can build one
# even after ``gd.httpx.AsyncClient`` has been monkey-patched to point at
# the factory itself (otherwise the factory would call itself).
_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _mock_client_factory(handler: Callable[[httpx.Request], httpx.Response]):
    """Return a factory usable in place of ``httpx.AsyncClient(...)``.

    The real module code calls ``httpx.AsyncClient(timeout=...)``; we
    strip the timeout kwarg and pin the transport to a MockTransport so
    no real socket is opened.
    """

    def factory(*args, **kwargs):
        kwargs.pop("timeout", None)
        kwargs.pop("transport", None)
        return _REAL_ASYNC_CLIENT(
            transport=httpx.MockTransport(handler),
            **kwargs,
        )

    return factory


@pytest.fixture
def provider(monkeypatch):
    # Make the module think credentials are configured.
    from backend import config

    monkeypatch.setattr(config.settings, "GOOGLE_DRIVE_CLIENT_ID", "cid", raising=False)
    monkeypatch.setattr(config.settings, "GOOGLE_DRIVE_CLIENT_SECRET", "csec", raising=False)
    monkeypatch.setattr(
        config.settings,
        "GOOGLE_DRIVE_REDIRECT_URI",
        "http://example.test/cb",
        raising=False,
    )
    return gd.GoogleDriveProvider()


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_build_authorize_url_contains_scope_and_state(provider):
    url = provider.build_authorize_url("s123")
    assert "client_id=cid" in url
    assert "state=s123" in url
    assert "drive.readonly" in url
    assert "access_type=offline" in url


def test_exchange_code_populates_expires_at(provider, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "oauth2.googleapis.com"
        return httpx.Response(
            200,
            json={
                "access_token": "AT",
                "refresh_token": "RT",
                "expires_in": 3600,
                "scope": "https://www.googleapis.com/auth/drive.readonly openid",
                "token_type": "Bearer",
            },
        )

    monkeypatch.setattr(gd.httpx, "AsyncClient", _mock_client_factory(handler))

    bundle = asyncio.get_event_loop().run_until_complete(provider.exchange_code("code"))
    assert bundle.access_token == "AT"
    assert bundle.refresh_token == "RT"
    assert bundle.expires_at is not None
    assert "https://www.googleapis.com/auth/drive.readonly" in bundle.scopes


def test_list_folder_filters_non_video(provider, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/drive/v3/files"
        assert "mimeType" in request.url.query.decode()
        return httpx.Response(
            200,
            json={
                "nextPageToken": "next-abc",
                "files": [
                    {
                        "id": "f1",
                        "name": "nested",
                        "mimeType": "application/vnd.google-apps.folder",
                    },
                    {
                        "id": "v1",
                        "name": "clip.mp4",
                        "mimeType": "video/mp4",
                        "size": "12345",
                        "modifiedTime": "2024-01-01T00:00:00Z",
                        "thumbnailLink": "https://thumb/1",
                    },
                    {
                        "id": "d1",
                        "name": "notes.txt",
                        "mimeType": "text/plain",
                    },
                ],
            },
        )

    monkeypatch.setattr(gd.httpx, "AsyncClient", _mock_client_factory(handler))

    page = asyncio.get_event_loop().run_until_complete(
        provider.list_folder("AT", None, None)
    )
    assert len(page.folders) == 1
    assert page.folders[0].id == "f1"
    assert len(page.files) == 1
    assert page.files[0].id == "v1"
    assert page.files[0].size == 12345
    assert page.next_page_token == "next-abc"


def test_search_uses_name_contains_query(provider, monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["q"] = dict(request.url.params).get("q", "")
        return httpx.Response(
            200,
            json={
                "files": [
                    {"id": "v2", "name": "match.mp4", "mimeType": "video/mp4", "size": "1"}
                ]
            },
        )

    monkeypatch.setattr(gd.httpx, "AsyncClient", _mock_client_factory(handler))
    page = asyncio.get_event_loop().run_until_complete(
        provider.search("AT", "match", None)
    )
    assert "name contains 'match'" in captured["q"]
    assert "mimeType contains 'video/'" in captured["q"]
    assert len(page.files) == 1


def test_get_file_metadata_401_raises_cloud_provider_error(provider, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    monkeypatch.setattr(gd.httpx, "AsyncClient", _mock_client_factory(handler))

    with pytest.raises(CloudProviderError) as info:
        asyncio.get_event_loop().run_until_complete(
            provider.get_file_metadata("AT", "v1")
        )
    assert info.value.status_code == 401
