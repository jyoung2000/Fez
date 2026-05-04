"""Tests for the soft fingerprint-mismatch behavior added by the
"don't delete session on fingerprint mismatch" fix.

Covers:
  * Fingerprint mismatch returns 401 + ``fingerprint_mismatch``
    detail, but leaves the session record intact.
  * Re-login with the same credentials works without requiring a
    password reset.
  * IP change alone does NOT trigger the fingerprint check.
  * Legacy sessions (V1: IP+UA fingerprint) are migrated to V2 on
    first access without a forced re-login.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import tempfile
from pathlib import Path

import pytest


_TMP = Path(tempfile.mkdtemp(prefix="clipai_fp_soft_test_"))
os.environ["HOME"] = str(_TMP)

from backend.app.auth import security  # noqa: E402
from backend.app.auth import store as auth_store  # noqa: E402


@pytest.fixture(autouse=True)
def _reset():
    asyncio.run(auth_store._reset_for_tests())
    yield
    asyncio.run(auth_store._reset_for_tests())


@pytest.fixture
def client():
    pytest.importorskip("fastapi.testclient")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.app.auth.middleware import AuthMiddleware
    from backend.routers import auth as auth_router

    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(auth_router.router)
    return TestClient(app)


def test_fingerprint_mismatch_returns_401_with_code(client):
    async def _seed():
        await auth_store.create_user("alice", "supersecret")
    asyncio.run(_seed())

    r = client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "supersecret"},
        headers={"user-agent": "BrowserA/1.0"},
    )
    assert r.status_code == 200
    token = r.cookies.get("clipai_session") or client.cookies.get("clipai_session")
    assert token

    # UA change → fingerprint mismatch → 401 with specific detail.
    r2 = client.get(
        "/api/auth/me",
        headers={"user-agent": "DifferentBrowser/9.9"},
        cookies={"clipai_session": token},
    )
    assert r2.status_code == 401
    assert r2.json()["detail"] == "fingerprint_mismatch"


def test_fingerprint_mismatch_does_not_delete_session(client):
    """Server-side session record survives the mismatch — the user
    can re-login on the same browser without a password reset."""
    async def _seed():
        await auth_store.create_user("alice", "supersecret")
    asyncio.run(_seed())

    r = client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "supersecret"},
        headers={"user-agent": "BrowserA/1.0"},
    )
    token = r.cookies.get("clipai_session") or client.cookies.get("clipai_session")

    # Trigger the mismatch.
    client.get(
        "/api/auth/me",
        headers={"user-agent": "DifferentBrowser/9.9"},
        cookies={"clipai_session": token},
    )

    # Session record is still present.
    async def _check():
        return await auth_store.get_session(token)
    got = asyncio.run(_check())
    assert got is not None
    assert got.token == token


def test_relogin_works_after_fingerprint_mismatch(client):
    async def _seed():
        await auth_store.create_user("alice", "supersecret")
    asyncio.run(_seed())

    # First login.
    r = client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "supersecret"},
        headers={"user-agent": "BrowserA/1.0"},
    )
    assert r.status_code == 200
    old_token = r.cookies.get("clipai_session") or client.cookies.get("clipai_session")

    # Simulate the cookie turning up from a different UA (mismatch).
    client.cookies.clear()
    r2 = client.get(
        "/api/auth/me",
        headers={"user-agent": "DifferentBrowser/9.9"},
        cookies={"clipai_session": old_token},
    )
    assert r2.status_code == 401

    # Re-login with the same creds — works, password unchanged.
    client.cookies.clear()
    r3 = client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "supersecret"},
        headers={"user-agent": "BrowserA/1.0"},
    )
    assert r3.status_code == 200


def test_ip_change_alone_is_accepted(client):
    """V2 fingerprint drops the IP component, so changing only the
    client's X-Forwarded-For does not trigger the fingerprint fail
    path at all."""
    async def _seed():
        await auth_store.create_user("alice", "supersecret")
    asyncio.run(_seed())

    r = client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "supersecret"},
        headers={
            "user-agent": "BrowserA/1.0",
            "x-forwarded-for": "1.1.1.1",
        },
    )
    token = r.cookies.get("clipai_session") or client.cookies.get("clipai_session")

    # Drastically different IP → still 200.
    r2 = client.get(
        "/api/auth/me",
        headers={
            "user-agent": "BrowserA/1.0",
            "x-forwarded-for": "9.9.9.9",
        },
        cookies={"clipai_session": token},
    )
    assert r2.status_code == 200


def test_legacy_session_migrates_to_current_version(client):
    """Pre-fix sessions on disk stored an older fingerprint (V1: IP+UA,
    V2: full UA). The middleware must accept such a session once under
    the current scheme and rewrite the stored fingerprint to the V3
    (browser-family + major-version) form.
    """
    async def _seed_legacy():
        u = await auth_store.create_user("alice", "supersecret")
        # Hand-craft a session with a V1-style fingerprint (IP+UA)
        # and both version flags left False to mark it as legacy.
        from backend.app.auth.models import Session
        from datetime import datetime, timedelta, timezone
        token = "legacy-token-abc"
        v1_fp = hashlib.sha256(b"1.1.1.0|BrowserA/1.0").hexdigest()[:32]
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        expires = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        async with auth_store._sessions_lock:
            data = await auth_store._read_json(
                auth_store.SESSIONS_PATH, {"sessions": []},
            )
            data.setdefault("sessions", []).append(Session(
                token=token, user_id=u.id,
                fingerprint=v1_fp,
                ip="1.1.1.1", user_agent="BrowserA/1.0",
                created_at=now, last_seen=now, expires_at=expires,
                remember=True, fp_v2=False, fp_v3=False,
            ).to_storage())
            await auth_store._atomic_write_json(auth_store.SESSIONS_PATH, data)
        return token
    token = asyncio.run(_seed_legacy())

    # Hit /me with the legacy cookie — migration lets it through.
    r = client.get(
        "/api/auth/me",
        headers={"user-agent": "BrowserA/1.0"},
        cookies={"clipai_session": token},
    )
    assert r.status_code == 200, r.text

    # The stored fingerprint was rewritten to V3.
    async def _reload():
        return await auth_store.get_session(token)
    got = asyncio.run(_reload())
    assert got is not None
    assert got.fp_v3 is True
    assert got.fingerprint == security.compute_fingerprint(
        "ignored", "BrowserA/1.0",
    )

    # A second request with a DIFFERENT browser family now fails the
    # mismatch check (no more migration path left).
    r2 = client.get(
        "/api/auth/me",
        headers={"user-agent": "AttackerBrowser/0.0"},
        cookies={"clipai_session": token},
    )
    assert r2.status_code == 401
    assert r2.json()["detail"] == "fingerprint_mismatch"
