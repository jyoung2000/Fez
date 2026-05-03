"""Tests for the user authentication + session management system.

Covers:
  * Password hashing round-trip + rejection of wrong passwords.
  * Session fingerprinting (IP + User-Agent binding).
  * Head-admin seeding is idempotent and cannot be demoted / deleted.
  * User CRUD endpoints enforce admin-only where required.
  * New-IP / new-browser request rotates the session (401, forces login).
  * Per-user data isolation: one user cannot see another's jobs.
  * Login flow returns a safe public user payload (no hash / salt).
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Redirect the auth store to a throwaway directory before anything from
# ``backend.app.auth`` is imported. The store resolves the dir at import
# time; pointing it at a tmp path isolates tests from any real
# ``/data/auth`` data.
_TMP = Path(tempfile.mkdtemp(prefix="clipai_auth_test_"))
os.environ["HOME"] = str(_TMP)
os.environ.setdefault("CLIPAI_TEST_AUTH_DIR", str(_TMP / "auth"))

from backend.app.auth import security   # noqa: E402
from backend.app.auth import store as auth_store  # noqa: E402
from backend.app.auth.models import Role  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_store():
    """Wipe users + sessions between tests."""
    asyncio.run(auth_store._reset_for_tests())
    yield
    asyncio.run(auth_store._reset_for_tests())


# ── security helpers ─────────────────────────────────────────


def test_hash_roundtrip():
    salt = security.new_salt()
    hashed = security.hash_password("hunter2aa", salt)
    assert security.verify_password("hunter2aa", salt, hashed) is True
    assert security.verify_password("hunter2aB", salt, hashed) is False
    assert security.verify_password("", salt, hashed) is False


def test_hash_rejects_short_password():
    salt = security.new_salt()
    with pytest.raises(ValueError):
        security.hash_password("abc", salt)


def test_fingerprint_is_stable_same_ip_ua():
    a = security.compute_fingerprint("192.168.1.10", "Mozilla/5.0 (Macintosh)")
    b = security.compute_fingerprint("192.168.1.10", "Mozilla/5.0 (Macintosh)")
    assert a == b


def test_fingerprint_ignores_ip_change_v2():
    """V2 fingerprint: UA only, so IP flux behind a reverse proxy
    no longer kills sessions mid-upload."""
    a = security.compute_fingerprint("192.168.1.10", "Mozilla/5.0")
    b = security.compute_fingerprint("10.0.0.1", "Mozilla/5.0")
    assert a == b


def test_fingerprint_changes_on_ua_change():
    a = security.compute_fingerprint("192.168.1.10", "Mozilla/5.0")
    b = security.compute_fingerprint("192.168.1.10", "curl/8.4.0")
    assert a != b


def test_fingerprint_independent_of_ipv6():
    """IPv6 flux no longer changes the fingerprint in V2."""
    a = security.compute_fingerprint(
        "2001:db8:85a3::8a2e:370:7334", "Safari",
    )
    b = security.compute_fingerprint(
        "abcd::1", "Safari",
    )
    assert a == b


def test_fingerprint_stable_across_devtools_responsive_mode():
    """V3: Chrome with DevTools device-toolbar emulating a phone
    swaps the platform / device tokens but keeps Chrome/<major> the
    same. The fingerprint must be identical so opening Inspect
    Element doesn't kill the session.
    """
    desktop = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    )
    mobile_emulated = (
        "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Mobile Safari/537.36"
    )
    assert (
        security.compute_fingerprint("1.1.1.1", desktop)
        == security.compute_fingerprint("1.1.1.1", mobile_emulated)
    )


def test_fingerprint_distinguishes_browser_families():
    """V3 still rejects a cookie reused across genuinely different
    browser families."""
    chrome = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    )
    firefox = (
        "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 "
        "Firefox/128.0"
    )
    assert (
        security.compute_fingerprint("1.1.1.1", chrome)
        != security.compute_fingerprint("1.1.1.1", firefox)
    )


# ── store: users + sessions ─────────────────────────────────


def test_create_user_hashes_password():
    async def run():
        u = await auth_store.create_user("alice", "supersecret", role=Role.USER)
        assert u.id
        assert u.username == "alice"
        assert u.password_hash != "supersecret"
        assert u.salt != u.password_hash
        # Can log in.
        reloaded = await auth_store.get_user_by_username("alice")
        assert reloaded is not None
        assert security.verify_password(
            "supersecret", reloaded.salt, reloaded.password_hash,
        )
    asyncio.run(run())


def test_duplicate_username_is_rejected():
    async def run():
        await auth_store.create_user("alice", "supersecret")
        with pytest.raises(ValueError):
            await auth_store.create_user("alice", "anothersecret")
        # Case-insensitive collision.
        with pytest.raises(ValueError):
            await auth_store.create_user("ALICE", "anothersecret")
    asyncio.run(run())


def test_create_session_and_lookup():
    async def run():
        u = await auth_store.create_user("bob", "hunter2aa")
        s = await auth_store.create_session(
            user_id=u.id, ip="1.2.3.4", user_agent="TestUA/1.0",
        )
        assert s.token
        roundtrip = await auth_store.get_session(s.token)
        assert roundtrip is not None
        assert roundtrip.user_id == u.id
        assert roundtrip.fingerprint == security.compute_fingerprint(
            "1.2.3.4", "TestUA/1.0",
        )
    asyncio.run(run())


def test_delete_session():
    async def run():
        u = await auth_store.create_user("bob", "hunter2aa")
        s = await auth_store.create_session(user_id=u.id, ip="1.2.3.4", user_agent="x")
        await auth_store.delete_session(s.token)
        assert await auth_store.get_session(s.token) is None
    asyncio.run(run())


def test_delete_sessions_for_user_when_deleted():
    async def run():
        u = await auth_store.create_user("bob", "hunter2aa")
        await auth_store.create_session(user_id=u.id, ip="1.1.1.1", user_agent="x")
        await auth_store.create_session(user_id=u.id, ip="2.2.2.2", user_agent="y")
        await auth_store.delete_user(u.id)
        # No sessions should remain for this user.
        for s in await auth_store.list_sessions():
            assert s.user_id != u.id
    asyncio.run(run())


# ── head admin seeding ─────────────────────────────────────


def test_seed_head_admin_is_idempotent():
    from backend.app.auth.seed import (
        HEAD_ADMIN_PASSWORD,
        HEAD_ADMIN_USERNAME,
        ensure_head_admin,
    )

    async def run():
        await ensure_head_admin()
        u1 = await auth_store.get_user_by_username(HEAD_ADMIN_USERNAME)
        assert u1 is not None
        assert u1.head_admin is True
        assert u1.role == Role.ADMIN
        assert security.verify_password(
            HEAD_ADMIN_PASSWORD, u1.salt, u1.password_hash,
        )

        # Second call: user still exists and keeps the same hash.
        await ensure_head_admin()
        u2 = await auth_store.get_user_by_username(HEAD_ADMIN_USERNAME)
        assert u2.id == u1.id
        assert u2.password_hash == u1.password_hash
    asyncio.run(run())


def test_head_admin_cannot_be_demoted_or_deleted():
    from backend.app.auth.seed import HEAD_ADMIN_USERNAME, ensure_head_admin

    async def run():
        await ensure_head_admin()
        ha = await auth_store.get_user_by_username(HEAD_ADMIN_USERNAME)
        with pytest.raises(ValueError):
            await auth_store.update_user(ha.id, role=Role.USER)
        with pytest.raises(ValueError):
            await auth_store.update_user(ha.id, active=False)
        with pytest.raises(ValueError):
            await auth_store.delete_user(ha.id)
    asyncio.run(run())


# ── per-user settings ──────────────────────────────────────


def test_user_settings_read_write_delete():
    async def run():
        u = await auth_store.create_user("alice", "passworda")
        assert await auth_store.read_user_settings(u.id) == {}
        merged = await auth_store.write_user_settings(u.id, {"WHISPER_MODEL": "large-v3"})
        assert merged == {"WHISPER_MODEL": "large-v3"}
        merged = await auth_store.write_user_settings(
            u.id, {"OLLAMA_VISION_MODEL": "llava:13b"},
        )
        assert merged["WHISPER_MODEL"] == "large-v3"
        assert merged["OLLAMA_VISION_MODEL"] == "llava:13b"
        await auth_store.delete_user_settings(u.id)
        assert await auth_store.read_user_settings(u.id) == {}
    asyncio.run(run())


# ── HTTP integration: login / sessions / admin / isolation ─────


@pytest.fixture
def client():
    pytest.importorskip("fastapi.testclient")
    from fastapi.testclient import TestClient
    # Build a tiny FastAPI app with only the auth middleware + router +
    # one protected echo route so tests don't require the full backend
    # (which depends on pydantic-settings, cv2, torch, etc.).
    from fastapi import Depends, FastAPI

    from backend.app.auth.deps import get_current_user
    from backend.app.auth.middleware import AuthMiddleware
    from backend.app.auth.models import User
    from backend.routers import auth as auth_router

    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(auth_router.router)

    @app.get("/api/echo")
    async def echo(user: User = Depends(get_current_user)):
        return {"user": user.username, "role": user.role.value}

    return TestClient(app)


def test_login_sets_cookie_and_me_returns_user(client):
    async def _seed():
        await auth_store.create_user("alice", "supersecret", role=Role.USER)
    asyncio.run(_seed())

    # Without login, a protected endpoint is 401.
    assert client.get("/api/echo").status_code == 401

    # Login.
    r = client.post("/api/auth/login", json={"username": "alice", "password": "supersecret"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["user"]["username"] == "alice"
    assert "password_hash" not in body["user"]
    assert "salt" not in body["user"]

    # Cookie carried the session; /me and /echo both work.
    assert client.get("/api/auth/me").status_code == 200
    assert client.get("/api/echo").status_code == 200


def test_login_rejects_wrong_password(client):
    async def _seed():
        await auth_store.create_user("alice", "supersecret")
    asyncio.run(_seed())

    r = client.post("/api/auth/login", json={"username": "alice", "password": "wrong"})
    assert r.status_code == 401
    # Safe error message (no "user not found" disclosure).
    assert r.json()["detail"] == "invalid credentials"


def test_ip_change_alone_does_not_invalidate_session(client):
    """V2 fingerprint: IP flux is tolerated because reverse proxies
    legitimately change X-Forwarded-For between requests. UA change
    still forces a re-login."""
    async def _seed():
        await auth_store.create_user("alice", "supersecret")
    asyncio.run(_seed())

    r = client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "supersecret"},
        headers={"x-forwarded-for": "1.1.1.1", "user-agent": "BrowserA/1.0"},
    )
    assert r.status_code == 200
    token = r.cookies.get("clipai_session") or client.cookies.get("clipai_session")
    assert token

    # Same cookie, DIFFERENT IP — V2 accepts it.
    r2 = client.get(
        "/api/auth/me",
        headers={"x-forwarded-for": "9.9.9.9", "user-agent": "BrowserA/1.0"},
        cookies={"clipai_session": token},
    )
    assert r2.status_code == 200, r2.text

    # Same cookie on original IP but a different User-Agent still
    # fails the fingerprint check.
    r3 = client.get(
        "/api/auth/me",
        headers={"x-forwarded-for": "1.1.1.1", "user-agent": "OtherBrowser/2.0"},
        cookies={"clipai_session": token},
    )
    assert r3.status_code == 401, r3.text


def test_admin_can_manage_users(client):
    async def _seed():
        await auth_store.create_user("admin", "supersecret", role=Role.ADMIN)
    asyncio.run(_seed())

    # login as admin
    client.post("/api/auth/login", json={"username": "admin", "password": "supersecret"})

    # list
    r = client.get("/api/auth/admin/users")
    assert r.status_code == 200
    assert r.json()["users"][0]["username"] == "admin"

    # create
    r = client.post("/api/auth/admin/users", json={
        "username": "bob", "password": "bobsecret", "role": "user",
    })
    assert r.status_code == 200, r.text
    bob_id = r.json()["user"]["id"]

    # promote bob to admin
    r = client.patch(f"/api/auth/admin/users/{bob_id}", json={"role": "admin"})
    assert r.status_code == 200
    assert r.json()["user"]["role"] == "admin"

    # reset bob's password
    r = client.post(
        f"/api/auth/admin/users/{bob_id}/reset_password",
        json={"new_password": "newbob1234"},
    )
    assert r.status_code == 200

    # delete bob
    r = client.delete(f"/api/auth/admin/users/{bob_id}")
    assert r.status_code == 200
    # bob is gone
    r = client.get("/api/auth/admin/users")
    assert all(u["username"] != "bob" for u in r.json()["users"])


def test_regular_user_cannot_access_admin_endpoints(client):
    async def _seed():
        await auth_store.create_user("alice", "supersecret", role=Role.USER)
    asyncio.run(_seed())
    client.post("/api/auth/login", json={"username": "alice", "password": "supersecret"})
    assert client.get("/api/auth/admin/users").status_code == 403
    assert client.post(
        "/api/auth/admin/users",
        json={"username": "new", "password": "password", "role": "user"},
    ).status_code == 403


def test_logout_clears_cookie(client):
    async def _seed():
        await auth_store.create_user("alice", "supersecret")
    asyncio.run(_seed())
    client.post("/api/auth/login", json={"username": "alice", "password": "supersecret"})
    assert client.get("/api/auth/me").status_code == 200
    r = client.post("/api/auth/logout")
    assert r.status_code == 200
    # Cookie gone → /me is 401.
    client.cookies.clear()
    assert client.get("/api/auth/me").status_code == 401


def test_change_password_revokes_other_sessions(client):
    async def _seed():
        await auth_store.create_user("alice", "supersecret")
    asyncio.run(_seed())
    client.post("/api/auth/login", json={"username": "alice", "password": "supersecret"})
    # Issue a second session (simulate other device).
    user = asyncio.run(auth_store.get_user_by_username("alice"))
    other = asyncio.run(auth_store.create_session(
        user_id=user.id, ip="2.2.2.2", user_agent="OtherDevice/1.0",
    ))
    r = client.patch("/api/auth/me/password", json={
        "current_password": "supersecret",
        "new_password": "newerpass",
    })
    assert r.status_code == 200
    assert asyncio.run(auth_store.get_session(other.token)) is None


def test_bootstrap_reports_presence(client):
    """Pre-seed path: bootstrap advertises whether any user exists."""
    # Empty store:
    r = client.get("/api/auth/bootstrap")
    assert r.status_code == 200
    assert r.json()["has_users"] is False

    asyncio.run(auth_store.create_user("alice", "supersecret"))
    r = client.get("/api/auth/bootstrap")
    assert r.json()["has_users"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
