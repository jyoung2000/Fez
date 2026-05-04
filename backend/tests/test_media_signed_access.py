"""End-to-end tests for the ``/api/files/{job_id}/{path}`` signed-URL
fast path added in the "signed media URLs" fix.

The real ``serve_file`` handler reaches into ``backend.database`` and
``backend.services.browser_preview`` which require the full backend
install. For a unit-test-scale build we stub those modules and
exercise only the auth branch: the middleware bypass for
``?exp=&sig=``, the signature check inside the handler, and the
fall-through to cookie auth when the signature is absent or
incomplete.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.testclient import TestClient


_TMP = Path(tempfile.mkdtemp(prefix="clipai_media_signed_test_"))
os.environ["HOME"] = str(_TMP)

from backend.app.auth import security  # noqa: E402
from backend.app.auth import store as auth_store  # noqa: E402
from backend.app.auth.middleware import AuthMiddleware  # noqa: E402
from backend.app.auth.store import get_user as _get_user  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_store():
    asyncio.run(auth_store._reset_for_tests())
    yield
    asyncio.run(auth_store._reset_for_tests())


_app = FastAPI()
_app.add_middleware(AuthMiddleware)


@_app.get("/api/files/{job_id}/{path:path}")
async def _serve(job_id: str, path: str, request: Request):
    exp_raw = request.query_params.get("exp")
    sig = request.query_params.get("sig")
    if exp_raw and sig:
        try:
            exp_int = int(exp_raw)
        except (TypeError, ValueError):
            return Response(status_code=401, content="invalid signature")
        claimed = request.query_params.get("u") or ""
        if not security.verify_media_signature(claimed, job_id, path, exp_int, sig):
            return Response(status_code=401, content="invalid signature")
        user = await _get_user(claimed)
        if user is None or not user.active:
            return Response(status_code=401, content="invalid signature")
        return Response(status_code=200, content=f"signed:{user.id}:{job_id}:{path}")
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    return Response(status_code=200, content=f"cookie:{user.id}:{job_id}:{path}")


client = TestClient(_app)


def test_signed_url_succeeds_without_cookie():
    async def _seed():
        return await auth_store.create_user("alice", "supersecret")
    u = asyncio.run(_seed())

    security._reset_signing_key_cache_for_tests()
    expiry, sig = security.sign_media_url(u.id, "job-1", "video.mp4")

    r = client.get(
        f"/api/files/job-1/video.mp4?exp={expiry}&sig={sig}&u={u.id}",
    )
    assert r.status_code == 200, r.text
    assert r.text == f"signed:{u.id}:job-1:video.mp4"


def test_expired_signed_url_fails():
    async def _seed():
        return await auth_store.create_user("alice", "supersecret")
    u = asyncio.run(_seed())

    security._reset_signing_key_cache_for_tests()
    past = time.time() - 1_000_000
    expiry, sig = security.sign_media_url(
        u.id, "job-1", "video.mp4", ttl=10, now=past,
    )

    r = client.get(
        f"/api/files/job-1/video.mp4?exp={expiry}&sig={sig}&u={u.id}",
    )
    assert r.status_code == 401


def test_signed_url_for_user_a_cannot_access_user_b_job():
    """The signature binds the user_id → claiming a different ``u`` in
    the query string breaks the HMAC check outright."""
    async def _seed():
        a = await auth_store.create_user("alice", "supersecret")
        b = await auth_store.create_user("bob", "bobsecret")
        return a, b
    a, b = asyncio.run(_seed())

    security._reset_signing_key_cache_for_tests()
    expiry, sig = security.sign_media_url(a.id, "job-1", "video.mp4")

    r = client.get(
        f"/api/files/job-1/video.mp4?exp={expiry}&sig={sig}&u={b.id}",
    )
    assert r.status_code == 401


def test_missing_sig_falls_through_to_cookie_auth():
    """If only ``exp`` is present, the middleware does NOT grant the
    bypass — the normal cookie path runs, which rejects the
    unauthenticated request with 401."""
    async def _seed():
        return await auth_store.create_user("alice", "supersecret")
    u = asyncio.run(_seed())

    security._reset_signing_key_cache_for_tests()
    expiry, _sig = security.sign_media_url(u.id, "job-1", "video.mp4")

    r = client.get(f"/api/files/job-1/video.mp4?exp={expiry}")
    assert r.status_code == 401


def test_tampered_signature_fails():
    async def _seed():
        return await auth_store.create_user("alice", "supersecret")
    u = asyncio.run(_seed())

    security._reset_signing_key_cache_for_tests()
    expiry, sig = security.sign_media_url(u.id, "job-1", "video.mp4")
    flipped = ("0" if sig[0] != "0" else "1") + sig[1:]

    r = client.get(
        f"/api/files/job-1/video.mp4?exp={expiry}&sig={flipped}&u={u.id}",
    )
    assert r.status_code == 401
