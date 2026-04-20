"""Tests for the session + user TTL cache added to
``backend/app/auth/store.py``.

Covers:
  * Concurrent cached lookups do one disk read.
  * Cache expires after the TTL.
  * ``delete_session`` and ``delete_user`` invalidate immediately.
  * ``touch_session`` invalidates so the next read picks up new
    ``expires_at`` / ``last_seen`` values.
  * Negative lookups are cached (shorter TTL).
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


_TMP = Path(tempfile.mkdtemp(prefix="clipai_auth_cache_test_"))
os.environ["HOME"] = str(_TMP)

from backend.app.auth import store as auth_store  # noqa: E402
from backend.app.auth.models import Role  # noqa: E402


@pytest.fixture(autouse=True)
def _reset():
    asyncio.run(auth_store._reset_for_tests())
    yield
    asyncio.run(auth_store._reset_for_tests())


def test_concurrent_session_cached_calls_do_one_read():
    """Two concurrent cached lookups for the same token hit the
    cache once populated. ``list_sessions`` (the disk read) runs
    exactly once — the miss populates the cache, and the hit returns
    the cached value.
    """
    async def _run():
        u = await auth_store.create_user("alice", "supersecret")
        s = await auth_store.create_session(
            user_id=u.id, ip="1.1.1.1", user_agent="UA/1.0",
        )

        # Prime the cache with one call.
        await auth_store.get_session_cached(s.token)

        # Now a storm of concurrent lookups should all hit the cache.
        with patch.object(auth_store, "list_sessions",
                          side_effect=AssertionError("disk read")) as mock_read:
            results = await asyncio.gather(*[
                auth_store.get_session_cached(s.token) for _ in range(20)
            ])
            mock_read.assert_not_called()
        assert all(r is not None and r.token == s.token for r in results)

        stats = auth_store.get_cache_stats()
        assert stats["session_hits"] >= 20

    asyncio.run(_run())


def test_cache_expires_after_ttl(monkeypatch):
    """After the TTL elapses, the next cached call does a fresh read."""
    async def _run():
        u = await auth_store.create_user("alice", "supersecret")
        s = await auth_store.create_session(
            user_id=u.id, ip="1.1.1.1", user_agent="UA/1.0",
        )
        # Prime.
        await auth_store.get_session_cached(s.token)
        # Fast-forward time past the TTL.
        fake_now = time.monotonic() + auth_store._SESSION_CACHE_TTL + 1
        monkeypatch.setattr(time, "monotonic", lambda: fake_now)
        miss_count_before = auth_store._cache_stats["session_misses"]
        got = await auth_store.get_session_cached(s.token)
        miss_count_after = auth_store._cache_stats["session_misses"]
        assert got is not None and got.token == s.token
        assert miss_count_after == miss_count_before + 1

    asyncio.run(_run())


def test_delete_session_invalidates_cache():
    async def _run():
        u = await auth_store.create_user("alice", "supersecret")
        s = await auth_store.create_session(
            user_id=u.id, ip="1.1.1.1", user_agent="UA/1.0",
        )
        # Prime.
        assert await auth_store.get_session_cached(s.token) is not None
        # Delete — cache should be dropped.
        await auth_store.delete_session(s.token)
        # Next cached call re-reads the session file and gets None
        # (the negative cache then stores it for a short window).
        assert await auth_store.get_session_cached(s.token) is None

    asyncio.run(_run())


def test_touch_session_invalidates_cache():
    """After ``touch_session`` updates ``expires_at``, the cached
    value is stale. A fresh cached read must pick up the new value.
    """
    async def _run():
        u = await auth_store.create_user("alice", "supersecret")
        s = await auth_store.create_session(
            user_id=u.id, ip="1.1.1.1", user_agent="UA/1.0",
        )
        first = await auth_store.get_session_cached(s.token)
        assert first is not None
        first_exp = first.expires_at
        # Touch moves the expiry forward.
        await auth_store.touch_session(s.token)
        second = await auth_store.get_session_cached(s.token)
        assert second is not None
        # The cache was invalidated, so we read the new expires_at.
        assert second.expires_at != first_exp

    asyncio.run(_run())


def test_user_update_invalidates_cache():
    async def _run():
        u = await auth_store.create_user("alice", "supersecret", role=Role.USER)
        cached = await auth_store.get_user_cached(u.id)
        assert cached is not None and cached.role == Role.USER

        await auth_store.update_user(u.id, role=Role.ADMIN)
        fresh = await auth_store.get_user_cached(u.id)
        assert fresh is not None and fresh.role == Role.ADMIN

    asyncio.run(_run())


def test_negative_lookup_is_cached():
    """An unknown token is cached as None so a brute-force token scan
    doesn't hammer the disk on every attempt."""
    async def _run():
        assert await auth_store.get_session_cached("nope-not-a-real-token") is None
        miss_before = auth_store._cache_stats["session_misses"]
        assert await auth_store.get_session_cached("nope-not-a-real-token") is None
        miss_after = auth_store._cache_stats["session_misses"]
        # Second call hit the negative cache; no extra miss.
        assert miss_after == miss_before
        assert auth_store._cache_stats["session_negative_hits"] >= 1

    asyncio.run(_run())


def test_cache_stats_reports_counts():
    async def _run():
        u = await auth_store.create_user("alice", "supersecret")
        s = await auth_store.create_session(
            user_id=u.id, ip="1.1.1.1", user_agent="UA/1.0",
        )
        # Prime then hit.
        await auth_store.get_session_cached(s.token)
        await auth_store.get_session_cached(s.token)
        await auth_store.get_user_cached(u.id)
        stats = auth_store.get_cache_stats()
        assert "session_hits" in stats
        assert "session_misses" in stats
        assert "user_hits" in stats
        assert "user_misses" in stats
        assert stats["session_cache_size"] >= 1
        assert stats["user_cache_size"] >= 1

    asyncio.run(_run())
