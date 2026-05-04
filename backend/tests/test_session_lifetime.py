"""Tests for the 25-hour session-cookie minimum contract.

The user-facing rule: once a user signs in successfully, the browser
keeps them signed in for at least 25 hours regardless of the
``remember`` flag, regardless of whether the browser was closed in
between visits.

The mechanism: ``set_session_cookie`` always writes a persistent
cookie (``Max-Age`` >= 25 h) — the previous behavior of writing a
session-scoped cookie when ``remember=False`` is gone, because it
caused the most-reported "I just signed in, why am I being asked to
sign in again" symptom.

Active users stay signed in indefinitely via the middleware's
rolling-touch refresh; the 25 h floor only matters for users who
close the browser between visits.
"""
from __future__ import annotations

from starlette.responses import Response

from backend.app.auth.middleware import (
    _MIN_SESSION_LIFETIME_SEC,
    set_session_cookie,
)


def _max_age_from_response(resp: Response) -> int | None:
    """Pull the Max-Age from a Set-Cookie header on the response.

    Returns None if no Max-Age is set (i.e., session-scoped cookie).
    """
    set_cookie = resp.headers.get("set-cookie")
    assert set_cookie, "no Set-Cookie header"
    for part in set_cookie.split(";"):
        part = part.strip()
        if part.lower().startswith("max-age="):
            return int(part.split("=", 1)[1])
    return None


# ──────────────────────────────────────────────────────────────────────────
# The 25-hour floor
# ──────────────────────────────────────────────────────────────────────────

def test_min_lifetime_constant_is_at_least_25_hours():
    """The constant the contract is built on. If somebody lowers this,
    the rest of the test suite enforces the user-facing promise via
    the cookie max-age."""
    assert _MIN_SESSION_LIFETIME_SEC >= 25 * 3600


def test_remember_true_writes_persistent_cookie_at_least_25h():
    resp = Response()
    set_session_cookie(resp, token="t-1", remember=True)
    max_age = _max_age_from_response(resp)
    assert max_age is not None, (
        "remember=True must produce a persistent cookie"
    )
    assert max_age >= 25 * 3600


def test_remember_false_now_writes_persistent_cookie_at_least_25h():
    """Previously a session-scoped cookie (no Max-Age, drops on browser
    quit). Updated contract: always at least 25 h so users never have
    to re-sign-in for >= 25 h after a successful login."""
    resp = Response()
    set_session_cookie(resp, token="t-2", remember=False)
    max_age = _max_age_from_response(resp)
    assert max_age is not None, (
        "remember=False must NOT produce a session-scoped cookie under "
        "the 25 h floor — users were being prompted to sign in on every "
        "browser restart with the old behavior."
    )
    assert max_age >= 25 * 3600


def test_remember_true_keeps_30_day_default():
    """When remember=True the default 30 d ceiling is preserved
    (longer is allowed, so the floor only floors)."""
    resp = Response()
    set_session_cookie(resp, token="t-3", remember=True)
    max_age = _max_age_from_response(resp)
    assert max_age == 30 * 24 * 3600


def test_remember_false_uses_exactly_25h():
    """When remember=False the cookie is the 25 h floor exactly —
    less persistent than remember=True but still meets the
    user-facing promise."""
    resp = Response()
    set_session_cookie(resp, token="t-4", remember=False)
    max_age = _max_age_from_response(resp)
    assert max_age == _MIN_SESSION_LIFETIME_SEC


def test_explicit_max_age_below_floor_is_floored_when_remember_true():
    """If a caller passes max_age_seconds smaller than the floor,
    the floor wins. Keeps the user-facing promise even when the
    caller is misconfigured."""
    resp = Response()
    set_session_cookie(
        resp, token="t-5", max_age_seconds=3600, remember=True,
    )
    max_age = _max_age_from_response(resp)
    assert max_age >= 25 * 3600


def test_explicit_max_age_above_floor_is_honored_when_remember_true():
    resp = Response()
    set_session_cookie(
        resp, token="t-6",
        max_age_seconds=7 * 24 * 3600,  # 7 days
        remember=True,
    )
    max_age = _max_age_from_response(resp)
    assert max_age == 7 * 24 * 3600


# ──────────────────────────────────────────────────────────────────────────
# Cookie-flag invariants
# ──────────────────────────────────────────────────────────────────────────

def test_cookie_is_httponly():
    resp = Response()
    set_session_cookie(resp, token="t-7", remember=True)
    set_cookie = resp.headers.get("set-cookie", "")
    assert "httponly" in set_cookie.lower()


def test_cookie_path_is_root():
    resp = Response()
    set_session_cookie(resp, token="t-8", remember=True)
    set_cookie = resp.headers.get("set-cookie", "")
    assert "path=/" in set_cookie.lower()


def test_cookie_samesite_is_lax():
    resp = Response()
    set_session_cookie(resp, token="t-9", remember=True)
    set_cookie = resp.headers.get("set-cookie", "")
    assert "samesite=lax" in set_cookie.lower()
