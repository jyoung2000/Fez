"""AuthMiddleware — resolves the session cookie before routers run.

Lifecycle per request:

  1. Check request path against allow-lists (static / public / auth
     endpoints themselves) — these short-circuit before any lookup.
  2. Read the ``clipai_session`` cookie.
  3. Look up the session; verify the IP+UA fingerprint hasn't changed.
     A changed fingerprint means the user is coming from a new IP or
     browser — force re-login by responding 401. The frontend catches
     this and redirects to ``/login``.
  4. Look up the user; skip the request when the user is deactivated.
  5. Attach ``request.state.user`` + ``request.state.session`` and
     continue the ASGI chain.
  6. Periodically ``touch_session`` so long-running use doesn't expire.

Endpoints that should be reachable without auth (e.g. ``/api/auth/login``
or the SPA's index page) bypass the middleware via
``_is_public_path``.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Iterable

from fastapi import Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.app.auth.security import compute_fingerprint
from backend.app.auth.store import (
    get_session_cached,
    get_user_cached,
    touch_session,
)

logger = logging.getLogger(__name__)


SESSION_COOKIE = "clipai_session"


# Paths that never require auth. SPA routes fall through to index.html
# which is public; the frontend then mounts AuthProvider and fetches
# /api/auth/me — a 401 there triggers the redirect to /login.
#
# Note: /api/auth/me is NOT in this list. It MUST flow through this
# middleware so the router handler can read ``request.state.user``.
# Unauthenticated callers still receive a 401 from the middleware —
# same signal the frontend needs to redirect to /login.
_PUBLIC_EXACT = {
    "/",
    "/login",
    "/api/auth/login",
    "/api/auth/bootstrap",
    "/health",
    "/healthz",
}

# Paths that are public only on GET. Site-config carries branding
# (title / favicon / logo URLs) that the SPA needs to render the
# /login page itself, before the user can possibly be authenticated.
# Mutating verbs on the same path stay protected — they're handled
# below in ``dispatch`` and re-enter the auth flow.
_PUBLIC_GET_EXACT = {
    "/api/site-config",
    # Read-only deploy-verification probe for the 2026 SOTA reframing
    # validation. Returns paths + boolean flags, no secrets, no
    # subprocess on the success path. Public so the verify script
    # (run from the Unraid host outside the browser session) can
    # confirm the image was built correctly. Mutating bench endpoints
    # stay auth-protected.
    "/api/diagnostics/sota-bench-status",
}

_PUBLIC_PREFIXES = (
    "/static/",
    "/assets/",
    "/favicon",
    "/api/site-uploads/",
    "/api/auth/login",   # handle query params
    "/api/auth/logout",
    # Signed share-link API: the token IS the credential. No session
    # cookie required so recipients can view the linked content
    # without needing an account. Issuance / revocation lives at
    # /api/share/links/* and DOES require auth (handled per-route).
    "/api/share/public/",
    # OG unfurl / preview pages live under /share/...
    "/share/",
    # SPA is a single index.html; any UI route is public — frontend
    # enforces the redirect.
)


_SPA_ROUTE_RE = re.compile(
    r"^/(login|signin|signup|upload|clips|media|logs|settings|analysis|seo|share)(/|$)"
)


def _is_public_path(path: str, method: str = "GET") -> bool:
    if path in _PUBLIC_EXACT:
        return True
    if method == "GET" and path in _PUBLIC_GET_EXACT:
        return True
    for p in _PUBLIC_PREFIXES:
        if path.startswith(p):
            return True
    if _SPA_ROUTE_RE.match(path):
        return True
    # Anything not under /api/ is a SPA route (index.html).
    if not path.startswith("/api/"):
        return True
    return False


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real = request.headers.get("x-real-ip")
    if real:
        return real.strip()
    return request.client.host if request.client else ""


class AuthMiddleware(BaseHTTPMiddleware):
    """Attach the authenticated user to ``request.state`` or respond 401."""

    # In-memory throttle: we only hit touch_session once per session
    # per ``_TOUCH_INTERVAL`` seconds of real time so we don't re-write
    # the sessions file on every single API call.
    _TOUCH_INTERVAL = 60.0

    def __init__(self, app, touch_cache: dict | None = None):
        super().__init__(app)
        self._touch_cache: dict = touch_cache or {}

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if _is_public_path(path, request.method):
            return await call_next(request)

        # Signed-URL fast path: GET /api/files/{job_id}/{path}?exp=&sig=
        # lets HTML5 <video> / <img> elements load media without the
        # session cookie. The signature itself is the credential; the
        # handler re-verifies it before serving bytes. Any other
        # method, or a request missing either query param, falls
        # through to the normal cookie path.
        if (
            request.method == "GET"
            and path.startswith("/api/files/")
            and request.query_params.get("exp")
            and request.query_params.get("sig")
        ):
            return await call_next(request)

        token = request.cookies.get(SESSION_COOKIE)
        if not token:
            return JSONResponse(
                {"detail": "not authenticated"}, status_code=401,
            )

        session = await get_session_cached(token)
        if session is None:
            return self._clear_and_reject("session not found")

        # Fingerprint check (UA-only in V2). On mismatch we return
        # 401 + clear the cookie on THIS response, but we do NOT
        # delete the server-side session record — behind a reverse
        # proxy the mismatch is often a transient header flake, and
        # killing the session mid-upload / mid-playback is
        # catastrophic. Re-login on the same browser restores access.
        ip = _client_ip(request)
        ua = request.headers.get("user-agent", "")
        current_fp = compute_fingerprint(ip, ua)
        if current_fp != session.fingerprint:
            # One-time migration: legacy sessions were stored with a
            # V1 fingerprint (IP + UA). Accept the session once on
            # the UA-only check and rewrite the stored fingerprint
            # so subsequent requests match cleanly. Guarded by the
            # fp_v2 flag on the session record.
            if not _session_is_v2(session):
                try:
                    await _migrate_session_to_v2(token, current_fp)
                except Exception as e:
                    logger.warning("fingerprint V2 migration failed: %s", e)
                # Continue as if the fingerprint matched — this was
                # a known-good cookie from the V1 scheme.
            else:
                return self._reject_fingerprint()

        user = await get_user_cached(session.user_id)
        if user is None or not user.active:
            return self._clear_and_reject("user not found or deactivated")

        # Attach to request.state for downstream deps.
        request.state.user = user
        request.state.session = session

        # Throttled touch_session — extends both the server-side
        # ``expires_at`` AND (via ``response.set_cookie`` below) the
        # browser-side cookie max-age, so an active user never has to
        # sign in again as long as they hit the app at least once
        # within the rolling window.
        now = time.monotonic()
        last = self._touch_cache.get(token, 0.0)
        refreshed_cookie = False
        if now - last > self._TOUCH_INTERVAL:
            try:
                await touch_session(token)
                self._touch_cache[token] = now
                refreshed_cookie = True
            except Exception:
                pass

        response = await call_next(request)
        if refreshed_cookie:
            try:
                # Honor the per-session ``remember`` flag — never
                # accidentally promote a session-scoped cookie to a
                # persistent one when the user explicitly opted out
                # of "Remember me" at login.
                set_session_cookie(
                    response, token,
                    remember=getattr(session, "remember", True),
                )
            except Exception:
                pass
        return response

    def _clear_and_reject(self, detail: str) -> Response:
        resp = JSONResponse({"detail": detail}, status_code=401)
        resp.delete_cookie(SESSION_COOKIE, path="/")
        return resp

    def _reject_fingerprint(self) -> Response:
        """Reject a fingerprint-mismatched request.

        Returns 401 + clears the cookie on the response, but leaves
        the server-side session record intact so the user can
        re-login on the same browser without requiring a password
        reset. The detail string is machine-checkable so the
        frontend can display a targeted prompt.
        """
        return self._clear_and_reject("fingerprint_mismatch")


_FINGERPRINT_V2_FLAG = "fp_v2"


def _session_is_v2(session) -> bool:
    """True if this session was stored with the V2 (UA-only)
    fingerprint. Legacy sessions lack the flag.
    """
    # Session is a frozen dataclass; check the raw storage too for
    # forward-compat.
    if getattr(session, _FINGERPRINT_V2_FLAG, False):
        return True
    return False


async def _migrate_session_to_v2(token: str, new_fingerprint: str) -> None:
    """Rewrite an existing session's fingerprint to the UA-only V2
    form and flip the ``fp_v2`` flag so the migration runs at most
    once per session.
    """
    from backend.app.auth.store import (
        SESSIONS_PATH,
        _atomic_write_json,
        _read_json,
        _sessions_lock,
        invalidate_session_cache,
    )
    async with _sessions_lock:
        data = await _read_json(SESSIONS_PATH, {"sessions": []})
        updated = False
        for s in data.get("sessions", []):
            if s.get("token") == token:
                s["fingerprint"] = new_fingerprint
                s[_FINGERPRINT_V2_FLAG] = True
                updated = True
                break
        if updated:
            await _atomic_write_json(SESSIONS_PATH, data)
    invalidate_session_cache(token)


# Minimum cookie lifetime after a successful sign-in. The cookie is
# always written with at least this much ``Max-Age`` so a user who
# signs in stays signed in for at least 25 hours regardless of the
# ``remember`` choice. The middleware's rolling touch loop refreshes
# the cookie on every API call (throttled to once per minute), so
# active users effectively never have to sign in again — the floor
# only kicks in for browsers that close mid-session and reopen later.
_MIN_SESSION_LIFETIME_SEC = 25 * 3600


def set_session_cookie(
    response: Response,
    token: str,
    *,
    max_age_seconds: int = 30 * 24 * 3600,
    remember: bool = True,
) -> None:
    """Attach the session cookie with safe defaults.

    The cookie is ALWAYS written as persistent with a ``Max-Age``
    floor of ``_MIN_SESSION_LIFETIME_SEC`` (25 h), regardless of the
    ``remember`` flag. This guarantees that a user who successfully
    signs in is not prompted to sign in again for at least 25 hours,
    even if they close the browser between visits.

    ``remember=True`` (default): cookie persists for ``max_age_seconds``
    (30 days by default). The middleware's rolling touch refreshes
    this on every authenticated request, so an active user never
    times out.

    ``remember=False``: cookie persists for exactly 25 h. The user
    has explicitly asked for a less-persistent session, but the
    25 h floor is the minimum useful session length on this app —
    we do not write session-scoped cookies that vanish on
    browser quit, because those produced the most-reported "I just
    signed in, why am I being asked to sign in again" bug.

    The server-side session record itself lives 30 days regardless;
    the cookie ``Max-Age`` and the server-side ``expires_at`` are
    independent ceilings.
    """
    if remember:
        effective_max_age = max(int(max_age_seconds), _MIN_SESSION_LIFETIME_SEC)
    else:
        effective_max_age = _MIN_SESSION_LIFETIME_SEC
    cookie_kwargs = dict(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,
        secure=False,  # localhost defaults; reverse proxy can override
        samesite="lax",
        path="/",
        max_age=effective_max_age,
    )
    response.set_cookie(**cookie_kwargs)


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")
