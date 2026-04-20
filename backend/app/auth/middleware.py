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
from backend.app.auth.store import get_session, get_user, touch_session

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

        session = await get_session(token)
        if session is None:
            return self._clear_and_reject("session not found")

        # Fingerprint check: new IP or browser → force re-login.
        ip = _client_ip(request)
        ua = request.headers.get("user-agent", "")
        if compute_fingerprint(ip, ua) != session.fingerprint:
            # Kill the session so re-use of the old cookie can't succeed.
            from backend.app.auth.store import delete_session
            await delete_session(token)
            return self._clear_and_reject("session bound to a different browser/IP")

        user = await get_user(session.user_id)
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


def set_session_cookie(
    response: Response,
    token: str,
    *,
    max_age_seconds: int = 30 * 24 * 3600,
    remember: bool = True,
) -> None:
    """Attach the session cookie with safe defaults.

    ``remember=True`` (default) writes a persistent cookie with
    ``max_age``: the browser keeps it across restarts and the user
    stays signed in for up to ``max_age_seconds``.

    ``remember=False`` writes a SESSION cookie (no ``Max-Age`` /
    ``Expires``): browsers drop it on quit, so the next time the
    user opens ClipAI on that device they have to sign in again.
    The server-side session record is unchanged either way; the
    auto-extending touch loop on the middleware still rolls the
    expiry forward, but no cookie persists past the browser tab.
    """
    cookie_kwargs = dict(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,
        secure=False,  # localhost defaults; reverse proxy can override
        samesite="lax",
        path="/",
    )
    if remember:
        cookie_kwargs["max_age"] = max_age_seconds
    response.set_cookie(**cookie_kwargs)


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")
