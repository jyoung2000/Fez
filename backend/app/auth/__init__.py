"""User authentication and session management.

Replaces the shared single-API-key model with per-user login. Three
layers:

  * :mod:`backend.app.auth.models` — ``User`` / ``Session`` / ``Role``
    dataclasses.
  * :mod:`backend.app.auth.store` — file-backed JSON stores for users
    and sessions, with atomic writes + async locks.
  * :mod:`backend.app.auth.security` — pbkdf2 password hashing (stdlib,
    no new dependency) + IP/UA fingerprint computation.
  * :mod:`backend.app.auth.deps` — FastAPI dependencies:
    ``get_current_user`` and ``require_admin``.
  * :mod:`backend.app.auth.middleware` — ``AuthMiddleware`` that
    resolves the session cookie before any router dependency runs and
    enforces IP/User-Agent binding so a new IP or browser forces a
    re-login.
  * :mod:`backend.app.auth.seed` — ensures the head admin account
    (``Jadmin``) exists on every startup.
"""

from backend.app.auth.deps import get_current_user, require_admin  # noqa: F401
from backend.app.auth.models import Role, Session, User  # noqa: F401
