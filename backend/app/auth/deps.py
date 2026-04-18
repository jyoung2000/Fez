"""FastAPI dependencies for auth-protected endpoints."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request

from backend.app.auth.models import Role, User


def _user_from_request(request: Request) -> User:
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    return user


async def get_current_user(request: Request) -> User:
    """Return the authenticated user for this request.

    Populated by :class:`backend.app.auth.middleware.AuthMiddleware`
    before router handlers execute. When absent, we respond 401; the
    frontend catches that and redirects to ``/login``.
    """
    return _user_from_request(request)


async def require_admin(request: Request) -> User:
    user = _user_from_request(request)
    if user.role != Role.ADMIN:
        raise HTTPException(status_code=403, detail="admin only")
    return user
