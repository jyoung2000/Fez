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


async def require_job_access(job_id: str, user: User):
    """Load a job and verify the caller owns it.

    Every user — including admins — only sees their own jobs.  Matching
    is by ``owner_user_id`` (primary) or ``owner_username`` (fallback).
    Legacy pre-auth jobs are adopted by the head admin on first access.
    """
    from backend import database

    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    owner = getattr(job, "owner_user_id", "") or ""
    owner_name = (getattr(job, "owner_username", "") or "").strip().lower()
    caller_name = (user.username or "").strip().lower()

    if owner and owner == user.id:
        return job
    if owner_name and caller_name and owner_name == caller_name:
        return job

    if not owner and getattr(user, "head_admin", False):
        job.owner_user_id = user.id
        job.owner_username = user.username or ""
        await database.save_job(job)
        return job

    raise HTTPException(status_code=404, detail="Job not found")
