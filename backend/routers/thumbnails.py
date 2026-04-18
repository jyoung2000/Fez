"""Thumbnail endpoints.

Historically unauthenticated for Slackbot / Twitterbot unfurl, but a
thumbnail IS an image of the user's video — leaking it to another
account violates cross-account privacy. Every request now requires
the caller to own the job. Social unfurl can be enabled per job via
the signed share-link mechanism in ``backend/routers/share.py``.
"""

import os
from datetime import datetime, timezone
from email.utils import format_datetime, parsedate_to_datetime
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import FileResponse, Response

from backend import database
from backend.app.auth.middleware import SESSION_COOKIE
from backend.app.auth.store import get_session, get_user
from backend.services.thumbnail_extractor import get_thumbnail_dir, get_clip_thumbnail_path

router = APIRouter()


async def _owns_job(request: Request, job_id: str) -> bool:
    """Resolve the session cookie manually and verify job ownership.

    ``/thumbnails/*`` lives outside ``/api/``, so the auth middleware
    treats it as a public SPA path and skips it. We auth here instead.
    """
    token = request.cookies.get(SESSION_COOKIE, "")
    if not token:
        return False
    session = await get_session(token)
    if session is None:
        return False
    user = await get_user(session.user_id)
    if user is None or not user.active:
        return False
    job = await database.load_job(job_id)
    if not job:
        return False
    owner = (getattr(job, "owner_user_id", "") or "")
    owner_name = (getattr(job, "owner_username", "") or "").strip().lower()
    caller_name = (user.username or "").strip().lower()
    if owner and owner == user.id:
        return True
    if owner_name and caller_name and owner_name == caller_name:
        return True
    if not owner and getattr(user, "head_admin", False):
        return True
    return False


@router.get("/thumbnails/{job_id}.jpg")
async def get_thumbnail(
    job_id: str,
    request: Request,
    if_modified_since: str = Header(None, alias="If-Modified-Since"),
):
    """Serve a job's thumbnail as image/jpeg. Requires job ownership."""
    # Sanitize job_id — only allow safe filename characters
    if not job_id or not all(c.isalnum() or c in "-_" for c in job_id):
        raise HTTPException(status_code=400, detail="invalid job_id")

    if not await _owns_job(request, job_id):
        raise HTTPException(status_code=404, detail="thumbnail not found")

    thumb_path = get_thumbnail_dir() / f"{job_id}.jpg"

    if not thumb_path.exists():
        # Fall back to a default placeholder shipped with the app
        default = Path(__file__).parent.parent / "static" / "default_thumbnail.jpg"
        if default.exists():
            thumb_path = default
        else:
            raise HTTPException(status_code=404, detail="thumbnail not found")

    # Conditional GET handling
    mtime = datetime.fromtimestamp(thumb_path.stat().st_mtime, tz=timezone.utc)
    if if_modified_since:
        try:
            client_mtime = parsedate_to_datetime(if_modified_since)
            if client_mtime >= mtime:
                return Response(status_code=304)
        except Exception:
            pass

    headers = {
        "Cache-Control": "private, max-age=86400",  # 1 day, user-scoped
        "Last-Modified": format_datetime(mtime, usegmt=True),
        "X-Content-Type-Options": "nosniff",
    }
    return FileResponse(
        path=str(thumb_path),
        media_type="image/jpeg",
        headers=headers,
    )


@router.get("/thumbnails/{job_id}/{clip_id}.jpg")
async def get_clip_thumbnail(
    job_id: str,
    clip_id: int,
    request: Request,
    if_modified_since: str = Header(None, alias="If-Modified-Since"),
):
    """Serve a per-clip thumbnail as image/jpeg. Requires job ownership."""
    if not job_id or not all(c.isalnum() or c in "-_" for c in job_id):
        raise HTTPException(status_code=400, detail="invalid job_id")

    if not await _owns_job(request, job_id):
        raise HTTPException(status_code=404, detail="thumbnail not found")

    # Try per-clip thumbnail first
    thumb_path = get_clip_thumbnail_path(job_id, clip_id)

    if not thumb_path.exists():
        # Fall back to job-level thumbnail
        thumb_path = get_thumbnail_dir() / f"{job_id}.jpg"

    if not thumb_path.exists():
        # Fall back to default placeholder
        default = Path(__file__).parent.parent / "static" / "default_thumbnail.jpg"
        if default.exists():
            thumb_path = default
        else:
            raise HTTPException(status_code=404, detail="thumbnail not found")

    # Conditional GET handling
    mtime = datetime.fromtimestamp(thumb_path.stat().st_mtime, tz=timezone.utc)
    if if_modified_since:
        try:
            client_mtime = parsedate_to_datetime(if_modified_since)
            if client_mtime >= mtime:
                return Response(status_code=304)
        except Exception:
            pass

    headers = {
        "Cache-Control": "private, max-age=86400",
        "Last-Modified": format_datetime(mtime, usegmt=True),
        "X-Content-Type-Options": "nosniff",
    }
    return FileResponse(
        path=str(thumb_path),
        media_type="image/jpeg",
        headers=headers,
    )
