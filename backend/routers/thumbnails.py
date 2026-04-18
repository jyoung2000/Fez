"""Public thumbnail endpoint for rich preview crawlers.

Unauthenticated by design — Slackbot, Twitterbot, etc. do not have sessions.
Cache headers are aggressive because crawlers re-check images frequently.
"""

import os
from datetime import datetime, timezone
from email.utils import format_datetime, parsedate_to_datetime
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse, Response

from backend.services.thumbnail_extractor import get_thumbnail_dir, get_clip_thumbnail_path

router = APIRouter()


@router.get("/thumbnails/{job_id}.jpg")
async def get_thumbnail(
    job_id: str,
    if_modified_since: str = Header(None, alias="If-Modified-Since"),
):
    """Serve a job's thumbnail as image/jpeg with public caching.

    Falls back to a default placeholder if the job's thumbnail is missing.
    """
    # Sanitize job_id — only allow safe filename characters
    if not job_id or not all(c.isalnum() or c in "-_" for c in job_id):
        raise HTTPException(status_code=400, detail="invalid job_id")

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
        "Cache-Control": "public, max-age=2592000",  # 30 days
        "Last-Modified": format_datetime(mtime, usegmt=True),
        "Access-Control-Allow-Origin": "*",
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
    if_modified_since: str = Header(None, alias="If-Modified-Since"),
):
    """Serve a per-clip thumbnail as image/jpeg with public caching.

    Falls back to the job-level thumbnail if no per-clip file exists,
    then to the default placeholder.
    """
    if not job_id or not all(c.isalnum() or c in "-_" for c in job_id):
        raise HTTPException(status_code=400, detail="invalid job_id")

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
        "Cache-Control": "public, max-age=2592000",
        "Last-Modified": format_datetime(mtime, usegmt=True),
        "Access-Control-Allow-Origin": "*",
        "X-Content-Type-Options": "nosniff",
    }
    return FileResponse(
        path=str(thumb_path),
        media_type="image/jpeg",
        headers=headers,
    )
