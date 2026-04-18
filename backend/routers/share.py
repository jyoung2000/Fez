"""Bulletproof share link routes + signed share-link API.

Two layers of "share":

1. **OG unfurl pages** under ``/share/...`` — server-rendered HTML
   for crawlers (Slack, Twitter, Facebook). These don't grant API
   access; they just produce nice link previews. Existed before the
   multi-user auth landed.

2. **Signed share links** under ``/api/share/...`` — opaque tokens
   issued by the job's owner that grant read-only API access to ONE
   job (or ONE clip) without requiring the recipient to log in.
   The middleware allow-lists ``/api/share/public/*`` so the token is
   the only credential the recipient needs.

Endpoints:

  Owner / admin only (auth required):
    POST   /api/share/links                     create a link
    GET    /api/share/links?job_id=...          list a user's links
    DELETE /api/share/links/{token}             revoke

  Public (no auth):
    GET    /api/share/public/{token}            link info + scope
    GET    /api/share/public/{token}/job        full job data (scoped)
    GET    /api/share/public/{token}/clip       single clip data (scoped)
    GET    /api/share/public/{token}/render_plan
    GET    /api/share/public/{token}/transcript
"""

import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, Response

from backend.services.range_stream import parse_range_header, stream_file_range
from pydantic import BaseModel, Field

from backend.middleware.og_injection import (
    render_og_html,
    _get_base_url_from_request,
    _resolve_clip_metadata,
)
from backend import database
from backend.app.auth import share_store
from backend.app.auth.deps import get_current_user
from backend.app.auth.models import Role, User

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/share/analysis/{job_id}", response_class=HTMLResponse)
async def share_analysis(job_id: str, request: Request):
    """Server-rendered share page for analysis results.

    Always returns HTML with OG tags regardless of user agent.
    Includes meta refresh to redirect human visitors to the SPA.
    """
    base_url = _get_base_url_from_request(request)
    if not base_url:
        raise HTTPException(status_code=500, detail="Cannot determine base URL")

    job = await database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    title = (getattr(job, 'filename', None)
             or f"ClipAI Analysis {job_id[:8]}")
    summary = getattr(job, 'summary', None)
    if summary and hasattr(summary, 'overview'):
        description = summary.overview
    else:
        description = "AI-powered video analysis"
    image_url = f"{base_url}/thumbnails/{job_id}.jpg"
    canonical_url = f"{base_url}/analysis/{job_id}"

    html = render_og_html(
        title=title,
        description=description,
        image_url=image_url,
        canonical_url=canonical_url,
        og_type="video.other",
    )
    return HTMLResponse(content=html)


@router.get("/share/clip/{job_id}/{clip_id}", response_class=HTMLResponse)
async def share_clip(job_id: str, clip_id: int, request: Request):
    """Server-rendered share page for a specific clip.

    Always returns HTML with OG tags regardless of user agent.
    Includes meta refresh to redirect human visitors to the SPA.
    """
    base_url = _get_base_url_from_request(request)
    if not base_url:
        raise HTTPException(status_code=500, detail="Cannot determine base URL")

    job = await database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    # Find the clip
    clip = None
    for c in getattr(job, 'clips', []):
        if getattr(c, 'id', None) == clip_id:
            clip = c
            break

    meta = _resolve_clip_metadata(clip, job, job_id, clip_id, base_url)

    html = render_og_html(
        title=meta["title"],
        description=meta["description"],
        image_url=meta["image_url"],
        canonical_url=meta["canonical_url"],
        og_type="video.other",
        video_url=meta["video_url"],
        image_width=meta["image_width"],
        image_height=meta["image_height"],
        video_width=meta["video_width"],
        video_height=meta["video_height"],
    )
    return HTMLResponse(content=html)


# ── Signed share-link API ──────────────────────────────────────


class CreateLinkRequest(BaseModel):
    job_id: str
    scope: str = Field(default="job", pattern="^(job|clip)$")
    clip_id: Optional[int] = None
    ttl_days: int = Field(default=365, ge=1, le=3650)
    note: str = ""


def _can_share_job(job, user) -> bool:
    """Owner or admin only."""
    if user.role == Role.ADMIN:
        return True
    owner = getattr(job, "owner_user_id", "") or ""
    return bool(owner) and owner == user.id


@router.post("/api/share/links")
async def create_share_link(
    payload: CreateLinkRequest,
    user: User = Depends(get_current_user),
):
    job = await database.load_job(payload.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not _can_share_job(job, user):
        raise HTTPException(status_code=403, detail="not allowed")
    if payload.scope == "clip":
        # Verify the clip exists.
        clip_ids = {getattr(c, "id", None) for c in (getattr(job, "clips", None) or [])}
        if payload.clip_id is None or payload.clip_id not in clip_ids:
            raise HTTPException(status_code=404, detail="clip not found in job")
    link = await share_store.create_link(
        job_id=payload.job_id,
        scope=payload.scope,
        clip_id=payload.clip_id,
        created_by=user.id,
        ttl_days=payload.ttl_days,
        note=payload.note,
    )
    return {"link": link.to_public()}


@router.get("/api/share/links")
async def list_share_links(
    job_id: Optional[str] = Query(default=None),
    user: User = Depends(get_current_user),
):
    # Admins see all; regular users only their own.
    owner_id = None if user.role == Role.ADMIN else user.id
    links = await share_store.list_links(owner_id=owner_id, job_id=job_id)
    return {"links": [l.to_public() for l in links]}


@router.delete("/api/share/links/{token}")
async def revoke_share_link(
    token: str,
    user: User = Depends(get_current_user),
):
    link = await share_store.get_link(token)
    if link is None:
        raise HTTPException(status_code=404, detail="link not found")
    if user.role != Role.ADMIN and link.created_by != user.id:
        raise HTTPException(status_code=403, detail="not allowed")
    await share_store.delete_link(token)
    return {"ok": True}


# ── Public read endpoints (NO auth required) ─────────────────


def _serialize_clip(clip) -> dict:
    if hasattr(clip, "model_dump"):
        return clip.model_dump(mode="json")
    if isinstance(clip, dict):
        return dict(clip)
    return {}


def _scope_clip(job_dict: dict, clip_id: int) -> dict:
    """Return a clip's slice of a job dict with everything the frontend
    VideoEditor needs to render an interactive preview of that clip.

    Previously this stripped scenes / transcript / subject_track so the
    recipient only saw static metadata. Now the share view embeds the
    same ``<VideoEditor>`` the SEO page uses, so it needs the same
    underlying data. Fields are still scoped to the clip's time window
    where it matters — transcript is trimmed, scenes / scene cuts /
    subject_track / layout_timeline are passed through whole because
    the editor handles time-windowing internally and shipping the full
    arrays is cheaper than re-slicing them here.
    """
    clips = job_dict.get("clips") or []
    chosen = next((c for c in clips if c.get("id") == clip_id), None)
    if chosen is None:
        return {}

    # Trim transcript to the clip window (± a small pad so boundary
    # segments still appear). The editor will re-clip internally, but
    # shipping only relevant segments keeps the public payload small.
    start = float(chosen.get("start_time") or 0.0)
    end = float(chosen.get("end_time") or 0.0)
    pad = 0.5
    full_transcript = job_dict.get("transcript") or []
    clip_transcript = [
        seg for seg in full_transcript
        if isinstance(seg, dict)
        and float(seg.get("end", 0)) >= (start - pad)
        and float(seg.get("start", 0)) <= (end + pad)
    ]

    return {
        "job_id": job_dict.get("job_id"),
        "filename": job_dict.get("filename"),
        "duration": job_dict.get("duration"),
        "fps": job_dict.get("fps"),
        "resolution": job_dict.get("resolution"),
        "speaker_names": job_dict.get("speaker_names", {}),
        "subtitle_settings": job_dict.get("subtitle_settings"),
        "clip": chosen,
        # Editor-required context — same fields ClipSEO feeds to
        # ``<VideoEditor>``. The recipient's preview now matches the
        # owner's SEO page instead of a static read-only card.
        "transcript": clip_transcript,
        "scenes": job_dict.get("scenes") or [],
        "scene_cut_timestamps": job_dict.get("scene_cut_timestamps") or [],
        "subject_track": job_dict.get("subject_track") or [],
        "layout_timeline": job_dict.get("layout_timeline") or [],
        "default_layout_mode": job_dict.get("default_layout_mode") or "single",
        "tracking_mode": job_dict.get("tracking_mode") or "",
        "content_type_override": job_dict.get("content_type_override") or "",
    }


def _scope_full_job(job_dict: dict) -> dict:
    """Return the full job dict with admin/owner-only fields stripped."""
    safe = dict(job_dict)
    # Drop fields that look internal / sensitive.
    for k in (
        "owner_user_id",
        "estimated_cost_usd",
        "provider_used",
        "face_registry_data",
    ):
        safe.pop(k, None)
    return safe


async def _resolve_link_or_404(token: str) -> share_store.ShareLink:
    link = await share_store.get_link(token)
    if link is None:
        raise HTTPException(status_code=404, detail="share link not found or expired")
    return link


@router.get("/api/share/public/{token}")
async def public_share_info(token: str):
    """Lightweight metadata so the frontend knows which view to render."""
    link = await _resolve_link_or_404(token)
    job = await database.load_job(link.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "scope": link.scope,
        "job_id": link.job_id,
        "clip_id": link.clip_id,
        "expires_at": link.expires_at,
        "filename": getattr(job, "filename", ""),
        "duration": getattr(job, "duration", 0),
        "resolution": getattr(job, "resolution", ""),
    }


@router.get("/api/share/public/{token}/job")
async def public_share_job(token: str):
    link = await _resolve_link_or_404(token)
    job = await database.load_job(link.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    job_dict = job.model_dump(mode="json")
    if link.scope == "clip":
        return _scope_clip(job_dict, link.clip_id)
    return _scope_full_job(job_dict)


@router.get("/api/share/public/{token}/clip")
async def public_share_clip(token: str):
    """Same payload as /job but always scoped to the link's clip. 404
    when the link is a full-job link without a clip_id."""
    link = await _resolve_link_or_404(token)
    if link.scope != "clip" or link.clip_id is None:
        raise HTTPException(status_code=400, detail="link is not clip-scoped")
    job = await database.load_job(link.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _scope_clip(job.model_dump(mode="json"), link.clip_id)


@router.get("/api/share/public/{token}/render_plan")
async def public_share_render_plan(token: str):
    """Expose the cached render_plan for previewing in the share view."""
    link = await _resolve_link_or_404(token)
    job = await database.get_job(link.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    rp = getattr(job, "render_plan", None) or {}
    if link.scope == "clip" and link.clip_id is not None:
        # Trim ops to the selected clip's time range when we can find it.
        clips = getattr(job, "clips", []) or []
        target = next((c for c in clips if getattr(c, "id", None) == link.clip_id), None)
        if target is not None and rp.get("ops"):
            start = float(getattr(target, "start", 0.0))
            end = float(getattr(target, "end", 0.0))
            rp = dict(rp)
            rp["ops"] = [
                op for op in rp["ops"]
                if op.get("end_sec", 0) > start and op.get("start_sec", 0) < end
            ]
    return {"render_plan": rp}


@router.get("/api/share/public/{token}/video")
async def public_share_video(token: str, request: Request):
    """Stream the shared job's source video to an unauthenticated
    recipient (with Range request support so HTML5 seek works).

    Gated solely by the share token — the middleware allow-lists this
    path, so the token itself is the credential. Mirrors the logic of
    the authenticated ``/api/files/{job_id}/{path}`` endpoint in
    ``backend/main.py`` but keyed off the share link's ``job_id``
    instead of accepting a job id from the URL.

    Used by ``SharedView`` on the frontend: the recipient sees the
    same interactive editor preview the owner sees on the SEO page —
    play / pause / scrub / subtitles / layout — without signing in.
    """
    link = await _resolve_link_or_404(token)
    job = await database.load_job(link.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    # Resolve the source video file. Prefer ``file_path`` on the job
    # record (canonical), fall back to scanning the uploads dir for
    # anything named ``video.*`` which is how ``chunked_upload`` lands
    # files during ingestion.
    candidates: list[str] = []
    fp = getattr(job, "file_path", "") or ""
    if fp and os.path.isfile(fp):
        candidates.append(fp)
    uploads_dir = os.path.join("/data/uploads", link.job_id)
    if os.path.isdir(uploads_dir):
        for name in os.listdir(uploads_dir):
            full = os.path.join(uploads_dir, name)
            if os.path.isfile(full) and name.startswith("video."):
                candidates.append(full)

    file_path = next((c for c in candidates if os.path.isfile(c)), None)
    if not file_path:
        raise HTTPException(status_code=404, detail="video file not found")

    # Swap in a browser-playable derivative when the source uses a
    # container + codec combo the HTML5 ``<video>`` element can't
    # decode. This is what fixed silent MKV previews (Speed Racer
    # ep.1 had AC3 audio — Chrome renders the H.264 video fine but
    # drops the AC3 audio track, so the recipient got a silent clip).
    # Idempotent + cached: first hit blocks for FFmpeg, later hits
    # serve the cached ``browser_preview.mp4`` instantly.
    try:
        from backend.services.browser_preview import ensure_browser_preview_async
        file_path = await ensure_browser_preview_async(file_path)
    except Exception as e:  # pragma: no cover — fallback path
        logger.warning("share: browser_preview generation failed: %s", e)

    ext = os.path.splitext(file_path)[1].lower()
    content_types = {
        ".mp4": "video/mp4", ".webm": "video/webm",
        ".mov": "video/quicktime", ".mkv": "video/x-matroska",
    }
    content_type = content_types.get(ext, "application/octet-stream")

    # Range request support — HTML5 video seeks with a
    # ``Range: bytes=start-end`` header. Without this, the browser
    # can't scrub the video.
    #
    # Shares a streaming helper with ``/api/files`` so the same
    # low-memory chunked delivery applies here. The previous
    # implementation buffered the whole range in RAM, which made
    # large seeks feel slow to unauthenticated viewers (who often
    # open the share on phones / spotty networks and retry seeks).
    file_size = os.path.getsize(file_path)
    # Shared previews never change for a given token → long-lived
    # cache makes revisits / repeat seeks much faster.
    _share_cache_headers = {"Cache-Control": "public, max-age=3600"}
    range_header = request.headers.get("range")
    if range_header:
        try:
            parsed = parse_range_header(range_header, file_size)
        except ValueError:
            raise HTTPException(status_code=416, detail="invalid range header")
        if parsed is not None:
            start, end = parsed
            return stream_file_range(
                file_path, file_size, start, end, content_type,
                extra_headers=_share_cache_headers,
            )

    response = FileResponse(file_path, media_type=content_type)
    for k, v in _share_cache_headers.items():
        response.headers[k] = v
    return response


@router.get("/api/share/public/{token}/transcript.srt")
async def public_share_transcript_srt(token: str):
    link = await _resolve_link_or_404(token)
    job = await database.load_job(link.job_id)
    if job is None or not getattr(job, "transcript", None):
        raise HTTPException(status_code=404, detail="no transcript")
    from backend.models import TranscriptSegment
    from backend.services.srt_generator import generate_srt
    segments = []
    if link.scope == "clip" and link.clip_id is not None:
        clips = getattr(job, "clips", []) or []
        target = next((c for c in clips if getattr(c, "id", None) == link.clip_id), None)
        s = float(getattr(target, "start", 0)) if target else 0.0
        e = float(getattr(target, "end", 0)) if target else 0.0
        for raw in job.transcript:
            seg = TranscriptSegment(**raw) if isinstance(raw, dict) else raw
            if seg.end < s or seg.start > e:
                continue
            segments.append(seg)
    else:
        segments = [TranscriptSegment(**raw) if isinstance(raw, dict) else raw
                    for raw in job.transcript]
    body = generate_srt(segments, include_speakers=True)
    base = (getattr(job, "filename", "") or token).rsplit(".", 1)[0]
    return Response(
        content=body,
        media_type="text/srt; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{base}.srt"'},
    )


# ── Public export endpoints (NO auth required — token is credential) ──
#
# A share-link recipient would previously hit a 401 when trying to
# export a clip because the clip endpoints require ``get_current_user``.
# These public shims validate the share token, enforce scope (a clip-
# scoped link can only export *that* clip), build a synthetic "caller
# context" attributing the export to the shared link rather than a
# specific user, then forward to the same underlying ``export_clip`` /
# ``export_full_video`` functions the authenticated endpoints use —
# so there's exactly one code path for actual export logic.


def _make_share_caller(token: str):
    """Build a synthetic ``User``-like object for export attribution.

    The real ``export_clip_endpoint`` only reads ``.id`` and ``.username``
    from the ``user`` parameter (for ``exported_by_user_id`` /
    ``exported_by_username`` on the job's ``exported_clips`` entry), so
    a ``SimpleNamespace`` with those two attributes is sufficient. We
    stamp the token prefix into the id so admins browsing the Exports
    tab can trace back to the originating share link.
    """
    from types import SimpleNamespace
    from backend.app.auth.models import Role
    short = token[:8] if token else "unknown"
    return SimpleNamespace(
        id=f"share:{short}",
        username=f"Shared link ({short})",
        role=Role.USER,
    )


@router.post("/api/share/public/{token}/export-clip")
async def public_share_export_clip(token: str, req_body: dict, request: Request):
    """Export a clip via a share link. Scope-enforced.

    * ``token``-validated through the shared-link store.
    * If the link is clip-scoped, ``req_body["clip_id"]`` is forced to
      ``link.clip_id`` — the recipient cannot export a different clip
      from the same job by tampering with the request body.
    * ``job_id`` is pulled from the link, not from the URL or body —
      the recipient cannot pivot to another job either.
    """
    from backend.models import ExportRequest
    from backend.routers.clips import export_clip_endpoint

    link = await _resolve_link_or_404(token)
    try:
        req = ExportRequest.model_validate(req_body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid export request: {e}")

    # Clip-scope enforcement: lock the clip_id to the one the link
    # was minted for. Ignore whatever the client sent.
    if link.scope == "clip":
        if link.clip_id is None:
            raise HTTPException(status_code=400, detail="malformed clip-scoped link (no clip_id)")
        req.clip_id = int(link.clip_id)
    elif link.scope == "job":
        # Job-scoped links can export any clip belonging to the job.
        # Verify the requested clip_id actually exists so the export
        # doesn't fail deep in the pipeline with a confusing error.
        job = await database.load_job(link.job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        valid_ids = {getattr(c, "id", None) for c in (getattr(job, "clips", None) or [])}
        # Allow clip_id==0 (full-video sentinel the frontend uses).
        if req.clip_id and req.clip_id != 0 and req.clip_id not in valid_ids:
            raise HTTPException(status_code=404, detail=f"clip {req.clip_id} not found in job")

    caller = _make_share_caller(token)
    # Forward to the authenticated endpoint function. Calling it
    # programmatically (instead of via HTTP) bypasses the ``Depends``
    # machinery — the synthetic caller is used directly. ``job_id`` is
    # overridden with the link's job_id so the URL has no authority
    # over which job gets written to.
    return await export_clip_endpoint(link.job_id, req, user=caller)


@router.post("/api/share/public/{token}/export-full-video")
async def public_share_export_full_video(token: str, req_body: dict):
    """Export the full video via a share link.

    Only available for job-scoped links. Clip-scoped shares cannot
    bypass their scope by exporting the entire video.
    """
    from backend.models import FullVideoExportRequest
    from backend.routers.clips import export_full_video_endpoint

    link = await _resolve_link_or_404(token)
    if link.scope != "job":
        raise HTTPException(
            status_code=403,
            detail=(
                "This share link is scoped to a single clip. Full-video "
                "export is only available to job-scoped share links."
            ),
        )

    try:
        req = FullVideoExportRequest.model_validate(req_body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid export request: {e}")

    caller = _make_share_caller(token)
    return await export_full_video_endpoint(link.job_id, req, user=caller)


@router.get("/api/share/public/{token}/download/{filename}")
async def public_share_download(token: str, filename: str, request: Request):
    """Stream an exported clip file to a share recipient.

    The export endpoints above produce files under
    ``/data/uploads/{job_id}/clips/`` (or ``/data/outputs/{job_id}/...``).
    The normal ``/api/files/{job_id}/clips/{filename}`` serve path is
    auth-gated, so a shared-link recipient can't fetch what they just
    exported. This shim token-gates the same file, restricting the
    filesystem search to the share's job directory.

    Clip-scoped links are additionally restricted: the filename must
    embed the linked ``clip_id`` (``clip_{id}.mp4`` is the pipeline's
    naming convention) so a recipient can't download a sibling clip's
    export just because it happened to be in the same folder.
    """
    # Basic path hygiene — the filename is used as a path component.
    # Reject anything that looks like it's trying to escape the clip
    # directory. ``os.path.basename`` strips slashes already; this is
    # belt-and-braces.
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(status_code=400, detail="invalid filename")

    link = await _resolve_link_or_404(token)

    # Resolve the file inside the share's job directory only.
    candidates = [
        os.path.join("/data/uploads", link.job_id, "clips", filename),
        os.path.join("/data/outputs", link.job_id, filename),
        os.path.join("/data/uploads", link.job_id, filename),
    ]
    file_path = next((c for c in candidates if os.path.isfile(c)), None)
    if not file_path:
        raise HTTPException(status_code=404, detail="file not found")

    # Clip-scope tightening: require ``clip_{id}`` somewhere in the
    # filename when the link is clip-scoped. The exporter names files
    # ``{clipTitle}_clip{id}_{timestamp}.mp4`` by default; the token
    # ``clip{id}`` (or ``clip_{id}``) appears there.
    if link.scope == "clip" and link.clip_id is not None:
        needle_a = f"clip{link.clip_id}"
        needle_b = f"clip_{link.clip_id}"
        full_export = "_full_" in filename  # allow "full video" style names
        if needle_a not in filename and needle_b not in filename and not full_export:
            raise HTTPException(
                status_code=403,
                detail="filename does not match clip-scoped share link",
            )

    ext = os.path.splitext(file_path)[1].lower()
    content_types = {
        ".mp4": "video/mp4", ".webm": "video/webm",
        ".mov": "video/quicktime", ".mkv": "video/x-matroska",
    }
    content_type = content_types.get(ext, "application/octet-stream")

    # Range-request support so the browser can stream / resume.
    # Exports re-use the same chunked streaming helper as the source
    # video endpoints to avoid pinning megabytes of clip bytes in
    # RAM during resumable downloads.
    file_size = os.path.getsize(file_path)
    _dl_headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Content-Disposition": f'attachment; filename="{os.path.basename(file_path)}"',
    }
    range_header = request.headers.get("range")
    if range_header:
        try:
            parsed = parse_range_header(range_header, file_size)
        except ValueError:
            raise HTTPException(status_code=416, detail="invalid range header")
        if parsed is not None:
            start, end = parsed
            return stream_file_range(
                file_path, file_size, start, end, content_type,
                extra_headers=_dl_headers,
            )

    return FileResponse(
        file_path,
        media_type=content_type,
        filename=os.path.basename(file_path),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@router.post("/api/share/public/{token}/cancel-export/{clip_id}")
async def public_share_cancel_export(token: str, clip_id: int):
    """Cancel an in-progress export started via the share link.

    Paired with the two endpoints above — share-users need a way to
    abort a long export without an account. The cancel endpoint on the
    authenticated side doesn't need the user at all (it's keyed off
    ``job_id + clip_id``), we just need to keep the share recipient
    out of the auth middleware.
    """
    from backend.routers.clips import cancel_export_endpoint

    link = await _resolve_link_or_404(token)
    # Clip-scope enforcement: can only cancel the linked clip.
    if link.scope == "clip" and link.clip_id is not None and clip_id not in (link.clip_id, 0):
        raise HTTPException(
            status_code=403,
            detail="share link is scoped to a different clip",
        )
    return await cancel_export_endpoint(link.job_id, clip_id)
