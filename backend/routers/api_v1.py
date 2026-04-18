"""Versioned REST API (v1) for AI agents and external integrations.

All endpoints require Bearer token authentication except /health, /info,
/skill, and /openapi.json.  The frontend continues to use the existing
/api/* routes without any auth.

Response envelope:
    {"success": true, "data": {...}, "error": null, "meta": {...}}
"""

import asyncio
import json
import logging
import os
import secrets
import shutil
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile, File
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from backend import database
from backend.auth import (
    get_or_create_api_key,
    mask_key,
    regenerate_api_key,
    verify_api_key,
)
from backend.config import settings
from backend.models import (
    ExportRequest,
    JobResult,
    JobStatus,
    SubtitleSettings,
)
from backend.webhooks import send_webhook

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["v1"])


# ── Response Envelope ──────────────────────────────────────────────────

class Meta(BaseModel):
    timestamp: str = ""
    request_id: str = ""


class ApiResponse(BaseModel):
    success: bool = True
    data: Any = None
    error: Any = None
    meta: Meta = Meta()


def _ok(data: Any, request_id: str = "") -> dict:
    """Build a success response envelope."""
    return {
        "success": True,
        "data": data,
        "error": None,
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id or f"req_{secrets.token_hex(6)}",
        },
    }


def _err(code: str, message: str, status: int = 400, request_id: str = "") -> HTTPException:
    """Build an error as an HTTPException with envelope body."""
    body = {
        "success": False,
        "data": None,
        "error": {"code": code, "message": message},
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id or f"req_{secrets.token_hex(6)}",
        },
    }
    raise HTTPException(status_code=status, detail=body)


# ── Health & Info (no auth) ────────────────────────────────────────────

@router.get("/health", summary="Health check (no auth required)")
async def health():
    """Basic health check for agents to verify the service is running."""
    whisper_loaded = False
    try:
        from backend.services.transcription import _model
        whisper_loaded = _model is not None
    except (ImportError, AttributeError):
        pass

    disk_free = 0.0
    try:
        usage = shutil.disk_usage("/data")
        disk_free = round(usage.free / (1024 ** 3), 1)
    except Exception:
        pass

    return _ok({
        "status": "healthy",
        "whisper_loaded": whisper_loaded,
        "disk_free_gb": disk_free,
    })


@router.get("/info", summary="System info (no auth required)")
async def info():
    """System information: version, capabilities, and configuration status."""
    jobs = await database.list_jobs()
    return _ok({
        "version": "1.0.0",
        "capabilities": [
            "video_upload", "transcription", "scene_analysis", "viral_clip_detection",
            "subtitle_rendering", "subject_tracking", "seo_generation", "batch_export",
            "workflow_pipeline",
        ],
        "providers_configured": {
            "openrouter": bool(settings.OPENROUTER_API_KEY),
            "anthropic": bool(settings.ANTHROPIC_API_KEY),
            "gemini": bool(settings.GEMINI_API_KEY),
            "groq": bool(settings.GROQ_API_KEY),
        },
        "total_jobs": len(jobs),
    })


@router.get("/skill", summary="OpenClaw AgentSkill SKILL.md (no auth required)")
async def serve_skill():
    """Serve the SKILL.md file for OpenClaw and other agent frameworks."""
    skill_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "static", "openclaw-skill", "SKILL.md",
    )
    if not os.path.isfile(skill_path):
        _err("SKILL_NOT_FOUND", "SKILL.md not found", 404)
    return FileResponse(skill_path, media_type="text/markdown")


# ── Auth ───────────────────────────────────────────────────────────────

@router.get("/auth/key", summary="Get current API key (masked)", dependencies=[Depends(verify_api_key)])
async def get_api_key():
    """Return the current API key (masked for display)."""
    key = get_or_create_api_key()
    return _ok({"key": key, "masked": mask_key(key)})


@router.post("/auth/key/regenerate", summary="Regenerate API key", dependencies=[Depends(verify_api_key)])
async def regenerate_key():
    """Generate a new API key. The old key is immediately invalidated."""
    new_key = regenerate_api_key()
    return _ok({"key": new_key, "masked": mask_key(new_key)})


# ── Videos ─────────────────────────────────────────────────────────────

class UploadUrlBody(BaseModel):
    url: str = Field(description="Video URL to download")
    language: str = Field(default="", description="ISO 639-1 language code (empty = auto-detect)")
    auto_analyze: bool = Field(default=True, description="Start analysis immediately")
    callback_url: str = Field(default="", description="Webhook URL for completion notification")


@router.post("/videos/upload", summary="Upload video file", dependencies=[Depends(verify_api_key)])
async def upload_video_file(file: UploadFile = File(...)):
    """Upload a video via multipart form data."""
    if not file.filename:
        _err("INVALID_FILE", "No file provided")

    job_id = str(uuid.uuid4())
    job_dir = f"/data/uploads/{job_id}"
    os.makedirs(job_dir, exist_ok=True)

    filename = file.filename or "video.mp4"
    video_path = os.path.join(job_dir, filename)

    try:
        with open(video_path, "wb") as f:
            while chunk := await file.read(65536):
                f.write(chunk)

        file_size = os.path.getsize(video_path)
        if file_size == 0:
            shutil.rmtree(job_dir, ignore_errors=True)
            _err("EMPTY_FILE", "Uploaded file is empty")

        now = datetime.now(timezone.utc).isoformat()
        job = JobResult(
            job_id=job_id, filename=filename, file_path=video_path,
            file_size_mb=round(file_size / (1024 * 1024), 2),
            status=JobStatus.QUEUED, progress=0,
            progress_message="Uploaded via API, waiting for analysis",
            created_at=now, updated_at=now,
        )
        await database.save_job(job)

        return _ok({"job_id": job_id, "status": "uploaded", "filename": filename})

    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        logger.exception("File upload failed: %s", e)
        _err("UPLOAD_FAILED", str(e), 500)


@router.post("/videos/upload-url", summary="Upload video from URL", dependencies=[Depends(verify_api_key)])
async def upload_video_url(body: UploadUrlBody):
    """Download a video from a URL and create a job."""
    from backend.routers.agent import upload_from_url, UploadUrlRequest
    from fastapi import BackgroundTasks

    req = UploadUrlRequest(
        url=body.url, language=body.language, auto_analyze=body.auto_analyze,
    )
    bg = BackgroundTasks()
    result = await upload_from_url(req, bg)
    # Execute background tasks
    for task in bg.tasks:
        asyncio.create_task(task["func"](*task["args"], **task["kwargs"]))

    return _ok({
        "job_id": result.job_id,
        "status": result.status,
        "filename": result.filename,
        "deduplicated": result.deduplicated,
    })


@router.get("/videos", summary="List all videos/jobs", dependencies=[Depends(verify_api_key)])
async def list_videos(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List all video processing jobs with pagination."""
    all_jobs = await database.list_jobs()

    if status:
        all_jobs = [j for j in all_jobs if j.status == status]

    all_jobs.sort(key=lambda j: j.created_at, reverse=True)
    total = len(all_jobs)
    page = all_jobs[offset:offset + limit]

    return _ok({
        "jobs": [
            {
                "job_id": j.job_id, "filename": j.filename,
                "duration": j.duration, "status": j.status,
                "progress": j.progress, "created_at": j.created_at,
                "clip_count": len(j.clips), "file_size_mb": j.file_size_mb,
            }
            for j in page
        ],
        "total": total, "limit": limit, "offset": offset,
    })


@router.get("/videos/{job_id}", summary="Get video/job details", dependencies=[Depends(verify_api_key)])
async def get_video(job_id: str):
    """Get full job details including transcript, scenes, clips."""
    job = await database.load_job(job_id)
    if not job:
        _err("VIDEO_NOT_FOUND", f"No video found with job_id '{job_id}'", 404)

    return _ok(job.model_dump())


@router.delete("/videos/{job_id}", summary="Delete video/job", dependencies=[Depends(verify_api_key)])
async def delete_video(job_id: str):
    """Delete a job and all associated files."""
    job = await database.load_job(job_id)
    if not job:
        _err("VIDEO_NOT_FOUND", f"No video found with job_id '{job_id}'", 404)

    await database.delete_job(job_id)
    return _ok({"job_id": job_id, "deleted": True})


# ── Analysis ───────────────────────────────────────────────────────────

class AnalyzeBody(BaseModel):
    callback_url: str = Field(default="", description="Webhook URL for completion notification")


@router.post("/videos/{job_id}/analyze", summary="Start video analysis", dependencies=[Depends(verify_api_key)])
async def start_analysis(job_id: str, body: AnalyzeBody = AnalyzeBody()):
    """Start or re-trigger the analysis pipeline for a video."""
    job = await database.load_job(job_id)
    if not job:
        _err("VIDEO_NOT_FOUND", f"No video found with job_id '{job_id}'", 404)

    from backend.services.pipeline import run_analysis

    async def _run_and_notify():
        await run_analysis(job_id)
        if body.callback_url:
            result = await database.load_job(job_id)
            await send_webhook(body.callback_url, "analysis_complete", {
                "job_id": job_id,
                "status": result.status if result else "unknown",
                "clip_count": len(result.clips) if result else 0,
            })

    asyncio.create_task(_run_and_notify())
    return _ok({"job_id": job_id, "status": "analyzing", "message": "Analysis started"})


@router.get("/videos/{job_id}/status", summary="Get analysis progress", dependencies=[Depends(verify_api_key)])
async def get_analysis_status(job_id: str):
    """Get current analysis status and progress."""
    job = await database.load_job(job_id)
    if not job:
        _err("VIDEO_NOT_FOUND", f"No video found with job_id '{job_id}'", 404)

    return _ok({
        "job_id": job_id,
        "status": job.status,
        "progress": job.progress,
        "progress_message": job.progress_message,
    })


@router.get("/videos/{job_id}/summary", summary="Get AI summary", dependencies=[Depends(verify_api_key)])
async def get_summary(job_id: str):
    """Get the AI-generated video summary."""
    job = await database.load_job(job_id)
    if not job:
        _err("VIDEO_NOT_FOUND", f"No video found with job_id '{job_id}'", 404)
    if not job.summary:
        _err("ANALYSIS_NOT_COMPLETE", "Summary not available yet", 404)

    return _ok(job.summary.model_dump())


@router.get("/videos/{job_id}/transcript", summary="Get transcript", dependencies=[Depends(verify_api_key)])
async def get_transcript(
    job_id: str,
    format: str = Query("json", description="Output format: json, srt, vtt, txt"),
):
    """Get the full transcript with timestamps and speaker diarization."""
    job = await database.load_job(job_id)
    if not job:
        _err("VIDEO_NOT_FOUND", f"No video found with job_id '{job_id}'", 404)
    if not job.transcript:
        _err("ANALYSIS_NOT_COMPLETE", "Transcript not available yet", 404)

    if format == "json":
        return _ok([s.model_dump() for s in job.transcript])

    if format == "srt":
        from backend.services.srt_generator import generate_srt
        content = generate_srt(job.transcript)
        return Response(content=content, media_type="text/plain")

    if format in ("vtt", "txt"):
        lines = []
        if format == "vtt":
            lines.append("WEBVTT\n")
        for s in job.transcript:
            if format == "vtt":
                start = _fmt_vtt_time(s.start)
                end = _fmt_vtt_time(s.end)
                lines.append(f"{start} --> {end}")
                lines.append(f"{s.speaker}: {s.text}\n")
            else:
                lines.append(f"[{s.speaker}] {s.text}")
        return Response(content="\n".join(lines), media_type="text/plain")

    _err("INVALID_FORMAT", f"Unknown format: {format}")


def _fmt_vtt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


@router.get("/videos/{job_id}/scenes", summary="Get detected scenes", dependencies=[Depends(verify_api_key)])
async def get_scenes(job_id: str):
    """Get all detected key scenes."""
    job = await database.load_job(job_id)
    if not job:
        _err("VIDEO_NOT_FOUND", f"No video found with job_id '{job_id}'", 404)

    return _ok([s.model_dump() for s in job.scenes])


@router.get("/videos/{job_id}/tracking-debug", summary="Debug subject tracking data", dependencies=[Depends(verify_api_key)])
async def get_tracking_debug(job_id: str):
    """Diagnostic endpoint to verify per-second tracking data in database."""
    job = await database.load_job(job_id)
    if not job:
        _err("VIDEO_NOT_FOUND", f"No video found with job_id '{job_id}'", 404)

    scenes = job.scenes or []
    dense_scenes = [s for s in scenes if s.description == "[dense face tracking]"]
    ai_scenes = [s for s in scenes if s.description != "[dense face tracking]"]
    sx_values = [s.subject_x for s in scenes]
    asx_values = [s.active_speaker_x for s in scenes if s.active_speaker_x is not None]

    return _ok({
        "total_scenes": len(scenes),
        "dense_face_scenes": len(dense_scenes),
        "ai_vision_scenes": len(ai_scenes),
        "subject_x_range": [min(sx_values), max(sx_values)] if sx_values else None,
        "active_speaker_x_count": len(asx_values),
        "unique_subject_x": len(set(sx_values)),
        "scenes_with_face_positions": sum(1 for s in scenes if s.face_positions),
        "dense_tracking_summary": job.dense_tracking_summary,
        "expected": "total_scenes should be ~648 (589 dense + 59 AI) for per-second tracking",
    })


@router.get("/videos/{job_id}/clips", summary="Get viral clip candidates", dependencies=[Depends(verify_api_key)])
async def get_clips(
    job_id: str,
    min_score: int = Query(0, ge=0, le=100),
    min_duration: float = Query(0, ge=0),
    max_duration: float = Query(9999, ge=0),
):
    """Get viral clip candidates with optional filters."""
    job = await database.load_job(job_id)
    if not job:
        _err("VIDEO_NOT_FOUND", f"No video found with job_id '{job_id}'", 404)

    clips = job.clips
    if min_score > 0:
        clips = [c for c in clips if c.viral_score >= min_score]
    if min_duration > 0:
        clips = [c for c in clips if c.duration >= min_duration]
    if max_duration < 9999:
        clips = [c for c in clips if c.duration <= max_duration]

    return _ok([c.model_dump() for c in clips])


# ── Clip Operations ────────────────────────────────────────────────────

@router.get("/videos/{job_id}/clips/{clip_id}", summary="Get specific clip details", dependencies=[Depends(verify_api_key)])
async def get_clip(job_id: str, clip_id: int):
    """Get details for a specific clip candidate."""
    job = await database.load_job(job_id)
    if not job:
        _err("VIDEO_NOT_FOUND", f"No video found with job_id '{job_id}'", 404)

    clip = next((c for c in job.clips if c.id == clip_id), None)
    if not clip:
        _err("CLIP_NOT_FOUND", f"Clip {clip_id} not found", 404)

    return _ok(clip.model_dump())


class ClipSettingsBody(BaseModel):
    aspect_ratio: Optional[str] = None
    subtitles_enabled: Optional[bool] = None
    subtitle_font: Optional[str] = None
    subtitle_font_size: Optional[int] = None
    subtitle_font_color: Optional[str] = None
    subtitle_stroke_color: Optional[str] = None
    subtitle_stroke_width: Optional[int] = None
    subtitle_position: Optional[str] = None
    subtitle_background_enabled: Optional[bool] = None
    subtitle_background_color: Optional[str] = None
    subtitle_background_opacity: Optional[float] = None
    active_word_highlight: Optional[bool] = None
    highlight_color: Optional[str] = None
    subject_tracking: Optional[bool] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    export_quality: Optional[str] = None


class ExportClipBody(BaseModel):
    settings: Optional[ClipSettingsBody] = None
    callback_url: str = Field(default="", description="Webhook URL for completion notification")


@router.put("/videos/{job_id}/clips/{clip_id}/settings", summary="Update clip settings", dependencies=[Depends(verify_api_key)])
async def update_clip_settings(job_id: str, clip_id: int, body: ClipSettingsBody):
    """Update settings for a specific clip."""
    job = await database.load_job(job_id)
    if not job:
        _err("VIDEO_NOT_FOUND", f"No video found with job_id '{job_id}'", 404)

    clip = next((c for c in job.clips if c.id == clip_id), None)
    if not clip:
        _err("CLIP_NOT_FOUND", f"Clip {clip_id} not found", 404)

    applied = {}
    if body.start_time is not None:
        clip.start_time = body.start_time
        applied["start_time"] = body.start_time
    if body.end_time is not None:
        clip.end_time = body.end_time
        applied["end_time"] = body.end_time
    if body.start_time is not None or body.end_time is not None:
        clip.duration = round(clip.end_time - clip.start_time, 2)
        applied["duration"] = clip.duration

    await database.save_job(job)
    return _ok({"clip_id": clip_id, "applied_settings": applied})


@router.post("/videos/{job_id}/clips/{clip_id}/export", summary="Export clip", dependencies=[Depends(verify_api_key)])
async def export_clip(job_id: str, clip_id: int, body: ExportClipBody = ExportClipBody()):
    """Export a single clip with optional settings override."""
    job = await database.load_job(job_id)
    if not job:
        _err("VIDEO_NOT_FOUND", f"No video found with job_id '{job_id}'", 404)

    clip = next((c for c in job.clips if c.id == clip_id), None)
    if not clip:
        _err("CLIP_NOT_FOUND", f"Clip {clip_id} not found", 404)

    s = body.settings or ClipSettingsBody()
    # Build SubtitleSettings from individual fields if any are provided
    sub_kwargs = {}
    if s.subtitle_font is not None:
        sub_kwargs["font"] = s.subtitle_font
    if s.subtitle_font_size is not None:
        sub_kwargs["size"] = s.subtitle_font_size
    if s.subtitle_font_color is not None:
        sub_kwargs["font_color"] = s.subtitle_font_color
    if s.subtitle_stroke_color is not None:
        sub_kwargs["outline_color"] = s.subtitle_stroke_color
    if s.subtitle_stroke_width is not None:
        sub_kwargs["outline_width"] = s.subtitle_stroke_width
    if s.subtitle_position is not None:
        sub_kwargs["position"] = s.subtitle_position
    if s.subtitle_background_enabled is not None:
        sub_kwargs["background_enabled"] = s.subtitle_background_enabled
    if s.subtitle_background_color is not None:
        sub_kwargs["background_color"] = s.subtitle_background_color
    if s.subtitle_background_opacity is not None:
        sub_kwargs["background_opacity"] = int(s.subtitle_background_opacity)
    if s.active_word_highlight is not None:
        sub_kwargs["active_word_enabled"] = s.active_word_highlight
    if s.highlight_color is not None:
        sub_kwargs["active_word_color"] = s.highlight_color
    subtitle_settings = SubtitleSettings(**sub_kwargs) if sub_kwargs else None

    export_req = ExportRequest(
        start=s.start_time or clip.start_time,
        end=s.end_time or clip.end_time,
        clip_id=clip_id,
        clip_title=clip.title,
        aspect_ratio=s.aspect_ratio or "9:16",
        subtitles_enabled=s.subtitles_enabled if s.subtitles_enabled is not None else True,
        subtitle_settings=subtitle_settings,
        export_quality=s.export_quality or "1080p",
    )

    from backend.routers.clips import export_clip_endpoint
    await export_clip_endpoint(job_id, export_req)

    export_id = f"{job_id}_{clip_id}"

    if body.callback_url:
        asyncio.create_task(_wait_and_notify_export(
            job_id, clip_id, export_id, body.callback_url,
        ))

    return _ok({"export_id": export_id, "status": "encoding", "message": "Export started"})


async def _wait_and_notify_export(job_id: str, clip_id: int, export_id: str, callback_url: str):
    """Wait for export to finish and send webhook."""
    from backend.routers.clips import _active_export_tasks
    task = _active_export_tasks.get(export_id)
    if task:
        try:
            await task
        except Exception:
            pass

    job = await database.load_job(job_id)
    exported = None
    if job:
        for exp in reversed(job.exported_clips):
            if exp.get("clip_id") == clip_id:
                exported = exp
                break

    await send_webhook(callback_url, "export_complete", {
        "export_id": export_id,
        "job_id": job_id,
        "clip_id": clip_id,
        "status": "complete" if exported else "failed",
        "download_url": f"/api/files/{job_id}/clips/{exported['filename']}" if exported else None,
    })


class BatchExportBody(BaseModel):
    indices: list[int] = Field(default=[], description="Clip IDs to export. Empty = all.")
    settings: Optional[ClipSettingsBody] = None
    callback_url: str = Field(default="", description="Webhook URL for completion notification")


@router.post("/videos/{job_id}/clips/batch-export", summary="Batch export clips", dependencies=[Depends(verify_api_key)])
async def batch_export_clips(job_id: str, body: BatchExportBody):
    """Export multiple clips from the same video."""
    from backend.models_api import BatchExportRequest
    from backend.routers.agent import batch_export

    req = BatchExportRequest(
        clip_ids=body.indices,
        callback_url=body.callback_url,
    )
    result = await batch_export(job_id, req)
    return _ok(result.model_dump())


@router.get("/exports/{export_id}", summary="Get export status", dependencies=[Depends(verify_api_key)])
async def get_export_status(export_id: str):
    """Check the status of a clip export. export_id format: {job_id}_{clip_id}."""
    parts = export_id.split("_", 1)
    if len(parts) != 2:
        _err("INVALID_EXPORT_ID", "export_id must be in format: job_id_clip_id")

    job_id, clip_id_str = parts
    try:
        clip_id = int(clip_id_str)
    except ValueError:
        _err("INVALID_EXPORT_ID", "clip_id must be an integer")

    from backend.routers.agent import get_export_status as agent_export_status
    result = await agent_export_status(job_id, clip_id)
    return _ok(result.model_dump())


@router.get("/exports/{export_id}/download", summary="Download exported clip", dependencies=[Depends(verify_api_key)])
async def download_export(export_id: str):
    """Download the exported MP4 file."""
    parts = export_id.split("_", 1)
    if len(parts) != 2:
        _err("INVALID_EXPORT_ID", "export_id must be in format: job_id_clip_id")

    job_id, clip_id_str = parts
    try:
        clip_id = int(clip_id_str)
    except ValueError:
        _err("INVALID_EXPORT_ID", "clip_id must be an integer")

    job = await database.load_job(job_id)
    if not job:
        _err("VIDEO_NOT_FOUND", "Job not found", 404)

    for exp in reversed(job.exported_clips):
        if exp.get("clip_id") == clip_id:
            filepath = exp.get("path", "")
            if filepath and os.path.isfile(filepath):
                return FileResponse(
                    filepath,
                    media_type="video/mp4",
                    filename=exp.get("filename", os.path.basename(filepath)),
                )
            break

    _err("EXPORT_NOT_FOUND", "Export file not found", 404)


# ── SEO ────────────────────────────────────────────────────────────────

class GenerateSeoBody(BaseModel):
    type: str = Field(
        default="all",
        description="Type of SEO to generate: 'shorts', 'youtube', 'title', 'tags', 'all'",
    )


@router.post("/videos/{job_id}/clips/{clip_id}/seo/generate", summary="Generate SEO", dependencies=[Depends(verify_api_key)])
async def generate_seo(job_id: str, clip_id: int, body: GenerateSeoBody = GenerateSeoBody()):
    """Generate SEO metadata for a clip using AI."""
    job = await database.load_job(job_id)
    if not job:
        _err("VIDEO_NOT_FOUND", f"No video found with job_id '{job_id}'", 404)

    has_any_key = any([
        settings.OPENROUTER_API_KEY, settings.ANTHROPIC_API_KEY,
        settings.GEMINI_API_KEY, settings.GROQ_API_KEY,
    ])
    if not has_any_key:
        _err("NO_API_KEY", "No AI provider API key configured. Please add your key in Settings.")

    result = {}

    if body.type in ("all", "title", "tags"):
        from backend.routers.clips import generate_seo_endpoint
        try:
            seo_result = await generate_seo_endpoint(job_id, clip_id)
            if isinstance(seo_result, dict):
                result["seo"] = seo_result.get("seo")
                result["provider"] = seo_result.get("provider")
        except HTTPException as e:
            _err("SEO_FAILED", str(e.detail), e.status_code)

    if body.type in ("all", "shorts"):
        from backend.routers.clips import generate_description_endpoint, GenerateDescriptionRequest
        try:
            desc_result = await generate_description_endpoint(
                job_id, clip_id, GenerateDescriptionRequest(description_type="shorts"),
            )
            if isinstance(desc_result, dict):
                result["shorts_description"] = desc_result.get("description")
        except HTTPException:
            pass

    if body.type in ("all", "youtube"):
        from backend.routers.clips import generate_description_endpoint, GenerateDescriptionRequest
        try:
            desc_result = await generate_description_endpoint(
                job_id, clip_id, GenerateDescriptionRequest(description_type="long_form"),
            )
            if isinstance(desc_result, dict):
                result["youtube_description"] = desc_result.get("description")
        except HTTPException:
            pass

    return _ok(result)


@router.get("/videos/{job_id}/clips/{clip_id}/seo", summary="Get SEO metadata", dependencies=[Depends(verify_api_key)])
async def get_seo(job_id: str, clip_id: int):
    """Get previously generated SEO metadata for a clip."""
    job = await database.load_job(job_id)
    if not job:
        _err("VIDEO_NOT_FOUND", f"No video found with job_id '{job_id}'", 404)

    clip = next((c for c in job.clips if c.id == clip_id), None)
    if not clip:
        _err("CLIP_NOT_FOUND", f"Clip {clip_id} not found", 404)

    return _ok({
        "clip_id": clip_id,
        "title": clip.title,
        "suggested_caption": clip.suggested_caption,
        "platform": clip.platform,
        "seo_title": clip.seo_title,
        "seo_description": clip.seo_description,
        "seo_tags": clip.seo_tags,
        "seo_platform_tips": clip.seo_platform_tips,
        "shorts_description": clip.shorts_description,
        "longform_description": clip.longform_description,
    })


# ── Settings ───────────────────────────────────────────────────────────

@router.get("/settings", summary="Get current settings", dependencies=[Depends(verify_api_key)])
async def get_settings():
    """Get current application settings. API keys are masked."""
    return _ok({
        "ai_provider": settings.AI_PROVIDER,
        "openrouter_preset": settings.OPENROUTER_PRESET,
        "openrouter_api_key": mask_key(settings.OPENROUTER_API_KEY) if settings.OPENROUTER_API_KEY else "",
        "anthropic_api_key": mask_key(settings.ANTHROPIC_API_KEY) if settings.ANTHROPIC_API_KEY else "",
        "gemini_api_key": mask_key(settings.GEMINI_API_KEY) if settings.GEMINI_API_KEY else "",
        "groq_api_key": mask_key(settings.GROQ_API_KEY) if settings.GROQ_API_KEY else "",
        "whisper_model": settings.WHISPER_MODEL,
        "frame_sample_rate": settings.FRAME_SAMPLE_RATE,
        "subject_tracking_enabled": settings.SUBJECT_TRACKING_ENABLED,
        "auto_analyze": settings.AUTO_ANALYZE,
        "ffmpeg_preset": settings.FFMPEG_PRESET,
        "ffmpeg_crf": settings.FFMPEG_CRF,
    })


class UpdateSettingsBody(BaseModel):
    openrouter_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    openrouter_preset: Optional[str] = None
    whisper_model: Optional[str] = None
    frame_sample_rate: Optional[int] = None
    subject_tracking_enabled: Optional[bool] = None
    auto_analyze: Optional[bool] = None


@router.patch("/settings", summary="Update settings", dependencies=[Depends(verify_api_key)])
async def update_settings(body: UpdateSettingsBody):
    """Partial update of application settings."""
    from backend.routers.settings import _persist_user_settings

    updated = []
    for field, value in body.model_dump(exclude_unset=True).items():
        if value is not None:
            attr = field.upper()
            if hasattr(settings, attr):
                setattr(settings, attr, value)
                updated.append(field)

    if updated:
        _persist_user_settings()

    return _ok({"updated_keys": updated})


@router.get("/settings/providers", summary="Get provider status", dependencies=[Depends(verify_api_key)])
async def get_providers():
    """Get status of all AI providers."""
    import httpx as hx

    ollama_status = "not_configured"
    try:
        async with hx.AsyncClient(timeout=3) as client:
            resp = await client.get(f"{settings.OLLAMA_HOST}/api/version")
            ollama_status = "connected" if resp.status_code == 200 else "error"
    except Exception:
        pass

    return _ok({
        "openrouter": "configured" if settings.OPENROUTER_API_KEY else "not_configured",
        "anthropic": "configured" if settings.ANTHROPIC_API_KEY else "not_configured",
        "gemini": "configured" if settings.GEMINI_API_KEY else "not_configured",
        "groq": "configured" if settings.GROQ_API_KEY else "not_configured",
        "ollama": ollama_status,
    })


# ── Workflows ──────────────────────────────────────────────────────────

class WorkflowBody(BaseModel):
    url: str = Field(description="Video URL to process")
    auto_export_top: int = Field(default=3, ge=1, le=50, description="Number of top clips to export")
    settings: Optional[ClipSettingsBody] = None
    callback_url: str = Field(default="", description="Webhook URL for completion notification")


@router.post("/workflows/process-video", summary="Full pipeline: upload + analyze + export", dependencies=[Depends(verify_api_key)])
async def process_video(body: WorkflowBody):
    """Execute the complete ClipAI pipeline in a single call."""
    from backend.models_api import PipelineRequest
    from backend.routers.agent import run_pipeline
    from fastapi import BackgroundTasks

    s = body.settings or ClipSettingsBody()
    req = PipelineRequest(
        url=body.url,
        auto_select="all",
        clip_count=body.auto_export_top,
        aspect_ratio=s.aspect_ratio or "9:16",
        subtitles_enabled=s.subtitles_enabled if s.subtitles_enabled is not None else True,
        export_quality=s.export_quality or "1080p",
        callback_url=body.callback_url,
    )
    bg = BackgroundTasks()
    result = await run_pipeline(req, bg)
    return _ok(result.model_dump())


@router.get("/workflows/{workflow_id}", summary="Check workflow progress", dependencies=[Depends(verify_api_key)])
async def get_workflow_status(workflow_id: str):
    """Check the status of a running workflow/pipeline."""
    from backend.routers.agent import get_pipeline_status
    try:
        result = await get_pipeline_status(workflow_id)
        return _ok(result.model_dump())
    except HTTPException as e:
        _err("WORKFLOW_NOT_FOUND", str(e.detail), e.status_code)


# ── Agent WebSocket ────────────────────────────────────────────────────

from fastapi import WebSocket, WebSocketDisconnect

@router.websocket("/ws")
async def agent_websocket(ws: WebSocket, api_key: str = Query("")):
    """General-purpose WebSocket for agents. Streams events for all jobs.

    Connect with: ws://host:1353/api/v1/ws?api_key=<your-key>
    """
    if api_key != get_or_create_api_key():
        await ws.close(code=4003, reason="Invalid API key")
        return

    await ws.accept()

    from backend.routers.agent import register_sse_subscriber, unregister_sse_subscriber
    queue: asyncio.Queue = asyncio.Queue(maxsize=200)

    # Subscribe to all job events by registering under a special key
    register_sse_subscriber("__agent_ws__", queue)

    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30)
                await ws.send_json(event)
            except asyncio.TimeoutError:
                # Send keepalive
                await ws.send_json({"type": "keepalive"})
            except WebSocketDisconnect:
                break
    finally:
        unregister_sse_subscriber("__agent_ws__", queue)


# ── API Key Management (no auth, for frontend) ────────────────────────

@router.get("/auth/current-key", summary="Get API key for frontend settings display")
async def get_current_key_for_frontend():
    """Return the current API key. Used by the frontend Settings page.

    This endpoint does NOT require auth since the frontend needs to
    display the key before the user knows it.
    """
    key = get_or_create_api_key()
    return _ok({"key": key, "masked": mask_key(key)})


@router.post("/auth/regenerate-key", summary="Regenerate API key from frontend")
async def regenerate_key_frontend():
    """Regenerate the API key. Used by the frontend Settings page."""
    new_key = regenerate_api_key()
    return _ok({"key": new_key, "masked": mask_key(new_key)})
