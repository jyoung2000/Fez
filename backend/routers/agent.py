"""Agent-friendly API endpoints.

This router provides endpoints specifically designed for AI agent consumption:
- Upload from URL (agents don't have local files)
- Export status polling (agents can't use WebSocket)
- One-shot pipeline (entire workflow in one call)
- Batch export
- Health check
- Quick setup
- SSE event streams
- Download all exports as ZIP
"""

import asyncio
import hashlib
import io
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
import zipfile
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse

from backend import database
from backend.config import settings
from backend.models import (
    ExportRequest,
    JobResult,
    JobStatus,
    SubtitleSettings,
)
from backend.models_api import (
    ActiveExportItem,
    BatchExportRequest,
    BatchExportResponse,
    BatchStatusResponse,
    ErrorResponse,
    ExportStatusResponse,
    HealthResponse,
    JobListResponse,
    JobSummaryResponse,
    PipelineClipResult,
    PipelineRequest,
    PipelineStartResponse,
    PipelineStatusResponse,
    SetupRequest,
    SetupResponse,
    UploadUrlRequest,
    UploadUrlResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["agent"])

# ── Tracking for async operations ────────────────────────────────────

# Export progress cache: export_key -> progress percentage
_export_progress_cache: dict[str, int] = {}

# Pipeline tracking
_active_pipelines: dict[str, asyncio.Task] = {}
_pipeline_status: dict[str, dict] = {}

# Batch export tracking
_active_batches: dict[str, asyncio.Task] = {}
_batch_status: dict[str, dict] = {}

# SSE subscriber queues: job_id -> list[asyncio.Queue]
_sse_subscribers: dict[str, list[asyncio.Queue]] = {}


def register_sse_subscriber(job_id: str, queue: asyncio.Queue):
    if job_id not in _sse_subscribers:
        _sse_subscribers[job_id] = []
    _sse_subscribers[job_id].append(queue)


def unregister_sse_subscriber(job_id: str, queue: asyncio.Queue):
    if job_id in _sse_subscribers:
        _sse_subscribers[job_id] = [q for q in _sse_subscribers[job_id] if q is not queue]
        if not _sse_subscribers[job_id]:
            del _sse_subscribers[job_id]


async def notify_sse_subscribers(job_id: str, event: dict):
    """Push an event to all SSE subscribers for a job."""
    for queue in _sse_subscribers.get(job_id, []):
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            pass


# ── Health ───────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, summary="Health check for agent pre-flight")
async def health():
    """Detailed health check. Agents should call this before starting work
    to verify the container is ready and has required API keys configured."""
    whisper_loaded = False
    try:
        from backend.services.transcription import _model
        whisper_loaded = _model is not None
    except (ImportError, AttributeError):
        pass

    ollama_available = False
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(f"{settings.OLLAMA_HOST}/api/version")
            ollama_available = resp.status_code == 200
    except Exception:
        pass

    disk_free = 0.0
    try:
        usage = shutil.disk_usage("/data")
        disk_free = round(usage.free / (1024 ** 3), 1)
    except Exception:
        pass

    jobs = await database.list_jobs()
    active = sum(1 for j in jobs if j.status not in (JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.CANCELLED))

    return HealthResponse(
        status="ready",
        whisper_loaded=whisper_loaded,
        ollama_available=ollama_available,
        openrouter_key_set=bool(settings.OPENROUTER_API_KEY),
        gemini_key_set=bool(settings.GEMINI_API_KEY),
        groq_key_set=bool(settings.GROQ_API_KEY),
        anthropic_key_set=bool(settings.ANTHROPIC_API_KEY),
        disk_free_gb=disk_free,
        active_jobs=active,
        version="1.0.0",
    )


# ── Quick Setup ──────────────────────────────────────────────────────

@router.post("/setup", response_model=SetupResponse, summary="One-call setup: configure API keys and defaults")
async def quick_setup(req: SetupRequest):
    """Configure the application in a single call. Sets API keys,
    AI preset, and default export settings. Useful for agents
    bootstrapping a fresh container."""
    from backend.routers.settings import _persist_user_settings

    providers_configured = []

    if req.openrouter_api_key:
        settings.OPENROUTER_API_KEY = req.openrouter_api_key
        providers_configured.append("openrouter")

    if req.anthropic_api_key:
        settings.ANTHROPIC_API_KEY = req.anthropic_api_key
        providers_configured.append("anthropic")

    if req.gemini_api_key:
        settings.GEMINI_API_KEY = req.gemini_api_key
        providers_configured.append("gemini")

    if req.groq_api_key:
        settings.GROQ_API_KEY = req.groq_api_key
        providers_configured.append("groq")

    if req.preset:
        settings.OPENROUTER_PRESET = req.preset

    if req.auto_analyze is not None:
        settings.AUTO_ANALYZE = req.auto_analyze

    _persist_user_settings()

    return SetupResponse(
        status="configured",
        providers_configured=providers_configured,
        message=f"Configured {len(providers_configured)} provider(s). Preset: {req.preset}.",
    )


# ── Upload from URL ──────────────────────────────────────────────────

def _compute_file_hash(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


async def _find_job_by_hash(content_hash: str) -> Optional[str]:
    """Check if any existing job has the same content hash."""
    jobs = await database.list_jobs()
    for j in jobs:
        hash_file = os.path.join(f"/data/uploads/{j.job_id}", ".content_hash")
        if os.path.isfile(hash_file):
            try:
                with open(hash_file) as f:
                    if f.read().strip() == content_hash:
                        return j.job_id
            except OSError:
                pass
    return None


@router.post(
    "/upload-url",
    response_model=UploadUrlResponse,
    summary="Upload a video from URL",
    responses={400: {"model": ErrorResponse}},
)
async def upload_from_url(req: UploadUrlRequest, background_tasks: BackgroundTasks):
    """Download a video from a URL and create a job.

    Supports direct video URLs (.mp4, .mkv, .webm, etc.).
    Also supports YouTube/TikTok/Instagram URLs if yt-dlp is installed.
    Deduplicates uploads by content hash — if the same video was previously
    uploaded, returns the existing job instead of re-downloading.
    """
    if not req.url or not req.url.strip():
        raise HTTPException(status_code=400, detail="URL is required")

    url = req.url.strip()
    job_id = str(uuid.uuid4())
    job_dir = f"/data/uploads/{job_id}"
    os.makedirs(job_dir, exist_ok=True)

    # Determine if this is a social media URL requiring yt-dlp
    social_patterns = [
        r"youtube\.com", r"youtu\.be", r"tiktok\.com",
        r"instagram\.com", r"twitter\.com", r"x\.com",
        r"facebook\.com", r"fb\.watch", r"vimeo\.com",
        r"twitch\.tv", r"reddit\.com",
    ]
    is_social = any(re.search(pat, url, re.I) for pat in social_patterns)

    filename = "video.mp4"
    video_path = os.path.join(job_dir, filename)

    try:
        if is_social:
            # Use yt-dlp for social media URLs
            try:
                import subprocess
                result = await asyncio.to_thread(
                    subprocess.run,
                    [
                        "yt-dlp",
                        "--no-playlist",
                        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                        "--merge-output-format", "mp4",
                        "-o", video_path,
                        "--no-warnings",
                        "--quiet",
                        url,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if result.returncode != 0:
                    # Clean up
                    shutil.rmtree(job_dir, ignore_errors=True)
                    raise HTTPException(
                        status_code=400,
                        detail=f"yt-dlp failed: {result.stderr[:500] if result.stderr else 'unknown error'}. "
                               "Ensure yt-dlp is installed in the container.",
                    )
                # yt-dlp may add different extension — find the actual output file
                if not os.path.isfile(video_path):
                    for f in os.listdir(job_dir):
                        if f.startswith("video.") and not f.endswith(".tmp"):
                            video_path = os.path.join(job_dir, f)
                            filename = f
                            break
            except FileNotFoundError:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(
                    status_code=501,
                    detail="yt-dlp is not installed. Install it with: pip install yt-dlp. "
                           "Social media URLs require yt-dlp for download.",
                )
        else:
            # Direct download for regular video URLs
            # Detect file extension from URL
            url_path = url.split("?")[0].split("#")[0]
            ext_match = re.search(r"\.(mp4|mkv|webm|mov|avi)$", url_path, re.I)
            if ext_match:
                ext = ext_match.group(1).lower()
                filename = f"video.{ext}"
                video_path = os.path.join(job_dir, filename)

            async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(connect=30, read=600, write=30, pool=30)) as client:
                async with client.stream("GET", url) as resp:
                    if resp.status_code != 200:
                        shutil.rmtree(job_dir, ignore_errors=True)
                        raise HTTPException(status_code=400, detail=f"Failed to download: HTTP {resp.status_code}")

                    # Try to detect extension from content-type
                    ct = resp.headers.get("content-type", "")
                    if "video/webm" in ct and not filename.endswith(".webm"):
                        filename = "video.webm"
                        video_path = os.path.join(job_dir, filename)
                    elif "video/quicktime" in ct and not filename.endswith(".mov"):
                        filename = "video.mov"
                        video_path = os.path.join(job_dir, filename)

                    with open(video_path, "wb") as f:
                        async for chunk in resp.aiter_bytes(chunk_size=65536):
                            f.write(chunk)

        if not os.path.isfile(video_path) or os.path.getsize(video_path) == 0:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="Downloaded file is empty or missing")

        # Compute content hash for dedup
        content_hash = await asyncio.to_thread(_compute_file_hash, video_path)
        existing_job_id = await _find_job_by_hash(content_hash)
        if existing_job_id:
            shutil.rmtree(job_dir, ignore_errors=True)
            existing_job = await database.load_job(existing_job_id)
            return UploadUrlResponse(
                job_id=existing_job_id,
                status=existing_job.status if existing_job else "exists",
                filename=existing_job.filename if existing_job else "",
                deduplicated=True,
            )

        # Save content hash
        with open(os.path.join(job_dir, ".content_hash"), "w") as f:
            f.write(content_hash)

        file_size = os.path.getsize(video_path)
        lang = req.language.strip().lower() if req.language else ""

        now = datetime.now(timezone.utc).isoformat()
        job = JobResult(
            job_id=job_id,
            filename=filename,
            file_path=video_path,
            file_size_mb=round(file_size / (1024 * 1024), 2),
            language=lang,
            status=JobStatus.QUEUED,
            progress=0,
            progress_message="Downloaded from URL, waiting for analysis",
            created_at=now,
            updated_at=now,
        )
        await database.save_job(job)

        status = "queued"
        if req.auto_analyze:
            from backend.services.pipeline import run_analysis
            background_tasks.add_task(run_analysis, job_id)
            job.progress_message = "Analysis starting..."
            await database.save_job(job)
            status = "analyzing"

        return UploadUrlResponse(
            job_id=job_id,
            status=status,
            filename=filename,
            deduplicated=False,
        )

    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        logger.exception("Upload from URL failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


# ── Export Status Polling ────────────────────────────────────────────

@router.get(
    "/jobs/{job_id}/export-status/{clip_id}",
    response_model=ExportStatusResponse,
    summary="Poll export progress (HTTP alternative to WebSocket)",
)
async def get_export_status(job_id: str, clip_id: int):
    """Poll the status of a clip export. Returns download_url when complete.

    This is the HTTP-polling alternative to the WebSocket export_complete
    message. Call this repeatedly (e.g. every 2-5 seconds) until
    status is 'complete' or 'failed'.
    """
    from backend.routers.clips import _active_export_tasks

    export_key = f"{job_id}_{clip_id}"
    task = _active_export_tasks.get(export_key)

    if task and not task.done():
        progress = _export_progress_cache.get(export_key, 0)
        return ExportStatusResponse(
            status="encoding",
            progress=progress,
            message=f"Encoding in progress ({progress}%)",
        )

    # Check if export exists in job record
    job = await database.load_job(job_id)
    if job:
        # Find the most recent export for this clip_id
        for exp in reversed(job.exported_clips):
            if exp.get("clip_id") == clip_id:
                return ExportStatusResponse(
                    status="complete",
                    progress=100,
                    download_url=f"/api/files/{job_id}/clips/{exp['filename']}",
                    filename=exp.get("filename"),
                    duration=exp.get("duration"),
                    export_quality=exp.get("export_quality"),
                    message="Export complete",
                )

    # Check if the task failed
    if task and task.done():
        exc = task.exception() if not task.cancelled() else None
        if exc:
            return ExportStatusResponse(
                status="failed",
                message=f"Export failed: {str(exc)}",
            )
        elif task.cancelled():
            return ExportStatusResponse(
                status="failed",
                message="Export was cancelled",
            )

    return ExportStatusResponse(status="not_found", message="No export found for this clip")


# ── SSE Event Stream ─────────────────────────────────────────────────

@router.get(
    "/jobs/{job_id}/events",
    summary="Stream job events as Server-Sent Events (HTTP alternative to WebSocket)",
)
async def job_events(job_id: str, request: Request):
    """Stream real-time job progress as Server-Sent Events.

    Usage: curl -N http://host:1353/api/jobs/{job_id}/events

    Each event has format:
      event: {type}
      data: {"type": "status", "progress": 45, "message": "Analyzing scenes..."}

    Stream ends when a 'complete' or 'error' event is sent.
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    register_sse_subscriber(job_id, queue)

    async def event_generator():
        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30)
                    event_type = event.get("type", "message")
                    data = json.dumps(event)
                    yield f"event: {event_type}\ndata: {data}\n\n"
                    if event_type in ("complete", "error", "export_complete"):
                        break
                except asyncio.TimeoutError:
                    # Send keepalive comment
                    yield ": keepalive\n\n"
        finally:
            unregister_sse_subscriber(job_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Batch Export ─────────────────────────────────────────────────────

@router.post(
    "/jobs/{job_id}/export-batch",
    response_model=BatchExportResponse,
    summary="Export multiple clips in one call",
)
async def batch_export(job_id: str, req: BatchExportRequest):
    """Export multiple clips in a single request. Returns a batch_id
    for polling via GET /api/jobs/{job_id}/batch-status/{batch_id}."""
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.clips:
        raise HTTPException(status_code=400, detail="No clips available to export")

    # Determine which clips to export
    if req.clip_ids:
        clips_to_export = [c for c in job.clips if c.id in req.clip_ids]
        if not clips_to_export:
            raise HTTPException(status_code=400, detail="None of the specified clip_ids found")
    else:
        clips_to_export = list(job.clips)

    batch_id = str(uuid.uuid4())[:8]

    _batch_status[batch_id] = {
        "batch_id": batch_id,
        "job_id": job_id,
        "status": "running",
        "total_clips": len(clips_to_export),
        "completed_clips": 0,
        "failed_clips": 0,
        "clips": [],
        "message": f"Exporting {len(clips_to_export)} clips...",
    }

    async def _do_batch():
        from backend.routers.clips import export_clip_endpoint
        try:
            for clip in clips_to_export:
                try:
                    export_req = ExportRequest(
                        start=clip.start_time,
                        end=clip.end_time,
                        clip_id=clip.id,
                        clip_title=clip.title,
                        aspect_ratio=req.aspect_ratio,
                        subtitles_enabled=req.subtitles_enabled,
                        subtitle_settings=req.subtitle_settings,
                        export_quality=req.export_quality,
                    )
                    await export_clip_endpoint(job_id, export_req)

                    # Wait for export to complete
                    from backend.routers.clips import _active_export_tasks
                    export_key = f"{job_id}_{clip.id}"
                    task = _active_export_tasks.get(export_key)
                    if task:
                        await task

                    # Check if export succeeded
                    j = await database.load_job(job_id)
                    exported = None
                    if j:
                        for exp in reversed(j.exported_clips):
                            if exp.get("clip_id") == clip.id:
                                exported = exp
                                break

                    clip_result = {
                        "clip_id": clip.id,
                        "title": clip.title,
                        "viral_score": clip.viral_score,
                        "status": "complete" if exported else "failed",
                        "download_url": f"/api/files/{job_id}/clips/{exported['filename']}" if exported else None,
                    }

                    # Generate SEO if requested
                    if req.generate_seo and exported:
                        try:
                            from backend.routers.clips import generate_seo_endpoint
                            seo_result = await generate_seo_endpoint(job_id, clip.id)
                            clip_result["seo"] = seo_result.get("seo") if isinstance(seo_result, dict) else None
                        except Exception as seo_err:
                            logger.warning("SEO generation failed for clip %s: %s", clip.id, seo_err)

                    _batch_status[batch_id]["clips"].append(clip_result)
                    _batch_status[batch_id]["completed_clips"] += 1
                    _batch_status[batch_id]["message"] = f"Exported {_batch_status[batch_id]['completed_clips']}/{len(clips_to_export)} clips"

                except Exception as clip_err:
                    logger.exception("Batch export failed for clip %s: %s", clip.id, clip_err)
                    _batch_status[batch_id]["clips"].append({
                        "clip_id": clip.id,
                        "title": clip.title,
                        "status": "failed",
                        "error": str(clip_err),
                    })
                    _batch_status[batch_id]["failed_clips"] += 1

            _batch_status[batch_id]["status"] = "complete"
            _batch_status[batch_id]["message"] = f"Batch complete: {_batch_status[batch_id]['completed_clips']} exported, {_batch_status[batch_id]['failed_clips']} failed"

            # Send webhook callback if configured
            if req.callback_url:
                await _send_callback(req.callback_url, {
                    "event": "batch_complete",
                    "batch_id": batch_id,
                    "job_id": job_id,
                    **_batch_status[batch_id],
                })

        except Exception as e:
            logger.exception("Batch export failed: %s", e)
            _batch_status[batch_id]["status"] = "failed"
            _batch_status[batch_id]["message"] = f"Batch failed: {str(e)}"
        finally:
            _active_batches.pop(batch_id, None)

    task = asyncio.create_task(_do_batch())
    _active_batches[batch_id] = task

    return BatchExportResponse(
        batch_id=batch_id,
        job_id=job_id,
        status="running",
        clip_count=len(clips_to_export),
        message=f"Exporting {len(clips_to_export)} clips",
    )


@router.get(
    "/jobs/{job_id}/batch-status/{batch_id}",
    response_model=BatchStatusResponse,
    summary="Poll batch export progress",
)
async def get_batch_status(job_id: str, batch_id: str):
    """Poll the status of a batch export operation."""
    status = _batch_status.get(batch_id)
    if not status:
        raise HTTPException(status_code=404, detail="Batch not found")
    return BatchStatusResponse(**status)


# ── Pipeline (One-Shot) ──────────────────────────────────────────────

@router.post(
    "/pipeline",
    response_model=PipelineStartResponse,
    summary="Full end-to-end pipeline: download, analyze, detect clips, export, SEO",
)
async def run_pipeline(req: PipelineRequest, background_tasks: BackgroundTasks):
    """Execute the complete ClipAI pipeline in a single API call.

    1. Download video from URL
    2. Analyze (transcribe, detect scenes, generate summary)
    3. Detect viral clips
    4. Export clips (top N or all)
    5. Generate SEO metadata

    Returns immediately with a pipeline_id. Poll GET /api/pipeline/{pipeline_id}
    for status updates. Optionally set callback_url to receive a webhook POST
    when the pipeline completes.
    """
    pipeline_id = str(uuid.uuid4())[:12]

    _pipeline_status[pipeline_id] = {
        "pipeline_id": pipeline_id,
        "job_id": "",
        "status": "downloading",
        "progress": 0,
        "progress_message": "Starting pipeline...",
        "clips": [],
        "error": None,
    }

    async def _run():
        try:
            # Step 1: Download video
            _pipeline_status[pipeline_id]["progress_message"] = f"Downloading video from {req.url[:80]}..."
            _pipeline_status[pipeline_id]["progress"] = 5

            # Create upload request
            upload_req = UploadUrlRequest(
                url=req.url,
                language=req.language,
                auto_analyze=False,  # We'll manage the pipeline manually
            )
            # Call upload directly (without background tasks for analysis)
            upload_result = await upload_from_url(upload_req, BackgroundTasks())
            job_id = upload_result.job_id
            _pipeline_status[pipeline_id]["job_id"] = job_id

            if upload_result.deduplicated:
                _pipeline_status[pipeline_id]["progress_message"] = "Video already uploaded (deduplicated)"
                _pipeline_status[pipeline_id]["progress"] = 10
            else:
                _pipeline_status[pipeline_id]["progress_message"] = "Video downloaded successfully"
                _pipeline_status[pipeline_id]["progress"] = 10

            # Step 2: Run analysis
            _pipeline_status[pipeline_id]["status"] = "analyzing"
            _pipeline_status[pipeline_id]["progress_message"] = "Running analysis (transcription, scene detection)..."
            _pipeline_status[pipeline_id]["progress"] = 15

            from backend.services.pipeline import run_analysis
            await run_analysis(job_id)

            # Check analysis result
            job = await database.load_job(job_id)
            if not job or job.status == JobStatus.FAILED:
                _pipeline_status[pipeline_id]["status"] = "failed"
                _pipeline_status[pipeline_id]["error"] = job.error if job else "Analysis failed"
                return

            _pipeline_status[pipeline_id]["progress"] = 60
            _pipeline_status[pipeline_id]["progress_message"] = "Analysis complete. Detecting viral clips..."

            # Step 3: Detect clips (if not already done)
            if not job.clips:
                _pipeline_status[pipeline_id]["status"] = "detecting_clips"
                _pipeline_status[pipeline_id]["progress"] = 65

                from backend.services.ai_orchestrator import AIOrchestrator
                from backend.services.prompts import load_prompts
                from backend.services.pipeline import broadcast_ws

                custom_prompts = load_prompts()
                orchestrator = AIOrchestrator(
                    ws_broadcast=broadcast_ws,
                    custom_prompts=custom_prompts,
                )

                summary_text = None
                if job.summary:
                    parts = [job.summary.overview]
                    if job.summary.key_topics:
                        parts.append(f"Key topics: {', '.join(job.summary.key_topics)}")
                    summary_text = "\n".join(parts)

                clips, clips_provider = await orchestrator.detect_viral_clips(
                    job.transcript, job.scenes, job.duration, job_id,
                    clip_count=req.clip_count,
                    min_duration=req.min_duration,
                    max_duration=req.max_duration,
                    video_summary=summary_text,
                )

                # Filter by duration
                filtered = [
                    c for c in clips
                    if req.min_duration <= c.duration <= req.max_duration
                ]
                for idx, clip in enumerate(filtered, start=1):
                    clip.id = idx

                await database.update_job_status(
                    job_id,
                    status="complete",
                    clips=filtered,
                    progress=100,
                    progress_message=f"Found {len(filtered)} clips via {clips_provider}",
                )
                job = await database.load_job(job_id)

            _pipeline_status[pipeline_id]["progress"] = 75
            _pipeline_status[pipeline_id]["progress_message"] = f"Found {len(job.clips)} clips. Starting export..."

            # Step 4: Export clips
            _pipeline_status[pipeline_id]["status"] = "exporting"
            clips_to_export = sorted(job.clips, key=lambda c: c.viral_score, reverse=True)
            if req.auto_select == "top":
                clips_to_export = clips_to_export[:1]

            from backend.services.clip_exporter import export_clip as do_export

            vid_w, vid_h = 1920, 1080
            if job.resolution:
                try:
                    parts = job.resolution.split("x")
                    vid_w, vid_h = int(parts[0]), int(parts[1])
                except (ValueError, IndexError):
                    pass

            results = []
            for i, clip in enumerate(clips_to_export):
                try:
                    _pipeline_status[pipeline_id]["progress_message"] = f"Exporting clip {i+1}/{len(clips_to_export)}: {clip.title[:50]}"
                    _pipeline_status[pipeline_id]["progress"] = 75 + int(15 * i / max(len(clips_to_export), 1))

                    # Gather subject tracking data
                    clip_subject_x = 50
                    clip_scenes = []
                    if settings.SUBJECT_TRACKING_ENABLED and job.scenes:
                        in_range = [s for s in job.scenes if clip.start_time <= s.timestamp <= clip.end_time]
                        if in_range:
                            clip_subject_x = round(sum(s.subject_x for s in in_range) / len(in_range))
                            clip_scenes = in_range

                    output_path = await do_export(
                        job_id=job_id,
                        video_path=job.file_path,
                        start=clip.start_time,
                        end=clip.end_time,
                        clip_id=clip.id,
                        clip_title=clip.title,
                        aspect_ratio=req.aspect_ratio,
                        subtitles_enabled=req.subtitles_enabled,
                        subtitle_settings=req.subtitle_settings.model_dump() if req.subtitle_settings else None,
                        transcript=[s.model_dump() for s in job.transcript] if req.subtitles_enabled else None,
                        video_width=vid_w,
                        video_height=vid_h,
                        subject_x=clip_subject_x,
                        subject_scenes=clip_scenes or None,
                        scene_cut_timestamps=getattr(job, 'scene_cut_timestamps', None) or None,
                        export_quality=req.export_quality,
                    )

                    # Save to job record
                    j = await database.load_job(job_id)
                    if j:
                        j.exported_clips.append({
                            "clip_id": clip.id,
                            "path": output_path,
                            "filename": os.path.basename(output_path),
                            "title": clip.title,
                            "start": clip.start_time,
                            "end": clip.end_time,
                            "exported_at": datetime.now(timezone.utc).isoformat(),
                            "duration": round(clip.end_time - clip.start_time, 2),
                            "export_quality": req.export_quality,
                            "aspect_ratio": req.aspect_ratio,
                            "subtitles_enabled": req.subtitles_enabled,
                        })
                        await database.save_job(j)

                    clip_result = PipelineClipResult(
                        clip_id=clip.id,
                        title=clip.title,
                        viral_score=clip.viral_score,
                        start_time=clip.start_time,
                        end_time=clip.end_time,
                        duration=round(clip.end_time - clip.start_time, 2),
                        download_url=f"/api/files/{job_id}/clips/{os.path.basename(output_path)}",
                    )

                    # Step 5: Generate SEO if requested
                    if req.generate_seo:
                        try:
                            _pipeline_status[pipeline_id]["progress_message"] = f"Generating SEO for clip {clip.id}..."
                            from backend.routers.clips import generate_seo_endpoint
                            seo_result = await generate_seo_endpoint(job_id, clip.id)
                            if isinstance(seo_result, dict) and "seo" in seo_result:
                                from backend.models import ClipSEO
                                clip_result.seo = ClipSEO(**seo_result["seo"])
                        except Exception as seo_err:
                            logger.warning("Pipeline SEO failed for clip %s: %s", clip.id, seo_err)

                    results.append(clip_result)

                except Exception as export_err:
                    logger.exception("Pipeline export failed for clip %s: %s", clip.id, export_err)

            _pipeline_status[pipeline_id]["clips"] = [r.model_dump() for r in results]
            _pipeline_status[pipeline_id]["status"] = "complete"
            _pipeline_status[pipeline_id]["progress"] = 100
            _pipeline_status[pipeline_id]["progress_message"] = f"Pipeline complete: {len(results)} clips exported"

            # Send webhook callback if configured
            if req.callback_url:
                await _send_callback(req.callback_url, {
                    "event": "pipeline_complete",
                    **_pipeline_status[pipeline_id],
                })

        except Exception as e:
            logger.exception("Pipeline failed: %s", e)
            _pipeline_status[pipeline_id]["status"] = "failed"
            _pipeline_status[pipeline_id]["error"] = str(e)
            _pipeline_status[pipeline_id]["progress_message"] = f"Pipeline failed: {str(e)}"

            if req.callback_url:
                await _send_callback(req.callback_url, {
                    "event": "pipeline_failed",
                    **_pipeline_status[pipeline_id],
                })
        finally:
            _active_pipelines.pop(pipeline_id, None)

    task = asyncio.create_task(_run())
    _active_pipelines[pipeline_id] = task

    return PipelineStartResponse(
        pipeline_id=pipeline_id,
        job_id="",  # Will be set once download completes
        status="running",
        message="Pipeline started. Poll GET /api/pipeline/{pipeline_id} for status.",
    )


@router.get(
    "/pipeline/{pipeline_id}",
    response_model=PipelineStatusResponse,
    summary="Poll pipeline progress",
)
async def get_pipeline_status(pipeline_id: str):
    """Check the status of a running pipeline. Returns clips with
    download URLs when the pipeline completes."""
    status = _pipeline_status.get(pipeline_id)
    if not status:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return PipelineStatusResponse(**status)


# ── Jobs with Pagination ─────────────────────────────────────────────

@router.get(
    "/jobs-paginated",
    response_model=JobListResponse,
    summary="List jobs with pagination and filtering",
)
async def list_jobs_paginated(
    limit: int = Query(20, ge=1, le=100, description="Max jobs to return"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip"),
    status: Optional[str] = Query(None, description="Filter by status (e.g. 'complete', 'running')"),
    sort: str = Query("created_at_desc", description="Sort order: created_at_desc, created_at_asc"),
):
    """List jobs with pagination, filtering, and sorting.
    More agent-friendly than the basic GET /api/jobs endpoint."""
    all_jobs = await database.list_jobs()

    # Filter by status
    if status:
        all_jobs = [j for j in all_jobs if j.status == status]

    # Sort
    if sort == "created_at_asc":
        all_jobs.sort(key=lambda j: j.created_at)
    else:
        all_jobs.sort(key=lambda j: j.created_at, reverse=True)

    total = len(all_jobs)
    page = all_jobs[offset:offset + limit]

    return JobListResponse(
        jobs=[
            JobSummaryResponse(
                job_id=j.job_id,
                filename=j.filename,
                duration=j.duration,
                status=j.status,
                progress=j.progress,
                progress_message=j.progress_message,
                created_at=j.created_at,
                clips_count=len(j.clips),
                provider_used=j.provider_used,
                file_size_mb=j.file_size_mb,
                estimated_cost_usd=j.estimated_cost_usd,
            )
            for j in page
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


# ── Download All Exports as ZIP ──────────────────────────────────────

@router.get(
    "/jobs/{job_id}/download-all",
    summary="Download all exported clips as a single ZIP file",
)
async def download_all_exports(job_id: str):
    """Download all exported clips for a job as a single ZIP file.
    Useful for agents that want to retrieve all outputs in one request."""
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.exported_clips:
        raise HTTPException(status_code=404, detail="No exported clips available")

    # Build ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for exp in job.exported_clips:
            filepath = exp.get("path", "")
            if filepath and os.path.isfile(filepath):
                arcname = exp.get("filename", os.path.basename(filepath))
                zf.write(filepath, arcname)

    zip_buffer.seek(0)
    zip_bytes = zip_buffer.getvalue()

    if len(zip_bytes) <= 22:  # Empty ZIP file
        raise HTTPException(status_code=404, detail="No clip files found on disk")

    base = job.filename.rsplit(".", 1)[0] if "." in job.filename else job.filename
    zip_filename = f"{base}_clips.zip"

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
    )


# ── Webhook Helper ───────────────────────────────────────────────────

async def _send_callback(url: str, payload: dict):
    """Send a webhook callback. Best-effort with retries."""
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code < 400:
                    logger.info("Webhook callback sent to %s (status %d)", url, resp.status_code)
                    return
                logger.warning("Webhook callback to %s returned %d", url, resp.status_code)
        except Exception as e:
            logger.warning("Webhook callback to %s failed (attempt %d): %s", url, attempt + 1, e)
        await asyncio.sleep(2 ** attempt)
