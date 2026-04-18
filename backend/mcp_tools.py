"""MCP tool definitions for the in-process FastMCP server.

These tools are thin wrappers around the same service-layer functions
the REST API endpoints call.  They are registered on the FastMCP instance
that is mounted at /mcp on the main FastAPI app.
"""

import logging
import os
from typing import Optional

from backend import database
from backend.config import settings
from backend.models import JobStatus

logger = logging.getLogger("clipai.mcp")


def register_tools(mcp):
    """Register all ClipAI tools and resources on a FastMCP instance."""

    # ── Video Management ──────────────────────────────────────────

    @mcp.tool()
    async def upload_video(
        url: str,
        language: str = "",
        auto_analyze: bool = True,
    ) -> dict:
        """Upload a video from a URL for AI analysis.

        Supports direct video URLs (.mp4, .mkv, .webm) and social media
        URLs (YouTube, TikTok, Instagram) if yt-dlp is installed.
        Deduplicates by content hash.

        Args:
            url: Video URL to download.
            language: ISO 639-1 language code (empty = auto-detect).
            auto_analyze: Start analysis immediately after download.

        Returns:
            dict with job_id, status, filename, deduplicated.
        """
        from backend.routers.agent import upload_from_url
        from backend.models_api import UploadUrlRequest
        from fastapi import BackgroundTasks

        req = UploadUrlRequest(url=url, language=language, auto_analyze=auto_analyze)
        bg = BackgroundTasks()
        result = await upload_from_url(req, bg)
        # Fire background tasks
        import asyncio
        for task in bg.tasks:
            asyncio.create_task(task["func"](*task["args"], **task["kwargs"]))
        return {
            "job_id": result.job_id,
            "status": result.status,
            "filename": result.filename,
            "deduplicated": result.deduplicated,
        }

    @mcp.tool()
    async def list_jobs(
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict:
        """List all video processing jobs with optional filtering.

        Args:
            status: Filter by status (e.g. "complete", "queued", "failed").
            limit: Max jobs to return.
            offset: Number of jobs to skip for pagination.

        Returns:
            dict with jobs array and total count.
        """
        all_jobs = await database.list_jobs()
        if status:
            all_jobs = [j for j in all_jobs if j.status == status]
        all_jobs.sort(key=lambda j: j.created_at, reverse=True)
        total = len(all_jobs)
        page = all_jobs[offset:offset + limit]
        return {
            "jobs": [
                {
                    "job_id": j.job_id, "filename": j.filename,
                    "duration": j.duration, "status": j.status,
                    "clip_count": len(j.clips), "created_at": j.created_at,
                }
                for j in page
            ],
            "total": total,
        }

    @mcp.tool()
    async def get_job_status(job_id: str) -> dict:
        """Get full details for a video analysis job.

        Args:
            job_id: The job ID returned from upload_video.

        Returns:
            Full job details including status, progress, clips, and errors.
        """
        job = await database.load_job(job_id)
        if not job:
            return {"error": "Job not found"}
        return {
            "job_id": job.job_id,
            "filename": job.filename,
            "status": job.status,
            "progress": job.progress,
            "progress_message": job.progress_message,
            "duration": job.duration,
            "clip_count": len(job.clips),
            "scene_count": len(job.scenes),
            "has_transcript": bool(job.transcript),
            "has_summary": bool(job.summary),
            "error": job.error,
        }

    @mcp.tool()
    async def delete_job(job_id: str) -> dict:
        """Delete a job and all associated files.

        Args:
            job_id: The job ID to delete.

        Returns:
            dict with success status.
        """
        job = await database.load_job(job_id)
        if not job:
            return {"error": "Job not found"}
        await database.delete_job(job_id)
        return {"success": True, "message": f"Job {job_id} deleted"}

    # ── Analysis ──────────────────────────────────────────────────

    @mcp.tool()
    async def start_analysis(job_id: str) -> dict:
        """Start or re-trigger the analysis pipeline for a video.

        This runs transcription, scene analysis, summary generation,
        and viral clip detection.

        Args:
            job_id: The job ID to analyze.

        Returns:
            dict with status confirmation.
        """
        import asyncio
        job = await database.load_job(job_id)
        if not job:
            return {"error": "Job not found"}

        from backend.services.pipeline import run_analysis
        asyncio.create_task(run_analysis(job_id))
        return {"job_id": job_id, "status": "analyzing", "message": "Analysis started"}

    @mcp.tool()
    async def get_summary(job_id: str) -> dict:
        """Get the AI-generated video summary.

        Args:
            job_id: The job ID.

        Returns:
            Video summary with overview, topics, tone, and category.
        """
        job = await database.load_job(job_id)
        if not job:
            return {"error": "Job not found"}
        if not job.summary:
            return {"error": "Summary not available yet"}
        return job.summary.model_dump()

    @mcp.tool()
    async def get_transcript(
        job_id: str,
        format: str = "json",
    ) -> dict:
        """Get the full transcript with timestamps and speaker labels.

        Args:
            job_id: The job ID.
            format: Output format: "json", "srt", "txt", "vtt".

        Returns:
            Transcript data in the requested format.
        """
        job = await database.load_job(job_id)
        if not job:
            return {"error": "Job not found"}
        if not job.transcript:
            return {"error": "Transcript not available yet"}

        if format == "srt":
            from backend.services.srt_generator import generate_srt
            return {"format": "srt", "content": generate_srt(job.transcript)}

        if format == "txt":
            lines = [f"[{s.speaker}] {s.text}" for s in job.transcript]
            return {"format": "txt", "content": "\n".join(lines)}

        return {
            "format": "json",
            "segments": [s.model_dump() for s in job.transcript],
        }

    @mcp.tool()
    async def get_scenes(job_id: str) -> dict:
        """Get all detected key scenes with timestamps and descriptions.

        Args:
            job_id: The job ID.

        Returns:
            Array of scene objects with timestamp, description, importance.
        """
        job = await database.load_job(job_id)
        if not job:
            return {"error": "Job not found"}
        return {"scenes": [s.model_dump() for s in job.scenes]}

    @mcp.tool()
    async def get_viral_clips(
        job_id: str,
        min_score: int = 0,
        min_duration: float = 0,
        max_duration: float = 9999,
    ) -> dict:
        """Get detected viral clip candidates with scores and metadata.

        Args:
            job_id: The job ID.
            min_score: Minimum viral score (0-100) to include.
            min_duration: Minimum clip duration in seconds.
            max_duration: Maximum clip duration in seconds.

        Returns:
            Array of clip candidates with title, viral_score, timing, etc.
        """
        job = await database.load_job(job_id)
        if not job:
            return {"error": "Job not found"}

        clips = job.clips
        if min_score > 0:
            clips = [c for c in clips if c.viral_score >= min_score]
        if min_duration > 0:
            clips = [c for c in clips if c.duration >= min_duration]
        if max_duration < 9999:
            clips = [c for c in clips if c.duration <= max_duration]

        return {"clips": [c.model_dump() for c in clips]}

    # ── Clip Operations ───────────────────────────────────────────

    @mcp.tool()
    async def configure_clip(
        job_id: str,
        clip_index: int,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> dict:
        """Update timing for a specific clip candidate.

        Args:
            job_id: The job ID.
            clip_index: The clip ID/index to configure.
            start_time: New start time in seconds.
            end_time: New end time in seconds.

        Returns:
            dict with applied settings.
        """
        job = await database.load_job(job_id)
        if not job:
            return {"error": "Job not found"}
        clip = next((c for c in job.clips if c.id == clip_index), None)
        if not clip:
            return {"error": f"Clip {clip_index} not found"}

        applied = {}
        if start_time is not None:
            clip.start_time = start_time
            applied["start_time"] = start_time
        if end_time is not None:
            clip.end_time = end_time
            applied["end_time"] = end_time
        if applied:
            clip.duration = round(clip.end_time - clip.start_time, 2)
            applied["duration"] = clip.duration
            await database.save_job(job)

        return {"success": True, "clip_id": clip_index, "applied_settings": applied}

    @mcp.tool()
    async def export_clip(
        job_id: str,
        clip_index: int,
        aspect_ratio: str = "9:16",
        export_quality: str = "1080p",
        subtitles_enabled: bool = True,
    ) -> dict:
        """Export a video clip with customization settings.

        Args:
            job_id: The job ID.
            clip_index: The clip ID to export.
            aspect_ratio: Target ratio: "9:16", "1:1", "16:9", "4:5".
            export_quality: Quality: "720p", "1080p", "4k".
            subtitles_enabled: Burn subtitles into the video.

        Returns:
            dict with export_id and status.
        """
        from backend.models import ExportRequest
        from backend.routers.clips import export_clip_endpoint

        job = await database.load_job(job_id)
        if not job:
            return {"error": "Job not found"}
        clip = next((c for c in job.clips if c.id == clip_index), None)
        if not clip:
            return {"error": f"Clip {clip_index} not found"}

        req = ExportRequest(
            start=clip.start_time, end=clip.end_time,
            clip_id=clip_index, clip_title=clip.title,
            aspect_ratio=aspect_ratio, subtitles_enabled=subtitles_enabled,
            export_quality=export_quality,
        )
        await export_clip_endpoint(job_id, req)
        return {
            "export_id": f"{job_id}_{clip_index}",
            "status": "encoding",
            "message": f"Exporting clip '{clip.title}'",
        }

    @mcp.tool()
    async def get_export_status(export_id: str) -> dict:
        """Check export progress. Returns download_url when complete.

        Args:
            export_id: The export ID (format: job_id_clip_id).

        Returns:
            dict with status, progress, and download_url if complete.
        """
        parts = export_id.split("_", 1)
        if len(parts) != 2:
            return {"error": "Invalid export_id format (expected: job_id_clip_id)"}

        job_id, clip_id_str = parts
        try:
            clip_id = int(clip_id_str)
        except ValueError:
            return {"error": "Invalid clip_id in export_id"}

        from backend.routers.agent import get_export_status as agent_es
        result = await agent_es(job_id, clip_id)
        return result.model_dump()

    @mcp.tool()
    async def download_clip(export_id: str) -> dict:
        """Get the download URL for an exported clip.

        Args:
            export_id: The export ID (format: job_id_clip_id).

        Returns:
            dict with download_url if the export is complete.
        """
        parts = export_id.split("_", 1)
        if len(parts) != 2:
            return {"error": "Invalid export_id"}

        job_id, clip_id_str = parts
        try:
            clip_id = int(clip_id_str)
        except ValueError:
            return {"error": "Invalid clip_id"}

        job = await database.load_job(job_id)
        if not job:
            return {"error": "Job not found"}

        for exp in reversed(job.exported_clips):
            if exp.get("clip_id") == clip_id:
                return {
                    "download_url": f"/api/files/{job_id}/clips/{exp['filename']}",
                    "filename": exp.get("filename"),
                }

        return {"error": "Export not found"}

    # ── SEO ───────────────────────────────────────────────────────

    @mcp.tool()
    async def generate_seo(
        job_id: str,
        clip_index: int,
        type: str = "all",
    ) -> dict:
        """Generate SEO metadata for a clip using AI.

        Args:
            job_id: The job ID.
            clip_index: The clip ID.
            type: What to generate: "shorts_description", "youtube_description", "title", "tags", "all".

        Returns:
            Generated SEO content.
        """
        has_any_key = any([
            settings.OPENROUTER_API_KEY, settings.ANTHROPIC_API_KEY,
            settings.GEMINI_API_KEY, settings.GROQ_API_KEY,
        ])
        if not has_any_key:
            return {"error": "No AI provider API key configured. Add your key in Settings."}

        result = {}

        if type in ("all", "title", "tags"):
            try:
                from backend.routers.clips import generate_seo_endpoint
                seo = await generate_seo_endpoint(job_id, clip_index)
                if isinstance(seo, dict):
                    result["seo"] = seo.get("seo")
            except Exception as e:
                result["seo_error"] = str(e)

        if type in ("all", "shorts_description"):
            try:
                from backend.routers.clips import generate_description_endpoint, GenerateDescriptionRequest
                desc = await generate_description_endpoint(
                    job_id, clip_index, GenerateDescriptionRequest(description_type="shorts"),
                )
                if isinstance(desc, dict):
                    result["shorts_description"] = desc.get("description")
            except Exception as e:
                result["shorts_error"] = str(e)

        if type in ("all", "youtube_description"):
            try:
                from backend.routers.clips import generate_description_endpoint, GenerateDescriptionRequest
                desc = await generate_description_endpoint(
                    job_id, clip_index, GenerateDescriptionRequest(description_type="long_form"),
                )
                if isinstance(desc, dict):
                    result["youtube_description"] = desc.get("description")
            except Exception as e:
                result["youtube_error"] = str(e)

        return result

    @mcp.tool()
    async def get_seo_metadata(job_id: str, clip_index: int) -> dict:
        """Get clip info that can be used for SEO purposes.

        Args:
            job_id: The job ID.
            clip_index: The clip ID.

        Returns:
            Clip title, caption, and platform info.
        """
        job = await database.load_job(job_id)
        if not job:
            return {"error": "Job not found"}
        clip = next((c for c in job.clips if c.id == clip_index), None)
        if not clip:
            return {"error": f"Clip {clip_index} not found"}
        return {
            "clip_id": clip_index,
            "title": clip.title,
            "suggested_caption": clip.suggested_caption,
            "platform": clip.platform,
            "viral_score": clip.viral_score,
        }

    # ── Settings ──────────────────────────────────────────────────

    @mcp.tool()
    async def get_settings() -> dict:
        """Get current ClipAI settings. API keys are masked.

        Returns:
            Current configuration including provider keys (masked) and defaults.
        """
        from backend.auth import mask_key
        return {
            "openrouter_key_set": bool(settings.OPENROUTER_API_KEY),
            "anthropic_key_set": bool(settings.ANTHROPIC_API_KEY),
            "gemini_key_set": bool(settings.GEMINI_API_KEY),
            "groq_key_set": bool(settings.GROQ_API_KEY),
            "whisper_model": settings.WHISPER_MODEL,
            "preset": settings.OPENROUTER_PRESET,
            "subject_tracking": settings.SUBJECT_TRACKING_ENABLED,
            "auto_analyze": settings.AUTO_ANALYZE,
        }

    @mcp.tool()
    async def update_settings(
        openrouter_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        preset: Optional[str] = None,
    ) -> dict:
        """Update ClipAI settings (API keys, preset, etc.).

        Args:
            openrouter_api_key: OpenRouter API key.
            anthropic_api_key: Anthropic API key.
            gemini_api_key: Google Gemini API key.
            groq_api_key: Groq API key.
            preset: AI preset: "free", "efficient", "balanced", "premium".

        Returns:
            dict with updated keys.
        """
        from backend.routers.settings import _persist_user_settings
        updated = []
        if openrouter_api_key:
            settings.OPENROUTER_API_KEY = openrouter_api_key
            updated.append("openrouter_api_key")
        if anthropic_api_key:
            settings.ANTHROPIC_API_KEY = anthropic_api_key
            updated.append("anthropic_api_key")
        if gemini_api_key:
            settings.GEMINI_API_KEY = gemini_api_key
            updated.append("gemini_api_key")
        if groq_api_key:
            settings.GROQ_API_KEY = groq_api_key
            updated.append("groq_api_key")
        if preset:
            settings.OPENROUTER_PRESET = preset
            updated.append("preset")
        if updated:
            _persist_user_settings()
        return {"success": True, "updated_keys": updated}

    @mcp.tool()
    async def get_provider_status() -> dict:
        """Check which AI providers are configured and available.

        Returns:
            Status of each provider: "configured", "not_configured", or "connected".
        """
        import httpx
        ollama_status = "not_configured"
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                resp = await client.get(f"{settings.OLLAMA_HOST}/api/version")
                ollama_status = "connected" if resp.status_code == 200 else "error"
        except Exception:
            pass

        return {
            "openrouter": "configured" if settings.OPENROUTER_API_KEY else "not_configured",
            "anthropic": "configured" if settings.ANTHROPIC_API_KEY else "not_configured",
            "gemini": "configured" if settings.GEMINI_API_KEY else "not_configured",
            "groq": "configured" if settings.GROQ_API_KEY else "not_configured",
            "ollama": ollama_status,
        }

    # ── Batch / Workflow ──────────────────────────────────────────

    @mcp.tool()
    async def process_video_full(
        url: str,
        auto_export_top: int = 3,
        aspect_ratio: str = "9:16",
        subtitles: bool = True,
        export_quality: str = "1080p",
    ) -> dict:
        """Full end-to-end pipeline: upload, analyze, and export top clips.

        This is a convenience tool that chains the entire workflow.
        Returns immediately with a pipeline_id for polling.

        Args:
            url: Video URL to process.
            auto_export_top: Number of top clips to export.
            aspect_ratio: Export aspect ratio.
            subtitles: Burn subtitles into clips.
            export_quality: Export quality.

        Returns:
            dict with pipeline_id for status polling via get_pipeline_status.
        """
        from backend.models_api import PipelineRequest
        from backend.routers.agent import run_pipeline
        from fastapi import BackgroundTasks

        req = PipelineRequest(
            url=url,
            auto_select="all" if auto_export_top > 1 else "top",
            clip_count=auto_export_top,
            aspect_ratio=aspect_ratio,
            subtitles_enabled=subtitles,
            export_quality=export_quality,
        )
        bg = BackgroundTasks()
        result = await run_pipeline(req, bg)
        return result.model_dump()

    @mcp.tool()
    async def batch_export(
        job_id: str,
        clip_indices: Optional[list[int]] = None,
        aspect_ratio: str = "9:16",
        export_quality: str = "1080p",
        subtitles_enabled: bool = True,
    ) -> dict:
        """Export multiple clips from the same video.

        Args:
            job_id: The job ID.
            clip_indices: List of clip IDs to export. Empty/None = all clips.
            aspect_ratio: Export aspect ratio.
            export_quality: Export quality.
            subtitles_enabled: Burn subtitles.

        Returns:
            dict with batch_id for status polling.
        """
        from backend.models_api import BatchExportRequest
        from backend.routers.agent import batch_export as do_batch

        req = BatchExportRequest(
            clip_ids=clip_indices or [],
            aspect_ratio=aspect_ratio,
            export_quality=export_quality,
            subtitles_enabled=subtitles_enabled,
        )
        result = await do_batch(job_id, req)
        return result.model_dump()

    # ── MCP Resources ─────────────────────────────────────────────

    @mcp.resource("clipai://jobs")
    async def resource_list_jobs() -> str:
        """List all video processing jobs."""
        all_jobs = await database.list_jobs()
        import json
        return json.dumps([
            {"job_id": j.job_id, "filename": j.filename, "status": j.status, "clip_count": len(j.clips)}
            for j in all_jobs
        ], default=str)

    @mcp.resource("clipai://jobs/{job_id}")
    async def resource_get_job(job_id: str) -> str:
        """Get full details for a specific job."""
        job = await database.load_job(job_id)
        if not job:
            return '{"error": "Job not found"}'
        import json
        return json.dumps(job.model_dump(), default=str)

    @mcp.resource("clipai://jobs/{job_id}/transcript")
    async def resource_get_transcript(job_id: str) -> str:
        """Get the transcript for a specific job."""
        job = await database.load_job(job_id)
        if not job:
            return '{"error": "Job not found"}'
        import json
        return json.dumps([s.model_dump() for s in job.transcript], default=str)

    @mcp.resource("clipai://settings")
    async def resource_get_settings() -> str:
        """Get current ClipAI configuration."""
        import json
        return json.dumps({
            "openrouter_key_set": bool(settings.OPENROUTER_API_KEY),
            "anthropic_key_set": bool(settings.ANTHROPIC_API_KEY),
            "whisper_model": settings.WHISPER_MODEL,
            "preset": settings.OPENROUTER_PRESET,
        }, default=str)
