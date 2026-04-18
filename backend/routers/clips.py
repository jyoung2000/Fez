import asyncio
import logging
import os
import time
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend import database
from backend.app.auth.deps import get_current_user, require_job_access
from backend.app.auth.models import User
from backend.config import settings
from backend.models import ExportRequest, FullVideoExportRequest, GenerateClipsRequest, TranslateRequest, TranscriptSegment, UpdateClipTimesRequest, UpdateClipTitleRequest
from backend.services.clip_exporter import export_clip
from backend.services.pipeline import broadcast_ws

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["clips"])


def _seg_attr(seg, key: str, default=None):
    """Get attribute from a segment (dict or Pydantic model)."""
    if isinstance(seg, dict):
        return seg.get(key, default)
    return getattr(seg, key, default)


def _compute_output_duration(
    start: float,
    end: float,
    global_speed: float,
    segments: list | None = None,
) -> float:
    """Compute expected output duration accounting for per-segment speeds.

    When segments have individual speed overrides, each segment's output
    duration is its input duration divided by its speed.  Gap regions
    (not covered by any segment) use the global speed.

    This matches the logic in clip_exporter._build_speed_timeline so the
    stored duration record matches the actual exported file.
    """
    clip_dur = end - start
    speed = global_speed if global_speed else 1.0
    if not segments:
        return clip_dur / speed

    # Check if any segment has a per-segment speed override
    has_seg_speed = any(
        abs(_seg_attr(s, "speed", 1.0) - 1.0) > 0.001
        for s in segments
    )
    if not has_seg_speed:
        return clip_dur / speed

    # Walk through segments and compute output duration per region
    sorted_segs = sorted(segments, key=lambda s: _seg_attr(s, "start", 0))
    total_output = 0.0
    pos = 0.0  # clip-relative position

    for seg in sorted_segs:
        seg_start = max(0.0, _seg_attr(seg, "start", 0) - start)
        seg_end = min(clip_dur, _seg_attr(seg, "end", 0) - start)
        if seg_end <= seg_start:
            continue

        # Gap before this segment — uses global speed
        if seg_start > pos + 0.01:
            total_output += (seg_start - pos) / speed

        seg_speed = _seg_attr(seg, "speed", speed)
        total_output += (seg_end - seg_start) / seg_speed
        pos = seg_end

    # Gap after last segment
    if pos < clip_dur - 0.01:
        total_output += (clip_dur - pos) / speed

    return total_output

# Track active clip generation tasks per job so we can cancel on re-trigger
_active_clip_tasks: dict[str, asyncio.Task] = {}
_clip_cancel_events: dict[str, asyncio.Event] = {}

# Track active export tasks and their cancellation events
_active_export_tasks: dict[str, asyncio.Task] = {}
_export_cancel_events: dict[str, asyncio.Event] = {}

# Timeout for the AI clip detection call (15 minutes — large videos with
# many transcript segments and scenes need more time for AI analysis)
_CLIP_DETECTION_TIMEOUT = 900


@router.post("/jobs/{job_id}/export-clip")
async def export_clip_endpoint(
    job_id: str,
    req: ExportRequest,
    user: User = Depends(get_current_user),
):
    job = await require_job_access(job_id, user)

    # Populate hook_text from stored clip data if not already set
    if not req.hook_text and job.clips:
        matching_clip = next(
            (c for c in job.clips if c.id == req.clip_id), None
        )
        if matching_clip and matching_clip.hook_text:
            req.hook_text = matching_clip.hook_text

    # Parse video resolution for crop/subtitle positioning
    vid_w, vid_h = 1920, 1080
    if job.resolution:
        try:
            parts = job.resolution.split("x")
            vid_w, vid_h = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            pass

    # Gather scenes for subject tracking — include nearest boundary scenes
    # so _build_subject_keyframes can interpolate at clip edges rather than
    # falling back to center (50) when no scenes fall strictly within range.
    clip_subject_x = 50
    clip_scenes = []
    logger.info(
        "[SubjectTracking] ═══ EXPORT REQUEST clip %s: %.1f-%.1fs, aspect=%s, quality=%s, tracking_enabled=%s, total_scenes=%d ═══",
        req.clip_id, req.start, req.end, req.aspect_ratio or "original",
        req.export_quality or "1080p",
        settings.SUBJECT_TRACKING_ENABLED,
        len(job.scenes) if job.scenes else 0,
    )
    if settings.SUBJECT_TRACKING_ENABLED and job.scenes:
        in_range = [s for s in job.scenes if req.start <= s.timestamp <= req.end]
        before = [s for s in job.scenes if s.timestamp < req.start]
        after = [s for s in job.scenes if s.timestamp > req.end]
        nearest_before = max(before, key=lambda s: s.timestamp) if before else None
        nearest_after = min(after, key=lambda s: s.timestamp) if after else None

        # Build scene list: nearest-before + in-range + nearest-after
        clip_scenes = []
        if nearest_before:
            clip_scenes.append(nearest_before)
        clip_scenes.extend(in_range)
        if nearest_after:
            clip_scenes.append(nearest_after)

        logger.info(
            "[SubjectTracking] clip %s: gathered %d scenes (in_range=%d, boundary=%d) — timestamps: %s",
            req.clip_id, len(clip_scenes), len(in_range),
            (1 if nearest_before else 0) + (1 if nearest_after else 0),
            [f"t={s.timestamp:.1f},sx={s.subject_x}" for s in clip_scenes],
        )

        if in_range:
            clip_subject_x = round(sum(s.subject_x for s in in_range) / len(in_range))
            sx_values = [s.subject_x for s in in_range]
            logger.info(
                "[SubjectTracking] clip %s: %d in-range scenes, subject_x range [%d, %d], avg=%d",
                req.clip_id, len(in_range),
                min(sx_values), max(sx_values), clip_subject_x,
            )
        elif nearest_before and nearest_after:
            # No scenes within range — interpolate from boundary scenes
            mid = (req.start + req.end) / 2
            dt = nearest_after.timestamp - nearest_before.timestamp
            if dt > 0:
                frac = (mid - nearest_before.timestamp) / dt
                clip_subject_x = round(nearest_before.subject_x + (nearest_after.subject_x - nearest_before.subject_x) * frac)
            else:
                clip_subject_x = nearest_before.subject_x
            logger.info(
                "[SubjectTracking] clip %s: no in-range scenes, interpolated subject_x=%d from boundary (before=t%.1f/sx=%d, after=t%.1f/sx=%d)",
                req.clip_id, clip_subject_x,
                nearest_before.timestamp, nearest_before.subject_x,
                nearest_after.timestamp, nearest_after.subject_x,
            )
        elif nearest_before:
            clip_subject_x = nearest_before.subject_x
            logger.info(
                "[SubjectTracking] clip %s: no in-range scenes, using nearest before scene (t=%.1f, sx=%d)",
                req.clip_id, nearest_before.timestamp, clip_subject_x,
            )
        elif nearest_after:
            clip_subject_x = nearest_after.subject_x
            logger.info(
                "[SubjectTracking] clip %s: no in-range scenes, using nearest after scene (t=%.1f, sx=%d)",
                req.clip_id, nearest_after.timestamp, clip_subject_x,
            )
        else:
            logger.info("[SubjectTracking] clip %s: no scenes available — using default center (50)", req.clip_id)
    elif not settings.SUBJECT_TRACKING_ENABLED:
        logger.info("[SubjectTracking] clip %s: TRACKING DISABLED — using default center crop", req.clip_id)
    else:
        logger.info("[SubjectTracking] clip %s: no scene data in job — using default center crop", req.clip_id)

    export_key = f"{job_id}_{req.clip_id}"

    # Prevent duplicate exports: if an export for this clip is already
    # in progress, reject the second request instead of creating a
    # parallel task that would race and double-export.
    existing_task = _active_export_tasks.get(export_key)
    if existing_task and not existing_task.done():
        logger.info("Export already in progress for %s — ignoring duplicate request", export_key)
        return {"export_id": export_key, "status": "exporting", "duplicate": True}

    cancel_event = asyncio.Event()
    _export_cancel_events[export_key] = cancel_event

    # Apply VideoEditor trim offsets
    actual_start = req.start + req.trim_start_offset
    actual_end = req.end - req.trim_end_offset

    # Adjust overlay times to account for trim offset.
    # Overlay times from the frontend are computed as (item.start + req.start),
    # but the backend uses actual_start (= req.start + trim_start_offset) as
    # clip_start for overlay positioning.  Without this correction, overlays
    # would be shifted earlier by trim_start_offset seconds.
    if req.trim_start_offset:
        _trim = req.trim_start_offset
        for _ol_list in (req.text_overlays, req.image_overlays,
                         req.shape_overlays, req.audio_overlays):
            if _ol_list:
                for _ol in _ol_list:
                    _ol.start_time += _trim
                    _ol.end_time += _trim

    # Warn if any image/audio overlay src looks like a blob URL (client-only)
    for _ol_list, _label in ((req.image_overlays, "image"), (req.audio_overlays, "audio")):
        if _ol_list:
            for _ol in _ol_list:
                _src = getattr(_ol, "src", "") or ""
                if _src.startswith("blob:"):
                    logger.warning(
                        "Export request for clip %s has %s overlay with blob URL src=%s — "
                        "this cannot be resolved server-side and will be skipped",
                        req.clip_id, _label, _src[:80],
                    )

    # ISSUE 14: Warn when export request is missing expected editor data.
    # If the clip was edited but the request has no effects/overlays, the
    # export won't match the preview.
    if (not req.segments and not req.video_effects
            and not req.text_overlays and not req.image_overlays
            and not req.audio_overlays):
        logger.warning(
            "Export request for clip %s has no segments, video_effects, "
            "text_overlays, image_overlays, or audio_overlays. If the user "
            "edited this clip in the VideoEditor, the export may not match "
            "the preview. Ensure the frontend sends all editor state.",
            req.clip_id,
        )

    async def _do_export():
        try:
            export_start = time.monotonic()
            clip_dur = actual_end - actual_start

            await broadcast_ws(job_id, {
                "type": "status",
                "status": "exporting",
                "progress": 0,
                "message": f"Starting export for clip {req.clip_id} ({clip_dur:.1f}s) [{req.export_quality or '1080p'}]",
            })

            # Pre-flight diagnostic logging
            logger.info(
                "EXPORT PRE-FLIGHT clip %s: text_overlays=%d, image_overlays=%d, "
                "shape_overlays=%d, audio_overlays=%d, video_effects=%s, "
                "subtitles=%s, aspect=%s, quality=%s, trim=(%.2f, %.2f)",
                req.clip_id,
                len(req.text_overlays or []),
                len(req.image_overlays or []),
                len(req.shape_overlays or []),
                len(req.audio_overlays or []),
                bool(req.video_effects),
                req.subtitles_enabled,
                req.aspect_ratio,
                req.export_quality,
                req.trim_start_offset,
                req.trim_end_offset,
            )
            for ti, t_ov in enumerate(req.text_overlays or []):
                logger.info("  TEXT[%d]: text=%r pos=(%.0f%%,%.0f%%) time=%.1f-%.1f",
                            ti, (t_ov.text or "")[:30], t_ov.x, t_ov.y, t_ov.start_time, t_ov.end_time)
            for ii, i_ov in enumerate(req.image_overlays or []):
                logger.info("  IMAGE[%d]: src=%r pos=(%.0f%%,%.0f%%) time=%.1f-%.1f",
                            ii, (i_ov.src or "")[:60], i_ov.x, i_ov.y, i_ov.start_time, i_ov.end_time)
            for ai, a_ov in enumerate(req.audio_overlays or []):
                logger.info("  AUDIO[%d]: src=%r time=%.1f-%.1f vol=%.1f",
                            ai, (a_ov.src or "")[:60], a_ov.start_time, a_ov.end_time, a_ov.volume)

            # Log subject tracking status for this export
            if settings.SUBJECT_TRACKING_ENABLED and clip_scenes:
                tracking_type = "dynamic" if len(clip_scenes) > 1 else "static"
                await broadcast_ws(job_id, {
                    "type": "subject_tracking",
                    "enabled": True,
                    "tracked_scenes": len(clip_scenes),
                    "message": f"Intelligent Dynamic Subject Tracking: AI is centering the subject in frame using {len(clip_scenes)} tracked positions ({tracking_type} crop)",
                })
                await broadcast_ws(job_id, {
                    "type": "status",
                    "status": "exporting",
                    "progress": 25,
                    "message": f"AI Subject Tracking: centering subject in {req.aspect_ratio or 'original'} frame (subject_x={clip_subject_x}, {len(clip_scenes)} scene positions)",
                })
            elif not settings.SUBJECT_TRACKING_ENABLED:
                await broadcast_ws(job_id, {
                    "type": "subject_tracking",
                    "enabled": False,
                    "message": "Subject tracking disabled — using center crop",
                })
            else:
                await broadcast_ws(job_id, {
                    "type": "status",
                    "status": "exporting",
                    "progress": 25,
                    "message": f"No subject tracking data available — using center crop for {req.aspect_ratio or 'original'} frame",
                })

            async def _export_progress(msg: str):
                # Extract real encoding percentage from message text.
                # When the message has no percentage (e.g. "Preparing clip..."
                # or "Subtitle file generated..."), omit the progress field so
                # the frontend keeps the previous value instead of jumping.
                import re as _re
                _m = _re.search(r'(\d+)%', msg)
                payload = {
                    "type": "status",
                    "status": "exporting",
                    "message": msg,
                }
                if _m:
                    payload["progress"] = int(_m.group(1))
                await broadcast_ws(job_id, payload)

            output_path = await export_clip(
                job_id=job_id,
                video_path=job.file_path,
                start=actual_start,
                end=actual_end,
                clip_id=req.clip_id,
                clip_title=req.clip_title,
                aspect_ratio=req.aspect_ratio,
                subtitles_enabled=req.subtitles_enabled,
                global_subtitles_enabled=req.global_subtitles_enabled,
                subtitle_settings=req.subtitle_settings.model_dump() if req.subtitle_settings else None,
                transcript=[s.model_dump() for s in (req.edited_subtitle_segments or (job.translated_transcript if job.translated_transcript else job.transcript))] if req.subtitles_enabled else None,
                video_width=vid_w,
                video_height=vid_h,
                subject_x=clip_subject_x,
                subject_scenes=clip_scenes or None,
                scene_cut_timestamps=getattr(job, 'scene_cut_timestamps', None) or None,
                progress_callback=_export_progress,
                cancel_event=cancel_event,
                export_quality=req.export_quality or "1080p",
                volume=req.volume,
                speed=req.speed,
                segments=[s.model_dump() for s in req.segments] if req.segments else None,
                video_effects=req.video_effects.model_dump() if req.video_effects else None,
                text_overlays=[t.model_dump() for t in req.text_overlays] if req.text_overlays else None,
                image_overlays=[i.model_dump() for i in req.image_overlays] if req.image_overlays else None,
                shape_overlays=[s.model_dump() for s in req.shape_overlays] if req.shape_overlays else None,
                audio_overlays=[a.model_dump() for a in req.audio_overlays] if req.audio_overlays else None,
                overlay_compositing_order=req.overlay_compositing_order if req.overlay_compositing_order else None,
                layout_mode=getattr(req, 'layout_mode', 'auto'),
                pip_position=getattr(req, 'pip_position', 'bottom_right'),
                pip_size_pct=getattr(req, 'pip_size_pct', 25.0),
                face_registry_data=getattr(job, 'face_registry_data', None),
                layout_timeline_data=getattr(job, 'layout_timeline', None),
                hook_text=req.hook_text,
                frontend_subject_keyframes=req.subject_keyframes,
            )

            elapsed = int(time.monotonic() - export_start)

            # Update job record
            j = await database.load_job(job_id)
            if j:
                from datetime import datetime, timezone
                j.exported_clips.append({
                    "clip_id": req.clip_id,
                    "path": output_path,
                    "filename": os.path.basename(output_path),
                    "title": req.clip_title or f"Clip {req.clip_id}",
                    "start": actual_start,
                    "end": actual_end,
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "duration": round(_compute_output_duration(
                        actual_start, actual_end, req.speed, req.segments,
                    ), 2),
                    "export_quality": req.export_quality or "1080p",
                    "aspect_ratio": req.aspect_ratio,
                    "subtitles_enabled": req.subtitles_enabled,
                    "subtitle_settings": req.subtitle_settings.model_dump() if req.subtitle_settings else None,
                    "volume": req.volume,
                    "speed": req.speed,
                    # Per-user attribution so the Exports tab only
                    # surfaces clips this user actually exported (admins
                    # can see every job but should not see every other
                    # user's exports of the same job in their own list).
                    "exported_by_user_id": user.id,
                    "exported_by_username": user.username,
                })
                await database.save_job(j)

            await broadcast_ws(job_id, {
                "type": "export_complete",
                "clip_id": req.clip_id,
                "download_url": f"/api/files/{job_id}/clips/{quote(os.path.basename(output_path))}?t={int(time.time())}",
                "message": f"Clip {req.clip_id} exported in {elapsed}s [{req.export_quality or '1080p'}]",
                "qa_passed": True,
            })
        except asyncio.CancelledError:
            await broadcast_ws(job_id, {
                "type": "error",
                "message": f"Clip {req.clip_id} export cancelled",
            })
        except Exception as e:
            logger.exception("Export failed for clip %s in job %s", req.clip_id, job_id)
            await broadcast_ws(job_id, {
                "type": "error",
                "message": f"Clip {req.clip_id} export failed: {str(e)}",
            })
        finally:
            _active_export_tasks.pop(export_key, None)
            _export_cancel_events.pop(export_key, None)

    task = asyncio.create_task(_do_export())
    _active_export_tasks[export_key] = task
    return {"export_id": export_key, "status": "exporting"}


@router.post("/jobs/{job_id}/cancel-export/{clip_id}")
async def cancel_export_endpoint(job_id: str, clip_id: int, user: User = Depends(get_current_user)):
    """Cancel an in-progress clip export."""
    export_key = f"{job_id}_{clip_id}"

    cancel_event = _export_cancel_events.get(export_key)
    if cancel_event:
        cancel_event.set()

    task = _active_export_tasks.get(export_key)
    if not task or task.done():
        raise HTTPException(status_code=404, detail="No active export found for this clip")

    # Give the task a moment to handle cancellation gracefully
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
    except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
        pass

    return {"export_id": export_key, "status": "cancelled"}


@router.post("/jobs/{job_id}/export-full-video")
async def export_full_video_endpoint(
    job_id: str,
    req: FullVideoExportRequest,
    user: User = Depends(get_current_user),
):
    """Export the entire video with clip settings (aspect ratio, subtitles, subject tracking) applied."""
    job = await require_job_access(job_id, user)
    if not job.file_path or not job.duration:
        raise HTTPException(status_code=400, detail="Video file or duration not available")

    vid_w, vid_h = 1920, 1080
    if job.resolution:
        try:
            parts = job.resolution.split("x")
            vid_w, vid_h = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            pass

    # Use all scenes for subject tracking across the full video
    full_subject_x = 50
    full_scenes = []
    if settings.SUBJECT_TRACKING_ENABLED and job.scenes:
        full_scenes = list(job.scenes)
        sx_values = [s.subject_x for s in full_scenes if hasattr(s, "subject_x")]
        if sx_values:
            full_subject_x = round(sum(sx_values) / len(sx_values))

    # Use clip_id=0 to denote full-video export (matches frontend's ${jobId}_0 convention)
    export_key = f"{job_id}_0"

    # Prevent duplicate exports
    existing_task = _active_export_tasks.get(export_key)
    if existing_task and not existing_task.done():
        logger.info("Full video export already in progress for %s — ignoring duplicate request", export_key)
        return {"export_id": export_key, "status": "exporting", "duplicate": True}

    cancel_event = asyncio.Event()
    _export_cancel_events[export_key] = cancel_event

    async def _do_export():
        try:
            export_start = time.monotonic()
            await broadcast_ws(job_id, {
                "type": "status",
                "status": "exporting",
                "progress": 0,
                "message": f"Starting full video export ({job.duration:.0f}s) [{req.export_quality or '1080p'}]",
            })

            if settings.SUBJECT_TRACKING_ENABLED and full_scenes:
                tracking_type = "dynamic" if len(full_scenes) > 1 else "static"
                await broadcast_ws(job_id, {
                    "type": "subject_tracking",
                    "enabled": True,
                    "tracked_scenes": len(full_scenes),
                    "message": f"Intelligent Dynamic Subject Tracking: centering subject using {len(full_scenes)} tracked positions ({tracking_type} crop)",
                })

            async def _export_progress(msg: str):
                import re as _re
                _m = _re.search(r'(\d+)%', msg)
                payload = {
                    "type": "status",
                    "status": "exporting",
                    "message": msg,
                }
                if _m:
                    payload["progress"] = int(_m.group(1))
                await broadcast_ws(job_id, payload)

            # Apply VideoEditor trim offsets for full video
            fv_start = 0 + req.trim_start_offset
            fv_end = job.duration - req.trim_end_offset

            output_path = await export_clip(
                job_id=job_id,
                video_path=job.file_path,
                start=fv_start,
                end=fv_end,
                clip_id=0,
                clip_title=os.path.splitext(job.filename or "full_video")[0],
                aspect_ratio=req.aspect_ratio,
                subtitles_enabled=req.subtitles_enabled,
                global_subtitles_enabled=req.global_subtitles_enabled,
                subtitle_settings=req.subtitle_settings.model_dump() if req.subtitle_settings else None,
                transcript=[s.model_dump() for s in (req.edited_subtitle_segments or (job.translated_transcript if job.translated_transcript else job.transcript))] if req.subtitles_enabled and (req.edited_subtitle_segments or job.translated_transcript or job.transcript) else None,
                video_width=vid_w,
                video_height=vid_h,
                subject_x=full_subject_x,
                subject_scenes=full_scenes or None,
                scene_cut_timestamps=getattr(job, 'scene_cut_timestamps', None) or None,
                progress_callback=_export_progress,
                cancel_event=cancel_event,
                export_quality=req.export_quality or "1080p",
                volume=req.volume,
                speed=req.speed,
                segments=[s.model_dump() for s in req.segments] if req.segments else None,
                video_effects=req.video_effects.model_dump() if req.video_effects else None,
                text_overlays=[t.model_dump() for t in req.text_overlays] if req.text_overlays else None,
                image_overlays=[i.model_dump() for i in req.image_overlays] if req.image_overlays else None,
                shape_overlays=[s.model_dump() for s in req.shape_overlays] if req.shape_overlays else None,
                audio_overlays=[a.model_dump() for a in req.audio_overlays] if req.audio_overlays else None,
                # Pass the global compositing order, layout/PIP picks
                # and hook text — previously dropped from full-video
                # exports because the model didn't declare them.
                overlay_compositing_order=req.overlay_compositing_order or None,
                layout_mode=req.layout_mode,
                pip_position=req.pip_position,
                pip_size_pct=req.pip_size_pct,
                hook_text=req.hook_text or "",
                frontend_subject_keyframes=req.subject_keyframes,
            )

            elapsed = int(time.monotonic() - export_start)

            j = await database.load_job(job_id)
            if j:
                from datetime import datetime, timezone
                j.exported_clips.append({
                    "clip_id": 0,
                    "path": output_path,
                    "filename": os.path.basename(output_path),
                    "title": os.path.splitext(job.filename or "full_video")[0],
                    "start": fv_start,
                    "end": fv_end,
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "duration": round(_compute_output_duration(
                        fv_start, fv_end, req.speed, req.segments,
                    ), 2),
                    "export_quality": req.export_quality or "1080p",
                    "aspect_ratio": req.aspect_ratio,
                    "subtitles_enabled": req.subtitles_enabled,
                    "subtitle_settings": req.subtitle_settings.model_dump() if req.subtitle_settings else None,
                    "is_full_video": True,
                    "exported_by_user_id": user.id,
                    "exported_by_username": user.username,
                })
                await database.save_job(j)

            await broadcast_ws(job_id, {
                "type": "export_complete",
                "clip_id": 0,
                "download_url": f"/api/files/{job_id}/clips/{quote(os.path.basename(output_path))}?t={int(time.time())}",
                "message": f"Full video exported in {elapsed}s [{req.export_quality or '1080p'}]",
                "qa_passed": True,
            })
        except asyncio.CancelledError:
            await broadcast_ws(job_id, {
                "type": "error",
                "message": "Full video export cancelled",
            })
        except Exception as e:
            logger.exception("Full video export failed for job %s", job_id)
            await broadcast_ws(job_id, {
                "type": "error",
                "message": f"Full video export failed: {str(e)}",
            })
        finally:
            _active_export_tasks.pop(export_key, None)
            _export_cancel_events.pop(export_key, None)

    task = asyncio.create_task(_do_export())
    _active_export_tasks[export_key] = task
    return {"export_id": export_key, "status": "exporting"}


@router.get("/active-exports")
async def list_active_exports(user: User = Depends(get_current_user)):
    """List active export tasks owned by the calling user."""
    owned_jobs: set[str] | None = None
    active = []
    for key, task in _active_export_tasks.items():
        if not task.done():
            parts = key.split("_", 1)
            jid = parts[0] if len(parts) > 1 else key
            if owned_jobs is None:
                owned_jobs = {
                    j.job_id
                    for j in await database.list_jobs(
                        owner_user_id=user.id,
                        owner_username=user.username,
                    )
                }
            if jid not in owned_jobs:
                continue
            active.append({
                "export_id": key,
                "job_id": jid,
                "clip_id": parts[1] if len(parts) > 1 else None,
                "status": "encoding",
            })
    return active


@router.put("/jobs/{job_id}/clips/{clip_id}/title")
async def update_clip_title(job_id: str, clip_id: int, req: UpdateClipTitleRequest, user: User = Depends(get_current_user)):
    """Update the title of a clip candidate."""
    job = await require_job_access(job_id, user)

    clip = next((c for c in job.clips if c.id == clip_id), None)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    clip.title = req.title
    await database.save_job(job)
    return {"job_id": job_id, "clip_id": clip_id, "title": req.title}


@router.put("/jobs/{job_id}/clips/{clip_id}/times")
async def update_clip_times(job_id: str, clip_id: int, req: UpdateClipTimesRequest, user: User = Depends(get_current_user)):
    """Update the start/end times of a clip candidate."""
    job = await require_job_access(job_id, user)

    clip = next((c for c in job.clips if c.id == clip_id), None)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    if req.start_time is not None:
        clip.start_time = req.start_time
    if req.end_time is not None:
        clip.end_time = req.end_time
    clip.duration = round(clip.end_time - clip.start_time, 2)

    await database.save_job(job)
    return {
        "job_id": job_id,
        "clip_id": clip_id,
        "start_time": clip.start_time,
        "end_time": clip.end_time,
        "duration": clip.duration,
    }


@router.delete("/jobs/{job_id}/clips/{clip_id}")
async def delete_clip(job_id: str, clip_id: int, user: User = Depends(get_current_user)):
    """Delete a single clip candidate and any exported files for it."""
    job = await require_job_access(job_id, user)

    clip = next((c for c in job.clips if c.id == clip_id), None)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    # Remove from clips list
    job.clips = [c for c in job.clips if c.id != clip_id]

    # Remove matching exported clips and their files from disk
    remaining_exports = []
    for ec in job.exported_clips:
        if ec.get("clip_id") == clip_id:
            filepath = ec.get("path", "")
            if filepath and os.path.isfile(filepath):
                try:
                    os.remove(filepath)
                    logger.info("Deleted exported file %s for clip %s/%s", filepath, job_id, clip_id)
                except OSError as e:
                    logger.warning("Failed to delete exported file %s: %s", filepath, e)
        else:
            remaining_exports.append(ec)
    job.exported_clips = remaining_exports

    await database.save_job(job)
    return {"job_id": job_id, "clip_id": clip_id, "deleted": True, "remaining_clips": len(job.clips)}


class DeleteClipsRequest(BaseModel):
    clip_ids: list[int]


@router.post("/jobs/{job_id}/delete-clips")
async def delete_clips_bulk(job_id: str, req: DeleteClipsRequest, user: User = Depends(get_current_user)):
    """Delete multiple clip candidates and their exported files."""
    job = await require_job_access(job_id, user)

    ids_to_delete = set(req.clip_ids)
    deleted_count = 0

    # Remove from clips list
    original_count = len(job.clips)
    job.clips = [c for c in job.clips if c.id not in ids_to_delete]
    deleted_count = original_count - len(job.clips)

    # Remove matching exported clips and their files
    remaining_exports = []
    for ec in job.exported_clips:
        if ec.get("clip_id") in ids_to_delete:
            filepath = ec.get("path", "")
            if filepath and os.path.isfile(filepath):
                try:
                    os.remove(filepath)
                except OSError as e:
                    logger.warning("Failed to delete exported file %s: %s", filepath, e)
        else:
            remaining_exports.append(ec)
    job.exported_clips = remaining_exports

    await database.save_job(job)
    return {
        "job_id": job_id,
        "deleted_count": deleted_count,
        "remaining_clips": len(job.clips),
    }


@router.post("/jobs/{job_id}/generate-clips")
async def generate_clips_endpoint(
    job_id: str,
    req: GenerateClipsRequest,
    user: User = Depends(get_current_user),
):
    """Re-run viral clip detection using existing transcript and scenes."""
    job = await require_job_access(job_id, user)

    if not job.transcript or not job.scenes:
        raise HTTPException(
            status_code=400,
            detail="Transcript and scenes must be available before generating clips",
        )

    # Cancel any existing generation for this job before starting a new one
    await _cancel_existing_generation(job_id)

    # Set up cancellation event for this generation
    cancel_event = asyncio.Event()
    _clip_cancel_events[job_id] = cancel_event

    # Capture job data at request time so background task uses fresh data
    transcript = job.transcript
    scenes = job.scenes
    duration = job.duration

    # Optional scope — narrow analysis to [scope_start, scope_end] for
    # "Find more like this" style per-clip regeneration. The clip window
    # still reports absolute times so downstream timestamps remain stable.
    if req.scope_start is not None and req.scope_end is not None and req.scope_end > req.scope_start:
        _s, _e = float(req.scope_start), float(req.scope_end)
        transcript = [t for t in transcript if t.end > _s and t.start < _e]
        scenes = [sc for sc in scenes if _s <= sc.timestamp <= _e]
        logger.info(
            "generate-clips: scope narrowed to [%.0f,%.0f] (%d segs, %d scenes)",
            _s, _e, len(transcript), len(scenes),
        )

    # Build a summary string for the AI to understand overall video context
    summary_text = None
    if job.summary:
        parts = [job.summary.overview]
        if job.summary.key_topics:
            parts.append(f"Key topics: {', '.join(job.summary.key_topics)}")
        if job.summary.tone:
            parts.append(f"Tone: {job.summary.tone}")
        if job.summary.content_category:
            parts.append(f"Category: {job.summary.content_category}")
        if job.summary.estimated_audience:
            parts.append(f"Audience: {job.summary.estimated_audience}")
        summary_text = "\n".join(parts)

    # Build existing clip info so the AI avoids duplicating already-found clips
    existing_clips_info = None
    if job.clips:
        existing_clips_info = "\n".join(
            f"  \u2022 [{c.start_time:.0f}-{c.end_time:.0f}s] \"{c.title}\""
            for c in job.clips
        )

    # ── Load persisted clip-detection context (Bug 4 in the clip-focus
    #    audit) — the initial pipeline run persists these on the job so
    #    every regenerate-mode call gets the same context the LLM had
    #    the first time. Legacy jobs that pre-date the persistence
    #    fields fall back to empty values and the route still works.
    from backend.services.hot_zone_scorer import HotZone
    from backend.services.content_classifier import ClipContentType
    _loaded_hot_zones: list = []
    try:
        for z in (job.hot_zones or []):
            _loaded_hot_zones.append(HotZone(
                start=float(z.get("start", 0)),
                end=float(z.get("end", 0)),
                composite_score=float(z.get("composite_score", 0)),
                audio_score=float(z.get("audio_score", 0)),
                transcript_score=float(z.get("transcript_score", 0)),
                scene_score=float(z.get("scene_score", 0)),
                speaker_score=float(z.get("speaker_score", 0)),
                signals=list(z.get("signals", []) or []),
            ))
    except Exception as _hz_err:
        logger.warning("Failed to reconstruct hot_zones for %s: %s", job_id, _hz_err)
        _loaded_hot_zones = []
    _loaded_chapters: list = []
    try:
        # Chapters accepted as plain dicts by ai_orchestrator (uses getattr
        # with defaults), so no dataclass reconstruction is needed.
        _loaded_chapters = list(job.chapters or [])
    except Exception:
        _loaded_chapters = []
    _loaded_trend_context = job.trend_context or None
    _loaded_sentiment_timeline = job.sentiment_timeline or None
    _loaded_content_type = None
    try:
        ct_str = (job.clip_content_type or "").strip()
        if ct_str:
            _loaded_content_type = ClipContentType(ct_str)
    except Exception:
        _loaded_content_type = None

    async def _do_generate():
        try:
            from backend.services.ai_orchestrator import AIOrchestrator
            from backend.services.prompts import load_prompts
            from backend.services.clip_scoring import deduplicate_overlapping_clips

            start_time = time.monotonic()

            def _check_cancelled():
                if cancel_event.is_set():
                    raise asyncio.CancelledError("Clip generation cancelled")

            _check_cancelled()

            is_focus_mode = bool(req.clip_focus and req.clip_focus.strip())
            focus_topic = req.clip_focus.strip() if is_focus_mode else None

            if is_focus_mode:
                await broadcast_ws(job_id, {
                    "type": "status",
                    "status": "detecting_clips",
                    "progress": 82,
                    "message": f"Clip Focus: searching for \"{focus_topic}\" — analyzing {len(transcript)} transcript segments, {len(scenes)} scenes...",
                })
            else:
                await broadcast_ws(job_id, {
                    "type": "status",
                    "status": "detecting_clips",
                    "progress": 82,
                    "message": f"Preparing viral clip detection — {len(transcript)} transcript segments, {len(scenes)} scenes...",
                })

            custom_prompts = load_prompts()
            orchestrator = AIOrchestrator(
                ws_broadcast=broadcast_ws,
                custom_prompts=custom_prompts,
                cancel_check=_check_cancelled,
            )

            if is_focus_mode:
                await broadcast_ws(job_id, {
                    "type": "status",
                    "status": "detecting_clips",
                    "progress": 85,
                    "message": f"AI is analyzing video for \"{focus_topic}\" — scanning transcript and scenes...",
                })
            else:
                await broadcast_ws(job_id, {
                    "type": "status",
                    "status": "detecting_clips",
                    "progress": 85,
                    "message": "Sending transcript and scenes to AI for viral clip analysis...",
                })

            # Heartbeat: broadcast progress updates while AI processes
            async def _heartbeat():
                step = 0
                if is_focus_mode:
                    phases = [
                        f"AI is scanning video for \"{focus_topic}\" content...",
                        f"Identifying segments related to \"{focus_topic}\"...",
                        f"Analyzing scene context for \"{focus_topic}\" relevance...",
                        f"Evaluating clip boundaries for \"{focus_topic}\" moments...",
                        f"Finalizing focused clips for \"{focus_topic}\"...",
                    ]
                else:
                    phases = [
                        "AI is analyzing transcript for viral moments...",
                        "Identifying high-engagement segments...",
                        "Scoring clip candidates by viral potential...",
                        "Evaluating hook strength and audience retention...",
                        "Finalizing clip boundaries and scores...",
                    ]

                # Estimate total time based on data size. Bug 11 in the
                # clip-focus audit: the old cap at 300s under-estimated
                # Ollama runs badly. Now we scale with the same heuristic
                # ai_orchestrator.detect_viral_clips uses for timeouts
                # when the primary provider is Ollama.
                seg_count = len(transcript)
                scene_count = len(scenes)
                existing_provider = (job.provider_used or {}).get("clips", "")
                is_ollama = "ollama" in (existing_provider or "").lower() or os.environ.get("AI_PROVIDER", "").lower() == "ollama"
                if is_ollama and duration:
                    est_windows = max(1, int(duration / 300))
                    estimated_total = max(420, est_windows * 300 + 120)
                else:
                    estimated_total = max(30, min(600, seg_count / 3 + scene_count / 2 + 20))

                await asyncio.sleep(8)
                while True:
                    elapsed = int(time.monotonic() - start_time)
                    phase = phases[min(step, len(phases) - 1)]
                    pct = min(95, 85 + step * 2)

                    # Calculate ETA from elapsed time and estimate
                    remaining = max(0, int(estimated_total - elapsed))
                    if elapsed > 10 and remaining > 0:
                        if remaining < 60:
                            eta = f" — ~{remaining}s remaining"
                        else:
                            m, s = divmod(remaining, 60)
                            eta = f" — ~{m}m {s}s remaining"
                    else:
                        eta = ""

                    await broadcast_ws(job_id, {
                        "type": "status",
                        "status": "detecting_clips",
                        "progress": pct,
                        "message": f"{phase} ({elapsed}s elapsed{eta})",
                    })
                    step += 1
                    await asyncio.sleep(6)

            heartbeat_task = asyncio.create_task(_heartbeat())
            try:
                clips, clips_provider = await asyncio.wait_for(
                    orchestrator.detect_viral_clips(
                        transcript, scenes, duration, job_id,
                        clip_count=req.clip_count,
                        min_duration=req.min_duration,
                        max_duration=req.max_duration,
                        clip_focus=req.clip_focus,
                        video_summary=summary_text,
                        existing_clips=existing_clips_info,
                        hot_zones=_loaded_hot_zones or None,
                        chapters=_loaded_chapters or None,
                        trend_context=_loaded_trend_context,
                        sentiment_timeline=_loaded_sentiment_timeline,
                        content_type=_loaded_content_type,
                        viral_score_min=req.viral_score_min,
                        viral_score_max=req.viral_score_max,
                        min_relevance=req.min_relevance,
                    ),
                    timeout=_CLIP_DETECTION_TIMEOUT,
                )
            except asyncio.TimeoutError:
                timeout_min = _CLIP_DETECTION_TIMEOUT // 60
                logger.warning(
                    f"Clip detection timed out for {job_id} after {_CLIP_DETECTION_TIMEOUT}s "
                    f"({len(transcript)} segments, {len(scenes)} scenes)"
                )
                await broadcast_ws(job_id, {
                    "type": "error",
                    "message": (
                        f"Clip detection timed out after {timeout_min} minutes "
                        f"({len(transcript)} transcript segments, {len(scenes)} scenes). "
                        f"Try reducing the number of clips or using a faster AI provider."
                    ),
                })
                return
            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass

            _check_cancelled()

            elapsed = int(time.monotonic() - start_time)

            if is_focus_mode:
                await broadcast_ws(job_id, {
                    "type": "status",
                    "status": "detecting_clips",
                    "progress": 96,
                    "message": f"AI found {len(clips)} clips for \"{focus_topic}\" via {clips_provider} — filtering by duration ({int(req.min_duration)}-{int(req.max_duration)}s)...",
                })
            else:
                await broadcast_ws(job_id, {
                    "type": "status",
                    "status": "detecting_clips",
                    "progress": 96,
                    "message": f"AI returned {len(clips)} viral candidates via {clips_provider} — filtering by duration ({int(req.min_duration)}-{int(req.max_duration)}s)...",
                })

            # Stamp clip_focus on each clip so the UI can distinguish
            # clips generated via focus from viral algorithm clips
            if req.clip_focus and req.clip_focus.strip():
                for c in clips:
                    c.clip_focus = req.clip_focus.strip()

            # Filter clips by user's duration preferences
            filtered = []
            skipped = 0
            for c in clips:
                if c.duration < req.min_duration or c.duration > req.max_duration:
                    skipped += 1
                    continue
                filtered.append(c)

            filter_note = f" ({skipped} outside {int(req.min_duration)}-{int(req.max_duration)}s range)" if skipped else ""

            # Filter by viral score range if specified. In focus mode
            # viral_score carries RELEVANCE (see Bug 3 in the audit), so
            # we gate on viral_score_composite if it exists, else fall
            # back to viral_score.
            if req.viral_score_min > 0 or req.viral_score_max < 100:
                def _virality(c):
                    if is_focus_mode and c.viral_score_composite is not None:
                        return c.viral_score_composite
                    return c.viral_score
                score_filtered = [
                    c for c in filtered
                    if req.viral_score_min <= _virality(c) <= req.viral_score_max
                ]
                score_skipped = len(filtered) - len(score_filtered)
                if score_skipped > 0:
                    filter_note += f" ({score_skipped} outside {req.viral_score_min}-{req.viral_score_max} viral score range)"
                filtered = score_filtered

            # Focus-mode relevance floor (applied after the LLM returns
            # even though the prompt also asks for it — belt-and-braces).
            if is_focus_mode and req.min_relevance > 0:
                before = len(filtered)
                filtered = [
                    c for c in filtered
                    if (c.focus_relevance if c.focus_relevance is not None else c.viral_score)
                    >= req.min_relevance
                ]
                rel_skipped = before - len(filtered)
                if rel_skipped > 0:
                    filter_note += f" ({rel_skipped} below relevance floor {req.min_relevance})"

            # Bug 10 — dedupe near-identical clips by time IoU. Without
            # this the post-filter could keep two Tier-1 matches of the
            # same "fighting scene" that overlap 90%.
            before_dedup = len(filtered)
            filtered, dropped = deduplicate_overlapping_clips(filtered, iou_threshold=0.6)
            if dropped:
                filter_note += f" ({dropped} overlapping duplicates removed)"

            j = await database.load_job(job_id)
            if j:
                if req.append:
                    existing_clips = j.clips or []
                    max_existing_id = max((c.id for c in existing_clips), default=0)
                    for idx, clip in enumerate(filtered, start=max_existing_id + 1):
                        clip.id = idx
                    merged_clips = existing_clips + list(filtered)
                else:
                    for idx, clip in enumerate(filtered, start=1):
                        clip.id = idx
                    merged_clips = list(filtered)
                provider_used = j.provider_used or {}
                provider_used["clips"] = clips_provider
                focus_label = f" for \"{focus_topic}\"" if is_focus_mode else ""
                await database.update_job_status(
                    job_id,
                    status="complete",
                    clips=merged_clips,
                    provider_used=provider_used,
                    progress=100,
                    progress_message=f"Found {len(filtered)} new clips{focus_label} in {elapsed}s{filter_note} ({len(merged_clips)} total)",
                )

            focus_label = f" for \"{focus_topic}\"" if is_focus_mode else ""
            # Structured filter metadata lets the UI render targeted
            # empty-state remediation (Bug 7 in the clip-focus audit)
            # instead of a generic "0 clips" toast.
            filter_state = {
                "focus_enabled": is_focus_mode,
                "focus_query": focus_topic or None,
                "min_relevance": int(req.min_relevance or 0),
                "viral_score_min": int(req.viral_score_min or 0),
                "viral_score_max": int(req.viral_score_max or 100),
            }
            await broadcast_ws(job_id, {
                "type": "clips_generated",
                "count": len(filtered),
                "total": len(merged_clips),
                "provider": clips_provider,
                "filter_state": filter_state,
                "message": f"Found {len(filtered)} new clips{focus_label} via {clips_provider} in {elapsed}s{filter_note} ({len(merged_clips)} total)",
            })
        except asyncio.CancelledError:
            logger.info(f"Clip generation for {job_id} was cancelled (replaced by new generation)")
        except Exception as e:
            logger.exception(f"Clip generation failed for {job_id}")
            await broadcast_ws(job_id, {
                "type": "error",
                "message": f"Clip generation failed: {str(e)}",
            })
        finally:
            _active_clip_tasks.pop(job_id, None)
            _clip_cancel_events.pop(job_id, None)

    task = asyncio.create_task(_do_generate())
    _active_clip_tasks[job_id] = task
    return {"status": "generating", "message": "Generating clips..."}


async def _cancel_existing_generation(job_id: str):
    """Cancel any in-progress clip generation for the given job."""
    # Signal cancellation via the event
    cancel_event = _clip_cancel_events.get(job_id)
    if cancel_event:
        cancel_event.set()

    # Cancel the asyncio task
    existing_task = _active_clip_tasks.pop(job_id, None)
    if existing_task and not existing_task.done():
        existing_task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(existing_task), timeout=3.0)
        except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
            pass
        logger.info(f"Cancelled previous clip generation for {job_id}")

    _clip_cancel_events.pop(job_id, None)


@router.post("/jobs/{job_id}/cancel-generate")
async def cancel_generate_clips(job_id: str, user: User = Depends(get_current_user)):
    """Cancel an in-progress clip generation for this job.

    Invoked by the Analysis page Cancel button (Enhancement 9 in the
    clip-focus audit). Returns immediately after signalling cancel; the
    cooperative cancel_event is checked at every phase boundary inside
    the generate task so the UI returns to a clean state within seconds.
    """
    running = job_id in _clip_cancel_events or job_id in _active_clip_tasks
    await _cancel_existing_generation(job_id)
    if running:
        try:
            await broadcast_ws(job_id, {
                "type": "cancelled",
                "message": "Clip generation cancelled by user",
            })
        except Exception:
            pass
        return {"status": "cancelled", "was_running": True}
    return {"status": "noop", "was_running": False}


@router.get("/jobs/{job_id}/clip-diagnostics")
async def clip_diagnostics(job_id: str, user: User = Depends(get_current_user)):
    """Return every piece of context + per-clip scoring used by the
    most recent clip detection. Powers the "Why these clips?" drawer
    on the Analysis page (Enhancement 6 in the clip-focus audit) and
    serves as a debug / regression fixture.

    Safe to call on any job — fields default to empty when the job is
    older than the persistence changes.
    """
    job = await require_job_access(job_id, user)

    clips_info = []
    for c in (job.clips or []):
        clips_info.append({
            "id": c.id,
            "title": c.title,
            "start": c.start_time,
            "end": c.end_time,
            "viral_score": c.viral_score,
            "viral_score_composite": c.viral_score_composite,
            "focus_relevance": c.focus_relevance,
            "focus_tier": c.focus_tier,
            "clip_focus": c.clip_focus,
            "hook_score": c.hook_score,
            "flow_score": c.flow_score,
            "value_score": c.value_score,
            "trend_score": c.trend_score,
            "score_diagnostics": c.score_diagnostics or {},
        })

    # Axis population stats let us see which providers are actually
    # returning the four axes vs. falling back to legacy fill.
    populated = {"hook": 0, "flow": 0, "value": 0, "trend": 0, "total": len(clips_info)}
    legacy_fill_count = 0
    for c in clips_info:
        diag = c["score_diagnostics"] or {}
        if diag.get("legacy_fill"):
            legacy_fill_count += 1
        axes = diag.get("axis_scores") or {}
        for k in ("hook", "flow", "value", "trend"):
            if axes.get(k, 0) > 0:
                populated[k] += 1

    return {
        "job_id": job_id,
        "clip_content_type": job.clip_content_type or None,
        "provider_used": job.provider_used or {},
        "hot_zones": job.hot_zones or [],
        "chapters": job.chapters or [],
        "trend_context": job.trend_context,
        "sentiment_timeline": job.sentiment_timeline,
        "clips": clips_info,
        "axis_population": populated,
        "legacy_fill_count": legacy_fill_count,
        "four_axis_enabled": bool(os.environ.get("USE_FOUR_AXIS_SCORING", "1").lower() in {"1", "true", "yes", "on"}),
    }


@router.post("/jobs/{job_id}/translate-subtitles")
async def translate_subtitles(job_id: str, req: TranslateRequest, user: User = Depends(get_current_user)):
    """Translate the transcript for a job into a target language."""
    from backend.services.translator import translate_segments_with_fallback, SUPPORTED_LANGUAGES
    from backend.services.ai_orchestrator import AIOrchestrator

    job = await require_job_access(job_id, user)
    if not job.transcript:
        raise HTTPException(404, "Job has no transcript")

    if req.target_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(400, f"Unsupported language: {req.target_language}")

    orchestrator = AIOrchestrator()
    segments = [TranscriptSegment(**s) if isinstance(s, dict) else s for s in job.transcript]

    translated = await translate_segments_with_fallback(
        segments,
        source_language=req.source_language or job.language or "en",
        target_language=req.target_language,
        orchestrator=orchestrator,
    )

    # Store translated transcript and update subtitle_language
    await database.update_job_status(
        job_id,
        translated_transcript=[s.model_dump() for s in translated],
        subtitle_language=req.target_language,
    )

    return {
        "status": "ok",
        "target_language": req.target_language,
        "segments": len(translated),
    }


@router.get("/jobs/{job_id}/clips")
async def list_clips(job_id: str, user: User = Depends(get_current_user)):
    job = await require_job_access(job_id, user)
    return [
        {
            "clip_id": c.get("clip_id"),
            "filename": c.get("filename"),
            "download_url": f"/api/files/{job_id}/clips/{quote(c.get('filename', ''))}",
            "start": c.get("start"),
            "end": c.get("end"),
        }
        for c in job.exported_clips
    ]


@router.get("/jobs/{job_id}/retention")
async def get_retention_predictions(job_id: str, user: User = Depends(get_current_user)):
    """Get audience retention predictions for all clips in a job."""
    from backend.services.retention_predictor import predict_retention_for_clips

    job = await require_job_access(job_id, user)
    if not job.clips:
        return {"predictions": {}}

    predictions = predict_retention_for_clips(job.clips, job.transcript, job.scenes)
    return {"predictions": predictions}


@router.post("/jobs/{job_id}/seo/{clip_id}")
async def generate_seo_endpoint(job_id: str, clip_id: int, user: User = Depends(get_current_user)):
    """Generate SEO-optimized title, description, and tags for a clip."""
    job = await require_job_access(job_id, user)

    clip = next((c for c in job.clips if c.id == clip_id), None)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    # Build clip transcript from segments within the clip time range
    clip_transcript = "\n".join(
        f"{s.speaker}: {s.text}"
        for s in job.transcript
        if s.start >= clip.start_time and s.end <= clip.end_time
    )
    if not clip_transcript:
        clip_transcript = clip.suggested_caption or clip.title

    video_summary = ""
    if job.summary:
        video_summary = job.summary.overview

    # Check that at least one AI provider has an API key configured
    has_any_key = any([
        settings.OPENROUTER_API_KEY,
        settings.ANTHROPIC_API_KEY,
        settings.GEMINI_API_KEY,
        settings.GROQ_API_KEY,
    ])
    if not has_any_key:
        raise HTTPException(
            status_code=400,
            detail="No OpenRouter API key configured. Please add your key in Settings.",
        )

    await broadcast_ws(job_id, {
        "type": "status",
        "status": "generating_seo",
        "progress": 10,
        "message": f"Generating SEO metadata for clip {clip_id}...",
    })

    try:
        from backend.services.ai_orchestrator import AIOrchestrator
        from backend.services.prompts import load_prompts

        orchestrator = AIOrchestrator(ws_broadcast=broadcast_ws, custom_prompts=load_prompts())

        seo, provider = await orchestrator.generate_seo(
            clip_title=clip.title,
            clip_transcript=clip_transcript,
            video_summary=video_summary,
            platform=clip.platform,
            job_id=job_id,
        )

        await broadcast_ws(job_id, {
            "type": "status",
            "status": "generating_seo",
            "progress": 100,
            "message": f"SEO generated via {provider}",
        })

        # Persist SEO data on the clip
        seo_data = seo.model_dump()
        clip.seo_title = seo_data.get("title", "")
        clip.seo_description = seo_data.get("description", "")
        clip.seo_tags = seo_data.get("tags", [])
        clip.seo_platform_tips = seo_data.get("platform_tips", "")
        await database.save_job(job)

        return {
            "clip_id": clip_id,
            "provider": provider,
            "seo": seo_data,
        }
    except Exception as e:
        logger.exception(f"SEO generation failed for {job_id}/{clip_id}")
        raise HTTPException(status_code=500, detail=f"SEO generation failed: {str(e)}")


# ── YouTube Description Generators ─────────────────────────────────────

_SHORTS_DESCRIPTION_PROMPT = (
    "You are a YouTube Shorts SEO expert. Generate a YouTube Shorts description "
    "that is optimized for discoverability and engagement.\n\n"
    "You will be given the FULL TRANSCRIPT of what is said in the clip plus KEY SCENES "
    "describing what visually happens. Use BOTH to write a description that tells "
    "viewers and the YouTube algorithm exactly what this video is about.\n\n"
    "Requirements:\n"
    "- Hook line in the first sentence that references the specific topic or moment in the clip\n"
    "- 3-5 sentences that describe what actually happens in the clip — reference specific things "
    "said, shown, or discussed. Mention key points, quotes, or moments from the transcript.\n"
    "- Weave in relevant keywords naturally so the algorithm understands the content\n"
    "- 5-8 relevant hashtags at the bottom (mix broad and niche tags related to the actual content)\n"
    "- Include a call-to-action (e.g., 'Follow for more', 'Like if you agree')\n"
    "- Total length: 400-800 characters\n"
    "- Write naturally — not like a marketer or robot. Sound like a real creator.\n"
    "- DO NOT be generic. Every sentence should contain specific information from the clip.\n\n"
    "Return ONLY valid JSON:\n"
    '{"description": "the full description text including hashtags"}'
)

_LONGFORM_DESCRIPTION_PROMPT = (
    "You are a YouTube SEO expert. Generate a detailed YouTube video description "
    "optimized for search ranking and viewer engagement.\n\n"
    "You will be given the FULL TRANSCRIPT of what is said in the clip plus KEY SCENES "
    "describing what visually happens. Use BOTH to write a rich, contextual description "
    "that tells viewers and the YouTube algorithm exactly what this video covers.\n\n"
    "Requirements:\n"
    "- Strong opening paragraph (first 2-3 lines appear in search results — front-load keywords). "
    "This paragraph must reference the specific topic and key takeaway of the clip.\n"
    "- A detailed body section (2-3 paragraphs) covering:\n"
    "  * What the viewer will see and learn — reference specific points discussed in the transcript\n"
    "  * Key quotes or statements from the speaker(s) that capture the main message\n"
    "  * The context and significance of what is being discussed or shown\n"
    "- Use the KEY SCENES provided to create a timestamp section with real timestamps "
    "(e.g., '0:00 - Introduction', '0:15 - Main topic begins'). Base these on the scene "
    "timestamps and descriptions given to you.\n"
    "- A keyword-rich paragraph that naturally summarizes the topics for search ranking\n"
    "- 5-8 hashtags section (specific to the actual content, not generic filler)\n"
    "- Social links placeholder section (e.g., 'Follow me on: [Instagram] [Twitter] [TikTok]')\n"
    "- Total length: 1000-3000 characters\n"
    "- Write naturally and engagingly — match the tone of the video content\n"
    "- DO NOT be generic or vague. Every paragraph should contain specific information from "
    "the transcript and scenes. A viewer reading this should understand what the video is about "
    "without watching it.\n\n"
    "Return ONLY valid JSON:\n"
    '{"description": "the full description text"}'
)


class GenerateDescriptionRequest(BaseModel):
    description_type: str  # "shorts" or "long_form"


@router.post("/jobs/{job_id}/generate-description/{clip_id}")
async def generate_description_endpoint(
    job_id: str, clip_id: int, req: GenerateDescriptionRequest,
    user: User = Depends(get_current_user),
):
    """Generate a YouTube Shorts or long-form description for a clip."""
    if req.description_type not in ("shorts", "long_form"):
        raise HTTPException(status_code=400, detail="description_type must be 'shorts' or 'long_form'")

    job = await require_job_access(job_id, user)

    clip = next((c for c in job.clips if c.id == clip_id), None)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    # Check that at least one AI provider has an API key configured
    has_any_key = any([
        settings.OPENROUTER_API_KEY,
        settings.ANTHROPIC_API_KEY,
        settings.GEMINI_API_KEY,
        settings.GROQ_API_KEY,
    ])
    if not has_any_key:
        raise HTTPException(
            status_code=400,
            detail="No OpenRouter API key configured. Please add your key in Settings.",
        )

    # Build clip transcript
    clip_transcript = "\n".join(
        f"{s.speaker}: {s.text}"
        for s in job.transcript
        if s.start >= clip.start_time and s.end <= clip.end_time
    )
    if not clip_transcript:
        clip_transcript = clip.suggested_caption or clip.title

    video_summary = ""
    if job.summary:
        video_summary = job.summary.overview
        if job.summary.key_topics:
            video_summary += "\nTopics: " + ", ".join(job.summary.key_topics)
        if job.summary.content_category:
            video_summary += "\nCategory: " + job.summary.content_category

    # Gather scene descriptions that fall within this clip's time range
    clip_scenes = [
        s for s in job.scenes
        if s.timestamp >= clip.start_time and s.timestamp <= clip.end_time
    ]
    scene_context = ""
    if clip_scenes:
        scene_lines = []
        for s in clip_scenes:
            # Show timestamp relative to clip start for timestamp generation
            rel_ts = s.timestamp - clip.start_time
            mins, secs = divmod(int(rel_ts), 60)
            scene_lines.append(
                f"  [{mins}:{secs:02d}] (importance {s.importance_score}/10) {s.description}"
            )
        scene_context = "\n".join(scene_lines)

    # Select the appropriate prompt prefix
    if req.description_type == "shorts":
        desc_prompt = _SHORTS_DESCRIPTION_PROMPT
    else:
        desc_prompt = _LONGFORM_DESCRIPTION_PROMPT

    # Build rich context with video summary and scene descriptions.
    # NOTE: clip_transcript is passed separately to generate_seo(), so we don't
    # duplicate it here — the provider already includes it in the prompt.
    context_parts = [f"VIDEO OVERVIEW:\n{video_summary}"] if video_summary else []
    if scene_context:
        context_parts.append(f"KEY SCENES IN THIS CLIP (with timestamps relative to clip start):\n{scene_context}")
    full_context = "\n\n".join(context_parts)

    # Use a marker prefix so providers can detect this is a description
    # generation call and skip the default SEO prompt (which conflicts with
    # our character-length and formatting requirements).
    enriched_summary = (
        f"DESCRIPTION_OVERRIDE\n"
        f"{desc_prompt}\n\n"
        f"{full_context}"
    )

    try:
        from backend.services.ai_orchestrator import AIOrchestrator
        from backend.services.prompts import load_prompts

        orchestrator = AIOrchestrator(ws_broadcast=broadcast_ws, custom_prompts=load_prompts())

        seo_result, provider = await orchestrator.generate_seo(
            clip_title=clip.title,
            clip_transcript=clip_transcript,
            video_summary=enriched_summary,
            platform=clip.platform,
            job_id=job_id,
        )

        # The AI should have returned the description in the description field
        description = seo_result.description or ""

        # Persist description on the clip
        if req.description_type == "shorts":
            clip.shorts_description = description
        else:
            clip.longform_description = description
        await database.save_job(job)

        return {
            "clip_id": clip_id,
            "description_type": req.description_type,
            "description": description,
            "provider": provider,
        }
    except Exception as e:
        logger.exception(f"Description generation failed for {job_id}/{clip_id}")
        raise HTTPException(
            status_code=500,
            detail=f"Description generation failed: {str(e)}",
        )


# ── Persist SEO edits ──────────────────────────────────────────────────

class UpdateClipSEORequest(BaseModel):
    seo_title: Optional[str] = None
    seo_description: Optional[str] = None
    seo_tags: Optional[list[str]] = None
    seo_platform_tips: Optional[str] = None
    shorts_description: Optional[str] = None
    longform_description: Optional[str] = None


@router.put("/jobs/{job_id}/clips/{clip_id}/seo")
async def update_clip_seo(job_id: str, clip_id: int, req: UpdateClipSEORequest, user: User = Depends(get_current_user)):
    """Persist user-edited SEO data on a clip."""
    job = await require_job_access(job_id, user)

    clip = next((c for c in job.clips if c.id == clip_id), None)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    if req.seo_title is not None:
        clip.seo_title = req.seo_title
    if req.seo_description is not None:
        clip.seo_description = req.seo_description
    if req.seo_tags is not None:
        clip.seo_tags = req.seo_tags
    if req.seo_platform_tips is not None:
        clip.seo_platform_tips = req.seo_platform_tips
    if req.shorts_description is not None:
        clip.shorts_description = req.shorts_description
    if req.longform_description is not None:
        clip.longform_description = req.longform_description

    await database.save_job(job)

    return {
        "clip_id": clip_id,
        "seo_title": clip.seo_title,
        "seo_description": clip.seo_description,
        "seo_tags": clip.seo_tags,
        "seo_platform_tips": clip.seo_platform_tips,
        "shorts_description": clip.shorts_description,
        "longform_description": clip.longform_description,
    }


# ═══════════════════════════════════════════════════════
# Multi-Track Editor State Endpoints
# ═══════════════════════════════════════════════════════

EDITOR_STATE_DIR = "/data/uploads"


@router.put("/jobs/{job_id}/clips/{clip_id}/editor-state")
async def save_editor_state(job_id: str, clip_id: int, state: dict, user: User = Depends(get_current_user)):
    """Persist editor timeline state for cross-session recovery."""
    import json
    state_dir = os.path.join(EDITOR_STATE_DIR, job_id, "editor-state")
    os.makedirs(state_dir, exist_ok=True)
    state_file = os.path.join(state_dir, f"clip_{clip_id}.json")
    with open(state_file, "w") as f:
        json.dump(state, f)
    logger.info("Saved editor state for job %s clip %d", job_id, clip_id)
    return {"status": "saved", "job_id": job_id, "clip_id": clip_id}


@router.get("/jobs/{job_id}/clips/{clip_id}/editor-state")
async def get_editor_state(job_id: str, clip_id: int, user: User = Depends(get_current_user)):
    """Retrieve saved editor state."""
    import json
    state_file = os.path.join(EDITOR_STATE_DIR, job_id, "editor-state", f"clip_{clip_id}.json")
    if not os.path.isfile(state_file):
        return {"state": None}
    try:
        with open(state_file, "r") as f:
            state = json.load(f)
        return {"state": state}
    except Exception:
        return {"state": None}


@router.post("/jobs/{job_id}/clips/{clip_id}/export-timeline", deprecated=True)
async def export_timeline(job_id: str, clip_id: int, timeline: dict, user: User = Depends(get_current_user)):
    """Deprecated: use POST /jobs/{job_id}/export-clip instead.

    The standard export-clip endpoint now supports all multi-track editor
    data including video_effects, text_overlays, image_overlays, segments,
    and per-segment volume/speed/subtitle overrides.
    """
    raise HTTPException(
        status_code=410,
        detail="This endpoint is deprecated. Use POST /api/jobs/{job_id}/export-clip "
               "with video_effects, text_overlays, and image_overlays fields instead.",
    )


# ── Export Parity Validation ──────────────────────────────────────────

@router.post("/jobs/{job_id}/validate-export-parity")
async def validate_export_parity(job_id: str, req: ExportRequest, user: User = Depends(get_current_user)):
    """Validate that an export request will produce output matching the preview.

    Returns a detailed report of which effects, overlays, and settings will be
    applied in the FFmpeg export, and flags any missing parameters.
    """
    report: dict = {"effects": [], "overlays": [], "segments": [], "warnings": []}

    # Video effects
    if req.video_effects:
        ve = req.video_effects.model_dump()
        for key, val in ve.items():
            default = 0 if key not in ("opacity", "position_x", "position_y", "width", "height") else {
                "opacity": 1.0, "position_x": 50, "position_y": 50, "width": 100, "height": 100,
            }.get(key, 0)
            if val != default:
                report["effects"].append({"name": key, "value": val, "has_ffmpeg_filter": True})
    else:
        report["warnings"].append("No video_effects in request — no brightness/contrast/etc will be applied")

    # Text overlays
    if req.text_overlays:
        for i, t in enumerate(req.text_overlays):
            report["overlays"].append({
                "type": "text", "index": i,
                "text": t.text[:50],
                "timing": f"{t.start_time:.1f}s - {t.end_time:.1f}s",
                "has_drawtext": True,
            })
    else:
        report["warnings"].append("No text_overlays in request")

    # Image overlays
    if req.image_overlays:
        for i, img in enumerate(req.image_overlays):
            report["overlays"].append({
                "type": "image", "index": i,
                "src": img.src[:80] if img.src else "(empty)",
                "timing": f"{img.start_time:.1f}s - {img.end_time:.1f}s",
                "has_overlay_filter": bool(img.src),
            })
    else:
        report["warnings"].append("No image_overlays in request")

    # Audio overlays
    if req.audio_overlays:
        for i, ao in enumerate(req.audio_overlays):
            report["overlays"].append({
                "type": "audio", "index": i,
                "src": ao.src[:80] if ao.src else "(empty)",
                "timing": f"{ao.start_time:.1f}s - {ao.end_time:.1f}s",
                "has_amix": bool(ao.src),
            })

    # Segments
    if req.segments:
        for i, seg in enumerate(req.segments):
            report["segments"].append({
                "index": i,
                "start": seg.start, "end": seg.end,
                "volume": seg.volume, "muted": seg.muted,
                "speed": seg.speed,
                "subtitles_enabled": seg.subtitles_enabled,
            })
    else:
        report["warnings"].append("No per-segment overrides in request")

    # Subtitle check
    if req.subtitles_enabled:
        report["subtitles"] = {
            "enabled": True,
            "font": req.subtitle_settings.font if req.subtitle_settings else "default",
            "burn_in": True,
        }
    else:
        report["subtitles"] = {"enabled": False, "burn_in": False}

    # Volume / speed
    report["audio"] = {"volume": req.volume, "speed": req.speed}

    # Trim
    report["trim"] = {
        "start_offset": req.trim_start_offset,
        "end_offset": req.trim_end_offset,
        "effective_start": req.start + req.trim_start_offset,
        "effective_end": req.end - req.trim_end_offset,
    }

    return {"job_id": job_id, "clip_id": req.clip_id, "parity_report": report}


# ── QA / Validation Endpoint ─────────────────────────────────────────

@router.get("/jobs/{job_id}/qa-validate")
async def qa_validate_job(job_id: str, user: User = Depends(get_current_user)):
    """Run QA validation checks on a completed analysis job."""
    job = await require_job_access(job_id, user)

    checks = []
    warnings = []
    errors = []

    # 1. Job status
    checks.append({
        "name": "Job Status",
        "status": "pass" if job.status == "complete" else "fail",
        "detail": f"Status: {job.status}",
    })

    # 2. Transcript quality
    if job.transcript:
        seg_count = len(job.transcript)
        speaker_count = len(set(s.speaker for s in job.transcript))
        total_words = sum(len(s.text.split()) for s in job.transcript)
        empty_segs = sum(1 for s in job.transcript if not s.text.strip())
        checks.append({
            "name": "Transcript",
            "status": "pass" if seg_count > 0 and total_words > 10 else "warn",
            "detail": f"{seg_count} segments, {speaker_count} speakers, {total_words} words",
        })
        if empty_segs > 0:
            warnings.append(f"{empty_segs} empty transcript segments detected")
        # Check for word-level timestamps
        has_words = sum(1 for s in job.transcript if s.words and len(s.words) > 0)
        checks.append({
            "name": "Word Timestamps",
            "status": "pass" if has_words > 0 else "warn",
            "detail": f"{has_words}/{seg_count} segments have word-level timestamps (needed for active word highlighting)",
        })
    else:
        checks.append({
            "name": "Transcript",
            "status": "fail",
            "detail": "No transcript generated",
        })
        errors.append("No transcript — AI clip detection relies on transcript data")

    # 3. Scene analysis quality
    if job.scenes:
        scene_count = len(job.scenes)
        high_importance = sum(1 for s in job.scenes if s.importance_score >= 7)
        avg_importance = round(sum(s.importance_score for s in job.scenes) / scene_count, 1)
        checks.append({
            "name": "Scene Analysis",
            "status": "pass",
            "detail": f"{scene_count} scenes analyzed, {high_importance} high-importance (7+), avg score {avg_importance}/10",
        })
        # Subject tracking check
        if settings.SUBJECT_TRACKING_ENABLED:
            tracked = [s for s in job.scenes if s.subject_x is not None]
            all_center = all(s.subject_x == 50 for s in tracked) if tracked else True
            checks.append({
                "name": "Subject Tracking",
                "status": "pass" if tracked and not all_center else "warn",
                "detail": f"{len(tracked)}/{scene_count} scenes tracked"
                    + (" — WARNING: all subject_x=50 (model may not have detected positions)" if all_center and tracked else ""),
            })
        else:
            checks.append({
                "name": "Subject Tracking",
                "status": "info",
                "detail": "Disabled in settings — exports use center crop",
            })
    else:
        checks.append({
            "name": "Scene Analysis",
            "status": "fail",
            "detail": "No scenes analyzed",
        })
        errors.append("No scene analysis — visual moment detection unavailable")

    # 4. Clip detection quality
    if job.clips:
        clip_count = len(job.clips)
        scores = [c.viral_score for c in job.clips]
        avg_score = round(sum(scores) / len(scores), 1)
        min_score = min(scores)
        max_score = max(scores)
        durations = [c.duration for c in job.clips]

        checks.append({
            "name": "Clip Detection",
            "status": "pass",
            "detail": f"{clip_count} clips found, viral scores: {min_score}-{max_score} (avg {avg_score})",
        })

        # Duration validation
        invalid_dur = []
        for c in job.clips:
            if c.duration < 5:
                invalid_dur.append(f"Clip {c.id}: too short ({c.duration:.1f}s)")
            elif c.duration > 600:
                invalid_dur.append(f"Clip {c.id}: too long ({c.duration:.1f}s)")
        if invalid_dur:
            checks.append({
                "name": "Clip Durations",
                "status": "warn",
                "detail": f"{len(invalid_dur)} clips with unusual durations",
            })
            warnings.extend(invalid_dur)
        else:
            checks.append({
                "name": "Clip Durations",
                "status": "pass",
                "detail": f"All clips within valid range ({min(durations):.0f}s - {max(durations):.0f}s)",
            })

        # Boundary validation
        out_of_bounds = []
        for c in job.clips:
            if c.start_time < 0:
                out_of_bounds.append(f"Clip {c.id}: negative start ({c.start_time:.1f}s)")
            if job.duration and c.end_time > job.duration + 1:
                out_of_bounds.append(f"Clip {c.id}: end ({c.end_time:.1f}s) exceeds video duration ({job.duration:.1f}s)")
            if c.end_time <= c.start_time:
                out_of_bounds.append(f"Clip {c.id}: end <= start ({c.start_time:.1f}-{c.end_time:.1f})")
        if out_of_bounds:
            checks.append({
                "name": "Clip Boundaries",
                "status": "fail",
                "detail": f"{len(out_of_bounds)} clips with invalid boundaries",
            })
            errors.extend(out_of_bounds)
        else:
            checks.append({
                "name": "Clip Boundaries",
                "status": "pass",
                "detail": "All clips within video boundaries",
            })

        # Overlap detection
        sorted_clips = sorted(job.clips, key=lambda c: c.start_time)
        overlaps = []
        for i in range(len(sorted_clips) - 1):
            a, b = sorted_clips[i], sorted_clips[i + 1]
            overlap = a.end_time - b.start_time
            if overlap > a.duration * 0.5:
                overlaps.append(f"Clips {a.id} and {b.id}: {overlap:.1f}s overlap (>{a.duration * 0.5:.0f}s)")
        if overlaps:
            checks.append({
                "name": "Clip Overlap",
                "status": "warn",
                "detail": f"{len(overlaps)} heavily overlapping clip pairs",
            })
            warnings.extend(overlaps)
        else:
            checks.append({
                "name": "Clip Overlap",
                "status": "pass",
                "detail": "No excessive clip overlap detected",
            })

        # ClipFocus metadata
        focus_clips = [c for c in job.clips if c.clip_focus]
        if focus_clips:
            with_relevance = sum(1 for c in focus_clips if c.focus_relevance is not None)
            with_tier = sum(1 for c in focus_clips if c.focus_tier)
            checks.append({
                "name": "ClipFocus Metadata",
                "status": "pass" if with_relevance == len(focus_clips) else "warn",
                "detail": f"{len(focus_clips)} focus clips, {with_relevance} with relevance scores, {with_tier} with tier labels",
            })
    else:
        checks.append({
            "name": "Clip Detection",
            "status": "fail" if job.status == "complete" else "info",
            "detail": "No clips detected" + (" — analysis may still be running" if job.status != "complete" else ""),
        })
        if job.status == "complete":
            errors.append("Analysis completed but no clips found — check AI provider configuration")

    # 5. Summary quality
    if job.summary:
        has_overview = bool(job.summary.overview and len(job.summary.overview) > 20)
        has_topics = bool(job.summary.key_topics and len(job.summary.key_topics) >= 2)
        has_category = bool(job.summary.content_category and job.summary.content_category != "uncategorized")
        checks.append({
            "name": "Video Summary",
            "status": "pass" if has_overview and has_topics else "warn",
            "detail": f"Category: {job.summary.content_category}, Tone: {job.summary.tone}, "
                f"{len(job.summary.key_topics)} topics, Audience: {job.summary.estimated_audience}",
        })
    else:
        checks.append({
            "name": "Video Summary",
            "status": "warn",
            "detail": "No summary generated — content-type guidance unavailable for clip detection",
        })

    # 6. Provider info
    if job.provider_used:
        provider_strs = [f"{task}: {provider}" for task, provider in job.provider_used.items()]
        has_none = any(v == "none" for v in job.provider_used.values())
        checks.append({
            "name": "AI Providers",
            "status": "warn" if has_none else "pass",
            "detail": ", ".join(provider_strs),
        })
        if has_none:
            warnings.append("Some pipeline stages used no AI provider (fallback/timeout)")
    else:
        checks.append({
            "name": "AI Providers",
            "status": "info",
            "detail": "No provider info recorded",
        })

    # 7. Pipeline timing
    if job.analysis_duration_seconds:
        dur = job.analysis_duration_seconds
        if dur < 60:
            dur_str = f"{int(dur)}s"
        else:
            m, s = divmod(int(dur), 60)
            dur_str = f"{m}m {s}s"
        checks.append({
            "name": "Pipeline Duration",
            "status": "pass",
            "detail": dur_str,
        })

    # Overall verdict
    pass_count = sum(1 for c in checks if c["status"] == "pass")
    warn_count = sum(1 for c in checks if c["status"] == "warn")
    fail_count = sum(1 for c in checks if c["status"] == "fail")

    if fail_count > 0:
        overall = "fail"
    elif warn_count > 0:
        overall = "warn"
    else:
        overall = "pass"

    return {
        "job_id": job_id,
        "overall": overall,
        "summary": f"{pass_count} passed, {warn_count} warnings, {fail_count} failures",
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
    }
