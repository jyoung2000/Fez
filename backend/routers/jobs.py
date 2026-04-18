import asyncio
import glob
import logging
import os
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from backend import database
from backend.app.auth.deps import get_current_user
from backend.app.auth.models import Role, User
from backend.models import FrameData, JobStatus, SceneDescription, TranscriptSegment, WordTimestamp
from backend.services.pipeline import run_analysis, request_cancel, is_cancel_requested
from backend.services.retranscribe import retranscribe_job
from backend.services.srt_generator import generate_srt

logger = logging.getLogger(__name__)


async def _require_job_access(job_id: str, user: User):
    """Load a job and verify the caller owns it (or is admin).

    Admins see every job (including legacy jobs with no owner recorded).
    Regular users only see their own jobs — either by matching the
    stored ``owner_user_id`` UUID, or (fallback) by matching the
    stored ``owner_username`` when the UUID no longer resolves. The
    username fallback recovers access to jobs generated before
    ``/data/auth`` was persisted as a docker volume; without it a
    restart that regenerated the admin's UUID would hide every job
    they owned from them even though the job files survive on disk.
    Anything else → 404 so we don't leak the existence of other
    users' jobs.
    """
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    owner = getattr(job, "owner_user_id", "") or ""
    owner_name = (getattr(job, "owner_username", "") or "").strip().lower()
    if user.role == Role.ADMIN:
        return job
    if owner and owner == user.id:
        return job
    if owner_name and owner_name == (user.username or "").strip().lower():
        return job
    if not owner:
        # Legacy pre-auth job: leave invisible to regular users.
        raise HTTPException(status_code=404, detail="Job not found")
    raise HTTPException(status_code=404, detail="Job not found")


class SpeakerRenameRequest(BaseModel):
    speaker_names: dict[str, str]  # {"Speaker 1": "Eric"}

router = APIRouter(prefix="/api", tags=["jobs"])


@router.get("/jobs")
async def list_jobs(user: User = Depends(get_current_user)):
    # Admin sees all jobs (including legacy unowned ones).
    # Regular users only see their own.
    if user.role == Role.ADMIN:
        jobs = await database.list_jobs()
    else:
        jobs = await database.list_jobs(
            owner_user_id=user.id,
            owner_username=user.username,
        )
    return [
        {
            "job_id": j.job_id,
            "filename": j.filename,
            "duration": j.duration,
            "status": j.status,
            "progress": j.progress,
            "progress_message": j.progress_message,
            "created_at": j.created_at,
            "clips_count": len(j.clips),
            "provider_used": j.provider_used,
            "file_size_mb": j.file_size_mb,
            "estimated_cost_usd": j.estimated_cost_usd,
            "owner_user_id": getattr(j, "owner_user_id", "") or "",
        }
        for j in jobs
    ]


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, user: User = Depends(get_current_user)):
    job = await _require_job_access(job_id, user)
    data = job.model_dump(mode="json")
    # Safety: ensure status is always a plain string (not enum remnant)
    if "status" in data and not isinstance(data["status"], str):
        data["status"] = str(data["status"])
    return data


# Per-job single-flight lock so a user can't fire ten concurrent
# retranscribe jobs on the same video and blow the GPU budget. The
# lock is keyed by job_id and lives as long as the process — releases
# automatically when the task finishes.
_retranscribe_locks: dict[str, asyncio.Lock] = {}
# Track the in-flight task per job so we can detect "already running"
# even after we've handed control back to the HTTP client.
_retranscribe_tasks: dict[str, asyncio.Task] = {}


def _get_retranscribe_lock(job_id: str) -> asyncio.Lock:
    lock = _retranscribe_locks.get(job_id)
    if lock is None:
        lock = asyncio.Lock()
        _retranscribe_locks[job_id] = lock
    return lock


async def _run_retranscribe_in_background(job_id: str) -> None:
    """Background worker for ``POST /jobs/{job_id}/retranscribe``.

    Whisper on full-length audio routinely takes minutes. Holding the
    HTTP request open for that long means: (a) reverse proxies time
    out at 60–120s and the user sees a generic "request failed"; (b)
    the response is a long-lived ``StreamingResponse`` chain that re-
    enters every middleware on each chunk, which historically tripped
    the access-log filter and corrupted the body. Running in the
    background lets the endpoint return immediately and the existing
    job-status polling / WebSocket picks up progress + the new
    transcript when ``update_job_status`` writes them.
    """
    lock = _get_retranscribe_lock(job_id)
    async with lock:
        try:
            summary = await retranscribe_job(job_id)
            logger.info(
                "[%s] Retranscribe finished: before=%d after=%d task=%s",
                job_id,
                int(summary.get("segments_before", 0) or 0),
                int(summary.get("segments_after", 0) or 0),
                summary.get("task", ""),
            )
        except FileNotFoundError as e:
            logger.warning("[%s] Retranscribe aborted: %s", job_id, e)
            try:
                await database.update_job_status(
                    job_id,
                    status=JobStatus.COMPLETE,
                    progress_message=f"Retranscribe aborted: {e}",
                )
            except Exception:
                pass
        except ValueError as e:
            logger.warning("[%s] Retranscribe rejected: %s", job_id, e)
            try:
                await database.update_job_status(
                    job_id,
                    status=JobStatus.COMPLETE,
                    progress_message=f"Retranscribe rejected: {e}",
                )
            except Exception:
                pass
        except Exception as e:
            logger.exception("[%s] Retranscribe crashed", job_id)
            try:
                await database.update_job_status(
                    job_id,
                    status=JobStatus.COMPLETE,
                    progress_message=f"Retranscribe failed: {e}",
                )
            except Exception:
                pass
        finally:
            # Drop our handle so the next request isn't rejected as
            # "already running" once this task is done.
            _retranscribe_tasks.pop(job_id, None)


@router.post("/jobs/{job_id}/retranscribe", status_code=202)
async def retranscribe_job_endpoint(
    job_id: str,
    user: User = Depends(get_current_user),
):
    """Kick off a background re-run of Whisper on the job's full audio.

    Uses the language + subtitle-language (translate target) the user
    selected when uploading. If the user chose English subtitles for a
    non-English audio track, Whisper runs in native ``translate`` mode
    — the same logic the initial pipeline uses for the first pass.

    Returns ``202 Accepted`` immediately. Progress is exposed via the
    same ``GET /jobs/{job_id}`` polling and ``/ws/jobs/{job_id}``
    WebSocket the analysis page already watches:

        * ``status`` flips to ``transcribing`` while Whisper runs.
        * ``progress_message`` carries the human-readable phase.
        * ``status`` returns to ``complete`` with the new transcript
          attached when the run finishes.

    Replaces the old gap-fill approach. Only runs on terminal jobs
    (complete / failed / cancelled). The audio.wav is reused from the
    job's work directory, so frame extraction, scene analysis, clip
    detection etc. are NOT re-run — just the transcription phase plus
    inline diarization.
    """
    job = await _require_job_access(job_id, user)

    terminal_statuses = {JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.CANCELLED}
    if job.status not in terminal_statuses:
        raise HTTPException(
            status_code=409,
            detail=(
                "Retranscribe requires a finished analysis. This job is "
                f"still running (status={job.status}). Wait for the "
                "initial analysis to finish and try again."
            ),
        )

    existing = _retranscribe_tasks.get(job_id)
    if existing is not None and not existing.done():
        raise HTTPException(
            status_code=409,
            detail="A retranscribe is already running for this job.",
        )

    # Pre-flight: make sure audio.wav is still on disk, and surface the
    # 404 to the user synchronously instead of swallowing it into the
    # background task. This avoids a "click does nothing" experience on
    # jobs whose work dir has been pruned.
    from backend.services.retranscribe import _locate_audio_path

    if _locate_audio_path(job) is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "audio.wav not found for this job. The source audio was "
                "likely purged — run the full analysis again."
            ),
        )

    # Mark the job as transcribing immediately so the UI button stays
    # disabled and the polling status reflects the new phase before the
    # first Whisper second has even elapsed.
    try:
        await database.update_job_status(
            job_id,
            status=JobStatus.TRANSCRIBING,
            progress_message="Retranscribe queued — Whisper warming up…",
        )
    except Exception:
        logger.exception("[%s] Failed to mark job as transcribing", job_id)

    task = asyncio.create_task(
        _run_retranscribe_in_background(job_id),
        name=f"retranscribe:{job_id}",
    )
    _retranscribe_tasks[job_id] = task

    return {
        "status": "accepted",
        "job_id": job_id,
        "message": (
            "Retranscribe started in the background. Watch the job status "
            "for progress; the transcript will refresh automatically when "
            "Whisper finishes."
        ),
    }


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, user: User = Depends(get_current_user)):
    """Cancel a running or queued job. Also cancels active exports/clip-generation
    for jobs that are already in a terminal state."""
    job = await _require_job_access(job_id, user)

    # Already cancelled — return success (idempotent for repeated clicks)
    if job.status == JobStatus.CANCELLED:
        return {"job_id": job_id, "status": "cancelled"}

    terminal = (JobStatus.COMPLETE, JobStatus.FAILED)
    if job.status in terminal:
        # Job is terminal, but there may be active exports or clip tasks —
        # try to cancel those before rejecting.
        from backend.routers.clips import _active_export_tasks, _export_cancel_events
        from backend.routers.clips import _active_clip_tasks, _clip_cancel_events

        cancelled_something = False

        cancel_evt = _clip_cancel_events.get(job_id)
        if cancel_evt:
            cancel_evt.set()
        clip_task = _active_clip_tasks.pop(job_id, None)
        if clip_task and not clip_task.done():
            clip_task.cancel()
            cancelled_something = True

        for key in list(_active_export_tasks.keys()):
            if key.startswith(f"{job_id}_"):
                evt = _export_cancel_events.get(key)
                if evt:
                    evt.set()
                t = _active_export_tasks.pop(key, None)
                if t and not t.done():
                    t.cancel()
                    cancelled_something = True

        if cancelled_something:
            return {"job_id": job_id, "status": "cancelled"}
        raise HTTPException(status_code=409, detail=f"Job is already {job.status}")

    # For queued jobs not yet running, mark cancelled directly
    if job.status == JobStatus.QUEUED:
        await database.update_job_status(
            job_id,
            status=JobStatus.CANCELLED,
            progress_message="Cancelled by user",
        )
        return {"job_id": job_id, "status": "cancelled"}

    # For running jobs: update DB immediately so polls see "cancelled",
    # then signal the pipeline to stop at its next checkpoint.
    await database.update_job_status(
        job_id,
        status=JobStatus.CANCELLED,
        progress_message="Cancelling...",
    )
    request_cancel(job_id)
    return {"job_id": job_id, "status": "cancelled"}


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str, user: User = Depends(get_current_user)):
    job = await _require_job_access(job_id, user)
    deletable = (JobStatus.FAILED, JobStatus.COMPLETE, JobStatus.QUEUED, JobStatus.CANCELLED)
    if job.status not in deletable:
        raise HTTPException(status_code=409, detail="Cancel the job first before deleting")
    await database.delete_job(job_id)
    return {"job_id": job_id, "deleted": True}


@router.post("/jobs/{job_id}/analyze")
async def trigger_analysis(job_id: str, background_tasks: BackgroundTasks):
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ("queued", "failed", "complete"):
        raise HTTPException(status_code=409, detail="Analysis already in progress")
    background_tasks.add_task(run_analysis, job_id)
    return {"job_id": job_id, "status": "analysis_started"}


@router.get("/jobs/{job_id}/transcript.srt")
async def download_srt(job_id: str, speakers: bool = True):
    """Download the transcript as a speaker-separated SRT subtitle file."""
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.transcript:
        raise HTTPException(status_code=404, detail="No transcript available")

    segments = [TranscriptSegment(**s) if isinstance(s, dict) else s for s in job.transcript]
    srt_content = generate_srt(segments, include_speakers=speakers)

    base = job.filename.rsplit(".", 1)[0] if "." in job.filename else job.filename
    filename = f"{base}.srt"

    return Response(
        content=srt_content,
        media_type="text/srt; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete("/jobs/{job_id}/speakers/{speaker}")
async def delete_speaker(
    job_id: str,
    speaker: str,
    reassign_to: str | None = None,
):
    """Remove a speaker from the transcript.

    If ``reassign_to`` is provided, every transcript segment from
    ``speaker`` is reassigned to ``reassign_to`` and the original name
    is dropped from the speaker-name map. Otherwise every segment from
    ``speaker`` is deleted outright.

    Idempotent: if no segments match ``speaker`` the call still succeeds
    (affected_segments=0) and the rename map is still cleaned up.
    """
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.transcript:
        raise HTTPException(status_code=400, detail="No transcript to edit")
    if reassign_to is not None and reassign_to == speaker:
        raise HTTPException(
            status_code=400,
            detail="Cannot reassign a speaker to itself",
        )

    kept: list = []
    affected = 0
    for seg in job.transcript:
        s = TranscriptSegment(**seg) if isinstance(seg, dict) else seg
        if s.speaker == speaker:
            affected += 1
            if reassign_to:
                kept.append(s.model_copy(update={"speaker": reassign_to}))
            # else: drop the segment entirely
            continue
        kept.append(s)

    # Clean the rename map: drop both direct references (key == speaker)
    # AND any rename whose current value points at the deleted speaker.
    # When reassigning we do NOT remap value==speaker to reassign_to,
    # since the map's purpose is auditing the original → current name
    # transition, not the final destination.
    name_map = {
        k: v
        for k, v in job.speaker_names.items()
        if k != speaker and v != speaker
    }

    await database.update_job_status(
        job_id,
        transcript=[s.model_dump() for s in kept],
        speaker_names=name_map,
    )

    return {
        "job_id": job_id,
        "speaker": speaker,
        "action": "reassign" if reassign_to else "delete",
        "reassign_to": reassign_to,
        "affected_segments": affected,
        "remaining_segments": len(kept),
        "speaker_names": name_map,
    }


@router.put("/jobs/{job_id}/speakers")
async def rename_speakers(job_id: str, req: SpeakerRenameRequest):
    """Rename speakers in the transcript.

    Accepts a mapping like {"Speaker 1": "Eric", "Speaker 2": "Alice"}.
    Updates all transcript segments and persists the mapping.
    """
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.transcript:
        raise HTTPException(status_code=400, detail="No transcript to rename speakers in")

    # Build the full rename map: start from any existing renames, then apply new ones
    name_map = dict(job.speaker_names)
    name_map.update(req.speaker_names)

    # Update transcript segments
    updated_transcript = []
    for seg in job.transcript:
        s = TranscriptSegment(**seg) if isinstance(seg, dict) else seg
        # Check if this speaker's name should be replaced
        if s.speaker in req.speaker_names:
            s = s.model_copy(update={"speaker": req.speaker_names[s.speaker]})
        updated_transcript.append(s)

    await database.update_job_status(
        job_id,
        transcript=[s.model_dump() for s in updated_transcript],
        speaker_names=name_map,
    )

    return {
        "job_id": job_id,
        "speaker_names": name_map,
        "transcript": [s.model_dump() for s in updated_transcript],
    }


# --- Transcript editing ---

class BulkUpdateSpeakerRequest(BaseModel):
    segment_indices: list[int]
    speaker: str


@router.put("/jobs/{job_id}/transcript/bulk-update-speaker")
async def bulk_update_speaker(job_id: str, req: BulkUpdateSpeakerRequest):
    """Update the speaker for multiple transcript segments at once."""
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.transcript:
        raise HTTPException(status_code=404, detail="No transcript")

    updated = []
    for idx in req.segment_indices:
        if idx < 0 or idx >= len(job.transcript):
            continue
        seg = job.transcript[idx]
        if isinstance(seg, dict):
            seg = TranscriptSegment(**seg)
        seg = seg.model_copy(update={"speaker": req.speaker})
        job.transcript[idx] = seg
        updated.append(idx)

    if updated:
        await database.save_job(job)
    return {"job_id": job_id, "updated_indices": updated, "speaker": req.speaker}


class UpdateTranscriptSegmentRequest(BaseModel):
    text: str | None = None
    speaker: str | None = None
    start: float | None = None
    end: float | None = None


@router.put("/jobs/{job_id}/transcript/{segment_index}")
async def update_transcript_segment(job_id: str, segment_index: int, req: UpdateTranscriptSegmentRequest):
    """Update the text and/or speaker of a single transcript segment."""
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.transcript or segment_index < 0 or segment_index >= len(job.transcript):
        raise HTTPException(status_code=404, detail="Segment not found")

    seg = job.transcript[segment_index]
    if isinstance(seg, dict):
        seg = TranscriptSegment(**seg)
    updates = {}
    if req.text is not None:
        updates["text"] = req.text
    if req.speaker is not None:
        updates["speaker"] = req.speaker
    if req.start is not None:
        updates["start"] = req.start
    if req.end is not None:
        updates["end"] = req.end
    if updates:
        seg = seg.model_copy(update=updates)
        job.transcript[segment_index] = seg
        await database.save_job(job)
    return {"job_id": job_id, "segment_index": segment_index, "text": seg.text, "speaker": seg.speaker, "start": seg.start, "end": seg.end}


@router.delete("/jobs/{job_id}/transcript/{segment_index}")
async def delete_transcript_segment(job_id: str, segment_index: int):
    """Delete a single transcript segment."""
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.transcript or segment_index < 0 or segment_index >= len(job.transcript):
        raise HTTPException(status_code=404, detail="Segment not found")

    job.transcript.pop(segment_index)
    await database.save_job(job)
    return {"job_id": job_id, "deleted_index": segment_index, "remaining": len(job.transcript)}


class InsertTranscriptSegmentRequest(BaseModel):
    start: float
    end: float
    text: str
    speaker: str


@router.post("/jobs/{job_id}/transcript")
async def insert_transcript_segment(job_id: str, req: InsertTranscriptSegmentRequest):
    """Insert a new transcript segment. It is placed in chronological order."""
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    new_seg = TranscriptSegment(start=req.start, end=req.end, text=req.text, speaker=req.speaker)

    if not job.transcript:
        job.transcript = [new_seg]
        insert_index = 0
    else:
        # Find insertion point to maintain chronological order
        insert_index = 0
        for i, seg in enumerate(job.transcript):
            s = seg if isinstance(seg, TranscriptSegment) else TranscriptSegment(**seg)
            if s.start > new_seg.start:
                break
            insert_index = i + 1
        job.transcript.insert(insert_index, new_seg)

    await database.save_job(job)
    return {"job_id": job_id, "inserted_index": insert_index, "segment": new_seg.model_dump()}


# --- Word timestamp refresh ---

# Track active refresh tasks to prevent duplicate runs
_active_word_refresh: dict[str, asyncio.Task] = {}


async def _refresh_word_timestamps(job_id: str):
    """Background task: re-run Whisper to extract per-word timestamps and
    merge them onto the existing transcript segments (preserving text/speaker edits)."""
    from backend.services.transcription import extract_word_timestamps
    from backend.services.ws_manager import broadcast_ws

    try:
        job = await database.load_job(job_id)
        if not job or not job.transcript:
            return

        audio_path = f"/data/uploads/{job_id}/audio.wav"
        if not os.path.isfile(audio_path):
            logger.warning("[%s] No audio.wav for word timestamp refresh", job_id)
            await broadcast_ws(job_id, {
                "type": "error",
                "message": "Cannot refresh word timestamps: audio file not found. Re-analyze the video to regenerate it.",
            })
            return

        await broadcast_ws(job_id, {
            "type": "status",
            "status": "refreshing_words",
            "message": "Extracting per-word timestamps from audio...",
            "progress": job.progress,
        })

        all_words = await extract_word_timestamps(audio_path, language=job.language or "")
        if not all_words:
            await broadcast_ws(job_id, {
                "type": "error",
                "message": "Word timestamp extraction returned no words",
            })
            return

        # Re-load job in case it was edited during transcription
        job = await database.load_job(job_id)
        if not job or not job.transcript:
            return

        # Map extracted words onto existing segments by time overlap.
        # For each segment, collect words whose midpoint falls within
        # the segment's time range.
        updated = 0
        for idx, seg in enumerate(job.transcript):
            s = TranscriptSegment(**seg) if isinstance(seg, dict) else seg
            seg_words = [
                w for w in all_words
                if (w.start + w.end) / 2 >= s.start and (w.start + w.end) / 2 < s.end
            ]
            if seg_words:
                s = s.model_copy(update={"words": seg_words})
                job.transcript[idx] = s
                updated += 1

        await database.save_job(job)
        logger.info(
            "[%s] Word timestamps refreshed: %d/%d segments updated (%d total words)",
            job_id, updated, len(job.transcript), len(all_words),
        )

        await broadcast_ws(job_id, {
            "type": "word_timestamps_refreshed",
            "status": "complete",
            "message": f"Word timestamps updated for {updated}/{len(job.transcript)} segments",
            "progress": 100,
        })
    except Exception as e:
        logger.exception("[%s] Word timestamp refresh failed: %s", job_id, e)
        from backend.services.ws_manager import broadcast_ws
        await broadcast_ws(job_id, {
            "type": "error",
            "message": f"Word timestamp refresh failed: {str(e)}",
        })
    finally:
        _active_word_refresh.pop(job_id, None)


@router.post("/jobs/{job_id}/refresh-word-timestamps")
async def refresh_word_timestamps(job_id: str):
    """Re-run Whisper on existing audio to extract per-word timestamps.

    Merges word-level timing onto the existing transcript segments without
    changing text or speaker assignments.  Useful for enabling accurate
    active-word highlighting on transcripts created before word timestamps
    were captured.
    """
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.transcript:
        raise HTTPException(status_code=400, detail="No transcript to refresh")

    # Check if already running
    existing = _active_word_refresh.get(job_id)
    if existing and not existing.done():
        return {"job_id": job_id, "status": "already_running"}

    # Check if all segments already have word timestamps
    segments = [
        TranscriptSegment(**s) if isinstance(s, dict) else s
        for s in job.transcript
    ]
    has_words = sum(1 for s in segments if s.words)
    if has_words == len(segments):
        return {"job_id": job_id, "status": "already_complete", "segments_with_words": has_words}

    task = asyncio.create_task(_refresh_word_timestamps(job_id))
    _active_word_refresh[job_id] = task

    return {
        "job_id": job_id,
        "status": "started",
        "segments_total": len(segments),
        "segments_with_words": has_words,
    }


# --- Post-processing diarization ---

class DiarizeRequest(BaseModel):
    num_speakers: int = 0  # 0 = auto-detect, >0 = exact count


@router.post("/jobs/{job_id}/diarize")
async def diarize_job(job_id: str, req: DiarizeRequest):
    """Run speaker diarization on an existing transcript (post-processing).

    The user specifies how many speakers are in the video. The system
    runs pyannote (if available) or the heuristic speaker assigner to
    label each segment with a speaker identity.
    """
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.transcript:
        raise HTTPException(status_code=400, detail="No transcript to diarize")

    # Find the audio file
    audio_path = None
    upload_dir = f"/data/uploads/{job_id}"
    for ext in ["wav", "mp3", "m4a", "aac", "ogg", "flac"]:
        matches = glob.glob(f"{upload_dir}/*.{ext}")
        if matches:
            audio_path = matches[0]
            break
    # Also check for extracted audio from video
    if not audio_path:
        extracted = os.path.join(upload_dir, "audio.wav")
        if os.path.isfile(extracted):
            audio_path = extracted

    if not audio_path:
        raise HTTPException(
            status_code=400,
            detail="Audio file not found — re-upload the video to enable diarization"
        )

    from backend.services.transcription import diarize_transcript_post

    try:
        diarized = await diarize_transcript_post(
            audio_path=audio_path,
            segments=job.transcript,
            num_speakers=req.num_speakers,
        )

        await database.update_job_status(job_id, transcript=list(diarized))

        speaker_set = set(s.speaker for s in diarized)
        return {
            "status": "ok",
            "speakers_detected": len(speaker_set),
            "speakers_requested": req.num_speakers if req.num_speakers > 0 else "auto",
            "segments_updated": len(diarized),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diarization failed: {str(e)[:200]}")


# --- Scene management ---

class AddSceneRequest(BaseModel):
    timestamp: float
    description: str
    importance_score: int = 7  # 1-10


class UpdateSceneRequest(BaseModel):
    description: str | None = None
    importance_score: int | None = None
    subject_x: int | None = None


@router.post("/jobs/{job_id}/scenes")
async def add_scene(job_id: str, req: AddSceneRequest):
    """Add a user-defined keyscene to the job."""
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Find closest frame for thumbnail (if frames exist)
    thumbnail_path = ""
    frames_dir = f"/data/uploads/{job_id}/frames"
    if os.path.isdir(frames_dir):
        frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png')))
        if frame_files and job.fps > 0:
            target_frame = int(req.timestamp * job.fps)
            # Find closest frame by index
            best = frame_files[0]
            best_diff = abs(target_frame)
            for ff in frame_files:
                try:
                    idx = int(ff.split('_')[1].split('.')[0])
                    diff = abs(idx - target_frame)
                    if diff < best_diff:
                        best_diff = diff
                        best = ff
                except (IndexError, ValueError):
                    pass
            thumbnail_path = os.path.join(frames_dir, best)

    scene = SceneDescription(
        timestamp=req.timestamp,
        description=req.description,
        importance_score=max(1, min(10, req.importance_score)),
        thumbnail_path=thumbnail_path,
        subject_x=50,
    )

    job.scenes.append(scene)
    # Keep scenes sorted by timestamp
    job.scenes.sort(key=lambda s: s.timestamp)
    await database.save_job(job)

    return {
        "job_id": job_id,
        "scene_count": len(job.scenes),
        "scene": scene.model_dump(),
    }


@router.put("/jobs/{job_id}/scenes/{scene_index}")
async def update_scene(job_id: str, scene_index: int, req: UpdateSceneRequest):
    """Update a scene's description, importance score, or subject_x."""
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.scenes or scene_index < 0 or scene_index >= len(job.scenes):
        raise HTTPException(status_code=404, detail="Scene not found")

    scene = job.scenes[scene_index]
    if isinstance(scene, dict):
        scene = SceneDescription(**scene)

    updates = {}
    if req.description is not None:
        updates["description"] = req.description
    if req.importance_score is not None:
        updates["importance_score"] = max(1, min(10, req.importance_score))
    if req.subject_x is not None:
        updates["subject_x"] = max(0, min(100, req.subject_x))

    if updates:
        scene = scene.model_copy(update=updates)
        job.scenes[scene_index] = scene
        await database.save_job(job)

    return {"job_id": job_id, "scene_index": scene_index, "scene": scene.model_dump()}


@router.delete("/jobs/{job_id}/scenes/{scene_index}")
async def delete_scene(job_id: str, scene_index: int):
    """Delete a scene."""
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.scenes or scene_index < 0 or scene_index >= len(job.scenes):
        raise HTTPException(status_code=404, detail="Scene not found")

    job.scenes.pop(scene_index)
    await database.save_job(job)
    return {"job_id": job_id, "scene_count": len(job.scenes)}


# --- Subtitle settings (server is source of truth) ---

@router.put("/jobs/{job_id}/subtitle-settings")
async def save_subtitle_settings(job_id: str, request: Request):
    """Save canonical subtitle settings for a job.

    The server stores these so the browser never relies on potentially-stale
    localStorage values.  On every export the frontend should read settings
    from the job object (populated by GET /api/jobs/{job_id}) rather than
    from local cache.
    """
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    settings = await request.json()
    job.subtitle_settings = settings
    await database.save_job(job)
    return {"job_id": job_id, "subtitle_settings": job.subtitle_settings}


# --- Subject tracking re-center ---

@router.post("/jobs/{job_id}/recenter-subject")
async def recenter_subject(job_id: str, background_tasks: BackgroundTasks):
    """Center the crop on the subject's detected position.

    Preserves per-scene AI-detected subject_x values so that both the
    preview player and the exported video use dynamic (keyframe-based)
    crop tracking.  If no AI analysis has been run yet (all subject_x
    still at the 50 default), triggers a background re-analysis first.
    The per-scene values are kept intact — no flattening to an average —
    so the crop follows the subject through the clip at every aspect ratio.
    """
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.scenes:
        raise HTTPException(status_code=400, detail="No scenes to recenter")

    # Collect current subject_x values
    sx_values = []
    for scene in job.scenes:
        if isinstance(scene, dict):
            sx_values.append(scene.get("subject_x", 50))
        else:
            sx_values.append(scene.subject_x if hasattr(scene, "subject_x") else 50)

    # Check if AI analysis has been run (at least one non-default value)
    has_ai_data = any(v != 50 for v in sx_values)

    if not has_ai_data:
        # No AI analysis data — trigger background re-analysis.
        # center_after=False: preserve per-scene values for dynamic tracking.
        background_tasks.add_task(_reanalyze_subject_tracking, job_id, center_after=False)
        return {
            "job_id": job_id,
            "scenes_recentered": 0,
            "status": "reanalyzing",
            "message": "No subject data available — running AI analysis to detect subject position",
        }

    # AI data already exists — return the per-scene values as-is.
    # The preview player and export pipeline both build keyframes from
    # per-scene subject_x, so the crop dynamically follows the subject
    # at whatever aspect ratio is applied.
    avg_sx = round(sum(sx_values) / len(sx_values))
    avg_sx = max(10, min(90, avg_sx))
    return {
        "job_id": job_id,
        "scenes_recentered": len(job.scenes),
        "subject_x": avg_sx,
        "per_scene": True,
    }


async def _reanalyze_subject_tracking(job_id: str, center_after: bool = False):
    """Background task: re-run AI subject tracking on existing frames.

    If center_after=True, also computes the average subject_x after analysis
    and sets all scenes to that value for a static centered crop.
    """
    from backend.services.ai_orchestrator import AIOrchestrator
    from backend.services.frame_extractor import frame_to_base64
    from backend.services.prompts import load_prompts
    from backend.services.ws_manager import broadcast_ws

    job = await database.load_job(job_id)
    if not job or not job.scenes:
        return

    frames_dir = f"/data/uploads/{job_id}/frames"
    if not os.path.isdir(frames_dir):
        logger.warning("[%s] No frames directory for re-analysis", job_id)
        await broadcast_ws(job_id, {
            "type": "error",
            "message": "No extracted frames found — cannot re-analyze subject tracking",
        })
        return

    await broadcast_ws(job_id, {
        "type": "status",
        "status": "reanalyzing",
        "message": "Re-analyzing subject positions with AI...",
        "progress": job.progress,
    })

    # Build FrameData for each scene from existing frames on disk
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    if not frame_files:
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not frame_files:
        await broadcast_ws(job_id, {
            "type": "error",
            "message": "No frame images found — cannot re-analyze subject tracking",
        })
        return

    # Match existing scenes to their frame files by timestamp
    scene_frames = []
    for scene in job.scenes:
        if isinstance(scene, dict):
            scene = SceneDescription(**scene)
        # Try to find the exact frame file for this scene's thumbnail
        if scene.thumbnail_path and os.path.exists(scene.thumbnail_path):
            scene_frames.append(FrameData(timestamp=scene.timestamp, path=scene.thumbnail_path))
        else:
            # Find closest frame file by name (frame_00123.jpg -> timestamp)
            thumb_name = scene.thumbnail_path.split("/")[-1] if scene.thumbnail_path else ""
            matched = [f for f in frame_files if os.path.basename(f) == thumb_name]
            if matched:
                scene_frames.append(FrameData(timestamp=scene.timestamp, path=matched[0]))
            elif frame_files:
                # Use closest frame by index
                idx = min(len(frame_files) - 1, max(0, round(scene.timestamp / (job.duration or 1) * len(frame_files))))
                scene_frames.append(FrameData(timestamp=scene.timestamp, path=frame_files[idx]))

    if not scene_frames:
        await broadcast_ws(job_id, {
            "type": "error",
            "message": "Could not match scenes to frame files",
        })
        return

    # Encode frames to base64
    loop = asyncio.get_event_loop()
    for fr in scene_frames:
        try:
            fr.base64 = await loop.run_in_executor(None, lambda p=fr.path: frame_to_base64(p, skip_resize=True))
        except Exception as e:
            logger.warning("[%s] Failed to encode frame %s: %s", job_id, fr.path, e)
            fr.base64 = None

    scene_frames = [fr for fr in scene_frames if fr.base64]
    if not scene_frames:
        await broadcast_ws(job_id, {
            "type": "error",
            "message": "Failed to encode frames for re-analysis",
        })
        return

    # Run AI analysis
    try:
        custom_prompts = load_prompts()
        orchestrator = AIOrchestrator(custom_prompts=custom_prompts)
        new_scenes, provider = await orchestrator.analyze_frames(scene_frames, job_id)

        # Map updated subject_x back to existing scenes by timestamp matching
        new_sx_map = {round(s.timestamp, 1): s.subject_x for s in new_scenes}
        updated_count = 0
        for i, scene in enumerate(job.scenes):
            if isinstance(scene, dict):
                scene = SceneDescription(**scene)
                job.scenes[i] = scene
            key = round(scene.timestamp, 1)
            if key in new_sx_map:
                scene.subject_x = new_sx_map[key]
                updated_count += 1

        # Per-scene AI-detected values are preserved (no averaging/flattening).
        # The preview player and export pipeline both build keyframes from
        # per-scene subject_x values, enabling dynamic crop tracking at any
        # aspect ratio.
        if center_after:
            sx_vals = [s.subject_x for s in job.scenes if hasattr(s, "subject_x")]
            non_default = [v for v in sx_vals if v != 50]
            if non_default:
                logger.info(
                    "[%s] Re-analysis complete: preserving %d per-scene subject_x values "
                    "(range %d-%d, avg %d) for dynamic tracking",
                    job_id, len(non_default),
                    min(non_default), max(non_default),
                    round(sum(non_default) / len(non_default)),
                )

        await database.save_job(job)
        logger.info("[%s] Subject re-analysis complete: %d/%d scenes updated via %s", job_id, updated_count, len(job.scenes), provider)

        await broadcast_ws(job_id, {
            "type": "complete",
            "status": "complete",
            "message": f"Subject tracking re-analyzed: {updated_count} scenes updated via {provider}",
            "progress": 100,
        })
    except Exception as e:
        logger.exception("[%s] Subject re-analysis failed: %s", job_id, e)
        await broadcast_ws(job_id, {
            "type": "error",
            "message": f"Subject re-analysis failed: {str(e)}",
        })


@router.post("/jobs/{job_id}/reanalyze-subject")
async def reanalyze_subject(job_id: str, background_tasks: BackgroundTasks):
    """Re-run AI subject tracking analysis on existing frames. Updates subject_x values."""
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.scenes:
        raise HTTPException(status_code=400, detail="No scenes to re-analyze")
    background_tasks.add_task(_reanalyze_subject_tracking, job_id)
    return {"job_id": job_id, "status": "reanalyzing", "scenes_count": len(job.scenes)}


class RescoreRequest(BaseModel):
    """Optional payload for ``POST /api/jobs/{id}/rescore``.

    Lets the caller experiment with a different ``content_type`` (e.g.
    "what would the rankings look like if this were tagged as
    podcast?") without re-running detection. When omitted, uses the
    job's existing classified content type.
    """
    content_type: str | None = None
    focus_mode: bool = False


@router.post("/jobs/{job_id}/rescore")
async def rescore_clips(
    job_id: str,
    payload: RescoreRequest | None = None,
    user: User = Depends(get_current_user),
):
    """Re-run only the genre-weighted scoring composite on existing
    clips. NO LLM / vision / Whisper calls are made — this is purely a
    deterministic Python pass over the clips already on the job.

    Use cases:
      * Experiment with different ``content_type`` weights without
        paying for vision again.
      * Recompute after the user fixed a bad classification in the UI.
      * Backfill ``viral_score_composite`` on legacy jobs that ran
        before the four-axis composite landed.

    Returns the updated clips. Side-effect: persists them onto the job.
    """
    job = await _require_job_access(job_id, user)
    if not job.clips:
        raise HTTPException(status_code=400, detail="Job has no clips to rescore")

    from backend.models import ClipCandidate
    from backend.services.clip_scoring import finalize_clip_scores
    from backend.services.content_classifier import ClipContentType

    # Resolve the content_type to use:
    #   1. explicit override from payload
    #   2. the job's classified content_type (stored on JobResult)
    #   3. the legacy ``content_type_override`` field
    #   4. None — uses the default weights table.
    requested = (payload.content_type if payload else None) or ""
    requested = requested.strip().lower()
    if not requested:
        requested = (
            getattr(job, "classified_content_type", "")
            or getattr(job, "content_type_override", "")
            or ""
        ).strip().lower()
    ct: ClipContentType | None = None
    if requested:
        try:
            ct = ClipContentType(requested)
        except ValueError:
            ct = None

    # Coerce dict clips back to model instances so `finalize_clip_scores`
    # can mutate them in place. Job storage round-trips them as dicts.
    clip_models: list[ClipCandidate] = []
    for raw in job.clips:
        if isinstance(raw, ClipCandidate):
            clip_models.append(raw)
        else:
            try:
                clip_models.append(ClipCandidate(**raw))
            except Exception as e:
                logger.warning("[%s] rescore: skipping malformed clip: %s", job_id, e)
    if not clip_models:
        raise HTTPException(status_code=400, detail="No valid clips found to rescore")

    rescored = finalize_clip_scores(
        clip_models,
        content_type=ct,
        focus_mode=bool(payload.focus_mode) if payload else False,
    )
    job.clips = [c.model_dump() for c in rescored]
    await database.save_job(job)
    return {
        "job_id": job_id,
        "rescored_count": len(rescored),
        "content_type": ct.value if ct else None,
        "focus_mode": bool(payload.focus_mode) if payload else False,
        "clips": job.clips,
    }


@router.get("/logs/export")
async def export_logs():
    """Export all application logs since container start as a downloadable text file.

    Reads from the rotating log file written by the root logger.
    Includes current log and up to 3 rotated backups (oldest first).
    """
    log_file = "/data/logs/app.log"
    parts = []

    # Read rotated backups oldest-first (app.log.3, app.log.2, app.log.1)
    for i in range(3, 0, -1):
        rotated = f"{log_file}.{i}"
        if os.path.isfile(rotated):
            try:
                with open(rotated, "r", encoding="utf-8", errors="replace") as f:
                    parts.append(f.read())
            except OSError:
                pass

    # Read current log file
    if os.path.isfile(log_file):
        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                parts.append(f.read())
        except OSError:
            pass

    if not parts:
        raise HTTPException(status_code=404, detail="No log files found")

    content = "".join(parts)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"clipai_logs_{ts}.log"

    return Response(
        content=content,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/allocation")
async def get_allocation():
    """Return all jobs/processes currently consuming container resources."""
    import psutil

    jobs = await database.list_jobs()
    active_jobs = []
    terminal_statuses = {JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.CANCELLED}
    for j in jobs:
        if j.status not in terminal_statuses:
            elapsed = None
            if j.created_at:
                try:
                    created = datetime.fromisoformat(str(j.created_at))
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                    elapsed = int((datetime.now(timezone.utc) - created).total_seconds())
                except Exception:
                    pass
            active_jobs.append({
                "job_id": j.job_id,
                "filename": j.filename,
                "status": j.status,
                "progress": j.progress,
                "progress_message": j.progress_message,
                "file_size_mb": j.file_size_mb,
                "duration": j.duration,
                "elapsed_seconds": elapsed,
                "type": "analysis",
            })

    # Get active exports from clips router
    from backend.routers.clips import _active_export_tasks, _active_clip_tasks
    for key, task in _active_export_tasks.items():
        if not task.done():
            parts = key.split("_", 1)
            active_jobs.append({
                "job_id": parts[0] if len(parts) > 1 else key,
                "clip_id": int(parts[1]) if len(parts) > 1 else 0,
                "export_key": key,
                "filename": f"Clip {parts[1]}" if len(parts) > 1 else key,
                "status": "encoding",
                "progress": None,
                "progress_message": "Encoding clip...",
                "type": "export",
            })
    for job_id, task in _active_clip_tasks.items():
        if not task.done():
            active_jobs.append({
                "job_id": job_id,
                "filename": "Clip Detection",
                "status": "detecting_clips",
                "progress": None,
                "progress_message": "AI generating clips...",
                "type": "clip_generation",
            })

    # System resource usage
    cpu_percent = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/data") if os.path.exists("/data") else psutil.disk_usage("/")

    return {
        "active_jobs": active_jobs,
        "resources": {
            "cpu_percent": round(cpu_percent, 1),
            "memory_used_mb": round(mem.used / (1024 * 1024)),
            "memory_total_mb": round(mem.total / (1024 * 1024)),
            "memory_percent": round(mem.percent, 1),
            "disk_used_gb": round(disk.used / (1024 ** 3), 1),
            "disk_total_gb": round(disk.total / (1024 ** 3), 1),
            "disk_percent": round(disk.percent, 1),
        },
    }


@router.post("/jobs/{job_id}/force-fail")
async def force_fail_job(job_id: str):
    """Force-fail a job to free up resources. Works on any non-terminal job,
    and also cancels active exports/clip-generation for already-terminal jobs."""
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    from backend.routers.clips import _active_clip_tasks, _clip_cancel_events
    from backend.routers.clips import _active_export_tasks, _export_cancel_events

    terminal = (JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.CANCELLED)
    cancelled_something = False

    # Cancel clip generation tasks for this job
    cancel_evt = _clip_cancel_events.get(job_id)
    if cancel_evt:
        cancel_evt.set()
    clip_task = _active_clip_tasks.pop(job_id, None)
    if clip_task and not clip_task.done():
        clip_task.cancel()
        cancelled_something = True

    # Cancel any exports for this job
    for key in list(_active_export_tasks.keys()):
        if key.startswith(f"{job_id}_"):
            evt = _export_cancel_events.get(key)
            if evt:
                evt.set()
            t = _active_export_tasks.pop(key, None)
            if t and not t.done():
                t.cancel()
                cancelled_something = True

    if job.status in terminal and not cancelled_something:
        raise HTTPException(status_code=409, detail=f"Job is already {job.status}")

    # Signal analysis pipeline cancellation and mark failed (if not already terminal)
    if job.status not in terminal:
        request_cancel(job_id)
        await database.update_job_status(
            job_id,
            status=JobStatus.FAILED,
            progress_message="Force-failed by admin to free resources",
        )

    return {"job_id": job_id, "status": "failed", "message": "Job force-failed"}
