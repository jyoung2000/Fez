"""Re-run Whisper on a finished job using the upload-page settings.

This replaces the old "Silero VAD → fill gaps" approach, which tried
to recover missed dialogue by stitching narrow re-transcriptions into
existing output. In practice that path produced noisy merges and did
little for users — the root cause of bad coverage is almost always a
single failed Whisper pass (silent CTranslate2 OOM, wrong language,
misread subtitle-target selection), and the reliable fix is to just
re-run the full transcription with the user's own settings.

Scope:
* Runs ONLY the transcription phase. Frame extraction, scene
  analysis, clip detection, etc. are NOT re-run — audio is reused
  from the job's work directory so this is 10–30× faster than a
  full pipeline rerun.
* Uses ``job.language`` (audio language) and ``job.subtitle_language``
  (translation target) stored from the upload page. If
  ``subtitle_language == "en"`` and the audio is non-English, Whisper
  runs in native translate mode — mirroring the initial pipeline.
* Re-runs inline heuristic / face-aware diarization so speaker labels
  stay consistent with the new segment boundaries.
* Persists the new transcript via ``database.update_job_status`` and
  returns a small summary dict for the frontend toast.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Optional

from backend import database
from backend.config import settings
from backend.models import JobResult, JobStatus

logger = logging.getLogger(__name__)


def _locate_audio_path(job: JobResult) -> Optional[str]:
    """Resolve the ``audio.wav`` path for ``job``.

    Mirrors the conventions used by ``backend.services.pipeline`` so a
    retranscribe on any job — cloud or local, old or new layout — can
    find the pre-extracted audio.
    """
    candidates: list[str] = []
    file_path = getattr(job, "file_path", "") or ""
    if file_path:
        candidates.append(os.path.join(os.path.dirname(file_path), "audio.wav"))
    jid = getattr(job, "job_id", "") or ""
    if jid:
        candidates.append(f"/data/uploads/{jid}/audio.wav")
        candidates.append(os.path.join("uploads", jid, "audio.wav"))
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def _derive_whisper_task(job: JobResult) -> str:
    """Mirror the pipeline's logic for picking ``transcribe`` vs ``translate``.

    Users select a ``subtitle_language`` at upload time. When that is
    ``en`` and the audio is non-English, Whisper's native translate mode
    is dramatically more accurate than transcribe + LLM translation
    because it uses the raw audio signal. Any other combination keeps
    the source language.
    """
    sub = (job.subtitle_language or "").strip().lower()
    if sub != "en":
        return "transcribe"
    audio_lang = (job.language or "").strip().lower()
    if audio_lang == "en":
        return "transcribe"
    return "translate"


def _build_initial_prompt(job: JobResult, whisper_task: str) -> str:
    """Build the Whisper initial_prompt the same way the pipeline does.

    The filename is deliberately filtered to 1–3 proper-noun tokens —
    Whisper will otherwise echo S01E01 / codec markers verbatim when
    the audio is quiet, which was the original motivation for the
    aggressive stripping.
    """
    parts: list[str] = []
    if whisper_task == "translate":
        audio_lang_label = (job.language or "").strip().lower()
        if audio_lang_label == "ja" or not audio_lang_label:
            parts.append(
                "This is a casual Japanese conversation translated to natural English. "
                "Use complete sentences. Keep names as-is."
            )

    if job.filename:
        name_clean = re.sub(r"\.[^.]+$", "", job.filename)
        name_clean = re.sub(r"[-_\[\](){}]", " ", name_clean)
        name_clean = re.sub(
            r"\b(?:s\d{1,2}e\d{1,3}|ep?\d{1,3}|season\s*\d+|episode\s*\d+|"
            r"\d{3,4}p|x264|x265|h26[45]|hevc|web.?dl|webrip|bluray|bdrip|"
            r"dvdrip|hdtv|amzn|hulu|nf|dsnp|mkv|mp4|avi|aac|flac)\b",
            " ", name_clean, flags=re.IGNORECASE,
        )
        name_clean = re.sub(r"\b\d+\b", " ", name_clean)
        name_clean = re.sub(r"\s+", " ", name_clean).strip()
        tokens = [t for t in name_clean.split() if len(t) >= 3]
        if 1 <= len(tokens) <= 3 and all(t.replace("'", "").isalpha() for t in tokens):
            parts.append(" ".join(tokens))

    return ". ".join(parts) if parts else ""


def _probe_audio_duration(audio_path: str) -> float:
    """Best-effort duration via ffprobe. Returns 0.0 on any failure —
    the Whisper subprocess gate still works without it, just without
    the per-second progress ETA."""
    try:
        import subprocess

        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return float((out.stdout or "").strip() or 0.0)
    except Exception as e:
        logger.debug("ffprobe audio duration failed: %s", e)
    return 0.0


async def _run_diarization(result: list) -> list:
    """Apply heuristic diarization to single-speaker Whisper output.

    Subprocess Whisper labels everything as "Speaker 1"; this restores
    speaker variation using the same pause-based heuristic the main
    pipeline falls back to when face-aware data is unavailable. Face
    data isn't re-extracted here — retranscribe reuses the job's
    existing frame assets; only the transcript changes.
    """
    if not result or len(set(s.speaker for s in result)) > 1:
        return result
    try:
        from backend.services.transcription import assign_speakers_heuristic

        raw_segs = [
            {
                "start": s.start, "end": s.end, "text": s.text,
                "words": (
                    [{"start": w.start, "end": w.end, "word": w.word} for w in s.words]
                    if s.words else None
                ),
                "confidence": s.confidence,
                "avg_logprob": s.avg_logprob,
                "no_speech_prob": s.no_speech_prob,
            }
            for s in result
        ]
        return assign_speakers_heuristic(raw_segs)
    except Exception as e:
        logger.warning("retranscribe: diarization failed (non-fatal): %s", e)
        return result


async def retranscribe_job(job_id: str) -> dict:
    """Re-run Whisper on ``job_id`` using its stored upload settings.

    Returns a small summary dict the API serializes to the caller:

    ``{
        "segments_before": int,
        "segments_after": int,
        "language": str,          # audio language (as selected / detected)
        "subtitle_language": str, # target (empty if user didn't translate)
        "task": "transcribe" | "translate",
        "audio_duration": float,
      }``
    """
    job = await database.load_job(job_id)
    if not job:
        raise ValueError(f"Job not found: {job_id}")

    terminal_statuses = {JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.CANCELLED}
    if job.status not in terminal_statuses:
        raise ValueError(
            f"Retranscribe requires a finished analysis (status={job.status})"
        )

    audio_path = _locate_audio_path(job)
    if not audio_path:
        raise FileNotFoundError(
            f"audio.wav not found for job {job_id}. The source audio was "
            "likely purged — run the full analysis again."
        )

    whisper_task = _derive_whisper_task(job)
    initial_prompt = _build_initial_prompt(job, whisper_task)
    audio_duration = _probe_audio_duration(audio_path)

    logger.info(
        "[%s] Retranscribe: audio=%s duration=%.1fs language=%r subtitle_language=%r task=%s",
        job_id, audio_path, audio_duration,
        job.language or "", job.subtitle_language or "", whisper_task,
    )

    previous = list(job.transcript or [])

    from backend.services.pipeline import broadcast_ws

    def _fmt_time(sec: float) -> str:
        m, s = divmod(max(0, int(sec)), 60)
        return f"{m}:{s:02d}"

    def _fmt_eta(sec: float) -> str:
        sec = max(0, int(sec))
        if sec < 60:
            return f"~{sec}s remaining"
        m, s = divmod(sec, 60)
        return f"~{m}m {s}s remaining"

    phase_label = "Retranslating" if whisper_task == "translate" else "Retranscribing"

    async def _progress(info: dict):
        phase = info.get("phase")
        if phase in ("model_loading", "model_loaded", "vad_start"):
            msg = info.get("message") or f"Whisper: {phase}…"
            await database.update_job_status(
                job_id, status=JobStatus.TRANSCRIBING,
                progress=1, progress_message=msg,
            )
            await broadcast_ws(job_id, {
                "type": "status", "status": JobStatus.TRANSCRIBING.value,
                "progress": 1, "message": msg,
            })
            return

        pct = int(info.get("pct") or 0)
        segments = int(info.get("segments") or 0)
        lang = (info.get("lang") or "").strip()
        position = float(info.get("position_sec") or 0.0)
        eta_sec = float(info.get("eta_sec") or 0.0)

        lang_info = f" [{lang}]" if lang else ""
        total = _fmt_time(audio_duration) if audio_duration > 0 else "?"
        parts = [f"{phase_label}{lang_info}: {_fmt_time(position)} / {total}",
                 f"{segments} segments"]
        if eta_sec > 0 and pct > 0:
            parts.append(_fmt_eta(eta_sec))
        msg = " \u2014 ".join(parts)

        await database.update_job_status(
            job_id, status=JobStatus.TRANSCRIBING,
            progress=max(1, pct), progress_message=msg,
        )
        await broadcast_ws(job_id, {
            "type": "status", "status": JobStatus.TRANSCRIBING.value,
            "progress": max(1, pct), "message": msg,
        })

    _start_msg = f"{phase_label} — warming up Whisper…"
    await database.update_job_status(
        job_id,
        status=JobStatus.TRANSCRIBING,
        progress=1,
        progress_message=_start_msg,
    )
    await broadcast_ws(job_id, {
        "type": "status", "status": JobStatus.TRANSCRIBING.value,
        "progress": 1, "message": _start_msg,
    })

    try:
        if settings.GPU_ACCELERATION_ENABLED:
            from backend.services.transcription import transcribe_audio_subprocess

            _whisper_timeout = (
                max(1800, int(audio_duration * 5)) if audio_duration > 0 else 3600
            )
            result = await asyncio.wait_for(
                transcribe_audio_subprocess(
                    audio_path,
                    language=job.language or "",
                    task=whisper_task,
                    initial_prompt=initial_prompt,
                    audio_duration=audio_duration,
                    progress_callback=_progress,
                ),
                timeout=_whisper_timeout,
            )
        else:
            from backend.services.transcription import transcribe_audio

            result = await transcribe_audio(
                audio_path,
                language=job.language or "",
                task=whisper_task,
                initial_prompt=initial_prompt,
                audio_duration=audio_duration,
                progress_callback=_progress,
            )
    except asyncio.TimeoutError:
        _fail_msg = "Retranscribe timed out — previous transcript kept."
        await database.update_job_status(
            job_id,
            status=JobStatus.COMPLETE,
            progress=100,
            progress_message=_fail_msg,
        )
        await broadcast_ws(job_id, {
            "type": "status", "status": JobStatus.COMPLETE.value,
            "progress": 100, "message": _fail_msg,
        })
        raise ValueError("Whisper timed out — kept the previous transcript.")
    except Exception:
        _fail_msg = "Retranscribe failed — previous transcript kept."
        await database.update_job_status(
            job_id,
            status=JobStatus.COMPLETE,
            progress=100,
            progress_message=_fail_msg,
        )
        await broadcast_ws(job_id, {
            "type": "status", "status": JobStatus.COMPLETE.value,
            "progress": 100, "message": _fail_msg,
        })
        raise

    if not result:
        _empty_msg = "Retranscribe returned no segments — previous transcript kept."
        await database.update_job_status(
            job_id,
            status=JobStatus.COMPLETE,
            progress=100,
            progress_message=_empty_msg,
        )
        await broadcast_ws(job_id, {
            "type": "status", "status": JobStatus.COMPLETE.value,
            "progress": 100, "message": _empty_msg,
        })
        raise ValueError(
            "Whisper returned 0 segments. Kept the previous transcript. "
            "Try running the full analysis again if this video's audio has changed."
        )

    result = await _run_diarization(result)

    speaker_set = sorted(set(s.speaker for s in result))
    speaker_names = {sp: sp for sp in speaker_set} if len(speaker_set) >= 2 else None

    _done_msg = f"Retranscribed — {len(result)} segments"
    update_kwargs: dict = {
        "transcript": list(result),
        "status": JobStatus.COMPLETE,
        "progress": 100,
        "progress_message": _done_msg,
    }
    if speaker_names:
        update_kwargs["speaker_names"] = speaker_names
    await database.update_job_status(job_id, **update_kwargs)
    await broadcast_ws(job_id, {
        "type": "status", "status": JobStatus.COMPLETE.value,
        "progress": 100, "message": _done_msg,
    })

    logger.info(
        "[%s] Retranscribe done: %d → %d segments (task=%s, language=%r)",
        job_id, len(previous), len(result), whisper_task, job.language or "",
    )

    return {
        "segments_before": len(previous),
        "segments_after": len(result),
        "language": job.language or "",
        "subtitle_language": job.subtitle_language or "",
        "task": whisper_task,
        "audio_duration": audio_duration,
    }


__all__ = ["retranscribe_job"]
