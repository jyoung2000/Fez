import asyncio
import functools
import logging
import json
import os
import shutil
import time as _time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx

from backend.config import settings
from backend.models import JobResult, JobStatus, FrameData, VideoSummary
from backend import database
from backend.services.frame_extractor import (
    get_video_metadata,
    extract_frames,
    extract_frames_at_timestamps,
    extract_audio,
    frame_to_base64,
)
from backend.services.adaptive_frame_sampler import (
    AdaptiveSamplingInputs,
    adaptive_sampling_enabled,
    compute_adaptive_frame_times,
)
from backend.services.vlm_fusion import apply_fusion_to_scene, fusion_enabled
from backend.services.transcription import transcribe_audio, transcribe_audio_subprocess
from backend.services.ai_orchestrator import AIOrchestrator
from backend.services.prompts import load_prompts
from backend.services.providers.base import build_summary_from_transcript, has_real_summary_content, AllProvidersFailedError
from backend.services.audio_analyzer import analyze_audio_energy, format_audio_energy_map
from backend.services.clip_boundary_snapper import snap_all_clips
from backend.services.subject_track import build_subject_track

logger = logging.getLogger(__name__)


# Re-export the human-parity bench entry point. Kept here so callers can
# ``from backend.services.pipeline import run_reframe_on_clip_for_bench``
# without depending on the bridge module's path directly. Import is lazy
# via ``__getattr__`` so importing pipeline.py doesn't pull the bridge's
# OpenCV / scipy dependencies into the hot path.
def __getattr__(name: str):
    if name == "run_reframe_on_clip_for_bench":
        from backend.services.human_reframe_bridge import (
            run_reframe_on_clip_for_bench as _fn,
        )
        return _fn
    raise AttributeError(name)


def _log_gpu_memory(job_id: str, label: str):
    """Log current GPU memory state for VRAM debugging."""
    try:
        import torch
        if torch.cuda.is_available():
            free_mb, total_mb = [x / (1024 * 1024) for x in torch.cuda.mem_get_info()]
            used_mb = total_mb - free_mb
            logger.info(
                "[%s] GPU VRAM [%s]: %.0f MB used / %.0f MB total (%.0f MB free)",
                job_id, label, used_mb, total_mb, free_mb,
            )
    except Exception:
        pass  # Non-critical — don't break pipeline if GPU query fails


async def _release_whisper_vram(job_id: str):
    """Aggressively release Whisper VRAM so Ollama CLIP can use GPU.

    On a 4GB GTX 1650, Whisper occupies 1-3GB VRAM depending on model.
    Without explicit release + verification, Ollama sees residual VRAM
    and falls CLIP back to CPU — making vision analysis 100x slower.
    """
    try:
        import gc
        from backend.services import transcription as _trans_mod

        # Step 1: Use the canonical cleanup (handles del + gc + CUDA)
        with _trans_mod._model_lock:
            if _trans_mod._whisper_model is not None:
                _trans_mod._cleanup_old_model()
                _trans_mod._loaded_model_name = None  # Reset identity tracking
                logger.info("[%s] Whisper model cleaned up via _cleanup_old_model()", job_id)
            else:
                logger.debug("[%s] Whisper model not loaded — nothing to release", job_id)
                return

        # Step 2: Force Python garbage collection (releases CTranslate2 C++ objects)
        gc.collect()
        gc.collect()  # Second pass catches ref cycles

        # Step 3: Force PyTorch CUDA cache release
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

                # Log VRAM state using PyTorch (works inside Docker without nvidia-smi)
                allocated = torch.cuda.memory_allocated() / 1024 / 1024
                reserved = torch.cuda.memory_reserved() / 1024 / 1024
                free_mb, total_mb = [x / (1024 * 1024) for x in torch.cuda.mem_get_info()]
                logger.info(
                    "[%s] Whisper VRAM released — %.0f MB free / %.0f MB total "
                    "(PyTorch: %.0fMB allocated, %.0fMB reserved)",
                    job_id, free_mb, total_mb, allocated, reserved,
                )

                # If PyTorch still holds reserved memory, force aggressive release.
                # PYTORCH_CUDA_ALLOC_CONF helps but doesn't guarantee full release.
                # The nuclear option is resetting the allocator settings.
                if reserved > 100:
                    logger.warning(
                        "[%s] PyTorch still reserving %.0fMB — forcing aggressive release",
                        job_id, reserved,
                    )
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.empty_cache()

                    # Try resetting the CUDA memory allocator (torch >= 2.0)
                    if reserved > 200:
                        try:
                            if hasattr(torch.cuda, 'memory') and hasattr(torch.cuda.memory, '_set_allocator_settings'):
                                torch.cuda.memory._set_allocator_settings("")
                                gc.collect()
                                torch.cuda.empty_cache()
                        except Exception as e:
                            logger.debug("[%s] Allocator reset unavailable: %s", job_id, e)

                    allocated = torch.cuda.memory_allocated() / 1024 / 1024
                    reserved = torch.cuda.memory_reserved() / 1024 / 1024
                    logger.info(
                        "[%s] After aggressive release: %.0fMB allocated, %.0fMB reserved",
                        job_id, allocated, reserved,
                    )
        except ImportError:
            pass
        except Exception as e:
            logger.debug("[%s] PyTorch cleanup skipped: %s", job_id, e)

        # Step 4: Final GC sweep
        gc.collect()

        logger.info("[%s] Whisper model unloaded from VRAM for Ollama", job_id)

    except Exception as e:
        logger.warning("[%s] VRAM release error: %s", job_id, e)


def release_torch_gpu_memory():
    """Release all torch GPU memory. Safe to call multiple times, even if torch not loaded."""
    try:
        import gc
        import torch
        if not torch.cuda.is_available():
            return
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        # Try allocator reset for stubborn cached memory
        try:
            if hasattr(torch.cuda, 'memory') and hasattr(torch.cuda.memory, '_set_allocator_settings'):
                torch.cuda.memory._set_allocator_settings("")
                gc.collect()
                torch.cuda.empty_cache()
        except Exception:
            pass
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        logger.info("Torch GPU memory released: %.0fMB still reserved", reserved)
    except ImportError:
        pass
    except Exception as e:
        logger.debug("Torch GPU release error: %s", e)


def _compute_l1_saliency_peaks(
    *,
    frames,
    audio_path,
    duration_sec: float,
    job_id: str,
    cache_dir,
):
    """Layer 1: per-second saliency peaks for the critic engine.

    Samples 1 frame per second from ``frames`` (the master FrameMeta
    list from ``extract_frames``), computes audio energy from the
    same WAV the rest of the pipeline uses, and asks ``AvSaliency``
    for top-K peaks. Backend selection is internal (TASED-Net first,
    spectral residual fallback) — see ``infra/saliency/README.md``.

    When ``cache_dir`` is provided, writes
    ``saliency_peaks_per_second.json`` so the SOTA bench's
    "extraction cache HIT" path on the same source reads the same
    peaks the export job used.

    Returns a list-of-lists: ``[[ [x_pct, y_pct, score], ... ]_for_sec_0,
    ...]`` matching the schema ``human_reframe_bridge`` and
    ``reframe_critic`` consume. ``None`` on failure.
    """
    try:
        from backend.services.av_saliency import (
            AvSaliency, audio_energy_per_second,
        )
    except Exception as exc:
        logger.info(
            "[%s] saliency import failed (%s); skipping L1", job_id, exc,
        )
        return None

    n_seconds = max(1, int(round(duration_sec or 0.0)))
    if n_seconds < 1:
        return None

    # ── Sample 1 frame per second from the existing FrameMeta list ──
    try:
        import numpy as _np
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            logger.info(
                "[%s] cv2 missing for saliency frame load: %s", job_id, exc,
            )
            return None
        # Pick the first frame ≥ each integer second up to duration.
        sampled_paths = []
        next_t = 0.0
        for f in frames:
            if getattr(f, "timestamp", None) is None or not getattr(f, "path", None):
                continue
            if f.timestamp + 1e-3 >= next_t:
                sampled_paths.append(f.path)
                next_t = float(int(f.timestamp)) + 1.0
                if len(sampled_paths) >= n_seconds:
                    break
        if not sampled_paths:
            return None
        from backend.services.av_saliency import HEATMAP_H, HEATMAP_W
        loaded = []
        for p in sampled_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            loaded.append(img)
        if not loaded:
            return None
        # Resize all frames to the same shape (256x144) BEFORE stacking
        # — source frames may be 1920x1080 or 1280x720, np.stack can't
        # handle mixed shapes.
        normalized = _np.stack([
            cv2.resize(img, (HEATMAP_W, HEATMAP_H))
            for img in loaded
        ]).astype(_np.uint8)
    except Exception as exc:
        logger.info(
            "[%s] saliency frame sampling failed (%s); skipping L1",
            job_id, exc,
        )
        return None

    # ── Audio energy (1 Hz). Tolerates missing wav by passing zeros. ──
    audio_energy_1hz = None
    try:
        import numpy as _np
        if audio_path and os.path.isfile(audio_path):
            try:
                import wave
                with wave.open(str(audio_path), "rb") as wf:
                    sr = wf.getframerate()
                    nchan = wf.getnchannels()
                    raw = wf.readframes(wf.getnframes())
                samples = _np.frombuffer(raw, dtype=_np.int16)
                if nchan > 1:
                    samples = samples.reshape(-1, nchan).mean(axis=1)
                samples = samples.astype(_np.float32) / 32768.0
                audio_energy_1hz = audio_energy_per_second(samples, sr)
            except Exception as exc:
                logger.info(
                    "[%s] audio energy read failed (%s); using zeros",
                    job_id, exc,
                )
        if audio_energy_1hz is None:
            audio_energy_1hz = _np.zeros(n_seconds, dtype=_np.float32)
    except Exception as exc:
        logger.info(
            "[%s] saliency audio prep failed (%s); skipping L1",
            job_id, exc,
        )
        return None

    # ── Predict ──
    try:
        device = "cuda" if release_torch_gpu_memory and _has_cuda() else "cpu"
    except NameError:
        device = "cpu"
    try:
        avs = AvSaliency(device=device)
        result = avs.predict(normalized, audio_energy_1hz)
    except Exception as exc:
        logger.warning(
            "[%s] AvSaliency.predict failed (%s); skipping L1",
            job_id, exc,
        )
        return None
    finally:
        try:
            del avs
        except Exception:
            pass
        if device == "cuda":
            try:
                import torch  # type: ignore
                torch.cuda.empty_cache()
            except Exception:
                pass

    peaks_per_second = [
        [[float(x), float(y), float(s)] for (x, y, s) in sec_peaks]
        for sec_peaks in result.peaks
    ]

    # ── Persist to extraction cache so the bench reads the same data ──
    if cache_dir:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(
                os.path.join(cache_dir, "saliency_peaks_per_second.json"),
                "w",
            ) as fh:
                json.dump(peaks_per_second, fh)
        except Exception as exc:
            logger.info(
                "[%s] saliency cache write failed (%s)", job_id, exc,
            )

    return peaks_per_second


def _has_cuda() -> bool:
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


async def _trigger_ollama_gpu_rediscovery(job_id: str, provider):
    """After releasing torch VRAM, force Ollama to re-discover GPU.

    Ollama caches GPU state from startup. If discovery failed (timeout) or
    the GPU was full (torch hogging VRAM), all subsequent loads use CPU.
    Loading a model with num_gpu=99 AND a vision request triggers a fresh
    GPU scan for both the LLM and the CLIP vision encoder.
    """
    if not hasattr(provider, '_host'):
        return False
    try:
        host = provider._host
        vision_model = provider._vision_model
        logger.info("[%s] Triggering Ollama GPU re-discovery after VRAM release...", job_id)

        # First clear any CPU-loaded models
        if hasattr(provider, 'clear_vram'):
            await provider.clear_vram()
        await asyncio.sleep(2)

        # Generate a tiny 1x1 test image for vision probe
        import base64 as _b64
        _tiny_img = _b64.b64encode(
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
            b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00'
            b'\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00'
            b'\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
        ).decode()

        # Load vision model with GPU forced AND an image to trigger CLIP GPU allocation
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{host}/api/chat",
                json={
                    "model": vision_model,
                    "messages": [{"role": "user", "content": "test", "images": [_tiny_img]}],
                    "stream": False,
                    "options": {"num_gpu": 99, "num_predict": 1},
                },
                timeout=120,
            )
            if resp.status_code == 200:
                ps = await client.get(f"{host}/api/ps", timeout=10)
                if ps.status_code == 200:
                    for m in ps.json().get("models", []):
                        if m.get("size_vram", 0) > 0:
                            logger.info(
                                "[%s] Ollama GPU re-discovery succeeded — %s on GPU (%.0fMB VRAM)",
                                job_id, m.get("name", ""), m.get("size_vram", 0) / 1024 / 1024,
                            )
                            # Don't clear — leave model loaded so scene analysis uses GPU
                            if hasattr(provider, '_force_cpu'):
                                provider._force_cpu = False
                            return True
                logger.warning("[%s] Ollama GPU re-discovery: model still on CPU", job_id)
                # Unload CPU model so scene analysis can retry on GPU
                if hasattr(provider, 'clear_vram'):
                    await provider.clear_vram()
                return False
            elif resp.status_code == 500:
                # OOM during GPU load — model won't fit. Let scene analysis handle CPU fallback
                logger.warning("[%s] Ollama GPU re-discovery: vision model OOM on GPU (HTTP 500)", job_id)
                if hasattr(provider, 'clear_vram'):
                    await provider.clear_vram()
                return False
            else:
                logger.warning("[%s] Ollama GPU re-discovery failed: HTTP %d", job_id, resp.status_code)
                return False
    except Exception as e:
        logger.warning("[%s] Ollama GPU re-discovery error: %s", job_id, e)
        return False


# Dedicated thread pool for base64 frame encoding so it never competes
# with the default executor or the Whisper transcription pool.
_b64_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="b64enc")

# ── Pipeline stage timeouts ──────────────────────────────────────────
_METADATA_TIMEOUT = 120            # 2 min for FFprobe metadata
_EXTRACTION_TIMEOUT = 600          # 10 min for frame + audio extraction
_SUMMARY_CLIP_TIMEOUT = 900        # 15 min for summary + clip detection
_B64_ENCODE_TIMEOUT = 300          # 5 min for base64 frame encoding


# Pure-Python helpers live in pipeline_helpers so unit tests can import
# them without pulling in the full LLM provider stack. Re-export the
# public names so existing pipeline.py call sites keep working.
from backend.services.pipeline_helpers import (  # noqa: E402
    _hash_file_sha256,
    _maybe_use_cached_extraction,
    _write_extraction_manifest,
    _stage_timer,
    _record_pipeline_warning,
    _drain_pipeline_telemetry,
    is_synthetic_scene,
)

# ── Pipeline heartbeat — prevents >15s gaps in progress updates ──────
class _PipelineHeartbeat:
    """Emits keepalive messages when no real progress update has been sent."""

    def __init__(self, job_id: str, interval: float = 15.0):
        self.job_id = job_id
        self.interval = interval
        self.last_emit = _time.monotonic()
        self.current_stage = ""
        self.stage_start = _time.monotonic()
        self._task: asyncio.Task | None = None

    def touch(self, stage: str = ""):
        """Call whenever a real progress event is emitted."""
        self.last_emit = _time.monotonic()
        if stage and stage != self.current_stage:
            self.current_stage = stage
            self.stage_start = _time.monotonic()

    async def _run(self):
        """Background loop that checks for staleness every 5 seconds."""
        try:
            while True:
                await asyncio.sleep(5.0)
                elapsed_since_emit = _time.monotonic() - self.last_emit
                if elapsed_since_emit >= self.interval and self.current_stage:
                    stage_elapsed = int(_time.monotonic() - self.stage_start)
                    mins, secs = divmod(stage_elapsed, 60)
                    msg = f"Still processing... ({self.current_stage} \u2014 {mins}m {secs}s elapsed)"
                    # Only broadcast via WebSocket — don't update DB to avoid
                    # overwriting real progress values with heartbeat messages.
                    await broadcast_ws(self.job_id, {
                        "type": "heartbeat",
                        "message": msg,
                    })
                    self.last_emit = _time.monotonic()
        except asyncio.CancelledError:
            pass

    def start(self):
        self._task = asyncio.create_task(self._run())

    def stop(self):
        if self._task:
            self._task.cancel()
            self._task = None


# Active heartbeats per job
_heartbeats: dict[str, _PipelineHeartbeat] = {}

# Semaphore to limit concurrent analyses
_analysis_semaphore: asyncio.Semaphore | None = None

# WebSocket broadcast registry
_ws_subscribers: dict[str, list] = {}

# Cancellation events — set() means "please cancel"
_cancel_events: dict[str, asyncio.Event] = {}


class CancelledError(Exception):
    """Raised when a job is cancelled by the user."""


def request_cancel(job_id: str):
    """Signal a running job to stop at the next checkpoint."""
    ev = _cancel_events.get(job_id)
    if ev:
        ev.set()
        logger.info(f"Cancellation requested for job {job_id}")


def is_cancel_requested(job_id: str) -> bool:
    ev = _cancel_events.get(job_id)
    return ev.is_set() if ev else False


def _check_cancelled(job_id: str):
    """Raise CancelledError if the job has been cancelled."""
    if is_cancel_requested(job_id):
        raise CancelledError(f"Job {job_id} was cancelled by user")


def get_semaphore() -> asyncio.Semaphore:
    global _analysis_semaphore
    if _analysis_semaphore is None:
        _analysis_semaphore = asyncio.Semaphore(settings.CONCURRENT_ANALYSES)
    return _analysis_semaphore


def register_ws_subscriber(job_id: str, ws):
    if job_id not in _ws_subscribers:
        _ws_subscribers[job_id] = []
    _ws_subscribers[job_id].append(ws)


def unregister_ws_subscriber(job_id: str, ws):
    if job_id in _ws_subscribers:
        _ws_subscribers[job_id] = [w for w in _ws_subscribers[job_id] if w is not ws]
        if not _ws_subscribers[job_id]:
            del _ws_subscribers[job_id]


async def broadcast_ws(job_id: str, message: dict):
    """Broadcast a message to all WebSocket subscribers for a job."""
    from enum import Enum
    # Pre-sanitize: ensure all values are JSON-safe primitives (no Enum remnants)
    safe_message = {}
    for k, v in message.items():
        if isinstance(v, Enum):
            safe_message[k] = str(v.value) if hasattr(v, 'value') else str(v)
        else:
            safe_message[k] = v
    subscribers = _ws_subscribers.get(job_id, [])
    dead = []
    for ws in subscribers:
        try:
            await ws.send_json(safe_message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        unregister_ws_subscriber(job_id, ws)


async def _update_progress(job_id: str, status: str, progress: int, message: str):
    """Update job progress in DB and broadcast via WebSocket.
    If a cancel has been requested, raises CancelledError instead of
    writing a stale progress update that would overwrite the 'cancelled' status."""
    if is_cancel_requested(job_id):
        raise CancelledError(f"Job {job_id} was cancelled by user")
    await database.update_job_status(
        job_id,
        status=status,
        progress=progress,
        progress_message=message,
    )
    await broadcast_ws(job_id, {
        "type": "status",
        "status": status,
        "progress": progress,
        "message": message,
    })
    # Touch heartbeat so it knows we just emitted a real update.
    # Use human-friendly stage names for heartbeat messages.
    hb = _heartbeats.get(job_id)
    if hb:
        _stage_labels = {
            "extracting_frames": "frame extraction",
            "transcribing": "transcription",
            "analyzing_scenes": "scene analysis",
            "generating_summary": "summary generation",
            "detecting_clips": "clip detection",
        }
        stage_label = _stage_labels.get(status, status) if isinstance(status, str) else str(status)
        hb.touch(stage_label)


async def _background_post_processing(job_id: str, transcript: list, orchestrator, job):
    """Run transcript polishing and subtitle translation in background after analysis.

    These are quality-of-life improvements that don't affect clip detection.
    Running them after COMPLETE status saves ~5+ minutes on the critical path.
    """
    # ── Transcript polishing ──
    if settings.AI_TRANSCRIPT_CORRECTION and transcript:
        try:
            from backend.services.transcript_corrector import correct_transcript, _adaptive_batch_size
            logger.info("[%s] Background transcript polishing started (%d segments)", job_id, len(transcript))

            await broadcast_ws(job_id, {
                "type": "background_task",
                "task": "transcript_polishing",
                "status": "running",
                "message": "Polishing transcript in background...",
            })

            _polish_info = orchestrator.get_text_model_info()
            _batch_size = _adaptive_batch_size(len(transcript))
            _total_batches = -(-len(transcript) // _batch_size)
            _remaining_waves = -(- max(0, _total_batches - 1) // 3)
            _per_batch = 150 if _polish_info.get("is_thinking") else 90
            # Scale timeout with segment count — 955 segments at ~7s/batch of 8 = ~835s
            _estimated_time = (_total_batches * _per_batch) * 1.5
            _correction_timeout = max(180, min(1800, int(_estimated_time) + 60))
            logger.info(
                "[%s] Polishing timeout: %ds (segments=%d, batches=%d, per_batch=%ds)",
                job_id, _correction_timeout, len(transcript), _total_batches, _per_batch,
            )

            # Get Whisper's detected language for the correction prompt
            from backend.services.transcription import _last_detected_language
            whisper_lang = _last_detected_language.get("lang", "")

            # If Whisper used task="translate", the transcript is already English
            # regardless of the source language. Tell the corrector it's English
            # so it doesn't apply Japanese-specific corrections to English text.
            _source = job.language.strip().lower() if job.language else ""
            if not _source:
                _source = whisper_lang
            _whisper_translated = (
                job.subtitle_language
                and job.subtitle_language.strip().lower() == "en"
                and _source and _source != "en"
            )
            correction_lang = "en" if _whisper_translated else whisper_lang

            polished = await asyncio.wait_for(
                correct_transcript(transcript, orchestrator, job_id=job_id, language=correction_lang),
                timeout=_correction_timeout,
            )
            await database.update_job_status(job_id, transcript=list(polished))
            transcript = polished  # Use polished version for translation below

            # If OpenRouter ran out of credits mid-polish the breaker
            # tripped and remaining batches were either skipped or
            # served by Ollama. Surface that as ``degraded`` so the
            # frontend can show a partial-polish indicator instead of
            # a misleading "complete" badge.
            from backend.services.ai_orchestrator import (
                _is_openrouter_credit_exhausted,
            )
            if _is_openrouter_credit_exhausted(job_id):
                final_status = "degraded"
                final_message = (
                    "Transcript partially polished — OpenRouter "
                    "credit limit reached"
                )
            else:
                final_status = "complete"
                final_message = "Transcript polished"
            await broadcast_ws(job_id, {
                "type": "background_task",
                "task": "transcript_polishing",
                "status": final_status,
                "message": final_message,
            })
            logger.info(
                "[%s] Background transcript polishing %s",
                job_id, final_status,
            )
        except Exception as e:
            logger.warning("[%s] Background transcript polishing failed: %s", job_id, e)
            await broadcast_ws(job_id, {
                "type": "background_task",
                "task": "transcript_polishing",
                "status": "failed",
                "message": f"Polishing skipped: {str(e)[:80]}",
            })

    # ── Subtitle translation ──
    if job.subtitle_language and transcript:
        source_lang = job.language or ""
        target_lang = job.subtitle_language.strip().lower()

        # If source language wasn't detected, try to get it from Whisper
        if not source_lang:
            from backend.services.transcription import _last_detected_language
            source_lang = _last_detected_language.get("lang", "")

        # If Whisper used task="translate", the transcript is already in English.
        # Store it as translated_transcript and skip LLM translation.
        whisper_did_translate = (
            target_lang == "en"
            and source_lang
            and source_lang != "en"
        )

        if whisper_did_translate:
            logger.info(
                "[%s] Whisper native translate produced English text — "
                "storing as translated_transcript (skipping LLM translation)",
                job_id,
            )
            await database.update_job_status(
                job_id,
                translated_transcript=list(transcript),
            )
            await broadcast_ws(job_id, {
                "type": "background_task",
                "task": "subtitle_translation",
                "status": "complete",
                "message": "English subtitles ready (Whisper native translation)",
            })
        elif target_lang and target_lang != source_lang:
            from backend.services.translator import translate_segments_with_fallback, SUPPORTED_LANGUAGES
            target_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang)
            source_name = SUPPORTED_LANGUAGES.get(source_lang, source_lang) if source_lang else "auto-detected"
            logger.info("[%s] Background subtitle translation: %s → %s (%d segments)",
                        job_id, source_name, target_name, len(transcript))

            await broadcast_ws(job_id, {
                "type": "background_task",
                "task": "subtitle_translation",
                "status": "running",
                "message": f"Translating subtitles to {target_name}...",
            })

            # Scale timeout with segment count — allow extra time for model pull + fallback
            _trans_timeout = max(600, len(transcript) * 4)
            try:
                orchestrator.reset_circuit_breaker()
                translated = await asyncio.wait_for(
                    translate_segments_with_fallback(
                        transcript,
                        source_language=source_lang if source_lang else "auto",
                        target_language=target_lang,
                        orchestrator=orchestrator,
                    ),
                    timeout=_trans_timeout,
                )

                changed = sum(1 for t, o in zip(translated, transcript) if t.text != o.text)
                await database.update_job_status(
                    job_id,
                    translated_transcript=list(translated),
                )

                await broadcast_ws(job_id, {
                    "type": "background_task",
                    "task": "subtitle_translation",
                    "status": "complete",
                    "message": f"Subtitles translated to {target_name} ({changed}/{len(translated)} segments)",
                })
                logger.info("[%s] Translated %d/%d segments to %s",
                            job_id, changed, len(translated), target_lang)

            except Exception as e:
                logger.error("[%s] Translation failed: %s", job_id, e)
                await broadcast_ws(job_id, {
                    "type": "background_task",
                    "task": "subtitle_translation",
                    "status": "failed",
                    "message": f"Translation failed: {str(e)[:80]}",
                })


async def run_analysis(job_id: str):
    """Execute the full analysis pipeline for a video job."""
    # Set up cancellation event for this job
    _cancel_events[job_id] = asyncio.Event()
    sem = get_semaphore()

    # Broadcast immediately so the Analysis page shows status while waiting
    # for the semaphore (especially when another analysis is already running)
    await _update_progress(
        job_id, JobStatus.QUEUED, 1,
        "Preparing analysis pipeline...",
    )

    async with sem:
        # Start heartbeat for this job
        hb = _PipelineHeartbeat(job_id, interval=15.0)
        _heartbeats[job_id] = hb
        hb.start()
        # Resolve the job's owner so we can apply their per-user
        # settings (API keys + model picks + whisper choices) over the
        # global config for the lifetime of this analysis. Falls back
        # to a no-op overlay when the job has no owner (legacy data).
        from backend.app.auth.settings_overlay import overlay_user_settings
        _job_for_owner = await database.load_job(job_id)
        _owner_id = getattr(_job_for_owner, "owner_user_id", "") if _job_for_owner else ""
        try:
            async with overlay_user_settings(_owner_id) as _overlay:
                if _overlay:
                    logger.info(
                        "[%s] applied per-user settings overlay: %d keys (owner=%s)",
                        job_id,
                        sum(1 for k, v in _overlay.items() if v),
                        (_owner_id or "")[:8],
                    )
                await _run_analysis_inner(job_id)
        except CancelledError:
            logger.info(f"Job {job_id} cancelled by user")
            await database.update_job_status(
                job_id,
                status=JobStatus.CANCELLED,
                progress_message="Cancelled by user",
            )
            await broadcast_ws(job_id, {
                "type": "cancelled",
                "message": "Job cancelled by user",
            })
        except Exception as e:
            logger.exception(f"Analysis pipeline failed for {job_id}")
            # Preserve last known progress so the frontend can show where it
            # failed instead of the bar collapsing to 0%.
            current = await database.load_job(job_id)
            last_pct = current.progress if current and current.progress else 0
            await database.update_job_status(
                job_id,
                status=JobStatus.FAILED,
                progress=last_pct,
                progress_message=f"Failed at {last_pct}%: {str(e)[:120]}",
                error=str(e),
            )
            await broadcast_ws(job_id, {
                "type": "error",
                "message": f"Analysis failed: {str(e)}",
            })
        finally:
            # Stop heartbeat and clean up
            hb.stop()
            _heartbeats.pop(job_id, None)
            _cancel_events.pop(job_id, None)
            # Even when the job didn't finish cleanly, persist whatever
            # stage timings + warnings we collected so the UI can show
            # where it died. Best-effort: any DB error is logged but
            # never re-raised from the cleanup path.
            try:
                _t, _w = _drain_pipeline_telemetry(job_id)
                if _t or _w:
                    await database.update_job_status(
                        job_id,
                        timings=_t,
                        pipeline_warnings=_w,
                    )
            except Exception as _te:
                logger.info("[%s] telemetry drain failed: %s", job_id, _te)


# Fix 2: bimodality test for seated-panel subject_x distributions.
# Extracted to backend.services.subject_motion so unit tests can import
# it without pulling in fastapi / PIL / cv2 via the pipeline module.
from backend.services.subject_motion import (
    subject_xs_look_like_motion as _subject_xs_look_like_motion,
    choose_snap_centers as _choose_snap_centers,
)


def _select_dominant_face(faces, last_x=None):
    """Pick the most prominent face from a list of FaceInfo objects.

    Scoring:
      - 35% face size (larger = closer to camera = more important)
      - 25% lip motion (speaking subject is usually the focus)
      - 15% centeredness (cinematographers frame subjects near center)
      - 25% continuity (prefer tracking the same subject as last frame)

    Returns the winning FaceInfo, or None if faces is empty.
    """
    if not faces:
        return None
    if len(faces) == 1:
        return faces[0]

    best = None
    best_score = -1.0
    for f in faces:
        # Size score — larger face area = more prominent (max at 5% of frame)
        area = f.width * f.height / 10000.0  # normalize: 100*100=10000
        size_score = min(1.0, area / 0.05)

        # Lip motion score — speaking subjects are the focus
        lip_score = min(1.0, f.lip_aperture / 0.05) if f.lip_aperture > 0 else 0.0

        # Centeredness — subjects near frame center are typically the focus
        center_dist = abs(f.nose_x - 50) / 50.0
        center_score = 1.0 - center_dist

        # Continuity — prefer the subject we were already tracking
        continuity_score = 0.0
        if last_x is not None:
            x_dist = abs(f.nose_x - last_x)
            if x_dist < 15:
                continuity_score = 1.0 - (x_dist / 15.0)

        total = (
            0.35 * size_score
            + 0.25 * lip_score
            + 0.15 * center_score
            + 0.25 * continuity_score
        )
        if total > best_score:
            best_score = total
            best = f

    return best


async def _run_analysis_inner(job_id: str):
    job = await database.load_job(job_id)
    if not job:
        raise RuntimeError(f"Job {job_id} not found")

    video_path = job.file_path
    job_dir = f"/data/uploads/{job_id}"
    frames_dir = os.path.join(job_dir, "frames")
    audio_path = os.path.join(job_dir, "audio.wav")

    # Record pipeline start time for ETA and total duration tracking
    _pipeline_start = _time.monotonic()
    _pipeline_start_iso = datetime.now(timezone.utc).isoformat()
    await database.update_job_status(job_id, analysis_started_at=_pipeline_start_iso)

    # Create a cancel checker bound to this job
    def cancel_check():
        _check_cancelled(job_id)

    custom_prompts = load_prompts()
    orchestrator = AIOrchestrator(
        ws_broadcast=broadcast_ws,
        custom_prompts=custom_prompts,
        cancel_check=cancel_check,
    )

    # ── Pre-flight: validate AI models are reachable ──
    try:
        model_warnings = await orchestrator.validate_models(job_id)
        for w in model_warnings:
            logger.warning("[%s] Model validation: %s", job_id, w)
            # Surface the same warning to the Analysis page banner.
            # ``validate_models`` already prefixes its messages with
            # the warning glyph; normalize to the ``warn:`` prefix the
            # frontend banner expects.
            try:
                msg = w if w.lower().startswith(("warn:", "info:")) else f"warn: {w}"
                _record_pipeline_warning(job_id, msg)
            except Exception:
                pass
    except Exception as e:
        logger.warning("[%s] Model validation failed (non-fatal): %s", job_id, e)

    # ── Pre-flight: record device used for each accelerated stage ──
    # Surfaced on the Analysis page via JobResult.pipeline_warnings so
    # the user can verify "is GPU actually being used for THIS job"
    # without scraping logs. We deliberately produce an INFO-style
    # entry per accelerated stage; the UI tags it differently from
    # actual warnings.
    try:
        _device_lines: list[str] = []
        # Whisper (faster-whisper)
        try:
            from backend.services.transcription import (
                _detect_cuda_available as _whisper_detect_cuda,
            )
            cuda_avail, cuda_count, gpu_name, _ = _whisper_detect_cuda()
            if settings.GPU_ACCELERATION_ENABLED and cuda_avail and cuda_count > 0:
                _device_lines.append(
                    f"info: Whisper will use GPU ({gpu_name or 'CUDA'})"
                )
            else:
                reason = ("disabled by GPU_ACCELERATION_ENABLED=false"
                          if not settings.GPU_ACCELERATION_ENABLED
                          else "no CUDA device detected")
                _device_lines.append(
                    f"info: Whisper will use CPU ({reason})"
                )
        except Exception as _werr:
            _device_lines.append(
                f"info: Whisper device probe failed: {str(_werr)[:120]}"
            )
        # YOLO / object detector
        try:
            from backend.services.object_detector import _resolve_yolo_device
            _device_lines.append(
                f"info: Object detector device = {_resolve_yolo_device()}"
            )
        except Exception:
            pass
        # Vision provider — cloud (OpenRouter / Gemini / etc.) doesn't
        # use the local GPU, so make that obvious instead of leaving
        # the user wondering why their RTX 4090 is idle during scene
        # analysis.
        try:
            chain = orchestrator._get_active_chain()
            vision_provider = next(
                (p.provider_name for p in chain if p.supports_vision), None
            )
            if vision_provider == "ollama":
                _device_lines.append(
                    "info: Vision provider = ollama (uses local GPU)"
                )
            elif vision_provider:
                _device_lines.append(
                    f"info: Vision provider = {vision_provider} (cloud — local GPU unused for scene analysis)"
                )
        except Exception:
            pass
        for line in _device_lines:
            _record_pipeline_warning(job_id, line)
    except Exception as e:
        logger.debug("[%s] Device probe skipped: %s", job_id, e)

    def _pipeline_elapsed():
        return _time.monotonic() - _pipeline_start

    _clips_phase_start = [0.0]  # mutable; set when clip detection starts
    # Scene analysis phase-local tracking for accurate ETA
    _scene_phase_start = [0.0]  # set when scene analysis begins
    _scene_recent_timestamps: list[float] = []  # timestamps of recent frame completions

    # Phase timing history — records (end_pct, elapsed_sec) for completed phases.
    # Used to estimate remaining phases more accurately than hardcoded constants.
    _phase_timings: dict[str, float] = {}  # phase_name → elapsed seconds
    _transcription_phase_start = [0.0]  # set when transcription begins

    # Last ETA value for smoothing (avoids jarring jumps)
    _last_eta_value = [0.0]
    _last_eta_time = [0.0]

    def _format_remaining(total_remaining: float) -> str:
        if total_remaining < 60:
            return f" — ~{int(total_remaining)}s remaining"
        m, s = divmod(int(total_remaining), 60)
        return f" — ~{m}m {s}s remaining"

    def _smooth_eta(raw_eta: float) -> float:
        """Smooth ETA to avoid jarring jumps between updates.

        Uses exponential moving average: new_eta = 0.3 * raw + 0.7 * prev.
        Resets if more than 30s have passed since last update (phase change).
        """
        now = _time.monotonic()
        prev = _last_eta_value[0]
        gap = now - _last_eta_time[0]

        if prev <= 0 or gap > 30:
            # First call or phase transition — use raw value
            _last_eta_value[0] = raw_eta
            _last_eta_time[0] = now
            return raw_eta

        # Exponential smoothing — heavily weight previous to reduce jitter
        smoothed = 0.3 * raw_eta + 0.7 * prev
        # Clamp: never increase by more than 20% in a single update
        if smoothed > prev * 1.2 and prev > 30:
            smoothed = prev * 1.05
        _last_eta_value[0] = smoothed
        _last_eta_time[0] = now
        return smoothed

    def _estimate_remaining_phases(current_phase: str) -> float:
        """Estimate time for phases after current_phase using actual timings.

        Phase order: extraction → transcription → scene_analysis → summary → clips → save
        Uses actual measured times for completed phases to estimate remaining ones.
        """
        vid_min = metadata["duration"] / 60 if metadata.get("duration") else 10

        # Rough per-phase estimates as fraction of video duration (minutes)
        # These are defaults; replaced by actual timings when available.
        _default_factors = {
            "extraction": 0.04,       # ~4% of video duration
            "transcription": 0.08,    # ~8% of video duration (GPU)
            "scene_analysis": 0.20,   # ~20% of video duration (Ollama)
            "summary": 0.03,          # ~3% of video duration
            "clips": 0.08,            # ~8% of video duration
            "save": 0.002,            # ~15s regardless
        }
        _phase_order = ["extraction", "transcription", "scene_analysis", "summary", "clips", "save"]

        # Find which phases are after current
        try:
            current_idx = _phase_order.index(current_phase)
        except ValueError:
            current_idx = len(_phase_order)

        remaining_secs = 0.0
        for phase_name in _phase_order[current_idx + 1:]:
            if phase_name in _phase_timings:
                # Use actual timing from a completed phase of similar cost
                remaining_secs += _phase_timings[phase_name]
            else:
                # Estimate from video duration
                remaining_secs += vid_min * 60 * _default_factors.get(phase_name, 0.05)

        return max(15, remaining_secs)

    def _pipeline_eta(current_pct):
        """Estimate remaining time based on progress.

        Uses phase-local estimation during scene analysis (15-62%) and clip
        detection (78-95%) to avoid nonsensical ETA drift from global rate.
        """
        if current_pct <= 2:
            return ""
        elapsed = _pipeline_elapsed()

        # During transcription (15-40% in sequential mode), use transcription-local ETA
        if 15 <= current_pct < 40 and _transcription_phase_start[0] > 0 and _scene_phase_start[0] == 0:
            trans_elapsed = _time.monotonic() - _transcription_phase_start[0]
            if trans_elapsed > 5:
                # Transcription maps to 15-40% range (25 points)
                trans_pct_done = current_pct - 15  # 0-25
                if trans_pct_done > 1:
                    trans_rate = trans_pct_done / trans_elapsed
                    trans_remaining = max(0, (40 - current_pct) / trans_rate)
                    # Add estimate for remaining phases
                    after_trans = _estimate_remaining_phases("transcription")
                    raw_eta = trans_remaining + after_trans
                    return _format_remaining(_smooth_eta(raw_eta))

        # During scene analysis (40-62% in sequential mode, or 15-62% concurrent),
        # use sliding window ETA
        if 15 < current_pct < 62 and _scene_phase_start[0] > 0:
            now = _time.monotonic()
            _scene_recent_timestamps.append(now)
            # Keep last 15 timestamps for sliding window average
            if len(_scene_recent_timestamps) > 15:
                _scene_recent_timestamps[:] = _scene_recent_timestamps[-15:]
            if len(_scene_recent_timestamps) >= 3:
                window = _scene_recent_timestamps
                recent_elapsed = window[-1] - window[0]
                recent_steps = len(window) - 1
                if recent_elapsed > 0:
                    recent_rate = recent_steps / recent_elapsed  # pct-steps per sec
                    # Map current_pct to remaining pct in scene phase
                    phase_remaining_pct = 62 - current_pct
                    # Estimate remaining using recent rate (steps map roughly to pct)
                    scene_remaining = phase_remaining_pct / max(0.001, recent_rate)
                    # Add estimate for remaining phases using actual timings
                    after_scenes = _estimate_remaining_phases("scene_analysis")
                    raw_eta = scene_remaining + after_scenes
                    return _format_remaining(_smooth_eta(raw_eta))

        # During summary (65-75%), use phase-local estimate
        if 65 <= current_pct < 75:
            phase_elapsed = elapsed - sum(_phase_timings.get(p, 0) for p in ["extraction", "transcription", "scene_analysis"])
            if phase_elapsed > 3:
                phase_pct = current_pct - 65  # 0-10
                if phase_pct > 0:
                    phase_rate = phase_pct / max(1, phase_elapsed)
                    summary_remaining = max(0, (75 - current_pct) / phase_rate)
                    after_summary = _estimate_remaining_phases("summary")
                    raw_eta = summary_remaining + after_summary
                    return _format_remaining(_smooth_eta(raw_eta))

        # During clip detection (78-95%), use phase-local ETA
        if 78 <= current_pct <= 95 and _clips_phase_start[0] > 0:
            phase_elapsed = _time.monotonic() - _clips_phase_start[0]
            phase_pct = current_pct - 78  # 0-17 within clip phase
            if phase_pct > 0 and phase_elapsed > 5:
                phase_rate = phase_pct / phase_elapsed
                phase_remaining = max(0, (95 - current_pct) / phase_rate)
                total_remaining = phase_remaining + 15  # ~15s for saving
            else:
                # Not enough data yet — rough estimate from video duration
                vid_min = metadata["duration"] / 60 if metadata.get("duration") else 10
                total_remaining = max(60, vid_min * 8)
            return _format_remaining(_smooth_eta(total_remaining))

        # Default: global pipeline rate (used for transitions, early stages)
        if elapsed <= 0:
            return ""
        rate = current_pct / elapsed
        if rate <= 0:
            return ""
        remaining = max(0, (100 - current_pct) / rate)
        return _format_remaining(_smooth_eta(remaining))

    # Step 1 — Video Metadata (0-5%)
    cancel_check()
    await _update_progress(job_id, JobStatus.EXTRACTING_FRAMES, 2, "Extracting video metadata...")
    logger.info("[%s] Pipeline started — video: %s", job_id, video_path)
    _log_gpu_memory(job_id, "pipeline start")
    async with _stage_timer(job_id, "metadata"):
        try:
            metadata = await asyncio.wait_for(
                get_video_metadata(video_path), timeout=_METADATA_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Video metadata extraction timed out after {_METADATA_TIMEOUT}s. "
                "The file may be corrupt or on slow storage."
            )
    # Backfill ``width``/``height`` from the legacy ``resolution``
    # string when the probe didn't expose them as first-class fields.
    # Without this, every consumer of ``metadata.get("width", 1920)``
    # silently defaults to 1920x1080 — breaking AutoFlip / reframe
    # pixel math for every non-FullHD source (4K, 720p, vertical
    # phone clips, etc.).
    if not metadata.get("width") or not metadata.get("height"):
        try:
            res = str(metadata.get("resolution") or "")
            if "x" in res:
                w_str, h_str = res.split("x", 1)
                _w = int("".join(ch for ch in w_str if ch.isdigit()) or 0)
                _h = int("".join(ch for ch in h_str if ch.isdigit()) or 0)
                if _w > 0 and _h > 0:
                    metadata["width"] = _w
                    metadata["height"] = _h
        except Exception:
            # Final safety net — leave the existing defaults in place
            # rather than crash the pipeline over a malformed string.
            pass
    await database.update_job_status(
        job_id,
        duration=metadata["duration"],
        resolution=metadata["resolution"],
        fps=metadata["fps"],
        file_size_mb=metadata["file_size_mb"],
    )
    dur_fmt = f"{int(metadata['duration'] // 60)}:{int(metadata['duration'] % 60):02d}"
    res = metadata.get("resolution", "?")
    fps_val = metadata.get("fps", 0)
    mb = metadata.get("file_size_mb", 0)

    # Duration tier system — auto-adjusts pipeline based on video length
    from backend.config import get_duration_tier, apply_ollama_overrides
    vid_minutes = metadata["duration"] / 60
    tier = get_duration_tier(metadata["duration"])

    # Detect if Ollama is the primary provider
    _active_chain = orchestrator._get_active_chain()
    _primary_provider = _active_chain[0] if _active_chain else None
    is_ollama_primary = _primary_provider and _primary_provider.provider_name == "ollama"

    if is_ollama_primary:
        tier = apply_ollama_overrides(tier, is_ollama=True)
        logger.info(
            "[%s] Ollama is primary provider — applying overrides: "
            "window=%ds, timeout=%ds, summary=%s, sequential=True",
            job_id, tier.window_duration, tier.per_call_timeout_base, tier.summary_strategy,
        )
        # Warm up models to detect capabilities and VRAM constraints,
        # then immediately unload so Whisper gets exclusive GPU access.
        # Models reload automatically when scene analysis starts.
        # Timeout: skip warmup if it takes too long — models will load lazily.
        try:
            await _update_progress(job_id, JobStatus.EXTRACTING_FRAMES, 3, "Warming up local AI models...")
            await asyncio.wait_for(_primary_provider.warmup(), timeout=60)
            # Log GPU status after warmup for diagnostics
            if hasattr(_primary_provider, 'log_gpu_status'):
                await _primary_provider.log_gpu_status()
        except asyncio.TimeoutError:
            logger.warning("[%s] Ollama warmup timed out after 60s — skipping (models will load lazily)", job_id)
        except Exception as e:
            logger.warning("[%s] Ollama warmup failed (non-fatal): %s", job_id, e)

        # ── Critical: free GPU for Whisper ──
        # warmup() loaded Ollama models (qwen2.5:3b = 2.3GB) onto the GPU.
        # On a 4GB GPU, this leaves only ~1.5GB for Whisper → silent OOM.
        # Unload now — models reload when pipeline reaches scene analysis.
        try:
            await _update_progress(job_id, JobStatus.EXTRACTING_FRAMES, 4, "Freeing GPU for transcription...")
            await asyncio.wait_for(_primary_provider.unload_models(), timeout=15)
            logger.info("[%s] Ollama models unloaded after warmup — GPU freed for Whisper", job_id)
            await asyncio.sleep(2)  # Let CUDA driver reclaim across containers
        except asyncio.TimeoutError:
            logger.warning("[%s] Ollama model unload timed out after 15s — proceeding anyway", job_id)
        except Exception as e:
            logger.warning("[%s] Failed to unload Ollama after warmup: %s", job_id, e)

    logger.info(
        "[%s] Duration tier: %s (%.1f min) — frame_rate=%ds, summary=%s, "
        "window=%ds, max_clips=%d, max_gaps=%d, ollama=%s",
        job_id, tier.name, vid_minutes, tier.frame_sample_rate,
        tier.summary_strategy, tier.window_duration,
        tier.max_clip_candidates, tier.max_gaps_pass2, is_ollama_primary,
    )

    # Adaptive timeouts based on video duration + provider type + tier
    _EXTRACTION_TIMEOUT = max(600, int(vid_minutes * 60))
    # Inner Whisper subprocess timeout (must stay in sync with
    # ``_whisper_timeout`` in _branch_transcription): covers CPU
    # fallback at ~0.5-1x real-time with safety margin. Factor this
    # out so the outer ``_trans_scene_timeout`` below can derive
    # from it — they MUST satisfy outer >= inner + overhead, else
    # the outer wait_for cancels an in-flight subprocess that was
    # still well within its own budget (which is exactly the
    # failure mode we hit in production: outer=1800s fired while
    # inner=3030s was still running).
    _INNER_WHISPER_TIMEOUT = max(1800, int(vid_minutes * 60 * 5))
    # Fixed overhead we budget on top of the inner Whisper timeout:
    # preflight model load + Ollama evict + HuggingFace metadata
    # fetch + scene-analysis work that shares this wait_for. 15
    # minutes is comfortably above what we've observed in the wild.
    _TRANS_OUTER_OVERHEAD = 900
    if is_ollama_primary:
        est_windows = max(1, int(metadata["duration"] / tier.window_duration)) if tier.window_duration > 0 else 1
        # Ollama timeouts: generous because local inference is slow but reliable
        _SUMMARY_CLIP_TIMEOUT = max(
            1800,  # Minimum 30 minutes for any video
            est_windows * tier.per_call_timeout_base + 900
        )
        _B64_ENCODE_TIMEOUT = max(300, int(vid_minutes * 10))
        # Outer transcription+scene timeout. Must strictly exceed
        # the inner Whisper subprocess timeout so CPU-fallback
        # transcriptions finish instead of being killed mid-run.
        _trans_scene_timeout = max(
            1800,
            _INNER_WHISPER_TIMEOUT + _TRANS_OUTER_OVERHEAD,
        )
        logger.info(
            "[%s] Ollama-scaled timeouts: extraction=%ds, summary_clip=%ds, "
            "trans_scene=%ds, b64=%ds (%d est. windows, %.1f min video)",
            job_id, _EXTRACTION_TIMEOUT, _SUMMARY_CLIP_TIMEOUT,
            _trans_scene_timeout, _B64_ENCODE_TIMEOUT, est_windows, vid_minutes,
        )
    else:
        _SUMMARY_CLIP_TIMEOUT = max(900, int(vid_minutes * 120))
        _B64_ENCODE_TIMEOUT = max(300, int(vid_minutes * 10))
        # Same invariant: outer must cover inner Whisper + overhead.
        _trans_scene_timeout = max(
            3600,
            _INNER_WHISPER_TIMEOUT + _TRANS_OUTER_OVERHEAD,
        )
    logger.info(
        "[%s] Adaptive timeouts: extraction=%ds, summary_clip=%ds, b64=%ds (%.1f min video)",
        job_id, _EXTRACTION_TIMEOUT, _SUMMARY_CLIP_TIMEOUT, _B64_ENCODE_TIMEOUT, vid_minutes,
    )

    codec_label = metadata.get("codec_name", "unknown")
    pix_fmt_label = metadata.get("pix_fmt", "")
    codec_info = f" [{codec_label}]" if codec_label else ""
    if pix_fmt_label and pix_fmt_label not in ("yuv420p", "yuvj420p"):
        codec_info += f" ({pix_fmt_label})"
    await _update_progress(
        job_id, JobStatus.EXTRACTING_FRAMES, 5,
        f"Metadata extracted — {res} @ {fps_val}fps, {dur_fmt} duration, {mb:.1f}MB{codec_info}",
    )

    # Disk space pre-check — estimate needed space from video metadata
    disk_usage = shutil.disk_usage("/data")
    # Estimate: audio WAV ~1.8MB/min + frames ~3MB + overhead
    estimated_need_mb = max(50, metadata.get("file_size_mb", 100) * 0.3)
    if disk_usage.free < estimated_need_mb * 1024 * 1024:
        raise RuntimeError(
            f"Insufficient disk space: {disk_usage.free // (1024*1024)}MB free, "
            f"estimated {int(estimated_need_mb)}MB needed. "
            f"Please free space on the /data volume."
        )

    # Broadcast GPU info early so user can see what hardware is available
    try:
        from backend.services.clip_exporter import detect_gpu_capabilities, get_encoder_label
        _gpu = detect_gpu_capabilities()
        _gpu_parts = []
        if _gpu.get("cuda_available"):
            _gpu_parts.append(f"Whisper: CUDA ({_gpu.get('gpu_name', 'GPU')})")
        else:
            # Check if GPU is detected but CUDA isn't available
            _gpu_name = _gpu.get("gpu_name", "")
            if _gpu_name and _gpu_name != "None (CPU only)":
                _gpu_parts.append(f"Whisper: CPU (GPU detected: {_gpu_name} — CUDA runtime not available)")
            else:
                _gpu_parts.append("Whisper: CPU")
        _enc_label = get_encoder_label()
        _gpu_parts.append(f"Encoding: {_enc_label}")
        # Add issues hint if GPU detected but encoder fell back to CPU
        _gpu_issues = _gpu.get("gpu_issues", [])
        if _gpu_issues:
            _gpu_parts.append("(GPU passthrough incomplete — check Settings > Advanced)")
        await broadcast_ws(job_id, {
            "type": "status",
            "status": "processing",
            "progress": 5,
            "message": f"Hardware — {' | '.join(_gpu_parts)}",
        })
    except Exception:
        pass  # Non-critical — don't break pipeline if GPU detection fails

    # Step 2 — Frame + Audio Extraction in parallel (5-15%)
    cancel_check()
    file_mb = metadata.get("file_size_mb", 0)
    duration_min = round(metadata["duration"] / 60, 1)
    size_note = f" ({file_mb:.0f}MB, {duration_min}min)" if file_mb > 50 else ""
    await _update_progress(
        job_id, JobStatus.EXTRACTING_FRAMES, 8,
        f"Extracting frames + audio{size_note}...",
    )

    # Estimate total frames for progress scaling
    _est_frame_rate = settings.FRAME_SAMPLE_RATE
    _est_total_frames = max(50, int(metadata["duration"] / _est_frame_rate)) if metadata["duration"] > 0 else 100

    async def _frame_progress(frames_so_far: int):
        extraction_pct = min(1.0, frames_so_far / _est_total_frames)
        pct = 8 + int(extraction_pct * 6)  # 8% to 14%
        await _update_progress(
            job_id, JobStatus.EXTRACTING_FRAMES, pct,
            f"Extracted {frames_so_far} frames so far{size_note}...",
        )

    # ── Re-analyze cache: if frames + audio already exist on disk AND
    #    the source SHA-256 matches what's recorded on the job, skip
    #    the (expensive) re-extract. Saves 30–60% of total wall time
    #    on iterative re-analysis runs (clip tuning loops).
    #
    # Disable with CLIPAI_FORCE_REEXTRACT=1.
    _force_reextract = os.environ.get("CLIPAI_FORCE_REEXTRACT", "").lower() in ("1", "true", "yes")
    cached_extraction = None
    if not _force_reextract:
        try:
            cached_extraction = await _maybe_use_cached_extraction(
                job_id=job_id,
                video_path=video_path,
                frames_dir=frames_dir,
                audio_path=audio_path,
                expected_sha=getattr(job, "source_sha256", "") or "",
            )
        except Exception as _ce:
            logger.info("[%s] Cache probe failed (%s); falling back to fresh extract", job_id, _ce)

    if cached_extraction is not None:
        frames, scene_cut_timestamps = cached_extraction
        async with _stage_timer(job_id, "frame+audio extraction (cached)"):
            logger.info(
                "[%s] Re-using cached extraction: %d frames, %s audio (SHA matched)",
                job_id, len(frames),
                "with" if os.path.isfile(audio_path) else "missing",
            )
            _record_pipeline_warning(
                job_id,
                "Re-used cached frames + audio (re-analyze without source change)",
            )
    else:
        # Run frame extraction and audio extraction concurrently — both are
        # independent FFmpeg reads of the source video, writing to different outputs.
        async with _stage_timer(job_id, "frame+audio extraction"):
            try:
                extraction_result, _ = await asyncio.wait_for(
                    asyncio.gather(
                        extract_frames(
                            video_path, frames_dir,
                            cancel_check=cancel_check, progress_callback=_frame_progress,
                            video_duration=metadata["duration"],
                            video_codec=metadata.get("codec_name", ""),
                        ),
                        extract_audio(
                            video_path, audio_path,
                            cancel_check=cancel_check,
                            precondition=bool(getattr(settings, "WHISPER_AUDIO_PRECONDITION", True)),
                        ),
                    ),
                    timeout=_EXTRACTION_TIMEOUT,
                )
                frames, scene_cut_timestamps = extraction_result
            except asyncio.TimeoutError:
                logger.error("[%s] Frame+audio extraction timed out after %ds", job_id, _EXTRACTION_TIMEOUT)
                raise RuntimeError(
                    f"Frame and audio extraction timed out after {_EXTRACTION_TIMEOUT // 60} minutes. "
                    "The video file may be very large or the container is under heavy load."
                )
        # Persist a tiny sidecar manifest so future cache hits can
        # reconstruct frame timestamps + scene cuts without ffprobe /
        # ffmpeg re-runs.
        try:
            _write_extraction_manifest(frames_dir, frames, scene_cut_timestamps)
        except Exception:
            pass
        # Hash the source AFTER the first successful extract so re-analyze
        # runs can skip re-extracting. Hashing is best-effort and runs in
        # a thread so it doesn't block the event loop.
        try:
            _sha = await asyncio.to_thread(_hash_file_sha256, video_path)
            if _sha:
                await database.update_job_status(job_id, source_sha256=_sha)
        except Exception as _he:
            logger.info("[%s] source hash failed (%s); cache disabled for next run", job_id, _he)
    total_frames = len(frames)

    # Store scene cut timestamps for shot-boundary-aware tracking
    if scene_cut_timestamps:
        await database.update_job_status(job_id, scene_cut_timestamps=scene_cut_timestamps)
        logger.info("[%s] Stored %d scene cut timestamps for tracking", job_id, len(scene_cut_timestamps))

    # ── Face detection (CPU-only, ~0.5s for 60 frames) ──
    # Runs pixel-accurate face detection on extracted frames to augment
    # the AI vision model's subject_x estimates. No GPU needed.
    face_registry = None
    face_results = []  # Will hold FrameFaces for active speaker detection
    dense_face_results = []  # Will hold dense 1fps FrameFaces
    _sparse_face_map = {}  # timestamp -> FrameFaces for scene enrichment
    if settings.SUBJECT_TRACKING_ENABLED:
        try:
            from backend.services.face_detector import detect_faces_batch
            frame_list = [(f.timestamp, f.path) for f in frames]
            logger.info("[%s] Running face detection on %d frames...", job_id, len(frame_list))
            face_results = detect_faces_batch(frame_list)
            # Attach face data to each frame
            for frame, face_data in zip(frames, face_results):
                frame.face_data = face_data
            faces_found = sum(1 for fd in face_results if fd.faces)
            logger.info("[%s] Face detection complete: %d/%d frames have faces", job_id, faces_found, len(frames))

            # Pre-populate sparse face map for scene enrichment later
            for fd in face_results:
                if fd.faces:
                    _sparse_face_map[round(fd.timestamp, 1)] = fd

            # Face detection is complete — all OpenCV resources released.
            import gc
            gc.collect()
        except ImportError:
            logger.info("[%s] Face detection unavailable (mediapipe not installed) — using AI estimates only", job_id)
        except Exception as e:
            logger.warning("[%s] Face detection failed (non-fatal): %s", job_id, e)

    # ── Early gameplay pre-detection ──
    #
    # Runs on the SPARSE face pass (FaceMesh only, no YuNet) +
    # filename hint, BEFORE the expensive dense pass. Fires on:
    #
    #   1. The user picked a gameplay override from the dropdown
    #   2. The filename matches a known game title (e.g. "TF2",
    #      "Valorant", "Apex Legends" — see
    #      :mod:`filename_gameplay_hint`)
    #   3. The sparse FaceMesh pass returned ~zero faces AND a
    #      cheap HUD / crosshair signal fires on sample frames
    #
    # When it fires we:
    #   * Skip the dense face pass entirely — saves ~90s on a
    #     2.5 min clip, and avoids letting YuNet's cartoon-face
    #     false positives override the gameplay decision
    #   * Set ``_is_gameplay = True`` so the downstream reframe
    #     path takes the center-crop fast-path
    #
    # Guarded by a try/except so any exception degrades cleanly
    # back to the legacy post-dense detection block.
    _early_is_gameplay = False
    try:
        from backend.services.content_type_strings import (
            is_gameplay_override as _early_is_gp_override,
        )
        _early_ct_override = getattr(
            job, "content_type_override", "",
        ) or ""
        if _early_is_gp_override(_early_ct_override):
            _early_is_gameplay = True
            logger.info(
                "[%s] Early gameplay pre-detect: user override "
                "'%s' — skipping dense face pass",
                job_id, _early_ct_override,
            )
        else:
            # Respect an EXPLICIT non-gameplay user override. The
            # ``classify_gameplay_content`` heuristic looks at sparse-face
            # ratio + HUD signals — both of which fire on anime title
            # cards and on talking-head streams shot in front of a
            # busy background. If the user already told us it's
            # ``anime`` / ``cartoon`` / ``vlog`` / ``narrative`` /
            # ``music`` etc. we have to BELIEVE them rather than let
            # a noisy heuristic flip the whole pipeline into gameplay
            # mode (which sets every subject_x = 50 and produces a
            # static center crop).
            _ct_lower = _early_ct_override.strip().lower() if _early_ct_override else ""
            _explicit_non_gameplay = _ct_lower in {
                "anime", "anime_dialogue", "animation", "animation_dialogue",
                "cartoon", "cartoon_dialogue",
                "vlog", "podcast", "interview", "narrative", "documentary",
                "music", "music_video", "concert", "performance",
                "sports", "sports_basketball", "sports_racing",
                "tutorial", "talking_head",
            }
            if _explicit_non_gameplay:
                logger.info(
                    "[%s] Early gameplay pre-detect: SKIPPED — user "
                    "explicitly set content_type=%r (non-gameplay). "
                    "Running the normal subject-tracking path so "
                    "subject_x isn't slammed to center.",
                    job_id, _early_ct_override,
                )
            else:
                from backend.services.face_detector import (
                    classify_gameplay_content as _early_cgc,
                )
                _early_sample_paths = [
                    str(f.path) for f in frames[:30]
                ] if frames else []
                _early_filename = getattr(job, "filename", None) or ""
                # Pass an EMPTY list for the dense slot. The early
                # pre-detect runs before the dense+verifier pass, so
                # ``face_results`` here is sparse-only data with no
                # ``is_human_verified`` annotations. Feeding it into
                # the dense slot would let Signal 3 (cartoon
                # contamination) fire on un-verified data because
                # ``verified_ratio == 0`` is indistinguishable from
                # "100% non-human rejection" — the exact bug that
                # branded the Tank vs Tyrese podcast as gameplay.
                _early_result = _early_cgc(
                    [],
                    len(face_results or []),
                    _early_sample_paths,
                    sparse_face_data=face_results or [],
                    filename=_early_filename,
                )
                if _early_result == "gameplay":
                    # Check if ANIME_MODE_DETECTED already fired — if so,
                    # the gameplay heuristic is a false positive from anime
                    # character faces, not actual gameplay content.
                    from backend.services.face_detector import ANIME_MODE_DETECTED as _amd
                    # Also treat a positive filename hint OR user
                    # override matching anime/cartoon as a veto, since
                    # ANIME_MODE_DETECTED needs >100 verified faces
                    # before flipping to True (see "[AnimeMode] deferred"
                    # in face_detector.py).
                    _filename_anime = bool(_early_filename) and any(
                        kw in _early_filename.lower()
                        for kw in ("anime", "ova", "subbed", "dubbed",
                                   "cartoon", "season ", " s0", "[crunchyroll]")
                    )
                    if _amd or _filename_anime:
                        logger.info(
                            "[%s] Early gameplay pre-detect: gameplay signal "
                            "overridden (anime_mode=%s, filename_hint=%s) — "
                            "treating as animated content, not gameplay",
                            job_id, _amd, _filename_anime,
                        )
                    else:
                        _early_is_gameplay = True
                        logger.info(
                            "[%s] Early gameplay pre-detect: gameplay "
                            "(filename='%s', sparse_faces=%d/%d) — "
                            "skipping dense face pass",
                            job_id, _early_filename,
                            sum(1 for f in face_results or [] if f.faces),
                            len(face_results or []),
                        )
    except Exception as _early_e:
        logger.debug(
            "[%s] Early gameplay pre-detect failed (non-fatal): %s",
            job_id, _early_e,
        )

    # ── Early anime hint (Week 2 Part A) ──
    # Hoisted ABOVE the dense-pass gate so a pure-anime job (where the
    # sparse FaceMesh pass returned ~zero faces) can still enter the
    # dense block via the cascade-augmentation path. The hint controls
    # whether detect_faces_dense runs its lbpcascade anime augmentation
    # pass alongside the main YuNet / FaceMesh detectors. Sources, in
    # priority order:
    #   1. User dropdown override (`anime` / `cartoon`) — the most
    #      reliable signal; trust it immediately.
    #   2. ANIME_MODE_DETECTED — flipped by the sparse pass above when
    #      the human-pose verifier rejected >50% of detections. Fires
    #      for stylized content even when the user didn't pick anime
    #      from the dropdown.
    # When neither hint is available, the first dense pass runs
    # live-action-only and the re-run fallback below (after
    # classify_content) picks up the slack.
    _early_anime_hint = False
    try:
        from backend.services.face_detector import ANIME_MODE_DETECTED
    except ImportError:
        ANIME_MODE_DETECTED = False
    try:
        from backend.services.content_type_strings import (
            normalize_ui_content_type,
        )
        _early_ct_override = getattr(
            job, "content_type_override", "",
        ) or ""
        _norm_early = (
            normalize_ui_content_type(_early_ct_override)
            if _early_ct_override else None
        )
        if _norm_early and _norm_early.is_animated:
            _early_anime_hint = True
            logger.info(
                "[%s] Early anime hint = True (user override=%s)",
                job_id, _norm_early.raw,
            )
    except Exception:
        pass
    if not _early_anime_hint and ANIME_MODE_DETECTED:
        _early_anime_hint = True
        logger.info(
            "[%s] Early anime hint = True (ANIME_MODE_DETECTED)",
            job_id,
        )

    # ── Dense face detection (1fps, CPU-only) ──
    # Runs on CPU, no GPU conflict. For 4K VP9 this can take 3-5 minutes.
    # Gate: enter the dense pass when EITHER the sparse pass found faces
    # OR the early anime hint is set. On pure-anime content the sparse
    # FaceMesh / YuNet pass commonly returns ~zero faces (the human-
    # trained models reject stylized characters), and the dense pass
    # is the only place that runs the lbpcascade_animeface augmentation
    # — so without this gate-loosening, animated jobs got 0 dense face
    # tracking and the L1 solver had no per-second face anchors.
    if (
        settings.SUBJECT_TRACKING_ENABLED
        and (face_results or _early_anime_hint)
        and not _early_is_gameplay
    ):
        try:
            from backend.services.face_detector import detect_faces_dense

            video_duration = metadata.get("duration", 0)
            dense_sample_rate = settings.DENSE_FACE_SAMPLE_RATE

            # Cap at 1800 dense frames (~30 min at 1fps)
            max_dense_duration = min(video_duration, 1800)
            if video_duration > 1800:
                dense_sample_rate = max(dense_sample_rate, video_duration / 1800)

            expected_frames = int(max_dense_duration / dense_sample_rate)
            logger.info(
                "[%s] Running dense face detection (%.1fs intervals, ~%d frames)...",
                job_id, dense_sample_rate, expected_frames,
            )
            await _update_progress(
                job_id, JobStatus.EXTRACTING_FRAMES, 15,
                f"Extracting {expected_frames} dense frames for face detection...",
            )

            # Thread-safe progress callback for dense face detection
            _dense_loop = asyncio.get_event_loop()

            def _dense_progress(stage, done, total, extra=None):
                """Called from executor thread — schedules async progress update."""
                if stage == "extracting_frames":
                    pct_done = min(99, 100 * done // max(total, 1))
                    msg = f"Extracting dense frames: {done}/{total} ({pct_done}%)..."
                    pct = 15
                elif stage == "extracting_done":
                    msg = f"Dense frames extracted ({done}/{total}) — running FaceMesh..."
                    pct = 15
                elif stage == "facemesh_start":
                    msg = f"Running FaceMesh on {total} frames (detecting faces + lip aperture)..."
                    pct = 15
                elif stage == "facemesh_progress":
                    pct_done = min(99, 100 * done // max(total, 1))
                    faces_info = f" — {extra} frames with faces" if extra else ""
                    msg = f"FaceMesh: {done}/{total} frames processed ({pct_done}%){faces_info}..."
                    pct = 15
                elif stage == "facemesh_done":
                    faces_found = done  # done = len(results)
                    msg = f"FaceMesh complete ({faces_found} frames) — running YuNet for speaker identification..."
                    pct = 15
                elif stage == "yunet_start":
                    msg = f"Running YuNet + SFace on {total} frames (face embeddings for speaker ID)..."
                    pct = 15
                elif stage == "complete":
                    msg = f"Dense face detection complete: {done} frames analyzed"
                    pct = 15
                else:
                    return
                asyncio.run_coroutine_threadsafe(
                    _update_progress(job_id, JobStatus.EXTRACTING_FRAMES, pct, msg),
                    _dense_loop,
                )

            # Run in executor to avoid blocking the async event loop
            loop = asyncio.get_event_loop()
            dense_face_results = await loop.run_in_executor(
                None,
                lambda: detect_faces_dense(
                    video_path,
                    start=0,
                    end=video_duration,
                    sample_rate=dense_sample_rate,
                    min_confidence=0.4,
                    extract_embeddings=True,
                    progress_callback=_dense_progress,
                    is_animated=_early_anime_hint,
                ),
            )
            dense_with_faces = sum(1 for r in dense_face_results if r.faces)
            logger.info(
                "[%s] Dense face detection: %d frames, %d with faces",
                job_id, len(dense_face_results), dense_with_faces,
            )
        except Exception as e:
            logger.warning("[%s] Dense face detection failed (non-fatal): %s", job_id, e)

    # ── Gameplay content detection ──
    # Check if content is FPS / hero shooter before spending effort on face
    # tracking. If the user declared a gameplay variant via
    # content_type_override (gameplay / gameplay_fps / gameplay_moba /
    # gameplay_tps / gameplay_racing) we trust it immediately. If they
    # declared *any other* recognized override (podcast, debate, vlog,
    # movie, anime, music_video, stream, sports, ...) we respect that
    # and skip the gameplay auto-detect. Otherwise (auto / unknown /
    # invalid) we run the face-rarity heuristic.
    #
    # Note: ``stream`` normalizes to ContentType.GAMING but does NOT
    # take the face-skipping fast path — it has a facecam and still
    # needs the face pipeline for the overlay.
    from backend.services.content_type_strings import (
        is_gameplay_override,
        is_user_override,
        normalize_ui_content_type,
    )

    _is_gameplay = _early_is_gameplay  # Carry forward the early pre-detect result
    _job_data = await database.load_job(job_id)
    _content_override = getattr(_job_data, "content_type_override", "") if _job_data else ""
    # Single canonical lowercase form of the user's content-type pick.
    # Used downstream by both the clip-detection content-type selector
    # AND the per-clip render-plan classifier so they BOTH honor the
    # same user choice without re-deriving it. Empty string when the
    # user didn't pick anything (auto-classify wins in that case).
    _user_ct_str = (_content_override or "").strip().lower()
    _game_type = getattr(_job_data, "game_type", "") if _job_data else ""
    # Phase 2: anime / music sub-dropdown values plumbed end-to-end so
    # the classifier's user-override branch can populate
    # profile.anime_subtype / profile.music_subtype, and Phase 6 / Phase 5
    # downstream tuning can read them off the profile.
    _anime_subtype = getattr(_job_data, "anime_subtype", "") if _job_data else ""
    _music_subtype = getattr(_job_data, "music_subtype", "") if _job_data else ""
    _sports_subtype = getattr(_job_data, "sports_subtype", "") if _job_data else ""
    _normalized_override = normalize_ui_content_type(_content_override)

    # If the filename hint fired in the early pre-detect, inherit the
    # resolved game slug + genre so the downstream center-bias guard
    # fires on the correct CENTER_BIAS_GENRES bucket.
    _filename_game_slug = ""
    _filename_game_genre = ""
    try:
        from backend.services.filename_gameplay_hint import (
            filename_gameplay_hint as _pipeline_fgh,
        )
        _hint_obj = _pipeline_fgh(getattr(job, "filename", ""))
        if _hint_obj.matched:
            _filename_game_slug = _hint_obj.slug
            _filename_game_genre = _hint_obj.genre
    except Exception:
        pass

    if is_gameplay_override(_content_override):
        _is_gameplay = True
        logger.info(
            "[%s] Content type override = %s (user-declared gameplay fast-path)",
            job_id, _normalized_override.raw if _normalized_override else _content_override,
        )
    elif _early_is_gameplay:
        # Early pre-detect already decided gameplay from the sparse
        # face pass + filename hint. Don't re-run the heuristic on
        # the (skipped) dense pass.
        logger.info(
            "[%s] Gameplay detected (early pre-detect) — "
            "filename_slug='%s' genre='%s'",
            job_id, _filename_game_slug or "unknown",
            _filename_game_genre or "unknown",
        )
    elif is_user_override(_content_override):
        # User picked a non-gameplay type (or stream) — respect it and
        # skip auto-detect so we don't silently upgrade their choice.
        logger.info(
            "[%s] Content type override = %s → %s (user-declared, "
            "skipping gameplay auto-detect)",
            job_id,
            _normalized_override.raw,
            _normalized_override.content_type.value,
        )
    else:
        # Auto-detect: run the (now-hardened) face-rarity heuristic
        # when the user picked "auto" / nothing / an unrecognized
        # token. Passes BOTH sparse and dense face data plus the
        # filename so the detector can cross-check YuNet's cartoon
        # false positives against FaceMesh-only sparse results and
        # any filename keyword match.
        try:
            from backend.services.face_detector import classify_gameplay_content
            _dense_or_sparse = dense_face_results or face_results
            _sample_paths = [f.path for f in frames[:30]] if frames else []
            _gp_result = classify_gameplay_content(
                _dense_or_sparse,
                len(_dense_or_sparse),
                _sample_paths,
                sparse_face_data=face_results or None,
                filename=getattr(job, "filename", None),
            )
            if _gp_result == "gameplay":
                _is_gameplay = True
                logger.info("[%s] Auto-detected gameplay content", job_id)
            else:
                logger.info("[%s] Gameplay auto-detection result: %s", job_id, _gp_result)
        except Exception as e:
            logger.warning("[%s] Gameplay detection failed (non-fatal): %s", job_id, e)

    # Default empty crosshair_path; populated below for FPS subtypes.
    crosshair_path: list = []
    if _is_gameplay:
        # Set tracking mode to gameplay — skip face-based tracking entirely.
        # Scene subject_x values will be overridden to 50 later when scenes are available.
        await database.update_job_status(job_id, tracking_mode="gameplay")
        await _update_progress(
            job_id, JobStatus.EXTRACTING_FRAMES, 15,
            "Gameplay content detected — using center-crop tracking with HUD compositing",
        )
        logger.info("[%s] Gameplay mode: will set all scene subject_x=50 (crosshair-centered)", job_id)

        # ── Phase 1: per-frame crosshair tracking for FPS subtypes ──
        # For FPS / hero shooter / Valorant-style content, run the
        # crosshair tracker so the reframe pipeline can use the
        # actual per-frame crosshair x,y instead of a hard-coded
        # (50, 50). Gated on the override matching an FPS-class
        # subtype so MOBA / TPS / racing don't pay the cost.
        try:
            from backend.services.crosshair_tracker import (
                USE_CROSSHAIR_TRACKER,
                track_crosshair_path,
            )
            _is_fps_class = _content_override.lower() in (
                "gameplay_fps", "fps", "hero_shooter", "gameplay",
            )
            if USE_CROSSHAIR_TRACKER and _is_fps_class and frames:
                logger.info(
                    "[%s] Crosshair tracker: scanning %d frames",
                    job_id, len(frames),
                )
                _ch_inputs = [
                    (float(f.timestamp), str(f.path)) for f in frames
                ]
                # Run in executor — cv2 template matching is CPU bound.
                loop = asyncio.get_event_loop()
                crosshair_path = await loop.run_in_executor(
                    None,
                    lambda: track_crosshair_path(_ch_inputs),
                )
                logger.info(
                    "[%s] Crosshair tracker: %d/%d frames have a "
                    "high-confidence crosshair detection",
                    job_id, len(crosshair_path), len(frames),
                )
        except Exception as e:
            logger.warning(
                "[%s] Crosshair tracker failed (non-fatal): %s",
                job_id, e,
            )

    # ── Build face registry ──
    # Prefer dense data with embeddings for identity-based clustering.
    # Falls back to sparse position-based clustering if dense unavailable.
    if settings.SUBJECT_TRACKING_ENABLED and (face_results or dense_face_results) and not _is_gameplay:
        try:
            await _update_progress(
                job_id, JobStatus.EXTRACTING_FRAMES, 15,
                f"Building face registry — identifying unique speakers from {len(dense_face_results or face_results)} frames...",
            )
            from backend.services.face_registry import (
                build_face_registry,
                build_face_registry_with_embeddings,
            )
            if dense_face_results:
                face_registry = build_face_registry_with_embeddings(
                    dense_face_results,
                    min_appearances=3,
                    cosine_threshold=0.25,
                )
                logger.info(
                    "[%s] Face registry built from %d dense frames (embedding-based)",
                    job_id, len(dense_face_results),
                )
            elif face_results:
                face_registry = build_face_registry(face_results)
                logger.info(
                    "[%s] Face registry built from %d sparse frames (position-based fallback)",
                    job_id, len(face_results),
                )

            # ── Anime character clustering re-ID (Week 2 Part C) ──
            # When the content is animated (per the early hint from the
            # dense-detection stage) and the face registry has ≥2 slots,
            # run HSV color fingerprinting over the anime-cascade
            # detections and merge slots whose color palettes collide
            # within the chi-squared threshold. This catches the case
            # where two shots of the same character got assigned
            # different slots because embedding-based clustering had no
            # signal (lbpcascade doesn't emit embeddings).
            #
            # Graceful degradation: when frame_path references are
            # stale (the dense-detector tempdir has been cleaned up)
            # cv2.imread returns None and that slot contributes zero
            # fingerprints. If no slot produces any fingerprints we
            # log and skip the cluster step entirely.
            _early_anime_hint_for_reid = locals().get(
                "_early_anime_hint", False
            )
            if (
                _early_anime_hint_for_reid
                and face_registry
                and len(getattr(face_registry, "slots", []) or []) >= 2
            ):
                try:
                    from backend.services.anime_character_clustering import (
                        USE_ANIME_CHARACTER_CLUSTERING,
                        extract_color_fingerprint,
                        cluster_fingerprints,
                    )
                except ImportError:
                    USE_ANIME_CHARACTER_CLUSTERING = False

                if USE_ANIME_CHARACTER_CLUSTERING:
                    try:
                        import cv2  # anime clustering needs cv2 anyway
                        slot_fingerprints: dict = {}
                        unreadable_frames = 0
                        _source_frames = dense_face_results or face_results or []
                        for ff in _source_frames:
                            frame_path_local = getattr(ff, "frame_path", None)
                            frame_bgr = (
                                cv2.imread(frame_path_local)
                                if frame_path_local else None
                            )
                            if frame_bgr is None:
                                if frame_path_local:
                                    unreadable_frames += 1
                                continue
                            for fi in getattr(ff, "faces", []) or []:
                                slot_id = int(
                                    getattr(fi, "identity_id", -1) or -1
                                )
                                if slot_id < 0:
                                    continue
                                bbox_pct = (
                                    float(getattr(fi, "x_center", 0.0)),
                                    float(getattr(fi, "y_center", 0.0)),
                                    float(getattr(fi, "width", 0.0)),
                                    float(getattr(fi, "height", 0.0)),
                                )
                                fp = extract_color_fingerprint(
                                    frame_bgr, bbox_pct,
                                )
                                slot_fingerprints.setdefault(
                                    slot_id, [],
                                ).append(fp)

                        if not slot_fingerprints:
                            logger.info(
                                "[%s] Anime character clustering skipped: "
                                "no readable frames (unreadable=%d). "
                                "Dense-detector tempdir likely cleaned up; "
                                "re-ID requires persisted frames (future "
                                "work).",
                                job_id, unreadable_frames,
                            )
                        elif len(slot_fingerprints) >= 2:
                            # Average per-slot fingerprints to one centroid.
                            slot_centroids = {
                                sid: [
                                    sum(col) / len(col)
                                    for col in zip(*fps)
                                ]
                                for sid, fps in slot_fingerprints.items()
                                if fps
                            }
                            slot_ids_list = list(slot_centroids.keys())
                            centroid_list = [
                                slot_centroids[s] for s in slot_ids_list
                            ]
                            cluster = cluster_fingerprints(centroid_list)
                            cluster_ids = list(cluster.cluster_ids)
                            # Build remap: slot_id -> canonical slot id
                            # (the lowest slot in the same cluster).
                            canonical: dict = {}
                            for old_id, cid in zip(slot_ids_list, cluster_ids):
                                same = [
                                    sid
                                    for sid, c in zip(
                                        slot_ids_list, cluster_ids,
                                    )
                                    if c == cid
                                ]
                                canonical[old_id] = min(same)
                            merges = [
                                (o, n) for o, n in canonical.items()
                                if o != n
                            ]
                            if merges:
                                logger.info(
                                    "[%s] Anime character clustering "
                                    "merged %d slot pair(s): %s",
                                    job_id, len(merges), merges,
                                )
                                # Apply remap to every detection's
                                # identity_id, then rebuild the registry.
                                for fr_list in (
                                    face_results, dense_face_results,
                                ):
                                    for fr in (fr_list or []):
                                        for fi in getattr(
                                            fr, "faces", [],
                                        ) or []:
                                            cur = getattr(
                                                fi, "identity_id", -1,
                                            )
                                            if cur in canonical:
                                                fi.identity_id = (
                                                    canonical[cur]
                                                )
                                if dense_face_results:
                                    face_registry = (
                                        build_face_registry_with_embeddings(
                                            dense_face_results,
                                            min_appearances=3,
                                            cosine_threshold=0.25,
                                        )
                                    )
                                elif face_results:
                                    face_registry = build_face_registry(
                                        face_results,
                                    )
                    except Exception as _cc_err:
                        logger.warning(
                            "[%s] Anime character clustering failed "
                            "(non-fatal): %s",
                            job_id, _cc_err,
                        )

            if face_registry:
                for frame in frames:
                    frame.face_registry = face_registry

                # ── Audio-first sanity log ──
                # Compare face registry slot count with transcript speaker count
                # (if available) as a sanity check. Large discrepancies suggest
                # the registry is undercounting.
                transcript_speakers = set()
                _trx_for_sanity = transcript if 'transcript' in dir() else None
                if _trx_for_sanity:
                    for seg in _trx_for_sanity:
                        spk = getattr(seg, 'speaker', None)
                        if spk:
                            transcript_speakers.add(spk)
                if transcript_speakers:
                    audio_est = len(transcript_speakers)
                    face_est = len(face_registry.slots)
                    if audio_est > face_est:
                        logger.info(
                            "[%s] audio_speaker_estimate=%d vs face_registry_slots=%d — "
                            "audio suggests more speakers than face registry detected",
                            job_id, audio_est, face_est,
                        )
                    else:
                        logger.info(
                            "[%s] audio_speaker_estimate=%d vs face_registry_slots=%d — consistent",
                            job_id, audio_est, face_est,
                        )

                if face_registry.multi_speaker:
                    slot_info = ", ".join(f"slot{s.slot_id}@{s.x_center:.0f}%" for s in face_registry.slots)
                    logger.info(
                        "[%s] Multi-speaker face registry: %d face slots (%s)",
                        job_id, len(face_registry.slots), slot_info,
                    )
                    await _update_progress(
                        job_id, JobStatus.EXTRACTING_FRAMES, 15,
                        f"Identified {len(face_registry.slots)} speakers — positions: {slot_info}",
                    )
            # ── Safety net: ensure all dense faces have identity_id assigned ──
            if face_registry and face_registry.slots and dense_face_results:
                unassigned = sum(
                    1 for fr in dense_face_results for f in fr.faces
                    if f.identity_id < 0
                )
                if unassigned > 0:
                    logger.warning(
                        "[%s] %d faces still have identity_id=-1 after registry build — "
                        "running assign_identities explicitly",
                        job_id, unassigned,
                    )
                    from backend.services.face_registry import assign_identities
                    assign_identities(dense_face_results, face_registry)
                    still_unassigned = sum(
                        1 for fr in dense_face_results for f in fr.faces
                        if f.identity_id < 0
                    )
                    logger.info(
                        "[%s] After explicit assignment: %d unassigned (was %d)",
                        job_id, still_unassigned, unassigned,
                    )
                else:
                    assigned_count = sum(
                        1 for fr in dense_face_results for f in fr.faces
                        if f.identity_id >= 0
                    )
                    logger.info(
                        "[%s] Identity assignment verified: %d faces have identity_id >= 0",
                        job_id, assigned_count,
                    )

        except Exception as e:
            logger.warning("[%s] Face registry build failed (non-fatal): %s", job_id, e)

    _phase_timings["extraction"] = _pipeline_elapsed()
    logger.info("[%s] Extracted %d frames + audio track", job_id, total_frames)
    _dense_count = len(dense_face_results) if dense_face_results else 0
    await _update_progress(
        job_id, JobStatus.EXTRACTING_FRAMES, 15,
        f"Extracted {total_frames} frames + {_dense_count} dense face frames + audio track{_pipeline_eta(15)}",
    )

    # ── Steps 3+4 — Run transcription and scene analysis CONCURRENTLY ──
    # These two branches are independent: transcription needs audio,
    # scene analysis needs frames. Running them in parallel saves ~50%
    # of total processing time for long videos.

    audio_duration = metadata.get("duration", 0)

    def _fmt_time(sec):
        m, s = divmod(int(sec), 60)
        return f"{m}:{s:02d}"

    def _fmt_eta(sec):
        sec = int(sec)
        if sec < 60:
            return f"~{sec}s remaining"
        m, s = divmod(sec, 60)
        return f"~{m}m {s}s remaining"

    # Track progress from both branches; combined maps to 15-62% of pipeline
    _branch_pct = {"transcription": 0.0, "scene_analysis": 0.0}

    async def _update_branch_progress(
        branch: str, branch_pct: float, status: str, message: str,
    ):
        """Update progress from one concurrent branch, computing combined pipeline %.

        In sequential mode (local GPU), transcription maps to 15-40% and scene
        analysis maps to 40-62%, so each phase gets meaningful progress movement.
        In concurrent mode, both branches share the 15-62% range equally.
        """
        _branch_pct[branch] = min(100.0, branch_pct)

        if _uses_local_gpu:
            # Sequential mode: transcription = 15-40% (25 points), scenes = 40-62% (22 points)
            if branch == "transcription":
                pipeline_pct = 15 + int((_branch_pct["transcription"] / 100) * 25)
            else:
                # Scene analysis starts at 40% (where transcription ended)
                pipeline_pct = 40 + int((_branch_pct["scene_analysis"] / 100) * 22)
        else:
            # Concurrent mode: each branch contributes half the 15-62% range
            combined = (_branch_pct["transcription"] + _branch_pct["scene_analysis"]) / 200
            pipeline_pct = 15 + int(combined * 47)

        eta = _pipeline_eta(pipeline_pct)
        await _update_progress(job_id, status, min(62, pipeline_pct), message + eta)

    # Shared flag: was subprocess Whisper used? (accessible from VRAM release code)
    _subprocess_whisper_used = [False]

    # ── Branch A: Transcription (audio already extracted in Step 2) ──
    async def _branch_transcription():
        cancel_check()
        lang_label = job.language if job.language else "auto-detect"

        # ── Ensure Whisper model is downloaded before starting ──
        # Without this, the subprocess tries to download from HuggingFace
        # during transcription, which can timeout and fail.
        _transcription_phase_start[0] = _time.monotonic()
        from backend.services.transcription import preflight_whisper_check, is_whisper_model_cached, ensure_whisper_model_downloaded
        if not is_whisper_model_cached(settings.WHISPER_MODEL):
            await _update_branch_progress("transcription", 1, JobStatus.TRANSCRIBING,
                f"Downloading Whisper model ({settings.WHISPER_MODEL})...")
            logger.info("[%s] Whisper model '%s' not cached — downloading before transcription", job_id, settings.WHISPER_MODEL)
            # Run download in thread to avoid blocking event loop
            _dl_ok = await asyncio.get_event_loop().run_in_executor(
                None, ensure_whisper_model_downloaded, settings.WHISPER_MODEL, 600,
            )
            if not _dl_ok:
                raise RuntimeError(
                    f"Failed to download Whisper model '{settings.WHISPER_MODEL}'. "
                    f"Check network connectivity or try a smaller model in Settings."
                )

        # ── Pre-flight: verify Whisper model loads and CUDA works ──
        # This catches CUDA OOM and corrupted models BEFORE committing
        # to a potentially hour-long transcription.
        from backend.services.transcription import preflight_whisper_check
        await _update_branch_progress("transcription", 2, JobStatus.TRANSCRIBING,
            f"Verifying Whisper model ({settings.WHISPER_MODEL})...")
        _preflight = await preflight_whisper_check(timeout=90)
        if _preflight["ok"]:
            _requested = _preflight.get("requested_device", _preflight["device"])
            _actual = _preflight["device"]
            _device_msg = f"device={_actual}" if _requested == _actual else f"requested={_requested} actual={_actual}"
            logger.info(
                "[%s] Whisper preflight passed: model=%s %s load_time=%dms",
                job_id, _preflight["model"], _device_msg,
                _preflight.get("load_time_ms", 0),
            )
        else:
            logger.error(
                "[%s] Whisper preflight FAILED: %s (model=%s device=%s)",
                job_id, _preflight.get("error", "unknown"), _preflight["model"], _preflight["device"],
            )
            # Try to provide actionable error message
            err = _preflight.get("error", "Unknown error")
            if "timed out" in err.lower():
                raise RuntimeError(
                    f"Whisper model '{_preflight['model']}' failed to load within 90 seconds. "
                    f"The model may be downloading for the first time, or GPU memory is exhausted. "
                    f"Try a smaller model (e.g., 'small') in Settings > AI Provider."
                )
            elif "cuda" in err.lower() or "gpu" in err.lower():
                raise RuntimeError(
                    f"Whisper failed on GPU: {err}. "
                    f"Try disabling GPU acceleration in Settings > Advanced, "
                    f"or use a smaller model."
                )
            else:
                raise RuntimeError(f"Whisper model check failed: {err}")

        # Include GPU/device info in the initial transcription message
        from backend.services.transcription import whisper_device_info
        _wdev = whisper_device_info
        if _wdev["device"] == "cuda" and _wdev["gpu_name"]:
            device_label = f"GPU: {_wdev['gpu_name']} ({_wdev['compute_type']})"
        elif _wdev["device"] == "cuda":
            device_label = f"GPU: CUDA ({_wdev['compute_type']})"
        else:
            device_label = f"CPU ({_wdev['compute_type']})"
        await _update_branch_progress("transcription", 5, JobStatus.TRANSCRIBING,
            f"Transcribing audio ({lang_label}) — {device_label}")

        async def _transcribe_progress(info: dict):
            # Handle phase-based progress from whisper worker (model loading, VAD, etc.)
            phase = info.get("phase")
            if phase and phase in ("model_loading", "model_loaded", "vad_start"):
                msg = info.get("message", f"Whisper: {phase}...")
                await _update_branch_progress("transcription", 3, JobStatus.TRANSCRIBING, msg)
                return

            pct = info["pct"]
            lang_info = f" [{info['lang']}]" if info.get("lang") else ""
            pos = _fmt_time(info["position_sec"])
            total = _fmt_time(audio_duration) if audio_duration > 0 else "?"
            # Pipeline-wide ETA is appended by _update_branch_progress — no
            # branch-specific ETA here to avoid confusing double "remaining" messages.
            parts = [f"Transcribing{lang_info}: {pos} / {total}"]
            parts.append(f"{info['segments']} segments")
            parts.append(f"via {device_label}")
            branch_pct = 5 + pct * 0.95  # 5-100% (audio already extracted in Step 2)
            await _update_branch_progress("transcription", branch_pct,
                JobStatus.TRANSCRIBING, " \u2014 ".join(parts))

        # Determine the Whisper task: "translate" for direct audio→English,
        # "transcribe" for same-language transcription.
        # Whisper's native translate is dramatically more accurate than
        # transcribe→LLM-translate because it uses the raw audio signal.
        whisper_task = "transcribe"
        if job.subtitle_language and job.subtitle_language.strip().lower() == "en":
            audio_lang = job.language.strip().lower() if job.language else ""
            if audio_lang and audio_lang != "en":
                whisper_task = "translate"
                logger.info("[%s] Using Whisper native translate: %s audio → English subtitles", job_id, audio_lang)
            elif not audio_lang:
                whisper_task = "translate"
                logger.info("[%s] Using Whisper native translate: auto-detect → English subtitles", job_id)

        # Build initial_prompt for Whisper from available context
        # This helps Whisper recognize proper nouns, technical terms, etc.
        import re
        initial_prompt_parts = []

        # For translate tasks, prime the decoder for natural English output
        if whisper_task == "translate":
            audio_lang_label = job.language.strip().lower() if job.language else ""
            if audio_lang_label == "ja" or not audio_lang_label:
                translate_prompt = (
                    "This is a casual Japanese conversation translated to natural English. "
                    "Use complete sentences. Keep names as-is."
                )
                initial_prompt_parts.append(translate_prompt)

        # Deliberately do NOT add the raw filename to the Whisper
        # initial_prompt. Whisper's decoder treats the prompt as prior
        # context and will happily echo it verbatim when the audio is
        # quiet or music-only — we saw this catastrophically on anime
        # episodes whose filename is "Show - S01E01 - Title.mkv". The
        # transcript filled up with repeated title strings, the
        # hallucination filter then stripped 39/44 segments, and the
        # resulting 0.8% transcription coverage starved downstream
        # subtitle / highlight selection.
        #
        # We still try to extract a small set of *names* (proper nouns
        # with no S01E01 / episode markers) as a weak pronunciation
        # hint, which is what initial_prompt is actually good for.
        if job.filename:
            name_clean = re.sub(r'\.[^.]+$', '', job.filename)
            name_clean = re.sub(r'[-_\[\](){}]', ' ', name_clean)
            # Drop episode/season markers that commonly get echoed
            # back as lyrics when Whisper runs dry.
            name_clean = re.sub(
                r'\b(?:s\d{1,2}e\d{1,3}|ep?\d{1,3}|season\s*\d+|episode\s*\d+|'
                r'\d{3,4}p|x264|x265|h26[45]|hevc|web.?dl|webrip|bluray|bdrip|'
                r'dvdrip|hdtv|amzn|hulu|nf|dsnp|mkv|mp4|avi|aac|flac)\b',
                ' ', name_clean, flags=re.IGNORECASE,
            )
            name_clean = re.sub(r'\b\d+\b', ' ', name_clean)
            name_clean = re.sub(r'\s+', ' ', name_clean).strip()
            # Only keep the cleaned string as a pronunciation hint when
            # it looks like a proper noun (short, alphabetic, 1-3
            # tokens). Longer strings are almost always re-echoed.
            tokens = [t for t in name_clean.split() if len(t) >= 3]
            if 1 <= len(tokens) <= 3 and all(t.replace("'", "").isalpha() for t in tokens):
                initial_prompt_parts.append(" ".join(tokens))
        initial_prompt = ". ".join(initial_prompt_parts) if initial_prompt_parts else ""

        # Use subprocess transcription when GPU is enabled to fully release
        # CTranslate2's CUDA context (~1.6GB) after Whisper completes.
        # torch.cuda.empty_cache() is a no-op (CUDA version mismatch).
        # Subprocess exit is the ONLY way to reclaim CTranslate2's VRAM.
        # Always use subprocess for Whisper when GPU is available.
        # The subprocess isolates CTranslate2's CUDA context and releases
        # ALL GPU memory when it exits. This matters for both Ollama (needs
        # GPU for vision/text) and cloud providers (GPU still used for
        # Whisper transcription + NVENC encoding). Without subprocess,
        # the in-process path on CPU is 10-30x slower.
        _use_subprocess_whisper = settings.GPU_ACCELERATION_ENABLED
        _subprocess_whisper_used[0] = _use_subprocess_whisper
        if _use_subprocess_whisper:
            logger.info("[%s] Using subprocess Whisper (GPU mode) to release CUDA memory after", job_id)
            # ── Pre-Whisper VRAM eviction ──
            # On 4GB-class GPUs (GTX 1650, RTX 3050 4GB, etc.) the
            # in-pipeline state at this point typically holds 1.5-2 GB
            # of CUDA context (Ollama models that haven't unloaded,
            # YOLO weights still resident, leftover torch reservation).
            # Whisper-medium needs ~2.2 GB on top of that, so it OOMs
            # mid-stream without intervention. Try to free everything
            # we can before the subprocess gate decides which model to
            # load — the gate then sees the actual free VRAM and can
            # downgrade gracefully rather than launching a doomed run.
            try:
                async with httpx.AsyncClient(timeout=8.0) as _hc:
                    # Unload every Ollama model so its CUDA context exits.
                    _ps = await _hc.get(f"{settings.OLLAMA_HOST}/api/ps")
                    if _ps.status_code == 200:
                        for _m in (_ps.json().get("models") or []):
                            _name = _m.get("name") or ""
                            if _name:
                                try:
                                    await _hc.post(
                                        f"{settings.OLLAMA_HOST}/api/generate",
                                        json={"model": _name, "keep_alive": 0},
                                    )
                                except Exception:
                                    pass
            except Exception as _evict_err:
                logger.debug("[%s] Pre-Whisper Ollama unload failed: %s", job_id, _evict_err)
            try:
                release_torch_gpu_memory()
            except Exception:
                pass
            # Brief settle so the CUDA driver actually returns the
            # blocks before nvidia-smi reports them free.
            await asyncio.sleep(2.5)
            try:
                from backend.services.transcription import _get_gpu_free_mb
                _free_now = _get_gpu_free_mb()
                if _free_now > 0:
                    logger.info(
                        "[%s] Pre-Whisper VRAM evict: %dMB free now",
                        job_id, _free_now,
                    )
                    _record_pipeline_warning(
                        job_id,
                        f"info: Whisper preflight: {_free_now}MB free GPU after evicting Ollama / torch",
                    )
            except Exception:
                pass

            # Timeout: audio_duration * 5 — accommodates CPU fallback (0.5-1x real-time)
            # GPU: ~10-30x real-time, CPU: ~0.5-1x real-time. Use 5x for safety.
            # Derived from the same formula as ``_INNER_WHISPER_TIMEOUT`` above
            # so the outer ``_trans_scene_timeout`` is guaranteed to cover us.
            _whisper_timeout = (
                max(1800, int(audio_duration * 5)) if audio_duration > 0 else 3600
            )
            # Clamp to the outer-scope computed value so the inner and outer
            # stay perfectly aligned even if future edits touch one formula
            # and miss the other.
            _whisper_timeout = min(_whisper_timeout, _INNER_WHISPER_TIMEOUT)
            logger.info("[%s] Whisper subprocess timeout: %ds for %.0fs audio", job_id, _whisper_timeout, audio_duration)
            try:
                result = await asyncio.wait_for(
                    transcribe_audio_subprocess(
                        audio_path, language=job.language, task=whisper_task,
                        initial_prompt=initial_prompt, audio_duration=audio_duration,
                        progress_callback=_transcribe_progress,
                        is_animated=_early_anime_hint,
                        cancel_check=cancel_check,
                    ),
                    timeout=_whisper_timeout,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "[%s] Whisper subprocess timed out after %ds — killing process",
                    job_id, _whisper_timeout,
                )
                raise RuntimeError(f"Whisper transcription timed out after {_whisper_timeout // 60} minutes")
            logger.info("[%s] Whisper subprocess exited — CTranslate2 CUDA memory fully reclaimed", job_id)
        else:
            result = await transcribe_audio(
                audio_path, language=job.language, task=whisper_task,
                initial_prompt=initial_prompt, cancel_check=cancel_check,
                progress_callback=_transcribe_progress, audio_duration=audio_duration,
                is_animated=_early_anime_hint,
            )

        # ── CRASH RECOVERY: If Whisper returned 0 segments on a video with
        # real audio, it likely OOM'd or crashed. Retry with a smaller model.
        # Use subprocess mode to preserve VRAM isolation (in-process retry
        # would re-create CTranslate2's CUDA context in the main process).
        if not result and audio_duration > 10:
            logger.error(
                "[%s] Whisper returned 0 segments for %.0fs audio (model=%s) — "
                "likely CTranslate2 silent OOM. Retrying with 'medium' on GPU...",
                job_id, audio_duration, settings.WHISPER_MODEL,
            )
            await _update_branch_progress("transcription", 10, JobStatus.TRANSCRIBING,
                "Transcription failed — retrying with medium model on GPU...")

            original_model = settings.WHISPER_MODEL
            original_beam = settings.WHISPER_BEAM_SIZE
            try:
                settings.WHISPER_MODEL = "medium"
                settings.WHISPER_BEAM_SIZE = 1  # Greedy decode — lowest VRAM usage
                result = await transcribe_audio_subprocess(
                    audio_path, language=job.language, task=whisper_task,
                    initial_prompt=initial_prompt, audio_duration=audio_duration,
                    progress_callback=_transcribe_progress,
                    is_animated=_early_anime_hint,
                    cancel_check=cancel_check,
                )
                logger.info(
                    "[%s] Retry transcription (medium/GPU) produced %d segments",
                    job_id, len(result),
                )
            finally:
                settings.WHISPER_MODEL = original_model
                settings.WHISPER_BEAM_SIZE = original_beam

            # If GPU retry with medium also failed, try CPU as last resort
            if not result and audio_duration > 10:
                logger.error(
                    "[%s] GPU retry with 'medium' also returned 0 segments. "
                    "Trying CPU as last resort (may take %.0f minutes)...",
                    job_id, audio_duration / 60,
                )
                await _update_branch_progress("transcription", 10, JobStatus.TRANSCRIBING,
                    "GPU transcription failed — retrying on CPU (slower)...")

                original_gpu = settings.GPU_ACCELERATION_ENABLED
                try:
                    settings.WHISPER_MODEL = "medium"
                    settings.WHISPER_BEAM_SIZE = 1
                    settings.GPU_ACCELERATION_ENABLED = False
                    result = await transcribe_audio_subprocess(
                        audio_path, language=job.language, task=whisper_task,
                        initial_prompt=initial_prompt, audio_duration=audio_duration,
                        progress_callback=_transcribe_progress,
                        is_animated=_early_anime_hint,
                        cancel_check=cancel_check,
                    )
                    logger.info(
                        "[%s] CPU fallback transcription produced %d segments",
                        job_id, len(result),
                    )
                finally:
                    settings.WHISPER_MODEL = original_model
                    settings.WHISPER_BEAM_SIZE = original_beam
                    settings.GPU_ACCELERATION_ENABLED = original_gpu

        # ── TACT Phases 2-5: Coverage ladder + reconciliation + translation ──
        # Replaces the legacy gap-fill + Phase 1 post-hoc ledger blocks.
        # The ledger is the source of truth: bootstrap from the
        # post-Whisper segments + quarantined segments, optionally run
        # disjoint-offset and consensus passes and reconcile, run the
        # ladder against (uncovered ∪ quarantined), persist the report.
        #
        # When TACT_LADDER_ENABLED=False, the elif at the bottom of
        # this block runs the legacy gap-fill path verbatim — this is
        # the byte-identical-with-flags-off contract.
        if (getattr(settings, "TACT_LADDER_ENABLED", True)
                and result and audio_duration > 30):
            try:
                from backend.services.transcription_ledger import (
                    CoverageLedger,
                )
                from backend.services.escalation_ladder import (
                    EscalationContext, escalate_uncovered_intervals_sync,
                )
                from backend.services.transcription import (
                    get_last_quarantined_segments,
                )
                from backend.services.transcription_reconciler import (
                    get_last_reconciliation_stats, reconcile_n_passes,
                )
                from backend.services.active_speaker import build_vad_presence

                # Collect passes for reconciliation. Primary always
                # present. Offset and consensus are conditional.
                passes = [("primary", list(result))]
                chunk_grid_offsets: dict[str, float] = {"primary": 0.0}

                # ── Phase 3: Disjoint-offset second pass ──
                if (getattr(settings, "TACT_DISJOINT_OFFSET_ENABLED", True)
                        and audio_duration > 60):
                    try:
                        offset_sec = float(getattr(
                            settings, "TACT_DISJOINT_OFFSET_SEC", 15.0,
                        ))
                        await _update_branch_progress(
                            "transcription", 90, JobStatus.TRANSCRIBING,
                            f"Running phase-shifted second pass "
                            f"(+{offset_sec}s offset)...",
                        )
                        offset_result = await transcribe_audio_subprocess(
                            audio_path, language=job.language,
                            task=whisper_task,
                            initial_prompt=initial_prompt,
                            audio_duration=audio_duration,
                            progress_callback=None,
                            is_animated=_early_anime_hint,
                            cancel_check=cancel_check,
                            offset_sec=offset_sec,
                        )
                        passes.append(
                            (f"offset_{int(offset_sec)}s", list(offset_result)),
                        )
                        chunk_grid_offsets[f"offset_{int(offset_sec)}s"] = offset_sec
                    except Exception as e:
                        logger.warning(
                            "[%s] offset pass failed (%s) — primary only",
                            job_id, e,
                        )

                # ── Phase 4: Independent-architecture consensus pass ──
                if getattr(settings, "TACT_CONSENSUS_ENABLED", False):
                    try:
                        from backend.services.parakeet_transcriber import (
                            can_run_consensus,
                            transcribe_with_parakeet_subprocess,
                        )
                        gate = can_run_consensus()
                        if gate.allowed:
                            await _update_branch_progress(
                                "transcription", 92, JobStatus.TRANSCRIBING,
                                "Running independent-architecture consensus "
                                "pass (Parakeet)...",
                            )
                            parakeet_result = await transcribe_with_parakeet_subprocess(
                                audio_path,
                                language=job.language or "",
                                cancel_check=cancel_check,
                            )
                            passes.append(("parakeet", list(parakeet_result)))
                            # Parakeet's TDT decoder doesn't have a fixed
                            # 30 s chunk grid, so distance-to-boundary
                            # tiebreaks favour it appropriately.
                            chunk_grid_offsets["parakeet"] = 0.0
                        else:
                            logger.info(
                                "[%s] consensus pass declined: %s",
                                job_id, gate.reason,
                            )
                    except Exception as e:
                        logger.warning(
                            "[%s] consensus pass failed (%s)", job_id, e,
                        )

                # ── Reconcile ──
                if len(passes) >= 2:
                    reconciled, _recon_stats = reconcile_n_passes(
                        passes,
                        chunk_grid_offsets=chunk_grid_offsets,
                        align_tolerance_sec=float(getattr(
                            settings, "TACT_RECONCILE_ALIGN_TOLERANCE_SEC", 0.2,
                        )),
                    )
                    logger.info(
                        "[%s] reconciliation: passes=%s words=%d "
                        "agreed=%d disagreed=%d single=%d contested=%d",
                        job_id, [n for n, _ in passes],
                        _recon_stats.total_words, _recon_stats.agreed,
                        _recon_stats.disagreed, _recon_stats.single_source,
                        _recon_stats.contested,
                    )
                    result = reconciled

                # ── Build the ledger ──
                await _update_branch_progress(
                    "transcription", 95, JobStatus.TRANSCRIBING,
                    "Building coverage ledger and escalating uncovered audio...",
                )
                bin_ms = int(getattr(settings, "TACT_LEDGER_BIN_MS", 20))
                ledger = CoverageLedger(
                    int(audio_duration * 1000), bin_ms=bin_ms,
                )
                ledger.from_segments(list(result), source_pass="whisper_main")
                if getattr(settings, "TACT_QUARANTINE_HALLUCINATIONS", True):
                    quarantined = get_last_quarantined_segments()
                    if quarantined:
                        ledger.from_segments(
                            quarantined,
                            source_pass="quarantined_hallucinations",
                            status="quarantined",
                            flag_key="quarantine_reason",
                        )

                try:
                    vad_intervals = build_vad_presence(audio_path) or []
                except Exception:
                    vad_intervals = []

                import threading as _threading
                _ladder_cancel_event = _threading.Event()
                ctx = EscalationContext(
                    audio_path=audio_path,
                    audio_duration_ms=int(audio_duration * 1000),
                    language=(job.language or "en"),
                    task=whisper_task,
                    initial_prompt=initial_prompt,
                    is_animated=_early_anime_hint,
                    neighbor_text_before="",
                    neighbor_text_after="",
                    vad_intervals=vad_intervals,
                    event_priors=None,
                    budget_ms_remaining=int(
                        float(getattr(
                            settings, "TACT_LADDER_MAX_AUDIO_SEC", 600.0,
                        )) * 1000,
                    ),
                    per_interval_timeout_sec=float(getattr(
                        settings, "TACT_LADDER_PER_INTERVAL_TIMEOUT_SEC", 30.0,
                    )),
                    cancel_event=_ladder_cancel_event,
                )

                # The ladder runs in a thread pool; asyncio cancellation
                # cannot kill the thread directly, so we set a
                # cancellation event the ladder polls between intervals.
                # The wall-clock cap also protects against pathological
                # inputs (dozens of uncovered intervals on CPU).
                _ladder_wall_clock_timeout_sec = float(getattr(
                    settings, "TACT_LADDER_WALL_CLOCK_TIMEOUT_SEC", 360.0,
                ))
                _loop = asyncio.get_event_loop()
                _ladder_future = _loop.run_in_executor(
                    None,
                    functools.partial(
                        escalate_uncovered_intervals_sync, ledger, ctx,
                        max_audio_sec=float(getattr(
                            settings, "TACT_LADDER_MAX_AUDIO_SEC", 600.0,
                        )),
                    ),
                )
                try:
                    stats = await asyncio.wait_for(
                        asyncio.shield(_ladder_future),
                        timeout=_ladder_wall_clock_timeout_sec,
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError) as _e:
                    logger.warning(
                        "[%s] escalation ladder cancelled (%s) — "
                        "signalling worker to drain remaining intervals "
                        "via Rung 6",
                        job_id, type(_e).__name__,
                    )
                    _ladder_cancel_event.set()
                    # Give the worker a brief grace window to finish the
                    # in-flight interval and unwind through Rung 6.
                    try:
                        stats = await asyncio.wait_for(
                            asyncio.shield(_ladder_future),
                            timeout=float(ctx.per_interval_timeout_sec) + 5.0,
                        )
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        logger.error(
                            "[%s] escalation ladder did not drain in time; "
                            "abandoning thread (orphaned subprocess will "
                            "exit on its own per-interval timeout)",
                            job_id,
                        )
                        if isinstance(_e, asyncio.CancelledError):
                            raise
                        # On wall-clock timeout, fall through with an
                        # empty stats record so the rest of the pipeline
                        # can continue with whatever the ledger has.
                        from backend.services.escalation_ladder import (
                            EscalationStats as _EscStats,
                        )
                        stats = _EscStats()
                        stats.skipped_reason = "wall_clock_timeout"

                # Coverage invariant. Rung 6 should make this impossible
                # to fire; explicit check makes a regression loud, not
                # silent.
                uncovered_after = ledger.query({"uncovered"})
                if uncovered_after:
                    logger.error(
                        "[%s] TACT coverage invariant violated: %d "
                        "uncovered intervals after ladder (rung_6 "
                        "should have caught these)",
                        job_id, len(uncovered_after),
                    )

                result = ledger.to_segments()
                coverage_report = ledger.to_report_dict()
                coverage_report["voiced_coverage_ratio"] = round(
                    ledger.voiced_coverage_ratio(vad_intervals), 4,
                )
                coverage_report["ladder_stats"] = stats.as_dict()
                _recon = get_last_reconciliation_stats()
                if _recon is not None:
                    coverage_report["reconciliation"] = _recon.as_dict()

                # ── Phase 5: Translation track ──
                if (whisper_task == "translate"
                        or getattr(settings, "TACT_TRANSLATION_TRACK_ENABLED", False)):
                    try:
                        from backend.services.translation_track import (
                            run_translation_track,
                        )
                        target_lang = (
                            getattr(job, "target_language", None)
                            or getattr(job, "subtitle_language", "")
                            or "en"
                        )
                        translation_ledger, _t_stats = await _loop.run_in_executor(
                            None,
                            functools.partial(
                                run_translation_track, ledger,
                                target_language=target_lang,
                                backend=str(getattr(
                                    settings, "TACT_TRANSLATION_BACKEND",
                                    "whisper_passthrough",
                                )),
                            ),
                        )
                        coverage_report = ledger.to_paired_report_dict(
                            translation_ledger,
                        )
                        coverage_report["ladder_stats"] = stats.as_dict()
                        if _recon is not None:
                            coverage_report["reconciliation"] = _recon.as_dict()
                        coverage_report["translation_stats"] = _t_stats.as_dict()
                    except Exception as e:
                        logger.warning(
                            "[%s] translation track failed (%s) — "
                            "source-only report", job_id, e,
                        )

                await database.update_job_status(
                    job_id, coverage_report=coverage_report,
                )
                logger.info(
                    "[%s] coverage_ledger: ratio=%.3f voiced=%.3f "
                    "uncovered=%.3f spans=%d ladder=%s",
                    job_id,
                    coverage_report.get("coverage_ratio")
                    or coverage_report.get("source", {}).get("coverage_ratio", 0.0),
                    coverage_report.get("voiced_coverage_ratio") or 0.0,
                    coverage_report.get("uncovered_ratio")
                    or coverage_report.get("source", {}).get("uncovered_ratio", 0.0),
                    coverage_report.get("span_count")
                    or coverage_report.get("source", {}).get("span_count", 0),
                    stats.as_dict(),
                )
            except Exception as e:
                logger.warning(
                    "[%s] TACT ladder failed (%s) — falling back to "
                    "legacy gap-fill", job_id, e,
                )
                if (getattr(settings, "WHISPER_GAP_FILL_ENABLED", True)
                        and result):
                    try:
                        from backend.services.transcription_gap_filler import (
                            fill_transcript_gaps,
                        )
                        result_list, _gap_stats = await fill_transcript_gaps(
                            audio_path, list(result), audio_duration,
                            language=job.language, task=whisper_task,
                            is_animated=_early_anime_hint,
                            initial_prompt=initial_prompt,
                        )
                        result = result_list
                    except Exception as e2:
                        logger.warning(
                            "[%s] legacy gap-fill also failed (%s)",
                            job_id, e2,
                        )
        elif (getattr(settings, "WHISPER_GAP_FILL_ENABLED", True)
                and result and audio_duration > 30):
            # TACT ladder disabled by config — preserve legacy gap-fill
            # path exactly as it was. Byte-identical-with-flags-off.
            try:
                from backend.services.transcription_gap_filler import (
                    fill_transcript_gaps,
                )
                await _update_branch_progress(
                    "transcription", 95, JobStatus.TRANSCRIBING,
                    "Checking transcript coverage for missed speech...",
                )
                result_list, gap_stats = await fill_transcript_gaps(
                    audio_path,
                    list(result),
                    audio_duration,
                    language=job.language,
                    task=whisper_task,
                    is_animated=_early_anime_hint,
                    initial_prompt=initial_prompt,
                    max_gaps=int(getattr(settings, "WHISPER_GAP_FILL_MAX_GAPS", 40)),
                    max_fill_audio_sec=float(getattr(settings, "WHISPER_GAP_FILL_MAX_AUDIO_SEC", 600.0)),
                )
                if gap_stats.segments_added > 0:
                    logger.info(
                        "[%s] gap-fill recovered %d segments across %d gaps "
                        "(voiced coverage %.1f%% → %.1f%%)",
                        job_id, gap_stats.segments_added, gap_stats.gaps_filled,
                        gap_stats.coverage_before * 100,
                        gap_stats.coverage_after * 100,
                    )
                    result = result_list
                else:
                    logger.info(
                        "[%s] gap-fill: no new segments (gaps_found=%d, reason=%s)",
                        job_id, gap_stats.gaps_found,
                        gap_stats.skipped_reason or "empty_after_filter",
                    )
            except Exception as e:
                logger.warning(
                    "[%s] gap-fill pass failed (%s) — continuing with main transcript",
                    job_id, e,
                )

        await database.update_job_status(job_id, transcript=list(result))

        # ── Inline heuristic diarization ──
        # Subprocess Whisper defers speaker assignment (all = "Speaker 1").
        # Run face-aware or heuristic diarization so active speaker detection
        # has real speaker labels. Instant, no GPU needed.
        if result and len(set(s.speaker for s in result)) <= 1:
            try:
                speaker_count_before = len(set(s.speaker for s in result))
                _face_data_for_diar = dense_face_results if dense_face_results else face_results

                # Try face-aware diarization first (uses face positions for speaker changes)
                if _face_data_for_diar and face_registry and face_registry.multi_speaker:
                    from backend.services.transcription import assign_speakers_with_face_data
                    raw_segs = [
                        {
                            "start": s.start, "end": s.end, "text": s.text,
                            "words": [{"start": w.start, "end": w.end, "word": w.word} for w in s.words] if s.words else None,
                            "confidence": s.confidence,
                            "avg_logprob": s.avg_logprob,
                            "no_speech_prob": s.no_speech_prob,
                        }
                        for s in result
                    ]
                    result = assign_speakers_with_face_data(
                        raw_segs, _face_data_for_diar, face_registry,
                        scene_descriptions=None,  # Scenes not yet available at this pipeline stage
                    )
                    logger.info(
                        "[%s] Face-aware diarization: %d -> %d speakers",
                        job_id, speaker_count_before, len(set(s.speaker for s in result)),
                    )

                # Fallback: pure heuristic diarization
                if len(set(s.speaker for s in result)) <= 1:
                    from backend.services.transcription import assign_speakers_heuristic
                    raw_segs = [
                        {
                            "start": s.start, "end": s.end, "text": s.text,
                            "words": [{"start": w.start, "end": w.end, "word": w.word} for w in s.words] if s.words else None,
                            "confidence": s.confidence,
                            "avg_logprob": s.avg_logprob,
                            "no_speech_prob": s.no_speech_prob,
                        }
                        for s in result
                    ]
                    result = assign_speakers_heuristic(raw_segs)
                    logger.info(
                        "[%s] Heuristic diarization: %d -> %d speakers",
                        job_id, speaker_count_before, len(set(s.speaker for s in result)),
                    )

                # Update stored transcript with speaker labels
                await database.update_job_status(job_id, transcript=list(result))

                # Store speaker names for frontend
                speaker_set = set(s.speaker for s in result)
                if len(speaker_set) >= 2:
                    speaker_names = {speaker: speaker for speaker in sorted(speaker_set)}
                    await database.update_job_status(job_id, speaker_names=speaker_names)
            except Exception as e:
                logger.warning("[%s] Inline diarization failed (non-fatal): %s", job_id, e)
        # so the translator knows the source language
        if not job.language and result:
            from backend.services.transcription import _last_detected_language
            detected = _last_detected_language.get("lang", "")
            if detected:
                job.language = detected
                await database.update_job_status(job_id, language=detected)
                logger.info("[%s] Auto-detected language: %s", job_id, detected)

        # NOTE: Transcript polishing and subtitle translation are deferred to
        # a background task that runs AFTER analysis completes (see
        # _background_post_processing). This saves ~5+ minutes on the critical
        # path — raw Whisper output is good enough for clip detection.

        speaker_count = len(set(s.speaker for s in result))
        from backend.services.transcription import _last_diarization_method
        diar_method = _last_diarization_method.get("method", "heuristic")
        if diar_method == "deferred":
            diar_label = "speaker detection deferred to post-processing"
        elif diar_method == "neural":
            diar_label = f"{speaker_count} speaker{'s' if speaker_count != 1 else ''} detected via neural (pyannote)"
        else:
            diar_label = f"{speaker_count} speaker{'s' if speaker_count != 1 else ''} detected via heuristic (pause-based)"

        # The Silero-VAD coverage audit + gap-fill feature was removed:
        # in practice the gap-fill rarely produced usable dialogue, and
        # when Whisper misses content the reliable fix is to re-run the
        # full transcription with the user's upload-page settings (see
        # POST /api/jobs/{id}/retranscribe).

        # Record phase timing for ETA estimation of remaining phases
        if _transcription_phase_start[0] > 0:
            _phase_timings["transcription"] = _time.monotonic() - _transcription_phase_start[0]

        await _update_branch_progress("transcription", 100, JobStatus.TRANSCRIBING,
            f"Transcribed {len(result)} segments \u2014 {diar_label}")
        return result

    # ── Branch B: Base64 encoding + AI scene analysis ──
    async def _branch_scene_analysis():
        cancel_check()

        # ── Phase 4 adaptive VLM frame sampling ──
        #
        # IMPORTANT: this block runs BEFORE base64 encoding so we never
        # spend CPU encoding frames the VLM will not see. Previously
        # the encode loop processed every uniform frame first; on long
        # 4K videos that wasted ~minute of CPU on dropped frames.
        #
        # After the uniform-stride extractor + face detect have run,
        # we have shot cuts and a face-conf timeline. Compute an
        # adaptive schedule and, if it is meaningfully smaller than
        # the uniform set, re-extract just those frames for VLM
        # analysis. The uniform set is kept for face/ASD — only the
        # VLM call uses the adaptive subset. ASD margin timeline and
        # clip-candidate starts are not available at this call site
        # (clip detection runs after scene analysis); the sampler
        # degrades gracefully when those signals are empty.
        vlm_frames = frames  # default: VLM analyzes the full uniform set
        if adaptive_sampling_enabled() and metadata["duration"] > 60:
            try:
                _src_face = dense_face_results or face_results
                face_conf_timeline = [
                    (fr.timestamp, max((f.confidence for f in fr.faces), default=0.0))
                    for fr in _src_face
                ]
                adaptive_inputs = AdaptiveSamplingInputs(
                    duration_seconds=float(metadata["duration"]),
                    shot_cuts=list(scene_cut_timestamps or []),
                    face_conf_timeline=face_conf_timeline,
                    asd_margin_timeline=[],
                    clip_candidate_starts=[],
                )
                adaptive_ts = compute_adaptive_frame_times(adaptive_inputs)
                # Only re-extract if the adaptive set is at least 25%
                # smaller — otherwise the I/O cost outweighs the VLM
                # savings.
                if adaptive_ts and len(adaptive_ts) <= int(0.75 * len(frames)):
                    adaptive_dir = os.path.join(
                        os.path.dirname(frames_dir), "frames_adaptive",
                    )
                    try:
                        vlm_frames = await extract_frames_at_timestamps(
                            video_path, adaptive_dir, adaptive_ts,
                            cancel_check=cancel_check,
                            video_codec=metadata.get("codec_name", ""),
                        )
                        logger.info(
                            "[%s] Adaptive VLM sampling: %d → %d frames (%.0f%% reduction)",
                            job_id, len(frames), len(vlm_frames),
                            100 * (1 - len(vlm_frames) / max(1, len(frames))),
                        )
                        # Attach face data to the adaptive frames
                        # (nearest-neighbor by timestamp).
                        if _src_face:
                            face_by_ts = sorted(
                                _src_face, key=lambda fr: fr.timestamp,
                            )
                            for vf in vlm_frames:
                                nearest = min(
                                    face_by_ts,
                                    key=lambda fr: abs(fr.timestamp - vf.timestamp),
                                )
                                if abs(nearest.timestamp - vf.timestamp) < 1.5:
                                    vf.face_data = nearest
                    except Exception as adapt_err:
                        logger.warning(
                            "[%s] Adaptive extraction failed (%s) — falling back to uniform set",
                            job_id, adapt_err,
                        )
                        vlm_frames = frames
                else:
                    logger.info(
                        "[%s] Adaptive sampling skipped: schedule %d vs uniform %d "
                        "(not >=25%% smaller)",
                        job_id, len(adaptive_ts) if adaptive_ts else 0, len(frames),
                    )
            except Exception as adapt_outer_err:
                logger.warning(
                    "[%s] Adaptive sampling preparation failed (%s) — using uniform set",
                    job_id, adapt_outer_err,
                )
                vlm_frames = frames

        # ── Phase 2: novelty-weighted VLM budget reallocation ──
        # Opt-in; reorders vlm_frames so semantically distinct frames are
        # analyzed first. If the provider hits rate limits mid-job or the
        # orchestrator aborts early, the most novel frames still get real
        # descriptions. Downstream re-sorts scenes_result by timestamp so
        # the UI timeline remains chronological.
        if settings.SCENE_NOVELTY_REWEIGHT_ENABLED and len(vlm_frames) > 20:
            try:
                from backend.services.frame_novelty import rank_frames_by_novelty
                _fc_timeline = [
                    (fr.timestamp, max((f.confidence for f in fr.faces), default=0.0))
                    for fr in (dense_face_results or face_results or [])
                ]
                ranked_idx = await asyncio.to_thread(
                    rank_frames_by_novelty,
                    vlm_frames,
                    scene_cuts=list(scene_cut_timestamps or []),
                    face_conf_timeline=_fc_timeline,
                    audio_energy_timeline=None,
                    distance_threshold=settings.SCENE_NOVELTY_DISTANCE_THRESHOLD,
                )
                vlm_frames = [vlm_frames[i] for i in ranked_idx]
                logger.info(
                    "[%s] Novelty reweight: reordered %d VLM frames (distinctive first)",
                    job_id, len(vlm_frames),
                )
            except Exception as e:
                logger.warning("[%s] Novelty reweight skipped: %s", job_id, e)

        # ── Encode ONLY the frames the VLM will actually see ──
        # On a 1h 4K clip with 50 % adaptive reduction this saves
        # ~1 minute of CPU vs encoding every uniform frame upfront.
        vlm_total = len(vlm_frames)
        await _update_branch_progress(
            "scene_analysis", 0, JobStatus.ANALYZING_SCENES,
            f"Preparing {vlm_total} frames for AI analysis...",
        )
        loop = asyncio.get_running_loop()
        _completed = 0
        _batch_size = min(16, max(1, vlm_total))

        for batch_start in range(0, vlm_total, _batch_size):
            cancel_check()
            batch_end = min(batch_start + _batch_size, vlm_total)
            batch = vlm_frames[batch_start:batch_end]

            def _encode(path):
                # Don't skip resize — 4K frames (3840x2160) overwhelm small
                # vision models like moondream:1.8b on 4GB GPUs. MAX_DIMENSION
                # (1568px) is plenty for scene understanding.
                return frame_to_base64(path, skip_resize=False)

            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*(
                        loop.run_in_executor(_b64_executor, _encode, fr.path)
                        for fr in batch
                    )),
                    timeout=_B64_ENCODE_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "[%s] Base64 encoding timed out for batch %d-%d",
                    job_id, batch_start, batch_end,
                )
                raise RuntimeError(
                    f"Frame encoding timed out — batch {batch_start}-{batch_end}"
                )
            for fr, b64 in zip(batch, results):
                fr.base64 = b64

            _completed = batch_end
            pct_done = int((_completed / max(vlm_total, 1)) * 100)
            branch_pct = pct_done * 0.1  # 0-10% of branch
            await _update_branch_progress(
                "scene_analysis", branch_pct, JobStatus.ANALYZING_SCENES,
                f"Preparing frames: {_completed}/{vlm_total} ({pct_done}%)",
            )

        # AI scene analysis
        await _update_branch_progress(
            "scene_analysis", 10, JobStatus.ANALYZING_SCENES,
            "Analyzing scenes with AI...",
        )

        _scene_start = _time.monotonic()
        _scene_phase_start[0] = _scene_start
        _last_eta_value[0] = 0.0  # Reset ETA smoothing for new phase

        async def _scene_progress(frames_done, frames_total, provider_name):
            pct = int((frames_done / max(frames_total, 1)) * 100)
            branch_pct = 10 + pct * 0.9  # 10-100% of branch
            await _update_branch_progress(
                "scene_analysis", branch_pct, JobStatus.ANALYZING_SCENES,
                f"Analyzing frame {frames_done}/{frames_total} via {provider_name} ({pct}%)",
            )

        # Phase 2 — content-type-aware vision routing. We pass the
        # user's explicit ``content_type_override`` (normalized) if
        # set; the OpenRouter provider uses that to swap in Qwen3-VL
        # for ANIME / GAMEPLAY and leave every other type on the
        # preset default. Jobs without an explicit override route
        # through the preset default unchanged.
        try:
            scenes_result, provider = await orchestrator.analyze_frames(
                vlm_frames, job_id, progress_callback=_scene_progress,
                content_type=_normalized_override or None,
            )
        except CancelledError:
            raise
        except Exception as e:
            error_str = str(e)
            if any(p in error_str.lower() for p in ["out of memory", "cudamalloc", "ggml_assert", "sigabrt"]):
                logger.warning(
                    "[%s] Scene analysis failed due to GPU memory — "
                    "continuing with empty scenes. Consider using smaller Ollama models "
                    "(e.g., moondream:1.8b for vision, qwen2.5:3b for text) or adding "
                    "more VRAM. Error: %s",
                    job_id, error_str[:200],
                )
                _record_pipeline_warning(
                    job_id,
                    (
                        f"warn: Scene analysis failed (GPU OOM): {error_str[:200]}. "
                        "Switch to a smaller vision model (moondream:1.8b) or "
                        "use a cloud provider (OpenRouter / Gemini) for vision."
                    ),
                )
                # Don't count OOM as a circuit breaker failure — it's a hardware limitation
                orchestrator.reset_circuit_breaker()
            else:
                logger.exception("[%s] Scene analysis failed, continuing with empty scenes", job_id)
                _record_pipeline_warning(
                    job_id,
                    (
                        f"warn: Scene analysis failed: {error_str[:300]}. "
                        "Key Scenes will be empty. Verify the configured vision "
                        "model in Settings \u2192 AI Provider."
                    ),
                )
            scenes_result = []
            provider = "none"

        # ── Phase 2: restore chronological order after novelty reweight ──
        # Upstream VLM ingestion may have been novelty-reordered, which
        # preserves distinctive-frame coverage under rate limits but would
        # leave the Key Scenes timeline shuffled. Re-sort BEFORE any
        # downstream consumer (face-registry check, slot snapper, clip
        # selector) touches the list.
        if scenes_result:
            scenes_result = sorted(scenes_result, key=lambda s: s.timestamp)

        # ── Face registry consistency check ──
        # After AI + face fusion produces subject_x values, validate every
        # value against the face registry and snap outliers to the nearest
        # known face position. For multi-speaker, this eliminates dead-zone
        # values between speakers. SKIPPED for continuous-motion content
        # (cartoons, sports) where snapping destroys real position data.
        # Fix 2: replace the coarse registry is_continuous_motion check
        # with a bimodality test on the actual subject_x distribution.
        # A seated 3-speaker panel has 2-4 histogram peaks even when the
        # raw values span 0..100; the old check flagged that as motion
        # and disabled slot snapping. If the bimodality test finds 1-4
        # clear peaks we run slot snapping — and use the peak centers as
        # the snap targets when face_registry is also fragmented.
        _registry_says_motion = face_registry.is_continuous_motion if face_registry else False
        _sxs = [s.subject_x for s in scenes_result] if scenes_result else []
        _looks_motion, _peak_centers = _subject_xs_look_like_motion(_sxs)
        # Final decision: if the bimodality test says NOT motion and we
        # have plausible peaks, override the registry and run snap. Otherwise
        # honor the registry's decision.
        _is_continuous = _looks_motion if _peak_centers else _registry_says_motion
        if _is_continuous:
            logger.info(
                "[%s] [SubjectTracking] Continuous motion detected — skipping slot snap (preserving raw positions)",
                job_id,
            )
        elif _registry_says_motion and not _looks_motion and _peak_centers:
            logger.info(
                "[%s] [SubjectTracking] registry=motion but subject_x bimodality "
                "override: %d peak(s) at %s — running slot snap",
                job_id, len(_peak_centers),
                [round(p, 1) for p in _peak_centers],
            )
        if face_registry and scenes_result and not _is_continuous and not _is_gameplay:
            # Prefer peak centers when the registry is empty OR fragmented
            # into more slots than the bimodality peaks; otherwise use the
            # registry slot centers. Peak centers are a reliable panel
            # signal when face_registry.slots got over-split.
            registry_centers = [s.x_center for s in face_registry.slots]
            slot_centers, _used_peaks = _choose_snap_centers(
                registry_centers, _peak_centers,
            )
            if _used_peaks:
                logger.info(
                    "[%s] [SubjectTracking] using %d bimodality peak(s) as snap "
                    "targets (registry had %d slots, overfragmented=%s)",
                    job_id, len(_peak_centers), len(registry_centers),
                    bool(registry_centers),
                )
            elif _peak_centers and len(registry_centers) > len(_peak_centers) + 1:
                logger.info(
                    "[%s] [SubjectTracking] keeping %d registry slot(s) over "
                    "%d bimodality peak(s) — registry not over-fragmented",
                    job_id, len(registry_centers), len(_peak_centers),
                )
            if not slot_centers:
                # Nothing to snap to; skip the correction pass entirely.
                slot_centers = []
            corrected = 0
            # For single-speaker, use a tighter threshold — any value far from
            # the one detected face is likely an AI error.
            snap_threshold = 15 if (face_registry.multi_speaker or len(slot_centers) >= 2) else 10
            for scene in scenes_result:
                if not slot_centers:
                    break
                sx = scene.subject_x
                min_dist_to_slot = min(abs(sx - sc) for sc in slot_centers)
                if min_dist_to_slot > snap_threshold:
                    nearest = int(round(min(slot_centers, key=lambda sc: abs(sc - sx))))
                    scene.subject_x = nearest
                    corrected += 1
            if corrected > 0:
                logger.info(
                    "[%s] Face consistency check: corrected %d/%d scenes (snapped to registry slots)",
                    job_id, corrected, len(scenes_result),
                )
            final_sxs = [s.subject_x for s in scenes_result]
            final_dist: dict[int, int] = {}
            for sx in final_sxs:
                final_dist[sx] = final_dist.get(sx, 0) + 1
            logger.info(
                "[SubjectTracking] Post-correction distribution: %s",
                dict(sorted(final_dist.items())),
            )

        # ── Center-default inheritance ──
        # When AI returned center (45-55) and no face data is available,
        # inherit subject_x from the nearest temporal neighbor with real data.
        if face_registry and face_registry.multi_speaker and scenes_result:
            center_lo, center_hi = 45, 55
            fixed_center = 0
            for i, scene in enumerate(scenes_result):
                if not (center_lo <= scene.subject_x <= center_hi):
                    continue
                best_sx = None
                best_dist = float('inf')
                for j in range(max(0, i - 5), min(len(scenes_result), i + 6)):
                    if j == i:
                        continue
                    other = scenes_result[j]
                    if center_lo <= other.subject_x <= center_hi:
                        continue
                    dist = abs(scene.timestamp - other.timestamp)
                    if dist < best_dist:
                        best_dist = dist
                        best_sx = other.subject_x
                if best_sx is not None:
                    scene.subject_x = int(best_sx)
                    fixed_center += 1
            if fixed_center > 0:
                logger.info(
                    "[%s] Center-default inheritance: fixed %d scenes with temporal neighbors",
                    job_id, fixed_center,
                )

        # Gameplay mode: override scene subject_x using the per-frame
        # crosshair path when available (Phase 1: FPS subtypes), or
        # fall back to the legacy hard-coded center (50) otherwise.
        #
        # Center-bias guard (Phase 1 hard-fallback spec): only
        # high-confidence crosshair detections
        # (``confidence >= GAMING_CROSSHAIR_CONF_THRESHOLD``, default
        # 0.6) override the scene center. Low-confidence entries —
        # including the hard-center breadcrumbs that
        # ``track_crosshair_path`` now emits for every missed frame —
        # resolve to exactly x=50. This kills the TF2 / Marvel
        # Rivals-style regression where a stale or noisy detection
        # would pull the crop off-axis on cartoon FPS content.
        #
        # Anime veto refresh: ``_early_anime_hint`` is set ONCE at
        # pipeline init (line ~1481), *before* the dense face pass
        # runs. The dense pass uses lbpcascade_animeface and can flip
        # ``face_detector.ANIME_MODE_DETECTED`` to True after seeing
        # >100 verified anime faces. Without re-reading that flag
        # here, an anime video with a generic filename
        # (``episode_01.mkv``, ``demon_slayer_S01E01.mkv``, etc.)
        # that triggered early pre-detect ``_is_gameplay=True`` on
        # sparse-face-ratio heuristics falls through the
        # ``not _early_anime_hint`` guard below and gets every
        # scene's subject_x slammed to 50 — producing a static
        # center crop on animated content. Also consults the
        # dense-face anime ratio as a secondary signal for short
        # clips where the dense pass didn't cross the 100-face
        # threshold but still saw majority-anime faces.
        _anime_veto = _early_anime_hint
        if not _anime_veto:
            try:
                # Re-import to get the freshest module-level flag —
                # the dense pass writes to it during execution.
                from backend.services.face_detector import (
                    ANIME_MODE_DETECTED as _amd_now,
                )
                if _amd_now:
                    _anime_veto = True
                    logger.info(
                        "[%s] Gameplay override veto: "
                        "ANIME_MODE_DETECTED flipped True after dense "
                        "pass — treating as animated, not gameplay",
                        job_id,
                    )
            except Exception:
                pass
        if not _anime_veto and dense_face_results:
            # Secondary signal for short clips (<100 faces): if the
            # dense pass tagged a majority of its faces as anime
            # (lbpcascade_animeface contributions), veto even if the
            # module-level threshold hasn't flipped yet. Each dense
            # face result may carry an ``is_anime`` boolean or an
            # ``anime_source`` tag depending on detector build — we
            # accept either.
            try:
                _total = 0
                _anime_n = 0
                for _fr in dense_face_results:
                    for _f in getattr(_fr, "faces", None) or []:
                        _total += 1
                        if (
                            getattr(_f, "is_anime", False)
                            or getattr(_f, "anime_source", "")
                            or getattr(_f, "source", "") == "anime"
                        ):
                            _anime_n += 1
                if _total >= 30 and _anime_n / _total >= 0.5:
                    _anime_veto = True
                    logger.info(
                        "[%s] Gameplay override veto: dense-face "
                        "anime ratio %d/%d (≥0.5) — treating as "
                        "animated, not gameplay",
                        job_id, _anime_n, _total,
                    )
            except Exception:
                pass
        if _is_gameplay and scenes_result and not _anime_veto:
            try:
                from backend.services.l1_camera_path import (
                    GAMING_CROSSHAIR_CONF_THRESHOLD,
                )
            except Exception:
                GAMING_CROSSHAIR_CONF_THRESHOLD = 0.6  # fallback literal
            ch_used = 0
            if crosshair_path:
                # Build a sorted list for nearest-time lookup.
                sorted_ch = sorted(
                    crosshair_path, key=lambda c: c.timestamp,
                )
                ch_times = [c.timestamp for c in sorted_ch]
                import bisect as _bisect
                for scene in scenes_result:
                    mid_t = (
                        (scene.start + scene.end) / 2.0
                        if hasattr(scene, "start")
                        else getattr(scene, "timestamp", 0.0)
                    )
                    idx = _bisect.bisect_left(ch_times, mid_t)
                    candidates = []
                    if idx < len(sorted_ch):
                        candidates.append(sorted_ch[idx])
                    if idx > 0:
                        candidates.append(sorted_ch[idx - 1])
                    if not candidates:
                        scene.subject_x = 50
                        continue
                    best = min(
                        candidates,
                        key=lambda c: abs(c.timestamp - mid_t),
                    )
                    if (
                        abs(best.timestamp - mid_t) <= 1.0
                        and getattr(best, "confidence", 0.0)
                        >= GAMING_CROSSHAIR_CONF_THRESHOLD
                    ):
                        scene.subject_x = int(round(best.x_pct))
                        ch_used += 1
                    else:
                        scene.subject_x = 50
            else:
                for scene in scenes_result:
                    scene.subject_x = 50
            logger.info(
                "[%s] Gameplay mode: set %d scene subject_x "
                "(crosshair-tracked=%d, fallback-center=%d)",
                job_id, len(scenes_result), ch_used,
                len(scenes_result) - ch_used,
            )

        # ── Phase 1.5: Final description sanitization ──
        # Guarantees no raw VLM junk (dict reprs, JSON leaks, refusals)
        # escapes to storage or the UI regardless of which provider path
        # produced the scene. Runs after every provider-side cleanup so
        # this is the last line of defense before persistence.
        from backend.services.scene_description_validator import (
            sanitize_description as _p15_sanitize,
            is_valid_description as _p15_is_valid,
            fallback_description_for_scene as _p15_fallback,
        )
        _cleaned = _replaced = 0
        for _i, _scene in enumerate(scenes_result):
            _original = _scene.description or ""
            _clean = _p15_sanitize(_original)
            _ok, _reason = _p15_is_valid(_clean)
            if _ok:
                if _clean != _original:
                    _scene.description = _clean
                    _cleaned += 1
            else:
                _scene.description = _p15_fallback(
                    timestamp=_scene.timestamp,
                    transcript_segments=None,
                    scene_index=_i + 1,
                )
                if hasattr(_scene, "description_source"):
                    _scene.description_source = "transcript_fallback"
                _replaced += 1
        if _cleaned or _replaced:
            logger.info(
                "[%s] Description sanitization: cleaned=%d, replaced=%d, total=%d",
                job_id, _cleaned, _replaced, len(scenes_result),
            )

        await database.update_job_status(
            job_id,
            scenes=list(scenes_result),
            provider_used={"scenes": provider},
        )

        # Count real vs synthetic scenes via the canonical filter in
        # ``pipeline_helpers.is_synthetic_scene`` so every code path
        # (warning UI, clip detector, render plan) agrees on what
        # counts as a real description.
        real_scenes = [s for s in scenes_result if not is_synthetic_scene(s)]
        fake_count = len(scenes_result) - len(real_scenes)

        # ── Heuristic key-scene fallback (scene-branch local) ───
        # If the VLM produced no usable scenes (empty, all-synthetic,
        # provider crashed, etc.) but the pipeline DOES have scene
        # cuts, faces, or even just a duration, synthesize key moments
        # so the Key Scenes tab is never blank when the pipeline ran
        # to completion. Transcript-aware enrichment happens AFTER the
        # transcription branch finishes (see ``_enrich_scenes_with_transcript``
        # below the gather). Real VLM scenes always win when they exist;
        # this only activates when ``real_scenes`` is empty.
        if len(real_scenes) == 0:
            try:
                from backend.services.scene_fallback import synthesize_key_scenes
                synth = synthesize_key_scenes(
                    duration=float(metadata.get("duration") or 0),
                    scene_cuts=list(scene_cut_timestamps or []),
                    transcript=None,    # filled in after the gather
                    face_results=list(face_results or []),
                    frames=list(frames or []),
                    target_count=12,
                )
                if synth:
                    logger.info(
                        "[%s] Heuristic key-scene fallback: synthesized %d "
                        "scenes from cuts/faces (VLM produced 0 real)",
                        job_id, len(synth),
                    )
                    _record_pipeline_warning(
                        job_id,
                        (
                            f"info: Synthesized {len(synth)} heuristic key scenes "
                            "from scene cuts + face tracks because the vision "
                            "provider returned no real descriptions. Switch "
                            "Settings \u2192 AI Provider \u2192 Vision to a "
                            "working model for AI-generated descriptions."
                        ),
                    )
                    # Replace the empty/synthetic list with the
                    # synthesized one so downstream consumers (clip
                    # detector, Key Scenes tab) have something to use.
                    scenes_result = synth
                    real_scenes = synth
                    fake_count = 0
                    if provider == "none" or "(all synthetic" in provider:
                        provider = "heuristic-fallback"
            except Exception as fallback_err:
                logger.warning(
                    "[%s] Heuristic key-scene fallback failed: %s",
                    job_id, fallback_err,
                )

        if fake_count > 0 and len(real_scenes) == 0:
            provider = f"{provider} (all synthetic — vision failed)"
            logger.warning(
                "[%s] Scene analysis produced 0 real descriptions — all %d are synthetic. "
                "Check Ollama logs for CLIP/vision model errors.",
                job_id, fake_count,
            )
            # Surface to the Analysis page so the user understands WHY
            # the Key Scenes tab is empty instead of silently shipping
            # synthetic placeholders. ``provider`` here may be "none"
            # (analyze_frames raised), or a real model name with all
            # batches having returned empty/fallback responses.
            _record_pipeline_warning(
                job_id,
                (
                    f"warn: Scene analysis returned 0 real descriptions "
                    f"({fake_count} fallback) via {provider!r}. Common causes: "
                    "vision model overloaded / OOM, OpenRouter API key out of "
                    "credit / wrong model id, or the configured vision model "
                    "doesn't support image input. Check Settings \u2192 AI Provider \u2192 "
                    "Vision and run Settings \u2192 Advanced \u2192 Pipeline Diagnostics."
                ),
            )
        elif len(scenes_result) == 0 and provider != "none":
            # Provider returned the empty list cleanly (no exception, no
            # synthetic placeholders) — just nothing. Without this branch
            # the user got an empty Key Scenes tab with no explanation.
            logger.warning(
                "[%s] Scene analysis returned an empty list via %r without errors.",
                job_id, provider,
            )
            _record_pipeline_warning(
                job_id,
                (
                    f"warn: Scene analysis returned 0 scenes via {provider!r} "
                    "with no error reported. The vision model accepted the "
                    "request but produced no output — this usually means "
                    "the configured model id is wrong, the model is text-only, "
                    "or the prompt was rejected. Verify Settings \u2192 AI Provider \u2192 Vision."
                ),
            )
        elif fake_count > 0:
            logger.info(
                "[%s] Scene analysis: %d real + %d synthetic descriptions",
                job_id, len(real_scenes), fake_count,
            )
            _record_pipeline_warning(
                job_id,
                (
                    f"info: Scene analysis: {len(real_scenes)} real + "
                    f"{fake_count} fallback (vision model returned empty for "
                    f"{fake_count} batches)."
                ),
            )

        # Record phase timing for ETA estimation of remaining phases
        if _scene_phase_start[0] > 0:
            _phase_timings["scene_analysis"] = _time.monotonic() - _scene_phase_start[0]

        await _update_branch_progress("scene_analysis", 100, JobStatus.ANALYZING_SCENES,
            f"Analyzed {len(real_scenes)} scenes via {provider}"
            + (f" ({fake_count} skipped)" if fake_count > 0 else ""))

        # Log subject tracking status
        if settings.SUBJECT_TRACKING_ENABLED:
            tracked = [s for s in scenes_result if hasattr(s, 'subject_x') and s.subject_x is not None]
            sx_values = [s.subject_x for s in tracked] if tracked else []
            logger.info(
                "[SubjectTracking] ═══ SCENE ANALYSIS COMPLETE for job %s ═══",
                job_id,
            )
            logger.info(
                "[SubjectTracking] %d/%d scenes tracked, subject_x values: %s",
                len(tracked), len(scenes_result),
                sx_values if sx_values else "(none)",
            )
            if sx_values:
                logger.info(
                    "[SubjectTracking] subject_x stats: min=%d, max=%d, mean=%.1f, "
                    "unique=%d, distribution=%s",
                    min(sx_values), max(sx_values),
                    sum(sx_values) / len(sx_values),
                    len(set(sx_values)),
                    {v: sx_values.count(v) for v in sorted(set(sx_values))},
                )
            await broadcast_ws(job_id, {
                "type": "subject_tracking",
                "enabled": True,
                "tracked_scenes": len(tracked),
                "total_scenes": len(scenes_result),
                "message": f"Subject tracking: {len(tracked)}/{len(scenes_result)} scenes tracked for dynamic crop positioning",
            })

            # Detect poor tracking quality from local AI and notify user
            if tracked and sx_values:
                at_center = sum(1 for sx in sx_values if sx == 50)
                center_pct = at_center / len(sx_values) * 100 if sx_values else 0
                if center_pct > 80 and len(sx_values) > 5:
                    logger.warning(
                        "[SubjectTracking] Poor quality for job %s: %d/%d frames (%.0f%%) at center. Provider: %s",
                        job_id, at_center, len(sx_values), center_pct, provider,
                    )
                    await broadcast_ws(job_id, {
                        "type": "subject_tracking_quality",
                        "quality": "low",
                        "center_pct": round(center_pct),
                        "total_frames": len(sx_values),
                        "message": (
                            f"Subject tracking quality is limited — {round(center_pct)}% of frames defaulted to center. "
                            f"Using {provider}. Cloud AI providers typically produce more accurate tracking."
                        ),
                    })
        else:
            logger.info(
                "[SubjectTracking] DISABLED for job %s — all crops will use center of frame",
                job_id,
            )
            await broadcast_ws(job_id, {
                "type": "subject_tracking",
                "enabled": False,
                "message": "Subject tracking disabled — crops will use center of frame",
            })

        return scenes_result, provider

    # When Ollama (local GPU inference) is the provider, run transcription
    # FIRST so Whisper gets exclusive GPU access, then run scene analysis.
    # On a 4GB GPU, concurrent execution pushes both to CPU (~3x slower).
    # With cloud providers, run concurrently since there's no VRAM contention.
    # Sequential mode is ONLY needed when Ollama handles vision AND Whisper
    # both compete for the same local GPU. When cloud providers (OpenRouter,
    # Gemini, Groq) handle vision, there's no GPU contention — run concurrently.
    _uses_local_gpu = is_ollama_primary
    # _trans_scene_timeout was already computed in the adaptive timeout block above

    if _uses_local_gpu:
        logger.info("[%s] Sequential mode: transcription first, then scene analysis (local GPU)", job_id)

        # Safety: ALWAYS unload Ollama models before Whisper, regardless of
        # which AI provider is primary. The Ollama sidecar container shares the
        # GPU and may have models loaded from startup pulls, diagnostic checks,
        # or previous pipeline runs. On a 4GB GPU, there's no room for both.
        try:
            import httpx as _httpx
            _ollama_host = settings.OLLAMA_HOST
            logger.info("[%s] Pre-transcription: ensuring ALL Ollama models are unloaded...", job_id)
            async with _httpx.AsyncClient(timeout=15) as _uc:
                _ps = await _uc.get(f"{_ollama_host}/api/ps")
                if _ps.status_code == 200:
                    for _m in _ps.json().get("models", []):
                        _mn = _m.get("name", "")
                        if _mn:
                            await _uc.post(f"{_ollama_host}/api/generate",
                                json={"model": _mn, "keep_alive": 0}, timeout=10)
                            logger.info("[%s] Unloaded Ollama model '%s' to free GPU for Whisper", job_id, _mn)
            await asyncio.sleep(2)  # Let CUDA driver release memory
        except Exception as _unload_err:
            logger.warning("[%s] Pre-transcription: Ollama unload failed (%s) — proceeding anyway", job_id, _unload_err)

        # Log VRAM state so we can verify GPU is actually free
        try:
            import torch
            if torch.cuda.is_available():
                free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
                total_mb = torch.cuda.mem_get_info()[1] / (1024 * 1024)
                logger.info(
                    "[%s] VRAM before transcription: %.0fMB free / %.0fMB total",
                    job_id, free_mb, total_mb,
                )
        except Exception:
            pass

        async with _stage_timer(job_id, "transcription+scene_analysis"):
            try:
                trans_result = await asyncio.wait_for(
                    _branch_transcription(),
                    timeout=_trans_scene_timeout,
                )
            except asyncio.TimeoutError:
                logger.error("[%s] Transcription timed out", job_id)
                raise RuntimeError("Transcription timed out")
            except Exception as e:
                logger.exception("[%s] Transcription branch failed", job_id)
                trans_result = e

            # Pre-compute transcript-only hot zones for frame triage
            # (runs before scene analysis so Ollama can skip cold-zone frames)
            if not isinstance(trans_result, BaseException) and trans_result:
                try:
                    from backend.services.hot_zone_scorer import score_hot_zones_transcript_only
                    import backend.services.providers.ollama_provider as _ollama_mod
                    pre_hot_zones = score_hot_zones_transcript_only(trans_result, metadata["duration"])
                    _ollama_mod._current_hot_zones = pre_hot_zones
                    logger.info("[%s] Pre-computed %d transcript-only hot zones for frame triage", job_id, len(pre_hot_zones))
                except Exception as e:
                    logger.warning("[%s] Hot zone pre-scoring failed (non-fatal): %s", job_id, e)

            # Release Whisper VRAM before Ollama loads its models
            if _subprocess_whisper_used[0]:
                logger.info("[%s] Subprocess Whisper — CUDA memory already released, skipping torch cleanup", job_id)
            else:
                await _release_whisper_vram(job_id)

            # ── Verified VRAM recovery ──
            # Use PyTorch's CUDA reporting (works inside Docker without nvidia-smi).
            # Poll until VRAM is free or max wait exceeded.
            _vram_target = 3000  # Need 3GB free for moondream CLIP + LLM
            _vram_wait_max = 15  # Max seconds
            _vram_poll_interval = 3
            try:
                import torch
                if torch.cuda.is_available():
                    for _attempt in range(int(_vram_wait_max / _vram_poll_interval) + 1):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        import gc
                        gc.collect()
                        free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
                        logger.info(
                            "[%s] VRAM recovery check %d: %.0fMB free (target: %dMB)",
                            job_id, _attempt + 1, free_mb, _vram_target,
                        )
                        if free_mb >= _vram_target:
                            break
                        if _attempt < int(_vram_wait_max / _vram_poll_interval):
                            await asyncio.sleep(_vram_poll_interval)
                    else:
                        logger.warning(
                            "[%s] VRAM did not fully recover after %ds (%.0fMB free) — "
                            "moondream CLIP may fall back to CPU",
                            job_id, _vram_wait_max, free_mb,
                        )
            except Exception as e:
                logger.debug("[%s] PyTorch VRAM check unavailable: %s", job_id, e)

            _log_gpu_memory(job_id, "after VRAM recovery")

            # Final torch release — ensure CUDA context isn't hogging VRAM
            release_torch_gpu_memory()

            # Wait for CUDA driver to fully reclaim VRAM across Docker containers.
            # CTranslate2's subprocess exit releases memory, but the NVIDIA driver
            # needs 10-15 seconds to actually free it on GTX 1650.
            logger.info("[%s] Waiting 15s for CUDA driver to reclaim Whisper VRAM...", job_id)
            await asyncio.sleep(15)

            # Trigger Ollama GPU re-discovery with retry
            # First attempt may fail if VRAM isn't fully released yet
            if is_ollama_primary and _primary_provider:
                gpu_ok = False
                for _rediscovery_attempt in range(3):
                    try:
                        gpu_ok = await _trigger_ollama_gpu_rediscovery(job_id, _primary_provider)
                        if gpu_ok:
                            logger.info("[%s] Ollama GPU access confirmed (attempt %d)", job_id, _rediscovery_attempt + 1)
                            # DON'T clear the model — leave it loaded on GPU for scene analysis
                            break
                        else:
                            logger.warning("[%s] GPU rediscovery attempt %d: CLIP still on CPU, waiting 10s...",
                                           job_id, _rediscovery_attempt + 1)
                            # Clear CPU-loaded model and wait before retry
                            if hasattr(_primary_provider, 'clear_vram'):
                                await _primary_provider.clear_vram()
                            await asyncio.sleep(10)
                    except Exception as e:
                        logger.warning("[%s] GPU rediscovery attempt %d failed: %s", job_id, _rediscovery_attempt + 1, e)
                        await asyncio.sleep(5)

                if not gpu_ok:
                    logger.warning("[%s] All GPU rediscovery attempts failed — scene analysis will use CPU (slow)", job_id)
                    if hasattr(_primary_provider, '_force_cpu'):
                        _primary_provider._force_cpu = False  # Still allow GPU attempt during analysis

            await _update_progress(
                job_id, JobStatus.ANALYZING_SCENES, 40,
                "Released transcription GPU memory — preparing scene analysis...",
            )

            try:
                scene_result = await asyncio.wait_for(
                    _branch_scene_analysis(),
                    timeout=_trans_scene_timeout,
                )
            except asyncio.TimeoutError:
                logger.error("[%s] Scene analysis timed out", job_id)
                raise RuntimeError("Scene analysis timed out")
            except Exception as e:
                logger.exception("[%s] Scene analysis branch failed", job_id)
                scene_result = e
    else:
        logger.info("[%s] Concurrent mode: transcription + scene analysis (cloud providers)", job_id)
        async with _stage_timer(job_id, "transcription+scene_analysis"):
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(
                        _branch_transcription(),
                        _branch_scene_analysis(),
                        return_exceptions=True,
                    ),
                    timeout=_trans_scene_timeout,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "[%s] Transcription+scene analysis timed out after %.0fs",
                    job_id, _trans_scene_timeout,
                )
                raise RuntimeError(
                    f"Transcription and scene analysis timed out after "
                    f"{int(_trans_scene_timeout // 60)} minutes."
                )
            trans_result, scene_result = results

    if isinstance(trans_result, BaseException):
        if isinstance(trans_result, CancelledError):
            raise trans_result
        logger.exception("[%s] Transcription branch failed", job_id, exc_info=trans_result)
        transcript = []
    else:
        transcript = trans_result

    if isinstance(scene_result, BaseException):
        if isinstance(scene_result, CancelledError):
            raise scene_result
        logger.exception("[%s] Scene analysis branch failed", job_id, exc_info=scene_result)
        scenes, scenes_provider = [], "none"
    else:
        scenes, scenes_provider = scene_result

    speaker_count = len(set(s.speaker for s in transcript))
    logger.info(
        "[%s] Branches complete: %d transcript segments (%d speakers), %d scenes via %s",
        job_id, len(transcript), speaker_count, len(scenes), scenes_provider,
    )

    # ── Transcript-aware enrichment of heuristic key-scenes ──────
    # If the scene branch fell back to the heuristic synthesizer
    # (which runs BEFORE transcription completes and therefore can't
    # quote any spoken text), re-run it now with the transcript so
    # the Key Scenes tab shows what was being said at each moment.
    if scenes_provider == "heuristic-fallback" and transcript:
        try:
            from backend.services.scene_fallback import synthesize_key_scenes
            enriched = synthesize_key_scenes(
                duration=float(metadata.get("duration") or 0),
                scene_cuts=list(scene_cut_timestamps or []),
                transcript=list(transcript or []),
                face_results=list(face_results or []),
                frames=list(frames or []),
                target_count=12,
            )
            if enriched:
                scenes = enriched
                logger.info(
                    "[%s] Heuristic key-scene fallback: enriched %d scenes "
                    "with transcript excerpts after transcription completed",
                    job_id, len(enriched),
                )
        except Exception as enrich_err:
            logger.warning(
                "[%s] Heuristic key-scene enrichment failed (non-fatal): %s",
                job_id, enrich_err,
            )

    # ── Phase 1.5 transcript upgrade pass ────────────────────────
    # Upgrade any "Key moment N at X.Xs" timestamp-only fallbacks (emitted
    # by the scene-branch sanitization pass BEFORE transcription finished)
    # with transcript excerpts now that transcript_segments is available.
    import re as _p15_re
    from backend.services.scene_description_validator import (
        fallback_description_for_scene as _p15_fb,
    )
    _p15_ts_only = _p15_re.compile(r"^Key moment \d+ at [\d.]+s$")
    _p15_upgraded = 0
    for _i, _scene in enumerate(scenes or []):
        if _p15_ts_only.match(_scene.description or ""):
            _up = _p15_fb(
                timestamp=_scene.timestamp,
                transcript_segments=list(transcript or []) or None,
                scene_index=_i + 1,
            )
            if _up != _scene.description:
                _scene.description = _up
                _p15_upgraded += 1
    if _p15_upgraded:
        logger.info(
            "[%s] Transcript enrichment: upgraded %d timestamp-only fallbacks",
            job_id, _p15_upgraded,
        )

    # ── Last-resort fallback: scenes still empty after the join ──
    # Covers the case where ``_branch_scene_analysis`` raised mid-flight
    # (CancelledError unblocked, the in-branch fallback didn't run).
    # If we have ANY signal (duration, transcript, scene cuts) we
    # synthesize a minimal Key Scenes list rather than show an empty tab.
    if not scenes and (transcript or scene_cut_timestamps or float(metadata.get("duration") or 0) > 0):
        try:
            from backend.services.scene_fallback import synthesize_key_scenes
            scenes = synthesize_key_scenes(
                duration=float(metadata.get("duration") or 0),
                scene_cuts=list(scene_cut_timestamps or []),
                transcript=list(transcript or []),
                face_results=list(face_results or []),
                frames=list(frames or []),
                target_count=12,
            )
            if scenes:
                scenes_provider = "heuristic-fallback"
                _record_pipeline_warning(
                    job_id,
                    (
                        f"info: Key Scenes tab populated by heuristic fallback "
                        f"({len(scenes)} moments) because the vision branch "
                        "produced no scenes. Configure a working vision model in "
                        "Settings \u2192 AI Provider for AI-generated descriptions."
                    ),
                )
                logger.info(
                    "[%s] Last-resort heuristic fallback synthesized %d scenes",
                    job_id, len(scenes),
                )
        except Exception as last_err:
            logger.warning(
                "[%s] Last-resort key-scene fallback failed: %s",
                job_id, last_err,
            )

    # ── Screen content detection (CPU, ~5ms/frame) ──
    if settings.SUBJECT_TRACKING_ENABLED and scenes:
        try:
            from backend.services.screen_detector import detect_screen_content_batch
            frame_list = [(f.timestamp, f.path) for f in frames]
            screen_map = detect_screen_content_batch(frame_list)

            # Video-level gate: only apply if a significant portion of frames
            # are screen content. A single positive frame is noise.
            total_positive = sum(1 for v in screen_map.values() if v)
            screen_ratio = total_positive / max(len(screen_map), 1)

            screen_count = 0
            if screen_ratio >= 0.30:  # At least 30% of frames must be screenshare
                screen_positive_times = {
                    round(ts, 2) for ts, is_sc in screen_map.items() if is_sc
                }
                for scene in scenes:
                    if round(scene.timestamp, 2) in screen_positive_times:
                        scene.has_screen_content = True
                        screen_count += 1
                if screen_count > 0:
                    logger.info(
                        "[%s] Screen content detected in %d/%d scenes (%.0f%% of frames positive)",
                        job_id, screen_count, len(scenes), screen_ratio * 100,
                    )
            else:
                if total_positive > 0:
                    logger.info(
                        "[%s] Screen detection: %d/%d frames positive (%.0f%%) — "
                        "below 30%% threshold, ignoring (likely text overlays)",
                        job_id, total_positive, len(screen_map), screen_ratio * 100,
                    )
        except Exception as e:
            logger.warning("[%s] Screen detection failed (non-fatal): %s", job_id, e)

    # ── Object tracking for faceless frames (CPU, ~15ms/frame) ──
    _saliency_keyframes = []  # populated for AutoFlip reframe path
    if settings.SUBJECT_TRACKING_ENABLED and face_results and scenes:
        try:
            from backend.services.object_tracker import track_objects_in_frames
            frame_list = [(f.timestamp, f.path) for f in frames]
            object_kf = track_objects_in_frames(frame_list, face_results)

            # Also collect confidence-bearing tuples for AutoFlip reframe path
            USE_AUTOFLIP_REFRAME = os.environ.get("USE_AUTOFLIP_REFRAME", "false").lower() in ("true", "1", "yes")
            if USE_AUTOFLIP_REFRAME:
                _saliency_keyframes = track_objects_in_frames(
                    frame_list, face_results, return_confidence=True,
                )
                if _saliency_keyframes:
                    logger.info(
                        "[%s] Saliency keyframes for AutoFlip: %d tuples",
                        job_id, len(_saliency_keyframes),
                    )

            if object_kf:
                logger.info("[%s] Object tracker: %d keyframes from faceless frames", job_id, len(object_kf))
                obj_map = {round(t, 1): sx for t, sx in object_kf}
                for scene in scenes:
                    if scene.subject_x == 50 and not scene.face_positions:
                        for obj_t, obj_sx in obj_map.items():
                            if abs(obj_t - scene.timestamp) < 2.0:
                                scene.primary_object_x = obj_sx
                                scene.primary_object_type = "saliency"
                                scene.subject_x = int(obj_sx)
                                break
        except Exception as e:
            logger.warning("[%s] Object tracking failed (non-fatal): %s", job_id, e)

    # ── Saliency + Object Detection (universal) ──
    # Run saliency for all content types (not just AutoFlip path). This is the
    # AutoFlip-equivalent foundation: the single signal that matters most when
    # no face is present. Content-type weighting in required_regions controls
    # how much influence it has on the solver.
    _saliency_regions = []
    _object_detections = []
    # Early init for _persistent_regions — it is properly populated later
    # in the content classification block (~line 3320+) but the saliency
    # tracker below references it. Without this early init the call
    # raises "local variable '_persistent_regions' referenced before
    # assignment" — see Problem 4 in the anime reframing issue.
    try:
        _persistent_regions  # noqa: F841 — test if already bound
    except NameError:
        _persistent_regions = None
    OBJECT_DETECTION_ENABLED = os.environ.get("OBJECT_DETECTION_ENABLED", "true").lower() in ("true", "1", "yes")
    USE_AUTOFLIP_REFRAME = os.environ.get("USE_AUTOFLIP_REFRAME", "false").lower() in ("true", "1", "yes")
    if OBJECT_DETECTION_ENABLED:
        try:
            from backend.services.saliency_tracker import track_saliency_in_frames
            frame_list_sal = [(f.timestamp, f.path) for f in frames]
            _saliency_regions = track_saliency_in_frames(frame_list_sal, face_results,
                                                             persistent_regions=_persistent_regions)
            logger.info("[%s] SaliencyTracker: %d regions (universal)", job_id, len(_saliency_regions))
        except Exception as e:
            logger.warning("[%s] Saliency tracker failed (non-fatal): %s", job_id, e)

        try:
            from backend.services.object_detector import detect_objects_in_frames, get_detector
            frame_list_det = [(f.timestamp, f.path) for f in frames]
            _object_detections = detect_objects_in_frames(frame_list_det, face_results)
            logger.info("[%s] ObjectDetector: %d detections (backend=%s)",
                        job_id, len(_object_detections), get_detector().backend_name)
        except Exception as e:
            logger.warning("[%s] Object detection failed (non-fatal): %s", job_id, e)

        # ── Subject-x rescue: when the VLM produced uniform center
        # values (because the configured vision model has poor spatial
        # reasoning OR the entire batch was discarded) fall back to
        # the saliency tracker → object detector → face detector
        # cascade so the reframer doesn't lock the camera dead-center
        # for the entire video. This is the difference between an
        # anime episode being croppable to 9:16 and being a static
        # center-cropped frame for 22 minutes.
        try:
            if scenes and not _is_gameplay:
                # PER-SCENE rescue. The old gate only triggered when
                # *every* scene was ~50 (90%+ at center, ≤2 distinct
                # values, ≥5 scenes). That missed the much more common
                # failure mode shown on the user's Attack on Titan
                # reframe: a character whose face clearly sits at x≈70
                # in the 16:9 frame, but the VLM tagged *that* single
                # scene with subject_x=50 while other scenes were fine.
                # Result: the 9:16 window snapped to center and the
                # subject was off-screen for half a second.
                #
                # Now we inspect every scene independently — if the VLM
                # gave us a center default AND we have a concrete face /
                # saliency / YOLO hit within the lookup window, we
                # override it. Scenes the VLM got right are untouched.
                if True:
                    rescued = 0
                    # Build per-frame nearest-neighbor lookups so we
                    # can answer "what's at time t" in O(log n).
                    _sal_sorted = sorted(
                        _saliency_regions or [],
                        key=lambda r: getattr(r, "timestamp", 0.0),
                    )
                    _obj_sorted = sorted(
                        _object_detections or [],
                        key=lambda d: getattr(d, "timestamp", 0.0),
                    )
                    _face_sorted = sorted(
                        face_results or [],
                        key=lambda f: getattr(f, "timestamp", 0.0),
                    )

                    def _nearest_x(arr, t, attr_x_pct, max_delta=2.5):
                        if not arr:
                            return None
                        best = None
                        best_dist = float("inf")
                        for item in arr:
                            it = getattr(item, "timestamp", 0.0)
                            d = abs(it - t)
                            if d < best_dist:
                                best_dist = d
                                best = item
                        if best is None or best_dist > max_delta:
                            return None
                        x = attr_x_pct(best)
                        if x is None:
                            return None
                        try:
                            return max(0, min(100, int(round(float(x)))))
                        except (TypeError, ValueError):
                            return None

                    def _saliency_x(r):
                        # Saliency regions expose ``x_center_pct`` (0-100)
                        # via the AutoFlip parity adapter.
                        x = getattr(r, "x_center_pct", None)
                        if x is None:
                            x = getattr(r, "x_center", None)
                        return x

                    def _object_x(d):
                        # ObjectDetector entries have ``x_center_pct``
                        # via SubjectPosition.
                        return getattr(d, "x_center_pct", None) or getattr(d, "primary_object_x", None)

                    def _face_x(f):
                        faces = getattr(f, "faces", None) or []
                        if not faces:
                            return None
                        primary = getattr(f, "primary_face_idx", -1)
                        if 0 <= primary < len(faces):
                            face = faces[primary]
                        else:
                            face = faces[0]
                        return getattr(face, "x_center", None) or getattr(face, "nose_x", None)

                    for s in scenes:
                        # Only override VLM scenes that look suspicious —
                        # i.e. the VLM punted and gave us a ~center default.
                        # Trusted non-center VLM positions are preserved
                        # (they already reflect real spatial reasoning).
                        cur_sx = getattr(s, "subject_x", 50)
                        if not (47 <= cur_sx <= 53):
                            continue
                        t = getattr(s, "timestamp", 0.0)
                        # Try tight windows first (prefer evidence
                        # close to the scene timestamp), then widen
                        # progressively. Long-form content (22-minute
                        # anime episodes, 45-minute podcasts, sports
                        # highlights) has sparse face detection
                        # keyframes — a 1.5s window missed most scenes
                        # and left the whole video glued center. The
                        # final 12s window catches anything within the
                        # surrounding shot, which is still a massive
                        # improvement over 50.
                        x = (
                            _nearest_x(_face_sorted, t, _face_x, max_delta=1.5)
                            or _nearest_x(_sal_sorted, t, _saliency_x, max_delta=2.5)
                            or _nearest_x(_obj_sorted, t, _object_x, max_delta=2.5)
                            or _nearest_x(_face_sorted, t, _face_x, max_delta=6.0)
                            or _nearest_x(_sal_sorted, t, _saliency_x, max_delta=8.0)
                            or _nearest_x(_obj_sorted, t, _object_x, max_delta=8.0)
                            or _nearest_x(_face_sorted, t, _face_x, max_delta=12.0)
                        )
                        if x is not None and not (48 <= x <= 52):
                            s.subject_x = x
                            try:
                                s.fusion_source = "saliency_object_face_rescue"
                            except Exception:
                                pass
                            rescued += 1
                    if rescued > 0:
                        logger.info(
                            "[%s] Subject-x rescue: %d/%d scenes re-positioned "
                            "from saliency/object/face after VLM returned center defaults",
                            job_id, rescued, len(scenes),
                        )
                        _record_pipeline_warning(
                            job_id,
                            (
                                f"info: Subject tracking rescue: re-derived "
                                f"{rescued}/{len(scenes)} subject positions from "
                                "saliency + object detection + face data because "
                                "the vision model returned center defaults. The "
                                "9:16 / 1:1 reframe will follow the action instead "
                                "of staying glued to the center of the frame."
                            ),
                        )

                    # Temporal carry-fill. Any scene still locked at
                    # ~50 after the cascade inherits subject_x from
                    # the nearest rescued neighbor in time. Without
                    # this step, scenes that happen to sit in a shot
                    # with no face/saliency/object hit stay glued
                    # center while the surrounding shots follow the
                    # action — producing visible "snap to middle"
                    # glitches in the 9:16 reframe. Works for every
                    # content genre because we only copy from scenes
                    # that were already rescued from real evidence.
                    _rescued_points = [
                        (getattr(s, "timestamp", 0.0), int(getattr(s, "subject_x", 50)))
                        for s in scenes
                        if not (47 <= int(getattr(s, "subject_x", 50)) <= 53)
                    ]
                    if _rescued_points:
                        _rescued_points.sort(key=lambda p: p[0])
                        filled = 0
                        for s in scenes:
                            cur_sx = int(getattr(s, "subject_x", 50))
                            if not (47 <= cur_sx <= 53):
                                continue
                            t = float(getattr(s, "timestamp", 0.0))
                            nearest = min(_rescued_points, key=lambda p: abs(p[0] - t))
                            if abs(nearest[0] - t) <= 30.0:
                                s.subject_x = int(nearest[1])
                                try:
                                    s.fusion_source = "temporal_carry_fill"
                                except Exception:
                                    pass
                                filled += 1
                        if filled > 0:
                            logger.info(
                                "[%s] Subject-x carry-fill: %d additional scenes "
                                "inherited nearest rescued subject_x (preventing "
                                "center snap on shots with no direct evidence)",
                                job_id, filled,
                            )
        except Exception as _rescue_err:
            logger.debug(
                "[%s] Subject-x rescue failed (non-fatal): %s",
                job_id, _rescue_err,
            )

    # Save original AI vision subject_x BEFORE dense face or lip-audio overwrites.
    # These originals are the AI model's spatial reasoning — not lip detection noise.
    _original_scene_sx = [(s.timestamp, s.subject_x) for s in scenes] if scenes else []

    # ── Merge dense face data into scene descriptions ──
    # Override AI vision model's subject_x with actual face positions from
    # dense detection. Dense data is pixel-accurate; AI estimates are guesses.
    # Classify tracking mode BEFORE any merge/synthesis that references it.
    # SKIP for gameplay content — crosshair is always at center, face data is noise.
    _is_continuous = face_registry.is_continuous_motion if face_registry else False
    if dense_face_results and scenes and not _is_gameplay:
        dense_map = {}
        for dfr in dense_face_results:
            dense_map[round(dfr.timestamp, 2)] = dfr
        enriched = 0
        for scene in scenes:
            best_dfr = None
            best_dist = float('inf')
            for dt, dfr in dense_map.items():
                dist = abs(dt - scene.timestamp)
                if dist < best_dist and dist < 1.0:
                    best_dist = dist
                    best_dfr = dfr
            if best_dfr and best_dfr.faces:
                scene.face_count = int(len(best_dfr.faces))
                scene.face_positions = [
                    {
                        "slot_id": int(f.identity_id),
                        "x": int(round(f.nose_x)),
                        "y": int(round(f.nose_y)),
                        "w": int(round(f.width)),
                        "h": int(round(f.height)),
                        "is_speaking": bool(f.is_speaking),
                        "identity_id": int(f.identity_id),
                    }
                    for f in best_dfr.faces
                ]
                # Determine best face for subject_x:
                # 1. Single large face (close-up) → use raw nose_x for precision
                # 2. Multi-speaker with active speaker → use slot center for stability
                # 3. Fallback → largest face with slot center snapping
                is_closeup = (len(best_dfr.faces) == 1 and best_dfr.faces[0].width > 12.0)

                if fusion_enabled():
                    # Phase 3 — confidence-weighted fusion. Pick the
                    # best face candidate (same selection as legacy)
                    # but blend with the VLM signal rather than
                    # hard-overwriting. The legacy branches still
                    # determine WHICH face we trust; fusion just
                    # decides HOW MUCH.
                    best_face = None
                    if _is_continuous or is_closeup:
                        best_face = _select_dominant_face(best_dfr.faces)
                    elif face_registry and face_registry.multi_speaker:
                        speaking = [f for f in best_dfr.faces if f.lip_aperture > 0.03]
                        if speaking:
                            best_face = max(speaking, key=lambda f: f.lip_aperture)
                        elif best_dfr.primary_face_idx >= 0:
                            best_face = best_dfr.faces[best_dfr.primary_face_idx]
                    elif best_dfr.primary_face_idx >= 0:
                        best_face = best_dfr.faces[best_dfr.primary_face_idx]

                    if best_face is not None:
                        old_sx = scene.subject_x
                        # VLM signal — derived from the parsed
                        # scene.subject_box if present (Phase 1
                        # grounded output), else fall back to the
                        # model's legacy subject_x.
                        vlm_x = float(scene.subject_x)
                        if scene.subject_box:
                            box = scene.subject_box
                            vlm_x = ((box[0] + box[2]) / 2.0) * 100.0
                        vlm_conf = float(scene.vlm_confidence or 0.0)
                        # If the VLM didn't report a confidence
                        # (legacy parser), treat its subject_x as a
                        # low-confidence hint.
                        if vlm_conf == 0.0 and scene.subject_x != 50:
                            vlm_conf = 0.5
                        apply_fusion_to_scene(
                            scene,
                            face_x=float(best_face.nose_x),
                            face_conf=float(best_face.confidence),
                            vlm_x=vlm_x,
                            vlm_conf=vlm_conf,
                        )
                        if abs(old_sx - scene.subject_x) > 5:
                            enriched += 1
                else:
                    # ── Legacy hard-override path (unchanged) ──
                    if _is_continuous or is_closeup:
                        # Dominant-subject / closeup: use raw face position (no slot snap)
                        dominant = _select_dominant_face(best_dfr.faces)
                        if dominant:
                            old_sx = scene.subject_x
                            scene.subject_x = int(round(dominant.nose_x))
                            if abs(old_sx - scene.subject_x) > 5:
                                enriched += 1
                    elif face_registry and face_registry.multi_speaker:
                        # Multi-speaker: prefer active speaker, snap to slot center
                        speaking = [f for f in best_dfr.faces if f.lip_aperture > 0.03]
                        target_face = None
                        if speaking:
                            target_face = max(speaking, key=lambda f: f.lip_aperture)
                        elif best_dfr.primary_face_idx >= 0:
                            target_face = best_dfr.faces[best_dfr.primary_face_idx]
                        if target_face:
                            slot = face_registry.nearest_slot(target_face.nose_x)
                            if slot:
                                old_sx = scene.subject_x
                                scene.subject_x = int(round(slot.x_center))
                                if abs(old_sx - scene.subject_x) > 5:
                                    enriched += 1
                    elif best_dfr.primary_face_idx >= 0:
                        primary = best_dfr.faces[best_dfr.primary_face_idx]
                        old_sx = scene.subject_x
                        scene.subject_x = int(round(primary.nose_x))
                        if abs(old_sx - scene.subject_x) > 5:
                            enriched += 1
        if enriched > 0:
            logger.info("[%s] Dense face data overrode subject_x on %d/%d scenes", job_id, enriched, len(scenes))

    # Backfill scene face data from sparse detection
    if _sparse_face_map and scenes:
        for scene in scenes:
            if scene.face_count > 0:
                continue
            ts_key = round(scene.timestamp, 1)
            if ts_key in _sparse_face_map:
                sfd = _sparse_face_map[ts_key]
                scene.face_count = len(sfd.faces)

    # ── Active Speaker Detection (lip-audio cross-correlation) ──
    await _update_progress(
        job_id, JobStatus.DETECTING_CLIPS, 65,
        "Detecting active speakers and building layout...",
    )
    active_speaker_events = []
    if face_registry and face_registry.multi_speaker and transcript and (face_results or dense_face_results):
        try:
            from backend.services.active_speaker import (
                build_active_speaker_timeline,
                build_active_speaker_timeline_v2,
                get_active_slot_at_time,
            )
            # Use V2 (identity-based) with dense data when available
            _speaker_face_data = dense_face_results if dense_face_results else face_results
            _shot_cuts_for_speaker = scene_cut_timestamps if scene_cut_timestamps else []
            # Phase C: Light-ASD audio-visual backend (default ON as
            # of Task C). v2's lip-aperture heuristic over-switched on
            # any non-speech mouth motion (laughing, eating, hand
            # gestures near the face) and couldn't represent
            # overlapping speech or off-camera speakers. Light-ASD's
            # audio-visual model handles all three; the v2 fallback
            # below preserves the old behavior automatically when the
            # model is unavailable or returns zero scores. Override
            # with CLIPAI_ASD_BACKEND=heuristic to revert without a
            # redeploy.
            asd_backend = os.environ.get("CLIPAI_ASD_BACKEND", "light_asd").lower()
            if asd_backend == "light_asd" and dense_face_results:
                try:
                    from backend.services.light_asd import score_faces_for_clip
                    from backend.services.active_speaker import (
                        build_active_speaker_timeline_v3,
                    )
                    asd_scores = score_faces_for_clip(
                        video_path=video_path,
                        face_results=_speaker_face_data,
                        audio_wav_path=audio_path,
                    )
                    if asd_scores:
                        active_speaker_events = build_active_speaker_timeline_v3(
                            _speaker_face_data, transcript,
                            asd_scores=asd_scores,
                            face_registry=face_registry,
                            window_seconds=0.5,
                            shot_cuts=_shot_cuts_for_speaker,
                            audio_path=audio_path,
                        )
                    else:
                        logger.info(
                            "[%s] Light-ASD produced no scores — falling "
                            "back to v2 heuristic ASD",
                            job_id,
                        )
                        asd_backend = "heuristic"
                except Exception as e:
                    logger.warning(
                        "[%s] Light-ASD path failed (non-fatal): %s — "
                        "falling back to v2",
                        job_id, e,
                    )
                    asd_backend = "heuristic"

            if not active_speaker_events and dense_face_results:
                active_speaker_events = build_active_speaker_timeline_v2(
                    _speaker_face_data, transcript, face_registry,
                    window_seconds=0.5,
                    shot_cuts=_shot_cuts_for_speaker,
                    audio_path=audio_path,
                )
            elif not active_speaker_events:
                active_speaker_events = build_active_speaker_timeline(
                    _speaker_face_data, transcript, face_registry,
                    window_seconds=0.5 if dense_face_results else 2.0,
                    shot_cuts=_shot_cuts_for_speaker,
                    audio_path=audio_path,
                )
            if active_speaker_events:
                logger.info(
                    "[%s] Active speaker timeline: %d events covering %.1fs",
                    job_id, len(active_speaker_events),
                    sum(e.end - e.start for e in active_speaker_events),
                )
                # Correct scenes where lip tracking disagrees with subject_x
                # Only apply correction when confidence is high enough to be reliable.
                # Low confidence (< 0.3) means lip aperture data was ambiguous —
                # the visual scene analysis is more trustworthy in that case.
                lip_corrected = 0
                avg_confidence = sum(e.confidence for e in active_speaker_events) / max(len(active_speaker_events), 1)
                min_correction_confidence = 0.3  # Don't trust lip-audio below 30%
                logger.info(
                    "[%s] Active speaker avg confidence=%.2f (threshold=%.2f for correction)",
                    job_id, avg_confidence, min_correction_confidence,
                )
                if avg_confidence >= min_correction_confidence:
                    for scene in scenes:
                        active_slot_id = get_active_slot_at_time(
                            active_speaker_events, scene.timestamp,
                        )
                        if active_slot_id >= 0:
                            # Find the event at this timestamp and check its confidence
                            event_confidence = 0.0
                            for ev in active_speaker_events:
                                if ev.start <= scene.timestamp <= ev.end:
                                    event_confidence = ev.confidence
                                    break
                            if event_confidence < min_correction_confidence:
                                continue  # Skip low-confidence corrections
                            slot = face_registry.slot_by_id(active_slot_id)
                            if slot and abs(scene.subject_x - slot.x_center) > 15:
                                scene.subject_x = int(round(slot.x_center))
                                scene.active_slot = active_slot_id if active_slot_id >= 0 else None
                                lip_corrected += 1
                    if lip_corrected > 0:
                        logger.info(
                            "[%s] Lip-audio correction: updated %d/%d scenes to match active speaker (avg_conf=%.2f)",
                            job_id, lip_corrected, len(scenes), avg_confidence,
                        )
                else:
                    logger.info(
                        "[%s] Lip-audio correction SKIPPED: avg confidence %.2f < threshold %.2f — "
                        "visual scene analysis is more reliable",
                        job_id, avg_confidence, min_correction_confidence,
                    )
        except Exception as e:
            logger.warning("[%s] Active speaker detection failed (non-fatal): %s", job_id, e)

    # ── Fallback: Lip-only speaker detection when diarization found 1 speaker ──
    if face_registry and face_registry.multi_speaker:
        try:
            unique_speakers_detected = len(set(
                seg.speaker for seg in transcript
                if hasattr(seg, 'speaker') and seg.speaker
            )) if transcript else 0
            active_with_slot = sum(1 for e in active_speaker_events if e.slot_id >= 0)

            if unique_speakers_detected <= 1 or active_with_slot < 5:
                logger.info(
                    "[%s] Diarization found %d speaker(s) but registry has %d slots — "
                    "building lip-only speaker timeline",
                    job_id, unique_speakers_detected, len(face_registry.slots),
                )
                _lip_events = []
                _face_data = dense_face_results if dense_face_results else face_results
                for fr in _face_data:
                    if len(fr.faces) < 2:
                        continue
                    speaking_face = max(fr.faces, key=lambda f: f.lip_aperture)
                    if speaking_face.lip_aperture < 0.02:
                        continue
                    slot = face_registry.nearest_slot(speaking_face.nose_x)
                    if slot:
                        from backend.services.active_speaker import SpeakerEvent
                        _lip_events.append(SpeakerEvent(
                            start=fr.timestamp - 0.5,
                            end=fr.timestamp + 0.5,
                            slot_id=slot.slot_id,
                            confidence=min(1.0, speaking_face.lip_aperture / 0.05),
                        ))

                if len(_lip_events) > max(len(active_speaker_events), 1) * 0.5:
                    _lip_events.sort(key=lambda e: e.start)
                    merged = [_lip_events[0]]
                    for ev in _lip_events[1:]:
                        if ev.slot_id == merged[-1].slot_id and ev.start - merged[-1].end < 2.0:
                            merged[-1] = SpeakerEvent(
                                start=merged[-1].start, end=ev.end,
                                slot_id=ev.slot_id,
                                confidence=max(merged[-1].confidence, ev.confidence),
                            )
                        else:
                            merged.append(ev)
                    active_speaker_events = merged
                    avg_conf = sum(e.confidence for e in merged) / max(len(merged), 1)
                    logger.info(
                        "[%s] Lip-only speaker timeline: %d events, avg confidence=%.2f",
                        job_id, len(merged), avg_conf,
                    )
        except Exception as e:
            logger.warning("[%s] Lip-only speaker fallback failed (non-fatal): %s", job_id, e)

    # ── Gap 5a: Diarization fusion (second independent vote) ──
    # Run a speaker-side audio diarization pass (pyannote → MFCC →
    # off) and fuse the resulting speaker clusters with the lip-
    # motion timeline produced above. The two signals are
    # independent — lip-motion sees faces, diarization sees
    # waveforms — so agreement is strong evidence and disagreement
    # reliably flags crosstalk frames. The fusion is strictly
    # additive: empty diarization or a single-cluster result leaves
    # ``active_speaker_events`` bit-identical to the pre-fusion
    # timeline. Phase B threads the cluster→slot mapping down to
    # the reframe segmenter via ``_diar_segments`` / ``_fusion``
    # locals that are read at the existing ``build_reframe_segments``
    # call below.
    _diar_segments: list = []
    _fusion = None
    try:
        from backend.services.speaker_diarization import (
            USE_DIARIZATION, diarize_audio, fuse_lip_and_diarization,
        )
        _num_slots = len(face_registry.slots) if face_registry else 0
        # Face-slot count is a NOISY LOWER BOUND on speaker count —
        # not an exact number. Always attempt diarization when we
        # have audio + a lip timeline; the slot count becomes a
        # hint that the bounded kwargs in ``_diarize_pyannote``
        # translate into ``min_speakers`` / ``max_speakers``. Passing
        # ``_num_slots == 1`` here still produces a 1..3 speaker
        # bracket which catches the profile-only / off-screen /
        # dark-scene cases where face detection missed a speaker.
        if USE_DIARIZATION and audio_path and active_speaker_events:
            _diar_start = _time.time()
            _diar_hint = max(_num_slots, 2) if _num_slots > 0 else None
            _diar_segments = diarize_audio(audio_path, num_speakers=_diar_hint)
            if _diar_segments:
                _fusion = fuse_lip_and_diarization(
                    active_speaker_events, _diar_segments,
                )
                active_speaker_events = _fusion.events
                logger.info(
                    "[%s] Diarization fusion: agree=%d disagree=%d "
                    "override=%d (%.1fs)",
                    job_id, _fusion.agree_count, _fusion.disagree_count,
                    _fusion.override_count, _time.time() - _diar_start,
                )
            else:
                logger.info(
                    "[%s] Diarization produced no segments; "
                    "lip-only path retained", job_id,
                )
    except Exception as _diar_exc:
        logger.warning(
            "[%s] Diarization fusion skipped: %s", job_id, _diar_exc,
        )

    # ── Speaker → Face Slot Mapping (BEFORE per-second synthesis) ──
    # Map Whisper speaker labels to face registry slots using audio diarization.
    # This must run before synthesis so transcript-driven tracking can be used.
    speaker_slot_map = {}
    transcript_speaker_events = []
    if face_registry and face_registry.multi_speaker and transcript:
        try:
            from backend.services.active_speaker import (
                map_speakers_to_face_slots,
                build_transcript_speaker_timeline,
            )
            _mapping_face_data = dense_face_results if dense_face_results else face_results
            speaker_slot_map = map_speakers_to_face_slots(
                transcript, face_registry, _mapping_face_data,
                scenes=scenes,
                original_scene_sx=_original_scene_sx,
            )
            if speaker_slot_map:
                logger.info("[%s] Speaker→slot mapping: %s", job_id, speaker_slot_map)
                transcript_speaker_events = build_transcript_speaker_timeline(
                    transcript, speaker_slot_map, face_registry,
                )
                logger.info(
                    "[%s] Transcript-driven timeline: %d events (PRIMARY source for tracking)",
                    job_id, len(transcript_speaker_events),
                )
        except Exception as e:
            logger.warning("[%s] Speaker→slot mapping failed (non-fatal): %s", job_id, e)

    # ── Content classification + persistent region detection ──
    _content_profile = None
    _persistent_regions = None
    if dense_face_results and face_registry and scenes:
        try:
            from backend.services.content_classifier import classify_content, USE_CONTENT_AWARE_REFRAME
            _video_dur = metadata.get("duration", 0)
            _shot_cuts = scene_cut_timestamps if scene_cut_timestamps else []
            # Inject the user's content-type override into the metadata
            # dict the classifier receives. The ffprobe metadata dict
            # historically did NOT carry ``content_type_override`` at
            # all, so the classifier's user-override branch only fired
            # by accident for "podcast" (whose spelling happened to
            # match the enum value) and silently never fired for
            # "gameplay" or "movie". This is the Phase 1 plumbing fix.
            #
            # Phase 2 also injects the anime / music sub-dropdown values
            # plus the game key so the classifier can populate the new
            # profile.anime_subtype / profile.music_subtype /
            # profile.game_type fields. classify_content gates each
            # sub-type to the matching parent (anime_subtype is ignored
            # if the user picked a non-anime parent type) so a stale
            # selection from a previous job can't leak through.
            _classifier_metadata = dict(metadata) if metadata else {}
            if _content_override:
                _classifier_metadata["content_type_override"] = _content_override
            if _anime_subtype:
                _classifier_metadata["anime_subtype"] = _anime_subtype
            if _music_subtype:
                _classifier_metadata["music_subtype"] = _music_subtype
            if _sports_subtype:
                _classifier_metadata["sports_subtype"] = _sports_subtype
            if _game_type:
                _classifier_metadata["game_type"] = _game_type
            # Week 2 Parts D + E: thread object detections + beat grid
            # into the classifier so the sports / music-video subtype
            # auto-promotion helpers can see them. Both degrade
            # gracefully when absent (the helpers return None).
            _classifier_metadata["frame_objects"] = (
                _object_detections
                if "_object_detections" in locals() and _object_detections
                else None
            )
            _classifier_metadata["music_beat_grid"] = (
                locals().get("_music_beat_grid_for_seg") or None
            )
            _content_profile = classify_content(
                shot_cuts=_shot_cuts,
                face_registry=face_registry,
                dense_faces=dense_face_results,
                scenes=scenes,
                video_duration=_video_dur,
                metadata=_classifier_metadata,
                job_id=job_id,
                transcript_segments=transcript,
            )
        except Exception as e:
            logger.warning("[%s] Content classification failed (non-fatal): %s", job_id, e)

        # ── Gameplay subtype backfill ──
        #
        # When the early pre-detect (filename hint / sparse faces)
        # flipped ``_is_gameplay=True`` but the content classifier
        # settled on a non-gameplay content_type (common on TF2
        # where cartoon-character face detection pushes the
        # classifier toward "sports" or "narrative"), the
        # resulting profile has ``gameplay_subtype=None`` — which
        # means the CENTER_BIAS_GENRES guard in
        # :mod:`reframe_segmenter` never fires. Backfill
        # ``gameplay_subtype`` from the filename hint's genre (or
        # a plain ``"gameplay"`` fallback) so the downstream
        # center-bias path is active for every segment.
        if _is_gameplay and _content_profile is not None:
            _existing_sub = getattr(
                _content_profile, "gameplay_subtype", None,
            )
            if not _existing_sub:
                _backfill_sub = _filename_game_genre or "gameplay"
                try:
                    _content_profile.gameplay_subtype = _backfill_sub
                    if _filename_game_slug and not getattr(
                        _content_profile, "game_type", "",
                    ):
                        _content_profile.game_type = _filename_game_slug
                    logger.info(
                        "[%s] Gameplay subtype backfill: "
                        "content_type=%s → gameplay_subtype='%s' "
                        "(filename_slug='%s')",
                        job_id,
                        getattr(_content_profile, "content_type", "?"),
                        _backfill_sub,
                        _filename_game_slug or "-",
                    )
                except Exception as _bf_e:
                    logger.debug(
                        "[%s] Gameplay subtype backfill failed "
                        "(non-fatal): %s",
                        job_id, _bf_e,
                    )

        # ── Anime dense re-run fallback (Week 2 Part A Option 1) ──
        # If the early anime hint was False (user didn't pick anime,
        # ANIME_MODE_DETECTED didn't fire during the sparse pass) but
        # the classifier concluded is_animated=True, the first dense
        # pass ran live-action-only. Re-run with is_animated=True so
        # the lbpcascade augmentation gets a shot at the weak frames.
        # Gated on a low face-per-frame rate (<0.4) so we don't spend
        # the extra dense minute on clips that already have good
        # live-action coverage.
        _anime_profile = (
            _content_profile
            and getattr(_content_profile, "is_animated", False)
        )
        if (
            _anime_profile
            and not _early_anime_hint
            and dense_face_results
            and settings.SUBJECT_TRACKING_ENABLED
        ):
            try:
                _faces_per_frame = sum(
                    1 for fr in dense_face_results if getattr(fr, "faces", None)
                ) / max(1, len(dense_face_results))
                if _faces_per_frame < 0.4:
                    logger.info(
                        "[%s] Anime re-run fallback: profile.is_animated=True "
                        "but early hint was False; face coverage %.2f < 0.4. "
                        "Re-running dense detection with is_animated=True.",
                        job_id, _faces_per_frame,
                    )
                    _dense_rerun_loop = asyncio.get_event_loop()
                    _video_dur_rr = metadata.get("duration", 0)
                    _dense_sr_rr = settings.DENSE_FACE_SAMPLE_RATE
                    _rerun = await _dense_rerun_loop.run_in_executor(
                        None,
                        lambda: detect_faces_dense(
                            video_path,
                            start=0,
                            end=_video_dur_rr,
                            sample_rate=_dense_sr_rr,
                            min_confidence=0.4,
                            extract_embeddings=True,
                            is_animated=True,
                        ),
                    )
                    if _rerun:
                        dense_face_results = _rerun
                        logger.info(
                            "[%s] Anime re-run: replaced dense results "
                            "(%d frames, %d with faces)",
                            job_id, len(_rerun),
                            sum(1 for r in _rerun if r.faces),
                        )
            except Exception as _re_err:
                logger.warning(
                    "[%s] Anime re-run fallback failed (non-fatal): %s",
                    job_id, _re_err,
                )

        # ── Anime shot detector augmentation (Week 2 Part B) ──
        # When the classifier concluded the clip is animated, run the
        # histogram / edge-density shot detector and merge its cut
        # timestamps into scene_cut_timestamps. Anime action scenes
        # have 6-8-frame shots that PySceneDetect's default luminance-
        # diff detector misses; the histogram signal catches them.
        # De-duplicate within ±0.3s so a single cut that both detectors
        # found doesn't double-count.
        if (
            _content_profile
            and getattr(_content_profile, "is_animated", False)
        ):
            try:
                from backend.services.anime_shot_detector import (
                    USE_ANIME_SHOT_DETECTOR,
                    detect_anime_shots,
                )
            except ImportError:
                USE_ANIME_SHOT_DETECTOR = False

            if USE_ANIME_SHOT_DETECTOR:
                try:
                    anime_shot_result = detect_anime_shots(
                        video_path,
                        video_duration=metadata.get("duration", 0) or None,
                    )
                    if (
                        anime_shot_result.cut_times
                        and not anime_shot_result.skipped_reason
                    ):
                        if scene_cut_timestamps is None:
                            scene_cut_timestamps = []
                        existing_cuts = [
                            float(t) for t in scene_cut_timestamps
                        ]
                        added = 0
                        for t in anime_shot_result.cut_times:
                            if any(abs(t - e) < 0.30 for e in existing_cuts):
                                continue
                            existing_cuts.append(float(t))
                            added += 1
                        existing_cuts.sort()
                        scene_cut_timestamps = existing_cuts
                        logger.info(
                            "[%s] Anime shot detector: +%d cuts "
                            "(total now %d)",
                            job_id, added, len(scene_cut_timestamps),
                        )
                    elif anime_shot_result.skipped_reason:
                        logger.info(
                            "[%s] Anime shot detector skipped: %s",
                            job_id, anime_shot_result.skipped_reason,
                        )
                except Exception as e:
                    logger.warning(
                        "[%s] Anime shot detector failed (non-fatal): %s",
                        job_id, e,
                    )

        try:
            from backend.services.persistent_region_detector import detect_persistent_regions
            _persistent_regions = detect_persistent_regions(
                dense_faces=dense_face_results,
                face_registry=face_registry,
                video_duration=metadata.get("duration", 0),
                job_id=job_id,
            )
        except Exception as e:
            logger.warning("[%s] Persistent region detection failed (non-fatal): %s", job_id, e)

    # ── Reframe Segmenter (replaces per-second synthesis when enabled) ──
    _reframe_segments_used = False
    # v2 Phase 11 (Fix 7): hoist so the downstream plan_layout branch
    # can see the segmenter output and skip a redundant second solve.
    reframe_segments: list = []
    _pacing_estimator = None
    # Hoisted so both AUTOFLIP and REFRAME_SEGMENTER branches can reference it
    # without UnboundLocalError when the AUTOFLIP path doesn't execute.
    _interpolated_timeline = None
    # v4.1 hotfix: hoist source_width / source_height to function scope.
    # Both the AUTOFLIP (line ~2668) and REFRAME_SEGMENTER (line ~2764)
    # branches previously referenced a bare `source_width` when converting
    # pixel subject_x to a 0-100 int for SceneDescription. That variable
    # only existed as a kwarg name inside build_reframe_segments —
    # NameError at pipeline scope. Hoist it here once so both branches
    # (and any future additions) can read the same local.
    source_width = int(metadata.get("width", 1920) or 1920)
    source_height = int(metadata.get("height", 1080) or 1080)
    _has_speaker_data = active_speaker_events or transcript_speaker_events

    # ── Layer 1: per-second saliency peaks for the critic engine ──
    # One-shot extraction shared by both reframer branches below
    # (USE_AUTOFLIP_REFRAME and the regular REFRAME_SEGMENTER path).
    # Output goes to ``human_reframe_bridge.maybe_override_render_plan``
    # which threads it into ``reframe_critic.auto_repair_plan`` so
    # windows that exclude the salient region get auto-widened.
    #
    # Backend selection chain (in av_saliency.AvSaliency.__post_init__):
    #   1. TASED-Net (operator-supplied — see infra/saliency/README.md)
    #   2. cv2.saliency.StaticSaliencySpectralResidual_create (always
    #      available via opencv-contrib-python-headless)
    #
    # The block is wrapped in a single try/except so a saliency
    # failure can't break the export. ``CLIPAI_SALIENCY_LAYER=0``
    # disables the layer entirely.
    _l1_saliency_peaks_per_second = None
    if (
        os.environ.get("CLIPAI_SALIENCY_LAYER", "1") == "1"
        and frames
    ):
        try:
            _l1_saliency_peaks_per_second = _compute_l1_saliency_peaks(
                frames=frames,
                audio_path=audio_path,
                duration_sec=float(metadata.get("duration", 0) or 0.0),
                job_id=job_id,
                cache_dir=os.path.dirname(audio_path) if audio_path else None,
            )
        except Exception as _l1_exc:
            logger.warning(
                "[%s] Layer 1 saliency failed (%s) — proceeding without",
                job_id, _l1_exc,
            )
            _l1_saliency_peaks_per_second = None

    USE_AUTOFLIP_REFRAME = os.environ.get("USE_AUTOFLIP_REFRAME", "false").lower() in ("true", "1", "yes")
    if dense_face_results and face_registry and scenes and _has_speaker_data:
        try:
            from backend.services.reframe_segmenter import USE_REFRAME_SEGMENTER, build_reframe_segments
            from backend.services.human_reframe_pipeline import (
                HUMAN_REFRAME_ALLOWED_CONTENT_TYPES,
                is_compare_mode_enabled,
                is_pipeline_flag_enabled as _is_human_reframe_flag_enabled,
                try_run_human_reframe_pipeline,
            )

            # Phase A: try the SOTA human-reframe path first when its
            # flag + content-type allowlist gates are open. Returns
            # None on any gating miss / runtime failure so we fall
            # through to the legacy autoflip / segmenter branches
            # below without behavior change.
            _ct_for_human = (
                getattr(_content_profile, "content_type", "") or ""
                if _content_profile else ""
            )
            _human_outcome = None
            if (
                _is_human_reframe_flag_enabled()
                and _ct_for_human in HUMAN_REFRAME_ALLOWED_CONTENT_TYPES
                and not _is_gameplay
            ):
                _human_outcome = try_run_human_reframe_pipeline(
                    duration_sec=float(metadata.get("duration", 0) or 0.0),
                    source_width=source_width,
                    source_height=source_height,
                    source_fps=float(metadata.get("fps", 30.0) or 30.0),
                    content_type=_ct_for_human,
                    dense_faces=dense_face_results,
                    active_speaker_events=active_speaker_events,
                    shot_boundaries=scene_cut_timestamps if scene_cut_timestamps else [],
                    job_id=job_id,
                )

            _reframe_path = "human_reframe" if _human_outcome is not None else (
                "autoflip" if USE_AUTOFLIP_REFRAME else
                "reframe_segmenter" if USE_REFRAME_SEGMENTER else "legacy"
            )
            logger.info("[%s] Reframe mode: %s", job_id, _reframe_path.upper())

            if _human_outcome is not None:
                # ── Phase A: human-reframe path ──
                reframe_segments, _human_rp = _human_outcome
                if reframe_segments:
                    from backend.models import SceneDescription
                    ai_scenes = [s for s in scenes if s.description != "[dense face tracking]"]
                    for seg in reframe_segments:
                        _desc = (
                            f"[human_reframe:{seg.reason}:{seg.ease_in_ms}:"
                            f"{seg.strategy}:{seg.confidence:.2f}]"
                        )
                        _sx_int = int(round(seg.subject_x / source_width * 100.0)) if source_width > 0 else 50
                        _asx = _sx_int if seg.active_slot is not None else None
                        ai_scenes.append(SceneDescription(
                            timestamp=float(seg.start),
                            description=_desc,
                            importance_score=5,
                            thumbnail_path="",
                            subject_x=_sx_int,
                            active_speaker_x=_asx,
                            active_slot=seg.active_slot,
                            layout_mode=seg.layout,
                            precise_x=float(seg.subject_x),
                            precise_y=float(seg.subject_y),
                            face_count=len(face_registry.slots) if face_registry else 0,
                            face_positions=[],
                        ))
                    ai_scenes.sort(key=lambda s: s.timestamp)
                    scenes = ai_scenes
                    _tracking_mode = "multi_cluster"
                    _human_rp_dict = _human_rp.to_dict()
                    await database.update_job_status(
                        job_id,
                        scenes=list(scenes),
                        tracking_mode=_tracking_mode,
                        render_plan=_human_rp_dict,
                    )
                    logger.info(
                        "[%s] *** HumanReframePipeline: %d segments → %d scenes ***",
                        job_id, len(reframe_segments), len(scenes),
                    )
                    _reframe_segments_used = True

                    # Compare mode: log the marker so the export step
                    # (or an offline eyeball script) knows both paths
                    # are wanted. The actual ``<job>_legacy.mp4``
                    # render is produced by re-running analysis with
                    # ``CLIPAI_HUMAN_REFRAME_PIPELINE=0`` and exporting
                    # again — see docs/human_reframe_rollout.md for
                    # the full eyeball workflow.
                    if is_compare_mode_enabled():
                        try:
                            await database.update_job_status(
                                job_id, reframe_compare_mode=True,
                            )
                            logger.info(
                                "[%s] HumanReframePipeline: compare mode "
                                "marker set; primary=human_reframe "
                                "(see docs/human_reframe_rollout.md "
                                "for the legacy-render workflow)",
                                job_id,
                            )
                        except Exception as _cmp_e:
                            logger.warning(
                                "[%s] compare-mode marker update failed (%s)",
                                job_id, _cmp_e,
                            )

            elif USE_AUTOFLIP_REFRAME and not _is_gameplay:
                # ── AutoFlip reframe path ──
                from backend.services.autoflip_segmenter import build_autoflip_segments
                _video_dur = metadata.get("duration", 0)
                _shot_cuts = scene_cut_timestamps if scene_cut_timestamps else []

                # ── Subject Fusion (feature-flagged or auto for action types) ──
                # v4: auto-enable subject fusion for content types where
                # non-face subjects persist across cuts (anime, sports,
                # music video, vlog). These are the cases where the
                # face_registry alone can't carry identity through the
                # long stretches without a clean face bbox.
                _ct_str_fusion = (
                    getattr(_content_profile, "content_type", "")
                    if _content_profile else ""
                )
                _fusion_auto = _ct_str_fusion in (
                    "anime",
                    "animation_dialogue",
                    "sports",
                    "sports_basketball",
                    "sports_racing",
                    "music_video",
                    "vlog",
                )
                USE_SUBJECT_FUSION = (
                    os.environ.get("USE_SUBJECT_FUSION", "false").lower() in ("true", "1", "yes")
                    or _fusion_auto
                )
                _subject_tracks = None
                if USE_SUBJECT_FUSION and face_registry and getattr(
                    face_registry, "slots", None,
                ):
                    try:
                        from backend.services.subject_fusion import build_subject_tracks
                        _subject_tracks = build_subject_tracks(
                            face_registry=face_registry,
                            dense_faces=dense_face_results,
                            saliency_regions=_saliency_regions if '_saliency_regions' in dir() else [],
                            frame_paths=[(f.timestamp, f.path) for f in frames],
                            source_width=metadata.get("width", 1920),
                            source_height=metadata.get("height", 1080),
                            shot_cuts=_shot_cuts,
                            job_id=job_id,
                        )
                        logger.info(
                            "[%s] SubjectFusion: %d unified tracks (auto=%s, ct=%s)",
                            job_id,
                            len(_subject_tracks) if _subject_tracks else 0,
                            _fusion_auto, _ct_str_fusion,
                        )
                    except Exception as e:
                        logger.warning("[%s] Subject fusion failed (non-fatal): %s", job_id, e)

                logger.info("[%s] Reframe mode: AUTOFLIP (fusion: %s)", job_id,
                            "enabled" if USE_SUBJECT_FUSION else "disabled")

                # ── Dense Propagation (optional, behind feature flag) ──
                USE_DENSE_PROPAGATION = os.environ.get("USE_DENSE_PROPAGATION", "false").lower() in ("true", "1", "yes")
                if USE_DENSE_PROPAGATION and dense_face_results and frames:
                    try:
                        from backend.services.dense_propagator import build_interpolated_timeline
                        _interpolated_timeline = build_interpolated_timeline(
                            dense_face_results=dense_face_results,
                            frame_paths=[(f.timestamp, f.path) for f in frames],
                            source_width=metadata.get("width", 1920),
                            source_height=metadata.get("height", 1080),
                            source_fps=metadata.get("fps", 30.0),
                            shot_cuts=_shot_cuts,
                            backend=os.environ.get("PROPAGATION_BACKEND", "KCF"),
                            runtime_budget_sec=240.0,
                            job_id=job_id,
                        )
                        if _interpolated_timeline:
                            logger.info("[%s] DensePropagation: %s",
                                        job_id, _interpolated_timeline.to_dict_summary())
                    except Exception as e:
                        logger.warning("[%s] Dense propagation failed (non-fatal): %s", job_id, e)
                        _interpolated_timeline = None

                reframe_segments = build_autoflip_segments(
                    shot_cuts=_shot_cuts,
                    face_registry=face_registry,
                    active_speaker_events=active_speaker_events,
                    dense_faces=dense_face_results,
                    saliency_keyframes=_saliency_keyframes,
                    transcript_segments=transcript,
                    speaker_to_slot=speaker_slot_map,
                    video_duration=_video_dur,
                    source_width=metadata.get("width", 1920),
                    source_height=metadata.get("height", 1080),
                    persistent_regions=_persistent_regions,
                    saliency_regions=_saliency_regions,
                    object_detections=_object_detections,
                    subject_tracks=_subject_tracks,
                    target_aspect=9/16,
                    job_id=job_id,
                    interpolated_timeline=_interpolated_timeline,
                    frame_saliency=_saliency_regions if '_saliency_regions' in dir() else None,
                )
                if reframe_segments:
                    from backend.models import SceneDescription
                    ai_scenes = [s for s in scenes if s.description != "[dense face tracking]"]
                    for seg in reframe_segments:
                        _desc = f"[autoflip:{seg.reason}:{seg.ease_in_ms}:{seg.strategy}:{seg.confidence:.2f}]"
                        # Convert pixel-space subject_x/y to 0-100 int for SceneDescription
                        _sx_int = int(round(seg.subject_x / source_width * 100.0)) if source_width > 0 else 50
                        _asx = _sx_int if seg.active_slot is not None else None
                        ai_scenes.append(SceneDescription(
                            timestamp=float(seg.start),
                            description=_desc,
                            importance_score=5,
                            thumbnail_path="",
                            subject_x=_sx_int,
                            active_speaker_x=_asx,
                            active_slot=seg.active_slot,
                            layout_mode=seg.layout,
                            precise_x=float(seg.subject_x),
                            precise_y=float(seg.subject_y),
                            face_count=len(face_registry.slots) if face_registry else 0,
                            face_positions=[],
                        ))
                    ai_scenes.sort(key=lambda s: s.timestamp)
                    scenes = ai_scenes
                    _tracking_mode = "multi_cluster"
                    await database.update_job_status(
                        job_id, scenes=list(scenes), tracking_mode=_tracking_mode,
                    )
                    logger.info(
                        "[%s] *** AutoFlipSegmenter: %d segments → %d scenes ***",
                        job_id, len(reframe_segments), len(scenes),
                    )
                    _reframe_segments_used = True

                    try:
                        from backend.services.render_plan import USE_RENDER_PLAN
                        if USE_RENDER_PLAN:
                            from backend.services.render_plan_builder import build_render_plan
                            from backend.services.render_plan_debug import build_debug_payload
                            _rp = build_render_plan(
                                segments=reframe_segments,
                                source_width=metadata.get("width", 1920),
                                source_height=metadata.get("height", 1080),
                                source_fps=metadata.get("fps", 30.0),
                                target_aspect="9:16",
                            )
                            try:
                                from backend.services.human_reframe_bridge import (
                                    maybe_override_render_plan as _maybe_human_rp,
                                )
                                _rp = _maybe_human_rp(
                                    _rp,
                                    dense_faces=dense_face_results,
                                    active_speaker_events=active_speaker_events,
                                    shot_boundaries=_shot_cuts,
                                    content_type=getattr(_content_profile, "content_type", "") if _content_profile else "",
                                    duration_sec=float(metadata.get("duration", 0) or 0.0),
                                    source_width=int(metadata.get("width", 1920) or 1920),
                                    source_height=int(metadata.get("height", 1080) or 1080),
                                    source_fps=float(metadata.get("fps", 30.0) or 30.0),
                                    job_id=job_id,
                                    saliency_peaks_per_second=_l1_saliency_peaks_per_second,
                                )
                            except Exception as _hr_e:
                                logger.info("[%s] human-reframe hook skipped: %s", job_id, _hr_e)
                            # Phase 10: attach debug payload (content
                            # routing + per-segment editorial + reason
                            # tags) so the ReframeDebugOverlay chips
                            # light up without any extra round-trip.
                            _rp_dict = _rp.to_dict()
                            _rp_dict["debug"] = build_debug_payload(
                                job=await database.get_job(job_id),
                                content_profile=_content_profile,
                                reframe_segments=reframe_segments,
                                editorial_report=None,
                                pacing_estimator=_pacing_estimator,
                            )
                            await database.update_job_status(
                                job_id, render_plan=_rp_dict,
                            )
                            logger.info(
                                "[%s] AutoFlip RenderPlan: %d ops, %.1fs duration",
                                job_id, len(_rp.ops), _rp.total_duration_sec,
                            )
                    except Exception as rp_e:
                        logger.warning("[%s] AutoFlip RenderPlan build failed (non-fatal): %s", job_id, rp_e)

            elif USE_REFRAME_SEGMENTER and not _is_gameplay and not _is_continuous:
                _video_dur = metadata.get("duration", 0)
                _shot_cuts = scene_cut_timestamps if scene_cut_timestamps else []

                # ── Build LocalPacingEstimator ──
                try:
                    from backend.services.local_pacing import LocalPacingEstimator, compute_motion_from_dense_faces
                    _ct_str = getattr(_content_profile, 'content_type', 'unknown') if _content_profile else 'unknown'
                    # Promote animated content to "animation_dialogue" pacing
                    # weights — mirrors the segmenter's Fix A routing so
                    # pacing is consistent across stages.
                    if (
                        _ct_str in ("narrative", "unknown", "")
                        and _content_profile
                        and getattr(_content_profile, 'is_animated', False)
                    ):
                        _ct_str = "animation_dialogue"
                    _pacing_estimator = LocalPacingEstimator(_video_dur, content_type=_ct_str or 'unknown')
                    _pacing_estimator.add_shot_cuts(_shot_cuts)
                    _pacing_estimator.add_speaker_turns(active_speaker_events)
                    _motion = compute_motion_from_dense_faces(dense_face_results, _video_dur)
                    _pacing_estimator.add_motion_energy(_motion)
                    _pacing_estimator.compute()
                    logger.info(
                        "[%s] LocalPacingEstimator: mean=%.2f, min_hold range=[%.2f, %.2f]",
                        job_id, float(_pacing_estimator.pacing.mean()),
                        min(_pacing_estimator.get_min_hold_array()),
                        max(_pacing_estimator.get_min_hold_array()),
                    )
                except Exception as pe:
                    logger.warning("[%s] LocalPacingEstimator failed (non-fatal): %s", job_id, pe)
                    _pacing_estimator = None

                # ── Phase 6: Anime saliency anchor extraction ──
                #
                # When the content profile is animated AND the
                # CLIPAI_ANIME_ANCHOR flag is on, build a per-frame
                # AnimeFrameFeatures stream from the existing dense
                # face data + lazy OpenCV motion / contrast /
                # saturation extraction, score it via
                # anime_anchor.score_anime_sequence, and pass the
                # result to build_reframe_segments via the new
                # anime_anchors kwarg.
                #
                # All gated behind the Phase 6 flags (default OFF) so
                # this is a no-op until in-docker validation flips
                # the flags. Wrapped in try/except so any OpenCV /
                # cascade failure stays non-fatal and the existing
                # speaker-tracking path still runs.
                _anime_anchors_for_seg = None
                try:
                    from backend.services.anime_anchor import (
                        USE_ANIME_ANCHOR as _USE_AA,
                        AnimeFrameFeatures,
                        score_anime_sequence,
                    )

                    _auto_anime_anchor = (
                        _content_profile is not None
                        and getattr(_content_profile, "content_type", None)
                        == "animation_dialogue"
                    )
                    if (_USE_AA or _auto_anime_anchor) and _content_profile and getattr(
                        _content_profile, "is_animated", False,
                    ):
                        _features_seq: list = []
                        # Build per-frame features from dense_face_results.
                        # The face signal comes from the highest-conf
                        # face on each frame; motion_energy is derived
                        # from the face-position delta frame-to-frame so
                        # impact-frame detection has real signal even
                        # without the OpenCV cascade. Contrast and
                        # saturation remain 0 until the Phase 6 cascade
                        # lands.
                        for df in dense_face_results:
                            best_face = None
                            best_score = -1.0
                            for f in getattr(df, "faces", []) or []:
                                conf = float(getattr(f, "confidence", 0.0) or 0.0)
                                if conf > best_score:
                                    best_score = conf
                                    best_face = f
                            face_x = float(getattr(best_face, "nose_x", 50.0)) if best_face else 50.0
                            face_y = float(getattr(best_face, "nose_y", 50.0)) if best_face else 50.0
                            face_score = (
                                max(0.0, min(1.0, best_score))
                                if best_face is not None else 0.0
                            )
                            # v4: compute motion_energy from face delta
                            # frame-to-frame. 20% movement = max energy.
                            _prev_face_x = (
                                _features_seq[-1].face_x_pct if _features_seq else face_x
                            )
                            _prev_face_y = (
                                _features_seq[-1].face_y_pct if _features_seq else face_y
                            )
                            _dx = abs(face_x - _prev_face_x)
                            _dy = abs(face_y - _prev_face_y)
                            _motion = min(1.0, (_dx + _dy) / 20.0)
                            _features_seq.append(AnimeFrameFeatures(
                                timestamp=float(df.timestamp),
                                face_x_pct=face_x,
                                face_y_pct=face_y,
                                face_score=face_score,
                                motion_energy=_motion,
                            ))
                        _anime_subtype_pipe = getattr(
                            _content_profile, "anime_subtype", None,
                        )
                        _anime_anchors_for_seg = score_anime_sequence(
                            _features_seq, anime_subtype=_anime_subtype_pipe,
                        )
                        logger.info(
                            "[%s] AnimeAnchor: built %d per-frame anchors (subtype=%s)",
                            job_id, len(_anime_anchors_for_seg), _anime_subtype_pipe,
                        )
                except Exception as _aa_e:
                    logger.warning(
                        "[%s] Anime anchor extraction failed (non-fatal): %s",
                        job_id, _aa_e,
                    )

                # ── Phase 5: Music-video beat detection ──
                #
                # When the content profile is music_video AND the
                # CLIPAI_MUSIC_BEAT_SNAP flag is on, run librosa beat
                # tracking on the audio file extracted by
                # audio_analyzer.py. Lazy import; graceful fallback
                # to None when librosa / audio file aren't available.
                _music_beat_grid_for_seg = None
                try:
                    from backend.services.beat_detector import (
                        USE_MUSIC_BEAT_SNAP as _USE_BS,
                        detect_beats,
                    )

                    _ct_for_beats = (
                        getattr(_content_profile, "content_type", None)
                        if _content_profile else None
                    )
                    _audio_path = f"/data/uploads/{job_id}/audio.wav"
                    if (
                        _USE_BS
                        and _ct_for_beats == "music_video"
                        and os.path.isfile(_audio_path)
                    ):
                        _grid = detect_beats(_audio_path)
                        if _grid.has_data:
                            _music_beat_grid_for_seg = _grid
                            logger.info(
                                "[%s] BeatDetector: tempo=%.1f BPM, %d beats, %d downbeats",
                                job_id, _grid.tempo_bpm,
                                len(_grid.beat_times), len(_grid.downbeat_times),
                            )
                        else:
                            logger.info(
                                "[%s] BeatDetector: empty grid (skipped: %s)",
                                job_id, _grid.lp_message if hasattr(_grid, "lp_message") else "",
                            )
                except Exception as _bs_e:
                    logger.warning(
                        "[%s] Beat detection failed (non-fatal): %s",
                        job_id, _bs_e,
                    )

                # Phase 10: give build_reframe_segments a sidecar
                # dict so the Stage 9b editorial state-machine
                # report is surfaced back to the caller without
                # changing the return signature. Gets attached to
                # the RenderPlan debug payload a few lines below.
                _reframe_debug_out: dict = {}
                reframe_segments = build_reframe_segments(
                    shot_cuts=_shot_cuts,
                    face_registry=face_registry,
                    active_speaker_events=active_speaker_events,
                    dense_faces=dense_face_results,
                    transcript_segments=transcript,
                    speaker_to_slot=speaker_slot_map,
                    video_duration=_video_dur,
                    source_width=metadata.get("width", 1920),
                    source_height=metadata.get("height", 1080),
                    job_id=job_id,
                    content_profile=_content_profile,
                    persistent_regions=_persistent_regions,
                    pacing_estimator=_pacing_estimator,
                    interpolated_timeline=_interpolated_timeline,
                    # Fix 2: feed saliency regions into the segmenter
                    # so the salient-fallback cascade has a non-face
                    # signal to resolve low-confidence segments.
                    frame_saliency=_saliency_regions if '_saliency_regions' in dir() else None,
                    # Phase 5 + Phase 6 follow-up wiring
                    music_beat_grid=_music_beat_grid_for_seg,
                    anime_anchors=_anime_anchors_for_seg,
                    # Gap 5b: fusion result propagation. When the
                    # upstream diarization block ran (≥ 2 slots, flag
                    # on, audio available), ``_diar_segments`` and
                    # ``_fusion`` are populated; otherwise they fall
                    # back to empty/None so the estimator defaults to
                    # the legacy 0.20 lip slice.
                    diarization_segments=(
                        _diar_segments if '_diar_segments' in dir() else None
                    ),
                    cluster_to_slot=(
                        _fusion.cluster_to_slot
                        if '_fusion' in dir() and _fusion is not None
                        else None
                    ),
                    debug_out=_reframe_debug_out,
                )
                if reframe_segments:
                    # Replace scenes with one scene per reframe segment
                    from backend.models import SceneDescription
                    # Keep original AI scenes (non-dense) for other pipeline stages
                    ai_scenes = [s for s in scenes if s.description != "[dense face tracking]"]
                    for seg in reframe_segments:
                        _desc = f"[reframe:{seg.reason}:{seg.ease_in_ms}:{seg.strategy}:{seg.confidence:.2f}:{seg.subject_source or 'unknown'}]"
                        # Convert pixel-space subject_x/y to 0-100 int for SceneDescription
                        _sx_int = int(round(seg.subject_x / source_width * 100.0)) if source_width > 0 else 50
                        _asx = _sx_int if seg.active_slot is not None else None
                        ai_scenes.append(SceneDescription(
                            timestamp=float(seg.start),
                            description=_desc,
                            importance_score=5,
                            thumbnail_path="",
                            subject_x=_sx_int,
                            active_speaker_x=_asx,
                            active_slot=seg.active_slot,
                            layout_mode=seg.layout,
                            precise_x=float(seg.subject_x),
                            precise_y=float(seg.subject_y),
                            face_count=len(face_registry.slots) if face_registry else 0,
                            face_positions=[],
                        ))
                    ai_scenes.sort(key=lambda s: s.timestamp)
                    scenes = ai_scenes
                    _tracking_mode = "multi_cluster"
                    await database.update_job_status(
                        job_id, scenes=list(scenes), tracking_mode=_tracking_mode,
                    )
                    logger.info(
                        "[%s] *** ReframeSegmenter: %d segments → %d scenes saved (was %d per-second) ***",
                        job_id, len(reframe_segments), len(scenes),
                        len(dense_face_results),
                    )
                    _reframe_segments_used = True

                    # ── Build RenderPlan for full video (preview-export parity) ──
                    try:
                        from backend.services.render_plan import USE_RENDER_PLAN
                        if USE_RENDER_PLAN:
                            from backend.services.render_plan_builder import build_render_plan
                            from backend.services.render_plan_debug import build_debug_payload
                            _rp = build_render_plan(
                                segments=reframe_segments,
                                source_width=metadata.get("width", 1920),
                                source_height=metadata.get("height", 1080),
                                source_fps=metadata.get("fps", 30.0),
                                target_aspect="9:16",
                            )
                            try:
                                from backend.services.human_reframe_bridge import (
                                    maybe_override_render_plan as _maybe_human_rp,
                                )
                                _rp = _maybe_human_rp(
                                    _rp,
                                    dense_faces=dense_face_results,
                                    active_speaker_events=active_speaker_events,
                                    shot_boundaries=_shot_cuts,
                                    content_type=getattr(_content_profile, "content_type", "") if _content_profile else "",
                                    duration_sec=float(metadata.get("duration", 0) or 0.0),
                                    source_width=int(metadata.get("width", 1920) or 1920),
                                    source_height=int(metadata.get("height", 1080) or 1080),
                                    source_fps=float(metadata.get("fps", 30.0) or 30.0),
                                    job_id=job_id,
                                    saliency_peaks_per_second=_l1_saliency_peaks_per_second,
                                )
                            except Exception as _hr_e:
                                logger.info("[%s] human-reframe hook skipped: %s", job_id, _hr_e)
                            # Phase 10: attach debug payload so the
                            # ReframeDebugOverlay chip row + per-segment
                            # editorial / reason tags light up with no
                            # extra API round-trip.
                            _rp_dict = _rp.to_dict()
                            _rp_dict["debug"] = build_debug_payload(
                                job=await database.get_job(job_id),
                                content_profile=_content_profile,
                                reframe_segments=reframe_segments,
                                editorial_report=_reframe_debug_out.get("editorial_report"),
                                pacing_estimator=_pacing_estimator,
                            )
                            await database.update_job_status(
                                job_id, render_plan=_rp_dict,
                            )
                            logger.info(
                                "[%s] RenderPlan built: %d ops, %.1fs duration",
                                job_id, len(_rp.ops), _rp.total_duration_sec,
                            )
                    except Exception as rp_e:
                        logger.warning("[%s] RenderPlan build failed (non-fatal): %s", job_id, rp_e)
        except Exception as e:
            logger.warning("[%s] ReframeSegmenter failed (falling back to per-second): %s", job_id, e)

    # ── Per-second scene synthesis from dense face + speaker data (legacy) ──
    # Only used when ReframeSegmenter is disabled or failed.
    # The AI vision model produces ~59 scenes (1 per 10s). Dense face detection
    # has 606 data points (1 per second). Create synthetic scene descriptions at
    # 1-second intervals using the ACTIVE SPEAKER's face position, so the
    # tracking pipeline gets 600+ keyframes instead of 59.
    if not _reframe_segments_used and dense_face_results and face_registry and scenes and _has_speaker_data:
        try:
            await _update_progress(
                job_id, JobStatus.DETECTING_CLIPS, 66,
                f"Building per-second tracking data from {len(dense_face_results)} face frames + "
                f"{len(transcript_speaker_events)} transcript events + {len(active_speaker_events)} lip events...",
            )
            from backend.services.active_speaker import get_active_slot_at_time
            from backend.models import SceneDescription

            def _get_speaker_at_time(timestamp):
                """Get speaker slot. Transcript (audio) > lip-based > fallback.
                With per-timestamp lip verification for couch/panel scenes."""
                transcript_slot = -1
                if transcript_speaker_events:
                    for ev in transcript_speaker_events:
                        if ev.start <= timestamp <= ev.end:
                            transcript_slot = ev.slot_id
                            break

                if transcript_slot >= 0:
                    # Verify: check if transcript's face is actually speaking
                    dfr = next((d for d in dense_face_results
                                if abs(d.timestamp - timestamp) < 0.3), None)
                    if dfr and dfr.faces and face_registry and face_registry.multi_speaker:
                        transcript_lip = 0.0
                        best_alt_lip = 0.0
                        best_alt_slot = -1
                        for f in dfr.faces:
                            if f.identity_id == transcript_slot:
                                transcript_lip = f.lip_aperture
                            elif f.identity_id >= 0 and f.lip_aperture > best_alt_lip:
                                best_alt_lip = f.lip_aperture
                                best_alt_slot = f.identity_id
                        # Only override if alternative has MUCH higher lip aperture
                        if (best_alt_slot >= 0 and
                                best_alt_lip > 0.05 and
                                transcript_lip < 0.02 and
                                best_alt_lip > transcript_lip * 5):
                            return best_alt_slot, 'lip-override'
                    return transcript_slot, 'transcript'

                if active_speaker_events:
                    sid = get_active_slot_at_time(active_speaker_events, timestamp)
                    if sid >= 0:
                        return sid, 'lip'
                return -1, 'none'

            synthetic_count = 0
            _last_dominant_x = None  # For dominant-subject continuity tracking
            source_counts = {'transcript': 0, 'lip': 0, 'lip-override': 0, 'none': 0}
            for dfr in dense_face_results:
                # Skip timestamps that already have a real scene
                has_real_scene = any(abs(s.timestamp - dfr.timestamp) < 0.5 for s in scenes)
                if has_real_scene:
                    continue
                if not dfr.faces:
                    continue

                # Find which face slot is the active speaker at this timestamp
                slot_id, speaker_source = _get_speaker_at_time(dfr.timestamp)
                source_counts[speaker_source] = source_counts.get(speaker_source, 0) + 1
                chosen_face = None
                identity_matched = False

                if slot_id >= 0:
                    # Find face matching the active speaker's slot
                    for f in dfr.faces:
                        if f.identity_id == slot_id:
                            chosen_face = f
                            identity_matched = True
                            break
                    if not chosen_face:
                        # Identity not found — different camera angle (e.g., 2-person couch
                        # shot where nobody sits at the panel center position).
                        # Prefer face with highest lip aperture (most likely actually speaking)
                        # over nearest-to-slot-center (which picks the wrong person).
                        speaking_faces = [f for f in dfr.faces if f.lip_aperture > 0.03]
                        if speaking_faces:
                            chosen_face = max(speaking_faces,
                                key=lambda f: f.lip_aperture * min(2.0, f.width / 8.0))
                        elif face_registry:
                            slot = face_registry.slot_by_id(slot_id)
                            if slot:
                                chosen_face = min(dfr.faces,
                                    key=lambda f: abs(f.nose_x - slot.x_center))

                # Fallback: largest face (most reliable in multi-speaker panels)
                if not chosen_face and dfr.faces:
                    chosen_face = max(dfr.faces,
                        key=lambda f: f.width * (f.height if hasattr(f, 'height') else f.width))

                if chosen_face:
                    is_closeup = (len(dfr.faces) == 1 and chosen_face.width > 12.0)

                    if _is_continuous:
                        # Dominant-subject tracking: pick the most prominent face
                        # using size + lip motion + centeredness + continuity scoring.
                        # Uses raw nose_x — no slot snapping.
                        dominant = _select_dominant_face(dfr.faces, last_x=_last_dominant_x)
                        if dominant:
                            chosen_face = dominant
                            _last_dominant_x = dominant.nose_x
                        sx_val = int(round(chosen_face.nose_x))
                    elif is_closeup:
                        sx_val = int(round(chosen_face.nose_x))
                    elif slot_id >= 0 and face_registry and identity_matched:
                        # Multi-speaker: identity matched — use assigned slot center (stable)
                        slot = face_registry.slot_by_id(slot_id)
                        if slot:
                            sx_val = int(round(slot.x_center))
                        else:
                            sx_val = int(round(chosen_face.nose_x))
                    elif face_registry and face_registry.multi_speaker:
                        # Multi-speaker: identity NOT matched — use nearest slot center
                        slot = face_registry.nearest_slot(chosen_face.nose_x)
                        if slot:
                            sx_val = int(round(slot.x_center))
                        else:
                            sx_val = int(round(chosen_face.nose_x))
                    else:
                        sx_val = int(round(chosen_face.nose_x))
                    # Build face_positions array for frontend vertical tracking
                    # IMPORTANT: Convert all values to native Python types (int/float/bool)
                    # because numpy.float32/int64 can't be serialized by Pydantic/JSON
                    fp = []
                    for f in dfr.faces:
                        fp.append({
                            "x": int(round(f.nose_x)),
                            "y": int(round(f.nose_y)),
                            "w": int(round(f.width)) if hasattr(f, "width") else 0,
                            "h": int(round(f.height)) if hasattr(f, "height") else 0,
                            "is_speaking": bool((f.identity_id == slot_id) if slot_id >= 0 else False),
                            "identity_id": int(f.identity_id) if hasattr(f, "identity_id") else -1,
                        })
                    scenes.append(SceneDescription(
                        timestamp=float(dfr.timestamp),
                        description="[dense face tracking]",
                        importance_score=5,
                        thumbnail_path="",
                        subject_x=sx_val,
                        active_speaker_x=sx_val if slot_id >= 0 else None,
                        active_slot=slot_id if slot_id >= 0 else None,
                        precise_x=float(round(chosen_face.nose_x, 1)),
                        precise_y=float(round(chosen_face.nose_y, 1)),
                        face_count=int(len(dfr.faces)),
                        face_positions=fp,
                    ))
                    synthetic_count += 1

            logger.info(
                "[%s] Per-second speaker sources: %d transcript-driven, %d lip-based, "
                "%d lip-override, %d fallback",
                job_id, source_counts.get('transcript', 0),
                source_counts.get('lip', 0),
                source_counts.get('lip-override', 0),
                source_counts.get('none', 0),
            )

            # ── Temporal hold: smooth isolated detection gaps only ──
            # Only smooth when a single scene disagrees with BOTH neighbors
            # AND those neighbors have the SAME speaker position. This catches
            # face detection dropouts (momentary tracking loss) without
            # suppressing genuine brief speaker interjections.
            if synthetic_count > 10 and face_registry and face_registry.multi_speaker:
                synth_scenes = [s for s in scenes if s.description == "[dense face tracking]"]
                synth_scenes.sort(key=lambda s: s.timestamp)

                smoothed = 0
                i = 1
                while i < len(synth_scenes) - 1:
                    prev_sx = synth_scenes[i - 1].subject_x
                    curr_sx = synth_scenes[i].subject_x
                    next_sx = synth_scenes[i + 1].subject_x

                    dt = synth_scenes[i + 1].timestamp - synth_scenes[i].timestamp
                    # Only smooth if: neighbors agree with each other (same speaker),
                    # current disagrees (detection gap), and it's a single sample (≤1s)
                    is_isolated_gap = (
                        abs(prev_sx - next_sx) <= 10 and  # neighbors same speaker
                        abs(curr_sx - prev_sx) > 10 and   # current is different
                        dt <= 1.0                          # single sample
                    )
                    if is_isolated_gap:
                        synth_scenes[i].subject_x = prev_sx
                        if synth_scenes[i].active_speaker_x is not None:
                            synth_scenes[i].active_speaker_x = prev_sx
                        if synth_scenes[i].precise_x is not None:
                            synth_scenes[i].precise_x = prev_sx
                        smoothed += 1
                    i += 1

                if smoothed > 0:
                    logger.info(
                        "[%s] Temporal hold: smoothed %d isolated detection gaps (≤1s, neighbors agree)",
                        job_id, smoothed,
                    )

            if synthetic_count > 0:
                scenes.sort(key=lambda s: s.timestamp)
                logger.info(
                    "[%s] Created %d synthetic per-second scenes from dense face + active speaker data "
                    "(total scenes: %d)",
                    job_id, synthetic_count, len(scenes),
                )
                # CRITICAL: Persist the expanded scene list (648 scenes) back to the database
                # so the frontend API receives ALL per-second tracking data, not just the 59 AI scenes.
                # Without this, the frontend only gets 59 scenes and isDense=false, breaking tracking.
                _tracking_mode = "gameplay" if _is_gameplay else ("continuous" if _is_continuous else "multi_cluster")
                await database.update_job_status(
                    job_id, scenes=list(scenes), tracking_mode=_tracking_mode,
                )
                logger.info(
                    "[%s] *** SAVED %d scenes to database (was 59 AI-only, now includes %d dense tracking scenes, mode=%s) ***",
                    job_id, len(scenes), synthetic_count, _tracking_mode,
                )
                await _update_progress(
                    job_id, JobStatus.DETECTING_CLIPS, 67,
                    f"Per-second tracking ready: {len(scenes)} total scenes ({synthetic_count} from face detection + {len(scenes) - synthetic_count} from AI)",
                )
        except Exception as e:
            logger.error(
                "[%s] Per-second scene synthesis save FAILED: %s — "
                "This means the frontend will only receive %d AI scenes instead of %d total. "
                "Subject tracking will be degraded (isDense=false).",
                job_id, e, len([s for s in scenes if s.description != '[dense face tracking]']), len(scenes),
            )
            # Attempt sanitization and retry
            try:
                for scene in scenes:
                    scene.subject_x = int(scene.subject_x)
                    if scene.active_speaker_x is not None:
                        scene.active_speaker_x = int(scene.active_speaker_x)
                    scene.face_count = int(scene.face_count)
                    scene.timestamp = float(scene.timestamp)
                    scene.face_positions = [
                        {k: (bool(v) if k == 'is_speaking'
                             else int(v) if isinstance(v, (int, float)) and not isinstance(v, bool)
                             else v)
                         for k, v in fp.items()}
                        for fp in scene.face_positions
                    ]
                await database.update_job_status(job_id, scenes=list(scenes))
                logger.info("[%s] Per-second scene save SUCCEEDED on retry after sanitization", job_id)
            except Exception as e2:
                logger.error("[%s] Per-second scene save FAILED even after sanitization: %s", job_id, e2)

    # ── Post-scene speaker refinement ──
    # Now that we have active speaker events (the best lip-audio correlation),
    # scene descriptions (vision model data), and dense face data — re-run
    # diarization to refine speaker labels. The initial diarization ran BEFORE
    # scenes and active speaker detection, so it had incomplete data.
    if (face_registry and face_registry.multi_speaker and transcript
            and (active_speaker_events or dense_face_results)):
        try:
            from backend.services.transcription import assign_speakers_with_face_data
            _refine_face_data = dense_face_results if dense_face_results else face_results
            raw_segs = [
                {
                    "start": s.start, "end": s.end, "text": s.text,
                    "words": [{"start": w.start, "end": w.end, "word": w.word} for w in s.words] if s.words else None,
                    "confidence": s.confidence,
                    "avg_logprob": s.avg_logprob,
                    "no_speech_prob": s.no_speech_prob,
                }
                for s in transcript
            ]
            speakers_before = len(set(s.speaker for s in transcript))
            refined = assign_speakers_with_face_data(
                raw_segs, _refine_face_data, face_registry,
                scene_descriptions=scenes,
                active_speaker_events=active_speaker_events,
            )
            speakers_after = len(set(s.speaker for s in refined))

            # Only accept the refined result if it found meaningful speaker diversity
            if speakers_after >= 2:
                transcript = refined
                await database.update_job_status(job_id, transcript=list(transcript))
                speaker_names = {spk: spk for spk in sorted(set(s.speaker for s in transcript))}
                await database.update_job_status(job_id, speaker_names=speaker_names)
                logger.info(
                    "[%s] Post-scene speaker refinement: %d → %d speakers "
                    "(using active_speaker=%d events, scenes=%d, dense_faces=%d)",
                    job_id, speakers_before, speakers_after,
                    len(active_speaker_events),
                    len(scenes) if scenes else 0,
                    len(dense_face_results),
                )
            else:
                logger.info(
                    "[%s] Post-scene speaker refinement: no improvement (%d speakers, keeping original %d)",
                    job_id, speakers_after, speakers_before,
                )
        except Exception as e:
            logger.warning("[%s] Post-scene speaker refinement failed (non-fatal): %s", job_id, e)

    # ── Layout Analysis ──
    # Determine optimal layout mode for the video based on face data + speaker data
    layout_timeline = None
    face_registry_dict = None
    default_layout_mode = "single"
    if face_registry:
        face_registry_dict = face_registry.to_dict()
        if face_registry.multi_speaker:
            # Try camera solver first (per-shot, eliminates cross-cut drift)
            _solver_on = os.environ.get("CLIPAI_CAMERA_SOLVER", settings.CLIPAI_CAMERA_SOLVER).lower() != "off"
            _layout_face_data = dense_face_results if dense_face_results else face_results

            # v2 Phase 11 (Fix 7): if the ReframeSegmenter already produced
            # a content-aware, L1-solved segment timeline, adapt it directly
            # into a LayoutTimeline instead of re-running shot detection +
            # required-regions + camera solver in plan_layout. The
            # second-solve path on the Verzuz/Tank-Tyrese clip was
            # reintroducing the exact 2-Hz stair-stepping the segmenter
            # was built to eliminate (179 opencv-fallback shots →
            # panel short-shot override hard-pinned 165 of them).
            if (_reframe_segments_used and reframe_segments
                    and not _is_gameplay):
                try:
                    from backend.services.layout_engine import (
                        layout_from_reframe_segments,
                    )
                    layout_timeline = layout_from_reframe_segments(
                        reframe_segments=reframe_segments,
                        face_registry=face_registry,
                        source_width=source_width,
                        source_height=source_height,
                        job_id=job_id,
                    )
                    default_layout_mode = layout_timeline.default_mode
                    logger.info(
                        "[%s] [Layout] Built from %d reframe segments "
                        "(skipped plan_layout second-solve)",
                        job_id, len(reframe_segments),
                    )
                except Exception as _lfr_e:
                    logger.warning(
                        "[%s] layout_from_reframe_segments failed "
                        "(%s) — falling back to plan_layout",
                        job_id, _lfr_e,
                    )
                    layout_timeline = None

            if layout_timeline is None and _solver_on and _layout_face_data and not _is_gameplay:
                try:
                    from backend.services.layout_engine import plan_layout

                    # ── Content-type classification for solver tuning ──
                    # IMPORTANT: _clip_content_type must flow to plan_layout
                    # unconditionally so the downstream per-content-type
                    # branches (saliency downrank, speaker weight rules,
                    # CINEMATIC_DIALOGUE / ANIMATION_DIALOGUE routing) can
                    # activate. The CLIPAI_CONTENT_ROUTING env var gates
                    # only the extra conditional object/saliency detection.
                    _clip_content_type = None
                    _solver_objects = None
                    _solver_saliency = None
                    _solver_persons = None
                    _content_routing = os.environ.get("CLIPAI_CONTENT_ROUTING", "off").lower()
                    _enabled_types = os.environ.get("CLIPAI_CONTENT_TYPES_ENABLED", "").lower().split(",")
                    _enabled_types = [t.strip() for t in _enabled_types if t.strip()]

                    if _content_profile:
                        try:
                            from backend.services.content_classifier import classify_clip, ClipContentType
                            # Same priority chain as the job-level
                            # content_type above: user override wins
                            # over auto-classification so the camera
                            # solver routes through the user's pick
                            # instead of whatever the auto-classifier
                            # decided.
                            if _job_content_type is not None and _user_ct_str:
                                _clip_content_type = _job_content_type
                                logger.info(
                                    "[%s] Per-clip content_type=%s (USER OVERRIDE: %r)",
                                    job_id, _clip_content_type.value, _user_ct_str,
                                )
                            else:
                                _clip_content_type = classify_clip(
                                    content_profile=_content_profile,
                                    persistent_regions=_persistent_regions,
                                    frame_faces=_layout_face_data,
                                    shot_count=len(scene_cut_timestamps) if scene_cut_timestamps else 0,
                                    duration=metadata.get("duration", 0),
                                )
                            # Filter by enabled types if specified —
                            # but ONLY when the type came from auto-
                            # classification. A user override is
                            # explicit consent and should bypass the
                            # ``CLIPAI_CONTENT_TYPES_ENABLED`` allow-list.
                            if _enabled_types and not _user_ct_str and _clip_content_type.value not in _enabled_types:
                                logger.info(
                                    "[%s] Content type %s not in enabled list %s — using generic",
                                    job_id, _clip_content_type.value, _enabled_types,
                                )
                                _clip_content_type = ClipContentType.GENERIC

                            # Conditional object/saliency detection by content type
                            # (still gated behind CLIPAI_CONTENT_ROUTING=on).
                            #
                            # Re-uses the universal pass result when present
                            # (``_object_detections`` populated above) so we
                            # don't run YOLO twice over the same frame list —
                            # YOLO is the most expensive non-VLM stage and a
                            # second pass adds tens of seconds on long jobs.
                            if _content_routing == "on" and _clip_content_type in (
                                ClipContentType.ANIMATION, ClipContentType.ANIMATION_DIALOGUE,
                                ClipContentType.MUSIC_VIDEO, ClipContentType.GENERIC,
                            ):
                                if _object_detections:
                                    _solver_objects = _object_detections
                                    logger.info(
                                        "[%s] Content-routed object detection: re-using %d objects "
                                        "from universal YOLO pass (no second run)",
                                        job_id, len(_solver_objects),
                                    )
                                else:
                                    try:
                                        from backend.services.object_detector import detect_objects_in_frames
                                        _frame_list_obj = [(f.timestamp, f.path) for f in frames]
                                        _solver_objects = detect_objects_in_frames(_frame_list_obj, face_results)
                                        # Cache for any further consumers in this job.
                                        _object_detections = _solver_objects
                                        logger.info(
                                            "[%s] Content-routed object detection: %d objects",
                                            job_id, len(_solver_objects),
                                        )
                                    except Exception as _oe:
                                        logger.warning(
                                            "[%s] Content-routed object detection failed: %s",
                                            job_id, _oe,
                                        )

                            # Saliency detection for the solver path.
                            #
                            # v3 wiring: animation modes need saliency
                            # *unconditionally* — not gated behind
                            # CLIPAI_CONTENT_ROUTING — so the SaliencyFallback
                            # adapter in build_required_regions has data to
                            # convert and the camera solver has anchors for
                            # faceless action beats. Without this the anime
                            # crop drifts onto background motion. Other
                            # content types still respect the env-var gate.
                            _animation_modes = (
                                ClipContentType.ANIMATION,
                                ClipContentType.ANIMATION_DIALOGUE,
                            )
                            _saliency_for_animation = _clip_content_type in _animation_modes
                            _saliency_routed = _content_routing == "on" and _clip_content_type in (
                                ClipContentType.ANIMATION, ClipContentType.ANIMATION_DIALOGUE,
                                ClipContentType.MUSIC_VIDEO, ClipContentType.GAMEPLAY,
                            )
                            if _saliency_for_animation or _saliency_routed:
                                try:
                                    from backend.services.saliency_tracker import track_saliency_in_frames
                                    _frame_list_sal = [(f.timestamp, f.path) for f in frames]
                                    _solver_saliency = track_saliency_in_frames(
                                        _frame_list_sal, face_results,
                                        persistent_regions=_persistent_regions,
                                    )
                                    logger.info(
                                        "[%s] Content-routed saliency detection: "
                                        "%d regions (animation_unconditional=%s)",
                                        job_id, len(_solver_saliency),
                                        _saliency_for_animation,
                                    )
                                except Exception as _se:
                                    logger.warning("[%s] Content-routed saliency detection failed: %s", job_id, _se)

                            # ── v4: Person body detection ──
                            # Faceless action beats and back-turned characters
                            # need a real subject for the camera solver to
                            # anchor on. PersonDetector reuses the existing
                            # YOLOv8n / MobileNet-SSD pipeline (no extra
                            # model load) and emits typed PersonRegion
                            # objects with has_face flagged for
                            # de-prioritization in the AttentionAnchor chain.
                            # Run unconditionally for animation modes
                            # (anime / cartoon dialogue) where the win is
                            # largest. Other types still respect the env-
                            # var gate when CLIPAI_CONTENT_ROUTING=on.
                            _persons_for_animation = _clip_content_type in _animation_modes
                            _persons_routed = _content_routing == "on" and _clip_content_type in (
                                ClipContentType.ANIMATION,
                                ClipContentType.ANIMATION_DIALOGUE,
                                ClipContentType.CINEMATIC_DIALOGUE,
                                ClipContentType.GENERIC,
                            )
                            if _persons_for_animation or _persons_routed:
                                try:
                                    from backend.services.person_detector import detect_persons_in_frames
                                    _frame_list_per = [(f.timestamp, f.path) for f in frames]
                                    _solver_persons = detect_persons_in_frames(
                                        _frame_list_per,
                                        face_results=face_results,
                                        job_id=job_id,
                                    )
                                except Exception as _pe:
                                    logger.warning("[%s] PersonDetector failed (non-fatal): %s", job_id, _pe)

                        except Exception as _ct_err:
                            logger.warning("[%s] Content classification for solver failed: %s", job_id, _ct_err)

                    layout_timeline = plan_layout(
                        video_path=video_path,
                        frame_faces=_layout_face_data,
                        face_registry=face_registry,
                        active_speaker_events=active_speaker_events,
                        source_width=metadata.get("width", 1920),
                        source_height=metadata.get("height", 1080),
                        scene_descriptions=scenes,
                        video_duration=metadata.get("duration", 0),
                        job_id=job_id,
                        content_type=_clip_content_type,
                        persistent_regions=_persistent_regions,
                        frame_objects=_solver_objects,
                        frame_saliency=_solver_saliency,
                        frame_persons=_solver_persons,
                    )
                    default_layout_mode = layout_timeline.default_mode
                    logger.info(
                        "[%s] [Layout+Solver] Video layout analysis: default=%s, %d segments, "
                        "%d layout changes, content_type=%s",
                        job_id, layout_timeline.default_mode,
                        len(layout_timeline.segments),
                        layout_timeline.total_layout_changes,
                        _clip_content_type.value if _clip_content_type else "none",
                    )
                except Exception as e:
                    logger.warning("[%s] Camera solver layout failed, falling back to legacy: %s", job_id, e)
                    layout_timeline = None

            # Legacy fallback (also runs when solver is off or failed)
            if layout_timeline is None:
                try:
                    from backend.services.layout_engine import build_layout_timeline
                    layout_timeline = build_layout_timeline(
                        face_results=_layout_face_data,
                        face_registry=face_registry,
                        active_speaker_events=active_speaker_events,
                        scene_descriptions=scenes,
                        clip_start=0,
                        clip_end=metadata.get("duration", 0),
                    )
                    default_layout_mode = layout_timeline.default_mode
                    logger.info(
                        "[%s] [Layout] Video layout analysis: default=%s, %d segments, %d layout changes",
                        job_id, layout_timeline.default_mode,
                        len(layout_timeline.segments),
                        layout_timeline.total_layout_changes,
                    )
                except Exception as e:
                    logger.warning("[%s] Layout analysis failed (non-fatal): %s", job_id, e)
        else:
            logger.info("[%s] [Layout] Single-speaker video — using SINGLE layout", job_id)

    # Save layout data to job
    try:
        layout_update = {
            "face_registry_data": face_registry_dict,
            "default_layout_mode": str(default_layout_mode),  # Ensure plain string, not LayoutMode enum
        }
        if layout_timeline:
            layout_update["layout_timeline"] = [s.to_dict() for s in layout_timeline.segments]
        await database.update_job_status(job_id, **layout_update)
    except Exception as e:
        logger.warning("[%s] Failed to save layout data (non-fatal): %s", job_id, e)

    # Store dense tracking summary
    if dense_face_results:
        try:
            dense_summary = {
                "total_frames": len(dense_face_results),
                "frames_with_faces": sum(1 for r in dense_face_results if r.faces),
                "frames_with_embeddings": sum(
                    1 for r in dense_face_results
                    for f in r.faces if f.identity_embedding is not None
                ),
                "sample_rate": settings.DENSE_FACE_SAMPLE_RATE,
                "coverage_seconds": len(dense_face_results) * settings.DENSE_FACE_SAMPLE_RATE,
            }
            await database.update_job_status(job_id, dense_tracking_summary=dense_summary)
        except Exception as e:
            logger.debug("[%s] Failed to store dense tracking summary: %s", job_id, e)

    # ── Dense subject-position track ──
    # Fuse every spatial signal we have (dense faces, saliency, YOLO,
    # VLM scenes) into a frame-dense ``(t, x, y)`` track. The frontend
    # preview and the FFmpeg export both consume this so the 9:16 / 1:1
    # reframe stays locked on the subject on every frame instead of
    # drifting to center between sparse VLM keyframes.
    if not _is_gameplay:
        try:
            _track_duration = float(metadata.get("duration", 0) or 0)
            if _track_duration > 0:
                subject_track = build_subject_track(
                    duration_s=_track_duration,
                    dense_face_results=dense_face_results,
                    saliency_regions=_saliency_regions,
                    object_detections=_object_detections,
                    scenes=scenes,
                    sample_rate_hz=max(1.0, 1.0 / float(settings.DENSE_FACE_SAMPLE_RATE or 0.5)),
                )
                if subject_track:
                    await database.update_job_status(job_id, subject_track=subject_track)
                    logger.info(
                        "[%s] Dense subject track stored (%d samples).",
                        job_id, len(subject_track),
                    )
        except Exception as _track_err:
            logger.warning(
                "[%s] Dense subject track build failed (non-fatal): %s",
                job_id, _track_err,
            )

    # ── Pipeline health check: detect total failure ──
    real_scenes = [s for s in scenes if not is_synthetic_scene(s)]

    if len(transcript) == 0 and len(real_scenes) == 0:
        logger.error(
            "[%s] TOTAL PIPELINE FAILURE: 0 transcript segments AND 0 real scene descriptions. "
            "Possible causes: (1) Whisper OOM on GPU, (2) Ollama vision model crashed, "
            "(3) Audio extraction failed. Check container logs for errors.",
            job_id,
        )
        await broadcast_ws(job_id, {
            "type": "warning",
            "message": (
                "Analysis produced no usable results. Whisper transcription and "
                "visual analysis both failed — likely due to GPU memory constraints. "
                "Try: (1) Use Whisper 'small' instead of 'medium', "
                "(2) Restart the Ollama container, "
                "(3) Check the Logs page for detailed errors."
            ),
        })
    elif len(transcript) == 0 and audio_duration > 10:
        logger.error(
            "[%s] Whisper returned 0 segments for %.0fs audio. "
            "Model=%s, language=%s. "
            "This usually means CUDA OOM on GPU.",
            job_id, audio_duration,
            settings.WHISPER_MODEL,
            job.language or "auto",
        )
        await broadcast_ws(job_id, {
            "type": "warning",
            "message": (
                f"Transcription produced 0 segments for {int(audio_duration / 60)} min audio. "
                f"Whisper '{settings.WHISPER_MODEL}' may have crashed on your GPU. "
                f"Try switching to 'small' model in Settings."
            ),
        })

    await _update_progress(
        job_id, JobStatus.ANALYZING_SCENES, 63,
        f"Transcribed {len(transcript)} segments ({speaker_count} speakers) + "
        f"{len(real_scenes) if real_scenes != scenes else len(scenes)} scenes via {scenes_provider}",
    )

    # ── Steps 5+6 — Summary + audio/hot-zone analysis, THEN clip detection ──
    # Summary runs first so clip detection can use content context.
    # Audio energy + hot zone scoring run concurrently with summary (no AI needed).
    cancel_check()

    # ── CRITICAL: Unload vision model before text summarization ──
    # On GTX 1650 (3.6GB VRAM), the vision model (~1.1GB) and text model (~2.2GB)
    # cannot coexist. If OLLAMA_KEEP_ALIVE keeps the vision model resident,
    # loading the text model causes cudaMalloc OOM → sticky CPU fallback.
    # Explicitly unload ALL models so the text model gets full GPU access.
    if is_ollama_primary and _primary_provider:
        try:
            await _primary_provider.clear_vram()
            logger.info("[%s] Vision model unload sent — waiting for VRAM release", job_id)
            # Poll until Ollama confirms no models loaded (VRAM takes 5-10s to free on GTX 1650)
            for _vram_wait in range(12):
                await asyncio.sleep(1)
                try:
                    async with httpx.AsyncClient(timeout=5) as _hc:
                        _ps = await _hc.get(f"{settings.OLLAMA_HOST}/api/ps")
                        if _ps.status_code == 200 and not _ps.json().get("models", []):
                            logger.info("[%s] Ollama reports no models after %ds — waiting 5s for CUDA driver + runners to settle", job_id, _vram_wait + 1)
                            await asyncio.sleep(5)  # Extra delay: CUDA driver reclaim + Ollama runner cleanup
                            break
                except Exception:
                    pass
            else:
                logger.warning("[%s] Models may still be unloading after 12s wait", job_id)
            # Reset force_cpu flag so text model tries GPU
            if hasattr(_primary_provider, '_force_cpu'):
                _primary_provider._force_cpu = False
            # Log GPU status for diagnostics
            if hasattr(_primary_provider, 'log_gpu_status'):
                await _primary_provider.log_gpu_status()
            # Verify GPU is not poisoned before loading text model
            if hasattr(_primary_provider, 'verify_gpu_health'):
                gpu_ok = await _primary_provider.verify_gpu_health()
                if not gpu_ok and hasattr(_primary_provider, 'reset_gpu_scheduler'):
                    logger.warning("[%s] GPU scheduler poisoned — attempting reset before text summarization", job_id)
                    reset_ok = await _primary_provider.reset_gpu_scheduler()
                    if reset_ok:
                        logger.info("[%s] GPU scheduler reset successful — text model will load on GPU", job_id)
                    else:
                        logger.warning("[%s] GPU scheduler reset failed — text model will run on CPU", job_id)
        except Exception as e:
            logger.warning("[%s] Failed to unload vision model before summary: %s", job_id, e)

    await _update_progress(
        job_id, JobStatus.GENERATING_SUMMARY, 65,
        f"Generating video summary...{_pipeline_eta(65)}",
    )

    # CRITICAL: Reset circuit breaker before critical AI operations.
    orchestrator.reset_circuit_breaker()

    # Launch summary generation as a task
    _summary_start = _time.monotonic()
    summary_task = asyncio.create_task(
        orchestrator.generate_summary(transcript, scenes, job_id, tier=tier)
    )

    # While summary runs, do audio energy analysis + hot zone scoring (no AI)
    audio_energy_text = ""
    audio_moments = []
    hot_zone_text = ""
    try:
        audio_moments = await analyze_audio_energy(audio_path)
        audio_energy_text = format_audio_energy_map(audio_moments)
        if audio_energy_text:
            logger.info("[%s] Audio energy analysis: %d spikes detected", job_id, len(audio_moments))
    except Exception as e:
        logger.warning("[%s] Audio energy analysis failed (non-critical): %s", job_id, e)

    # ── Filler word detection (instant, no AI calls) ──
    filler_events: list[dict] = []
    if transcript:
        try:
            from backend.services.transcript_utils import detect_filler_words
            detected_lang = job.language or "en"
            filler_events = detect_filler_words(transcript, language=detected_lang)
            if filler_events:
                filler_count = len([e for e in filler_events if e["type"] == "filler"])
                dead_air_count = len([e for e in filler_events if e["type"] == "dead_air"])
                logger.info(
                    "[%s] Filler detection: %d filler words, %d dead air pauses",
                    job_id, filler_count, dead_air_count,
                )
        except Exception as e:
            logger.warning("[%s] Filler detection failed (non-fatal): %s", job_id, e)

    # Hot zone pre-scoring (instant, no AI calls)
    # Phase 3 — pass the trend matcher in if the flag is on. The matcher
    # is built lazily so the import / lexicon read only fires when the
    # operator opts in.
    _hot_zone_trend_matcher = None
    try:
        from backend.services.trend_matcher import (
            build_default_trend_matcher, trend_matcher_enabled,
        )
        if trend_matcher_enabled():
            _hot_zone_trend_matcher = build_default_trend_matcher()
    except Exception as _tm_err:
        logger.warning("[%s] Trend matcher (hot zones) init failed: %s", job_id, _tm_err)

    from backend.services.hot_zone_scorer import score_hot_zones, format_hot_zones_for_prompt
    hot_zones = score_hot_zones(
        transcript, scenes, audio_moments, metadata["duration"],
        filler_events=filler_events,
        trend_matcher=_hot_zone_trend_matcher,
    )
    hot_zone_text = format_hot_zones_for_prompt(hot_zones, top_n=tier.hot_zone_top_n)
    logger.info(
        "[%s] Hot zone scoring: %d zones, top score=%.1f",
        job_id, len(hot_zones),
        hot_zones[0].composite_score if hot_zones else 0,
    )

    # Now await summary
    try:
        summary, summary_provider = await summary_task
    except CancelledError:
        raise
    except AllProvidersFailedError:
        logger.warning("[%s] All providers failed for summary — retrying in 5s", job_id)
        orchestrator.reset_circuit_breaker()
        await asyncio.sleep(5)
        try:
            summary, summary_provider = await orchestrator.generate_summary(
                transcript, scenes, job_id, tier=tier,
            )
        except Exception:
            logger.exception("[%s] Summary generation retry also failed, building from transcript", job_id)
            fb = build_summary_from_transcript(transcript, scenes)
            summary = VideoSummary(**fb)
            summary_provider = "none"
    except Exception as e:
        logger.exception("[%s] Summary generation failed, building from transcript", job_id)
        fb = build_summary_from_transcript(transcript, scenes)
        summary = VideoSummary(**fb)
        summary_provider = "none"

    # Record summary phase timing
    if _summary_start:
        _phase_timings["summary"] = _time.monotonic() - _summary_start

    await _update_progress(
        job_id, JobStatus.GENERATING_SUMMARY, 75,
        f"Summary generated via {summary_provider} — now detecting viral clips...{_pipeline_eta(75)}",
    )

    # Build summary context string for clip detection
    summary_text = summary.overview
    if summary.key_topics:
        summary_text += f"\nKey topics: {', '.join(summary.key_topics)}"
    if summary.content_category:
        summary_text += f"\nCategory: {summary.content_category}"
    if summary.tone:
        summary_text += f"\nTone: {summary.tone}"
    if audio_energy_text:
        summary_text += audio_energy_text
    if hot_zone_text:
        summary_text += hot_zone_text

    # Step 6: Clip detection with summary context
    _log_gpu_memory(job_id, "before clip detection")
    cancel_check()

    # ── Early exit: skip clip detection when there's no data to analyze ──
    # Without transcript, the AI has nothing to find clips in. Running 46 windows
    # of empty prompts wastes hours of CPU time for guaranteed 0 clips.
    # This saved 224 minutes in production on a 113-min video where Whisper crashed.
    real_scenes = [s for s in scenes if not is_synthetic_scene(s)]
    if not transcript and len(real_scenes) < 10:
        logger.warning(
            "[%s] Skipping clip detection: 0 transcript segments, %d useful scenes. "
            "Nothing for the AI to analyze. Check Whisper logs for transcription errors.",
            job_id, len(real_scenes),
        )
        clips = []
        clips_provider = "skipped (no transcript)"
        await _update_progress(
            job_id, JobStatus.DETECTING_CLIPS, 95,
            "Clip detection skipped — no transcript data available. "
            "Check logs for Whisper errors.",
        )
    else:

        # Reset again before clip detection — summary generation may have
        # had transient failures that shouldn't block clip detection.
        orchestrator.reset_circuit_breaker()

        # ── Phase 2 (OpusClip parity gap): job-level content type ──
        # The per-clip content type is computed later (line ~3944) for
        # camera-solver routing. For genre-aware clip DETECTION we need
        # one job-level type ahead of time.
        #
        # Priority:
        #   1. **User override** from the upload form (always wins).
        #   2. Auto-classify from the content profile.
        #   3. GENERIC fallback when neither signal is available.
        # Without this, a user who explicitly picked ``anime`` /
        # ``music_video`` / ``sports`` could see clip detection
        # ignore their choice in favor of whatever the auto-classifier
        # decided was the closest match.
        _job_content_type = None
        try:
            from backend.services.content_classifier import (
                classify_clip as _classify_clip_for_detection, ClipContentType,
            )
            # 1. Map the user's UI override (already lowercased above
            # into ``_user_ct_str``) to a ClipContentType enum value.
            _OVERRIDE_TO_CONTENT_TYPE = {
                # Animation / anime
                "anime": "ANIMATION",
                "anime_dialogue": "ANIMATION_DIALOGUE",
                "animation": "ANIMATION",
                "animation_dialogue": "ANIMATION_DIALOGUE",
                "cartoon": "ANIMATION",
                "cartoon_dialogue": "ANIMATION_DIALOGUE",
                # Music / performance
                "music": "MUSIC_VIDEO",
                "music_video": "MUSIC_VIDEO",
                "concert": "MUSIC_VIDEO",
                "performance": "MUSIC_VIDEO",
                # Sports
                "sports": "SPORTS",
                "sports_basketball": "SPORTS",
                "sports_racing": "SPORTS",
                # Talking-head genres
                "podcast": "PODCAST",
                "interview": "PODCAST",
                "vlog": "VLOG",
                "talking_head": "VLOG",
                "tutorial": "EDUCATIONAL",
                "documentary": "DOCUMENTARY",
                "narrative": "NARRATIVE",
                # Gameplay (any subtype)
                "gameplay": "GAMEPLAY",
                "gameplay_fps": "GAMEPLAY",
                "gameplay_tps": "GAMEPLAY",
                "gameplay_moba": "GAMEPLAY",
                "gameplay_racing": "GAMEPLAY",
                "stream": "GAMEPLAY",
            }
            if _user_ct_str in _OVERRIDE_TO_CONTENT_TYPE:
                _ct_member_name = _OVERRIDE_TO_CONTENT_TYPE[_user_ct_str]
                _job_content_type = getattr(
                    ClipContentType, _ct_member_name, ClipContentType.GENERIC,
                )
                logger.info(
                    "[%s] Clip detection content_type=%s (USER OVERRIDE: %r)",
                    job_id, _job_content_type.value, _user_ct_str,
                )
                _record_pipeline_warning(
                    job_id,
                    f"info: Honoring user content type override = {_user_ct_str!r} "
                    f"(routes clip detection through {_job_content_type.value})",
                )
            elif _content_profile is not None:
                _job_content_type = _classify_clip_for_detection(
                    content_profile=_content_profile,
                    persistent_regions=None,
                    frame_faces=None,
                    shot_count=len(scene_cut_timestamps) if scene_cut_timestamps else 0,
                    duration=metadata.get("duration", 0),
                )
                logger.info(
                    "[%s] Clip detection content_type=%s (auto-classified from profile=%s)",
                    job_id,
                    _job_content_type.value if _job_content_type else "none",
                    getattr(_content_profile, "content_type", "unknown"),
                )
            else:
                _job_content_type = ClipContentType.GENERIC
                logger.info("[%s] No content profile — using GENERIC for clip detection", job_id)
        except Exception as _ct_err:
            logger.warning("[%s] Failed to compute job content_type (non-fatal): %s", job_id, _ct_err)
            _job_content_type = None

        # ── Phase 3 — Trend matcher (off by default, opt-in via flag) ──
        _trend_context_text: str | None = None
        _trend_matcher = None
        try:
            from backend.services.trend_matcher import (
                build_default_trend_matcher, format_trend_context,
                trend_matcher_enabled,
            )
            if trend_matcher_enabled():
                _trend_matcher = build_default_trend_matcher(
                    emphasis_keywords=None,  # filled in after detect_emphasis_words below
                )
                _trend_context_text = format_trend_context(
                    _trend_matcher, transcript,
                    content_type=_job_content_type,
                )
                if _trend_context_text:
                    logger.info(
                        "[%s] Trend context built: %d chars", job_id, len(_trend_context_text),
                    )
        except Exception as _trend_err:
            logger.warning("[%s] Trend matcher init failed (non-fatal): %s", job_id, _trend_err)

        # ── Phase 5 — Sentiment timeline from audio_moments ──
        _sentiment_timeline_text: str | None = None
        try:
            from backend.services.audio_analyzer import format_sentiment_timeline
            _sentiment_timeline_text = format_sentiment_timeline(audio_moments)
        except Exception:
            _sentiment_timeline_text = None

        # ── Phase 6 — Chapter segmentation (opt-in via flag) ──
        _chapters = None
        try:
            from backend.services.chapter_segmenter import (
                segment_chapters, chapter_segmentation_enabled,
            )
            if (
                chapter_segmentation_enabled()
                and metadata.get("duration", 0) >= 180  # short videos: skip
            ):
                _chapters = segment_chapters(
                    transcript, scenes, audio_moments,
                )
                logger.info(
                    "[%s] Chapter segmentation produced %d chapters",
                    job_id, len(_chapters),
                )
        except Exception as _ch_err:
            logger.warning("[%s] Chapter segmentation failed (non-fatal): %s", job_id, _ch_err)
            _chapters = None

        _clips_start = _time.monotonic()
        _clips_phase_start[0] = _clips_start  # For phase-aware ETA
        _last_eta_value[0] = 0.0  # Reset ETA smoothing for new phase

        # Scale clip count with video duration — use tier if available
        dynamic_clip_count = tier.max_clip_candidates

        # Cap clip count for CPU-only processing to avoid excessive stalls
        if is_ollama_primary and _primary_provider and hasattr(_primary_provider, 'is_gpu_available'):
            try:
                _gpu_avail = await _primary_provider.is_gpu_available()
                if not _gpu_avail and dynamic_clip_count > 15:
                    logger.info(
                        "[%s] CPU-only mode: capping clip count from %d to 15",
                        job_id, dynamic_clip_count,
                    )
                    dynamic_clip_count = 15
            except Exception:
                pass

        logger.info(
            "[%s] Dynamic clip count: %d (%.0f min video, default=%d)",
            job_id, dynamic_clip_count, vid_minutes, settings.MAX_CLIP_CANDIDATES,
        )

        clip_detection_task = None
        _clips_max_pct = [78]  # Track highest progress seen (never go backward)

        async def _clip_progress(phase: str, info: dict):
            """Progress callback from multi-pass clip detection."""
            elapsed = int(_time.monotonic() - _clips_start)

            if phase == "pass1_start":
                n_windows = info.get("windows", 1)
                msg = f"Pass 1: scanning {n_windows} window{'s' if n_windows > 1 else ''}..."
                pct = 78
            elif phase == "pass1_window_done":
                idx = info.get("window_idx", 1)
                total = info.get("window_total", 1)
                clips_so_far = info.get("clips_so_far", 0)
                msg = f"Pass 1: window {idx}/{total} done ({clips_so_far} clips so far)..."
                pct = 78 + int((idx / max(total, 1)) * 12)  # 78-90%
            elif phase == "pass1_done":
                n_clips = info.get("clips", 0)
                msg = f"Pass 1 found {n_clips} clips — checking coverage..."
                pct = 90
            elif phase == "pass2_start":
                n_gaps = info.get("gaps", 0)
                msg = f"Pass 2: sweeping {n_gaps} gap{'s' if n_gaps != 1 else ''} for hidden moments..."
                pct = 91
            elif phase == "pass2_gap":
                idx = info.get("gap_idx", 1)
                total = info.get("gap_total", 1)
                start = info.get("start", 0)
                end = info.get("end", 0)
                msg = f"Pass 2: scanning gap {idx}/{total} ({start:.0f}-{end:.0f}s)..."
                pct = 91 + int((idx / max(total, 1)) * 3)  # 91-94%
            elif phase == "pass3_merge":
                raw = info.get("raw", 0)
                msg = f"Merging {raw} candidates..."
                pct = 94
            else:
                msg = f"Identifying viral moments... ({elapsed}s elapsed)"
                pct = min(94, 78 + elapsed // 15)

            # Never go backward
            pct = max(pct, _clips_max_pct[0])
            _clips_max_pct[0] = pct

            await _update_progress(
                job_id, JobStatus.DETECTING_CLIPS, min(95, pct),
                f"{msg}{_pipeline_eta(min(95, pct))}",
            )

        async def _clips_heartbeat():
            await asyncio.sleep(10)
            while True:
                try:
                    cancel_check()
                except Exception:
                    if clip_detection_task and not clip_detection_task.done():
                        clip_detection_task.cancel()
                    raise
                elapsed = int(_time.monotonic() - _clips_start)
                hb_pct = min(94, 78 + elapsed // 15)
                # Only update if heartbeat would ADVANCE progress (never regress)
                if hb_pct > _clips_max_pct[0]:
                    _clips_max_pct[0] = hb_pct
                    await _update_progress(
                        job_id, JobStatus.DETECTING_CLIPS, hb_pct,
                        f"Identifying viral moments... ({elapsed}s elapsed){_pipeline_eta(hb_pct)}",
                    )
                await asyncio.sleep(8)

        heartbeat_task = asyncio.create_task(_clips_heartbeat())
        try:
            try:
                clip_detection_task = asyncio.ensure_future(
                    orchestrator.detect_viral_clips(
                        transcript, scenes, metadata["duration"], job_id,
                        video_summary=summary_text,
                        hot_zones=hot_zones,
                        progress_callback=_clip_progress,
                        clip_count=dynamic_clip_count,
                        tier=tier,
                        content_type=_job_content_type,
                        chapters=_chapters,
                        trend_context=_trend_context_text,
                        sentiment_timeline=_sentiment_timeline_text,
                        # Task 6 — feed the visual verifier the
                        # extracted frames + cancellation primitive.
                        # apply_visual_verification no-ops cleanly
                        # when frames is empty / vision_provider is
                        # None / the feature flag is off.
                        frames=frames,
                        cancel_check=cancel_check,
                    )
                )
                clips, clips_provider = await asyncio.wait_for(
                    clip_detection_task,
                    timeout=_SUMMARY_CLIP_TIMEOUT,
                )
            except AllProvidersFailedError:
                # All providers failed on first attempt — wait briefly for any
                # transient rate limits to clear and retry once.
                logger.warning("[%s] All providers failed for clip detection — retrying in 10s", job_id)
                orchestrator.reset_circuit_breaker()
                await asyncio.sleep(10)
                try:
                    clips, clips_provider = await asyncio.wait_for(
                        orchestrator.detect_viral_clips(
                            transcript, scenes, metadata["duration"], job_id,
                            video_summary=summary_text,
                            hot_zones=hot_zones,
                            progress_callback=_clip_progress,
                            clip_count=dynamic_clip_count,
                            tier=tier,
                            content_type=_job_content_type,
                            chapters=_chapters,
                            trend_context=_trend_context_text,
                            sentiment_timeline=_sentiment_timeline_text,
                            frames=frames,
                            cancel_check=cancel_check,
                        ),
                        timeout=_SUMMARY_CLIP_TIMEOUT,
                    )
                except Exception:
                    logger.exception("[%s] Clip detection retry also failed", job_id)
                    clips = []
                    clips_provider = "none"
        except asyncio.TimeoutError:
            logger.error("[%s] Clip detection timed out", job_id)
            clips = []
            clips_provider = "none"
        except CancelledError:
            raise
        except Exception as e:
            logger.exception("[%s] Clip detection failed", job_id)
            clips = []
            clips_provider = "none"
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except (asyncio.CancelledError, CancelledError):
                pass

    await _update_progress(
        job_id, JobStatus.DETECTING_CLIPS, 95,
        f"Summary via {summary_provider} + {len(clips)} clips via {clips_provider}",
    )

    # Snap clip boundaries to word-level timestamps for clean cuts
    if clips and transcript:
        clips = snap_all_clips(clips, transcript)

    # Save both results
    job = await database.load_job(job_id)
    provider_used = job.provider_used if job else {}
    provider_used["summary"] = summary_provider
    provider_used["clips"] = clips_provider

    # Ensure clip IDs are unique and sequential (AI may return duplicates)
    for idx, clip in enumerate(clips, start=1):
        clip.id = idx

    # ── Emphasis keyword detection ──
    emphasis_keywords: list[str] = []
    if transcript:
        try:
            from backend.services.transcript_utils import detect_emphasis_words
            emphasis_keywords = sorted(detect_emphasis_words(
                transcript, video_summary=summary_text, max_keywords=20,
            ))
        except Exception as e:
            logger.warning("[%s] Emphasis keyword detection failed (non-fatal): %s", job_id, e)

    # Persist the clip-detection context so the regenerate route
    # (POST /api/jobs/{job_id}/generate-clips) can reuse the same inputs
    # the initial run used. See Bug 4 in the clip-focus audit.
    _persisted_hot_zones: list[dict] = []
    try:
        for z in (hot_zones or []):
            _persisted_hot_zones.append({
                "start": float(getattr(z, "start", 0)),
                "end": float(getattr(z, "end", 0)),
                "composite_score": float(getattr(z, "composite_score", 0)),
                "audio_score": float(getattr(z, "audio_score", 0)),
                "transcript_score": float(getattr(z, "transcript_score", 0)),
                "scene_score": float(getattr(z, "scene_score", 0)),
                "speaker_score": float(getattr(z, "speaker_score", 0)),
                "signals": list(getattr(z, "signals", []) or []),
            })
    except Exception:
        _persisted_hot_zones = []
    _persisted_chapters: list[dict] = []
    try:
        for ch in (_chapters or []):
            if hasattr(ch, "to_dict"):
                _persisted_chapters.append(ch.to_dict())
            else:
                _persisted_chapters.append(dict(ch) if isinstance(ch, dict) else {})
    except Exception:
        _persisted_chapters = []
    _ct_value = ""
    try:
        if _job_content_type is not None:
            _ct_value = _job_content_type.value if hasattr(_job_content_type, "value") else str(_job_content_type)
    except Exception:
        _ct_value = ""

    await database.update_job_status(
        job_id,
        summary=summary,
        clips=list(clips),
        provider_used=provider_used,
        filler_events=filler_events,
        emphasis_keywords=emphasis_keywords,
        hot_zones=_persisted_hot_zones,
        chapters=_persisted_chapters,
        trend_context=_trend_context_text,
        sentiment_timeline=_sentiment_timeline_text,
        clip_content_type=_ct_value,
    )
    await _update_progress(
        job_id, JobStatus.DETECTING_CLIPS, 95,
        f"Summary via {summary_provider} + {len(clips)} clips via {clips_provider}",
    )

    # Step 7 — Save Results
    total_elapsed = round(_pipeline_elapsed(), 1)
    if total_elapsed < 60:
        dur_str = f"{int(total_elapsed)}s"
    else:
        m, s = divmod(int(total_elapsed), 60)
        dur_str = f"{m}m {s}s"

    await _update_progress(
        job_id, JobStatus.DETECTING_CLIPS, 98,
        f"Saving results — {len(transcript)} transcript segments, {len(scenes)} scenes, {len(clips)} clips",
    )
    # Estimate cost from AI provider token usage
    estimated_cost = orchestrator.estimate_cost()
    total_tokens = orchestrator.get_total_tokens()
    if total_tokens > 0:
        logger.info("[%s] Total AI tokens used: %d, estimated cost: $%.6f", job_id, total_tokens, estimated_cost)

    completion_msg = (
        f"Analysis complete in {dur_str} — "
        f"{len(transcript)} segments, {len(scenes)} scenes, {len(clips)} clips"
    )
    # ── Thumbnail extraction (non-fatal) ──
    # Extract a representative thumbnail frame for rich preview unfurls (OG/Twitter Card).
    try:
        from backend.services.thumbnail_extractor import extract_thumbnail
        _video_dur = metadata.get("duration", 0) if metadata else 0
        _reframe_segs = reframe_segments if 'reframe_segments' in dir() else None
        thumb_path = extract_thumbnail(
            job_id=job_id,
            source_video_path=video_path,
            video_duration=_video_dur,
            scenes=scenes,
            reframe_segments=_reframe_segs,
        )
        if thumb_path:
            await database.update_job_thumbnail(job_id, str(thumb_path))
            logger.info("[%s] thumbnail saved: %s", job_id, thumb_path)
    except Exception as e:
        logger.warning("[%s] thumbnail step failed (non-fatal): %s", job_id, e)

    # FINAL SAVE: ensure ALL scenes (including 589+ synthetic per-second tracking scenes)
    # are persisted. This is the authoritative save — if the earlier save at synthetic
    # creation time was skipped or failed, this catches it.
    logger.info(
        "[%s] Final save: %d total scenes (%d with description '[dense face tracking]')",
        job_id, len(scenes),
        sum(1 for s in scenes if getattr(s, 'description', '') == '[dense face tracking]'),
    )
    # Drain the per-stage timings + soft warnings recorded during the
    # run and persist them on the JobResult so the UI can show "where
    # time went" without a separate metrics store.
    _final_timings, _final_warnings = _drain_pipeline_telemetry(job_id)
    await database.update_job_status(
        job_id,
        status=JobStatus.COMPLETE,
        progress=100,
        progress_message=completion_msg,
        scenes=list(scenes),
        analysis_duration_seconds=total_elapsed,
        estimated_cost_usd=estimated_cost if estimated_cost > 0 else None,
        timings=_final_timings,
        pipeline_warnings=_final_warnings,
    )
    await broadcast_ws(job_id, {
        "type": "complete",
        "progress": 100,
        "message": completion_msg,
    })
    logger.info(
        "[%s] %s (summary=%s, scenes=%s, clips=%s)",
        job_id, completion_msg,
        summary_provider, scenes_provider, clips_provider,
    )

    # Launch background post-processing (transcript polishing + subtitle translation)
    # These improve quality but don't affect clip detection — run after COMPLETE.
    if transcript:
        asyncio.create_task(
            _background_post_processing(job_id, transcript, orchestrator, job)
        )
