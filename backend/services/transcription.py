import logging
import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from backend.config import settings
from backend.models import TranscriptSegment, WordTimestamp

logger = logging.getLogger(__name__)

_whisper_model = None
_model_lock = threading.Lock()
_loaded_model_name = None   # Tracks which model is currently in the singleton

# Stores the last detected language from Whisper auto-detection so the
# pipeline can read it after transcription completes.
_last_detected_language = {}

# Stores which diarization method was used ("neural" or "heuristic")
_last_diarization_method = {"method": "heuristic"}

# Exposed after model loads so the pipeline can report GPU info in status messages
whisper_device_info = {"device": "cpu", "compute_type": "int8", "gpu_name": ""}

# Dedicated thread pool for transcription so it never competes with
# the default executor (used for base64 encoding, etc.).  A single
# worker is sufficient because Whisper already parallelises internally.
_transcription_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")

# Timeout for model loading (download + init).  The first run downloads
# the model from HuggingFace which can take a while, but should not hang
# indefinitely.
_MODEL_LOAD_TIMEOUT = 600  # 10 minutes

# Per-segment stall timeout: if no new segment is produced within this
# many seconds, assume the model is stuck and return partial results.
_SEGMENT_STALL_TIMEOUT = 120  # 2 minutes


def _get_gpu_vram_mb() -> int:
    """Get total GPU VRAM in MB. Returns 0 if unavailable."""
    gpus = _enumerate_gpus_nvidia_smi()
    if gpus:
        return gpus[0].get("vram_mb", 0)
    # CTranslate2 can see the GPU even when torch can't, but doesn't expose VRAM.
    # Use a conservative default for GTX 1650 class cards.
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return 4096
    except Exception:
        pass
    return 0


def _get_gpu_free_mb() -> int:
    """Get FREE GPU VRAM in MB right now (not total). Returns 0 if unavailable.

    Crucial difference from ``_get_gpu_vram_mb``: when Ollama / another
    process already holds part of the card, the subprocess only gets
    the remainder. Using total VRAM for a fits-in-memory check makes
    the Whisper subprocess "pass" the gate and then OOM mid-stream.
    """
    try:
        import subprocess
        smi = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if smi.returncode != 0 or not smi.stdout.strip():
            return 0
        first = smi.stdout.strip().splitlines()[0].strip()
        return int(float(first))
    except Exception:
        return 0


def is_whisper_model_cached(model_name: str) -> bool:
    """Check if a Whisper model is already downloaded/cached locally.

    Returns True if the model files exist in the HuggingFace cache,
    False if the model would need to be downloaded from HuggingFace.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
        # faster-whisper models are stored as "Systran/faster-whisper-{model}"
        repo_id = f"Systran/faster-whisper-{model_name}"
        # Check for the model config file — if it's cached, the model is downloaded
        result = try_to_load_from_cache(repo_id, "config.json")
        return result is not None and not isinstance(result, type(None))
    except ImportError:
        # huggingface_hub not available — can't check
        pass
    except Exception:
        pass
    # Fallback: check if the model directory exists in the cache
    try:
        import os
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        # HuggingFace stores models as models--Systran--faster-whisper-{model}
        model_dir = os.path.join(cache_dir, f"models--Systran--faster-whisper-{model_name}")
        return os.path.isdir(model_dir)
    except Exception:
        return False


def ensure_whisper_model_downloaded(model_name: str, timeout: float = 600) -> bool:
    """Ensure a Whisper model is downloaded and cached locally.

    If the model is not cached, downloads it from HuggingFace.
    Returns True if the model is available, False if download failed.

    This should be called BEFORE starting transcription to avoid
    download timeouts inside the subprocess.
    """
    if is_whisper_model_cached(model_name):
        logger.info("Whisper model '%s' is cached locally", model_name)
        return True

    logger.info("Whisper model '%s' not cached — downloading from HuggingFace (timeout=%ds)...", model_name, int(timeout))
    try:
        from faster_whisper import WhisperModel
        # Load on CPU with int8 — minimal resources, just triggers download
        m = WhisperModel(model_name, device="cpu", compute_type="int8")
        del m
        import gc
        gc.collect()
        logger.info("Whisper model '%s' downloaded and cached successfully", model_name)
        return True
    except Exception as e:
        logger.error("Failed to download Whisper model '%s': %s", model_name, e)
        return False


async def _evict_ollama_for_whisper() -> None:
    """Unload any loaded Ollama models to free VRAM before launching Whisper.

    v2 Phase 11 (Fix 6): Ollama's idle CUDA context (~1.6 GB) is enough
    to evict Whisper off the GPU on a 4 GB 1650, forcing the ctranslate2
    CPU int8 fallback which is 10-30x slower. Queries ``/api/ps`` and
    posts ``keep_alive=0`` to each loaded model, then sleeps briefly so
    the driver can reclaim the VRAM before the Whisper subprocess opens
    its CUDA context.

    Best-effort: any errors are logged and swallowed by the caller.
    """
    import asyncio as _aio
    try:
        import httpx  # noqa: WPS433 — local import keeps the hot-path slim
    except ImportError:
        logger.debug("httpx not available — skipping Ollama eviction")
        return

    ollama_url = (
        os.environ.get("OLLAMA_HOST")
        or getattr(settings, "OLLAMA_HOST", None)
        or "http://ollama:11434"
    ).rstrip("/")

    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.get(f"{ollama_url}/api/ps")
        except Exception as e:
            logger.debug("Ollama /api/ps unreachable (%s) — nothing to evict", e)
            return
        if resp.status_code != 200:
            logger.debug(
                "Ollama /api/ps returned %s — skipping eviction",
                resp.status_code,
            )
            return
        try:
            models = resp.json().get("models", [])
        except Exception:
            models = []

        if not models:
            logger.debug("Ollama has no models loaded — no eviction needed")
            return

        evicted = 0
        for m in models:
            name = m.get("name") or m.get("model")
            if not name:
                continue
            try:
                await client.post(
                    f"{ollama_url}/api/generate",
                    json={"model": name, "keep_alive": 0},
                )
                evicted += 1
            except Exception as e:
                logger.debug("Failed to evict Ollama model %s: %s", name, e)

        if evicted:
            logger.info(
                "Evicted %d Ollama model(s) from VRAM before Whisper CUDA",
                evicted,
            )
            # Give the driver a beat to actually release the memory.
            await _aio.sleep(2.0)


async def preflight_whisper_check(timeout: float = 90) -> dict:
    """Quick pre-flight check that the Whisper model loads and CUDA works.

    Spawns a subprocess that loads the model and immediately exits.
    Returns {"ok": True, "model": ..., "device": ..., "load_time_ms": ...}
    on success, or {"ok": False, "error": ...} on failure.
    """
    import asyncio
    import json
    import sys
    import tempfile

    model_name = settings.WHISPER_MODEL
    device = "cpu"
    compute_type = "int8"
    device_index = 0
    if settings.GPU_ACCELERATION_ENABLED:
        cuda_available, cuda_count, _, best_idx = _detect_cuda_available()
        if cuda_available and cuda_count > 0:
            device = "cuda"
            compute_type = "float16"
            device_index = best_idx

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, dir="/tmp") as tmp:
        output_path = tmp.name

    try:
        cmd = [
            sys.executable, "-m", "backend.services.whisper_worker",
            "--preflight",
            "--output", output_path,
            "--model", model_name,
            "--device", device,
            "--device-index", str(device_index),
            "--compute-type", compute_type,
        ]
        logger.info("Whisper preflight check: model=%s device=%s timeout=%ds", model_name, device, int(timeout))

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ},
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {
                "ok": False,
                "error": f"Model load timed out after {int(timeout)}s — model may be downloading or GPU unavailable",
                "model": model_name,
                "device": device,
            }

        if stderr:
            for line in stderr.decode(errors="replace").strip().split("\n"):
                if line.strip():
                    logger.info("[whisper-preflight] %s", line)

        if proc.returncode != 0:
            error_msg = stderr.decode(errors="replace")[-300:] if stderr else "Unknown error"
            return {
                "ok": False,
                "error": f"Model load failed (exit {proc.returncode}): {error_msg}",
                "model": model_name,
                "device": device,
            }

        try:
            with open(output_path, "r") as f:
                result = json.load(f)
            # Report the ACTUAL device used, not the requested one.
            # The subprocess may have fallen back from cuda to cpu.
            actual_device = result.get("actual_device", device)
            return {
                "ok": result.get("status") == "ok",
                "error": result.get("error"),
                "model": model_name,
                "device": actual_device,
                "requested_device": device,
                "load_time_ms": result.get("load_time_ms", 0),
            }
        except Exception as e:
            return {"ok": False, "error": f"Failed to read preflight result: {e}", "model": model_name, "device": device}
    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass


async def transcribe_audio_subprocess(
    audio_path: str,
    language: str = "",
    task: str = "transcribe",
    initial_prompt: str = "",
    audio_duration: float = 0,
    progress_callback=None,
    is_animated: bool = False,
    cancel_check=None,
) -> list[TranscriptSegment]:
    """Run Whisper in a subprocess to fully release CTranslate2's CUDA memory.

    CTranslate2 (used by faster-whisper) holds ~1.6GB VRAM in its CUDA context
    even after the model is deleted. torch.cuda.empty_cache() is a no-op because
    PyTorch's CUDA 13.0 is incompatible with the driver's CUDA 12.9.

    Running in a subprocess ensures ALL GPU memory is reclaimed when the process
    exits — model weights, CUDA context, memory pool, everything.

    Returns the same list[TranscriptSegment] as transcribe_audio().
    """
    import asyncio
    import json
    import sys
    import tempfile

    model_name = settings.WHISPER_MODEL
    beam_size = settings.WHISPER_BEAM_SIZE
    vad_filter = settings.WHISPER_VAD_FILTER

    # Detect device settings (reuse the same logic as in-process mode)
    device = "cpu"
    compute_type = "int8"
    device_index = 0
    if settings.GPU_ACCELERATION_ENABLED:
        cuda_available, cuda_count, _, best_idx = _detect_cuda_available()
        if cuda_available and cuda_count > 0:
            device = "cuda"
            compute_type = "float16"
            device_index = best_idx
            gpu_idx = (settings.GPU_DEVICE_INDEX or "").strip()
            if gpu_idx and gpu_idx.isdigit():
                idx = int(gpu_idx)
                if idx < cuda_count:
                    device_index = idx

    # v2 Phase 11 (Fix 6): evict any Ollama models from VRAM BEFORE
    # launching the Whisper subprocess. Ollama holds ~1.6GB in its
    # CUDA context even when idle (the new pipeline only pings /api/tags
    # for health checks), which is enough to evict Whisper off the GPU
    # on a 4 GB 1650 and force the ctranslate2 CPU int8 fallback. CPU
    # Whisper takes 10-30x longer. Eviction is best-effort — if Ollama
    # isn't reachable or returns errors, Whisper will still try CUDA
    # and fall back to CPU as before.
    if device == "cuda":
        try:
            await _evict_ollama_for_whisper()
        except Exception as _ollama_e:
            logger.warning(
                "Ollama VRAM eviction for Whisper failed (non-fatal): %s",
                _ollama_e,
            )

    # ── VRAM safety checks for subprocess (exclusive GPU access) ──
    # Since main process no longer preloads Whisper, the subprocess gets the
    # full GPU. Thresholds are LOWER than the in-process path because there's
    # no competing CUDA context.
    if device == "cuda":
        vram_mb = _get_gpu_vram_mb()
        # The subprocess does NOT actually get exclusive access — Ollama
        # and other CUDA contexts often hold 1-2 GB at pipeline start.
        # We must compare the model's VRAM requirement against the FREE
        # VRAM, not the total. Falls back to total when nvidia-smi
        # can't report free (e.g. driver in headless container).
        free_mb = _get_gpu_free_mb()
        # Use a conservative effective number: the smaller of (free,
        # total - 800) so a card that's idle but small (e.g. 4GB)
        # still respects model overhead beyond raw load.
        if free_mb > 0:
            effective_vram = free_mb
        else:
            # Headless probe failed; assume total minus a CUDA-context
            # safety margin (~800MB) so we don't keep launching Whisper
            # into an empty number.
            effective_vram = max(0, vram_mb - 800)
        # Subprocess-exclusive VRAM requirements (model + beam + CUDA context)
        # Lowered from original — subprocess has exclusive GPU, no competing CUDA contexts
        _VRAM_REQUIREMENTS = {
            "large-v3": 4000,       # 3.1GB model + 0.4GB context + 0.5GB beam (was 4500)
            "large-v3-turbo": 3500, # 2.8GB model + 0.4GB context (was 3800)
            "medium": 2200,         # 1.5GB model + 0.3GB beam + 0.2GB context (was 2400)
            "medium.en": 2200,
        }
        if model_name in _VRAM_REQUIREMENTS:
            min_vram = _VRAM_REQUIREMENTS[model_name]
            user_explicitly_set = getattr(settings, 'WHISPER_MODEL_USER_SET', False)

            if vram_mb == 0:
                # VRAM detection failed (nvidia-smi unavailable in container).
                # Don't block — the subprocess will try CUDA and fall back to CPU
                # if it fails. Log a warning so the user knows VRAM wasn't checked.
                logger.warning(
                    "SUBPROCESS VRAM UNKNOWN: Could not detect GPU VRAM (nvidia-smi unavailable). "
                    "Whisper will attempt CUDA and fall back to CPU if it fails. "
                    "Model='%s', min_vram=%dMB.",
                    model_name, min_vram,
                )
            elif effective_vram < min_vram:
                if user_explicitly_set:
                    # User explicitly chose this model — DON'T downgrade.
                    # Force beam_size=1 (greedy) to minimize VRAM usage.
                    beam_size = 1
                    logger.warning(
                        "SUBPROCESS VRAM TIGHT: '%s' needs ~%dMB but GPU has %dMB free "
                        "(%dMB total). User explicitly selected this model — keeping "
                        "it with beam=1 (greedy decode) to fit in VRAM.",
                        model_name, min_vram, effective_vram, vram_mb,
                    )
                else:
                    # Auto-selected model doesn't fit. Pick the LARGEST
                    # model that fits in the *free* VRAM right now, so
                    # we never start a job that's destined to OOM.
                    if effective_vram >= 2200:
                        downgrade_to = "medium"
                    elif effective_vram >= 1100:
                        downgrade_to = "small"
                    elif effective_vram >= 600:
                        downgrade_to = "base"
                    else:
                        downgrade_to = "tiny"
                    logger.warning(
                        "SUBPROCESS VRAM: '%s' needs ~%dMB but only %dMB free "
                        "(%dMB total — Ollama or another process is holding the rest). "
                        "Downgrading to '%s' to avoid mid-stream CUDA OOM.",
                        model_name, min_vram, effective_vram, vram_mb, downgrade_to,
                    )
                    model_name = downgrade_to
                    beam_size = min(beam_size, 3 if downgrade_to == "medium" else 1)
            else:
                logger.info(
                    "SUBPROCESS VRAM OK: '%s' needs ~%dMB, GPU has %dMB free / %dMB total",
                    model_name, min_vram, effective_vram, vram_mb,
                )

        # Auto-upgrade from 'small' on large GPUs (only if user didn't explicitly set).
        #
        # Prioritize ``large-v3-turbo`` over ``large-v3`` whenever VRAM
        # allows: turbo is ~6x faster on identical English audio with
        # the same WER (~5%). For very large GPUs we still default to
        # turbo unless the user explicitly opted for ``large-v3`` —
        # the 6x speed win dominates the marginal accuracy delta on
        # most realistic content (long-form podcasts / talks / vlogs).
        #
        # Phase 1 also lets us step up ``small → medium`` when the card
        # has 2.5-6 GB free. Medium has meaningfully better recall on
        # faint / accented speech than small, which is the single
        # biggest contributor to dropped segments on 4 GB cards.
        _auto_upgrade = bool(getattr(settings, "WHISPER_AUTO_UPGRADE", True))
        if model_name == "small" and _auto_upgrade and not getattr(settings, 'WHISPER_MODEL_USER_SET', False):
            if vram_mb >= 6000:
                model_name = "large-v3-turbo"
                logger.info(
                    "WHISPER_AUTO_UPGRADE: small → large-v3-turbo on %d MB GPU "
                    "(Phase 1 coverage booster)", vram_mb,
                )
            elif vram_mb >= 8000 and os.environ.get("CLIPAI_PREFER_LARGE_V3", "").lower() in ("1", "true", "yes"):
                # Opt-in for slightly higher accuracy at 6x the cost.
                model_name = "large-v3"
            elif effective_vram >= 2500:
                # Mid-tier GPU (4 GB class) — step up to medium. The
                # beam cap matches the existing downgrade path so the
                # model still fits with headroom for Ollama's idle
                # CUDA context.
                model_name = "medium"
                beam_size = min(beam_size, 3)
                logger.info(
                    "WHISPER_AUTO_UPGRADE: small → medium on %d MB free "
                    "(Phase 1 coverage booster — beam capped at %d)",
                    effective_vram, beam_size,
                )

        # For translate tasks with CJK source, upgrade to medium even on 4GB VRAM.
        # Medium model (~1.5GB) + CUDA context fits in 4GB for translate (shorter output).
        # Medium has dramatically better cross-lingual accuracy than small for CJK→EN.
        if task == "translate" and model_name == "small":
            is_cjk_source = language.lower() in ("ja", "ko", "zh", "zh-cn", "zh-tw") if language else False
            if is_cjk_source and vram_mb >= 3500:
                model_name = "medium"
                beam_size = min(beam_size, 3)
                logger.info(
                    "TRANSLATE UPGRADE: small→medium for CJK→EN translation on %dMB GPU",
                    vram_mb,
                )
            elif not language and vram_mb >= 3500:
                # Auto-detect language — try medium for potentially CJK content
                model_name = "medium"
                beam_size = min(beam_size, 3)
                logger.info(
                    "TRANSLATE UPGRADE: small→medium for auto-detect→EN translation on %dMB GPU",
                    vram_mb,
                )

        # Beam size safety: reduce if model+beam won't fit
        _MODEL_VRAM_MB = {
            "tiny": 400, "base": 500, "small": 1000,
            "medium": 1500, "large-v3-turbo": 3000, "large-v3": 3500,
        }
        _model_mb = _MODEL_VRAM_MB.get(model_name, 1000)
        _beam_overhead = beam_size * 50
        _cuda_overhead = 400  # CUDA driver/context baseline
        _estimated_peak = _model_mb + _beam_overhead + _cuda_overhead
        if vram_mb > 0 and _estimated_peak > vram_mb * 0.85:
            safe_beam = max(1, int((vram_mb * 0.85 - _model_mb - _cuda_overhead) / 50))
            if safe_beam < beam_size:
                logger.warning(
                    "SUBPROCESS: Reducing beam %d→%d for '%s' on %dMB GPU "
                    "(est. peak %dMB)",
                    beam_size, safe_beam, model_name, vram_mb, _estimated_peak,
                )
                beam_size = safe_beam

    logger.info(
        "Subprocess Whisper config: model=%s, device=%s, beam=%d "
        "(settings had: model=%s, beam=%d)",
        model_name, device, beam_size, settings.WHISPER_MODEL, settings.WHISPER_BEAM_SIZE,
    )

    # Detect CJK for compression ratio relaxation
    _is_cjk = language.lower() in ("ja", "ko", "zh", "zh-cn", "zh-tw") if language else False
    effective_best_of = 1 if beam_size <= 1 else 3

    # ── Phase 1 coverage boosters ──
    # Pull user-tunable thresholds from settings so power users can
    # dial sensitivity up/down per account. Defaults tuned to recover
    # 2-5 percentage points of transcript coverage on typical content
    # without materially raising the hallucination rate (the worker's
    # existing post-filter handles the edge cases).
    _vad_onset_default = float(getattr(settings, "WHISPER_VAD_ONSET", 0.10) or 0.10)
    _no_speech_default = float(getattr(settings, "WHISPER_NO_SPEECH_THRESHOLD", 0.5) or 0.5)
    # Clamp to safe ranges so a misconfigured overlay can't blow up the
    # worker (e.g. onset=2.0 would detect no speech at all).
    _vad_onset_default = max(0.05, min(0.35, _vad_onset_default))
    _no_speech_default = max(0.2, min(0.9, _no_speech_default))

    # Translate still needs its own, lower, threshold because the English
    # output of CJK translation is structurally shorter than transcribe.
    if task == "translate":
        no_speech_threshold = "0.45" if _is_cjk else "0.6"
    else:
        no_speech_threshold = f"{_no_speech_default:.2f}"

    # CJK / animation override: slightly lower onset than the global
    # default so we still catch whispered speech and mixed-with-music
    # dialogue that Whisper's standard gate rejects.
    if _is_cjk or is_animated:
        vad_onset = "0.08"
    else:
        vad_onset = f"{_vad_onset_default:.3f}"

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, dir="/tmp") as tmp:
        output_path = tmp.name

    try:
        cmd = [
            sys.executable, "-m", "backend.services.whisper_worker",
            "--audio", audio_path,
            "--output", output_path,
            "--model", model_name,
            "--device", device,
            "--device-index", str(device_index),
            "--compute-type", compute_type,
            "--beam-size", str(beam_size),
            "--best-of", str(effective_best_of),
            "--task", task,
            # Quality parameters (match in-process path exactly)
            # Transcribe: WHISPER_NO_SPEECH_THRESHOLD (default 0.5) replaces
            # the legacy 0.8 so quiet / faint / whispered speech is not
            # silently discarded by Whisper's own VAD gate.
            # Translate: lower threshold to capture quiet backchannel/whispered speech
            # CJK translate: even lower to catch soft-spoken moments, eating speech, ecstatic speech
            "--no-speech-threshold", no_speech_threshold,
            "--log-prob-threshold", "-1.5",
            "--compression-ratio-threshold", "3.0" if _is_cjk else "2.4",
            "--repetition-penalty", "1.1",
            "--no-repeat-ngram-size", "3",
            "--prompt-reset-on-temperature", "0.5",
            # VAD fine-tuning — CJK translate and anime/animated content get
            # more sensitive settings to catch soft-spoken moments, speech
            # while eating, whispered speech, and speech mixed with music.
            "--vad-min-silence-ms", "200" if (_is_cjk or is_animated) else ("250" if task == "translate" else "300"),
            "--vad-speech-pad-ms", "700" if is_animated else ("800" if (task == "translate" and _is_cjk) else "600"),
            # VAD onset: WHISPER_VAD_ONSET (default 0.10) replaces the
            # legacy 0.15 for generic content. CJK/anime paths still use
            # the more sensitive 0.08 (unchanged).
            "--vad-onset", vad_onset,
            "--vad-min-speech-ms", "50" if (_is_cjk or is_animated) else "100",
        ]
        if vad_filter:
            cmd.append("--vad-filter")
        cmd.append("--word-timestamps")
        # Disable condition_on_previous_text for translate tasks — conditioning
        # propagates translation errors forward, causing chains of wrong phrases
        if task != "translate":
            cmd.append("--condition-on-previous")
        else:
            cmd.append("--no-condition-on-previous")
        if _is_cjk:
            cmd.append("--cjk")
        if language:
            cmd.extend(["--language", language])
        if initial_prompt:
            cmd.extend(["--initial-prompt", initial_prompt])
        if audio_duration > 0:
            cmd.extend(["--audio-duration", str(audio_duration)])

        logger.info("Starting Whisper subprocess: model=%s device=%s", model_name, device)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ},
        )

        # Stream stderr line by line for real-time progress updates.
        # The worker emits "PROGRESS:{json}" lines during transcription.
        stderr_lines = []
        async def _stream_stderr():
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                decoded = line.decode(errors="replace").rstrip()
                if not decoded:
                    continue
                if decoded.startswith("PROGRESS:"):
                    try:
                        progress_data = json.loads(decoded[9:])
                        if progress_callback:
                            await progress_callback(progress_data)
                    except Exception:
                        pass
                else:
                    stderr_lines.append(decoded)
                    logger.info("[whisper-worker] %s", decoded)

        import asyncio as _aio
        stderr_task = _aio.create_task(_stream_stderr())

        # Cooperative cancellation: poll cancel_check while the worker
        # runs. On cancel, terminate the subprocess so the GPU is freed
        # immediately instead of waiting for transcription to finish.
        async def _cooperative_wait():
            while True:
                if cancel_check is not None:
                    try:
                        cancel_check()
                    except Exception:
                        # Cancel signal raised — kill the subprocess
                        # and bubble the exception up.
                        try:
                            proc.terminate()
                            try:
                                await _aio.wait_for(proc.wait(), timeout=3.0)
                            except _aio.TimeoutError:
                                proc.kill()
                                await proc.wait()
                        except ProcessLookupError:
                            pass
                        raise
                if proc.returncode is not None:
                    return
                try:
                    await _aio.wait_for(proc.wait(), timeout=1.0)
                    return
                except _aio.TimeoutError:
                    continue

        # Read stdout in parallel; the subprocess's stdout is the JSON
        # output written at the very end so reading it blocks until the
        # process is essentially done.
        stdout_task = _aio.create_task(proc.stdout.read())

        async def _terminate_proc(reason: str = ""):
            """Kill the worker subprocess and drain output tasks.

            Used by both the user-cancel path and the asyncio-cancel
            path (outer ``wait_for`` timeout, parent task cancel).
            Without this guard a cancelled awaiter leaves the worker
            running on the GPU/CPU until the OS reaps it on app exit.
            """
            if proc.returncode is None:
                try:
                    proc.terminate()
                    try:
                        await _aio.wait_for(proc.wait(), timeout=3.0)
                    except _aio.TimeoutError:
                        try:
                            proc.kill()
                        except ProcessLookupError:
                            pass
                        try:
                            await _aio.wait_for(proc.wait(), timeout=2.0)
                        except _aio.TimeoutError:
                            pass
                except ProcessLookupError:
                    pass
                if reason:
                    logger.warning(
                        "Whisper subprocess terminated mid-run: %s", reason,
                    )
            for t in (stdout_task, stderr_task):
                if not t.done():
                    t.cancel()
                try:
                    await t
                except (_aio.CancelledError, Exception):
                    pass

        try:
            await _cooperative_wait()
        except _aio.CancelledError:
            # Outer wait_for timed out OR parent task cancelled. Kill
            # the worker before re-raising so a cancelled / timed-out
            # transcription doesn't leak a 7-GB GPU resident model.
            await _terminate_proc("asyncio cancel / outer wait_for timeout")
            raise
        except BaseException:
            # Any user-side cancel raised by ``cancel_check`` already
            # killed the subprocess inside ``_cooperative_wait``; this
            # branch handles the rest (e.g. KeyboardInterrupt).
            await _terminate_proc("unexpected error in cooperative wait")
            raise
        finally:
            try:
                stderr_task.cancel()
            except Exception:
                pass
        stdout = await stdout_task
        try:
            await stderr_task
        except _aio.CancelledError:
            pass

        if proc.returncode != 0:
            error_msg = "\n".join(stderr_lines[-10:]) if stderr_lines else "Unknown error"
            raise RuntimeError(f"Whisper subprocess failed (exit {proc.returncode}): {error_msg}")

        with open(output_path, "r") as f:
            raw = json.load(f)

        # Log result summary for debugging
        _info = raw.get("info", {})
        _seg_count = len(raw.get("segments", []))
        logger.info(
            "Whisper subprocess result: %d segments, duration=%.1fs, language=%s, "
            "warning=%s, model=%s, device=%s",
            _seg_count, _info.get("duration", 0), _info.get("language", "?"),
            raw.get("warning", "none"), model_name, device,
        )

        if raw.get("status") == "error":
            raise RuntimeError(f"Whisper worker error: {raw.get('error')}")

        # Apply post-processing filters.
        # The worker already ran _filter_segments() internally.
        # For translate mode, skip the second hallucination filter pass to avoid
        # double-filtering that cumulatively removes ~19% of valid segments.
        raw_segments = raw.get("segments", [])
        if task != "translate":
            raw_segments = _filter_hallucinations(raw_segments, task=task)
        raw_segments = _consolidate_segments(raw_segments, task=task, is_animated=is_animated)
        raw_segments = _split_segments_at_sentence_boundaries(raw_segments, task=task)
        raw_segments, _ = _dedupe_long_range(raw_segments)
        segments = []
        for seg in raw_segments:
            words = None
            if seg.get("words"):
                words = [WordTimestamp(start=w["start"], end=w["end"], word=w["word"]) for w in seg["words"]]
            segments.append(TranscriptSegment(
                start=round(seg["start"], 2),
                end=round(seg["end"], 2),
                text=seg["text"],
                speaker="Speaker 1",
                words=words,
                avg_logprob=seg.get("avg_logprob"),
                no_speech_prob=seg.get("no_speech_prob"),
            ))

        # Store detected language for pipeline to read
        info = raw.get("info", {})
        if info.get("language"):
            _last_detected_language["lang"] = info["language"]

        _last_diarization_method["method"] = "deferred"

        logger.info(
            "Whisper subprocess completed: %d segments, CUDA memory fully released",
            len(segments),
        )
        return segments

    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass


def _detect_cuda_available() -> tuple[bool, int, str, int]:
    """Try multiple methods to detect CUDA GPU availability.

    Returns (cuda_available, device_count, gpu_name, best_device_index).
    The best_device_index is the index of the most capable GPU (highest VRAM).
    """
    best_name, best_idx = _get_best_gpu()

    # Method 0 (fast, no CUDA context): Check /dev/nvidia* device nodes
    # Avoids initializing CTranslate2's CUDA context (~200MB) just for detection.
    try:
        import glob as _glob
        nvidia_devs = _glob.glob("/dev/nvidia[0-9]*")
        if nvidia_devs:
            gpu_name = best_name or f"NVIDIA GPU ({len(nvidia_devs)} device{'s' if len(nvidia_devs) > 1 else ''})"
            return True, len(nvidia_devs), gpu_name, best_idx
    except Exception:
        pass

    # Method 1: ctranslate2 (fallback — creates a CUDA context)
    try:
        import ctranslate2
        cuda_count = ctranslate2.get_cuda_device_count()
        if cuda_count > 0:
            gpu_name = best_name or f"CUDA GPU ({cuda_count} device{'s' if cuda_count > 1 else ''})"
            return True, cuda_count, gpu_name, best_idx
    except Exception:
        pass

    # Method 2: PyTorch CUDA (if torch is installed)
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            name = best_name or (torch.cuda.get_device_name(0) if count > 0 else "CUDA GPU")
            return True, count, name, best_idx
    except Exception:
        pass

    # Method 3: Check for NVIDIA device nodes (GPU passed through but libraries incomplete)
    try:
        import glob
        nvidia_devs = glob.glob("/dev/nvidia[0-9]*")
        if nvidia_devs:
            gpu_name = best_name or _get_gpu_name_from_sysfs() or f"NVIDIA GPU ({len(nvidia_devs)} devices)"
            logger.info(
                "NVIDIA device nodes found (%s) but CUDA runtime not available. "
                "Install CUDA toolkit or use Dockerfile.gpu for GPU-accelerated Whisper.",
                nvidia_devs,
            )
            return False, 0, gpu_name, 0
    except Exception:
        pass

    # Method 4: Try loading CUDA library directly
    try:
        import ctypes
        for lib in ["libcuda.so.1", "libcuda.so", "nvcuda.dll"]:
            try:
                ctypes.cdll.LoadLibrary(lib)
                gpu_name = best_name or _get_gpu_name_from_sysfs() or "NVIDIA GPU"
                logger.info("CUDA library %s is loadable — GPU may be available for Whisper", lib)
                return True, 1, gpu_name, best_idx
            except OSError:
                continue
    except Exception:
        pass

    return False, 0, "", 0


def _enumerate_gpus_nvidia_smi() -> list[dict]:
    """Enumerate all NVIDIA GPUs via nvidia-smi.

    Returns list of {index, name, vram_mb} sorted by VRAM (most capable first).
    """
    try:
        import subprocess
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if smi.returncode != 0 or not smi.stdout.strip():
            return []
        gpus = []
        for line in smi.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "vram_mb": int(float(parts[2])),
                    })
                except (ValueError, IndexError):
                    continue
        # Sort by VRAM descending — prefer most capable GPU
        gpus.sort(key=lambda g: g["vram_mb"], reverse=True)
        return gpus
    except Exception:
        return []


def _get_best_gpu() -> tuple[str, int]:
    """Get the most capable GPU name and its device index.

    Returns (gpu_name, device_index). Prefers GPU with the most VRAM
    so a passed-through RTX 4070 is chosen over a server's GTX 1650.
    """
    gpus = _enumerate_gpus_nvidia_smi()
    if gpus:
        best = gpus[0]
        if len(gpus) > 1:
            logger.info(
                "Multiple GPUs detected: %s — selecting %s (index %d, %d MB VRAM)",
                ", ".join(f"{g['name']} [{g['index']}]" for g in gpus),
                best["name"], best["index"], best["vram_mb"],
            )
        return best["name"], best["index"]
    return "", 0


def _get_gpu_name_from_nvidia_smi() -> str:
    """Try to get GPU name from nvidia-smi (best/most capable GPU)."""
    name, _ = _get_best_gpu()
    return name


def _get_gpu_name_from_sysfs() -> str:
    """Try to get GPU name from /proc/driver/nvidia or sysfs."""
    try:
        import glob
        for info_path in glob.glob("/proc/driver/nvidia/gpus/*/information"):
            try:
                for line in open(info_path):
                    if line.startswith("Model:"):
                        return line.split(":", 1)[1].strip()
            except (OSError, IOError):
                continue
    except Exception:
        pass
    return ""


def _cleanup_old_model():
    """Release CUDA memory held by the current Whisper model.

    Must be called WITH _model_lock held. Handles the case where
    CTranslate2/PyTorch CUDA allocations linger after Python del.
    """
    global _whisper_model
    old = _whisper_model
    _whisper_model = None

    # Explicitly delete the model object to trigger CTranslate2's C++ destructor
    try:
        del old
    except Exception:
        pass

    # Force Python GC to run CTranslate2 destructor immediately
    import gc
    gc.collect()
    gc.collect()  # Second pass for reference cycles

    # Release PyTorch CUDA cache (CTranslate2 uses PyTorch under the hood)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
            logger.info("CUDA memory released after model cleanup — %.0fMB free", free_mb)
    except ImportError:
        pass
    except Exception as e:
        logger.debug("CUDA cleanup after model switch: %s", e)


def reload_model():
    """Force-reload the Whisper model on the next transcription call.

    Called when the model selection changes or GPU acceleration is toggled
    so the model can switch size/device without restarting the server.
    Properly cleans up CUDA memory from the old model.
    """
    global _whisper_model, _loaded_model_name
    with _model_lock:
        if _whisper_model is not None:
            _cleanup_old_model()
        _whisper_model = None
        _loaded_model_name = None
    logger.info("Whisper model cache cleared — will reload on next use")


def reload_diarization():
    """Force-reload the pyannote pipeline when HF_AUTH_TOKEN changes."""
    global _diarization_pipeline
    with _diarization_lock:
        _diarization_pipeline = None
    logger.info("pyannote diarization cache cleared — will reload on next use")


def _get_whisper_model():
    global _whisper_model, whisper_device_info, _loaded_model_name
    with _model_lock:
        # Reload if model is not loaded OR if settings changed since last load
        if _whisper_model is not None and _loaded_model_name != settings.WHISPER_MODEL:
            logger.info(
                "Whisper model mismatch: loaded='%s' but settings='%s' — reloading",
                _loaded_model_name, settings.WHISPER_MODEL,
            )
            _cleanup_old_model()
        if _whisper_model is None:
            from faster_whisper import WhisperModel

            # Pick best device and compute type using multi-method detection
            device = "cpu"
            compute_type = "int8"
            gpu_name = ""
            device_index = 0

            cuda_available, cuda_count, detected_name, best_device_idx = _detect_cuda_available()
            if not settings.GPU_ACCELERATION_ENABLED:
                # User has GPU acceleration disabled — force CPU even if
                # CUDA is available, but still record the GPU name for UI.
                gpu_name = detected_name
                logger.info(
                    "GPU acceleration disabled in settings — using CPU for Whisper"
                    + (f" (GPU detected: {gpu_name})" if gpu_name else "")
                )
            elif cuda_available and cuda_count > 0:
                device = "cuda"
                compute_type = "float16"
                gpu_name = detected_name

                # Use specific GPU device index if configured, otherwise use
                # the auto-detected best GPU (highest VRAM).
                gpu_device_idx = (settings.GPU_DEVICE_INDEX or "").strip()
                if gpu_device_idx and gpu_device_idx.isdigit():
                    idx = int(gpu_device_idx)
                    if idx < cuda_count:
                        device_index = idx
                    else:
                        logger.warning(
                            "GPU_DEVICE_INDEX=%s exceeds available CUDA devices (%d). Using best GPU at index %d.",
                            gpu_device_idx, cuda_count, best_device_idx,
                        )
                        device_index = best_device_idx
                else:
                    # Auto-select the most capable GPU (highest VRAM)
                    device_index = best_device_idx if best_device_idx < cuda_count else 0

                logger.info(
                    "CUDA GPU detected: %s — using float16 for Whisper on device %d (of %d)",
                    gpu_name or "unknown", device_index, cuda_count,
                )
            elif detected_name:
                # GPU detected but CUDA runtime not available
                logger.warning(
                    "GPU detected (%s) but CUDA runtime not available for Whisper. "
                    "Falling back to CPU. For GPU acceleration, ensure CUDA toolkit "
                    "is installed and the container has --gpus all.",
                    detected_name,
                )
                gpu_name = detected_name  # Record name even if can't use it

            whisper_device_info = {
                "device": device,
                "compute_type": compute_type,
                "gpu_name": gpu_name,
                "device_index": device_index,
            }

            # Build model kwargs — pass device_index for GPU selection
            model_kwargs = {
                "device": device,
                "compute_type": compute_type,
            }
            if device == "cuda":
                model_kwargs["device_index"] = device_index

            # Auto-upgrade model when GPU is available and user hasn't explicitly chosen.
            # VRAM-aware: large-v3-turbo needs ~3GB VRAM in float16. On 4GB GPUs,
            # this leaves <1GB headroom and crashes on complex audio segments
            # (multilingual, music, overlapping speakers cause transient VRAM spikes).
            # Skip if user explicitly selected a model in the UI (WHISPER_MODEL_USER_SET)
            # or if WHISPER_AUTO_UPGRADE is disabled.
            _auto_upgrade_inproc = bool(getattr(settings, "WHISPER_AUTO_UPGRADE", True))
            if (device == "cuda" and settings.WHISPER_MODEL == "small"
                    and _auto_upgrade_inproc
                    and not getattr(settings, 'WHISPER_MODEL_USER_SET', False)):
                gpus = _enumerate_gpus_nvidia_smi()
                vram_mb = gpus[0]["vram_mb"] if gpus else 0
                # Prefer large-v3-turbo whenever VRAM allows: ~6x faster
                # than large-v3 with the same WER on English. Opt back
                # in to large-v3 via CLIPAI_PREFER_LARGE_V3=1 for tasks
                # where the marginal accuracy delta matters.
                if vram_mb >= 6000:
                    if (vram_mb >= 8000
                            and os.environ.get("CLIPAI_PREFER_LARGE_V3", "").lower()
                                in ("1", "true", "yes")):
                        settings.WHISPER_MODEL = "large-v3"
                        logger.info("Auto-upgraded Whisper to large-v3 (%dMB VRAM, opt-in) — session only, not persisted", vram_mb)
                    else:
                        settings.WHISPER_MODEL = "large-v3-turbo"
                        logger.info("Auto-upgraded Whisper to large-v3-turbo (%dMB VRAM) — session only, not persisted", vram_mb)
                elif vram_mb >= 2800:
                    # Phase 1 mid-tier step: 4 GB class cards (GTX 1650,
                    # RTX 3050 4GB, etc.) get ``small → medium``.
                    # Medium has noticeably better recall on faint /
                    # accented speech than small, which is the top
                    # cause of missed segments on these GPUs.
                    settings.WHISPER_MODEL = "medium"
                    logger.info(
                        "Auto-upgraded Whisper to medium (%dMB VRAM, Phase 1 "
                        "mid-tier step) — session only, not persisted",
                        vram_mb,
                    )
                else:
                    # <2.8 GB: keep 'small' (~500MB VRAM) to avoid CUDA OOM.
                    # large-v3-turbo crashed at 4:31 on a 113-min Japanese video
                    # in production — 18 segments collected then lost.
                    logger.info(
                        "Keeping Whisper 'small' — %dMB VRAM insufficient for "
                        "medium upgrade (needs ≥2.8 GB)",
                        vram_mb,
                    )

            # ── VRAM safety: auto-DOWNGRADE user-selected models that won't fit ──
            # Whisper VRAM requirements (float16, approximate):
            #   tiny:  ~0.4GB    base:  ~0.5GB    small: ~1.0GB
            #   medium: ~2.5GB   large-v3: ~3.5GB  large-v3-turbo: ~3.0GB
            # On 4GB GPUs, medium+ models crash on long audio (KV cache grows
            # with duration). Downgrade to small with a loud warning.
            # VRAM requirements = minimum TOTAL GPU memory needed.
            # These assume exclusive GPU access (Ollama unloaded before Whisper).
            # medium (~1.5GB model + ~0.5GB beam/KV + ~0.3GB buffers = ~2.3GB peak)
            # needs ~3GB total to leave headroom for CUDA spikes on complex audio.
            _VRAM_REQUIREMENTS = {
                "large-v3": 4000,       # ~3.5GB model alone (was 6000)
                "large-v3-turbo": 3500, # ~3.0GB model alone (was 5000)
                "medium": 2200,         # ~1.5GB model + ~0.5GB beam + ~0.3GB buffers (was 3000)
                "medium.en": 2200,
            }
            if device == "cuda" and settings.WHISPER_MODEL in _VRAM_REQUIREMENTS:
                # Try nvidia-smi first, fall back to PyTorch CUDA reporting
                # (nvidia-smi is NOT available in Docker containers without NVIDIA runtime)
                gpus = _enumerate_gpus_nvidia_smi()
                vram_mb = gpus[0]["vram_mb"] if gpus else 0
                if vram_mb == 0:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            vram_mb = int(torch.cuda.get_device_properties(0).total_mem / 1024 / 1024)
                            logger.info("VRAM detected via PyTorch: %dMB", vram_mb)
                    except Exception:
                        pass
                min_vram = _VRAM_REQUIREMENTS[settings.WHISPER_MODEL]
                if 0 < vram_mb < min_vram:
                    if getattr(settings, 'WHISPER_MODEL_USER_SET', False):
                        # User explicitly chose — keep it, reduce beam
                        logger.warning(
                            "IN-PROCESS VRAM TIGHT: '%s' needs ~%dMB, GPU has %dMB. "
                            "Keeping user's choice with beam=1.",
                            settings.WHISPER_MODEL, min_vram, vram_mb,
                        )
                        whisper_device_info["recommended_beam_size"] = 1
                    else:
                        original = settings.WHISPER_MODEL
                        # Downgrade the LOCAL model name only — do NOT modify
                        # settings.WHISPER_MODEL because subsequent _persist_user_settings()
                        # calls would overwrite the user's saved model choice.
                        settings.WHISPER_MODEL = "medium"  # NEVER small on GPU
                        logger.warning(
                            "AUTO-DOWNGRADE (session only): Whisper '%s' needs ~%dMB VRAM but GPU only has %dMB. "
                            "Downgrading to 'medium' for this session. Saved model preference is preserved. "
                            "To force '%s', select it in Settings.",
                            original, min_vram, vram_mb, original,
                        )

            # ── VRAM-aware beam size for inference ──
            # Estimate whether model + beam_size will fit in available VRAM.
            # Only reduce beam when the combination would actually exceed safe limits.
            # CTranslate2 catches cudaMalloc failures silently — returns empty results.
            if device == "cuda":
                _gpus = _enumerate_gpus_nvidia_smi()
                _vram = _gpus[0]["vram_mb"] if _gpus else 0
                if _vram == 0:
                    try:
                        import torch as _torch
                        if _torch.cuda.is_available():
                            _vram = int(_torch.cuda.get_device_properties(0).total_mem / 1024 / 1024)
                    except Exception:
                        pass

                # Estimate peak VRAM for current model + beam=N
                # Model weights (float16): tiny=400, base=500, small=1000, medium=1500,
                #   large-v3-turbo=3000, large-v3=3500
                _MODEL_VRAM_MB = {
                    "tiny": 400, "tiny.en": 400,
                    "base": 500, "base.en": 500,
                    "small": 1000, "small.en": 1000,
                    "medium": 1500, "medium.en": 1500,
                    "large-v3-turbo": 3000,
                    "large-v3": 3500, "large-v2": 3500, "large": 3500,
                }
                _model_mb = _MODEL_VRAM_MB.get(settings.WHISPER_MODEL, 1000)
                # Beam overhead: ~50MB per beam (KV cache + workspace)
                _beam_overhead = settings.WHISPER_BEAM_SIZE * 50
                # CUDA driver/context: ~400MB
                _cuda_overhead = 400
                _estimated_peak = _model_mb + _beam_overhead + _cuda_overhead
                # Usable VRAM (total minus driver)
                _usable = _vram - _cuda_overhead if _vram > 0 else 0

                if _vram > 0 and _estimated_peak > _vram * 0.85:
                    # Would exceed 85% of total VRAM — find safe beam size
                    # Work backward: max_beam = (usable * 0.85 - model) / 50
                    _safe_beam = max(1, int((_usable * 0.85 - _model_mb) / 50))
                    _safe_beam = min(_safe_beam, settings.WHISPER_BEAM_SIZE)
                    if _safe_beam < settings.WHISPER_BEAM_SIZE:
                        whisper_device_info["recommended_beam_size"] = _safe_beam
                        logger.info(
                            "VRAM-aware: beam %d→%d for %s on %dMB GPU "
                            "(est. peak %dMB, usable %dMB)",
                            settings.WHISPER_BEAM_SIZE, _safe_beam,
                            settings.WHISPER_MODEL, _vram, _estimated_peak, _usable,
                        )
                    else:
                        whisper_device_info["recommended_beam_size"] = None
                        logger.info(
                            "VRAM OK: %s + beam=%d fits on %dMB GPU "
                            "(est. peak %dMB, usable %dMB)",
                            settings.WHISPER_MODEL, settings.WHISPER_BEAM_SIZE,
                            _vram, _estimated_peak, _usable,
                        )
                else:
                    whisper_device_info["recommended_beam_size"] = None  # Fits fine

            logger.info(
                "Loading Whisper model: %s (device=%s, compute=%s%s)",
                settings.WHISPER_MODEL, device, compute_type,
                f", device_index={device_index}" if device == "cuda" else "",
            )
            try:
                _whisper_model = WhisperModel(
                    settings.WHISPER_MODEL,
                    **model_kwargs,
                )
            except Exception as e:
                if device == "cuda":
                    logger.warning(
                        "Failed to load Whisper on CUDA (%s, device_index=%d), falling back to CPU: %s",
                        gpu_name, device_index, e,
                    )
                    device = "cpu"
                    compute_type = "int8"
                    # Downgrade model for CPU — large models are too slow on CPU
                    if settings.WHISPER_MODEL in ("large-v3", "large-v3-turbo"):
                        settings.WHISPER_MODEL = "medium"
                        logger.info("Downgraded Whisper model to 'medium' for CPU fallback (large models too slow on CPU, medium has much better accuracy than small)")
                    whisper_device_info.update({"device": device, "compute_type": compute_type})
                    _whisper_model = WhisperModel(
                        settings.WHISPER_MODEL,
                        device=device,
                        compute_type=compute_type,
                    )
                else:
                    raise

            # ── Verify GPU is actually being used ──
            if device == "cuda":
                _verify_whisper_gpu_usage(gpu_name, device_index)
            else:
                logger.info(
                    "Whisper model '%s' loaded on CPU (int8) — GPU not used for transcription",
                    settings.WHISPER_MODEL,
                )

            _loaded_model_name = settings.WHISPER_MODEL
    return _whisper_model


def _verify_whisper_gpu_usage(gpu_name: str, device_index: int = 0):
    """Verify that the Whisper model is actually running on the GPU.

    Checks ctranslate2 and torch to confirm CUDA is active,
    providing definitive proof the GPU is in use.
    """
    verification_methods = []

    # Method 1: Check ctranslate2 CUDA device count (confirms CUDA runtime is active)
    try:
        import ctranslate2
        cuda_count = ctranslate2.get_cuda_device_count()
        if cuda_count > 0:
            verification_methods.append(f"ctranslate2 sees {cuda_count} CUDA device(s)")
    except Exception:
        pass

    # Method 2: Check PyTorch CUDA memory allocation (if torch is available)
    try:
        import torch
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated(device_index)
            mem_reserved = torch.cuda.memory_reserved(device_index)
            if mem_allocated > 0 or mem_reserved > 0:
                verification_methods.append(
                    f"torch CUDA device {device_index}: "
                    f"{mem_allocated / 1024 / 1024:.1f}MB allocated, "
                    f"{mem_reserved / 1024 / 1024:.1f}MB reserved"
                )
            else:
                verification_methods.append(
                    f"torch CUDA device {device_index} available "
                    "(memory will allocate on first inference)"
                )
    except Exception:
        pass

    # Method 3: Check nvidia-smi for GPU processes
    try:
        import subprocess, os as _os
        smi = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if smi.returncode == 0 and smi.stdout.strip():
            our_pid = str(_os.getpid())
            for line in smi.stdout.strip().split("\n"):
                if our_pid in line:
                    verification_methods.append(f"nvidia-smi confirms GPU usage: {line.strip()}")
                    break
    except Exception:
        pass

    if verification_methods:
        logger.info(
            "GPU VERIFICATION for Whisper (device=%d, %s): CONFIRMED\n  - %s",
            device_index, gpu_name, "\n  - ".join(verification_methods),
        )
    else:
        logger.warning(
            "GPU VERIFICATION for Whisper (device=%d, %s): UNCONFIRMED — "
            "could not verify GPU memory allocation. Model loaded with device=cuda "
            "but GPU usage cannot be independently confirmed.",
            device_index, gpu_name,
        )


def preload_model():
    """Preload the Whisper model so the first transcription doesn't have to wait."""
    try:
        logger.info("Preloading Whisper model in background...")
        _get_whisper_model()

        # Log a summary of what GPU is used for each subsystem
        _wdev = whisper_device_info
        if _wdev["device"] == "cuda" and _wdev.get("gpu_name"):
            whisper_label = f"CUDA — {_wdev['gpu_name']} ({_wdev['compute_type']}, device {_wdev.get('device_index', 0)})"
        elif _wdev["device"] == "cuda":
            whisper_label = f"CUDA ({_wdev['compute_type']})"
        else:
            whisper_label = f"CPU ({_wdev['compute_type']})"

        try:
            from backend.services.clip_exporter import get_encoder_label
            encoder_label = get_encoder_label()
        except Exception:
            encoder_label = "unknown"

        logger.info("Whisper transcription: %s", whisper_label)
        logger.info("Video encoding: %s", encoder_label)
    except Exception as e:
        logger.warning(f"Whisper model preload failed (will retry on first use): {e}")


async def transcribe_audio(
    audio_path: str,
    language: str = "",
    task: str = "transcribe",
    initial_prompt: str = "",
    cancel_check: Optional[Callable] = None,
    progress_callback: Optional[Callable] = None,
    audio_duration: float = 0,
    is_animated: bool = False,
) -> list[TranscriptSegment]:
    """
    Transcribe audio using faster-whisper with live progress reporting.

    Args:
        audio_path: Path to the audio file.
        language: ISO 639-1 language code (e.g. "en", "es"). Empty = auto-detect.
        cancel_check: Optional callable that raises if the job was cancelled.
        progress_callback: Optional async callable(info_dict) for progress.
            info_dict keys: segments, pct, lang, position_sec, eta_sec, last_text
        audio_duration: Total audio duration in seconds (for progress estimation).
    """
    import asyncio
    import functools

    # Shared state for progress reporting from the sync thread
    progress_state = {
        "segments": 0,
        "latest_end": 0.0,
        "language": "",
        "last_text": "",
        "start_time": 0.0,
        "done": False,
    }
    lock = threading.Lock()

    loop = asyncio.get_running_loop()

    # Use the dedicated transcription executor so the Whisper thread is
    # never blocked behind base64-encoding or other default-pool tasks.
    future = loop.run_in_executor(
        _transcription_executor, functools.partial(
            _transcribe_sync, audio_path, language=language,
            task=task, initial_prompt=initial_prompt,
            progress_state=progress_state, progress_lock=lock,
            is_animated=is_animated,
        )
    )

    task = asyncio.ensure_future(future)
    last_reported = 0
    last_segment_time = time.monotonic()
    _last_progress_emit = time.monotonic()

    # Timeout: generous base + proportional to audio length
    # e.g. 10 min base + 3x audio duration (Whisper is typically ~1x real-time on CPU)
    max_timeout = _MODEL_LOAD_TIMEOUT + max(audio_duration * 3, 300)

    while not task.done():
        await asyncio.sleep(1.0)
        if not task.done():
            if cancel_check:
                cancel_check()

            # Detect stalls: if no new segment appears for _SEGMENT_STALL_TIMEOUT
            # seconds after at least one segment has been produced, and we're past
            # the model-loading phase, warn but keep waiting (model may be on a
            # long silence stretch).  If the overall timeout expires, we break out.
            with lock:
                seg_count = progress_state["segments"]
                done_flag = progress_state["done"]

            if seg_count > last_reported:
                last_segment_time = time.monotonic()

            # Overall timeout guard
            elapsed_total = time.monotonic() - (progress_state.get("start_time") or time.monotonic())
            if not done_flag and progress_state.get("start_time") and elapsed_total > max_timeout:
                logger.warning(
                    "Transcription timed out after %.0fs (audio=%.0fs). "
                    "Returning %d segments collected so far.",
                    elapsed_total, audio_duration, seg_count,
                )
                break

            # Report progress from the sync thread's shared state.
            # Emit on new segments OR every 15 seconds (time-based) to
            # prevent long gaps when Whisper is processing silently.
            if progress_callback:
                with lock:
                    seg_count = progress_state["segments"]
                    latest_end = progress_state["latest_end"]
                    lang = progress_state["language"]
                    last_text = progress_state["last_text"]
                    start_time = progress_state["start_time"]
                _now = time.monotonic()
                _new_segments = seg_count > last_reported
                _time_elapsed = (_now - _last_progress_emit) >= 15 if seg_count > 0 else False
                if _new_segments or _time_elapsed:
                    if _new_segments:
                        last_reported = seg_count
                    pct = min(99, int((latest_end / audio_duration) * 100)) if audio_duration > 0 else 0
                    # Calculate ETA
                    eta_sec = 0
                    if pct > 0 and start_time > 0:
                        elapsed = _now - start_time
                        eta_sec = max(0, (elapsed / (pct / 100)) * (1 - pct / 100))
                    await progress_callback({
                        "segments": seg_count,
                        "pct": pct,
                        "lang": lang,
                        "position_sec": latest_end,
                        "eta_sec": eta_sec,
                        "last_text": last_text,
                    })
                    _last_progress_emit = _now

    if task.done():
        try:
            return task.result()
        except Exception as e:
            # Whisper crashed mid-transcription (CUDA OOM, segfault, etc.)
            # Try to recover partial segments collected before the crash.
            with lock:
                partial = list(progress_state.get("raw_segments", []))
            if partial:
                logger.warning(
                    "Whisper crashed after %d segments (error: %s). "
                    "Returning partial transcription.",
                    len(partial), str(e)[:200],
                )
                partial = _filter_hallucinations(partial, task=task)
                partial = _consolidate_segments(partial, task=task)
                partial = _split_segments_at_sentence_boundaries(partial, task=task)
                partial, _ = _dedupe_long_range(partial)
                if partial:
                    result_segs = []
                    for seg in partial:
                        words = None
                        if seg.get("words"):
                            words = [WordTimestamp(**w) for w in seg["words"]]
                        result_segs.append(TranscriptSegment(
                            start=round(seg["start"], 2),
                            end=round(seg["end"], 2),
                            text=seg["text"],
                            speaker="Speaker 1",
                            words=words,
                            confidence=seg.get("confidence"),
                            avg_logprob=seg.get("avg_logprob"),
                            no_speech_prob=seg.get("no_speech_prob"),
                        ))
                    _last_diarization_method["method"] = "deferred"
                    logger.info(
                        "Recovered %d segments covering %.1fs of audio",
                        len(result_segs),
                        result_segs[-1].end if result_segs else 0,
                    )
                    return result_segs
            logger.error("Whisper crashed with no recoverable segments: %s", str(e)[:200])
            raise

    # Timeout path: return whatever segments were collected so far.
    # The sync thread stores partial results in progress_state["raw_segments"].
    logger.warning("Returning partial transcription due to timeout")
    with lock:
        partial = list(progress_state.get("raw_segments", []))
    if partial:
        partial, removed_intervals = _dedupe_long_range(partial)
        return _assign_speakers(partial, removed_intervals=removed_intervals)
    return []


def _chunk_audio(audio_path: str, chunk_duration: int = 600, overlap: int = 30) -> list[dict]:
    """Split audio into overlapping chunks for long-form transcription.

    Args:
        audio_path: Path to preprocessed audio file.
        chunk_duration: Duration of each chunk in seconds (default 10 minutes).
        overlap: Overlap between chunks in seconds (default 30s).

    Returns:
        List of {"path": str, "offset": float} dicts.
        For short audio (<= chunk_duration), returns single chunk with offset 0.
    """
    import subprocess, tempfile, os, json as _json

    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", audio_path],
        capture_output=True, text=True, timeout=30,
    )
    duration = float(_json.loads(probe.stdout)["format"]["duration"])

    if duration <= chunk_duration:
        return [{"path": audio_path, "offset": 0.0}]

    chunks = []
    start = 0.0
    chunk_dir = tempfile.mkdtemp(prefix="whisper_chunks_")

    while start < duration:
        chunk_end = min(start + chunk_duration, duration)
        chunk_path = os.path.join(chunk_dir, f"chunk_{len(chunks):04d}.wav")

        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-ss", str(start),
            "-t", str(chunk_end - start),
            "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
            chunk_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode == 0:
            chunks.append({"path": chunk_path, "offset": start})
        else:
            logger.warning("Failed to create chunk at %.1fs: %s",
                           start, result.stderr[-200:] if result.stderr else "unknown")

        start += chunk_duration - overlap

    logger.info("Split %.0fs audio into %d chunks (%ds each, %ds overlap)",
                duration, len(chunks), chunk_duration, overlap)
    return chunks


def _merge_chunk_segments(all_chunks: list[list[dict]], overlap: int = 30) -> list[dict]:
    """Merge segments from overlapping audio chunks, deduplicating overlap regions."""
    if len(all_chunks) <= 1:
        return all_chunks[0] if all_chunks else []

    merged = list(all_chunks[0])

    for chunk_idx in range(1, len(all_chunks)):
        next_segments = all_chunks[chunk_idx]
        if not next_segments:
            continue
        if not merged:
            merged = list(next_segments)
            continue

        prev_last_end = merged[-1]["end"]
        next_first_start = next_segments[0]["start"]
        overlap_start = next_first_start
        overlap_end = prev_last_end

        if overlap_start >= overlap_end:
            merged.extend(next_segments)
            continue

        midpoint = (overlap_start + overlap_end) / 2.0

        # Trim prev chunk: remove segments that START after midpoint
        while merged and merged[-1]["start"] > midpoint:
            merged.pop()

        # Trim next chunk: skip segments that END before midpoint
        skip = 0
        for seg in next_segments:
            if seg["end"] < midpoint:
                skip += 1
            else:
                break

        merged.extend(next_segments[skip:])

    # Final pass: remove remaining timestamp overlaps
    deduped = []
    for seg in merged:
        if deduped and seg["start"] < deduped[-1]["end"] - 0.1:
            if seg.get("confidence", 0) > deduped[-1].get("confidence", 0):
                deduped[-1] = seg
        else:
            deduped.append(seg)

    return deduped


def _transcribe_sync(
    audio_path: str,
    language: str = "",
    task: str = "transcribe",
    initial_prompt: str = "",
    progress_state: Optional[dict] = None,
    progress_lock: Optional[threading.Lock] = None,
    is_animated: bool = False,
) -> list[TranscriptSegment]:
    model = _get_whisper_model()
    # Use VRAM-aware beam size if available (prevents silent CUDA OOM on ≤4GB GPUs)
    _recommended_beam = whisper_device_info.get("recommended_beam_size")
    effective_beam = _recommended_beam if _recommended_beam is not None else settings.WHISPER_BEAM_SIZE
    effective_best_of = 1 if effective_beam <= 1 else 3
    # Translate tasks (e.g. Japanese→English) need lower no_speech_threshold
    # because quiet backchannel responses, whispered asides, and expressive
    # speech get high no_speech_prob from Whisper's English acoustic model.
    _is_cjk_inproc = language.lower() in ("ja", "ko", "zh", "zh-cn", "zh-tw") if language else False
    # Phase 1 coverage booster: use the WHISPER_NO_SPEECH_THRESHOLD
    # setting (default 0.5) for transcribe, same as the subprocess path.
    _ns_setting_inproc = float(getattr(settings, "WHISPER_NO_SPEECH_THRESHOLD", 0.5) or 0.5)
    _ns_setting_inproc = max(0.2, min(0.9, _ns_setting_inproc))
    # CJK translate: even lower threshold to catch soft-spoken moments
    if task == "translate":
        _no_speech_thresh = 0.45 if _is_cjk_inproc else 0.6
    else:
        _no_speech_thresh = _ns_setting_inproc
    transcribe_kwargs = {
        "task": task,
        "beam_size": effective_beam,
        "best_of": effective_best_of,
        "vad_filter": settings.WHISPER_VAD_FILTER,
        "condition_on_previous_text": (task != "translate"),
        "word_timestamps": True,
        "no_speech_threshold": _no_speech_thresh,
        "log_prob_threshold": -1.5,
        "compression_ratio_threshold": 2.4,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 3,           # Was 0 — prevents phrase-level repetition
        # Temperature fallback: if a segment fails quality checks at temp 0.0,
        # retry with progressively more randomness.
        "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        # Reset previous-text context when temp > 0.5 (failed segment).
        # Prevents hallucinated text from poisoning the next segment.
        "prompt_reset_on_temperature": 0.5,
    }
    if settings.WHISPER_VAD_FILTER:
        if task == "translate" and _is_cjk_inproc:
            # CJK translate: very sensitive VAD to catch soft-spoken moments,
            # speech while eating, whispers, and ecstatic outbursts
            transcribe_kwargs["vad_parameters"] = {
                "min_silence_duration_ms": 200,
                "speech_pad_ms": 800,
                "onset": 0.08,
                "min_speech_duration_ms": 50,
            }
        elif is_animated:
            # Anime / animated content: dialogue is mixed with background
            # music, SFX, and action sounds. Standard VAD onset (0.2) is
            # too aggressive and drops 10-17% of speech. Use CJK-level
            # sensitivity to preserve speech under music and catch short
            # exclamations / whispers (Problem 3 in the anime audit).
            transcribe_kwargs["vad_parameters"] = {
                "min_silence_duration_ms": 200,
                "speech_pad_ms": 700,     # Reduced from 1000 — 500ms/side shifted boundaries too far
                "onset": 0.08,            # Very sensitive — catch speech under music
                "min_speech_duration_ms": 50,  # Catch short exclamations
            }
            logger.info("VAD parameters: using anime-sensitive preset (onset=0.08, pad=700ms)")
        else:
            # Generic path — onset pulled from WHISPER_VAD_ONSET (Phase 1,
            # default 0.10). The silero-published 0.10 default catches
            # whispered / soft speech that the legacy 0.2 hard-coded
            # value was rejecting on podcasts / interviews / vlogs.
            _vad_onset_inproc = float(getattr(settings, "WHISPER_VAD_ONSET", 0.10) or 0.10)
            _vad_onset_inproc = max(0.05, min(0.35, _vad_onset_inproc))
            transcribe_kwargs["vad_parameters"] = {
                "min_silence_duration_ms": 300,   # Was 500 — shorter threshold preserves natural pauses
                "speech_pad_ms": 600,              # Wide padding captures trailing quiet words
                "onset": _vad_onset_inproc,        # Phase 1: default 0.10 (was 0.2)
                "min_speech_duration_ms": 100,     # Don't discard very short utterances
            }
    # CJK languages have higher natural compression ratios — relax threshold.
    # Also applies to translate tasks where the source is CJK (Whisper still
    # processes the CJK audio internally before translating to English).
    _is_cjk_hint = language.lower() in ("ja", "ko", "zh", "zh-cn", "zh-tw") if language else False
    if _is_cjk_hint:
        transcribe_kwargs["compression_ratio_threshold"] = 3.0
    elif task == "translate":
        # Translate output (English) from any source can have higher compression
        # due to structural differences between languages
        transcribe_kwargs["compression_ratio_threshold"] = 2.8

    if language:
        transcribe_kwargs["language"] = language
        logger.info(f"Transcribing with explicit language: {language}, task: {task}")
    else:
        logger.info(f"Transcribing with auto language detection, task: {task}")

    if initial_prompt:
        transcribe_kwargs["initial_prompt"] = initial_prompt
        logger.info(f"Using initial_prompt ({len(initial_prompt)} chars)")

    opts = (
        f"task={task}, beam={effective_beam}"
        f"{' (VRAM-reduced)' if _recommended_beam is not None else ''}, "
        f"best_of={effective_best_of}, "
        f"vad={'on' if settings.WHISPER_VAD_FILTER else 'off'}, "
        f"no_repeat_ngram=3, temp_fallback=6_steps"
    )
    logger.info(f"Whisper options: {opts}")

    # ── Audio preprocessing for accuracy ──
    # Normalize volume so Whisper gets consistent input levels.
    # Whisper was trained on -20 LUFS audio; quiet recordings or loud
    # ones with clipping both degrade accuracy.
    # For translate tasks: gentler noise gate and compression to preserve
    # quiet backchannel responses and expressive speech.
    if task == "translate":
        # CJK translate: disable noise gate entirely and use gentler compression
        # to preserve soft-spoken moments, speech while eating, whispers, and
        # ecstatic outbursts. Standard gates cut these as "silence".
        if _is_cjk_inproc:
            _af_base = (
                "highpass=f=30,"
                "acompressor=threshold=-40dB:ratio=1.5:attack=15:release=300:makeup=10dB"
            )
        else:
            _af_base = (
                "highpass=f=50,"
                "acompressor=threshold=-35dB:ratio=2:attack=10:release=200:makeup=8dB,"
                "agate=threshold=-55dB:attack=10:release=100"
            )
    else:
        _af_base = (
            "highpass=f=50,"
            "acompressor=threshold=-30dB:ratio=4:attack=5:release=100:makeup=6dB,"
            "agate=threshold=-45dB:attack=5:release=50"
        )
    preprocessed_path = audio_path
    try:
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            preprocessed_path = tmp.name

        # Two-pass loudnorm for precise normalization
        # Pass 1: Measure loudness statistics
        measure_cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-af", f"{_af_base},loudnorm=I=-20:TP=-1.5:LRA=7:print_format=json",
            "-f", "null", "-",
        ]
        measure_result = subprocess.run(measure_cmd, capture_output=True, text=True, timeout=120)

        loudnorm_stats = None
        if measure_result.returncode == 0 and measure_result.stderr:
            import json as _json
            stderr_text = measure_result.stderr
            json_start = stderr_text.rfind('{')
            json_end = stderr_text.rfind('}')
            if json_start >= 0 and json_end > json_start:
                try:
                    loudnorm_stats = _json.loads(stderr_text[json_start:json_end + 1])
                except (ValueError, KeyError):
                    pass

        if loudnorm_stats:
            # Pass 2: Apply measured corrections (precise normalization)
            measured_i = loudnorm_stats.get("input_i", "-24.0")
            measured_tp = loudnorm_stats.get("input_tp", "-2.0")
            measured_lra = loudnorm_stats.get("input_lra", "7.0")
            measured_thresh = loudnorm_stats.get("input_thresh", "-34.0")
            target_offset = loudnorm_stats.get("target_offset", "0.0")

            normalize_filter = (
                f"{_af_base},"
                f"loudnorm=I=-20:TP=-1.5:LRA=7:linear=true"
                f":measured_I={measured_i}:measured_TP={measured_tp}"
                f":measured_LRA={measured_lra}:measured_thresh={measured_thresh}"
                f":offset={target_offset}"
            )
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-af", normalize_filter,
                "-ar", "16000", "-ac", "1",
                preprocessed_path,
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode != 0:
                logger.warning("Two-pass loudnorm failed, falling back to single-pass")
                cmd_fallback = [
                    "ffmpeg", "-y", "-i", audio_path,
                    "-af", f"{_af_base},loudnorm=I=-20:TP=-1.5:LRA=7",
                    "-ar", "16000", "-ac", "1",
                    preprocessed_path,
                ]
                subprocess.run(cmd_fallback, capture_output=True, timeout=120)
            else:
                logger.info("Audio preprocessed: two-pass loudnorm to -20 LUFS, 16kHz mono")
        else:
            # Fallback: single-pass if measurement failed
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-af", f"{_af_base},loudnorm=I=-20:TP=-1.5:LRA=7",
                "-ar", "16000", "-ac", "1",
                preprocessed_path,
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode != 0:
                preprocessed_path = audio_path
            else:
                logger.info("Audio preprocessed: single-pass loudnorm (measurement failed)")
    except Exception as e:
        logger.warning("Audio preprocessing skipped: %s", e)
        preprocessed_path = audio_path

    # ── Validate preprocessed audio ──
    # If loudnorm produced a corrupt/empty file, fall back to the original.
    if preprocessed_path != audio_path:
        try:
            import os as _os
            file_size = _os.path.getsize(preprocessed_path)
            if file_size < 1000:  # WAV header alone is 44 bytes; <1KB = certainly empty
                logger.warning(
                    "Preprocessed audio is suspiciously small (%d bytes) — "
                    "falling back to original audio",
                    file_size,
                )
                preprocessed_path = audio_path
            else:
                import subprocess as _sp
                probe = _sp.run(
                    ["ffprobe", "-v", "error", "-select_streams", "a:0",
                     "-show_entries", "stream=duration,sample_rate,channels",
                     "-of", "csv=p=0", preprocessed_path],
                    capture_output=True, text=True, timeout=10,
                )
                if probe.returncode != 0 or not probe.stdout.strip():
                    logger.warning(
                        "Preprocessed audio has no valid audio stream — "
                        "falling back to original: %s",
                        (probe.stderr or "unknown error")[:200],
                    )
                    preprocessed_path = audio_path
                else:
                    logger.info(
                        "Preprocessed audio validated: %d bytes, %s",
                        file_size, probe.stdout.strip(),
                    )
        except Exception as val_err:
            logger.warning("Audio validation failed (%s) — using preprocessed file anyway", val_err)

    # ── Chunked transcription for long audio ──
    # Whisper's 30-second attention window causes accuracy degradation on long files.
    # Split into 10-min chunks with 30s overlap, then merge.
    chunks = _chunk_audio(preprocessed_path, chunk_duration=600, overlap=30)

    if len(chunks) > 1:
        logger.info("Using chunked transcription: %d chunks for long audio", len(chunks))
        all_chunk_segments: list[list[dict]] = []
        detected_lang = None

        wall_start = time.monotonic()
        if progress_state and progress_lock:
            with progress_lock:
                progress_state["start_time"] = wall_start

        for ci, chunk_info in enumerate(chunks):
            chunk_kwargs = dict(transcribe_kwargs)
            # For chunks after the first, use detected language (don't re-detect)
            if detected_lang and not language:
                chunk_kwargs["language"] = detected_lang

            try:
                chunk_segments_iter, chunk_info_obj = model.transcribe(
                    chunk_info["path"], **chunk_kwargs
                )
            except Exception as e:
                error_str = str(e).lower()
                if any(p in error_str for p in ["out of memory", "cuda", "cudamalloc", "oom"]):
                    logger.error(
                        "Whisper CUDA OOM during chunk %d/%d transcription "
                        "(model=%s, device=%s). Audio: %s",
                        ci + 1, len(chunks),
                        settings.WHISPER_MODEL, whisper_device_info.get("device", "?"),
                        chunk_info["path"],
                    )
                raise

            if ci == 0:
                detected_lang = chunk_info_obj.language
                _last_detected_language["lang"] = detected_lang
                logger.info("Detected language: %s (prob=%.2f)",
                            detected_lang, chunk_info_obj.language_probability)
                if progress_state and progress_lock:
                    with progress_lock:
                        progress_state["language"] = detected_lang

            chunk_raw = []
            try:
                for segment in chunk_segments_iter:
                    text = segment.text.strip()
                    word_list = None
                    if hasattr(segment, "words") and segment.words:
                        word_list = []
                        for w in segment.words:
                            word_text = w.word.strip()
                            if not word_text:
                                continue
                            # Skip phantom words with extremely low probability
                            word_prob = getattr(w, 'probability', 1.0)
                            if word_prob < 0.01:
                                logger.debug("Skipping low-probability word at %.2fs: '%s' (p=%.4f)",
                                             w.start + chunk_info["offset"], word_text, word_prob)
                                continue
                            word_list.append({
                                "start": round(w.start + chunk_info["offset"], 3),
                                "end": round(w.end + chunk_info["offset"], 3),
                                "word": word_text,
                            })

                    seg_end = segment.end + chunk_info["offset"]
                    seg_start = segment.start + chunk_info["offset"]
                    if word_list:
                        last_word_end = max(w["end"] for w in word_list)
                        if last_word_end > seg_end:
                            seg_end = last_word_end + 0.05

                    avg_lp = getattr(segment, 'avg_logprob', -1.0)
                    no_speech = getattr(segment, 'no_speech_prob', 0.0)
                    confidence = max(0.0, min(1.0, 1.0 + avg_lp))
                    if no_speech > 0.5:
                        confidence *= (1.0 - (no_speech - 0.5) * 2)

                    chunk_raw.append({
                        "start": seg_start,
                        "end": seg_end,
                        "text": text,
                        "words": word_list,
                        "confidence": round(confidence, 3),
                        "avg_logprob": round(avg_lp, 4) if avg_lp is not None else None,
                        "no_speech_prob": round(no_speech, 4) if no_speech is not None else None,
                    })
            except Exception as iter_err:
                error_str = str(iter_err).lower()
                is_oom = any(p in error_str for p in [
                    "out of memory", "cuda", "cudamalloc", "oom", "cublaslt",
                ])
                logger.error(
                    "Whisper %s during chunk %d/%d iteration "
                    "(model=%s, device=%s, beam=%d, collected %d segments before crash): %s",
                    "CUDA OOM" if is_oom else "error",
                    ci + 1, len(chunks),
                    settings.WHISPER_MODEL, whisper_device_info.get("device", "?"),
                    effective_beam, len(chunk_raw),
                    str(iter_err)[:300],
                )
                if is_oom:
                    logger.warning(
                        "Continuing with remaining chunks after CUDA OOM in chunk %d. "
                        "Collected %d segments from this chunk before failure.",
                        ci + 1, len(chunk_raw),
                    )
                else:
                    logger.warning(
                        "Non-CUDA error during chunk %d iteration — continuing with %d segments",
                        ci + 1, len(chunk_raw),
                    )

            all_chunk_segments.append(chunk_raw)

            # Update progress
            if progress_state and progress_lock:
                total_so_far = sum(len(c) for c in all_chunk_segments)
                last_end = chunk_raw[-1]["end"] if chunk_raw else 0
                with progress_lock:
                    progress_state["segments"] = total_so_far
                    progress_state["latest_end"] = last_end
                    progress_state["last_text"] = chunk_raw[-1]["text"][:80] if chunk_raw else ""
                    progress_state["language"] = detected_lang or ""
                    # Store flat list for partial recovery
                    progress_state["raw_segments"] = [
                        seg for chunk in all_chunk_segments for seg in chunk
                    ]

            if len(chunk_raw) == 0:
                _chunk_size = 0
                try:
                    import os as _os
                    _chunk_size = _os.path.getsize(chunk_info["path"])
                except Exception:
                    pass
                logger.warning(
                    "Chunk %d/%d: 0 SEGMENTS (offset=%.1fs, file=%s, size=%d bytes). "
                    "Possible causes: (1) CUDA OOM during beam search — reduce beam_size, "
                    "(2) audio chunk is silent/corrupt, (3) VAD filtered everything. "
                    "Model=%s, device=%s, beam=%d",
                    ci + 1, len(chunks), chunk_info["offset"],
                    chunk_info["path"], _chunk_size,
                    settings.WHISPER_MODEL, whisper_device_info.get("device", "?"),
                    effective_beam,
                )
            else:
                logger.info("Chunk %d/%d: %d segments (offset=%.1fs)",
                            ci + 1, len(chunks), len(chunk_raw), chunk_info["offset"])

        # Summary diagnostic: warn if most/all chunks produced nothing
        _empty_chunks = sum(1 for c in all_chunk_segments if len(c) == 0)
        _total_segs = sum(len(c) for c in all_chunk_segments)
        if _empty_chunks == len(all_chunk_segments):
            logger.error(
                "ALL %d chunks produced 0 segments (model=%s, device=%s, beam=%d). "
                "Whisper is completely failing on this audio. "
                "This is almost certainly a silent CUDA OOM — try beam_size=1 or CPU mode.",
                len(chunks), settings.WHISPER_MODEL,
                whisper_device_info.get("device", "?"), effective_beam,
            )
        elif _empty_chunks > 0:
            logger.warning(
                "%d/%d chunks produced 0 segments (%d total segments). "
                "Partial transcription — some chunks may have hit CUDA memory limits.",
                _empty_chunks, len(all_chunk_segments), _total_segs,
            )

        # Merge overlapping chunks
        raw_segments = _merge_chunk_segments(all_chunk_segments, overlap=30)

        # Cleanup chunk temp files
        import shutil, tempfile as _tmpmod
        chunk_dir = os.path.dirname(chunks[0]["path"]) if chunks[0]["path"] != preprocessed_path else None
        if chunk_dir and chunk_dir.startswith(_tmpmod.gettempdir()):
            shutil.rmtree(chunk_dir, ignore_errors=True)

        if progress_state and progress_lock:
            with progress_lock:
                progress_state["done"] = True
                progress_state["raw_segments"] = raw_segments
    else:
        # Original single-pass path for short audio
        try:
            segments_iter, info = model.transcribe(preprocessed_path, **transcribe_kwargs)
        except Exception as e:
            error_str = str(e).lower()
            if any(p in error_str for p in ["out of memory", "cuda", "cudamalloc", "oom"]):
                logger.error(
                    "Whisper CUDA OOM during transcription (model=%s, device=%s). "
                    "The model is too large for available VRAM. "
                    "Audio: %s, kwargs: beam=%s, best_of=%s",
                    settings.WHISPER_MODEL, whisper_device_info.get("device", "?"),
                    audio_path, transcribe_kwargs.get("beam_size"),
                    transcribe_kwargs.get("best_of"),
                )
            raise
        detected_lang = info.language
        _last_detected_language["lang"] = detected_lang
        logger.info(f"Detected language: {detected_lang} (prob={info.language_probability:.2f})")

        wall_start = time.monotonic()
        if progress_state and progress_lock:
            with progress_lock:
                progress_state["language"] = detected_lang
                progress_state["start_time"] = wall_start

        raw_segments = []
        try:
            for segment in segments_iter:
                text = segment.text.strip()
                word_list = None
                if hasattr(segment, "words") and segment.words:
                    word_list = []
                    for w in segment.words:
                        word_text = w.word.strip()
                        if not word_text:
                            continue
                        word_prob = getattr(w, 'probability', 1.0)
                        if word_prob < 0.01:
                            logger.debug("Skipping low-probability word at %.2fs: '%s' (p=%.4f)",
                                         w.start, word_text, word_prob)
                            continue
                        word_list.append({
                            "start": round(w.start, 3), "end": round(w.end, 3), "word": word_text,
                        })
                seg_end = segment.end
                if word_list:
                    last_word_end = max(w["end"] for w in word_list)
                    if last_word_end > seg_end:
                        seg_end = last_word_end + 0.05

                avg_lp = getattr(segment, 'avg_logprob', -1.0)
                no_speech = getattr(segment, 'no_speech_prob', 0.0)
                confidence = max(0.0, min(1.0, 1.0 + avg_lp))
                if no_speech > 0.5:
                    confidence *= (1.0 - (no_speech - 0.5) * 2)

                seg_dict = {
                    "start": segment.start,
                    "end": seg_end,
                    "text": text,
                    "words": word_list,
                    "confidence": round(confidence, 3),
                    "avg_logprob": round(avg_lp, 4) if avg_lp is not None else None,
                    "no_speech_prob": round(no_speech, 4) if no_speech is not None else None,
                }
                raw_segments.append(seg_dict)
                if progress_state and progress_lock:
                    with progress_lock:
                        progress_state["segments"] = len(raw_segments)
                        progress_state["latest_end"] = segment.end
                        progress_state["last_text"] = text[:80] if text else ""
                        progress_state["raw_segments"] = raw_segments
        except Exception as iter_err:
            error_str = str(iter_err).lower()
            is_oom = any(p in error_str for p in [
                "out of memory", "cuda", "cudamalloc", "oom", "cublaslt",
            ])
            logger.error(
                "Whisper %s during single-pass iteration "
                "(model=%s, device=%s, beam=%d, collected %d segments before crash): %s",
                "CUDA OOM" if is_oom else "error",
                settings.WHISPER_MODEL, whisper_device_info.get("device", "?"),
                effective_beam, len(raw_segments),
                str(iter_err)[:300],
            )

        if progress_state and progress_lock:
            with progress_lock:
                progress_state["done"] = True

    if not raw_segments:
        return []

    # Filter hallucinations and consolidate fragments before speaker assignment
    raw_segments = _filter_hallucinations(raw_segments, task=task)
    raw_segments = _consolidate_segments(raw_segments, task=task, is_animated=is_animated)
    raw_segments = _split_segments_at_sentence_boundaries(raw_segments, task=task)
    raw_segments, _dedup_removed = _dedupe_long_range(raw_segments)
    if not raw_segments:
        return []

    # ── Skip speaker diarization during initial transcription ──
    # All segments assigned to "Speaker 1" for maximum subtitle accuracy.
    # Diarization runs in post-processing when the user specifies speaker count
    # via POST /api/jobs/{job_id}/diarize — this gives the AI better guidance
    # and avoids diarization errors contaminating the initial transcript.
    transcript_segments = []
    for seg in raw_segments:
        words = None
        if seg.get("words"):
            words = [WordTimestamp(**w) for w in seg["words"]]
        transcript_segments.append(TranscriptSegment(
            start=round(seg["start"], 2),
            end=round(seg["end"], 2),
            text=seg["text"],
            speaker="Speaker 1",
            words=words,
            confidence=seg.get("confidence"),
            avg_logprob=seg.get("avg_logprob"),
            no_speech_prob=seg.get("no_speech_prob"),
        ))

    _last_diarization_method["method"] = "deferred"

    # Clean up preprocessed audio
    if preprocessed_path != audio_path:
        try:
            import os as _os
            _os.unlink(preprocessed_path)
        except OSError:
            pass

    logger.info(f"Transcription complete: {len(transcript_segments)} segments (diarization deferred to post)")
    return transcript_segments


def _extract_word_timestamps_sync(audio_path: str, language: str = "") -> list[WordTimestamp]:
    """Run Whisper with word_timestamps=True and return a flat sorted list of all words."""
    model = _get_whisper_model()
    kwargs = {
        "beam_size": settings.WHISPER_BEAM_SIZE,
        "best_of": 1,
        "vad_filter": settings.WHISPER_VAD_FILTER,
        "condition_on_previous_text": False,
        "word_timestamps": True,
    }
    if settings.WHISPER_VAD_FILTER:
        kwargs["vad_parameters"] = {
            "min_silence_duration_ms": 300,
            "speech_pad_ms": 600,
            "onset": 0.2,
            "min_speech_duration_ms": 100,
        }
    if language:
        kwargs["language"] = language

    segments_iter, _info = model.transcribe(audio_path, **kwargs)
    all_words: list[WordTimestamp] = []
    for segment in segments_iter:
        if hasattr(segment, "words") and segment.words:
            for w in segment.words:
                word_text = w.word.strip()
                if word_text:
                    all_words.append(WordTimestamp(
                        start=round(w.start, 3),
                        end=round(w.end, 3),
                        word=word_text,
                    ))
    all_words.sort(key=lambda w: w.start)
    return all_words


async def extract_word_timestamps(
    audio_path: str,
    language: str = "",
) -> list[WordTimestamp]:
    """Async wrapper: extract word-level timestamps from audio using Whisper."""
    import asyncio
    import functools

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _transcription_executor,
        functools.partial(_extract_word_timestamps_sync, audio_path, language=language),
    )


def _split_segments_on_word_boundaries(
    raw_segments: list[dict],
    word_split_gap: float,
    rate_change_threshold: float,
) -> list[dict]:
    """Split Whisper segments at strong intra-segment speaker boundaries.

    A Whisper segment can cover multiple speakers when the pause
    between them is short enough that VAD merged them into one
    block. When ``seg.words`` is present we can look at word-to-word
    gaps; a gap ≥ ``word_split_gap`` that also straddles a
    word-rate change ≥ ``rate_change_threshold`` is treated as a
    speaker boundary and the segment is split in two.

    No-op when ``words`` is missing. Preserves all non-word fields;
    copies them to both halves. The main pass (``_assign_speakers``)
    then sees independent segments and can attribute each to a
    different speaker.
    """
    def _wps(words):
        if not words:
            return 3.0
        dur = (words[-1]["end"] - words[0]["start"]) or 0.0
        if dur <= 0.5:
            return 3.0
        return len(words) / dur

    out: list[dict] = []
    for seg in raw_segments:
        words = seg.get("words") or []
        if len(words) < 4:
            out.append(seg)
            continue
        # Find the first internal gap that qualifies as a boundary.
        split_idx = -1
        for i in range(1, len(words) - 1):
            gap = float(words[i]["start"]) - float(words[i - 1]["end"])
            if gap < word_split_gap:
                continue
            left_rate = _wps(words[:i])
            right_rate = _wps(words[i:])
            denom = max(left_rate, right_rate, 0.1)
            if abs(left_rate - right_rate) / denom >= rate_change_threshold:
                split_idx = i
                break
        if split_idx < 0:
            out.append(seg)
            continue
        left_words = words[:split_idx]
        right_words = words[split_idx:]
        left_text = " ".join((w.get("word", "") or "").strip() for w in left_words).strip()
        right_text = " ".join((w.get("word", "") or "").strip() for w in right_words).strip()
        left_seg = dict(seg)
        left_seg["end"] = float(left_words[-1]["end"])
        left_seg["text"] = left_text or seg.get("text", "")
        left_seg["words"] = left_words
        right_seg = dict(seg)
        right_seg["start"] = float(right_words[0]["start"])
        right_seg["text"] = right_text or seg.get("text", "")
        right_seg["words"] = right_words
        out.append(left_seg)
        out.append(right_seg)
    return out


def _assign_speakers(
    raw_segments: list[dict],
    removed_intervals: list[tuple[float, float]] | None = None,
) -> list[TranscriptSegment]:
    """Assign speaker labels using enhanced pause-based turn detection.

    Caps speakers at MAX_HEURISTIC_SPEAKERS to avoid absurd counts (e.g. 34).
    Tracks speaker history and speech rate per speaker to make smarter toggle
    decisions for 3+ person conversations.

    ``removed_intervals`` is an optional list of `(start, end)` tuples
    for segments that ``_dedupe_long_range`` removed earlier in the
    pipeline. We subtract the duration of any interval that falls
    inside the gap between two surviving segments before checking the
    NEW_SPEAKER_GAP threshold; this stops a removed verbatim repeat
    from inflating a 1.5 s pause into a 9 s pause and triggering a
    spurious speaker flip on the duplicate.
    """
    # Speaker cap: configurable via settings, defaults to 20 (effectively unlimited
    # for most content). The old default of 8 was too restrictive for panel shows,
    # roundtables, and multi-person podcasts. Setting DIARIZATION_MAX_SPEAKERS=0
    # means auto (use 20 as practical limit to avoid noise).
    MAX_HEURISTIC_SPEAKERS = max(
        2,
        settings.DIARIZATION_MAX_SPEAKERS if settings.DIARIZATION_MAX_SPEAKERS > 0 else 20,
    )
    # Thresholds tightened for V2. Whisper large-v3 with VAD produces
    # 0.4-1.0 s intra-speaker gaps and 2-4 s inter-speaker gaps on
    # typical conversational content. The previous 1.2 s / 5.0 s
    # cutoffs meant fast back-and-forth never crossed either
    # threshold, which is the dominant path to the "every speaker is
    # Speaker 1" bug in the heuristic-fallback tier.
    TURN_GAP = 0.6
    NEW_SPEAKER_GAP = 2.5
    # Word-level split threshold: when seg.words is present, a gap >=
    # WORD_SPLIT_GAP inside a single Whisper segment AND a word-rate
    # change >= RATE_CHANGE_THRESHOLD produces an intra-segment
    # speaker boundary. Biggest accuracy win for conversational clips
    # where Whisper merges two speakers into one segment.
    WORD_SPLIT_GAP = 0.5
    MONOLOGUE_DURATION = 15.0
    INTERJECTION_WORDS = 4
    RATE_CHANGE_THRESHOLD = 0.4

    if not raw_segments:
        return []

    # Pre-pass: split any Whisper segment that has a large intra-gap
    # AND a rate change between the halves. Operates on a flat list
    # of virtual segments so the main loop stays unchanged.
    raw_segments = _split_segments_on_word_boundaries(
        raw_segments, WORD_SPLIT_GAP, RATE_CHANGE_THRESHOLD,
    )

    # Pre-sort the removed-intervals list so we can compute "removed
    # duration inside (gap_start, gap_end)" in O(log n) per segment.
    # Most calls pass either None or a short list (a handful of
    # removals on a long video), so the loop overhead is negligible.
    sorted_removed: list[tuple[float, float]] = (
        sorted(removed_intervals) if removed_intervals else []
    )

    def _removed_duration_in(gap_start: float, gap_end: float) -> float:
        if not sorted_removed or gap_end <= gap_start:
            return 0.0
        total = 0.0
        for r_start, r_end in sorted_removed:
            if r_end <= gap_start:
                continue
            if r_start >= gap_end:
                break
            overlap = min(r_end, gap_end) - max(r_start, gap_start)
            if overlap > 0:
                total += overlap
        return total

    transcript_segments = []
    current_speaker = 1
    speakers_seen = 1
    speaker_history: list[int] = [1]
    # Seed Speaker 1's rate with the first segment so
    # ``_most_likely_existing_speaker`` has a meaningful rate to
    # compare against on segment 2. Previously the dict started
    # empty → every rate-match call degenerated to ``current_speaker
    # == 1`` and every subsequent segment inherited the label.
    speaker_rates: dict[int, list[float]] = {1: []}
    if raw_segments:
        _first_dur = raw_segments[0]["end"] - raw_segments[0]["start"]
        _first_words = (
            len(raw_segments[0]["text"].split())
            if raw_segments[0].get("text") else 0
        )
        if _first_dur > 0.5 and _first_words > 0:
            speaker_rates[1].append(_first_words / _first_dur)

    def _words_per_sec(seg: dict) -> float:
        duration = seg["end"] - seg["start"]
        if duration <= 0:
            return 3.0
        word_count = len(seg["text"].split()) if seg["text"] else 0
        return word_count / duration if duration > 0.5 else 3.0

    def _avg_rate(speaker: int) -> float:
        rates = speaker_rates.get(speaker, [])
        return sum(rates) / len(rates) if rates else 3.0

    def _most_likely_existing_speaker(rate: float) -> int:
        best_speaker = current_speaker
        best_diff = float('inf')
        for sp, rates in speaker_rates.items():
            if not rates:
                continue
            avg = sum(rates) / len(rates)
            diff = abs(avg - rate)
            if diff < best_diff:
                best_diff = diff
                best_speaker = sp
        return best_speaker

    for i, seg in enumerate(raw_segments):
        seg_rate = _words_per_sec(seg)
        seg_word_count = len(seg["text"].split()) if seg["text"] else 0

        if i > 0:
            raw_gap_start = raw_segments[i - 1]["end"]
            raw_gap_end = seg["start"]
            gap = raw_gap_end - raw_gap_start
            # Subtract any duration that ``_dedupe_long_range`` removed
            # from inside this gap so a deleted duplicate doesn't fake
            # a NEW_SPEAKER_GAP-sized silence.
            if sorted_removed and gap > 0:
                gap -= _removed_duration_in(raw_gap_start, raw_gap_end)
                if gap < 0:
                    gap = 0.0
            prev_duration = raw_segments[i - 1]["end"] - raw_segments[i - 1]["start"]
            prev_rate = _words_per_sec(raw_segments[i - 1])

            if gap >= NEW_SPEAKER_GAP:
                # On a NEW_SPEAKER_GAP crossing, prefer to MINT a new
                # speaker. Only route back to an existing speaker
                # when there is a different one whose rate is a
                # strong match — not the current one, which is the
                # bug path: on segment 2 with only Speaker 1 in
                # ``speaker_rates``, the old code always matched
                # Speaker 1 and collapsed every subsequent segment.
                rate_match = _most_likely_existing_speaker(seg_rate)
                rate_diff = abs(seg_rate - _avg_rate(rate_match))
                strong_match = (
                    rate_match != current_speaker
                    and rate_diff < RATE_CHANGE_THRESHOLD
                )
                if strong_match:
                    current_speaker = rate_match
                elif speakers_seen < MAX_HEURISTIC_SPEAKERS:
                    speakers_seen += 1
                    current_speaker = speakers_seen
                else:
                    current_speaker = _most_likely_existing_speaker(seg_rate)
            elif gap >= TURN_GAP:
                other = None
                for sp in reversed(speaker_history):
                    if sp != current_speaker:
                        other = sp
                        break
                if other:
                    current_speaker = other
                elif speakers_seen < MAX_HEURISTIC_SPEAKERS:
                    speakers_seen += 1
                    current_speaker = speakers_seen
                else:
                    current_speaker = _most_likely_existing_speaker(seg_rate)
            elif (prev_duration > MONOLOGUE_DURATION
                  and seg_word_count <= INTERJECTION_WORDS and gap < 0.5):
                other = None
                for sp in reversed(speaker_history):
                    if sp != current_speaker:
                        other = sp
                        break
                if other:
                    current_speaker = other
                elif speakers_seen < MAX_HEURISTIC_SPEAKERS:
                    speakers_seen += 1
                    current_speaker = speakers_seen
                else:
                    current_speaker = _most_likely_existing_speaker(seg_rate)
            elif abs(seg_rate - prev_rate) > RATE_CHANGE_THRESHOLD * max(seg_rate, prev_rate, 0.1):
                rate_match = _most_likely_existing_speaker(seg_rate)
                if rate_match != current_speaker:
                    current_speaker = rate_match

        speaker_history.append(current_speaker)
        if len(speaker_history) > 20:
            speaker_history = speaker_history[-20:]

        if current_speaker not in speaker_rates:
            speaker_rates[current_speaker] = []
        speaker_rates[current_speaker].append(seg_rate)
        if len(speaker_rates[current_speaker]) > 10:
            speaker_rates[current_speaker] = speaker_rates[current_speaker][-10:]

        words = None
        if seg.get("words"):
            words = [WordTimestamp(**w) for w in seg["words"]]

        transcript_segments.append(TranscriptSegment(
            start=round(seg["start"], 2),
            end=round(seg["end"], 2),
            text=seg["text"],
            speaker=f"Speaker {current_speaker}",
            words=words,
            confidence=seg.get("confidence"),
            avg_logprob=seg.get("avg_logprob"),
            no_speech_prob=seg.get("no_speech_prob"),
        ))

    return transcript_segments


_WHISPER_BOILERPLATE = {
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "like and subscribe",
    "don't forget to subscribe",
    "see you in the next video",
    "bye bye",
    "thanks for listening",
    "music playing",
    "music",
    "applause",
    "subtitles by",
    "captions by",
}


# ── Long-range duplicate filter ─────────────────────────────────────────
#
# Whisper large-v3 with condition_on_previous_text=True occasionally
# loops the same dialogue verbatim minutes later in the file. Every
# pre-existing dedup check in this module caps its window at 30 s, so
# repeats from 60 s up to ~10 min away slip through. This helper looks
# at the WHOLE history (regardless of distance) and drops verbatim
# repeats while protecting short legitimate repeats ("right.", "no no
# no", translate-mode backchannels).

# Punctuation stripped to build word fingerprints. We KEEP apostrophes
# inside contractions so "don't" and "do not" don't false-positive
# against each other.
_FINGERPRINT_STRIP_RE = re.compile(r"[^\w\s'\u2019]+")


def _fingerprint_text(text: str) -> tuple[frozenset, str, list[str]]:
    """Return (token_set, normalized_text, token_list) for fuzzy compare."""
    if not text:
        return (frozenset(), "", [])
    norm = _FINGERPRINT_STRIP_RE.sub(" ", text.lower())
    tokens = norm.split()
    return (frozenset(tokens), " ".join(tokens), tokens)


def _dedupe_long_range(
    segments: list[dict],
    min_words: int = 5,
    min_chars: int = 20,
    jaccard_threshold: float = 0.92,
    exact_threshold: float = 0.97,
) -> tuple[list[dict], list[tuple[float, float]]]:
    """Drop verbatim repeats anywhere in the segment stream.

    Looks at every previously-kept segment regardless of time distance
    (the fix for AoT S3E10 where the same line repeated 8 minutes
    later) and drops any candidate whose word fingerprint matches by
    Jaccard or whose normalized text matches by SequenceMatcher.

    Short candidates (`< min_words` AND `< min_chars`) skip the check
    entirely so legitimate repetition like "right.", "yeah.", "no no
    no", and translate-mode backchannels stay intact. We also allow
    up to two occurrences of any short fingerprint (< 8 tokens) to
    cover host tics.

    Returns:
        (kept_segments, removed_intervals)

    ``removed_intervals`` is a list of `(start, end)` tuples for each
    dropped segment so the speaker assigner can subtract their
    duration from gap calculations and avoid spurious speaker flips.
    """
    if not segments:
        return segments, []

    from difflib import SequenceMatcher

    kept: list[dict] = []
    removed_intervals: list[tuple[float, float]] = []
    fingerprint_counts: dict[frozenset, int] = {}

    # Keep parallel arrays so we don't recompute fingerprints on every loop.
    kept_fps: list[tuple[frozenset, str, list[str]]] = []

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            kept.append(seg)
            continue

        fp_set, fp_norm, fp_tokens = _fingerprint_text(text)
        token_count = len(fp_tokens)
        char_count = len(fp_norm)

        # Too short to risk a false positive — short repeats are
        # legitimate conversational tics ("right.", "no no no", "yes.").
        if token_count < min_words and char_count < min_chars:
            kept.append(seg)
            kept_fps.append((fp_set, fp_norm, fp_tokens))
            continue

        is_dup = False
        match_at: float | None = None

        for prev_seg, (prev_set, prev_norm, prev_tokens) in zip(kept, kept_fps):
            if not prev_set:
                continue
            # Jaccard over token sets.
            inter = len(fp_set & prev_set)
            union = len(fp_set | prev_set)
            jaccard = (inter / union) if union else 0.0
            if jaccard >= jaccard_threshold:
                is_dup = True
                match_at = float(prev_seg.get("start", 0.0))
                break
            # Cheap pre-screen for SequenceMatcher: only run it when the
            # token sets overlap heavily enough that ratio() COULD clear
            # exact_threshold. SequenceMatcher.ratio() is O(n*m); skipping
            # obviously-different texts keeps this pass cheap on long videos.
            if jaccard >= 0.6:
                ratio = SequenceMatcher(None, fp_norm, prev_norm).ratio()
                if ratio >= exact_threshold:
                    is_dup = True
                    match_at = float(prev_seg.get("start", 0.0))
                    break

        # Allow up to 2 total occurrences of any short-ish fingerprint
        # to protect "let's go, let's go" hosts. Short here = under 8
        # tokens; the floor above already protected the truly tiny.
        # ``count`` is the number of copies ALREADY kept; allowing one
        # more gives a max of 2 in the output.
        if is_dup and token_count < 8:
            count = fingerprint_counts.get(fp_set, 0)
            if count < 2:
                is_dup = False

        if is_dup:
            logger.warning(
                "Long-range dedup: removed verbatim repeat at %.1fs "
                "(orig at %.1fs): %s",
                float(seg.get("start", 0.0)),
                float(match_at) if match_at is not None else -1.0,
                text[:80],
            )
            removed_intervals.append((
                float(seg.get("start", 0.0)),
                float(seg.get("end", 0.0)),
            ))
            continue

        kept.append(seg)
        kept_fps.append((fp_set, fp_norm, fp_tokens))
        fingerprint_counts[fp_set] = fingerprint_counts.get(fp_set, 0) + 1

    if removed_intervals:
        logger.info(
            "Long-range dedup: %d → %d segments (removed %d verbatim repeats)",
            len(segments), len(kept), len(removed_intervals),
        )
    return kept, removed_intervals


def _filter_hallucinations(raw_segments: list[dict], task: str = "transcribe") -> list[dict]:
    """Remove Whisper hallucination segments.

    Detects and filters:
    - Non-speech segments (high no_speech_prob + low confidence)
    - Whisper boilerplate phrases
    - Backward-jumping timestamps (temporal ordering violations)
    - Abnormally long single segments (>1500 chars = likely runaway)
    - Repeated n-grams (looping text like "Thank you. Thank you. Thank you.")
    - Segments that are near-exact duplicates of the previous segment (sequence-based)

    When task='translate', applies looser thresholds because English translations
    of non-English audio produce shorter text for the same audio duration.
    """
    is_translate = (task == "translate")
    if not raw_segments:
        return raw_segments

    filtered = []
    prev_text = ""

    for seg in raw_segments:
        text = seg["text"].strip()

        # Skip empty segments
        if not text:
            continue

        # Check 0a: Non-speech segment (silence/music hallucination)
        no_speech = seg.get("no_speech_prob", 0.0)
        confidence = seg.get("confidence", 1.0)
        if no_speech and no_speech > 0.9 and confidence is not None and confidence < 0.1:
            logger.warning(
                "Hallucination filter: removed non-speech segment at %.1fs (no_speech=%.2f, conf=%.2f): %s...",
                seg["start"], no_speech, confidence, text[:60],
            )
            continue

        # Check 0b: Whisper boilerplate phrases — only at transcript edges with low confidence
        boilerplate_text = text.lower().strip().rstrip('.!')
        if boilerplate_text in _WHISPER_BOILERPLATE:
            is_edge = (not filtered) or (seg is raw_segments[-1]) or (
                len(raw_segments) > 1 and seg is raw_segments[-2])
            is_low_conf = confidence is not None and confidence < 0.5
            if is_edge and is_low_conf:
                logger.warning(
                    "Hallucination filter: removed boilerplate at %.1fs: %s",
                    seg["start"], text[:60],
                )
                continue

        # Check 0e: Prompt echo detection — catches initial_prompt being
        # hallucinated as transcription during quiet audio sections.
        _PROMPT_ECHO_PATTERNS = [
            "this is a japanese conversation",
            "this is a conversation translated",
            "translated to natural english",
            "translate names as-is",
            "use complete sentences",
        ]
        text_lower_check = text.lower()
        if any(pat in text_lower_check for pat in _PROMPT_ECHO_PATTERNS):
            logger.warning(
                "Hallucination filter: prompt echo at %.1fs: %s...",
                seg["start"], text[:80],
            )
            continue

        # Check 0d: Text-to-duration ratio — catches ghosts that have low no_speech_prob
        # Translate: English translations are structurally shorter than source audio
        seg_duration = seg["end"] - seg["start"]
        if seg_duration > 0:
            chars_per_sec = len(text) / seg_duration
            min_ratio_duration = 60 if is_translate else 15
            min_chars_ratio = 0.1 if is_translate else 1.0
            if seg_duration > min_ratio_duration and chars_per_sec < min_chars_ratio:
                logger.warning(
                    "Hallucination filter: ghost (ratio) at %.1fs (%.0fs, %.2f c/s): %s...",
                    seg["start"], seg_duration, chars_per_sec, text[:60],
                )
                continue
            mega_threshold = 300 if is_translate else 120
            mega_min_chars = 10 if is_translate else 200
            if seg_duration > mega_threshold and len(text) < mega_min_chars:
                logger.warning(
                    "Hallucination filter: mega-ghost at %.1fs (%.0fs, %d chars): %s...",
                    seg["start"], seg_duration, len(text), text[:60],
                )
                continue

        # Check 0c: Temporal ordering — segment start must not jump backward
        if filtered and seg["start"] < filtered[-1]["start"]:
            logger.warning(
                "Hallucination filter: removed backward-jumping segment at %.1fs (prev started %.1fs): %s...",
                seg["start"], filtered[-1]["start"], text[:60],
            )
            continue

        # Check 1: Abnormally long segment (Whisper runaway)
        if len(text) > 1500:
            logger.warning(
                "Hallucination filter: removed runaway segment at %.1fs (%d chars): %s...",
                seg["start"], len(text), text[:80],
            )
            continue

        # Check 2: Repeated n-grams
        # For CJK languages (no spaces between words), use character-level n-grams
        # For space-delimited languages, use word-level trigrams
        _CJK_RANGES = (
            ('\u4e00', '\u9fff'),   # CJK Unified Ideographs
            ('\u3040', '\u309f'),   # Hiragana
            ('\u30a0', '\u30ff'),   # Katakana
            ('\uac00', '\ud7af'),   # Hangul Syllables
        )
        is_cjk = any(
            any(lo <= ch <= hi for lo, hi in _CJK_RANGES)
            for ch in text[:50]
        )

        if is_cjk:
            # Character-level n-gram detection for CJK
            chars = text.replace(" ", "")
            ngram_size = 6
            if len(chars) >= ngram_size * 3:
                ngrams = [chars[i:i+ngram_size] for i in range(len(chars) - ngram_size + 1)]
                ngram_counts: dict[str, int] = {}
                for ng in ngrams:
                    ngram_counts[ng] = ngram_counts.get(ng, 0) + 1
                max_repeat = max(ngram_counts.values()) if ngram_counts else 0
                if max_repeat >= 3 and max_repeat / len(ngrams) > 0.3:
                    logger.warning(
                        "Hallucination filter: removed CJK looping segment at %.1fs: %s...",
                        seg["start"], text[:80],
                    )
                    continue
        else:
            # Word-level trigram detection for space-delimited languages
            words = text.lower().split()
            if len(words) >= 9:
                trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
                trigram_counts_w: dict[tuple, int] = {}
                for tg in trigrams:
                    trigram_counts_w[tg] = trigram_counts_w.get(tg, 0) + 1
                max_repeat = max(trigram_counts_w.values()) if trigram_counts_w else 0
                if max_repeat >= 3 and max_repeat / len(trigrams) > 0.4:
                    logger.warning(
                        "Hallucination filter: removed looping segment at %.1fs: %s...",
                        seg["start"], text[:80],
                    )
                    continue

        # Check 3: Near-duplicate of previous segment (sequence-based)
        if prev_text and text and len(text.split()) > 3:
            from difflib import SequenceMatcher
            ratio = SequenceMatcher(None, prev_text.lower(), text.lower()).ratio()
            seg_duration = seg["end"] - seg["start"]
            # Only filter near-identical duplicates at nearly the same timestamp
            # (0.85 was catching legitimate repeated phrases like "let's go, let's go")
            if ratio > 0.92 and seg_duration < 3.0 and abs(seg["start"] - filtered[-1]["start"]) < 2.0:
                logger.warning(
                    "Hallucination filter: removed duplicate segment at %.1fs (%.0f%% similar): %s...",
                    seg["start"], ratio * 100, text[:60],
                )
                continue

        # Check 3b: Exact duplicate of any segment in the last 10
        # Translate: backchannel responses ("Is that so?", "Really?") repeat
        # legitimately — only filter if timestamps are also close
        if len(filtered) >= 2:
            recent_texts = {s["text"].strip().lower() for s in filtered[-10:]}
            if text.lower() in recent_texts:
                if is_translate:
                    is_time_dup = False
                    for recent in filtered[-10:]:
                        if (recent["text"].strip().lower() == text.lower()
                                and abs(seg["start"] - recent["end"]) < 5.0):
                            is_time_dup = True
                            break
                    if not is_time_dup:
                        filtered.append(seg)
                        prev_text = text
                        continue  # Keep it — legitimate repeat at different time
                logger.warning(
                    "Hallucination filter: near-dup (window) at %.1fs: %s...",
                    seg["start"], text[:60],
                )
                continue

        # Check 3c: Fuzzy near-duplicate within 30s (Jaccard word overlap)
        # large-v3 sometimes re-transcribes the same content 10-30s later
        # Translate: skip for short phrases (≤5 words) — backchannel responses
        # repeat legitimately. Use higher threshold for longer translated text.
        _is_fuzzy_dup = False
        _skip_jaccard = is_translate and len(text.split()) <= 5
        if not _skip_jaccard and len(filtered) >= 2 and len(text.split()) > 3 and len(text) > 15:
            text_words = set(text.lower().split())
            jaccard_threshold = 0.85 if is_translate else 0.7
            for recent in filtered[-10:]:
                recent_text = recent["text"].strip()
                time_gap = abs(seg["start"] - recent["start"])
                # 30s ceiling for close-range fuzzy repeats. The real
                # long-range fix is _dedupe_long_range with its own
                # (stricter) thresholds; this is belt-and-braces inside
                # _filter_hallucinations.
                if time_gap > 30 or time_gap < 2:
                    continue
                recent_words = set(recent_text.lower().split())
                if text_words and recent_words:
                    jaccard = len(text_words & recent_words) / len(text_words | recent_words)
                    if jaccard > jaccard_threshold:
                        logger.warning(
                            "Hallucination filter: fuzzy dup at %.1fs (%.0f%% similar to %.1fs): %s...",
                            seg["start"], jaccard * 100, recent["start"], text[:60],
                        )
                        _is_fuzzy_dup = True
                        break
        if _is_fuzzy_dup:
            continue

        filtered.append(seg)
        prev_text = text

    removed = len(raw_segments) - len(filtered)
    if removed > 0:
        logger.info("Hallucination filter: removed %d/%d segments", removed, len(raw_segments))
    return filtered


def _consolidate_segments(
    segments: list[dict],
    max_gap: float = 2.0,
    task: str = "transcribe",
    is_animated: bool = False,
) -> list[dict]:
    """Consolidate over-fragmented Whisper output into natural subtitle-length segments.

    Merges micro-segments (<0.5s), consecutive short fragments (1-3 words),
    and removes near-duplicate text within 30 seconds.

    When task='translate', uses gentler merging to preserve short backchannel
    responses ("Yes.", "Really?", "I see.") as separate segments.

    When is_animated=True, the near-duplicate Jaccard threshold is raised to
    0.90 (from 0.70) and the time window shrinks to 10s because anime
    characters often repeat words or phrases for emphasis, and narrators
    sometimes echo dialogue — these are intentional, not Whisper echoes
    (Problem 3B in the anime reframing audit).
    """
    if len(segments) <= 1:
        return segments

    is_translate = (task == "translate")

    # ── Pass 1: Merge micro-segments into neighbors ──
    # Translate: only merge truly tiny segments (< 0.2s, single word)
    micro_threshold = 0.2 if is_translate else 0.5
    micro_max_words = 1 if is_translate else 3

    merged = []
    i = 0
    while i < len(segments):
        seg = dict(segments[i])
        duration = seg["end"] - seg["start"]
        text = seg.get("text", "").strip()

        # Merge very short segment forward into next
        if duration < micro_threshold and len(text.split()) <= micro_max_words and i + 1 < len(segments):
            next_seg = segments[i + 1]
            gap = next_seg["start"] - seg["end"]
            if gap < max_gap:
                merged_seg = dict(next_seg)
                merged_seg["start"] = seg["start"]
                merged_seg["text"] = text + " " + next_seg.get("text", "").strip()
                if seg.get("words") and next_seg.get("words"):
                    merged_seg["words"] = list(seg["words"]) + list(next_seg["words"])
                elif next_seg.get("words"):
                    merged_seg["words"] = next_seg["words"]
                merged.append(merged_seg)
                i += 2
                continue

        # Merge very short segment backward into previous
        if duration < micro_threshold and len(text.split()) <= micro_max_words and merged:
            prev = merged[-1]
            gap = seg["start"] - prev["end"]
            if gap < max_gap:
                prev["end"] = seg["end"]
                prev["text"] = prev.get("text", "").strip() + " " + text
                if prev.get("words") and seg.get("words"):
                    prev["words"] = list(prev["words"]) + list(seg["words"])
                i += 1
                continue

        merged.append(seg)
        i += 1

    # ── Pass 2: Merge consecutive short fragments into sentences ──
    # Translate: skip this pass — short fragments are individual backchannel responses
    if is_translate:
        consolidated = merged
    else:
        consolidated = []
        i = 0
        while i < len(merged):
            seg = dict(merged[i])
            text = seg.get("text", "").strip()
            words = text.split()

            if len(words) <= 3 and i + 1 < len(merged):
                combined_text = text
                combined_end = seg["end"]
                combined_words = list(seg.get("words", []) or [])
                j = i + 1

                while j < len(merged):
                    next_seg = merged[j]
                    next_text = next_seg.get("text", "").strip()
                    gap = next_seg["start"] - combined_end

                    if gap > max_gap:
                        break
                    if len(combined_text.split()) + len(next_text.split()) > 15:
                        break
                    if len(combined_text) + len(next_text) > 80:
                        break
                    if len(next_text.split()) > 3 and len(combined_text.split()) > 3:
                        break
                    # Don't merge across a sentence boundary already present
                    # in combined_text. Stops "Hold on, Rod." from being glued
                    # to "Don't shoot.": _split_segments_at_sentence_boundaries
                    # would only have to undo that work later.
                    if combined_text and combined_text.rstrip()[-1:] in ".!?":
                        break

                    combined_text = combined_text + " " + next_text
                    combined_end = next_seg["end"]
                    if next_seg.get("words"):
                        combined_words.extend(next_seg["words"])
                    j += 1

                if j > i + 1:
                    seg["end"] = combined_end
                    seg["text"] = combined_text
                    if combined_words:
                        seg["words"] = combined_words
                    i = j
                    consolidated.append(seg)
                    continue

            consolidated.append(seg)
            i += 1

    # ── Pass 3: Remove near-duplicate text within 30 seconds ──
    # Translate: skip dedup pass — already handled by _filter_hallucinations
    # Animated: stricter threshold + smaller window — intentional repeats
    # (dramatic emphasis, narrator echo) are common in anime and should
    # be preserved; only catch actual Whisper echoes.
    _dedup_jaccard = 0.90 if is_animated else 0.7
    _dedup_window = 10 if is_animated else 180
    if is_translate:
        deduped = consolidated
    else:
        deduped = []
        for seg in consolidated:
            text = seg.get("text", "").strip().lower()
            is_dup = False

            for recent in deduped[-10:]:
                recent_text = recent.get("text", "").strip().lower()
                time_gap = abs(seg["start"] - recent["start"])
                if time_gap > _dedup_window:
                    continue
                if len(text) > 10 and len(recent_text) > 10:
                    if text in recent_text or recent_text in text:
                        is_dup = True
                        break
                    words_a = set(text.split())
                    words_b = set(recent_text.split())
                    if words_a and words_b:
                        jaccard = len(words_a & words_b) / len(words_a | words_b)
                        if jaccard > _dedup_jaccard:
                            is_dup = True
                            break

            if is_dup:
                logger.info("Consolidation: removed near-dup at %.1fs: %s", seg["start"], seg.get("text", "")[:60])
                continue
            deduped.append(seg)

    removed = len(segments) - len(deduped)
    if removed > 0:
        logger.info("Segment consolidation: %d → %d segments (removed %d)", len(segments), len(deduped), removed)
    return deduped


# ── Sentence-boundary splitter ──────────────────────────────────────────
#
# Whisper sometimes lumps 2-4 distinct utterances under a single segment
# when its decoder backs up after a chunk boundary. The consolidate pass
# above can also concatenate short fragments into a multi-sentence chunk.
# This helper undoes both: it finds sentence boundaries inside `text`,
# slices `words` (when present) so each output segment carries the right
# per-word timing, and otherwise falls back to proportional time
# splitting by character count. Sub-segments that would be too short
# fold forward into their successor instead of fragmenting.

# Latin sentence-ending punctuation followed by whitespace + uppercase.
# Tolerates trailing quote / paren marks. Ellipses (... or …) are
# matched as a single boundary by the standalone _ELLIPSIS_RE pre-pass
# below; we don't want to split on the first '.' of "..." then create a
# zero-length sub-segment for the next two.
_LATIN_SENTENCE_END_RE = re.compile(
    r"(?<=[.!?])[\"\u201d\u2019')\]]?\s+(?=[\"\u201c\u2018'(\[]?[A-Z])"
)
# Full-width sentence enders for CJK content. CJK doesn't gate on a
# following uppercase letter (CJK has no case), so we just split after
# the punctuation if there's any text following.
_CJK_SENTENCE_END_RE = re.compile(r"(?<=[\u3002\uff01\uff1f])(?=\S)")
# Collapse '...' and '…' into a sentinel before we run the splitter,
# then restore them after, so they only count as a SINGLE boundary.
_ELLIPSIS_RE = re.compile(r"\.{3,}|\u2026")
_ELLIPSIS_SENTINEL = "\u0001ELLIPSIS\u0001"

_SENTENCE_SPLIT_CJK_RANGES = (
    ("\u4e00", "\u9fff"),   # CJK Unified Ideographs
    ("\u3040", "\u309f"),   # Hiragana
    ("\u30a0", "\u30ff"),   # Katakana
    ("\uac00", "\ud7af"),   # Hangul Syllables
)


def _looks_cjk(text: str) -> bool:
    """Return True if any of the first 50 chars fall in a CJK block."""
    sample = text[:50]
    return any(
        any(lo <= ch <= hi for lo, hi in _SENTENCE_SPLIT_CJK_RANGES)
        for ch in sample
    )


def _find_sentence_split_indices(text: str, cjk: bool) -> list[int]:
    """Return character offsets at which to split `text` into sentences.

    Each returned index is the START of the next sentence. The list does
    NOT include 0 or len(text); callers handle the bounds. Ellipses are
    counted as a single boundary, not three.
    """
    if not text or len(text) < 2:
        return []
    # Replace ellipses with a sentinel so the regex sees one '.' worth
    # of boundary signal. We restore the original characters when we
    # slice out the actual sub-text below.
    masked = _ELLIPSIS_RE.sub(_ELLIPSIS_SENTINEL, text)

    indices: list[int] = []
    if cjk:
        for m in _CJK_SENTENCE_END_RE.finditer(masked):
            indices.append(m.start())
    else:
        for m in _LATIN_SENTENCE_END_RE.finditer(masked):
            # Split start = end of the matched whitespace
            indices.append(m.end())
    return indices


def _split_segments_at_sentence_boundaries(
    segments: list[dict],
    min_split_duration: float = 0.40,
    min_split_chars: int = 4,
    task: str = "transcribe",
) -> list[dict]:
    """Split multi-sentence segments into per-sentence sub-segments.

    Whisper sometimes returns 2-4 distinct utterances spoken seconds
    apart in the same segment, with only the start of the first
    utterance recorded. This helper reverses that:

    1. For each input segment, locate sentence boundaries inside `text`.
    2. If the segment carries per-word timestamps, split at the
       corresponding word index so each sub-segment has accurate
       start/end timing.
    3. Otherwise, split proportionally by character count using the
       segment's existing start/end as bounds.
    4. Sub-segments shorter than ``min_split_duration`` seconds OR
       ``min_split_chars`` characters re-merge forward (preferred) or
       backward to avoid over-fragmenting stutters like "I— I think".
    5. Per-segment scores (confidence / avg_logprob / no_speech_prob)
       carry through unchanged.

    For CJK content, splits on full-width 。！？ instead of Latin
    punctuation (mixed-script subtitles use '.' decoratively and would
    otherwise be over-split).
    """
    if not segments:
        return segments

    out: list[dict] = []
    changed = 0

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            out.append(seg)
            continue

        cjk = _looks_cjk(text)
        boundaries = _sentence_boundary_offsets(text, cjk)
        if len(boundaries) < 1:
            out.append(seg)
            continue

        # Build sub-text spans from boundary offsets.
        sub_texts: list[str] = []
        prev = 0
        for b in boundaries:
            chunk = text[prev:b].strip()
            if chunk:
                sub_texts.append(chunk)
            prev = b
        tail = text[prev:].strip()
        if tail:
            sub_texts.append(tail)

        if len(sub_texts) < 2:
            out.append(seg)
            continue

        # ── Build sub-segments ──
        sub_segments: list[dict] = []
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        carry_keys = ("confidence", "avg_logprob", "no_speech_prob")
        carry = {k: seg.get(k) for k in carry_keys if k in seg}

        words = seg.get("words")
        if words:
            sub_segments = _split_using_word_timestamps(
                seg=seg, sub_texts=sub_texts, words=list(words),
                seg_start=seg_start, seg_end=seg_end, carry=carry,
            )
        if not sub_segments:
            # Either no words OR the word splitter couldn't align; fall
            # back to proportional time splitting by character count.
            sub_segments = _split_proportionally(
                sub_texts=sub_texts,
                seg_start=seg_start, seg_end=seg_end,
                carry=carry,
            )

        # ── Re-merge under-floor sub-segments ──
        sub_segments = _merge_undersized_subsegments(
            sub_segments,
            min_split_duration=min_split_duration,
            min_split_chars=min_split_chars,
        )

        if len(sub_segments) <= 1:
            # Folding ate every split — keep the original segment so we
            # don't accidentally drop carry-over fields.
            out.append(seg)
            continue

        out.extend(sub_segments)
        changed += 1

    if changed:
        logger.info(
            "Sentence splitter: %d → %d segments", len(segments), len(out),
        )
    return out


def _sentence_boundary_offsets(text: str, cjk: bool) -> list[int]:
    """Return a list of character offsets at which `text` should split.

    Each offset is the START of the next sentence. Indices reference the
    ORIGINAL `text` (not a masked copy). Ellipses ('...' / '…') count as
    a single boundary, not three: we step the cursor past consecutive
    dots after spotting the first one.
    """
    if not text:
        return []
    indices: list[int] = []
    if cjk:
        # CJK: split immediately after 。！？ if any non-space char follows.
        for i, ch in enumerate(text):
            if ch in "\u3002\uff01\uff1f":
                j = i + 1
                if j < len(text) and not text[j].isspace():
                    indices.append(j)
        return indices

    # Latin: walk the string, find punctuation that's followed by
    # whitespace + an uppercase next-sentence start. Skip ellipses.
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in ".!?":
            # Detect ellipsis: '...' or '…' (already a single char) or
            # a dot followed by more dots. Treat as ONE boundary: walk
            # past the run, then evaluate the trailing whitespace +
            # next-sentence-start condition just once.
            run_end = i + 1
            if ch == ".":
                while run_end < n and text[run_end] == ".":
                    run_end += 1
            if ch == "\u2026":
                while run_end < n and text[run_end] == "\u2026":
                    run_end += 1
            # Skip optional close-quote / paren after the punctuation.
            j = run_end
            while j < n and text[j] in "\"\u201d\u2019')\u3009]":
                j += 1
            # Skip whitespace.
            ws_start = j
            while j < n and text[j].isspace():
                j += 1
            # Need at least one whitespace AND an uppercase / open-quote
            # next character to call this a real sentence boundary.
            if j > ws_start and j < n:
                next_ch = text[j]
                # Skip an opening quote/paren before checking the letter.
                if next_ch in "\"\u201c\u2018'(\u3008[":
                    if j + 1 < n:
                        next_ch = text[j + 1]
                if next_ch.isalpha() and next_ch.isupper():
                    indices.append(j)
            i = run_end
            continue
        i += 1
    return indices


def _split_using_word_timestamps(
    seg: dict,
    sub_texts: list[str],
    words: list[dict],
    seg_start: float,
    seg_end: float,
    carry: dict,
) -> list[dict]:
    """Split a segment's words across the given sub-texts by word matching.

    For each sub-text we match words from the word list by counting
    cleaned words (not characters), which avoids drift from Whisper's
    inconsistent leading-space handling.  The last sub-text sweeps any
    remaining words.  Returns an empty list if alignment fails so the
    caller can fall back to proportional time splitting.
    """
    if not words or not sub_texts:
        return []

    # Build a clean word list for matching — strip whitespace so
    # Whisper's inconsistent leading spaces don't affect alignment.
    word_texts = [(w.get("word") or "").strip().lower() for w in words]

    sub_segments: list[dict] = []
    word_idx = 0

    for s_i, sub_text in enumerate(sub_texts):
        is_last = (s_i == len(sub_texts) - 1)
        sub_words_clean = sub_text.strip().lower().split()
        slice_start = word_idx

        if is_last:
            # Last sub-text takes all remaining words
            slice_end = len(words)
        else:
            # Find where this sub-text's words end in the word list
            # by matching word-by-word (tolerating minor differences)
            target_count = len(sub_words_clean)
            matched = 0
            scan = word_idx
            while scan < len(words) and matched < target_count:
                wt = word_texts[scan]
                if wt:  # skip empty
                    matched += 1
                scan += 1
            slice_end = scan
            word_idx = scan

        if slice_end <= slice_start:
            return []  # alignment failed

        word_slice = words[slice_start:slice_end]
        try:
            sub_start = float(word_slice[0].get("start", seg_start))
            sub_end = float(word_slice[-1].get("end", seg_end))
        except (TypeError, ValueError):
            return []

        # Don't clamp sub_end to seg_end — word timestamps are more
        # accurate than segment boundaries.  Only clamp start.
        sub_start = max(seg_start, sub_start)

        sub: dict = {
            "start": sub_start,
            "end": sub_end,
            "text": sub_text,
            "words": word_slice,
        }
        sub.update(carry)
        sub_segments.append(sub)

    return sub_segments


def _split_proportionally(
    sub_texts: list[str],
    seg_start: float,
    seg_end: float,
    carry: dict,
) -> list[dict]:
    """Length-weighted time split when no per-word timestamps are available."""
    if not sub_texts:
        return []
    duration = max(seg_end - seg_start, 1e-6)
    # Don't proportionally split short segments — the timing will be
    # wrong because speech rate varies within a segment (pauses,
    # emphasis, fast runs).  Short multi-sentence segments with no
    # word timestamps should stay unsplit rather than get bad timing.
    if duration < 4.0:
        return []  # caller keeps the original unsplit segment
    total_chars = sum(max(len(s), 1) for s in sub_texts)
    if total_chars <= 0:
        return []
    out: list[dict] = []
    cursor = seg_start
    for i, sub_text in enumerate(sub_texts):
        share = max(len(sub_text), 1) / total_chars
        is_last = (i == len(sub_texts) - 1)
        sub_dur = duration * share
        sub_end = seg_end if is_last else cursor + sub_dur
        sub: dict = {
            "start": cursor,
            "end": sub_end,
            "text": sub_text,
            "words": None,
        }
        sub.update(carry)
        out.append(sub)
        cursor = sub_end
    return out


def _merge_undersized_subsegments(
    sub_segments: list[dict],
    min_split_duration: float,
    min_split_chars: int,
) -> list[dict]:
    """Fold sub-segments shorter than the floor into their neighbour.

    Folding is forward-preferred (merge into the next sub-segment).
    Falls back to backward folding for the trailing fragment.
    """
    if len(sub_segments) <= 1:
        return sub_segments
    merged: list[dict] = []
    i = 0
    n = len(sub_segments)
    while i < n:
        sub = sub_segments[i]
        dur = float(sub.get("end", 0.0)) - float(sub.get("start", 0.0))
        text_len = len((sub.get("text") or "").strip())
        too_small = (dur < min_split_duration) or (text_len < min_split_chars)
        if too_small and i + 1 < n:
            nxt = sub_segments[i + 1]
            nxt = dict(nxt)
            nxt["start"] = sub["start"]
            nxt["text"] = (
                (sub.get("text") or "").rstrip()
                + " "
                + (nxt.get("text") or "").lstrip()
            )
            if sub.get("words") and nxt.get("words"):
                nxt["words"] = list(sub["words"]) + list(nxt["words"])
            elif sub.get("words"):
                nxt["words"] = sub["words"]
            sub_segments[i + 1] = nxt
            i += 1
            continue
        if too_small and merged:
            prev = merged[-1]
            prev["end"] = sub["end"]
            prev["text"] = (
                (prev.get("text") or "").rstrip()
                + " "
                + (sub.get("text") or "").lstrip()
            )
            if prev.get("words") and sub.get("words"):
                prev["words"] = list(prev["words"]) + list(sub["words"])
            elif sub.get("words"):
                prev["words"] = sub["words"]
            i += 1
            continue
        merged.append(dict(sub))
        i += 1
    return merged


# ── Speaker Diarization (pyannote) ─────────────────────────────────────

_diarization_pipeline = None
_diarization_lock = threading.Lock()
# One-word status string updated on every pipeline load attempt so
# callers (and the surfaced API response) can tell WHY diarization
# fell back to the heuristic path. Values:
#
#   "ok"           — pyannote pipeline is loaded and ready.
#   "no_token"     — HF_AUTH_TOKEN is empty; pyannote can't download
#                    the gated model so the heuristic path runs.
#   "load_failed"  — token was present but Pipeline.from_pretrained
#                    raised (network error, revoked token, missing
#                    torch build). Heuristic path runs.
#   "disabled"     — settings.DIARIZATION_ENABLED is False or the
#                    top-level ``CLIPAI_USE_DIARIZATION`` flag is
#                    off. No attempt is made.
#   "unknown"      — the loader has not been asked yet.
#
# This is used by ``get_diarization_status()`` which is surfaced on
# the /jobs/{id}/diarize response so the frontend can warn users
# when the heuristic fallback is in play.
_diarization_status: str = "unknown"


def _token_present() -> bool:
    """True when any of the three HF-auth env names is set, or the
    on-disk HF cache already stores a token from ``huggingface-cli
    login``. Matches the probe used by
    ``speaker_diarization._pyannote_available``."""
    return bool(
        settings.HF_AUTH_TOKEN
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.path.exists(os.path.expanduser("~/.cache/huggingface/token"))
    )


def get_diarization_status() -> dict:
    """Report the current diarization tier + reason string.

    Callers: the ``/jobs/{id}/diarize`` route surfaces this in the
    response so the frontend can distinguish pyannote-backed results
    from the heuristic fallback (accuracy degraded). Pure read
    operation — does not touch the pipeline.

    Returns a dict with:

      * ``pipeline_ready`` — ``True`` when pyannote is loaded.
      * ``reason`` — current ``_diarization_status`` value.
      * ``token_present`` — whether any HF token source is visible.
      * ``backend`` — ``"pyannote"`` / ``"mfcc"`` / ``"heuristic"``.
    """
    reason = _diarization_status
    token_present = _token_present()
    if not settings.DIARIZATION_ENABLED:
        backend = "heuristic"
    elif reason == "ok" and _diarization_pipeline is not None:
        backend = "pyannote"
    else:
        backend = "heuristic"
    return {
        "pipeline_ready": (reason == "ok" and _diarization_pipeline is not None),
        "reason": reason,
        "token_present": token_present,
        "backend": backend,
    }


def _get_diarization_pipeline():
    """Load pyannote speaker diarization pipeline (lazy init).

    Also updates the module-level ``_diarization_status`` so callers
    (and ``get_diarization_status()``) can tell whether the fallback
    path ran because of a missing token, a failed load, or an
    intentionally disabled setting.
    """
    global _diarization_pipeline, _diarization_status
    if not settings.DIARIZATION_ENABLED:
        _diarization_status = "disabled"
        return None
    with _diarization_lock:
        if _diarization_pipeline is None:
            try:
                from pyannote.audio import Pipeline
                token = (
                    settings.HF_AUTH_TOKEN
                    or os.environ.get("HUGGINGFACE_TOKEN")
                    or os.environ.get("HF_TOKEN")
                )
                if not token and not os.path.exists(
                    os.path.expanduser("~/.cache/huggingface/token")
                ):
                    logger.warning(
                        "HF_AUTH_TOKEN not set — pyannote diarization "
                        "unavailable. Falling back to pause-based speaker "
                        "detection. Set HF_AUTH_TOKEN in settings for "
                        "proper speaker labels."
                    )
                    _diarization_status = "no_token"
                    return None
                _diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=token,
                )
                # Move to GPU if available
                import torch
                if torch.cuda.is_available():
                    _diarization_pipeline.to(torch.device("cuda"))
                    logger.info("pyannote diarization loaded on CUDA")
                else:
                    logger.info("pyannote diarization loaded on CPU")
                _diarization_status = "ok"
            except Exception as e:
                logger.warning("Failed to load pyannote diarization: %s", e)
                _diarization_status = "load_failed"
                return None
    return _diarization_pipeline


def _diarize_audio(audio_path: str):
    """Run speaker diarization and return a mapping of (start, end) → speaker_label."""
    if not settings.DIARIZATION_ENABLED:
        return None

    pipeline = _get_diarization_pipeline()
    if pipeline is None:
        return None

    try:
        max_spk = settings.DIARIZATION_MAX_SPEAKERS if settings.DIARIZATION_MAX_SPEAKERS > 0 else None
        diarization = pipeline(
            audio_path,
            min_speakers=settings.DIARIZATION_MIN_SPEAKERS,
            max_speakers=max_spk,
        )

        # Build time-to-speaker mapping
        speaker_map = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_map[(turn.start, turn.end)] = speaker

        logger.info(
            "Diarization complete: %d turns, %d speakers",
            len(speaker_map),
            len(set(speaker_map.values())),
        )
        return speaker_map

    except Exception as e:
        logger.warning("Diarization failed: %s — falling back to pause-based", e)
        return None


async def diarize_transcript_post(
    audio_path: str,
    segments: list[TranscriptSegment],
    num_speakers: int = 0,
) -> list[TranscriptSegment]:
    """Run speaker diarization on an existing transcript (post-processing).

    Called when the user requests diarization after reviewing the transcript.
    The user can specify the exact number of speakers for better accuracy.

    Args:
        audio_path: Path to the audio file (for pyannote analysis).
        segments: Existing transcript segments (all "Speaker 1").
        num_speakers: Expected number of speakers (0 = auto-detect).

    Returns:
        Updated TranscriptSegment[] with speaker labels assigned.
    """
    import asyncio
    import functools

    # Convert segments to raw_segments format for speaker assignment
    raw_segments = []
    for seg in segments:
        word_list = None
        if seg.words:
            word_list = [{"start": w.start, "end": w.end, "word": w.word} for w in seg.words]
        raw_segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "words": word_list,
            "confidence": seg.confidence,
            "avg_logprob": seg.avg_logprob,
            "no_speech_prob": seg.no_speech_prob,
        })

    # Try pyannote first (best quality)
    if settings.DIARIZATION_ENABLED and settings.HF_AUTH_TOKEN:
        pipeline = _get_diarization_pipeline()
        if pipeline:
            try:
                loop = asyncio.get_running_loop()

                def _run_pyannote():
                    diarize_kwargs = {"audio": audio_path}
                    if num_speakers > 0:
                        diarize_kwargs["min_speakers"] = num_speakers
                        diarize_kwargs["max_speakers"] = num_speakers
                    elif settings.DIARIZATION_MIN_SPEAKERS > 1:
                        diarize_kwargs["min_speakers"] = settings.DIARIZATION_MIN_SPEAKERS
                    if settings.DIARIZATION_MAX_SPEAKERS > 0 and num_speakers == 0:
                        diarize_kwargs["max_speakers"] = settings.DIARIZATION_MAX_SPEAKERS

                    diarization = pipeline(**diarize_kwargs)
                    speaker_map = {}
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        speaker_map[(turn.start, turn.end)] = speaker
                    return speaker_map

                speaker_map = await loop.run_in_executor(_transcription_executor, _run_pyannote)

                if speaker_map:
                    result = _assign_speakers_from_diarization(raw_segments, speaker_map)
                    num_detected = len(set(s.speaker for s in result))
                    status = get_diarization_status()
                    logger.info(
                        "Post-processing diarization (pyannote): %d speakers detected "
                        "(requested: %s, status=%s)",
                        num_detected, num_speakers if num_speakers > 0 else "auto",
                        status.get("reason"),
                    )
                    return result
            except Exception as e:
                logger.warning("Post-processing pyannote diarization failed: %s", e)

    # Fallback: heuristic speaker assignment. This is where every
    # audit bug converges; leave a loud warning so the ops log shows
    # exactly why the pyannote tier was skipped.
    status = get_diarization_status()
    logger.warning(
        "Post-processing diarization falling back to heuristic "
        "(reason=%s, token_present=%s). Speaker accuracy will be "
        "degraded — set HF_AUTH_TOKEN in settings to enable pyannote.",
        status.get("reason"), status.get("token_present"),
    )
    result = _assign_speakers(raw_segments)
    num_detected = len(set(s.speaker for s in result))
    logger.info(
        "Post-processing diarization (heuristic): %d speakers detected "
        "(requested: %s)",
        num_detected, num_speakers if num_speakers > 0 else "auto",
    )
    return result


def _assign_speakers_from_diarization(
    raw_segments: list[dict], speaker_map: dict
) -> list[TranscriptSegment]:
    """Align Whisper segments with pyannote diarization output.

    Speakers are numbered by order of first appearance in the audio
    (not alphabetically by pyannote's internal SPEAKER_XX labels).
    """
    # Sort diarization turns by start time
    turns = sorted(speaker_map.items(), key=lambda x: x[0][0])

    # Build speaker name mapping by first-appearance order
    seen_order: list[str] = []
    for (_, _), speaker in turns:
        if speaker not in seen_order:
            seen_order.append(speaker)
    speaker_names = {s: f"Speaker {i+1}" for i, s in enumerate(seen_order)}

    transcript_segments = []
    for seg in raw_segments:
        seg_start, seg_end = seg["start"], seg["end"]

        # Find the turn with maximum overlap
        best_speaker = None
        best_overlap = 0.0
        for (turn_start, turn_end), speaker in turns:
            overlap_start = max(seg_start, turn_start)
            overlap_end = min(seg_end, turn_end)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

        speaker_label = speaker_names.get(best_speaker, "Speaker 1") if best_speaker else "Speaker 1"

        words = None
        if seg.get("words"):
            words = [WordTimestamp(**w) for w in seg["words"]]

        transcript_segments.append(TranscriptSegment(
            start=round(seg["start"], 2),
            end=round(seg["end"], 2),
            text=seg["text"],
            speaker=speaker_label,
            words=words,
            confidence=seg.get("confidence"),
            avg_logprob=seg.get("avg_logprob"),
            no_speech_prob=seg.get("no_speech_prob"),
        ))

    return transcript_segments


def assign_speakers_heuristic(raw_segments: list[dict]) -> list:
    """Public wrapper for heuristic speaker assignment.

    Assigns speaker labels using pause-based turn detection and speech
    rate analysis. No model or GPU required — instant computation.
    """
    return _assign_speakers(raw_segments)


def assign_speakers_with_face_data(
    raw_segments: list[dict],
    face_results: list = None,
    face_registry=None,
    scene_descriptions: list = None,
    active_speaker_events: list = None,
) -> list:
    """Assign speaker labels using audio timing + face mesh + lip aperture + vision.

    Fuses audio cues (Whisper word timestamps, pause patterns, speech rate)
    with visual cues (face mesh lip aperture, identity embeddings, scene
    descriptions) to accurately identify who is speaking in each segment.

    No artificial speaker cap — speaker count = unique face slots detected.

    Data fusion (most reliable first):
    1. Active speaker events — pre-computed lip-audio cross-correlation per timestamp
    2. Per-word lip sync — correlate each word's timing with face lip movement
    3. Segment-level lip aperture — highest LAR face during speech = speaker
    4. Face identity embeddings — track who is who across frames
    5. Vision scene descriptions — AI subject_x as position fallback
    6. Audio heuristics — pause duration, speech rate changes (from _assign_speakers)
    7. Temporal continuity — inherit from previous segment
    """
    if not face_results or not face_registry or not getattr(face_registry, 'multi_speaker', False):
        return _assign_speakers(raw_segments)

    # Build time-indexed face data: ALL faces per frame with lip + identity data
    frame_data = {}  # timestamp -> [(slot_id, lip_aperture, identity_id, nose_x)]
    for fr in face_results:
        if not fr.faces:
            continue
        faces_in_frame = []
        for f in fr.faces:
            slot_id = f.identity_id if f.identity_id >= 0 else -1
            if slot_id < 0:
                slot = face_registry.nearest_slot(f.nose_x)
                slot_id = slot.slot_id if slot else -1
            if slot_id >= 0:
                faces_in_frame.append((slot_id, f.lip_aperture, f.identity_id, f.nose_x))
        if faces_in_frame:
            frame_data[fr.timestamp] = faces_in_frame

    if len(frame_data) < 3:
        return _assign_speakers(raw_segments)

    frame_times = sorted(frame_data.keys())

    # Build active speaker event lookup if available
    def _active_speaker_at(t):
        if not active_speaker_events:
            return -1
        for ev in active_speaker_events:
            if ev.start <= t <= ev.end and ev.slot_id >= 0:
                return ev.slot_id
        return -1

    # Build scene description lookup
    scene_sx_map = {}
    if scene_descriptions:
        for sc in scene_descriptions:
            ts = sc.timestamp if hasattr(sc, 'timestamp') else sc.get('timestamp', 0)
            sx = sc.subject_x if hasattr(sc, 'subject_x') else sc.get('subject_x', 50)
            scene_sx_map[ts] = sx

    def _lip_speaker_at(t, window=1.0):
        """Find which face slot has the highest lip movement near time t."""
        nearby = []
        for ft in frame_times:
            if abs(ft - t) <= window:
                nearby.extend(frame_data[ft])
            elif ft > t + window:
                break
        if not nearby:
            return -1

        speaking = [(sid, lar) for sid, lar, _, _ in nearby if lar > 0.02]
        if speaking:
            slot_lars = {}
            for sid, lar in speaking:
                slot_lars.setdefault(sid, []).append(lar)
            return max(slot_lars, key=lambda s: sum(slot_lars[s]) / len(slot_lars[s]))

        # No one clearly speaking (all lar <= 0.02) — return -1 so the
        # caller can use audio heuristics or create a virtual speaker.
        # Previously this fell back to "most common face slot" which
        # prevented detecting off-camera speakers (Bug D).
        return -1

    def _find_speaker_for_segment(seg):
        """Determine speaker for a segment using all available signals."""
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_mid = (seg_start + seg_end) / 2

        # Signal 1: Active speaker events (pre-computed lip-audio correlation)
        as_slot = _active_speaker_at(seg_mid)
        if as_slot >= 0:
            return as_slot

        # Signal 2: Per-word lip sync (correlate each word's timing with lip movement)
        if seg.get("words"):
            word_votes = []
            for w in seg["words"]:
                w_start = w.get("start", w.get("s", 0))
                w_end = w.get("end", w.get("e", 0))
                if w_end - w_start < 0.05:
                    continue
                w_mid = (w_start + w_end) / 2
                slot = _lip_speaker_at(w_mid, window=0.5)
                if slot >= 0:
                    word_votes.append(slot)
            if word_votes:
                from collections import Counter
                return Counter(word_votes).most_common(1)[0][0]

        # Signal 3: Segment-level lip aperture (wider ±2s window)
        lip_slot = _lip_speaker_at(seg_mid, window=2.0)
        if lip_slot >= 0:
            return lip_slot

        # Signal 4: Vision scene description
        if scene_sx_map:
            best_sc_t = min(scene_sx_map.keys(), key=lambda st: abs(st - seg_mid), default=None)
            if best_sc_t is not None and abs(best_sc_t - seg_mid) <= 10.0:
                sx = scene_sx_map[best_sc_t]
                slot = face_registry.nearest_slot(sx)
                if slot:
                    return slot.slot_id

        return -1

    # Assign speakers — face slots provide anchors but audio can exceed them.
    # DIARIZATION_ALLOW_EXCEED_FACE_SLOTS (default true): when audio heuristics
    # detect a speaker change but no face slot matches, create a virtual speaker
    # label not tied to any face slot (off-camera speaker). This decouples the
    # audio speaker count from the face registry ceiling (Bug D fix).
    ALLOW_EXCEED = os.environ.get("DIARIZATION_ALLOW_EXCEED_FACE_SLOTS", "true").lower() in ("true", "1", "yes")
    slot_to_speaker = {}
    next_spk = 1
    segments = []
    prev_spk = 1
    virtual_speakers = 0  # count of off-camera speaker labels created

    for i, seg in enumerate(raw_segments):
        slot_id = _find_speaker_for_segment(seg)

        # Audio heuristic: large gap = possible new speaker
        if slot_id < 0 and i > 0:
            gap = seg["start"] - raw_segments[i - 1]["end"]
            if gap >= 1.2:
                # Gap suggests speaker change — try a different slot than previous
                prev_slot = None
                for sid, spk in slot_to_speaker.items():
                    if spk == prev_spk:
                        prev_slot = sid
                        break
                # Look for any OTHER slot with active lip movement nearby
                # (not just any visible face — must be speaking)
                nearby_speaking_slots = set()
                for ft in frame_times:
                    if abs(ft - seg["start"]) <= 3.0:
                        for sid, lar, _, _ in frame_data.get(ft, []):
                            if sid != prev_slot and lar > 0.02:
                                nearby_speaking_slots.add(sid)
                if nearby_speaking_slots:
                    slot_id = min(nearby_speaking_slots)
                elif ALLOW_EXCEED:
                    # No face slot matches and audio says new speaker:
                    # create a virtual speaker (off-camera voice)
                    virtual_key = f"_virtual_{next_spk}"
                    slot_to_speaker[virtual_key] = next_spk
                    spk = next_spk
                    next_spk += 1
                    virtual_speakers += 1
                    prev_spk = spk
                    words = [WordTimestamp(**w) for w in seg["words"]] if seg.get("words") else None
                    segments.append(TranscriptSegment(
                        start=round(seg["start"], 2), end=round(seg["end"], 2),
                        text=seg["text"], speaker=f"Speaker {spk}", words=words,
                        confidence=seg.get("confidence"), avg_logprob=seg.get("avg_logprob"),
                        no_speech_prob=seg.get("no_speech_prob"),
                    ))
                    continue
            else:
                # Small gap + no face match: look for any other visible slot
                nearby_slots = set()
                for ft in frame_times:
                    if abs(ft - seg["start"]) <= 3.0:
                        prev_slot_id = None
                        for sid, spk_label in slot_to_speaker.items():
                            if spk_label == prev_spk:
                                prev_slot_id = sid
                                break
                        for sid, _, _, _ in frame_data.get(ft, []):
                            if sid != prev_slot_id:
                                nearby_slots.add(sid)
                if nearby_slots:
                    slot_id = min(nearby_slots)

        if slot_id >= 0:
            if slot_id not in slot_to_speaker:
                slot_to_speaker[slot_id] = next_spk
                next_spk += 1
            spk = slot_to_speaker[slot_id]
        else:
            spk = prev_spk

        prev_spk = spk
        words = [WordTimestamp(**w) for w in seg["words"]] if seg.get("words") else None
        segments.append(TranscriptSegment(
            start=round(seg["start"], 2), end=round(seg["end"], 2),
            text=seg["text"], speaker=f"Speaker {spk}", words=words,
            confidence=seg.get("confidence"), avg_logprob=seg.get("avg_logprob"),
            no_speech_prob=seg.get("no_speech_prob"),
        ))

    num_speakers = len(set(s.speaker for s in segments))
    logger.info(
        "Face-aware diarization: %d speakers from %d face slots "
        "(%d virtual/off-camera), %d/%d frames with face data, "
        "%d scene descriptions, %d active speaker events",
        num_speakers, len(face_registry.slots), virtual_speakers,
        len(frame_data), len(face_results), len(scene_sx_map),
        len(active_speaker_events) if active_speaker_events else 0,
    )
    return segments
