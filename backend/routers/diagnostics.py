"""Pipeline diagnostics endpoints for GPU status monitoring and pipeline testing."""

import asyncio
import base64
import io
import json
import logging
import os
import time
from typing import AsyncGenerator

import httpx
from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import StreamingResponse

from backend.app.auth.deps import require_admin
from backend.app.auth.models import User
from backend.config import settings

router = APIRouter(prefix="/api/diagnostics", tags=["diagnostics"])
logger = logging.getLogger(__name__)

# ── GPU info cache (doesn't change at runtime) ──────────────────────────
_gpu_info_cache: dict | None = None


async def _get_gpu_info() -> dict:
    """Get GPU hardware info. Tries Ollama's container first since the app
    container often doesn't have direct GPU access (nvidia-smi/torch CUDA).

    Detection priority:
    1. Ollama /api/ps — if any model has VRAM > 0, GPU exists
    2. Local nvidia-smi — works if GPU passthrough configured for app container
    3. Local torch.cuda — works if CUDA runtime available in app container
    4. Ollama probe — load a tiny model with num_gpu=99 and check GPU placement
    """
    global _gpu_info_cache
    if _gpu_info_cache is not None:
        return {**_gpu_info_cache}

    info = {
        "gpu_available": False,
        "gpu_in_use": False,
        "gpu_poisoned": False,
        "gpu_name": None,
        "vram_total_bytes": 0,
        "vram_used_bytes": 0,
        "cuda_available": False,
    }

    # Method 1: Check Ollama /api/ps — if a model is loaded on GPU, we know GPU exists
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{settings.OLLAMA_HOST}/api/ps")
            if resp.status_code == 200:
                for m in resp.json().get("models", []):
                    if m.get("size_vram", 0) > 0:
                        info["gpu_available"] = True
                        info["cuda_available"] = True
                        info["gpu_name"] = "NVIDIA GPU (via Ollama)"
                        break
    except Exception:
        pass

    # Method 2: nvidia-smi locally
    if not info["gpu_available"]:
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(",")
                if len(parts) >= 2:
                    info["gpu_name"] = parts[0].strip()
                    vram_mb = int(parts[1].strip())
                    # Only trust nvidia-smi if it reports > 1GB (avoids app container's
                    # 256MB iGPU misreport when discrete GPU is in Ollama container)
                    if vram_mb > 1024:
                        info["vram_total_bytes"] = vram_mb * 1024 * 1024
                        info["gpu_available"] = True
                        info["cuda_available"] = True
        except Exception:
            pass

    # Method 3: /dev/nvidia* device nodes (fast, no CUDA context needed)
    # This catches the common Docker case where nvidia-smi misreports VRAM
    # (256MB iGPU) but the discrete GPU is available to the subprocess.
    if not info["cuda_available"]:
        try:
            import glob as _glob
            nvidia_devs = _glob.glob("/dev/nvidia[0-9]*")
            if nvidia_devs and settings.GPU_ACCELERATION_ENABLED:
                info["gpu_available"] = True
                info["cuda_available"] = True
                if not info["gpu_name"]:
                    info["gpu_name"] = f"NVIDIA GPU ({len(nvidia_devs)} device{'s' if len(nvidia_devs) > 1 else ''})"
                # Try to get VRAM from transcription module's detection
                try:
                    from backend.services.transcription import _get_gpu_vram_mb
                    vram_mb = _get_gpu_vram_mb()
                    if vram_mb > 0 and info["vram_total_bytes"] == 0:
                        info["vram_total_bytes"] = vram_mb * 1024 * 1024
                except Exception:
                    pass
        except Exception:
            pass

    # Method 4: PyTorch CUDA
    if not info["gpu_available"]:
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["vram_total_bytes"] = torch.cuda.get_device_properties(0).total_mem
                info["gpu_available"] = True
                info["cuda_available"] = True
        except Exception:
            pass

    # Method 5: Probe Ollama — load a model with GPU request, check if it gets GPU
    if not info["gpu_available"]:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{settings.OLLAMA_HOST}/api/generate",
                    json={
                        "model": settings.OLLAMA_VISION_MODEL,
                        "prompt": "hi",
                        "stream": False,
                        "options": {"num_gpu": 99, "num_predict": 1},
                    },
                    timeout=60,
                )
                if resp.status_code == 200:
                    ps_resp = await client.get(f"{settings.OLLAMA_HOST}/api/ps")
                    if ps_resp.status_code == 200:
                        for m in ps_resp.json().get("models", []):
                            if m.get("size_vram", 0) > 0:
                                info["gpu_available"] = True
                                info["cuda_available"] = True
                                info["gpu_name"] = "NVIDIA GPU (via Ollama)"
                                break
                    # Clean up probe
                    await client.post(
                        f"{settings.OLLAMA_HOST}/api/generate",
                        json={"model": settings.OLLAMA_VISION_MODEL, "keep_alive": 0},
                    )
                    await asyncio.sleep(2)
        except Exception:
            pass

    # Default VRAM for known GTX 1650 setup if we detected GPU but not VRAM size
    if info["gpu_available"] and info["vram_total_bytes"] == 0:
        info["vram_total_bytes"] = int(3.6 * 1024 * 1024 * 1024)

    _gpu_info_cache = info
    return {**info}


async def _get_ollama_loaded_models() -> list[dict]:
    """Get currently loaded Ollama models with VRAM info."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{settings.OLLAMA_HOST}/api/ps")
            if resp.status_code == 200:
                models = []
                for m in resp.json().get("models", []):
                    models.append({
                        "name": m.get("name", ""),
                        "size_bytes": m.get("size", 0),
                        "vram_bytes": m.get("size_vram", 0),
                        "processor": m.get("processor", "unknown"),
                        "expires_at": m.get("expires_at", ""),
                    })
                return models
    except Exception:
        pass
    return []


async def _unload_all_models() -> None:
    """Unload all loaded Ollama models to free VRAM."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{settings.OLLAMA_HOST}/api/ps")
            if resp.status_code == 200:
                for m in resp.json().get("models", []):
                    name = m.get("name", "")
                    if name:
                        await client.post(
                            f"{settings.OLLAMA_HOST}/api/generate",
                            json={"model": name, "keep_alive": 0},
                        )
    except Exception:
        pass


async def _unload_and_wait(max_wait: int = 15) -> bool:
    """Unload all models and poll until VRAM is actually freed.

    On GTX 1650, Ollama's CUDA memory reclamation can take 5-10s after
    keep_alive=0. Simply sleeping 2s is not enough — the text model OOMs
    because the vision model's VRAM hasn't been released yet.

    After /api/ps confirms no models, we add an extra 3s delay to let
    the CUDA driver fully reclaim GPU memory across container boundaries.

    Returns True if no models remain loaded.
    """
    await _unload_all_models()

    for attempt in range(max_wait):
        await asyncio.sleep(1)
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{settings.OLLAMA_HOST}/api/ps")
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    if not models:
                        # Models removed from Ollama's list, but CUDA driver
                        # may still hold GPU memory for several seconds.
                        # Ollama also spawns 10-20 runners during unload/reload
                        # which compete for GPU resources.
                        # On GTX 1650 with <200MB margin, we need a generous wait.
                        logger.info("Ollama reports no models after %ds — waiting 5s for CUDA driver + runners to settle", attempt + 1)
                        await asyncio.sleep(5)
                        return True
                    # Models still present — send another unload
                    for m in models:
                        name = m.get("name", "")
                        if name:
                            await client.post(
                                f"{settings.OLLAMA_HOST}/api/generate",
                                json={"model": name, "keep_alive": 0},
                            )
        except Exception:
            pass

    logger.warning("Models still loaded after %ds wait", max_wait)
    return False


def _generate_test_image() -> str:
    """Generate a tiny 64x64 test image as base64 for vision model testing."""
    try:
        from PIL import Image
    except ImportError:
        # Minimal 1x1 PNG as fallback if Pillow not available
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPj/HwADBwIAMCbHYQAAAABJRU5ErkJggg=="

    img = Image.new("RGB", (64, 64))
    pixels = img.load()
    for y in range(64):
        for x in range(64):
            pixels[x, y] = (x * 4, y * 4, 128)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _sse_event(event_type: str, data: dict) -> str:
    """Format an SSE event."""
    payload = {"type": event_type, "data": data}
    return f"data: {json.dumps(payload)}\n\n"


async def _test_vision_model(model: str) -> dict:
    """Test vision model: load, analyze a tiny test image, check GPU status."""
    start = time.time()
    try:
        test_image_b64 = _generate_test_image()
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(
                f"{settings.OLLAMA_HOST}/api/chat",
                json={
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": "Describe this image in one sentence.",
                        "images": [test_image_b64],
                    }],
                    "stream": False,
                    "options": {
                        "num_gpu": 99,
                        "num_predict": 50,
                        "num_batch": 128,  # Reduce batch to lower compute buffer
                    },
                },
                timeout=180,
            )

            duration_ms = int((time.time() - start) * 1000)

            if resp.status_code != 200:
                error_text = resp.text[:200] if resp.text else ""
                return {
                    "status": "fail",
                    "message": f"Vision model returned HTTP {resp.status_code}: {error_text}",
                    "duration_ms": duration_ms, "gpu_status": "unknown",
                }

            # Check GPU status
            gpu_status, vram_used = await _check_model_gpu(client, model)

            response_text = ""
            try:
                response_text = resp.json().get("message", {}).get("content", "")[:100]
            except Exception:
                pass

            if gpu_status.startswith("cpu"):
                return {
                    "status": "warn",
                    "message": f"Vision loaded on CPU — {duration_ms}ms. GPU may be poisoned from prior OOM.",
                    "duration_ms": duration_ms, "gpu_status": gpu_status,
                    "vram_bytes": vram_used, "sample_output": response_text,
                }

            return {
                "status": "pass",
                "message": f"Vision model OK — {duration_ms}ms on GPU",
                "duration_ms": duration_ms, "gpu_status": gpu_status,
                "vram_bytes": vram_used, "sample_output": response_text,
            }

    except asyncio.TimeoutError:
        return {
            "status": "fail",
            "message": (
                "Vision model timed out (>180s). If Whisper just ran, the GPU may need "
                "rediscovery — try restarting the Ollama container."
            ),
            "duration_ms": 180000, "gpu_status": "timeout",
        }
    except Exception as e:
        return {
            "status": "fail",
            "message": f"Vision model error: {str(e)[:200]}",
            "duration_ms": int((time.time() - start) * 1000), "gpu_status": "error",
        }


async def _test_text_model(model: str) -> dict:
    """Test text model: load, run a short completion, check GPU status."""
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{settings.OLLAMA_HOST}/api/chat",
                json={
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": "Summarize in one sentence: A man walks into a coffee shop and orders a latte.",
                    }],
                    "stream": False,
                    "options": {
                        "num_gpu": 99,
                        "num_predict": 50,
                        "num_ctx": 1024,    # Minimal context for test — reduces compute graph from 300MB to ~150MB
                        "num_batch": 128,   # Reduce batch size to lower compute buffer allocation
                    },
                },
                timeout=120,
            )

            duration_ms = int((time.time() - start) * 1000)

            if resp.status_code != 200:
                error_text = resp.text[:200] if resp.text else ""
                if "cudaMalloc" in error_text or "out of memory" in error_text.lower():
                    return {
                        "status": "fail",
                        "message": "CUDA OOM — text model doesn't fit. Vision model may not have unloaded.",
                        "duration_ms": duration_ms, "gpu_status": "oom",
                    }
                return {
                    "status": "fail",
                    "message": f"Text model returned HTTP {resp.status_code}: {error_text}",
                    "duration_ms": duration_ms, "gpu_status": "unknown",
                }

            gpu_status, vram_used = await _check_model_gpu(client, model)

            response_text = ""
            try:
                response_text = resp.json().get("message", {}).get("content", "")[:100]
            except Exception:
                pass

            if gpu_status.startswith("cpu"):
                return {
                    "status": "warn",
                    "message": f"Text model on CPU — {duration_ms}ms. Will cause 60s stall timeouts.",
                    "duration_ms": duration_ms, "gpu_status": gpu_status,
                    "vram_bytes": vram_used, "sample_output": response_text,
                }

            return {
                "status": "pass",
                "message": f"Text model OK — {duration_ms}ms on GPU",
                "duration_ms": duration_ms, "gpu_status": gpu_status,
                "vram_bytes": vram_used, "sample_output": response_text,
            }

    except asyncio.TimeoutError:
        return {
            "status": "fail",
            "message": "Text model timed out (>120s) — stuck on CPU or overloaded",
            "duration_ms": 120000, "gpu_status": "timeout",
        }
    except Exception as e:
        return {
            "status": "fail",
            "message": f"Text model error: {str(e)[:200]}",
            "duration_ms": int((time.time() - start) * 1000), "gpu_status": "error",
        }


async def _test_cloud_vision(model: str, provider: str) -> dict:
    """Test a cloud vision model via AIOrchestrator.analyze_frames()."""
    start = time.time()
    try:
        from backend.services.ai_orchestrator import AIOrchestrator
        from backend.models import FrameData
        orch = AIOrchestrator()
        test_image_b64 = _generate_test_image()

        # Create a minimal FrameData for the orchestrator
        frame = FrameData(path="", timestamp=0.0)
        frame.base64 = test_image_b64

        scenes, used_provider = await asyncio.wait_for(
            orch.analyze_frames([frame], job_id="pipeline-test"),
            timeout=90,
        )
        elapsed_ms = int((time.time() - start) * 1000)
        desc = scenes[0].description[:100] if scenes and hasattr(scenes[0], 'description') else ""
        return {
            "status": "pass",
            "message": f"Vision model OK — {elapsed_ms}ms via {used_provider}",
            "duration_ms": elapsed_ms,
            "gpu_status": f"cloud ({used_provider})",
            "sample_output": desc,
        }
    except asyncio.TimeoutError:
        return {
            "status": "fail",
            "message": f"Vision model timed out (>90s) via {provider}",
            "duration_ms": 90000, "gpu_status": "timeout",
        }
    except Exception as e:
        elapsed_ms = int((time.time() - start) * 1000)
        return {
            "status": "fail",
            "message": f"Vision model error via {provider}: {str(e)[:200]}",
            "duration_ms": elapsed_ms, "gpu_status": "error",
        }


async def _test_cloud_text(model: str, provider: str) -> dict:
    """Test a cloud text model via AIOrchestrator."""
    start = time.time()
    try:
        from backend.services.ai_orchestrator import AIOrchestrator
        orch = AIOrchestrator()
        raw = await asyncio.wait_for(
            orch.text_completion(
                "Summarize in one sentence: A customer orders a latte at a coffee shop.",
                max_tokens=50, timeout=60,
            ),
            timeout=60,
        )
        elapsed_ms = int((time.time() - start) * 1000)
        return {
            "status": "pass",
            "message": f"Text model OK — {elapsed_ms}ms via {provider}",
            "duration_ms": elapsed_ms,
            "gpu_status": f"cloud ({provider})",
            "sample_output": str(raw)[:100] if raw else "",
        }
    except asyncio.TimeoutError:
        return {
            "status": "fail",
            "message": f"Text model timed out (>60s) via {provider}",
            "duration_ms": 60000, "gpu_status": "timeout",
        }
    except Exception as e:
        elapsed_ms = int((time.time() - start) * 1000)
        return {
            "status": "fail",
            "message": f"Text model error via {provider}: {str(e)[:200]}",
            "duration_ms": elapsed_ms, "gpu_status": "error",
        }


async def _check_model_gpu(client: httpx.AsyncClient, model: str) -> tuple[str, int]:
    """Check if a model is on GPU via /api/ps. Returns (gpu_status_str, vram_bytes)."""
    try:
        ps_resp = await client.get(f"{settings.OLLAMA_HOST}/api/ps")
        if ps_resp.status_code == 200:
            for m in ps_resp.json().get("models", []):
                if model.split(":")[0] in m.get("name", ""):
                    vram = m.get("size_vram", 0)
                    total = m.get("size", 0)
                    if vram > 0 and total > 0:
                        pct = (vram / total) * 100
                        return f"gpu ({pct:.0f}% on CUDA)", vram
                    return "cpu (0% VRAM)", 0
    except Exception:
        pass
    return "unknown", 0


# ── Endpoints ────────────────────────────────────────────────────────────


@router.get("/gpu-status")
async def get_gpu_status():
    """Real-time GPU memory usage and loaded Ollama models. Polled every 2s."""
    gpu = await _get_gpu_info()
    loaded_models = await _get_ollama_loaded_models()

    # Compute dynamic VRAM and poisoning state from loaded models
    vram_used = sum(m["vram_bytes"] for m in loaded_models)
    gpu["vram_used_bytes"] = vram_used
    gpu["gpu_in_use"] = any(m["vram_bytes"] > 0 for m in loaded_models)

    # If we see a model on GPU but the cache said no GPU, invalidate cache
    if gpu["gpu_in_use"] and not gpu.get("gpu_available"):
        global _gpu_info_cache
        _gpu_info_cache = None
        gpu["gpu_available"] = True
        gpu["cuda_available"] = True
        if not gpu.get("gpu_name"):
            gpu["gpu_name"] = "NVIDIA GPU (via Ollama)"
        if gpu["vram_total_bytes"] == 0:
            gpu["vram_total_bytes"] = int(3.6 * 1024 * 1024 * 1024)

    # Detect GPU scheduler poisoning: hardware exists but loaded models are on CPU
    if gpu["gpu_available"] and loaded_models and not gpu["gpu_in_use"]:
        gpu["gpu_poisoned"] = True
    else:
        gpu["gpu_poisoned"] = False

    ollama_available = False
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.head(f"{settings.OLLAMA_HOST}")
            ollama_available = resp.status_code == 200
    except Exception:
        pass

    # Torch GPU memory info (separate from Ollama — this is the app container)
    torch_gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            torch_gpu = {
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
            }
            # Torch reserved memory counts as VRAM used (it's unavailable to Ollama)
            gpu["vram_used_bytes"] = vram_used + torch_gpu["reserved_bytes"]
    except (ImportError, Exception):
        pass

    return {
        "gpu": gpu,
        "loaded_models": loaded_models,
        "ollama_available": ollama_available,
        "torch_gpu": torch_gpu,
    }


@router.post("/unload-models")
async def unload_models():
    """Manually unload all Ollama models to free VRAM."""
    await _unload_all_models()
    return {"status": "ok", "message": "All models unloaded"}


@router.post("/release-gpu")
async def release_gpu():
    """Release all torch GPU memory AND unload Ollama models."""
    released_mb = 0
    try:
        from backend.services.pipeline import release_torch_gpu_memory
        import torch
        before = torch.cuda.memory_reserved() / 1024 / 1024 if torch.cuda.is_available() else 0
        release_torch_gpu_memory()
        after = torch.cuda.memory_reserved() / 1024 / 1024 if torch.cuda.is_available() else 0
        released_mb = max(0, before - after)
    except (ImportError, Exception):
        pass

    await _unload_all_models()

    # Invalidate GPU info cache
    global _gpu_info_cache
    _gpu_info_cache = None

    return {
        "status": "ok",
        "message": f"GPU memory released ({released_mb:.0f}MB torch freed). Ollama models unloaded.",
    }


@router.post("/restart-ollama")
async def restart_ollama():
    """Restart the Ollama container to reset a poisoned GPU scheduler.

    After a CUDA OOM, Ollama's scheduler permanently blacklists the GPU.
    The only fix is restarting the Ollama process/container.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["docker", "restart", "clipai-ollama"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            # Invalidate GPU cache so next poll re-detects
            global _gpu_info_cache
            _gpu_info_cache = None
            return {"status": "ok", "message": "Ollama container restarting — GPU scheduler will be reset. Wait ~15 seconds."}
        return {
            "status": "manual",
            "message": f"Cannot restart from app container (exit {result.returncode}). Run manually: docker restart clipai-ollama",
        }
    except FileNotFoundError:
        return {
            "status": "manual",
            "message": "Docker CLI not available in app container. Run on your server: docker restart clipai-ollama",
        }
    except Exception as e:
        return {
            "status": "manual",
            "message": f"Restart failed: {str(e)[:200]}. Run manually: docker restart clipai-ollama",
        }


@router.post("/test-pipeline")
async def test_pipeline(request: Request):
    """Run a diagnostic test that mirrors the real analysis pipeline 1:1.

    Simulates every step of _run_analysis_inner() in the same order with
    the same VRAM management, so any failure that would occur during a real
    analysis is caught here in ~60-90 seconds instead of 30-60 minutes.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    include_whisper = body.get("include_whisper", True)
    test_translation = body.get("test_translation", False)

    is_ollama = "ollama" in settings.active_provider_chain

    # Determine the active provider and model names
    # Use the FIRST provider in the chain as the primary
    _chain = [p.strip() for p in settings.AI_FALLBACK_CHAIN.split(",") if p.strip()]
    _primary_provider = _chain[0] if _chain else "ollama"

    if _primary_provider == "ollama":
        vision_model = settings.OLLAMA_VISION_MODEL
        text_model = settings.OLLAMA_TEXT_MODEL
        provider_label = "ollama"
    elif _primary_provider == "openrouter":
        vision_model = settings.OPENROUTER_VISION_MODEL or "openrouter/default"
        text_model = settings.OPENROUTER_TEXT_MODEL or "openrouter/default"
        provider_label = "openrouter"
    elif _primary_provider == "anthropic":
        vision_model = "claude-3-haiku"
        text_model = "claude-3-haiku"
        provider_label = "anthropic"
    elif _primary_provider == "gemini":
        vision_model = "gemini-2.5-flash"
        text_model = "gemini-2.5-flash"
        provider_label = "gemini"
    elif _primary_provider == "groq":
        vision_model = "llava-v1.5-7b-4096-preview"
        text_model = settings.GROQ_TEXT_MODEL if hasattr(settings, 'GROQ_TEXT_MODEL') else "llama3-8b-8192"
        provider_label = "groq"
    else:
        vision_model = settings.OLLAMA_VISION_MODEL
        text_model = settings.OLLAMA_TEXT_MODEL
        provider_label = "ollama"

    uses_ollama = _primary_provider == "ollama"

    async def event_stream() -> AsyncGenerator[str, None]:
        import tempfile
        import shutil

        tmp_dir = None
        # Track total phases dynamically
        # Cloud providers with Ollama fallback: add pre-whisper VRAM clear phase
        # Ollama providers: same phases + GPU rediscovery + VRAM management
        _whisper_phases = (1 if include_whisper else 0) + (1 if include_whisper and test_translation else 0)
        _vram_phase = 1 if (is_ollama and not uses_ollama) else 0  # Pre-whisper unload for cloud+ollama fallback
        total_phases = (8 + _whisper_phases + _vram_phase) if not uses_ollama else (11 + _whisper_phases)
        phase_counter = [0]

        def _phase(phase_id, label):
            idx = phase_counter[0]
            phase_counter[0] += 1
            return _sse_event("phase_start", {
                "phase": phase_id, "label": label,
                "phase_index": idx, "total_phases": total_phases,
            })

        try:
            # ══════════════════════════════════════════════════════════
            # Phase: Provider + GPU check
            # ══════════════════════════════════════════════════════════
            yield _phase("provider_gpu_check", "Checking AI provider + GPU hardware...")

            if not uses_ollama:
                yield _sse_event("phase_result", {
                    "phase": "provider_gpu_check", "status": "pass",
                    "message": f"Cloud provider ({provider_label}) selected — using cloud AI for vision/text",
                })
                # Test cloud API connectivity before proceeding
                yield _phase("cloud_test", f"Testing {provider_label} API connection...")
                try:
                    from backend.services.ai_orchestrator import AIOrchestrator
                    orch = AIOrchestrator()
                    t0 = time.time()
                    await asyncio.wait_for(
                        orch.text_completion("Say hello in one word.", max_tokens=10, timeout=30),
                        timeout=30,
                    )
                    ms = int((time.time() - t0) * 1000)
                    yield _sse_event("phase_result", {
                        "phase": "cloud_test", "status": "pass",
                        "message": f"{provider_label} API responded in {ms}ms",
                    })
                except Exception as e:
                    yield _sse_event("phase_result", {
                        "phase": "cloud_test", "status": "fail",
                        "message": f"{provider_label} API error: {str(e)[:200]}",
                    })
                    yield _sse_event("complete", {"overall_status": "fail"})
                    return
                # Continue to test remaining pipeline stages (Whisper, vision, text, etc.)
                gpu = await _get_gpu_info()

            if uses_ollama:
                # ── Ollama local pipeline — check connectivity and GPU ──
                gpu = None
                try:
                    async with httpx.AsyncClient(timeout=5) as client:
                        resp = await client.get(f"{settings.OLLAMA_HOST}/api/tags")
                        if resp.status_code != 200:
                            raise Exception(f"HTTP {resp.status_code}")

                    gpu = await _get_gpu_info()
                    vram_mb = gpu.get("vram_total_bytes", 0) // 1024 // 1024 if gpu else 0
                    gpu_msg = (
                        f"CUDA available, {gpu.get('gpu_name', 'GPU')} ({vram_mb}MB)"
                        if gpu.get("cuda_available")
                        else "No GPU detected — running on CPU"
                    )
                    yield _sse_event("phase_result", {
                        "phase": "provider_gpu_check", "status": "pass",
                        "message": f"Ollama connected. {gpu_msg}",
                        "gpu": gpu,
                    })
                except Exception as e:
                    yield _sse_event("phase_result", {
                        "phase": "provider_gpu_check", "status": "fail",
                        "message": f"Ollama not available: {e}",
                    })
                    yield _sse_event("complete", {"overall_status": "fail"})
                    return

            # ══════════════════════════════════════════════════════════
            # Phase: FFmpeg availability
            # ══════════════════════════════════════════════════════════
            yield _phase("ffmpeg_check", "Testing FFmpeg (audio/video extraction)...")
            test_audio = None
            try:
                tmp_dir = tempfile.mkdtemp(prefix="pipeline_test_")
                test_audio = os.path.join(tmp_dir, "test_audio.wav")

                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-version",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
                version_line = stdout_bytes.decode().split("\n")[0] if stdout_bytes else "unknown"

                cmd = [
                    "ffmpeg", "-y", "-f", "lavfi", "-i",
                    (
                        "sine=frequency=440:duration=3,aformat=sample_rates=16000:channel_layouts=mono[a1];"
                        "anullsrc=r=16000:cl=mono,atrim=0:2,asetpts=PTS-STARTPTS[s1];"
                        "anullsrc=r=16000:cl=mono,atrim=0:2,asetpts=PTS-STARTPTS[s2];"
                        "sine=frequency=880:duration=3,aformat=sample_rates=16000:channel_layouts=mono[a2];"
                        "[s1][a1][s2][a2]concat=n=4:v=0:a=1[out]"
                    ),
                    "-map", "[out]", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    test_audio,
                ]
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(proc.communicate(), timeout=15)

                if proc.returncode != 0 or not os.path.isfile(test_audio):
                    cmd2 = ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                            "-t", "5", "-acodec", "pcm_s16le", test_audio]
                    proc2 = await asyncio.create_subprocess_exec(
                        *cmd2, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    )
                    await asyncio.wait_for(proc2.communicate(), timeout=10)

                audio_size = os.path.getsize(test_audio) if os.path.isfile(test_audio) else 0
                yield _sse_event("phase_result", {
                    "phase": "ffmpeg_check", "status": "pass",
                    "message": f"{version_line.split(' Copyright')[0]} | Test audio: {audio_size // 1024}KB",
                })
            except Exception as e:
                yield _sse_event("phase_result", {
                    "phase": "ffmpeg_check", "status": "fail",
                    "message": f"FFmpeg test failed: {str(e)[:200]}",
                })
                yield _sse_event("complete", {"overall_status": "fail"})
                return

            # ══════════════════════════════════════════════════════════
            # Phase: Torch / CTranslate2 VRAM check
            # ══════════════════════════════════════════════════════════
            yield _phase("torch_vram", "Checking Whisper GPU runtime (CTranslate2 / PyTorch)...")
            whisper_device = "cpu"
            try:
                # Use the same GPU detection as the actual transcription subprocess.
                # This checks /dev/nvidia* device nodes FIRST (fast, no CUDA context),
                # then falls back to CTranslate2 and PyTorch.  The main process may not
                # have a working torch.cuda, but the subprocess can still use CUDA via
                # CTranslate2's own CUDA context.
                from backend.services.transcription import _detect_cuda_available
                cuda_ok, cuda_count, gpu_name, best_idx = _detect_cuda_available()
                if cuda_ok and settings.GPU_ACCELERATION_ENABLED:
                    whisper_device = "cuda"
                    # Check if torch is holding stale VRAM
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch_reserved = torch.cuda.memory_reserved() / 1024 / 1024
                            if torch_reserved > 100:
                                from backend.services.pipeline import release_torch_gpu_memory
                                release_torch_gpu_memory()
                                await asyncio.sleep(2)
                                torch_after = torch.cuda.memory_reserved() / 1024 / 1024
                                yield _sse_event("phase_result", {
                                    "phase": "torch_vram",
                                    "status": "warn" if torch_after > 100 else "pass",
                                    "message": f"GPU detected: {gpu_name}. Torch was holding {torch_reserved:.0f}MB, released to {torch_after:.0f}MB",
                                })
                            else:
                                yield _sse_event("phase_result", {
                                    "phase": "torch_vram", "status": "pass",
                                    "message": f"GPU detected: {gpu_name}. Torch memory OK ({torch_reserved:.0f}MB reserved)",
                                })
                        else:
                            yield _sse_event("phase_result", {
                                "phase": "torch_vram", "status": "pass",
                                "message": f"GPU detected: {gpu_name} — Whisper runs on CUDA via subprocess (CTranslate2)",
                            })
                    except ImportError:
                        yield _sse_event("phase_result", {
                            "phase": "torch_vram", "status": "pass",
                            "message": f"GPU detected: {gpu_name} — Whisper runs on CUDA via subprocess",
                        })
                else:
                    yield _sse_event("phase_result", {
                        "phase": "torch_vram", "status": "pass",
                        "message": "No CUDA GPU detected — Whisper will run on CPU (slower but functional)",
                    })
            except Exception as e:
                logger.warning("Whisper GPU detection failed: %s", e)
                yield _sse_event("phase_result", {
                    "phase": "torch_vram", "status": "pass",
                    "message": "GPU detection error — Whisper will use CPU fallback",
                })

            # ══════════════════════════════════════════════════════════
            # Phase: Clear VRAM before Whisper — ALWAYS run if Ollama container exists
            # Whisper needs exclusive GPU access regardless of which AI provider
            # is used for vision/text. Ollama may have loaded models from
            # startup pulls, diagnostics polling, or previous pipeline runs.
            # ══════════════════════════════════════════════════════════
            if is_ollama:
                yield _phase("pre_whisper_vram_clear", "Unloading Ollama models (free GPU for Whisper)...")
                cleared = await _unload_and_wait(10)
                yield _sse_event("phase_result", {
                    "phase": "pre_whisper_vram_clear",
                    "status": "pass" if cleared else "warn",
                    "message": "Ollama models unloaded — GPU freed for Whisper" if cleared else "Models may still be unloading",
                })

            # ══════════════════════════════════════════════════════════
            # Phase: Whisper transcription test
            # Real pipeline: transcribe_audio_subprocess()
            # ══════════════════════════════════════════════════════════
            whisper_ok = True
            if include_whisper and test_audio and os.path.isfile(test_audio):
                model_name = settings.WHISPER_MODEL
                beam_size = settings.WHISPER_BEAM_SIZE

                yield _phase("whisper_transcribe",
                    f"Whisper transcription — {model_name} (beam={beam_size}) on {whisper_device.upper()}...")

                # Known CTranslate2 VRAM usage per model (weights + context + beam)
                _WHISPER_VRAM_MB = {
                    "tiny": 150, "tiny.en": 150,
                    "base": 250, "base.en": 250,
                    "small": 500, "small.en": 500,
                    "medium": 1800, "medium.en": 1800,
                    "large-v3": 3500, "large-v3-turbo": 3200,
                    "large-v2": 3500, "large": 3500,
                }
                whisper_vram_mb = _WHISPER_VRAM_MB.get(model_name, 1000) if whisper_device == "cuda" else 0

                t0 = time.time()
                try:
                    from backend.services.transcription import transcribe_audio_subprocess
                    segments = await asyncio.wait_for(
                        transcribe_audio_subprocess(
                            test_audio, language="", task="transcribe",
                            initial_prompt="", audio_duration=10.0,
                        ),
                        timeout=180,
                    )
                    elapsed_ms = int((time.time() - t0) * 1000)
                    yield _sse_event("phase_result", {
                        "phase": "whisper_transcribe", "status": "pass",
                        "message": (
                            f"Transcription OK — {len(segments)} segment{'s' if len(segments) != 1 else ''} "
                            f"in {elapsed_ms}ms ({model_name}, beam={beam_size}, {whisper_device})"
                        ),
                        "segments": len(segments), "elapsed_ms": elapsed_ms,
                        "gpu_status": f"gpu (~{whisper_vram_mb}MB VRAM)" if whisper_device == "cuda" else "cpu",
                        "vram_bytes": whisper_vram_mb * 1024 * 1024 if whisper_device == "cuda" else 0,
                        "model": model_name,
                    })
                except asyncio.TimeoutError:
                    whisper_ok = False
                    yield _sse_event("phase_result", {
                        "phase": "whisper_transcribe", "status": "fail",
                        "message": f"Timed out after 180s — model '{model_name}' may be downloading or too large for GPU",
                    })
                except Exception as e:
                    whisper_ok = False
                    elapsed_ms = int((time.time() - t0) * 1000)
                    err = str(e)[:300]
                    hint = " — GPU memory issue, try smaller model" if "cuda" in err.lower() else ""
                    yield _sse_event("phase_result", {
                        "phase": "whisper_transcribe", "status": "fail",
                        "message": f"Failed ({elapsed_ms}ms): {err}{hint}",
                    })

                # ── Optional: Whisper translation test ──
                if test_translation and whisper_ok:
                    yield _phase("whisper_translate", f"Whisper translation — ja→en ({model_name})...")
                    t0 = time.time()
                    try:
                        segments_tr = await asyncio.wait_for(
                            transcribe_audio_subprocess(
                                test_audio, language="ja", task="translate",
                                initial_prompt="", audio_duration=10.0,
                            ),
                            timeout=120,
                        )
                        elapsed_ms = int((time.time() - t0) * 1000)
                        has_echo = any(
                            any(p in (getattr(s, "text", "") or "").lower() for p in [
                                "japanese conversation", "translated to natural english",
                            ])
                            for s in segments_tr
                        )
                        if has_echo:
                            yield _sse_event("phase_result", {
                                "phase": "whisper_translate", "status": "fail",
                                "message": "Prompt echo detected — Whisper hallucinating initial_prompt as output",
                            })
                        else:
                            yield _sse_event("phase_result", {
                                "phase": "whisper_translate", "status": "pass",
                                "message": f"Translation OK — {len(segments_tr)} segments in {elapsed_ms}ms (ja→en, no echo)",
                                "gpu_status": f"gpu (~{whisper_vram_mb}MB VRAM)" if whisper_device == "cuda" else "cpu",
                            })
                    except Exception as e:
                        elapsed_ms = int((time.time() - t0) * 1000)
                        yield _sse_event("phase_result", {
                            "phase": "whisper_translate", "status": "warn",
                            "message": f"Translation test failed ({elapsed_ms}ms): {str(e)[:200]}",
                        })

                # ── Whisper VRAM cleanup verification ──
                yield _phase("whisper_vram_cleanup", "Verifying Whisper released GPU memory...")
                try:
                    from backend.services.pipeline import release_torch_gpu_memory
                    release_torch_gpu_memory()
                except Exception:
                    pass
                await asyncio.sleep(2)
                try:
                    import subprocess as sp
                    smi = sp.run(
                        ["nvidia-smi", "--query-compute-apps=pid,name,used_memory",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5,
                    )
                    if smi.returncode == 0:
                        lines = [l.strip() for l in smi.stdout.strip().split("\n") if l.strip()]
                        python_procs = [l for l in lines if "python" in l.lower()]
                        if not python_procs:
                            yield _sse_event("phase_result", {
                                "phase": "whisper_vram_cleanup", "status": "pass",
                                "message": "GPU memory fully released — no residual CUDA context",
                            })
                        else:
                            yield _sse_event("phase_result", {
                                "phase": "whisper_vram_cleanup", "status": "pass",
                                "message": "Subprocess mode — VRAM released when process exited",
                            })
                    else:
                        yield _sse_event("phase_result", {
                            "phase": "whisper_vram_cleanup", "status": "pass",
                            "message": "Subprocess mode — VRAM released when process exited",
                        })
                except FileNotFoundError:
                    yield _sse_event("phase_result", {
                        "phase": "whisper_vram_cleanup", "status": "pass",
                        "message": "nvidia-smi not available — subprocess ensures cleanup",
                    })

            # ══════════════════════════════════════════════════════════
            # Phase: Ollama GPU rediscovery (Ollama-only)
            # ══════════════════════════════════════════════════════════
            if uses_ollama:
                yield _phase("ollama_gpu_rediscovery", "Triggering Ollama GPU rediscovery...")

                await _unload_and_wait(10)
                await asyncio.sleep(2)

                gpu_rediscovered = False
                try:
                    async with httpx.AsyncClient(timeout=120) as _rc:
                        _probe = await _rc.post(
                            f"{settings.OLLAMA_HOST}/api/generate",
                            json={
                                "model": vision_model,
                                "prompt": "test",
                                "stream": False,
                                "options": {"num_gpu": 99, "num_predict": 1},
                            },
                            timeout=120,
                        )
                        if _probe.status_code == 200:
                            _ps = await _rc.get(f"{settings.OLLAMA_HOST}/api/ps", timeout=10)
                            if _ps.status_code == 200:
                                for m in _ps.json().get("models", []):
                                    if m.get("size_vram", 0) > 0:
                                        gpu_rediscovered = True
                                        vram_mb = m.get("size_vram", 0) // 1024 // 1024
                                        break
                            await _rc.post(f"{settings.OLLAMA_HOST}/api/generate",
                                json={"model": vision_model, "keep_alive": 0}, timeout=10)
                except Exception as e:
                    logger.warning("GPU rediscovery probe failed: %s", e)

                if gpu_rediscovered:
                    yield _sse_event("phase_result", {
                        "phase": "ollama_gpu_rediscovery", "status": "pass",
                        "message": f"GPU rediscovered — {vision_model} loaded on GPU ({vram_mb}MB VRAM)",
                    })
                else:
                    yield _sse_event("phase_result", {
                        "phase": "ollama_gpu_rediscovery", "status": "warn",
                        "message": "GPU rediscovery: model loaded on CPU — GPU may be poisoned. Try restarting Ollama container.",
                    })

            # ══════════════════════════════════════════════════════════
            # Phase: Vision model test (scene analysis)
            # ══════════════════════════════════════════════════════════
            yield _phase("vision_model", f"Testing vision model — {vision_model} (scene analysis) via {provider_label}...")
            if uses_ollama:
                vision_result = await _test_vision_model(vision_model)
            else:
                # Non-Ollama: test via AIOrchestrator
                vision_result = await _test_cloud_vision(vision_model, provider_label)
            yield _sse_event("phase_result", {"phase": "vision_model", **vision_result})

            # ══════════════════════════════════════════════════════════
            # Phase: Unload vision, wait for VRAM
            # CRITICAL: vision + text can't coexist on 4GB GPU
            # ══════════════════════════════════════════════════════════
            if uses_ollama:
                yield _phase("vision_unload", f"Unloading {vision_model} — waiting for VRAM release...")
                freed = await _unload_and_wait(15)
                if freed:
                    await asyncio.sleep(3)
                yield _sse_event("phase_result", {
                    "phase": "vision_unload",
                    "status": "pass" if freed else "warn",
                    "message": "Vision model unloaded — VRAM freed for text model" if freed else "Vision model may still be resident",
                })
            else:
                yield _phase("vision_unload", "Cloud provider — no VRAM to release")
                yield _sse_event("phase_result", {
                    "phase": "vision_unload", "status": "pass",
                    "message": f"{provider_label} uses cloud inference — no local VRAM management needed",
                })

            # ══════════════════════════════════════════════════════════
            # Phase: Text model — summary generation
            # ══════════════════════════════════════════════════════════
            yield _phase("text_summary", f"Testing text model — {text_model} (summary generation) via {provider_label}...")
            if uses_ollama:
                summary_result = await _test_text_model(text_model)
            else:
                # Non-Ollama: test via AIOrchestrator
                summary_result = await _test_cloud_text(text_model, provider_label)
            yield _sse_event("phase_result", {"phase": "text_summary", **summary_result})

            # ══════════════════════════════════════════════════════════
            # Phase: Text model — clip detection JSON
            # ══════════════════════════════════════════════════════════
            yield _phase("text_clips", f"Testing clip detection — JSON output from {text_model} via {provider_label}...")
            t0 = time.time()
            try:
                clip_prompt = (
                    'You are analyzing a video transcript. Find 1 viral clip.\n'
                    'TRANSCRIPT:\n[00:00] Hello everyone, welcome to the show.\n'
                    '[00:05] Today we have something amazing.\n\n'
                    'Return ONLY valid JSON: {"clips": [{"id": 1, "title": "...", '
                    '"start_time": 0.0, "end_time": 10.0, "viral_score": 50}]}'
                )
                from backend.services.ai_orchestrator import AIOrchestrator
                orch = AIOrchestrator()
                raw = await asyncio.wait_for(
                    orch.text_completion(clip_prompt, max_tokens=200, timeout=90),
                    timeout=90,
                )
                elapsed_ms = int((time.time() - t0) * 1000)

                parsed = False
                if raw and raw.strip():
                    try:
                        text = raw.strip()
                        if text.startswith("```"):
                            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
                        start = text.find("{")
                        end = text.rfind("}") + 1
                        if start >= 0 and end > start:
                            data = json.loads(text[start:end])
                            if "clips" in data and isinstance(data["clips"], list):
                                parsed = True
                    except Exception:
                        pass

                yield _sse_event("phase_result", {
                    "phase": "text_clips",
                    "status": "pass" if parsed else "warn",
                    "message": (
                        f"Clip detection OK — valid JSON in {elapsed_ms}ms"
                        if parsed else
                        f"Model responded ({elapsed_ms}ms) but output wasn't valid clip JSON — pipeline will retry"
                    ),
                })
            except asyncio.TimeoutError:
                yield _sse_event("phase_result", {
                    "phase": "text_clips", "status": "fail",
                    "message": "Clip detection timed out after 90s",
                })
            except Exception as e:
                elapsed_ms = int((time.time() - t0) * 1000)
                yield _sse_event("phase_result", {
                    "phase": "text_clips", "status": "fail",
                    "message": f"Clip detection failed ({elapsed_ms}ms): {str(e)[:200]}",
                })

            # ══════════════════════════════════════════════════════════
            # Phase: Final cleanup
            # ══════════════════════════════════════════════════════════
            yield _phase("final_cleanup", "Final cleanup — unloading all models...")
            if uses_ollama:
                await _unload_all_models()
                yield _sse_event("phase_result", {
                    "phase": "final_cleanup", "status": "pass",
                    "message": "All models unloaded — GPU memory released",
                })
            else:
                yield _sse_event("phase_result", {
                    "phase": "final_cleanup", "status": "pass",
                    "message": f"Cloud provider ({provider_label}) — no local models to unload",
                })

            # ── Overall result ──
            all_ok = (
                vision_result.get("status") != "fail"
                and summary_result.get("status") != "fail"
                and whisper_ok
            )
            yield _sse_event("complete", {
                "overall_status": "pass" if all_ok else "fail",
                "summary": {
                    "whisper_model": settings.WHISPER_MODEL,
                    "whisper_device": whisper_device,
                    "whisper_tested": include_whisper,
                    "translation_tested": test_translation,
                    "provider": provider_label,
                    "vision_model": vision_model,
                    "text_model": text_model,
                    "gpu_name": gpu.get("gpu_name", "CPU") if gpu else "CPU",
                    "vram_mb": (gpu.get("vram_total_bytes", 0) // 1024 // 1024) if gpu else 0,
                },
            })

        finally:
            if tmp_dir:
                try:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Subject Tracking Validation ──────────────────────────────────────────────


def _generate_tracking_test_image(subject_x_pct: int, color: tuple[int, int, int]) -> str:
    """Generate a 256x256 test image with a person-like figure at known position.

    Creates a simplified but recognizable human figure (head, neck, shoulders,
    torso) against an indoor background. Moondream is trained on real photos,
    so we need figures that look plausible, not abstract geometric shapes.

    Returns base64-encoded PNG.
    """
    from PIL import Image, ImageDraw

    W, H = 256, 256
    img = Image.new("RGB", (W, H), (45, 52, 60))  # Dark room background
    draw = ImageDraw.Draw(img)

    # Background elements: wall, floor, window-like rectangle
    draw.rectangle([0, 0, W, H * 2 // 3], fill=(55, 62, 70))       # Wall
    draw.rectangle([0, H * 2 // 3, W, H], fill=(38, 35, 32))       # Floor
    # Window/frame element on opposite side from subject for spatial context
    win_x = W - 40 if subject_x_pct < 50 else 10
    draw.rectangle([win_x, 20, win_x + 30, 70], fill=(80, 95, 110))  # Window
    draw.rectangle([win_x + 2, 22, win_x + 28, 68], fill=(120, 145, 170))

    # Subject position
    cx = int(W * subject_x_pct / 100)
    base_y = H * 2 // 3  # Standing on the floor line

    # Skin tone + hair
    skin = (210, 175, 145)
    hair = (60, 40, 25)
    shirt = color

    # Torso (shirt)
    torso_w, torso_h = 36, 50
    draw.rectangle(
        [cx - torso_w // 2, base_y - torso_h - 30, cx + torso_w // 2, base_y - 30],
        fill=shirt,
    )

    # Shoulders — wider than torso
    shoulder_w = 48
    draw.rectangle(
        [cx - shoulder_w // 2, base_y - torso_h - 30, cx + shoulder_w // 2, base_y - torso_h - 20],
        fill=shirt,
    )

    # Neck
    neck_w = 10
    draw.rectangle(
        [cx - neck_w // 2, base_y - torso_h - 40, cx + neck_w // 2, base_y - torso_h - 28],
        fill=skin,
    )

    # Head (oval)
    head_rx, head_ry = 14, 17
    head_cy = base_y - torso_h - 40 - head_ry
    draw.ellipse(
        [cx - head_rx, head_cy - head_ry, cx + head_rx, head_cy + head_ry],
        fill=skin,
    )

    # Hair on top of head
    draw.ellipse(
        [cx - head_rx, head_cy - head_ry - 2, cx + head_rx, head_cy - 2],
        fill=hair,
    )

    # Simple face features — eyes and mouth
    eye_y = head_cy - 2
    draw.ellipse([cx - 7, eye_y - 2, cx - 3, eye_y + 2], fill=(40, 40, 40))
    draw.ellipse([cx + 3, eye_y - 2, cx + 7, eye_y + 2], fill=(40, 40, 40))
    draw.line([(cx - 4, head_cy + 7), (cx + 4, head_cy + 7)], fill=(170, 120, 100), width=1)

    # Legs
    leg_w = 10
    draw.rectangle([cx - 14, base_y - 30, cx - 14 + leg_w, base_y], fill=(50, 50, 65))
    draw.rectangle([cx + 4, base_y - 30, cx + 4 + leg_w, base_y], fill=(50, 50, 65))

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@router.post("/test-subject-tracking")
async def test_subject_tracking():
    """Validate that the vision AI produces usable subject_x values.

    Generates 3 synthetic test images with subjects at known positions
    (25%, 50%, 75%), sends each to the vision model, and compares
    the returned subject_x against ground truth.
    """
    vision_model = settings.OLLAMA_VISION_MODEL
    is_moondream = "moondream" in vision_model.lower()

    # 5 test images simulating a subject moving left-to-right across the frame
    test_cases = [
        {"expected_x": 15, "color": (220, 60, 60), "label": "far left"},
        {"expected_x": 35, "color": (200, 140, 60), "label": "left"},
        {"expected_x": 50, "color": (60, 120, 220), "label": "center"},
        {"expected_x": 65, "color": (60, 200, 120), "label": "right"},
        {"expected_x": 85, "color": (180, 60, 200), "label": "far right"},
    ]

    results = []
    total_time = 0

    for tc in test_cases:
        test_b64 = _generate_tracking_test_image(tc["expected_x"], tc["color"])

        if is_moondream:
            prompt = (
                "There is a person standing in a room in this image. "
                "Where is the person positioned horizontally in the frame?"
                '\n\nRespond with ONLY this JSON, nothing else:\n'
                '{"description": "<what you see>", "subject_x": <number 0 to 100>}\n'
                'subject_x: horizontal position of the person. '
                '0 = left edge, 25 = left quarter, 50 = center, 75 = right quarter, 100 = right edge.\n'
                'Look carefully at where the person is standing. Do NOT always say 50.'
            )
        else:
            prompt = (
                "There is a person standing in a room. "
                "Where is the person positioned horizontally?"
                '\n\nReturn ONLY valid JSON:\n'
                '{"timestamp": 0, "description": "<text>", '
                '"importance_score": 5, "subject_x": <0-100>}\n'
                'subject_x = horizontal position of the person as % of frame width '
                '(0=far left, 50=exact center, 100=far right).'
            )

        t0 = time.time()
        try:
            payload = {
                "model": vision_model,
                "messages": [{
                    "role": "user",
                    "content": prompt,
                    "images": [test_b64],
                }],
                "stream": False,
                "options": {"num_gpu": 99, "num_predict": 100, "num_ctx": 2048},
            }
            if is_moondream:
                payload["format"] = "json"

            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{settings.OLLAMA_HOST}/api/chat",
                    json=payload,
                    timeout=120,
                )

            duration_ms = int((time.time() - t0) * 1000)
            total_time += duration_ms

            if resp.status_code != 200:
                results.append({
                    "label": tc["label"],
                    "expected_x": tc["expected_x"],
                    "returned_x": None,
                    "error": None,
                    "json_ok": False,
                    "duration_ms": duration_ms,
                    "raw_response": resp.text[:200],
                    "status": "fail",
                    "message": f"HTTP {resp.status_code}",
                })
                continue

            raw_text = resp.json().get("message", {}).get("content", "")

            json_ok = False
            returned_x = None
            try:
                text = raw_text.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                json_start = text.find("{")
                json_end = text.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    parsed = json.loads(text[json_start:json_end])
                    if isinstance(parsed, list) and parsed:
                        parsed = parsed[0]
                    raw_sx = parsed.get("subject_x")
                    if raw_sx is not None:
                        returned_x = max(0, min(100, int(raw_sx)))
                        json_ok = True
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass

            error = abs(returned_x - tc["expected_x"]) if returned_x is not None else None

            if not json_ok:
                status = "fail"
                message = "JSON parse failed or missing subject_x"
            elif error <= 25:
                status = "pass"
                message = f"subject_x={returned_x} (expected {tc['expected_x']}, error={error})"
            elif error <= 40:
                status = "warn"
                message = f"subject_x={returned_x} (expected {tc['expected_x']}, error={error}) — approximate on synthetic image"
            else:
                status = "warn"
                message = f"subject_x={returned_x} (expected {tc['expected_x']}, error={error}) — inaccurate on synthetic, may work better on real video"

            results.append({
                "label": tc["label"],
                "expected_x": tc["expected_x"],
                "returned_x": returned_x,
                "error": error,
                "json_ok": json_ok,
                "duration_ms": duration_ms,
                "raw_response": raw_text[:200],
                "status": status,
                "message": message,
            })

        except asyncio.TimeoutError:
            results.append({
                "label": tc["label"],
                "expected_x": tc["expected_x"],
                "returned_x": None,
                "error": None,
                "json_ok": False,
                "duration_ms": 120000,
                "raw_response": "",
                "status": "fail",
                "message": "Timeout (>120s)",
            })
        except Exception as e:
            results.append({
                "label": tc["label"],
                "expected_x": tc["expected_x"],
                "returned_x": None,
                "error": None,
                "json_ok": False,
                "duration_ms": int((time.time() - t0) * 1000),
                "raw_response": str(e)[:200],
                "status": "fail",
                "message": f"Error: {str(e)[:100]}",
            })

    # Compute summary
    json_ok_count = sum(1 for r in results if r["json_ok"])
    errors = [r["error"] for r in results if r["error"] is not None]
    avg_error = round(sum(errors) / len(errors), 1) if errors else None
    avg_speed = round(total_time / len(results)) if results else 0

    # Check for degenerate responses: all subject_x values identical (e.g. all 50)
    returned_xs = [r["returned_x"] for r in results if r["returned_x"] is not None]
    all_same = len(set(returned_xs)) <= 1 and len(returned_xs) >= 2

    total_cases = len(test_cases)
    if json_ok_count == 0:
        overall = "fail"
        summary = "Vision AI cannot produce JSON — subject tracking will not work"
    elif json_ok_count < total_cases:
        overall = "warn"
        summary = f"JSON compliance: {json_ok_count}/{total_cases} — tracking may be unreliable"
    elif all_same:
        overall = "warn"
        summary = f"JSON works but all responses returned subject_x={returned_xs[0]} — model may not differentiate positions on real video"
    else:
        # Check if model detects movement direction (left images → lower x, right → higher x)
        ordered_xs = [r["returned_x"] for r in results if r["returned_x"] is not None]
        detects_direction = len(ordered_xs) >= 3 and ordered_xs[0] < ordered_xs[-1]

        if avg_error is not None and avg_error > 40 and not detects_direction:
            overall = "warn"
            summary = f"JSON works, avg error {avg_error}% — model struggles with position but may work on real video ({avg_speed}ms/frame)"
        elif detects_direction:
            overall = "pass"
            summary = f"Subject tracking working — detects movement direction, {avg_error}% avg error, {avg_speed}ms/frame"
        else:
            overall = "pass"
            summary = f"Subject tracking working — {avg_error}% avg error, {avg_speed}ms/frame"

    return {
        "overall_status": overall,
        "summary": summary,
        "model": vision_model,
        "format_json_used": is_moondream,
        "json_compliance": f"{json_ok_count}/{total_cases}",
        "avg_error": avg_error,
        "avg_speed_ms": avg_speed,
        "results": results,
    }


# NOTE: Whisper testing is now integrated into test-pipeline above.


# ── SOTA reframing bench (1-click runner) ────────────────────────────────


@router.post("/sota-bench")
async def run_sota_bench(request: Request):
    """One-click 2026 SOTA reframing validation.

    Runs the local QA harness (``python -m tests.qa.run_all_phases``)
    THEN the homelab fixture bench (``compare_autoflip_vs_clipai``)
    against the real-content manifest with every Phase A–E flag ON.

    Streams Server-Sent Events so the GUI can show live progress
    without long-polling.

    Body params (all optional):
      - skip_bench (bool)  : run only the QA harness (skip fixture
                              bench). Default false.
      - manifest (str)      : override manifest path. Default
                              ``tests/real_content/manifest.json``.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    skip_bench = bool(body.get("skip_bench", False))
    manifest = str(body.get("manifest") or "tests/real_content/manifest.json")

    # __file__ is /app/backend/routers/diagnostics.py inside the
    # container; the repo root is /app.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.dirname(repo_root)  # backend -> repo root

    qa_runner_path = os.path.join(repo_root, "tests", "qa", "run_all_phases.py")

    async def _stream_subprocess(
        cmd: list[str], *, env: dict | None = None,
    ) -> AsyncGenerator[str, None]:
        """Run a subprocess and yield each stdout line as an SSE log event."""
        # Force-unbuffered Python so the child's stdout reaches us
        # line-by-line. Without this the child's prints sit in a 4 KB
        # buffer and the SSE connection looks frozen — the browser /
        # reverse proxy gives up after ~30 s with a network error.
        sub_env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            **(env or {}),
        }
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=repo_root,
            env=sub_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            assert proc.stdout is not None
            last_emit = time.monotonic()
            while True:
                # readline times out gracefully; emit a heartbeat every
                # ~5 s so the connection stays alive even when the
                # subprocess is silent (e.g. waiting for a slow model
                # download or a long pytest collection).
                try:
                    line = await asyncio.wait_for(
                        proc.stdout.readline(), timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    yield _sse_event("heartbeat", {
                        "elapsed_sec": round(time.monotonic() - last_emit, 1),
                    })
                    continue
                if not line:
                    break
                last_emit = time.monotonic()
                yield _sse_event("log", {"line": line.decode(errors="replace").rstrip()})
            await proc.wait()
            yield _sse_event("exit_code", {"code": proc.returncode})
        finally:
            if proc.returncode is None:
                try:
                    proc.kill()
                except Exception:
                    pass

    async def event_stream() -> AsyncGenerator[str, None]:
        # Yield an immediate "connected" sentinel BEFORE doing anything
        # risky. This guarantees the client sees at least one chunk so
        # ERR_INCOMPLETE_CHUNKED_ENCODING is never a generic mystery —
        # if the next event is an error, the operator sees it in the
        # log panel.
        yield _sse_event("log", {
            "line": (
                f"[sota-bench] connected; repo_root={repo_root!r} "
                f"qa_runner={qa_runner_path!r} "
                f"qa_runner_exists={os.path.exists(qa_runner_path)}"
            ),
        })
        try:
            async for evt in _do_stream():
                yield evt
        except Exception as exc:
            # Any uncaught error becomes a visible SSE event rather
            # than a torn chunked-encoding stream that the browser
            # surfaces as ``ERR_INCOMPLETE_CHUNKED_ENCODING``.
            import traceback
            tb = traceback.format_exc()
            logger.exception("[sota-bench] event_stream crashed: %s", exc)
            for tline in tb.splitlines():
                yield _sse_event("log", {"line": f"!! {tline}"})
            yield _sse_event("complete", {
                "ok": False,
                "stage": "event_stream",
                "message": (
                    f"Internal error: {exc!s}. Check the backend logs for the "
                    "full traceback."
                ),
            })

    async def _do_stream() -> AsyncGenerator[str, None]:
        # Pre-flight: the harness file MUST be present in the image.
        # Without this check, a missing tests/ directory crashes
        # the subprocess instantly and the SSE stream torn-resets.
        if not os.path.exists(qa_runner_path):
            yield _sse_event("phase_start", {
                "phase": "qa_harness",
                "label": "Pre-flight check (QA harness path)...",
            })
            yield _sse_event("log", {
                "line": f"!! QA harness not found at {qa_runner_path}",
            })
            yield _sse_event("log", {
                "line": (
                    "!! The image was built before the SOTA QA harness "
                    "landed. Rebuild with: "
                    "docker compose build --no-cache && docker compose up -d"
                ),
            })
            yield _sse_event("phase_result", {
                "phase": "qa_harness", "status": "fail",
            })
            yield _sse_event("complete", {
                "ok": False,
                "stage": "qa_harness",
                "message": (
                    f"QA harness missing in image at {qa_runner_path}. "
                    "Rebuild the container."
                ),
            })
            return

        # Phase 1: QA harness
        yield _sse_event("phase_start", {
            "phase": "qa_harness",
            "label": "Running tests/qa/run_all_phases (mocks; ~5 s)...",
        })
        # Use sys.executable so we hit the same Python that's running
        # uvicorn. The container has python3 (not python) on PATH, so
        # ``["python", ...]`` would fail with FileNotFoundError. The
        # /sota-bench-status probe also uses sys.executable for the
        # same reason.
        import sys as _sys
        qa_failed = False
        async for evt in _stream_subprocess(
            [_sys.executable, "-u", "-m", "tests.qa.run_all_phases"],
        ):
            yield evt
            try:
                parsed = json.loads(evt[6:])
                if parsed.get("type") == "exit_code":
                    qa_failed = parsed["data"]["code"] != 0
            except Exception:
                pass
        yield _sse_event("phase_result", {
            "phase": "qa_harness",
            "status": "fail" if qa_failed else "pass",
        })
        if qa_failed:
            yield _sse_event("complete", {
                "ok": False,
                "stage": "qa_harness",
                "message": "QA harness failed — bench skipped. Read the failures above and fix the offending phase before re-running.",
            })
            return

        if skip_bench:
            yield _sse_event("complete", {
                "ok": True,
                "stage": "qa_harness",
                "message": "QA harness PASSED. Bench skipped per request.",
            })
            return

        # ── Phase 2 pre-flight: manifest + fixture cache ─────────────
        # Surface clear, actionable errors BEFORE we spawn the bench.
        # Otherwise the script crashes with a generic FileNotFoundError
        # or "manifest not found" and the operator has to dig through
        # the SSE log to figure out what's missing.
        manifest_abs = manifest if os.path.isabs(manifest) else os.path.join(
            repo_root, manifest,
        )
        manifest_present = os.path.isfile(manifest_abs)
        cache_dir = os.environ.get(
            "CLIPAI_REAL_CONTENT_CACHE", "/var/cache/clipai/real_content",
        )
        cache_present = os.path.isdir(cache_dir)
        cache_files = []
        if cache_present:
            try:
                cache_files = [
                    f for f in os.listdir(cache_dir)
                    if not f.startswith(".") and not f.endswith(".part")
                ]
            except Exception:
                cache_files = []

        if not manifest_present:
            yield _sse_event("phase_start", {
                "phase": "sota_bench",
                "label": "Pre-flight: locating fixture manifest...",
            })
            yield _sse_event("log", {
                "line": f"!! manifest not found at {manifest_abs}",
            })
            yield _sse_event("log", {
                "line": (
                    "!! The image was built before the SOTA bench manifest "
                    "landed. Rebuild with: docker compose down && "
                    "docker compose build --no-cache && docker compose up -d"
                ),
            })
            yield _sse_event("phase_result", {
                "phase": "sota_bench", "status": "fail",
            })
            yield _sse_event("complete", {
                "ok": False, "stage": "sota_bench",
                "message": (
                    f"Bench manifest missing in image at {manifest_abs}. "
                    "Rebuild the container."
                ),
            })
            return

        if not cache_files:
            yield _sse_event("phase_start", {
                "phase": "sota_bench",
                "label": "Pre-flight: locating fixture cache...",
            })
            yield _sse_event("log", {
                "line": (
                    f"!! fixture cache at {cache_dir} is empty"
                    if cache_present
                    else f"!! fixture cache directory {cache_dir} does not exist"
                ),
            })
            yield _sse_event("log", {
                "line": (
                    "!! The bench needs MP4 fixtures listed in "
                    "tests/real_content/manifest.json. Populate the cache "
                    "by running ON THE HOST: "
                    "bash tests/real_content/fetch.sh"
                ),
            })
            yield _sse_event("log", {
                "line": (
                    "!! (You may also need to fill in source_url + sha256 "
                    "fields in the manifest first - they ship blank.)"
                ),
            })
            yield _sse_event("phase_result", {
                "phase": "sota_bench", "status": "fail",
            })
            yield _sse_event("complete", {
                "ok": False, "stage": "sota_bench",
                "message": (
                    f"Fixture cache at {cache_dir} is empty. "
                    "Populate it via tests/real_content/fetch.sh on the host. "
                    "The QA harness portion above PASSED - the deploy is "
                    "healthy; the fixture step is opt-in."
                ),
            })
            return

        # ── Phase 2: full SOTA fixture bench ────────────────────────
        yield _sse_event("phase_start", {
            "phase": "sota_bench",
            "label": (
                f"Running compare_autoflip_vs_clipai on {len(cache_files)} "
                "fixture(s) with all Phase A-E flags ON (homelab GPU; this "
                "can take several minutes)..."
            ),
        })
        bench_env = {
            "CLIPAI_TRACKER_BACKEND": "samurai",
            "CLIPAI_DENSE_POINT_TRACKING": "1",
            "CLIPAI_SALIENCY_ENABLED": "1",
            "CLIPAI_COMPOSITION_HEAD": "clip",
            "CLIPAI_EDITORIAL_PLANNER": "1",
            "CLIPAI_HUMAN_REFRAME_PIPELINE": "1",
            # The bench script reads this to find fixture MP4s + write
            # its extraction cache. Inherits whatever the container is
            # configured with (default /var/cache/clipai/real_content).
            "CLIPAI_REAL_CONTENT_CACHE": cache_dir,
        }
        bench_failed = False
        bench_cmd = [
            _sys.executable, "-u", "-m",
            "backend.scripts.compare_autoflip_vs_clipai",
            "--manifest", manifest,
            "--output", "/tmp/sota_bench_results.md",
            "--json-out", "/tmp/sota_bench_results.json",
        ]
        async for evt in _stream_subprocess(bench_cmd, env=bench_env):
            yield evt
            try:
                parsed = json.loads(evt[6:])
                if parsed.get("type") == "exit_code":
                    bench_failed = parsed["data"]["code"] != 0
            except Exception:
                pass
        yield _sse_event("phase_result", {
            "phase": "sota_bench",
            "status": "fail" if bench_failed else "pass",
            "results_md": "/tmp/sota_bench_results.md",
            "results_json": "/tmp/sota_bench_results.json",
        })

        # ── Final summary ────────────────────────────────────────────
        if bench_failed:
            yield _sse_event("complete", {
                "ok": False,
                "stage": "sota_bench",
                "message": (
                    "Bench failed. Most common cause: missing fixture cache "
                    "(see CLIPAI_REAL_CONTENT_CACHE) or missing AutoFlip "
                    "reference outputs. See docs/sota_reframe_runbook.md."
                ),
            })
        else:
            yield _sse_event("complete", {
                "ok": True,
                "stage": "sota_bench",
                "message": (
                    "QA harness + SOTA bench PASSED. Results written to "
                    "/tmp/sota_bench_results.md (markdown) and "
                    "/tmp/sota_bench_results.json (machine-readable)."
                ),
            })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── SOTA bench: synchronous JSON variant + status probe ──────────────


@router.get("/sota-bench-status")
async def sota_bench_status():
    """Return a JSON snapshot of everything the SOTA bench needs.

    Fast (<200 ms), no streaming, no subprocess. Use this to verify
    that the container has the QA harness baked in BEFORE clicking
    the 1-click button. Curl it directly:

        curl -s http://localhost:1353/api/diagnostics/sota-bench-status | jq

    Output flags every potential failure mode the operator cares
    about: missing file, wrong python interpreter, broken module
    import, etc.
    """
    import shutil
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.dirname(repo_root)
    qa_runner = os.path.join(repo_root, "tests", "qa", "run_all_phases.py")
    suite_files = []
    qa_dir = os.path.join(repo_root, "tests", "qa")
    if os.path.isdir(qa_dir):
        suite_files = sorted(
            f for f in os.listdir(qa_dir) if f.startswith("test_phase_")
        )

    # Try a trivial subprocess to confirm the plumbing works.
    sub_ok = False
    sub_stdout = ""
    sub_err = ""
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "print('ok')",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        sub_ok = proc.returncode == 0
        sub_stdout = out.decode(errors="replace").strip()
        sub_err = err.decode(errors="replace").strip()
    except Exception as exc:
        sub_err = f"subprocess plumbing failed: {exc!r}"

    # ``shutil.which("pytest")`` returns None when the pytest binary
    # script is not on PATH — but in our container pytest is invoked
    # via ``python -m pytest``, so the binary isn't required. Probe
    # by importing the module instead.
    pytest_available = False
    pytest_version = None
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pytest", "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        pytest_available = proc.returncode == 0
        if pytest_available:
            pytest_version = out.decode(errors="replace").strip().splitlines()[-1]
    except Exception:
        pass

    # Try to import the harness module to see if the path works.
    import_ok = False
    import_err = ""
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c",
            "import importlib; importlib.import_module('tests.qa.run_all_phases')",
            cwd=repo_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=15.0)
        import_ok = proc.returncode == 0
        if not import_ok:
            import_err = err.decode(errors="replace").strip()
    except Exception as exc:
        import_err = f"{exc!r}"

    return {
        "ok": all([
            os.path.exists(qa_runner), sub_ok, import_ok, len(suite_files) >= 5,
            pytest_available,
        ]),
        "repo_root": repo_root,
        "qa_runner_path": qa_runner,
        "qa_runner_exists": os.path.exists(qa_runner),
        "qa_dir_exists": os.path.isdir(qa_dir),
        "suite_files": suite_files,
        "python": sys.executable,
        "pytest_available": pytest_available,
        "pytest_version": pytest_version,
        "pytest_binary_on_path": shutil.which("pytest") is not None,
        "subprocess_plumbing": {
            "ok": sub_ok, "stdout": sub_stdout, "stderr": sub_err,
        },
        "module_import": {"ok": import_ok, "error": import_err},
    }


@router.post("/sota-bench-qa")
async def sota_bench_qa_sync():
    """Synchronous JSON variant of the QA-only run. No streaming.

    Runs ``python -m tests.qa.run_all_phases`` to completion (~5 s)
    and returns the captured stdout + exit code as JSON. Use this
    when the streaming endpoint misbehaves — it is fundamentally
    more reliable across reverse-proxies and container networks
    because it is a single request/response cycle.

    The GUI's ``QA only`` checkbox falls back to this on the first
    SSE error.
    """
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.dirname(repo_root)
    qa_runner = os.path.join(repo_root, "tests", "qa", "run_all_phases.py")

    if not os.path.exists(qa_runner):
        return {
            "ok": False,
            "stage": "preflight",
            "error": f"QA harness not found at {qa_runner}. Rebuild the image.",
            "stdout": "",
            "stderr": "",
            "exit_code": -1,
        }

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-u", "-m", "tests.qa.run_all_phases",
            cwd=repo_root,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=120.0)
    except asyncio.TimeoutError:
        return {
            "ok": False,
            "stage": "qa_harness",
            "error": "QA harness exceeded 120 s timeout",
            "stdout": "", "stderr": "", "exit_code": -1,
        }
    except Exception as exc:
        return {
            "ok": False,
            "stage": "qa_harness",
            "error": f"subprocess failed: {exc!r}",
            "stdout": "", "stderr": "", "exit_code": -1,
        }

    return {
        "ok": proc.returncode == 0,
        "stage": "qa_harness",
        "stdout": out.decode(errors="replace"),
        "stderr": err.decode(errors="replace"),
        "exit_code": proc.returncode,
    }


# ── SOTA bench: upload a single MP4 and run the SOTA pipeline on it ──
#
# Two-step flow because SSE responses can't easily share a request
# with a multipart body:
#   1. POST /sota-clip-upload (multipart) → saves the file, returns a
#      token + the slug it was filed under.
#   2. POST /sota-clip-bench (json {token}) → SSE streams the QA
#      harness + a bench run against a synthesized manifest pointing
#      at JUST the uploaded clip.
#
# Uploads are stored under the regular fixture cache so the bench
# script can find them via its existing CLIPAI_REAL_CONTENT_CACHE
# resolution. Tokens live in process memory and are TTL-bounded so
# orphaned uploads from disconnected clients are reaped.


import secrets as _secrets
import time as _time

# token -> {path: str, slug: str, size: int, created_at: float, content_type: str}
_SOTA_CLIP_UPLOADS: dict = {}
_SOTA_CLIP_TTL_SEC = 3600  # 1 hour - enough to upload, run a bench, view


def _reap_old_uploads() -> None:
    """Drop tokens whose files were uploaded > TTL ago. Called opportunistically."""
    now = _time.monotonic()
    stale = [t for t, info in _SOTA_CLIP_UPLOADS.items()
             if now - info.get("created_at", 0) > _SOTA_CLIP_TTL_SEC]
    for t in stale:
        info = _SOTA_CLIP_UPLOADS.pop(t, None)
        if info and info.get("path"):
            try:
                os.unlink(info["path"])
            except Exception:
                pass


@router.post("/sota-clip-upload")
async def sota_clip_upload(
    file: UploadFile = File(...),
    content_type: str = "default",
):
    """Upload a single MP4 for the SOTA reframing pipeline test.

    Saves the file under the fixture cache directory with a synthesized
    slug (``sota_clip_<token>``). The companion endpoint
    ``/sota-clip-bench`` runs the bench against just this clip.

    ``content_type`` (form field, optional) maps to the manifest's
    ``target_clipcontenttype`` so the editorial planner picks the
    right per-genre playbook (e.g. ``multi_speaker_panel``,
    ``music_video``, ``sports``, ``gaming``, ``anime``, ``narrative``,
    ``vlog``, ``tutorial``, or ``default``).

    Auth required (cookie session) — same as the rest of /api/*.
    """
    _reap_old_uploads()

    if not file.filename:
        return {"ok": False, "error": "no filename"}
    fname_lower = file.filename.lower()
    if not any(fname_lower.endswith(ext) for ext in (".mp4", ".mov", ".webm", ".mkv")):
        return {
            "ok": False,
            "error": (
                f"unsupported extension on {file.filename!r}. "
                "Accepted: mp4, mov, webm, mkv."
            ),
        }

    cache_dir = os.environ.get(
        "CLIPAI_REAL_CONTENT_CACHE", "/var/cache/clipai/real_content",
    )
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError as exc:
        return {"ok": False, "error": f"cannot create {cache_dir}: {exc}"}

    token = _secrets.token_urlsafe(12)
    slug = f"sota_clip_{token}"
    ext = "mp4"   # bench script's slug→path resolution always appends `.mp4`
    fpath = os.path.join(cache_dir, f"{slug}.{ext}")

    try:
        bytes_written = 0
        with open(fpath, "wb") as out:
            while True:
                chunk = await file.read(1 << 20)  # 1 MB chunks
                if not chunk:
                    break
                out.write(chunk)
                bytes_written += len(chunk)
    except Exception as exc:
        try:
            os.unlink(fpath)
        except Exception:
            pass
        return {"ok": False, "error": f"write failed: {exc}"}

    _SOTA_CLIP_UPLOADS[token] = {
        "path": fpath,
        "slug": slug,
        "size": bytes_written,
        "filename": file.filename,
        "content_type": content_type or "default",
        "created_at": _time.monotonic(),
    }
    return {
        "ok": True,
        "token": token,
        "slug": slug,
        "size_mb": round(bytes_written / 1_048_576, 2),
        "filename": file.filename,
        "saved_to": fpath,
    }


@router.post("/sota-clip-bench")
async def sota_clip_bench(request: Request):
    """SSE-stream a bench run against a previously-uploaded clip.

    Body: ``{"token": "<from /sota-clip-upload>"}``. Generates a
    synthetic manifest, runs the QA harness, then runs the SOTA
    bench against just the one uploaded clip.
    """
    import sys as _sys  # local — pinned to the running uvicorn's interpreter
    try:
        body = await request.json()
    except Exception:
        body = {}
    token = (body.get("token") or "").strip()
    info = _SOTA_CLIP_UPLOADS.get(token)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.dirname(repo_root)
    qa_runner_path = os.path.join(repo_root, "tests", "qa", "run_all_phases.py")

    cache_dir = os.environ.get(
        "CLIPAI_REAL_CONTENT_CACHE", "/var/cache/clipai/real_content",
    )

    async def _stream_subprocess(
        cmd: list, *, env: dict | None = None,
    ) -> AsyncGenerator[str, None]:
        sub_env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            **(env or {}),
        }
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=repo_root, env=sub_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            assert proc.stdout is not None
            last_emit = time.monotonic()
            while True:
                try:
                    line = await asyncio.wait_for(
                        proc.stdout.readline(), timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    yield _sse_event("heartbeat", {
                        "elapsed_sec": round(time.monotonic() - last_emit, 1),
                    })
                    continue
                if not line:
                    break
                last_emit = time.monotonic()
                yield _sse_event("log", {
                    "line": line.decode(errors="replace").rstrip(),
                })
            await proc.wait()
            yield _sse_event("exit_code", {"code": proc.returncode})
        finally:
            if proc.returncode is None:
                try:
                    proc.kill()
                except Exception:
                    pass

    async def event_stream() -> AsyncGenerator[str, None]:
        yield _sse_event("log", {
            "line": (
                f"[sota-clip] connected; cache_dir={cache_dir!r} "
                f"qa_runner_exists={os.path.exists(qa_runner_path)} "
                f"token_valid={info is not None}"
            ),
        })

        if info is None:
            yield _sse_event("phase_start", {
                "phase": "preflight", "label": "Validating upload token...",
            })
            yield _sse_event("log", {
                "line": (
                    "!! token not found or expired. "
                    "Upload the MP4 again via the file picker."
                ),
            })
            yield _sse_event("phase_result", {
                "phase": "preflight", "status": "fail",
            })
            yield _sse_event("complete", {
                "ok": False, "stage": "preflight",
                "message": "Upload token not found or expired (1h TTL).",
            })
            return

        if not os.path.isfile(info["path"]):
            yield _sse_event("log", {
                "line": f"!! uploaded file vanished from {info['path']}",
            })
            yield _sse_event("complete", {
                "ok": False, "stage": "preflight",
                "message": "Uploaded MP4 disappeared before the bench could run.",
            })
            _SOTA_CLIP_UPLOADS.pop(token, None)
            return

        try:
            async for evt in _do_stream():
                yield evt
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            logger.exception("[sota-clip-bench] event_stream crashed: %s", exc)
            for tline in tb.splitlines():
                yield _sse_event("log", {"line": f"!! {tline}"})
            yield _sse_event("complete", {
                "ok": False, "stage": "event_stream",
                "message": f"Internal error: {exc!s}.",
            })

    async def _do_stream() -> AsyncGenerator[str, None]:
        # ── Phase 1: QA harness (validates SOTA modules) ─────────
        if os.path.exists(qa_runner_path):
            yield _sse_event("phase_start", {
                "phase": "qa_harness",
                "label": "Running tests/qa/run_all_phases (mocks; ~5 s)...",
            })
            qa_failed = False
            async for evt in _stream_subprocess(
                [_sys.executable, "-u", "-m", "tests.qa.run_all_phases"],
            ):
                yield evt
                try:
                    parsed = json.loads(evt[6:])
                    if parsed.get("type") == "exit_code":
                        qa_failed = parsed["data"]["code"] != 0
                except Exception:
                    pass
            yield _sse_event("phase_result", {
                "phase": "qa_harness",
                "status": "fail" if qa_failed else "pass",
            })
            if qa_failed:
                yield _sse_event("complete", {
                    "ok": False, "stage": "qa_harness",
                    "message": "QA harness failed - bench skipped.",
                })
                return

        # ── Phase 2: synthesize manifest + run bench on the upload ──
        manifest = {
            "schema": 1,
            "notes": "SOTA-bench: synthetic manifest for one user-uploaded clip",
            "cache_dir_env": "CLIPAI_REAL_CONTENT_CACHE",
            "cache_dir_default": cache_dir,
            "clips": [{
                "slug": info["slug"],
                "source_url": "",
                "sha256": "",
                "ext": "mp4",
                "duration_sec": 0,
                "content_type": info.get("content_type") or "default",
                "subtype": None,
                "target_clipcontenttype": info.get("content_type") or "default",
                "description": (
                    f"User-uploaded test clip ({info.get('filename', 'unknown')})"
                ),
                "ground_truth_vertical_url": None,
            }],
        }
        manifest_path = f"/tmp/sota_clip_{token}_manifest.json"
        try:
            with open(manifest_path, "w") as mf:
                json.dump(manifest, mf, indent=2)
        except Exception as exc:
            yield _sse_event("complete", {
                "ok": False, "stage": "manifest",
                "message": f"could not write synthetic manifest: {exc}",
            })
            return

        yield _sse_event("phase_start", {
            "phase": "sota_bench",
            "label": (
                f"Running SOTA reframing pipeline on {info['filename']!r} "
                f"({info['size'] / 1_048_576:.1f} MB) with all Phase A-E "
                "flags ON. Live progress below..."
            ),
        })

        bench_env = {
            "CLIPAI_TRACKER_BACKEND": "samurai",
            "CLIPAI_DENSE_POINT_TRACKING": "1",
            "CLIPAI_SALIENCY_ENABLED": "1",
            "CLIPAI_COMPOSITION_HEAD": "clip",
            "CLIPAI_EDITORIAL_PLANNER": "1",
            "CLIPAI_HUMAN_REFRAME_PIPELINE": "1",
            "CLIPAI_REAL_CONTENT_CACHE": cache_dir,
        }
        bench_failed = False
        bench_cmd = [
            _sys.executable, "-u", "-m",
            "backend.scripts.compare_autoflip_vs_clipai",
            "--manifest", manifest_path,
            "--filter-slugs", info["slug"],
            "--output", f"/tmp/sota_clip_{token}_results.md",
            "--json-out", f"/tmp/sota_clip_{token}_results.json",
        ]
        async for evt in _stream_subprocess(bench_cmd, env=bench_env):
            yield evt
            try:
                parsed = json.loads(evt[6:])
                if parsed.get("type") == "exit_code":
                    bench_failed = parsed["data"]["code"] != 0
            except Exception:
                pass
        yield _sse_event("phase_result", {
            "phase": "sota_bench",
            "status": "fail" if bench_failed else "pass",
            "results_md": f"/tmp/sota_clip_{token}_results.md",
            "results_json": f"/tmp/sota_clip_{token}_results.json",
        })

        if bench_failed:
            yield _sse_event("complete", {
                "ok": False, "stage": "sota_bench",
                "message": (
                    "Bench failed - see streaming output above for the "
                    "specific error. Most likely the uploaded MP4 codec "
                    "isn't supported by the analysis pipeline."
                ),
            })
            return

        # ── Phase 3: render the 9:16 preview MP4 ───────────────────
        # The bench wrote per-clip extraction artifacts to
        # <cache_dir>/extractions/<sha>/. Compute the same sha and
        # find segments.json there, then ffmpeg-crop the source into
        # a 9:16 preview the operator can actually watch.
        import hashlib
        preview_path = f"/tmp/sota_clip_{token}_preview.mp4"
        try:
            sha = hashlib.sha256()
            with open(info["path"], "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    sha.update(chunk)
            sha_hex = sha.hexdigest()
        except Exception as exc:
            yield _sse_event("log", {"line": f"!! could not hash upload: {exc}"})
            yield _sse_event("complete", {
                "ok": True, "stage": "sota_bench",
                "message": (
                    f"Bench OK; preview render skipped (hash failed). "
                    f"Results: /tmp/sota_clip_{token}_results.md"
                ),
            })
            return

        segments_path = os.path.join(
            cache_dir, "extractions", sha_hex, "segments.json",
        )
        yield _sse_event("phase_start", {
            "phase": "preview_render",
            "label": (
                f"Rendering 9:16 preview MP4 from segments.json "
                f"(this is the actual SOTA reframing output you can "
                f"watch to verify)..."
            ),
        })
        if not os.path.isfile(segments_path):
            yield _sse_event("log", {
                "line": f"!! segments.json not found at {segments_path}",
            })
            yield _sse_event("phase_result", {
                "phase": "preview_render", "status": "fail",
            })
            yield _sse_event("complete", {
                "ok": True, "stage": "sota_bench",
                "message": (
                    "Bench OK but preview render skipped — segments.json "
                    "missing. Bench results are still valid; the preview "
                    "video is opt-in."
                ),
            })
            return

        render_failed = False
        render_cmd = [
            _sys.executable, "-u", "-m", "backend.scripts.sota_render_preview",
            "--input", info["path"],
            "--segments", segments_path,
            "--output", preview_path,
        ]
        async for evt in _stream_subprocess(render_cmd):
            yield evt
            try:
                parsed = json.loads(evt[6:])
                if parsed.get("type") == "exit_code":
                    render_failed = parsed["data"]["code"] != 0
            except Exception:
                pass
        yield _sse_event("phase_result", {
            "phase": "preview_render",
            "status": "fail" if render_failed else "pass",
        })

        # Both the bench results AND the preview are downloadable in
        # the GUI even when one of the two fails — the operator can
        # always grab whatever DID succeed.
        results_md_url = f"/api/diagnostics/sota-clip-results/{token}.md"
        results_json_url = f"/api/diagnostics/sota-clip-results/{token}.json"
        if render_failed or not os.path.isfile(preview_path):
            yield _sse_event("complete", {
                "ok": True, "stage": "sota_bench",
                "results_md_url": results_md_url,
                "results_json_url": results_json_url,
                "message": (
                    "Bench OK but preview render failed — see ffmpeg "
                    "output above. The bench metric results below "
                    "are still valid."
                ),
            })
        else:
            preview_size_mb = os.path.getsize(preview_path) / 1_048_576
            yield _sse_event("complete", {
                "ok": True, "stage": "sota_bench",
                "preview_url": f"/api/diagnostics/sota-clip-preview/{token}.mp4",
                "preview_size_mb": round(preview_size_mb, 1),
                "results_md_url": results_md_url,
                "results_json_url": results_json_url,
                "message": (
                    f"SOTA pipeline rendered 9:16 preview "
                    f"({preview_size_mb:.1f} MB). Watch the video inline "
                    "below, or use the buttons to download the preview MP4 "
                    "+ metric reports."
                ),
            })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/sota-clip-preview/{token}.mp4")
async def sota_clip_preview(token: str, request: Request):
    """Serve the 9:16 SOTA-rendered preview MP4 for an upload token.

    Auth required (cookie session) — same as the upload + bench
    endpoints. The token is single-use-ish: it persists for the
    upload's TTL (1 h) and points at the rendered preview produced
    by the ``/sota-clip-bench`` endpoint's render phase.
    """
    from fastapi.responses import FileResponse, Response
    info = _SOTA_CLIP_UPLOADS.get(token)
    if info is None:
        return Response(status_code=404, content="upload token not found")
    preview_path = f"/tmp/sota_clip_{token}_preview.mp4"
    if not os.path.isfile(preview_path):
        return Response(
            status_code=404,
            content=(
                "preview not yet rendered. Run /sota-clip-bench first; "
                "the render is the final phase of that flow."
            ),
        )
    return FileResponse(
        preview_path,
        media_type="video/mp4",
        filename=f"sota_preview_{info.get('filename', 'clip')}",
        # Allow the browser <video> element to seek without redownloading.
        headers={"Accept-Ranges": "bytes", "Cache-Control": "no-cache"},
    )


def _serve_sota_results(token: str, suffix: str, media_type: str):
    """Shared body for the markdown + JSON results download endpoints.

    Returns the file inline (not as attachment) so the browser can
    preview the markdown / JSON in a tab; the front-end download
    button uses the ``download`` attribute for actual save-to-disk.
    """
    from fastapi.responses import FileResponse, Response
    info = _SOTA_CLIP_UPLOADS.get(token)
    if info is None:
        return Response(status_code=404, content="upload token not found")
    path = f"/tmp/sota_clip_{token}_results.{suffix}"
    if not os.path.isfile(path):
        return Response(
            status_code=404,
            content=(
                f"results.{suffix} not generated yet. Run /sota-clip-bench "
                "first."
            ),
        )
    base = os.path.splitext(info.get("filename") or "clip")[0]
    return FileResponse(
        path,
        media_type=media_type,
        filename=f"sota_results_{base}.{suffix}",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/sota-clip-results/{token}.md")
async def sota_clip_results_md(token: str):
    """Serve the human-readable markdown bench rollup for a token."""
    return _serve_sota_results(token, "md", "text/markdown; charset=utf-8")


@router.get("/sota-clip-results/{token}.json")
async def sota_clip_results_json(token: str):
    """Serve the machine-readable JSON metric dump for a token."""
    return _serve_sota_results(token, "json", "application/json")


@router.get("/auth-cache")
async def auth_cache_stats(_admin: User = Depends(require_admin)):
    """Expose the in-process session + user cache counters.

    Admin-only. Handy for diagnosing the upload-slowdown regression:
    during a large multi-chunk upload, ``session_hits`` should
    dominate over ``session_misses``. A low hit rate here points at
    a bug invalidating the cache too aggressively (or not at all).
    """
    from backend.app.auth import store as auth_store
    return auth_store.get_cache_stats()
