import os
import logging
import logging.handlers
import subprocess
import threading
import time

# ── Prevent PyTorch from eagerly initializing CUDA in the main process ──
# PyTorch's torch.cuda.is_available() triggers full CUDA initialization which
# can interfere with CTranslate2's CUDA context in the Whisper subprocess.
# CUDA_MODULE_LOADING=LAZY defers CUDA module loading until actually needed,
# preventing the main process from corrupting the CUDA driver state.
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

from backend.app.auth.deps import get_current_user, require_job_access
from backend.app.auth.models import User

from backend.services.range_stream import parse_range_header, stream_file_range
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from backend.routers import upload, jobs, clips, fonts, presets, settings as settings_router, ws
from backend.routers import agent as agent_router
from backend.routers import media as media_router
from backend.routers import chunked_upload
from backend.routers import cloud as cloud_router
from backend.routers import diagnostics as diagnostics_router
from backend.routers import render_plan as render_plan_router
from backend.routers import thumbnails as thumbnails_router
from backend.routers import share as share_router
from backend.routers import auth as auth_router
from backend.routers.api_v1 import router as api_v1_router
from backend.app.cloud.logging_filter import install_redaction_filter

# Cloud storage tokens / OAuth credentials must never leak into logs.
# The redaction filter runs before anything else so even stray
# logger.debug() calls in third-party libraries get scrubbed.
install_redaction_filter()

LOG_FILE = "/data/logs/app.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Also write all logs to a file so they can be exported from the UI.
# Clear log files on every container start so /logs only shows the current run.
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
for _old_log in [LOG_FILE, LOG_FILE + ".1", LOG_FILE + ".2", LOG_FILE + ".3"]:
    try:
        open(_old_log, "w").close()
    except OSError:
        pass
_file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8",
)
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logging.getLogger().addHandler(_file_handler)

logger = logging.getLogger("clipai")

app = FastAPI(title="ClipAI", version="1.0.0")


class CrossOriginIsolationMiddleware(BaseHTTPMiddleware):
    """Add COOP/COEP headers for SharedArrayBuffer support (WebGPU/ONNX Runtime).

    Only sent when the origin is "potentially trustworthy" (localhost or HTTPS),
    because browsers ignore these headers on plain-HTTP non-localhost origins
    and log a noisy console warning.
    """
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        host = (request.headers.get("host") or "").split(":")[0]
        is_secure = request.url.scheme == "https"
        is_localhost = host in ("localhost", "127.0.0.1", "::1")
        if is_secure or is_localhost:
            response.headers["Cross-Origin-Embedder-Policy"] = "credentialless"
            response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        return response


app.add_middleware(CrossOriginIsolationMiddleware)

# Trust proxy headers (X-Forwarded-For, X-Forwarded-Proto) so request.url.scheme
# resolves to "https" behind nginx / Caddy / Cloudflare Tunnel.  Without this,
# all emitted og:image and og:url are http:// on HTTPS sites, which Facebook
# and iMessage silently drop and Discord may downgrade.
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

# Trusted host allowlist — prevents Host-header spoofing that could make OG URLs
# point at an attacker's domain.  Read from TRUSTED_HOSTS env (comma-separated)
# with sensible defaults including common LAN patterns.
_trusted_hosts = os.environ.get("TRUSTED_HOSTS", "").strip()
if _trusted_hosts:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=[h.strip() for h in _trusted_hosts.split(",")],
    )

# OG injection middleware: serves minimal HTML with Open Graph tags to crawler
# user agents (Slackbot, Twitterbot, etc.) so link previews show thumbnails.
# Must be added AFTER CrossOriginIsolation (runs BEFORE it in the request chain)
# so crawlers get the OG stub before any other middleware processes the request.
from backend.middleware.og_injection import OGInjectionMiddleware
app.add_middleware(OGInjectionMiddleware)

# Auth middleware: resolves the session cookie before router deps so
# every authenticated request gets ``request.state.user`` populated. A
# missing / expired / fingerprint-mismatched cookie returns 401 so the
# frontend can redirect to ``/login``. Public paths (``/api/auth/*``,
# ``/api/site-config``, SPA routes) bypass the check — see the
# module docstring for the allow list.
from backend.app.auth.middleware import AuthMiddleware
app.add_middleware(AuthMiddleware)


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(request: Request, exc: RequestValidationError):
    """Log full Pydantic validation errors so 422s are diagnosable."""
    logger.error(
        "422 Validation Error on %s %s: %s",
        request.method, request.url.path, exc.errors(),
    )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.on_event("startup")
async def _startup_seed_auth():
    """Ensure the head admin account exists on every boot.

    Seeds ``Jadmin`` / ``Ilovesnoop5994!`` the first time the app runs
    against an empty auth store. Idempotent on subsequent boots — if
    the admin already exists, the seeder only normalizes their role
    and head-admin flag (never the password).
    """
    try:
        from backend.app.auth.seed import ensure_head_admin
        await ensure_head_admin()
    except Exception as e:
        logger.error("auth seeding failed: %s", e)


@app.on_event("startup")
async def _startup_preload():
    """Auto-detect GPU and preload Whisper model at startup."""
    from backend.config import settings as cfg

    nvidia_found = False
    gpu_name = ""

    # Method 1: nvidia-smi (most reliable when installed)
    # Enumerate ALL GPUs and pick the most capable (highest VRAM) —
    # this ensures a locally passed-through high-end GPU is preferred
    # over a server's weaker GPU.
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if smi.returncode == 0 and smi.stdout.strip():
            gpu_list = []
            for line in smi.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    try:
                        gpu_list.append({
                            "index": int(parts[0]),
                            "name": parts[1],
                            "vram_mb": int(float(parts[2])),
                        })
                    except (ValueError, IndexError):
                        continue
            if gpu_list:
                # Sort by VRAM descending — prefer most capable GPU
                gpu_list.sort(key=lambda g: g["vram_mb"], reverse=True)
                best = gpu_list[0]
                gpu_name = best["name"]
                nvidia_found = True
                if len(gpu_list) > 1:
                    logger.info(
                        "Multiple NVIDIA GPUs detected: %s — selecting %s (index %d, %d MB VRAM)",
                        ", ".join(f"{g['name']} [{g['index']}]" for g in gpu_list),
                        best["name"], best["index"], best["vram_mb"],
                    )
                else:
                    logger.info("NVIDIA GPU detected via nvidia-smi: %s (%d MB)", gpu_name, best["vram_mb"])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Method 2: Check /dev/nvidia* device nodes (GPU passthrough without nvidia-smi)
    if not nvidia_found:
        try:
            import glob as _glob
            nvidia_devs = _glob.glob("/dev/nvidia[0-9]*")
            if nvidia_devs:
                nvidia_found = True
                gpu_name = f"NVIDIA GPU ({len(nvidia_devs)} device{'s' if len(nvidia_devs) > 1 else ''})"
                logger.info("NVIDIA GPU detected via /dev/nvidia* devices: %s", nvidia_devs)
        except Exception:
            pass

    # Method 3: Check /proc/driver/nvidia/gpus/ (kernel module loaded)
    if not nvidia_found:
        try:
            import glob as _glob
            nv_infos = _glob.glob("/proc/driver/nvidia/gpus/*/information")
            for info_path in nv_infos:
                try:
                    content = open(info_path).read()
                    for line in content.split("\n"):
                        if line.startswith("Model:"):
                            gpu_name = line.split(":", 1)[1].strip()
                            nvidia_found = True
                            break
                except (OSError, IOError):
                    continue
            if nvidia_found:
                logger.info("NVIDIA GPU detected via /proc/driver/nvidia: %s", gpu_name)
        except Exception:
            pass

    # Method 4: Check sysfs for NVIDIA vendor ID (0x10de)
    if not nvidia_found:
        try:
            import glob as _glob
            for vendor_path in _glob.glob("/sys/class/drm/card[0-9]*/device/vendor"):
                try:
                    vendor_id = open(vendor_path).read().strip().lower()
                    if vendor_id == "0x10de":
                        nvidia_found = True
                        # Try to get PCI slot name for identification
                        uevent_path = os.path.join(os.path.dirname(vendor_path), "uevent")
                        if os.path.isfile(uevent_path):
                            for line in open(uevent_path):
                                if line.startswith("PCI_SLOT_NAME="):
                                    slot = line.strip().split("=", 1)[1]
                                    gpu_name = gpu_name or f"NVIDIA GPU ({slot})"
                        if not gpu_name:
                            gpu_name = "NVIDIA GPU (sysfs)"
                        logger.info("NVIDIA GPU detected via sysfs: %s", gpu_name)
                        break
                except (OSError, IOError):
                    continue
        except Exception:
            pass

    # Method 5: Check if CUDA libraries are loadable
    if not nvidia_found:
        try:
            import ctypes
            for lib_name in ["libcuda.so.1", "libcuda.so", "nvcuda.dll"]:
                try:
                    ctypes.cdll.LoadLibrary(lib_name)
                    nvidia_found = True
                    gpu_name = gpu_name or "NVIDIA GPU (CUDA library)"
                    logger.info("NVIDIA GPU detected via CUDA library: %s", lib_name)
                    break
                except OSError:
                    continue
        except Exception:
            pass

    if nvidia_found:
        # Only auto-enable if the user hasn't explicitly saved a preference.
        # _restore_user_settings() runs at module import time, so if the user
        # previously disabled GPU acceleration, we respect that choice.
        import os as _os
        user_settings_path = _os.path.join(
            "/data/logs" if _os.path.isdir("/data/logs") else ".",
            "user_settings.json",
        )
        user_has_gpu_preference = False
        try:
            if _os.path.exists(user_settings_path):
                import json as _json
                with open(user_settings_path) as _f:
                    _saved = _json.load(_f)
                if "GPU_ACCELERATION_ENABLED" in _saved:
                    user_has_gpu_preference = True
                    logger.info(
                        "User has saved GPU preference: GPU_ACCELERATION_ENABLED=%s",
                        _saved["GPU_ACCELERATION_ENABLED"],
                    )
        except Exception:
            pass

        if not user_has_gpu_preference:
            cfg.GPU_ACCELERATION_ENABLED = True
            cfg.GPU_VENDOR_OVERRIDE = "nvidia"
            logger.info("GPU acceleration auto-enabled (first run): %s", gpu_name)
        elif cfg.GPU_ACCELERATION_ENABLED:
            # User previously enabled it — ensure vendor is set
            if not cfg.GPU_VENDOR_OVERRIDE:
                cfg.GPU_VENDOR_OVERRIDE = "nvidia"
            logger.info("GPU acceleration enabled (user preference): %s", gpu_name)
        else:
            logger.info(
                "NVIDIA GPU detected (%s) but GPU acceleration disabled by user preference",
                gpu_name,
            )

        try:
            from backend.services.clip_exporter import _gpu_info_cache_clear
            _gpu_info_cache_clear()
        except Exception:
            pass
    else:
        logger.info("No NVIDIA GPU detected — using CPU for encoding and Whisper")
        # Log diagnostic details to help users troubleshoot GPU passthrough
        diag_parts = []
        try:
            smi_check = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=5,
            )
            diag_parts.append(f"nvidia-smi exit={smi_check.returncode}")
            if smi_check.stderr:
                diag_parts.append(f"nvidia-smi stderr: {smi_check.stderr.strip()[:200]}")
        except FileNotFoundError:
            diag_parts.append("nvidia-smi not found (NVIDIA Container Toolkit may not be injecting GPU)")
        except Exception as _e:
            diag_parts.append(f"nvidia-smi error: {_e}")
        import glob as _glob
        diag_parts.append(f"/dev/nvidia*: {_glob.glob('/dev/nvidia*') or 'none'}")
        diag_parts.append(f"NVIDIA_VISIBLE_DEVICES={os.environ.get('NVIDIA_VISIBLE_DEVICES', 'unset')}")
        logger.info(
            "GPU troubleshooting: %s | "
            "Ensure: (1) Docker Desktop uses WSL2 backend, "
            "(2) run 'docker compose up' (v2, not docker-compose), "
            "(3) NVIDIA driver R525+ on Windows host",
            " | ".join(diag_parts),
        )

    # Only preload Whisper in-process if we won't use subprocess transcription.
    # When Ollama is the AI provider, the pipeline runs Whisper in a subprocess
    # to release CTranslate2's CUDA context (~1.6GB) after transcription.
    # Preloading here would permanently hold that VRAM, defeating subprocess
    # isolation and leaving insufficient memory for medium/large models.
    from backend.config import settings as cfg
    _use_subprocess_whisper = (
        cfg.GPU_ACCELERATION_ENABLED
        and "ollama" in getattr(cfg, "active_provider_chain", [])
    )
    if _use_subprocess_whisper:
        logger.info(
            "Skipping Whisper preload — subprocess mode active (Ollama primary). "
            "GPU VRAM stays free for subprocess to load %s model.",
            cfg.WHISPER_MODEL,
        )
        # Populate whisper_device_info for status display without loading the model
        from backend.services.transcription import _detect_cuda_available, whisper_device_info, is_whisper_model_cached, ensure_whisper_model_downloaded
        try:
            cuda_ok, cuda_count, gpu_name, best_idx = _detect_cuda_available()
            if cuda_ok:
                whisper_device_info.update({
                    "device": "cuda",
                    "compute_type": "float16",
                    "gpu_name": gpu_name,
                    "device_index": best_idx,
                })
                logger.info("Whisper will use GPU in subprocess: %s (float16)", gpu_name)
        except Exception:
            pass
        # Pre-download the configured Whisper model in background if not cached.
        # This ensures the model is ready when the user starts their first analysis.
        if not is_whisper_model_cached(cfg.WHISPER_MODEL):
            logger.info("Whisper model '%s' not cached — starting background download", cfg.WHISPER_MODEL)
            def _bg_download():
                ensure_whisper_model_downloaded(cfg.WHISPER_MODEL, timeout=600)
            threading.Thread(target=_bg_download, daemon=True, name="whisper-startup-download").start()
    else:
        from backend.services.transcription import preload_model
        threading.Thread(target=preload_model, daemon=True).start()

    # Pull Ollama models in the background only if Ollama is in the fallback chain.
    # When cloud providers are primary, only pull the configured Ollama models
    # (not the extras like llava, llama3.2) to minimize unnecessary downloads.
    if "ollama" in cfg.active_provider_chain:
        _chain = [p.strip() for p in cfg.AI_FALLBACK_CHAIN.split(",") if p.strip()]
        _ollama_is_primary = _chain and _chain[0] == "ollama"

        def _pull_ollama_models():
            import httpx
            host = cfg.OLLAMA_HOST
            if _ollama_is_primary:
                # Ollama is primary — pull all configured models + extras
                models = [cfg.OLLAMA_VISION_MODEL, cfg.OLLAMA_TEXT_MODEL]
                for extra in ("llava", "llama3.2:3b", cfg.OLLAMA_TRANSLATION_MODEL):
                    if extra and extra not in models:
                        models.append(extra)
            else:
                # Cloud provider is primary — only pull the configured fallback models
                # Don't pull extras to save bandwidth and avoid unnecessary GPU loading
                models = [cfg.OLLAMA_VISION_MODEL, cfg.OLLAMA_TEXT_MODEL]
                logger.info("Cloud provider is primary — pulling only Ollama fallback models: %s", models)
            for model in models:
                try:
                    logger.info("Background pull: requesting %s from Ollama...", model)
                    resp = httpx.post(
                        f"{host}/api/pull",
                        json={"name": model},
                        timeout=httpx.Timeout(connect=10, read=1800, write=10, pool=10),
                    )
                    if resp.status_code == 200:
                        logger.info("Background pull: %s ready", model)
                    else:
                        logger.warning("Background pull: %s returned %d", model, resp.status_code)
                except Exception as exc:
                    logger.warning("Background pull: %s failed (%s)", model, exc)
        threading.Thread(target=_pull_ollama_models, daemon=True, name="ollama-bg-pull").start()
    else:
        logger.info("Ollama not in fallback chain — skipping background model pull")

    # ── Persist combined settings state ──
    # At this point, settings contain: (1) code defaults, (2) .env overrides,
    # (3) restored user_settings.json values, (4) GPU auto-detection results.
    # Persist everything so that ALL of these survive the next container rebuild,
    # including API keys that were only set via .env and never saved through the UI.
    try:
        from backend.routers.settings import _persist_user_settings
        _persist_user_settings()
        logger.info("Settings persisted after startup initialization")
    except Exception as e:
        logger.warning("Failed to persist settings after startup: %s", e)

@app.on_event("startup")
async def warmup_ollama():
    """Detect Ollama model capabilities (without loading into VRAM).

    We skip the VRAM warmup at startup so that Whisper can use the GPU
    for transcription first.  Ollama models are loaded lazily when the
    pipeline reaches the scene-analysis phase — after Whisper is done.
    """
    from backend.config import settings as cfg
    if "ollama" in cfg.active_provider_chain:
        try:
            from backend.services.providers.ollama_provider import OllamaProvider
            provider = OllamaProvider()
            await provider._detect_capabilities()
            # NOTE: warmup() intentionally skipped — see docstring above.
            await provider.close()
        except Exception as e:
            logger.warning("Ollama capability detection at startup failed (non-fatal): %s", e)

@app.on_event("startup")
async def fetch_openrouter_model_caps():
    """Fetch OpenRouter model capabilities at startup so the pipeline knows
    each model's context_length, max_completion_tokens, and vision support
    BEFORE the first analysis runs.  Without this, the provider falls back
    to hardcoded pattern-based estimates which can be wrong (e.g. reka-edge
    getting 8K output tokens when it only supports 4K context total).

    The cache is stored on the Docker volume (/data/logs/model_cache.json)
    and refreshed if older than 24h.
    """
    from backend.config import settings as cfg
    if not cfg.OPENROUTER_API_KEY or cfg.OPENROUTER_API_KEY in {"", "sk-or-..."}:
        logger.info("OpenRouter API key not set — skipping model capability fetch")
        return
    try:
        from backend.routers.settings import _fetch_openrouter_models, MODEL_CACHE_PATH, MODEL_CACHE_TTL
        import os, time as _time
        # Check if cache is fresh enough (avoid hitting API on every restart)
        if os.path.exists(MODEL_CACHE_PATH):
            try:
                import json as _json
                with open(MODEL_CACHE_PATH, "r") as f:
                    cache = _json.load(f)
                age = _time.time() - cache.get("timestamp", 0)
                n_models = len(cache.get("raw_models", []))
                if age < MODEL_CACHE_TTL and n_models > 0:
                    logger.info(
                        "OpenRouter model cache is fresh (%d models, %.0fh old) — skipping fetch",
                        n_models, age / 3600,
                    )
                    return
            except Exception:
                pass
        models = await _fetch_openrouter_models()
        if models:
            logger.info("Fetched %d OpenRouter models at startup — capabilities cached", len(models))
        else:
            logger.warning("OpenRouter model fetch returned no models (API may be unreachable)")
    except Exception as e:
        logger.warning("OpenRouter model capability fetch failed at startup (non-fatal): %s", e)

@app.on_event("startup")
async def recover_uploads():
    """Recover in-progress chunked upload sessions from disk after restart."""
    from backend.routers.chunked_upload import restore_sessions
    await restore_sessions()


@app.on_event("startup")
async def start_cleanup_task():
    """Periodically expire stale upload sessions (every 10 minutes)."""
    import asyncio

    async def cleanup_loop():
        while True:
            await asyncio.sleep(600)  # 10 minutes
            try:
                from backend.routers.chunked_upload import expire_stale_uploads
                expire_stale_uploads()
            except Exception as exc:
                logger.warning("Stale upload cleanup error: %s", exc)

    asyncio.create_task(cleanup_loop())


# Register API routers
# Auth must come first so its public endpoints (login, bootstrap) win
# over any catch-all routes registered later.
app.include_router(auth_router.router)
app.include_router(upload.router)
app.include_router(chunked_upload.router)
app.include_router(cloud_router.router)
app.include_router(jobs.router)
app.include_router(clips.router)
app.include_router(fonts.router)
app.include_router(presets.router)
app.include_router(settings_router.router)
app.include_router(ws.router)
app.include_router(agent_router.router)
app.include_router(media_router.router)
app.include_router(diagnostics_router.router)
app.include_router(render_plan_router.router)
app.include_router(thumbnails_router.router)
app.include_router(share_router.router)
app.include_router(api_v1_router)

# Mount MCP server at /mcp (if mcp package is available)
try:
    from mcp.server.fastmcp import FastMCP

    clipai_mcp = FastMCP(
        name="ClipAI",
        stateless_http=True,
    )

    # Import and register MCP tools
    from backend.mcp_tools import register_tools
    register_tools(clipai_mcp)

    app.mount("/mcp", clipai_mcp.streamable_http_app())
    logger.info("MCP server mounted at /mcp")
except ImportError:
    logger.warning("mcp package not installed — MCP endpoint disabled. Install with: pip install 'mcp>=1.8.0'")
except Exception as exc:
    logger.warning("Failed to mount MCP server: %s", exc)

# Ensure data dirs exist
for d in ["/data/uploads", "/data/outputs", "/data/logs", "/data/fonts"]:
    os.makedirs(d, exist_ok=True)

# Ensure fontconfig knows about /data/fonts so FFmpeg/libass can discover
# custom uploaded fonts by family name.  The Dockerfile also sets this up,
# but this covers non-Docker and fresh environments.
_FC_CONF = "/etc/fonts/conf.d/99-custom-fonts.conf"
if not os.path.isfile(_FC_CONF):
    try:
        os.makedirs(os.path.dirname(_FC_CONF), exist_ok=True)
        with open(_FC_CONF, "w") as f:
            f.write('<?xml version="1.0"?>\n'
                    '<!DOCTYPE fontconfig SYSTEM "fonts.dtd">\n'
                    '<fontconfig><dir>/data/fonts</dir></fontconfig>\n')
        subprocess.run(["fc-cache", "-f"], capture_output=True, timeout=10)
        logger.info("Created fontconfig config for /data/fonts")
    except (PermissionError, OSError, FileNotFoundError) as exc:
        logger.warning("Could not create fontconfig config (%s) — custom fonts "
                       "will still work via fontsdir in FFmpeg filter", exc)

# Symlink system-installed fonts into /data/fonts/ so they are always available
# to FFmpeg's ass filter via the fontsdir parameter.  The ass filter's fontsdir
# is the *only* guaranteed font discovery path — fontconfig may silently fail
# in some libass builds, causing all fonts to fall back to DejaVu Sans.
from backend.routers.fonts import SYSTEM_FONT_PATHS
_symlinked = 0
for _fname, _syspath in SYSTEM_FONT_PATHS.items():
    _link = os.path.join("/data/fonts", _fname)
    if os.path.isfile(_syspath) and not os.path.exists(_link):
        try:
            os.symlink(_syspath, _link)
            _symlinked += 1
        except OSError:
            pass
if _symlinked:
    logger.info("Symlinked %d system fonts into /data/fonts for FFmpeg fontsdir", _symlinked)


# Short-lived cache of successful (user_id, job_id) access checks so
# HTML5 <video> range requests during scrubbing don't re-load job.json
# from disk on every seek. The real ownership check runs at least once
# per minute; on cache miss the full require_job_access validates.
_file_access_cache: dict[tuple[str, str], float] = {}
_FILE_ACCESS_TTL = 60.0


async def _fast_require_job_access(job_id: str, user: User) -> None:
    """Cached wrapper around require_job_access for hot file-serving path."""
    key = (user.id, job_id)
    now = time.monotonic()
    exp = _file_access_cache.get(key)
    if exp is not None and exp > now:
        return
    await require_job_access(job_id, user)
    _file_access_cache[key] = now + _FILE_ACCESS_TTL
    # Opportunistic eviction — keep the dict bounded without needing a
    # dedicated reaper thread. 4096 entries is plenty for concurrent
    # users and evicts FIFO-ish (dict insertion order).
    if len(_file_access_cache) > 4096:
        for _k in list(_file_access_cache.keys())[:256]:
            _file_access_cache.pop(_k, None)


@app.get("/api/files/{job_id}/{path:path}")
async def serve_file(
    job_id: str,
    path: str,
    request: Request,
    user: User = Depends(get_current_user),
):
    """Serve video files and exported clips with range request support.

    Every request is authenticated and checked against job ownership so
    one user (admin or otherwise) can't read another's source video or
    exported clips by guessing job IDs. The per-user media library
    directories (``_library_{user_id}``) match by ownership name too
    when ``job_id`` starts with the library sentinel.
    """
    # Per-user media library: ``_library_{user_id}`` directories are
    # owned by the user in their name; any other user is rejected.
    if job_id.startswith("_library_"):
        suffix = job_id[len("_library_"):]
        if suffix != user.id:
            return Response(status_code=404, content="File not found")
    elif job_id != "_library":
        # For real jobs, verify ownership before serving any bytes.
        # Cached so range-heavy video playback doesn't hammer job.json.
        await _fast_require_job_access(job_id, user)
    else:
        # Legacy unscoped ``_library`` — not owned by anyone; reject.
        return Response(status_code=404, content="File not found")

    # Check uploads first, then outputs
    file_path = None
    candidates = [
        os.path.join("/data/uploads", job_id, path),
        os.path.join("/data/outputs", job_id, path),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            file_path = candidate
            break

    if not file_path:
        return Response(status_code=404, content="File not found")

    # Transparently swap in a browser-friendly derivative for the
    # source video. Applies only to the uploaded source file (path
    # starts with ``video.``) — exported clips under ``/clips/`` are
    # already MP4/H.264/AAC from the exporter and must not be
    # touched, or we'd re-encode every download. Without this swap,
    # ``.mkv`` sources with AC3 / DTS / FLAC audio play silently in
    # every major browser (HTML5 ``<video>`` doesn't decode those
    # codecs). The helper is idempotent + cached; see
    # ``backend/services/browser_preview.py``.
    _basename = os.path.basename(file_path).lower()
    _is_source_video = _basename.startswith("video.") and "/clips/" not in path
    if _is_source_video:
        try:
            from backend.services.browser_preview import ensure_browser_preview_async
            file_path = await ensure_browser_preview_async(file_path)
        except Exception as e:  # pragma: no cover — fallback path
            logger.warning("serve_file: browser_preview generation failed: %s", e)

    file_size = os.path.getsize(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    _CONTENT_TYPES = {
        ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".gif": "image/gif", ".webp": "image/webp",
        ".mp3": "audio/mpeg", ".wav": "audio/wav", ".aac": "audio/aac",
        ".ogg": "audio/ogg", ".flac": "audio/flac",
    }
    content_type = _CONTENT_TYPES.get(ext, "application/octet-stream")

    # Handle Range requests for video seeking.
    #
    # Uses a streaming response (``stream_file_range``) rather than
    # ``f.read(content_length)`` into memory. HTML5 ``<video>`` seeks
    # can pull multi-megabyte ranges, and with many concurrent
    # viewers the old path pinned entire ranges in RAM on the event
    # loop thread. Streaming keeps memory flat and lets aborted
    # seeks free resources immediately — the browser often starts a
    # new Range the moment the user moves the scrubber, and the old
    # code still finished reading the stale range into RAM first.
    range_header = request.headers.get("range")
    # Cache-Control differs by what we're serving:
    #   * ``/clips/`` — exporter overwrites same filename on re-export,
    #     must not be cached.
    #   * Source video — stable bytes for the life of the preview;
    #     cache an hour so repeat seeks over the same region hit the
    #     browser cache instead of the server. Meaningful scrubbing
    #     speedup on replays of previously-visited timestamps.
    if "/clips/" in path:
        _extra_headers = {
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        }
    elif _is_source_video:
        _extra_headers = {"Cache-Control": "public, max-age=3600"}
    else:
        _extra_headers = None
    if range_header:
        try:
            parsed = parse_range_header(range_header, file_size)
        except ValueError:
            return Response(status_code=416, content="invalid range")
        if parsed is not None:
            start, end = parsed
            return stream_file_range(
                file_path, file_size, start, end, content_type,
                extra_headers=_extra_headers,
            )

    # For clip files, force browser download instead of inline playback.
    # Cache-Control: no-store prevents the browser from serving a stale
    # cached version when the user re-exports the same clip (same filename).
    if "/clips/" in path:
        response = FileResponse(
            file_path,
            media_type=content_type,
            filename=os.path.basename(file_path),
        )
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return response

    response = FileResponse(file_path, media_type=content_type)
    if _is_source_video:
        response.headers["Cache-Control"] = "public, max-age=3600"
    return response


# ── Cloud storage setup docs ─────────────────────────────────────────
# The CloudStorageSection in Settings links to
# /docs/cloud-storage/SETUP.md (and friends). The SPA has no route for
# that path, so without this endpoint the user lands on a blank React
# page. We render the repo's markdown files into a minimal HTML
# wrapper so the docs actually load. Served BEFORE the SPA catch-all
# so the specific path wins over the generic fallback.
_DOCS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "docs",
    "cloud-storage",
)
_ALLOWED_CLOUD_DOCS = {"SETUP.md", "RECON.md", "UNRAID.md"}


def _render_markdown_to_html(md_text: str, title: str) -> str:
    """Tiny pure-Python markdown → HTML renderer.

    Handles the subset of features used by the cloud-storage docs:
    fenced code blocks, headings, bold, inline code, links, horizontal
    rules, unordered + ordered lists, paragraphs.

    We roll our own so we don't add a new runtime dependency just for
    three static docs. It is NOT a general-purpose markdown parser —
    don't use it for user-supplied input.
    """
    import html as _html
    import re

    def _escape(s: str) -> str:
        return _html.escape(s, quote=False)

    def _inline(s: str) -> str:
        s = _escape(s)
        # Inline code first (so ** inside code isn't re-parsed)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        # Bold
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        # Links [text](url)
        s = re.sub(
            r"\[([^\]]+)\]\(([^)]+)\)",
            lambda m: f'<a href="{m.group(2)}" target="_blank" rel="noopener noreferrer">{m.group(1)}</a>',
            s,
        )
        return s

    lines = md_text.splitlines()
    out: list[str] = []
    i = 0
    in_code = False
    code_lines: list[str] = []
    list_stack: list[str] = []  # 'ul' or 'ol'

    def _close_lists():
        while list_stack:
            out.append(f"</{list_stack.pop()}>")

    while i < len(lines):
        line = lines[i]
        # Fenced code block
        if line.lstrip().startswith("```"):
            if in_code:
                out.append(
                    "<pre><code>"
                    + _escape("\n".join(code_lines))
                    + "</code></pre>"
                )
                code_lines = []
                in_code = False
            else:
                _close_lists()
                in_code = True
            i += 1
            continue
        if in_code:
            code_lines.append(line)
            i += 1
            continue

        stripped = line.strip()
        if not stripped:
            _close_lists()
            i += 1
            continue

        # Horizontal rule
        if stripped == "---":
            _close_lists()
            out.append("<hr />")
            i += 1
            continue

        # Headings
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            _close_lists()
            level = len(heading.group(1))
            out.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            i += 1
            continue

        # Unordered list
        ul = re.match(r"^[-*]\s+(.*)$", stripped)
        if ul:
            if not list_stack or list_stack[-1] != "ul":
                _close_lists()
                out.append("<ul>")
                list_stack.append("ul")
            out.append(f"<li>{_inline(ul.group(1))}</li>")
            i += 1
            continue

        # Ordered list
        ol = re.match(r"^\d+\.\s+(.*)$", stripped)
        if ol:
            if not list_stack or list_stack[-1] != "ol":
                _close_lists()
                out.append("<ol>")
                list_stack.append("ol")
            out.append(f"<li>{_inline(ol.group(1))}</li>")
            i += 1
            continue

        # Paragraph — coalesce consecutive non-empty, non-special lines
        _close_lists()
        para_lines = [line]
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            if not nxt.strip():
                break
            if nxt.lstrip().startswith(("#", "```", "- ", "* ")):
                break
            if re.match(r"^\d+\.\s", nxt.lstrip()):
                break
            para_lines.append(nxt)
            j += 1
        para_html = " ".join(_inline(pl.strip()) for pl in para_lines)
        out.append(f"<p>{para_html}</p>")
        i = j

    _close_lists()
    if in_code and code_lines:
        out.append("<pre><code>" + _escape("\n".join(code_lines)) + "</code></pre>")

    body = "\n".join(out)
    # Minimal dark-theme CSS so the doc matches the rest of ClipAI.
    css = """
    body {
        margin: 0;
        background: #0b0d10;
        color: #e8ecf0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        line-height: 1.55;
    }
    .wrap {
        max-width: 760px;
        margin: 0 auto;
        padding: 32px 24px 80px;
    }
    h1, h2, h3 { color: #fff; margin-top: 1.8em; }
    h1 { font-size: 28px; border-bottom: 1px solid #222; padding-bottom: 10px; }
    h2 { font-size: 22px; border-bottom: 1px solid #1a1a1a; padding-bottom: 6px; }
    h3 { font-size: 17px; }
    p, li { color: #d0d6dc; }
    a { color: #4ec8f4; text-decoration: none; }
    a:hover { text-decoration: underline; }
    code {
        background: #1a1e24;
        border: 1px solid #2a2f36;
        color: #f6d68d;
        padding: 1px 6px;
        border-radius: 4px;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 13px;
    }
    pre {
        background: #0e1116;
        border: 1px solid #1e232a;
        border-radius: 6px;
        padding: 12px 14px;
        overflow-x: auto;
    }
    pre code {
        background: transparent;
        border: none;
        padding: 0;
        color: #cfd6dd;
    }
    hr { border: none; border-top: 1px solid #1e232a; margin: 28px 0; }
    ul, ol { padding-left: 22px; }
    .backlink {
        display: inline-block;
        margin-bottom: 18px;
        font-size: 13px;
        color: #8892a0;
    }
    """
    safe_title = _html.escape(title, quote=True)
    return (
        "<!doctype html><html><head>"
        '<meta charset="utf-8" />'
        '<meta name="viewport" content="width=device-width, initial-scale=1" />'
        f"<title>{safe_title} — ClipAI</title>"
        f"<style>{css}</style>"
        "</head><body><div class=\"wrap\">"
        '<a class="backlink" href="/settings">&larr; Back to Settings</a>'
        f"{body}"
        "</div></body></html>"
    )


@app.get("/docs/cloud-storage/{doc_name}")
async def serve_cloud_storage_doc(doc_name: str):
    """Render one of the cloud-storage setup docs as HTML.

    Only a hard-coded allowlist of .md filenames is accepted so this
    endpoint can never be abused to read arbitrary files from the
    container.
    """
    if doc_name not in _ALLOWED_CLOUD_DOCS:
        return Response(
            content=f"Unknown cloud storage doc: {doc_name}",
            status_code=404,
            media_type="text/plain",
        )
    doc_path = os.path.join(_DOCS_ROOT, doc_name)
    if not os.path.isfile(doc_path):
        return Response(
            content=(
                f"Doc missing: {doc_name}. This usually means the "
                "docs/ directory was not included in the Docker image. "
                "Rebuild with: docker compose build --no-cache clipai-app"
            ),
            status_code=404,
            media_type="text/plain",
        )
    try:
        with open(doc_path, "r", encoding="utf-8") as f:
            md_text = f.read()
    except OSError as exc:
        return Response(
            content=f"Failed to read {doc_name}: {exc}",
            status_code=500,
            media_type="text/plain",
        )
    title = doc_name.replace(".md", "").replace("-", " ").title()
    html = _render_markdown_to_html(md_text, title)
    return Response(content=html, media_type="text/html")


# Serve frontend static files
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
_assets_dir = os.path.join(static_dir, "assets")
_index_path = os.path.join(static_dir, "index.html")

if os.path.isdir(static_dir) and os.path.isdir(_assets_dir) and os.path.isfile(_index_path):
    app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")

    @app.get("/{path:path}")
    async def serve_spa(path: str):
        # Serve index.html for all non-API, non-asset routes (SPA routing)
        file_path = os.path.join(static_dir, path)
        if path and os.path.isfile(file_path):
            return FileResponse(file_path)

        # Inject site customisation (title, favicon) into index.html so
        # user settings persist visually across restarts without a flash.
        index_path = os.path.join(static_dir, "index.html")
        if not os.path.isfile(index_path):
            return Response(
                content='{"error": "Frontend not built. Run: docker compose build clipai-app"}',
                media_type="application/json",
                status_code=503,
            )
        try:
            from backend.routers.settings import _load_site_config
            cfg = _load_site_config()
            if cfg.get("title") or cfg.get("favicon"):
                with open(index_path, "r") as f:
                    html = f.read()
                if cfg.get("title"):
                    html = html.replace(
                        "<title>ClipAI \u2014 Video Intelligence</title>",
                        f"<title>{cfg['title']}</title>",
                    )
                if cfg.get("favicon"):
                    import re
                    html = re.sub(
                        r'<link rel="icon"[^>]*/>',
                        f'<link rel="icon" href="/api/site-uploads/{cfg["favicon"]}" />',
                        html,
                    )
                return Response(content=html, media_type="text/html")
        except Exception:
            pass
        return FileResponse(index_path)
else:
    if os.path.isdir(static_dir):
        logger.warning(
            "Frontend not built — %s exists but index.html or assets/ missing. "
            "Run 'cd frontend && npm run build' or rebuild the Docker image.",
            static_dir,
        )

    @app.get("/")
    async def root():
        return {"message": "ClipAI API running. Frontend not built — rebuild the Docker image with: docker compose build clipai-app"}

    @app.get("/{path:path}")
    async def spa_fallback(path: str):
        return Response(
            content='{"error": "Frontend not built. Rebuild with: docker compose build --no-cache clipai-app"}',
            media_type="application/json",
            status_code=503,
        )
