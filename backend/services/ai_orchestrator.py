import asyncio
import logging
import re
import time
from typing import Optional

from backend.config import settings
from backend.models import (
    FrameData, SceneDescription, TranscriptSegment, VideoSummary, ClipCandidate, ClipSEO,
)
from backend.services.providers.base import (
    AIProvider, ProviderError, ProviderRateLimitError, AllProvidersFailedError,
)
from backend.services.providers.openrouter_provider import OpenRouterProvider
from backend.services.providers.anthropic_provider import AnthropicProvider
from backend.services.providers.gemini_provider import GeminiProvider
from backend.services.providers.groq_provider import GroqProvider
from backend.services.providers.ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)

# WebSocket broadcast callback type
WsBroadcastCallback = Optional[object]  # Will be a callable


class _CircuitBreaker:
    """Marks a provider degraded for 15 min after 3 failures in 10 min."""

    def __init__(self):
        self._failures: dict[str, list[float]] = {}
        self._degraded_until: dict[str, float] = {}

    def is_degraded(self, name: str) -> bool:
        if name in self._degraded_until:
            if time.monotonic() < self._degraded_until[name]:
                return True
            del self._degraded_until[name]
        return False

    def record_failure(self, name: str):
        now = time.monotonic()
        if name not in self._failures:
            self._failures[name] = []
        self._failures[name] = [t for t in self._failures[name] if now - t < 600]
        self._failures[name].append(now)
        failure_count = len(self._failures[name])
        logger.info("Circuit breaker: %s failure %d/3 in 10-min window", name, failure_count)
        if failure_count >= 3:
            self._degraded_until[name] = now + 900  # 15 min
            logger.warning("Circuit breaker: provider %s marked DEGRADED for 15 minutes", name)

    def record_success(self, name: str):
        was_degraded = name in self._degraded_until
        self._failures.pop(name, None)
        self._degraded_until.pop(name, None)
        if was_degraded:
            logger.info("Circuit breaker: provider %s RECOVERED (success after degraded)", name)

    def clear_degraded(self, name: str):
        """Immediately remove degraded status for a provider."""
        was_degraded = name in self._degraded_until
        self._degraded_until.pop(name, None)
        if was_degraded:
            logger.info("Circuit breaker: %s manually un-degraded before critical operation", name)

    def force_reset_all(self):
        """Reset ALL provider states. Used before critical pipeline stages."""
        had_degraded = list(self._degraded_until.keys())
        self._failures.clear()
        self._degraded_until.clear()
        if had_degraded:
            logger.info("Circuit breaker: RESET all states (was degraded: %s)", had_degraded)


def _build_provider(name: str) -> Optional[AIProvider]:
    try:
        if name == "openrouter" and settings.OPENROUTER_API_KEY:
            return OpenRouterProvider()
        elif name == "anthropic" and settings.ANTHROPIC_API_KEY:
            return AnthropicProvider()
        elif name == "gemini" and settings.GEMINI_API_KEY:
            return GeminiProvider()
        elif name == "groq" and settings.GROQ_API_KEY:
            return GroqProvider()
        elif name == "ollama":
            return OllamaProvider()
    except Exception as e:
        logger.warning(f"Failed to initialize provider {name}: {e}")
    return None


_OPENROUTER_MODEL_CACHE: dict[str, bool] = {}


async def _openrouter_model_exists(model_id: str, timeout: float = 8.0) -> bool | None:
    """Return True / False if ``model_id`` is in OpenRouter's catalog,
    or ``None`` when the catalog couldn't be fetched (network error).

    Cached per-process so we don't hit /models on every job start.
    """
    if not model_id:
        return None
    cached = _OPENROUTER_MODEL_CACHE.get(model_id)
    if cached is not None:
        return cached
    try:
        import httpx
        headers = {}
        key = getattr(settings, "OPENROUTER_API_KEY", "") or ""
        if key:
            headers["Authorization"] = f"Bearer {key}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get("https://openrouter.ai/api/v1/models", headers=headers)
            if r.status_code != 200:
                return None
            data = r.json() or {}
            ids = {m.get("id") for m in (data.get("data") or [])}
            for mid in ids:
                if mid:
                    _OPENROUTER_MODEL_CACHE[mid] = True
            exists = model_id in ids
            _OPENROUTER_MODEL_CACHE[model_id] = exists
            return exists
    except Exception as e:
        logger.debug("OpenRouter model-exists probe failed: %s", e)
        return None


def _probe_ollama_reachable(timeout: float = 1.5) -> bool:
    """Cheap synchronous TCP ping of the Ollama daemon.

    Returns True iff a TCP connection to ``settings.OLLAMA_HOST`` opens
    within ``timeout`` seconds. Used by the orchestrator to skip
    Ollama on jobs where the daemon is not running, instead of
    waiting for the first HTTP call to time out (which can take
    30+ seconds and cascade through every batch).
    """
    import socket
    from urllib.parse import urlparse
    try:
        host_url = getattr(settings, "OLLAMA_HOST", "") or ""
        parsed = urlparse(host_url if "://" in host_url else f"http://{host_url}")
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 11434
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ValueError):
        return False


class AIOrchestrator:
    """
    Tries providers in fallback chain order.
    Circuit breaker: marks provider degraded for 15 min after 3 failures in 10 min.
    """

    # Model downgrade fallback for consecutive Ollama failures
    _SMALLER_MODELS = ["qwen2.5:1.5b-instruct", "qwen2.5:0.5b-instruct", "tinyllama"]
    _FAILURE_THRESHOLD_FOR_DOWNGRADE = 3

    def __init__(self, ws_broadcast=None, custom_prompts=None, cancel_check=None):
        self._circuit_breaker = _CircuitBreaker()
        self._ws_broadcast = ws_broadcast
        self._custom_prompts = custom_prompts  # PromptSet or None
        self._cancel_check = cancel_check  # callable that raises on cancel
        self._providers: dict[str, AIProvider] = {}
        self._consecutive_ollama_failures: int = 0
        self._current_model_override: str | None = None
        # Provider names that we know are unreachable for the lifetime
        # of this orchestrator (e.g. Ollama daemon not running, but
        # listed in AI_FALLBACK_CHAIN). Excluded from ``_get_active_chain``
        # so we don't burn 30s on every job trying to hit a dead daemon
        # before falling back to OpenRouter.
        self._unreachable: set[str] = set()
        for name in settings.active_provider_chain:
            p = _build_provider(name)
            if p:
                self._providers[name] = p
        # Best-effort sync reachability probe for Ollama.
        # We do NOT probe cloud providers here — their auth check costs
        # money and the orchestrator does proper fallback on first call.
        if "ollama" in self._providers:
            try:
                if not _probe_ollama_reachable():
                    self._unreachable.add("ollama")
                    logger.warning(
                        "Ollama is in AI_FALLBACK_CHAIN but %s is not "
                        "reachable — skipping for this job.",
                        getattr(settings, "OLLAMA_HOST", "(host unset)"),
                    )
            except Exception as e:
                logger.debug("Ollama reachability probe failed: %s", e)

    def _wire_ws_to_providers(self, job_id: str):
        """Pass WebSocket broadcast to providers that support model-level notifications."""
        if not self._ws_broadcast:
            return
        for provider in self._providers.values():
            if hasattr(provider, 'set_ws_broadcast'):
                provider.set_ws_broadcast(self._ws_broadcast, job_id)

    def _get_model_info(self, provider) -> str:
        """Get human-readable model info string for a provider."""
        pname = provider.provider_name
        parts = []
        if hasattr(provider, '_vision_model'):
            parts.append(f"vision={provider._vision_model}")
        if hasattr(provider, '_text_model'):
            parts.append(f"text={provider._text_model}")
        if parts:
            return f"{pname} ({', '.join(parts)})"
        return pname

    def _get_task_model(self, provider, task: str) -> str:
        """Get the specific model name used for a task (vision/text/summary/clips).

        Returns a string like 'reka/reka-edge via openrouter' for storage
        in provider_used so the frontend can show exactly which model ran.
        """
        pname = provider.provider_name
        if pname == "openrouter":
            if task in ("scenes", "vision", "scene_analysis"):
                model = getattr(provider, '_vision_model', None)
            elif task in ("summary",):
                model = getattr(provider, '_summary_model', None) or getattr(provider, '_text_model', None)
            else:  # clips, seo, text
                model = getattr(provider, '_text_model', None)
            if model:
                return f"{model} via openrouter"
        elif pname == "ollama":
            if task in ("scenes", "vision", "scene_analysis"):
                model = getattr(provider, '_vision_model', None)
            else:
                model = getattr(provider, '_text_model', None)
            if model:
                return f"{model} via ollama"
        return pname

    # Rough cost per 1K tokens by provider (input+output blended average)
    _COST_PER_1K_TOKENS = {
        "openrouter": 0.0002,   # varies by model; free tier = 0
        "anthropic": 0.006,     # Claude Sonnet ~$3/$15 per M tokens blended
        "gemini": 0.0003,       # Gemini Flash is very cheap
        "groq": 0.0001,         # Groq is very cheap
        "ollama": 0.0,          # local, no cost
    }

    async def validate_models(self, job_id: str) -> list[str]:
        """Pre-flight check: verify that EVERY constructed provider is
        reachable, in parallel.

        Previously only the first provider in the chain was validated,
        which meant a degraded primary (e.g. Ollama down) hid the fact
        that the user's OpenRouter key was actually fine. This made
        it look like the whole pipeline was broken when only the first
        link was. We now run a fan-out check so the user gets one
        line of feedback per provider in the chain.

        Returns a list of warning messages for the frontend log. Does
        NOT fail the pipeline — warnings are informational.
        """
        self._wire_ws_to_providers(job_id)
        warnings: list[str] = []
        if not self._providers:
            warnings.append("No AI providers configured — check Settings")
            return warnings

        async def _validate_one(pname: str, provider) -> tuple[str, str | None]:
            """Returns (status_message, warning_or_None) for one provider."""
            model_info = self._get_model_info(provider)
            try:
                if pname == "openrouter":
                    if not settings.OPENROUTER_API_KEY or settings.OPENROUTER_API_KEY in {"", "sk-or-..."}:
                        return ("", "⚠ OpenRouter API key is not set — cloud models will fail")
                    # Probe the TEXT model. Vision-model id is also
                    # validated below so we don't burn 100 batches on a
                    # bad ``OPENROUTER_VISION_MODEL`` (e.g. "gpt-4o"
                    # instead of "openai/gpt-4o").
                    try:
                        await asyncio.wait_for(
                            provider.text_complete("Reply with OK", max_tokens=5, timeout=15),
                            timeout=20,
                        )
                    except asyncio.TimeoutError:
                        return ("", "⚠ OpenRouter text model timed out — may be overloaded. Will retry with fallbacks.")
                    except Exception as e:
                        return ("", f"⚠ OpenRouter text model check failed: {str(e)[:150]}")
                    # ── Vision-model existence check ──
                    # Hits OpenRouter's /api/v1/models (cached) and
                    # confirms the configured vision model is in the
                    # catalog. Quietly skipped on network failure so
                    # validate_models never becomes the reason a job
                    # can't start.
                    try:
                        vision_id = getattr(provider, "_vision_model", None)
                        if vision_id:
                            ok = await _openrouter_model_exists(vision_id)
                            if ok is False:
                                return (
                                    "",
                                    (
                                        f"\u26a0 OpenRouter vision model {vision_id!r} is "
                                        "not in the OpenRouter catalog. Scenes will be empty. "
                                        "Verify Settings \u2192 AI Provider \u2192 Vision."
                                    ),
                                )
                    except Exception:
                        pass
                    return (f"✓ {model_info} — models verified", None)

                elif pname == "ollama":
                    import httpx
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        resp = await client.head(f"{settings.OLLAMA_HOST}")
                        if resp.status_code != 200:
                            return ("", f"⚠ Ollama returned HTTP {resp.status_code}")
                    # ── Vision-model existence check ──
                    # ``/api/show`` returns 404 if the model isn't
                    # pulled. Avoids 30s of batch failures on first
                    # use of a model the user hasn't downloaded.
                    try:
                        vision_id = getattr(provider, "_vision_model", None)
                        if vision_id:
                            # Ollama's /api/show uses the ``model`` key and
                            # expects a bare tag (no ``ollama/`` prefix).
                            bare_id = vision_id
                            while isinstance(bare_id, str) and bare_id.lower().startswith("ollama/"):
                                bare_id = bare_id[len("ollama/"):]
                            async with httpx.AsyncClient(timeout=8.0) as client:
                                ping = await client.post(
                                    f"{settings.OLLAMA_HOST}/api/show",
                                    json={"model": bare_id},
                                )
                                if ping.status_code == 404:
                                    return (
                                        "",
                                        (
                                            f"\u26a0 Ollama vision model {bare_id!r} is not "
                                            f"pulled. Run `ollama pull {bare_id}` or pick a "
                                            "model that's already downloaded."
                                        ),
                                    )
                    except Exception:
                        pass
                    return (f"✓ {model_info} — Ollama connected", None)

                elif pname == "gemini":
                    if not settings.GEMINI_API_KEY:
                        return ("", "⚠ Gemini API key is not set")
                    return (f"✓ {model_info} — key present", None)

                elif pname == "anthropic":
                    if not settings.ANTHROPIC_API_KEY:
                        return ("", "⚠ Anthropic API key is not set")
                    return (f"✓ {model_info} — key present", None)

                elif pname == "groq":
                    if not settings.GROQ_API_KEY:
                        return ("", "⚠ Groq API key is not set")
                    return (f"✓ {model_info} — key present", None)
            except Exception as e:
                return ("", f"⚠ {pname} validation crashed: {str(e)[:150]}")
            return (f"Using {model_info}", None)

        # Notify frontend which models we're about to validate.
        for pname, provider in self._providers.items():
            await self._notify_attempt(job_id, provider, "model validation")

        results = await asyncio.gather(
            *(_validate_one(pname, p) for pname, p in self._providers.items()),
            return_exceptions=False,
        )
        for status_msg, warn_msg in results:
            if status_msg and self._ws_broadcast:
                try:
                    await self._ws_broadcast(job_id, {
                        "type": "status", "message": status_msg,
                    })
                except Exception:
                    pass
            if warn_msg:
                warnings.append(warn_msg)
                if self._ws_broadcast:
                    try:
                        await self._ws_broadcast(job_id, {
                            "type": "status", "message": warn_msg,
                        })
                    except Exception:
                        pass
        return warnings

    def get_total_tokens(self) -> int:
        """Return total tokens used across all provider instances."""
        return sum(p.total_tokens for p in self._providers.values())

    def estimate_cost(self) -> float:
        """Estimate total cost in USD from all provider token usage."""
        total = 0.0
        for name, provider in self._providers.items():
            tokens = provider.total_tokens
            if tokens > 0:
                rate = self._COST_PER_1K_TOKENS.get(name, 0.001)
                total += (tokens / 1000) * rate
        return round(total, 6)

    async def unload_local_models(self):
        """Unload Ollama models from VRAM so GPU is free for other tasks."""
        ollama = self._providers.get("ollama")
        if ollama and hasattr(ollama, "unload_models"):
            await ollama.unload_models()

    def reset_circuit_breaker(self):
        """Reset circuit breaker state before critical pipeline operations.

        Call this before summary generation and clip detection to ensure
        that failures from non-critical steps (transcript correction)
        don't block the core analysis pipeline.
        """
        self._circuit_breaker.force_reset_all()

    def get_text_model_info(self) -> dict:
        """Return info about the text model that will handle the next text_completion call.

        Used by transcript correction to:
        1. Log which model is polishing the transcript
        2. Set an appropriate timeout based on model type (thinking vs standard)
        """
        chain = self._get_active_chain()
        if not chain:
            return {"provider": "none", "model": "none", "is_thinking": False}
        provider = chain[0]
        return {
            "provider": provider.provider_name,
            "model": provider.text_model_name,
            "is_thinking": provider.is_thinking_model,
        }

    def _get_active_chain(self) -> list[AIProvider]:
        chain = []
        skipped = []
        unreachable = []
        for name in settings.active_provider_chain:
            if name in self._unreachable:
                unreachable.append(name)
                continue
            if name in self._providers and not self._circuit_breaker.is_degraded(name):
                chain.append(self._providers[name])
            elif name in self._providers:
                skipped.append(name)
        chain_names = [p.provider_name for p in chain]
        if skipped or unreachable:
            logger.info(
                "Active provider chain: %s (degraded: %s, unreachable: %s)",
                chain_names, skipped, unreachable,
            )
        else:
            logger.debug("Active provider chain: %s", chain_names)
        return chain

    async def _notify_attempt(self, job_id: str, provider, task: str):
        """Notify the frontend that a provider is being tried for a task.
        `provider` can be a string or an AIProvider instance."""
        if isinstance(provider, str):
            label = provider
        else:
            label = self._get_model_info(provider)
        logger.info("Attempting %s via %s", task, label)
        if self._ws_broadcast:
            try:
                await self._ws_broadcast(job_id, {
                    "type": "status",
                    "message": f"Attempting {task} via {label}...",
                })
            except Exception:
                pass

    async def _notify_fallback(self, job_id: str, from_provider: str, reason: str):
        reason_short = reason[:200] if len(reason) > 200 else reason
        logger.info("Falling back from %s: %s", from_provider, reason_short)
        if self._ws_broadcast:
            try:
                await self._ws_broadcast(job_id, {
                    "type": "fallback",
                    "from_provider": from_provider,
                    "to_provider": "next in chain",
                    "reason": reason_short,
                })
            except Exception:
                pass

    async def analyze_frames(
        self, frames: list[FrameData], job_id: str,
        progress_callback=None,
        content_type=None,
    ) -> tuple[list[SceneDescription], str]:
        """Returns (results, provider_name_used).
        progress_callback(frames_done, total_frames, provider_name) is called per batch.

        ``content_type`` is an optional ``ClipContentType`` (or
        string / None). When provided, the OpenRouter provider uses
        it to route ANIME / GAMEPLAY jobs to Qwen3-VL and leaves
        every other content type on the preset default. See
        ``select_vision_model_for_content`` in openrouter_provider.py.
        """
        self._wire_ws_to_providers(job_id)
        frame_prompt = self._custom_prompts.frame_analysis if self._custom_prompts else None
        # Append subject tracking instructions when enabled
        if settings.SUBJECT_TRACKING_ENABLED:
            st_prompt = (self._custom_prompts.subject_tracking if self._custom_prompts else None) or ""
            if st_prompt:
                base = frame_prompt or ""
                frame_prompt = f"{base}\n\n4. Subject position — {st_prompt}" if base else st_prompt
        for provider in self._get_active_chain():
            if not provider.supports_vision:
                continue
            try:
                await self._notify_attempt(job_id, provider, f"scene analysis ({len(frames)} frames)")

                async def _provider_progress(done, total):
                    if progress_callback:
                        await progress_callback(done, total, provider.provider_name)

                # Phase 2 — content-type-aware vision routing. Only
                # OpenRouter ships the preset system, so this is a
                # targeted call; other providers ignore the kwarg.
                _routing_ct = content_type
                try:
                    if hasattr(provider, "apply_vision_model_override"):
                        provider.apply_vision_model_override(_routing_ct)
                except Exception as err:
                    logger.debug("Vision model override skipped: %s", err)

                t0 = time.monotonic()
                result = await provider.analyze_frames(
                    frames, custom_prompt=frame_prompt,
                    cancel_check=self._cancel_check,
                    progress_callback=_provider_progress,
                )
                elapsed = time.monotonic() - t0
                logger.info("Scene analysis via %s completed in %.1fs (%d scenes)", provider.provider_name, elapsed, len(result))
                # Log subject tracking distribution for debugging
                if result:
                    sx_values = [s.subject_x for s in result]
                    sx_unique = len(set(sx_values))
                    all_default = all(v == 50 for v in sx_values)
                    logger.info(
                        "Subject tracking via %s: %d scenes, %d unique subject_x values, range [%d, %d], mean=%.1f%s",
                        provider.provider_name, len(sx_values), sx_unique,
                        min(sx_values), max(sx_values),
                        sum(sx_values) / len(sx_values),
                        " ⚠ ALL VALUES ARE 50 — model may not have detected subject positions" if all_default else "",
                    )
                self._circuit_breaker.record_success(provider.provider_name)
                return result, self._get_task_model(provider, "scenes")
            except (ProviderRateLimitError, ProviderError) as e:
                self._circuit_breaker.record_failure(provider.provider_name)
                await self._notify_fallback(job_id, provider.provider_name, str(e))
                continue
        raise AllProvidersFailedError("All vision providers failed")

    async def generate_summary(
        self,
        transcript: list[TranscriptSegment],
        scenes: list[SceneDescription],
        job_id: str,
        tier=None,
    ) -> tuple[VideoSummary, str]:
        """Returns (summary, provider_name_used).

        If tier specifies map_reduce strategy, splits transcript into chunks,
        summarizes each, then merges. This ensures long videos get full coverage.
        """
        if tier and tier.summary_strategy == "map_reduce" and tier.summary_chunk_minutes > 0:
            return await self._map_reduce_summary(transcript, scenes, job_id, tier)

        summary_prompt = self._custom_prompts.summary if self._custom_prompts else None
        for provider in self._get_active_chain():
            try:
                await self._notify_attempt(job_id, provider, "summary generation")
                t0 = time.monotonic()
                result = await provider.generate_summary(transcript, scenes, cancel_check=self._cancel_check, custom_prompt=summary_prompt)
                elapsed = time.monotonic() - t0
                logger.info("Summary generation via %s completed in %.1fs", provider.provider_name, elapsed)
                self._circuit_breaker.record_success(provider.provider_name)
                return result, self._get_task_model(provider, "summary")
            except (ProviderRateLimitError, ProviderError) as e:
                self._circuit_breaker.record_failure(provider.provider_name)
                await self._notify_fallback(job_id, provider.provider_name, str(e))
                continue
        raise AllProvidersFailedError("All providers failed for summary generation")

    async def _map_reduce_summary(
        self,
        transcript: list[TranscriptSegment],
        scenes: list[SceneDescription],
        job_id: str,
        tier,
    ) -> tuple[VideoSummary, str]:
        """Hierarchical map-reduce summary for long videos.

        Map: Split transcript into N-minute chunks, generate mini-summary per chunk.
        Reduce: Feed all mini-summaries into a final summary call.
        """
        from backend.services.providers.base import extract_json, has_real_summary_content

        chunk_seconds = tier.summary_chunk_minutes * 60
        if not transcript:
            from backend.services.providers.base import build_summary_from_transcript
            fb = build_summary_from_transcript(transcript, scenes)
            return VideoSummary(**fb), "fallback"

        video_end = max(s.end for s in transcript)
        chunk_overlap = 30.0  # 30 second overlap between chunks
        chunks: list[tuple[float, float, list[TranscriptSegment], list[SceneDescription]]] = []
        t = 0.0
        while t < video_end:
            chunk_end = min(t + chunk_seconds, video_end)
            # Extend segment selection by overlap into adjacent chunks
            chunk_segs = [s for s in transcript if s.start >= t - chunk_overlap and s.end <= chunk_end + chunk_overlap]
            chunk_scenes = [s for s in scenes if t - chunk_overlap <= s.timestamp <= chunk_end + chunk_overlap]
            chunks.append((t, chunk_end, chunk_segs, chunk_scenes))
            t = chunk_end

        chain = self._get_active_chain()
        is_ollama = chain and chain[0].provider_name == "ollama"
        max_segs = 20 if is_ollama else 50
        chunk_timeout = 90 if is_ollama else 60
        max_chunk_tokens = 300 if is_ollama else 500

        logger.info(
            "[%s] Map-reduce summary: %d chunks (%.0fs each), ollama=%s",
            job_id, len(chunks), chunk_seconds, is_ollama,
        )

        sem = asyncio.Semaphore(1 if is_ollama else 3)

        async def _summarize_chunk(idx, start, end, segs, scns):
            async with sem:
                time_label = f"{int(start//60)}:{int(start%60):02d}-{int(end//60)}:{int(end%60):02d}"
                text = "\n".join(
                    f"[{s.start:.0f}s] {s.speaker}: {s.text}"
                    for s in segs[:max_segs]
                )
                scene_text = "\n".join(
                    f"[{s.timestamp:.0f}s] {s.description[:60 if is_ollama else 100]}"
                    for s in scns[:5 if is_ollama else 10]
                )
                prompt = (
                    f"Summarize this {time_label} segment in 2-3 sentences. "
                    f"Include: main topic, key content, notable moments. "
                    f"Do not reference or speculate about speakers.\n\n"
                    f"TRANSCRIPT:\n{text}\n\nSCENES:\n{scene_text}\n\n"
                    f"Return a plain text summary (no JSON)."
                )
                try:
                    result = await self.text_completion(
                        prompt, max_tokens=max_chunk_tokens,
                        timeout=chunk_timeout,
                        job_id=job_id, skip_circuit_breaker=True,
                    )
                    # Broadcast chunk progress so the user sees activity
                    if self._ws_broadcast and job_id:
                        try:
                            await self._ws_broadcast(job_id, {
                                "type": "status",
                                "message": f"Summary: chunk {idx + 1}/{len(chunks)} complete...",
                            })
                        except Exception:
                            pass
                    return f"[{time_label}] {result.strip()}"
                except Exception as e:
                    logger.warning("[%s] Chunk %d summary failed: %s", job_id, idx, e)
                    return f"[{time_label}] {segs[0].text[:200] if segs else 'No content'}"

        results = await asyncio.gather(*[
            _summarize_chunk(i, s, e, segs, scns)
            for i, (s, e, segs, scns) in enumerate(chunks)
        ])
        mini_summaries = [r for r in results if r]

        # Reduce phase
        combined = "\n".join(mini_summaries)
        if is_ollama and len(combined) > 2500:
            combined = combined[:2500]

        reduce_prompt = (
            f"You have segment-by-segment summaries of a video. "
            f"Combine them into a cohesive summary. "
            f"Do not reference or speculate about speakers unless names are explicitly mentioned. "
            f"Focus on what is discussed, shown, and the key moments.\n\n"
            f"SEGMENT SUMMARIES:\n{combined}\n\n"
            "Return ONLY valid JSON:\n"
            '{"overview": "<2-4 sentence paragraph about the video content>", '
            '"key_topics": ["topic1", "topic2", "topic3"], '
            '"tone": "<1-2 words>", "estimated_audience": "<who would watch>", '
            '"content_category": "<specific category>"}\n'
            "key_topics MUST contain 3-6 specific topics from the video."
        )

        for provider in chain:
            try:
                raw = await asyncio.wait_for(
                    provider.text_complete(reduce_prompt, max_tokens=1000 if is_ollama else 2000),
                    timeout=120 if is_ollama else 90,
                )
                data = extract_json(raw)
                if has_real_summary_content(data):
                    return VideoSummary(**data), self._get_task_model(provider, "summary")
            except Exception as e:
                logger.warning("[%s] Reduce summary via %s failed: %s", job_id, provider.provider_name, e)
                continue

        # Fallback: use mini-summaries as overview
        overview = " ".join(mini_summaries[:6])
        if len(overview) > 500:
            overview = overview[:500].rsplit(" ", 1)[0] + "..."
        return VideoSummary(
            overview=overview,
            key_topics=[],
            tone="conversational",
            estimated_audience="general viewers",
            content_category="video content",
        ), "map_reduce_fallback"

    async def detect_viral_clips(
        self,
        transcript: list[TranscriptSegment],
        scenes: list[SceneDescription],
        video_duration: float,
        job_id: str,
        clip_count: Optional[int] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        clip_focus: Optional[str] = None,
        video_summary: Optional[str] = None,
        existing_clips: Optional[str] = None,
        hot_zones=None,
        progress_callback=None,
        tier=None,
        content_type=None,
        chapters=None,
        trend_context: Optional[str] = None,
        sentiment_timeline: Optional[str] = None,
        viral_score_min: int = 0,
        viral_score_max: int = 100,
        min_relevance: int = 0,
    ) -> tuple[list[ClipCandidate], str]:
        """Returns (clips, provider_name_used).

        ``content_type`` (Phase 2) is a ``ClipContentType`` enum value
        used to pick the genre-specific viral-clip prompt. ``chapters``
        (Phase 6), ``trend_context`` (Phase 3), and ``sentiment_timeline``
        (Phase 5) are optional context blocks injected into the prompt
        — providers that don't know what to do with them ignore them.
        """
        # Phase 2 — pick the genre-specific prompt unless the user has
        # an explicit override in custom_prompts. The user's custom
        # prompt always wins so they can fine-tune per deployment.
        custom_user_prompt = self._custom_prompts.viral_clip_detection if self._custom_prompts else None
        # ``PromptSet`` defaults to ``DEFAULT_VIRAL_CLIP_PROMPT``, so a
        # vanilla custom_prompts object would shadow the genre routing.
        # Detect the default-equals case and fall through to genre.
        from backend.services.prompts import (
            DEFAULT_VIRAL_CLIP_PROMPT, get_genre_prompt,
        )
        from backend.services.clip_scoring import (
            finalize_clip_scores, four_axis_scoring_enabled,
        )
        if custom_user_prompt and custom_user_prompt != DEFAULT_VIRAL_CLIP_PROMPT:
            clip_prompt = custom_user_prompt
            logger.info("Clip detection: using user custom prompt")
        else:
            clip_prompt = get_genre_prompt(content_type)
            ct_label = content_type.value if hasattr(content_type, "value") else (content_type or "generic")
            logger.info("Using genre prompt: %s", ct_label)
        # If clip_focus is provided, build an augmented focus prompt that
        # BUILDS ON the viral detection infrastructure rather than replacing it.
        # Enhancement 4 — the focus field accepts a comma- or newline-separated
        # list of queries which the LLM is instructed to treat as OR; each
        # returned clip's clip_focus field names the matching sub-query.
        if clip_focus and clip_focus.strip():
            focus_text = clip_focus.strip()
            focus_sub_queries = [
                q.strip() for q in re.split(r"[,\n]+", focus_text)
                if q.strip()
            ]
            is_multi_query = len(focus_sub_queries) > 1
            if is_multi_query:
                sub_bullets = "\n".join(f"  {i+1}. \"{q}\"" for i, q in enumerate(focus_sub_queries))
                clip_prompt = (
                    f"You are finding clips in a video that match ANY of several user-requested topics.\n\n"
                    f"USER'S FOCUS QUERIES (OR):\n{sub_bullets}\n\n"
                    f"For each clip, set clip_focus to the EXACT sub-query string it matched "
                    f"(one of the options above). If a clip strongly matches multiple sub-queries, "
                    f"pick the best match and note the overlap in viral_score_reasoning.\n\n"
                    f"SEMANTIC EXPANSION — expand each sub-query into synonyms and related concepts "
                    f"before searching. For example, \"fighting\" also covers combat, battle, "
                    f"argument, confrontation, sparring.\n\n"
                    f"RELEVANCE TIERS (apply per sub-query the clip matched):\n"
                    f"  Tier 1 (STRONG — score 80-100): the matched sub-query is the main subject.\n"
                    f"  Tier 2 (MODERATE — score 50-79): the matched sub-query is a significant part.\n"
                    f"  Tier 3 (WEAK — score 20-49): passing mention only (include only if <3 stronger).\n"
                    f"  EXCLUDE: mere keyword hits in passing or negations.\n\n"
                    f"SCORING: Put RELEVANCE (1-100) in 'viral_score' — do NOT use virality here. "
                    f"Also include 'focus_relevance' (1-100) and 'focus_tier' "
                    f"(\"strong\", \"moderate\", or \"weak\").\n\n"
                    f"SCENE & SUBJECT COHERENCE (CRITICAL):\n"
                    f"- The main subject MUST stay in focus throughout each clip\n"
                    f"- Keep the clip within ONE scene / exchange\n"
                    f"- Prefer natural speech boundaries for start/end\n\n"
                    f"BOUNDARY RULES:\n"
                    f"- Start at the beginning of a sentence\n"
                    f"- End at a clean exit point\n"
                    f"- Clips must work standalone without the rest of the video"
                )
                logger.info(
                    "Clip focus MULTI-QUERY mode for job %s: %d sub-queries",
                    job_id, len(focus_sub_queries),
                )
            else:
                clip_prompt = (
                    f"You are finding clips in a video that focus on a specific user-requested topic.\n\n"
                    f"USER'S FOCUS QUERY: \"{focus_text}\"\n\n"
                    f"SEMANTIC EXPANSION — Before searching, expand this query into related concepts:\n"
                    f"Think about synonyms, related terms, sub-topics, and adjacent concepts that someone "
                    f"searching for \"{focus_text}\" would also want to see. For example, if the focus is "
                    f"'fighting', also look for: combat, battle, argument, confrontation, sparring, conflict, "
                    f"physical altercation, self-defense, martial arts, etc.\n\n"
                    f"RELEVANCE TIERS:\n"
                    f"  Tier 1 (STRONG — score 80-100): The segment IS ABOUT '{focus_text}'. "
                    f"The topic is the main subject of discussion or the primary visual action.\n"
                    f"  Tier 2 (MODERATE — score 50-79): The segment discusses '{focus_text}' as a "
                    f"significant part of a broader conversation. Multiple sentences or visual moments relate to it.\n"
                    f"  Tier 3 (WEAK — score 20-49): The topic is mentioned briefly or tangentially. "
                    f"Only include Tier 3 clips if fewer than 3 Tier 1/2 clips exist.\n"
                    f"  EXCLUDE: Segments that merely mention a word related to '{focus_text}' in passing, "
                    f"negations ('I don't like {focus_text}'), or purely metaphorical usage.\n\n"
                    f"COMPOUND QUERIES: If the focus contains both a topic and a mood/quality "
                    f"(e.g., 'funny cooking moments'), prioritize segments matching BOTH aspects. "
                    f"Score clips higher when they combine the topic with the specified mood.\n\n"
                    f"SCORING: Use 'viral_score' to represent RELEVANCE to '{focus_text}' (not virality). "
                    f"A clip with 90 relevance means the segment is deeply, directly about the focus topic. "
                    f"In 'viral_score_reasoning', explain WHY this clip matches the focus query and which "
                    f"relevance tier it falls into.\n\n"
                    f"Additionally include 'focus_relevance' (1-100) and 'focus_tier' (\"strong\", \"moderate\", "
                    f"or \"weak\") in each clip's JSON.\n\n"
                    f"SCENE & SUBJECT COHERENCE (CRITICAL):\n"
                    f"- The main subject MUST stay in focus throughout the entire clip\n"
                    f"- NEVER cut across unrelated scenes or topics — the clip must feel like ONE moment\n"
                    f"- If a clip covers a conversation, keep it within the same exchange\n"
                    f"- The visual setting should remain consistent — don't span across location changes\n"
                    f"- Prefer segments where the camera stays on the main action without jarring cuts\n"
                    f"- If scene descriptions show different settings at different timestamps, do NOT combine them into one clip\n\n"
                    f"BOUNDARY RULES:\n"
                    f"- Start at natural speech boundaries — beginning of a sentence, after a pause, at a speaker change\n"
                    f"- End at natural conclusions — even if focus content extends further, find a clean exit point\n"
                    f"- Must work standalone without context from the full video\n"
                    f"- Prefer clips where the focus topic is introduced within the first 5 seconds"
                )
                logger.info("Clip focus mode active for job %s: '%s'", job_id, focus_text)

        # Bug 2 (clip-focus audit): inject score range and min-relevance
        # as *prompt* constraints so the LLM doesn't waste tokens
        # generating candidates we'll throw away. The post-hoc filter in
        # backend/routers/clips.py still runs as a safety net.
        is_focus_mode = bool(clip_focus and clip_focus.strip())
        try:
            viral_score_min = max(0, min(100, int(viral_score_min)))
            viral_score_max = max(0, min(100, int(viral_score_max)))
            min_relevance = max(0, min(100, int(min_relevance)))
        except (TypeError, ValueError):
            viral_score_min, viral_score_max, min_relevance = 0, 100, 0
        if viral_score_min > 0 or viral_score_max < 100:
            clip_prompt = clip_prompt + (
                f"\n\nSCORE RANGE HARD CONSTRAINT: only return clips whose composite "
                f"virality score falls inside [{viral_score_min}, {viral_score_max}]. "
                f"If you would score a clip below {viral_score_min} or above {viral_score_max}, "
                f"skip it entirely — DO NOT pad the clip count with borderline or off-target clips."
            )
        if is_focus_mode and min_relevance > 0:
            clip_prompt = clip_prompt + (
                f"\n\nRELEVANCE FLOOR: only return clips whose focus_relevance is "
                f">= {min_relevance}. If fewer than the requested count qualify, return "
                f"fewer clips — do not include off-topic filler."
            )

        # Phase 3 / 5 / 6 — append optional context blocks the LLM should
        # use for the four-axis scoring. Each block is fenced so the LLM
        # can clearly tell where the rubric ends and the data begins.
        appended_blocks: list[str] = []
        if trend_context:
            appended_blocks.append(
                "TREND CONTEXT (use this to populate trend_score):\n"
                + trend_context.strip()
            )
        if sentiment_timeline:
            appended_blocks.append(
                "SENTIMENT TIMELINE (audio sentiment moments — use for hook_score "
                "and value_score):\n" + sentiment_timeline.strip()
            )
        if chapters:
            try:
                chapter_lines = []
                for ch in chapters:
                    title = getattr(ch, "title", "") or ""
                    start = float(getattr(ch, "start", 0))
                    end = float(getattr(ch, "end", 0))
                    chapter_lines.append(
                        f"  [{start:.0f}-{end:.0f}s] {title}"
                    )
                if chapter_lines:
                    appended_blocks.append(
                        "CHAPTERS (the video has been pre-segmented into topic "
                        "chapters — try to find at least one strong clip per "
                        "chapter, do not cluster all clips in the first chapter):\n"
                        + "\n".join(chapter_lines)
                    )
            except Exception:
                pass
        if appended_blocks:
            clip_prompt = clip_prompt + "\n\n" + "\n\n".join(appended_blocks)

        # Per-provider timeout prevents any single provider from blocking the
        # fallback chain. Scale timeout based on video duration and provider type.
        vid_minutes = video_duration / 60 if video_duration else 0
        if vid_minutes > 30:
            # Windows processed 2 at a time (or sequentially for Ollama)
            est_windows = max(1, int(video_duration / 600) + 1)
            est_rounds = (est_windows + 1) // 2
            default_timeout = max(600, est_rounds * 330 + 300)
        else:
            default_timeout = 330

        # Shared partial results container. Providers populate this incrementally
        # so even if a timeout fires, we have whatever completed.
        _partial_clips: list = []

        for provider in self._get_active_chain():
            pname = provider.provider_name
            # Compute Ollama timeout: sequential windows need much more time
            if pname == "ollama":
                est_windows = max(1, int(video_duration / 300))
                timeout = max(420, est_windows * 300 + 120)
                logger.info("Ollama clip detection timeout: %ds (%d est. windows)", timeout, est_windows)
            else:
                timeout = default_timeout
            try:
                await self._notify_attempt(job_id, provider, "viral clip detection")
                t0 = time.monotonic()
                result = await asyncio.wait_for(
                    provider.detect_viral_clips(
                        transcript, scenes, video_duration,
                        custom_prompt=clip_prompt, cancel_check=self._cancel_check,
                        clip_count=clip_count, min_duration=min_duration,
                        max_duration=max_duration,
                        video_summary=video_summary,
                        existing_clips=existing_clips,
                        hot_zones=hot_zones,
                        progress_callback=progress_callback,
                        _partial_results=_partial_clips,
                        tier=tier,
                    ),
                    timeout=timeout,
                )
                elapsed = time.monotonic() - t0
                logger.info("Clip detection via %s completed in %.1fs (%d clips)", pname, elapsed, len(result))
                self._circuit_breaker.record_success(pname)
                # Phase 4 — recompose viral_score from the four axes
                # using the genre-aware weights. No-op when the four
                # axes are all zero (legacy clip path).
                if four_axis_scoring_enabled():
                    finalize_clip_scores(result, content_type, focus_mode=is_focus_mode)
                return result, self._get_task_model(provider, "clips")
            except asyncio.TimeoutError:
                elapsed = time.monotonic() - t0
                # Check if partial results were collected before timeout
                if _partial_clips:
                    if hasattr(provider, '_deduplicate_clips'):
                        deduped = provider._deduplicate_clips(_partial_clips)
                    else:
                        deduped = _partial_clips
                    logger.warning(
                        "Clip detection via %s timed out after %ds but recovered %d partial clips",
                        pname, timeout, len(deduped),
                    )
                    if four_axis_scoring_enabled():
                        finalize_clip_scores(deduped, content_type, focus_mode=is_focus_mode)
                    return deduped, f"{self._get_task_model(provider, 'clips')} (partial)"
                logger.warning("Clip detection via %s timed out after %ds", pname, timeout)
                self._circuit_breaker.record_failure(pname)
                await self._notify_fallback(job_id, pname, f"Timed out after {timeout}s")
                continue
            except (ProviderRateLimitError, ProviderError) as e:
                self._circuit_breaker.record_failure(pname)
                await self._notify_fallback(job_id, pname, str(e))
                continue

        # Even if all providers "failed", check partial results
        if _partial_clips:
            logger.warning(
                "All providers failed but recovered %d partial clips", len(_partial_clips),
            )
            if four_axis_scoring_enabled():
                finalize_clip_scores(_partial_clips, content_type, focus_mode=is_focus_mode)
            return _partial_clips, "partial"
        raise AllProvidersFailedError("All providers failed for viral clip detection")

    async def _maybe_downgrade_ollama_model(self, provider) -> None:
        """After consecutive Ollama failures, clear VRAM and try a smaller model."""
        self._consecutive_ollama_failures += 1
        if self._consecutive_ollama_failures < self._FAILURE_THRESHOLD_FOR_DOWNGRADE:
            return

        logger.warning(
            "Ollama model failed %d times consecutively. Attempting VRAM clear and model downgrade.",
            self._consecutive_ollama_failures,
        )

        # Clear VRAM
        if hasattr(provider, 'clear_vram'):
            await provider.clear_vram()

        # Try smaller models
        import httpx as _httpx
        for smaller_model in self._SMALLER_MODELS:
            try:
                async with _httpx.AsyncClient(timeout=10) as client:
                    resp = await client.post(
                        f"{provider._host}/api/show",
                        json={"model": smaller_model},
                    )
                    if resp.status_code == 200:
                        logger.info("Downgrading to smaller model: %s", smaller_model)
                        self._current_model_override = smaller_model
                        self._consecutive_ollama_failures = 0
                        return
            except Exception:
                continue

    async def text_completion(self, prompt: str, max_tokens: int = 4096, timeout: float = 60, job_id: str = "", skip_circuit_breaker: bool = False) -> str:
        """Generic text completion using the configured provider chain.

        Used by transcript correction, translation, and other text-only tasks.
        Falls back through the provider chain on failure.
        Returns the raw text response from the first successful provider.

        Args:
            skip_circuit_breaker: If True, failures are NOT recorded in the
                circuit breaker. Use this for non-critical/optional operations
                (like transcript polishing) that should not degrade the provider
                for subsequent critical operations (summary, clip detection).
        """
        for provider in self._get_active_chain():
            pname = provider.provider_name
            model_name = provider.text_model_name
            # Apply model override for Ollama if we've downgraded after failures
            if pname == "ollama" and self._current_model_override:
                model_name = self._current_model_override
                # Temporarily override the provider's text model
                original_model = provider._text_model
                provider._text_model = self._current_model_override
            else:
                original_model = None
            try:
                # Clear VRAM before first Ollama call in a job
                if pname == "ollama" and hasattr(provider, 'clear_vram') and self._consecutive_ollama_failures == 0 and not self._current_model_override:
                    await provider.clear_vram()
                logger.info("text_completion attempting via %s model=%s (%d chars prompt)", pname, model_name, len(prompt))
                t0 = time.monotonic()
                result = await asyncio.wait_for(
                    provider.text_complete(prompt, max_tokens=max_tokens, timeout=int(timeout)),
                    timeout=timeout,
                )
                elapsed = time.monotonic() - t0
                logger.info("text_completion via %s model=%s completed in %.1fs", pname, model_name, elapsed)
                if not skip_circuit_breaker:
                    self._circuit_breaker.record_success(pname)
                # Reset consecutive failure counter on success
                if pname == "ollama":
                    self._consecutive_ollama_failures = 0
                return result
            except asyncio.TimeoutError:
                if not skip_circuit_breaker:
                    self._circuit_breaker.record_failure(pname)
                logger.warning("text_completion via %s model=%s timed out after %.0fs — trying next provider", pname, model_name, timeout)
                if pname == "ollama":
                    await self._maybe_downgrade_ollama_model(provider)
                await self._notify_fallback(job_id, pname, f"Text completion timed out after {timeout:.0f}s (model={model_name})")
                continue
            except Exception as e:
                if not skip_circuit_breaker:
                    self._circuit_breaker.record_failure(pname)
                logger.warning("text_completion via %s model=%s failed: %s — trying next provider", pname, model_name, e)
                if pname == "ollama" and ("stalled" in str(e).lower() or "overloaded" in str(e).lower()):
                    await self._maybe_downgrade_ollama_model(provider)
                await self._notify_fallback(job_id, pname, str(e))
                continue
            finally:
                # Restore original model if we overrode it
                if original_model is not None:
                    provider._text_model = original_model
        raise AllProvidersFailedError("All providers failed for text completion")

    async def generate_seo(
        self,
        clip_title: str,
        clip_transcript: str,
        video_summary: str,
        platform: str,
        job_id: str,
    ) -> tuple[ClipSEO, str]:
        """Returns (seo, provider_name_used)."""
        seo_prompt = self._custom_prompts.seo if self._custom_prompts else None
        for provider in self._get_active_chain():
            try:
                await self._notify_attempt(job_id, provider, "SEO generation")
                t0 = time.monotonic()
                result = await provider.generate_seo(
                    clip_title, clip_transcript, video_summary,
                    platform, cancel_check=self._cancel_check,
                    custom_prompt=seo_prompt,
                )
                elapsed = time.monotonic() - t0
                logger.info("SEO generation via %s completed in %.1fs", provider.provider_name, elapsed)
                self._circuit_breaker.record_success(provider.provider_name)
                return result, self._get_task_model(provider, "seo")
            except (ProviderRateLimitError, ProviderError) as e:
                self._circuit_breaker.record_failure(provider.provider_name)
                await self._notify_fallback(job_id, provider.provider_name, str(e))
                continue
        raise AllProvidersFailedError("All providers failed for SEO generation")
