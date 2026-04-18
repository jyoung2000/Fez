"""Per-user settings overlay.

Applies a user's persisted settings (API keys, model picks, whisper
options) over the global ``backend.config.settings`` object for the
duration of a context block. Used by the analysis pipeline to make
``AIOrchestrator()`` see the *job-owner's* keys at construction
without rewriting every provider to accept an explicit ``api_key=``
kwarg.

Implementation notes:

  * pydantic-settings ``Settings`` is mutable at the attribute level
    (``object.__setattr__``); we snapshot the originals before
    overlaying and restore on exit even if the body raises.

  * Concurrency: the overlay mutates the global ``settings`` object,
    so two overlays running concurrently from different jobs would
    race. The pipeline holds the overlay open for only a few ms during
    orchestrator construction, so the race window is small but not
    zero. Future work: refactor providers to accept an explicit
    ``api_key`` so we can drop the global mutation.

  * Async-safe: an ``async with`` variant exists so the overlay can be
    awaited while the user's settings are read from disk.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager, contextmanager
from typing import Iterable, Optional

from backend.config import settings as _settings
from backend.app.auth.store import read_user_settings

logger = logging.getLogger(__name__)


# Keys we are willing to overlay. Matches the allow-list in the
# /api/auth/me/settings PUT endpoint.
OVERLAYABLE_KEYS: tuple[str, ...] = (
    "AI_FALLBACK_CHAIN",
    "OPENROUTER_PRESET",
    "OPENROUTER_VISION_MODEL",
    "OPENROUTER_TEXT_MODEL",
    "OPENROUTER_SUMMARY_MODEL",
    "OLLAMA_VISION_MODEL",
    "OLLAMA_TEXT_MODEL",
    "OLLAMA_TRANSLATION_MODEL",
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "HF_AUTH_TOKEN",
    "WHISPER_MODEL",
    "WHISPER_BEAM_SIZE",
    "WHISPER_VAD_FILTER",
    # Phase 1 transcript-coverage knobs — per-user overridable so power
    # users can tune sensitivity without affecting other users.
    "WHISPER_AUDIO_PRECONDITION",
    "WHISPER_VAD_ONSET",
    "WHISPER_NO_SPEECH_THRESHOLD",
    "WHISPER_AUTO_UPGRADE",
    # Per-user model picks for Anthropic / Gemini / Groq. Without
    # these, users could only override OpenRouter / Ollama models
    # via the settings overlay, leaving cloud-only providers stuck
    # on whatever the install default ``MODEL`` constant was.
    "ANTHROPIC_MODEL",
    "GEMINI_TEXT_MODEL",
    "GEMINI_VIDEO_MODEL",
    "GROQ_TEXT_MODEL",
)


_OLLAMA_MODEL_KEYS = (
    "OLLAMA_VISION_MODEL",
    "OLLAMA_TEXT_MODEL",
    "OLLAMA_TRANSLATION_MODEL",
)


def _strip_ollama_prefix(value: str) -> str:
    """Ollama's HTTP API only knows bare tags (``moondream:1.8b``).
    The UI stores model picks with a provider prefix (``ollama/...``)
    to disambiguate them from OpenRouter picks; strip it here so the
    overlaid setting is the form the Ollama client actually needs.
    """
    v = value.strip()
    while v.lower().startswith("ollama/"):
        v = v[len("ollama/"):]
    return v


def _coerce(key: str, value):
    """Best-effort conversion of stored str into the type ``settings``
    expects. Booleans / ints get parsed; everything else stays str."""
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    if key.endswith("_VAD_FILTER") or key in (
        "WHISPER_AUDIO_PRECONDITION", "WHISPER_AUTO_UPGRADE",
    ):
        return value.lower() in ("1", "true", "yes", "on")
    if key in ("WHISPER_BEAM_SIZE",):
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if key in ("WHISPER_VAD_ONSET", "WHISPER_NO_SPEECH_THRESHOLD"):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    if key in _OLLAMA_MODEL_KEYS:
        return _strip_ollama_prefix(value)
    return value


def apply_overlay_dict(overrides: dict) -> dict:
    """Apply ``overrides`` to ``settings`` in place. Returns a
    snapshot dict suitable for ``restore_overlay``.
    """
    snapshot: dict = {}
    if not overrides:
        return snapshot
    for key in OVERLAYABLE_KEYS:
        if key not in overrides:
            continue
        new_val = overrides.get(key)
        if new_val in (None, ""):
            continue
        try:
            snapshot[key] = getattr(_settings, key, None)
        except Exception:
            snapshot[key] = None
        try:
            object.__setattr__(_settings, key, _coerce(key, new_val))
            # Also expose to subprocesses via env so faster-whisper
            # / ollama subprocesses get the overlay.
            os.environ[key] = str(_coerce(key, new_val))
        except Exception as e:
            logger.warning("settings overlay: failed to set %s: %s", key, e)

    # Honor a per-user ``WHISPER_MODEL`` pin: the auto-tune logic in
    # ``backend/services/transcription.py`` upgrades / downgrades the
    # whisper model based on detected VRAM unless ``WHISPER_MODEL_USER_SET``
    # is truthy. When the overlay deliberately picks a model, set the
    # flag so the user's choice survives auto-tune.
    if "WHISPER_MODEL" in overrides and overrides.get("WHISPER_MODEL"):
        try:
            snapshot.setdefault(
                "WHISPER_MODEL_USER_SET",
                getattr(_settings, "WHISPER_MODEL_USER_SET", False),
            )
            object.__setattr__(_settings, "WHISPER_MODEL_USER_SET", True)
            os.environ["WHISPER_MODEL_USER_SET"] = "true"
        except Exception as e:
            logger.warning("settings overlay: failed to pin WHISPER_MODEL_USER_SET: %s", e)
    return snapshot


def restore_overlay(snapshot: dict) -> None:
    for key, val in (snapshot or {}).items():
        try:
            object.__setattr__(_settings, key, val)
            if val is None or val == "":
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(val)
        except Exception as e:
            logger.warning("settings overlay: failed to restore %s: %s", key, e)


@contextmanager
def overlay_user_settings_sync(overrides: dict):
    """Sync context manager. Use when overrides are already in hand."""
    snap = apply_overlay_dict(overrides)
    try:
        yield
    finally:
        restore_overlay(snap)


@asynccontextmanager
async def overlay_user_settings(user_id: Optional[str]):
    """Async overlay that pulls the user's persisted settings from disk.

    No-op when ``user_id`` is empty (legacy jobs / unauthenticated
    flow) so the global settings still apply.
    """
    if not user_id:
        yield {}
        return
    overrides = await read_user_settings(user_id)
    snap = apply_overlay_dict(overrides)
    try:
        yield overrides
    finally:
        restore_overlay(snap)
