import asyncio
import base64
import json
import logging
import os
import re
import threading
import time
import uuid

import httpx
from fastapi import APIRouter, Depends, File, Request, UploadFile, Form
from pydantic import BaseModel

from typing import Optional

from backend.app.auth.deps import get_current_user
from backend.app.auth.models import User
from backend.app.auth.store import read_user_settings, write_user_settings
from backend.config import Settings, settings, get_settings
from backend.services.prompts import (
    PromptSet, load_prompts, save_prompts, get_defaults, MAX_PROMPT_LENGTH,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["settings"])


def _resolve_data_dir() -> str:
    """Find a writable data directory for persisting settings and caches.
    Prefers /data/logs (Docker volume mount), falls back to a local .clipai dir."""
    docker_path = "/data/logs"
    if os.path.isdir(docker_path) and os.access(docker_path, os.W_OK):
        return docker_path
    # Fallback: project-local directory (works outside Docker)
    local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".clipai")
    os.makedirs(local_path, exist_ok=True)
    return local_path


_DATA_DIR = _resolve_data_dir()
MODEL_CACHE_PATH = os.path.join(_DATA_DIR, "model_cache.json")
MODEL_CACHE_TTL = 86400  # 24 hours

# Persistent user settings — saved so they survive container/process restarts.
USER_SETTINGS_PATH = os.path.join(_DATA_DIR, "user_settings.json")

_PLACEHOLDER_KEYS = {"sk-or-...", "sk-ant-...", "AIza...", "gsk_...", ""}

# Keys that are persisted to user_settings.json
_PERSISTABLE_KEYS = [
    "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY",
    "HF_AUTH_TOKEN",
    "OPENROUTER_PRESET", "OPENROUTER_VISION_MODEL", "OPENROUTER_TEXT_MODEL",
    "OPENROUTER_SUMMARY_MODEL", "OLLAMA_VISION_MODEL", "OLLAMA_TEXT_MODEL", "OLLAMA_TRANSLATION_MODEL",
    "WHISPER_MODEL", "WHISPER_MODEL_USER_SET", "WHISPER_BEAM_SIZE",
    "WHISPER_VAD_FILTER", "FRAME_SAMPLE_RATE", "SUBJECT_TRACKING_ENABLED",
    "FFMPEG_PRESET", "FFMPEG_CRF", "FFMPEG_THREADS", "FFMPEG_FASTSTART",
    "GPU_ACCELERATION_ENABLED", "GPU_VENDOR_OVERRIDE",
    "GPU_HWDECODE_ENABLED", "GPU_HEVC_FOR_4K", "GPU_DEVICE_INDEX",
    "AI_FALLBACK_CHAIN",
    # Cloud storage OAuth credentials — entered via the Settings > Cloud
    # Storage UI and persisted so containers without env vars can still
    # connect to Google Drive / Box after the user pastes their credentials.
    "CLIPAI_CLOUD_STORAGE_ENABLED",
    "GOOGLE_DRIVE_CLIENT_ID", "GOOGLE_DRIVE_CLIENT_SECRET", "GOOGLE_DRIVE_REDIRECT_URI",
    "BOX_CLIENT_ID", "BOX_CLIENT_SECRET", "BOX_REDIRECT_URI",
]

# API key fields specifically (used to filter out placeholder values)
_API_KEY_FIELDS = {
    "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY",
    "HF_AUTH_TOKEN",
    # Cloud client secrets — same "never overwrite with blank" rule.
    "GOOGLE_DRIVE_CLIENT_SECRET", "BOX_CLIENT_SECRET",
}


def _is_real_value(key: str, val: str) -> bool:
    """Check if a value is a real user-entered value (not empty or a placeholder)."""
    if not val:
        return False
    if val in _PLACEHOLDER_KEYS:
        return False
    return True


async def _user_setting(user_id: str, key: str) -> str | bool | int | None:
    """Read a single setting for ``user_id``, falling back to global."""
    us = await read_user_settings(user_id)
    if key in us:
        return us[key]
    return getattr(settings, key, "")


def _persist_user_settings() -> bool:
    """Save all user-mutable settings to a JSON file that survives restarts.

    Returns True if settings were persisted successfully, False otherwise.
    The file is stored on a Docker volume mount (/data/logs) so it
    survives container stop/restart/recreate cycles.

    IMPORTANT: Merges with existing file to prevent API key loss.
    If the current in-memory value for an API key is empty but the
    existing file has a real key, the persisted key is preserved.
    This prevents non-key setting changes from wiping out API keys
    that were saved earlier but not restored into memory.
    """
    # Load existing persisted data to preserve API keys we might not have in memory
    existing_data = {}
    try:
        if os.path.exists(USER_SETTINGS_PATH):
            with open(USER_SETTINGS_PATH, "r") as f:
                existing_data = json.load(f)
    except Exception:
        pass

    data = {}
    skipped_keys = []

    # Protect user-set WHISPER_MODEL from being overwritten by runtime auto-upgrades/downgrades.
    # If WHISPER_MODEL_USER_SET is True in the existing file, keep the file's WHISPER_MODEL
    # unless the user explicitly changed it via the save_models endpoint (which sets
    # WHISPER_MODEL_USER_SET = True in the current settings too).
    _whisper_user_set_in_file = existing_data.get("WHISPER_MODEL_USER_SET", False)
    _whisper_model_in_file = existing_data.get("WHISPER_MODEL")

    for key in _PERSISTABLE_KEYS:
        val = getattr(settings, key, "")
        # Non-string types (bool, int) are always persisted
        if isinstance(val, (bool, int)):
            data[key] = val
            continue
        # For API keys: if current in-memory value is empty/placeholder but
        # the existing file has a real key, preserve the persisted key.
        # This prevents non-key setting changes from wiping out saved keys.
        if key in _API_KEY_FIELDS:
            if _is_real_value(key, val):
                data[key] = val
            elif key in existing_data and _is_real_value(key, existing_data[key]):
                data[key] = existing_data[key]
                logger.debug("Preserving %s from existing file (in-memory is empty)", key)
            else:
                skipped_keys.append(key)
            continue

        # Protect WHISPER_MODEL: if the user explicitly set a model in the file
        # but the in-memory value differs (due to auto-upgrade/downgrade), keep
        # the user's saved choice. The auto-upgrade only affects the current session.
        if key == "WHISPER_MODEL" and _whisper_user_set_in_file and _whisper_model_in_file:
            if _is_real_value(key, val) and val != _whisper_model_in_file:
                # In-memory value differs from user's saved choice — check if the
                # current settings object also has WHISPER_MODEL_USER_SET=True
                # (meaning the user just changed it via the UI in this session)
                if not getattr(settings, "WHISPER_MODEL_USER_SET", False):
                    # Auto-change, not user change — preserve the file value
                    data[key] = _whisper_model_in_file
                    logger.info(
                        "Preserving user's saved WHISPER_MODEL='%s' (runtime has '%s' from auto-upgrade/downgrade)",
                        _whisper_model_in_file, val,
                    )
                    continue

        # Skip empty values for non-key settings
        if not _is_real_value(key, val):
            continue
        data[key] = val

    # Log what API keys are being saved (or not)
    for key in _API_KEY_FIELDS:
        if key in data:
            logger.info("Persisting %s: YES (value present, %d chars)", key, len(str(data[key])))
        elif key in skipped_keys:
            logger.debug("Persisting %s: NO (empty in memory and file)", key)

    try:
        os.makedirs(os.path.dirname(USER_SETTINGS_PATH), exist_ok=True)
        with open(USER_SETTINGS_PATH, "w") as f:
            json.dump(data, f, indent=2)
        # Verify critical values were written
        has_key = "OPENROUTER_API_KEY" in data
        whisper = data.get("WHISPER_MODEL", "?")
        chain = data.get("AI_FALLBACK_CHAIN", "?")
        logger.info(
            "Persisted %d settings to %s (WHISPER_MODEL=%s, API_KEY=%s, CHAIN=%s)",
            len(data), USER_SETTINGS_PATH, whisper,
            f"YES({len(data['OPENROUTER_API_KEY'])}ch)" if has_key else "NO",
            chain,
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to persist user settings to {USER_SETTINGS_PATH}: {e}")
        return False


def _restore_user_settings():
    """Load persisted settings and apply them to the settings object.
    Called once at module import time so saved API keys survive restarts.

    API keys are ALWAYS restored from the persisted file — they take priority
    over environment variables and defaults. This ensures user-entered keys
    survive container recreate cycles where the .env file is lost."""
    if not os.path.exists(USER_SETTINGS_PATH):
        logger.info(f"No persisted settings found at {USER_SETTINGS_PATH}")
        return
    try:
        with open(USER_SETTINGS_PATH, "r") as f:
            data = json.load(f)
        restored = 0

        # Log what's in the file for diagnostics
        api_keys_in_file = [k for k in _API_KEY_FIELDS if k in data and _is_real_value(k, data[k])]
        model_keys_in_file = [
            k for k in ["WHISPER_MODEL", "OPENROUTER_VISION_MODEL", "OPENROUTER_TEXT_MODEL",
                         "AI_FALLBACK_CHAIN", "WHISPER_MODEL_USER_SET"]
            if k in data
        ]
        logger.info(
            "Restoring from %s: %d keys total, API keys: %s, model settings: %s",
            USER_SETTINGS_PATH, len(data),
            api_keys_in_file or "none",
            {k: data[k] for k in model_keys_in_file},
        )

        for key, val in data.items():
            if key not in _PERSISTABLE_KEYS:
                continue
            # Bool/int types: always restore from persisted value
            if isinstance(val, (bool, int)):
                setattr(settings, key, val)
                restored += 1
                continue
            if not _is_real_value(key, val):
                continue
            # For API keys: ALWAYS restore persisted real keys.
            # The persisted value is the most recent user-entered key and
            # should override env defaults, placeholders, and even env vars
            # (the user explicitly saved via the UI after the env was set).
            if key in _API_KEY_FIELDS:
                setattr(settings, key, val)
                logger.info(f"Restored API key: {key} ({len(val)} chars)")
                restored += 1
            else:
                # For model/preset settings: always restore persisted values.
                # The persisted file represents the user's last explicit choice.
                setattr(settings, key, val)
                restored += 1
        logger.info(f"Restored {restored} persisted settings from {USER_SETTINGS_PATH}")

        # Warn about common misconfigurations after restore
        if "openrouter" in (getattr(settings, "AI_FALLBACK_CHAIN", "") or "").lower():
            key = getattr(settings, "OPENROUTER_API_KEY", "")
            if not key or key in _PLACEHOLDER_KEYS:
                logger.warning(
                    "⚠ OpenRouter is in the fallback chain but OPENROUTER_API_KEY is empty! "
                    "Cloud AI models will fail. Enter your API key in Settings > AI Provider."
                )
        whisper = getattr(settings, "WHISPER_MODEL", "small")
        user_set = getattr(settings, "WHISPER_MODEL_USER_SET", False)
        logger.info(
            "Whisper config after restore: model=%s, user_set=%s",
            whisper, user_set,
        )
    except Exception as e:
        logger.warning(f"Failed to restore user settings from {USER_SETTINGS_PATH}: {e}")


# Restore saved settings on module load
_restore_user_settings()


def _backfill_api_keys():
    """Ensure API keys loaded from .env are also persisted to user_settings.json.

    When a user upgrades from an older version that only saved keys to .env,
    the key is loaded by pydantic into memory but may not be in user_settings.json.
    On container recreate, .env is lost. This backfill captures any in-memory keys
    that are missing from the persisted file, preventing key loss on updates.
    """
    if not os.path.exists(USER_SETTINGS_PATH):
        return
    try:
        with open(USER_SETTINGS_PATH, "r") as f:
            data = json.load(f)
        updated = False
        for key in _API_KEY_FIELDS:
            in_memory = getattr(settings, key, "")
            in_file = data.get(key, "")
            if _is_real_value(key, in_memory) and not _is_real_value(key, in_file):
                data[key] = in_memory
                updated = True
                logger.info(
                    "Backfilling %s from memory to user_settings.json (%d chars)",
                    key, len(in_memory),
                )
        if updated:
            with open(USER_SETTINGS_PATH, "w") as f:
                json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning("API key backfill failed (non-fatal): %s", e)


_backfill_api_keys()


# ── One-time migration: add WHISPER_MODEL_USER_SET flag if missing ──
# Older versions didn't persist this flag. If the file has a non-default
# Whisper model but no flag, preserve the model and mark it as user-set
# (the user kept it, whether originally from auto-upgrade or manual choice).
# Previously this reset to "small", which wiped users' model selections.
def _migrate_stale_whisper():
    if not os.path.exists(USER_SETTINGS_PATH):
        return
    try:
        with open(USER_SETTINGS_PATH, "r") as f:
            data = json.load(f)
        whisper = data.get("WHISPER_MODEL")
        user_set = data.get("WHISPER_MODEL_USER_SET")
        if whisper and whisper != "small" and user_set is None:
            # Flag was missing — preserve the model and mark as user-set.
            # The user had this model saved, so it represents their preference
            # regardless of how it was originally selected.
            logger.info(
                "Migration: WHISPER_MODEL='%s' with no WHISPER_MODEL_USER_SET flag — "
                "preserving model and marking as user-set.",
                whisper,
            )
            data["WHISPER_MODEL_USER_SET"] = True
            with open(USER_SETTINGS_PATH, "w") as f:
                json.dump(data, f, indent=2)
            settings.WHISPER_MODEL_USER_SET = True
    except Exception as e:
        logger.warning("Migration check failed (non-fatal): %s", e)

_migrate_stale_whisper()

# Short-lived cache for /api/providers/status (avoid hammering Ollama on rapid re-renders)
_status_cache: dict = {}
_status_cache_ts: float = 0
_STATUS_CACHE_TTL = 5  # seconds

# Env var names per provider
_PROVIDER_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "huggingface": "HF_AUTH_TOKEN",
}

# -- Cost estimation for a 10-min video --
# 120 frames (sampled every 5s), each ~765 tokens as image input
# Vision: 120 images * 765 ≈ 92K input tokens, ~12K output tokens
# Text: ~15K input tokens (transcript+scenes), ~8K output tokens (summary+clips)
_VISION_INPUT_TOKENS_10MIN = 100_000
_VISION_OUTPUT_TOKENS_10MIN = 15_000
_TEXT_INPUT_TOKENS_10MIN = 15_000
_TEXT_OUTPUT_TOKENS_10MIN = 8_000

# Speed ratings for known model families (estimated minutes to analyze a 10-min video).
# Vision: frame analysis across ~60 frames. Text: summary + clip detection.
# "speed" = "fast" | "medium" | "slow", "est_minutes" = estimated wall-clock minutes
_MODEL_SPEED_PROFILES = {
    # --- Free auto-router ---
    # quality_score: 1=poor, 2=basic, 3=good, 4=excellent, 5=best
    "openrouter/free": {"speed": "medium", "est_minutes_vision": 5.0, "est_minutes_text": 2.0, "quality": "basic", "quality_score": 2},
    # --- Fast models (under 2 min for 10-min video) ---
    "gemini-2.5-flash": {"speed": "fast", "est_minutes_vision": 1.5, "est_minutes_text": 0.5, "quality": "good", "quality_score": 3},
    "gemini-2.0-flash": {"speed": "fast", "est_minutes_vision": 1.5, "est_minutes_text": 0.5, "quality": "good", "quality_score": 3},
    "gemini-flash": {"speed": "fast", "est_minutes_vision": 1.5, "est_minutes_text": 0.5, "quality": "good", "quality_score": 3},
    "llama-3.1-8b": {"speed": "fast", "est_minutes_vision": 0, "est_minutes_text": 0.5, "quality": "basic", "quality_score": 2},
    "llama-3.3-70b": {"speed": "fast", "est_minutes_vision": 0, "est_minutes_text": 1.0, "quality": "good", "quality_score": 3},
    "qwen": {"speed": "fast", "est_minutes_vision": 2.0, "est_minutes_text": 0.8, "quality": "good", "quality_score": 3},
    "mistral": {"speed": "fast", "est_minutes_vision": 0, "est_minutes_text": 0.7, "quality": "good", "quality_score": 3},
    "deepseek": {"speed": "fast", "est_minutes_vision": 0, "est_minutes_text": 1.0, "quality": "good", "quality_score": 3},
    # --- Medium models (2-5 min) ---
    "gemini-2.5-pro": {"speed": "medium", "est_minutes_vision": 3.0, "est_minutes_text": 1.5, "quality": "excellent", "quality_score": 4},
    "gpt-4o-mini": {"speed": "medium", "est_minutes_vision": 2.5, "est_minutes_text": 1.0, "quality": "good", "quality_score": 3},
    "gpt-4o": {"speed": "medium", "est_minutes_vision": 3.5, "est_minutes_text": 1.5, "quality": "excellent", "quality_score": 4},
    "claude-haiku": {"speed": "medium", "est_minutes_vision": 2.0, "est_minutes_text": 1.0, "quality": "good", "quality_score": 3},
    "pixtral": {"speed": "medium", "est_minutes_vision": 3.0, "est_minutes_text": 1.5, "quality": "good", "quality_score": 3},
    # --- Slow models (5+ min) ---
    "claude-sonnet": {"speed": "slow", "est_minutes_vision": 5.0, "est_minutes_text": 2.5, "quality": "excellent", "quality_score": 4},
    "claude-opus": {"speed": "slow", "est_minutes_vision": 8.0, "est_minutes_text": 4.0, "quality": "best", "quality_score": 5},
    "gpt-4-turbo": {"speed": "slow", "est_minutes_vision": 5.0, "est_minutes_text": 2.0, "quality": "excellent", "quality_score": 4},
    "o1": {"speed": "slow", "est_minutes_vision": 6.0, "est_minutes_text": 3.0, "quality": "excellent", "quality_score": 4},
    "o3": {"speed": "slow", "est_minutes_vision": 7.0, "est_minutes_text": 3.5, "quality": "best", "quality_score": 5},
    # --- Vision model families ---
    "qwen2.5-vl": {"speed": "fast", "est_minutes_vision": 2.0, "est_minutes_text": 0.8, "quality": "good", "quality_score": 3},
    "qwen3-vl": {"speed": "fast", "est_minutes_vision": 2.0, "est_minutes_text": 0.8, "quality": "excellent", "quality_score": 4},
    "kimi-vl": {"speed": "medium", "est_minutes_vision": 3.0, "est_minutes_text": 1.5, "quality": "good", "quality_score": 3},
    "kimi-k2": {"speed": "medium", "est_minutes_vision": 3.0, "est_minutes_text": 1.5, "quality": "excellent", "quality_score": 4},
    "internvl": {"speed": "medium", "est_minutes_vision": 3.0, "est_minutes_text": 1.5, "quality": "good", "quality_score": 3},
    "minicpm-v": {"speed": "fast", "est_minutes_vision": 2.0, "est_minutes_text": 1.0, "quality": "good", "quality_score": 3},
    "glm-4": {"speed": "medium", "est_minutes_vision": 3.0, "est_minutes_text": 1.5, "quality": "good", "quality_score": 3},
    "yi-vision": {"speed": "medium", "est_minutes_vision": 3.0, "est_minutes_text": 1.5, "quality": "good", "quality_score": 3},
    "step-3.5": {"speed": "fast", "est_minutes_vision": 2.0, "est_minutes_text": 0.8, "quality": "good", "quality_score": 3},
    "mimo": {"speed": "medium", "est_minutes_vision": 3.0, "est_minutes_text": 1.5, "quality": "good", "quality_score": 3},
    "deepseek-vl": {"speed": "medium", "est_minutes_vision": 3.0, "est_minutes_text": 1.5, "quality": "good", "quality_score": 3},
    "nemotron": {"speed": "fast", "est_minutes_vision": 2.0, "est_minutes_text": 0.8, "quality": "good", "quality_score": 3},
    "phi-4": {"speed": "fast", "est_minutes_vision": 2.0, "est_minutes_text": 0.8, "quality": "good", "quality_score": 3},
    "phi-3": {"speed": "fast", "est_minutes_vision": 2.0, "est_minutes_text": 0.8, "quality": "basic", "quality_score": 2},
    "gemma-3": {"speed": "fast", "est_minutes_vision": 2.0, "est_minutes_text": 0.8, "quality": "good", "quality_score": 3},
    "llama-3.2-90b": {"speed": "medium", "est_minutes_vision": 3.0, "est_minutes_text": 1.5, "quality": "excellent", "quality_score": 4},
    "llama-3.2-11b": {"speed": "fast", "est_minutes_vision": 2.0, "est_minutes_text": 0.8, "quality": "good", "quality_score": 3},
    "llama-4": {"speed": "medium", "est_minutes_vision": 3.0, "est_minutes_text": 1.5, "quality": "excellent", "quality_score": 4},
    "grok-3": {"speed": "medium", "est_minutes_vision": 3.0, "est_minutes_text": 1.5, "quality": "excellent", "quality_score": 4},
    "grok-4": {"speed": "medium", "est_minutes_vision": 3.5, "est_minutes_text": 2.0, "quality": "excellent", "quality_score": 4},
    "grok-2": {"speed": "medium", "est_minutes_vision": 3.0, "est_minutes_text": 1.5, "quality": "good", "quality_score": 3},
    # --- Models with limited/no vision tracking ---
    "reka-edge": {"speed": "fast", "est_minutes_vision": 2.0, "est_minutes_text": 0.5, "quality": "poor", "quality_score": 1},
    "reka-core": {"speed": "medium", "est_minutes_vision": 3.0, "est_minutes_text": 1.0, "quality": "basic", "quality_score": 2},
    "gemini-2.5-flash-lite": {"speed": "fast", "est_minutes_vision": 1.0, "est_minutes_text": 0.3, "quality": "good", "quality_score": 3},
}

# Free tier models are rate-limited (~20 RPM), multiply time by 3x
_FREE_SPEED_MULTIPLIER = 3.0


def _estimate_speed(model_id: str, role: str, is_free: bool) -> dict:
    """Estimate analysis speed for a model on a 10-minute video.

    Returns {"speed": "fast"|"medium"|"slow", "est_minutes": float, "quality": str}.
    """
    mid_lower = model_id.lower()

    # Try to match against known profiles
    best_match = None
    for pattern, profile in _MODEL_SPEED_PROFILES.items():
        if pattern in mid_lower:
            best_match = profile
            break

    if best_match:
        minutes = best_match.get(f"est_minutes_{role}", best_match.get("est_minutes_text", 2.0))
        if is_free:
            minutes *= _FREE_SPEED_MULTIPLIER
        speed = best_match["speed"]
        if is_free and speed == "fast":
            speed = "medium"
        quality = best_match["quality"]
        quality_score = best_match.get("quality_score", 3)
    else:
        # Unknown model — estimate based on whether it's free
        minutes = 4.0 if is_free else 2.0
        speed = "medium"
        quality = "good"
        quality_score = 3

    # Build display string
    if minutes < 1:
        time_str = f"~{int(minutes * 60)}s"
    elif minutes < 10:
        time_str = f"~{minutes:.1f}min"
    else:
        time_str = f"~{int(minutes)}min"

    return {
        "speed": speed,
        "est_minutes": round(minutes, 1),
        "est_time_display": time_str,
        "quality": quality,
        "quality_score": quality_score,
    }


def _key_is_set(key: str) -> bool:
    return bool(key) and key not in _PLACEHOLDER_KEYS


@router.get("/providers/status")
async def provider_status(user: User = Depends(get_current_user)):
    # Per-user: read from user's settings file, fall back to global.
    us = await read_user_settings(user.id)

    def _us(key: str) -> str | bool | int:
        if key in us:
            return us[key]
        return getattr(settings, key, "")

    from backend.services.providers.openrouter_provider import PRESETS

    statuses = {}

    # Ollama (shared resource — status is instance-wide)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.OLLAMA_HOST}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            statuses["ollama"] = {
                "status": "connected",
                "models_loaded": models,
                "host": settings.OLLAMA_HOST,
            }
    except Exception as e:
        statuses["ollama"] = {"status": "offline", "error": str(e)}

    # OpenRouter — per-user API key + model selections
    or_key = _us("OPENROUTER_API_KEY")
    if _key_is_set(or_key):
        preset_name = _us("OPENROUTER_PRESET") or "free"
        preset = PRESETS.get(preset_name, PRESETS["free"])
        vision_model = _us("OPENROUTER_VISION_MODEL") or preset["vision"]
        text_model = _us("OPENROUTER_TEXT_MODEL") or preset["text"]
        summary_model = _us("OPENROUTER_SUMMARY_MODEL") or text_model
        statuses["openrouter"] = {
            "status": "configured",
            "preset": preset_name,
            "vision_model": vision_model,
            "summary_model": summary_model,
            "text_model": text_model,
        }
    else:
        statuses["openrouter"] = {"status": "not_configured"}

    # Anthropic — per-user key
    if _key_is_set(_us("ANTHROPIC_API_KEY")):
        statuses["anthropic"] = {"status": "configured"}
    else:
        statuses["anthropic"] = {"status": "not_configured"}

    # Gemini — per-user key
    if _key_is_set(_us("GEMINI_API_KEY")):
        statuses["gemini"] = {"status": "configured"}
    else:
        statuses["gemini"] = {"status": "not_configured"}

    # Groq — per-user key
    if _key_is_set(_us("GROQ_API_KEY")):
        statuses["groq"] = {"status": "configured"}
    else:
        statuses["groq"] = {"status": "not_configured"}

    # HuggingFace — per-user token
    hf_token = _us("HF_AUTH_TOKEN")
    if hf_token and str(hf_token).strip():
        statuses["huggingface"] = {"status": "configured", "message": "Token set"}
    else:
        statuses["huggingface"] = {"status": "not_configured", "message": "No HF token"}

    # Determine the active provider and models based on fallback chain
    chain_raw = _us("AI_FALLBACK_CHAIN")
    chain = settings.active_provider_chain  # default
    if chain_raw and isinstance(chain_raw, str):
        chain = [p.strip() for p in chain_raw.split(",") if p.strip()]
    active_provider = None
    active_vision_model = None
    active_text_model = None
    active_summary_model = None
    for name in chain:
        info = statuses.get(name, {})
        st = info.get("status", "not_configured")
        if st in ("connected", "configured"):
            active_provider = name
            if name == "openrouter":
                active_vision_model = info.get("vision_model", "")
                active_summary_model = info.get("summary_model", "")
                active_text_model = info.get("text_model", "")
            elif name == "ollama":
                active_vision_model = _us("OLLAMA_VISION_MODEL") or settings.OLLAMA_VISION_MODEL
                active_text_model = _us("OLLAMA_TEXT_MODEL") or settings.OLLAMA_TEXT_MODEL
                active_summary_model = active_text_model
            elif name == "gemini":
                active_vision_model = "gemini-2.5-flash"
                active_text_model = "gemini-2.5-flash"
                active_summary_model = "gemini-2.5-flash"
            elif name == "anthropic":
                active_vision_model = "claude-sonnet-4"
                active_text_model = "claude-sonnet-4"
                active_summary_model = "claude-sonnet-4"
            elif name == "groq":
                active_vision_model = ""
                active_text_model = "llama-3.1-8b-instant"
                active_summary_model = "llama-3.1-8b-instant"
            break

    whisper_model = _us("WHISPER_MODEL") or settings.WHISPER_MODEL
    whisper_beam = _us("WHISPER_BEAM_SIZE")
    if not isinstance(whisper_beam, int):
        whisper_beam = settings.WHISPER_BEAM_SIZE
    whisper_vad = _us("WHISPER_VAD_FILTER")
    if not isinstance(whisper_vad, bool):
        whisper_vad = settings.WHISPER_VAD_FILTER

    statuses["_active"] = {
        "provider": active_provider or "none",
        "transcript_model": whisper_model,
        "whisper_beam_size": whisper_beam,
        "whisper_vad_filter": whisper_vad,
        "vision_model": active_vision_model or "",
        "summary_model": active_summary_model or "",
        "text_model": active_text_model or "",
        "preset": (_us("OPENROUTER_PRESET") or settings.OPENROUTER_PRESET) if active_provider == "openrouter" else "",
        "fallback_chain": chain,
        "ollama_enabled": "ollama" in chain,
    }

    return statuses


@router.post("/providers/test/{provider_name}")
async def test_provider(provider_name: str, user: User = Depends(get_current_user)):
    """Live-test a provider by making a real API call and returning detailed status."""

    if provider_name == "openrouter":
        return await _test_openrouter()
    elif provider_name == "ollama":
        return await _test_ollama()
    elif provider_name == "anthropic":
        return await _test_anthropic()
    elif provider_name == "gemini":
        return await _test_gemini()
    elif provider_name == "groq":
        return await _test_groq()
    elif provider_name == "huggingface":
        return await _test_huggingface()
    else:
        return {"status": "error", "message": f"Unknown provider: {provider_name}"}


async def _test_openrouter():
    key = settings.OPENROUTER_API_KEY
    if not _key_is_set(key):
        return {
            "status": "not_configured",
            "message": "OPENROUTER_API_KEY is not set. Add it to your .env file.",
            "help": "Get a free key at https://openrouter.ai/keys",
        }

    # Step 1: Validate key by fetching account info
    headers = {"Authorization": f"Bearer {key}"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Check key validity via auth/key endpoint
            auth_resp = await client.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers=headers,
            )
            if auth_resp.status_code == 401:
                return {
                    "status": "invalid_key",
                    "message": "API key is invalid or expired. Check your key at openrouter.ai/keys.",
                }
            if auth_resp.status_code == 403:
                return {
                    "status": "invalid_key",
                    "message": "API key is forbidden. It may have been revoked.",
                }

            key_info = {}
            if auth_resp.status_code == 200:
                key_data = auth_resp.json().get("data", {})
                key_info = {
                    "label": key_data.get("label", ""),
                    "usage_usd": key_data.get("usage", 0),
                    "limit_usd": key_data.get("limit"),
                    "is_free_tier": key_data.get("is_free_tier", False),
                    "rate_limit_rpm": key_data.get("rate_limit", {}).get("requests", None),
                }

            # Step 2: Quick model ping — try multiple models to handle unavailable ones
            from backend.services.providers.openrouter_provider import PRESETS
            preset_name = settings.OPENROUTER_PRESET
            preset = PRESETS.get(preset_name, PRESETS["free"])

            # Build a list of models to try: current text model from settings, then fallbacks
            effective_text = settings.OPENROUTER_TEXT_MODEL or preset["text"]
            test_models = [effective_text]
            for fb in (preset.get("text_fallbacks") or []):
                if fb not in test_models:
                    test_models.append(fb)
            if preset.get("text_fallback") and preset["text_fallback"] not in test_models:
                test_models.append(preset["text_fallback"])

            model_ok = False
            model_error = ""
            tested_model = test_models[0]
            for test_model in test_models:
                tested_model = test_model
                try:
                    test_resp = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={**headers, "Content-Type": "application/json"},
                        json={
                            "model": test_model,
                            "messages": [{"role": "user", "content": "Say OK"}],
                            "max_tokens": 5,
                        },
                        timeout=20.0,
                    )
                    if test_resp.status_code == 200:
                        model_ok = True
                        break
                    else:
                        try:
                            err = test_resp.json()
                            model_error = err.get("error", {}).get("message", test_resp.text[:200])
                        except Exception:
                            model_error = test_resp.text[:200]
                        logger.info(f"Model test failed for {test_model}: {model_error}")
                except httpx.TimeoutException:
                    model_error = f"Timeout testing {test_model}"
                    logger.info(model_error)

            effective_vision = settings.OPENROUTER_VISION_MODEL or preset["vision"]
            effective_summary = settings.OPENROUTER_SUMMARY_MODEL or effective_text
            return {
                "status": "connected" if model_ok else "key_valid_model_error",
                "message": "API key validated and model responded successfully." if model_ok
                    else f"Key is valid but model test failed: {model_error}. Try refreshing models to find available ones.",
                "preset": preset_name,
                "vision_model": effective_vision,
                "summary_model": effective_summary,
                "text_model": effective_text,
                "tested_model": tested_model,
                "model_test_passed": model_ok,
                **key_info,
            }

    except httpx.TimeoutException:
        return {"status": "timeout", "message": "Connection to OpenRouter timed out. Try again."}
    except Exception as e:
        logger.warning(f"OpenRouter test failed: {e}")
        return {"status": "error", "message": f"Connection failed: {str(e)[:200]}"}


async def _test_ollama():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.OLLAMA_HOST}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]

            has_vision = any(
                "moondream" in m or "llava" in m or "bakllava" in m
                for m in models
            )
            return {
                "status": "connected",
                "message": f"Ollama connected with {len(models)} model(s) loaded.",
                "models": models,
                "has_vision_model": has_vision,
                "host": settings.OLLAMA_HOST,
            }
    except Exception as e:
        return {
            "status": "offline",
            "message": f"Cannot reach Ollama at {settings.OLLAMA_HOST}: {str(e)[:200]}",
        }


async def _test_anthropic():
    key = settings.ANTHROPIC_API_KEY
    if not _key_is_set(key):
        return {"status": "not_configured", "message": "ANTHROPIC_API_KEY is not set."}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 5,
                    "messages": [{"role": "user", "content": "Say OK"}],
                },
            )
            if resp.status_code == 200:
                return {"status": "connected", "message": "Anthropic API key is valid."}
            elif resp.status_code == 401:
                return {"status": "invalid_key", "message": "API key is invalid."}
            else:
                return {"status": "error", "message": f"Anthropic responded with status {resp.status_code}."}
    except Exception as e:
        return {"status": "error", "message": f"Connection failed: {str(e)[:200]}"}


async def _test_gemini():
    key = settings.GEMINI_API_KEY
    if not _key_is_set(key):
        return {"status": "not_configured", "message": "GEMINI_API_KEY is not set."}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}",
                json={"contents": [{"parts": [{"text": "Say OK"}]}]},
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code == 200:
                return {"status": "connected", "message": "Gemini API key is valid."}
            elif resp.status_code == 400 and "API_KEY_INVALID" in resp.text:
                return {"status": "invalid_key", "message": "API key is invalid."}
            else:
                return {"status": "error", "message": f"Gemini responded with status {resp.status_code}."}
    except Exception as e:
        return {"status": "error", "message": f"Connection failed: {str(e)[:200]}"}


async def _test_groq():
    key = settings.GROQ_API_KEY
    if not _key_is_set(key):
        return {"status": "not_configured", "message": "GROQ_API_KEY is not set."}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": "Say OK"}],
                    "max_tokens": 5,
                },
            )
            if resp.status_code == 200:
                return {"status": "connected", "message": "Groq API key is valid."}
            elif resp.status_code == 401:
                return {"status": "invalid_key", "message": "API key is invalid."}
            else:
                return {"status": "error", "message": f"Groq responded with status {resp.status_code}."}
    except Exception as e:
        return {"status": "error", "message": f"Connection failed: {str(e)[:200]}"}


async def _test_huggingface():
    token = settings.HF_AUTH_TOKEN
    if not token or not token.strip():
        return {
            "status": "not_configured",
            "message": "HF_AUTH_TOKEN is not set. Add your HuggingFace access token to enable pyannote speaker diarization.",
            "help": "Get a free token at https://huggingface.co/settings/tokens",
        }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {token.strip()}"},
            )
            if resp.status_code == 200:
                username = resp.json().get("name", "unknown")
                model_resp = await client.get(
                    "https://huggingface.co/api/models/pyannote/speaker-diarization-3.1",
                    headers={"Authorization": f"Bearer {token.strip()}"},
                )
                if model_resp.status_code == 200:
                    return {
                        "status": "connected",
                        "message": f"Connected as '{username}'. pyannote model access confirmed — neural speaker diarization is enabled.",
                    }
                elif model_resp.status_code == 403:
                    return {
                        "status": "connected",
                        "message": f"Connected as '{username}', but you must accept the pyannote model terms at https://huggingface.co/pyannote/speaker-diarization-3.1 and click 'Agree and access repository'.",
                    }
                else:
                    return {
                        "status": "connected",
                        "message": f"Connected as '{username}'. Could not verify pyannote model access (HTTP {model_resp.status_code}).",
                    }
            elif resp.status_code == 401:
                return {"status": "invalid_key", "message": "Token is invalid or expired."}
            else:
                return {"status": "error", "message": f"HuggingFace API returned HTTP {resp.status_code}."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to reach HuggingFace API: {e}"}


class SaveKeyRequest(BaseModel):
    provider: str
    key: str


def _invalidate_status_cache():
    global _status_cache, _status_cache_ts
    _status_cache = {}
    _status_cache_ts = 0


@router.post("/providers/key")
async def save_provider_key(req: SaveKeyRequest, user: User = Depends(get_current_user)):
    """Save an API key to the calling user's private settings file."""
    env_var = _PROVIDER_KEY_ENV.get(req.provider)
    if not env_var:
        return {"status": "error", "message": f"Unknown provider: {req.provider}"}

    key_val = req.key.strip()
    if not key_val:
        return {"status": "error", "message": "Key cannot be empty"}

    # Persist to per-user settings file
    await write_user_settings(user.id, {env_var: key_val})
    _invalidate_status_cache()
    logger.info(
        "User %s saved %s key (%d chars)",
        user.username, req.provider, len(key_val),
    )

    return {"status": "saved", "provider": req.provider}


class SavePresetRequest(BaseModel):
    preset: str
    vision_model: str = ""
    text_model: str = ""
    summary_model: str = ""


@router.post("/providers/preset")
async def save_preset(req: SavePresetRequest, user: User = Depends(get_current_user)):
    """Save the active preset (and optional custom models) to the user's settings."""
    from backend.services.providers.openrouter_provider import PRESETS as _PRESETS

    preset_dict = _PRESETS.get(req.preset, _PRESETS["free"])
    vision_model = req.vision_model or preset_dict["vision"]
    text_model = req.text_model or preset_dict["text"]
    summary_model = req.summary_model or preset_dict.get("summary", text_model)

    await write_user_settings(user.id, {
        "OPENROUTER_PRESET": req.preset,
        "OPENROUTER_VISION_MODEL": vision_model,
        "OPENROUTER_TEXT_MODEL": text_model,
        "OPENROUTER_SUMMARY_MODEL": summary_model,
    })
    _invalidate_status_cache()
    logger.info(
        "User %s saved preset: %s (vision=%s, text=%s, summary=%s)",
        user.username, req.preset, vision_model, text_model, summary_model,
    )
    return {"status": "saved", "preset": req.preset}


class ToggleOllamaRequest(BaseModel):
    enabled: bool


def _pre_download_whisper_model(model_name: str):
    """Pre-download a Whisper model in a background thread.

    faster-whisper downloads models from HuggingFace on first use. Without
    pre-downloading, the first transcription attempt triggers a download
    inside the subprocess, which can timeout and fail. This ensures the
    model is cached locally before the user starts a video analysis.
    """
    import threading

    def _do_download():
        try:
            logger.info("Whisper pre-download: downloading '%s' from HuggingFace...", model_name)
            # Import and instantiate on CPU with int8 — minimal resources,
            # just triggers the HuggingFace download to cache
            from faster_whisper import WhisperModel
            m = WhisperModel(model_name, device="cpu", compute_type="int8")
            del m
            import gc
            gc.collect()
            logger.info("Whisper pre-download: '%s' is now cached locally", model_name)
        except Exception as e:
            logger.warning("Whisper pre-download failed for '%s': %s", model_name, e)

    t = threading.Thread(target=_do_download, daemon=True, name=f"whisper-download-{model_name}")
    t.start()


def _pull_ollama_models_background(models: list[str] | None = None):
    """Pull Ollama models in a background thread.

    If no models are specified, pulls the configured vision, text, and
    translation models.  This is called when Ollama is toggled on or when
    models are saved, so the models are ready by the time the user tries
    to use them.
    """
    if models is None:
        models = []
        for m in (settings.OLLAMA_VISION_MODEL, settings.OLLAMA_TEXT_MODEL,
                  settings.OLLAMA_TRANSLATION_MODEL):
            if m and m not in models:
                models.append(m)

    if not models:
        return

    def _do_pull():
        import httpx as _httpx
        host = settings.OLLAMA_HOST
        for model in models:
            try:
                logger.info("Background pull: requesting %s from Ollama...", model)
                resp = _httpx.post(
                    f"{host}/api/pull",
                    json={"name": model},
                    timeout=_httpx.Timeout(connect=10, read=1800, write=10, pool=10),
                )
                if resp.status_code == 200:
                    logger.info("Background pull: %s ready", model)
                else:
                    logger.warning("Background pull: %s returned %d", model, resp.status_code)
            except Exception as exc:
                logger.warning("Background pull: %s failed (%s)", model, exc)

    threading.Thread(target=_do_pull, daemon=True, name="ollama-bg-pull").start()


@router.post("/providers/ollama/toggle")
async def toggle_ollama(req: ToggleOllamaRequest, user: User = Depends(get_current_user)):
    """Add or remove Ollama from the user's fallback chain."""
    us = await read_user_settings(user.id)
    chain_raw = us.get("AI_FALLBACK_CHAIN") or settings.AI_FALLBACK_CHAIN
    chain = [p.strip() for p in chain_raw.split(",") if p.strip()]
    if req.enabled:
        if "ollama" not in chain:
            chain.insert(0, "ollama")
    else:
        chain = [p for p in chain if p != "ollama"]
    await write_user_settings(user.id, {"AI_FALLBACK_CHAIN": ",".join(chain)})
    _invalidate_status_cache()

    # When Ollama is enabled, pull configured models in the background
    # so they're ready when the user needs them.  The startup pull only
    # fires if Ollama was already in the chain at boot time.
    if req.enabled:
        _pull_ollama_models_background()

    return {
        "status": "saved",
        "ollama_enabled": "ollama" in chain,
        "chain": chain,
    }


def _find_env_file() -> str | None:
    """Find or create the .env file for persisting settings.

    Checks project root, cwd, and common Docker paths. If no .env file
    exists, creates one at the project root so API keys and settings
    can be written as a backup alongside user_settings.json.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates = [
        os.path.join(project_root, ".env"),
        ".env",
        "/app/.env",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    # No .env found — create one at project root so subsequent writes succeed
    env_path = candidates[0]
    try:
        with open(env_path, "w") as f:
            f.write("# ClipAI settings (auto-created)\n")
        logger.info(f"Created new .env file at {env_path}")
        return env_path
    except Exception as e:
        logger.warning(f"Could not create .env at {env_path}: {e}")
        return None


def _upsert_env_var(env_path: str, var_name: str, value: str):
    """Update or add an env var in a .env file."""
    try:
        with open(env_path, "r") as f:
            lines = f.readlines()

        pattern = re.compile(rf"^{re.escape(var_name)}\s*=")
        found = False
        new_lines = []
        for line in lines:
            if pattern.match(line):
                new_lines.append(f"{var_name}={value}\n")
                found = True
            else:
                new_lines.append(line)
        if not found:
            new_lines.append(f"{var_name}={value}\n")

        with open(env_path, "w") as f:
            f.writelines(new_lines)
    except Exception as e:
        logger.warning(f"Failed to update .env: {e}")


@router.get("/providers/models/recommended")
async def recommended_models(user: User = Depends(get_current_user)):
    """Dynamically discover vision-capable models from OpenRouter and build recommendations."""
    us = await read_user_settings(user.id)
    or_key = us.get("OPENROUTER_API_KEY") or settings.OPENROUTER_API_KEY
    if not _key_is_set(or_key):
        return {"models": [], "error": "OpenRouter API key not configured"}

    all_models = await _fetch_openrouter_models()
    if all_models is None:
        return {"models": [], "error": "Failed to fetch models from OpenRouter"}

    by_id = {m["id"]: m for m in all_models}

    # Dynamically discover free vision models
    free_vision = []
    free_text = []
    for m in all_models:
        mid = m.get("id", "")
        arch = m.get("architecture", {})
        modality = arch.get("modality", "")
        is_free = ":free" in mid or _is_zero_cost(m)
        has_vision = "image" in modality

        if is_free and has_vision:
            free_vision.append(m)
        if is_free:
            free_text.append(m)

    # Sort free vision by context length (larger = better, likely better model)
    free_vision.sort(key=lambda m: m.get("context_length", 0), reverse=True)
    free_text.sort(key=lambda m: m.get("context_length", 0), reverse=True)

    # Build dynamic free tier from discovered models
    best_free_vision = free_vision[0]["id"] if free_vision else None
    second_free_vision = free_vision[1]["id"] if len(free_vision) > 1 else None
    best_free_text = free_text[0]["id"] if free_text else None

    # Build recommendations: dynamic free tiers + static paid tiers
    curated = []

    # Add dynamic free vision combos
    if best_free_vision and best_free_text:
        curated.append({
            "id": "free-best",
            "label": _short_name(by_id.get(best_free_vision, {})) + " (Free)",
            "desc": "Best available free vision model",
            "vision": best_free_vision,
            "summary": best_free_text,
            "text": best_free_text,
            "tier": "free",
        })
    if second_free_vision and best_free_text:
        curated.append({
            "id": "free-alt",
            "label": _short_name(by_id.get(second_free_vision, {})) + " (Free)",
            "desc": "Alternative free vision model",
            "vision": second_free_vision,
            "summary": best_free_text,
            "text": best_free_text,
            "tier": "free",
        })

    # Add more free vision options (up to 4 total free combos)
    for i, fv in enumerate(free_vision[2:6], start=3):
        curated.append({
            "id": f"free-{i}",
            "label": _short_name(fv) + " (Free)",
            "desc": f"Free vision model #{i}",
            "vision": fv["id"],
            "summary": best_free_text or fv["id"],
            "text": best_free_text or fv["id"],
            "tier": "free",
        })

    # Static paid tiers (these are stable model IDs unlikely to vanish)
    paid_combos = [
        {
            "id": "efficient",
            "label": "Gemini 2.5 Flash",
            "desc": "Fastest paid model — great value",
            "vision": "google/gemini-2.5-flash",
            "summary": "google/gemini-2.5-flash",
            "text": "google/gemini-2.5-flash",
            "tier": "efficient",
        },
        {
            "id": "balanced",
            "label": "Flash + Gemini Pro",
            "desc": "Fast vision & summary, smart clip detection",
            "vision": "google/gemini-2.5-flash",
            "summary": "google/gemini-2.5-flash",
            "text": "google/gemini-2.5-pro",
            "tier": "balanced",
        },
        {
            "id": "premium",
            "label": "Gemini Pro + Claude",
            "desc": "Best quality, highest accuracy",
            "vision": "google/gemini-2.5-pro",
            "summary": "google/gemini-2.5-flash",
            "text": "anthropic/claude-sonnet-4",
            "tier": "premium",
        },
        {
            "id": "gemini-pro-full",
            "label": "Gemini 2.5 Pro (Full)",
            "desc": "Gemini Pro for everything",
            "vision": "google/gemini-2.5-pro",
            "summary": "google/gemini-2.5-pro",
            "text": "google/gemini-2.5-pro",
            "tier": "premium",
        },
    ]
    curated.extend(paid_combos)

    result = []
    for combo in curated:
        v_model = by_id.get(combo["vision"])
        s_model = by_id.get(combo.get("summary", combo["text"]))
        t_model = by_id.get(combo["text"])

        v_cost = _estimate_cost(v_model, "vision") if v_model else 0
        s_cost = _estimate_cost(s_model, "text") if s_model else 0
        t_cost = _estimate_cost(t_model, "text") if t_model else 0
        total_cost = v_cost + s_cost + t_cost

        summary_id = combo.get("summary", combo["text"])
        result.append({
            "id": combo["id"],
            "label": combo["label"],
            "desc": combo["desc"],
            "tier": combo["tier"],
            "vision_model": combo["vision"],
            "summary_model": summary_id,
            "text_model": combo["text"],
            "vision_model_name": v_model.get("name", combo["vision"]) if v_model else combo["vision"],
            "summary_model_name": s_model.get("name", summary_id) if s_model else summary_id,
            "text_model_name": t_model.get("name", combo["text"]) if t_model else combo["text"],
            "cost_per_10min": round(total_cost, 4),
            "cost_per_10min_display": f"${total_cost:.4f}" if total_cost > 0 else "FREE",
            "available": v_model is not None and t_model is not None,
        })

    # Count stats
    total_free_vision = len(free_vision)
    total_free_text = len(free_text)

    return {
        "models": result,
        "free_vision_count": total_free_vision,
        "free_text_count": total_free_text,
    }


def _short_name(model_data: dict) -> str:
    """Extract a short display name from a model's full name."""
    name = model_data.get("name", model_data.get("id", "Unknown"))
    # Remove common suffixes/prefixes for brevity
    for remove in ["(free)", "(Free)", ":free"]:
        name = name.replace(remove, "").strip()
    return name


def _is_zero_cost(model_data: dict) -> bool:
    """Check if a model has zero pricing."""
    pricing = model_data.get("pricing", {})
    try:
        prompt = float(pricing.get("prompt", "1"))
        completion = float(pricing.get("completion", "1"))
        return prompt == 0 and completion == 0
    except (ValueError, TypeError):
        return False


async def _fetch_openrouter_models() -> list | None:
    """Fetch model list from OpenRouter, using cache if fresh."""
    # Check cache first
    if os.path.exists(MODEL_CACHE_PATH):
        try:
            with open(MODEL_CACHE_PATH, "r") as f:
                cache = json.load(f)
            if time.time() - cache.get("timestamp", 0) < MODEL_CACHE_TTL:
                return cache.get("raw_models", [])
        except Exception:
            pass

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"},
            )
            resp.raise_for_status()
            models = resp.json().get("data", [])
    except Exception as e:
        logger.warning(f"Failed to fetch OpenRouter models: {e}")
        return None

    # Save to cache (include raw models for reuse)
    _save_model_cache(models)
    return models


def _save_model_cache(models: list):
    """Save raw model list to cache file."""
    vision_models = []
    text_models = []
    for m in models:
        model_id = m.get("id", "")
        top_provider = m.get("top_provider", {}) or {}
        model_info = {
            "id": model_id,
            "name": m.get("name", model_id),
            "pricing": m.get("pricing", {}),
            "context_length": m.get("context_length", 0),
            "max_completion_tokens": top_provider.get("max_completion_tokens", 0),
        }
        architecture = m.get("architecture", {})
        modality = architecture.get("modality", "")
        input_modalities = architecture.get("input_modalities", [])
        has_vision = "image" in str(modality).lower() or "image" in [
            str(x).lower() for x in input_modalities
        ]
        if has_vision:
            vision_models.append(model_info)
        text_models.append(model_info)

    cache_data = {
        "timestamp": time.time(),
        "raw_models": models,
        "data": {
            "vision_models": vision_models,
            "text_models": text_models,
            "cached": True,
        },
    }
    os.makedirs(os.path.dirname(MODEL_CACHE_PATH), exist_ok=True)
    try:
        with open(MODEL_CACHE_PATH, "w") as f:
            json.dump(cache_data, f)
    except Exception as e:
        logger.warning(f"Failed to save model cache: {e}")


def _estimate_cost(model_data: dict, role: str) -> float:
    """Estimate cost for a 10-minute video based on model pricing."""
    pricing = model_data.get("pricing", {})
    try:
        prompt_price = float(pricing.get("prompt", "0"))  # per token
        completion_price = float(pricing.get("completion", "0"))  # per token
    except (ValueError, TypeError):
        return 0.0

    if role == "vision":
        return (prompt_price * _VISION_INPUT_TOKENS_10MIN +
                completion_price * _VISION_OUTPUT_TOKENS_10MIN)
    else:
        return (prompt_price * _TEXT_INPUT_TOKENS_10MIN +
                completion_price * _TEXT_OUTPUT_TOKENS_10MIN)


@router.get("/providers/models")
async def list_models(user: User = Depends(get_current_user)):
    """Fetch OpenRouter model list, cached for 24h."""
    # Check cache
    if os.path.exists(MODEL_CACHE_PATH):
        try:
            with open(MODEL_CACHE_PATH, "r") as f:
                cache = json.load(f)
            if time.time() - cache.get("timestamp", 0) < MODEL_CACHE_TTL:
                cached_data = cache.get("data", {})
                if cached_data:
                    return cached_data
        except Exception:
            pass

    # Fetch from OpenRouter
    if not _key_is_set(settings.OPENROUTER_API_KEY):
        return {"vision_models": [], "text_models": [], "cached": False}

    models = await _fetch_openrouter_models()
    if models is None:
        return {"vision_models": [], "text_models": [], "cached": False}

    # Return from the cache that _fetch_openrouter_models just wrote
    try:
        with open(MODEL_CACHE_PATH, "r") as f:
            cache = json.load(f)
        return cache.get("data", {"vision_models": [], "text_models": [], "cached": True})
    except Exception:
        return {"vision_models": [], "text_models": [], "cached": False}


@router.post("/providers/models/refresh")
async def refresh_models(user: User = Depends(get_current_user)):
    """Force-refresh the model list from OpenRouter (clears cache)."""
    # Delete cache to force re-fetch
    if os.path.exists(MODEL_CACHE_PATH):
        try:
            os.remove(MODEL_CACHE_PATH)
        except Exception:
            pass

    if not _key_is_set(settings.OPENROUTER_API_KEY):
        return {"status": "error", "message": "OpenRouter API key not configured"}

    models = await _fetch_openrouter_models()
    if models is None:
        return {"status": "error", "message": "Failed to fetch models from OpenRouter"}

    # Count vision models
    vision_count = 0
    free_vision_count = 0
    for m in models:
        arch = m.get("architecture", {})
        modality = arch.get("modality", "")
        if "image" in modality:
            vision_count += 1
            if ":free" in m.get("id", "") or _is_zero_cost(m):
                free_vision_count += 1

    return {
        "status": "refreshed",
        "total_models": len(models),
        "vision_models": vision_count,
        "free_vision_models": free_vision_count,
        "message": f"Loaded {len(models)} models ({vision_count} with vision, {free_vision_count} free vision)",
    }


# ── Per-Task Model Selection ──────────────────────────────────────

# Whisper model options (always available locally, free)
_WHISPER_MODELS = [
    {"id": "tiny", "name": "Whisper Tiny", "provider": "local", "desc": "Fastest, least accurate (~39M params)", "cost_per_hour": 0, "is_free": True, "quality_score": 1, "quality": "poor"},
    {"id": "base", "name": "Whisper Base", "provider": "local", "desc": "Fast, low accuracy (~74M params)", "cost_per_hour": 0, "is_free": True, "quality_score": 2, "quality": "basic"},
    {"id": "small", "name": "Whisper Small", "provider": "local", "desc": "Good accuracy/speed balance (~244M params, default)", "cost_per_hour": 0, "is_free": True, "quality_score": 3, "quality": "good"},
    {"id": "medium", "name": "Whisper Medium", "provider": "local", "desc": "High accuracy, slower (~769M params)", "cost_per_hour": 0, "is_free": True, "quality_score": 4, "quality": "excellent"},
    {"id": "large-v3", "name": "Whisper Large V3", "provider": "local", "desc": "Best accuracy, needs GPU (~1.5B params)", "cost_per_hour": 0, "is_free": True, "quality_score": 5, "quality": "best"},
    {"id": "large-v3-turbo", "name": "Whisper Large V3 Turbo", "provider": "local", "desc": "Near large-v3 accuracy, 40% faster (~809M params)", "cost_per_hour": 0, "is_free": True, "quality_score": 5, "quality": "best"},
    {"id": "distil-large-v3", "name": "Whisper Distil Large V3", "provider": "local", "desc": "Distilled large-v3, 6x faster, English-optimized (~756M params)", "cost_per_hour": 0, "is_free": True, "quality_score": 4, "quality": "excellent"},
]

# Known models for direct providers (when user has their API key)
# "created" = approximate release Unix timestamp so sorting by recency works
_ANTHROPIC_MODELS = [
    {"id": "anthropic/claude-sonnet-4", "name": "Claude Sonnet 4", "provider": "anthropic", "context_length": 200000, "vision": True, "created": 1747872000, "quality_score": 4, "quality": "excellent"},
    {"id": "anthropic/claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5", "provider": "anthropic", "context_length": 200000, "vision": True, "created": 1727740800, "quality_score": 3, "quality": "good"},
]

_GEMINI_MODELS = [
    {"id": "google/gemini-2.5-flash", "name": "Gemini 2.5 Flash", "provider": "gemini", "context_length": 1000000, "vision": True, "created": 1744502400, "quality_score": 3, "quality": "good"},
    {"id": "google/gemini-2.5-pro", "name": "Gemini 2.5 Pro", "provider": "gemini", "context_length": 1000000, "vision": True, "created": 1742860800, "quality_score": 4, "quality": "excellent"},
    {"id": "google/gemini-2.0-flash", "name": "Gemini 2.0 Flash", "provider": "gemini", "context_length": 1000000, "vision": True, "created": 1738886400, "quality_score": 3, "quality": "good"},
]


def _cost_per_hour(model_data: dict, role: str) -> float:
    """Estimate cost for a 1-hour video based on model pricing."""
    return _estimate_cost(model_data, role) * 6  # 6 × 10-min segments


# ── Vision model compatibility for subject tracking ──
# Models need sufficient context for at least 1 image + prompt + output.
_MIN_VISION_CONTEXT = 16000  # Absolute minimum context length

# Models known to be incompatible with subject tracking despite having vision.
_VISION_BLOCKLIST_PATTERNS = [
    "reka/reka-edge",      # 16K context, ~5800 tok/img, can't fit 2 images, 94% default to center
    "reka-edge",           # Catch any variant
    "firellava",           # Poor JSON compliance, no spatial reasoning
    "llava:7b",            # Too small for reliable subject_x
    "nanollava",           # Too small for structured output
]

# ── Subject tracking quality scores ──
# Scoring criteria: can the model output structured JSON with a reliable
# subject_x (0-100) horizontal position? Most 7B+ vision models can.
#   5 = best:  Excellent spatial reasoning + JSON + large context + fast
#   4 = excellent: Strong spatial + reliable JSON + good context
#   3 = good:  Solid spatial awareness + JSON works + adequate context
#   2 = basic: Can estimate positions + JSON mostly works
#   1 = minimal: Marginal capability
_VISION_TRACKING_SCORES: dict[str, int] = {
    # ── Google ──
    "gemini-2.5-pro": 5, "gemini-2.5-flash": 5, "gemini-2.0-flash": 4,
    "gemini-2.5-flash-lite": 3, "gemini-flash": 4, "gemini-3": 5,
    # ── Anthropic ──
    "claude-sonnet-4": 5, "claude-opus": 5, "claude-sonnet": 4, "claude-haiku": 3,
    # ── OpenAI ──
    "gpt-4o": 4, "gpt-4o-mini": 3, "gpt-4-turbo": 4, "o1": 4, "o3": 5, "o4-mini": 4,
    # ── Qwen VL ──
    "qwen2.5-vl-72b": 4, "qwen2.5-vl-32b": 4, "qwen2.5-vl-7b": 3, "qwen2.5-vl-3b": 2,
    "qwen3-vl-32b": 4, "qwen3-vl-8b": 3, "qwen-vl-max": 4, "qwen-vl-plus": 3,
    "qwen-vl": 3, "qwq": 3,
    # ── Mistral / Pixtral ──
    "pixtral-large": 4, "pixtral-12b": 3, "pixtral": 3,
    "mistral-large-3": 4, "mistral-small-3.1": 3, "mistral-small-3": 3, "mistral-medium": 3,
    # ── Meta Llama ──
    "llama-4-maverick": 4, "llama-4-scout": 4,
    "llama-3.2-90b": 4, "llama-3.2-11b": 3, "llama-3.2-3b": 2,
    # ── Google Gemma ──
    "gemma-3-27b": 3, "gemma-3-12b": 3, "gemma-3-4b": 2, "gemma-2": 2,
    # ── DeepSeek ──
    "deepseek-r1": 3, "deepseek-v3": 3, "deepseek-vl2": 3, "deepseek": 3,
    # ── Moonshot / Kimi ──
    "kimi-vl": 3, "kimi-k2.5": 4, "kimi-k2": 3, "moonshot": 3,
    # ── NVIDIA ──
    "nemotron": 3, "nemotron-nano-2-vl": 3, "llama-3.1-nemotron": 3,
    # ── xAI Grok ──
    "grok-3": 4, "grok-2": 3, "grok-4": 4, "grok": 3,
    # ── Zhipu / GLM ──
    "glm-4v": 3, "glm-4.5": 3, "chatglm": 2,
    # ── InternLM / InternVL ──
    "internvl": 3, "internlm": 3,
    # ── Yi (01.AI) ──
    "yi-vision": 3, "yi-vl": 3,
    # ── MiniCPM ──
    "minicpm-v": 3, "minicpm": 2,
    # ── StepFun ──
    "step-3.5": 3,
    # ── Reka (only Core; Edge is blocklisted) ──
    "reka-core": 2,
    # ── Moondream ──
    "moondream": 2,
    # ── Cohere ──
    "command-r-plus": 3, "command-r": 2,
    # ── MiMo ──
    "mimo": 3,
    # ── Microsoft Phi ──
    "phi-4": 3, "phi-3.5-vision": 2, "phi-3-vision": 2,
}


def _vision_tracking_compat(model_id: str, context_length: int) -> tuple[bool, int]:
    """Check if a vision model is compatible with subject tracking.

    Returns (is_compatible, tracking_quality_score 0-5).
    Uses longest pattern match for specificity (e.g. "qwen2.5-vl-72b" wins over "qwen-vl").
    """
    mid_lower = model_id.lower()

    for pattern in _VISION_BLOCKLIST_PATTERNS:
        if pattern in mid_lower:
            return False, 0

    # Skip context check for Ollama local models (context_length=0 means unknown)
    if context_length > 0 and context_length < _MIN_VISION_CONTEXT:
        return False, 0

    # Longest match wins for specificity
    best_score = None
    best_len = 0
    for pattern, score in _VISION_TRACKING_SCORES.items():
        if pattern in mid_lower and len(pattern) > best_len:
            best_score = score
            best_len = len(pattern)
    if best_score is not None:
        return True, best_score

    # Context-based fallback for unknown models — most modern vision models
    # with decent context can do basic spatial estimation.
    if context_length == 0:
        return True, 2  # Ollama local, give benefit of doubt
    if context_length >= 200000:
        return True, 4  # 200K+ → likely capable frontier model
    if context_length >= 32000:
        return True, 3  # 32K+ → modern model, should work well
    if context_length >= 16000:
        return True, 2  # 16K → tight but workable
    return True, 1  # Below 16K but passed blocklist — marginal


@router.get("/providers/models/available")
async def available_models(user: User = Depends(get_current_user)):
    """Return all available models grouped by task (transcript, vision, text).
    Each list is sorted: free/cheapest first. Filtered by capability."""

    transcript = list(_WHISPER_MODELS)
    vision = []
    text = []

    # Always add the OpenRouter free auto-router at the top
    if _key_is_set(settings.OPENROUTER_API_KEY):
        _auto_speed = _estimate_speed("openrouter/free", "vision", True)
        _auto = {
            "id": "openrouter/free", "name": "Free Auto-Router",
            "provider": "openrouter", "cost_per_hour": 0, "is_free": True,
            "context_length": 0, "created": int(time.time()),
            "desc": "Auto-routes to best available free model",
            **_auto_speed,
        }
        vision.append(dict(_auto))
        text.append({**_auto, **_estimate_speed("openrouter/free", "text", True)})

    # Fetch OpenRouter models
    all_models = await _fetch_openrouter_models() if _key_is_set(settings.OPENROUTER_API_KEY) else None

    if all_models:
        for m in all_models:
            mid = m.get("id", "")
            if mid == "openrouter/free":
                continue  # already added above
            name = m.get("name", mid)
            arch = m.get("architecture", {})
            modality = arch.get("modality", "")
            has_vision = "image" in modality
            is_free = ":free" in mid or _is_zero_cost(m)
            ctx = m.get("context_length", 0)
            created = m.get("created", 0)  # Unix timestamp

            entry_base = {
                "id": mid, "name": name, "provider": "openrouter",
                "is_free": is_free, "context_length": ctx, "created": created,
            }

            if has_vision:
                compatible, tracking_score = _vision_tracking_compat(mid, ctx)
                if not compatible:
                    continue  # Skip models that can't do subject tracking
                v_cost = _cost_per_hour(m, "vision")
                v_speed = _estimate_speed(mid, "vision", is_free)
                # Use tracking-specific score when it's higher
                if tracking_score > v_speed.get("quality_score", 0):
                    v_speed["quality_score"] = tracking_score
                    v_speed["quality"] = {1: "minimal", 2: "basic", 3: "good", 4: "excellent", 5: "best"}.get(tracking_score, "good")
                vision.append({**entry_base, "cost_per_hour": round(v_cost, 4),
                               "desc": f"{'FREE' if is_free else f'~${v_cost:.3f}/hr'} — {ctx:,} ctx",
                               "tracking_score": tracking_score,
                               **v_speed})

            t_cost = _cost_per_hour(m, "text")
            t_speed = _estimate_speed(mid, "text", is_free)
            text.append({**entry_base, "cost_per_hour": round(t_cost, 4),
                         "desc": f"{'FREE' if is_free else f'~${t_cost:.3f}/hr'} — {ctx:,} ctx",
                         **t_speed})

    # Add direct provider models if keys are set
    if _key_is_set(settings.ANTHROPIC_API_KEY):
        for m in _ANTHROPIC_MODELS:
            mid = m["id"]
            entry = {**m, "cost_per_hour": 0, "is_free": False,
                     "desc": f"Direct Anthropic API — {m['context_length']:,} ctx"}
            if m.get("vision"):
                vision.append({**entry, **_estimate_speed(mid, "vision", False)})
            text.append({**entry, **_estimate_speed(mid, "text", False)})

    if _key_is_set(settings.GEMINI_API_KEY):
        for m in _GEMINI_MODELS:
            mid = m["id"]
            entry = {**m, "cost_per_hour": 0, "is_free": False,
                     "desc": f"Direct Gemini API — {m['context_length']:,} ctx"}
            if m.get("vision"):
                vision.append({**entry, **_estimate_speed(mid, "vision", False)})
            text.append({**entry, **_estimate_speed(mid, "text", False)})

    # Add Ollama local models if Ollama is in the chain and reachable
    _VISION_FAMILIES = {"llava", "moondream", "bakllava", "minicpm-v", "llava-llama3", "llava-phi3", "nanollava"}
    if "ollama" in settings.active_provider_chain:
        _ollama_seen_ids: set[str] = set()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{settings.OLLAMA_HOST}/api/tags")
                if resp.status_code == 200:
                    ollama_data = resp.json()
                    for m in ollama_data.get("models", []):
                        model_name = m.get("name", "")
                        model_family = model_name.split(":")[0].lower()
                        has_vision = any(vf in model_family for vf in _VISION_FAMILIES)

                        size_bytes = m.get("size", 0)
                        size_gb = round(size_bytes / (1024**3), 1) if size_bytes else 0
                        details = m.get("details", {})
                        param_size = details.get("parameter_size", "")
                        quant = details.get("quantization_level", "")

                        desc_parts = ["LOCAL", "FREE"]
                        if param_size:
                            desc_parts.append(param_size)
                        if quant:
                            desc_parts.append(quant)
                        if size_gb:
                            desc_parts.append(f"{size_gb}GB")
                        desc = " — ".join(desc_parts)

                        entry = {
                            "id": f"ollama/{model_name}",
                            "name": f"{model_name} (Ollama Local)",
                            "provider": "ollama",
                            "is_free": True,
                            "cost_per_hour": 0,
                            "context_length": 0,
                            "created": int(time.time()),  # Sort to top as "newest"
                            "desc": desc,
                            "speed": "balanced",
                            "est_time_display": "varies by GPU",
                            "quality_score": 3,
                            "quality": "good",
                        }

                        _ollama_seen_ids.add(f"ollama/{model_name}")
                        if has_vision:
                            compatible, tracking_score = _vision_tracking_compat(f"ollama/{model_name}", 0)
                            if compatible:
                                entry["tracking_score"] = tracking_score
                                if tracking_score > entry.get("quality_score", 0):
                                    entry["quality_score"] = tracking_score
                                    entry["quality"] = {1: "minimal", 2: "basic", 3: "good", 4: "excellent", 5: "best"}.get(tracking_score, "good")
                                vision.append(entry)
                        # All models can do text
                        text.append(entry)
        except Exception as e:
            logger.warning("Failed to fetch Ollama models for available list: %s", e)

        # Always show the configured default models even if they haven't been
        # pulled yet (e.g. Ollama was just toggled on and pulls are in progress).
        # This lets the user select them in the dropdown immediately.
        _defaults = [
            (settings.OLLAMA_VISION_MODEL, True),   # (model_name, is_vision)
            (settings.OLLAMA_TEXT_MODEL, False),
        ]
        for _def_name, _def_is_vision in _defaults:
            if not _def_name:
                continue
            _def_id = f"ollama/{_def_name}"
            if _def_id in _ollama_seen_ids:
                continue  # Already listed from /api/tags
            _def_family = _def_name.split(":")[0].lower()
            _is_vision = _def_is_vision or any(vf in _def_family for vf in _VISION_FAMILIES)
            _def_entry = {
                "id": _def_id,
                "name": f"{_def_name} (Ollama Local — pulling...)",
                "provider": "ollama",
                "is_free": True,
                "cost_per_hour": 0,
                "context_length": 0,
                "created": int(time.time()),
                "desc": "LOCAL — FREE — downloading...",
                "speed": "balanced",
                "est_time_display": "varies by GPU",
                "quality_score": 3,
                "quality": "good",
            }
            if _is_vision:
                vision.append(_def_entry)
            text.append(_def_entry)
            _ollama_seen_ids.add(_def_id)

    # Sort: free first, then newer + cheaper towards the top
    # Within free models: newest first.  Within paid: newest first, then cheapest.
    def _sort_key(m):
        is_paid = 0 if m.get("is_free") else 1
        newest_first = -(m.get("created", 0))  # negate so newer = smaller = first
        cost = m.get("cost_per_hour", 999)
        return (is_paid, newest_first, cost)

    vision.sort(key=_sort_key)
    text.sort(key=_sort_key)

    # Limit to top 100 per category to avoid overwhelming the UI
    # Return current models based on which provider is primary.
    # If Ollama is first in chain OR is in the chain and has models configured,
    # return Ollama models so the UI shows what the user actually selected.
    chain = settings.active_provider_chain
    ollama_is_primary = chain and chain[0] == "ollama"
    ollama_models_set = (
        "ollama" in chain
        and settings.OLLAMA_VISION_MODEL
        and settings.OLLAMA_TEXT_MODEL
    )
    if ollama_is_primary or (ollama_models_set and not _key_is_set(settings.OPENROUTER_API_KEY)):
        current_vision = f"ollama/{settings.OLLAMA_VISION_MODEL}"
        current_text = f"ollama/{settings.OLLAMA_TEXT_MODEL}"
    else:
        current_vision = settings.OPENROUTER_VISION_MODEL
        current_text = settings.OPENROUTER_TEXT_MODEL

    return {
        "transcript": transcript,
        "vision": vision[:100],
        "text": text[:100],
        "current": {
            "transcript_model": settings.WHISPER_MODEL,
            "vision_model": current_vision,
            "text_model": current_text,
        },
    }


class SaveModelsRequest(BaseModel):
    transcript_model: str = ""
    vision_model: str = ""
    text_model: str = ""


@router.post("/providers/models/save")
async def save_models(req: SaveModelsRequest, user: User = Depends(get_current_user)):
    """Save per-task model selections to the user's settings."""
    patch: dict = {}

    if req.transcript_model:
        patch["WHISPER_MODEL"] = req.transcript_model
        patch["WHISPER_MODEL_USER_SET"] = True
        _pre_download_whisper_model(req.transcript_model)

    if req.vision_model:
        if req.vision_model.startswith("ollama/"):
            patch["OLLAMA_VISION_MODEL"] = req.vision_model[len("ollama/"):]
        else:
            patch["OPENROUTER_VISION_MODEL"] = req.vision_model
            patch["OPENROUTER_PRESET"] = "custom"

    if req.text_model:
        if req.text_model.startswith("ollama/"):
            patch["OLLAMA_TEXT_MODEL"] = req.text_model[len("ollama/"):]
        else:
            patch["OPENROUTER_TEXT_MODEL"] = req.text_model
            patch["OPENROUTER_SUMMARY_MODEL"] = req.text_model
            patch["OPENROUTER_PRESET"] = "custom"

    # Auto-adjust fallback chain for the user based on model provider
    has_ollama = (
        (req.vision_model and req.vision_model.startswith("ollama/"))
        or (req.text_model and req.text_model.startswith("ollama/"))
    )
    has_openrouter = (
        (req.vision_model and not req.vision_model.startswith("ollama/") and req.vision_model)
        or (req.text_model and not req.text_model.startswith("ollama/") and req.text_model)
    )
    us = await read_user_settings(user.id)
    chain_raw = us.get("AI_FALLBACK_CHAIN") or settings.AI_FALLBACK_CHAIN
    chain = [p.strip() for p in chain_raw.split(",") if p.strip()]

    if has_ollama:
        if "ollama" not in chain:
            chain.insert(0, "ollama")
        elif chain[0] != "ollama":
            chain = ["ollama"] + [p for p in chain if p != "ollama"]
        patch["AI_FALLBACK_CHAIN"] = ",".join(chain)
    elif has_openrouter:
        if "openrouter" not in chain:
            chain.insert(0, "openrouter")
        elif chain[0] != "openrouter":
            chain = ["openrouter"] + [p for p in chain if p != "openrouter"]
        patch["AI_FALLBACK_CHAIN"] = ",".join(chain)

    await write_user_settings(user.id, patch)
    _invalidate_status_cache()

    if has_ollama:
        pull_models = []
        if req.vision_model and req.vision_model.startswith("ollama/"):
            pull_models.append(req.vision_model[len("ollama/"):])
        if req.text_model and req.text_model.startswith("ollama/"):
            pull_models.append(req.text_model[len("ollama/"):])
        if pull_models:
            _pull_ollama_models_background(pull_models)

    # Reflect the user's saved models in the response.
    merged = {**us, **patch}
    use_ollama = has_ollama or (chain and chain[0] == "ollama")
    if use_ollama:
        return {
            "status": "saved",
            "transcript_model": merged.get("WHISPER_MODEL", settings.WHISPER_MODEL),
            "vision_model": f"ollama/{merged.get('OLLAMA_VISION_MODEL', settings.OLLAMA_VISION_MODEL)}",
            "text_model": f"ollama/{merged.get('OLLAMA_TEXT_MODEL', settings.OLLAMA_TEXT_MODEL)}",
        }
    return {
        "status": "saved",
        "transcript_model": merged.get("WHISPER_MODEL", settings.WHISPER_MODEL),
        "vision_model": merged.get("OPENROUTER_VISION_MODEL", settings.OPENROUTER_VISION_MODEL),
        "text_model": merged.get("OPENROUTER_TEXT_MODEL", settings.OPENROUTER_TEXT_MODEL),
    }


# ── Transcription Settings ────────────────────────────────────────


class SaveTranscriptionSettingsRequest(BaseModel):
    beam_size: Optional[int] = None      # 1-5
    vad_filter: Optional[bool] = None
    frame_sample_rate: Optional[int] = None  # 5-30 seconds


@router.get("/transcription/settings")
async def get_transcription_settings(user: User = Depends(get_current_user)):
    """Return per-user transcription speed/quality settings."""
    us = await read_user_settings(user.id)
    return {
        "whisper_model": us.get("WHISPER_MODEL", settings.WHISPER_MODEL),
        "beam_size": us.get("WHISPER_BEAM_SIZE", settings.WHISPER_BEAM_SIZE),
        "vad_filter": us.get("WHISPER_VAD_FILTER", settings.WHISPER_VAD_FILTER),
        "frame_sample_rate": us.get("FRAME_SAMPLE_RATE", settings.FRAME_SAMPLE_RATE),
    }


@router.post("/transcription/settings")
async def save_transcription_settings(req: SaveTranscriptionSettingsRequest, user: User = Depends(get_current_user)):
    """Save per-user transcription speed/quality settings."""
    patch: dict = {}
    if req.beam_size is not None:
        patch["WHISPER_BEAM_SIZE"] = max(1, min(5, req.beam_size))
    if req.vad_filter is not None:
        patch["WHISPER_VAD_FILTER"] = req.vad_filter
    if req.frame_sample_rate is not None:
        patch["FRAME_SAMPLE_RATE"] = max(5, min(30, req.frame_sample_rate))
    if patch:
        await write_user_settings(user.id, patch)
        _invalidate_status_cache()

    us = await read_user_settings(user.id)
    return {
        "status": "saved",
        "beam_size": us.get("WHISPER_BEAM_SIZE", settings.WHISPER_BEAM_SIZE),
        "vad_filter": us.get("WHISPER_VAD_FILTER", settings.WHISPER_VAD_FILTER),
        "frame_sample_rate": us.get("FRAME_SAMPLE_RATE", settings.FRAME_SAMPLE_RATE),
    }


# ── Encoding Settings ─────────────────────────────────────────────

_VALID_PRESETS = {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"}


class SaveEncodingSettingsRequest(BaseModel):
    preset: Optional[str] = None
    crf: Optional[int] = None
    threads: Optional[int] = None
    faststart: Optional[bool] = None


@router.get("/encoding/settings")
async def get_encoding_settings(user: User = Depends(get_current_user)):
    """Return per-user FFmpeg encoding settings."""
    us = await read_user_settings(user.id)
    return {
        "preset": us.get("FFMPEG_PRESET", settings.FFMPEG_PRESET),
        "crf": us.get("FFMPEG_CRF", settings.FFMPEG_CRF),
        "threads": us.get("FFMPEG_THREADS", settings.FFMPEG_THREADS),
        "faststart": us.get("FFMPEG_FASTSTART", settings.FFMPEG_FASTSTART),
    }


@router.post("/encoding/settings")
async def save_encoding_settings(req: SaveEncodingSettingsRequest, user: User = Depends(get_current_user)):
    """Save per-user FFmpeg encoding settings."""
    patch: dict = {}
    if req.preset is not None and req.preset in _VALID_PRESETS:
        patch["FFMPEG_PRESET"] = req.preset
    if req.crf is not None:
        patch["FFMPEG_CRF"] = max(0, min(51, req.crf))
    if req.threads is not None:
        patch["FFMPEG_THREADS"] = max(0, min(32, req.threads))
    if req.faststart is not None:
        patch["FFMPEG_FASTSTART"] = req.faststart
    if patch:
        await write_user_settings(user.id, patch)

    us = await read_user_settings(user.id)
    return {
        "status": "saved",
        "preset": us.get("FFMPEG_PRESET", settings.FFMPEG_PRESET),
        "crf": us.get("FFMPEG_CRF", settings.FFMPEG_CRF),
        "threads": us.get("FFMPEG_THREADS", settings.FFMPEG_THREADS),
        "faststart": us.get("FFMPEG_FASTSTART", settings.FFMPEG_FASTSTART),
    }


# ── Subject Tracking ──────────────────────────────────────────────


@router.get("/subject-tracking")
async def get_subject_tracking(user: User = Depends(get_current_user)):
    """Return per-user subject tracking enabled state."""
    us = await read_user_settings(user.id)
    return {"enabled": us.get("SUBJECT_TRACKING_ENABLED", settings.SUBJECT_TRACKING_ENABLED)}


class SubjectTrackingRequest(BaseModel):
    enabled: bool


@router.post("/subject-tracking")
async def set_subject_tracking(req: SubjectTrackingRequest, user: User = Depends(get_current_user)):
    """Toggle per-user subject tracking on/off."""
    await write_user_settings(user.id, {"SUBJECT_TRACKING_ENABLED": req.enabled})
    return {"status": "saved", "enabled": req.enabled}


# ── GPU Hardware Acceleration ────────────────────────────────────


@router.get("/gpu-acceleration")
async def get_gpu_acceleration(user: User = Depends(get_current_user)):
    """Return current GPU acceleration toggle state and detected GPU info.

    When enabled, runs GPU detection and returns full hardware details.
    When disabled, returns minimal info with vendor='none'.
    """
    from backend.services.clip_exporter import detect_gpu_capabilities

    # Use cached detection result if available — the test-encode can fail
    # transiently when Ollama is using the GPU. Only force re-detect when
    # explicitly requested (POST toggle endpoint uses force_redetect=True).
    gpu_info = await asyncio.to_thread(detect_gpu_capabilities, force_redetect=False)

    return {
        "enabled": settings.GPU_ACCELERATION_ENABLED,
        "vendor_override": settings.GPU_VENDOR_OVERRIDE,
        "hwdecode_enabled": settings.GPU_HWDECODE_ENABLED,
        "hevc_for_4k": settings.GPU_HEVC_FOR_4K,
        "gpu_device_index": settings.GPU_DEVICE_INDEX,
        "detected": {
            "vendor": gpu_info["vendor"],
            "gpu_name": gpu_info.get("gpu_name", "Unknown"),
            "encoder": gpu_info["encoder"],
            "hevc_encoder": gpu_info.get("hevc_encoder"),
            "decoder": gpu_info["decoder"],
            "hwaccel": gpu_info["hwaccel"],
            "capabilities": gpu_info.get("capabilities", []),
            "vram_mb": gpu_info.get("vram_mb", 0),
            "driver_version": gpu_info.get("driver_version", ""),
            "cuda_available": gpu_info.get("cuda_available", False),
            "whisper_device": gpu_info.get("whisper_device", "cpu"),
            "gpus": gpu_info.get("gpus", []),
            "gpu_issues": gpu_info.get("gpu_issues", []),
        },
    }


class GpuAccelerationRequest(BaseModel):
    enabled: bool
    vendor_override: Optional[str] = None


@router.post("/gpu-acceleration")
async def set_gpu_acceleration(req: GpuAccelerationRequest, user: User = Depends(get_current_user)):
    """Toggle GPU acceleration on/off and optionally set vendor override.

    When toggled ON: clears cached GPU info, re-scans for available GPUs,
    runs test-encodes to confirm the encoder works, returns full GPU details.
    When toggled OFF: clears cache, returns CPU fallback info.
    """
    from backend.services.clip_exporter import detect_gpu_capabilities, _gpu_info_cache_clear
    from backend.services.transcription import reload_model as reload_whisper_model

    settings.GPU_ACCELERATION_ENABLED = req.enabled
    if req.vendor_override is not None:
        settings.GPU_VENDOR_OVERRIDE = req.vendor_override

    _persist_user_settings()

    # Force re-detection so the response includes fresh GPU info
    _gpu_info_cache_clear()
    # Reload Whisper model so it moves between CPU/CUDA to match the toggle
    reload_whisper_model()
    # Run in thread to avoid blocking the event loop (subprocess calls inside).
    gpu_info = await asyncio.to_thread(detect_gpu_capabilities, force_redetect=True)

    return {
        "status": "saved",
        "enabled": settings.GPU_ACCELERATION_ENABLED,
        "detected": {
            "vendor": gpu_info["vendor"],
            "gpu_name": gpu_info.get("gpu_name", "Unknown"),
            "encoder": gpu_info["encoder"],
            "decoder": gpu_info["decoder"],
            "hwaccel": gpu_info["hwaccel"],
            "vram_mb": gpu_info.get("vram_mb", 0),
            "driver_version": gpu_info.get("driver_version", ""),
            "cuda_available": gpu_info.get("cuda_available", False),
            "whisper_device": gpu_info.get("whisper_device", "cpu"),
            "gpus": gpu_info.get("gpus", []),
            "gpu_issues": gpu_info.get("gpu_issues", []),
        },
    }


# ── Client GPU (Browser) Report ──────────────────────────────────


class ClientGpuReport(BaseModel):
    """Reported by the browser after GPU detection."""
    webgpu_supported: bool = False
    webcodec_supported: bool = False
    gpu_name: str = ""
    gpu_vendor: str = ""
    estimated_vram_mb: int = 0
    has_fp16: bool = False
    whisper_capable: bool = False
    h264_hardware_encode: bool = False
    hevc_hardware_encode: bool = False
    h264_hw_encode: bool = False
    hevc_hw_encode: bool = False
    client_whisper_enabled: bool = False
    client_encoding_enabled: bool = False
    gpu_index: str = "0"
    gpu_backend: str = ""


@router.post("/client-gpu-report")
async def report_client_gpu(req: ClientGpuReport, user: User = Depends(get_current_user)):
    """Store client GPU capabilities so the pipeline can decide where to process.

    The server uses this to skip server-side transcription if the client will
    handle it, or to prepare server-side fallback if the client can't.
    Also stores the user's selected GPU index for FFmpeg device selection.

    When the client reports an NVIDIA GPU and server-side GPU acceleration
    is not yet enabled, this triggers auto-detection and enables it.
    """
    # Store the selected GPU index so FFmpeg can target the right device
    if req.gpu_index:
        settings.GPU_DEVICE_INDEX = req.gpu_index
        logger.info("GPU device index set to %s (%s)", req.gpu_index, req.gpu_name)
        _persist_user_settings()

    # Auto-enable server GPU acceleration if client reports NVIDIA GPU
    # and server hasn't enabled it yet
    if req.gpu_vendor and "nvidia" in req.gpu_vendor.lower() and not settings.GPU_ACCELERATION_ENABLED:
        logger.info(
            "Client reports NVIDIA GPU (%s) — auto-enabling server GPU acceleration",
            req.gpu_name,
        )
        settings.GPU_ACCELERATION_ENABLED = True
        settings.GPU_VENDOR_OVERRIDE = "nvidia"
        _persist_user_settings()
        # Force GPU re-detection and reload Whisper model for CUDA
        try:
            from backend.services.clip_exporter import _gpu_info_cache_clear
            _gpu_info_cache_clear()
            from backend.services.transcription import reload_model as reload_whisper_model
            reload_whisper_model()
        except Exception:
            pass

    logger.info(
        "Client GPU report: webgpu=%s gpu=%s vendor=%s whisper_capable=%s "
        "client_whisper=%s client_encoding=%s gpu_index=%s",
        req.webgpu_supported, req.gpu_name, req.gpu_vendor, req.whisper_capable,
        req.client_whisper_enabled, req.client_encoding_enabled, req.gpu_index,
    )
    return {"status": "received"}


# ── GPU QA & Validation ──────────────────────────────────────────────


@router.get("/gpu-qa")
async def gpu_qa_validation(user: User = Depends(get_current_user)):
    """Run comprehensive GPU QA validation.

    Checks that GPU acceleration is properly configured and actually
    being used for both Whisper transcription and FFmpeg video encoding.
    Returns a structured report with pass/fail checks, warnings, and
    actionable recommendations.
    """
    import subprocess as _subprocess

    from backend.services.clip_exporter import (
        detect_gpu_capabilities,
        _gpu_encode_args,
        _gpu_decode_args,
    )
    from backend.services.transcription import whisper_device_info

    checks = []
    warnings = []
    errors = []

    # ── 1. GPU Detection ──
    gpu_info = await asyncio.to_thread(detect_gpu_capabilities, force_redetect=True)
    gpu_vendor = gpu_info.get("vendor", "none")
    gpu_name = gpu_info.get("gpu_name", "Unknown")

    if gpu_vendor != "none":
        checks.append({
            "name": "GPU detected",
            "status": "pass",
            "detail": f"{gpu_name} (vendor: {gpu_vendor})",
        })
    else:
        checks.append({
            "name": "GPU detected",
            "status": "fail",
            "detail": "No GPU detected by FFmpeg/system probes",
        })
        errors.append("No GPU hardware detected — GPU acceleration cannot work")

    # ── 2. GPU Acceleration Setting ──
    if settings.GPU_ACCELERATION_ENABLED:
        checks.append({
            "name": "GPU acceleration enabled",
            "status": "pass",
            "detail": f"Enabled (vendor_override={settings.GPU_VENDOR_OVERRIDE or 'auto'})",
        })
    else:
        checks.append({
            "name": "GPU acceleration enabled",
            "status": "fail",
            "detail": "GPU acceleration is disabled in settings",
        })
        errors.append(
            "GPU acceleration is disabled — enable it in Settings > GPU Acceleration"
        )

    # ── 3. CUDA Runtime (for Whisper) ──
    cuda_available = False
    cuda_device_count = 0
    try:
        from backend.services.transcription import _detect_cuda_available
        cuda_available, cuda_device_count, _ = _detect_cuda_available()
    except Exception:
        pass

    if cuda_available and cuda_device_count > 0:
        checks.append({
            "name": "CUDA runtime available",
            "status": "pass",
            "detail": f"{cuda_device_count} CUDA device(s) found",
        })
    else:
        checks.append({
            "name": "CUDA runtime available",
            "status": "warn" if gpu_vendor != "none" else "fail",
            "detail": "CUDA runtime not available (ctranslate2/torch cannot use GPU)",
        })
        if gpu_vendor == "nvidia":
            warnings.append(
                "NVIDIA GPU detected but CUDA runtime is not available. "
                "Ensure CUDA toolkit is installed and container has --gpus all."
            )

    # ── 4. Whisper Model GPU Status ──
    whisper_dev = whisper_device_info.get("device", "cpu")
    whisper_idx = whisper_device_info.get("device_index", 0)
    whisper_compute = whisper_device_info.get("compute_type", "int8")

    if whisper_dev == "cuda":
        checks.append({
            "name": "Whisper using GPU",
            "status": "pass",
            "detail": f"device=cuda:{whisper_idx}, compute_type={whisper_compute}",
        })
    else:
        status = "fail" if settings.GPU_ACCELERATION_ENABLED and cuda_available else "warn"
        checks.append({
            "name": "Whisper using GPU",
            "status": status,
            "detail": f"Whisper running on CPU ({whisper_compute})",
        })
        if status == "fail":
            errors.append(
                "GPU is enabled and CUDA is available, but Whisper is running on CPU. "
                "Try toggling GPU acceleration off and on to reload the model."
            )

    # ── 5. Whisper GPU Verification (live check) ──
    if whisper_dev == "cuda":
        verification_results = []
        try:
            import ctranslate2
            ct2_count = ctranslate2.get_cuda_device_count()
            if ct2_count > 0:
                verification_results.append(f"ctranslate2: {ct2_count} CUDA device(s)")
        except Exception:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated(whisper_idx)
                verification_results.append(
                    f"torch: {mem / 1024 / 1024:.1f}MB allocated on device {whisper_idx}"
                )
        except Exception:
            pass

        try:
            smi = _subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if smi.returncode == 0 and smi.stdout.strip():
                our_pid = str(os.getpid())
                for line in smi.stdout.strip().split("\n"):
                    if our_pid in line:
                        verification_results.append(f"nvidia-smi: PID {our_pid} using GPU")
                        break
        except Exception:
            pass

        if verification_results:
            checks.append({
                "name": "Whisper GPU verification",
                "status": "pass",
                "detail": "; ".join(verification_results),
            })
        else:
            checks.append({
                "name": "Whisper GPU verification",
                "status": "warn",
                "detail": "Could not independently verify GPU memory usage",
            })
            warnings.append(
                "Whisper reports device=cuda but live GPU verification could not confirm usage"
            )

    # ── 6. FFmpeg GPU Encoder ──
    encoder = gpu_info.get("encoder", "")
    if encoder and encoder != "libx264":
        checks.append({
            "name": "FFmpeg GPU encoder available",
            "status": "pass",
            "detail": f"Encoder: {encoder}",
        })
    elif settings.GPU_ACCELERATION_ENABLED and gpu_vendor != "none":
        checks.append({
            "name": "FFmpeg GPU encoder available",
            "status": "fail",
            "detail": "GPU detected but no hardware encoder found by FFmpeg",
        })
        errors.append(
            "FFmpeg cannot find a GPU encoder. Ensure FFmpeg is built with NVENC/VAAPI/QSV support."
        )
    else:
        checks.append({
            "name": "FFmpeg GPU encoder available",
            "status": "info",
            "detail": "Using software encoder (libx264)",
        })

    # ── 7. FFmpeg GPU Decoder / HW Decode ──
    if settings.GPU_HWDECODE_ENABLED:
        hwaccel = gpu_info.get("hwaccel", "")
        if hwaccel:
            checks.append({
                "name": "FFmpeg GPU decoder available",
                "status": "pass",
                "detail": f"hwaccel: {hwaccel}",
            })
        else:
            checks.append({
                "name": "FFmpeg GPU decoder available",
                "status": "warn",
                "detail": "Hardware decode enabled but no hwaccel method detected",
            })
    else:
        checks.append({
            "name": "FFmpeg GPU decoder available",
            "status": "info",
            "detail": "Hardware decode disabled in settings",
        })

    # ── 8. Test Encode (quick NVENC/VAAPI probe) ──
    if settings.GPU_ACCELERATION_ENABLED and encoder and encoder != "libx264":
        try:
            # Provide a minimal quality preset dict for the test
            test_preset = {"crf": 23, "preset": "fast"}
            encode_args = await asyncio.to_thread(_gpu_encode_args, test_preset, "1080p")
            if encode_args:
                # Run a minimal test encode to verify GPU encoder actually works
                test_cmd = [
                    "ffmpeg", "-y", "-f", "lavfi", "-i",
                    "color=c=black:s=64x64:d=0.1:r=1",
                    *encode_args, "-frames:v", "1",
                    "-f", "null", "-",
                ]
                proc = await asyncio.create_subprocess_exec(
                    *test_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=15)
                if proc.returncode == 0:
                    checks.append({
                        "name": "GPU test encode",
                        "status": "pass",
                        "detail": f"Test encode succeeded with {encoder}",
                    })
                else:
                    stderr_text = stderr_bytes.decode(errors="replace")[-300:]
                    checks.append({
                        "name": "GPU test encode",
                        "status": "fail",
                        "detail": f"Test encode failed: {stderr_text}",
                    })
                    errors.append(
                        f"GPU encoder '{encoder}' failed test encode. "
                        "The GPU driver or FFmpeg build may not support this encoder."
                    )
            else:
                checks.append({
                    "name": "GPU test encode",
                    "status": "warn",
                    "detail": "No encode args returned — encoder may not be configured",
                })
        except asyncio.TimeoutError:
            checks.append({
                "name": "GPU test encode",
                "status": "warn",
                "detail": "Test encode timed out (15s)",
            })
        except Exception as exc:
            checks.append({
                "name": "GPU test encode",
                "status": "warn",
                "detail": f"Test encode error: {exc}",
            })

    # ── 9. Device Index Consistency ──
    configured_idx = (settings.GPU_DEVICE_INDEX or "0").strip()
    if whisper_dev == "cuda" and str(whisper_idx) != configured_idx:
        warnings.append(
            f"Whisper is on CUDA device {whisper_idx} but GPU_DEVICE_INDEX is '{configured_idx}'. "
            "Toggle GPU off/on to apply the new device index."
        )

    # ── Summary ──
    pass_count = sum(1 for c in checks if c["status"] == "pass")
    fail_count = sum(1 for c in checks if c["status"] == "fail")
    warn_count = sum(1 for c in checks if c["status"] == "warn")

    overall = "pass"
    if fail_count > 0:
        overall = "fail"
    elif warn_count > 0:
        overall = "warn"

    return {
        "overall": overall,
        "summary": f"{pass_count} passed, {fail_count} failed, {warn_count} warnings",
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "config": {
            "gpu_acceleration_enabled": settings.GPU_ACCELERATION_ENABLED,
            "gpu_vendor_override": settings.GPU_VENDOR_OVERRIDE,
            "gpu_hwdecode_enabled": settings.GPU_HWDECODE_ENABLED,
            "gpu_hevc_for_4k": settings.GPU_HEVC_FOR_4K,
            "gpu_device_index": settings.GPU_DEVICE_INDEX,
        },
        "whisper": {
            "device": whisper_dev,
            "compute_type": whisper_compute,
            "device_index": whisper_idx,
            "gpu_name": whisper_device_info.get("gpu_name", ""),
        },
        "ffmpeg": {
            "vendor": gpu_vendor,
            "gpu_name": gpu_name,
            "encoder": encoder,
            "hevc_encoder": gpu_info.get("hevc_encoder", ""),
            "decoder": gpu_info.get("decoder", ""),
            "hwaccel": gpu_info.get("hwaccel", ""),
        },
    }


# ── Prompt Management ──────────────────────────────────────────────


class SavePromptsRequest(BaseModel):
    frame_analysis: Optional[str] = None
    viral_clip_detection: Optional[str] = None
    subject_tracking: Optional[str] = None
    summary: Optional[str] = None
    seo: Optional[str] = None


@router.get("/prompts")
async def get_prompts(user: User = Depends(get_current_user)):
    """Return current custom prompts and defaults."""
    current = load_prompts()
    defaults = get_defaults()
    return {
        "current": current.model_dump(),
        "defaults": defaults.model_dump(),
    }


@router.post("/prompts")
async def update_prompts(req: SavePromptsRequest, user: User = Depends(get_current_user)):
    """Save custom prompts. Pass null/empty to reset a prompt to default."""
    current = load_prompts()
    defaults = get_defaults()

    if req.frame_analysis is not None:
        text = req.frame_analysis.strip()
        if len(text) > MAX_PROMPT_LENGTH:
            return {
                "status": "error",
                "message": f"Frame analysis prompt exceeds {MAX_PROMPT_LENGTH} characters",
            }
        current.frame_analysis = text if text else defaults.frame_analysis

    if req.viral_clip_detection is not None:
        text = req.viral_clip_detection.strip()
        if len(text) > MAX_PROMPT_LENGTH:
            return {
                "status": "error",
                "message": f"Viral clip detection prompt exceeds {MAX_PROMPT_LENGTH} characters",
            }
        current.viral_clip_detection = text if text else defaults.viral_clip_detection

    if req.subject_tracking is not None:
        text = req.subject_tracking.strip()
        if len(text) > MAX_PROMPT_LENGTH:
            return {
                "status": "error",
                "message": f"Subject tracking prompt exceeds {MAX_PROMPT_LENGTH} characters",
            }
        current.subject_tracking = text if text else defaults.subject_tracking

    if req.summary is not None:
        text = req.summary.strip()
        if len(text) > MAX_PROMPT_LENGTH:
            return {
                "status": "error",
                "message": f"Summary prompt exceeds {MAX_PROMPT_LENGTH} characters",
            }
        current.summary = text if text else defaults.summary

    if req.seo is not None:
        text = req.seo.strip()
        if len(text) > MAX_PROMPT_LENGTH:
            return {
                "status": "error",
                "message": f"SEO prompt exceeds {MAX_PROMPT_LENGTH} characters",
            }
        current.seo = text if text else defaults.seo

    save_prompts(current)
    return {"status": "saved", "prompts": current.model_dump()}


@router.post("/prompts/reset")
async def reset_prompts(user: User = Depends(get_current_user)):
    """Reset all prompts to defaults."""
    defaults = get_defaults()
    save_prompts(defaults)
    return {"status": "reset", "prompts": defaults.model_dump()}


# ═══════════════════════════════════════════════════════════════
# Site Customisation — title, favicon, logo
# ═══════════════════════════════════════════════════════════════

SITE_CONFIG_PATH = os.path.join(_DATA_DIR, "site_config.json")
SITE_UPLOADS_DIR = os.path.join(_DATA_DIR, "site_uploads")


def _load_site_config() -> dict:
    if os.path.exists(SITE_CONFIG_PATH):
        try:
            with open(SITE_CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_site_config(cfg: dict):
    os.makedirs(os.path.dirname(SITE_CONFIG_PATH), exist_ok=True)
    with open(SITE_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


@router.get("/site-config")
async def get_site_config(user: User = Depends(get_current_user)):
    """Return site customisation (title, favicon URL, logo URL)."""
    return _load_site_config()


@router.post("/site-config")
async def update_site_config(
    title: Optional[str] = Form(None),
    favicon: Optional[UploadFile] = File(None),
    logo: Optional[UploadFile] = File(None),
    remove_favicon: Optional[str] = Form(None),
    remove_logo: Optional[str] = Form(None),
    user: User = Depends(get_current_user),
):
    """Update site title, favicon, and/or logo."""
    cfg = _load_site_config()
    os.makedirs(SITE_UPLOADS_DIR, exist_ok=True)

    if title is not None:
        cfg["title"] = title.strip()

    if remove_favicon == "true":
        old = cfg.pop("favicon", None)
        if old:
            old_path = os.path.join(SITE_UPLOADS_DIR, os.path.basename(old))
            if os.path.isfile(old_path):
                os.remove(old_path)
    elif favicon and favicon.filename:
        ext = os.path.splitext(favicon.filename)[1].lower() or ".ico"
        fname = f"favicon-{uuid.uuid4().hex[:8]}{ext}"
        fpath = os.path.join(SITE_UPLOADS_DIR, fname)
        content = await favicon.read()
        with open(fpath, "wb") as f:
            f.write(content)
        cfg["favicon"] = fname

    if remove_logo == "true":
        old = cfg.pop("logo", None)
        if old:
            old_path = os.path.join(SITE_UPLOADS_DIR, os.path.basename(old))
            if os.path.isfile(old_path):
                os.remove(old_path)
    elif logo and logo.filename:
        ext = os.path.splitext(logo.filename)[1].lower() or ".png"
        fname = f"logo-{uuid.uuid4().hex[:8]}{ext}"
        fpath = os.path.join(SITE_UPLOADS_DIR, fname)
        content = await logo.read()
        with open(fpath, "wb") as f:
            f.write(content)
        cfg["logo"] = fname

    _save_site_config(cfg)
    return {"status": "saved", **cfg}


@router.get("/site-uploads/{filename}")
async def serve_site_upload(filename: str):
    """Serve uploaded site assets (favicon, logo)."""
    safe = os.path.basename(filename)
    fpath = os.path.join(SITE_UPLOADS_DIR, safe)
    if not os.path.isfile(fpath):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"error": "not found"})
    from fastapi.responses import FileResponse
    return FileResponse(fpath)


# ═══════════════════════════════════════════════════════════════
# UI State Sync — persists frontend localStorage to the server
# so settings stay consistent across browsers.
# ═══════════════════════════════════════════════════════════════

UI_STATE_PATH = os.path.join(_DATA_DIR, "ui_state.json")


def _load_ui_state() -> dict:
    if os.path.exists(UI_STATE_PATH):
        try:
            with open(UI_STATE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_ui_state(state: dict):
    os.makedirs(os.path.dirname(UI_STATE_PATH), exist_ok=True)
    with open(UI_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


@router.get("/ui-state")
async def get_ui_state(user: User = Depends(get_current_user)):
    """Return all persisted frontend UI state (settings, segments, etc.)."""
    return _load_ui_state()


@router.put("/ui-state")
async def put_ui_state(request: Request, user: User = Depends(get_current_user)):
    """Merge incoming UI state into the persisted file.

    Accepts a JSON object of localStorage key→value pairs.  Values that
    are ``null`` delete the key from the persisted state so that
    localStorage.removeItem propagates to other browsers.
    """
    incoming = await request.json()
    state = _load_ui_state()
    for key, value in incoming.items():
        if value is None:
            state.pop(key, None)
        else:
            state[key] = value
    _save_ui_state(state)
    return {"status": "saved", "keys": len(state)}
