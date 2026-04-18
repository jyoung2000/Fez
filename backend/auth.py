"""API key authentication for the v1 API.

The frontend continues to work without any auth — only /api/v1/* endpoints
require a Bearer token.  The API key is auto-generated on first boot if
not provided via the CLIPAI_API_KEY environment variable.
"""

import json
import logging
import os
import secrets

from fastapi import HTTPException, Header
from typing import Optional

logger = logging.getLogger(__name__)

_SETTINGS_DIR = "/data/logs"
_API_KEY_FILE = os.path.join(_SETTINGS_DIR, "api_key.json")


def _load_key_from_file() -> Optional[str]:
    """Load API key from persistent storage."""
    if os.path.isfile(_API_KEY_FILE):
        try:
            with open(_API_KEY_FILE, "r") as f:
                data = json.load(f)
            return data.get("api_key") or None
        except Exception:
            pass
    return None


def _save_key_to_file(key: str):
    """Persist API key to disk so it survives container restarts."""
    os.makedirs(os.path.dirname(_API_KEY_FILE), exist_ok=True)
    with open(_API_KEY_FILE, "w") as f:
        json.dump({"api_key": key}, f)


def get_or_create_api_key() -> str:
    """Return the current API key, generating one if needed.

    Priority:
    1. CLIPAI_API_KEY environment variable
    2. Persisted key from previous boot
    3. Auto-generate a new key
    """
    # Check env var first
    env_key = os.getenv("CLIPAI_API_KEY", "").strip()
    if env_key:
        # Persist env key so it's available via the settings endpoint
        _save_key_to_file(env_key)
        return env_key

    # Check persisted key
    saved_key = _load_key_from_file()
    if saved_key:
        return saved_key

    # Generate new key
    new_key = f"clipai-{secrets.token_urlsafe(32)}"
    _save_key_to_file(new_key)
    logger.info(
        "\n%s\nClipAI API Key (save this): %s\n%s",
        "=" * 60, new_key, "=" * 60,
    )
    return new_key


def regenerate_api_key() -> str:
    """Generate and persist a new API key, invalidating the old one."""
    new_key = f"clipai-{secrets.token_urlsafe(32)}"
    _save_key_to_file(new_key)
    logger.info("API key regenerated")
    return new_key


def mask_key(key: str) -> str:
    """Return a masked version of the key for display."""
    if len(key) <= 12:
        return key[:4] + "..." + key[-4:]
    return key[:10] + "..." + key[-4:]


async def verify_api_key(authorization: Optional[str] = Header(None)) -> str:
    """FastAPI dependency for protected endpoints.

    Validates Bearer token against the stored API key.
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header. Use: Bearer <your-api-key>",
        )
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid auth format. Use: Bearer <your-api-key>",
        )
    token = authorization.removeprefix("Bearer ").strip()
    if token != get_or_create_api_key():
        raise HTTPException(status_code=403, detail="Invalid API key")
    return token
