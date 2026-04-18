"""JSON-backed store for connected cloud accounts.

There is no relational DB in ClipAI (see ``docs/cloud-storage/RECON.md``).
Each connected account is persisted as ``/data/cloud_accounts/<id>.json``
with the access/refresh tokens encrypted via
:mod:`backend.app.cloud.crypto`.

The schema mirrors the table described in the cloud-storage spec even
though it's a flat directory on disk:

    id                 uuid
    user_id            "local"
    provider           "google_drive" | "box"
    provider_user_id   remote account id (for display + uniqueness)
    display_name       email / name shown in UI
    access_token       Fernet-encrypted bytes (stored base64)
    refresh_token      Fernet-encrypted bytes (stored base64), nullable
    token_expires_at   ISO-8601 timestamp, nullable
    scopes             list[str]
    created_at
    updated_at

Writes are atomic (``tempfile.mkstemp`` + ``os.replace``) and serialized
per account via an in-process ``asyncio.Lock`` dict, matching the style of
``backend/database.py``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import aiofiles

from .crypto import encrypt, decrypt

logger = logging.getLogger(__name__)


ACCOUNTS_DIR = "/data/cloud_accounts"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CloudAccount:
    id: str
    user_id: str
    provider: str
    provider_user_id: str
    display_name: str
    access_token: str  # plaintext in memory only; serialized encrypted
    refresh_token: Optional[str]
    token_expires_at: Optional[str]  # ISO-8601
    scopes: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def to_public(self) -> dict:
        """Return a dict safe to send to the frontend (no tokens)."""
        return {
            "id": self.id,
            "provider": self.provider,
            "provider_user_id": self.provider_user_id,
            "display_name": self.display_name,
            "scopes": self.scopes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "token_expires_at": self.token_expires_at,
        }


# ── serialization ────────────────────────────────────────────────────────────

def _serialize(account: CloudAccount) -> dict:
    enc_access = encrypt(account.access_token)
    enc_refresh = encrypt(account.refresh_token) if account.refresh_token is not None else None
    return {
        "id": account.id,
        "user_id": account.user_id,
        "provider": account.provider,
        "provider_user_id": account.provider_user_id,
        "display_name": account.display_name,
        "access_token_enc": base64.b64encode(enc_access).decode("ascii") if enc_access else None,
        "refresh_token_enc": base64.b64encode(enc_refresh).decode("ascii") if enc_refresh else None,
        "token_expires_at": account.token_expires_at,
        "scopes": account.scopes,
        "created_at": account.created_at,
        "updated_at": account.updated_at,
    }


def _deserialize(data: dict) -> CloudAccount:
    enc_access = data.get("access_token_enc")
    enc_refresh = data.get("refresh_token_enc")
    access_token = decrypt(base64.b64decode(enc_access)) if enc_access else ""
    refresh_token = decrypt(base64.b64decode(enc_refresh)) if enc_refresh else None
    return CloudAccount(
        id=data["id"],
        user_id=data["user_id"],
        provider=data["provider"],
        provider_user_id=data["provider_user_id"],
        display_name=data["display_name"],
        access_token=access_token or "",
        refresh_token=refresh_token,
        token_expires_at=data.get("token_expires_at"),
        scopes=list(data.get("scopes", [])),
        created_at=data.get("created_at", _now_iso()),
        updated_at=data.get("updated_at", _now_iso()),
    )


# ── filesystem store ─────────────────────────────────────────────────────────

_locks: dict[str, asyncio.Lock] = {}


def _lock_for(account_id: str) -> asyncio.Lock:
    lock = _locks.get(account_id)
    if lock is None:
        lock = asyncio.Lock()
        _locks[account_id] = lock
    return lock


def _path(account_id: str) -> str:
    return os.path.join(ACCOUNTS_DIR, f"{account_id}.json")


def _ensure_dir() -> None:
    os.makedirs(ACCOUNTS_DIR, exist_ok=True)


async def save(account: CloudAccount) -> CloudAccount:
    """Persist the account, updating ``updated_at``. Returns the saved row."""
    _ensure_dir()
    account.updated_at = _now_iso()
    payload = json.dumps(_serialize(account), indent=2)
    path = _path(account.id)
    async with _lock_for(account.id):
        fd, tmp = tempfile.mkstemp(dir=ACCOUNTS_DIR, suffix=".tmp")
        try:
            async with aiofiles.open(fd, "w", closefd=True) as f:
                await f.write(payload)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    return account


async def get(account_id: str) -> Optional[CloudAccount]:
    path = _path(account_id)
    if not os.path.isfile(path):
        return None
    async with _lock_for(account_id):
        try:
            async with aiofiles.open(path, "r") as f:
                raw = await f.read()
            return _deserialize(json.loads(raw))
        except Exception as exc:
            logger.warning("Failed to load cloud account %s: %s", account_id, exc)
            return None


async def list_for_user(user_id: str) -> list[CloudAccount]:
    _ensure_dir()
    out: list[CloudAccount] = []
    for entry in sorted(os.listdir(ACCOUNTS_DIR)):
        if not entry.endswith(".json"):
            continue
        account = await get(entry[:-5])
        if account and account.user_id == user_id:
            out.append(account)
    return out


async def get_for_provider(user_id: str, provider: str) -> Optional[CloudAccount]:
    """Return the (single) account connected for this user+provider, if any."""
    for account in await list_for_user(user_id):
        if account.provider == provider:
            return account
    return None


async def delete(account_id: str) -> bool:
    path = _path(account_id)
    if not os.path.isfile(path):
        return False
    async with _lock_for(account_id):
        try:
            os.remove(path)
            return True
        except OSError as exc:
            logger.warning("Failed to delete cloud account %s: %s", account_id, exc)
            return False


async def upsert_for_provider(
    *,
    user_id: str,
    provider: str,
    provider_user_id: str,
    display_name: str,
    access_token: str,
    refresh_token: Optional[str],
    token_expires_at: Optional[str],
    scopes: list[str],
) -> CloudAccount:
    """Create-or-replace the (user_id, provider, provider_user_id) row.

    Box rotates refresh tokens on every refresh, so the refresh flow also
    calls through here to persist the new one.
    """
    existing = None
    for acc in await list_for_user(user_id):
        if acc.provider == provider and acc.provider_user_id == provider_user_id:
            existing = acc
            break

    if existing is not None:
        existing.display_name = display_name
        existing.access_token = access_token
        if refresh_token is not None:
            existing.refresh_token = refresh_token
        existing.token_expires_at = token_expires_at
        existing.scopes = scopes
        return await save(existing)

    new = CloudAccount(
        id=str(uuid.uuid4()),
        user_id=user_id,
        provider=provider,
        provider_user_id=provider_user_id,
        display_name=display_name,
        access_token=access_token,
        refresh_token=refresh_token,
        token_expires_at=token_expires_at,
        scopes=scopes,
    )
    return await save(new)


# ── State cache for OAuth `state` param (CSRF protection) ────────────────────

_STATE_TTL_SECONDS = 10 * 60

_state_cache: dict[str, tuple[str, float]] = {}
_state_lock = asyncio.Lock()


async def remember_state(state: str, provider: str) -> None:
    async with _state_lock:
        now = time.time()
        # Expire stale entries lazily
        expired = [k for k, (_, ts) in _state_cache.items() if now - ts > _STATE_TTL_SECONDS]
        for k in expired:
            _state_cache.pop(k, None)
        _state_cache[state] = (provider, now)


async def consume_state(state: str) -> Optional[str]:
    """Return the provider the state was issued for, then invalidate it."""
    async with _state_lock:
        entry = _state_cache.pop(state, None)
    if entry is None:
        return None
    provider, ts = entry
    if time.time() - ts > _STATE_TTL_SECONDS:
        return None
    return provider
