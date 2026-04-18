"""File-backed user + session storage.

One JSON file per collection under ``/data/auth/``:

  * ``users.json`` — ``{"users": [User.to_storage(), ...]}``
  * ``sessions.json`` — ``{"sessions": [Session.to_storage(), ...]}``

Writes are atomic (``tempfile + os.replace``) to survive crashes; an
``asyncio.Lock`` per file serializes concurrent writers within the
process. Falls back to ``~/.clipai/auth/`` when ``/data/auth`` isn't
writable (matches the pattern used by settings storage).

All reads and writes are async so they mesh with FastAPI's event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import aiofiles

from backend.app.auth.models import Role, Session, User
from backend.app.auth.security import (
    compute_fingerprint,
    hash_password,
    new_salt,
    new_token,
)

logger = logging.getLogger(__name__)

_DEFAULT_DIR = "/data/auth"
_FALLBACK_DIR = os.path.join(
    os.path.expanduser("~"), ".clipai", "auth",
)
_SESSION_TTL_DAYS = 30


def _resolve_dir() -> str:
    """Return a writable directory for auth storage."""
    try:
        os.makedirs(_DEFAULT_DIR, exist_ok=True)
        # Probe write access
        probe = os.path.join(_DEFAULT_DIR, ".write_probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.unlink(probe)
        return _DEFAULT_DIR
    except Exception:
        os.makedirs(_FALLBACK_DIR, exist_ok=True)
        return _FALLBACK_DIR


AUTH_DIR = _resolve_dir()
USERS_PATH = os.path.join(AUTH_DIR, "users.json")
SESSIONS_PATH = os.path.join(AUTH_DIR, "sessions.json")
USER_SETTINGS_DIR = os.path.join(AUTH_DIR, "user_settings")
os.makedirs(USER_SETTINGS_DIR, exist_ok=True)

_users_lock = asyncio.Lock()
_sessions_lock = asyncio.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


async def _atomic_write_json(path: str, data: dict) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        async with aiofiles.open(fd, "w", closefd=True) as f:
            await f.write(json.dumps(data, indent=2, sort_keys=True))
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


async def _read_json(path: str, default: dict) -> dict:
    if not os.path.isfile(path):
        return dict(default)
    try:
        async with aiofiles.open(path, "r") as f:
            content = await f.read()
        if not content.strip():
            return dict(default)
        return json.loads(content)
    except Exception as e:
        logger.warning("failed to read %s: %s", path, e)
        return dict(default)


# ── Users ────────────────────────────────────────────────────────


async def list_users() -> list[User]:
    async with _users_lock:
        data = await _read_json(USERS_PATH, {"users": []})
    return [User.from_storage(u) for u in data.get("users", [])]


async def get_user(user_id: str) -> Optional[User]:
    for u in await list_users():
        if u.id == user_id:
            return u
    return None


async def get_user_by_username(username: str) -> Optional[User]:
    key = (username or "").strip().lower()
    for u in await list_users():
        if u.username.lower() == key:
            return u
    return None


async def create_user(
    username: str,
    password: str,
    role: Role = Role.USER,
    head_admin: bool = False,
) -> User:
    username = (username or "").strip()
    if not username or len(username) < 2:
        raise ValueError("username must be at least 2 characters")
    if await get_user_by_username(username) is not None:
        raise ValueError(f"username {username!r} already exists")
    salt = new_salt()
    pw_hash = hash_password(password, salt)
    user = User(
        id=str(uuid.uuid4()),
        username=username,
        password_hash=pw_hash,
        salt=salt,
        role=role,
        created_at=_now_iso(),
        head_admin=head_admin,
        active=True,
    )
    async with _users_lock:
        data = await _read_json(USERS_PATH, {"users": []})
        data.setdefault("users", []).append(user.to_storage())
        await _atomic_write_json(USERS_PATH, data)
    return user


async def update_user(user_id: str, *, role: Optional[Role] = None,
                      active: Optional[bool] = None) -> User:
    async with _users_lock:
        data = await _read_json(USERS_PATH, {"users": []})
        updated = None
        for u in data.get("users", []):
            if u["id"] == user_id:
                if u.get("head_admin"):
                    # The head admin's role/active cannot be changed
                    # even by another admin. This is the ejector-seat
                    # guarantee.
                    if role is not None and role != Role.ADMIN:
                        raise ValueError("head admin role cannot be changed")
                    if active is False:
                        raise ValueError("head admin cannot be deactivated")
                if role is not None:
                    u["role"] = role.value
                if active is not None:
                    u["active"] = bool(active)
                updated = User.from_storage(u)
                break
        if updated is None:
            raise KeyError(user_id)
        await _atomic_write_json(USERS_PATH, data)
    return updated


async def change_password(user_id: str, new_password: str) -> None:
    async with _users_lock:
        data = await _read_json(USERS_PATH, {"users": []})
        for u in data.get("users", []):
            if u["id"] == user_id:
                salt = new_salt()
                u["salt"] = salt
                u["password_hash"] = hash_password(new_password, salt)
                await _atomic_write_json(USERS_PATH, data)
                return
        raise KeyError(user_id)


async def delete_user(user_id: str) -> None:
    async with _users_lock:
        data = await _read_json(USERS_PATH, {"users": []})
        kept = []
        for u in data.get("users", []):
            if u["id"] == user_id:
                if u.get("head_admin"):
                    raise ValueError("head admin cannot be deleted")
                continue
            kept.append(u)
        data["users"] = kept
        await _atomic_write_json(USERS_PATH, data)
    # Tear down any open sessions for this user.
    await delete_sessions_for_user(user_id)


# ── Sessions ────────────────────────────────────────────────────


async def list_sessions() -> list[Session]:
    async with _sessions_lock:
        data = await _read_json(SESSIONS_PATH, {"sessions": []})
    return [Session.from_storage(s) for s in data.get("sessions", [])]


async def get_session(token: str) -> Optional[Session]:
    for s in await list_sessions():
        if s.token == token:
            return s
    return None


async def create_session(
    *,
    user_id: str,
    ip: str,
    user_agent: str,
    remember: bool = True,
) -> Session:
    token = new_token()
    fingerprint = compute_fingerprint(ip, user_agent)
    now = _now_iso()
    expires = (
        datetime.now(timezone.utc) + timedelta(days=_SESSION_TTL_DAYS)
    ).replace(microsecond=0).isoformat()
    session = Session(
        token=token, user_id=user_id,
        fingerprint=fingerprint,
        ip=ip or "", user_agent=user_agent or "",
        created_at=now, last_seen=now, expires_at=expires,
        remember=remember,
    )
    async with _sessions_lock:
        data = await _read_json(SESSIONS_PATH, {"sessions": []})
        # Garbage-collect expired sessions every time we write.
        data["sessions"] = _drop_expired(data.get("sessions", []))
        data["sessions"].append(session.to_storage())
        await _atomic_write_json(SESSIONS_PATH, data)
    return session


async def touch_session(token: str) -> None:
    """Update ``last_seen`` AND extend ``expires_at`` so active sessions
    don't expire while in use.

    Rolling expiration: every touch pushes the expiry forward by
    ``_SESSION_TTL_DAYS`` days from now. Combined with the
    ``set_session_cookie`` ``max_age`` the browser also refreshes,
    this means a user who hits the app at least once a month stays
    signed in indefinitely.
    """
    async with _sessions_lock:
        data = await _read_json(SESSIONS_PATH, {"sessions": []})
        touched = False
        new_expires = (
            datetime.now(timezone.utc) + timedelta(days=_SESSION_TTL_DAYS)
        ).isoformat()
        for s in data.get("sessions", []):
            if s["token"] == token:
                s["last_seen"] = _now_iso()
                s["expires_at"] = new_expires
                touched = True
                break
        if touched:
            await _atomic_write_json(SESSIONS_PATH, data)


async def delete_session(token: str) -> None:
    async with _sessions_lock:
        data = await _read_json(SESSIONS_PATH, {"sessions": []})
        data["sessions"] = [s for s in data.get("sessions", []) if s["token"] != token]
        await _atomic_write_json(SESSIONS_PATH, data)


async def delete_sessions_for_user(user_id: str) -> None:
    async with _sessions_lock:
        data = await _read_json(SESSIONS_PATH, {"sessions": []})
        data["sessions"] = [s for s in data.get("sessions", []) if s["user_id"] != user_id]
        await _atomic_write_json(SESSIONS_PATH, data)


def _drop_expired(rows: list[dict]) -> list[dict]:
    now = datetime.now(timezone.utc)
    out = []
    for r in rows:
        try:
            exp = datetime.fromisoformat(r["expires_at"])
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            if exp > now:
                out.append(r)
        except Exception:
            # Malformed expires — drop it.
            continue
    return out


# ── Per-user settings ──────────────────────────────────────────


def _user_settings_path(user_id: str) -> str:
    return os.path.join(USER_SETTINGS_DIR, f"{user_id}.json")


async def read_user_settings(user_id: str) -> dict:
    path = _user_settings_path(user_id)
    if not os.path.isfile(path):
        return {}
    try:
        async with aiofiles.open(path, "r") as f:
            return json.loads(await f.read() or "{}")
    except Exception as e:
        logger.warning("read_user_settings(%s) failed: %s", user_id, e)
        return {}


async def write_user_settings(user_id: str, patch: dict) -> dict:
    path = _user_settings_path(user_id)
    existing = await read_user_settings(user_id)
    existing.update({k: v for k, v in (patch or {}).items() if v is not None})
    await _atomic_write_json(path, existing)
    return existing


async def delete_user_settings(user_id: str) -> None:
    path = _user_settings_path(user_id)
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning("delete_user_settings(%s) failed: %s", user_id, e)


# ── Test-only helper: reset stores ─────────────────────────────


async def _reset_for_tests() -> None:
    """Wipe all auth state. Only used in tests."""
    for p in (USERS_PATH, SESSIONS_PATH):
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass
    if os.path.isdir(USER_SETTINGS_DIR):
        for name in os.listdir(USER_SETTINGS_DIR):
            try:
                os.unlink(os.path.join(USER_SETTINGS_DIR, name))
            except Exception:
                pass
