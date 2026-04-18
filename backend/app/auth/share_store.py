"""Public share-link tokens.

A share link is an opaque token that grants read-only access to ONE
specific job (or one specific clip inside a job) without requiring the
recipient to log in. Owners and admins create / list / revoke links;
anyone with the URL can view the linked content until the token is
revoked or its expiration time passes.

Stored as ``/data/auth/share_links.json`` with the same atomic-write
+ async-lock pattern as users / sessions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiofiles

from backend.app.auth.store import AUTH_DIR

logger = logging.getLogger(__name__)


SHARE_PATH = os.path.join(AUTH_DIR, "share_links.json")
_lock = asyncio.Lock()

DEFAULT_TTL_DAYS = 365
TOKEN_BYTES = 24


@dataclass
class ShareLink:
    """One share grant.

    ``scope`` constrains what the public endpoint can return:
      * ``"job"``  — the entire job + its clips + transcript + scenes.
      * ``"clip"`` — only the named clip; the parent job's other clips
                     and transcript are not exposed.

    ``clip_id`` is required when ``scope == "clip"`` and ignored
    otherwise.
    """

    token: str
    job_id: str
    scope: str          # "job" | "clip"
    clip_id: Optional[int]
    created_by: str     # User.id of the issuer
    created_at: str
    expires_at: str
    note: str = ""

    def to_storage(self) -> dict:
        return asdict(self)

    @classmethod
    def from_storage(cls, d: dict) -> "ShareLink":
        return cls(
            token=d["token"],
            job_id=d["job_id"],
            scope=d.get("scope", "job"),
            clip_id=d.get("clip_id"),
            created_by=d.get("created_by", ""),
            created_at=d["created_at"],
            expires_at=d["expires_at"],
            note=d.get("note", ""),
        )

    def to_public(self) -> dict:
        """Same shape as ``to_storage`` but used by the API/UI — kept
        as a separate method so we can later strip fields if we ever
        need to (none today)."""
        return self.to_storage()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_in(days: int) -> str:
    return (_now() + timedelta(days=days)).replace(microsecond=0).isoformat()


def new_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


async def _read() -> dict:
    if not os.path.isfile(SHARE_PATH):
        return {"links": []}
    try:
        async with aiofiles.open(SHARE_PATH, "r") as f:
            content = await f.read()
        return json.loads(content) if content.strip() else {"links": []}
    except Exception as e:
        logger.warning("share_store read failed: %s", e)
        return {"links": []}


async def _write(data: dict) -> None:
    os.makedirs(AUTH_DIR, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=AUTH_DIR, suffix=".tmp")
    try:
        async with aiofiles.open(fd, "w", closefd=True) as f:
            await f.write(json.dumps(data, indent=2, sort_keys=True))
        os.replace(tmp, SHARE_PATH)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _is_expired(row: dict) -> bool:
    try:
        exp = datetime.fromisoformat(row["expires_at"])
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        return exp <= _now()
    except Exception:
        return True


async def list_links(*, owner_id: Optional[str] = None,
                     job_id: Optional[str] = None) -> list[ShareLink]:
    async with _lock:
        data = await _read()
    rows = [ShareLink.from_storage(r) for r in data.get("links", [])
            if not _is_expired(r)]
    if owner_id is not None:
        rows = [r for r in rows if r.created_by == owner_id]
    if job_id is not None:
        rows = [r for r in rows if r.job_id == job_id]
    rows.sort(key=lambda r: r.created_at, reverse=True)
    return rows


async def get_link(token: str) -> Optional[ShareLink]:
    async with _lock:
        data = await _read()
    for r in data.get("links", []):
        if r.get("token") == token and not _is_expired(r):
            return ShareLink.from_storage(r)
    return None


async def create_link(
    *,
    job_id: str,
    scope: str,
    clip_id: Optional[int],
    created_by: str,
    ttl_days: int = DEFAULT_TTL_DAYS,
    note: str = "",
) -> ShareLink:
    if scope not in ("job", "clip"):
        raise ValueError("scope must be 'job' or 'clip'")
    if scope == "clip" and clip_id is None:
        raise ValueError("clip_id is required when scope is 'clip'")
    link = ShareLink(
        token=new_token(),
        job_id=job_id,
        scope=scope,
        clip_id=clip_id if scope == "clip" else None,
        created_by=created_by,
        created_at=_now_iso(),
        expires_at=_iso_in(max(1, ttl_days)),
        note=(note or "").strip()[:200],
    )
    async with _lock:
        data = await _read()
        # Garbage-collect expired links on every write.
        data["links"] = [r for r in data.get("links", []) if not _is_expired(r)]
        data["links"].append(link.to_storage())
        await _write(data)
    return link


async def delete_link(token: str) -> bool:
    async with _lock:
        data = await _read()
        before = len(data.get("links", []))
        data["links"] = [r for r in data.get("links", []) if r.get("token") != token]
        after = len(data["links"])
        if after != before:
            await _write(data)
            return True
        return False


async def delete_links_for_job(job_id: str) -> None:
    async with _lock:
        data = await _read()
        data["links"] = [r for r in data.get("links", []) if r.get("job_id") != job_id]
        await _write(data)


async def _reset_for_tests() -> None:
    try:
        os.unlink(SHARE_PATH)
    except FileNotFoundError:
        pass
