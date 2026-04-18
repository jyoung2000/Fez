"""High-level helpers used by the /api/cloud router.

The functions in this module compose :mod:`backend.app.cloud.accounts`,
:mod:`backend.app.cloud.providers`, and
:mod:`backend.services.ingest` so the router stays thin and stateless.

Two things that live here specifically because they involve cross-module
coordination:

1. :func:`resolve_token` — returns a *valid* access token for a given
   connected account. If the stored token is expired or the provider
   returns 401 on first use, it refreshes once against the provider and
   persists the new bundle (handling Box's refresh-token rotation).
2. :func:`import_cloud_file` — streams a remote video into a
   pre-allocated job directory, calls
   :func:`backend.services.ingest.ingest_video_from_path`, and reports
   download progress on the same websocket channel the local upload
   path uses (``downloading_from_cloud`` phase).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional

from backend.config import settings
from backend.services.ingest import (
    ALLOWED_EXTENSIONS,
    IngestError,
    IngestMetadata,
    ingest_video_from_path,
)

from . import accounts
from .accounts import CloudAccount
from .base import CloudProvider, FileMeta
from .providers import cloud_storage_enabled, get_provider
from .retry import CloudProviderError

logger = logging.getLogger(__name__)


ProgressCallback = Callable[[int, Optional[int]], Awaitable[None]]
"""Callback invoked as ``progress(bytes_downloaded, total_bytes)``."""


class CloudServiceError(Exception):
    """Raised when a cloud operation cannot complete cleanly."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


# ── feature flag / provider lookup ──────────────────────────────────────────

def require_cloud_enabled() -> None:
    if not cloud_storage_enabled():
        raise CloudServiceError(
            "Cloud storage is disabled. Set CLIPAI_CLOUD_STORAGE_ENABLED=true to enable it.",
            status_code=503,
        )


def require_provider(name: str) -> CloudProvider:
    require_cloud_enabled()
    provider = get_provider(name)
    if provider is None:
        raise CloudServiceError(f"Unknown cloud provider: {name}", status_code=404)
    if not provider.is_configured():
        raise CloudServiceError(
            f"Cloud provider '{name}' is not configured on the server. "
            f"See docs/cloud-storage/SETUP.md.",
            status_code=503,
        )
    return provider


# ── token resolution ────────────────────────────────────────────────────────

def _is_expired(account: CloudAccount) -> bool:
    if not account.token_expires_at:
        return False
    try:
        expires = datetime.fromisoformat(account.token_expires_at)
    except ValueError:
        return False
    return expires <= datetime.now(timezone.utc)


async def resolve_token(account: CloudAccount) -> str:
    """Return a valid access token for *account*, refreshing if needed."""
    if not _is_expired(account):
        return account.access_token
    return await _refresh_account(account)


async def _refresh_account(account: CloudAccount) -> str:
    provider = require_provider(account.provider)
    if not account.refresh_token:
        raise CloudServiceError(
            f"Account '{account.display_name}' has no refresh token — reconnect required.",
            status_code=401,
        )
    try:
        bundle = await provider.refresh(account.refresh_token)
    except CloudProviderError as exc:
        raise CloudServiceError(
            f"Refreshing {account.provider} token failed: {exc}",
            status_code=exc.status_code or 401,
        ) from exc
    account.access_token = bundle.access_token
    if bundle.refresh_token:
        account.refresh_token = bundle.refresh_token
    account.token_expires_at = (
        bundle.expires_at.isoformat() if bundle.expires_at else None
    )
    await accounts.save(account)
    return account.access_token


async def call_with_refresh(
    account: CloudAccount,
    op: Callable[[str], Awaitable],
):
    """Run *op(access_token)*, refreshing once on a 401 and retrying."""
    token = await resolve_token(account)
    try:
        return await op(token)
    except CloudProviderError as exc:
        if exc.status_code != 401:
            raise
        token = await _refresh_account(account)
        return await op(token)


# ── import orchestration ────────────────────────────────────────────────────

def _sanitize_filename(raw: str) -> str:
    """Strip path separators and pick a sane extension.

    Providers can technically return filenames with slashes, weird
    control characters, or no extension at all. We keep the basename,
    scrub obvious garbage, and fall back to ``.mp4`` if the extension
    isn't in our allow-list.
    """
    if not raw:
        return "video.mp4"
    basename = os.path.basename(raw).strip() or "video.mp4"
    # Replace anything that isn't a sensible filename char.
    basename = re.sub(r"[\x00-\x1f\\/]+", "_", basename)
    if "." in basename:
        stem, ext = basename.rsplit(".", 1)
        if ext.lower() in ALLOWED_EXTENSIONS:
            return f"{stem}.{ext.lower()}"
    return f"{basename}.mp4"


def _initial_temp_path(job_id: str, filename: str) -> tuple[str, str]:
    """Pre-allocate the job dir and return (tmp_path, final_path)."""
    job_dir = os.path.join("/data/uploads", job_id)
    os.makedirs(job_dir, exist_ok=True)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "mp4"
    final_path = os.path.join(job_dir, f"video.{ext}")
    tmp_path = os.path.join(job_dir, f"video.cloud-download.{ext}")
    return tmp_path, final_path


async def import_cloud_file(
    account: CloudAccount,
    file_id: str,
    *,
    metadata: IngestMetadata,
    progress: Optional[ProgressCallback] = None,
    pending_job_id: Optional[str] = None,
) -> str:
    """Stream *file_id* from the cloud into a new job and start analysis.

    Returns the job_id once the file is on disk and the ingest entry
    point has been called (which in turn schedules the pipeline).

    The function is intended to be awaited from a ``BackgroundTasks``
    instance or an ``asyncio.create_task`` so the HTTP response can
    land before the download completes; the cloud router wires it that
    way and passes ``pending_job_id`` so the response can include the
    id before streaming begins.
    """
    provider = require_provider(account.provider)

    # Pre-resolve the file metadata so we have name + size before we
    # start streaming.
    try:
        meta: FileMeta = await call_with_refresh(
            account, lambda token: provider.get_file_metadata(token, file_id)
        )
    except CloudProviderError as exc:
        raise CloudServiceError(
            f"Failed to load metadata for file {file_id}: {exc}",
            status_code=exc.status_code or 502,
        ) from exc

    filename = _sanitize_filename(meta.name)
    total_bytes = meta.size or 0
    job_id = pending_job_id or str(uuid.uuid4())
    tmp_path, _final_path = _initial_temp_path(job_id, filename)

    logger.info(
        "Starting cloud import: provider=%s file_id=%s filename=%s size=%s job_id=%s",
        account.provider,
        file_id,
        filename,
        total_bytes or "?",
        job_id,
    )

    # Stream the file to tmp_path. We deliberately do not hold the whole
    # thing in memory — these are videos and they can easily be 10+ GB.
    downloaded = 0
    try:
        token = await resolve_token(account)
        retried = False
        while True:
            try:
                async with _open_async_output(tmp_path) as handle:
                    async for chunk in await provider.open_stream(token, file_id):
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if progress is not None:
                            await progress(downloaded, total_bytes or None)
                break
            except CloudProviderError as exc:
                if exc.status_code == 401 and not retried:
                    logger.info(
                        "Cloud stream returned 401 — refreshing token and retrying once"
                    )
                    token = await _refresh_account(account)
                    downloaded = 0
                    retried = True
                    continue
                raise
    except CloudProviderError as exc:
        _cleanup(tmp_path)
        raise CloudServiceError(
            f"Cloud download failed: {exc}",
            status_code=exc.status_code or 502,
        ) from exc
    except Exception:
        _cleanup(tmp_path)
        raise

    # Hand off to the shared ingest entry point. This is the single
    # convergence point between cloud and local uploads.
    try:
        return await ingest_video_from_path(
            tmp_path,
            filename=filename,
            file_size_bytes=downloaded,
            metadata=metadata,
            job_id=job_id,
            move_into_job_dir=True,
        )
    except IngestError as exc:
        _cleanup(tmp_path)
        raise CloudServiceError(str(exc), status_code=exc.status_code) from exc


# ── tiny helpers ────────────────────────────────────────────────────────────

def _cleanup(path: str) -> None:
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


class _AsyncFileWriter:
    """Minimal context manager that writes through ``asyncio.to_thread``."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._handle = None

    async def __aenter__(self) -> "_AsyncFileWriter":
        self._handle = await asyncio.to_thread(open, self._path, "wb")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            await asyncio.to_thread(self._handle.close)
            self._handle = None

    def write(self, chunk: bytes) -> None:
        if self._handle is None:
            raise RuntimeError("writer used outside 'async with' block")
        self._handle.write(chunk)


def _open_async_output(path: str) -> _AsyncFileWriter:
    return _AsyncFileWriter(path)
