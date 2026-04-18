"""Shared ingestion entry point used by local + cloud uploads.

Before cloud storage was added, the "file is on disk, create a JobResult,
run the pipeline" tail existed twice: once in
:mod:`backend.routers.chunked_upload` and once in
:mod:`backend.routers.upload`. Both copies were nearly identical.

This module consolidates that tail so that:

1. The chunked and non-chunked local uploads share one code path.
2. The cloud-storage importer converges on the same function, which keeps
   the "single ingest entry point" non-negotiable in the cloud-storage
   spec honest — no behavior can diverge between local and cloud uploads.

The helper is deliberately small: it validates the extension and file
header, moves the file into ``/data/uploads/<job_id>/video.<ext>``,
constructs a :class:`~backend.models.JobResult`, persists it, and (if
``settings.AUTO_ANALYZE`` is on) schedules
:func:`backend.services.pipeline.run_analysis`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from backend import database
from backend.config import settings
from backend.models import JobResult, JobStatus
from backend.services.video_validation import (
    validate_video_header as _validate_video_header,
)


def _run_analysis_lazy(job_id: str):
    """Lazy import of ``run_analysis`` so importing this module does not
    drag in the entire analysis pipeline (which pulls PIL, torch, etc.).

    This is used both at call time below and as a test override target —
    ``monkeypatch.setattr(ingest, 'run_analysis', ...)`` is what the
    cloud import test uses to intercept the analysis kick-off.
    """
    from backend.services.pipeline import run_analysis  # noqa: WPS433 - deliberate local import

    return run_analysis(job_id)


# Exposed as a module-level attribute so tests can monkey-patch it
# cleanly. The production ``ingest_video_from_path`` awaits
# ``run_analysis(job_id)``.
run_analysis = _run_analysis_lazy

logger = logging.getLogger(__name__)


ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm", "m4v", "3gp"}
UPLOAD_DIR = "/data/uploads"


class IngestError(Exception):
    """Raised when a video cannot be ingested (bad extension, header, etc.)."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class IngestMetadata:
    """All the metadata the pipeline needs to start analyzing a video.

    These fields mirror the form fields the chunked upload accepts today,
    so every ingest entry point hands the pipeline an identical shape.
    """

    language: str = ""
    subtitle_language: str = ""
    content_type_override: str = ""
    game_type: str = ""
    anime_subtype: str = ""
    music_subtype: str = ""
    sports_subtype: str = ""
    # Free-form source tag, for observability/telemetry: "local", "chunked",
    # "google_drive", "box".
    source: str = "local"
    # Owner of the resulting job (``User.id``). Empty = legacy ingest
    # path that bypassed auth; those jobs are admin-only visible.
    owner_user_id: str = ""
    # Username snapshot at ingest time. Used as a stable fallback when
    # the ``owner_user_id`` UUID becomes stale — e.g. after an auth
    # store regeneration following a missing ``/data/auth`` volume mount.
    owner_username: str = ""
    # Optional extra fields to merge onto the JobResult (e.g. provider_used).
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_form(
        cls,
        *,
        language: str = "",
        subtitle_language: str = "",
        content_type_override: str = "",
        game_type: str = "",
        anime_subtype: str = "",
        music_subtype: str = "",
        sports_subtype: str = "",
        source: str = "local",
        owner_user_id: str = "",
        owner_username: str = "",
    ) -> "IngestMetadata":
        return cls(
            language=(language or "").strip().lower(),
            subtitle_language=(subtitle_language or "").strip().lower(),
            content_type_override=(content_type_override or "").strip().lower(),
            game_type=(game_type or "").strip().lower(),
            anime_subtype=(anime_subtype or "").strip().lower(),
            music_subtype=(music_subtype or "").strip().lower(),
            sports_subtype=(sports_subtype or "").strip().lower(),
            source=source,
            owner_user_id=(owner_user_id or "").strip(),
            owner_username=(owner_username or "").strip(),
        )


def _extension_of(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _move_or_copy(src: str, dst: str) -> None:
    """Move ``src`` to ``dst``, falling back to copy+unlink across devices."""
    try:
        os.rename(src, dst)
    except OSError:
        # Cross-device or FUSE/Unraid parity path — fall back to shutil.
        shutil.move(src, dst)


async def ingest_video_from_path(
    source_path: str,
    *,
    filename: str,
    file_size_bytes: int,
    metadata: IngestMetadata,
    job_id: Optional[str] = None,
    move_into_job_dir: bool = True,
) -> str:
    """Ingest a video that is already on the server's filesystem.

    Parameters
    ----------
    source_path:
        Current on-disk location of the video. When ``move_into_job_dir`` is
        true (the default) this file is *moved* to the canonical job path.
    filename:
        Original filename, used for extension inference and UI display.
    file_size_bytes:
        Size in bytes, used for reporting. Not re-read from disk because
        streaming importers already know it.
    metadata:
        UI + provenance metadata. See :class:`IngestMetadata`.
    job_id:
        Optional explicit job id; otherwise a fresh UUID is generated. The
        cloud importer pre-allocates a job id so it can expose it via the
        ``/import`` response before the download finishes.
    move_into_job_dir:
        Set to False only if the caller has already placed the video at
        ``/data/uploads/<job_id>/video.<ext>`` itself (the chunked upload
        path does this). The function still runs header validation and
        creates the job row.

    Returns
    -------
    str
        The ``job_id`` that now has a queued :class:`JobResult` row and,
        if ``settings.AUTO_ANALYZE`` is on, a running analysis task.

    Raises
    ------
    IngestError
        If the extension is unsupported or the header validation fails.
        The HTTP status code to surface is attached to the exception.
    """
    ext = _extension_of(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise IngestError(
            f"Unsupported format .{ext}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            status_code=400,
        )

    if job_id is None:
        job_id = str(uuid.uuid4())
    job_dir = os.path.join(UPLOAD_DIR, job_id)
    await asyncio.to_thread(os.makedirs, job_dir, exist_ok=True)

    final_path = os.path.join(job_dir, f"video.{ext}")

    if move_into_job_dir and os.path.abspath(source_path) != os.path.abspath(final_path):
        await asyncio.to_thread(_move_or_copy, source_path, final_path)

    # Header validation is the same check the old local-upload path ran.
    header_err = await asyncio.to_thread(_validate_video_header, final_path, ext)
    if header_err:
        # Clean up the bad file so we don't leave orphaned job dirs.
        try:
            os.remove(final_path)
        except OSError:
            pass
        try:
            os.rmdir(job_dir)
        except OSError:
            pass
        raise IngestError(header_err, status_code=422)

    now = datetime.now(timezone.utc).isoformat()
    job = JobResult(
        job_id=job_id,
        filename=filename,
        file_path=final_path,
        owner_user_id=metadata.owner_user_id,
        owner_username=metadata.owner_username,
        file_size_mb=round(file_size_bytes / (1024 * 1024), 2),
        language=metadata.language,
        subtitle_language=metadata.subtitle_language,
        content_type_override=metadata.content_type_override,
        game_type=metadata.game_type,
        anime_subtype=metadata.anime_subtype,
        music_subtype=metadata.music_subtype,
        sports_subtype=metadata.sports_subtype,
        status=JobStatus.QUEUED,
        progress=0,
        progress_message="Uploaded, waiting for analysis",
        created_at=now,
        updated_at=now,
    )
    await database.save_job(job)

    logger.info(
        "Ingested video for job %s: %s (%.1f MB, source=%s)",
        job_id,
        final_path,
        job.file_size_mb,
        metadata.source,
    )

    if settings.AUTO_ANALYZE:
        job.progress_message = "Analysis starting..."
        await database.save_job(job)
        asyncio.create_task(run_analysis(job_id))

    # Warm up the browser-friendly preview in the background. For
    # ``.mkv`` / non-AAC sources the HTML5 ``<video>`` element needs
    # a transcoded derivative to play audio — we generate it
    # lazily on first request, but doing it here too means the
    # owner's own preview is already ready by the time analysis
    # finishes (preview build takes ~30-60s, analysis takes longer).
    # Fire-and-forget: failures just fall back to the on-demand
    # generator wired into the serve-file / share endpoints.
    try:
        from backend.services.browser_preview import ensure_browser_preview_async
        asyncio.create_task(ensure_browser_preview_async(final_path))
    except Exception as e:  # pragma: no cover — purely cosmetic warmup
        logger.debug("ingest: could not schedule preview warmup: %s", e)

    return job_id
