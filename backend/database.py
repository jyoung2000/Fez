import json
import logging
import os
import shutil
import asyncio
import tempfile
from typing import Optional

import aiofiles

from backend.models import JobResult

logger = logging.getLogger(__name__)


def _numpy_safe_default(obj):
    """JSON serializer fallback for numpy types that slip through."""
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:
        pass
    return str(obj)


_file_locks: dict[str, asyncio.Lock] = {}


def _get_lock(job_id: str) -> asyncio.Lock:
    if job_id not in _file_locks:
        _file_locks[job_id] = asyncio.Lock()
    return _file_locks[job_id]


def _job_dir(job_id: str) -> str:
    return f"/data/uploads/{job_id}"


def _job_path(job_id: str) -> str:
    return os.path.join(_job_dir(job_id), "job.json")


async def save_job(job: JobResult) -> None:
    lock = _get_lock(job.job_id)
    async with lock:
        directory = _job_dir(job.job_id)
        os.makedirs(directory, exist_ok=True)
        path = _job_path(job.job_id)
        data = job.model_dump(mode="json")
        content = json.dumps(data, indent=2, default=_numpy_safe_default)
        # Atomic write: write to temp file then rename to prevent readers
        # from seeing a truncated/empty file during concurrent access.
        fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            async with aiofiles.open(fd, "w", closefd=True) as f:
                await f.write(content)
            os.replace(tmp_path, path)
        except BaseException:
            # Clean up temp file on any failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


async def load_job(job_id: str) -> Optional[JobResult]:
    path = _job_path(job_id)
    if not os.path.exists(path):
        return None
    lock = _get_lock(job_id)
    try:
        async with lock:
            async with aiofiles.open(path, "r") as f:
                content = await f.read()
            if not content.strip():
                logger.warning("Empty job.json for %s, treating as not found", job_id)
                return None
            data = json.loads(content)
            return JobResult(**data)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Failed to load job %s: %s", job_id, e)
        return None


async def list_jobs(
    *,
    owner_user_id: Optional[str] = None,
    owner_username: Optional[str] = None,
    include_unowned: bool = False,
) -> list[JobResult]:
    """List all jobs on disk.

    When ``owner_user_id`` is given, only that user's jobs are returned.
    ``owner_username`` is an additional stable fallback match — a job
    is included when EITHER its stored UUID matches ``owner_user_id``
    OR its stored ``owner_username`` matches (case-insensitive). The
    username match recovers jobs whose stamped UUID no longer resolves
    (e.g. the auth store was regenerated after a restart because
    ``/data/auth`` wasn't persisted as a docker volume yet).
    ``include_unowned`` additionally surfaces jobs created before
    multi-user auth shipped (``owner_user_id == ""``); this is used by
    the admin view so legacy jobs remain accessible.
    """
    jobs = []
    uploads_dir = "/data/uploads"
    if not os.path.exists(uploads_dir):
        return jobs
    norm_username = (owner_username or "").strip().lower() or None
    for entry in os.listdir(uploads_dir):
        job_path = os.path.join(uploads_dir, entry, "job.json")
        if os.path.isfile(job_path):
            try:
                async with aiofiles.open(job_path, "r") as f:
                    content = await f.read()
                data = json.loads(content)
                job = JobResult(**data)
                if owner_user_id is not None:
                    row_owner = job.owner_user_id or ""
                    row_name = (getattr(job, "owner_username", "") or "").strip().lower()
                    matches_uid = bool(row_owner) and row_owner == owner_user_id
                    matches_name = (
                        norm_username is not None
                        and bool(row_name)
                        and row_name == norm_username
                    )
                    if not (matches_uid or matches_name):
                        if not (include_unowned and row_owner == ""):
                            continue
                jobs.append(job)
            except Exception as e:
                logger.warning("Failed to load job from %s: %s", job_path, e)
                continue
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return jobs


async def delete_job(job_id: str) -> bool:
    """Delete a job and all its associated files (uploads + outputs). Returns True if deleted.

    IMPORTANT: The global media library (_library) is never deleted via this path.
    Job-specific user-uploaded media in /data/uploads/{job_id}/media/ is preserved
    by moving it to the global library before the job directory is removed, so that
    user uploads survive job deletion and can only be deleted explicitly via the
    DELETE /api/media/{id} endpoint.
    """
    directory = _job_dir(job_id)
    if not os.path.exists(directory):
        return False

    # Guard: never delete the global media library
    if job_id == "_library":
        logger.warning("Blocked attempt to delete global media library via delete_job")
        return False

    lock = _get_lock(job_id)
    async with lock:
        # Preserve user-uploaded media by moving files to the global library.
        # This ensures uploads survive job deletion and can only be removed by
        # explicit user action (DELETE /api/media/{id}).
        job_media_dir = os.path.join(directory, "media")
        if os.path.isdir(job_media_dir):
            global_media_dir = os.path.join("/data/uploads", "_library", "media")
            os.makedirs(global_media_dir, exist_ok=True)

            # Load metadata from both source and destination
            src_meta_path = os.path.join(job_media_dir, "_meta.json")
            dst_meta_path = os.path.join(global_media_dir, "_meta.json")
            src_meta = {}
            dst_meta = {}
            try:
                if os.path.isfile(src_meta_path):
                    with open(src_meta_path) as f:
                        src_meta = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
            try:
                if os.path.isfile(dst_meta_path):
                    with open(dst_meta_path) as f:
                        dst_meta = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

            # Move each media file to the global library
            for fname in os.listdir(job_media_dir):
                if fname.startswith("_"):
                    continue  # skip metadata files
                src_path = os.path.join(job_media_dir, fname)
                if not os.path.isfile(src_path):
                    continue
                dst_path = os.path.join(global_media_dir, fname)
                if not os.path.exists(dst_path):
                    try:
                        shutil.move(src_path, dst_path)
                        media_id = os.path.splitext(fname)[0]
                        if media_id in src_meta:
                            dst_meta[media_id] = src_meta[media_id]
                    except OSError:
                        pass

            # Save updated global metadata
            try:
                with open(dst_meta_path, "w") as f:
                    json.dump(dst_meta, f)
            except OSError:
                pass

        shutil.rmtree(directory, ignore_errors=True)
        # Also clean up exported clips / output files
        output_dir = f"/data/outputs/{job_id}"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
    _file_locks.pop(job_id, None)
    return True


async def update_job_thumbnail(job_id: str, thumbnail_path: str) -> Optional[JobResult]:
    """Store the thumbnail path for a job."""
    return await update_job_status(job_id, thumbnail_path=thumbnail_path)


async def get_job(job_id: str) -> Optional[JobResult]:
    """Alias for load_job, used by OG injection and share routes."""
    return await load_job(job_id)


async def update_job_status(
    job_id: str,
    status: Optional[str] = None,
    progress: Optional[int] = None,
    progress_message: Optional[str] = None,
    **kwargs,
) -> Optional[JobResult]:
    job = await load_job(job_id)
    if job is None:
        return None
    if status is not None:
        job.status = status
    if progress is not None:
        job.progress = progress
    if progress_message is not None:
        job.progress_message = progress_message
    from datetime import datetime, timezone
    job.updated_at = datetime.now(timezone.utc).isoformat()
    for key, value in kwargs.items():
        if hasattr(job, key):
            setattr(job, key, value)
    await save_job(job)
    return job
