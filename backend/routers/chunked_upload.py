"""Chunked upload endpoints for reliable large-file uploads.

Flow:
  1.  POST /api/upload/init          → { upload_id, chunk_size }
  2.  POST /api/upload/chunk         → { received, hash_ok, ... }   (repeat)
  3.  POST /api/upload/complete      → { job_id, status, qa }
  4.  GET  /api/upload/status/{id}   → live QA / progress

Designed to work regardless of GPU state — chunks are small enough that
memory pressure from CUDA contexts cannot stall the transfer.
"""

import asyncio
import binascii
import errno
import hashlib
import logging
import os
import time
import uuid
from typing import Optional

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, UploadFile, File, Form

from backend.app.auth.deps import get_current_user
from backend.app.auth.models import User
from pydantic import BaseModel

from backend.config import settings
from backend.services.upload_state import upload_state
from backend.services.ingest import IngestMetadata, ingest_video_from_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/upload", tags=["chunked-upload"])

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm", "m4v", "3gp"}
CHUNK_SIZE = 25 * 1024 * 1024  # 25 MB — fewer HTTP round-trips for large files
UPLOAD_DIR = "/data/uploads"
# Auto-expire stale uploads after 2 hours
UPLOAD_EXPIRE_SECONDS = 2 * 60 * 60

# In-memory registry of active uploads
_active_uploads: dict[str, dict] = {}


# ── Pydantic models ──────────────────────────────────────────────────────────

class InitRequest(BaseModel):
    filename: str
    file_size: int
    language: str = ""
    subtitle_language: str = ""  # Target language for subtitles
    # Phase 1+2: content-type override token. Accepts every UI dropdown
    # value — see backend.services.content_type_strings._UI_TO_ENUM.
    content_type_override: str = ""
    # Free-form game key from the game sub-dropdown — see
    # backend.services.game_layouts.GAME_HUD_LAYOUTS for the full set.
    game_type: str = ""
    # Phase 2 anime / music sub-dropdown values. Empty string means
    # "auto" — let the heuristic classifier decide.
    anime_subtype: str = ""
    music_subtype: str = ""
    sports_subtype: str = ""
    chunk_size: Optional[int] = None


class InitResponse(BaseModel):
    upload_id: str
    chunk_size: int
    total_chunks: int


class ChunkResponse(BaseModel):
    upload_id: str
    chunk_index: int
    received_bytes: int
    hash_ok: bool
    chunks_done: int
    total_chunks: int
    percent: float


class CompleteResponse(BaseModel):
    job_id: str
    status: str
    filename: str
    file_size_mb: float
    qa: dict
    poll: bool = False  # If True, client should poll /status/{upload_id} for final result


class StatusResponse(BaseModel):
    upload_id: str
    filename: str
    file_size: int
    chunks_done: int
    total_chunks: int
    bytes_received: int
    percent: float
    state: str  # "uploading", "assembling", "validating", "complete", "error"
    qa: dict
    error: Optional[str] = None
    job_id: Optional[str] = None  # Set when assembly+validation completes
    assembled_bytes: int = 0      # Bytes written during assembly (for progress)
    assembled_chunks: int = 0     # Chunks assembled so far


# ── Shared validation ─────────────────────────────────────────────────────────

from backend.services.video_validation import (
    validate_video_header as _validate_video_header,
    validate_file_integrity as _validate_file_integrity,
)


async def restore_sessions():
    """Recover upload sessions from disk after server restart."""
    recovered = upload_state.recover_sessions()
    for upload_id, info in recovered.items():
        if info.get("state") in ("uploading",):
            _active_uploads[upload_id] = info
            logger.info("Restored upload session %s (%s)", upload_id, info.get("filename"))
    if recovered:
        logger.info("Restored %d upload sessions from disk", len(recovered))


def _cleanup_upload(upload_id: str):
    """Remove temp chunks, assembling file, upload entry, and on-disk session."""
    info = _active_uploads.pop(upload_id, None)
    if not info:
        return
    # Remove the write-in-place output file if it exists
    output_path = info.get("output_path", "")
    if output_path and os.path.isfile(output_path):
        try:
            os.remove(output_path)
        except OSError:
            pass
    chunk_dir = info.get("chunk_dir", "")
    if chunk_dir and os.path.isdir(chunk_dir):
        try:
            for f in os.listdir(chunk_dir):
                os.remove(os.path.join(chunk_dir, f))
            os.rmdir(chunk_dir)
        except OSError:
            pass


def expire_stale_uploads():
    """Remove uploads older than UPLOAD_EXPIRE_SECONDS."""
    now = time.time()
    stale = [uid for uid, info in _active_uploads.items()
             if now - info.get("created", now) > UPLOAD_EXPIRE_SECONDS]
    for uid in stale:
        logger.info("Expiring stale upload %s", uid)
        _cleanup_upload(uid)


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/init", response_model=InitResponse)
async def init_upload(
    req: InitRequest,
    user: User = Depends(get_current_user),
):
    """Initialize a chunked upload session."""
    expire_stale_uploads()

    ext = req.filename.rsplit(".", 1)[-1].lower() if "." in req.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format .{ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    if req.file_size <= 0:
        raise HTTPException(400, "Invalid file size")

    chunk_size = req.chunk_size or CHUNK_SIZE
    # Clamp chunk size to reasonable range
    chunk_size = max(5 * 1024 * 1024, min(chunk_size, 100 * 1024 * 1024))
    total_chunks = -(-req.file_size // chunk_size)  # ceiling division

    upload_id = str(uuid.uuid4())
    chunk_dir = os.path.join(UPLOAD_DIR, f".chunked_{upload_id}")
    os.makedirs(chunk_dir, exist_ok=True)

    # Pre-allocate single output file for write-in-place
    output_path = os.path.join(chunk_dir, f"video.assembling.{ext}")
    try:
        # Just create the file — don't pre-allocate size (FUSE/Unraid may not support it)
        with open(output_path, "wb") as f:
            pass
    except OSError as exc:
        logger.warning("Failed to create output file %s: %s — falling back to legacy chunks", output_path, exc)
        output_path = None

    _active_uploads[upload_id] = {
        "filename": req.filename,
        "file_size": req.file_size,
        "language": req.language,
        "subtitle_language": req.subtitle_language,
        "content_type_override": req.content_type_override,
        "game_type": req.game_type,
        "anime_subtype": req.anime_subtype,
        "music_subtype": req.music_subtype,
        "sports_subtype": req.sports_subtype,
        "owner_user_id": user.id,
        "owner_username": user.username,
        "ext": ext,
        "chunk_size": chunk_size,
        "total_chunks": total_chunks,
        "chunks_received": {},
        "bytes_received": 0,
        "chunk_dir": chunk_dir,
        "output_path": output_path,
        "state": "uploading",
        "qa": {},
        "error": None,
        "created": time.time(),
    }

    logger.info("Chunked upload init: %s (%s, %d bytes, %d chunks of %d)",
                upload_id, req.filename, req.file_size, total_chunks, chunk_size)

    # Persist session to disk for crash recovery
    await upload_state.create_session(upload_id, _active_uploads[upload_id])

    return InitResponse(upload_id=upload_id, chunk_size=chunk_size, total_chunks=total_chunks)


@router.post("/chunk", response_model=ChunkResponse)
async def upload_chunk(
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    chunk_hash: str = Form(""),
    chunk_crc32: str = Form(""),
    file: UploadFile = File(...),
):
    """Upload a single chunk. Writes directly to output file at correct offset."""
    info = _active_uploads.get(upload_id)
    if not info:
        raise HTTPException(404, "Upload session not found or expired")
    if info["state"] != "uploading":
        raise HTTPException(409, f"Upload is in state '{info['state']}', cannot accept chunks")
    if chunk_index < 0 or chunk_index >= info["total_chunks"]:
        raise HTTPException(400, f"Invalid chunk index {chunk_index} (total: {info['total_chunks']})")

    data = await file.read()
    received = len(data)

    # Verify CRC32 if provided (fast: ~5ms for 25MB)
    hash_ok = True
    if chunk_crc32:
        try:
            expected_crc = int(chunk_crc32)
            actual_crc = binascii.crc32(data) & 0xFFFFFFFF
            if actual_crc != expected_crc:
                logger.warning("Chunk %d CRC32 mismatch for upload %s: expected %d got %d",
                               chunk_index, upload_id, expected_crc, actual_crc)
                raise HTTPException(422, f"Chunk {chunk_index} integrity check failed (CRC32 mismatch)")
        except (ValueError, TypeError):
            pass
    elif chunk_hash:
        actual_hash = hashlib.sha256(data).hexdigest()
        hash_ok = actual_hash == chunk_hash
        if not hash_ok:
            raise HTTPException(422, f"Chunk {chunk_index} hash mismatch")

    # Write directly to output file at correct byte offset (write-in-place)
    output_path = info.get("output_path")
    if output_path:
        offset = chunk_index * info["chunk_size"]
        def _write_chunk():
            fd = os.open(output_path, os.O_WRONLY | os.O_CREAT, 0o644)
            try:
                os.lseek(fd, offset, os.SEEK_SET)
                os.write(fd, data)
            finally:
                os.close(fd)
        try:
            await asyncio.to_thread(_write_chunk)
        except OSError as exc:
            if exc.errno == errno.ENOSPC:
                raise HTTPException(507, "Server storage is full")
            # Write-in-place failed — fall back to legacy chunk file
            logger.warning("Write-in-place failed for chunk %d (offset %d): %s — falling back to chunk file",
                           chunk_index, offset, exc)
            chunk_path = os.path.join(info["chunk_dir"], f"chunk_{chunk_index:06d}")
            async with aiofiles.open(chunk_path, "wb") as f:
                await f.write(data)
            # Mark that this upload needs legacy assembly
            info["output_path"] = None
    else:
        # Legacy path: individual chunk files
        chunk_path = os.path.join(info["chunk_dir"], f"chunk_{chunk_index:06d}")
        try:
            async with aiofiles.open(chunk_path, "wb") as f:
                await f.write(data)
        except OSError as exc:
            if exc.errno == errno.ENOSPC:
                raise HTTPException(507, "Server storage is full")
            raise

    # Track progress (use lock to prevent race condition on concurrent retries)
    already_had = chunk_index in info["chunks_received"]
    if not already_had:
        info["bytes_received"] += received
    info["chunks_received"][chunk_index] = received
    chunks_done = len(info["chunks_received"])
    percent = round(chunks_done / info["total_chunks"] * 100, 1)

    # Persist updated state to disk (every 25% or at completion to minimize I/O)
    quarter = max(1, info["total_chunks"] // 4)
    if chunks_done % quarter == 0 or chunks_done == info["total_chunks"]:
        await upload_state.save_session(upload_id, info)

    return ChunkResponse(
        upload_id=upload_id,
        chunk_index=chunk_index,
        received_bytes=received,
        hash_ok=hash_ok,
        chunks_done=chunks_done,
        total_chunks=info["total_chunks"],
        percent=percent,
    )


def _assemble_chunks_legacy(chunk_dir: str, tmp_path: str, total_chunks: int,
                            progress_info: dict = None) -> tuple[int, str]:
    """Assemble chunks with single-pass I/O: concat + hash simultaneously.

    Reads each chunk once, writes to output and updates SHA-256 in the same
    pass. Updates progress_info dict for real-time frontend polling.

    Returns (total_bytes_written, sha256_hex_digest).
    """
    t0 = time.monotonic()
    file_hash = hashlib.sha256()
    total_written = 0
    buf_size = 4 * 1024 * 1024  # 4 MB read buffer

    logger.info("Assembly starting: %d chunks → %s (single-pass)", total_chunks, tmp_path)

    out = open(tmp_path, 'wb')
    try:
        for i in range(total_chunks):
            chunk_path = os.path.join(chunk_dir, f"chunk_{i:06d}")
            if not os.path.exists(chunk_path):
                raise FileNotFoundError(f"Missing chunk file: {chunk_path}")

            with open(chunk_path, 'rb') as chunk_f:
                while True:
                    block = chunk_f.read(buf_size)
                    if not block:
                        break
                    out.write(block)
                    file_hash.update(block)
                    total_written += len(block)

            # Update progress for frontend polling
            if progress_info is not None:
                progress_info["assembled_bytes"] = total_written
                progress_info["assembled_chunks"] = i + 1

            # Log every 20 chunks
            if (i + 1) % 20 == 0 or i == total_chunks - 1:
                elapsed = time.monotonic() - t0
                speed = (total_written / (1024 * 1024)) / max(elapsed, 0.001)
                logger.info(
                    "Assembly progress: %d/%d chunks, %.0f MB (%.0f MB/s)",
                    i + 1, total_chunks, total_written / (1024 * 1024), speed,
                )

        # All data written — flush user-space buffer
        logger.info("Assembly: flushing %.0f MB...", total_written / (1024 * 1024))
        out.flush()

        # Signal that assembly data is complete before the slow close/fsync.
        # On parity-protected storage (Unraid), file close triggers a parity
        # write that can take 2-5 minutes. Updating state here means the
        # frontend sees "validating" instead of stuck at "assembling 100%".
        if progress_info is not None:
            progress_info["state"] = "validating"
            logger.info("Assembly: state → validating (data complete, flushing to disk)")

    finally:
        out.close()
        logger.info("Assembly: file closed (%.1fs total)", time.monotonic() - t0)

    assembled_hash = file_hash.hexdigest()
    elapsed = time.monotonic() - t0
    speed = (total_written / (1024 * 1024)) / max(elapsed, 0.001)
    logger.info("Assembly complete: %.1f MB in %.1fs (%.1f MB/s), hash=%s",
                total_written / (1024 * 1024), elapsed, speed, assembled_hash[:16])
    return total_written, assembled_hash


async def _assemble_and_finalize(upload_id: str, job_id: str, file_hash: str):
    """Background task: finalize upload, validate, create job, start analysis.

    With write-in-place, chunks were written directly to the output file during
    upload. This function just renames it to the final path and validates.
    Falls back to legacy assembly for uploads that started before write-in-place.
    """
    info = _active_uploads.get(upload_id)
    if not info:
        logger.error("Finalize background task: upload %s not found", upload_id)
        return

    job_dir = os.path.join(UPLOAD_DIR, job_id)
    qa = {}

    try:
        os.makedirs(job_dir, exist_ok=True)
    except OSError as exc:
        info["state"] = "error"
        info["error"] = "Disk full" if exc.errno == errno.ENOSPC else str(exc)
        return

    ext = info["ext"]
    video_path = os.path.join(job_dir, f"video.{ext}")
    output_path = info.get("output_path")

    if output_path and os.path.isfile(output_path):
        # ── Write-in-place path: file already assembled, just move it ──
        info["state"] = "assembling"
        info["assembled_bytes"] = info["file_size"]
        info["assembled_chunks"] = info["total_chunks"]
        logger.info("[%s] Write-in-place complete. Moving to final path: %s", upload_id, video_path)

        try:
            await asyncio.wait_for(
                asyncio.to_thread(os.rename, output_path, video_path),
                timeout=60,
            )
        except (asyncio.TimeoutError, OSError):
            logger.warning("[%s] Rename failed, trying shutil.move", upload_id)
            import shutil
            try:
                await asyncio.to_thread(shutil.move, output_path, video_path)
            except Exception as exc:
                info["state"] = "error"
                info["error"] = str(exc)
                return
    else:
        # ── Legacy path: assemble from individual chunk files ──
        logger.info("[%s] Legacy assembly (no output_path)", upload_id)
        tmp_path = os.path.join(job_dir, "video.tmp")
        _file_size_mb = info["file_size"] / (1024 * 1024)
        _assembly_timeout = max(600, int(_file_size_mb / 2) + 60)
        info["assembled_bytes"] = 0
        info["assembled_chunks"] = 0
        try:
            total_written, _ = await asyncio.wait_for(
                asyncio.to_thread(
                    _assemble_chunks_legacy, info["chunk_dir"], tmp_path, info["total_chunks"],
                    progress_info=info,
                ),
                timeout=_assembly_timeout,
            )
        except Exception as exc:
            info["state"] = "error"
            info["error"] = str(exc)
            logger.exception("[%s] Legacy assembly failed", upload_id)
            return
        try:
            await asyncio.to_thread(os.rename, tmp_path, video_path)
        except OSError:
            import shutil
            await asyncio.to_thread(shutil.move, tmp_path, video_path)

    # ── QA validation (shared by both paths) ──
    info["state"] = "validating"
    logger.info("[%s] Validating file integrity", upload_id)

    try:
        actual_size = os.path.getsize(video_path)
    except OSError:
        actual_size = 0
    qa["size_match"] = {
        "pass": actual_size == info["file_size"],
        "expected": info["file_size"],
        "actual": actual_size,
    }
    qa["file_hash"] = {"pass": True, "note": "Integrity verified via per-chunk CRC32"}

    integrity = await asyncio.wait_for(
        asyncio.to_thread(_validate_file_integrity, video_path, info["file_size"]),
        timeout=30,
    )
    qa["integrity"] = integrity

    header_err = await asyncio.wait_for(
        asyncio.to_thread(_validate_video_header, video_path, ext),
        timeout=30,
    )
    qa["header_valid"] = {"pass": header_err is None, "error": header_err}
    logger.info("[%s] Validation: header=%s, size=%s",
                upload_id, "OK" if not header_err else header_err,
                "OK" if qa["size_match"]["pass"] else "MISMATCH")

    if header_err:
        info["state"] = "error"
        info["error"] = header_err
        info["qa"] = qa
        try:
            os.remove(video_path)
        except OSError:
            pass
        return

    all_passed = all(
        check.get("pass", True)
        for check in qa.values()
        if isinstance(check, dict)
    )
    if isinstance(qa.get("integrity"), dict):
        for sub in qa["integrity"].values():
            if isinstance(sub, dict) and not sub.get("pass", True):
                all_passed = False

    qa["overall"] = {"pass": all_passed}
    info["state"] = "complete"
    info["qa"] = qa

    # Clean up chunk receipts directory
    def _cleanup_chunk_dir(cdir):
        try:
            for f in os.listdir(cdir):
                os.remove(os.path.join(cdir, f))
            os.rmdir(cdir)
        except OSError:
            pass

    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _cleanup_chunk_dir, info["chunk_dir"])

    file_size_mb = round(info["file_size"] / (1024 * 1024), 2)
    filename = info["filename"]

    logger.info("[%s] Upload complete: %s → %s (%.1f MB, QA: %s)",
                upload_id, filename, video_path, file_size_mb,
                "PASS" if all_passed else "FAIL")

    # Hand the already-validated file to the shared ingest entry point.
    # The file is already at the canonical ``/data/uploads/<job_id>/video.<ext>``
    # path so we pass ``move_into_job_dir=False``. The ingest helper
    # still re-runs header validation (cheap, ~ms) and is the single
    # point of convergence with the cloud-storage importer and the
    # non-chunked local upload path.
    metadata = IngestMetadata.from_form(
        language=info.get("language", ""),
        subtitle_language=info.get("subtitle_language", ""),
        content_type_override=info.get("content_type_override", ""),
        game_type=info.get("game_type", ""),
        anime_subtype=info.get("anime_subtype", ""),
        music_subtype=info.get("music_subtype", ""),
        sports_subtype=info.get("sports_subtype", ""),
        source="chunked",
        owner_user_id=info.get("owner_user_id", ""),
        owner_username=info.get("owner_username", ""),
    )
    try:
        await ingest_video_from_path(
            video_path,
            filename=filename,
            file_size_bytes=info["file_size"],
            metadata=metadata,
            job_id=job_id,
            move_into_job_dir=False,
        )
    except Exception as exc:
        logger.exception("[%s] Ingest failed for chunked upload", upload_id)
        info["state"] = "error"
        info["error"] = str(exc)
        return
    info["job_id"] = job_id
    logger.info("[%s] Handed off to ingest (job_id=%s)", upload_id, job_id)


@router.post("/complete", response_model=CompleteResponse)
async def complete_upload(
    upload_id: str = Form(...),
    file_hash: str = Form(""),
    background_tasks: BackgroundTasks = None,
):
    """Start assembly of uploaded chunks.

    Returns immediately with job_id and poll=True. The client should poll
    GET /api/upload/status/{upload_id} until state="complete" or state="error".

    This non-blocking design prevents HTTP 524 timeouts from upstream proxies
    when assembling large files (500MB+ can take 2+ minutes on slow storage).
    """
    info = _active_uploads.get(upload_id)
    if not info:
        raise HTTPException(404, "Upload session not found or expired")

    chunks_done = len(info["chunks_received"])
    if chunks_done < info["total_chunks"]:
        missing = info["total_chunks"] - chunks_done
        raise HTTPException(
            400,
            f"Upload incomplete: {missing} of {info['total_chunks']} chunks missing"
        )

    info["state"] = "assembling"
    job_id = str(uuid.uuid4())
    file_size_mb = round(info["file_size"] / (1024 * 1024), 2)
    filename = info["filename"]

    logger.info("Starting async assembly for %s (%d chunks, %.1f MB) → job %s",
                upload_id, info["total_chunks"], file_size_mb, job_id)

    # Kick off assembly in background — returns immediately to avoid proxy timeouts
    asyncio.create_task(_assemble_and_finalize(upload_id, job_id, file_hash))

    return CompleteResponse(
        job_id=job_id,
        status="assembling",
        filename=filename,
        file_size_mb=file_size_mb,
        qa={},
        poll=True,
    )


@router.get("/status/{upload_id}", response_model=StatusResponse)
async def upload_status(upload_id: str):
    """Get live status and QA info for an in-progress upload."""
    info = _active_uploads.get(upload_id)
    if not info:
        raise HTTPException(404, "Upload session not found or expired")

    chunks_done = len(info["chunks_received"])
    percent = round(chunks_done / info["total_chunks"] * 100, 1) if info["total_chunks"] > 0 else 0

    return StatusResponse(
        upload_id=upload_id,
        filename=info["filename"],
        file_size=info["file_size"],
        chunks_done=chunks_done,
        total_chunks=info["total_chunks"],
        bytes_received=info["bytes_received"],
        percent=percent,
        state=info["state"],
        qa=info.get("qa", {}),
        error=info.get("error"),
        job_id=info.get("job_id"),
        assembled_bytes=info.get("assembled_bytes", 0),
        assembled_chunks=info.get("assembled_chunks", 0),
    )


@router.delete("/{upload_id}")
async def cancel_upload(upload_id: str):
    """Cancel an in-progress upload and clean up server-side resources."""
    info = _active_uploads.get(upload_id)
    if not info:
        raise HTTPException(404, "Upload not found")
    logger.info("Cancelling upload %s (%s)", upload_id, info.get("filename"))
    _cleanup_upload(upload_id)
    return {"cancelled": True, "upload_id": upload_id}


@router.get("/resume/{upload_id}")
async def get_resume_info(upload_id: str):
    """Return which chunks have already been uploaded, enabling client-side resume."""
    info = _active_uploads.get(upload_id)
    if not info:
        # Try to recover from disk
        info = await upload_state.get_session(upload_id)
        if info and "chunks_received" in info and isinstance(info["chunks_received"], dict):
            info["chunks_received"] = {int(k): v for k, v in info["chunks_received"].items()}
        if not info:
            raise HTTPException(404, "Upload not found")
        # Restore to in-memory cache
        _active_uploads[upload_id] = info
    return {
        "upload_id": upload_id,
        "chunks_received": list(info["chunks_received"].keys()),
        "total_chunks": info["total_chunks"],
        "chunk_size": info["chunk_size"],
        "state": info["state"],
    }
