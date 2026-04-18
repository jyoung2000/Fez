"""Media upload endpoint for the multi-track video editor."""

import json
import os
import uuid
import logging
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger("clipai.media")

router = APIRouter()

UPLOAD_DIR = "/data/uploads"
# Special job_id used for the global media library (not tied to any job)
GLOBAL_LIBRARY_ID = "_library"
ALLOWED_EXTENSIONS = {
    "video": {".mp4", ".mov", ".webm", ".mkv"},
    "audio": {".mp3", ".wav", ".aac", ".ogg", ".flac"},
    "image": {".png", ".jpg", ".jpeg", ".gif", ".webp"},
}
MAX_SIZES = {
    "video": 2 * 1024 * 1024 * 1024,  # 2 GB
    "audio": 500 * 1024 * 1024,         # 500 MB
    "image": 50 * 1024 * 1024,          # 50 MB
}

# Stream buffer size for writing to disk
_STREAM_BUF = 1024 * 1024  # 1 MB


def _meta_path(media_dir: str) -> str:
    """Path to the JSON metadata file that stores original filenames."""
    return os.path.join(media_dir, "_meta.json")


def _load_meta(media_dir: str) -> dict:
    path = _meta_path(media_dir)
    if os.path.isfile(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_meta(media_dir: str, meta: dict):
    """Atomically write metadata to prevent corruption on crash.

    Writes to a temporary file first, then renames (atomic on POSIX).
    This ensures _meta.json is never partially written — if the container
    crashes mid-write, the old file remains intact.
    """
    path = _meta_path(media_dir)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(meta, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def detect_media_type(filename: str) -> str | None:
    ext = Path(filename).suffix.lower()
    for media_type, extensions in ALLOWED_EXTENSIONS.items():
        if ext in extensions:
            return media_type
    return None


@router.post("/api/media/upload")
async def upload_media(
    file: UploadFile = File(...),
    job_id: str = Query(default=GLOBAL_LIBRARY_ID),
):
    """Accept media upload for the multi-track editor, store to data/uploads/{job_id}/media/.

    When job_id is omitted it defaults to the global media library (_library).
    Streams file to disk in chunks to avoid loading large files into memory.
    """
    media_type = detect_media_type(file.filename or "")
    if not media_type:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")

    max_size = MAX_SIZES.get(media_type, 50 * 1024 * 1024)

    # Create upload directory
    media_dir = os.path.join(UPLOAD_DIR, job_id, "media")
    os.makedirs(media_dir, exist_ok=True)

    # Generate unique filename
    media_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename or "upload").suffix.lower()
    safe_filename = f"{media_id}{ext}"
    file_path = os.path.join(media_dir, safe_filename)

    # Stream to disk to avoid loading entire file into memory
    total_written = 0
    try:
        async with aiofiles.open(file_path, "wb") as out:
            while True:
                chunk = await file.read(_STREAM_BUF)
                if not chunk:
                    break
                total_written += len(chunk)
                if total_written > max_size:
                    # Clean up oversized file
                    await out.close()
                    os.remove(file_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {max_size // (1024 * 1024)} MB limit for {media_type}",
                    )
                await out.write(chunk)
    except HTTPException:
        raise
    except OSError as exc:
        # Clean up on disk error
        try:
            os.remove(file_path)
        except OSError:
            pass
        logger.error("Disk error writing media %s: %s", safe_filename, exc)
        raise HTTPException(status_code=507, detail="Server storage error")

    # Save original filename in metadata
    original_name = file.filename or safe_filename
    meta = _load_meta(media_dir)
    meta[media_id] = {"original_filename": original_name}
    _save_meta(media_dir, meta)

    logger.info("Uploaded media %s (%s, %d bytes) for job %s", safe_filename, media_type, total_written, job_id)

    # Build URL for frontend
    url = f"/api/files/{job_id}/media/{safe_filename}"

    return JSONResponse({
        "id": media_id,
        "filename": original_name,
        "type": media_type,
        "size": total_written,
        "url": url,
    })


@router.post("/api/media/register")
async def register_media(
    file_path: str = Query(...),
    filename: str = Query(...),
    media_type: str = Query(...),
    job_id: str = Query(default=GLOBAL_LIBRARY_ID),
):
    """Register an already-assembled file (from chunked upload) in the media library.

    Moves/links the file from the upload directory into the media directory
    so it appears in the media library alongside directly uploaded files.
    """
    # Security: ensure file_path is under /data/uploads/
    real_path = os.path.realpath(file_path)
    if not real_path.startswith("/data/uploads/"):
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not os.path.isfile(real_path):
        raise HTTPException(status_code=404, detail="File not found")

    if media_type not in ("video", "audio", "image"):
        raise HTTPException(status_code=400, detail=f"Invalid media type: {media_type}")

    media_dir = os.path.join(UPLOAD_DIR, job_id, "media")
    os.makedirs(media_dir, exist_ok=True)

    media_id = str(uuid.uuid4())[:8]
    ext = Path(filename).suffix.lower() or Path(real_path).suffix.lower()
    safe_filename = f"{media_id}{ext}"
    dest_path = os.path.join(media_dir, safe_filename)

    # Move the assembled file into the media directory
    try:
        os.rename(real_path, dest_path)
    except OSError:
        # Cross-device: fall back to copy
        import shutil
        shutil.move(real_path, dest_path)

    total_size = os.path.getsize(dest_path)

    # Save original filename in metadata
    meta = _load_meta(media_dir)
    meta[media_id] = {"original_filename": filename}
    _save_meta(media_dir, meta)

    url = f"/api/files/{job_id}/media/{safe_filename}"
    logger.info("Registered media %s (%s, %d bytes) for job %s", safe_filename, media_type, total_size, job_id)

    return JSONResponse({
        "id": media_id,
        "filename": filename,
        "type": media_type,
        "size": total_size,
        "url": url,
    })


@router.get("/api/media/list")
async def list_media(job_id: str = Query(default=GLOBAL_LIBRARY_ID)):
    """List all uploaded media files for a job (defaults to global library)."""
    media_dir = os.path.join(UPLOAD_DIR, job_id, "media")
    if not os.path.isdir(media_dir):
        return JSONResponse({"items": []})

    meta = _load_meta(media_dir)

    items = []
    for fname in sorted(os.listdir(media_dir)):
        if fname.startswith("_"):
            continue  # skip metadata files
        fpath = os.path.join(media_dir, fname)
        if not os.path.isfile(fpath):
            continue
        media_type = detect_media_type(fname)
        if not media_type:
            continue
        media_id = Path(fname).stem
        original_name = meta.get(media_id, {}).get("original_filename", fname)
        items.append({
            "id": media_id,
            "filename": original_name,
            "type": media_type,
            "size": os.path.getsize(fpath),
            "url": f"/api/files/{job_id}/media/{fname}",
        })
    return JSONResponse({"items": items})


@router.delete("/api/media/{media_id}")
async def delete_media(media_id: str, job_id: str = Query(default=GLOBAL_LIBRARY_ID)):
    """Delete an uploaded media file."""
    media_dir = os.path.join(UPLOAD_DIR, job_id, "media")
    if not os.path.isdir(media_dir):
        raise HTTPException(status_code=404, detail="Media not found")

    for fname in os.listdir(media_dir):
        if fname.startswith("_"):
            continue
        if Path(fname).stem == media_id:
            fpath = os.path.join(media_dir, fname)
            os.remove(fpath)
            # Clean up metadata
            meta = _load_meta(media_dir)
            meta.pop(media_id, None)
            _save_meta(media_dir, meta)
            logger.info("Deleted media %s for job %s", fname, job_id)
            return JSONResponse({"deleted": True, "id": media_id})

    raise HTTPException(status_code=404, detail="Media not found")
