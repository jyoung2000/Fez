"""Persistent upload session state manager.

Persists chunked upload session metadata to disk as JSON so that
in-progress uploads survive server restarts. Each session is stored
as ``/data/uploads/.chunked_{upload_id}/session.json``.
"""

import asyncio
import json
import logging
import os

logger = logging.getLogger(__name__)

UPLOAD_DIR = "/data/uploads"


class UploadStateManager:
    """Thread-safe, disk-backed upload session store."""

    def __init__(self):
        # Per-upload_id locks to serialize read-modify-write
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, upload_id: str) -> asyncio.Lock:
        if upload_id not in self._locks:
            self._locks[upload_id] = asyncio.Lock()
        return self._locks[upload_id]

    @staticmethod
    def _session_path(upload_id: str) -> str:
        return os.path.join(UPLOAD_DIR, f".chunked_{upload_id}", "session.json")

    async def create_session(self, upload_id: str, info: dict):
        """Write a new session to disk."""
        path = self._session_path(upload_id)
        async with self._lock_for(upload_id):
            self._write_json(path, info)

    async def update_session(self, upload_id: str, updates: dict):
        """Read-modify-write the session JSON atomically."""
        path = self._session_path(upload_id)
        async with self._lock_for(upload_id):
            info = self._read_json(path)
            if info is None:
                return
            info.update(updates)
            self._write_json(path, info)

    async def save_session(self, upload_id: str, info: dict):
        """Overwrite the full session data on disk."""
        path = self._session_path(upload_id)
        async with self._lock_for(upload_id):
            self._write_json(path, info)

    async def get_session(self, upload_id: str) -> dict | None:
        """Read session from disk. Returns None if missing."""
        path = self._session_path(upload_id)
        return self._read_json(path)

    async def delete_session(self, upload_id: str):
        """Remove the session JSON file."""
        path = self._session_path(upload_id)
        async with self._lock_for(upload_id):
            try:
                os.remove(path)
            except OSError:
                pass
        self._locks.pop(upload_id, None)

    def recover_sessions(self) -> dict[str, dict]:
        """Scan for .chunked_* dirs with session.json — called at startup."""
        sessions = {}
        if not os.path.isdir(UPLOAD_DIR):
            return sessions
        for entry in os.listdir(UPLOAD_DIR):
            if not entry.startswith(".chunked_"):
                continue
            upload_id = entry[len(".chunked_"):]
            session_path = os.path.join(UPLOAD_DIR, entry, "session.json")
            info = self._read_json(session_path)
            if info is not None:
                # Convert chunks_received keys back to int (JSON serialises as strings)
                if "chunks_received" in info and isinstance(info["chunks_received"], dict):
                    info["chunks_received"] = {
                        int(k): v for k, v in info["chunks_received"].items()
                    }
                sessions[upload_id] = info
                logger.info("Recovered upload session %s (state=%s)", upload_id, info.get("state"))
        return sessions

    @staticmethod
    def _read_json(path: str) -> dict | None:
        if not os.path.isfile(path):
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def _write_json(path: str, data: dict):
        tmp = path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except OSError as e:
            logger.warning("Failed to persist session to %s: %s", path, e)


# Module-level singleton
upload_state = UploadStateManager()
