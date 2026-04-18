import asyncio
import json
import os
import uuid
from datetime import datetime, timezone

import aiofiles
from fastapi import APIRouter, HTTPException

from backend.models import ClipPreset, SavePresetRequest, RenamePresetRequest

router = APIRouter(prefix="/api", tags=["presets"])

# Persistent storage path — lives on the Docker-mounted /data/logs volume
_PRESETS_DIR = "/data/logs"
_PRESETS_PATH = os.path.join(_PRESETS_DIR, "clip_presets.json")
_lock = asyncio.Lock()


async def _load_presets() -> list[dict]:
    """Load presets from disk. Returns empty list if file doesn't exist."""
    if not os.path.exists(_PRESETS_PATH):
        return []
    try:
        async with aiofiles.open(_PRESETS_PATH, "r") as f:
            content = await f.read()
        return json.loads(content) if content.strip() else []
    except (json.JSONDecodeError, OSError):
        return []


async def _save_presets(presets: list[dict]):
    """Write presets to disk atomically."""
    os.makedirs(_PRESETS_DIR, exist_ok=True)
    tmp_path = _PRESETS_PATH + ".tmp"
    async with aiofiles.open(tmp_path, "w") as f:
        await f.write(json.dumps(presets, indent=2))
    os.replace(tmp_path, _PRESETS_PATH)


@router.get("/clip-presets")
async def list_presets():
    """Return all saved clip setting presets."""
    async with _lock:
        return await _load_presets()


@router.post("/clip-presets")
async def create_preset(req: SavePresetRequest):
    """Save current clip settings as a new named preset."""
    if not req.name or not req.name.strip():
        raise HTTPException(status_code=400, detail="Preset name is required")

    preset = ClipPreset(
        id=str(uuid.uuid4()),
        name=req.name.strip(),
        settings=req.settings,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    async with _lock:
        presets = await _load_presets()
        presets.append(preset.model_dump())
        await _save_presets(presets)

    return preset.model_dump()


@router.put("/clip-presets/{preset_id}")
async def rename_preset(preset_id: str, req: RenamePresetRequest):
    """Rename an existing preset."""
    if not req.name or not req.name.strip():
        raise HTTPException(status_code=400, detail="Preset name is required")

    async with _lock:
        presets = await _load_presets()
        for p in presets:
            if p["id"] == preset_id:
                p["name"] = req.name.strip()
                await _save_presets(presets)
                return p
        raise HTTPException(status_code=404, detail="Preset not found")


@router.delete("/clip-presets/{preset_id}")
async def delete_preset(preset_id: str):
    """Delete a saved preset."""
    async with _lock:
        presets = await _load_presets()
        filtered = [p for p in presets if p["id"] != preset_id]
        if len(filtered) == len(presets):
            raise HTTPException(status_code=404, detail="Preset not found")
        await _save_presets(filtered)

    return {"status": "deleted", "id": preset_id}
