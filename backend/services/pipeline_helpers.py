"""Standalone helpers used by the analysis pipeline.

Extracted from ``pipeline.py`` so unit tests can import them without
pulling in the LLM provider stack (``openai``, ``anthropic``, ``groq``,
``ollama`` ...). The pipeline re-exports these from its module
namespace so existing call sites keep working.

Contents:
  * :func:`_hash_file_sha256` — SHA-256 a file in chunks.
  * :func:`_maybe_use_cached_extraction` — re-analyze cache probe.
  * :func:`_write_extraction_manifest` — sidecar so the cache hit can
    rebuild ``FrameData`` without ffprobe.
  * :func:`_stage_timer` — async ctx manager that logs + records
    per-stage wall-clock time.
  * :func:`_record_pipeline_warning` / :func:`_drain_pipeline_telemetry`
    — accumulators that surface on ``JobResult.timings`` and
    ``JobResult.pipeline_warnings``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time as _time
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


# ── Per-job stage-timing accumulator ────────────────────────────
#
# Keyed by job_id so concurrent analyses don't clobber each other;
# flushed onto JobResult.timings / pipeline_warnings at the end of the
# run by ``_drain_pipeline_telemetry``.

_pipeline_timings: dict[str, dict[str, float]] = {}
_pipeline_warnings: dict[str, list[str]] = {}


# ── Synthetic / fallback scene-description detection ────────────
#
# Every vision provider has its own way of saying "I couldn't analyze
# this frame": Ollama emits "Frame at <ts> (vision unavailable …)",
# OpenRouter emits "Frame analysis unavailable" or "Frame analysis
# unavailable — API key limit exceeded", the orchestrator-level fallback
# emits "Analysis failed", etc. Without a single source of truth the
# pipeline could either
#   (a) count error placeholders as real scenes (Key Scenes shows
#       junk), or
#   (b) miss legitimate descriptions whose wording happens to overlap
#       with a marker substring.
# This list is the canonical "this is NOT a real description" check.
# Keep all matches lowercase substrings; ``is_synthetic_scene`` does
# the case-insensitive comparison.
_SYNTHETIC_SCENE_MARKERS: tuple[str, ...] = (
    "vision skipped",
    "vision unavailable",
    "vision model crashed",
    "analysis skipped",
    "analysis failed",
    "frame analysis unavailable",
    "frame analysis failed",
    "frame not analyzed",
    "api key limit exceeded",
    "no description",
    "no analysis",
    "model returned no response",
    "model returned empty",
)


def is_synthetic_scene(scene) -> bool:
    """Return True iff ``scene.description`` is an error placeholder OR
    structurally invalid (dict repr, JSON leak, refusal, too short).

    Used by the pipeline's ``real_scenes`` filter, the warning logic
    that recommends "switch vision providers" when 0 real descriptions
    came back, and any caller that needs to know whether a scene is
    user-presentable.
    """
    desc = getattr(scene, "description", None)
    if not desc:
        return True
    low = str(desc).strip().lower()
    if not low:
        return True
    if low.startswith("frame at "):
        # Ollama's "Frame at 12.5s (vision unavailable — CLIP on CPU)"
        return True
    if any(marker in low for marker in _SYNTHETIC_SCENE_MARKERS):
        return True
    # Structural check — catches JSON dict reprs, refusals, unparsed
    # model output that doesn't match any known marker phrase.
    # Import inside the function to avoid circular imports at load.
    from backend.services.scene_description_validator import is_valid_description
    ok, _ = is_valid_description(desc)
    return not ok


def _record_pipeline_warning(job_id: str, message: str) -> None:
    """Append a soft-warning string to the per-job warnings list.

    Surfaced on ``JobResult.pipeline_warnings`` so the UI can show
    non-fatal issues alongside the result. Idempotent: duplicate
    messages are coalesced so the list stays short.
    """
    bag = _pipeline_warnings.setdefault(job_id, [])
    if message and message not in bag:
        bag.append(message)


def _drain_pipeline_telemetry(job_id: str) -> tuple[dict[str, float], list[str]]:
    """Pop the accumulated timings + warnings for ``job_id``.

    Returns ``({stage: seconds}, [warning, ...])``. Safe to call
    multiple times — the second call returns empty.
    """
    timings = _pipeline_timings.pop(job_id, {}) or {}
    warnings = _pipeline_warnings.pop(job_id, []) or []
    return timings, warnings


@asynccontextmanager
async def _stage_timer(job_id: str, stage: str):
    """Log + record wall-clock time for a pipeline stage.

    If a stage runs twice (rare — recovery retries) the cumulative
    time is kept. The UI cares about "where time went", not which
    retry attempt it was.
    """
    t0 = _time.monotonic()
    logger.info("[%s] Stage '%s' started", job_id, stage)
    try:
        yield
    finally:
        elapsed = _time.monotonic() - t0
        logger.info("[%s] Stage '%s' finished in %.1fs", job_id, stage, elapsed)
        bag = _pipeline_timings.setdefault(job_id, {})
        bag[stage] = round(bag.get(stage, 0.0) + elapsed, 2)


# ── Extraction cache (frame + audio re-use across re-analyze) ──


def _hash_file_sha256(path: str, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 of ``path`` in 1 MiB chunks. Returns hex digest.

    Returns the empty string when the file can't be read so callers
    don't have to special-case missing files. Hashing a 4 GB video
    takes ~4 s on an SSD; we run it in a thread to keep the event
    loop responsive.
    """
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


async def _maybe_use_cached_extraction(
    *,
    job_id: str,
    video_path: str,
    frames_dir: str,
    audio_path: str,
    expected_sha: str,
):
    """Return cached ``(frames, scene_cut_timestamps)`` if the source
    hasn't changed, else ``None``.

    Conservative: requires both the audio file and at least 5 frame
    images to be present, and the source SHA-256 to match the value
    recorded on the job. Falls back to a fresh extract on any
    mismatch — *never* silently uses stale data.
    """
    if not expected_sha:
        return None
    if not (os.path.isdir(frames_dir) and os.path.isfile(audio_path)):
        return None

    try:
        frame_files = sorted(
            f for f in os.listdir(frames_dir)
            if f.endswith((".jpg", ".jpeg", ".png"))
        )
    except OSError:
        return None
    if len(frame_files) < 5:
        return None

    # Compare hashes off the event loop.
    actual_sha = await asyncio.to_thread(_hash_file_sha256, video_path)
    if not actual_sha or actual_sha != expected_sha:
        return None

    # Reconstruct the FrameData list. Lazy import to avoid pulling
    # the whole ``backend.models`` chain into pure-Python sandboxes.
    from backend.models import FrameData

    sidecar = os.path.join(frames_dir, "manifest.json")
    timestamps: list[float] = []
    scene_cut_timestamps: list[float] = []
    if os.path.isfile(sidecar):
        try:
            with open(sidecar, "r") as f:
                data = json.load(f)
            timestamps = [float(t) for t in (data.get("timestamps") or [])]
            scene_cut_timestamps = [float(t) for t in (data.get("scene_cuts") or [])]
        except Exception:
            timestamps = []
            scene_cut_timestamps = []

    if not timestamps or len(timestamps) != len(frame_files):
        # Fall back to filename-index ordering with a 1 s grid placeholder.
        # We cannot recover scene cuts without a sidecar, so they degrade
        # to empty.
        timestamps = []
        for i, name in enumerate(frame_files):
            base = os.path.splitext(name)[0]
            tail = base.rsplit("_", 1)[-1] if "_" in base else base
            try:
                idx = int(tail)
                timestamps.append(idx * 1.0)
            except ValueError:
                timestamps.append(float(i))

    frames = [
        FrameData(timestamp=ts, path=os.path.join(frames_dir, name))
        for ts, name in zip(timestamps, frame_files)
    ]
    logger.info(
        "[%s] Cache hit: %d frames, audio=%.1f KB, %d scene cuts",
        job_id, len(frames),
        os.path.getsize(audio_path) / 1024.0,
        len(scene_cut_timestamps),
    )
    return frames, scene_cut_timestamps


def _write_extraction_manifest(
    frames_dir: str,
    frames,
    scene_cut_timestamps,
) -> None:
    """Persist a tiny sidecar so future cache hits can rebuild the
    ``FrameData`` list without re-decoding the video.

    Best-effort: any IO error is swallowed since we'll just fall back
    to filename parsing on the next run.
    """
    try:
        manifest = {
            "timestamps": [float(f.timestamp) for f in (frames or [])],
            "scene_cuts": [float(t) for t in (scene_cut_timestamps or [])],
        }
        os.makedirs(frames_dir, exist_ok=True)
        with open(os.path.join(frames_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f)
    except Exception:
        pass
