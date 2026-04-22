"""SAM 3 segmentation + tracking wrapper — Blueprint v2 Phase 4.

One ``segment_and_track`` entrypoint hides the choice of backend
(self-hosted HTTP service, Replicate, Modal, fal.ai, or disabled).
Callers pass a video path and a list of concept prompts; the wrapper
returns per-frame instance detections with stable IDs, which the
ClipAI pipeline converts into ``FrameFaces`` (via
``sam3_to_dense_faces``) and ``RequiredRegion`` entries (via
``sam3_to_required_regions``).

Safety-net contract: ``segment_and_track`` **never raises**. On any
failure — misconfigured backend, network error, timeout, unparseable
response — it returns ``None`` so the caller's legacy detector
fallback path runs. This is the only reason ClipAI can ship SAM 3 at
all on a cluster that can't self-host the model.

Backends selected by ``SAM3_BACKEND``:

    disabled         (default) — always return ``None``
    local_service    HTTP POST to SAM3_SERVICE_URL (recommended)
    api_replicate    Replicate endpoint (REPLICATE_API_TOKEN)
    api_modal        Modal endpoint (SAM3_MODAL_URL)
    api_fal          fal.ai endpoint (FAL_KEY)

The pipeline is gated by ``CLIPAI_SAM3_ENABLED=1`` — even when a
backend is configured, leaving the flag off keeps the whole path
inert.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# Prompt classification tables — callers can override by pointing
# ``SAM3_REQUIRED_PROMPTS`` / ``SAM3_PREFERRED_PROMPTS`` at a JSON
# file, but the defaults cover the v2 blueprint genre list.
_FACE_PROMPTS: frozenset = frozenset({
    "visible face", "anime face", "speaking person", "face",
    "narrator", "performer", "vocalist",
})

_REQUIRED_PROMPTS: frozenset = frozenset({
    "HUD elements", "score bug", "scoreboard", "minimap", "health bar",
    "speech bubble", "subtitle text", "screen capture region",
    "kill feed", "chat overlay", "alert overlay",
    "presentation slide",
})

_PREFERRED_PROMPTS: frozenset = frozenset({
    "the ball", "basketball", "lead car", "cars", "racing flag",
    "player character", "player champion", "player vehicle",
    "player crosshair", "crosshair", "instrument", "main subject",
    "webcam overlay", "cursor", "stage lighting",
})


# Rolling telemetry — last N SAM 3 calls. Exposed via the diagnostics
# endpoint. Not persisted, so it resets on process restart.
_TELEMETRY_RING: list = []
_TELEMETRY_CAP: int = 256


def _record_telemetry(entry: dict) -> None:
    _TELEMETRY_RING.append(entry)
    if len(_TELEMETRY_RING) > _TELEMETRY_CAP:
        del _TELEMETRY_RING[: len(_TELEMETRY_RING) - _TELEMETRY_CAP]


def get_telemetry_snapshot(limit: int = 100) -> list:
    """Return the most recent SAM 3 call telemetry entries."""
    return list(_TELEMETRY_RING[-max(1, int(limit)):])


@dataclass
class SAM3Detection:
    """One SAM 3 detection on one sampled frame."""
    frame_idx: int
    timestamp: float
    instance_id: str      # stable across the whole video
    prompt: str           # concept prompt that matched
    bbox: tuple           # (x1, y1, x2, y2) in pixels
    score: float = 0.0
    mask_rle: Optional[str] = None


@dataclass
class SAM3Result:
    detections: list = field(default_factory=list)
    backend: str = "disabled"
    latency_sec: float = 0.0
    frame_count: int = 0
    instances_by_prompt: dict = field(default_factory=dict)


def _count_instances_by_prompt(detections: list) -> dict:
    """Nested dict mapping ``prompt → {instance_id: count}``."""
    out: dict = {}
    for d in detections:
        by_id = out.setdefault(d.prompt, {})
        by_id[d.instance_id] = by_id.get(d.instance_id, 0) + 1
    return out


def _parse_detections_payload(
    payload, default_backend: str,
) -> Optional[SAM3Result]:
    """Normalize a SAM 3 HTTP response into a ``SAM3Result``.

    Accepts either a dict with a ``detections`` array or a bare list
    (some backends return one, some the other). Returns ``None`` on
    unparseable input so the caller's fallback kicks in.
    """
    if payload is None:
        return None
    if isinstance(payload, dict):
        raw = payload.get("detections")
        backend = str(payload.get("backend", default_backend))
        frame_count = int(payload.get("frame_count", 0) or 0)
        latency = float(payload.get("latency_sec", 0.0) or 0.0)
    elif isinstance(payload, list):
        raw = payload
        backend = default_backend
        frame_count = 0
        latency = 0.0
    else:
        return None
    if not isinstance(raw, list):
        return None

    detections: list[SAM3Detection] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            bbox = entry.get("bbox") or entry.get("box")
            if not bbox or len(bbox) < 4:
                continue
            detections.append(SAM3Detection(
                frame_idx=int(entry.get("frame", entry.get("frame_idx", 0))),
                timestamp=float(entry.get("t", entry.get("timestamp", 0.0))),
                instance_id=str(entry.get("id", entry.get("instance_id", ""))),
                prompt=str(entry.get("prompt", ""))[:120],
                bbox=(
                    float(bbox[0]), float(bbox[1]),
                    float(bbox[2]), float(bbox[3]),
                ),
                score=float(entry.get("score", 0.0) or 0.0),
                mask_rle=entry.get("mask_rle"),
            ))
        except (TypeError, ValueError):
            continue

    if frame_count == 0 and detections:
        frame_count = len({(d.frame_idx, d.timestamp) for d in detections})

    return SAM3Result(
        detections=detections,
        backend=backend,
        latency_sec=latency,
        frame_count=frame_count,
        instances_by_prompt=_count_instances_by_prompt(detections),
    )


async def segment_and_track(
    video_path: str,
    prompts: list,
    *,
    sample_fps: float = 6.0,
    max_frames: int = 600,
    job_id: str = "",
) -> Optional[SAM3Result]:
    """Blueprint v2 Phase 4 — run SAM 3 and return structured detections.

    Returns ``None`` on any failure. Callers should always have a
    legacy fallback path.
    """
    master = os.environ.get("CLIPAI_SAM3_ENABLED", "0").lower()
    if master not in ("1", "true", "yes"):
        return None

    backend = os.environ.get("SAM3_BACKEND", "disabled").lower()
    if backend == "disabled" or not prompts:
        return None

    dispatcher = {
        "local_service": _call_local_service,
        "api_replicate": _call_replicate,
        "api_modal": _call_modal,
        "api_fal": _call_fal,
    }.get(backend)

    if dispatcher is None:
        logger.warning("[%s] unknown SAM3_BACKEND=%r; skipping", job_id, backend)
        return None

    t0 = time.monotonic()
    try:
        result = await dispatcher(
            video_path=video_path,
            prompts=list(prompts),
            sample_fps=float(sample_fps),
            max_frames=int(max_frames),
            job_id=job_id,
        )
    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.warning(
            "[%s] SAM 3 call via %s failed after %.1fs: %s",
            job_id, backend, elapsed, e,
        )
        _record_telemetry({
            "ts": time.time(),
            "job_id": job_id,
            "backend": backend,
            "ok": False,
            "error": str(e)[:200],
            "latency_sec": elapsed,
            "prompts": len(prompts),
        })
        return None

    elapsed = time.monotonic() - t0
    if result is None:
        _record_telemetry({
            "ts": time.time(),
            "job_id": job_id,
            "backend": backend,
            "ok": False,
            "error": "empty-result",
            "latency_sec": elapsed,
            "prompts": len(prompts),
        })
        return None

    if not result.latency_sec:
        result.latency_sec = elapsed

    logger.info(
        "[%s] SAM 3 telemetry: frames=%d detections=%d backend=%s "
        "prompts=%d latency=%.1fs",
        job_id, result.frame_count, len(result.detections),
        result.backend, len(prompts), result.latency_sec,
    )
    _record_telemetry({
        "ts": time.time(),
        "job_id": job_id,
        "backend": result.backend,
        "ok": True,
        "frame_count": result.frame_count,
        "detection_count": len(result.detections),
        "latency_sec": result.latency_sec,
        "prompts": len(prompts),
    })
    return result


# ── Backend implementations ──────────────────────────────────────
#
# Each returns a ``SAM3Result`` or raises (the dispatcher converts to
# ``None``). Implementations intentionally stay close to the wire
# format so swapping backends is a config change, not a code change.


async def _call_local_service(
    *, video_path: str, prompts: list, sample_fps: float,
    max_frames: int, job_id: str,
) -> Optional[SAM3Result]:
    """HTTP POST to a self-hosted SAM 3 service.

    Expected endpoint: ``POST {SAM3_SERVICE_URL}/segment``
    Body: ``{"video_path": "...", "prompts": [...], "sample_fps": 6.0,
             "max_frames": 600}``
    Response: ``{"detections": [{"frame": N, "t": 1.23, "id": "inst_001",
                 "prompt": "the ball", "bbox": [x1,y1,x2,y2],
                 "score": 0.87}], "frame_count": 1440, "latency_sec": 8.1}``
    """
    import httpx

    base = os.environ.get("SAM3_SERVICE_URL")
    if not base:
        logger.warning(
            "[%s] SAM3_BACKEND=local_service but SAM3_SERVICE_URL is unset",
            job_id,
        )
        return None
    url = base.rstrip("/") + "/segment"
    timeout = float(os.environ.get("SAM3_HTTP_TIMEOUT", "300"))

    payload = {
        "video_path": video_path,
        "prompts": prompts,
        "sample_fps": sample_fps,
        "max_frames": max_frames,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        body = resp.json()
    return _parse_detections_payload(body, "local_service")


async def _call_replicate(
    *, video_path: str, prompts: list, sample_fps: float,
    max_frames: int, job_id: str,
) -> Optional[SAM3Result]:
    """Replicate SAM 3 endpoint. Token from REPLICATE_API_TOKEN.

    Implementation note: Replicate returns a webhook-polled job ID
    for video workloads. This wrapper POSTs to create, polls up to
    ``SAM3_REPLICATE_POLL_SEC`` (default 300s), then parses.
    """
    import httpx

    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        logger.warning("[%s] SAM3_BACKEND=api_replicate but REPLICATE_API_TOKEN is unset", job_id)
        return None
    model = os.environ.get("SAM3_REPLICATE_MODEL", "meta/sam-3")
    poll_budget = float(os.environ.get("SAM3_REPLICATE_POLL_SEC", "300"))

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        create = await client.post(
            "https://api.replicate.com/v1/predictions",
            headers=headers,
            json={
                "version": model,
                "input": {
                    "video": video_path,
                    "prompts": prompts,
                    "sample_fps": sample_fps,
                    "max_frames": max_frames,
                },
            },
        )
        create.raise_for_status()
        pred = create.json()
        poll_url = pred.get("urls", {}).get("get")
        if not poll_url:
            return None

        deadline = time.monotonic() + poll_budget
        while time.monotonic() < deadline:
            poll = await client.get(poll_url, headers=headers)
            poll.raise_for_status()
            body = poll.json()
            status = body.get("status")
            if status in ("succeeded",):
                return _parse_detections_payload(body.get("output"), "api_replicate")
            if status in ("failed", "canceled"):
                logger.warning(
                    "[%s] Replicate SAM 3 returned status=%s: %s",
                    job_id, status, body.get("error"),
                )
                return None
            await asyncio.sleep(2.0)
    return None


async def _call_modal(
    *, video_path: str, prompts: list, sample_fps: float,
    max_frames: int, job_id: str,
) -> Optional[SAM3Result]:
    """Modal endpoint — same wire format as local_service (POST /segment)."""
    import httpx
    url = os.environ.get("SAM3_MODAL_URL")
    token = os.environ.get("SAM3_MODAL_TOKEN")
    if not url:
        return None
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    timeout = float(os.environ.get("SAM3_HTTP_TIMEOUT", "300"))
    payload = {
        "video_path": video_path,
        "prompts": prompts,
        "sample_fps": sample_fps,
        "max_frames": max_frames,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return _parse_detections_payload(resp.json(), "api_modal")


async def _call_fal(
    *, video_path: str, prompts: list, sample_fps: float,
    max_frames: int, job_id: str,
) -> Optional[SAM3Result]:
    """fal.ai endpoint — same wire format, different auth header."""
    import httpx
    url = os.environ.get("SAM3_FAL_URL")
    key = os.environ.get("FAL_KEY")
    if not url or not key:
        return None
    headers = {"Authorization": f"Key {key}"}
    timeout = float(os.environ.get("SAM3_HTTP_TIMEOUT", "300"))
    payload = {
        "video_path": video_path,
        "prompts": prompts,
        "sample_fps": sample_fps,
        "max_frames": max_frames,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return _parse_detections_payload(resp.json(), "api_fal")


# ── Adapters: SAM 3 → ClipAI internal format ──────────────────────


def _instance_to_slot(instance_id: str) -> int:
    """Map a SAM 3 instance id to a stable ClipAI face slot.

    SAM 3 ids look like ``face:0``, ``face:1``, ``ball:0``. We pull the
    trailing integer where possible so slot assignments stay short +
    deterministic. When the id doesn't parse, fall back to a hash so
    different instances still land in different slots.
    """
    if not instance_id:
        return -1
    # Trailing digits → use them directly.
    tail = ""
    for ch in reversed(str(instance_id)):
        if ch.isdigit():
            tail = ch + tail
        else:
            break
    if tail:
        try:
            return int(tail)
        except ValueError:
            pass
    # Fall back to a stable hash in a small range.
    return abs(hash(instance_id)) % 1024


def sam3_to_dense_faces(
    result: SAM3Result,
    *,
    source_width: int,
    source_height: int,
) -> list:
    """Blueprint v2 Phase 4 — convert SAM 3 face detections into the
    ``FrameFaces`` list that the rest of the pipeline consumes.

    Coordinates are converted from SAM 3's pixel-space bbox into
    ClipAI's 0-100 percent convention. ``source`` is tagged as
    ``sam3`` so downstream code can distinguish SAM 3 vs legacy.
    """
    if result is None:
        return []
    from backend.services.face_detector import FaceInfo, FrameFaces

    sw = max(int(source_width), 1)
    sh = max(int(source_height), 1)

    by_key: dict = {}
    for d in result.detections:
        if d.prompt not in _FACE_PROMPTS:
            continue
        by_key.setdefault((d.frame_idx, d.timestamp), []).append(d)

    out: list = []
    for (idx, t), dets in sorted(by_key.items(), key=lambda kv: kv[0][1]):
        faces = []
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            w_pct = max(0.0, (x2 - x1) / sw * 100.0)
            h_pct = max(0.0, (y2 - y1) / sh * 100.0)
            cx_pct = max(0.0, min(100.0, ((x1 + x2) / 2) / sw * 100.0))
            cy_pct = max(0.0, min(100.0, ((y1 + y2) / 2) / sh * 100.0))
            faces.append(FaceInfo(
                x_center=cx_pct,
                y_center=cy_pct,
                width=w_pct,
                height=h_pct,
                nose_x=cx_pct,
                nose_y=cy_pct,
                confidence=float(d.score),
                identity_id=_instance_to_slot(d.instance_id),
                y_bottom=cy_pct + h_pct / 2,
                is_human=(d.prompt != "anime face"),
            ))
        ff = FrameFaces(timestamp=float(t), frame_path="", faces=faces)
        # Stamp provenance so downstream telemetry + tests can tell.
        ff.source = "sam3"  # type: ignore[attr-defined]
        out.append(ff)
    return out


def sam3_to_required_regions(
    result: SAM3Result,
    *,
    source_width: int,
    source_height: int,
    config=None,
) -> list:
    """Convert SAM 3 HUD / text / object detections into
    :class:`RequiredRegion` entries. Hard-constraint prompts ("HUD
    elements", "subtitle text", "score bug") land on the ``required``
    tier with ``importance.required_region_gain`` weight; genre
    object prompts ("the ball", "lead car") land on the ``preferred``
    tier with ``importance.object`` weight.
    """
    if result is None:
        return []
    from backend.services.reframe_config import get_default_config
    from backend.services.required_regions import RequiredRegion

    cfg = (config or get_default_config())
    gain = float(cfg.importance.required_region_gain)
    obj_w = float(cfg.importance.object)

    sw = max(int(source_width), 1)
    sh = max(int(source_height), 1)

    out: list = []
    for d in result.detections:
        if d.prompt in _REQUIRED_PROMPTS:
            tier = "required"
            weight = gain
        elif d.prompt in _PREFERRED_PROMPTS:
            tier = "preferred"
            # Most callers expect weights in the O(1) range; the
            # ``object`` multiplier already encodes the per-genre
            # importance so we don't scale by ``gain`` here.
            weight = max(0.1, obj_w)
        else:
            continue
        x1, y1, x2, y2 = d.bbox
        cx = ((x1 + x2) / 2) / sw
        cy = ((y1 + y2) / 2) / sh
        hw = max(0.0, (x2 - x1) / (2 * sw))
        hh = max(0.0, (y2 - y1) / (2 * sh))
        out.append(RequiredRegion(
            timestamp=float(d.timestamp),
            cx=max(0.0, min(1.0, cx)),
            cy=max(0.0, min(1.0, cy)),
            half_width=max(0.0, min(0.5, hw)),
            half_height=max(0.0, min(0.5, hh)),
            score=float(d.score),
            tier=tier,
            source="sam3",
            weight=float(weight),
        ))
    return out
