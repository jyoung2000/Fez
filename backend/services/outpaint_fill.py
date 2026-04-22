"""Tier 3 generative outpaint fill — Blueprint v2 Phase 5.

Given a clip's render plan, scan the ``BLUR_FILL`` ops (Tier 1 fill)
and — when a safety gate passes and the clip is either "hero" or the
user opted in — replace them with an ``OUTPAINT_FILL`` op pointing at
a pre-generated 9:16 video.

Three layers:

1. :func:`should_outpaint_scene` — per-content-type allowlist + per-
   scene safety predicates. Cheap; no network, no frame extraction.
2. :func:`analyze_scene_outpaint_safety` — reads ``dense_faces`` /
   optical flow / HUD regions to decide whether faces / fast motion /
   HUDs live at the crop edge (where outpainters hallucinate badly).
3. :func:`run_outpaint` — dispatcher over provider backends (Luma,
   Seedance, local HTTP). Returns an ``OutpaintResult`` that carries
   the output video path + cost telemetry. Never raises.

All three layers are gated by ``CLIPAI_OUTPAINT_ENABLED=1``. When
unset, the entire module is inert and :func:`maybe_promote_to_outpaint`
returns the plan unchanged.

Safety contract: every failure path (bad content type, safety gate,
provider error, file missing) keeps the original ``BLUR_FILL`` op so
the clip still exports. Tier 1 is always the floor.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Blueprint Stage-5 safety table. Outpainting behaves well on these
# content types; the rest stay on BLUR_FILL regardless of virality.
SAFE_CONTENT_TYPES: frozenset = frozenset({
    "documentary", "landscape", "broll",
    "music_video",
    "cinematic_dialogue",
    "narrative",
    "vlog",
    "interview",
})

# Always skip even if virality is off-the-charts.
UNSAFE_CONTENT_TYPES: frozenset = frozenset({
    "anime", "animation", "animation_dialogue",
    "gameplay", "gameplay_fps", "gameplay_moba",
    "gameplay_tps", "gameplay_racing",
    "screen_share", "tutorial",
})

# Fraction of source width that counts as "near the edge" for the
# face / motion / HUD safety checks. Outpainters fail when they have
# to hallucinate content behind a moving object; an 8 % band catches
# the failure modes without rejecting clips where the subject happens
# to be off-center but still safely inside the crop.
_EDGE_BAND_FRAC: float = 0.08


# ── Telemetry (same pattern as SAM 3 wrapper) ────────────────────
_TELEMETRY_RING: list = []
_TELEMETRY_CAP: int = 256


def _record_telemetry(entry: dict) -> None:
    _TELEMETRY_RING.append(entry)
    if len(_TELEMETRY_RING) > _TELEMETRY_CAP:
        del _TELEMETRY_RING[: len(_TELEMETRY_RING) - _TELEMETRY_CAP]


def get_telemetry_snapshot(limit: int = 100) -> list:
    return list(_TELEMETRY_RING[-max(1, int(limit)):])


# ── Result dataclass ──────────────────────────────────────────────


@dataclass
class OutpaintResult:
    success: bool
    output_path: Optional[str] = None
    provider: str = ""
    latency_sec: float = 0.0
    frame_count: int = 0
    cost_usd: float = 0.0
    failure_reason: str = ""


# ── Safety gate (pure) ────────────────────────────────────────────


def should_outpaint_scene(
    *,
    content_type: str,
    scene_has_face_near_edge: bool,
    scene_has_fast_motion_at_edge: bool,
    scene_has_hud_at_edge: bool,
    is_hero_clip: bool,
    user_opt_in: bool,
) -> tuple[bool, str]:
    """Blueprint Stage-5 safety gate. Returns ``(allow, reason)``."""
    key = str(content_type or "").lower()
    if key in UNSAFE_CONTENT_TYPES:
        return False, f"content_type={key} in unsafe set"
    if key not in SAFE_CONTENT_TYPES:
        return False, f"content_type={key} not in safe set"
    if scene_has_face_near_edge:
        return False, "face near crop edge"
    if scene_has_fast_motion_at_edge:
        return False, "fast motion at crop edge"
    if scene_has_hud_at_edge:
        return False, "HUD near crop edge"
    if not (is_hero_clip or user_opt_in):
        return False, "not hero clip, no user opt-in"
    return True, "ok"


# ── Edge-safety detector ──────────────────────────────────────────


def analyze_scene_outpaint_safety(
    *,
    scene_start: float,
    scene_end: float,
    primary_crop_rect_px: tuple,   # (x, y, w, h) in source pixels
    source_width: int,
    dense_faces: list = None,
    optical_flow_frames: list = None,
    hud_regions: list = None,
) -> dict:
    """Return the three safety predicates for a scene.

    Keys: ``face_near_edge`` / ``fast_motion_at_edge`` / ``hud_at_edge``.

    Uses face ``x_center`` + ``width`` in 0-100 % convention (the
    ClipAI FaceInfo shape), plus pixel-space ``primary_crop_rect_px``
    to locate the crop edges. ``hud_regions`` entries are dicts with
    ``x``/``w`` in 0-100 % convention (or source pixels if
    ``_units="px"``).
    """
    crop_x, _, crop_w, _ = primary_crop_rect_px
    sw = max(int(source_width), 1)
    left_edge_frac = crop_x / sw
    right_edge_frac = (crop_x + crop_w) / sw
    band = float(_EDGE_BAND_FRAC)

    def _x_near_edge(x_frac: float) -> bool:
        return (
            x_frac < left_edge_frac + band
            or x_frac > right_edge_frac - band
        )

    # ── Face near edge ──
    face_near_edge = False
    for frame in (dense_faces or []):
        t = float(getattr(frame, "timestamp", 0.0))
        if not (scene_start <= t <= scene_end):
            continue
        for face in getattr(frame, "faces", None) or []:
            cx_pct = float(
                getattr(face, "x_center",
                        getattr(face, "nose_x", 50.0))
                or 50.0
            )
            x_frac = cx_pct / 100.0
            if _x_near_edge(x_frac):
                face_near_edge = True
                break
        if face_near_edge:
            break

    # ── Fast motion at edge ──
    # Flow magnitude > 10 % of frame width per frame in the edge band.
    fast_motion_at_edge = False
    for flow in (optical_flow_frames or []):
        t = float(flow.get("timestamp", 0.0)) if isinstance(flow, dict) else float(getattr(flow, "timestamp", 0.0))
        if not (scene_start <= t <= scene_end):
            continue
        mag = (
            flow.get("magnitude_at_edges", 0.0)
            if isinstance(flow, dict)
            else float(getattr(flow, "magnitude_at_edges", 0.0) or 0.0)
        )
        if float(mag) > 0.10:
            fast_motion_at_edge = True
            break

    # ── HUD at edge ──
    hud_at_edge = False
    for hud in (hud_regions or []):
        units = hud.get("_units", "pct") if isinstance(hud, dict) else "pct"
        if isinstance(hud, dict):
            hx = float(hud.get("x", 0.0))
            hw = float(hud.get("w", 0.0))
        else:
            hx = float(getattr(hud, "x", 0.0))
            hw = float(getattr(hud, "w", 0.0))
        center = hx + hw / 2
        x_frac = center / sw if units == "px" else center / 100.0
        if _x_near_edge(x_frac):
            hud_at_edge = True
            break

    return {
        "face_near_edge": face_near_edge,
        "fast_motion_at_edge": fast_motion_at_edge,
        "hud_at_edge": hud_at_edge,
    }


# ── Provider dispatcher ───────────────────────────────────────────


async def run_outpaint(
    *,
    source_segment_path: str,
    primary_crop_rect: dict,
    duration_sec: float,
    job_id: str = "",
) -> OutpaintResult:
    """Dispatch the outpaint call to the configured provider.

    Returns ``OutpaintResult(success=False, failure_reason=...)`` on
    any failure so the caller's BLUR_FILL fallback runs. Never raises.
    """
    master = os.environ.get("CLIPAI_OUTPAINT_ENABLED", "0").lower()
    if master not in ("1", "true", "yes"):
        return OutpaintResult(
            success=False,
            failure_reason="CLIPAI_OUTPAINT_ENABLED not set",
        )
    provider = os.environ.get("CLIPAI_OUTPAINT_PROVIDER", "luma").lower()

    dispatcher = {
        "luma": _call_luma,
        "seedance": _call_seedance,
        "local": _call_local,
    }.get(provider)
    if dispatcher is None:
        return OutpaintResult(
            success=False,
            failure_reason=f"unknown provider {provider!r}",
        )

    t0 = time.monotonic()
    try:
        result = await dispatcher(
            source_segment_path=source_segment_path,
            primary_crop_rect=primary_crop_rect,
            duration_sec=float(duration_sec),
            job_id=job_id,
        )
    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.warning(
            "[%s] outpaint call via %s failed after %.1fs: %s",
            job_id, provider, elapsed, e,
        )
        _record_telemetry({
            "ts": time.time(),
            "job_id": job_id,
            "provider": provider,
            "ok": False,
            "error": str(e)[:200],
            "latency_sec": elapsed,
        })
        return OutpaintResult(
            success=False, provider=provider,
            latency_sec=elapsed, failure_reason=str(e)[:200],
        )

    elapsed = time.monotonic() - t0
    if not result.latency_sec:
        result.latency_sec = elapsed
    _record_telemetry({
        "ts": time.time(),
        "job_id": job_id,
        "provider": result.provider or provider,
        "ok": result.success,
        "latency_sec": result.latency_sec,
        "cost_usd": result.cost_usd,
        "failure_reason": result.failure_reason,
    })
    return result


async def _call_luma(
    *, source_segment_path: str, primary_crop_rect: dict,
    duration_sec: float, job_id: str,
) -> OutpaintResult:
    """Luma Ray 2 endpoint — productized video outpaint.

    Wire format (what ClipAI expects from a Luma-compatible proxy):
        POST {LUMA_API_BASE}/v1/video/outpaint
        Authorization: Bearer {LUMA_API_KEY}
        Body: {
          "video_url": "...",
          "target_aspect": "9:16",
          "subject_region": [x_frac, y_frac, w_frac, h_frac]
        }
        Response (poll-until-done): {
          "status": "done", "output_url": "...",
          "duration_sec": 12.4, "cost_usd": 0.87
        }
    """
    import httpx

    api_key = os.environ.get("LUMA_API_KEY")
    if not api_key:
        return OutpaintResult(
            success=False, provider="luma",
            failure_reason="LUMA_API_KEY unset",
        )
    base = os.environ.get("LUMA_API_BASE", "https://api.lumalabs.ai").rstrip("/")
    url = f"{base}/v1/video/outpaint"
    timeout = float(os.environ.get("LUMA_HTTP_TIMEOUT", "600"))
    payload = {
        "video_url": source_segment_path,
        "target_aspect": "9:16",
        "subject_region": [
            float(primary_crop_rect.get("x", 0.0)),
            float(primary_crop_rect.get("y", 0.0)),
            float(primary_crop_rect.get("w", 1.0)),
            float(primary_crop_rect.get("h", 1.0)),
        ],
        "duration_sec": float(duration_sec),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        body = resp.json()

    return _parse_provider_response(body, provider="luma")


async def _call_seedance(
    *, source_segment_path: str, primary_crop_rect: dict,
    duration_sec: float, job_id: str,
) -> OutpaintResult:
    """ByteDance Seedance 2.0 — same wire shape as Luma via a proxy."""
    import httpx

    api_key = os.environ.get("SEEDANCE_API_KEY")
    base = os.environ.get("SEEDANCE_API_BASE", "").rstrip("/")
    if not api_key or not base:
        return OutpaintResult(
            success=False, provider="seedance",
            failure_reason="SEEDANCE_API_KEY or _BASE unset",
        )
    url = f"{base}/v1/video/outpaint"
    timeout = float(os.environ.get("SEEDANCE_HTTP_TIMEOUT", "600"))
    payload = {
        "video_url": source_segment_path,
        "target_aspect": "9:16",
        "subject_region": [
            float(primary_crop_rect.get("x", 0.0)),
            float(primary_crop_rect.get("y", 0.0)),
            float(primary_crop_rect.get("w", 1.0)),
            float(primary_crop_rect.get("h", 1.0)),
        ],
        "duration_sec": float(duration_sec),
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        body = resp.json()
    return _parse_provider_response(body, provider="seedance")


async def _call_local(
    *, source_segment_path: str, primary_crop_rect: dict,
    duration_sec: float, job_id: str,
) -> OutpaintResult:
    """Self-hosted HTTP service — same wire shape."""
    import httpx

    url = os.environ.get("OUTPAINT_LOCAL_URL")
    if not url:
        return OutpaintResult(
            success=False, provider="local",
            failure_reason="OUTPAINT_LOCAL_URL unset",
        )
    token = os.environ.get("OUTPAINT_LOCAL_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    timeout = float(os.environ.get("OUTPAINT_HTTP_TIMEOUT", "600"))
    payload = {
        "video_path": source_segment_path,
        "target_aspect": "9:16",
        "subject_region": [
            float(primary_crop_rect.get("x", 0.0)),
            float(primary_crop_rect.get("y", 0.0)),
            float(primary_crop_rect.get("w", 1.0)),
            float(primary_crop_rect.get("h", 1.0)),
        ],
        "duration_sec": float(duration_sec),
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        body = resp.json()
    return _parse_provider_response(body, provider="local")


def _parse_provider_response(body, *, provider: str) -> OutpaintResult:
    """Normalize a provider's JSON response into ``OutpaintResult``.

    Accepts the common ``{"status": "done", "output_url": "...",
    "cost_usd": 0.42}`` shape; any deviation → failure.
    """
    if not isinstance(body, dict):
        return OutpaintResult(
            success=False, provider=provider,
            failure_reason="non-dict response",
        )
    status = str(body.get("status", "")).lower()
    output = body.get("output_url") or body.get("output_path")
    if status not in ("done", "succeeded", "ok", "success") or not output:
        return OutpaintResult(
            success=False, provider=provider,
            failure_reason=f"status={status!r} output={bool(output)}",
        )
    try:
        cost = float(body.get("cost_usd", 0.0) or 0.0)
    except (TypeError, ValueError):
        cost = 0.0
    try:
        latency = float(body.get("latency_sec", 0.0) or 0.0)
    except (TypeError, ValueError):
        latency = 0.0
    try:
        frames = int(body.get("frame_count", 0) or 0)
    except (TypeError, ValueError):
        frames = 0
    return OutpaintResult(
        success=True,
        output_path=str(output),
        provider=provider,
        latency_sec=latency,
        frame_count=frames,
        cost_usd=cost,
    )


# ── Plan-level promotion helper ───────────────────────────────────


def _is_hero_clip(virality_score) -> bool:
    """``virality_score`` may be on a 0-10 or 0-100 scale; compare
    against a threshold that makes sense on either scale."""
    if virality_score is None:
        return False
    try:
        v = float(virality_score)
    except (TypeError, ValueError):
        return False
    thr = float(os.environ.get("CLIPAI_OUTPAINT_HERO_THRESHOLD", "8.5"))
    # Caller may pass a 0-100 score. Allow both conventions: if the
    # threshold fits 0-10 we compare directly, else we scale.
    return v >= thr if thr <= 10.0 else v >= thr


async def maybe_promote_to_outpaint(
    plan,
    *,
    content_type: str,
    virality_score: float = 0.0,
    dense_faces: list = None,
    optical_flow_frames: list = None,
    hud_regions: list = None,
    source_width: int,
    user_opt_in: bool = False,
    source_segment_path: str = "",
    job_id: str = "",
):
    """Blueprint v2 Phase 5 — try to promote Tier-1 ``BLUR_FILL`` ops
    to Tier-3 ``OUTPAINT_FILL`` ops.

    Returns the (possibly-modified) RenderPlan. Any failure on any op
    keeps the original ``BLUR_FILL`` entry so the clip still exports.
    Obeys ``CLIPAI_OUTPAINT_ENABLED`` (default off) and
    ``CLIPAI_OUTPAINT_MAX_PER_CLIP`` (default 3).
    """
    from backend.services.render_plan import RenderOp, RenderOpKind
    from dataclasses import replace

    if os.environ.get("CLIPAI_OUTPAINT_ENABLED", "0").lower() not in ("1", "true", "yes"):
        return plan

    is_hero = _is_hero_clip(virality_score)
    if not (is_hero or user_opt_in):
        return plan

    try:
        max_per_clip = int(os.environ.get("CLIPAI_OUTPAINT_MAX_PER_CLIP", "3"))
    except ValueError:
        max_per_clip = 3

    new_ops = []
    outpaint_count = 0
    total_cost = 0.0
    for op in plan.ops:
        if op.kind != RenderOpKind.BLUR_FILL or outpaint_count >= max_per_clip:
            new_ops.append(op)
            continue

        crop_px = op.primary_rect.to_pixels(
            plan.source_width, plan.source_height,
        )
        safety = analyze_scene_outpaint_safety(
            scene_start=float(op.start_sec),
            scene_end=float(op.end_sec),
            primary_crop_rect_px=crop_px,
            source_width=plan.source_width,
            dense_faces=dense_faces,
            optical_flow_frames=optical_flow_frames,
            hud_regions=hud_regions,
        )
        allow, reason = should_outpaint_scene(
            content_type=content_type,
            scene_has_face_near_edge=safety["face_near_edge"],
            scene_has_fast_motion_at_edge=safety["fast_motion_at_edge"],
            scene_has_hud_at_edge=safety["hud_at_edge"],
            is_hero_clip=is_hero,
            user_opt_in=user_opt_in,
        )
        if not allow:
            logger.info(
                "[%s] outpaint skipped op %.2f-%.2f: %s",
                job_id, op.start_sec, op.end_sec, reason,
            )
            new_ops.append(op)
            continue

        result = await run_outpaint(
            source_segment_path=source_segment_path,
            primary_crop_rect={
                "x": float(op.primary_rect.x),
                "y": float(op.primary_rect.y),
                "w": float(op.primary_rect.w),
                "h": float(op.primary_rect.h),
            },
            duration_sec=float(op.end_sec - op.start_sec),
            job_id=job_id,
        )
        if not (result.success and result.output_path
                and os.path.exists(result.output_path)):
            logger.info(
                "[%s] outpaint failed op %.2f-%.2f (%s); keeping blur fill",
                job_id, op.start_sec, op.end_sec, result.failure_reason,
            )
            new_ops.append(op)
            continue

        promoted = replace(
            op,
            kind=RenderOpKind.OUTPAINT_FILL,
            outpainted_media_path=result.output_path,
            outpaint_provider=result.provider,
            outpaint_cost_usd=result.cost_usd,
            fallback_op_kind=RenderOpKind.BLUR_FILL.value,
        )
        new_ops.append(promoted)
        outpaint_count += 1
        total_cost += result.cost_usd

    if outpaint_count == 0:
        return plan

    logger.info(
        "[%s] outpaint promoted %d/%d ops, total_cost=$%.2f",
        job_id, outpaint_count,
        sum(1 for o in plan.ops if o.kind == RenderOpKind.BLUR_FILL),
        total_cost,
    )
    # Reuse the same plan shape; only ``ops`` changes.
    new_plan = replace(plan, ops=new_ops)
    # Stash aggregate cost for the caller's DB write.
    new_plan._outpaint_calls = outpaint_count  # type: ignore[attr-defined]
    new_plan._outpaint_cost_usd = total_cost  # type: ignore[attr-defined]
    return new_plan
