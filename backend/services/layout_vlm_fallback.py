"""Layout VLM fallback — Blueprint v2 Phase 2.

When the deterministic ``layout_confidence`` scorer returns a low
confidence on a scene, the layout engine escalates to a single
VLM call. The VLM receives:

  * four sampled frames from the scene,
  * the top-ranked candidates + rationales from the scorer,
  * ASD timeline + transcript excerpt (so it can reason about pacing
    without having to watch the whole clip),

and returns a JSON object with the chosen layout + optional subject
hint + short reason. Results are cached on ``hash(scene_id + source_sha)``
so re-renders are deterministic.

Disabled by default — the caller is expected to check
``CLIPAI_LAYOUT_VLM_ENABLED`` before invoking this module.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from backend.services.layout_confidence import VALID_LAYOUTS

logger = logging.getLogger(__name__)


_PROMPT_TEMPLATE = """You are a video editor choosing the 9:16 reframe layout for a scene.

Scene duration: {duration:.1f} seconds
Content type: {content_type}
Active-speaker timeline: {asd_timeline}
Transcript excerpt: {transcript}

Candidate layouts (ranked by the deterministic scorer):
{candidates}

You see {n_frames} sampled frames from this scene.

Pick ONE layout from: single, split, triple, pip, screenshare, gameplay, ken_burns, object_tracker.

If "single" or "object_tracker", name the subject (e.g. "left speaker", "ball", "main character"). Otherwise set subject to null.

Respond ONLY with a JSON object like:
{{"layout": "split", "subject": null, "reason": "two speakers alternate every 1-2 seconds"}}"""


@dataclass
class VLMLayoutDecision:
    layout: str
    subject: Optional[str] = None
    reason: str = ""
    raw: str = ""

    def to_dict(self) -> dict:
        return {
            "layout": str(self.layout),
            "subject": self.subject,
            "reason": str(self.reason),
            "raw": str(self.raw)[:1000],
        }


_CACHE_DIR_DEFAULT = "/tmp/clipai_layout_vlm_cache"


def _cache_key(scene_id: str, source_sha: str) -> str:
    return hashlib.sha256(
        f"{scene_id}|{source_sha}".encode("utf-8"),
    ).hexdigest()[:24]


def _cache_read(key: str, cache_dir: str) -> Optional[VLMLayoutDecision]:
    p = Path(cache_dir) / f"{key}.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        return VLMLayoutDecision(
            layout=str(data.get("layout", "")),
            subject=data.get("subject"),
            reason=str(data.get("reason", "")),
            raw=str(data.get("raw", "")),
        )
    except Exception as e:
        logger.info("layout VLM cache read failed for %s: %s", key, e)
        return None


def _cache_write(key: str, cache_dir: str, decision: VLMLayoutDecision) -> None:
    try:
        p = Path(cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / f"{key}.json").write_text(json.dumps(decision.to_dict()))
    except Exception as e:
        logger.info("layout VLM cache write failed: %s", e)


def _parse_decision(raw: str) -> Optional[VLMLayoutDecision]:
    if not raw:
        return None
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(),
                      flags=re.MULTILINE).strip()
    # Find the first {...} block if VLM wrapped in prose.
    if not stripped.startswith("{"):
        lo = stripped.find("{")
        hi = stripped.rfind("}")
        if 0 <= lo < hi:
            stripped = stripped[lo:hi + 1]
    try:
        data = json.loads(stripped)
    except Exception as e:
        logger.info("layout VLM JSON parse failed: %s", e)
        return None
    if not isinstance(data, dict):
        return None
    layout = str(data.get("layout", "")).lower().strip()
    if layout not in VALID_LAYOUTS:
        logger.info("layout VLM returned invalid layout %r", layout)
        return None
    subject = data.get("subject")
    if subject is not None:
        subject = str(subject)[:120]
    return VLMLayoutDecision(
        layout=layout,
        subject=subject,
        reason=str(data.get("reason", ""))[:200],
        raw=raw[:1000],
    )


async def request_layout_from_vlm(
    *,
    scene_start: float,
    scene_end: float,
    scene_id: str,
    source_sha: str,
    video_path: str,
    orchestrator,
    candidates: list,
    content_type: str,
    transcript_excerpt: str = "",
    asd_timeline: str = "",
    config=None,
    cache_dir: Optional[str] = None,
    max_frames: int = 4,
) -> Optional[VLMLayoutDecision]:
    """Ask the VLM to pick a layout. Returns ``None`` on any failure so
    callers fall back to the deterministic top-ranked candidate."""
    resolved_cache = (
        cache_dir
        or (getattr(config, "layout_vlm_cache_dir", None) if config else None)
        or _CACHE_DIR_DEFAULT
    )
    key = _cache_key(scene_id, source_sha)
    cached = _cache_read(key, resolved_cache)
    if cached is not None:
        logger.info(
            "layout VLM cache HIT scene_id=%s layout=%s",
            scene_id, cached.layout,
        )
        return cached

    if not os.path.exists(video_path):
        return None
    duration = max(0.0, float(scene_end) - float(scene_start))
    if duration <= 0:
        return None

    # Sample ~``max_frames`` frames evenly across the scene window.
    # ``extract_frames_for_critic`` seeks from t=0 in the source, so we
    # trim its output to the scene window.
    from backend.services.critic_loop import extract_frames_for_critic
    interval = max(duration / max(max_frames, 1), 0.5)
    samples = await asyncio.to_thread(
        extract_frames_for_critic,
        video_path,
        duration_sec=float(scene_end),
        interval_sec=interval,
    )
    window = [s for s in samples if scene_start <= s.t <= scene_end]
    window = window[:max_frames]
    if len(window) < 2:
        return None

    images: list[dict] = []
    for s in window:
        if not (s.frame_path and os.path.exists(s.frame_path)):
            continue
        try:
            with open(s.frame_path, "rb") as f:
                images.append({"data": f.read(), "timestamp": float(s.t)})
        except OSError:
            continue
    if len(images) < 2:
        return None

    candidates_str = "\n".join(
        f"  {i + 1}. {c.layout} (score={c.score:.2f}) — {c.rationale}"
        for i, c in enumerate(list(candidates)[:4])
    )
    prompt = _PROMPT_TEMPLATE.format(
        duration=duration,
        content_type=content_type or "generic",
        asd_timeline=(asd_timeline or "")[:500],
        transcript=(transcript_excerpt or "")[:500],
        candidates=candidates_str,
        n_frames=len(images),
    )

    try:
        raw = await orchestrator.vlm_critique(
            prompt=prompt, images=images, max_tokens=256,
        )
    except Exception as e:
        logger.warning("layout VLM call failed: %s", e)
        return None

    decision = _parse_decision(raw or "")
    if decision is None:
        logger.info("layout VLM returned unusable response scene_id=%s", scene_id)
        return None

    _cache_write(key, resolved_cache, decision)
    logger.info(
        "layout VLM: scene_id=%s layout=%s subject=%s reason=%s",
        scene_id, decision.layout, decision.subject, decision.reason,
    )
    return decision
