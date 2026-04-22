"""Stage 7 — Post-render VLM quality gate (Blueprint v2 Phase 0).

Samples the rendered 9:16 output, sends the frames to a VLM with a
single prompt that asks for a JSON list of flagged timestamps +
issues, parses the response, and returns a structured report that
the exporter stores alongside the clip record.

One VLM call per clip (cheap). Disabled by default — enable with
``CLIPAI_POST_RENDER_CRITIC=1`` once providers are configured.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from backend.services.reframe_config import ReframeConfig, get_default_config

logger = logging.getLogger(__name__)

_PROMPT = """You are a video editor reviewing a 9:16 vertical reframe.
Watch the attached frames (each has a timestamp) and list framing problems.

Look for:
- speaker off-frame or partially off-frame
- text or captions cut off at the edge
- jarring camera moves across scene cuts
- visible crop seams or artifacts
- subject suddenly leaving the frame
- chin or forehead clipped

Respond ONLY with a JSON array like:
[{"t": 3.2, "issue": "speaker's face is half off the right edge", "severity": "high"}]

Valid severity values: "low", "medium", "high". If no problems,
respond with an empty array: []
"""


@dataclass
class PostRenderIssue:
    t: float
    issue: str
    severity: str = "medium"  # "low" | "medium" | "high"


@dataclass
class PostRenderReport:
    ok: bool
    issues: list = field(default_factory=list)
    raw_response: str = ""
    sampled_frames: int = 0
    vlm_latency_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ok": bool(self.ok),
            "issues": [
                {"t": float(i.t), "issue": str(i.issue), "severity": str(i.severity)}
                for i in self.issues
            ],
            "sampled": int(self.sampled_frames),
            "latency_sec": float(self.vlm_latency_sec),
        }


def _parse_response(raw: str) -> list[PostRenderIssue]:
    if not raw:
        return []
    raw = raw.strip()
    # Strip ``` fences if VLM wrapped the JSON.
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    # Some models wrap with text before/after — grab the first [...] block.
    if not raw.startswith("["):
        start = raw.find("[")
        end = raw.rfind("]")
        if 0 <= start < end:
            raw = raw[start:end + 1]
    try:
        data = json.loads(raw)
    except Exception as e:
        logger.info("post-render critic: JSON parse failed: %s", e)
        return []
    if not isinstance(data, list):
        return []
    out: list[PostRenderIssue] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        try:
            t = float(entry.get("t", 0))
            issue = str(entry.get("issue", ""))[:200]
            sev = str(entry.get("severity", "medium"))[:10].lower()
            if sev not in ("low", "medium", "high"):
                sev = "medium"
            out.append(PostRenderIssue(t=t, issue=issue, severity=sev))
        except (ValueError, TypeError):
            continue
    return out


async def run_post_render_critic(
    rendered_path: str,
    *,
    duration_sec: float,
    orchestrator,
    job_id: str = "",
    config: Optional[ReframeConfig] = None,
    max_frames: int = 6,
) -> PostRenderReport:
    """Sample frames from the rendered clip, call the VLM, parse issues.

    ``orchestrator`` must expose an async ``vlm_critique(prompt, images,
    max_tokens)`` method returning the raw model text (empty string on
    failure — never raising). The return value is always a structured
    report; callers can treat it as advisory.
    """
    config = config or get_default_config()
    if not os.path.exists(rendered_path):
        return PostRenderReport(ok=True, raw_response="no-output")
    if duration_sec <= 0:
        return PostRenderReport(ok=True, raw_response="zero-duration")

    from backend.services.critic_loop import extract_frames_for_critic
    # Interval: spread max_frames over the clip (minimum 1s apart).
    interval = max(duration_sec / max(max_frames, 1), 1.0)
    try:
        samples = await asyncio.to_thread(
            extract_frames_for_critic,
            rendered_path,
            duration_sec=duration_sec,
            interval_sec=interval,
        )
    except Exception as e:
        logger.warning("[%s] post-render critic frame extract failed: %s", job_id, e)
        return PostRenderReport(ok=True, raw_response=f"extract-error: {e}")

    if not samples:
        return PostRenderReport(ok=True, raw_response="no-samples")

    images: list[dict] = []
    for s in samples[:max_frames]:
        if not (s.frame_path and os.path.exists(s.frame_path)):
            continue
        try:
            with open(s.frame_path, "rb") as f:
                images.append({"data": f.read(), "timestamp": float(s.t)})
        except OSError:
            continue

    if not images:
        return PostRenderReport(
            ok=True, raw_response="no-images",
            sampled_frames=len(samples),
        )

    t0 = time.monotonic()
    try:
        raw = await orchestrator.vlm_critique(
            prompt=_PROMPT,
            images=images,
            max_tokens=512,
        )
    except Exception as e:
        logger.warning("[%s] post-render critic VLM call failed: %s", job_id, e)
        return PostRenderReport(
            ok=True, raw_response=f"error: {e}",
            sampled_frames=len(samples),
        )
    latency = time.monotonic() - t0

    issues = _parse_response(raw or "")
    # NOT ok when there are >= 2 high-severity issues — those are the
    # ones worth raising a UI banner for.
    high = sum(1 for i in issues if i.severity == "high")
    return PostRenderReport(
        ok=(high < 2),
        issues=issues,
        raw_response=(raw or "")[:2000],
        sampled_frames=len(samples),
        vlm_latency_sec=latency,
    )
