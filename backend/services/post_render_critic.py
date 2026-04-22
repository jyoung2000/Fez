"""Stage 7 — Post-render VLM quality gate.

Blueprint v2 Phase 0 wired the read-only path: sample frames, send to
VLM, store issues on the clip record.

Blueprint v2 Phase 3 extends the prompt with an issue taxonomy so the
auto-fix module (``post_render_autofix.py``) can decide which issues
are fixable via a segment re-solve vs structural (source content
can't be reframed without losing the subject).

One VLM call per clip (cheap). Gated by ``CLIPAI_POST_RENDER_CRITIC=1``.
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


# ── Phase 3 issue taxonomy ────────────────────────────────────────
#
# Each VLM-flagged issue lands in one of these buckets. The autofix
# strategy table maps auto-fixable kinds to a concrete config override
# + optional crop-widen + window size. Structural kinds mark the clip
# ``low_confidence`` and down-weight its virality score (Task 3.6).

class IssueKind:
    # ── Auto-fixable (re-solve a segment with new params) ──
    TIGHT_FRAMING = "tight_framing"             # face clipped at edge
    PAN_ACROSS_CUT = "pan_across_cut"           # solver bled across shot boundary
    WRONG_SUBJECT = "wrong_subject"             # tracking the wrong face / object
    JITTER = "jitter"                            # oscillating crop
    TEXT_CUT_OFF = "text_cut_off"                # overlay text partially off
    HEAD_OR_CHIN_CLIP = "head_or_chin_clip"     # headroom / chin violation

    # ── Structural (source can't be reframed) ──
    SUBJECT_LEFT_FRAME = "subject_left_frame"
    OUTPAINT_HALLUC = "outpaint_hallucination"
    HUD_ADJACENT = "hud_adjacent_gameplay"

    # ── Cosmetic (no action) ──
    SINGLE_FRAME_GLITCH = "single_frame_glitch"

    # ── Unknown / unclassified (default bucket) ──
    OTHER = "other"


AUTO_FIX_KINDS: frozenset = frozenset({
    IssueKind.TIGHT_FRAMING,
    IssueKind.PAN_ACROSS_CUT,
    IssueKind.WRONG_SUBJECT,
    IssueKind.JITTER,
    IssueKind.TEXT_CUT_OFF,
    IssueKind.HEAD_OR_CHIN_CLIP,
})

STRUCTURAL_KINDS: frozenset = frozenset({
    IssueKind.SUBJECT_LEFT_FRAME,
    IssueKind.OUTPAINT_HALLUC,
    IssueKind.HUD_ADJACENT,
})

COSMETIC_KINDS: frozenset = frozenset({
    IssueKind.SINGLE_FRAME_GLITCH,
})

_KNOWN_KINDS: frozenset = (
    AUTO_FIX_KINDS | STRUCTURAL_KINDS | COSMETIC_KINDS | frozenset({IssueKind.OTHER})
)


_PROMPT = """You are a video editor reviewing a 9:16 vertical reframe.
Watch the attached frames (each has a timestamp) and list framing problems.

For each issue, classify it as ONE of:
- "tight_framing": subject partially clipped at a frame edge
- "pan_across_cut": camera panned across a visible scene cut (jarring)
- "wrong_subject": the wrong face / object is being tracked
- "jitter": crop oscillates back and forth on a static scene
- "text_cut_off": overlay text or captions are partially out of frame
- "head_or_chin_clip": forehead or chin is cut off
- "subject_left_frame": the subject has left the frame entirely (source issue, not fixable)
- "outpaint_hallucination": visible outpaint / generative-fill artifact
- "hud_adjacent_gameplay": gameplay HUD / minimap is cropped or unreadable
- "single_frame_glitch": a one-frame artifact, not worth fixing

Respond ONLY with a JSON array like:
[{"t": 3.2, "kind": "tight_framing", "severity": "high", "issue": "left speaker is half off the right edge"}]

Valid severity values: "low", "medium", "high". If no problems: []
"""


@dataclass
class PostRenderIssue:
    t: float
    issue: str
    severity: str = "medium"  # "low" | "medium" | "high"
    # Phase 3 taxonomy bucket. Defaults to ``other`` when the VLM
    # omits the field or returns an unknown value so the autofix
    # strategy table can ignore the issue safely.
    kind: str = IssueKind.OTHER


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
                {
                    "t": float(i.t),
                    "kind": str(i.kind),
                    "severity": str(i.severity),
                    "issue": str(i.issue),
                }
                for i in self.issues
            ],
            "sampled": int(self.sampled_frames),
            "latency_sec": float(self.vlm_latency_sec),
        }

    def structural_issues(self) -> list:
        return [i for i in self.issues if i.kind in STRUCTURAL_KINDS]

    def auto_fixable_issues(self) -> list:
        return [i for i in self.issues if i.kind in AUTO_FIX_KINDS]


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
            kind = str(entry.get("kind", IssueKind.OTHER)).lower().strip()
            if kind not in _KNOWN_KINDS:
                kind = IssueKind.OTHER
            out.append(PostRenderIssue(
                t=t, issue=issue, severity=sev, kind=kind,
            ))
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
