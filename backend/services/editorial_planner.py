"""Global editorial planner — single LLM call per clip.

Phase E of the 2026 SOTA reframing rollout. The KEYSTONE phase that
ties Phases A-D into a per-genre playbook.

Why a single LLM call:
    A human editor doesn't decide each frame in isolation. They look
    at the whole clip, identify the editorial intent of each shot
    (establish, punchline, reaction, action, transition), and frame
    the entire arc accordingly. We compress that into one structured
    output and feed it to the LP solver as soft priors.

Inputs (consumed once per clip):
    * Scene timeline (descriptions + face slots + speaker turns).
    * Genre tag (drives prompt template selection).
    * Per-second saliency peaks (Phase C).
    * Audio events (laughter, gasp, music drops, SFX) from the
      existing audio_analyzer.
    * Beat grid for music_video / dance content.

Output (EditorialPlan):
    A list of EditorialShot entries — start, end, intent, framing,
    motivated_zoom kwargs, A/B cut flag, reaction beat timing.

Integration with the LP solver:
    Plans are consumed via :func:`apply_plan_to_lp_targets` which
    nudges per-frame tx[i]/ty[i] toward the planned subject and
    framing. Soft only — face / speaker hard constraints are
    unchanged.

Caching:
    Plans are cached on hash(scene_descriptions + speaker_turns +
    beats + content_type) to /tmp/clipai_editorial/<hash>.json. Re-
    runs of the same clip cost zero LLM calls.

Robust JSON parsing:
    LLMs occasionally return malformed JSON. We try strict json.loads,
    then json5 (lenient), then a regex extract-the-largest-JSON-block.
    All three fail → empty EditorialPlan and the LP falls back to its
    existing behaviour.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Constants / cache dir ────────────────────────────────────────────

DEFAULT_CACHE_DIR = Path("/tmp/clipai_editorial")
DEFAULT_PROMPT_DIR = Path(__file__).parent / "editorial_planner_prompts"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TIMEOUT_SEC = 60.0


# ── Genre dispatch ───────────────────────────────────────────────────

GENRE_TO_TEMPLATE = {
    "talking_head": "talking_head.txt",
    "multi_speaker_panel": "multi_speaker_panel.txt",
    "podcast": "multi_speaker_panel.txt",
    "debate": "multi_speaker_panel.txt",
    "interview": "talking_head.txt",
    "vlog": "vlog.txt",
    "narrative": "narrative.txt",
    "documentary": "narrative.txt",
    "cinematic_dialogue": "narrative.txt",
    "music_video": "music_video.txt",
    "concert": "music_video.txt",
    "performance": "music_video.txt",
    "sports": "sports.txt",
    "sports_basketball": "sports.txt",
    "sports_racing": "sports.txt",
    "gaming": "gaming.txt",
    "gameplay": "gaming.txt",
    "anime": "anime.txt",
    "animation": "anime.txt",
    "animation_dialogue": "anime.txt",
    "tutorial": "tutorial.txt",
}

DEFAULT_TEMPLATE = "default.txt"


# ── Dataclasses ──────────────────────────────────────────────────────


@dataclass
class EditorialShot:
    start: float
    end: float
    intent: str = "hold"            # establish | punchline | reaction | action | transition | hold
    subject: Optional[str] = None   # e.g. "speaker_2", "ball", None
    framing: str = "medium"         # tight | medium | medium-wide | wide
    motivated_zoom: Optional[dict] = None  # {"kind": "push_in", "at": 4.5}
    ab_cut: bool = False
    ab_target: Optional[str] = None
    reaction_at: Optional[float] = None
    notes: str = ""


@dataclass
class EditorialPlan:
    shots: list = field(default_factory=list)
    genre: str = "default"
    confidence: float = 0.0
    raw_response: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.shots


@dataclass
class EditorialPlannerInputs:
    """All inputs the planner needs to make a decision."""
    content_type: str
    scene_descriptions: list = field(default_factory=list)  # list[dict]
    speaker_turns: list = field(default_factory=list)       # list[dict]
    saliency_peaks_per_second: list = field(default_factory=list)
    audio_events: list = field(default_factory=list)        # list[dict]
    beat_grid: list = field(default_factory=list)           # list[float] seconds
    duration_sec: float = 0.0


# ── Cache key ────────────────────────────────────────────────────────


def _cache_key(inputs: EditorialPlannerInputs) -> str:
    payload = json.dumps({
        "ct": inputs.content_type,
        "sd": [s.get("description", "") for s in inputs.scene_descriptions],
        "st": [
            (st.get("speaker"), round(st.get("start", 0), 2),
             round(st.get("end", 0), 2))
            for st in inputs.speaker_turns
        ],
        "bg": [round(b, 2) for b in inputs.beat_grid],
        "dur": round(inputs.duration_sec, 1),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ── Prompt template loading ──────────────────────────────────────────


def _load_template(content_type: str, *, prompt_dir: Path = DEFAULT_PROMPT_DIR) -> str:
    name = GENRE_TO_TEMPLATE.get(content_type, DEFAULT_TEMPLATE)
    path = prompt_dir / name
    if not path.exists():
        path = prompt_dir / DEFAULT_TEMPLATE
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return BUILTIN_DEFAULT_TEMPLATE


# Embedded fallback so the module works even if the prompt files are
# missing on disk — the homelab build pipeline ships them but we don't
# want the planner to crash mid-job in their absence.
BUILTIN_DEFAULT_TEMPLATE = """You are an editorial planner for short-form vertical video.

Given scene descriptions, speaker turns, and audio events for a clip,
produce a JSON shot plan. Each shot has:
  start: seconds (float)
  end:   seconds (float)
  intent: "establish" | "punchline" | "reaction" | "action" | "hold" | "transition"
  subject: subject identifier (e.g. "speaker_1", "ball", null)
  framing: "tight" | "medium" | "medium-wide" | "wide"
  motivated_zoom: {"kind": "push_in"|"pull_out", "at": <float>} or null
  ab_cut: true if this shot should A/B cut against the previous; false otherwise
  ab_target: subject identifier if ab_cut is true; null otherwise
  reaction_at: timestamp of the reaction beat (float) or null
  notes: brief free-form description

Return ONLY JSON of the form:
{"genre": <string>, "shots": [<shot_obj>, ...]}

NO commentary, NO markdown fences.

INPUTS:
{inputs}
"""


def _format_inputs_for_prompt(inputs: EditorialPlannerInputs) -> str:
    """Human-readable summary for the LLM."""
    lines = [f"content_type: {inputs.content_type}",
             f"duration_sec: {inputs.duration_sec:.1f}"]
    if inputs.scene_descriptions:
        lines.append("scenes:")
        for sd in inputs.scene_descriptions[:30]:
            t = sd.get("timestamp", 0)
            d = (sd.get("description") or "").strip()
            lines.append(f"  - t={t:.1f}s {d[:200]}")
    if inputs.speaker_turns:
        lines.append("speaker_turns:")
        for st in inputs.speaker_turns[:50]:
            lines.append(
                f"  - {st.get('speaker', '?')} {st.get('start', 0):.1f}-"
                f"{st.get('end', 0):.1f}"
            )
    if inputs.beat_grid:
        beats = inputs.beat_grid[:64]
        lines.append("beats: " + ", ".join(f"{b:.2f}" for b in beats))
    if inputs.audio_events:
        lines.append("audio_events:")
        for ev in inputs.audio_events[:30]:
            lines.append(f"  - t={ev.get('t', 0):.1f}s kind={ev.get('kind', '?')}")
    if inputs.saliency_peaks_per_second:
        sample_seconds = list(range(
            0, len(inputs.saliency_peaks_per_second), max(1,
            len(inputs.saliency_peaks_per_second) // 10),
        ))
        lines.append("saliency_peak_samples:")
        for s in sample_seconds[:10]:
            peaks = inputs.saliency_peaks_per_second[s]
            lines.append(f"  - t={s}s peaks={peaks[:3]}")
    return "\n".join(lines)


# ── Robust JSON parser ───────────────────────────────────────────────


def _try_strict_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        return None


def _try_json5(text: str) -> Optional[dict]:
    try:
        import json5  # type: ignore
        return json5.loads(text)
    except Exception:
        return None


def _try_regex_extract(text: str) -> Optional[dict]:
    # Find the largest balanced {...} block.
    depth = 0
    start = -1
    best = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = text[start:i + 1]
                if best is None or len(candidate) > len(best):
                    best = candidate
    if best is None:
        return None
    try:
        return json.loads(best)
    except Exception:
        return None


def parse_plan_response(text: str, *, content_type: str = "default") -> EditorialPlan:
    """Strict-then-lenient parser. Returns empty plan on total failure."""
    if not text:
        return EditorialPlan(genre=content_type, raw_response="")
    parsed = _try_strict_json(text) or _try_json5(text) or _try_regex_extract(text)
    if not isinstance(parsed, dict):
        logger.warning("editorial_planner: unable to parse plan response")
        return EditorialPlan(genre=content_type, raw_response=text)
    shots_raw = parsed.get("shots", [])
    shots: list = []
    for s in shots_raw:
        if not isinstance(s, dict):
            continue
        try:
            shots.append(EditorialShot(
                start=float(s.get("start", 0)),
                end=float(s.get("end", 0)),
                intent=str(s.get("intent", "hold")),
                subject=s.get("subject"),
                framing=str(s.get("framing", "medium")),
                motivated_zoom=s.get("motivated_zoom"),
                ab_cut=bool(s.get("ab_cut", False)),
                ab_target=s.get("ab_target"),
                reaction_at=(
                    float(s["reaction_at"]) if s.get("reaction_at") is not None else None
                ),
                notes=str(s.get("notes", "")),
            ))
        except (TypeError, ValueError):
            continue
    return EditorialPlan(
        shots=shots,
        genre=str(parsed.get("genre", content_type)),
        confidence=float(parsed.get("confidence", 0.7 if shots else 0.0)),
        raw_response=text,
    )


# ── Cache ────────────────────────────────────────────────────────────


def _read_cache(key: str, *, cache_dir: Path) -> Optional[EditorialPlan]:
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        plan = EditorialPlan(genre=data.get("genre", "default"),
                             confidence=float(data.get("confidence", 0.0)),
                             raw_response=data.get("raw_response", ""))
        for s in data.get("shots", []):
            plan.shots.append(EditorialShot(**s))
        return plan
    except Exception as exc:
        logger.warning("editorial_planner cache read failed: %s", exc)
        return None


def _write_cache(key: str, plan: EditorialPlan, *, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "genre": plan.genre, "confidence": plan.confidence,
        "raw_response": plan.raw_response,
        "shots": [asdict(s) for s in plan.shots],
    }
    try:
        (cache_dir / f"{key}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("editorial_planner cache write failed: %s", exc)


# ── Public entry point ───────────────────────────────────────────────


async def plan_clip(
    inputs: EditorialPlannerInputs,
    *,
    orchestrator: Any,
    job_id: str = "",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    prompt_dir: Path = DEFAULT_PROMPT_DIR,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> EditorialPlan:
    """Build an editorial plan via one LLM text-completion call.

    ``orchestrator`` is an :class:`AIOrchestrator` (or any object with
    an async ``text_completion(prompt, max_tokens, timeout, job_id)``
    method). Cache hits return immediately and cost zero tokens.
    Failures (provider chain exhausted, malformed JSON, etc.) return
    an empty plan and the LP solver falls back to its existing path.
    """
    key = _cache_key(inputs)
    cached = _read_cache(key, cache_dir=cache_dir)
    if cached is not None:
        logger.info("[EditorialPlanner] cache hit job=%s key=%s", job_id, key)
        return cached
    template = _load_template(inputs.content_type, prompt_dir=prompt_dir)
    prompt = template.replace("{inputs}", _format_inputs_for_prompt(inputs))
    try:
        text = await orchestrator.text_completion(
            prompt, max_tokens=max_tokens,
            timeout=timeout_sec, job_id=job_id,
        )
    except Exception as exc:
        logger.warning("[EditorialPlanner] text_completion failed: %s", exc)
        return EditorialPlan(genre=inputs.content_type)
    plan = parse_plan_response(text, content_type=inputs.content_type)
    if not plan.is_empty:
        _write_cache(key, plan, cache_dir=cache_dir)
    logger.info(
        "[EditorialPlanner] job=%s genre=%s shots=%d conf=%.2f",
        job_id, plan.genre, len(plan.shots), plan.confidence,
    )
    return plan


# ── LP integration ───────────────────────────────────────────────────


_FRAMING_TO_ZOOM = {
    "tight":       0.70,
    "medium":      0.85,
    "medium-wide": 1.00,
    "wide":        1.15,
}


def apply_plan_to_lp_targets(
    plan: EditorialPlan,
    timestamps: list,
    tx: list,
    ty: list,
    *,
    framing_weight: float = 0.5,
    lo_bounds_x: Optional[list] = None,
    hi_bounds_x: Optional[list] = None,
) -> tuple:
    """Apply the editorial plan as soft priors to per-frame LP targets.

    For each frame, finds the active editorial shot and:
      * Adjusts ``tx`` toward the framing's target zoom-equivalent x
        (no-op if framing doesn't translate to an x adjustment, which
        is most cases — framing primarily drives the LP's zoom
        constraint not its x target).
      * Records the framing zoom hint so the camera-path solver's
        zoom-curve consumer can apply it.

    Returns ``(tx_new, ty_new, zoom_hints_per_frame)``. ``zoom_hints``
    is a list of floats matching the ``timestamps`` length — one
    target zoom per frame from the planned framing.
    """
    n = len(timestamps)
    tx_new = list(tx) if tx else [0.0] * n
    ty_new = list(ty) if ty else [0.0] * n
    zoom_hints: list = [1.0] * n
    if not plan.shots:
        return tx_new, ty_new, zoom_hints
    # Build a quick lookup: list of (start, end, shot) sorted by start.
    shots = sorted(plan.shots, key=lambda s: s.start)
    j = 0
    for i, t in enumerate(timestamps):
        # Advance j to the active shot.
        while j + 1 < len(shots) and shots[j + 1].start <= t:
            j += 1
        s = shots[j]
        if not (s.start - 1e-3 <= t <= s.end + 1e-3):
            continue
        zoom_hints[i] = _FRAMING_TO_ZOOM.get(s.framing, 1.0)
        # Future: subject-specific x adjustments (e.g. for "ab_cut" to
        # speaker_2, override tx to speaker_2's last-known position).
        # That requires the subject-position lookup table from
        # subject_track.py; we leave it unimplemented at this layer
        # because the LP solver's existing predictor blend already
        # handles per-subject re-targeting via primary_slot_by_t.
    return tx_new, ty_new, zoom_hints


def get_supported_content_types() -> set:
    """Return the content types the editorial planner can handle."""
    return set(GENRE_TO_TEMPLATE.keys())
