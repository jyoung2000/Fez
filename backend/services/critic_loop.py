"""VLM / learned aesthetic critic loop.

Two-pass reframe critic that grades a rendered RenderPlan against a
cinematographic rubric and tells the segmenter which windows to
re-solve.

Modes (``config.critic_mode``):

    off      — pass-through, ``score_plan`` is a no-op.
    vlm      — sample frames from the rendered output, send to a VLM
               via :mod:`backend.services.providers.openrouter_provider`,
               parse per-sample scores on a 0-10 rubric.
    learned  — invoke the small ``aesthetic_scorer`` model from Phase 8
               (CPU, < 10 ms per frame). No external calls.
    both     — run both and average.

The VLM path is cached on ``hash(source_sha256 + render_plan_hash)``
so repeated bench / export runs on the same inputs cost zero. The
budget cap (``config.critic_budget_per_clip``) limits how many VLM
calls we'll make in one pass.

Windows that score below ``config.critic_threshold`` are emitted as
:class:`ReSolveRequest` that the segmenter consumes to tighten
constraints on a second pass — typically forcing more crop padding,
switching to a wider framing, or reverting to blur_fill.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from backend.services.reframe_config import ReframeConfig, get_default_config

logger = logging.getLogger(__name__)


# ── Rubric ─────────────────────────────────────────────────────────

# The VLM is asked to score each sampled frame on 5 axes, each 0-10.
# The final score is the min (a clip is only as good as its worst axis).
_RUBRIC_AXES = (
    "composition",      # thirds / balance / negative-space
    "subject_visible",  # is the subject fully in frame, no clip
    "headroom",         # headroom above face, chin-clip avoidance
    "lead_room",        # space in the direction subject is facing / moving
    "framing_choice",   # is the framing size appropriate for the beat
)

_VLM_PROMPT = """You are a professional video editor reviewing a 9:16 vertical reframe.

For the attached frame, rate each axis 0-10:
  composition: rule-of-thirds placement, overall balance
  subject_visible: subject fully in frame, no clipping
  headroom: appropriate space above head, no chin clipped
  lead_room: space in the direction subject faces / moves
  framing_choice: close-up / medium / wide appropriate for beat

Respond ONLY with a JSON object like {"composition": 8, "subject_visible": 9, ...}
and a one-line reason."""


# ── Inputs / outputs ──────────────────────────────────────────────


@dataclass
class FrameSample:
    t: float
    # Pre-extracted frame as bytes (caller supplies); kept generic so
    # we don't force a numpy / cv2 dependency on the metrics module.
    frame_bytes: Optional[bytes] = None
    # Alternatively a file path to the PNG we already rendered.
    frame_path: Optional[str] = None


@dataclass
class FrameScore:
    t: float
    score: float                 # 0-10, min across axes
    per_axis: dict = field(default_factory=dict)
    reason: str = ""


@dataclass
class ReSolveRequest:
    """One window flagged for re-solve on the next pass."""

    start: float
    end: float
    reason: str
    suggested_action: str   # "widen", "blur_fill", "hold", "more_padding"


@dataclass
class CriticReport:
    mode: str
    mean_score: float = 0.0
    per_frame: list[FrameScore] = field(default_factory=list)
    resolve_requests: list[ReSolveRequest] = field(default_factory=list)
    budget_used: int = 0


# ── Cache ─────────────────────────────────────────────────────────


def _cache_key(source_sha256: str, plan_hash: str, mode: str) -> str:
    return hashlib.sha256(
        f"{source_sha256}|{plan_hash}|{mode}".encode("utf-8")
    ).hexdigest()


def _cache_path(key: str, cache_dir: str) -> Path:
    return Path(cache_dir) / f"{key}.json"


def _load_cache(key: str, cache_dir: str) -> Optional[CriticReport]:
    path = _cache_path(key, cache_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return CriticReport(
            mode=data["mode"],
            mean_score=data["mean_score"],
            per_frame=[FrameScore(**f) for f in data["per_frame"]],
            resolve_requests=[ReSolveRequest(**r) for r in data["resolve_requests"]],
            budget_used=data["budget_used"],
        )
    except Exception as e:
        logger.warning("critic cache read failed for %s: %s", key, e)
        return None


def _store_cache(key: str, cache_dir: str, report: CriticReport) -> None:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    path = _cache_path(key, cache_dir)
    path.write_text(json.dumps({
        "mode": report.mode,
        "mean_score": report.mean_score,
        "per_frame": [asdict(f) for f in report.per_frame],
        "resolve_requests": [asdict(r) for r in report.resolve_requests],
        "budget_used": report.budget_used,
    }))


# ── VLM scorer (local-first) ─────────────────────────────────────


def _ollama_base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def _ollama_vision_model() -> str:
    return os.environ.get("OLLAMA_VISION_MODEL", "llava:7b")


def _ollama_reachable() -> bool:
    """Cheap check that a local Ollama server is up.

    Avoids waiting for a vision-model cold-start timeout just to fail
    into the fallback. Uses ``/api/tags`` which returns in < 5 ms.
    """
    try:
        import urllib.error
        import urllib.request
        req = urllib.request.Request(_ollama_base_url() + "/api/tags")
        with urllib.request.urlopen(req, timeout=0.5) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def _score_frame_vlm_ollama(sample: FrameSample) -> Optional[FrameScore]:
    """Score one frame using a local Ollama vision model. Returns
    ``None`` when Ollama isn't reachable so the caller can fall through
    to another backend."""
    if not sample.frame_path or not os.path.exists(sample.frame_path):
        return FrameScore(t=sample.t, score=5.0, reason="no-frame-file")
    if not _ollama_reachable():
        return None
    try:
        import base64
        import json as _json
        import urllib.request

        with open(sample.frame_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("ascii")
        payload = {
            "model": _ollama_vision_model(),
            "prompt": _VLM_PROMPT,
            "images": [img_b64],
            "stream": False,
            "options": {"temperature": 0.1},
        }
        data = _json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            _ollama_base_url() + "/api/generate",
            data=data, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            body = _json.loads(resp.read().decode("utf-8"))
        text = body.get("response", "")
        return _parse_vlm_response(sample.t, text)
    except Exception as e:
        logger.info("ollama critic scoring failed at t=%.2f: %s", sample.t, e)
        return None


def _score_frame_vlm_openrouter(sample: FrameSample) -> Optional[FrameScore]:
    """Score one frame via OpenRouter (cloud). Returns ``None`` if the
    provider isn't available so the caller can fall through."""
    try:
        from backend.services.providers import openrouter_provider  # type: ignore
    except Exception:
        return None
    if not sample.frame_path:
        return FrameScore(t=sample.t, score=5.0, reason="no-frame-file")
    try:
        resp = openrouter_provider.analyze_frames(
            [sample.frame_path], prompt=_VLM_PROMPT,
        )
        if isinstance(resp, list) and resp:
            resp = resp[0]
        text = resp.get("text") if isinstance(resp, dict) else str(resp)
        return _parse_vlm_response(sample.t, text)
    except Exception as e:
        logger.info("openrouter critic scoring failed at t=%.2f: %s", sample.t, e)
        return None


def _score_frame_vlm(
    sample: FrameSample,
    *,
    backend: str = "auto",
) -> FrameScore:
    """Dispatch one-frame VLM scoring to the configured backend.

    ``backend`` values:
      * ``auto`` (default): local Ollama if reachable, else OpenRouter,
        else heuristic fallback via :mod:`aesthetic_scorer`. Keeps the
        run 100% local whenever Ollama is up.
      * ``ollama``: force local; if unreachable fall through to the
        heuristic fallback (never reaches out to the network).
      * ``openrouter``: force OpenRouter; fail soft to heuristic.
    """
    if backend in ("auto", "ollama"):
        result = _score_frame_vlm_ollama(sample)
        if result is not None:
            return result
        if backend == "ollama":
            # Forced local mode — don't fall back to OpenRouter.
            return _score_frame_learned(sample)
    if backend in ("auto", "openrouter"):
        result = _score_frame_vlm_openrouter(sample)
        if result is not None:
            return result
    return _score_frame_learned(sample)


def _parse_vlm_response(t: float, text: str) -> FrameScore:
    """Parse ``{"composition": 8, ...}`` JSON out of the VLM response."""
    if not text:
        return FrameScore(t=t, score=5.0, reason="empty-response")
    # Crude: find the first {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return FrameScore(t=t, score=5.0, reason="no-json")
    try:
        obj = json.loads(text[start:end + 1])
    except Exception:
        return FrameScore(t=t, score=5.0, reason="bad-json")
    per_axis = {}
    for axis in _RUBRIC_AXES:
        v = obj.get(axis)
        if isinstance(v, (int, float)):
            per_axis[axis] = float(v)
    if not per_axis:
        return FrameScore(t=t, score=5.0, reason="no-axes")
    score = min(per_axis.values())
    return FrameScore(t=t, score=score, per_axis=per_axis,
                      reason=str(obj.get("reason", ""))[:200])


# ── Learned scorer (Phase 8) ──────────────────────────────────────


def _score_frame_learned(sample: FrameSample) -> FrameScore:
    try:
        from backend.services.aesthetic_scorer import score_frame
    except Exception:
        return FrameScore(t=sample.t, score=5.0, reason="learned-unavailable")
    try:
        s = float(score_frame(sample.frame_path))
        return FrameScore(t=sample.t, score=s * 10.0, reason="learned")
    except Exception as e:
        return FrameScore(t=sample.t, score=5.0, reason=f"learned-error:{e}")


# ── Resolver: low-score windows → ReSolveRequest ──────────────────


def _coalesce_low_score_windows(
    scores: list[FrameScore],
    threshold: float,
    *,
    min_gap_sec: float = 0.5,
) -> list[tuple[float, float, float]]:
    if not scores:
        return []
    low = [s for s in scores if s.score < threshold]
    if not low:
        return []
    low.sort(key=lambda s: s.t)
    windows: list[list[FrameScore]] = [[low[0]]]
    for s in low[1:]:
        if s.t - windows[-1][-1].t <= min_gap_sec:
            windows[-1].append(s)
        else:
            windows.append([s])
    return [
        (w[0].t, w[-1].t, sum(s.score for s in w) / len(w))
        for w in windows
    ]


def _suggest_action(mean_score: float) -> str:
    if mean_score < 3.0:
        return "blur_fill"
    if mean_score < 4.5:
        return "widen"
    if mean_score < 5.5:
        return "more_padding"
    return "hold"


# ── Public API ────────────────────────────────────────────────────


def score_plan(
    *,
    samples: list[FrameSample],
    source_sha256: str = "",
    plan_hash: str = "",
    config: Optional[ReframeConfig] = None,
) -> CriticReport:
    """Score a rendered plan and return re-solve requests.

    ``samples`` are pre-extracted frames at 1 / ``critic_sample_interval_sec``
    Hz. The pipeline is expected to render a preview pass first, sample
    those frames (e.g. with ffmpeg ``-ss`` + ``-vframes 1``), and pass
    the paths in.
    """
    config = config or get_default_config()
    mode = (config.critic_mode or "off").lower()
    if mode == "off" or not samples:
        return CriticReport(mode="off")

    # Cache check
    key = _cache_key(source_sha256, plan_hash, mode)
    cached = _load_cache(key, config.critic_cache_dir)
    if cached is not None:
        return cached

    budget = max(0, int(config.critic_budget_per_clip))
    sampled = samples[:budget] if mode == "vlm" else samples

    backend = getattr(config, "critic_vlm_backend", "auto")
    scores: list[FrameScore] = []
    for s in sampled:
        if mode == "vlm":
            scores.append(_score_frame_vlm(s, backend=backend))
        elif mode == "learned":
            scores.append(_score_frame_learned(s))
        elif mode == "both":
            a = _score_frame_vlm(s, backend=backend)
            b = _score_frame_learned(s)
            scores.append(FrameScore(
                t=s.t,
                score=(a.score + b.score) / 2.0,
                per_axis=a.per_axis,
                reason=f"vlm={a.score:.1f},learned={b.score:.1f}",
            ))
        else:
            scores.append(FrameScore(t=s.t, score=5.0, reason="unknown-mode"))

    mean_score = (
        sum(s.score for s in scores) / len(scores) if scores else 0.0
    )
    low_windows = _coalesce_low_score_windows(scores, config.critic_threshold)
    resolve_requests = [
        ReSolveRequest(
            start=t0, end=t1,
            reason=f"mean_score={ms:.2f} < {config.critic_threshold}",
            suggested_action=_suggest_action(ms),
        )
        for t0, t1, ms in low_windows
    ]

    report = CriticReport(
        mode=mode,
        mean_score=mean_score,
        per_frame=scores,
        resolve_requests=resolve_requests,
        budget_used=len(scores) if mode in ("vlm", "both") else 0,
    )
    try:
        _store_cache(key, config.critic_cache_dir, report)
    except Exception as e:
        logger.info("critic cache write failed: %s", e)
    return report


# ── Frame extraction helper ──────────────────────────────────────


def extract_frames_for_critic(
    rendered_path: str,
    *,
    duration_sec: float,
    interval_sec: float = 1.0,
    out_dir: str | None = None,
) -> list[FrameSample]:
    """Pull one frame every ``interval_sec`` from the rendered output
    using ffmpeg, returning :class:`FrameSample` entries pointing at
    PNG files on disk.

    Returns an empty list if ffmpeg is unavailable.
    """
    import shutil
    import subprocess
    import tempfile

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return []

    out_dir = out_dir or tempfile.mkdtemp(prefix="clipai_critic_")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    samples: list[FrameSample] = []
    t = 0.0
    idx = 0
    while t <= duration_sec:
        out_path = Path(out_dir) / f"s{idx:04d}.png"
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-ss", f"{t:.3f}", "-i", rendered_path,
            "-frames:v", "1", str(out_path),
        ]
        try:
            subprocess.run(cmd, check=True, timeout=20.0)
            if out_path.exists():
                samples.append(FrameSample(t=t, frame_path=str(out_path)))
        except Exception as e:
            logger.info("frame extract failed at t=%.2f: %s", t, e)
        t += interval_sec
        idx += 1
    return samples
