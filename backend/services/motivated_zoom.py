"""Motivated zoom scheduler.

Humans zoom in to build emotional intensity and pull out to reveal
scale / context. Our pipeline today never zooms for editorial reasons —
``STATIONARY_ZOOM_STEPS`` in ``camera_solver`` is a safety-fit widener,
not a dolly.

This module decides **where** a push-in or pull-out belongs, and for
**how long**. It consumes signals the pipeline already produces:

  * Transcript word-level timing (pauses + punctuation).
  * Audio volume / RMS peaks from ``audio_analyzer``.
  * Shot duration from ``shot_detector``.
  * Intent confidence from ``intent_tracker``.
  * Optical-flow motion energy from ``optical_flow``.

and outputs a list of :class:`ZoomMoment` entries with start time,
duration, direction (push-in / pull-out), and target scale. The
RenderPlan builder turns each into a ``MOTIVATED_PUSH_IN`` or
``MOTIVATED_PULL_OUT`` op, with ``motion_path`` ramping from current
crop to the zoomed crop over the duration.

Rules (from the spec):

  * Push in on:
      - A pause > 400 ms following a rising-pitch word.
      - A punch-line / sentiment peak inside a cinematic_dialogue /
        narrative / vlog segment.
      - Emotional lexical cue (laugh, curse, realization word) +
        volume peak.
  * Pull out on:
      - A scene reveal (new subject enters frame after > 1 s absence).
      - A sudden motion-energy peak (action beat, "spectacle").
  * Guardrails:
      - Max 1 zoom per ``zoom_min_gap_sec`` seconds.
      - Duration clamped to ``[zoom_min_duration_sec, zoom_max_duration_sec]``.
      - Content-type gating: off for ``gaming``, ``music_video``,
        ``multi_speaker_panel``. The per-content-type overrides in
        ``reframe_config`` already encode this via
        ``zoom_push_in_max_scale == 1.0``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from backend.services.reframe_config import ReframeConfig, get_default_config

logger = logging.getLogger(__name__)


class ZoomKind(str, Enum):
    PUSH_IN = "push_in"
    PULL_OUT = "pull_out"


@dataclass
class ZoomMoment:
    start: float
    end: float
    kind: ZoomKind
    # Target scale: > 1.0 = push-in (crop shrinks); < 1.0 = pull-out.
    target_scale: float
    reason: str = ""


# ── Input dataclasses ────────────────────────────────────────────


@dataclass
class WordTiming:
    start: float
    end: float
    text: str
    # Optional pitch cue: True when the word is on a rising intonation.
    rising: bool = False
    # Optional emotion label ("laughter", "anger", "surprise", ...)
    emotion: Optional[str] = None


@dataclass
class AudioPeak:
    t: float
    loudness_db: float


@dataclass
class MotionBeat:
    t: float
    intensity: float   # 0..1, relative to the rest of the clip


@dataclass
class SubjectEntrance:
    t: float
    slot_id: int


# ── Detection helpers ─────────────────────────────────────────────


# Words that commonly sit on emotional peaks. Not exhaustive; works in
# combination with pitch / volume cues so false positives are rare.
_EMOTIONAL_WORDS = {
    "wait", "stop", "look", "listen",
    "god", "damn", "hell", "fuck", "shit",
    "wow", "whoa", "oh", "no", "yes", "holy",
    "can't", "cannot", "never", "always",
}


def _punchline_moments(
    words: list[WordTiming],
    audio_peaks: list[AudioPeak],
) -> list[tuple[float, str]]:
    """Return ``[(t, reason)]`` candidate push-in moments from words."""
    moments: list[tuple[float, str]] = []
    peak_times = sorted(p.t for p in audio_peaks)

    def _peak_near(t: float, window: float = 0.3) -> bool:
        for pt in peak_times:
            if abs(pt - t) < window:
                return True
        return False

    for i, w in enumerate(words):
        # Pause after a rising-pitch word
        if w.rising:
            next_w = words[i + 1] if i + 1 < len(words) else None
            pause = (next_w.start - w.end) if next_w else 1.0
            if pause > 0.4:
                moments.append((w.end, "pause-after-rising"))
        # Emotional lexical cue + volume peak
        text = (w.text or "").lower().strip(".,!?")
        if text in _EMOTIONAL_WORDS and _peak_near(w.end):
            moments.append((w.end, f"emotional-word:{text}"))
        # Explicit emotion label
        if w.emotion in ("surprise", "anger", "realization"):
            moments.append((w.end, f"emotion:{w.emotion}"))
    return moments


def _reveal_moments(
    entrances: list[SubjectEntrance],
    motion_beats: list[MotionBeat],
) -> list[tuple[float, str]]:
    """Return candidate pull-out moments."""
    moments: list[tuple[float, str]] = []
    for e in entrances:
        moments.append((e.t, f"subject-{e.slot_id}-enters"))
    # Top 2 motion beats
    ranked = sorted(motion_beats, key=lambda b: -b.intensity)
    for mb in ranked[:2]:
        if mb.intensity > 0.7:
            moments.append((mb.t, f"motion-beat@{mb.intensity:.2f}"))
    return moments


# ── Public API ────────────────────────────────────────────────────


def plan_motivated_zooms(
    *,
    clip_duration: float,
    words: list[WordTiming] | None = None,
    audio_peaks: list[AudioPeak] | None = None,
    motion_beats: list[MotionBeat] | None = None,
    entrances: list[SubjectEntrance] | None = None,
    shot_boundaries: list[float] | None = None,
    content_type: str = "",
    config: Optional[ReframeConfig] = None,
) -> list[ZoomMoment]:
    """Plan a non-conflicting list of zoom moments over the clip."""
    config = (config or get_default_config()).for_content(content_type)

    # Content-type gating: when zoom scales collapse to 1.0, disable.
    push_max = config.zoom_push_in_max_scale
    pull_max = config.zoom_pull_out_max_scale
    if push_max <= 1.001 and pull_max >= 0.999:
        return []

    words = words or []
    audio_peaks = audio_peaks or []
    motion_beats = motion_beats or []
    entrances = entrances or []
    shot_boundaries = shot_boundaries or []

    push_candidates = _punchline_moments(words, audio_peaks) if push_max > 1.001 else []
    pull_candidates = _reveal_moments(entrances, motion_beats) if pull_max < 0.999 else []

    # Merge & rank by time
    combined: list[tuple[float, str, ZoomKind]] = []
    for t, reason in push_candidates:
        combined.append((t, reason, ZoomKind.PUSH_IN))
    for t, reason in pull_candidates:
        combined.append((t, reason, ZoomKind.PULL_OUT))
    combined.sort(key=lambda x: x[0])

    # Guard rails: no zoom across a shot boundary, respect min gap.
    out: list[ZoomMoment] = []
    min_gap = config.zoom_min_gap_sec
    duration_lo = config.zoom_min_duration_sec
    duration_hi = config.zoom_max_duration_sec

    def _crosses_shot(t0: float, t1: float) -> bool:
        for s in shot_boundaries:
            if t0 < s < t1:
                return True
        return False

    for t, reason, kind in combined:
        if out and t - out[-1].end < min_gap:
            continue
        duration = (duration_hi + duration_lo) * 0.5
        end = min(clip_duration, t + duration)
        if end - t < duration_lo:
            continue
        if _crosses_shot(t, end):
            # Shorten to end at the shot boundary
            for s in shot_boundaries:
                if t < s < end:
                    end = s - 0.05
                    break
            if end - t < duration_lo:
                continue
        target_scale = push_max if kind == ZoomKind.PUSH_IN else pull_max
        out.append(ZoomMoment(
            start=t,
            end=end,
            kind=kind,
            target_scale=target_scale,
            reason=reason,
        ))
    return out


# ── Helper for RenderPlan builder ─────────────────────────────────


def zoom_rect_at(base_rect: dict, target_scale: float) -> dict:
    """Return a Rect dict representing ``base_rect`` scaled by
    ``target_scale`` about its center. Used by the RenderPlan builder
    to produce the terminal motion_path keypoint for a zoom op.

    ``base_rect`` is a dict with ``{x, y, w, h}`` in source-frame
    fractions; ``target_scale > 1`` zooms in (w, h shrink), ``< 1`` zooms
    out (w, h grow). Cropped outputs are clamped to [0, 1].
    """
    s = max(0.05, float(target_scale))
    w = max(0.01, min(1.0, base_rect["w"] / s))
    h = max(0.01, min(1.0, base_rect["h"] / s))
    cx = base_rect["x"] + base_rect["w"] * 0.5
    cy = base_rect["y"] + base_rect["h"] * 0.5
    x = max(0.0, min(1.0 - w, cx - w * 0.5))
    y = max(0.0, min(1.0 - h, cy - h * 0.5))
    return {"x": x, "y": y, "w": w, "h": h}
