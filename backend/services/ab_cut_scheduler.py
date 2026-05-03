"""Editorial A/B reverse-cut scheduler for multi-speaker dialogue.

Today, two speakers that cannot fit simultaneously in a 9:16 crop get
either:

  * ``wide_master`` (letterboxed, both tiny), or
  * ``split_screen`` (two stacked crops, both fixed).

Humans cut A → B → A → B on each speaker turn, holding on the current
speaker for their beat plus a short tail so the listener's reaction is
caught. This module plans that A/B (or A/B/C for 3-seat panels)
schedule in terms of the existing ``ReframeSegment`` vocabulary.

Decision inputs:
  * two (or three) face slots present in the overlap region,
  * alternating speaker turns each ≥ ``ab_min_turn_sec``,
  * faces don't fit inside the same crop at ≥ 30 % of source width
    (i.e. ``multi_region_layout`` returned infeasible or wide-master),
  * content type is NOT ``multi_speaker_panel`` (panels prefer a
    fixed panoramic wide) UNLESS explicitly overridden for a 3-seat
    rotation.

Output: a list of :class:`AbCutSegment` that the reframe segmenter
splices into the overlap region as per-speaker tracking crops with
``SACCADE`` eases between them (see :mod:`camera_events`).

Cut-on-reaction: when the audio analyzer flags laughter / gasp / crowd
within ±0.4 s of a non-speaker face being off-screen, the scheduler
inserts a reaction beat pointing at that non-speaker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from backend.services.reframe_config import ReframeConfig, get_default_config

logger = logging.getLogger(__name__)


@dataclass
class SpeakerWindow:
    """One contiguous window where one slot is the active speaker."""

    start: float
    end: float
    slot_id: int


@dataclass
class ReactionEvent:
    """Audio-detected reaction (laughter / gasp / crowd) at time ``t``
    that should prompt a listener cut."""

    t: float
    kind: str    # "laughter" | "gasp" | "crowd"
    intensity: float = 1.0


@dataclass
class AbCutSegment:
    """One A/B/C sub-segment the reframe segmenter should render as a
    single-subject tracking crop for this slot."""

    start: float
    end: float
    slot_id: int
    # Saccade ease-ms INTO this segment from the previous one. 0 for
    # the first segment of the overlap.
    ease_ms: int = 150
    reason: str = ""


# ── Heuristics ────────────────────────────────────────────────────


def _merge_short_windows(
    windows: list[SpeakerWindow],
    min_sec: float,
) -> list[SpeakerWindow]:
    """Drop windows shorter than ``min_sec`` by merging them into their
    neighbor (prefer the longer one). Prevents A/B/A/B/A/B jitter on
    short back-channel "mhm" / "yeah"s.
    """
    if not windows:
        return []
    out: list[SpeakerWindow] = [windows[0]]
    for w in windows[1:]:
        if w.end - w.start >= min_sec:
            out.append(w)
            continue
        # Absorb into previous (extend end).
        out[-1] = SpeakerWindow(
            start=out[-1].start,
            end=w.end,
            slot_id=out[-1].slot_id,
        )
    return [w for w in out if w.end - w.start >= min_sec]


def _cap_cut_rate(
    segments: list[AbCutSegment],
    max_per_sec: float,
) -> list[AbCutSegment]:
    """Guarantee at most ``max_per_sec`` cuts / second by stretching
    the shortest segments until the rate falls below the threshold.
    """
    if not segments or max_per_sec <= 0:
        return segments
    min_dt = 1.0 / max_per_sec
    out = [segments[0]]
    for seg in segments[1:]:
        prev = out[-1]
        gap = seg.start - prev.start
        if gap < min_dt:
            # Push this segment's start (and end) forward.
            delta = min_dt - gap
            new_start = seg.start + delta
            if new_start >= seg.end:
                # The segment would collapse — drop it and extend prev.
                out[-1] = AbCutSegment(
                    start=prev.start,
                    end=seg.end,
                    slot_id=prev.slot_id,
                    ease_ms=prev.ease_ms,
                    reason=prev.reason + "+capped",
                )
                continue
            seg = AbCutSegment(
                start=new_start,
                end=seg.end,
                slot_id=seg.slot_id,
                ease_ms=seg.ease_ms,
                reason=seg.reason,
            )
        out.append(seg)
    return out


def _insert_reaction_cuts(
    segments: list[AbCutSegment],
    reactions: list[ReactionEvent],
    non_speaker_slots_at: callable,
    config: ReframeConfig,
) -> list[AbCutSegment]:
    """For each reaction event that lands inside a speaker window,
    steal the last ``ab_overlap_sec`` of the window and point it at the
    non-speaker (listener) slot.
    """
    if not reactions:
        return segments
    out: list[AbCutSegment] = []
    for seg in segments:
        fired = False
        for react in reactions:
            if react.t < seg.start or react.t > seg.end:
                continue
            candidates = non_speaker_slots_at(react.t) or []
            candidates = [s for s in candidates if s != seg.slot_id]
            if not candidates:
                continue
            listener = candidates[0]
            # Carve the tail of this segment into a reaction cut.
            reaction_start = max(seg.start,
                                 min(seg.end - 0.1, react.t + 0.05))
            if reaction_start <= seg.start:
                continue
            out.append(AbCutSegment(
                start=seg.start,
                end=reaction_start,
                slot_id=seg.slot_id,
                ease_ms=seg.ease_ms,
                reason=seg.reason,
            ))
            out.append(AbCutSegment(
                start=reaction_start,
                end=seg.end,
                slot_id=listener,
                ease_ms=config.saccade_ease_ms,
                reason=f"reaction:{react.kind}",
            ))
            fired = True
            break
        if not fired:
            out.append(seg)
    return out


# ── Public API ────────────────────────────────────────────────────


@dataclass
class AbScheduleResult:
    enabled: bool
    segments: list[AbCutSegment] = field(default_factory=list)
    fallback_reason: str = ""


def plan_ab_schedule(
    *,
    overlap_start: float,
    overlap_end: float,
    speaker_windows: list[SpeakerWindow],
    reactions: list[ReactionEvent] | None = None,
    non_speaker_slots_at: callable | None = None,
    config: Optional[ReframeConfig] = None,
    content_type: str = "",
    use_speaker_cut_engine: bool = False,
    speaker_positions: Optional[dict] = None,
    source_width: int = 1920,
) -> AbScheduleResult:
    """Plan an A/B (or A/B/C) cut schedule for an overlap region.

    Returns an :class:`AbScheduleResult`. When ``enabled == False`` the
    caller should fall back to the existing ``split_screen`` /
    ``wide_master`` decision; ``fallback_reason`` explains why.
    """
    config = config or get_default_config()
    reactions = reactions or []
    unique_slots = {w.slot_id for w in speaker_windows}
    if len(unique_slots) < 2:
        return AbScheduleResult(enabled=False, fallback_reason="one-speaker")

    # Constrain to the overlap window
    windowed = [
        SpeakerWindow(
            start=max(w.start, overlap_start),
            end=min(w.end, overlap_end),
            slot_id=w.slot_id,
        )
        for w in speaker_windows
        if w.end > overlap_start and w.start < overlap_end
    ]
    windowed = [w for w in windowed if w.end > w.start]
    if not windowed:
        return AbScheduleResult(enabled=False, fallback_reason="no-speaker-in-overlap")

    # Each speaker must have ≥ 1 turn of ab_min_turn_sec.
    long_enough = _merge_short_windows(windowed, config.ab_min_turn_sec)
    if len({w.slot_id for w in long_enough}) < 2:
        return AbScheduleResult(enabled=False, fallback_reason="turns-too-short")

    # Panel content: A/B/C rotation allowed if ≥ 3 distinct slots fire.
    if content_type == "multi_speaker_panel" and len(unique_slots) < 3:
        return AbScheduleResult(enabled=False, fallback_reason="panel-dyad-keep-split")

    # Phase 3: optional delegation to the speaker_cut_engine for cut
    # TIMING. When ``use_speaker_cut_engine=True``, the base segment
    # boundaries below are replaced by the engine's keyframe times;
    # reaction cuts and the rate cap are still layered on afterward.
    if use_speaker_cut_engine:
        try:
            from backend.services.speaker_cut_engine import (
                compute_cut_keyframes_from_speaker_engine,
            )
            kfs = compute_cut_keyframes_from_speaker_engine(
                overlap_start=overlap_start,
                overlap_end=overlap_end,
                speaker_windows=long_enough,
                speaker_positions=speaker_positions,
                source_width=source_width,
                config=config,
            )
            if kfs:
                segments = []
                n_kfs = len(kfs)
                for i, kf in enumerate(kfs):
                    seg_start = max(kf.time_sec, overlap_start)
                    seg_end = (
                        kfs[i + 1].time_sec if i + 1 < n_kfs else overlap_end
                    )
                    if seg_end <= seg_start:
                        continue
                    try:
                        sid = int(kf.speaker_id) if kf.speaker_id is not None else -1
                    except (TypeError, ValueError):
                        sid = -1
                    segments.append(AbCutSegment(
                        start=seg_start,
                        end=seg_end,
                        slot_id=sid,
                        ease_ms=0,  # always hard cut from this engine
                        reason="speaker-cut-engine",
                    ))
                # Reaction cuts + rate cap still apply on top.
                if non_speaker_slots_at is not None:
                    segments = _insert_reaction_cuts(
                        segments, reactions, non_speaker_slots_at, config,
                    )
                segments = _cap_cut_rate(segments, config.ab_max_cuts_per_sec)
                return AbScheduleResult(enabled=True, segments=segments)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "speaker_cut_engine delegation failed; falling back to "
                "legacy A/B scheduler: %s", exc,
            )

    segments = [
        AbCutSegment(
            start=w.start,
            end=w.end,
            slot_id=w.slot_id,
            ease_ms=(0 if i == 0 else config.saccade_ease_ms),
            reason="speaker-turn",
        )
        for i, w in enumerate(long_enough)
    ]

    # Add a small tail overlap so the next speaker's reaction is seen.
    overlap = config.ab_overlap_sec
    for i in range(len(segments) - 1):
        segments[i] = AbCutSegment(
            start=segments[i].start,
            end=min(segments[i].end + overlap, segments[i + 1].start + overlap * 0.5),
            slot_id=segments[i].slot_id,
            ease_ms=segments[i].ease_ms,
            reason=segments[i].reason,
        )

    # Reaction cuts
    if non_speaker_slots_at is not None:
        segments = _insert_reaction_cuts(
            segments, reactions, non_speaker_slots_at, config,
        )

    # Rate-limit
    segments = _cap_cut_rate(segments, config.ab_max_cuts_per_sec)

    return AbScheduleResult(enabled=True, segments=segments)
