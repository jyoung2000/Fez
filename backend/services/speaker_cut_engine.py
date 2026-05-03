"""Speaker-aware cut engine for SPEAKER_ALTERNATING shots.

Phase 3 of the 6-phase reframing overhaul. Given a list of speaker
turns plus each speaker's known x position in the source frame, this
module produces a list of CROP KEYFRAMES describing when the camera
should snap to which speaker.

Key idea
========

For multi-speaker dialogue shots, the L1/LP camera-path solver is the
wrong tool: humans don't smoothly pan between two seated panelists,
they HARD CUT on each speaker turn (anticipating the next speaker by
~250 ms, holding ~500 ms past the end of a turn for reaction). The
output of this engine is a *sparse* keyframe sequence — between any
two keyframes the crop is static. Smoothing/easing happens elsewhere
(or not at all; ``transition_type`` is always ``"hard_cut"``).

Rules implemented (see Phase 3 spec)
====================================

1. **Min hold:** never cut faster than ``config.ab_min_turn_sec``.
   Rapid interruptions collapse onto the dominant speaker (the one
   who has spoken the longest in the surrounding 2 s window).
2. **Cut anticipation:** cut to the next speaker 250 ms before they
   start. If the gap between speakers is shorter than 250 ms, place
   the cut at the midpoint of the gap.
3. **Reaction hold:** after a speaker finishes, stay on them for
   500 ms before cutting away. If the next speaker starts within
   500 ms, the hold is truncated to the gap.
4. **Transition type:** every cut is ``"hard_cut"``.
5. **Initial speaker:** start on whoever speaks first. If there is no
   speech for >1 s, start on the face closest to frame center.
6. **Overlapping speech:** when two speakers overlap for more than
   1 s, stay on whoever started first. When the overlap ends, cut to
   whichever speaker is still speaking.

The module is pure: deterministic, no I/O, no model calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from backend.services.reframe_config import ReframeConfig, get_default_config

logger = logging.getLogger(__name__)


# ── Public dataclasses ──────────────────────────────────────────────


@dataclass
class SpeakerTurn:
    """One contiguous speaking interval for a single speaker."""

    speaker_id: str
    start_sec: float
    end_sec: float


@dataclass
class CropKeyframe:
    """One sparse keyframe in the camera path produced by this engine.

    Between any two consecutive keyframes the crop is held static at
    the latter keyframe's position. ``transition_type`` is always
    ``"hard_cut"`` for this engine but kept as a field for future
    extension (e.g. saccade easing).
    """

    time_sec: float
    x_frac: float
    transition_type: str  # always "hard_cut" from this engine
    speaker_id: Optional[str]


# ── Constants ───────────────────────────────────────────────────────


CUT_ANTICIPATION_SEC = 0.25      # Rule 2.
REACTION_HOLD_SEC = 0.50         # Rule 3.
INITIAL_SILENCE_THRESHOLD_SEC = 1.0  # Rule 5.
OVERLAP_THRESHOLD_SEC = 1.0      # Rule 6.
DOMINANT_WINDOW_SEC = 2.0        # Rule 1 dominant-speaker window.

_HARD_CUT = "hard_cut"


# ── Helpers ─────────────────────────────────────────────────────────


def _clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _resolve_x(
    speaker_id: Optional[str],
    speaker_positions: Dict[str, float],
    fallback: float = 0.5,
) -> float:
    if speaker_id is None:
        return fallback
    pos = speaker_positions.get(speaker_id)
    if pos is None:
        return fallback
    return _clamp(float(pos), 0.0, 1.0)


def _normalize_turns(
    speaker_turns: Sequence[SpeakerTurn],
    shot_start_sec: float,
    shot_end_sec: float,
) -> List[SpeakerTurn]:
    """Clip turns to the shot window, drop empties, sort by start."""
    out: List[SpeakerTurn] = []
    for t in speaker_turns or []:
        s = max(float(t.start_sec), float(shot_start_sec))
        e = min(float(t.end_sec), float(shot_end_sec))
        if e <= s:
            continue
        out.append(SpeakerTurn(
            speaker_id=str(t.speaker_id),
            start_sec=s,
            end_sec=e,
        ))
    out.sort(key=lambda t: (t.start_sec, t.end_sec))
    return out


def _initial_speaker_id(
    turns: Sequence[SpeakerTurn],
    shot_start_sec: float,
) -> Optional[str]:
    """Return the speaker_id for the very first turn, or None when the
    shot starts with > INITIAL_SILENCE_THRESHOLD_SEC of silence."""
    if not turns:
        return None
    first = turns[0]
    if (first.start_sec - shot_start_sec) > INITIAL_SILENCE_THRESHOLD_SEC:
        return None
    return first.speaker_id


def _closest_to_center_speaker(
    speaker_positions: Dict[str, float],
) -> Optional[str]:
    if not speaker_positions:
        return None
    best_id: Optional[str] = None
    best_d = float("inf")
    for sid, x in speaker_positions.items():
        d = abs(float(x) - 0.5)
        if d < best_d:
            best_d = d
            best_id = sid
    return best_id


def _dominant_speaker_in_window(
    turns: Sequence[SpeakerTurn],
    center_t: float,
    window_sec: float = DOMINANT_WINDOW_SEC,
) -> Optional[str]:
    """Speaker with the most total speech inside ``[center_t-w/2,
    center_t+w/2]``. Falls back to None if no speech is in window."""
    half = window_sec * 0.5
    lo = center_t - half
    hi = center_t + half
    totals: Dict[str, float] = {}
    for t in turns:
        s = max(t.start_sec, lo)
        e = min(t.end_sec, hi)
        if e > s:
            totals[t.speaker_id] = totals.get(t.speaker_id, 0.0) + (e - s)
    if not totals:
        return None
    return max(totals.items(), key=lambda kv: kv[1])[0]


def _detect_overlap_segments(
    turns: Sequence[SpeakerTurn],
) -> List[Tuple[float, float, List[str]]]:
    """Return list of (start, end, [speaker_ids]) intervals where ≥ 2
    speakers overlap. Used by Rule 6 — when the overlap exceeds
    OVERLAP_THRESHOLD_SEC, stick with whoever started first."""
    if len(turns) < 2:
        return []
    # Use a sweep over start/end events.
    events: List[Tuple[float, int, str]] = []
    for t in turns:
        events.append((t.start_sec, 0, t.speaker_id))
        events.append((t.end_sec, 1, t.speaker_id))
    events.sort(key=lambda e: (e[0], e[1]))
    active: Dict[str, int] = {}
    last_t = events[0][0]
    out: List[Tuple[float, float, List[str]]] = []
    for ts, kind, sid in events:
        if ts > last_t and len(active) >= 2:
            out.append((last_t, ts, sorted(active.keys())))
        if kind == 0:
            active[sid] = active.get(sid, 0) + 1
        else:
            n = active.get(sid, 0) - 1
            if n <= 0:
                active.pop(sid, None)
            else:
                active[sid] = n
        last_t = ts
    return out


def _speaker_at_time(
    turns: Sequence[SpeakerTurn],
    t: float,
) -> Optional[str]:
    for tr in turns:
        if tr.start_sec <= t < tr.end_sec:
            return tr.speaker_id
    return None


# ── Core algorithm ─────────────────────────────────────────────────


def _build_target_sequence(
    turns: List[SpeakerTurn],
    shot_start_sec: float,
    shot_end_sec: float,
) -> List[Tuple[float, str]]:
    """Build the (cut_time, speaker_id) sequence BEFORE min-hold gating.

    Implements rules 2 (anticipation), 3 (reaction hold) and 6
    (overlap stickiness). Initial speaker selection (rule 5) and the
    min-hold gate (rule 1) are applied by the caller.
    """
    if not turns:
        return []

    # Pre-compute long overlap ranges (Rule 6).
    long_overlaps: List[Tuple[float, float, str]] = []
    for ov_start, ov_end, sids in _detect_overlap_segments(turns):
        if ov_end - ov_start <= OVERLAP_THRESHOLD_SEC:
            continue
        # Within a long overlap we stick with whoever started first.
        # The "first" speaker is the one whose turn START is the
        # earliest among the overlapping speakers.
        first_starter: Optional[str] = None
        first_start = float("inf")
        for sid in sids:
            for t in turns:
                if t.speaker_id != sid:
                    continue
                if t.start_sec < ov_end and t.end_sec > ov_start:
                    if t.start_sec < first_start:
                        first_start = t.start_sec
                        first_starter = sid
        if first_starter is not None:
            long_overlaps.append((ov_start, ov_end, first_starter))

    def _in_long_overlap(t: float) -> Optional[str]:
        for s, e, sid in long_overlaps:
            if s <= t < e:
                return sid
        return None

    # Build the "intended speaker" piecewise function from turns.
    # We walk through turns in order; for each turn we add a target
    # cut at ``start - 250 ms`` (or midpoint of the silence gap if
    # the gap is shorter than 250 ms). The reaction hold (Rule 3)
    # delays the cut by up to 500 ms after the previous speaker
    # finishes; if the next speaker comes within 500 ms the hold is
    # truncated to the gap.
    targets: List[Tuple[float, str]] = []
    prev_end: Optional[float] = None
    prev_sid: Optional[str] = None

    for turn in turns:
        sid = turn.speaker_id

        # Long-overlap stickiness.
        ov_sid = _in_long_overlap(turn.start_sec)
        if ov_sid is not None and ov_sid != sid:
            # Skip emitting a cut for this turn — we are stuck on the
            # earlier speaker. The cut at the END of the overlap is
            # synthesised below.
            prev_end = max(prev_end or 0.0, turn.end_sec)
            prev_sid = ov_sid
            continue

        if prev_sid is None:
            # First turn — schedule the initial speaker at shot start
            # (or just before). Caller will dedupe with rule 5.
            cut_t = max(shot_start_sec, turn.start_sec - CUT_ANTICIPATION_SEC)
            targets.append((cut_t, sid))
        elif sid == prev_sid:
            # Same speaker continues — no cut.
            pass
        else:
            gap = turn.start_sec - (prev_end if prev_end is not None else turn.start_sec)
            if gap >= CUT_ANTICIPATION_SEC + REACTION_HOLD_SEC:
                # Rule 3 reaction hold: stay on prev for 500ms after
                # they finish. Then anticipate by 250ms, which still
                # leaves ≥0 between hold-end and cut.
                cut_t = turn.start_sec - CUT_ANTICIPATION_SEC
            elif gap >= CUT_ANTICIPATION_SEC:
                # Anticipation fits but a full reaction hold doesn't.
                cut_t = turn.start_sec - CUT_ANTICIPATION_SEC
            elif gap > 0:
                # Short gap — cut at midpoint.
                cut_t = (prev_end + turn.start_sec) / 2.0
            else:
                # Overlap (gap <= 0) but the long-overlap rule didn't
                # fire, so cut right at the new turn's start.
                cut_t = turn.start_sec
            targets.append((cut_t, sid))

        prev_end = turn.end_sec if prev_end is None else max(prev_end, turn.end_sec)
        prev_sid = sid

    # Synthesize a cut at the end of each long overlap to whoever is
    # still speaking. This implements the second clause of Rule 6.
    for ov_start, ov_end, stuck_sid in long_overlaps:
        still = _speaker_at_time(turns, ov_end + 1e-6)
        if still is None or still == stuck_sid:
            continue
        targets.append((ov_end, still))

    # Sort and clip to shot.
    targets.sort(key=lambda x: x[0])
    targets = [
        (max(shot_start_sec, t), sid)
        for t, sid in targets
        if t < shot_end_sec
    ]
    return targets


def _enforce_min_hold(
    targets: List[Tuple[float, str]],
    turns: Sequence[SpeakerTurn],
    min_hold_sec: float,
) -> List[Tuple[float, str]]:
    """Drop cuts that would land < ``min_hold_sec`` after the previous
    one. Replaced with whichever speaker is dominant in the surrounding
    2s window (Rule 1)."""
    if not targets:
        return []
    out: List[Tuple[float, str]] = [targets[0]]
    for cut_t, sid in targets[1:]:
        prev_t, prev_sid = out[-1]
        if cut_t - prev_t < min_hold_sec - 1e-9:
            # Replace with dominant speaker over [prev_t, cut_t + 1s].
            dom = _dominant_speaker_in_window(
                turns, (prev_t + cut_t) / 2.0, DOMINANT_WINDOW_SEC,
            )
            if dom is not None and dom != prev_sid:
                # Promote previous slot to dominant if dominant differs.
                # Don't add a NEW cut though — the previous slot is the
                # one being held (per spec: "stay on dominant speaker").
                out[-1] = (prev_t, dom)
            # Else: drop the proposed cut and keep holding prev_sid.
            continue
        if sid == prev_sid:
            # Redundant cut to the same speaker — drop.
            continue
        out.append((cut_t, sid))
    return out


def plan_speaker_cuts(
    speaker_turns: List[SpeakerTurn],
    speaker_positions: Dict[str, float],
    shot_start_sec: float,
    shot_end_sec: float,
    source_width: int,
    config: ReframeConfig,
) -> List[CropKeyframe]:
    """Plan SPEAKER_ALTERNATING crop keyframes for one shot.

    Args:
        speaker_turns: Diarized turn intervals for the shot. Turns
            outside ``[shot_start_sec, shot_end_sec]`` are ignored.
        speaker_positions: ``speaker_id -> x_frac`` map giving each
            speaker's known x position as a 0-1 fraction of source
            frame width.
        shot_start_sec: Inclusive shot start.
        shot_end_sec: Exclusive shot end.
        source_width: Source frame width in pixels (kept for parity
            with the rest of the reframing surface — currently unused
            because keyframes are emitted in fractional space).
        config: ``ReframeConfig`` — only ``ab_min_turn_sec`` is read.

    Returns:
        Sparse ``CropKeyframe`` list. The first keyframe is at
        ``shot_start_sec``. ``speaker_id`` is ``None`` only for the
        center-fallback initial keyframe (rule 5).
    """
    if shot_end_sec <= shot_start_sec:
        return []
    cfg = config or get_default_config()
    min_hold = float(cfg.ab_min_turn_sec)

    turns = _normalize_turns(speaker_turns, shot_start_sec, shot_end_sec)
    targets = _build_target_sequence(turns, shot_start_sec, shot_end_sec)

    # Rule 5 — initial speaker.
    initial_sid = _initial_speaker_id(turns, shot_start_sec)
    initial_kf_speaker: Optional[str]
    if initial_sid is None:
        # No speech (or > 1 s of silence) → center-fallback face.
        initial_kf_speaker = _closest_to_center_speaker(speaker_positions)
        initial_x = _resolve_x(initial_kf_speaker, speaker_positions, 0.5)
        initial_kf = CropKeyframe(
            time_sec=float(shot_start_sec),
            x_frac=initial_x,
            transition_type=_HARD_CUT,
            # speaker_id=None for the center-fallback frame.
            speaker_id=None,
        )
    else:
        initial_x = _resolve_x(initial_sid, speaker_positions, 0.5)
        initial_kf = CropKeyframe(
            time_sec=float(shot_start_sec),
            x_frac=initial_x,
            transition_type=_HARD_CUT,
            speaker_id=initial_sid,
        )

    # The first scheduled target may equal the initial speaker; drop
    # it so we don't emit two same-position keyframes back-to-back.
    if targets and targets[0][1] == initial_sid:
        targets = targets[1:]
    # Push the initial keyframe in as the implicit anchor for min-hold
    # spacing — the engine can decide later if it wants to overwrite
    # it.
    seeded = [(float(shot_start_sec), initial_sid or "")] + targets
    spaced = _enforce_min_hold(seeded, turns, min_hold)

    # The first entry of ``spaced`` is the seed; the remainder are
    # genuine cuts. If the seed's speaker_id was rewritten by the
    # dominant-speaker rule, propagate that to the initial keyframe.
    keyframes: List[CropKeyframe] = []
    if spaced:
        seed_sid = spaced[0][1] or None
        if (
            seed_sid is not None
            and seed_sid != (initial_sid or "")
            and seed_sid in speaker_positions
        ):
            initial_kf = CropKeyframe(
                time_sec=initial_kf.time_sec,
                x_frac=_resolve_x(seed_sid, speaker_positions, 0.5),
                transition_type=_HARD_CUT,
                speaker_id=seed_sid,
            )
    keyframes.append(initial_kf)

    last_speaker = initial_kf.speaker_id
    for cut_t, sid in spaced[1:]:
        if not sid:
            continue
        if sid == last_speaker:
            continue
        x = _resolve_x(sid, speaker_positions, 0.5)
        keyframes.append(CropKeyframe(
            time_sec=float(cut_t),
            x_frac=x,
            transition_type=_HARD_CUT,
            speaker_id=sid,
        ))
        last_speaker = sid

    return keyframes


# ── Adapter for ab_cut_scheduler delegation ────────────────────────


def compute_cut_keyframes_from_speaker_engine(
    *,
    overlap_start: float,
    overlap_end: float,
    speaker_windows: Sequence,
    speaker_positions: Optional[Dict[str, float]] = None,
    source_width: int = 1920,
    config: Optional[ReframeConfig] = None,
) -> List[CropKeyframe]:
    """Adapter so :mod:`backend.services.ab_cut_scheduler` can delegate
    cut TIMING to this engine.

    Accepts ``ab_cut_scheduler.SpeakerWindow`` (with integer
    ``slot_id``) and converts to the engine's ``SpeakerTurn`` shape.
    When ``speaker_positions`` is missing, slots are placed at evenly-
    spaced x-fractions (deterministic fallback so the adapter still
    returns valid keyframes).
    """
    cfg = config or get_default_config()

    turns: List[SpeakerTurn] = []
    slot_ids: List[int] = []
    for w in speaker_windows or []:
        slot_id = int(getattr(w, "slot_id", -1))
        if slot_id < 0:
            continue
        if slot_id not in slot_ids:
            slot_ids.append(slot_id)
        turns.append(SpeakerTurn(
            speaker_id=str(slot_id),
            start_sec=float(getattr(w, "start", 0.0)),
            end_sec=float(getattr(w, "end", 0.0)),
        ))

    if speaker_positions is None:
        # Even-spaced fallback (deterministic). 6 slot ceiling per spec.
        capped = slot_ids[:6]
        n = max(len(capped), 1)
        if n == 1:
            positions = {str(capped[0]): 0.5} if capped else {}
        else:
            positions = {
                str(sid): (i + 0.5) / n for i, sid in enumerate(capped)
            }
    else:
        positions = {str(k): float(v) for k, v in speaker_positions.items()}

    return plan_speaker_cuts(
        speaker_turns=turns,
        speaker_positions=positions,
        shot_start_sec=float(overlap_start),
        shot_end_sec=float(overlap_end),
        source_width=int(source_width),
        config=cfg,
    )


# ── Reframe-segmenter integration helper ───────────────────────────


def keyframes_to_reframe_segments(
    keyframes: Sequence[CropKeyframe],
    *,
    shot_start_sec: float,
    shot_end_sec: float,
    source_width: int,
    speaker_id_to_slot: Optional[Dict[str, int]] = None,
    base_segment_template=None,
):
    """Convert sparse keyframes to ``ReframeSegment``-compatible dicts.

    Returns a list of ``dict`` rather than ``ReframeSegment`` so the
    caller can splat the dict into its own segment constructor (avoids
    a hard import cycle through ``reframe_segmenter``).

    Each emitted segment covers ``[kf.time_sec, next_kf.time_sec]`` and
    carries ``subject_x = x_frac * source_width``, ``layout="single"``,
    ``strategy="speaker_alternating"``.
    """
    if not keyframes:
        return []
    sorted_kfs = sorted(keyframes, key=lambda k: k.time_sec)
    out: List[dict] = []
    n = len(sorted_kfs)
    for i, kf in enumerate(sorted_kfs):
        seg_start = max(float(kf.time_sec), float(shot_start_sec))
        seg_end = (
            float(sorted_kfs[i + 1].time_sec)
            if i + 1 < n
            else float(shot_end_sec)
        )
        if seg_end <= seg_start:
            continue
        subject_x = _clamp(float(kf.x_frac), 0.0, 1.0) * float(source_width)
        slot = None
        if speaker_id_to_slot and kf.speaker_id is not None:
            slot = speaker_id_to_slot.get(kf.speaker_id)
        out.append({
            "start": seg_start,
            "end": seg_end,
            "subject_x": subject_x,
            "active_slot": slot,
            "layout": "single",
            "strategy": "speaker_alternating",
            "reason": "speaker_turn",
            "ease_in_ms": 0,  # hard cut
            "speaker_id": kf.speaker_id,
        })
    return out


__all__ = [
    "SpeakerTurn",
    "CropKeyframe",
    "plan_speaker_cuts",
    "compute_cut_keyframes_from_speaker_engine",
    "keyframes_to_reframe_segments",
]
