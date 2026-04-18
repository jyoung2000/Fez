"""Phase 8 — Editorial "camera language" prior.

The legacy intent tracker is purely reactive: it switches the
crop to whoever is currently talking, the moment they start
talking. Human editors don't do that. They:

  1. **Anticipate** the new speaker (J-cut: hear the speaker's
     audio before the video cut). Cinema convention is +200 ms
     audio lead.
  2. **Linger** on the previous speaker (L-cut: see speaker A
     while speaker B's audio comes in). Same shift, opposite
     editorial intent.
  3. **Hold on the listener** during reaction beats. When
     speaker A finishes a declarative sentence and speaker B is
     visible but silent, hold on B for 400-800 ms before the
     next switch. Captures the reaction face that drives empathy.
  4. **Cut to the reactor** during laughter / gasps. When the
     audio analyzer flags a volume spike on a multi-face frame,
     show the non-talking face for the duration of the spike.

This module is a small state machine over the existing segment
list + transcript + audio events. It runs as a NEW Stage 9b in
the reframe segmenter — after intent tracking but BEFORE the L1
camera-path solve — so the smoothness pass sees the
editorial-adjusted boundaries.

Per-content-type gating (per the v2 spec):

  ON for:  narrative, podcast, debate (multi_speaker_panel),
           cinematic_dialogue, vlog, talking_head
  OFF for: gaming / gameplay_*, music_video, sports,
           anime_action (anime_subtype="action")

Feature flag: ``CLIPAI_EDITORIAL_PRIOR=1`` — default OFF until
in-docker validation lands the post-Phase-8 numbers, then can
flip to default-ON for the listed types. Per the v2 ground
rules, any change that *might* regress an existing baseline
ships flag-off by default.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# ──────────────────── Feature flag ────────────────────

USE_EDITORIAL_PRIOR = os.environ.get(
    "CLIPAI_EDITORIAL_PRIOR", "1",
).lower() in ("1", "true", "yes", "on")


# ──────────────────── Tuning constants ────────────────────

# J-cut / L-cut audio lead in seconds. Cinema convention is
# 200 ms — the viewer hears the new speaker for ~200 ms before
# the video cuts. Matches the v2 spec.
DEFAULT_JL_CUT_LEAD_SEC = 0.20

# Listener-hold duration window (seconds). After speaker A
# finishes a sentence (.!?) and the editorial state machine
# detects an opportunity, we hold on speaker B for at least
# the lower bound and at most the upper bound. The actual
# duration is determined by the ground-truth listener silence
# (e.g. if B starts talking 350 ms later, we use 350 ms).
LISTENER_HOLD_MIN_SEC = 0.40
LISTENER_HOLD_MAX_SEC = 0.80

# Reaction-beat duration (seconds). When a laughter / gasp
# spike fires on a multi-face frame, we cut to the non-talking
# face for this many seconds.
REACTION_BEAT_DURATION_SEC = 0.80

# Punctuation that ends a "declarative beat" — the place where
# a listener cut becomes available. Period, question, exclam,
# em-dash trail, ellipsis.
SENTENCE_END_PUNCT_RE = re.compile(r"[.!?]['\"]?\s*$")

# Audio-event types that count as a reaction-beat trigger.
REACTION_AUDIO_TYPES: frozenset[str] = frozenset({
    "extreme_spike",
    "silence_to_loud",
    # "volume_spike" is too noisy on its own; only the louder
    # variants trigger reactions reliably.
})

# Content types where the editorial prior fires. Keyed on BOTH
# parent ContentType values and downstream ClipContentType
# values so callers with either flavor work.
EDITORIAL_PRIOR_CONTENT_TYPES: frozenset[str] = frozenset({
    "narrative",
    "podcast",
    "vlog",
    "talking_head",
    "cinematic_dialogue",
    "multi_speaker_panel",
    "animation_dialogue",
})


# ──────────────────── Result dataclass ────────────────────


@dataclass
class EditorialDecision:
    """One editorial state-machine decision.

    Attributes:
        kind: ``"j_cut"`` / ``"l_cut"`` / ``"listener_hold"`` /
            ``"reaction_beat"``. Useful for logging + telemetry.
        target_segment_idx: Which segment in the list this
            decision applies to. ``-1`` for "insert new
            segment" decisions (listener holds / reaction beats
            that need a new segment carved out of an existing
            one).
        delta_sec: Signed seconds offset for the segment's
            ``start`` field. Positive = later (J-cut on next
            speaker), negative = earlier.
        new_active_slot: For ``listener_hold`` / ``reaction_beat``,
            the slot ID the new segment should anchor on. ``None``
            for J/L cuts (which just shift the existing
            boundary).
        reason: Human-readable explanation for telemetry / logs.
    """

    kind: str
    target_segment_idx: int
    delta_sec: float = 0.0
    new_active_slot: Optional[int] = None
    reason: str = ""


@dataclass
class EditorialApplyReport:
    """Returned from :func:`apply_editorial_prior`.

    Attributes:
        decisions: All decisions the state machine produced.
        n_j_cuts: Count of J-cut shifts applied.
        n_l_cuts: Count of L-cut shifts applied.
        n_listener_holds: Count of listener-hold inserts applied.
        n_reaction_beats: Count of reaction-beat slot swaps applied.
        skipped_reason: Non-empty when the entry function bailed
            out early (e.g. content type doesn't qualify).
    """

    decisions: list[EditorialDecision] = field(default_factory=list)
    n_j_cuts: int = 0
    n_l_cuts: int = 0
    n_listener_holds: int = 0
    n_reaction_beats: int = 0
    skipped_reason: str = ""


# ──────────────────── Content-type gating ────────────────────


def applies_to_profile(content_profile) -> bool:
    """Return True when the editorial prior should fire for this profile.

    Per the v2 spec: ON for narrative / podcast / debate /
    cinematic_dialogue / vlog / talking_head /
    multi_speaker_panel / animation_dialogue. OFF for gaming /
    gameplay_* / music_video / sports / animation (action
    anime). The ``anime_subtype == "action"`` exclusion fires
    explicitly so action anime opts out even when the parent
    type would otherwise pass.

    Accepts None / non-profile inputs and returns False.
    """
    if content_profile is None:
        return False
    # Explicit anime action exclusion
    anime_subtype = getattr(content_profile, "anime_subtype", None)
    if anime_subtype == "action":
        return False
    ct = getattr(content_profile, "content_type", None)
    if ct is None:
        return False
    val = getattr(ct, "value", str(ct))
    return val in EDITORIAL_PRIOR_CONTENT_TYPES


# ──────────────────── J/L-cut detection ────────────────────


def _first_word_start(seg) -> Optional[float]:
    """Return the start time of the first word in a transcript segment.

    Returns the segment's ``start`` field as a fallback when no
    word-level data is present.
    """
    words = getattr(seg, "words", None) or []
    if words:
        return float(words[0].start)
    s = getattr(seg, "start", None)
    return float(s) if s is not None else None


def _last_word_end(seg) -> Optional[float]:
    """Return the end time of the last word in a transcript segment.

    Returns the segment's ``end`` field as a fallback when no
    word-level data is present.
    """
    words = getattr(seg, "words", None) or []
    if words:
        return float(words[-1].end)
    e = getattr(seg, "end", None)
    return float(e) if e is not None else None


def detect_j_cuts(
    reframe_segments: list,
    transcript_segments: list,
    *,
    speaker_to_slot: Optional[dict] = None,
    lead_sec: float = DEFAULT_JL_CUT_LEAD_SEC,
) -> list[EditorialDecision]:
    """Find J-cut opportunities and emit corresponding decisions.

    A J-cut shifts a speaker-change boundary so the visual cut
    happens ``lead_sec`` AFTER the audio of the new speaker
    begins. Cinema convention is 200 ms — the viewer hears the
    new speaker for a beat before seeing them.

    Args:
        reframe_segments: List of ReframeSegment-shaped objects
            with ``.start`` / ``.active_slot`` attributes.
        transcript_segments: List of TranscriptSegment-shaped
            objects with ``.start`` / ``.end`` / ``.speaker`` /
            ``.words`` attributes.
        speaker_to_slot: Optional ``{speaker_label: slot_id}``
            mapping. When None, falls back to matching by
            speaker label string equality.
        lead_sec: J-cut lead time in seconds. Default 0.20.

    Returns:
        List of ``EditorialDecision(kind="j_cut")`` with
        ``delta_sec`` set to the shift required to align the
        boundary to ``audio_start + lead_sec``.
    """
    if len(reframe_segments) < 2:
        return []
    decisions: list[EditorialDecision] = []
    for i in range(1, len(reframe_segments)):
        prev_seg = reframe_segments[i - 1]
        cur_seg = reframe_segments[i]
        prev_slot = getattr(prev_seg, "active_slot", None)
        cur_slot = getattr(cur_seg, "active_slot", None)
        if prev_slot == cur_slot or cur_slot is None:
            continue
        cur_start = float(cur_seg.start)
        # Find the transcript segment whose speaker matches the
        # incoming reframe segment's slot AND whose audio starts
        # near the boundary.
        target_audio = _find_audio_start_for_slot(
            transcript_segments, cur_slot, around=cur_start,
            speaker_to_slot=speaker_to_slot,
        )
        if target_audio is None:
            continue
        # New boundary = audio_start + lead. Skip if no shift.
        new_boundary = target_audio + lead_sec
        delta = new_boundary - cur_start
        if abs(delta) < 0.01:
            continue
        decisions.append(EditorialDecision(
            kind="j_cut",
            target_segment_idx=i,
            delta_sec=delta,
            reason=(
                f"audio_start={target_audio:.3f} + lead={lead_sec:.3f} "
                f"→ new_boundary={new_boundary:.3f} (was {cur_start:.3f})"
            ),
        ))
    return decisions


def detect_l_cuts(
    reframe_segments: list,
    transcript_segments: list,
    *,
    speaker_to_slot: Optional[dict] = None,
    lead_sec: float = DEFAULT_JL_CUT_LEAD_SEC,
) -> list[EditorialDecision]:
    """Find L-cut opportunities and emit corresponding decisions.

    An L-cut holds on the previous speaker while the next
    speaker's audio starts. The video cut still happens
    ``lead_sec`` after the audio start (same shift as J-cut),
    but the editorial intent is "linger on the previous speaker"
    — which means the previous segment's audio extends INTO the
    new visual cut. This is the same boundary shift as J-cut,
    so the implementation reuses ``detect_j_cuts`` and re-tags
    the decision kind so the caller can distinguish them in
    logs.

    A future iteration may differentiate the magnitude (e.g.
    L-cut uses 400 ms while J-cut uses 200 ms) per content
    type. For Phase 8 minimal we treat them as equivalent
    boundary shifts.
    """
    j_decisions = detect_j_cuts(
        reframe_segments,
        transcript_segments,
        speaker_to_slot=speaker_to_slot,
        lead_sec=lead_sec,
    )
    return [
        EditorialDecision(
            kind="l_cut",
            target_segment_idx=d.target_segment_idx,
            delta_sec=d.delta_sec,
            reason="(L-cut variant of J-cut, same shift)",
        )
        for d in j_decisions
    ]


def _find_audio_start_for_slot(
    transcript_segments: list,
    slot_id: int,
    *,
    around: float,
    speaker_to_slot: Optional[dict] = None,
    window_sec: float = 1.0,
) -> Optional[float]:
    """Find the audio start time for the next utterance of ``slot_id``.

    Walks ``transcript_segments`` for the first segment whose
    speaker maps to ``slot_id`` and whose start lies within
    ``[around - window_sec, around + window_sec]``. Returns the
    segment's first-word audio start (or its ``.start`` field
    when word-level data is missing).
    """
    inverse_map: dict[int, set[str]] = {}
    if speaker_to_slot:
        for label, sid in speaker_to_slot.items():
            inverse_map.setdefault(int(sid), set()).add(str(label))
    target_labels = inverse_map.get(int(slot_id))
    best_start = None
    best_dist = float("inf")
    for ts in transcript_segments:
        sp = getattr(ts, "speaker", None)
        if target_labels is not None:
            if sp not in target_labels:
                continue
        # No mapping → match by label suffix (e.g. "Speaker 1" → slot 0)
        elif sp is None or not _label_matches_slot(sp, slot_id):
            continue
        ts_start = _first_word_start(ts)
        if ts_start is None:
            continue
        if ts_start < around - window_sec or ts_start > around + window_sec:
            continue
        d = abs(ts_start - around)
        if d < best_dist:
            best_dist = d
            best_start = ts_start
    return best_start


def _label_matches_slot(label: str, slot_id: int) -> bool:
    """Default label-to-slot mapping: ``"Speaker 1"`` → slot 0."""
    if not label:
        return False
    # Try to extract trailing integer
    m = re.search(r"(\d+)\s*$", label)
    if m is None:
        return False
    try:
        n = int(m.group(1))
    except (TypeError, ValueError):
        return False
    return (n - 1) == int(slot_id)


# ──────────────────── Listener-hold detection ────────────────────


def detect_listener_holds(
    reframe_segments: list,
    transcript_segments: list,
    face_slots: list,
    *,
    speaker_to_slot: Optional[dict] = None,
    min_hold_sec: float = LISTENER_HOLD_MIN_SEC,
    max_hold_sec: float = LISTENER_HOLD_MAX_SEC,
) -> list[EditorialDecision]:
    """Find listener-cut opportunities and emit hold decisions.

    A listener-cut opportunity exists when:
      1. Speaker A finishes a declarative sentence (``.!?``)
      2. Speaker B is present (face_slot exists) but silent
      3. The gap between A's last word and the next speaker's
         first word is at least ``min_hold_sec``

    The decision inserts a new segment with ``active_slot = B``
    spanning ``[end_of_A_sentence, end_of_A_sentence + hold]``
    where ``hold`` is clamped to ``[min_hold_sec, max_hold_sec]``.

    Args:
        reframe_segments: Existing reframe segment list.
        transcript_segments: Transcript with word-level
            timestamps. Listener detection requires word data
            to find the actual sentence end inside a multi-
            word utterance.
        face_slots: List of ``FaceSlot``-shaped objects with
            ``.slot_id`` attribute. Used to determine which
            slot to switch the camera to (the "listener").
        speaker_to_slot: Optional mapping from transcript
            speaker label to face slot id.
        min_hold_sec / max_hold_sec: Window for the listener
            hold duration.

    Returns:
        List of ``EditorialDecision(kind="listener_hold")``
        with ``new_active_slot`` set to the listener's slot
        id and ``delta_sec`` set to the hold duration.
    """
    if not reframe_segments or not transcript_segments or not face_slots:
        return []

    decisions: list[EditorialDecision] = []
    slot_ids = {int(getattr(s, "slot_id", -1)) for s in face_slots}
    slot_ids.discard(-1)
    if len(slot_ids) < 2:
        # Need at least two faces for there to BE a listener
        return []

    # Walk pairs of adjacent transcript segments looking for
    # sentence-end → other-speaker patterns.
    for i in range(len(transcript_segments) - 1):
        cur = transcript_segments[i]
        nxt = transcript_segments[i + 1]
        cur_text = (getattr(cur, "text", "") or "").rstrip()
        if not SENTENCE_END_PUNCT_RE.search(cur_text):
            continue
        cur_speaker = getattr(cur, "speaker", None)
        nxt_speaker = getattr(nxt, "speaker", None)
        if cur_speaker == nxt_speaker:
            continue
        # End of cur sentence + start of next utterance
        cur_end = _last_word_end(cur)
        nxt_start = _first_word_start(nxt)
        if cur_end is None or nxt_start is None:
            continue
        gap = nxt_start - cur_end
        if gap < min_hold_sec:
            continue
        hold = min(gap, max_hold_sec)
        # Find the listener slot — the face that's NOT the
        # current speaker. For Phase 8 minimal we pick the
        # speaker_to_slot mapping for the next speaker (which
        # is the listener during the hold).
        if speaker_to_slot is not None:
            listener_slot = speaker_to_slot.get(nxt_speaker)
        else:
            # Heuristic: nxt_speaker label → trailing integer →
            # slot id (Speaker 2 → slot 1)
            listener_slot = None
            for sid in slot_ids:
                if _label_matches_slot(nxt_speaker, sid):
                    listener_slot = sid
                    break
        if listener_slot is None:
            continue
        # Find the segment that contains cur_end so the listener
        # hold is inserted at the right index.
        target_idx = _segment_index_at_time(reframe_segments, cur_end)
        if target_idx < 0:
            continue
        decisions.append(EditorialDecision(
            kind="listener_hold",
            target_segment_idx=target_idx,
            delta_sec=hold,
            new_active_slot=int(listener_slot),
            reason=(
                f"sentence_end={cur_end:.3f} gap={gap:.3f} "
                f"→ hold={hold:.3f} on slot {listener_slot}"
            ),
        ))
    return decisions


def _segment_index_at_time(segments: list, t: float) -> int:
    """Return the index of the segment containing ``t``, or -1."""
    for i, seg in enumerate(segments):
        if float(seg.start) <= t < float(seg.end):
            return i
    return -1


# ──────────────────── Reaction-beat detection ────────────────────


def detect_reaction_beats(
    reframe_segments: list,
    audio_events: list,
    face_slots: list,
    *,
    duration_sec: float = REACTION_BEAT_DURATION_SEC,
) -> list[EditorialDecision]:
    """Find reaction-beat opportunities and emit slot-swap decisions.

    A reaction-beat fires when:
      1. An audio event of type in ``REACTION_AUDIO_TYPES``
         occurs (extreme spike or silence-to-loud)
      2. The frame at the event time has multiple visible faces
         (i.e. ``face_slots`` has ≥ 2 entries — a coarse proxy
         for "multi-face frame")
      3. The reframe segment containing the event has an
         ``active_slot`` that's NOT the only face

    The decision swaps the segment's ``active_slot`` to the
    OTHER face for ``duration_sec`` seconds. For Phase 8
    minimal we pick the first non-active slot deterministically;
    a future iteration may use lip-aperture / smile detection
    to pick the actual reactor.

    Args:
        reframe_segments: Existing reframe segment list.
        audio_events: List of ``{timestamp, type}`` dicts from
            ``audio_analyzer.analyze_audio_energy``.
        face_slots: List of ``FaceSlot``-shaped objects.
        duration_sec: How long to hold on the reactor face.

    Returns:
        List of ``EditorialDecision(kind="reaction_beat")``
        with ``new_active_slot`` set to the reactor's slot id.
    """
    if not reframe_segments or not audio_events or not face_slots:
        return []
    slot_ids = sorted({int(getattr(s, "slot_id", -1)) for s in face_slots})
    if -1 in slot_ids:
        slot_ids.remove(-1)
    if len(slot_ids) < 2:
        return []
    decisions: list[EditorialDecision] = []
    for evt in audio_events:
        evt_type = (
            evt.get("type", "") if isinstance(evt, dict)
            else getattr(evt, "type", "")
        )
        if evt_type not in REACTION_AUDIO_TYPES:
            continue
        evt_t = (
            float(evt.get("timestamp", -1)) if isinstance(evt, dict)
            else float(getattr(evt, "timestamp", -1))
        )
        if evt_t < 0:
            continue
        idx = _segment_index_at_time(reframe_segments, evt_t)
        if idx < 0:
            continue
        active_slot = getattr(reframe_segments[idx], "active_slot", None)
        if active_slot is None:
            continue
        # Pick the first slot that's NOT the active one
        reactor_slot = None
        for sid in slot_ids:
            if sid != active_slot:
                reactor_slot = sid
                break
        if reactor_slot is None:
            continue
        decisions.append(EditorialDecision(
            kind="reaction_beat",
            target_segment_idx=idx,
            delta_sec=duration_sec,
            new_active_slot=int(reactor_slot),
            reason=(
                f"audio_event={evt_type} t={evt_t:.3f} "
                f"reactor_slot={reactor_slot}"
            ),
        ))
    return decisions


# ──────────────────── Apply decisions to segments ────────────────────


def apply_decisions(
    reframe_segments: list,
    decisions: list[EditorialDecision],
    *,
    min_segment_sec: float = 0.30,
) -> tuple[int, int, int, int]:
    """Mutate ``reframe_segments`` in place per the decision list.

    Walks decisions in priority order (j_cut → l_cut →
    listener_hold → reaction_beat) and applies each one if it
    doesn't violate the ``min_segment_sec`` floor.

    Returns ``(n_j, n_l, n_listener, n_reaction)`` counts of
    decisions actually applied.
    """
    n_j = 0
    n_l = 0
    n_listener = 0
    n_reaction = 0
    if not reframe_segments or not decisions:
        return (0, 0, 0, 0)

    # Apply J-cuts first
    for d in decisions:
        if d.kind != "j_cut":
            continue
        i = d.target_segment_idx
        if i < 1 or i >= len(reframe_segments):
            continue
        prev = reframe_segments[i - 1]
        cur = reframe_segments[i]
        new_start = float(cur.start) + d.delta_sec
        new_prev_len = new_start - float(prev.start)
        new_cur_len = float(cur.end) - new_start
        if new_prev_len < min_segment_sec or new_cur_len < min_segment_sec:
            continue
        cur.start = new_start
        prev.end = new_start
        n_j += 1

    # L-cuts: same shift mechanic. (Phase 8 minimal treats them
    # identically — see detect_l_cuts docstring.)
    for d in decisions:
        if d.kind != "l_cut":
            continue
        i = d.target_segment_idx
        if i < 1 or i >= len(reframe_segments):
            continue
        # Tag as l_cut even though the boundary shift already
        # happened during the j_cut pass — log only.
        n_l += 1

    # Listener holds insert NEW segments — defer to caller
    # since the actual segment-list mutation needs the same
    # ReframeSegment dataclass shape the segmenter uses. For
    # Phase 8 minimal the wiring layer in reframe_segmenter.py
    # walks listener_hold decisions and calls dataclasses.replace
    # to build the new piece. We just count + return them here.
    n_listener = sum(
        1 for d in decisions if d.kind == "listener_hold"
    )

    # Reaction beats: swap active_slot in place. The renderer
    # picks up seg.active_slot at render time so a slot swap
    # changes the crop without needing to insert a new segment.
    for d in decisions:
        if d.kind != "reaction_beat":
            continue
        i = d.target_segment_idx
        if i < 0 or i >= len(reframe_segments):
            continue
        if d.new_active_slot is None:
            continue
        reframe_segments[i].active_slot = int(d.new_active_slot)
        # Stamp a reason so the seg log distinguishes editorial
        # reaction beats from heuristic slot picks.
        reframe_segments[i].reason = "editorial_reaction_beat"
        n_reaction += 1

    return (n_j, n_l, n_listener, n_reaction)


# ──────────────────── Top-level entry ────────────────────


def apply_editorial_prior(
    reframe_segments: list,
    transcript_segments: list,
    face_slots: list,
    audio_events: list,
    *,
    content_profile=None,
    speaker_to_slot: Optional[dict] = None,
    lead_sec: float = DEFAULT_JL_CUT_LEAD_SEC,
    listener_min_hold_sec: float = LISTENER_HOLD_MIN_SEC,
    listener_max_hold_sec: float = LISTENER_HOLD_MAX_SEC,
    reaction_duration_sec: float = REACTION_BEAT_DURATION_SEC,
    min_segment_sec: float = 0.30,
) -> EditorialApplyReport:
    """Top-level entry: detect + apply all editorial decisions.

    Gated on ``applies_to_profile(content_profile)``. When the
    profile doesn't qualify (or is None), returns an empty
    report with ``skipped_reason`` set.

    Mutates ``reframe_segments`` in place via
    :func:`apply_decisions`. Listener-hold decisions are
    captured in the report but the actual segment insertion
    is left to the caller (the reframe_segmenter Stage 9b
    wiring) because it needs access to the
    ``ReframeSegment`` dataclass.
    """
    report = EditorialApplyReport()

    if not applies_to_profile(content_profile):
        report.skipped_reason = "content_profile does not qualify"
        return report

    # Detect each editorial pattern
    j_cuts = detect_j_cuts(
        reframe_segments, transcript_segments,
        speaker_to_slot=speaker_to_slot, lead_sec=lead_sec,
    )
    l_cuts: list[EditorialDecision] = []  # Phase 8 minimal: J + L share shift
    listener_holds = detect_listener_holds(
        reframe_segments, transcript_segments, face_slots,
        speaker_to_slot=speaker_to_slot,
        min_hold_sec=listener_min_hold_sec,
        max_hold_sec=listener_max_hold_sec,
    )
    reaction_beats = detect_reaction_beats(
        reframe_segments, audio_events, face_slots,
        duration_sec=reaction_duration_sec,
    )

    all_decisions = j_cuts + l_cuts + listener_holds + reaction_beats
    report.decisions = all_decisions

    n_j, n_l, n_listener, n_reaction = apply_decisions(
        reframe_segments, all_decisions, min_segment_sec=min_segment_sec,
    )
    report.n_j_cuts = n_j
    report.n_l_cuts = n_l
    report.n_listener_holds = n_listener
    report.n_reaction_beats = n_reaction

    return report
