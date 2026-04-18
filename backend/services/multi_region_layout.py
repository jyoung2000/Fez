"""Phase 3 — multi-region layout decision helper.

The Phase 3 spec asks for a function that takes per-frame required +
optional bboxes and decides whether the LP can fit them all into one
crop (``"fit"``) or whether the segment needs to fall back to
SPLIT_SCREEN / WIDE_MASTER. This module is the single source of truth
for that decision; the reframe segmenter calls it from a new
content-aware Stage 3.5 sweep behind ``CLIPAI_MULTI_REGION_LP=1``.

Two public entry points:

  - ``decide_multi_region_layout`` — runs the LP and maps its output
    onto a small set of layout decisions (``fit`` / ``fit_with_pad``
    / ``split`` / ``wide``) plus the solved camera path when feasible.

  - ``promote_required_regions_for_segment`` — implements the
    "required vs optional" rule from the spec: a face is required
    when it's the active speaker and has spoken in the last N seconds;
    it's optional (lower weight) otherwise.

The infeasibility-ratio thresholds live here so Phase 3 has one place
to tune. Phase-specific tuning (e.g. podcast wants split, narrative
wants wide) is encoded via ``per_content_fallback_preference``.

This module is **numpy-free** and depends only on
``backend.services._autoflip_lp.solve_multi_region_camera_path`` plus
stdlib. The LP call lazily imports scipy when actually invoked, so a
sandbox without scipy can still import this module and exercise the
pure-Python helpers.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────── Feature flag ───────────────────

# Default ON as of Week 1 flag audit (see
# docs/reframing_autoflip_parity.md "Week 1 flag audit"). The
# synthetic 3speaker_panel fixture passed all four safety metrics
# with this flag enabled, and Stage 10a now runs the multi-region
# LP fit-check on every panel-routed clip. Set
# CLIPAI_MULTI_REGION_LP=0 to disable if a real-content regression
# appears — isolated OFF-vs-ON comparison data is archived at
# /tmp/mrlp_off.json and /tmp/mrlp_on.json from the Week 1 run.
USE_MULTI_REGION_LP = os.environ.get(
    "CLIPAI_MULTI_REGION_LP", "1",
).lower() in ("1", "true", "yes", "on")


# ─────────────────── Tuning constants ───────────────────

# Frame-fraction thresholds that route the layout decision based on
# how many frames the LP can't fit. The LP hard-fails at the first
# infeasible frame (``status="infeasible"``), so these thresholds
# operate on the *fraction* of frames that would have been infeasible
# if we walked the per-frame bounds directly. They live here so Phase
# 4-8 can tune them on a per-content-type basis without editing the
# LP itself.
INFEASIBLE_THRESHOLD_FIT = 0.05      # ≤ 5% → "fit_with_pad"
INFEASIBLE_THRESHOLD_SPLIT = 0.50    # ≤ 50% → "split"
# > 50% → "wide" (or per-content fallback preference)

# Required vs optional promotion (spec: "promote a region from
# optional → required when the active-speaker confidence > 0.7 AND
# the person has spoken in the last 2s. Demote on silence > 3s").
REQUIRED_PROMOTION_CONFIDENCE = 0.7
REQUIRED_PROMOTION_RECENCY_SEC = 2.0
REQUIRED_DEMOTION_SILENCE_SEC = 3.0


# ─────────────────── Per-frame bounds calculator ───────────────────


def per_frame_bounds_from_required(
    required_per_frame: list,
    crop_half_width_px: float,
    source_width_px: float,
) -> tuple[list[float], list[float], list[int]]:
    """Compute per-frame ``[lo, hi]`` and infeasible frame indices.

    For each frame, the camera center ``cam[t]`` must satisfy:

        cam[t] >= max_required_right - half_crop  (right edges in-frame)
        cam[t] <= min_required_left  + half_crop  (left edges in-frame)

    and stay inside the source frame:

        half_crop <= cam[t] <= source_width - half_crop

    Returns ``(lo, hi, infeasible_indices)``. When ``lo[t] > hi[t]``
    the frame is infeasible and the bounds are clamped to a degenerate
    midpoint so the LP still produces something the caller can ignore.

    This function is the geometric heart of the multi-region solver.
    It lives in ``multi_region_layout`` (not ``_autoflip_lp``) so a
    sandbox without numpy can still import it for unit tests.
    """
    lo: list[float] = []
    hi: list[float] = []
    infeasible: list[int] = []
    src_lo = float(crop_half_width_px)
    src_hi = float(source_width_px - crop_half_width_px)
    for t, regions in enumerate(required_per_frame):
        cur_lo = src_lo
        cur_hi = src_hi
        for left, right in regions:
            cur_lo = max(cur_lo, float(right) - crop_half_width_px)
            cur_hi = min(cur_hi, float(left) + crop_half_width_px)
        if cur_lo > cur_hi + 1e-6:
            infeasible.append(t)
            mid = 0.5 * (cur_lo + cur_hi)
            lo.append(mid)
            hi.append(mid)
        else:
            lo.append(cur_lo)
            hi.append(cur_hi)
    return lo, hi, infeasible


# ─────────────────── Result dataclass ───────────────────


@dataclass
class MultiRegionLayoutDecision:
    """Output of :func:`decide_multi_region_layout`.

    Attributes:
        decision: ``"fit"`` (LP fully feasible — use the camera path),
            ``"fit_with_pad"`` (a small fraction of frames are
            infeasible — use the LP path with the affected frames
            clamped to the nearest feasible center), ``"split"``
            (route to SPLIT_SCREEN), ``"wide"`` (route to WIDE_MASTER /
            blur fill / per-content fallback), or ``"lp_failed"``
            (HiGHS error — caller should keep the existing single-
            subject behavior).
        camera_path: solved camera centers in pixels, length matches
            the input frame count. Empty when ``decision`` is ``split``
            / ``wide`` / ``lp_failed``.
        infeasible_frames: indices of frames where the required regions
            can't all fit. Empty when fully feasible.
        infeasibility_ratio: fraction of frames that were infeasible.
        n_required: total required-region count summed across frames.
        n_optional: total optional-region count summed across frames.
        lp_message: passthrough of the scipy linprog status message.
        solve_ms: LP solve wall-clock time in milliseconds.
        reason: short string explaining the decision (for logs).
    """

    decision: str
    camera_path: list[float] = field(default_factory=list)
    infeasible_frames: list[int] = field(default_factory=list)
    infeasibility_ratio: float = 0.0
    n_frames: int = 0
    n_required: int = 0
    n_optional: int = 0
    lp_message: str = ""
    solve_ms: float = 0.0
    reason: str = ""


# Per-content-type fallback when the LP says "doesn't fit". Mirrors
# the existing CONTENT_TYPE_CONFIG fallback_preference values but
# keyed on the values that classify_clip emits, not raw ContentType.
PER_CONTENT_FALLBACK = {
    # Talking-head modes prefer SPLIT_SCREEN — the natural framing
    # for two participants who don't fit in one crop.
    "talking_head": "split",
    "multi_speaker_panel": "split",
    "cinematic_dialogue": "split",
    # Narrative honors the DP's composition — fall to wide_master /
    # blur fill rather than chopping the frame.
    "generic": "wide",
    # Animation paths follow the cinematic-dialogue rule.
    "animation_dialogue": "split",
    "animation": "wide",
    # Music video / sports / gaming favor wide fills since their
    # tracking signal is motion, not speakers.
    "music_video": "wide",
    "gameplay": "wide",
    "gameplay_moba": "wide",
    "gameplay_tps": "wide",
    "gameplay_racing": "wide",
    "stream": "wide",
    # Sports: subject is motion (ball / car / player body) — wide fill
    # keeps the action in frame rather than split-screening players.
    "sports": "wide",
    "sports_basketball": "wide",
    "sports_racing": "wide",
    # Vlog / narrative: single subject, favor split over letterbox if
    # two people enter frame (matches cinematic_dialogue treatment).
    "vlog": "split",
    "narrative": "split",
}


def fallback_for_content(content_type: Optional[str]) -> str:
    """Return ``"split"`` or ``"wide"`` for a given clip content type.

    Defaults to ``"split"`` when the type is unknown — splitting is
    the safer fallback for face-driven content, which is the only
    case the multi-region LP fires on.
    """
    if content_type is None:
        return "split"
    val = getattr(content_type, "value", str(content_type))
    return PER_CONTENT_FALLBACK.get(val, "split")


# ─────────────────── Layout decision ───────────────────


def decide_multi_region_layout(
    required_per_frame: list,
    optional_per_frame: Optional[list] = None,
    *,
    crop_width_px: float,
    source_width_px: float,
    content_type: Optional[str] = None,
    lam_data: float = 1.0,
    lam_velocity: float = 20.0,
    lam_accel: float = 100.0,
    lam_jerk: float = 100.0,
    lam_soft: float = 5.0,
    time_limit_sec: float = 10.0,
) -> MultiRegionLayoutDecision:
    """Run the multi-region LP and pick a layout decision.

    Decision tree:

      1. Run :func:`solve_multi_region_camera_path` against the inputs.
      2. ``status == "feasible"`` → ``"fit"`` (use the camera path).
      3. ``status == "infeasible"``:
         - ``infeasibility_ratio <= INFEASIBLE_THRESHOLD_FIT`` →
           ``"fit_with_pad"`` (Phase 4+ work; for now we route this
           to the per-content fallback like a regular split).
         - ``infeasibility_ratio <= INFEASIBLE_THRESHOLD_SPLIT`` →
           ``"split"`` for talking-head / panel / cinematic-dialogue,
           ``"wide"`` for narrative / music / gaming.
         - ``infeasibility_ratio > INFEASIBLE_THRESHOLD_SPLIT`` →
           ``"wide"`` regardless of content (too many frames out).
      4. ``status == "lp_failed"`` → ``"lp_failed"`` (caller keeps
         the existing single-subject behavior).

    The per-content fallback in step 3 lets the same LP drive both
    podcast-style splits and narrative-style wide masters from one
    code path, replacing the heuristic Stage 3 logic that currently
    decides via face-count + speaker-overlap rules.
    """
    from backend.services._autoflip_lp import (
        MultiRegionLPResult,
        solve_multi_region_camera_path,
    )

    n_frames = len(required_per_frame)
    if n_frames == 0:
        return MultiRegionLayoutDecision(
            decision="fit",
            camera_path=[],
            n_frames=0,
            reason="empty_input",
        )

    result: MultiRegionLPResult = solve_multi_region_camera_path(
        required_per_frame,
        optional_per_frame,
        crop_width_px=crop_width_px,
        source_width_px=source_width_px,
        lam_data=lam_data,
        lam_velocity=lam_velocity,
        lam_accel=lam_accel,
        lam_jerk=lam_jerk,
        lam_soft=lam_soft,
        time_limit_sec=time_limit_sec,
    )

    base = MultiRegionLayoutDecision(
        decision="lp_failed",
        camera_path=list(result.camera_path),
        infeasible_frames=list(result.infeasible_frames),
        infeasibility_ratio=result.infeasibility_ratio,
        n_frames=result.n_frames,
        n_required=result.n_required,
        n_optional=result.n_optional,
        lp_message=result.lp_message,
        solve_ms=result.solve_ms,
        reason="lp_failed",
    )

    if result.status == "feasible":
        base.decision = "fit"
        base.reason = "lp_feasible"
        return base

    if result.status == "lp_failed":
        # HiGHS error — let the caller stick with the existing
        # single-subject path. Camera path is empty.
        base.reason = result.lp_message or "lp_failed"
        return base

    # status == "infeasible"
    ratio = result.infeasibility_ratio
    if ratio <= INFEASIBLE_THRESHOLD_FIT:
        # Mostly feasible; Phase 4+ may add the actual padding logic.
        # For Phase 3 minimal we route this through the per-content
        # fallback so the small minority of bad frames don't drag
        # the whole shot into a split.
        base.decision = "fit_with_pad"
        base.reason = f"ratio={ratio:.2%}≤{INFEASIBLE_THRESHOLD_FIT:.0%}"
        return base

    if ratio <= INFEASIBLE_THRESHOLD_SPLIT:
        base.decision = fallback_for_content(content_type)
        base.reason = (
            f"ratio={ratio:.2%}≤{INFEASIBLE_THRESHOLD_SPLIT:.0%},"
            f"content={content_type}"
        )
        return base

    # > 50% infeasible — too many frames out of fit. Force wide.
    base.decision = "wide"
    base.reason = f"ratio={ratio:.2%}>{INFEASIBLE_THRESHOLD_SPLIT:.0%}"
    return base


# ─────────────────── Required vs optional promotion ───────────────────


@dataclass
class _SpeakerActivity:
    """Lightweight per-slot recency/confidence state used by the
    required-vs-optional promoter.

    Mirrors the shape of ``backend.models.ActiveSpeakerEvent`` minus
    the irrelevant timing fields so callers can construct synthetic
    activity windows for tests without pulling the full pipeline.
    """

    slot_id: int
    last_speaking_at: float = -1.0  # latest timestamp where this slot was speaking
    max_confidence: float = 0.0


def _build_activity_index(
    active_speaker_events: list,
    seg_start: float,
    seg_end: float,
) -> dict[int, _SpeakerActivity]:
    """Index ``ActiveSpeakerEvent``-shaped objects by slot for the segment."""
    index: dict[int, _SpeakerActivity] = {}
    for ev in active_speaker_events or []:
        slot_id = int(getattr(ev, "slot_id", -1))
        if slot_id < 0:
            continue
        ev_start = float(getattr(ev, "start", 0.0))
        ev_end = float(getattr(ev, "end", 0.0))
        if ev_end < seg_start or ev_start > seg_end:
            continue
        cur = index.get(slot_id)
        if cur is None:
            cur = _SpeakerActivity(slot_id=slot_id)
            index[slot_id] = cur
        # Track the latest "speaking" anchor inside the segment window.
        clipped_end = min(ev_end, seg_end)
        if clipped_end > cur.last_speaking_at:
            cur.last_speaking_at = clipped_end
        conf = float(getattr(ev, "confidence", 0.0) or 0.0)
        if conf > cur.max_confidence:
            cur.max_confidence = conf
    return index


def promote_required_regions_for_segment(
    *,
    seg_start: float,
    seg_end: float,
    face_slots: list,
    active_speaker_events: list,
    now_t: Optional[float] = None,
    promotion_confidence: float = REQUIRED_PROMOTION_CONFIDENCE,
    promotion_recency_sec: float = REQUIRED_PROMOTION_RECENCY_SEC,
    demotion_silence_sec: float = REQUIRED_DEMOTION_SILENCE_SEC,
) -> tuple[list[int], list[int]]:
    """Decide which face slots are required vs optional inside a segment.

    Promotion rule (per the v2 spec):

        promote face_slot → required when active-speaker confidence
        for that slot > ``promotion_confidence`` (default 0.7) AND
        the slot has spoken in the last ``promotion_recency_sec``
        (default 2.0) seconds.

        demote face_slot → optional when the slot has been silent for
        more than ``demotion_silence_sec`` (default 3.0) seconds.

    Args:
        seg_start: segment start time in seconds.
        seg_end: segment end time in seconds.
        face_slots: iterable of ``FaceSlot``-shaped objects (must
            expose ``.slot_id``). Order is preserved in the output.
        active_speaker_events: list of ``ActiveSpeakerEvent``-shaped
            objects (``.slot_id``, ``.start``, ``.end``, ``.confidence``).
        now_t: the timestamp at which to evaluate recency. Defaults
            to ``seg_end`` so the recency window is "the last
            ``promotion_recency_sec`` seconds before the segment ends".
        promotion_confidence / promotion_recency_sec / demotion_silence_sec:
            tunable thresholds (defaults track the v2 spec).

    Returns:
        ``(required_slot_ids, optional_slot_ids)`` — two disjoint
        lists in slot-order. Slots not in either list have no signal
        in the segment window and are dropped (e.g. a face that
        appeared once at the start of the shot but is silent for the
        last 5s).
    """
    if now_t is None:
        now_t = seg_end
    activity = _build_activity_index(active_speaker_events, seg_start, seg_end)
    required: list[int] = []
    optional: list[int] = []
    for slot in face_slots:
        slot_id = int(getattr(slot, "slot_id", -1))
        if slot_id < 0:
            continue
        info = activity.get(slot_id)
        # No speaker activity → definitely optional (it's a passive
        # face the segment shouldn't anchor on, but the LP can still
        # try to keep it in-frame as a soft constraint).
        if info is None:
            optional.append(slot_id)
            continue
        recency = now_t - info.last_speaking_at
        if (
            info.max_confidence >= promotion_confidence
            and 0.0 <= recency <= promotion_recency_sec
        ):
            required.append(slot_id)
            continue
        if recency >= demotion_silence_sec:
            optional.append(slot_id)
            continue
        # In the gray zone (between promotion-recency and demotion-
        # silence) → still optional. The LP can fit the speaker as a
        # nice-to-have but no shot decision hangs on it.
        optional.append(slot_id)
    return required, optional


def slot_to_pixel_bbox(
    slot,
    *,
    source_width: float,
    half_width_px: Optional[float] = None,
) -> tuple[float, float]:
    """Convert a ``FaceSlot``-shaped object to a ``(left_px, right_px)`` pair.

    Reads ``slot.x_center`` (in 0–100 percent space, the existing
    convention used everywhere in the reframe pipeline) and either
    ``slot.avg_width`` (also in %) or the explicit ``half_width_px``
    override for tests.

    Returns the bbox in source-frame **pixels**, ready to feed into
    ``solve_multi_region_camera_path``.
    """
    cx_pct = float(getattr(slot, "x_center", 50.0))
    cx_px = cx_pct / 100.0 * float(source_width)
    if half_width_px is not None:
        half = float(half_width_px)
    else:
        avg_w_pct = float(getattr(slot, "avg_width", 10.0))
        half = (avg_w_pct / 100.0 * float(source_width)) / 2.0
    return (cx_px - half, cx_px + half)
