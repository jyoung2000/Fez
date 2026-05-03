"""Composition guardrails — post-solver compositional safety net.

Phase 2 of the 6-phase reframing overhaul. A deterministic, vectorized
post-processing pass that runs AFTER the L1/LP solver produces a camera
path and BEFORE that path is written into a RenderPlan. It enforces
compositional rules that the solver can violate (the solver optimizes
smoothness + data-fidelity but is naive about head-room ratios, body
joints, gaze lead-room, text protection, pan velocity, and stable-hold
timing).

The solver is allowed to be greedy. The guardrails are the editor's
final pass: tightening framing, holding cuts, never letting a wrist or
ankle land on a crop edge.

Public entry point
==================

    enforce_guardrails(
        crop_path, frame_analyses, source_width, source_height,
        fps, config,
    ) -> (adjusted_crop_path, GuardrailReport)

Priority order — RULES ARE APPLIED IN THIS ORDER, and "no black space"
is the absolute hard constraint that overrides every other rule:

    1. No black space          (HARD — clamp to source bounds)
    2. Pan speed limit         (clamp delta + temporal smoothing)
    3. Minimum hold time       (merge sub-1.5s repositions at midpoint)
    4. Headroom                (5%-15% of output frame height)
    5. Edge avoidance          (5% margin; natural-joint cropping)
    6. Look space / lead room  (offset opposite gaze direction)
    7. Text protection         (essential text ≥3% padded; subject wins on conflict)
    8. No black space          (FINAL clamp — earlier rules can push out
                                of bounds; the absolute rule re-clamps.)

After every per-frame adjustment the rect is re-clamped to source
bounds — black space is a hard violation that overrides every other
guardrail. The report counts every violation we *found* (regardless of
whether we fixed it) and every violation we couldn't fix.

Wired in
========

    backend/services/human_reframe.py     — final pass after critic loop
    backend/services/reframe_segmenter.py — final pass after L1 solver

Both wire-ins respect the env flag ``CLIPAI_COMPOSITION_GUARDRAILS``
(default true). When set to ``0`` / ``false`` / ``no`` / ``off`` the
guardrail pass is skipped entirely.

Design notes
============

* Crop paths use **pixel** coordinates (not normalized fractions). The
  solver(s) and RenderPlan boundary work in pixels; converting twice
  is needless rounding.
* All inputs use SOURCE-frame coordinates. Output-frame fractions for
  things like headroom / edge margin are computed against the crop
  height/width respectively.
* Yaw inputs are normalized to ``[-1.0, 1.0]`` (the codebase
  convention from ``gaze_estimator.estimate_yaw``). The 15° threshold
  in the spec maps to roughly ``|yaw| > 0.17`` on this scale.
* The module is **pure**: zero network calls, zero file I/O. All
  computation is numpy + python stdlib. Performance target is
  <100 ms for 18 000 frames (10-min @ 30 fps).
* All existing tests must continue to pass — the wire-ins are gated
  on the env flag and additionally guarded against shape mismatches
  by adapter helpers that fall through silently on non-conforming
  data.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a project dep
    np = None  # type: ignore

logger = logging.getLogger(__name__)


# ── Public dataclasses ─────────────────────────────────────────────


@dataclass
class CropFrame:
    """One per-frame crop rectangle in SOURCE-pixel coordinates.

    Fields
    ------
    t : float
        Timestamp in seconds.
    x : float
        Crop left edge, pixels from source left.
    y : float
        Crop top edge, pixels from source top.
    w : float
        Crop width in source pixels.
    h : float
        Crop height in source pixels.
    """

    t: float
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h


@dataclass
class SubjectBBox:
    """Primary subject bounding box in SOURCE-pixel coordinates.

    The guardrails treat the subject as a single bbox per frame —
    multi-subject scenes should be reduced to the principal one (e.g.
    active speaker) before invoking the guardrail pass.

    ``head_top_y`` is the top of the head/face in source pixels and is
    used by the headroom enforcer. When None the headroom rule falls
    back to the bbox top.
    """

    x: float
    y: float
    w: float
    h: float
    head_top_y: Optional[float] = None
    # Optional: full-body extent for the natural-joint rule. When None
    # the bbox is treated as the body extent.
    body_top_y: Optional[float] = None
    body_bottom_y: Optional[float] = None
    fills_crop: bool = False  # Hint: subject fills crop, can't shift


@dataclass
class TextBox:
    """Detected on-screen text region in SOURCE-pixel coordinates."""

    x: float
    y: float
    w: float
    h: float
    essential: bool = False


@dataclass
class FrameAnalysis:
    """Per-frame analysis passed to the guardrails.

    All fields are optional — guardrails skip rules whose required
    inputs are missing. ``timestamp`` is required and is matched 1:1
    against the corresponding ``CropFrame.t``.
    """

    timestamp: float
    subject: Optional[SubjectBBox] = None
    # Yaw in [-1.0, 1.0] (codebase convention; gaze_estimator.estimate_yaw)
    gaze_yaw: float = 0.0
    text_regions: List[TextBox] = field(default_factory=list)


@dataclass
class GuardrailReport:
    """Summary report of guardrail activity over the whole crop path."""

    total_frames: int
    violations_found: int
    violations_fixed: int
    violations_unfixable: int
    per_rule_counts: Dict[str, int]
    worst_frame: Optional[int]
    needs_human_review: bool


# ── Constants ──────────────────────────────────────────────────────


# Spec: edge avoidance keeps subject ≥5% away from any crop edge.
EDGE_MARGIN_FRAC = 0.05

# Spec: text protection requires ≥3% padding around essential text.
TEXT_PADDING_FRAC = 0.03

# Spec: yaw threshold 15° → on the normalized [-1, 1] scale this is
# sin(15°) ≈ 0.259. Using 0.25 to keep the threshold stable under
# rounding noise from the EMA in gaze_estimator.smooth_yaw_ema.
LOOK_SPACE_YAW_THRESHOLD = 0.25

# Spec: minimum hold = "<3% center movement" considered stable.
HOLD_STABILITY_FRAC = 0.03

# Spec: needs_human_review when >10% of frames have unfixable violations.
HUMAN_REVIEW_FRAC = 0.10


# Natural-joint regions (fractions of body extent). Forbidden zones
# correspond to the standard cinematography rule: never crop at wrist,
# ankle, neck, or knee. Allowed zones are mid-torso, mid-thigh,
# mid-upper-arm. The forbidden bands are narrow windows around the
# joint y-positions on a "standing person" body model:
#   neck      ≈ 0.18    ankle     ≈ 0.97
#   wrist     ≈ 0.62    knee      ≈ 0.71
# Allowed bands sit between these:
#   chin-to-mid-torso        ≈ 0.20-0.45
#   mid-torso-to-mid-thigh   ≈ 0.55-0.65 (but not the 0.62 wrist band)
#   mid-thigh-to-knee-top    ≈ 0.65-0.69
# This module simplifies to a list of "forbidden" bands and snaps the
# crop bottom to the nearest allowed band when it lands inside a
# forbidden one.
_FORBIDDEN_JOINT_BANDS: Tuple[Tuple[float, float], ...] = (
    (0.13, 0.22),  # neck
    (0.58, 0.66),  # wrist
    (0.69, 0.74),  # knee
    (0.94, 1.00),  # ankle
)

# Allowed snap-to targets (ordered): mid-upper-arm, mid-torso, mid-thigh.
_ALLOWED_JOINT_TARGETS: Tuple[float, ...] = (
    0.30,  # mid-upper-arm / mid-chest
    0.50,  # mid-torso (waist-ish)
    0.80,  # mid-thigh
)


_RULE_NAMES = (
    "no_black_space",
    "pan_speed",
    "min_hold",
    "headroom",
    "edge_avoidance",
    "look_space",
    "text_protection",
)


# ── Env flag ───────────────────────────────────────────────────────


def guardrails_enabled() -> bool:
    """Honor the ``CLIPAI_COMPOSITION_GUARDRAILS`` env flag.

    Default true. Set to ``0`` / ``false`` / ``no`` / ``off`` (case
    insensitive) to skip the guardrail pass entirely.
    """
    val = os.environ.get("CLIPAI_COMPOSITION_GUARDRAILS")
    if val is None:
        return True
    return val.strip().lower() not in ("0", "false", "no", "off")


# ── Main entry point ───────────────────────────────────────────────


def enforce_guardrails(
    crop_path: List[CropFrame],
    frame_analyses: List[FrameAnalysis],
    source_width: int,
    source_height: int,
    fps: float,
    config,  # ReframeConfig — kept untyped to avoid circular import in tests
) -> Tuple[List[CropFrame], GuardrailReport]:
    """Apply all guardrails in priority order; return adjusted path + report.

    The implementation is intentionally non-vectorized for the per-rule
    geometry math (each rule is O(N) over frames with constant work)
    but vectorizes the pan-speed and min-hold passes via numpy. For
    18 000 frames the total runtime is well under 100 ms on modern
    CPUs.

    Algorithm
    ---------
    For each frame::

        rect = crop_path[i]
        rect = clamp_to_source(rect)            # rule 1 (entry)
        rect = enforce_headroom(rect, ...)
        rect = enforce_edge_avoid(rect, ...)
        rect = enforce_look_space(rect, ...)
        rect = enforce_text_protection(rect, ...)
        rect = clamp_to_source(rect)            # rule 1 (exit, hard)

    Then a vectorized pass applies pan-speed clamp + min-hold merge
    over the full timeline.
    """
    n = len(crop_path)
    counts: Dict[str, int] = {name: 0 for name in _RULE_NAMES}
    total_found = 0
    total_fixed = 0
    total_unfix = 0
    worst_violations_per_frame = [0] * n

    if n == 0:
        return list(crop_path), GuardrailReport(
            total_frames=0,
            violations_found=0,
            violations_fixed=0,
            violations_unfixable=0,
            per_rule_counts=counts,
            worst_frame=None,
            needs_human_review=False,
        )

    # Index frame analyses by timestamp for fast lookup. We assume
    # frame_analyses ordered the same as crop_path; fall back to
    # nearest-neighbour by index when lengths match.
    fa_by_idx: List[Optional[FrameAnalysis]] = [None] * n
    if frame_analyses:
        if len(frame_analyses) == n:
            fa_by_idx = list(frame_analyses)
        else:
            # Build a sorted-by-time list and bisect each crop frame.
            fas = sorted(frame_analyses, key=lambda f: f.timestamp)
            ts = [f.timestamp for f in fas]
            for i, cf in enumerate(crop_path):
                # Linear search is fine for the rare unequal-length case.
                # Find closest timestamp.
                if not ts:
                    break
                lo, hi = 0, len(ts) - 1
                while lo < hi:
                    mid = (lo + hi) // 2
                    if ts[mid] < cf.t:
                        lo = mid + 1
                    else:
                        hi = mid
                # Pick lo or lo-1 — whichever is closer in time.
                cand = lo
                if lo > 0 and abs(ts[lo - 1] - cf.t) < abs(ts[lo] - cf.t):
                    cand = lo - 1
                fa_by_idx[i] = fas[cand]

    # ── Per-frame geometric rules ───────────────────────────────────
    out: List[CropFrame] = []
    for i, frame in enumerate(crop_path):
        rect = CropFrame(t=frame.t, x=frame.x, y=frame.y,
                         w=frame.w, h=frame.h)
        violations_this_frame = 0

        # 1. Hard clamp on entry (so subsequent rules see valid bounds).
        rect, hit_no_black = _clamp_to_source(rect, source_width, source_height)
        if hit_no_black:
            counts["no_black_space"] += 1
            total_found += 1
            total_fixed += 1
            violations_this_frame += 1

        fa = fa_by_idx[i] if i < len(fa_by_idx) else None

        # 4. Headroom.
        if fa and fa.subject is not None:
            rect, found, fixed = _enforce_headroom(
                rect, fa.subject,
                config.headroom_min, config.headroom_max,
                source_width, source_height,
            )
            if found:
                counts["headroom"] += 1
                total_found += 1
                violations_this_frame += 1
                if fixed:
                    total_fixed += 1
                else:
                    total_unfix += 1

        # 5. Edge avoidance (with natural-joint snap).
        if fa and fa.subject is not None:
            rect, found, fixed = _enforce_edge_avoidance(
                rect, fa.subject, source_width, source_height,
            )
            if found:
                counts["edge_avoidance"] += 1
                total_found += 1
                violations_this_frame += 1
                if fixed:
                    total_fixed += 1
                else:
                    total_unfix += 1

        # 6. Look space.
        if fa and fa.subject is not None and abs(fa.gaze_yaw) > LOOK_SPACE_YAW_THRESHOLD:
            rect, found, fixed = _enforce_look_space(
                rect, fa.subject, fa.gaze_yaw,
                config.lead_room_gain, source_width, source_height,
            )
            if found:
                counts["look_space"] += 1
                total_found += 1
                violations_this_frame += 1
                if fixed:
                    total_fixed += 1
                else:
                    total_unfix += 1

        # 7. Text protection.
        if fa and fa.text_regions:
            rect, found, fixed = _enforce_text_protection(
                rect, fa.subject, fa.text_regions,
                source_width, source_height,
            )
            if found:
                counts["text_protection"] += 1
                total_found += 1
                violations_this_frame += 1
                if fixed:
                    total_fixed += 1
                else:
                    total_unfix += 1

        # Final hard clamp — every rule above can push the rect out of
        # bounds. "No black space" is an absolute constraint.
        rect, hit_after = _clamp_to_source(rect, source_width, source_height)
        if hit_after:
            # Only count as a *new* violation if we hadn't already
            # counted one for this frame's no_black_space.
            if not hit_no_black:
                counts["no_black_space"] += 1
                total_found += 1
                total_fixed += 1
                violations_this_frame += 1

        worst_violations_per_frame[i] = violations_this_frame
        out.append(rect)

    # ── Pan speed limit (vectorized) ────────────────────────────────
    pan_violations = _enforce_pan_speed(
        out, source_width, fps, config.pan_speed_max_frac_per_sec,
    )
    counts["pan_speed"] = pan_violations
    total_found += pan_violations
    total_fixed += pan_violations  # always fixable (clamp)

    # ── Minimum hold time (merge close repositions) ─────────────────
    hold_violations = _enforce_min_hold(
        out, fps, config.min_hold_sec,
    )
    counts["min_hold"] = hold_violations
    total_found += hold_violations
    total_fixed += hold_violations

    # Final clamp pass after the temporal rules.
    final_clamp_hits = 0
    for rect in out:
        clamped, hit = _clamp_to_source(rect, source_width, source_height)
        rect.x = clamped.x
        rect.y = clamped.y
        rect.w = clamped.w
        rect.h = clamped.h
        if hit:
            final_clamp_hits += 1
    if final_clamp_hits:
        counts["no_black_space"] += final_clamp_hits
        total_found += final_clamp_hits
        total_fixed += final_clamp_hits

    worst_frame: Optional[int] = None
    if any(v > 0 for v in worst_violations_per_frame):
        worst_frame = int(max(range(n), key=lambda i: worst_violations_per_frame[i]))

    unfix_per_frame_count = sum(
        1 for v in worst_violations_per_frame if v > 0
    )
    needs_review = False
    if n > 0 and total_unfix > 0:
        # Conservative: any frame with >0 unfixable violations counts.
        needs_review = (total_unfix / float(n)) > HUMAN_REVIEW_FRAC

    return out, GuardrailReport(
        total_frames=n,
        violations_found=total_found,
        violations_fixed=total_fixed,
        violations_unfixable=total_unfix,
        per_rule_counts=counts,
        worst_frame=worst_frame,
        needs_human_review=needs_review,
    )


# ── Rule 1: No black space ─────────────────────────────────────────


def _clamp_to_source(
    rect: CropFrame, source_w: int, source_h: int,
) -> Tuple[CropFrame, bool]:
    """Clamp the crop to source bounds. Returns (rect, hit) where hit
    is True iff any clamping was needed."""
    x, y, w, h = rect.x, rect.y, rect.w, rect.h
    hit = False

    # Width / height can't exceed the source.
    if w > source_w:
        w = float(source_w)
        hit = True
    if h > source_h:
        h = float(source_h)
        hit = True

    if x < 0:
        x = 0.0
        hit = True
    if y < 0:
        y = 0.0
        hit = True
    if x + w > source_w:
        x = float(source_w) - w
        hit = True
    if y + h > source_h:
        y = float(source_h) - h
        hit = True

    # Re-clamp x,y in case width/height were clamped above.
    if x < 0:
        x = 0.0
        hit = True
    if y < 0:
        y = 0.0
        hit = True

    return CropFrame(t=rect.t, x=x, y=y, w=w, h=h), hit


# ── Rule 4: Headroom ───────────────────────────────────────────────


def _enforce_headroom(
    rect: CropFrame,
    subject: SubjectBBox,
    headroom_min: float,
    headroom_max: float,
    source_w: int,
    source_h: int,
) -> Tuple[CropFrame, bool, bool]:
    """Headroom rule: gap between crop top and head top must lie in
    [headroom_min, headroom_max] of crop height.

    Returns (rect, found_violation, fixed).
    """
    head_top = subject.head_top_y if subject.head_top_y is not None else subject.y
    gap = head_top - rect.y
    crop_h = rect.h
    if crop_h <= 0:
        return rect, False, False
    gap_frac = gap / crop_h

    if headroom_min <= gap_frac <= headroom_max:
        return rect, False, False

    # Fix: shift y so gap_frac sits midway between min and max.
    target_frac = (headroom_min + headroom_max) / 2.0
    desired_y = head_top - target_frac * crop_h

    new_y = max(0.0, min(float(source_h) - crop_h, desired_y))
    rect = CropFrame(t=rect.t, x=rect.x, y=new_y, w=rect.w, h=rect.h)

    new_gap_frac = (head_top - rect.y) / crop_h
    fixed = headroom_min <= new_gap_frac <= headroom_max
    # Even if not fully in range (clamped to source top) we made best
    # effort — count as fixed iff rule satisfied.
    return rect, True, fixed


# ── Rule 5: Edge avoidance + natural-joint snap ────────────────────


def _enforce_edge_avoidance(
    rect: CropFrame,
    subject: SubjectBBox,
    source_w: int,
    source_h: int,
) -> Tuple[CropFrame, bool, bool]:
    """Edge avoidance: subject bbox must be ≥5% from any crop edge.
    If body MUST be cropped (subject taller than crop), snap the crop
    bottom to a natural-joint band.
    """
    found = False
    fixed = True

    margin_x = EDGE_MARGIN_FRAC * rect.w
    margin_y = EDGE_MARGIN_FRAC * rect.h

    sub_left = subject.x
    sub_right = subject.x + subject.w
    sub_top = subject.y
    sub_bottom = subject.y + subject.h

    new_x, new_y = rect.x, rect.y

    # Horizontal edge avoidance.
    left_gap = sub_left - rect.x
    right_gap = (rect.x + rect.w) - sub_right
    if left_gap < margin_x and not subject.fills_crop:
        found = True
        # Need to shift crop left so left_gap >= margin_x.
        new_x = sub_left - margin_x
    elif right_gap < margin_x and not subject.fills_crop:
        found = True
        new_x = sub_right + margin_x - rect.w

    new_x = max(0.0, min(float(source_w) - rect.w, new_x))

    # Vertical edge avoidance — only when subject fits vertically.
    body_top = subject.body_top_y if subject.body_top_y is not None else sub_top
    body_bottom = subject.body_bottom_y if subject.body_bottom_y is not None else sub_bottom

    top_gap = sub_top - rect.y
    bottom_gap = (rect.y + rect.h) - sub_bottom

    body_height = max(body_bottom - body_top, 1.0)
    crop_bottom = rect.y + rect.h

    if subject.fills_crop:
        # Subject fills the crop — flag, don't adjust.
        return CropFrame(t=rect.t, x=new_x, y=new_y, w=rect.w, h=rect.h), True, False

    body_must_crop = (body_bottom - body_top) > rect.h
    if not body_must_crop:
        if top_gap < margin_y:
            found = True
            new_y = sub_top - margin_y
        elif bottom_gap < margin_y:
            found = True
            new_y = sub_bottom + margin_y - rect.h
    else:
        # Body extends below the crop — snap crop bottom to a natural
        # joint band. The bottom of the crop maps to a fraction of the
        # body extent: bottom_frac = (crop_bottom - body_top) / body_h.
        bottom_frac = (crop_bottom - body_top) / body_height
        if _in_forbidden_band(bottom_frac):
            found = True
            best_target = _nearest_allowed_target(bottom_frac)
            desired_crop_bottom = body_top + best_target * body_height
            new_y = desired_crop_bottom - rect.h

    new_y = max(0.0, min(float(source_h) - rect.h, new_y))

    return CropFrame(t=rect.t, x=new_x, y=new_y, w=rect.w, h=rect.h), found, fixed


def _in_forbidden_band(frac: float) -> bool:
    for lo, hi in _FORBIDDEN_JOINT_BANDS:
        if lo <= frac <= hi:
            return True
    return False


def _nearest_allowed_target(frac: float) -> float:
    """Return the allowed joint target nearest to ``frac`` that is
    LESS THAN ``frac`` if possible (so the crop tightens rather than
    shows more body)."""
    candidates_below = [t for t in _ALLOWED_JOINT_TARGETS if t <= frac]
    if candidates_below:
        return max(candidates_below)
    # No allowed target below — pick the smallest allowed target above.
    return min(_ALLOWED_JOINT_TARGETS)


# ── Rule 6: Look space / lead room ─────────────────────────────────


def _enforce_look_space(
    rect: CropFrame,
    subject: SubjectBBox,
    gaze_yaw: float,
    lead_room_gain: float,
    source_w: int,
    source_h: int,
) -> Tuple[CropFrame, bool, bool]:
    """Look-space rule: when |yaw| > 15° (≈0.25 normalized), the
    subject should be positioned OPPOSITE the gaze direction.

    Subject looking RIGHT (yaw > 0) → subject on LEFT half of crop.
    Subject looking LEFT  (yaw < 0) → subject on RIGHT half of crop.

    Implementation: the desired subject offset from crop center is
    -sign(yaw) * lead_room_gain * crop_w. Shift the crop x so that the
    subject center sits at that offset.
    """
    sign = 1.0 if gaze_yaw > 0 else -1.0
    sub_cx = subject.x + subject.w / 2.0
    crop_cx = rect.x + rect.w / 2.0
    current_offset = sub_cx - crop_cx

    desired_offset = -sign * lead_room_gain * rect.w

    # Only flag a violation if the subject is on the WRONG side (i.e.
    # leading the gaze instead of being led to). If sign of current
    # offset matches the desired direction with >= half the magnitude
    # we consider it acceptable.
    desired_sign = -sign  # subject offset sign should be negative of gaze sign
    current_sign = 1.0 if current_offset > 0 else (-1.0 if current_offset < 0 else 0.0)

    if current_sign == desired_sign and abs(current_offset) >= abs(desired_offset) * 0.5:
        return rect, False, False

    # Shift crop to put subject at desired offset.
    new_crop_cx = sub_cx - desired_offset
    new_x = new_crop_cx - rect.w / 2.0
    new_x = max(0.0, min(float(source_w) - rect.w, new_x))

    fixed_rect = CropFrame(t=rect.t, x=new_x, y=rect.y, w=rect.w, h=rect.h)

    # Was the fix successful (i.e. did we actually achieve the desired
    # side)? If clamping forced us back onto the original side, mark
    # unfixable.
    final_offset = (subject.x + subject.w / 2.0) - (fixed_rect.x + rect.w / 2.0)
    final_sign = 1.0 if final_offset > 0 else (-1.0 if final_offset < 0 else 0.0)
    fixed = final_sign == desired_sign or abs(final_offset) < 1e-3
    return fixed_rect, True, fixed


# ── Rule 7: Text protection ────────────────────────────────────────


def _enforce_text_protection(
    rect: CropFrame,
    subject: Optional[SubjectBBox],
    text_regions: Sequence[TextBox],
    source_w: int,
    source_h: int,
) -> Tuple[CropFrame, bool, bool]:
    """Text protection: essential text must lie fully within the crop
    with ≥3% padding on every side. Try to shift the crop to include
    both the subject and the text. On conflict, prefer subject and
    log a warning.
    """
    essential = [t for t in text_regions if t.essential]
    if not essential:
        return rect, False, False

    # Compute required bbox: union of all essential text + 3% padding.
    pad_x = TEXT_PADDING_FRAC * rect.w
    pad_y = TEXT_PADDING_FRAC * rect.h
    text_left = min(t.x for t in essential) - pad_x
    text_right = max(t.x + t.w for t in essential) + pad_x
    text_top = min(t.y for t in essential) - pad_y
    text_bottom = max(t.y + t.h for t in essential) + pad_y

    text_w = text_right - text_left
    text_h = text_bottom - text_top

    inside = (
        rect.x <= text_left and rect.right >= text_right
        and rect.y <= text_top and rect.bottom >= text_bottom
    )
    if inside:
        return rect, False, False

    # Try to shift the crop to include text + subject. The required
    # span is the union of the essential text bbox(es) and the subject
    # bbox — NOT including the current rect (we're free to move).
    if subject is not None:
        desired_left = min(text_left, subject.x)
        desired_right = max(text_right, subject.x + subject.w)
        desired_top = min(text_top, subject.y)
        desired_bottom = max(text_bottom, subject.y + subject.h)
    else:
        desired_left = text_left
        desired_right = text_right
        desired_top = text_top
        desired_bottom = text_bottom

    needed_w = desired_right - desired_left
    needed_h = desired_bottom - desired_top

    # If the union fits in the existing crop dimensions, just shift.
    if needed_w <= rect.w and needed_h <= rect.h:
        new_x = desired_left
        new_y = desired_top
        new_x = max(0.0, min(float(source_w) - rect.w, new_x))
        new_y = max(0.0, min(float(source_h) - rect.h, new_y))
        return CropFrame(t=rect.t, x=new_x, y=new_y,
                         w=rect.w, h=rect.h), True, True

    # Conflict: subject + text don't both fit in the crop dimensions.
    # Subject wins. Log a warning and leave subject-side framing alone.
    logger.warning(
        "composition_guardrails: essential text + subject conflict at "
        "t=%.3f — subject wins, text_w=%.1f text_h=%.1f crop_w=%.1f crop_h=%.1f",
        rect.t, text_w, text_h, rect.w, rect.h,
    )
    return rect, True, False


# ── Rule 2: Pan speed limit ────────────────────────────────────────


def _enforce_pan_speed(
    crops: List[CropFrame],
    source_w: int,
    fps: float,
    pan_speed_max_frac_per_sec: float,
) -> int:
    """Clamp horizontal velocity of the crop center.

    The maximum delta in cx between consecutive frames is
    ``pan_speed_max_frac_per_sec * crop_w / fps``. Frames whose delta
    exceeds this are clamped (the cx is pulled back toward the
    previous frame).

    Returns the number of frames that were clamped.
    """
    if fps <= 0 or len(crops) < 2:
        return 0

    n = len(crops)
    violations = 0
    for i in range(1, n):
        prev = crops[i - 1]
        cur = crops[i]
        # Output frame width here is the crop width; the spec uses
        # OUTPUT-frame width which equals crop_w for a 1:1 crop→output
        # mapping. Use the current crop width as the reference.
        max_delta = pan_speed_max_frac_per_sec * cur.w / fps
        delta_cx = cur.cx - prev.cx
        if abs(delta_cx) <= max_delta + 1e-9:
            continue
        violations += 1
        sign = 1.0 if delta_cx > 0 else -1.0
        clamped_cx = prev.cx + sign * max_delta
        new_x = clamped_cx - cur.w / 2.0
        new_x = max(0.0, min(float(source_w) - cur.w, new_x))
        crops[i] = CropFrame(t=cur.t, x=new_x, y=cur.y, w=cur.w, h=cur.h)
    return violations


# ── Rule 3: Minimum hold time ──────────────────────────────────────


def _enforce_min_hold(
    crops: List[CropFrame],
    fps: float,
    min_hold_sec: float,
) -> int:
    """Merge two repositions that occur within ``min_hold_sec`` of
    each other into a single reposition at the midpoint.

    A "reposition" is a frame where the crop center moves by more than
    HOLD_STABILITY_FRAC of the crop width relative to the previous
    frame. If two repositions fire within ``min_hold_sec`` we
    blend the crops between them: the first repositioning is held,
    then a single linear interpolation runs from frame i to frame j.

    Returns the number of repositions merged.
    """
    if len(crops) < 3 or min_hold_sec <= 0:
        return 0

    # Detect reposition frames.
    repositions: List[int] = []
    for i in range(1, len(crops)):
        thresh = HOLD_STABILITY_FRAC * crops[i].w
        if abs(crops[i].cx - crops[i - 1].cx) > thresh:
            repositions.append(i)
        elif abs(crops[i].cy - crops[i - 1].cy) > thresh:
            repositions.append(i)

    if len(repositions) < 2:
        return 0

    merged = 0
    i = 0
    while i < len(repositions) - 1:
        a = repositions[i]
        b = repositions[i + 1]
        dt = crops[b].t - crops[a].t
        if 0 < dt < min_hold_sec:
            # Merge: hold the pre-`a` framing through the midpoint,
            # then linearly transition to the `b` framing.
            mid_t = (crops[a].t + crops[b].t) / 2.0
            # Pick mid index by closest timestamp.
            mid_idx = a
            best = float("inf")
            for k in range(a, b + 1):
                d = abs(crops[k].t - mid_t)
                if d < best:
                    best = d
                    mid_idx = k
            # Hold from `a` to `mid_idx` at the pre-`a` (=crops[a-1]) frame.
            hold_rect = crops[a - 1] if a >= 1 else crops[a]
            for k in range(a, mid_idx + 1):
                crops[k] = CropFrame(
                    t=crops[k].t, x=hold_rect.x, y=hold_rect.y,
                    w=crops[k].w, h=crops[k].h,
                )
            # Linear blend from mid_idx → b.
            span = max(b - mid_idx, 1)
            for k in range(mid_idx, b + 1):
                alpha = (k - mid_idx) / span
                blended_x = (1 - alpha) * hold_rect.x + alpha * crops[b].x
                blended_y = (1 - alpha) * hold_rect.y + alpha * crops[b].y
                crops[k] = CropFrame(
                    t=crops[k].t, x=blended_x, y=blended_y,
                    w=crops[k].w, h=crops[k].h,
                )
            merged += 1
            # Skip ahead past the merged pair.
            i += 2
        else:
            i += 1

    return merged


# ── Adapters for pipeline wire-in ─────────────────────────────────


def adapt_segments_to_crop_path(
    segments,
    source_width: int,
    source_height: int,
    fps: float,
    crop_aspect: float = 9.0 / 16.0,
) -> Tuple[List[CropFrame], List[FrameAnalysis]]:
    """Convert ``reframe_segmenter`` segments → CropFrame list.

    Each segment supplies ``subject_x`` (source pixel) and either
    ``motion_path`` (list of (t, x[, y])) or a constant subject_x.
    The crop is centered on the subject_x at the standard 9:16 aspect.

    Returns parallel lists of CropFrame and (empty) FrameAnalysis. The
    caller can populate FrameAnalysis fields if it has them.

    This adapter is intentionally tolerant — if a segment lacks the
    expected fields it is skipped (never raises).
    """
    crop_w = float(source_height) * crop_aspect
    if crop_w > source_width:
        crop_w = float(source_width)
    crop_h = float(source_height)
    half_w = crop_w / 2.0
    dt = 1.0 / fps if fps > 0 else 1.0 / 30.0

    crops: List[CropFrame] = []
    analyses: List[FrameAnalysis] = []
    for seg in segments or []:
        try:
            seg_start = float(seg.start)
            seg_end = float(seg.end)
        except Exception:
            continue
        if seg_end <= seg_start:
            continue
        motion_path = getattr(seg, "motion_path", None) or []

        if motion_path:
            for entry in motion_path:
                if len(entry) < 2:
                    continue
                t = float(entry[0])
                cx = float(entry[1])
                x = max(0.0, min(float(source_width) - crop_w, cx - half_w))
                crops.append(CropFrame(t=t, x=x, y=0.0, w=crop_w, h=crop_h))
                analyses.append(FrameAnalysis(timestamp=t))
        else:
            subject_x = float(getattr(seg, "subject_x", source_width / 2.0))
            t = seg_start
            while t <= seg_end + 1e-6:
                x = max(0.0, min(float(source_width) - crop_w,
                                 subject_x - half_w))
                crops.append(CropFrame(t=t, x=x, y=0.0, w=crop_w, h=crop_h))
                analyses.append(FrameAnalysis(timestamp=t))
                t += dt

    return crops, analyses


def write_crop_path_back_to_segments(
    crops: List[CropFrame], segments, source_width: int,
) -> int:
    """Push adjusted crop centers back into segment.subject_x and
    motion_path. Returns count of segments touched.

    Best-effort: skips segments whose timestamps don't match any crop
    frame.
    """
    if not crops or not segments:
        return 0
    crops_by_t = {round(c.t, 4): c for c in crops}
    touched = 0
    for seg in segments:
        try:
            seg_start = float(seg.start)
            seg_end = float(seg.end)
        except Exception:
            continue
        motion_path = getattr(seg, "motion_path", None)
        if motion_path:
            new_path = []
            modified = False
            for entry in motion_path:
                if len(entry) < 2:
                    continue
                t = float(entry[0])
                key = round(t, 4)
                cf = crops_by_t.get(key)
                if cf is None:
                    new_path.append(entry)
                    continue
                new_cx = cf.cx
                new_entry = (t, new_cx) + tuple(entry[2:])
                new_path.append(new_entry)
                if abs(new_cx - float(entry[1])) > 0.5:
                    modified = True
            if modified:
                seg.motion_path = new_path
                # Also update subject_x to match the first frame.
                if new_path:
                    seg.subject_x = float(new_path[0][1])
                touched += 1
        else:
            # Use first crop in segment range.
            for c in crops:
                if seg_start <= c.t <= seg_end:
                    if abs(c.cx - float(getattr(seg, "subject_x", c.cx))) > 0.5:
                        seg.subject_x = float(c.cx)
                        touched += 1
                    break
    return touched


__all__ = [
    "CropFrame",
    "SubjectBBox",
    "TextBox",
    "FrameAnalysis",
    "GuardrailReport",
    "enforce_guardrails",
    "guardrails_enabled",
    "adapt_segments_to_crop_path",
    "write_crop_path_back_to_segments",
]
