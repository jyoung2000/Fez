"""Local reframe critic (Fix 3.8).

Scores every 1-second window of a :class:`RenderPlan` against the
subject-intent track and the dense face detections. Entirely local,
no network. Pairs with ``intent_track.IntentSample`` + dense faces
to catch chin-clips, head-clips, off-center framing, and jittery
cx/cy before the clip ships.

Score is 0–10; weights per check:

  subject-centering  3.0
  headroom           2.5
  chin-clip          2.0  (hard fail = 0 on the window)
  side-clip          1.5
  jitter             1.0

A window is "low" when score < config.critic_threshold (default 6.0).
``attempt_window_fix`` tries to repair a low-scoring window by
re-solving the shot with tighter headroom, higher jitter damping, or
a slot switch. ``apply_window_fix`` splices the repaired op back into
the plan. Windows are never re-solved twice (oscillation guard).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from backend.services.reframe_config import ReframeConfig, get_default_config
from backend.services.render_plan import Rect, RenderOp, RenderOpKind, RenderPlan

logger = logging.getLogger(__name__)


@dataclass
class CriticScore:
    t_start: float
    t_end: float
    score: float
    reasons: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────────


def _op_at(plan: RenderPlan, t: float) -> Optional[RenderOp]:
    for op in plan.ops:
        if op.start_sec <= t < op.end_sec:
            return op
    return None


def _rect_at(op: RenderOp, t: float) -> Optional[Rect]:
    """Return the primary_rect at t — handles motion_path interpolation."""
    if op.motion_path:
        # Linear interpolate between keypoints that bracket t.
        kps = sorted(op.motion_path, key=lambda kp: kp.t)
        if t <= kps[0].t:
            return kps[0].rect
        if t >= kps[-1].t:
            return kps[-1].rect
        for i in range(len(kps) - 1):
            a, b = kps[i], kps[i + 1]
            if a.t <= t <= b.t:
                dur = max(1e-6, b.t - a.t)
                u = (t - a.t) / dur
                return Rect(
                    x=a.rect.x + (b.rect.x - a.rect.x) * u,
                    y=a.rect.y + (b.rect.y - a.rect.y) * u,
                    w=a.rect.w + (b.rect.w - a.rect.w) * u,
                    h=a.rect.h + (b.rect.h - a.rect.h) * u,
                )
    return op.primary_rect


def _face_at(dense_faces: list, t: float, window: float = 0.15) -> Optional[object]:
    """Largest face at the nearest dense-face frame within ``window``."""
    best = None
    best_dt = window
    for df in dense_faces or []:
        if not df.faces:
            continue
        dt = abs(getattr(df, "timestamp", 0.0) - t)
        if dt <= best_dt:
            candidates = [f for f in df.faces if getattr(f, "is_human", True)]
            if candidates:
                best = max(candidates, key=lambda f: f.width * f.height)
                best_dt = dt
    return best


def _slot_face_at(
    dense_faces: list,
    t: float,
    slot_id: Optional[int],
    window: float = 0.15,
) -> Optional[object]:
    """Largest face whose ``identity_id == slot_id`` at the nearest
    dense-face frame within ``window``. Returns None when ``slot_id`` is
    None or when that slot has no face in the search window.

    Companion to :func:`_face_at`: the unrestricted lookup picks the
    largest face regardless of identity, which silently switches
    reference points across reaction-shot cuts in panel content. The
    critic uses this helper to detect "segmenter chose slot X but slot
    X is not visible" — the off-screen-active-speaker class of bug.
    """
    if slot_id is None:
        return None
    best = None
    best_dt = window
    for df in dense_faces or []:
        if not df.faces:
            continue
        dt = abs(getattr(df, "timestamp", 0.0) - t)
        if dt > best_dt:
            continue
        candidates = [
            f for f in df.faces
            if int(getattr(f, "identity_id", -1)) == int(slot_id)
            and getattr(f, "is_human", True)
        ]
        if candidates:
            best = max(candidates, key=lambda f: f.width * f.height)
            best_dt = dt
    return best


# ── Scoring ───────────────────────────────────────────────────────


def _saliency_in_crop_at(
    saliency_peaks_per_second: Optional[list],
    t: float,
    crop_left: float,
    crop_right: float,
) -> Optional[bool]:
    """Layer 1: does any top-K saliency peak at second ``int(t)`` lie
    horizontally inside ``[crop_left, crop_right]``?

    Returns None when no saliency data is available for this second
    (so the caller treats it as "not measured" rather than a
    pass/fail). Returns True when ≥1 peak is in-crop, False when peaks
    exist but all sit outside the crop.

    ``saliency_peaks_per_second[s]`` is a list of ``(x_pct, y_pct,
    score)`` triples in [0, 100] (the AvSaliency / parity-bench
    convention). ``crop_left`` / ``crop_right`` are source-frame
    fractions in [0, 1].
    """
    if not saliency_peaks_per_second:
        return None
    sec_idx = int(t)
    if sec_idx < 0 or sec_idx >= len(saliency_peaks_per_second):
        return None
    peaks = saliency_peaks_per_second[sec_idx] or []
    if not peaks:
        return None
    left_pct = crop_left * 100.0
    right_pct = crop_right * 100.0
    for peak in peaks:
        if not peak:
            continue
        try:
            px = float(peak[0])
        except (TypeError, ValueError, IndexError):
            continue
        if left_pct <= px <= right_pct:
            return True
    return False


def _score_window(
    t0: float, t1: float,
    plan: RenderPlan,
    *,
    dense_faces: list,
    config: ReframeConfig,
    saliency_peaks_per_second: Optional[list] = None,
) -> CriticScore:
    reasons: list = []
    metrics: dict = {}

    # Sample at 10 Hz inside the window.
    n_steps = max(2, int(round((t1 - t0) * 10.0)))
    step = (t1 - t0) / n_steps
    cx_samples: list[float] = []
    saliency_in_crop_hits: int = 0
    saliency_in_crop_total: int = 0

    centering_err = 0.0
    headroom_bad = 0
    headroom_unreachable = 0
    chin_clip_frames = 0
    side_clip_bad = 0
    n_scored = 0
    # Layer 6 (speaker_offscreen): count samples where the segmenter
    # picked a specific speaker_slot but that slot has no face in the
    # dense track at this timestamp, while some other face IS visible.
    # This is the "we cropped to the empty seat" failure mode — the
    # generic _face_at lookup hides it because it picks whatever face
    # happens to be largest, even if it's a different speaker.
    chosen_slot_invisible_frames = 0
    chosen_slot_present_frames = 0

    for i in range(n_steps + 1):
        t = t0 + i * step
        op = _op_at(plan, t)
        if op is None:
            continue
        rect = _rect_at(op, t)
        if rect is None:
            continue
        face = _face_at(dense_faces, t)
        # Layer 6: chosen-speaker visibility, evaluated whenever the op
        # has a speaker_slot set, regardless of whether ANY face was
        # found by _face_at. Tracks invisible-vs-present for the slot
        # the segmenter intended to frame.
        chosen_slot = getattr(op, "speaker_slot", None)
        if chosen_slot is not None:
            slot_face = _slot_face_at(dense_faces, t, chosen_slot)
            if slot_face is None and face is not None:
                chosen_slot_invisible_frames += 1
            elif slot_face is not None:
                chosen_slot_present_frames += 1
        if face is None:
            continue
        n_scored += 1

        # nose_x is in percent [0, 100].
        face_x = float(face.nose_x) / 100.0
        face_y = float(face.nose_y) / 100.0
        face_w = float(face.width) / 100.0
        face_h = float(face.height) / 100.0
        face_top = face_y - face_h * 0.5
        face_bot = face_y + face_h * 0.5
        face_left = face_x - face_w * 0.5
        face_right = face_x + face_w * 0.5

        # Crop rect in source-frame fractions.
        crop_left = rect.x
        crop_right = rect.x + rect.w
        crop_top = rect.y
        crop_bot = rect.y + rect.h
        crop_cx = rect.x + rect.w * 0.5
        crop_cy = rect.y + rect.h * 0.5
        cx_samples.append(crop_cx)

        # 1. Subject centering: face_x relative to crop center.
        centering_err += abs(face_x - crop_cx)
        # 2. Headroom: face_top position inside the crop as fraction.
        #    Split into reachable-but-bad (true solver miss) vs.
        #    unreachable (geometrically impossible given the source
        #    aspect / face position — no cy satisfies the constraint).
        if rect.h > 0:
            face_top_in_crop = (face_top - crop_top) / rect.h
            if not (config.headroom_min <= face_top_in_crop <= config.headroom_max):
                # Feasible crop_top interval is [0, 1 - rect.h]. The
                # headroom window requires crop_top in
                # [face_top - headroom_max*h, face_top - headroom_min*h].
                # If those intervals don't overlap, no solver choice
                # could have landed inside the window.
                need_lo = face_top - config.headroom_max * rect.h
                need_hi = face_top - config.headroom_min * rect.h
                feasible_hi = 1.0 - rect.h
                reachable = (need_lo <= feasible_hi) and (need_hi >= 0.0)
                if reachable:
                    headroom_bad += 1
                else:
                    headroom_unreachable += 1
        # 3. Chin clip: face_bot extends below crop_bot.
        if face_bot > crop_bot - 0.02:
            chin_clip_frames += 1
        # 4. Side clip: face extends past either crop edge.
        if face_left < crop_left - 0.02 or face_right > crop_right + 0.02:
            side_clip_bad += 1

        # Layer 1: saliency-in-crop. Check whether any top-K peak at
        # the current second lies inside the crop. We sample at the
        # 10 Hz grid above but only count peaks once per crop sample
        # — a "hit" is when any peak for that second lands in-crop.
        sal_hit = _saliency_in_crop_at(
            saliency_peaks_per_second, t, crop_left, crop_right,
        )
        if sal_hit is not None:
            saliency_in_crop_total += 1
            if sal_hit:
                saliency_in_crop_hits += 1

    # 5. Jitter: stddev of cx changes across the sampled cropping.
    jitter_std = 0.0
    if len(cx_samples) >= 3:
        deltas = [
            cx_samples[i + 1] - cx_samples[i]
            for i in range(len(cx_samples) - 1)
        ]
        mean_d = sum(deltas) / len(deltas)
        var = sum((d - mean_d) ** 2 for d in deltas) / len(deltas)
        jitter_std = var ** 0.5

    # Aggregate score 0..10.
    score = 10.0
    n_scored_safe = max(1, n_scored)

    avg_center_err = centering_err / n_scored_safe
    if avg_center_err > 0.10:
        penalty = min(3.0, (avg_center_err - 0.10) * 30.0)
        score -= penalty
        if penalty > 0.5:
            reasons.append("off_center")
    metrics["avg_center_err"] = round(avg_center_err, 4)

    headroom_bad_frac = headroom_bad / n_scored_safe
    if headroom_bad_frac > 0.30:
        penalty = min(2.5, headroom_bad_frac * 3.0)
        score -= penalty
        if penalty > 0.5:
            if headroom_bad_frac > 0.5:
                reasons.append("head_clip")
    metrics["headroom_bad_frac"] = round(headroom_bad_frac, 2)

    headroom_unreachable_frac = headroom_unreachable / n_scored_safe
    if headroom_unreachable_frac > 0.5:
        # Flag separately but do NOT penalize — solver had no choice.
        reasons.append("head_clip_unreachable")
    metrics["headroom_unreachable_frac"] = round(headroom_unreachable_frac, 2)

    chin_frac = chin_clip_frames / n_scored_safe
    if chin_frac > 0.10:
        # Hard fail: more than 10% of the window chin-clips.
        score = 0.0
        reasons.append("chin_clip")
    metrics["chin_frac"] = round(chin_frac, 2)

    side_frac = side_clip_bad / n_scored_safe
    if side_frac > 0.20:
        penalty = min(1.5, side_frac * 2.0)
        score -= penalty
        if penalty > 0.5:
            reasons.append("side_clip")
    metrics["side_frac"] = round(side_frac, 2)

    if jitter_std > 0.02:
        penalty = min(1.0, (jitter_std - 0.02) * 20.0)
        score -= penalty
        if penalty > 0.3:
            reasons.append("jitter_x")
    metrics["jitter_std"] = round(jitter_std, 4)

    # Missing subject: window had 0 scored frames.
    if n_scored == 0:
        score = 2.0
        reasons.append("subject_missing")

    # Layer 6: chosen-speaker visibility. Acts only when the op declared
    # a speaker_slot AND at least one sample resolved a face — otherwise
    # subject_missing already covers the case. A high invisible fraction
    # means the segmenter targeted a slot that is not on camera while
    # the panel cut to a reaction shot.
    chosen_total = chosen_slot_invisible_frames + chosen_slot_present_frames
    if chosen_total >= 3:
        chosen_invisible_frac = chosen_slot_invisible_frames / chosen_total
        metrics["chosen_invisible_frac"] = round(chosen_invisible_frac, 2)
        if chosen_invisible_frac > 0.50:
            score = min(score, 3.0)
            reasons.append("speaker_offscreen")

    # Layer 1: saliency-in-crop summary + flag. Only acts when we
    # actually saw saliency data for ≥3 sample frames in the window
    # (otherwise the signal is too noisy to penalize on).
    if saliency_in_crop_total >= 3:
        sal_in_crop_frac = saliency_in_crop_hits / saliency_in_crop_total
        metrics["saliency_in_crop_frac"] = round(sal_in_crop_frac, 3)
        if sal_in_crop_frac < 0.5:
            # 2-point penalty out of 10. Caps at 2.
            score -= 2.0
            reasons.append("saliency_excluded")

    return CriticScore(
        t_start=t0, t_end=t1,
        score=max(0.0, min(10.0, score)),
        reasons=reasons, metrics=metrics,
    )


def score_plan(
    plan: RenderPlan,
    *,
    dense_faces: list,
    config: Optional[ReframeConfig] = None,
    window_sec: float = 1.0,
    saliency_peaks_per_second: Optional[list] = None,
) -> list[CriticScore]:
    """Score every ``window_sec``-long window of the plan.

    Returns all windows sorted by score ascending, so the lowest-
    scoring windows come first — useful for budgeted auto-repair.

    ``saliency_peaks_per_second`` (Layer 1) is an optional list whose
    s-th entry is a list of ``(x_pct, y_pct, score)`` peak triples
    for second ``s``. When present, each window picks up a
    ``saliency_in_crop_frac`` metric and a ``saliency_excluded``
    reason when <50% of sampled crops contained any peak. When
    absent, the saliency check is silently skipped (legacy behavior).
    """
    if config is None:
        config = get_default_config()
    duration = plan.total_duration_sec
    if duration <= 0 or not plan.ops:
        return []
    scores: list[CriticScore] = []
    t = 0.0
    while t < duration:
        t_end = min(duration, t + window_sec)
        scores.append(_score_window(
            t, t_end, plan,
            dense_faces=dense_faces, config=config,
            saliency_peaks_per_second=saliency_peaks_per_second,
        ))
        t = t_end
    scores.sort(key=lambda s: s.score)
    return scores


# ── Auto-repair ──────────────────────────────────────────────────


@dataclass
class WindowFix:
    t_start: float
    t_end: float
    reason: str
    new_op: RenderOp


def attempt_window_fix(
    score: CriticScore,
    plan: RenderPlan,
    *,
    dense_faces: list,
    config: ReframeConfig,
) -> Optional[WindowFix]:
    """Try a deterministic local repair for a low-score window.

    Returns None if no tractable fix applies; the caller should fall
    through to the next-lowest window or leave the plan as-is.
    """
    op = _op_at(plan, score.t_start)
    if op is None:
        return None

    if "chin_clip" in score.reasons:
        # Chin clipping means the face's chin (bottom) extends below
        # crop_bot. Shifting the crop DOWN (larger y) extends crop_bot
        # lower in source and reveals the chin. Shift by face_bot -
        # crop_bot + 3% margin; fall back to 10% of crop height.
        shift = op.primary_rect.h * 0.10
        faces_in_window = []
        t = score.t_start
        while t < score.t_end:
            f = _face_at(dense_faces, t)
            if f is not None:
                faces_in_window.append(f)
            t += 0.1
        if faces_in_window:
            # Maximum face_bot in the window vs current crop_bot.
            crop_bot = op.primary_rect.y + op.primary_rect.h
            max_face_bot = max(
                (float(f.nose_y) + float(f.height) * 0.5) / 100.0
                for f in faces_in_window
            )
            needed = max_face_bot - crop_bot + 0.03
            if needed > 0:
                shift = min(
                    1.0 - (op.primary_rect.y + op.primary_rect.h),
                    needed,
                )
        new_y = min(
            1.0 - op.primary_rect.h,
            op.primary_rect.y + max(0.0, shift),
        )
        new_rect = Rect(
            x=op.primary_rect.x, y=new_y,
            w=op.primary_rect.w, h=op.primary_rect.h,
        )
        new_op = _clone_op(op, start=score.t_start, end=score.t_end,
                           rect=new_rect, reason="critic:chin_clip_shift")
        new_op.motion_path = None
        return WindowFix(
            t_start=score.t_start, t_end=score.t_end,
            reason="chin_clip_shift", new_op=new_op,
        )

    if "head_clip" in score.reasons:
        # Head clip: face top above crop top. Shift UP (smaller y).
        new_rect = Rect(
            x=op.primary_rect.x,
            y=max(0.0, op.primary_rect.y - op.primary_rect.h * 0.10),
            w=op.primary_rect.w,
            h=op.primary_rect.h,
        )
        new_op = _clone_op(op, start=score.t_start, end=score.t_end,
                           rect=new_rect, reason="critic:head_clip_shift")
        new_op.motion_path = None
        return WindowFix(
            t_start=score.t_start, t_end=score.t_end,
            reason="head_clip_shift", new_op=new_op,
        )

    if "speaker_offscreen" in score.reasons:
        # The segmenter's chosen speaker is not on camera during this
        # window. Two reasonable repairs in priority order:
        #   1. If a single visible slot dominates ≥60% of the window,
        #      retarget the crop onto that visible speaker (matches the
        #      "panel cut to reaction shot" editorial intent).
        #   2. Otherwise, wide_master so the audience at least sees
        #      whoever happens to be on camera.
        from collections import Counter
        visible_faces: list = []
        t = score.t_start
        while t < score.t_end:
            f = _face_at(dense_faces, t)
            if f is not None:
                visible_faces.append(f)
            t += 0.1
        slot_counts: Counter = Counter()
        for f in visible_faces:
            sid = int(getattr(f, "identity_id", -1))
            if sid >= 0:
                slot_counts[sid] += 1
        total = sum(slot_counts.values())
        dominant = slot_counts.most_common(1)[0] if slot_counts else None
        if dominant and total > 0 and dominant[1] / total >= 0.60:
            dominant_slot, _ = dominant
            new_slot_faces = [
                f for f in visible_faces
                if int(getattr(f, "identity_id", -1)) == dominant_slot
            ]
            mean_x = sum(
                float(f.nose_x) / 100.0 for f in new_slot_faces
            ) / len(new_slot_faces)
            new_x = max(0.0, min(
                1.0 - op.primary_rect.w,
                mean_x - op.primary_rect.w * 0.5,
            ))
            new_rect = Rect(
                x=new_x, y=op.primary_rect.y,
                w=op.primary_rect.w, h=op.primary_rect.h,
            )
            new_op = _clone_op(
                op, start=score.t_start, end=score.t_end,
                rect=new_rect,
                reason="critic:speaker_offscreen_retarget",
            )
            new_op.motion_path = None
            new_op.speaker_slot = dominant_slot
            return WindowFix(
                t_start=score.t_start, t_end=score.t_end,
                reason="speaker_offscreen_retarget", new_op=new_op,
            )
        new_op = RenderOp(
            kind=RenderOpKind.WIDE_MASTER,
            start_sec=score.t_start, end_sec=score.t_end,
            primary_rect=Rect(x=0.0, y=0.0, w=1.0, h=1.0),
            strategy_label="critic:speaker_offscreen:wide",
        )
        return WindowFix(
            t_start=score.t_start, t_end=score.t_end,
            reason="speaker_offscreen_wide", new_op=new_op,
        )

    if "subject_missing" in score.reasons:
        new_op = RenderOp(
            kind=RenderOpKind.WIDE_MASTER,
            start_sec=score.t_start, end_sec=score.t_end,
            primary_rect=Rect(x=0.0, y=0.0, w=1.0, h=1.0),
            strategy_label="critic:subject_missing:wide",
        )
        return WindowFix(
            t_start=score.t_start, t_end=score.t_end,
            reason="subject_missing_wide", new_op=new_op,
        )

    if "jitter_x" in score.reasons:
        # Replace with a STATIONARY op centered on the window-mean cx.
        mean_cx = score.metrics.get("mean_cx") or (
            op.primary_rect.x + op.primary_rect.w * 0.5
        )
        new_rect = Rect(
            x=max(0.0, min(1.0 - op.primary_rect.w,
                           mean_cx - op.primary_rect.w * 0.5)),
            y=op.primary_rect.y,
            w=op.primary_rect.w,
            h=op.primary_rect.h,
        )
        new_op = _clone_op(op, start=score.t_start, end=score.t_end,
                           rect=new_rect, reason="critic:jitter_damp")
        new_op.motion_path = None  # drop the jittery path
        return WindowFix(
            t_start=score.t_start, t_end=score.t_end,
            reason="jitter_damp", new_op=new_op,
        )

    if "saliency_excluded" in score.reasons:
        # Layer 1 fix: saliency hot-spots fell outside the crop. Widen
        # the crop to cover the full source frame for this window so
        # whichever peak the viewer's eye lands on is at least visible.
        # Wider framing trades subject-prominence for coverage — the
        # right call when the alternative is excluding the salient
        # region entirely.
        new_op = RenderOp(
            kind=RenderOpKind.WIDE_MASTER,
            start_sec=score.t_start, end_sec=score.t_end,
            primary_rect=Rect(x=0.0, y=0.0, w=1.0, h=1.0),
            strategy_label="critic:saliency_excluded:wide",
        )
        return WindowFix(
            t_start=score.t_start, t_end=score.t_end,
            reason="saliency_widen", new_op=new_op,
        )

    if "off_center" in score.reasons:
        # Recenter the crop on the face mean over the window.
        face_xs = []
        t = score.t_start
        while t < score.t_end:
            f = _face_at(dense_faces, t)
            if f is not None:
                face_xs.append(float(f.nose_x) / 100.0)
            t += 0.1
        if not face_xs:
            return None
        target_cx = sum(face_xs) / len(face_xs)
        new_rect = Rect(
            x=max(0.0, min(1.0 - op.primary_rect.w,
                           target_cx - op.primary_rect.w * 0.5)),
            y=op.primary_rect.y,
            w=op.primary_rect.w,
            h=op.primary_rect.h,
        )
        new_op = _clone_op(op, start=score.t_start, end=score.t_end,
                           rect=new_rect, reason="critic:recenter")
        new_op.motion_path = None
        return WindowFix(
            t_start=score.t_start, t_end=score.t_end,
            reason="recenter", new_op=new_op,
        )

    return None


def _clone_op(src: RenderOp, *, start: float, end: float,
              rect: Rect, reason: str) -> RenderOp:
    """Clone src with a new time window + primary_rect + strategy."""
    return RenderOp(
        kind=src.kind,
        start_sec=start, end_sec=end,
        primary_rect=rect,
        motion_path=src.motion_path,
        ease_in_ms=src.ease_in_ms,
        strategy_label=(src.strategy_label or "") + f"|{reason}",
        content_type=src.content_type,
        gaming_layout_mode=src.gaming_layout_mode,
        speaker_slot=src.speaker_slot,
        speaker_label=src.speaker_label,
        secondary_rect=src.secondary_rect,
        tertiary_rect=src.tertiary_rect,
        quaternary_rect=src.quaternary_rect,
    )


def apply_window_fix(plan: RenderPlan, fix: WindowFix) -> RenderPlan:
    """Splice ``fix.new_op`` into the plan over ``[t_start, t_end]``.

    The source op covering the window is trimmed on either side so
    the new op slots in without gaps. All other ops are kept as-is.
    """
    new_ops: list[RenderOp] = []
    for op in plan.ops:
        if op.end_sec <= fix.t_start or op.start_sec >= fix.t_end:
            new_ops.append(op)
            continue
        # This op overlaps the fix window.
        if op.start_sec < fix.t_start:
            # Preceding trimmed segment.
            new_ops.append(_clone_op(
                op, start=op.start_sec, end=fix.t_start,
                rect=op.primary_rect, reason="critic-trimmed-head",
            ))
        new_ops.append(fix.new_op)
        if op.end_sec > fix.t_end:
            new_ops.append(_clone_op(
                op, start=fix.t_end, end=op.end_sec,
                rect=op.primary_rect, reason="critic-trimmed-tail",
            ))
    # De-dup when multiple source ops covered the window.
    deduped: list[RenderOp] = []
    seen_fix = False
    for op in new_ops:
        if op is fix.new_op:
            if seen_fix:
                continue
            seen_fix = True
        deduped.append(op)
    return RenderPlan(
        source_width=plan.source_width,
        source_height=plan.source_height,
        target_width=plan.target_width,
        target_height=plan.target_height,
        total_duration_sec=plan.total_duration_sec,
        fps=plan.fps,
        ops=deduped,
        source_offset_sec=getattr(plan, "source_offset_sec", 0.0),
    )


def auto_repair_plan(
    plan: RenderPlan,
    *,
    dense_faces: list,
    config: Optional[ReframeConfig] = None,
    saliency_peaks_per_second: Optional[list] = None,
) -> tuple[RenderPlan, list[WindowFix], list[CriticScore]]:
    """Run the critic, apply fixes up to the config budget, return
    ``(repaired_plan, applied_fixes, final_scores)``.

    Oscillation guard: a window is never repaired twice in one pass.

    ``saliency_peaks_per_second`` (Layer 1) is forwarded to
    :func:`score_plan`. When provided, low-saliency-in-crop windows
    flag as ``saliency_excluded`` and become candidates for the
    wide-framing repair.
    """
    config = config or get_default_config()
    scores = score_plan(
        plan, dense_faces=dense_faces, config=config,
        saliency_peaks_per_second=saliency_peaks_per_second,
    )
    low = [s for s in scores if s.score < config.critic_threshold]
    budget = int(config.critic_budget_per_clip)
    applied: list[WindowFix] = []
    repaired_plan = plan
    repaired_windows: set[tuple[float, float]] = set()
    for s in low[:budget]:
        key = (round(s.t_start, 3), round(s.t_end, 3))
        if key in repaired_windows:
            continue
        fix = attempt_window_fix(
            s, repaired_plan, dense_faces=dense_faces, config=config,
        )
        if fix is None:
            continue
        repaired_plan = apply_window_fix(repaired_plan, fix)
        applied.append(fix)
        repaired_windows.add(key)
    final = score_plan(
        repaired_plan, dense_faces=dense_faces, config=config,
        saliency_peaks_per_second=saliency_peaks_per_second,
    )
    return repaired_plan, applied, final
