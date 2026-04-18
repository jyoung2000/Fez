"""Human-parity metrics for reframing.

These metrics replace the AutoFlip-parity gate. Each metric compares our
rendered 9:16 trajectory (or equivalent RenderPlan crop rects) against a
human-edited vertical reference trajectory recovered from actual vertical
releases (Triller Verzuz, NBA verticals, MJ / Chris Brown vertical cuts,
etc).

The reference trajectory is a list of ``(t, cx_frac, cy_frac, crop_w_frac,
crop_h_frac)`` 5-tuples in source-frame normalized coordinates. See
``backend/scripts/extract_human_trajectories.py`` for how these are
recovered from a (source, human-vertical) clip pair via homography.

Five metrics are produced per clip:

1. **Crop-center MAE (x, y)** — mean absolute error of the crop center in
   output-frame fractions. Lower is better. Fractions of output width so
   the scale is resolution-independent.
2. **Framing-type F1** — per-frame one-of {CU, MS, WS, 2SHOT, SPLIT}
   classification (from the normalized crop width and subject count) and
   the macro-F1 vs human.
3. **Cut-timing deviation (ms)** — signed ms between our re-frames /
   shot-boundary handoffs and the human's. A human J-cut ~100 ms before
   the audio turn is the reference.
4. **Lead-room correlation** — Pearson's r between our ``(crop_center -
   subject_center)`` offset and the human's, frame-by-frame. Captures
   whether we're leading the subject in the same direction.
5. **Aesthetic score** — optional mean VLM / learned-scorer score over
   sampled output frames (0-10). Populated by
   ``backend/services/critic_loop.py``; passes through as ``None`` when
   the critic is disabled.

All metrics are emitted as a single ``HumanParityReport`` dataclass that
serializes to JSON. ``validate_human_parity.py`` reads the JSON back and
diffs against a pinned baseline.

This module has no runtime dependency on a GPU, a VLM, or the full
pipeline — it operates purely on crop trajectories + optional face boxes
so it can be invoked both from the pipeline at export time and from
offline bench runs.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Framing-type classification ──────────────────────────────────
#
# Maps a normalized 9:16 crop width (as a fraction of source_width) plus
# the number of distinct subjects visible inside the crop at a given
# timestamp into a categorical framing label. The cut points come from
# standard cinematography practice:
#
#   Close-up (CU):        crop_w_frac < 0.25   (face spans ≥ 40% of crop)
#   Medium shot (MS):     0.25 ≤ w < 0.45      (head + shoulders)
#   Wide shot (WS):       0.45 ≤ w < 0.80      (body + space)
#   2-shot:               2 subjects in crop, any width
#   Split:                2 subjects rendered via SPLIT_SCREEN op
#
# The 2-shot and split labels override width-based classification because
# they're structurally different compositions.

FRAMING_CU = "CU"
FRAMING_MS = "MS"
FRAMING_WS = "WS"
FRAMING_2SHOT = "2SHOT"
FRAMING_SPLIT = "SPLIT"

FRAMING_LABELS = (FRAMING_CU, FRAMING_MS, FRAMING_WS, FRAMING_2SHOT, FRAMING_SPLIT)


def classify_framing(
    crop_w_frac: float,
    *,
    n_subjects_in_crop: int = 1,
    is_split_screen: bool = False,
) -> str:
    """Classify a single frame into one of the five framing labels."""
    if is_split_screen:
        return FRAMING_SPLIT
    if n_subjects_in_crop >= 2:
        return FRAMING_2SHOT
    if crop_w_frac < 0.25:
        return FRAMING_CU
    if crop_w_frac < 0.45:
        return FRAMING_MS
    return FRAMING_WS


# ── Dataclasses ─────────────────────────────────────────────────────


@dataclass
class TrajectorySample:
    """One row of a crop trajectory, normalized to source-frame fractions.

    All coordinates are floats in [0, 1]; the crop center is ``(cx, cy)``
    and the crop size is ``(w, h)``. Out-of-range values will be clamped
    by the caller before metrics are computed.
    """

    t: float
    cx: float
    cy: float
    w: float
    h: float
    # Optional per-frame subject center(s) in source-frame fractions.
    # When provided we compute lead-room (cx - subj_cx, cy - subj_cy).
    subject_cx: Optional[float] = None
    subject_cy: Optional[float] = None
    n_subjects_in_crop: int = 1
    is_split_screen: bool = False


@dataclass
class HumanParityReport:
    """One clip's worth of human-parity metrics."""

    clip_slug: str
    content_type: str
    n_frames: int

    # Crop-center MAE, output-frame fractions (1.0 = full output width / height)
    mae_cx: float = 0.0
    mae_cy: float = 0.0

    # Framing-type agreement
    framing_macro_f1: float = 0.0
    framing_per_label_f1: dict = field(default_factory=dict)

    # Cut-timing deviation
    cut_timing_deviation_ms_mean: float = 0.0
    cut_timing_deviation_ms_p90: float = 0.0
    n_cut_pairs: int = 0

    # Lead-room correlation (Pearson's r, -1..1)
    lead_room_correlation_x: float = 0.0
    lead_room_correlation_y: float = 0.0

    # Aesthetic score (0-10), optional
    aesthetic_score_mean: Optional[float] = None
    aesthetic_score_samples: int = 0

    # Implementation tag so diffs between bench versions are visible
    implementation_version: str = "human_parity_v1"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)


# ── Resampling + MAE ────────────────────────────────────────────────


def _resample_to_grid(
    traj: list[TrajectorySample],
    grid_ts: list[float],
) -> list[TrajectorySample]:
    """Piecewise-linear resample a trajectory onto the given timestamps.

    The input trajectory is assumed sorted by ``t`` and non-empty; the
    grid timestamps are clamped into the trajectory's span. Points outside
    the span are held at the nearest endpoint (no extrapolation).
    """
    if not traj:
        return []
    if len(traj) == 1:
        only = traj[0]
        return [
            TrajectorySample(
                t=ts,
                cx=only.cx, cy=only.cy, w=only.w, h=only.h,
                subject_cx=only.subject_cx, subject_cy=only.subject_cy,
                n_subjects_in_crop=only.n_subjects_in_crop,
                is_split_screen=only.is_split_screen,
            )
            for ts in grid_ts
        ]

    result: list[TrajectorySample] = []
    j = 0
    for ts in grid_ts:
        # Advance j so traj[j].t <= ts <= traj[j+1].t (clamped at edges).
        while j + 1 < len(traj) and traj[j + 1].t < ts:
            j += 1
        if ts <= traj[0].t:
            a = traj[0]
            result.append(TrajectorySample(
                t=ts, cx=a.cx, cy=a.cy, w=a.w, h=a.h,
                subject_cx=a.subject_cx, subject_cy=a.subject_cy,
                n_subjects_in_crop=a.n_subjects_in_crop,
                is_split_screen=a.is_split_screen,
            ))
            continue
        if ts >= traj[-1].t:
            a = traj[-1]
            result.append(TrajectorySample(
                t=ts, cx=a.cx, cy=a.cy, w=a.w, h=a.h,
                subject_cx=a.subject_cx, subject_cy=a.subject_cy,
                n_subjects_in_crop=a.n_subjects_in_crop,
                is_split_screen=a.is_split_screen,
            ))
            continue
        a = traj[j]
        b = traj[min(j + 1, len(traj) - 1)]
        dt = b.t - a.t
        u = 0.0 if dt <= 1e-9 else (ts - a.t) / dt
        u = max(0.0, min(1.0, u))

        def _lerp(va, vb) -> float:
            if va is None or vb is None:
                return va if va is not None else (vb if vb is not None else 0.0)
            return (1.0 - u) * float(va) + u * float(vb)

        result.append(TrajectorySample(
            t=ts,
            cx=_lerp(a.cx, b.cx),
            cy=_lerp(a.cy, b.cy),
            w=_lerp(a.w, b.w),
            h=_lerp(a.h, b.h),
            subject_cx=(
                None if a.subject_cx is None and b.subject_cx is None
                else _lerp(a.subject_cx, b.subject_cx)
            ),
            subject_cy=(
                None if a.subject_cy is None and b.subject_cy is None
                else _lerp(a.subject_cy, b.subject_cy)
            ),
            # nearest-neighbor for the categorical fields
            n_subjects_in_crop=a.n_subjects_in_crop if u < 0.5 else b.n_subjects_in_crop,
            is_split_screen=a.is_split_screen if u < 0.5 else b.is_split_screen,
        ))
    return result


def _mae_crop_center(
    ours: list[TrajectorySample],
    human: list[TrajectorySample],
) -> tuple[float, float]:
    """Return (mae_cx, mae_cy) in output-frame fractions."""
    if not ours or not human:
        return (0.0, 0.0)
    n = min(len(ours), len(human))
    ex = sum(abs(ours[i].cx - human[i].cx) for i in range(n)) / max(n, 1)
    ey = sum(abs(ours[i].cy - human[i].cy) for i in range(n)) / max(n, 1)
    return (ex, ey)


# ── Framing-type F1 ─────────────────────────────────────────────────


def _f1_from_confusion(tp: int, fp: int, fn: int) -> float:
    if tp + fp + fn == 0:
        return 1.0  # undefined → treat as match
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    if prec + rec <= 0.0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def _framing_f1(
    ours: list[TrajectorySample],
    human: list[TrajectorySample],
) -> tuple[float, dict]:
    n = min(len(ours), len(human))
    per_label: dict = {lbl: {"tp": 0, "fp": 0, "fn": 0} for lbl in FRAMING_LABELS}
    for i in range(n):
        a = classify_framing(
            ours[i].w,
            n_subjects_in_crop=ours[i].n_subjects_in_crop,
            is_split_screen=ours[i].is_split_screen,
        )
        b = classify_framing(
            human[i].w,
            n_subjects_in_crop=human[i].n_subjects_in_crop,
            is_split_screen=human[i].is_split_screen,
        )
        for lbl in FRAMING_LABELS:
            if a == lbl and b == lbl:
                per_label[lbl]["tp"] += 1
            elif a == lbl and b != lbl:
                per_label[lbl]["fp"] += 1
            elif a != lbl and b == lbl:
                per_label[lbl]["fn"] += 1
    per_f1 = {
        lbl: _f1_from_confusion(
            per_label[lbl]["tp"], per_label[lbl]["fp"], per_label[lbl]["fn"],
        )
        for lbl in FRAMING_LABELS
    }
    macro = sum(per_f1.values()) / len(FRAMING_LABELS)
    return macro, per_f1


# ── Cut-timing deviation ────────────────────────────────────────────


def _detect_cuts(traj: list[TrajectorySample], *, min_delta: float = 0.05) -> list[float]:
    """Return timestamps where the crop center jumps by > ``min_delta``
    in a single frame-to-frame step. These are our proxy for editorial
    cuts / saccades.
    """
    out: list[float] = []
    for i in range(1, len(traj)):
        dx = traj[i].cx - traj[i - 1].cx
        dy = traj[i].cy - traj[i - 1].cy
        if math.hypot(dx, dy) > min_delta:
            out.append(traj[i].t)
    return out


def _cut_timing_deviation(
    ours: list[TrajectorySample],
    human: list[TrajectorySample],
    *,
    match_window_ms: float = 500.0,
) -> tuple[float, float, int]:
    """For each human cut, find the nearest ours cut within the match
    window and record the signed offset in ms (negative = we cut earlier).
    Returns ``(mean_abs_ms, p90_abs_ms, n_pairs)``.
    """
    ours_cuts = _detect_cuts(ours)
    human_cuts = _detect_cuts(human)
    if not human_cuts:
        return (0.0, 0.0, 0)
    deltas: list[float] = []
    window = match_window_ms / 1000.0
    for ht in human_cuts:
        best = None
        for ot in ours_cuts:
            diff = ot - ht
            if abs(diff) <= window:
                if best is None or abs(diff) < abs(best):
                    best = diff
        if best is not None:
            deltas.append(best * 1000.0)
    if not deltas:
        return (0.0, 0.0, 0)
    abs_deltas = sorted(abs(d) for d in deltas)
    mean_abs = sum(abs_deltas) / len(abs_deltas)
    p90_idx = max(0, int(0.9 * len(abs_deltas)) - 1)
    return (mean_abs, abs_deltas[p90_idx], len(deltas))


# ── Lead-room correlation ──────────────────────────────────────────


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    mx = sum(xs[:n]) / n
    my = sum(ys[:n]) / n
    num = 0.0
    sx = 0.0
    sy = 0.0
    for i in range(n):
        dx = xs[i] - mx
        dy = ys[i] - my
        num += dx * dy
        sx += dx * dx
        sy += dy * dy
    if sx <= 0.0 or sy <= 0.0:
        return 0.0
    return num / math.sqrt(sx * sy)


def _lead_room_corr(
    ours: list[TrajectorySample],
    human: list[TrajectorySample],
) -> tuple[float, float]:
    ox: list[float] = []
    hx: list[float] = []
    oy: list[float] = []
    hy: list[float] = []
    n = min(len(ours), len(human))
    for i in range(n):
        if ours[i].subject_cx is None or human[i].subject_cx is None:
            continue
        ox.append(ours[i].cx - float(ours[i].subject_cx))
        hx.append(human[i].cx - float(human[i].subject_cx))
        if ours[i].subject_cy is not None and human[i].subject_cy is not None:
            oy.append(ours[i].cy - float(ours[i].subject_cy))
            hy.append(human[i].cy - float(human[i].subject_cy))
    return _pearson(ox, hx), _pearson(oy, hy)


# ── Public API ─────────────────────────────────────────────────────


def compute_human_parity(
    ours: list[TrajectorySample],
    human: list[TrajectorySample],
    *,
    clip_slug: str,
    content_type: str,
    grid_dt_sec: float = 1.0 / 30.0,
    aesthetic_score_mean: Optional[float] = None,
    aesthetic_score_samples: int = 0,
) -> HumanParityReport:
    """Compute the full human-parity report for a single clip.

    ``ours`` and ``human`` can be sampled at different rates / non-uniform
    grids. They are both resampled onto a common uniform grid at
    ``grid_dt_sec`` spanning the overlap of their time ranges before any
    metrics are computed, so the resulting MAE / correlation are rate-
    invariant.

    Aesthetic score is optional and passes through when populated by
    ``backend/services/critic_loop.py`` on a full pipeline run.
    """
    if not ours or not human:
        return HumanParityReport(
            clip_slug=clip_slug,
            content_type=content_type,
            n_frames=0,
        )

    ours_sorted = sorted(ours, key=lambda s: s.t)
    human_sorted = sorted(human, key=lambda s: s.t)
    t0 = max(ours_sorted[0].t, human_sorted[0].t)
    t1 = min(ours_sorted[-1].t, human_sorted[-1].t)
    if t1 <= t0:
        return HumanParityReport(
            clip_slug=clip_slug,
            content_type=content_type,
            n_frames=0,
        )

    n_grid = int(math.floor((t1 - t0) / grid_dt_sec)) + 1
    grid = [t0 + i * grid_dt_sec for i in range(n_grid)]
    o_g = _resample_to_grid(ours_sorted, grid)
    h_g = _resample_to_grid(human_sorted, grid)

    mae_cx, mae_cy = _mae_crop_center(o_g, h_g)
    macro_f1, per_f1 = _framing_f1(o_g, h_g)
    cut_mean, cut_p90, n_pairs = _cut_timing_deviation(o_g, h_g)
    lr_x, lr_y = _lead_room_corr(o_g, h_g)

    return HumanParityReport(
        clip_slug=clip_slug,
        content_type=content_type,
        n_frames=n_grid,
        mae_cx=mae_cx,
        mae_cy=mae_cy,
        framing_macro_f1=macro_f1,
        framing_per_label_f1=per_f1,
        cut_timing_deviation_ms_mean=cut_mean,
        cut_timing_deviation_ms_p90=cut_p90,
        n_cut_pairs=n_pairs,
        lead_room_correlation_x=lr_x,
        lead_room_correlation_y=lr_y,
        aesthetic_score_mean=aesthetic_score_mean,
        aesthetic_score_samples=aesthetic_score_samples,
    )


def render_plan_to_trajectory(
    render_plan_dict: dict,
    *,
    sample_fps: float = 30.0,
) -> list[TrajectorySample]:
    """Convert a serialized ``RenderPlan`` dict into a sampled trajectory.

    Walks the ``ops`` list, interpolates ``motion_path`` keypoints for
    ``tracking_crop``, and emits one ``TrajectorySample`` per frame at
    ``sample_fps`` for the full plan duration.
    """
    ops = render_plan_dict.get("ops") or []
    duration = float(render_plan_dict.get("total_duration_sec") or 0.0)
    if not ops or duration <= 0.0:
        return []

    dt = 1.0 / max(sample_fps, 1.0)
    out: list[TrajectorySample] = []
    t = 0.0
    op_idx = 0
    while t <= duration + 1e-6 and op_idx < len(ops):
        op = ops[op_idx]
        start = float(op["start_sec"])
        end = float(op["end_sec"])
        if t < start:
            t = start
            continue
        if t > end:
            op_idx += 1
            continue

        kind = op.get("kind", "crop")
        primary = op.get("primary_rect") or {}
        secondary = op.get("secondary_rect") or {}
        is_split = kind in ("split_screen", "stacked_gameplay", "grid_2x2")

        if kind == "tracking_crop" and op.get("motion_path"):
            mp = op["motion_path"]
            t_rel = t - start
            # find surrounding keypoints
            kp_a = mp[0]
            kp_b = mp[-1]
            for i in range(len(mp) - 1):
                if mp[i]["t"] <= t_rel <= mp[i + 1]["t"]:
                    kp_a = mp[i]
                    kp_b = mp[i + 1]
                    break
            span = (kp_b["t"] - kp_a["t"]) or 1e-9
            u = max(0.0, min(1.0, (t_rel - kp_a["t"]) / span))
            ra = kp_a["rect"]
            rb = kp_b["rect"]
            cx = (1 - u) * (ra["x"] + ra["w"] * 0.5) + u * (rb["x"] + rb["w"] * 0.5)
            cy = (1 - u) * (ra["y"] + ra["h"] * 0.5) + u * (rb["y"] + rb["h"] * 0.5)
            w = (1 - u) * ra["w"] + u * rb["w"]
            h = (1 - u) * ra["h"] + u * rb["h"]
        else:
            cx = primary.get("x", 0.0) + primary.get("w", 1.0) * 0.5
            cy = primary.get("y", 0.0) + primary.get("h", 1.0) * 0.5
            w = primary.get("w", 1.0)
            h = primary.get("h", 1.0)

        n_subj = 2 if (kind in ("split_screen",) and secondary) else 1
        out.append(TrajectorySample(
            t=t, cx=cx, cy=cy, w=w, h=h,
            n_subjects_in_crop=n_subj,
            is_split_screen=is_split,
        ))
        t += dt
    return out
