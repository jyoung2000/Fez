"""Saliency soft-prior wiring for the LP camera path.

Phase C of the 2026 SOTA reframing rollout.

The AutoFlip-style LP solver in :mod:`_autoflip_lp` already supports
per-frame target positions and per-frame data-fidelity weights. This
module produces both from the saliency heatmap so the LP can encourage
the crop to contain at least one local gaze-fixation maximum without
ever overriding face / speaker hard constraints.

Two integration points:

    1. ``apply_saliency_nudge`` — given the existing per-frame target
       (``tx``) plus the per-second top-3 saliency peaks, nudges ``tx``
       toward the nearest peak by ``lambda_saliency * peak_score``.
       Soft (default ``lambda_saliency=0.15``) and clamped to the
       per-frame feasibility bounds. Acts AFTER face-targeting but
       BEFORE the LP — so faces win when both are present.

    2. ``soft_promote_peaks_to_required`` — produces a list of
       ``RequiredRegion`` objects with ``tier='preferred'`` that the
       existing required-regions builder can append to the per-frame
       fusion. Lets the LP's required-region machinery score the same
       saliency peaks as soft hints.

Both functions are pure-numpy and unit-testable without sam2/torch
/saliency models — they consume the already-extracted peaks from
:class:`AvSaliency.predict()`.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


DEFAULT_LAMBDA_SALIENCY = 0.15
DEFAULT_PEAK_RADIUS_PCT = 18.0   # how close a peak counts as "in crop"


# ── Nudge: pre-blend into the LP target ──────────────────────────────


def apply_saliency_nudge(
    tx: list,
    timestamps: list,
    saliency_peaks_per_second: list,
    *,
    lambda_saliency: float = DEFAULT_LAMBDA_SALIENCY,
    lo_bounds: Optional[list] = None,
    hi_bounds: Optional[list] = None,
    peak_radius_pct: float = DEFAULT_PEAK_RADIUS_PCT,
) -> list:
    """Return a copy of ``tx`` nudged toward the nearest top-K peak.

    ``tx``: per-frame target X (% of source width).
    ``timestamps``: per-frame timestamp seconds.
    ``saliency_peaks_per_second[s]``: list of ``(x_pct, y_pct, score)``.

    The nudge is only applied when the existing target is OUTSIDE the
    peak's neighbourhood (``peak_radius_pct``). When a face / speaker
    already places the target on a salient peak, no nudge happens.
    Bounds-respecting via ``lo_bounds`` / ``hi_bounds``.
    """
    if not tx or not saliency_peaks_per_second:
        return list(tx)
    if lambda_saliency <= 0:
        return list(tx)
    out = list(tx)
    n = len(out)
    for i in range(n):
        sec_idx = int(timestamps[i]) if i < len(timestamps) else i
        if sec_idx < 0 or sec_idx >= len(saliency_peaks_per_second):
            continue
        peaks = saliency_peaks_per_second[sec_idx]
        if not peaks:
            continue
        # Find the peak nearest the current target.
        best = None
        best_dist = float("inf")
        for px, _py, score in peaks:
            d = abs(px - out[i] * 100.0) if out[i] <= 1.0 else abs(px - out[i])
            if d < best_dist:
                best_dist = d
                best = (px, score)
        if best is None or best_dist <= peak_radius_pct:
            continue
        peak_x_pct, peak_score = best
        # Convert nudge to the same units as ``tx``. ``tx`` may be in
        # fractional [0,1] (camera_path_2d's convention) OR percentage —
        # detect via the existing range.
        if all(0.0 <= v <= 1.0 for v in out[:5]):
            target = peak_x_pct / 100.0
        else:
            target = peak_x_pct
        nudged = (1.0 - lambda_saliency * peak_score) * out[i] + (
            lambda_saliency * peak_score
        ) * target
        if lo_bounds is not None and i < len(lo_bounds):
            nudged = max(lo_bounds[i], nudged)
        if hi_bounds is not None and i < len(hi_bounds):
            nudged = min(hi_bounds[i], nudged)
        out[i] = nudged
    return out


# ── Soft-promote peaks to required regions ──────────────────────────


def soft_promote_peaks_to_required(
    saliency_peaks_per_second: list,
    *,
    timestamps_per_second: Optional[list] = None,
    half_width: float = 0.08,
    half_height: float = 0.08,
    weight: float = 0.4,
):
    """Produce ``preferred``-tier required regions from saliency peaks.

    Returns a list of ``RequiredRegion`` instances suitable for
    appending to the per-frame fusion in
    :func:`required_regions.build_required_regions`. Soft only — the
    existing face / speaker hard constraints are unaffected.

    Coordinates: ``RequiredRegion`` uses normalized [0, 1] coords, so
    we convert from the saliency module's [0, 100] convention.
    """
    try:
        from backend.services.required_regions import RequiredRegion
    except Exception:
        # required_regions has heavier deps — fall back to a tuple
        # representation when its dependency tree isn't installed.
        RequiredRegion = None  # type: ignore
    out: list = []
    for s, peaks in enumerate(saliency_peaks_per_second):
        t = float(timestamps_per_second[s]) if (
            timestamps_per_second is not None and s < len(timestamps_per_second)
        ) else float(s)
        for (x_pct, y_pct, score) in peaks:
            cx = x_pct / 100.0
            cy = y_pct / 100.0
            if RequiredRegion is None:
                out.append({
                    "timestamp": t, "cx": cx, "cy": cy,
                    "half_width": half_width, "half_height": half_height,
                    "score": float(score), "tier": "preferred",
                    "source": "saliency", "weight": weight,
                })
            else:
                out.append(RequiredRegion(
                    timestamp=t, cx=cx, cy=cy,
                    half_width=half_width, half_height=half_height,
                    score=float(score), tier="preferred",
                    source="saliency", weight=weight,
                    saliency_score=float(score),
                ))
    return out


# ── LP weight helper ─────────────────────────────────────────────────


def saliency_data_weight(
    base_weight: float,
    saliency_at_target: float,
    *,
    lambda_saliency: float = DEFAULT_LAMBDA_SALIENCY,
) -> float:
    """Boost the data-fidelity weight when target lies on a salient peak.

    Returns a weight scalar consumable by :func:`solve_autoflip_lp`'s
    ``weights`` kwarg. Frames where the target sits on a high-saliency
    peak get up to ``1 + lambda_saliency`` of weight; off-peak frames
    are unchanged.
    """
    return float(base_weight) * (1.0 + lambda_saliency * float(saliency_at_target))
