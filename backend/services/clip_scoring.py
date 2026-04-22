"""Composite clip scoring — Phase 4 of the OpusClip parity gap.

The LLM no longer returns a single ``viral_score``. Instead it returns
four axis scores (hook / flow / value / trend) and we combine them into
a composite using genre-aware weights. This module owns the math.

The composite is deterministic and unit-tested. ``clip_verifier`` and
the existing hook-strength / retention adjustments still apply on top of
the composite, so the LLM's per-axis judgement remains the dominant
signal but local heuristics can nudge it.

Backward compatibility: when all four axis scores are zero (legacy
clips that never went through the new prompt) we leave ``viral_score``
untouched.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from backend.models import ClipCandidate
from backend.services.content_classifier import ClipContentType

logger = logging.getLogger(__name__)


# Default 4-axis weights when no genre is known. Hook + Value are the
# two strongest predictors of completion / re-watch on short-form
# platforms, with Flow and Trend as supporting signals.
GENRE_WEIGHTS_DEFAULT: dict[str, float] = {
    "hook": 0.30,
    "flow": 0.25,
    "value": 0.30,
    "trend": 0.15,
}


# Per-genre weight overrides. Each row must sum to 1.0 (we assert in
# unit tests). Tuned for the OpusClip parity gap — gameplay / sports
# weight Hook hardest because the first few frames of a clutch play
# decide whether a viewer keeps watching, while talking-head weights
# Value highest because the substance of what's said is what gets
# shared.
GENRE_WEIGHTS: dict[ClipContentType, dict[str, float]] = {
    ClipContentType.TALKING_HEAD: {
        "hook": 0.30, "flow": 0.25, "value": 0.30, "trend": 0.15,
    },
    ClipContentType.MULTI_SPEAKER_PANEL: {
        "hook": 0.25, "flow": 0.30, "value": 0.30, "trend": 0.15,
    },
    ClipContentType.CINEMATIC_DIALOGUE: {
        "hook": 0.25, "flow": 0.35, "value": 0.30, "trend": 0.10,
    },
    ClipContentType.GAMEPLAY: {
        "hook": 0.40, "flow": 0.15, "value": 0.30, "trend": 0.15,
    },
    ClipContentType.GAMEPLAY_MOBA: {
        "hook": 0.40, "flow": 0.15, "value": 0.30, "trend": 0.15,
    },
    ClipContentType.GAMEPLAY_TPS: {
        "hook": 0.40, "flow": 0.15, "value": 0.30, "trend": 0.15,
    },
    ClipContentType.GAMEPLAY_RACING: {
        "hook": 0.40, "flow": 0.15, "value": 0.30, "trend": 0.15,
    },
    ClipContentType.STREAM: {
        "hook": 0.35, "flow": 0.20, "value": 0.30, "trend": 0.15,
    },
    ClipContentType.SPORTS: {
        "hook": 0.45, "flow": 0.10, "value": 0.30, "trend": 0.15,
    },
    ClipContentType.SPORTS_BASKETBALL: {
        "hook": 0.45, "flow": 0.10, "value": 0.30, "trend": 0.15,
    },
    ClipContentType.SPORTS_RACING: {
        "hook": 0.45, "flow": 0.10, "value": 0.30, "trend": 0.15,
    },
    ClipContentType.MUSIC_VIDEO: {
        "hook": 0.40, "flow": 0.20, "value": 0.15, "trend": 0.25,
    },
    ClipContentType.ANIMATION: {
        "hook": 0.35, "flow": 0.25, "value": 0.25, "trend": 0.15,
    },
    ClipContentType.ANIMATION_DIALOGUE: {
        "hook": 0.30, "flow": 0.30, "value": 0.25, "trend": 0.15,
    },
    ClipContentType.GENERIC: GENRE_WEIGHTS_DEFAULT,
}


def _flag_enabled(name: str, default: bool) -> bool:
    """Read a feature flag from the environment.

    Truthy values: 1 / true / yes / on (case-insensitive). Anything else
    is falsy. Allows operators to flip Phase 1/2 behavior without code
    changes.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def four_axis_scoring_enabled() -> bool:
    """Phase 1 feature flag — defaults on per the gap-close brief."""
    return _flag_enabled("USE_FOUR_AXIS_SCORING", True)


def get_weights(content_type: Optional[ClipContentType]) -> dict[str, float]:
    """Return the axis weights to use for a given content type.

    Always returns a dict with keys hook/flow/value/trend that sum to
    ~1.0. Falls back to GENRE_WEIGHTS_DEFAULT for unknown types.
    """
    if content_type is None:
        return GENRE_WEIGHTS_DEFAULT
    return GENRE_WEIGHTS.get(content_type, GENRE_WEIGHTS_DEFAULT)


# Blueprint v2 Phase 3 — re-export the virality-by-confidence helpers
# from the stand-alone ``clip_confidence`` module. They live there so
# tests can exercise them without pulling pydantic / ClipCandidate
# into the import graph.
from backend.services.clip_confidence import (  # noqa: E402,F401
    CLIP_CONFIDENCE_VIRALITY_MULTIPLIER,
    _confidence_multiplier,
    apply_clip_confidence,
)


def composite_score(
    clip: ClipCandidate,
    content_type: Optional[ClipContentType] = None,
) -> int:
    """Compute the composite viral_score from the four axes.

    Returns an int 1-100. When all four axes are 0 (legacy clip from a
    pre-Phase-1 prompt), returns the existing viral_score unchanged so
    we never silently zero out a clip's score. When at least one axis
    is populated, the composite is the genre-weighted average of all
    four axes.
    """
    axes = (
        int(clip.hook_score),
        int(clip.flow_score),
        int(clip.value_score),
        int(clip.trend_score),
    )
    if all(a == 0 for a in axes):
        # Legacy path — nothing to compose. Keep the LLM-supplied score.
        return max(1, min(100, int(clip.viral_score) if clip.viral_score else 50))

    weights = get_weights(content_type)
    raw = (
        axes[0] * weights["hook"]
        + axes[1] * weights["flow"]
        + axes[2] * weights["value"]
        + axes[3] * weights["trend"]
    )
    return max(1, min(100, int(round(raw))))


def finalize_clip_scores(
    clips: list[ClipCandidate],
    content_type: Optional[ClipContentType] = None,
    focus_mode: bool = False,
) -> list[ClipCandidate]:
    """Compute composite viral_score for every clip in-place.

    Also populates ``clip.score_diagnostics`` with the weights used and
    the per-axis breakdown so the job result JSON can be inspected
    after the fact.

    When ``focus_mode`` is True the LLM was instructed to put RELEVANCE
    (not virality) in ``viral_score``; we still compute the genre-weighted
    virality composite, but we store it in ``viral_score_composite`` and
    leave ``viral_score`` = relevance untouched. This lets the UI show
    both numbers without contradicting the focus-mode prompt (see
    Bug 3 in the clip-focus audit).
    """
    if not four_axis_scoring_enabled():
        return clips

    weights = get_weights(content_type)
    for clip in clips:
        # Detect the legacy-fill case so the UI can render axis bars
        # desaturated and show a "estimated from legacy score" tooltip
        # (Bug 12 in the clip-focus audit).
        legacy_fill = all(
            int(getattr(clip, f) or 0) == 0
            for f in ("hook_score", "flow_score", "value_score", "trend_score")
        )
        new_score = composite_score(clip, content_type)
        prev_score = clip.viral_score
        if focus_mode:
            # Preserve LLM-reported relevance in viral_score.
            clip.viral_score_composite = new_score
        else:
            clip.viral_score = new_score
            clip.viral_score_composite = new_score
        diag = dict(clip.score_diagnostics or {})
        diag.update({
            "axis_scores": {
                "hook": int(clip.hook_score),
                "flow": int(clip.flow_score),
                "value": int(clip.value_score),
                "trend": int(clip.trend_score),
            },
            "axis_reasons": {
                "hook": clip.hook_reason,
                "flow": clip.flow_reason,
                "value": clip.value_reason,
                "trend": clip.trend_reason,
            },
            "weights": dict(weights),
            "content_type": content_type.value if content_type else None,
            "composite_before": int(prev_score),
            "composite_after": int(new_score),
            "legacy_fill": bool(legacy_fill),
            "focus_mode": bool(focus_mode),
        })
        clip.score_diagnostics = diag
    return clips


def deduplicate_overlapping_clips(
    clips: list[ClipCandidate],
    iou_threshold: float = 0.6,
) -> tuple[list[ClipCandidate], int]:
    """Drop near-duplicate clips by time IoU.

    For any two clips with time-IoU >= ``iou_threshold`` AND the same
    ``clip_focus`` (None == None counts as same), keep the one with the
    higher sort key (focus_relevance if set, else viral_score) and drop
    the other. Returns (kept_clips, dropped_count).

    Preserves the relative order of kept clips. The QA validator warns
    about overlaps (see backend/routers/clips.py ~1717-1737) but never
    acted on them — this helper is invoked from the regenerate path
    after the score filter. See Bug 10 in the clip-focus audit.
    """
    if not clips:
        return clips, 0

    def _sort_key(c: ClipCandidate) -> int:
        return int(c.focus_relevance if c.focus_relevance is not None else (c.viral_score or 0))

    ordered = sorted(enumerate(clips), key=lambda ic: _sort_key(ic[1]), reverse=True)
    kept_indices: set[int] = set()
    for orig_idx, clip in ordered:
        keep = True
        for kept_idx in kept_indices:
            other = clips[kept_idx]
            if (clip.clip_focus or None) != (other.clip_focus or None):
                continue
            a_start, a_end = clip.start_time, clip.end_time
            b_start, b_end = other.start_time, other.end_time
            inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
            if inter <= 0:
                continue
            union = max(a_end, b_end) - min(a_start, b_start)
            iou = inter / union if union > 0 else 0.0
            if iou >= iou_threshold:
                keep = False
                break
        if keep:
            kept_indices.add(orig_idx)

    kept = [c for i, c in enumerate(clips) if i in kept_indices]
    return kept, len(clips) - len(kept)


def fill_axes_from_legacy(clip: ClipCandidate) -> None:
    """Backfill axis scores from a legacy ``viral_score`` field.

    Used by the focus-query path (``clip_focus``) and by salvage paths
    where the LLM didn't return the four axes. We don't know how to
    decompose the score, so we splat it across all four axes — this
    keeps downstream UI code that reads ``hook_score`` etc. happy
    without overstating the signal.
    """
    score = max(1, min(100, int(clip.viral_score or 0)))
    filled_any = False
    if clip.hook_score == 0:
        clip.hook_score = score
        clip.hook_reason = clip.hook_reason or "derived from legacy viral_score"
        filled_any = True
    if clip.flow_score == 0:
        clip.flow_score = score
        clip.flow_reason = clip.flow_reason or "derived from legacy viral_score"
        filled_any = True
    if clip.value_score == 0:
        clip.value_score = score
        clip.value_reason = clip.value_reason or "derived from legacy viral_score"
        filled_any = True
    if clip.trend_score == 0:
        # Trend defaults to 50 (neutral) when no real signal exists,
        # not the legacy viral_score — overstating trend would let
        # neutral content beat genuinely on-trend clips.
        clip.trend_score = 50
        clip.trend_reason = clip.trend_reason or "no trend signal — neutral default"
        filled_any = True
    if filled_any:
        diag = dict(clip.score_diagnostics or {})
        diag["legacy_fill"] = True
        clip.score_diagnostics = diag
