"""Tests for the 4-axis composite scoring (Phase 1 + 4)."""

import pytest

from backend.models import ClipCandidate
from backend.services.clip_scoring import (
    GENRE_WEIGHTS,
    GENRE_WEIGHTS_DEFAULT,
    composite_score,
    fill_axes_from_legacy,
    finalize_clip_scores,
    get_weights,
)
from backend.services.content_classifier import ClipContentType


def _make_clip(
    hook=0, flow=0, value=0, trend=0, viral=50,
) -> ClipCandidate:
    return ClipCandidate(
        id=1,
        title="Test",
        start_time=0.0,
        end_time=30.0,
        duration=30.0,
        viral_score=viral,
        viral_score_reasoning="",
        clip_type="highlight",
        platform="both",
        suggested_caption="",
        hook_text="",
        why_this_works="",
        hook_score=hook,
        flow_score=flow,
        value_score=value,
        trend_score=trend,
    )


# ── composite_score ────────────────────────────────────────────────


def test_composite_score_gameplay_weights_match_brief():
    """Acceptance test from the gap-close brief.

    hook=90, flow=40, value=60, trend=50, gameplay weights:
    90*.40 + 40*.15 + 60*.30 + 50*.15 = 36 + 6 + 18 + 7.5 = 67.5.
    Python's banker's rounding rounds 67.5 → 68 (the brief said 67;
    we accept the math).
    """
    clip = _make_clip(hook=90, flow=40, value=60, trend=50)
    score = composite_score(clip, ClipContentType.GAMEPLAY)
    assert score in (67, 68)


def test_composite_score_gameplay_no_half_round_edge():
    """A second gameplay sample that doesn't hit the half-integer edge."""
    clip = _make_clip(hook=80, flow=40, value=70, trend=50)
    # 80*.40 + 40*.15 + 70*.30 + 50*.15 = 32 + 6 + 21 + 7.5 = 66.5 → 66 (banker)
    score = composite_score(clip, ClipContentType.GAMEPLAY)
    assert score in (66, 67)


def test_composite_score_default_weights_balanced():
    clip = _make_clip(hook=80, flow=60, value=70, trend=50)
    score = composite_score(clip, None)
    # 80*.30 + 60*.25 + 70*.30 + 50*.15 = 67.5 → 68
    assert score == 68


def test_composite_score_legacy_no_axes_passthrough():
    """When all four axes are 0, return the LLM viral_score unchanged."""
    clip = _make_clip(viral=82)
    assert composite_score(clip, ClipContentType.GAMEPLAY) == 82


def test_composite_score_clamps_to_1_100():
    clip = _make_clip(hook=0, flow=0, value=0, trend=0, viral=0)
    assert composite_score(clip) >= 1


def test_composite_score_unknown_genre_uses_default():
    clip = _make_clip(hook=80, flow=60, value=70, trend=50)
    assert composite_score(clip, None) == composite_score(clip, ClipContentType.GENERIC)


def test_genre_weights_sum_to_one():
    for genre, weights in GENRE_WEIGHTS.items():
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6, f"{genre} weights sum to {total}, not 1.0"
    assert abs(sum(GENRE_WEIGHTS_DEFAULT.values()) - 1.0) < 1e-6


def test_get_weights_falls_back_to_default():
    weights = get_weights(None)
    assert weights == GENRE_WEIGHTS_DEFAULT


# ── finalize_clip_scores ───────────────────────────────────────────


def test_finalize_clip_scores_writes_diagnostics():
    clip = _make_clip(hook=70, flow=60, value=80, trend=50)
    finalize_clip_scores([clip], ClipContentType.TALKING_HEAD)
    assert clip.score_diagnostics is not None
    diag = clip.score_diagnostics
    assert diag["axis_scores"]["hook"] == 70
    assert diag["weights"]["hook"] == GENRE_WEIGHTS[ClipContentType.TALKING_HEAD]["hook"]
    assert diag["content_type"] == "talking_head"
    assert diag["composite_after"] == clip.viral_score


def test_finalize_clip_scores_uses_genre_weights():
    """Same axes, different genres → different composites."""
    sports = _make_clip(hook=90, flow=20, value=60, trend=50)
    talking = _make_clip(hook=90, flow=20, value=60, trend=50)
    finalize_clip_scores([sports], ClipContentType.SPORTS)
    finalize_clip_scores([talking], ClipContentType.TALKING_HEAD)
    # SPORTS weights hook 0.45 → higher score for hook-heavy clip
    assert sports.viral_score > talking.viral_score


# ── fill_axes_from_legacy ──────────────────────────────────────────


def test_fill_axes_from_legacy_splats_score():
    clip = _make_clip(viral=78)
    fill_axes_from_legacy(clip)
    assert clip.hook_score == 78
    assert clip.flow_score == 78
    assert clip.value_score == 78
    # Trend is NOT filled from legacy — stays neutral
    assert clip.trend_score == 50


def test_fill_axes_from_legacy_preserves_existing():
    clip = _make_clip(hook=90, viral=50)
    fill_axes_from_legacy(clip)
    assert clip.hook_score == 90  # not overwritten
    assert clip.flow_score == 50
