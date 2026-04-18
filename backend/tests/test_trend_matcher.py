"""Tests for the trend matcher (Phase 3)."""

import json
import os
import tempfile

import pytest

from backend.models import TranscriptSegment
from backend.services.content_classifier import ClipContentType
from backend.services.trend_matcher import (
    TrendMatcher,
    build_default_trend_matcher,
    format_trend_context,
)


def _write_lexicon(entries: list[dict]) -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump({"version": "test", "entries": entries}, f)
    return path


def test_score_clip_default_neutral_no_lexicon():
    """Empty lexicon → default 50, floor enforced."""
    matcher = TrendMatcher(lexicon_path="/nonexistent/path.json")
    score, reason = matcher.score_clip("This is a normal clip about cats.")
    assert score == 50


def test_score_clip_phrase_match_boosts_above_70():
    path = _write_lexicon([
        {"phrase": "hot take", "genres": ["talking_head"], "platforms": ["tiktok"], "hotness": 1.0},
    ])
    matcher = TrendMatcher(lexicon_path=path)
    score, reason = matcher.score_clip(
        "Here's my hot take on this topic.",
        content_type=ClipContentType.TALKING_HEAD,
    )
    assert score >= 70
    assert "hot take" in reason


def test_score_clip_genre_mismatch_no_boost():
    path = _write_lexicon([
        {"phrase": "clutch play", "genres": ["gameplay"], "platforms": ["shorts"], "hotness": 0.9},
    ])
    matcher = TrendMatcher(lexicon_path=path)
    # talking_head clip mentioning gameplay phrase → off-genre, no boost
    score, _ = matcher.score_clip(
        "He had a clutch play in that game.",
        content_type=ClipContentType.TALKING_HEAD,
    )
    assert score == 50


def test_score_clip_floor_at_20():
    matcher = TrendMatcher(lexicon_path="/nonexistent/path.json")
    score, _ = matcher.score_clip("nothing", content_type=ClipContentType.GENERIC)
    assert score >= 20


def test_score_clip_emphasis_keywords_contribute():
    path = _write_lexicon([])
    matcher = TrendMatcher(lexicon_path=path, emphasis_keywords=["amazing", "incredible"])
    score, _ = matcher.score_clip(
        "This was an amazing and incredible moment.",
    )
    assert score >= 60  # 50 baseline + emphasis bonus


def test_format_trend_context_empty_when_no_match():
    matcher = TrendMatcher(lexicon_path="/nonexistent/path.json")
    transcript = [TranscriptSegment(start=0, end=5, text="random words", speaker="Speaker 1")]
    out = format_trend_context(matcher, transcript)
    assert out == ""


def test_format_trend_context_lists_matches():
    path = _write_lexicon([
        {"phrase": "plot twist", "genres": ["narrative"], "platforms": ["tiktok"], "hotness": 0.85},
    ])
    matcher = TrendMatcher(lexicon_path=path)
    transcript = [
        TranscriptSegment(start=0, end=5, text="And then the plot twist hits.", speaker="Speaker 1"),
    ]
    # content_type=None keeps the matcher's genre filter off so a
    # narrative-tagged phrase still matches a generic clip in the test.
    out = format_trend_context(matcher, transcript, content_type=None)
    assert "plot twist" in out


def test_acceptance_lexicon_emptied_neutral():
    """Brief acceptance: same clip with empty lexicon → score == 50."""
    empty = TrendMatcher(lexicon_path="/nonexistent.json")
    score, _ = empty.score_clip("Here's my hot take", content_type=ClipContentType.TALKING_HEAD)
    assert score == 50


def test_default_lexicon_loads():
    """The shipped backend/data/trend_lexicon.json must parse."""
    matcher = build_default_trend_matcher()
    # If the file exists and parses, score should still be deterministic.
    score, _ = matcher.score_clip("This is a clutch play.", content_type=ClipContentType.GAMEPLAY)
    assert score >= 70
