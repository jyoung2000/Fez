"""Tests for chapter segmentation (Phase 6)."""

import pytest

from backend.models import SceneDescription, TranscriptSegment
from backend.services.chapter_segmenter import (
    Chapter,
    segment_chapters,
)


def _seg(start: float, end: float, text: str, speaker: str = "Speaker 1") -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text=text, speaker=speaker)


def _scene(ts: float, importance: int = 5) -> SceneDescription:
    return SceneDescription(
        timestamp=ts,
        description="x",
        importance_score=importance,
        thumbnail_path="",
    )


def test_segment_chapters_short_video_returns_single_chapter():
    transcript = [_seg(0, 30, "talking about cooking pasta")]
    chapters = segment_chapters(transcript, scenes=[], min_len=45.0, max_len=300.0)
    assert len(chapters) == 1
    assert chapters[0].start == 0.0


def test_segment_chapters_topic_shift_detected():
    """Two unrelated 60-second buckets should produce a boundary."""
    transcript = [
        _seg(0, 30, "cooking pasta carbonara recipe italian kitchen"),
        _seg(30, 60, "boil water salt eggs bacon parmesan cheese"),
        _seg(60, 90, "rocket launch space exploration mars NASA"),
        _seg(90, 120, "astronaut orbit moon mission satellite"),
        _seg(120, 150, "carbon dioxide oxygen pressure"),
        _seg(150, 180, "more space mars lander rover"),
        _seg(180, 210, "more rover stuff"),
        _seg(210, 240, "more rover stuff continued"),
    ]
    chapters = segment_chapters(transcript, scenes=[], min_len=45.0, max_len=600.0)
    # We should produce more than one chapter when the topic shifts hard
    assert len(chapters) >= 2


def test_segment_chapters_respects_max_len():
    """A single long topic block should be split when it exceeds max_len."""
    transcript = [_seg(i, i + 5, "same topic repeat repeat repeat") for i in range(0, 600, 5)]
    chapters = segment_chapters(transcript, scenes=[], min_len=45.0, max_len=200.0)
    for ch in chapters:
        assert ch.end - ch.start <= 200.0 + 1e-3


def test_segment_chapters_empty_transcript_returns_empty():
    assert segment_chapters([], []) == []


def test_chapter_to_dict_round_trip():
    ch = Chapter(start=0, end=60, title="Test", topic_keywords=["a", "b"])
    d = ch.to_dict()
    assert d["start"] == 0
    assert d["title"] == "Test"
    assert d["topic_keywords"] == ["a", "b"]


def test_chapter_titles_use_keywords():
    transcript = [
        _seg(0, 30, "marathon training pace sprint speed"),
        _seg(30, 60, "marathon strategy hydration electrolytes"),
        _seg(60, 90, "marathon recovery stretches sleep"),
    ]
    chapters = segment_chapters(transcript, scenes=[], min_len=10.0, max_len=300.0)
    assert chapters
    # The single chapter title should mention at least one TF-relevant word.
    title = chapters[0].title.lower()
    assert "marathon" in title or any(kw in title for kw in ("training", "strategy", "recovery"))
