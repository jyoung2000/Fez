"""Tests for the Clip Focus fixes from the clip-focus / viral-settings audit.

Covers:
- Bug 3: focus_mode preserves LLM-reported relevance in viral_score and
  stashes the composite virality in viral_score_composite.
- Bug 10: deduplicate_overlapping_clips drops near-duplicate focus matches
  by time IoU.
- Bug 12: fill_axes_from_legacy tags score_diagnostics.legacy_fill=True
  and finalize_clip_scores flags the all-zero axis case similarly.
- GenerateClipsRequest gains min_relevance / append / scope fields.
- JobResult persists clip-detection context fields (hot_zones, chapters,
  trend_context, sentiment_timeline, clip_content_type).
"""

import pytest

from backend.models import ClipCandidate, GenerateClipsRequest, JobResult
from backend.services.clip_scoring import (
    composite_score,
    deduplicate_overlapping_clips,
    fill_axes_from_legacy,
    finalize_clip_scores,
)
from backend.services.content_classifier import ClipContentType


def _mk(
    id: int,
    start: float,
    end: float,
    viral: int = 50,
    *,
    focus: str | None = None,
    relevance: int | None = None,
    hook: int = 0,
    flow: int = 0,
    value: int = 0,
    trend: int = 0,
) -> ClipCandidate:
    return ClipCandidate(
        id=id,
        title=f"c{id}",
        start_time=start,
        end_time=end,
        duration=end - start,
        viral_score=viral,
        viral_score_reasoning="",
        clip_type="highlight",
        platform="both",
        suggested_caption="",
        hook_text="",
        why_this_works="",
        clip_focus=focus,
        focus_relevance=relevance,
        hook_score=hook,
        flow_score=flow,
        value_score=value,
        trend_score=trend,
    )


# ── Bug 3: focus_mode preserves relevance ─────────────────────────────


def test_focus_mode_preserves_relevance_in_viral_score():
    clip = _mk(
        1, 10, 40, viral=85, focus="fighting", relevance=85,
        hook=90, flow=80, value=70, trend=50,
    )
    finalize_clip_scores(
        [clip], ClipContentType.TALKING_HEAD, focus_mode=True,
    )
    # viral_score must stay at the LLM-reported relevance (85)
    assert clip.viral_score == 85
    # composite virality lives in the new field
    assert clip.viral_score_composite is not None
    assert clip.viral_score_composite != 85  # composite != relevance here
    assert clip.score_diagnostics["focus_mode"] is True


def test_non_focus_mode_overwrites_viral_score_with_composite():
    clip = _mk(
        1, 10, 40, viral=50,
        hook=90, flow=80, value=70, trend=50,
    )
    finalize_clip_scores(
        [clip], ClipContentType.TALKING_HEAD, focus_mode=False,
    )
    # viral_score should be recomputed, not stay at 50
    assert clip.viral_score != 50
    assert clip.viral_score_composite == clip.viral_score
    assert clip.score_diagnostics["focus_mode"] is False


# ── Bug 10: IoU dedup ────────────────────────────────────────────────


def test_dedupe_drops_near_duplicates_same_focus():
    clips = [
        _mk(1, 120, 150, focus="fight", relevance=85),  # winner
        _mk(2, 125, 155, focus="fight", relevance=70),  # 83% IoU with #1 → drop
        _mk(3, 300, 330, focus="fight", relevance=60),  # separate
    ]
    kept, dropped = deduplicate_overlapping_clips(clips, iou_threshold=0.6)
    assert dropped == 1
    assert {c.id for c in kept} == {1, 3}


def test_dedupe_keeps_overlap_across_different_focus():
    clips = [
        _mk(1, 120, 150, focus="fight", relevance=85),
        _mk(2, 125, 155, focus="advice", relevance=70),  # same window, different focus
    ]
    kept, dropped = deduplicate_overlapping_clips(clips, iou_threshold=0.6)
    assert dropped == 0
    assert len(kept) == 2


def test_dedupe_empty_list():
    kept, dropped = deduplicate_overlapping_clips([])
    assert kept == []
    assert dropped == 0


def test_dedupe_prefers_higher_relevance_when_present():
    # Two clips with same IoU; the one with higher focus_relevance wins
    # even if its viral_score is lower.
    clips = [
        _mk(1, 0, 30, viral=40, focus="x", relevance=95),  # high relevance
        _mk(2, 1, 29, viral=90, focus="x", relevance=60),  # low relevance, high viral
    ]
    kept, _ = deduplicate_overlapping_clips(clips, iou_threshold=0.5)
    assert len(kept) == 1
    assert kept[0].id == 1


# ── Bug 12: legacy_fill diagnostic ───────────────────────────────────


def test_fill_axes_from_legacy_marks_diagnostic():
    clip = _mk(1, 0, 30, viral=80)  # all axes default to 0
    fill_axes_from_legacy(clip)
    assert clip.hook_score == 80
    assert clip.flow_score == 80
    assert clip.value_score == 80
    assert clip.trend_score == 50  # neutral default
    assert clip.score_diagnostics is not None
    assert clip.score_diagnostics["legacy_fill"] is True


def test_finalize_clip_scores_marks_legacy_fill_when_axes_zero():
    # Axes all zero → composite falls back to viral_score; we should
    # still flag legacy_fill in the diagnostics so the UI knows.
    clip = _mk(1, 0, 30, viral=75)
    finalize_clip_scores([clip], ClipContentType.TALKING_HEAD, focus_mode=False)
    assert clip.score_diagnostics["legacy_fill"] is True


def test_finalize_clip_scores_does_not_mark_legacy_when_axes_populated():
    clip = _mk(1, 0, 30, viral=50, hook=80, flow=70, value=60, trend=50)
    finalize_clip_scores([clip], ClipContentType.TALKING_HEAD, focus_mode=False)
    assert clip.score_diagnostics["legacy_fill"] is False


# ── GenerateClipsRequest field additions ─────────────────────────────


def test_generate_clips_request_new_fields_defaults():
    req = GenerateClipsRequest()
    assert req.viral_score_min == 0
    assert req.viral_score_max == 100
    assert req.min_relevance == 0
    assert req.append is True  # preserves legacy append behavior
    assert req.scope_start is None
    assert req.scope_end is None


def test_generate_clips_request_accepts_scope_and_relevance():
    req = GenerateClipsRequest(
        clip_focus="fight scene",
        min_relevance=70,
        scope_start=120.0,
        scope_end=240.0,
        append=False,
    )
    assert req.clip_focus == "fight scene"
    assert req.min_relevance == 70
    assert req.scope_start == 120.0
    assert req.scope_end == 240.0
    assert req.append is False


# ── JobResult context persistence ────────────────────────────────────


def test_job_result_persists_clip_context():
    # Smoke test that JobResult round-trips the new fields.
    data = {
        "job_id": "test",
        "filename": "x.mp4",
        "file_path": "/tmp/x.mp4",
        "hot_zones": [
            {"start": 0, "end": 30, "composite_score": 85,
             "audio_score": 80, "transcript_score": 70,
             "scene_score": 60, "speaker_score": 50, "signals": ["loud"]},
        ],
        "chapters": [
            {"start": 0, "end": 60, "title": "Intro",
             "topic_keywords": ["welcome"], "speaker_turns": 3, "scene_cuts": 2},
        ],
        "trend_context": "trend=foo",
        "sentiment_timeline": "sentiment=bar",
        "clip_content_type": "talking_head",
    }
    job = JobResult(**data)
    assert len(job.hot_zones) == 1
    assert job.hot_zones[0]["composite_score"] == 85
    assert len(job.chapters) == 1
    assert job.chapters[0]["title"] == "Intro"
    assert job.trend_context == "trend=foo"
    assert job.sentiment_timeline == "sentiment=bar"
    assert job.clip_content_type == "talking_head"

    # Round-trip via JSON dump + reload
    dumped = job.model_dump()
    assert "hot_zones" in dumped
    assert dumped["hot_zones"][0]["start"] == 0
