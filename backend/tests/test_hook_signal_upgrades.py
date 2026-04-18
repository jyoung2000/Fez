"""Tests for the Phase 5 _score_hook_strength upgrades.

The original scorer only looked at the transcript opening word. Phase
5 adds optional audio-attack and visual-importance signals so a music
drop or a striking opener frame can rescue a clip that would otherwise
score "dead air".
"""

from __future__ import annotations

import pytest

from backend.models import ClipCandidate, SceneDescription, TranscriptSegment
from backend.services.providers.base import _score_hook_strength


def _make_clip(start: float = 0.0) -> ClipCandidate:
    return ClipCandidate(
        id=1,
        title="t",
        start_time=start,
        end_time=start + 30,
        duration=30.0,
        viral_score=50,
        viral_score_reasoning="",
        clip_type="highlight",
        platform="both",
        suggested_caption="",
        hook_text="",
        why_this_works="",
    )


def test_dead_air_with_audio_attack_rescued():
    """No speech in first 3s but a strong audio attack at t=0.2 → boost."""
    clip = _make_clip()
    audio = [{"timestamp": 0.2, "type": "extreme_spike", "delta_db": 14, "sentiment": "cheering"}]
    score, _ = _score_hook_strength(clip, transcript=[], audio_moments=audio)
    assert score > 30


def test_dead_air_no_audio_no_scenes_floors_at_15():
    clip = _make_clip()
    score, reason = _score_hook_strength(clip, transcript=[])
    assert score == 15
    assert reason == "dead_air_opening"


def test_question_opener_still_scores_high_with_audio():
    """Audio attack adds on top of the existing speech-based bonuses."""
    clip = _make_clip()
    transcript = [
        TranscriptSegment(start=0.0, end=2.5, text="Why does nobody talk about this?", speaker="Speaker 1"),
    ]
    audio = [{"timestamp": 0.1, "type": "volume_spike", "delta_db": 9, "sentiment": "laughter"}]
    score, reason = _score_hook_strength(clip, transcript=transcript, audio_moments=audio)
    # Question (+25) + audio_attack (+10) + sentiment_laughter (+12) = 50 + 47 = 97
    assert score >= 90
    assert "audio_attack_hook" in reason
    assert "sentiment_laughter_hook" in reason


def test_visual_peak_in_opener_boosts_score():
    clip = _make_clip()
    transcript = [
        TranscriptSegment(start=0.0, end=2.5, text="Today I want to share a thought.", speaker="Speaker 1"),
    ]
    scenes = [SceneDescription(timestamp=0.5, description="x", importance_score=9, thumbnail_path="")]
    score, reason = _score_hook_strength(clip, transcript=transcript, scenes=scenes)
    assert "visual_peak_hook" in reason


def test_visual_dead_opener_drops_score():
    clip = _make_clip()
    transcript = [
        TranscriptSegment(start=0.0, end=2.5, text="Today I want to share a thought.", speaker="Speaker 1"),
    ]
    scenes = [SceneDescription(timestamp=0.5, description="x", importance_score=2, thumbnail_path="")]
    score, reason = _score_hook_strength(clip, transcript=transcript, scenes=scenes)
    assert "visual_dead_hook" in reason


def test_filler_opener_still_penalised():
    clip = _make_clip()
    transcript = [
        TranscriptSegment(start=0.0, end=2.5, text="Um, so I was thinking about this.", speaker="Speaker 1"),
    ]
    score, reason = _score_hook_strength(clip, transcript=transcript)
    assert score < 50
    assert "filler_opener_um" in reason
