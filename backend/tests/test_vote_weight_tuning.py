"""Gap 5c — tests for per-content-type vote weight tuning.

Pins the ``ACTIVE_SPEAKER_VOTE_WEIGHTS`` table math and the
``get_vote_weights`` fallback behavior. These run in the sandbox
without any heavy deps.
"""

from __future__ import annotations

import pytest

from backend.services.content_type_config import (
    ACTIVE_SPEAKER_VOTE_WEIGHTS,
    ContentType,
    get_vote_weights,
)


def test_each_row_sums_to_vote_budget():
    """Every row must sum to 0.25 (the total speaker_agree budget)."""
    for ct, (lip_w, diar_w) in ACTIVE_SPEAKER_VOTE_WEIGHTS.items():
        total = lip_w + diar_w
        assert total == pytest.approx(0.25, abs=1e-9), \
            f"{ct} row sums to {total}, expected 0.25"


def test_legacy_mode_returns_full_lip_budget():
    """``has_diarization=False`` → (0.20, 0.0) regardless of type."""
    assert get_vote_weights("multi_speaker_panel", False) == (0.20, 0.0)
    assert get_vote_weights("anime", False) == (0.20, 0.0)
    assert get_vote_weights("vlog", False) == (0.20, 0.0)


def test_panel_diar_favored():
    """Multi-speaker panel should weight diarization ≥ lip."""
    lip_w, diar_w = get_vote_weights("multi_speaker_panel", True)
    assert diar_w >= lip_w, (
        f"panel expected diar-favored, got lip={lip_w} diar={diar_w}"
    )


def test_anime_diar_favored():
    """Anime should weight diarization ≥ lip (stylized mouth flaps)."""
    lip_w, diar_w = get_vote_weights("anime", True)
    assert diar_w >= lip_w


def test_animation_dialogue_diar_favored():
    """Anime dialogue subtype inherits anime's tuning."""
    lip_w, diar_w = get_vote_weights("animation_dialogue", True)
    assert diar_w >= lip_w


def test_vlog_lip_favored():
    """Vlog should weight lip > diar (B-roll audio is unreliable)."""
    lip_w, diar_w = get_vote_weights("vlog", True)
    assert lip_w > diar_w


def test_narrative_lip_favored():
    """Narrative should weight lip > diar (ADR contaminates audio)."""
    lip_w, diar_w = get_vote_weights("narrative", True)
    assert lip_w > diar_w


def test_music_video_lip_favored():
    """Music video lip > diar (lipsync is clean, audio is 1 cluster)."""
    lip_w, diar_w = get_vote_weights("music_video", True)
    assert lip_w > diar_w


def test_unknown_type_gets_default():
    """Unknown content type falls through to the generic 0.15 / 0.10."""
    lip_w, diar_w = get_vote_weights("unknown", True)
    assert (lip_w, diar_w) == (0.15, 0.10)


def test_invalid_type_string_falls_to_unknown():
    """Typo / garbage string should not crash; falls to UNKNOWN."""
    lip_w, diar_w = get_vote_weights("totally_made_up", True)
    assert (lip_w, diar_w) == (0.15, 0.10)
