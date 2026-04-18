"""Filename-based gameplay hint unit tests.

Targets the TF2 regression: ``TF2： Washed Up Tuber.mp4`` must
match the ``team_fortress_2`` keyword despite the halfwidth
colon, mixed case, and embedded spaces. Also covers the full
keyword bank + clean negatives.
"""

from __future__ import annotations

from backend.services.filename_gameplay_hint import (
    filename_gameplay_hint,
    is_center_bias_genre,
)


# ──────────────────── TF2 regression ────────────────────


def test_tf2_halfwidth_colon_filename_matches():
    """The exact filename from the bug report must hit
    the team_fortress_2 keyword."""
    hint = filename_gameplay_hint("TF2： Washed Up Tuber.mp4")
    assert hint.matched is True
    assert hint.slug == "team_fortress_2"
    assert hint.genre == "fps"
    assert is_center_bias_genre(hint.genre)


def test_tf2_plain_dash_form_matches():
    hint = filename_gameplay_hint("my_tf2_clip.mp4")
    assert hint.matched is True
    assert hint.slug == "team_fortress_2"


def test_tf2_team_fortress_2_spelled_out_matches():
    hint = filename_gameplay_hint("Team Fortress 2 highlights.mov")
    assert hint.matched is True
    assert hint.slug == "team_fortress_2"


def test_tf2_team_fortress_2_with_underscores():
    hint = filename_gameplay_hint("team_fortress_2-ep01.mp4")
    assert hint.matched is True


# ──────────────────── FPS / hero shooter hits ────────────────────


def test_valorant_filename():
    hint = filename_gameplay_hint("/tmp/Valorant ace pickup.mp4")
    assert hint.matched
    assert hint.slug == "valorant"
    assert hint.genre == "fps"
    assert is_center_bias_genre(hint.genre)


def test_apex_legends_filename():
    hint = filename_gameplay_hint("apex_legends_clutch.mp4")
    assert hint.matched
    assert hint.slug == "apex_legends"
    assert hint.genre == "fps"


def test_csgo_filename():
    hint = filename_gameplay_hint("csgo highlight reel.mkv")
    assert hint.matched
    assert hint.slug == "cs_go"


def test_cs2_filename():
    hint = filename_gameplay_hint("CS2 deagle ace.mp4")
    assert hint.matched
    assert hint.slug == "cs_2"


def test_fortnite_filename():
    hint = filename_gameplay_hint("fortnite clutch moment.mov")
    assert hint.matched
    assert hint.slug == "fortnite"
    assert hint.genre == "fps"


def test_warzone_filename():
    hint = filename_gameplay_hint("warzone solo win.mp4")
    assert hint.matched
    assert hint.slug == "warzone"


def test_overwatch_filename():
    hint = filename_gameplay_hint("overwatch ana sleep.mp4")
    assert hint.matched
    assert hint.slug == "overwatch"
    assert hint.genre == "hero_shooter"
    assert is_center_bias_genre(hint.genre)


def test_marvel_rivals_filename():
    hint = filename_gameplay_hint("Marvel Rivals first win.mp4")
    assert hint.matched
    assert hint.slug == "marvel_rivals"


def test_minecraft_sandbox():
    hint = filename_gameplay_hint("minecraft base tour.mp4")
    assert hint.matched
    assert hint.genre == "sandbox"
    assert is_center_bias_genre(hint.genre)


# ──────────────────── Non-center-bias genres ────────────────────


def test_lol_moba_not_center_bias():
    hint = filename_gameplay_hint("league_of_legends_penta.mp4")
    assert hint.matched
    assert hint.genre == "moba"
    assert not is_center_bias_genre(hint.genre)


def test_rocket_league_racing():
    hint = filename_gameplay_hint("rocket league airshot.mp4")
    assert hint.matched
    assert hint.genre == "racing"
    assert not is_center_bias_genre(hint.genre)


def test_gta_v_tps():
    hint = filename_gameplay_hint("GTA V heist finale.mp4")
    assert hint.matched
    assert hint.slug == "gta_v"
    assert hint.genre == "tps"
    assert not is_center_bias_genre(hint.genre)


def test_elden_ring_tps():
    hint = filename_gameplay_hint("elden ring boss fight.mp4")
    assert hint.matched
    assert hint.slug == "elden_ring"
    assert hint.genre == "tps"


# ──────────────────── Negatives ────────────────────


def test_empty_and_none_return_no_match():
    assert filename_gameplay_hint(None).matched is False
    assert filename_gameplay_hint("").matched is False
    assert filename_gameplay_hint("   ").matched is False


def test_talking_head_podcast_filename_no_match():
    for name in (
        "podcast_episode_12.mp4",
        "Rogan Interview.mp4",
        "vlog_morning_routine.mp4",
        "wedding_speech.mp4",
        "lecture_transformer_networks.mp4",
    ):
        hint = filename_gameplay_hint(name)
        assert hint.matched is False, f"{name} falsely matched {hint.slug}"


def test_movie_and_anime_titles_no_false_match():
    for name in (
        "Attack_on_Titan_S04E01.mkv",
        "Naruto_Shippuden_200.mp4",
        "The Dark Knight.mp4",
        "spirited_away_1080p.mp4",
    ):
        hint = filename_gameplay_hint(name)
        assert hint.matched is False, f"{name} falsely matched {hint.slug}"


def test_path_with_directories_matches_basename_only():
    hint = filename_gameplay_hint(
        "/data/uploads/abc123/Valorant_clutch.mp4",
    )
    assert hint.matched
    assert hint.slug == "valorant"


def test_short_abbreviations_require_word_boundary():
    # "lola" should not match "lol" at the start
    hint = filename_gameplay_hint("lola runs a marathon.mp4")
    assert hint.matched is False


# ──────────────────── Source-token bookkeeping ────────────────────


def test_source_token_populated_on_hit():
    hint = filename_gameplay_hint("My Apex Legends Clip.mp4")
    assert hint.matched
    assert "apex" in hint.source_token.lower()
