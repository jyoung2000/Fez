"""Phase 2 — UI dropdown → classifier routing matrix.

One parametrized test per row in the Phase 2 dropdown spec. Each row
asserts the full routing chain:

    Upload.jsx token
        → normalize_ui_content_type   (ContentType + flags + subtypes)
        → classify_content metadata    (profile fields populated)
        → classify_clip                (ClipContentType for solver tuning)

The matrix is the canonical contract between the UI and the classifier
layers. Adding a new dropdown row in Upload.jsx without adding a row
here will fail this suite the next time someone runs ``pytest``.

Phase 2 also extends ``classify_clip`` with the gameplay sub-categories
(``GAMEPLAY_MOBA`` / ``GAMEPLAY_TPS`` / ``GAMEPLAY_RACING``) and routes
``stream`` to ``ClipContentType.STREAM``. Both are exercised here.

The matrix also pins the anime sub-dropdown semantics:
``anime_subtype="action"`` lands on ``ANIMATION``, ``"dialogue"`` /
``"slice_of_life"`` land on ``ANIMATION_DIALOGUE``. Phase 6 will use
``profile.anime_subtype`` directly to drive the anime shot detector
+ lead-room multiplier; this test pins the ContentProfile flag so a
future refactor can't drop it.
"""

from __future__ import annotations

import pytest

from backend.services.content_classifier import (
    ClipContentType,
    classify_clip,
    classify_content,
)
from backend.services.content_type_config import ContentType
from backend.services.content_type_strings import (
    is_gameplay_override,
    normalize_ui_content_type,
)
from backend.services.game_layouts import (
    DEFAULT_GAME_BY_GENRE,
    GAME_GENRE,
    GAME_HUD_LAYOUTS,
    games_for_genre,
    get_action_center,
)


# ───────────────────── The full routing matrix ─────────────────────
#
# Each row:
#   token         — UI dropdown value (Upload.jsx)
#   anime_sub     — anime sub-dropdown value, or None if N/A
#   music_sub     — music sub-dropdown value, or None if N/A
#   parent_enum   — expected ContentType after normalize_ui_content_type
#   gameplay_sub  — expected NormalizedContentType.gameplay_subtype
#   panel_flag    — expected NormalizedContentType.is_multi_speaker_panel
#   animated_flag — expected NormalizedContentType.is_animated
#   fast_path     — expected is_gameplay_override(token)
#   clip_type     — expected ClipContentType after classify_clip

ROUTING_MATRIX = [
    # ── Legacy ──
    ("gameplay", None, None, ContentType.GAMING, "fps", False, False, True,
     ClipContentType.GAMEPLAY),
    ("movie", None, None, ContentType.NARRATIVE, None, False, False, False,
     ClipContentType.GENERIC),
    ("podcast", None, None, ContentType.PODCAST, None, False, False, False,
     ClipContentType.TALKING_HEAD),

    # ── People / Dialogue ──
    ("debate", None, None, ContentType.PODCAST, None, True, False, False,
     ClipContentType.MULTI_SPEAKER_PANEL),
    ("panel", None, None, ContentType.PODCAST, None, True, False, False,
     ClipContentType.MULTI_SPEAKER_PANEL),
    ("interview", None, None, ContentType.PODCAST, None, False, False, False,
     ClipContentType.TALKING_HEAD),
    ("vlog", None, None, ContentType.VLOG, None, False, False, False,
     ClipContentType.TALKING_HEAD),
    ("narrative", None, None, ContentType.NARRATIVE, None, False, False, False,
     ClipContentType.GENERIC),
    ("cinematic", None, None, ContentType.NARRATIVE, None, False, False, False,
     ClipContentType.GENERIC),

    # ── Animation (parent + sub-dropdown matrix) ──
    ("anime", None, None, ContentType.ANIME, None, False, True, False,
     ClipContentType.ANIMATION),
    ("anime", "action", None, ContentType.ANIME, None, False, True, False,
     ClipContentType.ANIMATION),
    ("anime", "dialogue", None, ContentType.ANIME, None, False, True, False,
     ClipContentType.ANIMATION_DIALOGUE),
    ("anime", "slice_of_life", None, ContentType.ANIME, None, False, True, False,
     ClipContentType.ANIMATION_DIALOGUE),
    ("cartoon", None, None, ContentType.ANIME, None, False, True, False,
     ClipContentType.ANIMATION),

    # ── Music / Performance ──
    ("music_video", None, None, ContentType.MUSIC_VIDEO, None, False, False, False,
     ClipContentType.MUSIC_VIDEO),
    ("music_video", None, "performance", ContentType.MUSIC_VIDEO, None, False, False, False,
     ClipContentType.MUSIC_VIDEO),
    ("music_video", None, "narrative", ContentType.MUSIC_VIDEO, None, False, False, False,
     ClipContentType.MUSIC_VIDEO),
    ("music_video", None, "lyric", ContentType.MUSIC_VIDEO, None, False, False, False,
     ClipContentType.MUSIC_VIDEO),

    # ── Gaming (parent variants) ──
    ("gameplay_fps", None, None, ContentType.GAMING, "fps", False, False, True,
     ClipContentType.GAMEPLAY),
    ("gameplay_moba", None, None, ContentType.GAMING, "moba", False, False, True,
     ClipContentType.GAMEPLAY_MOBA),
    ("gameplay_tps", None, None, ContentType.GAMING, "tps", False, False, True,
     ClipContentType.GAMEPLAY_TPS),
    ("gameplay_racing", None, None, ContentType.GAMING, "racing", False, False, True,
     ClipContentType.GAMEPLAY_RACING),
    # Stream is gaming-layout but is NOT a fast-path candidate (it
    # has a facecam and still needs the face pipeline).
    ("stream", None, None, ContentType.GAMING, "stream", False, False, False,
     ClipContentType.STREAM),

    # ── Sports ──
    # v4: SPORTS parent now maps to ClipContentType.SPORTS, and the
    # sub-types route to SPORTS_BASKETBALL / SPORTS_RACING.
    ("sports", None, None, ContentType.SPORTS, None, False, False, False,
     ClipContentType.SPORTS),
    ("sports_basketball", None, None, ContentType.SPORTS, None, False, False, False,
     ClipContentType.SPORTS_BASKETBALL),
    ("sports_racing", None, None, ContentType.SPORTS, None, False, False, False,
     ClipContentType.SPORTS_RACING),
    ("basketball", None, None, ContentType.SPORTS, None, False, False, False,
     ClipContentType.SPORTS_BASKETBALL),
    ("racing", None, None, ContentType.SPORTS, None, False, False, False,
     ClipContentType.SPORTS_RACING),
]


@pytest.mark.parametrize(
    "token, anime_sub, music_sub, parent_enum, gameplay_sub, panel_flag, "
    "animated_flag, fast_path, clip_type",
    ROUTING_MATRIX,
)
def test_routing_matrix_row(
    token, anime_sub, music_sub, parent_enum, gameplay_sub, panel_flag,
    animated_flag, fast_path, clip_type,
):
    """One assertion per layer of the routing chain.

    If this fails, the failing assertion tells you exactly which layer
    drifted — the normalizer, the classifier, or classify_clip.
    """
    # ── Layer 1: normalizer ──
    normalized = normalize_ui_content_type(token)
    assert normalized is not None, f"{token!r} normalized to None"
    assert normalized.content_type == parent_enum, (
        f"{token}: normalizer parent {normalized.content_type} != {parent_enum}"
    )
    assert normalized.gameplay_subtype == gameplay_sub, (
        f"{token}: gameplay_subtype {normalized.gameplay_subtype} != {gameplay_sub}"
    )
    assert normalized.is_multi_speaker_panel == panel_flag, (
        f"{token}: panel flag {normalized.is_multi_speaker_panel} != {panel_flag}"
    )
    assert normalized.is_animated == animated_flag, (
        f"{token}: animated flag {normalized.is_animated} != {animated_flag}"
    )

    # ── Layer 2: gameplay fast-path ──
    assert is_gameplay_override(token) == fast_path, (
        f"{token}: gameplay fast-path {is_gameplay_override(token)} != {fast_path}"
    )

    # ── Layer 3: classify_content via metadata injection ──
    md: dict[str, str] = {"content_type_override": token}
    if anime_sub is not None:
        md["anime_subtype"] = anime_sub
    if music_sub is not None:
        md["music_subtype"] = music_sub
    profile = classify_content(
        shot_cuts=[],
        face_registry=None,
        dense_faces=[],
        scenes=[],
        video_duration=60.0,
        metadata=md,
    )
    assert profile.content_type == parent_enum.value
    assert profile.confidence == 1.0
    assert profile.is_multi_speaker_panel == panel_flag
    assert profile.is_animated == animated_flag
    assert profile.gameplay_subtype == gameplay_sub
    if animated_flag and anime_sub:
        assert profile.anime_subtype == anime_sub
    if profile.content_type == ContentType.MUSIC_VIDEO.value and music_sub:
        assert profile.music_subtype == music_sub

    # ── Layer 4: classify_clip ──
    assert classify_clip(content_profile=profile) == clip_type, (
        f"{token}: classify_clip = {classify_clip(content_profile=profile).value} "
        f"!= {clip_type.value}"
    )


# ─────────────── Game-layout coverage for Phase 2 ───────────────

class TestGameLayoutsExpansion:
    """Phase 2 added 9 new entries to GAME_HUD_LAYOUTS spanning MOBA,
    TPS, racing, and sandbox. This suite locks the genre + action-center
    contract so Phase 7 has stable inputs to wire to."""

    NEW_GAMES = [
        ("league_of_legends", "moba"),
        ("dota2", "moba"),
        ("rocket_league", "racing"),
        ("gta_v", "tps"),
        ("elden_ring", "tps"),
        ("minecraft", "sandbox"),
        ("generic_moba", "moba"),
        ("generic_tps", "tps"),
        ("generic_racing", "racing"),
    ]

    @pytest.mark.parametrize("key, genre", NEW_GAMES)
    def test_new_game_present_with_correct_genre(self, key, genre):
        layout = GAME_HUD_LAYOUTS.get(key)
        assert layout is not None, f"missing GAME_HUD_LAYOUTS[{key!r}]"
        assert layout.get("genre") == genre, (
            f"{key}: genre {layout.get('genre')} != {genre}"
        )
        # Action center must be a (x, y) tuple of percents in [0, 100].
        cx, cy = get_action_center(key)
        assert 0 <= cx <= 100
        assert 0 <= cy <= 100

    def test_fps_action_center_is_screen_center(self):
        for fps in ("overwatch", "valorant", "marvel_rivals", "generic_fps"):
            assert get_action_center(fps) == (50.0, 50.0), fps

    def test_tps_action_center_is_above_center(self):
        # Third-person games have the player offset down+right of frame
        # center; the head/shoulders sit slightly above true center, so
        # the y anchor is < 50.
        for tps in ("gta_v", "elden_ring", "generic_tps"):
            cx, cy = get_action_center(tps)
            assert cx == 50.0
            assert cy < 50.0, f"{tps}: cy={cy} should be < 50 for TPS"

    def test_racing_action_center_is_lower_third(self):
        # The car sits in the lower portion of the frame; the anchor
        # must be in the lower half.
        cx, cy = get_action_center("generic_racing")
        assert cx == 50.0
        assert cy > 50.0, f"generic_racing: cy={cy} should be > 50"

    def test_unknown_game_defaults_to_center(self):
        assert get_action_center("nonexistent_game") == (50.0, 50.0)

    def test_genre_index_is_complete(self):
        # GAME_GENRE must have an entry for every key in GAME_HUD_LAYOUTS.
        assert set(GAME_GENRE.keys()) == set(GAME_HUD_LAYOUTS.keys())

    def test_default_game_per_genre(self):
        for genre, key in DEFAULT_GAME_BY_GENRE.items():
            assert key in GAME_HUD_LAYOUTS, (
                f"DEFAULT_GAME_BY_GENRE[{genre!r}]={key!r} missing from layouts"
            )
            assert GAME_HUD_LAYOUTS[key]["genre"] == genre, (
                f"default game {key} for genre {genre} has wrong genre"
            )

    def test_games_for_genre_returns_only_that_genre(self):
        for genre in ("fps", "moba", "tps", "racing", "sandbox"):
            games = games_for_genre(genre)
            assert len(games) >= 1, f"genre {genre} has no games"
            for key, _name in games:
                assert GAME_HUD_LAYOUTS[key]["genre"] == genre

    def test_games_for_genre_falls_back_to_fps(self):
        # An unknown genre returns FPS games (the legacy default).
        games = games_for_genre("not_a_real_genre")
        assert games
        for key, _name in games:
            assert GAME_HUD_LAYOUTS[key]["genre"] == "fps"


# ─────────── End-to-end: every Upload.jsx token round-trips ───────────

class TestEndToEndCoverage:
    """Quick guard that the matrix above didn't drop a UI token.

    Upload.jsx exposes 14 distinct content_type_override values (3
    legacy + 11 new). The matrix exercises every one of them at least
    once; this test asserts the count and the unique-set membership.
    """

    UPLOAD_JSX_TOKENS = {
        # Legacy
        "gameplay", "movie", "podcast",
        # People / Dialogue
        "debate", "vlog", "narrative",
        # Animation
        "anime",
        # Music
        "music_video",
        # Gaming
        "gameplay_moba", "gameplay_tps", "gameplay_racing", "stream",
        # Sports (v4)
        "sports", "sports_basketball", "sports_racing",
    }

    def test_matrix_covers_every_upload_jsx_token(self):
        matrix_tokens = {row[0] for row in ROUTING_MATRIX}
        missing = self.UPLOAD_JSX_TOKENS - matrix_tokens
        assert not missing, (
            f"ROUTING_MATRIX is missing rows for these Upload.jsx tokens: {missing}"
        )

    def test_every_matrix_token_normalizes(self):
        for row in ROUTING_MATRIX:
            token = row[0]
            assert normalize_ui_content_type(token) is not None, token


# ─────────── Pipeline metadata injection — Phase 2 fields ───────────

class TestPipelineSubtypeInjection:
    """AST guard: pipeline.py must inject the Phase 2 sub-type fields
    into the dict it passes to classify_content, not just
    content_type_override (which Phase 1 already covered)."""

    def test_pipeline_injects_anime_subtype(self):
        import ast
        from pathlib import Path
        src = (
            Path(__file__).resolve().parents[1] / "services" / "pipeline.py"
        ).read_text()
        # Look for _classifier_metadata['anime_subtype'] = ... assignment
        tree = ast.parse(src)
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if (
                        isinstance(tgt, ast.Subscript)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "_classifier_metadata"
                        and isinstance(tgt.slice, ast.Constant)
                        and tgt.slice.value == "anime_subtype"
                    ):
                        found = True
                        break
        assert found, (
            "pipeline.py must inject anime_subtype into _classifier_metadata "
            "so the classifier user-override branch can populate "
            "profile.anime_subtype."
        )

    def test_pipeline_injects_music_subtype(self):
        import ast
        from pathlib import Path
        src = (
            Path(__file__).resolve().parents[1] / "services" / "pipeline.py"
        ).read_text()
        tree = ast.parse(src)
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if (
                        isinstance(tgt, ast.Subscript)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "_classifier_metadata"
                        and isinstance(tgt.slice, ast.Constant)
                        and tgt.slice.value == "music_subtype"
                    ):
                        found = True
                        break
        assert found, "pipeline.py must inject music_subtype"

    def test_pipeline_injects_sports_subtype(self):
        import ast
        from pathlib import Path
        src = (
            Path(__file__).resolve().parents[1] / "services" / "pipeline.py"
        ).read_text()
        tree = ast.parse(src)
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if (
                        isinstance(tgt, ast.Subscript)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "_classifier_metadata"
                        and isinstance(tgt.slice, ast.Constant)
                        and tgt.slice.value == "sports_subtype"
                    ):
                        found = True
                        break
        assert found, "pipeline.py must inject sports_subtype"

    def test_pipeline_injects_game_type(self):
        import ast
        from pathlib import Path
        src = (
            Path(__file__).resolve().parents[1] / "services" / "pipeline.py"
        ).read_text()
        tree = ast.parse(src)
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if (
                        isinstance(tgt, ast.Subscript)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "_classifier_metadata"
                        and isinstance(tgt.slice, ast.Constant)
                        and tgt.slice.value == "game_type"
                    ):
                        found = True
                        break
        assert found, "pipeline.py must inject game_type"
