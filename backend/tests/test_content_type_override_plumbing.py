"""Phase 1 — content-type override plumbing.

Regression suite for the UI→enum→classifier plumbing. The upload
dropdown sends strings like "gameplay", "movie", "podcast" (plus the
Phase 2 values "debate", "vlog", "anime", "music_video",
"gameplay_moba", "gameplay_tps", "gameplay_racing", "stream",
"sports", "cartoon", "panel", "interview", "narrative", "cinematic").
The classifier's user-override branch previously only fired for
"podcast" by coincidence, because the UI strings didn't match the
``ContentType`` enum values ("gameplay" vs "gaming", "movie" vs
"narrative"). These tests lock the fix in.

The Phase 1 spec called out 5 test cases covering every UI string —
we ship an expanded suite (20+ cases) covering the normalizer, the
pipeline gameplay fast-path, and the classifier override branch end
to end.

These tests deliberately avoid imports that need numpy / OpenCV / DNS
so they run in a minimal sandbox. The pipeline-level paths are covered
via AST inspection (metadata injection) and unit tests on the
normalizer + ``classify_content`` rather than by spinning up the full
async pipeline.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from backend.services.content_type_config import ContentType
from backend.services.content_type_strings import (
    NormalizedContentType,
    is_gameplay_override,
    is_user_override,
    normalize_ui_content_type,
)
from backend.services.content_classifier import classify_content


# ─────────────────────── Normalizer — UI token → enum ───────────────────────

class TestNormalizeLegacyAliases:
    """The three UI strings that have always existed must keep working."""

    def test_gameplay_maps_to_gaming(self):
        n = normalize_ui_content_type("gameplay")
        assert n is not None
        assert n.content_type == ContentType.GAMING
        assert n.is_gameplay_fastpath is True
        assert n.is_multi_speaker_panel is False
        assert n.is_animated is False
        assert n.raw == "gameplay"

    def test_movie_maps_to_narrative(self):
        n = normalize_ui_content_type("movie")
        assert n is not None
        assert n.content_type == ContentType.NARRATIVE
        assert n.is_gameplay_fastpath is False

    def test_podcast_maps_to_podcast(self):
        n = normalize_ui_content_type("podcast")
        assert n is not None
        assert n.content_type == ContentType.PODCAST
        assert n.is_multi_speaker_panel is False


class TestNormalizePhase2Tokens:
    """Forward-compat tokens reserved for the Phase 2 dropdown."""

    @pytest.mark.parametrize("token", ["debate", "panel"])
    def test_debate_and_panel_set_panel_flag(self, token):
        n = normalize_ui_content_type(token)
        assert n.content_type == ContentType.PODCAST
        assert n.is_multi_speaker_panel is True

    def test_vlog(self):
        n = normalize_ui_content_type("vlog")
        assert n.content_type == ContentType.VLOG

    def test_interview_maps_to_podcast(self):
        n = normalize_ui_content_type("interview")
        assert n.content_type == ContentType.PODCAST
        assert n.is_multi_speaker_panel is False

    def test_narrative_and_cinematic(self):
        assert normalize_ui_content_type("narrative").content_type == ContentType.NARRATIVE
        assert normalize_ui_content_type("cinematic").content_type == ContentType.NARRATIVE

    @pytest.mark.parametrize("token", ["anime", "cartoon"])
    def test_anime_sets_animated_flag(self, token):
        n = normalize_ui_content_type(token)
        assert n.content_type == ContentType.ANIME
        assert n.is_animated is True

    def test_music_video(self):
        n = normalize_ui_content_type("music_video")
        assert n.content_type == ContentType.MUSIC_VIDEO
        assert n.is_animated is False

    @pytest.mark.parametrize(
        "token",
        ["gameplay_fps", "gameplay_moba", "gameplay_tps", "gameplay_racing"],
    )
    def test_gameplay_variants_all_route_to_gaming_fastpath(self, token):
        n = normalize_ui_content_type(token)
        assert n.content_type == ContentType.GAMING
        assert n.is_gameplay_fastpath is True

    def test_stream_is_gaming_but_not_fastpath(self):
        # Stream is gaming-layout but has a facecam that still needs
        # the face pipeline — it must NOT take the face-skipping fast
        # path. Phase 7 handles the STACKED_GAMEPLAY layout routing.
        n = normalize_ui_content_type("stream")
        assert n.content_type == ContentType.GAMING
        assert n.is_gameplay_fastpath is False

    def test_sports(self):
        n = normalize_ui_content_type("sports")
        assert n.content_type == ContentType.SPORTS


class TestNormalizeNullAndInvalid:
    @pytest.mark.parametrize("token", [None, "", "   ", "auto", "unknown", "AUTO"])
    def test_null_tokens_return_none(self, token):
        assert normalize_ui_content_type(token) is None

    @pytest.mark.parametrize("token", ["nonsense", "gamepley", "FOOBAR", "podcas"])
    def test_invalid_tokens_return_none(self, token):
        assert normalize_ui_content_type(token) is None


class TestNormalizeFormatTolerance:
    def test_leading_trailing_whitespace(self):
        assert normalize_ui_content_type("  GAMEPLAY  ").content_type == ContentType.GAMING

    def test_case_insensitive(self):
        assert normalize_ui_content_type("Podcast").content_type == ContentType.PODCAST
        assert normalize_ui_content_type("MOVIE").content_type == ContentType.NARRATIVE
        assert normalize_ui_content_type("Anime").is_animated is True


# ────────────────────── is_gameplay_override helper ──────────────────────

class TestIsGameplayOverride:
    @pytest.mark.parametrize(
        "token",
        ["gameplay", "gameplay_fps", "gameplay_moba", "gameplay_tps", "gameplay_racing"],
    )
    def test_gameplay_variants_are_fast_path(self, token):
        assert is_gameplay_override(token) is True

    def test_stream_is_not_fast_path(self):
        assert is_gameplay_override("stream") is False

    @pytest.mark.parametrize(
        "token",
        ["podcast", "movie", "debate", "vlog", "anime", "music_video", "sports",
         "narrative", "cinematic", "", None, "auto", "nonsense"],
    )
    def test_non_gameplay_tokens_are_not_fast_path(self, token):
        assert is_gameplay_override(token) is False


class TestIsUserOverride:
    @pytest.mark.parametrize(
        "token",
        ["gameplay", "movie", "podcast", "debate", "vlog", "anime",
         "music_video", "stream", "sports"],
    )
    def test_recognized_tokens(self, token):
        assert is_user_override(token) is True

    @pytest.mark.parametrize("token", [None, "", "auto", "unknown", "nonsense"])
    def test_non_overrides(self, token):
        assert is_user_override(token) is False


# ───────────────── classify_content respects the override ─────────────────

class TestClassifyContentOverride:
    """``classify_content`` must route on UI tokens via metadata injection.

    These cases call ``classify_content`` with empty face / scene /
    shot data — the heuristic branch would normally return ``unknown``
    or a low-confidence guess. With a valid override the branch must
    short-circuit to ``confidence=1.0`` on the correct enum value.
    """

    def _classify(self, override):
        return classify_content(
            shot_cuts=[],
            face_registry=None,
            dense_faces=[],
            scenes=[],
            video_duration=60.0,
            metadata={"content_type_override": override},
        )

    def test_legacy_gameplay_routes_to_gaming(self):
        p = self._classify("gameplay")
        assert p.content_type == ContentType.GAMING.value
        assert p.confidence == 1.0

    def test_legacy_movie_routes_to_narrative(self):
        p = self._classify("movie")
        assert p.content_type == ContentType.NARRATIVE.value
        assert p.confidence == 1.0

    def test_legacy_podcast_routes_to_podcast(self):
        p = self._classify("podcast")
        assert p.content_type == ContentType.PODCAST.value
        assert p.confidence == 1.0

    def test_debate_sets_panel_flag_on_profile(self):
        p = self._classify("debate")
        assert p.content_type == ContentType.PODCAST.value
        assert p.is_multi_speaker_panel is True
        assert p.confidence == 1.0

    def test_anime_sets_animated_flag_on_profile(self):
        p = self._classify("anime")
        assert p.content_type == ContentType.ANIME.value
        assert p.is_animated is True

    def test_music_video(self):
        p = self._classify("music_video")
        assert p.content_type == ContentType.MUSIC_VIDEO.value

    def test_vlog(self):
        p = self._classify("vlog")
        assert p.content_type == ContentType.VLOG.value

    def test_stream_routes_to_gaming(self):
        p = self._classify("stream")
        assert p.content_type == ContentType.GAMING.value
        assert p.confidence == 1.0

    def test_sports(self):
        p = self._classify("sports")
        assert p.content_type == ContentType.SPORTS.value

    def test_gameplay_variants_route_to_gaming(self):
        for token in ("gameplay_fps", "gameplay_moba", "gameplay_tps", "gameplay_racing"):
            p = self._classify(token)
            assert p.content_type == ContentType.GAMING.value, token

    def test_legacy_metadata_content_type_key_still_works(self):
        """Old callers pass ``content_type`` instead of ``content_type_override``."""
        p = classify_content(
            shot_cuts=[],
            face_registry=None,
            dense_faces=[],
            scenes=[],
            video_duration=60.0,
            metadata={"content_type": "podcast"},
        )
        assert p.content_type == ContentType.PODCAST.value
        assert p.confidence == 1.0

    def test_reframe_style_legacy_key_still_works(self):
        p = classify_content(
            shot_cuts=[],
            face_registry=None,
            dense_faces=[],
            scenes=[],
            video_duration=60.0,
            metadata={"reframe_style": "movie"},
        )
        assert p.content_type == ContentType.NARRATIVE.value

    def test_invalid_override_falls_through_to_heuristic(self):
        """An unknown token must NOT crash — it falls to the heuristic path."""
        p = classify_content(
            shot_cuts=[],
            face_registry=None,
            dense_faces=[],
            scenes=[],
            video_duration=60.0,
            metadata={"content_type_override": "nonsense"},
        )
        # Heuristic path ran — confidence is NOT the 1.0 user-override sentinel.
        assert p.confidence != 1.0

    def test_no_metadata_does_not_crash(self):
        p = classify_content(
            shot_cuts=[],
            face_registry=None,
            dense_faces=[],
            scenes=[],
            video_duration=60.0,
            metadata=None,
        )
        # Heuristic path ran — confidence is NOT the user-override sentinel.
        assert p.confidence != 1.0


# ──────────────── pipeline.py metadata-injection plumbing ────────────────

class TestPipelineMetadataInjection:
    """AST check: pipeline.py must inject the override into a dict it
    passes to ``classify_content`` so the user-override branch fires.

    We don't run the full pipeline here (it requires numpy, a real
    video, etc.) — we instead parse pipeline.py and assert the
    injection exists as a literal key assignment to
    ``_classifier_metadata``. That catches a future refactor that
    accidentally drops the injection.
    """

    PIPELINE_PATH = Path(__file__).resolve().parents[1] / "services" / "pipeline.py"

    def test_pipeline_has_metadata_injection(self):
        src = self.PIPELINE_PATH.read_text()
        tree = ast.parse(src)
        found_injection = False
        # Look for any Subscript store of the key
        # 'content_type_override' on a Name target named
        # '_classifier_metadata'.
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name):
                        if tgt.value.id != "_classifier_metadata":
                            continue
                        slc = tgt.slice
                        if isinstance(slc, ast.Constant) and slc.value == "content_type_override":
                            found_injection = True
                            break
        assert found_injection, (
            "pipeline.py must inject _content_override into "
            "_classifier_metadata['content_type_override'] before "
            "calling classify_content. Without this, the user's "
            "UI choice silently never reaches the classifier."
        )

    def test_pipeline_uses_is_gameplay_override_helper(self):
        """The pipeline's gameplay fast-path must use the normalizer's
        helper, NOT a hard-coded string literal comparison."""
        src = self.PIPELINE_PATH.read_text()
        # Must import is_gameplay_override
        assert "is_gameplay_override" in src, (
            "pipeline.py must import is_gameplay_override from "
            "content_type_strings so legacy 'gameplay' and the new "
            "gameplay_* variants all trigger the gameplay fast-path."
        )
        # Must NOT compare _content_override to a bare 'gameplay' literal
        # in the fast-path decision. Allow 'gameplay' to appear in log
        # messages but require the decision to go through the helper.
        tree = ast.parse(src)
        bad = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                if isinstance(node.left, ast.Name) and node.left.id == "_content_override":
                    for cmp_val in node.comparators:
                        if isinstance(cmp_val, ast.Constant) and cmp_val.value == "gameplay":
                            bad.append(node.lineno)
        assert not bad, (
            f"pipeline.py compares _content_override == 'gameplay' literally "
            f"at line(s) {bad}. Use is_gameplay_override() instead so the "
            f"new gameplay variants (gameplay_fps / gameplay_moba / etc.) "
            f"also route to the fast path."
        )
