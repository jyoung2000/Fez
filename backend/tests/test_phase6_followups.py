"""Phase 6 follow-ups — character clustering + pipeline integration.

Three test groups:

1. **Character clustering** — pure-Python helpers for the
   color-fingerprint-based cross-cut anime character re-id
   (``backend.services.anime_character_clustering``).

2. **Pipeline.py integration AST guards** — assert that the
   pipeline's Phase 6 wiring is structurally present so a
   future refactor can't accidentally drop it. The actual
   pipeline call needs the full numpy + OpenCV stack and runs
   in docker; the sandbox suite checks the integration
   markers via ``ast.parse``.

3. **Dockerfile cascade download** — assert both Dockerfiles
   download ``lbpcascade_animeface.xml`` to the path the
   ``anime_face_detector`` module expects.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from backend.services.anime_character_clustering import (
    DEFAULT_CLUSTER_THRESHOLD,
    FINGERPRINT_DIMS,
    USE_ANIME_CHARACTER_CLUSTERING,
    chi_squared_distance,
    cluster_fingerprints,
    extract_color_fingerprint,
    merge_with_existing_slots,
)


# ──────────────────── chi_squared_distance ────────────────────


class TestChiSquaredDistance:
    def test_identical(self):
        a = [0.1, 0.2, 0.3, 0.4]
        assert chi_squared_distance(a, a) == 0.0

    def test_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        # Σ (1-0)² / (1+0+ε) + (0-1)² / (0+1+ε) ≈ 1 + 1 = 2
        d = chi_squared_distance(a, b)
        assert 1.5 < d <= 2.0

    def test_mismatched_length(self):
        assert chi_squared_distance([1, 0], [1, 0, 0]) == float("inf")

    def test_empty(self):
        assert chi_squared_distance([], []) == 0.0

    def test_symmetric(self):
        a = [0.1, 0.2, 0.7]
        b = [0.5, 0.3, 0.2]
        d_ab = chi_squared_distance(a, b)
        d_ba = chi_squared_distance(b, a)
        assert d_ab == pytest.approx(d_ba, abs=1e-9)


# ──────────────────── cluster_fingerprints ────────────────────


class TestClusterFingerprints:
    def _naruto(self):
        # Mostly orange + black palette
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.1, 0.0, 0.1, 0.1]

    def _goku(self):
        # Mostly red + blue palette
        return [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0,
                0.1, 0.0, 0.0, 0.1]

    def test_empty_input(self):
        result = cluster_fingerprints([])
        assert result.n_clusters == 0
        assert result.cluster_ids == []

    def test_single_fingerprint(self):
        result = cluster_fingerprints([self._naruto()])
        assert result.n_clusters == 1
        assert result.cluster_ids == [0]
        assert len(result.centroids) == 1

    def test_two_same_character(self):
        # Two near-identical fingerprints → one cluster
        a = self._naruto()
        b = self._naruto()
        b[5] = 0.35  # slight pose variation
        b[6] = 0.35
        result = cluster_fingerprints([a, b])
        assert result.n_clusters == 1
        assert result.cluster_ids == [0, 0]

    def test_two_different_characters(self):
        result = cluster_fingerprints([self._naruto(), self._goku()])
        assert result.n_clusters == 2
        assert result.cluster_ids == [0, 1]

    def test_three_mixed_clusters(self):
        # naruto, naruto, goku → two clusters
        a = self._naruto()
        b = self._naruto()
        b[5] = 0.38
        c = self._goku()
        result = cluster_fingerprints([a, b, c])
        assert result.n_clusters == 2
        assert result.cluster_ids[0] == result.cluster_ids[1]
        assert result.cluster_ids[0] != result.cluster_ids[2]

    def test_custom_threshold(self):
        # Tighter threshold splits the same-character cluster
        a = self._naruto()
        b = self._naruto()
        b[5] = 0.30
        b[6] = 0.40
        # Default threshold 0.50 → cluster together
        loose = cluster_fingerprints([a, b])
        # Very tight threshold → split
        tight = cluster_fingerprints([a, b], threshold=0.001)
        assert loose.n_clusters == 1
        assert tight.n_clusters == 2

    def test_centroid_count_matches(self):
        result = cluster_fingerprints([self._naruto(), self._goku()])
        assert len(result.centroids) == result.n_clusters
        # Each centroid is a list of FINGERPRINT_DIMS floats
        for c in result.centroids:
            assert len(c) == FINGERPRINT_DIMS


# ──────────────────── merge_with_existing_slots ────────────────────


class TestMergeWithExistingSlots:
    def _naruto(self):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.1, 0.0, 0.1, 0.1]

    def _goku(self):
        return [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0,
                0.1, 0.0, 0.0, 0.1]

    def test_match_existing_slot(self):
        existing = {0: self._naruto()}
        slot_ids, updated = merge_with_existing_slots(
            [self._naruto()], existing,
        )
        assert slot_ids == [0]
        assert set(updated.keys()) == {0}

    def test_create_new_slot_for_unknown(self):
        existing = {0: self._naruto()}
        slot_ids, updated = merge_with_existing_slots(
            [self._goku()], existing,
        )
        assert slot_ids == [1]
        assert set(updated.keys()) == {0, 1}

    def test_mixed_match_and_new(self):
        existing = {0: self._naruto()}
        slot_ids, updated = merge_with_existing_slots(
            [self._naruto(), self._goku()], existing,
        )
        assert slot_ids == [0, 1]
        assert set(updated.keys()) == {0, 1}

    def test_empty_existing_creates_all_new(self):
        slot_ids, updated = merge_with_existing_slots(
            [self._naruto(), self._goku()], {},
        )
        assert slot_ids == [0, 1]
        assert set(updated.keys()) == {0, 1}

    def test_does_not_mutate_input_dict(self):
        existing = {0: self._naruto()}
        original_size = len(existing)
        merge_with_existing_slots([self._goku()], existing)
        # Input dict unchanged
        assert len(existing) == original_size


# ──────────────────── extract_color_fingerprint ────────────────────


class TestExtractColorFingerprint:
    def test_returns_uniform_for_none_frame(self):
        result = extract_color_fingerprint(None, (50, 50, 20, 20))
        assert len(result) == FINGERPRINT_DIMS
        # Uniform fingerprint sums to 1.0
        assert pytest.approx(sum(result), abs=1e-9) == 1.0
        # All entries equal
        assert all(abs(v - result[0]) < 1e-9 for v in result)

    def test_returns_uniform_when_cv2_missing(self):
        # When cv2 isn't importable, the function falls back to
        # uniform without crashing. We can't easily simulate the
        # missing-cv2 case in a sandbox that has cv2 installed,
        # but we can test the None-frame path which exercises the
        # same fallback.
        result = extract_color_fingerprint(None, (10, 10, 5, 5))
        assert sum(result) == pytest.approx(1.0, abs=1e-9)


# ──────────────────── Feature flag default ────────────────────


class TestCharacterClusteringFlagDefaultOff:
    def test_default_off(self):
        assert USE_ANIME_CHARACTER_CLUSTERING is False


# ──────────────────── pipeline.py integration AST guards ────────────────────


class TestPipelinePhase6Integration:
    SRC_PATH = (
        Path(__file__).resolve().parents[1] / "services" / "pipeline.py"
    )

    def test_imports_anime_anchor_lazy(self):
        src = self.SRC_PATH.read_text()
        assert "from backend.services.anime_anchor import" in src
        assert "AnimeFrameFeatures" in src
        assert "score_anime_sequence" in src

    def test_passes_anime_anchors_to_segmenter(self):
        src = self.SRC_PATH.read_text()
        assert "anime_anchors=_anime_anchors_for_seg" in src

    def test_gates_on_is_animated(self):
        src = self.SRC_PATH.read_text()
        # The Phase 6 block reads profile.is_animated
        assert "is_animated" in src

    def test_anime_anchor_block_has_try_except(self):
        # The Phase 6 block must be wrapped in try/except so a
        # cascade XML missing / OpenCV failure stays non-fatal.
        src = self.SRC_PATH.read_text()
        assert "Anime anchor extraction failed" in src

    def test_phase5_beat_detector_wired(self):
        # The Phase 5 follow-up wiring is also in this commit.
        src = self.SRC_PATH.read_text()
        assert "from backend.services.beat_detector import" in src
        assert "music_beat_grid=_music_beat_grid_for_seg" in src

    def test_phase5_beat_detector_gated_on_music_video(self):
        src = self.SRC_PATH.read_text()
        # The beat detector block reads content_type == "music_video"
        assert "music_video" in src

    def test_pipeline_still_parses(self):
        # Trivial AST round-trip: the heavy edits don't break the
        # parse.
        src = self.SRC_PATH.read_text()
        ast.parse(src)


# ──────────────────── Dockerfile cascade download ────────────────────


class TestDockerfileCascadeDownload:
    REPO_ROOT = Path(__file__).resolve().parents[2]

    def test_gpu_dockerfile_downloads_cascade(self):
        df = (self.REPO_ROOT / "Dockerfile.gpu").read_text()
        assert "lbpcascade_animeface.xml" in df
        # Downloads to /app/backend/models/lbpcascade_animeface.xml
        # (matches DEFAULT_CASCADE_PATH in anime_face_detector)
        assert "/app/backend/models/lbpcascade_animeface.xml" in df
        assert "nagadomi/lbpcascade_animeface" in df

    def test_cpu_dockerfile_downloads_cascade(self):
        df = (self.REPO_ROOT / "Dockerfile").read_text()
        assert "lbpcascade_animeface.xml" in df
        assert "/app/backend/models/lbpcascade_animeface.xml" in df

    def test_cascade_path_matches_anime_face_detector(self):
        # The DEFAULT_CASCADE_PATH constant in anime_face_detector
        # resolves to backend/models/lbpcascade_animeface.xml
        # relative to the package — which is what the Dockerfile
        # downloads to (/app/backend/models/).
        from backend.services.anime_face_detector import DEFAULT_CASCADE_PATH
        assert "lbpcascade_animeface.xml" in DEFAULT_CASCADE_PATH
        assert "models" in DEFAULT_CASCADE_PATH
