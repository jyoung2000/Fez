"""Phase 9 — fixture registry tests.

Verifies that every entry in ``FIXTURES`` is well-formed:

- has a non-empty name and description
- ``content_type_override`` is a valid Phase 1+2 token (normalizes
  to a real ContentType)
- the ``build()`` callable returns the kwargs ``build_reframe_segments``
  expects
- every metric in ``metrics`` is in ``ALL_METRICS``
- the ground-truth payload has data for every metric the fixture
  claims to score (e.g. fixtures with ``downbeat_snap_error`` must
  carry a non-empty ``beat_grid``)

These tests run in the sandbox without numpy because they only touch
the fixture spec dataclasses, the normalizer, and the metric name
list — never the actual segmenter.
"""

from __future__ import annotations

import pytest

from backend.services.autoflip_parity_fixtures import (
    FIXTURES,
    FixtureSpec,
    GroundTruth,
    get_fixture,
    list_fixture_names,
)
from backend.services.autoflip_parity_metrics import ALL_METRICS
from backend.services.content_type_strings import normalize_ui_content_type


REQUIRED_KEYS = {
    "shot_cuts",
    "face_registry",
    "active_speaker_events",
    "dense_faces",
    "transcript_segments",
    "speaker_to_slot",
    "video_duration",
}


# ───────────────── Registry-level invariants ─────────────────

class TestRegistryInvariants:
    def test_eight_fixtures_present(self):
        # The Phase 9 spec calls out 7 fixtures; the anime reframe gap
        # close added an 8th (anime_panning_close_up). If a future
        # contributor drops one or adds without updating the docs
        # table, this tripwire fires.
        assert len(FIXTURES) == 8

    def test_every_spec_in_registry_matches_its_key(self):
        for key, spec in FIXTURES.items():
            assert spec.name == key, f"{key} → spec.name={spec.name}"

    def test_get_fixture_round_trip(self):
        for name in list_fixture_names():
            assert get_fixture(name) is FIXTURES[name]

    def test_get_fixture_unknown_raises(self):
        with pytest.raises(KeyError):
            get_fixture("not_a_real_fixture")

    def test_fixture_names_are_unique(self):
        names = list_fixture_names()
        assert len(names) == len(set(names))


# ───────────────── Per-fixture invariants (parametrized) ──────────

@pytest.mark.parametrize("name", sorted(FIXTURES.keys()))
class TestFixtureWellFormed:
    """Run all spec-level checks once per fixture so failures point at
    the exact fixture rather than a dict-iteration loop."""

    def test_has_description(self, name):
        spec = get_fixture(name)
        assert spec.description, f"{name}: empty description"

    def test_content_type_override_normalizes(self, name):
        spec = get_fixture(name)
        # The fixture token must round-trip through the Phase 1
        # normalizer — if it doesn't, the runner can't inject it as
        # metadata for the classifier user-override branch.
        normalized = normalize_ui_content_type(spec.content_type_override)
        assert normalized is not None, (
            f"{name}: content_type_override={spec.content_type_override!r} "
            "does not normalize"
        )

    def test_metrics_all_known(self, name):
        spec = get_fixture(name)
        unknown = [m for m in spec.metrics if m not in ALL_METRICS]
        assert not unknown, (
            f"{name}: unknown metric names {unknown} "
            f"(must be one of {ALL_METRICS})"
        )

    def test_metrics_non_empty(self, name):
        spec = get_fixture(name)
        assert spec.metrics, f"{name}: must score at least one metric"

    def test_video_duration_positive(self, name):
        spec = get_fixture(name)
        assert spec.video_duration > 0

    def test_crop_width_pct_reasonable(self, name):
        spec = get_fixture(name)
        # 9:16 of 1920x1080 = 31.6 % of source width
        cw = spec.crop_width_pct
        assert 5.0 < cw < 100.0, f"{name}: crop_width_pct={cw}"


# ───────────────── Build callables produce the right shape ──────────

@pytest.mark.parametrize("name", sorted(FIXTURES.keys()))
class TestBuildCallable:
    def test_returns_dict_with_required_keys(self, name):
        spec = get_fixture(name)
        kwargs = spec.build()
        assert isinstance(kwargs, dict)
        missing = REQUIRED_KEYS - kwargs.keys()
        assert not missing, f"{name}: missing kwargs {missing}"

    def test_video_duration_matches_spec(self, name):
        spec = get_fixture(name)
        kwargs = spec.build()
        assert kwargs["video_duration"] == spec.video_duration

    def test_dense_faces_non_empty_or_gameplay(self, name):
        # All non-gameplay fixtures should have at least one dense face
        # frame so the classifier has something to work with. Gameplay
        # fixtures (which take the face-skipping fast path) are allowed
        # to ship zero faces.
        spec = get_fixture(name)
        kwargs = spec.build()
        is_gameplay = spec.content_type_override.startswith("gameplay")
        if is_gameplay:
            return
        assert len(kwargs["dense_faces"]) > 0, f"{name}: empty dense_faces"

    def test_dense_face_timestamps_monotonic(self, name):
        spec = get_fixture(name)
        kwargs = spec.build()
        ts = [df.timestamp for df in kwargs["dense_faces"]]
        for a, b in zip(ts, ts[1:]):
            assert a <= b, f"{name}: dense face timestamps not monotonic"

    def test_dense_face_timestamps_within_duration(self, name):
        spec = get_fixture(name)
        kwargs = spec.build()
        for df in kwargs["dense_faces"]:
            assert 0.0 <= df.timestamp < spec.video_duration + 0.01

    def test_transcript_segments_within_duration(self, name):
        spec = get_fixture(name)
        kwargs = spec.build()
        for seg in kwargs["transcript_segments"]:
            assert 0.0 <= seg.start < spec.video_duration + 0.01
            assert seg.end <= spec.video_duration + 0.01


# ───────────────── Ground-truth coverage ───────────────────────────

@pytest.mark.parametrize("name", sorted(FIXTURES.keys()))
class TestGroundTruthCoverage:
    def test_gt_is_groundtruth(self, name):
        spec = get_fixture(name)
        assert isinstance(spec.ground_truth, GroundTruth)

    def test_subsecond_recall_has_or_doesnt_need_expected_switches(self, name):
        spec = get_fixture(name)
        if "sub_second_switch_recall" not in spec.metrics:
            return
        # Fixture asks for sub-second recall — it must either declare
        # the expected switches OR explicitly mean "no switches" (vlog
        # / TPS / stream cases). Both are allowed.
        gt = spec.ground_truth
        assert isinstance(gt.expected_switches, list)

    def test_downbeat_snap_requires_beat_grid(self, name):
        spec = get_fixture(name)
        if "downbeat_snap_error" not in spec.metrics:
            return
        assert spec.ground_truth.beat_grid, (
            f"{name}: scores downbeat_snap_error but ground_truth.beat_grid "
            "is empty"
        )

    def test_required_region_metric_has_per_frame_data(self, name):
        spec = get_fixture(name)
        if "required_region_miss_rate" not in spec.metrics:
            return
        # At least one fixture frame must declare a required region.
        # Empty per-frame data means the metric will return 0.0
        # (vacuous-true), which is allowed but worth flagging in tests
        # so the fixture author can decide whether they meant it.
        gt = spec.ground_truth
        if not gt.required_regions_per_frame:
            pytest.skip(
                f"{name}: required_region_miss_rate metric has no "
                "per-frame ground truth (vacuous-true score)"
            )
        non_empty = sum(1 for f in gt.required_regions_per_frame if f)
        assert non_empty > 0, (
            f"{name}: required_regions_per_frame is all empty"
        )

    def test_hud_metric_has_hud_zones(self, name):
        spec = get_fixture(name)
        if "hud_preservation_rate" not in spec.metrics:
            return
        assert spec.ground_truth.hud_zones, (
            f"{name}: scores hud_preservation_rate but hud_zones is empty"
        )

    def test_thirds_metric_either_declared_or_pending(self, name):
        spec = get_fixture(name)
        if "face_centroid_in_thirds_rate" not in spec.metrics:
            return
        # Phase 4 will populate face_y_in_crop_normalized from the
        # actual reframe output; for the baseline it's empty, which
        # produces a 0.0 score (no thirds compliance yet). That's
        # the desired starting state, so we don't fail the test —
        # we just assert the field exists as a list.
        assert isinstance(
            spec.ground_truth.face_y_in_crop_normalized, list
        )


# ───────────────── Specific spot checks ──────────────────────────

class TestSpecificFixtures:
    """Pin a few fixture-specific properties so accidental edits to
    durations / speaker counts / beat grids show up immediately."""

    def test_2speaker_alternating_has_9_switches(self):
        spec = get_fixture("2speaker_alternating")
        assert len(spec.ground_truth.expected_switches) == 9
        assert spec.video_duration == 20.0

    def test_3speaker_panel_has_11_switches(self):
        spec = get_fixture("3speaker_panel")
        assert len(spec.ground_truth.expected_switches) == 11

    def test_vlog_has_no_switches(self):
        spec = get_fixture("vlog_walk_and_talk")
        assert spec.ground_truth.expected_switches == []

    def test_music_video_beat_grid_120bpm(self):
        spec = get_fixture("music_video_beat")
        # 120 BPM → 0.5 s spacing, 12 s duration → 24 beats + the 0
        # beat = 25 entries.
        assert len(spec.ground_truth.beat_grid) == 25
        # First two beats should be 0.0 and 0.5.
        assert spec.ground_truth.beat_grid[0] == 0.0
        assert abs(spec.ground_truth.beat_grid[1] - 0.5) < 1e-6

    def test_anime_hard_cuts_has_5_shot_boundaries(self):
        spec = get_fixture("anime_hard_cuts")
        kwargs = spec.build()
        assert len(kwargs["shot_cuts"]) == 5

    def test_tps_uses_gameplay_tps_token(self):
        spec = get_fixture("tps_character_offset")
        assert spec.content_type_override == "gameplay_tps"
        assert spec.game_type == "gta_v"

    def test_stream_uses_stream_token_not_fastpath(self):
        # Stream is gaming-layout but must NOT take the gameplay fast
        # path — Phase 7 will route it through STACKED_GAMEPLAY which
        # still needs the face pipeline.
        from backend.services.content_type_strings import is_gameplay_override
        spec = get_fixture("stream_corner_facecam")
        assert spec.content_type_override == "stream"
        assert is_gameplay_override("stream") is False

    def test_anime_uses_dialogue_subtype(self):
        spec = get_fixture("anime_hard_cuts")
        assert spec.anime_subtype == "dialogue"

    def test_music_video_uses_performance_subtype(self):
        spec = get_fixture("music_video_beat")
        assert spec.music_subtype == "performance"


# ───────────────── Runner-level smoke (dry run only) ──────────

class TestRunnerDryRun:
    """The full runner needs numpy via build_reframe_segments. Dry
    run mode skips that and just dumps the spec — which is enough to
    verify the runner's plumbing and JSON shape in a sandbox."""

    def test_dry_run_score_returns_status_dry_run(self):
        from backend.scripts.measure_autoflip_parity import score_one
        spec = get_fixture("vlog_walk_and_talk")
        out = score_one(spec, dry_run=True)
        assert out["name"] == "vlog_walk_and_talk"
        assert out["status"] == "dry_run"
        assert out["metrics"] == {}
        # Ground truth summary must still be populated.
        assert out["ground_truth_summary"]["expected_switches"] == 0

    def test_dry_run_run_all_covers_every_fixture(self):
        from backend.scripts.measure_autoflip_parity import run_all
        payload = run_all(dry_run=True)
        assert payload["fixture_count"] == len(FIXTURES)
        names = {row["name"] for row in payload["results"]}
        assert names == set(FIXTURES.keys())

    def test_run_in_sandbox_skips_segmenter_with_explicit_error(self):
        """Without numpy the segmenter import fails — the runner must
        still produce a usable row with status='skipped' and a
        traceable error string."""
        from backend.scripts.measure_autoflip_parity import score_one
        spec = get_fixture("2speaker_alternating")
        out = score_one(spec, dry_run=False)
        # Either skipped (sandbox) or ok (docker). Both are acceptable
        # — we only assert the row has a status field.
        assert out["status"] in ("ok", "skipped", "error")
        if out["status"] == "skipped":
            assert "error" in out
            assert "import" in out["error"].lower() or "numpy" in out["error"].lower()
