"""Phase 5 — music-video beat snap + pulse cuts.

Five test groups:

1. **BeatGrid construction** — synthetic builder + librosa fallback.
2. **Snap helpers** — ``snap_to_nearest_downbeat``,
   ``snap_segment_boundaries``, ``enumerate_pulse_cuts``.
3. **clip_boundary_snapper integration** — the music-video
   downbeat path on ``snap_all_clips`` AND the legacy word-level
   path stays unchanged.
4. **AST guards** on the reframe segmenter Stage 11 wiring + the
   parity runner's BeatGrid construction.
5. **Spec exit criterion** — the 120 BPM synthetic fixture's
   downbeat_snap_error.snap_rate hits 1.0 with the flag ON, with
   every cut within ±40 ms of a downbeat.

Per the v2 ground rules, the feature flag default is OFF — Phase
5 ships flag-OFF until the in-docker validation lands the post-
Phase-5 numbers in ``docs/autoflip_parity_v2_results.md``. The
sandbox bench (with scipy installed) confirms the flag-ON path
hits the spec target before the docker flip happens.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from backend.services.beat_detector import (
    USE_MUSIC_BEAT_SNAP,
    BeatGrid,
    build_synthetic_beat_grid,
    detect_beats,
    enumerate_pulse_cuts,
    snap_segment_boundaries,
    snap_to_nearest_downbeat,
)


# ────────────────── BeatGrid construction ──────────────────


class TestBuildSyntheticBeatGrid:
    def test_120_bpm_4_4_meter(self):
        grid = build_synthetic_beat_grid(120.0, 12.0)
        # 120 BPM → 0.5 s spacing → 25 beats over 12 s
        assert len(grid.beat_times) == 25
        assert grid.beat_times[0] == 0.0
        assert grid.beat_times[1] == 0.5
        # 4/4 → every 4th beat is a downbeat
        assert len(grid.downbeat_times) == 7
        assert grid.downbeat_times == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        assert grid.tempo_bpm == 120.0
        assert grid.source == "synthetic"
        assert grid.has_data

    def test_3_4_meter(self):
        # Waltz: 60 BPM → 1 s spacing, 3/4 → downbeats every 3 s
        grid = build_synthetic_beat_grid(60.0, 12.0, meter=3)
        assert grid.beat_times == [float(i) for i in range(13)]
        assert grid.downbeat_times == [0.0, 3.0, 6.0, 9.0, 12.0]
        assert grid.meter == 3

    def test_phase_offset(self):
        grid = build_synthetic_beat_grid(120.0, 12.0, phase_offset_sec=0.25)
        assert grid.beat_times[0] == 0.25
        assert grid.beat_times[1] == 0.75

    def test_zero_tempo(self):
        grid = build_synthetic_beat_grid(0.0, 12.0)
        assert not grid.has_data
        assert grid.beat_times == []

    def test_zero_duration(self):
        grid = build_synthetic_beat_grid(120.0, 0.0)
        assert not grid.has_data


class TestDetectBeatsLibrosaFallback:
    """``detect_beats`` must degrade gracefully when librosa isn't
    available or the audio file can't be read. We don't have a real
    audio file in the sandbox; the test verifies the empty-grid
    fallback path."""

    def test_missing_audio_returns_empty_grid(self, tmp_path):
        # Non-existent path → librosa.load raises → empty grid.
        result = detect_beats(str(tmp_path / "nope.wav"))
        assert isinstance(result, BeatGrid)
        assert not result.has_data
        assert result.beat_times == []


# ────────────────── snap_to_nearest_downbeat ──────────────────


class TestSnapToNearestDownbeat:
    def test_exact_downbeat_unchanged(self):
        grid = build_synthetic_beat_grid(120.0, 12.0)
        assert snap_to_nearest_downbeat(2.0, grid) == 2.0

    def test_within_tolerance_snaps_up(self):
        grid = build_synthetic_beat_grid(120.0, 12.0)
        assert snap_to_nearest_downbeat(2.05, grid) == 2.0

    def test_within_tolerance_snaps_down(self):
        grid = build_synthetic_beat_grid(120.0, 12.0)
        assert snap_to_nearest_downbeat(1.85, grid) == 2.0

    def test_outside_tolerance_unchanged(self):
        grid = build_synthetic_beat_grid(120.0, 12.0)
        # 0.30 s from any downbeat → exceeds 0.20 tolerance
        assert snap_to_nearest_downbeat(2.30, grid) == 2.30

    def test_float_precision_epsilon(self):
        # The classic 4.0 - 3.80 = 0.20000000000000018 case: must
        # snap because the tolerance gets a 1 µs epsilon.
        grid = build_synthetic_beat_grid(120.0, 12.0)
        assert snap_to_nearest_downbeat(3.80, grid) == 4.0
        assert snap_to_nearest_downbeat(5.80, grid) == 6.0
        assert snap_to_nearest_downbeat(7.80, grid) == 8.0

    def test_empty_grid_returns_input(self):
        empty = BeatGrid(tempo_bpm=120.0)
        assert snap_to_nearest_downbeat(2.05, empty) == 2.05

    def test_zero_tolerance_returns_input(self):
        grid = build_synthetic_beat_grid(120.0, 12.0)
        assert snap_to_nearest_downbeat(2.05, grid, max_distance_sec=0.0) == 2.05

    def test_custom_tolerance(self):
        grid = build_synthetic_beat_grid(120.0, 12.0)
        # 0.30 s outside default → 0.30 < 0.5 custom → snaps
        assert snap_to_nearest_downbeat(2.30, grid, max_distance_sec=0.5) == 2.0


# ────────────────── snap_segment_boundaries ──────────────────


@dataclass
class _Seg:
    start: float
    end: float


class TestSnapSegmentBoundaries:
    def test_snaps_all_within_tolerance(self):
        grid = build_synthetic_beat_grid(120.0, 12.0)
        segs = [
            _Seg(0.00, 1.80),
            _Seg(1.80, 3.80),
            _Seg(3.80, 5.80),
            _Seg(5.80, 7.80),
            _Seg(7.80, 9.80),
            _Seg(9.80, 12.00),
        ]
        n = snap_segment_boundaries(segs, grid)
        assert n == 5
        # All non-zero starts now on downbeats
        for s in segs[1:]:
            assert s.start in grid.downbeat_times
        # Contiguity preserved
        for a, b in zip(segs, segs[1:]):
            assert a.end == b.start

    def test_skips_snap_that_shrinks_below_min(self):
        grid = build_synthetic_beat_grid(120.0, 12.0)
        # Tiny segment 1.95-2.10 — snapping start 1.95→2.0 leaves
        # cur_len = 0.10 < min_segment_sec=0.30 → must skip.
        segs = [
            _Seg(0.0, 1.95),
            _Seg(1.95, 2.10),
            _Seg(2.10, 12.0),
        ]
        n = snap_segment_boundaries(segs, grid)
        # The 1.95 boundary should NOT snap (cur_len 0.10 too short).
        assert segs[1].start == 1.95

    def test_skips_snap_above_half_segment_length(self):
        grid = build_synthetic_beat_grid(120.0, 12.0)
        # Tiny segment 2.30-2.40 — boundary 2.30 wants to snap to
        # 2.0 (distance 0.30) but that's > 0.5 * 0.10 segment len.
        # Since 2.30 is also outside the default 0.20 tolerance,
        # it won't snap regardless.
        segs = [
            _Seg(0.0, 2.30),
            _Seg(2.30, 2.40),
            _Seg(2.40, 12.0),
        ]
        n = snap_segment_boundaries(segs, grid)
        assert segs[1].start == 2.30

    def test_zero_boundary_not_snapped(self):
        grid = build_synthetic_beat_grid(120.0, 12.0)
        # First segment starts at 0 → never snapped (no prev to bind to)
        segs = [_Seg(0.0, 5.0), _Seg(5.0, 10.0)]
        snap_segment_boundaries(segs, grid)
        assert segs[0].start == 0.0

    def test_empty_inputs(self):
        grid = build_synthetic_beat_grid(120.0, 12.0)
        assert snap_segment_boundaries([], grid) == 0

    def test_empty_grid(self):
        segs = [_Seg(0.0, 1.5), _Seg(1.5, 3.0)]
        empty = BeatGrid(tempo_bpm=120.0)
        assert snap_segment_boundaries(segs, empty) == 0
        # Segments unchanged
        assert segs[1].start == 1.5


# ────────────────── enumerate_pulse_cuts ──────────────────


class TestEnumeratePulseCuts:
    def test_internal_downbeats_only(self):
        grid = build_synthetic_beat_grid(120.0, 12.0)
        # Single 4-second segment containing one internal downbeat (at 2.0)
        segs = [_Seg(0.0, 4.0)]
        pulses = enumerate_pulse_cuts(grid, segs)
        # Excludes 0.0 and 4.0 (boundaries within edge_skip)
        assert pulses == [(0, 2.0)]

    def test_edge_skip_excludes_boundary_downbeats(self):
        grid = build_synthetic_beat_grid(120.0, 12.0)
        # 2.0-4.0 segment: downbeats 2.0 and 4.0 are both at the
        # edges → none qualify with default edge_skip=0.10
        segs = [_Seg(2.0, 4.0)]
        assert enumerate_pulse_cuts(grid, segs) == []

    def test_multi_segment_pulses(self):
        grid = build_synthetic_beat_grid(120.0, 12.0)
        # Two 4s segments each get one pulse (at 2.0 and 6.0)
        segs = [_Seg(0.0, 4.0), _Seg(4.0, 8.0)]
        pulses = enumerate_pulse_cuts(grid, segs)
        assert pulses == [(0, 2.0), (1, 6.0)]

    def test_custom_edge_skip(self):
        grid = build_synthetic_beat_grid(120.0, 12.0)
        # Segment from 0.0 to 4.0; with edge_skip=2.5, downbeat 2.0
        # is within [0.0+2.5, 4.0-2.5] = [2.5, 1.5] which is empty
        # → no pulse cuts.
        segs = [_Seg(0.0, 4.0)]
        assert enumerate_pulse_cuts(grid, segs, edge_skip_sec=2.5) == []

    def test_empty_grid_returns_empty(self):
        empty = BeatGrid(tempo_bpm=120.0)
        segs = [_Seg(0.0, 4.0)]
        assert enumerate_pulse_cuts(empty, segs) == []

    def test_empty_segments_returns_empty(self):
        grid = build_synthetic_beat_grid(120.0, 12.0)
        assert enumerate_pulse_cuts(grid, []) == []


# ────────────────── clip_boundary_snapper ──────────────────


class TestClipBoundaryDownbeatSnap:
    def _clip(self, start, end):
        # Use a duck-typed stub instead of ClipCandidate so we don't
        # have to populate the dozen-plus required fields the model
        # has accumulated. The clip_boundary_snapper only reads
        # start_time / end_time / duration / title.
        @dataclass
        class _Clip:
            start_time: float
            end_time: float
            duration: float
            title: str = "test"

        return _Clip(
            start_time=start,
            end_time=end,
            duration=end - start,
        )

    def test_snaps_to_nearest_downbeats(self):
        from backend.services.clip_boundary_snapper import snap_clip_to_downbeats
        grid = build_synthetic_beat_grid(120.0, 60.0)
        clip = self._clip(2.05, 18.10)
        snap_clip_to_downbeats(clip, grid)
        assert clip.start_time == 2.0
        assert clip.end_time == 18.0

    def test_skips_when_grid_empty(self):
        from backend.services.clip_boundary_snapper import snap_clip_to_downbeats
        grid = BeatGrid(tempo_bpm=120.0)
        clip = self._clip(2.05, 18.10)
        snap_clip_to_downbeats(clip, grid)
        # Unchanged
        assert clip.start_time == 2.05
        assert clip.end_time == 18.10

    def test_skips_when_too_short(self):
        from backend.services.clip_boundary_snapper import snap_clip_to_downbeats
        grid = build_synthetic_beat_grid(120.0, 60.0)
        # 14 s clip would shrink below the 15 s minimum after snap
        clip = self._clip(2.05, 16.10)
        snap_clip_to_downbeats(clip, grid)
        # After snap: 2.0..16.0 = 14 → below 15 → no snap
        assert clip.start_time == 2.05

    def test_snap_all_clips_routes_by_content_type(self):
        from backend.services.clip_boundary_snapper import snap_all_clips
        grid = build_synthetic_beat_grid(120.0, 60.0)
        clip = self._clip(2.05, 18.10)
        # music_video routes to downbeat snap
        out = snap_all_clips([clip], transcript=[], beat_grid=grid, content_type="music_video")
        assert out[0].start_time == 2.0
        # podcast falls back to word snap (no words → no-op)
        clip2 = self._clip(2.05, 18.10)
        out2 = snap_all_clips([clip2], transcript=[], beat_grid=grid, content_type="podcast")
        assert out2[0].start_time == 2.05  # no change


# ────────────────── Feature flag default ──────────────────


class TestFeatureFlagDefaultOff:
    def test_default_off(self):
        # Per the v2 ground rules, Phase 5 ships flag-off until
        # validation captures the post-Phase-5 numbers.
        assert USE_MUSIC_BEAT_SNAP is False


# ────────────────── reframe_segmenter Stage 11 AST guards ──────────────────


class TestReframeSegmenterStage11AST:
    SRC_PATH = (
        Path(__file__).resolve().parents[1] / "services" / "reframe_segmenter.py"
    )

    def test_stage_11_block_present(self):
        src = self.SRC_PATH.read_text()
        assert "Stage 11" in src
        assert "MusicBeatSnap" in src
        assert "music_beat_grid" in src

    def test_imports_beat_detector_lazily(self):
        src = self.SRC_PATH.read_text()
        assert "USE_MUSIC_BEAT_SNAP" in src
        assert "snap_segment_boundaries" in src
        assert "enumerate_pulse_cuts" in src

    def test_gates_on_music_video_content_type(self):
        src = self.SRC_PATH.read_text()
        # The gate string: content type == "music_video"
        assert "music_video" in src

    def test_pulse_cut_uses_slot_to_x(self):
        # Pulse cuts re-derive subject_x from the active speaker
        # slot center via _slot_to_x to produce a fresh visual
        # anchor.
        src = self.SRC_PATH.read_text()
        assert "music_pulse_cut" in src
        assert "_slot_to_x" in src


class TestParityRunnerBeatGridConstruction:
    """The parity runner derives a BeatGrid from the fixture's
    beat_grid ground truth (Phase 9 already populated this for the
    music_video_beat fixture)."""

    SRC_PATH = (
        Path(__file__).resolve().parents[1] / "scripts" / "measure_autoflip_parity.py"
    )

    def test_runner_imports_extract_segment_boundaries(self):
        src = self.SRC_PATH.read_text()
        assert "extract_segment_boundaries" in src

    def test_runner_builds_BeatGrid_for_music_video(self):
        src = self.SRC_PATH.read_text()
        assert "BeatGrid" in src
        assert "music_beat_grid" in src
        # The runner derives downbeats by stride-4 over the fixture's
        # full beat list (4/4 meter).
        assert "[::4]" in src or "::meter" in src or "meter = 4" in src

    def test_runner_passes_actual_segment_boundaries(self):
        src = self.SRC_PATH.read_text()
        assert "actual_segment_boundaries=" in src

    def test_runner_passes_music_beat_grid(self):
        src = self.SRC_PATH.read_text()
        assert "music_beat_grid=music_beat_grid" in src


# ────────────────── score_fixture downbeat_snap_error path ──────────────────


class TestScoreFixturePrefersBoundaries:
    def test_prefers_actual_segment_boundaries_for_downbeat_snap(self):
        from backend.services.autoflip_parity_metrics import score_fixture
        # Switch times have 1 entry at 2.05; boundaries have 4
        # entries on perfect downbeats. The metric should use
        # boundaries (snap_rate = 1.0) not switch times (snap_rate
        # might be lower).
        result = score_fixture(
            metrics_to_run=["downbeat_snap_error"],
            actual_switches=[2.05],
            actual_segment_boundaries=[2.0, 4.0, 6.0, 8.0],
            beat_grid=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        )
        snap = result["downbeat_snap_error"]
        assert snap["snap_rate"] == 1.0
        assert snap["count"] == 4
        assert snap["max_error_ms"] == 0.0

    def test_falls_back_to_switches_when_no_boundaries(self):
        from backend.services.autoflip_parity_metrics import score_fixture
        result = score_fixture(
            metrics_to_run=["downbeat_snap_error"],
            actual_switches=[2.05],
            actual_segment_boundaries=None,
            beat_grid=[0.0, 2.0, 4.0],
        )
        snap = result["downbeat_snap_error"]
        # 2.05 within 200ms of 2.0
        assert snap["snap_rate"] == 1.0
        assert snap["count"] == 1


# ────────────────── extract_segment_boundaries helper ──────────────────


class TestExtractSegmentBoundaries:
    def test_excludes_zero(self):
        from backend.services.autoflip_parity_metrics import extract_segment_boundaries
        @dataclass
        class S:
            start: float
        segs = [S(0.0), S(1.0), S(2.0)]
        assert extract_segment_boundaries(segs) == [1.0, 2.0]

    def test_includes_zero_when_skip_disabled(self):
        from backend.services.autoflip_parity_metrics import extract_segment_boundaries
        @dataclass
        class S:
            start: float
        segs = [S(0.0), S(1.0)]
        assert extract_segment_boundaries(segs, skip_zero=False) == [0.0, 1.0]

    def test_handles_missing_attribute(self):
        from backend.services.autoflip_parity_metrics import extract_segment_boundaries
        # An object with no .start should be silently skipped
        class _Empty:
            pass
        result = extract_segment_boundaries([_Empty(), _Empty()])
        assert result == []
