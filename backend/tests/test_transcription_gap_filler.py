"""Unit tests for transcription_gap_filler.

These tests focus on the pure functions:
  - find_transcript_gaps
  - compute_coverage
  - _merge_intervals
  - _vad_overlap
  - _is_hallucinated_fill

The Whisper-calling code path (fill_transcript_gaps_sync) is not tested
here because it requires a GPU-loaded model and real audio; it is exercised
by the existing integration-test harness.
"""
from __future__ import annotations

from backend.services.transcription_gap_filler import (
    _is_hallucinated_fill,
    _merge_intervals,
    _vad_overlap,
    compute_coverage,
    find_transcript_gaps,
)


# ──────────────────────────────────────────────────────────────────────────
# _merge_intervals
# ──────────────────────────────────────────────────────────────────────────

def test_merge_intervals_empty():
    assert _merge_intervals([]) == []


def test_merge_intervals_disjoint():
    assert _merge_intervals([(0, 1), (2, 3), (5, 6)]) == [(0, 1), (2, 3), (5, 6)]


def test_merge_intervals_overlapping():
    assert _merge_intervals([(0, 2), (1, 3), (4, 5)]) == [(0, 3), (4, 5)]


def test_merge_intervals_touching():
    # touching intervals (a[end] == b[start]) should merge
    assert _merge_intervals([(0, 2), (2, 4), (4, 6)]) == [(0, 6)]


def test_merge_intervals_unsorted_input():
    assert _merge_intervals([(5, 6), (0, 2), (2, 3)]) == [(0, 3), (5, 6)]


# ──────────────────────────────────────────────────────────────────────────
# _vad_overlap
# ──────────────────────────────────────────────────────────────────────────

def test_vad_overlap_empty_intervals():
    assert _vad_overlap([], 0.0, 10.0) == 0.0


def test_vad_overlap_empty_window():
    assert _vad_overlap([(0, 10)], 5.0, 5.0) == 0.0
    assert _vad_overlap([(0, 10)], 5.0, 3.0) == 0.0


def test_vad_overlap_full():
    assert _vad_overlap([(0, 10)], 2.0, 5.0) == 3.0


def test_vad_overlap_partial_left():
    assert _vad_overlap([(3, 10)], 0.0, 5.0) == 2.0


def test_vad_overlap_partial_right():
    assert _vad_overlap([(5, 15)], 3.0, 8.0) == 3.0


def test_vad_overlap_multiple_intervals():
    # Window 0..10, VAD at 1..2 and 5..9  →  overlap 1 + 4 = 5
    assert _vad_overlap([(1, 2), (5, 9)], 0.0, 10.0) == 5.0


def test_vad_overlap_no_intersection():
    assert _vad_overlap([(0, 2), (10, 12)], 4.0, 8.0) == 0.0


# ──────────────────────────────────────────────────────────────────────────
# find_transcript_gaps
# ──────────────────────────────────────────────────────────────────────────

def test_find_gaps_empty_segments_returns_whole_vad_region():
    # No transcript at all, VAD says 3s of speech in [5..8] inside a 30s audio.
    gaps = find_transcript_gaps(
        segments=[],
        vad_intervals=[(5.0, 8.0)],
        audio_duration=30.0,
    )
    # Gap is [0..30], VAD overlap = 3s, length = 30s, both exceed thresholds.
    assert len(gaps) == 1
    gs, ge, voiced = gaps[0]
    assert gs == 0.0
    assert ge == 30.0
    assert voiced == 3.0


def test_find_gaps_segments_fully_cover_vad():
    # Segments cover the entire VAD, no gaps.
    segs = [{"start": 0.0, "end": 30.0}]
    gaps = find_transcript_gaps(
        segments=segs,
        vad_intervals=[(5.0, 8.0), (15.0, 20.0)],
        audio_duration=30.0,
    )
    assert gaps == []


def test_find_gaps_small_gap_below_min_gap_sec_is_dropped():
    # Only a 1.5s hole between segments — below the 2.0s default.
    segs = [
        {"start": 0.0, "end": 10.0},
        {"start": 11.5, "end": 30.0},
    ]
    gaps = find_transcript_gaps(
        segments=segs,
        vad_intervals=[(10.2, 11.4)],  # VAD says speech inside the gap
        audio_duration=30.0,
    )
    assert gaps == []


def test_find_gaps_vad_overlap_below_threshold_is_dropped():
    # A 5s transcript hole, but VAD says only 0.2s of speech → below floor.
    segs = [
        {"start": 0.0, "end": 10.0},
        {"start": 15.0, "end": 30.0},
    ]
    gaps = find_transcript_gaps(
        segments=segs,
        vad_intervals=[(12.0, 12.2)],
        audio_duration=30.0,
    )
    assert gaps == []


def test_find_gaps_qualifying_middle_gap():
    # Real case: 10s hole with 4s of VAD-voiced audio inside it.
    segs = [
        {"start": 0.0, "end": 10.0},
        {"start": 20.0, "end": 30.0},
    ]
    gaps = find_transcript_gaps(
        segments=segs,
        vad_intervals=[(12.0, 16.0)],
        audio_duration=30.0,
    )
    assert len(gaps) == 1
    gs, ge, voiced = gaps[0]
    assert gs == 10.0
    assert ge == 20.0
    assert voiced == 4.0


def test_find_gaps_leading_and_trailing():
    # Transcript only covers middle; leading and trailing gaps both qualify.
    segs = [{"start": 10.0, "end": 20.0}]
    gaps = find_transcript_gaps(
        segments=segs,
        vad_intervals=[(2.0, 8.0), (22.0, 28.0)],
        audio_duration=30.0,
    )
    assert len(gaps) == 2
    leading = [g for g in gaps if g[0] == 0.0][0]
    trailing = [g for g in gaps if g[1] == 30.0][0]
    assert leading == (0.0, 10.0, 6.0)
    assert trailing == (20.0, 30.0, 6.0)


def test_find_gaps_tolerates_dataclass_and_dict_segments():
    """Segments may be dataclass-like objects OR dicts — both should work."""
    class _FakeSeg:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    segs = [
        _FakeSeg(0.0, 5.0),
        {"start": 15.0, "end": 30.0},
    ]
    gaps = find_transcript_gaps(
        segments=segs,
        vad_intervals=[(7.0, 12.0)],
        audio_duration=30.0,
    )
    assert len(gaps) == 1
    gs, ge, voiced = gaps[0]
    assert gs == 5.0
    assert ge == 15.0
    assert voiced == 5.0


def test_find_gaps_tolerates_malformed_segments():
    # Segments with None start/end or non-numeric values should be skipped,
    # not crash the function.
    segs = [
        {"start": None, "end": 5.0},
        {"start": 0.0, "end": None},
        {"start": "bad", "end": 5.0},
        {"start": 10.0, "end": 5.0},  # end before start
        {"start": 10.0, "end": 20.0},  # the only valid one
    ]
    gaps = find_transcript_gaps(
        segments=segs,
        vad_intervals=[(2.0, 8.0)],
        audio_duration=30.0,
    )
    # Leading gap [0..10] with 6s VAD → qualifies.
    assert len(gaps) == 1
    assert gaps[0][0] == 0.0
    assert gaps[0][1] == 10.0


def test_find_gaps_zero_duration_returns_empty():
    gaps = find_transcript_gaps(
        segments=[{"start": 0.0, "end": 5.0}],
        vad_intervals=[(0.0, 5.0)],
        audio_duration=0.0,
    )
    assert gaps == []


# ──────────────────────────────────────────────────────────────────────────
# compute_coverage
# ──────────────────────────────────────────────────────────────────────────

def test_coverage_empty_audio():
    assert compute_coverage([], [], 0.0) == (0.0, 0.0)


def test_coverage_full():
    t_cov, v_cov = compute_coverage(
        segments=[{"start": 0.0, "end": 30.0}],
        vad_intervals=[(5.0, 25.0)],
        audio_duration=30.0,
    )
    assert t_cov == 1.0
    assert v_cov == 1.0  # all 20s of voiced audio are inside the segment


def test_coverage_half_transcript_all_voiced_missed():
    # Transcript covers [0..15], VAD says voiced [15..30] → zero voiced coverage.
    t_cov, v_cov = compute_coverage(
        segments=[{"start": 0.0, "end": 15.0}],
        vad_intervals=[(15.0, 30.0)],
        audio_duration=30.0,
    )
    assert t_cov == 0.5
    assert v_cov == 0.0


def test_coverage_no_vad_returns_zero_voiced():
    t_cov, v_cov = compute_coverage(
        segments=[{"start": 0.0, "end": 15.0}],
        vad_intervals=[],
        audio_duration=30.0,
    )
    assert t_cov == 0.5
    assert v_cov == 0.0


def test_coverage_partial_intersection():
    # Transcript [0..10] and [20..30], VAD [5..15] and [25..28]
    # VAD total = 10 + 3 = 13s
    # Covered voiced = [5..10] (5) + [25..28] (3) = 8s
    # Voiced coverage = 8/13 ≈ 0.615
    t_cov, v_cov = compute_coverage(
        segments=[{"start": 0.0, "end": 10.0}, {"start": 20.0, "end": 30.0}],
        vad_intervals=[(5.0, 15.0), (25.0, 28.0)],
        audio_duration=30.0,
    )
    assert abs(t_cov - (20.0 / 30.0)) < 1e-6
    assert abs(v_cov - (8.0 / 13.0)) < 1e-6


# ──────────────────────────────────────────────────────────────────────────
# _is_hallucinated_fill
# ──────────────────────────────────────────────────────────────────────────

def test_hallucinated_empty():
    assert _is_hallucinated_fill("") is True
    assert _is_hallucinated_fill("   ") is True
    assert _is_hallucinated_fill(None or "") is True  # None → "" upstream


def test_hallucinated_boilerplate():
    assert _is_hallucinated_fill("Thanks for watching") is True
    assert _is_hallucinated_fill("Thanks for watching!") is True
    assert _is_hallucinated_fill("Subscribe") is True
    assert _is_hallucinated_fill("you.") is True


def test_hallucinated_punctuation_only():
    assert _is_hallucinated_fill("...") is True
    assert _is_hallucinated_fill(" . . . ") is True


def test_hallucinated_repeated_single_token_loop():
    assert _is_hallucinated_fill("yeah yeah yeah yeah") is True
    assert _is_hallucinated_fill("no no no no no") is True


def test_not_hallucinated_real_speech():
    assert _is_hallucinated_fill("I think we should go to the store") is False
    assert _is_hallucinated_fill("Yeah, that sounds good") is False
    # Borderline short but legitimate — should NOT be filtered out.
    assert _is_hallucinated_fill("Hello there") is False
    assert _is_hallucinated_fill("What?") is False
