"""Week 3 — unit tests for ``cut_to_hold_ratio``.

The metric is pure Python, takes an AutoFlip-compatible event list,
and returns a dict summarizing the per-segment hold distribution.
These tests pin the edge cases the real-content bench will hit.
"""
from __future__ import annotations

import pytest

from backend.services.autoflip_parity_metrics import cut_to_hold_ratio


def _ev(t: float, scene_change: bool = False) -> dict:
    return {"t": t, "scene_change": scene_change}


# ───────────────────────── empty / degenerate ─────────────────────────


def test_empty_events_returns_zero_segments():
    assert cut_to_hold_ratio([]) == {"n_segments": 0}


def test_no_scene_changes_treats_clip_as_one_segment():
    """When the event list lacks any scene_change flag, the whole clip counts
    as a single hold spanning first → last event t."""
    events = [_ev(0.0), _ev(1.0), _ev(2.0), _ev(3.0), _ev(4.0)]
    result = cut_to_hold_ratio(events)

    assert result["n_segments"] == 1
    assert result["median_hold_sec"] == 4.0
    assert result["segments_under_1s_rate"] == 0.0
    assert result["segments_over_8s_rate"] == 0.0


def test_single_scene_change_event_only_yields_empty():
    """A single event with scene_change=True has no following event to
    measure the hold against. segment_ends[0] = last_t = that same t,
    so hold = 0 which is filtered out."""
    events = [_ev(0.0, scene_change=True)]
    result = cut_to_hold_ratio(events)
    assert result == {"n_segments": 0}


# ───────────────────────── three-segment distribution ─────────────────────────


def test_three_segments_of_2_4_6_seconds():
    """Segments span [0,2), [2,6), [6,12]. Median = 4.0, n=3."""
    events = [
        _ev(0.0, scene_change=True),
        _ev(1.0),
        _ev(2.0, scene_change=True),
        _ev(4.0),
        _ev(6.0, scene_change=True),
        _ev(9.0),
        _ev(12.0),
    ]
    result = cut_to_hold_ratio(events)

    assert result["n_segments"] == 3
    assert result["median_hold_sec"] == 4.0
    # With sorted holds [2.0, 4.0, 6.0]: p25 = idx min(2, int(0.75)) = 0 → 2.0
    assert result["p25"] == 2.0
    # p75 = idx min(2, int(2.25)) = 2 → 6.0
    assert result["p75"] == 6.0
    # p95 = idx min(2, int(2.85)) = 2 → 6.0
    assert result["p95"] == 6.0
    assert result["segments_under_1s_rate"] == 0.0
    assert result["segments_over_8s_rate"] == 0.0


# ───────────────────────── red-flag rates ─────────────────────────


def test_all_segments_under_1s_flags_correctly():
    """5 segments of ~0.5s each — segments_under_1s_rate = 1.0."""
    events = []
    for i in range(5):
        events.append(_ev(i * 0.5, scene_change=True))
        events.append(_ev(i * 0.5 + 0.25))
    events.append(_ev(2.5))  # final end
    result = cut_to_hold_ratio(events)

    assert result["n_segments"] == 5
    assert result["segments_under_1s_rate"] == 1.0
    assert result["segments_over_8s_rate"] == 0.0
    assert result["median_hold_sec"] == 0.5


def test_all_segments_over_8s_flags_correctly():
    """2 segments of ~10s each — segments_over_8s_rate = 1.0."""
    events = [
        _ev(0.0, scene_change=True),
        _ev(5.0),
        _ev(10.0, scene_change=True),
        _ev(15.0),
        _ev(20.0),
    ]
    result = cut_to_hold_ratio(events)

    assert result["n_segments"] == 2
    assert result["segments_over_8s_rate"] == 1.0
    assert result["segments_under_1s_rate"] == 0.0
    assert result["median_hold_sec"] == 10.0


# ───────────────────────── mixed distribution ─────────────────────────


def test_mixed_distribution_partial_rates():
    """4 segments: 0.5s, 2s, 3s, 10s. under_1s = 1/4, over_8s = 1/4."""
    events = [
        _ev(0.0, scene_change=True),   # seg 0: ends at 0.5
        _ev(0.5, scene_change=True),   # seg 1: ends at 2.5
        _ev(2.5, scene_change=True),   # seg 2: ends at 5.5
        _ev(5.5, scene_change=True),   # seg 3: ends at 15.5
        _ev(15.5),                     # final timestamp
    ]
    result = cut_to_hold_ratio(events)

    assert result["n_segments"] == 4
    # Holds: [0.5, 2.0, 3.0, 10.0]; median of 4 = mean(2.0, 3.0) = 2.5
    assert result["median_hold_sec"] == 2.5
    assert result["segments_under_1s_rate"] == 0.25
    assert result["segments_over_8s_rate"] == 0.25


# ───────────────────────── quantile correctness ─────────────────────────


def test_quantiles_respect_nearest_rank():
    """10 segments with holds 1..10s; p25=3, p75=8, p95=10 (nearest-rank)."""
    events = [_ev(0.0, scene_change=True)]
    t = 0.0
    for hold in range(1, 11):  # 1, 2, ..., 10
        t += hold
        events.append(_ev(t, scene_change=True))
    events.append(_ev(t))  # close the final segment (hold=0, filtered)
    result = cut_to_hold_ratio(events)

    assert result["n_segments"] == 10
    # Holds: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], median = 5.5
    assert result["median_hold_sec"] == 5.5
    # p25: idx = int(10 * 0.25) = 2 → holds_sorted[2] = 3
    assert result["p25"] == 3.0
    # p75: idx = int(10 * 0.75) = 7 → holds_sorted[7] = 8
    assert result["p75"] == 8.0
    # p95: idx = min(9, int(10 * 0.95)) = 9 → holds_sorted[9] = 10
    assert result["p95"] == 10.0
