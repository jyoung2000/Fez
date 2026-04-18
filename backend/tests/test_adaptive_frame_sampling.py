"""Phase 4 acceptance tests — adaptive VLM frame scheduling.

The scheduler is pure: given a set of signals (shot cuts, face
confidence timeline, ASD margin timeline, clip-candidate starts,
duration), it returns a sorted list of timestamps to hand to the
VLM. These tests lock in the adaptive rule table.
"""

import pytest

from backend.services.adaptive_frame_sampler import (
    AdaptiveSamplingInputs,
    ADAPTIVE_SAMPLING_ENV,
    _total_cap,
    adaptive_sampling_enabled,
    compute_adaptive_frame_times,
)


def _nearest(ts, target, tol=0.1):
    return any(abs(t - target) <= tol for t in ts)


def test_adaptive_includes_shot_cut_neighborhoods():
    """Each shot cut contributes a frame at the cut and +1.0s later."""
    inp = AdaptiveSamplingInputs(
        duration_seconds=60.0,
        shot_cuts=[5.0, 12.0, 25.0, 40.0, 55.0],
    )
    ts = compute_adaptive_frame_times(inp)
    for cut in [5.0, 12.0, 25.0, 40.0, 55.0]:
        assert _nearest(ts, cut), f"missing cut frame at {cut}s"
        # +1.0 follow-up is added except when beyond duration.
        if cut + 1.0 <= inp.duration_seconds:
            assert _nearest(ts, cut + 1.0), f"missing +1s frame at {cut + 1.0}s"


def test_adaptive_oversamples_low_face_conf_window():
    """A face-conf dropout >0.5s produces a midpoint frame."""
    # Timeline: conf 0.9 until t=10, then 0.3 until t=13 (3s dropout), then 0.9 again.
    timeline = [(t * 0.5, 0.9) for t in range(21)]      # 0..10s high conf
    timeline += [(10.5 + i * 0.5, 0.3) for i in range(6)]  # 10.5..13s low
    timeline += [(13.5 + i * 0.5, 0.9) for i in range(20)]  # 13.5..23s high
    inp = AdaptiveSamplingInputs(
        duration_seconds=60.0,
        face_conf_timeline=timeline,
    )
    ts = compute_adaptive_frame_times(inp)
    # Dropout runs ~10.5-13s; midpoint ~11.75s. Allow a wide band
    # because the midpoint depends on which sample crossed the
    # threshold first vs last.
    assert any(10.0 <= t <= 14.0 for t in ts), ts


def test_adaptive_hook_window_has_three_frames():
    """Each clip candidate yields frames at start+{0.5,1.5,2.5}s."""
    inp = AdaptiveSamplingInputs(
        duration_seconds=60.0,
        clip_candidate_starts=[12.0],
    )
    ts = compute_adaptive_frame_times(inp)
    assert _nearest(ts, 12.5)
    assert _nearest(ts, 13.5)
    assert _nearest(ts, 14.5)


def test_adaptive_asd_ambiguity_window():
    """ASD margin < 0.3 for > 0.5s produces a midpoint frame."""
    # Stable period (margin 0.5), then 1s ambiguous window, then back.
    timeline = [(t * 0.2, 0.5) for t in range(25)]  # 0..5s
    timeline += [(5.2 + i * 0.2, 0.15) for i in range(6)]  # 5.2..6.2s
    timeline += [(6.4 + i * 0.2, 0.5) for i in range(20)]
    inp = AdaptiveSamplingInputs(
        duration_seconds=30.0,
        asd_margin_timeline=timeline,
    )
    ts = compute_adaptive_frame_times(inp)
    assert any(5.0 <= t <= 7.0 for t in ts), ts


def test_adaptive_stable_region_floor():
    """90s of totally stable content still gets at least one frame per 30s."""
    inp = AdaptiveSamplingInputs(duration_seconds=90.0)
    ts = compute_adaptive_frame_times(inp)
    assert len(ts) >= 3
    # One in each of [0,30), [30,60), [60,90).
    assert any(0.0 <= t < 30.0 for t in ts)
    assert any(30.0 <= t < 60.0 for t in ts)
    assert any(60.0 <= t <= 90.0 for t in ts)


def test_adaptive_total_capped_per_job():
    """Pathological input with 1000 cuts in 60s is capped at 60."""
    inp = AdaptiveSamplingInputs(
        duration_seconds=60.0,
        shot_cuts=[i * 0.05 for i in range(1000)],
    )
    ts = compute_adaptive_frame_times(inp)
    assert len(ts) <= 60
    assert len(ts) == _total_cap(60.0)


def test_adaptive_cap_scales_with_duration():
    """10-minute clip → cap is max(60, 40) = 60.
    30-minute clip → cap is max(60, 120) = 120."""
    assert _total_cap(600.0) == 60   # max(60, 40) = 60
    assert _total_cap(1800.0) == 120  # max(60, 120) = 120


def test_adaptive_dedupes_overlapping_rules():
    """Shot cut at 12.0 and hook window at 11.5 both produce a frame
    near 12.5; dedup should collapse them."""
    inp = AdaptiveSamplingInputs(
        duration_seconds=60.0,
        shot_cuts=[12.0],
        clip_candidate_starts=[11.0],  # hook @ 11.5, 12.5, 13.5
    )
    ts = compute_adaptive_frame_times(inp)
    # Sanity: result is sorted and monotonically increasing.
    assert ts == sorted(ts)
    # Dedup tol is 0.2 — no pair should be closer than that.
    for a, b in zip(ts, ts[1:]):
        assert b - a >= 0.2 - 1e-9, (a, b)


def test_adaptive_disabled_flag_default(monkeypatch):
    """Default env is OFF."""
    monkeypatch.delenv(ADAPTIVE_SAMPLING_ENV, raising=False)
    assert not adaptive_sampling_enabled()


def test_adaptive_enabled_flag_truthy_values(monkeypatch):
    for val in ("1", "true", "True", "YES", "on"):
        monkeypatch.setenv(ADAPTIVE_SAMPLING_ENV, val)
        assert adaptive_sampling_enabled(), val


def test_adaptive_empty_inputs_still_gets_floor():
    """Zero signals → scheduler still emits the stable-region floor."""
    inp = AdaptiveSamplingInputs(duration_seconds=75.0)
    ts = compute_adaptive_frame_times(inp)
    # 75s / 30s windows = 3 windows → 3 samples.
    assert len(ts) == 3


def test_adaptive_zero_duration_returns_empty():
    inp = AdaptiveSamplingInputs(duration_seconds=0.0)
    ts = compute_adaptive_frame_times(inp)
    assert ts == []
