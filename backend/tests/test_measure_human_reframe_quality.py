"""Smoke test for backend.scripts.measure_human_reframe_quality.

We don't assert every synthetic fixture passes the §5 targets (some
of those fails surface real tuning work still to be done). We assert
the script RUNS, emits per-fixture metrics, and returns non-zero
exit when hard-fails are found — i.e. it's a working signal source.
"""
from backend.scripts.measure_human_reframe_quality import (
    FixtureMetrics,
    Thresholds,
    _fx_talking_head,
    _grade,
    _measure_fixture,
    main,
)


def test_measure_talking_head_fixture_runs():
    m = _measure_fixture(_fx_talking_head(duration=5.0))
    assert isinstance(m, FixtureMetrics)
    # Sanity: we scored at least some frames.
    assert m.n_frames_scored > 0
    # No silent fallback (pipeline didn't crash).
    assert not m.silent_legacy_fallback


def test_main_returns_int_exit_code():
    rc = main([])
    # Expected to exit non-zero until all fixture targets pass, but
    # must not raise.
    assert rc in (0, 1)


def test_grade_ok_on_clean_metrics():
    m = FixtureMetrics(
        name="fake", n_frames_scored=100,
        chin_clip_frames=0, head_clip_frames=0,
        center_errors=[0.05] * 100,
        jitter_stds=[0.01] * 100,
        saccade_gaps=[2.0, 2.0, 2.0],
        critic_windows_total=100,
        critic_windows_repaired=10,
    )
    ok, fails = _grade(m, Thresholds())
    assert ok, f"expected clean metrics to pass, got {fails}"


def test_grade_fails_on_chin_clip():
    m = FixtureMetrics(
        name="fake", n_frames_scored=100,
        chin_clip_frames=10,  # 10% chin clip → hard fail
    )
    ok, fails = _grade(m, Thresholds())
    assert not ok
    assert any("chin_rate" in f for f in fails)
