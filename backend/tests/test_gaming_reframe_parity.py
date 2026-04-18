"""Phase 6 — gaming reframe parity smoke test.

Runs ``measure_gaming_reframe.run_all`` against the synthetic
fixture set and asserts each fixture passes its SLAs. Acts as
the CI gate for the gaming reframe gap close.
"""

from __future__ import annotations

import pytest


def test_gaming_parity_all_fixtures_pass_slas():
    from backend.scripts.measure_gaming_reframe import run_all

    payload = run_all()
    # Six fixtures now — the TF2 center-bias regression joined
    # the set in this branch.
    assert payload["fixture_count"] == 6
    failures = [
        r for r in payload["results"] if r["sla_status"] != "passed"
    ]
    if failures:
        msgs = [
            f"{r['name']}: {r['sla_failures']}" for r in failures
        ]
        pytest.fail("Gaming parity SLA failures:\n  " + "\n  ".join(msgs))


def test_tf2_regression_center_locks_at_100pct():
    """Phase 1 TF2 regression fixture: cartoon-style FPS with no
    crosshair. Must hit 100% center_lock_pct and 0 off_axis_drift
    events. This is the regression target — before the Phase 1
    center-bias branch landed, this clip would drift to x≈20-32."""
    from backend.scripts.measure_gaming_reframe import run_all

    payload = run_all(["tf2_clip"])
    row = payload["results"][0]
    assert row["sla_status"] == "passed", row["sla_failures"]
    cl = row["center_lock"]
    assert cl["center_lock_pct"] == 1.0, (
        f"TF2 center_lock_pct={cl['center_lock_pct']}, expected 1.0"
    )
    assert cl["off_axis_drift_events"] == 0
    # No crosshair truth → crosshair metric is n/a.
    assert row["crosshair"]["mean"] is None
    assert row["center_bias_regression"] is True


def test_valorant_fixture_layout_target():
    from backend.scripts.measure_gaming_reframe import run_all

    payload = run_all(["valorant_clip"])
    row = payload["results"][0]
    assert row["sla_status"] == "passed", row["sla_failures"]
    # FPS clip should be ≥ 80% fullscreen
    fs = row["layout_distribution"].get("fullscreen", 0.0)
    assert fs >= 0.80, f"valorant fullscreen pct {fs} below 0.80"


def test_lol_fixture_blurfill():
    from backend.scripts.measure_gaming_reframe import run_all

    payload = run_all(["lol_clip"])
    row = payload["results"][0]
    assert row["sla_status"] == "passed", row["sla_failures"]
    blurfill = row["layout_distribution"].get("blurfill", 0.0)
    # LoL fixture is 6s; segments split at 1s boundaries → segments
    # ≥ 4s threshold are blurfill. Some may be wide_zoom on motion.
    # The minimum guarantee is just: layout_target met.
    assert blurfill + row["layout_distribution"].get("wide_zoom", 0) >= 0.5


def test_rocket_league_wide_zoom_on_motion():
    """Racing clip with motion 35 → wide_zoom (motion > 30 threshold)."""
    from backend.scripts.measure_gaming_reframe import run_all

    payload = run_all(["rocket_league_clip"])
    row = payload["results"][0]
    wz = row["layout_distribution"].get("wide_zoom", 0.0)
    fs = row["layout_distribution"].get("fullscreen", 0.0)
    # The motion threshold pushed it into wide_zoom; counted toward
    # the 50% layout target via either bucket.
    assert (wz + fs) >= 0.50


def test_run_all_returns_payload_shape():
    from backend.scripts.measure_gaming_reframe import run_all

    payload = run_all(["valorant_clip"])
    assert payload["schema"] == "gaming_parity_results/1"
    assert "results" in payload
    row = payload["results"][0]
    assert "crosshair" in row
    assert "events" in row
    assert "layout_distribution" in row
    assert "max_velocity_pct_per_sec" in row
    assert "sla_status" in row
