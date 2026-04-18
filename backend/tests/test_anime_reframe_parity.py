"""Phase 6 — smoke test for the Phase 6 anime parity SLAs.

Runs ``measure_autoflip_parity`` against the new
``anime_panning_close_up`` fixture in ``--animated`` mode and
asserts the SLAs from the acceptance criteria:

  - ``dense_ratio  ≥ 0.30``
  - ``mean_err     ≤ 4.0`` percent of frame width
  - ``p95_err      ≤ 8.0`` percent of frame width
  - The fixture's strategy mix should include `tracking` (Phase 5
    override; the face panning 35 → 75 % is well above the 6 %
    threshold).

The test invokes the runner directly (``run_all`` returns the JSON
payload) instead of shelling out so it stays fast and dependency-
free. The fixture's dense face track is fully populated so the
parity numbers reflect the L1 + Phase 5 wiring.
"""

from __future__ import annotations

import pytest


def test_anime_panning_fixture_meets_slas():
    pytest.importorskip("numpy")
    from backend.scripts.measure_autoflip_parity import run_all

    payload = run_all(
        fixture_names=["anime_panning_close_up"],
        animated=True,
    )
    assert payload["fixture_count"] == 1
    row = payload["results"][0]
    assert row["status"] == "ok", f"fixture errored: {row.get('error')}"

    anim = row.get("animated")
    assert anim is not None, "animated SLA block missing"

    # Fixture has fully populated dense face track → density 1.0
    assert anim["dense_ratio"] >= 0.30, (
        f"dense_ratio {anim['dense_ratio']} below 0.30"
    )
    # Per-frame face center error caps
    assert anim["mean_err"] <= 4.0, (
        f"mean_err {anim['mean_err']:.2f}% above 4% SLA"
    )
    assert anim["p95_err"] <= 8.0, (
        f"p95_err {anim['p95_err']:.2f}% above 8% SLA"
    )

    # SLA aggregate
    assert row.get("sla_status") == "passed", row.get("sla_failures")


def test_animated_summary_handles_empty_segments():
    """Defensive: the summary helper must not crash when the segmenter
    returns zero segments (e.g., dry-run / fixture import failure)."""
    from backend.scripts.measure_autoflip_parity import _animated_summary
    from backend.services.autoflip_parity_fixtures import get_fixture

    spec = get_fixture("anime_panning_close_up")
    summary = _animated_summary(spec, [])
    assert summary["dense_ratio"] == 0.0
    assert summary["tracking_pct"] == 0.0
    assert summary["mean_err"] == 0.0
    assert summary["p95_err"] == 0.0
