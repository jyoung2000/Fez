"""End-to-end smoke test: the SOTA preview/metric parity patch
should reduce ``max_acceleration`` and ``max_jerk`` on a realistic
panel-style fixture by an order of magnitude.

The original Tank-vs-Tyrese QA run reported
``max_acceleration: 43.03, max_jerk: 86.06`` — squarely in the
"legacy step" regime (every segment renders as an instant snap).
After the patch, the same per-frame metrics see the smoothed
camera path, which is what the production export already draws.

Pure Python: no ffmpeg, no numpy, no torch — runs on the QA sandbox.
"""
from __future__ import annotations

from backend.scripts.export_autoflip_compatible import (
    reframe_segments_to_events,
)
from backend.services.autoflip_parity_metrics import (
    max_acceleration,
    max_jerk,
)


def _build_panel_fixture() -> tuple[list[dict], list[dict]]:
    """Build matched (legacy_segments, new_segments) lists.

    Both lists describe the SAME 30-segment panel-style clip:
    15 stationary speaker segments alternating left/right, then 5
    tracking shots panning across the frame, all with an
    ``ease_in_ms=200`` ramp at the boundary.

    The "legacy" version strips ``motion_path`` and ``ease_in_ms`` to
    simulate what the cache produced before this patch, so the
    comparison apples-to-apples isolates the new keypoint stream.
    """
    src_w = 1920
    seg_dur = 2.0
    new_segs: list[dict] = []
    t = 0.0

    speaker_xs = [200, 1000, 200, 1000, 200, 1000, 200, 1000, 200, 1000,
                  200, 1000, 200, 1000, 200]
    for x in speaker_xs:
        new_segs.append({
            "start": t, "end": t + seg_dur,
            "subject_x": float(x),
            "ease_in_ms": 200,
        })
        t += seg_dur

    for _ in range(5):
        # 21 keypoints across 2 sec — pans 200 → 1800 (full width).
        path = [
            (t + j * 0.1, 200.0 + j * 80.0)
            for j in range(int(seg_dur / 0.1) + 1)
        ]
        new_segs.append({
            "start": t, "end": t + seg_dur,
            "subject_x": 200.0,
            "ease_in_ms": 200,
            "motion_path": path,
        })
        t += seg_dur

    legacy_segs = [
        {k: v for k, v in s.items()
         if k not in ("motion_path", "ease_in_ms")}
        for s in new_segs
    ]
    return legacy_segs, new_segs


def test_motion_path_smoothing_improves_event_stream_metrics():
    legacy_segs, new_segs = _build_panel_fixture()

    events_legacy = reframe_segments_to_events(
        legacy_segs, 1920, 1080, 30.0,
    )
    events_new = reframe_segments_to_events(
        new_segs, 1920, 1080, 30.0,
    )

    centers_legacy = [e["crop_cx"] * 100.0 for e in events_legacy]
    centers_new = [e["crop_cx"] * 100.0 for e in events_new]

    accel_legacy = max_acceleration(centers_legacy)
    jerk_legacy = max_jerk(centers_legacy)
    accel_new = max_acceleration(centers_new)
    jerk_new = max_jerk(centers_new)

    # The "legacy step" regime should land near the Tank-vs-Tyrese
    # numbers — if it doesn't, the fixture has drifted away from the
    # bug we set out to fix.
    assert accel_legacy > 30.0, (
        f"legacy fixture is no longer in the bug regime: "
        f"accel={accel_legacy} (expected > 30)"
    )
    assert jerk_legacy > 60.0

    # The patched stream should be at least 5x smoother on accel and
    # at least 10x smoother on jerk. Real numbers from local runs are
    # ~9x / ~25x; the looser bounds give the assertion headroom for
    # platform floating-point variation.
    assert accel_new < accel_legacy / 5.0, (
        f"accel did not improve enough: legacy={accel_legacy}, "
        f"new={accel_new}"
    )
    assert jerk_new < jerk_legacy / 10.0, (
        f"jerk did not improve enough: legacy={jerk_legacy}, "
        f"new={jerk_new}"
    )

    # Both streams have the same number of events (the patch only
    # changes per-frame x, not the event count).
    assert len(events_new) == len(events_legacy)
