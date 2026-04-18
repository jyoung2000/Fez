"""v4.1 Fix 4: L1 short-shot bypass + velocity retry.

Short shots (T<=5) were landing in PADDED at ~20% rate because the full
LP with velocity / accel / jerk penalties overconstrained tiny problems.
The hotfix:
  1. T<=5 → skip LP, use stationary-median directly (short_bypass)
  2. LP failure → retry with lam2=2 (velocity relaxed)
  3. Second LP failure → stationary-median fallback
  4. Median fallback failure → PADDED (geometrically impossible)

These tests exercise each rung of the ladder.
"""

from backend.services.camera_solver import (
    CameraMode,
    L1_SHORT_SHOT_BYPASS,
    _l1_track_keyframes,
    _stationary_median_keyframes,
    solve_shot,
)


CROP_HALF = (9.0 / 16.0) / (16.0 / 9.0) / 2.0   # ≈ 0.158


def _bound(t, cx, half=0.02):
    return {
        "t": t, "left": cx - half, "right": cx + half,
        "center": cx, "cy": 0.5,
    }


def test_short_shot_t3_uses_bypass():
    # 3-frame shot — well under L1_SHORT_SHOT_BYPASS (5). LP is never
    # invoked; the bypass returns a 2-keyframe stationary path.
    per_frame_bounds = [
        _bound(0.0, 0.48),
        _bound(0.1, 0.50),
        _bound(0.2, 0.52),
    ]
    keyframes, status, held = _l1_track_keyframes(
        per_frame_bounds, CROP_HALF, job_id="test", shot_index=0,
    )
    assert status == "short_bypass"
    assert len(keyframes) == 2
    # Median of [0.48, 0.50, 0.52] is 0.50; clamped inside crop range.
    assert abs(keyframes[0][1] - 0.50) < 1e-6
    assert keyframes[0][1] == keyframes[1][1]
    assert held == 1.0


def test_short_shot_t5_uses_bypass():
    # Boundary case: T==L1_SHORT_SHOT_BYPASS should still bypass.
    per_frame_bounds = [_bound(i * 0.1, 0.50) for i in range(L1_SHORT_SHOT_BYPASS)]
    keyframes, status, _ = _l1_track_keyframes(
        per_frame_bounds, CROP_HALF, job_id="test", shot_index=1,
    )
    assert status == "short_bypass"
    assert len(keyframes) == 2


def test_long_shot_t10_uses_lp():
    # T>5 should go through the real LP path. A near-constant target
    # stream should return status="success" (not short_bypass).
    per_frame_bounds = [_bound(i * 0.1, 0.50) for i in range(10)]
    keyframes, status, held = _l1_track_keyframes(
        per_frame_bounds, CROP_HALF, job_id="test", shot_index=2,
    )
    assert status in ("success", "retry_success", "median_fallback")
    # Not the short-shot bypass — the LP did run.
    assert status != "short_bypass"
    assert len(keyframes) == 10


def test_stationary_median_helper_success():
    # Tight centers around 0.50 with small half_width so the median cx
    # cleanly covers every frame's required bounds.
    per_frame_bounds = [_bound(0.0, 0.48), _bound(0.1, 0.50), _bound(0.2, 0.52)]
    kf, status, held = _stationary_median_keyframes(
        per_frame_bounds, CROP_HALF, reason="test",
    )
    assert status == "success"
    # Median of [0.48, 0.50, 0.52] = 0.50
    assert abs(kf[0][1] - 0.50) < 1e-6


def test_stationary_median_helper_failure_on_impossible_geometry():
    # A region at cx=0.30 with half_width=0.15 sits at [0.15, 0.45]. Another
    # at cx=0.90 with half_width=0.15 sits at [0.75, 1.05] which is outside
    # the [0.158, 0.842] crop feasibility. Even the median cx=0.60 cannot
    # cover both, so the helper should return failed.
    per_frame_bounds = [
        {"t": 0.0, "left": 0.15, "right": 0.45, "center": 0.30, "cy": 0.5},
        {"t": 0.1, "left": 0.75, "right": 1.05, "center": 0.90, "cy": 0.5},
    ]
    kf, status, _ = _stationary_median_keyframes(
        per_frame_bounds, CROP_HALF, reason="impossible",
    )
    assert status == "failed"
    assert kf == []


def test_solve_shot_short_shot_ends_in_tracking_or_stationary():
    # A 3-frame monotonic shot: STATIONARY fires first because the union
    # width 0.04 fits. This also exercises the new solve_shot cascade:
    # short shots should NEVER land in PADDED.
    class _R:
        def __init__(self, t, cx):
            self.timestamp = t
            self.cx = cx
            self.cy = 0.5
            self.half_width = 0.02
            self.half_height = 0.04
            self.tier = "required"
            self.source = "face"

    class _Shot:
        def __init__(self):
            self.index = 0
            self.start = 0.0
            self.end = 0.2

    regions_per_frame = [
        [_R(0.0, 0.48)],
        [_R(0.1, 0.50)],
        [_R(0.2, 0.52)],
    ]
    sc = solve_shot(_Shot(), regions_per_frame, source_aspect=16 / 9)
    assert sc.mode != CameraMode.PADDED
