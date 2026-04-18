"""v4 Change 3 — L1 path solver bridges a step change in straight line.

Anchor stream that holds at cx=0.3 for 5 frames then jumps to cx=0.7
for 5 frames. The LP solver should produce a piecewise path that holds
on each side and transitions monotonically across the jump (not
oscillating, not overshooting).
"""

from backend.services.camera_solver import _l1_track_keyframes


def test_l1_path_bridges_jump_monotonically():
    per_frame_bounds = []
    # 5 frames at cx=0.3 with narrow bounds
    for i in range(5):
        per_frame_bounds.append({
            "t": i * 0.1,
            "left": 0.28, "right": 0.32, "center": 0.30, "cy": 0.5,
        })
    # 5 frames at cx=0.7 with narrow bounds
    for i in range(5):
        per_frame_bounds.append({
            "t": 0.5 + i * 0.1,
            "left": 0.68, "right": 0.72, "center": 0.70, "cy": 0.5,
        })

    crop_half_width = (9.0 / 16.0) / (16.0 / 9.0) / 2.0  # ≈ 0.158

    keyframes, status, held_frac = _l1_track_keyframes(
        per_frame_bounds, crop_half_width, job_id="test", shot_index=1,
    )

    assert status == "success"
    assert len(keyframes) == 10

    cxs = [kf[1] for kf in keyframes]
    # First half should sit close to 0.3, last half close to 0.7.
    first_mean = sum(cxs[:5]) / 5
    last_mean = sum(cxs[5:]) / 5
    assert abs(first_mean - 0.30) < 0.10, f"first half mean {first_mean}"
    assert abs(last_mean - 0.70) < 0.10, f"last half mean {last_mean}"

    # No oscillation: cumulative absolute change ≈ end-to-end change.
    # Total wiggle <= 1.5× direct distance is the AutoFlip "no wiggle"
    # property — the path can ease in but cannot zigzag.
    total_wiggle = sum(abs(cxs[i] - cxs[i - 1]) for i in range(1, len(cxs)))
    direct = abs(cxs[-1] - cxs[0])
    assert total_wiggle <= direct * 1.5 + 1e-6, (
        f"path oscillates: total wiggle {total_wiggle:.3f} vs direct {direct:.3f}"
    )

    # cxs should be (weakly) non-decreasing across the transition — the
    # solver should not back up.
    for i in range(1, len(cxs)):
        assert cxs[i] >= cxs[i - 1] - 1e-6, (
            f"path went backwards at i={i}: {cxs[i - 1]} → {cxs[i]}"
        )
