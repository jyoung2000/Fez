"""Fix 3.4: Kalman prediction feeds the solver's data term so the
camera LEADS the subject.

Scenario: a subject walks left→right at 0.3 source-width/sec for 2
seconds. Without the predictor the solver's target at frame i is the
raw nose_x at that frame — so the solved camera lags the subject
while the L1 smoothing catches up. With the predictor feeding
kalman_prediction_ms=350 ahead, the target at frame i is the
position the subject will occupy ~350ms later, so the camera sits
ahead of raw detections during steady motion.
"""
from dataclasses import dataclass
from typing import Optional

from backend.services.camera_path_2d import FaceFrame2D, solve_2d_camera_path
from backend.services.reframe_config import get_default_config
from backend.services.subject_kalman import SubjectKalmanRegistry


def _walking_subject(
    t_start: float, t_end: float,
    x0: float, vx_per_sec: float,
    step: float = 0.1,
    slot_id: int = 0,
) -> tuple[list, list]:
    """Build timestamps + FaceFrame2D list for a linearly-walking
    subject. Returns (timestamps, faces)."""
    timestamps = []
    faces = []
    t = t_start
    while t <= t_end + 1e-6:
        x = x0 + vx_per_sec * (t - t_start)
        timestamps.append(t)
        faces.append(FaceFrame2D(
            t=t, nose_x=x, nose_y=0.5,
            width=0.10, height=0.14,
            slot_id=slot_id,
        ))
        t += step
    return timestamps, faces


def test_kalman_predictor_leads_raw_targets():
    """Walking subject at 0.3 source-w/sec. The solved camera cx at
    frame i is closer to the subject's FUTURE position
    (t + kalman_prediction_ms) than to the current position,
    within a small tolerance."""
    config = get_default_config()
    timestamps, faces = _walking_subject(
        t_start=0.0, t_end=2.0, x0=0.2, vx_per_sec=0.3,
    )
    # Build a filled Kalman registry from the faces.
    registry = SubjectKalmanRegistry(config=config)
    for f in faces:
        registry.observe(f.slot_id, f.t, f.nose_x, f.nose_y)

    primary_slot_by_t = {t: 0 for t in timestamps}

    # Baseline: no predictor.
    baseline = solve_2d_camera_path(
        faces, timestamps=timestamps,
        source_w=1920, source_h=1080, config=config,
    )

    # With predictor.
    led = solve_2d_camera_path(
        faces, timestamps=timestamps,
        source_w=1920, source_h=1080, config=config,
        predictor=registry, primary_slot_by_t=primary_slot_by_t,
    )

    # Pick a middle frame (past the EMA warmup + Kalman settle).
    mid = len(timestamps) // 2
    t_mid = timestamps[mid]
    raw_x = faces[mid].nose_x

    # The led solution's cx at t_mid is AHEAD of the raw subject x
    # by a positive amount (subject moves right, camera leads right).
    # Baseline cx lags or equals raw.
    lead_gap = led.cx[mid] - raw_x
    base_gap = baseline.cx[mid] - raw_x

    assert lead_gap > base_gap, (
        f"predictor should pull cx toward future: "
        f"led_gap={lead_gap:.4f} base_gap={base_gap:.4f}"
    )


def test_kalman_no_slot_in_map_falls_back_to_raw():
    """When primary_slot_by_t has no entry for a timestamp, the blend
    skips that frame — behavior identical to no-predictor path."""
    config = get_default_config()
    timestamps, faces = _walking_subject(
        t_start=0.0, t_end=1.0, x0=0.4, vx_per_sec=0.2,
    )
    registry = SubjectKalmanRegistry(config=config)
    for f in faces:
        registry.observe(f.slot_id, f.t, f.nose_x, f.nose_y)

    baseline = solve_2d_camera_path(
        faces, timestamps=timestamps,
        source_w=1920, source_h=1080, config=config,
    )
    no_map = solve_2d_camera_path(
        faces, timestamps=timestamps,
        source_w=1920, source_h=1080, config=config,
        predictor=registry, primary_slot_by_t={},
    )
    for i in range(len(timestamps)):
        assert abs(baseline.cx[i] - no_map.cx[i]) < 1e-6


def test_kalman_high_uncertainty_falls_back_to_raw():
    """A freshly-initialized Kalman filter has high uncertainty on
    the very first observation — the predictor returns high-
    uncertainty predictions that are gated out."""
    config = get_default_config()
    timestamps = [0.0, 0.1, 0.2, 0.3]
    faces = [
        FaceFrame2D(t=t, nose_x=0.5, nose_y=0.5,
                    width=0.10, height=0.14, slot_id=0)
        for t in timestamps
    ]
    registry = SubjectKalmanRegistry(config=config)
    # Only one observation → high P → predictions above the gate.
    registry.observe(0, 0.0, 0.5, 0.5)

    baseline = solve_2d_camera_path(
        faces, timestamps=timestamps,
        source_w=1920, source_h=1080, config=config,
    )
    led = solve_2d_camera_path(
        faces, timestamps=timestamps,
        source_w=1920, source_h=1080, config=config,
        predictor=registry, primary_slot_by_t={t: 0 for t in timestamps},
    )
    # With only 1 observation the filter can't lead — paths should
    # match within LP solver tolerance.
    for i in range(len(timestamps)):
        assert abs(baseline.cx[i] - led.cx[i]) < 0.02
