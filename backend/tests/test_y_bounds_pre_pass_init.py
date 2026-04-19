"""Fix 3.5: y-axis (and x-axis) fallback uses last-known-good bounds,
with a pre-pass initializer so leading no-face frames are derived from
the first real face — not hardcoded frame center (0.5).

Also asserts ``y_uncertain_windows`` is populated when face detections
drop out for longer than ``kalman_prediction_ms * 3`` (~1s).
"""
from backend.services.camera_path_2d import (
    FaceFrame2D,
    _build_x_bounds,
    _build_y_bounds,
    solve_2d_camera_path,
)
from backend.services.reframe_config import get_default_config


def test_leading_no_face_frames_use_first_real_face_for_y():
    """Frames 0-4 have no face. Frame 5 has a face at y=0.75. The
    target_cy at frame 0 is derived from frame 5's face position,
    never 0.5 (the legacy center fallback)."""
    config = get_default_config()
    faces = [None] * 5 + [
        FaceFrame2D(t=0.5, nose_x=0.5, nose_y=0.75, width=0.10, height=0.15)
    ]
    targets, lo, hi = _build_y_bounds(
        faces, crop_h_frac=0.5, config=config,
    )
    # The frame-0 target must match the frame-5 target.
    assert targets[0] == targets[5]
    # And it must NOT equal 0.5 (legacy center fallback).
    assert abs(targets[0] - 0.5) > 0.001, (
        f"leading no-face frame should NOT be 0.5, got {targets[0]}"
    )


def test_leading_no_face_frames_use_first_real_face_for_x():
    """Symmetric pre-pass for the x-axis — leading no-face frames
    inherit the first real face's x bounds."""
    config = get_default_config()
    faces = [None] * 5 + [
        FaceFrame2D(t=0.5, nose_x=0.25, nose_y=0.5, width=0.10, height=0.15)
    ]
    targets, lo, hi = _build_x_bounds(
        faces, crop_w_frac=0.6, config=config,
    )
    assert targets[0] == targets[5]
    # Given nose_x=0.25 and crop_w_frac=0.6, the bounds clamp, but the
    # key invariant is: the first frame's target is NOT the center
    # default 0.5.
    assert targets[0] != 0.5 or targets[5] != 0.5


def test_all_no_face_falls_back_to_center_but_does_not_crash():
    """If every frame has no face, pre-pass returns None and the
    legacy 0.5 center fallback is used."""
    config = get_default_config()
    faces = [None] * 10
    targets, lo, hi = _build_y_bounds(
        faces, crop_h_frac=0.5, config=config,
    )
    # No crash, length preserved, all 0.5.
    assert len(targets) == 10
    assert targets[0] == 0.5


def test_y_uncertain_windows_populated_on_long_dropouts():
    """A 2.5s no-face window triggers a y_uncertain entry when
    kalman_prediction_ms=350 (threshold = 1.05s)."""
    config = get_default_config()
    timestamps = [i * 0.1 for i in range(40)]  # 4s @ 10Hz
    faces = [
        FaceFrame2D(t=t, nose_x=0.5, nose_y=0.5, width=0.10, height=0.15)
        if (i < 5 or i >= 30) else None  # 2.5s gap in middle
        for i, t in enumerate(timestamps)
    ]
    path = solve_2d_camera_path(
        faces,
        timestamps=timestamps,
        source_w=1920, source_h=1080,
        config=config,
    )
    assert path.y_uncertain_windows, (
        "expected at least one uncertain window, got "
        f"{path.y_uncertain_windows}"
    )
    start, end = path.y_uncertain_windows[0]
    # Window should cover most of the 2.5s gap.
    assert end - start >= 1.0


def test_y_uncertain_windows_empty_on_short_dropouts():
    """A 0.3s dropout is below the 1.05s threshold → no entry."""
    config = get_default_config()
    timestamps = [i * 0.1 for i in range(40)]
    faces = [
        FaceFrame2D(t=t, nose_x=0.5, nose_y=0.5, width=0.10, height=0.15)
        if (i < 15 or i >= 18) else None  # 0.3s gap
        for i, t in enumerate(timestamps)
    ]
    path = solve_2d_camera_path(
        faces,
        timestamps=timestamps,
        source_w=1920, source_h=1080,
        config=config,
    )
    assert path.y_uncertain_windows == []


def test_y_uncertain_window_at_end_of_clip():
    """A trailing no-face run also registers as uncertain."""
    config = get_default_config()
    timestamps = [i * 0.1 for i in range(40)]  # 4s
    faces = [
        FaceFrame2D(t=t, nose_x=0.5, nose_y=0.5, width=0.10, height=0.15)
        if i < 10 else None  # 3s trailing dropout
        for i, t in enumerate(timestamps)
    ]
    path = solve_2d_camera_path(
        faces,
        timestamps=timestamps,
        source_w=1920, source_h=1080,
        config=config,
    )
    assert path.y_uncertain_windows, "expected trailing uncertain window"
    start, end = path.y_uncertain_windows[-1]
    assert start >= 0.9
    # End should be at/near the last timestamp.
    assert end >= 3.5
