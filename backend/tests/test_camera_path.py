"""Tests for camera mode selector."""

from backend.services.camera_path import CameraMode, select_camera_mode
from backend.services.focus_model import SceneFocusRegion


def _make_focus(targets, fits=True):
    """Helper to build a SceneFocusRegion with per-frame targets."""
    return SceneFocusRegion(
        shot_start=targets[0][0] if targets else 0,
        shot_end=targets[-1][0] if targets else 1,
        required=[],
        optional=[],
        min_bounding_rect=(50, 50, 10, 13),
        fits_target_aspect=fits,
        optimal_crop_center=(50, 50),
        per_frame_target=targets,
    )


class TestCameraModeSelector:
    def test_single_stationary_face(self):
        """Single stationary face -> STATIONARY."""
        targets = [(t * 0.5, 50.0 + (t % 3) * 0.1, 40.0) for t in range(20)]
        focus = _make_focus(targets)
        mode = select_camera_mode(focus, source_width=1920, source_height=1080)
        assert mode == CameraMode.STATIONARY

    def test_face_walking_left_to_right(self):
        """Face walking steadily left-to-right -> TRACKING (catches linear motion too)."""
        # Linear motion from x=20 to x=80 over 10 seconds
        targets = [(t * 0.5, 20 + t * 3.0, 40.0) for t in range(20)]
        focus = _make_focus(targets)
        mode = select_camera_mode(focus, source_width=1920, source_height=1080)
        # TRACKING is preferred over PANNING when acceleration is low
        assert mode in (CameraMode.TRACKING, CameraMode.PANNING)

    def test_face_jittery_motion(self):
        """Face with small coherent motion + jitter -> TRACKING."""
        import math
        # Smooth sinusoidal motion with small amplitude — coherent but not linear
        targets = [(t * 0.5, 50 + 8 * math.sin(t * 0.3), 40.0) for t in range(20)]
        focus = _make_focus(targets)
        mode = select_camera_mode(focus, source_width=1920, source_height=1080)
        # Range is ~16, which fits in crop_width (~31.6)
        # Acceleration is low (smooth sinusoid) -> TRACKING
        assert mode == CameraMode.TRACKING

    def test_two_faces_80pct_apart(self):
        """Two faces 80% apart -> PADDING (doesn't fit)."""
        targets = [(0.5, 50.0, 40.0)]
        focus = _make_focus(targets, fits=False)
        mode = select_camera_mode(focus, source_width=1920, source_height=1080)
        assert mode == CameraMode.PADDING

    def test_face_teleports(self):
        """Face that teleports (simulating an undetected cut) -> PADDING."""
        targets = [
            (0.0, 20.0, 40.0),
            (0.5, 21.0, 40.0),
            (1.0, 80.0, 40.0),  # teleport!
            (1.5, 81.0, 40.0),
        ]
        focus = _make_focus(targets)
        mode = select_camera_mode(focus, source_width=1920, source_height=1080)
        # Large x_range (60) doesn't fit in crop_width_pct (~31.6)
        # and non-linear -> PADDING
        assert mode == CameraMode.PADDING

    def test_no_targets(self):
        """No per-frame targets -> STATIONARY (default)."""
        focus = _make_focus([], fits=True)
        mode = select_camera_mode(focus, source_width=1920, source_height=1080)
        assert mode == CameraMode.STATIONARY

    def test_single_target(self):
        """Single target -> STATIONARY."""
        focus = _make_focus([(0.0, 50.0, 40.0)])
        mode = select_camera_mode(focus, source_width=1920, source_height=1080)
        assert mode == CameraMode.STATIONARY
