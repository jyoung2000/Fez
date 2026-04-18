"""Regression tests for object_tracker.py after saliency refactor."""

import numpy as np
import cv2
import pytest
from dataclasses import dataclass, field

from backend.services.object_tracker import track_objects_in_frames


@dataclass
class MockFrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)
    path: str = ""


@dataclass
class MockFace:
    nose_x: float = 50.0
    nose_y: float = 40.0


def _write_frames_with_square(tmp_path, n=5):
    """Write n frames with a moving bright square, return frame_paths."""
    paths = []
    for i in range(n):
        frame = np.full((480, 640, 3), 30, dtype=np.uint8)
        x_off = 100 + i * 80
        cv2.rectangle(frame, (x_off, 150), (x_off + 100, 300), (220, 220, 220), -1)
        p = tmp_path / f"frame_{i}.png"
        cv2.imwrite(str(p), frame)
        paths.append((float(i) * 0.5, str(p)))
    return paths


class TestTrackObjectsLegacyAPI:
    def test_return_confidence_false_type(self, tmp_path):
        """return_confidence=False returns list[(timestamp, int)] -- legacy type."""
        frames = _write_frames_with_square(tmp_path)
        result = track_objects_in_frames(frames, return_confidence=False)
        assert isinstance(result, list)
        for item in result:
            assert len(item) == 2
            assert isinstance(item[0], float)
            assert isinstance(item[1], int)

    def test_return_confidence_true_type(self, tmp_path):
        """return_confidence=True returns list[(timestamp, float, float)]."""
        frames = _write_frames_with_square(tmp_path)
        result = track_objects_in_frames(frames, return_confidence=True)
        assert isinstance(result, list)
        for item in result:
            assert len(item) == 3
            assert isinstance(item[0], float)
            assert isinstance(item[1], float)
            assert isinstance(item[2], float)

    def test_empty_frame_list(self):
        """Empty frame list returns empty list."""
        result = track_objects_in_frames([])
        assert result == []

    def test_skips_frames_with_faces(self, tmp_path):
        """Frames where face_results has faces are skipped."""
        frames = _write_frames_with_square(tmp_path, n=5)
        face_results = [
            MockFrameFaces(timestamp=float(i) * 0.5, faces=[MockFace()] if i < 3 else [])
            for i in range(5)
        ]
        result = track_objects_in_frames(frames, face_results=face_results)
        # Only faceless frames (index 3 and 4) should produce results
        for t, x in result:
            assert t >= 1.5  # frame 3 = 1.5s

    def test_no_frames_no_crash(self, tmp_path):
        """No valid frames (all unreadable) -> empty list, no crash."""
        frames = [(0.0, "/nonexistent/path.png"), (0.5, "/nonexistent/path2.png")]
        result = track_objects_in_frames(frames)
        assert result == []
