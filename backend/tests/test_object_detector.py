"""Tests for class-aware object detector."""

import pytest
from dataclasses import dataclass, field

from backend.services.object_detector import (
    ObjectDetector,
    ObjectDetection,
    detect_objects_in_frames,
    get_class_priority,
    get_detector,
    reset_detector,
    DEFAULT_PRIORITY,
)


@dataclass
class MockFrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)
    path: str = ""


@dataclass
class MockFace:
    nose_x: float = 50.0
    nose_y: float = 40.0


class TestObjectDetectorImport:
    def test_module_imports(self):
        """Module imports without error even when no backends are available."""
        import backend.services.object_detector  # should not raise

    def test_backend_name_valid(self):
        """get_detector().backend_name returns a recognized value."""
        reset_detector()
        detector = get_detector()
        assert detector.backend_name in {"ultralytics-yolov8n", "opencv-mobilenet-ssd", "none"}
        reset_detector()


class TestDetectObjectsInFrames:
    def test_empty_list(self):
        """detect_objects_in_frames([]) returns []."""
        result = detect_objects_in_frames([])
        assert result == []

    def test_skips_frames_with_faces(self, tmp_path):
        """Frames with face data are skipped."""
        import cv2
        import numpy as np

        reset_detector()
        detector = get_detector()
        if detector.backend_name == "none":
            pytest.skip("No detector backend available")

        # Create frames
        paths = []
        face_results = []
        for i in range(4):
            frame = np.full((480, 640, 3), 128, dtype=np.uint8)
            p = tmp_path / f"frame_{i}.png"
            cv2.imwrite(str(p), frame)
            paths.append((float(i) * 0.5, str(p)))
            if i < 2:
                face_results.append(MockFrameFaces(timestamp=float(i) * 0.5, faces=[MockFace()]))
            else:
                face_results.append(MockFrameFaces(timestamp=float(i) * 0.5, faces=[]))

        result = detect_objects_in_frames(paths, face_results=face_results)
        # Any detections should only be from frames 2 and 3
        for det in result:
            assert det.timestamp >= 1.0
        reset_detector()


class TestGetClassPriority:
    def test_person_required(self):
        """person -> (True, 1.0)."""
        assert get_class_priority("person") == (True, 1.0)

    def test_car_stationary_not_required(self):
        """car, not moving -> (False, 0.7)."""
        assert get_class_priority("car", is_moving=False) == (False, 0.7)

    def test_car_moving_required(self):
        """car, moving -> (True, 0.7)."""
        assert get_class_priority("car", is_moving=True) == (True, 0.7)

    def test_unknown_class_default(self):
        """Unknown class -> DEFAULT_PRIORITY."""
        assert get_class_priority("unknown") == DEFAULT_PRIORITY

    def test_cat_always_required(self):
        """cat -> always required regardless of motion."""
        assert get_class_priority("cat") == (True, 0.9)
        assert get_class_priority("cat", is_moving=True) == (True, 0.9)

    def test_airplane_if_moving(self):
        """airplane -> (if_moving -> True/False, 0.8)."""
        assert get_class_priority("airplane", is_moving=False) == (False, 0.8)
        assert get_class_priority("airplane", is_moving=True) == (True, 0.8)


class TestObjectDetectionToDict:
    def test_to_dict_sanitizes(self):
        """ObjectDetection.to_dict() produces plain Python types."""
        import numpy as np
        det = ObjectDetection(
            timestamp=np.float64(1.0),
            x=np.float32(50.0), y=np.float32(40.0),
            w=np.float32(10.0), h=np.float32(13.0),
            class_name="person",
            confidence=np.float64(0.95),
        )
        d = det.to_dict()
        for k, v in d.items():
            if k == 'class_name':
                assert isinstance(v, str)
            else:
                assert not hasattr(v, 'item'), f"{k} is still numpy: {type(v)}"
