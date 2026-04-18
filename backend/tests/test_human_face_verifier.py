"""Tests for the HumanFaceVerifier.

Since MediaPipe Pose may not be available in CI, these tests cover:
  - Graceful fail-open behavior when MediaPipe is unavailable
  - Spatial consistency logic via mocked pose results
  - Edge cases (empty/tiny frames)
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from backend.services.human_face_verifier import HumanFaceVerifier, reset_verifier


class TestVerifierAvailability:
    def test_unavailable_when_mediapipe_missing(self):
        """Verifier reports available=False gracefully when MediaPipe can't import."""
        with patch.dict("sys.modules", {"mediapipe": None}):
            verifier = HumanFaceVerifier()
            assert verifier.available is False

    def test_unavailable_returns_false(self):
        """When unavailable, verify_face returns (False, 0.0)."""
        verifier = HumanFaceVerifier.__new__(HumanFaceVerifier)
        verifier._available = False
        verifier._pose = None
        verifier._mp = None

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        is_human, conf = verifier.verify_face(frame, 50, 50, 10, 12)
        assert is_human is False
        assert conf == 0.0


class TestVerifyFaceEdgeCases:
    def _make_verifier(self):
        """Create a verifier that reports available but has a mocked pose."""
        v = HumanFaceVerifier.__new__(HumanFaceVerifier)
        v._available = True
        v._pose = MagicMock()
        v._mp = MagicMock()
        return v

    def test_none_frame_returns_false(self):
        v = self._make_verifier()
        is_human, conf = v.verify_face(None, 50, 50, 10, 12)
        assert is_human is False

    def test_tiny_frame_returns_false(self):
        v = self._make_verifier()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        is_human, conf = v.verify_face(frame, 50, 50, 10, 12)
        assert is_human is False

    def test_no_pose_detected_returns_false(self):
        v = self._make_verifier()
        v._pose.process.return_value = MagicMock(pose_landmarks=None)

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        is_human, conf = v.verify_face(frame, 50, 50, 10, 12)
        assert is_human is False
        assert conf == 0.0


class TestSpatialConsistency:
    def _make_landmark(self, x, y, visibility=0.9):
        lm = MagicMock()
        lm.x = x
        lm.y = y
        lm.visibility = visibility
        return lm

    def _make_verifier_with_pose(self, nose, left_shoulder, right_shoulder):
        """Create a verifier with mock pose results."""
        v = HumanFaceVerifier.__new__(HumanFaceVerifier)
        v._available = True
        v._mp = MagicMock()

        # Set up PoseLandmark enum values
        v._mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value = 11
        v._mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value = 12
        v._mp.solutions.pose.PoseLandmark.NOSE.value = 0

        landmarks = [None] * 33
        landmarks[0] = nose
        landmarks[11] = left_shoulder
        landmarks[12] = right_shoulder

        mock_results = MagicMock()
        mock_results.pose_landmarks.landmark = landmarks

        v._pose = MagicMock()
        v._pose.process.return_value = mock_results
        return v

    def test_human_with_shoulders_below_face(self):
        """Face with visible shoulders directly below → human."""
        nose = self._make_landmark(0.30, 0.30)
        left_sh = self._make_landmark(0.25, 0.50)
        right_sh = self._make_landmark(0.35, 0.50)
        v = self._make_verifier_with_pose(nose, left_sh, right_sh)

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        is_human, conf = v.verify_face(frame, 30, 30, 10, 12)

        assert is_human is True
        assert conf > 0.5

    def test_figurine_no_body(self):
        """Face-shaped object with no pose detected → not human."""
        v = HumanFaceVerifier.__new__(HumanFaceVerifier)
        v._available = True
        v._mp = MagicMock()
        v._pose = MagicMock()
        v._pose.process.return_value = MagicMock(pose_landmarks=None)

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        is_human, conf = v.verify_face(frame, 70, 40, 8, 10)

        assert is_human is False

    def test_shoulders_too_far_horizontally(self):
        """Pose detected but shoulders are far from the face → different person."""
        # Face at x=20%, but shoulders at x=80%
        nose = self._make_landmark(0.80, 0.30)
        left_sh = self._make_landmark(0.75, 0.50)
        right_sh = self._make_landmark(0.85, 0.50)
        v = self._make_verifier_with_pose(nose, left_sh, right_sh)

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        is_human, conf = v.verify_face(frame, 20, 30, 10, 12)

        assert is_human is False

    def test_shoulders_above_face(self):
        """Shoulders ABOVE face (impossible anatomy) → not human."""
        nose = self._make_landmark(0.30, 0.50)
        left_sh = self._make_landmark(0.25, 0.20)  # above
        right_sh = self._make_landmark(0.35, 0.20)  # above
        v = self._make_verifier_with_pose(nose, left_sh, right_sh)

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        is_human, conf = v.verify_face(frame, 30, 50, 10, 12)

        assert is_human is False

    def test_low_shoulder_visibility_rejected(self):
        """Shoulders with very low visibility → not trusted."""
        nose = self._make_landmark(0.30, 0.30)
        left_sh = self._make_landmark(0.25, 0.50, visibility=0.1)
        right_sh = self._make_landmark(0.35, 0.50, visibility=0.1)
        v = self._make_verifier_with_pose(nose, left_sh, right_sh)

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        is_human, conf = v.verify_face(frame, 30, 30, 10, 12)

        assert is_human is False
