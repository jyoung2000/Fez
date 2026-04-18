"""Tests for strict pose-nose containment verification.

Tests the new verify_faces_in_frame API and strict containment logic.
Since MediaPipe may not be available, tests use mocked pose results.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from backend.services.human_face_verifier import HumanFaceVerifier


class _MockFace:
    """Minimal face mock for verification tests."""
    def __init__(self, nose_x=50, nose_y=50, x_center=50, y_center=50,
                 width=10, height=12):
        self.nose_x = nose_x
        self.nose_y = nose_y
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height
        self.is_human = True
        self.pose_confidence = 0.0


def _make_verifier_with_region_response(responses):
    """Create a verifier that returns pre-set nose positions per region call.

    `responses` is a list of Optional[(nose_x_px, nose_y_px)] — one per call
    to _get_pose_nose_for_region, consumed in order.
    """
    v = HumanFaceVerifier.__new__(HumanFaceVerifier)
    v._available = True
    v._mp = MagicMock()
    v._pose = MagicMock()
    v._min_pose_confidence = 0.4

    call_idx = [0]
    def fake_get_nose(frame_bgr, cx, cy, w, h, sw, sh):
        idx = call_idx[0]
        call_idx[0] += 1
        if idx < len(responses):
            return responses[idx]
        return None

    v._get_pose_nose_for_region = fake_get_nose
    return v


class TestFailOpen:
    def test_unavailable_verifier_accepts_all(self):
        """When verifier is unavailable, all faces get is_human=True."""
        v = HumanFaceVerifier.__new__(HumanFaceVerifier)
        v._available = False
        v._pose = None
        v._mp = None

        faces = [_MockFace(30, 40), _MockFace(70, 35)]
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        v.verify_faces_in_frame(frame, faces, 200, 200)

        assert all(f.is_human for f in faces)

    def test_empty_faces_list(self):
        """Empty faces list returns without error."""
        v = _make_verifier_with_region_response([])
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = v.verify_faces_in_frame(frame, [], 200, 200)
        assert result == []

    def test_none_frame_fails_open(self):
        """None frame fails open."""
        v = _make_verifier_with_region_response([])
        faces = [_MockFace()]
        v.verify_faces_in_frame(None, faces, 200, 200)
        assert faces[0].is_human is True

    def test_tiny_frame_fails_open(self):
        """Tiny frame (< 50px) fails open."""
        v = _make_verifier_with_region_response([])
        faces = [_MockFace()]
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        v.verify_faces_in_frame(frame, faces, 10, 10)
        assert faces[0].is_human is True


class TestStrictContainment:
    def test_nose_inside_bbox_passes(self):
        """A pose nose inside the face bbox → is_human=True."""
        # Face at center (50%, 50%), size 10%x12% of a 1000x1000 frame
        # So bbox is [450-550, 440-560] in pixels
        # Pose nose at (500, 500) = inside
        face = _MockFace(nose_x=50, nose_y=50, width=10, height=12)
        v = _make_verifier_with_region_response([(500, 500)])
        frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        v.verify_faces_in_frame(frame, [face], 1000, 1000)
        assert face.is_human is True

    def test_nose_outside_bbox_fails(self):
        """A pose nose far outside the face bbox → is_human=False."""
        # Face at (50%, 50%), bbox [450-550, 440-560]
        # Pose nose at (200, 200) = way outside
        face = _MockFace(nose_x=50, nose_y=50, width=10, height=12)
        v = _make_verifier_with_region_response([(200, 200)])
        frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        v.verify_faces_in_frame(frame, [face], 1000, 1000)
        assert face.is_human is False

    def test_no_pose_detected_fails(self):
        """No pose detected at all → is_human=False."""
        face = _MockFace(nose_x=50, nose_y=50)
        v = _make_verifier_with_region_response([None])
        frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        v.verify_faces_in_frame(frame, [face], 1000, 1000)
        assert face.is_human is False

    def test_figurine_in_front_of_torso(self):
        """Figurine face at (70, 35) with human torso behind it.

        The human's pose nose is at the HUMAN's face position (30, 40),
        not inside the figurine's bbox. The figurine should be rejected.
        """
        # 1920x1080 frame
        # Human face at (30%, 40%), figurine face at (70%, 35%)
        human_face = _MockFace(nose_x=30, nose_y=40, width=8, height=10)
        figurine_face = _MockFace(nose_x=70, nose_y=35, width=6, height=8)

        # Human's region finds pose nose at human's face position (576, 432)
        # Figurine's region finds pose nose at human's face position too (576, 432)
        # because the human's torso is visible in the figurine's expanded region
        human_nose_px = (576.0, 432.0)   # 30% of 1920, 40% of 1080
        # For the figurine's region, the pose detects the human behind it
        # The nose is at the human's position — NOT inside the figurine bbox
        v = _make_verifier_with_region_response([
            human_nose_px,   # human face region → finds own nose
            human_nose_px,   # figurine region → finds human's nose (behind it)
        ])

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        v.verify_faces_in_frame(frame, [human_face, figurine_face], 1920, 1080)

        assert human_face.is_human is True, "Human face should pass"
        assert figurine_face.is_human is False, "Figurine face should fail"

    def test_two_side_by_side_faces(self):
        """Two humans side by side: each matches its own pose nose."""
        # Person A at 25%, person B at 75%, on a 1000x1000 frame
        face_a = _MockFace(nose_x=25, nose_y=40, width=10, height=12)
        face_b = _MockFace(nose_x=75, nose_y=40, width=10, height=12)

        # Each region finds its own nose
        v = _make_verifier_with_region_response([
            (250, 400),  # face_a's pose nose inside bbox [200-300, 340-460]
            (750, 400),  # face_b's pose nose inside bbox [700-800, 340-460]
        ])

        frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        v.verify_faces_in_frame(frame, [face_a, face_b], 1000, 1000)

        assert face_a.is_human is True
        assert face_b.is_human is True

    def test_tolerance_allows_slight_offset(self):
        """Pose nose slightly outside bbox but within tolerance → passes."""
        # Face at (50%, 50%), size 10%x12% on 1000x1000
        # bbox is [450-550, 440-560]
        # tolerance is 20% of face_w = 20px, 20% of face_h = 24px
        # Nose at (555, 440) → x is 5px outside, within 20px tolerance
        face = _MockFace(nose_x=50, nose_y=50, width=10, height=12)
        v = _make_verifier_with_region_response([(555, 440)])
        frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        v.verify_faces_in_frame(frame, [face], 1000, 1000)
        assert face.is_human is True

    def test_well_outside_tolerance_fails(self):
        """Pose nose well outside tolerance → fails."""
        # Face at (50%, 50%), bbox [450-550, 440-560], tol_x=20
        # Nose at (600, 500) → 50px outside bbox right edge, > 20px tolerance
        face = _MockFace(nose_x=50, nose_y=50, width=10, height=12)
        v = _make_verifier_with_region_response([(600, 500)])
        frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        v.verify_faces_in_frame(frame, [face], 1000, 1000)
        assert face.is_human is False
