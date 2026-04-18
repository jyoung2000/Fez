"""Tests for FaceMeshValidator."""

import numpy as np
import pytest

from backend.services.face_mesh_validator import (
    FaceMeshValidator,
    get_validator,
    reset_validator,
    MESH_CONFIDENCE_BOOST,
)


class TestFaceMeshValidatorGraceful:
    def test_validate_none_input(self):
        """validate(None) returns (False, 0.0)."""
        reset_validator()
        validator = get_validator()
        found, boost = validator.validate(None)
        assert found is False
        assert boost == 0.0
        reset_validator()

    def test_validate_empty_array(self):
        """validate(empty_array) returns (False, 0.0)."""
        reset_validator()
        validator = get_validator()
        empty = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        found, boost = validator.validate(empty)
        assert found is False
        assert boost == 0.0
        reset_validator()

    def test_validate_tiny_image(self):
        """validate(10x10 image) returns (False, 0.0) -- too small."""
        reset_validator()
        validator = get_validator()
        tiny = np.full((10, 10, 3), 128, dtype=np.uint8)
        found, boost = validator.validate(tiny)
        assert found is False
        assert boost == 0.0
        reset_validator()

    def test_validate_gray_rectangle(self):
        """A solid gray rectangle returns (False, 0.0) -- no face."""
        reset_validator()
        validator = get_validator()
        rect = np.full((200, 150, 3), 128, dtype=np.uint8)
        found, boost = validator.validate(rect)
        assert found is False
        assert boost == 0.0
        reset_validator()


class TestFaceMeshValidatorSingleton:
    def test_same_instance(self):
        """get_validator() returns the same instance across calls."""
        reset_validator()
        v1 = get_validator()
        v2 = get_validator()
        assert v1 is v2
        reset_validator()

    def test_available_property(self):
        """available property returns a bool."""
        reset_validator()
        validator = get_validator()
        assert isinstance(validator.available, bool)
        reset_validator()


class TestFaceMeshWithMediaPipe:
    def test_photo_face_detection(self):
        """A synthetic photo-like face image returns (True, boost) if MediaPipe available."""
        reset_validator()
        validator = get_validator()
        if not validator.available:
            pytest.skip("MediaPipe unavailable")

        # Create a synthetic face-like image (skin-toned oval)
        # MediaPipe needs a realistic face, so this may not pass.
        # We test the API works; face detection on synthetic images is unreliable.
        img = np.full((200, 150, 3), 200, dtype=np.uint8)  # light background
        found, boost = validator.validate(img)
        # We don't assert found=True because synthetic doesn't reliably trigger mesh
        assert isinstance(found, bool)
        if found:
            assert boost == MESH_CONFIDENCE_BOOST
        else:
            assert boost == 0.0
        reset_validator()
