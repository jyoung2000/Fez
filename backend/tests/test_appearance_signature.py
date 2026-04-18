"""Tests for appearance signature module."""

import numpy as np
import cv2

from backend.services.appearance_signature import (
    compute_appearance_signature,
    appearance_distance,
    same_subject,
    merge_signatures,
)


def _make_solid_color(bgr, h=100, w=100):
    """Create a solid-color BGR image."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = bgr
    return img


class TestComputeAppearanceSignature:
    def test_none_input(self):
        """None input returns None."""
        assert compute_appearance_signature(None) is None

    def test_empty_input(self):
        """Empty array returns None."""
        assert compute_appearance_signature(np.array([], dtype=np.uint8).reshape(0, 0, 3)) is None

    def test_tiny_input(self):
        """Tiny (5x5) image returns None."""
        assert compute_appearance_signature(np.zeros((5, 5, 3), dtype=np.uint8)) is None

    def test_valid_red_square(self):
        """100x100 red square returns a valid histogram."""
        red = _make_solid_color((0, 0, 255))
        sig = compute_appearance_signature(red)
        assert sig is not None
        assert sig.dtype == np.float32
        assert sig.shape == (8 * 8 * 8,)  # 512 bins
        assert sig.max() > 0  # normalized, so should have non-zero values


class TestAppearanceDistance:
    def test_identical_signatures(self):
        """Distance of identical signatures is ~0.0."""
        red = _make_solid_color((0, 0, 255))
        sig = compute_appearance_signature(red)
        dist = appearance_distance(sig, sig)
        assert dist < 0.01

    def test_different_colors_large_distance(self):
        """Solid red vs solid blue returns distance >0.5."""
        red = _make_solid_color((0, 0, 255))  # BGR red
        blue = _make_solid_color((255, 0, 0))  # BGR blue
        sig_r = compute_appearance_signature(red)
        sig_b = compute_appearance_signature(blue)
        dist = appearance_distance(sig_r, sig_b)
        assert dist > 0.5

    def test_similar_colors_small_distance(self):
        """Red vs orange-red returns distance <0.35."""
        red = _make_solid_color((0, 0, 255))
        orange_red = _make_solid_color((0, 80, 255))
        sig_r = compute_appearance_signature(red)
        sig_o = compute_appearance_signature(orange_red)
        dist = appearance_distance(sig_r, sig_o)
        assert dist < 0.35

    def test_none_none(self):
        """Distance of (None, None) returns 1.0."""
        assert appearance_distance(None, None) == 1.0

    def test_none_valid(self):
        """Distance of (None, valid) returns 1.0."""
        red = _make_solid_color((0, 0, 255))
        sig = compute_appearance_signature(red)
        assert appearance_distance(None, sig) == 1.0
        assert appearance_distance(sig, None) == 1.0


class TestSameSubject:
    def test_same_image(self):
        """Same image passed twice -> True."""
        red = _make_solid_color((0, 0, 255))
        sig = compute_appearance_signature(red)
        assert same_subject(sig, sig) is True

    def test_different_colors(self):
        """Very different colors -> False."""
        red = _make_solid_color((0, 0, 255))
        blue = _make_solid_color((255, 0, 0))
        sig_r = compute_appearance_signature(red)
        sig_b = compute_appearance_signature(blue)
        assert same_subject(sig_r, sig_b) is False


class TestMergeSignatures:
    def test_merge_between_inputs(self):
        """Merged signature has distance between the two inputs."""
        red = _make_solid_color((0, 0, 255))
        blue = _make_solid_color((255, 0, 0))
        sig_r = compute_appearance_signature(red)
        sig_b = compute_appearance_signature(blue)
        merged = merge_signatures(sig_r, sig_b, weight_a=0.5)
        # Merged should be closer to each input than they are to each other
        d_rb = appearance_distance(sig_r, sig_b)
        d_r_merged = appearance_distance(sig_r, merged)
        d_b_merged = appearance_distance(sig_b, merged)
        assert d_r_merged < d_rb
        assert d_b_merged < d_rb

    def test_merge_none_a(self):
        """merge(None, sig) returns sig."""
        red = _make_solid_color((0, 0, 255))
        sig = compute_appearance_signature(red)
        result = merge_signatures(None, sig)
        assert np.array_equal(result, sig)

    def test_merge_none_b(self):
        """merge(sig, None) returns sig."""
        red = _make_solid_color((0, 0, 255))
        sig = compute_appearance_signature(red)
        result = merge_signatures(sig, None)
        assert np.array_equal(result, sig)
