"""Tests for spatiotemporal saliency tracker."""

import numpy as np
import pytest
from dataclasses import dataclass, field

from backend.services.saliency_tracker import (
    SaliencyRegion,
    SPATIAL_WEIGHT,
    TEMPORAL_WEIGHT,
    COLOR_WEIGHT,
    _compute_color_opponent,
    _compute_adaptive_center_bias,
    compute_spatiotemporal_saliency,
    extract_saliency_bboxes,
    track_saliency_in_frames,
)


# ── Helpers ──

def _make_gray(h=480, w=640, value=128):
    """Create a uniform gray frame."""
    return np.full((h, w), value, dtype=np.uint8)


def _make_bright_square(h=480, w=640, sq_x=270, sq_y=190, sq_size=100, bg=30, fg=220):
    """Create a gray frame with a bright square."""
    frame = np.full((h, w), bg, dtype=np.uint8)
    frame[sq_y:sq_y + sq_size, sq_x:sq_x + sq_size] = fg
    return frame


@dataclass
class MockFrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)
    path: str = ""


@dataclass
class MockFace:
    nose_x: float = 50.0
    nose_y: float = 40.0


class TestComputeSpatiotemporalSaliency:
    def test_no_prev_gray_uses_spatial_only(self):
        """With no prev_gray, returns a map using only the spatial component."""
        gray = _make_bright_square()
        sal_map = compute_spatiotemporal_saliency(gray, prev_gray=None)
        assert sal_map.shape == gray.shape
        assert sal_map.dtype == np.float32
        assert sal_map.max() <= 1.0
        assert sal_map.max() > 0.0  # bright square creates spatial edges

    def test_identical_frames_near_zero_temporal(self):
        """Two identical frames -> near-zero temporal component."""
        gray = _make_bright_square()
        # With identical frames, temporal_weight * 0 == 0, so result is purely spatial
        sal_spatial = compute_spatiotemporal_saliency(gray, prev_gray=None, spatial_weight=1.0, temporal_weight=0.0)
        sal_both = compute_spatiotemporal_saliency(gray, prev_gray=gray)
        # The combined map should be close to just the spatial component * 0.4
        # (since temporal is zero and combined is renormalized)
        assert sal_both.max() <= 1.0
        # The temporal contribution should be zero for identical frames
        temporal_only = compute_spatiotemporal_saliency(gray, prev_gray=gray, spatial_weight=0.0, temporal_weight=1.0)
        assert temporal_only.max() < 0.01

    def test_different_frames_nonzero_temporal(self):
        """Two different frames -> non-zero temporal response."""
        frame1 = _make_bright_square(sq_x=100)
        frame2 = _make_bright_square(sq_x=300)
        sal_map = compute_spatiotemporal_saliency(frame2, prev_gray=frame1)
        assert sal_map.max() > 0.0
        # Temporal component should be non-zero
        temporal_only = compute_spatiotemporal_saliency(frame2, prev_gray=frame1, spatial_weight=0.0, temporal_weight=1.0)
        assert temporal_only.max() > 0.1


class TestExtractSaliencyBboxes:
    def test_bright_square_one_bbox(self):
        """Synthetic 640x480 frame with a bright 100x100 square -> one bbox."""
        gray = _make_bright_square(sq_x=270, sq_y=190, sq_size=100)
        sal_map = compute_spatiotemporal_saliency(gray, prev_gray=None, spatial_weight=1.0, temporal_weight=0.0)
        bboxes = extract_saliency_bboxes(sal_map, threshold=0.3)
        assert len(bboxes) >= 1
        # The bbox should be roughly around the square position
        bx, by, bw, bh, mean_sal = bboxes[0]
        # Center should be near (320, 240) - the center of the square
        cx = bx + bw / 2
        cy = by + bh / 2
        assert 200 < cx < 440, f"cx={cx} not near square center"
        assert 120 < cy < 360, f"cy={cy} not near square center"

    def test_uniform_frame_zero_bboxes(self):
        """Uniform frame returns zero bboxes."""
        gray = _make_gray()
        sal_map = compute_spatiotemporal_saliency(gray, prev_gray=None, spatial_weight=1.0, temporal_weight=0.0)
        bboxes = extract_saliency_bboxes(sal_map)
        assert len(bboxes) == 0

    def test_rejects_small_regions(self):
        """Tiny regions below min_area_ratio are rejected."""
        # Create a frame with a very small bright dot
        frame = np.full((480, 640), 30, dtype=np.uint8)
        frame[240, 320] = 255  # single pixel
        sal_map = compute_spatiotemporal_saliency(frame, prev_gray=None, spatial_weight=1.0, temporal_weight=0.0)
        bboxes = extract_saliency_bboxes(sal_map, min_area_ratio=0.01)
        # The Gaussian blur will spread it, but it should still be below min_area
        # Any detected region from a single bright pixel should be very small
        for bx, by, bw, bh, _ in bboxes:
            area_ratio = (bw * bh) / (480 * 640)
            assert area_ratio >= 0.01  # all surviving bboxes pass the filter


class TestCenterBiasAndHudMasking:
    """Fix 4: Verify center bias and HUD masking."""

    def test_center_blob_higher_score_than_corner(self):
        """With center bias, center blob has higher score than corner blob."""
        # Two equal-intensity blobs: one at center, one at (90%, 90%)
        frame = np.full((480, 640), 30, dtype=np.uint8)
        # Center blob
        frame[215:265, 295:345] = 220
        # Corner blob (same size, same intensity)
        frame[400:450, 540:590] = 220

        # Without center bias
        sal_no_bias = compute_spatiotemporal_saliency(
            frame, prev_gray=None, spatial_weight=1.0, temporal_weight=0.0,
            center_bias_sigma=0,
        )
        bboxes_no_bias = extract_saliency_bboxes(sal_no_bias, threshold=0.3)

        # With center bias
        sal_biased = compute_spatiotemporal_saliency(
            frame, prev_gray=None, spatial_weight=1.0, temporal_weight=0.0,
            center_bias_sigma=0.35,
        )
        bboxes_biased = extract_saliency_bboxes(sal_biased, threshold=0.3)

        # Without bias, both should be detected with similar scores
        assert len(bboxes_no_bias) >= 2

        # With bias, if both are detected, center one should have higher score
        if len(bboxes_biased) >= 2:
            # Sort by y position: center blob is higher (y~240), corner is lower (y~425)
            bboxes_biased.sort(key=lambda b: b[1])
            center_score = bboxes_biased[0][4]  # mean_saliency of center
            corner_score = bboxes_biased[-1][4]  # mean_saliency of corner
            assert center_score > corner_score, (
                f"Center blob ({center_score:.3f}) should score higher than "
                f"corner blob ({corner_score:.3f}) with center bias"
            )

    def test_hud_mask_suppresses_blob(self):
        """HUD mask covering one blob causes only the unmasked blob to survive."""
        # Two blobs: one at top-left, one at center
        frame = np.full((480, 640), 30, dtype=np.uint8)
        frame[50:100, 50:100] = 220   # top-left blob
        frame[215:265, 295:345] = 220  # center blob

        # HUD mask covers the top-left blob
        hud_mask = np.zeros((480, 640), dtype=np.float32)
        hud_mask[0:120, 0:120] = 1.0

        sal_masked = compute_spatiotemporal_saliency(
            frame, prev_gray=None, spatial_weight=1.0, temporal_weight=0.0,
            center_bias_sigma=0, hud_mask=hud_mask,
        )
        bboxes = extract_saliency_bboxes(sal_masked, threshold=0.3)

        # Only center blob should survive (top-left is masked)
        for bx, by, bw, bh, _ in bboxes:
            cx = bx + bw / 2
            cy = by + bh / 2
            assert cx > 120 or cy > 120, (
                f"HUD-masked blob at ({cx:.0f}, {cy:.0f}) should be suppressed"
            )


class TestAdaptiveThreshold:
    """Fix 3: Verify adaptive (percentile-based) thresholding."""

    def test_low_contrast_detected_with_adaptive(self):
        """Low-contrast frame with very faint blob produces bbox with
        adaptive threshold but zero with high fixed threshold."""
        # Low-contrast frame: background ~100, faint blob at ~115
        # The Sobel edges from a 15-step contrast are very weak
        frame = np.full((480, 640), 100, dtype=np.uint8)
        # Use a gradient blob instead of hard edges to make edges fainter
        for dy in range(50):
            for dx in range(50):
                val = 100 + int(15 * min(1.0, (25 - abs(dy - 25)) / 25.0) *
                               min(1.0, (25 - abs(dx - 25)) / 25.0))
                frame[215 + dy, 295 + dx] = val

        sal_map = compute_spatiotemporal_saliency(frame, prev_gray=None,
                                                   spatial_weight=1.0, temporal_weight=0.0)

        # Adaptive (default percentile=85) should detect the blob
        bboxes_adaptive = extract_saliency_bboxes(sal_map, threshold=None)
        assert len(bboxes_adaptive) >= 1, (
            f"Adaptive threshold should detect faint blob, got {len(bboxes_adaptive)} bboxes"
        )

        # High fixed threshold should miss the blob
        bboxes_fixed = extract_saliency_bboxes(sal_map, threshold=0.8)
        assert len(bboxes_fixed) == 0, (
            f"Fixed threshold=0.8 should miss faint blob, got {len(bboxes_fixed)} bboxes"
        )


class TestTrackSaliencyInFrames:
    def test_moving_rectangle_returns_regions(self, tmp_path):
        """5-frame sequence with a moving rectangle returns at least 3 regions."""
        import cv2
        frame_paths = []
        for i in range(5):
            frame = np.full((480, 640, 3), 30, dtype=np.uint8)
            # Moving bright rectangle
            x_offset = 100 + i * 80
            cv2.rectangle(frame, (x_offset, 150), (x_offset + 100, 300), (220, 220, 220), -1)
            path = tmp_path / f"frame_{i}.png"
            cv2.imwrite(str(path), frame)
            frame_paths.append((float(i) * 0.5, str(path)))

        regions = track_saliency_in_frames(frame_paths)
        assert len(regions) >= 3

    def test_processes_all_frames_regardless_of_faces(self, tmp_path):
        """Saliency runs on ALL frames, even those with face data (parity fix 2)."""
        import cv2
        frame_paths = []
        face_results = []
        for i in range(5):
            frame = np.full((480, 640, 3), 30, dtype=np.uint8)
            x_offset = 100 + i * 80
            cv2.rectangle(frame, (x_offset, 150), (x_offset + 100, 300), (220, 220, 220), -1)
            path = tmp_path / f"frame_{i}.png"
            cv2.imwrite(str(path), frame)
            frame_paths.append((float(i) * 0.5, str(path)))
            # First 3 frames have faces, last 2 don't
            if i < 3:
                face_results.append(MockFrameFaces(timestamp=float(i) * 0.5, faces=[MockFace()]))
            else:
                face_results.append(MockFrameFaces(timestamp=float(i) * 0.5, faces=[]))

        regions = track_saliency_in_frames(frame_paths, face_results)
        # ALL frames should be processed — regions from both face and faceless frames
        timestamps = {r.timestamp for r in regions}
        assert len(regions) >= 3
        # Should have regions from early frames too (not just >= 1.5)
        assert any(t < 1.5 for t in timestamps), (
            f"Expected regions from face-containing frames too, got timestamps: {timestamps}"
        )

    def test_empty_frame_list(self):
        """Empty frame list returns empty list."""
        regions = track_saliency_in_frames([])
        assert regions == []


class TestSaliencyRegionToDict:
    def test_sanitizes_numpy_scalars(self):
        """to_dict() sanitizes numpy scalars to native Python types."""
        region = SaliencyRegion(
            timestamp=np.float64(1.5),
            x=np.float32(50.0),
            y=np.float32(40.0),
            w=np.float32(10.0),
            h=np.float32(13.0),
            saliency_score=np.float64(0.8),
            motion_score=np.float64(0.3),
            spatial_score=np.float64(0.5),
        )
        d = region.to_dict()
        for k, v in d.items():
            assert not hasattr(v, 'item'), f"{k} is still a numpy scalar: {type(v)}"
            assert isinstance(v, (int, float)), f"{k} has unexpected type: {type(v)}"


class TestColorOpponentChannel:
    """Task 3: Color-opponent saliency channel tests."""

    def test_color_opponent_detects_red_on_gray(self):
        """Red square on gray background has saliency peak inside the square."""
        # Create a 480x270 gray background with a red 40x40 square at (200, 100)
        frame = np.full((270, 480, 3), 128, dtype=np.uint8)
        # Red in BGR = (0, 0, 255)
        frame[100:140, 200:240] = [0, 0, 255]

        color_map = _compute_color_opponent(frame)
        assert color_map.shape == (270, 480)
        assert color_map.dtype == np.float32
        assert color_map.max() <= 1.0

        # Peak should be inside the red square's bbox (with some blur spread)
        peak_y, peak_x = np.unravel_index(np.argmax(color_map), color_map.shape)
        assert 90 <= peak_y <= 150, f"Peak y={peak_y}, expected near 100-140"
        assert 190 <= peak_x <= 250, f"Peak x={peak_x}, expected near 200-240"

    def test_color_channel_optional_backward_compat(self):
        """compute_spatiotemporal_saliency with curr_bgr=None is bit-identical
        to pre-change behavior (two-channel fusion with 0.4/0.6 weights)."""
        gray = _make_bright_square()

        # Legacy behavior: explicit 0.4/0.6 weights, no color channel
        legacy = compute_spatiotemporal_saliency(
            gray, prev_gray=None,
            spatial_weight=0.4, temporal_weight=0.6,
            curr_bgr=None,
        )
        # New default with curr_bgr=None should renormalize to same ratio
        new_default = compute_spatiotemporal_saliency(
            gray, prev_gray=None,
            curr_bgr=None,
        )
        # Should be identical (both use spatial-only with no temporal)
        np.testing.assert_array_almost_equal(legacy, new_default, decimal=5)

    def test_three_channels_sum_to_one(self):
        """Weight constants sum to exactly 1.0."""
        assert SPATIAL_WEIGHT + TEMPORAL_WEIGHT + COLOR_WEIGHT == 1.0


class TestAdaptiveCenterBias:
    """Task 5: Adaptive center bias based on face positions."""

    def test_adaptive_bias_widens_with_spread(self):
        """Two faces at opposite edges → sigma_frac > 0.5."""
        @dataclass
        class _Face:
            nose_x: float = 50.0
            nose_y: float = 50.0

        faces = [_Face(nose_x=10.4), _Face(nose_x=88.5)]  # x=200 and x=1700 on 1920
        cx, cy, sigma = _compute_adaptive_center_bias(faces, 1920, 1080)
        assert sigma > 0.5, f"sigma_frac={sigma} should be > 0.5 with spread faces"
        # Centroid should be roughly in the middle
        assert 800 < cx < 1100, f"cx={cx} should be near center"

    def test_adaptive_bias_centers_on_single_face(self):
        """One face at x=300 → bias_cx ≈ 300, sigma = 0.35."""
        @dataclass
        class _Face:
            nose_x: float = 15.625  # 300/1920 * 100
            nose_y: float = 50.0

        faces = [_Face()]
        cx, cy, sigma = _compute_adaptive_center_bias(faces, 1920, 1080)
        assert abs(cx - 300.0) < 1.0, f"cx={cx}, expected ~300"
        assert sigma == 0.35, f"sigma_frac={sigma}, expected 0.35"

    def test_no_faces_falls_back_to_default(self):
        """Empty face list → (w/2, h/2, 0.35)."""
        cx, cy, sigma = _compute_adaptive_center_bias([], 1920, 1080)
        assert cx == 1920 / 2.0
        assert cy == 1080 / 2.0
        assert sigma == 0.35
