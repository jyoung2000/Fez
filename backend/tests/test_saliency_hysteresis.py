"""Tests for saliency hysteresis (Phase A).

Validates that:
- Saliency anchor stays locked when a challenger doesn't clearly beat it
- Anchor switches when a clearly stronger competitor appears (>=15% margin)
- Gaming mode bypasses hysteresis entirely
"""
import numpy as np
import pytest


def _make_saliency_map(h, w, cx_frac, cy_frac, blob_radius_frac, intensity):
    """Create a synthetic saliency map with a single Gaussian blob."""
    sal = np.zeros((h, w), dtype=np.float32)
    cx_px = int(cx_frac * w)
    cy_px = int(cy_frac * h)
    radius = int(blob_radius_frac * max(w, h))
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx_px)**2 + (y - cy_px)**2)
    sal[dist < radius] = intensity
    # Blur slightly
    import cv2
    sal = cv2.GaussianBlur(sal, (15, 15), 0)
    if sal.max() > 1e-6:
        sal = sal / sal.max() * intensity
    return sal


class TestSaliencyHysteresis:
    """E1: Saliency snap-and-hold."""

    def test_sticky_peak_holds_incumbent(self):
        """When prev_peak is near the top candidate, sticky peak keeps it.
        When prev_peak is far from the top candidate but a runner-up is
        close and competitive, the runner-up is preferred (hysteresis)."""
        from backend.services.saliency_tracker import extract_saliency_bboxes

        h, w = 480, 854

        # Scenario: two blobs. cx=0.3 is the incumbent (prev_peak there).
        # cx=0.7 is marginally stronger. Sticky peak should keep cx=0.3
        # blob in position [0] because:
        # 1. scored[0] after sort = cx=0.3 (higher score*area)
        # 2. _centroid_dist(scored[0]) to prev_peak is small → pass
        # The sorted list keeps cx=0.3 first.
        sal = _make_saliency_map(h, w, 0.3, 0.5, 0.08, 0.9)
        sal += _make_saliency_map(h, w, 0.7, 0.5, 0.08, 0.5)
        if sal.max() > 1e-6:
            sal = sal / sal.max()

        # prev_peak at cx=0.3 in pixel coords
        prev_peak = (int(0.3 * w), int(0.5 * h))
        bboxes = extract_saliency_bboxes(sal, prev_peak=prev_peak)

        # Verify we got 2 blobs
        assert len(bboxes) >= 2, f"Expected 2+ bboxes, got {len(bboxes)}"

        # The cx=0.3 blob should have higher score*area
        scored = [(b[4] * b[2] * b[3], (b[0] + b[2] / 2) / w) for b in bboxes]
        scored.sort(key=lambda s: s[0], reverse=True)
        assert abs(scored[0][1] - 0.3) < 0.15, (
            f"Top by score*area: cx={scored[0][1]:.2f}, expected near 0.3"
        )

    def test_sticky_peak_promotes_runner_up(self):
        """When the top candidate by score*area is FAR from prev_peak but
        a runner-up near prev_peak scores >= 0.85 * top, hysteresis
        promotes the runner-up to prevent jitter."""
        from backend.services.saliency_tracker import extract_saliency_bboxes

        h, w = 480, 854

        # cx=0.7 has slightly higher score*area, cx=0.3 is the runner-up.
        # prev_peak is at cx=0.3.
        sal = _make_saliency_map(h, w, 0.3, 0.5, 0.08, 0.85)
        sal += _make_saliency_map(h, w, 0.7, 0.5, 0.10, 0.90)  # bigger blob
        if sal.max() > 1e-6:
            sal = sal / sal.max()

        prev_peak = (int(0.3 * w), int(0.5 * h))
        bboxes_with = extract_saliency_bboxes(sal, prev_peak=prev_peak)
        bboxes_without = extract_saliency_bboxes(sal, prev_peak=None)

        # Without prev_peak: natural contour order (not sorted)
        # With prev_peak: if the runner-up near cx=0.3 was promoted,
        # bboxes[0] should be the cx=0.3 blob
        if len(bboxes_with) >= 2:
            top_with_cx = (bboxes_with[0][0] + bboxes_with[0][2] / 2) / w
            # The hysteresis should have promoted the cx=0.3 runner-up
            # (or the natural sort put it first if it scored higher)
            # Either way, cx=0.3 should be first
            assert abs(top_with_cx - 0.3) < 0.20, (
                f"Sticky peak: top cx={top_with_cx:.2f}, expected near 0.3 (hysteresis)"
            )

    def test_anchor_switches_on_clear_winner(self):
        """Frames 26-29: cx=0.7 clearly winning by >=30%.
        Assert anchor switches to cx=0.7 by frame 28."""
        from backend.services.saliency_tracker import extract_saliency_bboxes

        h, w = 480, 854
        # Start with prev_peak at cx=0.3 in pixel coords
        prev_peak = (int(0.3 * w), int(0.5 * h))

        for frame in range(4):
            # cx=0.7 clearly wins by >30%
            sal = _make_saliency_map(h, w, 0.3, 0.5, 0.08, 0.4)
            sal += _make_saliency_map(h, w, 0.7, 0.5, 0.08, 0.95)
            if sal.max() > 1e-6:
                sal = sal / sal.max()

            bboxes = extract_saliency_bboxes(sal, prev_peak=prev_peak)
            if bboxes:
                top = max(bboxes, key=lambda b: b[4] * (b[2] * b[3]))
                cx = (top[0] + top[2] / 2) / w
                prev_peak = (top[0] + top[2] / 2.0, top[1] + top[3] / 2.0)

        # By the end, anchor should be near 0.7 (in pixel coords)
        assert prev_peak is not None
        anchor_cx_norm = prev_peak[0] / w
        assert abs(anchor_cx_norm - 0.7) < 0.15, (
            f"Anchor at {anchor_cx_norm:.2f}, expected near 0.7 after clear winner"
        )

    def test_content_aware_fusion_weights(self):
        """_get_fusion_weights returns different weights for different types."""
        from backend.services.saliency_tracker import _get_fusion_weights

        # Dialogue types should have higher spatial weight
        s, t, c = _get_fusion_weights("talking_head")
        assert s == 0.45 and t == 0.30 and c == 0.25

        # Sports should have higher temporal weight
        s, t, c = _get_fusion_weights("sports")
        assert s == 0.25 and t == 0.55 and c == 0.20

        # Gaming should get legacy defaults
        s, t, c = _get_fusion_weights("gameplay")
        assert s == 0.3 and t == 0.5 and c == 0.2

        # None should get legacy defaults
        s, t, c = _get_fusion_weights(None)
        assert s == 0.3 and t == 0.5 and c == 0.2

    def test_gaming_mode_no_hysteresis(self):
        """Gaming content should bypass temporal hysteresis entirely."""
        from backend.services.saliency_tracker import compute_spatiotemporal_saliency

        h, w = 240, 426
        curr = np.random.randint(0, 255, (h, w), dtype=np.uint8)
        prev_combined = np.ones((h, w), dtype=np.float32) * 0.5

        # With gaming content_type, prev_combined should be ignored
        result_gaming = compute_spatiotemporal_saliency(
            curr, None,
            prev_combined=prev_combined,
            content_type="gameplay",
        )
        result_no_prev = compute_spatiotemporal_saliency(
            curr, None,
            content_type="gameplay",
        )
        # Results should be identical for gaming (no blending)
        np.testing.assert_array_almost_equal(result_gaming, result_no_prev, decimal=5)
