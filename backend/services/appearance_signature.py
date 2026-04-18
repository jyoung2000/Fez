"""Appearance signatures for cross-shot subject identity.

Uses HSV color histograms and Bhattacharyya distance to match the same
subject across different shots. Handles the Speed Racer case where
a character's color palette is consistent across cuts even though
face detection doesn't bridge the cut.
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Distance threshold for "same subject" -- tuned empirically
# 0.0 = identical, 1.0 = completely different
APPEARANCE_MATCH_THRESHOLD = 0.35

# Histogram bin counts per channel (H, S, V)
# 8 bins per channel = 512 total dimensions, compact enough for fast compare
HIST_BINS = (8, 8, 8)


def compute_appearance_signature(image_crop_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Compute a normalized HSV histogram from a BGR image crop.

    Returns a flattened float32 array, or None if the crop is invalid.
    """
    if image_crop_bgr is None or image_crop_bgr.size == 0:
        return None
    h, w = image_crop_bgr.shape[:2]
    if h < 10 or w < 10:
        return None

    try:
        hsv = cv2.cvtColor(image_crop_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None, HIST_BINS,
            [0, 180, 0, 256, 0, 256],
        )
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten().astype(np.float32)
    except Exception as e:
        logger.warning("appearance_signature: compute failed: %s", e)
        return None


def appearance_distance(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Bhattacharyya distance between two signatures.

    Returns a value in [0, 1] where 0 means identical.
    """
    if sig_a is None or sig_b is None:
        return 1.0
    if sig_a.shape != sig_b.shape:
        return 1.0
    try:
        a = sig_a.reshape(HIST_BINS).astype(np.float32)
        b = sig_b.reshape(HIST_BINS).astype(np.float32)
        return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))
    except Exception as e:
        logger.warning("appearance_distance: compare failed: %s", e)
        return 1.0


def same_subject(sig_a: np.ndarray, sig_b: np.ndarray,
                 threshold: float = APPEARANCE_MATCH_THRESHOLD) -> bool:
    """True if two signatures match within threshold."""
    return appearance_distance(sig_a, sig_b) < threshold


def merge_signatures(sig_a: np.ndarray, sig_b: np.ndarray,
                     weight_a: float = 0.5) -> np.ndarray:
    """Running-average merge of two signatures for updating persistent identity."""
    if sig_a is None:
        return sig_b
    if sig_b is None:
        return sig_a
    return (sig_a * weight_a + sig_b * (1.0 - weight_a)).astype(np.float32)
