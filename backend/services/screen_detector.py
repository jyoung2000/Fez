"""Screen content detection using OpenCV heuristics.

Detects if a video frame contains screen share / slides / text-heavy content
using simple computer vision heuristics (no AI needed, runs in <5ms per frame).

CONSERVATIVE: designed to avoid false positives on videos with text
overlays (titles, lower thirds, graphics). Only returns True when the
frame looks like an actual screen recording, presentation slide, or
code editor — not a video with text on it.

Used by the layout engine to decide between SCREENSHARE layout mode and
standard speaker layouts.
"""
import logging

logger = logging.getLogger(__name__)


def detect_screen_content(frame_path: str) -> bool:
    """Detect if a frame contains screen share / slides / text-heavy content.

    Uses conservative heuristics to avoid false positives on videos with
    text overlays (titles, lower thirds like "VERZUZ", "TANK VS TYRESE").

    Requires ALL THREE indicators to trigger:
    1. High edge density (>12%) — UI elements, code lines
    2. Very low saturation (<40) — screenshare is mostly grayscale
    3. Large uniform regions (>50%) — flat-color UI panels, not video texture
    """
    import cv2
    import numpy as np

    img = cv2.imread(frame_path)
    if img is None:
        return False

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Heuristic 1: Edge density — screenshare has MANY straight edges
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / (h * w)

    # Heuristic 2: Saturation — screenshare is VERY desaturated
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    avg_saturation = np.mean(hsv[:, :, 1])

    # Heuristic 3: Texture uniformity — screenshare has large FLAT regions
    # (solid-color UI panels, IDE background). Video with text overlays
    # has complex natural textures behind the text.
    # Use local standard deviation to measure texture complexity.
    gray_f = gray.astype(np.float32)
    local_mean = cv2.blur(gray_f, (31, 31))
    local_sq_mean = cv2.blur(gray_f ** 2, (31, 31))
    local_var = local_sq_mean - local_mean ** 2
    local_std = np.sqrt(np.maximum(local_var, 0))
    uniform_ratio = float(np.mean(local_std < 8.0))

    # Require ALL THREE indicators (was >= 2, now >= 3)
    score = 0
    if edge_density > 0.12:    # Was 0.08 — raised to avoid text overlay edges
        score += 1
    if avg_saturation < 40:    # Was 60 — natural video rarely this desaturated
        score += 1
    if uniform_ratio > 0.50:   # NEW — must have >50% flat regions
        score += 1

    return score >= 3


def detect_screen_content_batch(frame_paths: list) -> dict:
    """Run screen content detection on multiple frames.

    Args:
        frame_paths: list of (timestamp, path) tuples

    Returns:
        dict mapping timestamp -> bool (True if screen content detected)
    """
    results = {}
    for timestamp, path in frame_paths:
        try:
            results[timestamp] = detect_screen_content(str(path))
        except Exception:
            results[timestamp] = False
    screen_count = sum(1 for v in results.values() if v)
    if screen_count > 0:
        logger.info(
            "[ScreenDetect] %d/%d frames detected as screen content",
            screen_count, len(frame_paths),
        )
    return results
