"""Lightweight object tracking for non-face subjects.

When no face is detected in a frame, we need SOMETHING to track for the
crop position. This module uses the spatiotemporal saliency tracker to find
the most visually important region and track it across frames.

For the GTX 1650 constraint: NO GPU usage. All CPU-based.

Pipeline:
1. If faces present -> use face tracking (face_detector.py handles this)
2. If no faces -> compute spatiotemporal saliency -> find salient bboxes
3. Pick dominant region per frame (highest saliency_score)
4. Output: (timestamp, object_x) tuples in same format as face subject_x
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    x_center: float    # 0-100
    y_center: float    # 0-100
    width: float       # % of frame
    height: float      # % of frame
    confidence: float  # 0-1
    object_type: str   # "saliency" | "motion" | "text"


def track_objects_in_frames(
    frame_paths: list,
    face_results: list = None,
    return_confidence: bool = False,
) -> list:
    """Track the primary non-face subject across frames.

    Returns (timestamp, subject_x) tuples compatible with the existing
    subject tracking pipeline -- can be merged directly into keyframes.

    When return_confidence=True, returns list[(timestamp, x_center, confidence)]
    instead, for use by the AutoFlip reframe path.

    Uses the spatiotemporal saliency tracker internally. For each saliency
    region found, picks the one with the highest saliency_score per timestamp.

    DEPRECATED: This function collapses SaliencyRegion bboxes to x-only tuples.
    Prefer passing the SaliencyRegion list directly as frame_saliency to
    scene_focus.aggregate_scene_focus for full bbox data.
    """
    from backend.services.saliency_tracker import track_saliency_in_frames

    try:
        regions = track_saliency_in_frames(frame_paths, face_results)
    except Exception as e:
        logger.warning("[ObjectTracker] saliency tracker failed: %s", e)
        return []

    # Group by timestamp, pick the dominant region per frame
    by_time = {}
    for r in regions:
        prev = by_time.get(r.timestamp)
        if prev is None or r.saliency_score > prev.saliency_score:
            by_time[r.timestamp] = r

    keyframes = []
    for t in sorted(by_time.keys()):
        r = by_time[t]
        if return_confidence:
            keyframes.append((t, r.x, r.saliency_score))
        else:
            keyframes.append((t, round(r.x)))

    logger.info("[ObjectTracker] %d dominant saliency keyframes from %d regions",
                len(keyframes), len(regions))
    return keyframes
