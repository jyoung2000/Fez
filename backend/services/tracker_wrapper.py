"""Thin wrapper around OpenCV trackers with backend selection.

Tries KCF first (fast and accurate), falls back to MOSSE (faster but less accurate)
if KCF cannot keep up with the runtime budget on a given clip.
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _create_tracker_instance(backend: str):
    """Create an OpenCV tracker instance for the given backend.

    Supports KCF (default), CSRT (opt-in), MOSSE (fast fallback), MIL (last resort).
    Falls through backends in order of preference if the requested one is unavailable.
    """
    backends = {
        "KCF": [
            lambda: cv2.TrackerKCF.create(),
            lambda: cv2.legacy.TrackerKCF.create() if hasattr(cv2, 'legacy') else None,
        ],
        "MOSSE": [
            lambda: cv2.legacy.TrackerMOSSE.create() if hasattr(cv2, 'legacy') else None,
            lambda: cv2.TrackerKCF.create(),
        ],
        "CSRT": [
            lambda: cv2.TrackerCSRT.create(),
            lambda: cv2.legacy.TrackerCSRT.create() if hasattr(cv2, 'legacy') else None,
            lambda: cv2.TrackerKCF.create(),
        ],
        "MIL": [
            lambda: cv2.TrackerMIL.create(),
        ],
    }

    for factory in backends.get(backend, backends["KCF"]):
        try:
            tracker = factory()
            if tracker is not None:
                return tracker
        except Exception:
            continue

    # Last resort — MIL is always available
    try:
        return cv2.TrackerMIL.create()
    except Exception as e:
        logger.error("No OpenCV tracker available: %s", e)
        return None


class SlotTracker:
    """One OpenCV tracker per FaceRegistry slot."""

    def __init__(self, slot_id: int, backend: str = "KCF"):
        self.slot_id = slot_id
        self.backend = backend
        self._tracker = _create_tracker_instance(backend)
        self._initialized = False
        self._last_bbox = None  # (x, y, w, h) in pixels

    def init(self, frame_bgr: np.ndarray, bbox_pixels: tuple) -> bool:
        """Initialize tracker on a frame with a bounding box (x, y, w, h) in pixels."""
        x, y, w, h = bbox_pixels
        if w < 10 or h < 10:
            return False
        if self._tracker is None:
            return False
        try:
            # OpenCV 4.x init() returns None (void) on success, raises on failure
            self._tracker.init(frame_bgr, (int(x), int(y), int(w), int(h)))
            self._initialized = True
            self._last_bbox = (float(x), float(y), float(w), float(h))
            return True
        except Exception:
            return False

    def update(self, frame_bgr: np.ndarray) -> Optional[tuple]:
        """Advance tracker by one frame. Returns (x, y, w, h) in pixels or None."""
        if not self._initialized or self._tracker is None:
            return None
        try:
            ok, bbox = self._tracker.update(frame_bgr)
            if ok:
                self._last_bbox = (float(bbox[0]), float(bbox[1]),
                                   float(bbox[2]), float(bbox[3]))
                return self._last_bbox
            return None
        except Exception:
            return None

    def reset(self, frame_bgr: np.ndarray, bbox_pixels: tuple):
        """Re-create tracker and re-initialize on the given bbox."""
        self._tracker = _create_tracker_instance(self.backend)
        self._initialized = False
        self.init(frame_bgr, bbox_pixels)
