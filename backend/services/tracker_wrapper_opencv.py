"""OpenCV tracker backend (legacy). KCF / MOSSE / CSRT / MIL.

Phase-A migration: this module owns the original ``SlotTracker``
implementation. The public interface is unchanged so the dispatcher
in :mod:`tracker_wrapper` can route to it without callers caring.

The OpenCV trackers are circa-2017 — they drift on rapid motion and
lose targets behind brief occlusions. They are kept as the CPU /
no-GPU fallback for SAMURAI (see :mod:`samurai_tracker`).
"""

from __future__ import annotations

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


class OpenCvSlotTracker:
    """One OpenCV tracker per FaceRegistry slot.

    Bbox-only output. ``update()`` returns ``(ok, (x, y, w, h), None)``
    so the signature matches :class:`SamuraiTracker.update` — the third
    component (mask) is always ``None`` from the OpenCV path.
    """

    def __init__(self, slot_id: int, backend: str = "KCF"):
        self.slot_id = slot_id
        self.backend = backend
        self._tracker = _create_tracker_instance(backend)
        self._initialized = False
        self._last_bbox: Optional[tuple] = None

    def init(self, frame_bgr: np.ndarray, bbox_pixels: tuple) -> bool:
        x, y, w, h = bbox_pixels
        if w < 10 or h < 10:
            return False
        if self._tracker is None:
            return False
        try:
            self._tracker.init(frame_bgr, (int(x), int(y), int(w), int(h)))
            self._initialized = True
            self._last_bbox = (float(x), float(y), float(w), float(h))
            return True
        except Exception:
            return False

    def update(self, frame_bgr: np.ndarray):
        """Returns ``(ok, bbox, None)`` — mask channel is always None for OpenCV."""
        if not self._initialized or self._tracker is None:
            return (False, self._last_bbox, None)
        try:
            ok, bbox = self._tracker.update(frame_bgr)
            if ok:
                self._last_bbox = (
                    float(bbox[0]), float(bbox[1]),
                    float(bbox[2]), float(bbox[3]),
                )
                return (True, self._last_bbox, None)
            return (False, self._last_bbox, None)
        except Exception:
            return (False, self._last_bbox, None)

    def reset(self, frame_bgr: Optional[np.ndarray] = None,
              bbox_pixels: Optional[tuple] = None) -> None:
        """Re-create tracker. Re-initialize if both frame + bbox given."""
        self._tracker = _create_tracker_instance(self.backend)
        self._initialized = False
        self._last_bbox = None
        if frame_bgr is not None and bbox_pixels is not None:
            self.init(frame_bgr, bbox_pixels)


# ── Legacy-shape adapter ─────────────────────────────────────────────
#
# Existing callers of ``SlotTracker`` use the old (.update returning
# Optional[tuple]) signature. Provide a class with the legacy shape so
# nothing in the existing codebase breaks during the Phase-A swap; the
# new dispatcher in :mod:`tracker_wrapper` returns this class when the
# backend is ``opencv``.


class SlotTracker(OpenCvSlotTracker):
    """Legacy-shape OpenCV tracker.

    Keeps ``update()`` returning ``Optional[tuple]`` (bbox or None) for
    pre-Phase-A callers. New code should use the dispatcher in
    :mod:`tracker_wrapper` directly.
    """

    def update(self, frame_bgr: np.ndarray):  # type: ignore[override]
        ok, bbox, _mask = super().update(frame_bgr)
        return bbox if ok else None
