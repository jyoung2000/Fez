"""Face mesh validator -- runs MediaPipe Face Mesh on a pre-cropped image
to boost confidence of saliency-promoted subject tracks.

Does NOT run as a primary detector -- it only validates candidates that
the promotion gate has already decided to promote.
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Confidence boost applied when face mesh successfully finds a face on the crop
MESH_CONFIDENCE_BOOST = 0.25


class FaceMeshValidator:
    """Singleton wrapper around MediaPipe Face Mesh for crop validation."""

    def __init__(self, min_detection_confidence: float = 0.3):
        self._mesh = None
        self._available = False
        try:
            import mediapipe as mp
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=min_detection_confidence,
            )
            self._available = True
            logger.info("FaceMeshValidator: mediapipe face mesh loaded")
        except Exception as e:
            logger.warning("FaceMeshValidator: unavailable (%s), validation disabled", e)

    @property
    def available(self) -> bool:
        return self._available

    def validate(self, image_crop_bgr: np.ndarray) -> tuple:
        """Run Face Mesh on a single BGR crop.

        Returns (found: bool, confidence_boost: float).
        `found=True` means Face Mesh found a face in the crop -- the caller
        should add `confidence_boost` to the saliency track's base confidence.
        """
        if not self._available or image_crop_bgr is None:
            return (False, 0.0)
        if image_crop_bgr.size == 0:
            return (False, 0.0)

        h, w = image_crop_bgr.shape[:2]
        if h < 20 or w < 20:
            return (False, 0.0)

        try:
            rgb = cv2.cvtColor(image_crop_bgr, cv2.COLOR_BGR2RGB)
            results = self._mesh.process(rgb)
            if results.multi_face_landmarks:
                return (True, MESH_CONFIDENCE_BOOST)
            return (False, 0.0)
        except Exception as e:
            logger.warning("FaceMeshValidator: validate failed: %s", e)
            return (False, 0.0)

    def close(self):
        if self._mesh is not None:
            try:
                self._mesh.close()
            except Exception:
                pass


# Module-level singleton -- MediaPipe is expensive to init
_VALIDATOR: Optional[FaceMeshValidator] = None


def get_validator() -> FaceMeshValidator:
    global _VALIDATOR
    if _VALIDATOR is None:
        _VALIDATOR = FaceMeshValidator()
    return _VALIDATOR


def reset_validator() -> None:
    """Reset the singleton (for testing)."""
    global _VALIDATOR
    if _VALIDATOR is not None:
        _VALIDATOR.close()
    _VALIDATOR = None
