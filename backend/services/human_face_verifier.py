"""Strict per-frame pose-based human face verification.

Runs MediaPipe Pose per face region and matches faces to pose noses by
STRICT containment: the pose nose MUST be inside the face bbox.

This fixes the figurine-in-front-of-torso case: the human's pose nose
is inside the human's face bbox (matched), the figurine's face bbox
contains no pose nose (rejected).

Feature-flagged via USE_HUMAN_VERIFICATION (default: true).
USE_STRICT_POSE_MATCHING (default: true) controls strict vs legacy mode.
When the verifier is unavailable (MediaPipe Pose not importable), it
fails open — all faces are treated as human.
"""

import logging
import os
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

USE_HUMAN_VERIFICATION = os.environ.get(
    "USE_HUMAN_VERIFICATION", "true"
).lower() in ("true", "1", "yes")

USE_STRICT_POSE_MATCHING = os.environ.get(
    "USE_STRICT_POSE_MATCHING", "true"
).lower() in ("true", "1", "yes")


class HumanFaceVerifier:
    """Runs MediaPipe Pose per face region to verify human faces via
    strict nose-in-bbox containment."""

    def __init__(self, min_pose_confidence: float = 0.4):
        self._pose = None
        self._available = False
        self._mp = None
        self._min_pose_confidence = min_pose_confidence
        try:
            import mediapipe as mp
            self._mp = mp
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=0,
                smooth_landmarks=False,
                min_detection_confidence=min_pose_confidence,
            )
            self._available = True
            logger.info("HumanFaceVerifier: mediapipe pose loaded (strict mode)")
        except Exception as e:
            logger.warning(
                "HumanFaceVerifier: unavailable (%s), fail-open enabled", e
            )

    @property
    def available(self) -> bool:
        return self._available

    def _get_pose_nose_for_region(
        self, frame_bgr: np.ndarray,
        face_cx_px: int, face_cy_px: int,
        face_w_px: int, face_h_px: int,
        source_width: int, source_height: int,
    ) -> Optional[Tuple[float, float]]:
        """Run pose on a sub-region around a face and return nose location
        in full-frame pixel coords, or None if no pose found."""
        import cv2

        # Expand region: wider horizontally (for shoulders) and downward (for torso)
        x0 = max(0, face_cx_px - face_w_px * 2)
        y0 = max(0, face_cy_px - face_h_px)
        x1 = min(source_width, face_cx_px + face_w_px * 2)
        y1 = min(source_height, face_cy_px + face_h_px * 4)
        if x1 - x0 < 30 or y1 - y0 < 30:
            return None

        region = frame_bgr[y0:y1, x0:x1]
        try:
            rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            results = self._pose.process(rgb)
            if not results.pose_landmarks:
                return None
            mp_pose = self._mp.solutions.pose.PoseLandmark
            nose = results.pose_landmarks.landmark[mp_pose.NOSE.value]
            left_shoulder = results.pose_landmarks.landmark[mp_pose.LEFT_SHOULDER.value]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.RIGHT_SHOULDER.value]

            # At least one shoulder must be visible
            if left_shoulder.visibility < 0.4 and right_shoulder.visibility < 0.4:
                return None

            # Convert nose from region-relative to full-frame pixels
            nose_x_full = x0 + nose.x * (x1 - x0)
            nose_y_full = y0 + nose.y * (y1 - y0)
            return (float(nose_x_full), float(nose_y_full))
        except Exception:
            return None

    def verify_faces_in_frame(
        self,
        frame_bgr: np.ndarray,
        faces: list,
        source_width: int,
        source_height: int,
    ) -> list:
        """Verify all faces in one frame using strict pose-nose containment.

        For each face, runs pose on a region around it and checks whether
        the detected pose nose falls INSIDE the face detection bbox.

        Mutates each face in-place:
          face.is_human = True  if pose nose is INSIDE the face bbox
          face.is_human = False otherwise
          face.pose_confidence = visibility score when matched, 0.0 otherwise

        Returns the faces list.
        """
        if not self._available or frame_bgr is None or not faces:
            for f in faces:
                f.is_human = True  # fail open
                f.pose_confidence = 0.0
            return faces

        H, W = frame_bgr.shape[:2]
        if H < 50 or W < 50:
            for f in faces:
                f.is_human = True
                f.pose_confidence = 0.0
            return faces

        for face in faces:
            face_x_pct = float(getattr(face, 'nose_x', getattr(face, 'x_center', 50)))
            face_y_pct = float(getattr(face, 'nose_y', getattr(face, 'y_center', 50)))
            face_w_pct = float(getattr(face, 'width', 10))
            face_h_pct = float(getattr(face, 'height', 12))

            # Face bbox in pixel coordinates
            face_cx_px = face_x_pct / 100 * source_width
            face_cy_px = face_y_pct / 100 * source_height
            face_w_px = face_w_pct / 100 * source_width
            face_h_px = face_h_pct / 100 * source_height
            face_x1 = face_cx_px - face_w_px / 2
            face_y1 = face_cy_px - face_h_px / 2
            face_x2 = face_cx_px + face_w_px / 2
            face_y2 = face_cy_px + face_h_px / 2

            # Run pose on region around this face
            pose_nose_px = self._get_pose_nose_for_region(
                frame_bgr,
                int(face_cx_px), int(face_cy_px),
                int(max(face_w_px, 20)), int(max(face_h_px, 20)),
                source_width, source_height,
            )

            if pose_nose_px is None:
                face.is_human = False
                face.pose_confidence = 0.0
                logger.debug("HumanFaceVerifier: rejected face at (%.1f, %.1f) — no pose",
                             face_x_pct, face_y_pct)
                continue

            nose_x_px, nose_y_px = pose_nose_px

            # STRICT containment: pose nose must be INSIDE the face bbox
            # Tolerance: 20% of face dimension for bbox imprecision
            tol_x = face_w_px * 0.2
            tol_y = face_h_px * 0.2
            contained = (face_x1 - tol_x <= nose_x_px <= face_x2 + tol_x
                         and face_y1 - tol_y <= nose_y_px <= face_y2 + tol_y)

            face.is_human = contained
            if contained:
                face.pose_confidence = 0.9
            else:
                face.pose_confidence = 0.0
                logger.debug("HumanFaceVerifier: rejected face at (%.1f, %.1f) — "
                             "pose nose at (%.1fpx, %.1fpx) not inside bbox "
                             "[%.1f-%.1f, %.1f-%.1f]",
                             face_x_pct, face_y_pct,
                             nose_x_px, nose_y_px,
                             face_x1, face_x2, face_y1, face_y2)

        n_human = sum(1 for f in faces if f.is_human)
        logger.info("HumanFaceVerifier: %d/%d faces verified as human",
                    n_human, len(faces))
        return faces

    def verify_face(
        self,
        full_frame_bgr: np.ndarray,
        face_x_pct: float,
        face_y_pct: float,
        face_w_pct: float,
        face_h_pct: float,
    ) -> Tuple[bool, float]:
        """Legacy per-face API for backward compatibility.

        When USE_STRICT_POSE_MATCHING=false, this is called instead of
        verify_faces_in_frame.
        """
        if not self._available or full_frame_bgr is None:
            return (False, 0.0)

        h, w = full_frame_bgr.shape[:2]
        if h < 50 or w < 50:
            return (False, 0.0)

        try:
            import cv2
            rgb = cv2.cvtColor(full_frame_bgr, cv2.COLOR_BGR2RGB)
            results = self._pose.process(rgb)
            if not results.pose_landmarks:
                return (False, 0.0)

            landmarks = results.pose_landmarks.landmark
            mp_pose = self._mp.solutions.pose.PoseLandmark

            left_shoulder = landmarks[mp_pose.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.RIGHT_SHOULDER.value]
            nose = landmarks[mp_pose.NOSE.value]

            if (left_shoulder.visibility < 0.5
                    and right_shoulder.visibility < 0.5):
                return (False, 0.0)

            shoulder_cx = (left_shoulder.x + right_shoulder.x) / 2 * 100
            shoulder_cy = (left_shoulder.y + right_shoulder.y) / 2 * 100
            nose_x = nose.x * 100
            nose_y = nose.y * 100

            x_distance = abs(shoulder_cx - face_x_pct)
            if x_distance > face_w_pct * 1.5:
                return (False, 0.0)

            if shoulder_cy < face_y_pct:
                return (False, 0.0)

            face_to_pose_distance = abs(nose_x - face_x_pct) + abs(nose_y - face_y_pct)
            if face_to_pose_distance > face_w_pct * 2:
                return (False, 0.0)

            avg_visibility = (left_shoulder.visibility + right_shoulder.visibility) / 2
            return (True, float(avg_visibility))

        except Exception as e:
            logger.debug("HumanFaceVerifier: verify failed: %s", e)
            return (False, 0.0)


_VERIFIER: Optional[HumanFaceVerifier] = None


def get_verifier() -> HumanFaceVerifier:
    """Get the singleton HumanFaceVerifier instance."""
    global _VERIFIER
    if _VERIFIER is None:
        _VERIFIER = HumanFaceVerifier()
    return _VERIFIER


def reset_verifier() -> None:
    """Reset the singleton (for testing)."""
    global _VERIFIER
    _VERIFIER = None
