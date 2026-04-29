"""Sparse-sampling content classifier for the SOTA bench's auto-detect pass.

Wraps :func:`backend.services.content_classifier.classify_content` with a
focused fast path: probe duration via ffprobe, sample N evenly-spaced
frames for face detection, sample the first ``audio_window_s`` seconds
of audio for shot-cut detection, skip transcription. Designed to run in
≤ 30 s on a 5-minute clip so it can fire as a pre-pass before the SOTA
bench starts.

When the heavy deps (cv2 / numpy / shot detection) aren't available, the
function logs a warning and returns a low-confidence default profile so
callers fall back gracefully.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


def _probe_duration_seconds(video_path: str) -> float:
    """ffprobe-based duration probe. Returns 0.0 on failure."""
    if not shutil.which("ffprobe"):
        return 0.0
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            timeout=10,
        )
        return float(out.decode().strip() or 0.0)
    except (subprocess.SubprocessError, ValueError, OSError) as exc:
        logger.warning("quick_classify ffprobe failed: %s", exc)
        return 0.0


def _empty_profile(content_type: str = "default", confidence: float = 0.0) -> Any:
    """Build a ContentProfile-like fallback that honors the
    ContentClassifier API contract without needing the heavy module."""
    try:
        from backend.services.content_classifier import ContentProfile
        prof = ContentProfile()
        prof.content_type = content_type
        prof.confidence = confidence
        return prof
    except Exception:
        # Heavy deps missing — return a duck-typed bag.
        class _Profile:
            def __init__(self, content_type, confidence):
                self.content_type = content_type
                self.confidence = confidence
        return _Profile(content_type, confidence)


def quick_classify_video(
    video_path: str,
    *,
    sample_count: int = 10,
    audio_window_s: float = 30.0,
    job_id: str = "",
) -> Any:
    """Fast content classification using sparse sampling.

    Returns a :class:`ContentProfile`-shaped object with at minimum
    ``content_type`` (str) and ``confidence`` (0.0-1.0). On any
    failure (file missing, deps missing, classifier crash) the
    return is a low-confidence default profile so callers can fall
    back to the manifest's existing ``"default"`` content type.

    Args:
        video_path: Path to the MP4 to classify.
        sample_count: How many evenly-spaced frames to sample for
            face detection. Lower = faster but noisier.
        audio_window_s: How many seconds of leading audio to use for
            shot-cut detection.
        job_id: Prefix for log lines.
    """
    if not video_path or not os.path.isfile(video_path):
        logger.warning("[%s] quick_classify: file not found %r", job_id, video_path)
        return _empty_profile()

    duration = _probe_duration_seconds(video_path)
    if duration <= 0:
        logger.warning("[%s] quick_classify: zero/unknown duration", job_id)
        return _empty_profile()

    # Try the full classifier path. Any failure → low-confidence default.
    try:
        from backend.services.content_classifier import classify_content
    except Exception as exc:
        logger.warning(
            "[%s] quick_classify: content_classifier import failed (%s) — "
            "returning default profile",
            job_id, exc,
        )
        return _empty_profile()

    # Sparse face detection over evenly-spaced frames.
    dense_faces: list = []
    shot_cuts: list = []
    face_registry: Any = None
    try:
        dense_faces, face_registry = _sparse_face_detection(
            video_path, duration, sample_count, job_id=job_id,
        )
    except Exception as exc:
        logger.warning(
            "[%s] quick_classify: sparse face detection failed (%s)",
            job_id, exc,
        )

    try:
        shot_cuts = _sparse_shot_detection(
            video_path, min(audio_window_s, duration), job_id=job_id,
        )
    except Exception as exc:
        logger.warning(
            "[%s] quick_classify: sparse shot detection failed (%s)",
            job_id, exc,
        )

    try:
        profile = classify_content(
            shot_cuts=shot_cuts,
            face_registry=face_registry,
            dense_faces=dense_faces,
            scenes=[],
            video_duration=duration,
            metadata=None,
            job_id=job_id,
            transcript_segments=None,
        )
        return profile
    except Exception as exc:
        logger.warning(
            "[%s] quick_classify: classify_content crashed (%s) — "
            "returning default profile",
            job_id, exc,
        )
        return _empty_profile()


def _sparse_face_detection(
    video_path: str, duration: float, sample_count: int, *, job_id: str = "",
) -> tuple[list, Any]:
    """Run face detection on N evenly-spaced frames.

    Returns ``(dense_faces, face_registry)``. Both empty when the
    detector deps are unavailable.
    """
    try:
        # Lazy heavy imports.
        from backend.services.face_detector import FaceDetector
        from backend.services.face_registry import build_face_registry
    except Exception as exc:
        logger.info(
            "[%s] quick_classify: heavy face deps missing (%s)", job_id, exc,
        )
        return [], None

    detector = FaceDetector()
    timestamps = [
        duration * (i + 0.5) / max(sample_count, 1) for i in range(sample_count)
    ]
    dense_faces: list = []
    for t in timestamps:
        try:
            frame_faces = detector.detect_at_timestamp(video_path, t)
            if frame_faces is not None:
                dense_faces.append(frame_faces)
        except Exception as exc:
            logger.debug("face detect at t=%.2fs failed: %s", t, exc)
            continue
    if not dense_faces:
        return [], None
    try:
        registry = build_face_registry(dense_faces)
    except Exception as exc:
        logger.debug("registry build failed: %s", exc)
        registry = None
    return dense_faces, registry


def _sparse_shot_detection(
    video_path: str, window_s: float, *, job_id: str = "",
) -> list[float]:
    """Sample-mode shot-cut detection on the leading window.

    Returns a list of cut timestamps (seconds). Empty when the heavy
    shot detector isn't available.
    """
    try:
        from backend.services.shot_detector import detect_shot_cuts
    except Exception as exc:
        logger.info(
            "[%s] quick_classify: shot detector missing (%s)", job_id, exc,
        )
        return []
    try:
        cuts = detect_shot_cuts(video_path, max_duration_s=window_s)
        return [float(c) for c in (cuts or [])]
    except Exception as exc:
        logger.debug("shot detection failed: %s", exc)
        return []
