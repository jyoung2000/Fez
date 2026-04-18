"""Phase 6 — Anime / cartoon face detection.

The standard live-action face detectors (MediaPipe FaceMesh, OpenCV
DNN, YuNet) struggle on anime / cartoon characters. Anime faces have
exaggerated eye proportions, simplified nose / mouth geometry, and
hard outlines instead of skin texture — none of which the
human-trained detectors handle reliably. The existing
``backend/services/face_detector.py`` papers over this with
``ANIME_MODE_DETECTED`` (a flag set when the human-pose verifier
rejects > 50 % of detections) which downgrades the verification
gate, but it doesn't actually find faces the live-action detector
missed in the first place.

This module is the dedicated anime detector. Two production tiers:

- **lbpcascade_animeface tier**: lazily loads the public
  ``lbpcascade_animeface`` Haar cascade
  (``https://github.com/nagadomi/lbpcascade_animeface``). The
  cascade XML ships at ``backend/models/lbpcascade_animeface.xml``
  and is downloaded at build time. Returns ``AnimeFaceDetection``
  results with bbox + confidence.

- **Heuristic tier**: when OpenCV / numpy / the cascade aren't
  available (sandbox path) the detector returns an empty result
  so callers fall through to the existing detection chain. The
  pure-Python helpers (``score_anime_face_density``,
  ``best_face_in_frame``) work without an imaging stack and are
  unit-testable in isolation.

Detection results are converted to ``backend.services.face_detector.FaceInfo``
shape via ``to_face_info`` so anime detections slot into the
existing dense-face stream / face_registry / camera_solver
pipeline without any segmenter changes.

The reframe pipeline calls this detector via:

    if profile.is_animated and CLIPAI_ANIME_FACE_DETECTOR:
        anime_faces = detect_anime_faces(frame_path)
        if anime_faces:
            face_results.extend(to_face_info(d) for d in anime_faces)

so anime detections AUGMENT the live-action stream rather than
replacing it. The face_registry's identity-clustering step then
folds them into the same slot system.

Feature flag: ``CLIPAI_ANIME_FACE_DETECTOR`` env var, default ON as
of Week 2. The dense-face augmentation call site lives in
``face_detector._augment_dense_with_anime``; set the env var to ``0``
to disable the augmentation pass without touching the rest of the
anime pipeline (shot detector, character clustering, saliency anchor
all have their own flags).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────── Feature flag ────────────────────

USE_ANIME_FACE_DETECTOR = os.environ.get(
    "CLIPAI_ANIME_FACE_DETECTOR", "1",
).lower() in ("1", "true", "yes", "on")


# ──────────────────── Tuning ────────────────────

# Default Haar cascade path. Download at:
# https://github.com/nagadomi/lbpcascade_animeface/raw/master/lbpcascade_animeface.xml
DEFAULT_CASCADE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "models", "lbpcascade_animeface.xml",
)

# Minimum face size as a fraction of frame width. Anime cuts often
# have full-frame face close-ups — the lower bound on bbox size
# kills false positives from background characters. Phase 3
# loosened from 0.05 → 0.025 so profile shots, 3/4 turns, and
# medium-distance characters survive (0.05 of 1080p is a 96px
# floor — anything smaller fell off a cliff).
MIN_FACE_FRAC = 0.025

# Max face size as a fraction of frame width. Anything larger is
# almost certainly a foreground prop or a logo, not a face.
MAX_FACE_FRAC = 0.95

# ── Phase 3: env-tunable cascade params ──
# Lower minNeighbors and tighter scaleFactor produce more candidate
# windows on motion frames where the anime cascade was previously
# rejecting profile / 3-quarter views. Defaults are 3 and 1.05; the
# old hardcoded values (5 / 1.1) were tuned for stills, not motion.
ANIME_CASCADE_MIN_NEIGHBORS = int(
    os.environ.get("ANIME_CASCADE_MIN_NEIGHBORS", "3")
)
ANIME_CASCADE_SCALE_FACTOR = float(
    os.environ.get("ANIME_CASCADE_SCALE_FACTOR", "1.05")
)


# ──────────────────── Result dataclass ────────────────────


@dataclass
class AnimeFaceDetection:
    """One anime face detection result.

    Coordinates are in **percent of frame** (0–100), matching the
    rest of the reframe pipeline. ``confidence`` reflects the
    raw cascade score normalized to [0, 1].
    """

    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float
    source: str = "lbpcascade_animeface"


@dataclass
class AnimeDetectionResult:
    """All anime faces detected in a single frame."""

    timestamp: float
    detections: list[AnimeFaceDetection] = field(default_factory=list)
    skipped_reason: str = ""

    @property
    def has_faces(self) -> bool:
        return bool(self.detections)


# ──────────────────── Pure-Python scoring helpers ────────────────────


def score_anime_face_density(
    detections: list[AnimeFaceDetection],
    *,
    min_faces: int = 1,
) -> float:
    """Return a dramatic-moment score in [0, 1] for an anime frame.

    Combines:
      - face count (weighted toward "more is more dramatic", capped at 4)
      - max face confidence (peak detection signal)
      - max face area (closer face = more dramatic)

    Returns 0.0 when ``detections`` is empty so the caller can
    fall through to motion / contrast signals.
    """
    if not detections:
        return 0.0
    n = min(len(detections), 4)
    count_score = n / 4.0
    max_conf = max(d.confidence for d in detections)
    max_area = max(d.width * d.height for d in detections) / 10000.0
    max_area = min(max_area, 1.0)
    score = 0.4 * count_score + 0.35 * max_conf + 0.25 * max_area
    return max(0.0, min(1.0, score))


def best_face_in_frame(
    detections: list[AnimeFaceDetection],
) -> Optional[AnimeFaceDetection]:
    """Pick the most prominent anime face in a frame.

    Ranks by (confidence × area), so a confident close-up beats
    a high-confidence wide background face. Returns None on
    empty input.
    """
    if not detections:
        return None
    return max(
        detections,
        key=lambda d: d.confidence * (d.width * d.height),
    )


def to_face_info(detection: AnimeFaceDetection, *, identity_id: int = -1):
    """Convert an ``AnimeFaceDetection`` to the existing ``FaceInfo`` shape.

    Lazily imports ``backend.services.face_detector`` so this module
    stays importable in a sandbox where the heavy face_detector
    dependencies (cv2 / mediapipe) aren't available.

    The output sets ``is_human=False`` so downstream verifiers know
    the detection came from the anime path and skip the human-pose
    check that would normally reject it.
    """
    from backend.services.face_detector import FaceInfo

    return FaceInfo(
        x_center=float(detection.x_center),
        y_center=float(detection.y_center),
        width=float(detection.width),
        height=float(detection.height),
        # The cascade doesn't emit landmarks, so use bbox center
        # for nose_x / nose_y. The estimate_yaw fall-through in
        # gaze_estimator returns 0 (forward) when nose == center,
        # which is the right default for anime characters where
        # we don't have reliable head-pose data.
        nose_x=float(detection.x_center),
        nose_y=float(detection.y_center),
        confidence=float(detection.confidence),
        identity_id=int(identity_id),
        # Mark non-human so the human-pose verifier doesn't reject
        # this detection.
        is_human=False,
        # No live-action speaker signal for anime — leave 0.
        lip_aperture=0.0,
        is_speaking=False,
        y_bottom=float(detection.y_center + detection.height / 2),
        pose_confidence=0.0,
    )


# ──────────────────── Phase D: YOLOv8-anime-face ONNX backend ────────────────────
#
# CLIPAI_ANIME_FACE_BACKEND=yolo_anime switches the detector from the
# 2014-era lbpcascade Haar cascade to a deepghs YOLOv8-anime-face ONNX
# model. The ONNX runs on CPU via onnxruntime CPUExecutionProvider — the
# GTX 1650 stays reserved for Whisper / Ollama. The flag defaults to
# lbpcascade so production behaviour is unchanged.

_YOLO_ANIME_SESSION = None
YOLO_ANIME_URL = (
    "https://huggingface.co/deepghs/anime_face_detection/resolve/main/"
    "face_detect_v1.4_s/model.onnx"
)


def _get_yolo_anime_session():
    """Lazy-load the YOLOv8-anime-face ONNX session.

    Returns None when onnxruntime isn't installed or the model can't be
    fetched / loaded so callers can fall back to the lbpcascade tier.
    Cached as a module-level singleton.
    """
    global _YOLO_ANIME_SESSION
    if _YOLO_ANIME_SESSION is not None:
        return _YOLO_ANIME_SESSION
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning(
            "onnxruntime not installed — YOLOv8-anime-face unavailable",
        )
        return None

    model_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "yolo_anime_face.onnx",
    )
    if not os.path.isfile(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        try:
            import urllib.request
            logger.info("Downloading YOLOv8-anime-face ONNX model...")
            urllib.request.urlretrieve(YOLO_ANIME_URL, model_path)
        except Exception as e:
            logger.warning("Failed to download YOLO anime face model: %s", e)
            return None

    try:
        _YOLO_ANIME_SESSION = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        logger.info("Loaded YOLOv8-anime-face ONNX session (CPU)")
        return _YOLO_ANIME_SESSION
    except Exception as e:
        logger.warning("YOLO anime session init failed: %s", e)
        return None


def _detect_with_yolo_anime(
    frame_path: str,
    *,
    timestamp: float = 0.0,
    min_confidence: float = 0.4,
) -> AnimeDetectionResult:
    """Run YOLOv8-anime-face ONNX on a single frame.

    Returns an AnimeDetectionResult with one AnimeFaceDetection per
    surviving box (after the model's built-in NMS). Falls through to an
    empty result on any failure so the caller can fall back to the
    lbpcascade tier without a crash.
    """
    sess = _get_yolo_anime_session()
    if sess is None:
        return AnimeDetectionResult(
            timestamp=timestamp, skipped_reason="yolo_anime: no session"
        )
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        return AnimeDetectionResult(
            timestamp=timestamp,
            skipped_reason=f"yolo_anime: missing deps: {exc}",
        )

    img = cv2.imread(frame_path)
    if img is None:
        return AnimeDetectionResult(
            timestamp=timestamp,
            skipped_reason=f"yolo_anime: cannot read frame: {frame_path}",
        )
    h, w = img.shape[:2]
    if w <= 0 or h <= 0:
        return AnimeDetectionResult(
            timestamp=timestamp,
            skipped_reason="yolo_anime: degenerate frame",
        )

    # Standard YOLOv8 preprocessing: resize to 640x640, BGR→RGB, normalize
    # to 0-1, CHW, NCHW. The deepghs models export with NMS included so
    # we get a flat (N, 6) output.
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    blob = img_rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    try:
        outputs = sess.run(None, {sess.get_inputs()[0].name: blob})
    except Exception as exc:
        logger.debug("yolo_anime forward failed: %s", exc)
        return AnimeDetectionResult(
            timestamp=timestamp,
            skipped_reason=f"yolo_anime: forward failed: {exc}",
        )

    preds = outputs[0]
    if hasattr(preds, "ndim") and preds.ndim == 3:
        preds = preds[0]

    detections: list[AnimeFaceDetection] = []
    sx, sy = w / 640.0, h / 640.0
    for det in preds:
        if len(det) < 5:
            continue
        conf = float(det[4])
        if conf < min_confidence:
            continue
        x1, y1, x2, y2 = float(det[0]), float(det[1]), float(det[2]), float(det[3])
        x1, x2 = x1 * sx, x2 * sx
        y1, y2 = y1 * sy, y2 * sy
        cx_pct = ((x1 + x2) / 2) / w * 100
        cy_pct = ((y1 + y2) / 2) / h * 100
        fw_pct = (x2 - x1) / w * 100
        fh_pct = (y2 - y1) / h * 100
        if fw_pct <= 0 or fh_pct <= 0:
            continue
        detections.append(AnimeFaceDetection(
            x_center=float(cx_pct),
            y_center=float(cy_pct),
            width=float(fw_pct),
            height=float(fh_pct),
            confidence=conf,
            source="yolo_anime",
        ))
    return AnimeDetectionResult(timestamp=timestamp, detections=detections)


# ──────────────────── Cascade-backed detector ────────────────────


def detect_anime_faces(
    frame_path: str,
    *,
    timestamp: float = 0.0,
    cascade_path: Optional[str] = None,
    min_face_frac: float = MIN_FACE_FRAC,
    max_face_frac: float = MAX_FACE_FRAC,
    scale_factor: Optional[float] = None,
    min_neighbors: Optional[int] = None,
) -> AnimeDetectionResult:
    """Run the lbpcascade_animeface detector on a single frame.

    Lazily imports OpenCV / numpy so this module loads in a
    sandbox without them. Returns
    ``AnimeDetectionResult(skipped_reason=...)`` when the cascade
    can't be loaded, the frame can't be read, or the imports fail.

    Args:
        frame_path: Filesystem path to the source frame.
        timestamp: Timestamp the frame represents (seconds).
        cascade_path: Optional override for the cascade XML file.
            Defaults to ``DEFAULT_CASCADE_PATH``.
        min_face_frac / max_face_frac: Bbox size gates as fractions
            of frame width.
        scale_factor / min_neighbors: Standard cv2 cascade params.

    Returns:
        ``AnimeDetectionResult`` with ``detections`` and either
        ``has_faces`` or a ``skipped_reason``.
    """
    # Phase D: route through YOLOv8-anime-face when requested. Falls
    # through to the lbpcascade tier on any failure / empty result so
    # the existing path still wins when the new model is unavailable.
    backend = os.environ.get("CLIPAI_ANIME_FACE_BACKEND", "lbpcascade").lower()
    if backend == "yolo_anime":
        yres = _detect_with_yolo_anime(
            frame_path,
            timestamp=timestamp,
            min_confidence=0.4,
        )
        if yres.has_faces:
            return yres
        # belt-and-suspenders: fall through to lbpcascade on empty.

    try:
        import cv2  # noqa: F401
    except ImportError as exc:
        return AnimeDetectionResult(
            timestamp=timestamp,
            skipped_reason=f"missing deps: {exc}",
        )

    cascade_file = cascade_path or DEFAULT_CASCADE_PATH
    if not os.path.isfile(cascade_file):
        return AnimeDetectionResult(
            timestamp=timestamp,
            skipped_reason=f"cascade not found: {cascade_file}",
        )

    cascade = cv2.CascadeClassifier(cascade_file)
    if cascade.empty():
        return AnimeDetectionResult(
            timestamp=timestamp,
            skipped_reason=f"cascade failed to load: {cascade_file}",
        )

    img = cv2.imread(frame_path)
    if img is None:
        return AnimeDetectionResult(
            timestamp=timestamp,
            skipped_reason=f"cannot read frame: {frame_path}",
        )

    h, w = img.shape[:2]
    if w <= 0 or h <= 0:
        return AnimeDetectionResult(
            timestamp=timestamp,
            skipped_reason="degenerate frame",
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    eff_scale = (
        float(scale_factor) if scale_factor is not None
        else float(ANIME_CASCADE_SCALE_FACTOR)
    )
    eff_neighbors = (
        int(min_neighbors) if min_neighbors is not None
        else int(ANIME_CASCADE_MIN_NEIGHBORS)
    )

    min_side = max(int(w * min_face_frac), 8)
    max_side = max(int(w * max_face_frac), min_side + 1)

    # Phase 3: detectMultiScale3 returns per-detection level weights
    # (unbounded reals from the cascade's stage scoring). We normalize
    # to a [0, 1] confidence as ``min(1.0, level_weight / 10.0)`` —
    # strong detections come out around 0.8–0.95 and weaker ones
    # around 0.3–0.5, which lines up with the YuNet confidence range
    # so the registry's confidence-weighted aggregation can compare
    # them apples-to-apples. Falls back to detectMultiScale when the
    # OpenCV build doesn't expose detectMultiScale3 (older bindings).
    rects = []
    weights = []
    try:
        rects_arr, _reject_levels, level_weights = cascade.detectMultiScale3(
            gray,
            scaleFactor=eff_scale,
            minNeighbors=eff_neighbors,
            minSize=(min_side, min_side),
            maxSize=(max_side, max_side),
            outputRejectLevels=True,
        )
        for r, lw in zip(rects_arr, level_weights):
            rects.append(tuple(int(v) for v in r))
            try:
                weights.append(float(lw))
            except Exception:
                weights.append(8.5)  # 0.85 default
    except Exception:
        # Older OpenCV without detectMultiScale3 — fall back, using
        # a conservative constant level weight that maps to 0.85.
        rects_arr = cascade.detectMultiScale(
            gray,
            scaleFactor=eff_scale,
            minNeighbors=eff_neighbors,
            minSize=(min_side, min_side),
            maxSize=(max_side, max_side),
        )
        for r in rects_arr:
            rects.append(tuple(int(v) for v in r))
            weights.append(8.5)

    detections: list[AnimeFaceDetection] = []
    for (x, y, fw, fh), lw in zip(rects, weights):
        cx = (x + fw / 2.0) / w * 100.0
        cy = (y + fh / 2.0) / h * 100.0
        # Normalize the level weight to [0, 1]. The cascade's
        # ``levelWeights`` are unbounded positive reals; divide by 10
        # so a typical "strong" stage score of ~9 lands at 0.9.
        conf = max(0.0, min(1.0, float(lw) / 10.0))
        detections.append(AnimeFaceDetection(
            x_center=cx,
            y_center=cy,
            width=fw / w * 100.0,
            height=fh / h * 100.0,
            confidence=conf,
        ))

    return AnimeDetectionResult(timestamp=timestamp, detections=detections)
