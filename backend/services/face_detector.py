"""Lightweight face detection for subject tracking.

Uses OpenCV's DNN face detector (always available) with optional MediaPipe
upgrade. Runs on CPU — no GPU competition with Ollama/Whisper.
Processes 60 frames in ~1-2 seconds.
"""
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Set to True by _verify_faces_in_results when the human-verifier
# rejection rate exceeds 50% (anime/cartoon signature). The content
# classifier reads this to force is_animated=True on the ContentProfile
# so downstream routing can pick ANIMATION_DIALOGUE. Reset to False at
# the start of each detect_faces_batch call.
ANIME_MODE_DETECTED = False


@dataclass
class FaceInfo:
    """Face detection result for a single face."""
    x_center: float      # Horizontal center of face bbox, 0-100
    y_center: float      # Vertical center of face bbox, 0-100
    width: float          # Face bbox width as % of frame
    height: float         # Face bbox height as % of frame
    nose_x: float         # Best estimate of face center x, 0-100
    nose_y: float         # Best estimate of face center y, 0-100
    confidence: float     # Detection confidence, 0-1
    lip_aperture: float = 0.0  # Mouth openness ratio (0=closed, 1=wide open)
    identity_embedding: list = None  # 128-d (SFace) or 512-d (ArcFace) face embedding for re-identification
    identity_id: int = -1       # Assigned face slot from registry (-1 = unassigned)
    is_speaking: bool = False   # Set by active speaker detection
    y_bottom: float = 0.0      # Bottom of face bbox as % of frame (for vertical positioning)
    is_human: bool = True       # Set by HumanFaceVerifier (default True = fail-open)
    pose_confidence: float = 0.0  # Pose verification confidence (0 if not verified)

    def __post_init__(self):
        """Ensure all numeric fields are native Python types, not numpy."""
        self.x_center = float(self.x_center)
        self.y_center = float(self.y_center)
        self.width = float(self.width)
        self.height = float(self.height)
        self.nose_x = float(self.nose_x)
        self.nose_y = float(self.nose_y)
        self.confidence = float(self.confidence)
        self.lip_aperture = float(self.lip_aperture)
        self.identity_id = int(self.identity_id)
        self.is_speaking = bool(self.is_speaking)
        self.y_bottom = float(self.y_bottom)


@dataclass
class FrameFaces:
    """All faces detected in a single frame."""
    timestamp: float
    frame_path: str
    faces: list[FaceInfo] = field(default_factory=list)
    primary_face_idx: int = -1  # Index of largest/most-prominent face


def _extract_face_embeddings(frame_img, faces_info: list, detector) -> list:
    """Extract 128-d identity embeddings using OpenCV SFace recognizer.

    Runs on CPU. ~5ms per face. No GPU competition with Whisper/Ollama.
    The embedding enables cross-frame re-identification: same person across
    different timestamps gets matched even if they move positions.
    """
    import cv2
    import numpy as np

    if not faces_info or detector is None:
        return faces_info

    recognizer = None
    sface_paths = [
        os.path.join(os.path.dirname(cv2.__file__), "data", "face_recognition_sface_2021dec.onnx"),
        "/usr/local/share/opencv4/face_recognition_sface_2021dec.onnx",
        os.path.join(os.path.dirname(__file__), "..", "models", "face_recognition_sface_2021dec.onnx"),
    ]
    sface_path = next((p for p in sface_paths if os.path.exists(p)), None)

    if sface_path is None:
        # Download one-time (~2MB)
        import urllib.request
        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        os.makedirs(model_dir, exist_ok=True)
        sface_path = os.path.join(model_dir, "face_recognition_sface_2021dec.onnx")
        if not os.path.exists(sface_path):
            url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
            logger.info("Downloading SFace recognition model...")
            try:
                urllib.request.urlretrieve(url, sface_path)
            except Exception as e:
                logger.warning("Failed to download SFace model: %s", e)
                return faces_info

    try:
        recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
    except Exception as e:
        logger.warning("SFace recognizer unavailable: %s — skipping embeddings", e)
        return faces_info

    h, w = frame_img.shape[:2]
    for face in faces_info:
        try:
            # Convert percentage coords back to pixels for SFace
            fx = int((face.x_center - face.width / 2) * w / 100)
            fy = int((face.y_center - face.height / 2) * h / 100)
            fw = int(face.width * w / 100)
            fh = int(face.height * h / 100)

            # Build the aligned face for SFace (it expects the YuNet detection format)
            face_array = [fx, fy, fw, fh,
                          face.nose_x * w / 100, face.nose_y * h / 100,
                          0, 0, 0, 0, 0, 0, 0, 0, face.confidence]
            det = np.array([face_array], dtype=np.float32)

            aligned = recognizer.alignCrop(frame_img, det[0])
            embedding = recognizer.feature(aligned)
            face.identity_embedding = embedding.flatten().tolist()
            face.y_bottom = round((face.y_center + face.height / 2), 1)
        except Exception:
            pass  # Non-critical — face still usable without embedding

    return faces_info


# ── ArcFace (InsightFace buffalo_s) embedding backend ──
# Module-level singleton so we don't reload the 16 MB ONNX per batch.
# Routed via CLIPAI_FACE_EMBEDDING=arcface; defaults off.
# CPU-only via onnxruntime CPUExecutionProvider — must NOT compete with
# Whisper / Ollama on the GTX 1650 (4 GB VRAM is reserved for them).
_ARCFACE_APP = None


def _get_arcface_app():
    """Lazy-load InsightFace ArcFace (buffalo_s pack), CPU-only.

    Returns None if insightface / onnxruntime are unavailable so callers
    can fall back to SFace. The model is cached after first load.
    """
    global _ARCFACE_APP
    if _ARCFACE_APP is not None:
        return _ARCFACE_APP
    try:
        from insightface.app import FaceAnalysis
    except ImportError as e:
        logger.warning("insightface not installed — ArcFace unavailable: %s", e)
        return None
    except Exception as e:
        logger.warning("insightface import failed: %s", e)
        return None

    model_root = os.path.join(os.path.dirname(__file__), "..", "models", "insightface")
    os.makedirs(model_root, exist_ok=True)
    try:
        app = FaceAnalysis(
            name="buffalo_s",
            root=model_root,
            # CPU ONLY — GPU is reserved for Whisper/Ollama. Do not change.
            providers=["CPUExecutionProvider"],
            allowed_modules=["recognition"],  # skip det/landmark — we use YuNet
        )
        app.prepare(ctx_id=-1, det_size=(320, 320))
        _ARCFACE_APP = app
        logger.info("Loaded InsightFace buffalo_s (ArcFace, 512-d, CPU)")
        return app
    except Exception as e:
        logger.warning("Failed to init InsightFace buffalo_s: %s", e)
        return None


def _extract_face_embeddings_arcface(frame_img, faces_info: list) -> list:
    """Extract 512-d ArcFace embeddings using InsightFace buffalo_s.

    We reuse the YuNet bbox/landmarks already attached to each FaceInfo —
    InsightFace's recognition module needs an aligned face crop, and we
    build it from the bbox + nose landmark to avoid running its own
    detector twice.
    """
    import numpy as np

    app = _get_arcface_app()
    if app is None or not faces_info:
        return faces_info

    try:
        from insightface.app.common import Face as IFace
    except ImportError:
        return faces_info

    h, w = frame_img.shape[:2]
    rec_model = None
    # Pull the recognition model directly from the analysis app.
    try:
        for m in app.models.values():
            if hasattr(m, "get_feat") or hasattr(m, "get"):
                rec_model = m
                break
    except Exception:
        rec_model = None
    if rec_model is None:
        logger.warning("ArcFace recognition model not found in buffalo_s pack")
        return faces_info

    for face in faces_info:
        try:
            # Convert our 0-100 percentage bbox back to absolute pixel coords
            fx1 = max(0, int((face.x_center - face.width / 2) * w / 100))
            fy1 = max(0, int((face.y_center - face.height / 2) * h / 100))
            fx2 = min(w, int((face.x_center + face.width / 2) * w / 100))
            fy2 = min(h, int((face.y_center + face.height / 2) * h / 100))
            if fx2 - fx1 < 12 or fy2 - fy1 < 12:
                continue  # too small to embed reliably

            # Build the 5-point landmark array InsightFace expects:
            # (right-eye, left-eye, nose, right-mouth, left-mouth) in pixels.
            # YuNet only stores the nose tip on FaceInfo, so synthesize the
            # other four from the bbox geometry — buffalo_s's alignCrop is
            # robust to the approximation.
            nose_x = face.nose_x * w / 100
            nose_y = face.nose_y * h / 100
            eye_y = fy1 + (fy2 - fy1) * 0.35
            mouth_y = fy1 + (fy2 - fy1) * 0.75
            kps = np.array([
                [fx1 + (fx2 - fx1) * 0.35, eye_y],
                [fx1 + (fx2 - fx1) * 0.65, eye_y],
                [nose_x, nose_y],
                [fx1 + (fx2 - fx1) * 0.38, mouth_y],
                [fx1 + (fx2 - fx1) * 0.62, mouth_y],
            ], dtype=np.float32)

            iface = IFace(
                bbox=np.array([fx1, fy1, fx2, fy2], dtype=np.float32),
                kps=kps,
            )
            # InsightFace recognition models expose `get(img, face)` which
            # returns a 512-d L2-normalized feature vector.
            feat = rec_model.get(frame_img, iface)
            if feat is None:
                continue
            face.identity_embedding = np.asarray(feat, dtype=np.float32).flatten().tolist()
            face.y_bottom = round((face.y_center + face.height / 2), 1)
        except Exception as e:
            logger.debug("ArcFace embed failed for one face: %s", e)

    return faces_info


def _detect_with_opencv_dnn(frame_paths, min_confidence, extract_embeddings=True):
    """Detect faces using OpenCV's built-in DNN face detector.

    Tries YuNet DNN detector first (much more accurate, gives nose/mouth landmarks),
    falls back to Haar cascade if YuNet model is unavailable.
    """
    import cv2

    results = []

    # Try YuNet DNN face detector first (much more accurate than Haar cascade)
    detector = None
    try:
        yunet_paths = [
            os.path.join(os.path.dirname(cv2.__file__), "data",
                         "face_detection_yunet_2023mar.onnx"),
            "/usr/local/share/opencv4/face_detection_yunet_2023mar.onnx",
            "/usr/share/opencv4/face_detection_yunet_2023mar.onnx",
            os.path.join(os.path.dirname(__file__), "..", "models",
                         "face_detection_yunet_2023mar.onnx"),
        ]
        yunet_path = next((p for p in yunet_paths if os.path.exists(p)), None)

        if yunet_path is None:
            # Download the model (one-time, ~350KB)
            import urllib.request
            model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
            os.makedirs(model_dir, exist_ok=True)
            yunet_path = os.path.join(model_dir, "face_detection_yunet_2023mar.onnx")
            if not os.path.exists(yunet_path):
                url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
                logger.info("Downloading YuNet face detection model...")
                urllib.request.urlretrieve(url, yunet_path)
                logger.info("YuNet model downloaded to %s", yunet_path)

        if yunet_path and os.path.exists(yunet_path):
            detector = cv2.FaceDetectorYN.create(
                yunet_path,
                "",
                (320, 320),  # Will be resized per-frame
                min_confidence,
                0.3,  # NMS threshold
            )
            logger.info("Using OpenCV YuNet DNN face detector")
    except (cv2.error, AttributeError, Exception) as e:
        logger.debug("YuNet not available: %s", e)

    use_haar = detector is None

    if use_haar:
        # Haar cascade fallback — always available in OpenCV
        cascade_paths = [
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"),
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml"),
        ]
        cascade_path = next((p for p in cascade_paths if os.path.exists(p)), None)
        if not cascade_path:
            logger.warning("No Haar cascade file found — face detection disabled")
            return None
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            logger.warning("Failed to load Haar cascade — face detection disabled")
            return None
        logger.info("Using OpenCV Haar cascade face detector")

    for timestamp, path in frame_paths:
        img = cv2.imread(str(path))
        if img is None:
            results.append(FrameFaces(timestamp=timestamp, frame_path=str(path)))
            continue

        h, w = img.shape[:2]
        faces: list[FaceInfo] = []

        if not use_haar and detector is not None:
            # YuNet DNN detection — much more accurate, gives landmarks
            detector.setInputSize((w, h))
            _, det_result = detector.detect(img)
            if det_result is not None:
                for face in det_result:
                    # YuNet returns: [x, y, w, h, right_eye_x, right_eye_y,
                    #   left_eye_x, left_eye_y, nose_x, nose_y,
                    #   right_mouth_x, right_mouth_y, left_mouth_x, left_mouth_y, score]
                    fx, fy, fw_px, fh_px = face[0], face[1], face[2], face[3]
                    nose_x_px, nose_y_px = face[8], face[9]
                    score = face[14]

                    cx = (fx + fw_px / 2) / w * 100
                    cy = (fy + fh_px / 2) / h * 100
                    fw_pct = fw_px / w * 100
                    fh_pct = fh_px / h * 100
                    nose_x_pct = nose_x_px / w * 100
                    nose_y_pct = nose_y_px / h * 100

                    # Compute lip aperture from mouth landmarks
                    right_mouth_y = face[11]
                    left_mouth_y = face[13]
                    mouth_center_y = (right_mouth_y + left_mouth_y) / 2
                    lip_aperture = abs(mouth_center_y - nose_y_px) / max(fh_px, 1) * 0.3

                    faces.append(FaceInfo(
                        x_center=float(round(cx, 1)),
                        y_center=float(round(cy, 1)),
                        width=float(round(fw_pct, 1)),
                        height=float(round(fh_pct, 1)),
                        nose_x=float(round(nose_x_pct, 1)),
                        nose_y=float(round(nose_y_pct, 1)),
                        confidence=float(round(float(score), 3)),
                        lip_aperture=float(round(lip_aperture, 3)),
                    ))

        elif use_haar:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detections = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(30, 30),
            )
            # NMS: remove overlapping detections (IoU > 0.3) to prevent
            # merged bounding boxes spanning multiple faces
            if len(detections) > 1:
                dets = sorted(detections.tolist() if hasattr(detections, 'tolist') else list(detections),
                              key=lambda f: f[2] * f[3], reverse=True)
                kept = [dets[0]]
                for det in dets[1:]:
                    x1, y1, w1, h1 = det
                    overlaps = False
                    for kx, ky, kw, kh in kept:
                        ix1 = max(x1, kx); iy1 = max(y1, ky)
                        ix2 = min(x1+w1, kx+kw); iy2 = min(y1+h1, ky+kh)
                        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                        union = w1*h1 + kw*kh - inter
                        if union > 0 and inter / union > 0.3:
                            overlaps = True
                            break
                    if not overlaps:
                        kept.append(det)
                detections = kept
            for (x, y, fw, fh) in detections:
                cx = (x + fw / 2) / w * 100
                cy = (y + fh / 2) / h * 100
                fw_pct = fw / w * 100
                fh_pct = fh / h * 100
                # Haar bboxes extend slightly more into background than toward
                # face center. Apply a small inward correction (max 3%) instead
                # of an aggressive heuristic that can overcorrect.
                edge_dist = abs(cx - 50)
                if edge_dist > 15:
                    pull = min(3.0, fw_pct * 0.15)
                    cx = cx + pull if cx < 50 else cx - pull
                faces.append(FaceInfo(
                    x_center=float(round(cx, 1)), y_center=float(round(cy, 1)),
                    width=float(round(fw_pct, 1)), height=float(round(fh_pct, 1)),
                    nose_x=float(round(cx, 1)), nose_y=float(round(cy, 1)),
                    confidence=0.8,  # Haar doesn't provide confidence
                ))

        # ── Reject merged detections ──
        # A single face wider than 18% of frame centered between 30-70%
        # is actually TWO faces merged into one box by Haar cascade.
        if len(faces) == 1 and faces[0].width > 18.0:
            if 30 < faces[0].x_center < 70:
                logger.debug(
                    "Frame %.1fs: rejecting merged face detection "
                    "(width=%.1f%%, center=%.1f%% — likely two speakers)",
                    timestamp, faces[0].width, faces[0].x_center,
                )
                faces = []

        # Extract identity embeddings (YuNet only — Haar doesn't provide landmarks)
        if extract_embeddings and not use_haar and detector is not None and faces and img is not None:
            backend = os.environ.get("CLIPAI_FACE_EMBEDDING", "sface").lower()
            if backend == "arcface":
                faces = _extract_face_embeddings_arcface(img, faces)
                # Graceful degrade: if ArcFace returned nothing, fall through to SFace
                if not any(f.identity_embedding is not None for f in faces):
                    logger.debug(
                        "ArcFace returned no embeddings — falling back to SFace for this frame"
                    )
                    faces = _extract_face_embeddings(img, faces, detector)
            else:
                faces = _extract_face_embeddings(img, faces, detector)

        primary = -1
        if faces:
            primary = max(range(len(faces)), key=lambda i: faces[i].width * faces[i].height)

        results.append(FrameFaces(
            timestamp=timestamp, frame_path=str(path),
            faces=faces, primary_face_idx=primary,
        ))

    # Explicitly release OpenCV resources
    if use_haar:
        del cascade
    elif detector is not None:
        del detector
    import gc
    gc.collect()

    return results


def _detect_with_facemesh(frame_paths, min_confidence, progress_callback=None):
    """Detect faces using MediaPipe FaceMesh — provides lip landmarks for active speaker detection.

    FaceMesh gives 468 landmarks per face including lip points.
    Lip Aperture Ratio = inner_lip_distance / face_height.
    When LAR > ~0.03, the person's mouth is open (likely speaking).
    Runs on CPU, ~3-5s for 60 frames.
    """
    import mediapipe as mp
    import cv2
    import time as _time

    face_mesh_module = None
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
        face_mesh_module = mp.solutions.face_mesh
    if face_mesh_module is None:
        try:
            from mediapipe.python.solutions import face_mesh as fm_mod
            face_mesh_module = fm_mod
        except (ImportError, AttributeError):
            pass
    if face_mesh_module is None:
        logger.info("MediaPipe FaceMesh not available — falling back to FaceDetection")
        return None

    UPPER_LIP_INNER = 82
    LOWER_LIP_INNER = 87
    NOSE_TIP = 1

    results = []
    total_frames = len(frame_paths)
    _last_progress_emit = _time.monotonic()

    with face_mesh_module.FaceMesh(
        static_image_mode=True,
        max_num_faces=10,  # Support up to 10 speakers (was 4)
        refine_landmarks=True,
        min_detection_confidence=min_confidence,
    ) as mesh:
        for frame_idx, (timestamp, path) in enumerate(frame_paths):
            img = cv2.imread(str(path))
            if img is None:
                results.append(FrameFaces(timestamp=timestamp, frame_path=str(path)))
                continue

            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            det_result = mesh.process(rgb)

            faces: list[FaceInfo] = []
            if det_result.multi_face_landmarks:
                for face_landmarks in det_result.multi_face_landmarks:
                    lm = face_landmarks.landmark

                    xs_px = [l.x * w for l in lm]
                    ys_px = [l.y * h for l in lm]
                    x_min, x_max = min(xs_px), max(xs_px)
                    y_min, y_max = min(ys_px), max(ys_px)
                    face_w = x_max - x_min
                    face_h = y_max - y_min

                    cx = ((x_min + x_max) / 2) / w * 100
                    cy = ((y_min + y_max) / 2) / h * 100
                    fw_pct = face_w / w * 100
                    fh_pct = face_h / h * 100

                    nose_x = lm[NOSE_TIP].x * 100
                    nose_y = lm[NOSE_TIP].y * 100

                    # Lip Aperture Ratio (LAR)
                    upper_lip_y = lm[UPPER_LIP_INNER].y * h
                    lower_lip_y = lm[LOWER_LIP_INNER].y * h
                    lip_distance = abs(lower_lip_y - upper_lip_y)
                    lip_aperture = lip_distance / max(face_h, 1)

                    faces.append(FaceInfo(
                        x_center=round(cx, 1),
                        y_center=round(cy, 1),
                        width=round(fw_pct, 1),
                        height=round(fh_pct, 1),
                        nose_x=round(nose_x, 1),
                        nose_y=round(nose_y, 1),
                        confidence=0.9,
                        lip_aperture=round(lip_aperture, 3),
                    ))

            # Reject merged detections
            if len(faces) == 1 and faces[0].width > 18.0:
                if 30 < faces[0].x_center < 70:
                    logger.debug("FaceMesh: rejecting merged detection w=%.1f%% c=%.1f%%",
                                 faces[0].width, faces[0].x_center)
                    faces = []

            primary = -1
            if faces:
                primary = max(range(len(faces)), key=lambda i: faces[i].width * faces[i].height)

            results.append(FrameFaces(
                timestamp=timestamp, frame_path=str(path),
                faces=faces, primary_face_idx=primary,
            ))

            # Emit per-frame progress every 50 frames or every 10 seconds
            if progress_callback and total_frames > 60:
                now = _time.monotonic()
                if (frame_idx + 1) % 50 == 0 or now - _last_progress_emit >= 10:
                    faces_so_far = sum(1 for r in results if r.faces)
                    progress_callback(
                        "facemesh_progress", frame_idx + 1, total_frames,
                        faces_so_far,
                    )
                    _last_progress_emit = now

    return results


def _detect_with_mediapipe(frame_paths, min_confidence):
    """Detect faces using MediaPipe — more accurate than Haar cascade.

    Handles both the legacy solutions API and newer API variants.
    """
    import mediapipe as mp
    import cv2

    # Try to find the right API
    face_detection_module = None
    FaceKeyPoint = None

    # Method 1: Legacy solutions API (mediapipe < 0.10.8)
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_detection'):
        face_detection_module = mp.solutions.face_detection
        FaceKeyPoint = face_detection_module.FaceKeyPoint

    # Method 2: Direct import (some versions)
    if face_detection_module is None:
        try:
            from mediapipe.python.solutions import face_detection as fd_mod
            face_detection_module = fd_mod
            FaceKeyPoint = fd_mod.FaceKeyPoint
        except (ImportError, AttributeError):
            pass

    if face_detection_module is None:
        logger.warning("MediaPipe face_detection API not available in this version")
        return None

    results = []
    get_key_point = getattr(face_detection_module, 'get_key_point', None)

    with face_detection_module.FaceDetection(
        model_selection=1,
        min_detection_confidence=min_confidence,
    ) as detector:
        for timestamp, path in frame_paths:
            img = cv2.imread(str(path))
            if img is None:
                results.append(FrameFaces(timestamp=timestamp, frame_path=str(path)))
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            det_result = detector.process(rgb)

            faces: list[FaceInfo] = []
            if det_result.detections:
                for det in det_result.detections:
                    bbox = det.location_data.relative_bounding_box
                    cx = (bbox.xmin + bbox.width / 2) * 100
                    cy = (bbox.ymin + bbox.height / 2) * 100
                    fw = bbox.width * 100
                    fh = bbox.height * 100

                    # Try to get nose tip for more accurate face center
                    nose_x, nose_y = cx, cy
                    if get_key_point and FaceKeyPoint:
                        try:
                            nose = get_key_point(det, FaceKeyPoint.NOSE_TIP)
                            if nose:
                                nose_x = nose.x * 100
                                nose_y = nose.y * 100
                        except Exception:
                            pass

                    conf = det.score[0] if det.score else 0.0
                    faces.append(FaceInfo(
                        x_center=round(cx, 1), y_center=round(cy, 1),
                        width=round(fw, 1), height=round(fh, 1),
                        nose_x=round(nose_x, 1), nose_y=round(nose_y, 1),
                        confidence=round(conf, 3),
                    ))

            primary = -1
            if faces:
                primary = max(range(len(faces)), key=lambda i: faces[i].width * faces[i].height)

            results.append(FrameFaces(
                timestamp=timestamp, frame_path=str(path),
                faces=faces, primary_face_idx=primary,
            ))

    return results


def _merge_detections(
    primary: list,  # list[FrameFaces] — FaceMesh results (have lip landmarks)
    secondary: list,  # list[FrameFaces] — YuNet results (may have more faces)
) -> list:
    """Merge face detections from two detectors.

    Keeps all primary (FaceMesh) faces — they have lip landmarks for active
    speaker detection. Adds secondary (YuNet) faces that don't overlap with
    any primary face (IoU by x_center distance > 10% of frame width).
    """
    merged = []
    for pri, sec in zip(primary, secondary):
        faces = list(pri.faces)  # Start with FaceMesh faces
        primary_xs = [f.x_center for f in faces]

        # Add non-overlapping secondary faces
        for sf in sec.faces:
            overlaps = any(abs(sf.x_center - px) < 10 for px in primary_xs)
            if not overlaps:
                faces.append(sf)
                primary_xs.append(sf.x_center)

        # Recompute primary face index
        primary_idx = -1
        if faces:
            primary_idx = max(range(len(faces)), key=lambda i: faces[i].width * faces[i].height)

        merged.append(FrameFaces(
            timestamp=pri.timestamp,
            frame_path=pri.frame_path,
            faces=faces,
            primary_face_idx=primary_idx,
        ))
    return merged


def _verify_faces_in_results(results: list, frame_paths: list) -> list:
    """Run human face verification on all detected faces.

    When USE_STRICT_POSE_MATCHING=true (default), uses per-frame strict
    containment: the pose nose must be INSIDE the face bbox.

    When USE_STRICT_POSE_MATCHING=false, uses the legacy per-face proximity
    check for backward compatibility.

    When USE_HUMAN_VERIFICATION=false or the verifier is unavailable,
    all faces are left as is_human=True (fail-open).
    """
    import os
    use_verification = os.environ.get(
        "USE_HUMAN_VERIFICATION", "true"
    ).lower() in ("true", "1", "yes")

    if not use_verification:
        logger.info("Human face verification disabled (USE_HUMAN_VERIFICATION=false)")
        return results

    use_strict = os.environ.get(
        "USE_STRICT_POSE_MATCHING", "true"
    ).lower() in ("true", "1", "yes")

    try:
        from backend.services.human_face_verifier import get_verifier
        verifier = get_verifier()
    except Exception as e:
        logger.warning("Human face verifier import failed: %s", e)
        return results

    if not verifier.available:
        logger.info("Human face verifier unavailable — all faces accepted (fail-open)")
        return results

    import cv2

    # Build a path lookup from frame_paths
    path_lookup = {round(t, 3): p for t, p in frame_paths}

    verified = 0
    non_human = 0
    for fr in results:
        if not fr.faces:
            continue

        # Try to load the frame image
        frame_path = path_lookup.get(round(fr.timestamp, 3), fr.frame_path)
        frame_bgr = cv2.imread(str(frame_path)) if frame_path else None
        if frame_bgr is None:
            continue

        if use_strict:
            # Strict mode: per-frame containment check
            h, w = frame_bgr.shape[:2]
            verifier.verify_faces_in_frame(frame_bgr, fr.faces, w, h)
            for face in fr.faces:
                verified += 1
                if not face.is_human:
                    non_human += 1
        else:
            # Legacy mode: per-face proximity check
            for face in fr.faces:
                is_human, pose_conf = verifier.verify_face(
                    frame_bgr,
                    face.nose_x,
                    face.nose_y,
                    face.width,
                    face.height,
                )
                face.is_human = is_human
                face.pose_confidence = pose_conf
                verified += 1
                if not is_human:
                    non_human += 1

    logger.info(
        "Human face verification (%s): %d faces checked, %d non-human detected",
        "strict" if use_strict else "legacy", verified, non_human,
    )

    # ── Auto-detect animated content ──
    # MediaPipe Pose is a human-proportions model; anime / cartoon / stylized
    # faces fail it ~70-80% of the time and get penalized to weight=0.35 in
    # the face registry, which drops anime speakers below the saliency ceiling
    # and causes the crop to drift onto background motion.
    #
    # Heuristic: if >50% of faces were rejected AND we have a reliable
    # sample size, this is almost certainly animated content. Reset
    # every face back to is_human=True so the registry treats them all
    # at full weight. Gated by the ALLOW_ANIMATED_AUTO_DETECT env var
    # (default on) for rollback.
    #
    # Fix 6: require at least MIN_FACES_FOR_ANIME_MODE verified faces
    # before triggering. The sparse 60-frame pass was flipping AnimeMode
    # on after seeing ~30 faces with 61% rejection — a statistical
    # fluke driven by YuNet false positives, not a real anime signal.
    # The dense 1212-frame pass then verified thousands of faces
    # normally, but by then sparse-pass AnimeMode had already flipped
    # is_human back to True for every face, polluting the registry
    # weighting downstream. Gating on minimum sample size defers the
    # decision to the dense pass where the statistics are reliable.
    import os as _os
    _allow_anim_auto = _os.environ.get(
        "ALLOW_ANIMATED_AUTO_DETECT", "true",
    ).lower() in ("true", "1", "yes")
    MIN_FACES_FOR_ANIME_MODE = 100
    global ANIME_MODE_DETECTED
    rejection_rate = (non_human / verified) if verified > 0 else 0.0
    if not _allow_anim_auto:
        # Flag left as-is for rollback. Don't clobber.
        pass
    elif verified < MIN_FACES_FOR_ANIME_MODE:
        logger.info(
            "[AnimeMode] deferred: verified=%d < %d minimum for reliable "
            "decision (rejection rate %.0f%%). AnimeMode decision will "
            "be taken on the dense-pass call.",
            verified, MIN_FACES_FOR_ANIME_MODE, rejection_rate * 100,
        )
        # Important: do NOT reset ANIME_MODE_DETECTED here — a previous
        # (dense) call may already have set it to True, and a later
        # sparse call should not unflip the reliable decision.
    elif rejection_rate > 0.5:
        _restored = 0
        for fr in results:
            for face in fr.faces:
                if not getattr(face, "is_human", True):
                    face.is_human = True
                    face.pose_confidence = 1.0
                    _restored += 1
        ANIME_MODE_DETECTED = True
        logger.info(
            "[AnimeMode] skipping human verifier, kept %d faces at weight=1.0 "
            "(rejection rate %.0f%% > 50%% — presumed animated content)",
            verified, rejection_rate * 100,
        )
    else:
        ANIME_MODE_DETECTED = False
        logger.info(
            "[AnimeMode] not triggered: verified=%d, rejection rate %.0f%% "
            "(<= 50%%)",
            verified, rejection_rate * 100,
        )

    return results


def detect_faces_batch_two_pass(
    frame_paths: list[tuple[float, str]],
    speech_intervals: list[tuple[float, float]],
    min_confidence: float = 0.3,
    progress_callback=None,
) -> list[FrameFaces]:
    """Speech-aware fast path for dense face detection.

    Splits ``frame_paths`` by whether the timestamp falls inside any of
    ``speech_intervals``:

      * **Inside speech** → full FaceMesh+YuNet pass (lip landmarks
        needed for active-speaker detection).
      * **Outside speech** → YuNet-only pass (much faster; no lip
        landmarks but they would not be used anyway).

    The resulting ``FrameFaces`` are merged back into the original
    timestamp order. When ``speech_intervals`` is empty or covers
    everything, this falls through to the regular ``detect_faces_batch``
    so existing callers see no behavior change.

    Typical speedup on dialogue / talking-head content: ~2× on the
    dense-face stage because most non-speech frames skip FaceMesh.
    """
    if not speech_intervals:
        return detect_faces_batch(frame_paths, min_confidence, progress_callback)

    # Build a sorted list of intervals and a quick predicate.
    intervals = sorted(
        (float(s), float(e)) for s, e in speech_intervals if e > s
    )
    if not intervals:
        return detect_faces_batch(frame_paths, min_confidence, progress_callback)

    def _in_speech(t: float) -> bool:
        # Linear scan is fine for typical few-hundred intervals; bisect
        # would be log(n) but the overhead isn't worth it here.
        for s, e in intervals:
            if t < s:
                return False
            if t <= e:
                return True
        return False

    indexed = list(enumerate(frame_paths))
    speech_subset = [(i, p) for i, p in indexed if _in_speech(p[0])]
    other_subset = [(i, p) for i, p in indexed if not _in_speech(p[0])]

    if not other_subset:
        # Everything's in speech — same cost as the original.
        return detect_faces_batch(frame_paths, min_confidence, progress_callback)
    if not speech_subset:
        # Everything's outside speech — YuNet-only on the whole list.
        try:
            results = _detect_with_opencv_dnn(frame_paths, min_confidence)
        except Exception as e:
            logger.warning("Two-pass YuNet-only fallback failed (%s); using full path", e)
            return detect_faces_batch(frame_paths, min_confidence, progress_callback)
        return list(results) if results is not None else []

    # Run the slow path only on speech-active frames; YuNet on the rest.
    speech_paths = [p for _, p in speech_subset]
    other_paths = [p for _, p in other_subset]

    try:
        speech_results = detect_faces_batch(speech_paths, min_confidence, progress_callback)
    except Exception as e:
        logger.warning(
            "Two-pass speech FaceMesh+YuNet failed (%s); falling back to full path",
            e,
        )
        return detect_faces_batch(frame_paths, min_confidence, progress_callback)

    other_results = []
    try:
        yunet = _detect_with_opencv_dnn(other_paths, min_confidence)
        if yunet is not None:
            other_results = list(yunet)
    except Exception as e:
        logger.info(
            "Two-pass YuNet (non-speech) failed (%s); using empty FrameFaces for %d frames",
            e, len(other_paths),
        )
    if not other_results:
        # Build placeholders with the right timestamps so downstream
        # length-keyed logic still works. ``frame_path`` is required on
        # FrameFaces — pass through the original path so any consumer
        # that wants to re-read the image can.
        other_results = [
            FrameFaces(timestamp=t, frame_path=p, faces=[]) for t, p in other_paths
        ]

    # Merge back in original order.
    merged: list[FrameFaces | None] = [None] * len(frame_paths)
    for (orig_idx, _), res in zip(speech_subset, speech_results):
        merged[orig_idx] = res
    for (orig_idx, _), res in zip(other_subset, other_results):
        merged[orig_idx] = res

    # Any None means a sub-call returned a shorter list than expected;
    # fill with empty FrameFaces so the list is well-formed.
    out = []
    for i, m in enumerate(merged):
        if m is not None:
            out.append(m)
        else:
            out.append(FrameFaces(
                timestamp=frame_paths[i][0],
                frame_path=frame_paths[i][1],
                faces=[],
            ))
    logger.info(
        "Two-pass face detection: %d frames in speech (FaceMesh+YuNet), "
        "%d outside speech (YuNet-only)",
        len(speech_subset), len(other_subset),
    )
    return out


def detect_faces_batch(
    frame_paths: list[tuple[float, str]],
    min_confidence: float = 0.3,
    progress_callback=None,
) -> list[FrameFaces]:
    """Detect faces in extracted frames.

    Tries MediaPipe first (more accurate), falls back to OpenCV Haar cascade.
    When FaceMesh finds very few multi-face frames, supplements with YuNet
    to catch additional speakers that FaceMesh missed.
    Returns list of FrameFaces, one per input frame.
    """
    import time as _t
    t0 = _t.monotonic()

    final_results = None

    # Try FaceMesh first (gives lip landmarks for active speaker detection)
    facemesh_results = None
    try:
        facemesh_results = _detect_with_facemesh(frame_paths, min_confidence, progress_callback=progress_callback)
        if facemesh_results is not None:
            elapsed = _t.monotonic() - t0
            logger.info("Face detection using MediaPipe FaceMesh (%.1fs for %d frames)", elapsed, len(frame_paths))
            _log_summary(facemesh_results)

            # Always supplement FaceMesh with YuNet to catch faces that FaceMesh missed.
            with_faces = sum(1 for r in facemesh_results if r.faces)
            multi = sum(1 for r in facemesh_results if len(r.faces) >= 2)
            max_faces_per_frame = max((len(r.faces) for r in facemesh_results), default=0)

            if with_faces >= 5:
                try:
                    yunet_results = _detect_with_opencv_dnn(frame_paths, min_confidence)
                    if yunet_results is not None:
                        yunet_total = sum(len(r.faces) for r in yunet_results)
                        fm_total = sum(len(r.faces) for r in facemesh_results)
                        if yunet_total > fm_total:
                            merged = _merge_detections(facemesh_results, yunet_results)
                            new_total = sum(len(r.faces) for r in merged)
                            new_multi = sum(1 for r in merged if len(r.faces) >= 2)
                            logger.info(
                                "FaceMesh+YuNet merge: %d→%d total faces, multi-face frames %d→%d "
                                "(FaceMesh max %d/frame, YuNet found %d extra)",
                                fm_total, new_total, multi, new_multi,
                                max_faces_per_frame, new_total - fm_total,
                            )
                            _log_summary(merged)
                            final_results = merged
                        else:
                            logger.info(
                                "YuNet supplement: no extra faces (FaceMesh=%d, YuNet=%d)",
                                fm_total, yunet_total,
                            )
                except Exception as e:
                    logger.debug("YuNet supplement failed: %s", e)

            if final_results is None:
                final_results = facemesh_results
    except Exception as e:
        logger.info("FaceMesh unavailable (%s), trying FaceDetection", e)

    # Try MediaPipe FaceDetection (no lip landmarks but more robust)
    if final_results is None:
        try:
            results = _detect_with_mediapipe(frame_paths, min_confidence)
            if results is not None:
                elapsed = _t.monotonic() - t0
                logger.info("Face detection using MediaPipe FaceDetection (%.1fs for %d frames)", elapsed, len(frame_paths))
                _log_summary(results)
                final_results = results
        except Exception as e:
            logger.info("MediaPipe FaceDetection unavailable (%s), trying OpenCV", e)

    # Fall back to OpenCV (YuNet DNN → Haar cascade)
    if final_results is None:
        try:
            results = _detect_with_opencv_dnn(frame_paths, min_confidence)
            if results is not None:
                elapsed = _t.monotonic() - t0
                logger.info("Face detection using OpenCV DNN/Haar (%.1fs for %d frames)", elapsed, len(frame_paths))
                _log_summary(results)
                final_results = results
        except Exception as e:
            logger.warning("OpenCV face detection failed: %s", e)

    # All methods failed — return empty results (graceful degradation)
    if final_results is None:
        elapsed = _t.monotonic() - t0
        logger.warning("All face detection methods failed (%.1fs) — using AI estimates only", elapsed)
        final_results = [
            FrameFaces(timestamp=ts, frame_path=str(p))
            for ts, p in frame_paths
        ]

    # Run human face verification on all detected faces
    final_results = _verify_faces_in_results(final_results, frame_paths)

    return final_results


def _augment_dense_with_anime(
    results: list,
    frame_paths: list,
    *,
    strong_conf_threshold: float = 0.55,
    job_log_prefix: str = "",
    anime_priority: bool = False,
) -> tuple[int, int]:
    """Augment ``results`` with anime-cascade detections on weak frames.

    Week 2 Part A. Runs ``detect_anime_faces`` on every result whose
    ``faces`` list is empty or whose strongest face is below
    ``strong_conf_threshold``. Detections are converted to ``FaceInfo``
    via ``anime_face_detector.to_face_info`` (which sets
    ``is_human=False`` so the human-pose verifier skips them) and
    appended to the frame's ``faces`` list — they never replace
    live-action detections that were already confident.

    The helper is module-level so tests can exercise it without
    standing up the whole ``detect_faces_dense`` pipeline.

    Args:
        results: Mutable list of ``FrameFaces`` from one of the
            ``_detect_with_*`` functions (same order as ``frame_paths``).
        frame_paths: Parallel list of ``(timestamp, path)`` tuples.
            The paths must still point to extant frame files — call
            this while the enclosing ``TemporaryDirectory`` is alive.
        strong_conf_threshold: Live-action confidence floor above
            which a frame is considered "already confident" and is
            skipped. Default 0.55 matches the YuNet / FaceMesh
            production threshold.
        job_log_prefix: Optional "[job_id] " string for log lines.
        anime_priority: When True, run the cascade on EVERY frame
            (not just sub-threshold ones). On anime content YuNet
            routinely scores mascots / eye-shaped patterns / hair
            highlights at 0.7–0.9 and locks out the cascade on the
            very frames where we need it. With the flag set we run
            the cascade unconditionally and demote any pre-existing
            live-action face that lacks an SFace embedding cluster
            match within ±5 frames (those are the YuNet anime false
            positives) so the registry weighting drops them.

    Returns:
        ``(augmented_frames, augmented_faces)`` counts. Both zero when
        the feature flag is OFF, the detector module can't be imported,
        or no frames needed augmentation.
    """
    try:
        from backend.services.anime_face_detector import (
            USE_ANIME_FACE_DETECTOR,
            detect_anime_faces,
            to_face_info,
        )
    except ImportError:
        return (0, 0)

    if not USE_ANIME_FACE_DETECTOR:
        return (0, 0)

    augmented_frames = 0
    augmented_faces = 0

    for i, fr in enumerate(results):
        # Skip frames where live-action already found a confident face,
        # UNLESS anime_priority is set — on anime content YuNet locks
        # in false positives we need to displace.
        if not anime_priority:
            strong_faces = [
                f for f in fr.faces
                if getattr(f, "confidence", 0.0) >= strong_conf_threshold
            ]
            if strong_faces:
                continue

        if i >= len(frame_paths):
            continue
        _ts, fp = frame_paths[i]
        anime_result = detect_anime_faces(fp, timestamp=_ts)
        if not anime_result.has_faces:
            continue

        # Append anime detections — do NOT replace the live-action list.
        # If YuNet picked up a faint face AND the anime cascade picked
        # up a different one, the face_registry identity-clustering step
        # resolves overlaps downstream.
        for det in anime_result.detections:
            try:
                fr.faces.append(to_face_info(det))
            except Exception:
                continue
        augmented_frames += 1
        augmented_faces += len(anime_result.detections)

    # ── Demote YuNet false positives on anime-priority runs ──
    # When anime_priority=True, any live-action face that does NOT have
    # an SFace embedding-cluster match within ±5 frames is almost
    # certainly a YuNet hit on a mascot / eye-shaped patch / hair
    # highlight. We don't drop it (the dense-propagator still wants the
    # bbox), but we scale its confidence by 0.3 so the registry's
    # confidence-weighted aggregation drops it relative to the cascade
    # detections.
    if anime_priority:
        _demoted = _demote_unmatched_live_action(results, window=5)
        if _demoted:
            logger.info(
                "%s[DenseFaces] Anime priority: demoted %d unmatched "
                "live-action detections",
                job_log_prefix, _demoted,
            )

    if augmented_frames:
        logger.info(
            "%s[DenseFaces] Anime augmentation: +%d faces across %d frames",
            job_log_prefix, augmented_faces, augmented_frames,
        )
    return (augmented_frames, augmented_faces)


def _demote_unmatched_live_action(results: list, *, window: int = 5) -> int:
    """Demote live-action faces that have no SFace cluster match nearby.

    For the anime-priority pass, any face with ``is_human=True`` whose
    identity_embedding cannot be matched (cosine similarity > 0.4) to
    another live-action face within ``±window`` frames is treated as a
    YuNet false positive on a mascot / hair highlight / eye-shaped
    background patch. We multiply its confidence by 0.3 in place so
    the downstream face_registry drops it without losing the bbox.

    Returns the number of demoted faces.
    """
    if not results:
        return 0

    # Build a lightweight index of (frame_idx, face_idx, embedding) for
    # all live-action faces with embeddings. Cascade detections are
    # marked is_human=False so they're naturally excluded.
    indexed = []
    for fi, fr in enumerate(results):
        for face in getattr(fr, "faces", []) or []:
            if not getattr(face, "is_human", True):
                continue
            emb = getattr(face, "identity_embedding", None)
            if emb is None:
                continue
            indexed.append((fi, face, emb))

    if len(indexed) < 2:
        return 0

    try:
        import numpy as _np
    except ImportError:
        return 0

    # Pre-normalize embeddings for cosine similarity.
    def _norm(v):
        arr = _np.asarray(v, dtype=_np.float32).flatten()
        n = float(_np.linalg.norm(arr))
        if n <= 1e-9:
            return None
        return arr / n

    normed = []
    for fi, face, emb in indexed:
        ne = _norm(emb)
        if ne is None:
            continue
        normed.append((fi, face, ne))

    if len(normed) < 2:
        return 0

    demoted = 0
    for i, (fi, face, ne) in enumerate(normed):
        matched = False
        for j, (fj, _other_face, oe) in enumerate(normed):
            if i == j:
                continue
            if abs(fj - fi) > window:
                continue
            sim = float(_np.dot(ne, oe))
            if sim > 0.4:
                matched = True
                break
        if not matched:
            try:
                face.confidence = float(face.confidence) * 0.3
                demoted += 1
            except Exception:
                continue
    return demoted


def detect_faces_dense(
    video_path: str,
    start: float,
    end: float,
    sample_rate: float = 0.5,
    min_confidence: float = 0.5,
    extract_embeddings: bool = True,
    progress_callback=None,
    is_animated: bool = False,
) -> list:
    """Dense face detection for a clip's time range.

    Unlike the pipeline's sparse detection (1 frame every ~5s), this extracts
    frames at 2 FPS and runs full face detection + embedding on each.
    Used at export time to build accurate per-frame face position data
    for layout decisions and smooth tracking.

    For a 30-second clip at 0.5s intervals = 60 frames.
    At ~10ms per frame (YuNet + SFace) = ~600ms total. Fast enough for export.
    """
    import re as _re
    import subprocess
    import tempfile
    import time as _time

    duration = end - start
    if duration <= 0:
        return []

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "ffmpeg", "-y", "-threads", "2",
            "-ss", str(start),
            "-t", str(duration),
            "-i", video_path,
            "-vf", (
                f"fps=1/{sample_rate},"
                "scale='min(1920,iw)':'min(1080,ih)'"
                ":force_original_aspect_ratio=decrease"
            ),
            "-vsync", "vfr", "-q:v", "5",
            os.path.join(tmpdir, "dense_%04d.jpg"),
        ]
        # Scale timeout: ~0.5s per expected frame + 30s base for FFmpeg startup.
        # VP9 4K is ~3x slower to decode than H.264 1080p.
        expected_frames = int(duration / sample_rate)
        ffmpeg_timeout = max(120, int(30 + expected_frames * 0.5))
        logger.info(
            "[DenseFaces] Starting FFmpeg extraction: %.0fs video, %d expected frames, timeout=%ds",
            duration, expected_frames, ffmpeg_timeout,
        )
        try:
            # Use Popen to stream stderr and report frame extraction progress
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            _last_reported_frame = 0
            _last_progress_time = _time.monotonic()
            _frame_re = _re.compile(r'frame=\s*(\d+)')
            try:
                for line in proc.stderr:
                    match = _frame_re.search(line)
                    if match and progress_callback:
                        current_frame = int(match.group(1))
                        now = _time.monotonic()
                        if (current_frame - _last_reported_frame >= 50
                                or now - _last_progress_time >= 5):
                            pct = min(99, 100 * current_frame // max(expected_frames, 1))
                            progress_callback(
                                "extracting_frames", current_frame, expected_frames)
                            _last_reported_frame = current_frame
                            _last_progress_time = now
                proc.wait(timeout=ffmpeg_timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                logger.warning(
                    "[DenseFaces] FFmpeg extraction timed out after %ds for %.0fs video (%d expected frames)",
                    ffmpeg_timeout, duration, expected_frames,
                )
                return []
            if proc.returncode != 0:
                logger.warning(
                    "[DenseFaces] FFmpeg extraction failed (rc=%d)",
                    proc.returncode,
                )
                return []
        except FileNotFoundError:
            logger.warning("[DenseFaces] FFmpeg not found")
            return []

        frame_files = sorted(
            f for f in os.listdir(tmpdir) if f.startswith("dense_")
        )
        if not frame_files:
            return []

        if progress_callback:
            progress_callback("extracting_done", len(frame_files), expected_frames)

        frame_paths = []
        for i, fname in enumerate(frame_files):
            ts = start + i * sample_rate
            frame_paths.append((ts, os.path.join(tmpdir, fname)))

        # Run face detection with FaceMesh
        if progress_callback:
            progress_callback("facemesh_start", 0, len(frame_paths))
        results = detect_faces_batch(
            frame_paths, min_confidence=min_confidence,
            progress_callback=progress_callback,
        )
        if progress_callback:
            progress_callback("facemesh_done", len(results), len(frame_paths))

        # If batch detection didn't produce embeddings (e.g. FaceMesh path),
        # supplement with YuNet+SFace for embeddings
        has_embeddings = any(
            f.identity_embedding is not None
            for fr in results for f in fr.faces
        )
        if not has_embeddings and extract_embeddings:
            if progress_callback:
                progress_callback("yunet_start", 0, len(frame_paths))
            yunet_results = _detect_with_opencv_dnn(
                frame_paths, min_confidence, extract_embeddings=True
            )
            if yunet_results:
                # Merge embeddings from YuNet into the main results
                for main_fr, yunet_fr in zip(results, yunet_results):
                    for mf in main_fr.faces:
                        if mf.identity_embedding is not None:
                            continue
                        # Find closest YuNet face by position
                        best_yf = None
                        best_dist = float('inf')
                        for yf in yunet_fr.faces:
                            dist = abs(mf.nose_x - yf.nose_x)
                            if dist < best_dist and yf.identity_embedding is not None:
                                best_dist = dist
                                best_yf = yf
                        if best_yf and best_dist < 10:
                            mf.identity_embedding = best_yf.identity_embedding

        # ── Anime face augmentation ──
        # When the caller signaled animated content, run the lbpcascade
        # anime detector on frames where live-action detection found
        # zero faces or only low-confidence (<0.55) detections. Anime
        # cascade results are marked is_human=False so the human-pose
        # verifier skips them. Runs inside the TemporaryDirectory scope
        # so the per-frame paths in frame_paths are still valid.
        if is_animated:
            _augment_dense_with_anime(
                results, frame_paths, anime_priority=True,
            )

        if progress_callback:
            progress_callback("complete", len(results), len(frame_paths))

        logger.info(
            "[DenseFaces] %d frames, %d with faces, %d with embeddings (%.1fs clip, %.1fs rate)",
            len(results),
            sum(1 for r in results if r.faces),
            sum(1 for r in results for f in r.faces if f.identity_embedding is not None),
            duration, sample_rate,
        )
        return results


def _log_summary(results: list[FrameFaces]):
    """Log face detection summary statistics."""
    total = sum(len(r.faces) for r in results)
    with_faces = sum(1 for r in results if r.faces)
    multi = sum(1 for r in results if len(r.faces) >= 2)
    logger.info(
        "Face detection: %d/%d frames have faces (%d total, %d multi-face)",
        with_faces, len(results), total, multi,
    )

    nose_xs = [
        r.faces[r.primary_face_idx].nose_x
        for r in results if r.faces and r.primary_face_idx >= 0
    ]
    if nose_xs:
        logger.info(
            "Face positions (nose_x): min=%.0f, max=%.0f, mean=%.1f, unique=%d",
            min(nose_xs), max(nose_xs),
            sum(nose_xs) / len(nose_xs),
            len(set(round(x) for x in nose_xs)),
        )

    # Lip aperture stats (for active speaker detection)
    all_lars = [f.lip_aperture for r in results for f in r.faces if f.lip_aperture > 0]
    if all_lars:
        logger.info(
            "Lip aperture: min=%.3f, max=%.3f, mean=%.3f, speaking_frames=%d (LAR>0.03)",
            min(all_lars), max(all_lars),
            sum(all_lars) / len(all_lars),
            sum(1 for l in all_lars if l > 0.03),
        )


# ── Gameplay Content Detection ───────────────────────────────────────────────


def detect_crosshair_persistence(sample_frame_paths: list[str]) -> float:
    """Look for a small high-contrast element at the exact center of frames.

    FPS games have a crosshair that stays fixed at the center while the
    background behind it changes.  We detect this by comparing temporal
    variance of the center 4×4 pixels vs the periphery of a 40×40 patch.

    Returns a score 0–1 indicating crosshair presence confidence.
    """
    import cv2
    import numpy as np

    if not sample_frame_paths:
        return 0.0

    center_patches = []
    for frame_path in sample_frame_paths[:30]:
        img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        cy, cx = h // 2, w // 2
        patch = img[cy - 20:cy + 20, cx - 20:cx + 20]
        if patch.shape == (40, 40):
            center_patches.append(patch)

    if len(center_patches) < 5:
        return 0.0

    # Center 4×4 should be stable (crosshair), periphery should vary (world moves)
    centers = np.stack([p[18:22, 18:22] for p in center_patches])
    center_temporal_var = np.var(centers, axis=0).mean()

    peripheries = np.stack([
        np.concatenate([p[:5, :].flatten(), p[-5:, :].flatten()])
        for p in center_patches
    ])
    periphery_temporal_var = np.var(peripheries, axis=0).mean()

    if periphery_temporal_var > 0:
        ratio = 1.0 - (center_temporal_var / periphery_temporal_var)
        return max(0.0, min(1.0, ratio))

    return 0.0


def detect_hud_corner_brightness(sample_frame_paths: list[str]) -> float:
    """Detect persistent high-saturation HUD elements in frame corners.

    Real-world footage has roughly uniform colour distribution.  Gameplay
    HUDs have brightly coloured, high-saturation overlays in fixed corner
    regions (top-right killfeed, bottom-center abilities, etc.).

    Returns a score 0–1 indicating HUD presence confidence.
    """
    import cv2
    import numpy as np

    if not sample_frame_paths:
        return 0.0

    corner_saturations: dict[str, list[float]] = {
        "tl": [], "tr": [], "bl": [], "br": [], "bc": [],
    }

    for frame_path in sample_frame_paths[:30]:
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32)

        ch, cw = h // 10, w // 10
        corner_saturations["tl"].append(float(sat[:ch, :cw].mean()))
        corner_saturations["tr"].append(float(sat[:ch, -cw:].mean()))
        corner_saturations["bl"].append(float(sat[-ch:, :cw].mean()))
        corner_saturations["br"].append(float(sat[-ch:, -cw:].mean()))
        corner_saturations["bc"].append(float(sat[-ch:, w // 2 - cw:w // 2 + cw].mean()))

    max_corner_score = 0.0
    for vals in corner_saturations.values():
        if vals:
            mean_sat = float(np.mean(vals))
            if mean_sat > 100:
                max_corner_score = max(max_corner_score, mean_sat / 255.0)

    return max_corner_score


def _verified_face_ratio(face_data: list) -> tuple[float, float]:
    """Return ``(raw_face_ratio, verified_human_ratio)`` for a list
    of :class:`FrameFaces`.

    The raw ratio uses every frame that has any detection; the
    verified ratio only counts frames that contain at least one
    detection flagged ``is_human_verified`` by
    :class:`HumanFaceVerifier`. On cartoon-FPS content like TF2,
    YuNet + MediaPipe find 70 %+ raw face rate but < 15 %
    verified rate — that divergence is the single most reliable
    signal that the video is gameplay content, not a talking
    head.

    Returns ``(0.0, 0.0)`` if no frames are supplied.
    """
    if not face_data:
        return 0.0, 0.0
    total = len(face_data)
    raw_hits = 0
    verified_hits = 0
    for f in face_data:
        faces = getattr(f, "faces", None) or []
        if not faces:
            continue
        raw_hits += 1
        # A frame counts as verified if ANY face on it passed the
        # human-face verifier.
        if any(getattr(face, "is_human_verified", False) for face in faces):
            verified_hits += 1
    return raw_hits / total, verified_hits / total


def classify_gameplay_content(
    dense_face_data: list,
    total_frames: int,
    sample_frame_paths: list[str],
    *,
    sparse_face_data=None,
    filename=None,
) -> str:
    """Classify whether video content is gameplay footage.

    Six signals, evaluated in order:

      1. **Filename keyword hint** — short-circuit match against
         the game-name keyword bank in
         :mod:`filename_gameplay_hint`. A hit returns
         ``"gameplay"`` immediately so the pipeline can skip the
         expensive dense face pass entirely.
      2. **Sparse face rarity** — if the sparse FaceMesh pass
         (passed via ``sparse_face_data``) found faces in fewer
         than 5 % of frames, that's a strong gameplay signal
         that is NOT polluted by YuNet's cartoon-character
         false positives. This is the fix for the TF2
         regression, where dense YuNet found 213 "faces" on
         cartoon models but sparse FaceMesh found 0.
      3. **Dense-vs-verified divergence** — if dense face
         detection found faces in ≥ 30 % of frames BUT the
         verified-human ratio is < 15 %, the dense pass is
         firing on cartoon characters / mannequins. Promote to
         ``"gameplay"`` instead of the legacy early-return
         that branded the clip as ``"not_gameplay"``.
      4. **Raw dense face rarity** — the legacy ≥ 30 % check.
         Only applies when the verified ratio is also healthy
         (i.e. not the cartoon case above).
      5. **Crosshair persistence** — stable center element
         across sample frames.
      6. **HUD corner brightness** — high saturation in corner
         regions indicating killfeed / minimap / ability bar.

    Returns: ``'gameplay'`` | ``'unknown'`` | ``'not_gameplay'``.
    """
    # Signal 1: filename keyword hint. Cheapest + most reliable
    # when it fires, and the short-circuit means we never let
    # YuNet's cartoon false positives cloud the decision.
    try:
        from backend.services.filename_gameplay_hint import (
            filename_gameplay_hint,
        )
        hint = filename_gameplay_hint(filename)
        if hint.matched:
            logger.info(
                "Gameplay classification: filename hint match "
                "'%s' → slug=%s genre=%s",
                hint.source_token, hint.slug, hint.genre,
            )
            return "gameplay"
    except Exception as e:
        logger.debug("filename_gameplay_hint failed: %s", e)

    # Signal 1.5: ANIME_MODE_DETECTED guard — if the human face verifier
    # already decided this is animated content (rejection rate >50%), the
    # high raw_ratio + low verified_ratio pattern is from anime character
    # faces, NOT from cartoon FPS game models. Do NOT promote to gameplay.
    # Let the anime tracking path handle it instead.
    global ANIME_MODE_DETECTED
    if ANIME_MODE_DETECTED:
        logger.info(
            "Gameplay classification: ANIME_MODE_DETECTED=True — "
            "skipping gameplay promotion (animated content, not gameplay)"
        )
        return "not_gameplay"

    # Signal 2: sparse-pass face rarity. On cartoon FPS content
    # the sparse FaceMesh pass (no YuNet augmentation) returns
    # near-zero faces — a signal that dense-pass YuNet false
    # positives will later overwrite.
    sparse_raw_ratio = 0.0
    if sparse_face_data is not None:
        sparse_raw_ratio, _sparse_verified = _verified_face_ratio(
            sparse_face_data,
        )
        if sparse_raw_ratio < 0.05 and len(sparse_face_data) >= 10:
            # Confirm with a cheap pixel signal so we don't return
            # "gameplay" on a black-frame clip.
            crosshair_score = detect_crosshair_persistence(sample_frame_paths)
            hud_score = detect_hud_corner_brightness(sample_frame_paths)
            logger.info(
                "Gameplay classification (sparse path): "
                "sparse_ratio=%.2f, crosshair=%.2f, hud=%.2f",
                sparse_raw_ratio, crosshair_score, hud_score,
            )
            if crosshair_score > 0.5 or hud_score > 0.4:
                return "gameplay"
            # Even without an HUD signal, a completely face-less
            # sparse pass is suspicious enough to tag as
            # ``"unknown"`` and let the filename / content
            # classifier take the next shot — the legacy caller
            # treats ``"unknown"`` as "skip the gameplay
            # fast-path but don't declare this talking-head"
            # which is the right middle-ground for faceless but
            # non-HUD clips (B-roll, slideshows, etc.).
            return "unknown"

    # Signal 3 + 4: dense face stats.
    raw_ratio, verified_ratio = _verified_face_ratio(dense_face_data)
    if raw_ratio == 0.0 and total_frames > 0:
        # Fallback to the legacy raw count when the verifier
        # wasn't populated on this data (backwards compat).
        frames_with_face = sum(
            1 for f in dense_face_data
            if hasattr(f, "faces") and f.faces
        )
        raw_ratio = frames_with_face / max(1, total_frames)

    # Cartoon-contamination check: if the raw rate is high but
    # the verified-human rate is very low, this is TF2-style
    # content where YuNet is firing on cartoon characters.
    if raw_ratio >= 0.30 and verified_ratio < 0.15:
        logger.info(
            "Gameplay classification: dense raw_ratio=%.2f but "
            "verified_ratio=%.2f (%.0f%% non-human rejection) — "
            "promoting to gameplay",
            raw_ratio, verified_ratio,
            (1.0 - verified_ratio / max(raw_ratio, 1e-6)) * 100.0,
        )
        return "gameplay"

    if raw_ratio > 0.30:
        return "not_gameplay"

    # Signal 5: crosshair detection
    crosshair_score = detect_crosshair_persistence(sample_frame_paths)

    # Signal 6: HUD edge density
    hud_score = detect_hud_corner_brightness(sample_frame_paths)

    logger.info(
        "Gameplay classification: face_ratio=%.2f, verified=%.2f, "
        "crosshair=%.2f, hud=%.2f",
        raw_ratio, verified_ratio, crosshair_score, hud_score,
    )

    if crosshair_score > 0.6 or (hud_score > 0.5 and raw_ratio < 0.10):
        return "gameplay"

    if raw_ratio < 0.05:
        return "unknown"

    return "not_gameplay"
