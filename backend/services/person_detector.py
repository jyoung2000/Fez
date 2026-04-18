"""Person body detector for the v4 AttentionAnchor fallback chain.

Why a separate module: 63% of faceless frames in dialogue / action footage
left the camera with nothing to anchor on after v3. A character with their
back turned, in profile, or far from camera produces no face but is clearly
locatable as a person bbox. Feeding those bboxes into the AttentionAnchor
priority chain (between "any face" and "last-face-decay") gives the
camera solver a real subject to lock onto instead of dropping into
last-face-decay or saliency-peak.

This module is a thin wrapper around the existing object_detector.py
YOLOv8n / MobileNet-SSD pipeline:
  - Reuses the singleton detector so we don't reload weights.
  - Filters to class_name == "person" + min confidence.
  - Computes `has_face` per detection by intersecting against the
    same-frame face list (the AttentionAnchor consumer de-prioritizes
    person bboxes whose face is already covered by the face anchor).

Returns a flat list[PersonRegion] (one per detection, multiple per frame
allowed) keyed by timestamp so callers can group by frame.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# Confidence floor — YOLO sometimes emits low-conf person boxes for
# silhouettes / TV screens. 0.40 keeps recall on real subjects without
# letting noise into the anchor chain.
DEFAULT_PERSON_CONF = 0.40


@dataclass
class PersonRegion:
    """Single person body detection.

    Coordinates are normalized (0-1) with origin at top-left of the source
    frame. Width / height are absolute fractions of the source dimensions.

    Phase B adds the pose-derived head anchor fields. ``anchor_x`` /
    ``anchor_y`` are populated when CLIPAI_PERSON_USE_POSE=true and a
    YOLO11n-pose run produced confident keypoints; otherwise they default
    to the bbox center so existing consumers see no behavioural change.
    Field ordering of the original four numeric coords is preserved
    (downstream consumers may read by position).
    """
    timestamp: float
    cx: float          # 0-1 normalized center x
    cy: float          # 0-1 normalized center y
    width: float       # 0-1 normalized width
    height: float      # 0-1 normalized height
    confidence: float  # 0-1 detection confidence
    has_face: bool     # True iff a same-frame face center lies inside this bbox
    # Phase B: pose-derived head anchor (added at end of dataclass).
    anchor_x: float = None  # 0-1 normalized; None means "use cx"
    anchor_y: float = None  # 0-1 normalized; None means "use cy"
    anchor_source: str = "bbox_center"  # "nose" | "shoulders" | "bbox_top" | "bbox_center"


# ── Phase B: YOLO11n-pose head anchor ──
# Module-level singleton so we don't reload the ~6 MB pose model per batch.
# Routed via CLIPAI_PERSON_USE_POSE=true; defaults off. Requires the
# yolo11n-pose.pt weights, which ultralytics auto-downloads on first use.
# CPU-only — matches object_detector's CPU-only invocation pattern below.
_PERSON_POSE_MODEL = None


def _get_pose_model():
    """Lazy-load YOLO11n-pose. Returns None when the flag is off or the
    weights / ultralytics import fails (graceful degrade to bbox center)."""
    global _PERSON_POSE_MODEL
    if _PERSON_POSE_MODEL is not None:
        return _PERSON_POSE_MODEL
    if os.environ.get("CLIPAI_PERSON_USE_POSE", "false").lower() not in (
        "true", "1", "yes",
    ):
        return None
    try:
        from ultralytics import YOLO
    except ImportError as e:
        logger.warning(
            "[PersonDetector] ultralytics not available — pose anchor disabled: %s",
            e,
        )
        return None
    try:
        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        model_path = os.path.join(model_dir, "yolo11n-pose.pt")
        if not os.path.exists(model_path):
            # Try the /data/models path used by object_detector first.
            alt = os.path.join("/data", "models", "yolo11n-pose.pt")
            if os.path.exists(alt):
                model_path = alt
            else:
                model_path = "yolo11n-pose.pt"  # auto-download
        _PERSON_POSE_MODEL = YOLO(model_path)
        logger.info(
            "[PersonDetector] Loaded yolo11n-pose for head-anchor extraction "
            "(path=%s)", model_path,
        )
        return _PERSON_POSE_MODEL
    except Exception as e:
        logger.warning(
            "[PersonDetector] yolo11n-pose unavailable: %s — using bbox centers",
            e,
        )
        return None


def _anchor_from_keypoints(kpts, bbox) -> tuple:
    """Return (anchor_x, anchor_y, source) in 0-1 coords.

    COCO keypoint order: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear,
    4=right_ear, 5=left_shoulder, 6=right_shoulder, ...

    Priority: nose > shoulder-midpoint > bbox top-third center > bbox center.
    Each kpt is (x, y, conf) in 0-1 coords. ``bbox`` is (x1, y1, x2, y2)
    normalized 0-1.
    """
    x1, y1, x2, y2 = bbox
    nose = kpts[0]
    ls, rs = kpts[5], kpts[6]

    if float(nose[2]) > 0.5:
        return float(nose[0]), float(nose[1]), "nose"
    if float(ls[2]) > 0.5 and float(rs[2]) > 0.5:
        return (
            (float(ls[0]) + float(rs[0])) / 2,
            (float(ls[1]) + float(rs[1])) / 2,
            "shoulders",
        )
    # Back-turned with occluded keypoints — aim at upper third of bbox
    return (x1 + x2) / 2, y1 + (y2 - y1) / 3, "bbox_top"


def _face_inside(face, person: PersonRegion) -> bool:
    """Return True if the face's nose lies within the person bbox.

    Face coordinates are 0-100 percent (face_detector convention); the
    PersonRegion is 0-1. Convert face to 0-1 and check containment.
    """
    fx = float(getattr(face, "nose_x", 50.0)) / 100.0
    fy = float(getattr(face, "nose_y", 50.0)) / 100.0
    half_w = person.width / 2.0
    half_h = person.height / 2.0
    return (
        person.cx - half_w <= fx <= person.cx + half_w
        and person.cy - half_h <= fy <= person.cy + half_h
    )


def detect_persons_in_frames(
    frame_paths: list,
    face_results: Optional[list] = None,
    conf_threshold: float = DEFAULT_PERSON_CONF,
    job_id: str = "",
) -> list:
    """Run YOLOv8n / MobileNet-SSD over frame_paths and return person bodies.

    Args:
        frame_paths: list of (timestamp, path) tuples — same shape the
            existing object_detector accepts. Pipeline.py constructs this
            from the frame_extractor output.
        face_results: optional list[FrameFaces] aligned with frame_paths
            (same length, same order) so we can mark has_face per detection.
        conf_threshold: minimum YOLO/SSD confidence to keep a detection.
        job_id: for logging.

    Returns:
        list[PersonRegion] — flat list, may contain multiple regions per
        frame. Empty if no detector backend is available.
    """
    # Lazy import so test files that don't need cv2 / ultralytics still
    # import this module without pulling in the heavy detector path.
    from backend.services.object_detector import get_detector
    import cv2  # local import — only used inside the detect loop

    detector = get_detector()
    if detector.backend_name == "none":
        logger.info(
            "[%s] PersonDetector: no backend available (object_detector=none)",
            job_id,
        )
        return []

    # Build a face index aligned with frame_paths so per-frame has_face
    # checks are O(faces in this frame), not O(all faces).
    face_by_index = {}
    if face_results:
        for i, ff in enumerate(face_results):
            face_by_index[i] = list(getattr(ff, "faces", []) or [])

    # Phase B: try to load the yolo11n-pose model. If it's available,
    # we'll run it once per frame (CPU) and attach pose-derived head
    # anchors to each PersonRegion. When the loader returns None
    # (flag off / weights missing) we fall back to bbox centers.
    pose_model = _get_pose_model()
    pose_anchor_count = 0  # how many regions got a pose anchor

    persons: list = []
    n_frames = 0
    n_with_face = 0
    n_without_face = 0
    for i, item in enumerate(frame_paths):
        # Accept (timestamp, path) tuples — same shape as object_detector
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            timestamp, path = item[0], item[1]
        else:
            continue
        n_frames += 1
        img = cv2.imread(str(path))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]

        try:
            raw_detections = detector.detect(img, float(timestamp))
        except Exception as exc:
            logger.warning(
                "[%s] PersonDetector: detect failed at t=%.2f: %s",
                job_id, float(timestamp), exc,
            )
            continue

        # Phase B: run pose extraction once per frame. Returns a list of
        # (bbox_xyxy_normalized, kpts_normalized) tuples, one per detected
        # person. Empty list when the pose model is disabled.
        pose_results = []
        if pose_model is not None:
            try:
                # Use the same device-resolution policy as ObjectDetector
                # so YOLO11n-pose runs on the GPU when available. Honors
                # the GPU_ACCELERATION_ENABLED kill switch.
                from backend.services.object_detector import _resolve_yolo_device
                _pose_device = _resolve_yolo_device()
                yres = pose_model(
                    img, verbose=False, device=_pose_device,
                    conf=max(0.25, conf_threshold - 0.1),
                )
                for r in yres:
                    if r.keypoints is None or r.boxes is None:
                        continue
                    kxy = getattr(r.keypoints, "xyn", None)
                    kconf = getattr(r.keypoints, "conf", None)
                    if kxy is None:
                        continue
                    kxy_arr = kxy.cpu().numpy() if hasattr(kxy, "cpu") else kxy
                    kconf_arr = (
                        kconf.cpu().numpy() if hasattr(kconf, "cpu") else kconf
                    )
                    boxes_xyxyn = r.boxes.xyxyn
                    boxes_arr = (
                        boxes_xyxyn.cpu().numpy()
                        if hasattr(boxes_xyxyn, "cpu") else boxes_xyxyn
                    )
                    for bi in range(len(boxes_arr)):
                        kps_xy = kxy_arr[bi]
                        kps_c = (
                            kconf_arr[bi] if kconf_arr is not None
                            else [1.0] * len(kps_xy)
                        )
                        kps = [
                            (float(kps_xy[k][0]), float(kps_xy[k][1]),
                             float(kps_c[k]))
                            for k in range(len(kps_xy))
                        ]
                        bx = boxes_arr[bi]
                        pose_results.append((
                            (float(bx[0]), float(bx[1]),
                             float(bx[2]), float(bx[3])),
                            kps,
                        ))
            except Exception as exc:
                logger.debug(
                    "[%s] PersonDetector: pose forward failed at t=%.2f: %s",
                    job_id, float(timestamp), exc,
                )

        def _match_pose_to_person(person_cx, person_cy, person_w, person_h):
            """Find the pose result whose bbox-center is closest to the
            person bbox-center. Returns (anchor_x, anchor_y, source) or
            (None, None, "bbox_center") on no match."""
            if not pose_results:
                return None, None, "bbox_center"
            best = None
            best_dist = float("inf")
            for bbox, kps in pose_results:
                bx1, by1, bx2, by2 = bbox
                pcx = (bx1 + bx2) / 2
                pcy = (by1 + by2) / 2
                dist = (pcx - person_cx) ** 2 + (pcy - person_cy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best = (bbox, kps)
            if best is None:
                return None, None, "bbox_center"
            # Require the matched pose center to lie inside the person bbox
            # (within 1.5x bbox half-extent — generous so YOLO bbox jitter
            # doesn't drop matches).
            bbox, kps = best
            bx1, by1, bx2, by2 = bbox
            pcx = (bx1 + bx2) / 2
            pcy = (by1 + by2) / 2
            if (
                abs(pcx - person_cx) > 1.5 * person_w
                or abs(pcy - person_cy) > 1.5 * person_h
            ):
                return None, None, "bbox_center"
            ax, ay, src = _anchor_from_keypoints(kps, bbox)
            return ax, ay, src

        # Filter to person class above conf threshold.
        for det in raw_detections:
            if getattr(det, "class_name", "") != "person":
                continue
            if float(getattr(det, "confidence", 0.0)) < conf_threshold:
                continue
            # ObjectDetection is in 0-100 percent, top-left origin via
            # bbox_center: cx, cy are centers, w/h are full widths in %.
            cx = float(det.x) / 100.0
            cy = float(det.y) / 100.0
            pw = float(det.w) / 100.0
            ph = float(det.h) / 100.0
            person = PersonRegion(
                timestamp=float(timestamp),
                cx=cx, cy=cy, width=pw, height=ph,
                confidence=float(det.confidence),
                has_face=False,
            )
            # Phase B: attach pose-derived head anchor when available.
            ax, ay, src = _match_pose_to_person(cx, cy, pw, ph)
            if ax is not None and ay is not None:
                person.anchor_x = float(max(0.0, min(1.0, ax)))
                person.anchor_y = float(max(0.0, min(1.0, ay)))
                person.anchor_source = src
                pose_anchor_count += 1
            else:
                # No pose match — leave anchor_x/y None so consumers fall
                # back to (cx, cy). anchor_source stays "bbox_center".
                pass
            # has_face: any face in this frame whose nose lies inside the
            # person bbox. We use the index because face_results is aligned
            # with frame_paths.
            faces_here = face_by_index.get(i, [])
            for face in faces_here:
                if _face_inside(face, person):
                    person.has_face = True
                    break
            persons.append(person)
            if person.has_face:
                n_with_face += 1
            else:
                n_without_face += 1

    logger.info(
        "[%s] [PersonDetector] %d frames processed, %d person regions detected "
        "(%d with face, %d without, %d pose-anchored)",
        job_id, n_frames, len(persons), n_with_face, n_without_face,
        pose_anchor_count,
    )
    return persons
