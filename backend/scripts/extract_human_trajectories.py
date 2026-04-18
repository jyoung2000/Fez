"""Extract per-frame crop-window trajectories from (source, human-vertical)
clip pairs.

For every manifest entry with ``ground_truth_vertical_url`` fetched
into the real-content cache, this script:

  1. Loads the source (16:9) and the human vertical (9:16) clips.
  2. Samples both at a common rate (default 6 fps for speed) and finds
     the per-frame homography mapping vertical-frame → source-frame
     coordinates via ORB features + RANSAC.
  3. Projects the vertical frame's four corners through the homography
     to recover the human's per-frame crop window ``(cx, cy, w, h)`` in
     source-frame normalized coordinates.
  4. Optionally runs a lightweight face detector on the source frame to
     record ``subject_cx/_cy`` so the lead-room metric works.
  5. Writes ``data/human_trajectories/<slug>.jsonl`` — one JSON object
     per sampled frame.

This is the single place homography / feature matching lives so the
metrics module stays numpy-free. It intentionally only runs offline —
not on the pipeline hot path.

If OpenCV is unavailable (`cv2` missing) this script logs and exits 0
without writing — CI skips it in that case.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_HUMAN_TRAJ_DIR = Path("data/human_trajectories")
_MIN_MATCHES = 12
_DEFAULT_SAMPLE_FPS = 6.0


def _iter_frames(path: str, target_fps: float):
    """Yield ``(t, bgr_frame)`` tuples sampled at ``target_fps``."""
    import cv2  # local import so CI without OpenCV still imports the module
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(src_fps / target_fps)))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            t = idx / src_fps
            yield t, frame
        idx += 1
    cap.release()


def _recover_crop_window(
    source_frame,
    vertical_frame,
) -> tuple[float, float, float, float] | None:
    """Given a 16:9 source frame and a 9:16 human-edited frame of the
    same moment, return the crop window used by the human as
    ``(cx, cy, w, h)`` in source-frame normalized coordinates.

    Returns None when feature matching fails.
    """
    import cv2
    import numpy as np

    src_h, src_w = source_frame.shape[:2]
    v_h, v_w = vertical_frame.shape[:2]

    orb = cv2.ORB_create(nfeatures=1500)
    kp_s, des_s = orb.detectAndCompute(source_frame, None)
    kp_v, des_v = orb.detectAndCompute(vertical_frame, None)
    if des_s is None or des_v is None:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_v, des_s)
    if len(matches) < _MIN_MATCHES:
        return None
    src_pts = np.float32([kp_v[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_s[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    if H is None:
        return None

    corners_v = np.float32([
        [0, 0], [v_w, 0], [v_w, v_h], [0, v_h],
    ]).reshape(-1, 1, 2)
    corners_in_src = cv2.perspectiveTransform(corners_v, H).reshape(-1, 2)
    xs = corners_in_src[:, 0] / src_w
    ys = corners_in_src[:, 1] / src_h
    x0 = float(max(0.0, min(1.0, xs.min())))
    x1 = float(max(0.0, min(1.0, xs.max())))
    y0 = float(max(0.0, min(1.0, ys.min())))
    y1 = float(max(0.0, min(1.0, ys.max())))
    w = max(1e-3, x1 - x0)
    h = max(1e-3, y1 - y0)
    cx = x0 + w * 0.5
    cy = y0 + h * 0.5
    return (cx, cy, w, h)


def _detect_face_center(frame) -> tuple[float, float] | None:
    """Best-effort largest face center in the source frame, normalized."""
    try:
        import cv2
    except Exception:
        return None
    try:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        clf = cv2.CascadeClassifier(path)
        if clf.empty():
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = clf.detectMultiScale(gray, 1.2, 4)
        if len(boxes) == 0:
            return None
        x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
        H, W = frame.shape[:2]
        return ((x + w * 0.5) / W, (y + h * 0.5) / H)
    except Exception:
        return None


def extract(source_path: str, vertical_path: str, *, target_fps: float) -> list[dict]:
    try:
        import cv2  # noqa: F401
    except Exception as e:
        logger.warning("cv2 unavailable (%s) — extraction skipped", e)
        return []

    out: list[dict] = []
    src_iter = _iter_frames(source_path, target_fps)
    v_iter = _iter_frames(vertical_path, target_fps)
    while True:
        try:
            t_s, f_s = next(src_iter)
        except StopIteration:
            break
        try:
            _, f_v = next(v_iter)
        except StopIteration:
            break
        rect = _recover_crop_window(f_s, f_v)
        if rect is None:
            continue
        cx, cy, w, h = rect
        face = _detect_face_center(f_s)
        rec = {
            "t": t_s,
            "cx": cx, "cy": cy, "w": w, "h": h,
            "n_subjects_in_crop": 1,
            "is_split_screen": False,
        }
        if face is not None:
            rec["subject_cx"], rec["subject_cy"] = face
        out.append(rec)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="tests/real_content/manifest.json")
    parser.add_argument("--cache-dir", default=os.environ.get(
        "CLIPAI_REAL_CONTENT_CACHE", "/var/cache/clipai/real_content"))
    parser.add_argument("--fps", type=float, default=_DEFAULT_SAMPLE_FPS)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    cache = Path(args.cache_dir)
    _HUMAN_TRAJ_DIR.mkdir(parents=True, exist_ok=True)

    for clip in manifest.get("clips", []):
        slug = clip["slug"]
        if not clip.get("ground_truth_vertical_url"):
            continue
        src = cache / f"{slug}.{clip.get('ext', 'mp4')}"
        vert = cache / f"{slug}.vertical.{clip.get('ext', 'mp4')}"
        if not src.exists() or not vert.exists():
            logger.info("[%s] missing source or vertical — skipped", slug)
            continue
        records = extract(str(src), str(vert), target_fps=args.fps)
        if not records:
            logger.warning("[%s] no records produced", slug)
            continue
        out_path = _HUMAN_TRAJ_DIR / f"{slug}.jsonl"
        with out_path.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        logger.info("[%s] %d samples → %s", slug, len(records), out_path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
