"""Dense per-frame subject propagation between sparse detector anchors.

Algorithm:
  1. Walk source frames in order
  2. At each frame:
     a. If a fresh detection exists at this timestamp, use it as the anchor:
        - Reset all trackers whose detection moved >drift_threshold from prediction
        - Initialize new trackers for new detections
     b. Otherwise, advance all active trackers by one frame
     c. If frame timestamp crossed a shot boundary, reset all trackers (wait for next anchor)
  3. Validate tracker output with optical flow on each subject's bbox
     - If LK flow disagrees with KCF center by >validation_threshold, mark frame for reset
  4. Emit a FrameSample for every source frame
"""

import logging
import time
from typing import Optional

import cv2
import numpy as np

from backend.services.interpolated_timeline import (
    FrameSample,
    InterpolatedFaceTimeline,
)
from backend.services.tracker_wrapper import SlotTracker

logger = logging.getLogger(__name__)

DRIFT_THRESHOLD_PCT = 8.0            # detection-vs-prediction reset threshold
LK_VALIDATION_THRESHOLD_PCT = 6.0    # LK-vs-KCF disagreement threshold
PROPAGATION_CONFIDENCE_DECAY = 0.05  # confidence drops per frame between anchors
ANCHOR_CONFIDENCE = 0.95
ABSENCE_KILL_THRESHOLD = 2           # consecutive anchors without detection → kill tracker


def build_interpolated_timeline(
    dense_face_results: list,
    frame_paths: list,
    source_width: int,
    source_height: int,
    source_fps: float,
    shot_cuts: list = None,
    backend: str = "KCF",
    runtime_budget_sec: float = 240.0,
    job_id: str = "",
) -> InterpolatedFaceTimeline:
    """Build a per-source-frame interpolated timeline from sparse detection anchors.

    Args:
        dense_face_results: Sparse detector output (list of FrameFaces at ~2 FPS).
        frame_paths: list[(timestamp, path)] at source FPS.
        source_width, source_height: Source video dimensions.
        source_fps: Source video FPS.
        shot_cuts: Shot boundary timestamps (tracker resets on each).
        backend: OpenCV tracker backend ("KCF", "MOSSE", "CSRT").
        runtime_budget_sec: Max wallclock time before aborting.
        job_id: For log correlation.
    """
    _log = lambda msg, *a: logger.info("[%s] DensePropagator: " + msg, job_id, *a)

    if not frame_paths:
        return InterpolatedFaceTimeline(
            samples=[], source_fps=source_fps,
            source_width=source_width, source_height=source_height,
        )

    sorted_frames = sorted(frame_paths, key=lambda f: f[0])
    sorted_anchors = sorted(dense_face_results or [], key=lambda d: d.timestamp)
    shot_cuts_sorted = sorted(shot_cuts or [])

    # Build anchor lookup: round(timestamp, 3) -> detection object
    anchor_lookup = {round(d.timestamp, 3): d for d in sorted_anchors}

    slot_trackers: dict = {}  # {slot_id: SlotTracker}
    slot_absence_counts: dict = {}  # {slot_id: consecutive anchors without detection}
    n_tracker_kills = 0
    samples = []
    prev_gray = None
    next_shot_cut_idx = 0
    n_resets = 0
    n_anchors_used = 0
    frames_since_anchor = 0

    started = time.monotonic()

    for frame_idx, (timestamp, path) in enumerate(sorted_frames):
        # Runtime budget check
        if time.monotonic() - started > runtime_budget_sec:
            _log("ABORT: runtime budget exceeded at frame %d/%d, returning partial",
                 frame_idx, len(sorted_frames))
            break

        # Shot cut check — reset all trackers on cut
        while (next_shot_cut_idx < len(shot_cuts_sorted)
               and shot_cuts_sorted[next_shot_cut_idx] <= timestamp):
            for st in slot_trackers.values():
                st._initialized = False
            slot_trackers.clear()
            slot_absence_counts.clear()
            prev_gray = None
            frames_since_anchor = 0
            next_shot_cut_idx += 1

        # Load frame
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sample = FrameSample(
            timestamp=float(timestamp),
            fps=float(source_fps),
            bboxes={},
            confidences={},
            is_anchor=False,
            had_reset=False,
        )

        # ── Advance existing trackers ──
        slots_before_update = set(slot_trackers.keys())
        for slot_id, tracker in list(slot_trackers.items()):
            bbox_px = tracker.update(img)
            if bbox_px is None:
                del slot_trackers[slot_id]
                continue
            x, y, w, h = bbox_px
            cx_pct = (x + w / 2) / source_width * 100
            cy_pct = (y + h / 2) / source_height * 100
            w_pct = w / source_width * 100
            h_pct = h / source_height * 100
            sample.bboxes[slot_id] = (float(cx_pct), float(cy_pct),
                                      float(w_pct), float(h_pct))
            # Confidence decays as we get further from the anchor
            sample.confidences[slot_id] = max(
                0.4, ANCHOR_CONFIDENCE - PROPAGATION_CONFIDENCE_DECAY * frames_since_anchor
            )

        # ── If anchor exists at this timestamp, reset/init from detection ──
        anchor_key = round(timestamp, 3)
        anchor = anchor_lookup.get(anchor_key)
        if anchor is not None:
            sample.is_anchor = True
            n_anchors_used += 1
            frames_since_anchor = 0

            for face in anchor.faces:
                slot_id = getattr(face, 'identity_id', -1)
                if slot_id < 0:
                    continue
                fx_pct = float(getattr(face, 'nose_x', getattr(face, 'x', 50)))
                fy_pct = float(getattr(face, 'nose_y', getattr(face, 'y', 50)))
                fw_pct = float(getattr(face, 'width', 10))
                fh_pct = float(getattr(face, 'height', 12))

                # Convert to pixels for tracker init
                cx_px = fx_pct / 100 * source_width
                cy_px = fy_pct / 100 * source_height
                fw_px = fw_pct / 100 * source_width
                fh_px = fh_pct / 100 * source_height
                x_px = cx_px - fw_px / 2
                y_px = cy_px - fh_px / 2

                # Drift check
                needs_reset = True
                if slot_id in sample.bboxes:
                    pred_cx, pred_cy, _, _ = sample.bboxes[slot_id]
                    drift = abs(pred_cx - fx_pct)
                    if drift < DRIFT_THRESHOLD_PCT:
                        needs_reset = False
                    else:
                        n_resets += 1
                        sample.had_reset = True
                        logger.debug("[%s] tracker_reset slot=%d t=%.2f reason=drift_%.1f%%",
                                     job_id, slot_id, timestamp, drift)
                elif slot_id in slots_before_update:
                    # Tracker was active but lost the target — also a reset
                    n_resets += 1
                    sample.had_reset = True
                    logger.debug("[%s] tracker_reset slot=%d t=%.2f reason=tracker_lost",
                                 job_id, slot_id, timestamp)

                if needs_reset or slot_id not in slot_trackers:
                    if slot_id in slot_trackers:
                        slot_trackers[slot_id].reset(img, (x_px, y_px, fw_px, fh_px))
                    else:
                        new_tracker = SlotTracker(slot_id, backend=backend)
                        if new_tracker.init(img, (x_px, y_px, fw_px, fh_px)):
                            slot_trackers[slot_id] = new_tracker

                # Always overwrite sample bbox with anchor truth
                sample.bboxes[slot_id] = (fx_pct, fy_pct, fw_pct, fh_pct)
                sample.confidences[slot_id] = ANCHOR_CONFIDENCE

            # ── Absence detection: kill trackers for slots missing from anchor ──
            anchor_slot_ids = set()
            for face in anchor.faces:
                if not getattr(face, 'is_human', True):
                    continue
                sid = getattr(face, 'identity_id', -1)
                if sid >= 0:
                    anchor_slot_ids.add(sid)

            for sid in list(slot_trackers.keys()):
                if sid in anchor_slot_ids:
                    slot_absence_counts[sid] = 0
                else:
                    slot_absence_counts[sid] = slot_absence_counts.get(sid, 0) + 1
                    if slot_absence_counts[sid] >= ABSENCE_KILL_THRESHOLD:
                        logger.info("[%s] tracker_killed slot=%d t=%.2f "
                                    "reason=absent_%d_anchors",
                                    job_id, sid, timestamp,
                                    slot_absence_counts[sid])
                        del slot_trackers[sid]
                        slot_absence_counts.pop(sid, None)
                        sample.bboxes.pop(sid, None)
                        sample.confidences.pop(sid, None)
                        n_tracker_kills += 1
        else:
            frames_since_anchor += 1

        # ── LK flow validation (cheap sanity check on propagated frames) ──
        if prev_gray is not None and slot_trackers and not sample.is_anchor:
            for slot_id in list(slot_trackers.keys()):
                if slot_id not in sample.bboxes:
                    continue
                cx_pct, cy_pct, _, _ = sample.bboxes[slot_id]
                pt_x = cx_pct / 100 * source_width
                pt_y = cy_pct / 100 * source_height
                pts = np.array([[[pt_x, pt_y]]], dtype=np.float32)
                try:
                    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                        prev_gray, gray, pts, None,
                        winSize=(21, 21), maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
                    )
                    if status is not None and status[0][0] == 1:
                        lk_x_pct = new_pts[0][0][0] / source_width * 100
                        disagreement = abs(lk_x_pct - cx_pct)
                        if disagreement > LK_VALIDATION_THRESHOLD_PCT:
                            sample.confidences[slot_id] *= 0.6
                except Exception:
                    pass

        samples.append(sample)
        prev_gray = gray

    timeline = InterpolatedFaceTimeline(
        samples=samples,
        source_fps=float(source_fps),
        source_width=int(source_width),
        source_height=int(source_height),
    )
    timeline.build_index()

    elapsed = time.monotonic() - started
    _log("FINAL: %d samples (%d anchors, %d resets, %d tracker_kills) in %.1fs "
         "(budget=%ds, backend=%s)",
         len(samples), n_anchors_used, n_resets, n_tracker_kills,
         elapsed, int(runtime_budget_sec), backend)
    return timeline
