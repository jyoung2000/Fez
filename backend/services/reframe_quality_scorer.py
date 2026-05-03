"""Universal reframe quality scorer.

Phase 6 (final phase) of the 6-phase reframing overhaul. A single
deterministic entry point that grades a finished :class:`RenderPlan`
along five quality axes and produces a single 0-100 overall score
together with a per-axis breakdown and human-readable notes.

The scorer has TWO modes:

1. **Estimated mode** (default, fast)
   Inputs are the render plan + the same source-side analysis that
   built it (face tracks, gaze, OCR text regions). NO video render is
   required. This is what CI / smoke tests / pipeline self-checks use.

2. **Rendered mode** (optional, ffmpeg-gated)
   When ``rendered_path`` is supplied, the scorer additionally runs
   :func:`backend.services.crop_qa.validate_no_black_bars` on the
   actual rendered file. Falls back gracefully — emits a note and
   keeps the estimated black-bar score — when ffmpeg / numpy / the
   face detector are unavailable.

Five axes (each 0-100, weighted to overall):

  Subject Visibility       30 %
  Composition              25 %
  Motion Smoothness        20 %
  Black Bar                15 %
  Genre Appropriateness    10 %

Pure module: stdlib only. Numpy is *not* required and is not imported
at module load time. ``RenderPlan`` and ``RenderOpKind`` are imported
without numpy. The module respects the constraint of running in <2 s
per minute of source on CPU.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.services.render_plan import (
    MotionKeypoint,
    Rect,
    RenderOp,
    RenderOpKind,
    RenderPlan,
)

logger = logging.getLogger(__name__)


# ── Public dataclass ───────────────────────────────────────────────


@dataclass
class QualityScores:
    """Aggregate quality scores for a render plan.

    All axis scores are floats in [0, 100]. ``overall`` is the
    weighted sum (Subject 30 + Composition 25 + Smoothness 20 +
    BlackBar 15 + Genre 10).

    ``strategy_distribution`` maps every strategy label that appears
    in the plan to its fraction of total duration in [0, 1].
    """

    subject_visibility: float
    composition: float
    motion_smoothness: float
    black_bar: float
    genre_appropriateness: float
    overall: float
    strategy_distribution: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_visibility": round(self.subject_visibility, 2),
            "composition": round(self.composition, 2),
            "motion_smoothness": round(self.motion_smoothness, 2),
            "black_bar": round(self.black_bar, 2),
            "genre_appropriateness": round(self.genre_appropriateness, 2),
            "overall": round(self.overall, 2),
            "strategy_distribution": {
                k: round(v, 4) for k, v in self.strategy_distribution.items()
            },
            "notes": list(self.notes),
        }


# ── Axis weights ───────────────────────────────────────────────────

WEIGHT_SUBJECT = 0.30
WEIGHT_COMPOSITION = 0.25
WEIGHT_SMOOTHNESS = 0.20
WEIGHT_BLACKBAR = 0.15
WEIGHT_GENRE = 0.10

# Reasonable upper-bound for crop-center acceleration (fraction of
# source width per second^2). Above this we treat the path as fully
# unsmooth and clip the smoothness score to 0.
MAX_EXPECTED_ACCEL_FRAC = 4.0


# ── Helpers — strategy label normalization ─────────────────────────


_OP_TO_STRATEGY = {
    RenderOpKind.CROP: "STATIC_CENTER",
    RenderOpKind.WIDE_MASTER: "BLUR_FILL_PRESERVE",
    RenderOpKind.BLUR_FILL: "BLUR_FILL_PRESERVE",
    RenderOpKind.TRACKING_CROP: "SUBJECT_TRACKING",
    RenderOpKind.CONTEXTUAL_PAN: "CONTEXTUAL_PAN",
    RenderOpKind.MOTIVATED_PUSH_IN: "SUBJECT_TRACKING",
    RenderOpKind.MOTIVATED_PULL_OUT: "SUBJECT_TRACKING",
    RenderOpKind.SPLIT_SCREEN: "MULTI_REGION",
    RenderOpKind.STACKED_GAMEPLAY: "MULTI_REGION",
    RenderOpKind.GRID_2X2: "MULTI_REGION",
    RenderOpKind.HUD_COMPOSITE: "MULTI_REGION",
}


def _strategy_label_for_op(op: RenderOp) -> str:
    """Normalize a RenderOp into one of the six advisor strategy labels.

    Falls back to the kind name if the op carries an explicit
    ``strategy_label`` set by upstream advisor wiring (recorded for
    debugging on the op).
    """
    label = (op.strategy_label or "").strip().lower()
    # If the upstream advisor stamped its own strategy label, prefer that.
    canonical = {
        "static_center": "STATIC_CENTER",
        "subject_tracking": "SUBJECT_TRACKING",
        "speaker_alternating": "SPEAKER_ALTERNATING",
        "multi_region": "MULTI_REGION",
        "contextual_pan": "CONTEXTUAL_PAN",
        "blur_fill_preserve": "BLUR_FILL_PRESERVE",
    }
    if label in canonical:
        return canonical[label]
    # Speaker-alternating is encoded in the plan as a back-to-back
    # series of CROP ops with distinct ``speaker_slot`` values. We
    # detect that at the plan level (not per-op).
    return _OP_TO_STRATEGY.get(op.kind, op.kind.value.upper())


def _build_strategy_distribution(plan: RenderPlan) -> Dict[str, float]:
    """Return ``{label: fraction_of_duration}`` for the plan."""
    if not plan.ops:
        return {}
    total = max(plan.total_duration_sec, 1e-6)

    # First pass: detect speaker-alternating runs of CROP ops with
    # distinct slot ids in close succession.
    labels = [_strategy_label_for_op(op) for op in plan.ops]
    n = len(plan.ops)
    for i in range(n):
        op = plan.ops[i]
        if labels[i] != "STATIC_CENTER":
            continue
        if op.speaker_slot is None:
            continue
        # Look at a small window of neighbors. If at least one neighbor
        # is also a CROP op with a *different* speaker_slot, this is a
        # speaker-alternating run.
        for j in (i - 1, i + 1):
            if j < 0 or j >= n:
                continue
            other = plan.ops[j]
            if (other.kind == RenderOpKind.CROP
                    and other.speaker_slot is not None
                    and other.speaker_slot != op.speaker_slot):
                labels[i] = "SPEAKER_ALTERNATING"
                break

    accum: Dict[str, float] = {}
    for op, label in zip(plan.ops, labels):
        dur = max(op.end_sec - op.start_sec, 0.0)
        accum[label] = accum.get(label, 0.0) + dur
    return {k: v / total for k, v in accum.items() if v > 0.0}


# ── Per-frame crop-rect interpolation ──────────────────────────────


def _rect_at_time(op: RenderOp, t_in_op: float) -> Rect:
    """Interpolate an op's primary rect at timestamp ``t_in_op``
    (seconds since op.start_sec).

    For ops with motion_path, linearly interpolate between
    primary_rect (at t=0) and the keypoints. For static crops, return
    primary_rect.
    """
    primary = op.primary_rect
    if not op.motion_path:
        return primary
    # Build keypoint list: implicit (0, primary_rect) then motion_path.
    kps: List[Tuple[float, Rect]] = [(0.0, primary)]
    for kp in op.motion_path:
        kps.append((float(kp.t), kp.rect))
    # Find bounding pair.
    if t_in_op <= kps[0][0]:
        return kps[0][1]
    if t_in_op >= kps[-1][0]:
        return kps[-1][1]
    for i in range(1, len(kps)):
        t0, r0 = kps[i - 1]
        t1, r1 = kps[i]
        if t0 <= t_in_op <= t1:
            span = max(t1 - t0, 1e-6)
            alpha = (t_in_op - t0) / span
            return Rect(
                x=r0.x + alpha * (r1.x - r0.x),
                y=r0.y + alpha * (r1.y - r0.y),
                w=r0.w + alpha * (r1.w - r0.w),
                h=r0.h + alpha * (r1.h - r0.h),
            )
    return primary


def _sampled_crop_path(
    plan: RenderPlan, fps: float,
) -> List[Tuple[float, Rect]]:
    """Walk the plan and emit a sampled (t, rect) per output frame.

    Sample rate: ``fps`` (defaults to plan.fps). The output path is
    used by the smoothness and visibility axes.
    """
    if not plan.ops:
        return []
    rate = max(fps if fps and fps > 0 else plan.fps, 1.0)
    dt = 1.0 / rate
    out: List[Tuple[float, Rect]] = []
    for op in plan.ops:
        op_dur = max(op.end_sec - op.start_sec, 0.0)
        if op_dur <= 0:
            continue
        # Emit at least one sample per op.
        n_samples = max(int(math.ceil(op_dur / dt)), 1)
        for k in range(n_samples):
            t_local = (k + 0.5) * (op_dur / n_samples)
            t_global = op.start_sec + t_local
            out.append((t_global, _rect_at_time(op, t_local)))
    return out


def _rect_intersection_over_subject(
    rect: Rect, subj: Tuple[float, float, float, float],
) -> float:
    """Fraction of the subject bbox area that lies inside ``rect``.

    Both rects are normalized 0-1 fractions of the source frame.
    ``subj`` is (x_center, y_center, w, h).
    """
    sx_c, sy_c, sw, sh = subj
    sx = max(0.0, sx_c - sw / 2.0)
    sy = max(0.0, sy_c - sh / 2.0)
    sx2 = min(1.0, sx_c + sw / 2.0)
    sy2 = min(1.0, sy_c + sh / 2.0)
    s_area = max(sx2 - sx, 0.0) * max(sy2 - sy, 0.0)
    if s_area <= 0.0:
        return 0.0

    rx2 = rect.x + rect.w
    ry2 = rect.y + rect.h
    ix = max(sx, rect.x)
    iy = max(sy, rect.y)
    ix2 = min(sx2, rx2)
    iy2 = min(sy2, ry2)
    inter = max(ix2 - ix, 0.0) * max(iy2 - iy, 0.0)
    return inter / s_area


# ── Axis 1: Subject Visibility ─────────────────────────────────────


def _score_subject_visibility(
    plan: RenderPlan,
    sampled: Sequence[Tuple[float, Rect]],
    source_face_tracks: Optional[List[Any]],
    notes: List[str],
) -> float:
    """Average % of expected primary-subject bbox visible per frame.

    Each face track entry is treated as a per-timestamp face cluster.
    Heterogeneous shapes accepted: dict, dataclass, ``FrameFaces``-like.
    When no faces are available the scorer falls back to "subject =
    centered region (40%x40%)" so multi-region / blur-fill plans still
    receive a sensible score.
    """
    if not sampled:
        return 100.0

    face_by_ts = _index_faces_by_timestamp(source_face_tracks)
    visibilities: List[float] = []
    for t, rect in sampled:
        subj = _nearest_face_bbox(face_by_ts, t)
        if subj is None:
            # No face available — assume a centered subject.
            subj = (0.5, 0.5, 0.40, 0.40)
        vis = _rect_intersection_over_subject(rect, subj)
        visibilities.append(vis)

    if not visibilities:
        return 100.0
    avg = sum(visibilities) / len(visibilities) * 100.0
    if avg < 85.0:
        notes.append(
            f"subject_visibility avg {avg:.1f}% below target 85%"
        )
    return max(0.0, min(100.0, avg))


def _index_faces_by_timestamp(
    face_tracks: Optional[List[Any]],
) -> List[Tuple[float, Tuple[float, float, float, float]]]:
    """Flatten ``face_tracks`` into a sorted list of (t, subject_bbox)
    tuples.

    Supported input shapes:

    * list of dicts ``{timestamp, x_center, y_center, width, height}``
      (all in 0-1 fractions OR 0-100 percent, auto-detected)
    * list of dataclass-like objects exposing ``timestamp`` and a
      ``faces`` list (FrameFaces-shape)
    * pre-flattened list of ``FaceSample``-like objects
    """
    out: List[Tuple[float, Tuple[float, float, float, float]]] = []
    if not face_tracks:
        return out
    for entry in face_tracks:
        ts = getattr(entry, "timestamp", None)
        if ts is None and isinstance(entry, dict):
            ts = entry.get("timestamp")
        # FrameFaces-like with a list of faces?
        faces = getattr(entry, "faces", None)
        if faces is None and isinstance(entry, dict):
            faces = entry.get("faces")
        if faces:
            largest = _largest_face(faces)
            if largest is None:
                continue
            bbox = _face_to_norm_bbox(largest)
            if bbox is not None and ts is not None:
                out.append((float(ts), bbox))
            continue
        # Single sample — try direct fields.
        bbox = _face_to_norm_bbox(entry if not isinstance(entry, dict) else entry)
        if bbox is not None and ts is not None:
            out.append((float(ts), bbox))
    out.sort(key=lambda kv: kv[0])
    return out


def _largest_face(faces: Sequence[Any]) -> Optional[Any]:
    if not faces:
        return None
    def area(f):
        return float(_get(f, "width", 0.0)) * float(_get(f, "height", 0.0))
    return max(faces, key=area)


def _get(obj: Any, name: str, default: Any) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _face_to_norm_bbox(face: Any) -> Optional[Tuple[float, float, float, float]]:
    if face is None:
        return None
    x_center = _get(face, "x_center", None)
    if x_center is None:
        x_center = _get(face, "nose_x", None)
    y_center = _get(face, "y_center", None)
    if y_center is None:
        y_center = _get(face, "nose_y", None)
    w = _get(face, "width", None)
    h = _get(face, "height", None)
    if x_center is None or y_center is None or w is None or h is None:
        return None
    # Normalize 0-100 percent → 0-1 fractions when needed.
    def _norm(v):
        v = float(v)
        return v / 100.0 if v > 1.5 else v
    return (_norm(x_center), _norm(y_center), _norm(w), _norm(h))


def _nearest_face_bbox(
    face_index: Sequence[Tuple[float, Tuple[float, float, float, float]]],
    t: float,
) -> Optional[Tuple[float, float, float, float]]:
    if not face_index:
        return None
    # Linear scan is fine — face tracks are typically <600 entries
    # (1 fps * 10 min). Bisect would be overkill.
    best_d = float("inf")
    best = None
    for ts, bbox in face_index:
        d = abs(ts - t)
        if d < best_d:
            best_d = d
            best = bbox
        elif ts > t and best_d != float("inf"):
            break
    # Only accept matches within 1 second.
    return best if best_d <= 1.0 else None


# ── Axis 2: Composition ────────────────────────────────────────────


def _score_composition(
    plan: RenderPlan,
    sampled: Sequence[Tuple[float, Rect]],
    source_face_tracks: Optional[List[Any]],
    source_gaze_per_second: Optional[List[Any]],
    source_text_regions: Optional[List[Any]],
    notes: List[str],
) -> float:
    """Per-frame composition score on detected faces; averaged.

    For each frame with a known face, accrue 20 points each for:
      * headroom in [5%, 15%]
      * subject not within 5% of any crop edge
      * subject near a rule-of-thirds vertical line (within 8% of x=1/3 or 2/3)
      * look-space when gaze is angled (subject on the side
        OPPOSITE the gaze direction)
      * essential text region(s) visible (or no text expected)
    """
    if not sampled:
        return 100.0
    face_index = _index_faces_by_timestamp(source_face_tracks)
    gaze_index = _index_gaze_by_second(source_gaze_per_second)
    text_norm = _normalize_text_regions(source_text_regions)

    if not face_index and not text_norm:
        return 100.0  # nothing to compose against

    per_frame: List[float] = []
    for t, rect in sampled:
        face = _nearest_face_bbox(face_index, t)
        if face is None:
            # Without a face, score only the text-visible criterion.
            if text_norm:
                per_frame.append(100.0 if _text_visible(rect, text_norm) else 60.0)
            continue
        fx, fy, fw, fh = face
        face_top = fy - fh / 2.0
        face_bottom = fy + fh / 2.0
        face_left = fx - fw / 2.0
        face_right = fx + fw / 2.0
        score = 0.0
        # Headroom — gap between face top and crop top, normalized to
        # crop height.
        if rect.h > 0:
            headroom_frac = (face_top - rect.y) / rect.h
            if 0.05 <= headroom_frac <= 0.15:
                score += 20.0
        # Edge avoidance — subject ≥5% from any crop edge.
        if rect.w > 0 and rect.h > 0:
            min_x_gap = min(face_left - rect.x,
                            (rect.x + rect.w) - face_right) / rect.w
            min_y_gap = min(face_top - rect.y,
                            (rect.y + rect.h) - face_bottom) / rect.h
            if min_x_gap >= 0.05 and min_y_gap >= 0.05:
                score += 20.0
        # Rule of thirds — subject x relative to crop near 1/3 or 2/3.
        if rect.w > 0:
            rel_x = (fx - rect.x) / rect.w
            if (abs(rel_x - 1.0 / 3.0) <= 0.08
                    or abs(rel_x - 2.0 / 3.0) <= 0.08
                    or abs(rel_x - 0.5) <= 0.05):
                score += 20.0
        # Look space.
        yaw = _nearest_gaze(gaze_index, t)
        if yaw is None or abs(yaw) <= 0.25:
            score += 20.0  # no penalty when gaze isn't angled
        else:
            sign = 1.0 if yaw > 0 else -1.0
            rel_x = (fx - (rect.x + rect.w / 2.0)) / max(rect.w, 1e-6)
            # Subject offset should be opposite the gaze sign.
            if (sign > 0 and rel_x < -0.02) or (sign < 0 and rel_x > 0.02):
                score += 20.0
        # Text visible.
        if not text_norm or _text_visible(rect, text_norm):
            score += 20.0
        per_frame.append(score)

    if not per_frame:
        return 100.0
    avg = sum(per_frame) / len(per_frame)
    if avg < 70.0:
        notes.append(f"composition avg {avg:.1f} below 70 (5 axes × 20)")
    return max(0.0, min(100.0, avg))


def _index_gaze_by_second(
    gaze: Optional[List[Any]],
) -> List[Tuple[float, float]]:
    if not gaze:
        return []
    out: List[Tuple[float, float]] = []
    for entry in gaze:
        if isinstance(entry, dict):
            t = entry.get("timestamp")
            yaw = entry.get("yaw")
        elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
            t, yaw = float(entry[0]), float(entry[1])
        else:
            t = getattr(entry, "timestamp", None)
            yaw = getattr(entry, "yaw", None)
        if t is None or yaw is None:
            continue
        out.append((float(t), float(yaw)))
    out.sort(key=lambda kv: kv[0])
    return out


def _nearest_gaze(
    index: Sequence[Tuple[float, float]], t: float,
) -> Optional[float]:
    if not index:
        return None
    best_d = float("inf")
    best = None
    for ts, yaw in index:
        d = abs(ts - t)
        if d < best_d:
            best_d = d
            best = yaw
    return best if best_d <= 1.0 else None


def _normalize_text_regions(
    regions: Optional[List[Any]],
) -> List[Tuple[float, float, float, float, bool]]:
    """Normalize OCR text regions into a list of (x_c, y_c, w, h,
    essential) tuples in 0-1 fractions."""
    if not regions:
        return []
    out: List[Tuple[float, float, float, float, bool]] = []
    for r in regions:
        x = _get(r, "x_center", _get(r, "x_pct", 0.5))
        y = _get(r, "y_center", _get(r, "y_pct", 0.5))
        w = _get(r, "width", _get(r, "w_pct", 0.0))
        h = _get(r, "height", _get(r, "h_pct", 0.0))
        ess = bool(_get(r, "essential", False) or _get(r, "is_essential", False))

        def _norm(v):
            v = float(v)
            return v / 100.0 if v > 1.5 else v

        out.append((_norm(x), _norm(y), _norm(w), _norm(h), ess))
    return out


def _text_visible(
    rect: Rect,
    regions: Sequence[Tuple[float, float, float, float, bool]],
) -> bool:
    """True when every essential text region lies within rect."""
    essential = [r for r in regions if r[4]]
    if not essential:
        return True
    rx2 = rect.x + rect.w
    ry2 = rect.y + rect.h
    for cx, cy, w, h, _ in essential:
        x = cx - w / 2.0
        y = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        if not (rect.x <= x and rx2 >= x2 and rect.y <= y and ry2 >= y2):
            return False
    return True


# ── Axis 3: Motion Smoothness ──────────────────────────────────────


def _score_motion_smoothness(
    sampled: Sequence[Tuple[float, Rect]],
    fps: float,
    notes: List[str],
) -> float:
    """100 − (mean |accel| / max_expected) * 100, clamped to [0, 100].

    Accel = second derivative of crop center x. The mean is taken over
    all frames with valid second-derivative samples (i.e. excluding
    the first and last). Hard cuts across op boundaries are excluded
    by detecting frame-to-frame deltas >50% of the source width.
    """
    if len(sampled) < 3:
        return 100.0
    dt = 1.0 / max(fps, 1.0)
    cxs: List[float] = []
    for _, rect in sampled:
        cxs.append(rect.x + rect.w / 2.0)

    # Discard hard-cut frame pairs (huge deltas).
    accel_samples: List[float] = []
    for i in range(1, len(cxs) - 1):
        d_prev = cxs[i] - cxs[i - 1]
        d_next = cxs[i + 1] - cxs[i]
        # Skip across-cut samples.
        if abs(d_prev) > 0.5 or abs(d_next) > 0.5:
            continue
        accel = (d_next - d_prev) / (dt * dt)
        accel_samples.append(abs(accel))

    if not accel_samples:
        return 100.0
    mean_accel = sum(accel_samples) / len(accel_samples)
    raw = 100.0 - (mean_accel / MAX_EXPECTED_ACCEL_FRAC) * 100.0
    score = max(0.0, min(100.0, raw))
    if score < 70.0:
        notes.append(
            f"motion_smoothness {score:.1f} (mean |accel|={mean_accel:.2f} "
            f"frac/s²)"
        )
    return score


# ── Axis 4: Black Bar ──────────────────────────────────────────────


def _score_black_bar_estimated(plan: RenderPlan, notes: List[str]) -> float:
    """Estimate the black-bar score from the render plan alone.

    Phase 4 of the overhaul guarantees that no op kind produces black
    bars: WIDE_MASTER and BLUR_FILL render with a blurred fill,
    multi-region ops have a 2-px separator that fully covers any gap.

    The estimator returns 100 unless the plan contains a primary_rect
    smaller than (target_w/source_w, target_h/source_h) for a non-fill
    op kind — which would imply uncovered area.
    """
    if not plan.ops:
        return 100.0
    bad = 0
    total = 0
    for op in plan.ops:
        total += 1
        if op.kind in (RenderOpKind.WIDE_MASTER, RenderOpKind.BLUR_FILL):
            continue  # blur fills the frame — never black
        if op.kind in (RenderOpKind.SPLIT_SCREEN, RenderOpKind.STACKED_GAMEPLAY,
                       RenderOpKind.GRID_2X2, RenderOpKind.HUD_COMPOSITE):
            continue  # multi-region tiles tile the full output
        # Single-crop ops: rect must be normalized w >= target_aspect /
        # source_aspect for the crop to upscale to fill the output.
        # If primary_rect.w * source_w / (primary_rect.h * source_h) is
        # significantly different from target_w/target_h, we'd be
        # producing letterbox/pillarbox.
        if not op.primary_rect:
            bad += 1
            continue
        crop_aspect = (op.primary_rect.w * plan.source_width) / max(
            op.primary_rect.h * plan.source_height, 1e-6,
        )
        target_aspect = plan.target_width / max(plan.target_height, 1)
        if abs(crop_aspect - target_aspect) > 0.15:
            bad += 1

    if bad == 0:
        return 100.0
    score = 100.0 * (1.0 - bad / float(total))
    notes.append(
        f"black_bar estimated {score:.1f} ({bad}/{total} ops have "
        "aspect mismatch — possible letterboxing)"
    )
    return max(0.0, score)


def _score_black_bar_rendered(
    rendered_path: str, notes: List[str],
) -> Tuple[Optional[float], bool]:
    """Run :func:`validate_no_black_bars` on the rendered file.

    Returns (score, ran). When ffmpeg/numpy aren't available the
    function returns (None, False); the caller should fall back to
    the estimated score and append a note.
    """
    try:
        from backend.services.crop_qa import validate_no_black_bars
    except Exception as exc:  # pragma: no cover - defensive
        notes.append(f"black_bar rendered check skipped: import failed ({exc})")
        return None, False

    report = validate_no_black_bars(rendered_path, fps=1.0)
    if report.note:
        notes.append(f"black_bar rendered: {report.note}")
    if report.total_frames_sampled == 0 and report.note:
        return None, False
    if report.total_frames_sampled == 0:
        return 100.0, True
    bad = len(report.frames_with_black)
    score = 100.0 * (1.0 - bad / float(report.total_frames_sampled))
    return max(0.0, min(100.0, score)), True


# ── Axis 5: Genre Appropriateness ──────────────────────────────────


def _score_genre_appropriateness(
    content_type: Optional[str],
    distribution: Dict[str, float],
    has_facecam: bool,
    notes: List[str],
) -> float:
    """Per-genre rules from the Phase 6 spec.

    Distribution values are fractions of total duration in [0, 1].
    """
    if not content_type:
        return 100.0
    ct = content_type.strip().lower()

    static_share = distribution.get("STATIC_CENTER", 0.0)
    speaker_share = distribution.get("SPEAKER_ALTERNATING", 0.0)
    blur_share = distribution.get("BLUR_FILL_PRESERVE", 0.0)
    multi_share = distribution.get("MULTI_REGION", 0.0)
    panel_share = static_share + speaker_share

    if ct in ("talking_head", "podcast", "vlog", "multi_speaker_panel"):
        if panel_share >= 0.80:
            return 100.0
        score = 100.0 * (panel_share / 0.80)
        notes.append(
            f"genre podcast: only {panel_share*100:.0f}% STATIC/SPEAKER_ALT "
            "(target >80%)"
        )
        return max(0.0, score)

    if ct.startswith("sports"):
        if blur_share == 0.0:
            return 100.0
        score = 100.0 * (1.0 - min(1.0, blur_share / 0.30))
        notes.append(
            f"genre sports: {blur_share*100:.0f}% BLUR_FILL "
            "(target 0%)"
        )
        return max(0.0, score)

    if ct in ("animation", "animation_dialogue", "anime"):
        if static_share >= 0.70:
            return 100.0
        score = 100.0 * (static_share / 0.70)
        notes.append(
            f"genre anime: only {static_share*100:.0f}% STATIC_CENTER "
            "(target >70%)"
        )
        return max(0.0, score)

    if ct.startswith("gameplay") or ct == "stream":
        if has_facecam:
            if multi_share > 0.0:
                return 100.0
            notes.append(
                "genre gaming+facecam: 0% MULTI_REGION "
                "(facecam should be tiled)"
            )
            return 50.0
        return 100.0

    if ct == "music_video":
        if speaker_share == 0.0:
            return 100.0
        score = 100.0 * (1.0 - min(1.0, speaker_share / 0.20))
        notes.append(
            f"genre music_video: {speaker_share*100:.0f}% "
            "SPEAKER_ALTERNATING (target 0%)"
        )
        return max(0.0, score)

    return 100.0


# ── Public entry point ────────────────────────────────────────────


def score_render_plan(
    render_plan: RenderPlan,
    *,
    source_face_tracks: Optional[List[Any]] = None,
    source_gaze_per_second: Optional[List[Any]] = None,
    source_text_regions: Optional[List[Any]] = None,
    content_type: Optional[str] = None,
    rendered_path: Optional[str] = None,
    fps: float = 30.0,
    has_facecam: bool = False,
) -> QualityScores:
    """Grade a finished :class:`RenderPlan` along five quality axes.

    See module docstring for axis weights, modes, and the exact
    per-genre rules. ``rendered_path`` is optional; when supplied the
    black-bar axis runs against the rendered file via
    :func:`backend.services.crop_qa.validate_no_black_bars`.
    """
    notes: List[str] = []
    distribution = _build_strategy_distribution(render_plan)

    sample_fps = float(fps if fps and fps > 0 else render_plan.fps)
    sampled = _sampled_crop_path(render_plan, sample_fps)

    subject = _score_subject_visibility(
        render_plan, sampled, source_face_tracks, notes,
    )
    composition = _score_composition(
        render_plan, sampled, source_face_tracks,
        source_gaze_per_second, source_text_regions, notes,
    )
    smoothness = _score_motion_smoothness(sampled, sample_fps, notes)

    estimated_blackbar = _score_black_bar_estimated(render_plan, notes)
    blackbar = estimated_blackbar
    if rendered_path:
        rendered_score, ran = _score_black_bar_rendered(rendered_path, notes)
        if ran and rendered_score is not None:
            # Take the WORSE of the two — render-side findings are
            # ground truth, but a clean estimated score with a missing
            # rendered file shouldn't be punished beyond what the
            # estimator already saw.
            blackbar = min(estimated_blackbar, rendered_score)

    genre = _score_genre_appropriateness(
        content_type, distribution, has_facecam, notes,
    )

    overall = (
        WEIGHT_SUBJECT * subject
        + WEIGHT_COMPOSITION * composition
        + WEIGHT_SMOOTHNESS * smoothness
        + WEIGHT_BLACKBAR * blackbar
        + WEIGHT_GENRE * genre
    )

    return QualityScores(
        subject_visibility=subject,
        composition=composition,
        motion_smoothness=smoothness,
        black_bar=blackbar,
        genre_appropriateness=genre,
        overall=overall,
        strategy_distribution=distribution,
        notes=notes,
    )


__all__ = [
    "QualityScores",
    "score_render_plan",
    "WEIGHT_SUBJECT",
    "WEIGHT_COMPOSITION",
    "WEIGHT_SMOOTHNESS",
    "WEIGHT_BLACKBAR",
    "WEIGHT_GENRE",
]
