"""Persistent per-frame attention anchor stream.

Opus-level saliency fallback: produces exactly one anchor per dense
frame by picking the highest-confidence signal available in priority
order:

  1. active-speaker face in this frame          (conf=1.0)
  2. any confirmed face in this frame            (conf=0.9)
  3. last-seen face, decaying, blended with next (conf=0.85 → 0.4)
  4. saliency peak (saliency_score · area)      (conf=0.3 + 0.5·score)
  5. motion centroid (frame-diff CoM)            (conf=0.3)

Why this exists: in long anime / TV dialogue scenes, ~60% of dense
frames contain zero detected faces, and the raw saliency tracker only
reports blobs on a sparse timeline. Feeding the camera solver a stream
that has a region for *every* frame stops the crop from drifting onto
background motion during faceless action beats — it keeps a coherent
trajectory between the previous and next human anchor.

This module is a pure data producer: it reads dense face results +
saliency regions and emits a dense timeline of AttentionAnchor objects.
Consumers (required_regions.build_required_regions) promote these to
RequiredRegions in dialogue modes where background motion is noise.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# Decay / bridging tunables
LAST_FACE_MAX_GAP_SECONDS = 1.5  # bridge faces within this window
LAST_FACE_DECAY_MAX = 0.85
LAST_FACE_DECAY_MIN = 0.4
SALIENCY_PEAK_MIN_SCORE = 0.25
SMOOTHER_WINDOW_SECONDS = 0.6


# v4: confidence assigned to person_body anchors. Sits below face/active_face
# (0.9/1.0) and above last_face_decay (0.85→0.40) so dialogue scenes prefer
# real face detections, but faceless action beats latch onto a person body
# rather than dropping into decay.
PERSON_BODY_CONFIDENCE = 0.65

# v4: cy is biased upward inside the person bbox so the resulting crop sits
# where a face would be if one had been detected — head-and-shoulders, not
# torso-centered.
PERSON_HEAD_CY_BIAS = 0.35


def _saliency_min_score_for(content_type) -> float:
    """Return the minimum saliency score threshold for _saliency_peak_anchor.

    Non-gaming content uses a higher bar (0.35) to suppress weak peaks.
    Gameplay and None fall through to legacy 0.25.
    """
    if content_type is None:
        return 0.25
    from backend.services.required_regions import _is_gaming_mode
    if _is_gaming_mode(content_type):
        return 0.25
    return 0.35


@dataclass
class AttentionAnchor:
    """Single per-frame attention anchor for the camera solver."""
    timestamp: float
    cx: float           # 0-1 normalized frame coordinates
    cy: float
    half_width: float
    half_height: float
    confidence: float   # 0-1, lower → more extrapolation
    source: str         # "active_face" | "face" | "person_body"
                        # | "last_face_decay" | "saliency_peak"
                        # | "motion_centroid"


def _active_slot_at(active_speaker_events, timestamp: float) -> int:
    if not active_speaker_events:
        return -1
    for ev in active_speaker_events:
        if ev.start <= timestamp <= ev.end and ev.slot_id >= 0:
            return ev.slot_id
    return -1


def _face_anchor(face, timestamp: float, conf: float, source: str) -> AttentionAnchor:
    cx = face.nose_x / 100.0
    cy = face.nose_y / 100.0
    hw = (face.width / 100.0) / 2.0
    hh = (face.height / 100.0) / 2.0
    return AttentionAnchor(
        timestamp=timestamp,
        cx=cx, cy=cy,
        half_width=hw, half_height=hh,
        confidence=conf, source=source,
    )


def _best_face_in_frame(ff, active_slot: int) -> tuple:
    """Return (best_face, confidence, source) for the face at ff.

    Priority: active speaker by identity_id → largest face in frame.
    """
    if not ff.faces:
        return (None, 0.0, "")
    # Filter to is_human=True; the AnimeMode override has already restored
    # these to True for anime content.
    humans = [f for f in ff.faces if getattr(f, "is_human", True)]
    if not humans:
        return (None, 0.0, "")
    # Prefer the active speaker if present.
    if active_slot >= 0:
        for face in humans:
            if getattr(face, "identity_id", -1) == active_slot:
                return (face, 1.0, "active_face")
    # Otherwise pick the largest face — it's the most likely subject.
    largest = max(humans, key=lambda f: f.width * f.height)
    return (largest, 0.9, "face")


def _person_body_anchor(
    persons_at_t: list, timestamp: float,
) -> Optional[AttentionAnchor]:
    """Build a person-body anchor for a frame with no face detection.

    v4 priority chain rule:
      - Only persons with has_face=False are eligible. has_face=True means
        the face anchor (a higher-priority source) already covers this
        person and we'd be doubling up.
      - Among eligible persons we pick the one with the largest bbox area
        (closest to camera, most likely the subject).
      - cy is biased upward by PERSON_HEAD_CY_BIAS × height so the crop
        framing lands on the head/shoulders region instead of the torso
        center, matching where a face anchor would have been.

    Returns None if no eligible person was found.
    """
    if not persons_at_t:
        return None
    eligible = [p for p in persons_at_t if not getattr(p, "has_face", False)]
    if not eligible:
        return None
    # Largest bbox wins (closest to camera).
    best = max(eligible, key=lambda p: float(p.width) * float(p.height))
    # Phase B: prefer the pose-derived head anchor when available
    # (CLIPAI_PERSON_USE_POSE=true). It lands on the actual nose /
    # shoulder midpoint instead of a torso-biased bbox center, which is
    # critical for back-turned, profile, and far-subject frames.
    anchor_cx = getattr(best, "anchor_x", None)
    anchor_cy = getattr(best, "anchor_y", None)
    if anchor_cx is not None and anchor_cy is not None:
        cx = float(anchor_cx)
        cy = float(anchor_cy)
    else:
        cx = float(best.cx)
        cy = float(best.cy) - PERSON_HEAD_CY_BIAS * float(best.height)
        cy = max(0.0, min(1.0, cy))
    return AttentionAnchor(
        timestamp=timestamp,
        cx=cx,
        cy=cy,
        half_width=float(best.width) / 2.0,
        half_height=float(best.height) / 2.0,
        confidence=PERSON_BODY_CONFIDENCE,
        source="person_body",
    )


def _saliency_peak_anchor(
    sal_frame,
    timestamp: float,
    prev_anchor: Optional[AttentionAnchor] = None,
    content_type=None,
) -> Optional[AttentionAnchor]:
    """Pick the saliency blob with the highest score*area product.

    A small bright blob in a corner is usually noise; a larger blob
    with decent score is typically the real subject for faceless frames.

    When *prev_anchor* is a saliency_peak anchor and the new candidate
    centroid is within 0.08 * max(w, h) (normalized) of the previous
    centroid, the previous centroid is kept to reduce jitter.
    """
    if sal_frame is None:
        return None
    score = getattr(sal_frame, "mean_score", 0.0)
    blobs = getattr(sal_frame, "blobs", None) or []
    _min_score = _saliency_min_score_for(content_type)
    if not blobs or score < _min_score:
        return None
    # blobs are (x, y, w, h) normalized 0-1 top-left.
    best_blob = None
    best_rank = -1.0
    for b in blobs:
        bx, by, bw, bh = b[0], b[1], b[2], b[3]
        area = bw * bh
        # Rank by score * area so small corner blobs lose to bigger ones.
        rank = float(score) * float(area)
        if rank > best_rank:
            best_rank = rank
            best_blob = (bx, by, bw, bh)
    if best_blob is None:
        return None
    bx, by, bw, bh = best_blob
    new_cx = bx + bw / 2.0
    new_cy = by + bh / 2.0

    # Saliency lock: if prev anchor was saliency_peak and the new
    # centroid is close, keep the previous centroid to reduce jitter.
    if (prev_anchor is not None
            and getattr(prev_anchor, "source", "") == "saliency_peak"):
        # Normalized distance threshold: 0.08 of the larger dimension.
        # Since coordinates are 0-1, max(w,h) in normalized space is 1.0,
        # so threshold is simply 0.08.
        dx = new_cx - prev_anchor.cx
        dy = new_cy - prev_anchor.cy
        dist = (dx * dx + dy * dy) ** 0.5
        if dist <= 0.08:
            new_cx = prev_anchor.cx
            new_cy = prev_anchor.cy

    conf = min(1.0, 0.3 + 0.5 * float(score))
    return AttentionAnchor(
        timestamp=timestamp,
        cx=new_cx,
        cy=new_cy,
        half_width=bw / 2.0,
        half_height=bh / 2.0,
        confidence=conf,
        source="saliency_peak",
    )


def _blend(a: AttentionAnchor, b: AttentionAnchor, t: float, t_a: float, t_b: float) -> tuple:
    """Linearly interpolate position between two anchors at time t."""
    if t_b == t_a:
        return (a.cx, a.cy, a.half_width, a.half_height)
    frac = (t - t_a) / (t_b - t_a)
    frac = max(0.0, min(1.0, frac))
    cx = a.cx + (b.cx - a.cx) * frac
    cy = a.cy + (b.cy - a.cy) * frac
    hw = a.half_width + (b.half_width - a.half_width) * frac
    hh = a.half_height + (b.half_height - a.half_height) * frac
    return (cx, cy, hw, hh)


def build_attention_anchors(
    frame_faces: list,
    active_speaker_events: Optional[list] = None,
    frame_saliency: Optional[list] = None,
    shot_cuts: Optional[list] = None,
    frame_persons: Optional[list] = None,
    content_type=None,
) -> list:
    """Build a dense per-frame AttentionAnchor timeline.

    Args:
        frame_faces: list[FrameFaces] from face_detector.py
        active_speaker_events: SpeakerEvent list from active_speaker.py
        frame_saliency: list of FrameSaliency-like objects (with .blobs
            and .mean_score). May be None.
        shot_cuts: list of timestamps at which shot cuts occur; the
            temporal smoother does not blend across these.
        frame_persons: optional flat list[PersonRegion] from
            person_detector.py. v4: when present, person bodies fill in
            faceless frames between the "face" and "last_face_decay"
            priority slots so the camera has a real subject to lock onto.
        content_type: Optional content type for saliency min-score
            threshold tuning. None = legacy behaviour.

    Returns:
        list[AttentionAnchor], one per input frame, in timestamp order.
    """
    if not frame_faces:
        return []

    # Index saliency by rounded timestamp.
    sal_by_time = {}
    if frame_saliency:
        for sf in frame_saliency:
            sal_by_time[round(getattr(sf, "timestamp", 0.0), 2)] = sf

    # Index persons by rounded timestamp (multiple per frame allowed).
    person_by_time: dict = {}
    if frame_persons:
        for p in frame_persons:
            key = round(float(getattr(p, "timestamp", 0.0)), 2)
            person_by_time.setdefault(key, []).append(p)

    # First pass — attach a "direct" anchor per frame in priority order:
    #   1. active_face / face         (handled by _best_face_in_frame)
    #   2. person_body (v4)           — eligible person bbox with no face inside
    #   3. saliency_peak              — score·area-ranked saliency blob
    # Frames that get nothing here are filled in the second pass by
    # last_face_decay or motion_centroid fallback.
    direct_anchors: list = [None] * len(frame_faces)
    _prev_sal_anchor: Optional[AttentionAnchor] = None
    for i, ff in enumerate(frame_faces):
        t = float(ff.timestamp)
        active_slot = _active_slot_at(active_speaker_events, t)
        face, conf, source = _best_face_in_frame(ff, active_slot)
        if face is not None:
            direct_anchors[i] = _face_anchor(face, t, conf, source)
            continue
        # v4: person body wins over saliency_peak when a faceless person
        # is visible. Pulls anchors away from background motion onto the
        # actual subject. _person_body_anchor enforces has_face=False so
        # we never double-anchor a covered face.
        persons_at_t = person_by_time.get(round(t, 2), [])
        pa = _person_body_anchor(persons_at_t, t)
        if pa is not None:
            direct_anchors[i] = pa
            continue
        sal_frame = sal_by_time.get(round(t, 2))
        sp = _saliency_peak_anchor(
            sal_frame, t,
            prev_anchor=_prev_sal_anchor,
            content_type=content_type,
        )
        if sp is not None:
            direct_anchors[i] = sp
            _prev_sal_anchor = sp
            continue
        # Motion centroid fallback: use frame-diff CoM if the saliency
        # frame exposes it. We don't have a separate signal here — the
        # saliency mean_score already encapsulates motion in the
        # existing tracker. Leave the slot blank and let the second
        # pass fill it with a decayed face.

    # Second pass — fill gaps by bridging between the nearest face
    # anchors on each side of the gap. Confidence decays linearly from
    # LAST_FACE_DECAY_MAX at the edge of a known face to
    # LAST_FACE_DECAY_MIN at LAST_FACE_MAX_GAP_SECONDS away.
    face_indices = [
        i for i, a in enumerate(direct_anchors)
        if a is not None and a.source in ("face", "active_face")
    ]
    anchors: list = []
    n = len(frame_faces)
    for i, ff in enumerate(frame_faces):
        t = float(ff.timestamp)
        if direct_anchors[i] is not None:
            anchors.append(direct_anchors[i])
            continue
        # Find nearest face anchor before and after.
        prev_idx = None
        next_idx = None
        for fi in face_indices:
            if fi <= i:
                prev_idx = fi
            elif fi > i and next_idx is None:
                next_idx = fi
                break
        prev_a = direct_anchors[prev_idx] if prev_idx is not None else None
        next_a = direct_anchors[next_idx] if next_idx is not None else None

        filled = None
        # Bridging: decide which side(s) are close enough to this frame
        # (within LAST_FACE_MAX_GAP_SECONDS). We decay confidence by the
        # distance to the NEAREST real face anchor, NOT by the gap
        # between the two face anchors — a long faceless stretch still
        # gets bridged near each end where a face was recently seen.
        dist_prev = (t - prev_a.timestamp) if prev_a is not None else float("inf")
        dist_next = (next_a.timestamp - t) if next_a is not None else float("inf")
        prev_ok = prev_a is not None and dist_prev <= LAST_FACE_MAX_GAP_SECONDS
        next_ok = next_a is not None and dist_next <= LAST_FACE_MAX_GAP_SECONDS

        if prev_ok and next_ok:
            cx, cy, hw, hh = _blend(
                prev_a, next_a, t, prev_a.timestamp, next_a.timestamp,
            )
            dist = min(dist_prev, dist_next)
            ratio = min(1.0, dist / max(LAST_FACE_MAX_GAP_SECONDS, 1e-6))
            conf = LAST_FACE_DECAY_MAX + (LAST_FACE_DECAY_MIN - LAST_FACE_DECAY_MAX) * ratio
            filled = AttentionAnchor(
                timestamp=t,
                cx=cx, cy=cy,
                half_width=hw, half_height=hh,
                confidence=conf,
                source="last_face_decay",
            )
        elif prev_a is not None:
            dist = t - prev_a.timestamp
            if dist <= LAST_FACE_MAX_GAP_SECONDS:
                ratio = min(1.0, dist / LAST_FACE_MAX_GAP_SECONDS)
                conf = LAST_FACE_DECAY_MAX + (LAST_FACE_DECAY_MIN - LAST_FACE_DECAY_MAX) * ratio
                filled = AttentionAnchor(
                    timestamp=t,
                    cx=prev_a.cx, cy=prev_a.cy,
                    half_width=prev_a.half_width, half_height=prev_a.half_height,
                    confidence=conf,
                    source="last_face_decay",
                )
        elif next_a is not None:
            dist = next_a.timestamp - t
            if dist <= LAST_FACE_MAX_GAP_SECONDS:
                ratio = min(1.0, dist / LAST_FACE_MAX_GAP_SECONDS)
                conf = LAST_FACE_DECAY_MAX + (LAST_FACE_DECAY_MIN - LAST_FACE_DECAY_MAX) * ratio
                filled = AttentionAnchor(
                    timestamp=t,
                    cx=next_a.cx, cy=next_a.cy,
                    half_width=next_a.half_width, half_height=next_a.half_height,
                    confidence=conf,
                    source="last_face_decay",
                )

        if filled is None:
            # Final fallback: a weak motion/center anchor — we still
            # want the solver to have an input here so it doesn't drift.
            filled = AttentionAnchor(
                timestamp=t,
                cx=0.5, cy=0.5,
                half_width=0.1, half_height=0.15,
                confidence=0.3,
                source="motion_centroid",
            )
        anchors.append(filled)

    # Temporal smoothing: 0.6s boxcar on cx/cy. Does not blend across
    # shot cuts — we reset the window whenever the cut lies inside it.
    cuts = sorted(float(c) for c in (shot_cuts or []))

    def _window_bounds(i: int) -> tuple:
        t0 = anchors[i].timestamp - SMOOTHER_WINDOW_SECONDS / 2.0
        t1 = anchors[i].timestamp + SMOOTHER_WINDOW_SECONDS / 2.0
        # Clamp to the nearest surrounding shot cuts.
        for c in cuts:
            if c < anchors[i].timestamp and c > t0:
                t0 = c + 1e-6
            if c > anchors[i].timestamp and c < t1:
                t1 = c - 1e-6
                break
        return (t0, t1)

    if len(anchors) >= 2:
        jitter_before = 0.0
        jitter_after = 0.0
        prev_cx = anchors[0].cx
        smoothed_cxs = []
        smoothed_cys = []
        for i in range(len(anchors)):
            t_i = anchors[i].timestamp
            t0, t1 = _window_bounds(i)
            sx = 0.0
            sy = 0.0
            n_win = 0
            # Expand left
            j = i
            while j >= 0 and anchors[j].timestamp >= t0:
                sx += anchors[j].cx
                sy += anchors[j].cy
                n_win += 1
                j -= 1
            # Expand right (skip i since we included it above)
            j = i + 1
            while j < len(anchors) and anchors[j].timestamp <= t1:
                sx += anchors[j].cx
                sy += anchors[j].cy
                n_win += 1
                j += 1
            if n_win > 0:
                smoothed_cxs.append(sx / n_win)
                smoothed_cys.append(sy / n_win)
            else:
                smoothed_cxs.append(anchors[i].cx)
                smoothed_cys.append(anchors[i].cy)

            if i > 0:
                jitter_before += abs(anchors[i].cx - prev_cx)
            prev_cx = anchors[i].cx

        # Write smoothed values back, compute after-jitter.
        prev_sc = smoothed_cxs[0]
        for i in range(len(anchors)):
            anchors[i] = AttentionAnchor(
                timestamp=anchors[i].timestamp,
                cx=smoothed_cxs[i],
                cy=smoothed_cys[i],
                half_width=anchors[i].half_width,
                half_height=anchors[i].half_height,
                confidence=anchors[i].confidence,
                source=anchors[i].source,
            )
            if i > 0:
                jitter_after += abs(smoothed_cxs[i] - prev_sc)
            prev_sc = smoothed_cxs[i]

        denom = max(len(anchors) - 1, 1)
        logger.info(
            "[AnchorStream] %d anchors, %d shot-cut breaks, "
            "mean cx-jitter before=%.4f after=%.4f",
            len(anchors), len(cuts),
            jitter_before / denom, jitter_after / denom,
        )
    else:
        logger.info(
            "[AnchorStream] %d anchors, %d shot-cut breaks",
            len(anchors), len(cuts),
        )

    # v4: per-source breakdown so the verifier can confirm person_body
    # is actually pulling its weight on faceless-frame heavy clips. The
    # six counters cover every value AttentionAnchor.source can take.
    src_counts = {
        "active_face": 0,
        "face": 0,
        "person_body": 0,
        "last_face_decay": 0,
        "saliency_peak": 0,
        "motion_centroid": 0,
    }
    for a in anchors:
        if a.source in src_counts:
            src_counts[a.source] += 1
        else:
            src_counts.setdefault(a.source, 0)
            src_counts[a.source] += 1
    logger.info(
        "[AttentionAnchor] source breakdown: active_speaker=%d, face=%d, "
        "person_body=%d, last_face_decay=%d, saliency_peak=%d, motion_centroid=%d",
        src_counts.get("active_face", 0),
        src_counts.get("face", 0),
        src_counts.get("person_body", 0),
        src_counts.get("last_face_decay", 0),
        src_counts.get("saliency_peak", 0),
        src_counts.get("motion_centroid", 0),
    )

    return anchors
