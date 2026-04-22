"""Required-region adapter: reshapes ClipAI's existing face detection output
into the solver's input format.

Supports face-only mode (standalone solver) and full fusion mode (face +
object + saliency). The solver treats "required" regions as hard constraints
and "preferred" regions as tiebreakers.

Blueprint v2 Phase 1: the passive-face score that used to be hardcoded
at 0.55 is now read from ``ReframeConfig.importance.passive_face_weight``
so it can vary by genre. Default is still 0.55 so unchanged content
types produce bit-identical required-region output.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from backend.services.reframe_config import ReframeConfig, get_default_config

logger = logging.getLogger(__name__)


@dataclass
class RequiredRegion:
    """A region that must stay inside the crop at a given timestamp."""
    timestamp: float
    cx: float            # normalized 0-1, center of the region
    cy: float            # normalized 0-1
    half_width: float    # normalized half-width
    half_height: float   # normalized half-height
    score: float         # 1.0 for active speaker, 0.55 passive (same shot), 0.8 passive (no speaker)
    tier: str = "required"   # "required" | "preferred"
    source: str = "face"     # "face" | "object" | "saliency"
    face_slot: int = -1
    saliency_score: float = 0.0  # original saliency score (0-1), preserved from tracker
    weight: float = 1.0  # camera-path data-term weight (see build_required_regions)
    is_active_speaker: bool = False  # True when this face is the active speaker


def _iou_normalized(a, b) -> float:
    """IoU between two regions (using cx/cy/half_width/half_height)."""
    a_l, a_r = a.cx - a.half_width, a.cx + a.half_width
    a_t, a_b = a.cy - a.half_height, a.cy + a.half_height
    b_l, b_r = b.cx - b.half_width, b.cx + b.half_width
    b_t, b_b = b.cy - b.half_height, b.cy + b.half_height

    ix_l, ix_r = max(a_l, b_l), min(a_r, b_r)
    iy_t, iy_b = max(a_t, b_t), min(a_b, b_b)
    if ix_r <= ix_l or iy_b <= iy_t:
        return 0.0
    inter = (ix_r - ix_l) * (iy_b - iy_t)
    area_a = (a_r - a_l) * (a_b - a_t)
    area_b = (b_r - b_l) * (b_b - b_t)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _overlaps_any(region, existing_regions, iou_threshold=0.3) -> bool:
    """Check if region overlaps any existing region above threshold."""
    for er in existing_regions:
        if _iou_normalized(region, er) > iou_threshold:
            return True
    return False


def _merge_overlapping_features(frame_regions: list) -> list:
    """Merge saliency features that overlap face/object features.

    For each saliency-source region, compute IoU against all face and
    object regions in the same frame. If IoU > 0.3 against any, absorb
    the saliency region into that other feature by adding its score
    (capped at 1.0) and dropping the standalone saliency entry.

    This prevents a saliency bbox overlapping a face bbox from over-
    constraining the crop solver.
    """
    if not frame_regions:
        return frame_regions

    non_sal = [r for r in frame_regions if r.source != "saliency"]
    sal = [r for r in frame_regions if r.source == "saliency"]

    if not sal or not non_sal:
        return frame_regions

    kept_sal = []
    for sr in sal:
        merged = False
        for nr in non_sal:
            iou = _iou_normalized(sr, nr)
            if iou > 0.3:
                nr.score = min(1.0, nr.score + sr.score)
                merged = True
                break
        if not merged:
            kept_sal.append(sr)

    return non_sal + kept_sal


def _is_dialogue_mode(content_type) -> bool:
    """True for content types where faces are load-bearing and background
    saliency should not compete: TALKING_HEAD, CINEMATIC_DIALOGUE,
    MULTI_SPEAKER_PANEL, and ANIMATION_DIALOGUE (talking anime characters).
    """
    if content_type is None:
        return False
    # Match by string value so this works whether content_type is the enum
    # or its .value — callers pass both forms.
    val = getattr(content_type, "value", content_type)
    return val in (
        "talking_head",
        "cinematic_dialogue",
        "multi_speaker_panel",
        "animation_dialogue",
    )


def _is_animated_mode(content_type) -> bool:
    """True for content types that are animated (anime / cartoons).
    Used to tune the lip-aperture promotion threshold since anime neutral
    expressions sit around 0.05 aperture and need a higher bar.
    """
    if content_type is None:
        return False
    val = getattr(content_type, "value", content_type)
    return val in ("animation", "animation_dialogue")


def _is_gaming_mode(content_type) -> bool:
    """True for gameplay content types. Used to gate non-gaming enhancements
    so gameplay paths take the existing code path unchanged."""
    if content_type is None:
        return False
    val = getattr(content_type, "value", content_type)
    return val in ("gameplay", "gameplay_moba", "gameplay_tps", "gameplay_racing")


def _is_action_mode(content_type) -> bool:
    """Content types where motion centroid is a useful fallback anchor.

    Used to decide whether to run the attention_anchor stream for
    non-dialogue content. Unlike dialogue modes (which need the anchor to
    bridge listener/reaction gaps), action content needs the anchor so
    the motion centroid priority level 5 can carry the camera through
    long faceless stretches.
    """
    if content_type is None:
        return False
    val = getattr(content_type, "value", content_type)
    return val in (
        "animation",           # anime action
        "sports",
        "sports_basketball",
        "sports_racing",
        "music_video",
        "gameplay",
        "gameplay_moba",
        "gameplay_tps",
        "gameplay_racing",
    )


@dataclass
class _FrameSaliencyAdapter:
    """FrameSaliency-compatible wrapper built from a group of
    SaliencyRegion objects sharing a timestamp.

    `blobs` is a list of (x, y, w, h) tuples in normalized 0-1
    coordinates with top-left origin — exactly the shape
    build_required_regions expects from saliency_detector's
    FrameSaliency output.
    """
    timestamp: float
    blobs: List[tuple]
    mean_score: float


def _saliency_regions_to_frame_saliency(saliency_regions: list) -> list:
    """Group a flat list of SaliencyRegion objects (from
    saliency_tracker.track_saliency_in_frames) into per-frame
    FrameSaliency-compatible adapters.

    SaliencyRegion coordinates are percentages (0-100) centered on the
    bbox. FrameSaliency consumers expect normalized 0-1 top-left
    (x, y, w, h). We convert and keep the top-3 regions per frame by
    saliency_score to cap per-frame saliency noise.

    Returns an empty list if the input isn't the expected shape —
    callers can then treat it as a no-op fallback.
    """
    if not saliency_regions:
        return []
    # Duck-type: SaliencyRegion has .x/.y/.w/.h/.saliency_score attrs.
    first = saliency_regions[0]
    if not all(hasattr(first, a) for a in ("x", "y", "w", "h", "saliency_score")):
        return []

    by_ts: dict = {}
    for sr in saliency_regions:
        key = round(float(sr.timestamp), 2)
        by_ts.setdefault(key, []).append(sr)

    adapters: list = []
    for ts, regs in by_ts.items():
        regs.sort(key=lambda r: float(r.saliency_score), reverse=True)
        top = regs[:3]
        blobs = []
        score_sum = 0.0
        for r in top:
            # Center-origin percent → top-left-origin normalized 0-1.
            w_norm = float(r.w) / 100.0
            h_norm = float(r.h) / 100.0
            x_norm = max(0.0, float(r.x) / 100.0 - w_norm / 2.0)
            y_norm = max(0.0, float(r.y) / 100.0 - h_norm / 2.0)
            blobs.append((x_norm, y_norm, w_norm, h_norm))
            score_sum += float(r.saliency_score)
        mean_score = score_sum / max(len(top), 1)
        adapters.append(_FrameSaliencyAdapter(
            timestamp=ts, blobs=blobs, mean_score=mean_score,
        ))
    return adapters


def _saliency_weight_for_content(content_type, sal_score: float):
    """Return (weight, min_area) tuple for saliency regions, per content type.

    Table (from the spec):
        TALKING_HEAD / CINEMATIC_DIALOGUE / ANIMATION_DIALOGUE:
            0.15 + 0.25·s, max 0.40, min_area 0.04
        ANIMATION / GAMEPLAY:   0.30 + 0.60·s (legacy), min_area 0.02
        STREAM / MUSIC_VIDEO / GENERIC: 0.25 + 0.45·s, max 0.70, min_area 0.03
    """
    val = getattr(content_type, "value", content_type) if content_type is not None else None
    if val in (
        "talking_head",
        "cinematic_dialogue",
        "animation_dialogue",
        "multi_speaker_panel",
    ):
        return (min(0.40, 0.15 + 0.25 * sal_score), 0.04)
    if val in ("animation", "gameplay"):
        return (0.30 + 0.60 * sal_score, 0.02)
    # STREAM / MUSIC_VIDEO / GENERIC / None
    return (min(0.70, 0.25 + 0.45 * sal_score), 0.03)


def build_required_regions(
    frame_faces: list,
    active_speaker_events: list = None,
    frame_objects: list = None,
    frame_saliency: list = None,
    content_type=None,
    shot_cuts: list = None,
    frame_persons: list = None,
    config: Optional[ReframeConfig] = None,
) -> List[List[RequiredRegion]]:
    """Return per-frame lists of required regions.

    Face regions are always built. Object and saliency regions are added
    when their data is provided (populated conditionally by content type
    in the pipeline).

    Args:
        frame_faces: list[FrameFaces] from face_detector.py
        active_speaker_events: active speaker timeline
        frame_objects: list[list[ObjectDetection]] — one list per frame,
            or flat list indexed by timestamp
        frame_saliency: list[FrameSaliency] from saliency_detector OR a
            flat list[SaliencyRegion] from saliency_tracker (both shapes
            are accepted; SaliencyRegion lists are auto-converted to
            FrameSaliency-compatible adapters).
        content_type: ClipContentType for per-type weighting / saliency
            downrank / lead-room bias.
        shot_cuts: optional list of shot-cut timestamps. When provided
            the dense AttentionAnchor stream (dialogue modes) uses them
            to break the boxcar smoother at cut boundaries.

    Returns:
        list of lists — one inner list per frame.
    """
    def _active_event_at(timestamp: float):
        if not active_speaker_events:
            return None
        for ev in active_speaker_events:
            if ev.start <= timestamp <= ev.end and ev.slot_id >= 0:
                return ev
        return None

    # Index object detections by rounded timestamp
    obj_by_time = {}
    if frame_objects:
        for obj in frame_objects:
            key = round(getattr(obj, 'timestamp', 0), 2)
            obj_by_time.setdefault(key, []).append(obj)

    # Index saliency by rounded timestamp.
    #
    # `frame_saliency` may arrive as:
    #   (a) a flat list[SaliencyRegion] from saliency_tracker
    #       (.x/.y/.w/.h/.saliency_score in 0-100 percent), or
    #   (b) a list of FrameSaliency-like objects from saliency_detector
    #       (already have .blobs as 0-1 top-left tuples and .mean_score).
    #
    # When we receive (a) we convert to (b) so the existing .blobs
    # consumer below just works. Without this conversion the sal_by_time
    # index silently produces zero saliency regions.
    sal_by_time = {}
    _fs_used = frame_saliency
    if _fs_used:
        _first = _fs_used[0] if len(_fs_used) > 0 else None
        _has_blobs = _first is not None and hasattr(_first, "blobs")
        if not _has_blobs:
            _converted = _saliency_regions_to_frame_saliency(_fs_used)
            if _converted:
                _fs_used = _converted
                logger.info(
                    "[SaliencyFallback] generated %d frame_saliency entries "
                    "from saliency_regions",
                    len(_converted),
                )
    if _fs_used:
        for sf in _fs_used:
            sal_by_time[round(sf.timestamp, 2)] = sf

    dialogue_mode = _is_dialogue_mode(content_type)
    animated_mode = _is_animated_mode(content_type)
    action_mode = _is_action_mode(content_type)

    # Blueprint v2 Phase 1: passive-face score comes from the Stage 3
    # importance matrix. Defaults to 0.55 so generic / unknown content
    # types stay bit-identical; panel / dialogue can override.
    _cfg = (config or get_default_config()).for_content(content_type)
    _passive_face_score = float(_cfg.importance.passive_face_weight)

    # ── Attention anchor stream ──
    # For dialogue modes: bridge faces across 1.5s gaps, fall back to
    # saliency peaks. For action modes (sports / anime action / music /
    # gameplay): same dense per-frame timeline, but lower promotion
    # confidence — motion centroid becomes the load-bearing fallback when
    # no face is available. Never leave a frame without an anchor in the
    # long run.
    anchor_by_time: dict = {}
    _run_anchor_stream = frame_faces and (dialogue_mode or action_mode)
    if _run_anchor_stream:
        try:
            from backend.services.attention_anchor import build_attention_anchors
            _anchors = build_attention_anchors(
                frame_faces=frame_faces,
                active_speaker_events=active_speaker_events,
                frame_saliency=_fs_used,
                shot_cuts=shot_cuts,
                frame_persons=frame_persons,
            )
            for a in _anchors:
                anchor_by_time[round(a.timestamp, 2)] = a
        except Exception as _ae:
            logger.info("[AnchorStream] build failed (non-fatal): %s", _ae)
            anchor_by_time = {}

    # ── Lip-aperture promotion threshold ──
    # Live-action neutral faces have lip_aperture ~0-0.03. Anime neutral
    # expressions can sit around 0.05 (drawn open-mouth style), so a
    # 0.05 threshold fires on listeners and wrongly promotes them. Bump
    # to 0.09 for animated content and require ≥3 consecutive frames to
    # hold before promoting (persist state across the per-frame loop).
    lip_thr = 0.09 if animated_mode else 0.05
    min_consec = 3 if animated_mode else 1
    # identity_id → consecutive count of frames with lip_aperture > lip_thr
    _lip_streak: dict = {}

    regions_per_frame = []
    _pre_merge_total = 0
    _post_merge_total = 0
    _active_count = 0
    _passive_same_count = 0
    _passive_no_speaker_count = 0
    # v4.1 Fix 3: split the saliency-gate counters three ways so the
    # verifier can tell whether saliency is being dropped by the
    # containment check, kept on faceless frames, or kept inside face
    # bboxes. The old single counter collapsed all three cases.
    _sal_face_outside_dropped = 0    # frame had faces; saliency was outside their bboxes
    _sal_face_inside_kept = 0        # frame had faces; saliency was inside
    _sal_faceless_kept = 0           # frame had no faces; saliency passed through unfiltered
    _lead_room_applied = 0
    _lead_room_reverted = 0
    _containment_emitted = 0
    _hard_floor_boosts = 0

    for ff in frame_faces:
        active_ev = _active_event_at(ff.timestamp)
        active_slot = active_ev.slot_id if active_ev else -1
        on_screen = getattr(active_ev, "on_screen", True) if active_ev else True
        frame_regions = []
        ts_key = round(ff.timestamp, 2)

        # Is there any active speaker in this frame at all?
        frame_has_active_speaker = (active_slot >= 0 and on_screen)

        # For the tightened lip-aperture override: check whether any face in
        # this frame already has identity_id == active_slot.
        slot_ids_in_frame = {
            getattr(f, 'identity_id', -1) for f in ff.faces
            if getattr(f, 'is_human', True)
        }
        active_slot_visible_as_identity = (
            active_slot >= 0 and active_slot in slot_ids_in_frame
        )

        # ── Face regions (required tier) ──
        # Update lip streak counters for this frame (per identity) before
        # building regions, so the consecutive-frame rule can fire.
        _seen_this_frame = set()
        for face in ff.faces:
            if not getattr(face, 'is_human', True):
                continue
            fid = getattr(face, 'identity_id', -1)
            if fid < 0:
                continue
            _seen_this_frame.add(fid)
            if getattr(face, 'lip_aperture', 0) > lip_thr:
                _lip_streak[fid] = _lip_streak.get(fid, 0) + 1
            else:
                _lip_streak[fid] = 0
        # Reset streak for any identity that wasn't in this frame
        for fid in list(_lip_streak.keys()):
            if fid not in _seen_this_frame:
                _lip_streak[fid] = 0

        for face in ff.faces:
            if not getattr(face, 'is_human', True):
                continue
            slot_id = getattr(face, 'identity_id', -1)
            is_active = (slot_id >= 0 and slot_id == active_slot)
            # Lip-aperture override: only promote passive→active when no
            # other face in this frame already has identity_id == active_slot
            # (stops a laughing/yawning listener from shadow-promoting).
            # For animated content the threshold is 0.09 and the face must
            # have held open lips for ≥3 consecutive frames.
            if (not is_active
                    and slot_id >= 0
                    and getattr(face, 'lip_aperture', 0) > lip_thr
                    and _lip_streak.get(slot_id, 0) >= min_consec
                    and not active_slot_visible_as_identity):
                is_active = True

            # Off-screen speaker: treat every visible face as passive-same-shot.
            # The crop then anchors on the listener (reaction-shot framing)
            # instead of fabricating a speaker anchor.
            if active_ev is not None and not on_screen:
                is_active = False

            cx = face.nose_x / 100.0
            cy = face.nose_y / 100.0
            hw = (face.width / 100.0) / 2.0
            hh = (face.height / 100.0) / 2.0

            # Weight rules:
            #   active speaker:                          score=1.00 weight=1.40
            #   passive, same shot as active speaker:    score=_passive_face_score weight=0.45
            #   passive, no active speaker in frame:     score=0.80 weight=0.80
            # Phase 1: passive-same-shot score comes from
            # ``config.importance.passive_face_weight`` (default 0.55).
            if is_active:
                score = 1.0
                weight = 1.4
                _active_count += 1
            elif frame_has_active_speaker or (active_ev is not None and not on_screen):
                # Another face in this frame is the speaker, OR the speaker
                # is off-screen (reaction shot) — demote heavily.
                score = _passive_face_score
                weight = 0.45
                _passive_same_count += 1
            else:
                # No active speaker attributed to this frame — fall back to
                # the legacy "faces matter" behavior.
                score = 0.8
                weight = 0.8
                _passive_no_speaker_count += 1

            # ── Lead-room bias ──
            # Only the active speaker; requires a `yaw` attribute populated
            # upstream. Shifts cx by -0.05·sign(yaw) (in the direction the
            # subject is looking) and clamps to [hw, 1 - hw].
            if is_active:
                yaw = getattr(face, "yaw", None)
                if yaw is not None:
                    try:
                        yaw_val = float(yaw)
                        if abs(yaw_val) > 15.0:
                            sign = 1.0 if yaw_val > 0 else -1.0
                            cx_shifted = cx - 0.05 * sign
                            # Clamp so the region stays in-bounds.
                            lo = hw
                            hi = 1.0 - hw
                            if lo <= hi:
                                cx_shifted = max(lo, min(hi, cx_shifted))
                            # Post-condition: verify cx ± (hw * 1.15) stays
                            # inside [0, 1]. If not, revert the shift.
                            padded_hw = hw * 1.15
                            if (cx_shifted - padded_hw >= 0.0
                                    and cx_shifted + padded_hw <= 1.0):
                                cx = cx_shifted
                                _lead_room_applied += 1
                            else:
                                _lead_room_reverted += 1
                    except (TypeError, ValueError):
                        pass

            frame_regions.append(RequiredRegion(
                timestamp=ff.timestamp,
                cx=cx, cy=cy,
                half_width=hw, half_height=hh,
                score=score,
                tier="required",
                source="face",
                face_slot=slot_id,
                weight=weight,
                is_active_speaker=is_active,
            ))

            # ── Active-speaker containment region (B1) ──
            # Emit an additional containment region for the active speaker
            # so the camera solver can enforce that the face stays fully
            # inside the 9:16 crop. Skipped for gaming modes where the
            # face cam overlay has different framing needs.
            if is_active and not _is_gaming_mode(content_type):
                frame_regions.append(RequiredRegion(
                    timestamp=ff.timestamp,
                    cx=cx, cy=cy,
                    half_width=hw * 1.15,
                    half_height=hh * 1.20,
                    score=1.0,
                    tier="required",
                    source="face_containment",
                    face_slot=slot_id,
                    weight=1.6,
                    is_active_speaker=True,
                ))
                _containment_emitted += 1

        # ── Object regions: person class → required, non-face ──
        for obj in obj_by_time.get(ts_key, []):
            class_name = getattr(obj, 'class_name', '')
            if class_name != "person":
                continue
            cx = obj.x / 100.0
            cy = obj.y / 100.0
            hw = (obj.w / 100.0) / 2.0
            hh = (obj.h / 100.0) / 2.0
            candidate = RequiredRegion(
                timestamp=ff.timestamp,
                cx=cx, cy=cy,
                half_width=hw, half_height=hh,
                score=0.6,
                tier="required",
                source="object",
                weight=0.8,
            )
            if not _overlaps_any(candidate, frame_regions, 0.3):
                frame_regions.append(candidate)

        # ── Sports object regions ──
        # For sports content: non-person objects (ball, car) become
        # preferred anchors so the solver follows the action even when
        # no face is visible. In basketball, the ball is the subject; in
        # racing, the car is; in generic sports, fall back to the largest
        # player body. Scoreboard preservation is handled via cy floor.
        _ct_val = getattr(content_type, "value", content_type) if content_type else None
        if _ct_val in ("sports", "sports_basketball", "sports_racing"):
            for obj in obj_by_time.get(ts_key, []):
                class_name = getattr(obj, "class_name", "")
                # Basketball: track the ball
                if _ct_val == "sports_basketball" and class_name == "sports ball":
                    cx = float(getattr(obj, "x", 50)) / 100.0
                    cy = float(getattr(obj, "y", 50)) / 100.0
                    hw = float(getattr(obj, "w", 10)) / 200.0
                    hh = float(getattr(obj, "h", 10)) / 200.0
                    ball_region = RequiredRegion(
                        timestamp=ff.timestamp,
                        cx=cx, cy=cy,
                        half_width=max(0.05, hw),
                        half_height=max(0.05, hh),
                        # Phase 1: match the passive-face baseline so
                        # ball preferred regions don't outscore passive
                        # faces. Default still 0.55 = legacy behavior.
                        score=_passive_face_score,
                        tier="preferred",
                        source="object",
                        weight=0.7,
                    )
                    if not _overlaps_any(ball_region, frame_regions, 0.3):
                        frame_regions.append(ball_region)
                # Racing: track the largest vehicle (lead car)
                elif _ct_val == "sports_racing" and class_name in (
                    "car", "truck", "motorcycle",
                ):
                    cx = float(getattr(obj, "x", 50)) / 100.0
                    cy = float(getattr(obj, "y", 50)) / 100.0
                    # Racing: bias toward lower third (car sits below midframe)
                    cy = min(0.75, max(0.50, cy))
                    hw = float(getattr(obj, "w", 15)) / 200.0
                    hh = float(getattr(obj, "h", 15)) / 200.0
                    car_region = RequiredRegion(
                        timestamp=ff.timestamp,
                        cx=cx, cy=cy,
                        half_width=max(0.08, hw),
                        half_height=max(0.05, hh),
                        score=0.65,   # higher than ball — car IS the subject
                        tier="preferred",
                        source="object",
                        weight=0.8,
                    )
                    if not _overlaps_any(car_region, frame_regions, 0.3):
                        frame_regions.append(car_region)
                # Generic sports: large moving person body (player) when no face
                elif _ct_val == "sports" and class_name == "person":
                    if not any(r.source == "face" for r in frame_regions):
                        cx = float(getattr(obj, "x", 50)) / 100.0
                        cy = float(getattr(obj, "y", 45)) / 100.0
                        hw = float(getattr(obj, "w", 12)) / 200.0
                        hh = float(getattr(obj, "h", 20)) / 200.0
                        player_region = RequiredRegion(
                            timestamp=ff.timestamp,
                            cx=cx, cy=cy,
                            half_width=max(0.06, hw),
                            half_height=max(0.08, hh),
                            score=0.50,
                            tier="preferred",
                            source="object",
                            weight=0.6,
                        )
                        if not _overlaps_any(player_region, frame_regions, 0.3):
                            frame_regions.append(player_region)

        # ── Saliency regions: preferred tier ──
        sal_frame = sal_by_time.get(ts_key)
        if sal_frame:
            face_regions_this_frame = [
                r for r in frame_regions if r.source == "face"
            ]
            # v4.1 Fix 3: on faceless frames in dialogue mode, saliency
            # is the ONLY anchor source the camera solver has for that
            # frame (nothing else can win the AttentionAnchor priority
            # chain between person_body and motion_centroid). Dropping
            # it because "it's outside a face bbox" — when there IS no
            # face bbox — kills the saliency exactly where it's needed
            # most. Bypass min_area, containment, and the overlap gate
            # on truly faceless frames so at least one saliency region
            # survives to feed promote_preferred_to_required.
            faceless_frame = not face_regions_this_frame
            for blob in getattr(sal_frame, 'blobs', []):
                bx, by, bw, bh = blob[0], blob[1], blob[2], blob[3]
                area = bw * bh
                _sal_score = getattr(sal_frame, 'mean_score', 0.5)
                # Per-content-type weight + min-area gate. Faceless
                # dialogue frames skip the min-area filter entirely —
                # any non-zero saliency signal is load-bearing there.
                sal_weight, min_area = _saliency_weight_for_content(
                    content_type, _sal_score,
                )
                if not (dialogue_mode and faceless_frame):
                    if area < min_area:
                        continue
                cand_cx = bx + bw / 2
                cand_cy = by + bh / 2
                # Dialogue-mode containment gate: drop saliency regions
                # whose center lies outside the union of face bboxes
                # expanded by 1.5× ONLY when faces exist in the frame.
                # Faceless frames skip this entirely.
                if dialogue_mode and face_regions_this_frame:
                    inside_any = False
                    for fr_reg in face_regions_this_frame:
                        ehw = fr_reg.half_width * 1.5
                        ehh = fr_reg.half_height * 1.5
                        if (fr_reg.cx - ehw <= cand_cx <= fr_reg.cx + ehw
                                and fr_reg.cy - ehh <= cand_cy <= fr_reg.cy + ehh):
                            inside_any = True
                            break
                    if not inside_any:
                        _sal_face_outside_dropped += 1
                        continue
                    else:
                        _sal_face_inside_kept += 1
                elif dialogue_mode and faceless_frame:
                    _sal_faceless_kept += 1
                candidate = RequiredRegion(
                    timestamp=ff.timestamp,
                    cx=cand_cx, cy=cand_cy,
                    half_width=bw / 2, half_height=bh / 2,
                    score=_sal_score * 0.5,
                    tier="preferred",
                    source="saliency",
                    saliency_score=_sal_score,
                    weight=sal_weight,
                )
                # Overlap gate: on faceless frames skip the overlap
                # check too (there are no faces to overlap with, and we
                # want multiple saliency regions per frame so the
                # anchor chain has options).
                if dialogue_mode and faceless_frame:
                    frame_regions.append(candidate)
                elif not _overlaps_any(candidate, frame_regions, 0.5):
                    frame_regions.append(candidate)

        # ── Hard floor: active speaker weight >= 1.5 × max other face weight ──
        # Guarantees the L1 camera solver's data term cannot pick a non-
        # speaking face over a speaking one, regardless of relative bbox
        # sizes. Only applies within the frame's face regions — saliency
        # and object regions are unaffected.
        face_regs = [r for r in frame_regions if r.source == "face"]
        active_regs = [r for r in face_regs if r.is_active_speaker]
        passive_regs = [r for r in face_regs if not r.is_active_speaker]
        if active_regs and passive_regs:
            max_other = max(r.weight for r in passive_regs)
            floor = 1.5 * max_other
            for ar in active_regs:
                if ar.weight < floor:
                    ar.weight = floor
                    _hard_floor_boosts += 1

        # ── Attention anchor fallback (dialogue + action modes) ──
        # If no face RequiredRegion survived for this frame, promote the
        # dense AttentionAnchor to a required region so the camera
        # solver always has something to lock onto. Without this the
        # crop drifts onto background motion during faceless action
        # beats (the primary symptom from the K S01E12 sanity run).
        #
        # Dialogue mode: min_conf=0.3 (bridge faces aggressively).
        # Action mode: min_conf=0.5 (only promote strong motion/saliency
        # centroids, and cap the region weight lower so the solver
        # treats it as a soft hint rather than a locked target).
        if _run_anchor_stream and not any(r.source == "face" for r in frame_regions):
            anchor = anchor_by_time.get(ts_key)
            _min_anchor_conf = 0.3 if dialogue_mode else 0.5
            if anchor is not None and float(getattr(anchor, "confidence", 0.0)) >= _min_anchor_conf:
                _anchor_conf = float(getattr(anchor, "confidence", 0.0))
                if dialogue_mode:
                    _score = max(0.3, min(0.9, 0.3 + 0.5 * _anchor_conf))
                    _weight = max(0.3, min(0.9, 0.3 + 0.5 * _anchor_conf))
                else:
                    # Action mode: cap score and weight lower so faces in
                    # adjacent frames still dominate the solver.
                    _score = max(0.3, min(0.7, _anchor_conf))
                    _weight = max(0.3, min(0.7, _anchor_conf))
                frame_regions.append(RequiredRegion(
                    timestamp=ff.timestamp,
                    cx=anchor.cx, cy=anchor.cy,
                    half_width=anchor.half_width, half_height=anchor.half_height,
                    score=_score,
                    tier="required",
                    source="saliency" if anchor.source in ("saliency_peak",) else "face",
                    weight=_weight,
                    is_active_speaker=False,
                ))

        _pre_merge_total += len(frame_regions)
        frame_regions = _merge_overlapping_features(frame_regions)
        _post_merge_total += len(frame_regions)
        regions_per_frame.append(frame_regions)

    logger.info(
        "[SpeakerV2] weights: active=%d passive_same_shot=%d passive_no_speaker=%d",
        _active_count, _passive_same_count, _passive_no_speaker_count,
    )
    if _hard_floor_boosts > 0:
        logger.info(
            "[SpeakerV2] hard-floor boosted %d active-speaker weights "
            "(>= 1.5x passive in-frame max)",
            _hard_floor_boosts,
        )
    # v4.1 Fix 3: three-way split log. N=kept on faceless frames,
    # M=dropped because outside face bboxes, K=kept inside face bboxes.
    # Lets the verifier tell at a glance whether the gate is working
    # or whether the SaliencyTracker just isn't emitting much.
    if dialogue_mode and (
        _sal_faceless_kept
        or _sal_face_outside_dropped
        or _sal_face_inside_kept
    ):
        logger.info(
            "[SaliencyV2] dialogue-mode gate: %d faceless-frame saliency kept, "
            "%d face-frame saliency dropped (outside face bboxes), "
            "%d face-frame saliency kept (inside face bboxes)",
            _sal_faceless_kept,
            _sal_face_outside_dropped,
            _sal_face_inside_kept,
        )
    if _lead_room_applied > 0 or _lead_room_reverted > 0:
        logger.info(
            "[SpeakerV2] lead-room bias applied to %d active-speaker regions",
            _lead_room_applied,
        )
    if _containment_emitted > 0 or _lead_room_reverted > 0:
        logger.info(
            "[SpeakerV2] active_containment: regions_emitted=%d, lead_room_reverted=%d",
            _containment_emitted, _lead_room_reverted,
        )

    if _pre_merge_total != _post_merge_total:
        logger.info("[SaliencyParity] feature merge: %d → %d regions (-%d absorbed)",
                    _pre_merge_total, _post_merge_total,
                    _pre_merge_total - _post_merge_total)

    total_regions = sum(len(r) for r in regions_per_frame)
    frames_with = sum(1 for r in regions_per_frame if r)
    n_face = sum(1 for r in regions_per_frame for rr in r if rr.source == "face")
    n_obj = sum(1 for r in regions_per_frame for rr in r if rr.source == "object")
    n_sal = sum(1 for r in regions_per_frame for rr in r if rr.source == "saliency")
    logger.info("RequiredRegions: %d frames, %d with data, %d total "
                "(face=%d obj=%d sal=%d)",
                len(regions_per_frame), frames_with, total_regions,
                n_face, n_obj, n_sal)
    return regions_per_frame


def promote_preferred_to_required(
    regions_per_frame: List[List[RequiredRegion]],
    content_type=None,
) -> None:
    """For frames with no required regions, promote the highest-scoring
    preferred region to required.

    This is what makes anime and gameplay work — saliency becomes
    load-bearing exactly when faces are absent.

    For dialogue modes (TALKING_HEAD / CINEMATIC_DIALOGUE): only promote
    high-confidence (score >= 0.6) preferred regions. This prevents
    fabricating a saliency anchor during a speaker pause while still
    giving the solver a target for truly faceless frames (cutaway b-roll,
    over-the-shoulder inserts) where the attention anchor stream didn't
    already bridge the gap.
    """
    promoted = 0
    dialogue_mode = _is_dialogue_mode(content_type)
    for frame_regions in regions_per_frame:
        has_required = any(r.tier == "required" for r in frame_regions)
        if has_required or not frame_regions:
            continue
        preferred = [r for r in frame_regions if r.tier == "preferred"]
        if not preferred:
            continue

        if dialogue_mode:
            # In dialogue mode: only promote high-confidence saliency so we
            # don't anchor on background motion during a brief speaker pause.
            high_conf = [r for r in preferred if r.score >= 0.6]
            if not high_conf:
                continue
            best = max(high_conf, key=lambda r: r.score)
        else:
            best = max(preferred, key=lambda r: r.score)

        best.tier = "required"
        promoted += 1

    if promoted > 0:
        logger.info("RequiredRegions: promoted %d preferred→required (no-face frames)",
                    promoted)
