"""Segment-based reframe timeline for human-like vertical crop.

Replaces per-scene / per-keyframe subject_x decisions with a segment-based
reframe timeline that mimics how a human editor cuts vertical reframes.
Eliminates jitter by making reframes *motivated editorial events* rather
than face-detection outputs.

Content-aware mode (gated by ContentProfile) applies different editorial
strategies per content type: narrative, podcast, gaming, vlog, sports,
music_video, anime.

Two-pass architecture:
  Pass 1 (measurement): Build per-second signal arrays
  Pass 2 (planning): Walk arrays with look-ahead to emit segments

Adaptive pacing: Timing (min_hold, anticipation) derived from
LocalPacingEstimator, not hardcoded config values.

Confidence-gated fallback ladder: Low confidence NEVER produces a center
crop. Instead falls back to blur_fill or wide_master showing the full source.

Consumes shot cuts, face registry, active speaker events, dense faces,
transcript segments, and speaker-to-slot mapping. Produces a list of
ReframeSegment objects that the pipeline emits as scenes.

Interval convention: all segments use half-open [start, end) semantics.
"""

import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Feature flags ──
USE_REFRAME_SEGMENTER = os.environ.get("USE_REFRAME_SEGMENTER", "true").lower() in ("true", "1", "yes")
# v2 Phase 11: default ON. The content-aware branches (panel hold, narrative,
# gaming, anime) were shipped dormant because the env var defaulted to "false",
# so classify_content's output never reached the segmenter and every panel
# clip ran as content_type=unknown. Parity bench (Phase 10) passed with these
# branches enabled; there is no reason to keep them behind a flag.
USE_CONTENT_AWARE_REFRAME = os.environ.get("USE_CONTENT_AWARE_REFRAME", "true").lower() in ("true", "1", "yes")
USE_INTENT_TRACKING = os.environ.get("USE_INTENT_TRACKING", "false").lower() in ("true", "1", "yes")

# ── Default tunables (used when no content profile is provided) ──
MIN_HOLD_SECONDS = 0.12  # 3 frames @ 24fps, 4 @ 30fps — absolute floor
ANTICIPATION_MS = 200
SPEAKER_CONFIDENCE_THRESHOLD = 0.6
SPEAKER_COVERAGE_THRESHOLD = 0.60
DENSE_DOMINANCE_THRESHOLD = 0.70
MULTI_SPEAKER_THRESHOLD = 0.20
WIDE_MASTER_X = 50
SUBJECT_Y_DEFAULT = 40
EASE_SHOT_CUT_MS = 0
EASE_SPEAKER_TURN_MS = 200
EASE_SUBJECT_WALK_MS = 250


def intra_shot_face_motion_pct(
    dense_faces: list,
    slot_id,
    start: float,
    end: float,
) -> float:
    """Return ``max(nose_x) - min(nose_x)`` across dense faces in a shot.

    Phase 5 — used by the segmenter to decide whether a `stationary`
    L1 result should be overridden to `tracking`. Returns 0.0 when
    there are no dense face samples in the time range.

    Args:
        dense_faces: list[FrameFaces] from dense face detection.
        slot_id: the active slot id, or None to consider any face.
        start, end: shot time range in seconds.

    Returns:
        The face motion within the shot, in *percent of frame width*.
    """
    if not dense_faces or end <= start:
        return 0.0
    xs: list[float] = []
    for df in dense_faces:
        ts = float(getattr(df, "timestamp", -1.0))
        if ts < start or ts >= end:
            continue
        for face in getattr(df, "faces", []) or []:
            sid = getattr(face, "identity_id", -1)
            if slot_id is not None and sid != slot_id:
                continue
            nx = getattr(face, "nose_x", None)
            if nx is None:
                continue
            xs.append(float(nx))
            break
    if len(xs) < 2:
        return 0.0
    return max(xs) - min(xs)


def should_force_tracking_for_motion(
    dense_faces: list,
    slot_id,
    start: float,
    end: float,
    *,
    motion_pct_threshold: float = 6.0,
    min_shot_seconds: float = 1.0,
) -> bool:
    """True when intra-shot face motion exceeds the override threshold.

    Phase 5 — when a shot's L1 solver result is `stationary` but the
    underlying face panned across the frame by more than
    ``motion_pct_threshold`` percent of frame width AND the shot is
    at least ``min_shot_seconds`` long, override to `tracking` so
    the L1 solver's per-frame anchors actually drive the camera.
    """
    if (end - start) < min_shot_seconds:
        return False
    return intra_shot_face_motion_pct(dense_faces, slot_id, start, end) > motion_pct_threshold


def _is_formation_frame(
    frame_faces,
    min_faces: int = 3,
    min_span_pct: float = 55.0,
) -> bool:
    """True when ``min_faces`` or more faces span ``>= min_span_pct`` of
    frame width.

    Signals group choreography (dance formation, ensemble cast) where a
    tight single-face crop loses the composition entirely. Human editors
    hold wide or pan slowly across the formation in these frames. Used
    by the music-video segment branch to force ``wide_master`` on
    formation-dominant windows.
    """
    faces = getattr(frame_faces, "faces", []) or []
    if len(faces) < min_faces:
        return False
    xs = [float(getattr(f, "nose_x", 50)) for f in faces]
    if not xs:
        return False
    return (max(xs) - min(xs)) >= min_span_pct


@dataclass
class ReframeSegment:
    start: float              # seconds
    end: float                # seconds, end - start >= MIN_HOLD
    subject_x: float          # Source pixel x-coordinate (pixel-precise crop center)
    subject_y: float          # Source pixel y-coordinate (pixel-precise crop center)
    layout: str               # "single" | "split" | "triple" | "wide_master" | "blur_fill" | "stacked_gameplay" | "grid"
    active_slot: Optional[int]  # which face registry slot, or None for wide
    confidence: float         # 0..1
    reason: str               # "speaker_turn" | "shot_cut" | "subject_walk" | "wide_fallback" | "hold" | "action_sequence" | "split_overlap"
    ease_in_ms: int           # 0 for snap, 400-600 for motivated in-shot move
    # ── Content-aware fields (populated when content profile is provided) ──
    strategy: str = "stationary"  # ReframeStrategy value
    content_type: str = "unknown"
    lead_room_direction: Optional[str] = None  # "left" | "right" | None
    motion_path: Optional[list] = None  # [(t, x, y), ...] for tracking/panning
    hard_constraints: Optional[list] = None  # [(x, y, w, h), ...] HUD rects
    subject_source: str = ""  # which code path produced subject_x
    fallback_reason: Optional[str] = None  # why a fallback was applied
    confidence_breakdown: Optional[dict] = None  # {face_in_crop, speaker_agree, stability, transcript}
    # ── Phase 4 (gaming): per-segment layout mode for gameplay clips ──
    # One of "fullscreen" | "blurfill" | "composite" | "wide_zoom".
    # Only populated when ``_content_profile.is_gaming`` is True; the
    # render plan + clip exporter read this to pick the FFmpeg
    # filter graph per segment instead of one mode per clip.
    gaming_layout_mode: Optional[str] = None


def build_reframe_segments(
    shot_cuts: list[float],
    face_registry,
    active_speaker_events: list,
    dense_faces: list,
    transcript_segments: list,
    speaker_to_slot: dict[str, int],
    video_duration: float,
    source_width: int = 1920,
    source_height: int = 1080,
    job_id: str = "",
    content_profile=None,
    persistent_regions=None,
    pacing_estimator=None,
    interpolated_timeline=None,
    frame_saliency: list = None,
    music_beat_grid=None,
    anime_anchors: list = None,
    gameplay_motion_centroids: list = None,
    audio_events: list = None,
    diarization_segments: list = None,
    cluster_to_slot: dict = None,
    debug_out: Optional[dict] = None,
) -> list[ReframeSegment]:
    """Build a segment-based reframe timeline.

    Args:
        shot_cuts: Scene-cut timestamps (hard boundaries).
        face_registry: FaceRegistry with N slots and their x-positions.
        active_speaker_events: SpeakerEvent list from active_speaker.py.
        dense_faces: FrameFaces list from dense face detection.
        transcript_segments: TranscriptSegment list with speaker_id assigned.
        speaker_to_slot: Mapping from speaker label to face slot id.
        video_duration: Total video duration in seconds.
        job_id: For logging correlation.
        content_profile: ContentProfile from content_classifier (optional).
        persistent_regions: RegionDetectionResult from persistent_region_detector (optional).
        pacing_estimator: LocalPacingEstimator (optional). When provided,
            replaces hardcoded min_hold and anticipation with data-driven values.

    Returns:
        List of ReframeSegment covering [0, video_duration].
    """
    _log = lambda msg, *a: logger.info("[%s] ReframeSegmenter: " + msg, job_id, *a)

    # ── Load content-type config ──
    ct = "unknown"
    cfg = None
    _tuning = None
    if content_profile and USE_CONTENT_AWARE_REFRAME:
        base_ct = getattr(content_profile, 'content_type', 'unknown') or 'unknown'
        # v2 Phase 11: a multi-speaker panel needs distinct editorial
        # routing (tighter holds, no in-shot tracking, per-turn snaps).
        # Previously the classifier would set content_type="podcast" and
        # is_multi_speaker_panel=True, but the segmenter only looked at
        # content_type so panels ran with vlog/podcast pacing. Promote
        # the flag to its own ct key here so get_config("multi_speaker_panel")
        # returns the panel preset.
        if getattr(content_profile, 'is_multi_speaker_panel', False):
            ct = "multi_speaker_panel"
        elif (
            getattr(content_profile, 'is_animated', False)
            and base_ct in ("narrative", "unknown", "")
        ):
            # Animated content that the video-level classifier promoted
            # to "narrative" for routing purposes should use the anime
            # preset — not the live-action narrative preset. Anime has
            # fast intent tracking, blur_fill fallback, and no
            # wide_master_on_multi_face (speaker should stay tracked).
            ct = "anime"
        else:
            ct = base_ct
        try:
            from backend.services.content_type_config import get_config, get_tuning_from_profile
            cfg = get_config(ct)
            _tuning = get_tuning_from_profile(content_profile)
        except ImportError:
            cfg = None
    if _tuning is None:
        try:
            from backend.services.content_type_config import get_tuning_from_profile
            _tuning = get_tuning_from_profile(None)
        except ImportError:
            pass

    # Use content-type-specific tunables or fall back to module defaults
    # NOTE: min_hold_seconds and anticipation_ms are NO LONGER from config.
    # They come from the pacing_estimator (adaptive) or module defaults (legacy).
    _has_pacing = pacing_estimator is not None
    _min_hold = MIN_HOLD_SECONDS  # Default; overridden per-segment by pacing estimator
    _anticipation_ms = ANTICIPATION_MS  # Default; overridden per-segment by pacing estimator
    _speaker_conf_thresh = cfg.get("speaker_confidence_threshold", SPEAKER_CONFIDENCE_THRESHOLD) if cfg else SPEAKER_CONFIDENCE_THRESHOLD
    _speaker_cov_thresh = cfg.get("speaker_coverage_threshold", SPEAKER_COVERAGE_THRESHOLD) if cfg else SPEAKER_COVERAGE_THRESHOLD
    _ease_speaker_ms = cfg.get("ease_speaker_turn_ms", EASE_SPEAKER_TURN_MS) if cfg else EASE_SPEAKER_TURN_MS
    _ease_cut_ms = cfg.get("ease_shot_cut_ms", EASE_SHOT_CUT_MS) if cfg else EASE_SHOT_CUT_MS
    _ease_walk_ms = cfg.get("ease_subject_walk_ms", EASE_SUBJECT_WALK_MS) if cfg else EASE_SUBJECT_WALK_MS
    _wide_on_multi = cfg.get("wide_master_on_multi_face", True) if cfg else True
    _use_split_on_overlap = cfg.get("use_split_screen_on_overlap", False) if cfg else False
    _apply_lead_room = cfg.get("apply_lead_room", False) if cfg else False
    _allow_motion_tracking = cfg.get("allow_motion_tracking", False) if cfg else False

    # ── Music video sub-type behavior branches ──
    # The music_video parent type routes all performance content by
    # default. Sub-types refine the editorial behavior:
    #   narrative: cinematic-dialogue feel (lead room + wide on multi)
    #   lyric:     minimal motion, long holds (override pacing floor)
    #   performance: snappy, wide on formation (formation detector does
    #                the wide work; keep lead room off)
    _music_subtype = (
        getattr(content_profile, "music_subtype", None)
        if content_profile else None
    )
    if ct == "music_video" and _music_subtype:
        if _music_subtype == "narrative":
            _apply_lead_room = True
            _wide_on_multi = True
        elif _music_subtype == "lyric":
            _apply_lead_room = False
            _wide_on_multi = False
            # Hold floor ≥ 2.5s: lyric videos should almost never cut
            _min_hold = max(_min_hold, 2.5)
        elif _music_subtype == "performance":
            _apply_lead_room = False
            _wide_on_multi = True

    # ── Initialize subject confidence estimator ──
    _confidence_estimator = None
    try:
        from backend.services.subject_confidence import SubjectConfidenceEstimator, get_fallback_strategy, face_in_proposed_crop
        _confidence_estimator = SubjectConfidenceEstimator(
            face_registry=face_registry,
            dense_faces=dense_faces,
            active_speaker_events=active_speaker_events,
            transcript_segments=transcript_segments,
            speaker_to_slot=speaker_to_slot,
            source_width=source_width,
            source_height=source_height,
            # Fix 2: frame_saliency feeds the salient-fallback cascade
            # so low-confidence segments can land on a real subject
            # (via saliency_peak / face_centroid / motion_proxy) instead
            # of hardcoded_center.
            frame_saliency=frame_saliency,
            persistent_regions=persistent_regions,
            # Gap 5b: diarization as a second independent vote. Phase C
            # adds ``content_type=ct`` here for the per-type weight
            # table lookup.
            diarization_segments=diarization_segments,
            cluster_to_slot=cluster_to_slot,
            content_type=ct,
        )
    except Exception as e:
        logger.warning("[%s] SubjectConfidenceEstimator init failed: %s", job_id, e)

    if _has_pacing:
        _log("using adaptive pacing from LocalPacingEstimator")

    n_scenes = len(dense_faces) if dense_faces else 0
    n_cuts = len(shot_cuts)
    n_as = len(active_speaker_events) if active_speaker_events else 0
    n_ts = len(transcript_segments) if transcript_segments else 0
    _log("input=%d scenes, %d shot cuts, %d AS events, %d transcript segs, content_type=%s",
         n_scenes, n_cuts, n_as, n_ts, ct)

    # ── Stage 1: Candidate boundary collection ──
    boundaries = set()
    boundary_reasons = {}  # time -> reason

    # 1a. Shot cuts are HARD boundaries
    for t in shot_cuts:
        if 0 < t < video_duration:
            boundaries.add(t)
            boundary_reasons[t] = "shot_cut"

    # 1b. Speaker turns in transcript
    if transcript_segments:
        prev_speaker = None
        for seg in transcript_segments:
            speaker = getattr(seg, 'speaker', None) or ''
            conf = getattr(seg, 'confidence', None)
            if conf is None:
                conf = 1.0
            if prev_speaker is not None and speaker != prev_speaker and conf > _speaker_conf_thresh:
                t = seg.start
                if 0 < t < video_duration:
                    # Don't overwrite shot_cut with speaker_turn
                    if t not in boundary_reasons or boundary_reasons[t] != "shot_cut":
                        boundaries.add(t)
                        boundary_reasons[t] = "speaker_turn"
            prev_speaker = speaker

    # 1c. Active-speaker slot changes with hold > min_hold (adaptive)
    if active_speaker_events and len(active_speaker_events) >= 2:
        for i in range(1, len(active_speaker_events)):
            prev_ev = active_speaker_events[i - 1]
            curr_ev = active_speaker_events[i]
            if curr_ev.slot_id != prev_ev.slot_id:
                # Check if the new slot holds long enough
                hold_dur = curr_ev.end - curr_ev.start
                local_min_hold = pacing_estimator.min_hold_at(curr_ev.start) if _has_pacing else _min_hold
                if hold_dur >= local_min_hold:
                    t = curr_ev.start
                    if 0 < t < video_duration and t not in boundary_reasons:
                        boundaries.add(t)
                        boundary_reasons[t] = "speaker_turn"

    # ── C4: Suppress tail-region splits ──
    # Audit (Phase C): do not insert a non-shot-cut boundary within the
    # last 300ms of a shot when content is not gaming. A speaker-turn
    # boundary here would split the tail of a shot into a tiny segment,
    # causing the camera solver's tail-lock constraint to lose effect.
    _tail_suppressed = 0
    _is_gaming_ct = ct in ("gameplay", "gameplay_fps", "gameplay_rts",
                           "gameplay_moba", "stacked_gameplay")
    if not _is_gaming_ct and shot_cuts:
        _suppress = set()
        for sc in shot_cuts:
            for t in boundaries:
                if t == sc:
                    continue  # don't suppress the shot cut itself
                if boundary_reasons.get(t) == "shot_cut":
                    continue  # never suppress hard shot-cut boundaries
                # Suppress non-shot-cut boundaries within 300ms before a shot cut
                if 0 < (sc - t) <= 0.30:
                    _suppress.add(t)
        for t in _suppress:
            boundaries.discard(t)
            boundary_reasons.pop(t, None)
            _tail_suppressed += 1
        if _tail_suppressed > 0:
            _log("C4: suppressed %d tail-region boundary(ies) within 300ms of shot cuts",
                 _tail_suppressed)

    # Add video start and end
    boundaries.add(0.0)
    boundaries.add(video_duration)

    sorted_boundaries = sorted(boundaries)
    _log("built %d candidate boundaries → %d raw segments",
         len(sorted_boundaries), len(sorted_boundaries) - 1)

    # ── Stage 2: Segment construction ──
    raw_segments = []
    for i in range(len(sorted_boundaries) - 1):
        seg_start = sorted_boundaries[i]
        seg_end = sorted_boundaries[i + 1]
        reason = boundary_reasons.get(seg_start, "hold")

        active_slot, confidence, layout, subject_source = _resolve_slot_for_interval(
            seg_start, seg_end,
            transcript_segments, speaker_to_slot,
            active_speaker_events,
            dense_faces, face_registry,
        )

        subject_x = _slot_to_x(active_slot, face_registry, source_width)

        # ── Music video: formation detection ──
        # For music videos with group choreography (3+ faces spanning
        # >55% of the frame), force a wide master crop instead of a
        # tight single-face follow. Human editors hold wide or pan
        # slowly across formation shots.
        _formation_override = False
        if ct == "music_video" and dense_faces:
            _window_frames = [
                df for df in dense_faces
                if seg_start <= df.timestamp < seg_end
            ]
            if _window_frames:
                _formation_count = sum(
                    1 for df in _window_frames if _is_formation_frame(df)
                )
                if _formation_count / len(_window_frames) >= 0.5:
                    _formation_override = True

        if _formation_override:
            raw_segments.append(ReframeSegment(
                start=seg_start,
                end=seg_end,
                subject_x=source_width * 0.5,
                subject_y=SUBJECT_Y_DEFAULT / 100.0 * source_height,
                layout="wide_master",
                active_slot=None,
                confidence=0.85,
                reason="formation_shot",
                ease_in_ms=0,
                strategy="wide_master",
                content_type=ct,
                subject_source="formation",
            ))
        else:
            raw_segments.append(ReframeSegment(
                start=seg_start,
                end=seg_end,
                subject_x=subject_x,
                subject_y=SUBJECT_Y_DEFAULT / 100.0 * source_height,
                layout=layout,
                active_slot=active_slot,
                confidence=confidence,
                reason=reason,
                ease_in_ms=0,
                strategy="stationary",
                content_type=ct,
                subject_source=subject_source,
            ))

    # ── Confidence-gated merge of short segments ──
    # A short segment is merged ONLY if:
    #   - confidence < 0.6 (low-confidence blip), OR
    #   - it has no shot-cut boundary on either side
    # High-confidence short segments at shot-cut boundaries are preserved.
    _shot_cut_set = set(shot_cuts)

    def _is_at_shot_cut(t: float) -> bool:
        return any(abs(t - sc) < 0.05 for sc in _shot_cut_set)

    merged_count = 0
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(raw_segments):
            seg = raw_segments[i]
            dur = seg.end - seg.start
            # Use adaptive min_hold at the segment's midpoint, clamped to 0.12 floor
            seg_mid = (seg.start + seg.end) / 2.0
            local_min_hold = pacing_estimator.min_hold_at(seg_mid) if _has_pacing else _min_hold
            local_min_hold = max(local_min_hold, MIN_HOLD_SECONDS)  # enforce 0.12s floor
            if dur < local_min_hold and len(raw_segments) > 1:
                # Gate: preserve high-confidence segments at shot boundaries
                at_shot_boundary = (
                    _is_at_shot_cut(seg.start) or _is_at_shot_cut(seg.end)
                    or seg.reason == "shot_cut"
                )
                if seg.confidence >= 0.6 and at_shot_boundary:
                    i += 1
                    continue  # preserve: high confidence + shot-cut anchor
                # Gate: never merge across shot cuts
                merged_into = _merge_short_segment(raw_segments, i, _shot_cut_set)
                if merged_into is not None:
                    merged_count += 1
                    changed = True
                    continue  # restart from same index
            i += 1

    if merged_count > 0:
        _log("merged %d short segments (confidence-gated, min_hold=%.2f)", merged_count, MIN_HOLD_SECONDS)

    # ── Stage 3: Multi-speaker detection (content-aware) ──
    wide_count = 0
    split_count = 0
    if dense_faces and face_registry and len(face_registry.slots) >= 2:
        for seg in raw_segments:
            active_slot_count = _count_active_slots(seg.start, seg.end, dense_faces)
            if active_slot_count >= 3 and _wide_on_multi:
                # Don't override if the segment already has a confident single-speaker
                # assignment (active speaker or transcript speaker picked one)
                if seg.active_slot is not None and seg.confidence >= 0.6:
                    continue  # Already tracking a specific speaker, keep it
                if _use_split_on_overlap and active_slot_count <= 4:
                    seg.layout = "grid"
                    seg.strategy = "grid"
                    seg.reason = "wide_fallback"
                    seg.confidence = 0.8
                    split_count += 1
                else:
                    seg.layout = "wide_master"
                    seg.strategy = "wide_master"
                    seg.active_slot = None
                    seg.subject_x = source_width / 2.0
                    seg.reason = "wide_fallback"
                    seg.confidence = 0.8
                    wide_count += 1
            elif active_slot_count == 2 and _use_split_on_overlap:
                # Check for speaker overlap (both active simultaneously)
                overlap_dur = _check_speaker_overlap(
                    seg.start, seg.end, active_speaker_events)
                overlap_thresh = cfg.get("overlap_threshold_seconds", 1.0) if cfg else 1.0
                if overlap_dur >= overlap_thresh:
                    seg.layout = "split"
                    seg.strategy = "split_screen"
                    seg.reason = "split_overlap"
                    seg.confidence = 0.8
                    split_count += 1
            elif active_slot_count >= 3 and _is_multi_speaker_crowd(seg.start, seg.end, dense_faces, face_registry):
                seg.layout = "wide_master"
                seg.strategy = "wide_master"
                seg.active_slot = None
                seg.subject_x = source_width / 2.0
                seg.reason = "wide_fallback"
                seg.confidence = 0.8
                wide_count += 1

    if wide_count > 0:
        _log("%d segments forced to WIDE_MASTER (multi-speaker)", wide_count)
    if split_count > 0:
        _log("%d segments set to SPLIT_SCREEN/GRID (overlap/multi)", split_count)

    # ── Stage 3b: Narrative action-sequence detection ──
    # For narrative content, detect clusters of rapid shot cuts and force WIDE_MASTER
    action_wide_count = 0
    if ct == "narrative" and cfg and shot_cuts:
        action_cut_rate = cfg.get("action_cut_rate_threshold", 6.0)
        action_window = cfg.get("action_window_seconds", 10.0)
        for seg in raw_segments:
            if seg.layout in ("wide_master", "split", "grid"):
                continue
            cuts_in_window = sum(
                1 for sc in shot_cuts
                if seg.start <= sc <= seg.end
            )
            seg_dur = seg.end - seg.start
            if seg_dur > 0 and cuts_in_window / (seg_dur / action_window) >= action_cut_rate:
                seg.layout = "wide_master"
                seg.strategy = "wide_master"
                seg.active_slot = None
                seg.subject_x = source_width / 2.0
                seg.reason = "action_sequence"
                action_wide_count += 1

    if action_wide_count > 0:
        _log("%d segments forced to WIDE_MASTER (action sequence, >%.0f cuts/%.0fs)",
             action_wide_count, cfg.get("action_cut_rate_threshold", 6.0),
             cfg.get("action_window_seconds", 10.0))

    # ── Stage 3c: Gaming layout selection ──
    gaming_stacked_count = 0
    if ct == "gaming" and persistent_regions:
        has_facecam = getattr(persistent_regions, 'has_facecam', False)
        has_hud = getattr(persistent_regions, 'has_hud', False)
        if has_facecam and cfg and cfg.get("prefer_stacked_gameplay"):
            for seg in raw_segments:
                seg.layout = "stacked_gameplay"
                seg.strategy = "stacked_gameplay"
                gaming_stacked_count += 1
        elif has_hud:
            for seg in raw_segments:
                seg.layout = "blur_fill"
                seg.strategy = "blur_fill"
        if gaming_stacked_count > 0:
            _log("%d segments set to STACKED_GAMEPLAY", gaming_stacked_count)

    # ── Stage 3d: Confidence-gated fallback ladder ──
    # Evaluate each segment's confidence and apply fallbacks.
    # Low confidence NEVER produces a center crop — falls back to blur_fill or wide_master.
    fallback_count = 0
    last_confident_x = None
    last_confident_slot = None
    if _confidence_estimator:
        for seg in raw_segments:
            if seg.layout in ("wide_master", "split", "grid", "blur_fill", "stacked_gameplay"):
                # Already a multi-speaker or special layout — skip
                if seg.confidence >= 0.70:
                    last_confident_x = seg.subject_x
                    last_confident_slot = seg.active_slot
                continue

            # Confidence estimator works in 0-100 space; convert at boundary
            _sx_pct = int(round(seg.subject_x / source_width * 100.0))
            conf, conf_reason = _confidence_estimator.evaluate(
                seg.start, seg.end, seg.active_slot, _sx_pct,
            )
            seg.confidence = conf

            # Fix 5: per-frame robustness safeguard. Before committing
            # a confidence-driven fallback, check whether a tracked
            # face was actually inside the proposed crop for ≥40 %% of
            # the segment's dense frames. If yes, the aggregated
            # confidence dropped below the threshold because of
            # transient occlusions / profile shots / detection jitter
            # — not because the subject left the crop. Restore the
            # candidate and skip the fallback.
            from backend.services.subject_confidence import CONFIDENCE_HIGH as _CONF_HI
            if conf < _CONF_HI and _confidence_estimator is not None:
                try:
                    pass_rate = _confidence_estimator.per_frame_in_crop_pass_rate(
                        seg.start, seg.end, _sx_pct,
                    )
                    if pass_rate >= 0.40:
                        seg.confidence = max(conf, _CONF_HI)
                        last_confident_x = seg.subject_x
                        last_confident_slot = seg.active_slot
                        _log(
                            "segment %.1f-%.1fs: per-frame safeguard rescued "
                            "(pass_rate=%.0f%% ≥ 40%%), keeping candidate x=%d",
                            seg.start, seg.end, pass_rate * 100, _sx_pct,
                        )
                        continue
                except Exception as _pfe:
                    logger.debug(
                        "[%s] per-frame safeguard failed %.1f-%.1f: %s",
                        job_id, seg.start, seg.end, _pfe,
                    )

            if conf >= 0.70:
                # High confidence — keep as-is
                last_confident_x = seg.subject_x
                last_confident_slot = seg.active_slot
                continue

            # Apply fallback ladder.
            #
            # Fix 2+3: run the salient-fallback cascade BEFORE calling
            # get_fallback_strategy so the strategy has a real candidate
            # center to promote instead of defaulting to 50. The cascade
            # picks the best non-letterbox tier for the content type
            # (active_speaker_slot → prev_crop_continuity → saliency_peak
            # → face_centroid → motion_proxy → prev_anywhere →
            # hardcoded_center).
            try:
                from backend.services.subject_confidence import (
                    get_fallback_strategy,
                    resolve_fallback_center,
                )
                _last_x_pct = (
                    int(round(last_confident_x / source_width * 100.0))
                    if last_confident_x is not None else None
                )
                _cand_x_pct = int(round(seg.subject_x / source_width * 100.0))

                # Build a "last_seg_like" object carrying the fields the
                # resolver reads: end, subject_x (as 0-100 pct), subject_y
                # (0-100 pct), confidence. We don't have a real prev
                # ReframeSegment in scope for the very first iteration,
                # so fall back to None there.
                _last_seg_like = None
                if last_confident_x is not None:
                    class _LS:
                        pass
                    _last_seg_like = _LS()
                    _last_seg_like.end = seg.start  # within the ≤2s gap window
                    _last_seg_like.subject_x = _last_x_pct
                    _last_seg_like.subject_y = 50
                    _last_seg_like.confidence = 0.7  # last_confident_* was set at >= 0.70

                fb_x_pct, fb_y_pct, fb_source = resolve_fallback_center(
                    _confidence_estimator,
                    ct,
                    seg.start, seg.end,
                    _last_seg_like,
                    candidate_x=_cand_x_pct,
                    candidate_y=50,
                )

                # Zero-face guard: if the proposed crop has essentially no
                # faces in it AND confidence is already low, the fallback
                # cascade returned a stale slot position (empty seat / couch
                # scene). Forcing wide_master is always better than a crop of
                # set dressing. Only applies below CONFIDENCE_MEDIUM — medium+
                # confidence means the estimator itself saw a face.
                from backend.services.subject_confidence import (
                    CONFIDENCE_MEDIUM as _CONF_MED,
                )
                _zero_face_override = False
                if conf < _CONF_MED and _confidence_estimator is not None:
                    try:
                        # Prefer the slot-strict rate when an active_slot
                        # is present: the looser per_frame_in_crop check
                        # accepts any boundary slot whose face center
                        # happens to fall inside the candidate crop, which
                        # lets the guard sleep when the actual chosen
                        # speaker is entirely off-camera.
                        if seg.active_slot is not None:
                            _zero_face_pass_rate = (
                                _confidence_estimator.per_frame_slot_visible_rate(
                                    seg.start, seg.end, seg.active_slot,
                                )
                            )
                        else:
                            _zero_face_pass_rate = (
                                _confidence_estimator.per_frame_in_crop_pass_rate(
                                    seg.start, seg.end, fb_x_pct,
                                )
                            )
                        if _zero_face_pass_rate < 0.10:
                            seg.strategy = "wide_master"
                            seg.layout = "wide_master"
                            seg.active_slot = None
                            seg.subject_x = source_width / 2.0
                            seg.reason = "zero_face_in_crop_guard"
                            seg.subject_source = "zero_face_guard"
                            seg.fallback_reason = conf_reason
                            fallback_count += 1
                            _log(
                                "segment %.1f-%.1fs: zero-face guard fired "
                                "(pass_rate=%.0f%% < 10%%, conf=%.2f) → WIDE_MASTER",
                                seg.start, seg.end,
                                _zero_face_pass_rate * 100, conf,
                            )
                            _zero_face_override = True
                    except Exception as _zfg_e:
                        logger.debug(
                            "[%s] zero-face guard check failed %.1f-%.1f: %s",
                            job_id, seg.start, seg.end, _zfg_e,
                        )

                if _zero_face_override:
                    continue  # skip get_fallback_strategy for this segment

                fallback = get_fallback_strategy(
                    conf, ct,
                    last_confident_x=_last_x_pct,
                    last_confident_slot=last_confident_slot,
                    candidate_x=fb_x_pct,
                    candidate_slot=seg.active_slot,
                )
                if fallback is not None:
                    strategy, layout, chosen_x_pct, active_slot, reason = fallback
                    seg.strategy = strategy
                    seg.layout = layout
                    # Convert fallback subject_x from 0-100 back to pixel space
                    seg.subject_x = float(chosen_x_pct) / 100.0 * source_width
                    seg.active_slot = active_slot
                    seg.reason = reason
                    # Fix 2: record the cascade tier that produced the
                    # center so logs tell us which fallback fired. No
                    # more blind "hardcoded_center" on every bad segment.
                    seg.subject_source = (
                        fb_source if strategy != "wide_master" else "wide_master"
                    )
                    seg.fallback_reason = conf_reason
                    fallback_count += 1
                    _log(
                        "segment %.1f-%.1fs: confidence=%.2f, fallback=%s "
                        "(source=%s, reason=%s)",
                        seg.start, seg.end, conf, strategy.upper(),
                        seg.subject_source, conf_reason,
                    )
            except Exception as e:
                logger.warning("[%s] Fallback ladder failed for segment %.1f-%.1f: %s",
                               job_id, seg.start, seg.end, e)

    if fallback_count > 0:
        _log("%d segments received confidence-gated fallbacks", fallback_count)

    # ── Stage 3e: Motion-aware tracking for fast content ──
    # Skip optical flow entirely on stable talking-head segments (podcast with
    # stable face positions). This saves significant compute on podcast exports.
    motion_tracking_count = 0
    _skip_motion = False
    if ct == "podcast" and face_registry and not face_registry.is_continuous_motion:
        # Stable talking-head: face positions have low stdev, no motion tracking needed
        _skip_motion = True
        _log("skipping motion tracking (stable %s content)", ct)

    if _allow_motion_tracking and dense_faces and not _skip_motion:
        try:
            from backend.services.optical_flow import (
                compute_motion_energy_per_second,
                should_use_motion_tracking,
                is_motion_chaotic,
                build_motion_tracking_path,
            )
            motion_energy = compute_motion_energy_per_second(dense_faces, video_duration)
            for seg in raw_segments:
                if seg.layout in ("wide_master", "split", "grid", "blur_fill", "stacked_gameplay"):
                    continue
                if should_use_motion_tracking(motion_energy, seg.start, seg.end):
                    motion_path = build_motion_tracking_path(
                        dense_faces, seg.start, seg.end, shot_cuts,
                    )
                    if len(motion_path) >= 2:
                        seg.strategy = "tracking"
                        seg.motion_path = motion_path
                        motion_tracking_count += 1
                elif is_motion_chaotic(motion_energy, seg.start, seg.end):
                    # Too chaotic for tracking — use wide_master
                    seg.strategy = "wide_master"
                    seg.layout = "wide_master"
                    seg.active_slot = None
                    seg.subject_x = source_width / 2.0
                    seg.reason = "chaotic_motion"
        except Exception as e:
            logger.warning("[%s] Motion tracking failed (non-fatal): %s", job_id, e)

    if motion_tracking_count > 0:
        _log("%d segments set to TRACKING (motion-aware)", motion_tracking_count)

    # ── Stage 4: Legacy look-ahead hysteresis — REMOVED (Bug 2) ──
    # The legacy two-pass hysteresis was a second independent filter that deleted
    # valid short speaker switches. Removed entirely; the confidence-gated merge
    # in the merge loop above is now the sole short-segment filter.

    # ── Stage 4b: Consolidate consecutive same-slot segments ──
    consolidated = 0
    i = 0
    while i < len(raw_segments) - 1:
        a = raw_segments[i]
        b = raw_segments[i + 1]
        if a.active_slot == b.active_slot and a.layout == b.layout:
            a.end = b.end
            raw_segments.pop(i + 1)
            consolidated += 1
        else:
            i += 1

    if consolidated > 0:
        _log("consolidated %d consecutive same-slot segments", consolidated)

    # ── Stage 5: Anticipation offset (adaptive) ──
    # When shifting a segment's start earlier, also trim the predecessor's end
    # to maintain half-open [start, end) contiguity — no overlaps.
    shot_cut_set = set(shot_cuts)
    anticipated = 0
    for idx_seg, seg in enumerate(raw_segments):
        if seg.reason == "speaker_turn":
            at_shot_cut = any(abs(seg.start - sc) < 0.15 for sc in shot_cuts)
            if at_shot_cut:
                continue
            # Adaptive anticipation: frantic content → short, calm → full
            local_anticipation_ms = pacing_estimator.anticipation_ms_at(seg.start) if _has_pacing else _anticipation_ms
            shift_s = local_anticipation_ms / 1000.0
            # Find previous shot cut to clamp (don't shift past it)
            prev_cut = 0.0
            for sc in sorted(shot_cut_set):
                if sc < seg.start:
                    prev_cut = sc
                else:
                    break
            new_start = max(prev_cut, seg.start - shift_s)
            if new_start < seg.start:
                # Trim predecessor's end to match so intervals stay contiguous
                if idx_seg > 0:
                    raw_segments[idx_seg - 1].end = new_start
                seg.start = new_start
                anticipated += 1

    if anticipated > 0:
        _log("%d speaker-turn segments shifted (adaptive anticipation)", anticipated)

    # ── Stage 6: Ease vs snap decision ──
    anticipation_s = _anticipation_ms / 1000.0
    for i, seg in enumerate(raw_segments):
        if i == 0:
            seg.ease_in_ms = 0
            continue

        is_near_cut = any(
            abs(seg.start - sc) < 0.15 or abs(seg.start + anticipation_s - sc) < 0.15
            for sc in shot_cuts
        )
        if is_near_cut:
            seg.ease_in_ms = _ease_cut_ms
        elif seg.reason == "speaker_turn":
            seg.ease_in_ms = _ease_speaker_ms
        elif seg.reason == "subject_walk":
            seg.ease_in_ms = _ease_walk_ms
        else:
            seg.ease_in_ms = 0

    # ── Stage 7: Position snapping (pixel-precise) ──
    for seg in raw_segments:
        seg.subject_x = _slot_to_x(seg.active_slot, face_registry, source_width)

    # ── Stage 7b: Anime saliency anchor override (Phase 6) ──
    #
    # Behind ``CLIPAI_ANIME_ANCHOR=1`` (default OFF). When the
    # content profile is animated AND the caller passed a list of
    # pre-computed ``AnimeAnchor`` objects via the new
    # ``anime_anchors`` kwarg, this sub-block walks each segment
    # and replaces ``seg.subject_x`` with the segment's strongest
    # anime anchor (face / motion / contrast / saturation peak).
    # Falls back to the existing ``_slot_to_x`` value when the
    # segment has no qualifying anchor — so the anime override
    # only fires on segments where the saliency pipeline produced
    # a strong-enough signal.
    #
    # The anime anchor is the answer to "always reframe the right
    # moment": for action moments it lands on the motion centroid;
    # for dialogue / reaction shots it lands on the dramatic face;
    # for spell effects it lands on the saturation peak.
    #
    # Production callers (``pipeline.py``) build ``anime_anchors``
    # by extracting per-frame features (anime_face_detector +
    # motion + contrast + saturation) and running them through
    # ``anime_anchor.score_anime_sequence``. The parity runner
    # builds them from the fixture's pre-baked features.
    anime_anchor_count = 0
    try:
        from backend.services.anime_anchor import (
            USE_ANIME_ANCHOR,
            aggregate_anchors_to_segment_x,
        )

        _is_animated = bool(
            getattr(content_profile, "is_animated", False)
            if content_profile else False
        )
        if (
            USE_ANIME_ANCHOR
            and _is_animated
            and anime_anchors
        ):
            _anime_subtype = (
                getattr(content_profile, "anime_subtype", None)
                if content_profile else None
            )
            for seg in raw_segments:
                if seg.layout in (
                    "wide_master", "split", "grid", "blur_fill", "stacked_gameplay",
                ):
                    continue
                fallback_x_pct = (
                    seg.subject_x / source_width * 100.0
                    if source_width > 0 else 50.0
                )
                anchor_x_pct, anchor_src = aggregate_anchors_to_segment_x(
                    anime_anchors,
                    start=seg.start,
                    end=seg.end,
                    fallback_x_pct=fallback_x_pct,
                )
                if anchor_src != "fallback":
                    seg.subject_x = anchor_x_pct / 100.0 * source_width
                    seg.subject_source = f"anime_anchor_{anchor_src}"
                    anime_anchor_count += 1
    except Exception as e:
        logger.warning(
            "[%s] Anime anchor override failed (non-fatal): %s",
            job_id, e,
        )
    if anime_anchor_count > 0:
        _log("AnimeAnchor: %d segments overridden by anime saliency", anime_anchor_count)

    # ── Stage 7c: Gameplay subject tracker override (Phase 7) ──
    #
    # Behind ``CLIPAI_GAMEPLAY_TRACKER=1`` (default OFF). When the
    # content profile has a ``gameplay_subtype`` (set by Phase 2's
    # normalizer when the user picks a gameplay variant) AND the
    # caller passed a list of pre-computed
    # ``GameplayMotionCentroid`` objects via the new
    # ``gameplay_motion_centroids`` kwarg, this sub-block walks
    # each segment and replaces ``seg.subject_x`` / ``seg.subject_y``
    # with the gameplay-tracker output.
    #
    # The tracker uses the per-game / per-genre ``action_center_pct``
    # from ``game_layouts.GAME_HUD_LAYOUTS`` (Phase 2) as the
    # fallback when motion is weak. So for a TPS clip with no
    # motion data the camera lands at (50, 45) — character offset
    # — instead of the legacy hard-coded (50, 50). For racing,
    # the fallback is (50, 65) — car in lower third.
    #
    # When motion centroids ARE strong (e.g. a TPS character
    # walks across the frame), the tracker overrides the
    # action-center fallback with the smoothed motion centroid,
    # producing per-segment ``subject_x`` that follows the player.
    #
    # Note that gameplay clips currently take a separate fast-path
    # in pipeline.py BEFORE the reframe segmenter even runs (the
    # ``_is_gameplay`` branch). For Phase 7, the production
    # integration in pipeline.py disables that fast-path when a
    # gameplay_subtype is set so the segmenter sees the gameplay
    # clip and the tracker can fire — that wiring lives in the
    # Phase 7 follow-up.
    gameplay_track_count = 0
    try:
        from backend.services.gameplay_subject_tracker import (
            USE_GAMEPLAY_TRACKER,
            aggregate_subject_x_for_segment,
            subject_anchor_for_game,
            subject_anchor_for_genre,
        )
        from backend.services.l1_camera_path import CENTER_BIAS_GENRES

        _gp_subtype = (
            getattr(content_profile, "gameplay_subtype", None)
            if content_profile else None
        )
        _gp_game_key = (
            getattr(content_profile, "game_type", "")
            if content_profile else ""
        )
        # Phase 1 center-bias guard: for FPS / hero-shooter / sandbox
        # content the motion-centroid tracker is NEVER allowed to
        # override the scene-level subject_x — those genres have
        # their action locked to dead center, and the tracker was
        # the single biggest source of off-center drift on cartoon
        # FPS clips (TF2 / Marvel Rivals) where character models
        # trip face/motion detection at (x ≈ 20-32). Instead we
        # force every non-composite segment in those genres to
        # ``subject_x = source_width / 2`` so the L1 solver starts
        # from the hard center anchor. Non-center-bias gaming
        # genres (MOBA / TPS / racing) still flow through the
        # tracker because their action genuinely moves off-center.
        _center_bias_active = (
            _gp_subtype is not None
            and _gp_subtype in CENTER_BIAS_GENRES
        )
        if _center_bias_active:
            _center_px = source_width / 2.0
            _bias_count = 0
            for seg in raw_segments:
                if seg.layout in (
                    "split", "grid", "wide_master", "blur_fill",
                    "stacked_gameplay",
                ):
                    continue
                # Only overwrite segments whose current subject_x
                # drifted off center — preserves the pipeline's
                # high-confidence crosshair override when it ran.
                if abs(seg.subject_x - _center_px) > 0.03 * source_width:
                    seg.subject_x = _center_px
                    seg.subject_source = (
                        seg.subject_source or ""
                    ) + "+center_bias"
                    _bias_count += 1
            if _bias_count:
                _log(
                    "Center-bias (%s): reset %d segment subject_x "
                    "to dead center (source_width/2)",
                    _gp_subtype, _bias_count,
                )
        if (
            USE_GAMEPLAY_TRACKER
            and _gp_subtype
            and not _center_bias_active
            and gameplay_motion_centroids is not None
        ):
            # Resolve the fallback anchor: prefer the specific
            # game key (e.g. "gta_v") over the genre default
            # (e.g. "tps").
            if _gp_game_key:
                fallback_xy = subject_anchor_for_game(_gp_game_key)
            else:
                fallback_xy = subject_anchor_for_genre(_gp_subtype)

            for seg in raw_segments:
                if seg.layout in (
                    "split", "grid", "wide_master", "blur_fill", "stacked_gameplay",
                ):
                    continue
                x_pct, y_pct, source = aggregate_subject_x_for_segment(
                    gameplay_motion_centroids,
                    start=seg.start,
                    end=seg.end,
                    fallback_xy_pct=fallback_xy,
                )
                seg.subject_x = x_pct / 100.0 * source_width
                seg.subject_y = y_pct / 100.0 * source_height
                seg.subject_source = f"gameplay_tracker_{source}"
                gameplay_track_count += 1
    except Exception as e:
        logger.warning(
            "[%s] Gameplay tracker override failed (non-fatal): %s",
            job_id, e,
        )
    if gameplay_track_count > 0:
        _log("GameplayTracker: %d segments overridden by per-genre tracker", gameplay_track_count)

    # ── Stage 7d: Stream layout routing (Phase 7) ──
    #
    # When the profile's gameplay_subtype is "stream" (Phase 2:
    # gaming + facecam), force the segment layout to
    # STACKED_GAMEPLAY so the renderer knows to stack the
    # gameplay portion (top) and the facecam portion (bottom).
    # The single-subject path still picks the facecam slot from
    # the face registry; stacked_gameplay is just a layout
    # marker for the renderer.
    #
    # No feature flag — this fires whenever the user explicitly
    # picked the "stream" content type, since stream is its own
    # editorial decision rather than a quality knob. The
    # downstream renderer already supports STACKED_GAMEPLAY from
    # the existing CONTENT_TYPE_CONFIG.GAMING.prefer_stacked_gameplay
    # entry — Phase 7 just routes to it via the explicit gate.
    stream_count = 0
    try:
        _gp_sub_for_stream = (
            getattr(content_profile, "gameplay_subtype", None)
            if content_profile else None
        )
        if _gp_sub_for_stream == "stream":
            for seg in raw_segments:
                if seg.layout not in (
                    "split", "grid", "wide_master", "blur_fill",
                ):
                    if seg.layout != "stacked_gameplay":
                        seg.layout = "stacked_gameplay"
                        seg.strategy = "stacked_gameplay"
                        seg.reason = seg.reason or "stream_layout"
                        stream_count += 1
    except Exception as e:
        logger.warning(
            "[%s] Stream layout routing failed (non-fatal): %s",
            job_id, e,
        )
    if stream_count > 0:
        _log("Stream: %d segments routed to STACKED_GAMEPLAY", stream_count)

    # ── Stage 8: Lead-room application (narrative/vlog) ──
    #
    # Two tiers, controlled by two independent feature flags:
    #
    #   * Categorical (legacy)   — ``estimate_gaze_from_dense``
    #     returns "left" / "right" / "center"; ``apply_lead_room``
    #     applies a step shift of ~5 % of the viewport width. This
    #     is what the segmenter has used since the original AutoFlip
    #     work and what runs by default.
    #
    #   * Continuous V2 (Phase 4) — ``estimate_yaw_from_dense``
    #     returns a smoothed float in [-1, 1] and
    #     ``lead_room_offset_px`` produces a linearly-scaled pixel
    #     offset capped at ``0.08 * crop_width`` at full yaw. Fires
    #     ONLY when ``CLIPAI_GAZE_LEAD_ROOM_V2=1`` (default OFF
    #     until the in-docker validation lands the post-Phase-4
    #     numbers — see docs/autoflip_parity_v2_results.md).
    #
    # Phase 4 also adds an optional thirds-bias x-offset
    # (``CLIPAI_THIRDS_BIAS=1``, default OFF) that runs AFTER the
    # lead-room step on segments whose content type is in
    # ``thirds_bias.THIRDS_BIAS_CONTENT_TYPES``. The thirds offset
    # is composed with whatever lead-room offset already fired, so
    # both flags can be ON simultaneously.
    # Auto-enable lead room and thirds bias for animated content regardless
    # of env var. Anime/animation_dialogue almost always has clear gaze
    # direction (characters face the person they're talking to). Lead room
    # converts "face centered in crop" to "face on the looking side" — the
    # primary difference between ClipAI and a human editor on this content
    # type. Thirds bias adds the natural off-center placement editors use.
    _is_animated_content = (
        content_profile is not None
        and getattr(content_profile, 'is_animated', False)
    )
    lead_room_count = 0
    thirds_count = 0
    if _apply_lead_room and dense_faces:
        try:
            from backend.services.gaze_estimator import (
                USE_GAZE_LEAD_ROOM_V2 as _USE_LEAD_RAW,
                apply_lead_room as _apply_lr,
                estimate_gaze_from_dense,
                estimate_yaw_from_dense,
                lead_room_offset_px,
                yaw_to_categorical,
            )
            from backend.services.thirds_bias import (
                USE_THIRDS_BIAS,
                applies_to_profile as _thirds_applies_profile,
                thirds_x_offset_px,
            )
            USE_GAZE_LEAD_ROOM_V2 = _USE_LEAD_RAW or _is_animated_content

            # Crop width in source pixels (16:9 → 9:16). Match the
            # value used by l1_camera_path / multi_region_layout so
            # offsets here are commensurate with the LP's box.
            _crop_aspect = 9.0 / 16.0
            _crop_width_px = float(source_height) * _crop_aspect
            if _crop_width_px > source_width:
                _crop_width_px = float(source_width)
            # The profile-aware check folds in the
            # is_multi_speaker_panel exclusion so debate / panel
            # shots don't get a thirds offset even though their
            # parent ContentType is "podcast".
            _thirds_auto = _is_animated_content and _thirds_applies_profile(content_profile)
            _thirds_on = (USE_THIRDS_BIAS or _thirds_auto) and _thirds_applies_profile(content_profile)
            # Phase 6: anime-action multiplier on the lead-room
            # offset. Applied to BOTH Stage 8 (stationary) and
            # Stage 10c (tracking / panning). Other anime sub-types
            # use the default 1.0.
            _anime_subtype_now = (
                getattr(content_profile, "anime_subtype", None)
                if content_profile else None
            )
            _anime_action_mult = (
                1.5 if _anime_subtype_now == "action" else 1.0
            )

            for seg in raw_segments:
                if seg.active_slot is None or seg.layout in (
                    "wide_master", "split", "grid", "blur_fill", "stacked_gameplay",
                ):
                    continue

                if USE_GAZE_LEAD_ROOM_V2:
                    # ── Phase 4 continuous-yaw path ──
                    yaw = estimate_yaw_from_dense(
                        dense_faces, seg.active_slot, seg.start, seg.end,
                    )
                    if abs(yaw) > 0.01:
                        offset_px = lead_room_offset_px(
                            yaw, _crop_width_px,
                            anime_action_multiplier=_anime_action_mult,
                        )
                        seg.subject_x = float(seg.subject_x) + offset_px
                        seg.lead_room_direction = yaw_to_categorical(yaw)
                        lead_room_count += 1
                else:
                    # ── Legacy categorical path ──
                    gaze = estimate_gaze_from_dense(
                        dense_faces, seg.active_slot, seg.start, seg.end,
                    )
                    if gaze != "center":
                        sx_pct = seg.subject_x / source_width * 100.0
                        adjusted_pct = _apply_lr(int(round(sx_pct)), gaze)
                        seg.subject_x = adjusted_pct / 100.0 * source_width
                        seg.lead_room_direction = gaze
                        lead_room_count += 1

                # ── Phase 4 thirds-bias x-offset ──
                # Optional, on top of the lead-room offset above.
                # Defers to the post-lead-room yaw (or zero when the
                # categorical path is in effect) for the left-vs-right
                # decision so both offsets compose coherently.
                if _thirds_on:
                    if USE_GAZE_LEAD_ROOM_V2:
                        _yaw_for_thirds = yaw  # already computed above
                    else:
                        # Re-derive yaw for the thirds decision so
                        # the categorical path can still pick the
                        # correct third without breaking its lead-room
                        # behavior.
                        _yaw_for_thirds = estimate_yaw_from_dense(
                            dense_faces, seg.active_slot, seg.start, seg.end,
                        )
                    _face_pct = float(seg.subject_x) / max(source_width, 1) * 100.0
                    thirds_offset = thirds_x_offset_px(
                        face_x_pct=_face_pct,
                        crop_width_px=_crop_width_px,
                        source_width_px=float(source_width),
                        yaw=_yaw_for_thirds,
                    )
                    if thirds_offset != 0.0:
                        seg.subject_x = float(seg.subject_x) + thirds_offset
                        thirds_count += 1

                # Clamp so the crop center stays inside the source
                # frame even after composing the two offsets — without
                # this, an aggressive yaw on a face near the edge
                # could push subject_x outside [half, src_w - half]
                # and downstream renderers would clip.
                _half = _crop_width_px / 2.0
                seg.subject_x = max(_half, min(source_width - _half, float(seg.subject_x)))
        except Exception as e:
            logger.warning("[%s] Lead-room application failed (non-fatal): %s", job_id, e)

    if lead_room_count > 0:
        _log("lead room applied to %d segments", lead_room_count)
    if thirds_count > 0:
        _log("thirds-bias x-offset applied to %d segments", thirds_count)

    # ── Stage 9: Hard constraints from persistent regions ──
    if persistent_regions and hasattr(persistent_regions, 'as_rects'):
        rects = persistent_regions.as_rects()
        if rects:
            for seg in raw_segments:
                seg.hard_constraints = rects

    # ── Stage 9b: Editorial "camera language" prior (Phase 8) ──
    #
    # Behind ``CLIPAI_EDITORIAL_PRIOR=1`` (default OFF until
    # in-docker validation lands the post-Phase-8 numbers). When
    # ON AND the content profile qualifies (narrative / podcast /
    # debate / cinematic_dialogue / vlog / talking_head /
    # multi_speaker_panel / animation_dialogue), this sub-block
    # runs the editorial state machine over the raw segments:
    #
    #   - **J-cuts**: shift speaker-change boundaries +200 ms
    #     after the new speaker's first audio word, so the viewer
    #     hears the new voice for a beat before seeing them.
    #   - **L-cuts**: same shift, paired editorial intent (Phase
    #     8 minimal treats them as equivalent — see editorial_prior
    #     docstring).
    #   - **Listener holds**: when speaker A finishes a declarative
    #     sentence and speaker B is silent for 400-800 ms,
    #     emit a hold decision on slot B. (For Phase 8 minimal
    #     the hold decision is COUNTED but the actual segment
    #     insertion is deferred; the L1 solver in Stage 10
    #     already smooths through the residual gap.)
    #   - **Reaction beats**: when an extreme audio spike fires
    #     on a multi-face segment, swap ``seg.active_slot`` to
    #     the non-talking face for the spike duration.
    #
    # Wrapped in try/except so any state-machine failure stays
    # non-fatal and the existing reactive intent tracker output
    # still drives the L1 solver.
    editorial_report = None
    try:
        from backend.services.editorial_prior import (
            USE_EDITORIAL_PRIOR,
            applies_to_profile as _editorial_applies,
            apply_editorial_prior,
        )

        if USE_EDITORIAL_PRIOR and _editorial_applies(content_profile):
            editorial_report = apply_editorial_prior(
                raw_segments,
                transcript_segments,
                face_registry.slots if face_registry else [],
                audio_events or [],
                content_profile=content_profile,
                speaker_to_slot=speaker_to_slot,
            )
            if editorial_report.n_j_cuts + editorial_report.n_l_cuts \
                    + editorial_report.n_listener_holds \
                    + editorial_report.n_reaction_beats > 0:
                _log(
                    "EditorialPrior: %d J-cuts, %d L-cuts, %d listener holds, %d reaction beats",
                    editorial_report.n_j_cuts,
                    editorial_report.n_l_cuts,
                    editorial_report.n_listener_holds,
                    editorial_report.n_reaction_beats,
                )
            # Phase 10: surface the editorial report to the caller
            # via the optional ``debug_out`` out-parameter so
            # pipeline.py can attach per-segment j_cut / l_cut /
            # listener_hold / reaction_beat tags onto the cached
            # RenderPlan debug payload.
            if debug_out is not None and editorial_report is not None:
                debug_out["editorial_report"] = editorial_report
    except Exception as e:
        logger.warning(
            "[%s] Editorial prior failed (non-fatal): %s",
            job_id, e,
        )

    # ── Stage 10a: Multi-region LP fit-check (Phase 3) ──
    # Behind CLIPAI_MULTI_REGION_LP=1 (default OFF until in-docker
    # validation lands the post-phase-3 numbers in
    # docs/autoflip_parity_v2_results.md). When ON, this sweep
    # re-evaluates every segment that Stage 3 routed to split / grid
    # / wide_master via the heuristic active-slot count: it builds
    # per-frame required + optional bboxes from the active speaker
    # events and runs solve_multi_region_camera_path. If the LP says
    # "all required regions fit in one crop", the segment is
    # downgraded back to single-subject (with subject_x = LP center)
    # so the existing Stage 10 single-subject loop produces the
    # smooth path. If the LP says "infeasible", the per-content
    # fallback (split_screen for podcast/debate, wide_master for
    # narrative) is honored — same decision the heuristic would have
    # made, but driven by actual geometry instead of face-count
    # rules.
    multi_region_count = 0
    multi_region_split = 0
    multi_region_wide = 0
    try:
        from backend.services.multi_region_layout import (
            USE_MULTI_REGION_LP,
            decide_multi_region_layout,
            promote_required_regions_for_segment,
            slot_to_pixel_bbox,
        )

        if USE_MULTI_REGION_LP and face_registry and face_registry.slots:
            # The crop window for 16:9 → 9:16 is source_height *
            # crop_aspect = 1080 * 9/16 = 607 px wide on a 1920-wide
            # source. Match the value used by l1_camera_path so the
            # fit decision is comparable to the single-subject solver.
            crop_aspect = 9.0 / 16.0
            crop_width_px = float(source_height) * crop_aspect
            if crop_width_px > source_width:
                crop_width_px = float(source_width)

            for seg_idx, seg in enumerate(raw_segments):
                # Only re-examine segments that the heuristic Stage 3
                # routed to a multi-subject layout. Single-subject
                # segments already get the right treatment from the
                # existing single-subject L1 solver.
                if seg.layout not in ("split", "grid", "wide_master"):
                    continue
                req_ids, opt_ids = promote_required_regions_for_segment(
                    seg_start=seg.start,
                    seg_end=seg.end,
                    face_slots=face_registry.slots,
                    active_speaker_events=active_speaker_events,
                )
                if not req_ids and not opt_ids:
                    continue
                # Snapshot per-frame regions: for the segment-level
                # fit check we sample at the segment endpoints (the
                # face slots are stationary across the segment so
                # one sample suffices). When we wire per-frame
                # tracks in Phase 4 this becomes the dense-faces
                # walk. n_samples > 1 keeps the LP smoothness terms
                # well-defined.
                n_samples = 4
                required_per_frame: list = []
                optional_per_frame: list = []
                slot_by_id = {int(s.slot_id): s for s in face_registry.slots}
                for _ in range(n_samples):
                    req_bboxes = [
                        slot_to_pixel_bbox(slot_by_id[sid], source_width=source_width)
                        for sid in req_ids
                        if sid in slot_by_id
                    ]
                    opt_bboxes = [
                        (*slot_to_pixel_bbox(slot_by_id[sid], source_width=source_width), 0.5)
                        for sid in opt_ids
                        if sid in slot_by_id
                    ]
                    required_per_frame.append(req_bboxes)
                    optional_per_frame.append(opt_bboxes)

                # ContentType for fallback choice. We use the clip-
                # level value when available; reframe_segmenter
                # doesn't currently know it directly so we look at
                # content_profile.content_type as a proxy.
                _ct = getattr(content_profile, "content_type", None) if content_profile else None
                decision = decide_multi_region_layout(
                    required_per_frame,
                    optional_per_frame,
                    crop_width_px=crop_width_px,
                    source_width_px=float(source_width),
                    content_type=_ct,
                )
                logger.info(
                    "[%s] MultiRegionLP seg %.2f-%.2fs layout=%s decision=%s "
                    "ratio=%.2f reason=%s n_req=%d n_opt=%d",
                    job_id, seg.start, seg.end, seg.layout, decision.decision,
                    decision.infeasibility_ratio, decision.reason,
                    decision.n_required, decision.n_optional,
                )
                if decision.decision == "fit" and decision.camera_path:
                    # Downgrade to single-subject with the LP center.
                    seg.layout = "single"
                    # Pick the active speaker as the slot for the
                    # downstream Stage 10 single-subject path. If
                    # there are multiple required slots, prefer the
                    # one whose pixel position is closest to the
                    # LP-solved camera center.
                    cam_center = float(decision.camera_path[0])
                    best_slot = None
                    best_dist = float("inf")
                    for sid in req_ids:
                        slot = slot_by_id.get(sid)
                        if slot is None:
                            continue
                        slot_px = float(slot.x_center) / 100.0 * source_width
                        d = abs(slot_px - cam_center)
                        if d < best_dist:
                            best_dist = d
                            best_slot = sid
                    if best_slot is None and req_ids:
                        best_slot = req_ids[0]
                    if best_slot is not None:
                        seg.active_slot = best_slot
                    seg.subject_x = cam_center
                    seg.strategy = "tracking"
                    seg.reason = "multi_region_lp_fit"
                    seg.confidence = max(seg.confidence, 0.85)
                    multi_region_count += 1
                elif decision.decision in ("fit_with_pad",):
                    # Same as fit for now — Phase 4+ may add the
                    # actual padding logic. The LP ran cleanly on
                    # ≥95% of frames so the camera path is usable.
                    if decision.camera_path:
                        seg.layout = "single"
                        seg.subject_x = float(decision.camera_path[0])
                        seg.strategy = "tracking"
                        seg.reason = "multi_region_lp_fit_with_pad"
                        seg.confidence = max(seg.confidence, 0.75)
                        multi_region_count += 1
                elif decision.decision == "split":
                    seg.layout = "split"
                    seg.strategy = "split_screen"
                    seg.reason = "multi_region_lp_split"
                    multi_region_split += 1
                elif decision.decision == "wide":
                    seg.layout = "wide_master"
                    seg.strategy = "wide_master"
                    seg.subject_x = source_width / 2.0
                    seg.reason = "multi_region_lp_wide"
                    multi_region_wide += 1
                # decision == "lp_failed" → leave the segment alone;
                # the existing heuristic decision stands.
    except Exception as e:
        logger.warning(
            "[%s] Multi-region LP sweep failed (non-fatal): %s",
            job_id, e,
        )

    if multi_region_count + multi_region_split + multi_region_wide > 0:
        _log(
            "MultiRegionLP: %d fit-downgrade, %d split, %d wide",
            multi_region_count, multi_region_split, multi_region_wide,
        )

    # ── Stage 10: L1 camera path — shot-level solving with lookahead ──
    # Group segments by shot (using shot_cuts), then solve the TV-denoised
    # camera path ONCE per shot instead of per segment. This eliminates
    # pops at segment boundaries and provides non-causal lookahead so the
    # camera begins easing BEFORE the subject moves.
    l1_count = 0
    try:
        from backend.services.l1_camera_path import (
            solve_camera_path,
            solve_camera_path_for_shot,
            get_dense_face_positions_for_segment,
            get_propagated_positions_for_segment,
        )

        source = interpolated_timeline if interpolated_timeline is not None else dense_faces
        if not source:
            raise ValueError("No face data for L1 solver")

        # Group single-layout segments by shot boundaries
        sorted_cuts = sorted(shot_cuts) if shot_cuts else []

        def _shot_for_time(t):
            """Return the shot index for a given timestamp."""
            for i, cut in enumerate(sorted_cuts):
                if t < cut:
                    return i
            return len(sorted_cuts)

        # Build shot groups: {shot_idx: [segment_indices]}
        shot_groups = {}
        eligible_indices = []
        for i, seg in enumerate(raw_segments):
            if seg.layout not in ("single",) or seg.active_slot is None:
                continue
            eligible_indices.append(i)
            shot_idx = _shot_for_time(seg.start)
            shot_groups.setdefault(shot_idx, []).append(i)

        # Solve per shot group
        _n_shots = len(shot_groups)
        logger.info(
            "[%s] L1 camera path: solving %d shot groups (%d eligible segments)",
            job_id, _n_shots, len(eligible_indices),
        )
        for _k, (shot_idx, seg_indices) in enumerate(shot_groups.items()):
            shot_segs = [raw_segments[i] for i in seg_indices]
            if not shot_segs:
                continue

            shot_start = shot_segs[0].start
            shot_end = shot_segs[-1].end

            # Per-iteration progress log — if a specific shot hangs the
            # solver, this line identifies it before the hang happens.
            # (The solver's own "L1 solver: ... starting" line is also
            # emitted but only from solve_camera_path_for_shot; this is
            # the caller-side breadcrumb so it stays attributable.)
            logger.debug(
                "[%s] L1 shot %d/%d: shot_idx=%s segs=%d span=%.2f-%.2fs",
                job_id, _k + 1, _n_shots, shot_idx,
                len(shot_segs), float(shot_start), float(shot_end),
            )

            try:
                # Anime/animation_dialogue dialogue content: use higher
                # TV lambda for longer, stickier holds that match human
                # editorial pacing. 0.03 × 1920 = 57.6 vs default 38.4.
                # Action content handled by pacing estimator reducing min_hold.
                _shot_lam_frac = (
                    0.03 if _is_animated_content else 0.02
                )
                results = solve_camera_path_for_shot(
                    shot_start=shot_start,
                    shot_end=shot_end,
                    segments_in_shot=shot_segs,
                    propagated_path=source,
                    source_width=source_width,
                    source_height=source_height,
                    lam=_shot_lam_frac * source_width,
                    job_id=job_id,
                    # Phase 4: animated content with a populated dense
                    # face track switches the L1 unary from the
                    # propagated slot center to per-frame nose_x with a
                    # higher data-fidelity weight. Gated to anime so
                    # live-action behavior is unchanged.
                    is_animated=bool(_is_animated_content),
                    dense_faces_for_anchors=dense_faces if _is_animated_content else None,
                )
            except Exception as shot_exc:
                # A single shot failing must not abort Stage 10 — skip
                # it and leave the affected segments with their
                # existing face-registry positions.
                logger.warning(
                    "[%s] L1 shot %d/%d failed (non-fatal): %s: %s",
                    job_id, _k + 1, _n_shots,
                    type(shot_exc).__name__, shot_exc,
                )
                continue

            for seg_i, result in zip(seg_indices, results):
                seg = raw_segments[seg_i]
                seg.strategy = result["mode"]

                # Phase 5: Force tracking on intra-shot face motion.
                # On animated content the L1 solver sometimes settles
                # on `stationary` for a shot whose face pans across
                # the frame — the face center error within close-ups
                # is what makes the subject drift off-center. Override
                # to `tracking` (and reuse the L1 path) when the dense
                # face data shows >6% intra-shot motion.
                if (
                    result["mode"] == "stationary"
                    and _is_animated_content
                    and dense_faces
                    and should_force_tracking_for_motion(
                        dense_faces, seg.active_slot, seg.start, seg.end,
                    )
                ):
                    if result.get("path"):
                        seg.strategy = "tracking"
                        seg.subject_x = result["path"][0][1]
                        seg.motion_path = result["path"]
                    elif result.get("center") is not None:
                        # Solver collapsed to a constant; build a flat
                        # motion_path so downstream tracking-aware
                        # post-processing fires.
                        seg.strategy = "tracking"
                        center = float(result["center"])
                        seg.subject_x = center
                        seg.motion_path = [
                            (float(seg.start), center),
                            (float(seg.end), center),
                        ]
                elif result["mode"] == "stationary":
                    # Keep face-registry-based subject_x (more stable)
                    pass
                elif result["mode"] in ("tracking", "panning"):
                    seg.subject_x = result["path"][0][1] if result["path"] else seg.subject_x
                    seg.motion_path = result["path"]

                l1_count += 1

    except Exception as e:
        logger.warning("[%s] L1 camera path failed (non-fatal): %s", job_id, e)

    if l1_count > 0:
        _log("L1 camera path solved for %d segments", l1_count)

    # ── Stage 10c: Phase 4 V2 lead-room + thirds-bias post-process ──
    #
    # Stage 8 applies Phase 4 offsets to ``seg.subject_x`` BEFORE the
    # L1 solver runs, but the L1 solver in Stage 10 OVERWRITES
    # ``seg.subject_x`` (and writes ``seg.motion_path``) for tracking
    # / panning segments — clobbering the offset. This post-process
    # re-applies the same lead-room + thirds-bias offsets to the
    # L1-solved path so the bias survives.
    #
    # For stationary segments Stage 8 already did the right thing
    # and the L1 solver leaves ``seg.subject_x`` alone; this loop
    # detects that and skips them so the offset isn't applied twice.
    #
    # For tracking / panning segments the same offset is added to
    # every entry in ``seg.motion_path`` AND to ``seg.subject_x``,
    # producing a uniformly-shifted smooth path. The shift is by a
    # constant within a single segment, which preserves smoothness
    # (max accel / max jerk are unchanged) while moving the subject
    # to the lead-room / thirds position.
    #
    # Both flags default OFF; the post-process is a no-op when neither
    # is set.
    phase4_post_count = 0
    if dense_faces:
        try:
            from backend.services.gaze_estimator import (
                USE_GAZE_LEAD_ROOM_V2 as _USE_V2,
                estimate_yaw_from_dense as _est_yaw,
                lead_room_offset_px as _lr_off,
            )
            from backend.services.thirds_bias import (
                USE_THIRDS_BIAS as _USE_TH,
                applies_to_profile as _th_applies,
                thirds_x_offset_px as _th_off,
            )
            if _USE_V2 or (_USE_TH and _th_applies(content_profile)):
                _crop_aspect = 9.0 / 16.0
                _crop_w = float(source_height) * _crop_aspect
                if _crop_w > source_width:
                    _crop_w = float(source_width)
                _half = _crop_w / 2.0
                _do_thirds = _USE_TH and _th_applies(content_profile)
                # Phase 6: anime action multiplier — passed to
                # lead_room_offset_px when the segment's parent
                # is anime + action subtype so head-turn lead-
                # room is exaggerated to match the cinematography.
                _anime_subtype_post = (
                    getattr(content_profile, "anime_subtype", None)
                    if content_profile else None
                )
                _anime_mult_post = (
                    1.5 if _anime_subtype_post == "action" else 1.0
                )
                for seg in raw_segments:
                    if seg.active_slot is None or seg.layout in (
                        "wide_master", "split", "grid", "blur_fill", "stacked_gameplay",
                    ):
                        continue
                    if seg.strategy not in ("tracking", "panning"):
                        # Stationary segments already got their offset
                        # in Stage 8 — Stage 10 didn't touch
                        # subject_x, so nothing to repair here.
                        continue
                    yaw = _est_yaw(
                        dense_faces, seg.active_slot, seg.start, seg.end,
                    ) if _USE_V2 else 0.0
                    offset = 0.0
                    if _USE_V2 and abs(yaw) > 0.01:
                        offset += _lr_off(
                            yaw, _crop_w,
                            anime_action_multiplier=_anime_mult_post,
                        )
                    if _do_thirds:
                        _face_pct = float(seg.subject_x) / max(source_width, 1) * 100.0
                        offset += _th_off(
                            face_x_pct=_face_pct,
                            crop_width_px=_crop_w,
                            source_width_px=float(source_width),
                            yaw=yaw,
                        )
                    if offset == 0.0:
                        continue
                    # Apply uniformly to subject_x AND every motion_path entry
                    seg.subject_x = max(_half, min(source_width - _half, float(seg.subject_x) + offset))
                    if seg.motion_path:
                        new_path = []
                        for entry in seg.motion_path:
                            t = entry[0]
                            x = float(entry[1]) + offset
                            x = max(_half, min(source_width - _half, x))
                            # Preserve any extra tuple elements (e.g. y)
                            new_path.append((t, x) + tuple(entry[2:]))
                        seg.motion_path = new_path
                    phase4_post_count += 1
        except Exception as e:
            logger.warning(
                "[%s] Phase 4 V2 post-process failed (non-fatal): %s",
                job_id, e,
            )
    if phase4_post_count > 0:
        _log(
            "Phase4 V2 post-process: %d tracking/panning segments shifted",
            phase4_post_count,
        )

    # ── Fix 5: surrounding-mean center for wide_master segments ──
    # Any segment still showing strategy=wide_master with subject_x
    # exactly at source_width/2 is the hardcoded-center fallback from
    # Stage 3. Replace with the mean subject_x of its nearest confident
    # neighbors within ±5 s so panning continuity is preserved instead
    # of yanking the crop to the middle of the frame.
    try:
        _center_px = source_width / 2.0
        _surround_window = 5.0
        _rewritten = 0
        for i, seg in enumerate(raw_segments):
            if seg.strategy != "wide_master":
                continue
            if abs(seg.subject_x - _center_px) > 1.0:
                continue  # already has a custom center, leave alone
            neigh_xs = []
            seg_mid = (seg.start + seg.end) / 2.0
            for j, nseg in enumerate(raw_segments):
                if j == i or nseg.strategy == "wide_master":
                    continue
                n_mid = (nseg.start + nseg.end) / 2.0
                if abs(n_mid - seg_mid) <= _surround_window:
                    neigh_xs.append(nseg.subject_x)
            if neigh_xs:
                seg.subject_x = float(sum(neigh_xs)) / len(neigh_xs)
                seg.subject_source = "wide_master_surround_mean"
                _rewritten += 1
        if _rewritten > 0:
            _log(
                "wide_master surround-mean: %d segments rewritten "
                "from hardcoded center to neighbor mean (±%.0fs)",
                _rewritten, _surround_window,
            )
    except Exception as _wmr:
        logger.warning(
            "[%s] wide_master surround-mean rewrite failed: %s",
            job_id, _wmr,
        )

    # ── Stage 11 (gaming): per-segment layout mode chooser (Phase 4) ──
    # Gated on content_profile having a gameplay_subtype. Picks one of
    # fullscreen / blurfill / composite / wide_zoom per segment based on
    # genre + segment activity (events + motion). Non-gaming segments
    # are untouched. Caller-provided ``gaming_layout_mode`` overrides
    # are preserved.
    try:
        _gp_subtype_for_layout = (
            getattr(content_profile, "gameplay_subtype", None)
            if content_profile else None
        )
        if _gp_subtype_for_layout:
            from backend.services.gaming_layout_chooser import (
                USE_GAMING_LAYOUT_CHOOSER,
                assign_gaming_layouts,
            )
            if USE_GAMING_LAYOUT_CHOOSER:
                # Optional gaming events list — passed via job_id /
                # global state in production, plumbed through here as
                # a no-op since the segmenter doesn't yet receive it.
                _events_for_layout: list = []
                # Optional per-segment motion magnitude. We don't yet
                # have a clean dict here so leave it None — the
                # chooser falls through to the genre default.
                n_assigned = assign_gaming_layouts(
                    raw_segments,
                    events=_events_for_layout,
                    motion_by_segment=None,
                    profile=content_profile,
                )
                if n_assigned:
                    _log(
                        "Gaming layout chooser: assigned %d segments "
                        "(%s genre)",
                        n_assigned, _gp_subtype_for_layout,
                    )
    except Exception as e:
        logger.warning(
            "[%s] Gaming layout chooser failed (non-fatal): %s",
            job_id, e,
        )

    # ── Summary logging ──
    slot_counts = Counter()
    strategy_counts = Counter()
    for seg in raw_segments:
        if seg.active_slot is not None:
            slot_counts[seg.active_slot] += 1
        else:
            slot_counts["wide"] += 1
        strategy_counts[seg.strategy] += 1

    unique_x = set(seg.subject_x for seg in raw_segments)
    slot_str = ", ".join(f"{k}: {v}" for k, v in sorted(slot_counts.items(), key=lambda kv: str(kv[0])))
    total_segs = len(raw_segments)

    strategy_str = ", ".join(
        f"{k}={v} ({100*v//max(total_segs,1)}%)"
        for k, v in sorted(strategy_counts.items(), key=lambda x: -x[1])
    )
    _log("FINAL (%s): %d segments", ct, total_segs)
    _log("  strategies: %s", strategy_str)

    x_parts = []
    wide_px = source_width / 2.0
    if face_registry:
        for slot in face_registry.slots:
            slot_px = slot.x_center / 100.0 * source_width
            x_parts.append(f"slot{slot.slot_id}={slot_px:.1f}px")
    if any(abs(x - wide_px) < 1.0 for x in unique_x):
        x_parts.append(f"wide={wide_px:.0f}px")
    _log("  unique_x: %d (%s)", len(unique_x), ", ".join(x_parts))
    _log("  slots: {%s}", slot_str)
    if lead_room_count > 0:
        _log("  lead_room_applied: %d segments", lead_room_count)

    # ── Per-segment diagnostic logging ──
    for seg in raw_segments:
        face_slots_at_mid = []
        seg_mid = (seg.start + seg.end) / 2.0
        if dense_faces:
            # Find face slots visible near segment midpoint
            for df in dense_faces:
                if abs(df.timestamp - seg_mid) < 0.5 and df.faces:
                    for f in df.faces:
                        sid = getattr(f, 'identity_id', -1)
                        if sid >= 0:
                            fx = getattr(f, 'nose_x', getattr(f, 'x', 50))
                            face_slots_at_mid.append((sid, round(fx, 1)))
                    break

        try:
            from backend.services.subject_confidence import face_in_proposed_crop as _fipc
            in_crop = _fipc(seg, face_registry, dense_faces)
        except Exception:
            in_crop = "unknown"
        logger.info(
            "[reframe] segment t=%.2f-%.2f strategy=%s subject_x=%.1f subject_y=%.1f "
            "source_path=%s face_slots=%s active_slot=%s "
            "confidence=%.2f fallback_reason=%s face_in_crop_rect=%s",
            seg.start, seg.end, seg.strategy, seg.subject_x, seg.subject_y,
            seg.subject_source or "unknown",
            face_slots_at_mid,
            seg.active_slot,
            seg.confidence,
            seg.fallback_reason or "none",
            in_crop,
        )

    # ── Stage 11: Music-video beat snap + pulse cuts (Phase 5) ──
    #
    # Behind ``CLIPAI_MUSIC_BEAT_SNAP=1`` (default OFF). Two passes
    # over the segment list, both gated on:
    #
    #   - the feature flag is on
    #   - ``content_profile.content_type == "music_video"``
    #   - a ``music_beat_grid`` was passed in (production path:
    #     pipeline calls ``beat_detector.detect_beats(audio_path)``;
    #     test path: the parity runner pulls the pre-baked grid from
    #     the fixture's ``ground_truth.beat_grid``)
    #
    # Pass A — boundary snap. Walks pairs (prev, cur) and snaps
    # ``cur.start`` to the nearest downbeat within ±200 ms,
    # propagating the snap to ``prev.end`` so contiguity is
    # preserved. Skips snaps that would shrink either neighbor
    # below 0.30 s OR that exceed half the segment length (per the
    # v2 spec: "preserve sub-second switches — only snap if snap
    # distance < half the segment length").
    #
    # Pass B — pulse cuts. For every downbeat that lies STRICTLY
    # INSIDE an existing segment (more than 100 ms from either
    # boundary), splits the segment in two. The new segment
    # inherits the active speaker slot and its subject_x is
    # re-derived from the slot's center via ``_slot_to_x`` so the
    # pulse cut produces a visible "fresh anchor" even though the
    # speaker hasn't changed. The motion_path of a tracking /
    # panning segment is sliced at the split point so each half
    # carries its own path.
    #
    # The exit-invariant pass below normalizes any small float
    # drift introduced by these mutations.
    music_snap_count = 0
    pulse_cut_count = 0
    formation_snap_count = 0
    try:
        from backend.services.beat_detector import (
            USE_MUSIC_BEAT_SNAP,
            BeatGrid,
            enumerate_pulse_cuts,
            snap_segment_boundaries,
        )

        _ct_for_music = (
            getattr(content_profile, "content_type", None)
            if content_profile else None
        )
        if (
            USE_MUSIC_BEAT_SNAP
            and _ct_for_music == "music_video"
            and isinstance(music_beat_grid, BeatGrid)
            and music_beat_grid.has_data
        ):
            # Pass A: snap segment boundaries
            music_snap_count = snap_segment_boundaries(
                raw_segments, music_beat_grid,
                max_distance_sec=0.20,
                min_segment_sec=0.30,
            )

            # Pass B: insert pulse cuts at internal downbeats
            #
            # Iterate by *original* segment index, generate splits,
            # then collapse them all into the new list. Generates
            # at most one new segment per downbeat per existing
            # segment, with the original segment's active_slot and
            # the slot-center subject_x.
            pulse_pairs = enumerate_pulse_cuts(
                music_beat_grid, raw_segments, edge_skip_sec=0.10,
            )
            if pulse_pairs:
                # Group splits by segment index for batch insertion
                splits_by_seg: dict[int, list[float]] = {}
                for seg_idx, db in pulse_pairs:
                    splits_by_seg.setdefault(seg_idx, []).append(db)

                new_segments: list = []
                for i, seg in enumerate(raw_segments):
                    splits = sorted(splits_by_seg.get(i, []))
                    if not splits:
                        new_segments.append(seg)
                        continue
                    # Walk the segment, emitting one piece per split.
                    cursor = float(seg.start)
                    end = float(seg.end)
                    last_template = seg
                    for db in splits:
                        if db <= cursor + 1e-6 or db >= end - 1e-6:
                            continue
                        # Trim the existing piece up to the downbeat
                        last_template.end = db
                        # Slice motion_path if present so the split
                        # halves carry their own paths
                        if getattr(last_template, "motion_path", None):
                            last_template.motion_path = [
                                e for e in last_template.motion_path
                                if e[0] <= db
                            ]
                        new_segments.append(last_template)

                        # Build the post-downbeat piece via
                        # dataclass replace; copy all attributes
                        # then override start / end / subject_x.
                        from dataclasses import replace as _dc_replace
                        try:
                            piece = _dc_replace(
                                seg,
                                start=db,
                                end=end,
                                reason="music_pulse_cut",
                                subject_x=_slot_to_x(
                                    seg.active_slot, face_registry, source_width,
                                ),
                                motion_path=(
                                    [e for e in seg.motion_path if e[0] >= db]
                                    if getattr(seg, "motion_path", None)
                                    else None
                                ),
                            )
                        except TypeError:
                            # If replace fails (e.g. dataclass shape
                            # changed), fall back to manual copy.
                            piece = seg
                            piece.start = db
                        cursor = db
                        last_template = piece
                        pulse_cut_count += 1
                    new_segments.append(last_template)
                raw_segments = new_segments

            # ── Pass C: formation → downbeat snap (Week 2 Part E) ──
            # For any segment whose LAST probeable frame is a formation
            # shot (3+ faces spanning >=55% of frame width), extend the
            # cut point to the next downbeat (up to 1.5s out) so the
            # camera holds the formation wide past the natural boundary
            # and lands the tight crop on the beat. Only applies when
            # the downbeat falls inside the NEXT segment and extending
            # it doesn't shrink the next segment below 0.30s.
            if dense_faces:
                faces_by_t: dict = {}
                for _df in dense_faces:
                    _ts = getattr(_df, "timestamp", None)
                    if _ts is not None:
                        faces_by_t[round(float(_ts), 2)] = _df
                _downbeats = (
                    music_beat_grid.downbeat_times
                    if music_beat_grid is not None else []
                )
                for i, seg in enumerate(raw_segments[:-1]):
                    end_probe_t = round(max(0.0, seg.end - 0.15), 2)
                    _fr = faces_by_t.get(end_probe_t)
                    if _fr is None or not _is_formation_frame(_fr):
                        continue
                    # Find the next downbeat strictly after seg.end.
                    next_db = None
                    for db in _downbeats:
                        if db > seg.end + 0.05:
                            next_db = db
                            break
                    if next_db is None or (next_db - seg.end) > 1.5:
                        continue
                    nxt = raw_segments[i + 1]
                    if not (nxt.start < next_db < nxt.end):
                        continue
                    if (nxt.end - next_db) < 0.30:
                        continue
                    seg.end = float(next_db)
                    nxt.start = float(next_db)
                    formation_snap_count += 1
    except Exception as e:
        logger.warning(
            "[%s] Music-video beat snap failed (non-fatal): %s",
            job_id, e,
        )

    if (
        music_snap_count > 0
        or pulse_cut_count > 0
        or formation_snap_count > 0
    ):
        _log(
            "MusicBeatSnap: %d boundary snaps, %d pulse cuts, %d formation snaps",
            music_snap_count, pulse_cut_count, formation_snap_count,
        )

    # ── Exit: enforce half-open [start, end) contiguity ──
    # Sort, snap adjacent boundaries to exact equality, and assert no overlaps.
    raw_segments.sort(key=lambda s: s.start)
    for a, b in zip(raw_segments, raw_segments[1:]):
        # Snap to exact equality to kill float drift
        b.start = a.end
    if raw_segments:
        assert raw_segments[0].start == 0.0, (
            f"First segment starts at {raw_segments[0].start}, expected 0.0"
        )
        assert abs(raw_segments[-1].end - video_duration) < 1e-6, (
            f"Last segment ends at {raw_segments[-1].end}, expected {video_duration}"
        )
    for a, b in zip(raw_segments, raw_segments[1:]):
        assert a.end <= b.start, (
            f"Overlap: segment ending at {a.end} > next starting at {b.start}"
        )

    return raw_segments


# ── Internal helpers ──

def _count_active_slots(
    start: float,
    end: float,
    dense_faces: list,
) -> int:
    """Count how many distinct face slots appear in this interval."""
    slots_seen = set()
    for df in dense_faces:
        if df.timestamp < start or df.timestamp > end:
            continue
        for f in df.faces:
            sid = getattr(f, 'identity_id', -1)
            if sid >= 0:
                slots_seen.add(sid)
    return len(slots_seen)


def _check_speaker_overlap(
    start: float,
    end: float,
    active_speaker_events: list,
) -> float:
    """Check if two different speakers are both active simultaneously.

    Returns the total duration of overlap in seconds.
    """
    if not active_speaker_events:
        return 0.0

    # Collect time ranges per slot
    from collections import defaultdict
    slot_ranges = defaultdict(list)
    for ev in active_speaker_events:
        overlap_start = max(start, ev.start)
        overlap_end = min(end, ev.end)
        if overlap_start < overlap_end:
            slot_ranges[ev.slot_id].append((overlap_start, overlap_end))

    if len(slot_ranges) < 2:
        return 0.0

    # Find pairwise overlap between the two most active slots
    slots = sorted(slot_ranges.keys(), key=lambda s: sum(e - s_ for s_, e in slot_ranges[s]), reverse=True)
    if len(slots) < 2:
        return 0.0

    ranges_a = slot_ranges[slots[0]]
    ranges_b = slot_ranges[slots[1]]
    total_overlap = 0.0
    for a_start, a_end in ranges_a:
        for b_start, b_end in ranges_b:
            ov_start = max(a_start, b_start)
            ov_end = min(a_end, b_end)
            if ov_start < ov_end:
                total_overlap += ov_end - ov_start

    return total_overlap


def _slot_visible_in_interval(
    slot_id: Optional[int],
    start: float,
    end: float,
    dense_faces: list,
    *,
    min_frame_fraction: float = 0.30,
) -> bool:
    """True when ≥``min_frame_fraction`` of dense-face frames in
    ``[start, end]`` contain at least one face whose identity_id matches
    ``slot_id``. Used to gate audio-attributed slot picks: if the
    audio-attributed speaker has no visible face in the interval, the
    segmenter must NOT force a tight crop on the empty seat — it should
    fall through to dense-face dominance / spread resolution.
    """
    if slot_id is None or not dense_faces:
        return False
    n_total = 0
    n_with_slot = 0
    for df in dense_faces:
        ts = float(getattr(df, "timestamp", 0.0))
        if ts < start or ts > end:
            continue
        n_total += 1
        for f in getattr(df, "faces", []) or []:
            sid = getattr(f, "identity_id", -1)
            if sid is not None and int(sid) == int(slot_id):
                n_with_slot += 1
                break
    if n_total == 0:
        return False
    return (n_with_slot / n_total) >= min_frame_fraction


def _resolve_slot_for_interval(
    start: float,
    end: float,
    transcript_segments: list,
    speaker_to_slot: dict[str, int],
    active_speaker_events: list,
    dense_faces: list,
    face_registry,
    source_width: int = 1920,
    source_height: int = 1080,
) -> tuple[Optional[int], float, str, str]:
    """Determine the active slot for a time interval.

    Priority:
      1. Transcript-speaker override (>60% coverage, confidence > 0.6)
         — gated on visible-slot evidence in dense_faces
      2. Active-speaker majority (mode > 50%)
         — gated on visible-slot evidence in dense_faces
      3. Dense face dominance (one slot in 70%+ frames)
      4. Multi-face spread check — pick active speaker or split/blur
      5. Wide master fallback

    Returns:
        (active_slot, confidence, layout, subject_source)
    """
    duration = end - start
    if duration <= 0:
        return None, 0.0, "wide_master", "hardcoded_center"

    # Priority 1: Transcript-speaker override
    if transcript_segments and speaker_to_slot:
        slot, coverage = _transcript_slot_coverage(
            start, end, transcript_segments, speaker_to_slot)
        if slot is not None and coverage >= SPEAKER_COVERAGE_THRESHOLD:
            # Off-screen-speaker guard: only honor the audio-attributed
            # slot when the dense face track confirms that slot is
            # actually visible in this interval. Reaction-shot edits and
            # B-roll cuts away from the talking head are common in panel
            # shows; forcing a tight crop on the empty seat is wrong.
            if _slot_visible_in_interval(slot, start, end, dense_faces):
                return slot, min(1.0, coverage), "single", "active_speaker_slot"
            # else: fall through to visible-face priorities below.

    # Priority 2: Active-speaker majority
    if active_speaker_events:
        slot, coverage = _active_speaker_majority(start, end, active_speaker_events)
        if slot is not None and coverage >= 0.5:
            if _slot_visible_in_interval(slot, start, end, dense_faces):
                return slot, min(1.0, coverage), "single", "active_speaker_slot"
            # else: fall through.

    # Priority 3: Dense face dominance
    if dense_faces and face_registry:
        slot = _dense_face_dominant_slot(
            start, end, dense_faces, face_registry,
            active_speaker_events=active_speaker_events,
        )
        if slot is not None and _slot_visible_in_interval(
            slot, start, end, dense_faces, min_frame_fraction=0.30,
        ):
            return slot, 0.7, "single", "dense_face_dominant"
        # else: slot exists in aggregate but not in this interval — fall through.

    # Priority 4: Multi-face spread check
    # When multiple confident faces are visible but none dominates,
    # check if they fit in a single crop. If not, pick active speaker
    # or use split/blur. NEVER average distant face positions.
    if dense_faces and face_registry and len(face_registry.slots) >= 2:
        result = _resolve_multi_face_spread(
            start, end, face_registry, active_speaker_events,
            dense_faces, source_width, source_height,
        )
        if result is not None:
            return result

    # Priority 5: Wide master fallback
    return None, 0.3, "wide_master", "hardcoded_center"


def _transcript_slot_coverage(
    start: float,
    end: float,
    transcript_segments: list,
    speaker_to_slot: dict[str, int],
) -> tuple[Optional[int], float]:
    """Find transcript speaker covering the most of this interval."""
    duration = end - start
    if duration <= 0:
        return None, 0.0

    slot_time = Counter()
    for seg in transcript_segments:
        overlap_start = max(start, seg.start)
        overlap_end = min(end, seg.end)
        if overlap_start >= overlap_end:
            continue
        speaker = getattr(seg, 'speaker', None) or ''
        conf = getattr(seg, 'confidence', None)
        if conf is None:
            conf = 1.0
        if conf < SPEAKER_CONFIDENCE_THRESHOLD:
            continue
        slot_id = speaker_to_slot.get(speaker)
        if slot_id is not None:
            slot_time[slot_id] += overlap_end - overlap_start

    if not slot_time:
        return None, 0.0

    best_slot = slot_time.most_common(1)[0]
    coverage = best_slot[1] / duration
    return best_slot[0], coverage


def _active_speaker_majority(
    start: float,
    end: float,
    active_speaker_events: list,
) -> tuple[Optional[int], float]:
    """Find the mode active-speaker slot in the interval."""
    duration = end - start
    if duration <= 0:
        return None, 0.0

    slot_time = Counter()
    for ev in active_speaker_events:
        overlap_start = max(start, ev.start)
        overlap_end = min(end, ev.end)
        if overlap_start >= overlap_end:
            continue
        slot_time[ev.slot_id] += overlap_end - overlap_start

    if not slot_time:
        return None, 0.0

    best_slot = slot_time.most_common(1)[0]
    coverage = best_slot[1] / duration
    return best_slot[0], coverage


def _dense_face_dominant_slot(
    start: float,
    end: float,
    dense_faces: list,
    face_registry,
    active_speaker_events: Optional[list] = None,
) -> Optional[int]:
    """Check if one face slot dominates the dense frames in this interval.

    A slot is considered dominant when:
      * it appears in >= DENSE_DOMINANCE_THRESHOLD of frames with faces, AND
      * no other slot appears in >= 30% of frames, AND
      * if active_speaker_events are provided, no other slot has stronger
        active-speaker majority in the same interval (which would mean we
        should be tracking the talker, not the most-photographed seat).

    Returns the dominant slot id, or None if no slot meets the bar.
    """
    frames_in_range = [
        df for df in dense_faces
        if start <= df.timestamp <= end and df.faces
    ]
    if not frames_in_range:
        return None

    total = len(frames_in_range)
    slot_counts = Counter()
    for df in frames_in_range:
        seen_slots = set()
        for f in df.faces:
            sid = getattr(f, 'identity_id', -1)
            if sid >= 0 and sid not in seen_slots:
                slot_counts[sid] += 1
                seen_slots.add(sid)

    if not slot_counts:
        return None

    best_slot_id, best_count = slot_counts.most_common(1)[0]
    if best_count / total < DENSE_DOMINANCE_THRESHOLD:
        return None

    # Other slots <30% rule.
    for sid, cnt in slot_counts.items():
        if sid != best_slot_id and cnt / total >= 0.30:
            return None  # Multiple strong faces — not dominant

    # Active-speaker tiebreaker: the most-photographed seat is not always
    # the talker (panel reaction shots, B-roll cuts). If a different slot
    # has voting majority in active_speaker_events for this interval AND
    # is sufficiently visible, prefer that one.
    if active_speaker_events:
        as_slot, as_coverage = _active_speaker_majority(
            start, end, active_speaker_events,
        )
        if (
            as_slot is not None
            and as_slot != best_slot_id
            and as_coverage >= 0.50
            and slot_counts.get(as_slot, 0) / total >= 0.30
        ):
            return as_slot

    return best_slot_id


def _resolve_multi_face_spread(
    start: float,
    end: float,
    face_registry,
    active_speaker_events: list,
    dense_faces: list,
    source_width: int = 1920,
    source_height: int = 1080,
) -> Optional[tuple[Optional[int], float, str, str]]:
    """Handle multi-face segments where no single face dominates.

    When multiple confident face slots are visible but none wins majority,
    checks whether they fit in a single crop. If they do, returns their
    centroid. If they don't, picks the active speaker or falls back to
    split/blur. NEVER averages positions of faces that don't fit in one crop.

    Returns:
        (active_slot, confidence, layout, subject_source) or None if not applicable.
    """
    # Find slots with faces visible in this interval
    frames_in_range = [
        df for df in dense_faces
        if start <= df.timestamp <= end and df.faces
    ]
    if not frames_in_range:
        return None

    total = len(frames_in_range)
    slot_counts = Counter()
    for df in frames_in_range:
        seen = set()
        for f in df.faces:
            sid = getattr(f, 'identity_id', -1)
            if sid >= 0 and sid not in seen:
                slot_counts[sid] += 1
                seen.add(sid)

    # Only applies when 2+ slots are confidently visible (>20% of frames)
    confident_slot_ids = [sid for sid, cnt in slot_counts.items()
                          if cnt / total >= MULTI_SPEAKER_THRESHOLD]
    if len(confident_slot_ids) < 2:
        return None

    confident_slots = [face_registry.slot_by_id(sid)
                       for sid in confident_slot_ids]
    confident_slots = [s for s in confident_slots if s is not None]
    if len(confident_slots) < 2:
        return None

    # Compute the spread (max distance between face slot centers)
    xs = [s.x_center for s in confident_slots]
    spread = max(xs) - min(xs)

    # Compute crop width as percentage of source frame
    src_aspect = source_width / source_height if source_height > 0 else 16 / 9
    target_aspect = 9 / 16
    crop_width_pct = (target_aspect / src_aspect) * 100  # ~31.6% for 16:9→9:16

    # If all confident faces fit inside ONE crop window (with 15% padding), centroid is safe
    if spread < crop_width_pct * 0.85:
        cx = sum(xs) / len(xs)
        # Find the slot closest to centroid
        nearest = min(confident_slots, key=lambda s: abs(s.x_center - cx))
        return nearest.slot_id, 0.75, "single", "dense_face_dominant"

    # Faces are spread wider than a single crop can contain.
    # NEVER average — that lands between them on empty space.

    # Tiebreaker 1: Active speaker picks the slot
    if active_speaker_events:
        slot_id, coverage = _active_speaker_majority(start, end, active_speaker_events)
        if slot_id is not None and slot_id in confident_slot_ids:
            return slot_id, min(1.0, max(0.6, coverage)), "single", "active_speaker_slot"

    # Tiebreaker 2: Check for simultaneous speaking (overlap)
    if active_speaker_events:
        overlap_dur = _check_speaker_overlap(start, end, active_speaker_events)
        seg_dur = end - start
        if seg_dur > 0 and overlap_dur / seg_dur > 0.30:
            # Both speaking >30% of the segment — split screen
            return None, 0.8, "split", "multi_face_split"

    # Tiebreaker 3: Pick the slot with the most frames (most recently dominant)
    best_sid = max(confident_slot_ids, key=lambda sid: slot_counts[sid])
    return best_sid, 0.55, "single", "dense_face_dominant"


def _is_multi_speaker_crowd(
    start: float,
    end: float,
    dense_faces: list,
    face_registry,
) -> bool:
    """Check if 3+ face slots each have faces in >20% of dense frames."""
    frames_in_range = [
        df for df in dense_faces
        if start <= df.timestamp <= end and df.faces
    ]
    if not frames_in_range:
        return False

    total = len(frames_in_range)
    if total == 0:
        return False

    slot_counts = Counter()
    for df in frames_in_range:
        seen_slots = set()
        for f in df.faces:
            sid = getattr(f, 'identity_id', -1)
            if sid >= 0 and sid not in seen_slots:
                slot_counts[sid] += 1
                seen_slots.add(sid)

    active_slots = sum(1 for cnt in slot_counts.values()
                       if cnt / total >= MULTI_SPEAKER_THRESHOLD)
    return active_slots >= 3


def _slot_to_x(active_slot: Optional[int], face_registry, source_width: int = 1920) -> float:
    """Convert a slot id to pixel-precise subject_x.

    Returns source pixel coordinate for the face slot's center,
    or the source center pixel for None (wide fallback).
    """
    if active_slot is None:
        return source_width / 2.0  # center pixel
    if face_registry:
        slot = face_registry.slot_by_id(active_slot)
        if slot:
            # x_center is 0-100 in the face registry; convert to pixels
            return slot.x_center / 100.0 * source_width
    return source_width / 2.0  # center pixel


def _merge_short_segment(
    segments: list[ReframeSegment],
    idx: int,
    shot_cut_set: set[float] = frozenset(),
) -> Optional[int]:
    """Merge a short segment into the best neighbor.

    Uses half-open [start, end) semantics. After merge the absorbed segment's
    entire time range is covered by the neighbor and contiguity is preserved.

    Never merges across shot-cut boundaries — shot cuts are hard boundaries.

    Returns neighbor index or None.
    """
    seg = segments[idx]

    def _is_shot_cut_between(t: float) -> bool:
        return any(abs(t - sc) < 0.05 for sc in shot_cut_set)

    left = segments[idx - 1] if idx > 0 else None
    right = segments[idx + 1] if idx < len(segments) - 1 else None

    # Never merge across shot cuts
    left_ok = left and not _is_shot_cut_between(seg.start)
    right_ok = right and not _is_shot_cut_between(seg.end)

    def _absorb_into_left():
        left.end = seg.end
        segments.pop(idx)
        return idx - 1

    def _absorb_into_right():
        right.start = seg.start
        segments.pop(idx)
        return idx

    # Prefer neighbor with same active_slot
    if left_ok and left.active_slot == seg.active_slot:
        return _absorb_into_left()
    if right_ok and right.active_slot == seg.active_slot:
        return _absorb_into_right()
    # Merge into the longer neighbor (respecting shot-cut barriers)
    if left_ok and right_ok:
        left_dur = left.end - left.start
        right_dur = right.end - right.start
        if left_dur >= right_dur:
            return _absorb_into_left()
        else:
            return _absorb_into_right()
    elif left_ok:
        return _absorb_into_left()
    elif right_ok:
        return _absorb_into_right()
    return None
