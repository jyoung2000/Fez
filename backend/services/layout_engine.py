"""Layout decision engine for multi-speaker video reframing.

Analyzes face positions, speaker activity, and frame content to decide
the optimal layout mode at each timestamp. Produces a layout timeline
that the FFmpeg export pipeline uses to composite the output.

Layout modes:
  SINGLE     — One subject. Crop + pan to follow them. (default, only path)
  SPLIT      — Two subjects visible. Side-by-side vertical split.
  TRIPLE     — Three subjects. 2-up top + 1 bottom, or 1 top + 2 bottom.
  PIP        — Primary speaker large + secondary speaker small overlay.
  SCREENSHARE — Screen/slides content top half + speaker bottom half.
  GAMEPLAY   — Game footage top 70% + speaker webcam bottom 30%.

Multi-layout modes (SPLIT, TRIPLE, etc.) are reachable only behind the
ALLOW_MULTI_LAYOUT flag (default False). When disabled, the engine always
emits SINGLE — pure reframing, no layout gimmicks.
"""
import logging
import os
from collections import Counter
from dataclasses import dataclass, field

# When False (default), layout is always SINGLE — pure reframing.
# Multi-layout (split, triple, pip, etc.) is dead code behind this flag.
ALLOW_MULTI_LAYOUT = os.environ.get("ALLOW_MULTI_LAYOUT", "false").lower() in ("true", "1", "yes")

logger = logging.getLogger(__name__)


@dataclass
class LayoutSegment:
    """A continuous time range with a fixed layout mode."""
    start: float
    end: float
    layout_mode: str
    face_positions: list = field(default_factory=list)
    transition_type: str = "cut"  # "cut" or "dissolve"

    # For SPLIT mode:
    left_face_slot: int = -1
    right_face_slot: int = -1

    # For PIP mode:
    primary_face_slot: int = -1
    pip_face_slot: int = -1
    pip_position: str = "bottom_right"
    pip_size_pct: float = 25.0

    def to_dict(self) -> dict:
        # Ensure all values are plain Python types (no numpy.float32 etc.)
        def _sanitize(v):
            if hasattr(v, 'item'):  # numpy scalar
                return v.item()
            return v

        clean_positions = []
        for fp in self.face_positions:
            clean_positions.append({k: _sanitize(v) for k, v in fp.items()})

        return {
            "start": float(self.start),
            "end": float(self.end),
            "layout_mode": str(self.layout_mode),
            "face_positions": clean_positions,
            "transition_type": self.transition_type,
            "left_face_slot": int(self.left_face_slot),
            "right_face_slot": int(self.right_face_slot),
            "primary_face_slot": int(self.primary_face_slot),
            "pip_face_slot": int(self.pip_face_slot),
            "pip_position": self.pip_position,
            "pip_size_pct": float(self.pip_size_pct),
        }


@dataclass
class LayoutTimeline:
    """Complete layout plan for a clip."""
    segments: list  # list[LayoutSegment]
    default_mode: str
    face_registry: object = None
    total_layout_changes: int = 0

    def to_dict(self) -> dict:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "default_mode": self.default_mode,
            "total_layout_changes": self.total_layout_changes,
        }


def _vote_layout_for_frame(
    frame_faces,
    active_speaker_slot: int,
    has_screen_content: bool,
    registry=None,
) -> str:
    """Vote for a layout mode based on a single frame's data."""
    from backend.models import LayoutMode

    n_faces = len(frame_faces.faces)

    # Rule 1: Screen content detected → screenshare if there's also a face
    if has_screen_content and n_faces >= 1:
        return LayoutMode.SCREENSHARE

    # Rule 2: 3+ distinct face identities → triple
    distinct_ids = set(f.identity_id for f in frame_faces.faces if f.identity_id >= 0)
    if len(distinct_ids) >= 3:
        return LayoutMode.TRIPLE

    # Rule 3: 2 distinct identities
    if len(distinct_ids) == 2:
        # If both speakers occupy >5% of frame width each → SPLIT
        # If one is much smaller (background person) → SINGLE focused on larger
        id_sizes = {}
        for f in frame_faces.faces:
            if f.identity_id >= 0:
                id_sizes[f.identity_id] = max(id_sizes.get(f.identity_id, 0), f.width)
        sizes = sorted(id_sizes.values(), reverse=True)
        if len(sizes) >= 2 and sizes[1] > 5.0:
            return LayoutMode.SPLIT
        else:
            return LayoutMode.SINGLE

    # Rule 4: 2 faces detected but without identity data
    if n_faces >= 2 and len(distinct_ids) < 2:
        # Check by raw face sizes
        face_widths = sorted([f.width for f in frame_faces.faces], reverse=True)
        if len(face_widths) >= 2 and face_widths[1] > 5.0:
            return LayoutMode.SPLIT

    # Rule 5: 0-1 faces → single
    return LayoutMode.SINGLE


def _smooth_layout_votes(
    votes: list,
    min_duration: float = 2.0,
) -> list:
    """Smooth raw per-frame layout votes into stable segments.

    Uses majority voting within sliding windows to prevent rapid switching.
    Short segments (<min_duration) are absorbed into their neighbors.
    """
    if not votes:
        return []

    # Build raw segments from consecutive same-mode votes
    raw_segments = []
    current_mode = votes[0][1]
    current_start = votes[0][0]

    for i in range(1, len(votes)):
        t, mode = votes[i]
        if mode != current_mode:
            raw_segments.append((current_start, t, current_mode))
            current_mode = mode
            current_start = t
    # Final segment
    if votes:
        raw_segments.append((current_start, votes[-1][0] + 0.5, current_mode))

    if not raw_segments:
        return []

    # Helper: merge consecutive same-mode segments
    def _merge_consecutive(segments):
        if not segments:
            return []
        merged = [segments[0]]
        for start, end, mode in segments[1:]:
            if mode == merged[-1][2]:
                merged[-1] = (merged[-1][0], end, mode)
            else:
                merged.append((start, end, mode))
        return merged

    # Deterministic two-pass absorb: O(n), no loops, no `changed` flag.
    # Forward pass: walk left-to-right, absorb short segments into predecessor.
    # Backward pass: walk right-to-left, absorb remaining short segments.
    stabilized = _merge_consecutive(raw_segments)

    # Forward pass
    forward = [stabilized[0]]
    for i in range(1, len(stabilized)):
        start, end, mode = stabilized[i]
        duration = end - start
        if duration < min_duration and len(stabilized) > 1:
            # Extend predecessor
            prev = forward[-1]
            forward[-1] = (prev[0], end, prev[2])
        else:
            forward.append((start, end, mode))
    forward = _merge_consecutive(forward)

    # Backward pass
    backward = [forward[-1]]
    for i in range(len(forward) - 2, -1, -1):
        start, end, mode = forward[i]
        duration = end - start
        if duration < min_duration and len(forward) > 1:
            # Extend successor (which is backward[0] since we're going right-to-left)
            succ = backward[0]
            backward[0] = (start, succ[1], succ[2])
        else:
            backward.insert(0, (start, end, mode))
    stabilized = _merge_consecutive(backward)

    return stabilized


def build_layout_timeline(
    face_results: list,
    face_registry,
    active_speaker_events: list,
    scene_descriptions: list = None,
    clip_start: float = 0,
    clip_end: float = 0,
    min_segment_duration: float = 2.0,
    prefer_single: bool = False,
) -> LayoutTimeline:
    """Build a layout timeline for a clip.

    Algorithm:
    1. For each dense frame, count faces and identify who's speaking
    2. Build raw layout votes per frame based on face count + speaker state
    3. Apply temporal smoothing: don't switch modes for <min_segment_duration
    4. Merge consecutive segments with the same mode
    5. Add transition markers at segment boundaries
    """
    from backend.models import LayoutMode
    from backend.services.active_speaker import get_active_slot_at_time

    if prefer_single or not face_results or not ALLOW_MULTI_LAYOUT:
        return LayoutTimeline(
            segments=[LayoutSegment(
                start=clip_start, end=clip_end,
                layout_mode=LayoutMode.SINGLE,
            )],
            default_mode=LayoutMode.SINGLE,
            face_registry=face_registry,
            total_layout_changes=0,
        )

    # Build screen content map from scene descriptions
    screen_timestamps = set()
    if scene_descriptions:
        for scene in scene_descriptions:
            if hasattr(scene, 'has_screen_content') and scene.has_screen_content:
                screen_timestamps.add(round(scene.timestamp, 1))

    # Build raw votes
    votes = []
    for fr in face_results:
        if clip_start <= fr.timestamp <= clip_end:
            active_slot = get_active_slot_at_time(active_speaker_events, fr.timestamp)
            has_screen = round(fr.timestamp, 1) in screen_timestamps
            mode = _vote_layout_for_frame(fr, active_slot, has_screen, face_registry)
            votes.append((fr.timestamp, mode))

    if not votes:
        return LayoutTimeline(
            segments=[LayoutSegment(
                start=clip_start, end=clip_end,
                layout_mode=LayoutMode.SINGLE,
            )],
            default_mode=LayoutMode.SINGLE,
            face_registry=face_registry,
            total_layout_changes=0,
        )

    # Smooth votes into stable segments
    smoothed = _smooth_layout_votes(votes, min_segment_duration)

    # Build LayoutSegments
    segments = []
    for start, end, mode in smoothed:
        seg = LayoutSegment(
            start=start,
            end=end,
            layout_mode=mode,
            transition_type="cut" if not segments else "dissolve",
        )

        # Assign face slots for SPLIT mode
        if mode == LayoutMode.SPLIT and face_registry and face_registry.multi_speaker:
            sorted_slots = sorted(face_registry.slots, key=lambda s: s.x_center)
            if len(sorted_slots) >= 2:
                seg.left_face_slot = sorted_slots[0].slot_id
                seg.right_face_slot = sorted_slots[1].slot_id

        # Build face positions for this segment
        face_pos = []
        for fr in face_results:
            if start <= fr.timestamp < end:
                for f in fr.faces:
                    face_pos.append({
                        "timestamp": fr.timestamp,
                        "identity_id": f.identity_id,
                        "x": round(f.nose_x, 1),
                        "y": round(f.nose_y, 1),
                        "w": round(f.width, 1),
                        "h": round(f.height, 1),
                        "is_speaking": f.is_speaking,
                    })
        seg.face_positions = face_pos

        segments.append(seg)

    # Determine default mode (most common by duration)
    mode_durations: dict[str, float] = {}
    for seg in segments:
        dur = seg.end - seg.start
        mode_durations[seg.layout_mode] = mode_durations.get(seg.layout_mode, 0) + dur
    default_mode = max(mode_durations, key=mode_durations.get) if mode_durations else LayoutMode.SINGLE

    total_changes = max(0, len(segments) - 1)

    timeline = LayoutTimeline(
        segments=segments,
        default_mode=default_mode,
        face_registry=face_registry,
        total_layout_changes=total_changes,
    )

    logger.info(
        "[Layout] Timeline: default=%s, %d segments, %d changes, modes=%s",
        default_mode, len(segments), total_changes,
        dict(Counter(s.layout_mode for s in segments)),
    )

    return timeline


# ─────────────────────── Camera Solver Integration ────────────────────────

def plan_layout(
    video_path: str,
    frame_faces: list,
    face_registry,
    active_speaker_events: list,
    source_width: int = 1920,
    source_height: int = 1080,
    scene_descriptions: list = None,
    video_duration: float = 0,
    job_id: str = "",
    content_type=None,
    persistent_regions=None,
    frame_objects: list = None,
    frame_saliency: list = None,
    frame_persons: list = None,
) -> LayoutTimeline:
    """Solver-first layout planner with content-type-aware routing.

    Runs shot detection → required regions → per-shot camera solver.
    For solver-handled shots (STATIONARY/PANNING/TRACKING), emits SINGLE
    layout with the solver's keyframes stored in face_positions.
    For PADDED shots, routes by content_type to specialized layout helpers.

    Wrapped in a try/except: on any failure, falls back to the existing
    build_layout_timeline().
    """
    try:
        return _plan_layout_impl(
            video_path, frame_faces, face_registry, active_speaker_events,
            source_width, source_height, scene_descriptions, video_duration,
            job_id, content_type, persistent_regions, frame_objects,
            frame_saliency, frame_persons,
        )
    except Exception as e:
        logger.warning("[%s] plan_layout failed (%s) — falling back to legacy layout", job_id, e)
        clip_end = video_duration or (frame_faces[-1].timestamp if frame_faces else 0)
        return build_layout_timeline(
            face_results=frame_faces,
            face_registry=face_registry,
            active_speaker_events=active_speaker_events,
            scene_descriptions=scene_descriptions,
            clip_start=0,
            clip_end=clip_end,
        )


def _plan_layout_impl(
    video_path, frame_faces, face_registry, active_speaker_events,
    source_width, source_height, scene_descriptions, video_duration, job_id,
    content_type, persistent_regions, frame_objects, frame_saliency,
    frame_persons=None,
):
    from backend.models import LayoutMode
    from backend.services.shot_detector import detect_shots
    from backend.services.camera_solver import (
        solve_all_shots, CameraMode, get_params_for_content_type,
    )
    from backend.services.required_regions import (
        build_required_regions, promote_preferred_to_required,
    )

    # Get content-type-specific solver params for shot detection threshold
    params = get_params_for_content_type(content_type) if content_type else None
    shot_threshold = params.shot_threshold if params else 27.0

    # 1. Detect shots (with content-type-tuned threshold)
    shots = detect_shots(video_path, threshold=shot_threshold,
                         video_duration=video_duration)
    logger.info("[%s] plan_layout: %d shots detected (threshold=%.0f)",
                job_id, len(shots), shot_threshold)

    # ── Anime shot detector augmentation (Week 2 Part B) ──
    # When the ClipContentType resolves to an animation variant, run
    # the histogram / edge-density detector and splice its cuts into
    # the live-action Shot list. Each new cut at time t finds the
    # containing Shot and splits it into two new Shot objects; indices
    # get renumbered after all splits land.
    _is_animated_ct = False
    try:
        _ct_val = getattr(content_type, "value", content_type)
        _is_animated_ct = _ct_val in ("animation", "animation_dialogue")
    except Exception:
        _is_animated_ct = False

    if _is_animated_ct:
        try:
            from backend.services.anime_shot_detector import (
                USE_ANIME_SHOT_DETECTOR,
                detect_anime_shots,
            )
            from backend.services.shot_detector import Shot as _Shot
        except ImportError:
            USE_ANIME_SHOT_DETECTOR = False
            _Shot = None

        if USE_ANIME_SHOT_DETECTOR and _Shot is not None:
            try:
                anime_result = detect_anime_shots(
                    video_path, video_duration=video_duration,
                )
                if (
                    anime_result.cut_times
                    and not anime_result.skipped_reason
                ):
                    existing_bounds = (
                        {round(s.start, 2) for s in shots}
                        | {round(s.end, 2) for s in shots}
                    )
                    merged_shots = list(shots)
                    for t in anime_result.cut_times:
                        if any(abs(t - b) < 0.30 for b in existing_bounds):
                            continue
                        # Find the containing Shot and split.
                        for idx, sh in enumerate(merged_shots):
                            if sh.start < t < sh.end:
                                new_left = _Shot(
                                    index=sh.index,
                                    start=sh.start,
                                    end=float(t),
                                    detector_confidence="high",
                                )
                                new_right = _Shot(
                                    index=sh.index + 1,
                                    start=float(t),
                                    end=sh.end,
                                    detector_confidence="high",
                                )
                                merged_shots[idx:idx + 1] = [
                                    new_left, new_right,
                                ]
                                existing_bounds.add(round(t, 2))
                                break
                    # Re-number indices after all splits.
                    for i, sh in enumerate(merged_shots):
                        sh.index = i
                    added = len(merged_shots) - len(shots)
                    if added:
                        logger.info(
                            "[%s] Anime shot detector added %d cuts "
                            "(now %d shots)",
                            job_id, added, len(merged_shots),
                        )
                    shots = merged_shots
            except Exception as e:
                logger.warning(
                    "[%s] Anime shot detector (AUTOFLIP path) failed: %s",
                    job_id, e,
                )

    # Extract shot cut timestamps for the AttentionAnchor stream so the
    # boxcar smoother breaks at each cut.
    _shot_cut_times = [float(getattr(s, "start", 0.0)) for s in shots if getattr(s, "start", 0.0) > 0]

    # 2. Build required regions (with optional object/saliency/person fusion)
    # v4: frame_persons feeds the AttentionAnchor person_body fallback for
    # faceless dialogue/action frames so the camera anchors on a real
    # subject instead of decay or background saliency.
    regions = build_required_regions(
        frame_faces, active_speaker_events,
        frame_objects=frame_objects,
        frame_saliency=frame_saliency,
        content_type=content_type,
        shot_cuts=_shot_cut_times,
        frame_persons=frame_persons,
    )

    # Promote preferred → required for frames with no faces
    # (critical for anime and gameplay where saliency is load-bearing).
    # For TALKING_HEAD / CINEMATIC_DIALOGUE this is a no-op — a no-face
    # frame means "wait", not "invent a saliency anchor".
    promote_preferred_to_required(regions, content_type=content_type)

    # Fix 5: panel-aware short-shot override.
    # For MULTI_SPEAKER_PANEL content, when a shot is <20 frames AND the
    # active speaker is known for that shot's time range, skip the L1
    # solver entirely and emit a static crop at the active-speaker
    # slot's mean_x. Short cut-between-speakers shots were the dominant
    # source of padded-fallback on the Verzuz run — 57%. A stationary
    # crop on a known seat is always better than padded.
    from backend.services.content_classifier import ClipContentType
    from backend.services.camera_solver import (
        ShotCamera as _ShotCamera,
        CameraMode as _CameraMode,
        CROP_ASPECT as _CROP_ASPECT,
    )
    is_panel = (
        content_type == ClipContentType.MULTI_SPEAKER_PANEL
        if content_type else False
    )
    _panel_short_shot_overrides: dict = {}
    # v2 Phase 11 (Fix 3): gate the short-shot override on shot-detector
    # confidence. The opencv frame-diff fallback fires on static panel
    # content (lighting flicker, compression noise) — 179 "shots" on a
    # 10-minute clip. Each fake shot was getting hard-pinned to an active
    # speaker slot via a 2-keyframe stationary crop, bypassing the L1
    # solver and reintroducing the 2-Hz stair-stepping this phase was
    # supposed to eliminate. If ANY shot is low-confidence, skip the
    # override and let the L1 solver handle smoothing.
    # v4: only skip the override when >30% of shots are low-confidence.
    # A single flicker shot was killing the panel override for the whole
    # clip; allowing up to 30% low-conf tolerates lighting noise while
    # still falling back to the L1 solver when the shot detector is
    # genuinely unreliable.
    _low_conf_count = sum(
        1 for s in shots
        if getattr(s, "detector_confidence", "high") == "low"
    )
    _low_conf_ratio = _low_conf_count / max(len(shots), 1)
    _any_low_conf_shots = _low_conf_ratio > 0.30
    if _any_low_conf_shots:
        logger.info(
            "[%s] [Layout+Solver] panel short-shot override SKIPPED: "
            "%.0f%% low-conf shots (%d/%d). Trusting L1 solver.",
            job_id, _low_conf_ratio * 100, _low_conf_count, len(shots),
        )
    if is_panel and not _any_low_conf_shots and face_registry and getattr(face_registry, "slots", None):
        _source_aspect = (
            source_width / source_height if source_height > 0 else 16 / 9
        )
        _crop_half = (_CROP_ASPECT / _source_aspect) / 2.0

        def _slot_center_normalized(slot_id: int):
            for _s in face_registry.slots:
                if _s.slot_id == slot_id:
                    # FaceSlot.x_center is 0-100 percent; normalize to 0-1.
                    return max(_crop_half, min(1.0 - _crop_half,
                                                float(_s.x_center) / 100.0))
            return None

        def _active_slot_in_range(t_start: float, t_end: float):
            if not active_speaker_events:
                return None
            # Pick the slot that covers the largest share of the shot.
            coverage: dict = {}
            for ev in active_speaker_events:
                ov_start = max(t_start, ev.start)
                ov_end = min(t_end, ev.end)
                if ov_end > ov_start and ev.slot_id >= 0:
                    coverage[ev.slot_id] = (
                        coverage.get(ev.slot_id, 0.0) + (ov_end - ov_start)
                    )
            if not coverage:
                return None
            return max(coverage, key=coverage.get)

        _panel_short_threshold_frames = 20
        for _sh in shots:
            _shot_duration = float(getattr(_sh, "end", 0.0)) - float(getattr(_sh, "start", 0.0))
            if _shot_duration <= 0:
                continue
            # Count required-region frames inside this shot as "frames".
            _shot_n = sum(
                1 for regs in regions
                if regs and _sh.start <= regs[0].timestamp <= _sh.end
            )
            if _shot_n >= _panel_short_threshold_frames:
                continue
            _active_slot = _active_slot_in_range(_sh.start, _sh.end)
            if _active_slot is None:
                continue
            _cx_norm = _slot_center_normalized(_active_slot)
            if _cx_norm is None:
                continue
            _panel_short_shot_overrides[_sh.index] = _ShotCamera(
                shot_index=_sh.index,
                start=_sh.start,
                end=_sh.end,
                mode=_CameraMode.STATIONARY,
                keyframes=[
                    (_sh.start, _cx_norm, 0.5),
                    (_sh.end, _cx_norm, 0.5),
                ],
                reason=f"panel_short_shot_slot{_active_slot}",
                zoom=1.0,
            )
        if _panel_short_shot_overrides:
            logger.info(
                "[%s] [Layout+Solver] panel short-shot override: %d shots "
                "(< %d frames) mapped to active-speaker slot",
                job_id, len(_panel_short_shot_overrides),
                _panel_short_threshold_frames,
            )

    # 3. Solve camera mode per shot (with content-type tuning)
    shot_cameras = solve_all_shots(
        shots, regions,
        source_width=source_width,
        source_height=source_height,
        content_type=content_type,
        job_id=job_id,
    )
    # Fix 5: apply the precomputed panel short-shot overrides so they
    # bypass the L1 solver entirely. Keeping the solve_all_shots call
    # above lets us still log the full mode breakdown for the shots
    # that DIDN'T qualify for the override.
    if _panel_short_shot_overrides:
        shot_cameras = [
            _panel_short_shot_overrides.get(sc.shot_index, sc)
            for sc in shot_cameras
        ]

    # 4. Build LayoutSegments
    segments = []
    padded_count = 0

    for sc in shot_cameras:
        if sc.mode in (
            CameraMode.STATIONARY,
            CameraMode.STATIONARY_ZOOMED,  # v4: discrete-zoom variant of STATIONARY
            CameraMode.PANNING,
            CameraMode.TRACKING,
        ):
            # Solver output → SINGLE layout with keyframes. v4: STATIONARY_ZOOMED
            # carries an extra .zoom factor (1.1/1.2/1.3) that the renderer
            # multiplies by the base crop width.
            kf_positions = [
                {
                    "timestamp": float(t),
                    "x": float(cx * 100),
                    "y": float(cy * 100),
                    "solver_mode": sc.mode.value,
                    "solver_zoom": float(getattr(sc, "zoom", 1.0)),
                }
                for t, cx, cy in sc.keyframes
            ]
            seg = LayoutSegment(
                start=sc.start,
                end=sc.end,
                layout_mode=LayoutMode.SINGLE,
                face_positions=kf_positions,
                transition_type="cut" if not segments else "dissolve",
            )
            segments.append(seg)
        else:
            # PADDED → route by content type
            padded_count += 1
            seg = _build_padded_segment(
                sc, content_type, frame_faces, face_registry,
                active_speaker_events, scene_descriptions, regions,
                source_width, source_height,
            )
            if isinstance(seg, list):
                segments.extend(seg)
            else:
                segments.append(seg)

    segments.sort(key=lambda s: s.start)

    mode_durations = {}
    for seg in segments:
        dur = seg.end - seg.start
        mode_durations[seg.layout_mode] = mode_durations.get(seg.layout_mode, 0) + dur
    default_mode = max(mode_durations, key=mode_durations.get) if mode_durations else LayoutMode.SINGLE

    total_changes = max(0, len(segments) - 1)
    timeline = LayoutTimeline(
        segments=segments,
        default_mode=default_mode,
        face_registry=face_registry,
        total_layout_changes=total_changes,
    )

    solver_handled = sum(1 for sc in shot_cameras if sc.mode != CameraMode.PADDED)
    logger.info(
        "[%s] [Layout+Solver] %d segments, %d solver-handled, %d padded-fallback, "
        "content_type=%s, default=%s",
        job_id, len(segments), solver_handled, padded_count,
        content_type.value if content_type else "none", default_mode,
    )

    return timeline


# ─────────────── Content-type-aware PADDED routing ────────────────────────

def _build_padded_segment(
    sc, content_type, frame_faces, face_registry,
    active_speaker_events, scene_descriptions, regions_per_frame,
    source_width, source_height,
):
    """Route a PADDED shot to a content-type-appropriate layout.

    Returns a single LayoutSegment or a list of LayoutSegments.
    """
    from backend.models import LayoutMode

    try:
        from backend.services.content_classifier import ClipContentType
    except ImportError:
        ClipContentType = None

    # Content-type-specific routing
    if content_type and ClipContentType:
        # CINEMATIC_DIALOGUE / ANIMATION_DIALOGUE: never SPLIT, never PIP.
        # Always follow the active speaker as a single-subject crop. A
        # split-screen layout on a drama or anime dialogue scene looks
        # like a zoom call and kills the cinematography. PADDING
        # (blur-fill) is acceptable when the active speaker's bbox alone
        # doesn't fit 9:16.
        if content_type in (
            ClipContentType.CINEMATIC_DIALOGUE,
            ClipContentType.ANIMATION_DIALOGUE,
        ):
            return LayoutSegment(
                start=sc.start, end=sc.end,
                layout_mode=LayoutMode.SINGLE,
                transition_type="cut",
            )

        if content_type == ClipContentType.TALKING_HEAD:
            # Debates/podcasts: SPLIT layout with two speakers
            pad_faces = [fr for fr in frame_faces if sc.start <= fr.timestamp < sc.end]
            if pad_faces and face_registry and face_registry.multi_speaker and ALLOW_MULTI_LAYOUT:
                sub = build_layout_timeline(
                    face_results=pad_faces,
                    face_registry=face_registry,
                    active_speaker_events=active_speaker_events,
                    scene_descriptions=scene_descriptions,
                    clip_start=sc.start,
                    clip_end=sc.end,
                )
                return sub.segments

        elif content_type == ClipContentType.STREAM:
            # Twitch streams: gameplay PIP layout (facecam in corner)
            if ALLOW_MULTI_LAYOUT:
                return LayoutSegment(
                    start=sc.start, end=sc.end,
                    layout_mode="gameplay",
                    transition_type="cut",
                )

        elif content_type == ClipContentType.GAMEPLAY:
            # Pure gameplay: center crop
            return LayoutSegment(
                start=sc.start, end=sc.end,
                layout_mode=LayoutMode.SINGLE,
                transition_type="cut",
            )

        elif content_type == ClipContentType.ANIMATION:
            # Anime: saliency-dominant crop (no splitting)
            seg = _build_saliency_dominant_crop(
                sc, regions_per_frame, source_width, source_height,
            )
            if seg:
                return seg

        elif content_type == ClipContentType.MUSIC_VIDEO:
            # Music video: center crop (widened aspect handled downstream)
            return LayoutSegment(
                start=sc.start, end=sc.end,
                layout_mode=LayoutMode.SINGLE,
                transition_type="cut",
            )

    # Generic fallback: existing layout voter or SINGLE
    pad_faces = [fr for fr in frame_faces if sc.start <= fr.timestamp < sc.end]
    if pad_faces and ALLOW_MULTI_LAYOUT:
        sub = build_layout_timeline(
            face_results=pad_faces,
            face_registry=face_registry,
            active_speaker_events=active_speaker_events,
            scene_descriptions=scene_descriptions,
            clip_start=sc.start,
            clip_end=sc.end,
        )
        return sub.segments

    return LayoutSegment(
        start=sc.start, end=sc.end,
        layout_mode=LayoutMode.SINGLE,
        transition_type="cut",
    )


def _build_saliency_dominant_crop(sc, regions_per_frame, source_width, source_height):
    """For anime PADDED shots: pick the single highest-score region per frame
    and track it, rather than splitting (anime doesn't have two speakers).
    """
    from backend.models import LayoutMode

    shot_regions = []
    for regs in regions_per_frame:
        if not regs:
            continue
        if not (sc.start <= regs[0].timestamp <= sc.end):
            continue
        best = max(regs, key=lambda r: r.score)
        shot_regions.append(best)

    if not shot_regions:
        return None

    kf_positions = [
        {
            "timestamp": float(r.timestamp),
            "x": float(r.cx * 100),
            "y": float(r.cy * 100),
            "solver_mode": "saliency_dominant",
        }
        for r in shot_regions
    ]

    return LayoutSegment(
        start=sc.start, end=sc.end,
        layout_mode=LayoutMode.SINGLE,
        face_positions=kf_positions,
        transition_type="cut",
    )


def layout_from_reframe_segments(
    reframe_segments: list,
    face_registry,
    source_width: int,
    source_height: int,
    job_id: str = "",
) -> LayoutTimeline:
    """Build a LayoutTimeline directly from ReframeSegments.

    v2 Phase 11 (Fix 7): the ReframeSegmenter already produces a
    content-aware, L1-solved, confidence-gated timeline. The legacy
    ``plan_layout`` path runs its own shot detection + required-regions
    + camera solver pass, which on the Verzuz/Tank-Tyrese clip
    produced a second, incompatible decomposition (179 opencv-fallback
    "shots", a panel short-shot override that hard-pinned 165 of them
    to single-slot centers, and a different content_type than what
    the segmenter was using). The two paths running in series means
    the final keyframes come from whichever one wrote last — with
    no smoothing guarantees.

    This adapter emits one ``LayoutSegment`` per ``ReframeSegment``
    without re-running any detection, re-solving, or re-smoothing.
    ``ReframeSegment.subject_x`` / ``subject_y`` are in source-pixel
    space; this function converts to the 0-100 normalized coordinate
    that downstream consumers expect in ``face_positions``.

    Any ``ReframeSegment.motion_path`` values (already L1-solved for
    tracking segments) are threaded through as keyframes so the
    renderer gets smoothing without a second solver pass.
    """
    from backend.models import LayoutMode

    if not reframe_segments:
        return LayoutTimeline(
            segments=[],
            default_mode="single",
            face_registry=face_registry,
            total_layout_changes=0,
        )

    _sw = max(source_width, 1)
    _sh = max(source_height, 1)

    # ReframeSegment.layout → LayoutMode
    _layout_map = {
        "single": LayoutMode.SINGLE,
        "split": LayoutMode.SPLIT,
        "triple": LayoutMode.TRIPLE,
        "wide_master": LayoutMode.SINGLE,   # Renderer treats wide_master
        "blur_fill": LayoutMode.SINGLE,     # and blur_fill as SINGLE + bg
        "stacked_gameplay": LayoutMode.GAMEPLAY,
        "grid": LayoutMode.TRIPLE,
    }

    segments: list = []
    layout_changes = 0
    prev_mode: str = None

    for rseg in reframe_segments:
        _layout_mode = _layout_map.get(
            getattr(rseg, "layout", "single"), LayoutMode.SINGLE,
        )

        # Build keyframes. Use motion_path if present (tracking/panning
        # already L1-solved in the segmenter), otherwise emit stationary
        # start/end kf pair at the subject_x/y.
        motion_path = getattr(rseg, "motion_path", None)
        if motion_path:
            _default_py_pct = float(rseg.subject_y / _sh * 100.0)
            kf_positions = []
            for _mp_entry in motion_path:
                _mp_t = float(_mp_entry[0])
                _mp_px = float(_mp_entry[1])
                # l1_camera_path returns 2-tuples (t, x); optical_flow returns
                # 3-tuples (t, x, y). Handle both without crashing.
                _mp_py_pct = (
                    float(_mp_entry[2] / _sh * 100.0)
                    if len(_mp_entry) > 2
                    else _default_py_pct
                )
                kf_positions.append({
                    "timestamp": _mp_t,
                    "x": float(_mp_px / _sw * 100.0),
                    "y": _mp_py_pct,
                    "solver_mode": str(getattr(rseg, "strategy", "stationary")),
                    "solver_zoom": 1.0,
                })
        else:
            _cx = float(rseg.subject_x / _sw * 100.0)
            _cy = float(rseg.subject_y / _sh * 100.0)
            kf_positions = [
                {
                    "timestamp": float(rseg.start),
                    "x": _cx, "y": _cy,
                    "solver_mode": str(getattr(rseg, "strategy", "stationary")),
                    "solver_zoom": 1.0,
                },
                {
                    "timestamp": float(rseg.end),
                    "x": _cx, "y": _cy,
                    "solver_mode": str(getattr(rseg, "strategy", "stationary")),
                    "solver_zoom": 1.0,
                },
            ]

        # Count layout transitions.
        if prev_mode is not None and _layout_mode != prev_mode:
            layout_changes += 1
        prev_mode = _layout_mode

        seg = LayoutSegment(
            start=float(rseg.start),
            end=float(rseg.end),
            layout_mode=_layout_mode,
            face_positions=kf_positions,
            transition_type="cut" if not segments else "dissolve",
        )
        segments.append(seg)

    # Determine default mode as the most-used mode.
    if segments:
        mode_counts = Counter(s.layout_mode for s in segments)
        default_mode = mode_counts.most_common(1)[0][0]
    else:
        default_mode = LayoutMode.SINGLE

    logger.info(
        "[%s] [Layout] built from %d reframe segments (skipped re-detection, "
        "re-solving, and short-shot override); default=%s, transitions=%d",
        job_id, len(reframe_segments), default_mode, layout_changes,
    )

    return LayoutTimeline(
        segments=segments,
        default_mode=default_mode,
        face_registry=face_registry,
        total_layout_changes=layout_changes,
    )
