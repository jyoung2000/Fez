"""Top-level integration for the human-reframing overhaul.

This is the single entry point the pipeline invokes when
``CLIPAI_HUMAN_REFRAME=1``. It threads the Phase 2-9 subsystems
together:

    faces + speakers + motion + content-type
        │
        ├─▶ subject_kalman.build_registry_from_dense_faces
        │
        ├─▶ camera_path_2d.solve_2d_camera_path  (2-D LP with headroom
        │                                         + eye-line + lead-room)
        │
        ├─▶ camera_events.schedule_camera_events (saccade vs pursuit)
        │
        ├─▶ ab_cut_scheduler.plan_ab_schedule    (A/B/C dialogue cuts)
        │
        ├─▶ motivated_zoom.plan_motivated_zooms  (push-in / pull-out)
        │
        ├─▶ genre_refinements.apply_genre_refinements
        │
        └─▶ critic_loop.score_plan               (two-pass VLM/learned)

It returns a :class:`HumanReframePlan` that the existing RenderPlan
builder can consume — a thin adapter turns it into
``list[RenderOp]`` with the right op kinds.

Kept as its own module so the legacy AutoFlip path in
:mod:`backend.services.reframe_segmenter` can remain unchanged for
existing fixtures (``CLIPAI_HUMAN_REFRAME`` off).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from backend.services.ab_cut_scheduler import (
    AbScheduleResult,
    ReactionEvent,
    SpeakerWindow,
    plan_ab_schedule,
)
from backend.services.camera_events import (
    CameraEvent,
    ShotInfo,
    SpeakerTurn,
    ZoomWindow,
    schedule_camera_events,
)
from backend.services.camera_path_2d import (
    CameraPath2D,
    faces_from_dense,
    solve_2d_camera_path,
)
from backend.services.genre_refinements import (
    GenreInputs,
    GenreRefinementResult,
    apply_genre_refinements,
)
from backend.services.motivated_zoom import (
    MotionBeat,
    SubjectEntrance,
    WordTiming,
    ZoomMoment,
    plan_motivated_zooms,
)
from backend.services.reframe_config import ReframeConfig, get_default_config
from backend.services.shot_classifier import ShotProfile, classify_shots
from backend.services.subject_kalman import (
    SubjectKalmanRegistry,
    build_registry_from_dense_faces,
)

logger = logging.getLogger(__name__)


@dataclass
class HumanReframePlan:
    """Top-level output the RenderPlan builder consumes."""

    path: CameraPath2D
    events: list[CameraEvent]
    ab: AbScheduleResult
    zooms: list[ZoomMoment]
    genre: GenreRefinementResult
    kalman: SubjectKalmanRegistry
    notes: list[str] = field(default_factory=list)
    # Fix 3.2: per-shot content classification. Empty when no shot
    # boundaries were passed in. The adapter uses this to emit per-op
    # content_type telemetry and (once wired) to pick per-op configs.
    shot_profiles: list[ShotProfile] = field(default_factory=list)


@dataclass
class HumanReframeInputs:
    """Bundle of inputs from the pipeline (all optional where sensible)."""

    duration_sec: float
    source_w: int
    source_h: int
    content_type: str = "other"

    dense_faces: list = field(default_factory=list)
    active_speaker_events: list = field(default_factory=list)
    shot_boundaries: list[float] = field(default_factory=list)

    # Optional: beats for music-video genre refinements
    beats: list[float] = field(default_factory=list)
    downbeats: list[float] = field(default_factory=list)

    # Optional: words + audio peaks for motivated zoom
    words: list[WordTiming] = field(default_factory=list)
    audio_peaks: list = field(default_factory=list)
    motion_beats: list[MotionBeat] = field(default_factory=list)
    entrances: list[SubjectEntrance] = field(default_factory=list)

    # Optional: audio-analyzer reactions (laughter / gasp) for A/B cuts
    reactions: list[ReactionEvent] = field(default_factory=list)

    # Optional: ball / car / HUD for genre refinements
    ball_detections: list = field(default_factory=list)
    car_detections: list = field(default_factory=list)
    hud_regions: list[dict] = field(default_factory=list)
    impact_frames: list[float] = field(default_factory=list)

    # Phase 3 (reframing overhaul): per-shot strategy advice. When any
    # shot's strategy is SPEAKER_ALTERNATING the A/B scheduler delegates
    # cut timing to the speaker_cut_engine. ``speaker_positions`` is a
    # ``slot_id -> x_frac`` map used by the engine to know where to
    # crop on each cut.
    shot_advice_list: list = field(default_factory=list)
    speaker_positions: dict = field(default_factory=dict)


def _speaker_windows_from_events(events: list) -> list[SpeakerWindow]:
    out: list[SpeakerWindow] = []
    for e in events or []:
        slot = getattr(e, "slot_id", -1)
        if slot < 0:
            continue
        out.append(SpeakerWindow(
            start=float(getattr(e, "start", 0.0)),
            end=float(getattr(e, "end", 0.0)),
            slot_id=int(slot),
        ))
    return out


def _speaker_turns_from_events(events: list) -> list[SpeakerTurn]:
    out: list[SpeakerTurn] = []
    prev_slot = -1
    for e in sorted(events or [], key=lambda x: getattr(x, "start", 0.0)):
        slot = getattr(e, "slot_id", -1)
        if slot < 0 or slot == prev_slot:
            continue
        out.append(SpeakerTurn(
            t=float(getattr(e, "start", 0.0)),
            from_slot=prev_slot,
            to_slot=int(slot),
        ))
        prev_slot = slot
    return out


def _timestamps_from_dense(dense_faces: list) -> list[float]:
    return [float(ff.timestamp) for ff in dense_faces]


def run_human_reframe(
    inputs: HumanReframeInputs,
    config: Optional[ReframeConfig] = None,
) -> HumanReframePlan:
    """Run the full human-reframe pipeline. Returns a plan."""
    config = (config or get_default_config()).for_content(inputs.content_type)

    # 0. Fix 3.2: per-shot classification. Each shot gets its own
    # ShotProfile that drives the downstream config choice. For now
    # the 2-D solver still runs once over the full clip (stitching
    # per-shot paths is a future task), but the per-shot information
    # flows through HumanReframePlan.shot_profiles so the adapter
    # and critic can read it.
    from backend.services.content_classifier import ContentProfile
    clip_prior = ContentProfile(
        content_type=inputs.content_type or "unknown",
        confidence=0.5,
    )
    shot_profiles = classify_shots(
        shot_boundaries=inputs.shot_boundaries,
        dense_faces=inputs.dense_faces,
        duration_sec=inputs.duration_sec,
        clip_profile=clip_prior,
    )

    # 1. Kalman registry (predictive motion).
    registry = build_registry_from_dense_faces(
        inputs.dense_faces, inputs.active_speaker_events, config=config,
    )

    # 2. 2-D camera path.
    timestamps = _timestamps_from_dense(inputs.dense_faces)
    faces_2d = faces_from_dense(
        inputs.dense_faces,
        active_speaker_events=inputs.active_speaker_events,
    )
    # Fix 3.4: build a primary-slot map keyed on timestamp so the
    # solver can pull the right Kalman-predicted lead per frame.
    primary_slot_by_t: dict[float, int] = {}
    for ev in inputs.active_speaker_events or []:
        slot = getattr(ev, "slot_id", -1)
        if slot < 0:
            continue
        for t in timestamps:
            if ev.start <= t <= ev.end:
                primary_slot_by_t[t] = slot
    path = solve_2d_camera_path(
        faces_2d,
        timestamps=timestamps,
        source_w=inputs.source_w,
        source_h=inputs.source_h,
        config=config,
        predictor=registry,
        primary_slot_by_t=primary_slot_by_t or None,
    )

    # 3. Genre refinements (produces zoom candidates + HUD exclusions).
    genre = apply_genre_refinements(
        GenreInputs(
            content_type=inputs.content_type,
            clip_duration=inputs.duration_sec,
            ball_detections=inputs.ball_detections,
            car_detections=inputs.car_detections,
            beats=inputs.beats,
            downbeats=inputs.downbeats,
            impact_frames=inputs.impact_frames,
            hud_regions=inputs.hud_regions,
            speaker_turns=[t.t for t in _speaker_turns_from_events(
                inputs.active_speaker_events,
            )],
            mean_dwell_sec=0.0,
            shot_boundaries=inputs.shot_boundaries,
        ),
        config=config,
    )

    # 4. A/B cut scheduler over the whole clip's speaker windows.
    sp_windows = _speaker_windows_from_events(inputs.active_speaker_events)
    # Phase 3: when any shot is flagged SPEAKER_ALTERNATING, delegate
    # cut TIMING to speaker_cut_engine; ab_cut_scheduler still owns the
    # motivated zoom + reaction layering on top of that timing.
    _use_scs = False
    try:
        from backend.services.shot_reframe_advisor import ReframeStrategy
        _use_scs = any(
            getattr(a, "strategy", None) == ReframeStrategy.SPEAKER_ALTERNATING
            for a in (inputs.shot_advice_list or [])
        )
    except Exception:
        _use_scs = False
    ab = plan_ab_schedule(
        overlap_start=0.0,
        overlap_end=inputs.duration_sec,
        speaker_windows=sp_windows,
        reactions=inputs.reactions,
        non_speaker_slots_at=lambda _t: [],
        config=config,
        content_type=inputs.content_type,
        use_speaker_cut_engine=_use_scs,
        speaker_positions=inputs.speaker_positions or None,
        source_width=inputs.source_w,
    )

    # 5. Motivated zoom (merged with genre-forced zooms).
    zooms = plan_motivated_zooms(
        clip_duration=inputs.duration_sec,
        words=inputs.words,
        audio_peaks=inputs.audio_peaks,
        motion_beats=inputs.motion_beats,
        entrances=inputs.entrances,
        shot_boundaries=inputs.shot_boundaries,
        content_type=inputs.content_type,
        config=config,
    )
    zooms = list(zooms) + list(genre.forced_zooms)
    zooms.sort(key=lambda z: z.start)

    # 6. Camera event scheduler.
    turns = _speaker_turns_from_events(inputs.active_speaker_events)
    shots = [
        ShotInfo(
            start=(inputs.shot_boundaries[i - 1] if i > 0 else 0.0),
            end=inputs.shot_boundaries[i] if i < len(inputs.shot_boundaries) else inputs.duration_sec,
        )
        for i in range(len(inputs.shot_boundaries) + 1)
    ]
    zoom_windows = [ZoomWindow(start=z.start, end=z.end) for z in zooms]
    # Sample Kalman speeds across the clip for the pursue/hold decision.
    vel_samples: list[tuple[float, float]] = []
    for i, t in enumerate(timestamps):
        speeds = [reg.speed for reg in registry.filters.values()]
        vel_samples.append((t, max(speeds) if speeds else 0.0))

    events = schedule_camera_events(
        duration=inputs.duration_sec,
        shots=shots,
        speaker_turns=turns,
        occlusions=[],
        zoom_windows=zoom_windows,
        velocity_by_t=vel_samples,
        config=config,
    )

    # Inject genre-forced saccade timestamps (e.g. anime impacts, beats)
    # into the event schedule as extra SACCADE markers.
    if genre.saccade_timestamps:
        from backend.services.camera_events import CameraEvent, CameraMode
        for t in genre.saccade_timestamps:
            events.append(CameraEvent(
                mode=CameraMode.SACCADE,
                start=t, end=t,
                ease_ms=config.saccade_ease_ms,
                reason="genre-injected",
            ))
        events.sort(key=lambda e: (e.start, e.end))

    notes = list(genre.notes)
    if ab.enabled:
        notes.append(f"A/B schedule: {len(ab.segments)} sub-segments")
    else:
        notes.append(f"A/B disabled: {ab.fallback_reason}")

    # ── Phase 2 composition guardrails (post-critic, pre-RenderPlan) ──
    # Apply hard compositional rules (headroom, edge margins, lead room,
    # text protection, pan speed, min hold) to the solved 2-D camera
    # path. Best-effort: any failure logs and leaves the path untouched.
    # Gated on the CLIPAI_COMPOSITION_GUARDRAILS env flag (default on).
    try:
        from backend.services.composition_guardrails import (
            CropFrame,
            FrameAnalysis,
            enforce_guardrails,
            guardrails_enabled,
        )
        if guardrails_enabled() and getattr(path, "x", None) is not None:
            xs = getattr(path, "x", None) or []
            ys = getattr(path, "y", None) or []
            ts = list(timestamps)
            crop_w_pred = float(inputs.source_h) * 9.0 / 16.0
            if crop_w_pred > inputs.source_w:
                crop_w_pred = float(inputs.source_w)
            crop_h_pred = float(inputs.source_h)
            crops = []
            analyses = []
            n_path = min(len(xs), len(ts))
            for i in range(n_path):
                cx = float(xs[i])
                cy = float(ys[i]) if i < len(ys) else crop_h_pred / 2.0
                crops.append(CropFrame(
                    t=float(ts[i]),
                    x=max(0.0, min(float(inputs.source_w) - crop_w_pred,
                                   cx - crop_w_pred / 2.0)),
                    y=max(0.0, min(float(inputs.source_h) - crop_h_pred,
                                   cy - crop_h_pred / 2.0)),
                    w=crop_w_pred,
                    h=crop_h_pred,
                ))
                analyses.append(FrameAnalysis(timestamp=float(ts[i])))
            if crops:
                fps_est = 30.0
                if len(ts) >= 2:
                    dts = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)
                           if ts[i + 1] > ts[i]]
                    if dts:
                        fps_est = 1.0 / (sum(dts) / len(dts))
                adjusted, report = enforce_guardrails(
                    crops, analyses,
                    inputs.source_w, inputs.source_h,
                    fps=fps_est, config=config,
                )
                # Write adjusted cx/cy back into the camera path.
                for i, cf in enumerate(adjusted):
                    if i < len(xs):
                        xs[i] = cf.cx
                    if i < len(ys):
                        ys[i] = cf.cy
                if hasattr(path, "x"):
                    try:
                        path.x = xs
                    except Exception:
                        pass
                if hasattr(path, "y"):
                    try:
                        path.y = ys
                    except Exception:
                        pass
                if report.violations_found > 0:
                    notes.append(
                        f"guardrails: {report.violations_found} found, "
                        f"{report.violations_fixed} fixed, "
                        f"{report.violations_unfixable} unfixable"
                    )
                if report.needs_human_review:
                    notes.append("guardrails: needs human review")
    except Exception as gr_exc:
        logger.warning(
            "composition guardrails failed (non-fatal): %s", gr_exc,
        )

    return HumanReframePlan(
        path=path,
        events=events,
        ab=ab,
        zooms=zooms,
        genre=genre,
        kalman=registry,
        notes=notes,
        shot_profiles=shot_profiles,
    )
