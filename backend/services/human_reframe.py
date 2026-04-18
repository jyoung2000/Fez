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
    path = solve_2d_camera_path(
        faces_2d,
        timestamps=timestamps,
        source_w=inputs.source_w,
        source_h=inputs.source_h,
        config=config,
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
    ab = plan_ab_schedule(
        overlap_start=0.0,
        overlap_end=inputs.duration_sec,
        speaker_windows=sp_windows,
        reactions=inputs.reactions,
        non_speaker_slots_at=lambda _t: [],
        config=config,
        content_type=inputs.content_type,
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

    return HumanReframePlan(
        path=path,
        events=events,
        ab=ab,
        zooms=zooms,
        genre=genre,
        kalman=registry,
        notes=notes,
    )
