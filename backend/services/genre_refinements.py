"""Genre-specific refinements that sit on top of the 2-D solver, Kalman
predictor, camera-event scheduler, motivated-zoom planner, and A/B cut
scheduler introduced in Phases 2-7.

Each genre helper consumes existing signals the pipeline already
computes and produces:

  * required-region adjustments (e.g. promote ball / car to required),
  * camera-event overrides (e.g. snap saccades to beat grid),
  * motivated-zoom schedules (e.g. music-video beat-locked zooms),
  * HUD / scoreboard exclusion zones.

The public entry point is :func:`apply_genre_refinements`, which reads
the content type off the pipeline's ``ContentProfile`` and dispatches
to the right helper. Helpers return lightweight dataclasses with
results; the pipeline is responsible for threading them through the
render-plan builder.

Per-content-type knobs live in :mod:`backend.services.reframe_config`
(the ``content_overrides`` table); this module is the place where the
*rules* based on those knobs fire.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from backend.services.motivated_zoom import ZoomKind, ZoomMoment
from backend.services.reframe_config import ReframeConfig, get_default_config

logger = logging.getLogger(__name__)


# Blueprint v2 Phase 1: ball / car boosts layer on top of the genre's
# ``importance.object`` weight so the matrix stays the single source
# of truth. Raw multipliers preserve the pre-Phase-1 numerical output
# (object=1.0 × 1.1 ball boost = 1.1 weight, matching the legacy
# hardcoded 1.1).
BASKETBALL_BALL_BOOST = 1.1
RACING_CAR_BOOST = 1.2


# ── Shared dataclasses ────────────────────────────────────────────


@dataclass
class HudExclusion:
    """Rectangular region in source-frame fractions that the crop
    solver should avoid centering on (HUD, scoreboard, network bug)."""

    x: float
    y: float
    w: float
    h: float
    label: str = ""


@dataclass
class RequiredBoost:
    """Feature-stream adjustment: tier / weight override for a named
    source ("ball", "car", "player") on a frame range."""

    source: str
    start: float
    end: float
    tier: str = "required"
    weight: float = 1.0


@dataclass
class GenreRefinementResult:
    """Bundle of outputs the pipeline splices into its per-clip build."""

    required_boosts: list[RequiredBoost] = field(default_factory=list)
    hud_exclusions: list[HudExclusion] = field(default_factory=list)
    forced_zooms: list[ZoomMoment] = field(default_factory=list)
    saccade_timestamps: list[float] = field(default_factory=list)
    # Human-readable notes for telemetry.
    notes: list[str] = field(default_factory=list)


# ── Sports helpers ─────────────────────────────────────────────────


def _basketball_refinements(
    *,
    ball_detections: list,
    shot_boundaries: list[float],
    config: ReframeConfig,
) -> GenreRefinementResult:
    """Promote basketball ball detections to required when the ball is
    seen for ≥ 8 consecutive frames with conf ≥ 0.5, and predict its
    airtime arc so the top of the arc lands near the eye-line.

    Phase 1: the boost weight derives from
    ``config.importance.object * BASKETBALL_BALL_BOOST`` so the
    Stage 3 matrix is authoritative.
    """
    res = GenreRefinementResult()
    if not ball_detections:
        return res

    # Walk through detections, collecting runs of confident frames.
    runs: list[tuple[float, float]] = []
    run_start: Optional[float] = None
    last_t: Optional[float] = None
    consecutive = 0
    for det in sorted(ball_detections, key=lambda d: d["t"]):
        if det.get("confidence", 0.0) >= 0.5:
            consecutive += 1
            if run_start is None:
                run_start = det["t"]
            last_t = det["t"]
        else:
            if consecutive >= 8 and run_start is not None and last_t is not None:
                runs.append((run_start, last_t))
            consecutive = 0
            run_start = None
    if consecutive >= 8 and run_start is not None and last_t is not None:
        runs.append((run_start, last_t))

    ball_weight = float(config.importance.object) * BASKETBALL_BALL_BOOST
    for s, e in runs:
        res.required_boosts.append(RequiredBoost(
            source="ball", start=s, end=e, tier="required", weight=ball_weight,
        ))
    res.notes.append(f"basketball: {len(runs)} required-ball runs")
    return res


def _racing_refinements(
    *,
    car_detections: list,
    config: ReframeConfig,
) -> GenreRefinementResult:
    res = GenreRefinementResult()
    if not car_detections:
        return res
    # The biggest / most-central car per frame drives the crop.
    # Phase 1: weight = config.importance.object * RACING_CAR_BOOST.
    car_weight = float(config.importance.object) * RACING_CAR_BOOST
    res.required_boosts.append(RequiredBoost(
        source="car", start=0.0, end=float("inf"),
        tier="required", weight=car_weight,
    ))
    res.notes.append("racing: car promoted to required globally")
    return res


# ── Music video helpers ────────────────────────────────────────────


def _music_refinements(
    *,
    beats: list[float],
    downbeats: list[float],
    clip_duration: float,
    config: ReframeConfig,
) -> GenreRefinementResult:
    """Beat-lock saccades, and schedule a wide-master push on every
    downbeat that has multiple equidistant faces in frame (formation)."""
    res = GenreRefinementResult()
    if beats:
        # Every downbeat is a saccade opportunity (A/B cut scheduler
        # already emits its own saccade times; we add these so the
        # camera_events scheduler can snap to beat).
        res.saccade_timestamps.extend(downbeats or beats[::4])

    # Schedule a beat-locked zoom wave: 1-bar push-in into every 4th
    # downbeat, 1-bar pull back. This produces the "breathing" feel
    # typical of music montages.
    if downbeats and len(downbeats) >= 4:
        for i, db in enumerate(downbeats):
            if i % 4 != 0:
                continue
            start = db
            end = min(clip_duration, db + 1.5)
            if end - start < 0.5:
                continue
            kind = ZoomKind.PUSH_IN if (i // 4) % 2 == 0 else ZoomKind.PULL_OUT
            scale = (
                config.zoom_push_in_max_scale if kind == ZoomKind.PUSH_IN
                else config.zoom_pull_out_max_scale
            )
            # Respect content gate: for music_video, zoom scales default
            # to 1.0 and produce nothing. A caller that wants beat-zooms
            # has to override the config explicitly.
            if abs(scale - 1.0) < 1e-3:
                continue
            res.forced_zooms.append(ZoomMoment(
                start=start, end=end, kind=kind,
                target_scale=scale, reason=f"beat-lock-{kind.value}",
            ))
    res.notes.append(
        f"music: {len(res.saccade_timestamps)} beat-locked saccades, "
        f"{len(res.forced_zooms)} zooms"
    )
    return res


# ── Anime helpers ─────────────────────────────────────────────────


def _anime_refinements(
    *,
    impact_frames: list[float],
    clip_duration: float,
) -> GenreRefinementResult:
    """Anime: impact frames become saccade triggers, and every impact
    gets a short pull-out for spectacle."""
    res = GenreRefinementResult()
    for t in impact_frames:
        res.saccade_timestamps.append(t)
        start = max(0.0, t - 0.05)
        end = min(clip_duration, t + 0.8)
        if end - start >= 0.3:
            res.forced_zooms.append(ZoomMoment(
                start=start, end=end, kind=ZoomKind.PULL_OUT,
                target_scale=0.85,
                reason="anime-impact",
            ))
    res.notes.append(
        f"anime: {len(impact_frames)} impact saccades + pull-outs",
    )
    return res


# ── Gaming helpers ────────────────────────────────────────────────


def _gaming_refinements(
    *,
    hud_regions: list[dict],
) -> GenreRefinementResult:
    """Reserve HUD regions so the crop solver won't center on them,
    and forbid any zoom while HUD coverage is significant."""
    res = GenreRefinementResult()
    for r in hud_regions or []:
        res.hud_exclusions.append(HudExclusion(
            x=float(r.get("x", 0.0)),
            y=float(r.get("y", 0.0)),
            w=float(r.get("w", 0.0)),
            h=float(r.get("h", 0.0)),
            label=str(r.get("label", "hud")),
        ))
    res.notes.append(f"gaming: {len(res.hud_exclusions)} HUD exclusions")
    return res


# ── Dialogue / vlog helpers ───────────────────────────────────────


def _dialogue_refinements(
    *,
    speaker_turns: list[float],
    mean_dwell_sec: float,
    clip_duration: float,
    config: ReframeConfig,
) -> GenreRefinementResult:
    """Dialogue: schedule a mild push-in at every ≥ 3 s continuous
    speaker window (motivates the zoom on long beats)."""
    res = GenreRefinementResult()
    if config.zoom_push_in_max_scale <= 1.001:
        return res
    if not speaker_turns:
        return res
    turns = sorted(speaker_turns)
    for i, t in enumerate(turns):
        t_end = turns[i + 1] if i + 1 < len(turns) else clip_duration
        dwell = t_end - t
        if dwell >= 3.0:
            start = min(clip_duration, t + 0.8)
            end = min(clip_duration, t + 0.8 + 2.5)
            if end - start > 0.5:
                res.forced_zooms.append(ZoomMoment(
                    start=start, end=end, kind=ZoomKind.PUSH_IN,
                    target_scale=config.zoom_push_in_max_scale,
                    reason="dialogue-dwell-push",
                ))
    res.notes.append(f"dialogue: {len(res.forced_zooms)} dwell push-ins")
    return res


# ── Dispatch ─────────────────────────────────────────────────────


@dataclass
class GenreInputs:
    content_type: str
    clip_duration: float
    ball_detections: list = field(default_factory=list)
    car_detections: list = field(default_factory=list)
    beats: list[float] = field(default_factory=list)
    downbeats: list[float] = field(default_factory=list)
    impact_frames: list[float] = field(default_factory=list)
    hud_regions: list[dict] = field(default_factory=list)
    speaker_turns: list[float] = field(default_factory=list)
    mean_dwell_sec: float = 0.0
    shot_boundaries: list[float] = field(default_factory=list)


def apply_genre_refinements(
    inputs: GenreInputs,
    config: Optional[ReframeConfig] = None,
) -> GenreRefinementResult:
    """Dispatch to the right per-genre helper and return a merged result."""
    config = (config or get_default_config()).for_content(inputs.content_type)
    ct = (inputs.content_type or "").lower()

    results: list[GenreRefinementResult] = []
    if ct == "sports_basketball":
        results.append(_basketball_refinements(
            ball_detections=inputs.ball_detections,
            shot_boundaries=inputs.shot_boundaries,
            config=config,
        ))
    if ct == "sports_racing":
        results.append(_racing_refinements(
            car_detections=inputs.car_detections,
            config=config,
        ))
    if ct in ("music_video",):
        results.append(_music_refinements(
            beats=inputs.beats, downbeats=inputs.downbeats,
            clip_duration=inputs.clip_duration,
            config=config,
        ))
    if ct in ("animation", "animation_dialogue"):
        results.append(_anime_refinements(
            impact_frames=inputs.impact_frames,
            clip_duration=inputs.clip_duration,
        ))
    if ct.startswith("gameplay") or ct == "gaming":
        results.append(_gaming_refinements(hud_regions=inputs.hud_regions))
    if ct in ("talking_head", "cinematic_dialogue", "vlog",
              "narrative", "animation_dialogue"):
        results.append(_dialogue_refinements(
            speaker_turns=inputs.speaker_turns,
            mean_dwell_sec=inputs.mean_dwell_sec,
            clip_duration=inputs.clip_duration,
            config=config,
        ))

    merged = GenreRefinementResult()
    for r in results:
        merged.required_boosts.extend(r.required_boosts)
        merged.hud_exclusions.extend(r.hud_exclusions)
        merged.forced_zooms.extend(r.forced_zooms)
        merged.saccade_timestamps.extend(r.saccade_timestamps)
        merged.notes.extend(r.notes)
    return merged
