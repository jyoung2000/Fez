"""Saccade-vs-smooth-pursuit event scheduler.

Phase 4 of the human-reframing overhaul. Replaces the hand-picked ease
durations (``EASE_SHOT_CUT_MS=0``, ``EASE_SPEAKER_TURN_MS=200``) and the
uniform L1 smoothing with a five-state machine that models how a human
camera operator shifts their attention:

  HOLD         — subject isn't moving and nothing asks us to relocate.
                 Tight dead-zone. Camera freezes at sub-pixel.
  SMOOTH_PURSUE — subject is moving continuously (walking, running,
                 racing). L1 velocity penalty relaxes; acceleration
                 penalty tightens. Camera follows without jitter.
  SACCADE      — discrete event demands a fast re-locate: speaker
                 switch, new dominant subject, subject reappears after
                 occlusion, intent switch. Cosine ease of 100-200 ms
                 that the LP is forbidden to smooth through.
  MATCH_CUT    — shot boundary where the next shot's subject is
                 geometrically close to current camera. Maintain
                 framing across the cut instead of recentering.
  MICRO_ZOOM   — the ``motivated_zoom`` scheduler has scheduled a
                 push-in or pull-out that starts here. The camera path
                 is otherwise held to avoid fighting the zoom.

The scheduler consumes:

    * Shot cuts ``list[float]``
    * Speaker events ``list[SpeakerEvent]``
    * Kalman subject velocities ``dict[slot_id, (vx, vy)]`` from
      :mod:`backend.services.subject_kalman`
    * Occlusion / reappearance timestamps
    * Zoom windows from :mod:`backend.services.motivated_zoom`

and emits a ``list[CameraEvent]`` tagged with start, end, mode, and an
``ease_ms`` value the downstream FFmpeg filter / RenderPlan builder uses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from backend.services.reframe_config import ReframeConfig, get_default_config

logger = logging.getLogger(__name__)


class CameraMode(str, Enum):
    HOLD = "hold"
    SMOOTH_PURSUE = "smooth_pursue"
    SACCADE = "saccade"
    MATCH_CUT = "match_cut"
    MICRO_ZOOM = "micro_zoom"


# ── Configuration ─────────────────────────────────────────────────

# Velocity threshold (normalized source-frame units per second) above
# which a subject is considered to be in smooth-pursuit territory.
# Human smooth pursuit tops out around 30°/s; mapped to a 16:9 frame
# with a 50° horizontal FOV that's ~0.6 of frame width per second.
_PURSUE_SPEED_THRESHOLD = 0.15  # 15% of source width per second

# Minimum separation between consecutive saccades in the same segment
# to avoid nausea-inducing flicker. Humans make at most ~3 saccades /
# second; we aim lower since each saccade in our case is ~150 ms long.
_MIN_SACCADE_GAP_SEC = 0.4

# If a shot cut lands closer than this to a scheduled saccade, it is
# absorbed into the saccade (same destination, same ease).
_CUT_ABSORB_WINDOW_SEC = 0.25


@dataclass
class CameraEvent:
    """One state-machine entry in the output schedule."""

    mode: CameraMode
    start: float
    end: float
    # How long the transition INTO this event should take in ms.
    # 0 = hard cut; > 0 = cosine ease.
    ease_ms: int
    # Trigger annotation for debugging / telemetry.
    reason: str = ""
    # For SACCADE: target slot or subject id, when known.
    target_slot: Optional[int] = None

    def duration(self) -> float:
        return max(0.0, self.end - self.start)


# ── Input adapters ────────────────────────────────────────────────


@dataclass
class ShotInfo:
    start: float
    end: float


@dataclass
class SpeakerTurn:
    t: float
    from_slot: int
    to_slot: int


@dataclass
class OcclusionGap:
    start: float
    end: float
    slot_id: int


@dataclass
class ZoomWindow:
    start: float
    end: float


# ── Scheduler ─────────────────────────────────────────────────────


@dataclass
class _Trigger:
    """Internal: one candidate event, annotated."""
    t: float
    kind: str
    reason: str
    slot: Optional[int] = None
    target_pos: Optional[tuple[float, float]] = None


def _collect_triggers(
    *,
    shots: list[ShotInfo],
    speaker_turns: list[SpeakerTurn],
    occlusions: list[OcclusionGap],
    zoom_windows: list[ZoomWindow],
    duration: float,
) -> list[_Trigger]:
    triggers: list[_Trigger] = []
    for sh in shots[1:]:
        triggers.append(_Trigger(
            t=sh.start, kind="shot_cut",
            reason=f"shot-cut at {sh.start:.2f}s",
        ))
    for tu in speaker_turns:
        triggers.append(_Trigger(
            t=tu.t, kind="speaker_turn",
            reason=f"speaker {tu.from_slot}→{tu.to_slot}",
            slot=tu.to_slot,
        ))
    for oc in occlusions:
        if oc.end - oc.start >= 0.5:
            triggers.append(_Trigger(
                t=oc.end, kind="reappear",
                reason=f"slot {oc.slot_id} reappears after {oc.end - oc.start:.2f}s",
                slot=oc.slot_id,
            ))
    for zw in zoom_windows:
        triggers.append(_Trigger(
            t=zw.start, kind="zoom_start",
            reason=f"motivated zoom starts",
        ))
        triggers.append(_Trigger(
            t=zw.end, kind="zoom_end",
            reason=f"motivated zoom ends",
        ))
    triggers.sort(key=lambda x: x.t)

    # Absorb shot cuts into adjacent speaker saccades.
    merged: list[_Trigger] = []
    for tr in triggers:
        if (merged
                and tr.kind == "shot_cut"
                and merged[-1].kind in ("speaker_turn", "reappear")
                and tr.t - merged[-1].t < _CUT_ABSORB_WINDOW_SEC):
            continue
        if (merged
                and tr.kind in ("speaker_turn", "reappear")
                and merged[-1].kind == "shot_cut"
                and tr.t - merged[-1].t < _CUT_ABSORB_WINDOW_SEC):
            merged[-1] = tr
            continue
        merged.append(tr)
    return merged


def _is_match_cut(
    tr: _Trigger,
    *,
    prev_pos: tuple[float, float],
    next_pos: tuple[float, float] | None,
    config: ReframeConfig,
) -> bool:
    if next_pos is None:
        return False
    dx = next_pos[0] - prev_pos[0]
    dy = next_pos[1] - prev_pos[1]
    return (dx * dx + dy * dy) ** 0.5 < config.match_cut_threshold


def schedule_camera_events(
    *,
    duration: float,
    shots: list[ShotInfo],
    speaker_turns: list[SpeakerTurn],
    occlusions: list[OcclusionGap] | None = None,
    zoom_windows: list[ZoomWindow] | None = None,
    velocity_by_t: list[tuple[float, float]] | None = None,
    positions_by_trigger: dict[float, tuple[float, float]] | None = None,
    config: Optional[ReframeConfig] = None,
) -> list[CameraEvent]:
    """Compose a full ``CameraEvent`` timeline for a segment.

    ``velocity_by_t``: sparse ``[(t, speed)]`` from the Kalman registry.
    Between triggers, the scheduler picks SMOOTH_PURSUE vs HOLD based
    on whether the average speed in the window exceeds the threshold.

    ``positions_by_trigger``: map from trigger timestamp → expected
    subject position, for the MATCH_CUT test. Optional; absent
    positions conservatively get a SACCADE.
    """
    config = config or get_default_config()
    occlusions = occlusions or []
    zoom_windows = zoom_windows or []
    velocity_by_t = velocity_by_t or []
    positions_by_trigger = positions_by_trigger or {}

    triggers = _collect_triggers(
        shots=shots, speaker_turns=speaker_turns,
        occlusions=occlusions, zoom_windows=zoom_windows,
        duration=duration,
    )
    events: list[CameraEvent] = []

    last_t = 0.0
    last_saccade_end = -1.0
    prev_pos: tuple[float, float] = (0.5, 0.5)

    def _avg_speed(t0: float, t1: float) -> float:
        samples = [s for ts, s in velocity_by_t if t0 <= ts <= t1]
        if not samples:
            return 0.0
        return sum(samples) / len(samples)

    for tr in triggers:
        if tr.t <= last_t + 1e-6:
            continue

        speed = _avg_speed(last_t, tr.t)
        mode = CameraMode.SMOOTH_PURSUE if speed > _PURSUE_SPEED_THRESHOLD else CameraMode.HOLD
        events.append(CameraEvent(
            mode=mode,
            start=last_t,
            end=tr.t,
            ease_ms=0,
            reason="between-triggers",
        ))

        # Decide the trigger's event type
        next_pos = positions_by_trigger.get(tr.t)
        ease_ms = config.saccade_ease_ms
        if tr.kind == "shot_cut" and _is_match_cut(
                tr, prev_pos=prev_pos, next_pos=next_pos, config=config):
            mode = CameraMode.MATCH_CUT
            ease_ms = 0
        elif tr.kind == "zoom_start":
            mode = CameraMode.MICRO_ZOOM
            ease_ms = 0
        elif tr.kind == "zoom_end":
            mode = CameraMode.HOLD
            ease_ms = 0
        else:
            # SACCADE, but guard against flicker.
            if tr.t - last_saccade_end < _MIN_SACCADE_GAP_SEC:
                mode = CameraMode.HOLD
                ease_ms = 0
            else:
                mode = CameraMode.SACCADE
                ease_ms = config.saccade_ease_ms
                last_saccade_end = tr.t + ease_ms / 1000.0

        # Saccades / match cuts / zoom markers are instantaneous
        # boundaries in the output; downstream renders ``ease_ms``
        # INTO the next state, so we emit them as zero-length markers.
        events.append(CameraEvent(
            mode=mode,
            start=tr.t,
            end=tr.t,
            ease_ms=ease_ms,
            reason=tr.reason,
            target_slot=tr.slot,
        ))

        last_t = tr.t
        if next_pos is not None:
            prev_pos = next_pos

    # Tail segment up to ``duration``.
    if last_t < duration:
        speed = _avg_speed(last_t, duration)
        mode = CameraMode.SMOOTH_PURSUE if speed > _PURSUE_SPEED_THRESHOLD else CameraMode.HOLD
        events.append(CameraEvent(
            mode=mode,
            start=last_t,
            end=duration,
            ease_ms=0,
            reason="tail",
        ))

    return events


# ── Public helper: ease-ms lookup for reframe_segmenter ───────────


def ease_ms_for_boundary(
    prev_event: Optional[CameraEvent],
    curr_event: CameraEvent,
    config: Optional[ReframeConfig] = None,
) -> int:
    """Return the ease-ms a RenderPlan builder should use when
    transitioning into ``curr_event``.

    Called by :mod:`backend.services.reframe_segmenter` in place of the
    ``EASE_SHOT_CUT_MS`` / ``EASE_SPEAKER_TURN_MS`` constants when the
    human-reframe flag is enabled. Falls back to ``curr_event.ease_ms``.
    """
    config = config or get_default_config()
    if curr_event.mode == CameraMode.MATCH_CUT:
        return 0
    if curr_event.mode == CameraMode.SACCADE:
        return config.saccade_ease_ms
    if curr_event.mode == CameraMode.MICRO_ZOOM:
        return 0
    # SMOOTH_PURSUE / HOLD transitions carry their ease as-is.
    return curr_event.ease_ms
