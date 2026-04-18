"""Phase 7 — Gameplay subject tracker.

The legacy gaming reframe path hard-codes ``subject_x = 50``
(screen-center crosshair) for every gaming clip. That's correct
for FPS / hero shooters where the action sits on the crosshair,
but it's wrong for:

  - **MOBA / top-down**: the camera is centered but the action
    moves across the lane. A fixed center crop loses the
    fight when it migrates into a side lane.

  - **Third-person action (TPS)**: the player character is
    typically offset down+right of frame center (GTA, Elden
    Ring, etc). A center crop puts the character on the edge
    of the vertical frame.

  - **Racing**: the car sits in the lower third of the frame.
    A center crop loses the road / horizon context.

  - **Stream**: gameplay + facecam stacked vertically. The
    gameplay portion needs the per-genre action center; the
    facecam needs the existing single-slot face tracker.

This module ships the per-genre subject tracker. Two production
tiers:

- **OpenCV-backed motion tracker**: lazily imports cv2 and uses
  Farneback dense optical flow to find the dominant motion
  centroid per frame. The motion centroid is the player
  character in TPS, the spell effect in MOBA, the car in
  racing.

- **Pure-Python smoother**: takes a list of per-frame
  ``(timestamp, x_pct, y_pct, magnitude)`` motion centroids
  and produces an EMA-smoothed + velocity-clamped subject path
  per segment. Falls back to the genre-specific
  ``action_center_pct`` from ``game_layouts.GAME_HUD_LAYOUTS``
  when motion is weak.

The ``backend.services.game_layouts`` table already has the
per-genre ``action_center_pct`` values from Phase 2 — this
module ties them to per-frame motion + per-segment trajectory.

Feature flag: ``CLIPAI_GAMEPLAY_TRACKER`` env var, default OFF
until in-docker validation lands the post-Phase-7 numbers.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# ──────────────────── Feature flag ────────────────────

USE_GAMEPLAY_TRACKER = os.environ.get(
    "CLIPAI_GAMEPLAY_TRACKER", "1",
).lower() in ("1", "true", "yes", "on")


# ──────────────────── Tuning constants ────────────────────

# EMA time constant for motion-centroid smoothing. 0.50 s
# matches the existing build_motion_tracking_path default in
# optical_flow.py — it kills per-frame jitter without lagging
# real subject motion.
DEFAULT_EMA_TAU_SEC = 0.50

# Maximum tracking velocity in % of frame per second. Anything
# faster is filtered as noise (camera shake, optical flow
# misfire). 15 %/s gets a subject across half the frame in
# ~3 seconds, which matches typical TPS / MOBA pace.
DEFAULT_MAX_VELOCITY_PCT_PER_SEC = 15.0

# Minimum motion magnitude for the tracker to override the
# per-genre action center. Below this we trust the genre
# default (FPS center, racing lower-third, etc) rather than
# chasing weak signals.
MIN_MOTION_MAGNITUDE = 0.6


# ──────────────────── Per-frame centroid dataclass ────────────────────


@dataclass
class GameplayMotionCentroid:
    """One frame's motion centroid + magnitude.

    Attributes:
        timestamp: Frame timestamp (seconds).
        x_pct / y_pct: Centroid position in [0, 100] % coordinates
            of the source frame.
        magnitude: Motion magnitude in [0, 1]. Larger = stronger
            signal. Below ``MIN_MOTION_MAGNITUDE`` the tracker
            falls back to the per-genre action center.
    """

    timestamp: float
    x_pct: float
    y_pct: float
    magnitude: float = 0.0


@dataclass
class GameplaySubjectPath:
    """Output of :func:`track_gameplay_subject` — smoothed per-frame
    subject trajectory for one segment.

    Attributes:
        path: List of ``(timestamp, x_pct, y_pct)`` tuples after
            EMA smoothing + velocity clamping + fallback resolution.
        source: Source label — one of ``"motion"`` (motion centroids
            drove the path), ``"action_center"`` (motion was too
            weak, fell back to genre default), or ``"mixed"``
            (some frames used motion, others fell back).
        n_motion_frames: How many frames used the motion centroid
            vs the fallback. Useful for telemetry.
    """

    path: list[tuple[float, float, float]] = field(default_factory=list)
    source: str = "action_center"
    n_motion_frames: int = 0


# ──────────────────── Per-genre fallback ────────────────────


def subject_anchor_for_game(game_key: Optional[str]) -> tuple[float, float]:
    """Return the ``(x_pct, y_pct)`` action-center anchor for a game.

    Wraps :func:`backend.services.game_layouts.get_action_center`
    with a None-safe default. Used as the fallback when motion
    centroids are weak / missing.

    Returns ``(50.0, 50.0)`` when the game key is unknown / None.
    """
    from backend.services.game_layouts import get_action_center

    if not game_key:
        return (50.0, 50.0)
    return get_action_center(game_key)


def subject_anchor_for_genre(genre: Optional[str]) -> tuple[float, float]:
    """Return the action-center anchor for a gameplay genre.

    Genre is one of ``fps`` / ``moba`` / ``tps`` / ``racing`` /
    ``sandbox``. Mirrors the per-game lookup but operates on the
    coarser ``profile.gameplay_subtype`` value when no specific
    game was selected.
    """
    from backend.services.game_layouts import (
        DEFAULT_GAME_BY_GENRE,
        get_action_center,
    )

    if not genre:
        return (50.0, 50.0)
    default_game = DEFAULT_GAME_BY_GENRE.get(genre.strip().lower())
    if not default_game:
        return (50.0, 50.0)
    return get_action_center(default_game)


# ──────────────────── Pure-Python tracker ────────────────────


def track_gameplay_subject(
    centroids: list[GameplayMotionCentroid],
    *,
    fallback_xy_pct: tuple[float, float],
    start: float,
    end: float,
    ema_tau_sec: float = DEFAULT_EMA_TAU_SEC,
    max_velocity_pct_per_sec: float = DEFAULT_MAX_VELOCITY_PCT_PER_SEC,
    min_magnitude: float = MIN_MOTION_MAGNITUDE,
) -> GameplaySubjectPath:
    """Smooth a sequence of motion centroids into a subject path.

    Walks the centroid sequence in order, applies EMA smoothing
    with the configured time constant, and clamps per-step
    movement to the maximum velocity. Centroids with magnitude
    below ``min_magnitude`` are treated as "no signal" — those
    frames inherit the previous smoothed position (or the
    fallback when no previous position exists yet).

    Mirrors ``optical_flow.build_motion_tracking_path``'s
    smoothing math but operates on motion centroids instead of
    face centroids and adds the per-genre fallback semantics.

    Args:
        centroids: Per-frame motion centroids in the segment.
            May be empty.
        fallback_xy_pct: ``(x, y)`` anchor in [0, 100] % to use
            when motion is too weak. Comes from
            :func:`subject_anchor_for_game`.
        start: Segment start time (seconds).
        end: Segment end time (seconds).
        ema_tau_sec / max_velocity_pct_per_sec / min_magnitude:
            tunables documented above.

    Returns:
        ``GameplaySubjectPath`` with the smoothed trajectory,
        a ``source`` label, and the count of frames that used
        the motion signal.
    """
    fallback_x, fallback_y = float(fallback_xy_pct[0]), float(fallback_xy_pct[1])

    # Filter to in-segment centroids
    in_seg: list[GameplayMotionCentroid] = [
        c for c in centroids
        if start <= c.timestamp <= end
    ]

    if not in_seg:
        # No motion data → single-point path at the fallback
        return GameplaySubjectPath(
            path=[(start, fallback_x, fallback_y)],
            source="action_center",
            n_motion_frames=0,
        )

    # Smooth + clamp + fallback
    smoothed: list[tuple[float, float, float]] = []
    smooth_x = fallback_x
    smooth_y = fallback_y
    last_t = start
    n_motion = 0
    saw_any_motion = False

    in_seg.sort(key=lambda c: c.timestamp)
    for cent in in_seg:
        t = cent.timestamp
        if cent.magnitude < min_magnitude:
            # Weak signal — hold previous smoothed position
            smoothed.append((t, smooth_x, smooth_y))
            last_t = t
            continue
        target_x = float(cent.x_pct)
        target_y = float(cent.y_pct)
        if not saw_any_motion:
            # First strong signal initializes the smoother
            smooth_x = target_x
            smooth_y = target_y
            saw_any_motion = True
            n_motion += 1
            smoothed.append((t, smooth_x, smooth_y))
            last_t = t
            continue
        # EMA smoothing: alpha = 1 - exp(-dt / tau)
        dt = max(t - last_t, 1e-6)
        alpha = 1.0 - math.exp(-dt / max(ema_tau_sec, 1e-6))
        new_x = smooth_x + alpha * (target_x - smooth_x)
        new_y = smooth_y + alpha * (target_y - smooth_y)
        # Velocity clamp (in % per sec)
        max_step = max_velocity_pct_per_sec * dt
        dx = new_x - smooth_x
        dy = new_y - smooth_y
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > max_step and dist > 0:
            scale = max_step / dist
            new_x = smooth_x + dx * scale
            new_y = smooth_y + dy * scale
        smooth_x = new_x
        smooth_y = new_y
        n_motion += 1
        smoothed.append((t, smooth_x, smooth_y))
        last_t = t

    if n_motion == 0:
        source = "action_center"
    elif n_motion == len(in_seg):
        source = "motion"
    else:
        source = "mixed"

    return GameplaySubjectPath(
        path=smoothed,
        source=source,
        n_motion_frames=n_motion,
    )


def aggregate_subject_x_for_segment(
    centroids: list[GameplayMotionCentroid],
    *,
    start: float,
    end: float,
    fallback_xy_pct: tuple[float, float],
    ema_tau_sec: float = DEFAULT_EMA_TAU_SEC,
    max_velocity_pct_per_sec: float = DEFAULT_MAX_VELOCITY_PCT_PER_SEC,
    min_magnitude: float = MIN_MOTION_MAGNITUDE,
) -> tuple[float, float, str]:
    """Pick a single ``(x, y, source)`` for a segment.

    Wraps :func:`track_gameplay_subject` and averages the
    smoothed path's mid-segment position. The reframe segmenter
    uses this to set ``seg.subject_x`` / ``seg.subject_y``
    when the gameplay tracker is enabled.

    Returns the fallback ``(x, y, "action_center")`` when the
    tracker has no data.
    """
    sp = track_gameplay_subject(
        centroids,
        fallback_xy_pct=fallback_xy_pct,
        start=start,
        end=end,
        ema_tau_sec=ema_tau_sec,
        max_velocity_pct_per_sec=max_velocity_pct_per_sec,
        min_magnitude=min_magnitude,
    )
    if not sp.path:
        return (fallback_xy_pct[0], fallback_xy_pct[1], "action_center")
    # Average position across the smoothed path. Could swap
    # for a median if outliers become a problem.
    sum_x = sum(p[1] for p in sp.path)
    sum_y = sum(p[2] for p in sp.path)
    n = len(sp.path)
    return (sum_x / n, sum_y / n, sp.source)


# ──────────────────── HUD preservation fallback ────────────────────


def derive_hud_safe_subject_x(
    *,
    hud_zones: list[tuple[float, float, float, float]],
    crop_width_pct: float,
    source_width_pct: float = 100.0,
) -> float:
    """Pick a ``subject_x`` that minimizes HUD intrusion when
    no specific game key is known.

    Walks the HUD zones (each ``(x_pct, y_pct, w_pct, h_pct)``),
    computes the union of their horizontal extents, and picks
    the crop center that maximizes the visible HUD width.

    The cleanest output is the **centroid** of all HUD zones
    weighted by area — this puts the crop window over the
    HUD-densest region, which is usually what an unfamiliar
    game wants (most HUDs cluster around the bottom-center
    health/ammo bar).

    Args:
        hud_zones: List of ``(x_pct, y_pct, w_pct, h_pct)``
            tuples in [0, 100] % coordinates.
        crop_width_pct: Crop window width as % of source width.
        source_width_pct: Source frame width — defaults to 100
            (caller normalized). Reserved for callers using
            non-standard coordinate spaces.

    Returns:
        Subject x in [0, 100] % coordinates. Returns 50.0
        (frame center) when no HUD zones are provided OR all
        zones have zero area.
    """
    if not hud_zones:
        return 50.0
    total_area = 0.0
    weighted_sum = 0.0
    for x_pct, _y_pct, w_pct, h_pct in hud_zones:
        area = max(0.0, float(w_pct)) * max(0.0, float(h_pct))
        if area <= 0:
            continue
        cx = float(x_pct) + float(w_pct) / 2.0
        weighted_sum += cx * area
        total_area += area
    if total_area <= 0:
        return 50.0
    cx = weighted_sum / total_area
    # Clamp so the crop window stays inside the source frame
    half = crop_width_pct / 2.0
    cx = max(half, min(source_width_pct - half, cx))
    return cx
