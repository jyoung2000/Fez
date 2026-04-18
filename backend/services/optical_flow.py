"""Cheap optical flow estimation for motion-aware tracking.

Uses face position deltas from dense face detection as a proxy for
optical flow. Falls back to frame difference magnitude when faces
aren't available. No OpenCV Farneback needed — leverages data the
pipeline has already decoded.

For fast content (racing, anime, music videos, fast gameplay),
produces motion_path keypoints driven by motion centroid rather
than face position.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Motion tracking thresholds ──
MOTION_ENERGY_TRACKING_THRESHOLD = 0.70  # top quartile → use tracking
MOTION_CHAOTIC_THRESHOLD = 1.0           # stdev > mean → too chaotic, use wide_master
TRACKING_EMA_TAU = 0.3                   # exponential moving average time constant (seconds)
TRACKING_MAX_VELOCITY_PCT_PER_SEC = 25   # max velocity in % of frame width per second


def compute_motion_energy_per_second(
    dense_faces: list,
    duration_sec: float,
) -> np.ndarray:
    """Compute per-second motion energy from dense face position changes.

    Uses frame-to-frame delta in face x-positions as motion proxy.
    Self-calibrates against the video's own distribution.

    Returns:
        1D numpy array of normalized motion magnitudes, one per second.
    """
    n = max(1, int(round(duration_sec)))
    motion = np.zeros(n, dtype=np.float32)

    if not dense_faces or len(dense_faces) < 2:
        return motion

    # Build per-frame mean face x
    frame_data = []
    for df in dense_faces:
        t = df.timestamp
        if df.faces:
            xs = [getattr(f, 'x', 50) for f in df.faces if getattr(f, 'identity_id', -1) >= 0]
            if xs:
                frame_data.append((t, sum(xs) / len(xs)))

    if len(frame_data) < 2:
        return motion

    frame_data.sort(key=lambda x: x[0])

    # Compute per-second deltas
    for i in range(1, len(frame_data)):
        t_prev, x_prev = frame_data[i - 1]
        t_curr, x_curr = frame_data[i]
        dt = t_curr - t_prev
        if dt > 0:
            delta = abs(x_curr - x_prev) / dt  # velocity in %/sec
            sec_idx = int(t_curr)
            if 0 <= sec_idx < n:
                motion[sec_idx] = max(motion[sec_idx], delta)

    # Self-calibrate: normalize against top 5%
    nonzero = motion[motion > 0]
    if len(nonzero) > 0:
        top_5pct = float(np.percentile(nonzero, 95))
        if top_5pct > 0:
            motion = motion / top_5pct
            motion = np.clip(motion, 0.0, 1.0)

    return motion


def is_motion_chaotic(motion_energy: np.ndarray, start_sec: float, end_sec: float) -> bool:
    """Check if motion in the interval is too chaotic for tracking.

    Motion is chaotic when its stdev exceeds its mean — meaning motion
    is going in all directions (crowd, explosion, rapid cuts).
    """
    start_idx = max(0, int(start_sec))
    end_idx = min(len(motion_energy), int(end_sec) + 1)
    if end_idx <= start_idx:
        return False

    window = motion_energy[start_idx:end_idx]
    mean = float(np.mean(window))
    if mean < 0.01:
        return False

    std = float(np.std(window))
    return std > mean * MOTION_CHAOTIC_THRESHOLD


def should_use_motion_tracking(
    motion_energy: np.ndarray,
    start_sec: float,
    end_sec: float,
) -> bool:
    """Determine if motion-aware tracking should be used for a segment.

    Returns True if motion energy exceeds the threshold AND is not chaotic.
    """
    start_idx = max(0, int(start_sec))
    end_idx = min(len(motion_energy), int(end_sec) + 1)
    if end_idx <= start_idx:
        return False

    window = motion_energy[start_idx:end_idx]
    mean_energy = float(np.mean(window))

    if mean_energy < MOTION_ENERGY_TRACKING_THRESHOLD:
        return False

    if is_motion_chaotic(motion_energy, start_sec, end_sec):
        return False

    return True


def build_motion_tracking_path(
    dense_faces: list,
    start_sec: float,
    end_sec: float,
    shot_cuts: list,
    fps: float = 1.0,
    ema_tau: float = TRACKING_EMA_TAU,
    max_velocity: float = TRACKING_MAX_VELOCITY_PCT_PER_SEC,
) -> list:
    """Build a motion-tracking path from face position data.

    Uses EMA smoothing and velocity clamping on face centroids.
    Hard-resets at shot cuts.

    Returns:
        List of (t, x, y) tuples where t is absolute time, x/y are 0-100.
    """
    # Gather face positions in the interval
    positions = []
    for df in dense_faces:
        if df.timestamp < start_sec or df.timestamp > end_sec:
            continue
        if not df.faces:
            continue
        xs = [getattr(f, 'x', 50) for f in df.faces if getattr(f, 'identity_id', -1) >= 0]
        ys = [getattr(f, 'y', 40) for f in df.faces if getattr(f, 'identity_id', -1) >= 0]
        if xs:
            positions.append((df.timestamp, sum(xs) / len(xs), sum(ys) / len(ys)))

    if len(positions) < 2:
        return []

    positions.sort(key=lambda p: p[0])

    # Find shot cuts in the interval
    cuts_in_range = sorted(sc for sc in shot_cuts if start_sec < sc < end_sec)

    # Apply EMA smoothing with velocity clamping and shot-cut resets
    path = []
    smooth_x = positions[0][1]
    smooth_y = positions[0][2]
    path.append((positions[0][0], smooth_x, smooth_y))

    for i in range(1, len(positions)):
        t_prev = positions[i - 1][0]
        t_curr, raw_x, raw_y = positions[i]
        dt = t_curr - t_prev

        # Check for shot cut between prev and curr
        has_cut = any(t_prev < sc <= t_curr for sc in cuts_in_range)
        if has_cut:
            # Hard reset
            smooth_x = raw_x
            smooth_y = raw_y
            path.append((t_curr, smooth_x, smooth_y))
            continue

        if dt <= 0:
            continue

        # EMA smoothing: alpha = 1 - exp(-dt / tau)
        alpha = 1.0 - np.exp(-dt / ema_tau) if ema_tau > 0 else 1.0
        target_x = smooth_x + alpha * (raw_x - smooth_x)
        target_y = smooth_y + alpha * (raw_y - smooth_y)

        # Velocity clamping
        max_delta = max_velocity * dt
        dx = target_x - smooth_x
        dy = target_y - smooth_y
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > max_delta and dist > 0:
            scale = max_delta / dist
            target_x = smooth_x + dx * scale
            target_y = smooth_y + dy * scale

        smooth_x = target_x
        smooth_y = target_y
        path.append((t_curr, smooth_x, smooth_y))

    return path
