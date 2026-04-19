"""Per-subject constant-acceleration Kalman filter.

Replaces the ad-hoc EMAs scattered through ``crosshair_tracker``,
``gameplay_subject_tracker``, ``gaze_estimator``, and the attention
anchor smoother with a single 6-state Kalman per tracked subject:

    state = [x, y, vx, vy, ax, ay]

Observations feed from every detector (face, body, saliency peak,
crosshair, gameplay action center, VLM subject_x). Observation noise
is per-source so faces contribute more signal than saliency.

The filter publishes, at every update:

    * A smoothed estimate of the current state.
    * A prediction of the subject's position ``prediction_ms`` ahead.
      This is what the 2-D solver consumes instead of raw observations,
      so the camera *leads* the subject like a human operator does.

Pure-Python + numpy only; no pytorch / cv2 dependency. Cheap enough
(O(1) per frame, no matrix inversion > 2x2) to run on every face /
saliency / crosshair update.

Integration points:

  * ``backend.services.camera_path_2d.solve_2d_camera_path`` accepts a
    ``predictor`` argument (when the master human-reframe flag is on)
    and uses predicted positions as the data term.
  * ``backend.services.required_regions`` uses prediction covariance
    to decide whether to downweight a noisy observation.
  * Gameplay / crosshair trackers wrap themselves in ``KalmanSubject``
    instead of maintaining their own EMA (Phase 9 genre pass).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from backend.services.reframe_config import ReframeConfig, get_default_config

logger = logging.getLogger(__name__)


# ── Observation source noise table ─────────────────────────────────
#
# Per-source standard deviations in normalized source-frame units.
# Faces are tight (~2% of source width), saliency is noisy (~6%),
# crosshair templates are tight (~1.5%), VLM answers are rarely
# precise (~8% if available).

_OBS_NOISE = {
    "face":            0.02,
    "face_dense":      0.025,
    "active_speaker":  0.02,
    "person":          0.04,
    "saliency":        0.06,
    "crosshair":       0.015,
    "gameplay_motion": 0.05,
    "attention":       0.05,
    "vlm":             0.08,
    "gaze":            0.04,
}


@dataclass
class KalmanSubject:
    """Per-subject Kalman filter state.

    ``slot_id`` identifies which face-registry / subject_fusion track
    this filter belongs to. One instance per tracked subject; the
    pipeline creates filters lazily on first observation.
    """

    slot_id: int
    config: ReframeConfig = field(default_factory=get_default_config)

    # State vector: [x, y, vx, vy, ax, ay]
    x: np.ndarray = field(default_factory=lambda: np.zeros(6))
    # State covariance
    P: np.ndarray = field(default_factory=lambda: np.eye(6) * 0.1)

    # Process noise base (scaled by dt^2 in predict step)
    q_base: float = 0.0

    # Timestamp of last update (seconds); None = uninitialized
    last_t: Optional[float] = None

    def __post_init__(self):
        # Rescale process-noise base from config
        self.q_base = max(1e-6, float(self.config.kalman_process_noise)) ** 2

    # ── Transition / observation matrices ──────────────────────────
    @staticmethod
    def _F(dt: float) -> np.ndarray:
        """Constant-acceleration transition matrix for dt seconds."""
        F = np.eye(6)
        F[0, 2] = dt
        F[1, 3] = dt
        F[0, 4] = 0.5 * dt * dt
        F[1, 5] = 0.5 * dt * dt
        F[2, 4] = dt
        F[3, 5] = dt
        return F

    @staticmethod
    def _H() -> np.ndarray:
        """Observation matrix: we only observe ``(x, y)``."""
        H = np.zeros((2, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        return H

    def _Q(self, dt: float) -> np.ndarray:
        """Continuous-white-noise acceleration discretization."""
        q = self.q_base
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        # Block diagonal over (x-axis, y-axis) with coupling between
        # position, velocity and acceleration for a constant-accel model.
        block = np.array([
            [dt4 / 4.0, dt3 / 2.0, dt2 / 2.0],
            [dt3 / 2.0, dt2,       dt],
            [dt2 / 2.0, dt,        1.0],
        ]) * q
        Q = np.zeros((6, 6))
        # Rearrange indices to match our state layout [x, y, vx, vy, ax, ay]
        idx_x = [0, 2, 4]
        idx_y = [1, 3, 5]
        for i, a in enumerate(idx_x):
            for j, b in enumerate(idx_x):
                Q[a, b] = block[i, j]
        for i, a in enumerate(idx_y):
            for j, b in enumerate(idx_y):
                Q[a, b] = block[i, j]
        return Q

    # ── Public update API ─────────────────────────────────────────

    def observe(self, t: float, x: float, y: float, *, source: str = "face") -> None:
        """Consume one observation at time ``t`` from ``source``."""
        if self.last_t is None:
            self.x = np.array([x, y, 0.0, 0.0, 0.0, 0.0], dtype=float)
            self.P = np.eye(6) * 0.05
            self.P[2:4, 2:4] *= 4.0   # larger initial velocity uncertainty
            self.P[4:6, 4:6] *= 16.0
            self.last_t = t
            return

        dt = max(1e-3, t - self.last_t)
        F = self._F(dt)
        Q = self._Q(dt)

        # Predict
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q

        # Measurement
        H = self._H()
        z = np.array([x, y], dtype=float)

        r = _OBS_NOISE.get(source, 0.05)
        # Scale by config's measurement-noise multiplier
        r = r * max(1e-3, float(self.config.kalman_measurement_noise)) / 0.16
        R = np.eye(2) * (r * r)

        y_res = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        self.x = x_pred + K @ y_res
        I = np.eye(6)
        self.P = (I - K @ H) @ P_pred
        self.last_t = t

    def predict(self, t: float) -> tuple[float, float, float]:
        """Predict state at time ``t``.

        Returns ``(x, y, uncertainty)`` where ``uncertainty`` is the
        trace of the position covariance (useful for gating).
        """
        if self.last_t is None:
            return (float(self.x[0]), float(self.x[1]), 1.0)
        dt = max(0.0, t - self.last_t)
        F = self._F(dt)
        xp = F @ self.x
        Pp = F @ self.P @ F.T + self._Q(max(dt, 1e-3))
        uncertainty = float(Pp[0, 0] + Pp[1, 1])
        return (float(xp[0]), float(xp[1]), uncertainty)

    @property
    def velocity(self) -> tuple[float, float]:
        return (float(self.x[2]), float(self.x[3]))

    @property
    def acceleration(self) -> tuple[float, float]:
        return (float(self.x[4]), float(self.x[5]))

    @property
    def speed(self) -> float:
        return math.hypot(float(self.x[2]), float(self.x[3]))


# ── Multi-subject registry ────────────────────────────────────────


@dataclass
class SubjectKalmanRegistry:
    """Holds one :class:`KalmanSubject` per slot.

    Threadsafe use is NOT a goal — the pipeline is single-threaded at
    the reframing stage, and the consumer copies out of the registry
    before the render path.
    """

    config: ReframeConfig = field(default_factory=get_default_config)
    filters: dict[int, KalmanSubject] = field(default_factory=dict)

    def observe(self, slot_id: int, t: float, x: float, y: float, *,
                source: str = "face") -> KalmanSubject:
        k = self.filters.get(slot_id)
        if k is None:
            k = KalmanSubject(slot_id=slot_id, config=self.config)
            self.filters[slot_id] = k
        k.observe(t, x, y, source=source)
        return k

    def predicted_positions(self, t: float) -> dict[int, tuple[float, float, float]]:
        return {sid: k.predict(t) for sid, k in self.filters.items()}

    def prediction_for_lead(
        self, slot_id: int, t_now: float,
    ) -> Optional[tuple[float, float, float]]:
        """Return ``(x, y, uncertainty)`` the slot is expected to occupy
        ``config.kalman_prediction_ms`` ahead of ``t_now``.

        Used as the solver's data target so the camera leads the
        subject. ``uncertainty`` is the trace of the position
        covariance; callers should fall back to the raw observation
        when ``uncertainty > 0.05`` (filter widened too much, e.g. a
        no-observation gap).
        """
        k = self.filters.get(slot_id)
        if k is None or k.last_t is None:
            return None
        t_future = t_now + self.config.kalman_prediction_ms / 1000.0
        x, y, u = k.predict(t_future)
        return (x, y, u)


# ── Integration helper: feed dense faces into a registry ──────────


def build_registry_from_dense_faces(
    dense_faces: list,
    active_speaker_events: list | None = None,
    config: Optional[ReframeConfig] = None,
) -> SubjectKalmanRegistry:
    """Initialize + fill a Kalman registry from a dense-face stream.

    Does two passes internally:
      1. Walk frames in order, observe the active-speaker face (if in
         frame) OR every face by identity_id.
      2. Return the registry with filters advanced to the last frame.
    """
    reg = SubjectKalmanRegistry(config=config or get_default_config())

    def _active_slot_at(t: float) -> int:
        if not active_speaker_events:
            return -1
        for ev in active_speaker_events:
            if ev.start <= t <= ev.end and getattr(ev, "slot_id", -1) >= 0:
                return ev.slot_id
        return -1

    for ff in dense_faces:
        t = ff.timestamp
        active = _active_slot_at(t)
        for face in ff.faces:
            if not getattr(face, "is_human", True):
                continue
            sid = getattr(face, "identity_id", -1)
            if sid < 0:
                continue
            x = face.nose_x / 100.0
            y = face.nose_y / 100.0
            source = "active_speaker" if sid == active else "face_dense"
            reg.observe(sid, t, x, y, source=source)
    return reg
