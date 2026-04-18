"""Local pacing estimator for adaptive minimum-hold derivation.

Replaces hardcoded per-content-type min_hold_seconds with a data-driven
approach. Measures shot cut density, speaker turn density, motion energy,
and audio dynamics over a rolling 8-second window, then derives the local
minimum hold via inverse lerp with a power curve.

A calm sit-down interview and a rapid-fire debate are both "podcast"
but produce different cut cadences because their measured pacing differs.
"""

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Adaptive hold constants ──
MIN_HOLD_FLOOR = 0.25      # frantic content: sub-second cuts allowed
MIN_HOLD_CEILING = 2.00    # calm content: 2.0s minimum hold (snappier response)
HOLD_CURVE_POWER = 0.75    # gentle power curve so mid-range isn't a plateau

# ── Anticipation constants ──
ANTICIPATION_MIN_MS = 120  # frantic: short anticipation
ANTICIPATION_MAX_MS = 400  # calm: full anticipation for snappy lead-in

# ── Pacing window ──
PACING_WINDOW_SEC = 8      # rolling window size in seconds
PACING_SMOOTHING_TAPS = 3  # median filter kernel size

# ── Normalization references (self-calibrated per video) ──
CUT_RATE_REF = 2.0         # 2 cuts/sec = pacing 1.0
TURN_RATE_REF = 1.0        # 1 speaker turn/sec = pacing 1.0

# ── Per-content-type signal weights ──
# Which signal to trust for each content type. The weights DON'T change pacing;
# the measured signal values do. Weights only reflect "which signal should we
# trust for this content type."
PACING_WEIGHTS = {
    "narrative":    {"cuts": 0.45, "speakers": 0.10, "motion": 0.25, "audio": 0.20},
    "podcast":      {"cuts": 0.05, "speakers": 0.55, "motion": 0.05, "audio": 0.35},
    "gaming":       {"cuts": 0.05, "speakers": 0.00, "motion": 0.60, "audio": 0.35},
    "vlog":         {"cuts": 0.15, "speakers": 0.25, "motion": 0.35, "audio": 0.25},
    "sports":       {"cuts": 0.20, "speakers": 0.05, "motion": 0.55, "audio": 0.20},
    "music_video":  {"cuts": 0.40, "speakers": 0.00, "motion": 0.45, "audio": 0.15},
    "anime":        {"cuts": 0.35, "speakers": 0.15, "motion": 0.30, "audio": 0.20},
    # animation_dialogue = anime dialogue scenes: cut-driven pacing with
    # meaningful speaker turns. Higher speaker weight than pure anime
    # because BSD-style content has clear per-character speech runs.
    "animation_dialogue": {"cuts": 0.30, "speakers": 0.30, "motion": 0.20, "audio": 0.20},
    "unknown":      {"cuts": 0.25, "speakers": 0.25, "motion": 0.25, "audio": 0.25},
}


class LocalPacingEstimator:
    """Computes per-second pacing scores from video signals.

    Usage:
        estimator = LocalPacingEstimator(duration, content_type="podcast")
        estimator.add_shot_cuts(shot_cuts)
        estimator.add_speaker_turns(active_speaker_events)
        estimator.add_motion_energy(motion_magnitudes)
        estimator.add_audio_energy(audio_rms_buckets)
        estimator.compute()

        pacing_at_5s = estimator.pacing[5]
        min_hold_at_5s = estimator.min_hold_at(5.0)
    """

    def __init__(self, duration_sec: float, content_type: str = "unknown",
                 window_sec: int = PACING_WINDOW_SEC):
        self.duration = max(1, int(round(duration_sec)))
        self.content_type = content_type
        self.window = window_sec
        self.half_win = window_sec // 2

        weights = PACING_WEIGHTS.get(content_type, PACING_WEIGHTS["unknown"])
        self.w_cuts = weights["cuts"]
        self.w_speakers = weights["speakers"]
        self.w_motion = weights["motion"]
        self.w_audio = weights["audio"]

        # Per-second signal arrays (initialized to 0)
        self.cut_density = np.zeros(self.duration, dtype=np.float32)
        self.speaker_density = np.zeros(self.duration, dtype=np.float32)
        self.motion_energy = np.zeros(self.duration, dtype=np.float32)
        self.audio_dynamics = np.zeros(self.duration, dtype=np.float32)

        # Output
        self.pacing = np.zeros(self.duration, dtype=np.float32)
        self._computed = False

    def add_shot_cuts(self, shot_cuts: list):
        """Add shot cut timestamps. Each cut increments the density at its second."""
        if not shot_cuts:
            return
        for t in shot_cuts:
            idx = int(t)
            if 0 <= idx < self.duration:
                self.cut_density[idx] += 1.0

    def add_speaker_turns(self, active_speaker_events: list):
        """Add active-speaker events and count slot changes per second."""
        if not active_speaker_events or len(active_speaker_events) < 2:
            return
        for i in range(1, len(active_speaker_events)):
            prev_ev = active_speaker_events[i - 1]
            curr_ev = active_speaker_events[i]
            if curr_ev.slot_id != prev_ev.slot_id:
                idx = int(curr_ev.start)
                if 0 <= idx < self.duration:
                    self.speaker_density[idx] += 1.0

    def add_motion_energy(self, motion_magnitudes: Optional[np.ndarray]):
        """Add per-second motion magnitude (from optical flow or dense face movement).

        motion_magnitudes should be a 1D array with one value per second.
        Values are self-calibrated against the video's top 5% magnitude.
        """
        if motion_magnitudes is None or len(motion_magnitudes) == 0:
            return
        n = min(len(motion_magnitudes), self.duration)
        self.motion_energy[:n] = motion_magnitudes[:n]

    def add_audio_energy(self, rms_buckets: Optional[np.ndarray],
                         bucket_duration_sec: float = 0.25):
        """Add audio RMS per bucket (e.g., 250ms buckets).

        Computes "peakiness" per window: stdev(rms) / mean(rms).
        High peakiness = loud-quiet-loud patterns (debate, action).
        """
        if rms_buckets is None or len(rms_buckets) == 0:
            return

        buckets_per_sec = max(1, int(round(1.0 / bucket_duration_sec)))
        for t in range(self.duration):
            bucket_start = t * buckets_per_sec
            bucket_end = min(len(rms_buckets), (t + 1) * buckets_per_sec)
            if bucket_start >= bucket_end:
                continue
            window_rms = rms_buckets[bucket_start:bucket_end]
            mean_rms = float(np.mean(window_rms))
            if mean_rms > 1e-6:
                self.audio_dynamics[t] = float(np.std(window_rms)) / mean_rms
            else:
                self.audio_dynamics[t] = 0.0

    def compute(self):
        """Compute per-second pacing scores from all added signals.

        Uses prefix sums for O(N) windowed computation instead of O(N×W).
        Applies rolling window smoothing and signal normalization.
        """
        # Build prefix sums for O(1) window queries
        cut_prefix = np.zeros(self.duration + 1, dtype=np.float64)
        speaker_prefix = np.zeros(self.duration + 1, dtype=np.float64)
        motion_prefix = np.zeros(self.duration + 1, dtype=np.float64)
        audio_prefix = np.zeros(self.duration + 1, dtype=np.float64)

        np.cumsum(self.cut_density, out=cut_prefix[1:])
        np.cumsum(self.speaker_density, out=speaker_prefix[1:])
        np.cumsum(self.motion_energy, out=motion_prefix[1:])
        np.cumsum(self.audio_dynamics, out=audio_prefix[1:])

        raw_pacing = np.zeros(self.duration, dtype=np.float32)

        for t in range(self.duration):
            win_start = max(0, t - self.half_win)
            win_end = min(self.duration, t + self.half_win + 1)
            win_size = win_end - win_start

            # O(1) window sums via prefix arrays
            cuts_in_win = float(cut_prefix[win_end] - cut_prefix[win_start])
            norm_cuts = min(1.0, (cuts_in_win / max(1, win_size)) / CUT_RATE_REF)

            turns_in_win = float(speaker_prefix[win_end] - speaker_prefix[win_start])
            norm_speakers = min(1.0, (turns_in_win / max(1, win_size)) / TURN_RATE_REF)

            motion_sum = float(motion_prefix[win_end] - motion_prefix[win_start])
            norm_motion = min(1.0, motion_sum / max(1, win_size))

            audio_sum = float(audio_prefix[win_end] - audio_prefix[win_start])
            norm_audio = min(1.0, audio_sum / max(1, win_size))

            raw_pacing[t] = (
                self.w_cuts * norm_cuts
                + self.w_speakers * norm_speakers
                + self.w_motion * norm_motion
                + self.w_audio * norm_audio
            )

        # Clamp to [0, 1]
        raw_pacing = np.clip(raw_pacing, 0.0, 1.0)

        # Median filter for spike suppression
        self.pacing = _median_filter_1d(raw_pacing, PACING_SMOOTHING_TAPS)
        self._computed = True

    def min_hold_at(self, t: float) -> float:
        """Get the adaptive minimum hold at time t (seconds)."""
        if not self._computed:
            self.compute()
        idx = max(0, min(int(t), self.duration - 1))
        return derive_min_hold_sec(float(self.pacing[idx]))

    def anticipation_ms_at(self, t: float) -> int:
        """Get the adaptive anticipation in ms at time t."""
        if not self._computed:
            self.compute()
        idx = max(0, min(int(t), self.duration - 1))
        score = float(self.pacing[idx])
        # Frantic content → short anticipation; calm content → full anticipation
        return int(round(ANTICIPATION_MAX_MS - score * (ANTICIPATION_MAX_MS - ANTICIPATION_MIN_MS)))

    def get_pacing_array(self) -> list:
        """Return pacing array as a Python list (for JSON serialization)."""
        if not self._computed:
            self.compute()
        return [round(float(v), 3) for v in self.pacing]

    def get_min_hold_array(self) -> list:
        """Return derived min_hold per second as a Python list."""
        if not self._computed:
            self.compute()
        return [round(derive_min_hold_sec(float(v)), 3) for v in self.pacing]


def derive_min_hold_sec(pacing_score: float) -> float:
    """Derive minimum hold duration from a pacing score.

    Uses a power curve so the middle of the range doesn't hover
    around an awkward plateau.

    Args:
        pacing_score: 0.0 (calm) to 1.0 (frantic)

    Returns:
        Minimum hold duration in seconds.
    """
    pacing_score = max(0.0, min(1.0, pacing_score))
    curved = pacing_score ** HOLD_CURVE_POWER
    return MIN_HOLD_CEILING - curved * (MIN_HOLD_CEILING - MIN_HOLD_FLOOR)


def compute_motion_from_dense_faces(dense_faces: list, duration_sec: float) -> np.ndarray:
    """Cheap motion estimation from dense face position changes.

    Uses the frame-to-frame change in face x-positions as a proxy for
    scene motion. This is computed from data already decoded — no extra
    OpenCV optical flow needed.

    Returns a per-second normalized motion magnitude array.
    """
    n = max(1, int(round(duration_sec)))
    motion = np.zeros(n, dtype=np.float32)

    if not dense_faces or len(dense_faces) < 2:
        return motion

    # Build per-second average face x
    sec_x = {}
    for df in dense_faces:
        t = int(df.timestamp)
        if 0 <= t < n and df.faces:
            xs = [getattr(f, 'x', 50) for f in df.faces if getattr(f, 'identity_id', -1) >= 0]
            if xs:
                sec_x[t] = sum(xs) / len(xs)

    # Frame-to-frame delta
    prev_x = None
    for t in range(n):
        if t in sec_x:
            if prev_x is not None:
                motion[t] = abs(sec_x[t] - prev_x)
            prev_x = sec_x[t]

    # Self-calibrate: normalize against top 5% magnitude
    if np.max(motion) > 0:
        top_5pct = float(np.percentile(motion[motion > 0], 95)) if np.any(motion > 0) else 1.0
        if top_5pct > 0:
            motion = motion / top_5pct
            motion = np.clip(motion, 0.0, 1.0)

    return motion


def _median_filter_1d(arr: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Simple 1D median filter for spike suppression."""
    if len(arr) < kernel_size:
        return arr.copy()
    result = arr.copy()
    half = kernel_size // 2
    for i in range(half, len(arr) - half):
        result[i] = np.median(arr[i - half:i + half + 1])
    return result
