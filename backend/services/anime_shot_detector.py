"""Phase 6 — Anime / cartoon shot detection.

Anime defeats PySceneDetect's default ``ContentDetector``: flat color
regions, 2-on-3 holds (the same drawing held for 2-3 frames before
the next), and limited motion all pull the per-frame HSV-MSE delta
toward zero, so the detector misses real cuts and false-positives
on lip-sync flicker.

This module ships a complementary detector that uses two signals
better suited to anime:

  1. **Histogram correlation** — for each adjacent frame pair,
     compute the correlation (Pearson) of their normalized
     8-bin grayscale histograms. Anime cuts produce sharp drops
     in correlation because the new shot's color palette is
     usually different even when the motion delta is small.
     Live-action cuts also show this signal, but the threshold
     is tuned for anime's sparse-color regime.

  2. **Edge density delta** — Sobel-magnitude average per frame
     gives a "drawing complexity" signal. Anime cuts swap from
     a sparse keyframe to a denser background to a different
     character composition; the absolute delta in edge density
     captures the shot change even when the histogram is
     coincidentally similar.

A frame pair is flagged as a cut when EITHER:
  - histogram correlation drops below ``hist_corr_threshold``
    (default 0.55), OR
  - edge density delta exceeds ``edge_delta_threshold`` (default
    0.30 of the rolling mean).

Both signals are computed on **down-sampled** grayscale crops
(default 160 × 90) so the detector is fast even on long runs.

The numpy / OpenCV dependencies are imported lazily so the module
loads in a sandbox without them — the pure-Python helpers
(``score_pair_metrics``, ``cut_indices_from_metrics``) work
without the imaging stack and are unit-testable in isolation.

Production wiring is a follow-up: ``shot_detector.detect_shots``
will call ``detect_anime_shots`` when ``is_animated=True`` is set
on the content profile. For Phase 6 minimal we ship the detector
+ tests; ``shot_detector.py`` itself is unchanged so the existing
PySceneDetect path keeps producing the same numbers on non-anime
fixtures.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# ──────────────────── Feature flag ────────────────────

USE_ANIME_SHOT_DETECTOR = os.environ.get(
    "CLIPAI_ANIME_SHOT_DETECTOR", "1",
).lower() in ("1", "true", "yes", "on")


# ──────────────────── Tuning constants ────────────────────

# Histogram correlation threshold below which a pair is flagged
# as a cut. 0.55 is conservative — anime shot pairs usually
# correlate above 0.80 within a shot and below 0.40 across cuts.
HIST_CORR_THRESHOLD = 0.55

# Edge-density delta threshold (relative to rolling mean) above
# which a pair is flagged as a cut. 0.30 = "edge density of the
# new frame is 30 % higher or lower than the recent average".
EDGE_DELTA_THRESHOLD = 0.30

# Down-sample target for both signals. Matches the existing
# OpenCV fallback in shot_detector.py so the two detectors see
# the same spatial granularity.
DOWNSAMPLE_W = 160
DOWNSAMPLE_H = 90

# Number of histogram bins. 8 is a coarse-but-fast default that
# preserves shot-level color signatures without overfitting on
# JPEG noise.
HIST_BINS = 8

# Minimum gap between consecutive cuts (seconds). Below this we
# coalesce into a single cut to suppress the rapid-fire false
# positives that 2-on-3 holds produce.
MIN_CUT_GAP_SEC = 0.30


# ──────────────────── Per-pair metrics dataclass ────────────────────


@dataclass
class PairMetrics:
    """Metrics for one adjacent frame pair.

    Attributes:
        index: Index of the SECOND frame in the pair (the "after"
            frame). The "before" frame is implicitly ``index - 1``.
        timestamp: Timestamp of the second frame (seconds).
        hist_corr: Pearson correlation of the two frames'
            normalized grayscale histograms in [-1, 1]. Higher =
            more similar. Anime within-shot pairs typically score
            > 0.80, cuts < 0.40.
        edge_density: Edge density of the second frame in [0, 1]
            (Sobel magnitude average / 255).
        edge_delta: Absolute fractional delta of the edge density
            relative to the rolling mean over the previous N
            frames.
    """

    index: int
    timestamp: float
    hist_corr: float
    edge_density: float
    edge_delta: float


@dataclass
class AnimeShotResult:
    """Result of an anime-shot-detector run.

    Attributes:
        cut_times: Sorted list of cut timestamps (seconds).
        pair_metrics: One ``PairMetrics`` per adjacent frame pair.
            Useful for telemetry and threshold tuning.
        n_frames_scanned: Total number of frames the detector saw.
        skipped_reason: Non-empty when the detector bailed out
            (missing numpy, missing video file, etc).
    """

    cut_times: list[float] = field(default_factory=list)
    pair_metrics: list[PairMetrics] = field(default_factory=list)
    n_frames_scanned: int = 0
    skipped_reason: str = ""

    @property
    def has_data(self) -> bool:
        return self.n_frames_scanned > 0


# ──────────────────── Pure-Python helpers ────────────────────


def histogram_correlation(
    hist_a: list[float],
    hist_b: list[float],
) -> float:
    """Pearson correlation of two normalized histograms.

    Returns 1.0 for identical, -1.0 for perfectly anti-correlated,
    0.0 for uncorrelated. Returns 1.0 for two empty histograms
    and 0.0 for histograms of mismatched length.
    """
    if len(hist_a) != len(hist_b):
        return 0.0
    n = len(hist_a)
    if n == 0:
        return 1.0
    mean_a = sum(hist_a) / n
    mean_b = sum(hist_b) / n
    num = 0.0
    den_a = 0.0
    den_b = 0.0
    for a, b in zip(hist_a, hist_b):
        da = a - mean_a
        db = b - mean_b
        num += da * db
        den_a += da * da
        den_b += db * db
    den = (den_a * den_b) ** 0.5
    if den <= 0:
        # Degenerate cases: if both histograms are flat (zero
        # variance) treat them as perfectly correlated.
        if den_a == 0 and den_b == 0:
            return 1.0
        return 0.0
    return num / den


def score_pair_metrics(
    hist_pairs: list[tuple[list[float], list[float]]],
    edge_densities: list[float],
    timestamps: list[float],
    *,
    rolling_window: int = 5,
) -> list[PairMetrics]:
    """Compute per-pair metrics from extracted hist + edge data.

    This is the numpy-free entry point — it accepts already-
    computed histograms and edge densities so unit tests can
    exercise the threshold logic without an imaging stack.

    Args:
        hist_pairs: List of ``(hist_prev, hist_cur)`` tuples,
            length N-1 where N is the number of frames scanned.
        edge_densities: List of length N — edge density per frame.
        timestamps: List of length N — timestamp per frame.
        rolling_window: Window size for the edge-delta running
            mean. Default 5 frames matches the spec's 2-on-3
            anime hold cadence.

    Returns:
        List of ``PairMetrics``, one per pair. The first pair's
        edge_delta uses the actual frame-1 density (no history).
    """
    n_pairs = len(hist_pairs)
    if n_pairs == 0:
        return []
    if len(edge_densities) != n_pairs + 1:
        raise ValueError(
            f"edge_densities length {len(edge_densities)} != "
            f"hist_pairs length+1 ({n_pairs + 1})"
        )
    if len(timestamps) != n_pairs + 1:
        raise ValueError(
            f"timestamps length {len(timestamps)} != "
            f"hist_pairs length+1 ({n_pairs + 1})"
        )
    out: list[PairMetrics] = []
    rolling: list[float] = [edge_densities[0]]
    for i, (hist_prev, hist_cur) in enumerate(hist_pairs):
        cur_idx = i + 1
        cur_density = float(edge_densities[cur_idx])
        # Rolling mean over the previous ``rolling_window`` densities
        roll_mean = sum(rolling) / len(rolling) if rolling else cur_density
        edge_delta = (
            abs(cur_density - roll_mean) / max(roll_mean, 1e-6)
        )
        out.append(PairMetrics(
            index=cur_idx,
            timestamp=float(timestamps[cur_idx]),
            hist_corr=histogram_correlation(hist_prev, hist_cur),
            edge_density=cur_density,
            edge_delta=edge_delta,
        ))
        rolling.append(cur_density)
        if len(rolling) > rolling_window:
            rolling.pop(0)
    return out


def cut_indices_from_metrics(
    metrics: list[PairMetrics],
    *,
    hist_corr_threshold: float = HIST_CORR_THRESHOLD,
    edge_delta_threshold: float = EDGE_DELTA_THRESHOLD,
    min_cut_gap_sec: float = MIN_CUT_GAP_SEC,
) -> list[int]:
    """Walk per-pair metrics → sorted list of pair indices flagged as cuts.

    Cut criterion: histogram correlation < ``hist_corr_threshold``
    OR edge delta > ``edge_delta_threshold``. Consecutive cuts
    within ``min_cut_gap_sec`` are coalesced — only the first is
    kept — to suppress 2-on-3 hold flicker.
    """
    cuts: list[int] = []
    last_t = -float("inf")
    for m in metrics:
        is_cut = (
            m.hist_corr < hist_corr_threshold
            or m.edge_delta > edge_delta_threshold
        )
        if not is_cut:
            continue
        if m.timestamp - last_t < min_cut_gap_sec:
            continue
        cuts.append(m.index)
        last_t = m.timestamp
    return cuts


def cut_times_from_metrics(
    metrics: list[PairMetrics],
    *,
    hist_corr_threshold: float = HIST_CORR_THRESHOLD,
    edge_delta_threshold: float = EDGE_DELTA_THRESHOLD,
    min_cut_gap_sec: float = MIN_CUT_GAP_SEC,
) -> list[float]:
    """Convenience: return the cut timestamps directly."""
    cut_indices = cut_indices_from_metrics(
        metrics,
        hist_corr_threshold=hist_corr_threshold,
        edge_delta_threshold=edge_delta_threshold,
        min_cut_gap_sec=min_cut_gap_sec,
    )
    by_idx = {m.index: m.timestamp for m in metrics}
    return [by_idx[i] for i in cut_indices]


# ──────────────────── OpenCV-backed entry point ────────────────────


def detect_anime_shots(
    video_path: str,
    *,
    video_duration: Optional[float] = None,
    sample_step: int = 3,
    hist_corr_threshold: float = HIST_CORR_THRESHOLD,
    edge_delta_threshold: float = EDGE_DELTA_THRESHOLD,
    min_cut_gap_sec: float = MIN_CUT_GAP_SEC,
) -> AnimeShotResult:
    """Run the anime shot detector on a video file.

    Lazily imports OpenCV + numpy so this module loads in a
    sandbox without them. Returns an empty
    ``AnimeShotResult(skipped_reason=...)`` when the imports
    fail or the video can't be opened.

    Args:
        video_path: Path to the source video.
        video_duration: Optional precomputed duration in seconds.
        sample_step: Frame stride for sampling. Default 3 matches
            the OpenCV fallback in shot_detector.py — we look at
            every 3rd frame at 30 fps = 10 fps which captures
            most anime cut cadence.
        hist_corr_threshold / edge_delta_threshold / min_cut_gap_sec:
            Tunables forwarded to ``cut_indices_from_metrics``.

    Returns:
        ``AnimeShotResult`` with ``cut_times``, per-pair telemetry,
        and a ``skipped_reason`` when the detector couldn't run.
    """
    try:
        import cv2  # noqa: F401
        import numpy as np  # noqa: F401
    except ImportError as exc:
        return AnimeShotResult(skipped_reason=f"missing deps: {exc}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return AnimeShotResult(skipped_reason=f"cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    timestamps: list[float] = []
    histograms: list[list[float]] = []
    edge_densities: list[float] = []
    n_frames = 0
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % max(int(sample_step), 1) == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (DOWNSAMPLE_W, DOWNSAMPLE_H))
                # Histogram (8 bins, normalized)
                hist = cv2.calcHist([gray], [0], None, [HIST_BINS], [0, 256])
                hist = hist.flatten()
                total = float(hist.sum())
                if total > 0:
                    hist = (hist / total).tolist()
                else:
                    hist = [0.0] * HIST_BINS
                histograms.append(hist)
                # Edge density: Sobel magnitude average / 255
                sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                mag = (sobel_x ** 2 + sobel_y ** 2) ** 0.5
                density = float(mag.mean()) / 255.0
                edge_densities.append(density)
                timestamps.append(frame_idx / fps)
                n_frames += 1
            frame_idx += 1
    finally:
        cap.release()

    if n_frames < 2:
        return AnimeShotResult(
            n_frames_scanned=n_frames,
            skipped_reason="too few frames sampled",
        )

    hist_pairs = [
        (histograms[i], histograms[i + 1])
        for i in range(n_frames - 1)
    ]
    metrics = score_pair_metrics(hist_pairs, edge_densities, timestamps)
    cut_times = cut_times_from_metrics(
        metrics,
        hist_corr_threshold=hist_corr_threshold,
        edge_delta_threshold=edge_delta_threshold,
        min_cut_gap_sec=min_cut_gap_sec,
    )

    logger.info(
        "AnimeShotDetector: %d frames scanned, %d cuts (hist_thresh=%.2f, edge_thresh=%.2f)",
        n_frames, len(cut_times), hist_corr_threshold, edge_delta_threshold,
    )
    return AnimeShotResult(
        cut_times=cut_times,
        pair_metrics=metrics,
        n_frames_scanned=n_frames,
    )
