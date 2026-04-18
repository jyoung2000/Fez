"""Phase 3 — glance-trigger sources for the HUD glance stream.

Three independent trigger types that all emit the common
:class:`GlanceTrigger` shape. Callers wire them up once per
scene, feed frames + optional audio samples into each one per
frame, and pass the union of their outputs into
:meth:`HudGlanceStream.update`.

Trigger types:

  * **PixelDeltaTrigger** — running-mean vs. current-frame MAD
    inside each HUD bbox. Fires on kill-feed text changes,
    health-bar depletion, killcam flashes, etc.

  * **AudioPeakTrigger** — short-time audio envelope anomaly
    detection with σ-based thresholding. Used as a
    co-occurrence signal: within ±``audio_peak_window_ms`` of a
    pixel-delta candidate, sensitivity for health / killfeed
    regions is boosted. Also fires standalone when a loud
    sound has no pixel correlate (e.g. ult call on audio
    without an HUD spark) to opportunistically glance at the
    highest-confidence region.

  * **ScheduledFallback** — if no other trigger has fired in
    the last ``scheduled_glance_interval_s`` seconds, emit a
    synthetic trigger on the highest-confidence HUD region
    that hasn't been glanced at recently. Guarantees the
    viewer gets periodic HUD context even on quiet stretches.

The three can be combined via :class:`GlanceTriggerBus` — a
tiny dispatcher that runs each trigger on the current frame
and dedupes via a per-region refractory window.

The audio branch has an explicit interface stub so this module
doesn't hard-depend on ``audio_analyzer.py`` wiring that may
not be plumbed for every clip; if no envelope is passed, the
audio path is simply skipped.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal, Optional

logger = logging.getLogger(__name__)


TriggerSource = Literal["pixel_delta", "audio_peak", "scheduled"]


@dataclass
class GlanceTrigger:
    """A single glance-fire event.

    ``region_id`` is the Python ``id()`` of the target
    :class:`HudRegion` — keeps the interface agnostic to the
    region class and lets :class:`HudGlanceStream` dispatch in
    O(1). ``peak_weight`` overrides the stream default when
    non-zero.
    """

    region_id: int
    peak_weight: float
    source: TriggerSource
    frame_idx: int
    confidence: float = 1.0
    label: str = ""


# ──────────────────── Helper ────────────────────


def _crop_mean(img, bbox: tuple[int, int, int, int]) -> float:
    """Return the mean pixel value of ``img`` inside ``bbox``.

    Tolerates bboxes that overrun the frame edge (clamps). Uses
    numpy when available; falls back to a pure-Python sum for
    small fixtures.
    """
    x, y, w, h = bbox
    try:
        import numpy as np  # type: ignore
    except ImportError:
        np = None  # type: ignore

    if np is not None and hasattr(img, "shape"):
        H, W = img.shape[:2]
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(W, int(x + w))
        y1 = min(H, int(y + h))
        if x0 >= x1 or y0 >= y1:
            return 0.0
        return float(img[y0:y1, x0:x1].mean())

    # Pure-Python fallback for list-of-lists inputs used by tests.
    rows = img[y:y + h]
    total = 0.0
    n = 0
    for row in rows:
        sub = row[x:x + w]
        total += sum(sub)
        n += len(sub)
    return (total / n) if n else 0.0


# ──────────────────── 1. Pixel-delta trigger ────────────────────


class PixelDeltaTrigger:
    """Per-region running-mean pixel-delta detector.

    For each :class:`HudRegion` we store a single running-mean
    scalar (mean intensity inside the bbox). On each new frame
    we compute the mean absolute difference between the
    region's current mean and the running mean; values above
    ``threshold`` emit a trigger, and the running mean is then
    updated with α = ``ema_alpha`` (default 0.1).
    """

    def __init__(
        self,
        hud_regions: list,
        *,
        threshold: float = 8.0,
        ema_alpha: float = 0.1,
    ):
        self.regions = list(hud_regions)
        self.threshold = float(threshold)
        self.alpha = float(ema_alpha)
        self._running_mean: dict[int, Optional[float]] = {
            id(r): None for r in self.regions
        }

    def set_threshold(self, value: float) -> None:
        self.threshold = float(value)

    def process(
        self,
        frame,
        frame_idx: int,
        *,
        region_threshold_boost: Optional[dict[int, float]] = None,
    ) -> list[GlanceTrigger]:
        """Run the detector on one frame.

        Args:
            frame: HxW grayscale numpy array (or nested list).
            frame_idx: Current frame index.
            region_threshold_boost: Optional per-region
                threshold multiplier in (0, 1]. Used by the
                :class:`GlanceTriggerBus` to raise sensitivity
                for health / killfeed regions near audio peaks
                (a lower effective threshold = fires more
                easily).

        Returns:
            Zero or more :class:`GlanceTrigger` for regions
            whose MAD crossed the (possibly boosted) threshold.
        """
        triggers: list[GlanceTrigger] = []
        for r in self.regions:
            rid = id(r)
            bbox = tuple(getattr(r, "bbox", (0, 0, 0, 0)))
            try:
                current = _crop_mean(frame, bbox)  # type: ignore[arg-type]
            except Exception as e:
                logger.debug("PixelDeltaTrigger: _crop_mean failed: %s", e)
                continue
            running = self._running_mean[rid]
            thresh = self.threshold
            if region_threshold_boost and rid in region_threshold_boost:
                # Multiplier < 1 lowers the threshold (boosts sensitivity).
                thresh *= float(region_threshold_boost[rid])

            if running is None:
                self._running_mean[rid] = current
                continue
            mad = abs(current - running)
            if mad >= thresh:
                triggers.append(GlanceTrigger(
                    region_id=rid,
                    peak_weight=0.0,  # defer to stream default
                    source="pixel_delta",
                    frame_idx=int(frame_idx),
                    confidence=min(1.0, mad / max(thresh, 1e-6)),
                    label=getattr(r, "semantic_hint", "") or "",
                ))
            # Always update the running mean so the detector
            # eventually adapts to the new baseline after a
            # persistent change.
            self._running_mean[rid] = (
                (1.0 - self.alpha) * running + self.alpha * current
            )
        return triggers


# ──────────────────── 2. Audio-peak trigger (interface) ────────────────────


class AudioPeakTrigger:
    """Short-time audio envelope anomaly detector.

    This is the minimal viable implementation: the caller is
    expected to pass a list of ``(timestamp, envelope)``
    samples covering the scene (same shape that
    ``audio_analyzer.py`` already produces). The trigger
    computes a rolling mean + std on a 2 s window and flags
    any sample whose envelope is more than ``sigma``
    deviations above the rolling mean as a candidate peak.

    It does NOT fire triggers on its own — its role is to
    return ``peak_timestamps`` that :class:`GlanceTriggerBus`
    uses to boost pixel-delta sensitivity inside the audio
    window. When no envelope is provided (common on B-roll
    clips without an audio pipeline), the class degrades to a
    no-op.
    """

    def __init__(
        self,
        *,
        sigma: float = 2.5,
        window_s: float = 2.0,
    ):
        self.sigma = float(sigma)
        self.window_s = float(window_s)
        self._peak_times: list[float] = []

    def compute_peaks(
        self,
        envelope: list[tuple[float, float]],
    ) -> list[float]:
        """Return a list of peak timestamps in seconds."""
        if not envelope:
            self._peak_times = []
            return []
        sorted_env = sorted(envelope, key=lambda p: p[0])
        peaks: list[float] = []
        # Rolling mean / std via a simple deque pattern.
        window: list[float] = []
        window_t: list[float] = []
        for t, v in sorted_env:
            # Evict old samples
            while window_t and (t - window_t[0]) > self.window_s:
                window.pop(0)
                window_t.pop(0)
            if len(window) >= 3:
                mean = sum(window) / len(window)
                var = sum((x - mean) ** 2 for x in window) / len(window)
                std = math.sqrt(var)
                # σ-based threshold for noisy backgrounds, OR a
                # 2× relative threshold for near-flat ones (where
                # std collapses to ~0 and would otherwise mask
                # perfectly obvious spikes).
                sigma_threshold = (
                    mean + self.sigma * std if std > 1e-9 else float("inf")
                )
                relative_threshold = max(mean * 2.0, mean + 0.5)
                if v >= min(sigma_threshold, relative_threshold):
                    peaks.append(float(t))
            window.append(float(v))
            window_t.append(float(t))
        self._peak_times = peaks
        return peaks

    @property
    def peak_times(self) -> list[float]:
        return list(self._peak_times)

    def is_near_peak(self, timestamp: float, window_ms: float = 200.0) -> bool:
        half = window_ms / 1000.0 / 2.0
        for pt in self._peak_times:
            if abs(pt - timestamp) <= half:
                return True
        return False


# ──────────────────── 3. Scheduled fallback ────────────────────


class ScheduledGlanceFallback:
    """Synthetic glance emitter for quiet stretches.

    Fires a trigger on the highest-confidence HUD region that
    (a) hasn't been glanced at in the last
    ``scheduled_glance_interval_s`` seconds AND (b) whose last
    glance is older than the refractory floor. If no region
    satisfies both, nothing fires for this frame.
    """

    def __init__(
        self,
        hud_regions: list,
        *,
        interval_s: float = 10.0,
        frame_rate: float = 30.0,
    ):
        self.regions = list(hud_regions)
        self.interval_s = float(interval_s)
        self.frame_rate = max(1.0, float(frame_rate))
        # Last trigger frame per region — initialized to "never
        # fired" sentinel but treated as 0 for dt computation so
        # the very first frame doesn't count as "interval seconds
        # idle."
        self._last_frame: dict[int, int] = {
            id(r): 0 for r in self.regions
        }
        self._last_any_trigger_frame: int = 0

    def note_trigger(self, region_id: int, frame_idx: int) -> None:
        """Called by the bus whenever any other trigger fires
        so the fallback can back off."""
        self._last_frame[region_id] = int(frame_idx)
        self._last_any_trigger_frame = int(frame_idx)

    def process(self, frame_idx: int) -> list[GlanceTrigger]:
        dt_frames = max(0, frame_idx - self._last_any_trigger_frame)
        dt_s = dt_frames / self.frame_rate
        if dt_s < self.interval_s:
            return []

        # Pick the best region to synthesize a glance for —
        # highest confidence, least recently visited.
        best: Optional[tuple[float, object]] = None
        for r in self.regions:
            conf = float(getattr(r, "confidence", 0.0))
            last = self._last_frame.get(id(r), -10_000_000)
            # Score prefers high confidence AND longer gap.
            age_s = (frame_idx - last) / self.frame_rate
            # Exclude regions that were just visited (refractory).
            if age_s < self.interval_s * 0.5:
                continue
            score = conf * min(age_s / self.interval_s, 3.0)
            if best is None or score > best[0]:
                best = (score, r)

        if best is None:
            return []
        _score, chosen = best
        rid = id(chosen)
        self._last_frame[rid] = int(frame_idx)
        self._last_any_trigger_frame = int(frame_idx)
        return [GlanceTrigger(
            region_id=rid,
            peak_weight=0.0,
            source="scheduled",
            frame_idx=int(frame_idx),
            confidence=float(getattr(chosen, "confidence", 1.0)),
            label=str(
                getattr(chosen, "semantic_hint", "")
                or getattr(chosen, "label", "")
                or "scheduled",
            ),
        )]


# ──────────────────── Bus ────────────────────


class GlanceTriggerBus:
    """Compose pixel delta + audio peak + scheduled fallback.

    Maintains a per-region refractory window so two triggers on
    the same region within a short window merge into one
    glance. Applies audio-peak boosts to health / killfeed
    regions for frames near an audio peak.
    """

    def __init__(
        self,
        hud_regions: list,
        *,
        frame_rate: float = 30.0,
        pixel_delta_threshold: float = 8.0,
        scheduled_interval_s: float = 10.0,
        refractory_s: float = 2.0,
        audio_boost: float = 2.0,  # higher = more sensitivity inside window
    ):
        self.regions = list(hud_regions)
        self.frame_rate = max(1.0, float(frame_rate))
        self.refractory_frames = int(round(refractory_s * self.frame_rate))
        self.pixel = PixelDeltaTrigger(
            hud_regions, threshold=pixel_delta_threshold,
        )
        self.audio = AudioPeakTrigger()
        self.scheduled = ScheduledGlanceFallback(
            hud_regions,
            interval_s=scheduled_interval_s,
            frame_rate=self.frame_rate,
        )
        self._last_fired: dict[int, int] = {
            id(r): -10_000_000 for r in self.regions
        }
        self._audio_boost = float(audio_boost)

    def set_audio_envelope(
        self, envelope: list[tuple[float, float]],
    ) -> None:
        """Pre-compute audio peaks across the whole scene."""
        self.audio.compute_peaks(envelope)

    def _boosts_for_frame(
        self, timestamp: float,
    ) -> dict[int, float]:
        """Return per-region threshold multipliers for frames
        inside an audio-peak window. Values < 1.0 lower the
        threshold (= higher sensitivity)."""
        boosts: dict[int, float] = {}
        if not self.audio.peak_times:
            return boosts
        if not self.audio.is_near_peak(timestamp):
            return boosts
        boost_mult = 1.0 / max(self._audio_boost, 1e-6)
        for r in self.regions:
            hint = getattr(r, "semantic_hint", "")
            if hint in ("health", "killfeed"):
                boosts[id(r)] = boost_mult
        return boosts

    def process_frame(
        self,
        frame,
        *,
        frame_idx: int,
        timestamp: float,
    ) -> list[GlanceTrigger]:
        triggers: list[GlanceTrigger] = []

        # 1. Pixel-delta with optional audio boost.
        boosts = self._boosts_for_frame(timestamp)
        pd = self.pixel.process(
            frame, frame_idx, region_threshold_boost=boosts,
        )
        for trig in pd:
            if not self._in_refractory(trig.region_id, frame_idx):
                triggers.append(trig)
                self._last_fired[trig.region_id] = frame_idx
                self.scheduled.note_trigger(trig.region_id, frame_idx)

        # 2. Scheduled fallback — only if pixel delta didn't
        # fire anything this frame.
        if not triggers:
            sched = self.scheduled.process(frame_idx)
            for trig in sched:
                if not self._in_refractory(trig.region_id, frame_idx):
                    triggers.append(trig)
                    self._last_fired[trig.region_id] = frame_idx

        return triggers

    def _in_refractory(self, region_id: int, frame_idx: int) -> bool:
        last = self._last_fired.get(region_id, -10_000_000)
        return (frame_idx - last) < self.refractory_frames
