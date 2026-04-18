"""Camera mode selector: AutoFlip-style hierarchical mode selection.

For each shot, picks the cheapest camera mode that keeps all required
features in frame:
  1. STATIONARY -- subject barely moves, static crop
  2. TRACKING -- subject moves coherently, camera follows
  3. PANNING -- subject moves linearly, camera pans
  4. PADDING -- content cannot be cropped without clipping, use blur_fill
"""

import logging
from enum import Enum
from typing import Callable, Optional

from backend.services.focus_model import SceneFocusRegion

logger = logging.getLogger(__name__)

SHORT_SHOT_THRESHOLD_SEC = 1.0  # shots under this always get STATIONARY


class CameraMode(str, Enum):
    STATIONARY = "stationary"
    TRACKING = "tracking"
    PANNING = "panning"
    PADDING = "padding"


def select_camera_mode(
    focus: SceneFocusRegion,
    motion_energy_fn: Optional[Callable] = None,
    target_aspect: float = 9 / 16,
    source_width: int = 1920,
    source_height: int = 1080,
    job_id: str = "",
) -> CameraMode:
    """AutoFlip-style hierarchical mode selection, cheapest mode first.

    Order of preference:
      1. STATIONARY if bounding rect fits and per-frame targets stay within
         a stationary window of width `crop_width_pct` centered on optimal_crop_center.
      2. TRACKING if per-frame targets move coherently (low acceleration) and
         the full motion envelope fits within the target aspect window.
      3. PANNING if motion is linear (high direction consistency, low turn-around count).
      4. PADDING fallback otherwise -- content cannot be cropped without clipping required features.
    """
    # Compute crop width in percentage space
    src_aspect = source_width / source_height if source_height > 0 else 16 / 9
    if target_aspect < src_aspect:
        crop_width_pct = (target_aspect / src_aspect) * 100
    else:
        crop_width_pct = 100.0

    # Collect hard-required features for Rule B
    hard_required = [rf for rf in focus.required if rf.must_be_in_frame]

    # Compute aspect mismatch for Rule C
    aspect_mismatch = abs(src_aspect - target_aspect) / max(src_aspect, target_aspect)

    # ── Rule A: Short-shot override ──
    # For very short shots (<1 second), any camera motion is distracting
    # regardless of subject movement. Always use STATIONARY on the best
    # available crop center.
    shot_duration = focus.shot_end - focus.shot_start
    if shot_duration < SHORT_SHOT_THRESHOLD_SEC:
        if focus.fits_target_aspect:
            logger.info("[%s] Camera mode: STATIONARY (short shot, %.2fs)", job_id, shot_duration)
            return CameraMode.STATIONARY
        else:
            logger.info("[%s] Camera mode: PADDING (short shot can't fit, %.2fs)", job_id, shot_duration)
            return CameraMode.PADDING

    # ── Rule B: Multi-feature geometric infeasibility ──
    # When the shot has multiple hard-required features that need to appear
    # simultaneously (same timestamp) and their combined bounding rect
    # exceeds the crop window, PADDING is the only correct answer.
    # A single subject walking across the frame has one feature per timestamp,
    # which is handled fine by TRACKING — Rule B only catches genuinely
    # multi-subject frames.
    if hard_required and not focus.fits_target_aspect:
        from collections import defaultdict as _defaultdict_b
        _by_time = _defaultdict_b(list)
        for rf in hard_required:
            _by_time[rf.t_start].append(rf)
        _has_simultaneous_multi = any(len(fs) >= 2 for fs in _by_time.values())
        if _has_simultaneous_multi:
            logger.info("[%s] Camera mode: PADDING (simultaneous multi-feature doesn't fit)",
                        job_id)
            return CameraMode.PADDING

    # No per-frame targets means no motion data -- STATIONARY if fits, PADDING otherwise
    if len(focus.per_frame_target) < 2:
        if not focus.fits_target_aspect:
            logger.info("[%s] Camera mode: PADDING (insufficient targets and doesn't fit)", job_id)
            return CameraMode.PADDING
        logger.info("[%s] Camera mode: STATIONARY (insufficient per-frame targets: %d)",
                    job_id, len(focus.per_frame_target))
        return CameraMode.STATIONARY

    # Check per-frame feature spread: do any individual frames have required
    # features wider than the crop? If so, PADDING is the only option.
    _per_frame_fits = True
    if focus.required:
        from collections import defaultdict
        by_time = defaultdict(list)
        for rf in focus.required:
            by_time[rf.t_start].append(rf)
        for t, features in by_time.items():
            if len(features) < 2:
                continue
            lefts = [rf.left for rf in features]
            rights = [rf.right for rf in features]
            frame_spread = max(rights) - min(lefts)
            if frame_spread > crop_width_pct:
                _per_frame_fits = False
                break

    if not _per_frame_fits:
        logger.info("[%s] Camera mode: PADDING (per-frame feature spread exceeds crop width)", job_id)
        return CameraMode.PADDING

    # Extract x positions from per-frame targets
    target_xs = [t[1] for t in focus.per_frame_target]
    max_x = max(target_xs)
    min_x = min(target_xs)
    x_range = max_x - min_x

    # ── Rule C: Aspect-aware stationary tolerance ──
    # The more extreme the source-to-target aspect mismatch, the tighter
    # the crop, and the less tolerance we have for drift before switching
    # to TRACKING. Scale down from 10% when aspect mismatch is large.
    stationary_tolerance = crop_width_pct * 0.10 * (1.0 - min(aspect_mismatch * 0.5, 0.5))

    # ── STATIONARY test ──
    # If subjects barely move (within tolerance of crop width), static crop suffices
    # AND the overall bounding rect fits in a single static window
    if x_range <= stationary_tolerance and focus.fits_target_aspect:
        logger.info("[%s] Camera mode: STATIONARY (x_range=%.1f <= tolerance=%.1f)",
                    job_id, x_range, stationary_tolerance)
        return CameraMode.STATIONARY

    # ── TRACKING test ──
    # Compute second derivative (acceleration) numerically
    mean_accel_log = 0.0
    r_squared_log = 0.0
    chosen_mode = CameraMode.PADDING  # default fallback

    if len(target_xs) >= 3:
        timestamps = [t[0] for t in focus.per_frame_target]
        accels = []
        for i in range(1, len(target_xs) - 1):
            dt1 = timestamps[i] - timestamps[i - 1]
            dt2 = timestamps[i + 1] - timestamps[i]
            if dt1 > 0 and dt2 > 0:
                v1 = (target_xs[i] - target_xs[i - 1]) / dt1
                v2 = (target_xs[i + 1] - target_xs[i]) / dt2
                accel = abs(v2 - v1) / ((dt1 + dt2) / 2)
                accels.append(accel)

        if accels:
            mean_accel_log = sum(accels) / len(accels)
            # Threshold: 5%/sec^2 -- smooth motion
            if mean_accel_log < 5.0:
                logger.info("[%s] Camera mode: TRACKING (mean_accel=%.2f, x_range=%.1f)",
                            job_id, mean_accel_log, x_range)
                chosen_mode = CameraMode.TRACKING
                logger.info(
                    "[%s] mode_decision shot=%.2f-%.2f chosen=%s x_range=%.1f "
                    "tolerance=%.1f mean_accel=%.2f r_sq=%.3f aspect_mismatch=%.2f "
                    "n_required=%d fits=%s",
                    job_id, focus.shot_start, focus.shot_end, chosen_mode.value,
                    x_range, stationary_tolerance, mean_accel_log, r_squared_log,
                    aspect_mismatch, len(hard_required), focus.fits_target_aspect,
                )
                return chosen_mode

    # ── PANNING test ──
    # Fit linear regression x = a*t + b, check R^2
    # Rule D: hysteresis — raise threshold from 0.85 to 0.90 to prevent
    # mode oscillation in the 0.85-0.90 band.
    if len(target_xs) >= 3:
        timestamps = [t[0] for t in focus.per_frame_target]
        n = len(timestamps)
        mean_t = sum(timestamps) / n
        mean_x = sum(target_xs) / n

        ss_tt = sum((t - mean_t) ** 2 for t in timestamps)
        ss_tx = sum((t - mean_t) * (x - mean_x) for t, x in zip(timestamps, target_xs))
        ss_xx = sum((x - mean_x) ** 2 for x in target_xs)

        if ss_tt > 0 and ss_xx > 0:
            a = ss_tx / ss_tt
            b = mean_x - a * mean_t
            ss_res = sum((x - (a * t + b)) ** 2 for t, x in zip(timestamps, target_xs))
            r_squared_log = 1 - ss_res / ss_xx if ss_xx > 0 else 0

            if r_squared_log > 0.90:
                logger.info("[%s] Camera mode: PANNING (R^2=%.3f clear linear, slope=%.2f)",
                            job_id, r_squared_log, a)
                chosen_mode = CameraMode.PANNING
                logger.info(
                    "[%s] mode_decision shot=%.2f-%.2f chosen=%s x_range=%.1f "
                    "tolerance=%.1f mean_accel=%.2f r_sq=%.3f aspect_mismatch=%.2f "
                    "n_required=%d fits=%s",
                    job_id, focus.shot_start, focus.shot_end, chosen_mode.value,
                    x_range, stationary_tolerance, mean_accel_log, r_squared_log,
                    aspect_mismatch, len(hard_required), focus.fits_target_aspect,
                )
                return chosen_mode

    # ── PADDING fallback ──
    chosen_mode = CameraMode.PADDING
    logger.info("[%s] Camera mode: PADDING (no mode fits -- x_range=%.1f, crop_width=%.1f)",
                job_id, x_range, crop_width_pct)
    logger.info(
        "[%s] mode_decision shot=%.2f-%.2f chosen=%s x_range=%.1f "
        "tolerance=%.1f mean_accel=%.2f r_sq=%.3f aspect_mismatch=%.2f "
        "n_required=%d fits=%s",
        job_id, focus.shot_start, focus.shot_end, chosen_mode.value,
        x_range, stationary_tolerance, mean_accel_log, r_squared_log,
        aspect_mismatch, len(hard_required), focus.fits_target_aspect,
    )
    return chosen_mode
