"""Per-frame crosshair coordinate detection for FPS / hero shooter clips.

The legacy ``face_detector.detect_crosshair_persistence`` returns a 0-1
score for "this clip has a persistent crosshair" but throws away the
actual position. This module ships a tracker that returns
``(x_pct, y_pct, confidence)`` per frame so the reframe pipeline can
use the crosshair as the per-frame subject anchor instead of the
hard-coded ``(50, 50)``.

Two production tiers:

- **OpenCV-backed template matcher**: lazily imports cv2 and runs
  ``matchTemplate`` against a small bank of synthetic crosshair
  patterns (dot, plus, T, X, circle) at multiple scales. Confidence
  is the peak normalized cross-correlation. Searches a window
  around a prior centroid so per-frame cost stays bounded.

- **Pure-Python fallback**: scores the central low-temporal-variance
  patch as a "persistent crosshair" without trying to localize it,
  matching the legacy ``detect_crosshair_persistence`` behavior.
  Used by tests that don't have cv2.

Feature flag: ``CLIPAI_CROSSHAIR_TRACKER`` env var, default ON.

Usage:

    from backend.services.crosshair_tracker import (
        track_crosshair_path,
        CrosshairFrame,
    )
    path = track_crosshair_path(frame_paths)  # [(t, x, y, conf), ...]
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────── Feature flag ────────────────────

USE_CROSSHAIR_TRACKER = os.environ.get(
    "CLIPAI_CROSSHAIR_TRACKER", "1",
).lower() in ("1", "true", "yes", "on")


# ──────────────────── Tunables ────────────────────

# Confidence floor below which a detection is dropped. Template
# matching peak NCC is in [-1, 1]; production patterns score
# 0.4–0.95 on a real crosshair.
DEFAULT_CONFIDENCE_FLOOR = 0.4

# Initial seed when no prior centroid is available — center-screen.
DEFAULT_SEED_XY = (50.0, 50.0)

# Search window radius around the prior centroid, as % of frame
# width. 25 % covers an aggressive aim swing without doing a
# full-frame template scan every frame.
DEFAULT_SEARCH_RADIUS_PCT = 25.0

# EMA time constant for path smoothing. 0.30 s kills frame-to-frame
# jitter without lagging real aim swings.
DEFAULT_EMA_TAU_SEC = 0.30

# Reject jumps larger than this when confidence is below floor —
# prevents the tracker from teleporting on a misfire.
JITTER_REJECT_PCT_PER_100MS = 8.0


# ──────────────────── Result dataclass ────────────────────


@dataclass
class CrosshairFrame:
    """One frame's tracked crosshair position.

    Attributes:
        timestamp: Frame timestamp (seconds).
        x_pct / y_pct: Position in [0, 100] % of source frame.
        confidence: Tracker confidence in [0, 1]. Drops below
            ``DEFAULT_CONFIDENCE_FLOOR`` are discarded by
            ``track_crosshair_path``.
    """

    timestamp: float
    x_pct: float
    y_pct: float
    confidence: float


# ──────────────────── Synthetic pattern bank ────────────────────


def _make_crosshair_patterns():
    """Build a small bank of synthetic crosshair templates.

    Returns a list of (name, ndarray) tuples covering the five
    most common crosshair shapes (dot, plus, T, X, circle) at
    three sizes (8, 12, 16 px). Each template is a uint8
    grayscale image with a black foreground on a 128-gray
    background.

    Lazily imports numpy so the module remains importable in
    sandboxes without it.
    """
    try:
        import numpy as np
    except ImportError:
        return []

    patterns = []
    sizes = (8, 12, 16, 24)

    for size in sizes:
        # Dot
        img = np.full((size, size), 128, dtype=np.uint8)
        c = size // 2
        r = max(1, size // 6)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    img[c + dy, c + dx] = 0
        patterns.append((f"dot_{size}", img))

        # Plus
        img = np.full((size, size), 128, dtype=np.uint8)
        thick = max(1, size // 8)
        img[c - thick:c + thick + 1, :] = 0
        img[:, c - thick:c + thick + 1] = 0
        patterns.append((f"plus_{size}", img))

        # X
        img = np.full((size, size), 128, dtype=np.uint8)
        for i in range(size):
            for off in range(-thick, thick + 1):
                if 0 <= i + off < size:
                    img[i, i + off] = 0
                if 0 <= (size - 1 - i) + off < size:
                    img[i, (size - 1 - i) + off] = 0
        patterns.append((f"x_{size}", img))

        # T (top-down crosshair)
        img = np.full((size, size), 128, dtype=np.uint8)
        img[c - thick:c + thick + 1, :] = 0  # horizontal bar
        img[c:c + size // 2, c - thick:c + thick + 1] = 0  # vertical down
        patterns.append((f"t_{size}", img))

        # Circle (outline)
        img = np.full((size, size), 128, dtype=np.uint8)
        r_outer = c - 1
        r_inner = max(0, r_outer - 1)
        for dy in range(-r_outer, r_outer + 1):
            for dx in range(-r_outer, r_outer + 1):
                d2 = dx * dx + dy * dy
                if r_inner * r_inner <= d2 <= r_outer * r_outer:
                    img[c + dy, c + dx] = 0
        patterns.append((f"circle_{size}", img))

    return patterns


_PATTERN_CACHE: Optional[list] = None


def _get_patterns():
    global _PATTERN_CACHE
    if _PATTERN_CACHE is None:
        _PATTERN_CACHE = _make_crosshair_patterns()
    return _PATTERN_CACHE


# ──────────────────── Single-frame detector ────────────────────


def detect_crosshair_xy(
    frame_path: str,
    *,
    prior_xy: tuple[float, float] = DEFAULT_SEED_XY,
    search_radius_pct: float = DEFAULT_SEARCH_RADIUS_PCT,
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
) -> Optional[tuple[float, float, float]]:
    """Detect crosshair position in a single frame.

    Searches a window of ``search_radius_pct`` around ``prior_xy``
    using template matching against the synthetic crosshair bank.
    Returns ``(x_pct, y_pct, confidence)`` for the highest-scoring
    template position, or ``None`` if no template scored above
    ``confidence_floor``.

    Args:
        frame_path: Filesystem path to a frame image.
        prior_xy: Best-guess centroid in [0, 100] % to seed the
            search. Defaults to frame center.
        search_radius_pct: Search window half-width as % of frame.
        confidence_floor: Minimum normalized cross-correlation
            score to accept a detection.

    Returns:
        ``(x_pct, y_pct, confidence)`` tuple in [0, 100] % space
        for the position and [0, 1] for confidence, or ``None``.
    """
    if not USE_CROSSHAIR_TRACKER:
        return None

    try:
        import cv2  # type: ignore
        import numpy as np
    except ImportError:
        return None

    img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w = img.shape[:2]
    if w <= 0 or h <= 0:
        return None

    # Search window in pixels.
    radius_px = max(int(w * search_radius_pct / 100.0), 16)
    cx_px = int(w * float(prior_xy[0]) / 100.0)
    cy_px = int(h * float(prior_xy[1]) / 100.0)
    x0 = max(0, cx_px - radius_px)
    y0 = max(0, cy_px - radius_px)
    x1 = min(w, cx_px + radius_px)
    y1 = min(h, cy_px + radius_px)
    if x1 - x0 < 24 or y1 - y0 < 24:
        return None

    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    patterns = _get_patterns()
    if not patterns:
        return None

    best_score = -1.0
    best_x = float(prior_xy[0])
    best_y = float(prior_xy[1])

    for _name, tpl in patterns:
        th, tw = tpl.shape[:2]
        if roi.shape[0] < th + 2 or roi.shape[1] < tw + 2:
            continue
        try:
            res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
        except Exception:
            continue
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = float(max_val)
            # Pixel coordinates of the template's top-left in the
            # full source frame; add half template size for centroid.
            best_x_px = x0 + max_loc[0] + tw / 2.0
            best_y_px = y0 + max_loc[1] + th / 2.0
            best_x = best_x_px / max(w, 1) * 100.0
            best_y = best_y_px / max(h, 1) * 100.0

    if best_score < confidence_floor:
        return None
    return (best_x, best_y, max(0.0, min(1.0, best_score)))


# ──────────────────── Multi-frame walker ────────────────────


# ──────────────────── Fallback semantics ────────────────────
#
# How many consecutive missed detections we tolerate before the
# EMA state is re-seeded back to ``DEFAULT_SEED_XY``. A short
# blip (single frame miss inside a continuous detection) should
# retain the prior position so the search resumes where the
# crosshair actually is; a longer outage means the prior is
# stale and the next real detection should start its search
# from center.
FALLBACK_RESEED_AFTER_DROPS = 3


def track_crosshair_path(
    frame_paths: list,
    *,
    seed_xy: tuple[float, float] = DEFAULT_SEED_XY,
    search_radius_pct: float = DEFAULT_SEARCH_RADIUS_PCT,
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
    ema_tau_sec: float = DEFAULT_EMA_TAU_SEC,
) -> list[CrosshairFrame]:
    """Walk frames in order, returning a per-frame crosshair path.

    Seeds the first frame at ``seed_xy``, then uses each previous
    *smoothed* position as the prior for the next frame. Per-frame
    detections are EMA-smoothed with ``ema_tau_sec`` to kill jitter.
    Detections that move more than ``JITTER_REJECT_PCT_PER_100MS``
    in under 100 ms AND have confidence < ``confidence_floor + 0.1``
    are treated as jitter and emitted as fallback-center entries.

    **Always-emit fallback semantics (Phase 1 center-bias):**

    Every input frame produces exactly one :class:`CrosshairFrame`
    in the returned list. Frames where the template matcher fails
    (or the confidence floor / jitter guard rejects the detection)
    emit a ``(timestamp, 50.0, 50.0, 0.0)`` entry — *never* a stale
    prior position. The downstream L1 solver uses the ``confidence``
    field to decide whether to weigh the crosshair anchor
    (``confidence ≥ GAMING_CROSSHAIR_CONF_THRESHOLD``, default 0.6)
    or fall back to the mandatory dead-center anchor. Without the
    0.0-confidence breadcrumb the solver would have no way to
    distinguish "no crosshair here" from "no frames observed" and
    could inherit stale face-cluster / motion-centroid anchors on
    cartoon-style FPS content like TF2 or Marvel Rivals.

    When a run of ``FALLBACK_RESEED_AFTER_DROPS`` consecutive
    fallback frames occurs, the internal EMA state is re-seeded
    back to ``seed_xy`` so the next real detection begins its
    search from center rather than from a long-stale prior.

    Args:
        frame_paths: List of ``(timestamp, path)`` tuples sorted
            by timestamp.
        seed_xy: Seed position for the first frame.
        search_radius_pct / confidence_floor: Forwarded to
            :func:`detect_crosshair_xy`.
        ema_tau_sec: EMA time constant.

    Returns:
        One :class:`CrosshairFrame` per input frame in timestamp
        order. Low-confidence / missing frames carry
        ``x_pct=50.0, y_pct=50.0, confidence=0.0``.
    """
    if not frame_paths:
        return []

    sorted_frames = sorted(frame_paths, key=lambda p: p[0])

    # Fallback breadcrumbs always emit the hard-coded dead-center
    # (50, 50) position per the Phase 1 center-bias spec — the
    # downstream L1 solver treats ``confidence == 0.0`` as "anchor
    # to center" and this guarantees it sees 50, 50 regardless of
    # how the caller seeded the initial search prior.
    fallback_x = 50.0
    fallback_y = 50.0
    seed_x = float(seed_xy[0])
    seed_y = float(seed_xy[1])
    smoothed_x = seed_x
    smoothed_y = seed_y
    last_real_t: Optional[float] = None
    saw_any = False
    consecutive_drops = 0
    out: list[CrosshairFrame] = []

    def _emit_fallback(ts: float) -> None:
        """Append a hard-center breadcrumb for this frame."""
        nonlocal consecutive_drops, smoothed_x, smoothed_y
        out.append(CrosshairFrame(
            timestamp=float(ts),
            x_pct=fallback_x,
            y_pct=fallback_y,
            confidence=0.0,
        ))
        consecutive_drops += 1
        if consecutive_drops >= FALLBACK_RESEED_AFTER_DROPS:
            # Re-seed the search prior back to hard center so the
            # next real detection doesn't resume its search window
            # at a long-stale (and possibly off-axis) position.
            smoothed_x = fallback_x
            smoothed_y = fallback_y

    for ts, path in sorted_frames:
        det = detect_crosshair_xy(
            path,
            prior_xy=(smoothed_x, smoothed_y),
            search_radius_pct=search_radius_pct,
            confidence_floor=confidence_floor,
        )
        if det is None:
            _emit_fallback(ts)
            continue
        raw_x, raw_y, conf = det

        # Jitter reject: if the raw detection moved a lot in a
        # very short time AND confidence is borderline, emit a
        # center-anchor fallback instead of inheriting the
        # likely-wrong position.
        if last_real_t is not None:
            dt = ts - last_real_t
            if 0 < dt < 0.1 and conf < confidence_floor + 0.1:
                dist = math.hypot(raw_x - smoothed_x, raw_y - smoothed_y)
                if dist > JITTER_REJECT_PCT_PER_100MS:
                    _emit_fallback(ts)
                    continue

        consecutive_drops = 0
        if not saw_any:
            smoothed_x = raw_x
            smoothed_y = raw_y
            saw_any = True
        else:
            dt = ts - last_real_t if last_real_t is not None else 0.033
            dt = max(dt, 1e-6)
            alpha = 1.0 - math.exp(-dt / max(ema_tau_sec, 1e-6))
            smoothed_x = smoothed_x + alpha * (raw_x - smoothed_x)
            smoothed_y = smoothed_y + alpha * (raw_y - smoothed_y)

        out.append(CrosshairFrame(
            timestamp=float(ts),
            x_pct=float(smoothed_x),
            y_pct=float(smoothed_y),
            confidence=float(conf),
        ))
        last_real_t = ts

    return out


def crosshair_persistence_score(path: list[CrosshairFrame]) -> float:
    """Backward-compat helper: how persistent is the crosshair?

    Returns the fraction of frames in ``path`` where the smoothed
    centroid stayed within 8 % of frame width of the cluster center.
    Used by the legacy classifier code that wants a 0-1 score.

    Only high-confidence entries (``confidence >= 0.4``) contribute
    to the score; the Phase 1 fallback-center breadcrumbs emitted
    for low-confidence frames are excluded so a clip with no
    detectable crosshair doesn't score as "perfectly persistent at
    (50, 50)."
    """
    real = [p for p in path if p.confidence >= 0.4]
    if len(real) < 3:
        return 0.0
    xs = [p.x_pct for p in real]
    ys = [p.y_pct for p in real]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    n_close = sum(
        1 for x, y in zip(xs, ys)
        if math.hypot(x - cx, y - cy) <= 8.0
    )
    return n_close / len(real)
