"""Phase 1 — per-scene auto-detection of gameplay HUD regions.

The existing ``game_layouts.GAME_HUD_LAYOUTS`` table ships a
hand-curated bbox for the ~20 supported titles. This module adds
a complementary, game-agnostic detector: sample N frames across
a scene, find pixel regions that are **temporally stable** (low
variance) and **spatially edgy** (high Sobel magnitude), cluster
the survivors into bounding boxes, and label each by its frame
quadrant with a cheap semantic hint.

The output complements the static layout table rather than
replacing it — callers that know the specific game should still
prefer the exact bbox from ``game_layouts``; the auto-detector
is the fallback when the game is unknown (e.g. clip dropped
from a third-party storefront, or an indie title ClipAI hasn't
seen before).

Feature flag: ``USE_GAMING_REFRAME`` (via
:mod:`gaming_reframe_config`). No runtime cost when the flag is
off — the module is importable, but the top-level detector
short-circuits without cv2 or when the config disables gaming.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal, Optional

logger = logging.getLogger(__name__)


# ──────────────────── Result dataclass ────────────────────


Quadrant = Literal["TL", "TR", "BL", "BR", "BC", "TC"]
SemanticHint = Literal[
    "unknown", "minimap", "health", "scoreboard", "killfeed",
    "abilities",
]


@dataclass
class HudRegion:
    """One auto-detected gameplay HUD region.

    ``bbox`` is stored in the **original** frame pixel coordinate
    system (not the downsampled working resolution), so callers
    can hand it directly to the frontend / FFmpeg without a
    rescale.
    """

    bbox: tuple[int, int, int, int]  # x, y, w, h — original frame pixels
    quadrant: Quadrant
    confidence: float            # mean HUD score inside bbox, 0-1
    semantic_hint: SemanticHint = "unknown"
    # Shape metadata kept around for tests / debug overlays.
    aspect_ratio: float = 1.0
    frame_width: int = 0
    frame_height: int = 0

    def as_pct(self) -> tuple[float, float, float, float]:
        """Return ``(x_pct, y_pct, w_pct, h_pct)`` in [0, 100]."""
        if self.frame_width <= 0 or self.frame_height <= 0:
            return (0.0, 0.0, 0.0, 0.0)
        x, y, w, h = self.bbox
        return (
            x / self.frame_width * 100.0,
            y / self.frame_height * 100.0,
            w / self.frame_width * 100.0,
            h / self.frame_height * 100.0,
        )


# ──────────────────── Quadrant classification ────────────────────


def classify_quadrant(
    bbox: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
) -> Quadrant:
    """Return which frame quadrant the bbox center lives in.

    Uses a 3×2 grid (top/middle/bottom × left/center/right) so
    the ``BC`` / ``TC`` center-strip buckets can catch
    health-bar / scoreboard layouts that aren't strictly in a
    corner.
    """
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    horiz = "L" if cx < frame_w / 3 else ("R" if cx > 2 * frame_w / 3 else "C")
    vert = "T" if cy < frame_h / 3 else ("B" if cy > 2 * frame_h / 3 else "M")
    if horiz == "C" and vert == "B":
        return "BC"
    if horiz == "C" and vert == "T":
        return "TC"
    if vert == "M":
        # Middle rows are rarely real HUDs — snap to the nearest
        # top or bottom edge to keep the label set small.
        vert = "T" if cy < frame_h / 2 else "B"
    if horiz == "C":
        horiz = "L" if cx < frame_w / 2 else "R"
    return f"{vert}{horiz}"  # type: ignore[return-value]


def classify_semantic_hint(
    bbox: tuple[int, int, int, int],
    quadrant: Quadrant,
    frame_w: int,
    frame_h: int,
) -> SemanticHint:
    """Cheap heuristic — good enough for visual debugging and
    for HUD-glance triggers to pick a sensible default."""
    _x, _y, w, h = bbox
    aspect = w / max(h, 1)

    if quadrant == "BR" and 0.6 <= aspect <= 1.6:
        return "minimap"
    if quadrant in ("BL", "BC") and aspect >= 2.0:
        return "health"
    if quadrant == "BC" and 0.6 <= aspect <= 2.0 and (h / frame_h) < 0.10:
        return "abilities"
    if quadrant == "TR" and aspect >= 1.8:
        return "killfeed"
    if quadrant == "TC" and aspect >= 2.5:
        return "scoreboard"
    return "unknown"


# ──────────────────── Core detector ────────────────────


def detect_hud_regions(
    frame_paths: list[str],
    *,
    score_threshold: float = 0.55,
    downsample_width: int = 320,
    min_area_pct: float = 0.003,
    max_area_pct: float = 0.08,
    edge_margin_pct: float = 0.15,
    max_regions: int = 6,
) -> list[HudRegion]:
    """Detect auto-HUD regions across a set of sample frames.

    Algorithm:

      1. Load each frame as grayscale and downsample to
         ``downsample_width`` preserving aspect ratio.
      2. Stack into a (N, H, W) uint8 array.
      3. **Temporal variance**: per-pixel std-dev across N. Low
         values = stable UI. Normalized to [0, 1].
      4. **Edge magnitude**: mean Sobel magnitude across N (we
         average across frames so brief motion flashes don't
         dominate). Normalized to [0, 1].
      5. **HUD score**: ``edge_density * (1 - normalized_variance)``.
      6. Threshold → binary mask.
      7. ``cv2.connectedComponentsWithStats`` → candidate bboxes.
      8. Filter by area and edge-proximity constraints, rescale
         to the **original** frame coordinate system, classify
         quadrant + semantic hint, sort by confidence, keep the
         top ``max_regions``.

    Args:
        frame_paths: List of sample frame image paths (jpeg/png).
            The caller is responsible for picking temporally
            spaced samples across the scene.
        score_threshold: HUD score floor for the binary mask.
        downsample_width: Working resolution (original frames
            are rescaled to this width with preserved aspect).
        min_area_pct / max_area_pct: Bbox area filters as a
            fraction of the frame.
        edge_margin_pct: Distance from the nearest frame edge
            below which a bbox is considered "near an edge"
            (HUDs live at edges).
        max_regions: Hard cap on the returned region count.

    Returns:
        A list of :class:`HudRegion` objects in descending
        confidence order, each coordinate-expressed in the
        original frame pixel system.
    """
    if not frame_paths:
        return []

    try:
        import cv2  # type: ignore
        import numpy as np
    except ImportError:
        logger.debug("gaming_hud_detector: cv2/numpy unavailable")
        return []

    # ── Step 1-2: load + downsample + stack ──
    stack = []
    original_w = 0
    original_h = 0
    working_w = downsample_width
    working_h = 0

    for fp in frame_paths:
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h0, w0 = img.shape[:2]
        if w0 <= 0 or h0 <= 0:
            continue
        if original_w == 0:
            original_w = w0
            original_h = h0
            working_h = max(8, int(round(h0 * working_w / w0)))
        # Resize to the working resolution
        if (w0, h0) != (working_w, working_h):
            img = cv2.resize(
                img, (working_w, working_h),
                interpolation=cv2.INTER_AREA,
            )
        stack.append(img)

    if len(stack) < 2:
        return []

    arr = np.stack(stack, axis=0).astype(np.float32)
    # arr: (N, H, W)

    # ── Step 3: temporal variance ──
    std_map = arr.std(axis=0)
    std_norm = std_map / (std_map.max() + 1e-6)
    stability = 1.0 - std_norm  # high = stable

    # ── Step 4: edge magnitude (averaged across frames) ──
    # Sobel per frame, averaged, normalized.
    sobel_accum = np.zeros_like(arr[0])
    for i in range(arr.shape[0]):
        f = arr[i]
        gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
        sobel_accum += np.hypot(gx, gy)
    sobel_mean = sobel_accum / arr.shape[0]
    edge_norm = sobel_mean / (sobel_mean.max() + 1e-6)

    # ── Step 5: HUD score ──
    hud_score = edge_norm * stability
    # Smooth a little to reduce speckle clusters before CC
    hud_score_smooth = cv2.GaussianBlur(hud_score, (5, 5), 1.0)

    # ── Step 6: threshold ──
    mask = (hud_score_smooth >= score_threshold).astype(np.uint8) * 255

    # ── Step 7: connected components ──
    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8,
    )

    total_area = working_w * working_h
    edge_margin_px = int(round(edge_margin_pct * max(working_w, working_h)))

    candidates: list[HudRegion] = []
    for label_idx in range(1, n_labels):  # skip background (0)
        x, y, w, h, area = stats[label_idx]
        if area <= 0:
            continue
        area_pct = area / total_area
        if area_pct < min_area_pct or area_pct > max_area_pct:
            continue
        # Edge-proximity filter: at least one side must be within
        # edge_margin_px of the frame edge.
        near_edge = (
            x <= edge_margin_px
            or y <= edge_margin_px
            or (x + w) >= (working_w - edge_margin_px)
            or (y + h) >= (working_h - edge_margin_px)
        )
        if not near_edge:
            continue

        # Per-bbox confidence = mean HUD score inside.
        region_score = float(
            hud_score_smooth[y:y + h, x:x + w].mean()
        )
        if not math.isfinite(region_score):
            continue

        # Rescale back to original frame pixels.
        scale_x = original_w / working_w
        scale_y = original_h / working_h
        orig_bbox = (
            int(round(x * scale_x)),
            int(round(y * scale_y)),
            int(round(w * scale_x)),
            int(round(h * scale_y)),
        )
        quadrant = classify_quadrant(orig_bbox, original_w, original_h)
        hint = classify_semantic_hint(
            orig_bbox, quadrant, original_w, original_h,
        )
        candidates.append(HudRegion(
            bbox=orig_bbox,
            quadrant=quadrant,
            confidence=float(region_score),
            semantic_hint=hint,
            aspect_ratio=orig_bbox[2] / max(orig_bbox[3], 1),
            frame_width=original_w,
            frame_height=original_h,
        ))

    # Deduplicate quadrants — keep only the top-confidence region
    # per quadrant (the cheap clustering asked for in the spec).
    best_per_quadrant: dict[str, HudRegion] = {}
    for r in candidates:
        prev = best_per_quadrant.get(r.quadrant)
        if prev is None or r.confidence > prev.confidence:
            best_per_quadrant[r.quadrant] = r

    sorted_regions = sorted(
        best_per_quadrant.values(),
        key=lambda r: r.confidence,
        reverse=True,
    )
    return sorted_regions[:max_regions]


# ──────────────────── Scene-level cache ────────────────────


@dataclass
class _HudCacheEntry:
    scene_id: str
    regions: list[HudRegion] = field(default_factory=list)


_HUD_CACHE: dict[str, _HudCacheEntry] = {}


def detect_hud_regions_for_scene(
    scene_id: str,
    frame_paths: list[str],
    *,
    config=None,
) -> list[HudRegion]:
    """Cached wrapper — keyed on ``scene_id``.

    Re-runs detection on cache miss and stores the result under
    the scene id so repeat calls during a single export are
    free. Callers should invalidate the cache on shot boundary;
    :func:`invalidate_hud_cache` is provided for that.
    """
    entry = _HUD_CACHE.get(scene_id)
    if entry is not None:
        return entry.regions

    cfg_kwargs: dict = {}
    if config is not None:
        cfg_kwargs = {
            "score_threshold": config.hud_detection_score_threshold,
            "downsample_width": config.hud_downsample_width,
            "min_area_pct": config.hud_min_area_pct,
            "max_area_pct": config.hud_max_area_pct,
            "edge_margin_pct": config.hud_edge_margin_pct,
        }

    regions = detect_hud_regions(frame_paths, **cfg_kwargs)
    _HUD_CACHE[scene_id] = _HudCacheEntry(scene_id=scene_id, regions=regions)
    return regions


def invalidate_hud_cache(scene_id: Optional[str] = None) -> None:
    """Drop a cached entry (or the entire cache when None)."""
    if scene_id is None:
        _HUD_CACHE.clear()
    else:
        _HUD_CACHE.pop(scene_id, None)
