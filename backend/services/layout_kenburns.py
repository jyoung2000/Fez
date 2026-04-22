"""Ken Burns layout — Blueprint v2 Phase 0.

When a scene has no tracked subject (no face, diffuse saliency) and
is long enough to push meaningfully, emit a ``KEN_BURNS`` RenderOp
instead of a static crop. The L1 solver otherwise chases noise on
these scenes; a committed slow push toward the scene's mean
saliency centroid reads as cinematic instead of twitchy.

Detection is deterministic and conservative: if the scene has a
tracked face at ANY point, we stay on the current path. Ken Burns
only fires on landscape / B-roll / establishing shots.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from backend.services.reframe_config import ReframeConfig, get_default_config
from backend.services.render_plan import MotionKeypoint, Rect, RenderOp, RenderOpKind

logger = logging.getLogger(__name__)


_KEN_BURNS_CONTENT_TYPES = frozenset({
    "landscape", "documentary", "broll", "establishing", "music_video",
})


@dataclass
class SaliencyCentroid:
    """A single representative (x, y) saliency peak for a scene."""
    cx: float      # 0-1 fraction of source width
    cy: float      # 0-1 fraction of source height
    spread: float  # 0-1 fraction of source area covered by top-20% saliency


def _content_type_allows_ken_burns(content_type) -> bool:
    key = getattr(content_type, "value", content_type)
    if not key:
        return False
    return str(key).lower() in _KEN_BURNS_CONTENT_TYPES


def should_use_ken_burns(
    *,
    scene_start: float,
    scene_end: float,
    has_face_in_scene: bool,
    saliency_spread: float,
    content_type=None,
    config: Optional[ReframeConfig] = None,
) -> bool:
    """Decide whether a scene qualifies for Ken Burns.

    Criteria (all must hold):
      * No face tracked anywhere in the scene
      * Saliency is diffuse (top-20% covers > 30% of frame area),
        which is the signal for "no dominant subject"
      * Scene duration >= ``ken_burns_min_duration_sec``
      * Content type is in the landscape / B-roll allowlist
    """
    config = config or get_default_config()
    if has_face_in_scene:
        return False
    if saliency_spread <= 0.30:
        return False
    if (scene_end - scene_start) < config.ken_burns_min_duration_sec:
        return False
    if not _content_type_allows_ken_burns(content_type):
        return False
    if config.ken_burns_max_zoom <= 1.0:
        # Explicit disable via config.
        return False
    return True


def build_ken_burns_op(
    *,
    scene_start: float,
    scene_end: float,
    centroid: SaliencyCentroid,
    source_width: int,
    source_height: int,
    target_aspect: float = 9 / 16,
    config: Optional[ReframeConfig] = None,
) -> RenderOp:
    """Build a KEN_BURNS RenderOp with a slow push toward the centroid.

    Start: centered 1.0x 9:16 crop.
    End:   ``ken_burns_max_zoom`` crop, offset toward the centroid,
           capped so total travel <= ``ken_burns_max_travel_frac`` of
           source width.
    """
    config = config or get_default_config()
    max_zoom = max(config.ken_burns_max_zoom, 1.0)
    travel_cap = max(config.ken_burns_max_travel_frac, 0.0)

    # Base 9:16 crop that fits inside the source. Normalized to 0-1.
    # We treat target_aspect as W/H.
    src_ar = source_width / max(source_height, 1)
    if src_ar >= target_aspect:
        # Source is wider than target — crop height = full, width = H*target_aspect.
        base_h_norm = 1.0
        base_w_norm = (source_height * target_aspect) / source_width
    else:
        base_w_norm = 1.0
        base_h_norm = (source_width / target_aspect) / source_height

    base_w_norm = max(0.05, min(1.0, base_w_norm))
    base_h_norm = max(0.05, min(1.0, base_h_norm))

    # Start: centered crop, 1.0x.
    start_x = (1.0 - base_w_norm) / 2.0
    start_y = (1.0 - base_h_norm) / 2.0
    start_rect = Rect(x=start_x, y=start_y, w=base_w_norm, h=base_h_norm)

    # End: zoom in toward centroid. The ZOOMED crop is smaller.
    end_w = base_w_norm / max_zoom
    end_h = base_h_norm / max_zoom

    # Target center (in normalized source coords) is the centroid
    # clamped so the end crop still fits inside the source.
    tgt_cx = max(end_w / 2.0, min(1.0 - end_w / 2.0, centroid.cx))
    tgt_cy = max(end_h / 2.0, min(1.0 - end_h / 2.0, centroid.cy))
    desired_end_x = tgt_cx - end_w / 2.0
    desired_end_y = tgt_cy - end_h / 2.0

    # Cap total travel. Travel is measured as the L1 distance between
    # the start and end crop centers (normalized to source dims).
    start_cx = start_x + base_w_norm / 2.0
    start_cy = start_y + base_h_norm / 2.0
    end_cx = desired_end_x + end_w / 2.0
    end_cy = desired_end_y + end_h / 2.0
    dx = end_cx - start_cx
    dy = end_cy - start_cy
    travel = abs(dx) + abs(dy)
    if travel > travel_cap and travel > 1e-6:
        scale = travel_cap / travel
        end_cx = start_cx + dx * scale
        end_cy = start_cy + dy * scale
        desired_end_x = end_cx - end_w / 2.0
        desired_end_y = end_cy - end_h / 2.0

    end_rect = Rect(
        x=max(0.0, min(1.0 - end_w, desired_end_x)),
        y=max(0.0, min(1.0 - end_h, desired_end_y)),
        w=end_w,
        h=end_h,
    )

    op = RenderOp(
        kind=RenderOpKind.KEN_BURNS,
        start_sec=float(scene_start),
        end_sec=float(scene_end),
        primary_rect=start_rect,
        motion_path=[
            MotionKeypoint(t=0.0, rect=start_rect),
            MotionKeypoint(t=float(scene_end - scene_start), rect=end_rect),
        ],
        strategy_label="ken_burns",
    )
    logger.info(
        "[ken_burns] %.2f-%.2fs centroid=(%.2f,%.2f) start=(%.3f,%.3f,%.3f,%.3f) "
        "end=(%.3f,%.3f,%.3f,%.3f)",
        scene_start, scene_end, centroid.cx, centroid.cy,
        start_rect.x, start_rect.y, start_rect.w, start_rect.h,
        end_rect.x, end_rect.y, end_rect.w, end_rect.h,
    )
    return op
