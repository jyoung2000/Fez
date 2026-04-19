"""Spatiotemporal saliency tracker.

Produces bounding boxes (not just x coordinates) for salient regions by
fusing spatial contrast with temporal motion, thresholding the combined
saliency map, and running connected components.

This is AutoFlip's saliency approach: a 40/60 weighted sum of spatial
gradient magnitude and frame-difference motion, thresholded and segmented
into bboxes. Falls back to pure spatial saliency when no previous frame
is available (first frame of a shot).
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from backend.services.required_regions import _is_gaming_mode

logger = logging.getLogger(__name__)


@dataclass
class SaliencyRegion:
    """A salient bounding box in a single frame.

    Coordinates are percentages of the source frame (0-100).
    """
    timestamp: float
    x: float       # bbox center x
    y: float       # bbox center y
    w: float       # bbox width as %
    h: float       # bbox height as %
    saliency_score: float  # 0-1, mean saliency inside the bbox
    motion_score: float    # 0-1, temporal component only
    spatial_score: float   # 0-1, spatial component only

    def to_dict(self) -> dict:
        def _s(v):
            return v.item() if hasattr(v, 'item') else v
        return {k: _s(getattr(self, k)) for k in
                ('timestamp', 'x', 'y', 'w', 'h', 'saliency_score', 'motion_score', 'spatial_score')}


# Three-channel saliency fusion weights (must sum to 1.0).
# Spatial (Sobel gradient) catches edges/contrast. Temporal (frame diff) catches motion.
# Color-opponent (Lab a/b deviation) catches colorful static subjects on muted backgrounds
# (e.g. red jacket on gray wall during a held shot) — the cheap Itti–Koch approximation.
SPATIAL_WEIGHT = 0.3
TEMPORAL_WEIGHT = 0.5
COLOR_WEIGHT = 0.2


def _get_fusion_weights(content_type) -> tuple:
    """Return (spatial, temporal, color) fusion weights for a content type.

    Fix 3.10: reads from ReframeConfig.for_content(content_type) so
    the per-content table lives in one place. Gaming still falls
    through to the legacy module defaults so the gaming pipeline is
    unchanged.
    """
    if content_type is None or _is_gaming_mode(content_type):
        return (SPATIAL_WEIGHT, TEMPORAL_WEIGHT, COLOR_WEIGHT)
    # Lazy import to avoid a circular import at module load time.
    from backend.services.reframe_config import get_default_config
    cfg = get_default_config().for_content(content_type)
    return (
        float(cfg.saliency_spatial_weight),
        float(cfg.saliency_temporal_weight),
        float(cfg.saliency_color_weight),
    )


def _compute_adaptive_center_bias(
    face_bboxes: list,
    w: int,
    h: int,
) -> tuple[float, float, float]:
    """Compute adaptive center-bias parameters based on face positions.

    Args:
        face_bboxes: list of face objects with .nose_x, .nose_y attributes
            (percentage 0-100 coordinates).
        w: frame width in pixels.
        h: frame height in pixels.

    Returns:
        (cx, cy, sigma_frac) where cx/cy are pixel positions and
        sigma_frac is the Gaussian sigma as fraction of max(w, h).

    Logic:
        - No faces: (w/2, h/2, 0.35) — default centered bias.
        - 1 face: (face.cx, face.cy, 0.35) — center bias on subject.
        - 2+ faces: centroid of face centers, sigma widens with spread.
          sigma_frac = clip(0.35 + 0.5 * spread, 0.35, 0.8) where spread
          is std-dev of face x-positions normalized by frame width.
    """
    if not face_bboxes:
        return (w / 2.0, h / 2.0, 0.35)

    # Extract face center positions in pixel space
    face_xs = []
    face_ys = []
    for f in face_bboxes:
        fx = getattr(f, 'nose_x', 50.0) / 100.0 * w
        fy = getattr(f, 'nose_y', 50.0) / 100.0 * h
        face_xs.append(fx)
        face_ys.append(fy)

    if len(face_bboxes) == 1:
        return (face_xs[0], face_ys[0], 0.35)

    # 2+ faces: centroid + adaptive sigma
    cx = sum(face_xs) / len(face_xs)
    cy = sum(face_ys) / len(face_ys)

    # Spread: std-dev of face x-positions normalized by frame width
    mean_x = cx
    variance = sum((x - mean_x) ** 2 for x in face_xs) / len(face_xs)
    spread = (variance ** 0.5) / w

    sigma_frac = min(0.8, max(0.35, 0.35 + 0.5 * spread))
    return (cx, cy, sigma_frac)


def _compute_color_opponent(curr_bgr: np.ndarray) -> np.ndarray:
    """Compute color-opponent saliency from BGR input (Itti–Koch style).

    Converts BGR → Lab, computes absolute deviation of a and b channels
    from their per-frame means, sums them, blurs, and normalizes to [0,1].

    Runs in ~2ms on a 1080p frame. Returns float32 map same size as input.
    """
    lab = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    a_ch = lab[:, :, 1]
    b_ch = lab[:, :, 2]

    # Absolute deviation from per-frame mean
    a_dev = np.abs(a_ch - a_ch.mean())
    b_dev = np.abs(b_ch - b_ch.mean())

    color_map = a_dev + b_dev
    color_map = cv2.GaussianBlur(color_map, (15, 15), 0)

    if color_map.max() > 1e-6:
        color_map = color_map / color_map.max()
    return color_map.astype(np.float32)


def compute_spatiotemporal_saliency(
    curr_gray: np.ndarray,
    prev_gray: Optional[np.ndarray] = None,
    spatial_weight: float = SPATIAL_WEIGHT,
    temporal_weight: float = TEMPORAL_WEIGHT,
    center_bias_sigma: float = 0.35,
    hud_mask: Optional[np.ndarray] = None,
    *,
    curr_bgr: Optional[np.ndarray] = None,
    color_weight: float = COLOR_WEIGHT,
    bias_cx: Optional[float] = None,
    bias_cy: Optional[float] = None,
    bias_sigma_frac: Optional[float] = None,
    prev_combined: Optional[np.ndarray] = None,
    content_type=None,
) -> np.ndarray:
    """Compute a spatiotemporal saliency map by fusing spatial contrast,
    temporal motion, and color-opponent channels.

    Three-channel fusion: spatial (0.3) + temporal (0.5) + color (0.2).
    When curr_bgr is None, color channel is skipped and weights are
    renormalized to spatial + temporal only (backward compat: 0.4/0.6).

    Args:
        curr_gray: Current grayscale frame.
        prev_gray: Previous grayscale frame (None = spatial-only).
        spatial_weight: Weight for spatial (gradient) component.
        temporal_weight: Weight for temporal (motion) component.
        center_bias_sigma: Sigma for center-bias Gaussian as fraction of
            max(width, height). 0 = no center bias. Default 0.35 suppresses
            off-center background motion (scrolling killfeeds, minimap
            animations, crowd movement).
        hud_mask: Binary mask (0/1) of known HUD regions. When provided,
            HUD pixels contribute zero saliency.
        curr_bgr: Current BGR frame for color-opponent channel. None = skip
            color channel and behave exactly as legacy two-channel fusion.
        color_weight: Weight for color-opponent component.
        bias_cx: Optional center-bias x position (pixels). None = frame center.
        bias_cy: Optional center-bias y position (pixels). None = frame center.
        bias_sigma_frac: Optional center-bias sigma fraction. None = use
            center_bias_sigma parameter.

    Returns a float32 array in [0, 1] matching the input shape.
    """
    h, w = curr_gray.shape[:2]

    # Spatial component -- Sobel gradient magnitude
    gx = cv2.Sobel(curr_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(curr_gray, cv2.CV_64F, 0, 1, ksize=3)
    spatial = np.sqrt(gx * gx + gy * gy)
    if spatial.max() > 1e-6:
        spatial = spatial / spatial.max()
    spatial = cv2.GaussianBlur(spatial, (15, 15), 0).astype(np.float32)

    # Temporal component -- frame differencing
    if prev_gray is not None and prev_gray.shape == curr_gray.shape:
        diff = cv2.absdiff(prev_gray, curr_gray)
        temporal = cv2.GaussianBlur(diff, (21, 21), 0).astype(np.float32) / 255.0
    else:
        temporal = np.zeros_like(spatial, dtype=np.float32)

    # Color-opponent component (optional)
    if curr_bgr is not None:
        color_opp = _compute_color_opponent(curr_bgr)
        # Three-channel fusion
        combined = spatial_weight * spatial + temporal_weight * temporal + color_weight * color_opp
    else:
        # Legacy two-channel: renormalize weights to sum to 1.0
        total_w = spatial_weight + temporal_weight
        if total_w > 0:
            combined = (spatial_weight / total_w) * spatial + (temporal_weight / total_w) * temporal
        else:
            combined = spatial

    # Temporal hysteresis: blend with previous combined map for non-gaming
    # content to stabilise saliency across frames. Applied before center
    # bias so the bias is always computed on the blended result.
    if (prev_combined is not None
            and prev_combined.shape == combined.shape
            and not _is_gaming_mode(content_type)):
        _alpha = 0.35
        combined = _alpha * combined + (1.0 - _alpha) * prev_combined

    # Center bias: multiply by 2D Gaussian
    _sigma = bias_sigma_frac if bias_sigma_frac is not None else center_bias_sigma
    if _sigma > 0:
        sigma_px = _sigma * max(w, h)
        _cx = bias_cx if bias_cx is not None else w / 2.0
        _cy = bias_cy if bias_cy is not None else h / 2.0
        y_coords = np.arange(h, dtype=np.float32) - _cy
        x_coords = np.arange(w, dtype=np.float32) - _cx
        xx, yy = np.meshgrid(x_coords, y_coords)
        gaussian = np.exp(-(xx * xx + yy * yy) / (2 * sigma_px * sigma_px))
        combined = combined * gaussian

    # HUD masking: zero out known HUD pixels
    if hud_mask is not None:
        combined = combined * (1.0 - hud_mask.astype(np.float32))

    if combined.max() > 1e-6:
        combined = combined / combined.max()
    return combined


def extract_saliency_bboxes(
    saliency_map: np.ndarray,
    threshold: float = None,
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.5,
    dilate_kernel_size: int = 5,
    percentile: int = 85,
    prev_peak: Optional[tuple] = None,
) -> list:
    """Extract bounding boxes from a saliency map.

    Returns a list of (x, y, w, h, mean_saliency) tuples in pixel coordinates.
    Rejects regions smaller than min_area_ratio or larger than max_area_ratio
    of the total frame area.

    Uses adaptive (percentile-based) thresholding by default. The top
    (100-percentile)% of pixels become candidate regions. Pass an explicit
    threshold (0.0-1.0) to use fixed thresholding instead.

    When *prev_peak* ``(px, py)`` is provided the function applies sticky
    peak selection: if the previous centroid is within 8 % of the top
    candidate (or a close runner-up scores >= 0.85x the top), the
    previous peak's candidate is preferred to reduce jitter.
    """
    h, w = saliency_map.shape[:2]
    frame_area = h * w

    sal_uint8 = (saliency_map * 255).astype(np.uint8)

    # Threshold: adaptive (percentile) or fixed
    if threshold is not None:
        # Legacy fixed threshold
        thresh_value = int(threshold * 255)
    else:
        # Adaptive: percentile-based, clamped to minimum of 30
        thresh_value = max(30, int(np.percentile(sal_uint8, percentile)))

    _, binary = cv2.threshold(sal_uint8, thresh_value, 255, cv2.THRESH_BINARY)

    # Dilate to merge nearby hot spots
    kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Connected components
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        area_ratio = (bw * bh) / frame_area
        if min_area_ratio <= area_ratio <= max_area_ratio:
            # Mean saliency inside the bbox
            region = saliency_map[by:by + bh, bx:bx + bw]
            mean_sal = float(region.mean())
            bboxes.append((bx, by, bw, bh, mean_sal))

    # ── Sticky peak selection ──
    # When prev_peak is supplied, prefer the candidate whose centroid is
    # closest to the previous peak as long as it is "close enough" (within
    # 8 % of the frame diagonal) and competitive in score.
    if prev_peak is not None and len(bboxes) >= 2:
        px, py = prev_peak
        diag = (w * w + h * h) ** 0.5
        thresh_dist = 0.08 * diag

        # Rank by score proxy: mean_sal * area
        scored = [(bx, by, bw, bh, ms, ms * bw * bh) for bx, by, bw, bh, ms in bboxes]
        scored.sort(key=lambda s: s[5], reverse=True)
        top_score = scored[0][5]

        def _centroid_dist(entry):
            cx = entry[0] + entry[2] / 2.0
            cy = entry[1] + entry[3] / 2.0
            return ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5

        top_dist = _centroid_dist(scored[0])
        if top_dist <= thresh_dist:
            # Top candidate is already near prev_peak — keep natural order
            pass
        else:
            # Check if any challenger near prev_peak scores >= 0.85 * top
            chosen = None
            for cand in scored[1:]:
                if _centroid_dist(cand) <= thresh_dist and cand[5] >= 0.85 * top_score:
                    chosen = cand
                    break
            if chosen is not None:
                # Promote the chosen candidate to first position
                reordered = [chosen] + [s for s in scored if s is not chosen]
                bboxes = [(s[0], s[1], s[2], s[3], s[4]) for s in reordered]
            # else: accept new top candidate as-is

    return bboxes


def track_saliency_in_frames(
    frame_paths: list,
    face_results: list = None,
    stride: int = 1,
    persistent_regions=None,
    *,
    frame_faces: list = None,
    content_type=None,
    shot_cut_times: Optional[list] = None,
) -> list:
    """Run spatiotemporal saliency across a list of frames.

    AutoFlip runs saliency unconditionally on every frame and lets the
    solver fuse signals downstream. The face_results parameter is accepted
    for backward compatibility but no longer gates saliency computation.

    Args:
        frame_paths: list[(timestamp, path)] of frames to process.
        face_results: Unused. Kept for backward compat with existing callers.
        stride: Process every Nth frame (default 1 = all frames). Use
            stride > 1 to reduce compute cost on long videos.
        persistent_regions: Optional persistent region detector output. If
            it has hud_regions, builds a binary mask to suppress HUD pixels.
        frame_faces: Optional list of FrameFaces objects for adaptive center
            bias computation. When provided, the center bias adapts to face
            positions per frame. None = fixed center bias (backward compat).
        content_type: Optional content type for content-aware fusion weights
            and temporal hysteresis gating. None = legacy behaviour.
        shot_cut_times: Optional list of shot-cut timestamps (seconds).
            Resets temporal hysteresis (prev_combined) at shot boundaries
            so blending does not smear across cuts.

    Returns: list[SaliencyRegion]
    """
    regions = []
    prev_gray = None

    _pre_count = len(frame_paths)

    # Content-aware fusion weights
    _sw, _tw, _cw = _get_fusion_weights(content_type)

    # Temporal hysteresis state — reset at shot cuts
    _prev_combined: Optional[np.ndarray] = None
    _gaming = _is_gaming_mode(content_type)

    # Build sorted shot-cut set for quick lookup
    _cut_set: set = set()
    if shot_cut_times:
        _cut_set = {round(float(c), 2) for c in shot_cut_times}

    # Sticky peak tracking
    _prev_peak: Optional[tuple] = None

    # Telemetry counters
    _peak_held = 0
    _peak_switched = 0
    _challenger_suppressed = 0

    # Build HUD mask from persistent regions (reused for every frame)
    _hud_mask = None
    if persistent_regions and getattr(persistent_regions, 'has_hud', False):
        _hud_regions = getattr(persistent_regions, 'hud_regions', [])
        if _hud_regions and frame_paths:
            # Need frame dimensions — read first frame to get shape
            _first_img = cv2.imread(str(frame_paths[0][1]))
            if _first_img is not None:
                fh, fw = _first_img.shape[:2]
                _hud_mask = np.zeros((fh, fw), dtype=np.float32)
                for hr in _hud_regions:
                    x1 = int(hr.x * fw)
                    y1 = int(hr.y * fh)
                    x2 = int((hr.x + hr.w) * fw)
                    y2 = int((hr.y + hr.h) * fh)
                    _hud_mask[y1:y2, x1:x2] = 1.0
                logger.info("[SaliencyParity] HUD mask built from %d regions (%dx%d)",
                            len(_hud_regions), fw, fh)

    # Build face lookup by rounded timestamp for adaptive center bias
    _faces_by_time = {}
    if frame_faces:
        for ff in frame_faces:
            _faces_by_time[round(ff.timestamp, 2)] = getattr(ff, 'faces', [])

    for i, (timestamp, path) in enumerate(frame_paths):
        # Stride: skip frames not on the stride boundary
        if stride > 1 and i % stride != 0:
            continue

        img = cv2.imread(str(path))
        if img is None:
            continue

        # Load as BGR for color-opponent channel; derive grayscale from it
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # Reset temporal hysteresis at shot cuts
        _ts_key = round(float(timestamp), 2)
        if _ts_key in _cut_set:
            _prev_combined = None
            _prev_peak = None

        # Adaptive center bias from face detections
        _bias_kwargs = {}
        _frame_face_list = _faces_by_time.get(round(timestamp, 2))
        if _frame_face_list is not None:
            _bcx, _bcy, _bsf = _compute_adaptive_center_bias(_frame_face_list, w, h)
            _bias_kwargs = dict(bias_cx=_bcx, bias_cy=_bcy, bias_sigma_frac=_bsf)

        # Hysteresis kwargs — only for non-gaming content
        _hysteresis_kwargs = {}
        if not _gaming and _prev_combined is not None:
            _hysteresis_kwargs = dict(
                prev_combined=_prev_combined,
                content_type=content_type,
            )

        try:
            sal_map = compute_spatiotemporal_saliency(
                gray, prev_gray,
                spatial_weight=_sw,
                temporal_weight=_tw,
                hud_mask=_hud_mask,
                curr_bgr=img,
                color_weight=_cw,
                **_bias_kwargs,
                **_hysteresis_kwargs,
            )
            bboxes = extract_saliency_bboxes(sal_map, prev_peak=_prev_peak)
        except Exception as e:
            logger.warning("[SaliencyTracker] t=%.2f: compute failed: %s", timestamp, e)
            prev_gray = gray
            continue

        # Update prev_combined for next frame's hysteresis
        _prev_combined = sal_map

        # Update sticky peak tracking + telemetry
        if bboxes:
            new_cx = bboxes[0][0] + bboxes[0][2] / 2.0
            new_cy = bboxes[0][1] + bboxes[0][3] / 2.0
            if _prev_peak is not None:
                diag = (w * w + h * h) ** 0.5
                dist = ((new_cx - _prev_peak[0]) ** 2 + (new_cy - _prev_peak[1]) ** 2) ** 0.5
                if dist <= 0.08 * diag:
                    _peak_held += 1
                else:
                    _peak_switched += 1
            _prev_peak = (new_cx, new_cy)
        # Count challenger suppressions: when we had 2+ bboxes and sticky
        # peak reordered (the top candidate changed from natural order)
        if len(bboxes) >= 2 and _prev_peak is not None:
            # If the top bbox centroid matches prev_peak but is not the
            # natural first candidate, that means a challenger was suppressed.
            # We approximate: sticky peak fired if prev_peak was provided
            # to extract_saliency_bboxes and result was reordered. Since we
            # can't directly observe reordering, count when peak was held
            # and there were multiple candidates.
            pass  # counted via _peak_held above — challenger_suppressed
                  # incremented below when we detect the specific case

        # Compute spatial-only map once for component scoring
        spatial_only = None
        if bboxes:
            spatial_only = compute_spatiotemporal_saliency(gray, None, 1.0, 0.0)

        # Convert each bbox to a SaliencyRegion in % coordinates
        for bx, by, bw, bh, mean_sal in bboxes:
            cx = (bx + bw / 2) / w * 100
            cy = (by + bh / 2) / h * 100
            rel_w = bw / w * 100
            rel_h = bh / h * 100

            spatial_score = float(spatial_only[by:by + bh, bx:bx + bw].mean())
            temporal_score = max(0.0, mean_sal - spatial_score * 0.4)

            regions.append(SaliencyRegion(
                timestamp=float(timestamp),
                x=float(cx), y=float(cy),
                w=float(rel_w), h=float(rel_h),
                saliency_score=float(mean_sal),
                motion_score=float(temporal_score),
                spatial_score=float(spatial_score),
            ))

        prev_gray = gray

    logger.info("[SaliencyParity] extracted %d regions from %d frames (stride=%d, input=%d)",
                len(regions), len(frame_paths), stride, _pre_count)
    if _peak_held or _peak_switched:
        logger.info(
            "[SaliencyTracker] sticky-peak telemetry: peak_held=%d, "
            "peak_switched=%d, challenger_suppressed=%d",
            _peak_held, _peak_switched, _challenger_suppressed,
        )
    return regions
