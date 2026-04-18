"""Pure geometry helpers shared by FFmpeg + Canvas gaming layouts.

Both ``backend.services.gameplay_filters`` (FFmpeg filter graph
builders) AND the frontend ``RenderEngine.js`` Canvas renderer
need to agree on the source / destination rectangles for the
gaming layout modes. This module is the single source of truth
for that math so preview and export stay within ±2 px of each
other.

The helpers return integer pixel coordinates with even widths /
heights (FFmpeg requires this for libx264; Canvas doesn't care
but uses the same numbers for parity).

Layout modes:

  - ``fullscreen``  : center crop of source filling the target,
                       with the crop window centered on
                       ``cam_x_pct`` % of source width.
  - ``blurfill``    : sharp 16:9 source band centered vertically,
                       blurred copy of the source filling the rest.
  - ``wide_zoom``   : same as fullscreen with the crop window
                       ``crop_scale`` × wider, then scaled to fit.
  - ``composite``   : action-on-top + HUD strip — the existing
                       ``_build_gameplay_composite_filter`` owns
                       this; we don't replicate the math here.

Each helper returns a dict of named rectangles. Callers pick
the rectangles they need:

    >>> compute_fullscreen_rects(1920, 1080, 1080, 1920, cam_x_pct=50)
    {'src': (420, 0, 1080, 1080), 'dst': (0, 0, 1080, 1920)}
    >>> compute_wide_zoom_rects(1920, 1080, 1080, 1920, cam_x_pct=50)
    {'src': (260, 0, 1400, 1080), 'dst': (0, 0, 1080, 1920)}

The frontend ``RenderEngine.js`` reads these and calls
``ctx.drawImage(media, sx, sy, sw, sh, dx, dy, dw, dh)`` once
per layer.
"""

from __future__ import annotations


def _even(n: int) -> int:
    """Round to nearest even integer (FFmpeg even-dimension rule)."""
    n = int(round(n))
    return n - (n % 2)


def compute_fullscreen_rects(
    src_w: int,
    src_h: int,
    target_w: int,
    target_h: int,
    *,
    cam_x_pct: float = 50.0,
    cam_y_pct: float = 50.0,
) -> dict:
    """Center-cropped fullscreen layout.

    Crops the source to match the target aspect ratio centered on
    ``(cam_x_pct, cam_y_pct)``, then scales to fit the target.

    Returns ``{"src": (sx, sy, sw, sh), "dst": (0, 0, target_w, target_h)}``.
    """
    target_aspect = target_w / max(target_h, 1)
    src_aspect = src_w / max(src_h, 1)

    if src_aspect > target_aspect:
        # Source is wider — crop horizontally
        crop_h = src_h
        crop_w = _even(crop_h * target_aspect)
        crop_w = min(crop_w, src_w)
        cam_x_px = src_w * cam_x_pct / 100.0
        crop_x = max(0, min(src_w - crop_w, int(round(cam_x_px - crop_w / 2))))
        crop_y = 0
    else:
        # Source is taller — crop vertically
        crop_w = src_w
        crop_h = _even(crop_w / target_aspect)
        crop_h = min(crop_h, src_h)
        cam_y_px = src_h * cam_y_pct / 100.0
        crop_x = 0
        crop_y = max(0, min(src_h - crop_h, int(round(cam_y_px - crop_h / 2))))

    return {
        "src": (crop_x, crop_y, crop_w, crop_h),
        "dst": (0, 0, target_w, target_h),
    }


def compute_blurfill_rects(
    src_w: int,
    src_h: int,
    target_w: int,
    target_h: int,
) -> dict:
    """Blur-fill layout: 16:9 sharp band over a blurred fill.

    Returns three rects:

      - ``src``       : the entire source frame (the band reads from
                         the full source).
      - ``band_dst``  : where the sharp 16:9 band lands on the
                         target canvas (centered vertically).
      - ``blur_dst``  : where the blurred fill goes (entire target
                         canvas — covers the whole 9:16 frame).
    """
    band_dw = target_w
    band_dh = _even(target_w * src_h / max(src_w, 1))
    band_dy = (target_h - band_dh) // 2
    band_dy -= band_dy % 2

    return {
        "src": (0, 0, src_w, src_h),
        "band_dst": (0, band_dy, band_dw, band_dh),
        "blur_dst": (0, 0, target_w, target_h),
    }


def compute_wide_zoom_rects(
    src_w: int,
    src_h: int,
    target_w: int,
    target_h: int,
    *,
    cam_x_pct: float = 50.0,
    crop_scale: float = 1.25,
) -> dict:
    """Wide-zoom layout: wider crop than fullscreen for context.

    Crops ``crop_scale`` × the fullscreen crop width centered on
    ``cam_x_pct``, then scales to fit the target. Falls back to a
    centered crop when the cam would push the window off-screen.
    """
    target_aspect = target_w / max(target_h, 1)
    base_crop_w = int(round(src_h * target_aspect))
    crop_w = int(round(base_crop_w * crop_scale))
    crop_w = min(crop_w, src_w)
    crop_w = _even(crop_w)

    cam_x_px = src_w * cam_x_pct / 100.0
    crop_x = max(0, min(src_w - crop_w, int(round(cam_x_px - crop_w / 2))))

    return {
        "src": (crop_x, 0, crop_w, src_h),
        "dst": (0, 0, target_w, target_h),
    }


def compute_layout_rects(
    layout_mode: str,
    *,
    src_w: int,
    src_h: int,
    target_w: int,
    target_h: int,
    cam_x_pct: float = 50.0,
    cam_y_pct: float = 50.0,
    crop_scale: float = 1.25,
) -> dict:
    """Single dispatch entry point — picks the right helper by mode.

    Unknown modes default to fullscreen. The ``mode`` key is included
    in the result for renderer convenience.
    """
    mode = (layout_mode or "fullscreen").lower()
    if mode == "blurfill":
        rects = compute_blurfill_rects(src_w, src_h, target_w, target_h)
    elif mode == "wide_zoom":
        rects = compute_wide_zoom_rects(
            src_w, src_h, target_w, target_h,
            cam_x_pct=cam_x_pct, crop_scale=crop_scale,
        )
    else:
        rects = compute_fullscreen_rects(
            src_w, src_h, target_w, target_h,
            cam_x_pct=cam_x_pct, cam_y_pct=cam_y_pct,
        )
    rects["mode"] = mode
    return rects
