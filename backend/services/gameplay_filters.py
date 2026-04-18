"""Pure FFmpeg filtergraph builders for the gameplay layout modes.

Lives in its own module so the filter graphs are unit-testable
without pulling in ``clip_exporter``'s database / aiofiles /
pydantic-settings dependency chain. ``clip_exporter`` re-exports
these helpers for production callers.

Each builder takes source + target dimensions in pixels and
returns a single FFmpeg filtergraph string ending with ``[v]``.

The rectangle math is delegated to
``backend.services.gameplay_geometry`` so the FFmpeg filter
graphs and the frontend Canvas renderer use the SAME pixel
coordinates — that's the preview/export parity contract for
Phase 5.
"""

from __future__ import annotations

from backend.services.gameplay_geometry import (
    compute_blurfill_rects,
    compute_wide_zoom_rects,
)


def build_gameplay_blurfill_filter(
    src_w: int,
    src_h: int,
    target_w: int,
    target_h: int,
    *,
    blur_radius: int = 20,
) -> str:
    """Build FFmpeg filtergraph for the gameplay blur-fill layout.

    A scaled, blurred copy of the source fills the entire 9:16
    output frame; the sharp 16:9 source band is then composited
    centered on top. Industry standard for MOBA / lane-fight
    moments where the whole horizontal play area matters.

    Args:
        src_w / src_h: Source video dimensions in pixels.
        target_w / target_h: Output (9:16) dimensions in pixels.
        blur_radius: ``boxblur`` radius. 20 ≈ Gaussian σ=60 in
            CSS units (matches industry norm).

    Returns:
        FFmpeg filter graph string ending with ``[v]``.
    """
    rects = compute_blurfill_rects(src_w, src_h, target_w, target_h)
    _band_dx, band_dy, band_w, band_h = rects["band_dst"]
    band_h = max(2, band_h)

    return (
        f"split=2[band_src][blur_src];"
        f"[band_src]scale={band_w}:{band_h}[band];"
        f"[blur_src]scale={target_w}:{target_h}:"
        f"force_original_aspect_ratio=increase,"
        f"crop={target_w}:{target_h},"
        f"boxblur={blur_radius}:{blur_radius}[blur];"
        f"[blur][band]overlay=0:{band_dy}[v]"
    )


def build_gameplay_wide_zoom_filter(
    src_w: int,
    src_h: int,
    target_w: int,
    target_h: int,
    *,
    crop_scale: float = 1.25,
    cam_x_pct: float = 50.0,
) -> str:
    """Build FFmpeg filtergraph for the wide-zoom gameplay layout.

    Crops the source ``crop_scale`` × wider than the default 9:16
    crop window centered at ``cam_x_pct`` % of source width, then
    scales to fit the target. The result shows ~25 % more
    horizontal context than ``fullscreen``, useful for chaos
    moments where a single subject doesn't capture the action.

    Falls back to a centered crop when the requested cam_x would
    push the crop window outside the source frame.
    """
    rects = compute_wide_zoom_rects(
        src_w, src_h, target_w, target_h,
        cam_x_pct=cam_x_pct, crop_scale=crop_scale,
    )
    crop_x, crop_y, crop_w, crop_h = rects["src"]

    return (
        f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},"
        f"scale={target_w}:{target_h}[v]"
    )
