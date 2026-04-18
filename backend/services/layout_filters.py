"""FFmpeg filter-complex templates for composite layout modes.

Each function returns an FFmpeg filtergraph string that can be passed
to clip_exporter for rendering. The preview (CSS) and export (FFmpeg)
use matching parameters for preview/export parity.
"""


def build_blur_fill_filter(
    input_w: int,
    input_h: int,
    output_w: int = 1080,
    output_h: int = 1920,
    blur_sigma: int = 50,
    brightness: float = -0.1,
) -> str:
    """Blur-fill: original centered, blurred duplicate behind.

    Used when the full 16:9 frame must be preserved (e.g., gaming with HUD)
    but the output is 9:16. The original is scaled to fit width, then
    overlaid on a heavily blurred, slightly darkened version that fills
    the vertical space.
    """
    return (
        f"[0:v]split=2[orig][bg];"
        f"[bg]scale={output_w * 2}:{output_h * 2}:force_original_aspect_ratio=increase,"
        f"crop={output_w}:{output_h},gblur=sigma={blur_sigma},"
        f"eq=brightness={brightness}[blurred];"
        f"[orig]scale={output_w}:-1[fg];"
        f"[blurred][fg]overlay=(W-w)/2:(H-h)/2[out]"
    )


def build_wide_master_filter(
    input_w: int,
    input_h: int,
    output_w: int = 1080,
    output_h: int = 1920,
    bg_color: str = "black",
) -> str:
    """Wide-master: original 16:9 centered vertically, bars top/bottom.

    Used for action scenes in narrative or default sports framing.
    The original is scaled to fit width, centered vertically.
    """
    return (
        f"[0:v]scale={output_w}:-1[fg];"
        f"color=c={bg_color}:s={output_w}x{output_h}:d=1[bg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2[out]"
    )


def build_stacked_gameplay_filter(
    gameplay_crop_x: int,
    gameplay_crop_y: int,
    gameplay_crop_w: int,
    gameplay_crop_h: int,
    facecam_crop_x: int,
    facecam_crop_y: int,
    facecam_crop_w: int,
    facecam_crop_h: int,
    output_w: int = 1080,
    output_h: int = 1920,
    gameplay_pct: float = 0.6,
) -> str:
    """Stacked gameplay: gameplay on top, facecam on bottom.

    The gameplay region is cropped and scaled to fill the top portion.
    The facecam region is cropped and scaled to fill the bottom portion.
    """
    top_h = int(output_h * gameplay_pct)
    bot_h = output_h - top_h
    return (
        f"[0:v]split=2[gp][fc];"
        f"[gp]crop={gameplay_crop_w}:{gameplay_crop_h}:{gameplay_crop_x}:{gameplay_crop_y},"
        f"scale={output_w}:{top_h}[gameplay];"
        f"[fc]crop={facecam_crop_w}:{facecam_crop_h}:{facecam_crop_x}:{facecam_crop_y},"
        f"scale={output_w}:{bot_h}[facecam];"
        f"[gameplay][facecam]vstack[out]"
    )


def build_split_screen_filter(
    face1_x: int,
    face2_x: int,
    input_w: int,
    input_h: int,
    output_w: int = 1080,
    output_h: int = 1920,
) -> str:
    """Split-screen: two speakers stacked vertically.

    Each speaker gets a crop region centered on their face position,
    scaled to fill half the output height.
    """
    half_h = output_h // 2
    crop_w = int(input_h * (output_w / half_h))  # Maintain output AR per half
    crop_w = min(crop_w, input_w)

    def make_crop(face_x_pct):
        cx = int(face_x_pct / 100 * input_w)
        x = max(0, min(input_w - crop_w, cx - crop_w // 2))
        return x

    x1 = make_crop(face1_x)
    x2 = make_crop(face2_x)

    return (
        f"[0:v]split=2[top][bot];"
        f"[top]crop={crop_w}:{input_h}:{x1}:0,scale={output_w}:{half_h}[t];"
        f"[bot]crop={crop_w}:{input_h}:{x2}:0,scale={output_w}:{half_h}[b];"
        f"[t][b]vstack[out]"
    )


def build_grid_filter(
    face_positions: list[int],
    input_w: int,
    input_h: int,
    output_w: int = 1080,
    output_h: int = 1920,
) -> str:
    """Grid layout: 2x2 tiles for 3-4 speakers.

    Each tile is a crop of the source centered on the speaker's face.
    """
    n = min(4, len(face_positions))
    tile_w = output_w // 2
    tile_h = output_h // 2
    crop_w = int(input_h * (tile_w / tile_h))
    crop_w = min(crop_w, input_w)

    parts = []
    labels = []
    for i in range(n):
        cx = int(face_positions[i] / 100 * input_w)
        x = max(0, min(input_w - crop_w, cx - crop_w // 2))
        parts.append(
            f"[0:v]crop={crop_w}:{input_h}:{x}:0,scale={tile_w}:{tile_h}[tile{i}]"
        )
        labels.append(f"tile{i}")

    # Fill empty tiles with black
    for i in range(n, 4):
        parts.append(f"color=c=black:s={tile_w}x{tile_h}:d=1[tile{i}]")
        labels.append(f"tile{i}")

    parts.append(f"[{labels[0]}][{labels[1]}]hstack[row0]")
    parts.append(f"[{labels[2]}][{labels[3]}]hstack[row1]")
    parts.append("[row0][row1]vstack[out]")

    return ";".join(parts)
