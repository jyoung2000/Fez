"""FFmpeg filter builder consuming a RenderPlan.

Generates a filter_complex_script file and the ffmpeg command args.
Both the full-video and clip export paths call build_ffmpeg_command()
with a RenderPlan produced by build_render_plan().

Design principles:
  - One video stream label per op: [v0], [v1], ..., final [outv] via concat.
  - Uses -filter_complex_script (temp file) to avoid argv length limits.
  - For clips, uses -ss before -i for fast seek; segment times are rebased to 0.
  - Never uses enable='between(t,...)' — splits/trims into discrete streams.
"""

import logging
import os
import tempfile
import uuid
from typing import List, Tuple

from backend.services.render_plan import (
    RenderOp,
    RenderOpKind,
    RenderPlan,
    Rect,
)

logger = logging.getLogger(__name__)

# Blur params must match frontend Canvas renderer exactly
BLUR_SIGMA = 50
BLUR_BRIGHTNESS = -0.1  # eq filter brightness offset


def build_ffmpeg_command(
    plan: RenderPlan,
    source_path: str,
    output_path: str,
    use_gpu: bool = True,
    interpolated_timeline=None,
) -> Tuple[List[str], str]:
    """Build an FFmpeg command from a RenderPlan.

    Returns:
        (cmd_args_list, filter_script_path) — the script path should be
        cleaned up after ffmpeg finishes.
    """
    filter_graph = _build_filter_graph(plan, interpolated_timeline=interpolated_timeline)
    script_path = _write_filter_script(filter_graph)

    cmd = []

    # Input seeking for clip exports
    if plan.source_offset_sec > 0:
        cmd.extend(["-ss", f"{plan.source_offset_sec:.3f}"])

    cmd.extend(["-i", source_path])

    if plan.source_offset_sec > 0:
        cmd.extend(["-t", f"{plan.total_duration_sec:.3f}"])

    cmd.extend([
        "-filter_complex_script", script_path,
        "-map", "[outv]",
    ])

    # Audio: map from input, copy if possible
    cmd.extend(["-map", "0:a?", "-c:a", "aac"])

    # Video encoding
    if use_gpu:
        cmd.extend(["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"])
    else:
        cmd.extend(["-c:v", "libx264", "-preset", "medium", "-crf", "23"])

    cmd.extend([
        "-movflags", "+faststart",
        "-y",
        output_path,
    ])

    logger.info(
        "FFmpeg command built: %d ops, filter script at %s (%d bytes)",
        len(plan.ops), script_path, len(filter_graph),
    )

    return cmd, script_path


def _build_filter_graph(plan: RenderPlan, interpolated_timeline=None) -> str:
    """Build the complete filter_complex string for a RenderPlan."""
    lines = []
    op_labels = []  # [v0], [v1], ...
    src_w = plan.source_width
    src_h = plan.source_height
    tgt_w = plan.target_width
    tgt_h = plan.target_height

    for i, op in enumerate(plan.ops):
        op_filter = _build_op_filter(op, i, src_w, src_h, tgt_w, tgt_h,
                                     interpolated_timeline=interpolated_timeline)
        lines.append(op_filter)
        op_labels.append(f"[v{i}]")

    # Handle transitions (xfade between ops with ease_in_ms > 0)
    final_labels = _apply_transitions(plan.ops, op_labels, lines)

    # Concat all final labels into [outv]
    if len(final_labels) == 1:
        # Single op — just rename
        lines.append(f"{final_labels[0]}copy[outv]")
    else:
        concat_inputs = "".join(final_labels)
        lines.append(
            f"{concat_inputs}concat=n={len(final_labels)}:v=1:a=0[outv]"
        )

    return ";\n".join(lines)


def _build_op_filter(
    op: RenderOp,
    index: int,
    src_w: int,
    src_h: int,
    tgt_w: int,
    tgt_h: int,
    interpolated_timeline=None,
) -> str:
    """Build the filter chain for a single RenderOp."""
    label = f"v{index}"
    start = op.start_sec
    end = op.end_sec

    if op.kind == RenderOpKind.CROP:
        return _filter_crop(op, label, src_w, src_h, tgt_w, tgt_h, start, end)
    elif op.kind == RenderOpKind.TRACKING_CROP:
        if interpolated_timeline is not None:
            result = _filter_tracking_crop_from_timeline(
                op, label, src_w, src_h, tgt_w, tgt_h, start, end,
                interpolated_timeline,
            )
            if result is not None:
                return result
        return _filter_tracking_crop(op, label, src_w, src_h, tgt_w, tgt_h, start, end)
    elif op.kind == RenderOpKind.WIDE_MASTER:
        return _filter_wide_master(op, label, tgt_w, tgt_h, start, end)
    elif op.kind == RenderOpKind.BLUR_FILL:
        return _filter_blur_fill(op, label, tgt_w, tgt_h, start, end)
    elif op.kind == RenderOpKind.SPLIT_SCREEN:
        return _filter_split_screen(op, label, src_w, src_h, tgt_w, tgt_h, start, end)
    elif op.kind == RenderOpKind.STACKED_GAMEPLAY:
        return _filter_stacked_gameplay(op, label, src_w, src_h, tgt_w, tgt_h, start, end)
    elif op.kind == RenderOpKind.GRID_2X2:
        return _filter_grid_2x2(op, label, src_w, src_h, tgt_w, tgt_h, start, end)
    elif op.kind in (RenderOpKind.MOTIVATED_PUSH_IN, RenderOpKind.MOTIVATED_PULL_OUT):
        return _filter_motivated_zoom(op, label, src_w, src_h, tgt_w, tgt_h, start, end)
    else:
        # Fallback to crop
        return _filter_crop(op, label, src_w, src_h, tgt_w, tgt_h, start, end)


def _filter_motivated_zoom(op, label, src_w, src_h, tgt_w, tgt_h, start, end) -> str:
    """MOTIVATED_PUSH_IN / MOTIVATED_PULL_OUT: time-varying crop w/h/x/y.

    Two keypoints in ``motion_path`` are expected: the starting crop and
    the final crop. The crop ``w/h`` interpolate linearly alongside
    ``x/y``. Uses FFmpeg's expression parser (not ``zoompan``, which
    pixelates) so the downscale to output runs through the lanczos
    scaler at full source resolution.

    For simplicity and to stay within the expression-length budget,
    motivated zooms emit a single interpolation segment. The scheduler
    in ``motivated_zoom.py`` caps durations at a few seconds, so one
    segment is enough.
    """
    kps = op.motion_path or []
    if len(kps) < 2:
        return _filter_crop(op, label, src_w, src_h, tgt_w, tgt_h, start, end)

    kp0, kp1 = kps[0], kps[-1]
    t0 = max(0.0, kp0.t)
    t1 = max(t0 + 1e-3, kp1.t)

    x0 = kp0.rect.x * src_w
    y0 = kp0.rect.y * src_h
    w0 = kp0.rect.w * src_w
    h0 = kp0.rect.h * src_h
    x1 = kp1.rect.x * src_w
    y1 = kp1.rect.y * src_h
    w1 = kp1.rect.w * src_w
    h1 = kp1.rect.h * src_h

    def _lin(a: float, b: float) -> str:
        if abs(b - a) < 1e-3:
            return f"{a:.2f}"
        return f"{a:.2f}+({b - a:.2f})*clip((t-{t0:.3f})/{t1 - t0:.3f}\\,0\\,1)"

    x_expr = _lin(x0, x1)
    y_expr = _lin(y0, y1)
    w_expr = _lin(w0, w1)
    h_expr = _lin(h0, h1)

    # Use max dimensions for the crop filter's static size; the
    # ``w_expr`` / ``h_expr`` are applied via the crop expression
    # which FFmpeg evaluates per frame.
    return (
        f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS,"
        f"crop=w={w_expr}:h={h_expr}:x={x_expr}:y={y_expr}:exact=1,"
        f"scale={tgt_w}:{tgt_h}:flags=lanczos[{label}]"
    )


def _filter_crop(op, label, src_w, src_h, tgt_w, tgt_h, start, end) -> str:
    """CROP: static rectangle crop."""
    px, py, pw, ph = op.primary_rect.to_pixels(src_w, src_h)
    return (
        f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS,"
        f"crop={pw}:{ph}:{px}:{py},"
        f"scale={tgt_w}:{tgt_h}:flags=lanczos[{label}]"
    )


def _filter_tracking_crop(op, label, src_w, src_h, tgt_w, tgt_h, start, end) -> str:
    """TRACKING_CROP: animated crop via motion_path keypoints.

    Uses a piecewise-linear x expression driven by motion_path.
    """
    # Build the crop dimensions from the first keypoint rect
    first_rect = op.motion_path[0].rect if op.motion_path else op.primary_rect
    _, _, pw, ph = first_rect.to_pixels(src_w, src_h)

    # Build piecewise-linear x expression from keypoints
    x_expr = _build_piecewise_x_expr(op.motion_path, src_w, src_h, pw)
    max_x = src_w - pw
    # Clamp the expression
    x_expr_clamped = f"clip({x_expr}\\,0\\,{max_x})"

    return (
        f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS,"
        f"crop={pw}:{ph}:{x_expr_clamped}:0,"
        f"scale={tgt_w}:{tgt_h}:flags=lanczos[{label}]"
    )


FFMPEG_EXPR_MAX_LEN = 8000  # FFmpeg expression parser limit


def _filter_tracking_crop_from_timeline(
    op, label, src_w, src_h, tgt_w, tgt_h, start, end,
    interpolated_timeline,
) -> str:
    """Build a TRACKING_CROP filter from interpolated timeline samples.

    Returns None if timeline doesn't have enough samples, falling back
    to the standard piecewise expression.
    """
    # Pull per-frame samples from the timeline within this op's range
    samples = []
    for s in interpolated_timeline.samples:
        if s.timestamp < start or s.timestamp > end:
            continue
        if not s.bboxes:
            continue
        # Use the first available slot's bbox center
        slot_id, bbox = next(iter(s.bboxes.items()))
        cx_pct, cy_pct, _, _ = bbox
        samples.append((s.timestamp - start, cx_pct))  # Rebased to 0

    if len(samples) < 2:
        return None

    first_rect = op.motion_path[0].rect if op.motion_path else op.primary_rect
    _, _, pw, ph = first_rect.to_pixels(src_w, src_h)
    max_x = src_w - pw

    # Downsample if expression would be too long
    # Each segment adds ~60 chars: "if(between(t,0.033,0.067),100.0+(5.0)*(t-0.033)/(0.033),"
    max_segments = FFMPEG_EXPR_MAX_LEN // 65
    if len(samples) - 1 > max_segments:
        step = max(1, len(samples) // max_segments)
        downsampled = samples[::step]
        if downsampled[-1] != samples[-1]:
            downsampled.append(samples[-1])
        logger.warning(
            "FFmpeg timeline expression downsampled from %d to %d samples "
            "(max expression length %d chars)",
            len(samples), len(downsampled), FFMPEG_EXPR_MAX_LEN,
        )
        samples = downsampled

    # Build piecewise-linear expression
    expr_parts = []
    for i in range(len(samples) - 1):
        t0, x0_pct = samples[i]
        t1, x1_pct = samples[i + 1]
        x0_px = (x0_pct / 100 * src_w) - pw / 2
        x1_px = (x1_pct / 100 * src_w) - pw / 2
        dt = t1 - t0
        if dt <= 0:
            continue
        dx = x1_px - x0_px
        if dx == 0:
            seg_expr = f"{x0_px:.1f}"
        else:
            seg_expr = f"{x0_px:.1f}+{dx:.1f}*(t-{t0:.3f})/{dt:.3f}"
        expr_parts.append(f"if(between(t\\,{t0:.3f}\\,{t1:.3f})\\,{seg_expr}\\,")

    if not expr_parts:
        return None

    # Final fallback value (last sample's position)
    last_x_px = (samples[-1][1] / 100 * src_w) - pw / 2
    expr = "".join(expr_parts) + f"{last_x_px:.1f}" + ")" * len(expr_parts)
    expr_clamped = f"clip({expr}\\,0\\,{max_x})"

    return (
        f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS,"
        f"crop={pw}:{ph}:{expr_clamped}:0,"
        f"scale={tgt_w}:{tgt_h}:flags=lanczos[{label}]"
    )


def _build_piecewise_x_expr(keypoints, src_w, src_h, crop_w) -> str:
    """Build a piecewise-linear FFmpeg expression for x offset from keypoints."""
    if not keypoints:
        return "0"

    if len(keypoints) == 1:
        px, _, _, _ = keypoints[0].rect.to_pixels(src_w, src_h)
        return str(px)

    # Build nested if(lt(t,...), ...) expression
    # For each segment between keypoints i and i+1:
    #   x = x_i + (x_{i+1} - x_i) * (t - t_i) / (t_{i+1} - t_i)
    expr = ""
    for i in range(len(keypoints) - 1):
        t0 = keypoints[i].t
        t1 = keypoints[i + 1].t
        px0, _, _, _ = keypoints[i].rect.to_pixels(src_w, src_h)
        px1, _, _, _ = keypoints[i + 1].rect.to_pixels(src_w, src_h)

        dt = t1 - t0
        if dt <= 0:
            continue

        dx = px1 - px0
        # Linear interpolation: px0 + dx * (t - t0) / dt
        if dx == 0:
            segment_expr = str(px0)
        else:
            segment_expr = f"{px0}+{dx}*(t-{t0:.3f})/{dt:.3f}"

        if i == 0:
            expr = f"if(lt(t\\,{t1:.3f})\\,{segment_expr}\\,"
        elif i < len(keypoints) - 2:
            expr += f"if(lt(t\\,{t1:.3f})\\,{segment_expr}\\,"
        else:
            # Last segment — no condition needed, just the expression
            expr += segment_expr

    # Close all the if() parentheses
    close_count = max(0, len(keypoints) - 2)
    expr += ")" * close_count

    return expr


def _filter_wide_master(op, label, tgt_w, tgt_h, start, end) -> str:
    """WIDE_MASTER: letterbox with black bars."""
    return (
        f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS,"
        f"scale={tgt_w}:-1:flags=lanczos,"
        f"pad={tgt_w}:{tgt_h}:0:(oh-ih)/2:black[{label}]"
    )


def _filter_blur_fill(op, label, tgt_w, tgt_h, start, end) -> str:
    """BLUR_FILL: source centered, blurred duplicate as background."""
    return (
        f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS,"
        f"split=2[fg{label}][bg{label}];\n"
        f"[bg{label}]scale={tgt_w}:{tgt_h}:force_original_aspect_ratio=increase,"
        f"crop={tgt_w}:{tgt_h},"
        f"gblur=sigma={BLUR_SIGMA},eq=brightness={BLUR_BRIGHTNESS}[bgblur{label}];\n"
        f"[fg{label}]scale={tgt_w}:-1:flags=lanczos[fgs{label}];\n"
        f"[bgblur{label}][fgs{label}]overlay=(W-w)/2:(H-h)/2[{label}]"
    )


def _filter_split_screen(op, label, src_w, src_h, tgt_w, tgt_h, start, end) -> str:
    """SPLIT_SCREEN: two crops stacked vertically, 50/50."""
    half_h = tgt_h // 2
    half_h = half_h - (half_h % 2)

    px, py, pw, ph = op.primary_rect.to_pixels(src_w, src_h)
    sx, sy, sw, sh = op.secondary_rect.to_pixels(src_w, src_h)

    return (
        f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS,"
        f"split=2[top_src{label}][bot_src{label}];\n"
        f"[top_src{label}]crop={pw}:{ph}:{px}:{py},"
        f"scale={tgt_w}:{half_h}:flags=lanczos[top{label}];\n"
        f"[bot_src{label}]crop={sw}:{sh}:{sx}:{sy},"
        f"scale={tgt_w}:{half_h}:flags=lanczos[bot{label}];\n"
        f"[top{label}][bot{label}]vstack[{label}]"
    )


def _filter_stacked_gameplay(op, label, src_w, src_h, tgt_w, tgt_h, start, end) -> str:
    """STACKED_GAMEPLAY: gameplay top 60%, facecam bottom 40%."""
    top_h = int(tgt_h * 0.6)
    top_h = top_h - (top_h % 2)
    bot_h = tgt_h - top_h

    px, py, pw, ph = op.primary_rect.to_pixels(src_w, src_h)
    sx, sy, sw, sh = op.secondary_rect.to_pixels(src_w, src_h)

    return (
        f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS,"
        f"split=2[game_src{label}][cam_src{label}];\n"
        f"[game_src{label}]crop={pw}:{ph}:{px}:{py},"
        f"scale={tgt_w}:{top_h}:flags=lanczos[game{label}];\n"
        f"[cam_src{label}]crop={sw}:{sh}:{sx}:{sy},"
        f"scale={tgt_w}:{bot_h}:flags=lanczos[cam{label}];\n"
        f"[game{label}][cam{label}]vstack[{label}]"
    )


def _filter_grid_2x2(op, label, src_w, src_h, tgt_w, tgt_h, start, end) -> str:
    """GRID_2X2: 4 tiles in 2x2 layout using xstack."""
    tile_w = tgt_w // 2
    tile_w = tile_w - (tile_w % 2)
    tile_h = tgt_h // 2
    tile_h = tile_h - (tile_h % 2)

    rects = [op.primary_rect, op.secondary_rect, op.tertiary_rect, op.quaternary_rect]
    lines = [
        f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS,"
        f"split=4[g0_{label}][g1_{label}][g2_{label}][g3_{label}]"
    ]

    for j, rect in enumerate(rects):
        px, py, pw, ph = rect.to_pixels(src_w, src_h)
        lines.append(
            f"[g{j}_{label}]crop={pw}:{ph}:{px}:{py},"
            f"scale={tile_w}:{tile_h}:flags=lanczos[gt{j}_{label}]"
        )

    # xstack layout: 2x2 grid
    lines.append(
        f"[gt0_{label}][gt1_{label}][gt2_{label}][gt3_{label}]"
        f"xstack=inputs=4:layout=0_0|{tile_w}_0|0_{tile_h}|{tile_w}_{tile_h}[{label}]"
    )

    return ";\n".join(lines)


def _apply_transitions(ops, op_labels, lines) -> List[str]:
    """Apply xfade transitions between ops where ease_in_ms > 0.

    Returns the final list of stream labels to concat.
    """
    if len(ops) <= 1:
        return list(op_labels)

    final_labels = [op_labels[0]]

    for i in range(1, len(ops)):
        ease_ms = ops[i].ease_in_ms
        if ease_ms > 0:
            # Insert xfade transition
            dur = ease_ms / 1000.0
            offset = ops[i].start_sec - dur / 2.0
            offset = max(0, offset)

            prev_label = final_labels[-1]
            curr_label = op_labels[i]
            xfade_label = f"[vx{i}]"

            lines.append(
                f"{prev_label}{curr_label}"
                f"xfade=transition=fade:duration={dur:.3f}:offset={offset:.3f}"
                f"{xfade_label}"
            )
            final_labels[-1] = xfade_label
        else:
            final_labels.append(op_labels[i])

    return final_labels


def _write_filter_script(filter_graph: str) -> str:
    """Write filter graph to a temp file, return the path."""
    script_dir = tempfile.gettempdir()
    script_name = f"filter_graph_{uuid.uuid4().hex[:12]}.txt"
    script_path = os.path.join(script_dir, script_name)
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(filter_graph)
    return script_path


def cleanup_filter_script(script_path: str):
    """Remove the filter script temp file."""
    try:
        if script_path and os.path.exists(script_path):
            os.remove(script_path)
    except OSError:
        pass
