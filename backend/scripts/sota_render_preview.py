"""Render a 9:16 preview MP4 from the SOTA reframing pipeline output.

Reads ``segments.json`` (the per-clip extraction-cache file produced
by ``compare_autoflip_vs_clipai`` when it runs the full pipeline)
and crops the source MP4 segment-by-segment using ffmpeg's ``crop``
filter with a time-piecewise ``x`` expression.

Output: a 1080x1920 MP4 with the source's audio passed through.

Usage:

    python -m backend.scripts.sota_render_preview \\
        --input  /path/to/source.mp4 \\
        --segments /path/to/segments.json \\
        --output /path/to/preview.mp4

Exits 0 on success, non-zero with a human-readable message on
failure. Writes nothing to stdout EXCEPT progress events the SSE
stream can forward to the GUI.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUT_W = 1080
OUTPUT_H = 1920


def _build_x_expression(segments: list[dict], src_w: int, crop_w: int) -> str:
    """Build an ffmpeg crop-x expression that picks the right x per second.

    Each keypoint produced by ``build_camera_path_keypoints`` is the
    pixel position of the subject centroid at a given time in the
    SOURCE frame. We crop a window of width ``crop_w`` centred on that
    x, clamped so the crop window stays inside the source frame.

    The output is a FLAT SUM of piecewise-LINEAR lerp terms gated by
    half-open ``gte(t,t0)*lt(t,t1)`` intervals, plus a final
    ``gte(t,t_last)`` hold so the crop doesn't snap back to 0 past the
    last keypoint. Adjacent intervals are non-overlapping, so exactly
    one term is non-zero at any given t and the sum equals the
    interpolated x for that time. The flat sum has no recursion
    depth, so it scales to thousands of keypoints without hitting
    ffmpeg's expression-parser depth limit.

    For segments with a ``motion_path`` the renderer draws a smooth
    pan along the path. For segments with ``ease_in_ms > 0`` at a
    motivated cut, the keypoint builder emits smoothstep samples so
    the linear lerp between them traces an S-curve.

    For OLD cached ``segments.json`` blobs that pre-date the
    ``motion_path`` serialization, every segment falls back to a
    held ``(start, subject_x)`` → ``(end, subject_x)`` keypoint pair,
    so the rendered preview is bit-identical to the legacy
    step-function behavior.
    """
    if not segments:
        # Centre crop fallback when segments are missing.
        return str((src_w - crop_w) // 2)

    half = crop_w / 2.0
    max_x = max(0, src_w - crop_w)

    def _clamp(x: float) -> int:
        return max(0, min(max_x, int(round(x - half))))

    # Lazy import: the renderer is occasionally invoked on hosts
    # without the full backend stack installed, but
    # ``export_autoflip_compatible`` is pure Python with no heavy
    # dependencies so the import is safe.
    from backend.scripts.export_autoflip_compatible import (
        build_camera_path_keypoints,
    )

    keypoints = build_camera_path_keypoints(segments)
    if not keypoints:
        return str((src_w - crop_w) // 2)

    # Single keypoint → degenerate hold.
    if len(keypoints) == 1:
        x = _clamp(keypoints[0][1])
        t0 = float(keypoints[0][0])
        return f"{x}*gte(t,{t0:.3f})"

    terms: list[str] = []
    for i in range(len(keypoints) - 1):
        t0 = float(keypoints[i][0])
        t1 = float(keypoints[i + 1][0])
        if t1 <= t0:
            continue
        x0 = _clamp(keypoints[i][1])
        x1 = _clamp(keypoints[i + 1][1])
        if x0 == x1:
            terms.append(f"{x0}*gte(t,{t0:.3f})*lt(t,{t1:.3f})")
        else:
            dx = x1 - x0
            dt = t1 - t0
            terms.append(
                f"({x0}+({dx})*(t-{t0:.3f})/({dt:.3f}))"
                f"*gte(t,{t0:.3f})*lt(t,{t1:.3f})"
            )

    # Tail hold so x doesn't drop to 0 once we pass the last keypoint
    # (ffmpeg keeps evaluating the expression for the trailing tail of
    # the source clip).
    t_last = float(keypoints[-1][0])
    x_last = _clamp(keypoints[-1][1])
    terms.append(f"{x_last}*gte(t,{t_last:.3f})")

    if not terms:
        return str((src_w - crop_w) // 2)

    # The expression sits inside single quotes in the -vf string, so
    # commas inside between(...) are not interpreted as filter-arg
    # separators by ffmpeg's outer parser. The terms join with '+'
    # because '+' is never an argument separator.
    return "+".join(terms)


def _probe_source(input_path: Path) -> tuple[int, int, float]:
    """Return ``(width, height, duration_sec)`` via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-show_entries", "format=duration",
        "-of", "json", str(input_path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {out.stderr.strip()}")
    data = json.loads(out.stdout or "{}")
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    if not streams:
        raise RuntimeError("ffprobe found no video stream")
    return (
        int(streams[0]["width"]),
        int(streams[0]["height"]),
        float(fmt.get("duration", 0.0)),
    )


def _print(line: str) -> None:
    """Flushed print so the SSE stream sees output line-by-line."""
    print(line, flush=True)


def render_preview(
    input_path: Path,
    segments_path: Path,
    output_path: Path,
    *,
    target_w: int = OUTPUT_W,
    target_h: int = OUTPUT_H,
) -> int:
    if not input_path.is_file():
        _print(f"!! source not found: {input_path}")
        return 2
    if not segments_path.is_file():
        _print(f"!! segments.json not found: {segments_path}")
        _print(
            "!! the bench did not finish, OR the extraction cache was "
            "deleted between bench and render. Re-run the bench first."
        )
        return 2

    _print(f"[render] probing {input_path.name}...")
    src_w, src_h, duration = _probe_source(input_path)
    _print(f"[render] source: {src_w}x{src_h}, {duration:.1f}s")

    segments = json.loads(segments_path.read_text())
    if not isinstance(segments, list) or not segments:
        _print(f"!! segments.json is empty or wrong shape: {type(segments).__name__}")
        return 2
    _print(f"[render] {len(segments)} segments to render")

    # Crop window: the largest 9:16 rectangle that fits inside the source
    # frame's height. For a 1920x1080 source -> crop_w = 1080 * 9/16 = 608.
    aspect_target = target_w / target_h
    aspect_src = src_w / max(src_h, 1)
    if aspect_target < aspect_src:
        # Source wider than target — crop horizontally.
        crop_w = int(round(src_h * aspect_target))
        crop_h = src_h
        x_expr = _build_x_expression(segments, src_w, crop_w)
        y_expr = "0"
    else:
        # Source narrower than target (vertical or square source) — crop vertically.
        crop_w = src_w
        crop_h = int(round(src_w / aspect_target))
        x_expr = "0"
        y_expr = str(max(0, (src_h - crop_h) // 2))

    _print(
        f"[render] crop window {crop_w}x{crop_h} @ x=<piecewise>, y={y_expr}; "
        f"scale -> {target_w}x{target_h}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ffmpeg command. We use libx264 (not nvenc) so the output renders
    # the same on any host; the diagnostics SSE pipeline doesn't have
    # access to the GPU anyway. Audio passes through.
    vf = (
        f"crop=w={crop_w}:h={crop_h}:x='{x_expr}':y={y_expr},"
        f"scale={target_w}:{target_h}:flags=lanczos,setsar=1"
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "info", "-stats",
        "-i", str(input_path),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path),
    ]

    _print("[render] running ffmpeg...")
    t0 = time.monotonic()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.monotonic() - t0
    if proc.returncode != 0:
        # Surface the last 30 ffmpeg lines to help debug.
        _print(f"!! ffmpeg failed in {elapsed:.1f}s (rc={proc.returncode})")
        for line in (proc.stdout or "").splitlines()[-30:]:
            _print(f"   {line}")
        return proc.returncode

    if not output_path.is_file() or output_path.stat().st_size < 1024:
        _print(f"!! ffmpeg succeeded but output is empty: {output_path}")
        return 3

    _print(
        f"[render] OK in {elapsed:.1f}s -> {output_path} "
        f"({output_path.stat().st_size / 1_048_576:.1f} MB)"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--segments", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return render_preview(args.input, args.segments, args.output)


if __name__ == "__main__":
    sys.exit(main())
