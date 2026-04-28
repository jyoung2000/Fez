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

    Each segment's ``subject_x`` is the pixel position of the subject
    centroid in the SOURCE frame. We crop a window of width ``crop_w``
    centred on that x, clamped so the crop window stays inside the
    source frame.

    The output is a nested if() expression of the form:
        if(lt(t,e1), x1, if(lt(t,e2), x2, ... last_x))
    """
    if not segments:
        # Centre crop fallback when segments are missing.
        return str((src_w - crop_w) // 2)

    half = crop_w // 2
    max_x = max(0, src_w - crop_w)

    def _clamp(x: float) -> int:
        clamped = max(0, min(max_x, int(round(x - half))))
        return clamped

    parts = []
    for s in segments:
        end = float(s.get("end", 0.0))
        x = float(s.get("subject_x", src_w / 2.0))
        parts.append((end, _clamp(x)))
    parts.sort(key=lambda p: p[0])

    # Build right-to-left: ... if(lt(t, e_n-1), x_n-1, x_n) ...
    last_x = parts[-1][1]
    expr = str(last_x)
    for end, x in reversed(parts[:-1]):
        expr = f"if(lt(t,{end:.3f}),{x},{expr})"
    return expr


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
