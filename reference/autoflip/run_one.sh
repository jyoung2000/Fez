#!/usr/bin/env bash
# Run MediaPipe AutoFlip on one clip and emit a stable JSON timeline.
#
# Usage (inside the clipai/autoflip-ref Docker image):
#   run_one.sh <input.mp4> <output.json> [aspect_ratio]
#
# aspect_ratio defaults to "9:16". The emitted JSON schema matches
# backend/scripts/export_autoflip_compatible.py so the Week 3
# comparison harness can diff AutoFlip and ClipAI outputs uniformly.
#
# See reference/autoflip/REFERENCE_OUTPUTS.md for the pinned
# MEDIAPIPE_REF and cache-generation workflow.

set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "usage: run_one.sh <input.mp4> <output.json> [aspect_ratio]" >&2
  exit 1
fi

INPUT="$1"
OUTPUT_JSON="$2"
ASPECT="${3:-9:16}"

if [[ ! -f "$INPUT" ]]; then
  echo "input not found: $INPUT" >&2
  exit 2
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
OUT_MP4="$WORK/out.mp4"

# Run AutoFlip. stderr carries per-frame crop decisions; we parse them
# below. stdout is discarded. A non-zero exit from the binary surfaces
# through ``set -e`` and is treated as a hard failure.
run_autoflip \
  --calculator_graph_config_file=/etc/autoflip/autoflip_graph.pbtxt \
  --input_side_packets="input_video_path=${INPUT},output_video_path=${OUT_MP4},aspect_ratio=${ASPECT}" \
  2> "$WORK/stderr.log"

# The AutoFlip scene_cropping_calculator logs per-frame crop decisions
# to stderr in the form:
#
#   I<date> <time>  <threadid> scene_cropping_calculator.cc:<line>] \
#     frame=<N> crop_x=<X> crop_y=<Y> crop_w=<W> crop_h=<H> \
#     [scene_change=0|1]
#
# The log format drifts between MediaPipe releases. If this regex
# produces zero events, dump the first 50 lines of
# /tmp/autoflip-stderr.log and grep for crop_x to see the actual
# format in the pinned release, then update the pattern below.
python3 - <<PYEOF
import json, re, subprocess, os, sys

STDERR_LOG = "$WORK/stderr.log"
OUTPUT_JSON = "$OUTPUT_JSON"
INPUT = "$INPUT"
ASPECT = "$ASPECT"

stderr_text = open(STDERR_LOG).read()

# Probe source dimensions + fps via ffprobe for coordinate normalization.
probe = subprocess.run(
    ["ffprobe", "-v", "error", "-select_streams", "v:0",
     "-show_entries", "stream=width,height,r_frame_rate",
     "-of", "csv=p=0", INPUT],
    capture_output=True, text=True, check=True,
)
w_str, h_str, fps_str = probe.stdout.strip().split(",")
src_w, src_h = int(w_str), int(h_str)
num, den = fps_str.split("/")
fps = float(num) / float(den) if float(den) else float(num)

# Regex: scene_change field is optional (older MediaPipe versions
# omitted it). crop_x / crop_y may be signed.
pattern = re.compile(
    r"frame=(\d+)\s+crop_x=(-?\d+)\s+crop_y=(-?\d+)"
    r"\s+crop_w=(\d+)\s+crop_h=(\d+)"
    r"(?:\s+scene_change=(\d))?"
)

events = []
for m in pattern.finditer(stderr_text):
    fi, cx, cy, cw, ch, sc = m.groups()
    cx_px = int(cx) + int(cw) / 2.0
    cy_px = int(cy) + int(ch) / 2.0
    events.append({
        "frame": int(fi),
        "t": int(fi) / fps,
        "crop_cx": cx_px / src_w,
        "crop_cy": cy_px / src_h,
        "crop_w": int(cw) / src_w,
        "crop_h": int(ch) / src_h,
        "scene_change": bool(int(sc)) if sc is not None else False,
    })

if not events:
    sys.stderr.write(
        "WARN: no events parsed from AutoFlip stderr. "
        "First 50 lines follow for debugging:\n"
    )
    sys.stderr.write("".join(stderr_text.splitlines(True)[:50]))

timeline = {
    "tool": "mediapipe_autoflip",
    "version": os.environ.get("MEDIAPIPE_REF", "unknown"),
    "source_width": src_w,
    "source_height": src_h,
    "source_fps": fps,
    "aspect_ratio": ASPECT,
    "events": events,
}
with open(OUTPUT_JSON, "w") as f:
    json.dump(timeline, f, separators=(",", ":"))
sys.stderr.write(f"wrote {len(events)} events to {OUTPUT_JSON}\n")
PYEOF
