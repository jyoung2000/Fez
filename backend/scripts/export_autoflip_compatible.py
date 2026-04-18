#!/usr/bin/env python3
"""Week 3 — translate a ClipAI render plan or reframe-segment list
into the MediaPipe AutoFlip JSON shape for apples-to-apples comparison.

Output schema matches ``reference/autoflip/run_one.sh``:

    {
      "tool": "clipai",
      "version": "week3",
      "source_width":  int,
      "source_height": int,
      "source_fps":    float,
      "aspect_ratio":  "9:16",
      "events": [
        {
          "frame": int,
          "t": float,            // seconds
          "crop_cx": float,      // 0-1 normalized by source width
          "crop_cy": float,      // 0-1 normalized by source height
          "crop_w": float,
          "crop_h": float,
          "scene_change": bool   // True on the first frame of each op/segment
        },
        ...
      ]
    }

The translator interpolates the op-based ClipAI plan to per-frame
events at the source fps. The Week 3 comparison harness
(``backend/scripts/compare_autoflip_vs_clipai.py``) runs the metric
library over both the AutoFlip and ClipAI event lists in the same
coordinate system so the numbers are directly comparable.

Two input modes are supported (use exactly one):

    --render-plan-json <path>
        Path to a JSON dump of ``RenderPlan.to_dict()`` — the
        render-plan-builder output. Each op's ``primary_rect`` is
        assumed to use 0-1 normalized source coords.

    --reframe-segments-json <path>
        Path to a JSON list of ``ReframeSegment`` dicts with
        ``start``, ``end``, ``subject_x``, ``subject_y`` fields.
        Subject coords are in source pixels (matching the pipeline's
        convention after Phase 0) and are normalized here.

Example:

    python -m backend.scripts.export_autoflip_compatible \\
        --video /tmp/clip.mp4 \\
        --reframe-segments-json /tmp/clip.segments.json \\
        --output /tmp/clip.clipai_timeline.json

Notes on faithfulness to the pipeline's actual output:

  * Tracking ops with a ``motion_path`` collapse to a single
    ``primary_rect`` per op at the render-plan layer. That matches
    what AutoFlip's per-shot crop motion does under the hood — both
    tools are sampled at source fps and emit stepwise crop centers
    around the shot boundaries.

  * The ``scene_change`` flag is set to True on the first event of
    each op/segment and False for every interior event. This is the
    same convention the AutoFlip parser uses, so
    ``cut_to_hold_ratio`` can score both tools uniformly.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional


# ───────────────────────── helpers ─────────────────────────


def probe_source(video_path: str) -> tuple[int, int, float]:
    """Return ``(width, height, fps)`` for the source video via ffprobe.

    Raises ``subprocess.CalledProcessError`` when ffprobe fails, so
    the caller sees a clean traceback when the video file is missing
    or unreadable.
    """
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "csv=p=0", video_path,
        ],
        capture_output=True, text=True, check=True,
    )
    parts = result.stdout.strip().split(",")
    if len(parts) != 3:
        raise ValueError(
            f"unexpected ffprobe output for {video_path!r}: {result.stdout!r}"
        )
    w_str, h_str, fps_str = parts
    if "/" in fps_str:
        num, den = fps_str.split("/")
        den_f = float(den)
        fps = float(num) / den_f if den_f else float(num)
    else:
        fps = float(fps_str)
    return int(w_str), int(h_str), float(fps)


def _aspect_to_float(aspect_ratio: str) -> float:
    """Parse ``"9:16"`` → ``0.5625``. Any parse error falls back to 9:16."""
    try:
        num, den = aspect_ratio.split(":")
        return float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        return 9.0 / 16.0


# ───────────────────────── translators ─────────────────────────


def render_plan_to_events(
    plan_dict: dict,
    src_w: int,
    src_h: int,
    fps: float,
) -> list[dict]:
    """Translate a ``RenderPlan.to_dict()`` dump into AutoFlip events.

    Each op emits one event per source frame inside its
    ``[start_sec, end_sec)`` window. The first frame of every op is
    flagged ``scene_change=True``; interior frames are False.

    ``primary_rect`` is assumed to carry 0-1 normalized source coords
    with ``x`` / ``y`` as the top-left and ``w`` / ``h`` as the box
    dimensions. The event ``crop_cx`` / ``crop_cy`` are derived from
    the box center.
    """
    events: list[dict] = []
    for op in plan_dict.get("ops", []):
        rect = op.get("primary_rect") or {}
        start = float(op.get("start_sec", 0.0))
        end = float(op.get("end_sec", 0.0))
        first_frame = int(round(start * fps))
        last_frame = int(round(end * fps))
        if last_frame <= first_frame:
            continue
        scene_change = True
        rx = float(rect.get("x", 0.0))
        ry = float(rect.get("y", 0.0))
        rw = float(rect.get("w", 9.0 / 16.0))
        rh = float(rect.get("h", 1.0))
        cx = rx + rw / 2.0
        cy = ry + rh / 2.0
        for fi in range(first_frame, last_frame):
            events.append({
                "frame": fi,
                "t": fi / fps,
                "crop_cx": cx,
                "crop_cy": cy,
                "crop_w": rw,
                "crop_h": rh,
                "scene_change": scene_change,
            })
            scene_change = False
    return events


def reframe_segments_to_events(
    segments: list,
    src_w: int,
    src_h: int,
    fps: float,
    *,
    aspect_ratio: str = "9:16",
) -> list[dict]:
    """Translate a list of ``ReframeSegment`` dicts into AutoFlip events.

    Subject coordinates in ``ReframeSegment`` are source pixels (post-
    Phase-0 convention); this helper normalizes them to 0-1. The crop
    is assumed to fill source height with width set by the target
    aspect ratio, matching the default ClipAI vertical export.

    ``segments`` may be either a raw JSON list of dicts or a list of
    ``ReframeSegment`` dataclass instances (``__dict__`` is inspected).
    """
    events: list[dict] = []
    if not segments:
        return events

    target_aspect = _aspect_to_float(aspect_ratio)
    crop_h = 1.0  # fill source height
    crop_w = (src_h * target_aspect) / src_w if src_w > 0 else target_aspect

    for seg in segments:
        # Allow both dict and dataclass.
        if isinstance(seg, dict):
            g = seg.get
        else:
            g = lambda k, default=None: getattr(seg, k, default)
        start = float(g("start", 0.0) or 0.0)
        end = float(g("end", 0.0) or 0.0)
        sx_px = float(g("subject_x", src_w / 2.0) or src_w / 2.0)
        sy_px = float(g("subject_y", src_h / 2.0) or src_h / 2.0)
        sx = sx_px / src_w if src_w else 0.5
        sy = sy_px / src_h if src_h else 0.5

        first_frame = int(round(start * fps))
        last_frame = int(round(end * fps))
        if last_frame <= first_frame:
            continue
        scene_change = True
        for fi in range(first_frame, last_frame):
            events.append({
                "frame": fi,
                "t": fi / fps,
                "crop_cx": sx,
                "crop_cy": sy,
                "crop_w": crop_w,
                "crop_h": crop_h,
                "scene_change": scene_change,
            })
            scene_change = False
    return events


# ───────────────────────── CLI ─────────────────────────


def build_timeline(
    *,
    video: str,
    render_plan_json: Optional[str] = None,
    reframe_segments_json: Optional[str] = None,
    aspect_ratio: str = "9:16",
    clipai_version: str = "week3",
) -> dict:
    """Produce an AutoFlip-shape timeline dict. Pure function; no I/O
    beyond the ffprobe call and the two input JSON reads.

    Raises ``ValueError`` if neither input source is provided.
    """
    src_w, src_h, fps = probe_source(video)
    if render_plan_json:
        plan = json.loads(Path(render_plan_json).read_text())
        events = render_plan_to_events(plan, src_w, src_h, fps)
    elif reframe_segments_json:
        segs = json.loads(Path(reframe_segments_json).read_text())
        events = reframe_segments_to_events(
            segs, src_w, src_h, fps, aspect_ratio=aspect_ratio,
        )
    else:
        raise ValueError(
            "need --render-plan-json or --reframe-segments-json",
        )
    return {
        "tool": "clipai",
        "version": clipai_version,
        "source_width": src_w,
        "source_height": src_h,
        "source_fps": fps,
        "aspect_ratio": aspect_ratio,
        "events": events,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video", required=True,
        help="Source .mp4 (needed for fps/dims via ffprobe)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--render-plan-json",
        help="Path to a ClipAI render plan JSON "
             "(RenderPlan.to_dict() dump)",
    )
    group.add_argument(
        "--reframe-segments-json",
        help="Path to a ClipAI reframe-segment JSON "
             "(list of dicts with start/end/subject_x/subject_y)",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--aspect-ratio", default="9:16")
    parser.add_argument(
        "--clipai-version", default="week3",
        help="Version string stamped into the output JSON",
    )
    args = parser.parse_args()

    timeline = build_timeline(
        video=args.video,
        render_plan_json=args.render_plan_json,
        reframe_segments_json=args.reframe_segments_json,
        aspect_ratio=args.aspect_ratio,
        clipai_version=args.clipai_version,
    )
    Path(args.output).write_text(
        json.dumps(timeline, separators=(",", ":")),
    )
    print(
        f"wrote {len(timeline['events'])} events to {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
