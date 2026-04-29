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
from typing import Optional, Sequence


# ───────────────────────── helpers ─────────────────────────


def _smoothstep_ease(
    t: float, t0: float, t1: float, x0: float, x1: float,
) -> float:
    """Smoothstep blend from ``x0`` at ``t == t0`` to ``x1`` at ``t == t1``.

    Outside ``[t0, t1]`` the function clamps to ``x0`` / ``x1``. The
    interior follows ``s = u^2 * (3 - 2u)`` (cubic Hermite, zero
    derivative at both ends), the same easing curve the clip exporter
    uses for motivated pans so the diagnostic preview matches the
    eventual render.
    """
    if t1 <= t0:
        return x1
    if t <= t0:
        return x0
    if t >= t1:
        return x1
    u = (t - t0) / (t1 - t0)
    s = u * u * (3.0 - 2.0 * u)
    return x0 + (x1 - x0) * s


def _seg_get(seg, key, default=None):
    """Read ``key`` from a dict OR a dataclass-like instance."""
    if isinstance(seg, dict):
        return seg.get(key, default)
    return getattr(seg, key, default)


def build_camera_path_keypoints(
    segments: Sequence,
    *,
    ease_sample_dt: float = 1.0 / 30.0,
) -> list[tuple[float, float]]:
    """Build a flat ``(t_abs_sec, x_pixel_center)`` keypoint stream.

    The renderer (``sota_render_preview._build_x_expression``) and the
    events translator (``reframe_segments_to_events``) both consume
    this stream so the diagnostic preview, the event-stream parity
    metrics, and the production-export camera path stay in lock-step.

    Per segment:

      * If ``motion_path`` is populated, emit one keypoint per entry.
        Each entry is treated as ``(t_abs_sec, x_pixel, [y_pixel])``
        — the same units as ``subject_x`` (matches the L1 solver
        output convention; see ``backend/services/l1_camera_path.py``).
      * Otherwise emit ``(start, subject_x)`` and ``(end, subject_x)``
        so a linear lerp degenerates to a hold.

    At each segment boundary with ``ease_in_ms > 0``, prepend a
    smoothstep ramp from the previous segment's exit x to the current
    segment's first keypoint x, sampled at ``ease_sample_dt`` so the
    downstream linear-lerp consumers approximate the cubic curve
    closely. The ease replaces the segment's first ``ease_in_ms`` of
    its own keypoints.

    The output is sorted, monotonically non-decreasing in t, and
    contains no zero-width adjacent duplicates. Returns an empty list
    when ``segments`` is empty or contains no usable entries.
    """
    if not segments:
        return []

    # Sort by start so ease blending walks the timeline in order.
    parts = sorted(
        list(segments),
        key=lambda s: float(_seg_get(s, "start", 0.0) or 0.0),
    )

    flat: list[tuple[float, float]] = []
    prev_exit_x: Optional[float] = None
    prev_exit_t: Optional[float] = None

    for seg in parts:
        start = float(_seg_get(seg, "start", 0.0) or 0.0)
        end = float(_seg_get(seg, "end", 0.0) or 0.0)
        if end <= start:
            continue
        subject_x = float(_seg_get(seg, "subject_x", 0.0) or 0.0)
        ease_ms = int(_seg_get(seg, "ease_in_ms", 0) or 0)

        # Build raw keypoint sequence for THIS segment (no ease yet).
        raw_path = _seg_get(seg, "motion_path", None) or None
        seg_kps: list[tuple[float, float]] = []
        if raw_path:
            for entry in raw_path:
                try:
                    t_e = float(entry[0])
                    x_e = float(entry[1])
                except (TypeError, ValueError, IndexError):
                    continue
                # Clamp into the segment window so a stray entry from a
                # path-slicing bug can't pull the camera off-screen.
                if t_e < start - 1e-6 or t_e > end + 1e-6:
                    continue
                t_e = max(start, min(end, t_e))
                seg_kps.append((t_e, x_e))
            seg_kps.sort(key=lambda p: p[0])
            # Make sure the segment is anchored at its start / end so
            # the lerp covers the full span even when the path is
            # short.
            if not seg_kps or seg_kps[0][0] > start + 1e-6:
                seg_kps.insert(0, (start, seg_kps[0][1] if seg_kps else subject_x))
            if seg_kps[-1][0] < end - 1e-6:
                seg_kps.append((end, seg_kps[-1][1]))
        else:
            seg_kps = [(start, subject_x), (end, subject_x)]

        # Apply ease if there's a previous exit x and ease_in_ms > 0.
        if (
            prev_exit_x is not None
            and ease_ms > 0
            and seg_kps
        ):
            ease_dur = ease_ms / 1000.0
            ease_end_t = min(start + ease_dur, end)
            # Target x at the end of the ease = the segment's x at
            # ease_end_t, interpolated along its own raw keypoints.
            target_x = _interpolate_keypoints(seg_kps, ease_end_t)
            ramp: list[tuple[float, float]] = []
            t = start
            # Start the ramp slightly past the prev exit time when
            # they coincide so the keypoints stay strictly increasing.
            if prev_exit_t is not None and t <= prev_exit_t:
                t = prev_exit_t + max(1e-6, ease_sample_dt * 0.0)
            sample_dt = max(1e-3, ease_sample_dt)
            while t < ease_end_t:
                x = _smoothstep_ease(
                    t, start, ease_end_t, prev_exit_x, target_x,
                )
                ramp.append((t, x))
                t += sample_dt
            ramp.append((ease_end_t, target_x))
            # Drop the segment's own keypoints that fall inside the
            # ease window — the ramp owns that interval.
            tail = [kp for kp in seg_kps if kp[0] > ease_end_t + 1e-6]
            seg_kps = ramp + tail

        # Splice into the flat stream:
        #   * Drop a kp that exactly matches the previous one
        #     (zero-width zero-displacement duplicate).
        #   * Keep two kps at the SAME time with DIFFERENT x — that's
        #     a hard step at the segment boundary, exactly what the
        #     legacy back-compat path needs. The half-open gating in
        #     the consumers makes the zero-width interval contribute
        #     nothing, so the step lands cleanly.
        for kp in seg_kps:
            if (
                flat
                and abs(flat[-1][0] - kp[0]) < 1e-6
                and abs(flat[-1][1] - kp[1]) < 1e-6
            ):
                continue
            flat.append(kp)

        prev_exit_t = flat[-1][0] if flat else end
        # Use the segment's TRUE end x (not the post-splice flat tail,
        # which may have been bumped backwards by an ease ramp from
        # the next iteration). For the simple case both agree.
        prev_exit_x = flat[-1][1] if flat else subject_x

    return flat


def _interpolate_keypoints(
    keypoints: Sequence[tuple[float, float]],
    t: float,
    fallback: Optional[float] = None,
) -> float:
    """Linear-lerp evaluator for the keypoint stream produced by
    ``build_camera_path_keypoints``.

    Each consecutive pair ``(t0, x0), (t1, x1)`` covers the half-open
    interval ``[t0, t1)``; zero-width pairs (``t1 == t0``) are
    skipped, so a hard step at a segment boundary correctly picks up
    the LATER x at the boundary timestamp. The very last keypoint
    holds for any ``t >= t_last``.
    """
    if not keypoints:
        return float(fallback) if fallback is not None else 0.0
    if t < keypoints[0][0]:
        return float(keypoints[0][1])
    if t >= keypoints[-1][0]:
        return float(keypoints[-1][1])
    # Linear scan — keypoint counts are small (10s-100s per clip) so
    # the constant factor on a binary search isn't worth it.
    for i in range(len(keypoints) - 1):
        t0, x0 = keypoints[i]
        t1, x1 = keypoints[i + 1]
        if t1 <= t0:
            # Zero-width step; the next iteration starts at this t1
            # (== t0), so the post-step x wins at the boundary.
            continue
        if t0 <= t < t1:
            u = (t - t0) / (t1 - t0)
            return float(x0 + (x1 - x0) * u)
    return float(keypoints[-1][1])


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

    The per-frame ``crop_cx`` is sampled from the camera-path keypoint
    stream (``build_camera_path_keypoints``) so motion_path tracking
    and smoothstep ease at motivated cuts both show up in the event
    stream — and therefore in the ``max_acceleration`` / ``max_jerk``
    parity metrics. Segments without a ``motion_path`` collapse to a
    held subject_x exactly as before, so cached ``segments.json`` blobs
    that pre-date the motion_path serialization keep their old
    behavior.

    ``segments`` may be either a raw JSON list of dicts or a list of
    ``ReframeSegment`` dataclass instances (``__dict__`` is inspected).
    """
    events: list[dict] = []
    if not segments:
        return events

    target_aspect = _aspect_to_float(aspect_ratio)
    crop_h = 1.0  # fill source height
    crop_w = (src_h * target_aspect) / src_w if src_w > 0 else target_aspect

    keypoints = build_camera_path_keypoints(segments)

    for seg in segments:
        g = (seg.get if isinstance(seg, dict)
             else (lambda k, default=None: getattr(seg, k, default)))
        start = float(g("start", 0.0) or 0.0)
        end = float(g("end", 0.0) or 0.0)
        sx_px_default = float(g("subject_x", src_w / 2.0) or src_w / 2.0)
        sy_px = float(g("subject_y", src_h / 2.0) or src_h / 2.0)
        sy = sy_px / src_h if src_h else 0.5

        first_frame = int(round(start * fps))
        last_frame = int(round(end * fps))
        if last_frame <= first_frame:
            continue
        scene_change = True
        for fi in range(first_frame, last_frame):
            t = fi / fps if fps else 0.0
            sx_px = _interpolate_keypoints(
                keypoints, t, fallback=sx_px_default,
            )
            sx = sx_px / src_w if src_w else 0.5
            events.append({
                "frame": fi,
                "t": t,
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
