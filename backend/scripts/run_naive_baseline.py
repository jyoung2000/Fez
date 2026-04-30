"""Naive center-crop baseline reference generator (Task 5, Path B).

Produces AutoFlip-shape JSON references for each clip in a manifest by
running the simplest possible "follow the largest face" tracker. The
output lives where the bench expects it
(``tests/autoflip_reference_outputs/<slug>.json``) so the existing
:func:`load_autoflip_timeline` reads it without changes.

This is a deliberately weaker comparator than MediaPipe AutoFlip — it
gives the bench a side-by-side baseline that's clearly inferior, so
"ClipAI better than naive" is the floor we have to clear before any
SOTA claims. Mark loudly in the rollup that this is naive_baseline,
not real AutoFlip.

Usage:
    python -m backend.scripts.run_naive_baseline \\
        --manifest tests/real_content/manifest.json \\
        --output-dir tests/autoflip_reference_outputs/

Algorithm:
    1. Probe video duration via ffprobe.
    2. Sample faces at 1 fps via the same FaceDetector the pipeline
       uses.
    3. Take the largest face's center as the per-second crop x.
    4. Hold the previous crop center when no face is detected.
    5. Smooth with a box filter (default 0.5 s window).
    6. Emit one event per FPS frame, all sharing the smoothed center.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _probe_metadata(video_path: str) -> dict:
    """Return ``{duration, width, height, fps}`` via ffprobe."""
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe not on PATH")
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate:format=duration",
            "-of", "json",
            video_path,
        ],
        timeout=15,
    )
    info = json.loads(out.decode())
    streams = info.get("streams") or [{}]
    fmt = info.get("format") or {}
    s = streams[0]
    fps_num, fps_den = (s.get("r_frame_rate") or "30/1").split("/")
    try:
        fps = float(fps_num) / max(float(fps_den), 1.0)
    except (TypeError, ValueError, ZeroDivisionError):
        fps = 30.0
    return {
        "duration": float(fmt.get("duration") or 0.0),
        "width": int(s.get("width") or 1920),
        "height": int(s.get("height") or 1080),
        "fps": fps,
    }


def _sparse_face_centers(
    video_path: str, duration: float, sample_count: int,
) -> list[tuple[float, Optional[float]]]:
    """Sample face centers at sample_count evenly-spaced frames.

    Returns ``[(t, x_pct or None), ...]``. None when no face was found
    at that sample. Defensive: any detector failure returns None for
    that sample, never raises.
    """
    out: list[tuple[float, Optional[float]]] = []
    timestamps = [
        duration * (i + 0.5) / max(sample_count, 1)
        for i in range(sample_count)
    ]
    try:
        from backend.services.face_detector import FaceDetector
    except Exception as exc:
        logger.warning(
            "naive_baseline: face detector unavailable (%s) — "
            "emitting center-only baseline", exc,
        )
        return [(t, None) for t in timestamps]

    detector = FaceDetector()
    for t in timestamps:
        try:
            frame_faces = detector.detect_at_timestamp(video_path, t)
            if not frame_faces or not getattr(frame_faces, "faces", []):
                out.append((t, None))
                continue
            largest = max(
                frame_faces.faces,
                key=lambda f: float(getattr(f, "width", 0))
                * float(getattr(f, "height", 0)),
            )
            x_pct = float(getattr(largest, "nose_x", 50.0))
            out.append((t, x_pct))
        except Exception as exc:
            logger.debug("face detect failed at t=%.2fs: %s", t, exc)
            out.append((t, None))
    return out


def _box_smooth(values: list[float], window_s: float, dt: float) -> list[float]:
    """Centered box filter; window in seconds → integer span in samples."""
    if not values:
        return values
    span = max(int(window_s / max(dt, 0.001)), 1)
    half = span // 2
    n = len(values)
    out: list[float] = []
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        out.append(sum(values[a:b]) / (b - a))
    return out


def _hold_through_gaps(
    samples: list[tuple[float, Optional[float]]],
    default_x: float,
) -> list[tuple[float, float]]:
    """Replace None centers with the previous sample's center.

    No previous sample yet → ``default_x``.
    """
    last = default_x
    out: list[tuple[float, float]] = []
    for t, x in samples:
        if x is not None:
            last = x
        out.append((t, last))
    return out


def naive_baseline_events(
    video_path: str,
    *,
    sample_fps: float = 1.0,
    smooth_window_s: float = 0.5,
    aspect_ratio: str = "9:16",
) -> dict:
    """Compute the naive baseline reference for one clip.

    Returns a dict in the shape :func:`load_autoflip_timeline` expects:

        {"events": [...], "tool": "naive_baseline", "metadata": {...}}

    Each event has ``t``, ``crop_cx``, ``crop_cy``, ``crop_w``,
    ``crop_h``, ``scene_change``.
    """
    meta = _probe_metadata(video_path)
    duration = max(meta["duration"], 0.001)
    fps = meta["fps"]
    crop_aspect = 9.0 / 16.0
    if aspect_ratio == "16:9":
        crop_aspect = 16.0 / 9.0

    sample_count = max(int(duration * sample_fps), 1)
    samples = _sparse_face_centers(video_path, duration, sample_count)
    held = _hold_through_gaps(samples, default_x=50.0)
    xs_pct = _box_smooth(
        [x for _t, x in held],
        window_s=smooth_window_s,
        dt=duration / max(sample_count, 1),
    )

    # Render one event per output frame at the source fps.
    n_out = max(int(duration * fps), 1)
    events: list[dict] = []
    for i in range(n_out):
        t = i / fps
        # Find the sample-window x for this output time.
        sample_idx = min(int(t * sample_fps), len(xs_pct) - 1)
        x_norm = xs_pct[sample_idx] / 100.0
        events.append({
            "t": round(t, 4),
            "crop_cx": round(x_norm, 4),
            "crop_cy": 0.5,
            "crop_w": round(crop_aspect / (16.0 / 9.0), 4),
            "crop_h": 1.0,
            "scene_change": False,
        })

    return {
        "tool": "naive_baseline",
        "metadata": meta,
        "events": events,
    }


def run_for_manifest(
    manifest_path: Path,
    output_dir: Path,
    *,
    real_content_dir: Optional[Path] = None,
    only_slugs: Optional[set[str]] = None,
) -> dict:
    """Generate baseline JSON for every clip in the manifest.

    Returns ``{"generated": [...], "skipped": [...]}``.
    """
    manifest = json.loads(manifest_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    real_root = real_content_dir or Path(os.environ.get(
        "CLIPAI_REAL_CONTENT_CACHE", "/var/cache/clipai/real_content",
    ))

    generated: list[str] = []
    skipped: list[tuple[str, str]] = []

    for clip in manifest.get("clips", []):
        slug = clip.get("slug") or ""
        if only_slugs and slug not in only_slugs:
            continue
        ext = clip.get("ext") or "mp4"
        video_path = real_root / f"{slug}.{ext}"
        if not video_path.is_file():
            skipped.append((slug, f"video missing at {video_path}"))
            continue
        try:
            ref = naive_baseline_events(str(video_path))
        except Exception as exc:
            skipped.append((slug, f"baseline failed: {exc!s}"))
            continue
        out_path = output_dir / f"{slug}.json"
        out_path.write_text(json.dumps(ref))
        generated.append(slug)
        logger.info("[naive_baseline] wrote %s (%d events)", out_path, len(ref["events"]))

    return {"generated": generated, "skipped": skipped}


def generate_single_clip_reference(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    slug: Optional[str] = None,
) -> Optional[Path]:
    """Generate a naive_baseline reference JSON for one clip on disk.

    Used by both the ``--single-clip`` CLI mode and the
    ``/sota-clip-bench`` SSE pre-phase so single-clip runs always have
    a comparator without requiring the operator to run
    ``make naive-references`` out-of-band.

    Returns the output path on success or ``None`` on failure.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        logger.error("naive_baseline: source not found: %s", video_path)
        return None
    out_slug = slug or video_path.stem
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        ref = naive_baseline_events(str(video_path))
    except Exception as exc:
        logger.error("naive_baseline: failed for %s: %s", video_path, exc)
        return None
    out_path = out_dir / f"{out_slug}.json"
    out_path.write_text(json.dumps(ref))
    logger.info(
        "[naive_baseline] wrote %s (%d events)",
        out_path, len(ref["events"]),
    )
    return out_path


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    # ``--manifest`` is required for the manifest-iteration path but
    # not for ``--single-clip``; we enforce mutual-exclusivity manually
    # below so the help text stays simple.
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--output-dir", default="tests/autoflip_reference_outputs/",
    )
    parser.add_argument("--real-content-dir", default=None)
    parser.add_argument("--filter-slugs", default=None)
    parser.add_argument(
        "--single-clip", default=None,
        help=(
            "Generate a reference for one clip at this path "
            "(skips manifest iteration). Pair with --slug to override "
            "the output filename stem."
        ),
    )
    parser.add_argument(
        "--slug", default=None,
        help=(
            "Slug to use when --single-clip is set. Defaults to the "
            "source filename stem."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.single_clip:
        if args.manifest:
            print(
                "!! --single-clip and --manifest are mutually exclusive",
                file=sys.stderr,
            )
            return 2
        out_path = generate_single_clip_reference(
            args.single_clip, args.output_dir, slug=args.slug,
        )
        if out_path is None:
            return 1
        print(json.dumps({"single_clip": str(out_path)}, indent=2))
        return 0

    if not args.manifest:
        print(
            "!! one of --manifest or --single-clip is required",
            file=sys.stderr,
        )
        return 2

    only = None
    if args.filter_slugs:
        only = {s.strip() for s in args.filter_slugs.split(",") if s.strip()}

    summary = run_for_manifest(
        Path(args.manifest),
        Path(args.output_dir),
        real_content_dir=Path(args.real_content_dir) if args.real_content_dir else None,
        only_slugs=only,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["generated"] else 1


if __name__ == "__main__":
    sys.exit(main())
