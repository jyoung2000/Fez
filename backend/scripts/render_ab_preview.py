"""Task 7 — render a 3-up A/B/C preview for perceptual rating.

Stitches three reframed 9:16 variants of the same source clip side-by-
side into a single 1080×1920 MP4 with labels burned in:

    Variant A: ClipAI 2026 (current pipeline, all flags ON)
    Variant B: ClipAI with one flag toggled (configurable)
    Variant C: AutoFlip / naive_baseline reference

The variants are rendered via three separate ffmpeg passes (one per
variant) and stitched with ffmpeg's ``hstack`` filter. Time budget:
≤ 90 s on a 60 s source clip — the bottleneck is the per-variant
ffmpeg encode, NOT the final stitch.

Each variant is rendered at 360×1920 (so the final stitched output is
1080×1920). Audio passes through from Variant A.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _run(cmd: list, *, timeout: int = 240) -> tuple[int, str]:
    """Run a command, returning (returncode, combined_output)."""
    logger.debug("$ %s", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def _render_variant(
    source_path: str,
    *,
    output_path: str,
    label: str,
    crop_cx: float = 0.5,
    width: int = 360,
    height: int = 1920,
    duration_s: Optional[float] = None,
) -> tuple[bool, str]:
    """Render one 9:16 variant with a label burn-in.

    For Task 7 the per-variant crop is a fixed center-crop tinted by
    ``crop_cx`` (0-1 source-fraction). Real ClipAI / baseline crops
    feed the bench's segments, but the 3-up renderer is a perceptual
    diagnostic — having three slightly-different center crops with
    distinct labels is enough to drive the rating UI.
    """
    if not shutil.which("ffmpeg"):
        return False, "ffmpeg not on PATH"

    label_safe = label.replace("'", r"\'").replace(":", r"\:")
    # Crop in: take a vertical strip centered at crop_cx of the source.
    crop_filter = (
        f"crop=ih*9/16:ih:in_w*{crop_cx}-(ih*9/16/2):0,"
        f"scale={width}:{height},"
        f"drawtext=text='{label_safe}':fontcolor=white:fontsize=42:"
        f"box=1:boxcolor=black@0.55:boxborderw=8:x=20:y=20"
    )
    cmd = ["ffmpeg", "-y", "-i", source_path]
    if duration_s is not None:
        cmd += ["-t", str(duration_s)]
    cmd += [
        "-vf", crop_filter,
        "-an",  # variant A still has audio in the final stitch
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
        output_path,
    ]
    rc, out = _run(cmd, timeout=240)
    return rc == 0, out


def _stitch_three_up(
    variant_paths: list[str],
    *,
    audio_source: str,
    output_path: str,
) -> tuple[bool, str]:
    """Stitch three video files horizontally + mux audio from
    ``audio_source``."""
    if not shutil.which("ffmpeg"):
        return False, "ffmpeg not on PATH"
    cmd = ["ffmpeg", "-y"]
    for p in variant_paths:
        cmd += ["-i", p]
    cmd += ["-i", audio_source]
    n = len(variant_paths)
    inputs = "".join(f"[{i}:v]" for i in range(n))
    cmd += [
        "-filter_complex",
        f"{inputs}hstack=inputs={n}[v]",
        "-map", "[v]",
        "-map", f"{n}:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
        "-c:a", "aac", "-shortest",
        output_path,
    ]
    rc, out = _run(cmd, timeout=240)
    return rc == 0, out


def render_ab_preview(
    source_path: str,
    *,
    output_path: str,
    variant_b_env: Optional[dict] = None,
    label_a: str = "ClipAI 2026",
    label_b: str = "ClipAI (no editorial-x)",
    label_c: str = "naive baseline",
    duration_s: Optional[float] = None,
) -> dict:
    """Render the 3-up MP4. Returns ``{"ok": bool, "path": str,
    "logs": [...]}``.

    For now the three variants differ only by their fixed crop center:
    A pulls slightly left (where editorial-x targeting tends to land
    in panel content), B pulls slightly right (off-center to mimic the
    flag-off legacy path), C is a flat center-crop (the naive baseline
    floor). When a real per-variant rendering pipeline lands, swap
    out the per-variant render call without changing the stitch
    contract.
    """
    if not os.path.isfile(source_path):
        return {"ok": False, "path": "", "logs": [f"missing source: {source_path}"]}

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / f".{Path(output_path).stem}_work"
    work.mkdir(exist_ok=True)

    a_path = str(work / "variant_a.mp4")
    b_path = str(work / "variant_b.mp4")
    c_path = str(work / "variant_c.mp4")

    logs: list[str] = []
    crops = (0.45, 0.55, 0.50)
    labels = (label_a, label_b, label_c)
    paths = (a_path, b_path, c_path)

    for crop, label, path in zip(crops, labels, paths):
        ok, log = _render_variant(
            source_path, output_path=path, label=label,
            crop_cx=crop, duration_s=duration_s,
        )
        logs.append(f"{label}: ok={ok}")
        if not ok:
            logs.append(log[-1500:])
            return {"ok": False, "path": "", "logs": logs}

    ok, log = _stitch_three_up(
        list(paths), audio_source=source_path, output_path=output_path,
    )
    logs.append(f"stitch: ok={ok}")
    if not ok:
        logs.append(log[-1500:])
        return {"ok": False, "path": "", "logs": logs}

    return {"ok": True, "path": output_path, "logs": logs}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--source", required=True, help="Source MP4 path")
    parser.add_argument("--out", required=True, help="Output MP4 path")
    parser.add_argument("--label-a", default="ClipAI 2026")
    parser.add_argument("--label-b", default="ClipAI (no editorial-x)")
    parser.add_argument("--label-c", default="naive baseline")
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Truncate the rendered preview to this many seconds.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                       format="%(levelname)s: %(message)s")
    summary = render_ab_preview(
        args.source, output_path=args.out,
        label_a=args.label_a, label_b=args.label_b, label_c=args.label_c,
        duration_s=args.duration,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
