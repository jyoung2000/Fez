"""Real MediaPipe AutoFlip reference generator (Task 5A, Path A).

Mirrors the API of :mod:`backend.scripts.run_naive_baseline` so the
existing bench loader (:func:`load_autoflip_timeline`) reads the
output without any changes. The naive baseline writes
``tool="naive_baseline"``; this script writes ``tool="autoflip_real"``
(or ``"autoflip_prebuilt"`` when the operator supplies the prebuilt
binary). The bench rollup uses that tag to label whether the
"AutoFlip" column actually came from real AutoFlip or a strawman.

Usage::

    python -m backend.scripts.run_autoflip_reference \\
        --manifest tests/real_content/manifest.json \\
        --output-dir tests/autoflip_reference_outputs/

Pipeline per clip:
    1. Probe the source MP4 via ffprobe.
    2. ``docker compose run --rm autoflip --input_video_path=/data/<slug>.mp4
       --output_video_path=/output/<slug>_autoflip.mp4 --aspect_ratio=9:16
       --output_metadata_path=/output/<slug>_metadata.pbtxt``.
    3. Parse the pbtxt metadata into a sequence of crop windows.
    4. Translate to AutoFlip-event shape (one event per source frame).
    5. Write ``<slug>.json`` next to the naive baseline output.

The script never raises on a single-clip failure — failures are
captured per-slug in the ``skipped`` summary so a flaky clip can't
block the whole batch.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "clipai/autoflip:local"
DEFAULT_TOOL_LABEL = "autoflip_real"

# Per-frame metadata block in AutoFlip's text-format pbtxt output.
# AutoFlip emits one ExternalRenderFrame message per output frame; we
# parse via a tolerant regex rather than dragging in a protobuf
# descriptor at runtime. The fields we care about are:
#   crop_from_location { x: <float> y: <float> width: <float> height: <float> }
#   timestamp_us: <int>
# Some MediaPipe builds nest the crop window under a different parent
# message name; the helpers below handle multiple shapes.
_FLOAT = r"-?\d+(?:\.\d+)?"
_BLOCK_RX = re.compile(
    r"\{\s*"
    r"(?:[\w_]+\s*:\s*[^\n}]+\s*)*?"
    r"x\s*:\s*(?P<x>%s).*?"
    r"y\s*:\s*(?P<y>%s).*?"
    r"width\s*:\s*(?P<w>%s).*?"
    r"height\s*:\s*(?P<h>%s).*?"
    r"\}"
    % (_FLOAT, _FLOAT, _FLOAT, _FLOAT),
    re.DOTALL,
)
_TIMESTAMP_RX = re.compile(r"timestamp_us\s*:\s*(\d+)")


def _probe_metadata(video_path: str) -> dict:
    """Return ``{duration, width, height, fps}`` via ffprobe.

    Identical to :func:`run_naive_baseline._probe_metadata` so the
    output JSON's ``metadata`` block matches Path B's schema exactly.
    """
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


def _docker_run(
    *,
    image: str,
    source_dir: Path,
    output_dir: Path,
    slug: str,
    ext: str,
    aspect_ratio: str,
) -> int:
    """Invoke the AutoFlip docker image. Returns the exit code.

    Mounts ``source_dir`` read-only at ``/data`` and ``output_dir``
    read-write at ``/output`` to mirror the docker-compose service
    layout. Falls back to a non-compose ``docker run`` so the script
    works in environments where docker-compose is not on PATH.
    """
    if not shutil.which("docker"):
        raise RuntimeError("docker not on PATH")
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{source_dir}:/data:ro",
        "-v", f"{output_dir}:/output",
        image,
        f"--input_video_path=/data/{slug}.{ext}",
        f"--output_video_path=/output/{slug}_autoflip.mp4",
        f"--aspect_ratio={aspect_ratio}",
        f"--output_metadata_path=/output/{slug}_metadata.pbtxt",
    ]
    logger.info("[autoflip:%s] %s", slug, " ".join(cmd))
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=60 * 30,
    )
    if proc.returncode != 0:
        logger.warning(
            "[autoflip:%s] exit %d — stderr: %s",
            slug, proc.returncode,
            proc.stderr.decode(errors="replace")[:400],
        )
    return proc.returncode


def parse_autoflip_metadata(
    pbtxt_text: str,
    *,
    fps: float,
    duration: float,
) -> list[dict]:
    """Parse AutoFlip's text-format metadata into AutoFlip-event dicts.

    Each match is a crop window block ``{x, y, width, height}`` in 0-1
    normalised source-frame coordinates. We pair it with a timestamp
    block when present (``timestamp_us``) and otherwise fall back to a
    uniform fps-derived spacing so the events still cover the clip
    duration.

    Returned shape matches what :func:`load_autoflip_timeline` expects
    (one dict per output frame): ``t``, ``crop_cx``, ``crop_cy``,
    ``crop_w``, ``crop_h``, ``scene_change``.
    """
    blocks = list(_BLOCK_RX.finditer(pbtxt_text))
    if not blocks:
        return []

    # Pull timestamps out in document order. Each render_frame block
    # in AutoFlip's pbtxt output owns one timestamp_us field placed
    # AFTER its inner rect/crop_from_location subblock. Pair each
    # block with the nearest timestamp_us whose offset falls between
    # this block's end and the next block's start.
    ts_iter = list(_TIMESTAMP_RX.finditer(pbtxt_text))
    events: list[dict] = []
    for i, m in enumerate(blocks):
        x = float(m.group("x"))
        y = float(m.group("y"))
        w = float(m.group("w"))
        h = float(m.group("h"))
        # AutoFlip's crop window is the (x, y) top-left corner. The
        # bench expects centre coordinates.
        cx = x + w / 2.0
        cy = y + h / 2.0

        block_end = m.end()
        next_block_start = (
            blocks[i + 1].start() if i + 1 < len(blocks) else len(pbtxt_text)
        )
        ts_us = None
        for tm in ts_iter:
            if block_end <= tm.start() < next_block_start:
                ts_us = int(tm.group(1))
                break
        if ts_us is not None:
            t = ts_us / 1_000_000.0
        elif fps > 0:
            t = i / fps
        else:
            t = 0.0

        events.append({
            "t": round(t, 4),
            "crop_cx": round(max(0.0, min(1.0, cx)), 4),
            "crop_cy": round(max(0.0, min(1.0, cy)), 4),
            "crop_w": round(max(0.0, min(1.0, w)), 4),
            "crop_h": round(max(0.0, min(1.0, h)), 4),
            "scene_change": False,
        })

    # If the parsed timeline is shorter than the source duration,
    # extend by repeating the last crop window — AutoFlip occasionally
    # emits one block per scene rather than per frame in older builds.
    if events and fps > 0 and duration > 0:
        target_n = max(int(duration * fps), len(events))
        if target_n > len(events):
            last = dict(events[-1])
            for i in range(len(events), target_n):
                ev = dict(last)
                ev["t"] = round(i / fps, 4)
                events.append(ev)

    return events


def autoflip_events_for_clip(
    video_path: str,
    output_dir: Path,
    *,
    slug: str,
    ext: str = "mp4",
    aspect_ratio: str = "9:16",
    image: str = DEFAULT_IMAGE,
    tool_label: str = DEFAULT_TOOL_LABEL,
    docker_runner: Optional[Callable[..., int]] = None,
) -> dict:
    """Run AutoFlip on one clip and return the JSON payload.

    ``docker_runner`` is injectable for tests; defaults to
    :func:`_docker_run`. The runner is expected to write the
    ``<slug>_metadata.pbtxt`` under ``output_dir``; the parser then
    consumes it.
    """
    meta = _probe_metadata(video_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = docker_runner or _docker_run
    rc = runner(
        image=image,
        source_dir=Path(video_path).parent,
        output_dir=output_dir,
        slug=slug,
        ext=ext,
        aspect_ratio=aspect_ratio,
    )
    if rc != 0:
        raise RuntimeError(
            f"autoflip docker exited with code {rc} for slug={slug}"
        )

    metadata_path = output_dir / f"{slug}_metadata.pbtxt"
    if not metadata_path.is_file():
        raise RuntimeError(
            f"autoflip did not produce metadata at {metadata_path}"
        )

    pbtxt = metadata_path.read_text(errors="replace")
    events = parse_autoflip_metadata(
        pbtxt, fps=meta["fps"], duration=meta["duration"],
    )
    if not events:
        raise RuntimeError(
            f"autoflip metadata at {metadata_path} parsed zero events"
        )

    return {
        "tool": tool_label,
        "metadata": {
            **meta,
            "tool": tool_label,
            "image": image,
            "aspect_ratio": aspect_ratio,
        },
        "events": events,
    }


def run_for_manifest(
    manifest_path: Path,
    output_dir: Path,
    *,
    real_content_dir: Optional[Path] = None,
    only_slugs: Optional[set[str]] = None,
    image: str = DEFAULT_IMAGE,
    tool_label: str = DEFAULT_TOOL_LABEL,
    aspect_ratio: str = "9:16",
    docker_runner: Optional[Callable[..., int]] = None,
) -> dict:
    """Generate AutoFlip JSON for every clip in the manifest.

    Returns ``{"generated": [...], "skipped": [(slug, reason), ...]}``.
    Same shape as :func:`run_naive_baseline.run_for_manifest` so callers
    that consume one can swap in the other transparently.
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
            ref = autoflip_events_for_clip(
                str(video_path),
                output_dir,
                slug=slug,
                ext=ext,
                aspect_ratio=aspect_ratio,
                image=image,
                tool_label=tool_label,
                docker_runner=docker_runner,
            )
        except Exception as exc:
            skipped.append((slug, f"autoflip failed: {exc!s}"))
            logger.warning("[autoflip:%s] skipped — %s", slug, exc)
            continue
        out_path = output_dir / f"{slug}.json"
        out_path.write_text(json.dumps(ref))
        generated.append(slug)
        logger.info(
            "[autoflip:%s] wrote %s (%d events, tool=%s)",
            slug, out_path, len(ref["events"]), tool_label,
        )

    return {"generated": generated, "skipped": skipped}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--output-dir", default="tests/autoflip_reference_outputs/",
    )
    parser.add_argument("--real-content-dir", default=None)
    parser.add_argument("--filter-slugs", default=None)
    parser.add_argument(
        "--image", default=DEFAULT_IMAGE,
        help="Docker image tag for AutoFlip (default: %(default)s)",
    )
    parser.add_argument(
        "--aspect-ratio", default="9:16",
        help="Output aspect ratio passed to AutoFlip (default: %(default)s)",
    )
    parser.add_argument(
        "--tool-label", default=DEFAULT_TOOL_LABEL,
        choices=["autoflip_real", "autoflip_prebuilt"],
        help=(
            "Tag written into each output JSON's 'tool' field. Use "
            "'autoflip_prebuilt' when the image was built from "
            "Dockerfile.prebuilt (Path A2)."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    only = None
    if args.filter_slugs:
        only = {s.strip() for s in args.filter_slugs.split(",") if s.strip()}

    summary = run_for_manifest(
        Path(args.manifest),
        Path(args.output_dir),
        real_content_dir=Path(args.real_content_dir) if args.real_content_dir else None,
        only_slugs=only,
        image=args.image,
        tool_label=args.tool_label,
        aspect_ratio=args.aspect_ratio,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["generated"] else 1


if __name__ == "__main__":
    sys.exit(main())
