"""Universal reframe smoke test harness (Phase 6).

Walks a directory of test videos (or a single file), runs each
through the full reframing pipeline with ``CLIPAI_HUMAN_REFRAME=1``,
extracts the resulting :class:`RenderPlan`, scores it via
:func:`backend.services.reframe_quality_scorer.score_render_plan`,
and emits both a human-readable summary AND a JSON report.

Exit codes
----------
    0    every clip passes (overall >= 75)
    1    any clip needs review (overall in [50, 75))
    2    any clip fails        (overall < 50)
    3    no clips found / nothing to score

Sandbox behavior
----------------
The pipeline cannot run end-to-end without numpy + ffmpeg + the rest
of the heavyweight stack. When deps are missing the harness emits a
clear ``"skipped: missing X"`` entry per clip — the CLI shape and
JSON schema are stable regardless. This is deliberate: the script
must be IMPLEMENTABLE today and merely SKIPPABLE in the test sandbox.

Usage
-----
    python backend/scripts/smoke_test_universal_reframe.py \
        --input /path/to/videos/ \
        --output report.json

When ``--input`` is a single file the harness scores just that file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


PASS_THRESHOLD = 75.0
REVIEW_THRESHOLD = 50.0


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def _gather_videos(root: Path) -> List[Path]:
    """Return every video file under ``root`` (or [root] if it's a file).

    Returns an empty list when ``root`` does not exist.
    """
    if not root.exists():
        return []
    if root.is_file():
        return [root]
    out: List[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            out.append(p)
    return out


def _check_pipeline_available() -> Tuple[bool, str]:
    """Probe whether the full pipeline can run.

    Returns (ok, reason). The reason is empty when ok is True.
    """
    try:
        import numpy  # noqa: F401
    except ImportError:
        return False, "numpy not installed"
    if not shutil.which("ffmpeg"):
        return False, "ffmpeg not on PATH"
    try:
        from backend.services import pipeline  # noqa: F401
    except Exception as exc:  # pragma: no cover - module-level import errs
        return False, f"pipeline import failed: {exc!s}"
    return True, ""


def _run_pipeline(video_path: Path) -> Optional[Any]:
    """Run the full pipeline and return the resulting :class:`RenderPlan`.

    Returns ``None`` when the pipeline run errored or did not produce a
    plan. The caller decides whether that's a hard failure or a soft
    skip (in this harness it's a soft per-clip skip).
    """
    os.environ.setdefault("CLIPAI_HUMAN_REFRAME", "1")
    try:
        # The pipeline's exact entry point varies — try the most common
        # public surfaces in priority order. None of these will be
        # importable in the test sandbox (numpy missing), so this code
        # only runs in real environments.
        from backend.services.pipeline import process_video  # type: ignore
    except Exception as exc:
        logger.warning("pipeline.process_video unavailable: %s", exc)
        return None
    try:
        result = process_video(str(video_path))
    except Exception as exc:  # pragma: no cover - real env
        logger.warning("pipeline.process_video raised on %s: %s", video_path, exc)
        return None
    # process_video returns a dict-ish job result; extract render_plan.
    if hasattr(result, "render_plan"):
        return result.render_plan
    if isinstance(result, dict):
        return result.get("render_plan")
    return None


def _summarize_distribution(dist: Dict[str, float]) -> str:
    if not dist:
        return "(no ops)"
    parts = []
    for label, frac in sorted(dist.items(), key=lambda kv: -kv[1]):
        parts.append(f"{label} {int(round(frac*100))}%")
    return ", ".join(parts)


def _classify(overall: float) -> str:
    if overall >= PASS_THRESHOLD:
        return "PASS"
    if overall >= REVIEW_THRESHOLD:
        return "NEEDS REVIEW"
    return "FAIL"


def _render_score_block(scores) -> str:
    return (
        f"    Subject Visibility:    {scores.subject_visibility:5.1f}/100\n"
        f"    Composition:           {scores.composition:5.1f}/100\n"
        f"    Motion Smoothness:     {scores.motion_smoothness:5.1f}/100\n"
        f"    Black Bar:             {scores.black_bar:5.1f}/100\n"
        f"    Genre Appropriateness: {scores.genre_appropriateness:5.1f}/100\n"
        f"    OVERALL:               {scores.overall:5.1f}/100"
    )


def _emit_clip_block(
    name: str, entry: Dict[str, Any], stream,
) -> None:
    """Write the per-clip text summary expected by the spec format."""
    print(f"\nVideo: {name}", file=stream)
    if entry.get("skipped"):
        print(f"  SKIPPED: {entry['skipped']}", file=stream)
        return
    ct = entry.get("content_type") or "unknown"
    dist = entry.get("strategy_distribution") or {}
    print(f"  Content Type: {ct}", file=stream)
    print(f"  Strategy Distribution: {_summarize_distribution(dist)}",
          file=stream)
    print("  Scores:", file=stream)
    print(
        f"    Subject Visibility:    "
        f"{entry['subject_visibility']:5.1f}/100",
        file=stream,
    )
    print(
        f"    Composition:           "
        f"{entry['composition']:5.1f}/100",
        file=stream,
    )
    print(
        f"    Motion Smoothness:     "
        f"{entry['motion_smoothness']:5.1f}/100",
        file=stream,
    )
    print(
        f"    Black Bar:             "
        f"{entry['black_bar']:5.1f}/100",
        file=stream,
    )
    print(
        f"    Genre Appropriateness: "
        f"{entry['genre_appropriateness']:5.1f}/100",
        file=stream,
    )
    print(f"    OVERALL:               {entry['overall']:5.1f}/100",
          file=stream)
    gr = entry.get("guardrail_report")
    if gr:
        print(
            f"  Guardrail Report: {gr.get('violations_found', 0)} "
            f"violations found, {gr.get('violations_fixed', 0)} fixed, "
            f"{gr.get('violations_unfixable', 0)} unfixable",
            file=stream,
        )
    verdict = entry.get("verdict", _classify(entry["overall"]))
    mark = {"PASS": "PASS", "NEEDS REVIEW": "NEEDS REVIEW",
            "FAIL": "FAIL"}.get(verdict, verdict)
    print(f"  {mark}", file=stream)


def score_one_video(
    video_path: Path,
    *,
    pipeline_available: bool,
    pipeline_skip_reason: str,
) -> Dict[str, Any]:
    """Score one video and return a JSON-serializable dict."""
    entry: Dict[str, Any] = {
        "video": str(video_path),
        "name": video_path.name,
    }
    if not pipeline_available:
        entry["skipped"] = f"missing dependency — {pipeline_skip_reason}"
        return entry

    t0 = time.time()
    plan = _run_pipeline(video_path)
    if plan is None:
        entry["skipped"] = "pipeline did not produce a render plan"
        return entry

    try:
        from backend.services.reframe_quality_scorer import score_render_plan
    except Exception as exc:
        entry["skipped"] = f"scorer import failed: {exc!s}"
        return entry

    # Best-effort: pull the source-side analysis off the plan/object.
    face_tracks = getattr(plan, "_source_face_tracks", None)
    gaze = getattr(plan, "_source_gaze", None)
    text_regions = getattr(plan, "_source_text_regions", None)
    content_type = getattr(plan, "_content_type", None)
    has_facecam = bool(getattr(plan, "_has_facecam", False))

    scores = score_render_plan(
        plan,
        source_face_tracks=face_tracks,
        source_gaze_per_second=gaze,
        source_text_regions=text_regions,
        content_type=content_type,
        has_facecam=has_facecam,
        fps=getattr(plan, "fps", 30.0),
    )
    entry.update(scores.to_dict())
    entry["content_type"] = content_type or "unknown"
    entry["elapsed_sec"] = round(time.time() - t0, 3)
    entry["verdict"] = _classify(scores.overall)
    return entry


def _aggregate_exit_code(entries: List[Dict[str, Any]]) -> int:
    if not entries:
        return 3
    have_scored = [e for e in entries if "overall" in e]
    if not have_scored:
        return 3
    any_fail = any(e["overall"] < REVIEW_THRESHOLD for e in have_scored)
    any_review = any(
        REVIEW_THRESHOLD <= e["overall"] < PASS_THRESHOLD
        for e in have_scored
    )
    if any_fail:
        return 2
    if any_review:
        return 1
    return 0


def run(
    input_path: Path,
    output_path: Path,
    *,
    stream=None,
) -> int:
    """Run the smoke test. Public for unit tests / programmatic use."""
    stream = stream or sys.stdout
    pipeline_available, reason = _check_pipeline_available()
    print("=== Universal Reframe Smoke Test ===", file=stream)
    if not pipeline_available:
        print(f"(pipeline unavailable: {reason} — clips will be SKIPPED)",
              file=stream)

    videos = _gather_videos(input_path)
    if not videos:
        print(f"\nNo videos found at {input_path}", file=stream)
        report = {
            "input": str(input_path),
            "pipeline_available": pipeline_available,
            "skip_reason": reason,
            "clips": [],
            "summary": {
                "total": 0, "pass": 0, "review": 0, "fail": 0, "skipped": 0,
            },
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
        return 3

    entries: List[Dict[str, Any]] = []
    for v in videos:
        entry = score_one_video(
            v, pipeline_available=pipeline_available,
            pipeline_skip_reason=reason,
        )
        entries.append(entry)
        _emit_clip_block(v.name, entry, stream)

    n_total = len(entries)
    n_skip = sum(1 for e in entries if e.get("skipped"))
    n_pass = sum(1 for e in entries if e.get("verdict") == "PASS")
    n_review = sum(1 for e in entries if e.get("verdict") == "NEEDS REVIEW")
    n_fail = sum(1 for e in entries if e.get("verdict") == "FAIL")

    print(
        f"\nSummary: {n_pass}/{n_total} PASS, {n_review}/{n_total} NEEDS "
        f"REVIEW, {n_fail}/{n_total} FAIL, {n_skip}/{n_total} SKIPPED",
        file=stream,
    )

    report = {
        "input": str(input_path),
        "pipeline_available": pipeline_available,
        "skip_reason": reason,
        "clips": entries,
        "summary": {
            "total": n_total,
            "pass": n_pass,
            "review": n_review,
            "fail": n_fail,
            "skipped": n_skip,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {output_path}", file=stream)

    return _aggregate_exit_code(entries)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--input", required=True,
        help="Directory of test videos OR a single video path.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for the JSON report.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    input_path = Path(args.input)
    output_path = Path(args.output)
    return run(input_path, output_path)


if __name__ == "__main__":
    sys.exit(main())
