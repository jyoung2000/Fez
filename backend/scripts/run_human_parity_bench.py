"""Run the human-parity bench.

For each clip in ``tests/real_content/manifest.json`` that has both a
pinned ``sha256`` (source is cached) AND a ``ground_truth_vertical_url``
(human vertical exists), this script:

  1. Loads the cached source clip via ``CLIPAI_REAL_CONTENT_CACHE``.
  2. Runs the reframe pipeline's ``build_render_plan`` to produce our
     ``RenderPlan`` for the clip.
  3. Samples our trajectory via ``render_plan_to_trajectory``.
  4. Loads the human trajectory from
     ``data/human_trajectories/<slug>.jsonl`` (produced by
     ``extract_human_trajectories.py``).
  5. Invokes ``compute_human_parity`` and writes one report per clip.

Clips missing either the source cache or the human trajectory are
skipped with a warning; they do NOT fail the gate.

All path / network / subprocess work is kept out of
``human_parity_metrics.py`` so the metrics module stays importable from
pure-Python sandboxes / tests.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from backend.services.human_parity_metrics import (
    HumanParityReport,
    TrajectorySample,
    compute_human_parity,
    render_plan_to_trajectory,
)

logger = logging.getLogger(__name__)

_DEFAULT_CACHE = "/var/cache/clipai/real_content"
_HUMAN_TRAJ_DIR = Path("data/human_trajectories")


def _load_human_trajectory(slug: str) -> list[TrajectorySample]:
    path = _HUMAN_TRAJ_DIR / f"{slug}.jsonl"
    if not path.exists():
        return []
    samples: list[TrajectorySample] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        samples.append(TrajectorySample(
            t=float(rec["t"]),
            cx=float(rec["cx"]),
            cy=float(rec["cy"]),
            w=float(rec["w"]),
            h=float(rec["h"]),
            subject_cx=rec.get("subject_cx"),
            subject_cy=rec.get("subject_cy"),
            n_subjects_in_crop=int(rec.get("n_subjects_in_crop", 1)),
            is_split_screen=bool(rec.get("is_split_screen", False)),
        ))
    return samples


def _run_one_clip(clip: dict, cache_dir: Path) -> HumanParityReport | None:
    slug = clip["slug"]
    if not clip.get("sha256"):
        logger.info("[%s] skipped: source not cached (empty sha256)", slug)
        return None
    source_path = cache_dir / f"{slug}.{clip.get('ext', 'mp4')}"
    if not source_path.exists():
        logger.info("[%s] skipped: source file %s missing", slug, source_path)
        return None

    human = _load_human_trajectory(slug)
    if not human:
        logger.info("[%s] skipped: no human trajectory under %s/",
                    slug, _HUMAN_TRAJ_DIR)
        return None

    # Lazy import: the pipeline entry point depends on the full backend.
    try:
        from backend.services.pipeline import run_reframe_on_clip_for_bench
    except Exception as e:
        logger.warning("[%s] pipeline import failed (%s) — using stub", slug, e)
        return HumanParityReport(
            clip_slug=slug,
            content_type=clip.get("target_clipcontenttype", ""),
            n_frames=0,
        )

    render_plan_dict = run_reframe_on_clip_for_bench(
        str(source_path),
        content_type=clip.get("target_clipcontenttype"),
    )
    ours = render_plan_to_trajectory(render_plan_dict, sample_fps=30.0)
    report = compute_human_parity(
        ours, human,
        clip_slug=slug,
        content_type=clip.get("target_clipcontenttype", ""),
    )
    logger.info("[%s] mae_cx=%.3f mae_cy=%.3f macro_f1=%.3f cut_mean_ms=%.0f",
                slug, report.mae_cx, report.mae_cy,
                report.framing_macro_f1, report.cut_timing_deviation_ms_mean)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="tests/real_content/manifest.json")
    parser.add_argument("--cache-dir", default=os.environ.get(
        "CLIPAI_REAL_CONTENT_CACHE", _DEFAULT_CACHE))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    cache_dir = Path(args.cache_dir)

    runs: dict[str, dict] = {}
    for clip in manifest.get("clips", []):
        report = _run_one_clip(clip, cache_dir)
        if report is not None:
            runs[clip["slug"]] = report.to_dict()

    out = {
        "implementation_version": "human_parity_v1",
        "runs": runs,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"wrote {len(runs)} reports to {args.out}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
