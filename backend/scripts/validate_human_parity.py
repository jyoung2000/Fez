"""Validate a bench run against a pinned human-parity baseline.

Usage:

    # First time — pin a baseline from the current pipeline:
    python -m backend.scripts.validate_human_parity \\
        --results /tmp/human_parity_run.json \\
        --pin docs/human_parity_baseline.json

    # Subsequent CI runs:
    python -m backend.scripts.validate_human_parity \\
        --results /tmp/human_parity_run.json \\
        --baseline docs/human_parity_baseline.json

Results JSON schema (produced by the bench runner):

    {
        "implementation_version": "human_parity_v1",
        "runs": {
            "<clip_slug>": { ...HumanParityReport.to_dict()... },
            ...
        }
    }

Regression rules (per clip, per metric):

    mae_cx, mae_cy                        : must not increase by > 0.02
    framing_macro_f1                      : must not drop by > 0.05
    cut_timing_deviation_ms_mean          : must not increase by > 50 ms
    lead_room_correlation_x / _y          : must not drop by > 0.10
    aesthetic_score_mean                  : must not drop by > 0.5 (if both populated)

This file replaces ``validate_v2_phases.py`` as the CI gate. The
synthetic AutoFlip fixtures remain as smoke-only and are invoked with
``--smoke-only``; they no longer block merges.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Regression thresholds ──────────────────────────────────────────

_THRESHOLDS = {
    "mae_cx":                           ("inc", 0.02),
    "mae_cy":                           ("inc", 0.02),
    "framing_macro_f1":                 ("dec", 0.05),
    "cut_timing_deviation_ms_mean":     ("inc", 50.0),
    "cut_timing_deviation_ms_p90":      ("inc", 100.0),
    "lead_room_correlation_x":          ("dec", 0.10),
    "lead_room_correlation_y":          ("dec", 0.10),
    "aesthetic_score_mean":             ("dec", 0.5),
}


def _check_clip(slug: str, baseline: dict, current: dict) -> list[str]:
    violations: list[str] = []
    for metric, (direction, thresh) in _THRESHOLDS.items():
        b = baseline.get(metric)
        c = current.get(metric)
        if b is None or c is None:
            continue
        if direction == "inc":
            if c - b > thresh:
                violations.append(
                    f"[{slug}] {metric} regressed: baseline={b:.4f}, "
                    f"current={c:.4f}, delta=+{c - b:.4f} > {thresh}"
                )
        else:
            if b - c > thresh:
                violations.append(
                    f"[{slug}] {metric} regressed: baseline={b:.4f}, "
                    f"current={c:.4f}, delta=-{b - c:.4f} > {thresh}"
                )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="current run JSON")
    parser.add_argument("--baseline", help="pinned baseline JSON to compare against")
    parser.add_argument("--pin", help="write the current results as the new baseline")
    parser.add_argument("--smoke-only", action="store_true",
                        help="run only the legacy synthetic fixtures (no human gate)")
    args = parser.parse_args()

    if args.smoke_only:
        # The old AutoFlip fixtures live under tests/ and run via
        # pytest; this is a pass-through marker so CI can invoke the
        # same entry point.
        logger.info("smoke-only: invoke the legacy fixture suite via pytest")
        return 0

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: results file {results_path} not found", file=sys.stderr)
        return 2
    current = json.loads(results_path.read_text())
    if "runs" not in current:
        print("ERROR: results JSON missing 'runs' key", file=sys.stderr)
        return 2

    if args.pin:
        Path(args.pin).write_text(json.dumps(current, indent=2, sort_keys=True))
        print(f"PINNED baseline at {args.pin} ({len(current['runs'])} clips)")
        return 0

    if not args.baseline:
        print("ERROR: either --baseline or --pin is required", file=sys.stderr)
        return 2

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(f"ERROR: baseline {baseline_path} not found — "
              f"run with --pin first", file=sys.stderr)
        return 2
    baseline = json.loads(baseline_path.read_text())

    all_violations: list[str] = []
    for slug, cur in current["runs"].items():
        base = baseline.get("runs", {}).get(slug)
        if base is None:
            print(f"[{slug}] NEW clip — recording only, no gate applied")
            continue
        all_violations.extend(_check_clip(slug, base, cur))

    if all_violations:
        print("HUMAN-PARITY REGRESSIONS:")
        for v in all_violations:
            print(f"  - {v}")
        return 1

    print(f"OK: {len(current['runs'])} clips, no regressions.")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
