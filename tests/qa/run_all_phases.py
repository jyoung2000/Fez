"""Master QA harness — runs every phase suite end-to-end with one report.

Designed for Claude Code Opus 4.7 to call as a single command before
pushing. Outputs a machine-readable JSON summary + a human-readable
console table.

Usage:
    python -m tests.qa.run_all_phases
    python -m tests.qa.run_all_phases --json-out /tmp/qa_report.json

Exits with code 0 if every suite passes (skips allowed), 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]


PHASES: list[dict[str, Any]] = [
    {
        "phase": "A",
        "name": "SAMURAI subject tracking",
        "suite": "tests/qa/test_phase_a_samurai.py",
        "min_tests": 24,
        "modules": ["backend/services/samurai_tracker.py",
                    "backend/services/tracker_wrapper.py",
                    "backend/services/tracker_wrapper_opencv.py"],
    },
    {
        "phase": "B",
        "name": "CoTracker3 dense + LP camera-motion subtraction",
        "suite": "tests/qa/test_phase_b_cotracker.py",
        "min_tests": 24,
        "modules": ["backend/services/cotracker3_dense.py"],
    },
    {
        "phase": "C",
        "name": "AV saliency + OCR text exclusion + LP saliency cost",
        "suite": "tests/qa/test_phase_c_saliency.py",
        "min_tests": 25,
        "modules": ["backend/services/av_saliency.py",
                    "backend/services/ocr_regions.py",
                    "backend/services/saliency_lp_term.py"],
    },
    {
        "phase": "D",
        "name": "CLIP-conditioned composition head",
        "suite": "tests/qa/test_phase_d_clip_head.py",
        "min_tests": 14,
        "modules": ["backend/services/composition_head_clip.py"],
    },
    {
        "phase": "E",
        "name": "Editorial planner + per-genre playbooks",
        "suite": "tests/qa/test_phase_e_editorial.py",
        "min_tests": 19,
        "modules": ["backend/services/editorial_planner.py",
                    "backend/services/editorial_planner_prompts/"],
    },
]


def _run_pytest(suite_path: str) -> dict:
    """Invoke pytest on a single suite and parse the summary line."""
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", suite_path, "-v",
         "--tb=short", "--no-header"],
        cwd=str(REPO),
        capture_output=True, text=True,
    )
    elapsed = time.monotonic() - t0
    out = proc.stdout + proc.stderr
    # Last line typically: "X passed, Y skipped in Z.Zs"
    summary_line = ""
    for line in reversed(out.splitlines()):
        if "passed" in line or "failed" in line or "error" in line:
            summary_line = line.strip()
            break
    passed = failed = skipped = 0
    for token in summary_line.replace(",", " ").split():
        try:
            n = int(token)
        except ValueError:
            continue
        idx = summary_line.split().index(token)
        # Tokens in pytest summary alternate "<n> <verb>".
        rest = summary_line.split()[idx + 1] if idx + 1 < len(summary_line.split()) else ""
        if "passed" in rest:
            passed = n
        elif "failed" in rest:
            failed = n
        elif "skipped" in rest:
            skipped = n
    return {
        "suite": suite_path,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "exit_code": proc.returncode,
        "elapsed_sec": round(elapsed, 2),
        "summary_line": summary_line,
    }


def _check_modules_present(modules: list[str]) -> dict:
    out = {"missing": [], "present": []}
    for m in modules:
        path = REPO / m
        if path.exists():
            out["present"].append(m)
        else:
            out["missing"].append(m)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    print("=" * 78)
    print("ClipAI 2026 SOTA Reframing — Master QA Harness")
    print("=" * 78)
    print()

    results = []
    overall_pass = True
    for p in PHASES:
        print(f"Phase {p['phase']}: {p['name']}")
        print("-" * 78)

        # 1. Module presence
        modcheck = _check_modules_present(p["modules"])
        if modcheck["missing"]:
            print(f"  modules MISSING: {modcheck['missing']}")
            overall_pass = False
            results.append({
                "phase": p["phase"], "ok": False,
                "reason": "missing_modules",
                "modules_missing": modcheck["missing"],
            })
            print()
            continue
        print(f"  modules: {len(modcheck['present'])} present")

        # 2. Run pytest. We require failed == 0 and that
        # passed + skipped meets the suite's expected size — skips are
        # legitimate when a test depends on an optional sandbox dep
        # (e.g. pydantic_settings). The homelab gate covers the
        # skipped-here cases against the real environment.
        r = _run_pytest(p["suite"])
        run_total = r["passed"] + r["skipped"]
        ok = r["exit_code"] == 0 and r["failed"] == 0 and run_total >= p["min_tests"]
        if not ok:
            overall_pass = False
        status = "PASS" if ok else "FAIL"
        print(
            f"  suite: {r['summary_line']}  ({r['elapsed_sec']}s)   [{status}]"
        )
        if r["failed"] > 0:
            print(f"  >>> {r['failed']} test(s) FAILED — see {p['suite']}")
        if run_total < p["min_tests"]:
            print(
                f"  >>> only {run_total}/{p['min_tests']} expected tests "
                f"ran (passed={r['passed']} skipped={r['skipped']})"
            )
        results.append({
            "phase": p["phase"],
            "ok": ok,
            "passed": r["passed"],
            "failed": r["failed"],
            "skipped": r["skipped"],
            "min_tests": p["min_tests"],
            "elapsed_sec": r["elapsed_sec"],
        })
        print()

    print("=" * 78)
    if overall_pass:
        print("OVERALL: PASS — all five phase QA suites green.")
        print()
        print("Next step: run the homelab bench against the fixture set.")
        print("           See tests/qa/MASTER_QA_HARNESS.md for the procedure.")
    else:
        print("OVERALL: FAIL — at least one phase regressed.")
        print()
        print("Do NOT push. Read the failures above and fix the offending phase.")
    print("=" * 78)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "ok": overall_pass,
            "phases": results,
        }, indent=2))
        print(f"\nWrote JSON report to {args.json_out}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
