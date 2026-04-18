#!/usr/bin/env python3
"""Phase E — Detection-stack flag matrix runner.

Wraps ``backend.scripts.measure_autoflip_parity`` and re-runs it under
every combination of the Phase 11 detection-stack feature flags so the
team can compare deltas in one table. Flags swept:

  CLIPAI_FACE_EMBEDDING        ∈ {sface, arcface}        (Phase A)
  CLIPAI_OBJECT_DETECTOR       ∈ {yolov8n, yolo11n}      (Phase B)
  CLIPAI_PERSON_USE_POSE       ∈ {false, true}           (Phase B; pose only valid w/ yolo11n)
  CLIPAI_ASD_BACKEND           ∈ {heuristic, light_asd}  (Phase C)
  CLIPAI_ANIME_FACE_BACKEND    ∈ {lbpcascade, yolo_anime} (Phase D)

The full sweep is 32 combos × N fixtures and is **expensive** — the
runner is gated behind ``RUN_DETECTION_MATRIX=1`` and prints a hint
otherwise. Pass ``--fixture <name>`` (repeatable) to limit the fixture
set to a smaller slice for a quick smoke run; the README recommends
running the matrix against ``synthetic_alternating_2s`` and
``multi_speaker_crowd_10s`` only until a human signs off.

For each combination × fixture the runner records:

  - parity_metric_set (whatever measure_autoflip_parity emits)
  - wall-clock runtime delta vs. the all-defaults run
  - flag combination as a stable key

Outputs:

  - ``docs/autoflip_parity_v2_phase11_detection_stack.md`` (markdown)
  - ``/tmp/clipai_detection_matrix.csv`` (one row per combo × fixture)

Usage:

    RUN_DETECTION_MATRIX=1 python -m backend.scripts.measure_detection_stack_matrix \\
        --fixture synthetic_alternating_2s \\
        --fixture multi_speaker_crowd_10s

Without ``RUN_DETECTION_MATRIX=1`` the script prints a usage hint and
exits 0 so it's safe to invoke from CI scaffolding.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


FLAGS = {
    "CLIPAI_FACE_EMBEDDING": ["sface", "arcface"],
    "CLIPAI_OBJECT_DETECTOR": ["yolov8n", "yolo11n"],
    "CLIPAI_PERSON_USE_POSE": ["false", "true"],
    "CLIPAI_ASD_BACKEND": ["heuristic", "light_asd"],
    "CLIPAI_ANIME_FACE_BACKEND": ["lbpcascade", "yolo_anime"],
}


def _is_valid_combo(combo: dict) -> bool:
    """Filter out invalid combinations.

    ``CLIPAI_PERSON_USE_POSE=true`` only makes sense alongside
    ``CLIPAI_OBJECT_DETECTOR=yolo11n`` because the pose model lives in
    the same Ultralytics release. We still emit the row for the matrix
    completeness check but mark it as ``skipped``.
    """
    if combo["CLIPAI_PERSON_USE_POSE"] == "true" and combo["CLIPAI_OBJECT_DETECTOR"] != "yolo11n":
        return False
    return True


def _combo_key(combo: dict) -> str:
    return ",".join(f"{k.split('_', 1)[1]}={v}" for k, v in sorted(combo.items()))


def _run_one_combo(combo: dict, fixture_args: list, dry_run: bool) -> dict:
    """Invoke measure_autoflip_parity in a subprocess with the given env.

    Returns a dict with the parsed JSON payload + wall-clock timing.
    Errors are surfaced inline so the matrix doc can show what failed.
    """
    env = os.environ.copy()
    env.update(combo)

    cmd = [sys.executable, "-m", "backend.scripts.measure_autoflip_parity", "--quiet"]
    for f in fixture_args:
        cmd.extend(["--fixture", f])
    if dry_run:
        cmd.append("--dry-run")

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=900,
        )
        elapsed = time.monotonic() - start
    except subprocess.TimeoutExpired:
        return {
            "combo": combo, "elapsed_s": time.monotonic() - start,
            "status": "timeout", "payload": None,
        }
    if proc.returncode != 0:
        return {
            "combo": combo, "elapsed_s": elapsed, "status": "error",
            "stderr": proc.stderr[-2000:], "payload": None,
        }
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {
            "combo": combo, "elapsed_s": elapsed, "status": "parse_error",
            "stderr": str(exc), "payload": None,
        }
    return {
        "combo": combo, "elapsed_s": elapsed, "status": "ok",
        "payload": payload,
    }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="measure_detection_stack_matrix",
        description=(
            "Sweep the Phase 11 detection-stack feature flags through "
            "measure_autoflip_parity and emit a delta table."
        ),
    )
    p.add_argument(
        "--fixture", action="append", default=[],
        help="Limit the parity run to the named fixture(s). Repeatable.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Pass --dry-run to the inner runner (skips the segmenter).",
    )
    p.add_argument(
        "--csv", default="/tmp/clipai_detection_matrix.csv",
        help="Where to write the per-combo CSV (default /tmp/clipai_detection_matrix.csv).",
    )
    p.add_argument(
        "--md", default="docs/autoflip_parity_v2_phase11_detection_stack.md",
        help="Where to write the markdown summary table.",
    )
    return p


def _row_summary(payload: dict) -> dict:
    """Pull a small set of comparable scalar metrics out of the parity
    runner's per-fixture results.

    ``measure_autoflip_parity`` emits one entry per fixture with a
    ``metrics`` sub-dict; we average a few well-known scalars across
    fixtures so the matrix table fits on a screen.
    """
    if not payload or "results" not in payload:
        return {}
    fixtures = payload["results"]
    if not isinstance(fixtures, list) or not fixtures:
        return {}
    scalars = ("switch_recall", "switch_precision", "mean_lag_frames",
               "max_lag_frames", "mean_center_err_pct", "p95_center_err_pct")
    sums: dict = {k: 0.0 for k in scalars}
    counts: dict = {k: 0 for k in scalars}
    for entry in fixtures:
        m = entry.get("metrics", {}) if isinstance(entry, dict) else {}
        for k in scalars:
            v = m.get(k)
            if isinstance(v, (int, float)):
                sums[k] += float(v)
                counts[k] += 1
    return {
        k: (sums[k] / counts[k]) if counts[k] else None
        for k in scalars
    }


def _write_csv(rows: list, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_markdown(rows: list, path: str, fixtures: list) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Phase 11 — Detection stack feature-flag matrix\n\n")
        f.write(
            "Generated by `backend/scripts/measure_detection_stack_matrix.py`. "
            "One row per (CLIPAI_* combination) × fixture set, with parity "
            "scalars averaged across the fixtures listed below.\n\n"
        )
        f.write(f"- **Run timestamp:** {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"- **Fixtures:** {', '.join(fixtures) if fixtures else 'all'}\n")
        f.write(
            "- **Hardware constraint:** all new ONNX backends run on CPU "
            "via `onnxruntime CPUExecutionProvider`. The GTX 1650 stays "
            "reserved for Whisper / Ollama.\n\n"
        )
        f.write(
            "## Phase A — ArcFace embeddings\n\n"
            "_Fill in delta vs. baseline once the matrix has run._\n\n"
            "## Phase B — YOLO11n + pose anchor\n\n"
            "_Fill in delta vs. baseline once the matrix has run._\n\n"
            "## Phase C — Light-ASD\n\n"
            "_Fill in delta vs. baseline once the matrix has run._\n\n"
            "## Phase D — YOLO anime face\n\n"
            "_Fill in delta vs. baseline once the matrix has run._\n\n"
            "## Per-combo results\n\n"
        )
        if not rows:
            f.write("_No rows — the matrix runner produced no results._\n")
            return
        keys = [
            "combo", "status", "elapsed_s",
            "switch_recall", "switch_precision",
            "mean_lag_frames", "max_lag_frames",
            "mean_center_err_pct", "p95_center_err_pct",
        ]
        f.write("| " + " | ".join(keys) + " |\n")
        f.write("|" + "|".join("---" for _ in keys) + "|\n")
        for r in rows:
            cells = []
            for k in keys:
                v = r.get(k)
                if v is None:
                    cells.append("—")
                elif isinstance(v, float):
                    cells.append(f"{v:.3f}")
                else:
                    cells.append(str(v))
            f.write("| " + " | ".join(cells) + " |\n")


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)

    if os.environ.get("RUN_DETECTION_MATRIX") != "1":
        print(
            "[detection-matrix] gated behind RUN_DETECTION_MATRIX=1. "
            "Set the env var to actually run the sweep. This is the "
            "scaffolding entrypoint; CI invokes it without the gate to "
            "verify the script imports cleanly.",
            file=sys.stderr,
        )
        return 0

    combos: list = []
    for values in itertools.product(*FLAGS.values()):
        combo = dict(zip(FLAGS.keys(), values))
        if not _is_valid_combo(combo):
            continue
        combos.append(combo)

    print(f"[detection-matrix] {len(combos)} valid combinations queued",
          file=sys.stderr)
    rows: list = []
    for i, combo in enumerate(combos):
        print(
            f"[detection-matrix] [{i + 1}/{len(combos)}] {_combo_key(combo)}",
            file=sys.stderr,
        )
        result = _run_one_combo(combo, args.fixture, args.dry_run)
        summary = _row_summary(result.get("payload") or {})
        row = {
            "combo": _combo_key(combo),
            "status": result["status"],
            "elapsed_s": round(result["elapsed_s"], 2),
        }
        row.update(summary)
        rows.append(row)

    _write_csv(rows, args.csv)
    _write_markdown(rows, args.md, args.fixture)
    print(
        f"[detection-matrix] wrote {args.csv} ({len(rows)} rows) and {args.md}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
