#!/usr/bin/env python3
"""Plot L1 solver telemetry CSVs produced by CLIPAI_DUMP_L1=1.

Reads CSVs from /tmp/clipai_l1/ and renders target vs solved overlays.

Usage:
    python -m backend.scripts.plot_l1_solve [--dir /tmp/clipai_l1] [--out /tmp/clipai_l1/plots]
"""

import argparse
import csv
import sys
from pathlib import Path


def _read_csv(path: Path):
    """Read a telemetry CSV and return column lists."""
    times, targets, solved, lo_bounds, hi_bounds = [], [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["t"]))
            targets.append(float(row["target_x"]))
            solved.append(float(row["solved_x"]))
            lo_bounds.append(float(row["lo_bound"]) if row["lo_bound"] else None)
            hi_bounds.append(float(row["hi_bound"]) if row["hi_bound"] else None)
    return times, targets, solved, lo_bounds, hi_bounds


def plot_one(csv_path: Path, out_dir: Path):
    """Plot a single CSV file."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot", file=sys.stderr)
        return

    times, targets, solved, lo_bounds, hi_bounds = _read_csv(csv_path)
    if not times:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, targets, "o-", markersize=2, alpha=0.5, label="target_x", color="tab:blue")
    ax.plot(times, solved, "-", linewidth=2, label="solved_x", color="tab:orange")

    # Plot bounds as shaded region
    lo_filled = [v if v is not None else min(targets) for v in lo_bounds]
    hi_filled = [v if v is not None else max(targets) for v in hi_bounds]
    has_bounds = any(v is not None for v in lo_bounds)
    if has_bounds:
        ax.fill_between(times, lo_filled, hi_filled, alpha=0.15, color="green", label="bounds")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("x (pixels)")
    ax.set_title(csv_path.stem)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{csv_path.stem}.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot L1 solver telemetry")
    parser.add_argument("--dir", default="/tmp/clipai_l1", help="CSV input directory")
    parser.add_argument("--out", default=None, help="Plot output directory (default: <dir>/plots)")
    args = parser.parse_args()

    csv_dir = Path(args.dir)
    out_dir = Path(args.out) if args.out else csv_dir / "plots"

    csvs = sorted(csv_dir.glob("*.csv"))
    if not csvs:
        print(f"No CSVs found in {csv_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Plotting {len(csvs)} CSVs from {csv_dir}")
    for p in csvs:
        plot_one(p, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
