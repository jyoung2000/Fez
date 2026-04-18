#!/usr/bin/env python3
"""Compare Condat TV and AutoFlip LP solvers on the same shot.

Runs both solvers on a synthetic shot with a sharp subject transition
and plots them overlaid. Also reports timing for each.

Usage:
    python -m backend.scripts.compare_l1_solvers [--out /tmp/compare_l1.png]
"""

import argparse
import math
import time
import sys


def _build_test_signal(n=150, transition_at=75):
    """Synthetic shot: subject at x=400 then transitions to x=1500."""
    targets = []
    lo_bounds = []
    hi_bounds = []
    for i in range(n):
        if i < transition_at:
            targets.append(400.0)
        else:
            targets.append(1500.0)
        lo_bounds.append(0.0)
        hi_bounds.append(1920.0)
    return targets, lo_bounds, hi_bounds


def main():
    parser = argparse.ArgumentParser(description="Compare L1 solvers")
    parser.add_argument("--out", default="/tmp/compare_l1.png", help="Output plot path")
    parser.add_argument("--n", type=int, default=150, help="Signal length")
    args = parser.parse_args()

    targets, lo_bounds, hi_bounds = _build_test_signal(n=args.n)

    # ── Condat TV solver ──
    from backend.services._condat_tv import condat_tv_l1
    from backend.services.l1_camera_path import TV_LAMBDA_FRAC

    lam = TV_LAMBDA_FRAC * 1920
    t0 = time.perf_counter()
    condat_result = condat_tv_l1(targets, lam)
    condat_ms = (time.perf_counter() - t0) * 1000

    # ── AutoFlip LP solver ──
    from backend.services._autoflip_lp import solve_autoflip_lp

    t0 = time.perf_counter()
    lp_result = solve_autoflip_lp(
        targets, lo_bounds, hi_bounds,
        lam1=1.0, lam2=10.0, lam3=100.0, lam4=1000.0,
    )
    lp_ms = (time.perf_counter() - t0) * 1000

    print(f"Signal length: {args.n}")
    print(f"Condat TV:  {condat_ms:.1f}ms")
    print(f"AutoFlip LP: {lp_ms:.1f}ms")

    # ── Compute smoothness metrics ──
    def tv(x):
        return sum(abs(x[i + 1] - x[i]) for i in range(len(x) - 1))

    def accel(x):
        return sum(abs(x[i + 2] - 2 * x[i + 1] + x[i]) for i in range(len(x) - 2))

    def jerk(x):
        return sum(abs(x[i + 3] - 3 * x[i + 2] + 3 * x[i + 1] - x[i]) for i in range(len(x) - 3))

    print(f"\nTV (velocity):     Condat={tv(condat_result):.1f}  LP={tv(lp_result):.1f}")
    print(f"Acceleration:      Condat={accel(condat_result):.1f}  LP={accel(lp_result):.1f}")
    print(f"Jerk:              Condat={jerk(condat_result):.1f}  LP={jerk(lp_result):.1f}")

    # ── Plot ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 5))
        times = [i / 30.0 for i in range(args.n)]

        ax.plot(times, targets, "--", alpha=0.4, label="target", color="gray", linewidth=1)
        ax.plot(times, condat_result, "-", label=f"Condat TV ({condat_ms:.1f}ms)", color="tab:blue", linewidth=2)
        ax.plot(times, lp_result, "-", label=f"AutoFlip LP ({lp_ms:.1f}ms)", color="tab:orange", linewidth=2)

        ax.set_xlabel("time (s)")
        ax.set_ylabel("x (pixels)")
        ax.set_title("Condat TV vs AutoFlip LP — subject transition")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out, dpi=120)
        plt.close(fig)
        print(f"\nPlot saved to {args.out}")
    except ImportError:
        print("\nmatplotlib not installed — skipping plot", file=sys.stderr)


if __name__ == "__main__":
    main()
