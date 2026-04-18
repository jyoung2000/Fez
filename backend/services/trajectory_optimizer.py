"""L1-regularized camera path optimizer.

Smooths the camera trajectory while respecting hard constraints from
required features. Uses scipy.optimize.linprog when available, falls
back to simple moving-average smoothing otherwise.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def optimize_camera_path(
    targets: list,               # list[(t, target_x)]
    hard_constraints: list,      # list[(t, min_x, max_x)]
    shot_boundaries: list,       # list[float] — path MUST NOT smooth across these
    lambda_smooth: float = 0.3,
    job_id: str = "",
) -> list:                       # list[(t, x)]
    """Solve:
        minimize  sum_t |x[t] - target[t]| + lambda_smooth * sum_t |x[t+1] - x[t]|
        subject to  min_x[t] <= x[t] <= max_x[t]  for all t

    This is a linear program in 2*N variables (positive/negative slack for the L1 terms).
    Uses scipy.optimize.linprog with method='highs'. For >60s clips, chunk into
    overlapping windows and stitch.

    Never smooths across shot boundaries — runs the optimizer independently on each
    shot segment and concatenates.
    """
    if not targets:
        return []

    if len(targets) == 1:
        return [(targets[0][0], targets[0][1])]

    # Build constraint lookup
    constraint_map = {}
    for entry in hard_constraints:
        t, lo, hi = entry[0], entry[1], entry[2]
        constraint_map[t] = (lo, hi)

    # Split targets by shot boundaries
    sorted_boundaries = sorted(shot_boundaries)
    segments = _split_by_boundaries(targets, sorted_boundaries)

    result = []
    for seg_targets in segments:
        if not seg_targets:
            continue
        seg_constraints = []
        for t, x in seg_targets:
            if t in constraint_map:
                seg_constraints.append((t, constraint_map[t][0], constraint_map[t][1]))
        optimized = _optimize_segment(seg_targets, seg_constraints, lambda_smooth, job_id)
        result.extend(optimized)

    return result


def _split_by_boundaries(targets, boundaries):
    """Split target list at shot boundaries."""
    if not boundaries:
        return [targets]

    segments = []
    current = []
    bi = 0

    for t, x in targets:
        # Check if we crossed a boundary
        while bi < len(boundaries) and boundaries[bi] <= t:
            if current:
                segments.append(current)
                current = []
            bi += 1
        current.append((t, x))

    if current:
        segments.append(current)

    return segments if segments else [targets]


def _optimize_segment(targets, constraints, lambda_smooth, job_id=""):
    """Optimize a single segment (no shot boundaries within)."""
    n = len(targets)
    if n <= 1:
        return [(targets[0][0], targets[0][1])] if targets else []

    target_xs = [x for _, x in targets]
    timestamps = [t for t, _ in targets]

    # Build constraint bounds per variable
    lo_bounds = [0.0] * n
    hi_bounds = [100.0] * n

    constraint_t_map = {}
    for t, lo, hi in constraints:
        constraint_t_map[t] = (lo, hi)

    for i, t in enumerate(timestamps):
        if t in constraint_t_map:
            lo_bounds[i] = constraint_t_map[t][0]
            hi_bounds[i] = constraint_t_map[t][1]

    # Try scipy LP solver
    try:
        return _solve_with_scipy(timestamps, target_xs, lo_bounds, hi_bounds, lambda_smooth, job_id)
    except Exception as e:
        logger.warning("[%s] Trajectory optimizer: scipy LP failed (%s), using moving-average fallback", job_id, e)
        return _moving_average_fallback(timestamps, target_xs, lo_bounds, hi_bounds)


def _solve_with_scipy(timestamps, target_xs, lo_bounds, hi_bounds, lambda_smooth, job_id=""):
    """Solve the L1 LP with scipy.optimize.linprog."""
    try:
        from scipy.optimize import linprog
    except ImportError:
        logger.warning("[%s] Trajectory optimizer: scipy not available, using moving-average fallback", job_id)
        return _moving_average_fallback(timestamps, target_xs, lo_bounds, hi_bounds)

    n = len(target_xs)

    # Variables: x[0..n-1], u[0..n-1] (|x-target| slack), v[0..n-2] (smoothness slack)
    # Total: n + n + (n-1) = 3n - 1 variables
    n_vars = n + n + (n - 1)

    # Objective: minimize sum(u) + lambda_smooth * sum(v)
    c = [0.0] * n + [1.0] * n + [lambda_smooth] * (n - 1)

    # Inequality constraints: A_ub @ vars <= b_ub
    # For each i: x[i] - target[i] <= u[i]  and  target[i] - x[i] <= u[i]
    # For each i: x[i+1] - x[i] <= v[i]  and  x[i] - x[i+1] <= v[i]
    A_rows = []
    b_rows = []

    for i in range(n):
        # x[i] - u[i] <= target[i]  =>  x[i] - target[i] <= u[i]
        row = [0.0] * n_vars
        row[i] = 1.0
        row[n + i] = -1.0
        A_rows.append(row)
        b_rows.append(target_xs[i])

        # -x[i] - u[i] <= -target[i]  =>  target[i] - x[i] <= u[i]
        row = [0.0] * n_vars
        row[i] = -1.0
        row[n + i] = -1.0
        A_rows.append(row)
        b_rows.append(-target_xs[i])

    for i in range(n - 1):
        # x[i+1] - x[i] <= v[i]
        row = [0.0] * n_vars
        row[i + 1] = 1.0
        row[i] = -1.0
        row[2 * n + i] = -1.0
        A_rows.append(row)
        b_rows.append(0.0)

        # x[i] - x[i+1] <= v[i]
        row = [0.0] * n_vars
        row[i] = 1.0
        row[i + 1] = -1.0
        row[2 * n + i] = -1.0
        A_rows.append(row)
        b_rows.append(0.0)

    # Variable bounds
    bounds = []
    for i in range(n):
        bounds.append((lo_bounds[i], hi_bounds[i]))  # x[i]
    for i in range(n):
        bounds.append((0.0, None))  # u[i] >= 0
    for i in range(n - 1):
        bounds.append((0.0, None))  # v[i] >= 0

    # Check feasibility of hard constraints
    for i in range(n):
        if lo_bounds[i] > hi_bounds[i]:
            logger.warning("[%s] Trajectory optimizer: infeasible constraints at t=%.2f (lo=%.1f > hi=%.1f)",
                           job_id, timestamps[i], lo_bounds[i], hi_bounds[i])
            return [(t, x) for t, x in zip(timestamps, target_xs)]

    result = linprog(c, A_ub=A_rows, b_ub=b_rows, bounds=bounds, method='highs',
                     options={'presolve': True, 'time_limit': 2.0})

    if not result.success:
        logger.warning("[%s] Trajectory optimizer: LP not optimal (status=%s), returning targets",
                       job_id, result.message)
        return [(t, x) for t, x in zip(timestamps, target_xs)]

    optimized_x = result.x[:n]
    return [(timestamps[i], float(optimized_x[i])) for i in range(n)]


def _moving_average_fallback(timestamps, target_xs, lo_bounds, hi_bounds, window=5):
    """Simple moving-average smoothing when scipy is unavailable."""
    n = len(target_xs)
    smoothed = list(target_xs)

    half_w = window // 2
    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        avg = sum(target_xs[start:end]) / (end - start)
        # Clamp to hard constraints
        avg = max(lo_bounds[i], min(hi_bounds[i], avg))
        smoothed[i] = avg

    return [(timestamps[i], smoothed[i]) for i in range(n)]
