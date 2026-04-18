"""LP-based AutoFlip camera path solver with acceleration and jerk penalties.

Two solvers live in this module:

1. ``solve_autoflip_lp`` — the **single-subject** L1 path. One target
   per frame, hard box constraints per frame, smoothness penalties on
   velocity / acceleration / jerk. This is what the reframe segmenter
   has used since Phase 0.

2. ``solve_multi_region_camera_path`` (Phase 3) — the **multi-region**
   generalization. Takes per-frame lists of required + optional bboxes
   instead of a scalar target. Required regions become hard box
   constraints (the crop window must contain every required bbox);
   optional regions become soft penalties (the LP is rewarded for
   keeping them in-frame but won't fail if they spill out). When the
   required regions can't all fit inside the crop on some frame, the
   LP reports the infeasible frame indices so the caller can fall back
   to SPLIT_SCREEN / WIDE_MASTER.

Both formulate the AutoFlip objective (Grundmann et al. 2011, Section
4.2) as a linear program using scipy's HiGHS backend:

    min  lam1 * sum |cam[t] - target[t]|                          (data fidelity)
       + lam2 * sum |cam[t] - cam[t-1]|                           (velocity)
       + lam3 * sum |cam[t] - 2*cam[t-1] + cam[t-2]|              (acceleration)
       + lam4 * sum |cam[t] - 3*cam[t-1] + 3*cam[t-2] - cam[t-3]| (jerk)
       + lam_soft * sum w_k * outside_amount_k                    (multi-region only)
    s.t. lo[t] <= cam[t] <= hi[t]
         (hard box bounds derived from required regions)

Each |·| term is linearized via an auxiliary slack variable and two
inequality constraints: ``|z| <= s  <=>  z <= s, -z <= s, s >= 0``.

Reference:
    M. Grundmann, V. Kwatra, and I. Essa, "Auto-Directed Video Stabilization
    with Robust L1 Optimal Camera Paths," CVPR 2011, Section 4.2.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

logger = logging.getLogger(__name__)


def solve_autoflip_lp(
    targets: list[float],
    lo_bounds: list[float],
    hi_bounds: list[float],
    lam1: float = 1.0,
    lam2: float = 10.0,
    lam3: float = 100.0,
    lam4: float = 1000.0,
    *,
    weights: list[float] | None = None,
) -> list[float]:
    """Solve the AutoFlip L1-optimal camera path via linear programming.

    Args:
        targets: Target face positions (pixels), length n.
        lo_bounds: Per-frame lower bounds on camera position.
        hi_bounds: Per-frame upper bounds on camera position.
        lam1: Data fidelity weight.
        lam2: Velocity smoothness weight.
        lam3: Acceleration smoothness weight.
        lam4: Jerk smoothness weight.
        weights: Optional per-frame weights for data fidelity term.
            When provided, scales lam1 per frame: lam1 * weights[t].
            None = uniform weights of 1.0 (bit-identical to unweighted).

    Returns:
        Solved camera path, length n.
    """
    n = len(targets)
    if n <= 1:
        return list(targets)

    # Variable layout:
    #   cam[0..n-1]       — camera positions (n vars)
    #   s1[0..n-1]        — |cam[t] - target[t]| slack (n vars)
    #   s2[0..n-2]        — |cam[t+1] - cam[t]| slack (n-1 vars)
    #   s3[0..n-3]        — |accel| slack (n-2 vars)
    #   s4[0..n-4]        — |jerk| slack (n-3 vars)
    #
    # Total variables: n + n + (n-1) + (n-2) + (n-3) = 5n - 6

    n_s2 = max(n - 1, 0)
    n_s3 = max(n - 2, 0)
    n_s4 = max(n - 3, 0)
    n_vars = n + n + n_s2 + n_s3 + n_s4

    # Offsets into the variable vector
    off_cam = 0
    off_s1 = n
    off_s2 = off_s1 + n
    off_s3 = off_s2 + n_s2
    off_s4 = off_s3 + n_s3

    # ── Objective: min lam1*w[t]*sum(s1) + lam2*sum(s2) + lam3*sum(s3) + lam4*sum(s4)
    c = np.zeros(n_vars)
    if weights is not None:
        for t in range(n):
            c[off_s1 + t] = lam1 * weights[t]
    else:
        c[off_s1:off_s1 + n] = lam1
    c[off_s2:off_s2 + n_s2] = lam2
    c[off_s3:off_s3 + n_s3] = lam3
    c[off_s4:off_s4 + n_s4] = lam4

    # ── Inequality constraints: A_ub @ x <= b_ub ──
    # Each |z| <= s  =>  z - s <= 0  AND  -z - s <= 0
    # So 2 rows per |·| term.
    n_constraints = 2 * n + 2 * n_s2 + 2 * n_s3 + 2 * n_s4
    A = lil_matrix((n_constraints, n_vars))
    b = np.zeros(n_constraints)
    row = 0

    # s1: |cam[t] - target[t]| <= s1[t]
    #   cam[t] - s1[t] <= target[t]
    #   -cam[t] - s1[t] <= -target[t]
    for t in range(n):
        A[row, off_cam + t] = 1.0
        A[row, off_s1 + t] = -1.0
        b[row] = targets[t]
        row += 1
        A[row, off_cam + t] = -1.0
        A[row, off_s1 + t] = -1.0
        b[row] = -targets[t]
        row += 1

    # s2: |cam[t+1] - cam[t]| <= s2[t]
    #   cam[t+1] - cam[t] - s2[t] <= 0
    #   cam[t] - cam[t+1] - s2[t] <= 0
    for t in range(n_s2):
        A[row, off_cam + t + 1] = 1.0
        A[row, off_cam + t] = -1.0
        A[row, off_s2 + t] = -1.0
        b[row] = 0.0
        row += 1
        A[row, off_cam + t] = 1.0
        A[row, off_cam + t + 1] = -1.0
        A[row, off_s2 + t] = -1.0
        b[row] = 0.0
        row += 1

    # s3: |cam[t+2] - 2*cam[t+1] + cam[t]| <= s3[t]
    for t in range(n_s3):
        # cam[t+2] - 2*cam[t+1] + cam[t] - s3[t] <= 0
        A[row, off_cam + t + 2] = 1.0
        A[row, off_cam + t + 1] = -2.0
        A[row, off_cam + t] = 1.0
        A[row, off_s3 + t] = -1.0
        b[row] = 0.0
        row += 1
        # -(cam[t+2] - 2*cam[t+1] + cam[t]) - s3[t] <= 0
        A[row, off_cam + t + 2] = -1.0
        A[row, off_cam + t + 1] = 2.0
        A[row, off_cam + t] = -1.0
        A[row, off_s3 + t] = -1.0
        b[row] = 0.0
        row += 1

    # s4: |cam[t+3] - 3*cam[t+2] + 3*cam[t+1] - cam[t]| <= s4[t]
    for t in range(n_s4):
        # cam[t+3] - 3*cam[t+2] + 3*cam[t+1] - cam[t] - s4[t] <= 0
        A[row, off_cam + t + 3] = 1.0
        A[row, off_cam + t + 2] = -3.0
        A[row, off_cam + t + 1] = 3.0
        A[row, off_cam + t] = -1.0
        A[row, off_s4 + t] = -1.0
        b[row] = 0.0
        row += 1
        # -(cam[t+3] - 3*cam[t+2] + 3*cam[t+1] - cam[t]) - s4[t] <= 0
        A[row, off_cam + t + 3] = -1.0
        A[row, off_cam + t + 2] = 3.0
        A[row, off_cam + t + 1] = -3.0
        A[row, off_cam + t] = 1.0
        A[row, off_s4 + t] = -1.0
        b[row] = 0.0
        row += 1

    A = A.tocsc()

    # ── Variable bounds ──
    bounds = []
    for t in range(n):
        bounds.append((lo_bounds[t], hi_bounds[t]))  # cam[t]
    for _ in range(n):
        bounds.append((0.0, None))  # s1[t] >= 0
    for _ in range(n_s2):
        bounds.append((0.0, None))  # s2[t] >= 0
    for _ in range(n_s3):
        bounds.append((0.0, None))  # s3[t] >= 0
    for _ in range(n_s4):
        bounds.append((0.0, None))  # s4[t] >= 0

    # ── Solve ──
    result = linprog(
        c, A_ub=A, b_ub=b, bounds=bounds,
        method="highs",
        options={"presolve": True, "time_limit": 10.0},
    )

    if not result.success:
        logger.warning("AutoFlip LP solver failed: %s — falling back to targets", result.message)
        return list(targets)

    cam = result.x[off_cam:off_cam + n].tolist()
    return cam


# ── Phase 3: multi-region LP ────────────────────────────────────────


@dataclass
class MultiRegionLPResult:
    """Result bundle from ``solve_multi_region_camera_path``.

    Attributes:
        status: ``"feasible"`` if the LP found a path that satisfies
            every required region on every frame; ``"infeasible"`` if
            at least one frame's per-frame box collapses (i.e. two
            required bboxes are too far apart to fit inside the crop
            window simultaneously); ``"lp_failed"`` for everything
            else (HiGHS time-limit, numerical failure, etc).
        camera_path: per-frame solved camera center in **pixels**.
            Length matches ``n_frames`` when status is ``"feasible"``;
            empty otherwise.
        infeasible_frames: indices of frames where the required-region
            box collapsed (``lo > hi``). Always populated when
            ``status == "infeasible"``; may be partially populated
            when the caller uses the soft-feasibility threshold.
        infeasibility_ratio: ``len(infeasible_frames) / max(n_frames, 1)``.
            The layout-decision helper uses this as the "fits / split
            / wide" routing input.
        n_frames: total number of frames in the input.
        n_required: total count of required bboxes across all frames.
        n_optional: total count of optional bboxes across all frames.
        lp_message: the scipy linprog `result.message` string for
            telemetry / debugging.
        solve_ms: wall-clock LP solve time in milliseconds.
    """

    status: str
    camera_path: list = field(default_factory=list)
    infeasible_frames: list = field(default_factory=list)
    infeasibility_ratio: float = 0.0
    n_frames: int = 0
    n_required: int = 0
    n_optional: int = 0
    lp_message: str = ""
    solve_ms: float = 0.0


# NOTE: ``_per_frame_bounds_from_required`` lives in
# ``backend.services.multi_region_layout`` so it can be imported from
# a numpy-free sandbox. The LP function below imports it lazily.


def solve_multi_region_camera_path(
    required_per_frame: list,
    optional_per_frame: Optional[list] = None,
    *,
    crop_width_px: float,
    source_width_px: float,
    lam_data: float = 1.0,
    lam_velocity: float = 20.0,
    lam_accel: float = 100.0,
    lam_jerk: float = 100.0,
    lam_soft: float = 5.0,
    time_limit_sec: float = 10.0,
) -> MultiRegionLPResult:
    """Solve an L1 camera path that keeps every required bbox in-crop.

    Inputs are length-N lists indexed by frame. Each frame entry is a
    list of bbox tuples in **pixel** coordinates:

      ``required_per_frame[t]``: list of ``(left_px, right_px)`` —
        every entry MUST land inside the crop window
        ``[cam[t] - half, cam[t] + half]`` for the LP to be feasible.

      ``optional_per_frame[t]``: list of ``(left_px, right_px, weight)``
        — entries are penalized for being outside the crop in proportion
        to ``weight`` but never make the LP fail. ``weight`` is a
        positive float; a region with weight 1.0 contributes
        ``lam_soft * outside_amount_px`` per frame to the objective.

    Set ``optional_per_frame`` to ``None`` (or an empty list per frame)
    to skip the soft-penalty machinery entirely.

    The objective combines:
      * a small data-fidelity pull toward the midpoint of the per-frame
        feasible interval (``(lo + hi) / 2``) so the LP has a defined
        absolute position when smoothness alone would be ambiguous,
      * velocity / acceleration / jerk smoothness penalties identical
        to ``solve_autoflip_lp`` (same default weights from the
        AutoFlip paper, plus our 2× velocity bias for stickier holds),
      * one ``outside_amount`` slack per optional region, summed with
        ``lam_soft * weight`` into the objective.

    Returns a :class:`MultiRegionLPResult`. The caller should:
      1. Check ``status``. ``"feasible"`` means use the camera path.
      2. ``"infeasible"`` means at least one frame's required regions
         can't fit in one crop — call ``decide_multi_region_layout``
         to pick SPLIT_SCREEN vs WIDE_MASTER.
      3. ``"lp_failed"`` is a HiGHS error — log and fall back.
    """
    import time as _time

    n = len(required_per_frame)
    if optional_per_frame is None:
        optional_per_frame = [[] for _ in range(n)]
    if len(optional_per_frame) != n:
        raise ValueError(
            f"optional_per_frame length {len(optional_per_frame)} != "
            f"required_per_frame length {n}"
        )
    if crop_width_px <= 0 or source_width_px <= 0:
        raise ValueError("crop_width_px and source_width_px must be positive")
    if crop_width_px > source_width_px + 1e-6:
        raise ValueError(
            f"crop_width_px ({crop_width_px}) > source_width_px "
            f"({source_width_px})"
        )

    n_required = sum(len(f) for f in required_per_frame)
    n_optional = sum(len(f) for f in optional_per_frame)

    if n == 0:
        return MultiRegionLPResult(
            status="feasible",
            camera_path=[],
            n_frames=0,
            n_required=0,
            n_optional=0,
            lp_message="empty input",
        )

    from backend.services.multi_region_layout import (
        per_frame_bounds_from_required as _per_frame_bounds_from_required,
    )

    half_crop = crop_width_px / 2.0
    lo_bounds, hi_bounds, infeasible_idx = _per_frame_bounds_from_required(
        required_per_frame, half_crop, source_width_px,
    )
    infeasibility_ratio = len(infeasible_idx) / max(n, 1)

    # Hard infeasibility: bail out so the layout decider can pick a
    # split / wide / blur fallback. The LP would still produce a
    # solution by clamping to the degenerate midpoint, but its camera
    # path would be misleading — the caller should NOT use it.
    if infeasible_idx:
        return MultiRegionLPResult(
            status="infeasible",
            camera_path=[],
            infeasible_frames=infeasible_idx,
            infeasibility_ratio=infeasibility_ratio,
            n_frames=n,
            n_required=n_required,
            n_optional=n_optional,
            lp_message=(
                f"{len(infeasible_idx)}/{n} frame(s) have required regions "
                f"that cannot fit in one crop (crop_width_px={crop_width_px})"
            ),
        )

    # ── Per-frame data-fidelity targets ──
    # The LP smoothness terms only see differences, so without an
    # absolute pull the camera could drift to any feasible position.
    # We anchor it to the midpoint of each frame's feasible interval
    # so when both sides have headroom the camera sits between the
    # speakers (the natural composition).
    targets = [0.5 * (lo_bounds[i] + hi_bounds[i]) for i in range(n)]

    # ── Single-frame degenerate case: just pick the midpoint ──
    if n == 1:
        return MultiRegionLPResult(
            status="feasible",
            camera_path=[float(targets[0])],
            n_frames=1,
            n_required=n_required,
            n_optional=n_optional,
            lp_message="n=1 trivial",
        )

    # ── Variable layout ──
    # cam[0..n-1]                       : n
    # s1[0..n-1]   (data fidelity)      : n
    # s2[0..n-2]   (velocity)           : n-1
    # s3[0..n-3]   (accel)              : n-2
    # s4[0..n-4]   (jerk)               : n-3
    # opt_slack[0..K-1]                 : K = n_optional
    n_s2 = max(n - 1, 0)
    n_s3 = max(n - 2, 0)
    n_s4 = max(n - 3, 0)
    n_opt = n_optional
    n_vars = n + n + n_s2 + n_s3 + n_s4 + n_opt

    off_cam = 0
    off_s1 = n
    off_s2 = off_s1 + n
    off_s3 = off_s2 + n_s2
    off_s4 = off_s3 + n_s3
    off_opt = off_s4 + n_s4

    # ── Objective ──
    c = np.zeros(n_vars)
    c[off_s1:off_s1 + n] = lam_data
    c[off_s2:off_s2 + n_s2] = lam_velocity
    c[off_s3:off_s3 + n_s3] = lam_accel
    c[off_s4:off_s4 + n_s4] = lam_jerk
    # Optional slack weights are filled in below as we walk the
    # optional regions to assign each its variable index.

    # ── Inequality constraints ──
    # Counts:
    #   2n     (s1 |cam-target|)
    #   2(n-1) (s2 velocity)
    #   2(n-2) (s3 accel)
    #   2(n-3) (s4 jerk)
    #   2K     (optional region soft slack)
    n_constraints = 2 * n + 2 * n_s2 + 2 * n_s3 + 2 * n_s4 + 2 * n_opt
    A = lil_matrix((n_constraints, n_vars))
    b = np.zeros(n_constraints)
    row = 0

    # s1: |cam[t] - target[t]| <= s1[t]
    for t in range(n):
        A[row, off_cam + t] = 1.0
        A[row, off_s1 + t] = -1.0
        b[row] = targets[t]
        row += 1
        A[row, off_cam + t] = -1.0
        A[row, off_s1 + t] = -1.0
        b[row] = -targets[t]
        row += 1

    # s2: velocity smoothness
    for t in range(n_s2):
        A[row, off_cam + t + 1] = 1.0
        A[row, off_cam + t] = -1.0
        A[row, off_s2 + t] = -1.0
        b[row] = 0.0
        row += 1
        A[row, off_cam + t] = 1.0
        A[row, off_cam + t + 1] = -1.0
        A[row, off_s2 + t] = -1.0
        b[row] = 0.0
        row += 1

    # s3: acceleration smoothness
    for t in range(n_s3):
        A[row, off_cam + t + 2] = 1.0
        A[row, off_cam + t + 1] = -2.0
        A[row, off_cam + t] = 1.0
        A[row, off_s3 + t] = -1.0
        b[row] = 0.0
        row += 1
        A[row, off_cam + t + 2] = -1.0
        A[row, off_cam + t + 1] = 2.0
        A[row, off_cam + t] = -1.0
        A[row, off_s3 + t] = -1.0
        b[row] = 0.0
        row += 1

    # s4: jerk smoothness
    for t in range(n_s4):
        A[row, off_cam + t + 3] = 1.0
        A[row, off_cam + t + 2] = -3.0
        A[row, off_cam + t + 1] = 3.0
        A[row, off_cam + t] = -1.0
        A[row, off_s4 + t] = -1.0
        b[row] = 0.0
        row += 1
        A[row, off_cam + t + 3] = -1.0
        A[row, off_cam + t + 2] = 3.0
        A[row, off_cam + t + 1] = -3.0
        A[row, off_cam + t] = 1.0
        A[row, off_s4 + t] = -1.0
        b[row] = 0.0
        row += 1

    # ── Optional region soft penalty ──
    # For each (frame t, region (l, r, w)):
    #   amount_outside = max(0, (cam[t] - half) - l, r - (cam[t] + half))
    # Linearize via slack o[k] >= 0 and:
    #   o[k] >= (cam[t] - half) - l    →  cam[t] - o[k] <= half + l
    #   o[k] >= r - (cam[t] + half)    → -cam[t] - o[k] <= half - r
    opt_idx = 0
    for t, regions in enumerate(optional_per_frame):
        for entry in regions:
            if len(entry) < 3:
                # Allow (left, right) with default weight 1.0 for callers
                # that don't care about per-region importance.
                left, right = float(entry[0]), float(entry[1])
                weight = 1.0
            else:
                left, right, weight = float(entry[0]), float(entry[1]), float(entry[2])
            opt_var = off_opt + opt_idx
            # cam[t] - o[k] <= half_crop + left
            A[row, off_cam + t] = 1.0
            A[row, opt_var] = -1.0
            b[row] = half_crop + left
            row += 1
            # -cam[t] - o[k] <= half_crop - right
            A[row, off_cam + t] = -1.0
            A[row, opt_var] = -1.0
            b[row] = half_crop - right
            row += 1
            c[opt_var] = lam_soft * max(weight, 0.0)
            opt_idx += 1

    A = A.tocsc()

    # ── Variable bounds ──
    bounds: list[tuple] = []
    for t in range(n):
        bounds.append((lo_bounds[t], hi_bounds[t]))  # cam[t]
    for _ in range(n):
        bounds.append((0.0, None))  # s1
    for _ in range(n_s2):
        bounds.append((0.0, None))  # s2
    for _ in range(n_s3):
        bounds.append((0.0, None))  # s3
    for _ in range(n_s4):
        bounds.append((0.0, None))  # s4
    for _ in range(n_opt):
        bounds.append((0.0, None))  # optional slack

    # ── Solve ──
    _t0 = _time.perf_counter()
    result = linprog(
        c, A_ub=A, b_ub=b, bounds=bounds,
        method="highs",
        options={"presolve": True, "time_limit": float(time_limit_sec)},
    )
    solve_ms = (_time.perf_counter() - _t0) * 1000.0

    if not result.success:
        logger.warning(
            "Multi-region LP failed: %s — n=%d, n_required=%d, n_optional=%d",
            result.message, n, n_required, n_optional,
        )
        return MultiRegionLPResult(
            status="lp_failed",
            camera_path=[],
            infeasible_frames=[],
            infeasibility_ratio=0.0,
            n_frames=n,
            n_required=n_required,
            n_optional=n_optional,
            lp_message=str(result.message),
            solve_ms=solve_ms,
        )

    cam = result.x[off_cam:off_cam + n].tolist()
    return MultiRegionLPResult(
        status="feasible",
        camera_path=cam,
        infeasible_frames=[],
        infeasibility_ratio=0.0,
        n_frames=n,
        n_required=n_required,
        n_optional=n_optional,
        lp_message=str(result.message),
        solve_ms=solve_ms,
    )


# ── Phase 2 (human reframe): 2-D LP convenience ────────────────────


def solve_autoflip_lp_2d(
    targets_x: list[float],
    targets_y: list[float],
    lo_bounds_x: list[float],
    hi_bounds_x: list[float],
    lo_bounds_y: list[float],
    hi_bounds_y: list[float],
    lam1: float = 1.0,
    lam2_x: float = 20.0,
    lam2_y: float = 40.0,
    lam3_x: float = 100.0,
    lam3_y: float = 200.0,
    lam4_x: float = 100.0,
    lam4_y: float = 200.0,
    *,
    weights: list[float] | None = None,
) -> tuple[list[float], list[float]]:
    """Solve x and y camera paths independently with per-axis smoothness.

    The AutoFlip paper's LP separates additively over axes when there is
    no cross-axis coupling in the cost function, so running two 1-D LPs
    back-to-back is mathematically identical to a block-diagonal 2-D LP
    and is 2× faster for the same solution. Y-axis smoothness weights
    are typically higher than X's because a human camera op moves the
    tilt axis much more slowly than pan.

    Returns ``(cam_x, cam_y)``.
    """
    cam_x = solve_autoflip_lp(
        targets_x, lo_bounds_x, hi_bounds_x,
        lam1=lam1, lam2=lam2_x, lam3=lam3_x, lam4=lam4_x,
        weights=weights,
    )
    cam_y = solve_autoflip_lp(
        targets_y, lo_bounds_y, hi_bounds_y,
        lam1=lam1, lam2=lam2_y, lam3=lam3_y, lam4=lam4_y,
        weights=weights,
    )
    return cam_x, cam_y
