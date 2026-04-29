"""L1-optimal camera path solver for AutoFlip-quality reframing.

Solves a 1D total-variation denoising problem per segment to produce
"hold still, snap, hold still" or constant-velocity pan motion — the
signature of professional camera operation.

    min  (1/2) sum (cam[t] - target[t])^2  +  lambda * sum |cam[t] - cam[t-1]|
    s.t. min_x[t] <= cam[t] <= max_x[t]   (hard constraints from must_be_in_frame features)

Uses an exact dual proximal gradient TV solver (Chambolle 2004 / Condat 2013)
with per-solve box constraint projection. No iterative approximation —
produces exact piecewise-constant paths with zero residual tilt on holds.

Per-segment mode selection from the solved path:
  STATIONARY: max(path) - min(path) < threshold * source_width  -> static center
  TRACKING:   otherwise -> emit the L1 path as motion_path
  PANNING:    near-linear path (R^2 > 0.95) and |slope| > threshold -> sweep
"""

import csv
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Debug telemetry: set CLIPAI_DUMP_L1=1 to write per-call CSVs to /tmp/clipai_l1/
_DUMP_L1 = os.environ.get("CLIPAI_DUMP_L1", "0") in ("1", "true", "yes")
_DUMP_DIR = Path("/tmp/clipai_l1")
_dump_counter = 0

# ── Solver selection ──
# CLIPAI_L1_SOLVER controls which solver is used:
#   "auto"   (default) — LP for shots <= LP_MAX_FRAMES, Condat above.
#   "lp"     — force LP; error if shot exceeds LP_MAX_FRAMES * 2.
#   "condat" — force Condat (legacy default, kept as escape hatch).
#
# Deprecated: CLIPAI_L1_LP=1 maps to "lp" with a DeprecationWarning.
_DEPRECATED_LP_FLAG = os.environ.get("CLIPAI_L1_LP", "0") in ("1", "true", "yes")
_SOLVER_MODE = os.environ.get("CLIPAI_L1_SOLVER", "auto").lower()
if _DEPRECATED_LP_FLAG and _SOLVER_MODE == "auto":
    _SOLVER_MODE = "lp"
    warnings.warn(
        "CLIPAI_L1_LP is deprecated; use CLIPAI_L1_SOLVER=lp instead.",
        DeprecationWarning,
        stacklevel=1,
    )
# Keep USE_LP_SOLVER for any downstream readers (read-only alias)
USE_LP_SOLVER = _SOLVER_MODE in ("lp", "auto")

# Performance guardrail: LP is O(n³) worst case. Fall back to Condat for
# shots longer than this many frames (30s at 30fps = 900).
LP_MAX_FRAMES = 900

# LP cost weights (Grundmann et al. 2011, Section 4.2).
# λ₁ = data fidelity (always 1.0), λ₂ = velocity, λ₃ = acceleration, λ₄ = jerk.
# Paper defaults: (10, 100, 100). We use λ₂=20 (2× paper) to bias toward
# holding still — face-keypoint noise in our pipeline is noisier than
# AutoFlip's feature tracks, so the solver needs more velocity resistance
# to avoid chasing jitter and to hold on subjects longer.
LP_LAMBDA_V = 20.0    # λ₂: velocity penalty (paper: 10, ours: 20 for stickier holds)
LP_LAMBDA_A = 100.0   # λ₃: acceleration penalty
LP_LAMBDA_J = 100.0   # λ₄: jerk penalty

# TV denoise regularization: fraction of source_width for resolution independence.
# Actual lambda = TV_LAMBDA_FRAC * source_width. 0.02 * 1920 = 38.4
# Higher = smoother/stickier holds. 0.02 produces longer holds than 0.015
# while still snapping cleanly to new subjects.
TV_LAMBDA_FRAC = 0.02
# Legacy constant (kept for backward-compat callers passing lam= explicitly)
TV_LAMBDA = 10.0
# Dead-zone: ignore target jitter smaller than this fraction of source_width.
# 0.018 * 1920 ≈ 35px — covers face-keypoint noise (which can be 20-30px
# frame-to-frame) without masking genuine subject motion.
DEADZONE_FRAC = 0.018
# Stationary threshold: if total movement < this fraction of source width, static crop
# 0.08 = ~154px on 1920 — covers normal face-detection noise without triggering
STATIONARY_THRESHOLD = 0.08

# Phase 4: per-frame face-unary weight multiplier. When an animated
# content profile + dense face anchors (density ≥ 40%) are available
# we override the slot-center unary with the actual per-frame nose_x
# and scale its data-fidelity weight by this factor relative to the
# slot-center fallback (which keeps weight 1.0). 1.5 is enough to pull
# the L1 solver onto the real face position within close-up shots
# without overwhelming the velocity / accel terms.
W_FACE_UNARY = float(os.environ.get("CLIPAI_W_FACE_UNARY", "1.5"))

# Phase 4: density floor for swapping to per-frame face anchors.
# Below this fraction of frames having a real face, the per-frame
# signal would be more noise than signal — fall back to slot-center.
FACE_ANCHOR_DENSITY_FLOOR = float(
    os.environ.get("CLIPAI_FACE_ANCHOR_DENSITY", "0.4")
)

# ── Phase 3 (gaming): pan-and-recenter knobs ──
#
# For a GamingEvent at time T with duration D and target_region
# (x_pct, y_pct, w, h), we ramp the data-fidelity weight on the
# target region for ``T - GAMING_PAN_RAMP_SEC`` to ``T``, hold it
# for the event duration, then ramp back. The action anchor's
# weight is scaled DOWN to ``GAMING_PAN_ACTION_HOLD_SCALE`` during
# the hold so the pan target wins the L1 fight.
GAMING_PAN_WEIGHT_RATIO = float(
    os.environ.get("GAMING_PAN_WEIGHT_RATIO", "2.0"),
)
GAMING_PAN_HOLD_SEC = float(
    os.environ.get("GAMING_PAN_HOLD_SEC", "0.8"),
)
GAMING_PAN_RAMP_SEC = float(
    os.environ.get("GAMING_PAN_RAMP_SEC", "0.3"),
)
GAMING_PAN_ACTION_HOLD_SCALE = float(
    os.environ.get("GAMING_PAN_ACTION_HOLD_SCALE", "0.3"),
)

# ── Phase 3a (gaming): center-anchor unary weights ──
#
# Center bias is the safety net for everything else in the gaming
# reframe pipeline. A strong per-frame center anchor is ALWAYS
# present for shots in ``CENTER_BIAS_GENRES`` — when the crosshair
# tracker is confident, a stronger crosshair unary wins; when the
# crosshair is missing, only the center unary fires and the solver
# locks to x = source_width / 2.  Saliency, motion, and
# face-cluster anchors are explicitly suppressed for these genres
# (they're the source of the TF2 off-center regression). Tunable
# via env vars but the defaults match the spec.
GAMING_CENTER_WEIGHT_DEFAULT = float(
    os.environ.get("GAMING_CENTER_WEIGHT_DEFAULT", "1.5"),
)
GAMING_CENTER_WEIGHT_ACTIVE = float(
    os.environ.get("GAMING_CENTER_WEIGHT_ACTIVE", "0.4"),
)
GAMING_CROSSHAIR_WEIGHT = float(
    os.environ.get("GAMING_CROSSHAIR_WEIGHT", "2.0"),
)
GAMING_PAN_WEIGHT = float(
    os.environ.get("GAMING_PAN_WEIGHT", "3.0"),
)
GAMING_CROSSHAIR_CONF_THRESHOLD = float(
    os.environ.get("GAMING_CROSSHAIR_CONF_THRESHOLD", "0.6"),
)

# Gameplay_subtype values that qualify for the hard center bias.
# MOBA / TPS / racing are deliberately excluded — their action
# genuinely moves off center (lane fights, third-person character
# offset, car in lower third) so they keep the existing
# slot-center / motion-centroid / character-tracker anchor chain.
CENTER_BIAS_GENRES = frozenset({
    "fps",
    "gameplay_fps",
    "hero_shooter",
    "sandbox",
    "gameplay",
})


def apply_glance_deviation_clamp(
    solved_path: list[tuple[float, float]],
    *,
    source_width: int,
    crop_width: float,
    max_glance_deviation_pct: float = 0.65,
) -> list[tuple[float, float]]:
    """Hard-clamp a solved L1 camera path to the gaming mode's
    maximum horizontal glance deviation.

    Phase 4 — gaming mode adds a mandatory asymmetric clamp on
    top of whatever pan the solver produces: no frame's crop
    center may deviate from the horizontal source center by
    more than ``max_glance_deviation_pct * max_pan``, where
    ``max_pan = (source_width - crop_width) / 2``. Keeps at
    least part of the actual gameplay visible during HUD
    glances — without the clamp, a kill-feed glance on a 16:9
    source could slide the 9:16 crop fully off the action.

    The clamp is applied after the TV / LP solve so it
    composes with the existing ``compute_hard_bounds`` soft
    box-constraint machinery rather than fighting it. Callers
    that want a hard constraint baked into the solve itself
    can alternatively plumb ``max_glance_deviation_pct`` into
    the ``lo_bounds`` / ``hi_bounds`` arguments of
    :func:`solve_camera_path`.

    Args:
        solved_path: List of ``(timestamp, x_pixel)`` pairs
            from :func:`solve_camera_path` / its shot variant.
        source_width: Source frame width in pixels.
        crop_width: Target crop width in pixels (e.g. for
            9:16 from 1920×1080 this is 607.5).
        max_glance_deviation_pct: Fraction of the available
            pan range that a glance may traverse. Must be in
            ``[0, 1]``. Default 0.65 matches the spec.

    Returns:
        A new path with the same timestamps, where each x
        value is clamped into
        ``[center - pct*max_pan, center + pct*max_pan]``.
    """
    if not solved_path:
        return []
    if not 0.0 <= max_glance_deviation_pct <= 1.0:
        raise ValueError(
            f"max_glance_deviation_pct must be in [0, 1]; got "
            f"{max_glance_deviation_pct}"
        )
    src_w = float(source_width)
    crop_w = float(crop_width)
    if crop_w >= src_w:
        # Nothing to clamp — crop already covers the whole width.
        return list(solved_path)
    center = src_w / 2.0
    max_pan = (src_w - crop_w) / 2.0
    max_dev = float(max_glance_deviation_pct) * max_pan
    lo = center - max_dev
    hi = center + max_dev
    out: list[tuple[float, float]] = []
    for t, x in solved_path:
        xc = float(x)
        if xc < lo:
            xc = lo
        elif xc > hi:
            xc = hi
        out.append((float(t), xc))
    return out


def center_distance_cost(
    path: list[tuple[float, float]],
    *,
    source_width: int,
    crop_width: float,
    lambda_center: float = 0.4,
) -> float:
    """Compute the asymmetric center-distance cost for a path.

    Phase 4 term: ``Σ_t lambda_center * ((x_t - x_center) /
    max_pan)^2``. Used in tests to verify that a centered
    solution has strictly lower cost than an off-center one
    for the same set of saliency targets. Production callers
    fold the term into the L1 objective via the center-anchor
    unary — this helper exists so the tunable can be
    evaluated as a standalone scalar.
    """
    if not path:
        return 0.0
    src_w = float(source_width)
    crop_w = float(crop_width)
    if crop_w >= src_w:
        return 0.0
    center = src_w / 2.0
    max_pan = (src_w - crop_w) / 2.0
    total = 0.0
    for _t, x in path:
        delta = (float(x) - center) / max(max_pan, 1e-6)
        total += lambda_center * delta * delta
    return float(total)


def build_face_anchor_targets(
    dense_faces: list,
    slot_id: Optional[int],
    slot_center_pct: float,
    start: float,
    end: float,
    *,
    target_fps: float = 30.0,
    is_animated: bool = False,
    face_unary_weight: float = W_FACE_UNARY,
    density_floor: float = FACE_ANCHOR_DENSITY_FLOOR,
) -> tuple[list[tuple[float, float]], list[float], bool]:
    """Build a uniform-fps per-frame anchor signal + weights for the L1 solver.

    Phase 4 — animated content with dense face data switches the unary
    anchor from the static slot-center to the actual per-frame
    ``dense_face.nose_x``. Frames with a real face anchor get
    ``face_unary_weight`` (default 1.5×); frames with no face anchor
    fall back to ``slot_center_pct`` with weight 1.0.

    The swap is gated behind:

      1. ``is_animated`` is True (caller passes
         ``_content_profile.is_animated``)
      2. The per-shot density of frames-with-face for the requested slot
         is at least ``density_floor`` (default 0.4)

    When either gate fails the helper returns the legacy slot-center
    signal at unit weights AND ``False`` for ``used_face_anchors`` so
    the caller can log/branch on that.

    Args:
        dense_faces: list of FrameFaces objects (one per dense sample).
        slot_id: face registry slot to track. ``None`` = any face.
        slot_center_pct: fallback x-position in percent.
        start, end: shot time range in seconds.
        target_fps: uniform sampling rate.
        is_animated: anime/cartoon gate.
        face_unary_weight: weight multiplier for face anchor frames.
        density_floor: min frames-with-face ratio to enable the swap.

    Returns:
        ``(positions, weights, used_face_anchors)`` where ``positions``
        is a list of ``(timestamp, x_pct)`` pairs at uniform spacing
        and ``weights`` is parallel — ``face_unary_weight`` for real
        face frames, ``1.0`` for slot-center fallback frames.
    """
    if end <= start:
        return [], [], False

    dt = 1.0 / target_fps
    n_frames = max(1, int((end - start) * target_fps))

    # Step 1: collect per-frame face nose_x for the requested slot.
    face_x_by_t: dict[float, float] = {}
    if dense_faces:
        for df in dense_faces:
            ts = float(getattr(df, "timestamp", -1.0))
            if ts < start or ts >= end:
                continue
            faces = getattr(df, "faces", []) or []
            chosen_x: Optional[float] = None
            for face in faces:
                sid = getattr(face, "identity_id", -1)
                if slot_id is not None and sid != slot_id:
                    continue
                nx = getattr(face, "nose_x", None)
                if nx is None:
                    continue
                chosen_x = float(nx)
                break
            if chosen_x is None and slot_id is None and faces:
                nx = getattr(faces[0], "nose_x", None)
                if nx is not None:
                    chosen_x = float(nx)
            if chosen_x is not None:
                face_x_by_t[round(ts, 4)] = chosen_x

    # Step 2: compute density (fraction of dense samples in [start,end)
    # that contributed a face_x). If we don't have ANY face data we
    # implicitly hit density 0.0 and skip the swap.
    density_window = [
        df for df in (dense_faces or [])
        if start <= float(getattr(df, "timestamp", -1.0)) < end
    ]
    n_window = max(len(density_window), 1)
    density = len(face_x_by_t) / n_window

    use_anchors = bool(
        is_animated
        and face_x_by_t
        and density >= density_floor
    )

    # Step 3: build the uniform-fps signal. For each frame timestamp
    # interpolate linearly between the two nearest face samples (so a
    # 1 Hz dense stream still drives the unary at 30 fps); when the
    # uniform timestamp is outside the face range, hold the nearest
    # endpoint. Frames within ``±max_gap`` of a real face sample carry
    # ``face_unary_weight``; outside that window we fall back to the
    # slot center at unit weight. When the gate is closed every frame
    # falls back to slot.
    positions: list[tuple[float, float]] = []
    weights: list[float] = []
    sorted_face_keys = sorted(face_x_by_t.keys()) if face_x_by_t else []
    # Extrapolation window: face data at production-default 1 Hz with
    # ±0.6s window covers every uniform-fps target between successive
    # face frames. Animated content tends to hold faces for ≥1s so
    # this is rarely the limiting factor.
    max_gap = 0.6
    raw_idx = 0

    for i in range(n_frames):
        t = start + i * dt
        if t >= end:
            break
        anchor_x: Optional[float] = None
        is_real_anchor = False
        if use_anchors and sorted_face_keys:
            # Bracket t between sorted_face_keys[raw_idx] and [raw_idx+1]
            while (
                raw_idx < len(sorted_face_keys) - 1
                and sorted_face_keys[raw_idx + 1] <= t
            ):
                raw_idx += 1

            t0 = sorted_face_keys[raw_idx]
            x0 = face_x_by_t[t0]
            if raw_idx >= len(sorted_face_keys) - 1:
                # Past the last face sample — hold last known if within gap.
                if abs(t - t0) <= max_gap:
                    anchor_x = x0
                    is_real_anchor = True
            elif sorted_face_keys[raw_idx] >= t:
                # Before the first relevant sample.
                if abs(t0 - t) <= max_gap:
                    anchor_x = x0
                    is_real_anchor = True
            else:
                t1 = sorted_face_keys[raw_idx + 1]
                x1 = face_x_by_t[t1]
                # Linear interpolation between bracketing face frames.
                # Allow it as long as t is "near" either bracket.
                if (t - t0) <= max_gap and (t1 - t) <= max_gap:
                    span = t1 - t0
                    alpha = (t - t0) / span if span > 0 else 0.0
                    anchor_x = x0 + alpha * (x1 - x0)
                    is_real_anchor = True

        if is_real_anchor and anchor_x is not None:
            positions.append((round(t, 6), float(anchor_x)))
            weights.append(float(face_unary_weight))
        else:
            positions.append((round(t, 6), float(slot_center_pct)))
            weights.append(1.0)

    return positions, weights, use_anchors
# Panning R^2 threshold for linear-fit detection (0.90 for pre-solve noisy data)
PANNING_R2_THRESHOLD = 0.90
# Minimum slope (pixels per second) to qualify as a pan
PANNING_MIN_SLOPE = 50.0


def _tv_denoise_1d(
    signal: list[float],
    lam: float,
    n_iter: int = 100,
    lo_bounds: Optional[list[float]] = None,
    hi_bounds: Optional[list[float]] = None,
) -> list[float]:
    """1D total-variation denoising via exact dual proximal gradient
    with box constraint projection.

    Produces a piecewise-constant approximation of the input signal,
    which gives "snap and hold" camera motion. Uses the exact solver
    from _condat_tv (Chambolle 2004 / Condat 2013 dual formulation)
    with a two-pass projection onto [lo_bounds, hi_bounds].

    Args:
        signal: Input 1D signal (target face positions over time).
        lam: Regularization weight. Higher = smoother.
        n_iter: Ignored (kept for backward compat). The solver
            runs to convergence automatically.
        lo_bounds: Per-element lower bounds (None = unconstrained).
        hi_bounds: Per-element upper bounds (None = unconstrained).

    Returns:
        Denoised signal of the same length.
    """
    from backend.services._condat_tv import condat_tv_l1

    n = len(signal)
    if n <= 1:
        return list(signal)

    # Pass 1: unconstrained exact TV denoise
    x = condat_tv_l1(signal, lam)

    # Pass 2: project onto box constraints and re-denoise
    if lo_bounds is not None and hi_bounds is not None:
        for i in range(n):
            if lo_bounds[i] is not None:
                x[i] = max(x[i], lo_bounds[i])
            if hi_bounds[i] is not None:
                x[i] = min(x[i], hi_bounds[i])
        # Second Condat pass on the projected signal to restore TV smoothness
        x = condat_tv_l1(x, lam)
        # Final projection to ensure feasibility
        for i in range(n):
            if lo_bounds[i] is not None:
                x[i] = max(x[i], lo_bounds[i])
            if hi_bounds[i] is not None:
                x[i] = min(x[i], hi_bounds[i])

    return x


def _linear_fit_r2(values: list[float]) -> tuple[float, float]:
    """Compute R^2 and slope of a linear fit to evenly-spaced values.

    Returns (r_squared, slope_per_step).
    """
    n = len(values)
    if n < 3:
        return 0.0, 0.0

    # Simple linear regression on indices
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n

    ss_xy = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    ss_xx = sum((i - x_mean) ** 2 for i in range(n))
    ss_yy = sum((v - y_mean) ** 2 for v in values)

    if ss_xx == 0 or ss_yy == 0:
        return 1.0 if ss_yy == 0 else 0.0, 0.0

    slope = ss_xy / ss_xx
    r_squared = (ss_xy ** 2) / (ss_xx * ss_yy)

    return r_squared, slope


def compute_hard_bounds(
    features: list,
    timestamps: list[float],
    source_width: int,
    crop_aspect: float = 9 / 16,
    source_height: int = 1080,
) -> tuple[list[Optional[float]], list[Optional[float]], list[bool]]:
    """Compute per-frame hard bounds from must_be_in_frame features.

    For each timestamp, computes the union bbox of all must_be_in_frame=True
    features active at that time. The crop window centered at cam[t] must
    contain this union bbox.

    Args:
        features: list[RequiredFeature] with must_be_in_frame, left, right, t_start
        timestamps: sorted list of frame timestamps
        source_width: source video width in pixels
        crop_aspect: target aspect ratio (width/height)
        source_height: source video height in pixels

    Returns:
        (lo_bounds, hi_bounds, infeasible) — per-timestamp lower/upper bounds
        on the crop center x (in pixels), and a boolean flag per frame for
        infeasible frames where the union bbox exceeds crop width.
    """
    src_aspect = source_width / source_height if source_height > 0 else 16 / 9
    if crop_aspect < src_aspect:
        crop_width_px = int(source_height * crop_aspect)
    else:
        crop_width_px = source_width
    half_crop = crop_width_px / 2.0

    # Filter to hard-required features
    hard = [f for f in features if getattr(f, 'must_be_in_frame', False)]
    if not hard:
        return (
            [None] * len(timestamps),
            [None] * len(timestamps),
            [False] * len(timestamps),
        )

    lo_bounds = []
    hi_bounds = []
    infeasible = []

    for t in timestamps:
        # Find all hard features active at this timestamp
        active = [f for f in hard if abs(f.t_start - t) < 0.05]
        if not active:
            lo_bounds.append(None)
            hi_bounds.append(None)
            infeasible.append(False)
            continue

        # Union bbox in percentage space
        union_left_pct = min(f.left for f in active)
        union_right_pct = max(f.right for f in active)

        # Convert to pixel space
        union_left_px = union_left_pct / 100.0 * source_width
        union_right_px = union_right_pct / 100.0 * source_width
        union_width_px = union_right_px - union_left_px

        if union_width_px > crop_width_px:
            # Infeasible: union bbox wider than crop
            lo_bounds.append(None)
            hi_bounds.append(None)
            infeasible.append(True)
            continue

        # The crop [cam_x - half_crop, cam_x + half_crop] must contain
        # [union_left_px, union_right_px].
        # => cam_x >= union_right_px - half_crop  (right edge of union must be inside crop)
        # => cam_x <= union_left_px + half_crop   (left edge of union must be inside crop)
        lo = union_right_px - half_crop
        hi = union_left_px + half_crop

        # Also clamp to valid range [half_crop, source_width - half_crop]
        lo = max(half_crop, lo)
        hi = min(source_width - half_crop, hi)

        lo_bounds.append(lo)
        hi_bounds.append(hi)
        infeasible.append(False)

    return lo_bounds, hi_bounds, infeasible


def assert_required_in_frame(
    path: list[tuple[float, float]],
    features: list,
    source_width: int,
    crop_aspect: float = 9 / 16,
    source_height: int = 1080,
    tolerance_px: float = 2.0,
) -> list[tuple[float, list]]:
    """Verify that all must_be_in_frame features are covered by the solved path.

    Returns a list of (timestamp, [offending_feature_identities]) for frames
    where the crop does NOT contain all required features. Empty list = all good.
    """
    src_aspect = source_width / source_height if source_height > 0 else 16 / 9
    if crop_aspect < src_aspect:
        crop_width_px = int(source_height * crop_aspect)
    else:
        crop_width_px = source_width
    half_crop = crop_width_px / 2.0

    hard = [f for f in features if getattr(f, 'must_be_in_frame', False)]
    if not hard:
        return []

    violations = []
    for t, cam_x in path:
        crop_left = cam_x - half_crop
        crop_right = cam_x + half_crop

        active = [f for f in hard if abs(f.t_start - t) < 0.05]
        offenders = []
        for f in active:
            f_left_px = f.left / 100.0 * source_width
            f_right_px = f.right / 100.0 * source_width
            if f_left_px < crop_left - tolerance_px or f_right_px > crop_right + tolerance_px:
                offenders.append(getattr(f, 'identity', None))
        if offenders:
            violations.append((t, offenders))

    return violations


def _post_solve_smooth(solved: list[float], alpha: float = 0.15) -> list[float]:
    """Light bidirectional exponential smoothing to kill residual jitter.

    Runs a forward EMA then a backward EMA and averages them (Holt-style
    symmetric smoothing). This preserves transition timing while suppressing
    sub-pixel wiggles left by the LP/Condat solver.

    Args:
        solved: Solved camera path from TV/LP.
        alpha: Smoothing factor (0 = no smoothing, 1 = no smoothing).
            0.15 kills 1-2px jitter without visibly delaying transitions.

    Returns:
        Smoothed path of the same length.
    """
    if len(solved) < 3 or alpha >= 1.0:
        return list(solved)

    n = len(solved)
    # Forward pass
    fwd = [0.0] * n
    fwd[0] = solved[0]
    for i in range(1, n):
        fwd[i] = alpha * solved[i] + (1 - alpha) * fwd[i - 1]

    # Backward pass
    bwd = [0.0] * n
    bwd[n - 1] = solved[n - 1]
    for i in range(n - 2, -1, -1):
        bwd[i] = alpha * solved[i] + (1 - alpha) * bwd[i + 1]

    # Average — symmetric, so transitions aren't shifted in time
    return [(f + b) / 2.0 for f, b in zip(fwd, bwd)]


def _apply_condat_weights(
    targets: list[float],
    weights: list[float],
    lam: float,
) -> list[float]:
    """Pre-weight signal for diagonal data-fidelity weighting with Condat TV.

    For the L2-TV problem:  min sum w[t]*(cam[t]-target[t])^2 + lam*TV(cam)
    This is equivalent to solving with adjusted targets when weights differ.
    The standard trick: stronger weight pulls the solution closer to the target
    at that frame. We scale the effective lambda inversely with weight so that
    high-weight frames get lower regularization relative to data fidelity.

    For simplicity, we return a weighted signal: when all weights are 1.0,
    the output is identical to the input (bit-identical backward compat).
    """
    # When weight > 1, the target pulls harder; < 1, pulls weaker.
    # We implement this by duplicating the signal but biasing toward
    # a running mean for low-weight frames (effectively softening their pull).
    if not weights:
        return list(targets)

    # Compute the global mean as the "neutral" position
    mean_x = sum(targets) / len(targets) if targets else 0.0
    result = []
    for i, (t, w) in enumerate(zip(targets, weights)):
        # Blend between target and global mean based on weight
        # w=1.0 → pure target, w→0 → pulls toward mean (less influence)
        result.append(w * t + (1.0 - w) * mean_x)
    return result


def _apply_deadzone(targets: list[float], deadzone_px: float) -> list[float]:
    """Replace target values within deadzone of a running hold with the hold value.

    Uses a running median over the previous 1.0s (30 samples at 30fps) as
    the hold reference. If a target is within deadzone_px of the hold,
    it's replaced by the hold value — this suppresses keypoint jitter
    without masking genuine subject motion.

    The 1s window keeps the hold reference stable during stationary shots,
    preventing the camera from drifting with noise. Genuine motion (subject
    walks, speaker switch) produces deltas >> deadzone_px and breaks through.
    """
    if not targets or deadzone_px <= 0:
        return list(targets)

    result = list(targets)
    window = 30  # ~1.0s at 30fps — long enough to anchor holds
    hold = targets[0]

    for i in range(len(targets)):
        # Update hold as running median of recent window
        start = max(0, i - window)
        recent = sorted(targets[start:i + 1])
        hold = recent[len(recent) // 2]

        if abs(targets[i] - hold) < deadzone_px:
            result[i] = hold

    return result


def _dump_solve_csv(
    job_id: str, seg_idx: int,
    times: list[float], targets: list[float], solved: list[float],
    lo_bounds: Optional[list[float]], hi_bounds: Optional[list[float]],
    *,
    solver: str = "condat",
    weights: Optional[list[float]] = None,
) -> None:
    """Write a per-call CSV for telemetry analysis (behind CLIPAI_DUMP_L1=1)."""
    global _dump_counter
    try:
        _DUMP_DIR.mkdir(parents=True, exist_ok=True)
        jid = job_id or "unknown"
        fname = f"{jid}_{seg_idx:04d}.csv"
        path = _DUMP_DIR / fname
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "target_x", "solved_x", "lo_bound", "hi_bound", "solver", "weight"])
            for i, t in enumerate(times):
                lo = lo_bounds[i] if lo_bounds and lo_bounds[i] is not None else ""
                hi = hi_bounds[i] if hi_bounds and hi_bounds[i] is not None else ""
                w = weights[i] if weights else 1.0
                writer.writerow([f"{t:.4f}", f"{targets[i]:.2f}", f"{solved[i]:.2f}",
                                 lo, hi, solver, f"{w:.4f}"])
        _dump_counter += 1
        logger.debug("L1 telemetry: wrote %s (%d rows)", path, len(times))
    except Exception as e:
        logger.warning("L1 telemetry dump failed: %s", e)


def _solve_camera_path_with_discontinuities(
    *,
    face_positions: list[tuple[float, float]],
    source_width: int,
    lam: float,
    hard_features: Optional[list],
    source_height: int,
    crop_aspect: float,
    job_id: str,
    seg_idx: int,
    weights: Optional[list[float]],
    discontinuity_marks: list[float],
) -> dict:
    """Split ``face_positions`` at each mark, solve each piece alone.

    Sub-problems are solved by recursing into :func:`solve_camera_path`
    without ``discontinuity_marks`` so the recursion bottoms out. The
    LP / Condat solvers are smoothness-preserving, so the only way to
    get a hard cut at a mark is to refuse to let the solver see across
    the boundary — the whole point of the marks.

    Result aggregation: each sub-problem produces its own mode. The
    aggregate is returned as ``mode="tracking"`` with the concatenated
    path so downstream consumers (the segmenter's per-segment path
    consumer) treat the segment as a tracking move. ``center`` is the
    mean of the concatenated path; ``slope`` is 0 (panning at the
    aggregate level is meaningless across a cut). If any sub-problem
    is infeasible the overall result is infeasible.

    Marks are sorted and deduped within 50 ms (matches the dedupe
    window in :func:`apply_plan_to_lp_targets`).
    """
    if not face_positions:
        return solve_camera_path(
            face_positions=face_positions,
            source_width=source_width,
            lam=lam,
            hard_features=hard_features,
            source_height=source_height,
            crop_aspect=crop_aspect,
            job_id=job_id,
            seg_idx=seg_idx,
            weights=weights,
        )

    # Sort + dedupe marks within 50ms.
    marks_sorted = sorted(float(m) for m in discontinuity_marks)
    deduped: list[float] = []
    for m in marks_sorted:
        if not deduped or (m - deduped[-1]) > 0.05:
            deduped.append(m)

    if not deduped:
        return solve_camera_path(
            face_positions=face_positions,
            source_width=source_width,
            lam=lam,
            hard_features=hard_features,
            source_height=source_height,
            crop_aspect=crop_aspect,
            job_id=job_id,
            seg_idx=seg_idx,
            weights=weights,
        )

    # Sort positions + weights jointly by timestamp so the split is
    # well-defined.
    pairs: list[tuple[int, tuple[float, float]]] = list(enumerate(face_positions))
    pairs.sort(key=lambda kv: kv[1][0])
    sorted_idx = [k for k, _ in pairs]
    sorted_pos = [p for _, p in pairs]
    sorted_weights: Optional[list[float]] = None
    if weights is not None and len(weights) == len(face_positions):
        sorted_weights = [weights[k] for k in sorted_idx]

    # Build splits: each segment is [start_idx, end_idx_exclusive).
    splits: list[tuple[int, int]] = []
    cursor = 0
    for mark in deduped:
        # First index whose timestamp >= mark.
        boundary = cursor
        while boundary < len(sorted_pos) and sorted_pos[boundary][0] < mark:
            boundary += 1
        if boundary > cursor:
            splits.append((cursor, boundary))
        cursor = boundary
    if cursor < len(sorted_pos):
        splits.append((cursor, len(sorted_pos)))

    # Degenerate: all marks fell outside [t_min, t_max] → solve normally.
    if len(splits) <= 1:
        return solve_camera_path(
            face_positions=face_positions,
            source_width=source_width,
            lam=lam,
            hard_features=hard_features,
            source_height=source_height,
            crop_aspect=crop_aspect,
            job_id=job_id,
            seg_idx=seg_idx,
            weights=weights,
        )

    aggregate_path: list[tuple[float, float]] = []
    any_infeasible = False
    infeasible_frames: list = []
    for sub_idx, (a, b) in enumerate(splits):
        sub_pos = sorted_pos[a:b]
        sub_weights = sorted_weights[a:b] if sorted_weights is not None else None
        sub_result = solve_camera_path(
            face_positions=sub_pos,
            source_width=source_width,
            lam=lam,
            hard_features=hard_features,
            source_height=source_height,
            crop_aspect=crop_aspect,
            job_id=job_id,
            seg_idx=seg_idx * 100 + sub_idx,
            weights=sub_weights,
        )
        if sub_result.get("mode") == "infeasible":
            any_infeasible = True
            infeasible_frames.extend(sub_result.get("infeasible_frames", []))
            continue
        path = sub_result.get("path") or []
        if not path:
            center = float(sub_result.get("center", source_width / 2.0))
            path = [(t, center) for (t, _) in sub_pos]
        aggregate_path.extend(path)

    if any_infeasible and not aggregate_path:
        return {
            "mode": "infeasible",
            "center": source_width / 2.0,
            "path": [],
            "slope": 0.0,
            "ease_in_ms": 0,
            "infeasible_frames": infeasible_frames,
        }

    aggregate_path.sort(key=lambda p: p[0])
    if not aggregate_path:
        return {
            "mode": "stationary",
            "center": source_width / 2.0,
            "path": [],
            "slope": 0.0,
            "ease_in_ms": 0,
            "infeasible_frames": [],
        }
    mean_x = sum(x for _, x in aggregate_path) / len(aggregate_path)
    return {
        "mode": "tracking",
        "center": mean_x,
        "path": aggregate_path,
        "slope": 0.0,
        "ease_in_ms": 0,
        "infeasible_frames": infeasible_frames,
    }


def solve_camera_path(
    face_positions: list[tuple[float, float]],
    source_width: int = 1920,
    lam: float = TV_LAMBDA,
    hard_features: Optional[list] = None,
    source_height: int = 1080,
    crop_aspect: float = 9 / 16,
    job_id: str = "",
    seg_idx: int = 0,
    *,
    weights: Optional[list[float]] = None,
    discontinuity_marks: Optional[list[float]] = None,
) -> dict:
    """Solve L1-optimal camera path for a segment.

    Args:
        face_positions: [(timestamp, x_pixel), ...] dense face centroids.
        source_width: Source video width in pixels.
        lam: TV regularization weight.
        hard_features: list[RequiredFeature] for hard constraint enforcement.
            When provided, the solver projects cam[t] into the feasible
            interval at each iteration and checks feasibility afterward.
        source_height: Source video height in pixels.
        crop_aspect: Target crop aspect ratio (width/height).
        weights: Optional per-frame weights (same length as face_positions).
            When provided, scales the data-fidelity term per frame.
            None = all weights 1.0 (bit-identical to unweighted path).
        discontinuity_marks: Optional list of timestamps at which the
            solver is allowed to introduce a hard cut. The path is
            split at each mark into independent sub-problems whose
            solutions are concatenated without smoothing across the
            boundary — exactly what the editorial planner needs for
            ab_cut beats. Marks are sorted and deduped within 50 ms.

    Returns:
        {
            "mode": "stationary" | "tracking" | "panning" | "infeasible",
            "center": float (pixel) — for stationary mode,
            "path": [(t, x), ...] — for tracking mode,
            "slope": float — for panning mode (px/sec),
            "ease_in_ms": int — 0 for snap, >0 for smooth,
            "infeasible_frames": [(t, [identities]), ...] — if infeasible,
        }
    """
    if discontinuity_marks:
        return _solve_camera_path_with_discontinuities(
            face_positions=face_positions,
            source_width=source_width,
            lam=lam,
            hard_features=hard_features,
            source_height=source_height,
            crop_aspect=crop_aspect,
            job_id=job_id,
            seg_idx=seg_idx,
            weights=weights,
            discontinuity_marks=discontinuity_marks,
        )

    if not face_positions:
        return {
            "mode": "stationary",
            "center": source_width / 2.0,
            "path": [],
            "slope": 0.0,
            "ease_in_ms": 0,
            "infeasible_frames": [],
        }

    # Sort by time
    sorted_pos = sorted(face_positions, key=lambda p: p[0])
    times = [p[0] for p in sorted_pos]
    targets = [p[1] for p in sorted_pos]

    if len(targets) < 2:
        return {
            "mode": "stationary",
            "center": targets[0],
            "path": [],
            "slope": 0.0,
            "ease_in_ms": 0,
            "infeasible_frames": [],
        }

    # Compute hard bounds if features provided
    lo_bounds = None
    hi_bounds = None
    infeasible_frames = []

    if hard_features:
        lo_bounds, hi_bounds, infeasible_flags = compute_hard_bounds(
            hard_features, times, source_width, crop_aspect, source_height,
        )

        # Check for infeasible frames
        infeasible_times = [t for t, inf in zip(times, infeasible_flags) if inf]
        if infeasible_times:
            # Signal infeasibility — caller should promote to PADDED
            logger.warning(
                "L1 solver: %d/%d frames infeasible (required features exceed crop width)",
                len(infeasible_times), len(times),
            )
            return {
                "mode": "infeasible",
                "center": source_width / 2.0,
                "path": [],
                "slope": 0.0,
                "ease_in_ms": 0,
                "infeasible_frames": [(t, []) for t in infeasible_times],
            }

    # ── Resolution-independent lambda ──
    # If caller used the legacy default TV_LAMBDA=10.0, upgrade to fraction-based
    if lam == TV_LAMBDA:
        lam = TV_LAMBDA_FRAC * source_width  # 0.015 * 1920 = 28.8

    # ── Pre-solve dead-zone: suppress keypoint noise during holds ──
    deadzone_px = DEADZONE_FRAC * source_width  # ~25px at 1920
    targets_dz = _apply_deadzone(targets, deadzone_px)

    # ── Pre-solve panning detection on target signal ──
    # Detect panning BEFORE solving — the raw (dead-zoned) target is noisier
    # but won't be staircase-smoothed by TV, so a genuine linear pan has
    # higher R² here than in post-solve Condat output.
    if len(targets_dz) >= 3:
        _pre_r2, _pre_slope_step = _linear_fit_r2(targets_dz)
        _pre_dt = (times[-1] - times[0]) / max(len(times) - 1, 1)
        _pre_slope_pps = _pre_slope_step / max(_pre_dt, 0.001)

        if (_pre_r2 > PANNING_R2_THRESHOLD
                and abs(_pre_slope_pps) > PANNING_MIN_SLOPE):
            # Verify the linear fit stays within hard_bounds everywhere
            n_t = len(targets_dz)
            y_mean = sum(targets_dz) / n_t
            x_mean = (n_t - 1) / 2.0
            _fit_vals = [_pre_slope_step * (i - x_mean) + y_mean for i in range(n_t)]
            bounds_ok = True
            if lo_bounds is not None and hi_bounds is not None:
                for i in range(n_t):
                    lo_v = lo_bounds[i]
                    hi_v = hi_bounds[i]
                    if lo_v is not None and _fit_vals[i] < lo_v - 1.0:
                        bounds_ok = False
                        break
                    if hi_v is not None and _fit_vals[i] > hi_v + 1.0:
                        bounds_ok = False
                        break

            if bounds_ok:
                logger.info(
                    "L1 solver: shot=%d frames=%d mode=panning solver=pre_solve_linear solve_ms=0.0",
                    seg_idx, n_t,
                )
                return {
                    "mode": "panning",
                    "center": sum(_fit_vals) / len(_fit_vals),
                    "path": list(zip(times, _fit_vals)),
                    "slope": _pre_slope_pps,
                    "ease_in_ms": 0,
                    "infeasible_frames": [],
                }

    # ── Solver selection ──
    n_frames = len(targets_dz)
    if _SOLVER_MODE == "lp" and n_frames > LP_MAX_FRAMES * 2:
        raise ValueError(
            f"LP solver forced but shot has {n_frames} frames "
            f"(> {LP_MAX_FRAMES * 2} max). Use CLIPAI_L1_SOLVER=auto."
        )
    use_lp = _SOLVER_MODE == "lp" or (_SOLVER_MODE == "auto" and n_frames <= LP_MAX_FRAMES)
    _solver_used = "lp" if use_lp else "condat"
    _t0 = time.perf_counter()
    try:
        if use_lp:
            from backend.services._autoflip_lp import solve_autoflip_lp
            _lp_lo = [lo_bounds[i] if lo_bounds and lo_bounds[i] is not None else 0.0
                      for i in range(n_frames)]
            _lp_hi = [hi_bounds[i] if hi_bounds and hi_bounds[i] is not None else float(source_width)
                      for i in range(n_frames)]
            # Per-frame data-term weights: scale lam1 per frame in the LP
            _lp_weights = weights if weights is not None else None
            solved = solve_autoflip_lp(
                targets_dz, _lp_lo, _lp_hi,
                lam1=1.0, lam2=LP_LAMBDA_V, lam3=LP_LAMBDA_A, lam4=LP_LAMBDA_J,
                weights=_lp_weights,
            )
        else:
            # For Condat: pre-weight signal for diagonal data-fidelity weighting.
            # Replace target[t] with weighted version that pulls harder/softer.
            if weights is not None:
                _condat_input = _apply_condat_weights(targets_dz, weights, lam)
            else:
                _condat_input = targets_dz
            solved = _tv_denoise_1d(_condat_input, lam, 0, lo_bounds, hi_bounds)
    except Exception as solve_exc:
        logger.warning(
            "L1 solver: seg=%d frames=%d solver=%s EXCEPTION %s: %s — "
            "falling back to raw targets",
            seg_idx, n_frames, _solver_used,
            type(solve_exc).__name__, solve_exc,
        )
        solved = list(targets_dz)
    _solve_ms = (time.perf_counter() - _t0) * 1000

    # ── Post-solve smoothing: kill residual jitter ──
    solved = _post_solve_smooth(solved)

    logger.info(
        "L1 solver: shot=%d frames=%d mode=%s solver=%s solve_ms=%.1f",
        seg_idx, n_frames, "pending", _solver_used, _solve_ms,
    )

    # ── Debug telemetry dump ──
    if _DUMP_L1:
        _dump_solve_csv(job_id, seg_idx, times, targets, solved, lo_bounds, hi_bounds,
                        solver=_solver_used, weights=weights)

    # Post-solve verification: check that all hard features are in frame
    if hard_features:
        path_pairs = list(zip(times, solved))
        violations = assert_required_in_frame(
            path_pairs, hard_features, source_width, crop_aspect, source_height,
        )
        if violations:
            logger.warning(
                "L1 solver: %d frames have required features out of crop after solve "
                "(triggering PADDED fallback): %s",
                len(violations),
                [(f"t={t:.2f}", ids) for t, ids in violations[:5]],
            )
            return {
                "mode": "infeasible",
                "center": source_width / 2.0,
                "path": [],
                "slope": 0.0,
                "ease_in_ms": 0,
                "infeasible_frames": violations,
            }

    # Analyze the solved path
    path_min = min(solved)
    path_max = max(solved)
    path_range = path_max - path_min

    # Mode selection
    if path_range < STATIONARY_THRESHOLD * source_width:
        # Stationary: tiny movement, emit single static center
        center = sum(solved) / len(solved)
        return {
            "mode": "stationary",
            "center": center,
            "path": [],
            "slope": 0.0,
            "ease_in_ms": 0,
            "infeasible_frames": [],
        }

    # Post-solve panning check (fallback for cases where pre-solve fit was
    # close-but-not-quite and the solver smoothed it into something more linear)
    r2, slope_per_step = _linear_fit_r2(solved)
    if len(times) >= 3:
        dt = (times[-1] - times[0]) / max(len(times) - 1, 1)
        slope_px_per_sec = slope_per_step / max(dt, 0.001)
    else:
        slope_px_per_sec = 0.0

    if r2 > PANNING_R2_THRESHOLD and abs(slope_px_per_sec) > PANNING_MIN_SLOPE:
        return {
            "mode": "panning",
            "center": sum(solved) / len(solved),
            "path": list(zip(times, solved)),
            "slope": slope_px_per_sec,
            "ease_in_ms": 0,  # constant velocity, no ease
            "infeasible_frames": [],
        }

    # Tracking: non-trivial movement
    return {
        "mode": "tracking",
        "center": solved[0],
        "path": list(zip(times, solved)),
        "slope": 0.0,
        "ease_in_ms": 0,  # L1 solver says snap
        "infeasible_frames": [],
    }


def _classify_segment_mode(
    times: list[float],
    solved: list[float],
    source_width: int,
) -> dict:
    """Classify a segment slice as stationary/tracking/panning.

    Returns the same dict shape as solve_camera_path.
    """
    if not solved:
        return {
            "mode": "stationary",
            "center": source_width / 2.0,
            "path": [],
            "slope": 0.0,
            "ease_in_ms": 0,
            "infeasible_frames": [],
        }

    path_range = max(solved) - min(solved)

    if path_range < STATIONARY_THRESHOLD * source_width:
        center = sum(solved) / len(solved)
        return {
            "mode": "stationary",
            "center": center,
            "path": [],
            "slope": 0.0,
            "ease_in_ms": 0,
            "infeasible_frames": [],
        }

    r2, slope_per_step = _linear_fit_r2(solved)
    if len(times) >= 3:
        dt = (times[-1] - times[0]) / max(len(times) - 1, 1)
        slope_px_per_sec = slope_per_step / max(dt, 0.001)
    else:
        slope_px_per_sec = 0.0

    if r2 > PANNING_R2_THRESHOLD and abs(slope_px_per_sec) > PANNING_MIN_SLOPE:
        return {
            "mode": "panning",
            "center": sum(solved) / len(solved),
            "path": list(zip(times, solved)),
            "slope": slope_px_per_sec,
            "ease_in_ms": 0,
            "infeasible_frames": [],
        }

    return {
        "mode": "tracking",
        "center": solved[0],
        "path": list(zip(times, solved)),
        "slope": 0.0,
        "ease_in_ms": 0,
        "infeasible_frames": [],
    }


def build_gaming_pan_targets(
    *,
    action_xy_pct: tuple[float, float],
    events: list,
    start: float,
    end: float,
    target_fps: float = 30.0,
    pan_weight_ratio: float = GAMING_PAN_WEIGHT_RATIO,
    pan_hold_sec: float = GAMING_PAN_HOLD_SEC,
    pan_ramp_sec: float = GAMING_PAN_RAMP_SEC,
    action_hold_scale: float = GAMING_PAN_ACTION_HOLD_SCALE,
) -> tuple[list[tuple[float, float]], list[float]]:
    """Build a per-frame target signal + weights for a gaming shot.

    Phase 3 — for a gaming shot that has zero or more
    :class:`gaming_event_detector.GamingEvent` markers, this helper
    blends the action anchor (typically the crosshair x or the
    per-genre action center) with each event's ``target_region``
    using a triangular temporal weight:

      - ``[T - ramp, T]``      → action weight ramps DOWN to
                                  ``action_hold_scale``,
                                  pan weight ramps UP from 0 to
                                  ``pan_weight_ratio``.
      - ``[T, T + duration]``  → pan weight held at full strength,
                                  action weight at ``action_hold_scale``.
      - ``[T + d, T + d + r]`` → both ramp back to baseline.

    The L1 solver's existing total-variation smoothness term keeps
    the transition smooth — no hand-rolled easing math required.

    Args:
        action_xy_pct: ``(x, y)`` baseline anchor in % of source.
        events: List of GamingEvent (only those with
            ``target_region != None`` produce a pan; ``None``
            target regions are treated as zoom-outs and skipped
            here — Phase 4's layout chooser handles those).
        start, end: Shot time range (seconds).
        target_fps: Output sample rate.
        pan_weight_ratio: Peak weight on the target during the
            hold, as a multiple of the baseline action weight (1.0).
        pan_hold_sec / pan_ramp_sec: Window shape.
        action_hold_scale: How much we attenuate the action
            anchor while panning (0 = full kill, 1 = no scaling).

    Returns:
        ``(positions, weights)`` parallel lists at uniform target_fps.
        ``positions`` is ``[(timestamp, x_pct), ...]`` and
        ``weights`` is the data-fidelity weight per frame (1.0 =
        baseline, > 1.0 means the solver is pulled harder onto
        that point).
    """
    if end <= start:
        return [], []

    dt = 1.0 / target_fps
    n_frames = max(1, int((end - start) * target_fps))

    action_x = float(action_xy_pct[0])

    # Pre-build a list of (event_t_start, event_t_end, target_x, peak_weight)
    # for easy per-frame lookup.
    pan_windows: list[tuple[float, float, float, float, float]] = []
    for ev in events or []:
        target_region = getattr(ev, "target_region", None)
        if target_region is None:
            continue
        try:
            tx = float(target_region[0]) + float(target_region[2]) / 2.0
        except (TypeError, IndexError):
            continue
        ev_t = float(getattr(ev, "timestamp", 0.0))
        ev_d = float(getattr(ev, "duration", pan_hold_sec))
        win_start = ev_t - pan_ramp_sec
        win_end = ev_t + ev_d + pan_ramp_sec
        # Skip events fully outside the shot
        if win_end < start or win_start > end:
            continue
        pan_windows.append((win_start, ev_t, ev_t + ev_d, win_end, tx))

    positions: list[tuple[float, float]] = []
    weights: list[float] = []

    for i in range(n_frames):
        t = start + i * dt
        if t >= end:
            break

        # Find the active pan window (at most one — events should
        # have been merged upstream when they overlap).
        target_x = action_x
        target_w = 1.0
        for win_start, hold_start, hold_end, win_end, tx in pan_windows:
            if win_start <= t <= win_end:
                # Compute interpolation weight in [0, 1]
                if t < hold_start:
                    span = max(hold_start - win_start, 1e-6)
                    interp = (t - win_start) / span
                elif t <= hold_end:
                    interp = 1.0
                else:
                    span = max(win_end - hold_end, 1e-6)
                    interp = max(0.0, 1.0 - (t - hold_end) / span)

                # Blend action and target positions by interp.
                target_x = (1.0 - interp) * action_x + interp * tx
                # Weight scales between baseline (1.0) and
                # pan_weight_ratio at full hold; action gets
                # attenuated to action_hold_scale at full hold.
                pan_w = 1.0 + (pan_weight_ratio - 1.0) * interp
                action_attn = 1.0 - (1.0 - action_hold_scale) * interp
                # The blended weight is the convex combination of
                # the two weights — pan weight when target_x = tx,
                # attenuated action weight when target_x = action_x.
                target_w = (1.0 - interp) * action_attn + interp * pan_w
                break

        positions.append((round(t, 6), float(target_x)))
        weights.append(float(target_w))

    return positions, weights


def build_gaming_center_biased_targets(
    *,
    crosshair_path: Optional[list] = None,
    events: Optional[list] = None,
    start: float,
    end: float,
    target_fps: float = 30.0,
    conf_threshold: float = None,  # type: ignore[assignment]
    w_center_default: float = None,  # type: ignore[assignment]
    w_center_active: float = None,  # type: ignore[assignment]
    w_crosshair: float = None,  # type: ignore[assignment]
    w_pan: float = None,  # type: ignore[assignment]
    pan_hold_sec: float = None,  # type: ignore[assignment]
    pan_ramp_sec: float = None,  # type: ignore[assignment]
) -> tuple[list[tuple[float, float]], list[float]]:
    """Build the center-biased gaming target stream for the L1 solver.

    Phase 3a — this is the canonical target builder for shots whose
    ``gameplay_subtype`` lives in :data:`CENTER_BIAS_GENRES`. It
    combines three anchor behaviors into a single per-frame
    ``(target_x, weight)`` stream the existing
    :func:`solve_camera_path` / :func:`solve_camera_path_for_shot`
    pipeline already consumes:

    1. **Hard center anchor (default).** Every frame starts with a
       unary term pulling toward ``target_x = 50`` (% of frame
       width) at weight :data:`GAMING_CENTER_WEIGHT_DEFAULT`. This
       is the safety net that guarantees a rollback to the
       rock-solid "center-crop the action" default whenever every
       other signal fails.

    2. **Per-frame crosshair override.** When the crosshair
       tracker entry at ``t`` carries
       ``confidence >= conf_threshold`` (default 0.6), the unary
       target switches to the tracked crosshair x and the weight
       rises to :data:`GAMING_CROSSHAIR_WEIGHT`. The center anchor
       is implicitly attenuated to
       :data:`GAMING_CENTER_WEIGHT_ACTIVE` in this frame — since
       the L1 solver consumes a single ``(target, weight)`` pair
       per frame, the effective anchor is the crosshair position
       at the higher weight (the attenuated center term is folded
       into the smoothness prior). On low-confidence or missing
       frames the crosshair override does NOT fire; saliency /
       motion / face-cluster anchors are NEVER consulted here by
       design.

    3. **Time-windowed pan anchor (gaming events).** For each
       :class:`gaming_event_detector.GamingEvent` with a
       ``target_region`` the helper blends the current anchor
       (center or crosshair) toward the region's centroid via a
       triangular temporal weight — 0.3 s ease-in, ``pan_hold_sec``
       hold at :data:`GAMING_PAN_WEIGHT`, 0.3 s ease-out. After the
       pan window closes the stream returns to the center/crosshair
       anchor and the solver's TV smoothness prior naturally eases
       the crop back over ~0.3-0.5 s.

    Args:
        crosshair_path: List of :class:`CrosshairFrame`-shaped
            objects (or ``(t, x, y, conf)`` tuples). May be empty
            or ``None`` — every frame falls back to hard center.
        events: List of :class:`GamingEvent` objects. Events whose
            ``target_region`` is ``None`` (zoom-out signals) are
            skipped here; Phase 4's layout chooser handles those.
        start, end: Shot time range (seconds).
        target_fps: Output sample rate.
        conf_threshold: Crosshair confidence floor to accept the
            per-frame override. Default
            :data:`GAMING_CROSSHAIR_CONF_THRESHOLD`.
        w_center_default / w_center_active: Center-anchor weights.
            Defaults from the :data:`GAMING_CENTER_WEIGHT_*` env
            vars.
        w_crosshair: Crosshair override weight. Default
            :data:`GAMING_CROSSHAIR_WEIGHT`.
        w_pan: Pan target weight during the hold. Default
            :data:`GAMING_PAN_WEIGHT`.
        pan_hold_sec / pan_ramp_sec: Pan window shape. Defaults
            from :data:`GAMING_PAN_HOLD_SEC` /
            :data:`GAMING_PAN_RAMP_SEC`.

    Returns:
        ``(positions, weights)`` parallel lists at uniform
        ``target_fps``. Positions are ``[(timestamp, x_pct), ...]``
        in percent of source width; weights are the per-frame
        data-fidelity weights for the L1 unary term. Callers that
        want pixel coordinates should multiply by
        ``source_width / 100`` before feeding the solver.
    """
    if end <= start:
        return [], []

    if conf_threshold is None:
        conf_threshold = GAMING_CROSSHAIR_CONF_THRESHOLD
    if w_center_default is None:
        w_center_default = GAMING_CENTER_WEIGHT_DEFAULT
    if w_center_active is None:
        w_center_active = GAMING_CENTER_WEIGHT_ACTIVE
    if w_crosshair is None:
        w_crosshair = GAMING_CROSSHAIR_WEIGHT
    if w_pan is None:
        w_pan = GAMING_PAN_WEIGHT
    if pan_hold_sec is None:
        pan_hold_sec = GAMING_PAN_HOLD_SEC
    if pan_ramp_sec is None:
        pan_ramp_sec = GAMING_PAN_RAMP_SEC

    dt = 1.0 / target_fps
    n_frames = max(1, int((end - start) * target_fps))

    # ── Normalize the crosshair path to (t, x, y, conf) tuples ──
    ch_entries: list[tuple[float, float, float, float]] = []
    for entry in crosshair_path or []:
        if hasattr(entry, "timestamp"):
            ch_entries.append((
                float(entry.timestamp),
                float(entry.x_pct),
                float(getattr(entry, "y_pct", 50.0)),
                float(getattr(entry, "confidence", 0.0)),
            ))
        else:
            # (t, x, y, conf) tuple — pad with defaults if shorter.
            t = float(entry[0])
            x = float(entry[1]) if len(entry) > 1 else 50.0
            y = float(entry[2]) if len(entry) > 2 else 50.0
            c = float(entry[3]) if len(entry) > 3 else 0.0
            ch_entries.append((t, x, y, c))
    ch_entries.sort(key=lambda p: p[0])
    ch_times = [e[0] for e in ch_entries]

    def _lookup_crosshair(t: float) -> tuple[float, float] | None:
        """Return (x_pct, conf) for the crosshair entry closest to
        ``t`` within 0.1 s. None if no entry is close enough."""
        if not ch_entries:
            return None
        import bisect
        idx = bisect.bisect_left(ch_times, t)
        candidates = []
        if idx < len(ch_entries):
            candidates.append(ch_entries[idx])
        if idx > 0:
            candidates.append(ch_entries[idx - 1])
        if not candidates:
            return None
        best = min(candidates, key=lambda e: abs(e[0] - t))
        if abs(best[0] - t) > 0.1:
            return None
        return (best[1], best[3])

    # ── Pre-compute pan windows ──
    pan_windows: list[tuple[float, float, float, float, float]] = []
    for ev in events or []:
        target_region = getattr(ev, "target_region", None)
        if target_region is None:
            continue
        try:
            tx = float(target_region[0]) + float(target_region[2]) / 2.0
        except (TypeError, IndexError):
            continue
        ev_t = float(getattr(ev, "timestamp", 0.0))
        ev_d = float(getattr(ev, "duration", pan_hold_sec))
        win_start = ev_t - pan_ramp_sec
        win_end = ev_t + ev_d + pan_ramp_sec
        if win_end < start or win_start > end:
            continue
        pan_windows.append((win_start, ev_t, ev_t + ev_d, win_end, tx))

    positions: list[tuple[float, float]] = []
    weights: list[float] = []

    for i in range(n_frames):
        t = start + i * dt
        if t >= end:
            break

        # Step 1: baseline center-or-crosshair anchor.
        ch = _lookup_crosshair(t)
        if ch is not None and ch[1] >= conf_threshold:
            base_x = ch[0]
            base_w = float(w_crosshair)
            anchor_is_crosshair = True
        else:
            base_x = 50.0
            base_w = float(w_center_default)
            anchor_is_crosshair = False

        # Step 2: pan window override.
        target_x = base_x
        target_w = base_w
        for win_start, hold_start, hold_end, win_end, pan_x in pan_windows:
            if win_start <= t <= win_end:
                # Triangular weight in [0, 1]
                if t < hold_start:
                    span = max(hold_start - win_start, 1e-6)
                    interp = (t - win_start) / span
                elif t <= hold_end:
                    interp = 1.0
                else:
                    span = max(win_end - hold_end, 1e-6)
                    interp = max(0.0, 1.0 - (t - hold_end) / span)

                # Blend the pan target into the base anchor.
                target_x = (1.0 - interp) * base_x + interp * pan_x
                # Weight scales between the base weight and the
                # full pan weight. When the anchor is a confident
                # crosshair the base weight is already higher than
                # the center default; preserve that and ramp up
                # to the pan weight only if it's strictly larger.
                peak_w = max(float(w_pan), base_w)
                target_w = (1.0 - interp) * base_w + interp * peak_w
                break

        positions.append((round(t, 6), float(target_x)))
        weights.append(float(target_w))

        # anchor_is_crosshair is unused in the current single-target
        # form but retained for future callers that may want to
        # plumb a second unary term per frame.
        _ = anchor_is_crosshair

    return positions, weights


def solve_camera_path_for_shot(
    shot_start: float,
    shot_end: float,
    segments_in_shot: list,
    propagated_path,
    source_width: int = 1920,
    source_height: int = 1080,
    crop_aspect: float = 9 / 16,
    target_fps: float = 30.0,
    lam: float = TV_LAMBDA,
    hard_features: Optional[list] = None,
    job_id: str = "",
    *,
    is_animated: bool = False,
    dense_faces_for_anchors: Optional[list] = None,
) -> list[dict]:
    """Solve L1-optimal camera path for an entire shot, then slice per segment.

    Runs the TV solver once over the full shot (not per-segment), which:
    - Eliminates pops at segment boundaries (they're inside one solve)
    - Provides lookahead: the solver can begin easing BEFORE a target moves

    Mirror-reflects the first/last 30 frames at shot boundaries for
    symmetric lookahead at the edges.

    Args:
        shot_start: Shot start time (seconds).
        shot_end: Shot end time (seconds).
        segments_in_shot: list of segment objects with .start, .end, .active_slot.
        propagated_path: InterpolatedFaceTimeline or dense_faces list.
        source_width, source_height, crop_aspect: Video dimensions.
        target_fps: Uniform sample rate for the solver.
        lam: TV regularization weight.
        hard_features: Required features for hard constraint enforcement.
        job_id: For telemetry.

    Returns:
        list[dict] — one result dict per segment (same shape as solve_camera_path).
    """
    if not segments_in_shot:
        return []

    # ── Resolution-independent lambda ──
    if lam == TV_LAMBDA:
        lam = TV_LAMBDA_FRAC * source_width

    # ── Build full-shot target at uniform fps ──
    dt = 1.0 / target_fps
    pad_frames = 45  # ~1.5s lookahead padding — longer than the original 1s
    # to give the solver more context for hold decisions and prevent early snap-off

    # Collect per-segment targets, concatenated
    all_times = []
    all_targets = []
    segment_boundaries = []  # [(start_idx, end_idx)] into the arrays

    all_weights: list[float] = []
    _used_face_anchors_any = False

    for seg in segments_in_shot:
        seg_start_idx = len(all_times)
        seg_positions: list[tuple[float, float]] = []
        seg_weights: list[float] = []

        # Phase 4: animated content with a dense face track switches the
        # unary to per-frame nose_x with a higher data-fidelity weight.
        # Falls through to the legacy propagated-positions extraction
        # when the gate is closed (live action, sparse face data).
        if is_animated and dense_faces_for_anchors:
            slot_center_pct = 50.0
            try:
                if seg.active_slot is not None:
                    # Look up the slot's nose_x from any frame that has it
                    for df in dense_faces_for_anchors:
                        for face in getattr(df, "faces", []) or []:
                            if getattr(face, "identity_id", -1) == seg.active_slot:
                                nx = getattr(face, "nose_x", None)
                                if nx is not None:
                                    slot_center_pct = float(nx)
                                    break
                        else:
                            continue
                        break
            except Exception:
                pass

            anchor_pos, anchor_w, used = build_face_anchor_targets(
                dense_faces_for_anchors,
                slot_id=seg.active_slot,
                slot_center_pct=slot_center_pct,
                start=seg.start,
                end=seg.end,
                target_fps=target_fps,
                is_animated=True,
            )
            if used and anchor_pos:
                # Convert percent → pixel for the solver
                seg_positions = [
                    (t, x / 100.0 * source_width)
                    for t, x in anchor_pos
                ]
                seg_weights = list(anchor_w)
                _used_face_anchors_any = True

        if not seg_positions:
            positions = get_propagated_positions_for_segment(
                propagated_path, seg.active_slot, seg.start, seg.end,
                source_width=source_width, target_fps=target_fps,
            )
            seg_positions = list(positions)
            seg_weights = [1.0] * len(seg_positions)

        for (t, x), w in zip(seg_positions, seg_weights):
            all_times.append(t)
            all_targets.append(x)
            all_weights.append(w)
        seg_end_idx = len(all_times)
        segment_boundaries.append((seg_start_idx, seg_end_idx))

    if _used_face_anchors_any:
        logger.info(
            "[%s] L1 shot solver: face-anchor unary in use (anime gate open)",
            job_id,
        )

    if len(all_targets) < 2:
        # Not enough data — fall back to per-segment solving
        results = []
        for seg in segments_in_shot:
            results.append(solve_camera_path(
                [], source_width, lam, hard_features,
                source_height, crop_aspect, job_id,
            ))
        return results

    # ── Pre-solve dead-zone ──
    deadzone_px = DEADZONE_FRAC * source_width
    all_targets = _apply_deadzone(all_targets, deadzone_px)

    # ── Mirror-reflection padding at shot boundaries ──
    n_orig = len(all_targets)
    pad_left = min(pad_frames, n_orig)
    pad_right = min(pad_frames, n_orig)

    # Pad left: mirror the first pad_left samples
    left_pad = list(reversed(all_targets[:pad_left]))
    # Pad right: mirror the last pad_right samples
    right_pad = list(reversed(all_targets[n_orig - pad_right:]))

    padded_targets = left_pad + all_targets + right_pad
    padded_times = []
    for i in range(len(padded_targets)):
        padded_times.append(shot_start + (i - pad_left) * dt)

    # ── Compute hard bounds for padded signal (if any) ──
    lo_bounds = None
    hi_bounds = None
    if hard_features:
        lo_bounds, hi_bounds, infeasible_flags = compute_hard_bounds(
            hard_features, padded_times, source_width, crop_aspect, source_height,
        )
        infeasible_times = [t for t, inf in zip(padded_times, infeasible_flags) if inf]
        if infeasible_times:
            logger.warning(
                "L1 shot solver: %d/%d frames infeasible",
                len(infeasible_times), len(padded_times),
            )
            # Return infeasible for all segments
            return [{
                "mode": "infeasible",
                "center": source_width / 2.0,
                "path": [],
                "slope": 0.0,
                "ease_in_ms": 0,
                "infeasible_frames": [(t, []) for t in infeasible_times],
            } for _ in segments_in_shot]

    # ── Solve: LP (with accel+jerk) or Condat TV ──
    n_padded = len(padded_targets)
    use_lp = (
        _SOLVER_MODE == "lp"
        or (_SOLVER_MODE == "auto" and n_padded <= LP_MAX_FRAMES)
    )
    if _SOLVER_MODE == "lp" and n_padded > LP_MAX_FRAMES * 2:
        raise ValueError(
            f"LP solver forced but shot has {n_padded} frames "
            f"(> {LP_MAX_FRAMES * 2} max). Use CLIPAI_L1_SOLVER=auto."
        )
    if use_lp and n_padded > LP_MAX_FRAMES and _SOLVER_MODE == "auto":
        use_lp = False

    _solver_used = "lp" if use_lp else "condat"
    # Pre-solve log so a hang is attributable to a specific shot. Without
    # this, previously the "L1 solver: ..." line only fired AFTER the
    # solve returned, which made it impossible to tell from logs which
    # shot was the one that froze the container.
    logger.info(
        "L1 solver: shot=%d frames=%d n_segments=%d solver=%s span=%.2f-%.2fs — starting",
        0, n_padded, len(segments_in_shot), _solver_used,
        float(shot_start), float(shot_end),
    )
    _t0 = time.perf_counter()

    try:
        if use_lp:
            from backend.services._autoflip_lp import solve_autoflip_lp
            # Build bounds for LP solver (None → unconstrained as source_width bounds)
            lp_lo = []
            lp_hi = []
            for i in range(n_padded):
                lo_val = lo_bounds[i] if lo_bounds and lo_bounds[i] is not None else 0.0
                hi_val = hi_bounds[i] if hi_bounds and hi_bounds[i] is not None else float(source_width)
                lp_lo.append(lo_val)
                lp_hi.append(hi_val)
            padded_solved = solve_autoflip_lp(
                padded_targets, lp_lo, lp_hi,
                lam1=1.0, lam2=LP_LAMBDA_V, lam3=LP_LAMBDA_A, lam4=LP_LAMBDA_J,
            )
        else:
            padded_solved = _tv_denoise_1d(padded_targets, lam, 0, lo_bounds, hi_bounds)
    except Exception as solve_exc:
        # Any solver explosion (scipy time-limit, infeasible LP, numpy
        # error, memory pressure) must NOT take the worker down — fall
        # back to the raw dead-zoned target signal. The downstream mode
        # classifier still gets a coherent path, and the surrounding
        # pipeline continues.
        logger.warning(
            "L1 solver: shot=0 frames=%d solver=%s EXCEPTION %s: %s — "
            "falling back to raw targets",
            n_padded, _solver_used, type(solve_exc).__name__, solve_exc,
        )
        padded_solved = list(padded_targets)

    _solve_ms = (time.perf_counter() - _t0) * 1000

    logger.info(
        "L1 solver: shot=%d frames=%d mode=%s solver=%s solve_ms=%.1f",
        0, n_padded, "shot", _solver_used, _solve_ms,
    )

    # ── Discard padding ──
    solved = padded_solved[pad_left:pad_left + n_orig]
    times = all_times

    # ── Post-solve smoothing: kill residual jitter ──
    solved = _post_solve_smooth(solved)

    # ── Debug telemetry dump ──
    if _DUMP_L1:
        _dump_solve_csv(
            job_id, 9000, times, all_targets, solved,
            lo_bounds[pad_left:pad_left + n_orig] if lo_bounds else None,
            hi_bounds[pad_left:pad_left + n_orig] if hi_bounds else None,
            solver=_solver_used,
        )

    # ── Post-solve verification ──
    if hard_features:
        path_pairs = list(zip(times, solved))
        violations = assert_required_in_frame(
            path_pairs, hard_features, source_width, crop_aspect, source_height,
        )
        if violations:
            logger.warning(
                "L1 shot solver: %d frames have required features out of crop",
                len(violations),
            )
            return [{
                "mode": "infeasible",
                "center": source_width / 2.0,
                "path": [],
                "slope": 0.0,
                "ease_in_ms": 0,
                "infeasible_frames": violations,
            } for _ in segments_in_shot]

    # ── Slice solved path back into per-segment results ──
    results = []
    for (seg_start_idx, seg_end_idx) in segment_boundaries:
        seg_times = times[seg_start_idx:seg_end_idx]
        seg_solved = solved[seg_start_idx:seg_end_idx]
        result = _classify_segment_mode(seg_times, seg_solved, source_width)
        results.append(result)

    return results


def get_dense_face_positions_for_segment(
    dense_faces: list,
    active_slot: Optional[int],
    start: float,
    end: float,
) -> list[tuple[float, float]]:
    """Extract (timestamp, x_pixel) pairs for a face slot in a time range.

    Uses nose_x from dense face data. Falls back to any face if slot is None.

    DEPRECATED: Prefer get_propagated_positions_for_segment which produces
    uniform-fps output suitable for the TV solver.
    """
    positions = []
    for df in dense_faces:
        if df.timestamp < start or df.timestamp >= end:
            continue
        for f in df.faces:
            sid = getattr(f, 'identity_id', -1)
            if active_slot is not None and sid != active_slot:
                continue
            x = getattr(f, 'nose_x', None)
            if x is not None:
                positions.append((df.timestamp, x))
                break  # one face per frame
    return positions


def get_propagated_positions_for_segment(
    propagated_path,
    active_slot: Optional[int],
    start: float,
    end: float,
    source_width: int = 1920,
    target_fps: float = 30.0,
) -> list[tuple[float, float]]:
    """Extract uniform-fps (timestamp, x_pixel) pairs from a propagated timeline.

    Produces a uniformly-spaced trajectory at target_fps by resampling
    the propagated path via linear interpolation. Gaps (occlusion, identity
    drop) are filled by holding the last known x-position.

    Works with either an InterpolatedFaceTimeline (from dense_propagator)
    or a raw list of dense_faces (sparse detections). When given sparse
    input, resamples onto a uniform grid so the TV solver's finite-difference
    operator sees uniform dt.

    Args:
        propagated_path: InterpolatedFaceTimeline or list of FrameFaces/dense_faces.
        active_slot: Face registry slot ID to track (None = any face).
        start: Segment start time (seconds).
        end: Segment end time (seconds, exclusive).
        source_width: Source video width (for percentage-to-pixel conversion).
        target_fps: Output sample rate (default 30.0).

    Returns:
        list[(timestamp, x_pixel)] at uniform target_fps spacing.
        Returns the same signature as get_dense_face_positions_for_segment.
    """
    # ── Step 1: Extract raw (t, x_pct) pairs from the source ──
    raw_samples = []  # [(t, x_pct)]

    if hasattr(propagated_path, 'slot_positions_in_range'):
        # InterpolatedFaceTimeline path — per-source-frame data
        if active_slot is not None:
            positions = propagated_path.slot_positions_in_range(active_slot, start, end)
            for t, cx, _cy, _w, _h, _conf in positions:
                if start <= t < end:
                    raw_samples.append((t, cx))
        else:
            # No specific slot — use first available slot per frame
            for s in propagated_path.samples:
                if s.timestamp < start or s.timestamp >= end:
                    continue
                if s.bboxes:
                    first_slot = next(iter(s.bboxes))
                    cx, _cy, _w, _h = s.bboxes[first_slot]
                    raw_samples.append((s.timestamp, cx))
    else:
        # Sparse dense_faces list — extract like get_dense_face_positions_for_segment
        for df in propagated_path:
            if df.timestamp < start or df.timestamp >= end:
                continue
            for f in df.faces:
                sid = getattr(f, 'identity_id', -1)
                if active_slot is not None and sid != active_slot:
                    continue
                x = getattr(f, 'nose_x', None)
                if x is not None:
                    raw_samples.append((df.timestamp, x))
                    break

    if not raw_samples:
        return []

    raw_samples.sort(key=lambda p: p[0])

    # ── Step 2: Resample onto uniform grid at target_fps ──
    dt = 1.0 / target_fps
    n_frames = max(1, int((end - start) * target_fps))
    uniform = []
    last_x = raw_samples[0][1]  # hold for gaps
    raw_idx = 0

    for i in range(n_frames):
        t = start + i * dt
        if t >= end:
            break

        # Find the two raw samples bracketing t for linear interpolation
        while raw_idx < len(raw_samples) - 1 and raw_samples[raw_idx + 1][0] <= t:
            raw_idx += 1

        if raw_idx >= len(raw_samples) - 1:
            # Past the last raw sample — hold last known
            x_pct = raw_samples[-1][1]
        elif raw_samples[raw_idx][0] >= t:
            # Before or at the first relevant sample
            x_pct = raw_samples[raw_idx][1]
        else:
            # Linear interpolation between raw_samples[raw_idx] and [raw_idx + 1]
            t0, x0 = raw_samples[raw_idx]
            t1, x1 = raw_samples[raw_idx + 1]
            span = t1 - t0
            if span > 0:
                alpha = (t - t0) / span
                x_pct = x0 + alpha * (x1 - x0)
            else:
                x_pct = x0

        last_x = x_pct
        # Convert from percentage to pixel space
        x_px = x_pct / 100.0 * source_width
        uniform.append((round(t, 6), x_px))

    return uniform
