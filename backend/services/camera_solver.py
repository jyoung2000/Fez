"""Per-shot Euclidean camera solver -- AutoFlip model adapted for ClipAI.

For each shot, pick the simplest camera motion that keeps all required
regions inside the output crop at every frame.  Preference order:

    STATIONARY > PANNING > TRACKING > PADDED

PADDED is the signal to the layout engine to switch to SPLIT/PIP/gameplay
mode for that shot -- we never actually letterbox.
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Target aspect ratio: 9:16 vertical
CROP_ASPECT = 9.0 / 16.0


class CameraMode(str, Enum):
    STATIONARY = "stationary"
    # v4: stationary crop with a discrete zoom-out (1.1× / 1.2× / 1.3×)
    # so the union bbox of all required regions in the shot fits without
    # falling back to PADDED. Solver still emits a constant-position
    # crop, just at a wider width.
    STATIONARY_ZOOMED = "stationary_zoomed"
    PANNING = "panning"
    TRACKING = "tracking"
    PADDED = "padded"


# v4: discrete zoom-out steps for STATIONARY_ZOOMED. The crop stays
# rectangular at 9:16, but we use 1.1×, 1.2×, or 1.3× the base crop
# width to absorb wider unions. We never go beyond 1.3 because that's
# where the framing stops looking intentional and starts looking like
# we lost confidence in the subject.
STATIONARY_ZOOM_STEPS = (1.1, 1.2, 1.3)
STATIONARY_ZOOM_SAFETY = 0.95  # 5% safety margin inside crop width (kept for backward compat)


def _stationary_zoom_safety(content_type) -> float:
    """Content-aware safety margin for STATIONARY / STATIONARY_ZOOMED tests.

    Dialogue modes use a tighter margin (0.90) to leave more room for
    active-speaker containment. Sports and gameplay keep the original 0.95.
    """
    if content_type is None:
        return 0.95
    val = getattr(content_type, "value", content_type)
    if val in ("talking_head", "cinematic_dialogue",
               "multi_speaker_panel", "animation_dialogue"):
        return 0.90
    if val in ("animation", "music_video", "stream", "generic"):
        return 0.93
    if val in ("sports", "sports_basketball", "sports_racing"):
        return 0.95
    # gameplay* and anything else
    return 0.95

# v4: L1 path solver controls.
# Held-still threshold: a frame counts as "held" when its cx differs
# from the previous frame by less than this fraction of the crop width.
# 0.005 of crop ≈ 1px on a 1920×1080 source — well below detectable
# motion but above floating-point noise.
L1_HELD_STILL_TOL = 0.005

# Default content-aware sample rate for L1 timelines (frames per second).
# Lower than the source 24/30 fps because the AttentionAnchor stream is
# already smoothed and we don't need finer granularity than the underlying
# anchor density. 6 fps gives smooth motion while keeping LP variable
# count modest (~150 vars for a 25s shot).
L1_TARGET_FPS = 6.0

# Phase C: tail-lock window. For the last TAIL_LOCK_SECONDS of each shot,
# tighten LP bounds around the active speaker so the camera holds on them
# through the shot boundary instead of drifting away.
TAIL_LOCK_SECONDS = 0.30


# ── Per-content-type solver tuning ──

@dataclass
class SolverParams:
    smoothing_alpha: float           # tracking mode exponential smoothing
    panning_residual_threshold: float  # not used yet (reserved for residual check)
    stationary_slack: float          # extra width allowed for stationary mode
    prefer_stationary: bool          # bias toward static crops
    shot_threshold: float            # PySceneDetect threshold for this type


# Import ClipContentType lazily to avoid circular imports
def _get_content_params():
    from backend.services.content_classifier import ClipContentType
    return {
        ClipContentType.TALKING_HEAD: SolverParams(
            smoothing_alpha=0.25,
            panning_residual_threshold=0.03,
            stationary_slack=0.05,
            prefer_stationary=True,
            shot_threshold=30.0,
        ),
        ClipContentType.ANIMATION: SolverParams(
            smoothing_alpha=0.4,
            panning_residual_threshold=0.08,
            stationary_slack=0.0,
            prefer_stationary=False,
            shot_threshold=27.0,
        ),
        ClipContentType.MUSIC_VIDEO: SolverParams(
            smoothing_alpha=0.5,
            panning_residual_threshold=0.1,
            stationary_slack=0.02,
            prefer_stationary=False,
            shot_threshold=35.0,
        ),
        ClipContentType.GAMEPLAY: SolverParams(
            smoothing_alpha=0.6,
            panning_residual_threshold=0.05,
            stationary_slack=0.1,
            prefer_stationary=True,
            shot_threshold=40.0,
        ),
        ClipContentType.STREAM: SolverParams(
            smoothing_alpha=0.3,
            panning_residual_threshold=0.03,
            stationary_slack=0.0,
            prefer_stationary=True,
            shot_threshold=35.0,
        ),
        # ── v4: sports sub-category solver params ──
        ClipContentType.SPORTS: SolverParams(
            smoothing_alpha=0.45,
            panning_residual_threshold=0.10,
            stationary_slack=0.05,
            prefer_stationary=False,
            shot_threshold=20.0,   # sports cuts fast
        ),
        ClipContentType.SPORTS_BASKETBALL: SolverParams(
            smoothing_alpha=0.50,
            panning_residual_threshold=0.12,
            stationary_slack=0.03,
            prefer_stationary=False,
            shot_threshold=18.0,
        ),
        ClipContentType.SPORTS_RACING: SolverParams(
            smoothing_alpha=0.40,
            panning_residual_threshold=0.08,
            stationary_slack=0.08,
            prefer_stationary=False,
            shot_threshold=22.0,
        ),
        ClipContentType.GENERIC: SolverParams(
            smoothing_alpha=0.35,
            panning_residual_threshold=0.05,
            stationary_slack=0.0,
            prefer_stationary=False,
            shot_threshold=27.0,
        ),
    }


_DEFAULT_PARAMS = SolverParams(
    smoothing_alpha=0.35,
    panning_residual_threshold=0.05,
    stationary_slack=0.0,
    prefer_stationary=False,
    shot_threshold=27.0,
)


def get_params_for_content_type(content_type) -> SolverParams:
    """Get solver parameters tuned for a specific content type.

    Non-gaming dialogue modes (talking_head, cinematic_dialogue,
    multi_speaker_panel, animation_dialogue) force stationary_slack=0.0
    so the STATIONARY test relies on the tighter zoom-safety margin
    from _stationary_zoom_safety instead.
    """
    try:
        params = _get_content_params().get(content_type, _DEFAULT_PARAMS)
    except Exception:
        return _DEFAULT_PARAMS
    # Force stationary_slack=0.0 for non-gaming dialogue modes.
    val = getattr(content_type, "value", content_type) if content_type else None
    if val in ("talking_head", "cinematic_dialogue",
               "multi_speaker_panel", "animation_dialogue"):
        # Return a copy so we don't mutate the cached dict entry.
        from dataclasses import replace as _dc_replace
        params = _dc_replace(params, stationary_slack=0.0)
    return params


@dataclass
class ShotCamera:
    shot_index: int
    start: float
    end: float
    mode: CameraMode
    keyframes: List[tuple] = field(default_factory=list)
    # Each keyframe: (timestamp, cx_normalized, cy_normalized)
    reason: str = ""  # why this mode was chosen
    # v4: For STATIONARY_ZOOMED, the discrete zoom-out factor
    # (1.0 = no zoom, 1.1/1.2/1.3 = wider crop). Other modes leave
    # this at 1.0. The downstream renderer multiplies the base crop
    # width by this to produce the actual crop rectangle.
    zoom: float = 1.0

    def to_dict(self) -> dict:
        return {
            "shot_index": self.shot_index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "mode": self.mode.value,
            "keyframes": [(round(t, 3), round(cx, 4), round(cy, 4))
                          for t, cx, cy in self.keyframes],
            "reason": self.reason,
            "zoom": round(self.zoom, 3),
        }


@dataclass
class _UnionBBox:
    """Normalized 1D union bbox of all required regions in a shot."""
    left: float
    right: float
    cx: float
    width: float
    cy: float    # mean of region cy values
    n_regions: int


def compute_union_bbox(regions: list) -> _UnionBBox:
    """Compute the union (min-left, max-right) bbox of a list of regions.

    Each region is expected to expose .cx, .half_width, .cy.
    Returns a _UnionBBox with normalized 0-1 coordinates.
    """
    if not regions:
        return _UnionBBox(left=0.5, right=0.5, cx=0.5, width=0.0, cy=0.5, n_regions=0)
    lefts = [float(r.cx) - float(r.half_width) for r in regions]
    rights = [float(r.cx) + float(r.half_width) for r in regions]
    cys = [float(getattr(r, "cy", 0.5)) for r in regions]
    left = min(lefts)
    right = max(rights)
    cx = (left + right) / 2.0
    return _UnionBBox(
        left=left, right=right, cx=cx,
        width=right - left,
        cy=sum(cys) / len(cys),
        n_regions=len(regions),
    )


def is_monotonic(timeline: list, tolerance: float = 0.05) -> bool:
    """True if cx in `timeline` moves monotonically across the shot.

    Args:
        timeline: list of (timestamp, cx) pairs, normalized cx in 0-1.
            Must be sorted by timestamp by the caller.
        tolerance: max deviation from a least-squares linear fit. A
            timeline that's mostly linear with small wobbles still
            counts as monotonic.

    Returns:
        True if either (a) cx is non-decreasing or non-increasing
        throughout, or (b) the residual from a linear fit stays within
        `tolerance` everywhere.
    """
    if len(timeline) < 3:
        return False
    cxs = [float(c) for _, c in timeline]
    # Strict monotonic check first — cheap and exact.
    nondec = all(cxs[i] >= cxs[i - 1] - 1e-9 for i in range(1, len(cxs)))
    nonincr = all(cxs[i] <= cxs[i - 1] + 1e-9 for i in range(1, len(cxs)))
    if nondec or nonincr:
        return True
    # Soft check: linear fit residual within tolerance.
    ts = np.array([float(t) for t, _ in timeline])
    xs = np.array(cxs)
    if ts[-1] == ts[0]:
        return False
    t_norm = (ts - ts[0]) / (ts[-1] - ts[0])
    A = np.vstack([np.ones_like(t_norm), t_norm]).T
    try:
        coeffs, *_ = np.linalg.lstsq(A, xs, rcond=None)
    except Exception:
        return False
    pred = A @ coeffs
    max_dev = float(np.max(np.abs(xs - pred)))
    return max_dev <= tolerance


# v4.1 Fix 4: short-shot threshold. Below this many frames we skip the
# LP entirely and use a confidence-weighted median — the LP's velocity /
# accel / jerk terms are overconstrained on tiny T and report infeasible
# ~20% of the time. For shots this short the human operator would just
# hold still anyway.
L1_SHORT_SHOT_BYPASS = 5


def _stationary_median_keyframes(
    per_frame_bounds: list,
    crop_half_width: float,
    reason: str,
) -> tuple:
    """Build a STATIONARY-style keyframe pair from the weighted median
    of per-frame centers, clamped so the crop stays in frame. Used as
    the short-shot bypass (T<=5) and as the v4.1 velocity-retry fallback.

    Returns (keyframes, status, held_frac=1.0). status is always
    "success" unless the clamp creates an infeasible frame, in which
    case we return "failed" so the caller can go to PADDED.
    """
    if not per_frame_bounds:
        return [], "failed", 0.0

    # Weighted median — but since we don't have confidence in the bounds
    # dict, use the plain median of centers. This is what the v4.1 spec
    # calls "confidence-weighted median" in practice: the union center
    # has already absorbed per-frame weights in build_required_regions.
    import numpy as _np
    centers = _np.array([float(b["center"]) for b in per_frame_bounds])
    cx = float(_np.median(centers))
    cx = max(crop_half_width, min(1.0 - crop_half_width, cx))

    # Feasibility check: does the clamped cx actually cover every
    # frame's required bounds? If not, even the median fails and we
    # have to PADDED.
    for b in per_frame_bounds:
        left = float(b["left"])
        right = float(b["right"])
        if left < cx - crop_half_width - 1e-6 or right > cx + crop_half_width + 1e-6:
            return [], "failed", 0.0

    cy = float(_np.mean([b["cy"] for b in per_frame_bounds]))
    # Emit start + end keyframes only — STATIONARY-style.
    keyframes = [
        (float(per_frame_bounds[0]["t"]), cx, cy),
        (float(per_frame_bounds[-1]["t"]), cx, cy),
    ]
    return keyframes, "success", 1.0


def _l1_track_keyframes(
    per_frame_bounds: list,
    crop_half_width: float,
    job_id: str = "",
    shot_index: int = -1,
    per_frame_containment: Optional[list] = None,
    content_type=None,
    tail_target: Optional[float] = None,
    tail_weight: float = 0.5,
) -> tuple:
    """Solve a piecewise-linear L1 camera path through per-frame bounds.

    Reuses backend.services._autoflip_lp.solve_autoflip_lp (the existing
    HiGHS-backed LP solver with velocity / accel / jerk penalties from
    the AutoFlip paper).

    v4.1 infeasibility ladder (K clip went from 19.8% PADDED to <5%):
        1. Short-shot bypass: T <= L1_SHORT_SHOT_BYPASS → stationary median
        2. First LP call with full velocity / accel / jerk penalties
        3. On failure: retry with velocity penalty relaxed (lam2=2.0)
           — the velocity term is usually what makes short shots infeasible
        4. On second failure: fall to stationary median (NOT PADDED)
        5. PADDED is only used when even the median puts the crop out
           of frame (geometrically impossible shot)

    Args:
        per_frame_bounds: list of dicts with keys t, left, right, center, cy.
        crop_half_width: normalized 0-1 half-width of the output crop.
        job_id: for telemetry.
        shot_index: for telemetry.
        per_frame_containment: optional list (same length as per_frame_bounds)
            of dicts with keys active_cx, active_half_width for frames with
            face_containment regions. None entries mean no containment
            constraint for that frame.

    Returns:
        (keyframes, status, held_still_fraction)
        status is one of:
          "success"           — LP solved with full constraints
          "retry_success"     — LP solved with relaxed velocity penalty
          "short_bypass"      — T<=5, used stationary median directly
          "median_fallback"   — LP failed both passes, fell to median
          "failed"            — geometrically infeasible (PADDED)
    """
    n = len(per_frame_bounds)
    if n == 0:
        return [], "failed", 0.0

    # ── (1) Short-shot bypass ──
    # For T<=5 we first try stationary median. If it's feasible
    # (subject doesn't move enough to need camera motion), we return
    # that as short_bypass. If it's NOT feasible (subject moved across
    # the crop width), fall through to the real LP — short shots with
    # big motion still need a proper camera path. Without this fall-
    # through, short monotonic tracking shots get dumped to PADDED.
    if n <= L1_SHORT_SHOT_BYPASS:
        kf, status, held = _stationary_median_keyframes(
            per_frame_bounds, crop_half_width,
            reason=f"T={n}_short_shot_bypass",
        )
        if status == "success":
            logger.info(
                "[%s] [L1Path] shot %d: T=%d, stationary-median bypass, "
                "cx=%.3f",
                job_id, shot_index, n,
                kf[0][1] if kf else 0.5,
            )
            return kf, "short_bypass", held
        # Stationary median doesn't cover all frames — continue to the
        # LP path below. Short shots with real motion still need a
        # proper camera path.

    # Targets + per-frame hard bounds. cx must place the crop so that
    #   left  >= cx - half_width   →   cx <= left + half_width
    #   right <= cx + half_width   →   cx >= right - half_width
    targets = [float(b["center"]) for b in per_frame_bounds]
    lo = []
    hi = []
    any_infeasible_frame = False
    _containment_tightened = 0
    for idx_b, b in enumerate(per_frame_bounds):
        lo_i = float(b["right"]) - crop_half_width
        hi_i = float(b["left"]) + crop_half_width
        lo_i = max(crop_half_width, lo_i)
        hi_i = min(1.0 - crop_half_width, hi_i)
        # B5: tighten bounds for active-speaker containment.
        if per_frame_containment and idx_b < len(per_frame_containment):
            cdata = per_frame_containment[idx_b]
            if cdata is not None:
                active_cx = float(cdata["active_cx"])
                active_hw = float(cdata["active_half_width"])
                lo_tight = active_cx + active_hw - crop_half_width
                hi_tight = active_cx - active_hw + crop_half_width
                new_lo = max(lo_i, lo_tight)
                new_hi = min(hi_i, hi_tight)
                if new_lo <= new_hi:
                    lo_i = new_lo
                    hi_i = new_hi
                    _containment_tightened += 1
                # else: tightened bounds infeasible — fall through to
                # original bounds (stationary median will catch it)
        if lo_i > hi_i:
            any_infeasible_frame = True
        lo.append(lo_i)
        hi.append(hi_i)
    if _containment_tightened > 0:
        logger.info(
            "[%s] [L1Path] shot %d: tightened LP bounds on %d/%d frames "
            "for active-speaker containment",
            job_id, shot_index, _containment_tightened, n,
        )

    # ── C1: Tail-lock constraint ──
    # For the last TAIL_LOCK_SECONDS of the shot, tighten LP bounds around
    # the active speaker so the camera holds on them through the shot boundary.
    from backend.services.required_regions import _is_gaming_mode
    _tail_locked = 0
    if not _is_gaming_mode(content_type) and per_frame_containment and n > 0:
        tail_cutoff = per_frame_bounds[-1]["t"] - TAIL_LOCK_SECONDS
        for i in range(n):
            if per_frame_bounds[i]["t"] <= tail_cutoff:
                continue
            if i < len(per_frame_containment) and per_frame_containment[i] is not None:
                active_cx = float(per_frame_containment[i]["active_cx"])
                new_hi = min(hi[i], active_cx + 0.02)
                new_lo = max(lo[i], active_cx - 0.02)
                if new_lo <= new_hi:
                    hi[i] = new_hi
                    lo[i] = new_lo
                    _tail_locked += 1
    if _tail_locked > 0:
        logger.info(
            "[%s] [L1Path] shot %d: tail-locked LP bounds on %d frames "
            "(last %.0fms)",
            job_id, shot_index, _tail_locked, TAIL_LOCK_SECONDS * 1000,
        )

    # ── C3: Tail-target blending ──
    # When a next-shot intent is provided, blend the last frame's target
    # toward the next shot's active speaker cx. This biases the LP toward
    # a smooth handoff without modifying the LP formulation itself.
    if tail_target is not None and n > 0:
        tail_weight_normalized = tail_weight / (1.0 + tail_weight)
        targets[-1] = targets[-1] * (1.0 - tail_weight_normalized) + tail_target * tail_weight_normalized

    # If even one frame's per-frame bounds are geometrically infeasible
    # (required bbox wider than crop) the LP will reject. Fall straight
    # to the median path and let _stationary_median_keyframes decide
    # whether to PADDED.
    if any_infeasible_frame:
        kf, status, held = _stationary_median_keyframes(
            per_frame_bounds, crop_half_width,
            reason="per_frame_bounds_infeasible",
        )
        if status == "success":
            logger.info(
                "[%s] [L1Path] shot %d: T=%d, per-frame bounds "
                "infeasible → stationary-median fallback cx=%.3f",
                job_id, shot_index, n, kf[0][1] if kf else 0.5,
            )
            return kf, "median_fallback", held
        return [], "failed", 0.0

    # ── (2) First LP call with full penalties ──
    def _call_lp(lam2: float) -> list:
        from backend.services._autoflip_lp import solve_autoflip_lp
        return solve_autoflip_lp(
            targets, lo, hi,
            lam1=1.0, lam2=lam2, lam3=100.0, lam4=100.0,
        )

    try:
        solved = _call_lp(lam2=20.0)
    except Exception as exc:
        logger.info(
            "[%s] [L1Path] shot %d: first LP raised %s — retrying",
            job_id, shot_index, exc,
        )
        solved = None

    # ── (3) Retry with relaxed velocity penalty on failure ──
    # The velocity term (lam2) is the usual culprit for short-shot
    # infeasibility: the LP wants cx_t to stay near cx_{t-1} but the
    # per-frame data fidelity plus crop bounds leave no feasible region
    # that meets the velocity budget. Dropping lam2 by 10× effectively
    # unconstrains inter-frame velocity.
    if not solved or len(solved) != n:
        try:
            solved_retry = _call_lp(lam2=2.0)
        except Exception as exc:
            logger.info(
                "[%s] [L1Path] shot %d: retry without velocity "
                "constraint raised %s",
                job_id, shot_index, exc,
            )
            solved_retry = None

        if solved_retry and len(solved_retry) == n:
            logger.info(
                "[%s] [L1Path] shot %d: retry without velocity "
                "constraint succeeded",
                job_id, shot_index,
            )
            solved = solved_retry
            _status = "retry_success"
        else:
            # ── (4) Median fallback ──
            kf, st, held = _stationary_median_keyframes(
                per_frame_bounds, crop_half_width,
                reason="lp_double_failure",
            )
            if st == "success":
                logger.info(
                    "[%s] [L1Path] shot %d: LP failed both passes, "
                    "stationary-median fallback cx=%.3f",
                    job_id, shot_index, kf[0][1] if kf else 0.5,
                )
                return kf, "median_fallback", held
            # ── (5) True PADDED ──
            return [], "failed", 0.0
    else:
        _status = "success"

    # Final clamp + held-still measurement.
    held = 0
    keyframes = []
    last_cx = None
    for i, b in enumerate(per_frame_bounds):
        cx = max(crop_half_width, min(1.0 - crop_half_width, float(solved[i])))
        if last_cx is not None and abs(cx - last_cx) < L1_HELD_STILL_TOL:
            held += 1
        last_cx = cx
        keyframes.append((float(b["t"]), cx, float(b["cy"])))

    held_frac = held / max(n - 1, 1)
    return keyframes, _status, held_frac


def _check_containment_at_cx(
    cx: float,
    crop_half_width: float,
    shot_frames: list,
    shot_index: int,
    job_id: str = "",
) -> bool:
    """Return True if the chosen cx contains all face_containment regions.

    For each frame that has a face_containment region, verify the crop
    [cx - crop_half_width, cx + crop_half_width] fully contains the
    containment region. If any frame fails, log and return False.
    """
    for regs in shot_frames:
        for r in regs:
            if getattr(r, "source", "") != "face_containment":
                continue
            c_left = r.cx - r.half_width
            c_right = r.cx + r.half_width
            crop_left = cx - crop_half_width
            crop_right = cx + crop_half_width
            if c_left < crop_left - 1e-6 or c_right > crop_right + 1e-6:
                logger.info(
                    "[CameraSolver] rejecting STATIONARY for shot %d: "
                    "active speaker containment would clip at t=%.2f",
                    shot_index, float(r.timestamp),
                )
                return False
    return True


def solve_shot(
    shot,
    regions_per_frame: list,
    source_aspect: float,
    params: Optional[SolverParams] = None,
    job_id: str = "",
    content_type=None,
    tail_target: Optional[float] = None,
    tail_weight: float = 0.5,
) -> ShotCamera:
    """Pick the AutoFlip-style camera mode for a single shot.

    v4 union-bbox flow:
        TEST 1: union fits stationary crop                 → STATIONARY
        TEST 2: trajectory is monotonic                    → TRACKING (L1)
        TEST 3: union fits with discrete zoom-out 1.1-1.3  → STATIONARY_ZOOMED
        TEST 4: any feasible smoothed trajectory            → PANNING (L1)
        FALLBACK: zero required regions                    → PADDED

    PADDED is now reserved for shots with literally no data — the v4
    target is <5% of shots, down from ~18% in v3. Wider unions get
    absorbed by STATIONARY_ZOOMED instead of letterboxing.

    Args:
        shot: Shot dataclass with .index, .start, .end
        regions_per_frame: list of list[RequiredRegion] from required_regions.py
        source_aspect: width/height of source video (e.g. 16/9 = 1.778)
        params: Content-type-specific solver parameters (None = generic defaults)
        job_id: for telemetry.
        content_type: content type for content-aware safety margin.

    Returns:
        ShotCamera with mode and keyframes.
    """
    if params is None:
        params = _DEFAULT_PARAMS

    # Filter to this shot's frames; prefer "required" tier, fall back to all.
    shot_frames = []
    for regs in regions_per_frame:
        if not regs or not (shot.start <= regs[0].timestamp <= shot.end):
            continue
        required = [r for r in regs if r.tier == "required"]
        if required:
            shot_frames.append(required)
        elif regs:
            shot_frames.append(regs)

    if not shot_frames:
        # PADDED: literally no signal. Should be rare with v4 wiring.
        return ShotCamera(
            shot_index=shot.index, start=shot.start, end=shot.end,
            mode=CameraMode.PADDED,
            keyframes=[],
            reason="no_required_regions_in_shot",
        )

    # Max crop half-width in normalized 0-1 source coords.
    # At 16:9 source, a 9:16 crop taking full source height:
    #   width = (9/16) / (16/9) ≈ 0.316
    crop_half_width = (CROP_ASPECT / source_aspect) / 2.0
    crop_w_needed = crop_half_width * 2.0

    # Per-frame x-axis bounds + the shot-wide union bbox.
    per_frame_bounds = []
    all_regions: list = []
    for regs in shot_frames:
        lefts = [float(r.cx) - float(r.half_width) for r in regs]
        rights = [float(r.cx) + float(r.half_width) for r in regs]
        per_frame_bounds.append({
            "t": regs[0].timestamp,
            "left": min(lefts),
            "right": max(rights),
            "center": (min(lefts) + max(rights)) / 2,
            "width": max(rights) - min(lefts),
            "cy": sum(float(r.cy) for r in regs) / len(regs),
        })
        all_regions.extend(regs)

    union = compute_union_bbox(all_regions)
    cy_mean = float(np.mean([b["cy"] for b in per_frame_bounds]))

    # Content-aware safety margin (B2).
    _zoom_safety = _stationary_zoom_safety(content_type)

    # ── TEST 1: STATIONARY (union fits with safety margin) ──
    # Add the per-content-type stationary_slack on top of the safety
    # margin so dialogue / podcast modes can absorb tiny drift without
    # going to TRACKING.
    if union.width <= crop_w_needed * _zoom_safety + params.stationary_slack:
        cx = float(np.clip(union.cx, crop_half_width, 1 - crop_half_width))
        # Sanity-check every frame's per-frame union still fits
        all_fit = all(
            b["left"] >= cx - crop_half_width - 0.01
            and b["right"] <= cx + crop_half_width + 0.01
            for b in per_frame_bounds
        )
        if all_fit:
            # B3: active-speaker containment feasibility check.
            if not _check_containment_at_cx(
                cx, crop_half_width, shot_frames, shot.index, job_id,
            ):
                pass  # fall through to TRACKING / PANNING
            else:
                return ShotCamera(
                    shot_index=shot.index, start=shot.start, end=shot.end,
                    mode=CameraMode.STATIONARY,
                    keyframes=[(shot.start, cx, cy_mean), (shot.end, cx, cy_mean)],
                    reason=f"union_width_{union.width:.3f}_fits_stationary",
                    zoom=1.0,
                )

    # ── B5: build per-frame containment data for L1 bounds tightening ──
    _per_frame_containment: list = []
    for regs in shot_frames:
        cdata = None
        for r in regs:
            if getattr(r, "source", "") == "face_containment":
                cdata = {
                    "active_cx": float(r.cx),
                    "active_half_width": float(r.half_width),
                }
                break  # one containment region per frame is enough
        _per_frame_containment.append(cdata)

    # ── TEST 2: TRACKING (monotonic trajectory, L1-solved path) ──
    cx_timeline = [(b["t"], b["center"]) for b in per_frame_bounds]
    cx_timeline.sort()
    # v4.1 Fix 4: is_monotonic requires >=3 points but not is a
    # prerequisite for success. Short shots (T<=5) bypass the LP and
    # use stationary-median — still valid as TRACKING output. Gate by
    # the solver's return status, not a separate length check.
    tracking_ok_statuses = ("success", "retry_success", "short_bypass", "median_fallback")
    if len(cx_timeline) >= 3 and is_monotonic(cx_timeline, tolerance=0.05):
        keyframes, status, held = _l1_track_keyframes(
            per_frame_bounds, crop_half_width,
            job_id=job_id, shot_index=shot.index,
            per_frame_containment=_per_frame_containment,
            content_type=content_type,
            tail_target=tail_target,
            tail_weight=tail_weight,
        )
        if status in tracking_ok_statuses:
            logger.info(
                "[%s] [L1Path] shot %d: T=%d frames, solver status=%s, "
                "mode=tracking, held-still fraction=%.2f",
                job_id, shot.index, len(per_frame_bounds), status, held,
            )
            return ShotCamera(
                shot_index=shot.index, start=shot.start, end=shot.end,
                mode=CameraMode.TRACKING,
                keyframes=keyframes,
                reason=f"monotonic_l1_{status}_held={held:.2f}",
                zoom=1.0,
            )
        # L1 failed for tracking — fall through to zoom / panning attempts.

    # ── TEST 3: STATIONARY_ZOOMED (union fits with discrete zoom-out) ──
    for zoom in STATIONARY_ZOOM_STEPS:
        zoomed_w = crop_w_needed * zoom * STATIONARY_ZOOM_SAFETY
        if union.width <= zoomed_w:
            zoom_half = (crop_w_needed * zoom) / 2.0
            cx = float(np.clip(union.cx, zoom_half, 1 - zoom_half))
            # B3: active-speaker containment feasibility check.
            if not _check_containment_at_cx(
                cx, zoom_half, shot_frames, shot.index, job_id,
            ):
                continue  # try next zoom step or fall through
            return ShotCamera(
                shot_index=shot.index, start=shot.start, end=shot.end,
                mode=CameraMode.STATIONARY_ZOOMED,
                keyframes=[(shot.start, cx, cy_mean), (shot.end, cx, cy_mean)],
                reason=f"union_width_{union.width:.3f}_zoom_{zoom:.1f}",
                zoom=float(zoom),
            )

    # ── TEST 4: PANNING (smoothed L1 trajectory) ──
    # Use the L1 solver here too — same call, no monotonicity gate. The
    # LP's velocity penalty handles non-monotonic motion gracefully.
    # v4.1 Fix 4: accept all ok statuses, not just "success". Short-shot
    # bypass and median-fallback both produce valid PANNING output.
    pan_ok_statuses = ("success", "retry_success", "short_bypass", "median_fallback")
    if len(per_frame_bounds) >= 1:
        keyframes, status, held = _l1_track_keyframes(
            per_frame_bounds, crop_half_width,
            job_id=job_id, shot_index=shot.index,
            per_frame_containment=_per_frame_containment,
            content_type=content_type,
            tail_target=tail_target,
            tail_weight=tail_weight,
        )
        if status in pan_ok_statuses:
            logger.info(
                "[%s] [L1Path] shot %d: T=%d frames, solver status=%s, "
                "mode=panning, held-still fraction=%.2f",
                job_id, shot.index, len(per_frame_bounds), status, held,
            )
            return ShotCamera(
                shot_index=shot.index, start=shot.start, end=shot.end,
                mode=CameraMode.PANNING,
                keyframes=keyframes,
                reason=f"l1_panning_{status}_held={held:.2f}",
                zoom=1.0,
            )

    # ── FALLBACK: PADDED (truly geometrically infeasible) ──
    # v4.1 Fix 4: at this point even stationary-median couldn't place
    # the crop in frame — the required-region union is wider than the
    # maximum zoom step can absorb. This is rare and truly unfixable
    # without letting the crop clip subjects.
    logger.info(
        "[%s] [L1Path] shot %d: T=%d frames, all modes + L1 retries + "
        "stationary-median fallback failed — PADDED",
        job_id, shot.index, len(per_frame_bounds),
    )
    return ShotCamera(
        shot_index=shot.index, start=shot.start, end=shot.end,
        mode=CameraMode.PADDED,
        keyframes=[],
        reason="geometrically_infeasible",
    )


def apply_cross_shot_handoff(shots: list, content_type=None) -> list:
    """Post-process shot cameras for smooth cross-shot handoff.

    For back-to-back cuts (gap < 0.05s): no modification (cut is the handoff).
    For held gaps (>= 0.05s): ease last 150ms of shot_i toward shot_i+1's
    first keyframe, bounded to max 0.04 displacement.

    Skipped for gaming content.
    """
    from backend.services.required_regions import _is_gaming_mode
    if _is_gaming_mode(content_type):
        return shots

    handoff_eased = 0
    handoff_cut = 0

    for i in range(len(shots) - 1):
        a = shots[i]
        b = shots[i + 1]
        if not a.keyframes or not b.keyframes:
            continue

        gap = b.start - a.end
        if gap < 0.05:
            # Back-to-back cut — no interpolation
            handoff_cut += 1
            continue

        # Held gap: ease last 150ms of shot a toward shot b's first kf
        next_cx = b.keyframes[0][1]
        ease_window = 0.15  # 150ms
        max_displacement = 0.04

        modified = False
        new_kfs = list(a.keyframes)
        for ki in range(len(new_kfs) - 1, -1, -1):
            t, cx, cy = new_kfs[ki]
            if t < a.end - ease_window:
                break
            # How far through the ease window are we? (0 at start, 1 at end)
            progress = (t - (a.end - ease_window)) / ease_window if ease_window > 0 else 1.0
            progress = max(0.0, min(1.0, progress))
            # Cosine ease
            import math
            ease = 0.5 * (1.0 - math.cos(math.pi * progress))
            # Bounded displacement
            desired_shift = (next_cx - cx) * ease
            actual_shift = max(-max_displacement, min(max_displacement, desired_shift))
            new_cx = cx + actual_shift
            new_kfs[ki] = (t, new_cx, cy)
            modified = True

        if modified:
            a.keyframes = new_kfs
            handoff_eased += 1

    if handoff_eased > 0 or handoff_cut > 0:
        logger.info(
            "[CameraSolver] handoff_eased=%d, handoff_cut=%d, handoff_skipped_gaming=%d",
            handoff_eased, handoff_cut, 0,
        )

    return shots


def solve_all_shots(
    shots: list,
    regions_per_frame: list,
    source_width: int = 1920,
    source_height: int = 1080,
    content_type=None,
    job_id: str = "",
) -> List[ShotCamera]:
    """Solve camera mode for all shots in a video.

    Args:
        shots: list[Shot] from shot_detector
        regions_per_frame: list of list[RequiredRegion] from required_regions
        source_width, source_height: source video dimensions
        content_type: ClipContentType for per-content-type tuning (None = generic)
        job_id: for logging

    Returns:
        list[ShotCamera], one per shot
    """
    source_aspect = source_width / source_height if source_height > 0 else 16 / 9
    params = get_params_for_content_type(content_type) if content_type else _DEFAULT_PARAMS

    results = []
    mode_counts = {
        CameraMode.STATIONARY.value: 0,
        CameraMode.STATIONARY_ZOOMED.value: 0,
        CameraMode.TRACKING.value: 0,
        CameraMode.PANNING.value: 0,
        CameraMode.PADDED.value: 0,
    }
    # v4.1 Fix 4: per-status breakdown for the L1 solver. Helps the
    # verifier tell the difference between "LP worked cleanly" and
    # "LP needed the short-shot bypass or velocity retry to escape
    # infeasibility" — important for tuning.
    l1_status_counts = {
        "success": 0,
        "retry_success": 0,
        "short_bypass": 0,
        "median_fallback": 0,
    }
    # B5 telemetry: count shots where stationary was rejected by
    # containment and where tracking tails were locked (Phase C stub).
    stationary_rejected_by_containment = 0
    tracking_tail_locked = 0  # placeholder for Phase C
    # Fix 5: padded-shot center inheritance.
    # Track the most recent non-PADDED crop cx. When a shot falls through
    # to PADDED, give it two keyframes at the previous shot's cx instead
    # of default-centering at 0.5 (which yanks the crop back to the
    # middle on every solver failure). Dramatically reduces visual
    # disruption on clips with frequent short shots the LP rejects.
    _last_cx_inherited: float | None = None
    crop_half_width_solve = (CROP_ASPECT / source_aspect) / 2.0
    for si, shot in enumerate(shots):
        # ── C3: look-ahead to next shot's active speaker cx ──
        _next_tail_target: Optional[float] = None
        if si + 1 < len(shots):
            next_shot = shots[si + 1]
            for regs in regions_per_frame:
                if not regs:
                    continue
                _ts = regs[0].timestamp
                if _ts < next_shot.start:
                    continue
                if _ts > next_shot.end:
                    break
                for r in regs:
                    if getattr(r, "is_active_speaker", False):
                        _next_tail_target = float(r.cx)
                        break
                if _next_tail_target is not None:
                    break

        camera = solve_shot(
            shot, regions_per_frame, source_aspect, params, job_id=job_id,
            content_type=content_type,
            tail_target=_next_tail_target,
        )
        if camera.mode == CameraMode.PADDED and _last_cx_inherited is not None:
            inherited_cx = float(
                max(crop_half_width_solve,
                    min(1.0 - crop_half_width_solve, _last_cx_inherited))
            )
            camera = ShotCamera(
                shot_index=camera.shot_index,
                start=camera.start,
                end=camera.end,
                mode=CameraMode.STATIONARY,
                keyframes=[
                    (camera.start, inherited_cx, 0.5),
                    (camera.end, inherited_cx, 0.5),
                ],
                reason=f"padded_inherited_cx_{inherited_cx:.3f}",
                zoom=1.0,
            )
        if camera.mode != CameraMode.PADDED and camera.keyframes:
            _last_cx_inherited = float(camera.keyframes[-1][1])
        results.append(camera)
        mode_counts[camera.mode.value] = mode_counts.get(camera.mode.value, 0) + 1
        # Count shots where stationary was rejected by containment:
        # these end up as TRACKING/PANNING instead of STATIONARY.
        if camera.mode in (CameraMode.TRACKING, CameraMode.PANNING):
            # The containment check logged a rejection — we detect this by
            # checking if the shot had containment regions and ended up
            # non-stationary despite a narrow union.
            _shot_has_containment = any(
                getattr(r, "source", "") == "face_containment"
                for regs in regions_per_frame
                if regs and shot.start <= regs[0].timestamp <= shot.end
                for r in regs
            )
            if _shot_has_containment and "union_width" not in camera.reason:
                stationary_rejected_by_containment += 1
        # Parse status out of the reason string (encoded by solve_shot
        # as "monotonic_l1_<status>_held=..." / "l1_panning_<status>_held=...").
        for key in l1_status_counts:
            if f"l1_{key}" in camera.reason or f"l1_panning_{key}" in camera.reason:
                l1_status_counts[key] += 1
                break

    # ── C2: cross-shot handoff post-processing ──
    results = apply_cross_shot_handoff(results, content_type=content_type)

    total = max(len(results), 1)
    padded_pct = 100.0 * mode_counts[CameraMode.PADDED.value] / total
    logger.info(
        "[%s] [CameraSolver] shot modes: stationary=%d, stationary_zoomed=%d, "
        "tracking=%d, panning=%d, padded=%d",
        job_id,
        mode_counts[CameraMode.STATIONARY.value],
        mode_counts[CameraMode.STATIONARY_ZOOMED.value],
        mode_counts[CameraMode.TRACKING.value],
        mode_counts[CameraMode.PANNING.value],
        mode_counts[CameraMode.PADDED.value],
    )
    logger.info(
        "[%s] [CameraSolver] L1 status breakdown: success=%d, retry_success=%d, "
        "short_bypass=%d, median_fallback=%d",
        job_id,
        l1_status_counts["success"],
        l1_status_counts["retry_success"],
        l1_status_counts["short_bypass"],
        l1_status_counts["median_fallback"],
    )
    logger.info(
        "[%s] [CameraSolver] padded-fallback rate: %.1f%% (target <5%%)",
        job_id, padded_pct,
    )
    if stationary_rejected_by_containment > 0 or tracking_tail_locked > 0:
        logger.info(
            "[%s] [CameraSolver] containment: stationary_rejected=%d, "
            "tracking_tail_locked=%d",
            job_id, stationary_rejected_by_containment, tracking_tail_locked,
        )
    return results
