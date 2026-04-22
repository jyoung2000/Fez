"""Stage-7 auto-fix — Blueprint v2 Phase 3.

For each ``PostRenderIssue`` the critic flags, look up a
``FixStrategy`` that translates the issue into concrete
``ReframeConfig`` overrides + crop padding + resolve window. The
strategy is then applied to a segment of the render plan via
``apply_fix_to_segment``: re-solve the window with the overrides,
splice the re-solved RenderOps back into the original plan, verify
coverage.

This module is advisory — it never raises. Callers should treat its
output as "try this plan, fall back to the original if it doesn't
work". Tests pin each strategy's overrides so later matrix edits
can't silently change fix behavior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from backend.services.post_render_critic import (
    AUTO_FIX_KINDS,
    IssueKind,
    PostRenderIssue,
)
from backend.services.reframe_config import ReframeConfig, get_default_config

logger = logging.getLogger(__name__)


@dataclass
class FixStrategy:
    """What to change when a specific issue kind shows up."""

    issue_kind: str
    # Field → value dict applied via ``ReframeConfig.override``.
    config_overrides: dict = field(default_factory=dict)
    # Additional padding around required regions for the re-solved
    # window (fraction of source width). 0 = no padding.
    widen_crop_frac: float = 0.0
    # Force a specific layout mode for the window (e.g. "single" to
    # dodge a bad split-screen pick). ``None`` = let the solver pick.
    force_layout: Optional[str] = None
    # Symmetric window around the issue timestamp that gets re-solved.
    resolve_window_sec: float = 2.0


# Blueprint v2 Phase 3 — strategy table keyed by issue kind.
#
# Every auto-fix kind has an entry. Strategies intentionally touch
# only a handful of fields so the re-solve is a surgical tweak rather
# than a whole-new-config blast.
_STRATEGIES: dict[str, FixStrategy] = {
    IssueKind.TIGHT_FRAMING: FixStrategy(
        issue_kind=IssueKind.TIGHT_FRAMING,
        # Slightly larger headroom band + pad the crop so the subject
        # sits further from the edges.
        config_overrides={"headroom_min": 0.08, "headroom_max": 0.18},
        widen_crop_frac=0.08,
    ),
    IssueKind.HEAD_OR_CHIN_CLIP: FixStrategy(
        issue_kind=IssueKind.HEAD_OR_CHIN_CLIP,
        config_overrides={"headroom_min": 0.10, "headroom_max": 0.20},
        widen_crop_frac=0.05,
    ),
    IssueKind.PAN_ACROSS_CUT: FixStrategy(
        issue_kind=IssueKind.PAN_ACROSS_CUT,
        # Hard-cut on shot boundaries + tighten the match-cut gate so
        # the solver is more likely to reset than pan across.
        config_overrides={"ease_shot_cut_ms": 0, "match_cut_threshold": 0.05},
    ),
    IssueKind.JITTER: FixStrategy(
        issue_kind=IssueKind.JITTER,
        # Double the velocity penalty and widen the static-hold
        # deadband so small oscillations fall below the tracking gate.
        config_overrides={"lp_lambda_v": 40.0, "deadband_frac": 0.14},
    ),
    IssueKind.WRONG_SUBJECT: FixStrategy(
        issue_kind=IssueKind.WRONG_SUBJECT,
        # The subject hint from the VLM is the actionable signal here;
        # no config tweak on its own. Caller is expected to promote
        # the hinted subject to a required region.
        config_overrides={},
    ),
    IssueKind.TEXT_CUT_OFF: FixStrategy(
        issue_kind=IssueKind.TEXT_CUT_OFF,
        # Text detection should promote to a required region in a
        # later phase; for now widen the crop so edge text survives.
        config_overrides={},
        widen_crop_frac=0.05,
    ),
}


def strategy_for(issue: PostRenderIssue) -> Optional[FixStrategy]:
    """Return a ``FixStrategy`` for an auto-fixable issue, else ``None``."""
    kind = getattr(issue, "kind", IssueKind.OTHER)
    if kind not in AUTO_FIX_KINDS:
        return None
    return _STRATEGIES.get(kind)


def snap_to_shot_boundaries(
    start: float, end: float, shot_boundaries: list, tol: float = 0.5,
) -> tuple[float, float]:
    """Snap ``start`` / ``end`` to nearby shot-cut timestamps.

    Re-solving across a shot cut usually makes the underlying issue
    worse; snapping the window to the nearest boundary keeps the
    segment coherent. ``tol`` is the max snap distance in seconds.
    """
    new_start = float(start)
    new_end = float(end)
    for b in shot_boundaries or []:
        try:
            bf = float(b)
        except (TypeError, ValueError):
            continue
        if abs(bf - new_start) < tol:
            new_start = bf
        if abs(bf - new_end) < tol:
            new_end = bf
    if new_end <= new_start:
        return float(start), float(end)
    return new_start, new_end


def _resolve_window(
    issue: PostRenderIssue,
    strategy: FixStrategy,
    *,
    shot_boundaries: list,
    total_duration: float,
) -> tuple[float, float]:
    """Compute the ``[start, end]`` to re-solve around an issue."""
    half = max(0.25, float(strategy.resolve_window_sec))
    start = max(0.0, float(issue.t) - half)
    end = min(float(total_duration), float(issue.t) + half)
    return snap_to_shot_boundaries(start, end, shot_boundaries or [])


def apply_fix_to_segment(
    original_plan,
    issue: PostRenderIssue,
    strategy: FixStrategy,
    *,
    dense_faces: list,
    active_speaker_events: list,
    shot_boundaries: list,
    content_type: str,
    source_width: int,
    source_height: int,
    source_fps: float,
    config: Optional[ReframeConfig] = None,
    job_id: str = "",
):
    """Re-solve the segment around ``issue.t`` with the strategy's
    overrides and splice the new ops back into the original plan.

    Returns the new ``RenderPlan`` on success, or ``None`` if the
    re-solve failed / the spliced plan doesn't pass coverage
    verification. Never raises.
    """
    from backend.services.human_reframe import (
        HumanReframeInputs, run_human_reframe,
    )
    from backend.services.human_render_plan_adapter import (
        render_plan_from_human_plan,
        verify_frame_coverage,
    )
    from backend.services.render_plan_splice import splice_segment

    total_duration = float(getattr(original_plan, "total_duration_sec", 0.0))
    start, end = _resolve_window(
        issue, strategy,
        shot_boundaries=shot_boundaries,
        total_duration=total_duration,
    )
    if end - start <= 0.1:
        logger.info(
            "[%s] autofix: resolve window collapsed around t=%.2f; skipping",
            job_id, issue.t,
        )
        return None

    seg_faces = [
        f for f in (dense_faces or [])
        if start <= float(getattr(f, "timestamp", 0.0)) <= end
    ]
    seg_events = [
        e for e in (active_speaker_events or [])
        if float(getattr(e, "end", 0.0)) >= start
        and float(getattr(e, "start", 0.0)) <= end
    ]
    seg_shots = [
        float(b) for b in (shot_boundaries or [])
        if start <= float(b) <= end
    ]

    base_cfg = (config or get_default_config())
    local_cfg = base_cfg.override(**strategy.config_overrides)

    inputs = HumanReframeInputs(
        duration_sec=end - start,
        source_w=int(source_width),
        source_h=int(source_height),
        content_type=str(content_type or "generic"),
        dense_faces=seg_faces,
        active_speaker_events=seg_events,
        shot_boundaries=seg_shots,
    )
    try:
        new_human = run_human_reframe(inputs, config=local_cfg)
    except Exception as e:
        logger.warning(
            "[%s] autofix segment re-solve failed at t=%.2f: %s",
            job_id, issue.t, e,
        )
        return None

    try:
        seg_plan = render_plan_from_human_plan(
            new_human,
            source_width=int(source_width),
            source_height=int(source_height),
            source_fps=float(source_fps or 30.0),
            config=local_cfg,
            content_type=str(content_type or ""),
        )
    except Exception as e:
        logger.warning(
            "[%s] autofix render_plan_from_human_plan failed at t=%.2f: %s",
            job_id, issue.t, e,
        )
        return None

    try:
        spliced = splice_segment(
            original_plan, seg_plan, start=start, end=end,
        )
    except Exception as e:
        logger.warning(
            "[%s] autofix splice failed at [%.2f, %.2f]: %s",
            job_id, start, end, e,
        )
        return None

    report = verify_frame_coverage(spliced)
    if not report.ok:
        logger.info(
            "[%s] autofix coverage check failed at [%.2f, %.2f]: "
            "gaps=%d overlaps=%d oor=%d — reverting",
            job_id, start, end,
            len(report.gaps), len(report.overlaps), len(report.out_of_range_rects),
        )
        return None

    logger.info(
        "[%s] autofix applied kind=%s t=%.2f window=[%.2f,%.2f] ops=%d",
        job_id, issue.kind, issue.t, start, end, len(spliced.ops),
    )
    return spliced


def classify_clip_confidence(
    report,
    *,
    applied_fixes: int = 0,
) -> str:
    """Blueprint v2 Phase 3 — translate a ``PostRenderReport`` into
    one of ``high`` | ``medium`` | ``low``.

    ``low`` fires when the report has any structural issue — the
    source content has genuinely lost the subject (or similar) and
    no amount of re-solving will help.

    ``high`` fires when there are no remaining issues (``ok`` and no
    flagged items). Everything in between is ``medium`` — fixable or
    cosmetic issues that we either patched or chose to live with.
    """
    if report is None:
        return "high"
    structural = [
        i for i in getattr(report, "issues", []) or []
        if getattr(i, "kind", "") in (
            IssueKind.SUBJECT_LEFT_FRAME,
            IssueKind.OUTPAINT_HALLUC,
            IssueKind.HUD_ADJACENT,
        )
    ]
    if structural:
        return "low"
    if not getattr(report, "issues", None):
        return "high"
    return "medium"
