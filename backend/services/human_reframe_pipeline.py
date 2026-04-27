"""Phase A — wire the SOTA human-reframe path into the production pipeline.

This module owns three concerns so :mod:`backend.services.pipeline`
itself stays nearly unchanged:

1. **Flag + allowlist gating.** ``CLIPAI_HUMAN_REFRAME_PIPELINE`` must
   be truthy AND the clip's ``content_type`` must be in
   :data:`HUMAN_REFRAME_ALLOWED_CONTENT_TYPES` for the human path to be
   picked. Default OFF — the legacy segmenter remains the production
   default until enough comparison renders justify the switch.

2. **A shim** :func:`render_plan_to_reframe_segments` that converts a
   :class:`RenderPlan` (the human-reframe output) into the
   ``ReframeSegment`` list the rest of the pipeline already consumes
   (``scenes``, ``layout_timeline``, exporter wiring etc.). One
   ``RenderOp`` becomes one ``ReframeSegment`` — the output is
   structurally identical to what
   ``reframe_segmenter.build_reframe_segments`` would produce, so the
   downstream code path is unchanged.

3. **A single-call entry point** :func:`try_run_human_reframe_pipeline`
   that runs the full ``run_human_reframe →
   render_plan_from_human_plan`` chain, verifies coverage, and returns
   ``(segments, render_plan)`` on success or ``None`` on any
   gating-miss / inputs-missing / runtime failure. Callers fall through
   to the legacy segmenter when ``None`` is returned.

The Phase A goal is to MAKE the SOTA path runnable in production for
two specific content types, not to USE it as the default. Adding a
content type to the allowlist requires its own bench-validation
evidence (every quality axis within spec on at least one real clip of
that type).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


# ── Allowlist ────────────────────────────────────────────────────
#
# Content types the human-reframe path is cleared for in production.
#
# Phase E (2026-SOTA rollout): the allowlist now covers every
# content type because the editorial planner (Phase E) provides a
# per-genre playbook that drives the LP solver from a single LLM
# call. The previous gate (multi_speaker_panel + talking_head only)
# was needed when the only "genre intelligence" was the per-content-
# type code in genre_refinements.py — that limitation no longer
# applies. See docs/sota_reframe_rollout.md for the migration ADR.
#
# Emergency rollback: set CLIPAI_LEGACY_REFRAME=1 to force the
# legacy reframe_segmenter path for one release.
HUMAN_REFRAME_ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset({
    "multi_speaker_panel", "talking_head",
    "podcast", "vlog", "interview",
    "narrative", "documentary", "cinematic_dialogue",
    "music_video", "concert", "performance",
    "sports", "sports_basketball", "sports_racing",
    "gaming", "gameplay",
    "anime", "animation", "animation_dialogue",
    "tutorial",
    "debate",
})


# ── Flags ────────────────────────────────────────────────────────


def _truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


def is_pipeline_flag_enabled() -> bool:
    """``CLIPAI_HUMAN_REFRAME_PIPELINE`` toggle. Default OFF."""
    return _truthy(os.environ.get("CLIPAI_HUMAN_REFRAME_PIPELINE"))


def is_compare_mode_enabled() -> bool:
    """``CLIPAI_REFRAME_COMPARE`` toggle. Default OFF.

    When set together with the pipeline flag, the analysis stage runs
    BOTH the human-reframe path and the legacy segmenter and keeps
    both render plans available for side-by-side eyeballing. The
    actual two-MP4 export is a follow-up — see
    ``docs/human_reframe_rollout.md``.
    """
    return _truthy(os.environ.get("CLIPAI_REFRAME_COMPARE"))


def pick_reframe_path(content_type: str) -> str:
    """Return ``"human_reframe"`` | ``"autoflip"`` | ``"reframe_segmenter"``.

    Pure function of the env flags + content type. Callers still apply
    their own ``not _is_gameplay`` / ``not _is_continuous`` guards
    after this returns.
    """
    if (
        is_pipeline_flag_enabled()
        and content_type in HUMAN_REFRAME_ALLOWED_CONTENT_TYPES
    ):
        return "human_reframe"
    if _truthy(os.environ.get("USE_AUTOFLIP_REFRAME")):
        return "autoflip"
    return "reframe_segmenter"


# ── RenderPlan → ReframeSegment shim ─────────────────────────────


_OP_KIND_TO_LAYOUT = {
    "crop": "single",
    "tracking_crop": "single",
    "wide_master": "wide_master",
    "blur_fill": "blur_fill",
    "split_screen": "split",
    "stacked_gameplay": "stacked_gameplay",
    "grid_2x2": "grid",
    "motivated_push_in": "single",
    "motivated_pull_out": "single",
}


def _kind_value(kind) -> str:
    return getattr(kind, "value", str(kind))


def render_plan_to_reframe_segments(
    rp,
    *,
    source_width: int,
    source_height: int,
    content_type: str,
):
    """Convert a :class:`RenderPlan` into a list of
    :class:`ReframeSegment`. One RenderOp per ReframeSegment.

    Coordinate conventions match the legacy segmenter:
      * ``subject_x`` / ``subject_y`` are SOURCE pixel coordinates of
        the crop center.
      * ``motion_path`` entries are ``(t_seconds, x_px, y_px)``.
      * ``ease_in_ms`` is the transition duration into this segment.
    """
    from backend.services.reframe_segmenter import ReframeSegment

    segments: list = []
    for op in rp.ops:
        rect = op.primary_rect
        cx = rect.x + rect.w * 0.5
        cy = rect.y + rect.h * 0.5
        subject_x_px = float(cx * source_width)
        subject_y_px = float(cy * source_height)

        motion_path = []
        for kp in (op.motion_path or []):
            kp_cx = kp.rect.x + kp.rect.w * 0.5
            kp_cy = kp.rect.y + kp.rect.h * 0.5
            motion_path.append((
                float(op.start_sec + kp.t),
                float(kp_cx * source_width),
                float(kp_cy * source_height),
            ))

        kind_v = _kind_value(op.kind)
        layout = _OP_KIND_TO_LAYOUT.get(kind_v, "single")
        strategy = "tracking" if op.motion_path else "stationary"

        segments.append(ReframeSegment(
            start=float(op.start_sec),
            end=float(op.end_sec),
            subject_x=subject_x_px,
            subject_y=subject_y_px,
            layout=layout,
            active_slot=op.speaker_slot,
            confidence=0.85,
            reason=f"human_reframe:{op.strategy_label or kind_v}",
            ease_in_ms=int(op.ease_in_ms or 0),
            strategy=strategy,
            content_type=op.content_type or content_type or "unknown",
            motion_path=motion_path or None,
            gaming_layout_mode=op.gaming_layout_mode,
            subject_source="human_reframe",
        ))
    return segments


# ── Single-call entry point ──────────────────────────────────────


def try_run_human_reframe_pipeline(
    *,
    duration_sec: float,
    source_width: int,
    source_height: int,
    source_fps: float,
    content_type: str,
    dense_faces: list,
    active_speaker_events: list,
    shot_boundaries: list,
    beats: Optional[list] = None,
    downbeats: Optional[list] = None,
    job_id: str = "",
):
    """Run the human-reframe pipeline end-to-end.

    Returns ``(segments, render_plan)`` on success, ``None`` on any
    gating miss / insufficient inputs / runtime failure. Callers
    treat ``None`` as "fall through to the legacy segmenter."
    """
    if not (
        is_pipeline_flag_enabled()
        and content_type in HUMAN_REFRAME_ALLOWED_CONTENT_TYPES
    ):
        return None
    if not dense_faces or duration_sec <= 0:
        logger.info(
            "[%s] human-reframe pipeline: insufficient inputs "
            "(dense_faces=%d, duration=%.1f); falling back to legacy",
            job_id, len(dense_faces or []), float(duration_sec),
        )
        return None

    try:
        from backend.services.human_reframe import (
            HumanReframeInputs, run_human_reframe,
        )
        from backend.services.human_render_plan_adapter import (
            render_plan_from_human_plan,
            verify_frame_coverage,
        )
    except Exception as e:
        logger.warning(
            "[%s] human-reframe pipeline: imports unavailable (%s); falling back",
            job_id, e,
        )
        return None

    try:
        inputs = HumanReframeInputs(
            duration_sec=float(duration_sec),
            source_w=int(source_width),
            source_h=int(source_height),
            content_type=content_type or "",
            dense_faces=list(dense_faces),
            active_speaker_events=list(active_speaker_events or []),
            shot_boundaries=list(shot_boundaries or []),
            beats=list(beats or []),
            downbeats=list(downbeats or []),
        )
        human = run_human_reframe(inputs)
        rp = render_plan_from_human_plan(
            human,
            source_width=int(source_width),
            source_height=int(source_height),
            source_fps=float(source_fps),
            content_type=content_type or "",
        )
    except Exception as e:
        logger.warning(
            "[%s] human-reframe pipeline: failed (%s); falling back to legacy",
            job_id, e,
        )
        return None

    try:
        coverage = verify_frame_coverage(rp)
        if not coverage.ok:
            logger.warning(
                "[%s] human-reframe pipeline: coverage failed "
                "(gaps=%d oor=%d); falling back to legacy",
                job_id, len(coverage.gaps),
                len(coverage.out_of_range_rects),
            )
            return None
    except Exception as e:
        logger.warning(
            "[%s] human-reframe pipeline: coverage verify failed (%s); "
            "falling back to legacy", job_id, e,
        )
        return None

    segments = render_plan_to_reframe_segments(
        rp,
        source_width=int(source_width),
        source_height=int(source_height),
        content_type=content_type or "",
    )
    logger.info(
        "[%s] human-reframe pipeline: %d ops → %d segments (ct=%s)",
        job_id, len(rp.ops), len(segments), content_type,
    )
    return segments, rp
