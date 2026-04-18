"""Cached RenderPlan → legacy export keyframes converter.

Phase 10 follow-up: the preview player reads the cached
``job.render_plan`` dict verbatim via
``GET /api/jobs/{id}/render_plan``. That dict is built from the
full-fidelity ``ReframeSegment`` list at pipeline completion time
(``pipeline.py:_rp_dict["debug"] = build_debug_payload(...)``), so
it preserves:

- ``motion_path`` — per-frame walking tracks for VLOG subjects,
  Stage 10 L1 camera paths, anime action pans.
- ``content_type`` — the per-clip routing the segmenter produced.
- Normalized primary/secondary rects for split / stacked / grid
  layouts.

The **export** path, historically, has rebuilt its own RenderPlan
from the stripped ``SceneDescription`` list via
``clip_exporter._extract_render_plan_segments``. That helper
hard-codes ``motion_path=None`` / ``content_type="unknown"`` /
``hard_constraints=None`` / ``active_slot=None`` (see
``clip_exporter.py:2679``), which meant a walking vlog rendered as
a smooth pan in the preview but as a static centre crop at the
segment midpoint in the exported MP4.

This module fixes that for the one content type where it's a real
editorial regression — VLOG with ``allow_motion_tracking=True``.
It walks the cached plan's ``ops`` list (already JSON-serialized
via ``RenderPlan.to_dict()``) and emits the same
``list[tuple[float, int]]`` keyframe schedule that the legacy
ffmpeg filter builder consumes via the ``frontend_subject_keyframes``
and camera-solver paths in ``clip_exporter.export_clip``.

Every tracking_crop op with a ``motion_path`` contributes one
keyframe per motion keypoint so the export filter interpolates
along the same camera path the preview shows. Static ops
(crop / wide_master / split_screen / stacked_gameplay / blur_fill /
grid_2x2) contribute one keyframe at their start so segment
boundaries still land as hard cuts.

Pure, side-effect-free, import-lightweight — the tests in
``backend/tests/test_export_render_plan_parity.py`` exercise it
without pulling in the rest of ``clip_exporter``'s dependency
graph (cv2, MediaPipe, ffmpeg wrappers, pydantic_settings, ...).
"""

from __future__ import annotations

from typing import Optional


def keyframes_from_cached_render_plan(
    cached_plan: dict,
    *,
    clip_start: float,
    clip_end: float,
) -> Optional[list[tuple[float, int]]]:
    """Convert a cached ``RenderPlan`` dict into export keyframes.

    Args:
        cached_plan: The ``job.render_plan`` dict as written by
            ``RenderPlan.to_dict()``. Each op has ``start_sec`` /
            ``end_sec`` in source-video-absolute seconds,
            ``primary_rect`` with normalized ``x`` / ``y`` / ``w`` /
            ``h`` in [0, 1], optional ``motion_path`` (list of
            ``{"t": float, "rect": {x, y, w, h}}`` entries where ``t``
            is **op-relative**), and a ``kind`` string.
        clip_start: Clip start in source-video-absolute seconds.
        clip_end: Clip end in source-video-absolute seconds. Ops
            outside this window are skipped; straddling ops
            contribute only their in-range keyframes. Times are
            rebased so ``t=0`` is ``clip_start``.

    Returns:
        A sorted, de-duplicated list of ``(t_seconds_clip_relative,
        crop_center_x_percent_0_100)`` tuples, or ``None`` when the
        plan has no ops overlapping the clip window or the input is
        malformed. Never raises — the caller falls back to the
        legacy per-clip rebuild pipeline on ``None``.
    """
    if not isinstance(cached_plan, dict):
        return None
    ops = cached_plan.get("ops") or []
    if not ops:
        return None

    kf: list[tuple[float, int]] = []

    for op in ops:
        if not isinstance(op, dict):
            continue
        try:
            op_start = float(op.get("start_sec", 0.0))
            op_end = float(op.get("end_sec", 0.0))
        except (TypeError, ValueError):
            continue
        if op_end <= clip_start or op_start >= clip_end:
            continue

        primary_rect = op.get("primary_rect") or {}
        center_pct = _rect_center_pct(primary_rect)

        motion_path = op.get("motion_path") or []
        if motion_path and isinstance(motion_path, list):
            any_in_range = False
            for mk in motion_path:
                if not isinstance(mk, dict):
                    continue
                try:
                    mk_t_op_rel = float(mk.get("t", 0.0))
                except (TypeError, ValueError):
                    continue
                mk_t_abs = op_start + mk_t_op_rel
                if mk_t_abs < clip_start - 1e-6:
                    continue
                if mk_t_abs > clip_end + 1e-6:
                    continue
                mk_center = _rect_center_pct(
                    mk.get("rect"), fallback=center_pct,
                )
                kf.append((
                    round(max(0.0, mk_t_abs - clip_start), 3),
                    int(round(mk_center)),
                ))
                any_in_range = True
            # Fall through to a static boundary keyframe only if
            # the motion path contributed nothing inside the clip
            # window — otherwise it's already represented.
            if any_in_range:
                continue

        t_boundary = max(op_start, clip_start) - clip_start
        kf.append((round(max(0.0, t_boundary), 3), int(round(center_pct))))

    if not kf:
        return None

    kf.sort()

    # Drop exact duplicates the motion_path / boundary emitters
    # can produce at segment junctions — they confuse the
    # downstream easing code which keys on (t, x) identity.
    deduped: list[tuple[float, int]] = []
    for entry in kf:
        if deduped and deduped[-1] == entry:
            continue
        deduped.append(entry)
    return deduped


def _rect_center_pct(rect: dict | None, fallback: float = 50.0) -> float:
    """Convert a normalized [0, 1] ``Rect`` dict to a crop-center x %.

    Clamped to [0, 100] so downstream integer rounding never produces
    a negative keyframe value even if the cached plan has a rect that
    spills outside the source frame.
    """
    if not isinstance(rect, dict):
        return fallback
    try:
        x = float(rect.get("x", 0.0))
        w = float(rect.get("w", 1.0))
    except (TypeError, ValueError):
        return fallback
    return max(0.0, min(100.0, (x + w / 2.0) * 100.0))
