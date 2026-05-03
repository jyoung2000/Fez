"""Pipeline ↔ human-reframe bridge.

Owns two concerns so :mod:`backend.services.pipeline` can stay nearly
unchanged:

  1. ``maybe_override_render_plan`` — one-line hook the pipeline calls
     after the legacy segmenter builds ``_rp``. When the master flag
     ``CLIPAI_HUMAN_REFRAME`` is on AND we have enough inputs (dense
     faces + shot cuts + content type + duration), this runs the full
     human-reframe pipeline and returns the resulting
     :class:`RenderPlan`. Otherwise it returns the input plan unchanged.
     No behavior change is possible with the flag off.

  2. ``run_reframe_on_clip_for_bench`` — the entry point the human-parity
     bench runner calls. Takes only a source video path + content type
     and produces a render-plan dict. Does not require the full async
     pipeline; extracts frames + detects faces on demand using the
     codebase's existing helpers.

Everything about orchestration, safety fallback, and coverage
verification lives here so the pipeline's 4000-line function doesn't
need to grow further. The bridge is completely independent from the
legacy segmenter / AutoFlip segmenter — it consumes the same inputs
from a different angle.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from backend.services.human_parity_metrics import render_plan_to_trajectory
from backend.services.human_reframe import (
    HumanReframeInputs,
    HumanReframePlan,
    run_human_reframe,
)
from backend.services.human_render_plan_adapter import (
    CoverageReport,
    fill_gaps_with_blur,
    render_plan_from_human_plan,
    try_repair_coverage,
    verify_frame_coverage,
)
from backend.services.reframe_config import (
    ReframeConfig,
    get_default_config,
)
from backend.services.render_plan import RenderPlan

logger = logging.getLogger(__name__)


def _flag_enabled() -> bool:
    """Is the human-reframe path enabled?

    Default ON: every run gets the best framing possible. Set
    ``CLIPAI_HUMAN_REFRAME=0`` to force the legacy path (useful for
    AutoFlip-parity fixture runs / A-B comparisons).
    """
    val = os.environ.get("CLIPAI_HUMAN_REFRAME")
    if val is None:
        return True
    return val.lower() in ("1", "true", "yes", "on")


# ── Pipeline hook ─────────────────────────────────────────────────


def maybe_override_render_plan(
    legacy_rp: RenderPlan,
    *,
    dense_faces: list,
    active_speaker_events: list,
    shot_boundaries: list[float],
    content_type: str,
    duration_sec: float,
    source_width: int,
    source_height: int,
    source_fps: float,
    beats: Optional[list[float]] = None,
    downbeats: Optional[list[float]] = None,
    ball_detections: Optional[list] = None,
    car_detections: Optional[list] = None,
    hud_regions: Optional[list] = None,
    impact_frames: Optional[list[float]] = None,
    words: Optional[list] = None,
    audio_peaks: Optional[list] = None,
    motion_beats: Optional[list] = None,
    entrances: Optional[list] = None,
    reactions: Optional[list] = None,
    config: Optional[ReframeConfig] = None,
    job_id: str = "",
    saliency_peaks_per_second: Optional[list] = None,
    shot_advice_list: Optional[list] = None,
    speaker_positions: Optional[dict] = None,
) -> RenderPlan:
    """Return ``legacy_rp`` unchanged unless the human-reframe flag is on
    and the inputs are sufficient to produce a better plan.

    ``saliency_peaks_per_second`` (Layer 1, optional): when supplied,
    the critic auto-repair pass below picks up the saliency-in-crop
    check and may emit ``saliency_widen`` fixes for windows where the
    crop excludes the salient region.

    The contract: caller can treat this as idempotent and safe — never
    raises, always returns a validated plan. On any failure or coverage
    gap we fall back to ``legacy_rp``.
    """
    if not _flag_enabled():
        return legacy_rp
    if not dense_faces:
        logger.info("[%s] human-reframe flag on but no dense faces; keeping legacy",
                    job_id)
        return legacy_rp
    if duration_sec <= 0:
        return legacy_rp

    config = config or get_default_config()

    # Scale the critic repair budget with clip duration so long-form
    # content (full episodes, multi-hour streams) gets enough fixes to
    # cover the back half of the video. The 5-per-minute slope keeps a
    # 60 s clip at the 50-fix floor and lets a 60-minute movie reach
    # 300 fixes. Operators can still pin a fixed budget via
    # ``CLIPAI_CRITIC_BUDGET`` (the env var is honored at config-build
    # time; we only scale up when the explicit budget is at the
    # default-or-lower).
    try:
        _scaled_budget = max(
            int(config.critic_budget_per_clip),
            int(float(duration_sec) / 60.0 * 5.0),
        )
        if _scaled_budget != int(config.critic_budget_per_clip):
            config = config.override(critic_budget_per_clip=_scaled_budget)
    except Exception:
        # Never let budget scaling fail the pipeline.
        pass

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
            ball_detections=list(ball_detections or []),
            car_detections=list(car_detections or []),
            hud_regions=list(hud_regions or []),
            impact_frames=list(impact_frames or []),
            words=list(words or []),
            audio_peaks=list(audio_peaks or []),
            motion_beats=list(motion_beats or []),
            entrances=list(entrances or []),
            reactions=list(reactions or []),
            shot_advice_list=list(shot_advice_list or []),
            speaker_positions=dict(speaker_positions or {}),
        )
        human: HumanReframePlan = run_human_reframe(inputs, config=config)
        new_rp = render_plan_from_human_plan(
            human,
            source_width=source_width,
            source_height=source_height,
            source_fps=source_fps,
            config=config,
            content_type=content_type or "",
        )
    except Exception as e:
        logger.warning("[%s] human-reframe failed (%s); keeping legacy plan",
                       job_id, e)
        return legacy_rp

    # Fix 3.8: local critic + auto-repair pass on the new plan. Local
    # heuristics only; no network. Budget-limited by config.
    try:
        from backend.services.reframe_critic import auto_repair_plan
        if getattr(config, "critic_mode", "learned") != "off":
            new_rp, fixes, _ = auto_repair_plan(
                new_rp, dense_faces=list(dense_faces or []), config=config,
                saliency_peaks_per_second=saliency_peaks_per_second,
            )
            if fixes:
                logger.info(
                    "[%s] human-reframe critic: applied %d fixes (%s)",
                    job_id, len(fixes),
                    ", ".join(sorted({f.reason for f in fixes}))[:120],
                )
    except Exception as e:
        logger.warning("[%s] critic auto-repair failed (%s); continuing", job_id, e)

    # Fix 3.7: coverage is now a repair-in-place cascade. Only fall
    # back to the legacy plan when every repair attempt fails.
    coverage: CoverageReport = verify_frame_coverage(new_rp)
    if not coverage.ok:
        repaired = try_repair_coverage(new_rp, coverage, float(duration_sec))
        if repaired is not None:
            new_rp = repaired
            coverage = verify_frame_coverage(new_rp)
            logger.info(
                "[%s] human-reframe: repaired coverage (%d gaps, %d oor)",
                job_id, len(coverage.gaps), len(coverage.out_of_range_rects),
            )
    if not coverage.ok:
        # Last resort before legacy fallback: fill remaining gaps with
        # blur_fill from the new plan's timeline. Still preserves the
        # new plan's shot/speaker decisions for the segments that work.
        new_rp = fill_gaps_with_blur(new_rp, float(duration_sec))
        coverage = verify_frame_coverage(new_rp)
    if not coverage.ok:
        logger.error(
            "[%s] human-reframe: unrecoverable coverage failure "
            "(gaps=%d overlaps=%d oor=%d zero=%d); reverting to legacy",
            job_id,
            len(coverage.gaps), len(coverage.overlaps),
            len(coverage.out_of_range_rects), coverage.zero_duration_ops,
        )
        return legacy_rp

    logger.info(
        "[%s] human-reframe: %d ops, %.2fs covered, %d zooms, ab=%s, notes=%s",
        job_id,
        len(new_rp.ops), coverage.total_covered_sec,
        len(human.zooms),
        "on" if human.ab.enabled else "off",
        "; ".join(human.notes[:3]),
    )
    return new_rp


# ── Bench entry point ────────────────────────────────────────────


def run_reframe_on_clip_for_bench(
    source_path: str,
    *,
    content_type: Optional[str] = None,
    sample_fps: float = 6.0,
    target_height_px: int = 1920,
    target_aspect: str = "9:16",
    config: Optional[ReframeConfig] = None,
) -> dict:
    """Produce a ``RenderPlan`` dict for one clip without running the
    full async pipeline.

    Used by :mod:`backend.scripts.run_human_parity_bench` to compare
    ours-vs-human crop trajectories offline. Degrades gracefully when
    OpenCV is missing or the source can't be opened by emitting a
    single blur_fill plan covering the clip duration so the bench
    still gets *some* trajectory to compare against.
    """
    config = config or get_default_config()
    try:
        width, height, fps, duration = _probe_video(source_path)
    except Exception as e:
        logger.warning("bench probe failed (%s); returning minimal plan", e)
        return _minimal_plan_dict(1920, 1080, target_height_px, target_aspect)

    if duration <= 0:
        return _minimal_plan_dict(width, height, target_height_px, target_aspect)

    dense_faces, shot_boundaries = _bench_extract_signals(
        source_path, duration, sample_fps=sample_fps,
    )

    inputs = HumanReframeInputs(
        duration_sec=duration,
        source_w=width,
        source_h=height,
        content_type=content_type or "",
        dense_faces=dense_faces,
        active_speaker_events=[],
        shot_boundaries=shot_boundaries,
    )
    try:
        human = run_human_reframe(inputs, config=config)
        rp = render_plan_from_human_plan(
            human,
            source_width=width,
            source_height=height,
            source_fps=fps,
            target_aspect=target_aspect,
            target_height_px=target_height_px,
            config=config,
            content_type=content_type or "",
        )
    except Exception as e:
        logger.warning("bench run failed (%s); emitting minimal plan", e)
        return _minimal_plan_dict(width, height, target_height_px, target_aspect)

    return rp.to_dict()


# ── Bench helpers ────────────────────────────────────────────────


def _probe_video(path: str) -> tuple[int, int, float, float]:
    """Return ``(width, height, fps, duration_sec)`` for a file."""
    try:
        import cv2
    except Exception:
        return (1920, 1080, 30.0, 0.0)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return (1920, 1080, 30.0, 0.0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    duration = frames / fps if fps > 0 else 0.0
    cap.release()
    return (width, height, fps, duration)


def _bench_extract_signals(
    path: str, duration: float, *, sample_fps: float,
) -> tuple[list, list[float]]:
    """Extract a sparse dense-face stream + shot boundaries from the
    source using OpenCV Haar cascade + histogram cut detection.

    Cheap, deterministic, and doesn't depend on the full pipeline's
    YuNet / MediaPipe / PySceneDetect stack — this is only used by the
    offline bench, where the goal is "something reasonable" not
    production-grade.
    """
    try:
        import cv2
    except Exception:
        return [], []

    from dataclasses import dataclass

    @dataclass
    class _Face:
        nose_x: float
        nose_y: float
        width: float
        height: float
        identity_id: int = 0
        is_human: bool = True
        lip_aperture: float = 0.0
        yaw: float = 0.0

    @dataclass
    class _FrameFaces:
        timestamp: float
        faces: list

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps / sample_fps)))
    haar = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    clf = cv2.CascadeClassifier(haar)

    dense: list = []
    cuts: list[float] = []
    prev_hist = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        if idx % stride == 0:
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = []
            try:
                boxes = clf.detectMultiScale(gray, 1.2, 4) if not clf.empty() else []
                for (bx, by, bw, bh) in boxes:
                    faces.append(_Face(
                        nose_x=(bx + bw * 0.5) / w * 100.0,
                        nose_y=(by + bh * 0.5) / h * 100.0,
                        width=bw / w * 100.0,
                        height=bh / h * 100.0,
                    ))
            except Exception:
                faces = []
            dense.append(_FrameFaces(timestamp=t, faces=faces))
            # shot-cut detection via histogram correlation
            small = cv2.resize(frame, (64, 36))
            hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8],
                                [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            if prev_hist is not None:
                corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if corr < 0.6:
                    cuts.append(t)
            prev_hist = hist
        idx += 1
    cap.release()
    return dense, cuts


def _minimal_plan_dict(width: int, height: int, target_h: int,
                      target_aspect: str) -> dict:
    aspect = {"9:16": 9 / 16, "1:1": 1.0, "16:9": 16 / 9}.get(target_aspect, 9 / 16)
    target_w = int(round(target_h * aspect))
    target_w -= target_w % 2
    return {
        "source_width": width,
        "source_height": height,
        "target_width": target_w,
        "target_height": target_h,
        "total_duration_sec": 0.0,
        "fps": 30.0,
        "ops": [{
            "kind": "blur_fill",
            "start_sec": 0.0,
            "end_sec": 0.0,
            "primary_rect": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
            "motion_path": [],
            "ease_in_ms": 0,
            "strategy_label": "bench-minimal",
            "content_type": "unknown",
            "gaming_layout_mode": None,
            "speaker_slot": None,
            "speaker_label": None,
            "secondary_rect": None,
            "tertiary_rect": None,
            "quaternary_rect": None,
        }],
        "source_offset_sec": 0.0,
    }
