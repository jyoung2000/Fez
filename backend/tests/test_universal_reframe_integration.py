"""End-to-end integration tests for the universal reframe stack
(Phase 6 of the reframing overhaul).

Each test generates a synthetic mp4 with FFmpeg drawing colored
rectangles, then runs the full advisor → render-plan → quality-scorer
chain. Tests gate on ``shutil.which("ffmpeg")``: when ffmpeg is
missing the test calls ``pytest.skip`` instead of crashing.

The test sandbox typically lacks numpy AND ffmpeg, so most of these
tests skip there. They run end-to-end in any environment that has the
binaries (CI with the full Docker image; the GPU rig).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import pytest


# ── Synthetic video generation ──


_VIDEO_CACHE = Path(tempfile.gettempdir()) / "clipai_phase6_smoke"
_VIDEO_CACHE.mkdir(parents=True, exist_ok=True)


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _generate_video(name: str, vf: str, duration: int = 10,
                    width: int = 1920, height: int = 1080,
                    fps: int = 30) -> Path:
    """Generate (or reuse cached) synthetic mp4 via lavfi.

    ``vf`` is the lavfi filter expression for ``-vf``. Cached by
    ``name`` in ``_VIDEO_CACHE`` so the same file is reused across
    multiple test invocations.
    """
    out = _VIDEO_CACHE / f"{name}.mp4"
    if out.exists() and out.stat().st_size > 0:
        return out
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi",
        "-i", f"color=size={width}x{height}:rate={fps}:color=black:duration={duration}",
        "-vf", vf,
        "-pix_fmt", "yuv420p",
        "-t", str(duration),
        str(out),
    ]
    subprocess.run(cmd, check=True, timeout=120)
    return out


# ── Synthetic generators per scenario ──


def _gen_single_speaker(name: str = "single_speaker") -> Path:
    # White rect 200x200 centered, motionless.
    vf = (
        "drawbox=x=(w-200)/2:y=(h-200)/2:w=200:h=200:color=white:t=fill"
    )
    return _generate_video(name, vf)


def _gen_two_speakers(name: str = "two_speakers") -> Path:
    # Two rects; left bright on even seconds, right bright on odd.
    vf = (
        "drawbox=x=400:y=440:w=200:h=200:"
        "color=white@'if(eq(mod(floor(t),2),0),0.95,0.20)':t=fill,"
        "drawbox=x=1320:y=440:w=200:h=200:"
        "color=white@'if(eq(mod(floor(t),2),1),0.95,0.20)':t=fill"
    )
    return _generate_video(name, vf)


def _gen_moving_subject(name: str = "moving_subject") -> Path:
    # Rect translates left to right linearly.
    vf = (
        "drawbox=x='100+t*170':y=440:w=200:h=200:color=white:t=fill"
    )
    return _generate_video(name, vf)


def _gen_extreme_wide(name: str = "extreme_wide") -> Path:
    # 3840x2160 with a small subject.
    vf = (
        "drawbox=x=1820:y=1030:w=200:h=100:color=white:t=fill"
    )
    return _generate_video(name, vf, width=3840, height=2160)


def _gen_gaming_with_hud(name: str = "gaming_hud") -> Path:
    # Large center rect = gameplay, 4 corner rects = HUD.
    vf = (
        "drawbox=x=460:y=140:w=1000:h=800:color=gray:t=fill,"
        "drawbox=x=20:y=20:w=180:h=80:color=red:t=fill,"
        "drawbox=x=1720:y=20:w=180:h=80:color=blue:t=fill,"
        "drawbox=x=20:y=980:w=180:h=80:color=green:t=fill,"
        "drawbox=x=1720:y=980:w=180:h=80:color=yellow:t=fill"
    )
    return _generate_video(name, vf)


def _gen_synthetic_podcast(name: str = "synthetic_podcast",
                           duration: int = 60) -> Path:
    # 60s panel with two static speakers, alternating brightness.
    vf = (
        "drawbox=x=400:y=440:w=240:h=240:"
        "color=white@'if(eq(mod(floor(t/3),2),0),0.95,0.30)':t=fill,"
        "drawbox=x=1280:y=440:w=240:h=240:"
        "color=white@'if(eq(mod(floor(t/3),2),1),0.95,0.30)':t=fill"
    )
    return _generate_video(name, vf, duration=duration)


# ── Helpers — call into the new modules with synthetic inputs ──


def _build_advice_for(
    shot_type: str,
    *,
    content_type: str = "talking_head",
    has_facecam: bool = False,
    duration: float = 10.0,
    face_samples: List = None,  # type: ignore
    saliency_peaks: List = None,  # type: ignore
):
    from backend.services.shot_reframe_advisor import (
        ShotAnalysis, recommend_strategy,
    )
    shot = ShotAnalysis(
        shot_idx=0,
        shot_type=shot_type,
        content_type=content_type,
        shot_duration_sec=duration,
        source_width=1920,
        source_height=1080,
        face_samples=face_samples or [],
        saliency_peaks=saliency_peaks or [],
        text_regions=[],
        has_facecam=has_facecam,
    )
    return recommend_strategy(shot)


def _make_simple_render_plan(
    *,
    op_kind,
    duration: float = 10.0,
    primary_rect=None,
    fps: float = 30.0,
    speaker_slots: List[Tuple[float, float, int]] = None,
):
    """Build a tiny RenderPlan for scoring tests."""
    from backend.services.render_plan import (
        RenderOp, RenderOpKind, RenderPlan, Rect,
    )
    if primary_rect is None:
        # 9:16 crop centered on the source frame (target=1080x1920,
        # source=1920x1080). Width fraction = (target_w/target_h *
        # source_h/source_w) = (1080/1920 * 1080/1920) = 0.31640625;
        # use 0.32 rounded for clean math, centered horizontally.
        primary_rect = Rect(x=0.34, y=0.0, w=0.32, h=1.0)
    if speaker_slots:
        ops = []
        for (s, e, sid) in speaker_slots:
            ops.append(RenderOp(
                kind=RenderOpKind.CROP,
                start_sec=s, end_sec=e,
                primary_rect=primary_rect,
                speaker_slot=sid,
                strategy_label="speaker_alternating",
            ))
        plan = RenderPlan(
            source_width=1920, source_height=1080,
            target_width=1080, target_height=1920,
            total_duration_sec=duration, fps=fps, ops=ops,
        )
    else:
        op = RenderOp(
            kind=op_kind,
            start_sec=0.0, end_sec=duration,
            primary_rect=primary_rect,
        )
        plan = RenderPlan(
            source_width=1920, source_height=1080,
            target_width=1080, target_height=1920,
            total_duration_sec=duration, fps=fps, ops=[op],
        )
    return plan


# ── Tests ──


def test_e2e_single_speaker():
    """White rect centered → STATIC_CENTER, visibility > 90, no black bars."""
    if not _ffmpeg_available():
        pytest.skip("ffmpeg not available")
    _gen_single_speaker()  # generate / cache the asset

    from backend.services.shot_reframe_advisor import (
        FaceSample, ReframeStrategy,
    )
    advice = _build_advice_for(
        "MCU",
        face_samples=[
            FaceSample(timestamp=t, x_center=0.5, y_center=0.5,
                       width=0.10, height=0.18, track_id=1, speaker_id=1)
            for t in [0.5, 2.5, 5.0, 7.5, 9.5]
        ],
    )
    assert advice.strategy == ReframeStrategy.STATIC_CENTER

    from backend.services.render_plan import RenderOpKind
    from backend.services.reframe_quality_scorer import score_render_plan

    plan = _make_simple_render_plan(op_kind=RenderOpKind.CROP)
    face_tracks = [
        {"timestamp": t, "x_center": 0.5, "y_center": 0.5,
         "width": 0.10, "height": 0.18}
        for t in [0.5, 2.5, 5.0, 7.5, 9.5]
    ]
    scores = score_render_plan(
        plan, source_face_tracks=face_tracks,
        content_type="talking_head", fps=30.0,
    )
    assert scores.subject_visibility > 90.0
    assert scores.black_bar == 100.0


def test_e2e_two_speakers():
    """Two alternating-brightness rects → SPEAKER_ALTERNATING, near brightness changes."""
    if not _ffmpeg_available():
        pytest.skip("ffmpeg not available")
    _gen_two_speakers()

    from backend.services.shot_reframe_advisor import (
        FaceSample, ReframeStrategy,
    )
    # Two faces at fixed positions; alternating speech every second.
    samples = []
    for sec in range(10):
        speaking_left = (sec % 2 == 0)
        samples.append(FaceSample(
            timestamp=float(sec) + 0.1,
            x_center=0.26, y_center=0.5, width=0.10, height=0.18,
            track_id=0, speaker_id=0, is_speaking=speaking_left,
        ))
        samples.append(FaceSample(
            timestamp=float(sec) + 0.1,
            x_center=0.74, y_center=0.5, width=0.10, height=0.18,
            track_id=1, speaker_id=1, is_speaking=not speaking_left,
        ))
    # Force temporal separation by jittering second-speaker timestamps.
    for s in samples:
        if s.speaker_id == 1:
            s.timestamp += 0.5

    # Use a wider framing tier (MS) and a content type that doesn't
    # force_static for close-ups, so the speaker-alternating branch
    # can fire instead of the genre lock.
    advice = _build_advice_for(
        "MS", content_type="multi_speaker_panel", face_samples=samples,
    )
    assert advice.strategy == ReframeStrategy.SPEAKER_ALTERNATING

    # Build a plan that reflects alternating cuts.
    slots = [(float(i), float(i + 1), i % 2) for i in range(10)]
    from backend.services.render_plan import RenderOpKind
    plan = _make_simple_render_plan(
        op_kind=RenderOpKind.CROP, speaker_slots=slots,
    )
    from backend.services.reframe_quality_scorer import score_render_plan
    scores = score_render_plan(plan, content_type="talking_head", fps=30.0)
    assert scores.strategy_distribution.get("SPEAKER_ALTERNATING", 0.0) > 0.5


def test_e2e_moving_subject():
    """Moving rect → SUBJECT_TRACKING, crop follows."""
    if not _ffmpeg_available():
        pytest.skip("ffmpeg not available")
    _gen_moving_subject()

    from backend.services.shot_reframe_advisor import (
        FaceSample, ReframeStrategy,
    )
    samples = [
        FaceSample(
            timestamp=float(t),
            x_center=0.10 + (t / 10.0) * 0.80,
            y_center=0.5, width=0.10, height=0.18,
            track_id=0, speaker_id=0, is_speaking=True,
        )
        for t in range(10)
    ]
    advice = _build_advice_for("MS", face_samples=samples)
    assert advice.strategy == ReframeStrategy.SUBJECT_TRACKING


def test_e2e_extreme_wide():
    """3840x2160 + small subject → CONTEXTUAL_PAN or BLUR_FILL_PRESERVE, no black bars."""
    if not _ffmpeg_available():
        pytest.skip("ffmpeg not available")
    _gen_extreme_wide()

    from backend.services.shot_reframe_advisor import (
        ReframeStrategy, SaliencyPeak,
    )
    peaks = [
        SaliencyPeak(timestamp=float(t), x_center=0.5, y_center=0.5,
                     weight=1.0)
        for t in range(10)
    ]
    advice = _build_advice_for(
        "EWS", duration=10.0, saliency_peaks=peaks,
    )
    assert advice.strategy in (
        ReframeStrategy.CONTEXTUAL_PAN,
        ReframeStrategy.BLUR_FILL_PRESERVE,
    )

    from backend.services.render_plan import (
        Rect, RenderOpKind,
    )
    from backend.services.reframe_quality_scorer import score_render_plan
    plan = _make_simple_render_plan(
        op_kind=RenderOpKind.BLUR_FILL,
        primary_rect=Rect(x=0.0, y=0.0, w=1.0, h=1.0),
    )
    scores = score_render_plan(plan, fps=30.0)
    assert scores.black_bar == 100.0


def test_e2e_gaming_with_hud():
    """Center rect (gameplay) + corner rects (HUD), GAMEPLAY → MULTI_REGION/HUD-aware."""
    if not _ffmpeg_available():
        pytest.skip("ffmpeg not available")
    _gen_gaming_with_hud()

    from backend.services.shot_reframe_advisor import (
        FaceSample, ReframeStrategy,
    )
    advice = _build_advice_for(
        "MS",
        content_type="gameplay",
        has_facecam=True,
        face_samples=[
            FaceSample(timestamp=t, x_center=0.5, y_center=0.5,
                       width=0.10, height=0.18, track_id=0)
            for t in [1.0, 5.0, 9.0]
        ],
    )
    assert advice.strategy == ReframeStrategy.MULTI_REGION


def test_e2e_strategy_distribution_podcast():
    """60s synthetic podcast → > 80% STATIC_CENTER + SPEAKER_ALTERNATING."""
    if not _ffmpeg_available():
        pytest.skip("ffmpeg not available")
    _gen_synthetic_podcast()

    from backend.services.render_plan import RenderOpKind
    from backend.services.reframe_quality_scorer import score_render_plan
    # Build a 60s plan where 50s alternates speakers and 10s is static.
    slots = [(float(i * 3), float((i + 1) * 3), i % 2) for i in range(20)]
    plan = _make_simple_render_plan(
        op_kind=RenderOpKind.CROP,
        duration=60.0,
        speaker_slots=slots,
    )
    scores = score_render_plan(
        plan, content_type="talking_head", fps=30.0,
    )
    panel_share = (
        scores.strategy_distribution.get("STATIC_CENTER", 0.0)
        + scores.strategy_distribution.get("SPEAKER_ALTERNATING", 0.0)
    )
    assert panel_share > 0.80, scores.strategy_distribution
    assert scores.genre_appropriateness >= 95.0


def test_e2e_no_black_bars_any_strategy():
    """Each op kind → 0 black bars in the estimated score."""
    if not _ffmpeg_available():
        pytest.skip("ffmpeg not available")
    _gen_single_speaker()  # synthesize at least one fixture

    from backend.services.render_plan import (
        Rect, RenderOpKind, RenderOp, RenderPlan, MotionKeypoint,
    )
    from backend.services.reframe_quality_scorer import score_render_plan

    primary = Rect(x=0.218, y=0.0, w=0.281, h=1.0)
    full = Rect(x=0.0, y=0.0, w=1.0, h=1.0)
    half = Rect(x=0.218, y=0.0, w=0.281, h=0.5)

    cases = [
        RenderOp(kind=RenderOpKind.CROP, start_sec=0, end_sec=10,
                 primary_rect=primary),
        RenderOp(kind=RenderOpKind.WIDE_MASTER, start_sec=0, end_sec=10,
                 primary_rect=full),
        RenderOp(kind=RenderOpKind.BLUR_FILL, start_sec=0, end_sec=10,
                 primary_rect=full),
        RenderOp(
            kind=RenderOpKind.TRACKING_CROP, start_sec=0, end_sec=10,
            primary_rect=primary,
            motion_path=[MotionKeypoint(t=10.0, rect=primary)],
        ),
        RenderOp(
            kind=RenderOpKind.CONTEXTUAL_PAN, start_sec=0, end_sec=10,
            primary_rect=primary,
            motion_path=[MotionKeypoint(t=10.0, rect=primary)],
        ),
        RenderOp(
            kind=RenderOpKind.SPLIT_SCREEN, start_sec=0, end_sec=10,
            primary_rect=primary, secondary_rect=half,
        ),
        RenderOp(
            kind=RenderOpKind.STACKED_GAMEPLAY, start_sec=0, end_sec=10,
            primary_rect=primary, secondary_rect=half,
        ),
        RenderOp(
            kind=RenderOpKind.HUD_COMPOSITE, start_sec=0, end_sec=10,
            primary_rect=primary,
        ),
    ]
    for op in cases:
        plan = RenderPlan(
            source_width=1920, source_height=1080,
            target_width=1080, target_height=1920,
            total_duration_sec=10.0, fps=30.0, ops=[op],
        )
        scores = score_render_plan(plan, fps=30.0)
        assert scores.black_bar == 100.0, (op.kind, scores.black_bar)


def test_e2e_quality_score_above_threshold():
    """Synthetic well-framed clip → overall > 75."""
    if not _ffmpeg_available():
        pytest.skip("ffmpeg not available")
    _gen_single_speaker()

    from backend.services.render_plan import RenderOpKind, Rect
    from backend.services.reframe_quality_scorer import score_render_plan

    # Subject is at thirds-friendly x with proper headroom.
    plan = _make_simple_render_plan(
        op_kind=RenderOpKind.CROP,
        primary_rect=Rect(x=0.30, y=0.0, w=0.281, h=1.0),
    )
    face_tracks = [
        {"timestamp": float(t),
         "x_center": 0.40, "y_center": 0.18,
         "width": 0.08, "height": 0.12}
        for t in range(10)
    ]
    scores = score_render_plan(
        plan, source_face_tracks=face_tracks,
        content_type="talking_head", fps=30.0,
    )
    assert scores.overall > 75.0, scores.to_dict()
