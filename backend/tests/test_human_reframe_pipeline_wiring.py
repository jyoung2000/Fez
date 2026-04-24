"""Smoke test for the Phase A human-reframe pipeline wiring.

These tests target ``backend.services.human_reframe_pipeline`` directly
because the full ``backend.services.pipeline.process_video`` async
entry point is a 5000-line function with dozens of upstream
dependencies (face detection, AVD, transcript, scene detection, …)
that aren't trivially mockable. The wiring it owns is exactly the
flag/allowlist gating, the RenderPlan→ReframeSegment shim, and the
single-call entry point — all of which live in
``human_reframe_pipeline.py``. Verifying them here is sufficient to
prove the pipeline's branch-pick is correct.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from backend.services.human_reframe_pipeline import (
    HUMAN_REFRAME_ALLOWED_CONTENT_TYPES,
    is_compare_mode_enabled,
    is_pipeline_flag_enabled,
    pick_reframe_path,
    render_plan_to_reframe_segments,
    try_run_human_reframe_pipeline,
)


# ── Flag gating ────────────────────────────────────────────────


def test_flag_off_default():
    """The pipeline flag is OFF by default: legacy path stays primary."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CLIPAI_HUMAN_REFRAME_PIPELINE", None)
        assert is_pipeline_flag_enabled() is False
        assert pick_reframe_path("multi_speaker_panel") == "reframe_segmenter"


def test_flag_on_panel_picks_human():
    """Flag on + allowlisted content type → human_reframe."""
    with patch.dict(os.environ, {"CLIPAI_HUMAN_REFRAME_PIPELINE": "true"}, clear=False):
        os.environ.pop("USE_AUTOFLIP_REFRAME", None)
        assert pick_reframe_path("multi_speaker_panel") == "human_reframe"
        assert pick_reframe_path("talking_head") == "human_reframe"


def test_flag_on_anime_falls_through_to_legacy():
    """Flag on but content type not in allowlist → falls through."""
    with patch.dict(os.environ, {"CLIPAI_HUMAN_REFRAME_PIPELINE": "true"}, clear=False):
        os.environ.pop("USE_AUTOFLIP_REFRAME", None)
        # animation + animation_dialogue are NOT allowlisted (each
        # needs its own bench-validation evidence before joining).
        assert pick_reframe_path("animation") == "reframe_segmenter"
        assert pick_reframe_path("animation_dialogue") == "reframe_segmenter"
        assert pick_reframe_path("") == "reframe_segmenter"


def test_flag_on_autoflip_priority_when_unallowlisted():
    """USE_AUTOFLIP_REFRAME wins over the legacy segmenter when the
    human flag is on but content type isn't allowlisted."""
    with patch.dict(os.environ, {
        "CLIPAI_HUMAN_REFRAME_PIPELINE": "true",
        "USE_AUTOFLIP_REFRAME": "1",
    }, clear=False):
        assert pick_reframe_path("animation") == "autoflip"


def test_compare_mode_flag():
    """The compare mode is independently togglable."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CLIPAI_REFRAME_COMPARE", None)
        assert is_compare_mode_enabled() is False
    with patch.dict(os.environ, {"CLIPAI_REFRAME_COMPARE": "1"}, clear=False):
        assert is_compare_mode_enabled() is True


def test_allowlist_membership_is_explicit():
    """The allowlist is a frozenset and only contains the bench-cleared
    content types. Adding to it requires bench evidence, not a config
    nudge — guard against accidental membership growth."""
    assert HUMAN_REFRAME_ALLOWED_CONTENT_TYPES == frozenset({
        "multi_speaker_panel",
        "talking_head",
    })


# ── try_run_human_reframe_pipeline gating ─────────────────────────


def test_try_run_returns_none_when_flag_off():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CLIPAI_HUMAN_REFRAME_PIPELINE", None)
        result = try_run_human_reframe_pipeline(
            duration_sec=10.0,
            source_width=1920, source_height=1080, source_fps=30.0,
            content_type="multi_speaker_panel",
            dense_faces=[], active_speaker_events=[],
            shot_boundaries=[], job_id="test",
        )
        assert result is None


def test_try_run_returns_none_when_content_type_not_allowlisted():
    with patch.dict(os.environ, {"CLIPAI_HUMAN_REFRAME_PIPELINE": "true"}, clear=False):
        result = try_run_human_reframe_pipeline(
            duration_sec=10.0,
            source_width=1920, source_height=1080, source_fps=30.0,
            content_type="animation",
            dense_faces=[("placeholder",)],
            active_speaker_events=[],
            shot_boundaries=[], job_id="test",
        )
        assert result is None


def test_try_run_returns_none_when_no_dense_faces():
    """Even with the flag on and an allowlisted content type, no dense
    faces means we have no input — fall back to legacy."""
    with patch.dict(os.environ, {"CLIPAI_HUMAN_REFRAME_PIPELINE": "true"}, clear=False):
        result = try_run_human_reframe_pipeline(
            duration_sec=10.0,
            source_width=1920, source_height=1080, source_fps=30.0,
            content_type="multi_speaker_panel",
            dense_faces=[],
            active_speaker_events=[],
            shot_boundaries=[], job_id="test",
        )
        assert result is None


# ── End-to-end wiring on a synthetic fixture ─────────────────────


def _talking_head_dense(duration: float = 6.0, hz: float = 10.0) -> list:
    """Mirror of the bench fixture so the pipeline runs without ffmpeg."""
    from backend.scripts.measure_human_reframe_quality import (
        _fx_talking_head,
    )
    return _fx_talking_head(duration=duration, hz=hz)["dense_faces"]


def test_try_run_talking_head_returns_segments_and_plan():
    """End-to-end: flag on + allowlisted CT + valid inputs → returns
    a non-empty (segments, render_plan) tuple."""
    with patch.dict(os.environ, {"CLIPAI_HUMAN_REFRAME_PIPELINE": "true"}, clear=False):
        result = try_run_human_reframe_pipeline(
            duration_sec=6.0,
            source_width=1920, source_height=1080, source_fps=30.0,
            content_type="talking_head",
            dense_faces=_talking_head_dense(duration=6.0),
            active_speaker_events=[],
            shot_boundaries=[],
            job_id="test",
        )
    assert result is not None, "expected the pipeline to run end-to-end"
    segments, rp = result
    assert segments, "expected at least one ReframeSegment"
    assert rp.ops, "expected a non-empty RenderPlan"
    # The shim must produce contiguous segments matching the plan.
    assert len(segments) == len(rp.ops)


def test_render_plan_to_segments_preserves_timing():
    """Per-op timestamps round-trip through the shim unchanged."""
    with patch.dict(os.environ, {"CLIPAI_HUMAN_REFRAME_PIPELINE": "true"}, clear=False):
        result = try_run_human_reframe_pipeline(
            duration_sec=6.0,
            source_width=1920, source_height=1080, source_fps=30.0,
            content_type="talking_head",
            dense_faces=_talking_head_dense(duration=6.0),
            active_speaker_events=[],
            shot_boundaries=[],
            job_id="test",
        )
    assert result is not None
    segments, rp = result
    for seg, op in zip(segments, rp.ops):
        assert seg.start == pytest.approx(op.start_sec)
        assert seg.end == pytest.approx(op.end_sec)
        # subject_x is in pixels — must lie inside the source frame.
        assert 0.0 <= seg.subject_x <= 1920.0
        assert seg.subject_source == "human_reframe"
        assert seg.reason.startswith("human_reframe:")
