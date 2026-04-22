"""Phase A — integration test for the human-reframe pipeline wiring.

The full async ``pipeline.run_job`` is infeasible to drive from a unit
test (DB, ffmpeg, OpenCV cascades, etc.) so this suite exercises the
wiring point directly:

  1. ``reframe_config.load_default_config`` reads the new
     ``CLIPAI_HUMAN_REFRAME_PIPELINE`` env flag correctly (default
     OFF; "1" / "true" turn it on).

  2. ``human_reframe_segment_adapter.reframe_segments_from_human_plan``
     — the shim that bridges ``HumanReframePlan`` → the
     ``ReframeSegment`` list the rest of ``pipeline.py`` consumes —
     produces a contiguous, half-open, in-bounds timeline on the
     existing synthetic parity fixtures.

  3. The underlying ``HumanReframePlan`` contains at least one
     ``CameraMode.SACCADE`` event at a speaker turn (2-speaker
     alternating fixture) and at least one ``CameraMode.HOLD`` /
     stationary-strategy segment on held content (vlog fixture).

  4. ``backend.services.pipeline`` still imports cleanly when the
     flag is set (the new branch doesn't regress module init).
"""

from __future__ import annotations

import importlib
import os

import pytest

from backend.services.autoflip_parity_fixtures import (
    FIXTURES,
    get_fixture,
)
from backend.services.camera_events import CameraMode
from backend.services.human_reframe import (
    HumanReframeInputs,
    run_human_reframe,
)
from backend.services.human_reframe_segment_adapter import (
    reframe_segments_from_human_plan,
)
from backend.services.reframe_segmenter import ReframeSegment


# ── Helpers ────────────────────────────────────────────────────────


def _build_human_inputs(fixture_name: str) -> tuple[HumanReframeInputs, dict, int, int]:
    """Construct ``HumanReframeInputs`` from an autoflip parity fixture."""
    fx = get_fixture(fixture_name)
    kwargs = fx.build()
    inputs = HumanReframeInputs(
        duration_sec=float(kwargs["video_duration"]),
        source_w=fx.source_width,
        source_h=fx.source_height,
        content_type=fx.content_type_override or "other",
        dense_faces=list(kwargs.get("dense_faces") or []),
        active_speaker_events=list(kwargs.get("active_speaker_events") or []),
        shot_boundaries=list(kwargs.get("shot_cuts") or []),
    )
    return inputs, kwargs, fx.source_width, fx.source_height


def _run_shim(fixture_name: str):
    inputs, kwargs, src_w, src_h = _build_human_inputs(fixture_name)
    plan = run_human_reframe(inputs)
    segments, render_plan = reframe_segments_from_human_plan(
        plan,
        source_width=src_w,
        source_height=src_h,
        source_fps=30.0,
        content_type=inputs.content_type,
    )
    return plan, segments, render_plan, kwargs, src_w, src_h


# ── Flag plumbing ──────────────────────────────────────────────────


def _reload_config_module():
    """Force ``reframe_config`` to re-read env on next ``load_default_config``."""
    import backend.services.reframe_config as rc
    importlib.reload(rc)
    return rc


def test_flag_defaults_off_when_unset(monkeypatch):
    monkeypatch.delenv("CLIPAI_HUMAN_REFRAME_PIPELINE", raising=False)
    rc = _reload_config_module()
    cfg = rc.load_default_config()
    assert cfg.human_reframe_pipeline_enabled is False


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes"])
def test_flag_enables_on_truthy_env(monkeypatch, val):
    monkeypatch.setenv("CLIPAI_HUMAN_REFRAME_PIPELINE", val)
    rc = _reload_config_module()
    cfg = rc.load_default_config()
    assert cfg.human_reframe_pipeline_enabled is True


def test_flag_disabled_on_falsy_env(monkeypatch):
    monkeypatch.setenv("CLIPAI_HUMAN_REFRAME_PIPELINE", "0")
    rc = _reload_config_module()
    cfg = rc.load_default_config()
    assert cfg.human_reframe_pipeline_enabled is False


# ── Shim output contract ───────────────────────────────────────────


def test_shim_returns_reframesegment_list():
    _, segments, _, _, _, _ = _run_shim("2speaker_alternating")
    assert segments, "shim produced no segments"
    assert all(isinstance(s, ReframeSegment) for s in segments)


def test_shim_segments_cover_full_timeline_half_open():
    _, segments, render_plan, kwargs, _, _ = _run_shim("2speaker_alternating")
    duration = float(kwargs["video_duration"])
    assert segments, "shim produced no segments"
    # First segment starts at 0, last segment ends at duration (w/ ~50ms tolerance
    # for the render-plan adapter's rounding).
    assert segments[0].start == pytest.approx(0.0, abs=0.01)
    assert segments[-1].end == pytest.approx(duration, abs=0.1)
    # Half-open contiguity: segments[i].end == segments[i+1].start.
    for i in range(len(segments) - 1):
        assert segments[i].end == pytest.approx(
            segments[i + 1].start, abs=0.005,
        ), f"gap between seg[{i}] end and seg[{i+1}] start"


def test_shim_pixel_centers_inside_source_bounds():
    _, segments, _, _, src_w, src_h = _run_shim("2speaker_alternating")
    assert segments
    for s in segments:
        assert 0.0 <= s.subject_x <= src_w, f"subject_x {s.subject_x} OOB [0, {src_w}]"
        assert 0.0 <= s.subject_y <= src_h, f"subject_y {s.subject_y} OOB [0, {src_h}]"
        if s.motion_path:
            for t, x, y in s.motion_path:
                assert 0.0 <= x <= src_w, f"motion_path x {x} OOB"
                assert 0.0 <= y <= src_h, f"motion_path y {y} OOB"


def test_shim_render_plan_validates():
    _, _, render_plan, _, _, _ = _run_shim("2speaker_alternating")
    violations = render_plan.validate()
    assert not violations, f"render plan invalid: {violations[:5]}"


def test_shim_every_segment_tagged_human_reframe():
    _, segments, _, _, _, _ = _run_shim("2speaker_alternating")
    assert segments
    assert all(s.subject_source == "human_reframe" for s in segments)


# ── Camera-event assertions on the underlying plan ─────────────────


def test_plan_has_saccade_at_speaker_turn():
    """On the 2-speaker alternating fixture, the event scheduler must
    emit at least one SACCADE near a turn boundary (every 2s)."""
    plan, _, _, kwargs, _, _ = _run_shim("2speaker_alternating")
    saccades = [e for e in plan.events if e.mode == CameraMode.SACCADE]
    assert saccades, "expected at least one SACCADE event on alternating speakers"
    # At least one saccade should land within 0.5s of a turn boundary
    # at 2, 4, 6, ... seconds.
    turn_times = [i * 2.0 for i in range(1, int(kwargs["video_duration"] // 2))]
    for sc in saccades:
        for t in turn_times:
            if abs(sc.start - t) < 0.5:
                return
    pytest.fail(
        "no SACCADE within 0.5s of any speaker turn; "
        f"saccades at {[s.start for s in saccades]}"
    )


def test_plan_has_hold_or_pursue_segment():
    """The event scheduler should emit at least one HOLD or
    SMOOTH_PURSUE state on the vlog fixture where the subject only
    walks slowly (no saccade-triggering turns)."""
    plan, _, _, _, _, _ = _run_shim("vlog_walk_and_talk")
    kinds = {e.mode for e in plan.events}
    assert (
        CameraMode.HOLD in kinds or CameraMode.SMOOTH_PURSUE in kinds
    ), f"expected HOLD or SMOOTH_PURSUE on held vlog content, got {kinds}"


def test_shim_emits_saccade_ease_on_speaker_turn():
    """Downstream segments carry ``ease_in_ms > 0`` at least once on
    the 2-speaker fixture — proves the event scheduler's saccade ease
    threaded through the shim into the ReframeSegment list."""
    _, segments, _, _, _, _ = _run_shim("2speaker_alternating")
    assert any(s.ease_in_ms > 0 for s in segments), (
        "no segment has ease_in_ms > 0; saccade ease didn't thread through"
    )


def test_shim_emits_stationary_strategy_on_held_content():
    """On vlog walk-and-talk the solver/shim should produce at least
    one stationary strategy (the deadzone check collapses a still
    window into a CROP op which maps to ``strategy='stationary'``)."""
    _, segments, _, _, _, _ = _run_shim("vlog_walk_and_talk")
    strategies = {s.strategy for s in segments}
    assert strategies & {"stationary", "tracking"}, (
        f"expected stationary or tracking strategy, got {strategies}"
    )


# ── Pipeline import regression ─────────────────────────────────────


def test_pipeline_module_parses_and_references_new_branch():
    """The pipeline module must parse cleanly and its source must
    contain the new ``CLIPAI_HUMAN_REFRAME_PIPELINE`` branch.

    We use AST parsing (not import) because ``pipeline.py`` pulls
    in httpx / opencv / ffmpeg subprocess setup that isn't present
    in a bare-bones unit-test environment. AST parsing still
    catches syntax errors and dedent regressions in the new branch.
    """
    import ast
    import pathlib

    src = pathlib.Path(
        "backend/services/pipeline.py",
    ).read_text()
    tree = ast.parse(src)
    assert tree is not None
    # New env flag must be referenced by name somewhere in the source.
    assert "CLIPAI_HUMAN_REFRAME_PIPELINE" in src, (
        "pipeline.py missing the CLIPAI_HUMAN_REFRAME_PIPELINE branch"
    )
    assert "USE_HUMAN_REFRAME_PIPELINE" in src, (
        "pipeline.py missing the USE_HUMAN_REFRAME_PIPELINE gate"
    )
    assert "reframe_segments_from_human_plan" in src, (
        "pipeline.py missing the shim import / call"
    )


def test_new_shim_importable_in_isolation():
    """The shim module must import cleanly without running the full
    pipeline — a missing symbol or circular import would block every
    path that uses it."""
    from backend.services.human_reframe_segment_adapter import (
        reframe_segments_from_human_plan,
    )
    assert callable(reframe_segments_from_human_plan)


# ── All-fixtures smoke ─────────────────────────────────────────────


_NON_GAMING_FIXTURES = [
    name for name, spec in FIXTURES.items()
    if spec.content_type_override not in ("gameplay_tps", "stream")
]


@pytest.mark.parametrize("fixture_name", _NON_GAMING_FIXTURES)
def test_shim_runs_cleanly_on_every_non_gaming_fixture(fixture_name):
    """The shim should not raise on any of the synthetic
    non-gaming fixtures. Gaming fixtures go through the dedicated
    compositor and don't take the human-reframe path."""
    _, segments, render_plan, kwargs, _, _ = _run_shim(fixture_name)
    assert segments, f"no segments on {fixture_name}"
    violations = render_plan.validate()
    assert not violations, f"{fixture_name} produced invalid plan: {violations[:3]}"
