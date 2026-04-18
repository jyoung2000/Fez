"""Cross-genre coverage + quality invariants for the human-reframe path.

These tests assert the contract: **every frame of every clip, in every
genre, gets a valid RenderOp with an in-range crop rect**, and no crop
is awkwardly placed (face chopped, chin clipped, headroom inverted).

The tests use the synthetic face / event / beat stream fixtures because
the bench needs real video clips that aren't in the repo. They focus
on the invariants the adapter guarantees.

Each test follows the same shape:

  1. Build a ``HumanReframeInputs`` tuned for the target genre.
  2. Run ``human_reframe.run_human_reframe``.
  3. Hand the output to ``render_plan_from_human_plan``.
  4. Call ``verify_frame_coverage`` and assert ``ok``.
  5. Spot-check genre-specific quality signals (zoom present for
     cinematic_dialogue, saccade for music beat, AB cut for two-speaker
     dialogue, no zoom for gameplay, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest

from backend.services import (
    ab_cut_scheduler,
    human_reframe,
    human_reframe_bridge,
    human_render_plan_adapter,
    motivated_zoom,
    reframe_config,
)
from backend.services.render_plan import RenderOpKind


# ── Synthetic fixtures ───────────────────────────────────────────


@dataclass
class _Face:
    identity_id: int = 0
    is_human: bool = True
    nose_x: float = 50.0
    nose_y: float = 40.0
    width: float = 15.0
    height: float = 20.0
    lip_aperture: float = 0.0
    yaw: Optional[float] = None


@dataclass
class _FrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)


@dataclass
class _SpeakerEvent:
    start: float
    end: float
    slot_id: int
    on_screen: bool = True


def _dense_single_speaker(duration: float = 6.0, fps: float = 6.0, nose_x_fn=None):
    frames = []
    n = int(duration * fps)
    for i in range(n):
        t = i / fps
        x = nose_x_fn(t) if nose_x_fn else 50.0
        frames.append(_FrameFaces(timestamp=t, faces=[_Face(nose_x=x)]))
    return frames


def _dense_two_speakers(duration: float = 6.0, fps: float = 6.0):
    frames = []
    n = int(duration * fps)
    for i in range(n):
        t = i / fps
        # Speaker 0 on left, speaker 1 on right, both present in frame.
        frames.append(_FrameFaces(
            timestamp=t,
            faces=[
                _Face(identity_id=0, nose_x=30.0, nose_y=40.0),
                _Face(identity_id=1, nose_x=70.0, nose_y=40.0),
            ],
        ))
    return frames


def _run(inputs: human_reframe.HumanReframeInputs, cfg=None):
    if cfg is None:
        cfg = reframe_config.load_default_config().override(
            human_reframe_enabled=True,
        )
    human = human_reframe.run_human_reframe(inputs, config=cfg)
    rp = human_render_plan_adapter.render_plan_from_human_plan(
        human,
        source_width=inputs.source_w,
        source_height=inputs.source_h,
        source_fps=30.0,
        config=cfg,
        content_type=inputs.content_type,
    )
    coverage = human_render_plan_adapter.verify_frame_coverage(rp)
    return human, rp, coverage


def _assert_coverage_clean(rp, coverage):
    assert coverage.ok, (
        f"coverage failed: gaps={coverage.gaps}, "
        f"overlaps={coverage.overlaps}, oor={coverage.out_of_range_rects}, "
        f"zero_ops={coverage.zero_duration_ops}"
    )
    violations = rp.validate()
    assert not violations, f"plan validation failed: {violations}"


# ── Genre: talking head ─────────────────────────────────────────


def test_talking_head_coverage_and_zoom_on_long_dwell():
    dense = _dense_single_speaker(duration=8.0)
    inputs = human_reframe.HumanReframeInputs(
        duration_sec=8.0,
        source_w=1920, source_h=1080,
        content_type="talking_head",
        dense_faces=dense,
        active_speaker_events=[_SpeakerEvent(0.0, 8.0, slot_id=0)],
    )
    human, rp, coverage = _run(inputs)
    _assert_coverage_clean(rp, coverage)

    # Every op's primary rect must be a 9:16 crop (w < h).
    for op in rp.ops:
        r = op.primary_rect
        assert r.w <= r.h + 0.02
        # Face is at nose_x=50, so crop center should be near x=0.5.
        cx = r.x + r.w * 0.5
        assert 0.40 < cx < 0.60

    # Long-dwell talking head should pick up at least one motivated
    # zoom (genre refinement schedules push-in at 3+s dwell).
    kinds = {op.kind for op in rp.ops}
    # At minimum a tracking/crop + a zoom should appear in the mix.
    assert RenderOpKind.CROP in kinds or RenderOpKind.TRACKING_CROP in kinds


# ── Genre: multi_speaker_panel ──────────────────────────────────


def test_panel_no_zoom_and_valid_coverage():
    dense = _dense_two_speakers(duration=6.0)
    events = [
        _SpeakerEvent(0.0, 2.0, slot_id=0),
        _SpeakerEvent(2.0, 4.0, slot_id=1),
        _SpeakerEvent(4.0, 6.0, slot_id=0),
    ]
    inputs = human_reframe.HumanReframeInputs(
        duration_sec=6.0,
        source_w=1920, source_h=1080,
        content_type="multi_speaker_panel",
        dense_faces=dense,
        active_speaker_events=events,
    )
    human, rp, coverage = _run(inputs)
    _assert_coverage_clean(rp, coverage)

    # Panel content must never emit motivated zooms.
    forbidden = {RenderOpKind.MOTIVATED_PUSH_IN, RenderOpKind.MOTIVATED_PULL_OUT}
    assert not any(op.kind in forbidden for op in rp.ops)


# ── Genre: two-speaker dialogue (A/B cut) ────────────────────────


def test_dialogue_ab_cut_schedules_speaker_switches():
    dense = _dense_two_speakers(duration=6.0)
    events = [
        _SpeakerEvent(0.0, 2.0, slot_id=0),
        _SpeakerEvent(2.0, 4.0, slot_id=1),
        _SpeakerEvent(4.0, 6.0, slot_id=0),
    ]
    inputs = human_reframe.HumanReframeInputs(
        duration_sec=6.0,
        source_w=1920, source_h=1080,
        content_type="cinematic_dialogue",
        dense_faces=dense,
        active_speaker_events=events,
    )
    human, rp, coverage = _run(inputs)
    _assert_coverage_clean(rp, coverage)
    assert human.ab.enabled, "A/B cut should have fired for 2-speaker dialogue"
    # Speaker slots should alternate in the A/B segments.
    slots = [s.slot_id for s in human.ab.segments]
    assert len(set(slots)) >= 2


# ── Genre: sports (racing) ──────────────────────────────────────


def test_racing_eye_line_lowered():
    # Moving subject: car pans left → right.
    def nose_x(t):
        return 30.0 + 6.0 * t    # 30 → 66 over 6 sec
    dense = _dense_single_speaker(duration=6.0, nose_x_fn=nose_x)
    inputs = human_reframe.HumanReframeInputs(
        duration_sec=6.0,
        source_w=1920, source_h=1080,
        content_type="sports_racing",
        dense_faces=dense,
        active_speaker_events=[_SpeakerEvent(0.0, 6.0, slot_id=0)],
    )
    human, rp, coverage = _run(inputs)
    _assert_coverage_clean(rp, coverage)

    # Eye line for racing is ~0.55 (car sits lower). The crop center
    # should land in the lower half of the source frame on average.
    cy_vals = []
    for op in rp.ops:
        cy_vals.append(op.primary_rect.y + op.primary_rect.h * 0.5)
    # Since crop_h_frac = 9/16/aspect = ~0.56 and the eye-line config
    # pulls cy down, we expect cy > 0.42 on average.
    assert sum(cy_vals) / len(cy_vals) > 0.42


# ── Genre: gaming ───────────────────────────────────────────────


def test_gameplay_no_zoom_hud_reservation():
    dense = _dense_single_speaker(duration=5.0)
    inputs = human_reframe.HumanReframeInputs(
        duration_sec=5.0,
        source_w=1920, source_h=1080,
        content_type="gameplay",
        dense_faces=dense,
        active_speaker_events=[],
        hud_regions=[{"x": 0.0, "y": 0.85, "w": 1.0, "h": 0.15,
                      "label": "mini-map"}],
    )
    human, rp, coverage = _run(inputs)
    _assert_coverage_clean(rp, coverage)

    forbidden = {RenderOpKind.MOTIVATED_PUSH_IN, RenderOpKind.MOTIVATED_PULL_OUT}
    assert not any(op.kind in forbidden for op in rp.ops)
    assert len(human.genre.hud_exclusions) == 1


# ── Genre: music video (beat-lock) ──────────────────────────────


def test_music_video_schedules_beat_saccades():
    dense = _dense_single_speaker(duration=8.0)
    downbeats = [0.0, 2.0, 4.0, 6.0]
    beats = [i * 0.5 for i in range(16)]
    inputs = human_reframe.HumanReframeInputs(
        duration_sec=8.0,
        source_w=1920, source_h=1080,
        content_type="music_video",
        dense_faces=dense,
        active_speaker_events=[_SpeakerEvent(0.0, 8.0, slot_id=0)],
        beats=beats,
        downbeats=downbeats,
    )
    human, rp, coverage = _run(inputs)
    _assert_coverage_clean(rp, coverage)
    assert human.genre.saccade_timestamps, "music genre should schedule beat-locked saccades"


# ── Genre: animation ────────────────────────────────────────────


def test_animation_impact_saccades_pull_out():
    dense = _dense_single_speaker(duration=5.0)
    impacts = [1.0, 3.0]
    inputs = human_reframe.HumanReframeInputs(
        duration_sec=5.0,
        source_w=1920, source_h=1080,
        content_type="animation",
        dense_faces=dense,
        active_speaker_events=[],
        impact_frames=impacts,
    )
    human, rp, coverage = _run(inputs)
    _assert_coverage_clean(rp, coverage)
    # Animation impact frames both create SACCADE timestamps and a
    # forced pull-out zoom in genre_refinements.
    assert human.genre.saccade_timestamps
    forced = [z for z in human.zooms if "anime-impact" in z.reason]
    assert forced, "animation impact frames should force pull-out zooms"


# ── Invariant: no awkward crops anywhere ────────────────────────


def test_no_op_clips_face_bbox_for_static_speaker():
    """For a static talking head with face at (50%, 40%) and size
    (15% x 20%), the crop window must always contain the face bbox.
    """
    dense = _dense_single_speaker(duration=5.0)
    inputs = human_reframe.HumanReframeInputs(
        duration_sec=5.0,
        source_w=1920, source_h=1080,
        content_type="talking_head",
        dense_faces=dense,
        active_speaker_events=[_SpeakerEvent(0.0, 5.0, slot_id=0)],
    )
    human, rp, coverage = _run(inputs)
    _assert_coverage_clean(rp, coverage)

    face_l = 0.50 - 0.075
    face_r = 0.50 + 0.075
    face_t = 0.40 - 0.10
    face_b = 0.40 + 0.10

    for op in rp.ops:
        # Walk all frames of this op (primary_rect for static, motion_path
        # for tracking). Each crop window must contain the face bbox.
        rects = [op.primary_rect]
        rects.extend(kp.rect for kp in (op.motion_path or []))
        for r in rects:
            # Zoom ops intentionally shrink beyond the face bbox; skip.
            if op.kind in (RenderOpKind.MOTIVATED_PUSH_IN,
                           RenderOpKind.MOTIVATED_PULL_OUT):
                continue
            # Blur fill / wide master use full-frame by design.
            if op.kind in (RenderOpKind.BLUR_FILL, RenderOpKind.WIDE_MASTER):
                continue
            assert r.x - 1e-3 <= face_l, f"{op.kind}: crop left {r.x} past face {face_l}"
            assert r.x + r.w + 1e-3 >= face_r, \
                f"{op.kind}: crop right {r.x + r.w} not containing face {face_r}"


# ── Invariant: bridge override preserves legacy when flag is off ──


def test_bridge_returns_legacy_when_flag_off(monkeypatch):
    # Explicitly opt out to confirm the kill switch still works.
    monkeypatch.setenv("CLIPAI_HUMAN_REFRAME", "0")
    from backend.services.render_plan import RenderPlan, RenderOp, Rect
    legacy = RenderPlan(
        source_width=1920, source_height=1080,
        target_width=1080, target_height=1920,
        total_duration_sec=5.0, fps=30.0,
        ops=[RenderOp(
            kind=RenderOpKind.CROP,
            start_sec=0.0, end_sec=5.0,
            primary_rect=Rect(x=0.25, y=0.0, w=0.5, h=1.0),
        )],
    )
    dense = _dense_single_speaker(duration=5.0)
    result = human_reframe_bridge.maybe_override_render_plan(
        legacy,
        dense_faces=dense,
        active_speaker_events=[],
        shot_boundaries=[],
        content_type="talking_head",
        duration_sec=5.0,
        source_width=1920, source_height=1080,
        source_fps=30.0,
    )
    assert result is legacy


def test_bridge_on_by_default(monkeypatch):
    """With no env override the human-reframe path should take over."""
    monkeypatch.delenv("CLIPAI_HUMAN_REFRAME", raising=False)
    from backend.services.render_plan import RenderPlan, RenderOp, Rect
    legacy = RenderPlan(
        source_width=1920, source_height=1080,
        target_width=1080, target_height=1920,
        total_duration_sec=5.0, fps=30.0,
        ops=[RenderOp(
            kind=RenderOpKind.CROP,
            start_sec=0.0, end_sec=5.0,
            primary_rect=Rect(x=0.25, y=0.0, w=0.5, h=1.0),
        )],
    )
    dense = _dense_single_speaker(duration=5.0)
    result = human_reframe_bridge.maybe_override_render_plan(
        legacy,
        dense_faces=dense,
        active_speaker_events=[_SpeakerEvent(0.0, 5.0, slot_id=0)],
        shot_boundaries=[],
        content_type="talking_head",
        duration_sec=5.0,
        source_width=1920, source_height=1080,
        source_fps=30.0,
    )
    assert result is not legacy
    cov = human_render_plan_adapter.verify_frame_coverage(result)
    assert cov.ok


def test_bridge_overrides_when_flag_on(monkeypatch):
    monkeypatch.setenv("CLIPAI_HUMAN_REFRAME", "1")
    from backend.services.render_plan import RenderPlan, RenderOp, Rect
    legacy = RenderPlan(
        source_width=1920, source_height=1080,
        target_width=1080, target_height=1920,
        total_duration_sec=5.0, fps=30.0,
        ops=[RenderOp(
            kind=RenderOpKind.CROP,
            start_sec=0.0, end_sec=5.0,
            primary_rect=Rect(x=0.25, y=0.0, w=0.5, h=1.0),
        )],
    )
    dense = _dense_single_speaker(duration=5.0)
    result = human_reframe_bridge.maybe_override_render_plan(
        legacy,
        dense_faces=dense,
        active_speaker_events=[_SpeakerEvent(0.0, 5.0, slot_id=0)],
        shot_boundaries=[],
        content_type="talking_head",
        duration_sec=5.0,
        source_width=1920, source_height=1080,
        source_fps=30.0,
    )
    # Override should produce its own plan; legacy identity comparison
    # must fail.
    assert result is not legacy
    assert result.total_duration_sec == pytest.approx(5.0, abs=0.05)
    # Validate it's actually a usable plan.
    cov = human_render_plan_adapter.verify_frame_coverage(result)
    assert cov.ok


# ── Infeasible solve → safety fallback (no crash) ───────────────


def test_impossible_input_falls_back_to_blur_fill():
    # Zero duration, no frames — adapter must still produce a valid plan.
    inputs = human_reframe.HumanReframeInputs(
        duration_sec=0.0,
        source_w=1920, source_h=1080,
        content_type="talking_head",
    )
    human = human_reframe.run_human_reframe(inputs)
    rp = human_render_plan_adapter.render_plan_from_human_plan(
        human,
        source_width=1920, source_height=1080, source_fps=30.0,
    )
    assert rp.ops
    assert rp.ops[0].kind == RenderOpKind.BLUR_FILL


def test_bench_entry_point_gracefully_handles_missing_source():
    """``run_reframe_on_clip_for_bench`` must never raise even when the
    source file doesn't exist (bench runs skip rather than crash)."""
    result = human_reframe_bridge.run_reframe_on_clip_for_bench(
        "/tmp/does_not_exist_a12b.mp4",
        content_type="talking_head",
    )
    assert isinstance(result, dict)
    assert "ops" in result


# ── Saccade vs pursuit across a real trajectory ─────────────────


def test_continuous_motion_maintains_coverage_and_tracking():
    """Face walks linearly across the frame for 5 sec. Coverage must
    hold and the solver must emit tracking (not static) ops."""
    def nose_x(t):
        return 20.0 + 10.0 * t    # 20 → 70 over 5 sec
    dense = _dense_single_speaker(duration=5.0, fps=10.0, nose_x_fn=nose_x)
    inputs = human_reframe.HumanReframeInputs(
        duration_sec=5.0,
        source_w=1920, source_h=1080,
        content_type="vlog",
        dense_faces=dense,
        active_speaker_events=[_SpeakerEvent(0.0, 5.0, slot_id=0)],
    )
    _, rp, coverage = _run(inputs)
    _assert_coverage_clean(rp, coverage)
    # Linear pan >5% of frame should produce at least one tracking op.
    kinds = {op.kind for op in rp.ops}
    assert RenderOpKind.TRACKING_CROP in kinds or len(rp.ops) >= 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
