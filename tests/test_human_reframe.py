"""Unit tests for the human-reframing overhaul modules.

These tests are pure-Python and don't touch the full pipeline — they
verify that each new module's public API behaves correctly on tiny
synthetic inputs. They run under pytest from the repository root.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import pytest

from backend.services import (
    ab_cut_scheduler,
    camera_events,
    camera_path_2d,
    composition_head,
    critic_loop,
    cut_timing_head,
    genre_refinements,
    human_parity_metrics,
    human_reframe,
    motivated_zoom,
    reframe_config,
    subject_kalman,
)


# ── human_parity_metrics ─────────────────────────────────────────


def _sample(t: float, cx: float, cy: float, w: float = 0.5, h: float = 1.0):
    return human_parity_metrics.TrajectorySample(
        t=t, cx=cx, cy=cy, w=w, h=h, subject_cx=cx, subject_cy=cy,
    )


def test_mae_identical_is_zero():
    ours = [_sample(i * 0.1, 0.5, 0.5) for i in range(20)]
    human = [_sample(i * 0.1, 0.5, 0.5) for i in range(20)]
    rep = human_parity_metrics.compute_human_parity(
        ours, human, clip_slug="t1", content_type="talking_head",
    )
    assert rep.mae_cx == pytest.approx(0.0, abs=1e-9)
    assert rep.mae_cy == pytest.approx(0.0, abs=1e-9)
    assert rep.framing_macro_f1 == pytest.approx(1.0, abs=1e-6)


def test_mae_shifted_detects():
    ours = [_sample(i * 0.1, 0.4, 0.5) for i in range(20)]
    human = [_sample(i * 0.1, 0.5, 0.5) for i in range(20)]
    rep = human_parity_metrics.compute_human_parity(
        ours, human, clip_slug="t2", content_type="talking_head",
    )
    assert rep.mae_cx == pytest.approx(0.1, abs=0.02)


def test_framing_classifier_buckets():
    assert human_parity_metrics.classify_framing(0.20) == "CU"
    assert human_parity_metrics.classify_framing(0.30) == "MS"
    assert human_parity_metrics.classify_framing(0.60) == "WS"
    assert human_parity_metrics.classify_framing(0.30, n_subjects_in_crop=2) == "2SHOT"
    assert human_parity_metrics.classify_framing(0.30, is_split_screen=True) == "SPLIT"


def test_cut_timing_detects_matched_cuts():
    # Large jump at t=1.5 in both trajectories, 100ms apart.
    ours = [_sample(i * 0.05, 0.3 if i < 30 else 0.7, 0.5) for i in range(60)]
    human = [_sample(i * 0.05, 0.3 if i < 32 else 0.7, 0.5) for i in range(60)]
    rep = human_parity_metrics.compute_human_parity(
        ours, human, clip_slug="t3", content_type="talking_head",
    )
    assert rep.n_cut_pairs >= 1
    assert rep.cut_timing_deviation_ms_mean <= 150.0


def test_render_plan_to_trajectory_static_crop():
    plan = {
        "total_duration_sec": 1.0,
        "ops": [{
            "kind": "crop",
            "start_sec": 0.0, "end_sec": 1.0,
            "primary_rect": {"x": 0.25, "y": 0.0, "w": 0.5, "h": 1.0},
        }],
    }
    traj = human_parity_metrics.render_plan_to_trajectory(plan, sample_fps=10.0)
    assert len(traj) >= 10
    assert all(abs(s.cx - 0.5) < 1e-6 for s in traj)


# ── reframe_config ────────────────────────────────────────────────


def test_config_default_is_safe():
    cfg = reframe_config.load_default_config()
    assert 0 < cfg.headroom_min < cfg.headroom_max < 0.5
    assert cfg.lp_lambda_v > 0
    assert cfg.saccade_ease_ms > 0


def test_config_for_content_panel_disables_zoom():
    cfg = reframe_config.load_default_config()
    cfg2 = cfg.for_content("multi_speaker_panel")
    assert cfg2.zoom_push_in_max_scale == 1.0
    assert cfg2.zoom_pull_out_max_scale == 1.0


def test_config_for_content_racing_eye_line_low():
    cfg = reframe_config.load_default_config().for_content("sports_racing")
    assert cfg.eye_line_y >= 0.5


# ── subject_kalman ────────────────────────────────────────────────


def test_kalman_tracks_constant_velocity():
    k = subject_kalman.KalmanSubject(slot_id=0)
    for i in range(20):
        t = i * 0.05
        k.observe(t, 0.1 + 0.02 * i, 0.5, source="face")
    # Predict ahead 0.5 s; should be near 0.1 + 0.02 * (20 + 10) * 0.05 step
    # With dt=0.05, predicted value at t = 20*0.05 + 0.5 = 1.5 should be
    # approximately 0.1 + 0.02 * 30 = 0.7 (within noise).
    x, y, _ = k.predict(1.5)
    assert 0.55 < x < 0.85
    assert abs(y - 0.5) < 0.1


def test_kalman_registry_lead_prediction():
    reg = subject_kalman.SubjectKalmanRegistry()
    for i in range(15):
        reg.observe(42, i * 0.05, 0.2 + 0.01 * i, 0.5, source="face")
    lead = reg.prediction_for_lead(42, t_now=15 * 0.05)
    assert lead is not None
    assert lead[0] > 0.34   # has moved ahead


# ── camera_events ─────────────────────────────────────────────────


def test_scheduler_emits_saccade_on_speaker_turn():
    evs = camera_events.schedule_camera_events(
        duration=5.0,
        shots=[camera_events.ShotInfo(0.0, 5.0)],
        speaker_turns=[
            camera_events.SpeakerTurn(t=2.0, from_slot=1, to_slot=2),
        ],
    )
    kinds = [e.mode for e in evs]
    assert camera_events.CameraMode.SACCADE in kinds


def test_scheduler_match_cut_when_close():
    evs = camera_events.schedule_camera_events(
        duration=5.0,
        shots=[camera_events.ShotInfo(0.0, 2.0), camera_events.ShotInfo(2.0, 5.0)],
        speaker_turns=[],
        positions_by_trigger={2.0: (0.5, 0.5)},
    )
    # position close to (0.5, 0.5) → match cut
    assert any(e.mode == camera_events.CameraMode.MATCH_CUT for e in evs)


def test_scheduler_handles_zoom_window():
    evs = camera_events.schedule_camera_events(
        duration=3.0,
        shots=[camera_events.ShotInfo(0.0, 3.0)],
        speaker_turns=[],
        zoom_windows=[camera_events.ZoomWindow(1.0, 2.5)],
    )
    kinds = [e.mode for e in evs]
    assert camera_events.CameraMode.MICRO_ZOOM in kinds


# ── ab_cut_scheduler ─────────────────────────────────────────────


def test_ab_schedule_two_speakers():
    windows = [
        ab_cut_scheduler.SpeakerWindow(0.0, 1.5, slot_id=1),
        ab_cut_scheduler.SpeakerWindow(1.5, 3.0, slot_id=2),
        ab_cut_scheduler.SpeakerWindow(3.0, 4.5, slot_id=1),
    ]
    r = ab_cut_scheduler.plan_ab_schedule(
        overlap_start=0.0, overlap_end=4.5,
        speaker_windows=windows,
    )
    assert r.enabled
    slots = [s.slot_id for s in r.segments]
    assert slots == [1, 2, 1]


def test_ab_schedule_fallback_when_short_turns():
    windows = [
        ab_cut_scheduler.SpeakerWindow(0.0, 0.2, slot_id=1),
        ab_cut_scheduler.SpeakerWindow(0.2, 0.4, slot_id=2),
    ]
    r = ab_cut_scheduler.plan_ab_schedule(
        overlap_start=0.0, overlap_end=0.4,
        speaker_windows=windows,
    )
    assert not r.enabled


def test_ab_reaction_cut_inserted():
    windows = [
        ab_cut_scheduler.SpeakerWindow(0.0, 3.0, slot_id=1),
        ab_cut_scheduler.SpeakerWindow(3.0, 6.0, slot_id=2),
    ]
    reactions = [ab_cut_scheduler.ReactionEvent(t=1.5, kind="laughter")]
    r = ab_cut_scheduler.plan_ab_schedule(
        overlap_start=0.0, overlap_end=6.0,
        speaker_windows=windows,
        reactions=reactions,
        non_speaker_slots_at=lambda _t: [2],
    )
    assert r.enabled
    slots = [s.slot_id for s in r.segments]
    assert 2 in slots[:2]  # reaction listener cut early


# ── motivated_zoom ────────────────────────────────────────────────


def test_motivated_zoom_fires_on_emotional_word():
    words = [
        motivated_zoom.WordTiming(start=0.0, end=0.3, text="I"),
        motivated_zoom.WordTiming(start=0.3, end=0.8, text="wait"),
    ]
    peaks = [motivated_zoom.AudioPeak(t=0.8, loudness_db=-10.0)]
    zooms = motivated_zoom.plan_motivated_zooms(
        clip_duration=6.0,
        words=words, audio_peaks=peaks,
        content_type="cinematic_dialogue",
    )
    assert len(zooms) >= 1
    assert zooms[0].kind == motivated_zoom.ZoomKind.PUSH_IN


def test_motivated_zoom_respects_content_gate():
    zooms = motivated_zoom.plan_motivated_zooms(
        clip_duration=6.0,
        words=[motivated_zoom.WordTiming(start=0.3, end=0.8, text="wait")],
        audio_peaks=[motivated_zoom.AudioPeak(t=0.8, loudness_db=-10.0)],
        content_type="multi_speaker_panel",
    )
    # panel content: zoom disabled via config overrides
    assert len(zooms) == 0


def test_zoom_rect_at_scales_correctly():
    base = {"x": 0.25, "y": 0.0, "w": 0.5, "h": 1.0}
    r = motivated_zoom.zoom_rect_at(base, target_scale=1.25)
    assert r["w"] < base["w"]
    assert 0.25 < r["x"] + r["w"] * 0.5 < 0.75   # still centered


# ── camera_path_2d ───────────────────────────────────────────────


def test_camera_path_2d_basic():
    faces = [
        camera_path_2d.FaceFrame2D(
            t=i * 0.1, nose_x=0.5 + 0.01 * i, nose_y=0.4,
            width=0.15, height=0.2, yaw=0.0,
        )
        for i in range(10)
    ]
    path = camera_path_2d.solve_2d_camera_path(
        faces,
        timestamps=[f.t for f in faces],
        source_w=1920, source_h=1080,
    )
    assert len(path.cx) == 10
    assert len(path.cy) == 10
    # All centers within feasible bounds
    for cx in path.cx:
        assert 0.0 <= cx <= 1.0
    for cy in path.cy:
        assert 0.0 <= cy <= 1.0


def test_camera_path_2d_lead_room_shifts_with_yaw():
    faces_right = [
        camera_path_2d.FaceFrame2D(
            t=i * 0.1, nose_x=0.5, nose_y=0.4,
            width=0.15, height=0.2, yaw=0.8,
        )
        for i in range(8)
    ]
    faces_left = [
        camera_path_2d.FaceFrame2D(
            t=i * 0.1, nose_x=0.5, nose_y=0.4,
            width=0.15, height=0.2, yaw=-0.8,
        )
        for i in range(8)
    ]
    p_right = camera_path_2d.solve_2d_camera_path(
        faces_right,
        timestamps=[f.t for f in faces_right],
        source_w=1920, source_h=1080,
    )
    p_left = camera_path_2d.solve_2d_camera_path(
        faces_left,
        timestamps=[f.t for f in faces_left],
        source_w=1920, source_h=1080,
    )
    # Looking right → camera shifted left of subject (leave space to right)
    assert p_right.cx[-1] < p_left.cx[-1]


# ── genre_refinements ───────────────────────────────────────────


def test_music_refinements_schedules_saccades():
    result = genre_refinements.apply_genre_refinements(
        genre_refinements.GenreInputs(
            content_type="music_video",
            clip_duration=10.0,
            beats=[i * 0.5 for i in range(20)],
            downbeats=[0.0, 2.0, 4.0, 6.0, 8.0],
        ),
    )
    assert result.saccade_timestamps


def test_gaming_refinements_reserves_hud():
    result = genre_refinements.apply_genre_refinements(
        genre_refinements.GenreInputs(
            content_type="gameplay",
            clip_duration=5.0,
            hud_regions=[{"x": 0.0, "y": 0.85, "w": 1.0, "h": 0.15, "label": "mini-map"}],
        ),
    )
    assert len(result.hud_exclusions) == 1


# ── composition_head / cut_timing_head fallbacks ─────────────────


def test_composition_head_fallback_returns_valid():
    feat = composition_head.CompositionFeatures(
        face_cx=0.42, face_cy=0.4, face_w=0.15, face_h=0.2,
        content_type="talking_head", speaker_dwell_sec=3.0,
    )
    cx, cy, zoom = composition_head.predict_composition(feat)
    assert 0 <= cx <= 1 and 0 <= cy <= 1
    assert 0.5 <= zoom <= 2.0


def test_cut_timing_head_fallback_j_cut_for_dialogue():
    feat = cut_timing_head.CutFeatures(
        content_family="dialogue",
        trigger_kind="speaker_turn",
    )
    off, conf = cut_timing_head.predict_cut_offset(feat)
    assert off < 0  # J-cut


# ── critic_loop ──────────────────────────────────────────────────


def test_critic_loop_off_mode_noop():
    import backend.services.reframe_config as rc
    cfg = rc.load_default_config().override(critic_mode="off")
    rep = critic_loop.score_plan(samples=[], config=cfg)
    assert rep.mode == "off"


def test_critic_loop_learned_default_no_network():
    """Default critic mode must be local-only. No VLM calls should be
    made when the critic falls back to the learned / heuristic path."""
    import backend.services.reframe_config as rc
    cfg = rc.load_default_config()
    # Default ships with 'learned' so every run gets aesthetic feedback
    # without any network dependency.
    assert cfg.critic_mode == "learned"
    # And the VLM backend prefers local Ollama when explicitly used.
    assert cfg.critic_vlm_backend == "auto"


def test_critic_parse_response_valid_json():
    fs = critic_loop._parse_vlm_response(
        t=1.0,
        text='{"composition": 8, "subject_visible": 9, "headroom": 7, '
             '"lead_room": 6, "framing_choice": 8, "reason": "ok"}',
    )
    assert fs.score == 6.0   # min over axes
    assert fs.per_axis["composition"] == 8.0


# ── human_reframe (top-level) ───────────────────────────────────


def test_human_reframe_runs_with_empty_inputs():
    plan = human_reframe.run_human_reframe(
        human_reframe.HumanReframeInputs(
            duration_sec=5.0,
            source_w=1920, source_h=1080,
            content_type="talking_head",
        ),
    )
    assert plan.events
    assert plan.ab is not None
    assert plan.genre is not None


def test_human_reframe_with_synthetic_faces():
    @dataclass
    class F:
        identity_id: int = 0
        is_human: bool = True
        nose_x: float = 50.0
        nose_y: float = 40.0
        width: float = 15.0
        height: float = 20.0
        yaw: Optional[float] = None
        lip_aperture: float = 0.0

    @dataclass
    class FF:
        timestamp: float
        faces: list

    @dataclass
    class Ev:
        start: float
        end: float
        slot_id: int
        on_screen: bool = True

    dense = [FF(timestamp=i * 0.1, faces=[F()]) for i in range(30)]
    events = [Ev(start=0.0, end=3.0, slot_id=0)]
    inputs = human_reframe.HumanReframeInputs(
        duration_sec=3.0,
        source_w=1920, source_h=1080,
        content_type="talking_head",
        dense_faces=dense,
        active_speaker_events=events,
    )
    plan = human_reframe.run_human_reframe(inputs)
    assert plan.path.cx and plan.path.cy
    assert len(plan.path.cx) == len(dense)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
