"""Week 3 — unit tests for ``backend/scripts/export_autoflip_compatible.py``.

Exercises the render-plan and reframe-segment translators in
isolation, and monkey-patches ``subprocess.run`` to fake ffprobe so
``build_timeline`` can run in the sandbox without a real video file.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.scripts.export_autoflip_compatible import (
    _aspect_to_float,
    build_timeline,
    probe_source,
    reframe_segments_to_events,
    render_plan_to_events,
)


# ───────────────────────── _aspect_to_float ─────────────────────────


def test_aspect_to_float_standard_ratios():
    assert _aspect_to_float("9:16") == pytest.approx(0.5625)
    assert _aspect_to_float("16:9") == pytest.approx(16.0 / 9.0)
    assert _aspect_to_float("1:1") == 1.0


def test_aspect_to_float_malformed_falls_back_to_9_16():
    assert _aspect_to_float("garbage") == pytest.approx(0.5625)
    assert _aspect_to_float("9:0") == pytest.approx(0.5625)
    assert _aspect_to_float("") == pytest.approx(0.5625)


# ───────────────────────── render_plan_to_events ─────────────────────────


def test_render_plan_two_ops_produces_expected_frame_count():
    """Two ops covering 0–2s and 2–4s at 30 fps → 120 events."""
    plan = {
        "ops": [
            {
                "start_sec": 0.0,
                "end_sec": 2.0,
                "primary_rect": {"x": 0.3, "y": 0.0, "w": 0.3, "h": 1.0},
            },
            {
                "start_sec": 2.0,
                "end_sec": 4.0,
                "primary_rect": {"x": 0.5, "y": 0.0, "w": 0.3, "h": 1.0},
            },
        ]
    }
    events = render_plan_to_events(plan, src_w=1920, src_h=1080, fps=30.0)
    assert len(events) == 120


def test_render_plan_first_frame_of_each_op_is_scene_change():
    """Frame 0 and frame 60 (start of op 2 at 2.0s × 30 fps) are marked."""
    plan = {
        "ops": [
            {
                "start_sec": 0.0,
                "end_sec": 2.0,
                "primary_rect": {"x": 0.3, "y": 0.0, "w": 0.3, "h": 1.0},
            },
            {
                "start_sec": 2.0,
                "end_sec": 4.0,
                "primary_rect": {"x": 0.5, "y": 0.0, "w": 0.3, "h": 1.0},
            },
        ]
    }
    events = render_plan_to_events(plan, src_w=1920, src_h=1080, fps=30.0)
    assert events[0]["scene_change"] is True
    assert events[1]["scene_change"] is False
    assert events[59]["scene_change"] is False
    assert events[60]["scene_change"] is True
    assert events[61]["scene_change"] is False


def test_render_plan_crop_center_from_rect():
    """crop_cx = x + w/2, crop_cy = y + h/2."""
    plan = {
        "ops": [
            {
                "start_sec": 0.0, "end_sec": 1.0,
                "primary_rect": {
                    "x": 0.2, "y": 0.1, "w": 0.3, "h": 0.8,
                },
            },
        ]
    }
    events = render_plan_to_events(plan, src_w=1920, src_h=1080, fps=30.0)
    assert len(events) == 30
    assert events[0]["crop_cx"] == pytest.approx(0.35)   # 0.2 + 0.15
    assert events[0]["crop_cy"] == pytest.approx(0.50)   # 0.1 + 0.4
    assert events[0]["crop_w"] == pytest.approx(0.3)
    assert events[0]["crop_h"] == pytest.approx(0.8)


def test_render_plan_zero_duration_op_dropped():
    """An op whose end rounds to the same frame as its start emits nothing."""
    plan = {
        "ops": [
            {
                "start_sec": 0.0, "end_sec": 0.0,
                "primary_rect": {"x": 0.0, "y": 0.0, "w": 0.3, "h": 1.0},
            },
        ]
    }
    events = render_plan_to_events(plan, src_w=1920, src_h=1080, fps=30.0)
    assert events == []


def test_render_plan_empty_ops_list():
    assert render_plan_to_events({}, src_w=1920, src_h=1080, fps=30.0) == []
    assert render_plan_to_events({"ops": []}, 1920, 1080, 30.0) == []


# ───────────────────────── reframe_segments_to_events ─────────────────────────


def test_reframe_segments_normalizes_pixel_coords():
    """ReframeSegment subject_x is in source pixels; event crop_cx is 0-1."""
    segs = [
        {
            "start": 0.0,
            "end": 1.0,
            "subject_x": 960.0,   # mid-frame
            "subject_y": 540.0,
        },
        {
            "start": 1.0,
            "end": 2.0,
            "subject_x": 480.0,
            "subject_y": 270.0,
        },
    ]
    events = reframe_segments_to_events(
        segs, src_w=1920, src_h=1080, fps=30.0,
    )
    assert len(events) == 60
    assert events[0]["crop_cx"] == pytest.approx(0.5)
    assert events[0]["crop_cy"] == pytest.approx(0.5)
    assert events[30]["crop_cx"] == pytest.approx(0.25)
    assert events[30]["crop_cy"] == pytest.approx(0.25)


def test_reframe_segments_crop_w_matches_target_aspect():
    """Crop width = source_height × target_aspect / source_width."""
    segs = [
        {"start": 0.0, "end": 1.0, "subject_x": 960.0, "subject_y": 540.0},
    ]
    events = reframe_segments_to_events(
        segs, src_w=1920, src_h=1080, fps=30.0, aspect_ratio="9:16",
    )
    # 1080 * (9/16) / 1920 = 607.5 / 1920 = 0.3164...
    expected_crop_w = (1080 * 0.5625) / 1920
    assert events[0]["crop_w"] == pytest.approx(expected_crop_w, rel=1e-6)
    assert events[0]["crop_h"] == 1.0


def test_reframe_segments_first_event_scene_change():
    """First frame of each segment is a scene change."""
    segs = [
        {"start": 0.0, "end": 1.0, "subject_x": 960.0, "subject_y": 540.0},
        {"start": 1.0, "end": 2.0, "subject_x": 480.0, "subject_y": 540.0},
    ]
    events = reframe_segments_to_events(
        segs, src_w=1920, src_h=1080, fps=30.0,
    )
    assert events[0]["scene_change"] is True
    assert events[1]["scene_change"] is False
    assert events[30]["scene_change"] is True


def test_reframe_segments_accepts_dataclass_instances():
    """Translator tolerates object-style segments (not just dicts)."""
    class _Seg:
        def __init__(self, start, end, subject_x, subject_y):
            self.start = start
            self.end = end
            self.subject_x = subject_x
            self.subject_y = subject_y

    segs = [_Seg(0.0, 1.0, 960.0, 540.0)]
    events = reframe_segments_to_events(
        segs, src_w=1920, src_h=1080, fps=30.0,
    )
    assert len(events) == 30
    assert events[0]["crop_cx"] == pytest.approx(0.5)


def test_reframe_segments_empty_list_returns_empty_events():
    assert reframe_segments_to_events(
        [], src_w=1920, src_h=1080, fps=30.0,
    ) == []


# ───────────────────────── probe_source + build_timeline ─────────────────────────


def _fake_ffprobe(width: int, height: int, fps_str: str):
    """Return a subprocess.run replacement that emits the given dimensions."""
    def _run(cmd, capture_output=True, text=True, check=True):
        stdout = f"{width},{height},{fps_str}"
        return SimpleNamespace(stdout=stdout, stderr="", returncode=0)
    return _run


def test_build_timeline_with_reframe_segments(tmp_path, monkeypatch):
    """End-to-end: build_timeline reads a segment JSON + fakes ffprobe."""
    import backend.scripts.export_autoflip_compatible as mod

    segs_path = tmp_path / "segs.json"
    segs_path.write_text(json.dumps([
        {"start": 0.0, "end": 2.0, "subject_x": 960, "subject_y": 540},
        {"start": 2.0, "end": 4.0, "subject_x": 480, "subject_y": 540},
    ]))
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"")  # path must exist for the caller convention

    monkeypatch.setattr(mod.subprocess, "run",
                        _fake_ffprobe(1920, 1080, "30/1"))

    timeline = mod.build_timeline(
        video=str(video_path),
        reframe_segments_json=str(segs_path),
    )

    assert timeline["tool"] == "clipai"
    assert timeline["source_width"] == 1920
    assert timeline["source_height"] == 1080
    assert timeline["source_fps"] == pytest.approx(30.0)
    assert timeline["aspect_ratio"] == "9:16"
    assert len(timeline["events"]) == 120


def test_build_timeline_with_render_plan(tmp_path, monkeypatch):
    import backend.scripts.export_autoflip_compatible as mod

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "ops": [
            {
                "start_sec": 0.0, "end_sec": 3.0,
                "primary_rect": {
                    "x": 0.3, "y": 0.0, "w": 0.3, "h": 1.0,
                },
            },
        ]
    }))
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"")

    monkeypatch.setattr(mod.subprocess, "run",
                        _fake_ffprobe(1920, 1080, "30/1"))

    timeline = mod.build_timeline(
        video=str(video_path),
        render_plan_json=str(plan_path),
    )
    assert len(timeline["events"]) == 90  # 3s × 30fps


def test_build_timeline_requires_one_input(tmp_path, monkeypatch):
    import backend.scripts.export_autoflip_compatible as mod
    monkeypatch.setattr(mod.subprocess, "run",
                        _fake_ffprobe(1920, 1080, "30/1"))
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"")

    with pytest.raises(ValueError, match="render-plan-json"):
        mod.build_timeline(video=str(video_path))


def test_probe_source_parses_ffprobe_output(monkeypatch):
    import backend.scripts.export_autoflip_compatible as mod
    monkeypatch.setattr(mod.subprocess, "run",
                        _fake_ffprobe(1280, 720, "29970/1000"))
    w, h, fps = probe_source("/dev/null")
    assert (w, h) == (1280, 720)
    assert fps == pytest.approx(29.97, rel=1e-4)


def test_probe_source_rejects_malformed_output(monkeypatch):
    import backend.scripts.export_autoflip_compatible as mod

    def _bad_run(*args, **kwargs):
        return SimpleNamespace(stdout="garbage", stderr="", returncode=0)

    monkeypatch.setattr(mod.subprocess, "run", _bad_run)
    with pytest.raises(ValueError, match="unexpected ffprobe output"):
        probe_source("/dev/null")
