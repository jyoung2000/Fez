"""Task 5 — naive-baseline reference generator + bench reads it.

Path B implementation. The MediaPipe AutoFlip Docker sidecar (Path A)
is heavier than this homelab can run, so we use a deterministic
center-crop baseline as the comparator. ClipAI being better than this
floor is the bare-minimum SOTA validation.

Run:  pytest tests/qa/test_naive_baseline_reference.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def test_box_smooth_window_one_pass_through():
    from backend.scripts.run_naive_baseline import _box_smooth
    out = _box_smooth([10.0, 20.0, 30.0, 40.0], window_s=0.0, dt=0.1)
    assert out == [10.0, 20.0, 30.0, 40.0]


def test_box_smooth_attenuates_step():
    from backend.scripts.run_naive_baseline import _box_smooth
    out = _box_smooth(
        [0.0, 0.0, 100.0, 100.0, 100.0], window_s=0.5, dt=0.1,
    )
    # The middle sample's box-window straddles the step.
    assert 0 < out[1] < 100
    # Edges are still close to their originals (tail averaging).
    assert out[0] < 50
    assert out[-1] > 50


def test_hold_through_gaps_carries_last_known():
    from backend.scripts.run_naive_baseline import _hold_through_gaps
    samples = [(0.0, 25.0), (1.0, None), (2.0, None), (3.0, 75.0)]
    held = _hold_through_gaps(samples, default_x=50.0)
    assert held == [(0.0, 25.0), (1.0, 25.0), (2.0, 25.0), (3.0, 75.0)]


def test_hold_through_gaps_uses_default_when_no_prior():
    from backend.scripts.run_naive_baseline import _hold_through_gaps
    samples = [(0.0, None), (1.0, None), (2.0, 80.0)]
    held = _hold_through_gaps(samples, default_x=50.0)
    assert held[0] == (0.0, 50.0)
    assert held[1] == (1.0, 50.0)
    assert held[2] == (2.0, 80.0)


def test_naive_baseline_events_shape(tmp_path):
    """Mock ffprobe + face detection to verify the event-list shape
    matches what the bench's load_autoflip_timeline + score_timeline
    expect."""
    from backend.scripts import run_naive_baseline

    fake = tmp_path / "stub.mp4"
    fake.write_bytes(b"\x00")

    with mock.patch.object(
        run_naive_baseline, "_probe_metadata",
        return_value={"duration": 5.0, "width": 1920, "height": 1080,
                      "fps": 10.0},
    ), mock.patch.object(
        run_naive_baseline, "_sparse_face_centers",
        return_value=[(0.5, 25.0), (1.5, 25.0), (2.5, 50.0),
                      (3.5, 75.0), (4.5, 75.0)],
    ):
        ref = run_naive_baseline.naive_baseline_events(str(fake))

    assert ref["tool"] == "naive_baseline"
    events = ref["events"]
    # 5 s × 10 fps = 50 events.
    assert len(events) == 50
    for e in events:
        assert {"t", "crop_cx", "crop_cy", "crop_w", "crop_h", "scene_change"} <= set(e)
        assert 0.0 <= e["crop_cx"] <= 1.0
    # Late frames should have moved toward the right (face migrated 25 → 75).
    assert events[-1]["crop_cx"] > events[0]["crop_cx"]


def test_score_timeline_consumes_baseline_events(tmp_path):
    from backend.scripts.compare_autoflip_vs_clipai import score_timeline
    from backend.scripts import run_naive_baseline

    fake = tmp_path / "stub.mp4"
    fake.write_bytes(b"\x00")
    with mock.patch.object(
        run_naive_baseline, "_probe_metadata",
        return_value={"duration": 3.0, "width": 1920, "height": 1080,
                      "fps": 10.0},
    ), mock.patch.object(
        run_naive_baseline, "_sparse_face_centers",
        return_value=[(0.5, 50.0), (1.5, 50.0), (2.5, 50.0)],
    ):
        ref = run_naive_baseline.naive_baseline_events(str(fake))

    metrics = score_timeline(ref["events"])
    assert metrics["n_segments"] >= 0
    assert metrics["max_acceleration"] is not None


def test_run_for_manifest_skips_missing_videos(tmp_path):
    from backend.scripts.run_naive_baseline import run_for_manifest

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "clips": [{"slug": "missing_clip", "ext": "mp4"}],
    }))
    output_dir = tmp_path / "out"

    summary = run_for_manifest(
        manifest_path, output_dir,
        real_content_dir=tmp_path / "videos_dont_exist",
    )
    assert summary["generated"] == []
    assert len(summary["skipped"]) == 1
    assert "video missing" in summary["skipped"][0][1]


def test_load_autoflip_timeline_consumes_naive_baseline(tmp_path):
    """The bench's load_autoflip_timeline reads the naive baseline
    references the same way it reads real AutoFlip output."""
    from backend.scripts.compare_autoflip_vs_clipai import load_autoflip_timeline
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()
    payload = {
        "tool": "naive_baseline",
        "metadata": {"duration": 1.0, "width": 1920, "height": 1080, "fps": 10.0},
        "events": [
            {"t": 0.0, "crop_cx": 0.5, "crop_cy": 0.5,
             "crop_w": 0.5625, "crop_h": 1.0, "scene_change": False},
        ],
    }
    (ref_dir / "panel_clip.json").write_text(json.dumps(payload))
    events = load_autoflip_timeline({"slug": "panel_clip"}, ref_dir)
    assert events is not None
    assert len(events) == 1


def test_load_autoflip_timeline_handles_missing_file(tmp_path):
    from backend.scripts.compare_autoflip_vs_clipai import load_autoflip_timeline
    out = load_autoflip_timeline({"slug": "no_such_slug"}, tmp_path)
    assert out is None
