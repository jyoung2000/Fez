"""Task B: --single-clip mode of run_naive_baseline.

Locks down the contract that lets the SOTA bench's pre-phase generate
a naive_baseline AutoFlip reference inline. Without this, every
single-clip run against an unfamiliar fixture reports
``AutoFlip references: 0 real, 0 naive_baseline`` — the bench's
``autoflip`` column stays null and operators can't compare.

Mocks ``_probe_metadata`` + ``_sparse_face_centers`` so the suite
runs in seconds without ffprobe / OpenCV.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import backend.scripts.run_naive_baseline as naive


@pytest.fixture(autouse=True)
def _stub_video_pipeline(monkeypatch):
    """Skip ffprobe + FaceDetector in every test in this module."""
    monkeypatch.setattr(
        naive, "_probe_metadata",
        lambda _video_path: {
            "duration": 6.0, "width": 1920, "height": 1080, "fps": 30.0,
        },
    )
    monkeypatch.setattr(
        naive, "_sparse_face_centers",
        lambda _video_path, _duration, sample_count: [
            (i * 1.0 + 0.5, 50.0 + i * 5.0) for i in range(sample_count)
        ],
    )


# ── 1. Single-clip mode emits one well-shaped JSON file ───────────────


def test_run_naive_baseline_single_clip_mode(tmp_path):
    src = tmp_path / "panel_test_clip.mp4"
    src.write_bytes(b"\x00" * 16)  # placeholder; mocked probe ignores content
    out_dir = tmp_path / "refs"

    rc = naive.main([
        "--single-clip", str(src),
        "--slug", "panel_test_clip",
        "--output-dir", str(out_dir),
    ])
    assert rc == 0

    out_path = out_dir / "panel_test_clip.json"
    assert out_path.is_file()
    payload = json.loads(out_path.read_text())
    assert payload["tool"] == "naive_baseline"
    assert isinstance(payload["events"], list)
    assert len(payload["events"]) > 0
    # Each event has the AutoFlip-shape keys.
    for ev in payload["events"][:3]:
        assert {"t", "crop_cx", "crop_cy", "crop_w", "crop_h"} <= set(ev)


# ── 2. Missing source file → exit 1, no output ────────────────────────


def test_run_naive_baseline_single_clip_handles_missing(tmp_path, caplog):
    out_dir = tmp_path / "refs"
    rc = naive.main([
        "--single-clip", str(tmp_path / "does_not_exist.mp4"),
        "--output-dir", str(out_dir),
    ])
    assert rc == 1
    # No file written on failure.
    if out_dir.is_dir():
        assert list(out_dir.iterdir()) == []


# ── 3. Default slug from source basename ──────────────────────────────


def test_run_naive_baseline_single_clip_uses_basename_slug(tmp_path):
    src = tmp_path / "vlog_morning_walk.mp4"
    src.write_bytes(b"\x00" * 16)
    out_dir = tmp_path / "refs"

    rc = naive.main([
        "--single-clip", str(src),
        "--output-dir", str(out_dir),
    ])
    assert rc == 0
    assert (out_dir / "vlog_morning_walk.json").is_file()


# ── 4. SSE pre-phase: skip when reference exists ──────────────────────
# Asserted via the helper directly to avoid spinning up FastAPI in this
# test module — the diagnostics endpoint just calls the helper.


def test_generate_single_clip_reference_writes_when_missing(tmp_path):
    src = tmp_path / "podcast_clip.mp4"
    src.write_bytes(b"\x00")
    out_dir = tmp_path / "refs"

    out_path = naive.generate_single_clip_reference(
        src, out_dir, slug="podcast_clip",
    )
    assert out_path is not None
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["tool"] == "naive_baseline"


def test_generate_single_clip_reference_returns_none_on_missing(tmp_path):
    out_dir = tmp_path / "refs"
    result = naive.generate_single_clip_reference(
        tmp_path / "missing.mp4", out_dir, slug="missing",
    )
    assert result is None


# ── 5. Mutual-exclusivity guard ────────────────────────────────────────


def test_single_clip_and_manifest_are_mutually_exclusive(tmp_path):
    src = tmp_path / "x.mp4"
    src.write_bytes(b"\x00")
    fake_manifest = tmp_path / "m.json"
    fake_manifest.write_text("{}")
    rc = naive.main([
        "--single-clip", str(src),
        "--manifest", str(fake_manifest),
        "--output-dir", str(tmp_path / "refs"),
    ])
    assert rc == 2
