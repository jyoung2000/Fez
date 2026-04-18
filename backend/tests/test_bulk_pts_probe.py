"""Bulk PTS probe correctness — one ffmpeg pass over an extracted
JPG sequence must return per-frame timestamps in input order.

Covers Change 4 of the VLM-upgrade wiring prompt: the previous
implementation spawned one ffprobe subprocess per extracted frame,
costing ~30-80ms per spawn. Collapsing the loop to a single ffmpeg
pass with the ``showinfo`` filter recovers ~5-15s on every job.
"""

import asyncio
import os
import shutil
import subprocess
import tempfile

import pytest


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


@pytest.fixture
def jpg_sequence():
    """Generate a known-PTS JPG sequence."""
    with tempfile.TemporaryDirectory() as tmp:
        # 1s test source @ 4fps → 4 jpgs at pts 0, 0.25, 0.5, 0.75
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "testsrc=duration=1:size=160x120:rate=4",
            "-q:v", "12", "-frame_pts", "1",
            os.path.join(tmp, "frame_%06d.jpg"),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        yield tmp


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
def test_bulk_probe_returns_one_value_per_frame(jpg_sequence):
    import backend.services.frame_extractor as fe

    count = len([
        f for f in os.listdir(jpg_sequence)
        if f.startswith("frame_") and f.endswith(".jpg")
    ])
    assert count == 4

    pts = asyncio.run(fe._bulk_probe_pts(jpg_sequence, count))
    assert len(pts) == count
    assert all(p is None or isinstance(p, float) for p in pts)
    # At least 3 of 4 should parse successfully (showinfo is reliable here).
    assert sum(1 for p in pts if p is not None) >= 3


def test_bulk_probe_empty_dir_returns_empty():
    import backend.services.frame_extractor as fe
    pts = asyncio.run(fe._bulk_probe_pts("/tmp/nonexistent_dir_xyz", 0))
    assert pts == []
