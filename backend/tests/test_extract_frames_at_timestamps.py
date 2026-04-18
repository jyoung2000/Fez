"""Tests for the Phase 4 timestamp-based extractor wrapper.

Covers:
  - Return value shape (FrameData list, paths exist, timestamps sorted)
  - Empty input raises RuntimeError
  - Dedupe + sort behavior on out-of-order / duplicate inputs

These tests shell out to real ffmpeg to verify the wrapper end-to-end.
They skip gracefully when ffmpeg is not installed (e.g. unit-test CI
containers without the video toolchain).
"""

import asyncio
import os
import shutil
import subprocess
import tempfile

import pytest


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _make_test_video(path: str, duration: int = 10) -> None:
    """Generate a deterministic test video with a moving rectangle
    (testsrc) so every frame is visually distinct."""
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"testsrc=duration={duration}:size=320x240:rate=10",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


@pytest.fixture
def test_video():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.mp4")
        _make_test_video(path)
        yield path, tmp


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
def test_extract_at_timestamps_returns_requested_frames(test_video):
    from backend.services.frame_extractor import extract_frames_at_timestamps

    video_path, tmp = test_video
    out_dir = os.path.join(tmp, "out")
    timestamps = [1.0, 3.0, 5.0, 7.0]
    frames = asyncio.run(extract_frames_at_timestamps(
        video_path, out_dir, timestamps, video_codec="h264",
    ))
    assert len(frames) >= 3  # ffmpeg may drop the last one near EOF
    assert all(os.path.exists(f.path) for f in frames)
    assert all(f.timestamp >= 0 for f in frames)


def test_extract_at_timestamps_empty_raises():
    from backend.services.frame_extractor import extract_frames_at_timestamps

    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(RuntimeError):
            asyncio.run(extract_frames_at_timestamps(
                "/nonexistent.mp4", os.path.join(tmp, "out"), [],
            ))


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
def test_extract_at_timestamps_dedupes_and_sorts(test_video):
    from backend.services.frame_extractor import extract_frames_at_timestamps

    video_path, tmp = test_video
    out_dir = os.path.join(tmp, "out")
    # Out of order with duplicates — the wrapper should dedupe + sort.
    frames = asyncio.run(extract_frames_at_timestamps(
        video_path, out_dir, [5.0, 1.0, 5.0, 3.0, 1.0], video_codec="h264",
    ))
    timestamps = [f.timestamp for f in frames]
    assert timestamps == sorted(timestamps)
    assert len(set(timestamps)) == len(timestamps)
