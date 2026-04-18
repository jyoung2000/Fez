"""Tests for the thumbnail extractor service."""

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.services.thumbnail_extractor import (
    select_best_frame_timestamp,
    extract_thumbnail,
    get_thumbnail_dir,
)


class TestSelectBestFrameTimestamp:
    def test_high_importance_scene_selected(self):
        """Highest-importance scene's timestamp is chosen."""
        scenes = [
            MagicMock(importance_score=3, timestamp=2.0),
            MagicMock(importance_score=9, timestamp=15.5),
            MagicMock(importance_score=5, timestamp=8.0),
        ]
        ts = select_best_frame_timestamp(60.0, scenes=scenes)
        assert ts == 15.5

    def test_stationary_segment_fallback(self):
        """When no scenes, longest stationary segment midpoint is used."""
        seg1 = MagicMock(strategy="stationary", start=0.0, end=2.0)
        seg2 = MagicMock(strategy="stationary", start=5.0, end=15.0)
        seg3 = MagicMock(strategy="tracking", start=15.0, end=30.0)
        ts = select_best_frame_timestamp(30.0, reframe_segments=[seg1, seg2, seg3])
        assert ts == 10.0  # midpoint of seg2

    def test_25_percent_fallback(self):
        """With no scenes or segments, returns 25% of duration."""
        ts = select_best_frame_timestamp(100.0)
        assert ts == 25.0

    def test_zero_duration_returns_1(self):
        """Zero duration returns 1.0 (safe fallback)."""
        ts = select_best_frame_timestamp(0.0)
        assert ts == 1.0

    def test_scenes_with_zero_importance_ignored(self):
        """Scenes with importance_score=0 are skipped."""
        scenes = [MagicMock(importance_score=0, timestamp=5.0)]
        ts = select_best_frame_timestamp(40.0, scenes=scenes)
        assert ts == 10.0  # 25% of 40

    def test_scenes_with_none_timestamp_ignored(self):
        """Scenes with timestamp=None are skipped."""
        scenes = [MagicMock(importance_score=8, timestamp=None)]
        ts = select_best_frame_timestamp(20.0, scenes=scenes)
        assert ts == 5.0  # 25% of 20


def _ffmpeg_available():
    """Check if ffmpeg is installed."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class TestExtractThumbnail:
    def test_missing_source_returns_none(self, caplog):
        """Missing source video returns None with warning."""
        result = extract_thumbnail(
            job_id="test123",
            source_video_path="/nonexistent/video.mp4",
            video_duration=60.0,
        )
        assert result is None
        assert "source video missing" in caplog.text

    @pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
    def test_ffmpeg_extraction_on_synthetic_video(self, tmp_path):
        """Extract thumbnail from a synthetic FFmpeg-generated test video."""
        # Generate a 5-second test video with FFmpeg lavfi
        video_path = tmp_path / "test_video.mp4"
        gen_cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i",
            "testsrc=duration=5:size=640x360:rate=25",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(video_path),
        ]
        gen_result = subprocess.run(gen_cmd, capture_output=True, timeout=30)
        if gen_result.returncode != 0:
            pytest.skip("ffmpeg not available or lavfi test source failed")

        output_dir = tmp_path / "thumbs"
        output_dir.mkdir()

        result = extract_thumbnail(
            job_id="synth_test",
            source_video_path=str(video_path),
            video_duration=5.0,
            output_dir=output_dir,
        )

        assert result is not None
        assert result.exists()
        assert result.stat().st_size >= 1000
        assert result.name == "synth_test.jpg"

    @pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
    def test_thumbnail_dimensions(self, tmp_path):
        """Extracted thumbnail has correct 1200x630 dimensions."""
        video_path = tmp_path / "test_video.mp4"
        gen_cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i",
            "testsrc=duration=2:size=1920x1080:rate=25",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(video_path),
        ]
        gen_result = subprocess.run(gen_cmd, capture_output=True, timeout=30)
        if gen_result.returncode != 0:
            pytest.skip("ffmpeg not available")

        output_dir = tmp_path / "thumbs"
        output_dir.mkdir()

        result = extract_thumbnail(
            job_id="dim_test",
            source_video_path=str(video_path),
            video_duration=2.0,
            output_dir=output_dir,
        )
        assert result is not None

        # Verify dimensions via ffprobe
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x",
            str(result),
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
        if probe_result.returncode == 0:
            dims = probe_result.stdout.strip()
            assert dims == "1200x630", f"Expected 1200x630, got {dims}"

    def test_ffmpeg_failure_returns_none(self, tmp_path):
        """When ffmpeg fails, returns None gracefully."""
        # Create a file that is not a valid video
        fake_video = tmp_path / "fake.mp4"
        fake_video.write_text("not a video")

        result = extract_thumbnail(
            job_id="fail_test",
            source_video_path=str(fake_video),
            video_duration=10.0,
            output_dir=tmp_path,
        )
        assert result is None
