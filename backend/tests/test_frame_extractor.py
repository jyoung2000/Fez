"""Tests for frame extraction utilities.

Covers:
  - frame_to_base64 produces valid base64
  - resize_frame_if_needed handles oversized and normal images
  - get_video_metadata with mock FFprobe output
  - FFmpeg command errors raise RuntimeError
"""
import asyncio
import base64
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.services.frame_extractor import (
    frame_to_base64,
    resize_frame_if_needed,
    get_video_metadata,
    MAX_DIMENSION,
)


class TestFrameToBase64(unittest.TestCase):
    """Test base64 encoding of frame images."""

    def setUp(self):
        """Create a small test image."""
        self._tmpdir = tempfile.mkdtemp()
        self._img_path = os.path.join(self._tmpdir, "test.jpg")
        # Create a minimal JPEG using PIL
        try:
            from PIL import Image
            img = Image.new("RGB", (100, 100), color="red")
            img.save(self._img_path, "JPEG")
        except ImportError:
            # If PIL not available, write minimal binary data
            with open(self._img_path, "wb") as f:
                f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_returns_valid_base64(self):
        result = frame_to_base64(self._img_path, skip_resize=True)
        # Should be valid base64
        decoded = base64.b64decode(result)
        self.assertGreater(len(decoded), 0)

    def test_returns_string(self):
        result = frame_to_base64(self._img_path, skip_resize=True)
        self.assertIsInstance(result, str)


class TestResizeFrame(unittest.TestCase):
    """Test frame resizing logic."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_small_image_not_resized(self):
        """Images within MAX_DIMENSION should not be changed."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not available")

        path = os.path.join(self._tmpdir, "small.jpg")
        img = Image.new("RGB", (800, 600), color="blue")
        img.save(path, "JPEG")

        result_path = resize_frame_if_needed(path)
        self.assertEqual(result_path, path)

        # Verify dimensions unchanged
        img_after = Image.open(path)
        self.assertEqual(img_after.size, (800, 600))

    def test_oversized_image_resized(self):
        """Images larger than MAX_DIMENSION should be scaled down."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not available")

        path = os.path.join(self._tmpdir, "big.jpg")
        big_size = MAX_DIMENSION + 500
        img = Image.new("RGB", (big_size, big_size), color="green")
        img.save(path, "JPEG")

        resize_frame_if_needed(path)
        img_after = Image.open(path)
        self.assertLessEqual(max(img_after.size), MAX_DIMENSION)


class TestGetVideoMetadata(unittest.TestCase):
    """Test FFprobe metadata extraction with mocked subprocess."""

    def _make_ffprobe_output(self, duration=120.5, width=1920, height=1080, fps="30000/1001", size=52428800):
        return json.dumps({
            "format": {
                "duration": str(duration),
                "size": str(size),
            },
            "streams": [
                {
                    "codec_type": "video",
                    "width": width,
                    "height": height,
                    "r_frame_rate": fps,
                }
            ],
        }).encode()

    @patch("backend.services.frame_extractor.asyncio.create_subprocess_exec")
    def test_parses_metadata_correctly(self, mock_exec):
        proc = AsyncMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(
            self._make_ffprobe_output(),
            b"",
        ))
        mock_exec.return_value = proc

        result = asyncio.get_event_loop().run_until_complete(
            get_video_metadata("/fake/video.mp4")
        )
        self.assertAlmostEqual(result["duration"], 120.5, places=1)
        self.assertEqual(result["resolution"], "1920x1080")
        self.assertGreater(result["fps"], 0)
        self.assertGreater(result["file_size_mb"], 0)

    @patch("backend.services.frame_extractor.asyncio.create_subprocess_exec")
    def test_ffprobe_failure_raises(self, mock_exec):
        proc = AsyncMock()
        proc.returncode = 1
        proc.communicate = AsyncMock(return_value=(b"", b"Error reading file"))
        mock_exec.return_value = proc

        with self.assertRaises(RuntimeError):
            asyncio.get_event_loop().run_until_complete(
                get_video_metadata("/fake/corrupt.mp4")
            )

    @patch("backend.services.frame_extractor.asyncio.create_subprocess_exec")
    def test_corrupt_file_friendly_message(self, mock_exec):
        proc = AsyncMock()
        proc.returncode = 1
        proc.communicate = AsyncMock(return_value=(b"", b"Invalid data found"))
        mock_exec.return_value = proc

        with self.assertRaises(RuntimeError) as ctx:
            asyncio.get_event_loop().run_until_complete(
                get_video_metadata("/fake/corrupt.mp4")
            )
        self.assertIn("corrupt", str(ctx.exception).lower())

    @patch("backend.services.frame_extractor.asyncio.create_subprocess_exec")
    def test_no_video_stream(self, mock_exec):
        """Audio-only file should return empty resolution and 0 fps."""
        proc = AsyncMock()
        proc.returncode = 0
        output = json.dumps({
            "format": {"duration": "60.0", "size": "1048576"},
            "streams": [{"codec_type": "audio"}],
        }).encode()
        proc.communicate = AsyncMock(return_value=(output, b""))
        mock_exec.return_value = proc

        result = asyncio.get_event_loop().run_until_complete(
            get_video_metadata("/fake/audio.mp3")
        )
        self.assertEqual(result["resolution"], "")
        self.assertEqual(result["fps"], 0.0)


if __name__ == "__main__":
    unittest.main()
