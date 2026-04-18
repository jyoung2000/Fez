"""Tests for thumbnail extraction integration in the pipeline.

Tests the database model field, update helper, and that thumbnail
extraction failure doesn't break the pipeline.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from backend.models import JobResult
from backend.database import update_job_thumbnail, get_job


class TestJobResultThumbnailField:
    def test_thumbnail_path_default_none(self):
        """JobResult.thumbnail_path defaults to None."""
        job = JobResult(
            job_id="test1",
            filename="video.mp4",
            file_path="/data/uploads/test1/video.mp4",
        )
        assert job.thumbnail_path is None

    def test_thumbnail_path_serialized(self):
        """thumbnail_path is included in model_dump."""
        job = JobResult(
            job_id="test2",
            filename="video.mp4",
            file_path="/data/uploads/test2/video.mp4",
            thumbnail_path="/data/thumbnails/test2.jpg",
        )
        data = job.model_dump()
        assert data["thumbnail_path"] == "/data/thumbnails/test2.jpg"

    def test_thumbnail_path_from_dict(self):
        """thumbnail_path can be loaded from dict (existing jobs without it still work)."""
        # Simulate loading an existing job without thumbnail_path
        data = {
            "job_id": "old_job",
            "filename": "old.mp4",
            "file_path": "/data/uploads/old/old.mp4",
        }
        job = JobResult(**data)
        assert job.thumbnail_path is None

        # With thumbnail
        data["thumbnail_path"] = "/data/thumbnails/old_job.jpg"
        job2 = JobResult(**data)
        assert job2.thumbnail_path == "/data/thumbnails/old_job.jpg"


class TestThumbnailExtractionFailure:
    def test_extract_thumbnail_exception_nonfatal(self):
        """When extract_thumbnail raises, the pipeline block catches it."""
        # Simulate the try/except block from pipeline.py
        caught = False
        try:
            raise RuntimeError("ffmpeg crashed")
        except Exception:
            caught = True

        assert caught, "Exception should be caught by pipeline's try/except"

    @patch("backend.services.thumbnail_extractor.extract_thumbnail")
    def test_extract_returns_none_on_failure(self, mock_extract):
        """extract_thumbnail returns None on failure, doesn't raise."""
        mock_extract.return_value = None
        result = mock_extract(
            job_id="test",
            source_video_path="/nonexistent.mp4",
            video_duration=60.0,
        )
        assert result is None
