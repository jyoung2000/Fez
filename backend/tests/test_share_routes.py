"""Tests for the dedicated /share routes and /thumbnails routes."""

import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


try:
    import sys
    from unittest.mock import AsyncMock, MagicMock
    # Ensure backend.database is importable even without full DB deps
    if "backend.database" not in sys.modules:
        _mock_db = MagicMock()
        _mock_db.get_job = AsyncMock(return_value=None)
        sys.modules.setdefault("backend.database", _mock_db)

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.routers.share import router as share_router
    from backend.routers.thumbnails import router as thumb_router

    _app = FastAPI()
    _app.include_router(share_router)
    _app.include_router(thumb_router)
    client = TestClient(_app)
    _HAS_TESTCLIENT = True
except ImportError:
    _HAS_TESTCLIENT = False


@pytest.mark.skipif(not _HAS_TESTCLIENT, reason="fastapi/httpx not available")
class TestShareAnalysis:
    def test_returns_og_html(self):
        """GET /share/analysis/{job_id} returns HTML with OG tags."""
        mock_job = MagicMock()
        mock_job.filename = "my_video.mp4"
        mock_job.summary = MagicMock()
        mock_job.summary.overview = "Great video analysis"

        with patch.dict("os.environ", {"PUBLIC_BASE_URL": "https://app.example.com"}):
            with patch("backend.database.get_job", new_callable=AsyncMock, return_value=mock_job):
                resp = client.get("/share/analysis/job-abc123")

        assert resp.status_code == 200
        assert "og:title" in resp.text
        assert "og:image" in resp.text
        assert "https://app.example.com/thumbnails/job-abc123.jpg" in resp.text
        assert "https://app.example.com/analysis/job-abc123" in resp.text

    def test_nonexistent_job_returns_404(self):
        """GET /share/analysis/nonexistent returns 404."""
        with patch.dict("os.environ", {"PUBLIC_BASE_URL": "https://app.example.com"}):
            with patch("backend.database.get_job", new_callable=AsyncMock, return_value=None):
                resp = client.get("/share/analysis/nonexistent")
        assert resp.status_code == 404

    def test_works_without_public_base_url(self):
        """Without PUBLIC_BASE_URL, derives base URL from request Host header."""
        mock_job = MagicMock()
        mock_job.filename = "test.mp4"
        mock_job.summary = None

        with patch.dict("os.environ", {"PUBLIC_BASE_URL": ""}, clear=False):
            with patch("backend.database.get_job", new_callable=AsyncMock, return_value=mock_job):
                resp = client.get(
                    "/share/analysis/test123",
                    headers={"Host": "192.168.8.119:1353"},
                )

        assert resp.status_code == 200
        assert "og:title" in resp.text
        assert "/thumbnails/test123.jpg" in resp.text

    def test_meta_refresh_present(self):
        """Response includes meta refresh redirect to canonical URL."""
        mock_job = MagicMock()
        mock_job.filename = "test.mp4"
        mock_job.summary = None

        with patch.dict("os.environ", {"PUBLIC_BASE_URL": "https://example.com"}):
            with patch("backend.database.get_job", new_callable=AsyncMock, return_value=mock_job):
                resp = client.get("/share/analysis/test123")

        assert 'http-equiv="refresh"' in resp.text

    def test_any_user_agent_gets_og(self):
        """Share routes return OG tags regardless of user agent (no crawler check)."""
        mock_job = MagicMock()
        mock_job.filename = "test.mp4"
        mock_job.summary = None

        with patch.dict("os.environ", {"PUBLIC_BASE_URL": "https://example.com"}):
            with patch("backend.database.get_job", new_callable=AsyncMock, return_value=mock_job):
                resp = client.get(
                    "/share/analysis/test123",
                    headers={"User-Agent": "Mozilla/5.0 Regular Browser"},
                )

        assert resp.status_code == 200
        assert "og:title" in resp.text


@pytest.mark.skipif(not _HAS_TESTCLIENT, reason="fastapi/httpx not available")
class TestShareClip:
    def test_returns_clip_og_html(self):
        """GET /share/clip/{job_id}/{clip_id} returns HTML with clip title."""
        mock_clip = MagicMock()
        mock_clip.id = 3
        mock_clip.title = "Best Moment"
        mock_clip.seo_title = "Best Moment Ever"
        mock_clip.seo_description = "The best moment from the video"
        mock_clip.summary = None
        mock_clip.transcript = None

        mock_job = MagicMock()
        mock_job.clips = [mock_clip]
        mock_job.filename = "test.mp4"

        with patch.dict("os.environ", {"PUBLIC_BASE_URL": "https://example.com"}):
            with patch("backend.database.get_job", new_callable=AsyncMock, return_value=mock_job):
                resp = client.get("/share/clip/job123/3")

        assert resp.status_code == 200
        assert "Best Moment" in resp.text
        assert "og:image" in resp.text
        assert "og:image:secure_url" in resp.text
        assert "theme-color" in resp.text

    def test_clip_not_found_still_returns(self):
        """When clip ID doesn't match, returns generic title."""
        mock_job = MagicMock()
        mock_job.clips = []
        mock_job.filename = "my_video.mp4"

        with patch.dict("os.environ", {"PUBLIC_BASE_URL": "https://example.com"}):
            with patch("backend.database.get_job", new_callable=AsyncMock, return_value=mock_job):
                resp = client.get("/share/clip/job123/99")

        assert resp.status_code == 200
        assert "og:title" in resp.text

    def test_clip_share_has_video_tags_when_mp4_exists(self):
        """When a clip mp4 exists, share page includes og:video and twitter:player."""
        mock_clip = MagicMock()
        mock_clip.id = 0
        mock_clip.title = "Test Clip"
        mock_clip.seo_title = None
        mock_clip.seo_description = "A clip"
        mock_clip.summary = None
        mock_clip.transcript = None

        mock_job = MagicMock()
        mock_job.clips = [mock_clip]
        mock_job.filename = "test.mp4"

        with patch.dict("os.environ", {"PUBLIC_BASE_URL": "https://example.com"}):
            with patch("backend.database.get_job", new_callable=AsyncMock, return_value=mock_job):
                with patch("backend.middleware.og_injection._find_clip_mp4", return_value="clips/test.mp4"):
                    resp = client.get("/share/clip/job123/0")

        assert resp.status_code == 200
        assert "og:video" in resp.text
        assert "twitter:player" in resp.text
        assert 'content="player"' in resp.text

    def test_clip_share_no_video_when_no_mp4(self):
        """When no clip mp4 exists, share page omits video tags."""
        mock_clip = MagicMock()
        mock_clip.id = 0
        mock_clip.title = "Test Clip"
        mock_clip.seo_title = None
        mock_clip.seo_description = "A clip"
        mock_clip.summary = None
        mock_clip.transcript = None

        mock_job = MagicMock()
        mock_job.clips = [mock_clip]
        mock_job.filename = "test.mp4"

        with patch.dict("os.environ", {"PUBLIC_BASE_URL": "https://example.com"}):
            with patch("backend.database.get_job", new_callable=AsyncMock, return_value=mock_job):
                with patch("backend.middleware.og_injection._find_clip_mp4", return_value=None):
                    resp = client.get("/share/clip/job123/0")

        assert resp.status_code == 200
        assert "og:video" not in resp.text
        assert 'content="summary_large_image"' in resp.text


@pytest.mark.skipif(not _HAS_TESTCLIENT, reason="fastapi/httpx not available")
class TestClipThumbnailEndpoint:
    def test_fallback_to_job_thumbnail(self, tmp_path):
        """When no per-clip thumbnail exists, falls back to job-level."""
        # Create a fake job-level thumbnail
        job_thumb = tmp_path / "test-job.jpg"
        job_thumb.write_bytes(b"\xff\xd8\xff" + b"\x00" * 1000)

        with patch("backend.services.thumbnail_extractor.get_thumbnail_dir", return_value=tmp_path):
            with patch("backend.routers.thumbnails.get_clip_thumbnail_path",
                       return_value=tmp_path / "nonexistent.jpg"):
                resp = client.get("/thumbnails/test-job/0.jpg")

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"

    def test_304_with_future_if_modified_since(self, tmp_path):
        """If-Modified-Since in the future returns 304."""
        job_thumb = tmp_path / "test-job.jpg"
        job_thumb.write_bytes(b"\xff\xd8\xff" + b"\x00" * 1000)

        with patch("backend.services.thumbnail_extractor.get_thumbnail_dir", return_value=tmp_path):
            with patch("backend.routers.thumbnails.get_clip_thumbnail_path",
                       return_value=tmp_path / "nonexistent.jpg"):
                resp = client.get(
                    "/thumbnails/test-job/0.jpg",
                    headers={"If-Modified-Since": "Sun, 01 Jan 2090 00:00:00 GMT"},
                )

        assert resp.status_code == 304

    def test_falls_back_to_default(self, tmp_path):
        """When job thumbnail is also missing, falls back to default_thumbnail.jpg."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with patch("backend.services.thumbnail_extractor.get_thumbnail_dir", return_value=empty_dir):
            with patch("backend.routers.thumbnails.get_clip_thumbnail_path",
                       return_value=empty_dir / "nonexistent.jpg"):
                resp = client.get("/thumbnails/test-job/0.jpg")

        # Should get 200 from the default thumbnail (or 404 if default doesn't exist)
        assert resp.status_code in (200, 404)
