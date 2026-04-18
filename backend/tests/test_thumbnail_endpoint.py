"""Tests for the public thumbnail serving endpoint."""

import os
import tempfile
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.routers.thumbnails import router


# Use FastAPI's test client
try:
    from fastapi.testclient import TestClient
    from fastapi import FastAPI

    _app = FastAPI()
    _app.include_router(router)
    client = TestClient(_app)
    _HAS_TESTCLIENT = True
except ImportError:
    _HAS_TESTCLIENT = False


@pytest.mark.skipif(not _HAS_TESTCLIENT, reason="fastapi/httpx not available")
class TestThumbnailEndpoint:
    def test_valid_thumbnail_returns_200(self, tmp_path):
        """GET /thumbnails/{job_id}.jpg returns 200 with jpeg content type."""
        thumb_dir = tmp_path / "thumbs"
        thumb_dir.mkdir()
        # Create a dummy jpeg
        thumb_file = thumb_dir / "test_job.jpg"
        thumb_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 2000)

        with patch("backend.routers.thumbnails.get_thumbnail_dir", return_value=thumb_dir):
            resp = client.get("/thumbnails/test_job.jpg")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"

    def test_cache_control_header(self, tmp_path):
        """Response includes Cache-Control: public, max-age=2592000."""
        thumb_dir = tmp_path / "thumbs"
        thumb_dir.mkdir()
        thumb_file = thumb_dir / "cache_test.jpg"
        thumb_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 2000)

        with patch("backend.routers.thumbnails.get_thumbnail_dir", return_value=thumb_dir):
            resp = client.get("/thumbnails/cache_test.jpg")
        assert "public, max-age=2592000" in resp.headers.get("cache-control", "")

    def test_conditional_get_304(self, tmp_path):
        """Returns 304 on conditional GET with current Last-Modified."""
        thumb_dir = tmp_path / "thumbs"
        thumb_dir.mkdir()
        thumb_file = thumb_dir / "cond_test.jpg"
        thumb_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 2000)

        # Get the file's mtime
        mtime = datetime.fromtimestamp(thumb_file.stat().st_mtime, tz=timezone.utc)
        # Use a future date to ensure 304
        from datetime import timedelta
        future = mtime + timedelta(hours=1)
        ims_header = format_datetime(future, usegmt=True)

        with patch("backend.routers.thumbnails.get_thumbnail_dir", return_value=thumb_dir):
            resp = client.get(
                "/thumbnails/cond_test.jpg",
                headers={"If-Modified-Since": ims_header},
            )
        assert resp.status_code == 304

    def test_missing_thumbnail_returns_default_or_404(self, tmp_path):
        """Returns default placeholder when thumbnail is missing, or 404 if no default."""
        thumb_dir = tmp_path / "thumbs"
        thumb_dir.mkdir()

        with patch("backend.routers.thumbnails.get_thumbnail_dir", return_value=thumb_dir):
            resp = client.get("/thumbnails/nonexistent.jpg")
        # Should return 200 (default) or 404 (no default)
        assert resp.status_code in (200, 404)

    def test_path_traversal_returns_safe(self):
        """Path traversal attempts don't serve arbitrary files."""
        # URL-encoded traversal gets decoded by FastAPI, but our sanitization
        # catches non-alphanumeric characters in job_id
        resp = client.get("/thumbnails/../../etc/passwd.jpg")
        assert resp.status_code in (400, 404, 422)

    def test_invalid_job_id_returns_safe(self):
        """Job IDs with special characters don't serve files."""
        # Spaces and semicolons make FastAPI route matching fail (404)
        # or our validation catches them (400)
        resp = client.get("/thumbnails/test%3Becho.jpg")
        assert resp.status_code in (400, 404)

    def test_empty_job_id_returns_400(self):
        """Empty job_id in the path returns 400 or 404."""
        resp = client.get("/thumbnails/.jpg")
        # FastAPI may not match this route at all, or match with empty string
        assert resp.status_code in (400, 404, 422)

    def test_last_modified_header_present(self, tmp_path):
        """Response includes Last-Modified header."""
        thumb_dir = tmp_path / "thumbs"
        thumb_dir.mkdir()
        thumb_file = thumb_dir / "lm_test.jpg"
        thumb_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 2000)

        with patch("backend.routers.thumbnails.get_thumbnail_dir", return_value=thumb_dir):
            resp = client.get("/thumbnails/lm_test.jpg")
        assert "last-modified" in resp.headers
