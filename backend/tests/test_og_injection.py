"""Tests for the crawler-aware OG injection middleware."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.middleware.og_injection import (
    is_crawler,
    render_og_html,
    CRAWLER_AGENTS,
    ANALYSIS_PATH_RE,
    SEO_PATH_RE,
    THEME_COLOR,
)


class TestIsCrawler:
    def test_slackbot(self):
        assert is_crawler("Slackbot-LinkExpanding 1.0 (+https://api.slack.com/robots)") is True

    def test_twitterbot(self):
        assert is_crawler("Twitterbot/1.0") is True

    def test_facebookbot(self):
        assert is_crawler("facebookexternalhit/1.1") is True

    def test_discordbot(self):
        assert is_crawler("Mozilla/5.0 (compatible; Discordbot/2.0; +https://discordapp.com)") is True

    def test_regular_browser(self):
        assert is_crawler("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36") is False

    def test_empty_ua(self):
        assert is_crawler("") is False

    def test_none_ua(self):
        assert is_crawler(None) is False

    def test_whatsapp(self):
        assert is_crawler("WhatsApp/2.21.4.22") is True

    def test_linkedinbot(self):
        assert is_crawler("LinkedInBot/1.0") is True

    def test_telegrambot(self):
        assert is_crawler("TelegramBot (like TwitterBot)") is True

    def test_applebot(self):
        assert is_crawler("Applebot/0.1 +http://www.apple.com/go/applebot") is True

    def test_googlebot(self):
        assert is_crawler("Googlebot/2.1") is True


class TestRenderOGHtml:
    def test_contains_all_og_tags(self):
        html = render_og_html(
            title="Test Video",
            description="A test video description",
            image_url="https://example.com/thumbnails/abc.jpg",
            canonical_url="https://example.com/analysis/abc",
        )
        assert 'og:title' in html
        assert 'og:description' in html
        assert 'og:image' in html
        assert 'og:url' in html
        assert 'og:type' in html
        assert 'og:site_name' in html
        assert 'og:image:secure_url' in html
        assert 'og:image:type' in html
        assert 'og:image:alt' in html

    def test_contains_twitter_card_tags(self):
        html = render_og_html(
            title="Test",
            description="Desc",
            image_url="https://example.com/thumb.jpg",
            canonical_url="https://example.com/analysis/x",
        )
        assert 'twitter:card' in html
        assert 'summary_large_image' in html
        assert 'twitter:title' in html
        assert 'twitter:image' in html

    def test_xss_protection(self):
        """HTML-escapes titles to prevent XSS."""
        html = render_og_html(
            title='<script>alert("xss")</script>',
            description='<img src=x onerror=alert(1)>',
            image_url="https://example.com/thumb.jpg",
            canonical_url="https://example.com/analysis/x",
        )
        assert '<script>' not in html
        assert '&lt;script&gt;' in html
        assert '<img src=x' not in html

    def test_contains_meta_refresh(self):
        """HTML includes meta refresh redirect for humans hitting the URL."""
        html = render_og_html(
            title="Test",
            description="Desc",
            image_url="https://example.com/thumb.jpg",
            canonical_url="https://example.com/analysis/x",
        )
        assert 'http-equiv="refresh"' in html

    def test_image_dimensions_from_params(self):
        html = render_og_html(
            title="Test",
            description="Desc",
            image_url="https://example.com/thumb.jpg",
            canonical_url="https://example.com/test",
            image_width=1080,
            image_height=1920,
        )
        assert 'og:image:width' in html
        assert '1080' in html
        assert 'og:image:height' in html
        assert '1920' in html

    def test_video_tags_full_when_video_url_provided(self):
        html = render_og_html(
            title="Test",
            description="Desc",
            image_url="https://example.com/thumb.jpg",
            canonical_url="https://example.com/test",
            video_url="https://example.com/video.mp4",
            video_width=1080,
            video_height=1920,
        )
        assert 'og:video' in html
        assert 'og:video:secure_url' in html
        assert 'og:video:type' in html
        assert 'og:video:width' in html
        assert 'og:video:height' in html
        assert 'twitter:player' in html
        assert 'twitter:player:width' in html
        assert 'twitter:player:height' in html
        assert 'twitter:player:stream' in html
        assert 'twitter:player:stream:content_type' in html
        # twitter:card switches to "player"
        assert 'content="player"' in html
        assert 'summary_large_image' not in html

    def test_no_video_tags_when_no_video_url(self):
        html = render_og_html(
            title="Test",
            description="Desc",
            image_url="https://example.com/thumb.jpg",
            canonical_url="https://example.com/test",
        )
        assert 'og:video' not in html
        assert 'content="summary_large_image"' in html

    def test_theme_color_present(self):
        html = render_og_html(
            title="Test",
            description="Desc",
            image_url="https://example.com/thumb.jpg",
            canonical_url="https://example.com/test",
        )
        assert 'theme-color' in html
        assert THEME_COLOR in html


class TestPathRegex:
    def test_analysis_path_matches(self):
        m = ANALYSIS_PATH_RE.match("/analysis/abc123-def")
        assert m is not None
        assert m.group("job_id") == "abc123-def"

    def test_analysis_path_with_trailing_slash(self):
        m = ANALYSIS_PATH_RE.match("/analysis/abc123/")
        assert m is not None

    def test_analysis_path_no_match_api(self):
        m = ANALYSIS_PATH_RE.match("/api/analysis/abc123")
        assert m is None

    def test_seo_path_matches(self):
        m = SEO_PATH_RE.match("/seo/job123/5")
        assert m is not None
        assert m.group("job_id") == "job123"
        assert m.group("clip_id") == "5"

    def test_seo_path_no_match_without_clip_id(self):
        m = SEO_PATH_RE.match("/seo/job123")
        assert m is None


class TestOGMiddleware:
    """Integration tests for the middleware using FastAPI TestClient."""

    @pytest.fixture
    def app_client(self):
        try:
            import sys
            # Ensure backend.database is importable even without full DB deps
            if "backend.database" not in sys.modules:
                mock_db = MagicMock()
                mock_db.get_job = AsyncMock(return_value=None)
                sys.modules.setdefault("backend.database", mock_db)

            from fastapi import FastAPI
            from fastapi.testclient import TestClient
            from backend.middleware.og_injection import OGInjectionMiddleware

            app = FastAPI()
            app.add_middleware(OGInjectionMiddleware)

            @app.get("/analysis/{job_id}")
            async def analysis(job_id: str):
                return {"page": "spa", "job_id": job_id}

            @app.get("/seo/{job_id}/{clip_id}")
            async def seo(job_id: str, clip_id: int):
                return {"page": "spa", "job_id": job_id, "clip_id": clip_id}

            return TestClient(app)
        except ImportError:
            pytest.skip("fastapi/httpx not available")

    def test_browser_passes_through(self, app_client):
        """Regular browser gets the SPA response, not the OG stub."""
        resp = app_client.get(
            "/analysis/test123",
            headers={"User-Agent": "Mozilla/5.0 (Macintosh)"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["page"] == "spa"

    def test_crawler_gets_og_stub(self, app_client):
        """Crawler gets HTML with OG tags when PUBLIC_BASE_URL is set."""
        mock_job = MagicMock()
        mock_job.filename = "test_video.mp4"
        mock_job.summary = MagicMock()
        mock_job.summary.overview = "A test description"

        with patch.dict("os.environ", {"PUBLIC_BASE_URL": "https://example.com"}):
            with patch("backend.database.get_job", new_callable=AsyncMock, return_value=mock_job):
                resp = app_client.get(
                    "/analysis/test123",
                    headers={"User-Agent": "Slackbot-LinkExpanding 1.0"},
                )

        assert resp.status_code == 200
        assert "og:image" in resp.text
        assert "https://example.com/thumbnails/test123.jpg" in resp.text

    def test_crawler_falls_through_when_no_base_url(self, app_client):
        """Without PUBLIC_BASE_URL, crawler gets the SPA response."""
        with patch.dict("os.environ", {"PUBLIC_BASE_URL": ""}, clear=False):
            resp = app_client.get(
                "/analysis/test123",
                headers={"User-Agent": "Slackbot-LinkExpanding 1.0"},
            )
        # Falls through to the SPA handler
        assert resp.status_code == 200
        data = resp.json()
        assert data["page"] == "spa"

    def test_crawler_falls_through_when_job_not_found(self, app_client):
        """When job doesn't exist, crawler gets the SPA response."""
        with patch.dict("os.environ", {"PUBLIC_BASE_URL": "https://example.com"}):
            with patch("backend.database.get_job", new_callable=AsyncMock, return_value=None):
                resp = app_client.get(
                    "/analysis/nonexistent",
                    headers={"User-Agent": "Slackbot-LinkExpanding 1.0"},
                )
        assert resp.status_code == 200
        data = resp.json()
        assert data["page"] == "spa"

    def test_seo_crawler_gets_og_with_https(self, app_client):
        """Each crawler UA on /seo/{job_id}/{clip_id} gets og:image with https://."""
        mock_clip = MagicMock()
        mock_clip.id = 2
        mock_clip.seo_title = "Best Clip"
        mock_clip.title = "Best Clip"
        mock_clip.seo_description = "Great moment"
        mock_clip.summary = None
        mock_clip.transcript = None

        mock_job = MagicMock()
        mock_job.filename = "test.mp4"
        mock_job.clips = [mock_clip]

        test_uas = [
            "Twitterbot/1.0",
            "facebookexternalhit/1.1",
            "Mozilla/5.0 (compatible; Discordbot/2.0)",
            "Slackbot-LinkExpanding 1.0",
            "WhatsApp/2.21.4.22",
            "TelegramBot",
            "LinkedInBot/1.0",
        ]

        for ua in test_uas:
            with patch.dict("os.environ", {"PUBLIC_BASE_URL": "https://clipai.example.com"}):
                with patch("backend.database.get_job", new_callable=AsyncMock, return_value=mock_job):
                    resp = app_client.get(
                        "/seo/job123/2",
                        headers={"User-Agent": ua},
                    )
            assert resp.status_code == 200, f"Failed for UA: {ua}"
            assert "og:image" in resp.text, f"Missing og:image for UA: {ua}"
            assert "https://" in resp.text, f"Missing https:// in og:image for UA: {ua}"

    def test_seo_browser_passes_through(self, app_client):
        """Normal browser on /seo path gets SPA, not OG stub."""
        resp = app_client.get(
            "/seo/job123/2",
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["page"] == "spa"
