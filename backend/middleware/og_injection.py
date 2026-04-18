"""Crawler-aware Open Graph tag injection.

When a known crawler user agent requests an analysis or SEO page, returns a
minimal HTML stub with OG and Twitter Card tags filled from the database.
Real users get the SPA passed through unchanged.
"""

import logging
import os
import re
from html import escape
from pathlib import Path
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse

logger = logging.getLogger(__name__)

# Hardcoded crawler user agents — case-insensitive substring match
CRAWLER_AGENTS = [
    "slackbot-linkexpanding",
    "slack-imgproxy",
    "twitterbot",
    "facebookexternalhit",
    "facebookcatalog",
    "discordbot",
    "linkedinbot",
    "whatsapp",
    "telegrambot",
    "redditbot",
    "applebot",
    "pinterest",
    "skypeuripreview",
    "embedly",
    "quora link preview",
    "outbrain",
    "vkshare",
    "w3c_validator",
    "googlebot",
]

# Routes that get OG injection. Matches the SPA's analysis and SEO page paths.
ANALYSIS_PATH_RE = re.compile(r"^/analysis/(?P<job_id>[a-zA-Z0-9_-]+)/?$")
SEO_PATH_RE = re.compile(r"^/seo/(?P<job_id>[a-zA-Z0-9_-]+)/(?P<clip_id>\d+)/?$")


def is_crawler(user_agent: str) -> bool:
    """Check if the user agent belongs to a known link preview crawler."""
    if not user_agent:
        return False
    ua_lower = user_agent.lower()
    return any(agent in ua_lower for agent in CRAWLER_AGENTS)


THEME_COLOR = "#0ff1ce"  # ClipAI brand accent — Discord embed stripe color


def render_og_html(
    title: str,
    description: str,
    image_url: str,
    canonical_url: str,
    og_type: str = "website",
    video_url: Optional[str] = None,
    *,
    image_width: int = 1200,
    image_height: int = 630,
    image_type: str = "image/jpeg",
    video_width: int = 0,
    video_height: int = 0,
) -> str:
    """Render the minimal HTML stub with all OG and Twitter Card tags.

    Args:
        image_width/image_height: Actual thumbnail dimensions (read from disk
            via Pillow). Lying about these causes letterboxing on Facebook/LinkedIn.
        image_type: MIME type of the thumbnail (image/jpeg, image/png).
        video_width/video_height: Clip dimensions for og:video and twitter:player.
        video_url: Public mp4 URL. When set, emits og:video and twitter:player
            tags and switches twitter:card to "player" for inline playback.
    """
    title_e = escape(title or "ClipAI")
    desc_e = escape(description or "AI-powered video analysis and reframing")
    image_e = escape(image_url)
    url_e = escape(canonical_url)

    # Video tags — emitted only when a rendered mp4 exists
    video_tags = ""
    twitter_card = "summary_large_image"
    if video_url:
        video_e = escape(video_url)
        vw = video_width or 1080
        vh = video_height or 1920
        video_tags = (
            f'<meta property="og:video" content="{video_e}">\n'
            f'<meta property="og:video:secure_url" content="{video_e}">\n'
            f'<meta property="og:video:type" content="video/mp4">\n'
            f'<meta property="og:video:width" content="{vw}">\n'
            f'<meta property="og:video:height" content="{vh}">\n'
            f'<meta name="twitter:player" content="{video_e}">\n'
            f'<meta name="twitter:player:width" content="{vw}">\n'
            f'<meta name="twitter:player:height" content="{vh}">\n'
            f'<meta name="twitter:player:stream" content="{video_e}">\n'
            f'<meta name="twitter:player:stream:content_type" content="video/mp4">\n'
        )
        twitter_card = "player"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title_e}</title>
<meta name="description" content="{desc_e}">
<meta property="og:title" content="{title_e}">
<meta property="og:description" content="{desc_e}">
<meta property="og:image" content="{image_e}">
<meta property="og:image:secure_url" content="{image_e}">
<meta property="og:image:type" content="{escape(image_type)}">
<meta property="og:image:alt" content="{title_e}">
<meta property="og:image:width" content="{image_width}">
<meta property="og:image:height" content="{image_height}">
<meta property="og:url" content="{url_e}">
<meta property="og:type" content="{escape(og_type)}">
<meta property="og:site_name" content="ClipAI">
<meta name="twitter:card" content="{twitter_card}">
<meta name="twitter:title" content="{title_e}">
<meta name="twitter:description" content="{desc_e}">
<meta name="twitter:image" content="{image_e}">
<meta name="theme-color" content="{THEME_COLOR}">
{video_tags}<link rel="canonical" href="{url_e}">
<meta http-equiv="refresh" content="0; url={url_e}">
</head>
<body>
<p><a href="{url_e}">{title_e}</a></p>
</body>
</html>"""


def _get_base_url() -> str:
    """Read the public base URL from env or config."""
    base = os.environ.get("PUBLIC_BASE_URL", "")
    if not base:
        try:
            from backend.config import settings
            base = getattr(settings, "PUBLIC_BASE_URL", "")
        except Exception:
            pass
    return base.rstrip("/")


def _get_base_url_from_request(request) -> str:
    """Derive the base URL from the incoming request when PUBLIC_BASE_URL is not set."""
    base = _get_base_url()
    if base:
        return base
    # Fall back to deriving from the request
    scheme = request.url.scheme or "http"
    host = request.headers.get("host") or request.url.netloc
    if host:
        return f"{scheme}://{host}"
    return ""


def _find_clip_mp4(job_id: str, clip_id: int) -> Optional[str]:
    """Find the rendered mp4 for a clip under /data/outputs/{job_id}/clips/.

    Returns the filename (relative to job output dir) or None if no mp4 exists.
    """
    clips_dir = Path(f"/data/outputs/{job_id}/clips")
    if not clips_dir.is_dir():
        return None
    # Clip files are named like "[1080P] Title.mp4" — scan for any mp4
    # The clip_exporter doesn't embed clip_id in the filename, so we can't
    # match by ID. Return the first mp4 if only one clip, or try matching.
    mp4s = sorted(clips_dir.glob("*.mp4"))
    if not mp4s:
        return None
    # If there's a per-clip index we could use job metadata, but for now
    # try to match by index (clip_id is 0-based or 1-based in the list)
    if clip_id < len(mp4s):
        chosen = mp4s[clip_id]
    else:
        chosen = mp4s[0]
    return f"clips/{chosen.name}"


def _get_thumbnail_dimensions(thumb_path: str) -> tuple[int, int]:
    """Read actual thumbnail dimensions via Pillow. Returns (width, height)."""
    try:
        from PIL import Image
        with Image.open(thumb_path) as img:
            return img.size  # (width, height)
    except Exception:
        return (1200, 630)  # safe default


def _resolve_clip_metadata(clip, job, job_id: str, clip_id: int, base_url: str) -> dict:
    """Build title, description, image_url, video_url, and image dimensions for a clip.

    Returns a dict with keys: title, description, image_url, canonical_url,
    video_url, image_width, image_height, video_width, video_height.
    """
    from backend.services.thumbnail_extractor import get_clip_thumbnail_path, get_thumbnail_dir

    # Title cascade: seo_title → title → "Filename — Clip N" → fallback
    title = None
    if clip:
        title = getattr(clip, 'seo_title', None) or getattr(clip, 'title', None)
    if not title:
        fname = getattr(job, 'filename', None)
        title = f"{fname} — Clip {clip_id}" if fname else "ClipAI Clip"
    title = title[:70]  # Twitter truncates at ~70 chars

    # Description cascade: seo_description → summary → transcript[:200] → fallback
    description = None
    if clip:
        description = getattr(clip, 'seo_description', None)
        if not description:
            description = getattr(clip, 'summary', None)
        if not description:
            transcript = getattr(clip, 'transcript', None)
            if transcript and isinstance(transcript, str):
                description = transcript[:200]
    if not description:
        description = "AI-reframed video clip"
    description = description[:160]

    # Per-clip thumbnail → job-level → default
    clip_thumb = get_clip_thumbnail_path(job_id, clip_id)
    if clip_thumb.exists():
        image_url = f"{base_url}/thumbnails/{job_id}/{clip_id}.jpg"
        iw, ih = _get_thumbnail_dimensions(str(clip_thumb))
    else:
        job_thumb = get_thumbnail_dir() / f"{job_id}.jpg"
        if job_thumb.exists():
            image_url = f"{base_url}/thumbnails/{job_id}.jpg"
            iw, ih = _get_thumbnail_dimensions(str(job_thumb))
        else:
            image_url = f"{base_url}/thumbnails/{job_id}.jpg"
            iw, ih = 1200, 630

    canonical_url = f"{base_url}/seo/{job_id}/{clip_id}"

    # Video URL — only if a rendered mp4 exists
    video_url = None
    video_width = 0
    video_height = 0
    clip_mp4 = _find_clip_mp4(job_id, clip_id)
    if clip_mp4:
        video_url = f"{base_url}/api/files/{job_id}/{clip_mp4}"
        video_width = 1080
        video_height = 1920

    return {
        "title": title,
        "description": description,
        "image_url": image_url,
        "canonical_url": canonical_url,
        "video_url": video_url,
        "image_width": iw,
        "image_height": ih,
        "video_width": video_width,
        "video_height": video_height,
    }


class OGInjectionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        ua = request.headers.get("user-agent", "")

        if not is_crawler(ua):
            return await call_next(request)

        path = request.url.path

        analysis_match = ANALYSIS_PATH_RE.match(path)
        seo_match = SEO_PATH_RE.match(path)

        if not analysis_match and not seo_match:
            return await call_next(request)

        # Crawler hit on an analysis or SEO route — fetch metadata and render stub
        try:
            from backend import database

            base_url = _get_base_url_from_request(request)
            if not base_url:
                logger.warning("OGInjection: cannot determine base URL, falling through")
                return await call_next(request)

            if analysis_match:
                job_id = analysis_match.group("job_id")
                job = await database.get_job(job_id)
                if not job:
                    return await call_next(request)
                title = (getattr(job, 'filename', None)
                         or f"ClipAI Analysis {job_id[:8]}")
                summary = getattr(job, 'summary', None)
                if summary and hasattr(summary, 'overview'):
                    description = summary.overview
                else:
                    description = "AI-powered video analysis"
                image_url = f"{base_url}/thumbnails/{job_id}.jpg"
                canonical_url = f"{base_url}/analysis/{job_id}"
                og_type = "video.other"
            else:
                job_id = seo_match.group("job_id")
                clip_id = int(seo_match.group("clip_id"))
                job = await database.get_job(job_id)
                if not job:
                    return await call_next(request)
                clip = None
                for c in getattr(job, 'clips', []):
                    if getattr(c, 'id', None) == clip_id:
                        clip = c
                        break
                meta = _resolve_clip_metadata(clip, job, job_id, clip_id, base_url)
                title = meta["title"]
                description = meta["description"]
                image_url = meta["image_url"]
                canonical_url = meta["canonical_url"]
                og_type = "video.other"

            html = render_og_html(
                title=title,
                description=description,
                image_url=image_url,
                canonical_url=canonical_url,
                og_type=og_type,
                video_url=meta["video_url"] if seo_match else None,
                image_width=meta["image_width"] if seo_match else 1200,
                image_height=meta["image_height"] if seo_match else 630,
                video_width=meta.get("video_width", 0) if seo_match else 0,
                video_height=meta.get("video_height", 0) if seo_match else 0,
            )
            logger.info("OGInjection: served crawler stub for ua=%s path=%s",
                        ua[:50], path)
            return HTMLResponse(content=html, status_code=200)
        except Exception as e:
            logger.warning("OGInjection: failed for path=%s ua=%s: %s",
                           path, ua[:50], e)
            return await call_next(request)
