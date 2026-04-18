"""Allowed video MIME types for cloud browse/search results.

We filter *server-side* so users never see documents or images in the
cloud picker. The frontend repeats the check as defense-in-depth.

The set mirrors the extensions accepted by the local upload path
(``backend/services/ingest.py::ALLOWED_EXTENSIONS``) plus a handful of
provider-native video types (Google encodes certain YouTube-backed files
with ``application/vnd.google-apps.*`` sentinels).
"""

from __future__ import annotations

# Standard MIME types that map 1:1 with our accepted upload extensions.
VIDEO_MIME_TYPES: frozenset[str] = frozenset(
    {
        "video/mp4",
        "video/quicktime",           # .mov
        "video/x-matroska",          # .mkv
        "video/webm",
        "video/x-msvideo",            # .avi
        "video/x-m4v",
        "video/3gpp",
        "video/mpeg",
        "video/ogg",
    }
)

# Google Drive "native" types. Google sometimes wraps imported videos with
# a Drive-specific MIME, but for our purposes we still treat any type
# starting with ``video/`` as a candidate, so these are only used to
# explicitly allow Google's own video app exports.
GOOGLE_NATIVE_VIDEO_TYPES: frozenset[str] = frozenset(
    {
        "application/vnd.google-apps.video",
    }
)

# Box exposes extensions directly on items; we use this list for
# ``GET /search?file_extensions=`` which takes a CSV of extensions.
BOX_SEARCH_EXTENSIONS: tuple[str, ...] = (
    "mp4",
    "mov",
    "mkv",
    "webm",
    "avi",
    "m4v",
    "3gp",
    "mpeg",
    "mpg",
)


def is_video_mime(mime: str) -> bool:
    """Return True if *mime* is a video type we can ingest."""
    if not mime:
        return False
    mime = mime.lower()
    if mime.startswith("video/"):
        return True
    return mime in GOOGLE_NATIVE_VIDEO_TYPES


def is_video_extension(name: str) -> bool:
    """Defensive fallback for providers that don't return a MIME."""
    if not name or "." not in name:
        return False
    ext = name.rsplit(".", 1)[-1].lower()
    return ext in {"mp4", "mov", "mkv", "webm", "avi", "m4v", "3gp", "mpeg", "mpg"}
