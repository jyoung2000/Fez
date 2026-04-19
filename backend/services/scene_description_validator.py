"""Validates and sanitizes VLM-returned scene descriptions.

Every VLM provider eventually funnels untrusted model output into
`SceneDescription.description`. This module is the last line of
defense before that string is persisted and rendered to the user.

Two public entry points:

- `sanitize_description(raw)` cleans benign formatting issues
  (whitespace, control chars, code fences, trailing JSON fragments)
  and returns a presentable string, or `""` if nothing survives.

- `is_valid_description(desc)` returns `(ok, reason)` — True only when
  the string reads as natural-language prose and doesn't match any
  known failure pattern (dict reprs, JSON leaks, refusals, errors).
"""
from __future__ import annotations
import re
from typing import Optional

_JUNK_START_PATTERNS = (
    re.compile(r"^\s*[\{\[]"),
    re.compile(r"^\s*['\"]?(raw_response|error|description|importance_score|subject_box|subject_x|traceback|stack)['\"]?\s*[:=]", re.IGNORECASE),
    re.compile(r"^\s*(Error|Traceback|Exception|Failed to|RuntimeError|undefined|null|NaN|<html|<!DOCTYPE)\b", re.IGNORECASE),
)

_REFUSAL_PATTERNS = (
    re.compile(r"^\s*(?:I can'?t|I cannot|I am unable|I'm unable|I am not able|I don'?t see)\b", re.IGNORECASE),
    re.compile(r"^\s*(As an AI|Sorry,|I'm sorry|Unfortunately, I)", re.IGNORECASE),
    re.compile(r"^\s*\[?image (unavailable|not (loaded|provided|visible))\]?", re.IGNORECASE),
    re.compile(r"^\s*(no image|image is blank|image appears (blank|empty))", re.IGNORECASE),
)

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]")
_CODE_FENCE_RE = re.compile(r"^```[a-z]*\s*\n(.*?)\n?```\s*$", re.DOTALL)
_MARKDOWN_JSON_INLINE = re.compile(r"^`[\{\[].*[\}\]]`$", re.DOTALL)

_MIN_WORDS = 3
_MIN_CHARS = 10
_MAX_CHARS = 400

# Scenes produced by our own fallback synthesizers — these ARE valid.
# Keep in sync with scene_fallback.py and fallback_description_for_scene.
_SYNTHETIC_PREFIXES = (
    "key moment ",
    "video frame at ",
)


def sanitize_description(raw: Optional[str]) -> str:
    """Clean a description for display. Returns '' if unsalvageable."""
    if not raw or not isinstance(raw, str):
        return ""
    text = raw.strip()
    if not text:
        return ""
    m = _CODE_FENCE_RE.match(text)
    if m:
        inner = m.group(1).strip()
        if inner.startswith(("{", "[")):
            return ""
        text = inner
    if _MARKDOWN_JSON_INLINE.match(text):
        return ""
    for prefix in ("Description: ", "Scene: ", "This frame shows ", "In this frame, "):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):]
            break
    text = _CONTROL_CHAR_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.rstrip(' "\',;:}])')
    text = text.lstrip(' "\',;:{[(')
    if len(text) > _MAX_CHARS:
        cut = text.rfind(". ", 0, _MAX_CHARS - 1)
        text = text[: cut + 1] if cut > _MAX_CHARS // 2 else text[: _MAX_CHARS - 1] + "\u2026"
    return text


def is_valid_description(desc: Optional[str]) -> tuple[bool, str]:
    """Return (is_valid, reason). Treats synthesized markers as valid."""
    if not desc or not isinstance(desc, str):
        return False, "empty"
    text = desc.strip()
    if not text:
        return False, "whitespace_only"
    low = text.lower()
    if any(low.startswith(p) for p in _SYNTHETIC_PREFIXES):
        return True, "synthetic_ok"
    # Check junk / refusal patterns BEFORE length checks so callers get
    # the most descriptive reason (a dict repr is "junk_prefix", not
    # "too_few_words", even though both technically apply).
    for rx in _JUNK_START_PATTERNS:
        if rx.search(text):
            return False, "junk_prefix"
    for rx in _REFUSAL_PATTERNS:
        if rx.search(text):
            return False, "refusal"
    if len(text) < _MIN_CHARS:
        return False, "too_short"
    if len(text.split()) < _MIN_WORDS:
        return False, "too_few_words"
    non_alpha = sum(1 for c in text if not (c.isalnum() or c in " .,;:!?'-\u2026\u2013\u2014"))
    if non_alpha / max(1, len(text)) > 0.15:
        return False, "high_symbol_ratio"
    return True, "ok"


def fallback_description_for_scene(
    *,
    timestamp: float,
    transcript_segments: list | None = None,
    scene_index: int = 0,
) -> str:
    """Build a clean transcript-aware fallback. Prefer nearby speech;
    fall back to a timestamp-only 'Key moment' if no speech is nearby.

    Kept in sync with scene_fallback._excerpt_transcript but callable
    per-scene rather than per-job.
    """
    excerpt = ""
    if transcript_segments:
        pieces = []
        total = 0
        window_start = max(0.0, timestamp - 4.0)
        window_end = timestamp + 4.0
        for seg in transcript_segments:
            seg_start = float(getattr(seg, "start", 0) or 0)
            seg_end = float(getattr(seg, "end", 0) or 0)
            if seg_end < window_start:
                continue
            if seg_start > window_end:
                break
            text = (getattr(seg, "text", "") or "").strip()
            if not text:
                continue
            pieces.append(text)
            total += len(text) + 1
            if total >= 140:
                break
        excerpt = " ".join(pieces).strip()
        if len(excerpt) > 140:
            excerpt = excerpt[:139].rstrip() + "\u2026"
    if excerpt:
        return f"Key moment {scene_index}: {excerpt}"
    return f"Key moment {scene_index} at {timestamp:.1f}s"
