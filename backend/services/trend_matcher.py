"""Trend matching layer — Phase 3 of the OpusClip parity gap.

OpusClip claims to compare each candidate clip against current short-
form trend signals. A real live scraper is out of scope (and fragile)
so we use three local inputs:

  1. A static, version-controlled lexicon of phrases known to perform
     on short-form platforms, tagged by genre and platform.
  2. The pipeline's existing emphasis-keyword detection, fed in as a
     "what this video itself emphasises" signal.
  3. (Optional, behind ``ENABLE_TREND_WEB_FETCH``) a web-fetched trend
     summary string from a configured endpoint.

The output is consumed two ways:

  * ``score_clip(text, content_type, platform)`` returns a ``(score,
    reason)`` tuple used by both ``hot_zone_scorer`` (as a fifth
    weighted axis) and ``ai_orchestrator`` (as the TREND CONTEXT block
    appended to the LLM clip-detection prompt).
  * ``format_trend_context`` renders a short summary suitable for
    inclusion in the LLM prompt.

Feature-flagged via ``USE_TREND_MATCHER`` — defaults off until the
first end-to-end test passes per the gap-close brief.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from backend.models import TranscriptSegment

logger = logging.getLogger(__name__)


_DEFAULT_LEXICON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),  # backend/
    "data",
    "trend_lexicon.json",
)


def trend_matcher_enabled() -> bool:
    """Phase 3 feature flag — defaults off per the gap-close brief."""
    raw = os.environ.get("USE_TREND_MATCHER")
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def web_trend_fetch_enabled() -> bool:
    """Optional web-fetch sub-flag, defaulting off."""
    raw = os.environ.get("ENABLE_TREND_WEB_FETCH")
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class TrendEntry:
    phrase: str
    genres: list[str] = field(default_factory=list)
    platforms: list[str] = field(default_factory=list)
    hotness: float = 0.5

    @property
    def normalized_phrase(self) -> str:
        return self.phrase.lower().strip()


class TrendMatcher:
    """Score arbitrary text against a curated trend lexicon.

    The matcher is deliberately simple — exact phrase, fuzzy keyword,
    and emphasis-keyword overlap. No ML, no vectorisation, no third-
    party deps. The point is "is this clip riding any of the trend
    signals we know about right now?" and the answer is robust as
    long as the lexicon is kept fresh.
    """

    def __init__(
        self,
        lexicon_path: Optional[str] = None,
        emphasis_keywords: Optional[list[str]] = None,
        web_summary: Optional[str] = None,
    ):
        self._entries: list[TrendEntry] = []
        self._emphasis = [e.lower() for e in (emphasis_keywords or [])]
        self._web_summary = (web_summary or "").lower()
        self._load_lexicon(lexicon_path or _DEFAULT_LEXICON_PATH)

    # ── Public API ──────────────────────────────────────────────

    def score_clip(
        self,
        clip_text: str,
        content_type=None,
        platform: str = "generic",
    ) -> tuple[int, str]:
        """Score how trend-aligned a clip text is.

        Returns ``(score, reason)`` with score floored at 20 and
        capped at 100. Default 50 (neutral) when no signal fires.
        """
        if not clip_text or not clip_text.strip():
            return 50, "no transcript text"

        text_lower = clip_text.lower()
        genre_key = self._content_type_key(content_type)

        score = 50.0
        reasons: list[str] = []

        # 1. Exact-phrase lexicon hits — strongest signal.
        for entry in self._entries:
            if entry.normalized_phrase in text_lower:
                if genre_key and entry.genres and genre_key not in entry.genres:
                    # Genre mismatch — don't let an off-genre trend
                    # phrase boost the clip.
                    continue
                score += 30 * float(entry.hotness)
                reasons.append(entry.phrase)
                if len(reasons) >= 3:
                    break

        # 2. Fuzzy keyword overlap — weaker, deduplicated against #1.
        if not reasons:
            tokens = set(_tokenize(text_lower))
            for entry in self._entries:
                phrase_tokens = set(_tokenize(entry.normalized_phrase))
                if not phrase_tokens:
                    continue
                overlap = phrase_tokens & tokens
                if len(overlap) >= max(1, len(phrase_tokens) - 1):
                    if genre_key and entry.genres and genre_key not in entry.genres:
                        continue
                    score += 15 * float(entry.hotness)
                    reasons.append(f"~{entry.phrase}")
                    if len(reasons) >= 3:
                        break

        # 3. Local emphasis-keyword overlap — the user's own video tells
        # us which words matter to it. Each hit is +10, capped at +25.
        if self._emphasis:
            emphasis_hits = sum(1 for kw in self._emphasis if kw and kw in text_lower)
            if emphasis_hits:
                score += min(25, emphasis_hits * 10)
                reasons.append(f"{emphasis_hits} emphasized")

        # 4. Optional web summary overlap.
        if self._web_summary:
            web_tokens = set(_tokenize(self._web_summary))
            text_tokens = set(_tokenize(text_lower))
            shared = web_tokens & text_tokens
            if len(shared) >= 2:
                score += 10
                reasons.append("web trend overlap")

        score = max(20, min(100, score))
        if not reasons:
            return int(round(score)), "no trend signal — neutral"
        return int(round(score)), ", ".join(reasons)

    def matched_phrases(self, clip_text: str, content_type=None) -> list[str]:
        """Return the lexicon phrases that fire for ``clip_text``."""
        if not clip_text:
            return []
        text_lower = clip_text.lower()
        genre_key = self._content_type_key(content_type)
        out: list[str] = []
        for entry in self._entries:
            if entry.normalized_phrase in text_lower:
                if genre_key and entry.genres and genre_key not in entry.genres:
                    continue
                out.append(entry.phrase)
        return out

    # ── Internals ───────────────────────────────────────────────

    def _load_lexicon(self, path: str) -> None:
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.info("TrendMatcher: no lexicon at %s — running empty", path)
            return
        except Exception as e:
            logger.warning("TrendMatcher: failed to load lexicon %s: %s", path, e)
            return

        for raw in data.get("entries", []):
            try:
                self._entries.append(TrendEntry(
                    phrase=str(raw["phrase"]),
                    genres=list(raw.get("genres", [])),
                    platforms=list(raw.get("platforms", [])),
                    hotness=float(raw.get("hotness", 0.5)),
                ))
            except Exception:
                continue
        logger.info(
            "TrendMatcher: loaded %d entries from %s (version=%s)",
            len(self._entries), path, data.get("version", "unknown"),
        )

    @staticmethod
    def _content_type_key(content_type) -> Optional[str]:
        if content_type is None:
            return None
        if hasattr(content_type, "value"):
            return content_type.value
        return str(content_type)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z0-9'-]{1,}", text.lower())


def build_default_trend_matcher(
    emphasis_keywords: Optional[list[str]] = None,
    web_summary: Optional[str] = None,
) -> TrendMatcher:
    """Convenience constructor used by the pipeline."""
    return TrendMatcher(
        lexicon_path=_DEFAULT_LEXICON_PATH,
        emphasis_keywords=emphasis_keywords,
        web_summary=web_summary,
    )


def format_trend_context(
    matcher: TrendMatcher,
    transcript: list[TranscriptSegment],
    content_type=None,
    max_phrases: int = 6,
) -> str:
    """Render a 2-line trend context block for the LLM prompt.

    Walks the full transcript, collects the unique lexicon phrases
    that fire across the video, and renders them as a list. Returns
    an empty string when nothing fires so the orchestrator can skip
    the section entirely.
    """
    if not transcript or not matcher:
        return ""
    full_text = " ".join(s.text for s in transcript)
    phrases = matcher.matched_phrases(full_text, content_type=content_type)
    if not phrases:
        return ""
    unique = []
    for p in phrases:
        if p not in unique:
            unique.append(p)
        if len(unique) >= max_phrases:
            break
    return (
        "The following trending short-form phrases appeared in this video — "
        "clips that contain them should score higher on the trend axis: "
        + ", ".join(unique)
    )
