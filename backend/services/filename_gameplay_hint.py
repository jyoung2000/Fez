"""Filename-based gameplay hint heuristic.

When the user uploads a video whose title contains a recognizable
game name, we can short-circuit the expensive dense-face +
MediaPipe + YuNet pipeline that would otherwise run to decide
"is this gameplay?" — a decision that gets gated on a cartoon-
character-contaminated face ratio.

This module exposes a single function :func:`filename_gameplay_hint`
that returns a ``(matched: bool, slug: str, genre_hint: str)``
tuple for any filename containing a supported game keyword. The
slug is suitable for passing to :mod:`game_layouts` and the
genre hint is suitable for
:data:`l1_camera_path.CENTER_BIAS_GENRES`.

The keyword list is intentionally conservative — false positives
would incorrectly skip the face pipeline on non-gameplay clips.
When adding new entries, prefer unambiguous multi-character
tokens (``"valorant"``, ``"team fortress"``) over short
abbreviations (``"cs"``, ``"cod"``) that collide with ordinary
English words. Short abbreviations must be guarded with word
boundaries and an additional disambiguator.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional


# ──────────────────── Keyword bank ────────────────────


@dataclass(frozen=True)
class _GameKeyword:
    """One filename keyword + its gameplay-subtype hint."""

    pattern: re.Pattern[str]
    slug: str
    genre: str  # one of the CENTER_BIAS_GENRES values or "moba" / "tps" / "racing"


def _kw(regex: str, slug: str, genre: str) -> _GameKeyword:
    # Always case-insensitive; callers normalize the filename to
    # lowercase before matching.
    return _GameKeyword(
        pattern=re.compile(regex, re.IGNORECASE),
        slug=slug,
        genre=genre,
    )


# Order matters — first hit wins, so more-specific tokens come first.
GAME_KEYWORDS: tuple[_GameKeyword, ...] = (
    # FPS / hero shooter (center-bias genres)
    _kw(r"\bteam[\s_\-\.]?fortress[\s_\-\.]?2\b", "team_fortress_2", "fps"),
    _kw(r"\btf\s?2\b", "team_fortress_2", "fps"),
    _kw(r"\bvalorant\b", "valorant", "fps"),
    _kw(r"\bapex[\s_\-\.]?legends?\b", "apex_legends", "fps"),
    _kw(r"\bapex\b", "apex_legends", "fps"),
    _kw(r"\bcs\s?go\b", "cs_go", "fps"),
    _kw(r"\bcs\s?2\b", "cs_2", "fps"),
    _kw(r"\bcounter[\s_\-]?strike\b", "cs_go", "fps"),
    _kw(r"\bcall[\s_\-]of[\s_\-]duty\b", "call_of_duty", "fps"),
    _kw(r"\bwarzone\b", "warzone", "fps"),
    _kw(r"\bmodern[\s_\-]warfare\b", "modern_warfare", "fps"),
    _kw(r"\bovervatch\b|\boverwatch\b", "overwatch", "hero_shooter"),
    _kw(r"\bmarvel[\s_\-]?rivals\b", "marvel_rivals", "hero_shooter"),
    _kw(r"\brainbow[\s_\-]?six\b", "rainbow_six", "fps"),
    _kw(r"\br6[\s_\-]?siege\b", "rainbow_six", "fps"),
    _kw(r"\btitanfall\b", "titanfall", "fps"),
    _kw(r"\bhalo\s?(infinite|reach|mcc|\d)?\b", "halo", "fps"),
    _kw(r"\bdestiny\s?2?\b", "destiny", "fps"),
    _kw(r"\bdoom[\s_\-]?(eternal|2016)?\b", "doom", "fps"),
    _kw(r"\bbattlefield\s?\d*\b", "battlefield", "fps"),
    _kw(r"\bescape[\s_\-]from[\s_\-]tarkov\b", "tarkov", "fps"),

    # Battle royale / sandbox FPS
    _kw(r"\bfortnite\b", "fortnite", "fps"),
    _kw(r"\bpubg\b", "pubg", "fps"),
    _kw(r"\bminecraft\b", "minecraft", "sandbox"),
    _kw(r"\broblox\b", "roblox", "sandbox"),

    # Non-center-bias genres (still useful as hints, no crop bias)
    _kw(r"\bleague[\s_\-]of[\s_\-]legends\b", "league_of_legends", "moba"),
    _kw(r"\blol\b", "league_of_legends", "moba"),
    _kw(r"\bdota\s?2?\b", "dota_2", "moba"),
    _kw(r"\brocket[\s_\-]?league\b", "rocket_league", "racing"),
    _kw(r"\bforza\b", "forza", "racing"),
    _kw(r"\bgran[\s_\-]?turismo\b", "gran_turismo", "racing"),
    _kw(r"\belden[\s_\-]?ring\b", "elden_ring", "tps"),
    _kw(r"\bdark[\s_\-]?souls\b", "dark_souls", "tps"),
    _kw(r"\bgta\s?[v5]?\b", "gta_v", "tps"),
    _kw(r"\bred[\s_\-]?dead\b", "red_dead", "tps"),
    _kw(r"\bgears[\s_\-]of[\s_\-]war\b", "gears_of_war", "tps"),
    _kw(r"\bsplit[\s_\-]?gate\b", "splitgate", "fps"),
)


# ──────────────────── Public API ────────────────────


@dataclass
class FilenameGameplayHint:
    """Result of :func:`filename_gameplay_hint`."""

    matched: bool
    slug: str = ""          # e.g. "team_fortress_2"
    genre: str = ""         # e.g. "fps" — matches CENTER_BIAS_GENRES
    source_token: str = ""  # the substring that matched, for logging


def filename_gameplay_hint(
    filename: Optional[str],
) -> FilenameGameplayHint:
    """Check whether a filename matches a known gameplay title.

    Args:
        filename: Video filename or full path. Both basename-only
            and full-path inputs are supported. When ``None``
            or empty, returns a ``matched=False`` result.

    Returns:
        A :class:`FilenameGameplayHint` — ``matched=False`` on a
        miss, otherwise the first keyword hit in the keyword bank.
    """
    if not filename:
        return FilenameGameplayHint(matched=False)
    basename = os.path.basename(str(filename))
    # Normalize separator-ish punctuation so "TF2_Washed-Up.Tuber"
    # still matches ``tf 2``.
    normalized = basename.replace("_", " ").replace(".", " ")
    normalized = normalized.replace("-", " ")
    # Collapse ``Ｕｎｉｃｏｄｅ`` separators too.
    normalized = normalized.replace("：", " ").replace(":", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()

    for kw in GAME_KEYWORDS:
        m = kw.pattern.search(normalized)
        if m:
            return FilenameGameplayHint(
                matched=True,
                slug=kw.slug,
                genre=kw.genre,
                source_token=m.group(0),
            )
    return FilenameGameplayHint(matched=False)


def is_center_bias_genre(genre: str) -> bool:
    """Whether a genre hint should trigger center-bias reframing."""
    return genre in {"fps", "gameplay_fps", "hero_shooter", "sandbox", "gameplay"}
