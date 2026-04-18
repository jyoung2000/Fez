"""Canonical string mapping for UI-selected content types.

The reframe pipeline has multiple layers of content-type bookkeeping:

- The upload UI dropdown sends a string token (``content_type_override``).
- ``content_classifier.classify_content`` picks a ``ContentType`` enum value.
- ``classify_clip`` maps that to a ``ClipContentType`` for solver tuning.

Historically these layers had mismatched string spellings — the UI
would send ``"gameplay"`` but the enum value is ``"gaming"``; the UI
would send ``"movie"`` but the enum value is ``"narrative"``; only
``"podcast"`` matched by coincidence. The result was that the
classifier's user-override branch silently never fired for two of the
three existing dropdown values, and new UI values (``debate``,
``vlog``, ``anime``, ``music_video``, the gameplay variants, etc.)
would all hit the same dead code path.

This module is the single source of truth for the UI → enum mapping
so:

1. Every UI token is normalized to a ``ContentType`` enum + a bundle
   of editorial flags (``is_multi_speaker_panel``, ``is_animated``).
2. Legacy UI aliases (``"gameplay"`` / ``"movie"``) keep working
   forever.
3. Forward-compat tokens for Phase 2 (``debate``, ``vlog``, ``anime``,
   ``music_video``, ``gameplay_moba``, ``gameplay_tps``,
   ``gameplay_racing``, ``stream``, ``sports``) all normalize today so
   we don't have to touch ``content_classifier.py`` again when the
   dropdown grows.
4. Unknown / empty / ``"auto"`` tokens return ``None`` so the caller
   falls through to heuristic classification.

``is_gameplay_override`` is a convenience for the pipeline's early
gameplay fast-path, which skips face tracking entirely. Note that
``"stream"`` is deliberately NOT a gameplay fast-path candidate — a
stream has a facecam which requires the face pipeline — so it
normalizes to ``ContentType.GAMING`` but is_gameplay_override is
False for it. Phase 7 handles the stream layout routing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from backend.services.content_type_config import ContentType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NormalizedContentType:
    """Result of normalizing a UI content-type string.

    Attributes:
        content_type: The canonical ``ContentType`` enum value.
        is_multi_speaker_panel: True when the UI explicitly picked a
            debate / panel format. Drives ``ClipContentType.MULTI_SPEAKER_PANEL``
            routing in ``classify_clip``.
        is_animated: True when the UI picked anime/cartoon. Routes to
            ``ClipContentType.ANIMATION`` / ``ANIMATION_DIALOGUE`` downstream.
        anime_subtype: "action" | "dialogue" | "slice_of_life" | None.
            Populated by the Phase 2 UI sub-dropdown via
            ``apply_subtypes_to_metadata`` (read from
            ``metadata['anime_subtype']`` by the classifier).
        music_subtype: "performance" | "narrative" | "lyric" | None.
            Populated by the Phase 2 UI sub-dropdown.
        gameplay_subtype: "fps" | "moba" | "tps" | "racing" | "stream" | None.
            Encoded directly in the UI token (``gameplay_moba`` →
            ``"moba"``) so ``classify_clip`` can route to the right
            ``ClipContentType.GAMEPLAY_*`` value.
        is_gameplay_fastpath: True when the pipeline should skip face
            tracking entirely (pure gameplay). False for ``stream``,
            which is gaming-layout but still needs the face pipeline.
        raw: The original UI token (lowercased + stripped). Preserved
            for logging / telemetry.
    """

    content_type: ContentType
    is_multi_speaker_panel: bool = False
    is_animated: bool = False
    anime_subtype: Optional[str] = None
    music_subtype: Optional[str] = None
    gameplay_subtype: Optional[str] = None
    sports_subtype: Optional[str] = None
    is_gameplay_fastpath: bool = False
    raw: str = ""


# Every dropdown value in Upload.jsx MUST appear here. The legacy
# tokens ("gameplay", "movie", "podcast") are kept forever as aliases
# so old jobs and old UI builds still route correctly.
#
# ``gameplay_subtype`` encodes which gameplay variant the user picked
# so ``classify_clip`` can route to the right ``ClipContentType.GAMEPLAY_*``
# value without needing to re-parse the UI token. The bare ``gameplay``
# legacy token defaults to FPS (matching the legacy pipeline behavior).
_UI_TO_ENUM: dict[str, dict] = {
    # ── Legacy aliases (must work forever) ──
    "gameplay": {
        "content_type": ContentType.GAMING,
        "gameplay_fastpath": True,
        "gameplay_subtype": "fps",
    },
    "movie": {"content_type": ContentType.NARRATIVE},
    "podcast": {"content_type": ContentType.PODCAST},

    # ── Phase 2 dropdown entries ──
    "debate": {
        "content_type": ContentType.PODCAST,
        "panel": True,
    },
    "panel": {
        "content_type": ContentType.PODCAST,
        "panel": True,
    },
    "interview": {"content_type": ContentType.PODCAST},
    "vlog": {"content_type": ContentType.VLOG},
    "narrative": {"content_type": ContentType.NARRATIVE},
    "cinematic": {"content_type": ContentType.NARRATIVE},
    "anime": {
        "content_type": ContentType.ANIME,
        "animated": True,
    },
    "cartoon": {
        "content_type": ContentType.ANIME,
        "animated": True,
    },
    "music_video": {"content_type": ContentType.MUSIC_VIDEO},
    "gameplay_fps": {
        "content_type": ContentType.GAMING,
        "gameplay_fastpath": True,
        "gameplay_subtype": "fps",
    },
    "gameplay_moba": {
        "content_type": ContentType.GAMING,
        "gameplay_fastpath": True,
        "gameplay_subtype": "moba",
    },
    "gameplay_tps": {
        "content_type": ContentType.GAMING,
        "gameplay_fastpath": True,
        "gameplay_subtype": "tps",
    },
    "gameplay_racing": {
        "content_type": ContentType.GAMING,
        "gameplay_fastpath": True,
        "gameplay_subtype": "racing",
    },
    # Stream is GAMING + facecam. It must NOT take the face-skipping
    # fast path — Phase 7 handles the STACKED_GAMEPLAY layout routing
    # and the facecam portion still needs the regular face pipeline.
    "stream": {
        "content_type": ContentType.GAMING,
        "gameplay_fastpath": False,
        "gameplay_subtype": "stream",
    },
    "sports": {"content_type": ContentType.SPORTS},
    "sports_basketball": {
        "content_type": ContentType.SPORTS,
        "sports_subtype": "basketball",
    },
    "sports_racing": {
        "content_type": ContentType.SPORTS,
        "sports_subtype": "racing",
    },
    "basketball": {
        "content_type": ContentType.SPORTS,
        "sports_subtype": "basketball",
    },
    "racing": {
        "content_type": ContentType.SPORTS,
        "sports_subtype": "racing",
    },

    # ── Special no-op tokens ──
    "": None,
    "auto": None,
    "unknown": None,
}


# Allowed sub-type values per parent — the classifier rejects garbage
# from the UI rather than passing it through unchecked. Phase 6 (anime)
# and Phase 5 (music) wire these to actual tuning behavior.
_VALID_ANIME_SUBTYPES: frozenset[str] = frozenset({
    "action", "dialogue", "slice_of_life", "auto",
})
_VALID_MUSIC_SUBTYPES: frozenset[str] = frozenset({
    "performance", "narrative", "lyric", "auto",
})


def normalize_anime_subtype(value: Optional[str]) -> Optional[str]:
    """Validate an ``anime_subtype`` UI value.

    Returns the lowercased token if valid, ``None`` otherwise (which
    also means "auto" / unset / invalid). The classifier treats
    ``None`` as "no subtype hint" and falls back to heuristic anime
    classification (Phase 6).
    """
    if not value:
        return None
    key = str(value).strip().lower()
    if not key or key == "auto":
        return None
    if key not in _VALID_ANIME_SUBTYPES:
        logger.debug(
            "content_type_strings: unknown anime_subtype %r — ignoring",
            value,
        )
        return None
    return key


def normalize_music_subtype(value: Optional[str]) -> Optional[str]:
    """Validate a ``music_subtype`` UI value. See ``normalize_anime_subtype``."""
    if not value:
        return None
    key = str(value).strip().lower()
    if not key or key == "auto":
        return None
    if key not in _VALID_MUSIC_SUBTYPES:
        logger.debug(
            "content_type_strings: unknown music_subtype %r — ignoring",
            value,
        )
        return None
    return key


def normalize_ui_content_type(token: Optional[str]) -> Optional[NormalizedContentType]:
    """Normalize a UI content-type token to an enum + flag bundle.

    Accepts ``None``, the empty string, ``"auto"``, ``"unknown"``, and
    every registered UI dropdown token. Unknown / invalid tokens return
    ``None`` so the caller falls through to heuristic classification
    rather than crashing.

    Case-insensitive; leading / trailing whitespace is stripped. An
    unknown token emits a debug log so misconfigured dropdown values
    are traceable.
    """
    if token is None:
        return None
    key = token.strip().lower()
    if not key:
        return None
    if key in {"auto", "unknown"}:
        return None
    spec = _UI_TO_ENUM.get(key)
    if spec is None:
        logger.debug(
            "content_type_strings: unknown UI token %r — falling through "
            "to heuristic classification",
            token,
        )
        return None
    return NormalizedContentType(
        content_type=spec["content_type"],
        is_multi_speaker_panel=bool(spec.get("panel", False)),
        is_animated=bool(spec.get("animated", False)),
        anime_subtype=spec.get("anime_subtype"),
        music_subtype=spec.get("music_subtype"),
        gameplay_subtype=spec.get("gameplay_subtype"),
        sports_subtype=spec.get("sports_subtype"),
        is_gameplay_fastpath=bool(spec.get("gameplay_fastpath", False)),
        raw=key,
    )


def is_gameplay_override(token: Optional[str]) -> bool:
    """Return True when the UI token picks a pure gameplay variant.

    Used by the pipeline's early gameplay fast-path which skips face
    tracking entirely. Covers:

    - the legacy ``gameplay`` value
    - the Phase 2 ``gameplay_fps`` / ``gameplay_moba`` / ``gameplay_tps``
      / ``gameplay_racing`` variants

    Does **not** include ``stream`` — a stream is gaming-layout but has
    a facecam and still needs the face pipeline. Phase 7 routes stream
    content through ``ClipContentType.STREAM`` downstream.
    """
    n = normalize_ui_content_type(token)
    return n is not None and n.is_gameplay_fastpath


def is_user_override(token: Optional[str]) -> bool:
    """Return True when the UI token is a recognized, non-auto override.

    Used by the pipeline to decide whether to skip gameplay auto-detect
    (if the user already declared a non-gameplay type, we should not
    override their choice with a heuristic).
    """
    return normalize_ui_content_type(token) is not None
