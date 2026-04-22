"""Per-genre SAM 3 prompt lists — Blueprint v2 Phase 4.

Matches the blueprint's Stage 2 table: what concept prompts drive
detection + tracking for each genre. ``prompts_for`` is keyed by
``ClipContentType`` value strings; unknown keys fall back to
``generic``.

Keeping this as a plain dict (not on ``ReframeConfig``) because the
prompt list is a SAM-3-specific knob that doesn't generalize to the
rest of the reframing stack. When Phase 4.5 retires the legacy
detector path, these prompts become the definition of what ClipAI
can "see" on each genre.
"""

from __future__ import annotations


PROMPTS_BY_GENRE: dict[str, list[str]] = {
    # Talking head / single speaker
    "talking_head":        ["visible face", "speaking person"],
    "podcast":             ["visible face", "speaking person"],
    "interview":           ["visible face", "speaking person"],
    "multi_speaker_panel": ["visible face", "speaking person"],

    # Cinematic / narrative
    "cinematic_dialogue":  ["visible face", "speaking person", "main subject"],
    "narrative":           ["visible face", "speaking person", "main subject"],

    # Sports
    "sports":              ["the ball", "players", "scoreboard", "score bug"],
    "sports_basketball":   ["basketball", "players", "scoreboard"],
    "sports_racing":       ["lead car", "cars", "racing flag"],

    # Gameplay (includes legacy ``gaming`` alias)
    "gameplay":            [
        "player character", "HUD elements", "minimap", "health bar",
        "webcam overlay", "crosshair",
    ],
    "gameplay_fps":        [
        "player crosshair", "HUD elements", "minimap",
        "webcam overlay", "kill feed",
    ],
    "gameplay_moba":       [
        "player champion", "minimap", "HUD elements", "webcam overlay",
    ],
    "gameplay_tps":        [
        "player character", "HUD elements", "webcam overlay",
    ],
    "gameplay_racing":     [
        "player vehicle", "HUD elements", "minimap", "webcam overlay",
    ],
    "gaming":              [
        "player character", "HUD elements", "minimap", "webcam overlay",
    ],

    # Stream (gameplay + webcam)
    "stream":              [
        "webcam overlay", "game content", "chat overlay", "alert overlay",
    ],

    # Animation
    "animation":           ["anime face", "speech bubble", "subtitle text", "character"],
    "anime":               ["anime face", "speech bubble", "subtitle text", "character"],
    "animation_dialogue":  ["anime face", "speech bubble"],

    # Music
    "music_video":         ["performer", "instrument", "vocalist"],
    "music_performance":   ["performer", "instrument", "vocalist", "stage lighting"],

    # Documentary / B-roll
    "documentary":         ["main subject", "narrator"],
    "landscape":           [],
    "broll":               ["visible face", "main subject"],

    # Screen-share / tutorial
    "tutorial":            [
        "face", "screen capture region", "cursor", "presentation slide",
    ],
    "screen_share":        ["screen capture region", "cursor", "face"],

    # Vlog
    "vlog":                ["face", "main subject"],

    # Fallback
    "generic":             ["visible face", "main subject"],
    "unknown":             ["visible face", "main subject"],
}


def prompts_for(content_type) -> list[str]:
    """Return the SAM 3 prompt list for a content type.

    Accepts enum (``.value``-bearing) or string. Missing keys fall
    back to ``generic``.
    """
    key = getattr(content_type, "value", content_type)
    if key is None:
        return list(PROMPTS_BY_GENRE["generic"])
    return list(PROMPTS_BY_GENRE.get(str(key), PROMPTS_BY_GENRE["generic"]))
