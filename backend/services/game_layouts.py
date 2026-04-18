"""HUD layout database for known FPS / hero shooter / MOBA / TPS / racing games.

Each layout defines the source-frame positions (as percentages) of critical
HUD elements that should be preserved when compositing a 9:16 vertical
frame from 16:9 gameplay footage.

HUD-zone keys:
    x_pct, y_pct  — top-left corner of the HUD element (% of source frame)
    w_pct, h_pct  — width and height of the element (% of source frame)

Layout-level metadata keys:
    name            — display name for the UI dropdown / logs
    genre           — "fps" | "moba" | "tps" | "racing" | "sandbox"
    action_center_pct: (x, y)
        The on-screen anchor where the camera should be biased. For
        FPS this is (50, 50) — crosshair-centered. For MOBA / top-down
        the action also sits at center but with a wider safe-zone. For
        third-person action games (GTA, Elden Ring) the player character
        is offset down+right of frame center, so the anchor sits at
        roughly (50, 45) — slightly above center vertically. For racing
        the car sits in the lower third, so the anchor is (50, 65).
        Phase 7's gameplay subject tracker reads this to override the
        hard-coded ``subject_x = 50`` baked into the legacy gameplay path.

NOTE: The HUD bbox values for the new MOBA / TPS / racing entries are
conservative starting points — designed to err on the side of preserving
critical UI rather than maximizing crop area. Phase 7 will refine them
with telemetry from real footage.
"""

GAME_HUD_LAYOUTS = {
    # ─── FPS / hero shooters ───────────────────────────────────────────
    "overwatch": {
        "name": "Overwatch / Overwatch 2",
        "genre": "fps",
        "action_center_pct": (50, 50),
        "killfeed": {"x_pct": 70, "y_pct": 5, "w_pct": 28, "h_pct": 18},
        "abilities": {"x_pct": 38, "y_pct": 85, "w_pct": 24, "h_pct": 12},
        "ultimate": {"x_pct": 45, "y_pct": 80, "w_pct": 10, "h_pct": 10},
        "health": {"x_pct": 5, "y_pct": 87, "w_pct": 20, "h_pct": 10},
    },
    "marvel_rivals": {
        "name": "Marvel Rivals",
        "genre": "fps",
        "action_center_pct": (50, 50),
        "killfeed": {"x_pct": 72, "y_pct": 8, "w_pct": 26, "h_pct": 18},
        "abilities": {"x_pct": 35, "y_pct": 85, "w_pct": 30, "h_pct": 12},
        "health": {"x_pct": 35, "y_pct": 92, "w_pct": 30, "h_pct": 5},
    },
    "valorant": {
        "name": "Valorant",
        "genre": "fps",
        "action_center_pct": (50, 50),
        "killfeed": {"x_pct": 75, "y_pct": 5, "w_pct": 24, "h_pct": 25},
        "minimap": {"x_pct": 0, "y_pct": 0, "w_pct": 18, "h_pct": 25},
        "abilities": {"x_pct": 35, "y_pct": 88, "w_pct": 30, "h_pct": 10},
        "health": {"x_pct": 35, "y_pct": 94, "w_pct": 30, "h_pct": 5},
    },
    "apex_legends": {
        "name": "Apex Legends",
        "genre": "fps",
        "action_center_pct": (50, 50),
        "killfeed": {"x_pct": 0, "y_pct": 5, "w_pct": 30, "h_pct": 15},
        "minimap": {"x_pct": 85, "y_pct": 5, "w_pct": 15, "h_pct": 18},
        "health": {"x_pct": 35, "y_pct": 85, "w_pct": 30, "h_pct": 8},
    },
    "fortnite": {
        "name": "Fortnite",
        "genre": "fps",
        "action_center_pct": (50, 50),
        "killfeed": {"x_pct": 55, "y_pct": 3, "w_pct": 42, "h_pct": 15},
        "minimap": {"x_pct": 78, "y_pct": 65, "w_pct": 20, "h_pct": 30},
        "health": {"x_pct": 30, "y_pct": 90, "w_pct": 40, "h_pct": 8},
    },
    "generic_fps": {
        "name": "Generic FPS",
        "genre": "fps",
        "action_center_pct": (50, 50),
        "killfeed": {"x_pct": 70, "y_pct": 5, "w_pct": 28, "h_pct": 18},
        "health": {"x_pct": 35, "y_pct": 88, "w_pct": 30, "h_pct": 10},
    },

    # ─── MOBA / top-down ───────────────────────────────────────────────
    # Action sits near the centered camera but the safe-zone is much
    # wider since lane fights happen across the visible play area.
    "league_of_legends": {
        "name": "League of Legends",
        "genre": "moba",
        "action_center_pct": (50, 50),
        # Bottom-right minimap is the most critical preserve.
        "minimap": {"x_pct": 82, "y_pct": 70, "w_pct": 18, "h_pct": 30},
        "abilities": {"x_pct": 30, "y_pct": 85, "w_pct": 40, "h_pct": 15},
        "scoreboard": {"x_pct": 35, "y_pct": 0, "w_pct": 30, "h_pct": 8},
        "shop": {"x_pct": 0, "y_pct": 80, "w_pct": 20, "h_pct": 20},
    },
    "dota2": {
        "name": "Dota 2",
        "genre": "moba",
        "action_center_pct": (50, 50),
        # Dota's minimap is bottom-LEFT (opposite of LoL).
        "minimap": {"x_pct": 0, "y_pct": 70, "w_pct": 18, "h_pct": 30},
        "abilities": {"x_pct": 30, "y_pct": 85, "w_pct": 40, "h_pct": 15},
        "scoreboard": {"x_pct": 30, "y_pct": 0, "w_pct": 40, "h_pct": 6},
        "shop": {"x_pct": 82, "y_pct": 80, "w_pct": 18, "h_pct": 20},
    },
    "generic_moba": {
        "name": "Generic MOBA",
        "genre": "moba",
        "action_center_pct": (50, 50),
        "minimap": {"x_pct": 82, "y_pct": 70, "w_pct": 18, "h_pct": 30},
        "abilities": {"x_pct": 30, "y_pct": 85, "w_pct": 40, "h_pct": 15},
    },

    # ─── Third-person action / TPS ─────────────────────────────────────
    # Player character is offset down+right of frame center — anchor
    # sits slightly above true center so the head/shoulders land in
    # the upper-third of the vertical crop.
    "gta_v": {
        "name": "Grand Theft Auto V",
        "genre": "tps",
        "action_center_pct": (50, 45),
        "minimap": {"x_pct": 0, "y_pct": 75, "w_pct": 18, "h_pct": 25},
        "weapon_wheel": {"x_pct": 80, "y_pct": 80, "w_pct": 20, "h_pct": 20},
        "health": {"x_pct": 70, "y_pct": 88, "w_pct": 28, "h_pct": 8},
    },
    "elden_ring": {
        "name": "Elden Ring",
        "genre": "tps",
        "action_center_pct": (50, 45),
        "health": {"x_pct": 2, "y_pct": 85, "w_pct": 30, "h_pct": 5},
        "stamina": {"x_pct": 2, "y_pct": 90, "w_pct": 30, "h_pct": 4},
        "fp": {"x_pct": 2, "y_pct": 80, "w_pct": 30, "h_pct": 5},
        "items": {"x_pct": 2, "y_pct": 70, "w_pct": 12, "h_pct": 12},
    },
    "generic_tps": {
        "name": "Generic Third-Person Action",
        "genre": "tps",
        "action_center_pct": (50, 45),
        "minimap": {"x_pct": 0, "y_pct": 75, "w_pct": 18, "h_pct": 25},
        "health": {"x_pct": 70, "y_pct": 88, "w_pct": 28, "h_pct": 8},
    },

    # ─── Racing / driving ──────────────────────────────────────────────
    # Car sits in the lower-third, so anchor low to keep the road and
    # car body both visible in 9:16.
    "rocket_league": {
        "name": "Rocket League",
        "genre": "racing",
        # Rocket League is somewhere between racing and sports — the
        # camera is high-and-back so the action is near vertical center.
        "action_center_pct": (50, 55),
        "scoreboard": {"x_pct": 30, "y_pct": 0, "w_pct": 40, "h_pct": 10},
        "boost": {"x_pct": 78, "y_pct": 80, "w_pct": 20, "h_pct": 15},
        "timer": {"x_pct": 45, "y_pct": 0, "w_pct": 10, "h_pct": 8},
    },
    "generic_racing": {
        "name": "Generic Racing / Driving",
        "genre": "racing",
        "action_center_pct": (50, 65),
        "speedo": {"x_pct": 78, "y_pct": 80, "w_pct": 20, "h_pct": 18},
        "minimap": {"x_pct": 0, "y_pct": 78, "w_pct": 18, "h_pct": 22},
        "position": {"x_pct": 0, "y_pct": 0, "w_pct": 18, "h_pct": 10},
    },

    # ─── Sandbox ───────────────────────────────────────────────────────
    "minecraft": {
        "name": "Minecraft",
        "genre": "sandbox",
        # First-person hotbar + crosshair = FPS-like center anchor.
        "action_center_pct": (50, 50),
        "hotbar": {"x_pct": 30, "y_pct": 90, "w_pct": 40, "h_pct": 10},
        "health": {"x_pct": 30, "y_pct": 82, "w_pct": 18, "h_pct": 5},
        "hunger": {"x_pct": 52, "y_pct": 82, "w_pct": 18, "h_pct": 5},
        "exp_bar": {"x_pct": 30, "y_pct": 87, "w_pct": 40, "h_pct": 3},
    },
}

# All known game keys for the frontend dropdown, grouped by genre.
# The frontend filters this by the user's gameplay_* parent selection
# so an FPS user only sees FPS games, a MOBA user only sees MOBAs, etc.
GAME_CHOICES = [
    # FPS / hero shooters
    ("auto", "Auto-detect"),
    ("overwatch", "Overwatch / Overwatch 2"),
    ("marvel_rivals", "Marvel Rivals"),
    ("valorant", "Valorant"),
    ("apex_legends", "Apex Legends"),
    ("fortnite", "Fortnite"),
    ("generic_fps", "Other FPS"),
    # MOBAs
    ("league_of_legends", "League of Legends"),
    ("dota2", "Dota 2"),
    ("generic_moba", "Other MOBA"),
    # TPS / action
    ("gta_v", "Grand Theft Auto V"),
    ("elden_ring", "Elden Ring"),
    ("generic_tps", "Other Third-Person Action"),
    # Racing
    ("rocket_league", "Rocket League"),
    ("generic_racing", "Other Racing / Driving"),
    # Sandbox
    ("minecraft", "Minecraft"),
]

# Per-genre default game key. Used by the frontend when the user picks
# a genre dropdown but hasn't yet picked a specific game, and by Phase 7
# downstream when ``profile.game_type`` is empty/auto for a given
# ``gameplay_subtype``.
DEFAULT_GAME_BY_GENRE: dict[str, str] = {
    "fps": "generic_fps",
    "moba": "generic_moba",
    "tps": "generic_tps",
    "racing": "generic_racing",
    "sandbox": "minecraft",  # Minecraft is the dominant sandbox case
}

# Reverse lookup: which gameplay_subtype each game belongs to. Used to
# validate user input + filter the dropdown.
GAME_GENRE: dict[str, str] = {
    key: layout["genre"] for key, layout in GAME_HUD_LAYOUTS.items()
}


def get_hud_layout(game_key: str) -> dict:
    """Return the HUD layout for a game, falling back to generic_fps."""
    return GAME_HUD_LAYOUTS.get(game_key, GAME_HUD_LAYOUTS["generic_fps"])


def get_action_center(game_key: str) -> tuple[float, float]:
    """Return the (x, y) action-center anchor for a game in % coordinates.

    Defaults to ``(50.0, 50.0)`` (FPS center-crop) when the game is
    unknown. Phase 7's gameplay subject tracker uses this to override
    the legacy hard-coded ``subject_x = 50`` for non-FPS genres.
    """
    layout = GAME_HUD_LAYOUTS.get(game_key)
    if not layout:
        return (50.0, 50.0)
    cx, cy = layout.get("action_center_pct", (50, 50))
    return (float(cx), float(cy))


def games_for_genre(genre: str) -> list[tuple[str, str]]:
    """Return ``[(key, display_name), ...]`` for all games in a genre.

    Genre is one of: ``fps``, ``moba``, ``tps``, ``racing``, ``sandbox``.
    Falls back to all FPS games for an unknown genre.
    """
    target = (genre or "").strip().lower() or "fps"
    matches = [
        (key, layout["name"])
        for key, layout in GAME_HUD_LAYOUTS.items()
        if layout.get("genre") == target
    ]
    if not matches:
        return [
            (key, layout["name"])
            for key, layout in GAME_HUD_LAYOUTS.items()
            if layout.get("genre") == "fps"
        ]
    return matches
