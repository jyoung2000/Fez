"""HUD layout database for known FPS / hero shooter / MOBA / TPS / racing games and HUD-aware crop helpers (Phase 5).

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

from typing import Optional

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


# ──────────────────── Phase 5: HUD-aware gaming crops ────────────────────


def _bbox_pct_to_norm(bbox_pct: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Convert a (x, y, w, h) percent tuple to normalized 0-1."""
    x, y, w, h = bbox_pct
    return (x / 100.0, y / 100.0, w / 100.0, h / 100.0)


def compute_hud_aware_crop_window(
    *,
    action_center_pct: tuple[float, float],
    hud_bboxes_norm: list[tuple[float, float, float, float]],
    target_aspect: float,
    source_w: int,
    source_h: int,
) -> tuple[tuple[float, float, float, float], bool]:
    """Compute the minimal 9:16 crop including action center + HUD.

    ``action_center_pct`` is ``(x_pct, y_pct)`` in [0, 100].
    ``hud_bboxes_norm`` is a list of (x, y, w, h) in normalized [0, 1].
    ``target_aspect`` is e.g. 9/16 for a vertical output.

    Returns ``((x, y, w, h), fits)`` where:
      - the rect is normalized [0, 1] in source coordinates
      - ``fits=True`` means BOTH action and all HUD bboxes fit inside
      - ``fits=False`` means the caller should fall back to HUD_COMPOSITE
        (stacked viewport + horizontal HUD strip)

    The window is the smallest 9:16 (or other ``target_aspect``) crop
    centered on a point that minimizes distance to the action center
    while still covering the HUD bboxes' horizontal span. The result
    is clamped to the source bounds.
    """
    src_aspect = source_w / source_h if source_h > 0 else 1.0
    if target_aspect < src_aspect:
        crop_w_norm = (target_aspect * source_h) / source_w
        crop_h_norm = 1.0
    else:
        crop_w_norm = 1.0
        crop_h_norm = (source_w / target_aspect) / source_h
        crop_h_norm = min(1.0, crop_h_norm)

    action_cx = action_center_pct[0] / 100.0

    # Compute the horizontal span of all HUD bboxes (left edge, right edge).
    if hud_bboxes_norm:
        hud_left = min(b[0] for b in hud_bboxes_norm)
        hud_right = max(b[0] + b[2] for b in hud_bboxes_norm)
    else:
        hud_left = action_cx
        hud_right = action_cx

    # Span we need to cover: union of action point and HUD horizontal span.
    span_left = min(hud_left, action_cx)
    span_right = max(hud_right, action_cx)
    span_w = span_right - span_left

    fits = span_w <= crop_w_norm + 1e-6
    if fits:
        # Center the crop on the action center, then nudge to cover HUD.
        cx = action_cx
        x = cx - crop_w_norm / 2.0
        # Ensure HUD left/right are inside [x, x + crop_w_norm].
        if hud_left < x:
            x = hud_left
        if hud_right > x + crop_w_norm:
            x = hud_right - crop_w_norm
        # Clamp to source bounds.
        x = max(0.0, min(x, 1.0 - crop_w_norm))
    else:
        # HUD + action don't both fit horizontally → caller should
        # use HUD_COMPOSITE. Return a center-on-action crop anyway so
        # the viewport is well-defined.
        cx = action_cx
        x = cx - crop_w_norm / 2.0
        x = max(0.0, min(x, 1.0 - crop_w_norm))

    return ((float(x), 0.0, float(crop_w_norm), float(crop_h_norm)), bool(fits))


def select_hud_strip_rects(
    hud_bboxes_norm: list[tuple[float, float, float, float]],
    update_frequencies: Optional[list[float]] = None,
    *,
    max_count: int = 4,
) -> list[tuple[float, float, float, float]]:
    """Pick the HUD bboxes most worth displaying in the strip.

    "Most important" = highest update frequency (changing HUD = relevant
    HUD). When ``update_frequencies`` is None, returns ``hud_bboxes_norm``
    in input order, capped at ``max_count``.
    """
    if not hud_bboxes_norm:
        return []
    if update_frequencies is None or len(update_frequencies) != len(hud_bboxes_norm):
        return list(hud_bboxes_norm)[:max_count]
    paired = list(zip(hud_bboxes_norm, update_frequencies))
    paired.sort(key=lambda p: float(p[1]), reverse=True)
    return [p[0] for p in paired[:max_count]]


def arrange_hud_strip_horizontally(
    hud_bboxes_norm: list[tuple[float, float, float, float]],
    *,
    target_strip_height_px: int,
    target_strip_width_px: int,
    min_element_height_px: int = 30,
) -> dict:
    """Compute the per-element layout in the HUD strip.

    Returns a dict with:
      - ``slot_widths``: per-element output pixel widths summing to
        ``target_strip_width_px``.
      - ``element_height_px``: the actual rendered height; equals
        ``target_strip_height_px`` clamped to be >= ``min_element_height_px``
        (caller is responsible for upscaling small bboxes to maintain
        readability on 1080×1920).
      - ``needs_scale_up``: True when any element would render below
        ``min_element_height_px`` without scaling.

    The arrangement is purely horizontal (1 row, equal-share slots).
    """
    n = len(hud_bboxes_norm)
    if n == 0:
        return {
            "slot_widths": [],
            "element_height_px": int(target_strip_height_px),
            "needs_scale_up": False,
        }
    base_w = target_strip_width_px // n
    base_w = base_w - (base_w % 2)
    last_w = target_strip_width_px - base_w * (n - 1)
    last_w = last_w - (last_w % 2)
    widths = [base_w] * (n - 1) + [last_w]

    needs_scale_up = int(target_strip_height_px) < int(min_element_height_px)
    rendered_h = max(int(target_strip_height_px), int(min_element_height_px))
    return {
        "slot_widths": widths,
        "element_height_px": rendered_h,
        "needs_scale_up": needs_scale_up,
    }


def choose_hud_layout(
    *,
    action_center_pct: tuple[float, float],
    hud_bboxes_norm: list[tuple[float, float, float, float]],
    target_aspect: float,
    source_w: int,
    source_h: int,
) -> dict:
    """Top-level decision: single CROP or HUD_COMPOSITE.

    Returns a dict:
        {
          "kind": "crop" | "hud_composite",
          "viewport_rect": (x, y, w, h) normalized,
          "hud_strip_rects": list of (x, y, w, h) normalized (only for
              hud_composite).
        }
    """
    rect, fits = compute_hud_aware_crop_window(
        action_center_pct=action_center_pct,
        hud_bboxes_norm=hud_bboxes_norm,
        target_aspect=target_aspect,
        source_w=source_w,
        source_h=source_h,
    )
    if fits:
        return {
            "kind": "crop",
            "viewport_rect": rect,
            "hud_strip_rects": [],
        }
    # Doesn't fit → HUD_COMPOSITE. Viewport is the action-centered
    # crop above; HUD strip carries the original (un-cropped) HUD
    # source rects so the strip filter can extract them at full
    # resolution.
    return {
        "kind": "hud_composite",
        "viewport_rect": rect,
        "hud_strip_rects": list(hud_bboxes_norm),
    }


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
