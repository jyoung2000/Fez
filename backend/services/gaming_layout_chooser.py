"""Phase 4 — per-segment layout mode chooser for gaming clips.

The legacy gaming reframe path picks one strategy per clip
(fullscreen center-crop OR composite). Real esports editors
switch layouts per moment: fullscreen for the action, blurfill
letterbox during a team fight, wide-zoom for chaos / clutch
moments, scoreboard view for the tab screen.

This module ships the chooser as a pure function so it can be
unit tested without standing up the full segmenter. The reframe
segmenter calls it once per segment after Stage 10 for every
gaming clip.

Layout modes:

  - ``"fullscreen"``  — center-crop fills the 9:16 frame
                        (default for FPS / TPS / sandbox).
  - ``"blurfill"``    — full 16:9 source band centered, blurred
                        fill above and below (industry standard
                        for MOBA / RTS lane fights).
  - ``"composite"``   — legacy action-on-top + HUD strip.
  - ``"wide_zoom"``   — fullscreen with the action scaled to
                        80 % of crop width so more horizontal
                        context is visible (chaos moments).

Feature flag: ``CLIPAI_GAMING_LAYOUT_CHOOSER`` env var, default ON.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# ──────────────────── Feature flag ────────────────────

USE_GAMING_LAYOUT_CHOOSER = os.environ.get(
    "CLIPAI_GAMING_LAYOUT_CHOOSER", "1",
).lower() in ("1", "true", "yes", "on")


# ──────────────────── Tunables ────────────────────

# A MOBA / RTS segment shorter than this stays on the composite
# layout. Long lane fights switch to blurfill so the whole team
# fight is visible.
GAMING_BLURFILL_MIN_DURATION = float(
    os.environ.get("GAMING_BLURFILL_MIN_DURATION", "4.0"),
)

# Motion magnitude (loose 0–100 metric) above which we switch to
# wide_zoom so the viewer sees more horizontal context.
GAMING_WIDE_ZOOM_MOTION_THRESHOLD = float(
    os.environ.get("GAMING_WIDE_ZOOM_MOTION_THRESHOLD", "30.0"),
)

# Racing motion threshold for keeping fullscreen (above = follow
# the car, below = composite to show speedo + minimap).
GAMING_RACING_FULLSCREEN_MOTION = float(
    os.environ.get("GAMING_RACING_FULLSCREEN_MOTION", "20.0"),
)

VALID_LAYOUTS = ("fullscreen", "blurfill", "composite", "wide_zoom")


# ──────────────────── Helpers ────────────────────


def has_event_kind(events: Iterable, *kinds: str) -> bool:
    """True when any event in the iterable matches one of ``kinds``."""
    if not events:
        return False
    target = set(kinds)
    for ev in events:
        if getattr(ev, "kind", None) in target:
            return True
    return False


def events_in_range(events: Iterable, start: float, end: float) -> list:
    """Return events whose timestamp + duration overlaps [start, end)."""
    if not events:
        return []
    out = []
    for ev in events:
        ev_t = float(getattr(ev, "timestamp", 0.0))
        ev_d = float(getattr(ev, "duration", 0.0))
        if ev_t + ev_d < start or ev_t >= end:
            continue
        out.append(ev)
    return out


def normalize_genre(profile, fallback: str = "fps") -> str:
    """Pull the gameplay genre off a content profile.

    Looks at ``profile.gameplay_subtype`` then falls back to the
    raw ``content_type`` lowercase. Returns ``fallback`` when no
    gameplay info is present.
    """
    if profile is None:
        return fallback
    sub = getattr(profile, "gameplay_subtype", None)
    if sub:
        return str(sub).strip().lower()
    ct = getattr(profile, "content_type", "") or ""
    ct = str(ct).strip().lower()
    if ct.startswith("gameplay_"):
        return ct.split("_", 1)[1]
    if ct in ("fps", "moba", "tps", "racing", "sandbox", "stream", "br"):
        return ct
    return fallback


# ──────────────────── Per-segment chooser ────────────────────


def choose_gaming_layout(
    *,
    seg_start: float,
    seg_end: float,
    events_in_seg: list,
    motion_in_seg: float,
    profile=None,
    blurfill_min_duration: float = GAMING_BLURFILL_MIN_DURATION,
    wide_zoom_motion_threshold: float = GAMING_WIDE_ZOOM_MOTION_THRESHOLD,
) -> str:
    """Pick the per-segment gaming layout mode.

    The chooser is a flat decision tree gated on:

      1. Presence of a scoreboard / tab event (always blurfill so
         the full HUD is visible).
      2. Chaos events or high motion → wide_zoom.
      3. Genre default with duration / motion overrides:
         - MOBA / RTS: blurfill on long segments, composite on short.
         - Racing: fullscreen on high motion, composite otherwise.
         - TPS / sandbox: fullscreen.
         - FPS / hero shooter / default: fullscreen.

    Args:
        seg_start, seg_end: Segment time range.
        events_in_seg: List of GamingEvent inside the segment
            (caller pre-filters via :func:`events_in_range`).
        motion_in_seg: Loose motion magnitude estimator in [0, 100].
            Higher = more on-screen activity.
        profile: ContentProfile or None. Used to read the genre.
        blurfill_min_duration: Per-genre threshold for blurfill.
        wide_zoom_motion_threshold: Threshold for wide-zoom switch.

    Returns:
        One of ``VALID_LAYOUTS``.
    """
    if not USE_GAMING_LAYOUT_CHOOSER:
        return "fullscreen"

    # Tab / scoreboard always shows the whole HUD.
    if has_event_kind(events_in_seg, "scoreboard"):
        return "blurfill"

    # Chaos events (merged kills / ults) or extreme motion → wide_zoom.
    if has_event_kind(events_in_seg, "chaos"):
        return "wide_zoom"
    if motion_in_seg > wide_zoom_motion_threshold:
        return "wide_zoom"

    genre = normalize_genre(profile)
    duration = max(0.0, seg_end - seg_start)

    if genre in ("moba", "rts"):
        if duration >= blurfill_min_duration:
            return "blurfill"
        return "composite"

    if genre in ("racing",):
        if motion_in_seg >= GAMING_RACING_FULLSCREEN_MOTION:
            return "fullscreen"
        return "composite"

    if genre in ("tps", "sandbox"):
        return "fullscreen"

    # FPS / hero_shooter / generic
    return "fullscreen"


# ──────────────────── Bulk helper ────────────────────


def assign_gaming_layouts(
    segments: list,
    *,
    events: Optional[list] = None,
    motion_by_segment: Optional[dict] = None,
    profile=None,
) -> int:
    """Assign ``gaming_layout_mode`` to each segment in place.

    Returns the count of segments that received a mode. Segments
    that already have a non-empty ``gaming_layout_mode`` are left
    alone (caller-provided overrides win).

    Args:
        segments: List of segments with ``start``, ``end``,
            ``gaming_layout_mode`` attrs.
        events: Optional list of GamingEvent for the full clip.
        motion_by_segment: Optional dict keyed by id(segment) →
            motion magnitude for that segment.
        profile: ContentProfile or None.
    """
    if not segments:
        return 0
    n_assigned = 0
    for seg in segments:
        existing = getattr(seg, "gaming_layout_mode", None)
        if existing:
            continue
        seg_events = events_in_range(events or [], seg.start, seg.end)
        motion = (
            (motion_by_segment or {}).get(id(seg), 0.0)
            if motion_by_segment else 0.0
        )
        mode = choose_gaming_layout(
            seg_start=float(seg.start),
            seg_end=float(seg.end),
            events_in_seg=seg_events,
            motion_in_seg=float(motion),
            profile=profile,
        )
        try:
            seg.gaming_layout_mode = mode
        except Exception:
            continue
        n_assigned += 1
    return n_assigned
