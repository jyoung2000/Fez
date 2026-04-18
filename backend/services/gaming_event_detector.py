"""Phase 2 — Critical-moment detector for gaming clips.

Fuses cheap signals (HUD region pixel diffs, audio energy spikes,
crosshair velocity, screen-content classifier) into per-clip
``GamingEvent`` markers. The downstream L1 solver (Phase 3) reads
these to drive per-segment pans and zoom-outs that mirror what
real esports editors do.

Signals (all ≤ 5 ms per frame):

  - **Killfeed diff**: sample the known ``hud_layout['killfeed']``
    bbox at 4 fps, compute mean abs diff between consecutive
    samples. Spike > 25 (8-bit) for > 0.3 s = kill event.
  - **Ult / ability flash**: sample the abilities bbox; spike +
    nearby audio peak (within ±0.2 s) = ult cast.
  - **Audio energy spike**: > 2 σ above local mean. Confirmatory
    only — not a standalone trigger.
  - **Crosshair stillness + audio**: crosshair velocity < 2 %/s
    for > 0.5 s WITH an audio spike → "locked onto a target",
    candidate for zoom-out.
  - **Scoreboard / tab screen**: high screen-text classifier
    score → "tab pressed", emit with ``target_region=None``.
  - **Minimap activity**: dot-density spike in the minimap bbox
    for > 1 s → team fight forming.

The module is pure-Python except for the optional cv2 path used
to read frame pixels. Tests stub the frame reader so they can
exercise the fusion logic without real images.

Feature flag: ``CLIPAI_GAMING_EVENT_DETECTOR`` env var, default ON.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ──────────────────── Feature flag ────────────────────

USE_GAMING_EVENT_DETECTOR = os.environ.get(
    "CLIPAI_GAMING_EVENT_DETECTOR", "1",
).lower() in ("1", "true", "yes", "on")


# ──────────────────── Tunables ────────────────────

# Killfeed mean-abs-diff threshold (8-bit). Real killfeed updates
# produce 30–80 of diff because of the icon/text appearing on
# black-ish background. 25 picks them up reliably without
# triggering on background flicker.
KILLFEED_DIFF_THRESHOLD = 25.0
KILLFEED_MIN_DURATION = 0.3
KILLFEED_SAMPLE_FPS = 4.0

# Ability/ult flash threshold — abilities glow when they go off
# cooldown so the diff is even sharper than killfeed.
ABILITY_DIFF_THRESHOLD = 30.0

# Audio confirmation window (seconds before / after a visual spike).
AUDIO_CONFIRM_WINDOW = 0.2

# Audio spike: > N σ above local mean. 2.0 σ filters background
# noise without missing dramatic peaks (gunshot, ult voice line).
AUDIO_SIGMA_THRESHOLD = 2.0
AUDIO_LOCAL_WINDOW = 2.0  # seconds

# Crosshair stillness: velocity < this %/s for > MIN_DURATION
# seconds AND coincident audio spike → "locked on" event.
CROSSHAIR_STILL_VEL_PCT_PER_SEC = 2.0
CROSSHAIR_STILL_MIN_DURATION = 0.5

# Default event hold duration when not otherwise specified. The
# pan stays on the target for this many seconds before snapping
# back to the action anchor.
DEFAULT_EVENT_DURATION = 0.8

# Two events within this many seconds get merged into a single
# zoom-out instead of two consecutive pans.
EVENT_MERGE_WINDOW = 1.5


# ──────────────────── Result dataclass ────────────────────


@dataclass
class GamingEvent:
    """One critical moment in a gaming clip.

    Attributes:
        timestamp: When the moment happens (seconds).
        duration: How long the pan / zoom should hold (seconds).
        kind: One of ``"kill"``, ``"ult"``, ``"rotation"``,
            ``"scoreboard"``, ``"chaos"`` (merged), ``"locked_on"``.
        target_region: ``(x_pct, y_pct, w_pct, h_pct)`` in source
            % space for the pan target. ``None`` → no pan, just
            zoom out / show full frame.
        confidence: 0–1 score; higher = more confident.
        source: Free-form provenance string for telemetry
            (e.g. ``"killfeed_diff+audio"``).
    """

    timestamp: float
    duration: float
    kind: str
    target_region: Optional[tuple[float, float, float, float]] = None
    confidence: float = 0.5
    source: str = ""


# ──────────────────── HUD region helpers ────────────────────


def _hud_bbox(hud_layout: dict, key: str) -> Optional[
    tuple[float, float, float, float]
]:
    """Return ``(x_pct, y_pct, w_pct, h_pct)`` for a HUD key.

    Returns ``None`` when the layout doesn't define ``key``.
    """
    if not hud_layout:
        return None
    elem = hud_layout.get(key)
    if not isinstance(elem, dict):
        return None
    try:
        return (
            float(elem.get("x_pct", 0.0)),
            float(elem.get("y_pct", 0.0)),
            float(elem.get("w_pct", 0.0)),
            float(elem.get("h_pct", 0.0)),
        )
    except (TypeError, ValueError):
        return None


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


# ──────────────────── HUD region pixel diff ────────────────────


def _read_hud_patch(
    frame_path: str,
    bbox_pct: tuple[float, float, float, float],
):
    """Read the HUD bbox patch from a frame as a uint8 grayscale array.

    Returns ``None`` on any failure (cv2 missing, read failure,
    degenerate bbox). Tests can monkey-patch this to inject
    canned arrays without writing real images.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape[:2]
    x, y, bw, bh = bbox_pct
    x0 = max(0, int(w * x / 100.0))
    y0 = max(0, int(h * y / 100.0))
    x1 = min(w, int(w * (x + bw) / 100.0))
    y1 = min(h, int(h * (y + bh) / 100.0))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return img[y0:y1, x0:x1]


def _patch_diff(a, b) -> float:
    """Mean absolute difference between two grayscale patches.

    Returns 0.0 when shapes mismatch or numpy is missing.
    """
    if a is None or b is None:
        return 0.0
    try:
        import numpy as np
    except ImportError:
        return 0.0
    if a.shape != b.shape:
        # Resize b to match a — coarse but cheap.
        try:
            import cv2
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        except Exception:
            return 0.0
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def detect_hud_diff_spikes(
    frame_paths: list,
    hud_bbox_pct: tuple[float, float, float, float],
    *,
    threshold: float,
    min_duration: float,
    sample_fps: float = KILLFEED_SAMPLE_FPS,
    read_patch: Callable = _read_hud_patch,
) -> list[tuple[float, float]]:
    """Walk frames at ``sample_fps``, return list of (start, end) of spikes.

    A "spike" is a contiguous run of consecutive samples with
    ``mean_abs_diff > threshold`` that lasts at least
    ``min_duration`` seconds.

    Args:
        frame_paths: List of ``(timestamp, path)`` tuples in any
            order; the function sorts them.
        hud_bbox_pct: Region to sample.
        threshold: Per-sample mean-abs-diff threshold.
        min_duration: Minimum spike duration in seconds.
        sample_fps: How often to read a frame (samples / s).
        read_patch: Hook for tests; default reads pixels via cv2.

    Returns:
        List of ``(start_t, end_t)`` tuples for each spike.
    """
    if not frame_paths or not hud_bbox_pct:
        return []

    sorted_frames = sorted(frame_paths, key=lambda p: p[0])
    # Subsample at sample_fps
    sample_dt = 1.0 / max(sample_fps, 0.1)
    samples: list[tuple[float, str]] = []
    next_t = sorted_frames[0][0]
    for ts, path in sorted_frames:
        if ts >= next_t:
            samples.append((ts, path))
            next_t = ts + sample_dt

    if len(samples) < 2:
        return []

    diffs: list[tuple[float, float]] = []  # (timestamp, mean_abs_diff)
    prev_patch = None
    for ts, path in samples:
        patch = read_patch(path, hud_bbox_pct)
        if prev_patch is not None and patch is not None:
            diffs.append((ts, _patch_diff(prev_patch, patch)))
        prev_patch = patch

    if not diffs:
        return []

    # Scan for contiguous runs above threshold.
    spikes: list[tuple[float, float]] = []
    run_start: Optional[float] = None
    last_above_t: Optional[float] = None
    for ts, d in diffs:
        if d >= threshold:
            if run_start is None:
                run_start = ts
            last_above_t = ts
        else:
            if run_start is not None and last_above_t is not None:
                if last_above_t - run_start >= min_duration:
                    spikes.append((run_start, last_above_t))
            run_start = None
            last_above_t = None
    if run_start is not None and last_above_t is not None:
        if last_above_t - run_start >= min_duration:
            spikes.append((run_start, last_above_t))
    return spikes


# ──────────────────── Audio envelope helpers ────────────────────


def _audio_spike_at(
    audio_envelope: Optional[list],
    target_t: float,
    *,
    window: float = AUDIO_CONFIRM_WINDOW,
    sigma: float = AUDIO_SIGMA_THRESHOLD,
    local_window: float = AUDIO_LOCAL_WINDOW,
) -> bool:
    """True when the audio envelope has a >sigma spike near ``target_t``.

    ``audio_envelope`` is a list of ``(timestamp, energy)`` pairs
    sorted by timestamp. Returns False on empty / None input so
    the visual signals still function in audio-less tests.
    """
    if not audio_envelope:
        return False
    # Find values within target_t ± window for the candidate spike
    near = [
        (t, e) for t, e in audio_envelope
        if abs(t - target_t) <= window
    ]
    if not near:
        return False
    peak = max(e for _t, e in near)

    # Local window for mean / std
    local = [
        e for t, e in audio_envelope
        if abs(t - target_t) <= local_window
    ]
    if len(local) < 3:
        return False
    mean = sum(local) / len(local)
    var = sum((e - mean) ** 2 for e in local) / len(local)
    std = math.sqrt(var)
    if std <= 1e-9:
        return False
    return (peak - mean) >= sigma * std


# ──────────────────── Crosshair stillness ────────────────────


def detect_crosshair_lock_events(
    crosshair_path: Optional[list],
    audio_envelope: Optional[list],
    *,
    still_vel: float = CROSSHAIR_STILL_VEL_PCT_PER_SEC,
    min_duration: float = CROSSHAIR_STILL_MIN_DURATION,
) -> list[GamingEvent]:
    """Find frames where the crosshair locked onto a target.

    A "lock" is a stretch of crosshair frames whose pairwise
    per-second velocity stays below ``still_vel`` for at least
    ``min_duration``, AND coincides with an audio spike. Returns
    one ``GamingEvent(kind="locked_on", target_region=None)``
    centered on the lock window — Phase 3 reads this as a
    candidate for a brief zoom-out.
    """
    if not crosshair_path or len(crosshair_path) < 3:
        return []

    out: list[GamingEvent] = []
    sorted_path = sorted(crosshair_path, key=lambda c: c.timestamp)
    run_start: Optional[float] = None
    last_t: Optional[float] = None
    last_x: Optional[float] = None
    last_y: Optional[float] = None
    for cf in sorted_path:
        t = float(cf.timestamp)
        x = float(cf.x_pct)
        y = float(cf.y_pct)
        if last_t is None:
            run_start = t
            last_t = t
            last_x = x
            last_y = y
            continue
        dt = max(t - last_t, 1e-6)
        d = math.hypot(x - last_x, y - last_y)
        vel = d / dt
        if vel <= still_vel:
            if run_start is None:
                run_start = last_t
        else:
            if run_start is not None and (last_t - run_start) >= min_duration:
                mid = (run_start + last_t) / 2.0
                if _audio_spike_at(audio_envelope, mid):
                    out.append(GamingEvent(
                        timestamp=mid,
                        duration=DEFAULT_EVENT_DURATION,
                        kind="locked_on",
                        target_region=None,
                        confidence=0.65,
                        source="crosshair_still+audio",
                    ))
            run_start = None
        last_t = t
        last_x = x
        last_y = y

    if run_start is not None and last_t is not None and (last_t - run_start) >= min_duration:
        mid = (run_start + last_t) / 2.0
        if _audio_spike_at(audio_envelope, mid):
            out.append(GamingEvent(
                timestamp=mid,
                duration=DEFAULT_EVENT_DURATION,
                kind="locked_on",
                target_region=None,
                confidence=0.65,
                source="crosshair_still+audio",
            ))
    return out


# ──────────────────── Event merging ────────────────────


def merge_close_events(
    events: list[GamingEvent],
    *,
    window: float = EVENT_MERGE_WINDOW,
) -> list[GamingEvent]:
    """Merge events within ``window`` seconds into a single ``chaos`` event.

    When two HUD events fire close together the right reaction is
    a single zoom-out (so the viewer sees both regions at once),
    not two consecutive pans. Returns a new list with the merged
    events; emits a ``GamingEvent(kind="chaos", target_region=None)``
    for each merged group.
    """
    if not events:
        return []
    sorted_events = sorted(events, key=lambda e: e.timestamp)
    out: list[GamingEvent] = []
    i = 0
    while i < len(sorted_events):
        cur = sorted_events[i]
        group = [cur]
        j = i + 1
        while j < len(sorted_events) and (
            sorted_events[j].timestamp - cur.timestamp <= window
        ):
            group.append(sorted_events[j])
            j += 1
        if len(group) == 1:
            out.append(cur)
        else:
            t0 = min(e.timestamp for e in group)
            t1 = max(e.timestamp + e.duration for e in group)
            out.append(GamingEvent(
                timestamp=t0,
                duration=max(t1 - t0, DEFAULT_EVENT_DURATION),
                kind="chaos",
                target_region=None,
                confidence=max(e.confidence for e in group),
                source="merged:" + "+".join(sorted(set(e.kind for e in group))),
            ))
        i = j
    return out


# ──────────────────── Top-level fusion ────────────────────


def detect_gaming_events(
    *,
    frame_paths: list,
    audio_envelope: Optional[list] = None,
    hud_layout: Optional[dict] = None,
    crosshair_path: Optional[list] = None,
    gameplay_subtype: Optional[str] = None,
    read_patch: Callable = _read_hud_patch,
) -> list[GamingEvent]:
    """Fuse all signals into a list of :class:`GamingEvent`.

    Args:
        frame_paths: ``[(timestamp, path)]`` for the clip's
            extracted frames.
        audio_envelope: Optional ``[(timestamp, energy)]`` envelope
            from ``audio_analyzer``; used to confirm visual spikes.
            ``None`` disables audio fusion (visual signals still
            fire on their own).
        hud_layout: ``game_layouts.get_hud_layout`` output for the
            game. ``None`` disables HUD-region signals.
        crosshair_path: Optional list of ``CrosshairFrame`` from
            Phase 1's tracker. Drives the ``locked_on`` signal.
        gameplay_subtype: ``fps`` / ``moba`` / ``tps`` / ``racing``;
            controls which signals are enabled.
        read_patch: Hook for tests; defaults to cv2 pixel read.

    Returns:
        Sorted list of ``GamingEvent``.
    """
    if not USE_GAMING_EVENT_DETECTOR:
        return []
    if not frame_paths:
        return []

    events: list[GamingEvent] = []

    # ── Killfeed ──
    killfeed_bbox = _hud_bbox(hud_layout or {}, "killfeed")
    if killfeed_bbox:
        spikes = detect_hud_diff_spikes(
            frame_paths,
            killfeed_bbox,
            threshold=KILLFEED_DIFF_THRESHOLD,
            min_duration=KILLFEED_MIN_DURATION,
            read_patch=read_patch,
        )
        for s_start, s_end in spikes:
            mid = (s_start + s_end) / 2.0
            target_xy = _bbox_center(killfeed_bbox)
            audio_ok = (
                audio_envelope is None
                or _audio_spike_at(audio_envelope, mid)
            )
            confidence = 0.75 if audio_ok else 0.55
            events.append(GamingEvent(
                timestamp=mid,
                duration=max(s_end - s_start, DEFAULT_EVENT_DURATION),
                kind="kill",
                target_region=killfeed_bbox,
                confidence=confidence,
                source="killfeed_diff" + ("+audio" if audio_ok else ""),
            ))

    # ── Abilities / ult ──
    for abil_key in ("ultimate", "abilities"):
        bbox = _hud_bbox(hud_layout or {}, abil_key)
        if not bbox:
            continue
        spikes = detect_hud_diff_spikes(
            frame_paths,
            bbox,
            threshold=ABILITY_DIFF_THRESHOLD,
            min_duration=KILLFEED_MIN_DURATION,
            read_patch=read_patch,
        )
        for s_start, s_end in spikes:
            mid = (s_start + s_end) / 2.0
            audio_ok = (
                audio_envelope is None
                or _audio_spike_at(audio_envelope, mid)
            )
            if not audio_ok:
                # Ult/ability spikes need audio confirmation —
                # otherwise we trip on every cooldown flash.
                continue
            events.append(GamingEvent(
                timestamp=mid,
                duration=max(s_end - s_start, DEFAULT_EVENT_DURATION),
                kind="ult",
                target_region=bbox,
                confidence=0.7,
                source=f"{abil_key}_diff+audio",
            ))

    # ── Minimap rotations (BR / MOBA) ──
    if (gameplay_subtype or "").lower() in ("moba", "br", "battle_royale", "fps"):
        bbox = _hud_bbox(hud_layout or {}, "minimap")
        if bbox:
            spikes = detect_hud_diff_spikes(
                frame_paths,
                bbox,
                threshold=20.0,  # minimap dot density spikes are softer
                min_duration=1.0,
                read_patch=read_patch,
            )
            for s_start, s_end in spikes:
                mid = (s_start + s_end) / 2.0
                events.append(GamingEvent(
                    timestamp=mid,
                    duration=max(s_end - s_start, 1.2),
                    kind="rotation",
                    target_region=bbox,
                    confidence=0.55,
                    source="minimap_diff",
                ))

    # ── Crosshair lock-on ──
    if crosshair_path:
        events.extend(detect_crosshair_lock_events(
            crosshair_path, audio_envelope,
        ))

    events.sort(key=lambda e: e.timestamp)

    # ── Suppress kills near aim swings ──
    # If a kill event fires while the crosshair is moving fast
    # (>20 %/s), suppress the killfeed pan — the player is
    # already reacting and the camera fighting them feels wrong.
    if crosshair_path:
        events = _suppress_kills_during_aim_swings(events, crosshair_path)

    # ── Merge close events ──
    events = merge_close_events(events)
    return events


def _suppress_kills_during_aim_swings(
    events: list[GamingEvent],
    crosshair_path: list,
    *,
    swing_threshold_pct_per_sec: float = 20.0,
) -> list[GamingEvent]:
    """Drop kill events whose timestamp lands during a fast aim swing."""
    if not events or len(crosshair_path) < 2:
        return events

    sorted_path = sorted(crosshair_path, key=lambda c: c.timestamp)
    out: list[GamingEvent] = []
    for ev in events:
        if ev.kind != "kill":
            out.append(ev)
            continue
        # Find the velocity around ev.timestamp
        before = None
        after = None
        for cf in sorted_path:
            if cf.timestamp <= ev.timestamp:
                before = cf
            elif after is None:
                after = cf
                break
        if before is None or after is None:
            out.append(ev)
            continue
        dt = max(after.timestamp - before.timestamp, 1e-6)
        d = math.hypot(after.x_pct - before.x_pct, after.y_pct - before.y_pct)
        vel = d / dt
        if vel > swing_threshold_pct_per_sec:
            logger.debug(
                "GamingEventDetector: suppressed kill at t=%.2f "
                "(crosshair vel=%.1f %%/s > %.1f)",
                ev.timestamp, vel, swing_threshold_pct_per_sec,
            )
            continue
        out.append(ev)
    return out
