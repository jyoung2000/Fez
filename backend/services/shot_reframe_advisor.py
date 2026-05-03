"""Per-shot reframe strategy advisor.

Phase 1 of the 6-phase reframing overhaul. A deterministic, zero-LLM,
zero-network module that examines each shot's signals and recommends
ONE high-level reframe strategy from
:class:`ReframeStrategy`. Downstream consumers
(``reframe_segmenter.build_reframe_segments`` and
``human_reframe_bridge.maybe_override_render_plan``) read the advice
list to pick the right tunables / layout / fallback for each shot.

Design notes
============

The codebase does NOT define a canonical ``ShotType`` enum (ECU / CU
/ MCU / MS / MW / WS / EWS / TWO_SHOT / OTS / INSERT) at the time of
this module's introduction. The advisor accepts the shot framing tier
as a free-form string and matches against the canonical names
case-insensitively. ``shot_classifier.ShotProfile`` stores
``content_type`` and ``motion_class`` but does not classify framing
tier — callers that want sophisticated routing should compute the
tier (typical signal: face_size as a fraction of frame height) and
pass it in. When unknown, framing-tier branches are skipped and the
advisor falls through to the saliency / face-motion branches.

Naming reconciliation against
:class:`backend.services.content_classifier.ClipContentType`:

* spec ``GAMEPLAY`` / ``STREAM``                  → both kept; ``STREAM``
  is the facecam-on-gameplay variant. The ``MULTI_REGION`` gate fires
  when ``has_facecam=True`` regardless of which gameplay sub-type.
* spec "tutorial/lecture"                          → no enum member.
  The advisor exposes a ``has_screen_or_slides`` input boolean to
  cover this case; it is independent of ``ClipContentType`` and the
  "tutorial / lecture" gate fires on ``GENERIC`` / ``TALKING_HEAD``
  with that flag set.
* spec ``TALKING_HEAD``                            → matches enum.
* spec ``ANIMATION``                               → matches enum
  (``ANIMATION_DIALOGUE`` is treated the same way for the override).
* spec ``MUSIC_VIDEO``                             → matches enum.
* spec ``SPORTS``                                  → matches enum
  (sub-types ``SPORTS_BASKETBALL`` / ``SPORTS_RACING`` are also
  treated as ``SPORTS`` for the override).

The module is pure: every function is deterministic and uses only
numpy + python stdlib. No I/O. No model calls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Optional, Sequence, Tuple

try:  # numpy is already a dep of the project.
    import numpy as np
except ImportError:  # pragma: no cover - numpy always present in this repo
    np = None  # type: ignore

# Best-effort import — used only for genre overrides. Falling back to
# string comparisons keeps the advisor importable in slimmed contexts.
try:
    from backend.services.content_classifier import ClipContentType
except Exception:  # pragma: no cover
    ClipContentType = None  # type: ignore


# ── Public enum ─────────────────────────────────────────────────────


class ReframeStrategy(str, Enum):
    """High-level per-shot reframe strategy.

    Distinct from ``content_type_config.ReframeStrategy`` (which is a
    layout-level enum: STATIONARY / TRACKING / SPLIT_SCREEN / ...).
    The advisor's strategies sit one level higher: they describe the
    *editorial intent* for the shot. The downstream segmenter and
    layout picker map advisor strategies to layout-level strategies.
    """

    STATIC_CENTER = "static_center"
    SUBJECT_TRACKING = "subject_tracking"
    SPEAKER_ALTERNATING = "speaker_alternating"
    MULTI_REGION = "multi_region"
    CONTEXTUAL_PAN = "contextual_pan"
    BLUR_FILL_PRESERVE = "blur_fill_preserve"


BBox = Tuple[float, float, float, float]
"""(x_center, y_center, width, height) — fractions of source frame in [0, 1]."""


# ── Input dataclasses ───────────────────────────────────────────────


@dataclass
class FaceSample:
    """One face sample inside a shot.

    Coordinates are 0-1 fractions of the source frame. ``speaker_id``
    is -1 when the active-speaker pipeline did not assign this face.
    """

    timestamp: float
    x_center: float
    y_center: float
    width: float
    height: float
    track_id: int = -1
    speaker_id: int = -1
    is_speaking: bool = False


@dataclass
class SaliencyPeak:
    """One top-K saliency peak inside a shot, in 0-1 frame fractions."""

    timestamp: float
    x_center: float
    y_center: float
    weight: float = 1.0


@dataclass
class TextRegionInput:
    """One OCR text region to protect, in 0-1 frame fractions."""

    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class ShotAnalysis:
    """Bundle of per-shot signals used by ``recommend_strategy``.

    ``shot_type`` is a free-form framing-tier string (case-insensitive
    match against ECU / CU / MCU / MS / MW / WS / EWS / TWO_SHOT / OTS
    / INSERT). ``content_type`` is a ClipContentType value or its
    string name; both are accepted.
    """

    shot_idx: int
    shot_type: str
    content_type: object  # ClipContentType | str
    shot_duration_sec: float
    source_width: int
    source_height: int

    face_samples: List[FaceSample] = field(default_factory=list)
    saliency_peaks: List[SaliencyPeak] = field(default_factory=list)
    text_regions: List[TextRegionInput] = field(default_factory=list)

    # Optional flags lifted from ContentProfile / supplementary detectors.
    has_facecam: bool = False
    has_screen_or_slides: bool = False


# ── Output dataclass ────────────────────────────────────────────────


@dataclass
class ShotReframeAdvice:
    """Advisor output for a single shot."""

    strategy: ReframeStrategy
    confidence: float
    primary_subject_bbox: Optional[BBox]
    secondary_subjects: List[BBox]
    text_regions_to_protect: List[BBox]
    genre_override_applied: Optional[str]
    fallback_strategy: ReframeStrategy
    shot_idx: int = -1


# ── Genre overrides ─────────────────────────────────────────────────


def _content_key(content_type) -> str:
    """Normalize ContentType-like objects into a comparable string."""
    if content_type is None:
        return ""
    if hasattr(content_type, "value"):
        return str(content_type.value)
    return str(content_type)


# Override table keyed by ClipContentType *value strings* so we can
# compare against either the enum or its string name.
GENRE_OVERRIDES = {
    "talking_head": {
        "force_static_for_shot_types": {"ECU", "CU", "MCU"},
    },
    "cinematic_dialogue": {
        # Same family as talking_head — close-ups should hold steady.
        "force_static_for_shot_types": {"ECU", "CU", "MCU"},
    },
    "animation": {
        "force_static_for_shot_types": {"ECU", "CU", "MCU", "MS"},
    },
    "animation_dialogue": {
        "force_static_for_shot_types": {"ECU", "CU", "MCU", "MS"},
    },
    "music_video": {
        "block_strategies": {ReframeStrategy.SPEAKER_ALTERNATING.value},
    },
    "sports": {
        "remap": {
            ReframeStrategy.BLUR_FILL_PRESERVE.value:
                ReframeStrategy.SUBJECT_TRACKING.value,
        },
    },
    "sports_basketball": {
        "remap": {
            ReframeStrategy.BLUR_FILL_PRESERVE.value:
                ReframeStrategy.SUBJECT_TRACKING.value,
        },
    },
    "sports_racing": {
        "remap": {
            ReframeStrategy.BLUR_FILL_PRESERVE.value:
                ReframeStrategy.SUBJECT_TRACKING.value,
        },
    },
    "gameplay": {
        "block_strategies": {ReframeStrategy.CONTEXTUAL_PAN.value},
    },
    "gameplay_moba": {
        "block_strategies": {ReframeStrategy.CONTEXTUAL_PAN.value},
    },
    "gameplay_tps": {
        "block_strategies": {ReframeStrategy.CONTEXTUAL_PAN.value},
    },
    "gameplay_racing": {
        "block_strategies": {ReframeStrategy.CONTEXTUAL_PAN.value},
    },
}


# ── Internal helpers ────────────────────────────────────────────────


_WIDE_SHOT_TYPES = {"WS", "MW", "EWS"}
_ESTABLISHING_SHOT_TYPES = {"EWS"}
_TUTORIAL_CONTENT_KEYS = {
    "tutorial",
    "lecture",
    "talking_head",   # tutorial/lecture often classifies as talking_head
    "generic",
}


def _norm_shot_type(s: str) -> str:
    return (s or "").strip().upper()


def _faces_by_track(samples: Sequence[FaceSample]) -> dict:
    out: dict = {}
    for f in samples:
        out.setdefault(int(f.track_id), []).append(f)
    return out


def _faces_by_speaker(samples: Sequence[FaceSample]) -> dict:
    out: dict = {}
    for f in samples:
        sid = int(f.speaker_id)
        if sid < 0:
            continue
        out.setdefault(sid, []).append(f)
    return out


def _bbox_from_face(f: FaceSample) -> BBox:
    return (float(f.x_center), float(f.y_center),
            float(f.width), float(f.height))


def _avg_bbox(samples: Sequence[FaceSample]) -> Optional[BBox]:
    if not samples:
        return None
    n = float(len(samples))
    return (
        sum(f.x_center for f in samples) / n,
        sum(f.y_center for f in samples) / n,
        sum(f.width for f in samples) / n,
        sum(f.height for f in samples) / n,
    )


def _detect_alternating_speakers(
    samples: Sequence[FaceSample],
    *,
    max_gap_sec: float = 3.0,
) -> Tuple[bool, List[int]]:
    """Return ``(alternates, ordered_speaker_ids)``.

    Two speakers are considered alternating when:

    1. There are at least 2 distinct speaker_ids that ever speak, AND
    2. They DO NOT speak in the same frame (no temporal overlap), AND
    3. The maximum gap between consecutive turns is < ``max_gap_sec``.

    "Speak in the same frame" is a strong overlap signal — if both
    voices are on at the same time we're either in a debate or a
    cross-talk segment, neither of which benefits from a snap-to-
    speaker layout.
    """
    speaking = [f for f in samples if f.is_speaking and f.speaker_id >= 0]
    if not speaking:
        return False, []

    # Overlap check: any timestamp where >= 2 distinct speakers speak.
    by_ts: dict = {}
    for f in speaking:
        ts_key = round(f.timestamp, 3)
        by_ts.setdefault(ts_key, set()).add(f.speaker_id)
    for sids in by_ts.values():
        if len(sids) >= 2:
            return False, sorted({f.speaker_id for f in speaking})

    speaking.sort(key=lambda f: f.timestamp)
    turns: List[Tuple[float, int]] = []
    last_sid = -2
    for f in speaking:
        if f.speaker_id != last_sid:
            turns.append((f.timestamp, f.speaker_id))
            last_sid = f.speaker_id

    distinct = sorted({sid for _, sid in turns})
    if len(distinct) < 2 or len(turns) < 2:
        return False, distinct

    max_gap = 0.0
    for (t0, _), (t1, _) in zip(turns[:-1], turns[1:]):
        max_gap = max(max_gap, t1 - t0)

    return (max_gap < max_gap_sec), distinct


def _dominant_speaker_track(
    samples: Sequence[FaceSample],
) -> Optional[int]:
    """Return the speaker_id with the most speaking samples, or None."""
    counts: dict = {}
    for f in samples:
        if f.is_speaking and f.speaker_id >= 0:
            counts[f.speaker_id] = counts.get(f.speaker_id, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _face_x_motion_pct(samples: Sequence[FaceSample]) -> float:
    """Range of face x_center across the shot, as a 0-1 fraction."""
    if not samples:
        return 0.0
    xs = [f.x_center for f in samples]
    return max(xs) - min(xs)


def _saliency_x_std(peaks: Sequence[SaliencyPeak]) -> float:
    if not peaks:
        return 0.0
    xs = [p.x_center for p in peaks]
    if len(xs) < 2:
        return 0.0
    mean = sum(xs) / len(xs)
    variance = sum((x - mean) ** 2 for x in xs) / len(xs)
    return math.sqrt(variance)


def _saliency_centroid(peaks: Sequence[SaliencyPeak]) -> Optional[BBox]:
    if not peaks:
        return None
    total_w = sum(max(p.weight, 0.0) for p in peaks) or 1.0
    cx = sum(p.x_center * max(p.weight, 0.0) for p in peaks) / total_w
    cy = sum(p.y_center * max(p.weight, 0.0) for p in peaks) / total_w
    # Synthetic bbox: use a 30% width/height heuristic so downstream
    # consumers always have something to crop around.
    return (cx, cy, 0.30, 0.40)


def _highest_saliency_bbox(peaks: Sequence[SaliencyPeak]) -> Optional[BBox]:
    if not peaks:
        return None
    top = max(peaks, key=lambda p: p.weight)
    return (top.x_center, top.y_center, 0.30, 0.40)


def _to_protect_bboxes(regions: Sequence[TextRegionInput]) -> List[BBox]:
    return [(r.x_center, r.y_center, r.width, r.height) for r in regions]


def _is_tutorial_or_lecture(ct_key: str, has_screen: bool) -> bool:
    return has_screen and ct_key in _TUTORIAL_CONTENT_KEYS


# ── Core decision logic ─────────────────────────────────────────────


def _base_strategy(
    shot: ShotAnalysis,
) -> Tuple[ReframeStrategy, float, Optional[BBox], List[BBox]]:
    """Run the spec's base decision tree.

    Returns ``(strategy, confidence, primary_bbox, secondary_bboxes)``
    BEFORE genre overrides are applied.
    """
    ct_key = _content_key(shot.content_type)
    shot_type = _norm_shot_type(shot.shot_type)
    duration = float(shot.shot_duration_sec or 0.0)

    samples = list(shot.face_samples or [])
    peaks = list(shot.saliency_peaks or [])

    # 1. Multi-region: gameplay/stream + facecam.
    if ct_key in ("gameplay", "gameplay_moba", "gameplay_tps",
                  "gameplay_racing", "stream") and shot.has_facecam:
        primary = _avg_bbox(samples) if samples else None
        return ReframeStrategy.MULTI_REGION, 0.95, primary, []

    # 2. Tutorial/lecture + screen/slides.
    if _is_tutorial_or_lecture(ct_key, shot.has_screen_or_slides):
        primary = _avg_bbox(samples) if samples else None
        return ReframeStrategy.MULTI_REGION, 0.85, primary, []

    # 3 & 4. Establishing wide shots.
    if shot_type in _ESTABLISHING_SHOT_TYPES:
        if duration > 2.0:
            return (
                ReframeStrategy.CONTEXTUAL_PAN,
                0.80,
                _highest_saliency_bbox(peaks),
                [],
            )
        return ReframeStrategy.BLUR_FILL_PRESERVE, 0.75, None, []

    # 5. Speaker count check.
    by_speaker = _faces_by_speaker(samples)
    if len(by_speaker) >= 2:
        alternates, _ = _detect_alternating_speakers(samples)
        if alternates:
            # Build secondary bboxes for the two most-active speakers.
            sorted_speakers = sorted(
                by_speaker.items(),
                key=lambda kv: -sum(1 for f in kv[1] if f.is_speaking),
            )
            primary = _avg_bbox(sorted_speakers[0][1])
            secondary = [
                bb for bb in (
                    _avg_bbox(s[1]) for s in sorted_speakers[1:3]
                ) if bb is not None
            ]
            return ReframeStrategy.SPEAKER_ALTERNATING, 0.90, primary, secondary
        # Two+ speakers but they overlap or one dominates → SUBJECT_TRACKING
        # on the dominant speaker (or, failing that, the most-tracked face).
        dom_sid = _dominant_speaker_track(samples)
        if dom_sid is not None:
            primary = _avg_bbox(by_speaker[dom_sid])
        else:
            # Fallback: pick the speaker with the most samples.
            dom_kv = max(by_speaker.items(), key=lambda kv: len(kv[1]))
            primary = _avg_bbox(dom_kv[1])
        return ReframeStrategy.SUBJECT_TRACKING, 0.75, primary, []

    # 6 & 7. Single face.
    by_track = _faces_by_track(samples) if samples else {}
    distinct_tracks = [k for k in by_track if k != -1]
    # Treat "exactly 1 face" as either: (a) one explicit track id, or
    # (b) only one face per frame on average.
    avg_faces_per_frame = (
        len(samples) / max(1, len({round(f.timestamp, 3) for f in samples}))
        if samples else 0.0
    )
    if (len(distinct_tracks) == 1 or
            (samples and avg_faces_per_frame <= 1.05)):
        primary = _avg_bbox(samples)
        face_x_range = _face_x_motion_pct(samples)
        center_x = primary[0] if primary else 0.5
        if abs(center_x - 0.5) <= 0.20 and face_x_range <= 0.15:
            return ReframeStrategy.STATIC_CENTER, 0.85, primary, []
        if face_x_range > 0.15:
            return ReframeStrategy.SUBJECT_TRACKING, 0.85, primary, []
        # Off-center but stationary → tracking (so the crop snaps to subject).
        return ReframeStrategy.SUBJECT_TRACKING, 0.70, primary, []

    # 8 & 9. No faces: use saliency std.
    if not samples and peaks:
        std = _saliency_x_std(peaks)
        if std < 0.10:
            return (
                ReframeStrategy.STATIC_CENTER,
                0.70,
                _saliency_centroid(peaks),
                [],
            )
        if std > 0.25:
            return (
                ReframeStrategy.CONTEXTUAL_PAN,
                0.65,
                _highest_saliency_bbox(peaks),
                [],
            )

    # 10. Default — subject tracking on the highest-weight saliency
    # region (or center if there is nothing to anchor on).
    primary = _highest_saliency_bbox(peaks) or _avg_bbox(samples)
    return ReframeStrategy.SUBJECT_TRACKING, 0.50, primary, []


def _apply_genre_overrides(
    strategy: ReframeStrategy,
    shot: ShotAnalysis,
) -> Tuple[ReframeStrategy, Optional[str]]:
    """Apply :data:`GENRE_OVERRIDES`, returning the final strategy and
    a label describing the override that fired (or ``None``)."""
    ct_key = _content_key(shot.content_type)
    rules = GENRE_OVERRIDES.get(ct_key)
    if not rules:
        return strategy, None

    shot_type = _norm_shot_type(shot.shot_type)

    # force_static_for_shot_types: snap to STATIC_CENTER on the listed
    # framing tiers, regardless of base. We record the override label
    # even when the base already produced STATIC_CENTER, so debug
    # tooling can attribute the lock to genre policy rather than
    # geometry alone.
    forced = rules.get("force_static_for_shot_types")
    if forced and shot_type in forced:
        return (
            ReframeStrategy.STATIC_CENTER,
            f"{ct_key}:force_static[{shot_type}]",
        )

    # block_strategies: if the base strategy is in the blocklist,
    # remap to SUBJECT_TRACKING (a safe, almost-always-acceptable
    # fallback).
    blocked = rules.get("block_strategies")
    if blocked and strategy.value in blocked:
        return (
            ReframeStrategy.SUBJECT_TRACKING,
            f"{ct_key}:block[{strategy.value}]",
        )

    # remap: explicit base→override mapping.
    remap = rules.get("remap")
    if remap and strategy.value in remap:
        new_val = remap[strategy.value]
        return (
            ReframeStrategy(new_val),
            f"{ct_key}:remap[{strategy.value}->{new_val}]",
        )

    return strategy, None


def _fallback_for(strategy: ReframeStrategy) -> ReframeStrategy:
    """Pick a safe fallback strategy. The chosen fallback is the one
    used when the primary fails coverage checks downstream — it is
    NEVER the same as ``strategy``."""
    table = {
        ReframeStrategy.STATIC_CENTER: ReframeStrategy.SUBJECT_TRACKING,
        ReframeStrategy.SUBJECT_TRACKING: ReframeStrategy.STATIC_CENTER,
        ReframeStrategy.SPEAKER_ALTERNATING: ReframeStrategy.SUBJECT_TRACKING,
        ReframeStrategy.MULTI_REGION: ReframeStrategy.SUBJECT_TRACKING,
        ReframeStrategy.CONTEXTUAL_PAN: ReframeStrategy.BLUR_FILL_PRESERVE,
        ReframeStrategy.BLUR_FILL_PRESERVE: ReframeStrategy.STATIC_CENTER,
    }
    return table[strategy]


# ── Public API ──────────────────────────────────────────────────────


def recommend_strategy(shot: ShotAnalysis) -> ShotReframeAdvice:
    """Run the full base + genre-override decision pipeline for one shot.

    Pure: no I/O, no model calls.
    """
    strategy, confidence, primary, secondary = _base_strategy(shot)
    final, override_label = _apply_genre_overrides(strategy, shot)

    # If genre override fired, keep the same primary subject (it was
    # picked by the base logic) but give a small confidence boost or
    # cut depending on the override kind.
    if override_label is not None:
        if override_label.startswith(_content_key(shot.content_type) + ":force_static"):
            confidence = max(confidence, 0.85)
        elif "block" in override_label:
            confidence = min(confidence, 0.65)

    return ShotReframeAdvice(
        strategy=final,
        confidence=float(confidence),
        primary_subject_bbox=primary,
        secondary_subjects=list(secondary),
        text_regions_to_protect=_to_protect_bboxes(shot.text_regions),
        genre_override_applied=override_label,
        fallback_strategy=_fallback_for(final),
        shot_idx=int(shot.shot_idx),
    )


def _build_shot_analyses(
    shots: Sequence,
    content_profile,
    face_tracks: Sequence,
    speaker_events: Sequence,
    saliency_data: Sequence,
    text_regions: Sequence,
    source_w: int,
    source_h: int,
) -> List[ShotAnalysis]:
    """Construct a ``ShotAnalysis`` per shot from raw pipeline inputs.

    Accepts heterogeneous shapes (the pipeline's existing wiring uses a
    mix of dataclasses and lists); falls back to attribute-getters with
    safe defaults so a partial input never crashes the advisor.
    """
    sw = max(int(source_w or 1), 1)
    sh = max(int(source_h or 1), 1)

    has_facecam = bool(getattr(content_profile, "has_facecam", False))
    # has_screen_or_slides: optional; treat a non-empty hud_regions list
    # as "this looks like a slide deck / shared screen overlay".
    hud_regions = getattr(content_profile, "hud_regions", None) or []
    has_screen_or_slides = bool(getattr(
        content_profile, "has_screen_or_slides", False,
    )) or len(hud_regions) > 0

    # Derive the per-clip ClipContentType once. Allow ContentProfile,
    # ClipContentType, or raw string.
    if hasattr(content_profile, "content_type"):
        ct = getattr(content_profile, "content_type", "")
    else:
        ct = content_profile

    # Pre-bucket the per-frame inputs by shot start/end for O(N) lookup.
    text_inputs = [
        TextRegionInput(
            x_center=_to_frac(getattr(r, "x_pct", getattr(r, "x_center", 0.0)), 100.0),
            y_center=_to_frac(getattr(r, "y_pct", getattr(r, "y_center", 0.0)), 100.0),
            width=_to_frac(getattr(r, "w_pct", getattr(r, "width", 0.0)), 100.0),
            height=_to_frac(getattr(r, "h_pct", getattr(r, "height", 0.0)), 100.0),
        )
        for r in (text_regions or [])
    ]

    out: List[ShotAnalysis] = []
    for idx, shot in enumerate(shots or []):
        start = float(getattr(shot, "start", 0.0))
        end = float(getattr(shot, "end", 0.0))
        duration = max(end - start, 0.0)
        # The advisor's free-form framing tier: prefer attribute, else "".
        shot_type = str(
            getattr(shot, "shot_type", "")
            or getattr(shot, "framing_tier", "")
            or ""
        )

        # Filter face_tracks into this shot.
        face_samples = _filter_faces_for_shot(face_tracks, start, end, sw, sh)
        # Annotate is_speaking from speaker events.
        _apply_speaker_events(face_samples, speaker_events, start, end)

        # Saliency.
        peaks = _filter_saliency_for_shot(saliency_data, start, end)

        out.append(ShotAnalysis(
            shot_idx=idx,
            shot_type=shot_type,
            content_type=ct,
            shot_duration_sec=duration,
            source_width=sw,
            source_height=sh,
            face_samples=face_samples,
            saliency_peaks=peaks,
            text_regions=text_inputs,
            has_facecam=has_facecam,
            has_screen_or_slides=has_screen_or_slides,
        ))
    return out


def advise_all_shots(
    shots: Sequence,
    content_profile,
    face_tracks: Sequence,
    speaker_events: Sequence,
    saliency_data: Sequence,
    text_regions: Sequence,
    source_w: int,
    source_h: int,
) -> List[ShotReframeAdvice]:
    """Run :func:`recommend_strategy` on every shot in ``shots``.

    The pipeline integration point. Designed to be safe even when some
    inputs are missing (defaults to empty lists). Cost: O(N_shots *
    avg signals/shot); typical < 50 ms for a 10-min clip.
    """
    analyses = _build_shot_analyses(
        shots, content_profile, face_tracks, speaker_events,
        saliency_data, text_regions, source_w, source_h,
    )
    return [recommend_strategy(a) for a in analyses]


# ── Pipeline-shape adapters ─────────────────────────────────────────


def _to_frac(val: float, denom: float) -> float:
    """Normalize a possibly-percent value to a 0-1 fraction.

    Heuristic: anything > 1.5 is assumed to be 0-100 percent and is
    divided by ``denom`` (typically 100). Already-fractional values
    pass through unchanged.
    """
    if val is None:
        return 0.0
    v = float(val)
    if v > 1.5:
        return v / float(denom or 1.0)
    return v


def _filter_faces_for_shot(
    face_tracks: Sequence,
    start: float,
    end: float,
    sw: int,
    sh: int,
) -> List[FaceSample]:
    """Convert a ``list[FrameFaces]`` (or pre-built ``list[FaceSample]``)
    into ``FaceSample``s falling inside ``[start, end)``."""
    out: List[FaceSample] = []
    for fr in face_tracks or []:
        # Already a FaceSample? Pass through if in range.
        if isinstance(fr, FaceSample):
            if start <= fr.timestamp < end:
                out.append(fr)
            continue
        ts = float(getattr(fr, "timestamp", 0.0))
        if not (start <= ts < end):
            continue
        faces = getattr(fr, "faces", None) or []
        for f in faces:
            out.append(FaceSample(
                timestamp=ts,
                x_center=_to_frac(getattr(f, "x_center", getattr(f, "nose_x", 50.0)), 100.0),
                y_center=_to_frac(getattr(f, "y_center", getattr(f, "nose_y", 50.0)), 100.0),
                width=_to_frac(getattr(f, "width", 10.0), 100.0),
                height=_to_frac(getattr(f, "height", 12.0), 100.0),
                track_id=int(getattr(f, "identity_id", -1)),
                speaker_id=int(getattr(f, "speaker_id", getattr(f, "identity_id", -1))),
                is_speaking=bool(getattr(f, "is_speaking", False)),
            ))
    return out


def _apply_speaker_events(
    samples: List[FaceSample],
    speaker_events: Sequence,
    start: float,
    end: float,
) -> None:
    """Mutate ``samples`` in place: set ``is_speaking`` and
    ``speaker_id`` from ``SpeakerEvent``-shaped entries."""
    if not samples or not speaker_events:
        return
    relevant = []
    for ev in speaker_events:
        ev_start = float(getattr(ev, "start", 0.0))
        ev_end = float(getattr(ev, "end", 0.0))
        if ev_end < start or ev_start > end:
            continue
        relevant.append((
            ev_start, ev_end,
            int(getattr(ev, "slot_id", -1)),
        ))
    if not relevant:
        return
    for s in samples:
        for ev_start, ev_end, slot_id in relevant:
            if ev_start <= s.timestamp <= ev_end:
                s.is_speaking = True
                if s.speaker_id < 0:
                    s.speaker_id = slot_id
                break


def _filter_saliency_for_shot(
    saliency_data: Sequence,
    start: float,
    end: float,
) -> List[SaliencyPeak]:
    out: List[SaliencyPeak] = []
    for s in saliency_data or []:
        if isinstance(s, SaliencyPeak):
            if start <= s.timestamp < end:
                out.append(s)
            continue
        ts = float(getattr(s, "timestamp", 0.0))
        if not (start <= ts < end):
            continue
        out.append(SaliencyPeak(
            timestamp=ts,
            x_center=_to_frac(getattr(s, "x", getattr(s, "x_center", 50.0)), 100.0),
            y_center=_to_frac(getattr(s, "y", getattr(s, "y_center", 50.0)), 100.0),
            weight=float(getattr(s, "saliency_score", getattr(s, "weight", 1.0))),
        ))
    return out


__all__ = [
    "ReframeStrategy",
    "FaceSample",
    "SaliencyPeak",
    "TextRegionInput",
    "ShotAnalysis",
    "ShotReframeAdvice",
    "GENRE_OVERRIDES",
    "recommend_strategy",
    "advise_all_shots",
]
