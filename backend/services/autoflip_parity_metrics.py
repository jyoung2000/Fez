"""Phase 9 — AutoFlip parity metrics (pure-Python, no numpy / cv2 / ffmpeg).

These functions take simple Python data (lists of times, lists of crop
centers, ground-truth bbox tracks) and emit floats / ints / dicts.
They are deliberately decoupled from the reframe pipeline so they can
be unit-tested in isolation without the full numpy + OpenCV +
MediaPipe stack — every Phase-X PR can land its quality numbers by
running the runner script in docker and pasting the JSON into the
results doc, but the metric definitions themselves are testable in a
30-line sandbox.

The 8 metrics here implement the spec from the v2 plan:

    1. sub_second_switch_recall        — speaker change responsiveness
    2. overlap_count                   — segment-boundary invariant
    3. max_acceleration                — |Δ²x| on the crop path
    4. max_jerk                        — |Δ³x| on the crop path
    5. required_region_miss_rate       — % frames with a required bbox out of crop
    6. downbeat_snap_error             — music-video beat alignment
    7. hud_preservation_rate           — gaming HUD-zone visibility
    8. face_centroid_in_thirds_rate    — narrative / vlog rule-of-thirds bias

Coordinate convention: positions and bboxes are in **percentage of
source frame width** (0–100), matching the rest of the reframe
pipeline. Crop widths are also in % of source width. Timestamps are
in seconds.

Each function is total — empty-input edge cases return a sensible
default (1.0 for "everything passed", 0.0 for "no signal", or a dict
with ``None`` placeholders) so the runner can score every applicable
metric for every fixture without crashing on missing data.
"""

from __future__ import annotations

from statistics import median as _median
from typing import Iterable, Optional


# ── Helpers ─────────────────────────────────────────────────────────

def extract_switch_times(
    segments: Iterable,
    slot_attr: str = "active_slot",
    start_attr: str = "start",
) -> list[float]:
    """Extract the start times of segments where the active slot changes.

    Mirrors the ``_find_crop_switch_time`` helper in
    ``measure_reframe_lag.py`` but iterates the whole sequence so the
    parity runner gets every transition in one pass. Returns an empty
    list when no switches exist (e.g. a video with one speaker).

    The function is tolerant of mixed-attribute objects — segments
    that don't expose ``slot_attr`` simply contribute ``None`` and
    don't count toward switches against other ``None`` neighbors.
    """
    switches: list[float] = []
    last_slot = _SENTINEL
    for seg in segments:
        slot = getattr(seg, slot_attr, None)
        if last_slot is not _SENTINEL and slot != last_slot:
            try:
                switches.append(float(getattr(seg, start_attr)))
            except (TypeError, ValueError):
                pass
        last_slot = slot
    return switches


def extract_segment_boundaries(
    segments: Iterable,
    *,
    start_attr: str = "start",
    skip_zero: bool = True,
) -> list[float]:
    """Extract every segment ``start`` time (excluding t=0 by default).

    Distinct from ``extract_switch_times`` in that it counts EVERY
    segment boundary as a "cut", regardless of whether the active
    slot changed. Phase 5's pulse cuts produce new segments with
    the SAME active speaker — they're visual cuts (re-anchors)
    that ``extract_switch_times`` would miss.

    Used by the ``downbeat_snap_error`` metric so a music video
    fixture that pulse-cuts on every downbeat scores
    ``snap_rate == 1.0`` even though no speaker change occurred.
    """
    out: list[float] = []
    for seg in segments:
        raw = getattr(seg, start_attr, None)
        if raw is None:
            continue
        try:
            t = float(raw)
        except (TypeError, ValueError):
            continue
        if skip_zero and t <= 0.0:
            continue
        out.append(t)
    return out


_SENTINEL = object()  # used by extract_switch_times to distinguish first iteration


# ── Metric 1: sub-second speaker-change recall ──────────────────────

def sub_second_switch_recall(
    expected_switches: list[float],
    actual_switches: list[float],
    tolerance_sec: float = 0.5,
) -> float:
    """Fraction of expected speaker-change events the crop responded to.

    A "responded" expected switch is one where at least one actual
    switch lies within ``tolerance_sec`` of the expected time. The
    default 0.5 s tolerance matches the spec's "sub-second switch
    recall" metric — anything tighter is up to the per-fixture caller.

    Returns 1.0 when ``expected_switches`` is empty (vacuous-true: no
    expectations to miss). Returns 0.0 when expected switches exist
    but none were matched.
    """
    if not expected_switches:
        return 1.0
    if not actual_switches:
        return 0.0
    matched = 0
    for exp in expected_switches:
        if any(abs(act - exp) <= tolerance_sec for act in actual_switches):
            matched += 1
    return matched / len(expected_switches)


# ── Metric 2: segment-overlap count ─────────────────────────────────

def overlap_count(segments: list[tuple[float, float]]) -> int:
    """Count adjacent segments where ``end_i > start_{i+1}``.

    The reframe pipeline guarantees half-open contiguous segments
    (``end_i == start_{i+1}``) — the overlap count must stay at zero
    across every phase. Any positive return is a regression.

    Accepts either ``(start, end)`` tuples or any object with
    ``.start`` / ``.end`` attributes.
    """
    if not segments:
        return 0
    spans = [_get_span(s) for s in segments]
    return sum(1 for i in range(len(spans) - 1) if spans[i][1] > spans[i + 1][0] + 1e-9)


def _get_span(seg) -> tuple[float, float]:
    if isinstance(seg, tuple) and len(seg) >= 2:
        return float(seg[0]), float(seg[1])
    return float(getattr(seg, "start")), float(getattr(seg, "end"))


# ── Metric 3 & 4: smoothness on the crop path ───────────────────────

def max_acceleration(crop_centers: list[float]) -> float:
    """Max ``|x[i+1] - 2·x[i] + x[i-1]|`` over the crop center sequence.

    Phase 0 (the LP solver work) baselined this at ≤ 3.0 px on the
    linear-pan fixture. Each v2 phase must keep it under that ceiling
    or feature-flag the regressing change off by default.
    """
    n = len(crop_centers)
    if n < 3:
        return 0.0
    return max(
        abs(crop_centers[i + 1] - 2.0 * crop_centers[i] + crop_centers[i - 1])
        for i in range(1, n - 1)
    )


def max_jerk(crop_centers: list[float]) -> float:
    """Max ``|x[i+1] - 3·x[i] + 3·x[i-1] - x[i-2]|`` over the crop path.

    Baselined at ≤ 3.0 px on the tracking-shot fixture by Phase 0. The
    metric is a discrete third-difference operator and approximates the
    classical smoothness-of-motion penalty used by AutoFlip's LP.
    """
    n = len(crop_centers)
    if n < 4:
        return 0.0
    return max(
        abs(
            crop_centers[i + 1]
            - 3.0 * crop_centers[i]
            + 3.0 * crop_centers[i - 1]
            - crop_centers[i - 2]
        )
        for i in range(2, n - 1)
    )


# ── Metric 5: required-region miss rate ─────────────────────────────

def required_region_miss_rate(
    crop_centers: list[float],
    required_regions_per_frame: list[list[tuple[float, float]]],
    crop_width_pct: float,
) -> float:
    """Fraction of frames where a required region (face / sub bar / HUD)
    falls outside the cropping window.

    ``crop_centers`` is a per-frame list of crop center x in % (0–100).
    ``required_regions_per_frame`` is the same length; each element is
    a list of ``(left_pct, right_pct)`` bboxes that MUST be in-frame
    on that frame. ``crop_width_pct`` is the crop width as % of the
    source frame width.

    A frame is counted as a miss the first time any required region
    has either endpoint outside the crop bounds (with a 0.01 % epsilon
    for floating-point slop). Returns 0.0 when there are no frames or
    no required regions on any frame (vacuous-true).
    """
    if not crop_centers:
        return 0.0
    if not required_regions_per_frame:
        return 0.0
    half = crop_width_pct / 2.0
    misses = 0
    counted = 0
    pairs = list(zip(crop_centers, required_regions_per_frame))
    for cx, regions in pairs:
        if not regions:
            continue
        counted += 1
        crop_left = cx - half
        crop_right = cx + half
        for left, right in regions:
            if left < crop_left - 0.01 or right > crop_right + 0.01:
                misses += 1
                break
    if counted == 0:
        return 0.0
    return misses / counted


# ── Metric 6: downbeat snap error (music video) ─────────────────────

def downbeat_snap_error(
    switch_times: list[float],
    beat_grid: list[float],
    snap_tolerance_ms: float = 200.0,
) -> dict:
    """Distance from each crop-switch to the nearest downbeat.

    Returns a dict with:
      - ``mean_error_ms``  — mean absolute distance, in ms
      - ``max_error_ms``   — largest single distance, in ms
      - ``snap_rate``      — fraction of switches within
        ``snap_tolerance_ms`` (the spec asks for ±200 ms)
      - ``count``          — number of switches measured
      - ``snapped``        — number of switches inside the tolerance

    Returns sentinel ``None`` values when either input is empty so the
    runner can still emit a row without crashing.
    """
    if not switch_times or not beat_grid:
        return {
            "mean_error_ms": None,
            "max_error_ms": None,
            "snap_rate": 0.0,
            "count": len(switch_times),
            "snapped": 0,
        }
    errors_ms: list[float] = []
    snapped = 0
    for sw in switch_times:
        nearest = min(abs(sw - b) for b in beat_grid)
        ms = nearest * 1000.0
        errors_ms.append(ms)
        if ms <= snap_tolerance_ms:
            snapped += 1
    return {
        "mean_error_ms": round(sum(errors_ms) / len(errors_ms), 2),
        "max_error_ms": round(max(errors_ms), 2),
        "snap_rate": round(snapped / len(switch_times), 4),
        "count": len(switch_times),
        "snapped": snapped,
    }


# ── Metric 7: HUD preservation rate (gaming) ────────────────────────

def hud_preservation_rate(
    crop_centers: list[float],
    crop_width_pct: float,
    hud_zones: list[tuple[float, float, float, float]],
    min_visible_fraction: float = 0.8,
) -> float:
    """Fraction of frames where every HUD zone is mostly inside the crop.

    ``hud_zones`` is a list of ``(x_pct, y_pct, w_pct, h_pct)`` boxes
    in source-frame % coordinates — the same shape as
    ``backend.services.game_layouts.GAME_HUD_LAYOUTS`` entries. Phase 7
    will pull the genre-correct list out of that table at runtime.

    A zone is "preserved" on a frame when at least
    ``min_visible_fraction`` (default 80%) of its **horizontal** width
    sits inside the crop window. We don't check vertical visibility
    here because the cropping pipeline operates only in x for 16:9
    → 9:16 conversion (the source y is preserved 1:1).

    Returns 1.0 when there are no HUD zones (vacuous-true) and 0.0
    when there are no crop frames.
    """
    if not crop_centers:
        return 0.0
    if not hud_zones:
        return 1.0
    half = crop_width_pct / 2.0
    preserved = 0
    for cx in crop_centers:
        crop_left = cx - half
        crop_right = cx + half
        all_visible = True
        for hx, _hy, hw, _hh in hud_zones:
            zone_left = hx
            zone_right = hx + hw
            visible_left = max(crop_left, zone_left)
            visible_right = min(crop_right, zone_right)
            visible_w = max(0.0, visible_right - visible_left)
            zone_w = zone_right - zone_left
            if zone_w > 0 and (visible_w / zone_w) < min_visible_fraction:
                all_visible = False
                break
        if all_visible:
            preserved += 1
    return preserved / len(crop_centers)


# ── Metric 8: face-centroid-in-thirds rate ──────────────────────────

def face_centroid_in_thirds_rate(
    face_y_in_crop_normalized: list[float],
    target_thirds: float = 1.0 / 3.0,
    tolerance: float = 0.10,
) -> float:
    """Fraction of frames where the face Y lies near rule-of-thirds.

    ``face_y_in_crop_normalized`` is the per-frame face centroid Y
    expressed as a fraction in ``[0, 1]`` of the **vertical crop**
    (not the source frame). The target is the upper-third line
    (``y = 1/3``) by default, which matches the cinematography
    convention for talking-head and vlog framing.

    A frame "passes" when ``|y - target_thirds| <= tolerance``. The
    spec calls for this metric on narrative + vlog routes; Phase 4
    will tune the actual y-anchor logic.

    Returns 0.0 on empty input — a route that produced no face data
    can't claim thirds compliance.
    """
    if not face_y_in_crop_normalized:
        return 0.0
    in_thirds = sum(
        1
        for y in face_y_in_crop_normalized
        if abs(y - target_thirds) <= tolerance
    )
    return in_thirds / len(face_y_in_crop_normalized)


# ── Metric 9 (Phase A): identity-switch count ───────────────────────


def identity_switch_count(
    timeline: list[dict],
    *,
    iou_threshold: float = 0.30,
    min_gap_sec: float = 0.20,
) -> int:
    """Count tracker identity switches across a multi-subject timeline.

    A "switch" is when a tracker slot's bounding box at frame ``t``
    has IoU below ``iou_threshold`` with the same slot's bbox at
    frame ``t-1``, with at least ``min_gap_sec`` between frames (so
    sub-frame jitter doesn't count). The metric is monotonic — every
    real ID swap counts exactly once.

    Input shape (``timeline`` is per-frame):

        [
            {"t": float, "slots": {slot_id: (x, y, w, h)}, ...},
            ...
        ]

    Coordinates are in any consistent unit (pixels OR % of source).
    The metric only uses IoU, which is unit-agnostic.

    Phase A target: identity-switch rate ≥ 30 % lower than the
    OpenCV-tracker baseline on the four-speaker panel fixture.
    """
    if len(timeline) < 2:
        return 0
    switches = 0
    last_slots: dict = {}
    last_t: Optional[float] = None
    for frame in timeline:
        try:
            t = float(frame.get("t", 0.0))
        except (TypeError, ValueError):
            continue
        slots = frame.get("slots", {}) or {}
        if last_t is not None and (t - last_t) < min_gap_sec:
            # Sub-frame jitter — skip but still update last_slots.
            last_slots = dict(slots)
            last_t = t
            continue
        for slot_id, bbox in slots.items():
            if slot_id in last_slots:
                if _iou(last_slots[slot_id], bbox) < iou_threshold:
                    switches += 1
        last_slots = dict(slots)
        last_t = t
    return switches


def _iou(box_a, box_b) -> float:
    """Pixel-space IoU for two ``(x, y, w, h)`` tuples."""
    if box_a is None or box_b is None:
        return 0.0
    ax, ay, aw, ah = (float(v) for v in box_a)
    bx, by, bw, bh = (float(v) for v in box_b)
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ax1, ay1 = ax + aw, ay + ah
    bx1, by1 = bx + bw, by + bh
    ix0 = max(ax, bx)
    iy0 = max(ay, by)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


# ── Metric 10 (Phase C): saliency-in-crop fraction ──────────────────


def saliency_in_crop_fraction(
    crop_centers: list[float],
    crop_width_pct: float,
    saliency_peaks_per_second: list[list[tuple[float, float, float]]],
    *,
    timestamps: Optional[list[float]] = None,
) -> float:
    """Fraction of frames where ≥1 top-3 saliency peak lies inside the crop.

    Phase C target: ≥ 0.80 on dynamic content (sports / music_video /
    anime). Soft metric — saliency is a hint, not a constraint.

    Coordinates are in % of source frame width to match the rest of
    the parity bench. ``saliency_peaks_per_second[i]`` is a list of
    ``(x, y, score)`` peaks for second ``i``; ``timestamps[j]`` (or
    ``j``-as-second when ``timestamps`` is None) maps each crop frame
    to its peaks-per-second slot.
    """
    if not crop_centers or not saliency_peaks_per_second:
        return 0.0
    half = crop_width_pct / 2.0
    n_in_crop = 0
    for i, cx in enumerate(crop_centers):
        if timestamps is not None:
            sec_idx = int(timestamps[i]) if i < len(timestamps) else i
        else:
            sec_idx = i
        if sec_idx < 0 or sec_idx >= len(saliency_peaks_per_second):
            continue
        peaks = saliency_peaks_per_second[sec_idx]
        if not peaks:
            continue
        for (px, _py, _score) in peaks:
            if cx - half <= px <= cx + half:
                n_in_crop += 1
                break
    return n_in_crop / len(crop_centers)


# ── Metric 11 (Phase D): face-clipping rate ─────────────────────────


def face_clipping_rate(
    crop_centers: list[float],
    crop_width_pct: float,
    face_bboxes_per_frame: list[list[tuple[float, float, float, float]]],
    *,
    margin_pct: float = 0.5,
) -> float:
    """Fraction of frames where any face's bbox extends outside the crop.

    Phase D target: ≥ 50 % reduction vs Phase C baseline. A face is
    "clipped" when more than ``margin_pct`` % of source-width sticks
    out of either crop edge — a tiny tolerance avoids false positives
    on faces that are exactly tangent to the crop edge.

    ``face_bboxes_per_frame[i]`` is the list of ``(x, y, w, h)`` faces
    for crop frame ``i`` in % of source width (the same units as the
    crop center).
    """
    if not crop_centers:
        return 0.0
    half = crop_width_pct / 2.0
    clipped = 0
    for i, cx in enumerate(crop_centers):
        crop_left = cx - half
        crop_right = cx + half
        faces = face_bboxes_per_frame[i] if i < len(face_bboxes_per_frame) else []
        for (fx, _fy, fw, _fh) in faces:
            f_left = fx
            f_right = fx + fw
            if (crop_left - f_left > margin_pct) or (f_right - crop_right > margin_pct):
                clipped += 1
                break
    return clipped / len(crop_centers)


# ── Metric 12 (Phase C): text-region clipping rate ──────────────────


def text_region_clipping_rate(
    crop_centers: list[float],
    crop_width_pct: float,
    text_regions_per_frame: list[list[tuple[float, float, float, float]]],
    *,
    min_visible_fraction: float = 0.95,
) -> float:
    """Fraction of frames where at least one OCR text region is clipped.

    A region is considered clipped when less than ``min_visible_fraction``
    of its horizontal extent sits inside the crop. The default 95 %
    threshold protects readable text — anything tighter cuts mid-letter.

    Phase C target: < 0.05 (≤ 5 % of frames have any text clipped).
    """
    if not crop_centers:
        return 0.0
    half = crop_width_pct / 2.0
    clipped = 0
    for i, cx in enumerate(crop_centers):
        crop_left = cx - half
        crop_right = cx + half
        regions = text_regions_per_frame[i] if i < len(text_regions_per_frame) else []
        for (rx, _ry, rw, _rh) in regions:
            r_left = rx
            r_right = rx + rw
            visible_left = max(crop_left, r_left)
            visible_right = min(crop_right, r_right)
            visible_w = max(0.0, visible_right - visible_left)
            r_w = r_right - r_left
            if r_w > 0 and (visible_w / r_w) < min_visible_fraction:
                clipped += 1
                break
    return clipped / len(crop_centers)


# ── Metric 13: cut-to-hold ratio (Week 3) ────────────────────────────

def cut_to_hold_ratio(events: list[dict]) -> dict:
    """Distribution of per-segment hold durations.

    Week 3 real-content bench metric. Human professional editors hold
    shots a median of ~3-8 seconds on dialogue / panel / narrative
    content; closer to 1.5-3 seconds on fast music / action. AutoFlip
    tends to hold shorter than humans across all content types because
    its optimization is local (scene-cropping calculator operates per
    shot without an editorial prior). The ClipAI editorial prior is
    designed to pull hold distributions toward the human median; this
    metric is what proves it on real content.

    Input shape is the AutoFlip-compatible event list:

        [
            {"t": float, "scene_change": bool, ...},
            ...
        ]

    Events with ``scene_change == True`` mark the start of a new
    segment. Hold duration for segment ``i`` is
    ``next_segment_start - this_segment_start``, with the final
    segment ending at the last event's ``t``. If no event has
    ``scene_change == True`` the entire clip is treated as one
    segment (hold = last - first).

    Returns a dict with:

        n_segments:            int
        median_hold_sec:       float  (0.0 on empty)
        p25, p75, p95:         float  (quantile holds)
        segments_under_1s_rate: float — red flag if high on dialogue
        segments_over_8s_rate:  float — red flag on music / action

    An empty event list returns ``{"n_segments": 0}``.
    """
    if not events:
        return {"n_segments": 0}

    # Segment start times: every event flagged as a scene change.
    segment_starts = [
        float(e["t"]) for e in events if e.get("scene_change")
    ]
    # If no scene-change flag fired, treat the whole clip as one segment.
    if not segment_starts:
        segment_starts = [float(events[0]["t"])]

    last_t = float(events[-1]["t"])
    segment_ends = segment_starts[1:] + [last_t]
    holds = [
        e - s for s, e in zip(segment_starts, segment_ends) if e > s
    ]
    if not holds:
        return {"n_segments": 0}

    holds_sorted = sorted(holds)
    n = len(holds_sorted)

    def _pct(p: float) -> float:
        """Nearest-rank quantile; n*p clamped to last index."""
        idx = min(n - 1, max(0, int(n * p)))
        return holds_sorted[idx]

    return {
        "n_segments": n,
        "median_hold_sec": round(_median(holds_sorted), 3),
        "p25": round(_pct(0.25), 3),
        "p75": round(_pct(0.75), 3),
        "p95": round(_pct(0.95), 3),
        "segments_under_1s_rate": round(
            sum(1 for h in holds_sorted if h < 1.0) / n, 3,
        ),
        "segments_over_8s_rate": round(
            sum(1 for h in holds_sorted if h > 8.0) / n, 3,
        ),
    }


# ── Aggregator: score a fixture ────────────────────────────────────

def score_fixture(
    *,
    metrics_to_run: list[str],
    expected_switches: Optional[list[float]] = None,
    actual_switches: Optional[list[float]] = None,
    actual_segment_boundaries: Optional[list[float]] = None,
    segment_spans: Optional[list[tuple[float, float]]] = None,
    crop_centers: Optional[list[float]] = None,
    required_regions_per_frame: Optional[list[list[tuple[float, float]]]] = None,
    crop_width_pct: Optional[float] = None,
    beat_grid: Optional[list[float]] = None,
    hud_zones: Optional[list[tuple[float, float, float, float]]] = None,
    face_y_in_crop_normalized: Optional[list[float]] = None,
    sub_second_tolerance_sec: float = 0.5,
    downbeat_snap_tolerance_ms: float = 200.0,
    hud_min_visible_fraction: float = 0.8,
    thirds_tolerance: float = 0.10,
    # 2026 SOTA additions — kwargs are optional so existing callers
    # don't have to change.
    identity_timeline: Optional[list[dict]] = None,
    saliency_peaks_per_second: Optional[list[list[tuple[float, float, float]]]] = None,
    saliency_timestamps: Optional[list[float]] = None,
    face_bboxes_per_frame: Optional[list[list[tuple[float, float, float, float]]]] = None,
    text_regions_per_frame: Optional[list[list[tuple[float, float, float, float]]]] = None,
) -> dict:
    """Run every requested metric and return a JSON-friendly dict.

    Unknown metric names are silently skipped (the runner emits a
    warning) so an outdated fixture spec doesn't crash the whole pass.
    """
    out: dict = {}
    for metric in metrics_to_run:
        if metric == "sub_second_switch_recall":
            out[metric] = sub_second_switch_recall(
                expected_switches or [],
                actual_switches or [],
                tolerance_sec=sub_second_tolerance_sec,
            )
        elif metric == "overlap_count":
            out[metric] = overlap_count(segment_spans or [])
        elif metric == "max_acceleration":
            out[metric] = max_acceleration(crop_centers or [])
        elif metric == "max_jerk":
            out[metric] = max_jerk(crop_centers or [])
        elif metric == "required_region_miss_rate":
            if crop_width_pct is None:
                out[metric] = None
            else:
                out[metric] = required_region_miss_rate(
                    crop_centers or [],
                    required_regions_per_frame or [],
                    crop_width_pct=crop_width_pct,
                )
        elif metric == "downbeat_snap_error":
            # Prefer the segment-boundary list (Phase 5: pulse cuts
            # are visual cuts that don't change the active slot, so
            # actual_switches misses them; actual_segment_boundaries
            # captures every cut).
            cut_times = (
                actual_segment_boundaries
                if actual_segment_boundaries is not None
                else actual_switches
            ) or []
            out[metric] = downbeat_snap_error(
                cut_times,
                beat_grid or [],
                snap_tolerance_ms=downbeat_snap_tolerance_ms,
            )
        elif metric == "hud_preservation_rate":
            if crop_width_pct is None:
                out[metric] = None
            else:
                out[metric] = hud_preservation_rate(
                    crop_centers or [],
                    crop_width_pct=crop_width_pct,
                    hud_zones=hud_zones or [],
                    min_visible_fraction=hud_min_visible_fraction,
                )
        elif metric == "face_centroid_in_thirds_rate":
            out[metric] = face_centroid_in_thirds_rate(
                face_y_in_crop_normalized or [],
                tolerance=thirds_tolerance,
            )
        elif metric == "identity_switch_count":
            if identity_timeline is None:
                out[metric] = None
            else:
                out[metric] = identity_switch_count(identity_timeline)
        elif metric == "saliency_in_crop_fraction":
            if crop_width_pct is None or saliency_peaks_per_second is None:
                out[metric] = None
            else:
                out[metric] = saliency_in_crop_fraction(
                    crop_centers or [],
                    crop_width_pct,
                    saliency_peaks_per_second,
                    timestamps=saliency_timestamps,
                )
        elif metric == "face_clipping_rate":
            if crop_width_pct is None or face_bboxes_per_frame is None:
                out[metric] = None
            else:
                out[metric] = face_clipping_rate(
                    crop_centers or [],
                    crop_width_pct,
                    face_bboxes_per_frame,
                )
        elif metric == "text_region_clipping_rate":
            if crop_width_pct is None or text_regions_per_frame is None:
                out[metric] = None
            else:
                out[metric] = text_region_clipping_rate(
                    crop_centers or [],
                    crop_width_pct,
                    text_regions_per_frame,
                )
        else:
            out[metric] = {"error": f"unknown metric {metric!r}"}
    return out


# ── Public API ─────────────────────────────────────────────────────

ALL_METRICS: tuple[str, ...] = (
    "sub_second_switch_recall",
    "overlap_count",
    "max_acceleration",
    "max_jerk",
    "required_region_miss_rate",
    "downbeat_snap_error",
    "hud_preservation_rate",
    "face_centroid_in_thirds_rate",
    # 2026 SOTA additions
    "identity_switch_count",          # Phase A
    "saliency_in_crop_fraction",      # Phase C
    "face_clipping_rate",             # Phase D
    "text_region_clipping_rate",      # Phase C
)
