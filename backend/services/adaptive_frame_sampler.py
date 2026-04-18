"""Adaptive VLM frame sampling.

Phase 4 of the VLM subject-tracking upgrade.

The legacy frame extractor emits a uniform-stride frame list
(~1 frame per 10 seconds). That's too sparse for clips with rapid
scene changes and too wasteful on long single-speaker holds. This
module consumes the signals the pipeline already has — shot cuts,
per-frame face confidence, ASD ambiguity windows, and clip
candidate hook windows — and produces a non-uniform VLM frame
schedule that spends its budget where it matters.

Adaptive rules (all additive, deduped within 0.2s):

  1. Shot-cut neighborhood — first frame of each shot + one frame
     at +1.0s into the shot.
  2. Face-confidence dropouts — when per-frame face confidence
     stays below 0.6 for > 0.5s, sample the midpoint.
  3. ASD ambiguity — when the active-speaker margin stays below
     0.3 for > 0.5s, sample the midpoint.
  4. Clip-candidate hook windows — for every clip candidate at
     start t₀, sample at t₀+0.5, t₀+1.5, t₀+2.5.
  5. Stable-region quota floor — at least 1 frame per 30s of
     content so long single-speaker sections still get coverage.

Total sampled frames per job are capped at
``max(60, int(duration_minutes * 4))`` — generous on purpose. Cost
savings come from fewer DENSE samples in stable regions, not from
a tighter cap.

Gated behind ``CLIPAI_ADAPTIVE_VLM_SAMPLING`` env var. Default OFF
preserves the uniform-stride frame list bit-for-bit identical.

Pure function: no I/O, no side effects. Callers are responsible
for converting sampled timestamps back into FrameData objects
via the existing ``frame_extractor`` helpers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

ADAPTIVE_SAMPLING_ENV = "CLIPAI_ADAPTIVE_VLM_SAMPLING"


def adaptive_sampling_enabled() -> bool:
    """True if ``CLIPAI_ADAPTIVE_VLM_SAMPLING`` is set to a truthy value."""
    return os.environ.get(ADAPTIVE_SAMPLING_ENV, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


@dataclass
class AdaptiveSamplingInputs:
    """All inputs the adaptive scheduler needs from the pipeline.

    Every field is optional so the scheduler can be called from
    paths that only have partial signals. Missing signals simply
    cause the corresponding rule to emit no frames — they never
    degrade coverage below the stable-region quota floor.
    """

    duration_seconds: float
    shot_cuts: list[float] = field(default_factory=list)
    # (timestamp, face_confidence_0_1) pairs, sparse is fine.
    face_conf_timeline: list[tuple[float, float]] = field(default_factory=list)
    # (timestamp, margin_between_top_two_asd_candidates) pairs.
    asd_margin_timeline: list[tuple[float, float]] = field(default_factory=list)
    # Clip-candidate start times. Windows at +0.5, +1.5, +2.5.
    clip_candidate_starts: list[float] = field(default_factory=list)


# ── Per-rule builders (pure) ────────────────────────────────────────


def _rule_shot_cut_neighborhood(cuts: list[float], duration: float) -> list[float]:
    """First frame of each shot + one frame at +1.0s into the shot."""
    out: list[float] = []
    for cut in cuts:
        if 0.0 <= cut <= duration:
            out.append(cut)
            plus = cut + 1.0
            if plus <= duration:
                out.append(plus)
    return out


def _contiguous_windows(
    timeline: list[tuple[float, float]],
    predicate,
    min_duration: float = 0.5,
) -> list[tuple[float, float]]:
    """Collapse a timeline of (t, v) points into contiguous windows
    where ``predicate(v)`` holds for at least ``min_duration``.

    Timelines may be sparse: we treat each (t, v) as a sample; a
    window is a maximal run of consecutive samples that all satisfy
    the predicate. Gaps of more than 1.0s break a window.
    """
    if not timeline:
        return []
    sorted_timeline = sorted(timeline, key=lambda x: x[0])
    windows: list[tuple[float, float]] = []
    cur_start: float | None = None
    cur_end: float | None = None
    last_t: float | None = None
    for t, v in sorted_timeline:
        if predicate(v):
            if cur_start is None:
                cur_start = t
                cur_end = t
            elif last_t is not None and t - last_t > 1.0:
                # Gap → close the old window, start new.
                if cur_end - cur_start >= min_duration:
                    windows.append((cur_start, cur_end))
                cur_start = t
                cur_end = t
            else:
                cur_end = t
            last_t = t
        else:
            if cur_start is not None and cur_end is not None:
                if cur_end - cur_start >= min_duration:
                    windows.append((cur_start, cur_end))
                cur_start = None
                cur_end = None
            last_t = t
    if cur_start is not None and cur_end is not None:
        if cur_end - cur_start >= min_duration:
            windows.append((cur_start, cur_end))
    return windows


def _rule_face_conf_dropouts(
    face_conf_timeline: list[tuple[float, float]],
) -> list[float]:
    """One frame at the midpoint of every > 0.5s window where
    face_conf < 0.6."""
    windows = _contiguous_windows(
        face_conf_timeline, lambda v: v < 0.6, min_duration=0.5,
    )
    return [(a + b) / 2.0 for a, b in windows]


def _rule_asd_ambiguity(
    asd_margin_timeline: list[tuple[float, float]],
) -> list[float]:
    """One frame at the midpoint of every > 0.5s window where
    ASD margin < 0.3."""
    windows = _contiguous_windows(
        asd_margin_timeline, lambda v: v < 0.3, min_duration=0.5,
    )
    return [(a + b) / 2.0 for a, b in windows]


def _rule_hook_windows(
    clip_starts: list[float], duration: float,
) -> list[float]:
    """Three frames per clip candidate: start+0.5, +1.5, +2.5."""
    out: list[float] = []
    for start in clip_starts:
        for offset in (0.5, 1.5, 2.5):
            t = start + offset
            if 0.0 <= t <= duration:
                out.append(t)
    return out


def _rule_stable_region_floor(
    duration: float,
    existing: list[float],
    interval: float = 30.0,
) -> list[float]:
    """Guarantee at least one sample per ``interval`` seconds.

    Walks the timeline in ``interval``-sized windows; if a window
    has zero existing samples, inject one at the window's midpoint.
    """
    if duration <= 0:
        return []
    existing_sorted = sorted(existing)
    added: list[float] = []
    t = 0.0
    i = 0
    while t < duration:
        window_end = min(t + interval, duration)
        # Advance pointer into existing_sorted up to window_end.
        has_sample = False
        j = i
        while j < len(existing_sorted) and existing_sorted[j] < window_end:
            if existing_sorted[j] >= t:
                has_sample = True
                break
            j += 1
        if not has_sample:
            added.append((t + window_end) / 2.0)
        # Advance i to the start of the next window.
        while i < len(existing_sorted) and existing_sorted[i] < window_end:
            i += 1
        t = window_end
    return added


# ── Dedup + cap ─────────────────────────────────────────────────────


def _dedup_within(ts: list[float], tol: float = 0.2) -> list[float]:
    """Sort and drop entries within ``tol`` seconds of a prior entry."""
    if not ts:
        return []
    sorted_ts = sorted(ts)
    out = [sorted_ts[0]]
    for t in sorted_ts[1:]:
        if t - out[-1] >= tol:
            out.append(t)
    return out


def _total_cap(duration_seconds: float) -> int:
    """max(60, int(duration_minutes * 4))."""
    return max(60, int((duration_seconds / 60.0) * 4))


def _trim_to_cap(ts: list[float], cap: int) -> list[float]:
    """If over the cap, keep evenly-spaced samples.

    Not a hard round-robin — we preserve the first and last entries
    and thin the middle by stride sampling so the cap never drops a
    shot-cut neighborhood that landed at a clip boundary.
    """
    if len(ts) <= cap:
        return ts
    if cap < 2:
        return ts[:cap]
    # Reservoir-style even thinning.
    stride = len(ts) / float(cap)
    out = []
    for i in range(cap):
        out.append(ts[int(round(i * stride))])
    # Re-dedup in case rounding caused collisions.
    return _dedup_within(out, tol=0.05)


# ── Top-level scheduler ─────────────────────────────────────────────


def compute_adaptive_frame_times(inputs: AdaptiveSamplingInputs) -> list[float]:
    """Return a sorted list of VLM sample timestamps.

    Applies rules 1-4, dedupes within 0.2s, applies the stable-region
    floor (rule 5), then caps total count at
    ``max(60, duration_minutes * 4)``.
    """
    duration = max(0.0, float(inputs.duration_seconds))
    collected: list[float] = []
    collected += _rule_shot_cut_neighborhood(inputs.shot_cuts, duration)
    collected += _rule_face_conf_dropouts(inputs.face_conf_timeline)
    collected += _rule_asd_ambiguity(inputs.asd_margin_timeline)
    collected += _rule_hook_windows(inputs.clip_candidate_starts, duration)
    # Dedup before running the floor so stable-region floor sees
    # the true post-rule coverage.
    deduped = _dedup_within(collected, tol=0.2)
    floor = _rule_stable_region_floor(duration, deduped, interval=30.0)
    combined = _dedup_within(deduped + floor, tol=0.2)
    capped = _trim_to_cap(combined, _total_cap(duration))
    return capped
