"""ASD (active-speaker-detection) VLM tiebreaker.

Phase 6C of the VLM subject-tracking upgrade.

The Phase 11 doc calls out that Light-ASD is a risky default-flip on
gaming / anime / heavy-music-over-dialogue content. The resolution:
keep the heuristic ASD path on globally, but when the top-two
candidate margin is < 0.3 for > 0.5s (the same signal Phase 4 uses
to oversample VLM frames), consult Light-ASD on those frames only.
If Light-ASD agrees with the heuristic, boost confidence; if it
disagrees, the next VLM batch gets a "which of these people is
talking?" prompt and the answer wins.

This narrow path is strictly safer than an unmodified heuristic
because it only fires on cases where the heuristic was already
unsure — there's no quality floor to drop below.

Gated behind ``CLIPAI_ASD_TIEBREAKER``. Default ON (new path is
strictly ≥ heuristic-only).

This module is orchestration glue: it takes a stream of ASD
candidate margins, identifies the ambiguous windows, and returns a
list of timestamps for Light-ASD and/or VLM follow-up. Actual
Light-ASD and VLM invocations happen in the caller so this module
can be tested without opencv / ffmpeg.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

TIEBREAKER_ENV = "CLIPAI_ASD_TIEBREAKER"
# Ambiguity thresholds — same as the Phase 4 adaptive sampler's ASD
# rule so every rule fires on the same windows.
DEFAULT_MARGIN_THRESHOLD = 0.3
DEFAULT_WINDOW_MIN_DURATION = 0.5
# When the heuristic and Light-ASD agree on an ambiguous window, we
# boost the heuristic confidence by this amount (capped at 1.0).
HEURISTIC_BOOST_ON_AGREEMENT = 0.6


def tiebreaker_enabled() -> bool:
    """Default ON — the tiebreaker path is strictly safer than the
    unmodified heuristic. Returns True unless explicitly disabled."""
    val = os.environ.get(TIEBREAKER_ENV, "1").strip().lower()
    return val not in ("0", "false", "no", "off")


@dataclass
class AmbiguityWindow:
    """A contiguous window where the ASD margin stayed below the
    ambiguity threshold for long enough to warrant a tiebreaker."""

    start: float
    end: float
    margin_min: float
    # Candidate timestamps to send to Light-ASD / VLM — one per
    # window's midpoint so we don't spam the tiebreaker pipeline.
    sample_timestamps: list[float] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def find_ambiguous_windows(
    asd_margin_timeline: list[tuple[float, float]],
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    min_duration: float = DEFAULT_WINDOW_MIN_DURATION,
) -> list[AmbiguityWindow]:
    """Collapse an ASD margin timeline into ambiguity windows.

    A window is a maximal contiguous run of samples where
    ``margin < margin_threshold``, lasting at least ``min_duration``
    seconds. Gaps of > 1.0s between samples break a run.
    """
    if not asd_margin_timeline:
        return []
    sorted_tl = sorted(asd_margin_timeline, key=lambda x: x[0])
    windows: list[AmbiguityWindow] = []
    cur_start: float | None = None
    cur_end: float | None = None
    cur_min: float = 1.0
    last_t: float | None = None
    for t, margin in sorted_tl:
        if margin < margin_threshold:
            if cur_start is None:
                cur_start = t
                cur_end = t
                cur_min = margin
            elif last_t is not None and t - last_t > 1.0:
                if cur_end - cur_start >= min_duration:
                    windows.append(AmbiguityWindow(
                        start=cur_start, end=cur_end,
                        margin_min=cur_min,
                        sample_timestamps=[(cur_start + cur_end) / 2.0],
                    ))
                cur_start = t
                cur_end = t
                cur_min = margin
            else:
                cur_end = t
                cur_min = min(cur_min, margin)
            last_t = t
        else:
            if cur_start is not None and cur_end is not None:
                if cur_end - cur_start >= min_duration:
                    windows.append(AmbiguityWindow(
                        start=cur_start, end=cur_end,
                        margin_min=cur_min,
                        sample_timestamps=[(cur_start + cur_end) / 2.0],
                    ))
                cur_start = None
                cur_end = None
                cur_min = 1.0
            last_t = t
    if cur_start is not None and cur_end is not None:
        if cur_end - cur_start >= min_duration:
            windows.append(AmbiguityWindow(
                start=cur_start, end=cur_end,
                margin_min=cur_min,
                sample_timestamps=[(cur_start + cur_end) / 2.0],
            ))
    return windows


@dataclass
class TiebreakerDecision:
    """Per-window decision after consulting Light-ASD / VLM."""

    window: AmbiguityWindow
    chosen_speaker: str | int
    source: str  # "heuristic" | "lightasd_agrees" | "vlm_tiebreak"
    confidence: float


def resolve_tiebreaker(
    window: AmbiguityWindow,
    heuristic_choice,
    lightasd_choice=None,
    vlm_choice=None,
    heuristic_confidence: float = 0.4,
) -> TiebreakerDecision:
    """Pick a final speaker for an ambiguous window.

    Priority:
      1. Light-ASD agrees with heuristic → keep heuristic pick,
         boost confidence to ``min(1.0, heuristic_confidence +
         HEURISTIC_BOOST_ON_AGREEMENT)``.
      2. Light-ASD disagrees with heuristic → use ``vlm_choice`` if
         provided (and non-None); else stick with the heuristic.
      3. Light-ASD unavailable (None) → use VLM if provided, else
         stick with the heuristic.
    """
    if lightasd_choice is not None and lightasd_choice == heuristic_choice:
        boosted = min(1.0, heuristic_confidence + HEURISTIC_BOOST_ON_AGREEMENT)
        return TiebreakerDecision(
            window=window,
            chosen_speaker=heuristic_choice,
            source="lightasd_agrees",
            confidence=boosted,
        )
    if lightasd_choice is not None and lightasd_choice != heuristic_choice:
        if vlm_choice is not None:
            return TiebreakerDecision(
                window=window,
                chosen_speaker=vlm_choice,
                source="vlm_tiebreak",
                confidence=0.8,
            )
        return TiebreakerDecision(
            window=window,
            chosen_speaker=heuristic_choice,
            source="heuristic",
            confidence=heuristic_confidence,
        )
    # lightasd_choice is None
    if vlm_choice is not None:
        return TiebreakerDecision(
            window=window,
            chosen_speaker=vlm_choice,
            source="vlm_tiebreak",
            confidence=0.7,
        )
    return TiebreakerDecision(
        window=window,
        chosen_speaker=heuristic_choice,
        source="heuristic",
        confidence=heuristic_confidence,
    )
