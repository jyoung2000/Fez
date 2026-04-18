"""Snap clip boundaries to word-level timestamps for clean cuts.

Two snap modes:

- **Word-level** (Phase 0+): snaps to the start/end of the nearest
  transcribed word, with pre/post roll for breath. The default
  for every content type.

- **Downbeat-level** (Phase 5): snaps to the nearest downbeat in a
  ``BeatGrid`` for music-video content. Activated by passing a
  ``beat_grid`` to ``snap_clip_boundaries`` / ``snap_all_clips``.
  Falls back to word-level when the grid is empty or unavailable.
"""

import logging
from typing import Optional

from backend.models import TranscriptSegment, ClipCandidate

logger = logging.getLogger(__name__)

PRE_ROLL_S = 0.3   # Silence before first word
POST_ROLL_S = 0.5  # Silence after last word

# Default snap window for downbeat snapping (per the v2 spec).
DOWNBEAT_SNAP_TOLERANCE_SEC = 0.20


def snap_clip_boundaries(
    clip: ClipCandidate,
    transcript: list[TranscriptSegment],
) -> ClipCandidate:
    """Snap a clip's start/end to the nearest word boundary.

    - start_time snaps to the START of the nearest word (minus pre-roll)
    - end_time snaps to the END of the nearest word (plus post-roll)
    """
    # Collect all words in the clip's time range (with some margin)
    margin = 2.0
    words_in_range = []
    for seg in transcript:
        if seg.words and seg.end >= clip.start_time - margin and seg.start <= clip.end_time + margin:
            for w in seg.words:
                words_in_range.append(w)

    if not words_in_range:
        return clip  # No word timestamps available

    words_in_range.sort(key=lambda w: w.start)

    # Snap start: find the word whose start is closest to clip.start_time
    best_start_word = min(words_in_range, key=lambda w: abs(w.start - clip.start_time))
    snapped_start = max(0, best_start_word.start - PRE_ROLL_S)

    # Snap end: find the word whose end is closest to clip.end_time
    best_end_word = min(words_in_range, key=lambda w: abs(w.end - clip.end_time))
    snapped_end = best_end_word.end + POST_ROLL_S

    # Guard: snapping must not invert start/end
    if snapped_start >= snapped_end:
        logger.warning(
            "Boundary snap would invert start/end for '%s' (%.1f >= %.1f) — skipping",
            clip.title, snapped_start, snapped_end,
        )
        return clip

    # Ensure minimum duration is maintained
    if snapped_end - snapped_start < 15:
        return clip  # Don't snap if it would make the clip too short

    if abs(snapped_start - clip.start_time) > 3.0 or abs(snapped_end - clip.end_time) > 3.0:
        logger.warning(
            "Large snap adjustment for '%s': start %.1f->%.1f, end %.1f->%.1f",
            clip.title, clip.start_time, snapped_start, clip.end_time, snapped_end,
        )

    clip.start_time = round(snapped_start, 3)
    clip.end_time = round(snapped_end, 3)
    clip.duration = round(clip.end_time - clip.start_time, 1)
    return clip


def snap_clip_to_downbeats(
    clip: ClipCandidate,
    beat_grid,
    *,
    tolerance_sec: float = DOWNBEAT_SNAP_TOLERANCE_SEC,
) -> ClipCandidate:
    """Snap a clip's start/end to the nearest downbeats in a BeatGrid.

    Phase 5 entry point for music-video content. ``beat_grid`` is a
    :class:`backend.services.beat_detector.BeatGrid` (lazily imported
    so this module loads without librosa). Snaps each boundary
    independently within ``tolerance_sec`` of the closest downbeat
    and skips snaps that would invert start/end or shrink the clip
    below the existing 15 s minimum from
    ``snap_clip_boundaries``.

    Returns the clip mutated in place.
    """
    from backend.services.beat_detector import (
        BeatGrid,
        snap_to_nearest_downbeat,
    )

    if not isinstance(beat_grid, BeatGrid) or not beat_grid.has_data:
        return clip

    new_start = snap_to_nearest_downbeat(
        clip.start_time, beat_grid, max_distance_sec=tolerance_sec,
    )
    new_end = snap_to_nearest_downbeat(
        clip.end_time, beat_grid, max_distance_sec=tolerance_sec,
    )

    if new_start >= new_end:
        return clip
    if new_end - new_start < 15:
        return clip

    if (
        abs(new_start - clip.start_time) > 0
        or abs(new_end - clip.end_time) > 0
    ):
        logger.debug(
            "Downbeat-snapped clip '%s': start %.3f→%.3f, end %.3f→%.3f",
            clip.title, clip.start_time, new_start, clip.end_time, new_end,
        )

    clip.start_time = round(new_start, 3)
    clip.end_time = round(new_end, 3)
    clip.duration = round(clip.end_time - clip.start_time, 1)
    return clip


def snap_all_clips(
    clips: list[ClipCandidate],
    transcript: list[TranscriptSegment],
    *,
    beat_grid=None,
    content_type: Optional[str] = None,
) -> list[ClipCandidate]:
    """Snap all clip boundaries to word- (or downbeat-) timestamps.

    For music-video content with a populated ``beat_grid``, snaps
    to nearest downbeats (Phase 5); otherwise falls back to the
    word-level snap (Phase 0). Pure word-level callers can omit
    both new kwargs and get exactly the legacy behavior.
    """
    # Music-video downbeat path
    use_beat_snap = (
        beat_grid is not None
        and getattr(beat_grid, "has_data", False)
        and content_type == "music_video"
    )

    if use_beat_snap:
        snapped = [snap_clip_to_downbeats(clip, beat_grid) for clip in clips]
        snapped_count = sum(
            1 for orig, new in zip(clips, snapped)
            if orig.start_time != new.start_time
            or orig.end_time != new.end_time
        )
        if snapped_count:
            logger.info(
                "Snapped %d/%d clip boundaries to downbeats (music_video)",
                snapped_count, len(clips),
            )
        return snapped

    # Word-level fallback (legacy path)
    has_words = any(seg.words for seg in transcript)
    if not has_words:
        logger.info("No word-level timestamps available — skipping boundary snapping")
        return clips

    snapped = []
    for clip in clips:
        snapped.append(snap_clip_boundaries(clip, transcript))

    snapped_count = sum(
        1 for orig, new in zip(clips, snapped)
        if orig.start_time != new.start_time or orig.end_time != new.end_time
    )
    if snapped_count:
        logger.info("Snapped %d/%d clip boundaries to word timestamps", snapped_count, len(clips))
    return snapped
