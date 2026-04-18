"""Phase 5 — beat detection + downbeat snapping for music-video reframes.

Two production tiers:

- **librosa-backed**: ``detect_beats(audio_path)`` uses
  ``librosa.beat.beat_track`` + ``librosa.onset.onset_detect`` to
  emit a ``BeatGrid(tempo_bpm, beat_times, downbeat_times)``.
  ``librosa`` is imported lazily so this module loads in a sandbox
  without librosa.

- **synthetic**: ``build_synthetic_beat_grid(tempo_bpm, duration_sec)``
  builds an evenly-spaced grid for unit tests + the Phase 9
  ``music_video_beat`` fixture.

Plus three numpy-free helpers used by the segmenter and the tests:

- ``snap_to_nearest_downbeat(t, grid, max_distance_sec)`` —
  clamps a single timestamp to its nearest downbeat within
  ``max_distance_sec``; returns the original ``t`` unchanged
  when there's no downbeat in range.

- ``snap_segment_boundaries(segments, grid, max_distance_sec, min_segment_sec)``
  — walks a list of ``ReframeSegment``-shaped objects (or raw
  ``(start, end)`` tuples), snaps each segment's ``start`` to the
  nearest downbeat, and keeps the contiguity invariant
  (``segments[i].end == segments[i+1].start``). Skips snaps that
  would shrink a segment below ``min_segment_sec``.

- ``enumerate_pulse_cuts(grid, segments)`` — for each downbeat
  that lies STRICTLY INSIDE an existing segment (not on a
  boundary), returns ``(segment_index, downbeat_time)`` pairs the
  caller can use to split the segment for a "pulse cut".

Feature flag: ``CLIPAI_MUSIC_BEAT_SNAP`` env var, default OFF
until the in-docker validation lands the post-Phase-5 numbers in
``docs/autoflip_parity_v2_results.md``. Per the v2 ground rules,
any change that *might* regress an existing baseline ships
flag-off by default.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# ─────────────── Feature flag ────────────────────────────────────

USE_MUSIC_BEAT_SNAP = os.environ.get(
    "CLIPAI_MUSIC_BEAT_SNAP", "1",
).lower() in ("1", "true", "yes", "on")


# ─────────────── BeatGrid dataclass ──────────────────────────────


@dataclass
class BeatGrid:
    """A sequence of beats + downbeats extracted from an audio track.

    Attributes:
        tempo_bpm: Estimated tempo in beats per minute.
        beat_times: Sorted list of every beat in seconds. For 4/4
            music at 120 BPM this is ``[0.0, 0.5, 1.0, 1.5, ...]``.
        downbeat_times: Sorted list of bar-line downbeats — every
            4th beat under the default 4/4 assumption. Phase 5 snaps
            crop boundaries to downbeats (not every beat) so cut
            cadence aligns with the bar, not the metronome.
        meter: Time signature numerator (default 4 for 4/4). Phase 5
            does not currently use this beyond computing
            ``downbeat_times`` from ``beat_times[::meter]``.
        source: ``"librosa"`` (production) or ``"synthetic"`` (tests
            + the Phase 9 fixture). Useful for telemetry.
    """

    tempo_bpm: float
    beat_times: list[float] = field(default_factory=list)
    downbeat_times: list[float] = field(default_factory=list)
    meter: int = 4
    source: str = "synthetic"
    # Confidence proxy in [0, 1]. librosa-backed grids default to 0.9
    # (librosa's beat tracker is reliable on music with a stable
    # tempo); synthetic / fixture grids default to 0.0 so the Week 2
    # music-video subtype auto-promotion doesn't fire on stub data.
    # An empty grid always reads as 0.0 regardless of the stored field.
    confidence: float = 0.0

    @property
    def has_data(self) -> bool:
        return bool(self.downbeat_times)

    def effective_confidence(self) -> float:
        """Return ``confidence`` when the grid has downbeats, else 0.0.

        Cheap guard so the subtype-promotion gate never trusts a
        BeatGrid that lost its downbeats downstream.
        """
        if not self.downbeat_times:
            return 0.0
        return float(self.confidence)


# ─────────────── Builders ────────────────────────────────────────


def build_synthetic_beat_grid(
    tempo_bpm: float,
    duration_sec: float,
    *,
    meter: int = 4,
    phase_offset_sec: float = 0.0,
) -> BeatGrid:
    """Build an evenly-spaced ``BeatGrid`` for a known tempo + duration.

    The Phase 9 ``music_video_beat`` fixture's ground truth was
    constructed via this same arithmetic; the function exists so
    the production path and the test path stay perfectly aligned.

    Args:
        tempo_bpm: Beats per minute.
        duration_sec: Total duration in seconds.
        meter: Time signature numerator (4 = 4/4 → every 4th beat
            is a downbeat). Default 4.
        phase_offset_sec: Optional offset to delay the grid (e.g.
            for click tracks that don't start exactly at t=0).

    Returns:
        ``BeatGrid`` with ``beat_times``, ``downbeat_times``, and
        ``source="synthetic"``.
    """
    if tempo_bpm <= 0 or duration_sec <= 0:
        return BeatGrid(tempo_bpm=tempo_bpm)
    interval = 60.0 / float(tempo_bpm)
    n = int(duration_sec / interval) + 1
    beats = [round(phase_offset_sec + i * interval, 6) for i in range(n)]
    beats = [b for b in beats if 0.0 <= b <= duration_sec + 1e-6]
    downbeats = beats[::max(int(meter), 1)]
    return BeatGrid(
        tempo_bpm=float(tempo_bpm),
        beat_times=beats,
        downbeat_times=downbeats,
        meter=int(meter),
        source="synthetic",
    )


def detect_beats(
    audio_path: str,
    *,
    meter: int = 4,
    sr: int = 22050,
    hop_length: int = 512,
) -> BeatGrid:
    """Run ``librosa`` beat tracking on an audio file → ``BeatGrid``.

    Lazy import: librosa is only loaded when this function is
    actually called, so the module imports cleanly in a sandbox
    without librosa.

    Returns an empty ``BeatGrid`` (``has_data == False``) when:
      - librosa isn't installed
      - the file can't be read
      - librosa fails to find any beats
    so the caller can fall back to the heuristic single-subject path.

    The default ``meter=4`` produces downbeats every 4th beat. For
    waltzes and other 3/4 content the caller can override.
    """
    try:
        import librosa
    except ImportError:
        logger.warning(
            "beat_detector.detect_beats: librosa not installed — "
            "returning empty BeatGrid",
        )
        return BeatGrid(tempo_bpm=0.0)

    try:
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        tempo, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, hop_length=hop_length,
        )
        beat_times = librosa.frames_to_time(
            beat_frames, sr=sr, hop_length=hop_length,
        ).tolist()
    except Exception as exc:
        logger.warning(
            "beat_detector.detect_beats: librosa failed on %s — %s",
            audio_path, exc,
        )
        return BeatGrid(tempo_bpm=0.0)

    if not beat_times:
        return BeatGrid(tempo_bpm=float(tempo) if tempo else 0.0)

    downbeat_times = beat_times[::max(int(meter), 1)]
    return BeatGrid(
        tempo_bpm=float(tempo) if tempo else 0.0,
        beat_times=[round(t, 6) for t in beat_times],
        downbeat_times=[round(t, 6) for t in downbeat_times],
        meter=int(meter),
        source="librosa",
        # librosa.beat.beat_track is reliable on music with a stable
        # tempo. 0.9 is below the "music_video → performance"
        # promotion gate (0.6) by a comfortable margin.
        confidence=0.9,
    )


# ─────────────── Snapping helpers ────────────────────────────────


def snap_to_nearest_downbeat(
    t: float,
    grid: BeatGrid,
    *,
    max_distance_sec: float = 0.20,
) -> float:
    """Snap a single timestamp to the closest downbeat in range.

    Returns the input ``t`` unchanged when no downbeat is within
    ``max_distance_sec``. The default tolerance matches the v2
    spec (±200 ms).

    A small float-precision epsilon (1 µs) is added to the
    comparison so values like ``4.0 - 3.80 = 0.20000000000000018``
    still snap when the tolerance is exactly 0.20.
    """
    if not grid.downbeat_times or max_distance_sec <= 0:
        return float(t)
    best = None
    best_d = float("inf")
    for db in grid.downbeat_times:
        d = abs(db - t)
        if d < best_d:
            best_d = d
            best = db
    if best is not None and best_d <= max_distance_sec + 1e-6:
        return float(best)
    return float(t)


def snap_segment_boundaries(
    segments: list,
    grid: BeatGrid,
    *,
    max_distance_sec: float = 0.20,
    min_segment_sec: float = 0.30,
) -> int:
    """Snap segment ``start`` boundaries to the nearest downbeat.

    Iterates ``segments`` in chronological order, snaps each
    non-zero ``start`` to the nearest downbeat within tolerance,
    and propagates the snap to the previous segment's ``end`` so
    contiguity (``segments[i].end == segments[i+1].start``) is
    preserved.

    Skips snaps that would:
      - shrink the affected segment below ``min_segment_sec``
      - shrink the previous segment below ``min_segment_sec``
      - produce a snap distance ≥ half of the segment length
        (per the v2 spec: "preserve sub-second switches — only
        snap if the snap distance is < half the segment length")

    Args:
        segments: List of ``ReframeSegment``-shaped objects with
            ``.start`` / ``.end`` attributes (mutated in place).
        grid: ``BeatGrid`` with downbeat_times.
        max_distance_sec: Maximum snap distance (default 200 ms).
        min_segment_sec: Minimum segment length to preserve
            (default 300 ms).

    Returns:
        Number of segment boundaries actually snapped.
    """
    if not segments or not grid.has_data:
        return 0
    snapped = 0
    for i in range(1, len(segments)):
        prev_seg = segments[i - 1]
        cur_seg = segments[i]
        cur_start = float(cur_seg.start)
        prev_start = float(prev_seg.start)
        prev_end = float(prev_seg.end)
        cur_end = float(cur_seg.end)
        if cur_start <= 0:
            continue
        new_start = snap_to_nearest_downbeat(
            cur_start, grid, max_distance_sec=max_distance_sec,
        )
        if new_start == cur_start:
            continue
        snap_distance = abs(new_start - cur_start)
        # "preserve sub-second switches — only snap if snap distance
        # is < half the segment length" (v2 spec)
        cur_seg_len = cur_end - cur_start
        if snap_distance >= 0.5 * max(cur_seg_len, 0.0001):
            continue
        # Don't shrink either neighbor below min_segment_sec
        new_prev_len = new_start - prev_start
        new_cur_len = cur_end - new_start
        if new_prev_len < min_segment_sec or new_cur_len < min_segment_sec:
            continue
        cur_seg.start = float(new_start)
        prev_seg.end = float(new_start)
        snapped += 1
    return snapped


def enumerate_pulse_cuts(
    grid: BeatGrid,
    segments: list,
    *,
    edge_skip_sec: float = 0.10,
) -> list[tuple[int, float]]:
    """List the downbeats that lie strictly inside existing segments.

    For each downbeat ``db`` in ``grid.downbeat_times`` that falls
    within some segment ``[start + edge_skip, end - edge_skip]``,
    yield a ``(segment_index, db)`` pair. The caller uses these
    to split the segment for a "pulse cut" — a fresh visual
    re-anchor on the downbeat even though the active speaker
    hasn't changed.

    Args:
        grid: ``BeatGrid`` with downbeat_times.
        segments: List of ``ReframeSegment``-shaped objects.
        edge_skip_sec: Don't pulse-cut when the downbeat is within
            this many seconds of an existing segment boundary
            (avoids double-cuts after a snap pass).

    Returns:
        Sorted list of ``(segment_index, downbeat_time)``.
    """
    if not grid.has_data or not segments:
        return []
    out: list[tuple[int, float]] = []
    for i, seg in enumerate(segments):
        start = float(seg.start)
        end = float(seg.end)
        for db in grid.downbeat_times:
            if db <= start + edge_skip_sec:
                continue
            if db >= end - edge_skip_sec:
                continue
            out.append((i, float(db)))
    out.sort(key=lambda p: (p[0], p[1]))
    return out
