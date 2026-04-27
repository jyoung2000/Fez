"""Transcript coverage verification + gap fill (recall pass).

The normal Whisper pipeline is tuned for a precision/recall tradeoff that
drops short, soft, or unclear speech by design — Whisper's own internal
``no_speech_threshold`` and the Silero VAD gate both discard audio that
they are not confident is speech, and downstream hallucination filters
(whisper_worker._filter_segments, transcription._filter_hallucinations)
discard low-ratio / long-duration / repeat segments that sometimes
correspond to real but faint speech.

This module adds a SECOND PASS that:

  1. Runs an independent VAD (webrtcvad via ``active_speaker.build_vad_presence``)
     over the same audio to get a speech-presence mask that is NOT influenced
     by Whisper's internal gates.
  2. Compares that mask to what the main transcript actually covered.
  3. For every gap where VAD says "there is speech here" but the transcript
     is empty for more than ``min_gap_sec`` seconds, slices the audio and
     re-runs Whisper on just that slice with recall-first parameters:
       - vad_filter=False            (we already know speech is there)
       - no_speech_threshold=0.2     (accept low-confidence speech)
       - temperature=[0.0]           (no temp fallback → no hallucination chain)
       - condition_on_previous=False (prevent prompt-echo hallucinations)
       - word_timestamps=True        (for downstream alignment)
  4. Merges the recovered segments back into the transcript, sorted by time.

Why this is additive (safe):
  - It only produces NEW segments in time windows that the main pass
    returned empty. It cannot delete existing segments, corrupt their
    text, or shift their timestamps.
  - It is guarded by ``settings.WHISPER_GAP_FILL_ENABLED`` so it can be
    toggled off per-user.
  - If VAD or the Whisper slice call raises, the pass short-circuits
    and returns the original transcript unchanged.

Why the parameters are different from the main pass:
  - ``vad_filter=False``: the main pass' VAD is what dropped the audio;
    running it again here would drop it again.
  - ``no_speech_threshold=0.2``: we have external evidence (webrtcvad)
    that speech is present; Whisper's internal gate can be relaxed.
  - ``temperature=[0.0]``: six-step temp fallback on very short clips
    produces chains of hallucinated text ("Thank you. Thank you...") —
    the Whisper paper (Radford et al. 2022, §4.4) documents this. We
    keep only the low-temp pass and let the outer pipeline retry if
    this single pass produces nothing useful.
  - ``condition_on_previous_text=False``: short slices have no context
    anyway, and conditioning on the previous segment's text causes the
    model to echo that text back as the transcript for the slice.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

from backend.models import TranscriptSegment, WordTimestamp

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Tunables
# ──────────────────────────────────────────────────────────────────────────

# Minimum gap length (seconds) to be considered a coverage hole worth
# re-transcribing. Below this, gaps are almost always legitimate pauses
# (breath, beat between sentences) and re-running Whisper on them
# produces hallucinations more often than real words.
_MIN_GAP_SEC = 2.0

# Minimum VAD-labelled voiced duration inside a gap for it to be a
# candidate. Gaps with VAD < 0.5s of voiced audio are silence/music
# artifacts, not speech.
_MIN_VAD_OVERLAP_SEC = 0.5

# Maximum number of gaps we will re-transcribe per video. Hard cap to
# prevent a pathologically under-transcribed video from running Whisper
# hundreds of times. Sorted by gap length descending so we fill the
# biggest holes first.
_MAX_GAPS_TO_FILL = 40

# Hard cap on the combined audio duration we will re-transcribe. On a
# GTX 1650 with medium model, Whisper runs around 0.5-1x real-time, so
# 600s of re-transcription adds up to ~10-20 min of extra pipeline time.
_MAX_FILL_AUDIO_SEC = 600.0

# Minimum confidence (1 + avg_logprob, clipped to [0,1]) for a recovered
# segment to be kept. Below this, it is more likely a hallucination than
# real recovered speech.
_MIN_FILL_CONFIDENCE = 0.2

# Skip boilerplate phrases Whisper is known to hallucinate on silence /
# music. If a gap produces one of these and nothing else, we discard.
_BOILERPLATE_LOWERED = frozenset([
    "thanks for watching",
    "thank you for watching",
    "thank you",
    "thanks for watching!",
    "subscribe",
    "please subscribe",
    "like and subscribe",
    "see you next time",
    "see you in the next video",
    "bye",
    "bye bye",
    "goodbye",
    ".",
    "you",
    "you.",
])


@dataclass
class GapFillStats:
    """Observability data emitted so the pipeline can log a summary."""

    gaps_found: int = 0
    gaps_attempted: int = 0
    gaps_filled: int = 0
    segments_added: int = 0
    fill_audio_sec: float = 0.0
    coverage_before: float = 0.0
    coverage_after: float = 0.0
    vad_speech_sec: float = 0.0
    elapsed_sec: float = 0.0
    skipped_reason: Optional[str] = None
    per_gap: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "gaps_found": self.gaps_found,
            "gaps_attempted": self.gaps_attempted,
            "gaps_filled": self.gaps_filled,
            "segments_added": self.segments_added,
            "fill_audio_sec": round(self.fill_audio_sec, 2),
            "coverage_before": round(self.coverage_before, 4),
            "coverage_after": round(self.coverage_after, 4),
            "vad_speech_sec": round(self.vad_speech_sec, 2),
            "elapsed_sec": round(self.elapsed_sec, 2),
            "skipped_reason": self.skipped_reason,
            "per_gap": self.per_gap,
        }


# ──────────────────────────────────────────────────────────────────────────
# Gap discovery (pure function, easy to unit-test)
# ──────────────────────────────────────────────────────────────────────────

def _merge_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Merge overlapping / touching intervals, return sorted disjoint list."""
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    merged: list[list[float]] = [[sorted_iv[0][0], sorted_iv[0][1]]]
    for start, end in sorted_iv[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _vad_overlap(
    vad_intervals: list[tuple[float, float]],
    window_start: float,
    window_end: float,
) -> float:
    """Total VAD-voiced duration inside ``[window_start, window_end]``."""
    if window_end <= window_start or not vad_intervals:
        return 0.0
    total = 0.0
    for vs, ve in vad_intervals:
        if ve <= window_start:
            continue
        if vs >= window_end:
            break
        total += min(ve, window_end) - max(vs, window_start)
    return max(0.0, total)


def find_transcript_gaps(
    segments: list,
    vad_intervals: list[tuple[float, float]],
    audio_duration: float,
    min_gap_sec: float = _MIN_GAP_SEC,
    min_vad_overlap_sec: float = _MIN_VAD_OVERLAP_SEC,
) -> list[tuple[float, float, float]]:
    """Find (gap_start, gap_end, vad_voiced_sec) windows worth re-transcribing.

    A "gap" is a time window inside [0, audio_duration) that:
      - contains no transcript segment, and
      - contains at least ``min_vad_overlap_sec`` seconds of VAD-voiced
        audio (i.e. the independent VAD says there IS speech here), and
      - is at least ``min_gap_sec`` long.

    Segments may be ``TranscriptSegment`` dataclass instances OR plain
    dicts with ``start``/``end`` keys — we support both so this function
    can run against either the pre-model dict form or the post-model
    dataclass form used in the pipeline.
    """
    if audio_duration <= 0:
        return []

    # Normalize segments → (start, end) tuples, tolerate both shapes.
    seg_iv: list[tuple[float, float]] = []
    for s in segments:
        if s is None:
            continue
        if isinstance(s, dict):
            start = s.get("start")
            end = s.get("end")
        else:
            start = getattr(s, "start", None)
            end = getattr(s, "end", None)
        if start is None or end is None:
            continue
        try:
            fs, fe = float(start), float(end)
        except (TypeError, ValueError):
            continue
        if fe > fs:
            seg_iv.append((fs, fe))

    seg_merged = _merge_intervals(seg_iv)
    vad_sorted = _merge_intervals(vad_intervals or [])

    # Walk complement of seg_merged within [0, audio_duration).
    gap_candidates: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in seg_merged:
        if start > cursor:
            gap_candidates.append((cursor, min(start, audio_duration)))
        cursor = max(cursor, end)
        if cursor >= audio_duration:
            break
    if cursor < audio_duration:
        gap_candidates.append((cursor, audio_duration))

    qualified: list[tuple[float, float, float]] = []
    for gs, ge in gap_candidates:
        if ge - gs < min_gap_sec:
            continue
        overlap = _vad_overlap(vad_sorted, gs, ge)
        if overlap < min_vad_overlap_sec:
            continue
        qualified.append((gs, ge, overlap))

    return qualified


def compute_coverage(
    segments: list,
    vad_intervals: list[tuple[float, float]],
    audio_duration: float,
) -> tuple[float, float]:
    """Return (transcript_coverage, vad_coverage_of_transcript).

    - transcript_coverage = (union of segment durations) / audio_duration
    - vad_coverage_of_transcript = fraction of VAD-voiced seconds that
      are covered by at least one transcript segment. This is the more
      meaningful number because it ignores silence.

    Returns (0.0, 0.0) if audio_duration <= 0.
    """
    if audio_duration <= 0:
        return 0.0, 0.0

    seg_iv: list[tuple[float, float]] = []
    for s in segments:
        if s is None:
            continue
        if isinstance(s, dict):
            start = s.get("start")
            end = s.get("end")
        else:
            start = getattr(s, "start", None)
            end = getattr(s, "end", None)
        if start is None or end is None:
            continue
        try:
            fs, fe = float(start), float(end)
        except (TypeError, ValueError):
            continue
        if fe > fs:
            seg_iv.append((fs, fe))

    seg_merged = _merge_intervals(seg_iv)
    seg_total = sum(e - s for s, e in seg_merged)
    transcript_cov = min(1.0, seg_total / audio_duration)

    vad_sorted = _merge_intervals(vad_intervals or [])
    vad_total = sum(e - s for s, e in vad_sorted)
    if vad_total <= 0:
        return transcript_cov, 0.0

    # Intersection of VAD intervals with segment intervals
    covered_voiced = 0.0
    i = j = 0
    while i < len(seg_merged) and j < len(vad_sorted):
        ss, se = seg_merged[i]
        vs, ve = vad_sorted[j]
        lo, hi = max(ss, vs), min(se, ve)
        if hi > lo:
            covered_voiced += hi - lo
        if se < ve:
            i += 1
        else:
            j += 1
    voiced_cov = covered_voiced / vad_total if vad_total > 0 else 0.0
    return transcript_cov, voiced_cov


# ──────────────────────────────────────────────────────────────────────────
# Audio slicing (ffmpeg) — kept tiny and side-effect-free
# ──────────────────────────────────────────────────────────────────────────

def _extract_audio_slice(
    audio_path: str,
    start_sec: float,
    end_sec: float,
    out_path: str,
) -> bool:
    """Copy ``[start, end)`` from ``audio_path`` to ``out_path`` as 16 kHz mono WAV.

    Returns True on success, False on any ffmpeg failure. Callers should
    treat False as "skip this gap".
    """
    duration = max(0.0, end_sec - start_sec)
    if duration <= 0:
        return False
    try:
        cmd = [
            "ffmpeg", "-y",
            "-nostdin", "-loglevel", "error",
            "-ss", f"{start_sec:.3f}",
            "-i", audio_path,
            "-t", f"{duration:.3f}",
            "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le",
            out_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, timeout=30,
        )
        if result.returncode != 0:
            logger.debug(
                "gap_filler: ffmpeg slice failed [%s..%s]: %s",
                start_sec, end_sec,
                (result.stderr or b"").decode(errors="replace")[:200],
            )
            return False
        # Sanity: file must be non-trivial.
        if not os.path.exists(out_path) or os.path.getsize(out_path) < 200:
            return False
        return True
    except Exception as e:
        logger.debug("gap_filler: slice exception %s", e)
        return False


# ──────────────────────────────────────────────────────────────────────────
# Slice filter — very conservative. Anything stricter belongs upstream.
# ──────────────────────────────────────────────────────────────────────────

def _is_hallucinated_fill(text: str) -> bool:
    """Cheap heuristic to reject common Whisper-on-silence outputs.

    Used only inside gap-fill segments where we have little context.
    Does NOT touch the main-pass hallucination filter.
    """
    if not text:
        return True
    t = text.strip().lower().rstrip(".!? ")
    if not t:
        return True
    if t in _BOILERPLATE_LOWERED:
        return True
    # Punctuation / pure whitespace remnants
    if all(not ch.isalnum() for ch in t):
        return True
    # Looping detection — same token repeated 4+ times
    toks = t.split()
    if len(toks) >= 4 and len(set(toks)) == 1:
        return True
    return False


# ──────────────────────────────────────────────────────────────────────────
# Main entrypoint
# ──────────────────────────────────────────────────────────────────────────

def fill_transcript_gaps_sync(
    audio_path: str,
    segments: list,
    audio_duration: float,
    language: str = "",
    task: str = "transcribe",
    is_animated: bool = False,
    initial_prompt: str = "",
    max_gaps: int = _MAX_GAPS_TO_FILL,
    max_fill_audio_sec: float = _MAX_FILL_AUDIO_SEC,
) -> tuple[list, GapFillStats]:
    """Synchronous entrypoint. Returns (possibly-augmented segments, stats).

    This function is sync because faster-whisper's Python API is sync; the
    outer pipeline wraps it in ``run_in_executor``. Keeping it sync makes
    unit testing much easier (no event loop needed).

    Never raises for expected failure modes — it always returns a (list,
    stats) tuple. On any unexpected exception, it logs and returns the
    original segments unchanged with ``stats.skipped_reason`` set.
    """
    wall_start = time.monotonic()
    stats = GapFillStats()

    if audio_duration <= 0 or not audio_path:
        stats.skipped_reason = "no_audio"
        return segments, stats

    # 1. Build VAD mask with the existing project helper. Importing lazily
    #    so unit tests can monkey-patch build_vad_presence cleanly.
    try:
        from backend.services.active_speaker import build_vad_presence
        vad_intervals = build_vad_presence(audio_path)
    except Exception as e:
        logger.info("gap_filler: VAD unavailable (%s) — skipping", e)
        stats.skipped_reason = "vad_failed"
        stats.elapsed_sec = time.monotonic() - wall_start
        return segments, stats

    stats.vad_speech_sec = sum(e - s for s, e in vad_intervals)
    stats.coverage_before = compute_coverage(
        segments, vad_intervals, audio_duration,
    )[1]  # voiced coverage is the one we care about

    # 2. Find gaps worth filling.
    gaps = find_transcript_gaps(segments, vad_intervals, audio_duration)
    stats.gaps_found = len(gaps)
    if not gaps:
        stats.coverage_after = stats.coverage_before
        stats.elapsed_sec = time.monotonic() - wall_start
        logger.info(
            "gap_filler: no gaps to fill "
            "(vad_speech=%.1fs, voiced_coverage=%.1f%%)",
            stats.vad_speech_sec, stats.coverage_before * 100,
        )
        return segments, stats

    # 3. Prioritize by VAD-voiced length inside the gap (bigger first),
    #    then cap by count and total duration.
    gaps_sorted = sorted(gaps, key=lambda g: -g[2])
    budget_audio = max_fill_audio_sec
    to_fill: list[tuple[float, float, float]] = []
    for gs, ge, voiced in gaps_sorted[:max_gaps]:
        dur = ge - gs
        if dur > budget_audio:
            # Truncate the tail — still worth filling the first chunk.
            ge = gs + budget_audio
            dur = budget_audio
        to_fill.append((gs, ge, voiced))
        budget_audio -= dur
        if budget_audio <= 0:
            break

    stats.gaps_attempted = len(to_fill)

    # 4. Resolve the slice transcriber. The gap-filler used to inherit
    #    the in-process Whisper singleton via ``_get_whisper_model``,
    #    but right after the main transcription subprocess exits that
    #    singleton is either uninitialised or stale, and reloading it
    #    re-attempts CUDA against an Ollama-occupied / passthrough-
    #    missing GPU on every single gap. The subprocess path gives
    #    fresh CUDA-context isolation per slice (or falls back to CPU
    #    cleanly if GPU is unavailable) without polluting this thread.
    try:
        from backend.services.transcription import (
            transcribe_audio_slice_subprocess,
        )
    except Exception as e:
        logger.warning(
            "gap_filler: subprocess slice helper unavailable (%s) — skipping",
            e,
        )
        stats.skipped_reason = "model_unavailable"
        stats.coverage_after = stats.coverage_before
        stats.elapsed_sec = time.monotonic() - wall_start
        return segments, stats

    added_segments: list[TranscriptSegment] = []

    # 5. For each gap: slice → transcribe (subprocess) → shift
    #    timestamps → filter → append. Recall-first knobs are baked
    #    into ``transcribe_audio_slice_subprocess``.
    for gs, ge, voiced in to_fill:
        per_gap: dict = {
            "gap_start": round(gs, 2),
            "gap_end": round(ge, 2),
            "vad_voiced_sec": round(voiced, 2),
            "segments_added": 0,
            "status": "unknown",
        }
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False, dir=tempfile.gettempdir(),
            ) as tmp:
                slice_path = tmp.name
            ok = _extract_audio_slice(audio_path, gs, ge, slice_path)
            if not ok:
                per_gap["status"] = "slice_failed"
                stats.per_gap.append(per_gap)
                try:
                    os.unlink(slice_path)
                except OSError:
                    pass
                continue
            try:
                slice_dur = max(15.0, (ge - gs) * 4.0 + 30.0)
                segments_raw = transcribe_audio_slice_subprocess(
                    slice_path,
                    language=language,
                    task=task,
                    initial_prompt=initial_prompt,
                    timeout=slice_dur,
                    is_animated=is_animated,
                )
                collected = []
                for seg in segments_raw:
                    text = (seg.get("text") or "").strip()
                    if _is_hallucinated_fill(text):
                        continue

                    avg_lp = float(seg.get("avg_logprob") or -1.0)
                    no_speech = float(seg.get("no_speech_prob") or 0.0)
                    confidence = max(0.0, min(1.0, 1.0 + avg_lp))
                    if no_speech > 0.85:
                        continue
                    if confidence < _MIN_FILL_CONFIDENCE:
                        continue

                    words = None
                    raw_words = seg.get("words") or []
                    if raw_words:
                        words = []
                        for w in raw_words:
                            wt = (w.get("word") or "").strip()
                            if not wt:
                                continue
                            words.append(WordTimestamp(
                                start=round(float(w["start"]) + gs, 3),
                                end=round(float(w["end"]) + gs, 3),
                                word=wt,
                            ))

                    collected.append(TranscriptSegment(
                        start=round(float(seg["start"]) + gs, 2),
                        end=round(float(seg["end"]) + gs, 2),
                        text=text,
                        speaker="Speaker ?",  # deferred; diarization assigns later
                        words=words,
                        confidence=round(confidence, 3),
                        avg_logprob=round(avg_lp, 4),
                        no_speech_prob=round(no_speech, 4),
                    ))
                if collected:
                    added_segments.extend(collected)
                    per_gap["segments_added"] = len(collected)
                    per_gap["status"] = "filled"
                    stats.gaps_filled += 1
                    stats.segments_added += len(collected)
                else:
                    per_gap["status"] = "empty_after_filter"
                stats.fill_audio_sec += (ge - gs)
            except Exception as e:
                logger.info(
                    "gap_filler: transcribe slice [%.1f..%.1f] failed: %s",
                    gs, ge, e,
                )
                per_gap["status"] = f"transcribe_failed: {type(e).__name__}"
            finally:
                try:
                    os.unlink(slice_path)
                except OSError:
                    pass
        except Exception as e:
            per_gap["status"] = f"slice_exception: {type(e).__name__}"
            logger.info("gap_filler: gap processing exception: %s", e)

        stats.per_gap.append(per_gap)

    # 7. Merge & return. We sort by start, but we DO NOT dedupe or
    #    consolidate — those are the job of the caller, who has the
    #    full context (turn gaps, speaker labels, etc.).
    if not added_segments:
        stats.coverage_after = stats.coverage_before
        stats.elapsed_sec = time.monotonic() - wall_start
        logger.info(
            "gap_filler: attempted %d gap(s), filled 0 — coverage unchanged at %.1f%%",
            stats.gaps_attempted, stats.coverage_before * 100,
        )
        return segments, stats

    merged: list = list(segments) + list(added_segments)
    merged.sort(key=lambda s: (
        float(s["start"]) if isinstance(s, dict) else float(s.start)
    ))

    stats.coverage_after = compute_coverage(
        merged, vad_intervals, audio_duration,
    )[1]
    stats.elapsed_sec = time.monotonic() - wall_start

    delta = (stats.coverage_after - stats.coverage_before) * 100
    logger.info(
        "gap_filler: filled %d/%d gaps, added %d segments (+%.1fs audio), "
        "voiced_coverage %.1f%% → %.1f%% (+%.1fpp) in %.1fs",
        stats.gaps_filled, stats.gaps_attempted,
        stats.segments_added, stats.fill_audio_sec,
        stats.coverage_before * 100, stats.coverage_after * 100,
        delta, stats.elapsed_sec,
    )

    return merged, stats


async def fill_transcript_gaps(
    audio_path: str,
    segments: list,
    audio_duration: float,
    language: str = "",
    task: str = "transcribe",
    is_animated: bool = False,
    initial_prompt: str = "",
    **kwargs,
) -> tuple[list, GapFillStats]:
    """Async wrapper — runs the sync worker on the default executor.

    Kept async so callers in the existing pipeline (all async) can
    ``await`` this without blocking the event loop on a multi-minute
    Whisper pass.
    """
    import asyncio
    import functools

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        functools.partial(
            fill_transcript_gaps_sync,
            audio_path=audio_path,
            segments=segments,
            audio_duration=audio_duration,
            language=language,
            task=task,
            is_animated=is_animated,
            initial_prompt=initial_prompt,
            **kwargs,
        ),
    )
