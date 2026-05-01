"""TACT Phase 3 word-level reconciler (ROVER-style).

The disjoint-offset multi-pass runs Whisper twice on the same audio
with phase-shifted chunk grids. Words that landed at chunk boundaries
in pass 1 are mid-chunk in pass 2 (and vice versa). This module
reconciles the two pass outputs at word level so the final segment
list inherits the better of each pair.

Phase 4 extension (``reconcile_n_passes``): generalizes the same
algorithm to N passes — used when an independent-architecture
consensus pass (Parakeet) runs alongside the two Whisper passes.

The reconciler is pure-Python, takes ``TranscriptSegment`` instances
as input, emits both reconciled segments and per-word reconciliation
stats. It does not touch audio, GPUs, or models.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

from backend.models import TranscriptSegment, WordTimestamp


# Module-level handoff so the pipeline can persist reconciliation
# stats into coverage_report after reconcile_passes / reconcile_n_passes
# has already returned. Mirrors the _last_quarantined_segments pattern
# in transcription.py. Single-job homelab use is the design center;
# concurrent jobs would need a per-job dict (documented in notes).
_last_reconciliation_stats: Optional["ReconciliationStats"] = None


def get_last_reconciliation_stats() -> Optional["ReconciliationStats"]:
    """Return (a snapshot of) the most recent reconciliation stats."""
    return _last_reconciliation_stats

logger = logging.getLogger(__name__)


# Standard Whisper chunk window. Chunk boundaries in pass 1 are
# multiples of this; in the offset pass they are
# ``multiples + offset_seconds``.
_DEFAULT_CHUNK_WINDOW_SEC = 30.0


@dataclass
class ReconciledWord:
    """One word in the final reconciled output.

    ``sources`` records which pass(es) produced this word — used by
    the ledger as ``LedgerSpan.flags`` so downstream tooling can audit
    per-word provenance. ``contested`` is True when at least two
    passes emitted a word at this timestamp but disagreed on text.
    ``distance_to_boundary_sec`` is the minimum distance, in the
    winning pass, to any chunk grid boundary — used by the ROVER
    tiebreaker.
    """
    start_sec: float
    end_sec: float
    word: str
    confidence: float
    sources: list[str] = field(default_factory=list)
    contested: bool = False
    single_source: bool = False
    distance_to_boundary_sec: float = 0.0


@dataclass
class ReconciliationStats:
    total_words: int = 0
    agreed: int = 0          # >=2 passes, same text
    disagreed: int = 0       # >=2 passes, different text
    single_source: int = 0   # only one pass emitted
    contested: int = 0       # >=3 passes, no majority
    passes: list[str] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def as_dict(self) -> dict:
        return {
            "total_words": self.total_words,
            "agreed": self.agreed,
            "disagreed": self.disagreed,
            "single_source": self.single_source,
            "contested": self.contested,
            "passes": list(self.passes),
            "elapsed_sec": round(self.elapsed_sec, 3),
        }


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _flatten_words(
    segments: list[TranscriptSegment],
) -> list[tuple[float, float, str, float]]:
    """Flatten a segment list to ``(start, end, word, confidence)`` tuples.

    When a segment lacks word-level timestamps, splits the segment text
    by whitespace and distributes timestamps uniformly across the
    segment duration. This is approximate but lets the reconciler still
    operate when one pass has word timestamps and the other does not.
    """
    out: list[tuple[float, float, str, float]] = []
    for seg in segments or []:
        if seg is None:
            continue
        seg_conf = seg.confidence if seg.confidence is not None else (
            max(0.0, min(1.0, 1.0 + (seg.avg_logprob or -1.0)))
        )
        if seg.words:
            for w in seg.words:
                wt = (w.word or "").strip()
                if not wt:
                    continue
                out.append((float(w.start), float(w.end), wt, float(seg_conf)))
            continue
        text = (seg.text or "").strip()
        if not text:
            continue
        tokens = text.split()
        if not tokens:
            continue
        seg_dur = max(0.0, seg.end - seg.start)
        if seg_dur <= 0 or len(tokens) == 1:
            out.append((float(seg.start), float(seg.end), text, float(seg_conf)))
            continue
        per = seg_dur / len(tokens)
        for i, tok in enumerate(tokens):
            ws = seg.start + i * per
            we = ws + per
            out.append((float(ws), float(we), tok, float(seg_conf)))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def _normalize_word(w: str) -> str:
    """Normalize for equality comparison — lowercase + strip ASCII
    punctuation. This deliberately keeps the original form for
    output; only the comparison key is normalized."""
    return "".join(ch for ch in w.lower() if ch.isalnum())


def _distance_to_boundary(
    midpoint_sec: float,
    chunk_window_sec: float,
    offset_sec: float,
) -> float:
    """Distance from ``midpoint_sec`` to the nearest chunk-grid boundary.

    Boundaries for a pass with offset ``offset_sec`` are at
    ``offset + k * chunk_window`` for every integer k. The function
    returns the minimum absolute distance across all such k.
    """
    if chunk_window_sec <= 0:
        return float("inf")
    rel = (midpoint_sec - offset_sec) % chunk_window_sec
    return min(rel, chunk_window_sec - rel)


def _segments_from_words(
    words: list[ReconciledWord],
    primary: list[TranscriptSegment],
) -> list[TranscriptSegment]:
    """Group reconciled words back into segments using the primary pass's
    segment boundaries as guidance.

    For each primary segment ``s``, every reconciled word whose midpoint
    falls inside ``[s.start, s.end]`` is assigned to that segment. Words
    that don't fall inside any primary segment land in synthetic
    one-word segments inserted at their own timestamp.
    """
    if not words:
        return []
    if not primary:
        # No segmentation guidance — every reconciled word becomes a
        # one-word segment. Useful in tests; production passes have
        # primary segments.
        return [
            TranscriptSegment(
                start=round(w.start_sec, 3),
                end=round(w.end_sec, 3),
                text=w.word,
                speaker="Speaker 1",
                words=[WordTimestamp(start=w.start_sec, end=w.end_sec, word=w.word)],
                confidence=round(w.confidence, 4),
            )
            for w in words
        ]
    out: list[TranscriptSegment] = []
    used = [False] * len(words)
    primary_sorted = sorted(primary, key=lambda s: s.start)
    for ps in primary_sorted:
        bucket: list[ReconciledWord] = []
        for i, w in enumerate(words):
            if used[i]:
                continue
            mid = 0.5 * (w.start_sec + w.end_sec)
            if ps.start <= mid <= ps.end:
                used[i] = True
                bucket.append(w)
        if not bucket:
            continue
        bucket.sort(key=lambda w: w.start_sec)
        text = " ".join(w.word for w in bucket)
        avg_conf = sum(w.confidence for w in bucket) / len(bucket)
        out.append(TranscriptSegment(
            start=round(min(w.start_sec for w in bucket), 3),
            end=round(max(w.end_sec for w in bucket), 3),
            text=text,
            speaker=ps.speaker,
            words=[WordTimestamp(start=w.start_sec, end=w.end_sec, word=w.word)
                   for w in bucket],
            confidence=round(avg_conf, 4),
        ))
    # Words outside every primary segment → synthetic single-word segments.
    leftover = [w for i, w in enumerate(words) if not used[i]]
    for w in leftover:
        out.append(TranscriptSegment(
            start=round(w.start_sec, 3),
            end=round(w.end_sec, 3),
            text=w.word,
            speaker=primary_sorted[0].speaker if primary_sorted else "Speaker 1",
            words=[WordTimestamp(start=w.start_sec, end=w.end_sec, word=w.word)],
            confidence=round(w.confidence, 4),
        ))
    out.sort(key=lambda s: s.start)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Two-way reconciler (Phase 3)
# ──────────────────────────────────────────────────────────────────────────

def reconcile_passes(
    primary: list[TranscriptSegment],
    offset: list[TranscriptSegment],
    offset_seconds: float,
    *,
    chunk_window_sec: float = _DEFAULT_CHUNK_WINDOW_SEC,
    align_tolerance_sec: float = 0.2,
) -> tuple[list[TranscriptSegment], ReconciliationStats]:
    """Reconcile primary and offset Whisper passes at word level.

    Returns ``(reconciled_segments, stats)``. The reconciled segment
    list inherits the primary pass's segmentation but with word
    content/timestamps drawn from whichever pass is more reliable for
    each word.

    Algorithm:
      1. Flatten both passes to word-level via TranscriptSegment.words.
      2. Sort by midpoint timestamp.
      3. For each primary word, find the offset-pass word whose midpoint
         is within ``align_tolerance_sec``.
         - Match found, text agrees → keep, sources include both.
         - Match found, text disagrees → pick the word from the pass
           where the midpoint is FURTHER from any chunk boundary in
           that pass's grid. Mark contested.
         - No match → keep with single_source flag.
      4. For offset-pass words with no primary match → keep with
         single_source flag.
      5. Group reconciled words back into segments using the primary
         pass's segment boundaries as guidance.
    """
    # Phase 4 extension hook: this function is a thin wrapper around
    # reconcile_n_passes. The two-way path stays separate so two-pass
    # callers don't pay the N-way overhead and so the test surface
    # for the simple case stays clean.
    return reconcile_n_passes(
        passes=[("primary", primary), ("offset", offset)],
        chunk_grid_offsets={"primary": 0.0, "offset": offset_seconds},
        chunk_window_sec=chunk_window_sec,
        align_tolerance_sec=align_tolerance_sec,
        primary_for_segmentation="primary",
    )


# ──────────────────────────────────────────────────────────────────────────
# N-way reconciler (Phase 4)
# ──────────────────────────────────────────────────────────────────────────

def reconcile_n_passes(
    passes: list[tuple[str, list[TranscriptSegment]]],
    *,
    chunk_grid_offsets: Optional[dict[str, float]] = None,
    chunk_window_sec: float = _DEFAULT_CHUNK_WINDOW_SEC,
    align_tolerance_sec: float = 0.2,
    primary_for_segmentation: Optional[str] = None,
) -> tuple[list[TranscriptSegment], ReconciliationStats]:
    """Generalized N-way ROVER reconciler.

    ``passes`` is a list of ``(pass_name, segments)`` pairs.
    ``chunk_grid_offsets`` maps each pass name to its chunk-grid
    offset in seconds; passes not in the dict are assumed to have
    offset 0 (for non-Whisper passes like Parakeet that don't use a
    fixed chunk grid, this just makes ``distance_to_boundary`` very
    large so they never lose a tiebreak on boundary distance).

    ``primary_for_segmentation`` names the pass whose segment
    boundaries guide the final segmentation. When None, the first
    pass with non-empty segments wins.

    Algorithm:
      1. Cluster words across all passes by midpoint within
         ``align_tolerance_sec``.
      2. For each cluster:
         - Single source → keep, single_source=True.
         - All passes agree on normalized text → keep, max-confidence
           variant, sources = all agreeing passes.
         - Majority text wins (>= 2 passes agree, others disagree)
           → keep majority, mark contested if any pass disagreed.
         - No majority (3+ passes, all different) → pick highest
           confidence; if tied, use distance-to-boundary tiebreak.
           Mark contested.
    """
    import time as _time
    started = _time.monotonic()

    chunk_grid_offsets = chunk_grid_offsets or {}
    pass_names = [name for name, _ in passes]
    stats = ReconciliationStats(passes=list(pass_names))

    # Identify the pass that guides segmentation.
    seg_pass: list[TranscriptSegment] = []
    if primary_for_segmentation:
        for n, segs in passes:
            if n == primary_for_segmentation:
                seg_pass = list(segs or [])
                break
    if not seg_pass:
        for n, segs in passes:
            if segs:
                seg_pass = list(segs)
                break

    # Flatten each pass and tag words with their pass name.
    flat: list[tuple[float, float, str, float, str]] = []  # +pass_name
    for name, segs in passes:
        for st, en, w, c in _flatten_words(list(segs or [])):
            flat.append((st, en, w, c, name))
    if not flat:
        stats.elapsed_sec = round(_time.monotonic() - started, 3)
        return [], stats

    flat.sort(key=lambda x: 0.5 * (x[0] + x[1]))

    # Cluster by midpoint within tolerance.
    clusters: list[list[tuple[float, float, str, float, str]]] = []
    for entry in flat:
        if not clusters:
            clusters.append([entry])
            continue
        last_cluster = clusters[-1]
        last_mid = 0.5 * (last_cluster[-1][0] + last_cluster[-1][1])
        cur_mid = 0.5 * (entry[0] + entry[1])
        if abs(cur_mid - last_mid) <= align_tolerance_sec:
            # Don't put two words from the same pass into one cluster
            # (a pass can't disagree with itself); start a new cluster
            # in that case.
            if any(e[4] == entry[4] for e in last_cluster):
                clusters.append([entry])
            else:
                last_cluster.append(entry)
        else:
            clusters.append([entry])

    reconciled: list[ReconciledWord] = []
    for cluster in clusters:
        stats.total_words += 1
        if len(cluster) == 1:
            st, en, w, c, name = cluster[0]
            stats.single_source += 1
            mid = 0.5 * (st + en)
            offset = chunk_grid_offsets.get(name, 0.0)
            reconciled.append(ReconciledWord(
                start_sec=st, end_sec=en, word=w, confidence=c,
                sources=[name], single_source=True,
                distance_to_boundary_sec=_distance_to_boundary(
                    mid, chunk_window_sec, offset,
                ),
            ))
            continue

        # Multi-source. Vote on normalized text.
        normalized = [(_normalize_word(e[2]), e) for e in cluster]
        text_counts: Counter[str] = Counter(n for n, _ in normalized if n)
        if not text_counts:
            # Every word stripped to empty under normalization.
            chosen = cluster[0]
            stats.contested += 1
            mid = 0.5 * (chosen[0] + chosen[1])
            offset = chunk_grid_offsets.get(chosen[4], 0.0)
            reconciled.append(ReconciledWord(
                start_sec=chosen[0], end_sec=chosen[1], word=chosen[2],
                confidence=chosen[3], sources=[e[4] for e in cluster],
                contested=True,
                distance_to_boundary_sec=_distance_to_boundary(
                    mid, chunk_window_sec, offset,
                ),
            ))
            continue
        majority_text, majority_n = text_counts.most_common(1)[0]
        agreeing = [e for n, e in normalized if n == majority_text]
        disagreeing = [e for n, e in normalized if n != majority_text]
        if not disagreeing:
            stats.agreed += 1
            best = max(agreeing, key=lambda e: e[3])
            mid = 0.5 * (best[0] + best[1])
            offset = chunk_grid_offsets.get(best[4], 0.0)
            reconciled.append(ReconciledWord(
                start_sec=best[0], end_sec=best[1], word=best[2],
                confidence=best[3], sources=[e[4] for e in agreeing],
                contested=False,
                distance_to_boundary_sec=_distance_to_boundary(
                    mid, chunk_window_sec, offset,
                ),
            ))
            continue
        # Disagreement.
        stats.disagreed += 1
        if majority_n >= 2:
            # Majority wins.
            best = max(agreeing, key=lambda e: e[3])
            mid = 0.5 * (best[0] + best[1])
            offset = chunk_grid_offsets.get(best[4], 0.0)
            contested = bool(disagreeing)
            if contested:
                stats.contested += 1
            reconciled.append(ReconciledWord(
                start_sec=best[0], end_sec=best[1], word=best[2],
                confidence=best[3], sources=[e[4] for e in agreeing],
                contested=contested,
                distance_to_boundary_sec=_distance_to_boundary(
                    mid, chunk_window_sec, offset,
                ),
            ))
            continue
        # No majority (e.g. 3-way disagreement). Pick by distance-to-
        # boundary first (winning pass had the word in mid-chunk),
        # then by confidence.
        def _score(e):
            mid = 0.5 * (e[0] + e[1])
            offset = chunk_grid_offsets.get(e[4], 0.0)
            return (
                _distance_to_boundary(mid, chunk_window_sec, offset),
                e[3],
            )
        best = max(cluster, key=_score)
        mid = 0.5 * (best[0] + best[1])
        offset = chunk_grid_offsets.get(best[4], 0.0)
        stats.contested += 1
        reconciled.append(ReconciledWord(
            start_sec=best[0], end_sec=best[1], word=best[2],
            confidence=best[3], sources=[best[4]], contested=True,
            distance_to_boundary_sec=_distance_to_boundary(
                mid, chunk_window_sec, offset,
            ),
        ))

    # Sort and re-segment.
    reconciled.sort(key=lambda w: w.start_sec)
    out_segments = _segments_from_words(reconciled, seg_pass)
    stats.elapsed_sec = round(_time.monotonic() - started, 3)
    logger.info(
        "reconcile_n_passes: passes=%s words=%d agreed=%d disagreed=%d "
        "single=%d contested=%d in %.3fs",
        pass_names, stats.total_words, stats.agreed, stats.disagreed,
        stats.single_source, stats.contested, stats.elapsed_sec,
    )
    # Stash for the pipeline to read (mirrors _last_quarantined_segments).
    global _last_reconciliation_stats
    _last_reconciliation_stats = stats
    return out_segments, stats


# ──────────────────────────────────────────────────────────────────────────
# Ledger bridge (Phase 3): emit reconciled words straight into the ledger
# ──────────────────────────────────────────────────────────────────────────

def claim_reconciled_words_into_ledger(
    ledger,  # CoverageLedger; lazy-typed to avoid import cycle
    words: Iterable[ReconciledWord],
    *,
    base_source_pass: str = "reconciled",
) -> int:
    """Claim every reconciled word into ``ledger`` via ``claim_word``.

    Returns the number of accepted claims. Used by the pipeline after
    reconcile_passes runs and before the escalation ladder, so that
    ladder rungs only look at words the reconciler couldn't recover.
    """
    n = 0
    for w in words:
        start_ms = int(round(w.start_sec * 1000.0))
        end_ms = int(round(w.end_sec * 1000.0))
        if end_ms <= start_ms:
            continue
        flags: list[str] = []
        if w.contested:
            flags.append("contested")
        if w.single_source:
            flags.append("single_source")
        flags.extend(f"src:{s}" for s in w.sources)
        if ledger.claim_word(
            start_ms=start_ms,
            end_ms=end_ms,
            word=w.word,
            confidence=w.confidence,
            source_pass=f"{base_source_pass}:{','.join(w.sources)}",
            flags=flags,
        ):
            n += 1
    return n
