"""Temporal Coverage Ledger for TACT (Total Audio Coverage Transcription).

The ledger is a millisecond-resolution record of every input audio frame
and what is known about it. Every TACT stage reads from and writes to
the ledger; coverage stops being a derived metric and becomes a
first-class invariant the pipeline maintains as it runs.

Phase 1 scope (this file):
  * Pure data structure + claim/query/audit API.
  * ``from_segments`` to bootstrap the ledger from the existing post-
    gap-fill TranscriptSegment list. This is purely observability —
    the pipeline's transcript output is unchanged in Phase 1.
  * ``to_report_dict`` produces the JSON blob persisted on
    ``JobResult.coverage_report``.

Phase 2+ extends this with: contested/quarantined claims from the
hallucination filter, low_confidence claims from the consensus
reconciler, ``to_segments`` round-trip after the escalation ladder,
and word-level claims from the disjoint-offset reconciler.

Storage model
-------------
Run-length encoded list of ``LedgerSpan`` records, sorted by
``start_ms``, disjoint by construction. Bin width is purely a
reporting parameter (we report ``coverage_ratio`` against the
``audio_duration_ms / bin_ms`` denominator); the spans themselves
are stored at full ms precision.

For a 90-minute file with thousands of segments the RLE storage is
``O(spans)`` not ``O(bins)`` — typical memory footprint is a few KB
to a few hundred KB, well under the 5 MB target named in the design
doc. A hard cap of ``_MAX_SPANS`` guards against pathological input.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

from backend.models import TranscriptSegment

logger = logging.getLogger(__name__)


# Status values. Phase 1 only writes ``covered_speech`` and ``uncovered``
# (the latter being the implicit complement). The remaining values are
# reserved keys so ``to_report_dict()``'s schema is forward-compatible
# with Phase 2+ — they appear in the histogram with count 0 today.
LedgerStatus = Literal[
    "covered_speech",
    "covered_event",
    "covered_silence",
    "low_confidence",
    "contested",
    "uncovered",
    "quarantined",
]

ContentType = Literal["word", "phrase", "event", "silence", "unintelligible"]

# Statuses that count as "positively labeled" for coverage_ratio.
# Matches the TACT design doc §11 definition.
COVERED_STATUSES: frozenset[str] = frozenset({
    "covered_speech",
    "covered_event",
    "covered_silence",
})

# All statuses, in the canonical order used by ``status_distribution``.
ALL_STATUSES: tuple[str, ...] = (
    "covered_speech",
    "covered_event",
    "covered_silence",
    "low_confidence",
    "contested",
    "uncovered",
    "quarantined",
)

# Hard cap on stored spans. Aggressive coalescing keeps real-world
# ledgers far below this; we just need a guardrail.
_MAX_SPANS = 100_000


@dataclass
class LedgerSpan:
    """One contiguous range of the ledger with a single status / source."""

    start_ms: int
    end_ms: int
    status: str
    content: Optional[str] = None
    content_type: str = "phrase"
    source_pass: str = ""
    confidence: float = 0.0
    speaker: Optional[str] = None
    flags: list[str] = field(default_factory=list)

    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    def to_dict(self) -> dict:
        d = {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "status": self.status,
            "content_type": self.content_type,
            "source_pass": self.source_pass,
            "confidence": round(float(self.confidence), 4),
        }
        if self.content is not None:
            d["content"] = self.content
        if self.speaker is not None:
            d["speaker"] = self.speaker
        if self.flags:
            d["flags"] = list(self.flags)
        return d


def _seg_confidence(seg) -> float:
    """Derive confidence from a TranscriptSegment-like object.

    Mirrors the gap-filler's derivation
    (transcription_gap_filler.py:540) so the two confidences are on
    the same scale.
    """
    # Direct confidence has priority.
    conf = getattr(seg, "confidence", None)
    if conf is None and isinstance(seg, dict):
        conf = seg.get("confidence")
    if conf is not None:
        try:
            return max(0.0, min(1.0, float(conf)))
        except (TypeError, ValueError):
            pass
    # Fall back to ``1 + avg_logprob`` like the gap-filler.
    alp = getattr(seg, "avg_logprob", None)
    if alp is None and isinstance(seg, dict):
        alp = seg.get("avg_logprob")
    if alp is not None:
        try:
            return max(0.0, min(1.0, 1.0 + float(alp)))
        except (TypeError, ValueError):
            pass
    return 1.0


class CoverageLedger:
    """Run-length encoded coverage ledger over ``[0, audio_duration_ms]``.

    Invariants:
      * ``_spans`` is sorted by ``start_ms`` and contains no overlaps.
      * Every ms in ``[0, audio_duration_ms]`` not present in ``_spans``
        is implicitly ``uncovered``.
      * ``claim`` resolves overlap by confidence: a higher-confidence
        new span supersedes any overlapping existing span; an equal or
        lower one is rejected and the conflict is logged as ``contested``
        (only when ``allow_overlap=False``, which is the default).
    """

    def __init__(self, audio_duration_ms: int, bin_ms: int = 20) -> None:
        if audio_duration_ms < 0:
            raise ValueError(
                f"audio_duration_ms must be >= 0, got {audio_duration_ms}"
            )
        if bin_ms <= 0:
            raise ValueError(f"bin_ms must be > 0, got {bin_ms}")
        self.audio_duration_ms = int(audio_duration_ms)
        self.bin_ms = int(bin_ms)
        self._spans: list[LedgerSpan] = []
        # Tracks claim attempts that lost to a higher-confidence
        # incumbent. Reported in ``to_report_dict`` as
        # ``contested_attempts``. Phase 1 won't normally see any of
        # these — they're a Phase 2+ signal.
        self._contested_attempts = 0

    # ── Internal helpers ──────────────────────────────────────────

    def _clamp(self, start_ms: int, end_ms: int) -> tuple[int, int]:
        s = max(0, int(start_ms))
        e = min(self.audio_duration_ms, int(end_ms))
        return s, e

    def _coalesce(self) -> None:
        """Merge adjacent spans with identical status/source/content.

        Called after every claim. Keeps the span count from blowing up
        when many small adjacent claims arrive (e.g. consecutive words
        from the same pass).
        """
        if len(self._spans) < 2:
            return
        merged: list[LedgerSpan] = [self._spans[0]]
        for s in self._spans[1:]:
            prev = merged[-1]
            if (
                prev.end_ms == s.start_ms
                and prev.status == s.status
                and prev.source_pass == s.source_pass
                and prev.content_type == s.content_type
                and prev.speaker == s.speaker
                and prev.content == s.content
                and abs(prev.confidence - s.confidence) < 1e-6
                and prev.flags == s.flags
            ):
                prev.end_ms = s.end_ms
            else:
                merged.append(s)
        self._spans = merged

    # ── Mutators ──────────────────────────────────────────────────

    def claim(self, span: LedgerSpan, *, allow_overlap: bool = False) -> bool:
        """Claim a span. Returns True if any portion of the span was written.

        Overlap resolution: incumbents at >= the new span's confidence
        keep their bins; bins where the new span has strictly higher
        confidence are reassigned to the new span. Lost regions are
        counted in ``_contested_attempts`` for the report.

        ``allow_overlap=True`` skips overlap resolution and just
        unconditionally writes — used by tests; no production caller
        in Phase 1.
        """
        if span is None:
            return False
        start_ms, end_ms = self._clamp(span.start_ms, span.end_ms)
        if end_ms <= start_ms:
            return False
        if len(self._spans) >= _MAX_SPANS:
            logger.warning(
                "CoverageLedger: span cap (%d) reached — dropping claim "
                "[%dms..%dms] status=%s source=%s",
                _MAX_SPANS, start_ms, end_ms, span.status, span.source_pass,
            )
            return False

        # Normalize the input span so we always store an instance with
        # the clamped bounds.
        new_span = LedgerSpan(
            start_ms=start_ms,
            end_ms=end_ms,
            status=span.status,
            content=span.content,
            content_type=span.content_type,
            source_pass=span.source_pass,
            confidence=float(span.confidence),
            speaker=span.speaker,
            flags=list(span.flags),
        )

        if allow_overlap or not self._spans:
            self._spans.append(new_span)
            self._spans.sort(key=lambda s: s.start_ms)
            self._coalesce()
            return True

        # Resolve overlaps. Build a new span list by walking
        # incumbents and either keeping them whole, splitting them, or
        # replacing them with the new claim.
        rebuilt: list[LedgerSpan] = []
        new_start = new_span.start_ms
        new_end = new_span.end_ms
        any_written = False

        for inc in self._spans:
            if inc.end_ms <= new_start or inc.start_ms >= new_end:
                rebuilt.append(inc)
                continue
            # There is overlap with ``inc``. Decide which piece wins.
            if new_span.confidence > inc.confidence:
                # New wins on the overlapping region. Keep non-
                # overlapping incumbent fragments.
                if inc.start_ms < new_start:
                    rebuilt.append(LedgerSpan(
                        start_ms=inc.start_ms,
                        end_ms=new_start,
                        status=inc.status,
                        content=inc.content,
                        content_type=inc.content_type,
                        source_pass=inc.source_pass,
                        confidence=inc.confidence,
                        speaker=inc.speaker,
                        flags=list(inc.flags),
                    ))
                if inc.end_ms > new_end:
                    rebuilt.append(LedgerSpan(
                        start_ms=new_end,
                        end_ms=inc.end_ms,
                        status=inc.status,
                        content=inc.content,
                        content_type=inc.content_type,
                        source_pass=inc.source_pass,
                        confidence=inc.confidence,
                        speaker=inc.speaker,
                        flags=list(inc.flags),
                    ))
                self._contested_attempts += 1
            else:
                # Incumbent wins. Carve the new span around it.
                if inc.start_ms > new_start:
                    rebuilt.append(LedgerSpan(
                        start_ms=new_start,
                        end_ms=min(inc.start_ms, new_end),
                        status=new_span.status,
                        content=new_span.content,
                        content_type=new_span.content_type,
                        source_pass=new_span.source_pass,
                        confidence=new_span.confidence,
                        speaker=new_span.speaker,
                        flags=list(new_span.flags),
                    ))
                    any_written = True
                rebuilt.append(inc)
                new_start = max(new_start, inc.end_ms)
                if new_start >= new_end:
                    # New span is fully consumed by incumbents.
                    self._contested_attempts += 1
                    self._spans = sorted(rebuilt, key=lambda s: s.start_ms)
                    self._coalesce()
                    return any_written

        # Append whatever's left of the new span after walking all
        # incumbents — this is either the whole span (no overlap) or
        # the trailing portion past the last consumed incumbent.
        if new_start < new_end and new_span.confidence > -1.0:
            rebuilt.append(LedgerSpan(
                start_ms=new_start,
                end_ms=new_end,
                status=new_span.status,
                content=new_span.content,
                content_type=new_span.content_type,
                source_pass=new_span.source_pass,
                confidence=new_span.confidence,
                speaker=new_span.speaker,
                flags=list(new_span.flags),
            ))
            any_written = True

        self._spans = sorted(rebuilt, key=lambda s: s.start_ms)
        self._coalesce()
        return any_written

    def from_segments(
        self,
        segments: Iterable,
        source_pass: str = "whisper_main",
        *,
        status: str = "covered_speech",
        flag_key: Optional[str] = None,
    ) -> None:
        """Bulk-claim from a segment-shaped list (dict or TranscriptSegment).

        Phase 1 entrypoint: bootstraps the ledger from the post-Whisper
        segment list. Each segment becomes one span at
        ``content_type="phrase"`` (multi-word).

        Phase 2 additions:

        ``status`` is configurable so the same primitive can be used to
        claim quarantined segments with ``status="quarantined"``. The
        ledger doesn't dedupe across different statuses, so quarantined
        spans coexist with covered_speech spans even at the same
        timestamp.

        ``flag_key`` names a key on each segment dict whose value is
        appended to ``LedgerSpan.flags``. Used by the pipeline to
        forward ``quarantine_reason`` from the hallucination filter
        into the ledger so it survives into ``to_report_dict``. When
        ``None`` (default), no flag is added — preserves Phase 1
        behavior on existing callers.
        """
        for seg in segments or []:
            if seg is None:
                continue
            if isinstance(seg, dict):
                start = seg.get("start")
                end = seg.get("end")
                text = (seg.get("text") or "").strip()
                speaker = seg.get("speaker")
                flag_value = seg.get(flag_key) if flag_key else None
            else:
                start = getattr(seg, "start", None)
                end = getattr(seg, "end", None)
                text = (getattr(seg, "text", "") or "").strip()
                speaker = getattr(seg, "speaker", None)
                flag_value = getattr(seg, flag_key, None) if flag_key else None
            if start is None or end is None:
                continue
            try:
                start_ms = int(round(float(start) * 1000.0))
                end_ms = int(round(float(end) * 1000.0))
            except (TypeError, ValueError):
                continue
            if end_ms <= start_ms:
                continue
            self.claim(LedgerSpan(
                start_ms=start_ms,
                end_ms=end_ms,
                status=status,
                content=text or None,
                content_type="phrase",
                source_pass=source_pass,
                confidence=_seg_confidence(seg),
                speaker=speaker,
                flags=[str(flag_value)] if flag_value else [],
            ))

    def claim_word(
        self,
        start_ms: int,
        end_ms: int,
        word: str,
        confidence: float,
        *,
        source_pass: str,
        speaker: Optional[str] = None,
        flags: Optional[list[str]] = None,
        status: str = "covered_speech",
    ) -> bool:
        """Word-level claim helper. Used by the Phase 3 word-level
        reconciler and by Rung 4 forced alignment when emitting
        recovered words from a forced-alignment pass.

        Wraps ``claim`` with ``content_type="word"`` so the ledger's
        ``to_report_dict`` source-pass distribution counts word-level
        coverage separately from phrase-level coverage."""
        return self.claim(LedgerSpan(
            start_ms=start_ms,
            end_ms=end_ms,
            status=status,
            content=word,
            content_type="word",
            source_pass=source_pass,
            confidence=float(confidence),
            speaker=speaker,
            flags=list(flags or []),
        ))

    # ── Queries / audit ──────────────────────────────────────────

    def query(
        self,
        status_filter: Optional[Iterable[str]] = None,
        min_duration_ms: int = 0,
    ) -> list[tuple[int, int]]:
        """Return ``[(start_ms, end_ms)]`` intervals matching ``status_filter``.

        ``status_filter=None`` → every covered span. Pass
        ``{"uncovered"}`` to get the complement of all stored spans
        (used by the escalation ladder in Phase 2).
        """
        wanted: Optional[set[str]]
        if status_filter is None:
            wanted = None
        else:
            wanted = set(status_filter)

        intervals: list[tuple[int, int]] = []

        if wanted is None or "uncovered" not in wanted:
            for s in self._spans:
                if wanted is None or s.status in wanted:
                    if s.duration_ms() >= min_duration_ms:
                        intervals.append((s.start_ms, s.end_ms))
            return intervals

        # ``uncovered`` is the implicit complement; compute it.
        cursor = 0
        for s in self._spans:
            if s.start_ms > cursor:
                gap_end = min(s.start_ms, self.audio_duration_ms)
                if gap_end > cursor and (gap_end - cursor) >= min_duration_ms:
                    intervals.append((cursor, gap_end))
            cursor = max(cursor, s.end_ms)
            if cursor >= self.audio_duration_ms:
                break
        if cursor < self.audio_duration_ms:
            tail = self.audio_duration_ms - cursor
            if tail >= min_duration_ms:
                intervals.append((cursor, self.audio_duration_ms))

        # If other statuses were also requested, fold them in.
        if wanted - {"uncovered"}:
            for s in self._spans:
                if s.status in wanted and s.duration_ms() >= min_duration_ms:
                    intervals.append((s.start_ms, s.end_ms))
            intervals.sort()

        return intervals

    def coverage_ratio(
        self,
        statuses: Optional[Iterable[str]] = None,
    ) -> float:
        """Covered duration / total audio duration.

        ``statuses=None`` → uses the canonical COVERED_STATUSES.
        """
        if self.audio_duration_ms <= 0:
            return 0.0
        wanted = set(statuses) if statuses is not None else set(COVERED_STATUSES)
        covered = sum(
            s.duration_ms() for s in self._spans if s.status in wanted
        )
        return min(1.0, covered / self.audio_duration_ms)

    def voiced_coverage_ratio(
        self,
        vad_intervals: list[tuple[float, float]],
    ) -> float:
        """Fraction of VAD-voiced ms covered by ``covered_speech`` spans.

        Mirrors ``transcription_gap_filler.compute_coverage`` second
        return value. Useful for cross-checking Phase 1 ledger output
        against the gap-filler's own log line.
        """
        if not vad_intervals or self.audio_duration_ms <= 0:
            return 0.0
        # Convert VAD seconds → ms intervals, clamp, and merge.
        vad_ms: list[tuple[int, int]] = []
        for vs, ve in vad_intervals:
            try:
                a = max(0, int(round(float(vs) * 1000.0)))
                b = min(self.audio_duration_ms, int(round(float(ve) * 1000.0)))
            except (TypeError, ValueError):
                continue
            if b > a:
                vad_ms.append((a, b))
        if not vad_ms:
            return 0.0
        vad_ms.sort()
        merged: list[list[int]] = [list(vad_ms[0])]
        for a, b in vad_ms[1:]:
            if a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        vad_total = sum(b - a for a, b in merged)
        if vad_total <= 0:
            return 0.0

        speech_spans = [
            (s.start_ms, s.end_ms) for s in self._spans
            if s.status == "covered_speech"
        ]
        # Sweep-line intersection.
        covered = 0
        i = j = 0
        while i < len(speech_spans) and j < len(merged):
            ss, se = speech_spans[i]
            vs, ve = merged[j]
            lo, hi = max(ss, vs), min(se, ve)
            if hi > lo:
                covered += hi - lo
            if se < ve:
                i += 1
            else:
                j += 1
        return covered / vad_total

    def status_distribution(self) -> dict[str, int]:
        """Total ms per status across the ledger.

        The implicit ``uncovered`` ms are computed as
        ``audio_duration_ms - sum(stored span durations)``. Every key
        in ``ALL_STATUSES`` is present, possibly with value 0, so the
        report schema is stable.
        """
        dist: dict[str, int] = {k: 0 for k in ALL_STATUSES}
        stored_ms = 0
        for s in self._spans:
            d = s.duration_ms()
            dist[s.status] = dist.get(s.status, 0) + d
            stored_ms += d
        # Anything not stored is implicitly uncovered.
        if self.audio_duration_ms > stored_ms:
            dist["uncovered"] += self.audio_duration_ms - stored_ms
        return dist

    def confidence_histogram(self, n_bins: int = 10) -> list[int]:
        """Bin counts (in ms) of covered spans by confidence in [0, 1].

        Bin ``i`` covers ``[i/n_bins, (i+1)/n_bins)``; the last bin is
        closed on the right. Useful for "how much of my transcript is
        high-confidence?" charts.
        """
        if n_bins <= 0:
            return []
        bins = [0] * n_bins
        for s in self._spans:
            if s.status not in COVERED_STATUSES:
                continue
            c = max(0.0, min(1.0, float(s.confidence)))
            idx = min(n_bins - 1, int(c * n_bins))
            bins[idx] += s.duration_ms()
        return bins

    def source_pass_breakdown(self) -> dict[str, int]:
        """Total covered ms attributed to each ``source_pass`` label."""
        out: dict[str, int] = {}
        for s in self._spans:
            if s.status not in COVERED_STATUSES:
                continue
            key = s.source_pass or "unknown"
            out[key] = out.get(key, 0) + s.duration_ms()
        return out

    # ── Output ───────────────────────────────────────────────────

    def to_segments(self) -> list[TranscriptSegment]:
        """Render covered-speech spans back to TranscriptSegment.

        Phase 1 callers won't normally use this — the existing pipeline
        is the source of truth for segments. Provided so Phase 2's
        ladder can render escalation results without each rung
        knowing about the segment dataclass shape.
        """
        out: list[TranscriptSegment] = []
        for s in self._spans:
            if s.status != "covered_speech":
                continue
            text = s.content or ""
            out.append(TranscriptSegment(
                start=round(s.start_ms / 1000.0, 3),
                end=round(s.end_ms / 1000.0, 3),
                text=text,
                speaker=s.speaker or "Speaker ?",
                confidence=round(s.confidence, 3),
            ))
        return out

    def to_report_dict(self) -> dict:
        """The JSON blob persisted to ``JobResult.coverage_report``.

        Schema is intentionally stable across phases; Phase 2+ only
        adds keys, never removes them. Bin-derived ratios are reported
        instead of raw bin counts to keep the payload small on long
        files.
        """
        dist = self.status_distribution()
        total = self.audio_duration_ms or 1
        covered_ms = sum(dist[s] for s in COVERED_STATUSES)
        contested_ms = dist.get("contested", 0) + dist.get("low_confidence", 0)
        return {
            "version": 1,
            "audio_duration_ms": self.audio_duration_ms,
            "bin_ms": self.bin_ms,
            "span_count": len(self._spans),
            "coverage_ratio": round(min(1.0, covered_ms / total), 4),
            "voiced_coverage_ratio": None,  # Filled by caller when VAD known.
            "uncovered_ratio": round(dist.get("uncovered", 0) / total, 4),
            "contested_ratio": round(contested_ms / total, 4),
            "quarantined_ratio": round(dist.get("quarantined", 0) / total, 4),
            "status_distribution_ms": dist,
            "confidence_histogram": self.confidence_histogram(10),
            "source_pass_ms": self.source_pass_breakdown(),
            "contested_attempts": self._contested_attempts,
        }

    # ── Misc ─────────────────────────────────────────────────────

    @property
    def spans(self) -> list[LedgerSpan]:
        """Read-only view of stored spans, sorted by start_ms."""
        return list(self._spans)

    def __len__(self) -> int:
        return len(self._spans)
