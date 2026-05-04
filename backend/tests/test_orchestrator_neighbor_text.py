"""Tests for the orchestrator's per-interval neighbor text + the
reconciler stats accessor (Phase 2 wire-up Item 2).

The orchestrator uses ``dataclasses.replace`` to build a per-interval
EscalationContext so the caller's original ctx is not mutated. Rung
4 (forced alignment) reads neighbor text; this test confirms the
neighbor strings change between intervals and don't leak back into
the caller's ctx.

The reconciler stats accessor is a module-level handoff that mirrors
``_last_quarantined_segments`` in transcription.py.
"""
from __future__ import annotations

from backend.models import TranscriptSegment, WordTimestamp
from backend.services.escalation_ladder import (
    EscalationContext,
    LedgerSpan,
    RungResult,
    escalate_uncovered_intervals_sync,
)
from backend.services.transcription_ledger import CoverageLedger
from backend.services.transcription_reconciler import (
    get_last_reconciliation_stats,
    reconcile_passes,
)


# ──────────────────────────────────────────────────────────────────────────
# Per-interval neighbor text
# ──────────────────────────────────────────────────────────────────────────

def test_orchestrator_does_not_mutate_caller_ctx():
    ledger = CoverageLedger(audio_duration_ms=20_000)
    ledger.claim(LedgerSpan(
        start_ms=0, end_ms=5_000, status="covered_speech",
        content="hello there friend", source_pass="primary",
        confidence=0.9,
    ))
    ledger.claim(LedgerSpan(
        start_ms=10_000, end_ms=15_000, status="covered_speech",
        content="goodbye for now", source_pass="primary",
        confidence=0.9,
    ))
    ctx = EscalationContext(
        audio_path="/dev/null", audio_duration_ms=20_000,
        language="en", task="transcribe",
        neighbor_text_before="ORIGINAL_BEFORE",
        neighbor_text_after="ORIGINAL_AFTER",
    )

    # Custom rungs to inspect ctx as seen by each rung call.
    seen_neighbors: list[tuple[str, str]] = []

    def inspect(audio_path, start_ms, end_ms, rung_ctx):
        seen_neighbors.append((
            rung_ctx.neighbor_text_before,
            rung_ctx.neighbor_text_after,
        ))
        return RungResult(rung_name="inspect")

    rungs = [
        ("inspect", inspect),
        ("terminator",
         lambda ap, s, e, c: RungResult(
             rung_name="terminator",
             spans=[LedgerSpan(
                 start_ms=s, end_ms=e, status="covered_event",
                 content="[stub]", content_type="event",
                 source_pass="terminator", confidence=0.5,
             )],
         )),
    ]
    escalate_uncovered_intervals_sync(ledger, ctx, rungs=rungs)

    # The caller's ctx is unchanged.
    assert ctx.neighbor_text_before == "ORIGINAL_BEFORE"
    assert ctx.neighbor_text_after == "ORIGINAL_AFTER"
    # And the rung saw at least one non-original neighbor (the
    # neighbor text was filled in from the ledger by the orchestrator).
    seen_originals = sum(
        1 for b, a in seen_neighbors
        if b == "ORIGINAL_BEFORE" and a == "ORIGINAL_AFTER"
    )
    assert seen_originals == 0, f"caller ctx leaked into rungs: {seen_neighbors}"


def test_orchestrator_populates_neighbor_per_interval():
    """Two intervals between three covered spans → the rung sees
    different neighbor text on each interval (not the same stale
    string repeatedly)."""
    ledger = CoverageLedger(audio_duration_ms=30_000)
    ledger.claim(LedgerSpan(
        start_ms=0, end_ms=5_000, status="covered_speech",
        content="alpha beta gamma", source_pass="primary",
        confidence=0.9,
    ))
    ledger.claim(LedgerSpan(
        start_ms=10_000, end_ms=15_000, status="covered_speech",
        content="middle words here", source_pass="primary",
        confidence=0.9,
    ))
    ledger.claim(LedgerSpan(
        start_ms=20_000, end_ms=25_000, status="covered_speech",
        content="final part end", source_pass="primary",
        confidence=0.9,
    ))
    # Uncovered intervals: [5000, 10000] and [15000, 20000] and
    # [25000, 30000]. Three intervals; each has different neighbor
    # context.
    ctx = EscalationContext(
        audio_path="/dev/null", audio_duration_ms=30_000,
        language="en", task="transcribe",
    )
    seen: list[tuple[str, str]] = []

    def inspect(audio_path, start_ms, end_ms, rung_ctx):
        seen.append((
            rung_ctx.neighbor_text_before,
            rung_ctx.neighbor_text_after,
        ))
        return RungResult(rung_name="inspect")

    rungs = [
        ("inspect", inspect),
        ("terminator",
         lambda ap, s, e, c: RungResult(
             rung_name="terminator",
             spans=[LedgerSpan(
                 start_ms=s, end_ms=e, status="covered_event",
                 content="[stub]", content_type="event",
                 source_pass="terminator", confidence=0.5,
             )],
         )),
    ]
    escalate_uncovered_intervals_sync(ledger, ctx, rungs=rungs)
    assert len(seen) == 3
    # All three (before, after) tuples are distinct — the orchestrator
    # is genuinely refreshing per-interval, not reusing a stale ctx.
    assert len(set(seen)) == 3


# ──────────────────────────────────────────────────────────────────────────
# Reconciler stats accessor
# ──────────────────────────────────────────────────────────────────────────

def test_get_last_reconciliation_stats_after_reconcile():
    p = [TranscriptSegment(
        start=0.0, end=1.0, text="hello", speaker="Speaker 1",
        words=[WordTimestamp(start=0.0, end=1.0, word="hello")],
        confidence=0.9,
    )]
    o = [TranscriptSegment(
        start=0.0, end=1.0, text="hello", speaker="Speaker 1",
        words=[WordTimestamp(start=0.0, end=1.0, word="hello")],
        confidence=0.9,
    )]
    reconcile_passes(p, o, offset_seconds=15.0)
    stats = get_last_reconciliation_stats()
    assert stats is not None
    assert stats.total_words >= 1
    # The accessor returns the live object (not a copy); subsequent
    # reconciliation calls overwrite it. That's the documented
    # behavior — same as _last_quarantined_segments.


def test_get_last_reconciliation_stats_overwritten_each_call():
    """Two reconciliations in sequence: the accessor returns the
    second call's stats, not the first."""
    p1 = [TranscriptSegment(
        start=0.0, end=1.0, text="alpha", speaker="Speaker 1",
        words=[WordTimestamp(start=0.0, end=1.0, word="alpha")],
        confidence=0.9,
    )]
    o1 = [TranscriptSegment(
        start=0.0, end=1.0, text="alpha", speaker="Speaker 1",
        words=[WordTimestamp(start=0.0, end=1.0, word="alpha")],
        confidence=0.9,
    )]
    reconcile_passes(p1, o1, offset_seconds=15.0)
    first = get_last_reconciliation_stats()
    first_total = first.total_words

    p2 = [
        TranscriptSegment(
            start=0.0, end=0.5, text="one", speaker="Speaker 1",
            words=[WordTimestamp(start=0.0, end=0.5, word="one")], confidence=0.9,
        ),
        TranscriptSegment(
            start=0.5, end=1.0, text="two", speaker="Speaker 1",
            words=[WordTimestamp(start=0.5, end=1.0, word="two")], confidence=0.9,
        ),
    ]
    o2 = [
        TranscriptSegment(
            start=0.0, end=0.5, text="one", speaker="Speaker 1",
            words=[WordTimestamp(start=0.0, end=0.5, word="one")], confidence=0.9,
        ),
        TranscriptSegment(
            start=0.5, end=1.0, text="two", speaker="Speaker 1",
            words=[WordTimestamp(start=0.5, end=1.0, word="two")], confidence=0.9,
        ),
    ]
    reconcile_passes(p2, o2, offset_seconds=15.0)
    second = get_last_reconciliation_stats()
    assert second.total_words >= first_total
    # In the simple case here, second has 2 words, first had 1.
    assert second.total_words == 2
