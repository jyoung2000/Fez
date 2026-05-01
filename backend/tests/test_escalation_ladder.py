"""Tests for the TACT Phase 2 escalation ladder skeleton.

Covers:
  * Rung ordering — first non-empty result wins for an interval.
  * Rung 6 always-claims invariant: after the ladder runs with the
    default rungs (or any rung list ending in Rung 6), no uncovered
    bins remain in the ledger.
  * Per-rung kill switch via custom rung lists (used by Phase 2
    follow-up commits to gate Rungs 2/4/5 individually).
  * Budget exhaustion routes to Rungs 5/6 only.
  * Coverage invariant: ``coverage_ratio == 1.0`` after the ladder
    runs against an empty ledger with default rungs.

The Whisper-calling Rung 1 is exercised indirectly — the ladder is
tested with mock rungs so no real Whisper / ffmpeg is required.
"""
from __future__ import annotations

from backend.services.escalation_ladder import (
    DEFAULT_RUNGS,
    EscalationContext,
    EscalationStats,
    LedgerSpan,
    RungResult,
    _rung_mark_unintelligible,
    escalate_uncovered_intervals_sync,
)
from backend.services.transcription_ledger import CoverageLedger


def _make_ctx(audio_duration_ms: int = 10_000) -> EscalationContext:
    return EscalationContext(
        audio_path="/dev/null",
        audio_duration_ms=audio_duration_ms,
        language="en",
        task="transcribe",
    )


# ──────────────────────────────────────────────────────────────────────────
# Rung 6 invariant — always claims
# ──────────────────────────────────────────────────────────────────────────

def test_rung_6_terminator_always_claims():
    ctx = _make_ctx()
    result = _rung_mark_unintelligible("/dev/null", 1_000, 4_000, ctx)
    assert len(result.spans) == 1
    span = result.spans[0]
    assert span.start_ms == 1_000
    assert span.end_ms == 4_000
    assert span.content == "[unintelligible]"
    assert span.content_type == "unintelligible"
    assert span.status == "covered_event"
    assert span.confidence == 0.0


def test_rung_6_rejects_zero_or_negative_intervals():
    ctx = _make_ctx()
    assert _rung_mark_unintelligible("/dev/null", 5_000, 5_000, ctx).spans == []
    assert _rung_mark_unintelligible("/dev/null", 5_000, 1_000, ctx).spans == []


# ──────────────────────────────────────────────────────────────────────────
# Default-rung-list end-to-end: empty ledger → 100% coverage
# ──────────────────────────────────────────────────────────────────────────

def test_default_ladder_terminates_with_full_coverage():
    """Empty ledger → ladder fills every uncovered second with Rung 6
    (since the Whisper rungs aren't reachable in unit tests). After
    the ladder, coverage_ratio is exactly 1.0."""
    ledger = CoverageLedger(audio_duration_ms=10_000)
    ctx = _make_ctx()
    # Rung 1 isn't callable in this environment (no Whisper subprocess
    # / no audio file). The ladder catches the rung's exception, logs
    # it, and moves on. Eventually Rung 6 catches the interval.
    rungs_without_whisper = [
        # Skip Rung 1 (would need real audio + Whisper).
        ("rung_2", lambda *a, **kw: RungResult("rung_2")),
        ("rung_3", lambda *a, **kw: RungResult("rung_3")),
        ("rung_4", lambda *a, **kw: RungResult("rung_4")),
        ("rung_5", lambda *a, **kw: RungResult("rung_5")),
        ("rung_6_unintelligible", _rung_mark_unintelligible),
    ]
    stats = escalate_uncovered_intervals_sync(
        ledger, ctx, rungs=rungs_without_whisper, max_audio_sec=600.0,
    )
    # The whole 10 s was uncovered → one interval, resolved by Rung 6.
    assert stats.intervals_total == 1
    assert stats.intervals_resolved == 1
    assert stats.rung_resolutions == {"rung_6_unintelligible": 1}
    assert ledger.coverage_ratio() == 1.0
    assert ledger.query({"uncovered"}) == []


# ──────────────────────────────────────────────────────────────────────────
# Rung ordering
# ──────────────────────────────────────────────────────────────────────────

def test_first_non_empty_rung_wins():
    ledger = CoverageLedger(audio_duration_ms=10_000)
    ctx = _make_ctx()

    rung_3_called = {"called": False}

    def winning_rung(audio_path, start_ms, end_ms, ctx):
        return RungResult(
            rung_name="winning",
            spans=[LedgerSpan(
                start_ms=start_ms, end_ms=end_ms,
                status="covered_speech",
                content="winner",
                source_pass="winning",
                confidence=0.9,
            )],
            confidence=0.9,
        )

    def should_not_run(audio_path, start_ms, end_ms, ctx):
        rung_3_called["called"] = True
        return RungResult(rung_name="rung_3")

    rungs = [
        ("losing", lambda *a, **kw: RungResult("losing")),  # empty
        ("winning", winning_rung),
        ("rung_3", should_not_run),
        ("rung_6_unintelligible", _rung_mark_unintelligible),
    ]
    stats = escalate_uncovered_intervals_sync(ledger, ctx, rungs=rungs)
    assert stats.rung_resolutions == {"winning": 1}
    assert rung_3_called["called"] is False
    assert ledger.coverage_ratio() == 1.0


def test_rungs_called_in_registered_order():
    """The ladder must hit rungs left-to-right; reordering the list
    changes which rung wins."""
    call_order: list[str] = []

    def make_logger(name, claims=False):
        def fn(audio_path, start_ms, end_ms, ctx):
            call_order.append(name)
            if not claims:
                return RungResult(rung_name=name)
            return RungResult(
                rung_name=name,
                spans=[LedgerSpan(
                    start_ms=start_ms, end_ms=end_ms,
                    status="covered_event", content="[stub]",
                    content_type="event",
                    source_pass=name, confidence=0.5,
                )],
                confidence=0.5,
            )
        return fn

    ledger = CoverageLedger(audio_duration_ms=5_000)
    ctx = _make_ctx(5_000)
    rungs = [
        ("a", make_logger("a", claims=False)),
        ("b", make_logger("b", claims=False)),
        ("c", make_logger("c", claims=True)),
        ("d", make_logger("d", claims=False)),
    ]
    escalate_uncovered_intervals_sync(ledger, ctx, rungs=rungs)
    assert call_order == ["a", "b", "c"]


# ──────────────────────────────────────────────────────────────────────────
# Budget
# ──────────────────────────────────────────────────────────────────────────

def test_budget_exhaustion_routes_to_rung_5_or_6():
    """Once the ladder marks budget_exhausted, only Rungs 5 and 6 may
    run. Rungs 1-4 are skipped for remaining intervals."""
    ledger = CoverageLedger(audio_duration_ms=20_000)
    # Two uncovered intervals: [0, 8000] and [9000, 20000].
    ledger.claim(LedgerSpan(
        start_ms=8_000, end_ms=9_000,
        status="covered_speech", source_pass="primer",
        confidence=1.0,
    ))
    ctx = _make_ctx(20_000)

    expensive_calls = {"count": 0}

    def expensive_rung_1(audio_path, start_ms, end_ms, ctx):
        expensive_calls["count"] += 1
        # Burn the whole budget on the first interval.
        return RungResult(
            rung_name="rung_1_relaxed_whisper",
            elapsed_ms=0,
            notes="burned_budget",
        )

    rungs = [
        ("rung_1_relaxed_whisper", expensive_rung_1),
        ("rung_5_event_classifier", lambda *a, **kw: RungResult(
            rung_name="rung_5_event_classifier")),
        ("rung_6_unintelligible", _rung_mark_unintelligible),
    ]
    # Tiny budget. The ladder consumes it on the first interval
    # immediately; the second interval should only see Rungs 5 + 6.
    stats = escalate_uncovered_intervals_sync(
        ledger, ctx, rungs=rungs, max_audio_sec=0.0,  # zero budget
    )
    assert stats.budget_exhausted
    # Even with zero budget, Rung 6 must still claim every interval.
    assert ledger.coverage_ratio() == 1.0
    # Rung 1 was not called for the second interval (budget gate).
    # It was either not called at all, or only for the first.
    assert expensive_calls["count"] <= 1


# ──────────────────────────────────────────────────────────────────────────
# Quarantined-status integration
# ──────────────────────────────────────────────────────────────────────────

def test_ladder_iterates_over_quarantined_intervals():
    """A region marked ``quarantined`` should also be presented to the
    ladder so a hallucinated transcription can be replaced by a
    correct event tag."""
    ledger = CoverageLedger(audio_duration_ms=10_000)
    # Cover [0, 3000] as real speech.
    ledger.claim(LedgerSpan(
        start_ms=0, end_ms=3_000, status="covered_speech",
        source_pass="whisper_main", confidence=0.9,
    ))
    # Quarantine [4000, 6000] — Whisper hallucinated here.
    ledger.claim(LedgerSpan(
        start_ms=4_000, end_ms=6_000, status="quarantined",
        source_pass="whisper_main", confidence=0.4,
        flags=["non_speech"],
    ))
    # Leave [3000-4000] and [6000-10000] uncovered.

    ctx = _make_ctx()
    visited: list[tuple[int, int]] = []

    def visitor(audio_path, start_ms, end_ms, ctx):
        visited.append((start_ms, end_ms))
        return RungResult(rung_name="visitor")

    rungs = [
        ("visitor", visitor),
        ("rung_6_unintelligible", _rung_mark_unintelligible),
    ]
    escalate_uncovered_intervals_sync(ledger, ctx, rungs=rungs)
    # The visitor is called once per (uncovered ∪ quarantined) interval.
    visited_ranges = sorted(visited)
    assert (3_000, 4_000) in visited_ranges
    assert (4_000, 6_000) in visited_ranges
    assert (6_000, 10_000) in visited_ranges


# ──────────────────────────────────────────────────────────────────────────
# Stats shape
# ──────────────────────────────────────────────────────────────────────────

def test_stats_as_dict_has_expected_keys():
    stats = EscalationStats(intervals_total=2, intervals_resolved=1)
    d = stats.as_dict()
    expected = {
        "intervals_total", "intervals_resolved",
        "rung_resolutions", "rung_elapsed_ms",
        "spans_added", "audio_sec_processed",
        "elapsed_sec", "budget_exhausted", "skipped_reason",
    }
    assert set(d.keys()) == expected


def test_default_rungs_includes_every_phase_2_slot():
    names = [n for n, _ in DEFAULT_RUNGS]
    assert names == [
        "rung_1_relaxed_whisper",
        "rung_2_alt_whisper_checkpoint",
        "rung_3_consensus_model",
        "rung_4_forced_alignment",
        "rung_5_event_classifier",
        "rung_6_unintelligible",
    ]
