"""Unit tests for TACT Phase 1 CoverageLedger.

These tests cover the pure-Python data-structure layer:

  * empty audio
  * single-segment claim round-trip via ``from_segments``
  * overlapping claims with confidence-based winner
  * ``query`` with status filters including the implicit ``uncovered``
  * coverage_ratio + voiced_coverage_ratio
  * bin-width invariance (10/20/40 ms produce equivalent ratios)
  * ``to_report_dict`` schema is stable

The tests intentionally do not exercise the pipeline integration; that
path needs real audio + Whisper and is covered by the existing
integration harness.
"""
from __future__ import annotations

from backend.models import TranscriptSegment
from backend.services.transcription_ledger import (
    ALL_STATUSES,
    COVERED_STATUSES,
    CoverageLedger,
    LedgerSpan,
)


# ──────────────────────────────────────────────────────────────────────────
# Construction / empty cases
# ──────────────────────────────────────────────────────────────────────────

def test_empty_ledger_is_fully_uncovered():
    led = CoverageLedger(audio_duration_ms=10_000)
    assert led.coverage_ratio() == 0.0
    dist = led.status_distribution()
    assert dist["uncovered"] == 10_000
    # Every covered status is zero.
    for k in COVERED_STATUSES:
        assert dist[k] == 0
    # Schema completeness: every canonical status key is present.
    for k in ALL_STATUSES:
        assert k in dist


def test_zero_duration_audio_is_zero_coverage():
    led = CoverageLedger(audio_duration_ms=0)
    assert led.coverage_ratio() == 0.0
    assert led.status_distribution()["uncovered"] == 0


def test_negative_duration_rejected():
    import pytest
    with pytest.raises(ValueError):
        CoverageLedger(audio_duration_ms=-1)


def test_non_positive_bin_rejected():
    import pytest
    with pytest.raises(ValueError):
        CoverageLedger(audio_duration_ms=1000, bin_ms=0)


# ──────────────────────────────────────────────────────────────────────────
# claim() basics
# ──────────────────────────────────────────────────────────────────────────

def test_single_claim_covers_expected_duration():
    led = CoverageLedger(audio_duration_ms=10_000)
    led.claim(LedgerSpan(
        start_ms=1_000, end_ms=3_000,
        status="covered_speech", source_pass="t", confidence=0.9,
    ))
    assert led.coverage_ratio() == 0.2  # 2000 / 10000
    dist = led.status_distribution()
    assert dist["covered_speech"] == 2_000
    assert dist["uncovered"] == 8_000


def test_claim_clamps_to_audio_bounds():
    led = CoverageLedger(audio_duration_ms=5_000)
    led.claim(LedgerSpan(
        start_ms=-500, end_ms=10_000,
        status="covered_speech", source_pass="t", confidence=1.0,
    ))
    assert led.coverage_ratio() == 1.0
    assert led.status_distribution()["uncovered"] == 0


def test_zero_or_negative_span_rejected():
    led = CoverageLedger(audio_duration_ms=5_000)
    assert not led.claim(LedgerSpan(
        start_ms=1_000, end_ms=1_000,
        status="covered_speech", source_pass="t", confidence=0.5,
    ))
    assert not led.claim(LedgerSpan(
        start_ms=2_000, end_ms=1_000,
        status="covered_speech", source_pass="t", confidence=0.5,
    ))
    assert led.coverage_ratio() == 0.0


def test_adjacent_same_status_claims_coalesce():
    led = CoverageLedger(audio_duration_ms=10_000)
    led.claim(LedgerSpan(
        start_ms=0, end_ms=1_000,
        status="covered_speech", source_pass="p1",
        content_type="phrase", confidence=0.7,
    ))
    led.claim(LedgerSpan(
        start_ms=1_000, end_ms=2_000,
        status="covered_speech", source_pass="p1",
        content_type="phrase", confidence=0.7,
    ))
    # Same status/source/content_type/confidence → should coalesce.
    assert len(led) == 1


def test_adjacent_different_source_does_not_coalesce():
    led = CoverageLedger(audio_duration_ms=10_000)
    led.claim(LedgerSpan(
        start_ms=0, end_ms=1_000, status="covered_speech",
        source_pass="a", confidence=0.5,
    ))
    led.claim(LedgerSpan(
        start_ms=1_000, end_ms=2_000, status="covered_speech",
        source_pass="b", confidence=0.5,
    ))
    assert len(led) == 2


# ──────────────────────────────────────────────────────────────────────────
# Overlap resolution by confidence
# ──────────────────────────────────────────────────────────────────────────

def test_higher_confidence_overrides_lower():
    led = CoverageLedger(audio_duration_ms=10_000)
    # Existing low-confidence claim covering [1000, 4000].
    led.claim(LedgerSpan(
        start_ms=1_000, end_ms=4_000, status="covered_speech",
        source_pass="weak", confidence=0.2,
    ))
    # Higher-confidence claim covering the middle.
    led.claim(LedgerSpan(
        start_ms=2_000, end_ms=3_000, status="covered_speech",
        source_pass="strong", confidence=0.9,
    ))
    breakdown = led.source_pass_breakdown()
    # Strong wins the middle 1000ms; weak retains the two flanks.
    assert breakdown["strong"] == 1_000
    assert breakdown["weak"] == 2_000  # 1000-2000 + 3000-4000
    assert led.coverage_ratio() == 0.3


def test_lower_confidence_yields_to_existing():
    led = CoverageLedger(audio_duration_ms=10_000)
    led.claim(LedgerSpan(
        start_ms=1_000, end_ms=4_000, status="covered_speech",
        source_pass="strong", confidence=0.9,
    ))
    led.claim(LedgerSpan(
        start_ms=2_000, end_ms=3_000, status="covered_speech",
        source_pass="weak", confidence=0.2,
    ))
    breakdown = led.source_pass_breakdown()
    assert breakdown.get("weak", 0) == 0
    assert breakdown["strong"] == 3_000


def test_partial_overlap_partial_yield():
    """Lower-conf new span partially overlaps incumbent — non-overlapping
    portion is still written."""
    led = CoverageLedger(audio_duration_ms=10_000)
    led.claim(LedgerSpan(
        start_ms=2_000, end_ms=4_000, status="covered_speech",
        source_pass="strong", confidence=0.9,
    ))
    led.claim(LedgerSpan(
        start_ms=3_000, end_ms=6_000, status="covered_speech",
        source_pass="weak", confidence=0.4,
    ))
    breakdown = led.source_pass_breakdown()
    assert breakdown["strong"] == 2_000  # untouched
    assert breakdown["weak"] == 2_000    # 4000-6000


def test_contested_attempts_counted():
    led = CoverageLedger(audio_duration_ms=10_000)
    led.claim(LedgerSpan(
        start_ms=0, end_ms=5_000, status="covered_speech",
        source_pass="strong", confidence=0.9,
    ))
    # Two losing attempts — each increments the counter once.
    led.claim(LedgerSpan(
        start_ms=1_000, end_ms=2_000, status="covered_speech",
        source_pass="weak", confidence=0.1,
    ))
    led.claim(LedgerSpan(
        start_ms=3_000, end_ms=4_000, status="covered_speech",
        source_pass="weak", confidence=0.1,
    ))
    report = led.to_report_dict()
    assert report["contested_attempts"] == 2


# ──────────────────────────────────────────────────────────────────────────
# query()
# ──────────────────────────────────────────────────────────────────────────

def test_query_uncovered_complement():
    led = CoverageLedger(audio_duration_ms=10_000)
    led.claim(LedgerSpan(
        start_ms=2_000, end_ms=4_000, status="covered_speech",
        source_pass="t", confidence=1.0,
    ))
    led.claim(LedgerSpan(
        start_ms=6_000, end_ms=7_000, status="covered_speech",
        source_pass="t", confidence=1.0,
    ))
    intervals = led.query({"uncovered"})
    assert intervals == [(0, 2_000), (4_000, 6_000), (7_000, 10_000)]


def test_query_uncovered_with_min_duration_filters_short_gaps():
    led = CoverageLedger(audio_duration_ms=10_000)
    led.claim(LedgerSpan(
        start_ms=0, end_ms=4_900, status="covered_speech",
        source_pass="t", confidence=1.0,
    ))
    led.claim(LedgerSpan(
        start_ms=5_000, end_ms=10_000, status="covered_speech",
        source_pass="t", confidence=1.0,
    ))
    # The 100 ms gap is too short for a 200 ms minimum.
    assert led.query({"uncovered"}, min_duration_ms=200) == []
    # But it's visible at min_duration_ms=50.
    assert led.query({"uncovered"}, min_duration_ms=50) == [(4_900, 5_000)]


def test_query_status_filter():
    led = CoverageLedger(audio_duration_ms=10_000)
    led.claim(LedgerSpan(
        start_ms=0, end_ms=2_000, status="covered_speech",
        source_pass="t", confidence=1.0,
    ))
    led.claim(LedgerSpan(
        start_ms=3_000, end_ms=4_000, status="covered_event",
        source_pass="t", confidence=1.0,
    ))
    speech_only = led.query({"covered_speech"})
    assert speech_only == [(0, 2_000)]
    events_only = led.query({"covered_event"})
    assert events_only == [(3_000, 4_000)]
    both = led.query({"covered_speech", "covered_event"})
    assert sorted(both) == [(0, 2_000), (3_000, 4_000)]


# ──────────────────────────────────────────────────────────────────────────
# from_segments round-trip
# ──────────────────────────────────────────────────────────────────────────

def test_from_segments_roundtrip_dict_input():
    led = CoverageLedger(audio_duration_ms=10_000)
    segs = [
        {"start": 0.5, "end": 1.2, "text": "hello", "speaker": "Speaker 1",
         "avg_logprob": -0.2},
        {"start": 2.0, "end": 3.5, "text": "world", "speaker": "Speaker 1",
         "confidence": 0.85},
    ]
    led.from_segments(segs, source_pass="whisper_main")
    # 0.7 + 1.5 = 2.2 seconds covered out of 10.
    assert led.coverage_ratio() == 0.22
    assert led.source_pass_breakdown() == {"whisper_main": 2_200}


def test_from_segments_roundtrip_dataclass_input():
    led = CoverageLedger(audio_duration_ms=10_000)
    segs = [
        TranscriptSegment(
            start=0.5, end=1.2, text="hello", speaker="Speaker 1",
            avg_logprob=-0.2,
        ),
        TranscriptSegment(
            start=2.0, end=3.5, text="world", speaker="Speaker 1",
            confidence=0.85,
        ),
    ]
    led.from_segments(segs, source_pass="whisper_main+gap_fill")
    assert led.coverage_ratio() == 0.22
    assert led.source_pass_breakdown() == {"whisper_main+gap_fill": 2_200}


def test_from_segments_skips_malformed():
    led = CoverageLedger(audio_duration_ms=10_000)
    led.from_segments([
        None,
        {"start": "abc", "end": "def", "text": "garbage"},
        {"start": 1.0, "end": 1.0, "text": "zero"},  # zero-duration
        {"start": 2.0, "end": 1.0, "text": "inverted"},  # negative
        {"start": 0.0, "end": 1.0, "text": "ok"},  # only this one survives
    ])
    assert led.coverage_ratio() == 0.1


# ──────────────────────────────────────────────────────────────────────────
# coverage / voiced_coverage
# ──────────────────────────────────────────────────────────────────────────

def test_voiced_coverage_matches_intersection():
    led = CoverageLedger(audio_duration_ms=10_000)
    # Speech covers [1, 4] and [6, 8] seconds.
    led.from_segments([
        {"start": 1.0, "end": 4.0, "text": "a"},
        {"start": 6.0, "end": 8.0, "text": "b"},
    ])
    # VAD says voiced [0, 5] and [7, 9] = 7 seconds total.
    # Speech ∩ VAD = [1, 4] (3s) + [7, 8] (1s) = 4s out of 7.
    voiced = led.voiced_coverage_ratio([(0.0, 5.0), (7.0, 9.0)])
    assert abs(voiced - 4.0 / 7.0) < 1e-6


def test_voiced_coverage_empty_vad_is_zero():
    led = CoverageLedger(audio_duration_ms=10_000)
    led.from_segments([{"start": 1.0, "end": 4.0, "text": "a"}])
    assert led.voiced_coverage_ratio([]) == 0.0


# ──────────────────────────────────────────────────────────────────────────
# Bin-width invariance
# ──────────────────────────────────────────────────────────────────────────

def test_bin_width_invariance():
    """Coverage ratio is bin-independent (within float tolerance)."""
    segs = [
        {"start": 0.5, "end": 1.2, "text": "a"},
        {"start": 2.0, "end": 3.5, "text": "b"},
        {"start": 7.123, "end": 8.456, "text": "c"},
    ]
    ratios = []
    for bin_ms in (10, 20, 40):
        led = CoverageLedger(audio_duration_ms=10_000, bin_ms=bin_ms)
        led.from_segments(segs)
        ratios.append(led.coverage_ratio())
    assert max(ratios) - min(ratios) < 0.01


# ──────────────────────────────────────────────────────────────────────────
# to_report_dict schema
# ──────────────────────────────────────────────────────────────────────────

def test_report_dict_schema_is_stable():
    led = CoverageLedger(audio_duration_ms=10_000)
    led.from_segments([{"start": 0.0, "end": 5.0, "text": "x"}])
    report = led.to_report_dict()
    # Top-level keys, in stable order, every release.
    expected_keys = {
        "version",
        "audio_duration_ms",
        "bin_ms",
        "span_count",
        "coverage_ratio",
        "voiced_coverage_ratio",
        "uncovered_ratio",
        "contested_ratio",
        "quarantined_ratio",
        "status_distribution_ms",
        "confidence_histogram",
        "source_pass_ms",
        "contested_attempts",
    }
    assert set(report.keys()) == expected_keys
    # status_distribution is always keyed by every canonical status,
    # even when the status is unused this run.
    for k in ALL_STATUSES:
        assert k in report["status_distribution_ms"]
    # confidence_histogram always has 10 buckets.
    assert len(report["confidence_histogram"]) == 10
    assert report["version"] == 1
    assert report["audio_duration_ms"] == 10_000


def test_report_dict_uncovered_ratio_matches_complement():
    led = CoverageLedger(audio_duration_ms=10_000)
    led.from_segments([{"start": 0.0, "end": 3.0, "text": "x"}])
    r = led.to_report_dict()
    # 7000 ms uncovered out of 10000 → 0.70.
    assert abs(r["uncovered_ratio"] - 0.70) < 1e-6
    assert abs(r["coverage_ratio"] - 0.30) < 1e-6


# ──────────────────────────────────────────────────────────────────────────
# to_segments round-trip
# ──────────────────────────────────────────────────────────────────────────

def test_to_segments_round_trip_only_emits_speech():
    led = CoverageLedger(audio_duration_ms=10_000)
    led.from_segments([
        {"start": 0.5, "end": 1.2, "text": "hello", "speaker": "Speaker 1"},
    ])
    led.claim(LedgerSpan(
        start_ms=2_000, end_ms=4_000,
        status="covered_event", content="[music]",
        content_type="event", source_pass="event_classifier",
        confidence=0.8,
    ))
    out = led.to_segments()
    assert len(out) == 1
    assert out[0].text == "hello"
    assert out[0].speaker == "Speaker 1"


# ──────────────────────────────────────────────────────────────────────────
# Confidence histogram
# ──────────────────────────────────────────────────────────────────────────

def test_confidence_histogram_buckets_correctly():
    led = CoverageLedger(audio_duration_ms=10_000)
    # 100ms at confidence 0.05 → bucket 0
    # 100ms at confidence 0.95 → bucket 9
    # 100ms at confidence 0.5  → bucket 5
    led.claim(LedgerSpan(
        start_ms=0, end_ms=100, status="covered_speech",
        source_pass="t", confidence=0.05,
    ))
    led.claim(LedgerSpan(
        start_ms=200, end_ms=300, status="covered_speech",
        source_pass="t", confidence=0.95,
    ))
    led.claim(LedgerSpan(
        start_ms=400, end_ms=500, status="covered_speech",
        source_pass="t", confidence=0.5,
    ))
    hist = led.confidence_histogram(10)
    assert hist[0] == 100
    assert hist[5] == 100
    assert hist[9] == 100
    assert sum(hist) == 300
