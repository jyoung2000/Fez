"""Tests for the TACT Phase 3 word-level reconciler.

Covers:
  * Two-way agreement and disagreement.
  * Single-source words flagged correctly.
  * ROVER tiebreak by distance-to-boundary on disagreement.
  * Re-segmentation honors the primary pass's segment boundaries.
  * N-way generalization (Phase 4 extension): majority vote, tied
    no-majority cluster, multi-source agreement counting.
  * Ledger bridge produces word-level claims with provenance flags.
"""
from __future__ import annotations

from backend.models import TranscriptSegment, WordTimestamp
from backend.services.transcription_ledger import CoverageLedger
from backend.services.transcription_reconciler import (
    ReconciledWord,
    claim_reconciled_words_into_ledger,
    reconcile_n_passes,
    reconcile_passes,
)


def _seg(start, end, text, words=None, conf=0.9):
    if words is not None:
        words = [WordTimestamp(start=s, end=e, word=w) for s, e, w in words]
    return TranscriptSegment(
        start=start, end=end, text=text, speaker="Speaker 1",
        words=words, confidence=conf,
    )


# ──────────────────────────────────────────────────────────────────────────
# Two-way reconciliation
# ──────────────────────────────────────────────────────────────────────────

def test_both_passes_agree_keeps_one_word():
    primary = [_seg(1.0, 1.5, "hello",
                    words=[(1.0, 1.5, "hello")])]
    offset = [_seg(1.0, 1.5, "hello",
                   words=[(1.0, 1.5, "hello")])]
    out, stats = reconcile_passes(primary, offset, offset_seconds=15.0)
    assert stats.total_words == 1
    assert stats.agreed == 1
    assert stats.disagreed == 0
    # Output has one word, sourced from both passes.
    assert len(out) == 1
    assert out[0].text == "hello"
    assert out[0].words is not None and len(out[0].words) == 1


def test_disagreement_picks_far_from_boundary():
    """Word at primary-time 29.8 s is 0.2 s from the 30 s boundary in
    the primary pass. With offset=15, the same time in offset-pass
    coordinates is 14.8 s from the nearest boundary at 15 s. Offset
    wins."""
    primary = [_seg(29.5, 30.1, "near",
                    words=[(29.7, 29.9, "near")])]
    offset = [_seg(29.5, 30.1, "far",
                   words=[(29.7, 29.9, "far")])]
    out, stats = reconcile_passes(primary, offset, offset_seconds=15.0)
    assert stats.disagreed == 1
    # The offset pass wins because the word is mid-chunk in its grid.
    assert out[0].text == "far"


def test_only_primary_emits_word_kept_single_source():
    primary = [_seg(0.0, 0.5, "alpha",
                    words=[(0.0, 0.5, "alpha")])]
    offset = []
    out, stats = reconcile_passes(primary, offset, offset_seconds=15.0)
    assert stats.single_source == 1
    assert out[0].text == "alpha"


def test_only_offset_emits_word_kept_single_source():
    primary = []
    offset = [_seg(0.0, 0.5, "beta",
                   words=[(0.0, 0.5, "beta")])]
    out, stats = reconcile_passes(primary, offset, offset_seconds=15.0)
    assert stats.single_source == 1
    assert out[0].text == "beta"


def test_text_normalization_treats_punctuation_as_match():
    """'hello,' and 'hello' should agree — punctuation only differs."""
    primary = [_seg(1.0, 1.5, "hello,",
                    words=[(1.0, 1.5, "hello,")])]
    offset = [_seg(1.0, 1.5, "hello",
                   words=[(1.0, 1.5, "hello")])]
    out, stats = reconcile_passes(primary, offset, offset_seconds=15.0)
    assert stats.agreed == 1
    assert stats.disagreed == 0


def test_resegmentation_uses_primary_segment_boundaries():
    """Three reconciled words split into two primary segments → output
    has two segments, words assigned by midpoint."""
    primary = [
        _seg(0.0, 1.0, "hello world",
             words=[(0.1, 0.4, "hello"), (0.5, 0.9, "world")]),
        _seg(2.0, 2.5, "again",
             words=[(2.0, 2.5, "again")]),
    ]
    offset = [
        _seg(0.0, 1.0, "hello world",
             words=[(0.1, 0.4, "hello"), (0.5, 0.9, "world")]),
        _seg(2.0, 2.5, "again",
             words=[(2.0, 2.5, "again")]),
    ]
    out, _ = reconcile_passes(primary, offset, offset_seconds=15.0)
    assert len(out) == 2
    assert out[0].text == "hello world"
    assert out[1].text == "again"


def test_works_when_words_missing_via_uniform_distribution():
    """Reconciler still operates when one pass lacks word_timestamps
    (segment-only). Approximation; not exact, but exercises the
    fallback."""
    primary = [_seg(0.0, 1.0, "hello world", words=None)]
    offset = [_seg(0.0, 1.0, "hello world", words=None)]
    out, stats = reconcile_passes(primary, offset, offset_seconds=15.0)
    # Two tokens per pass clustered together → 2 agreed.
    assert stats.total_words == 2
    assert stats.agreed == 2


# ──────────────────────────────────────────────────────────────────────────
# N-way reconciliation
# ──────────────────────────────────────────────────────────────────────────

def test_three_way_majority_wins():
    """Two passes say 'hello', one says 'help' — majority wins, marked
    contested."""
    p1 = [_seg(0.0, 0.5, "hello", words=[(0.0, 0.5, "hello")], conf=0.9)]
    p2 = [_seg(0.0, 0.5, "hello", words=[(0.0, 0.5, "hello")], conf=0.85)]
    p3 = [_seg(0.0, 0.5, "help",  words=[(0.0, 0.5, "help")], conf=0.8)]
    out, stats = reconcile_n_passes(
        passes=[("a", p1), ("b", p2), ("c", p3)],
        chunk_grid_offsets={"a": 0.0, "b": 15.0, "c": 7.5},
    )
    assert stats.disagreed == 1
    assert stats.contested == 1
    assert out[0].text == "hello"


def test_three_way_no_majority_uses_distance_to_boundary():
    """Three passes, three different texts. Pass with the largest
    distance-to-boundary wins. With offsets 0, 15, 7.5 and a word at
    t=14.0:
      a: distance = min(14, 16) = 14
      b: distance = (14 - 15) % 30 = 29; min(29, 1) = 1
      c: distance = (14 - 7.5) % 30 = 6.5; min(6.5, 23.5) = 6.5
    Pass a wins.
    """
    p_a = [_seg(13.5, 14.5, "alpha", words=[(13.5, 14.5, "alpha")], conf=0.7)]
    p_b = [_seg(13.5, 14.5, "beta",  words=[(13.5, 14.5, "beta")], conf=0.7)]
    p_c = [_seg(13.5, 14.5, "gamma", words=[(13.5, 14.5, "gamma")], conf=0.7)]
    out, stats = reconcile_n_passes(
        passes=[("a", p_a), ("b", p_b), ("c", p_c)],
        chunk_grid_offsets={"a": 0.0, "b": 15.0, "c": 7.5},
    )
    assert stats.contested == 1
    assert out[0].text == "alpha"


def test_n_way_two_pass_falls_through_to_two_way_logic():
    """``reconcile_passes`` is a thin wrapper around ``reconcile_n_passes``
    — confirm equivalence."""
    p1 = [_seg(0.0, 0.5, "hello", words=[(0.0, 0.5, "hello")])]
    p2 = [_seg(0.0, 0.5, "hello", words=[(0.0, 0.5, "hello")])]
    a, sa = reconcile_passes(p1, p2, offset_seconds=15.0)
    b, sb = reconcile_n_passes(
        passes=[("primary", p1), ("offset", p2)],
        chunk_grid_offsets={"primary": 0.0, "offset": 15.0},
        primary_for_segmentation="primary",
    )
    assert sa.agreed == sb.agreed
    assert [s.text for s in a] == [s.text for s in b]


# ──────────────────────────────────────────────────────────────────────────
# Ledger bridge
# ──────────────────────────────────────────────────────────────────────────

def test_claim_reconciled_words_into_ledger():
    led = CoverageLedger(audio_duration_ms=10_000)
    words = [
        ReconciledWord(start_sec=1.0, end_sec=1.5, word="hello",
                       confidence=0.9, sources=["primary", "offset"]),
        ReconciledWord(start_sec=2.0, end_sec=2.5, word="world",
                       confidence=0.7, sources=["primary"],
                       single_source=True),
        ReconciledWord(start_sec=3.0, end_sec=3.5, word="contested",
                       confidence=0.5, sources=["primary"],
                       contested=True),
    ]
    n = claim_reconciled_words_into_ledger(led, words)
    assert n == 3
    # 0.5 + 0.5 + 0.5 = 1.5 s covered out of 10 s.
    assert abs(led.coverage_ratio() - 0.15) < 1e-6
    # Source-pass breakdown distinguishes by sources.
    breakdown = led.source_pass_breakdown()
    assert any("primary" in k for k in breakdown.keys())
    # Contested word's flags include "contested".
    contested_spans = [s for s in led.spans if "contested" in s.flags]
    assert len(contested_spans) == 1
