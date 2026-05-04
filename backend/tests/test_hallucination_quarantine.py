"""Tests for the TACT Phase 2 _filter_hallucinations tuple-return refactor.

The function previously returned ``list[dict]`` (kept). It now returns
``tuple[list[dict], list[dict]]`` (kept, quarantined). Each entry in
the quarantined list carries a ``quarantine_reason`` from a closed set
of reasons. The kept-list logic is byte-identical to the previous
behavior — these tests confirm both halves.
"""
from __future__ import annotations

from backend.services.transcription import (
    QUARANTINE_REASONS,
    _filter_hallucinations,
    get_last_quarantined_segments,
)


# ──────────────────────────────────────────────────────────────────────────
# Shape and invariants
# ──────────────────────────────────────────────────────────────────────────

def test_returns_tuple_of_two_lists():
    kept, q = _filter_hallucinations([])
    assert isinstance(kept, list)
    assert isinstance(q, list)


def test_empty_input_returns_two_empty_lists():
    kept, q = _filter_hallucinations([])
    assert kept == []
    assert q == []
    assert get_last_quarantined_segments() == []


def test_clean_segments_pass_through():
    segs = [
        {"start": 0.0, "end": 1.0, "text": "hello world",
         "no_speech_prob": 0.05, "confidence": 0.9},
        {"start": 1.0, "end": 2.0, "text": "this is fine",
         "no_speech_prob": 0.05, "confidence": 0.9},
    ]
    kept, q = _filter_hallucinations(segs)
    assert len(kept) == 2
    assert q == []


def test_kept_plus_quarantined_partitions_input():
    """No segment that would have been kept ends up quarantined and vice
    versa. Empty-text segments are dropped (not quarantined) per spec.

    The filter requires boilerplate to be at the transcript edge (first
    or last/second-to-last in raw_segments) AND low-confidence to fire,
    so the boilerplate entry is placed at index 0 here.
    """
    segs = [
        # Boilerplate at the leading edge, low confidence → quarantined.
        {"start": 0.0, "end": 1.0, "text": "thank you for watching",
         "no_speech_prob": 0.5, "confidence": 0.3},
        # Real speech, kept.
        {"start": 1.0, "end": 2.0, "text": "hello",
         "no_speech_prob": 0.05, "confidence": 0.9},
        # Empty text → dropped without quarantine.
        {"start": 2.0, "end": 2.5, "text": ""},
        # Non-speech segment → quarantined.
        {"start": 3.0, "end": 30.0, "text": ".",
         "no_speech_prob": 0.95, "confidence": 0.05},
    ]
    kept, q = _filter_hallucinations(segs)
    # 1 kept, 2 quarantined, 1 silently dropped (empty text).
    assert len(kept) == 1
    assert len(q) == 2
    # No segment appears in both lists — partition invariant.
    kept_keys = {(s["start"], s["end"], s["text"]) for s in kept}
    q_keys = {(s["start"], s["end"], s["text"]) for s in q}
    assert kept_keys.isdisjoint(q_keys)


# ──────────────────────────────────────────────────────────────────────────
# Reason tagging
# ──────────────────────────────────────────────────────────────────────────

def test_every_quarantined_has_a_known_reason():
    segs = [
        {"start": 3.0, "end": 30.0, "text": ".",
         "no_speech_prob": 0.95, "confidence": 0.05},  # non_speech
        {"start": 0.0, "end": 1.0, "text": "thank you for watching",
         "no_speech_prob": 0.5, "confidence": 0.3},  # boilerplate
        # Note: ghost_by_ratio fires only when seg_duration > 15s AND
        # chars_per_sec < 1.0; this is 60 s with 14 chars = ~0.23 c/s.
        {"start": 100.0, "end": 160.0, "text": "ok ok ok ok ok",
         "no_speech_prob": 0.05, "confidence": 0.9},  # ghost_by_ratio
    ]
    _, q = _filter_hallucinations(segs)
    for entry in q:
        assert "quarantine_reason" in entry
        assert entry["quarantine_reason"] in QUARANTINE_REASONS


def test_non_speech_reason_assigned():
    segs = [{
        "start": 0.0, "end": 1.0, "text": ".",
        "no_speech_prob": 0.95, "confidence": 0.05,
    }]
    _, q = _filter_hallucinations(segs)
    assert len(q) == 1
    assert q[0]["quarantine_reason"] == "non_speech"


def test_boilerplate_reason_assigned():
    # Boilerplate fires only at edge with low confidence.
    segs = [{
        "start": 0.0, "end": 1.0, "text": "thank you for watching",
        "no_speech_prob": 0.5, "confidence": 0.3,
    }]
    _, q = _filter_hallucinations(segs)
    assert len(q) == 1
    assert q[0]["quarantine_reason"] == "boilerplate"


def test_runaway_reason_assigned():
    long_text = "word " * 400  # 2000 chars
    segs = [{
        "start": 0.0, "end": 5.0, "text": long_text,
        "no_speech_prob": 0.05, "confidence": 0.9,
    }]
    _, q = _filter_hallucinations(segs)
    assert len(q) == 1
    assert q[0]["quarantine_reason"] == "runaway"


# ──────────────────────────────────────────────────────────────────────────
# Module-level accessor
# ──────────────────────────────────────────────────────────────────────────

def test_module_accessor_reflects_last_call():
    _filter_hallucinations([{
        "start": 0.0, "end": 1.0, "text": ".",
        "no_speech_prob": 0.95, "confidence": 0.05,
    }])
    last = get_last_quarantined_segments()
    assert len(last) == 1
    assert last[0]["quarantine_reason"] == "non_speech"
    # Subsequent call resets.
    _filter_hallucinations([])
    assert get_last_quarantined_segments() == []


def test_accessor_returns_independent_copies():
    """Mutating the returned list must not affect the module state."""
    _filter_hallucinations([{
        "start": 0.0, "end": 1.0, "text": ".",
        "no_speech_prob": 0.95, "confidence": 0.05,
    }])
    first = get_last_quarantined_segments()
    first.clear()
    second = get_last_quarantined_segments()
    assert len(second) == 1


# ──────────────────────────────────────────────────────────────────────────
# Backwards-compatibility: kept-list logic unchanged
# ──────────────────────────────────────────────────────────────────────────

def test_kept_list_unchanged_for_clean_input():
    """Previously the function returned `list[dict]`. Confirm that on
    realistic non-hallucination input the kept list is still the
    everything-kept set, in the same order."""
    segs = [
        {"start": 0.0, "end": 1.5, "text": "the quick brown fox",
         "no_speech_prob": 0.05, "confidence": 0.9},
        {"start": 2.0, "end": 3.5, "text": "jumps over the lazy dog",
         "no_speech_prob": 0.05, "confidence": 0.9},
        {"start": 4.0, "end": 5.5, "text": "and runs away into the night",
         "no_speech_prob": 0.05, "confidence": 0.9},
    ]
    kept, q = _filter_hallucinations(segs)
    assert kept == segs
    assert q == []
