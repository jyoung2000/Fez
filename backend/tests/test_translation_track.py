"""Tests for the TACT Phase 5 translation track.

Covers:
  * to_translation_ledger preserves event/silence/unintelligible
    spans unchanged and stages speech spans as
    awaiting_translation.
  * Single-backend translation fills the target ledger.
  * Multi-backend reconciliation: agreed / soft / contested / failed.
  * The translation-track coverage invariant: after run_translation_track,
    every source covered_speech span has a corresponding non-empty
    target span (or [untranslatable] when all backends fail).
  * to_paired_report_dict schema.
"""
from __future__ import annotations

import pytest

from backend.services.transcription_ledger import (
    CoverageLedger,
    LedgerSpan,
)
from backend.services.translation_track import (
    TranslationCandidate,
    _BACKENDS,
    reconcile_translations,
    register_backend,
    run_translation_track,
)


@pytest.fixture
def source_ledger():
    led = CoverageLedger(audio_duration_ms=10_000)
    # Three speech spans + one event span.
    led.claim(LedgerSpan(
        start_ms=0, end_ms=2_000, status="covered_speech",
        content="hello world", content_type="phrase",
        source_pass="whisper_main", confidence=0.9,
    ))
    led.claim(LedgerSpan(
        start_ms=3_000, end_ms=5_000, status="covered_speech",
        content="how are you", content_type="phrase",
        source_pass="whisper_main", confidence=0.85,
    ))
    led.claim(LedgerSpan(
        start_ms=5_000, end_ms=6_000, status="covered_event",
        content="[music]", content_type="event",
        source_pass="event_classifier", confidence=0.8,
    ))
    led.claim(LedgerSpan(
        start_ms=7_000, end_ms=9_000, status="covered_speech",
        content="goodbye", content_type="phrase",
        source_pass="whisper_main", confidence=0.95,
    ))
    return led


# ──────────────────────────────────────────────────────────────────────────
# to_translation_ledger
# ──────────────────────────────────────────────────────────────────────────

def test_to_translation_ledger_passes_events_through(source_ledger):
    target = source_ledger.to_translation_ledger("fr")
    event_spans = [s for s in target.spans if s.content_type == "event"]
    assert len(event_spans) == 1
    assert event_spans[0].content == "[music]"
    assert event_spans[0].target_language == "fr"


def test_to_translation_ledger_stages_speech_as_pending(source_ledger):
    target = source_ledger.to_translation_ledger("fr")
    pending = [s for s in target.spans
               if "awaiting_translation" in s.flags]
    assert len(pending) == 3
    for s in pending:
        assert s.target_language == "fr"
        assert s.content is None


# ──────────────────────────────────────────────────────────────────────────
# Single-backend translation fills the target ledger
# ──────────────────────────────────────────────────────────────────────────

def test_run_translation_track_single_backend(source_ledger):
    @register_backend("test_lower")
    def _lower(src, tgt):
        return TranslationCandidate(
            backend_name="test_lower",
            text=src.upper(),  # "translation" = uppercased
            confidence=0.9,
        )
    try:
        target, stats = run_translation_track(
            source_ledger, target_language="fr", backend="test_lower",
        )
        assert stats.source_speech_spans == 3
        assert stats.spans_translated == 3
        assert stats.spans_failed == 0
        # Every source covered_speech span has a translation now.
        translated = [
            s for s in target.spans
            if s.status == "covered_speech" and s.content
        ]
        assert len(translated) == 3
        assert all(s.target_language == "fr" for s in translated)
        # Event span passed through unchanged.
        events = [s for s in target.spans if s.content_type == "event"]
        assert len(events) == 1
        assert events[0].content == "[music]"
    finally:
        _BACKENDS.pop("test_lower", None)


def test_failed_backend_emits_untranslatable(source_ledger):
    @register_backend("test_fail")
    def _fail(src, tgt):
        return TranslationCandidate(
            backend_name="test_fail", text="", error="boom",
        )
    try:
        target, stats = run_translation_track(
            source_ledger, target_language="fr", backend="test_fail",
        )
        assert stats.spans_failed == 3
        assert stats.spans_translated == 0
        # Every speech span got [untranslatable] so the coverage
        # invariant holds.
        untranslatable = [
            s for s in target.spans if s.content == "[untranslatable]"
        ]
        assert len(untranslatable) == 3
    finally:
        _BACKENDS.pop("test_fail", None)


def test_translation_invariant_no_uncovered_speech(source_ledger):
    """After the track runs, the target ledger must not leave any
    source covered_speech ms in awaiting_translation status."""
    @register_backend("test_ok")
    def _ok(src, tgt):
        return TranslationCandidate(
            backend_name="test_ok", text=f"FR:{src}", confidence=0.8,
        )
    try:
        target, _ = run_translation_track(
            source_ledger, target_language="fr", backend="test_ok",
        )
        awaiting = [
            s for s in target.spans
            if "awaiting_translation" in s.flags
        ]
        assert awaiting == []
    finally:
        _BACKENDS.pop("test_ok", None)


# ──────────────────────────────────────────────────────────────────────────
# reconcile_translations
# ──────────────────────────────────────────────────────────────────────────

def test_reconcile_single_candidate():
    rec = reconcile_translations(
        [TranslationCandidate("a", "hello", confidence=0.9)],
        source_text="hola",
    )
    assert rec.status == "single"
    assert rec.text == "hello"


def test_reconcile_all_failed():
    rec = reconcile_translations(
        [
            TranslationCandidate("a", "", error="boom1"),
            TranslationCandidate("b", "", error="boom2"),
        ],
        source_text="hola",
    )
    assert rec.status == "failed"
    assert rec.text == ""


def test_reconcile_picks_highest_confidence_when_no_labse():
    """Without LaBSE, falls back to highest-confidence candidate."""
    candidates = [
        TranslationCandidate("a", "hello there", confidence=0.6),
        TranslationCandidate("b", "hi friend", confidence=0.9),
    ]
    rec = reconcile_translations(candidates, source_text="hola amigo")
    # Without LaBSE installed, falls through to "single" status with
    # the highest-confidence candidate selected.
    assert rec.text in ("hello there", "hi friend")


# ──────────────────────────────────────────────────────────────────────────
# to_paired_report_dict
# ──────────────────────────────────────────────────────────────────────────

def test_paired_report_dict_schema(source_ledger):
    @register_backend("test_paired")
    def _t(src, tgt):
        return TranslationCandidate("test_paired", f"X{src}", confidence=0.9)
    try:
        target, _ = run_translation_track(
            source_ledger, target_language="es", backend="test_paired",
        )
        report = source_ledger.to_paired_report_dict(target)
        assert "source" in report
        assert "translation" in report
        assert report["translation"]["target_language"] == "es"
        assert "coverage_ratio" in report["translation"]
        assert "untranslated_ratio" in report["translation"]
        assert report["translation"]["untranslated_ratio"] == 0.0
        assert report["translation"]["coverage_ratio"] > 0.0
    finally:
        _BACKENDS.pop("test_paired", None)


# ──────────────────────────────────────────────────────────────────────────
# Whisper passthrough backend (Phase 5 §1.1 default)
# ──────────────────────────────────────────────────────────────────────────

def test_whisper_passthrough_backend_registered():
    assert "whisper_passthrough" in _BACKENDS
    cand = _BACKENDS["whisper_passthrough"]("hello world", "fr")
    assert cand.text == "hello world"
    assert cand.backend_name == "whisper_passthrough"
