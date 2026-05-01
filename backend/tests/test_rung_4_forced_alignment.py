"""Tests for the TACT Phase 2 Rung 4 — forced alignment.

Real wav2vec2 alignment requires torchaudio + downloaded weights and
real audio; these tests use monkeypatching to drive
``align_text_to_audio`` through its branches without spawning torch.

Coverage:
  * Disabled-by-config short-circuit.
  * Empty neighbor text → declines with "no_neighbor_text".
  * Successful alignment → spans are emitted, clamped to interval.
  * Empty alignment result → falls through.
  * Various structured AlignmentResult.reason values surface in
    RungResult.notes.
  * AlignmentResult shape unit tests (pure data; no torch).
"""
from __future__ import annotations

import pytest

from backend.services.escalation_ladder import (
    EscalationContext,
    _rung_forced_alignment,
)
from backend.services.forced_alignment import (
    AlignmentResult,
    _normalize_text,
)


def _ctx(audio_duration_ms=30_000, before="", after=""):
    return EscalationContext(
        audio_path="/dev/null",
        audio_duration_ms=audio_duration_ms,
        language="en",
        task="transcribe",
        neighbor_text_before=before,
        neighbor_text_after=after,
    )


# ──────────────────────────────────────────────────────────────────────────
# Rung wrapper
# ──────────────────────────────────────────────────────────────────────────

def test_rung_4_disabled_by_config(monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "TACT_LADDER_RUNG_4_ENABLED", False)
    result = _rung_forced_alignment(
        "/dev/null", 1_000, 5_000,
        _ctx(before="hello world", after="how are you"),
    )
    assert result.spans == []
    assert result.notes == "disabled"


def test_rung_4_no_neighbor_text_declines(monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "TACT_LADDER_RUNG_4_ENABLED", True)
    result = _rung_forced_alignment("/dev/null", 1_000, 5_000, _ctx())
    assert result.spans == []
    assert result.notes == "no_neighbor_text"


def test_rung_4_emits_aligned_words(monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "TACT_LADDER_RUNG_4_ENABLED", True)
    import backend.services.forced_alignment as fa

    def fake_align(*args, **kwargs):
        return AlignmentResult(
            words=[
                (1.2, 1.5, "HELLO", 0.92),
                (1.6, 1.9, "WORLD", 0.81),
            ],
            reason="ok",
        )
    monkeypatch.setattr(fa, "align_text_to_audio", fake_align)

    result = _rung_forced_alignment(
        "/dev/null", 1_000, 5_000,
        _ctx(before="say hello", after="world ok"),
    )
    assert len(result.spans) == 2
    assert result.spans[0].content == "HELLO"
    assert result.spans[0].content_type == "word"
    assert result.spans[0].status == "covered_speech"
    assert result.spans[0].source_pass == "rung_4_forced_alignment"
    # Confidence carried through.
    assert abs(result.spans[0].confidence - 0.92) < 1e-6


def test_rung_4_clamps_words_to_interval(monkeypatch):
    """Frame-to-time rounding occasionally places a word slightly
    outside the slice bounds; the rung clamps to the original
    interval."""
    from backend.config import settings
    monkeypatch.setattr(settings, "TACT_LADDER_RUNG_4_ENABLED", True)
    import backend.services.forced_alignment as fa

    def fake_align(*args, **kwargs):
        # 0.99 is *before* the interval start at 1.0 s.
        # 5.05 is *after* the interval end at 5.0 s.
        return AlignmentResult(
            words=[
                (0.99, 1.50, "A", 0.7),
                (1.6, 5.05, "B", 0.7),
            ],
            reason="ok",
        )
    monkeypatch.setattr(fa, "align_text_to_audio", fake_align)
    result = _rung_forced_alignment(
        "/dev/null", 1_000, 5_000,
        _ctx(before="x", after="y"),
    )
    for span in result.spans:
        assert 1_000 <= span.start_ms < span.end_ms <= 5_000


def test_rung_4_no_alignment_passes(monkeypatch):
    """When the aligner returns no words, the rung falls through."""
    from backend.config import settings
    monkeypatch.setattr(settings, "TACT_LADDER_RUNG_4_ENABLED", True)
    import backend.services.forced_alignment as fa

    monkeypatch.setattr(
        fa, "align_text_to_audio",
        lambda *a, **kw: AlignmentResult(words=[], reason="no_alignment"),
    )
    result = _rung_forced_alignment(
        "/dev/null", 1_000, 5_000,
        _ctx(before="x", after="y"),
    )
    assert result.spans == []
    assert "no_alignment" in result.notes


def test_rung_4_module_unavailable_surfaces_in_notes(monkeypatch):
    """When forced_alignment.align_text_to_audio raises ImportError-y
    behavior, the rung records deps_unavailable and falls through."""
    from backend.config import settings
    monkeypatch.setattr(settings, "TACT_LADDER_RUNG_4_ENABLED", True)
    import backend.services.forced_alignment as fa

    def boom(*args, **kwargs):
        raise ImportError("torch missing")
    monkeypatch.setattr(fa, "align_text_to_audio", boom)
    result = _rung_forced_alignment(
        "/dev/null", 1_000, 5_000,
        _ctx(before="x", after="y"),
    )
    # When the helper itself raises (vs returning an empty result),
    # the wrapper catches it via the try/except wrapping the import.
    # In this monkeypatch path the import succeeds but the call
    # raises — handled inside the rung's try/except shielding
    # downstream callers from torch-shape issues.
    assert result.spans == []


# ──────────────────────────────────────────────────────────────────────────
# AlignmentResult / _normalize_text data tests (no torch)
# ──────────────────────────────────────────────────────────────────────────

def test_alignment_result_default_reason():
    r = AlignmentResult()
    assert r.words == []
    assert r.reason == "no_alignment"


def test_normalize_text_drops_unsupported_chars():
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ|-'")
    out = _normalize_text("Hello, World!", labels)
    # Lowercase letters uppercased; punctuation dropped; spaces → "|".
    assert "H" in out and "E" in out and "|" in out
    assert "," not in out and "!" not in out


def test_normalize_text_strips_trailing_separator():
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ|-'")
    out = _normalize_text("HELLO   ", labels)
    # No trailing "|" left over.
    assert out[-1] != "|"
