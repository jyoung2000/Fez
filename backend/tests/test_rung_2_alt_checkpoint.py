"""Tests for the TACT Phase 2 Rung 2 — alternate-checkpoint Whisper.

Real Whisper invocation needs CUDA + downloaded weights and is not
exercised here; these tests use monkeypatching to mock
``transcribe_audio_slice_subprocess`` and ``_extract_audio_slice`` so
the rung's logic is verified without spawning a subprocess.

Coverage:
  * Selection rule (`_select_alt_checkpoint`) — every primary
    branch of the table maps to the documented alternate.
  * Disabled-by-config short-circuit.
  * Decline when no alternate exists (e.g., primary == "tiny").
  * Clamping spans to the original interval (padding-zone words
    are dropped).
  * Subprocess failure path is non-fatal.
  * Hallucinated-fill text is filtered.
"""
from __future__ import annotations

import pytest

from backend.services.escalation_ladder import (
    EscalationContext,
    _rung_alt_whisper_checkpoint,
    _select_alt_checkpoint,
)


# ──────────────────────────────────────────────────────────────────────────
# _select_alt_checkpoint table
# ──────────────────────────────────────────────────────────────────────────

def test_alt_checkpoint_large_v3_turbo_to_large_v3():
    assert _select_alt_checkpoint("large-v3-turbo") == "large-v3"


def test_alt_checkpoint_small_to_medium():
    assert _select_alt_checkpoint("small") == "medium"


def test_alt_checkpoint_unknown_returns_none():
    assert _select_alt_checkpoint("tiny") is None
    assert _select_alt_checkpoint("") is None
    assert _select_alt_checkpoint("base") is None


def test_alt_checkpoint_medium_picks_turbo_when_vram_allows(monkeypatch):
    import backend.services.transcription as t
    monkeypatch.setattr(t, "_get_gpu_free_mb", lambda: 5000)
    assert _select_alt_checkpoint("medium") == "large-v3-turbo"


def test_alt_checkpoint_medium_picks_large_v3_when_vram_low(monkeypatch):
    import backend.services.transcription as t
    monkeypatch.setattr(t, "_get_gpu_free_mb", lambda: 1500)
    assert _select_alt_checkpoint("medium") == "large-v3"


# ──────────────────────────────────────────────────────────────────────────
# Rung wrapper behavior
# ──────────────────────────────────────────────────────────────────────────

def _ctx(audio_duration_ms=30_000):
    return EscalationContext(
        audio_path="/dev/null",
        audio_duration_ms=audio_duration_ms,
        language="en",
        task="transcribe",
    )


def test_rung_2_disabled_by_config(monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "TACT_LADDER_RUNG_2_ENABLED", False)
    result = _rung_alt_whisper_checkpoint("/dev/null", 1_000, 5_000, _ctx())
    assert result.spans == []
    assert result.notes == "disabled"


def test_rung_2_declines_when_no_alternate(monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "TACT_LADDER_RUNG_2_ENABLED", True)
    monkeypatch.setattr(settings, "WHISPER_MODEL", "tiny")
    result = _rung_alt_whisper_checkpoint("/dev/null", 1_000, 5_000, _ctx())
    assert result.spans == []
    assert result.notes.startswith("no_alt_checkpoint_for_")


def test_rung_2_zero_interval_returns_empty(monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "TACT_LADDER_RUNG_2_ENABLED", True)
    monkeypatch.setattr(settings, "WHISPER_MODEL", "small")
    result = _rung_alt_whisper_checkpoint("/dev/null", 5_000, 5_000, _ctx())
    assert result.spans == []


def test_rung_2_emits_spans_clamped_to_interval(monkeypatch, tmp_path):
    """Padded slice runs from [start - 2s, end + 2s]; words landing
    inside the padding zone (i.e., outside the original interval) are
    dropped. Words inside are emitted with timestamps in the original
    audio's frame of reference."""
    from backend.config import settings
    monkeypatch.setattr(settings, "TACT_LADDER_RUNG_2_ENABLED", True)
    monkeypatch.setattr(settings, "WHISPER_MODEL", "small")
    audio_path = str(tmp_path / "fake.wav")
    interval_start_ms = 5_000
    interval_end_ms = 8_000
    pad_sec = 2.0
    # In the padded slice, the offset from the slice-start (3.0s) is
    # the segment's reported "start" in the worker JSON. So a word
    # that should land at 6.0 s in the original audio reports
    # start = 3.0 (because slice begins at 3.0 s in original time).
    # padded_start_sec for [5000, 8000] is max(0, 3.0) = 3.0, so a
    # segment with start=3.0 emits at 6.0 in original time.
    fake_segments = [
        # Word at 4.0 s (inside padding before, should be dropped).
        {"start": 1.0, "end": 1.5, "text": "ignore_me",
         "avg_logprob": -0.2, "no_speech_prob": 0.05},
        # Word at 6.0 s (inside original interval, should be emitted).
        {"start": 3.0, "end": 3.5, "text": "keep_me",
         "avg_logprob": -0.2, "no_speech_prob": 0.05},
        # Word at 9.0 s (inside padding after, should be dropped).
        {"start": 6.0, "end": 6.5, "text": "ignore_me_too",
         "avg_logprob": -0.2, "no_speech_prob": 0.05},
    ]

    import backend.services.transcription as t
    monkeypatch.setattr(
        t, "transcribe_audio_slice_subprocess",
        lambda *args, **kwargs: fake_segments,
    )
    import backend.services.transcription_gap_filler as gf
    monkeypatch.setattr(gf, "_extract_audio_slice",
                        lambda *args, **kwargs: True)

    result = _rung_alt_whisper_checkpoint(
        audio_path, interval_start_ms, interval_end_ms, _ctx(),
    )
    # Only the middle word survives the padding-zone clamp.
    assert len(result.spans) == 1
    span = result.spans[0]
    assert span.content == "keep_me"
    assert interval_start_ms <= span.start_ms < span.end_ms <= interval_end_ms


def test_rung_2_subprocess_failure_returns_empty(monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "TACT_LADDER_RUNG_2_ENABLED", True)
    monkeypatch.setattr(settings, "WHISPER_MODEL", "small")
    import backend.services.transcription as t
    import backend.services.transcription_gap_filler as gf

    def boom(*args, **kwargs):
        raise RuntimeError("subprocess died")
    monkeypatch.setattr(t, "transcribe_audio_slice_subprocess", boom)
    monkeypatch.setattr(gf, "_extract_audio_slice",
                        lambda *args, **kwargs: True)
    result = _rung_alt_whisper_checkpoint("/dev/null", 1_000, 5_000, _ctx())
    assert result.spans == []
    assert "transcribe_failed" in result.notes


def test_rung_2_filters_hallucinated_text(monkeypatch):
    """Boilerplate text from `_is_hallucinated_fill` is filtered."""
    from backend.config import settings
    monkeypatch.setattr(settings, "TACT_LADDER_RUNG_2_ENABLED", True)
    monkeypatch.setattr(settings, "WHISPER_MODEL", "small")
    import backend.services.transcription as t
    import backend.services.transcription_gap_filler as gf
    monkeypatch.setattr(
        t, "transcribe_audio_slice_subprocess",
        lambda *args, **kwargs: [
            {"start": 3.0, "end": 3.5, "text": "thank you for watching",
             "avg_logprob": -0.2, "no_speech_prob": 0.05},
        ],
    )
    monkeypatch.setattr(gf, "_extract_audio_slice",
                        lambda *args, **kwargs: True)
    result = _rung_alt_whisper_checkpoint(
        "/dev/null", 5_000, 8_000, _ctx(),
    )
    # _is_hallucinated_fill drops boilerplate; spans is empty.
    assert result.spans == []
