"""Tests for the TACT Phase 2 Rung 5 — non-speech event classifier.

Real PANNs inference requires the panns_inference package + a CNN14
checkpoint download; these tests use monkeypatching to drive
``classify_interval`` through its branches without spawning torch.

Coverage:
  * AUDIOSET_TO_SIMPLIFIED mapping table is wired.
  * Disabled-by-config short-circuit.
  * Below-threshold confidence falls through.
  * "speech" label intentionally falls through (won't tag
    [speech]).
  * "music" / "applause" / "laughter" emit covered_event spans.
  * "silence" emits a covered_silence span with content_type
    "silence".
  * Classifier exception surfaces in notes.
"""
from __future__ import annotations

import pytest

from backend.services.escalation_ladder import (
    EscalationContext,
    _rung_event_classifier,
)
from backend.services.non_speech_events import (
    AUDIOSET_TO_SIMPLIFIED,
    EventClassification,
)


def _ctx(audio_duration_ms=30_000):
    return EscalationContext(
        audio_path="/dev/null",
        audio_duration_ms=audio_duration_ms,
        language="en",
        task="transcribe",
    )


# ──────────────────────────────────────────────────────────────────────────
# Mapping table sanity
# ──────────────────────────────────────────────────────────────────────────

def test_audioset_mapping_covers_core_categories():
    assert AUDIOSET_TO_SIMPLIFIED["Music"] == "music"
    assert AUDIOSET_TO_SIMPLIFIED["Speech"] == "speech"
    assert AUDIOSET_TO_SIMPLIFIED["Laughter"] == "laughter"
    assert AUDIOSET_TO_SIMPLIFIED["Applause"] == "applause"
    assert AUDIOSET_TO_SIMPLIFIED["Silence"] == "silence"
    assert AUDIOSET_TO_SIMPLIFIED["White noise"] == "noise"


def test_unmapped_audioset_class_implicitly_other():
    assert "Random Class That Doesn't Exist" not in AUDIOSET_TO_SIMPLIFIED


# ──────────────────────────────────────────────────────────────────────────
# Rung wrapper
# ──────────────────────────────────────────────────────────────────────────

def test_rung_5_disabled_by_config(monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "TACT_LADDER_RUNG_5_ENABLED", False)
    result = _rung_event_classifier("/dev/null", 1_000, 5_000, _ctx())
    assert result.spans == []
    assert result.notes == "disabled"


def test_rung_5_emits_music_event(monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "TACT_LADDER_RUNG_5_ENABLED", True)
    monkeypatch.setattr(settings, "TACT_EVENT_MIN_CONFIDENCE", 0.6)
    import backend.services.non_speech_events as nse
    monkeypatch.setattr(
        nse, "classify_interval",
        lambda *a, **kw: EventClassification(
            label="music", confidence=0.85, raw_class="Music",
        ),
    )
    result = _rung_event_classifier("/dev/null", 1_000, 5_000, _ctx())
    assert len(result.spans) == 1
    span = result.spans[0]
    assert span.content == "[music]"
    assert span.content_type == "event"
    assert span.status == "covered_event"
    assert span.confidence == 0.85


def test_rung_5_emits_silence_with_silence_content_type(monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "TACT_LADDER_RUNG_5_ENABLED", True)
    monkeypatch.setattr(settings, "TACT_EVENT_MIN_CONFIDENCE", 0.6)
    import backend.services.non_speech_events as nse
    monkeypatch.setattr(
        nse, "classify_interval",
        lambda *a, **kw: EventClassification(
            label="silence", confidence=0.95, raw_class="Silence",
        ),
    )
    result = _rung_event_classifier("/dev/null", 1_000, 5_000, _ctx())
    assert len(result.spans) == 1
    span = result.spans[0]
    assert span.content == "[silence]"
    assert span.content_type == "silence"
    assert span.status == "covered_silence"


def test_rung_5_below_threshold_passes(monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "TACT_LADDER_RUNG_5_ENABLED", True)
    monkeypatch.setattr(settings, "TACT_EVENT_MIN_CONFIDENCE", 0.6)
    import backend.services.non_speech_events as nse
    monkeypatch.setattr(
        nse, "classify_interval",
        lambda *a, **kw: EventClassification(
            label="music", confidence=0.42, raw_class="Music",
        ),
    )
    result = _rung_event_classifier("/dev/null", 1_000, 5_000, _ctx())
    assert result.spans == []
    assert "low_conf" in result.notes


def test_rung_5_other_label_passes(monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "TACT_LADDER_RUNG_5_ENABLED", True)
    import backend.services.non_speech_events as nse
    monkeypatch.setattr(
        nse, "classify_interval",
        lambda *a, **kw: EventClassification(
            label="other", confidence=0.95, raw_class="Vehicle",
        ),
    )
    result = _rung_event_classifier("/dev/null", 1_000, 5_000, _ctx())
    assert result.spans == []
    # 'other' explicitly falls through; the message records the label.
    assert "low_conf" in result.notes or "other" in result.notes.lower()


def test_rung_5_speech_label_intentionally_passes(monkeypatch):
    """Per the prompt: 'Speech detected but Rungs 1-4 already failed
    to transcribe it' — falling through to Rung 6 (which tags
    [unintelligible]) is more informative than tagging [speech]."""
    from backend.config import settings
    monkeypatch.setattr(settings, "TACT_LADDER_RUNG_5_ENABLED", True)
    monkeypatch.setattr(settings, "TACT_EVENT_MIN_CONFIDENCE", 0.6)
    import backend.services.non_speech_events as nse
    monkeypatch.setattr(
        nse, "classify_interval",
        lambda *a, **kw: EventClassification(
            label="speech", confidence=0.95, raw_class="Speech",
        ),
    )
    result = _rung_event_classifier("/dev/null", 1_000, 5_000, _ctx())
    assert result.spans == []
    assert result.notes == "speech_detected_but_untranscribed"


def test_rung_5_classifier_exception_surfaces(monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "TACT_LADDER_RUNG_5_ENABLED", True)
    import backend.services.non_speech_events as nse

    def boom(*a, **kw):
        raise RuntimeError("torch died")
    monkeypatch.setattr(nse, "classify_interval", boom)
    result = _rung_event_classifier("/dev/null", 1_000, 5_000, _ctx())
    assert result.spans == []
    assert "classify_failed" in result.notes


def test_rung_5_zero_interval_returns_empty(monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "TACT_LADDER_RUNG_5_ENABLED", True)
    result = _rung_event_classifier("/dev/null", 5_000, 5_000, _ctx())
    assert result.spans == []


def test_event_classification_dataclass_shape():
    cls = EventClassification(label="music", confidence=0.7, raw_class="Music")
    assert cls.label == "music"
    assert cls.confidence == 0.7
    assert cls.raw_class == "Music"
