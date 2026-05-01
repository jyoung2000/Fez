"""Tests for the TACT Phase 4 Parakeet consensus orchestrator.

Real NeMo transcription requires CUDA + downloaded weights and is
not exercised here; these tests cover the gating logic, the
declined-gracefully contract, the worker JSON-shape translation, and
that NeMo is *not* imported when consensus is disabled.

The acceptance gate (§3.7 of the prompt) explicitly requires:
``test_consensus_disabled_no_import`` — confirm nemo_toolkit is
absent from sys.modules of a fresh interpreter when
``TACT_CONSENSUS_ENABLED=False``.
"""
from __future__ import annotations

import subprocess
import sys

from backend.services.parakeet_transcriber import (
    ConsensusGateResult,
    _can_run_consensus,
)
from backend.services.parakeet_worker import _segments_from_nemo


# ──────────────────────────────────────────────────────────────────────────
# Gate logic
# ──────────────────────────────────────────────────────────────────────────

def test_gate_disabled_when_flag_off(monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "TACT_CONSENSUS_ENABLED", False)
    result = _can_run_consensus()
    assert isinstance(result, ConsensusGateResult)
    assert not result.allowed
    assert result.reason == "disabled_by_config"


def test_gate_blocks_when_vram_below_threshold(monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "TACT_CONSENSUS_ENABLED", True)
    monkeypatch.setattr(settings, "TACT_CONSENSUS_MIN_FREE_VRAM_MB", 8000)
    # Force a known low VRAM probe.
    import backend.services.transcription as t
    monkeypatch.setattr(t, "_get_gpu_free_mb", lambda: 1500)
    result = _can_run_consensus()
    assert not result.allowed
    assert "insufficient_vram" in result.reason
    assert result.free_vram_mb == 1500


def test_gate_allows_when_vram_meets_threshold(monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "TACT_CONSENSUS_ENABLED", True)
    monkeypatch.setattr(settings, "TACT_CONSENSUS_MIN_FREE_VRAM_MB", 1000)
    import backend.services.transcription as t
    monkeypatch.setattr(t, "_get_gpu_free_mb", lambda: 4000)
    result = _can_run_consensus()
    assert result.allowed
    assert result.reason == "ok"
    assert result.free_vram_mb == 4000


def test_gate_handles_probe_exception(monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "TACT_CONSENSUS_ENABLED", True)
    import backend.services.transcription as t

    def _boom():
        raise RuntimeError("driver died")
    monkeypatch.setattr(t, "_get_gpu_free_mb", _boom)
    result = _can_run_consensus()
    assert not result.allowed
    assert result.reason == "vram_probe_failed"


# ──────────────────────────────────────────────────────────────────────────
# Worker JSON-shape translation
# ──────────────────────────────────────────────────────────────────────────

class _FakeHyp:
    def __init__(self, text, words=None, segment=None):
        self.text = text
        self.timestamp = {
            "word": words or [],
            "segment": segment or [],
        }


def test_segments_from_nemo_with_word_timestamps():
    hyp = _FakeHyp(
        text="hello world",
        words=[
            {"start": 0.1, "end": 0.5, "word": "hello"},
            {"start": 0.6, "end": 1.0, "word": "world"},
        ],
        segment=[{"start": 0.1, "end": 1.0}],
    )
    out = _segments_from_nemo([hyp])
    assert len(out) == 1
    seg = out[0]
    assert seg["text"] == "hello world"
    assert seg["start"] == 0.1
    assert seg["end"] == 1.0
    assert len(seg["words"]) == 2
    assert seg["words"][0]["word"] == "hello"


def test_segments_from_nemo_falls_back_to_word_bounds():
    """No segment-level timestamp → use first/last word bounds."""
    hyp = _FakeHyp(
        text="alpha",
        words=[{"start": 5.0, "end": 5.5, "word": "alpha"}],
        segment=[],
    )
    out = _segments_from_nemo([hyp])
    assert len(out) == 1
    assert out[0]["start"] == 5.0
    assert out[0]["end"] == 5.5


def test_segments_from_nemo_empty_input():
    assert _segments_from_nemo(None) == []
    assert _segments_from_nemo([]) == []


def test_segments_from_nemo_skips_empty_text():
    hyp = _FakeHyp(text="   ", words=[], segment=[])
    assert _segments_from_nemo([hyp]) == []


# ──────────────────────────────────────────────────────────────────────────
# Acceptance gate §3.7: nemo_toolkit not imported when consensus disabled
# ──────────────────────────────────────────────────────────────────────────

def test_consensus_disabled_no_import():
    """Spin up a fresh interpreter with consensus disabled, import the
    parakeet_transcriber + the pipeline import path, confirm
    nemo_toolkit is *not* in sys.modules. Acceptance contract from
    the Phase 4 prompt §3.7."""
    code = (
        "import sys; "
        "from backend.config import settings; "
        "settings.TACT_CONSENSUS_ENABLED = False; "
        "from backend.services import parakeet_transcriber; "
        "assert 'nemo' not in sys.modules, "
        "'nemo unexpectedly imported when consensus disabled'; "
        "print('OK')"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=60,
    )
    assert res.returncode == 0, (
        f"subprocess failed: stderr={res.stderr[-500:]} stdout={res.stdout[-500:]}"
    )
    assert "OK" in res.stdout
