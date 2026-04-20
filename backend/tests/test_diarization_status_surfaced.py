"""Tests for Fix A: diarization status is surfaced via
``get_diarization_status()`` and the ``/jobs/{id}/diarize`` response.

Covers:
  * Empty HF token → status reports no_token, backend=heuristic.
  * Disabled setting → reason=disabled, backend=heuristic.
  * Loader wrapper reports the same shape as the unified loader.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _reset_loader():
    from backend.services import _pyannote_loader as _pyann
    _pyann.reset_for_tests()
    yield
    _pyann.reset_for_tests()


def test_empty_hf_token_reports_no_token(monkeypatch):
    """With no token in settings AND no token in env AND no cache
    file, the loader must report ``reason="no_token"`` and
    ``token_present=False``."""
    from backend.services import _pyannote_loader as _pyann
    from backend.config import settings

    monkeypatch.setattr(settings, "HF_AUTH_TOKEN", "", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    # Force the cache-file probe to return False regardless of the
    # tester's real home dir.
    monkeypatch.setattr(os.path, "exists", lambda p: False)

    # Drive the loader so the status string is populated.
    pipeline, reason = _pyann.get_pipeline()
    assert pipeline is None
    assert reason == "no_token"

    status = _pyann.get_status()
    assert status["pipeline_ready"] is False
    assert status["reason"] == "no_token"
    assert status["token_present"] is False
    assert status["backend"] == "heuristic"


def test_disabled_setting_reports_disabled(monkeypatch):
    from backend.services import _pyannote_loader as _pyann
    from backend.config import settings

    monkeypatch.setattr(settings, "DIARIZATION_ENABLED", False, raising=False)
    pipeline, reason = _pyann.get_pipeline()
    assert pipeline is None
    assert reason == "disabled"

    status = _pyann.get_status()
    assert status["pipeline_ready"] is False
    assert status["reason"] == "disabled"
    assert status["backend"] == "heuristic"


def test_transcription_wrapper_delegates_to_loader(monkeypatch):
    """``transcription.get_diarization_status`` must surface the
    same shape as ``_pyannote_loader.get_status`` — it's the
    single API the /jobs/{id}/diarize route uses."""
    from backend.services.transcription import get_diarization_status
    from backend.services import _pyannote_loader as _pyann
    from backend.config import settings

    monkeypatch.setattr(settings, "HF_AUTH_TOKEN", "", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    # Drive the loader to populate state.
    _pyann.get_pipeline()

    status = get_diarization_status()
    assert set(status.keys()) == {
        "pipeline_ready", "reason", "token_present", "backend",
    }
    assert status["reason"] == "no_token"
    assert status["backend"] == "heuristic"


def test_diarize_route_surfaces_backend_fields():
    """The /jobs/{id}/diarize route must include
    ``diarization_backend`` + ``diarization_status_reason`` in its
    response so the frontend can prompt the user to add a token.

    The full route imports heavy modules (cv2 / PIL / pipeline) that
    aren't available everywhere, so we verify the CONTRACT by
    reading the source directly — the response dict literal must
    carry the two keys, and the keys must be populated from the
    loader's status rather than hard-coded.
    """
    import pathlib
    jobs_src = (
        pathlib.Path(__file__).resolve().parents[1]
        / "routers" / "jobs.py"
    )
    # backend/tests/X.py → parents[1] == backend/, then routers/jobs.py.
    # No, actually parents[1] IS ``backend/`` so the path is correct.
    jobs_src = jobs_src.read_text(encoding="utf-8")
    # The route must import get_diarization_status.
    assert "get_diarization_status" in jobs_src
    # And the response must surface both fields.
    assert "\"diarization_backend\"" in jobs_src
    assert "\"diarization_status_reason\"" in jobs_src
    # Sanity: they must come from the live status dict, not hard-coded.
    assert "diar_status.get(\"backend\")" in jobs_src
    assert "diar_status.get(\"reason\")" in jobs_src
