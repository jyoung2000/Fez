"""Shared singleton loader for the pyannote speaker-diarization pipeline.

Two call sites — ``backend.services.transcription`` (post-processing
``diarize_transcript_post``) and ``backend.services.speaker_diarization``
(the lip/audio fusion path) — used to each keep their own
``Pipeline.from_pretrained`` result behind their own lock with
slightly different token-probing logic. They could disagree on
speaker count or load/fail independently on the same clip.

Centralize the load here so both paths share:

  * One token probe (``HF_AUTH_TOKEN`` / ``HUGGINGFACE_TOKEN`` /
    ``HF_TOKEN`` / on-disk HF cache — whichever is set).
  * One singleton + lock.
  * One status enum so callers / the ``/jobs/{id}/diarize`` route
    can report WHY the pipeline isn't available.

Public API:

    get_pipeline() -> tuple[Optional[Pipeline], str]
    get_status() -> dict
    reset_for_tests() -> None  # used by unit tests only
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# Status values (string enum — easy to log and serialize).
STATUS_UNKNOWN = "unknown"       # loader hasn't been asked yet
STATUS_OK = "ok"                 # pipeline loaded + ready
STATUS_NO_TOKEN = "no_token"     # no token source visible
STATUS_LOAD_FAILED = "load_failed"  # token present but load raised
STATUS_DISABLED = "disabled"     # settings.DIARIZATION_ENABLED is False


_pipeline = None
_status = STATUS_UNKNOWN
_lock = threading.Lock()


def _resolve_token() -> Optional[str]:
    """Return the first non-empty token from the supported sources.

    Order matches the long-standing convention in this codebase:
    settings first (configurable via the admin UI), then the two
    common env vars. The ``~/.cache/huggingface/token`` file — set
    by ``huggingface-cli login`` — is a separate signal checked by
    :func:`token_present` but isn't returned here because pyannote
    reads it transparently when ``token=None`` is passed.
    """
    try:
        # Import lazily — this module is imported from both the
        # transcription module (which always has settings loaded)
        # and from unit tests that stub out config.
        from backend.config import settings
        if settings.HF_AUTH_TOKEN:
            return settings.HF_AUTH_TOKEN
    except Exception:
        pass
    return (
        os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_TOKEN")
        or None
    )


def _cache_file_token_present() -> bool:
    return os.path.exists(os.path.expanduser("~/.cache/huggingface/token"))


def token_present() -> bool:
    """True when any token source is visible (env var or cache file)."""
    return bool(_resolve_token() or _cache_file_token_present())


def _disabled() -> bool:
    """True when the DIARIZATION_ENABLED setting is off.

    Swallows any import error so unit tests that don't have
    ``backend.config`` fully set up can still exercise this module.
    """
    try:
        from backend.config import settings
        return not bool(settings.DIARIZATION_ENABLED)
    except Exception:
        return False


def get_pipeline() -> Tuple[Optional[object], str]:
    """Return ``(pipeline, status_str)``.

    The pipeline is ``None`` whenever ``status_str != "ok"``; the
    caller is expected to fall back to a lower tier (MFCC or
    heuristic) in that case. Safe to call from multiple threads.
    """
    global _pipeline, _status

    if _disabled():
        _status = STATUS_DISABLED
        return None, _status

    with _lock:
        if _pipeline is not None and _status == STATUS_OK:
            return _pipeline, _status

        token = _resolve_token()
        if not token and not _cache_file_token_present():
            logger.warning(
                "[Diarization] HF token missing — pyannote unavailable. "
                "Set HF_AUTH_TOKEN (settings) or HUGGINGFACE_TOKEN / "
                "HF_TOKEN (env) to enable real speaker diarization."
            )
            _status = STATUS_NO_TOKEN
            return None, _status

        try:
            from pyannote.audio import Pipeline
            _pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", token=token,
            )
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    _pipeline.to(torch.device("cuda"))
                    logger.info("[Diarization] pyannote loaded on CUDA")
                else:
                    logger.info("[Diarization] pyannote loaded on CPU")
            except Exception:
                # torch not importable — pyannote still works CPU-only.
                logger.info("[Diarization] pyannote loaded (torch unavailable)")
            _status = STATUS_OK
            return _pipeline, _status
        except Exception as e:
            logger.warning(
                "[Diarization] pyannote load failed: %s — falling back "
                "to heuristic / MFCC tier", e,
            )
            _pipeline = None
            _status = STATUS_LOAD_FAILED
            return None, _status


def get_status() -> dict:
    """Report loader state without triggering a load.

    Returns:
        ``{"pipeline_ready": bool, "reason": str,
           "token_present": bool, "backend": str}``

    ``backend`` is ``"pyannote"`` when the pipeline is loaded and
    ``"heuristic"`` otherwise. Keeps the top-level ``/jobs/{id}/diarize``
    response flat.
    """
    reason = _status
    is_disabled = _disabled()
    if is_disabled:
        reason = STATUS_DISABLED
    ready = (_pipeline is not None and reason == STATUS_OK)
    return {
        "pipeline_ready": ready,
        "reason": reason,
        "token_present": token_present(),
        "backend": "pyannote" if ready else "heuristic",
    }


def reset_for_tests() -> None:
    """Drop the cached pipeline + status. Tests only."""
    global _pipeline, _status
    with _lock:
        _pipeline = None
        _status = STATUS_UNKNOWN
