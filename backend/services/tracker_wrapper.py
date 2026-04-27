"""Tracker dispatcher — picks SAMURAI when GPU is visible, OpenCV otherwise.

Phase A of the 2026 SOTA reframing rollout.

Public surface (unchanged from pre-Phase-A):

    SlotTracker(slot_id, backend="KCF")
        Legacy-shape OpenCV slot tracker. Keeps ``update()`` returning
        ``Optional[tuple]`` for compatibility with pre-Phase-A callers
        (``dense_propagator.py``, ``test_interpolated_timeline.py``).

New surface (used by Phase-B+):

    create_tracker(slot_id, *, backend="auto") -> Tracker
        Returns a tracker matching the unified Phase-A/B API where
        ``update(frame)`` returns ``(ok, bbox, mask | None)``. The
        backend is chosen per the ``CLIPAI_TRACKER_BACKEND`` env var
        (``samurai`` | ``opencv`` | ``auto``) — ``auto`` picks SAMURAI
        when CUDA is visible, OpenCV otherwise.

Routing rules:

    * ``CLIPAI_TRACKER_BACKEND=samurai`` — SAMURAI required. Raises
      ``RuntimeError`` if GPU not visible.
    * ``CLIPAI_TRACKER_BACKEND=opencv`` — OpenCV always.
    * ``CLIPAI_TRACKER_BACKEND=auto`` (default) — SAMURAI if GPU
      visible, OpenCV otherwise. Logs the choice once per process.

This module never imports ``sam2`` at module-load time. The import is
deferred to inside :class:`SamuraiTracker.init` so machines without
the upstream wheel can still ``import tracker_wrapper`` and use the
OpenCV path.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Re-export the legacy OpenCV slot tracker so existing imports of
# ``from .tracker_wrapper import SlotTracker`` keep working.
from backend.services.tracker_wrapper_opencv import (  # noqa: E402
    OpenCvSlotTracker,
    SlotTracker,
    _create_tracker_instance,
)


_BACKEND_ENV_VAR = "CLIPAI_TRACKER_BACKEND"
_VALID_BACKENDS = ("samurai", "opencv", "auto")
_BACKEND_LOGGED = False


def _resolve_backend(requested: Optional[str]) -> str:
    """Resolve the backend name. Lower-cases + validates."""
    if requested is None:
        requested = os.environ.get(_BACKEND_ENV_VAR, "auto")
    requested = (requested or "auto").strip().lower()
    if requested not in _VALID_BACKENDS:
        logger.warning(
            "Unknown %s=%r — defaulting to 'auto'. Valid: %s",
            _BACKEND_ENV_VAR, requested, ",".join(_VALID_BACKENDS),
        )
        return "auto"
    return requested


def _gpu_visible() -> bool:
    try:
        from backend.services.transcription import _probe_gpu_availability  # type: ignore
        return bool(_probe_gpu_availability().get("visible"))
    except Exception:
        return False


def _log_backend_choice(chosen: str, requested: str) -> None:
    global _BACKEND_LOGGED
    if _BACKEND_LOGGED:
        return
    _BACKEND_LOGGED = True
    logger.info(
        "[Samurai] tracker dispatcher: %s=%r → using %s backend",
        _BACKEND_ENV_VAR, requested, chosen,
    )


def create_tracker(
    slot_id: int = 0,
    *,
    backend: Optional[str] = None,
    opencv_backend: str = "KCF",
):
    """Construct a tracker instance per the resolved backend.

    Returns an object with the unified API:

        ok: bool, bbox: tuple | None, mask: ndarray | None = tracker.update(frame)
    """
    requested = _resolve_backend(backend)

    if requested == "opencv":
        _log_backend_choice("opencv", requested)
        return OpenCvSlotTracker(slot_id, backend=opencv_backend)

    # samurai or auto
    gpu_ok = _gpu_visible()
    if requested == "samurai" and not gpu_ok:
        raise RuntimeError(
            f"{_BACKEND_ENV_VAR}=samurai requested but no CUDA GPU visible. "
            "Set CLIPAI_TRACKER_BACKEND=auto to fall back automatically, or "
            "CLIPAI_TRACKER_BACKEND=opencv to force CPU."
        )

    if requested == "auto" and not gpu_ok:
        _log_backend_choice("opencv", requested)
        return OpenCvSlotTracker(slot_id, backend=opencv_backend)

    # GPU is visible — try SAMURAI, fall back to OpenCV on construction
    # failure (e.g. sam2 wheel missing in this image).
    try:
        from backend.services.samurai_tracker import SamuraiTracker
        tracker = SamuraiTracker(slot_id=slot_id)
        _log_backend_choice("samurai", requested)
        return tracker
    except RuntimeError as exc:
        if requested == "samurai":
            raise  # explicit ask — surface the error
        logger.warning(
            "[Samurai] auto-fallback to opencv: %s", exc,
        )
        _log_backend_choice("opencv", requested)
        return OpenCvSlotTracker(slot_id, backend=opencv_backend)


# ── Diagnostic helper ───────────────────────────────────────────────


def get_active_backend(backend: Optional[str] = None) -> str:
    """Return the backend ``create_tracker`` would pick (without constructing).

    Useful for telemetry / WebSocket logs without paying the model
    download cost.
    """
    requested = _resolve_backend(backend)
    if requested == "opencv":
        return "opencv"
    if not _gpu_visible():
        return "opencv"
    return "samurai"


__all__ = [
    "create_tracker",
    "get_active_backend",
    "OpenCvSlotTracker",
    "SlotTracker",
    "_create_tracker_instance",
]
