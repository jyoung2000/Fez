"""TACT Phase 4 — Parakeet (independent-architecture) transcription.

This module is the orchestrator for the consensus pass. It runs
``parakeet_worker.py`` as a subprocess (mirroring the pattern used by
``whisper_worker.py``) so NeMo's CUDA context is fully released when
the worker exits, freeing VRAM for downstream stages.

Entirely opt-in. The whole module is a no-op unless
``TACT_CONSENSUS_ENABLED=True`` AND a VRAM probe passes the
configured threshold. With consensus disabled, ``nemo_toolkit`` is
never imported in the parent process — confirmed by
``test_parakeet_transcriber.py::test_consensus_disabled_no_import``.

The consensus pass complements Whisper rather than replacing it.
Reconciliation across the two architectures (Whisper Transformer
encoder-decoder + Parakeet Conformer-TDT) catches errors that share
no failure mode — Whisper's silence hallucinations and Parakeet's
overlap collapses are largely orthogonal.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional

from backend.config import settings
from backend.models import TranscriptSegment, WordTimestamp

logger = logging.getLogger(__name__)


@dataclass
class ConsensusGateResult:
    """Why the consensus pass was (or was not) attempted."""
    allowed: bool
    reason: str
    free_vram_mb: int = 0


def can_run_consensus(*, override_min_vram_mb: Optional[int] = None) -> ConsensusGateResult:
    """Public alias for ``_can_run_consensus``.

    The pipeline imports this name; the underscore-prefixed version is
    kept for backwards compatibility with code already inside this
    module.
    """
    return _can_run_consensus(override_min_vram_mb=override_min_vram_mb)


def _can_run_consensus(*, override_min_vram_mb: Optional[int] = None) -> ConsensusGateResult:
    """Decide whether Parakeet may run for this job.

    The decision is conservative on purpose. Three reasons it can
    return ``allowed=False``:

      * The user disabled consensus globally (or for this job).
      * The VRAM probe failed (no GPU visible / nvidia-smi missing).
      * Free VRAM after the Whisper pass is below the configured
        threshold. Parakeet-tdt-0.6b int8 needs ~1.8 GB and we want
        a safety margin on top.

    Returns a structured result so the ladder can record the reason
    in stats — important for the 4 GB GPU contract where declining
    is the expected behavior, not an error.
    """
    if not bool(getattr(settings, "TACT_CONSENSUS_ENABLED", False)):
        return ConsensusGateResult(False, "disabled_by_config")
    # Lazy-imported so this module is import-cheap when the flag is off.
    try:
        from backend.services.transcription import _get_gpu_free_mb
    except Exception as e:
        return ConsensusGateResult(False, f"vram_probe_unavailable:{e}")
    try:
        free_mb = _get_gpu_free_mb()
    except Exception:
        return ConsensusGateResult(False, "vram_probe_failed")
    threshold = override_min_vram_mb or int(
        getattr(settings, "TACT_CONSENSUS_MIN_FREE_VRAM_MB", 2200)
    )
    if not free_mb or free_mb < threshold:
        return ConsensusGateResult(
            False,
            f"insufficient_vram_{free_mb}mb_lt_{threshold}mb",
            free_vram_mb=free_mb or 0,
        )
    return ConsensusGateResult(True, "ok", free_vram_mb=free_mb)


async def transcribe_with_parakeet_subprocess(
    audio_path: str,
    *,
    language: str = "",
    timeout_sec: float = 1800.0,
    cancel_check=None,
) -> list[TranscriptSegment]:
    """Run Parakeet in an isolated subprocess. Returns segment list.

    Mirrors ``transcribe_audio_subprocess`` from ``transcription.py``:
    spawns ``backend.services.parakeet_worker``, streams progress on
    stderr, parses JSON from a temp output file, releases all CUDA
    state when the subprocess exits.

    Raises ``RuntimeError`` on subprocess failure. Callers should
    treat exceptions as "fall back to whatever Whisper produced" —
    Parakeet is consensus, not a primary path.
    """
    gate = _can_run_consensus()
    if not gate.allowed:
        logger.info("parakeet: skipping (%s)", gate.reason)
        return []

    model_name = str(getattr(
        settings, "TACT_CONSENSUS_MODEL", "nvidia/parakeet-tdt-0.6b-v3",
    ))

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, dir="/tmp") as tmp:
        output_path = tmp.name

    try:
        cmd = [
            sys.executable, "-m", "backend.services.parakeet_worker",
            "--audio", audio_path,
            "--output", output_path,
            "--model", model_name,
        ]
        if language:
            cmd.extend(["--language", language])

        logger.info(
            "parakeet: starting subprocess model=%s free_vram=%dMB",
            model_name, gate.free_vram_mb,
        )

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ},
        )

        async def _wait_with_cancel():
            while True:
                if cancel_check is not None:
                    try:
                        cancel_check()
                    except Exception:
                        try:
                            proc.terminate()
                            await asyncio.wait_for(proc.wait(), timeout=3.0)
                        except (asyncio.TimeoutError, ProcessLookupError):
                            try:
                                proc.kill()
                            except ProcessLookupError:
                                pass
                        raise
                if proc.returncode is not None:
                    return
                try:
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
                    return
                except asyncio.TimeoutError:
                    continue

        try:
            await asyncio.wait_for(_wait_with_cancel(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            raise RuntimeError(
                f"parakeet subprocess timed out after {timeout_sec}s"
            )

        stderr_bytes = await proc.stderr.read() if proc.stderr else b""
        if proc.returncode != 0:
            tail = stderr_bytes.decode(errors="replace")[-500:]
            raise RuntimeError(
                f"parakeet subprocess exit {proc.returncode}: {tail}"
            )

        with open(output_path, "r") as f:
            raw = json.load(f)
        if raw.get("status") == "error":
            raise RuntimeError(f"parakeet worker error: {raw.get('error')}")

        segments: list[TranscriptSegment] = []
        for seg in raw.get("segments") or []:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            words = None
            if seg.get("words"):
                words = [
                    WordTimestamp(
                        start=float(w["start"]),
                        end=float(w["end"]),
                        word=w["word"],
                    )
                    for w in seg["words"]
                ]
            segments.append(TranscriptSegment(
                start=round(float(seg["start"]), 3),
                end=round(float(seg["end"]), 3),
                text=text,
                speaker="Speaker 1",
                words=words,
                confidence=seg.get("confidence"),
                avg_logprob=seg.get("avg_logprob"),
                no_speech_prob=seg.get("no_speech_prob"),
            ))
        logger.info(
            "parakeet: %d segments, CUDA released on exit", len(segments),
        )
        return segments

    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass


def transcribe_with_parakeet_subprocess_sync(
    audio_path: str,
    *,
    language: str = "",
    timeout_sec: float = 1800.0,
) -> list[TranscriptSegment]:
    """Sync entry point for the escalation ladder (which is sync).

    Wraps ``transcribe_with_parakeet_subprocess`` in a small event
    loop. The ladder's per-rung budget is enforced via
    ``timeout_sec``.
    """
    try:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                transcribe_with_parakeet_subprocess(
                    audio_path,
                    language=language,
                    timeout_sec=timeout_sec,
                )
            )
        finally:
            loop.close()
    except Exception as e:
        logger.info("parakeet: sync wrapper failed (%s)", e)
        return []
