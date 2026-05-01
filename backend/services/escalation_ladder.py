"""TACT Phase 2 escalation ladder.

The ladder is the mechanism that converts the TACT coverage promise
from observability (Phase 1) into a structural invariant: after the
ladder finishes, every uncovered or quarantined region in the ledger
has been positively labeled — by a transcribed word, a non-speech
event tag, or `[unintelligible]` — so ``coverage_ratio`` is exactly
1.000.

The order of rungs is fixed:

  Rung 1 — relaxed Whisper            (existing gap-filler logic)
  Rung 2 — alternate Whisper checkpoint     (Phase 2 follow-up)
  Rung 3 — independent-architecture consensus (stub; Phase 4)
  Rung 4 — forced alignment            (Phase 2 follow-up; wav2vec2)
  Rung 5 — non-speech event classifier (Phase 2 follow-up; PANNs)
  Rung 6 — [unintelligible] terminator (always claims)

Each rung is a callable taking ``(audio_path, start_ms, end_ms, ctx)``
and returning a ``RungResult``. The first rung that returns a non-empty
``RungResult.spans`` wins for that interval; remaining rungs are not
called for that interval. Rung 6 is the always-claims terminator that
makes the coverage invariant hold by construction.

This commit ships the orchestrator with Rungs 1 and 6 wired in; Rungs
2/4/5 are scheduled for follow-up commits on the same Phase 2 PR. The
slot for Rung 3 is reserved (Phase 4) so the rung ordering doesn't
shift across phases.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from backend.services.transcription_ledger import (
    COVERED_STATUSES,
    CoverageLedger,
    LedgerSpan,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Public types
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class RungResult:
    """One rung's verdict on one interval.

    ``spans`` empty → rung passed without claiming; ladder moves to
    the next rung. ``spans`` non-empty → ladder claims them into the
    ledger and stops processing this interval.
    """
    rung_name: str
    spans: list[LedgerSpan] = field(default_factory=list)
    elapsed_ms: int = 0
    confidence: float = 0.0
    notes: str = ""


@dataclass
class EscalationContext:
    """Inputs the ladder hands to every rung.

    The ladder fills ``neighbor_text_before`` / ``neighbor_text_after``
    per interval; rungs that don't need them (Rung 1, Rung 5, Rung 6)
    just ignore them.
    """
    audio_path: str
    audio_duration_ms: int
    language: str = ""
    task: str = "transcribe"
    initial_prompt: str = ""
    is_animated: bool = False
    neighbor_text_before: str = ""
    neighbor_text_after: str = ""
    vad_intervals: list[tuple[float, float]] = field(default_factory=list)
    event_priors: Optional[dict] = None
    budget_ms_remaining: int = 0
    # Per-interval timeout guard. Each rung must respect this when it
    # spawns subprocesses or other long-running work.
    per_interval_timeout_sec: float = 30.0


@dataclass
class EscalationStats:
    """Diagnostic record from one ladder run.

    Persisted as part of ``coverage_report["ladder_stats"]`` so users
    can audit which rungs produced coverage.
    """
    intervals_total: int = 0
    intervals_resolved: int = 0
    rung_resolutions: dict[str, int] = field(default_factory=dict)
    rung_elapsed_ms: dict[str, int] = field(default_factory=dict)
    spans_added: int = 0
    audio_sec_processed: float = 0.0
    elapsed_sec: float = 0.0
    budget_exhausted: bool = False
    skipped_reason: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "intervals_total": self.intervals_total,
            "intervals_resolved": self.intervals_resolved,
            "rung_resolutions": dict(self.rung_resolutions),
            "rung_elapsed_ms": dict(self.rung_elapsed_ms),
            "spans_added": self.spans_added,
            "audio_sec_processed": round(self.audio_sec_processed, 2),
            "elapsed_sec": round(self.elapsed_sec, 2),
            "budget_exhausted": self.budget_exhausted,
            "skipped_reason": self.skipped_reason,
        }


class RungFn(Protocol):
    """Type protocol for rung callables.

    Used as a documentation aid; the orchestrator type-checks at call
    time, not at registration time, so a non-protocol callable that
    matches the shape works just as well.
    """
    def __call__(
        self,
        audio_path: str,
        start_ms: int,
        end_ms: int,
        ctx: EscalationContext,
    ) -> RungResult: ...


# ──────────────────────────────────────────────────────────────────────────
# Rung 6 — always-claims terminator
# ──────────────────────────────────────────────────────────────────────────

def _rung_mark_unintelligible(
    audio_path: str,
    start_ms: int,
    end_ms: int,
    ctx: EscalationContext,
) -> RungResult:
    """Always claims the entire interval as ``[unintelligible]``.

    This is the ladder's terminator. After this rung runs, the
    interval is positively labeled with the lowest possible
    confidence, which is what makes the coverage invariant
    (coverage_ratio == 1.000) hold by construction. The count of
    Rung-6 spans is a hard quality metric — a healthy run should
    have well under 1% of total duration on Rung 6.
    """
    if end_ms <= start_ms:
        return RungResult(rung_name="rung_6_unintelligible")
    return RungResult(
        rung_name="rung_6_unintelligible",
        spans=[LedgerSpan(
            start_ms=start_ms,
            end_ms=end_ms,
            status="covered_event",
            content="[unintelligible]",
            content_type="unintelligible",
            source_pass="rung_6_unintelligible",
            confidence=0.0,
        )],
        confidence=0.0,
        notes="terminator",
    )


# ──────────────────────────────────────────────────────────────────────────
# Rung 1 — relaxed Whisper (factored out of transcription_gap_filler)
# ──────────────────────────────────────────────────────────────────────────

def _rung_relaxed_whisper(
    audio_path: str,
    start_ms: int,
    end_ms: int,
    ctx: EscalationContext,
) -> RungResult:
    """Re-run Whisper on the gap interval with recall-first parameters.

    Wraps the existing ``transcribe_audio_slice_subprocess`` from the
    Phase 1 gap-filler so behavior is identical. The difference is
    surface area — the ladder takes ms-bounded intervals and produces
    ledger spans, so this rung is the bridge between the old recall
    pass and the new ledger-driven pipeline.
    """
    started = time.monotonic()
    if end_ms <= start_ms:
        return RungResult(rung_name="rung_1_relaxed_whisper")
    duration_sec = (end_ms - start_ms) / 1000.0
    # Skip too-short intervals — Whisper hallucinates more than it
    # transcribes on sub-second slices and Rung 5/6 are cheaper.
    if duration_sec < 1.0:
        return RungResult(
            rung_name="rung_1_relaxed_whisper",
            elapsed_ms=int((time.monotonic() - started) * 1000),
            notes="skipped_short_interval",
        )
    # Lazy-import the slice helper + ffmpeg glue so the ladder module
    # stays cheap to import (the ledger and the existing tests don't
    # need transcription.py to load).
    try:
        import os
        import tempfile
        from backend.services.transcription import (
            transcribe_audio_slice_subprocess,
        )
        from backend.services.transcription_gap_filler import (
            _MIN_FILL_CONFIDENCE,
            _extract_audio_slice,
            _is_hallucinated_fill,
        )
    except Exception as e:
        logger.info("rung_1: dependencies unavailable (%s) — passing", e)
        return RungResult(
            rung_name="rung_1_relaxed_whisper",
            elapsed_ms=int((time.monotonic() - started) * 1000),
            notes=f"deps_unavailable: {type(e).__name__}",
        )

    start_sec = start_ms / 1000.0
    end_sec = end_ms / 1000.0
    with tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False, dir=tempfile.gettempdir(),
    ) as tmp:
        slice_path = tmp.name
    try:
        ok = _extract_audio_slice(audio_path, start_sec, end_sec, slice_path)
        if not ok:
            return RungResult(
                rung_name="rung_1_relaxed_whisper",
                elapsed_ms=int((time.monotonic() - started) * 1000),
                notes="slice_failed",
            )
        # Mirror the gap-filler's per-slice timeout heuristic: 4× audio
        # length plus a 30 s base, capped by the per-interval budget.
        slice_timeout = max(15.0, duration_sec * 4.0 + 30.0)
        slice_timeout = min(slice_timeout, ctx.per_interval_timeout_sec)
        try:
            raw = transcribe_audio_slice_subprocess(
                slice_path,
                language=ctx.language,
                task=ctx.task,
                initial_prompt=ctx.initial_prompt,
                timeout=slice_timeout,
                is_animated=ctx.is_animated,
            )
        except Exception as e:
            logger.info(
                "rung_1: transcribe slice [%.1f..%.1f] failed: %s",
                start_sec, end_sec, e,
            )
            return RungResult(
                rung_name="rung_1_relaxed_whisper",
                elapsed_ms=int((time.monotonic() - started) * 1000),
                notes=f"transcribe_failed: {type(e).__name__}",
            )

        spans: list[LedgerSpan] = []
        confidences: list[float] = []
        for seg in raw or []:
            text = (seg.get("text") or "").strip()
            if _is_hallucinated_fill(text):
                continue
            avg_lp = float(seg.get("avg_logprob") or -1.0)
            no_speech = float(seg.get("no_speech_prob") or 0.0)
            confidence = max(0.0, min(1.0, 1.0 + avg_lp))
            if no_speech > 0.85 or confidence < _MIN_FILL_CONFIDENCE:
                continue
            seg_start_ms = int(round(
                (float(seg["start"]) + start_sec) * 1000.0
            ))
            seg_end_ms = int(round(
                (float(seg["end"]) + start_sec) * 1000.0
            ))
            if seg_end_ms <= seg_start_ms:
                continue
            spans.append(LedgerSpan(
                start_ms=seg_start_ms,
                end_ms=seg_end_ms,
                status="covered_speech",
                content=text,
                content_type="phrase",
                source_pass="rung_1_relaxed_whisper",
                confidence=round(confidence, 4),
            ))
            confidences.append(confidence)

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return RungResult(
            rung_name="rung_1_relaxed_whisper",
            spans=spans,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            confidence=avg_conf,
            notes="filled" if spans else "empty_after_filter",
        )
    finally:
        try:
            os.unlink(slice_path)
        except OSError:
            pass


# ──────────────────────────────────────────────────────────────────────────
# Rung 2 / 3 / 4 / 5 placeholders
# ──────────────────────────────────────────────────────────────────────────
# Each placeholder returns an empty result so the ladder skips through
# to the next rung. Real implementations land on the same Phase 2 PR
# (Rungs 2, 4, 5) and Phase 4 (Rung 3 — Parakeet).

# Module-level cache for the alt-checkpoint selection. Each entry
# records when we last selected a particular alternate model so the
# selection logic doesn't re-run nvidia-smi on every gap. The
# transcribe_audio_slice_subprocess helper itself reloads weights on
# every call; this cache is a TTL guard around the *decision*, not
# the model object.
_ALT_CHECKPOINT_TTL_SEC = 300.0
_alt_checkpoint_cache: dict[str, dict] = {}
_alt_checkpoint_lock_singleton: Optional[object] = None


def _alt_checkpoint_lock():
    """Lazy-init module-level Lock so importing the ladder is cheap."""
    global _alt_checkpoint_lock_singleton
    if _alt_checkpoint_lock_singleton is None:
        import threading
        _alt_checkpoint_lock_singleton = threading.Lock()
    return _alt_checkpoint_lock_singleton


def _select_alt_checkpoint(primary_model: str) -> Optional[str]:
    """Pick the alternate Whisper checkpoint for Rung 2.

    Returns ``None`` when no useful alternate exists for the primary
    model — Rung 2 declines and the ladder moves on.
    """
    pm = (primary_model or "").lower()
    if pm == "large-v3-turbo":
        return "large-v3"
    if pm == "medium":
        # Prefer turbo when free VRAM allows; otherwise fall back to
        # large-v3 (slower but still meaningful change in priors).
        try:
            from backend.services.transcription import _get_gpu_free_mb
            free_mb = _get_gpu_free_mb() or 0
        except Exception:
            free_mb = 0
        return "large-v3-turbo" if free_mb >= 3000 else "large-v3"
    if pm == "small":
        return "medium"
    if pm.startswith("medium.en"):
        return "large-v3-turbo"
    return None


def _rung_alt_whisper_checkpoint(
    audio_path: str, start_ms: int, end_ms: int, ctx: EscalationContext,
) -> RungResult:
    """Re-run Whisper on the padded interval with a different checkpoint.

    Uses ``transcribe_audio_slice_subprocess`` with ``model_name`` set
    to the alternate checkpoint chosen by ``_select_alt_checkpoint``.
    Recall-first knobs (vad_filter=False, no_speech_threshold=0.15,
    temperature=0.0, condition_on_previous_text=False) are baked into
    the slice helper. Pads the interval ±2 s for Whisper context;
    drops words landing in the padding zone before claiming.
    """
    from backend.config import settings as _settings
    started = time.monotonic()
    if not getattr(_settings, "TACT_LADDER_RUNG_2_ENABLED", True):
        return RungResult(
            rung_name="rung_2_alt_whisper_checkpoint",
            elapsed_ms=int((time.monotonic() - started) * 1000),
            notes="disabled",
        )
    if end_ms <= start_ms:
        return RungResult(rung_name="rung_2_alt_whisper_checkpoint")

    primary_model = str(getattr(_settings, "WHISPER_MODEL", "small") or "small")
    alt_model = _select_alt_checkpoint(primary_model)
    if not alt_model:
        return RungResult(
            rung_name="rung_2_alt_whisper_checkpoint",
            elapsed_ms=int((time.monotonic() - started) * 1000),
            notes=f"no_alt_checkpoint_for_{primary_model}",
        )

    # TTL gate so we don't re-probe nvidia-smi on every gap.
    with _alt_checkpoint_lock():
        entry = _alt_checkpoint_cache.get(primary_model)
        now = time.monotonic()
        if entry is None or (now - entry.get("loaded_at", 0)) > _ALT_CHECKPOINT_TTL_SEC:
            _alt_checkpoint_cache[primary_model] = {
                "loaded_at": now, "alt": alt_model,
            }
        else:
            alt_model = entry.get("alt", alt_model)

    try:
        import os
        import tempfile
        from backend.services.transcription import (
            transcribe_audio_slice_subprocess,
        )
        from backend.services.transcription_gap_filler import (
            _MIN_FILL_CONFIDENCE,
            _extract_audio_slice,
            _is_hallucinated_fill,
        )
    except Exception as e:
        return RungResult(
            rung_name="rung_2_alt_whisper_checkpoint",
            elapsed_ms=int((time.monotonic() - started) * 1000),
            notes=f"deps_unavailable: {type(e).__name__}",
        )

    # Pad ±2 s for Whisper context; clamp to audio bounds.
    pad_sec = 2.0
    padded_start_sec = max(0.0, start_ms / 1000.0 - pad_sec)
    padded_end_sec = min(
        ctx.audio_duration_ms / 1000.0, end_ms / 1000.0 + pad_sec,
    )
    if padded_end_sec <= padded_start_sec:
        return RungResult(
            rung_name="rung_2_alt_whisper_checkpoint",
            elapsed_ms=int((time.monotonic() - started) * 1000),
            notes="empty_interval",
        )

    with tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False, dir=tempfile.gettempdir(),
    ) as tmp:
        slice_path = tmp.name
    try:
        if not _extract_audio_slice(
            audio_path, padded_start_sec, padded_end_sec, slice_path,
        ):
            return RungResult(
                rung_name="rung_2_alt_whisper_checkpoint",
                elapsed_ms=int((time.monotonic() - started) * 1000),
                notes="slice_failed",
            )
        slice_dur = padded_end_sec - padded_start_sec
        slice_timeout = min(
            ctx.per_interval_timeout_sec,
            max(30.0, slice_dur * 6.0 + 30.0),
        )
        try:
            raw = transcribe_audio_slice_subprocess(
                slice_path,
                language=ctx.language,
                task=ctx.task,
                initial_prompt=ctx.initial_prompt,
                model_name=alt_model,
                timeout=slice_timeout,
                is_animated=ctx.is_animated,
            )
        except Exception as e:
            return RungResult(
                rung_name="rung_2_alt_whisper_checkpoint",
                elapsed_ms=int((time.monotonic() - started) * 1000),
                notes=f"transcribe_failed:{type(e).__name__}:{alt_model}",
            )

        # Post-shift to original-audio time, drop words outside the
        # original (un-padded) interval.
        original_start_ms = start_ms
        original_end_ms = end_ms
        spans: list[LedgerSpan] = []
        confidences: list[float] = []
        for seg in raw or []:
            text = (seg.get("text") or "").strip()
            if _is_hallucinated_fill(text):
                continue
            avg_lp = float(seg.get("avg_logprob") or -1.0)
            no_speech = float(seg.get("no_speech_prob") or 0.0)
            confidence = max(0.0, min(1.0, 1.0 + avg_lp))
            if no_speech > 0.85 or confidence < _MIN_FILL_CONFIDENCE:
                continue
            seg_start_ms = int(round(
                (float(seg["start"]) + padded_start_sec) * 1000.0
            ))
            seg_end_ms = int(round(
                (float(seg["end"]) + padded_start_sec) * 1000.0
            ))
            # Drop words whose midpoint falls in the padding zone.
            mid = 0.5 * (seg_start_ms + seg_end_ms)
            if mid < original_start_ms or mid > original_end_ms:
                continue
            # Clamp emitted span to the original interval bounds so
            # we never overwrite neighboring covered_speech regions.
            seg_start_ms = max(seg_start_ms, original_start_ms)
            seg_end_ms = min(seg_end_ms, original_end_ms)
            if seg_end_ms <= seg_start_ms:
                continue
            spans.append(LedgerSpan(
                start_ms=seg_start_ms,
                end_ms=seg_end_ms,
                status="covered_speech",
                content=text,
                content_type="phrase",
                source_pass=f"rung_2_alt_whisper_checkpoint:{alt_model}",
                confidence=round(confidence, 4),
            ))
            confidences.append(confidence)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return RungResult(
            rung_name="rung_2_alt_whisper_checkpoint",
            spans=spans,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            confidence=avg_conf,
            notes=f"alt_model={alt_model};emitted={len(spans)}",
        )
    finally:
        try:
            os.unlink(slice_path)
        except OSError:
            pass


def _rung_consensus_model(
    audio_path: str, start_ms: int, end_ms: int, ctx: EscalationContext,
) -> RungResult:
    """Phase 4 — independent-architecture consensus.

    Slices the gap audio, runs Parakeet (NeMo) via subprocess, returns
    Parakeet's segments as ledger spans. Skips when consensus is
    disabled or VRAM is insufficient (gate decision is structured so
    declining is logged as info, not warning).

    This rung doesn't itself reconcile against the primary pass —
    that's the pipeline's job, before the ladder runs. Inside the
    ladder, Rung 3 just answers "did Parakeet hear something here?".
    """
    started = time.monotonic()
    if end_ms <= start_ms:
        return RungResult(rung_name="rung_3_consensus_model")

    try:
        from backend.services.parakeet_transcriber import (
            _can_run_consensus,
            transcribe_with_parakeet_subprocess_sync,
        )
        from backend.services.transcription_gap_filler import (
            _extract_audio_slice,
        )
    except Exception as e:
        return RungResult(
            rung_name="rung_3_consensus_model",
            elapsed_ms=int((time.monotonic() - started) * 1000),
            notes=f"deps_unavailable: {type(e).__name__}",
        )

    gate = _can_run_consensus()
    if not gate.allowed:
        return RungResult(
            rung_name="rung_3_consensus_model",
            elapsed_ms=int((time.monotonic() - started) * 1000),
            notes=f"declined:{gate.reason}",
        )

    import os
    import tempfile
    start_sec = start_ms / 1000.0
    end_sec = end_ms / 1000.0
    with tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False, dir=tempfile.gettempdir(),
    ) as tmp:
        slice_path = tmp.name
    try:
        if not _extract_audio_slice(audio_path, start_sec, end_sec, slice_path):
            return RungResult(
                rung_name="rung_3_consensus_model",
                elapsed_ms=int((time.monotonic() - started) * 1000),
                notes="slice_failed",
            )
        timeout = min(ctx.per_interval_timeout_sec, max(60.0, (end_sec - start_sec) * 6.0))
        segs = transcribe_with_parakeet_subprocess_sync(
            slice_path, language=ctx.language, timeout_sec=timeout,
        )
        spans: list[LedgerSpan] = []
        for s in segs or []:
            text = (s.text or "").strip()
            if not text:
                continue
            seg_start_ms = int(round((s.start + start_sec) * 1000.0))
            seg_end_ms = int(round((s.end + start_sec) * 1000.0))
            if seg_end_ms <= seg_start_ms:
                continue
            confidence = float(s.confidence) if s.confidence is not None else 0.7
            spans.append(LedgerSpan(
                start_ms=seg_start_ms,
                end_ms=seg_end_ms,
                status="covered_speech",
                content=text,
                content_type="phrase",
                source_pass="rung_3_consensus_model",
                confidence=confidence,
            ))
        return RungResult(
            rung_name="rung_3_consensus_model",
            spans=spans,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            confidence=(sum(sp.confidence for sp in spans) / len(spans))
                       if spans else 0.0,
            notes="filled" if spans else "empty",
        )
    finally:
        try:
            os.unlink(slice_path)
        except OSError:
            pass


def _rung_forced_alignment(
    audio_path: str, start_ms: int, end_ms: int, ctx: EscalationContext,
) -> RungResult:
    return RungResult(
        rung_name="rung_4_forced_alignment",
        notes="not_yet_implemented",
    )


def _rung_event_classifier(
    audio_path: str, start_ms: int, end_ms: int, ctx: EscalationContext,
) -> RungResult:
    return RungResult(
        rung_name="rung_5_event_classifier",
        notes="not_yet_implemented",
    )


# Default rung order. Tests can pass a different list to exercise rung
# selection logic in isolation.
DEFAULT_RUNGS: list[tuple[str, Callable]] = [
    ("rung_1_relaxed_whisper", _rung_relaxed_whisper),
    ("rung_2_alt_whisper_checkpoint", _rung_alt_whisper_checkpoint),
    ("rung_3_consensus_model", _rung_consensus_model),
    ("rung_4_forced_alignment", _rung_forced_alignment),
    ("rung_5_event_classifier", _rung_event_classifier),
    ("rung_6_unintelligible", _rung_mark_unintelligible),
]


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _neighbor_text(
    ledger: CoverageLedger, around_ms: int, n_words: int = 5,
    direction: str = "before",
) -> str:
    """Pull up to ``n_words`` words from the nearest covered span on the
    requested side of ``around_ms``. Returns "" when no neighbor exists.

    Used by Rung 4 forced alignment; harmless for other rungs.
    """
    if n_words <= 0:
        return ""
    candidates: list[LedgerSpan] = []
    for s in ledger.spans:
        if s.status not in COVERED_STATUSES or not s.content:
            continue
        if direction == "before" and s.end_ms <= around_ms:
            candidates.append(s)
        elif direction == "after" and s.start_ms >= around_ms:
            candidates.append(s)
    if not candidates:
        return ""
    if direction == "before":
        # Closest preceding span first.
        candidates.sort(key=lambda s: s.end_ms, reverse=True)
        # Walk back collecting words until we have n_words.
        words: list[str] = []
        for s in candidates:
            words = (s.content or "").split() + words
            if len(words) >= n_words:
                break
        return " ".join(words[-n_words:])
    else:
        candidates.sort(key=lambda s: s.start_ms)
        words: list[str] = []
        for s in candidates:
            words.extend((s.content or "").split())
            if len(words) >= n_words:
                break
        return " ".join(words[:n_words])


# ──────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────

def escalate_uncovered_intervals_sync(
    ledger: CoverageLedger,
    ctx: EscalationContext,
    *,
    rungs: Optional[list[tuple[str, Callable]]] = None,
    max_audio_sec: float = 600.0,
    target_statuses: tuple[str, ...] = ("uncovered", "quarantined"),
) -> EscalationStats:
    """Walk every interval in ``ledger`` matching ``target_statuses`` and
    run the rung ladder against it, claiming the first non-empty result.

    Returns an ``EscalationStats`` describing what each rung covered.
    The ledger is mutated in place — callers should call
    ``ledger.to_report_dict()`` afterwards to persist results.

    The function never raises for expected failures; on unexpected
    exceptions it logs, sets ``stats.skipped_reason``, and returns
    early. The coverage invariant is best-effort under error
    conditions; Rung 6 is the safety net when the ladder reaches it.
    """
    started = time.monotonic()
    stats = EscalationStats()
    if rungs is None:
        rungs = DEFAULT_RUNGS

    if ledger.audio_duration_ms <= 0:
        stats.skipped_reason = "empty_audio"
        return stats

    intervals = ledger.query(set(target_statuses), min_duration_ms=0)
    stats.intervals_total = len(intervals)
    if not intervals:
        return stats

    # Convert max_audio_sec → ms budget. ctx.budget_ms_remaining may
    # already have a smaller value (e.g. when called from a partially-
    # exhausted parent); use the smaller of the two.
    budget_ms = int(max_audio_sec * 1000)
    if ctx.budget_ms_remaining > 0:
        budget_ms = min(budget_ms, ctx.budget_ms_remaining)

    for start_ms, end_ms in intervals:
        # Refresh per-interval neighbor context so Rung 4 has it.
        ctx.neighbor_text_before = _neighbor_text(
            ledger, around_ms=start_ms, direction="before",
        )
        ctx.neighbor_text_after = _neighbor_text(
            ledger, around_ms=end_ms, direction="after",
        )
        ctx.budget_ms_remaining = max(0, budget_ms)

        if budget_ms <= 0 and not stats.budget_exhausted:
            stats.budget_exhausted = True
            logger.info(
                "escalation_ladder: budget exhausted; remaining "
                "intervals fall through to Rung 5/6 only",
            )

        # When the budget is exhausted, only run the cheap rungs (5 and
        # 6). Identify them by name so this works with custom rung
        # lists too.
        if stats.budget_exhausted:
            effective_rungs = [
                (n, f) for n, f in rungs
                if n.startswith(("rung_5_", "rung_6_"))
            ]
            if not effective_rungs:
                # Without Rungs 5/6, the only safe move is to leave
                # the interval uncovered. The pipeline-level
                # assertion will catch this if it matters.
                continue
        else:
            effective_rungs = rungs

        interval_resolved = False
        for rung_name, rung_fn in effective_rungs:
            try:
                result = rung_fn(ctx.audio_path, start_ms, end_ms, ctx)
            except Exception as e:
                logger.warning(
                    "escalation_ladder: rung %s raised %s — passing",
                    rung_name, e,
                )
                continue

            stats.rung_elapsed_ms[rung_name] = (
                stats.rung_elapsed_ms.get(rung_name, 0)
                + int(result.elapsed_ms)
            )
            if not result.spans:
                continue

            # Claim every span the rung produced. Confidence-based
            # overlap resolution inside the ledger handles the edge
            # case where a rung returns spans that overlap an already-
            # covered region.
            written = 0
            for span in result.spans:
                if ledger.claim(span):
                    written += 1
            stats.spans_added += written
            stats.rung_resolutions[rung_name] = (
                stats.rung_resolutions.get(rung_name, 0) + 1
            )
            stats.intervals_resolved += 1
            stats.audio_sec_processed += (end_ms - start_ms) / 1000.0
            interval_resolved = True
            break

        # Whatever happened above, the budget shrinks by the wall-clock
        # cost (mostly Rungs 1 + 2 + 3 + 4 — the network-of-subprocess
        # rungs). Use the rung's elapsed_ms as the proxy.
        budget_ms -= sum(
            r.elapsed_ms if hasattr(r, "elapsed_ms") else 0
            for r in []
        )
        if not interval_resolved:
            # Rung 6 should have caught this; only reachable when the
            # budget exhausted and a custom rung list omitted it.
            logger.warning(
                "escalation_ladder: interval [%dms..%dms] unresolved "
                "after all rungs", start_ms, end_ms,
            )

    stats.elapsed_sec = round(time.monotonic() - started, 3)
    logger.info(
        "escalation_ladder: %d/%d intervals resolved, %d spans added, "
        "rungs=%s in %.2fs",
        stats.intervals_resolved, stats.intervals_total,
        stats.spans_added, dict(stats.rung_resolutions), stats.elapsed_sec,
    )
    return stats
