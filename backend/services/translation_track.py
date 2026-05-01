"""TACT Phase 5 — translation track.

After the transcription ledger reaches ``coverage_ratio == 1.000``,
the translation track:

  1. Builds a paired translation ledger anchored to source-language
     span boundaries via ``CoverageLedger.to_translation_ledger``.
  2. For each source ``covered_speech`` span, runs translation.
  3. Writes the translated text into the target-language ledger with
     the same time bounds.
  4. Optionally reconciles N candidate translations via semantic
     similarity (LaBSE).

Forced alignment (Rung 4 of the transcription ladder) is **not**
used in the translation track — phonemes don't transfer cross-
language. Event spans (``[music]`` / ``[silence]`` etc.) pass through
unchanged in ``to_translation_ledger``; this module only fills in the
``covered_speech`` spans.

The translation backends are pluggable. Whisper's ``task=translate``
is the default (English target only); NLLB-200 / Seamless-M4T are
opt-in via ``TACT_TRANSLATION_BACKEND``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

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
class TranslationCandidate:
    """One model's translation of one source span."""
    backend_name: str
    text: str
    confidence: float = 0.0
    error: Optional[str] = None


@dataclass
class ReconciledTranslation:
    """Result of reconciling N candidate translations against the source.

    ``status`` is one of ``"agreed"``, ``"soft_agreement"``,
    ``"contested"``, ``"single"``, ``"failed"``.
    """
    text: str
    confidence: float
    sources: list[str] = field(default_factory=list)
    status: str = "single"
    similarity_to_source: float = 0.0


# ──────────────────────────────────────────────────────────────────────────
# Semantic reconciler
# ──────────────────────────────────────────────────────────────────────────

def _embed(texts: list[str]):
    """Lazy LaBSE embeddings via sentence-transformers.

    Returns ``None`` when the dependency isn't installed; callers
    treat None as "skip semantic reconciliation, pick the
    highest-confidence candidate".
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        return None
    try:
        # Cache the model on first use.
        global _labse_model
        if "_labse_model" not in globals() or _labse_model is None:  # type: ignore
            _labse_model = SentenceTransformer(  # type: ignore
                "sentence-transformers/LaBSE"
            )
        emb = _labse_model.encode(texts, normalize_embeddings=True)  # type: ignore
        return emb
    except Exception as e:
        logger.info("translation_track: LaBSE unavailable (%s)", e)
        return None


def reconcile_translations(
    candidates: list[TranslationCandidate],
    source_text: str,
    *,
    similarity_threshold_agree: float = 0.85,
    similarity_threshold_contest: float = 0.70,
) -> ReconciledTranslation:
    """Reconcile multiple translation candidates via semantic similarity.

    Algorithm:
      * Single non-error candidate → return it directly, status="single".
      * All non-error candidates within ``agree`` similarity to each
        other → return longest fluent variant, status="agreed".
      * Any pair below ``contest`` → status="contested",
        emit highest-confidence candidate.
      * Otherwise → soft agreement; pick semantic-best (closest to
        source).

    When LaBSE isn't installed, falls back to "highest-confidence
    candidate wins, status=single" — semantic reconciliation is a
    capability dep, not a hard requirement.
    """
    valid = [c for c in candidates if not c.error and c.text.strip()]
    if not valid:
        return ReconciledTranslation(
            text="", confidence=0.0,
            sources=[c.backend_name for c in candidates],
            status="failed",
        )
    if len(valid) == 1:
        return ReconciledTranslation(
            text=valid[0].text,
            confidence=valid[0].confidence,
            sources=[valid[0].backend_name],
            status="single",
        )

    # Try semantic reconciliation; fall back to confidence-pick.
    texts = [c.text for c in valid] + [source_text]
    emb = _embed(texts)
    if emb is None:
        best = max(valid, key=lambda c: c.confidence)
        return ReconciledTranslation(
            text=best.text,
            confidence=best.confidence,
            sources=[c.backend_name for c in valid],
            status="single",
        )

    import numpy as np  # safe — already a project dep
    cand_emb = emb[:-1]
    src_emb = emb[-1]
    # Pairwise candidate similarity matrix.
    sim = cand_emb @ cand_emb.T
    n = len(valid)
    min_pair = 1.0
    max_pair = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            v = float(sim[i][j])
            if v < min_pair:
                min_pair = v
            if v > max_pair:
                max_pair = v
    sim_to_src = cand_emb @ src_emb
    if min_pair >= similarity_threshold_agree:
        # All agree. Prefer longest non-empty variant (more fluent
        # readouts tend to be longer in target languages).
        best = max(valid, key=lambda c: len(c.text))
        idx = valid.index(best)
        return ReconciledTranslation(
            text=best.text,
            confidence=max(c.confidence for c in valid),
            sources=[c.backend_name for c in valid],
            status="agreed",
            similarity_to_source=float(sim_to_src[idx]),
        )
    if min_pair < similarity_threshold_contest:
        best = max(valid, key=lambda c: c.confidence)
        idx = valid.index(best)
        return ReconciledTranslation(
            text=best.text,
            confidence=best.confidence,
            sources=[c.backend_name for c in valid],
            status="contested",
            similarity_to_source=float(sim_to_src[idx]),
        )
    # Soft agreement → pick by similarity to source.
    best_idx = int(np.argmax(sim_to_src))
    best = valid[best_idx]
    return ReconciledTranslation(
        text=best.text,
        confidence=best.confidence,
        sources=[c.backend_name for c in valid],
        status="soft_agreement",
        similarity_to_source=float(sim_to_src[best_idx]),
    )


# ──────────────────────────────────────────────────────────────────────────
# Translation backends — pluggable
# ──────────────────────────────────────────────────────────────────────────

# A translator function takes (source_text, target_language) and
# returns a TranslationCandidate. Backends are looked up by name from
# this registry; each is wrapped with a try/except so a missing
# backend never crashes the track.
_BACKENDS: dict[str, Callable[[str, str], TranslationCandidate]] = {}


def register_backend(name: str):
    """Decorator-style registration. Used by tests to inject mock
    backends; production backends register themselves on import."""
    def _wrap(fn: Callable[[str, str], TranslationCandidate]):
        _BACKENDS[name] = fn
        return fn
    return _wrap


def _whisper_translate_passthrough(
    source_text: str, target_language: str,
) -> TranslationCandidate:
    """Fallback when the source ledger already came from a
    ``task=translate`` Whisper run. The 'translation' is just the
    transcription text; we surface it as the Whisper backend so the
    coverage report is internally consistent."""
    return TranslationCandidate(
        backend_name="whisper_passthrough",
        text=source_text,
        confidence=0.5,
    )


_BACKENDS["whisper_passthrough"] = _whisper_translate_passthrough


def list_backends() -> list[str]:
    return sorted(_BACKENDS.keys())


# ──────────────────────────────────────────────────────────────────────────
# Track runner
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class TranslationTrackStats:
    target_language: str = ""
    source_speech_spans: int = 0
    spans_translated: int = 0
    spans_failed: int = 0
    spans_reconciled: int = 0
    elapsed_sec: float = 0.0

    def as_dict(self) -> dict:
        return {
            "target_language": self.target_language,
            "source_speech_spans": self.source_speech_spans,
            "spans_translated": self.spans_translated,
            "spans_failed": self.spans_failed,
            "spans_reconciled": self.spans_reconciled,
            "elapsed_sec": round(self.elapsed_sec, 3),
        }


def run_translation_track(
    source_ledger: CoverageLedger,
    *,
    target_language: str,
    backend: str = "whisper_passthrough",
    consensus_backends: Optional[list[str]] = None,
    similarity_threshold_agree: float = 0.85,
    similarity_threshold_contest: float = 0.70,
) -> tuple[CoverageLedger, TranslationTrackStats]:
    """Build the paired translation ledger and fill its speech spans.

    Returns ``(translation_ledger, stats)``. The translation ledger's
    coverage_ratio reaches 1.000 when every source covered_speech
    span has a non-empty translation; spans where every backend
    failed are marked ``[untranslatable]`` (Phase 5 §4.3 — analogous
    to Rung 6's ``[unintelligible]`` for transcription).
    """
    import time as _time
    started = _time.monotonic()
    stats = TranslationTrackStats(target_language=target_language)

    target = source_ledger.to_translation_ledger(target_language)

    # The track operates on covered_speech spans only.
    # Other spans (events, silence) were already mirrored unchanged
    # by to_translation_ledger.
    backends_to_use: list[str] = []
    if backend in _BACKENDS:
        backends_to_use.append(backend)
    if consensus_backends:
        for b in consensus_backends:
            if b in _BACKENDS and b not in backends_to_use:
                backends_to_use.append(b)
    if not backends_to_use:
        logger.warning(
            "translation_track: no backends available — leaving "
            "translation ledger empty",
        )
        stats.elapsed_sec = round(_time.monotonic() - started, 3)
        return target, stats

    for s in source_ledger.spans:
        if s.status not in COVERED_STATUSES:
            continue
        if s.content_type in ("event", "silence", "unintelligible"):
            continue
        source_text = (s.content or "").strip()
        if not source_text:
            continue
        stats.source_speech_spans += 1

        # Run each backend; collect candidates.
        candidates: list[TranslationCandidate] = []
        for b_name in backends_to_use:
            fn = _BACKENDS.get(b_name)
            if fn is None:
                continue
            try:
                candidates.append(fn(source_text, target_language))
            except Exception as e:
                candidates.append(TranslationCandidate(
                    backend_name=b_name, text="", error=str(e),
                ))

        if not candidates or all(c.error or not c.text.strip() for c in candidates):
            # All backends failed — emit [untranslatable] so the
            # translation coverage invariant still holds.
            target.claim(LedgerSpan(
                start_ms=s.start_ms,
                end_ms=s.end_ms,
                status="covered_event",
                content="[untranslatable]",
                content_type="unintelligible",
                source_pass="translation_failed",
                # Tiny but non-zero so it supersedes the
                # awaiting_translation placeholder (which is at 0.0).
                # Still effectively "no useful content" — the
                # confidence reported in the report dict reflects
                # this.
                confidence=0.01,
                speaker=s.speaker,
                target_language=target_language,
                flags=["untranslatable"],
            ))
            stats.spans_failed += 1
            continue

        if len(candidates) == 1:
            cand = candidates[0]
            target.claim(LedgerSpan(
                start_ms=s.start_ms,
                end_ms=s.end_ms,
                status="covered_speech",
                content=cand.text,
                content_type=s.content_type,
                source_pass=f"translation:{cand.backend_name}",
                confidence=max(cand.confidence, 0.5),
                speaker=s.speaker,
                target_language=target_language,
            ))
            stats.spans_translated += 1
            continue

        rec = reconcile_translations(
            candidates, source_text,
            similarity_threshold_agree=similarity_threshold_agree,
            similarity_threshold_contest=similarity_threshold_contest,
        )
        if rec.status == "failed" or not rec.text.strip():
            target.claim(LedgerSpan(
                start_ms=s.start_ms,
                end_ms=s.end_ms,
                status="covered_event",
                content="[untranslatable]",
                content_type="unintelligible",
                source_pass="translation_failed",
                # Tiny but non-zero so it supersedes the
                # awaiting_translation placeholder (which is at 0.0).
                # Still effectively "no useful content" — the
                # confidence reported in the report dict reflects
                # this.
                confidence=0.01,
                speaker=s.speaker,
                target_language=target_language,
                flags=["untranslatable"],
            ))
            stats.spans_failed += 1
            continue
        target.claim(LedgerSpan(
            start_ms=s.start_ms,
            end_ms=s.end_ms,
            status="covered_speech",
            content=rec.text,
            content_type=s.content_type,
            source_pass="translation:" + ",".join(rec.sources),
            confidence=max(rec.confidence, 0.5),
            speaker=s.speaker,
            target_language=target_language,
            flags=[f"reconcile:{rec.status}"],
        ))
        stats.spans_translated += 1
        if rec.status in ("agreed", "soft_agreement", "contested"):
            stats.spans_reconciled += 1

    stats.elapsed_sec = round(_time.monotonic() - started, 3)
    logger.info(
        "translation_track: %s spans=%d translated=%d failed=%d in %.2fs",
        target_language, stats.source_speech_spans, stats.spans_translated,
        stats.spans_failed, stats.elapsed_sec,
    )
    return target, stats
