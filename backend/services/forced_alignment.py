"""TACT Phase 2 Rung 4 — wav2vec2 forced alignment.

Forced alignment is the technique of taking *known* text and aligning
it to *audio* — producing per-word timestamps and per-word
confidence. The TACT use case for this is narrow: when Rungs 1–3
fail to transcribe a gap interval, we ask "do the words from the
neighboring covered spans actually appear inside this gap?". For
chunk-edge truncation (where a 15-word sentence got chopped at
word 10 by Whisper's 30-s window), the last 5 words *are* in the gap
and the aligner finds them. For genuinely non-lexical gaps (music,
silence), alignment confidence stays low and the rung falls through.

The aligner is wav2vec2 via torchaudio's bundled pipelines:
  * English (and English-only): WAV2VEC2_ASR_BASE_960H
  * Multilingual: MMS_FA (Meta's Massively Multilingual Speech)

torchaudio.functional.forced_align is the CTC-based alignment helper
that does the actual work; the pipeline objects above provide the
acoustic model + tokenizer.

This module is import-cheap (torch is pulled in lazily inside
``align_text_to_audio``) so the ladder's import doesn't drag torch
in unconditionally.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    """Per-word alignments + a structured reason for empty results.

    ``words`` is a list of ``(start_sec, end_sec, text, confidence)``
    tuples in the *original audio's* frame of reference (the caller
    passes start_sec/end_sec for the gap; this module post-shifts
    timestamps so callers don't have to).

    ``reason`` is one of:
      * ``"ok"`` — alignment produced at least one word above
        the confidence threshold.
      * ``"no_alignment"`` — model loaded, ran, but no word
        cleared the threshold.
      * ``"model_unavailable"`` — torchaudio / weights missing.
      * ``"language_unsupported"`` — non-English language and
        MMS_FA unavailable or doesn't support the code.
      * ``"interval_too_long"`` — gap exceeds the 60-s cap.
      * ``"empty_text"`` — caller passed no neighbor text.
      * ``"slice_failed"`` — ffmpeg couldn't extract the slice.
    """
    words: list[tuple[float, float, str, float]] = field(default_factory=list)
    reason: str = "no_alignment"


# Cap on alignment input length. Beyond this the OOM risk on a 4 GB
# GPU dominates; gaps longer than 60 s are also rare and Rung 5
# (event classifier) handles long non-speech regions cheaper anyway.
_MAX_ALIGN_DURATION_SEC = 60.0

# Module-level model cache. Loading the bundle takes ~2 s; we keep
# one per language family across calls. Lazy-init guarded by a Lock.
_model_cache: dict[str, dict] = {}
_model_lock = threading.Lock()


def _slice_audio(
    audio_path: str, start_sec: float, end_sec: float, out_path: str,
) -> bool:
    """Extract [start, end] from ``audio_path`` as 16 kHz mono WAV."""
    duration = max(0.0, end_sec - start_sec)
    if duration <= 0:
        return False
    try:
        cmd = [
            "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
            "-ss", f"{start_sec:.3f}",
            "-i", audio_path,
            "-t", f"{duration:.3f}",
            "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            return False
        if not os.path.exists(out_path) or os.path.getsize(out_path) < 200:
            return False
        return True
    except Exception:
        return False


def _load_aligner(language: str, device: str) -> Optional[dict]:
    """Lazy-load and cache the appropriate alignment pipeline.

    Returns a dict with keys ``model``, ``tokenizer``, ``labels``,
    ``device`` on success, or None when the dependency / weights are
    unavailable. Picks WAV2VEC2_ASR_BASE_960H for English, MMS_FA
    otherwise.
    """
    lang = (language or "en").lower()
    is_english = lang in ("en", "en-us", "en-gb", "english")
    cache_key = "wav2vec2_en_base_960h" if is_english else f"mms_fa_{lang}"

    with _model_lock:
        cached = _model_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            import torch
            import torchaudio
        except Exception as e:
            logger.info("forced_alignment: torch/torchaudio unavailable (%s)", e)
            return None
        try:
            if is_english:
                bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
            else:
                # MMS_FA handles 1100+ languages.
                bundle = getattr(
                    torchaudio.pipelines, "MMS_FA", None,
                )
                if bundle is None:
                    logger.info(
                        "forced_alignment: MMS_FA pipeline not in this "
                        "torchaudio (need >=2.2)",
                    )
                    return None
            try:
                model = bundle.get_model()
            except Exception as e:
                logger.info("forced_alignment: weights unavailable (%s)", e)
                return None
            target_device = (
                "cuda" if device == "cuda" and torch.cuda.is_available()
                else "cpu"
            )
            model = model.to(target_device).eval()
            entry = {
                "model": model,
                "labels": list(bundle.get_labels()),
                "tokenizer": getattr(bundle, "get_tokenizer", lambda: None)(),
                "bundle": bundle,
                "device": target_device,
                "is_english": is_english,
            }
            _model_cache[cache_key] = entry
            return entry
        except Exception as e:
            logger.warning("forced_alignment: load failed (%s)", e)
            return None


def _normalize_text(text: str, labels: list[str]) -> list[str]:
    """Tokenize text into characters present in the bundle's label set.

    For wav2vec2 base 960h the labels are upper-case letters + ``|``
    (word separator) + ``-`` + ``'``. Characters outside the set are
    dropped; spaces become ``|``.
    """
    label_set = {ch for ch in labels}
    tokens: list[str] = []
    for raw_word in text.upper().split():
        word_tokens = [ch for ch in raw_word if ch in label_set]
        if not word_tokens:
            continue
        tokens.extend(word_tokens)
        tokens.append("|")
    if tokens and tokens[-1] == "|":
        tokens.pop()
    return tokens


def align_text_to_audio(
    audio_path: str,
    start_sec: float,
    end_sec: float,
    text: str,
    *,
    language: str = "en",
    device: str = "cpu",
    min_word_confidence: float = 0.6,
) -> AlignmentResult:
    """Force-align ``text`` to ``audio_path[start_sec:end_sec]``.

    Returns AlignmentResult with per-word timestamps (in original-
    audio time) and confidences. Words below ``min_word_confidence``
    are dropped. When alignment cannot be performed (deps missing,
    interval too long, etc.), returns an empty result with a
    structured ``reason``.

    For chunk-edge truncation specifically — the most common
    application of Rung 4 — the neighbor text concatenation
    (last 5 before + first 5 after) usually has a few words that
    actually do fall inside the gap; those align cleanly with high
    confidence.
    """
    if not text or not text.strip():
        return AlignmentResult(reason="empty_text")
    duration = end_sec - start_sec
    if duration <= 0:
        return AlignmentResult(reason="empty_text")
    if duration > _MAX_ALIGN_DURATION_SEC:
        return AlignmentResult(reason="interval_too_long")

    aligner = _load_aligner(language, device)
    if aligner is None:
        if (language or "").lower() not in ("en", "en-us", "en-gb", "english"):
            return AlignmentResult(reason="language_unsupported")
        return AlignmentResult(reason="model_unavailable")

    # Slice audio.
    with tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False, dir=tempfile.gettempdir(),
    ) as tmp:
        slice_path = tmp.name
    try:
        if not _slice_audio(audio_path, start_sec, end_sec, slice_path):
            return AlignmentResult(reason="slice_failed")

        try:
            import torch
            import torchaudio
        except Exception:
            return AlignmentResult(reason="model_unavailable")

        try:
            waveform, sr = torchaudio.load(slice_path)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.to(aligner["device"])

            tokens = _normalize_text(text, aligner["labels"])
            if not tokens:
                return AlignmentResult(reason="empty_text")

            with torch.inference_mode():
                emissions, _ = aligner["model"](waveform)
                # Convert label-strings back to token IDs.
                label_to_id = {ch: i for i, ch in enumerate(aligner["labels"])}
                token_ids = torch.tensor(
                    [label_to_id[t] for t in tokens if t in label_to_id],
                    dtype=torch.int32, device=aligner["device"],
                ).unsqueeze(0)
                if token_ids.numel() == 0:
                    return AlignmentResult(reason="empty_text")
                blank_id = label_to_id.get("-", 0)
                # forced_align is the standard CTC alignment helper.
                aligned, scores = torchaudio.functional.forced_align(
                    emissions.contiguous(),
                    token_ids,
                    blank=blank_id,
                )

            # Rebuild words from the aligned token sequence. ``|`` is
            # the word separator. Each surviving word emits one tuple
            # (start_sec, end_sec, text, confidence).
            ratio = waveform.shape[1] / emissions.shape[1]
            sample_rate = 16000.0
            tokens_flat = tokens
            words_out: list[tuple[float, float, str, float]] = []
            buf: list[tuple[int, str, float]] = []  # (frame_idx, char, score)
            aligned_seq = aligned[0].tolist()
            score_seq = scores[0].tolist() if scores.dim() == 2 else scores.tolist()
            tok_iter = iter(zip(tokens_flat, aligned_seq, score_seq))

            def _flush_word(items):
                if not items:
                    return
                first_frame = items[0][0]
                last_frame = items[-1][0]
                start_t = (first_frame * ratio) / sample_rate + start_sec
                end_t = ((last_frame + 1) * ratio) / sample_rate + start_sec
                conf = sum(s for _, _, s in items) / len(items)
                # Reverse: turn the per-character ID stream back into
                # the original word by joining the chars (already
                # uppercase from _normalize_text).
                word_text = "".join(c for _, c, _ in items)
                words_out.append((start_t, end_t, word_text, float(conf)))

            for tok, frame, score in tok_iter:
                if tok == "|":
                    _flush_word(buf)
                    buf = []
                    continue
                buf.append((frame, tok, float(score)))
            _flush_word(buf)

            # Confidence threshold filter.
            kept = [w for w in words_out if w[3] >= min_word_confidence]
            return AlignmentResult(
                words=kept,
                reason="ok" if kept else "no_alignment",
            )
        except Exception as e:
            logger.info("forced_alignment: align failed (%s)", e)
            return AlignmentResult(reason="no_alignment")
    finally:
        try:
            os.unlink(slice_path)
        except OSError:
            pass
