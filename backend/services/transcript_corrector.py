"""AI-powered transcript post-correction.

Uses the configured LLM provider to fix common Whisper transcription issues:
- Proper noun capitalization (people, brands, places)
- Punctuation and sentence boundaries
- Filler word removal (um, uh, like, you know)
- Number formatting inconsistencies
"""

import asyncio
import json
import logging
from difflib import SequenceMatcher
from backend.config import settings
from backend.models import TranscriptSegment, WordTimestamp
from backend.services.ai_orchestrator import AIOrchestrator

logger = logging.getLogger(__name__)

CORRECTION_PROMPT_EN = """You are a transcript correction assistant. Fix the following ENGLISH transcript segments while preserving their exact timing and structure.

Rules:
1. Fix capitalization of proper nouns (names, brands, places)
2. Fix punctuation and sentence boundaries
3. Remove filler words: "um", "uh", "like" (when used as filler), "you know", "I mean" (when used as filler)
4. Fix obvious transcription errors (e.g., "there" → "their" based on context)
5. DO NOT change the meaning, add content, or rephrase
6. DO NOT merge or split segments — return exactly the same number of segments
7. Return ONLY a JSON array of corrected text strings, one per input segment

Input segments:
{segments_json}

Return a JSON array of corrected text strings, exactly {count} elements.
Example: ["Fixed text one.", "Fixed text two."]"""

CORRECTION_PROMPT_JA = """You are a Japanese transcript correction assistant. Fix the following JAPANESE transcript segments while preserving their exact timing and structure.

Rules:
1. Fix incorrect kanji (e.g., wrong homophone kanji)
2. Fix broken sentence boundaries — merge fragments that were split mid-sentence
3. Remove speech disfluencies: えーと, あの, まあ, その (when used as fillers)
4. Fix mixed-script errors (e.g., random katakana in hiragana words)
5. DO NOT translate to English — keep ALL text in Japanese
6. DO NOT change the meaning, add content, or rephrase
7. DO NOT merge or split segments — return exactly the same number of segments
8. Return ONLY a JSON array of corrected text strings, one per input segment

Input segments:
{segments_json}

Return a JSON array of corrected text strings, exactly {count} elements."""

CORRECTION_PROMPT_GENERIC = """You are a transcript correction assistant. Fix the following transcript segments (detected language: {language}) while preserving their exact timing and structure.

Rules:
1. Fix obvious transcription errors appropriate for {language}
2. Fix punctuation and sentence boundaries
3. Remove speech disfluencies and filler words common in {language}
4. DO NOT translate to a different language — keep ALL text in {language}
5. DO NOT change the meaning, add content, or rephrase
6. DO NOT merge or split segments — return exactly the same number of segments
7. Return ONLY a JSON array of corrected text strings, one per input segment

Input segments:
{segments_json}

Return a JSON array of corrected text strings, exactly {count} elements."""


def _get_correction_prompt(language: str) -> str:
    """Get the appropriate correction prompt for the detected language."""
    lang = language.lower().strip() if language else "en"
    if lang == "en":
        return CORRECTION_PROMPT_EN
    elif lang == "ja":
        return CORRECTION_PROMPT_JA
    else:
        return CORRECTION_PROMPT_GENERIC


_CJK_RANGES_CORRECTOR = [('\u3040', '\u30ff'), ('\u4e00', '\u9fff'), ('\uac00', '\ud7af')]


def _detect_transcript_language(segments: list) -> str:
    """Detect the language of transcript segments from their text content."""
    if not segments:
        return "en"
    sample_text = " ".join(s.text for s in segments[:5])
    cjk_count = sum(1 for ch in sample_text if any(lo <= ch <= hi for lo, hi in _CJK_RANGES_CORRECTOR))
    if cjk_count / max(len(sample_text), 1) > 0.3:
        jp_count = sum(1 for ch in sample_text if '\u3040' <= ch <= '\u30ff')
        return "ja" if jp_count > 0 else "zh"
    if any('\u0600' <= ch <= '\u06ff' for ch in sample_text):
        return "ar"
    return "en"


MAX_BATCH_RETRIES = 1  # Only 1 retry (2 total attempts) — polishing is optional


def _realign_word_timestamps(segment: TranscriptSegment, original_text: str) -> TranscriptSegment:
    """Re-align word timestamps after AI text correction.

    When the AI removes filler words or changes text, the word-level
    timestamps may no longer match. This function attempts to preserve
    timestamps for words that survived the correction.
    """
    if not segment.words:
        return segment

    corrected_words = segment.text.split()
    original_words = [w.word for w in segment.words]

    if not corrected_words:
        return segment

    # Build a mapping from corrected words to original word timestamps
    # using a greedy alignment via SequenceMatcher
    matcher = SequenceMatcher(
        None,
        [w.lower().strip('.,!?;:') for w in original_words],
        [w.lower().strip('.,!?;:') for w in corrected_words],
    )

    new_word_timestamps = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for orig_idx, corr_idx in zip(range(i1, i2), range(j1, j2)):
                orig_wt = segment.words[orig_idx]
                new_word_timestamps.append(WordTimestamp(
                    start=orig_wt.start,
                    end=orig_wt.end,
                    word=corrected_words[corr_idx],
                ))
        elif tag == 'replace':
            # Map replacement words to the original time span
            if i1 < len(segment.words) and i2 <= len(segment.words):
                span_start = segment.words[i1].start
                span_end = segment.words[i2 - 1].end
                corr_count = j2 - j1
                if corr_count > 0:
                    span_duration = span_end - span_start
                    per_word = span_duration / corr_count
                    for k, corr_idx in enumerate(range(j1, j2)):
                        new_word_timestamps.append(WordTimestamp(
                            start=round(span_start + k * per_word, 3),
                            end=round(span_start + (k + 1) * per_word, 3),
                            word=corrected_words[corr_idx],
                        ))
        elif tag == 'insert':
            # New words inserted by AI — interpolate timestamps
            if new_word_timestamps:
                last_end = new_word_timestamps[-1].end
                next_start = segment.words[i1].start if i1 < len(segment.words) else last_end + 0.1
                gap = next_start - last_end
                corr_count = j2 - j1
                per_word = gap / max(corr_count, 1)
                for k, corr_idx in enumerate(range(j1, j2)):
                    new_word_timestamps.append(WordTimestamp(
                        start=round(last_end + k * per_word, 3),
                        end=round(last_end + (k + 1) * per_word, 3),
                        word=corrected_words[corr_idx],
                    ))
        # tag == 'delete' — filler words removed, skip their timestamps

    if new_word_timestamps:
        return segment.model_copy(update={"words": new_word_timestamps})
    return segment


def _parse_correction_response(response: str, expected_count: int) -> list[str] | None:
    """Parse an AI correction response into a list of strings.

    Returns the list on success, or None if the response is unparseable or
    has the wrong number of elements.
    """
    cleaned = response.strip()
    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        corrections = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    if isinstance(corrections, list) and len(corrections) == expected_count:
        return corrections
    return None


# ── Timeouts ──────────────────────────────────────────────────────────
_STANDARD_BATCH_TIMEOUT = 90       # per-batch timeout for standard models
_THINKING_BATCH_TIMEOUT = 150      # per-batch timeout for thinking models
_OUTER_TIMEOUT_MARGIN = 30         # outer = inner + margin
_MAX_CONCURRENCY = 3               # max parallel batch requests


def _adaptive_batch_size(segment_count: int) -> int:
    """Scale batch size inversely with transcript length.

    Smaller batches for large transcripts:
    - Faster per-batch inference (shorter prompts)
    - Better concurrency utilization
    - Smaller blast radius on failure (fewer segments lost)
    """
    if segment_count <= 50:
        return 25
    elif segment_count <= 150:
        return 15
    else:
        return 10


async def correct_transcript(
    segments: list[TranscriptSegment],
    orchestrator: AIOrchestrator,
    batch_size: int | None = None,
    job_id: str = "",
    language: str = "",
) -> list[TranscriptSegment]:
    """Correct transcript text using the user's configured AI text model.

    Improvements over the original implementation:
    - Adaptive batch sizing based on transcript length
    - Concurrent batch processing (up to 3 parallel requests)
    - Extended per-batch timeouts (90s standard, 150s thinking)
    - Graceful partial success (batches that succeed are kept, failures stay raw)

    Uses skip_circuit_breaker=True so failures here do NOT degrade providers
    for subsequent critical operations (summary, clip detection).
    """
    if not settings.AI_TRANSCRIPT_CORRECTION:
        return segments

    if not segments:
        return segments

    # Query which model will handle polishing
    model_info = orchestrator.get_text_model_info()
    model_name = model_info["model"]
    is_thinking = model_info["is_thinking"]

    if model_info["provider"] == "none":
        logger.warning("No AI providers available for transcript correction — skipping")
        return segments

    # Adaptive batch size
    if batch_size is None:
        batch_size = _adaptive_batch_size(len(segments))

    # Set timeout based on model type
    inner_timeout = _THINKING_BATCH_TIMEOUT if is_thinking else _STANDARD_BATCH_TIMEOUT
    outer_timeout = inner_timeout + _OUTER_TIMEOUT_MARGIN

    logger.info(
        "Transcript correction: %d segments, batch_size=%d, model=%s via %s, "
        "timeout=%ds/batch, concurrency=%d",
        len(segments), batch_size, model_name, model_info["provider"],
        inner_timeout, _MAX_CONCURRENCY,
    )

    # Use Whisper's detected language if provided, fall back to heuristic
    _detected_lang = language.lower().strip() if language else ""
    if not _detected_lang:
        _detected_lang = _detect_transcript_language(segments)
    _correction_prompt_template = _get_correction_prompt(_detected_lang)
    logger.info("Transcript correction: language=%s (source=%s)",
                _detected_lang, "whisper" if language else "heuristic")

    corrected = list(segments)  # Copy to avoid mutating input
    total_batches = -(-len(segments) // batch_size)  # ceiling division
    semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)

    # Track first-batch probe result for early exit
    first_batch_result = None  # Will be set by batch 0

    async def _process_batch(batch_idx: int, batch_start: int) -> bool:
        """Process a single batch. Returns True on success, False on failure."""
        nonlocal first_batch_result

        batch = segments[batch_start : batch_start + batch_size]
        batch_label = f"batch {batch_idx + 1}/{total_batches} (segs {batch_start}-{batch_start + len(batch) - 1})"

        async with semaphore:
            # Early exit: if batch 0 already failed, skip this batch
            if first_batch_result is False and batch_idx > 0:
                logger.info("Skipping %s — probe batch failed", batch_label)
                return False

            seg_texts = [{"index": i, "text": seg.text} for i, seg in enumerate(batch)]
            prompt = _correction_prompt_template.format(
                segments_json=json.dumps(seg_texts, ensure_ascii=False, indent=2),
                count=len(batch),
                language=_detected_lang,
            )

            max_attempts = MAX_BATCH_RETRIES + 1
            for attempt in range(max_attempts):
                try:
                    response = await asyncio.wait_for(
                        orchestrator.text_completion(
                            prompt, timeout=inner_timeout, job_id=job_id,
                            skip_circuit_breaker=True,
                        ),
                        timeout=outer_timeout,
                    )

                    corrections = _parse_correction_response(response, len(batch))
                    if corrections is None:
                        logger.warning(
                            "AI correction %s returned unparseable response (attempt %d/%d)",
                            batch_label, attempt + 1, max_attempts,
                        )
                        if attempt < max_attempts - 1:
                            continue
                        if batch_idx == 0:
                            first_batch_result = False
                        return False

                    # Apply corrections
                    for i, new_text in enumerate(corrections):
                        if isinstance(new_text, str) and new_text.strip():
                            idx = batch_start + i
                            old_text = corrected[idx].text
                            new_stripped = new_text.strip()

                            # Safety: reject corrections that remove >30% of words
                            old_word_count = len(old_text.split())
                            new_word_count = len(new_stripped.split())
                            if old_word_count > 3 and new_word_count < old_word_count * 0.7:
                                logger.warning(
                                    "Rejecting correction that drops too many words "
                                    "(was %d words, now %d): '%s' → '%s'",
                                    old_word_count, new_word_count,
                                    old_text[:60], new_stripped[:60],
                                )
                                continue

                            corrected[idx] = corrected[idx].model_copy(
                                update={"text": new_stripped}
                            )
                            if old_text != new_stripped and corrected[idx].words:
                                corrected[idx] = _realign_word_timestamps(corrected[idx], old_text)

                    logger.info("Corrected %s successfully", batch_label)
                    if batch_idx == 0:
                        first_batch_result = True
                    return True

                except asyncio.TimeoutError:
                    logger.warning(
                        "AI correction %s timed out (%ds, attempt %d/%d)",
                        batch_label, inner_timeout, attempt + 1, max_attempts,
                    )
                    if attempt < max_attempts - 1:
                        continue
                except Exception as e:
                    logger.warning(
                        "AI correction %s failed (attempt %d/%d): %s",
                        batch_label, attempt + 1, max_attempts, e,
                    )
                    if attempt < max_attempts - 1:
                        continue

            # All attempts exhausted
            logger.warning("All attempts exhausted for %s — keeping raw transcript", batch_label)
            if batch_idx == 0:
                first_batch_result = False
            return False

    # ── Probe-then-fan-out strategy ──
    # Run batch 0 first as a probe. If the model/provider is completely
    # unavailable, we skip all remaining batches immediately instead of
    # firing N concurrent requests that all timeout.
    batch_ranges = list(range(0, len(segments), batch_size))

    if batch_ranges:
        # Probe: run batch 0 alone
        await _process_batch(0, batch_ranges[0])

        if first_batch_result is False:
            logger.warning(
                "Probe batch failed — skipping remaining %d batches (model %s likely unavailable)",
                len(batch_ranges) - 1, model_name,
            )
        elif len(batch_ranges) > 1:
            # Fan out remaining batches concurrently
            tasks = [
                _process_batch(idx, start)
                for idx, start in enumerate(batch_ranges[1:], start=1)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            succeeded = sum(1 for r in results if r is True)
            failed = sum(1 for r in results if r is not True)
            logger.info(
                "Transcript correction: %d/%d batches succeeded (probe + %d concurrent)",
                succeeded + 1, total_batches, len(tasks),
            )

    # ── Second pass: contextual correction with sliding window ──
    # Each batch includes surrounding segments as read-only context.
    # This fixes cross-segment errors (broken sentences, pronoun resolution).
    if _detected_lang in ("en", "") and first_batch_result is not False:
        _CONTEXT_PROMPT = """You are a transcript refinement assistant. Review these segments WITH their surrounding context and fix cross-segment issues.

Rules:
1. Fix sentences that were split across segments (ensure both halves are grammatically complete)
2. Fix inconsistent proper noun spelling across segments
3. Fix pronoun ambiguity ONLY if the previous context makes it obvious
4. DO NOT merge or split segments — return exactly {count} corrected strings
5. DO NOT change content that is already correct
6. Return ONLY a JSON array of {count} strings

Context (DO NOT include in output — for reference only):
{context}

Segments to refine:
{segments_json}

Return a JSON array of {count} refined strings."""

        context_batch_size = min(10, batch_size or 10)
        for batch_start in range(0, len(corrected), context_batch_size):
            batch_end = min(batch_start + context_batch_size, len(corrected))
            batch = corrected[batch_start:batch_end]

            ctx_before = corrected[max(0, batch_start - 3):batch_start]
            ctx_after = corrected[batch_end:min(len(corrected), batch_end + 3)]
            context_lines = []
            for s in ctx_before:
                context_lines.append(f"[BEFORE] {s.text}")
            for s in ctx_after:
                context_lines.append(f"[AFTER] {s.text}")

            seg_texts = [{"index": i, "text": s.text} for i, s in enumerate(batch)]
            prompt = _CONTEXT_PROMPT.format(
                count=len(batch),
                context="\n".join(context_lines) if context_lines else "(start/end of transcript)",
                segments_json=json.dumps(seg_texts, ensure_ascii=False, indent=2),
            )

            try:
                response = await asyncio.wait_for(
                    orchestrator.text_completion(
                        prompt, timeout=inner_timeout, job_id=job_id,
                        skip_circuit_breaker=True,
                    ),
                    timeout=outer_timeout,
                )
                refinements = _parse_correction_response(response, len(batch))
                if refinements:
                    for i, new_text in enumerate(refinements):
                        if isinstance(new_text, str) and new_text.strip():
                            idx = batch_start + i
                            old_text = corrected[idx].text
                            if new_text.strip() != old_text:
                                corrected[idx] = corrected[idx].model_copy(
                                    update={"text": new_text.strip()}
                                )
                                if corrected[idx].words:
                                    corrected[idx] = _realign_word_timestamps(corrected[idx], old_text)
                    logger.info("Context refinement batch %d-%d succeeded", batch_start, batch_end - 1)
            except Exception as e:
                logger.debug("Context refinement batch %d failed (non-critical): %s", batch_start, e)

    return corrected
