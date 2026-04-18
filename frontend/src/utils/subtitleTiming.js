// Shared subtitle / transcript timing helpers.
//
// Both ``SubtitleOverlay`` (burned-in preview subtitles) and
// ``TranscriptViewer`` (Analysis / ClipSEO active-line highlight) need
// to answer the same question: "at time ``t``, which transcript segment
// is actually being spoken?" The honest answer isn't
// ``seg.start <= t < seg.end`` — Whisper segment-level timestamps are
// boundary estimates that routinely drift ±100–300 ms from real speech
// onset/offset, *and* diarization can split a single Whisper segment
// into two rows with the same text and neighbouring time ranges, which
// means a naive ``first contiguous match`` loop rushes the highlight to
// the next row before the current one's audio has actually ended.
//
// ``spokenWindow(seg)`` returns a ``{ start, end }`` tuple that:
//   1. Uses ``words[0].start`` (with a small head anticipation) for the
//      window start when per-word timestamps are present and plausible,
//      so the highlight lands *with* the first syllable instead of
//      after it.
//   2. Uses ``max(seg.end, words[-1].end) + WORD_TAIL_S`` for the
//      window end — ``max``, not ``min``. This is the important fix
//      for "the next line is highlighted while the current one is
//      still being spoken": if Whisper's per-word timestamps end
//      before ``seg.end`` (common when a line trails off into
//      silence or when the speaker's last syllable is long), the old
//      ``min`` clamp cut the segment short and let the next row win
//      the direct-match race even though the current row's line was
//      still audibly playing. Using ``max`` keeps the current row
//      active through its full Whisper segment window *and* through
//      any trailing words that extend past it.
//   3. Falls back to ``[seg.start, seg.end + WORD_TAIL_S]`` when the
//      word list is missing or broken. Extending ``seg.end`` by
//      ``WORD_TAIL_S`` in the fallback path is what produces "late
//      bias" on transitions between back-to-back non-word-timestamped
//      segments: adjacent windows overlap, and the first-match loop in
//      the consumers picks the earlier-started one so the highlight
//      hangs on the previous line for ``WORD_TAIL_S`` extra seconds.
//
// ``WORD_TAIL_S`` absorbs Whisper's ±100–300 ms boundary drift.
// Kept small (0.10 s) because the backend VAD padding already extends
// segment ends; larger values compound with backend drift and make
// subtitles linger visibly past the spoken word.

// Tail padding (s) added to every segment's effective end. Absorbs
// Whisper's boundary drift without compounding with backend VAD
// padding.
export const WORD_TAIL_S = 0.10;

// Head padding (s) before the first word's Whisper start.  Set to 0
// because the backend's per-speaker anticipation in SubtitleOverlay
// already handles onset timing; double-applying head shift made
// subtitles visibly premature.
export const WORD_HEAD_S = 0.00;

/**
 * Compute the ``[start, end]`` window during which a transcript segment
 * is actually being spoken. Prefers per-word timestamps when available
 * and plausible; falls back to segment-level timestamps otherwise.
 *
 * The returned shape matches ``{ start, end }`` so callers can treat
 * the output as a drop-in replacement for ``seg.start`` / ``seg.end``.
 */
export function spokenWindow(seg) {
  if (!seg) return { start: 0, end: 0 };
  const segStart = seg.start ?? 0;
  const segEnd = seg.end ?? segStart;
  const words = Array.isArray(seg.words) ? seg.words : null;
  if (!words || words.length === 0) {
    // No per-word timestamps — extend the effective end by WORD_TAIL_S
    // so adjacent back-to-back segments overlap and the first-match
    // loop in the consumers picks the earlier-started one, producing
    // the late-bias transition described at the top of this file.
    return { start: segStart, end: segEnd + WORD_TAIL_S };
  }
  // Guard: the word list must bracket the segment plausibly. If word
  // timestamps are broken (zero-duration, inverted, or non-numeric)
  // fall back to segment-level.
  const first = words[0];
  const last = words[words.length - 1];
  const ws = first?.start ?? first?.startTime;
  const we = last?.end ?? last?.endTime;
  if (typeof ws !== 'number' || typeof we !== 'number' || we <= ws) {
    return { start: segStart, end: segEnd + WORD_TAIL_S };
  }
  // Start: tight to the first word's onset with a small head
  // anticipation, but never earlier than ``segStart - 0.02`` so we
  // can't overlap the *previous* segment's end.
  const start = Math.max(segStart - 0.02, ws - WORD_HEAD_S);
  // End: take the LATER of ``segEnd`` and ``we`` and then add
  // ``WORD_TAIL_S``. Using MAX (not MIN) is the fix for "the next line
  // is highlighted while the current line is still being spoken" —
  // the old ``min(segEnd + 0.15, we + WORD_TAIL_S)`` formula cut
  // segments short whenever ``we`` landed before ``segEnd``, which is
  // very common for lines that trail off into silence. Using ``max``
  // keeps the current row active through whichever boundary is later.
  const end = Math.max(segEnd, we) + WORD_TAIL_S;
  if (end <= start) return { start: segStart, end: segEnd + WORD_TAIL_S };
  return { start, end };
}

/**
 * True iff ``t`` is inside the spoken window of ``seg``. Shared
 * predicate used by ``SubtitleOverlay`` and ``TranscriptViewer`` so
 * their notion of "active" stays consistent.
 */
export function isSpokenAt(seg, t) {
  const { start, end } = spokenWindow(seg);
  return start <= t && t < end;
}
