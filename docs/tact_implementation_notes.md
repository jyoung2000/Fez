# TACT implementation notes

Notes from reading the §0 reference files before writing any TACT code.
Captures the integration ambiguities the prompt could not anticipate from
outside the repo, and the decisions made for Phase 1.

## Branch

The system harness designates `claude/implement-tact-transcription-XHVao`
as the development branch; the prompt body specifies
`claude/critic-engine-implementation-cTvFT`. Defaulted to the harness
branch (the one we are checked out on). If the user wants the other
branch, the work is rebaseable onto it — Phase 1 touches only additive
files plus a small block in `pipeline.py` and a few keys in `config.py`.

## What's already in the repo (and not what the prompt assumed)

1. **`coverage_report` already exists on `JobResult`** —
   `backend/models.py:269` declares
   `coverage_report: Optional[dict] = None`, with a comment that it is a
   "legacy" field from a retired Silero-VAD gap-fill feature, "no longer
   produced or read." The prompt's §1 entry for "add optional
   coverage_report JSON column on jobs" therefore needs **no schema
   change** — the field is already there. We just start writing to it.

2. **`database.py` is JSON-on-disk, not SQL.** Each job is a `JobResult`
   pydantic model serialized to `/data/uploads/{job_id}/job.json`.
   Updating fields is done via `update_job_status(job_id, **kwargs)`
   which `setattr`s any attribute that exists on the model. So
   "`update_job_coverage_report`" as a separate function is unnecessary
   — `update_job_status(job_id, coverage_report=...)` works today.
   Phase 1 calls that.

3. **`active_speaker.build_vad_presence`** at
   `backend/services/active_speaker.py:169` returns
   `list[tuple[float, float]]` of voiced intervals using webrtcvad
   (preferred) or a numpy energy+ZCR fallback. This is the independent
   VAD the gap-filler already consumes; Phase 1 reuses it transitively
   via the existing gap-filler.

4. **The gap-filler already implements `compute_coverage`** at
   `transcription_gap_filler.py:257` and emits `GapFillStats` (logged
   to stdout but not persisted). The prompt's §0 statement that the
   gap-filler is "TACT Stage 4 Rung 1 already" is accurate; Phase 1's
   ledger is genuinely a new layer that reuses these primitives, not a
   rewrite.

5. **Pipeline integration site is at `pipeline.py:2607`** (the line
   range in the prompt is correct), inside the `if (... and result and
   audio_duration > 30):` guard around `fill_transcript_gaps`. After
   that block ends (around line 2647) is where the ledger emission goes.

## Key design decisions for Phase 1 (zero behavior change)

**Goal:** add a `CoverageLedger` data structure, emit its
`to_report_dict()` to `JobResult.coverage_report` for every job, and
log a one-line summary. No transcript text changes; no segment changes.

### Ledger storage

Run-length encoded list of `LedgerSpan` records, sorted by `start_ms`,
disjoint by construction (overlapping claims resolved at `claim()`
time by confidence). 20 ms bin default — a 90-min file is 270k bins,
but with RLE the actual storage is `O(spans)` ≈ a few thousand at
most, well under the 5 MB target. Hard cap at 100k spans with
aggressive coalescing as a guardrail (per prompt §5).

### Status taxonomy (Phase 1 only uses a subset)

The prompt specifies seven statuses:
`covered_speech | covered_event | covered_silence | low_confidence |
contested | uncovered | quarantined`. Phase 1 only writes
`covered_speech` (from existing transcript segments) and `uncovered`
(implicit complement). The remaining statuses are reserved keys for
Phase 2+ so the `to_report_dict()` schema is forward-compatible — they
appear in the histogram with count 0 today.

### `from_segments` mapping

Existing pipeline emits `TranscriptSegment` instances with
`start`/`end`/`text`/`speaker`/`words`/`confidence`/`avg_logprob`/
`no_speech_prob`. `from_segments` maps each segment to a single
`covered_speech` `LedgerSpan` with the segment's text as `content` and
`content_type="phrase"` (not `"word"` — segment text is multi-word).
Word-level claims become a Phase 3 concern when the reconciler needs
per-word provenance.

### Confidence

Use `seg.confidence` if present, else `max(0, min(1, 1 + avg_logprob))`,
else `1.0`. Matches the gap-filler's own confidence derivation
(`transcription_gap_filler.py:540`).

### Coverage ratio definition

The prompt is silent on whether coverage ratio is over total audio or
over voiced audio. We expose **both** in the report dict
(`coverage_ratio` = covered/total, `voiced_coverage_ratio` =
covered_speech ∩ vad_voiced / vad_voiced) so downstream consumers
don't have to recompute. Phase 1 only fills `covered_speech`, so
`coverage_ratio` is just transcript density and `voiced_coverage_ratio`
matches the existing `compute_coverage` voiced number — useful for
sanity-checking Phase 1 against the gap-filler's logs.

### Failure mode

Wrapped in `try/except` in the pipeline. Any failure logs a warning
and proceeds — the ledger is observability, not control flow.

## Integration ordering vs the existing gap-filler

The gap-filler runs *before* the ledger emission (the prompt is
explicit about this — `from_segments` should be called with
`source_pass="whisper_main+gap_fill"`). That means the ledger
captures the post-gap-fill segment list and the
`coverage_report.coverage_ratio` we log will already include any gap
recovery. This is the right boundary: in Phase 2, the ladder
**replaces** the gap-filler call, and the ledger becomes the source of
truth for which intervals need escalation. Phase 1 ledger is purely
post-hoc.

## Out-of-scope (per prompt §3)

- `ass_generator.py`, `clip_exporter.py` — untouched.
- Frontend — untouched. Coverage report appears in `JobResult` JSON
  only.
- `_chunk_audio`/`_merge_chunk_segments` — untouched.

## Phase 1 acceptance verification plan

- All existing tests pass: `pytest backend/tests/test_transcription_gap_filler.py
  backend/tests/test_coverage_repair_in_place.py` (+ any other
  transcription tests).
- New tests in `backend/tests/test_transcription_ledger.py`:
  - empty audio
  - single-segment claim round-trip via `from_segments`
  - overlap with confidence-based winner
  - bin-width invariance (10/20/40 ms produce equivalent ratio
    within 1%)
  - schema snapshot for `to_report_dict()`
- With `TACT_LEDGER_ENABLED=False` (or any failure path),
  `pipeline.py` emits zero new fields → byte-identical `job.json`
  output for the previous test fixtures.

## Open questions deferred to later phases

- Phase 2's `_filter_hallucinations` signature change `-> tuple[list,
  list]` is a breaking change at every call site. Need to grep and
  enumerate callers when we get there. (Already noted in the prompt;
  deferred.)
- The `TranscriptSegment` model has no `flags` field. The prompt's
  `LedgerSpan.flags` exists at the ledger level, which is correct;
  emission back to `TranscriptSegment` via `to_segments()` will lose
  that data. We accept that — flags are diagnostic, not transport.
- Phase 4 Parakeet co-residency on a 4 GB 1650 alongside the existing
  Whisper subprocess and Ollama is genuinely tight per the existing
  VRAM-management code in `transcription.py:510-665`. The
  `_get_gpu_free_mb()` machinery is already there to support the
  graceful-fallback contract.

---

## Phase 2 callers to update (`_filter_hallucinations`)

Mandatory grep, run before changing the signature. From
`grep -rn "_filter_hallucinations" backend/ --include='*.py'`:

| File | Line | Kind | Action when signature changes |
|------|------|------|-------------------------------|
| `backend/services/transcription.py` | 917 | call (inside `transcribe_audio_subprocess`) | Consume `(kept, quarantined)` — bubble `quarantined` up so the pipeline can claim it into the ledger as `status="quarantined"`. |
| `backend/services/transcription.py` | 1810 | call (partial-recovery path inside `transcribe_audio` exception handler) | `kept, _ = _filter_hallucinations(...)` — explicit discard. Partial recovery is already a degraded-output path; quarantine is not actionable here. |
| `backend/services/transcription.py` | 2488 | call (inside `_transcribe_sync` final filter pass) | Bubble up — same pattern as line 917. |
| `backend/services/transcription.py` | 3023 | the function definition itself | Change return type `list[dict] -> tuple[list[dict], list[dict]]`; preserve every existing `continue` site, but tag and append to `quarantined` instead of dropping. |
| `backend/services/transcription.py` | 3227 | comment only | No code change. |
| `backend/services/transcription.py` | 3377 | comment only | No code change. |
| `backend/services/transcription_gap_filler.py` | 7 | docstring reference | No code change. |
| `backend/tests/test_transcription.py` | 447 | comment only | No code change. |

Both call-site updates at lines 917 and 2488 currently happen deep
inside subprocess-result-handling code that returns `list[TranscriptSegment]`.
The transport mechanism for the quarantined list back up to the
pipeline integration site is the cleanest open question; two options:

  (a) Stash quarantined dicts on a module-level `_last_quarantined`
      dict keyed off the audio_path / job_id so the pipeline retrieves
      it after `transcribe_audio_subprocess` returns. Mirrors the
      `_last_detected_language` and `_last_diarization_method`
      pattern already in this file.
  (b) Change the subprocess-fronting helpers to return
      `tuple[list[TranscriptSegment], list[dict]]`. Cleaner but
      touches more callers.

Going with (a) — module-level `_last_quarantined_segments`. Same
pattern as `_last_detected_language` (line 20) and
`_last_diarization_method` (line 23). Avoids changing the public
shape of `transcribe_audio_subprocess` / `_transcribe_sync`. The
pipeline reads it after transcription, claims the entries into the
ledger with `status="quarantined"`, and the dict is reset on the
next `transcribe_audio_*` call.

When `TACT_QUARANTINE_HALLUCINATIONS=False`, every dropped segment
still ends up dropped (compatible with pre-TACT behavior); the
quarantined list is still populated for inspection but the pipeline
ignores it. This means the function-level signature change is
permanent (removing the `if flag` branch keeps the call sites
uniform), and only the *consumption* of the quarantined list is
flag-gated.

## Phase 2 scope for this session

This session's commit covers the foundation pieces of Phase 2 — the
parts that other Rungs and the pipeline integration build on:

1. `_filter_hallucinations` tuple-return signature change with all
   three callers updated and a module-level `_last_quarantined_segments`
   accessor.
2. Ledger surface additions for Phase 2: `status` kwarg on
   `from_segments`, `claim_word` for forthcoming Phase 3 word-level
   claims (used in tests now to lock the API).
3. `escalation_ladder.py` skeleton with `EscalationContext`,
   `RungResult`, `EscalationStats`, and the Rung-1/Rung-6 rungs.
   Rung 1 is the existing gap-filler logic, factored into a callable.
   Rung 6 is the always-claims terminator that makes the coverage
   invariant hold.
4. `TACT_LADDER_*` and `TACT_QUARANTINE_HALLUCINATIONS` config keys.
5. Unit tests for the quarantine path, ladder ordering, and the
   coverage invariant after Rung 6.

Rungs 2, 3, 4, 5, the PANNs/wav2vec2 integrations, the fixture pack,
and the full pipeline.py rewire are deferred to follow-up commits on
the same Phase 2 PR. The ledger's `from_segments` post-hoc emission
block in pipeline.py stays in place this commit — replacing it with
the ladder integration is a coordinated change that needs the rest
of the rungs first.
