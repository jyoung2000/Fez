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

---

## Phase 3 design decisions

**ROVER vs alignment-then-pick.** Phase 3 implements word-level ROVER
in `transcription_reconciler.py` rather than full Levenshtein
alignment. ROVER assumes per-word timestamps, which Whisper provides
when `word_timestamps=True` is set on `model.transcribe`. Both passes
use that flag in the existing pipeline so the assumption holds.
Levenshtein alignment would handle the case where one pass merges or
splits words differently from the other, but the additional accuracy
on Whisper-vs-Whisper agreement was not measurable in the design-doc
references. The reconciler also has a uniform-distribution fallback
when one pass lacks word timestamps — degraded, but never crashes.

**Punctuation normalization.** Word equality uses `lowercase + alnum-
only` for the comparison key but keeps the original form for emission.
"hello," and "hello" agree; "thanks" and "thanks!" agree. This avoids
inflating the disagreement count on cosmetic differences that would
flip the wrong word into a contested ROVER tiebreak.

**Distance-to-boundary tiebreak.** When two passes disagree, we pick
the word from the pass where the word is *furthest* from any chunk-
grid boundary. The grid for pass `p` is at `offset_p + k * 30` for
every integer k. Implemented in `_distance_to_boundary` as
`((midpoint - offset) % 30)` then `min(rel, 30 - rel)`. For a non-
chunked pass (Parakeet), we set offset=0 and rely on confidence
ordering — Parakeet's TDT decoder doesn't have a fixed 30-s window so
"distance to boundary" isn't meaningful for it. This is also why the
chunk_grid_offsets dict in `reconcile_n_passes` defaults to 0 for
unspecified passes.

**Re-segmentation.** After reconciliation, words are re-grouped using
the primary pass's segment boundaries via midpoint assignment. Words
falling outside every primary segment land in synthetic single-word
segments — covers the case where the offset pass found a word the
primary missed entirely.

**Worker offset trim ordering.** The `--input-offset-sec` flag in
`whisper_worker.py` runs ffmpeg trim *after* the loudnorm /
noise-gate preprocessing chain. This way both passes see identical
preprocessed audio levels (loudnorm measures from the full file).
Doing the trim before preprocessing would re-measure loudness on a
shorter sample and produce slightly different gain — an unnecessary
source of pass divergence.

**No pipeline orchestration in this commit.** `transcribe_audio_subprocess`
gains the `offset_sec` kwarg but `pipeline.py` is unchanged; the legacy
single-pass path stays in place. Wiring the second pass + reconciler
into the pipeline is a same-PR follow-up that depends on Phase 2's
ladder integration so the reconciled output flows into the ladder
instead of into the legacy gap-filler.

## Phase 4 design decisions

**Subprocess isolation, not in-process.** Mirrors `whisper_worker.py`.
NeMo's import alone is ~1 GB; even with consensus disabled, accidentally
importing it in the parent process would balloon every job's memory
footprint. The `test_consensus_disabled_no_import` acceptance test
spawns a fresh interpreter and asserts `'nemo' not in sys.modules` —
this is the structural contract that makes the opt-in flag meaningful.

**ConsensusGateResult is structured, not a bool.** Three production
ways for the gate to refuse: disabled, vram_probe_failed,
insufficient_vram_NNNNmb_lt_NNNNmb. Each is logged as info (not
warning) — declining gracefully on a 4 GB GPU is the expected
behavior, not a failure mode. The structured reason flows into
`RungResult.notes` so the ladder stats show why Rung 3 didn't run for
the post-hoc audit.

**NeMo API drift tolerance.** `_segments_from_nemo` in
`parakeet_worker.py` handles three NeMo API generations:
`hyp.timestamp["word"]` (newer), `hyp.word_timings` (older), and
dict-shaped hypotheses (some intermediate versions). All three round-
trip through the same JSON schema as `whisper_worker.py` so the
reconciler doesn't care which Parakeet version produced the segments.

**Rung 3 is per-gap, not whole-audio.** The ladder calls Parakeet on a
sliced audio range corresponding to one uncovered/quarantined
interval, not on the full file. The full-file Parakeet pass for
three-way reconciliation against primary + offset is a pipeline-level
orchestration that hasn't landed yet. Rung 3 inside the ladder works
today as a focused per-gap consensus pass and will continue to work
when the pipeline adds the full-file consensus pass — the two paths
are complementary, not redundant.

**N-way reconciler shipped in Phase 3.** `reconcile_n_passes` was
written in Phase 3 with N=2 callers, anticipating the Phase 4
requirement. No reconciler changes were needed in Phase 4 — just a
config key and the orchestrator that drives three-pass calls.

## Phase 5 design decisions

**Translation as a paired ledger, not parallel transcription.** The
key insight from the design doc §10: translation re-uses the source
ledger's span structure. `to_translation_ledger` copies time bounds
and event/silence/unintelligible spans verbatim, then stages every
covered_speech span as `low_confidence` with the
`awaiting_translation` flag. The track runner fills those in. Coverage
invariant in the target language: every awaiting_translation flag is
gone after the track completes (replaced either by translated text or
by `[untranslatable]`).

**Failed-translation confidence ≠ 0.** First pass at the implementation
had failed translations claim at confidence=0.0, which tied with the
awaiting_translation incumbent (also conf 0.0) and lost the overlap
resolution. Bumped to 0.01 so it strictly supersedes, while still
reporting essentially-no-useful-content. Caught by
`test_failed_backend_emits_untranslatable`.

**Backends are pluggable + lazy.** `register_backend` is a decorator
that adds to a module-level dict; tests inject mock backends and pop
them in `finally:` blocks. Production backends register themselves on
import. The default backend is `whisper_passthrough` — when the source
ledger came from `task=translate` Whisper, the "translation" already
happened upstream and we just surface it. NLLB and Seamless are
opt-in backends to be registered when those models are available.

**Semantic reconciliation is opt-in via LaBSE.** The
`sentence-transformers` dep is listed in requirements.txt as a
capability dep but never imported unless
`TACT_TRANSLATION_CONSENSUS_ENABLED=True` AND multiple backends are
registered. When LaBSE isn't installed,
`reconcile_translations` falls back to "highest-confidence candidate
wins, status=single". Same graceful-degradation pattern as the
Phase 4 NeMo gate.

**Forced alignment (Rung 4) is skipped in the translation track.**
Phonemes don't transfer cross-language. Rungs 1, 2, 3, 5, 6 still
apply but use translation-capable models. Rung 6's terminator
becomes `[untranslatable]` instead of `[unintelligible]`.

## Pipeline integration scope still pending

After Phases 1–5 commit, the following pipeline integration work is
still needed before the user-facing `coverage_ratio == 1.000`
contract holds:

1. Replace `pipeline.py` lines 2607–2647 (existing gap-fill block)
   with a call to `escalate_uncovered_intervals_sync` plus the
   pre-claimed quarantined regions from `_last_quarantined_segments`.
2. Drop the Phase 1 post-hoc ledger emission block (now redundant —
   ledger is the source of truth).
3. Wire the disjoint-offset second pass: serial Whisper run with
   `offset_sec=15.0`, then `reconcile_passes` before the ladder.
4. Wire the consensus pass when `TACT_CONSENSUS_ENABLED=True`:
   third Whisper-equivalent pass via `transcribe_with_parakeet_subprocess`,
   then `reconcile_n_passes` instead of `reconcile_passes`.
5. Wire the translation track when `task=="translate"` or
   `TACT_TRANSLATION_TRACK_ENABLED=True`: after the source ledger
   reaches 1.000, call `run_translation_track` and persist
   `to_paired_report_dict` instead of `to_report_dict`.
6. Implement Rungs 2, 4, 5 (alt-checkpoint Whisper, wav2vec2 forced
   alignment, PANNs event classifier) — currently stubs returning
   empty `RungResult`.
7. Build the fixture pack under `backend/tests/fixtures/coverage/`
   and the integration tests asserting `coverage_ratio == 1.0` per
   fixture.

The ordering of these pieces follows the Phase 2 → 3 → 4 → 5 PR
sequence; each one merges the corresponding pipeline-integration
slice. The data structures, types, and unit-tested algorithms are all
in place — the remaining work is wiring them into the existing
pipeline orchestration.

---

## Pipeline wire-up plan

This is the operationalization of the seven items above. Every item
is plumbing — no new architectural decisions, no new TACT modules
beyond the two needed for Rungs 4 and 5 (which the previous prompts
explicitly anticipated as new files).

### Pre-flight grep result

`grep -rn "TACT\|coverage_ledger\|escalation_ladder\|reconcile_passes\|run_translation_track" backend/services/pipeline.py`

Only matches: the Phase 1 ledger emission block at lines 2649–2704
and the legacy gap-fill at lines 2607–2647. Confirmed nothing else is
wired yet.

### API verifications before code edits

* `transcribe_audio_slice_subprocess(slice_path, *, model_name=None, ...)` —
  accepts a `model_name` kwarg (transcription.py:1011). Used by Rung 2.
* `transcribe_audio_subprocess(..., offset_sec=0.0)` — kwarg landed in
  Phase 3 (transcription.py:489). Used by the disjoint-offset pass.
* `_can_run_consensus()` exists in parakeet_transcriber.py:45 as a
  private helper. Will expose a public alias `can_run_consensus` per
  the prompt's instruction.
* `reconcile_passes` does not currently take a `boundary_window_sec`
  kwarg — only `chunk_window_sec` and `align_tolerance_sec`. The
  prompt's Item 4 example passes `boundary_window_sec`; will skip
  passing it from the pipeline since the reconciler doesn't use it
  yet, rather than adding a no-op kwarg to a stable signature.
* `EscalationContext` is a `@dataclass` so `dataclasses.replace` works
  for per-interval neighbor-text rewriting.

### The seven items, mapped

| Item | What | Files | Pipeline lines |
|------|------|-------|----------------|
| 1a | Rung 2 implementation | `escalation_ladder.py:315` (replace stub) | n/a |
| 1b | Rung 4 implementation | new `forced_alignment.py` + `escalation_ladder.py:418` | n/a |
| 1c | Rung 5 implementation | new `non_speech_events.py` + `escalation_ladder.py:427` | n/a |
| 2 | Per-interval neighbor text | `escalation_ladder.py:497` (orchestrator loop) | n/a |
| 3 | Replace gap-fill + Phase 1 ledger blocks with ladder | `pipeline.py:2607-2704` | 2607–2704 (entire region replaced) |
| 4 | Disjoint-offset pass + reconciler | `pipeline.py` (before Item 3 block); `transcription_reconciler.py` (add `_last_reconciliation_stats`) | new code immediately before Item 3 region |
| 5 | Consensus pass (opt-in) | same region as Item 4 | same |
| 6 | Translation track | inside Item 3 block, after `coverage_report` is built | inside the new Item 3 region |
| 7 | Fixture pack + integration test | new `backend/tests/fixtures/coverage/` and `backend/tests/test_tact_integration.py` | n/a |

### Items 4, 5, 6 collapse into the same `pipeline.py` region

The §1 prompt presents the disjoint-offset pass, the consensus pass,
and the translation track as three separate items, but in the
pipeline they're a single contiguous block of code: the multi-pass
collection feeding `reconcile_n_passes` (Items 4 + 5), then the
ledger build + ladder (Item 3), then the optional translation track
(Item 6). Implementing them in one commit avoids three rounds of
"now move this block before that block." One commit covers all four
of these (3 + 4 + 5 + 6).

### Rung 2 design pinning

* Selection rule: large-v3-turbo → large-v3; medium → large-v3-turbo
  (if free VRAM ≥ 3000 MB) else large-v3 (CPU int8); small → medium;
  anything else → decline.
* Module-level cache `_alt_checkpoint_cache` is `dict[str, dict]` mapping
  model name to `{"loaded_at": float, "model_name": str}` — actually
  there's no in-process model object to cache (the slice helper
  spawns a fresh subprocess each call), so the "cache" is really just
  a TTL guard that skips re-validating that an alternate is selectable.
  The Whisper subprocess itself reloads weights every call — that's
  the contract of `transcribe_audio_slice_subprocess`. The cache as
  described in the prompt is therefore largely cosmetic. I'll keep
  the lock + TTL structure for forward-compat with an in-process
  variant but acknowledge it's a minimal TTL gate today.

### Open ambiguities resolved

1. **`reconcile_passes` no `boundary_window_sec`.** Skip from
   pipeline call site rather than touch the reconciler signature.
   The kwarg is documented in the design doc but the implementation
   doesn't use it; adding a no-op kwarg violates "don't change
   public APIs."
2. **Translation track has no `target_language` on `JobResult`.**
   `backend/models.py` doesn't expose this field today. Adding it as
   `target_language: Optional[str] = None` is a one-line additive
   change to JobResult (Pydantic). Acceptable per the prompt's "if
   it's not there, add it" clause.
3. **PANNs vendoring in Dockerfile.** Out of scope for this commit
   — the rung gracefully declines (returns "other" with low
   confidence → falls through) when the model isn't available, so
   Rung 6 still catches every interval. The vendor step is a
   follow-up.
4. **Acceptance criterion (real-video coverage_ratio == 1.0)**
   cannot be verified in the current dev container — no GPU, no
   Whisper weights, no ffmpeg-decoded audio. The integration test in
   Item 7 uses synthesized fixtures plus mocking to prove the wiring
   is correct by construction; the real-video smoke test is a
   homelab step the user runs.

### Commit plan

1. `docs(tact)`: this section. (current commit)
2. `feat(tact)`: Rung 2 (`_rung_alt_whisper_checkpoint`) + tests.
3. `feat(tact)`: `forced_alignment.py` + Rung 4 + tests.
4. `feat(tact)`: `non_speech_events.py` + Rung 5 + tests.
5. `feat(tact)`: orchestrator per-interval neighbor text +
   reconciler stats accessor + tests.
6. `feat(tact)`: `pipeline.py` rewire (Items 3 + 4 + 5 + 6) +
   `JobResult.target_language` + `can_run_consensus` public alias.
7. `test(tact)`: fixture pack + `test_tact_integration.py`.

After the seven commits, push and add a "Pipeline wire-up complete"
section to this notes file with commit SHAs and the smoke-test
result placeholder for the user to fill.

---

## Pipeline wire-up complete

| # | Item | Commit |
|---|------|--------|
| 0 | Wire-up plan | `3e2fde6` |
| 1a | Rung 2 — alt-checkpoint Whisper | `3d27ec6` |
| 1b | Rung 4 — wav2vec2 forced alignment | `0cd89da` |
| 1c | Rung 5 — PANNs event classifier | `2088bd6` |
| 2 | Per-interval neighbor text + reconciler stats accessor | `cfc6901` |
| 3-6 | Pipeline.py rewire (ladder + offset + consensus + translation) | `ff4979f` |
| 7 | Fixture pack + `test_tact_integration.py` | (this commit) |

**Test count after wire-up:** 156 TACT-related tests passing.

### Real-video smoke test (homelab acceptance step)

The §4 single acceptance criterion ("on a real video,
`coverage_report["coverage_ratio"] == 1.0`") cannot be verified in
the dev container — no GPU, no Whisper model, no real audio.
Instead, the homelab runs:

```bash
python -m backend.cli transcribe --input some_30min_video.mp4 \
    --job-id smoke_test
python -c "
import json
from pathlib import Path
job = json.loads(Path('jobs/smoke_test/job.json').read_text())
cr = job['coverage_report']
src = cr.get('source', cr)  # paired-report shape vs single-ledger shape
total_ms = sum(src['status_distribution_ms'].values())
print('coverage_ratio:', src['coverage_ratio'])
print('voiced_coverage_ratio:', cr.get('voiced_coverage_ratio'))
print('total_ms:', total_ms,
      'audio_duration_ms:', src['audio_duration_ms'])
print('ladder_stats:', cr.get('ladder_stats'))
assert src['coverage_ratio'] == 1.0, 'coverage invariant violated'
"
```

Expected output structure:

```
coverage_ratio: 1.0
voiced_coverage_ratio: <typically 0.95+>
total_ms: <equals audio_duration_ms exactly>
ladder_stats: {
  'intervals_total': N,
  'intervals_resolved': N,
  'rung_resolutions': {'rung_1_relaxed_whisper': K1, 'rung_5_event_classifier': K5, ...},
  ...
}
```

If the assertion fires, look at `ladder_stats["intervals_resolved"]`
vs `intervals_total`. The most common cause is the orchestrator
skipping a catch-all interval — fix in the orchestrator (Item 2),
not the rungs.

### Deviations from the wire-up prompt

1. **Reconciler `boundary_window_sec` kwarg.** The prompt's example
   pipeline code passes `boundary_window_sec` to `reconcile_passes`,
   but the actual reconciler signature doesn't take that kwarg.
   Skipped from the pipeline call site rather than adding a no-op
   parameter. The reconciler's existing distance-to-boundary logic
   already does what `boundary_window_sec` was meant to gate.

2. **`JobResult.target_language` added; existing
   `subtitle_language` preserved.** The pipeline reads
   `target_language` first, then falls back to `subtitle_language`,
   then to `"en"`. This avoids breaking existing job.json files
   that only have `subtitle_language`.

3. **Quarantine recovery requires a higher rung than Rung 6.**
   The integration test
   (`test_quarantined_segments_re_presented_to_ladder`) uses a
   stub "replacer" rung at confidence 0.7 to demonstrate proper
   quarantine recovery. Rung 6's confidence=0.0 cannot supersede a
   typical quarantined claim (Whisper hallucinations carry
   non-trivial avg_logprob). In production, Rung 5 (PANNs event
   classifier) is the rung that actually replaces music/applause-
   region quarantined hallucinations with correct event tags.

### Follow-up work that surfaced during implementation

1. **PANNs CNN14 vendoring in Dockerfile.gpu** — the rung
   gracefully falls through when the model isn't available, but
   first-run downloads block the production pipeline ~30 s. Vendor
   step should be added to the GPU image build.

2. **Concurrent-job thread safety of module-level accessors** —
   `_last_quarantined_segments` (transcription.py) and
   `_last_reconciliation_stats` (transcription_reconciler.py) are
   module globals that race across simultaneous jobs. Single-job
   homelab use is the design center today. If concurrent jobs
   become a concern, refactor both into per-job dicts keyed on
   `job_id`. Documented as a known limitation.

3. **Real-audio integration tests** beyond the synthetic-WAV
   fixtures depend on TTS-generated speech samples. Festival or
   piper-tts could populate
   `backend/tests/fixtures/coverage/quiet_speech_60db.wav` etc.
   with actual speech — would catch regressions in
   reconciliation that synthetic tones can't surface.

4. **Coverage report payload size on long files.** A 90-min file
   with hundreds of source-pass entries can produce
   `coverage_report` >100 KB. Monitor on long-form test content;
   if it gets unwieldy, trim `confidence_histogram` resolution or
   summarize per-rung detail.

5. **Translation backends** beyond `whisper_passthrough` (NLLB,
   Seamless-M4T) need to be registered at startup time when the
   user opts into them. Out of scope for the wire-up commit; the
   plug-in mechanism (`register_backend` decorator) is in place.

The wire-up code paths are exercised by 156 unit/integration tests
that run in any environment. The end-to-end real-video smoke is the
final acceptance gate the homelab user runs.
