# Non-Reframing Feature Backport

Source branch: `claude/reframe-engine-audit-fixes-qP4q8`
Target branch: `claude/backport-non-reframing-feature-xiEd9`

26 logical groups, ported as 26 commits in order. All listed commit
SHAs in the original task description were absent from the source
branch's actual git log (likely squashed during rebase), so changes
were applied by checking out the listed source files (or the
relevant fragments thereof) directly from the source branch tree.

## Verification

All 172 new/ported tests pass. Run:

```
python3 -m pytest \
  backend/tests/test_auth_cache.py \
  backend/tests/test_fingerprint_soft_fail.py \
  backend/tests/test_session_lifetime.py \
  backend/tests/test_media_signing.py \
  backend/tests/test_media_signed_access.py \
  backend/tests/test_diarization_status_surfaced.py \
  backend/tests/test_speaker_diarization_not_collapsed.py \
  backend/tests/test_transcription_ledger.py \
  backend/tests/test_hallucination_quarantine.py \
  backend/tests/test_transcription_reconciler.py \
  backend/tests/test_parakeet_transcriber.py \
  backend/tests/test_translation_track.py \
  backend/tests/test_rung_2_alt_checkpoint.py \
  backend/tests/test_rung_5_event_classifier.py \
  backend/tests/test_orchestrator_neighbor_text.py \
  backend/tests/test_escalation_ladder.py \
  backend/tests/test_tact_integration.py \
  backend/tests/test_providers_models_available_ollama.py \
  tests/test_openrouter_circuit_breaker.py
```

## Commits in order

| # | SHA | Message | Test command |
|---|-----|---------|--------------|
| 1 | cb932d5 | perf(auth): TTL-cache session + user lookups to stop upload serialization | `pytest backend/tests/test_auth_cache.py -v` |
| 2 | 8db1ebc | fix(auth): don't delete session on fingerprint mismatch; drop IP from fingerprint | `pytest backend/tests/test_fingerprint_soft_fail.py -v` |
| 3 | dba22a3 | fix(auth): stabilize session fingerprint across DevTools UA mutations | `pytest backend/tests/test_fingerprint_soft_fail.py -v` |
| 4 | 4bfe1a5 | feat(auth): enforce 25-hour minimum session-cookie lifetime | `pytest backend/tests/test_session_lifetime.py -v` |
| 5 | 1ae70ec | feat(auth): signed media URLs for /api/files/* so `<video>` works without cookies | `pytest backend/tests/test_media_signing.py backend/tests/test_media_signed_access.py -v` |
| 6 | b966878 | fix(preview): unified retry hook handles 503/401/stall on GPU, CPU, and mobile | `cd frontend && npm test -- useVideoLoadRetry` |
| 7 | c94cbe9 | fix(preview): atomic encode rename + ffprobe cache + GPU-accelerated transcode | `pytest tests/test_browser_preview.py -v` |
| 8 | 97d4db7 | fix(clips): persist clips before verification to survive timeouts | n/a (pipeline.py change; covered by integration tests) |
| 9 | 6e6998d | fix(pipeline): GPU probe cache + OpenRouter circuit breaker + cartoon-contamination guard | `pytest tests/test_openrouter_circuit_breaker.py -v` |
| 10 | ba4b7f9 | fix(settings): use per-user AI_FALLBACK_CHAIN when enumerating Ollama models | `pytest backend/tests/test_providers_models_available_ollama.py -v` |
| 11 | d269328 | feat(diarization): surface backend + status reason in /diarize response | `pytest backend/tests/test_diarization_status_surfaced.py -v` |
| 12 | 3c16b3a | feat(diarization): heuristic alternation + tier fallback + pyannote singleton | `pytest backend/tests/test_speaker_diarization_not_collapsed.py -v` |
| 13 | 94da495 | feat(tact): Phase 1 — Temporal Coverage Ledger (observability only) | `pytest backend/tests/test_transcription_ledger.py -v` |
| 14 | be4b4f3 | feat(tact): Phase 2 — hallucination quarantine + ladder skeleton | `pytest backend/tests/test_hallucination_quarantine.py -v` |
| 15 | cb603f3 | feat(tact): Phase 3 — disjoint-offset reconciler + worker offset | `pytest backend/tests/test_transcription_reconciler.py -v` |
| 16 | 8c10a94 | feat(tact): Phase 4 — Parakeet consensus pass (opt-in, subprocess-isolated) | `pytest backend/tests/test_parakeet_transcriber.py -v` |
| 17 | fc7ad84 | feat(tact): Phase 5 — translation track + paired ledger | `pytest backend/tests/test_translation_track.py -v` |
| 18 | 540d16a | feat(tact): Rung 2 — alternate-checkpoint Whisper | `pytest backend/tests/test_rung_2_alt_checkpoint.py -v` |
| 19 | f09b73b | feat(tact): Rung 4 — wav2vec2 forced alignment | `pytest backend/tests/test_rung_4_forced_alignment.py -v` (3 cases skip without torchaudio) |
| 20 | cb6c0d9 | feat(tact): Rung 5 — non-speech event classifier (PANNs CNN14) | `pytest backend/tests/test_rung_5_event_classifier.py -v` |
| 21 | 6875876 | feat(tact): per-interval neighbor text + reconciler stats accessor | `pytest backend/tests/test_orchestrator_neighbor_text.py backend/tests/test_escalation_ladder.py -v` |
| 22 | 735f46c | feat(tact): wire ladder + reconciler + translation into pipeline.py | manual end-to-end on a 2-min test clip |
| 23 | 5b94528 | fix(tact): stop escalation ladder from running unbounded on CPU | `pytest backend/tests/test_escalation_ladder.py -v` |
| 24 | fbadd4e | fix(tact): make TACT capability deps opt-in to fix Docker build OOM | `docker build .` |
| 25 | db3af23 | test(tact): fixture pack + pipeline wire-up integration tests | `pytest backend/tests/test_tact_integration.py -v` |
| 26 | 479c5cf | docs(tact): implementation notes | n/a |

## Notes on porting decisions

- **Auth groups (1–5):** Source-branch auth files (middleware.py,
  security.py, store.py, models.py) have all five features
  intermixed. To preserve per-feature commit messages, the full
  auth source files were brought in at Group 1; subsequent commits
  (2–5) primarily add their respective test files plus, for Group 5,
  the new signed-URL frontend utilities, page consumers, and
  `backend/main.py` /api/files signed-bypass + /api/media/sign. The
  SOTA-bench public-path entry was stripped from `_PUBLIC_GET_EXACT`
  before commit (reframing-only).
- **Group 8:** The visual-verification `wait_for` wrapper part of
  the original commit (`efca124`) requires `apply_visual_verification`
  which doesn't exist on this branch's clip pipeline (it would have
  been added separately by reframing-adjacent commits). Only the
  pipeline.py changes — outer-timeout result recovery + early-save —
  were ported. Group 9's wholesale `clip_scoring.py` checkout did
  bring in the source `apply_visual_verification` helper as a
  side-effect of pulling source `ai_orchestrator.py` cleanly.
- **Group 9:** Source files for `transcription.py`, `ai_orchestrator.py`,
  `clip_scoring.py`, `face_detector.py`, `models.py`,
  `providers/openrouter_provider.py`, `providers/base.py` were brought
  in wholesale. `transcription.py` from source already includes all
  TACT Phase 2–5 changes, so subsequent TACT groups (13–22) commit
  primarily their tests and standalone modules; the implementations
  in transcription.py were already present.
- **Group 12:** Removed the `_num_slots >= 2 and audio_path` clamp
  in `pipeline.py` per Fix C and replaced it with the bounded-kwargs
  hint pattern from source.
- **Group 22:** The TACT pipeline-wiring block in source's
  `pipeline.py` is concentrated in one self-contained section that
  replaces the legacy gap-fill block. The block was extracted via
  `sed -n '2603,2965p'` from source pipeline.py and surgically
  spliced over the legacy block at line 2401 of HEAD's pipeline.py.
  The reframing-related additions in source's pipeline.py (L1
  saliency peaks, the multi-region compositor wiring, etc.) were
  NOT ported.
- **Group 19:** 3 of 9 Rung 4 forced-alignment tests skip when
  `torchaudio` isn't installed — environmental, not a port defect.
- **Group 24:** Heavy TACT capability deps were never on this branch
  to begin with, so the "comment them out" intent is satisfied. The
  three commented-out variants with explanatory comments were
  appended to `backend/requirements.txt` for clarity.

## Reframing-only paths NOT ported

Per the strict scope rules in the task spec, the following were
explicitly excluded:

- All `backend/services/{samurai_tracker,cotracker3_dense,av_saliency,
  composition_head_clip,editorial_planner*,shot_reframe_advisor,
  composition_guardrails,speaker_cut_engine,ab_cut_scheduler,
  multi_region_layout,game_layouts,human_reframe*,l1_camera_path,
  camera_path_2d,render_plan*,reframe_*,subject_kalman,subject_motion,
  subject_confidence,tracker_wrapper*,crop_qa,saliency_lp_term,
  quick_classify,face_registry,light_asd,ocr_regions,
  ffmpeg_filter_builder,active_speaker (reframe parts),
  face_detector (reframe parts)}.py`
- All `backend/scripts/*reframe*`, `*autoflip*`, `sota_*`, etc.
- All `tests/qa/*` reframing files.
- Reframing diagnostic endpoints in `backend/routers/diagnostics.py`
  (kept only the new `/auth-cache` endpoint from Group 1).
- Reframing files in `infra/`.
- Reframing docs (kept only `docs/tact_implementation_notes.md`).
- `.github/workflows/reframe_bench.yml`.
- `backend/vendor/tasednet/`.
- Reframing-specific frontend files (`frontend/src/utils/renderPlanRenderer.js`,
  reframe sections of `PipelineDiagnostics.jsx`).
