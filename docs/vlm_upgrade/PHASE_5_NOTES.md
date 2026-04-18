# Phase 5 — Editorial crop QA pass

## What changed

New module **`backend/services/crop_qa.py`**:

- `CropQualitySample` dataclass — one VLM-scored output frame.
  Fields: `head_in_frame`, `awkward_crop`,
  `subject_partially_off_frame`, `dead_space_dominant`,
  `quality_score` (0-10).
- `CropQualityReport` dataclass — aggregate for a segment.
  `samples`, `aggregate_score`, `triggered_recovery`,
  `any_subject_off_frame`, `any_awkward_crop` properties.
- `score_crop_quality(segment_index, sample_timestamps, vlm_scorer)`
  — the main entry point. Takes an injected `vlm_scorer` callable
  so the module is testable without opencv / ffmpeg / httpx.
- `decide_recovery(report)` — pure function mapping a report to a
  recovery strategy string.
- `apply_recovery_params(strategy, dead_zone_px, lambda2,
  crop_padding_frac)` — returns tuned solver parameters per
  strategy. No strategy touches the L1 solver math directly.
- `crop_qa_enabled()` — reads `CLIPAI_CROP_QA` env var (default OFF).

New prompt **`DEFAULT_CROP_QA_PROMPT`** in
`backend/services/prompts.py` matches the Phase 5 spec exactly.

## Recovery rule table

| Condition | Recovery |
|---|---|
| no samples | None |
| aggregate < 5 | `safety_center` |
| aggregate < 7 AND any `subject_partially_off_frame` | `loose_dead_zone` |
| aggregate < 7 AND any `awkward_crop` | `padding` |
| aggregate < 7 (no specific failure fingerprint) | `loose_dead_zone` (default) |
| aggregate ≥ 7 | None (accept crop) |

## Solver param tuning

- `loose_dead_zone` → `dead_zone_px *= 1.5`, `lambda2 *= 0.5`
- `padding` → `crop_padding_frac += 0.05`
- `safety_center` → caller forces a centered crop; no solver tuning.

## Env flag

`CLIPAI_CROP_QA` — default OFF. The module is importable and
callable in unit tests regardless; only the
`backend.services.clip_exporter` integration (the post-render
hook) checks `crop_qa_enabled()`.

## What did NOT change in Phase 5

- **`backend/services/clip_exporter.py` is not yet wired.** The
  post-render hook that calls `score_crop_quality` with a real
  ffmpeg-frame-extract + VLM scorer is a separate integration
  step. The scoring + decision + param-tuning logic is complete,
  tested, and ready for the hook to import. Doing the export
  integration in-place requires opencv + a real VLM round-trip,
  which is better done once Phase 4's adaptive sampling ships so
  the cost budget can be verified in one measurement pass.
- **No real-content cost measurement.** Per the prompt, promoting
  `CLIPAI_CROP_QA` to default ON requires ≤ 3% of segments
  triggering recovery on `tests/real_content/` fixtures AND QA
  tokens < 5% of total VLM tokens. Neither can be measured until
  the clip_exporter hook is in place.

## Tests

`backend/tests/test_crop_qa.py` — 15 tests, all passing:

Aggregate / acceptance (6):
1. `test_qa_perfect_crop_no_recovery`
2. `test_qa_subject_off_frame_triggers_loose_dead_zone`
3. `test_qa_awkward_crop_triggers_padding`
4. `test_qa_severe_failure_falls_to_safety_center`
5. `test_qa_borderline_7_is_accepted`
6. `test_qa_weak_without_specific_failure_defaults_to_loose_dead_zone`

Decide recovery edges (1):
7. `test_decide_recovery_threshold_edges`

apply_recovery_params (4):
8. `test_apply_recovery_params_loose_dead_zone`
9. `test_apply_recovery_params_padding`
10. `test_apply_recovery_params_safety_center_passthrough`
11. `test_apply_recovery_params_unknown_returns_inputs`

Env flag + threshold constants (4):
12. `test_qa_default_flag_off`
13. `test_qa_flag_on`
14. `test_qa_disabled_no_calls`
15. `test_threshold_constants_match_spec`
