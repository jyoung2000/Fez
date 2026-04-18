# Phase 4 — Adaptive VLM frame sampling

## What changed

New module **`backend/services/adaptive_frame_sampler.py`**:

- `AdaptiveSamplingInputs` dataclass — duration, shot cuts, face
  confidence timeline, ASD margin timeline, clip candidate starts.
  Every field is optional; missing signals simply skip the
  corresponding rule.
- `compute_adaptive_frame_times(inputs)` — pure scheduler. Applies
  rules 1-4, dedupes within 0.2s, applies the stable-region quota
  floor, then caps total count at `max(60, duration_minutes * 4)`.
- `adaptive_sampling_enabled()` — reads
  `CLIPAI_ADAPTIVE_VLM_SAMPLING` from the environment. Default OFF.

## Adaptive rules

| # | Rule | Output |
|---|---|---|
| 1 | Shot-cut neighborhood | first frame of each shot + frame at +1.0s |
| 2 | Face-confidence dropout (> 0.5s window, conf < 0.6) | midpoint frame |
| 3 | ASD ambiguity (> 0.5s window, margin < 0.3) | midpoint frame |
| 4 | Clip candidate hook windows | start+{0.5, 1.5, 2.5}s |
| 5 | Stable-region quota floor | 1 frame per 30s minimum |

All rules are additive. Dedup tolerance is 0.2s. Cap is applied
after dedup; over-cap results are thinned by even-stride sampling
(preserving first and last) rather than random drop so a clip
boundary near a shot cut never loses its hook window.

## Env flag

`CLIPAI_ADAPTIVE_VLM_SAMPLING`:

- **Unset / empty / false** (default) — the uniform-stride frame
  list from the existing frame extractor runs bit-for-bit
  identically. The module is importable but dormant.
- **Truthy** (`1`, `true`, `yes`, `on`, case-insensitive) — call
  sites that check `adaptive_sampling_enabled()` switch to
  `compute_adaptive_frame_times`.

## What did NOT change in Phase 4

- **`backend/services/frame_extractor.py` is untouched.** The new
  scheduler is a pure function consuming timestamps. Converting
  the scheduled timestamps into `FrameData` objects still uses the
  existing extractor primitives; a thin wrapper that calls those
  primitives on the scheduled timestamps is the natural landing
  spot, but it should be added in the same commit that flips the
  default (promotion pass). Phase 4 delivers the scheduler + tests
  so the wrapper can be written against a locked-in contract.
- **No cost telemetry wiring yet.** The prompt calls for a
  comparison against the full `tests/real_content/` fixture set to
  verify the 1.5× budget cap and to confirm that every fixture's
  required-region miss rate improves or stays flat. That
  measurement needs real fixtures + a pipeline run, which is out
  of scope for a pure-module phase.

## Tests

`backend/tests/test_adaptive_frame_sampling.py` — 12 tests, all
passing:

1. `test_adaptive_includes_shot_cut_neighborhoods`
2. `test_adaptive_oversamples_low_face_conf_window`
3. `test_adaptive_hook_window_has_three_frames`
4. `test_adaptive_asd_ambiguity_window`
5. `test_adaptive_stable_region_floor`
6. `test_adaptive_total_capped_per_job`
7. `test_adaptive_cap_scales_with_duration`
8. `test_adaptive_dedupes_overlapping_rules`
9. `test_adaptive_disabled_flag_default`
10. `test_adaptive_enabled_flag_truthy_values`
11. `test_adaptive_empty_inputs_still_gets_floor`
12. `test_adaptive_zero_duration_returns_empty`

## Promotion criteria (deferred)

Per the prompt: promote `CLIPAI_ADAPTIVE_VLM_SAMPLING` to default-on
ONLY if on a full `tests/real_content/` run:

1. The adaptive path uses ≤ 1.5× the baseline VLM tokens.
2. Required-region miss rate improves or stays flat on every fixture.

Phase 4 leaves the flag OFF.
