# Phase 3 — Confidence-weighted VLM / face fusion

## What changed

New module **`backend/services/vlm_fusion.py`**:

- `fuse_vlm_and_face(face_x, face_conf, vlm_x, vlm_conf)` — pure
  function returning a `FusionResult(subject_x, subject_confidence,
  fusion_source)`. Implements the rule table below.
- `apply_fusion_to_scene(scene, face_x, face_conf, vlm_x, vlm_conf)`
  — writes the fused result back onto a `SceneDescription`,
  populating `subject_x`, `precise_x`, `subject_confidence`,
  `vlm_confidence`, `face_confidence`, `fusion_source`.
- `fusion_mode()` / `fusion_enabled()` — read the `CLIPAI_VLM_FUSION`
  env var at call time. Default is `"hard"` (legacy behavior).

## Fusion rule

```
if face_conf >= 0.80:
    subject_x = face_x                       → "face_high_conf"
elif face_conf >= 0.40:
    w_face = (face_conf - 0.40) / 0.40
    w_vlm  = vlm_conf * (1 - w_face)
    if w_face + w_vlm < 1e-6:
        subject_x = face_x                   → "face_high_conf"
    else:
        subject_x = (face_x*w_face + vlm_x*w_vlm) / (w_face+w_vlm)
                                             → "blended"
elif vlm_conf >= 0.30:
    subject_x = vlm_x                        → "vlm_only"
else:
    subject_x = 50                           → "fallback_center"
```

- The 0.80 boundary is inclusive (face_conf == 0.80 → face_high_conf).
- The 0.40 boundary is inclusive at the bottom of the blend range.
- `vlm_conf >= 0.30` is the minimum for a VLM-only override; 0.29
  still falls through to `fallback_center`.
- Inputs are clamped to their valid ranges (face/vlm_x in [0,100],
  confidences in [0,1]).

## Env flag

`CLIPAI_VLM_FUSION`:

- `"hard"` (default) — **current production behavior preserved
  bit-for-bit.** The existing pipeline.py:2842 face-detector
  override path is unchanged. The module is importable but dormant.
- `"weighted"` — callers that consume `fusion_enabled()` switch to
  the new path.

## What did NOT change in Phase 3

- **pipeline.py:2842 is untouched.** The monster ~350-line dense
  face merge block is fragile enough that swapping the override
  for the fusion rule inline is a high-risk change. Phase 3 ships
  the fusion helper + tests so every caller can be migrated
  incrementally once Phase 4/5 ground-truth fixtures exist. The
  env flag gates the change; until `CLIPAI_VLM_FUSION=weighted`,
  all existing code paths run identically.
- **No float-x parity test for FFmpeg crop yet.** The pipeline
  already uses float `precise_x` (see `_merge_dense_face`),
  VideoEditor preview already handles floats, and the crop filter
  builder already rounds to int at the boundary. A dedicated
  `test_render_plan_float_x.py` parity run against `synthetic_*`
  fixtures is deferred to the Phase 5 QA module (which exercises
  the rendered output anyway).

## Tests

`backend/tests/test_vlm_fusion.py` — 17 tests, all passing:

Rule table (8 tests):
1. `test_fusion_high_face_conf_uses_face`
2. `test_fusion_face_at_exact_high_threshold_uses_face`
3. `test_fusion_uncertain_face_blends`
4. `test_fusion_blended_face_at_lower_threshold`
5. `test_fusion_blended_zero_vlm_conf_falls_back_to_face`
6. `test_fusion_no_face_vlm_takes_over`
7. `test_fusion_low_face_and_low_vlm_falls_back_center`
8. `test_fusion_vlm_just_below_minimum`

Input sanitation (3 tests):
9. `test_fusion_none_face_x_treated_as_zero`
10. `test_fusion_none_vlm_x_treated_as_center`
11. `test_fusion_confidence_clamped_to_unit_interval`

Scene writeback (2 tests):
12. `test_apply_fusion_writes_all_scene_fields`
13. `test_apply_fusion_blended_writes_float_precise_x`

Env flag (4 tests):
14. `test_fusion_default_mode_is_hard`
15. `test_fusion_weighted_mode_enables`
16. `test_fusion_mode_case_insensitive`
17. `test_fusion_disabled_preserves_legacy_behavior`

## Promotion criteria (for a future change that flips the default)

Per the prompt: promote `CLIPAI_VLM_FUSION=weighted` to the default
ONLY if:

1. `validate_v2_phases.py --quick` passes.
2. Required-region miss rate on `vlog_walk_and_talk` improves by
   ≥ 5 percentage points vs legacy hard-override.

Phase 3 leaves the flag at `hard`. The promotion pass is deferred
until real-content fixtures can be measured (Phase 4 adaptive
sampling enables proper per-fixture telemetry; Phase 5 crop QA
lets us quantify downstream impact).
