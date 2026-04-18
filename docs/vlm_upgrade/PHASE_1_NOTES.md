# Phase 1 — Grounding-output prompt + schema

## What changed

1. **`backend/services/prompts.py`** — `DEFAULT_SUBJECT_TRACKING_PROMPT`
   rewritten to ask for a normalized `subject_box`, a
   `subject_confidence` float, a `secondary_subjects` list, and a
   structured `no_subject_reason`. The legacy 0-100 percentage prompt
   is preserved as `LEGACY_SUBJECT_TRACKING_PROMPT` for any provider
   whose backend model can't do grounding output (local Ollama,
   older free-tier models). `parse_scene_dict` handles both.

2. **`backend/models.py`** — `SceneDescription` gains six additive,
   default-valued fields:
   - `subject_box: Optional[list[float]]`
   - `subject_confidence: float = 0.0` — the **merged** confidence
     (populated by Phase 3 fusion; raw VLM goes in `vlm_confidence`).
   - `vlm_confidence: float = 0.0`
   - `face_confidence: float = 0.0`
   - `fusion_source: Optional[str]`
   - `secondary_subjects: list[dict] = []`
   - `no_subject_reason: Optional[str]`

   The legacy `subject_x: int` field is **preserved**. Every existing
   test that constructs `SceneDescription` from kwargs passes without
   change (see `test_scene_description_construction_back_compat`).

3. **`backend/services/providers/base.py`** — new centralized helpers:
   - `SCENE_JSON_SCHEMA_GROUNDED` — schema string for vision requests.
   - `_clamp_box(box)` — coerce + clamp + swap-coordinate-correct.
   - `_coerce_confidence(raw)` — 0-1 float coercion (accepts 0-100%).
   - `fill_box_from_legacy(subject_x)` — synthesize a 10%-wide
     vertical strip from a legacy 0-100 percentage.
   - `parse_scene_dict(d, frame_timestamp=None)` — returns a
     SceneDescription-ready kwargs dict. **Back-compat contract:**
     - grounded path: derives `subject_x` and `precise_x` from the
       bbox center so every legacy consumer works unchanged.
     - legacy path: synthesizes a box from `subject_x` and marks
       `vlm_confidence = 0.5` (legacy uncertainty default).
     - no-signal path: returns centered defaults with
       `no_subject_reason = "empty_frame"`.

## What did NOT change in Phase 1

- **No provider was modified.** The existing openrouter
  `analyze_frames` still uses its inline legacy parser. Phase 2 wires
  `parse_scene_dict` into the providers when it flips the PRESETS
  over to Qwen3-VL / Gemini 3 — those models are trained on
  grounding output, so that's the natural place to start consuming
  the new schema. Until Phase 2 lands, the old path runs unchanged.
- **No pipeline behavior changed.** `subject_x` stays an int, face
  detection still overrides at `pipeline.py:2842`. Phase 3 is where
  the unconditional override becomes a confidence-weighted blend.
- **No env flag was added.** Phase 1 is purely additive: new fields
  default to None/zero, new functions are opt-in.

## Tests

`backend/tests/test_scene_dict_grounded.py` — 10 tests, all passing:

1. `test_parse_scene_dict_grounded_box`
2. `test_parse_scene_dict_legacy_subject_x`
3. `test_parse_scene_dict_no_subject_reason`
4. `test_parse_scene_dict_invalid_box_bounds`
5. `test_parse_scene_dict_secondary_subjects_capped`
6. `test_parse_scene_dict_box_swapped_coords`
7. `test_parse_scene_dict_missing_box_and_legacy`
8. `test_fill_box_from_legacy_centered`
9. `test_fill_box_from_legacy_edge_clamped`
10. `test_scene_description_construction_back_compat`

## Back-compat verification

`SceneDescription(timestamp=..., description=..., importance_score=...,
thumbnail_path=..., subject_x=42)` works identically. All new fields
get their defaults. `test_clip_scoring.py` (22 tests, touches models)
stays green.

## Cost / budget impact

Zero. Phase 1 is schema / parser infrastructure only; no VLM calls
change shape until Phase 2 flips the providers over.
