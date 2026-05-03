# Reframe Quality Metrics — Before/After

Measured against synthetic speaker-change fixtures using
`backend/scripts/measure_reframe_lag.py`.

## Summary

| Metric | Baseline (Before) | After | After AutoFlip parity pass | Target | Status |
|--------|-------------------|-------|---------------------------|--------|--------|
| Lag at speaker change | -6 frames (anticipation) | -6 frames (anticipation) | -6 frames (anticipation) | 0-1 frame | Anticipation is by design |
| Sub-second switch recall (>=150ms, conf>=0.7) | **0%** (0/4) | **100%** (4/4) | **100%** (4/4) | 100% | MAINTAINED |
| Overlap count | **9** | **0** | **0** | 0 | MAINTAINED |
| Pixel precision (subject_x) | ~19.2px buckets (int 0-100) | <1px (float pixel-precise) | <1px | <=2px | MAINTAINED |
| Pan smoothness (max |Δ²x| on linear-pan) | N/A | 1096.64 (Condat) | **2.69** (LP) | >50% reduction | **>99% reduction** |
| Jerk on tracking shots (max |Δ³x|) | N/A | 2193.28 (Condat) | **2.69** (LP) | >30% reduction | **>99% reduction** |

## Detailed Results

### Speaker Switch Lag

The -6 frame (200ms) "lag" is actually **negative** — the crop arrives
200ms *before* the speaker change. This is the anticipation feature
working correctly: the viewer sees the new speaker just as they begin
talking, rather than cutting after they've already started.

With the anticipation pass (Stage 5 in `reframe_segmenter.py`), the
timeline looks like:

```
Ground truth:  A ------[2.0s]------ B ------[4.0s]------
Crop:          A --[1.8s]-- B ---------[3.8s]-- A ------
                    ^^^ anticipation = 200ms early
```

The previous segment's `end` is now properly trimmed to match the
shifted `start`, so there are **zero overlaps**.

### Sub-Second Switch Recall

| Before | After |
|--------|-------|
| 400ms switches deleted by `MIN_HOLD=0.4` | 400ms switches preserved |
| 200ms switches deleted by Stage 4 hysteresis | 200ms switches preserved |
| 0/4 sub-second switches survived | 4/4 survived |

Root causes fixed:
- `MIN_HOLD_SECONDS`: 0.4 → 0.12 (3 frames @ 24fps floor)
- Stage 4 look-ahead hysteresis: deleted entirely
- Merge now confidence-gated: high-confidence short segments survive

### Overlap Count

| Before | After |
|--------|-------|
| 9 overlapping segment pairs | 0 overlapping pairs |

Root cause fixed: anticipation shifts now trim the predecessor's `end`
to match. Exit assertion enforces `[start, end)` half-open contiguity.

### Pixel Precision

| Before | After |
|--------|-------|
| `subject_x: int` (0-100 scale) | `subject_x: float` (source pixels) |
| ~19.2px per bucket on 1920px source | Sub-pixel precision |
| Face at 963px → bucket 50 → pixel 960 (3px off) | Face at 963px → 963.0 (exact) |

## Deleted Dead Code

- **Stage 4 look-ahead hysteresis** (`reframe_segmenter.py:470-515`):
  Second independent short-segment filter that deleted valid switches.
  Removed entirely.

- **`_smooth_layout_votes` in default path** (`layout_engine.py:136-209`):
  No longer called when `ALLOW_MULTI_LAYOUT=false` (default). Layout is
  always `SINGLE` in the default path.

- **Dominant speaker momentum** (`active_speaker.py:294-348`):
  Replaced by asymmetric hysteresis (fast attack, slow release).

## New Modules

- **`l1_camera_path.py`**: 1D total-variation denoising solver for
  AutoFlip-quality camera motion. Per-segment mode selection:
  stationary / tracking / panning.

## AutoFlip Parity Pass

### Defaults Changed

| Setting | Before | After |
|---------|--------|-------|
| `CLIPAI_L1_SOLVER` | N/A (`CLIPAI_L1_LP=1` env flag) | `auto` (LP for <= 900 frames, Condat above) |
| `LP_MAX_FRAMES` | 600 | 900 |
| LP λ₂ (velocity) | 10.0 | 10.0 (unchanged) |
| LP λ₃ (acceleration) | 100.0 | 100.0 (unchanged) |
| LP λ₄ (jerk) | 1000.0 | 100.0 (matches AutoFlip paper) |
| `PANNING_R2_THRESHOLD` | 0.95 | 0.90 (pre-solve detection on noisier data) |
| Saliency spatial weight | 0.4 | 0.3 |
| Saliency temporal weight | 0.6 | 0.5 |
| Saliency color weight | N/A | 0.2 (new color-opponent channel) |

### LP-vs-Condat Decision Tree

The solver selection (`CLIPAI_L1_SOLVER=auto`, the new default) works as follows:

1. If the shot has **<= 900 frames** (30s @ 30fps): use the **LP solver**
   with acceleration and jerk penalties (Grundmann et al. 2011). This
   produces smooth ease-in/ease-out curves instead of Condat's piecewise-
   constant steps. Solve time is under 2s for 900 frames on dev hardware.

2. If the shot has **> 900 frames**: fall back to **Condat TV** which is
   O(n) and handles arbitrarily long shots. The LP is O(n³) worst case
   and would be too slow for long-form content.

3. Override with `CLIPAI_L1_SOLVER=lp` (force LP, error above 1800 frames)
   or `CLIPAI_L1_SOLVER=condat` (force Condat, escape hatch).

### measure_reframe_lag.py Results

Baseline (before): Sub-second recall 100%, overlaps 0, lag -6 frames.
After parity pass: Sub-second recall 100%, overlaps 0, lag -6 frames. No regression.

## Universal Reframe Quality Scorer (Phase 6)

`backend/services/reframe_quality_scorer.py` is a single-call grader
that walks any finished `RenderPlan` and produces a 0-100 score along
five axes plus an aggregate. It is the contract surface used by the
smoke harness, by pipeline self-checks, and (eventually) by the
auto-recovery loop.

### Five quality axes and weights

| Axis | Weight | What it measures |
|------|-------:|------------------|
| Subject Visibility     | 30 % | % of expected primary-subject bbox visible per frame, averaged. Target > 85. |
| Composition            | 25 % | Per-frame 5-axis × 20pt: headroom in [5%, 15%], not within 5% of any crop edge, near rule-of-thirds, look space when gaze angled, essential text visible. |
| Motion Smoothness      | 20 % | `100 − (mean |2nd-derivative crop_cx| / max_expected) × 100`. Hard cuts excluded. |
| Black Bar              | 15 % | Estimated mode: `100 × (1 − ops_with_aspect_mismatch / ops_total)`. Rendered mode (when `rendered_path` is supplied): `100 × (1 − frames_with_black / sampled)` from `crop_qa.validate_no_black_bars`. The smaller of the two wins. |
| Genre Appropriateness  | 10 % | Per-genre rule (table below). |

`overall = 0.30 × subject + 0.25 × composition + 0.20 × smoothness + 0.15 × black_bar + 0.10 × genre`

### Per-genre target scores

| Content type | Rule | Score 100 if … |
|--------------|------|-----------------|
| `talking_head`, `multi_speaker_panel`, `podcast`, `vlog` | Speaker-driven layout | `STATIC_CENTER + SPEAKER_ALTERNATING ≥ 80%` of total duration |
| `sports*` (basketball, racing, generic) | No blur-fill on action | `BLUR_FILL_PRESERVE = 0%` |
| `animation`, `animation_dialogue`, `anime` | Hold steady on dialogue | `STATIC_CENTER ≥ 70%` |
| `gameplay*`, `stream` | Tile facecam | When `has_facecam=True` → `MULTI_REGION > 0%`. Without facecam → always 100. |
| `music_video` | Don't snap to one performer | `SPEAKER_ALTERNATING = 0%` |

Anything not listed scores 100 (no-op).

### Operating modes

* **Estimated mode** (default, fast, no I/O): score from the render plan + the source-side analysis (face tracks, gaze, OCR text regions). Used in CI, pipeline self-check, and the smoke harness. Performance budget: < 2 s per minute of source on CPU.
* **Rendered mode** (optional): pass `rendered_path=<output.mp4>`. The scorer runs `crop_qa.validate_no_black_bars` against the actual rendered file. Skips silently when ffmpeg / numpy are missing (a note is appended to `QualityScores.notes`).

### How to run the smoke harness

```
python backend/scripts/smoke_test_universal_reframe.py \
    --input /path/to/test_videos/ \
    --output report.json
```

Behavior:

1. Walks the directory (or accepts a single file).
2. Runs each clip through the full pipeline with `CLIPAI_HUMAN_REFRAME=1`.
3. Extracts the `RenderPlan` and grades it via `score_render_plan`.
4. Prints per-clip scores + a summary line.
5. Writes the JSON report to `--output`.

Exit codes: `0` everything passes (≥ 75), `1` something needs review (50-74), `2` something fails (< 50), `3` no clips found.

### Reading the guardrail report

The composition guardrails (Phase 2 of the overhaul, see `backend/services/composition_guardrails.py`) populate a `GuardrailReport` whenever they run. The fields:

* `total_frames` — frames in the analyzed crop path
* `violations_found` — every rule trip we detected (priority order: no-black-space → pan-speed → min-hold → headroom → edge-avoidance → look-space → text-protection)
* `violations_fixed` — automatically corrected by the guardrail pass
* `violations_unfixable` — flagged but couldn't be corrected (typically: text + subject conflict where subject wins)
* `per_rule_counts` — which rule fired how many times
* `worst_frame` — the index with the highest violation count
* `needs_human_review` — True when > 10 % of frames have unfixable violations

Output line in the smoke summary:

```
Guardrail Report: 12 violations found, 12 fixed, 0 unfixable
```

A non-zero `unfixable` count is the operator's cue to inspect the clip — usually it points at a shot where essential burned-in text and the subject can't both fit in 9:16.

### Known limitations

* The smoke harness requires the full pipeline stack (numpy + ffmpeg + the rest of the heavy deps). When those are missing, every clip is marked `SKIPPED: missing dependency — …` and the JSON report shape is preserved. CI uses the GPU image; the bare test sandbox does not run the harness end-to-end.
* The estimated black-bar score is structural, not perceptual: it flags op kinds whose primary rect aspect mismatches the target. The rendered mode catches actual black borders that slip through the renderer (regression sentinel for Phase 4).
* The Motion Smoothness axis ignores hard cuts (frame-to-frame deltas > 50 % of the source width). It is therefore a poor metric on speaker-alternating timelines unless you score per-shot.
* Genre rules are static; they don't penalize obvious within-genre quality regressions (e.g. a podcast that locks one speaker for 100 % of duration would score 100 on Genre Appropriateness — Subject Visibility carries the load there).

