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

## Stage 3 Importance Matrix (Blueprint v2 Phase 1)

The reframing stack weights five signals when building the per-pixel
importance map:

| Signal     | Source                               | When it dominates          |
|------------|--------------------------------------|----------------------------|
| `face`     | `face_detector.py` + `face_registry` | Talking-head / dialogue    |
| `saliency` | `saliency_tracker.py`                | Anime / documentary / B-roll |
| `motion`   | `optical_flow.py`                    | Sports / action / racing   |
| `depth`    | monocular-depth (future)             | Cinematic close-ups        |
| `object`   | `object_detector.py` (YOLO)          | Ball / car / product       |

Weights live in `backend/services/reframe_config.py` in the
`_IMPORTANCE_MATRIX` dict. One row per `ClipContentType`:

```python
_IMPORTANCE_MATRIX = {
    "talking_head":       ImportanceWeights(face=1.0, saliency=0.3, motion=0.1),
    "sports_racing":      ImportanceWeights(face=0.2, saliency=0.4, motion=0.6, object=1.0),
    "landscape":          ImportanceWeights(face=0.0, saliency=1.0, motion=0.2, depth=0.3),
    ...
}
```

### How to read the matrix

- Rows with `face=1.0` treat every detected face as a required region.
  Panels / podcasts / talking heads are the extreme case.
- Rows with `object=1.0` promote their genre's object class (ball in
  basketball, car in racing) so it outscores passive faces.
- `required_region_gain=1000.0` is the hard-constraint multiplier
  (blueprint's `H`). It must dominate the soft-weight sum by >= 100x;
  this is verified by `test_required_gain_dominates_soft_weights`.
- `passive_face_weight=0.55` is the score given to non-speaking faces
  in the same shot as an active speaker. Larger = softer demotion.

### How to tune the matrix

1. Inspect the current row: `GET /api/diagnostics/importance-matrix`
   returns the full matrix as JSON.
2. Change the desired row in `_IMPORTANCE_MATRIX` in
   `backend/services/reframe_config.py`. New fields go on
   `ImportanceWeights` (not hardcoded at the call site).
3. Run the parity test: `pytest
   backend/tests/test_importance_migration_parity.py -x` to confirm
   downstream weights still match.
4. Add a regression entry in `backend/tests/test_importance_matrix.py`
   if you're pinning a new numerical invariant.

### Where weights are consumed

- `required_regions.py` — `passive_face_weight` sets the score of
  passive same-shot faces (the previously hardcoded 0.55).
- `genre_refinements.py` — `importance.object * BASKETBALL_BALL_BOOST`
  and `importance.object * RACING_CAR_BOOST` drive the required-boost
  weights for ball / car detections.
- `content_type_config.py` — `get_face_weight(ct)` /
  `get_saliency_weight(ct)` / `get_motion_weight(ct)` /
  `get_object_weight(ct)` / `get_depth_weight(ct)` are public shims
  for callers that need a single weight without importing the whole
  config object.

### Migration status

Phase 1 migrated the two highest-leverage hardcoded constants
(`passive_face_weight` 0.55 and the ball/car boosts). Other legacy
values (e.g. 0.8 score for passive-no-speaker, 1.4 solver weight for
active speakers, containment bbox expansion factors) remain hardcoded
in `required_regions.py` and will migrate in later phases if they
prove to need per-genre tuning.
