# Master QA Harness — 2026 SOTA Reframing

This is the local-validation gate Claude Code Opus 4.7 (and any
worker doing follow-up work) runs **before pushing** to
`claude/clipai-sota-reframing-tOo41`.

## Quick run

```bash
# In the repo root, no GPU needed:
pytest tests/qa/ -v
```

Expected: ~98 unit tests, ~5 skipped (the skips are config tests
that need `pydantic_settings` — present in production, not in the
sandboxed CI).

## What each suite covers

| Suite                              | Tests | What it validates                    |
|------------------------------------|-------|--------------------------------------|
| `test_phase_a_samurai.py`          | 24    | OpenCV fallback, dispatcher routing, |
|                                    |       | GPU gate, motion-aware memory state, |
|                                    |       | synthetic occlusion fixture,         |
|                                    |       | identity_switch_count metric,        |
|                                    |       | config flag.                         |
| `test_phase_b_cotracker.py`        | 24    | Background grid seeding (with        |
|                                    |       | low-saliency masking), RANSAC inlier |
|                                    |       | translation, cumulative camera-      |
|                                    |       | motion estimation, Huber median,     |
|                                    |       | Kalman dense observer, LP camera-    |
|                                    |       | motion kwarg.                        |
| `test_phase_c_saliency.py`         | 25    | Peak NMS, audio energy 1Hz, OCR      |
|                                    |       | bbox conversion, saliency nudge      |
|                                    |       | (3 scenarios + bounds), soft-promote |
|                                    |       | peaks, saliency_in_crop_fraction,    |
|                                    |       | text_region_clipping_rate.           |
| `test_phase_d_clip_head.py`        | 14    | Hints vector shape + content-type    |
|                                    |       | one-hot, mocked CLIP head, heuristic |
|                                    |       | blend fallback, face_clipping_rate,  |
|                                    |       | training script wiring.              |
| `test_phase_e_editorial.py`        | 20    | Genre dispatch (10 templates),       |
|                                    |       | robust JSON parsing, cache hit/miss, |
|                                    |       | LP soft-prior integration, allowlist |
|                                    |       | removal, cache key stability,        |
|                                    |       | Phase E config defaults.             |

## Per-phase gating

Each Phase commit must satisfy:

1. The Phase's QA suite passes locally (`pytest tests/qa/test_phase_<x>_*.py -v`).
2. The previous Phases' suites still pass.
3. The full QA harness (`pytest tests/qa/`) still passes.

If any of those fails, the commit is NOT shippable.

## Homelab gate (separate from this harness)

The QA harness is mock-heavy by design — it runs without GPU /
torch / sam2 / cotracker / open_clip / paddle. The **homelab gate**
runs the actual model integrations against the real-content fixtures
and IS the second half of the validation.

To run the homelab bench (Unraid container):

```bash
# Phase A only
docker compose exec backend pytest tests/qa/test_phase_a_samurai.py -v
docker compose exec backend python -m \
  backend.scripts.compare_autoflip_vs_clipai \
  --manifest tests/real_content/manifest.json \
  --output /tmp/phase_a_results.md

# Cumulative through Phase E
docker compose exec backend env \
  CLIPAI_TRACKER_BACKEND=samurai \
  CLIPAI_DENSE_POINT_TRACKING=1 \
  CLIPAI_SALIENCY_ENABLED=1 \
  CLIPAI_COMPOSITION_HEAD=clip \
  CLIPAI_EDITORIAL_PLANNER=1 \
  CLIPAI_HUMAN_REFRAME_PIPELINE=1 \
  python -m backend.scripts.compare_autoflip_vs_clipai \
  --manifest tests/real_content/manifest.json \
  --output /tmp/phase_e_results.md \
  --json-out /tmp/phase_e_results.json
```

Pass criteria (homelab):

| Metric                         | Target                              |
|--------------------------------|-------------------------------------|
| `identity_switch_count`        | ≥ 50 % reduction on multi-speaker   |
|                                | fixtures vs Phase A baseline.       |
| `face_clipping_rate`           | ≥ 50 % reduction on every fixture.  |
| `saliency_in_crop_fraction`    | ≥ 0.80 on dynamic content (sports / |
|                                | music_video / anime).               |
| `cut_to_hold_ratio`            | Within ±10 % of MediaPipe AutoFlip  |
|                                | per clip.                           |
| Existing parity metrics        | No clip regresses by more than 5 %. |

A bench result that meets all of the above is shippable; anything
short is NOT shippable until the offending phase is fixed (or the
phase's flag is flipped OFF for one release).

## What to do if QA fails

* **One test fails** — read the failure, fix the underlying bug, re-run.
  Don't soften the assertion. Don't `pytest.skip` it.
* **Multiple tests fail** — likely a regression in a shared module
  (`autoflip_parity_metrics` or `tracker_wrapper`). Check the most
  recent commit's diff.
* **All Phase X tests fail** — the Phase X module either fails to
  import or has a syntax error. Check the import chain by hand:
  `python -c "import backend.services.<module>"`.

## What to do if the homelab bench fails

* If it's a NEW regression (vs the previous bench), the offending
  phase is NOT shippable — fix or flip its flag OFF.
* If it's a known regression that the spec accepts as the cost of a
  bigger win elsewhere, document it in `sota_reframe_rollout.md` and
  flag-gate the trade-off.

## Adding a new phase QA suite

When the next round of work adds a Phase F (whatever that turns out
to be), follow the convention:

1. New module in `backend/services/<module>.py`.
2. New test file in `tests/qa/test_phase_f_<name>.py`.
3. Mock all GPU / external dependencies — the QA gate must run on a
   hosted CI runner.
4. Add a row to the table in this doc.
5. Update `.github/workflows/reframe_bench.yml`'s `paths:` filter.
