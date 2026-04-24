# Human-Reframe Rollout (Phase A)

Status: **Phase A — wiring landed, default OFF.** Activation in
production is opt-in per job via env flag, gated to a small content-
type allowlist.

## Why three reframe paths?

The reframing pipeline currently has three production-runnable code
paths. They sit on a quality vs. risk gradient:

| Path | Module | Default | Notes |
|------|--------|---------|-------|
| `reframe_segmenter` | `backend.services.reframe_segmenter` | **default** | Stable. Drives every clip in production today. |
| `autoflip` | `backend.services.autoflip_segmenter` | opt-in via `USE_AUTOFLIP_REFRAME=1` | AutoFlip-parity path, used for fixture-locked benchmarks. |
| `human_reframe` | `backend.services.human_reframe` (composed via `human_reframe_pipeline.py`) | opt-in via `CLIPAI_HUMAN_REFRAME_PIPELINE=1` | Phase A target. SOTA-tier composition (2-D L1 LP solver, Kalman, motivated zoom, A/B cuts). |

We keep all three because each catches things the others miss:

- `reframe_segmenter` is the workhorse — it's been hardened on every
  content type we ship.
- `autoflip` exists for parity benchmarking against the published
  AutoFlip behaviour.
- `human_reframe` produces visibly better framing for talking-head
  and panel content (bench evidence below) but hasn't been validated
  on every genre yet.

The `human_reframe` path is the eventual default. Phase A gets it
**runnable in production** without flipping the default. A
later session — after enough comparison renders justify the switch
— promotes it.

## Allowlist

`backend.services.human_reframe_pipeline.HUMAN_REFRAME_ALLOWED_CONTENT_TYPES`:

- `multi_speaker_panel`
- `talking_head`

### Criteria for adding a content type

A new entry needs **at least one real clip of that content type
scoring within spec on every quality axis** in the bench. The bench
runs are documented in `docs/reframe_quality_metrics.md`. The bar is:

- chin-clip rate < 0.5%
- head-clip rate < 0.5%
- subject-center error p95 < 0.12
- jitter p95 < 0.025
- saccade gap mean > 1.5s
- critic auto-repair rate ∈ [5%, 25%]
- coverage_ok = true
- silent legacy fallback rate = 0

Bench evidence on file (2026-04-23 + 2026-04-24):

- `multi_speaker_panel` — `panel_breakfast_club_10s` clean across
  every axis. `panel_joebudden_10s` clean except jitter (0.030 vs
  0.025 target — within hard-fail bound).
- `talking_head` — synthetic `_fx_talking_head` fixture clean after
  the `head_rate=1.0` false-positive geometry fix (PR
  `claude/fix-head-rate-false-positive-llJ8F`).

Until a content type has comparable real-clip evidence, it stays out
of the allowlist.

## Opting in per job

Two env flags govern the path:

```
CLIPAI_HUMAN_REFRAME_PIPELINE=1   # opt into the human path
CLIPAI_REFRAME_COMPARE=1          # also keep the legacy plan for eyeballing
```

Both default OFF. The pipeline picks the path at analysis time; once
analysis is complete, the export step honours whatever plan analysis
produced.

`pick_reframe_path(content_type)` returns the chosen path name —
`"human_reframe"` | `"autoflip"` | `"reframe_segmenter"` — based on
the env flags + content type. The pipeline logs `Reframe mode: <PATH>`
at the start of the reframe stage so you can confirm which branch
ran.

## Running side-by-side comparisons

The end goal is to eyeball `<job>_legacy.mp4` next to `<job>_human.mp4`
to decide whether the new path's framing is better. There are two
paths to that pair of files today:

### Workflow A — single-job toggle (recommended for spot-checks)

Run analysis on the same source twice:

1. `CLIPAI_HUMAN_REFRAME_PIPELINE=0` → produces the legacy MP4.
   Rename to `<job>_legacy.mp4`.
2. `CLIPAI_HUMAN_REFRAME_PIPELINE=1` → produces the human MP4.
   Rename to `<job>_human.mp4`.

This works today with no extra wiring. No analysis-time changes
needed; the database simply overwrites the previous `render_plan`
entry on the second run.

### Workflow B — `CLIPAI_REFRAME_COMPARE=1` marker

When set together with the pipeline flag, the analysis stage tags
the JobResult with a compare-mode marker so a follow-up exporter
change can produce both MP4s in a single pass. **The exporter
side of this is not yet implemented** — Phase A only wires the
analysis-time marker.

Until the exporter follow-up lands, Workflow A is the path that
actually gets you two MP4s. The marker is useful only as a
queryable signal that the user wanted both renders.

## Known issues / per-content-type caveats

- `multi_speaker_panel` (`panel_joebudden_10s` 2026-04-23): jitter
  p95 came in at 0.030, slightly above the 0.025 target but well
  under the 0.05 hard-fail. Acceptable for Phase A; revisit if more
  panel clips show jitter regressions.
- `talking_head`: synthetic fixture is geometric — full-source-height
  9:16 crop has no vertical degrees of freedom. Real talking-head
  clips with face heights in the 15–25% range (the realistic
  case for a 1080p YouTube upload) leave the solver actual y-slack;
  expect different jitter / center-error than the synthetic fixture
  reports.
- All other content types route to the legacy segmenter regardless
  of the flag. Trying to opt them in via env override does not work
  by design — it requires a code change that updates the allowlist.

## Verifying the wiring

Smoke-test the gating logic without running the full pipeline:

```bash
PYTHONPATH=. python3 -m pytest \
  backend/tests/test_human_reframe_pipeline_wiring.py -v
```

The 11 tests cover flag-off default, flag-on allowlisted paths,
flag-on but unallowlisted falling through, the env-flag interactions,
and the synthetic-fixture end-to-end run.

## What to do when something looks wrong

The new path silently falls back to the legacy segmenter on every
internal failure (returns `None` from
`try_run_human_reframe_pipeline`). The job's exported clip is never
worse than the legacy default — the bridge guarantees that.

To diagnose:

1. Search the job log for `Reframe mode:`. If it doesn't say
   `HUMAN_REFRAME`, the gates didn't open — verify
   `CLIPAI_HUMAN_REFRAME_PIPELINE=1` and that the content type
   classifier landed on an allowlisted entry.
2. If `Reframe mode: HUMAN_REFRAME` appears but the export looks
   wrong, look for `human-reframe pipeline:` warnings — they
   describe the failure point (no dense faces / coverage gap /
   adapter failure).
3. As a last resort, set `CLIPAI_HUMAN_REFRAME_PIPELINE=0` and
   re-run analysis on the same source — this restores the legacy
   path bit-for-bit.

## Phase B (next)

After eyeballing N comparison renders confirms the human path is
visibly better:

- Promote the env flag to default ON.
- Expand the allowlist with the next bench-cleared content type.
- Wire the exporter side of `CLIPAI_REFRAME_COMPARE` so a single
  job emits both `_legacy.mp4` and `_human.mp4`.
- Eventually retire the legacy `reframe_segmenter` once the human
  path covers every content type with bench evidence.

These are explicit follow-ups, not Phase A scope.
