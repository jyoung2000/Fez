# Week 3 gap analysis

**Status: TEMPLATE.** This file is produced by hand at the end of a
Week 3 harness run. It's the decision document that drives Week 4's
targeted fixes. Fill in the sections below after running
`backend/scripts/compare_autoflip_vs_clipai.py` against the real
clip set AND the cached MediaPipe AutoFlip reference outputs.

The session that landed the Week 3 harness produced this template
with placeholders (`<fill in>`) so the next session has a scaffold
to populate without re-deriving the format.

---

## How to read this file

Three columns of comparison per clip:

- **Ground truth** — a human-edited 9:16 version when one exists
  (Triller Verzuz vertical releases, official MJ / Chris Brown
  vertical cuts, NBA vertical highlights). When available, this is
  the authoritative target — how close is each tool to what a human
  editor actually did?
- **ClipAI** — our pipeline.
- **AutoFlip** — MediaPipe reference.

When ground truth is absent, the per-content-type **target zone**
from `compare_autoflip_vs_clipai.TARGET_ZONES` is the fallback.
Both tools are measured against the same zone.

---

## Where ClipAI beats AutoFlip (keep these wins)

Content types where ClipAI lands inside the target zone (or closer
to ground truth than AutoFlip).

- `<fill in>` — which content type, which metric, how much better.
- `<fill in>` — ...

## Where ClipAI ties AutoFlip (no action needed)

Content types where both tools land inside the target zone. These
are lower-priority — both tools are already doing fine.

- `<fill in>`
- `<fill in>`

## Where ClipAI loses to AutoFlip (Week 4 targets)

Each bullet names the failing content type, the specific metric
that regressed, the magnitude, and a one-sentence root-cause
hypothesis naming the specific module + function to touch.

- **`<content_type>` — `<metric>` off by `<delta>`.**
  Hypothesis: `<fill in>`. Module: `<backend/services/XXX.py>` ::
  `<function_name>`.
- `<fill in second target>`
- `<fill in third target>`

---

## Hypotheses for Week 4 fixes (ranked)

Each entry is a candidate Week-4 session with a specific
before/after metric target pulled from
`/tmp/week3_results.json`. Week 4 only runs after Week 3 has a real
roll-up populated.

### Candidate 1: `<fill in>`

- **Metric:** `<which metric>`
- **Current:** `<value from Week 3 JSON>`
- **Target:** `<target from zone or ground truth>`
- **Proposed change:** `<specific module, function, single-line summary>`
- **Synthetic fixture risk:** `<which of the 7 fixtures might regress; mitigation plan>`

### Candidate 2: `<fill in>`

...

### Candidate 3: `<fill in>`

...

---

## What does NOT go into Week 4

Things the session explicitly defers even if Week 3 reveals them:

1. **No new dormant modules.** Week 2 just wired three; slow down.
2. **No solver weight retuning beyond `MIN_HOLD_SECONDS`.** The L1 +
   LP weights have strong synthetic-fixture coverage and aren't a
   Week-4 target.
3. **No new `ClipContentType` enum values.** Only justified if the
   real clip set reveals ≥2 content types with clearly different
   needs from existing categories. The manifest's 12 clips aren't
   enough signal to carve a new category out.
4. **No changes to the `_KNOWN_DIVERGENCES` whitelist in
   `validate_v2_phases.py`.** The Week 1 `phase8-editorial-prior-fp-drift`
   entries stand; the gate stays strict.

---

## Week 3 harness + metric state summary

- New Week 3 artifacts committed:
  - `backend/services/autoflip_parity_metrics.py::cut_to_hold_ratio`
    (9th metric, 8 unit tests)
  - `backend/scripts/export_autoflip_compatible.py`
    (17 unit tests)
  - `backend/scripts/compare_autoflip_vs_clipai.py` with target-zone
    verdict layer (16 unit tests)
  - `reference/autoflip/Dockerfile` + `run_one.sh` +
    `REFERENCE_OUTPUTS.md`
  - `tests/real_content/manifest.json` + `fetch.sh`
  - `docs/week3_real_content_results.md` (placeholder; becomes real
    output on first harness run)
- **`validate_v2_phases --quick` exit 0** (Week 3 added no solver
  behavior changes, so the synthetic fixtures should be bit-for-bit
  identical to the Week 2 baseline).
- **41 new unit tests, 0 failures.**

### Known blockers (these need to clear before a real gap analysis)

1. **AutoFlip reference cache empty.** `reference/autoflip/Dockerfile`
   exists but the sandbox didn't attempt the Bazel build (30-60 min,
   heavy). Cache generation is a one-time expense on a machine that
   can finish the build. See `reference/autoflip/REFERENCE_OUTPUTS.md`.
2. **Clips not fetched.** `manifest.json` has placeholder
   `source_url` fields — Jalon fills them with his own archive
   paths, then `fetch.sh` pulls and pins the hashes.
3. **`run_clipai_on_clip` is a stub.** The real implementation
   wires `measure_autoflip_parity.py`'s inline segmenter path to a
   real-video extraction cache. Touches `frame_extractor`,
   `face_detector.detect_faces_dense`, and `transcription` inline
   or via a persisted extraction cache keyed on clip sha256. This
   is the single biggest remaining Week 3 work item.

### Part E — `MIN_HOLD_SECONDS` tuning (not executed)

The Week 3 prompt's Part E is a tuning pass on
`content_type_config.py::MIN_HOLD_SECONDS` **conditional** on the
harness showing a systematic skew on a content type. With the
harness stub producing synthetic single-segment ClipAI timelines,
the skew signal is meaningless — any real tuning based on it would
just chase the stub's shape. **No changes to `MIN_HOLD_SECONDS`
landed in this session.**

When the harness runs for real, follow the Part-E protocol:

1. For each content type with verdict MARGINAL or MISS, snapshot
   the current `MIN_HOLD_SECONDS` in the config.
2. Bump by 0.5 (or 0.25 if the parity gate regresses).
3. Re-run the harness filtered to that content type's clips.
4. Re-run `validate_v2_phases --quick` to confirm no synthetic
   regression.
5. Accept the bump if median improved AND fixtures stayed green.
6. Hard rules: never above 3.0, never below 0.4.

Record the final values in a "Week 3 `MIN_HOLD_SECONDS` tuning"
table here and append to `docs/reframing_autoflip_parity.md`.
