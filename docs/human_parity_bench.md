# Human-Parity Bench

Replaces the AutoFlip-parity gate as the CI regression gate for
reframing. The AutoFlip synthetic fixtures remain under
`backend/scripts/validate_v2_phases.py --quick` and run as smoke-only;
they no longer block merges.

## Metrics

Computed by [`backend/services/human_parity_metrics.py`](../backend/services/human_parity_metrics.py)
per clip:

| Metric | Meaning | Regression threshold |
|--------|---------|----------------------|
| `mae_cx`, `mae_cy` | Mean absolute crop-center error vs human edit, output-frame fraction | may not increase by > 0.02 |
| `framing_macro_f1` | Macro-F1 over {CU, MS, WS, 2SHOT, SPLIT} | may not drop by > 0.05 |
| `cut_timing_deviation_ms_mean` | Mean \|our cut − human cut\| for cuts matched within 500 ms | may not increase by > 50 ms |
| `cut_timing_deviation_ms_p90` | p90 of above | may not increase by > 100 ms |
| `lead_room_correlation_x/_y` | Pearson r of (crop − subject) offset, frame-wise | may not drop by > 0.10 |
| `aesthetic_score_mean` | 0-10 VLM / learned critic score (optional) | may not drop by > 0.5 |

## Reference data

Populated from [`tests/real_content/manifest.json`](../tests/real_content/manifest.json).
Each clip gets a `ground_truth_vertical_url` pointing at the
human-edited 9:16 release (Triller Verzuz vertical, NBA vertical, MJ /
Chris Brown vertical cuts, etc). The extractor script at
[`backend/scripts/extract_human_trajectories.py`](../backend/scripts/extract_human_trajectories.py)
recovers the human's per-frame crop window by aligning the human
vertical to the source via ORB feature matching + homography, then
emits `data/human_trajectories/<slug>.jsonl`.

Clips without a vertical release fall back to the `TARGET_ZONES`
heuristic from `backend/scripts/compare_autoflip_vs_clipai.py` — those
clips have weaker signal but still contribute to the aesthetic score
metric.

## Running

```bash
# 1. populate tests/real_content/manifest.json source_url + ground_truth_vertical_url
bash tests/real_content/fetch.sh

# 2. extract human trajectories (one-time per clip; cached by sha256)
python -m backend.scripts.extract_human_trajectories

# 3. run the bench
python -m backend.scripts.run_human_parity_bench \
    --out /tmp/human_parity_run.json

# 4. first time only: pin the baseline
python -m backend.scripts.validate_human_parity \
    --results /tmp/human_parity_run.json \
    --pin docs/human_parity_baseline.json

# 5. every subsequent run (CI gate)
python -m backend.scripts.validate_human_parity \
    --results /tmp/human_parity_run.json \
    --baseline docs/human_parity_baseline.json
```

## What replaced

- `backend/scripts/validate_v2_phases.py` — AutoFlip LP / Condat / editorial-prior
  synthetic fixtures. Kept as `--smoke-only` invocation; no longer gating.
- `docs/autoflip_parity_v2_*.md` — historical record, not updated.
- `docs/week3_gap_analysis.md` — template. Superseded by this file.

## New aesthetic pipeline

With `CLIPAI_CRITIC_MODE` set to `vlm`, `learned`, or `both`, the bench
run additionally invokes [`backend/services/critic_loop.py`](../backend/services/critic_loop.py)
which samples output frames and scores them. The mean score lands in
`aesthetic_score_mean` on the report.
