# Feature flags & operational env vars

Compact reference for the env vars that gate behavior changes in
ClipAI. Production defaults are listed first; the alternative values
exist primarily so the operator can roll back a regression without
redeploying.

## Active speaker detection — `CLIPAI_ASD_BACKEND`

| Value | Behavior |
|-------|----------|
| `light_asd` (default) | Run the Light-ASD audio-visual model to score per-face speaking probability, then emit a speaker timeline that supports overlapping speech and off-camera speakers. Falls back to `heuristic` automatically if the model can't be loaded or returns zero scores. |
| `heuristic` | Use the legacy v2 lip-aperture + audio-energy heuristic. Useful when the operator wants to compare a regression head-to-head, or when the Light-ASD model file isn't present in the image. |

Set the variable in `docker-compose.yml` (`environment:` section) or
`.env`. The flag is read both by the production pipeline
(`backend/services/pipeline.py`) and by the bench
(`backend/scripts/compare_autoflip_vs_clipai.py`) so they stay in
agreement on which timeline shape the cache holds.

When `CLIPAI_ASD_BACKEND=light_asd`, the bench cache requires
`asd_scores.json` to be present alongside the other extraction
artifacts. Switching backends invalidates affected clips because the
cache lacks the required file — that re-extraction is the desired
behavior. `EXTRACTION_CACHE_VERSION` was bumped to 5 to make this
explicit.

## Force re-extraction — bench script `--force-reextract`

Pass `--force-reextract` to
`python -m backend.scripts.compare_autoflip_vs_clipai` to ignore
the on-disk extraction cache and rebuild every clip from scratch.
The Settings → SOTA test panel exposes this as a "Force re-extract"
button that appears when the cache-state badge flags a stale cache;
the button POSTs `{"force": true}` to `/api/diagnostics/sota-clip-bench`
which then forwards `--force-reextract` to the bench subprocess.

## Build timestamp — `/etc/build_info`

Final RUN layer in `Dockerfile` / `Dockerfile.gpu` writes the UTC
ISO-8601 build timestamp into `/etc/build_info`. The diagnostics
endpoint reads it and includes it in the SSE `cache_check` payload
so the UI can compare cache mtime to image age and flag the "stale
cache, fresh code" mismatch (which the in-process mtime watch list
can miss when a newly-added module isn't yet in
`_EXTRACTOR_MODULES_FOR_CACHE`).

## Light-ASD model pin — `LIGHT_ASD_SHA256`

The Dockerfile pre-downloads the Light-ASD ONNX model from
`https://github.com/Junhua-Liao/Light-ASD/releases/download/v1.0/light_asd.onnx`
and verifies it against `LIGHT_ASD_SHA256`. To compute or refresh
the pin locally:

```bash
make download-light-asd-model
# Pastes the printed sha256 hex into LIGHT_ASD_SHA256 in
# Dockerfile and Dockerfile.gpu.
```

When the env var is unset (default in this branch until the
operator records a hash), the build prints the hash but doesn't
enforce it — the file is still kept and used. Once you record the
hash, a substituted artifact will fail the build instead of landing
in production.

## Other env vars touched by the SOTA bench

- `CLIPAI_REAL_CONTENT_CACHE` — root for the per-clip extraction
  cache. Defaults to `/var/cache/clipai/real_content`.
- `CLIPAI_TRACKER_BACKEND`, `CLIPAI_DENSE_POINT_TRACKING`,
  `CLIPAI_SALIENCY_ENABLED`, `CLIPAI_COMPOSITION_HEAD`,
  `CLIPAI_EDITORIAL_PLANNER`, `CLIPAI_HUMAN_REFRAME_PIPELINE` — the
  Phase A-E feature flags that the SOTA test panel forces ON for
  every bench run.
