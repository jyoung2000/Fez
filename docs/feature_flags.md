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

## AutoFlip docker-out-of-docker — host-path env vars

The Settings → "Build AutoFlip image" button and the
`/api/diagnostics/build-autoflip-image` SSE endpoint shell out to
`docker build` and `docker run` from inside the running app
container. To make that work end-to-end, three things must hold:

1. **docker CLI** is installed in the app image
   (`apt-get install docker.io` in both `Dockerfile` and
   `Dockerfile.gpu`).
2. **The host's docker socket** is bind-mounted into the app
   container at `/var/run/docker.sock` (`docker-compose.yml`
   `app.volumes`). SECURITY NOTE: this gives the container
   effectively-root on the host. For a homelab behind LAN this is
   the standard tradeoff; for anything internet-exposed, comment
   the mount out and run `docker build infra/autoflip/` and
   `python -m backend.scripts.run_autoflip_reference …` from the
   host shell instead.
3. **Host-side path env vars** are set so volume-mount paths can
   be translated from the app container's mount namespace to the
   host filesystem (which is what the host docker daemon
   interprets):

| Env var | Default | Purpose |
|---------|---------|---------|
| `CLIPAI_HOST_REAL_CONTENT_CACHE` | `${PWD}/data/real_content_cache` | Host path to the bench fixture cache. Used by `_container_path_to_host` to rewrite `/var/cache/clipai/real_content` (the in-container view) to its host-side location for `docker run -v` invocations. |
| `CLIPAI_HOST_AUTOFLIP_REF_DIR` | `${PWD}/tests/autoflip_reference_outputs` | Host path to the AutoFlip reference output directory, used the same way. |
| `CLIPAI_HOST_AUTOFLIP_DOCKERFILE_DIR` | `${PWD}/infra/autoflip` | Host path to the AutoFlip sidecar Dockerfile context. The `/build-autoflip-image` endpoint passes this to `docker build`. |

When any of these are unset (typical when running the script from
a developer's host with python directly), path translation is a
no-op and the script falls back to passing the supplied paths
unchanged — preserving backwards-compatibility with host-side
workflows.

## Other env vars touched by the SOTA bench

- `CLIPAI_REAL_CONTENT_CACHE` — root for the per-clip extraction
  cache. Defaults to `/var/cache/clipai/real_content`.
- `CLIPAI_TRACKER_BACKEND`, `CLIPAI_DENSE_POINT_TRACKING`,
  `CLIPAI_SALIENCY_ENABLED`, `CLIPAI_COMPOSITION_HEAD`,
  `CLIPAI_EDITORIAL_PLANNER`, `CLIPAI_HUMAN_REFRAME_PIPELINE` — the
  Phase A-E feature flags that the SOTA test panel forces ON for
  every bench run.

## Critic engine (Layer 1+)

The 5-layer critic engine grades reframes the way a professional
editor would, then re-solves windows that fail. Layer 1 ships with
the saliency-in-crop check + auto-repair widening; Layers 2-5 land
in subsequent PRs.

| Env var | Default | Purpose |
|---------|---------|---------|
| `CLIPAI_SALIENCY_LAYER` | `1` | Layer 1 toggle. ON → `_extract_and_cache` writes `saliency_peaks_per_second.json` (TASED-Net top-K peaks per second) and `reframe_critic.score_plan` flags windows whose crop excludes the salient region. OFF → skip extraction; legacy behavior preserved. |
| `CLIPAI_FACE_REGISTRY_BURN_IN_S` | `5.0` | XC.1 burn-in window. Detections in the first N seconds are dropped from face-registry slot discovery (so B-roll / title-card frames don't pin slots wrong). Falls back to unfiltered when the clip is shorter than the window or post-burn-in is too sparse. Set to `0` for legacy. |
| `CLIPAI_ASD_CONFIDENCE_MARGIN` | `0.15` | XC.3 confidence floor for v3 active-speaker timeline. When exactly one identity crosses the speaking threshold but the runner-up is within the margin, v3 emits `slot_id=-1` (unsure). Multi-speaker overlap (≥2 above threshold) is unaffected. Set to `0` to disable. |

**TASED-Net availability.** Layer 1's saliency extraction depends on
the `tasednet` Python wheel + a CUDA GPU. When the wheel is missing or
the GPU isn't visible, the helper degrades gracefully — logs a warning
and writes an empty peaks file. The cache stays valid (the file just
has no peaks); the critic treats no-data as "not measured" and skips
the saliency check. A homelab that hasn't installed `tasednet` still
produces a fully-valid v6 cache.
