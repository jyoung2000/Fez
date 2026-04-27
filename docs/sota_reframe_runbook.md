# 2026 SOTA Reframing — Operator Runbook

For Unraid / homelab / GTX 1650 deployments. Companion doc to the
ADR in `docs/sota_reframe_rollout.md`.

---

## Verifying a deploy

After pushing to `claude/clipai-sota-reframing-tOo41` and rebuilding:

```bash
# 1. Local check BEFORE pushing (run from the repo root):
bash scripts/verify_sota_bench.sh local

# 2. Container check AFTER `docker compose up -d`:
docker compose exec backend bash scripts/verify_sota_bench.sh container

# 3. Quickest manual probe of the diagnostics endpoint:
curl -s http://localhost:1353/api/diagnostics/sota-bench-status | python -m json.tool
curl -s -X POST -H 'Content-Type: application/json' -d '{}' \
  http://localhost:1353/api/diagnostics/sota-bench-qa | python -m json.tool

docker compose down && docker compose build --no-cache && docker compose up -d

# 1. Confirm GPU passthrough
docker compose exec backend python -c "
from backend.services.transcription import _probe_gpu_availability
print(_probe_gpu_availability())
"

# 2. Confirm tracker dispatcher picks SAMURAI
docker compose exec backend python -c "
from backend.services.tracker_wrapper import get_active_backend
print('tracker:', get_active_backend())
"

# 3. Run the QA suites for every phase
docker compose exec backend pytest tests/qa/ -v

# 4. Run the bench against the fixture set
docker compose exec backend python -m \
  backend.scripts.compare_autoflip_vs_clipai \
  --manifest tests/real_content/manifest.json \
  --autoflip-outputs tests/autoflip_reference_outputs/ \
  --output /tmp/post_deploy_results.md \
  --json-out /tmp/post_deploy_results.json

# 5. Eyeball the production asset
docker compose exec backend python -m \
  backend.scripts.smoke_reframe \
  --clip panel_verzuz_tank_tyrese_10s \
  --output /tmp/post_deploy_smoke.mp4
```

Pass criteria: every clip ≥ baseline on every pre-existing metric +
the four new SOTA metrics meet their targets per `sota_reframe_rollout.md`.

## Disabling individual phases

Each phase has an env-var toggle. Set in your Unraid template:

| Phase | Disable                                       |
|-------|-----------------------------------------------|
| A     | `CLIPAI_TRACKER_BACKEND=opencv`               |
| B     | `CLIPAI_DENSE_POINT_TRACKING=0`               |
| C     | `CLIPAI_SALIENCY_ENABLED=0`                   |
| D     | `CLIPAI_COMPOSITION_HEAD=legacy`              |
| E     | `CLIPAI_EDITORIAL_PLANNER=0`                  |
| All   | `CLIPAI_LEGACY_REFRAME=1` (forces v1 path)    |

Restart the container after flipping any flag.

## Rolling back

Two options depending on severity:

### Soft rollback (per-phase)

If only one phase regressed (e.g. the trained CLIP head produces a
specific pathology):

```bash
# In .env:
CLIPAI_COMPOSITION_HEAD=legacy
# Restart only the backend container.
docker compose restart backend
```

### Hard rollback (full v1 path)

```bash
# In .env:
CLIPAI_LEGACY_REFRAME=1
# Restart everything.
docker compose down && docker compose up -d
```

`CLIPAI_LEGACY_REFRAME=1` short-circuits at the `human_reframe_pipeline`
gate — no flags below it matter, the legacy `reframe_segmenter` path
runs end-to-end.

## Common failures

### "ImportError: sam2 not installed"

Cause: the upstream `sam2` git pin in `requirements.txt` failed to
resolve during `docker compose build`. The dispatcher will fall back
to OpenCV at runtime (logs `[Samurai] auto-fallback to opencv`), so
service stays up but you lose the SAMURAI quality gain.

Fix: rebuild the container with the pin updated to a fresh upstream
commit. Verify with:

```bash
docker compose exec backend python -c "import sam2; print(sam2.__file__)"
```

### "RuntimeError: CUDA library libcuda.so.1"

Cause: nvidia-container-toolkit isn't passing the GPU into the
container. SAMURAI / CoTracker3 / TASED-Net / CLIP all degrade
gracefully (CoTracker3 raises, the rest fall back to OpenCV / CPU).

Fix: confirm the Unraid template has `--runtime=nvidia` and
`/dev/nvidia*` mounted. Verify with:

```bash
docker compose exec backend nvidia-smi
```

### "TASED-Net checkpoint download failed"

Cause: outbound network blocked, or the upstream URL changed.

Fix: download the checkpoint manually to `data/models/tasednet/` and
restart. The module will skip the download on subsequent boots.

### Editorial planner returns empty plans

Cause: the LLM provider chain is exhausted (OpenRouter credit, Ollama
OOM, etc.). The planner returns an empty plan and the LP falls back
to its existing path — no user-visible failure, but the per-genre
playbook is missing.

Fix: check `[EditorialPlanner]` log lines. If you see `text_completion
failed`, the orchestrator's circuit breaker has tripped — see
`backend/services/ai_orchestrator.py` for recovery (`reset_circuit_breaker()`).

## Editorial-planner cache management

Plans are cached in `/tmp/clipai_editorial/<sha16>.json`. To reset:

```bash
docker compose exec backend rm -rf /tmp/clipai_editorial
```

Re-runs of the same clip will pay one LLM call to regenerate.

## Saliency / OCR cache management

```bash
# Saliency heatmaps
docker compose exec backend rm -rf /tmp/clipai_saliency
# CoTracker3 dense tracks
docker compose exec backend rm -rf /tmp/clipai_cotracker
```

## Telemetry / WebSocket events

Each phase emits structured log lines under a tag the frontend
displays in `Analysis.jsx`:

| Tag                      | What                                       |
|--------------------------|--------------------------------------------|
| `[Samurai]`              | Mask coverage, motion-memory state         |
| `[CoTracker3]`           | Track count, occlusion %, camera motion    |
| `[Saliency]`             | Peak count, audio contribution             |
| `[CompositionHead]`      | Predicted (cx, cy, zoom), CLIP feat norm   |
| `[EditorialPlanner]`     | Shot count, LLM provider, cache hit/miss   |

Filter the WebSocket log via the prefix dropdown to drill into a
single phase's behaviour.
