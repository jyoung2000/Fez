# SAM 3 Deployment — ClipAI Blueprint v2 Phase 4

Meta SAM 3 does promptable concept segmentation: detect + track +
re-identify every instance of a concept (e.g. `"the ball"`,
`"anime face"`, `"HUD elements"`) across a video given a text prompt.
It consolidates four separate ClipAI detectors — `face_detector`,
`anime_face_detector`, `object_detector` / `object_tracker`, and
`gaming_hud_detector` — into one model.

## Deployment decision

ClipAI's reference deployment target (a GTX 1650 with 4 GB VRAM) cannot
self-host SAM 3. The model weights are multi-GB and the inference
path wants ~10 GB VRAM for video-length tracking. Planning options:

| Option | Pros | Cons | Ship risk |
|---|---|---|---|
| A. Replicate / fal.ai API | No infra; pay-per-call | $0.01-0.05 per video-second; online dep; latency | low |
| B. Self-host on 3090 / 4090 | Free per call; low latency; offline-capable | Needs the hardware | medium (hardware) |
| C. Hybrid (B or A + legacy fallback) | Graceful degradation; no hard-fail | Two code paths | low |
| D. Smaller SAM 3 variant on GTX 1650 | No new hardware | Likely quality hit; may not fit 4 GB | high |

**Chosen: C (hybrid).** Primary path is a self-hosted SAM 3 service
reachable over HTTP (either on a larger home-lab GPU or a managed
endpoint like Runpod / Modal). Fallback is the existing detector
stack, so ClipAI never hard-fails if SAM 3 is unavailable.

### Rough cost estimate

Assumptions: 6 fps sample rate, 10 concept prompts per clip, average
4-minute source, Replicate pricing as of 2026-04.

| Scenario | Frames sampled | Estimated cost |
|---|---|---|
| Talking-head (2 prompts) | 4 × 60 × 6 = 1 440 | ~$0.02 per clip |
| Gameplay (6 prompts) | 1 440 | ~$0.06 per clip |
| Sports (4 prompts) | 1 440 | ~$0.04 per clip |

Per hour of processed video (~15 4-min clips), worst-case: ~$1.00.
Self-hosted marginal cost is electricity only.

### Start / stop the service

Self-hosted path (recommended for production):

```bash
# On the GPU host:
docker run --gpus all -p 8188:8188 \
  -e MODEL=sam3-large \
  clipai/sam3-service:latest

# In ClipAI:
export CLIPAI_SAM3_ENABLED=1
export SAM3_BACKEND=local_service
export SAM3_SERVICE_URL=http://gpu-host.lan:8188
```

API endpoint: Replicate path (zero infra):

```bash
export CLIPAI_SAM3_ENABLED=1
export SAM3_BACKEND=api_replicate
export REPLICATE_API_TOKEN=r8_xxx
```

### Fallback behavior

When `CLIPAI_SAM3_ENABLED=0` (default), unset, or the backend call
fails for any reason, the pipeline silently reverts to the legacy
detector stack. A `pipeline_warnings` entry is recorded so the
Analysis-page banner surfaces the fallback:

```
warn: SAM 3 call failed (HTTPError: connection refused) — using legacy detectors
```

### Acceptance targets

After one week of production use:

- **Fallback rate** on clean runs: < 5 %
- **Face recall** on anime content: +10 % over legacy
- **HUD precision** on gameplay: +20 % over legacy
- **Latency** on talking-head content: within 20 % of legacy

### Retiring legacy code (Phase 4.5)

Once the fallback rate holds below 1 % for four weeks:

- `anime_face_detector.py` → remove
- `face_registry.py` ArcFace re-ID → swap in SAM 3 instance IDs (keep registry layout)
- `object_tracker.py` → remove
- `gaming_hud_detector.py` → remove HUD-region code, keep the content-classification hint

Not in Phase 4 — these live on as the fallback path until SAM 3 has
earned the trust to be the only detector.
