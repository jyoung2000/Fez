# Layer 2 — DOVER-Mobile (operator setup)

DOVER-Mobile is the no-reference VQA model that powers Layer 2 of
the critic engine. It returns aesthetic + technical scores, each in
[0, 1]. We compute them on both the source clip and the rendered
reframe and surface the deltas in the Validate Test panel.

## Vendor + bake at build time

The main image expects an ONNX export of DOVER-Mobile at
`/opt/clipai/models/dover_mobile.onnx`. The build-stage Dockerfile
in this directory produces that artifact from the upstream
PyTorch checkpoint.

**One-time operator step:** compute the SHA256 of the upstream
`DOVER-Mobile.pth` checkpoint and record it both here and in
`docs/feature_flags.md`:

```bash
curl -L https://github.com/VQAssessment/DOVER/releases/download/v0.5.0/DOVER-Mobile.pth \
  | sha256sum
```

Then build the export image:

```bash
docker build \
    --build-arg DOVER_MOBILE_SHA256=<paste hash here> \
    -t clipai/dover-mobile-export:latest \
    infra/dover_mobile/
```

Watch the build log for the final `sha256sum /opt/dover_mobile.onnx`
line; that's the value you'd pin into the main image's
`DOVER_MOBILE_SHA256` env var if you want runtime SHA verification of
the exported ONNX (optional — the wrapper skips when the env var is
empty).

## Wiring into the main image

The main `Dockerfile` references this stage:

```dockerfile
COPY --from=dover_export /opt/dover_mobile.onnx /opt/clipai/models/dover_mobile.onnx
ENV DOVER_MOBILE_MODEL_PATH=/opt/clipai/models/dover_mobile.onnx
ENV DOVER_MOBILE_SHA256=""
```

When you've pinned the ONNX SHA, set `DOVER_MOBILE_SHA256` to the
hex string. The runtime `_verify_model_sha` helper in
`backend/services/dover_quality.py` will refuse to load a model
that doesn't match — an extra layer of defense against substitution
attacks on the cached image layer.

## What the wrapper expects

`backend/services/dover_quality.py` samples 32 frames evenly across
the input video, resizes to 224×224, applies ImageNet normalization
in BGR→RGB order, and feeds the resulting `(1, 3, 32, 224, 224)`
tensor to the ONNX model. The export must produce that input layout
for two outputs (aesthetic, technical) — typical for `dover.export_onnx`.

If your export uses a different layout, fix the wrapper's
`_preprocess_for_dover` accordingly. The unit tests in
`tests/qa/test_critic_l2_dover.py` pin the math so a wrapper-side
mistake doesn't go unnoticed.

## Failure modes (graceful)

The `dover_quality` module degrades to "L2 unavailable" — never
raises — when:

  * `onnxruntime` isn't installed
  * the ONNX file is missing on disk
  * the SHA256 doesn't match `DOVER_MOBILE_SHA256` (when set)
  * the source / reframed video is unreadable
  * `cv2.VideoCapture` can't decode any frame

In all cases the bench's `result.critic.quality_delta` is absent,
the SSE stream emits `phase_result: l2_quality_delta status:
skipped`, and the UI's Critic engine row simply doesn't render the
quality row. Production exports are unaffected because L2 is
bench-only this PR (see prompt's "Out of scope").
