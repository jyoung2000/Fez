# Layer 1 saliency — operator setup

The L1 critic flags windows whose crop excludes the salient region of
the source frame. Two backends are available, in priority order:

## 1. Spectral residual (default — always works)

Pure OpenCV (`cv2.saliency.StaticSaliencySpectralResidual_create`).
No external weights, no PyTorch, no GPU. Quality is below TASED-Net
but produces real heatmaps the critic can act on. Image already
ships with `opencv-contrib-python-headless` in `backend/requirements.txt`,
so this backend is available out of the box.

## 2. TASED-Net (opt-in — better quality, operator-supplied)

[MichiganCOG/TASED-Net](https://github.com/MichiganCOG/TASED-Net)
publishes a stronger spatiotemporal saliency model but the upstream
repo ships without an explicit license. We don't bake the model
definition or weights into the image; operators supply them
themselves with whatever attribution they choose.

**To enable TASED-Net:**

1. Drop the upstream `model.py` (defining `TASED_v2`) at
   `backend/vendor/tasednet/model.py`. Preserve the upstream
   copyright header. Add a `NOTICE` file in the same directory
   crediting the source and your interpretation of upstream terms.
2. Download the canonical `TASED_v2.pth` checkpoint and mount it at
   the path indicated by the `TASED_NET_MODEL_PATH` env var
   (default: `/opt/clipai/models/tased_v2.pth`).
3. Rebuild / restart the container. `_TasedNetAdapter.load()` will
   pick up both artefacts on the next saliency phase. The bench's
   SSE stream emits `phase_result.backend = "tased_net"` once
   loaded; the spectral fallback emits `"spectral_residual"`.

Failure modes degrade gracefully: missing module → fallback;
missing weights → fallback; CUDA requested + no GPU → CPU-only;
`torch.load` exception → fallback. The pipeline never raises on a
saliency failure.

## Verifying which backend is active

- Check the SSE stream during a Validate Test run. The `phase_result`
  event for the saliency phase carries a `backend` field.
- Check the result panel — when the spectral fallback fires, the
  Saliency (L1) row shows a small `(spectral fallback)` annotation.
- From a shell:
  ```bash
  docker compose exec backend python -c \
    'from backend.services.av_saliency import AvSaliency; \
     a = AvSaliency(device="cpu"); print(a.backend_name)'
  ```

## Why we don't bake TASED-Net by default

`MichiganCOG/TASED-Net` has no LICENSE file. Under default GitHub
copyright that means "all rights reserved" — we shouldn't ship the
model definition or weights in a public image without operator-side
clearance. The vendored-placeholder + fallback pattern lets
operators who've cleared upstream terms enable the better backend
without committing the project to a licensing position it can't
defend.
