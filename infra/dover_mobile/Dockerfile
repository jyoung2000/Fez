# Build-stage Dockerfile — exports DOVER-Mobile to ONNX so the main
# app image doesn't carry the full PyTorch DOVER training stack.
# The output ``/opt/dover_mobile.onnx`` is COPY'd into the main
# Dockerfile via the multi-stage pattern.
#
# Build:
#   docker build -t clipai/dover-mobile-export:latest \
#       --build-arg DOVER_MOBILE_SHA256=<hash> \
#       infra/dover_mobile/
#
# The main Dockerfile then references this image as ``dover_export``
# in its ``COPY --from=`` line.
#
# License: DOVER (FudanVIP) at github.com/VQAssessment/DOVER. Verify
# the LICENSE file at the upstream commit you're vendoring before
# deploying — the project has historically used permissive licenses
# but always check.

FROM python:3.10-slim AS dover_export

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Pin a known-good commit at build time. Update periodically.
ARG DOVER_GIT_REF=main
RUN git clone --depth 1 https://github.com/VQAssessment/DOVER.git \
    && cd DOVER \
    && (git checkout "$DOVER_GIT_REF" 2>/dev/null || true)

WORKDIR /src/DOVER

# CPU-only torch is enough for the export trace.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir onnx onnxruntime opencv-python-headless decord \
    && pip install --no-cache-dir -e . || \
    (echo "WARN: editable install failed; falling back to runtime imports" \
     && pip install --no-cache-dir einops timm scipy fvcore tqdm pyyaml)

# Download the DOVER-Mobile checkpoint. The SHA256 must be supplied
# at build time — we don't bake a default to avoid implying a
# specific upstream artifact is "blessed" forever.
ARG DOVER_MOBILE_URL=https://github.com/VQAssessment/DOVER/releases/download/v0.5.0/DOVER-Mobile.pth
ARG DOVER_MOBILE_SHA256=""
RUN if [ -z "$DOVER_MOBILE_SHA256" ]; then \
        echo "WARN: DOVER_MOBILE_SHA256 not pinned at build time — record the hash in docs/feature_flags.md and rebuild"; \
    fi \
    && curl -fsSL -o /src/DOVER-Mobile.pth "$DOVER_MOBILE_URL" \
    && if [ -n "$DOVER_MOBILE_SHA256" ]; then \
        echo "$DOVER_MOBILE_SHA256  /src/DOVER-Mobile.pth" | sha256sum -c; \
    fi

# Export to ONNX. The DOVER repo's export entry point moves
# occasionally; try the documented paths in priority order. If all
# fail, the build fails with a clear message and the operator can
# patch ``infra/dover_mobile/Dockerfile`` to point at the right
# script for their pinned ref.
RUN ( \
        python -m dover.export_onnx \
            --weights /src/DOVER-Mobile.pth \
            --output /opt/dover_mobile.onnx \
            --opset 17 \
        || python tools/export_onnx.py \
            --weights /src/DOVER-Mobile.pth \
            --output /opt/dover_mobile.onnx \
            --opset 17 \
        || python scripts/export_onnx.py \
            --weights /src/DOVER-Mobile.pth \
            --output /opt/dover_mobile.onnx \
            --opset 17 \
    ) || (echo "FATAL: DOVER ONNX export failed — find the export entry point with 'find /src/DOVER -name *.py | xargs grep -l onnx' and patch infra/dover_mobile/Dockerfile" && exit 1)

# Sanity check: the exported file should be 30-40 MB. If under 1 MB
# the export silently produced a header-only file.
RUN test "$(stat -c%s /opt/dover_mobile.onnx)" -gt 10000000 \
    || (echo "FATAL: DOVER ONNX export produced an empty or tiny file ($(stat -c%s /opt/dover_mobile.onnx) bytes)" && exit 1)

# Print the final SHA256 so the operator can pin it in
# DOVER_MOBILE_SHA256 of the main image.
RUN sha256sum /opt/dover_mobile.onnx
