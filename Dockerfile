## Stage 1: Build frontend with Node
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

## Stage 2: Runtime
FROM python:3.11-slim

# Make NVIDIA GPUs visible when passed through with --gpus
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,video,utility
# Force pure-Python protobuf so MediaPipe 0.10.8 graph configs parse correctly
# with protobuf>=4 (required by torch/pyannote). The C++ implementation
# rejects 3.x-format graph definitions under protobuf 4.x.
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Install system dependencies (ca-certificates ensures HTTPS model downloads work)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    git \
    ca-certificates \
    fontconfig \
    fonts-dejavu-core \
    fonts-freefont-ttf \
    fonts-liberation2 \
    unzip \
    libgl1-mesa-glx libglib2.0-0 \
    docker.io \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install DM Sans font (default subtitle font) so FFmpeg/libass can find it
# Downloaded directly from the canonical Google Fonts GitHub repo (stable raw URLs)
RUN mkdir -p /usr/share/fonts/truetype/dmsans && \
    curl -fsSL -o /usr/share/fonts/truetype/dmsans/DMSans.ttf \
      "https://github.com/google/fonts/raw/main/ofl/dmsans/DMSans%5Bopsz%2Cwght%5D.ttf" && \
    curl -fsSL -o /usr/share/fonts/truetype/dmsans/DMSans-Italic.ttf \
      "https://github.com/google/fonts/raw/main/ofl/dmsans/DMSans-Italic%5Bopsz%2Cwght%5D.ttf" && \
    fc-cache -f -v

# Install popular Google Fonts for subtitle use (variable + static weight files)
RUN mkdir -p /usr/share/fonts/truetype/google-fonts && \
    cd /usr/share/fonts/truetype/google-fonts && \
    curl -fsSL -o Montserrat.ttf "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf" && \
    curl -fsSL -o OpenSans.ttf "https://github.com/google/fonts/raw/main/ofl/opensans/OpenSans%5Bwdth%2Cwght%5D.ttf" && \
    curl -fsSL -o Roboto.ttf "https://github.com/google/fonts/raw/main/ofl/roboto/Roboto%5Bwdth%2Cwght%5D.ttf" && \
    curl -fsSL -o Poppins-Regular.ttf "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Regular.ttf" && \
    curl -fsSL -o Poppins-Bold.ttf "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Bold.ttf" && \
    curl -fsSL -o Inter.ttf "https://github.com/google/fonts/raw/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf" && \
    curl -fsSL -o Nunito.ttf "https://github.com/google/fonts/raw/main/ofl/nunito/Nunito%5Bwght%5D.ttf" && \
    curl -fsSL -o Lato-Regular.ttf "https://github.com/google/fonts/raw/main/ofl/lato/Lato-Regular.ttf" && \
    curl -fsSL -o Lato-Bold.ttf "https://github.com/google/fonts/raw/main/ofl/lato/Lato-Bold.ttf" && \
    curl -fsSL -o Oswald.ttf "https://github.com/google/fonts/raw/main/ofl/oswald/Oswald%5Bwght%5D.ttf" && \
    curl -fsSL -o PlayfairDisplay.ttf "https://github.com/google/fonts/raw/main/ofl/playfairdisplay/PlayfairDisplay%5Bwght%5D.ttf" && \
    curl -fsSL -o BebasNeue-Regular.ttf "https://github.com/google/fonts/raw/main/ofl/bebasneue/BebasNeue-Regular.ttf" && \
    fc-cache -f -v

# Register /data/fonts with fontconfig so libass picks up custom fonts
RUN mkdir -p /data/fonts && \
    echo '<?xml version="1.0"?>\n<!DOCTYPE fontconfig SYSTEM "fonts.dtd">\n<fontconfig><dir>/data/fonts</dir></fontconfig>' \
    > /etc/fonts/conf.d/99-custom-fonts.conf

WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install pyannote.audio for neural speaker diarization (CPU torch for non-GPU builds)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir pyannote.audio>=3.1.0 && \
    # MediaPipe runtime deps (installed separately to avoid protobuf conflict)
    pip install --no-cache-dir \
        flatbuffers>=23.1.4 \
        attrs>=23.1.0 \
        sounddevice>=0.4.6 \
        absl-py>=1.0.0 && \
    # MediaPipe itself — skip deps to avoid protobuf<4 constraint
    pip install --no-cache-dir --no-deps mediapipe==0.10.8 && \
    # Verify MediaPipe can actually load (fail build early if broken)
    python3 -c "import mediapipe; print(f'MediaPipe {mediapipe.__version__} installed')" && \
    python3 -c "import mediapipe.python.solutions.face_mesh; print('FaceMesh available')"

# Download YuNet model for face detection fallback (~350KB, one-time)
# Download lbpcascade_animeface for v2 Phase 6 anime face detection (~110KB)
RUN mkdir -p /app/backend/models && \
    curl -sL -o /app/backend/models/face_detection_yunet_2023mar.onnx \
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" && \
    curl --retry 4 --retry-delay 5 --retry-all-errors -fsSL \
    -o /app/backend/models/lbpcascade_animeface.xml \
    "https://github.com/nagadomi/lbpcascade_animeface/raw/master/lbpcascade_animeface.xml"

# v4.1: pre-download YOLOv8n weights for PersonDetector / ObjectDetector
# (~5.5MB). object_detector.py looks at /data/models/yolov8n.pt first,
# so stash the weights there at build time. Without this step the first
# analysis run stalls while ultralytics downloads from GitHub, or fails
# silently on offline containers. Also verify load so broken builds
# fail early instead of silently landing backend=none.
RUN mkdir -p /data/models && \
    curl --retry 4 --retry-delay 5 --retry-all-errors -fsSL \
      -o /data/models/yolov8n.pt \
      "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt" && \
    python3 -c "from ultralytics import YOLO; m = YOLO('/data/models/yolov8n.pt'); print(f'YOLOv8n loaded: {len(m.names)} classes')"

# Pre-download InsightFace buffalo_s ArcFace pack so the first run with
# CLIPAI_FACE_EMBEDDING=arcface doesn't stall while the model downloads.
# CPU-only via onnxruntime — never touches the GPU. The "|| echo" keeps
# the image build green if the deepghs CDN is unreachable; the lazy
# loader will retry at runtime.
RUN mkdir -p /app/backend/models/insightface && \
    python3 -c "from insightface.app import FaceAnalysis; \
                FaceAnalysis(name='buffalo_s', root='/app/backend/models/insightface', \
                             providers=['CPUExecutionProvider']).prepare(ctx_id=-1, det_size=(320,320))" \
    || echo "WARN: buffalo_s pre-download failed — will download at runtime"

# Install CUDA runtime libraries via pip for GPU passthrough support.
# These PyPI packages provide the CUDA shared libraries that ctranslate2
# and faster-whisper need — no NVIDIA apt repo or system CUDA required.
# The "|| true" ensures the build succeeds on non-x86 architectures
# where these wheels may not be available.
RUN pip install --no-cache-dir \
    nvidia-cuda-runtime-cu12 \
    nvidia-cublas-cu12 \
    nvidia-cufft-cu12 \
    nvidia-cudnn-cu12 \
    nvidia-cuda-nvrtc-cu12 \
    2>/dev/null || true

# Point LD_LIBRARY_PATH at the pip-installed NVIDIA libs so ctranslate2 finds them
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cuda_runtime/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/cufft/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/cuda_nvrtc/lib:\
${LD_LIBRARY_PATH}

# Copy backend source
COPY backend/ ./backend/

# Copy the markdown docs so the cloud storage setup guide (and friends)
# can be rendered at /docs/cloud-storage/SETUP.md by backend/main.py.
# This is a few KB of markdown; excluding it makes the Settings page's
# "Setup guide →" link land on a blank page.
COPY docs/ ./docs/

# Copy the QA harness (Phase A-E mock-heavy unit tests + the master
# runner) so the Settings -> Advanced -> "2026 SOTA Reframing" 1-click
# button can spawn `python -m tests.qa.run_all_phases` against the
# real container. ~50 KB of Python; never imported at startup, only
# spawned on demand by the diagnostics router.
COPY tests/qa/ ./tests/qa/
# AutoFlip sidecar Dockerfile — shipped INSIDE the app image so the
# /api/diagnostics/build-autoflip-image endpoint can run
# ``docker build`` against it without depending on a host-side
# checkout. Requires the docker CLI (installed above) and the host's
# Docker socket to be bind-mounted at /var/run/docker.sock.
COPY infra/ ./infra/
# Make ``tests`` importable as a top-level package (the QA runner uses
# ``python -m tests.qa.run_all_phases``).
RUN touch ./tests/__init__.py
# Real-content bench manifest. Tiny JSON file (~5 KB) listing the
# fixture clips the bench scores against; the actual MP4s stay
# external and are mounted from the host (see docker-compose.yml's
# CLIPAI_REAL_CONTENT_CACHE volume + env var).
COPY tests/real_content/manifest.json ./tests/real_content/manifest.json
# Make sure the AutoFlip-reference-outputs directory exists so the
# bench's --autoflip-outputs default path doesn't hit
# FileNotFoundError.
RUN mkdir -p ./tests/autoflip_reference_outputs ./tests/real_content
# Operator helper scripts (verify_sota_bench.sh etc) so they can be
# invoked via `docker compose exec app bash scripts/<name>.sh`.
COPY scripts/ ./scripts/
RUN chmod +x ./scripts/*.sh 2>/dev/null || true

# Copy built frontend from stage 1
COPY --from=frontend-build /app/frontend/dist ./static

# Build timestamp for cache-staleness detection. The /sota-clip-bench
# SSE endpoint compares this against the per-clip extraction cache mtime
# to flag "stale cache, fresh code" mismatches that the in-process
# mtime watch list might miss (e.g. when a newly-added module's path
# hasn't been added to _EXTRACTOR_MODULES_FOR_CACHE yet).
RUN date -u +"%Y-%m-%dT%H:%M:%SZ" > /etc/build_info \
    && cat /etc/build_info

# Pre-download Light-ASD ONNX model so first-run inference doesn't
# require network egress. The SHA256 pin protects against substitution
# on GitHub Releases. To compute / refresh the pin locally:
#     make download-light-asd-model
# then paste the printed sha256 into LIGHT_ASD_SHA256 below. The
# ``|| echo`` keeps the build green when the operator hasn't pinned
# the hash yet — the lazy loader in ``backend/services/light_asd.py``
# falls back to runtime download in that case (same behavior as
# pre-Task-C builds), and the v3 path falls back to v2 if the model
# is unavailable.
ENV LIGHT_ASD_SHA256=""
ENV LIGHT_ASD_MODEL_PATH=/app/backend/models/light_asd.onnx
RUN mkdir -p /app/backend/models && \
    if curl --retry 4 --retry-delay 5 --retry-all-errors -fsSL \
         -o "$LIGHT_ASD_MODEL_PATH" \
         "https://github.com/Junhua-Liao/Light-ASD/releases/download/v1.0/light_asd.onnx"; then \
       if [ -n "$LIGHT_ASD_SHA256" ]; then \
         echo "$LIGHT_ASD_SHA256  $LIGHT_ASD_MODEL_PATH" | sha256sum -c \
           || (echo "WARN: Light-ASD SHA256 mismatch — keeping file but flagging" \
               && rm -f "$LIGHT_ASD_MODEL_PATH"); \
       else \
         echo "WARN: LIGHT_ASD_SHA256 not pinned; record sha256:" \
           && sha256sum "$LIGHT_ASD_MODEL_PATH"; \
       fi; \
    else \
       echo "WARN: Light-ASD pre-download failed — will retry at runtime"; \
    fi

EXPOSE 1353

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "1353", "--workers", "1"]
