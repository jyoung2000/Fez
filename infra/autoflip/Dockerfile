# MediaPipe AutoFlip — CPU-only build for the SOTA reframing bench (Task 5A).
#
# Builds the canonical run_autoflip example binary from the upstream
# MediaPipe repo, pinned to a known-good tag. CPU-only because the
# homelab GTX 1650 has 4 GB VRAM and the upstream AutoFlip example does
# not have a CUDA path in any released MediaPipe tag — this is the same
# baseline academic / industry papers report against.
#
# Image is opt-in: docker compose build only runs it under the "bench"
# profile so default-up workflows are unaffected.
#
# If the Bazel build can't complete in the homelab, switch to
# Dockerfile.prebuilt per infra/autoflip/README.md and document the
# failure in BUILD_FAILURE.md.

# ── Build stage ─────────────────────────────────────────────────────────
FROM ubuntu:22.04 AS builder

# Pin to a known-good MediaPipe tag. Update both this and the README
# whenever the pinned SHA changes.
ARG MEDIAPIPE_REF=v0.10.14

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl unzip ca-certificates \
        python3 python3-pip python3-dev \
        ffmpeg libopencv-dev libopencv-contrib-dev \
        openjdk-11-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

# MediaPipe's Bazel WORKSPACE introspects the system Python's numpy
# include path during analysis (org_tensorflow → python_configure.bzl
# calls ``python -c "import numpy; print(numpy.get_include())"``). If
# numpy isn't importable on system Python the build aborts during
# repository fetch with "ModuleNotFoundError: No module named 'numpy'"
# before any compile step runs. Pin to numpy<2 because MediaPipe v0.10.x
# was developed against numpy 1.x and v2.0's API rename breaks header
# layout for some of the first-party C extensions.
RUN pip3 install --no-cache-dir "numpy<2"

# Bazelisk picks up the .bazelversion the MediaPipe repo ships with.
RUN curl -fsSL -o /usr/local/bin/bazelisk \
        https://github.com/bazelbuild/bazelisk/releases/download/v1.21.0/bazelisk-linux-amd64 \
    && chmod +x /usr/local/bin/bazelisk \
    && ln -s /usr/local/bin/bazelisk /usr/local/bin/bazel

WORKDIR /src
RUN git clone --depth 1 --branch ${MEDIAPIPE_REF} \
        https://github.com/google/mediapipe.git

WORKDIR /src/mediapipe

# Build the autoflip binary, GPU disabled. The exact target name may
# move across MediaPipe tags — if this fails, check the upstream
# mediapipe/examples/desktop/autoflip/BUILD file in the cloned tree and
# update the target accordingly.
RUN bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
        //mediapipe/examples/desktop/autoflip:run_autoflip

# ── Runtime stage ───────────────────────────────────────────────────────
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg ca-certificates \
        libopencv-core4.5d libopencv-imgproc4.5d \
        libopencv-imgcodecs4.5d libopencv-videoio4.5d \
        libopencv-calib3d4.5d libopencv-features2d4.5d \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder \
        /src/mediapipe/bazel-bin/mediapipe/examples/desktop/autoflip/run_autoflip \
        /usr/local/bin/run_autoflip
COPY --from=builder \
        /src/mediapipe/mediapipe/examples/desktop/autoflip \
        /usr/local/share/autoflip

# AutoFlip looks up its tflite/landmark model files relative to the
# graph config. Mirror the upstream layout so the canonical
# autoflip_graph.pbtxt resolves out-of-the-box.
ENV AUTOFLIP_GRAPH=/usr/local/share/autoflip/autoflip_graph.pbtxt

ENTRYPOINT ["/usr/local/bin/run_autoflip"]
