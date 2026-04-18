# MediaPipe AutoFlip reference runner — Week 3 comparison harness input.
#
# This Dockerfile builds Google's MediaPipe AutoFlip from source and
# produces a deterministic black-box reference implementation we can
# compare ClipAI against on real content. AutoFlip is the publicly
# available reference implementation of Google's scene-cropping
# algorithm described in the AutoFlip paper (2019).
#
# Build is 30-60 minutes on a first pass (Bazel + OpenCV + protobuf
# build graph is large). Subsequent incremental builds with ccache are
# 2-5 minutes. For day-to-day harness runs, use the cached output JSONs
# committed at tests/autoflip_reference_outputs/ — those are the
# actual comparison inputs the harness reads. This Docker image only
# exists to REGENERATE those cached outputs when the clip set changes.
#
# See reference/autoflip/REFERENCE_OUTPUTS.md for the cache-generation
# workflow and the pinned MediaPipe ref used when the cache was last
# refreshed.

# ─────────────── Stage 1: build AutoFlip from source ───────────────
FROM ubuntu:20.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl wget unzip \
    python3 python3-dev python3-pip \
    libopencv-core-dev libopencv-imgproc-dev \
    libopencv-calib3d-dev libopencv-features2d-dev \
    libopencv-highgui-dev libopencv-video-dev \
    libopencv-videoio-dev \
    ffmpeg ca-certificates ccache \
    protobuf-compiler \
  && rm -rf /var/lib/apt/lists/*

# Bazel 6.1.1 is the newest known-good for MediaPipe AutoFlip as of
# v0.10.9. Newer Bazel versions have had breaking changes to the
# WORKSPACE file that MediaPipe hasn't patched.
ARG BAZEL_VERSION=6.1.1
RUN curl -fsSL "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh" -o /tmp/bazel.sh \
  && chmod +x /tmp/bazel.sh && /tmp/bazel.sh && rm /tmp/bazel.sh

# Pin MediaPipe to v0.10.9 — stable landing point for the AutoFlip
# binary. Bump the ARG and rebuild if a newer release is preferred;
# note the ref in REFERENCE_OUTPUTS.md when regenerating the cache.
ARG MEDIAPIPE_REF=v0.10.9
ENV MEDIAPIPE_REF=${MEDIAPIPE_REF}
RUN git clone --depth 1 --branch ${MEDIAPIPE_REF} \
        https://github.com/google/mediapipe /mediapipe
WORKDIR /mediapipe

# OpenCV 4 include-path fix. MediaPipe's upstream third_party/opencv_linux.BUILD
# expects OpenCV 3 headers; Debian 20.04 ships OpenCV 4 under
# /usr/include/opencv4. This sed is the minimal in-place patch.
RUN sed -i 's|/usr/include/opencv|/usr/include/opencv4|g' \
    third_party/opencv_linux.BUILD || true

ENV USE_CCACHE=1
RUN bazel build -c opt \
    --define MEDIAPIPE_DISABLE_GPU=1 \
    mediapipe/examples/desktop/autoflip:run_autoflip

# ─────────────── Stage 2: minimal runtime ───────────────
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV MEDIAPIPE_REF=v0.10.9

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core4.2 libopencv-imgproc4.2 libopencv-calib3d4.2 \
    libopencv-features2d4.2 libopencv-highgui4.2 libopencv-video4.2 \
    libopencv-videoio4.2 ffmpeg python3 python3-pip \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY --from=builder /mediapipe/bazel-bin/mediapipe/examples/desktop/autoflip/run_autoflip /usr/local/bin/run_autoflip
COPY --from=builder /mediapipe/mediapipe/examples/desktop/autoflip/autoflip_graph.pbtxt /etc/autoflip/autoflip_graph.pbtxt
COPY reference/autoflip/run_one.sh /usr/local/bin/run_one.sh
RUN chmod +x /usr/local/bin/run_one.sh

ENTRYPOINT ["/usr/local/bin/run_one.sh"]
