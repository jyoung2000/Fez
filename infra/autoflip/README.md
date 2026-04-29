# MediaPipe AutoFlip sidecar (Task 5A)

Real MediaPipe AutoFlip references for the SOTA reframing bench
(`backend/scripts/compare_autoflip_vs_clipai.py`). Without this the
"AutoFlip" column is the naive center-crop face tracker from
`run_naive_baseline.py` — a strawman, not a SOTA comparator.

## Versioning

| Path | MediaPipe ref | Notes |
| --- | --- | --- |
| `Dockerfile` (default) | `v0.10.14` | Built from source via Bazel, CPU-only (`MEDIAPIPE_DISABLE_GPU=1`). First build takes ~30–60 minutes on the homelab; subsequent rebuilds reuse the build cache. |
| `Dockerfile.prebuilt` | n/a | Drops in a binary you supply via `AUTOFLIP_BINARY_URL` + `AUTOFLIP_BINARY_SHA256`. Use only when the Bazel build cannot complete. |

The MediaPipe AutoFlip example was last touched upstream in 2020. The
graph config + protos at `mediapipe/examples/desktop/autoflip/` are the
canonical comparator that academic and industry reframing papers
benchmark against. Do not replace this with a non-published reframer.

## Build

The image lives under the `bench` compose profile so a vanilla
`docker compose up` does not pull or build it.

```bash
make autoflip-references-build
# equivalent to:
docker compose --profile bench build autoflip
```

Override the MediaPipe ref by editing `MEDIAPIPE_REF` at the top of
`Dockerfile`. Whenever you bump the ref, update this README's table and
the `BUILD_FAILURE.md` status if applicable.

## Run

The bench wrapper drives the container:

```bash
make autoflip-references          # writes tests/autoflip_reference_outputs/<slug>.json
                                  # one JSON per fixture clip with tool="autoflip_real"
```

Behind the scenes:

```bash
docker compose run --rm autoflip \
    --input_video_path=/data/<slug>.mp4 \
    --output_video_path=/output/<slug>_autoflip.mp4 \
    --aspect_ratio=9:16 \
    --output_metadata_path=/output/<slug>_metadata.pbtxt
```

`/data` is the host's `${CLIPAI_REAL_CONTENT_CACHE}` (read-only).
`/output` is `tests/autoflip_reference_outputs` (read-write).

## Path A2 — prebuilt binary

If the Bazel build cannot complete in the homelab:

1. Build the binary once from a workstation:
   ```bash
   docker build -f infra/autoflip/Dockerfile -t clipai/autoflip:local infra/autoflip
   docker create --name af clipai/autoflip:local
   docker cp af:/usr/local/bin/run_autoflip /tmp/run_autoflip
   docker rm af
   sha256sum /tmp/run_autoflip
   ```
2. Publish that binary to a mirror you trust (S3 bucket, internal
   registry, etc.) and record the SHA256 below.
3. Build the Path A2 image with:
   ```bash
   docker build \
     -f infra/autoflip/Dockerfile.prebuilt \
     --build-arg AUTOFLIP_BINARY_URL=https://your-mirror/run_autoflip \
     --build-arg AUTOFLIP_BINARY_SHA256=<sha256-from-step-1> \
     -t clipai/autoflip:local infra/autoflip
   ```

References generated through this path emit `tool="autoflip_prebuilt"`
in the JSON so the bench rollup can label them distinct from
`autoflip_real`.

### Trusted-source policy

Never `curl` an AutoFlip binary from a random GitHub Actions artifact,
unsigned release page, or community fork. The audit trail must be:
*you built it from a pinned MediaPipe ref* OR *you trust the
publishing org's signing process explicitly*.

## When the build fails

Document failures in `infra/autoflip/BUILD_FAILURE.md` with:

- The MediaPipe ref attempted.
- The exact Bazel command executed.
- The error and where it occurred.
- Estimated wall-clock spent.

Then ship Path A2 OR fall back to the naive baseline (`make
naive-references`). The bench rollup distinguishes all three states
(`autoflip_real`, `autoflip_prebuilt`, `naive_baseline`); the only
unacceptable outcome is shipping a "real AutoFlip" claim while the
data is actually `naive_baseline`.
