# AutoFlip reference outputs

## What this directory is

`reference/autoflip/` contains the tooling to produce
**reference JSON timelines** from Google's open-source MediaPipe
AutoFlip (the 2019 scene-cropping reference implementation) for every
clip in `tests/real_content/manifest.json`. The outputs live at
`tests/autoflip_reference_outputs/` as one JSON per clip and are the
"AutoFlip" half of the Week 3 comparison harness at
`backend/scripts/compare_autoflip_vs_clipai.py`.

## Why we cache the outputs instead of running live

MediaPipe AutoFlip is a ~30-60-minute Bazel + OpenCV + protobuf build
that doesn't integrate cleanly into the ClipAI Python test harness.
We pay the build cost **once**, produce a JSON per clip, commit those
JSONs to the repo, and the harness reads them offline forever after.
Regenerate only when:

- The real-content clip set changes (new clips added to
  `tests/real_content/manifest.json`).
- MediaPipe upstream fixes a bug that would meaningfully change
  AutoFlip's per-frame crop decisions on the existing clips.

## Pinned environment

| Component | Pinned ref |
|---|---|
| MediaPipe | `v0.10.9` (git tag) — set via `--build-arg MEDIAPIPE_REF=v0.10.9` |
| Bazel | `6.1.1` |
| OpenCV (Ubuntu 20.04 debs) | `4.2.0-dfsg` |
| Base image | `ubuntu:20.04` |
| Aspect ratio | `9:16` |

Bumping MediaPipe requires re-running the cache generation AND
updating the ClipAI harness if the stderr log format changed — the
`run_one.sh` parser's regex is the fragile seam.

## Build the Docker image

```bash
cd /path/to/fasd
docker build -t clipai/autoflip-ref -f reference/autoflip/Dockerfile .
```

First build is 30–60 minutes; Bazel prints a lot of noise. Look for
`run_autoflip` in the final output — that's the binary. If Bazel
fails:

1. First attempt: run `docker system prune` and rebuild. Bazel's
   incremental cache can get into a bad state.
2. Second attempt: look at `mediapipe/third_party/opencv_linux.BUILD`
   errors. MediaPipe patches that file to point at the system OpenCV.
   Newer Debian packages may have moved headers — update the `sed`
   in the Dockerfile to match.
3. Third attempt: drop to a community prebuilt image like
   `jerkeeler/mediapipe-autoflip` and PIN THE TAG by SHA. Community
   images are fine for cache generation as long as the SHA stays
   recorded here alongside the MediaPipe ref they wrap.

If all three attempts fail, the harness still runs — it reads cached
JSONs. Generate the cache on a different machine that can complete
the build (cloud VM, homelab box with ccache populated, etc.) and
copy the resulting JSONs into `tests/autoflip_reference_outputs/`.

## Generate the cache (one-time)

```bash
# From the repo root, with clipai/autoflip-ref built and the clip
# files cached at $CLIPAI_REAL_CONTENT_CACHE per the manifest.
mkdir -p tests/autoflip_reference_outputs/

jq -r '.clips[].slug' tests/real_content/manifest.json | while read -r slug; do
  clip_path=$(jq -r ".clips[] | select(.slug==\"$slug\") | .local_path // .slug + \".mp4\"" \
              tests/real_content/manifest.json)
  docker run --rm \
    -v "$CLIPAI_REAL_CONTENT_CACHE:/in:ro" \
    -v "$(pwd)/tests/autoflip_reference_outputs:/out" \
    clipai/autoflip-ref "/in/$clip_path" "/out/$slug.json" "9:16"
done
```

Each JSON is ~50-200 KB depending on clip duration. All 12 clips
should fit under 2 MB committed. Commit the JSONs as part of the same
change that adds a new clip to the manifest.

## JSON schema

Matches `backend/scripts/export_autoflip_compatible.py` so the
comparison harness can diff AutoFlip and ClipAI shapes uniformly.

```jsonc
{
  "tool": "mediapipe_autoflip",
  "version": "v0.10.9",
  "source_width": 1920,
  "source_height": 1080,
  "source_fps": 29.97,
  "aspect_ratio": "9:16",
  "events": [
    {
      "frame": 0,
      "t": 0.0,
      "crop_cx": 0.5,      // 0-1 normalized by source width
      "crop_cy": 0.5,      // 0-1 normalized by source height
      "crop_w": 0.2813,    // 9:16 crop on 16:9 source
      "crop_h": 1.0,
      "scene_change": true
    }
    // ... one event per source frame
  ]
}
```

## Known issues

### Stderr parser regex drift

`run_one.sh` parses per-frame crop decisions from MediaPipe's stderr
output. The format has shifted between releases. If a regenerated
cache has suspiciously few events (say, <10% of the source frame
count), dump the first 50 lines of the stderr log from inside the
container and grep for `crop_x`:

```bash
docker run --rm --entrypoint /bin/bash \
  -v "$CLIPAI_REAL_CONTENT_CACHE:/in:ro" \
  clipai/autoflip-ref -c '
    run_autoflip \
      --calculator_graph_config_file=/etc/autoflip/autoflip_graph.pbtxt \
      --input_side_packets="input_video_path=/in/panel_verzuz_tank_tyrese_10s.mp4,output_video_path=/tmp/out.mp4,aspect_ratio=9:16" \
    2>&1 | head -100
'
```

Update the regex in `run_one.sh` to match the observed format.

### `scene_change` field missing

Older MediaPipe builds don't emit the `scene_change` flag. The
parser already tolerates this (`bool(int(sc)) if sc is not None`)
and emits `scene_change=false` for every event. That means
`cut_to_hold_ratio` will treat the whole clip as one segment and
produce pessimistic numbers. If you see `n_segments=1` across
every AutoFlip output, the scene_change parser branch is the
culprit — MediaPipe needs to be bumped to a version that emits
the field.

## Build status in this sandbox

The Week 3 session that landed this file did not attempt the Docker
build. The Dockerfile and `run_one.sh` exist as deterministic
documentation of the reference environment; regeneration of the
cache is expected to happen on a machine where the Bazel build can
actually complete. The comparison harness at
`backend/scripts/compare_autoflip_vs_clipai.py` is written to read
whatever cached JSONs are present and gracefully skip clips that
have no AutoFlip output yet.
