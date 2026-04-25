# scripts/

## `bench.sh` — Fez human-parity bench runner

End-to-end runner for the human-reframe quality + parity bench. Drives the
flow from `tests/real_content/manifest.json` through fetch → extract →
bench → validate, all inside the running `clipai-app` container.

It survived ~15 iterations during the 2026-04-23 debugging session and is
the canonical way to reproduce that day's bench results.

### Prerequisites

Required before running:

- A running Docker container named `clipai-app`
  (`docker compose up -d` from the repo root brings it up).
- A Netscape-format YouTube cookies file at `<repo>/.secrets/cookies.txt`
  with `chmod 600`. The `.secrets/` directory is gitignored — never commit
  cookies. Use `./scripts/bench.sh --paste-cookies` to install them
  interactively.
- Inside the container: `yt-dlp`, `jq`, `ffmpeg`, `ffprobe`.
- On the host: `jq`, `docker`.

### Usage

```bash
# First time / after cookies expire — paste Netscape cookies, Ctrl-D:
./scripts/bench.sh --paste-cookies

# Normal run (interactive URL fixes if manifest has stale slugs):
./scripts/bench.sh

# Non-interactive run with explicit URL overrides:
./scripts/bench.sh --non-interactive \
    --set panel_joebudden_10s=https://www.youtube.com/watch?v=XXX \
    --set sports_boxing_8s=https://www.youtube.com/watch?v=YYY

# Help / full flag list:
./scripts/bench.sh --help
```

The script handles cookies validation, yt-dlp fetch with rate limiting and
H.264 preference, ffprobe-validated cache, slug-aware trim, sha256
auto-pinning, the human-parity bench, and a quality-bench fallback.

### Companion script

`trim_and_pin.sh` (repo root, not yet checked in) trims oversized cache
files to their slug-declared duration, clears stale sha256 pins, and
re-pins the trimmed files. Run it when a cache hash mismatch surfaces.
