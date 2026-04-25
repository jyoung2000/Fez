#!/usr/bin/env bash
# bench.sh — all-in-one Fez human-parity bench runner.
#
# Replaces oneshot_bench.sh. Adds:
#   - Netscape-format validation for cookies file (no more silent overwrite)
#   - --paste-cookies mode: read cookies from stdin, validate, install 600
#   - Interactive URL fixer for manifest slugs with playlist/channel URLs
#   - --set SLUG=URL flag for non-interactive URL fixes (repeatable)
#   - Everything oneshot_bench.sh did
#
# Running this from scratch:
#   1. First time, or after cookies expire:
#        ./bench.sh --paste-cookies
#      then paste your Netscape-format cookies, Ctrl-D to finish.
#
#   2. Then just:
#        ./bench.sh
#      and it walks you through any URL fixes interactively, then runs
#      fetch → extract → bench → validate automatically.
#
#   3. For CI / non-interactive runs:
#        ./bench.sh --non-interactive \
#            --set panel_joebudden_10s=https://www.youtube.com/watch?v=XXX \
#            --set sports_boxing_8s=https://www.youtube.com/watch?v=YYY
#
# Options:
#   --paste-cookies           read cookies from stdin, install, exit
#   --cookies PATH            use cookies from PATH (default: repo/.secrets/cookies.txt)
#   --container NAME          docker container name (default: clipai-app)
#   --repo-host PATH          host path to repo (default: /mnt/user/appdata/clipai)
#   --set SLUG=URL            patch a slug's source_url (repeatable)
#   --non-interactive         refuse to prompt; exit if manifest has bad URLs
#   --fetch-only              stop after fetch
#   --skip-fetch              skip fetch phase
#   --skip-extract            skip trajectory extraction
#   --skip-bench              skip bench
#   --pin                     force re-pin baseline after bench
#   --no-pin                  do not pin or validate baseline
#   --help

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────
CONTAINER="clipai-app"
REPO_HOST="/mnt/user/appdata/clipai"
COOKIES_HOST=""       # resolved below; default is $REPO_HOST/.secrets/cookies.txt

CACHE_CONTAINER="/app/bench_cache/real_content"
BASELINE_PATH="/app/docs/human_parity_baseline.json"
RESULTS_PATH="/app/bench_cache/human_parity_run.json"

MODE="bench"           # bench | paste-cookies
INTERACTIVE=1
SKIP_SETUP=0
SKIP_FETCH=0
SKIP_EXTRACT=0
SKIP_BENCH=0
FETCH_ONLY=0
PIN_MODE="auto"        # auto | force | never

URL_FIX_SLUGS=()       # parallel to URL_FIX_URLS — regular arrays survive set -u
URL_FIX_URLS=()

# ── Logging helpers ────────────────────────────────────────────────
say()  { printf '\033[1;36m==\033[0m %s\n' "$*"; }
ok()   { printf '  \033[1;32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[1;33mWARN\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR\033[0m %s\n' "$*" >&2; exit 2; }
note() { printf '  \033[1;34m›\033[0m %s\n' "$*"; }

# ── Arg parsing ────────────────────────────────────────────────────
print_help() { sed -n '2,45p' "$0"; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --paste-cookies)     MODE="paste-cookies"; shift ;;
    --cookies)           COOKIES_HOST="$2"; shift 2 ;;
    --container)         CONTAINER="$2"; shift 2 ;;
    --repo-host)         REPO_HOST="$2"; shift 2 ;;
    --set)
      pair="$2"
      slug="${pair%%=*}"
      url="${pair#*=}"
      if [[ -z "$slug" || -z "$url" || "$slug" == "$url" ]]; then
        die "--set expected SLUG=URL, got '$pair'"
      fi
      URL_FIX_SLUGS+=("$slug")
      URL_FIX_URLS+=("$url")
      shift 2 ;;
    --non-interactive)   INTERACTIVE=0; shift ;;
    --fetch-only)        FETCH_ONLY=1; shift ;;
    --skip-setup)        SKIP_SETUP=1; shift ;;
    --skip-fetch)        SKIP_FETCH=1; shift ;;
    --skip-extract)      SKIP_EXTRACT=1; shift ;;
    --skip-bench)        SKIP_BENCH=1; shift ;;
    --pin)               PIN_MODE="force"; shift ;;
    --no-pin)            PIN_MODE="never"; shift ;;
    -h|--help)           print_help ;;
    *)                   die "unknown option: $1" ;;
  esac
done

[[ -z "$COOKIES_HOST" ]] && COOKIES_HOST="$REPO_HOST/.secrets/cookies.txt"
MANIFEST="$REPO_HOST/tests/real_content/manifest.json"

# ── Mode: paste-cookies ────────────────────────────────────────────
if [[ "$MODE" == "paste-cookies" ]]; then
  say "installing cookies to $COOKIES_HOST"
  mkdir -p "$(dirname "$COOKIES_HOST")"
  chmod 700 "$(dirname "$COOKIES_HOST")"

  echo ""
  echo "Paste your Netscape-format cookies below (the whole file contents,"
  echo "starting with '# Netscape HTTP Cookie File'), then press Ctrl-D:"
  echo ""

  tmp=$(mktemp)
  trap 'rm -f "$tmp"' EXIT
  cat > "$tmp"

  # Validate before installing
  first=$(head -1 "$tmp" || true)
  if [[ "$first" != *"Netscape HTTP Cookie File"* ]]; then
    die "pasted content is NOT a Netscape cookies file.
  First line was:  $first
  Expected:        # Netscape HTTP Cookie File
  Re-export cookies with 'Get cookies.txt LOCALLY' (Chrome/Firefox
  extension) and retry."
  fi

  # Basic sanity: should have at least one cookie line with youtube.com
  if ! grep -q "^\.youtube\.com" "$tmp"; then
    warn "no .youtube.com cookies found in pasted content"
    warn "install proceeding but yt-dlp may not authenticate"
  fi

  install -m 600 "$tmp" "$COOKIES_HOST"
  ok "installed $(wc -l < "$COOKIES_HOST") lines to $COOKIES_HOST (mode 600)"
  exit 0
fi

# ── Host preflight ─────────────────────────────────────────────────
command -v docker >/dev/null || die "docker not found on host"
command -v jq     >/dev/null || die "jq not found on host (apt install jq)"

if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  echo "Running containers:" >&2
  docker ps --format '  {{.Names}}' >&2
  die "container '$CONTAINER' not running"
fi

[[ -d "$REPO_HOST/tests/real_content" ]] || die "$REPO_HOST/tests/real_content not found"
[[ -f "$MANIFEST" ]] || die "manifest not found at $MANIFEST"

# ── Cookies validation ─────────────────────────────────────────────
say "validating cookies"

if [[ ! -f "$COOKIES_HOST" ]]; then
  die "cookies file not found at $COOKIES_HOST.
  Install with:  $0 --paste-cookies
  or supply a path with:  $0 --cookies /path/to/cookies.txt"
fi

first_line=$(head -1 "$COOKIES_HOST" || true)
if [[ "$first_line" != *"Netscape HTTP Cookie File"* ]]; then
  die "$COOKIES_HOST is NOT a Netscape cookies file.
  First line was:  $first_line
  Expected:        # Netscape HTTP Cookie File
  This usually means something overwrote the cookies file with other
  content (a script, JSON, etc.). Re-install with:
    $0 --paste-cookies"
fi

# Make sure perms are tight — will fail if we're not owner, which is fine
if [[ "$(stat -c '%a' "$COOKIES_HOST" 2>/dev/null || echo '')" != "600" ]]; then
  chmod 600 "$COOKIES_HOST" 2>/dev/null || warn "couldn't set 600 on $COOKIES_HOST"
fi
ok "cookies valid ($(wc -l < "$COOKIES_HOST") lines)"

# ── Apply --set URL fixes ──────────────────────────────────────────
if [[ ${#URL_FIX_SLUGS[@]} -gt 0 ]]; then
  say "applying ${#URL_FIX_SLUGS[@]} --set URL fix(es)"
  for i in "${!URL_FIX_SLUGS[@]}"; do
    slug="${URL_FIX_SLUGS[$i]}"
    url="${URL_FIX_URLS[$i]}"
    if ! jq -e --arg s "$slug" '.clips[] | select(.slug == $s)' "$MANIFEST" >/dev/null; then
      warn "slug '$slug' not found in manifest — skipping"
      continue
    fi
    # Idempotent: if URL already matches, don't touch sha256. Otherwise
    # a no-op re-run would wipe the cache every time.
    current_url=$(jq -r --arg s "$slug" '.clips[] | select(.slug == $s) | .source_url' "$MANIFEST")
    if [[ "$current_url" == "$url" ]]; then
      note "$slug already set to $url — no change"
      continue
    fi
    tmp=$(mktemp)
    jq --arg slug "$slug" --arg url "$url" \
      '.clips |= map(if .slug == $slug then .source_url = $url | .sha256 = "" else . end)' \
      "$MANIFEST" > "$tmp" && mv "$tmp" "$MANIFEST"
    ok "patched $slug → $url"
  done
fi

# ── Lint manifest for bad URLs ─────────────────────────────────────
lint_bad_urls() {
  jq -r '.clips[] |
    select(.source_url != "" and .source_url != null) |
    select(.source_url | test("playlist\\?list=|/@[A-Za-z0-9_-]+/?$|/@[A-Za-z0-9_-]+/videos/?$|/channel/")) |
    .slug + "|" + .source_url' "$MANIFEST"
}

bad_urls=$(lint_bad_urls)

if [[ -n "$bad_urls" ]]; then
  say "bad URLs detected in manifest (playlist or channel, not single video)"
  echo "$bad_urls" | awk -F'|' '{printf "  %-40s  %s\n", $1, $2}' >&2

  if [[ "$INTERACTIVE" != "1" ]] || [[ ! -t 0 ]]; then
    die "stdin is not a TTY or --non-interactive was set. Fix with:
  $0 --set SLUG=URL --set SLUG=URL ...
or run interactively from a terminal."
  fi

  # Need the yt-dlp shim inside the container for suggestions. Copy
  # cookies + build shim now (harmless if done twice).
  say "prepping container for suggestions"
  docker cp "$COOKIES_HOST" "$CONTAINER:/root/.yt-dlp-cookies.txt" >/dev/null
  docker exec "$CONTAINER" chmod 600 /root/.yt-dlp-cookies.txt

  docker exec "$CONTAINER" bash -c '
    set -e
    if ! command -v yt-dlp >/dev/null 2>&1; then
      pip install -q yt-dlp >/dev/null
    fi
    REAL=$(which -a yt-dlp | grep -v /opt/bench-shims | head -1)
    mkdir -p /opt/bench-shims
    cat > /opt/bench-shims/yt-dlp <<WRAPPER
#!/usr/bin/env bash
set -eu
COOKIES=/root/.yt-dlp-cookies.txt
if [[ -r "\$COOKIES" ]]; then
  exec "$REAL" --cookies "\$COOKIES" "\$@"
else
  exec "$REAL" "\$@"
fi
WRAPPER
    chmod +x /opt/bench-shims/yt-dlp
  ' >/dev/null

  say "entering interactive URL fix mode"

  while IFS='|' read -r slug url; do
    [[ -z "$slug" ]] && continue

    echo ""
    echo "── $slug ──"
    echo "   current: $url"
    echo ""
    note "fetching 5 recent uploads from that channel/playlist..."

    # Query yt-dlp for 5 recent uploads. Use --playlist-end 5 so we don't
    # blow through the whole channel. The shim handles cookies.
    sug=$(docker exec "$CONTAINER" /opt/bench-shims/yt-dlp \
      --flat-playlist --no-warnings --skip-download --ignore-errors \
      --print '%(id)s|%(title).80B' \
      --playlist-end 5 \
      "$url" 2>/dev/null || true)

    declare -a cand_urls=()
    declare -a cand_titles=()
    idx=0
    if [[ -n "$sug" ]]; then
      while IFS='|' read -r vid title; do
        [[ -z "$vid" ]] && continue
        # Filter out shorts-only accounts that return hashes etc.
        [[ "$vid" =~ ^[A-Za-z0-9_-]{11}$ ]] || continue
        idx=$((idx+1))
        cand_urls[$idx]="https://www.youtube.com/watch?v=$vid"
        cand_titles[$idx]="$title"
        printf "   [%d] %s\n       %s\n" "$idx" "${cand_urls[$idx]}" "$title"
      done <<< "$sug"
    fi
    if [[ $idx -eq 0 ]]; then
      echo "   (no suggestions — enter a URL manually or skip)"
    fi

    echo ""
    while true; do
      printf "   choose [1-%d], paste URL, or 'skip': " "$idx"
      read -r choice < /dev/tty
      if [[ "$choice" == "skip" || "$choice" == "s" ]]; then
        warn "skipping $slug — next run will still error on this URL"
        new_url=""
        break
      elif [[ "$choice" =~ ^[0-9]+$ ]] && [[ -n "${cand_urls[$choice]:-}" ]]; then
        new_url="${cand_urls[$choice]}"
        break
      elif [[ "$choice" =~ ^https?://.*watch\?v= ]] || \
           [[ "$choice" =~ ^https?://youtu\.be/ ]]; then
        new_url="$choice"
        break
      else
        echo "   invalid — enter a number from the list, a full watch?v= URL, or 'skip'"
      fi
    done

    if [[ -n "$new_url" ]]; then
      tmp=$(mktemp)
      jq --arg slug "$slug" --arg url "$new_url" \
        '.clips |= map(if .slug == $slug then .source_url = $url | .sha256 = "" else . end)' \
        "$MANIFEST" > "$tmp" && mv "$tmp" "$MANIFEST"
      ok "patched $slug → $new_url"
    fi

    # Clean per-iteration arrays
    unset cand_urls cand_titles
    declare -a cand_urls=()
    declare -a cand_titles=()
  done <<< "$bad_urls"

  # Re-lint to confirm
  bad_urls=$(lint_bad_urls)
  if [[ -n "$bad_urls" ]]; then
    say "still have bad URLs after fix round:"
    echo "$bad_urls" | awk -F'|' '{printf "  %-40s  %s\n", $1, $2}' >&2
    die "re-run to continue fixing, or pass --non-interactive to skip the remaining"
  fi
  ok "all URLs clean"
fi

# ── Ship artifacts into container ──────────────────────────────────
say "shipping cookies + tests into $CONTAINER"
docker cp "$COOKIES_HOST" "$CONTAINER:/root/.yt-dlp-cookies.txt" >/dev/null
docker exec "$CONTAINER" chmod 600 /root/.yt-dlp-cookies.txt
# Delete existing /app/tests first — docker cp nests directories when
# the target already exists, which would leave a stale manifest at
# /app/tests/real_content/manifest.json while the new patched one ends
# up uselessly at /app/tests/tests/real_content/manifest.json.
docker exec "$CONTAINER" rm -rf /app/tests
docker cp "$REPO_HOST/tests" "$CONTAINER:/app/tests" >/dev/null

# Build comma-separated list of slugs being modified by --set (for
# targeted cache wipe inside the container — must not wipe unrelated
# slugs that still have valid pinned cache files).
FIXED_SLUGS_CSV=""
if [[ ${#URL_FIX_SLUGS[@]} -gt 0 ]]; then
  FIXED_SLUGS_CSV=$(IFS=,; echo "${URL_FIX_SLUGS[*]}")
fi

# ── Dispatch into container ────────────────────────────────────────
say "dispatching bench pipeline into $CONTAINER"
# Single-quoted heredoc — no host-side expansion. Values cross via -e.
# Crucially: disable `set -e` around docker exec so a non-zero exit from
# the inner pipeline (e.g. bench produced 0 reports, or YouTube rate
# limits) doesn't kill this outer script before the sync-back runs.
# Without this, auto-pasted sha256 pins never make it back to the host.
set +e
docker exec \
  -e BENCH_CACHE="$CACHE_CONTAINER" \
  -e BASELINE_PATH="$BASELINE_PATH" \
  -e RESULTS_PATH="$RESULTS_PATH" \
  -e SKIP_SETUP="$SKIP_SETUP" \
  -e SKIP_FETCH="$SKIP_FETCH" \
  -e SKIP_EXTRACT="$SKIP_EXTRACT" \
  -e SKIP_BENCH="$SKIP_BENCH" \
  -e FETCH_ONLY="$FETCH_ONLY" \
  -e PIN_MODE="$PIN_MODE" \
  -e FIXED_SLUGS_CSV="$FIXED_SLUGS_CSV" \
  -i "$CONTAINER" bash <<'INNER'
set -euo pipefail

log()  { printf '  \033[1;34m›\033[0m %s\n' "$*"; }
ok()   { printf '  \033[1;32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[1;33mWARN\033[0m %s\n' "$*" >&2; }
die()  { printf '  \033[1;31mERROR\033[0m %s\n' "$*" >&2; exit 2; }

cd /app
export PYTHONPATH=/app

MANIFEST=/app/tests/real_content/manifest.json
HUMAN_TRAJ_DIR=/app/data/human_trajectories

# ── 1. Dependencies + shim ─────────────────────────────────────────
if [[ "$SKIP_SETUP" != "1" ]]; then
  log "ensuring jq + yt-dlp + shim"
  if ! command -v jq >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq jq >/dev/null
  fi
  if ! command -v yt-dlp >/dev/null 2>&1; then
    pip install -q yt-dlp
  fi

  REAL_YT_DLP="$(which -a yt-dlp | grep -v /opt/bench-shims | head -1)"
  [[ -n "$REAL_YT_DLP" && -x "$REAL_YT_DLP" ]] || die "can't find underlying yt-dlp"

  mkdir -p /opt/bench-shims
  cat > /opt/bench-shims/yt-dlp <<WRAPPER
#!/usr/bin/env bash
# Auto-injects authenticated cookies + rate-limit friendly defaults +
# H.264 format preference. We explicitly avoid AV1 because OpenCV /
# ffmpeg inside the clipai container doesn't have AV1 decode enabled;
# any AV1-encoded download would silently produce zero frames during
# extraction. H.264 / VP9 fall back work on standard ffmpeg builds.
set -eu
COOKIES=/root/.yt-dlp-cookies.txt
RATE_ARGS=(--sleep-interval 3 --max-sleep-interval 8 --sleep-requests 1 --retries 3)
# Prefer H.264 bestvideo + best audio, fall back to "best" (which may
# still be AV1, but at least we tried the codec we know works first).
FORMAT_ARGS=(--format 'bestvideo[vcodec^=avc1]+bestaudio/best[vcodec^=avc1]/bestvideo[vcodec!=av01]+bestaudio/best' --merge-output-format mp4)
if [[ -r "\$COOKIES" ]]; then
  exec "$REAL_YT_DLP" --cookies "\$COOKIES" "\${RATE_ARGS[@]}" "\${FORMAT_ARGS[@]}" "\$@"
else
  exec "$REAL_YT_DLP" "\${RATE_ARGS[@]}" "\${FORMAT_ARGS[@]}" "\$@"
fi
WRAPPER
  chmod +x /opt/bench-shims/yt-dlp
  ok "shim at /opt/bench-shims/yt-dlp (real: $REAL_YT_DLP)"
fi

export PATH=/opt/bench-shims:$PATH
export CLIPAI_YT_DLP=/opt/bench-shims/yt-dlp
export CLIPAI_REAL_CONTENT_CACHE="$BENCH_CACHE"
mkdir -p "$BENCH_CACHE"

# ── Cache coherence: wipe cache files ONLY for slugs the user just
# changed via --set. Do NOT wipe clips that merely have empty sha256
# for other reasons (e.g., a previous run's pin didn't persist) —
# those files might still be valid and re-downloading during rate
# limits just wastes time.
log "cache coherence check"
wiped=0
if [[ -n "${FIXED_SLUGS_CSV:-}" ]] && command -v jq >/dev/null 2>&1; then
  # Map each --set slug to its extension, wipe both horizontal + vertical cache files
  IFS=',' read -ra _fixed_slugs <<< "$FIXED_SLUGS_CSV"
  for slug in "${_fixed_slugs[@]}"; do
    [[ -z "$slug" ]] && continue
    ext=$(jq -r --arg s "$slug" '.clips[] | select(.slug == $s) | .ext // "mp4"' "$MANIFEST")
    [[ -z "$ext" ]] && ext="mp4"
    for f in "$BENCH_CACHE/$slug.$ext" "$BENCH_CACHE/$slug.vertical.$ext"; do
      if [[ -f "$f" ]]; then
        rm -f "$f"
        log "  wiped stale (from --set): $f"
        wiped=$((wiped + 1))
      fi
    done
  done
fi
[[ "$wiped" -eq 0 ]] && log "  (nothing to wipe — no --set fixes, or no stale files)"

# ── 2. Download missing clips directly with yt-dlp / curl ──────────
# NOTE: repo's fetch.sh uses plain curl, which saves YouTube's HTML
# embed page as an .mp4. We bypass it for all URL-backed clips and
# only let fetch.sh do the hash-verification pass at the end.
FETCH_LOG=/tmp/fetch.log
DOWNLOAD_LOG=/tmp/download.log
: > "$DOWNLOAD_LOG"

if [[ "$SKIP_FETCH" != "1" ]]; then
  log "downloading source clips (log: $DOWNLOAD_LOG)"

  dl_fetched=0 dl_cached=0 dl_failed=0 dl_missing=0
  src_list=$(jq -r '.clips[] |
    (.slug + "\t" + (.source_url // "") + "\t" + (.ext // "mp4"))' \
    "$MANIFEST")

  while IFS=$'\t' read -r slug src_url ext; do
    [[ -z "$slug" ]] && continue
    out="$BENCH_CACHE/$slug.$ext"

    if [[ -z "$src_url" ]]; then
      log "  [$slug] no source_url — place file at $out manually"
      dl_missing=$((dl_missing + 1))
      continue
    fi
    if [[ -f "$out" ]] && [[ $(stat -c %s "$out" 2>/dev/null || echo 0) -gt 100000 ]]; then
      # Present and >100KB. Also verify it's actually playable video —
      # ffprobe catches HTML stubs and truncated MP4s that fetch.sh's
      # naive curl saved earlier.
      if ffprobe -v error -show_entries format=duration -of csv=p=0 "$out" >/dev/null 2>&1; then
        log "  [$slug] cached, size=$(du -h "$out" | cut -f1)"
        dl_cached=$((dl_cached + 1))
        continue
      else
        warn "  [$slug] cached file is broken (ffprobe failed) — wiping and re-fetching"
        rm -f "$out"
      fi
    fi

    # Wipe any stub file under the size floor before re-downloading
    [[ -f "$out" ]] && rm -f "$out"

    # Decide route
    case "$src_url" in
      *youtube.com*|*youtu.be*|*tiktok.com*|*instagram.com*|*vimeo.com*)
        log "  [$slug] yt-dlp → $src_url"
        if /opt/bench-shims/yt-dlp \
            --no-warnings --no-playlist \
            --merge-output-format mp4 \
            --output "$out" \
            "$src_url" >>"$DOWNLOAD_LOG" 2>&1; then
          dl_fetched=$((dl_fetched + 1))
        else
          warn "  [$slug] yt-dlp failed (see $DOWNLOAD_LOG)"
          dl_failed=$((dl_failed + 1))
        fi
        ;;
      file://*)
        src_path="${src_url#file://}"
        if [[ -f "$src_path" ]]; then
          cp "$src_path" "$out"
          log "  [$slug] file:// copied"
          dl_fetched=$((dl_fetched + 1))
        else
          warn "  [$slug] file not found at $src_path"
          dl_failed=$((dl_failed + 1))
        fi
        ;;
      http*|https*)
        log "  [$slug] curl → $src_url"
        if curl -fL --retry 3 --retry-delay 2 -s \
            -o "$out" "$src_url" 2>>"$DOWNLOAD_LOG"; then
          dl_fetched=$((dl_fetched + 1))
        else
          warn "  [$slug] curl failed"
          dl_failed=$((dl_failed + 1))
        fi
        ;;
      *)
        warn "  [$slug] unrecognized source_url scheme: $src_url"
        dl_failed=$((dl_failed + 1))
        ;;
    esac

    # Post-download sanity check: non-empty and ffprobe-readable
    if [[ -f "$out" ]]; then
      size_kb=$(( $(stat -c %s "$out") / 1024 ))
      if [[ $size_kb -lt 100 ]]; then
        warn "  [$slug] downloaded but only ${size_kb}KB — probably broken, wiping"
        rm -f "$out"
        dl_failed=$((dl_failed + 1))
        dl_fetched=$((dl_fetched - 1))
      elif ! ffprobe -v error -show_entries format=duration -of csv=p=0 "$out" >/dev/null 2>&1; then
        warn "  [$slug] downloaded but ffprobe can't read it — wiping"
        rm -f "$out"
        dl_failed=$((dl_failed + 1))
        dl_fetched=$((dl_fetched - 1))
      else
        dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$out" 2>/dev/null || echo "?")
        log "  [$slug] ✓ ${size_kb}KB duration=${dur}s"
      fi
    fi
  done <<< "$src_list"

  log "downloads: fetched=$dl_fetched cached=$dl_cached failed=$dl_failed no_url=$dl_missing"

  # Now let fetch.sh do its hash-verify pass over the real files we just
  # downloaded. It'll emit "new sha256" for unpinned clips; we auto-paste.
  log "running fetch.sh for hash verification (log: $FETCH_LOG)"
  set +e
  bash /app/tests/real_content/fetch.sh 2>&1 | tee "$FETCH_LOG"
  set -e

  new_hashes=$(grep -E '^new sha256 for ' "$FETCH_LOG" || true)
  if [[ -n "$new_hashes" ]]; then
    n=$(echo "$new_hashes" | wc -l)
    log "auto-pinning $n new sha256s into manifest"
    while IFS= read -r line; do
      slug=$(echo "$line" | awk '{print $4}' | sed 's/:$//')
      hash=$(echo "$line" | awk '{print $5}')
      [[ -z "$slug" || -z "$hash" ]] && continue
      tmp=$(mktemp)
      jq --arg slug "$slug" --arg hash "$hash" \
        '.clips |= map(if .slug == $slug then .sha256 = $hash else . end)' \
        "$MANIFEST" > "$tmp"
      mv "$tmp" "$MANIFEST"
      ok "  pinned $slug → $hash"
    done <<< "$new_hashes"
  fi

  ok_count=$(jq -r '[.clips[] | select(.sha256 != null and .sha256 != "")] | length' "$MANIFEST")
  log "fetch done: $ok_count clips pinned"
fi

if [[ "$FETCH_ONLY" == "1" ]]; then
  log "--fetch-only set, stopping here"
  exit 0
fi

# ── 2b. Fetch ground-truth vertical clips ──────────────────────────
# extract_human_trajectories.py expects $CACHE/$slug.vertical.<ext> to
# exist for every slug with a ground_truth_vertical_url. fetch.sh only
# handles horizontal sources. Handle verticals here via yt-dlp directly.
#
# Known limitation: many verticals in the manifest are channel handles
# (@nba, @daznboxing) rather than specific videos. yt-dlp can't download
# a generic channel as one video, so we skip those — they'll need
# specific watch?v= or /video/ URLs to work. Not a blocker for the
# quality bench, which doesn't need verticals.
log "fetching ground-truth vertical clips"
v_fetched=0 v_skipped=0 v_failed=0 v_missing_src=0 v_bad_url=0

vert_list=$(jq -r '.clips[] |
  select(.ground_truth_vertical_url != null and .ground_truth_vertical_url != "") |
  (.slug + "\t" + .ground_truth_vertical_url + "\t" + (.ext // "mp4"))' \
  "$MANIFEST")

while IFS=$'\t' read -r slug vert_url ext; do
  [[ -z "$slug" || -z "$vert_url" ]] && continue
  src_file="$BENCH_CACHE/$slug.$ext"
  vert_file="$BENCH_CACHE/$slug.vertical.$ext"

  if [[ ! -f "$src_file" ]]; then
    log "  [$slug] no source cached — vertical would be useless, skipping"
    v_missing_src=$((v_missing_src + 1))
    continue
  fi
  if [[ -f "$vert_file" ]]; then
    log "  [$slug] vertical already cached — ok"
    v_skipped=$((v_skipped + 1))
    continue
  fi

  # Bail on channel/playlist URLs — yt-dlp would try to grab the whole
  # channel. These are manifest-population bugs, not fetch bugs.
  if [[ "$vert_url" =~ /@[A-Za-z0-9_.]+/?$ ]] || \
     [[ "$vert_url" =~ /@[A-Za-z0-9_.]+/(videos|reels|shorts)/?$ ]] || \
     [[ "$vert_url" =~ /channel/ ]] || \
     [[ "$vert_url" =~ playlist\?list= ]]; then
    log "  [$slug] vertical URL is a channel/playlist ($vert_url) — skipping"
    log "         fix: populate with a specific video URL in the manifest"
    v_bad_url=$((v_bad_url + 1))
    continue
  fi

  log "  [$slug] fetching vertical from $vert_url"
  if /opt/bench-shims/yt-dlp \
      --no-warnings --no-playlist \
      --format "best[ext=mp4]/best" \
      --merge-output-format mp4 \
      --output "$vert_file" \
      "$vert_url" 2>>/tmp/vertical_fetch.log >/dev/null; then
    v_fetched=$((v_fetched + 1))
    log "  [$slug] ✓ vertical fetched"
  else
    warn "  [$slug] vertical fetch failed (see /tmp/vertical_fetch.log)"
    v_failed=$((v_failed + 1))
  fi
done <<< "$vert_list"

log "verticals: fetched=$v_fetched cached=$v_skipped failed=$v_failed missing_src=$v_missing_src bad_url=$v_bad_url"

# ── 3. Extract trajectories ────────────────────────────────────────
if [[ "$SKIP_EXTRACT" != "1" ]]; then
  log "extracting human ground-truth trajectories"
  python3 -m backend.scripts.extract_human_trajectories || \
    warn "extract_human_trajectories exited non-zero — continuing"
  traj_count=$(find "$HUMAN_TRAJ_DIR" -name '*.jsonl' 2>/dev/null | wc -l)
  ok "extracted $traj_count trajectory files"
fi

# ── 4. Bench ───────────────────────────────────────────────────────
if [[ "$SKIP_BENCH" != "1" ]]; then
  mkdir -p "$(dirname "$RESULTS_PATH")"
  log "running human-parity bench (needs trajectory files) → $RESULTS_PATH"
  python3 -m backend.scripts.run_human_parity_bench --out "$RESULTS_PATH" || true
  run_count=$(jq '.runs | length' "$RESULTS_PATH" 2>/dev/null || echo 0)
  log "human-parity bench: $run_count reports"

  # Always also run the quality-only bench. This one scores intrinsic
  # reframing quality (chin-clip rate, jitter, saccade spacing, etc.)
  # against any cached source files and does not need human trajectories.
  # That's our "at minimum we got SOME signal" fallback when the
  # ground-truth vertical pairs aren't populated.
  QUALITY_RESULTS="$(dirname "$RESULTS_PATH")/human_reframe_quality.json"
  log "running quality bench (no trajectory needed) → $QUALITY_RESULTS"
  python3 -m backend.scripts.measure_human_reframe_quality --json \
    > "$QUALITY_RESULTS" 2>&1 || warn "quality bench exited non-zero"

  if [[ -s "$QUALITY_RESULTS" ]]; then
    log "quality bench — full JSON at $QUALITY_RESULTS"
    # Try to extract just the summary line for a quick human read
    quality_summary=$(jq -r '. | "  overall_pass=\(.overall_pass // "?")  fixtures=\(.fixtures | length)  fails=\([.fixtures[] | select(.pass==false) | .fixture] | join(","))"' "$QUALITY_RESULTS" 2>/dev/null || true)
    if [[ -n "$quality_summary" ]]; then
      log "quality bench summary:"
      echo "$quality_summary"
    fi
    # Then the full set of per-fixture pass/fail, without the metric
    # details (those are in the full JSON on disk)
    log "per-fixture results:"
    jq -r '.fixtures[]? |
      "  \(.fixture): pass=\(.pass)  head_rate=\(.head_rate // "n/a")  chin_rate=\(.chin_rate // "n/a")  jitter_p95=\(.jitter_p95 // "n/a")  critic_repair=\(.critic_repair_rate // "n/a")  " +
      (if (.fails // []) | length > 0 then "FAILS: " + (.fails | join(", ")) else "" end)' \
      "$QUALITY_RESULTS" 2>/dev/null | sed 's/^/    /' || cat "$QUALITY_RESULTS" | tail -40 | sed 's/^/    /'
  fi

  if [[ "$run_count" == "0" ]]; then
    warn "human-parity bench produced zero reports (no trajectory files)"
    warn "this is expected if ground_truth_vertical_url values are channel handles"
    warn "quality bench above has your SOTA-path numbers for now"
  else
    # ── 5. Pin or validate (only runs if we got parity numbers) ─────
    case "$PIN_MODE" in
      force)
        log "forcing new baseline at $BASELINE_PATH"
        python3 -m backend.scripts.validate_human_parity \
          --results "$RESULTS_PATH" --pin "$BASELINE_PATH" ;;
      never)
        log "--no-pin: skipping pin/validate" ;;
      auto)
        if [[ -f "$BASELINE_PATH" ]]; then
          log "validating vs $BASELINE_PATH"
          python3 -m backend.scripts.validate_human_parity \
            --results "$RESULTS_PATH" --baseline "$BASELINE_PATH"
        else
          log "no baseline — pinning current run"
          python3 -m backend.scripts.validate_human_parity \
            --results "$RESULTS_PATH" --pin "$BASELINE_PATH"
        fi ;;
    esac
  fi

  ok "bench complete"
  echo ""
  echo "  human-parity:  $RESULTS_PATH ($run_count clips)"
  echo "  quality:       $QUALITY_RESULTS"
  echo "  baseline:      $BASELINE_PATH"
  echo ""
  if [[ "$run_count" -gt 0 ]]; then
    # Per-clip summary
    jq -r '.runs | to_entries[] |
      "  \(.key):  mae_cx=\(.value.mae_cx // "n/a")  mae_cy=\(.value.mae_cy // "n/a")  macro_f1=\(.value.framing_macro_f1 // "n/a")"' \
      "$RESULTS_PATH"
  fi
fi
INNER

inner_rc=$?
set -e

# ── Sync updated manifest + baseline back to host ──────────────────
# This MUST run regardless of inner_rc so sha256 pins written inside
# the container always propagate to the host manifest. A failed inner
# pipeline is still often a partial success.
say "syncing manifest back to host"
docker cp "$CONTAINER:/app/tests/real_content/manifest.json" \
  "$MANIFEST" >/dev/null

if docker exec "$CONTAINER" test -f "$BASELINE_PATH"; then
  mkdir -p "$REPO_HOST/docs"
  docker cp "$CONTAINER:$BASELINE_PATH" \
    "$REPO_HOST/docs/human_parity_baseline.json" >/dev/null
  ok "baseline synced to $REPO_HOST/docs/human_parity_baseline.json"
fi

[[ "$inner_rc" -ne 0 ]] && die "container run exited $inner_rc"
say "done"
