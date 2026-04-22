#!/usr/bin/env bash
# Pull every clip listed in manifest.json into the cache dir,
# verifying sha256 when pinned. Skip clips that already exist and
# hash correctly.
#
# Usage:
#   ./fetch.sh              — fetch every slug
#   ./fetch.sh "panel_*"    — fetch only matching slugs (glob)
#   ./fetch.sh "anime_*"    — same, by content group
#
# Environment:
#   CLIPAI_REAL_CONTENT_CACHE   Cache directory. Defaults to
#                               /var/cache/clipai/real_content.
#   CLIPAI_YT_DLP               Explicit path to yt-dlp binary. If
#                               unset, the script looks up yt-dlp on
#                               $PATH. If not found, YouTube-style
#                               URLs fail with a helpful message.
#
# URL routing:
#   Direct media URLs (https://cdn.example.com/x.mp4, file://…,
#   s3://… signed link) resolve via curl.
#   Streaming-platform URLs — youtube.com, youtu.be, vimeo.com,
#   tiktok.com, instagram.com — resolve via yt-dlp when it is
#   available on PATH (or CLIPAI_YT_DLP). yt-dlp is NOT required;
#   clips that use direct URLs continue to work without it.
#
# First-fetch workflow (when the manifest has empty source_urls and
# empty sha256 entries):
#
#   1. Edit manifest.json to fill in source_url for each slug. For
#      local archive paths, use file://. For YouTube / TikTok, paste
#      the watch URL directly — yt-dlp picks the best 9:16-compatible
#      format automatically.
#   2. bash tests/real_content/fetch.sh
#   3. Copy the "new sha256 for <slug>" stderr lines back into the
#      manifest and re-run fetch.sh to lock them in.
#   4. Commit the updated manifest.json (clips stay uncommitted).

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
MANIFEST="$HERE/manifest.json"
CACHE="${CLIPAI_REAL_CONTENT_CACHE:-/var/cache/clipai/real_content}"

if ! command -v jq >/dev/null 2>&1; then
  echo "fetch.sh: jq is required but not installed" >&2
  exit 2
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "fetch.sh: curl is required but not installed" >&2
  exit 2
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "fetch.sh: manifest not found at $MANIFEST" >&2
  exit 2
fi

# Optional yt-dlp resolver. We look it up lazily so slugs that use
# direct URLs (file://, s3, signed CDN) work on hosts without yt-dlp.
YT_DLP_BIN="${CLIPAI_YT_DLP:-}"
if [[ -z "$YT_DLP_BIN" ]] && command -v yt-dlp >/dev/null 2>&1; then
  YT_DLP_BIN="$(command -v yt-dlp)"
fi

# Return 0 if the URL looks like a streaming-platform URL that needs
# yt-dlp; return 1 for direct-download URLs that curl handles.
_needs_yt_dlp() {
  local u="$1"
  case "$u" in
    *youtube.com/*|*youtu.be/*|*vimeo.com/*|*tiktok.com/*|*instagram.com/*)
      return 0 ;;
    *) return 1 ;;
  esac
}

mkdir -p "$CACHE"
filter="${1:-*}"

ok_count=0
fetched_count=0
skipped_count=0
failed_count=0

while IFS= read -r entry; do
  slug=$(jq -r '.slug' <<<"$entry")

  # Shell glob match against filter
  case "$slug" in
    $filter) : ;;
    *) continue ;;
  esac

  url=$(jq -r '.source_url // empty' <<<"$entry")
  sha=$(jq -r '.sha256 // empty' <<<"$entry")
  ext=$(jq -r '.ext // "mp4"' <<<"$entry")
  out="$CACHE/$slug.$ext"

  # Already present? Check hash and skip.
  if [[ -f "$out" ]]; then
    if [[ -n "$sha" ]]; then
      if echo "$sha  $out" | sha256sum -c --status - 2>/dev/null; then
        echo "ok: $slug (hash matches)" >&2
        ok_count=$((ok_count + 1))
        continue
      else
        echo "BAD HASH: $out — refetching" >&2
        rm -f "$out"
      fi
    else
      echo "present: $slug (no hash pinned — leave as-is)" >&2
      skipped_count=$((skipped_count + 1))
      continue
    fi
  fi

  if [[ -z "$url" ]]; then
    echo "no source_url for $slug; manually place file at $out" >&2
    skipped_count=$((skipped_count + 1))
    continue
  fi

  echo "fetching: $slug <- $url" >&2
  if _needs_yt_dlp "$url"; then
    if [[ -z "$YT_DLP_BIN" ]]; then
      echo "FETCH FAILED: $slug — url needs yt-dlp but it is not installed (pip install yt-dlp, or set CLIPAI_YT_DLP=/path/to/yt-dlp)" >&2
      failed_count=$((failed_count + 1))
      continue
    fi
    # yt-dlp picks the best progressive mp4 ≤ 1080p for reframing
    # benchmarks; adjust --format if the harness needs a different
    # resolution. --no-playlist so playlist URLs don't drag in the
    # whole channel.
    if ! "$YT_DLP_BIN" \
        --no-playlist \
        --format "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best" \
        --merge-output-format mp4 \
        --output "$out" \
        --quiet --no-warnings \
        "$url"; then
      echo "FETCH FAILED (yt-dlp): $slug" >&2
      failed_count=$((failed_count + 1))
      continue
    fi
  else
    if ! curl -fL "$url" -o "$out"; then
      echo "FETCH FAILED: $slug" >&2
      failed_count=$((failed_count + 1))
      continue
    fi
  fi

  if [[ -n "$sha" ]]; then
    if echo "$sha  $out" | sha256sum -c --status - 2>/dev/null; then
      echo "ok: $slug (fetched + hash verified)" >&2
      fetched_count=$((fetched_count + 1))
    else
      echo "HASH MISMATCH after fetch: $slug — the pinned sha256 is wrong or the source changed" >&2
      failed_count=$((failed_count + 1))
    fi
  else
    new_hash=$(sha256sum "$out" | awk '{print $1}')
    echo "new sha256 for $slug: $new_hash (paste into manifest.json)" >&2
    fetched_count=$((fetched_count + 1))
  fi
done < <(jq -c '.clips[]' "$MANIFEST")

echo "" >&2
echo "fetch summary: ok=$ok_count fetched=$fetched_count skipped=$skipped_count failed=$failed_count" >&2

if [[ $failed_count -gt 0 ]]; then
  exit 1
fi
