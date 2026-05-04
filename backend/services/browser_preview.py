"""Generate browser-playable previews of source videos.

The share endpoint (and the authenticated ``/api/files`` endpoint)
stream the *source* video to the recipient's browser so the in-page
``<video>`` element can scrub and play it. The source is whatever the
owner uploaded — frequently ``.mkv`` with AC3 / DTS / EAC3 / FLAC audio
(standard for anime and Blu-ray rips).

HTML5 ``<video>`` only guarantees playback for the codec triple
``MP4 (H.264 + AAC)`` and ``WebM (VP8/VP9 + Opus/Vorbis)``. Browsers
will happily play an ``.mkv`` with H.264 video but silently drop the
audio track when it's AC3 / DTS / EAC3 / TrueHD / FLAC. That's why a
Speed Racer ``.mkv`` rendered a silent preview on the share page — the
video stream decoded fine, the audio stream was a codec the browser
couldn't decode.

This module detects those cases via ``ffprobe`` and produces a cached
browser-friendly derivative (``browser_preview.mp4``) next to the
source. The video stream is **copied** when it's already H.264 (fast,
lossless); audio is re-encoded to AAC. For non-H.264 sources we
fall back to a full re-encode.

Generation is idempotent and process-safe via a sentinel lock file so
concurrent Range requests during the first scrub don't race and
produce two overlapping FFmpegs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# Canonical filename of the cached browser preview. The version
# suffix (currently ``v3``) bumps whenever the encoding pipeline
# changes in a way that invalidates older cached previews. Operators
# can grep the startup log for "browser_preview: module loaded" to
# confirm at a glance whether the running container is on the latest
# code — the previous regression had a fresh build emitting v2 names
# while the source had moved to v3, and the only signal was a 60-min
# retry storm of ``Unable to choose an output format`` errors.
_PREVIEW_FILENAME = "browser_preview.v3.mp4"

logger.info(
    "browser_preview: module loaded, target filename = %s",
    _PREVIEW_FILENAME,
)


# Short-lived cache of ``_probe`` results keyed on ``(path, mtime)``.
# Every HTML5 ``<video>`` range request flows through
# ``ensure_browser_preview_status``. For sources that don't need a
# preview (already browser-friendly MP4s — the common case) the fast
# path at ``isfile(target_path)`` misses, we fall through to ``_probe``,
# which forks ``ffprobe``. Heavy scrubbing spawns dozens of ffprobes
# per second. Caching by ``(path, mtime)`` keeps us correct when the
# source is replaced (the mtime invalidates the entry) while avoiding
# the subprocess storm during a single scrub session.
_probe_cache: dict[tuple[str, float], "_ProbeResult"] = {}
_PROBE_CACHE_TTL = 300.0  # seconds
_probe_cache_expiry: dict[tuple[str, float], float] = {}
_PROBE_CACHE_MAX = 256


def _probe_cache_get(path: str, mtime: float) -> Optional["_ProbeResult"]:
    key = (path, mtime)
    exp = _probe_cache_expiry.get(key)
    if exp is None or exp <= time.monotonic():
        _probe_cache.pop(key, None)
        _probe_cache_expiry.pop(key, None)
        return None
    return _probe_cache.get(key)


def _probe_cache_put(path: str, mtime: float, result: "_ProbeResult") -> None:
    key = (path, mtime)
    _probe_cache[key] = result
    _probe_cache_expiry[key] = time.monotonic() + _PROBE_CACHE_TTL
    # Opportunistic eviction — FIFO-ish by dict insertion order.
    if len(_probe_cache) > _PROBE_CACHE_MAX:
        for _k in list(_probe_cache.keys())[:32]:
            _probe_cache.pop(_k, None)
            _probe_cache_expiry.pop(_k, None)


# Audio codecs an HTML5 ``<video>`` / ``<audio>`` element can decode
# reliably across Chrome/Edge/Safari/Firefox. Anything else needs to
# be transcoded to AAC before the browser can hear it.
#
# Notable NOT in this list and therefore always transcoded:
#   ac3, eac3, dts, dts-hd, truehd, flac, pcm_*, wmav2
_BROWSER_AUDIO_CODECS = frozenset({
    "aac",
    "mp3",
    "opus",
    "vorbis",
})

# Video codecs the big-four browsers can render. H.265/HEVC is
# intentionally excluded — only Safari plays it reliably and even then
# only when wrapped in the right container.
_BROWSER_VIDEO_CODECS = frozenset({
    "h264",
    "avc1",
    "vp8",
    "vp9",
    "av1",
})

# Containers where the browser-friendly codecs above are actually
# playable. MKV is excluded by design: Safari and iOS don't support
# it at all, and Chromium's support is patchy.
_BROWSER_CONTAINERS = frozenset({
    ".mp4",
    ".m4v",
    ".webm",
})


# Preview-encoding targets. Chosen so the encode itself completes in
# the 15-30 second window the in-page player budget allows:
#   * 720p (max width 1280) is more than enough for an in-browser
#     preview window. Encoding 1080p was the largest single cost in
#     the old pipeline — dropping to 720p cuts encoder pixel work by
#     ~2.25× without any visible quality loss in a preview-sized
#     viewport.
#   * 3.5 Mbps is plenty for 720p, fits comfortably inside the
#     progressive-download budget on ordinary connections, and the
#     lower target rate also speeds up the encoder (less rate-control
#     work per frame). Blu-ray rips often sit at 15-40 Mbps; this is
#     6-10× lower and that's the whole point of a preview.
_PREVIEW_MAX_WIDTH = 1280
_PREVIEW_MAX_BITRATE_KBPS = 3500
# Short GOP → fine-grained seek. 2 seconds lets the browser land
# within one keyframe of any scrub target instead of jumping 4–10 s
# at a time the way factory-default libx264 does.
_PREVIEW_KEYFRAME_INTERVAL_SEC = 2.0

# Copy-mode thresholds. These are intentionally **larger** than the
# re-encode preview cap above: copy is essentially instantaneous (a
# remux, not a re-encode), so as long as the source is something the
# browser can decode comfortably (H.264 up to 1080p at sane bitrates),
# we'd rather keep the user's bytes than spend 15+ s re-encoding to a
# slightly smaller derivative. Re-encoding only kicks in for
# genuinely browser-hostile sources (H.265, 4K, ≥8 Mbps rips) — see
# :func:`_needs_preview` and :func:`_should_copy_video`.
_COPY_MAX_WIDTH = 1920
_COPY_MAX_BITRATE_KBPS = 8000


@dataclass
class _ProbeResult:
    video_codec: str
    audio_codec: str
    has_audio: bool
    width: int
    height: int
    fps: float
    # Video-stream bitrate if ffprobe reported one, else 0. Some
    # containers (MKV variable-rate streams) don't carry a per-stream
    # bitrate and the field is simply absent — we treat that as
    # "unknown, don't base decisions on it".
    bitrate_kbps: int


def _parse_fps(rate: str) -> float:
    """Parse ffprobe's ``num/den`` rational fps string. Returns 0 on
    parse failure so callers can fall back to a safe default."""
    try:
        if "/" in rate:
            num, den = rate.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        return float(rate)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _probe(source_path: str) -> Optional[_ProbeResult]:
    """Run ``ffprobe`` on ``source_path`` and return codec + shape info.

    Returns ``None`` on any probe failure — the caller treats that as
    "don't touch the source" and serves it as-is, which preserves the
    pre-fix behaviour instead of hard-failing the share endpoint.

    Results are cached by ``(source_path, mtime)`` with a 5-minute TTL
    so the hot HTML5 range-request path doesn't fork ``ffprobe`` per
    seek. A replaced source invalidates the cache via the mtime
    component of the key.
    """
    try:
        mtime = os.path.getmtime(source_path)
    except OSError:
        mtime = 0.0
    cached = _probe_cache_get(source_path, mtime)
    if cached is not None:
        return cached
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-print_format", "json",
                "-show_streams",
                source_path,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0 or not proc.stdout:
            logger.debug(
                "browser_preview: ffprobe failed on %s (rc=%s): %s",
                source_path, proc.returncode, proc.stderr[:200],
            )
            return None
        data = json.loads(proc.stdout)
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        logger.debug("browser_preview: probe error on %s: %s", source_path, e)
        return None

    v_codec = ""
    a_codec = ""
    has_audio = False
    width = 0
    height = 0
    fps = 0.0
    bitrate_kbps = 0
    for stream in data.get("streams", []) or []:
        kind = stream.get("codec_type")
        name = (stream.get("codec_name") or "").lower()
        if kind == "video" and not v_codec:
            v_codec = name
            try:
                width = int(stream.get("width") or 0)
                height = int(stream.get("height") or 0)
            except (TypeError, ValueError):
                width = height = 0
            fps = _parse_fps(
                stream.get("avg_frame_rate")
                or stream.get("r_frame_rate")
                or "0/1"
            )
            try:
                br_raw = stream.get("bit_rate")
                if br_raw:
                    bitrate_kbps = int(int(br_raw) / 1000)
            except (TypeError, ValueError):
                bitrate_kbps = 0
        elif kind == "audio" and not a_codec:
            a_codec = name
            has_audio = True
    result = _ProbeResult(
        video_codec=v_codec,
        audio_codec=a_codec,
        has_audio=has_audio,
        width=width,
        height=height,
        fps=fps,
        bitrate_kbps=bitrate_kbps,
    )
    _probe_cache_put(source_path, mtime, result)
    return result


def _needs_preview(source_path: str, probe: _ProbeResult) -> bool:
    """Return True when the source can't be played smoothly in a browser.

    Triggers on any of:
      * container / codec incompatibility (the original reason this
        module exists — MKV + AC3 etc.)
      * oversize source (resolution above ``_COPY_MAX_WIDTH`` or
        bitrate above ``_COPY_MAX_BITRATE_KBPS``). Even when the
        codec is browser-supported, 4K Blu-ray sources stutter on
        laptop CPUs and every seek pulls MB-per-second chunks that
        don't fit the scrubbing budget — so we build a smaller cap
        preview for smooth playback.

    These thresholds intentionally match ``_should_copy_video``'s
    1080p / 8 Mbps comfort zone: a source the browser plays smoothly
    needs no preview at all (fastest possible path), while anything
    above gets either a remuxed copy preview (also fast) or a
    re-encoded preview at the lower ``_PREVIEW_MAX_WIDTH`` /
    ``_PREVIEW_MAX_BITRATE_KBPS`` target.
    """
    ext = os.path.splitext(source_path)[1].lower()
    if ext not in _BROWSER_CONTAINERS:
        return True
    if probe.video_codec and probe.video_codec not in _BROWSER_VIDEO_CODECS:
        return True
    if probe.has_audio and probe.audio_codec not in _BROWSER_AUDIO_CODECS:
        return True
    if probe.width and probe.width > _COPY_MAX_WIDTH:
        return True
    if probe.bitrate_kbps and probe.bitrate_kbps > _COPY_MAX_BITRATE_KBPS:
        return True
    return False


def _preview_path_for(source_path: str) -> str:
    """Return the canonical browser-preview path for a source video.

    The ``.v3`` tag in the filename bumps whenever the encoding
    pipeline changes in a way that invalidates older cached previews
    (e.g. added GOP tuning, scale cap, bitrate cap, faster preset).
    The previous ``browser_preview.v2.mp4`` files stay on disk but
    are simply not looked up anymore — new previews land at the
    versioned name and playback starts faster on the next request.
    """
    directory = os.path.dirname(source_path) or "."
    return os.path.join(directory, _PREVIEW_FILENAME)


# Cached HW-encoder spec for the preview transcode. Populated lazily on
# the first encode and reused for the rest of the process lifetime —
# capability detection forks ``ffmpeg -encoders`` plus a tiny test
# encode and shouldn't run per request.
_preview_encoder_cache: Optional[dict] = None


def _detect_preview_encoder() -> dict:
    """Pick the fastest available H.264 encoder for browser previews.

    The clip-export path already detects available HW encoders
    (NVENC / QSV / VAAPI / VideoToolbox); we reuse that detection so
    a deployment with a GPU gets a 5-10× faster preview encode for
    free. Falls back to ``libx264 -preset ultrafast`` everywhere else.

    Returns a spec dict consumed by :func:`_build_ffmpeg_cmd`:
      * ``encoder``    — ``"h264_nvenc"`` / ``"h264_qsv"`` /
                          ``"h264_videotoolbox"`` / ``"libx264"``
      * ``rate_args``  — encoder-specific quality / rate-control args
      * ``preset``     — encoder-specific speed preset (None → omit)

    On any detection error we silently fall back to libx264 — the old
    behaviour. This keeps the path working on minimal deployments
    that have ffmpeg but no GPU runtime.
    """
    global _preview_encoder_cache
    if _preview_encoder_cache is not None:
        return _preview_encoder_cache

    # Default: software libx264 at ultrafast preset. ``ultrafast`` is
    # ~2× quicker than ``veryfast`` on the same CPU, with quality
    # loss that's invisible in a preview-sized viewport. ``-threads 0``
    # tells libx264 to use every CPU core for the encode (default
    # behaviour, but spelled out here so it can't be lost to a future
    # global ffmpeg flag).
    spec: dict = {
        "encoder": "libx264",
        "preset": "ultrafast",
        "rate_args": [
            "-crf", "26",
            "-maxrate", f"{_PREVIEW_MAX_BITRATE_KBPS}k",
            "-bufsize", f"{_PREVIEW_MAX_BITRATE_KBPS * 2}k",
            "-threads", "0",
        ],
    }

    try:
        from backend.services.clip_exporter import detect_gpu_capabilities
        gpu = detect_gpu_capabilities()
        encoder = (gpu or {}).get("encoder") or "libx264"
        if encoder == "h264_nvenc":
            # NVENC ``p1`` is the fastest preset (lowest latency, lowest
            # quality). For a 720p preview this is more than enough,
            # and it routinely encodes at 100-300 fps even on an
            # entry-level GPU — well inside the 15-30 s budget for
            # any reasonable source.
            spec = {
                "encoder": "h264_nvenc",
                "preset": "p1",
                "rate_args": [
                    "-tune", "ll",
                    "-rc", "vbr",
                    "-cq", "26",
                    "-b:v", f"{_PREVIEW_MAX_BITRATE_KBPS}k",
                    "-maxrate", f"{_PREVIEW_MAX_BITRATE_KBPS}k",
                    "-bufsize", f"{_PREVIEW_MAX_BITRATE_KBPS * 2}k",
                ],
            }
        elif encoder == "h264_qsv":
            # Intel QuickSync — comparable to NVENC for our use case.
            spec = {
                "encoder": "h264_qsv",
                "preset": "veryfast",
                "rate_args": [
                    "-global_quality", "26",
                    "-maxrate", f"{_PREVIEW_MAX_BITRATE_KBPS}k",
                    "-bufsize", f"{_PREVIEW_MAX_BITRATE_KBPS * 2}k",
                ],
            }
        elif encoder == "h264_videotoolbox":
            # Apple VideoToolbox doesn't take ``-preset``; ``-q:v`` on
            # a 0-100 scale picks the rate-control quality. 55 is
            # roughly equivalent to libx264 CRF 26.
            spec = {
                "encoder": "h264_videotoolbox",
                "preset": None,
                "rate_args": [
                    "-q:v", "55",
                    "-maxrate", f"{_PREVIEW_MAX_BITRATE_KBPS}k",
                    "-bufsize", f"{_PREVIEW_MAX_BITRATE_KBPS * 2}k",
                ],
            }
        # h264_vaapi is intentionally not wired up here — it requires
        # ``-vaapi_device`` before the input plus a hwupload filter
        # chain, which would tangle with the existing scale filter.
        # libx264 ultrafast is fast enough as a fallback.
        if spec["encoder"] != "libx264":
            logger.info(
                "browser_preview: using HW encoder %s (preset=%s) for previews",
                spec["encoder"], spec.get("preset"),
            )
    except Exception as e:  # pragma: no cover — detect path is best-effort
        logger.debug(
            "browser_preview: HW encoder detect failed, using libx264: %s", e,
        )

    _preview_encoder_cache = spec
    return spec


def _should_copy_video(probe: _ProbeResult) -> bool:
    """Return True when it is safe to stream-copy the source video.

    We copy when ALL of the following hold:
      * Codec is H.264 (browser plays it without a transcode).
      * Resolution is at or below 1080p (above that, even H.264
        decode stresses laptop CPUs and every seek pulls megabytes).
      * Reported bitrate is sane (≤ 8 Mbps). Anything higher implies
        a rip-grade bitstream that scrubs poorly in the browser even
        when it's technically H.264.

    The thresholds are deliberately *higher* than the re-encode cap
    (``_PREVIEW_MAX_WIDTH``, ``_PREVIEW_MAX_BITRATE_KBPS``): copy is
    a remux that finishes in single-digit seconds even for a feature-
    length film, so we always prefer it when the source is already
    inside the browser's comfort zone. Re-encoding kicks in only for
    sources that *can't* play in-browser without it.
    """
    if probe.video_codec != "h264":
        return False
    if probe.width and probe.width > _COPY_MAX_WIDTH:
        return False
    if probe.bitrate_kbps and probe.bitrate_kbps > _COPY_MAX_BITRATE_KBPS:
        return False
    return True


def _tmp_path_for(target_path: str) -> str:
    """Scratch path FFmpeg writes to before the atomic rename.

    FFmpeg writes bytes directly to its output argument as it
    encodes — the file exists on disk (with a fresh mtime) during
    the entire encode, not just at the end. Without writing through
    a temp name, a concurrent request hitting :func:`ensure_browser_preview_status`
    passes the ``isfile(target) and mtime(target) >= mtime(source)``
    fast-path check mid-encode and ``serve_file`` streams partial
    bytes to the browser.

    Using ``target + ".tmp"`` keeps the scratch file in the same
    directory so the final ``os.rename`` is a same-filesystem
    rename (atomic on POSIX) and not a cross-device copy.
    """
    return target_path + ".tmp"


def _build_ffmpeg_cmd(source_path: str, target_path: str, probe: _ProbeResult) -> list[str]:
    """Build the FFmpeg command that produces the browser preview.

    Strategy:
      * Video: stream-copy when the source is already H.264 **and**
        within the preview size/bitrate budget. Otherwise re-encode
        with the fastest available encoder (NVENC / QSV /
        VideoToolbox / libx264 ultrafast) at the preview profile —
        needed for H.265, 4K rips, or very-high-bitrate sources that
        stutter in-browser.
      * Re-encode path is tuned for **encode speed**, not file size:
        ultrafast preset, 720p scale cap, capped bitrate, fully
        threaded. These together let a multi-minute Blu-ray rip
        finish encoding inside the player's 15-30 s budget on
        ordinary CPUs (and well under that with a GPU).
      * Audio: always transcode to AAC at 160 kbps stereo. The
        original track might be AC3, DTS, FLAC, etc. Transcoding
        costs almost nothing on top of the audio duration.
      * ``+faststart`` moves the moov atom up front so the
        ``<video>`` element can start playback before the whole
        file is buffered.
      * Output is directed at :func:`_tmp_path_for` (``<target>.tmp``)
        so the final ``target_path`` only appears once the encode
        is complete and has been atomically renamed into place. This
        closes the partial-file race where a concurrent request
        could read a half-written MP4.
    """
    video_opts: list[str]
    video_filters: list[str] = []
    if _should_copy_video(probe):
        video_opts = ["-c:v", "copy"]
    else:
        # Re-encode at the preview profile.
        #
        # * ``-g`` / ``-keyint_min`` = ``fps * 2`` — a keyframe every
        #   two seconds of real video. libx264's default GOP is 250
        #   frames and it effectively disables scene-cut keyframes
        #   when we also pass ``-sc_threshold 0``, so without this
        #   the user's seek lands on a long inter frame and the
        #   decoder has to walk backwards several seconds before it
        #   can paint a picture.
        # * ``scale='min(w,iw)':-2`` caps the width at 720p-ish
        #   while preserving aspect ratio; ``-2`` keeps height even
        #   (yuv420p requires that).
        # * The actual encoder + rate-control args come from
        #   :func:`_detect_preview_encoder` so a deployment with a
        #   GPU gets NVENC / QSV / VideoToolbox automatically. The
        #   software fallback is libx264 at ``ultrafast``, which is
        #   ~2× faster than the previous ``veryfast`` setting.
        fps_for_keyint = probe.fps if probe.fps and probe.fps > 1 else 30.0
        gop = max(24, int(round(fps_for_keyint * _PREVIEW_KEYFRAME_INTERVAL_SEC)))
        video_filters.append(
            f"scale='min({_PREVIEW_MAX_WIDTH},iw)':-2"
        )
        encoder_spec = _detect_preview_encoder()
        video_opts = ["-c:v", encoder_spec["encoder"]]
        if encoder_spec.get("preset"):
            video_opts += ["-preset", encoder_spec["preset"]]
        video_opts += [
            "-profile:v", "main",
            "-level", "4.0",
            *encoder_spec.get("rate_args", []),
            "-g", str(gop),
            "-keyint_min", str(gop),
            "-sc_threshold", "0",
            "-pix_fmt", "yuv420p",
        ]

    audio_opts: list[str]
    if probe.has_audio:
        # 160 kbps AAC is transparent for stereo content and saves
        # encoder cycles vs the old 192 kbps target. Audio cost is
        # never the bottleneck but every saved cycle compounds when
        # the goal is sub-30s end-to-end.
        audio_opts = [
            "-c:a", "aac",
            "-b:a", "160k",
            "-ac", "2",
        ]
    else:
        audio_opts = ["-an"]

    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", source_path,
        "-map", "0:v:0?",
        *(["-map", "0:a:0?"] if probe.has_audio else []),
        *video_opts,
    ]
    if video_filters:
        cmd += ["-vf", ",".join(video_filters)]
    cmd += [
        *audio_opts,
        # ``+faststart`` for progressive download; ``+frag_keyframe``
        # is deliberately *not* added — fragmented MP4 plays back
        # fine but some older Safari builds scrub poorly on it.
        "-movflags", "+faststart",
        # Force MP4 container explicitly so FFmpeg never has to infer
        # it from the temp filename's extension. Without this, a
        # ``.mp4.tmp`` suffix produces ``Unable to choose an output
        # format for '<...>.mp4.tmp'; use a standard extension for
        # the filename or specify the format manually`` and every
        # preview build fails identically until the cache name is
        # bumped on the next deploy.
        "-f", "mp4",
        # Write through a scratch path — _run_ffmpeg atomically
        # renames onto target_path only after a successful exit.
        # See _tmp_path_for for the partial-file race this closes.
        _tmp_path_for(target_path),
    ]
    return cmd


def _lock_path_for(target_path: str) -> str:
    return target_path + ".lock"


def _is_stale_lock(lock_path: str, max_age_sec: int = 3600) -> bool:
    """Locks older than ``max_age_sec`` are assumed abandoned by a
    crashed generator and get cleared. Without this, a single killed
    FFmpeg would wedge preview generation forever."""
    try:
        mtime = os.path.getmtime(lock_path)
    except OSError:
        return False
    return (time.time() - mtime) > max_age_sec


def _acquire_lock(lock_path: str) -> bool:
    """Atomically claim the preview-generation lock.

    Uses ``O_CREAT | O_EXCL`` so two concurrent generators can't both
    succeed. Returns True on acquisition, False if another process
    already holds it (and the lock isn't stale).
    """
    if os.path.exists(lock_path) and _is_stale_lock(lock_path):
        try:
            os.remove(lock_path)
        except OSError:
            pass
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    except OSError as e:
        logger.warning("browser_preview: lock acquire failed: %s", e)
        return False
    try:
        os.write(fd, f"pid={os.getpid()} ts={int(time.time())}\n".encode())
    finally:
        os.close(fd)
    return True


def _release_lock(lock_path: str) -> None:
    try:
        os.remove(lock_path)
    except OSError:
        pass


def _wait_for_peer(target_path: str, lock_path: str, timeout_sec: float = 300.0) -> bool:
    """Block until a peer generator finishes (or the timeout fires).

    Polling is fine here — preview generation is rare and the polling
    interval is coarse enough not to burn CPU. Returns True when the
    preview file eventually appears.
    """
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if os.path.isfile(target_path) and os.path.getsize(target_path) > 0:
            return True
        if not os.path.exists(lock_path):
            # Peer released the lock without producing output — give
            # up so the caller can fall back to the source.
            return os.path.isfile(target_path)
        time.sleep(0.5)
    return False


def _run_ffmpeg(cmd: list[str], target_path: str) -> bool:
    """Run FFmpeg and return True on a non-empty successful output.

    FFmpeg writes to :func:`_tmp_path_for` (``target_path + ".tmp"``)
    and we atomically ``os.rename`` into ``target_path`` on success.
    That way the final path only ever exists as a complete, playable
    file — closing the partial-file race where concurrent HTTP
    requests could hit the ``isfile(target_path)`` fast path mid-encode
    and stream half-written MP4 bytes to the browser.
    """
    tmp_path = _tmp_path_for(target_path)
    # Clear any stale scratch file from a previous crashed encode so
    # ffmpeg's ``-y`` doesn't spend time truncating a multi-GB leftover.
    try:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
    except OSError:
        pass
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30-minute ceiling — generous for a full re-encode
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("browser_preview: ffmpeg launch failed: %s", e)
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return False
    if proc.returncode != 0:
        logger.warning(
            "browser_preview: ffmpeg exit %s: %s",
            proc.returncode, (proc.stderr or "")[:500],
        )
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return False
    if not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) == 0:
        logger.warning("browser_preview: ffmpeg produced empty output at %s", tmp_path)
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return False
    # Atomic swap — same filesystem so os.rename is a single
    # directory-entry update, never observable as a partial file.
    try:
        os.replace(tmp_path, target_path)
    except OSError as e:
        logger.warning(
            "browser_preview: rename %s -> %s failed: %s",
            tmp_path, target_path, e,
        )
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return False
    return True


def ensure_browser_preview_status(
    source_path: str, *, wait_for_peer: bool = True,
) -> tuple[str, bool]:
    """Same as :func:`ensure_browser_preview` but also reports whether
    the returned path is the final playable artifact.

    Returns ``(path, ready)``:
      * ``ready=True`` — ``path`` is safe to hand to ``<video>``: either
        the finished preview, or a source that's already browser-
        compatible (``_needs_preview`` returned False), or an unprobable
        source we can't do better with.
      * ``ready=False`` — ``path`` is the source file but the source is
        known to be unplayable in-browser (needs a preview) and the
        preview isn't available yet. Callers in HTTP handlers should
        signal the client to retry rather than serve unplayable bytes.
    """
    if not source_path or not os.path.isfile(source_path):
        return source_path, True

    target_path = _preview_path_for(source_path)

    # Fast path: cached preview already exists and is newer than the
    # source. Use ``os.path.getmtime`` rather than size because the
    # source can be larger than a copy-mode preview (container overhead).
    if os.path.isfile(target_path):
        try:
            if os.path.getmtime(target_path) >= os.path.getmtime(source_path):
                return target_path, True
            # Source was replaced — invalidate.
            logger.info(
                "browser_preview: source newer than cached preview, rebuilding %s",
                target_path,
            )
            try:
                os.remove(target_path)
            except OSError:
                pass
        except OSError:
            return target_path, True

    probe = _probe(source_path)
    if probe is None:
        # Can't probe — fall back to the source so the player at least
        # attempts playback. This keeps the endpoint working on
        # deployments without ffprobe available.
        return source_path, True
    if not _needs_preview(source_path, probe):
        return source_path, True

    lock_path = _lock_path_for(target_path)
    if not _acquire_lock(lock_path):
        # Another worker is already generating. For HTTP handlers we
        # don't want to stall the response for minutes, so we bail
        # out and signal ``ready=False``; the handler will translate
        # that into a 503 + Retry-After, and the browser will pick
        # up the finished preview on a later request. The background
        # warm-up path keeps the old wait semantics so it doesn't
        # race itself.
        if not wait_for_peer:
            logger.info(
                "browser_preview: peer generating %s — signalling not-ready",
                target_path,
            )
            return source_path, False
        logger.info(
            "browser_preview: peer is generating %s, waiting", target_path,
        )
        if _wait_for_peer(target_path, lock_path):
            return target_path, True
        # Peer failed or timed out — fall back to source rather than
        # blocking the recipient forever.
        return source_path, False

    try:
        cmd = _build_ffmpeg_cmd(source_path, target_path, probe)
        logger.info(
            "browser_preview: building %s from %s (video=%s, audio=%s)",
            target_path, source_path, probe.video_codec, probe.audio_codec,
        )
        t0 = time.time()
        ok = _run_ffmpeg(cmd, target_path)
        if not ok and _should_retry_with_software(cmd):
            # HW encoder failed on this specific input (driver hiccup,
            # GPU OOM, exotic pixel format the encoder rejects). Burn
            # the cached HW spec so subsequent previews use libx264
            # immediately, then re-run *this* encode in software so the
            # user gets a playable preview instead of the silent-audio
            # source fallback. This is the difference between "the
            # preview always loads" and "the preview loads except when
            # NVENC has a bad day".
            logger.warning(
                "browser_preview: HW encoder failed on %s, retrying with libx264",
                source_path,
            )
            global _preview_encoder_cache
            _preview_encoder_cache = {
                "encoder": "libx264",
                "preset": "ultrafast",
                "rate_args": [
                    "-crf", "26",
                    "-maxrate", f"{_PREVIEW_MAX_BITRATE_KBPS}k",
                    "-bufsize", f"{_PREVIEW_MAX_BITRATE_KBPS * 2}k",
                    "-threads", "0",
                ],
            }
            cmd = _build_ffmpeg_cmd(source_path, target_path, probe)
            ok = _run_ffmpeg(cmd, target_path)
        if not ok:
            return source_path, False
        logger.info(
            "browser_preview: built %s in %.1fs", target_path, time.time() - t0,
        )
        return target_path, True
    finally:
        _release_lock(lock_path)


def _should_retry_with_software(cmd: list[str]) -> bool:
    """Return True if ``cmd`` used a hardware encoder we can fall back from.

    The fallback only fires when the original encode actually used a HW
    encoder — re-running an already-libx264 encode would hit the same
    failure twice. Inspecting the command (rather than threading a flag
    through the call stack) keeps the retry branch a one-line check.
    """
    try:
        idx = cmd.index("-c:v")
    except ValueError:
        return False
    encoder = cmd[idx + 1] if idx + 1 < len(cmd) else ""
    return encoder in {"h264_nvenc", "h264_qsv", "h264_videotoolbox", "h264_vaapi"}


def ensure_browser_preview(source_path: str, *, wait_for_peer: bool = True) -> str:
    """Return a path the browser can actually play.

    Thin wrapper around :func:`ensure_browser_preview_status` for
    callers that don't need the readiness flag.
    """
    path, _ready = ensure_browser_preview_status(
        source_path, wait_for_peer=wait_for_peer,
    )
    return path


async def ensure_browser_preview_status_async(
    source_path: str, *, wait_for_peer: bool = False,
) -> tuple[str, bool]:
    """Async variant of :func:`ensure_browser_preview_status` — runs
    the blocking work on the default executor so the event loop isn't
    stalled while FFmpeg churns.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: ensure_browser_preview_status(
            source_path, wait_for_peer=wait_for_peer,
        ),
    )


async def ensure_browser_preview_async(
    source_path: str, *, wait_for_peer: bool = False,
) -> str:
    """Async shim for FastAPI handlers — runs the blocking work on
    the default executor so the event loop isn't stalled while
    FFmpeg churns.

    Defaults to ``wait_for_peer=False`` because the two callers
    inside request handlers (``/api/files`` and the share route)
    must not block for minutes when the background warm-up is still
    generating the preview. The ingest warm-up explicitly passes
    ``wait_for_peer=True`` so it does the work instead of bailing.
    """
    path, _ready = await ensure_browser_preview_status_async(
        source_path, wait_for_peer=wait_for_peer,
    )
    return path
