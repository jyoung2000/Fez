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


# Preview-encoding targets. Chosen for smooth scrubbing on a typical
# laptop over a typical home connection:
#   * 1080p is more than enough for an in-browser preview even on a
#     4K monitor; decoding 4K H.264 in software stalls low-end CPUs
#     and slows every seek because the browser has to decode more
#     data to land on a keyframe.
#   * 6 Mbps is within the progressive-download budget of ordinary
#     connections and cheap for a CPU decoder. Blu-ray rips often
#     sit at 15-40 Mbps, which is far above that budget.
_PREVIEW_MAX_WIDTH = 1920
_PREVIEW_MAX_BITRATE_KBPS = 6000
# Short GOP → fine-grained seek. 2 seconds lets the browser land
# within one keyframe of any scrub target instead of jumping 4–10 s
# at a time the way factory-default libx264 does.
_PREVIEW_KEYFRAME_INTERVAL_SEC = 2.0


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
    """
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
    return _ProbeResult(
        video_codec=v_codec,
        audio_codec=a_codec,
        has_audio=has_audio,
        width=width,
        height=height,
        fps=fps,
        bitrate_kbps=bitrate_kbps,
    )


def _needs_preview(source_path: str, probe: _ProbeResult) -> bool:
    """Return True when the source can't be played smoothly in a browser.

    Triggers on any of:
      * container / codec incompatibility (the original reason this
        module exists — MKV + AC3 etc.)
      * oversize source (resolution above ``_PREVIEW_MAX_WIDTH`` or
        bitrate above ``_PREVIEW_MAX_BITRATE_KBPS``). Even when the
        codec is browser-supported, 4K Blu-ray sources stutter on
        laptop CPUs and every seek pulls MB-per-second chunks that
        don't fit the scrubbing budget — so we build a 1080p cap
        preview for smooth playback.
    """
    ext = os.path.splitext(source_path)[1].lower()
    if ext not in _BROWSER_CONTAINERS:
        return True
    if probe.video_codec and probe.video_codec not in _BROWSER_VIDEO_CODECS:
        return True
    if probe.has_audio and probe.audio_codec not in _BROWSER_AUDIO_CODECS:
        return True
    if probe.width and probe.width > _PREVIEW_MAX_WIDTH:
        return True
    if probe.bitrate_kbps and probe.bitrate_kbps > _PREVIEW_MAX_BITRATE_KBPS * 1.2:
        return True
    return False


def _preview_path_for(source_path: str) -> str:
    """Return the canonical browser-preview path for a source video.

    The ``.v2`` tag in the filename bumps whenever the encoding
    pipeline changes in a way that invalidates older cached previews
    (e.g. added GOP tuning, scale cap, bitrate cap). The previous
    ``browser_preview.mp4`` files stay on disk but are simply not
    looked up anymore — new previews land at the versioned name and
    playback smoothness improves on the next request.
    """
    directory = os.path.dirname(source_path) or "."
    return os.path.join(directory, "browser_preview.v2.mp4")


def _should_copy_video(probe: _ProbeResult) -> bool:
    """Return True when it is safe to stream-copy the source video.

    We only copy when ALL of the following hold:
      * Codec is H.264 (browser plays it without a transcode).
      * Resolution is at or below the preview cap (otherwise the
        player has to decode 4K frames on every seek → stutter).
      * Reported bitrate is sane (≤ 1.2× the preview target). Anything
        higher implies a rip-grade bitstream that scrubs poorly in
        the browser even when it's technically H.264.

    If any of those fail we fall through to a re-encode at the
    preview profile so seeks feel responsive. Copy is still the
    preferred path — it's seconds, not minutes — so we leave it in
    place for the common case of already-web-friendly source MP4s.
    """
    if probe.video_codec != "h264":
        return False
    if probe.width and probe.width > _PREVIEW_MAX_WIDTH:
        return False
    if probe.bitrate_kbps and probe.bitrate_kbps > _PREVIEW_MAX_BITRATE_KBPS * 1.2:
        return False
    return True


def _build_ffmpeg_cmd(source_path: str, target_path: str, probe: _ProbeResult) -> list[str]:
    """Build the FFmpeg command that produces the browser preview.

    Strategy:
      * Video: stream-copy when the source is already H.264 **and**
        within the preview size/bitrate budget. Otherwise re-encode
        with libx264 at the preview profile — needed for H.265, 4K
        rips, or very-high-bitrate sources that stutter in-browser.
      * Re-encode path is tuned for smooth **seeking**, not file
        size: short GOP (every ~2 s of video), a 1080p scale cap,
        and a capped bitrate. These three together let the browser
        land within one keyframe of any scrub target and finish the
        Range fetch quickly.
      * Audio: always transcode to AAC at 192 kbps stereo. The
        original track might be AC3, DTS, FLAC, etc. Transcoding
        costs almost nothing on top of the audio duration.
      * ``+faststart`` moves the moov atom up front so the
        ``<video>`` element can start playback before the whole
        file is buffered.
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
        # * ``scale='min(w,iw)':-2`` caps the width at 1080p-ish
        #   while preserving aspect ratio; ``-2`` keeps height even
        #   (yuv420p requires that).
        # * ``-maxrate`` / ``-bufsize`` clamp the bitrate so 4K
        #   sources don't emit 40 Mbps previews that defeat the
        #   point of a preview.
        # * ``-tune fastdecode`` reduces CPU cost on the client
        #   decoder (disables CABAC and a few decoder-unfriendly
        #   tools). Visible quality loss is imperceptible at CRF 23
        #   and the smoothness win on low-end laptops is real.
        # * ``-profile:v main`` + ``-level 4.0`` keep the file
        #   within the baseline every current browser accepts and
        #   avoids triggering a hardware-decode compatibility mode
        #   on some Safari/iOS builds.
        fps_for_keyint = probe.fps if probe.fps and probe.fps > 1 else 30.0
        gop = max(24, int(round(fps_for_keyint * _PREVIEW_KEYFRAME_INTERVAL_SEC)))
        video_filters.append(
            f"scale='min({_PREVIEW_MAX_WIDTH},iw)':-2"
        )
        video_opts = [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "fastdecode",
            "-profile:v", "main",
            "-level", "4.0",
            "-crf", "23",
            "-maxrate", f"{_PREVIEW_MAX_BITRATE_KBPS}k",
            "-bufsize", f"{_PREVIEW_MAX_BITRATE_KBPS * 2}k",
            "-g", str(gop),
            "-keyint_min", str(gop),
            "-sc_threshold", "0",
            "-pix_fmt", "yuv420p",
        ]

    audio_opts: list[str]
    if probe.has_audio:
        audio_opts = [
            "-c:a", "aac",
            "-b:a", "192k",
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
        target_path,
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
    """Run FFmpeg and return True on a non-empty successful output."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30-minute ceiling — generous for a full re-encode
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("browser_preview: ffmpeg launch failed: %s", e)
        return False
    if proc.returncode != 0:
        logger.warning(
            "browser_preview: ffmpeg exit %s: %s",
            proc.returncode, (proc.stderr or "")[:500],
        )
        try:
            if os.path.isfile(target_path):
                os.remove(target_path)
        except OSError:
            pass
        return False
    if not os.path.isfile(target_path) or os.path.getsize(target_path) == 0:
        logger.warning("browser_preview: ffmpeg produced empty output at %s", target_path)
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
        if not ok:
            return source_path, False
        logger.info(
            "browser_preview: built %s in %.1fs", target_path, time.time() - t0,
        )
        return target_path, True
    finally:
        _release_lock(lock_path)


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
