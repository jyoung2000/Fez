import asyncio
import base64
import json
import logging
import os
import shutil
from typing import Callable, Optional

from PIL import Image

from backend.config import settings
from backend.models import FrameData

logger = logging.getLogger(__name__)

MAX_DIMENSION = 1568


def _extract_ffmpeg_error(stderr_bytes: bytes) -> str:
    """Extract the useful error message from ffmpeg stderr.

    ffmpeg prints a ~400 char version banner to stderr on EVERY run (even
    successful ones). The actual error is at the END. Previous code used
    stderr[:500] which captured only the banner and missed the real error.
    """
    text = stderr_bytes.decode(errors="replace") if stderr_bytes else ""

    # Find specific error lines
    error_lines = [
        line.strip() for line in text.split("\n")
        if line.strip() and any(kw in line.lower() for kw in [
            "error", "no space", "permission denied", "no such file",
            "invalid", "corrupt", "failed", "cannot", "not found",
            "out of memory", "killed", "broken pipe", "disk full",
        ])
    ]
    if error_lines:
        return "\n".join(error_lines[-5:])

    # Fallback: return the tail (skips the version banner)
    return text[-1500:].strip() if len(text) > 1500 else text.strip()


def _check_disk_space(output_path: str, required_mb: int = 500) -> None:
    """Check that enough disk space is available before extraction."""
    output_dir = os.path.dirname(output_path) or "."
    try:
        disk = shutil.disk_usage(output_dir)
        free_mb = disk.free / (1024 * 1024)
        if free_mb < required_mb:
            raise RuntimeError(
                f"Insufficient disk space: {free_mb:.0f}MB free, need {required_mb}MB. "
                f"Disk is {disk.used / disk.total * 100:.0f}% full. "
                f"Clear old job files from /data/uploads/ or increase disk size."
            )
        elif free_mb < required_mb * 2:
            logger.warning(
                "Low disk space: %.0fMB free (%.0f%% full). "
                "Extraction may fail for long videos.",
                free_mb, disk.used / disk.total * 100,
            )
    except RuntimeError:
        raise
    except OSError as e:
        logger.warning("Could not check disk space: %s", e)

# After this many seconds with 0 frames produced, kill FFmpeg and retry
# with a fallback strategy (no GPU / no scene detection).
_STALL_TIMEOUT = 30   # Kill extraction if 0 frames after 30s (was 45)


async def _run_subprocess_cancellable(
    cmd: list[str],
    cancel_check: Optional[Callable] = None,
    poll_interval: float = 1.0,
) -> tuple[int, bytes]:
    """Run a subprocess with periodic cancellation checks.

    If cancel_check raises, the subprocess is terminated/killed and the
    exception propagates immediately instead of waiting for completion.
    Returns (returncode, stderr_bytes).
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    if not cancel_check:
        _, stderr = await proc.communicate()
        return proc.returncode, stderr

    comm_task = asyncio.ensure_future(proc.communicate())
    try:
        while not comm_task.done():
            await asyncio.sleep(poll_interval)
            if not comm_task.done():
                cancel_check()  # raises CancelledError if cancelled
        _, stderr = comm_task.result()
        return proc.returncode, stderr
    except BaseException:
        # Cancel requested or other error — kill the subprocess
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            proc.kill()
        comm_task.cancel()
        raise


async def get_video_metadata(video_path: str) -> dict:
    """Extract video metadata using FFprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", video_path,
    ]
    logger.info("FFprobe command: %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(
            "FFprobe timed out after 2 minutes — the video file may be corrupt or on slow storage."
        )
    if proc.returncode != 0:
        err_msg = stderr.decode().strip() or "(no output)"
        logger.error(f"FFprobe failed (rc={proc.returncode}) for {video_path}: {err_msg}")
        # Provide a user-friendly message for common corruption patterns
        if "Invalid data found" in err_msg or "EBML header" in err_msg:
            raise RuntimeError(
                "The video file could not be read — it may be corrupt or an incomplete download. "
                "Please check that it plays correctly on your device and try again."
            )
        raise RuntimeError(f"FFprobe failed: {err_msg}")

    data = json.loads(stdout.decode())
    fmt = data.get("format", {})
    video_stream = None
    for s in data.get("streams", []):
        if s.get("codec_type") == "video":
            video_stream = s
            break

    duration = float(fmt.get("duration", 0))
    file_size_mb = round(int(fmt.get("size", 0)) / (1024 * 1024), 2)
    resolution = ""
    fps = 0.0
    if video_stream:
        w = video_stream.get("width", 0)
        h = video_stream.get("height", 0)

        # Account for non-square pixels (SAR != 1:1).  Many cameras and
        # encoding tools produce videos where coded dimensions differ from
        # the actual display dimensions.  FFprobe reports SAR as "N:M";
        # the display width = coded_width * (SAR_num / SAR_den).
        sar_str = video_stream.get("sample_aspect_ratio", "1:1")
        try:
            sar_parts = sar_str.split(":")
            sar_num = int(sar_parts[0])
            sar_den = int(sar_parts[1]) if len(sar_parts) > 1 else 1
            if sar_num > 0 and sar_den > 0 and sar_num != sar_den:
                display_w = round(w * sar_num / sar_den)
                # Ensure even dimensions for video encoding
                display_w = display_w - (display_w % 2)
                logger.info(
                    "SAR correction: coded=%dx%d, SAR=%s, display=%dx%d",
                    w, h, sar_str, display_w, h,
                )
                w = display_w
        except (ValueError, IndexError, ZeroDivisionError):
            pass  # Malformed SAR — keep coded dimensions

        resolution = f"{w}x{h}"
        r_frame_rate = video_stream.get("r_frame_rate", "0/1")
        try:
            num, den = r_frame_rate.split("/")
            fps = round(int(num) / max(int(den), 1), 2)
        except (ValueError, ZeroDivisionError):
            fps = 0.0

    # Extract codec info for GPU compatibility checks
    codec_name = ""
    pix_fmt = ""
    if video_stream:
        codec_name = video_stream.get("codec_name", "")
        pix_fmt = video_stream.get("pix_fmt", "")

    # Expose ``width`` / ``height`` as first-class fields (in addition
    # to the legacy ``resolution`` string) so the reframer doesn't fall
    # back to a hard-coded 1920x1080 default for non-FullHD sources.
    # ``width``/``height`` are the SAR-corrected display dimensions so
    # crop/zoom math operates in display-pixel space.
    width_px = 0
    height_px = 0
    if video_stream:
        width_px = int(w) if w else 0
        height_px = int(h) if h else 0

    return {
        "duration": duration,
        "resolution": resolution,
        "width": width_px,
        "height": height_px,
        "fps": fps,
        "file_size_mb": file_size_mb,
        "codec_name": codec_name,
        "pix_fmt": pix_fmt,
    }


def _build_scene_filter(rate: int) -> str:
    """Build the hybrid scene detection + interval filter string.

    Adaptive scene threshold: for long videos with high sample rates,
    raise the threshold to avoid scene detection overwhelming the interval cap.
    Standard rate=10 uses threshold 0.3.
    rate=30+ uses threshold 0.45 (only major scene changes).
    """
    if rate >= 30:
        threshold = 0.45  # Only major scene changes for long videos
    elif rate >= 20:
        threshold = 0.38
    else:
        threshold = 0.3  # Default for short/medium videos

    return (
        f"select='gt(scene\\,{threshold})+isnan(prev_selected_t)"
        f"+gte(t-prev_selected_t\\,{rate})',"
        f"scale='min(1024\\,iw)':'min(576\\,ih)':force_original_aspect_ratio=decrease,"
        f"format=pix_fmts=yuvj420p"
    )


def _build_interval_filter(rate: int) -> str:
    """Build a simple interval-only filter (no scene detection)."""
    return (
        f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{rate})',"
        f"scale='min(1024\\,iw)':'min(576\\,ih)':force_original_aspect_ratio=decrease,"
        f"format=pix_fmts=yuvj420p"
    )


def _get_gpu_decode_args() -> list[str]:
    """Get GPU hardware decode args, logging any failures."""
    try:
        from backend.services.clip_exporter import _gpu_decode_args_for_filter
        args = _gpu_decode_args_for_filter()
        if args:
            logger.info("GPU decode args for frame extraction: %s", " ".join(args))
        return args
    except Exception as e:
        logger.warning("GPU decode args failed (falling back to CPU): %s", e)
        return []


async def _bulk_probe_pts(jpg_dir: str, count: int) -> list[float | None]:
    """Run one ffmpeg pass over the extracted JPGs and parse pts_time
    for every frame from the showinfo filter's stderr output.

    Returns a list of length ``count``, with ``None`` for any frame
    whose pts couldn't be parsed (caller falls back to the rough
    timestamp already on the FrameData).

    This replaces the previous per-frame ffprobe-spawn loop — each
    spawn cost ~30-80ms of subprocess overhead, so collapsing to a
    single ffmpeg pass recovers ~5-15s on every job with >50 frames.
    """
    if count == 0:
        return []
    # Globbing the jpg sequence is faster than feeding individual files
    # because ffmpeg only opens one demuxer.
    cmd = [
        "ffmpeg", "-y",
        "-pattern_type", "glob",
        "-i", os.path.join(jpg_dir, "frame_*.jpg"),
        "-vf", "showinfo",
        "-f", "null", "-",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning("Bulk PTS probe failed (%s); falling back to per-frame", e)
        return [None] * count

    text = stderr.decode(errors="replace")
    # showinfo lines look like:
    # [Parsed_showinfo_0 @ 0x...] n: 0   pts: 0       pts_time:0
    import re
    pts_values: list[float | None] = []
    for line in text.splitlines():
        m = re.search(r"pts_time:([\d.]+)", line)
        if m:
            try:
                pts_values.append(float(m.group(1)))
            except ValueError:
                pts_values.append(None)
    # Pad / truncate to match expected count.
    if len(pts_values) < count:
        pts_values.extend([None] * (count - len(pts_values)))
    return pts_values[:count]


async def _run_ffmpeg_extraction(
    video_path: str,
    output_dir: str,
    vf_filter: str,
    hw_dec: list[str],
    cancel_check: Optional[Callable],
    progress_callback: Optional[Callable],
    label: str,
) -> tuple[int, bytes]:
    """Run a single FFmpeg extraction attempt with stall detection.

    Returns (returncode, stderr).  Kills the process early if no frames
    appear within _STALL_TIMEOUT seconds.
    """
    cmd = [
        "ffmpeg", "-y",
        "-threads", "0",
        *hw_dec,
        "-i", video_path,
        "-an",
        "-vf", vf_filter,
        "-vsync", "vfr",
        "-q:v", "12",
        "-frame_pts", "1",
        os.path.join(output_dir, "frame_%06d.jpg"),
    ]

    logger.info("FFmpeg frame extraction (%s): %s", label, " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    comm_task = asyncio.ensure_future(proc.communicate())

    poll_count = 0
    stall_polls = 0  # consecutive polls with 0 frames
    try:
        while not comm_task.done():
            await asyncio.sleep(1.0)
            poll_count += 1
            if cancel_check and not comm_task.done():
                cancel_check()
            # Check frame count every 3 seconds
            if poll_count % 3 == 0:
                try:
                    frames_so_far = len([
                        f for f in os.listdir(output_dir)
                        if f.startswith("frame_") and f.endswith(".jpg")
                    ])
                    if progress_callback:
                        await progress_callback(frames_so_far)
                    # Stall detection: if 0 frames for too long, abort
                    if frames_so_far == 0:
                        stall_polls += 1
                        elapsed_stall = stall_polls * 3
                        if elapsed_stall >= _STALL_TIMEOUT:
                            logger.warning(
                                "FFmpeg stall detected (%s): 0 frames after %ds — aborting",
                                label, elapsed_stall,
                            )
                            proc.terminate()
                            try:
                                await asyncio.wait_for(proc.wait(), timeout=5.0)
                            except asyncio.TimeoutError:
                                proc.kill()
                            comm_task.cancel()
                            return -1, b"stall: 0 frames produced"
                    else:
                        stall_polls = 0  # reset once frames start appearing
                except Exception:
                    pass
        _, stderr = comm_task.result()
    except BaseException:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            proc.kill()
        comm_task.cancel()
        raise

    return proc.returncode, stderr


async def extract_frames(
    video_path: str,
    output_dir: str,
    sample_rate: Optional[int] = None,
    cancel_check: Optional[Callable] = None,
    progress_callback: Optional[Callable] = None,
    max_frames: Optional[int] = None,
    video_duration: Optional[float] = None,
    video_codec: Optional[str] = None,
) -> tuple[list[FrameData], list[float]]:
    """Extract frames using scene detection + minimum interval fallback.

    Returns (frames, scene_cut_timestamps) where scene_cut_timestamps contains
    the timestamps of frames triggered by scene detection (camera cuts).

    Strategy:
    1. Scene detection (threshold 0.3) captures visual transitions
    2. Minimum interval ensures coverage during static scenes
    3. Maximum frame cap prevents API cost explosion on long videos
    4. Adaptive frame count scales with video duration

    Fallback chain (if earlier attempts produce 0 frames):
    1. GPU decode + scene detection filter
    2. CPU decode + scene detection filter (GPU may be incompatible)
    3. CPU decode + interval-only filter (scene detection may be failing)
    """
    # Adaptive max_frames — no upper ceiling. Guarantees at least 3 frames/min
    # for any video length, with 6 frames/min for shorter content:
    #   0-60 min:  6 frames/min  (e.g. 10 min → 60, 60 min → 360)
    #   60+ min:   3 frames/min  (e.g. 120 min → 540, 3 hrs → 720)
    if max_frames is None:
        if video_duration and video_duration > 0:
            minutes = video_duration / 60
            if minutes <= 60:
                target = int(minutes * settings.FRAMES_PER_MINUTE)  # 6/min
            else:
                target = int(60 * settings.FRAMES_PER_MINUTE + (minutes - 60) * 3)
            max_frames = max(settings.MIN_FRAMES, target)
        else:
            max_frames = 60  # fallback

    rate = sample_rate or settings.FRAME_SAMPLE_RATE

    # For long videos (>30 min), enforce a minimum 30s interval to avoid
    # extracting 400+ frames that overwhelm vision analysis downstream.
    # 113-min video: 1 frame/30s ≈ 226 frames (vs 472 at 1/14.4s).
    if video_duration and video_duration > 1800 and rate < 30:
        logger.info(
            "Long video (%.0fs) — raising frame interval from %ds to 30s",
            video_duration, rate,
        )
        rate = 30

    # Adjust sample rate to avoid over-extraction (extracting 100+ frames then
    # discarding 40% wastes I/O time). Scene detection adds ~30-40% bonus
    # frames on top of interval-based frames, so set the interval so that
    # total (interval + scene) ≈ max_frames.
    if video_duration and video_duration > 0:
        ideal_rate = int(video_duration * 1.35 / max_frames)
        if ideal_rate > rate:
            logger.info(
                "Adaptive frame rate: default=%ds, ideal=%ds (%.0fs video, %d max frames)",
                rate, ideal_rate, video_duration, max_frames,
            )
            rate = ideal_rate
    os.makedirs(output_dir, exist_ok=True)

    # Get GPU decode args — but only if the codec is hardware-supported.
    # NVIDIA NVDEC codec support by GPU generation:
    #   All: h264, hevc, vp8, vp9, mpeg1video, mpeg2video, mpeg4, vc1
    #   RTX 30xx+: av1
    # When the codec isn't supported, skip GPU decode entirely to avoid
    # FFmpeg churning through per-frame CUDA failures for minutes.
    _NVDEC_SUPPORTED_CODECS = {
        "h264", "hevc", "h265", "vp8", "vp9",
        "mpeg1video", "mpeg2video", "mpeg4", "vc1",
    }
    _hw_dec = []
    codec_lower = (video_codec or "").lower()
    if codec_lower and codec_lower not in _NVDEC_SUPPORTED_CODECS:
        logger.info(
            "Skipping GPU decode: codec '%s' not in NVDEC supported set %s",
            codec_lower, _NVDEC_SUPPORTED_CODECS,
        )
    else:
        _hw_dec = _get_gpu_decode_args()
        if _hw_dec and codec_lower:
            logger.info("GPU decode enabled for codec '%s'", codec_lower)

    scene_filter = _build_scene_filter(rate)
    interval_filter = _build_interval_filter(rate)

    # Build fallback chain: try progressively simpler extraction strategies
    attempts = []
    if _hw_dec:
        # Attempt 1: GPU decode + scene detection
        attempts.append((_hw_dec, scene_filter, "GPU+scene"))
    # Attempt 2 (or 1 if no GPU): CPU decode + scene detection
    attempts.append(([], scene_filter, "CPU+scene"))
    # Attempt 3: CPU decode + interval-only (no scene detection at all)
    attempts.append(([], interval_filter, "CPU+interval"))

    # Quick-test GPU decode: if GPU args are present, run a 5-second probe
    # to verify the GPU can actually decode this codec. This prevents the
    # main extraction from churning for minutes on unsupported codecs that
    # slip past the NVDEC_SUPPORTED_CODECS check.
    if _hw_dec and attempts[0][2] == "GPU+scene":
        quick_test_filter = _build_interval_filter(2)  # 1 frame every 2s
        quick_cmd = [
            "ffmpeg", "-y", "-threads", "0",
            *_hw_dec,
            "-t", "5",  # Only process first 5 seconds
            "-i", video_path,
            "-an", "-vf", quick_test_filter,
            "-vsync", "vfr", "-q:v", "12",
            os.path.join(output_dir, "gpu_test_%06d.jpg"),
        ]
        logger.info("GPU quick-test: probing first 5s with %s", " ".join(_hw_dec))
        try:
            qt_proc = await asyncio.create_subprocess_exec(
                *quick_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, qt_stderr = await asyncio.wait_for(qt_proc.communicate(), timeout=15)
            qt_frames = len([
                f for f in os.listdir(output_dir)
                if f.startswith("gpu_test_") and f.endswith(".jpg")
            ])
            # Clean up test frames
            for f in os.listdir(output_dir):
                if f.startswith("gpu_test_"):
                    try:
                        os.remove(os.path.join(output_dir, f))
                    except OSError:
                        pass

            if qt_frames == 0 or qt_proc.returncode != 0:
                stderr_preview = qt_stderr.decode(errors='replace')[:200] if qt_stderr else ""
                logger.warning(
                    "GPU quick-test failed: %d frames, rc=%d, stderr=%s — removing GPU from fallback chain",
                    qt_frames, qt_proc.returncode, stderr_preview,
                )
                # Remove the GPU attempt from the chain
                attempts = [a for a in attempts if a[2] != "GPU+scene"]
            else:
                logger.info("GPU quick-test passed: %d frames in 5s", qt_frames)
        except asyncio.TimeoutError:
            logger.warning("GPU quick-test timed out after 15s — removing GPU from fallback chain")
            attempts = [a for a in attempts if a[2] != "GPU+scene"]
            # Kill the timed-out process (may have already exited)
            try:
                qt_proc.terminate()
                await asyncio.wait_for(qt_proc.wait(), timeout=5)
            except ProcessLookupError:
                pass  # Process already exited
            except Exception:
                try:
                    qt_proc.kill()
                except ProcessLookupError:
                    pass  # Process already exited
        except Exception as e:
            logger.warning("GPU quick-test error: %s — removing GPU from fallback chain", e)
            attempts = [a for a in attempts if a[2] != "GPU+scene"]

    returncode = -1
    stderr = b""
    for hw_args, vf_filter, label in attempts:
        # Clean up any frames from previous failed attempt
        for old_frame in os.listdir(output_dir):
            if old_frame.startswith("frame_") and old_frame.endswith(".jpg"):
                try:
                    os.remove(os.path.join(output_dir, old_frame))
                except OSError:
                    pass

        returncode, stderr = await _run_ffmpeg_extraction(
            video_path, output_dir, vf_filter, hw_args,
            cancel_check, progress_callback, label,
        )

        # Check if this attempt produced frames
        frame_count = len([
            f for f in os.listdir(output_dir)
            if f.startswith("frame_") and f.endswith(".jpg")
        ])

        if returncode == 0 and frame_count > 0:
            logger.info(
                "Frame extraction succeeded (%s): %d frames",
                label, frame_count,
            )
            break

        # Log the failure and try next strategy
        if returncode == -1:
            logger.warning(
                "Frame extraction stalled (%s): 0 frames after %ds, trying next strategy",
                label, _STALL_TIMEOUT,
            )
        elif returncode != 0:
            logger.warning(
                "Frame extraction failed (%s, rc=%d): %s — trying next strategy",
                label, returncode, _extract_ffmpeg_error(stderr),
            )
        else:
            logger.warning(
                "Frame extraction produced 0 frames (%s, rc=0) — trying next strategy",
                label,
            )

    if returncode != 0 and returncode != -1:
        error_msg = _extract_ffmpeg_error(stderr)
        logger.error("FFmpeg frame extraction failed (all strategies): %s", error_msg)
        raise RuntimeError(f"FFmpeg frame extraction failed:\n{error_msg}")

    # Collect extracted frames with actual timestamps from PTS
    frame_files = sorted(
        f for f in os.listdir(output_dir) if f.startswith("frame_") and f.endswith(".jpg")
    )

    if not frame_files:
        raise RuntimeError(
            "FFmpeg extracted 0 frames from the video. The file may be too short, "
            "contain only audio, or use an unsupported codec."
        )

    # Build ALL frames first with rough timestamps — we need timestamps
    # BEFORE capping so we can identify scene-change frames to preserve.
    all_frames = []
    for idx, fname in enumerate(frame_files):
        path = os.path.join(output_dir, fname)
        if video_duration and video_duration > 0 and len(frame_files) > 1:
            timestamp = (idx / (len(frame_files) - 1)) * video_duration
        else:
            timestamp = idx * rate
        all_frames.append(FrameData(timestamp=float(timestamp), path=path))

    # Refine timestamps using a single ffmpeg pass with the showinfo
    # filter. The previous implementation spawned one ffprobe process
    # per frame (up to 200 concurrently); this collapses those spawns
    # into one subprocess that emits every frame's pts_time on stderr.
    if len(all_frames) <= 240:
        try:
            pts_results = await _bulk_probe_pts(output_dir, len(all_frames))
            for frame, pts in zip(all_frames, pts_results):
                if isinstance(pts, float):
                    frame.timestamp = pts
        except Exception as e:
            logger.warning("Could not refine frame timestamps: %s", e)
    else:
        logger.info(
            "Skipping bulk PTS refinement for %d frames (>240)", len(all_frames),
        )

    # ── Smart frame capping: preserve scene-change frames ──
    # Scene-change frames have irregular spacing (not multiples of rate).
    # They mark camera cuts where crop position must snap instantly —
    # the MOST important frames for subject tracking.
    scene_cut_timestamps: list[float] = []
    was_capped = False

    # Identify scene-change frames from irregular timestamp gaps
    scene_change_indices: set[int] = set()
    if len(all_frames) >= 2:
        for i in range(1, len(all_frames)):
            gap = all_frames[i].timestamp - all_frames[i - 1].timestamp
            if gap < rate * 0.7:
                scene_change_indices.add(i)
                scene_cut_timestamps.append(all_frames[i].timestamp)

    if len(all_frames) > max_frames:
        original_count = len(all_frames)

        # Anchors we always want to keep — first + last frame so
        # coverage actually spans the full video duration instead of
        # skewing toward the start. ``sorted(priority)[:max_frames]``
        # used to lose the last frame (and the latest scene cuts)
        # whenever scene-change frames alone exceeded the cap —
        # producing "first few scenes analyzed, late video skipped"
        # on 30+ minute anime / sports / podcast uploads.
        last_idx = len(all_frames) - 1
        anchors = {0, last_idx}

        if len(scene_change_indices) + len(anchors) <= max_frames:
            # Common case: scene cuts + anchors fit — fill the rest
            # with evenly-spaced interval frames as before.
            priority_indices = set(anchors) | scene_change_indices
            remaining_budget = max_frames - len(priority_indices)
            if remaining_budget > 0:
                non_priority = [i for i in range(len(all_frames)) if i not in priority_indices]
                if non_priority:
                    step = max(1, len(non_priority) / remaining_budget)
                    for j in range(min(remaining_budget, len(non_priority))):
                        priority_indices.add(non_priority[int(j * step)])
            indices = sorted(priority_indices)
        else:
            # Scene cuts alone overflow the budget. Evenly downsample
            # across the *sorted scene-cut list* so coverage stays
            # uniform over the whole timeline instead of collapsing
            # to just the earliest cuts. Anchors are always kept.
            sorted_cuts = sorted(scene_change_indices)
            cut_slots = max(0, max_frames - len(anchors))
            if cut_slots > 0 and sorted_cuts:
                step = len(sorted_cuts) / cut_slots
                sampled_cuts = {
                    sorted_cuts[min(int(j * step), len(sorted_cuts) - 1)]
                    for j in range(cut_slots)
                }
            else:
                sampled_cuts = set()
            indices = sorted(anchors | sampled_cuts)

        # Hard guard — even in pathological cases we never exceed the
        # budget, and we always include the first + last anchor.
        if len(indices) > max_frames:
            step = len(indices) / max_frames
            kept = [indices[min(int(j * step), len(indices) - 1)] for j in range(max_frames)]
            # Re-insert anchors if stride-sampling dropped them.
            kept_set = set(kept)
            for a in (0, last_idx):
                if a not in kept_set and len(kept) > 0:
                    kept[-1 if a == last_idx else 0] = a
            indices = sorted(set(kept))

        kept_set = set(indices)
        frames = [all_frames[i] for i in indices]
        # Filter scene cuts to only those that survived capping
        scene_cut_timestamps = [
            all_frames[i].timestamp for i in scene_change_indices if i in kept_set
        ]
        was_capped = True
        scene_kept = len(scene_change_indices & kept_set)
        # Coverage span — how much of the video timeline made it into
        # the capped set. Values near 1.0 mean scenes span first-to-
        # last; values near 0.5 mean we skewed to one half.
        if len(frames) >= 2:
            cov_span = (frames[-1].timestamp - frames[0].timestamp)
            cov_total = (all_frames[-1].timestamp - all_frames[0].timestamp) or 1.0
            coverage_pct = 100.0 * cov_span / cov_total
        else:
            coverage_pct = 0.0
        logger.info(
            "Smart-capped frames from %d to %d (kept %d/%d scene-change frames, "
            "timeline coverage %.0f%%)",
            original_count, len(frames), scene_kept, len(scene_change_indices),
            coverage_pct,
        )
    else:
        frames = all_frames

    logger.info(
        "Scene-aware extraction complete: %d frames from %s "
        "(scene detection + %ds interval, %d scene cuts identified)",
        len(frames), video_path, rate, len(scene_cut_timestamps),
    )
    return frames, scene_cut_timestamps


async def extract_frames_at_timestamps(
    video_path: str,
    output_dir: str,
    timestamps: list[float],
    cancel_check: Optional[Callable] = None,
    progress_callback: Optional[Callable] = None,
    video_codec: Optional[str] = None,
) -> list[FrameData]:
    """Extract frames at a precise list of timestamps (Phase 4 adaptive path).

    Uses ffmpeg's ``select='between(t,T1,T1+d)+...'`` filter for accurate
    timestamp-based selection. Falls back to GPU→CPU decode chain identical
    to extract_frames(). Returns FrameData in the same shape so downstream
    code is interchangeable.

    Constraints:
      - timestamps must be sorted ascending and within video duration.
      - Caps at 240 timestamps per call (ffmpeg select filter has a
        practical limit around that).
      - Each timestamp is rounded to 0.01s so the between() comparison
        doesn't fail on float precision.
    """
    if not timestamps:
        raise RuntimeError("extract_frames_at_timestamps: empty timestamp list")

    os.makedirs(output_dir, exist_ok=True)
    _deduped = sorted({round(float(t), 2) for t in timestamps if t >= 0})
    if not _deduped:
        raise RuntimeError("extract_frames_at_timestamps: no valid timestamps")

    # ffmpeg's ``select='between(...)+between(...)'`` filter starts
    # failing near ~240 clauses per invocation. Previously the code
    # did ``[:240]`` — which silently drops everything chronologically
    # past the cap, so a 2-hour video whose adaptive sampler asked
    # for 480 timestamps would only re-extract the *first half* and
    # the back half of the episode was invisible to the vision model.
    # Stride-sample evenly across the whole list instead so coverage
    # stays uniform over the full duration, and keep first+last as
    # anchors so the clip selector still sees the start and end.
    _MAX_PER_CALL = 240
    if len(_deduped) > _MAX_PER_CALL:
        step = len(_deduped) / _MAX_PER_CALL
        sampled = [_deduped[min(int(j * step), len(_deduped) - 1)] for j in range(_MAX_PER_CALL)]
        # Preserve first + last to pin the coverage window.
        sampled_set = set(sampled)
        sampled_set.add(_deduped[0])
        sampled_set.add(_deduped[-1])
        if len(sampled_set) > _MAX_PER_CALL:
            # Strided sample could collide with anchors — re-stride.
            extras = sorted(sampled_set - {_deduped[0], _deduped[-1]})
            keep_count = _MAX_PER_CALL - 2
            step = len(extras) / max(1, keep_count)
            extras = [extras[min(int(j * step), len(extras) - 1)] for j in range(keep_count)]
            sampled_set = set(extras) | {_deduped[0], _deduped[-1]}
        sorted_ts = sorted(sampled_set)
        logger.info(
            "extract_frames_at_timestamps: downsampled %d → %d timestamps "
            "via uniform stride (preserving first+last; coverage %.0fs-%.0fs)",
            len(_deduped), len(sorted_ts), sorted_ts[0], sorted_ts[-1],
        )
    else:
        sorted_ts = _deduped

    # Build the select filter: 'between(t,1.5,1.55)+between(t,3.2,3.25)+...'
    # The +0.05 tolerance handles ffmpeg's floating-point pts comparison.
    select_clauses = "+".join(
        f"between(t\\,{t:.2f}\\,{t + 0.05:.2f})" for t in sorted_ts
    )
    vf_filter = (
        f"select='{select_clauses}',"
        "scale='min(1024\\,iw)':'min(576\\,ih)':force_original_aspect_ratio=decrease,"
        "format=pix_fmts=yuvj420p"
    )

    # Reuse existing GPU-decode probe logic from extract_frames.
    _NVDEC_SUPPORTED_CODECS = {
        "h264", "hevc", "h265", "vp8", "vp9",
        "mpeg1video", "mpeg2video", "mpeg4", "vc1",
    }
    codec_lower = (video_codec or "").lower()
    hw_dec: list[str] = []
    if not codec_lower or codec_lower in _NVDEC_SUPPORTED_CODECS:
        hw_dec = _get_gpu_decode_args()

    attempts: list[tuple[list[str], str]] = []
    if hw_dec:
        attempts.append((hw_dec, "GPU+adaptive"))
    attempts.append(([], "CPU+adaptive"))

    returncode = -1
    stderr = b""
    for hw_args, label in attempts:
        for old in os.listdir(output_dir):
            if old.startswith("frame_") and old.endswith(".jpg"):
                try:
                    os.remove(os.path.join(output_dir, old))
                except OSError:
                    pass

        returncode, stderr = await _run_ffmpeg_extraction(
            video_path, output_dir, vf_filter, hw_args,
            cancel_check, progress_callback, label,
        )
        frame_count = len([
            f for f in os.listdir(output_dir)
            if f.startswith("frame_") and f.endswith(".jpg")
        ])
        if returncode == 0 and frame_count > 0:
            logger.info(
                "Adaptive extraction succeeded (%s): %d/%d frames",
                label, frame_count, len(sorted_ts),
            )
            break
        logger.warning(
            "Adaptive extraction (%s) failed: rc=%d, %s — trying next",
            label, returncode, _extract_ffmpeg_error(stderr),
        )

    if returncode != 0:
        raise RuntimeError(
            "Adaptive frame extraction failed:\n"
            + _extract_ffmpeg_error(stderr)
        )

    frame_files = sorted(
        f for f in os.listdir(output_dir)
        if f.startswith("frame_") and f.endswith(".jpg")
    )
    # We requested N timestamps, ffmpeg returned M files in order.
    # Map them back by index — fewer-than-requested is OK (e.g. last
    # timestamp past EOF) but log it.
    if len(frame_files) < len(sorted_ts):
        logger.info(
            "Adaptive extraction: requested %d frames, got %d (%.0f%%)",
            len(sorted_ts), len(frame_files),
            100 * len(frame_files) / len(sorted_ts),
        )

    frames = [
        FrameData(timestamp=float(sorted_ts[i]), path=os.path.join(output_dir, fname))
        for i, fname in enumerate(frame_files[:len(sorted_ts)])
    ]
    return frames


def resize_frame_if_needed(path: str) -> str:
    """Resize frame to max 1568px on longest side. Returns path (may be same)."""
    try:
        img = Image.open(path)
        w, h = img.size
        if max(w, h) > MAX_DIMENSION:
            if w > h:
                new_w = MAX_DIMENSION
                new_h = int(h * MAX_DIMENSION / w)
            else:
                new_h = MAX_DIMENSION
                new_w = int(w * MAX_DIMENSION / h)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            img.save(path, "JPEG", quality=85)
    except Exception as e:
        logger.warning(f"Failed to resize frame {path}: {e}")
    return path


def frame_to_base64(path: str, skip_resize: bool = False) -> str:
    """Read frame and return base64 encoded string.

    Reads the file in a single pass and encodes to base64.
    skip_resize=True is recommended when frames were already
    downscaled during extraction (the default pipeline path).
    """
    if not skip_resize:
        resize_frame_if_needed(path)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def extract_audio(
    video_path: str,
    output_path: str,
    cancel_check: Optional[Callable] = None,
    *,
    precondition: bool = False,
) -> str:
    """Extract audio track from video as WAV for Whisper.

    When ``precondition=True`` the audio is pre-conditioned in the same
    FFmpeg pass before being written to disk. Three cheap filters,
    applied in order:

      1. ``highpass=f=80`` — strip AC hum / room rumble below 80 Hz.
         Invisible on voice, kills the low-frequency mud that makes
         Whisper's VAD think silence is speech.
      2. ``afftdn=nf=-25`` — adaptive FFT denoiser targeting a -25 dB
         noise floor. Pulls quiet speech out of constant backgrounds
         (HVAC, traffic, crowd, music beds) without chewing on plosives.
      3. ``loudnorm=I=-18:LRA=11:TP=-1.5`` — ITU BS.1770-3 single-pass
         loudness normalization. Whispered / off-mic speech that was
         at -40 dBFS in the source gets lifted into Whisper's sensitive
         band, which is the single biggest reason transcript coverage
         drops on mixed or poorly-mastered audio.

    The filters cost a few extra seconds on long videos but are
    decisively worth it — each one is the output of testing against
    the transcript-coverage regressions we hit on music-heavy /
    cinematic / anime content.

    Preconditioning is OFF by default for backward compat; the
    pipeline flips it on via ``settings.WHISPER_AUDIO_PRECONDITION``.
    """
    _check_disk_space(output_path, required_mb=500)

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
    ]
    if precondition:
        af_chain = "highpass=f=80,afftdn=nf=-25,loudnorm=I=-18:LRA=11:TP=-1.5"
        cmd.extend(["-af", af_chain])
    cmd.extend([
        "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        output_path,
    ])
    logger.info(
        "FFmpeg audio extraction command (precondition=%s): %s",
        precondition, " ".join(cmd),
    )
    returncode, stderr = await _run_subprocess_cancellable(cmd, cancel_check)
    if returncode != 0:
        # Preconditioning can fail on unusual container layouts / corrupt
        # audio streams. Retry once without filters so the analysis can
        # still proceed — degraded coverage beats a hard failure.
        if precondition:
            logger.warning(
                "Preconditioned audio extraction failed (exit %d). Retrying "
                "without filters: %s",
                returncode, _extract_ffmpeg_error(stderr),
            )
            fallback_cmd = [
                "ffmpeg", "-y", "-i", video_path, "-vn",
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                output_path,
            ]
            returncode, stderr = await _run_subprocess_cancellable(fallback_cmd, cancel_check)
        if returncode != 0:
            error_msg = _extract_ffmpeg_error(stderr)
            input_mb = os.path.getsize(video_path) / (1024 * 1024) if os.path.exists(video_path) else 0
            try:
                free_mb = shutil.disk_usage(os.path.dirname(output_path)).free / (1024 * 1024)
            except OSError:
                free_mb = -1
            logger.error(
                "ffmpeg audio extraction failed (exit %d)\n"
                "Input: %s (%.1fMB)\nOutput: %s\nDisk free: %.0fMB\nError: %s",
                returncode, video_path, input_mb, output_path, free_mb, error_msg,
            )
            raise RuntimeError(f"Audio extraction failed:\n{error_msg}")
    logger.info("Audio extraction complete: %s", output_path)
    return output_path
