import asyncio
import glob
import json
import logging
import math
import os
import platform
import subprocess

import re

_IS_WINDOWS = platform.system() == "Windows"
_IS_MACOS = platform.system() == "Darwin"

# Check if ffmpeg has drawtext filter (requires libfreetype at compile time)
_HAS_DRAWTEXT = False
try:
    _dt_check = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True, timeout=5)
    _HAS_DRAWTEXT = "drawtext" in _dt_check.stdout
except Exception:
    pass

from backend.config import settings as app_settings
from backend import database
from backend.models import TranscriptSegment
from backend.services.ass_generator import (
    generate_ass,
    FONT_SIZE_MAP as ASS_FONT_SIZE_MAP,
    FONT_WEIGHT_MAP as ASS_FONT_WEIGHT_MAP,
    POSITION_ALIGNMENT as ASS_POSITION_ALIGNMENT,
    REF_W as ASS_REF_W,
    REF_H as ASS_REF_H,
    _normalize_font_weight,
    MIN_TEXT_AREA_W,
    MIN_TEXT_AREA_H,
    _hex_to_ass_color,
    _hex_to_ass_color_with_alpha,
)

logger = logging.getLogger(__name__)

if not _HAS_DRAWTEXT:
    logger.warning(
        "FFmpeg 'drawtext' filter not available — "
        "text overlays will be rendered via ASS subtitles instead"
    )


# ---------------------------------------------------------------------------
# GPU hardware acceleration detection & encoder selection
# ---------------------------------------------------------------------------

_gpu_info: dict | None = None


def get_encoder_label() -> str:
    """Return a user-friendly label describing the current video encoder + GPU.

    Examples:
      "NVENC (NVIDIA GeForce RTX 4070)"
      "VideoToolbox (Apple M2 Pro)"
      "libx264 (CPU)"
    """
    if not app_settings.GPU_ACCELERATION_ENABLED:
        return "libx264 (CPU)"
    gpu = detect_gpu_capabilities()
    encoder = gpu.get("encoder", "libx264")
    gpu_name = gpu.get("gpu_name", "")
    if encoder == "libx264":
        return "libx264 (CPU)"
    # Map encoder codecs to friendly names
    _ENCODER_NAMES = {
        "h264_nvenc": "NVENC H.264",
        "hevc_nvenc": "NVENC HEVC",
        "h264_qsv": "QuickSync H.264",
        "hevc_qsv": "QuickSync HEVC",
        "h264_vaapi": "VAAPI H.264",
        "hevc_vaapi": "VAAPI HEVC",
        "h264_videotoolbox": "VideoToolbox H.264",
        "hevc_videotoolbox": "VideoToolbox HEVC",
    }
    friendly = _ENCODER_NAMES.get(encoder, encoder)
    if gpu_name and gpu_name != "None (CPU only)":
        return f"{friendly} ({gpu_name})"
    return friendly


def _gpu_info_cache_clear():
    """Clear the cached GPU info so next detect_gpu_capabilities() call re-scans."""
    global _gpu_info
    _gpu_info = None


def detect_gpu_capabilities(force_redetect: bool = False) -> dict:
    """Detect available GPU hardware encoders/decoders.

    Called when:
    - User toggles GPU acceleration ON in Settings (force_redetect=True)
    - First FFmpeg export after container start (lazy init from _gpu_encode_args)
    - GET /api/gpu-acceleration endpoint

    Detection steps:
    1. Check GPU_ACCELERATION_ENABLED — if False, return CPU fallback immediately
    2. Run nvidia-smi to detect NVIDIA GPU (name, VRAM, driver version)
    3. Probe FFmpeg for compiled-in encoders (h264_nvenc, h264_vaapi, h264_qsv)
    4. Test-encode a tiny null video with each detected encoder to confirm it works
    5. Check CUDA availability for faster-whisper transcription
    6. Return best available encoder + full GPU info dict
    """
    global _gpu_info

    if _gpu_info is not None and not force_redetect:
        return _gpu_info

    # Default: CPU-only fallback
    info = {
        "vendor": "none",
        "gpu_name": "None (CPU only)",
        "encoder": "libx264",
        "hevc_encoder": None,
        "decoder": None,
        "hwaccel": None,
        "hwaccel_device": None,
        "scale_filter": "scale",
        "capabilities": ["encode_cpu"],
        "vram_mb": 0,
        "driver_version": "",
        "cuda_available": False,
        "whisper_device": "cpu",
    }

    # Gate on user toggle — if GPU acceleration is toggled OFF in Settings,
    # return CPU immediately but still detect GPUs so the UI can show them.
    if not app_settings.GPU_ACCELERATION_ENABLED:
        info["gpus"] = _detect_all_gpus()
        _gpu_info = info
        return info

    forced_vendor = (app_settings.GPU_VENDOR_OVERRIDE or "").lower().strip() or "auto"

    # ── Step 1: Detect NVIDIA GPU(s) via multiple methods ──
    nvidia_detected = False
    nvidia_gpus = []
    if forced_vendor in ("auto", "nvidia"):
        # Method 1a: nvidia-smi (most reliable)
        try:
            smi = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if smi.returncode == 0 and smi.stdout.strip():
                for line in smi.stdout.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    parts = [p.strip() for p in line.split(",")]
                    gpu_idx = parts[0] if len(parts) > 0 else "0"
                    gpu_name = parts[1] if len(parts) > 1 else "NVIDIA GPU"
                    vram = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
                    driver = parts[3] if len(parts) > 3 else ""
                    nvidia_gpus.append({
                        "index": gpu_idx, "name": gpu_name,
                        "vram_mb": vram, "driver_version": driver,
                        "vendor": "nvidia",
                    })
                    logger.info("NVIDIA GPU %s detected: %s (%d MB VRAM, driver %s)", gpu_idx, gpu_name, vram, driver)
                if nvidia_gpus:
                    nvidia_detected = True
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.debug("nvidia-smi not available: %s", e)

        # Method 1b: /dev/nvidia* device nodes (GPU passthrough without nvidia-smi)
        if not nvidia_detected:
            try:
                nvidia_devs = sorted(glob.glob("/dev/nvidia[0-9]*"))
                if nvidia_devs:
                    nvidia_detected = True
                    for i, dev in enumerate(nvidia_devs):
                        gpu_name = _get_proc_nvidia_name(i) or f"NVIDIA GPU ({os.path.basename(dev)})"
                        nvidia_gpus.append({
                            "index": str(i), "name": gpu_name,
                            "vram_mb": 0, "driver_version": "", "vendor": "nvidia",
                        })
                        logger.info("NVIDIA GPU %d detected via device node: %s", i, gpu_name)
            except Exception:
                pass

        # Method 1c: /proc/driver/nvidia/gpus/ (kernel module loaded)
        if not nvidia_detected:
            try:
                nv_info_paths = sorted(glob.glob("/proc/driver/nvidia/gpus/*/information"))
                for i, info_path in enumerate(nv_info_paths):
                    try:
                        content = open(info_path).read()
                        gpu_name = "NVIDIA GPU"
                        for line in content.split("\n"):
                            if line.startswith("Model:"):
                                gpu_name = line.split(":", 1)[1].strip()
                                break
                        nvidia_gpus.append({
                            "index": str(i), "name": gpu_name,
                            "vram_mb": 0, "driver_version": "", "vendor": "nvidia",
                        })
                        nvidia_detected = True
                        logger.info("NVIDIA GPU %d detected via /proc: %s", i, gpu_name)
                    except (OSError, IOError):
                        continue
            except Exception:
                pass

        # Method 1d: sysfs vendor ID check (container with kernel access)
        if not nvidia_detected:
            try:
                for vendor_path in sorted(glob.glob("/sys/class/drm/card[0-9]*/device/vendor")):
                    try:
                        vendor_id = open(vendor_path).read().strip().lower()
                        if vendor_id == "0x10de":
                            device_dir = os.path.dirname(vendor_path)
                            gpu_name = "NVIDIA GPU"
                            uevent_path = os.path.join(device_dir, "uevent")
                            if os.path.isfile(uevent_path):
                                for line in open(uevent_path):
                                    if line.startswith("PCI_SLOT_NAME="):
                                        slot = line.strip().split("=", 1)[1]
                                        gpu_name = f"NVIDIA GPU ({slot})"
                                        break
                            # Read VRAM from BAR sizes
                            vram_mb = 0
                            resource_path = os.path.join(device_dir, "resource")
                            if os.path.isfile(resource_path):
                                try:
                                    for res_line in open(resource_path):
                                        parts = res_line.strip().split()
                                        if len(parts) >= 2:
                                            start = int(parts[0], 16)
                                            end = int(parts[1], 16)
                                            size_mb = (end - start + 1) // (1024 * 1024)
                                            if size_mb > vram_mb:
                                                vram_mb = size_mb
                                except (ValueError, IndexError):
                                    pass
                            nvidia_gpus.append({
                                "index": str(len(nvidia_gpus)), "name": gpu_name,
                                "vram_mb": vram_mb, "driver_version": "", "vendor": "nvidia",
                            })
                            nvidia_detected = True
                            logger.info("NVIDIA GPU detected via sysfs: %s (%d MB)", gpu_name, vram_mb)
                    except (OSError, IOError):
                        continue
            except Exception:
                pass

        if nvidia_detected and nvidia_gpus:
            # Select the most capable GPU (highest VRAM) — ensures a locally
            # passed-through high-end GPU is preferred over a server's weaker GPU.
            best_gpu = max(nvidia_gpus, key=lambda g: g.get("vram_mb", 0))
            if len(nvidia_gpus) > 1:
                logger.info(
                    "Multiple NVIDIA GPUs detected — selecting %s (index %s, %d MB) over %s",
                    best_gpu["name"], best_gpu["index"], best_gpu.get("vram_mb", 0),
                    ", ".join(g["name"] for g in nvidia_gpus if g["index"] != best_gpu["index"]),
                )
            info.update({
                "gpu_name": best_gpu["name"],
                "vram_mb": best_gpu.get("vram_mb", 0),
                "driver_version": best_gpu.get("driver_version", ""),
                "gpu_device_index": best_gpu["index"],
            })

    # ── Step 2: Check CUDA for Whisper ──
    # Use multiple methods — ctranslate2 and /dev/nvidia* device nodes.
    # The main process may not have a working CUDA context, but the Whisper
    # subprocess creates its own via CTranslate2 and can use CUDA fine.
    if not info.get("cuda_available"):
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                info["cuda_available"] = True
                info["whisper_device"] = "cuda"
                if "whisper_cuda" not in info["capabilities"]:
                    info["capabilities"].append("whisper_cuda")
                logger.info("CUDA available for Whisper transcription (via CTranslate2)")
        except Exception:
            pass

    # Fallback: if ctranslate2 CUDA check failed but /dev/nvidia* exists,
    # Whisper can still use CUDA in subprocess mode (CTranslate2 creates its
    # own CUDA context in the subprocess).
    if not info.get("cuda_available") and nvidia_detected:
        info["cuda_available"] = True
        info["whisper_device"] = "cuda"
        if "whisper_cuda" not in info["capabilities"]:
            info["capabilities"].append("whisper_cuda")
        logger.info("CUDA available for Whisper transcription (via /dev/nvidia* device nodes, subprocess mode)")

    # ── Step 3: Probe FFmpeg for available HW encoders ──
    available_encoders: set[str] = set()
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        for enc in (
            "h264_nvenc", "h264_vaapi", "h264_qsv",
            "hevc_nvenc", "hevc_vaapi", "hevc_qsv",
            "h264_videotoolbox", "hevc_videotoolbox",
        ):
            if enc in result.stdout:
                available_encoders.add(enc)
        logger.info("FFmpeg HW encoders available: %s", available_encoders or "none")
    except Exception as e:
        logger.warning("FFmpeg encoder probe failed: %s", e)

    # ── Step 4: Test each encoder (priority: NVENC > QSV > VAAPI) ──
    encoder_configs = []

    if "h264_nvenc" in available_encoders and (forced_vendor in ("auto", "nvidia") or nvidia_detected):
        # Use the best GPU's device index for the test encode
        nvenc_gpu_idx = info.get("gpu_device_index", "")
        nvenc_test_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "nullsrc=s=256x256:d=0.1",
        ]
        if nvenc_gpu_idx:
            nvenc_test_cmd += ["-gpu", str(nvenc_gpu_idx)]
        nvenc_test_cmd += ["-c:v", "h264_nvenc", "-f", "null", "-"]
        encoder_configs.append({
            "vendor": "nvidia", "encoder": "h264_nvenc", "decoder": "h264_cuvid",
            "hwaccel": "cuda", "hwaccel_device": None,
            "test_cmd": nvenc_test_cmd,
        })

    if "h264_qsv" in available_encoders and forced_vendor in ("auto", "intel"):
        # On Windows, QSV works without an explicit device path; on Linux
        # it needs the DRI render node.
        qsv_device = None if _IS_WINDOWS else "/dev/dri/renderD128"
        qsv_test_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "nullsrc=s=256x256:d=0.1",
            "-c:v", "h264_qsv", "-f", "null", "-",
        ]
        # On Windows, QSV auto-discovers the Intel GPU via DXVA2/D3D11
        if not _IS_WINDOWS and qsv_device:
            qsv_ok = os.path.exists(qsv_device)
        else:
            qsv_ok = True
        if qsv_ok:
            encoder_configs.append({
                "vendor": "intel", "encoder": "h264_qsv", "decoder": "h264_qsv",
                "hwaccel": "qsv", "hwaccel_device": qsv_device,
                "test_cmd": qsv_test_cmd,
            })

    if "h264_vaapi" in available_encoders and forced_vendor in ("auto", "intel", "amd"):
        # VAAPI is Linux-only (not available on Windows)
        vaapi_dev = "/dev/dri/renderD128"
        if not _IS_WINDOWS and os.path.exists(vaapi_dev):
            encoder_configs.append({
                "vendor": "amd" if forced_vendor == "amd" else "intel",
                "encoder": "h264_vaapi", "decoder": "h264_vaapi",
                "hwaccel": "vaapi", "hwaccel_device": vaapi_dev,
                "test_cmd": [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-vaapi_device", vaapi_dev,
                    "-f", "lavfi", "-i", "nullsrc=s=256x256:d=0.1",
                    "-vf", "format=nv12,hwupload",
                    "-c:v", "h264_vaapi", "-f", "null", "-",
                ],
            })

    # ── macOS VideoToolbox (Apple Silicon & Intel Macs) ──
    # VideoToolbox uses the Apple Media Engine for H.264/HEVC encode/decode
    # on M1/M2/M3/M4 chips and the Intel iGPU on older Macs.
    if _IS_MACOS and "h264_videotoolbox" in available_encoders and forced_vendor in ("auto", "apple"):
        encoder_configs.append({
            "vendor": "apple", "encoder": "h264_videotoolbox", "decoder": None,
            "hwaccel": "videotoolbox", "hwaccel_device": None,
            "test_cmd": [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "nullsrc=s=256x256:d=0.1",
                "-c:v", "h264_videotoolbox", "-f", "null", "-",
            ],
        })

    for cfg in encoder_configs:
        try:
            test = subprocess.run(cfg["test_cmd"], capture_output=True, text=True, timeout=15)
            if test.returncode == 0:
                info.update({
                    "vendor": cfg["vendor"], "encoder": cfg["encoder"],
                    "decoder": cfg["decoder"], "hwaccel": cfg["hwaccel"],
                    "hwaccel_device": cfg["hwaccel_device"],
                    "capabilities": ["encode_gpu", "decode_gpu"],
                })
                if info["cuda_available"]:
                    info["capabilities"].append("whisper_cuda")
                # Probe HEVC encoder availability (for 4K exports)
                hevc_enc = {
                    "nvidia": "hevc_nvenc", "intel": "hevc_qsv",
                    "amd": "hevc_vaapi", "apple": "hevc_videotoolbox",
                }.get(cfg["vendor"])
                if hevc_enc and hevc_enc in available_encoders:
                    info["hevc_encoder"] = hevc_enc
                    info["capabilities"].append("encode_hevc_gpu")
                    logger.info("HEVC GPU encoder available: %s", hevc_enc)
                # Detect GPU name for Intel/AMD if not set by nvidia-smi
                if cfg["vendor"] != "nvidia" and "None" in info["gpu_name"]:
                    _detect_non_nvidia_gpu_name(info, cfg["vendor"])
                logger.info("GPU acceleration active: %s encoder=%s", cfg["vendor"].upper(), cfg["encoder"])
                info["gpus"] = _detect_all_gpus()
                _gpu_info = info
                return info
            else:
                logger.debug("Encoder %s test failed: %s", cfg["encoder"], test.stderr[:200])
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug("Encoder %s test error: %s", cfg["encoder"], e)

    # No working GPU encoder found via test-encode. If NVIDIA GPU was detected
    # via device nodes, still report it as available — the test may have failed
    # transiently because Ollama is using the GPU VRAM. The actual export will
    # retry with the encoder and fail gracefully to CPU if needed.
    if nvidia_detected and nvidia_gpus and "h264_nvenc" in available_encoders:
        logger.warning(
            "NVENC test-encode failed (GPU may be busy with Ollama) but h264_nvenc is "
            "compiled into FFmpeg and /dev/nvidia* exists — reporting GPU as available"
        )
        info.update({
            "vendor": "nvidia",
            "encoder": "h264_nvenc",
            "decoder": "h264_cuvid",
            "hwaccel": "cuda",
            "hwaccel_device": None,
            "capabilities": ["encode_gpu", "decode_gpu"],
        })
        if info.get("cuda_available"):
            info["capabilities"].append("whisper_cuda")
        hevc_enc = "hevc_nvenc"
        if hevc_enc in available_encoders:
            info["hevc_encoder"] = hevc_enc
            info["capabilities"].append("encode_hevc_gpu")
        info["gpus"] = _detect_all_gpus()
        _gpu_info = info
        return info

    # No working GPU encoder and no NVENC fallback — provide actionable guidance
    if nvidia_detected:
        # Check specifically what's missing
        missing_parts = []
        if not available_encoders:
            missing_parts.append("FFmpeg is not compiled with NVENC support (need --enable-nvenc --enable-cuda-llvm)")
        elif "h264_nvenc" not in available_encoders:
            missing_parts.append("h264_nvenc encoder not found in FFmpeg")
        else:
            missing_parts.append("NVENC test-encode failed (GPU may not be accessible to FFmpeg)")

        # Check for nvidia device access
        nv_devs = glob.glob("/dev/nvidia*")
        if not nv_devs:
            missing_parts.append(
                "No /dev/nvidia* devices found — run Docker with: "
                "--gpus all --runtime=nvidia (or add deploy.resources.reservations.devices in docker-compose)"
            )

        # Check for nvidia driver library
        try:
            import ctypes
            ctypes.cdll.LoadLibrary("libnvidia-encode.so.1")
        except OSError:
            missing_parts.append(
                "libnvidia-encode.so.1 not found — install NVIDIA driver libraries "
                "(apt install libnvidia-encode-XXX or use nvidia/cuda base image)"
            )
        except Exception:
            pass

        logger.warning(
            "NVIDIA GPU detected (%s) but NVENC encoding unavailable. Issues:\n  - %s\n"
            "CUDA Whisper transcription will still work if CUDA runtime is available.",
            info["gpu_name"],
            "\n  - ".join(missing_parts),
        )

        # Store diagnostics for the API response
        info["gpu_issues"] = missing_parts
    else:
        logger.info("No GPU detected — using CPU encoding (libx264)")

    info["gpus"] = _detect_all_gpus()
    _gpu_info = info
    return info


def _get_proc_nvidia_name(index: int = 0) -> str:
    """Try to read NVIDIA GPU model name from /proc/driver/nvidia/gpus/."""
    try:
        nv_info_paths = sorted(glob.glob("/proc/driver/nvidia/gpus/*/information"))
        if index < len(nv_info_paths):
            for line in open(nv_info_paths[index]):
                if line.startswith("Model:"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return ""


def _detect_non_nvidia_gpu_name(info: dict, vendor: str):
    """Try to detect Intel/AMD/Apple GPU name from platform-specific tools."""
    try:
        if _IS_MACOS:
            # On macOS, use system_profiler to get GPU name
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-detailLevel", "mini"],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.split("\n"):
                stripped = line.strip()
                if stripped.startswith("Chipset Model:"):
                    info["gpu_name"] = stripped.split(":", 1)[1].strip()
                    break
            return
        elif _IS_WINDOWS:
            # Use WMIC to enumerate display adapters on Windows
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "Name"],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.split("\n"):
                lower = line.strip().lower()
                if not lower or "name" in lower:
                    continue
                if vendor == "intel" and "intel" in lower:
                    info["gpu_name"] = line.strip()
                    break
                elif vendor == "amd" and ("amd" in lower or "radeon" in lower):
                    info["gpu_name"] = line.strip()
                    break
        else:
            result = subprocess.run(["lspci"], capture_output=True, text=True, timeout=5)
            for line in result.stdout.split("\n"):
                lower = line.lower()
                if "vga" in lower or "3d controller" in lower or "display" in lower:
                    if vendor == "intel" and "intel" in lower:
                        info["gpu_name"] = line.split(": ", 1)[-1].strip() if ": " in line else "Intel GPU"
                        break
                    elif vendor == "amd" and ("amd" in lower or "radeon" in lower):
                        info["gpu_name"] = line.split(": ", 1)[-1].strip() if ": " in line else "AMD GPU"
                        break
    except Exception:
        pass


def _detect_all_gpus() -> list[dict]:
    """Detect ALL GPUs visible to the container (NVIDIA, Intel, AMD).

    Returns a list of dicts with: name, vendor, vram_mb, driver_version, type.
    This is called alongside detect_gpu_capabilities() to populate the gpus[] list.
    """
    gpus = []
    seen_names = set()

    # ── NVIDIA GPUs via nvidia-smi ──
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if smi.returncode == 0 and smi.stdout.strip():
            for line in smi.stdout.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                gpu_idx = parts[0] if len(parts) > 0 else "0"
                gpu_name = parts[1] if len(parts) > 1 else "NVIDIA GPU"
                vram = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
                driver = parts[3] if len(parts) > 3 else ""
                gpus.append({
                    "index": gpu_idx, "name": gpu_name, "vendor": "nvidia",
                    "vram_mb": vram, "driver_version": driver, "type": "discrete",
                })
                seen_names.add(gpu_name.lower())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # ── All GPUs via lspci (Linux) — catches Intel iGPU and AMD GPUs ──
    lspci_found = False
    if not _IS_MACOS and not _IS_WINDOWS:
        try:
            result = subprocess.run(["lspci"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lspci_found = True
                for line in result.stdout.split("\n"):
                    lower = line.lower()
                    if not ("vga" in lower or "3d controller" in lower or "display" in lower):
                        continue
                    gpu_name = line.split(": ", 1)[-1].strip() if ": " in line else "Unknown GPU"
                    # Skip if already found via nvidia-smi (avoid duplicates)
                    if any(sn in gpu_name.lower() for sn in seen_names):
                        continue
                    if "nvidia" in lower and any(sn in lower for sn in seen_names):
                        continue
                    vendor = "intel" if "intel" in lower else "amd" if ("amd" in lower or "radeon" in lower) else "nvidia" if "nvidia" in lower else "unknown"
                    gpu_type = "integrated" if vendor == "intel" else "discrete"
                    gpus.append({
                        "index": str(len(gpus)), "name": gpu_name, "vendor": vendor,
                        "vram_mb": 0, "driver_version": "", "type": gpu_type,
                    })
                    seen_names.add(gpu_name.lower())
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # ── Fallback: /sys/class/drm/ (Linux containers without lspci) ──
    # Docker containers often lack lspci (pciutils package).  This fallback
    # reads directly from sysfs to discover Intel/AMD/NVIDIA GPUs that the
    # kernel exposes to the container.
    if not _IS_MACOS and not _IS_WINDOWS and not lspci_found:
        _PCI_GPU_VENDORS = {
            "0x10de": "nvidia", "0x8086": "intel", "0x1002": "amd",
        }
        try:
            for card_dir in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
                try:
                    vendor_path = os.path.join(card_dir, "vendor")
                    if not os.path.isfile(vendor_path):
                        continue
                    vendor_id = open(vendor_path).read().strip().lower()
                    vendor = _PCI_GPU_VENDORS.get(vendor_id)
                    if not vendor:
                        continue
                    # Try to read device name from uevent or construct one
                    gpu_name = f"{vendor.upper()} GPU"
                    uevent_path = os.path.join(card_dir, "uevent")
                    if os.path.isfile(uevent_path):
                        for ue_line in open(uevent_path):
                            if ue_line.startswith("PCI_SLOT_NAME="):
                                slot = ue_line.strip().split("=", 1)[1]
                                gpu_name = f"{vendor.upper()} GPU ({slot})"
                                break
                    # Try to read VRAM from resource (BAR sizes)
                    vram_mb = 0
                    resource_path = os.path.join(card_dir, "resource")
                    if os.path.isfile(resource_path):
                        try:
                            for res_line in open(resource_path):
                                parts = res_line.strip().split()
                                if len(parts) >= 2:
                                    start = int(parts[0], 16)
                                    end = int(parts[1], 16)
                                    size_mb = (end - start + 1) // (1024 * 1024)
                                    if size_mb > vram_mb:
                                        vram_mb = size_mb
                        except (ValueError, IndexError):
                            pass
                    # Skip if already detected (e.g., NVIDIA via nvidia-smi)
                    if vendor == "nvidia" and any(g["vendor"] == "nvidia" for g in gpus):
                        continue
                    if gpu_name.lower() in seen_names:
                        continue
                    gpu_type = "integrated" if vendor == "intel" else "discrete"
                    gpus.append({
                        "index": str(len(gpus)), "name": gpu_name, "vendor": vendor,
                        "vram_mb": vram_mb, "driver_version": "", "type": gpu_type,
                    })
                    seen_names.add(gpu_name.lower())
                except (OSError, IOError):
                    continue
        except Exception:
            pass

    # ── Fallback: /proc/driver/nvidia/gpus/ (multi-NVIDIA without nvidia-smi) ──
    # Some containers have the NVIDIA kernel module loaded but nvidia-smi
    # is not installed.  /proc/driver/nvidia/gpus/ lists all NVIDIA GPUs.
    if not gpus or (not any(g["vendor"] == "nvidia" for g in gpus)):
        try:
            nv_gpu_dirs = sorted(glob.glob("/proc/driver/nvidia/gpus/*/information"))
            for info_path in nv_gpu_dirs:
                try:
                    content = open(info_path).read()
                    gpu_name = "NVIDIA GPU"
                    for line in content.split("\n"):
                        if line.startswith("Model:"):
                            gpu_name = line.split(":", 1)[1].strip()
                            break
                    if gpu_name.lower() in seen_names:
                        continue
                    gpus.append({
                        "index": str(len(gpus)), "name": gpu_name, "vendor": "nvidia",
                        "vram_mb": 0, "driver_version": "", "type": "discrete",
                    })
                    seen_names.add(gpu_name.lower())
                except (OSError, IOError):
                    continue
        except Exception:
            pass

    # ── macOS GPUs via system_profiler ──
    if _IS_MACOS:
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-detailLevel", "mini"],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.split("\n"):
                stripped = line.strip()
                if stripped.startswith("Chipset Model:"):
                    gpu_name = stripped.split(":", 1)[1].strip()
                    gpus.append({
                        "index": str(len(gpus)), "name": gpu_name, "vendor": "apple",
                        "vram_mb": 0, "driver_version": "", "type": "integrated",
                    })
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # ── Windows GPUs via WMIC ──
    if _IS_WINDOWS:
        try:
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "Name"],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.split("\n"):
                name = line.strip()
                lower = name.lower()
                if not lower or "name" in lower:
                    continue
                if any(sn in lower for sn in seen_names):
                    continue
                vendor = "intel" if "intel" in lower else "amd" if ("amd" in lower or "radeon" in lower) else "nvidia" if "nvidia" in lower else "unknown"
                gpu_type = "integrated" if vendor == "intel" else "discrete"
                gpus.append({
                    "index": str(len(gpus)), "name": name, "vendor": vendor,
                    "vram_mb": 0, "driver_version": "", "type": gpu_type,
                })
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    logger.info("All GPUs detected: %s", [g["name"] for g in gpus])
    return gpus


def _gpu_encode_args(quality_preset: dict, export_quality: str = "1080p") -> list[str]:
    """Return FFmpeg encoder arguments based on user toggle + detected GPU.

    Critical: checks GPU_ACCELERATION_ENABLED first. If the user has the
    toggle OFF in Settings, always returns libx264 CPU args regardless of
    what GPUs are available. Only when ON does it use the detected GPU encoder.
    """
    # Respect user toggle — if OFF, always CPU
    if not app_settings.GPU_ACCELERATION_ENABLED:
        logger.info(
            "GPU ENCODE: using CPU (libx264) — GPU_ACCELERATION_ENABLED=False"
        )
        return [
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", quality_preset.get("preset", "medium"),
            "-crf", str(quality_preset.get("crf", 23)),
        ]

    gpu = detect_gpu_capabilities()
    crf = quality_preset.get("crf", 23)

    # Use HEVC for 4K exports when the GPU supports it — HEVC delivers
    # ~40% better compression at 4K than H.264 with similar quality.
    use_hevc = (
        export_quality == "4k"
        and gpu.get("hevc_encoder")
        and app_settings.GPU_HEVC_FOR_4K
    )

    # GPU device index — user preference, or auto-detected best GPU (highest VRAM)
    gpu_device = (app_settings.GPU_DEVICE_INDEX or "").strip()
    if not gpu_device and gpu.get("gpu_device_index"):
        gpu_device = str(gpu["gpu_device_index"])

    if gpu["encoder"] == "h264_nvenc":
        # NVENC supports -gpu N to select a specific NVIDIA GPU
        device_args = ["-gpu", gpu_device] if gpu_device else []
        encoder = "hevc_nvenc" if use_hevc else "h264_nvenc"
        logger.info(
            "GPU ENCODE: using %s on %s (device=%s, crf=%d, quality=%s)",
            encoder, gpu.get("gpu_name", "NVIDIA GPU"),
            gpu_device or "auto", crf, export_quality,
        )
        if use_hevc:
            return device_args + [
                "-c:v", "hevc_nvenc",
                "-preset", "p5",
                "-rc", "vbr",
                "-cq", str(max(crf - 2, 0)),  # HEVC CQ is slightly different
                "-b:v", "0",
                "-pix_fmt", "yuv420p",
                "-tag:v", "hvc1",  # Apple/browser compatibility
            ]
        return device_args + [
            "-c:v", "h264_nvenc",
            "-preset", "p5",
            "-rc", "vbr",
            "-cq", str(crf),
            "-b:v", "0",
            "-pix_fmt", "yuv420p",
        ]
    elif gpu["encoder"] == "h264_vaapi":
        logger.info(
            "GPU ENCODE: using h264_vaapi on %s (device=%s, crf=%d)",
            gpu.get("gpu_name", "GPU"), gpu.get("hwaccel_device", ""), crf,
        )
        return [
            "-vaapi_device", gpu["hwaccel_device"],
            "-c:v", "h264_vaapi",
            "-qp", str(crf),
            "-pix_fmt", "vaapi",
        ]
    elif gpu["encoder"] == "h264_qsv":
        encoder = "hevc_qsv" if use_hevc else "h264_qsv"
        logger.info(
            "GPU ENCODE: using %s on %s (crf=%d, quality=%s)",
            encoder, gpu.get("gpu_name", "Intel GPU"), crf, export_quality,
        )
        if use_hevc:
            return [
                "-c:v", "hevc_qsv",
                "-global_quality", str(max(crf - 2, 0)),
                "-preset", "medium",
                "-pix_fmt", "yuv420p",
            ]
        return [
            "-c:v", "h264_qsv",
            "-global_quality", str(crf),
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
        ]
    elif gpu["encoder"] == "h264_videotoolbox":
        vt_quality = max(1, min(100, int(100 - (crf * 3))))
        encoder = "hevc_videotoolbox" if use_hevc else "h264_videotoolbox"
        logger.info(
            "GPU ENCODE: using %s on %s (vt_quality=%d, quality=%s)",
            encoder, gpu.get("gpu_name", "Apple GPU"), vt_quality, export_quality,
        )
        if use_hevc:
            return [
                "-c:v", "hevc_videotoolbox",
                "-q:v", str(vt_quality),
                "-pix_fmt", "yuv420p",
                "-tag:v", "hvc1",
            ]
        return [
            "-c:v", "h264_videotoolbox",
            "-q:v", str(vt_quality),
            "-pix_fmt", "yuv420p",
        ]
    else:
        logger.warning(
            "GPU ENCODE: falling back to CPU (libx264) — GPU acceleration enabled "
            "but no working GPU encoder detected (gpu_name=%s, encoder=%s)",
            gpu.get("gpu_name", "unknown"), gpu.get("encoder", "none"),
        )
        return [
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", quality_preset.get("preset", "medium"),
            "-crf", str(quality_preset.get("crf", 23)),
        ]


def _gpu_decode_args() -> list[str]:
    """Return FFmpeg input-side args for hardware-accelerated decoding.

    These args must be placed BEFORE the -i input flag.  When GPU
    acceleration is disabled or no suitable decoder is available,
    returns an empty list (software decode).

    Supported paths:
    - NVIDIA CUDA/CUVID: ``-hwaccel cuda`` (D3D11VA interop on Windows)
    - Intel QSV: ``-hwaccel qsv`` (DXVA2 on Windows, VAAPI on Linux)
    - Apple VideoToolbox: ``-hwaccel videotoolbox`` (macOS M-series & Intel)
    - Windows D3D11VA: ``-hwaccel d3d11va`` (universal Windows fallback)
    - Intel/AMD VAAPI (Linux only): ``-hwaccel vaapi -hwaccel_device /dev/dri/renderD128``

    Note: When using filter chains (subtitles, crop, scale, speed) the
    decoded frames must be downloaded back to system memory for the
    filters to work.  We intentionally omit ``-hwaccel_output_format``
    for the CUDA path when filters are likely, and let FFmpeg handle the
    automatic download.  For simple encode-only paths (no complex
    filters) we keep frames on the GPU for maximum throughput.
    """
    if not app_settings.GPU_ACCELERATION_ENABLED:
        logger.info("GPU DECODE: skipped — GPU acceleration is disabled")
        return []
    if not app_settings.GPU_HWDECODE_ENABLED:
        logger.info("GPU DECODE: skipped — hardware decode is disabled (GPU_HWDECODE_ENABLED=False)")
        return []

    gpu = detect_gpu_capabilities()

    if gpu["hwaccel"] == "cuda":
        # CUDA hardware decoding — works on both Windows (DXVA2/D3D11VA
        # backed) and Linux.  We omit -hwaccel_output_format here so
        # frames are auto-downloaded to system memory for filter
        # compatibility.  The encode side (_gpu_encode_args) uses NVENC
        # which can accept system memory input efficiently.
        logger.info(
            "GPU DECODE: using CUDA hwaccel on %s",
            gpu.get("gpu_name", "NVIDIA GPU"),
        )
        return ["-hwaccel", "cuda"]
    elif gpu["hwaccel"] == "qsv":
        args = ["-hwaccel", "qsv"]
        # On Linux, QSV needs an explicit device path
        if gpu["hwaccel_device"]:
            args += ["-hwaccel_device", gpu["hwaccel_device"]]
        logger.info(
            "GPU DECODE: using QSV hwaccel (device=%s)",
            gpu.get("hwaccel_device", "default"),
        )
        return args
    elif gpu["hwaccel"] == "d3d11va":
        logger.info("GPU DECODE: using D3D11VA hwaccel")
        return ["-hwaccel", "d3d11va"]
    elif gpu["hwaccel"] == "videotoolbox":
        logger.info("GPU DECODE: using VideoToolbox hwaccel")
        return ["-hwaccel", "videotoolbox"]
    elif gpu["hwaccel"] == "vaapi" and gpu["hwaccel_device"]:
        logger.info(
            "GPU DECODE: using VAAPI hwaccel (device=%s)",
            gpu["hwaccel_device"],
        )
        return ["-hwaccel", "vaapi", "-hwaccel_device", gpu["hwaccel_device"]]

    # On Windows, try D3D11VA as a universal fallback for decoding even
    # if no GPU encoder was detected — all modern Windows GPUs (Intel,
    # NVIDIA, AMD) support DXVA2/D3D11VA for video decoding.
    if _IS_WINDOWS and gpu["vendor"] == "none":
        logger.info("GPU DECODE: using D3D11VA hwaccel (Windows fallback)")
        return ["-hwaccel", "d3d11va"]

    # On macOS, try VideoToolbox as a universal fallback — all Macs have
    # hardware decode support via VideoToolbox.
    if _IS_MACOS and gpu["vendor"] == "none":
        logger.info("GPU DECODE: using VideoToolbox hwaccel (macOS fallback)")
        return ["-hwaccel", "videotoolbox"]

    logger.info("GPU DECODE: no hardware decoder available, using CPU software decode")
    return []


def _gpu_decode_args_for_filter() -> list[str]:
    """Like _gpu_decode_args but ensures frames end up in system memory.

    Used when the FFmpeg command includes filter chains (crop, scale,
    subtitles, speed) that require software-accessible frames.
    Identical to _gpu_decode_args() — both paths already output to
    system memory — but kept as a distinct function for clarity and
    future optimization (e.g. hwdownload insertion for VAAPI).
    """
    return _gpu_decode_args()


async def _validate_export(
    output_path: str,
    expected_duration: float,
    aspect_ratio: str | None,
    subtitles_enabled: bool,
    export_quality: str = "1080p",
) -> None:
    """QA check: verify exported clip is valid and matches expected parameters.

    Hard failures (missing file, empty file, no video stream, unreadable) raise
    RuntimeError.  Soft mismatches (duration, resolution) are logged as warnings.
    """
    if not os.path.exists(output_path):
        raise RuntimeError("Export QA failed: output file does not exist")

    file_size = os.path.getsize(output_path)
    if file_size == 0:
        raise RuntimeError("Export QA failed: output file is empty")

    # Probe the output video
    probe_cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", output_path,
    ]
    proc = await asyncio.create_subprocess_exec(
        *probe_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError("Export QA failed: ffprobe could not read output file")

    probe = json.loads(stdout.decode())

    # Must have at least one video stream
    video_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
    if not video_streams:
        raise RuntimeError("Export QA failed: no video stream in output")

    # Duration validation — two-tier check:
    # 1. Warning for >5% or >0.5s drift (soft mismatch, often harmless
    #    keyframe alignment or container overhead).
    # 2. Hard failure for >20% or >5s drift, which indicates a real bug
    #    (e.g. speed not applied, -t truncation, wrong segment math).
    fmt_duration = float(probe.get("format", {}).get("duration", 0))
    dur_diff = abs(fmt_duration - expected_duration)
    dur_pct = (dur_diff / expected_duration * 100) if expected_duration > 0 else 0
    if dur_diff > 5.0 or (expected_duration > 0 and dur_pct > 20):
        raise RuntimeError(
            f"Export QA failed: duration mismatch — expected {expected_duration:.1f}s, "
            f"got {fmt_duration:.1f}s (diff={dur_diff:.1f}s, {dur_pct:.0f}%). "
            f"This likely indicates a speed or trimming bug."
        )
    elif dur_diff > 0.5 or (expected_duration > 0 and dur_pct > 5):
        logger.warning(
            "Export QA: duration drift — expected %.1fs, got %.1fs (diff=%.2fs, %.1f%%)",
            expected_duration, fmt_duration, dur_diff, dur_pct,
        )

    # Resolution check against target aspect ratio and quality
    actual_w = int(video_streams[0].get("width", 0))
    actual_h = int(video_streams[0].get("height", 0))
    if aspect_ratio and aspect_ratio in ASPECT_RATIO_DIMS:
        dims_table = ASPECT_RATIO_DIMS_BY_QUALITY.get(export_quality, ASPECT_RATIO_DIMS)
        expected_w, expected_h = dims_table.get(aspect_ratio, (1920, 1080))
        if actual_w != expected_w or actual_h != expected_h:
            logger.warning(
                "Export QA: resolution mismatch — expected %dx%d, got %dx%d (quality=%s)",
                expected_w, expected_h, actual_w, actual_h, export_quality,
            )
    else:
        # No aspect ratio — verify height matches quality tier
        expected_h = QUALITY_MAX_HEIGHT.get(export_quality, 1080)
        if actual_h != expected_h:
            logger.warning(
                "Export QA: height mismatch — expected %dp, got %dp (quality=%s)",
                expected_h, actual_h, export_quality,
            )

    # When subtitles are burned in, confirm re-encoding happened (not stream copy)
    if subtitles_enabled:
        codec = video_streams[0].get("codec_name", "")
        if codec != "h264":
            logger.warning(
                "Export QA: expected h264 codec for subtitle burn-in, got %s", codec,
            )

    # --- Social-media readiness checks ---

    # Audio stream — most social platforms require audio
    audio_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio_streams:
        logger.warning("Export QA: no audio stream — social platforms may reject this video")

    # File size — conservative limit for social platforms
    MAX_SOCIAL_SIZE_MB = 500  # Twitter=512MB, IG=250MB for Reels
    file_size_mb = file_size / (1024 * 1024)
    if file_size_mb > MAX_SOCIAL_SIZE_MB:
        logger.warning(
            "Export QA: file size %.1fMB exceeds %dMB — may be too large for some platforms",
            file_size_mb, MAX_SOCIAL_SIZE_MB,
        )

    # Pixel format — H.264 yuv420p is universally compatible
    pix_fmt = video_streams[0].get("pix_fmt", "")
    if pix_fmt and pix_fmt != "yuv420p":
        logger.warning(
            "Export QA: pixel format '%s' — yuv420p recommended for social media compatibility",
            pix_fmt,
        )

    logger.info("Export QA passed for %s (%.1fs, %.1fMB, %s, %s)", output_path,
                fmt_duration, file_size_mb,
                video_streams[0].get("codec_name", "unknown"),
                pix_fmt or "unknown")


def _parse_ass_styles(ass_content: str) -> list[dict]:
    """Parse ASS Style lines into a list of dicts keyed by field name."""
    styles = []
    field_names = [
        "Name", "Fontname", "Fontsize", "PrimaryColour", "SecondaryColour",
        "OutlineColour", "BackColour", "Bold", "Italic", "Underline", "StrikeOut",
        "ScaleX", "ScaleY", "Spacing", "Angle", "BorderStyle", "Outline",
        "Shadow", "Alignment", "MarginL", "MarginR", "MarginV", "Encoding",
    ]
    for line in ass_content.split("\n"):
        if line.startswith("Style:"):
            parts = line[len("Style:"):].strip().split(",")
            if len(parts) >= len(field_names):
                styles.append({k: v.strip() for k, v in zip(field_names, parts)})
    return styles


def _parse_ass_dialogue(ass_content: str) -> list[dict]:
    """Parse ASS Dialogue lines into a list of dicts."""
    events = []
    for line in ass_content.split("\n"):
        if line.startswith("Dialogue:"):
            # Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
            parts = line[len("Dialogue:"):].strip().split(",", 9)
            if len(parts) >= 10:
                events.append({
                    "Layer": parts[0].strip(),
                    "Start": parts[1].strip(),
                    "End": parts[2].strip(),
                    "Style": parts[3].strip(),
                    "Name": parts[4].strip(),
                    "MarginL": parts[5].strip(),
                    "MarginR": parts[6].strip(),
                    "MarginV": parts[7].strip(),
                    "Effect": parts[8].strip(),
                    "Text": parts[9],
                })
    return events


def _validate_ass_settings(
    ass_content: str,
    settings: dict,
    video_width: int,
    video_height: int,
) -> list[str]:
    """Validate that ASS subtitle content accurately reflects the input settings.

    Cross-checks every style property in the generated ASS against the expected
    values computed from the input settings dict.  This ensures the exported
    video's subtitles will match what the frontend preview displayed.

    Returns a list of mismatch descriptions.  Empty list = all checks passed.
    Raises RuntimeError for structural failures (no styles, no events).
    """
    warnings: list[str] = []

    # --- Parse ASS structure ---
    styles = _parse_ass_styles(ass_content)
    events = _parse_ass_dialogue(ass_content)

    if not styles:
        raise RuntimeError("ASS QA failed: no styles found in generated ASS")
    if not events:
        raise RuntimeError("ASS QA failed: no dialogue events found in generated ASS")

    # --- Recompute expected values from settings (mirrors ass_generator logic) ---
    font = settings.get("font", "DM Sans")
    font_size = settings.get("size", "medium")
    font_weight = settings.get("font_weight", "bold")
    font_color = settings.get("font_color", "#FFFFFF")
    position = settings.get("position", "bottom")
    background_enabled = settings.get("background_enabled", False)
    background_color = settings.get("background_color", "#000000")
    background_opacity = settings.get("background_opacity", 75)
    outline_color = settings.get("outline_color", "#000000")
    outline_opacity = settings.get("outline_opacity", 100)
    outline_width = max(0, min(10, settings.get("outline_width", 2)))
    show_speaker_labels = settings.get("show_speaker_labels", False)
    max_width_pct = max(20, min(100, settings.get("max_width", 90)))
    offset_v_pct = max(0, min(100, settings.get("offset_v", 4)))
    max_words = settings.get("max_words", 0)
    active_word_enabled = settings.get("active_word_enabled", False)
    active_word_color = settings.get("active_word_color", "#FFD700")
    active_word_outline_color = settings.get("active_word_outline_color", "#000000")

    font_scale = min(video_width, video_height) / min(ASS_REF_W, ASS_REF_H)
    # Handle both numeric (12-72) and string ("small"/"medium"/"large") sizes
    # — must mirror generate_ass() logic (ass_generator.py:197-200)
    if isinstance(font_size, (int, float)):
        base_size_px = int(font_size)
    else:
        base_size_px = ASS_FONT_SIZE_MAP.get(font_size, 30)
    # Must match generate_ass() — no compensation factor, just base * scale.
    # Use round() to match frontend Math.round() and generate_ass().
    expected_font_size = max(16, round(base_size_px * font_scale))
    _, expected_bold = _normalize_font_weight(font_weight)
    expected_alignment = 2  # Always bottom-center for absolute vertical positioning
    scaled_outline_width = max(0, round(outline_width * font_scale)) if outline_width > 0 else 0

    # Expected margins
    expected_margin_h = max(20, int(video_width * (100 - max_width_pct) / 100 / 2))
    max_margin_h = int(video_width * (1 - MIN_TEXT_AREA_W) / 2)
    expected_margin_h = min(expected_margin_h, max_margin_h)

    expected_margin_v = int(video_height * offset_v_pct / 100)

    # Expected outline/background style values
    background_radius = settings.get("background_radius", 0)
    _bg_split = bool(background_enabled and background_radius > 0)
    if background_enabled and not _bg_split:
        expected_border_style = 3
        expected_outline_colour = _hex_to_ass_color_with_alpha(background_color, background_opacity)
        expected_back_colour = "&HFF000000&"  # fully transparent — matches ass_generator fix
        expected_ol_width = max(int(4 * font_scale), 2)
        expected_shadow = 0
    elif _bg_split:
        # Background via drawing commands: base style has no box.
        expected_border_style = 1
        expected_outline_colour = "&HFF000000&"  # fully transparent
        expected_back_colour = "&HFF000000&"  # fully transparent
        expected_ol_width = 0
        expected_shadow = 0
    else:
        expected_border_style = 1
        expected_outline_colour = _hex_to_ass_color_with_alpha(outline_color, outline_opacity)
        expected_ol_width = scaled_outline_width
        # With two-layer architecture, shadow is rendered correctly on
        # Layer 0 (uniform color → continuous shadow, no per-word boxes).
        expected_shadow = max(1, min(4, round(scaled_outline_width * 0.75))) if scaled_outline_width > 0 else 0

    # --- 1. PlayRes dimensions ---
    playres_x = re.search(r"PlayResX:\s*(\d+)", ass_content)
    playres_y = re.search(r"PlayResY:\s*(\d+)", ass_content)
    if playres_x and int(playres_x.group(1)) != video_width:
        warnings.append(f"PlayResX mismatch: expected {video_width}, got {playres_x.group(1)}")
    if playres_y and int(playres_y.group(1)) != video_height:
        warnings.append(f"PlayResY mismatch: expected {video_height}, got {playres_y.group(1)}")

    # --- Validate each style ---
    # Auxiliary styles (_AWBG, _AW, _OL, AWDRAW) intentionally differ from
    # the base speaker style (different BorderStyle, Outline, Shadow, etc.)
    # and should not be validated against base style expectations.
    _AUX_SUFFIXES = ("_AWBG", "_AW", "_OL", "AWDRAW", "BGDRAW")
    for style in styles:
        style_name = style["Name"]
        if any(style_name.endswith(s) or style_name == s for s in _AUX_SUFFIXES):
            continue

        # 2. Font name
        if style["Fontname"] != font:
            warnings.append(f"Style '{style_name}': font mismatch — expected '{font}', got '{style['Fontname']}'")

        # 3. Font size
        actual_size = int(style["Fontsize"])
        if actual_size != expected_font_size:
            warnings.append(f"Style '{style_name}': font size mismatch — expected {expected_font_size}, got {actual_size}")

        # 4. Bold flag
        actual_bold = int(style["Bold"])
        if actual_bold != expected_bold:
            warnings.append(f"Style '{style_name}': bold mismatch — expected {expected_bold}, got {actual_bold}")

        # 4b. Primary font color (only when speaker colors are off — otherwise
        #     each speaker style intentionally has a different PrimaryColour)
        if not settings.get("use_speaker_colors", True):
            expected_primary = _hex_to_ass_color(font_color)
            if style["PrimaryColour"].upper() != expected_primary.upper():
                warnings.append(
                    f"Style '{style_name}': PrimaryColour mismatch — "
                    f"expected {expected_primary}, got {style['PrimaryColour']}"
                )

        # 4c. Speaker colors from override dict — when speaker colors are on
        #     and a specific color was provided for this speaker, validate it.
        if settings.get("use_speaker_colors", True):
            speaker_colors_dict = settings.get("speaker_colors", {})
            if style_name in speaker_colors_dict:
                expected_speaker_color = _hex_to_ass_color(speaker_colors_dict[style_name])
                if style["PrimaryColour"].upper() != expected_speaker_color.upper():
                    warnings.append(
                        f"Style '{style_name}': speaker color mismatch — "
                        f"expected {expected_speaker_color}, got {style['PrimaryColour']}"
                    )

        # 5. Outline color
        if style["OutlineColour"].upper() != expected_outline_colour.upper():
            warnings.append(
                f"Style '{style_name}': OutlineColour mismatch — "
                f"expected {expected_outline_colour}, got {style['OutlineColour']}"
            )

        # 6. BorderStyle
        actual_border = int(style["BorderStyle"])
        if actual_border != expected_border_style:
            warnings.append(
                f"Style '{style_name}': BorderStyle mismatch — "
                f"expected {expected_border_style}, got {actual_border}"
            )

        # 7. Outline width (box padding when background enabled)
        actual_ol = int(style["Outline"])
        if actual_ol != expected_ol_width:
            warnings.append(
                f"Style '{style_name}': Outline width mismatch — "
                f"expected {expected_ol_width}, got {actual_ol}"
            )

        # 8. Shadow
        actual_shadow = int(style["Shadow"])
        if actual_shadow != expected_shadow:
            warnings.append(
                f"Style '{style_name}': Shadow mismatch — "
                f"expected {expected_shadow}, got {actual_shadow}"
            )

        # 9. Alignment
        actual_align = int(style["Alignment"])
        if actual_align != expected_alignment:
            warnings.append(
                f"Style '{style_name}': Alignment mismatch — "
                f"expected {expected_alignment}, got {actual_align}"
            )

        # 10. Margins
        actual_ml = int(style["MarginL"])
        actual_mr = int(style["MarginR"])
        actual_mv = int(style["MarginV"])
        if actual_ml != expected_margin_h:
            warnings.append(
                f"Style '{style_name}': MarginL mismatch — "
                f"expected {expected_margin_h}, got {actual_ml}"
            )
        if actual_mr != expected_margin_h:
            warnings.append(
                f"Style '{style_name}': MarginR mismatch — "
                f"expected {expected_margin_h}, got {actual_mr}"
            )
        if actual_mv != expected_margin_v:
            warnings.append(
                f"Style '{style_name}': MarginV mismatch — "
                f"expected {expected_margin_v}, got {actual_mv}"
            )

        # 11. BackColour (when background enabled)
        if background_enabled:
            if style["BackColour"].upper() != expected_back_colour.upper():
                warnings.append(
                    f"Style '{style_name}': BackColour mismatch — "
                    f"expected {expected_back_colour}, got {style['BackColour']}"
                )

    # --- Validate dialogue events ---

    # 12. Speaker labels
    for ev in events:
        text = ev["Text"]
        # Strip inline override tags for content analysis
        plain = re.sub(r"\{[^}]*\}", "", text)
        has_label = re.match(r"^.+:\s", plain)
        if show_speaker_labels and not has_label:
            warnings.append(
                f"Speaker label missing in dialogue but show_speaker_labels=True: '{plain[:60]}'"
            )
            break  # One warning is enough
        if not show_speaker_labels and has_label:
            # Only flag if no active word (active word events don't have labels in the plain text path)
            if not active_word_enabled:
                warnings.append(
                    f"Speaker label present in dialogue but show_speaker_labels=False: '{plain[:60]}'"
                )
                break

    # 13. Active word override tags
    has_override_tags = any("\\c" in ev["Text"] for ev in events)
    if active_word_enabled and not has_override_tags:
        warnings.append("active_word_enabled=True but no inline color override tags found in events")
    if not active_word_enabled and has_override_tags:
        warnings.append("active_word_enabled=False but inline color override tags found in events")

    # 14. Active word colors (spot-check)
    if active_word_enabled and has_override_tags:
        expected_aw_color = _hex_to_ass_color(active_word_color)
        # Check that at least one event contains the expected active word color
        aw_color_found = any(expected_aw_color.upper() in ev["Text"].upper() for ev in events)
        if not aw_color_found:
            warnings.append(
                f"Active word color {expected_aw_color} not found in any dialogue event"
            )
        # Active word outline: with the minimal-tag approach, \3c is set
        # once in the bord_tag prefix (not per-word).  Verify the outline
        # color appears in the event prefix, not as per-word inline tags.
        if not background_enabled:
            expected_outline_in_prefix = _hex_to_ass_color_with_alpha(
                settings.get("outline_color", "#000000"),
                settings.get("outline_opacity", 100),
            )
            outline_in_prefix = any(
                expected_outline_in_prefix.upper() in ev["Text"].upper()
                for ev in events
            )
            if not outline_in_prefix:
                warnings.append(
                    f"Active word outline color {expected_outline_in_prefix} "
                    f"not found in any dialogue event prefix"
                )

    # 14aa. Per-word \3c/\bord/\shad conflict detection for active word mode.
    # When active word highlighting is enabled, only \c (text color) should
    # be used per-word.  Per-word \3c/\bord/\shad tags cause libass to
    # render each word as a separate segment with an independent shadow box,
    # producing visible "black bars" around words.  This applies to ALL
    # BorderStyles, not just BorderStyle=3.
    if active_word_enabled and has_override_tags:
        # Count events where \3c appears MORE than once (the bord_tag prefix
        # is allowed to set \3c once — multiple \3c means per-word overrides).
        events_with_multi_3c = sum(
            1 for ev in events if ev["Text"].count("\\3c") > 1
        )
        if events_with_multi_3c > 0:
            warnings.append(
                f"Per-word \\3c overrides detected in {events_with_multi_3c} events — "
                "this causes libass to render separate shadow boxes per word "
                "(black bars). Only \\c should vary per word."
            )
        events_with_multi_bord = sum(
            1 for ev in events if ev["Text"].count("\\bord") > 1
        )
        if events_with_multi_bord > 0:
            warnings.append(
                f"Per-word \\bord overrides detected in {events_with_multi_bord} events — "
                "this causes per-word shadow box rendering (black bars)"
            )
        events_with_multi_shad = sum(
            1 for ev in events if ev["Text"].count("\\shad") > 1
        )
        if events_with_multi_shad > 0:
            warnings.append(
                f"Per-word \\shad overrides detected in {events_with_multi_shad} events — "
                "this causes per-word shadow box rendering (black bars)"
            )

    # 14ab. Two-layer architecture validation for active word mode.
    # Layer 0 = border layer (uniform color, continuous outline/shadow)
    # Layer 1 = color layer (per-word \c overrides, \bord0\shad0)
    if active_word_enabled and events:
        layer_0_events = [ev for ev in events if ev["Layer"] == "0"]
        layer_1_events = [ev for ev in events if ev["Layer"] == "1"]

        if not layer_0_events:
            warnings.append(
                "Active word mode: no Layer 0 (border) events found — "
                "export will be missing continuous outline/shadow"
            )
        if not layer_1_events:
            warnings.append(
                "Active word mode: no Layer 1 (color) events found — "
                "export will be missing per-word color highlighting"
            )

        # Layer 0 events must NOT contain \c overrides (would break seamless border)
        for ev in layer_0_events:
            text_after_bord = ev["Text"].split("}", 1)[-1] if "}" in ev["Text"] else ev["Text"]
            if "\\c&H" in text_after_bord or "\\c" in text_after_bord.replace("\\3c", ""):
                warnings.append(
                    "Layer 0 (border) event contains \\c color overrides — "
                    "this WILL cause per-word border segmentation (black bars). "
                    "Layer 0 must use uniform color only."
                )
                break

        # Layer 1 events must suppress duplicate borders/shadows.
        # Two valid patterns:
        #   A) _AW style (BorderStyle=1): needs \bord0\shad0 to suppress outline
        #   B) _AWBG style (BorderStyle=3): needs \shad0\3a (box is intentional,
        #      \3a&HFF& makes non-active word boxes transparent)
        for ev in layer_1_events:
            has_bord0 = "\\bord0" in ev["Text"]
            has_shad0 = "\\shad0" in ev["Text"]
            has_3a_transparent = "\\3a&HFF&" in ev["Text"]
            # Pattern A: explicit \bord0 (outline suppression)
            # Pattern B: \shad0 + \3a&HFF& (AWBG box mode)
            if not has_bord0 and not (has_shad0 and has_3a_transparent):
                warnings.append(
                    "Layer 1 (color) event missing border/box suppression tags"
                )
                break

        # CRITICAL: detect events with \bord>0 AND multiple \c overrides
        # (the exact anti-pattern that causes black bars)
        # Skip box-only events (\1a&HFF& = invisible text) — border
        # segmentation is irrelevant when no text is visible.
        if active_word_enabled:
            bad_events = 0
            for ev in events:
                if "\\1a&HFF&" in ev["Text"]:
                    continue  # box-only layer, text invisible
                has_border = "\\bord" in ev["Text"] and "\\bord0" not in ev["Text"]
                color_changes = ev["Text"].count("\\c&H")
                if has_border and color_changes > 1:
                    bad_events += 1
            if bad_events > 0:
                warnings.append(
                    f"CRITICAL: {bad_events} events have \\bord>0 WITH multiple \\c color "
                    f"overrides — this causes libass per-word border segmentation (black bars). "
                    f"Must use two-layer architecture: Layer 0 = border only (no \\c), "
                    f"Layer 1 = color only (\\bord0)."
                )

    # 14b. Active word background color (\4c)
    # With BorderStyle=3, \4c sets BackColour which causes shadow artifacts
    # even with Shadow=0 — it must NEVER appear when background is enabled.
    # With BorderStyle=1, \4c is only valid when aw_bg_opacity > 0.
    aw_bg_opacity = settings.get("active_word_bg_opacity", 0)
    if active_word_enabled and has_override_tags and aw_bg_opacity > 0:
        # Active word background uses BorderStyle=3 (_AWBG style) with \3c/\3a
        # override tags to show a colored box behind only the active word.
        aw_bg_color = settings.get("active_word_bg_color", "#000000")
        expected_3c = _hex_to_ass_color(aw_bg_color)
        bg_3c_found = any(f"\\3c{expected_3c}".upper() in ev["Text"].upper() for ev in events)
        if not bg_3c_found:
            warnings.append(
                f"Active word background \\3c{expected_3c} not found in any dialogue event"
            )

    # 15. Max words
    if max_words > 0:
        for ev in events:
            text = ev["Text"]
            plain = re.sub(r"\{[^}]*\}", "", text)
            # Strip speaker label prefix if present
            if show_speaker_labels:
                plain = re.sub(r"^.+?:\s", "", plain, count=1)
            words = plain.split()
            if len(words) > max_words:
                warnings.append(
                    f"Dialogue exceeds max_words={max_words}: "
                    f"'{plain[:60]}' has {len(words)} words"
                )
                break

    # 16. Font availability — verify fontconfig can actually resolve the font
    #     so FFmpeg/libass won't silently fall back to a different font.
    try:
        result = subprocess.run(
            ["fc-match", "--format", "%{family}", font],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            matched_family = result.stdout.strip().split(",")[0].strip()
            if matched_family.lower() != font.lower():
                warnings.append(
                    f"Font '{font}' not installed — fontconfig resolved to "
                    f"'{matched_family}' instead (exported video will use wrong font)"
                )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        warnings.append(f"Could not verify font availability for '{font}' (fc-match unavailable)")

    # 17. Temporal overlap — overlapping Dialogue events with the same layer
    #     cause libass to render both simultaneously, stacking them vertically
    #     (the "bouncing subtitle" bug).
    def _ass_ts_to_secs(ts: str) -> float:
        """Parse ASS timestamp H:MM:SS.cc to seconds."""
        parts = ts.split(":")
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        return 0.0

    overlap_count = 0
    if len(events) > 1:
        # Group events by layer — cross-layer overlaps are intentional
        # (the 2-layer active-word architecture has Layer 0 base text and
        # Layer 1 per-word highlights that overlap in time by design).
        layers: dict[str, list] = {}
        for ev in events:
            layer = ev.get("Layer", "0")
            layers.setdefault(layer, []).append(ev)

        for layer_events in layers.values():
            parsed = [
                (ev, _ass_ts_to_secs(ev["Start"]), _ass_ts_to_secs(ev["End"]))
                for ev in layer_events
            ]
            parsed.sort(key=lambda x: x[1])
            for i in range(len(parsed) - 1):
                _, _, end_a = parsed[i]
                _, start_b, _ = parsed[i + 1]
                if end_a > start_b + 0.005:  # tolerance for centisecond rounding
                    overlap_count += 1
    if overlap_count > 0:
        warnings.append(
            f"Temporal overlap detected in {overlap_count} event pair(s) — "
            f"overlapping events cause subtitle stacking/bouncing"
        )

    if warnings:
        logger.warning("ASS QA found %d issue(s) for %dx%d output", len(warnings), video_width, video_height)
    else:
        logger.info("ASS QA passed: all %d styles and %d events verified", len(styles), len(events))

    return warnings


def _validate_text_overlay_parity(
    text_overlays: list[dict],
    video_width: int,
    video_height: int,
) -> list[str]:
    """Validate text overlay properties for preview-export parity.

    Checks that text overlay values sent by the frontend will produce
    reasonable FFmpeg drawtext output matching the preview.

    Returns list of warning strings. Empty = all checks passed.
    """
    warnings: list[str] = []

    for i, ov in enumerate(text_overlays):
        label = f"Text overlay {i+1}"
        text = ov.get("text", "")
        if not text:
            warnings.append(f"{label}: empty text")
            continue

        # Position validation
        x = ov.get("x", 50)
        y = ov.get("y", 50)
        if not (0 <= x <= 100):
            warnings.append(f"{label}: x={x}% outside [0,100]")
        if not (0 <= y <= 100):
            warnings.append(f"{label}: y={y}% outside [0,100]")

        # Font size validation
        font_size = ov.get("font_size", 48)
        min_reasonable = max(8, video_height * 0.01)
        max_reasonable = video_height * 0.3
        if font_size < min_reasonable:
            warnings.append(f"{label}: font_size={font_size}px may be too small for {video_height}p output")
        if font_size > max_reasonable:
            warnings.append(f"{label}: font_size={font_size}px may be too large for {video_height}p output")

        # Color validation
        font_color = ov.get("font_color", "#FFFFFF")
        if font_color and not font_color.startswith("#"):
            warnings.append(f"{label}: font_color='{font_color}' not hex format")

        # Outline sanity
        ol_width = ov.get("outline_width", 0)
        if ol_width > 0:
            ol_color = ov.get("outline_color", "")
            if not ol_color:
                warnings.append(f"{label}: outline_width={ol_width} but no outline_color")

        # FFmpeg drawtext limitations
        bg_radius = ov.get("bg_radius", 0)
        if bg_radius > 0:
            warnings.append(
                f"{label}: bg_radius={bg_radius} — FFmpeg drawtext only supports "
                f"rectangular backgrounds (rounded corners not supported)"
            )

        shadow_blur = ov.get("shadow_blur", 0)
        if shadow_blur > 0:
            warnings.append(
                f"{label}: shadow_blur={shadow_blur} — FFmpeg drawtext does not "
                f"support shadow blur (only offset + color)"
            )

        # Timing validation
        start_t = ov.get("start_time", 0)
        end_t = ov.get("end_time", 0)
        if end_t <= start_t:
            warnings.append(f"{label}: end_time ({end_t}) <= start_time ({start_t})")

    return warnings


def _validate_subject_tracking(
    filter_chain: str | None,
    aspect_ratio: str | None,
    src_w: int,
    src_h: int,
    subject_x: int,
    subject_keyframes: list[tuple[float, int]] | None,
) -> list[str]:
    """Validate that subject tracking is correctly applied in the FFmpeg filter chain.

    Cross-checks the crop filter in the filter chain against the expected
    behaviour given the aspect ratio, source dimensions, and subject data.

    Returns a list of issue descriptions.  Empty list = all checks passed.
    """
    warnings: list[str] = []

    has_aspect = aspect_ratio and aspect_ratio in ASPECT_RATIO_VALUES
    has_dynamic = (
        subject_keyframes
        and len(subject_keyframes) > 1
        and len(set(kf[1] for kf in subject_keyframes)) > 1
    )
    has_static_kf = subject_keyframes and not has_dynamic

    if not has_aspect:
        # No aspect ratio → no crop → no subject tracking to validate
        if filter_chain and "crop=" in filter_chain:
            warnings.append("crop filter present but no aspect ratio specified")
        return warnings

    # Compute expected crop dimensions
    target_ratio = ASPECT_RATIO_VALUES[aspect_ratio]
    src_ratio = src_w / src_h if src_h else 1

    if abs(src_ratio - target_ratio) <= 0.01:
        # Same aspect ratio → no crop needed, skip validation
        return warnings

    if not filter_chain:
        warnings.append(f"aspect_ratio={aspect_ratio} but no filter chain produced")
        return warnings

    # Verify crop filter exists
    if "crop=" not in filter_chain:
        warnings.append(f"aspect_ratio={aspect_ratio} but no crop filter in chain")
        return warnings

    if target_ratio < src_ratio:
        crop_h = src_h
        crop_w = int(src_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(src_w / target_ratio)
    crop_w = crop_w - (crop_w % 2)
    crop_h = crop_h - (crop_h % 2)
    max_x_offset = src_w - crop_w

    # Parse the crop filter from the chain.
    # For dynamic expressions, the x parameter can contain nested if()/clip()
    # with escaped commas (\,), so we can't use a simple split.
    # Strategy: match crop=W:H: then grab everything until the last :Y before
    # the next comma-separated filter or end of string.
    crop_match = re.search(r"crop=(\d+):(\d+):(.+)", filter_chain)
    if not crop_match:
        warnings.append("Could not parse crop filter parameters from filter chain")
        return warnings

    actual_crop_w = int(crop_match.group(1))
    actual_crop_h = int(crop_match.group(2))
    remainder = crop_match.group(3)

    # For static crops: "123:0,scale=..." → x_param="123", y from "0"
    # For dynamic crops: "clip(if(...)\\,0\\,1314):0,scale=..." → x_param="clip(...)", y from "0"
    # Split on the LAST unescaped colon before the y-offset
    # The y-offset is always a plain integer followed by comma or end-of-string
    y_match = re.search(r":(\d+)(?:,|$)", remainder)
    if y_match:
        actual_y = int(y_match.group(1))
        x_param = remainder[:y_match.start()]
    else:
        # Fallback: try simple static parse
        static_match = re.match(r"(\d+):(\d+)", remainder)
        if static_match:
            x_param = static_match.group(1)
            actual_y = int(static_match.group(2))
        else:
            warnings.append("Could not parse crop x/y parameters from filter chain")
            return warnings

    # Validate crop dimensions
    if actual_crop_w != crop_w:
        warnings.append(
            f"crop width mismatch: expected {crop_w}, got {actual_crop_w}"
        )
    if actual_crop_h != crop_h:
        warnings.append(
            f"crop height mismatch: expected {crop_h}, got {actual_crop_h}"
        )

    # Validate subject tracking type (dynamic vs static)
    is_dynamic_expr = "if(lt(t" in x_param or "clip(" in x_param

    if has_dynamic:
        # Expect dynamic expression
        if not is_dynamic_expr:
            warnings.append(
                f"Dynamic keyframes provided ({len(subject_keyframes)} kfs, "
                f"{len(set(kf[1] for kf in subject_keyframes))} unique) "
                f"but crop uses static x={x_param}"
            )
        else:
            # Validate the expression is clipped to max_offset
            if f"\\,{max_x_offset})" in x_param or f",{max_x_offset})" in x_param:
                pass  # Good
            elif f"\\,0\\,{max_x_offset}" in x_param:
                pass  # clip(expr, 0, max_offset) format
            else:
                # Try to find max_offset in expression
                if str(max_x_offset) not in x_param:
                    warnings.append(
                        f"Dynamic crop expression doesn't reference max_offset={max_x_offset}"
                    )
    else:
        # Expect static offset
        if is_dynamic_expr:
            warnings.append(
                "Static subject tracking expected but dynamic expression found in crop"
            )
        else:
            # Validate the static offset is correctly computed
            try:
                actual_x = int(x_param)
            except ValueError:
                warnings.append(f"Expected static integer x offset, got '{x_param}'")
                return warnings

            # Compute expected offset
            if has_static_kf:
                sx = _safe_subject_x(subject_keyframes[0][1])
            else:
                sx = _safe_subject_x(subject_x)
            expected_x = _center_crop_offset(sx, src_w, crop_w)

            if actual_x != expected_x:
                warnings.append(
                    f"Static crop x offset mismatch: expected {expected_x} "
                    f"(from subject_x={sx}), got {actual_x}"
                )

    # Validate subject is actually centered — for static crops, verify the
    # subject pixel position falls within the crop window
    if not is_dynamic_expr:
        try:
            actual_x = int(x_param)
            if has_static_kf:
                sx = _safe_subject_x(subject_keyframes[0][1])
            else:
                sx = _safe_subject_x(subject_x)
            subject_pixel = src_w * sx / 100
            crop_left = actual_x
            crop_right = actual_x + crop_w
            if subject_pixel < crop_left or subject_pixel > crop_right:
                warnings.append(
                    f"Subject at pixel {subject_pixel:.0f} is outside crop window "
                    f"[{crop_left}, {crop_right}]"
                )
            elif actual_x > 0 and actual_x < max_x_offset:
                # Only check centering when the crop isn't edge-clamped.
                # At the frame edges, perfect centering is impossible.
                crop_center = crop_left + crop_w / 2
                center_offset_pct = abs(subject_pixel - crop_center) / crop_w * 100
                if center_offset_pct > 10:
                    warnings.append(
                        f"Subject/face at pixel {subject_pixel:.0f} is {center_offset_pct:.0f}% "
                        f"off-center in crop window [{crop_left}, {crop_right}] "
                        f"(target: <10% for face centering)"
                    )
        except ValueError:
            pass  # Already reported above

    # Dynamic keyframe bounds verification — ensure every keyframe
    # produces a valid crop offset that covers the full output frame
    # (no off-screen crops, no black bars)
    if has_dynamic and subject_keyframes:
        safe_lo, safe_hi = _compute_safe_range(src_ratio, target_ratio)
        out_of_bounds = 0
        off_center_count = 0
        for t, sx_val in subject_keyframes:
            # Check keyframe values are within safe range
            if sx_val < safe_lo or sx_val > safe_hi:
                out_of_bounds += 1
            # Verify the resulting crop offset is valid
            offset = _center_crop_offset(_safe_subject_x(sx_val), src_w, crop_w)
            if offset < 0:
                warnings.append(
                    f"Keyframe t={t:.2f}s sx={sx_val}: crop offset {offset} is negative (video would crop off-screen left)"
                )
            if offset > max_x_offset:
                warnings.append(
                    f"Keyframe t={t:.2f}s sx={sx_val}: crop offset {offset} > max {max_x_offset} (video would crop off-screen right)"
                )
            # Check the crop fills the full output width
            actual_crop_end = offset + crop_w
            if actual_crop_end > src_w + 1:  # +1 for rounding tolerance
                warnings.append(
                    f"Keyframe t={t:.2f}s sx={sx_val}: crop extends beyond source frame ({actual_crop_end} > {src_w})"
                )
            # Check centering quality (when not edge-clamped)
            if offset > 0 and offset < max_x_offset:
                subject_pixel = src_w * _safe_subject_x(sx_val) / 100
                subject_in_crop = subject_pixel - offset
                center_error_pct = abs(subject_in_crop - crop_w / 2) / crop_w * 100
                if center_error_pct > 15:
                    off_center_count += 1

        if out_of_bounds > 0:
            warnings.append(
                f"{out_of_bounds}/{len(subject_keyframes)} keyframes outside safe range "
                f"[{safe_lo}, {safe_hi}] — subject may be edge-clamped"
            )
        if off_center_count > 0:
            warnings.append(
                f"{off_center_count}/{len(subject_keyframes)} keyframes >15% off-center — "
                f"tracking may not look human-edited"
            )

        # Verify smoothstep interpolation at intermediate points
        # Sample at midpoints between consecutive keyframes to ensure
        # intermediate values don't exceed bounds
        for i in range(len(subject_keyframes) - 1):
            t0, sx0 = subject_keyframes[i]
            t1, sx1 = subject_keyframes[i + 1]
            dt = t1 - t0
            if dt <= 0:
                continue
            # Sample at 25%, 50%, 75% through the segment
            for frac in (0.25, 0.5, 0.75):
                # Smoothstep: f(p) = p^2 * (3 - 2p)
                p = frac
                eased = p * p * (3 - 2 * p)
                interp_sx = sx0 + (sx1 - sx0) * eased
                interp_offset = _center_crop_offset(
                    _safe_subject_x(int(round(interp_sx))), src_w, crop_w
                )
                if interp_offset < 0 or interp_offset > max_x_offset:
                    warnings.append(
                        f"Smoothstep interpolation at t={t0 + dt * frac:.2f}s produces "
                        f"offset={interp_offset} outside [0, {max_x_offset}] — video may crop off-screen"
                    )
                    break  # One warning per segment is enough

    if warnings:
        logger.warning(
            "Subject tracking QA found %d issue(s) for %s crop (%dx%d → %dx%d)",
            len(warnings), "dynamic" if has_dynamic else "static",
            src_w, src_h, crop_w, crop_h,
        )
    else:
        logger.info(
            "Subject tracking QA passed: %s crop verified for %s (%dx%d → %dx%d)",
            "dynamic" if has_dynamic else "static",
            aspect_ratio, src_w, src_h, crop_w, crop_h,
        )

    return warnings


def _verify_centering_math(
    clip_id: int | str,
    aspect_ratio: str,
    src_w: int,
    src_h: int,
    subject_x: int,
    subject_keyframes: list[tuple[float, int]] | None,
) -> None:
    """Log detailed centering verification for every keyframe (or static value).

    For each subject_x value, compute:
    - The pixel position of the subject in the source frame
    - The crop window offset
    - The pixel position of the subject within the crop (should be center)
    - The centering error in pixels and as a percentage of crop width

    This runs during every export and logs at INFO level so the container
    logs always show exactly how well subjects are centered.
    """
    target_ratio = ASPECT_RATIO_VALUES.get(aspect_ratio)
    if not target_ratio:
        return

    src_ratio = src_w / src_h if src_h else 1
    if abs(src_ratio - target_ratio) <= 0.01:
        return

    # Compute crop dimensions
    if target_ratio < src_ratio:
        crop_h = src_h
        crop_w = int(src_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(src_w / target_ratio)
    crop_w = crop_w - (crop_w % 2)
    crop_h = crop_h - (crop_h % 2)

    def _check_sx(sx_val: int, label: str) -> None:
        sx_safe = _safe_subject_x(sx_val)
        subject_pixel = src_w * sx_safe / 100
        x_offset = _center_crop_offset(sx_safe, src_w, crop_w)
        # Where the subject lands within the crop window
        subject_in_crop = subject_pixel - x_offset
        crop_center = crop_w / 2
        error_px = abs(subject_in_crop - crop_center)
        error_pct = error_px / crop_w * 100 if crop_w > 0 else 0

        is_edge_clamped = x_offset == 0 or x_offset == (src_w - crop_w)
        if error_pct < 1.0:
            status = "FACE-CENTERED"
        elif error_pct < 5.0:
            status = "NEAR-CENTER"
        elif is_edge_clamped:
            status = "EDGE-CLAMPED"
        else:
            status = "OFF-CENTER"

        logger.info(
            "[SubjectTracking] VERIFY %s clip %s: sx=%d → safe_sx=%d, "
            "subject@%.0fpx, crop_offset=%d, subject_in_crop=%.0f/%.0f (center=%.0f), "
            "error=%.1fpx (%.2f%%) → %s",
            label, clip_id, sx_val, sx_safe,
            subject_pixel, x_offset, subject_in_crop, crop_w, crop_center,
            error_px, error_pct, status,
        )

    if subject_keyframes and len(subject_keyframes) > 1:
        unique_sx = set(kf[1] for kf in subject_keyframes)
        if len(unique_sx) > 1:
            for i, (t, sx) in enumerate(subject_keyframes):
                _check_sx(sx, f"kf[{i}] t={t:.2f}s")
            return

    # Static — verify the single value
    sx_val = subject_keyframes[0][1] if subject_keyframes else subject_x
    _check_sx(sx_val, "static")

    # Also verify the frontend-equivalent objectPosition math
    R = src_ratio / target_ratio
    if R > 1.01:
        sx_safe = _safe_subject_x(sx_val)
        center_pct = (R * sx_safe - 50) / (R - 1)
        center_pct = max(0, min(100, center_pct))
        logger.info(
            "[SubjectTracking] VERIFY frontend-equivalent clip %s: "
            "R=%.3f, sx=%d → objectPosition=%.2f%% (50%%=perfect center)",
            clip_id, R, sx_safe, center_pct,
        )

    logger.info(
        "═══════════════════════════════════════════════════════════════════════════",
    )


# ---------------------------------------------------------------------------
# Export quality presets
# ---------------------------------------------------------------------------

QUALITY_PRESETS = {
    "720p":  {"crf": 20, "preset": "fast"},
    "1080p": {"crf": 18, "preset": "fast"},
    "4k":    {"crf": 18, "preset": "fast"},
}

# Standard output resolutions by aspect ratio, keyed by quality tier
ASPECT_RATIO_DIMS_BY_QUALITY = {
    "720p": {
        "16:9": (1280, 720),
        "9:16": (720, 1280),
        "1:1":  (720, 720),
        "4:5":  (720, 900),
    },
    "1080p": {
        "16:9": (1920, 1080),
        "9:16": (1080, 1920),
        "1:1":  (1080, 1080),
        "4:5":  (1080, 1350),
    },
    "4k": {
        "16:9": (3840, 2160),
        "9:16": (2160, 3840),
        "1:1":  (2160, 2160),
        "4:5":  (2160, 2700),
    },
}

# Default (1080p) — used when no quality specified
ASPECT_RATIO_DIMS = ASPECT_RATIO_DIMS_BY_QUALITY["1080p"]

# Max output height per quality tier (for scaling without aspect ratio change)
QUALITY_MAX_HEIGHT = {
    "720p": 720,
    "1080p": 1080,
    "4k": 2160,
}

ASPECT_RATIO_VALUES = {
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "1:1": 1.0,
    "4:5": 4 / 5,
}

def _compute_video_out_dims(
    src_w: int, src_h: int,
    aspect_ratio: str | None,
    export_quality: str,
) -> tuple[int, int]:
    """Compute the output video dimensions after crop/scale.

    Returns (out_w, out_h) matching what _build_filter_chain produces.
    """
    dims_table = ASPECT_RATIO_DIMS_BY_QUALITY.get(export_quality, ASPECT_RATIO_DIMS)
    if aspect_ratio and aspect_ratio in dims_table:
        out_w, out_h = dims_table[aspect_ratio]
        return out_w - (out_w % 2), out_h - (out_h % 2)
    target_h = QUALITY_MAX_HEIGHT.get(export_quality, 1080)
    if src_h != target_h:
        out_h = target_h
        out_w = round(src_w * target_h / src_h / 2) * 2 if src_h > 0 else src_w
        return out_w, out_h
    return src_w, src_h


def _compute_safe_range(src_ratio: float, target_ratio: float, edge_buffer: int = 3) -> tuple[int, int]:
    """Compute safe subject_x range for a given aspect ratio conversion.

    Ensures that any subject_x within this range will produce a non-clamped
    objectPosition value — meaning the subject can actually be centered.

    Matches frontend computeSafeRange() exactly for preview-export parity.
    """
    R = src_ratio / target_ratio
    if R <= 1.01:
        return (5, 95)
    # Scale edge buffer with magnification ratio so high-R crops (16:9→9:16)
    # don't push faces to the frame edge. A face is ~10% of source width;
    # at R≈3.16 the crop window is ~31% of source, so we need more margin.
    # Use gentler scaling (2x instead of 3x) to avoid over-clamping edge
    # speakers in 2-person podcast layouts (e.g. faces at x=20% and x=80%).
    scaled_buffer = min(15, round(edge_buffer + (R - 1) * 2))
    sx_at_min = (scaled_buffer * (R - 1) + 50) / R
    sx_at_max = ((100 - scaled_buffer) * (R - 1) + 50) / R
    return (
        max(5, math.ceil(sx_at_min)),
        min(95, math.floor(sx_at_max)),
    )


def _safe_subject_x(sx: int, margin: int = 10, src_ratio: float = 0, target_ratio: float = 0) -> int:
    """Clamp subject_x to safe range, dynamically if aspect ratios provided.

    When src_ratio and target_ratio are provided, computes the safe range
    based on the actual aspect ratio conversion. Otherwise falls back to
    static margin.

    Matches frontend safeSubjectX() exactly for preview-export parity.
    """
    if src_ratio > 0 and target_ratio > 0:
        lo, hi = _compute_safe_range(src_ratio, target_ratio)
        return max(lo, min(hi, round(sx)))
    return max(margin, min(100 - margin, round(sx)))


def _center_crop_offset(sx: int, src_w: int, crop_w: int) -> int:
    """Compute horizontal crop offset that centers the subject in the output.

    Places the crop window so that the subject (at sx% of src_w) ends up
    at the center of the cropped frame.

    Args:
        sx: Subject x position as 0-100 percentage of source width.
        src_w: Source video width in pixels.
        crop_w: Crop window width in pixels.

    Returns:
        Clamped pixel offset for the crop x parameter.
    """
    subject_pixel = src_w * sx / 100
    x_offset = round(subject_pixel - crop_w / 2)
    max_offset = src_w - crop_w
    return max(0, min(max_offset, x_offset))


def _verify_face_centering(
    sx: int, src_w: int, crop_w: int,
    label: str = "",
) -> int:
    """Verify the face is well-centered in the crop and adjust if needed.

    Computes where the subject (at sx% of source) ends up in the crop.
    If the face is more than 10% off-center, adjusts the offset to
    improve centering while staying within bounds.

    Returns the adjusted crop x_offset.
    """
    x_offset = _center_crop_offset(sx, src_w, crop_w)
    subject_pixel = src_w * sx / 100
    if crop_w <= 0:
        return x_offset
    face_in_crop_pct = (subject_pixel - x_offset) / crop_w * 100
    off_center = face_in_crop_pct - 50

    if abs(off_center) > 10:
        # Face is significantly off-center — try to improve
        ideal_offset = round(subject_pixel - crop_w / 2)
        max_offset = src_w - crop_w
        adjusted = max(0, min(max_offset, ideal_offset))

        new_face_pct = (subject_pixel - adjusted) / crop_w * 100
        new_off = new_face_pct - 50

        if abs(new_off) < abs(off_center):
            logger.info(
                "[SubjectTracking] Face centering fix%s: sx=%d, was %.1f%% in crop "
                "(%.1f%% off), now %.1f%% (%.1f%% off), offset %d→%d",
                f" ({label})" if label else "",
                sx, face_in_crop_pct, off_center, new_face_pct, new_off,
                x_offset, adjusted,
            )
            return adjusted

    return x_offset


def _build_speaker_position_map(scenes: list, transcript: list | None) -> dict[str, int]:
    """Correlate scene positions with transcript speaker labels.
    Matches frontend buildSpeakerPositionMap() exactly."""
    if not scenes or not transcript:
        return {}
    speaker_x_values: dict[str, list[int]] = {}
    for scene in scenes:
        ts = float(scene.timestamp if hasattr(scene, "timestamp") else scene.get("timestamp", 0))
        sx = int(scene.subject_x if hasattr(scene, "subject_x") else scene.get("subject_x", 50))
        active_seg = None
        for seg in transcript:
            seg_start = float(seg.get("start", seg.get("start_time", 0)))
            seg_end = float(seg.get("end", seg.get("end_time", 0)))
            if ts >= seg_start - 0.5 and ts <= seg_end + 0.5:
                active_seg = seg
                break
        if active_seg and active_seg.get("speaker"):
            speaker = active_seg["speaker"]
            if speaker not in speaker_x_values:
                speaker_x_values[speaker] = []
            speaker_x_values[speaker].append(sx)
    return {sp: round(sum(vs) / len(vs)) for sp, vs in speaker_x_values.items() if vs}


def _build_speaker_keyframes(
    transcript: list, speaker_map: dict[str, int],
    clip_start: float, clip_end: float,
    src_ratio: float = 0, target_ratio: float = 0,
) -> list[tuple[float, int]] | None:
    """Build dense keyframes at every speaker change.
    Matches frontend buildSpeakerKeyframes() exactly."""
    if not transcript or len(speaker_map) < 2:
        return None
    clip_dur = clip_end - clip_start
    if clip_dur <= 0:
        return None
    overlapping = sorted(
        [s for s in transcript
         if float(s.get("end", s.get("end_time", 0))) > clip_start
         and float(s.get("start", s.get("start_time", 0))) < clip_end],
        key=lambda s: float(s.get("start", s.get("start_time", 0))),
    )
    if not overlapping:
        return None
    keyframes = []
    last_speaker = None
    for seg in overlapping:
        seg_start = max(clip_start, float(seg.get("start", seg.get("start_time", 0))))
        speaker = seg.get("speaker")
        if not speaker or speaker not in speaker_map:
            continue
        if speaker == last_speaker:
            continue
        sx = _safe_subject_x(speaker_map[speaker], src_ratio=src_ratio, target_ratio=target_ratio)
        keyframes.append((max(0.0, seg_start - clip_start), sx))
        last_speaker = speaker
    if not keyframes:
        return None
    if keyframes[0][0] > 0:
        keyframes.insert(0, (0.0, keyframes[0][1]))
    if keyframes[-1][0] < clip_dur:
        keyframes.append((clip_dur, keyframes[-1][1]))
    return keyframes if len(keyframes) >= 2 else None


def _detect_position_clusters(
    keyframes: list[tuple[float, int]],
    gap_threshold: int = 10,
    min_cluster_size: int = 2,
) -> list[dict] | None:
    """Detect N position clusters using recursive gap splitting.

    Strips center noise zone (47-53) when it prevents detection.
    Matches frontend detectPositionClusters() exactly for preview-export parity.
    """
    if len(keyframes) < 4:
        return None

    xs = [kf[1] for kf in keyframes]

    def _split(values):
        if len(values) < min_cluster_size * 2:
            return [values]
        sv = sorted(values)
        mg = 0
        si = -1
        for i in range(1, len(sv)):
            g = sv[i] - sv[i - 1]
            if g > mg:
                mg = g
                si = i
        if mg < gap_threshold or si < 0:
            return [values]
        left = sv[:si]
        right = sv[si:]
        if len(left) < min_cluster_size or len(right) < min_cluster_size:
            return [values]
        return _split(left) + _split(right)

    def _median(arr):
        s = sorted(arr)
        mid = len(s) // 2
        return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2

    def _build_result(clusters):
        if len(clusters) < 2:
            return None
        result = []
        for v in clusters:
            sv = sorted(v)
            trim = max(1, len(sv) // 10)
            trimmed = sv[trim:len(sv) - trim] if len(sv) > 2 else sv
            center = round(sum(trimmed) / len(trimmed)) if trimmed else sv[len(sv) // 2]
            result.append({"center": center, "count": len(v)})
        result.sort(key=lambda c: c["center"])
        for i in range(1, len(result)):
            if result[i]["center"] - result[i - 1]["center"] < 8:
                return None
        # Midpoint cluster rejection — matches frontend exactly
        if len(result) >= 3:
            for i in range(len(result) - 2, 0, -1):
                mid = (result[i - 1]["center"] + result[i + 1]["center"]) / 2
                span = result[i + 1]["center"] - result[i - 1]["center"]
                if (abs(result[i]["center"] - mid) < span * 0.3 and
                        result[i]["count"] < max(result[i - 1]["count"], result[i + 1]["count"])):
                    result.pop(i)
            if len(result) < 2:
                return None
        return result

    # ── Run ALL passes and pick the best result ──
    # Pass 1 might find a midpoint phantom cluster that has MORE samples
    # than real speaker clusters (from Haar cascade merged detections).
    # Center stripping in Pass 2/3 removes those values and often produces
    # a cleaner 2-cluster result that should be preferred.

    result1 = _build_result(_split(xs))

    # Pass 2: strip center noise [47, 53]
    CENTER_LO, CENTER_HI = 47, 53
    non_center = [x for x in xs if x < CENTER_LO or x > CENTER_HI]
    center_count = len(xs) - len(non_center)
    result2 = None
    if (center_count > len(xs) * 0.10
            and len(non_center) >= min_cluster_size * 2
            and any(x < CENTER_LO for x in non_center)
            and any(x > CENTER_HI for x in non_center)):
        result2 = _build_result(_split(non_center))

    # Pass 3: aggressive strip [44, 56]
    WIDE_LO, WIDE_HI = 44, 56
    far = [x for x in xs if x < WIDE_LO or x > WIDE_HI]
    wide_count = len(xs) - len(far)
    result3 = None
    if (wide_count > len(xs) * 0.15
            and len(far) >= min_cluster_size * 2
            and any(x < WIDE_LO for x in far)
            and any(x > WIDE_HI for x in far)):
        result3 = _build_result(_split(far))

    # Pick the best result: prefer fewer clusters (cleaner tracking)
    candidates = [r for r in (result1, result2, result3) if r]
    if not candidates:
        return None

    candidates.sort(key=lambda r: (len(r), [result1, result2, result3].index(r)))
    return candidates[0]


def _snap_to_clusters(
    keyframes: list[tuple[float, int]],
    clusters: list[dict],
) -> list[tuple[float, int]]:
    """Snap each keyframe to nearest cluster center.

    When equidistant, prefers the cluster with more samples.
    Matches frontend snapToClusters() exactly for preview-export parity.
    """
    result = []
    for t, sx in keyframes:
        best = clusters[0]
        best_dist = abs(sx - best["center"])
        for c in clusters[1:]:
            d = abs(sx - c["center"])
            if d < best_dist or (d == best_dist and c["count"] > best["count"]):
                best = c
                best_dist = d
        result.append((t, best["center"]))
    return result


def _validate_tracking(
    keyframes: list[tuple[float, int]],
    clusters: list[dict] | None,
    clip_duration: float,
) -> list[tuple[float, int]]:
    """QA validation: fix extended center holds and missing instant-cut markers.

    Matches frontend validateTracking() exactly for preview-export parity.
    """
    if not keyframes:
        return [(0.0, 50)]
    fixed = list(keyframes)

    # Check 1: no extended center holds when multi-position data exists
    if clusters and len(clusters) >= 2:
        for i in range(len(fixed) - 1):
            hold = fixed[i + 1][0] - fixed[i][0]
            if 47 <= fixed[i][1] <= 53 and hold > 3.0:
                prev = fixed[i - 1][1] if i > 0 else None
                if prev is not None and any(c["center"] == prev for c in clusters):
                    fixed[i] = (fixed[i][0], prev)

    # Check 2: large position changes must have instant-cut markers
    i = 0
    while i < len(fixed) - 1:
        delta = abs(fixed[i + 1][1] - fixed[i][1])
        dt = fixed[i + 1][0] - fixed[i][0]
        if delta > 15 and dt > 0.01:
            cut_time = round(fixed[i + 1][0] - 0.001, 3)
            if cut_time > fixed[i][0]:
                fixed.insert(i + 1, (cut_time, fixed[i][1]))
                i += 1
        i += 1

    return fixed


def _speaker_aware_keyframes(
    face_results: list,
    clip_start: float,
    transcript: list = None,
    face_registry=None,
    active_speaker_events: list = None,
) -> list[tuple[float, int]]:
    """Build keyframes by tracking the SPEAKING face, not the largest face.

    For each dense frame:
    1. If active_speaker_events exist -> use the face matching the active slot
    2. Elif transcript has speech at this time -> pick face with highest lip aperture
    3. Else -> fall back to largest face (primary_face_idx)
    """
    from backend.services.active_speaker import get_active_slot_at_time

    keyframes = []
    face_ys = []
    face_widths = []

    for fd in face_results:
        if not fd.faces:
            continue
        rel_t = fd.timestamp - clip_start
        abs_t = fd.timestamp
        chosen_face = None

        # Method 1: Active speaker events (most accurate)
        if active_speaker_events:
            slot_id = get_active_slot_at_time(active_speaker_events, abs_t)
            if slot_id >= 0:
                for f in fd.faces:
                    if f.identity_id == slot_id:
                        chosen_face = f
                        break
                if not chosen_face and face_registry:
                    slot = face_registry.slot_by_id(slot_id)
                    if slot:
                        chosen_face = min(fd.faces,
                            key=lambda f: abs(f.nose_x - slot.x_center))

        # Method 2: Highest lip aperture during speech
        if not chosen_face and transcript:
            is_speech = any(
                (seg.start if hasattr(seg, 'start') else seg.get('start', 0)) <= abs_t <=
                (seg.end if hasattr(seg, 'end') else seg.get('end', 0))
                for seg in transcript
            )
            if is_speech:
                speaking_faces = [f for f in fd.faces if f.lip_aperture > 0.02]
                if speaking_faces:
                    chosen_face = max(speaking_faces, key=lambda f: f.lip_aperture)

        # Method 3: Largest face (existing behavior)
        if not chosen_face and fd.primary_face_idx >= 0:
            chosen_face = fd.faces[fd.primary_face_idx]

        if chosen_face:
            # Use actual face position for crop centering.
            # The pipeline's cluster-snap logic determines WHICH speaker to track;
            # here we use the precise face nose_x for WHERE to center the crop.
            # This matches the frontend's precise_x approach for preview-export parity.
            sx = int(round(chosen_face.nose_x))

            keyframes.append((rel_t, sx))
            face_ys.append(float(chosen_face.nose_y))
            face_widths.append(float(chosen_face.width))

    return keyframes, face_ys, face_widths


def _insert_snap_transitions(
    keyframes: list[tuple[float, int]],
    jump_threshold: int = 15,
    snap_duration: float = 0.15,
) -> list[tuple[float, int]]:
    """Insert synthetic keyframes to convert long interpolations into hold-then-snap.

    Without this, two keyframes at t=0,sx=30 and t=10,sx=70 produce a
    10-second slow pan. With this, they become:
        t=0.000, sx=30  (hold)
        t=4.925, sx=30  (hold ends)
        t=5.075, sx=70  (snap complete - 150ms transition)
        t=10.00, sx=70  (hold)
    """
    if len(keyframes) <= 1:
        return keyframes

    result = [keyframes[0]]
    for i in range(1, len(keyframes)):
        t0, sx0 = result[-1]
        t1, sx1 = keyframes[i]
        dt = t1 - t0
        jump = abs(sx1 - sx0)

        if jump >= jump_threshold and dt > snap_duration * 4:
            mid_t = (t0 + t1) / 2
            half_snap = snap_duration / 2
            result.append((mid_t - half_snap, sx0))
            result.append((mid_t + half_snap, sx1))
        result.append(keyframes[i])

    return result


def _compute_face_y_offset(
    face_y_center_pct: float,
    face_height_pct: float,
    src_h: int,
    crop_h: int,
    target_face_position: float = 0.38,
) -> int:
    """Compute vertical crop offset to position face with proper headroom.

    Places the face center at target_face_position (default 0.38 = upper third)
    following broadcast framing conventions (rule of thirds).
    """
    if crop_h >= src_h:
        return 0
    max_y_offset = src_h - crop_h
    face_center_px = src_h * face_y_center_pct / 100
    target_face_in_crop = crop_h * target_face_position
    y_offset = int(face_center_px - target_face_in_crop)
    return max(0, min(max_y_offset, y_offset))


def _compute_zoom_factor(
    avg_face_width_pct: float,
    target_face_pct: float = 15.0,
    min_zoom: float = 0.85,
    max_zoom: float = 1.15,
) -> float:
    """Compute zoom factor to keep face at consistent apparent size.

    Static per-clip, not per-frame (per-frame zoom causes breathing effect).
    """
    if avg_face_width_pct <= 0:
        return 1.0
    ratio = target_face_pct / max(avg_face_width_pct, 1)
    return round(max(min_zoom, min(max_zoom, ratio)), 3)

def _dense_face_detection_for_clip(
    video_path: str,
    clip_start: float,
    clip_end: float,
    sample_rate: float = 2.0,
) -> list[tuple[float, int]]:
    """Extract dense frames for a clip and run face detection.

    For a 30s clip at 2s intervals, this gives 15 face samples
    instead of the 2-3 from the sparse full-video analysis.
    Runs on CPU in ~1-2 seconds.

    Returns list of (clip_relative_time, subject_x) keyframes from face data.
    """
    import tempfile

    duration = clip_end - clip_start
    if duration <= 0:
        return []

    num_frames = min(30, max(5, int(duration / sample_rate)))

    try:
        from backend.services.face_detector import detect_faces_batch
    except ImportError:
        logger.debug("Face detector unavailable for dense clip detection")
        return []

    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract frames just for this clip at lower resolution
        cmd = [
            "ffmpeg", "-y", "-threads", "2",
            "-ss", str(clip_start),
            "-t", str(duration),
            "-i", video_path,
            "-vf", (
                f"fps=1/{sample_rate},"
                "scale='min(640,iw)':'min(360,ih)'"
                ":force_original_aspect_ratio=decrease"
            ),
            "-vsync", "vfr", "-q:v", "15",
            os.path.join(tmpdir, "clip_frame_%04d.jpg"),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=30)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

        # Build (timestamp, path) list
        frame_files = sorted(
            f for f in os.listdir(tmpdir) if f.startswith("clip_frame_")
        )
        if not frame_files:
            return []

        frame_paths = []
        for i, fname in enumerate(frame_files):
            ts = clip_start + i * sample_rate
            frame_paths.append((ts, os.path.join(tmpdir, fname)))

        # Run face detection (CPU, ~0.5-1s)
        face_results = detect_faces_batch(frame_paths, min_confidence=0.4)

        # Convert to keyframes using primary face position
        keyframes = []
        for fd in face_results:
            if fd.faces and fd.primary_face_idx >= 0:
                face = fd.faces[fd.primary_face_idx]
                rel_t = fd.timestamp - clip_start
                keyframes.append((rel_t, round(face.nose_x)))

        logger.info(
            "Dense clip face detection: %d/%d frames with faces (%.1fs clip, %.1fs intervals)",
            len(keyframes), len(frame_paths), duration, sample_rate,
        )
        return keyframes


def _keyframes_from_cached_render_plan(
    cached_plan: dict,
    clip_start: float,
    clip_end: float,
) -> list[tuple[float, int]] | None:
    """Thin wrapper around ``render_plan_keyframes.keyframes_from_cached_render_plan``.

    Kept as a module-level alias so the call sites in ``export_clip``
    stay unchanged and the test suite can import the helper without
    pulling in the rest of ``clip_exporter``'s heavyweight dependency
    graph (cv2 / MediaPipe / ffmpeg / pydantic_settings).
    """
    from backend.services.render_plan_keyframes import (
        keyframes_from_cached_render_plan,
    )
    return keyframes_from_cached_render_plan(
        cached_plan, clip_start=clip_start, clip_end=clip_end,
    )


def _extract_render_plan_segments(scenes: list):
    """Extract lightweight segment objects from SceneDescription data for the RenderPlan builder.

    Returns a list of objects with the attributes the RenderPlan builder expects,
    or None if the scenes don't contain reframe metadata.
    """
    if not scenes:
        return None

    segments = []
    has_reframe = False

    for i, scene in enumerate(scenes):
        ts = float(scene.timestamp if hasattr(scene, "timestamp") else scene.get("timestamp", 0))
        sx = int(scene.subject_x if hasattr(scene, "subject_x") else scene.get("subject_x", 50))
        sy = int(getattr(scene, "subject_y", 40) if hasattr(scene, "subject_y") else scene.get("subject_y", 40) if isinstance(scene, dict) else 40)
        layout = scene.layout_mode if hasattr(scene, "layout_mode") else scene.get("layout_mode", "single")
        desc = scene.description if hasattr(scene, "description") else scene.get("description", "")

        strategy = "stationary"
        reason = "hold"
        ease_in_ms = 0
        if desc.startswith("[reframe:"):
            has_reframe = True
            parts = desc.strip("[]").split(":")
            if len(parts) >= 2:
                reason = parts[1]
            if len(parts) >= 3:
                try:
                    ease_in_ms = int(parts[2])
                except ValueError:
                    pass
            if len(parts) >= 4:
                strategy = parts[3]

        # Compute end time from next scene
        next_ts = None
        if i + 1 < len(scenes):
            ns = scenes[i + 1]
            next_ts = float(ns.timestamp if hasattr(ns, "timestamp") else ns.get("timestamp", 0))
        if next_ts is None or next_ts <= ts:
            next_ts = ts + 5.0

        seg = type("_Seg", (), {
            "start": ts, "end": next_ts, "subject_x": sx, "subject_y": sy,
            "layout": layout, "strategy": strategy, "reason": reason,
            "ease_in_ms": ease_in_ms, "content_type": "unknown",
            "motion_path": None, "hard_constraints": None,
            "active_slot": None, "confidence": 1.0,
            "lead_room_direction": None,
        })()
        segments.append(seg)

    return segments if has_reframe and segments else None


def _extract_solver_keyframes(
    layout_timeline_data: list,
    clip_start: float,
    clip_end: float,
) -> list[tuple[float, int]] | None:
    """Extract solver-generated keyframes from layout_timeline_data.

    When the camera solver has produced per-shot keyframes, they are stored
    in the LayoutSegment's face_positions list as dicts with a 'solver_mode'
    key. This function finds those and converts them to (relative_time,
    subject_x_0_100) tuples for direct use in the FFmpeg crop pipeline.

    Returns None if no solver keyframes found.
    """
    solver_kf = []
    for seg_data in (layout_timeline_data or []):
        fps = seg_data if isinstance(seg_data, dict) else {}
        face_positions = fps.get("face_positions", [])
        if not face_positions:
            continue
        # Check if this segment has solver keyframes
        if not any(fp.get("solver_mode") for fp in face_positions):
            continue
        seg_start = fps.get("start", 0)
        seg_end = fps.get("end", 0)
        # Only include segments that overlap with this clip
        if seg_end <= clip_start or seg_start >= clip_end:
            continue
        for fp in face_positions:
            if not fp.get("solver_mode"):
                continue
            abs_t = float(fp.get("timestamp", 0))
            if abs_t < clip_start or abs_t > clip_end:
                continue
            rel_t = round(abs_t - clip_start, 3)
            sx = int(round(float(fp.get("x", 50))))
            sx = max(5, min(95, sx))  # basic safety clamp
            solver_kf.append((rel_t, sx))

    if not solver_kf:
        return None

    solver_kf.sort()
    # Ensure coverage at t=0 and t=end
    if solver_kf[0][0] > 0.01:
        solver_kf.insert(0, (0.0, solver_kf[0][1]))
    clip_dur = clip_end - clip_start
    if solver_kf[-1][0] < clip_dur - 0.01:
        solver_kf.append((round(clip_dur, 3), solver_kf[-1][1]))

    return solver_kf


def _build_subject_keyframes(
    scenes: list,
    clip_start: float,
    clip_end: float,
    src_ratio: float = 0,
    target_ratio: float = 0,
) -> list[tuple[float, int]]:
    """Build sorted (relative_time, subject_x) keyframes from scenes for a clip.

    Accepts scenes both within and outside the clip range.  Scenes outside
    the clip boundaries are used to interpolate accurate subject_x values
    at the clip start/end rather than falling back to center (50).  This
    ensures short clips between scene timestamps still get proper tracking.

    When src_ratio and target_ratio are provided, uses dynamic safe margin
    based on the aspect ratio conversion instead of the static margin.

    Returns at least one keyframe.  If no scenes are provided, returns [(0.0, 50)].
    """
    clip_dur = max(0.0, clip_end - clip_start)

    # Extract and sort all scene data (including scenes outside clip range)
    all_data = []
    for s in scenes:
        ts = float(s.timestamp if hasattr(s, "timestamp") else s.get("timestamp", 0))
        # Prefer active_speaker_x (when AI detected who is talking) over generic subject_x
        asx = s.active_speaker_x if hasattr(s, "active_speaker_x") else s.get("active_speaker_x")
        sx = s.subject_x if hasattr(s, "subject_x") else s.get("subject_x", 50)
        raw_sx = int(asx if asx is not None else sx)
        safe_sx = _safe_subject_x(raw_sx, src_ratio=src_ratio, target_ratio=target_ratio)
        all_data.append((ts, safe_sx))

    if not all_data:
        logger.info("[SubjectTracking] _build_subject_keyframes: no scene data — returning default center (50)")
        return [(0.0, 50)]

    all_data.sort(key=lambda k: k[0])

    logger.info(
        "[SubjectTracking] _build_subject_keyframes: clip=%.1f-%.1f (%.1fs), %d scenes, "
        "timestamps=[%.1f..%.1f], subject_x values=%s",
        clip_start, clip_end, clip_dur, len(all_data),
        all_data[0][0], all_data[-1][0],
        [sx for _, sx in all_data],
    )

    # Separate into before, within, and after clip boundaries
    before = [(ts, sx) for ts, sx in all_data if ts < clip_start]
    within = [(ts, sx) for ts, sx in all_data if clip_start <= ts <= clip_end]
    after = [(ts, sx) for ts, sx in all_data if ts > clip_end]

    logger.info(
        "[SubjectTracking] _build_subject_keyframes: %d before, %d within, %d after clip range",
        len(before), len(within), len(after),
    )

    # Build keyframes from within-range scenes (convert to relative time)
    raw = [(round(ts - clip_start, 3), sx) for ts, sx in within]

    def _interp(t_abs, s1, s2):
        """Linearly interpolate subject_x at absolute time t between two scenes."""
        t1, sx1 = s1
        t2, sx2 = s2
        dt = t2 - t1
        if dt <= 0:
            return sx1
        frac = min(1.0, max(0.0, (t_abs - t1) / dt))
        result = _safe_subject_x(int(round(sx1 + (sx2 - sx1) * frac)), src_ratio=src_ratio, target_ratio=target_ratio)
        logger.debug(
            "[SubjectTracking] interpolate t=%.2f between (%.1f,sx=%d) and (%.1f,sx=%d): frac=%.3f → sx=%d",
            t_abs, t1, sx1, t2, sx2, frac, result,
        )
        return result

    # Compute accurate boundary value at t=0 (clip_start)
    if not raw or raw[0][0] > 0.0:
        if before and within:
            sx0 = _interp(clip_start, before[-1], within[0])
            logger.info("[SubjectTracking] boundary t=0: interpolated from before→within, sx=%d", sx0)
        elif before and after and not within:
            sx0 = _interp(clip_start, before[-1], after[0])
            logger.info("[SubjectTracking] boundary t=0: interpolated from before→after (no within), sx=%d", sx0)
        elif before:
            sx0 = before[-1][1]
            logger.info("[SubjectTracking] boundary t=0: using last before scene, sx=%d", sx0)
        elif within:
            sx0 = within[0][1]
            logger.info("[SubjectTracking] boundary t=0: using first within scene, sx=%d", sx0)
        elif after:
            sx0 = after[0][1]
            logger.info("[SubjectTracking] boundary t=0: using first after scene, sx=%d", sx0)
        else:
            sx0 = 50
            logger.info("[SubjectTracking] boundary t=0: no scenes available, using default center (50)")
        raw.insert(0, (0.0, sx0))

    # Compute accurate boundary value at t=clip_dur (clip_end)
    if clip_dur > 0 and (not raw or raw[-1][0] < clip_dur):
        if after and within:
            sx_end = _interp(clip_end, within[-1], after[0])
            logger.info("[SubjectTracking] boundary t=%.1f: interpolated from within→after, sx=%d", clip_dur, sx_end)
        elif before and after and not within:
            sx_end = _interp(clip_end, before[-1], after[0])
            logger.info("[SubjectTracking] boundary t=%.1f: interpolated from before→after (no within), sx=%d", clip_dur, sx_end)
        elif after:
            sx_end = after[0][1]
            logger.info("[SubjectTracking] boundary t=%.1f: using first after scene, sx=%d", clip_dur, sx_end)
        elif within:
            sx_end = within[-1][1]
            logger.info("[SubjectTracking] boundary t=%.1f: using last within scene, sx=%d", clip_dur, sx_end)
        elif before:
            sx_end = before[-1][1]
            logger.info("[SubjectTracking] boundary t=%.1f: using last before scene, sx=%d", clip_dur, sx_end)
        else:
            sx_end = 50
            logger.info("[SubjectTracking] boundary t=%.1f: no scenes, using default center (50)", clip_dur)
        raw.append((clip_dur, sx_end))

    # Sort by time (boundary insertions should be in order but ensure it)
    raw.sort(key=lambda k: k[0])

    # ── Wide-shot fallback for distant 2-speaker setups ──
    # If exactly 2 speakers are too far apart for either to fit in the crop
    # window, center between them so both are partially visible.
    # For 3+ speakers (panel shows), DON'T apply — let the tracking follow
    # the active speaker and snap between positions.
    if len(raw) >= 4 and src_ratio > 0 and target_ratio > 0:
        all_sx = [sx for _, sx in raw]
        sx_min, sx_max = min(all_sx), max(all_sx)
        sx_range = sx_max - sx_min
        unique_positions = len(set(round(sx / 10) * 10 for sx in all_sx))  # Count ~10% clusters

        R = src_ratio / target_ratio if target_ratio > 0 else 1
        crop_coverage = 100.0 / R if R > 1 else 100.0

        if sx_range > crop_coverage * 0.7 and unique_positions <= 2:
            midpoint = int((sx_min + sx_max) / 2)
            safe_mid = _safe_subject_x(midpoint, src_ratio=src_ratio, target_ratio=target_ratio)
            logger.info(
                "[SubjectTracking] Wide-shot fallback: 2 speakers at sx=%d and sx=%d "
                "(range=%d > %.0f%% of crop coverage %.0f%%). Centering at %d",
                sx_min, sx_max, sx_range, 70, crop_coverage, safe_mid,
            )
            raw = [(t, safe_mid) for t, _ in raw]
        elif sx_range > crop_coverage * 0.7 and unique_positions > 2:
            logger.info(
                "[SubjectTracking] Multi-speaker panel (%d positions, range=%d) — "
                "tracking active speaker, NOT applying wide-shot fallback",
                unique_positions, sx_range,
            )

    logger.info(
        "[SubjectTracking] _build_subject_keyframes result: %d keyframes — %s",
        len(raw), [(f"t={t:.2f}s,sx={sx}") for t, sx in raw],
    )

    return raw if raw else [(0.0, 50)]


def _smooth_keyframes(
    keyframes: list[tuple[float, int]],
    max_speed: float = 50.0,
) -> list[tuple[float, int]]:
    """Smooth keyframes to limit maximum subject_x change rate.

    max_speed: maximum subject_x units per second (e.g. 50 = 50% per second).
    Performs a forward pass clamping each keyframe so the change from the
    previous one doesn't exceed max_speed * dt.

    Returns a new list of smoothed keyframes.

    DEPRECATED: Use _smooth_keyframes_bidirectional() for human-feeling movement.
    Kept for reference only.
    """
    if len(keyframes) <= 1:
        return list(keyframes)

    smoothed = [keyframes[0]]
    clamped_count = 0
    for i in range(1, len(keyframes)):
        t_prev, sx_prev = smoothed[-1]
        t_cur, sx_cur = keyframes[i]
        dt = t_cur - t_prev
        if dt <= 0:
            smoothed.append((t_cur, sx_prev))
            continue
        max_delta = max_speed * dt
        delta = sx_cur - sx_prev
        if abs(delta) > max_delta:
            original_sx = sx_cur
            sx_cur = int(sx_prev + max_delta * (1 if delta > 0 else -1))
            sx_cur = max(0, min(100, sx_cur))
            clamped_count += 1
            logger.debug(
                "[SubjectTracking] _smooth_keyframes: clamped kf[%d] t=%.2f sx %d→%d (delta=%d, maxDelta=%.1f)",
                i, t_cur, original_sx, sx_cur, delta, max_delta,
            )
        smoothed.append((t_cur, sx_cur))

    if clamped_count > 0:
        logger.info(
            "[SubjectTracking] _smooth_keyframes: %d/%d keyframes clamped (max_speed=%.0f/s)",
            clamped_count, len(keyframes) - 1, max_speed,
        )

    return smoothed


def _handle_scene_cuts(
    keyframes: list[tuple[float, int]],
    jump_threshold: int = 15,
) -> list[tuple[float, int]]:
    """Insert instant-jump keyframes at likely scene cuts.

    When subject_x changes by more than jump_threshold between consecutive
    keyframes, this is likely a scene cut — the subject didn't physically
    move, the camera cut to a new shot.  Human editors cut-to instantly,
    they never pan across a scene cut.

    Inserts a keyframe 1ms before the cut with the OLD position, so the
    transition is truly instant — below one frame at any display rate.

    Matches frontend handleSceneCuts() exactly for preview-export parity.
    """
    if len(keyframes) <= 1:
        return list(keyframes)

    result = [keyframes[0]]
    for i in range(1, len(keyframes)):
        t_prev, sx_prev = result[-1]
        t_cur, sx_cur = keyframes[i]
        delta = abs(sx_cur - sx_prev)

        if delta >= jump_threshold and (t_cur - t_prev) > 0.1:
            # Large jump detected — insert instant cut
            # 1ms gap: below one frame at any frame rate, so smoothstep can't catch it
            cut_time = round(t_cur - 0.001, 3)
            if cut_time > t_prev:
                result.append((cut_time, sx_prev))  # Hold old position until cut
                logger.debug(
                    "[SubjectTracking] _handle_scene_cuts: instant cut at t=%.3f (delta=%d, threshold=%d)",
                    t_cur, delta, jump_threshold,
                )

        result.append((t_cur, sx_cur))

    cuts_inserted = len(result) - len(keyframes)
    if cuts_inserted > 0:
        logger.info(
            "[SubjectTracking] _handle_scene_cuts: %d instant-cut keyframes inserted (threshold=%d)",
            cuts_inserted, jump_threshold,
        )

    return result


def _inject_shot_boundary_cuts(
    keyframes: list[tuple[float, int]],
    scene_cut_timestamps: list[float] | None,
    clip_start: float,
    clip_end: float,
) -> list[tuple[float, int]]:
    """Force instant cuts at camera shot boundaries.

    Scene cuts (camera angle changes) should ALWAYS trigger instant reframe,
    regardless of how much subject_x changed. This matches how professional
    editors work — they cut-to instantly, never pan across a camera cut.
    """
    if not scene_cut_timestamps or len(keyframes) <= 1:
        return list(keyframes)

    clip_dur = clip_end - clip_start
    # Convert to clip-relative time, filter to within clip bounds
    clip_cuts = sorted(
        t - clip_start
        for t in scene_cut_timestamps
        if clip_start + 0.1 < t < clip_end - 0.1
    )
    if not clip_cuts:
        return list(keyframes)

    result = list(keyframes)
    inserted = 0
    for cut_time in clip_cuts:
        # Find the keyframe just before and after this cut
        insert_idx = len(result)
        for i in range(len(result)):
            if result[i][0] >= cut_time:
                insert_idx = i
                break

        before_sx = result[insert_idx - 1][1] if insert_idx > 0 else result[0][1]
        after_sx = result[insert_idx][1] if insert_idx < len(result) else before_sx

        # If positions differ by more than 3 units, inject instant cut
        if abs(after_sx - before_sx) > 3:
            hold_time = round(cut_time - 0.001, 3)
            prev_t = result[insert_idx - 1][0] if insert_idx > 0 else 0
            if hold_time > prev_t:
                result.insert(insert_idx, (hold_time, before_sx))
                result.insert(insert_idx + 1, (cut_time, after_sx))
                inserted += 2

    if inserted > 0:
        logger.info(
            "[SubjectTracking] _inject_shot_boundary_cuts: %d instant-cut keyframes "
            "from %d scene boundaries", inserted, len(clip_cuts),
        )

    return result


def _apply_dead_zone(
    keyframes: list[tuple[float, int]],
    threshold: int = 5,
    src_ratio: float = 0,
    target_ratio: float = 0,
) -> list[tuple[float, int]]:
    """Hysteresis dead zone — higher threshold to start, lower to stop.

    Matches frontend applyDeadZone() exactly for preview-export parity.
    """
    if len(keyframes) <= 1:
        return list(keyframes)

    VISIBLE_THRESHOLD = 6
    effective_threshold = threshold
    if src_ratio > 0 and target_ratio > 0:
        R = src_ratio / target_ratio
        if R > 1.01:
            effective_threshold = max(3, round(VISIBLE_THRESHOLD * (R - 1) / R))

    engage_threshold = effective_threshold
    disengage_threshold = max(1, round(effective_threshold * 0.4))

    result = [keyframes[0]]
    anchor = keyframes[0][1]
    is_tracking = False
    snapped_count = 0

    for i in range(1, len(keyframes)):
        t, sx = keyframes[i]
        drift_from_anchor = abs(sx - anchor)

        if not is_tracking:
            if drift_from_anchor >= engage_threshold:
                is_tracking = True
                result.append((t, sx))
                anchor = sx
            else:
                result.append((t, anchor))
                snapped_count += 1
        else:
            if drift_from_anchor <= disengage_threshold:
                is_tracking = False
                result.append((t, anchor))
                snapped_count += 1
            else:
                result.append((t, sx))
                anchor = sx

    if snapped_count > 0:
        logger.info(
            "[SubjectTracking] _apply_dead_zone: %d/%d keyframes snapped (hysteresis engage=%d, disengage=%d)",
            snapped_count, len(keyframes) - 1, engage_threshold, disengage_threshold,
        )

    return result


def _smooth_keyframes_bidirectional(
    keyframes: list[tuple[float, int]],
    max_speed: float = 22.0,
    src_ratio: float = 0,
    target_ratio: float = 0,
) -> list[tuple[float, int]]:
    """Hold-then-snap smoother for human-edited camera feel.

    Holds camera COMPLETELY STILL until the subject drifts far enough to
    warrant a reframe, then snaps FAST (200-400ms) with ease-out curve.
    Pattern: HOLD -> SNAP -> HOLD -> SNAP (never continuous drift).

    Matches frontend smoothKeyframesBidirectional() exactly for preview-export parity.
    """
    if len(keyframes) <= 1:
        return list(keyframes)

    # Scale parameters for aspect ratio magnification
    reframe_threshold = 8
    reframe_duration = 0.30
    effective_max_speed = max_speed

    if src_ratio > 0 and target_ratio > 0:
        R = src_ratio / target_ratio
        if R > 1.5:
            reframe_threshold = max(3, round(8 / math.sqrt(R)))
            reframe_duration = max(0.15, 0.30 / math.sqrt(R))
            effective_max_speed = min(80, max_speed * math.sqrt(R))

    dt_step = 0.016
    result = [keyframes[0]]
    hold_pos = float(keyframes[0][1])
    pos = float(keyframes[0][1])
    reframing = False
    reframe_target = float(keyframes[0][1])
    reframe_progress = 0.0
    reframe_start_pos = float(keyframes[0][1])
    last_direction = 0

    for i in range(1, len(keyframes)):
        target = float(keyframes[i][1])
        seg_dt = keyframes[i][0] - keyframes[i - 1][0]

        if seg_dt <= 0.002:
            # Scene cut — instant snap
            pos = target
            hold_pos = target
            reframing = False
            reframe_progress = 0.0
            last_direction = 0
            result.append((keyframes[i][0], pos))
            continue

        # Track direction for hold-on-reversal logic
        new_direction = 1 if target > hold_pos else (-1 if target < hold_pos else 0)

        # Decide whether to start a reframe
        drift_from_hold = abs(target - hold_pos)
        if not reframing and drift_from_hold >= reframe_threshold:
            reframing = True
            reframe_target = target
            reframe_start_pos = pos
            reframe_progress = 0.0
        elif reframing:
            # Already reframing — update target if same direction
            if new_direction != 0 and last_direction != 0 and new_direction != last_direction:
                pass  # Direction reversed mid-reframe — finish current, don't chase
            else:
                reframe_target = target

        if new_direction != 0:
            last_direction = new_direction

        # Simulate movement
        sim_time = 0.0
        while sim_time < seg_dt:
            step = min(dt_step, seg_dt - sim_time)

            if reframing:
                reframe_progress += step / reframe_duration

                if reframe_progress >= 1.0:
                    pos = reframe_target
                    hold_pos = reframe_target
                    reframing = False
                    reframe_progress = 0.0
                else:
                    # Ease-out: 1 - (1-t)^3
                    eased = 1.0 - (1.0 - min(1.0, reframe_progress)) ** 3
                    pos = reframe_start_pos + (reframe_target - reframe_start_pos) * eased

            sim_time += step

        pos = max(0.0, min(100.0, pos))
        result.append((keyframes[i][0], pos))

    logger.info(
        "[SubjectTracking] _smooth_keyframes_bidirectional: %d keyframes processed (hold-then-snap, reframe_threshold=%d, reframe_duration=%.2fs)",
        len(keyframes), reframe_threshold, reframe_duration,
    )

    return result


def _merge_holds(
    keyframes: list[tuple[float, int]],
    tolerance: int = 3,
) -> list[tuple[float, int]]:
    """Merge consecutive keyframes with similar values into holds.

    If several consecutive keyframes are within tolerance of each other,
    snap them all to the first value — creating a visible 'rest' period
    where the crop holds steady.

    Matches frontend mergeHolds() exactly for preview-export parity.
    """
    if len(keyframes) <= 1:
        return list(keyframes)

    result = [keyframes[0]]
    merged_count = 0
    for i in range(1, len(keyframes)):
        t, sx = keyframes[i]
        _, prev_sx = result[-1]
        if abs(sx - prev_sx) <= tolerance:
            result.append((t, prev_sx))  # Hold at previous position
            merged_count += 1
        else:
            result.append((t, sx))

    if merged_count > 0:
        logger.info(
            "[SubjectTracking] _merge_holds: %d/%d keyframes merged into holds (tolerance=%d)",
            merged_count, len(keyframes) - 1, tolerance,
        )

    return result


def _compress_range(
    keyframes: list[tuple[float, int]],
    max_range: int = 30,
    src_ratio: float = 0,
    target_ratio: float = 0,
) -> list[tuple[float, int]]:
    """Compress the range of subject_x values to prevent erratic swinging.

    If the full range of sx values exceeds max_range, compress toward the
    median so total motion stays within bounds.  Preserves relative timing
    and direction of motion — just reduces amplitude.

    When src_ratio and target_ratio are provided, scales max_range up for
    extreme aspect ratio conversions where the visible crop window is much
    narrower than the source frame.

    Matches frontend compressRange() exactly for preview-export parity.
    """
    if len(keyframes) <= 1:
        return list(keyframes)

    # Scale max_range based on aspect ratio magnification
    effective_max_range = max_range
    if src_ratio > 0 and target_ratio > 0:
        R = src_ratio / target_ratio
        if R > 1.01:
            safe_lo, safe_hi = _compute_safe_range(src_ratio, target_ratio)
            effective_max_range = max(max_range, safe_hi - safe_lo)

    xs = [kf[1] for kf in keyframes]
    min_x = min(xs)
    max_x = max(xs)
    current_range = max_x - min_x

    if current_range <= effective_max_range:
        return list(keyframes)

    # Compress toward median
    sorted_xs = sorted(xs)
    median = sorted_xs[len(sorted_xs) // 2]
    scale = effective_max_range / current_range

    result = []
    for t, sx in keyframes:
        new_sx = max(0, min(100, median + (sx - median) * scale))
        result.append((t, new_sx))

    logger.info(
        "[SubjectTracking] _compress_range: range %d→%d (effective_max=%d, base=%d), median=%d, compressed %d keyframes",
        current_range, effective_max_range, effective_max_range, max_range, median, len(keyframes),
    )

    return result


def _build_crop_x_expr(
    keyframes: list[tuple[float, int]],
    max_offset: int,
    src_w: int = 0,
    crop_w: int = 0,
    step_mode: bool = False,
) -> str:
    """Build an FFmpeg expression for time-varying horizontal crop offset.

    Two modes:
    - step_mode=False (default): Piecewise smoothstep (3p²-2p³) interpolation.
    - step_mode=True: Step function — instant snap at each keyframe, matching
      the frontend's interpolateSubjectX() hold-until-next behavior exactly.

    Each keyframe's subject_x is converted to a centering offset so the
    subject ends up at the horizontal center of the cropped frame.

    If all keyframes share the same subject_x (or there's only one), returns
    a plain integer string for a static crop — no expression overhead.

    Args:
        keyframes: sorted list of (time_seconds, subject_x_0_to_100)
        max_offset: maximum x_offset in pixels (src_w - crop_w)
        src_w: source video width in pixels (for centering calculation)
        crop_w: crop window width in pixels (for centering calculation)
        step_mode: if True, use instant-snap (step) instead of smoothstep

    Returns:
        FFmpeg expression string for the x parameter of the crop filter.
    """
    if max_offset <= 0:
        return "0"

    # Compute aspect ratio for safe clamping (closure over outer variables)
    _expr_src_ratio = src_w / crop_w if crop_w > 0 else 0
    # The target ratio for the crop window is 1.0 (crop_w is already target-sized)
    # but we need the original source-to-target ratio for _safe_subject_x.
    # Infer it: crop_w = src_h * target_ratio → target_ratio = crop_w / src_h
    # src_ratio = src_w / src_h
    # However, we don't have src_h here. Since the keyframes are already
    # clamped by the pipeline, just pass the values through with centering.
    def _sx_to_offset(sx: int) -> int:
        """Convert subject_x to a centering crop offset.

        In step_mode (frontend keyframes), use _center_crop_offset directly —
        this is mathematically equivalent to the CSS objectPosition formula:
          objectPosition% = (R*sx - 50) / (R - 1)
          crop_x = objectPosition% / 100 * max_offset
        which simplifies to: sx * src_w / 100 - crop_w / 2

        Without step_mode (backend keyframes), use _verify_face_centering
        which may adjust the offset for better centering.
        """
        if src_w > 0 and crop_w > 0:
            if step_mode:
                # Exact CSS parity — no adjustment
                return _center_crop_offset(sx, src_w, crop_w)
            return _verify_face_centering(sx, src_w, crop_w)
        # Fallback to proportional if dimensions not provided
        return max(0, min(max_offset, int(max_offset * sx / 100)))

    # Check if all keyframes have the same value → static
    unique_sx = set(kf[1] for kf in keyframes)
    if len(unique_sx) <= 1:
        sx = keyframes[0][1] if keyframes else 50
        offset = _sx_to_offset(sx)
        logger.info(
            "[SubjectTracking] _build_crop_x_expr: static — all keyframes sx=%d → offset=%d (src_w=%d, crop_w=%d, max_offset=%d)",
            sx, offset, src_w, crop_w, max_offset,
        )
        return str(offset)

    offsets = [(t, _sx_to_offset(sx)) for t, sx in keyframes]

    # Collapse consecutive identical offsets — many keyframes may map to
    # the same pixel offset (e.g., cluster-snapped positions).  Keeping
    # only the first of each run halves expression nesting in practice.
    collapsed: list[tuple[float, int]] = [offsets[0]]
    for i in range(1, len(offsets)):
        if offsets[i][1] != collapsed[-1][1]:
            collapsed.append(offsets[i])
    if collapsed[-1] != offsets[-1]:
        collapsed.append(offsets[-1])  # ensure final time is present
    if len(collapsed) < len(offsets):
        logger.info(
            "[SubjectTracking] _build_crop_x_expr: collapsed %d → %d offsets (removed consecutive duplicates)",
            len(offsets), len(collapsed),
        )
    offsets = collapsed

    # FFmpeg nested if() expressions have a depth limit (~100-200 depending
    # on build).  If we still have too many offsets, downsample to stay safe.
    MAX_EXPR_DEPTH = 80
    if len(offsets) > MAX_EXPR_DEPTH:
        # Keep first, last, and evenly spaced keyframes
        step = max(1, (len(offsets) - 2) // (MAX_EXPR_DEPTH - 2))
        sampled = [offsets[0]]
        for i in range(step, len(offsets) - 1, step):
            sampled.append(offsets[i])
        sampled.append(offsets[-1])
        logger.warning(
            "[SubjectTracking] _build_crop_x_expr: downsampled %d → %d offsets (FFmpeg expression depth limit)",
            len(offsets), len(sampled),
        )
        offsets = sampled

    # Force step mode for high keyframe counts — smoothstep triples nesting
    # depth (hold + transition + hold per segment) which can exceed limits
    if not step_mode and len(offsets) > 40:
        step_mode = True
        logger.info(
            "[SubjectTracking] _build_crop_x_expr: forcing step mode (%d offsets — smoothstep would exceed nesting limit)",
            len(offsets),
        )

    interp_label = "step" if step_mode else "smoothstep"
    logger.info(
        "[SubjectTracking] _build_crop_x_expr: dynamic (%s) — %d keyframes, offsets=%s (src_w=%d, crop_w=%d, max_offset=%d)",
        interp_label, len(offsets),
        [(f"t={t:.2f}→{off}px") for t, off in offsets[:20]],
        src_w, crop_w, max_offset,
    )

    # Build nested if(lt(t,...), segment, ...) from last to first
    # Final fallback: last offset
    expr = str(offsets[-1][1])

    if step_mode:
        # Step function: hold each offset until the next keyframe (matches
        # frontend interpolateSubjectX() exactly — instant snap, no easing)
        for i in range(len(offsets) - 2, -1, -1):
            t1 = offsets[i + 1][0]
            off0 = offsets[i][1]
            expr = f"if(lt(t\\,{t1:.3f})\\,{off0}\\,{expr})"
    else:
        # Smoothstep: piecewise cubic Hermite interpolation between keyframes
        for i in range(len(offsets) - 2, -1, -1):
            t0, off0 = offsets[i]
            t1, off1 = offsets[i + 1]
            dt = t1 - t0
            if dt <= 0 or off0 == off1:
                segment = str(off0)
            else:
                d_off = off1 - off0
                p_expr = f"(t-{t0:.3f})/{dt:.3f}"
                segment = f"{off0}+{d_off}*st(0\\,{p_expr})*ld(0)*(3-2*ld(0))"
            expr = f"if(lt(t\\,{t1:.3f})\\,{segment}\\,{expr})"

    # Clamp to valid range
    final_expr = f"clip({expr}\\,0\\,{max_offset})"
    logger.debug("[SubjectTracking] _build_crop_x_expr: FFmpeg expression=%s", final_expr)
    return final_expr


CUSTOM_FONTS_DIR = "/data/fonts"


def _extract_force_style_from_ass(ass_content: str) -> str:
    """Extract outline/border settings from ASS content for force_style override.

    Parses the first Style line and builds a force_style string that
    guarantees outline rendering via FFmpeg's subtitles filter.

    The ASS v4+ Style format fields (0-indexed after 'Style:'):
      0:Name, 1:Fontname, 2:Fontsize, 3:PrimaryColour, 4:SecondaryColour,
      5:OutlineColour, 6:BackColour, 7:Bold, ... 15:BorderStyle,
      16:Outline, 17:Shadow, ...
    """
    for line in ass_content.split("\n"):
        if line.startswith("Style:"):
            fields = line[len("Style:"):].strip().split(",")
            if len(fields) >= 18:
                outline_colour = fields[5].strip()
                back_colour = fields[6].strip()
                border_style = fields[15].strip()
                outline = fields[16].strip()
                shadow = fields[17].strip()
                return (
                    f"BorderStyle={border_style},"
                    f"Outline={outline},"
                    f"Shadow={shadow},"
                    f"OutlineColour={outline_colour},"
                    f"BackColour={back_colour}"
                )
            break
    return ""


def _ass_time_to_sec(t: str) -> float:
    """Parse ASS timestamp 'H:MM:SS.cc' to seconds."""
    parts = t.strip().split(":")
    h = int(parts[0])
    m = int(parts[1])
    s_cc = parts[2].split(".")
    s = int(s_cc[0])
    cs = int(s_cc[1]) if len(s_cc) > 1 else 0
    return h * 3600 + m * 60 + s + cs / 100.0


def _filter_ass_by_segments(ass_content: str, subs_off_ranges: list[tuple[float, float]]) -> str:
    """Remove ASS Dialogue lines that overlap with subtitle-off time ranges.

    subs_off_ranges: list of (start, end) in clip-relative seconds where subs should be hidden.
    """
    import re

    lines = ass_content.split("\n")
    result = []
    dialogue_re = re.compile(r"^Dialogue:\s*\d+,\s*(\d+:\d+:\d+\.\d+),\s*(\d+:\d+:\d+\.\d+),")
    for line in lines:
        match = dialogue_re.match(line)
        if match:
            d_start = _ass_time_to_sec(match.group(1))
            d_end = _ass_time_to_sec(match.group(2))
            # Check if this dialogue overlaps with any subs-off range
            skip = False
            for off_start, off_end in subs_off_ranges:
                if d_start < off_end and d_end > off_start:
                    skip = True
                    break
            if skip:
                continue
        result.append(line)
    return "\n".join(result)


def _filter_ass_keep_only_ranges(ass_content: str, subs_on_ranges: list[tuple[float, float]]) -> str:
    """Keep only ASS Dialogue lines that overlap with subtitle-on time ranges.

    Used when global subtitles are off but specific segments have subtitles enabled.
    subs_on_ranges: list of (start, end) in clip-relative seconds where subs should be shown.
    """
    import re

    lines = ass_content.split("\n")
    result = []
    dialogue_re = re.compile(r"^Dialogue:\s*\d+,\s*(\d+:\d+:\d+\.\d+),\s*(\d+:\d+:\d+\.\d+),")
    for line in lines:
        match = dialogue_re.match(line)
        if match:
            d_start = _ass_time_to_sec(match.group(1))
            d_end = _ass_time_to_sec(match.group(2))
            # Only keep if this dialogue overlaps with a subs-on range
            keep = False
            for on_start, on_end in subs_on_ranges:
                if d_start < on_end and d_end > on_start:
                    keep = True
                    break
            if not keep:
                continue
        result.append(line)
    return "\n".join(result)


def _subtitle_filter(ass_path: str, force_style: str = "") -> str:
    """Return the FFmpeg subtitle filter string for a given .ass path.

    Uses the 'subtitles' filter (not 'ass') so that force_style can be
    applied.  force_style directly modifies the libass style objects via
    ass_process_force_style(), guaranteeing that outline/border settings
    are rendered even when certain libass builds silently drop them from
    the parsed ASS Style line.

    Both the 'ass' and 'subtitles' filters internally call
    ass_set_fonts(..., ASS_FONTPROVIDER_AUTODETECT, ...) which enables
    fontconfig for system font discovery automatically — no explicit
    fontprovider option is needed (and FFmpeg doesn't expose one).
    """
    safe_path = ass_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    safe_fonts_dir = CUSTOM_FONTS_DIR.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    if force_style:
        safe_force = force_style.replace("'", "\\'")
        return (
            f"subtitles=filename='{safe_path}'"
            f":fontsdir='{safe_fonts_dir}'"
            f":force_style='{safe_force}'"
        )
    return f"subtitles=filename='{safe_path}':fontsdir='{safe_fonts_dir}'"


def _detect_crop(video_path: str, start: float = 0, duration: float = 5.0) -> tuple[int, int, int, int] | None:
    """Use FFmpeg cropdetect to find baked-in black bars in the source video.

    Returns (crop_w, crop_h, crop_x, crop_y) of the detected content area,
    or None if detection fails or the entire frame is content.
    """
    try:
        # Sample a few seconds from the middle-ish of the clip
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-vf", "cropdetect=24:2:0",
            "-f", "null", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        # Parse the last cropdetect line (most stable after initial frames)
        crop_line = None
        for line in result.stderr.split("\n"):
            if "crop=" in line:
                crop_line = line
        if not crop_line:
            return None
        # Extract crop=W:H:X:Y
        match = re.search(r"crop=(\d+):(\d+):(\d+):(\d+)", crop_line)
        if not match:
            return None
        cw, ch, cx, cy = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
        return (cw, ch, cx, cy)
    except Exception as e:
        logger.debug("cropdetect failed: %s", e)
        return None


def _atempo_chain(spd: float) -> str:
    """Build chained atempo filters for a given speed value.

    FFmpeg's atempo filter only supports the 0.5–2.0 range, so extreme
    values must be split into multiple chained filters.
    """
    parts: list[str] = []
    remaining = spd
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    if abs(remaining - 1.0) > 0.001:
        parts.append(f"atempo={remaining:.4f}")
    return ",".join(parts)


def _build_speed_timeline(
    segments: list[dict],
    clip_dur: float,
    global_speed: float,
    global_volume: float,
    clip_start: float,
) -> list[dict]:
    """Build a complete timeline covering 0..clip_dur from segment overrides.

    Fills gaps between segments with global speed/volume settings.
    Returns list of dicts with keys: start, end, speed, volume, muted.
    Times are clip-relative (0-based).
    """
    sorted_segs = sorted(segments, key=lambda s: s["start"])
    timeline: list[dict] = []
    pos = 0.0

    for seg in sorted_segs:
        seg_start = max(0.0, seg["start"] - clip_start)
        seg_end = min(clip_dur, seg["end"] - clip_start)
        if seg_end <= seg_start:
            continue

        # Gap before this segment
        if seg_start > pos + 0.01:
            timeline.append({
                "start": pos, "end": seg_start,
                "speed": global_speed, "volume": global_volume, "muted": False,
            })

        seg_vol = 0.0 if seg.get("muted", False) else seg.get("volume", 1.0)
        timeline.append({
            "start": seg_start, "end": seg_end,
            "speed": seg.get("speed", global_speed),
            "volume": seg_vol, "muted": seg.get("muted", False),
        })
        pos = seg_end

    # Gap after last segment
    if pos < clip_dur - 0.01:
        timeline.append({
            "start": pos, "end": clip_dur,
            "speed": global_speed, "volume": global_volume, "muted": False,
        })

    return timeline


def _filter_keyframes_by_segment_tracking(
    keyframes: list[tuple[float, int]],
    segments: list[dict],
    clip_start: float,
    clip_end: float,
    static_sx: int = 50,
) -> list[tuple[float, int]]:
    """Filter dynamic keyframes based on per-segment subject_tracking_enabled.

    For segments where tracking is disabled, replaces keyframes within that
    time range with static subject_x fallback values.  For segments where
    tracking is enabled (or the default), keyframes are preserved.

    Args:
        keyframes: processed subject tracking keyframes [(time_relative, subject_x)]
        segments: list of segment dicts with start/end (absolute) and subject_tracking_enabled
        clip_start: absolute start time of the clip
        clip_end: absolute end time of the clip
        static_sx: fallback subject_x for tracking-off segments (default 50 = center)

    Returns:
        New keyframes list with tracking-off ranges replaced by static values.
    """
    if not keyframes or not segments:
        return keyframes

    # Check if any segments have tracking disabled
    has_tracking_off = any(
        not seg.get("subject_tracking_enabled", True) for seg in segments
    )
    if not has_tracking_off:
        return keyframes

    clip_dur = clip_end - clip_start

    # Build list of tracking-off ranges (relative to clip start)
    off_ranges = []
    for seg in segments:
        if not seg.get("subject_tracking_enabled", True):
            seg_start_rel = max(0.0, seg["start"] - clip_start)
            seg_end_rel = min(clip_dur, seg["end"] - clip_start)
            if seg_end_rel > seg_start_rel:
                off_ranges.append((seg_start_rel, seg_end_rel))

    if not off_ranges:
        return keyframes

    off_ranges.sort()

    def _in_off_range(t: float) -> bool:
        for rs, re in off_ranges:
            if rs - 0.01 <= t <= re + 0.01:
                return True
        return False

    # Build new keyframes: keep originals in tracking-on ranges,
    # replace with static in tracking-off ranges, and add boundary
    # keyframes at range edges for smooth transitions.
    result = []
    for rs, re in off_ranges:
        # Add static keyframes at boundary of each off range
        result.append((rs, static_sx))
        result.append((re, static_sx))

    # Keep original keyframes that fall in tracking-on ranges
    for t, sx in keyframes:
        if not _in_off_range(t):
            result.append((t, sx))

    # Sort by time and deduplicate close timestamps (keep latest)
    result.sort(key=lambda kf: kf[0])
    deduped = []
    for t, sx in result:
        if deduped and abs(deduped[-1][0] - t) < 0.05:
            deduped[-1] = (t, sx)
        else:
            deduped.append((t, sx))

    logger.info(
        "[SubjectTracking] _filter_keyframes_by_segment_tracking: "
        "%d original keyframes → %d filtered (%d tracking-off ranges, static_sx=%d)",
        len(keyframes), len(deduped), len(off_ranges), static_sx,
    )

    return deduped


def _build_split_filter(
    src_w: int, src_h: int,
    out_w: int, out_h: int,
    left_x_pct: float,
    right_x_pct: float,
    separator_px: int = 3,
    active_speaker: str = "none",
) -> str:
    """Build FFmpeg filtergraph for side-by-side split layout.

    Each speaker gets a horizontal strip of the source cropped to their face,
    then scaled to fit half the output height.
    active_speaker: "left", "right", or "none" — adds subtle 5% zoom to speaker.
    """
    half_h = (out_h - separator_px) // 2
    half_h = half_h - (half_h % 2)
    half_ratio = out_w / half_h

    crop_h = src_h
    crop_w = min(src_w, int(src_h * half_ratio))
    crop_w = crop_w - (crop_w % 2)

    def sx_to_offset(sx_pct):
        subject_px = src_w * sx_pct / 100
        offset = int(subject_px - crop_w / 2)
        return max(0, min(src_w - crop_w, offset))

    x_left = sx_to_offset(left_x_pct)
    x_right = sx_to_offset(right_x_pct)

    # Active speaker highlight: 5% zoom
    zoom_w = int(out_w * 1.05)
    zoom_h = int(half_h * 1.05)
    if active_speaker == "left":
        top_scale = f"scale={zoom_w}:{zoom_h},crop={out_w}:{half_h}"
        bot_scale = f"scale={out_w}:{half_h}"
    elif active_speaker == "right":
        top_scale = f"scale={out_w}:{half_h}"
        bot_scale = f"scale={zoom_w}:{zoom_h},crop={out_w}:{half_h}"
    else:
        top_scale = f"scale={out_w}:{half_h}"
        bot_scale = f"scale={out_w}:{half_h}"

    # Separator line drawn via pad + overlay isn't needed — just use vstack
    # with a small gap handled by pad filter
    return (
        f"split[s1][s2];"
        f"[s1]crop={crop_w}:{crop_h}:{x_left}:0,{top_scale}[top];"
        f"[s2]crop={crop_w}:{crop_h}:{x_right}:0,{bot_scale}[bot];"
        f"[top]pad={out_w}:{half_h + separator_px}:0:0:color=black[padtop];"
        f"[padtop][bot]vstack=inputs=2[v]"
    )


def _build_pip_filter(
    src_w: int, src_h: int,
    out_w: int, out_h: int,
    main_x_pct: float,
    pip_x_pct: float,
    pip_position: str = "bottom_right",
    pip_size_pct: float = 25.0,
) -> str:
    """Build FFmpeg filtergraph for picture-in-picture layout."""
    target_ratio = out_w / out_h

    # Main speaker: full frame crop centered on their face
    crop_h = src_h
    crop_w = min(src_w, int(src_h * target_ratio))
    crop_w = crop_w - (crop_w % 2)
    main_offset = max(0, min(src_w - crop_w, int(src_w * main_x_pct / 100 - crop_w / 2)))

    # PIP speaker: same crop approach but smaller output
    pip_w = int(out_w * pip_size_pct / 100)
    pip_h = int(out_h * pip_size_pct / 100)
    pip_w = pip_w - (pip_w % 2)
    pip_h = pip_h - (pip_h % 2)
    pip_offset = max(0, min(src_w - crop_w, int(src_w * pip_x_pct / 100 - crop_w / 2)))

    # PIP overlay position
    margin = int(out_w * 0.03)  # 3% margin
    pip_positions = {
        "bottom_right": (out_w - pip_w - margin, out_h - pip_h - margin),
        "bottom_left": (margin, out_h - pip_h - margin),
        "top_right": (out_w - pip_w - margin, margin),
        "top_left": (margin, margin),
    }
    pip_x, pip_y = pip_positions.get(pip_position, pip_positions["bottom_right"])

    return (
        f"split[main][pip];"
        f"[main]crop={crop_w}:{crop_h}:{main_offset}:0,scale={out_w}:{out_h}[mainsc];"
        f"[pip]crop={crop_w}:{crop_h}:{pip_offset}:0,scale={pip_w}:{pip_h}[pipsc];"
        f"[mainsc][pipsc]overlay={pip_x}:{pip_y}[v]"
    )


def _build_triple_filter(
    src_w: int, src_h: int,
    out_w: int, out_h: int,
    face_x_positions: list,
) -> str:
    """Build FFmpeg filtergraph for 3-speaker grid layout.

    Top row: two speakers side by side (each half width, ~40% height)
    Bottom: one speaker centered (~60% height)
    """
    top_h = int(out_h * 0.4)
    top_h = top_h - (top_h % 2)
    bot_h = out_h - top_h
    bot_h = bot_h - (bot_h % 2)
    half_w = out_w // 2
    half_w = half_w - (half_w % 2)

    # Sort faces by x position
    sorted_x = sorted(face_x_positions[:3])
    while len(sorted_x) < 3:
        sorted_x.append(50)

    # Compute crop for each speaker
    def crop_for_speaker(sx_pct, target_w, target_h):
        ratio = target_w / target_h
        cw = min(src_w, int(src_h * ratio))
        ch = src_h
        cw = cw - (cw % 2)
        offset = max(0, min(src_w - cw, int(src_w * sx_pct / 100 - cw / 2)))
        return cw, ch, offset

    cw1, ch1, x1 = crop_for_speaker(sorted_x[0], half_w, top_h)
    cw2, ch2, x2 = crop_for_speaker(sorted_x[1], half_w, top_h)
    cw3, ch3, x3 = crop_for_speaker(sorted_x[2], out_w, bot_h)

    return (
        f"split=3[a][b][c];"
        f"[a]crop={cw1}:{ch1}:{x1}:0,scale={half_w}:{top_h}[tl];"
        f"[b]crop={cw2}:{ch2}:{x2}:0,scale={half_w}:{top_h}[tr];"
        f"[c]crop={cw3}:{ch3}:{x3}:0,scale={out_w}:{bot_h}[bot];"
        f"[tl][tr]hstack=inputs=2[toprow];"
        f"[toprow][bot]vstack=inputs=2[v]"
    )


def _build_screenshare_filter(
    src_w: int, src_h: int,
    out_w: int, out_h: int,
    speaker_x_pct: float = 50,
    screen_pct: float = 60,
) -> str:
    """Build FFmpeg filtergraph for screenshare layout.

    Top: screen content (60% of output height)
    Bottom: speaker face (40% of output height)
    """
    screen_h = int(out_h * screen_pct / 100)
    screen_h = screen_h - (screen_h % 2)
    speaker_h = out_h - screen_h
    speaker_h = speaker_h - (speaker_h % 2)

    # Screen: center crop from full frame
    screen_ratio = out_w / screen_h
    scr_cw = min(src_w, int(src_h * screen_ratio))
    scr_cw = scr_cw - (scr_cw % 2)
    scr_x = (src_w - scr_cw) // 2

    # Speaker: crop centered on face
    spk_ratio = out_w / speaker_h
    spk_cw = min(src_w, int(src_h * spk_ratio))
    spk_cw = spk_cw - (spk_cw % 2)
    spk_x = max(0, min(src_w - spk_cw, int(src_w * speaker_x_pct / 100 - spk_cw / 2)))

    return (
        f"split[scr][spk];"
        f"[scr]crop={scr_cw}:{src_h}:{scr_x}:0,scale={out_w}:{screen_h}[screen];"
        f"[spk]crop={spk_cw}:{src_h}:{spk_x}:0,scale={out_w}:{speaker_h}[speaker];"
        f"[screen][speaker]vstack=inputs=2[v]"
    )


# Phase 4 (gaming): blur-fill + wide-zoom filter graph builders live
# in ``backend.services.gameplay_filters`` so they're unit-testable
# without the database / aiofiles import chain. Re-exported here so
# existing callers can keep importing them from clip_exporter.
from backend.services.gameplay_filters import (
    build_gameplay_blurfill_filter as _build_gameplay_blurfill_filter,
    build_gameplay_wide_zoom_filter as _build_gameplay_wide_zoom_filter,
)


def _build_gameplay_composite_filter(
    src_w: int,
    src_h: int,
    target_w: int,
    target_h: int,
    hud_layout: dict,
) -> str:
    """Build FFmpeg filtergraph for gameplay composite layout (9:16 from 16:9).

    Top 65%: center crop of the action area (crosshair-centered)
    Bottom 35%: HUD strip composited from killfeed, health, abilities etc.
    """
    action_h = int(target_h * 0.65)
    action_h = action_h - (action_h % 2)
    action_w = target_w

    # Center crop of source for action area
    action_aspect = action_w / action_h
    crop_w = int(src_h * action_aspect)
    crop_w = min(crop_w, src_w)
    crop_w = crop_w - (crop_w % 2)
    crop_x = (src_w - crop_w) // 2

    hud_h = target_h - action_h
    hud_h = hud_h - (hud_h % 2)
    hud_w = target_w

    # Count how many splits we need: 1 (action) + number of HUD elements
    hud_elements = {k: v for k, v in hud_layout.items() if k != "name" and isinstance(v, dict)}
    num_splits = 1 + len(hud_elements)  # action + each HUD crop

    parts = []

    if num_splits <= 1 or not hud_elements:
        # No HUD elements — just action crop + black bar
        parts.append(
            f"split=2[action_src][_dummy];"
            f"[action_src]crop={crop_w}:{src_h}:{crop_x}:0,scale={action_w}:{action_h}[action];"
            f"[_dummy]nullsink;"
            f"color=c=black:s={hud_w}x{hud_h}:d=999[hud_bg];"
            f"[action][hud_bg]vstack=inputs=2[v]"
        )
        return ";".join(parts)

    # Split input for action + each HUD element
    split_labels = ["[action_src]"] + [f"[hud_src_{i}]" for i in range(len(hud_elements))]
    parts.append(f"split={num_splits}{''.join(split_labels)}")

    # Action area
    parts.append(f"[action_src]crop={crop_w}:{src_h}:{crop_x}:0,scale={action_w}:{action_h}[action]")

    # Black background for HUD strip
    parts.append(f"color=c=black:s={hud_w}x{hud_h}:d=999[hud_bg]")

    # Extract and position each HUD element
    hud_overlay_chain = []
    for i, (elem_name, elem) in enumerate(hud_elements.items()):
        src_label = f"hud_src_{i}"
        out_label = f"hud_{elem_name}"

        # Source crop coordinates
        ex = int((elem["x_pct"] / 100) * src_w)
        ey = int((elem["y_pct"] / 100) * src_h)
        ew = int((elem["w_pct"] / 100) * src_w)
        eh = int((elem["h_pct"] / 100) * src_h)
        ew = max(2, ew - (ew % 2))
        eh = max(2, eh - (eh % 2))
        ex = min(ex, src_w - ew)
        ey = min(ey, src_h - eh)

        # Scale to fit HUD strip — each element gets proportional width
        target_elem_w = int(hud_w * 0.45)
        target_elem_h = int(target_elem_w * (eh / max(ew, 1)))
        target_elem_w = max(2, target_elem_w - (target_elem_w % 2))
        target_elem_h = max(2, min(target_elem_h, hud_h - 10))
        target_elem_h = target_elem_h - (target_elem_h % 2)

        parts.append(
            f"[{src_label}]crop={ew}:{eh}:{ex}:{ey},"
            f"scale={target_elem_w}:{target_elem_h}[{out_label}]"
        )

        # Position in HUD strip — distribute elements horizontally
        if "killfeed" in elem_name:
            pos_x = hud_w - target_elem_w - 10
            pos_y = 5
        elif "minimap" in elem_name:
            pos_x = 10
            pos_y = 5
        elif "health" in elem_name:
            pos_x = 10
            pos_y = hud_h - target_elem_h - 5
        elif "abilities" in elem_name or "ultimate" in elem_name:
            pos_x = (hud_w - target_elem_w) // 2
            pos_y = hud_h - target_elem_h - 5
        else:
            pos_x = 10 + i * (target_elem_w + 10)
            pos_y = 5

        hud_overlay_chain.append((out_label, pos_x, pos_y))

    # Build overlay chain onto hud_bg
    last_layer = "hud_bg"
    for idx, (name, x, y) in enumerate(hud_overlay_chain):
        next_layer = f"hud_step{idx}"
        parts.append(f"[{last_layer}][{name}]overlay={x}:{y}:shortest=1[{next_layer}]")
        last_layer = next_layer

    # Stack action on top of HUD strip
    parts.append(f"[action][{last_layer}]vstack=inputs=2[v]")

    return ";".join(parts)


def _build_layout_filter_chain(
    layout_timeline,
    src_w: int,
    src_h: int,
    target_aspect: str,
    export_quality: str,
    clip_start: float,
    clip_end: float,
    ass_path: str = None,
    subtitle_force_style: str = "",
    video_effects: dict = None,
) -> tuple:
    """Build FFmpeg filter chain for multi-layout compositing.

    Returns (filter_chain, is_complex, subtitle_filter).

    For SINGLE mode: returns None so caller delegates to existing _build_filter_chain.
    For SPLIT/TRIPLE/PIP/SCREENSHARE: returns a complex filtergraph.
    """
    from backend.models import LayoutMode

    if not layout_timeline or not layout_timeline.segments:
        return None, False, ""

    # If all segments are SINGLE, let the caller use the existing path
    all_single = all(s.layout_mode == LayoutMode.SINGLE for s in layout_timeline.segments)
    if all_single:
        return None, False, ""

    # Get output dimensions
    dims_table = ASPECT_RATIO_DIMS_BY_QUALITY.get(export_quality, ASPECT_RATIO_DIMS)
    out_w, out_h = dims_table.get(target_aspect, (1920, 1080))
    out_w = out_w - (out_w % 2)
    out_h = out_h - (out_h % 2)

    # For now, use the dominant non-single layout for the whole clip
    # (dynamic mid-clip switching requires segment-based encoding)
    primary_seg = max(
        layout_timeline.segments,
        key=lambda s: s.end - s.start if s.layout_mode != LayoutMode.SINGLE else 0,
    )
    mode = primary_seg.layout_mode
    registry = layout_timeline.face_registry

    filter_chain = None

    if mode == LayoutMode.SPLIT and registry and len(registry.slots) >= 2:
        sorted_slots = sorted(registry.slots, key=lambda s: s.x_center)
        left_x = sorted_slots[0].x_center
        right_x = sorted_slots[1].x_center
        filter_chain = _build_split_filter(
            src_w, src_h, out_w, out_h,
            left_x, right_x,
        )
        logger.info(
            "[Layout] SPLIT filter: left=%.0f%%, right=%.0f%%, output=%dx%d",
            left_x, right_x, out_w, out_h,
        )

    elif mode == LayoutMode.PICTURE_IN_PICTURE and registry and len(registry.slots) >= 2:
        sorted_slots = sorted(registry.slots, key=lambda s: s.frame_count, reverse=True)
        main_x = sorted_slots[0].x_center
        pip_x = sorted_slots[1].x_center
        pip_pos = primary_seg.pip_position
        pip_size = primary_seg.pip_size_pct
        filter_chain = _build_pip_filter(
            src_w, src_h, out_w, out_h,
            main_x, pip_x,
            pip_position=pip_pos, pip_size_pct=pip_size,
        )
        logger.info(
            "[Layout] PIP filter: main=%.0f%%, pip=%.0f%% (%s, %.0f%%), output=%dx%d",
            main_x, pip_x, pip_pos, pip_size, out_w, out_h,
        )

    elif mode == LayoutMode.TRIPLE and registry and len(registry.slots) >= 3:
        face_xs = [s.x_center for s in sorted(registry.slots, key=lambda s: s.x_center)[:3]]
        filter_chain = _build_triple_filter(
            src_w, src_h, out_w, out_h, face_xs,
        )
        logger.info("[Layout] TRIPLE filter: faces at %s, output=%dx%d", face_xs, out_w, out_h)

    elif mode == LayoutMode.SCREENSHARE:
        # Find the speaker's face position
        speaker_x = 50
        if registry and registry.slots:
            speaker_x = registry.slots[0].x_center
        filter_chain = _build_screenshare_filter(
            src_w, src_h, out_w, out_h,
            speaker_x_pct=speaker_x,
        )
        logger.info("[Layout] SCREENSHARE filter: speaker at %.0f%%, output=%dx%d", speaker_x, out_w, out_h)

    if filter_chain is None:
        return None, False, ""

    # The layout filters produce [v] output — it's a complex filtergraph
    # For SPLIT mode, position subtitles in the bottom panel
    sub = ""
    if ass_path:
        if mode == LayoutMode.SPLIT:
            # Push subtitles to bottom panel: offset MarginV by half the frame height
            half_h = out_h // 2
            split_margin_v = max(10, out_h - half_h - 40)
            extra_style = f"MarginV={split_margin_v}"
            if subtitle_force_style:
                extra_style = f"{subtitle_force_style},{extra_style}"
            sub = _subtitle_filter(ass_path, force_style=extra_style)
        else:
            sub = _subtitle_filter(ass_path, force_style=subtitle_force_style)

    return filter_chain, True, sub


def _build_filter_chain(
    aspect_ratio: str | None,
    src_w: int,
    src_h: int,
    ass_path: str | None,
    subject_x: int = 50,
    subject_keyframes: list[tuple[float, int]] | None = None,
    export_quality: str = "1080p",
    subtitle_force_style: str = "",
    video_path: str | None = None,
    start_time: float = 0,
    video_effects: dict | None = None,
    clip_duration: float = 0,
    face_y_center: float = 50.0,
    face_width_pct: float = 0.0,
    use_step_interpolation: bool = False,
) -> tuple[str | None, bool]:
    """Build FFmpeg video filter chain.

    Always crops to fill the target aspect ratio (no blur-background).
    Uses subject_keyframes (if provided) for time-varying crop that follows
    the subject through the clip.  Falls back to static subject_x.

    subtitle_force_style: optional force_style string (e.g.
    'BorderStyle=1,Outline=6,...') extracted from the generated ASS
    content.  Passed to the subtitles filter to guarantee outline
    rendering regardless of libass build quirks.

    video_path / start_time: used for cropdetect to remove baked-in
    black bars (pillarboxing/letterboxing) from the source video.

    Returns (filter_string, is_complex_graph).
    """
    # Determine if quality requires resolution scaling (even without aspect ratio)
    target_h = QUALITY_MAX_HEIGHT.get(export_quality, 1080)
    needs_quality_scale = (src_h != target_h)

    # Detect and remove baked-in black bars from the source video.
    # Many source videos (screen recordings, re-encoded clips) have
    # pillarboxing or letterboxing baked into the pixel data.  Without
    # this, the exported video inherits those black bars, and subtitles
    # span the full frame (including the bars) instead of the content.
    effective_w, effective_h = src_w, src_h
    precrop_filter = None
    if video_path:
        crop_result = _detect_crop(video_path, start=start_time, duration=3.0)
        if crop_result:
            cw, ch, cx, cy = crop_result
            # Only apply if cropdetect found significant black bars
            # (at least 4% removed from any dimension)
            w_removed_pct = (src_w - cw) / src_w * 100 if src_w > 0 else 0
            h_removed_pct = (src_h - ch) / src_h * 100 if src_h > 0 else 0
            if w_removed_pct >= 4 or h_removed_pct >= 4:
                # Ensure even dimensions
                cw = cw - (cw % 2)
                ch = ch - (ch % 2)
                precrop_filter = f"crop={cw}:{ch}:{cx}:{cy}"
                effective_w, effective_h = cw, ch
                logger.info(
                    "Detected baked-in black bars: source %dx%d → content %dx%d "
                    "(removed %.1f%% width, %.1f%% height)",
                    src_w, src_h, cw, ch, w_removed_pct, h_removed_pct,
                )

    # Recompute after precrop may have changed effective dimensions
    needs_quality_scale = (effective_h != target_h)

    if not aspect_ratio and not ass_path and not needs_quality_scale and not precrop_filter and not video_effects:
        return None, False, ""

    out_w, out_h = effective_w, effective_h
    sub = _subtitle_filter(ass_path, force_style=subtitle_force_style) if ass_path else ""

    # Simple linear chain: setsar → precrop → crop → scale
    # Subtitles are returned separately so they can be applied AFTER overlay
    # compositing (shapes, images, text) — ensuring subtitles render on top.
    # Start with setsar=1 to normalize non-square pixels (SAR != 1:1).
    # Many source videos have non-square SAR which causes FFmpeg's crop
    # and scale filters to produce slightly wrong dimensions, resulting
    # in thin black bars at the edges of the exported video.
    filters = ["setsar=1"]

    # Remove baked-in black bars before any aspect ratio cropping
    if precrop_filter:
        filters.append(precrop_filter)
        # Update src dimensions for subsequent aspect ratio calculations
        src_w, src_h = effective_w, effective_h

    if aspect_ratio and aspect_ratio in ASPECT_RATIO_VALUES:
        target_ratio = ASPECT_RATIO_VALUES[aspect_ratio]
        src_ratio = src_w / src_h if src_h else 1
        dims_table = ASPECT_RATIO_DIMS_BY_QUALITY.get(export_quality, ASPECT_RATIO_DIMS)
        out_w, out_h = dims_table.get(aspect_ratio, (1920, 1080))
        out_w = out_w - (out_w % 2)
        out_h = out_h - (out_h % 2)

        if abs(src_ratio - target_ratio) > 0.01:
            if target_ratio < src_ratio:
                # Crop width (e.g. 16:9 → 9:16: need narrower crop)
                crop_h = src_h
                crop_w = int(src_h * target_ratio)
            else:
                # Crop height (e.g. 9:16 → 16:9: need shorter crop)
                crop_w = src_w
                crop_h = int(src_w / target_ratio)

            crop_w = crop_w - (crop_w % 2)
            crop_h = crop_h - (crop_h % 2)

            # Apply dynamic zoom based on face size (static per-clip)
            if face_width_pct > 0:
                zoom = _compute_zoom_factor(face_width_pct)
                if abs(zoom - 1.0) > 0.02:
                    crop_w = int(crop_w / zoom)
                    crop_h = int(crop_h / zoom)
                    crop_w = crop_w - (crop_w % 2)
                    crop_h = crop_h - (crop_h % 2)
                    logger.info(
                        "[SubjectTracking] Zoom factor=%.3f (face_w=%.1f%%), adjusted crop=%dx%d",
                        zoom, face_width_pct, crop_w, crop_h,
                    )

            # Clamp crop dimensions to never exceed source frame
            crop_w = min(crop_w, src_w)
            crop_h = min(crop_h, src_h)

            max_x_offset = max(0, src_w - crop_w)

            # Vertical offset — face-aware headroom (rule of thirds)
            max_y_offset = max(0, src_h - crop_h)
            if max_y_offset > 0 and face_y_center != 50.0:
                y_offset = _compute_face_y_offset(
                    face_y_center, face_height_pct=10.0,
                    src_h=src_h, crop_h=crop_h,
                    target_face_position=0.38,
                )
                logger.info(
                    "[SubjectTracking] Face-aware y_offset=%d (face_y=%.1f%%, headroom at 38%%)",
                    y_offset, face_y_center,
                )
            else:
                y_offset = (src_h - crop_h) // 2
            y_offset = max(0, min(max_y_offset, y_offset))

            logger.info(
                "[SubjectTracking] _build_filter_chain: %s→%s, src=%dx%d, crop=%dx%d, max_x_offset=%d",
                f"{src_ratio:.3f}", f"{target_ratio:.3f}", src_w, src_h, crop_w, crop_h, max_x_offset,
            )

            # Dynamic or static horizontal offset — center subject in frame
            if subject_keyframes and len(subject_keyframes) > 1:
                unique_sx = set(kf[1] for kf in subject_keyframes)
                if len(unique_sx) > 1:
                    # Dynamic crop: time-varying x offset
                    if use_step_interpolation:
                        # Step mode (frontend keyframes): instant snap, no transitions
                        _final_kf = subject_keyframes
                    else:
                        # Smoothstep mode (backend keyframes): insert snap transitions
                        # for large jumps so they don't produce slow pans
                        _final_kf = _insert_snap_transitions(subject_keyframes)
                    x_expr = _build_crop_x_expr(
                        _final_kf, max_x_offset, src_w, crop_w,
                        step_mode=use_step_interpolation,
                    )
                    filters.append(f"crop={crop_w}:{crop_h}:{x_expr}:{y_offset}")
                    logger.info(
                        "[SubjectTracking] DYNAMIC CROP (%s): %d keyframes, %d unique sx values, "
                        "crop=%dx%d, y_offset=%d",
                        "step" if use_step_interpolation else "smoothstep",
                        len(subject_keyframes), len(unique_sx), crop_w, crop_h, y_offset,
                    )
                else:
                    # All keyframes same value → static centered
                    sx = _safe_subject_x(subject_keyframes[0][1])
                    x_offset = _verify_face_centering(sx, src_w, crop_w, label="converged")
                    filters.append(f"crop={crop_w}:{crop_h}:{x_offset}:{y_offset}")
                    logger.info(
                        "[SubjectTracking] STATIC CROP (converged keyframes): sx=%d → x_offset=%d, "
                        "subject_pixel=%.0f, crop_center=%.0f, crop=%dx%d",
                        sx, x_offset, src_w * sx / 100, x_offset + crop_w / 2, crop_w, crop_h,
                    )
            else:
                # Static crop: single subject_x value, centered
                if subject_keyframes:
                    sx = _safe_subject_x(subject_keyframes[0][1])
                else:
                    sx = _safe_subject_x(subject_x)
                x_offset = _verify_face_centering(sx, src_w, crop_w, label="static")
                filters.append(f"crop={crop_w}:{crop_h}:{x_offset}:{y_offset}")
                subject_pixel = src_w * sx / 100
                crop_center = x_offset + crop_w / 2
                centering_error = abs(subject_pixel - crop_center)
                logger.info(
                    "[SubjectTracking] STATIC CROP: sx=%d → x_offset=%d, "
                    "subject at pixel %.0f, crop center at pixel %.0f (error=%.1fpx), crop=%dx%d",
                    sx, x_offset, subject_pixel, crop_center, centering_error, crop_w, crop_h,
                )

        filters.append(f"scale={out_w}:{out_h}")

        # ── Centering verification for preview-export parity ──
        # Only runs when a crop actually happened (src_ratio != target_ratio).
        # When the source already matches the target ratio, crop_w/crop_h
        # are never defined and there's nothing to verify — the full frame
        # is passed through and centering is trivially correct.
        if subject_keyframes and abs(src_ratio - target_ratio) > 0.01:
            for label_kf, kf in [("first", subject_keyframes[0]), ("last", subject_keyframes[-1])]:
                t_kf, sx_kf = kf
                sx_safe = _safe_subject_x(sx_kf, src_ratio=src_ratio, target_ratio=target_ratio)
                offset_kf = _center_crop_offset(sx_safe, src_w, crop_w)
                face_pixel = src_w * sx_kf / 100
                face_in_crop = (face_pixel - offset_kf) / crop_w * 100 if crop_w > 0 else 50

                # Compute what the preview shows
                R = src_ratio / target_ratio if target_ratio > 0 else 1
                preview_pct = (R * sx_safe - 50) / (R - 1) if R > 1.01 else sx_safe
                preview_pct = max(0, min(100, preview_pct))

                logger.info(
                    "[SubjectTracking] Parity check (%s kf): sx=%d → safe=%d, "
                    "export: offset=%dpx face@%.0f%% of crop, "
                    "preview: objectPosition=%.1f%%, "
                    "error=%.1f%%",
                    label_kf, sx_kf, sx_safe, offset_kf, face_in_crop,
                    preview_pct, abs(face_in_crop - 50),
                )

    elif needs_quality_scale:
        # No aspect ratio change but quality requires resizing — scale preserving aspect ratio
        # Use -2 for width so FFmpeg auto-computes an even width from the target height
        out_h = target_h
        out_w = -2
        filters.append(f"scale={out_w}:{out_h}")

    # Apply video effects (brightness/contrast/saturation/blur/hue/sepia)
    # These must match the CSS filter() values used in the frontend preview.
    #
    # CSS filter mappings:
    #   brightness(factor) — multiplies each RGB channel by factor
    #   contrast(factor)   — scales around midpoint: factor*(c-0.5)+0.5
    #   saturate(factor)   — saturation multiplier
    #
    # FFmpeg eq filter:
    #   brightness — ADDITIVE offset on luma (NOT the same as CSS brightness!)
    #   contrast   — multiplier around midpoint (matches CSS)
    #   saturation — multiplier (matches CSS)
    if video_effects:
        # brightness: CSS brightness(1+val/100) is a linear RGB multiply.
        # FFmpeg eq brightness is ADDITIVE on luma, which produces a
        # completely different look.  Use colorlevels to scale the RGB
        # input range which gives a true multiplicative brightness.
        brightness_val = video_effects.get("brightness", 0)
        if brightness_val != 0:
            factor = 1 + brightness_val / 100
            if factor >= 1.0:
                # Brighten: map input [0, 1/factor] → output [0, 1]
                inv = 1.0 / factor
                filters.append(
                    f"colorlevels=rimin=0:rimax={inv:.4f}"
                    f":gimin=0:gimax={inv:.4f}"
                    f":bimin=0:bimax={inv:.4f}"
                )
            else:
                # Darken: map input [0, 1] → output [0, factor]
                filters.append(
                    f"colorlevels=romin=0:romax={factor:.4f}"
                    f":gomin=0:gomax={factor:.4f}"
                    f":bomin=0:bomax={factor:.4f}"
                )

        eq_parts = []
        # contrast: CSS contrast(1+val/100) matches FFmpeg eq contrast multiplier
        if video_effects.get("contrast", 0) != 0:
            eq_parts.append(f"contrast={1 + video_effects['contrast'] / 100:.4f}")
        # saturation: CSS saturate(1+val/100) matches FFmpeg eq saturation multiplier
        if video_effects.get("saturation", 0) != 0:
            eq_parts.append(f"saturation={1 + video_effects['saturation'] / 100:.4f}")
        if eq_parts:
            filters.append(f"eq={':'.join(eq_parts)}")

        # hue rotation: frontend uses hue-rotate(Xdeg), FFmpeg uses hue=h=X
        if video_effects.get("hue_rotate", 0) != 0:
            filters.append(f"hue=h={video_effects['hue_rotate']:.1f}")

        # blur: frontend uses blur(Xpx), FFmpeg uses boxblur=X:X
        blur_val = video_effects.get("blur", 0)
        if blur_val > 0:
            # Scale blur from CSS px to FFmpeg boxblur radius (approx mapping)
            ffmpeg_blur = max(1, int(blur_val * 1.5))
            filters.append(f"boxblur={ffmpeg_blur}:{ffmpeg_blur}")

        # sepia: apply via colorchannelmixer to approximate CSS sepia()
        sepia_val = video_effects.get("sepia", 0)
        if sepia_val > 0:
            s = sepia_val / 100.0
            # Standard sepia matrix blended with identity by amount s
            rr = 1 - s + s * 0.393
            rg = s * 0.769
            rb = s * 0.189
            gr = s * 0.349
            gg = 1 - s + s * 0.686
            gb = s * 0.168
            br = s * 0.272
            bg = s * 0.534
            bb = 1 - s + s * 0.131
            filters.append(
                f"colorchannelmixer={rr:.3f}:{rg:.3f}:{rb:.3f}:0:"
                f"{gr:.3f}:{gg:.3f}:{gb:.3f}:0:"
                f"{br:.3f}:{bg:.3f}:{bb:.3f}:0"
            )

        # opacity: applied as alpha blend with black if < 1
        opacity_val = video_effects.get("opacity", 1.0)
        if opacity_val < 1.0:
            filters.append(f"colorchannelmixer=aa={opacity_val:.3f}")

        # ── Video transform: position, size, rotation, fades ──
        # These match the CSS transform applied in the preview viewport.
        pos_x = video_effects.get("position_x", 50)
        pos_y = video_effects.get("position_y", 50)
        vid_w = video_effects.get("width", 100)
        vid_h = video_effects.get("height", 100)
        vid_rot = video_effects.get("rotation", 0)
        vid_fade_in = video_effects.get("fade_in", 0)
        vid_fade_out = video_effects.get("fade_out", 0)

        has_transform = (
            pos_x != 50 or pos_y != 50 or
            vid_w != 100 or vid_h != 100 or
            vid_rot != 0
        )

        if has_transform:
            # To apply position/size/rotation we need a complex filter:
            # 1. Scale the video to the desired size
            # 2. Rotate if needed
            # 3. Overlay onto a background at the desired position
            #
            # Since this changes from a simple chain to a complex graph,
            # we handle it by inserting scale + rotate + crop filters.
            #
            # Size: scale the video relative to original dimensions
            if vid_w != 100 or vid_h != 100:
                sw = vid_w / 100.0
                sh = vid_h / 100.0
                filters.append(
                    f"scale=trunc(iw*{sw:.4f}/2)*2:trunc(ih*{sh:.4f}/2)*2"
                )

            # Rotation: FFmpeg rotate filter (radians, with transparent fill)
            if vid_rot != 0:
                rad = vid_rot * 3.14159265 / 180.0
                filters.append(
                    f"rotate={rad:.6f}:fillcolor=black:ow=rotw({rad:.6f}):oh=roth({rad:.6f})"
                )

            # Position offset: pad to a larger canvas, overlay at offset, then crop.
            # Preview positions the center of the video at (pos_x%, pos_y%)
            # relative to the viewport. We use pad+crop so that after any
            # prior scale/rotate the dimensions are handled correctly via iw/ih.
            if pos_x != 50 or pos_y != 50:
                # Offset in pixels relative to current stream dimensions
                ox_pct = (pos_x - 50) / 100.0
                oy_pct = (pos_y - 50) / 100.0
                # Pad to 3x canvas centered, then crop back with offset
                filters.append(
                    f"pad=w=3*iw:h=3*ih:x=iw:y=ih:color=black"
                )
                filters.append(
                    f"crop=w={src_w}:h={src_h}"
                    f":x=iw/2-{src_w}/2-({src_w}*{ox_pct:.4f})"
                    f":y=ih/2-{src_h}/2-({src_h}*{oy_pct:.4f})"
                )

        # Fade in/out on main video
        if vid_fade_in > 0:
            filters.append(f"fade=t=in:st=0:d={vid_fade_in:.3f}")
        if vid_fade_out > 0 and clip_duration > 0:
            fade_out_start = max(0, clip_duration - vid_fade_out)
            filters.append(f"fade=t=out:st={fade_out_start:.3f}:d={vid_fade_out:.3f}")

        logger.info(
            "Video effects applied: brightness=%.1f contrast=%.1f saturation=%.1f "
            "blur=%.1f hue=%.1f sepia=%.1f opacity=%.2f "
            "pos=(%.1f,%.1f) size=(%.1f,%.1f) rot=%.1f fade_in=%.1f fade_out=%.1f",
            video_effects.get("brightness", 0), video_effects.get("contrast", 0),
            video_effects.get("saturation", 0), video_effects.get("blur", 0),
            video_effects.get("hue_rotate", 0), video_effects.get("sepia", 0),
            video_effects.get("opacity", 1.0),
            pos_x, pos_y, vid_w, vid_h, vid_rot, vid_fade_in, vid_fade_out,
        )

    # NOTE: subtitle filter (sub) is NOT appended here — it is returned
    # separately so the caller can apply it AFTER overlay compositing,
    # ensuring subtitles render on top of shapes/images/text overlays.

    # VA-API needs frames uploaded to GPU after CPU-side filters
    if app_settings.GPU_ACCELERATION_ENABLED:
        gpu = detect_gpu_capabilities()
        if gpu["encoder"] == "h264_vaapi" and filters:
            filters.append("format=nv12")
            filters.append("hwupload")

    return (",".join(filters) if filters else None, False, sub)


# ── Font family → file path mapping for drawtext ─────────────────────
# Maps CSS/frontend font family names to absolute font file paths.
_FONT_FAMILY_MAP: dict[str, str] = {
    "DM Sans": "/usr/share/fonts/truetype/dmsans/DMSans.ttf",
    "dm sans": "/usr/share/fonts/truetype/dmsans/DMSans.ttf",
    "Liberation Sans": "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "Liberation Serif": "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "Liberation Mono": "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "DejaVu Sans": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVu Serif": "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "DejaVu Sans Mono": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "FreeSans": "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "Montserrat": "/usr/share/fonts/truetype/google-fonts/Montserrat.ttf",
    "Open Sans": "/usr/share/fonts/truetype/google-fonts/OpenSans.ttf",
    "Roboto": "/usr/share/fonts/truetype/google-fonts/Roboto.ttf",
    "Poppins": "/usr/share/fonts/truetype/google-fonts/Poppins-Regular.ttf",
    "Inter": "/usr/share/fonts/truetype/google-fonts/Inter.ttf",
    "Nunito": "/usr/share/fonts/truetype/google-fonts/Nunito.ttf",
    "Lato": "/usr/share/fonts/truetype/google-fonts/Lato-Regular.ttf",
    "Oswald": "/usr/share/fonts/truetype/google-fonts/Oswald.ttf",
    "Playfair Display": "/usr/share/fonts/truetype/google-fonts/PlayfairDisplay.ttf",
    "Bebas Neue": "/usr/share/fonts/truetype/google-fonts/BebasNeue-Regular.ttf",
    # Common CSS fallbacks
    "sans-serif": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "serif": "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "monospace": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "Arial": "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "Helvetica": "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "Times New Roman": "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "Courier New": "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
}
# Bold font variants — used when font_weight >= 600
_FONT_BOLD_MAP: dict[str, str] = {
    "DM Sans": "/usr/share/fonts/truetype/dmsans/DMSans-Bold.ttf",
    "dm sans": "/usr/share/fonts/truetype/dmsans/DMSans-Bold.ttf",
    "Liberation Sans": "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "Liberation Serif": "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "Liberation Mono": "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "DejaVu Sans": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "DejaVu Serif": "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "DejaVu Sans Mono": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "FreeSans": "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "Poppins": "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
    "Lato": "/usr/share/fonts/truetype/google-fonts/Lato-Bold.ttf",
    "Bebas Neue": "/usr/share/fonts/truetype/google-fonts/BebasNeue-Regular.ttf",
    # NOTE: Variable fonts (Montserrat, Open Sans, Roboto, Inter, Nunito,
    # Oswald, Playfair Display) are intentionally NOT in this bold map.
    # They are in _FONT_FAMILY_MAP and _resolve_font_path uses fonttools
    # to instantiate a static bold instance at the requested weight.
    # Having them here would short-circuit the instantiation code.
    # Common CSS fallbacks
    "sans-serif": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "serif": "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "monospace": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "Arial": "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "Helvetica": "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "Times New Roman": "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "Courier New": "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
}
_DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_DEFAULT_FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_CUSTOM_FONTS_DIR = "/data/fonts"
_VARIABLE_FONT_INSTANCE_DIR = "/tmp/font-instances"

# Cache for fc-query font family name lookups (avoids repeated subprocess calls)
_fc_query_cache: dict[str, str | None] = {}

# Cache for instantiated variable font paths: (font_path, weight) -> static_ttf_path
_variable_font_cache: dict[tuple[str, int], str | None] = {}


def _instantiate_variable_font(font_path: str, weight: int) -> str | None:
    """Create a static font instance from a variable font at the given weight.

    Uses fonttools to pin the wght axis to the requested value.
    Returns the path to the static .ttf, or None on failure.
    Results are cached so each (font_path, weight) pair is only generated once.
    """
    cache_key = (font_path, weight)
    if cache_key in _variable_font_cache:
        return _variable_font_cache[cache_key]

    try:
        from fontTools.ttLib import TTFont
        from fontTools.varLib.instancer import instantiateVariableFont
    except ImportError:
        logger.warning("fonttools not available — cannot instantiate variable font %s at weight %d", font_path, weight)
        _variable_font_cache[cache_key] = None
        return None

    try:
        tt = TTFont(font_path)
        # Check if it actually has a wght axis
        if "fvar" not in tt:
            tt.close()
            _variable_font_cache[cache_key] = None
            return None

        axes = {a.axisTag: a for a in tt["fvar"].axes}
        if "wght" not in axes:
            tt.close()
            _variable_font_cache[cache_key] = None
            return None

        # Clamp weight to the font's supported range
        wght_axis = axes["wght"]
        clamped_weight = int(max(wght_axis.minValue, min(wght_axis.maxValue, weight)))

        os.makedirs(_VARIABLE_FONT_INSTANCE_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(font_path))[0]
        # v2 suffix invalidates stale cache from buggy inplace=False code
        out_path = os.path.join(_VARIABLE_FONT_INSTANCE_DIR, f"{base}-w{clamped_weight}-v2.ttf")

        if os.path.isfile(out_path):
            logger.info("Using cached variable font instance: %s", out_path)
            tt.close()
            _variable_font_cache[cache_key] = out_path
            return out_path

        instantiateVariableFont(tt, {"wght": clamped_weight}, inplace=True)
        tt.save(out_path)
        tt.close()
        logger.info("Instantiated variable font: %s weight=%d → %s", font_path, clamped_weight, out_path)
        _variable_font_cache[cache_key] = out_path
        return out_path
    except Exception as exc:
        logger.error("Failed to instantiate variable font %s at weight %d: %s", font_path, weight, exc, exc_info=True)
        _variable_font_cache[cache_key] = None
        return None


def _font_family_name_cached(font_path: str) -> str | None:
    """Extract the internal font family name via fc-query, with caching."""
    if font_path not in _fc_query_cache:
        try:
            result = subprocess.run(
                ["fc-query", "--format", "%{family}", font_path],
                capture_output=True, text=True, timeout=5,
            )
            name = result.stdout.strip().split(",")[0].strip() if result.returncode == 0 else None
            _fc_query_cache[font_path] = name if name else None
        except Exception:
            _fc_query_cache[font_path] = None
    return _fc_query_cache[font_path]


def _resolve_font_path(font_family: str, font_weight: int = 400) -> str:
    """Resolve a CSS font family name to an absolute .ttf path.

    Checks the built-in mapping first, then custom uploaded fonts dir.
    When font_weight >= 600 (semi-bold/bold), prefer the bold variant.
    For variable fonts, instantiates a static instance at the requested weight
    using fonttools so FFmpeg drawtext renders the correct weight.
    """
    is_bold = font_weight >= 600

    # Try bold variant first when weight is bold
    if is_bold:
        if font_family in _FONT_BOLD_MAP:
            path = _FONT_BOLD_MAP[font_family]
            if os.path.isfile(path):
                return path
        # Case-insensitive bold fallback
        for name, path in _FONT_BOLD_MAP.items():
            if name.lower() == font_family.lower():
                if os.path.isfile(path):
                    return path

    # Direct match (regular weight)
    if font_family in _FONT_FAMILY_MAP:
        path = _FONT_FAMILY_MAP[font_family]
        if os.path.isfile(path):
            # For non-default weights, try to instantiate variable font
            if font_weight != 400:
                instance = _instantiate_variable_font(path, font_weight)
                if instance:
                    logger.info("Variable font instantiated: '%s' weight=%d → %s", font_family, font_weight, instance)
                    return instance
                else:
                    logger.warning("Variable font instantiation failed for '%s' weight=%d, using base file: %s", font_family, font_weight, path)
            return path
    # Case-insensitive fallback
    for name, path in _FONT_FAMILY_MAP.items():
        if name.lower() == font_family.lower():
            if os.path.isfile(path):
                if font_weight != 400:
                    instance = _instantiate_variable_font(path, font_weight)
                    if instance:
                        return instance
                return path
    # Check custom uploaded fonts in multiple directories
    _font_search_dirs = [_CUSTOM_FONTS_DIR]
    # Also check job-specific font uploads if job_id context is available
    for fdir in _font_search_dirs:
        if os.path.isdir(fdir):
            normalized_family = font_family.lower().replace(" ", "").replace("-", "")
            for fname in os.listdir(fdir):
                fpath = os.path.join(fdir, fname)
                if not os.path.isfile(fpath):
                    continue
                ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
                if ext not in ("ttf", "otf"):
                    continue
                base = os.path.splitext(fname)[0]
                base_normalized = base.lower().replace(" ", "").replace("-", "")
                # Strategy 1: filename-based match (fast)
                if base_normalized == normalized_family:
                    logger.info("Custom font resolved via filename: '%s' → %s", font_family, fpath)
                    if font_weight != 400:
                        instance = _instantiate_variable_font(fpath, font_weight)
                        if instance:
                            return instance
                    return fpath
                # Strategy 2: fc-query family name match (authoritative)
                actual_family = _font_family_name_cached(fpath)
                if actual_family and actual_family.lower().replace(" ", "").replace("-", "") == normalized_family:
                    logger.info("Custom font resolved via fc-query: '%s' → %s", font_family, fpath)
                    if font_weight != 400:
                        instance = _instantiate_variable_font(fpath, font_weight)
                        if instance:
                            return instance
                    return fpath
                # Strategy 3: bold variant matching
                if is_bold and (
                    base_normalized == f"{normalized_family}bold"
                ):
                    logger.info("Custom bold font resolved: '%s' (weight=%d) → %s", font_family, font_weight, fpath)
                    return fpath
    # Use bold fallback when weight is bold and the bold file exists
    fallback = _DEFAULT_FONT
    if is_bold and os.path.isfile(_DEFAULT_FONT_BOLD):
        fallback = _DEFAULT_FONT_BOLD
    logger.warning(
        "Font '%s' (weight=%d) not found in built-in maps or custom dirs (%s) — "
        "falling back to default: %s",
        font_family, font_weight, _CUSTOM_FONTS_DIR, fallback,
    )
    return fallback


def _build_text_overlay_filters(text_overlays: list, clip_start: float = 0, video_out_w: int = 1920, video_out_h: int = 1080) -> tuple[str, list[str]]:
    """Build FFmpeg drawtext filter chain for text overlays.

    Each text overlay becomes a drawtext filter with enable/disable based on timing.
    Position is given as percentage (0-100) and converted to pixel expressions.
    Font size and pixel-based properties are scaled up so the export matches the
    proportional size seen in the browser preview (which renders fontSize pixels
    inside a container roughly half the output resolution).

    Returns:
        (filter_chain_string, list_of_warning_messages)
    """
    # The frontend preview renders text overlays at `fontSize px` inside a viewport
    # container whose width ≈ 50% of the user's screen height × aspect ratio.
    # For typical desktop displays this means the preview container is roughly half
    # the output frame width.  To make the export text look the same proportional
    # size we scale pixel values by output_longest_edge / PREVIEW_REF.
    _PREVIEW_REF = 960
    res_scale = max(video_out_w, video_out_h) / _PREVIEW_REF

    parts = []
    warnings: list[str] = []
    for i, overlay in enumerate(text_overlays):
        text = overlay.get("text", "")
        # Escape all FFmpeg drawtext special characters (backslash first)
        for ch in ('\\', "'", ':', '%', '{', '}', ';', '[', ']'):
            text = text.replace(ch, f'\\{ch}')
        if not text:
            warnings.append(f"Text overlay {i+1}: empty text — skipped")
            continue
        x_pct = overlay.get("x", 50) / 100.0
        y_pct = overlay.get("y", 50) / 100.0
        # Scale font_size from preview-canvas pixels to output-video pixels
        font_size = max(8, int(round(overlay.get("font_size", 48) * res_scale)))
        font_color = overlay.get("font_color", "#FFFFFF")
        font_family = overlay.get("font_family", "sans-serif")
        opacity = overlay.get("opacity", 1.0)
        start_t = overlay.get("start_time", 0) - clip_start
        end_t = overlay.get("end_time", 0) - clip_start
        # Skip overlays entirely outside the clip time range
        if end_t <= 0:
            warnings.append(f"Text overlay {i+1}: outside clip time range (end_t={end_t:.2f}s) — skipped")
            continue
        font_weight = overlay.get("font_weight", 400)
        if isinstance(font_weight, str):
            font_weight = 700 if font_weight.lower() == "bold" else 400
        else:
            font_weight = int(round(font_weight))
        font_path = _resolve_font_path(font_family, font_weight=font_weight)

        # Validate font file exists — fallback to default if missing
        if not os.path.isfile(font_path):
            logger.error(
                "FONT NOT FOUND: %s (family='%s', weight=%d) — text overlay %d will use fallback",
                font_path, font_family, font_weight, i + 1,
            )
            warnings.append(f"Text overlay {i+1}: font '{font_family}' not found, using default font")
            font_path = _DEFAULT_FONT_BOLD if font_weight >= 600 and os.path.isfile(_DEFAULT_FONT_BOLD) else _DEFAULT_FONT
            if not os.path.isfile(font_path):
                logger.error("FALLBACK FONT ALSO MISSING: %s — skipping text overlay %d", font_path, i + 1)
                warnings.append(f"Text overlay {i+1}: no fonts available — skipped entirely")
                continue

        # CSS -webkit-text-stroke: Npx with paint-order:stroke fill shows N/2
        # visible per side (fill covers inner half).  FFmpeg borderw renders
        # the full value outward.  So: borderw = outlineWidth/2 * res_scale
        # to match the visible stroke thickness in the preview.
        outline_width = max(0, int(round(overlay.get("outline_width", 0) * res_scale / 2)))
        outline_color = overlay.get("outline_color", "#000000")
        fade_in = overlay.get("fade_in", 0)
        fade_out = overlay.get("fade_out", 0)

        # Build alpha expression with fade in/out support
        # FFmpeg drawtext alpha accepts an expression evaluated per frame
        safe_start = max(0, start_t)
        has_fade = (fade_in > 0 or fade_out > 0) and end_t > start_t
        if has_fade:
            alpha_parts = [f"{opacity:.2f}"]
            if fade_in > 0:
                # Ramp from 0 to 1 over fade_in seconds after start
                alpha_parts.append(f"if(lt(t-{safe_start:.3f},{fade_in:.3f}),(t-{safe_start:.3f})/{fade_in:.3f},1)")
            if fade_out > 0:
                # Ramp from 1 to 0 over fade_out seconds before end
                fo_start = end_t - fade_out
                alpha_parts.append(f"if(gt(t,{fo_start:.3f}),({end_t:.3f}-t)/{fade_out:.3f},1)")
            alpha_expr = "*".join(alpha_parts)
        else:
            alpha_expr = f"{opacity:.2f}"

        # Text animation support (matches RenderEngine canvas animations)
        animation = overlay.get("animation", "")
        y_expr = f"h*{y_pct:.4f}-th/2"
        fontsize_expr = str(font_size)

        slide_offset = int(round(30 * res_scale))
        if animation == "slide-up" and end_t > start_t:
            # Slide up from below over 0.5s (offset scaled for output resolution)
            anim_dur = 0.5
            y_expr = (
                f"h*{y_pct:.4f}-th/2"
                f"+if(lt(t-{safe_start:.3f},{anim_dur})"
                f",(1-(t-{safe_start:.3f})/{anim_dur})*{slide_offset},0)"
            )
        elif animation == "pop" and end_t > start_t:
            # Scale from 50% to 100% over 0.3s with overshoot
            anim_dur = 0.3
            fontsize_expr = (
                f"if(lt(t-{safe_start:.3f},{anim_dur})"
                f",{font_size}*(0.5+0.5*(t-{safe_start:.3f})/{anim_dur})"
                f",{font_size})"
            )

        # Text alignment — adjust x expression
        text_align = overlay.get("text_align", "center")
        if text_align == "left":
            x_expr = f"w*{x_pct:.4f}"
        elif text_align == "right":
            x_expr = f"w*{x_pct:.4f}-tw"
        else:
            x_expr = f"w*{x_pct:.4f}-tw/2"

        # Build drawtext with enable expression for timing
        dt = (
            f"drawtext=text='{text}'"
            f":x={x_expr}"
            f":y={y_expr}"
            f":fontsize='{fontsize_expr}'"
            f":fontcolor={font_color}"
            f":fontfile={font_path}"
            f":alpha='{alpha_expr}'"
        )
        if outline_width > 0:
            dt += f":borderw={outline_width}:bordercolor={outline_color}"

        # Shadow support (FFmpeg drawtext shadowcolor/shadowx/shadowy)
        # Scale shadow offsets for output resolution
        shadow_x = int(round(overlay.get("shadow_offset_x", 0) * res_scale))
        shadow_y = int(round(overlay.get("shadow_offset_y", 0) * res_scale))
        shadow_color = overlay.get("shadow_color", "")
        if shadow_x or shadow_y or shadow_color:
            # Parse rgba() or hex shadow color to hex for FFmpeg
            hex_shadow = "#000000"
            if shadow_color.startswith("rgba("):
                parts_c = shadow_color.replace("rgba(", "").replace(")", "").split(",")
                if len(parts_c) >= 3:
                    try:
                        hex_shadow = "#{:02x}{:02x}{:02x}".format(
                            int(parts_c[0].strip()), int(parts_c[1].strip()), int(parts_c[2].strip())
                        )
                    except ValueError:
                        pass
            elif shadow_color.startswith("#"):
                hex_shadow = shadow_color
            dt += f":shadowcolor={hex_shadow}:shadowx={shadow_x}:shadowy={shadow_y}"

        # NOTE: FFmpeg drawtext box=1 only supports rectangular backgrounds.
        # Frontend rounded corners (bgRadius) are approximated as sharp rectangles.
        # Shadow blur (shadowBlur) is also not supported — only offset + color.
        bg_color = overlay.get("background_color")
        bg_opacity_pct = overlay.get("bg_opacity", 0)
        # Scale bg_padding for output resolution
        bg_padding = int(round(overlay.get("bg_padding", 8) * res_scale))
        if bg_color and bg_opacity_pct > 0:
            bg_alpha = bg_opacity_pct / 100.0
            dt += f":box=1:boxcolor={bg_color}@{bg_alpha:.2f}:boxborderw={bg_padding}"
        elif bg_color:
            dt += f":box=1:boxcolor={bg_color}@0.5:boxborderw={bg_padding}"

        if end_t > start_t:
            dt += f":enable='between(t,{safe_start:.3f},{end_t:.3f})'"
        parts.append(dt)

    return ",".join(parts), warnings


def _resolve_media_path(src: str, job_id: str) -> str | None:
    """Resolve an overlay src (URL or path) to a filesystem path.

    Handles:
    - /api/files/{job_id}/media/{filename} → /data/uploads/{job_id}/media/{filename}
    - Absolute filesystem paths
    - Bare filenames looked up in the job's media directory
    """
    if not src:
        return None
    # URL path from frontend: /api/files/{job_id}/media/{filename}
    api_prefix = f"/api/files/{job_id}/media/"
    if src.startswith(api_prefix):
        filename = src[len(api_prefix):]
        path = f"/data/uploads/{job_id}/media/{filename}"
        if os.path.isfile(path):
            return path
        logger.warning("Overlay media file not found: %s", path)
        return None
    # Generic /api/files/ pattern (different job_id in path)
    if src.startswith("/api/files/"):
        # Extract job_id and filename from URL: /api/files/<jid>/media/<fname>
        parts = src.split("/")
        if len(parts) >= 5 and parts[3] == "media" or (len(parts) >= 6 and parts[4] == "media"):
            # /api/files/<jid>/media/<fname>
            try:
                idx = parts.index("media")
                jid = parts[idx - 1]
                fname = "/".join(parts[idx + 1:])
                path = f"/data/uploads/{jid}/media/{fname}"
                if os.path.isfile(path):
                    return path
            except (ValueError, IndexError):
                pass
        # Also try global media library
        global_dir = "/data/uploads/_library/media"
        basename = parts[-1] if parts else ""
        if basename:
            gpath = os.path.join(global_dir, basename)
            if os.path.isfile(gpath):
                return gpath
        logger.warning("Could not resolve overlay media URL: %s", src)
        return None
    # Absolute path
    if os.path.isabs(src) and os.path.isfile(src):
        return src
    # Bare filename — look in job media dir, then global library
    media_dir = f"/data/uploads/{job_id}/media"
    path = os.path.join(media_dir, os.path.basename(src))
    if os.path.isfile(path):
        return path
    global_path = os.path.join("/data/uploads/_library/media", os.path.basename(src))
    if os.path.isfile(global_path):
        return global_path
    logger.warning("Overlay media not found: src=%s, job_id=%s", src, job_id)
    return None


async def _render_shape_to_png(shape: dict, video_width: int, video_height: int, output_dir: str, idx: int) -> str | None:
    """Render a shape overlay to a temporary PNG file for FFmpeg compositing.

    Generates the shape as a simple colored rectangle/ellipse using FFmpeg lavfi
    filters (no SVG dependency). Returns the path to the generated PNG, or None on failure.
    """
    shape_type = shape.get("shape_type", "rectangle")
    w_pct = shape.get("width", 20) / 100.0
    h_pct = shape.get("height", 20) / 100.0
    fill_color = shape.get("fill_color", "#FF3B30")
    stroke_color = shape.get("stroke_color", "#FFFFFF")
    stroke_width_raw = shape.get("stroke_width", 2)
    corner_radius_raw = shape.get("corner_radius", 0)

    # Compute pixel dimensions (even numbers for FFmpeg compatibility)
    px_w = max(4, int(video_width * w_pct) // 2 * 2)
    px_h = max(4, int(video_height * h_pct) // 2 * 2)

    # stroke_width and corner_radius are authored in CSS pixels at the DOM
    # preview resolution, which is typically ~50vh tall (~540px on a 1080p
    # display).  Scale them to the export video resolution so the visual
    # proportions in the exported video match what the user sees in the editor.
    SHAPE_PROPERTY_REF_HEIGHT = 540
    shape_scale = video_height / SHAPE_PROPERTY_REF_HEIGHT
    stroke_width = max(0, int(round(stroke_width_raw * shape_scale)))
    corner_radius = max(0, int(round(corner_radius_raw * shape_scale)))

    png_path = os.path.join(output_dir, f"_shape_{idx}.png")
    hex_fill = fill_color.lstrip('#')[:6]
    hex_stroke = stroke_color.lstrip('#')[:6]

    try:
        # Rectangles: always use PIL for consistent rounded corners + stroke
        if shape_type == "rectangle":
            from PIL import Image, ImageDraw
            # corner_radius is already in video pixels — clamp to half the
            # shortest side so Pillow doesn't error out.
            radius = min(corner_radius, min(px_w, px_h) // 2)

            img = Image.new("RGBA", (px_w, px_h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            fill_rgb = tuple(int(hex_fill[i:i+2], 16) for i in (0, 2, 4)) + (255,)
            if stroke_width > 0:
                stroke_rgb = tuple(int(hex_stroke[i:i+2], 16) for i in (0, 2, 4)) + (255,)
                draw.rounded_rectangle(
                    [0, 0, px_w - 1, px_h - 1],
                    radius=radius,
                    fill=fill_rgb,
                    outline=stroke_rgb,
                    width=stroke_width,
                )
            else:
                draw.rounded_rectangle(
                    [0, 0, px_w - 1, px_h - 1],
                    radius=radius,
                    fill=fill_rgb,
                )
            img.save(png_path, "PNG")
            if os.path.isfile(png_path) and os.path.getsize(png_path) > 0:
                logger.info("Rendered rect shape %d to %s (%dx%d, radius=%d, stroke=%d)",
                            idx, png_path, px_w, px_h, radius, stroke_width)
                return png_path
            return None

        # Use FFmpeg lavfi to generate shape PNGs directly (no SVG dependency)
        if shape_type in ("circle", "ellipse"):
            # Draw filled ellipse: create colored canvas, then mask with drawbox for border
            # FFmpeg doesn't have native ellipse, so we use a round approach:
            # 1. Create a transparent canvas
            # 2. Draw the ellipse using the geq filter
            inner_w = max(2, px_w - stroke_width * 2)
            inner_h = max(2, px_h - stroke_width * 2)
            # Use a radial gradient approach: paint pixels inside the ellipse equation
            vf = (
                f"format=rgba,"
                f"geq="
                f"r='if(lte(hypot((X-{px_w/2})/{px_w/2},(Y-{px_h/2})/{px_h/2}),1.0)"
                f",if(lte(hypot((X-{px_w/2})/{inner_w/2},(Y-{px_h/2})/{inner_h/2}),1.0)"
                f",{int(hex_fill[0:2],16)},{int(hex_stroke[0:2],16)}),0)'"
                f":g='if(lte(hypot((X-{px_w/2})/{px_w/2},(Y-{px_h/2})/{px_h/2}),1.0)"
                f",if(lte(hypot((X-{px_w/2})/{inner_w/2},(Y-{px_h/2})/{inner_h/2}),1.0)"
                f",{int(hex_fill[2:4],16)},{int(hex_stroke[2:4],16)}),0)'"
                f":b='if(lte(hypot((X-{px_w/2})/{px_w/2},(Y-{px_h/2})/{px_h/2}),1.0)"
                f",if(lte(hypot((X-{px_w/2})/{inner_w/2},(Y-{px_h/2})/{inner_h/2}),1.0)"
                f",{int(hex_fill[4:6],16)},{int(hex_stroke[4:6],16)}),0)'"
                f":a='if(lte(hypot((X-{px_w/2})/{px_w/2},(Y-{px_h/2})/{px_h/2}),1.0),255,0)'"
            )
            cmd = [
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", f"color=c=black@0:s={px_w}x{px_h}:d=1,format=rgba",
                "-vf", vf,
                "-frames:v", "1",
                "-update", "1",
                png_path,
            ]
        else:
            # Arrow, line, or unknown — simple colored rectangle with optional border
            if stroke_width > 0:
                vf = (
                    f"drawbox=x=0:y=0:w={px_w}:h={px_h}:c=0x{hex_stroke}@1:t=fill,"
                    f"drawbox=x={stroke_width}:y={stroke_width}"
                    f":w={max(2,px_w-stroke_width*2)}:h={max(2,px_h-stroke_width*2)}"
                    f":c=0x{hex_fill}@1:t=fill"
                )
            else:
                vf = f"drawbox=x=0:y=0:w={px_w}:h={px_h}:c=0x{hex_fill}@1:t=fill"
            cmd = [
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", f"color=c=0x{hex_fill}@1:s={px_w}x{px_h}:d=1,format=rgba",
                "-vf", vf,
                "-frames:v", "1",
                "-update", "1",
                png_path,
            ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            proc.kill()
            logger.warning("Shape %d (%s) rendering timed out", idx, shape_type)
            return None

        if proc.returncode != 0:
            stderr_text = stderr.decode(errors="replace")[-1000:] if stderr else "unknown error"
            logger.warning("Shape %d (%s) render failed: %s", idx, shape_type, stderr_text)
            return None

        if os.path.isfile(png_path):
            file_size = os.path.getsize(png_path)
            logger.info("Rendered shape %d (%s) to %s (%dx%d, %d bytes)", idx, shape_type, png_path, px_w, px_h, file_size)
            if file_size == 0:
                logger.warning("Shape %d (%s) produced empty PNG (0 bytes): %s", idx, shape_type, png_path)
                return None
            return png_path
        else:
            logger.warning("Shape %d (%s) produced no output: %s", idx, shape_type, png_path)
            return None
    except Exception as e:
        logger.warning("Failed to render shape %d (%s): %s", idx, shape_type, e)
        return None


def _build_image_overlay_data(
    image_overlays: list,
    job_id: str,
    clip_start: float,
    base_input_idx: int,
    video_out_w: int = 1920,
    video_out_h: int = 1080,
    overlay_compositing_order: list | None = None,
) -> tuple[list[str], str, int, list[str]]:
    """Build FFmpeg input args and overlay filter chain for image overlays.

    Returns:
        (extra_input_args, overlay_filter_chain_suffix, num_valid_images, warnings)

    The overlay_filter_chain_suffix is a semicolon-separated filter segment
    that should be appended to the filter_complex.  It expects the base video
    stream to be labeled ``[vbase]`` and produces a final label ``[vimg]``.

    video_out_w / video_out_h: the output video dimensions (after crop/scale).
    Used to compute image overlay pixel sizes.  NOTE: FFmpeg's ``main_w`` /
    ``main_h`` variables are NOT available inside the ``scale`` filter (only
    in ``overlay`` and ``scale2ref``), so we must pre-compute pixel sizes.

    If no valid images are found, returns ([], "", 0, warnings).
    """
    # ── DEFENSE-IN-DEPTH: Sort image_overlays to match compositing order ──
    # This ensures correct Z-order regardless of how items were appended.
    # Items processed FIRST in the overlay chain render BELOW items processed LATER.
    if overlay_compositing_order and len(image_overlays) > 1:
        _order_ids = [e.get("id") for e in overlay_compositing_order]
        def _z_sort_key(ov):
            iid = ov.get("item_id", "")
            try:
                return _order_ids.index(iid)
            except ValueError:
                return 999999
        image_overlays = sorted(image_overlays, key=_z_sort_key)
        logger.info(
            "image_overlay_data: Z-sorted %d overlays → [%s]",
            len(image_overlays),
            ", ".join(ov.get("item_id", "?")[:15] for ov in image_overlays),
        )
    elif len(image_overlays) > 1:
        # Fallback: no compositing order available — sort by item_id numeric suffix.
        # Item IDs like "item-5", "item-6" have incrementing numbers matching creation order.
        # Earlier-created items (lower number) should render BELOW (processed first).
        def _id_num(ov):
            m = re.search(r'(\d+)$', ov.get("item_id", "") or "")
            return int(m.group(1)) if m else 0
        image_overlays = sorted(image_overlays, key=_id_num)
        logger.info(
            "image_overlay_data: fallback sort by item_id number → [%s]",
            ", ".join(ov.get("item_id", "?")[:15] for ov in image_overlays),
        )

    extra_args: list[str] = []
    valid_overlays: list[tuple[int, dict]] = []  # (input_idx, overlay_dict)
    warnings: list[str] = []

    for i, overlay in enumerate(image_overlays):
        # Skip overlays entirely outside the clip time range
        overlay_end_t = overlay.get("end_time", 0) - clip_start
        if overlay_end_t <= 0:
            warnings.append(f"Image overlay {i+1}: outside clip time range — skipped")
            continue
        src = overlay.get("src", "")
        img_path = _resolve_media_path(src, job_id)
        if not img_path:
            # Log detailed resolution failure for debugging
            basename = os.path.basename(src) if src else ""
            tried_paths = []
            if src:
                tried_paths.append(f"/data/uploads/{job_id}/media/{basename}")
                tried_paths.append(f"/data/uploads/_library/media/{basename}")
            logger.warning(
                "Image overlay %d: src '%s' could not be resolved — SKIPPED. Tried paths: %s",
                i, src[:80] if src else "(empty)", tried_paths,
            )
            warnings.append(f"Image overlay {i+1}: source file not found ({basename or 'unknown'})")
            continue
        else:
            logger.info("Image overlay %d: src '%s' → resolved to %s", i, src[:60] if src else "(empty)", img_path)
        input_idx = base_input_idx + len(extra_args) // 2  # each image adds -i path (2 args)
        extra_args += ["-i", img_path]
        valid_overlays.append((input_idx, overlay))

    if not valid_overlays:
        return [], "", 0, warnings

    # Build overlay filter chain: [vbase][N:v]overlay=...[tmp0]; [tmp0][N+1:v]overlay=...[tmp1]; ...
    fc_parts: list[str] = []
    for i, (input_idx, overlay) in enumerate(valid_overlays):
        x_pct = overlay.get("x", 50) / 100.0
        y_pct = overlay.get("y", 50) / 100.0
        w_pct = overlay.get("width", 30) / 100.0
        h_pct = overlay.get("height", 30) / 100.0
        opacity = overlay.get("opacity", 1.0)
        start_t = overlay.get("start_time", 0) - clip_start
        end_t = overlay.get("end_time", 0) - clip_start

        in_label = "[vbase]" if i == 0 else f"[vtmp{i - 1}]"
        out_label = f"[vtmp{i}]" if i < len(valid_overlays) - 1 else "[vimg]"

        fade_in = overlay.get("fade_in", 0)
        fade_out = overlay.get("fade_out", 0)

        # Compute pixel dimensions for the image overlay.
        # We use pre-computed video output dimensions instead of FFmpeg's
        # main_w/main_h which are NOT available in the scale filter context
        # (only in overlay and scale2ref filters).
        scaled_w = max(2, int(video_out_w * w_pct) // 2 * 2)
        scaled_h = max(2, int(video_out_h * h_pct) // 2 * 2)
        img_scale = (
            f"[{input_idx}:v]"
            f"scale={scaled_w}:{scaled_h},"
            f"format=rgba"
        )

        # Apply image effects (brightness/contrast/saturation/blur/hue/sepia)
        img_brightness = overlay.get("brightness", 0)
        if img_brightness != 0:
            factor = 1 + img_brightness / 100
            if factor >= 1.0:
                inv = 1.0 / factor
                img_scale += (
                    f",colorlevels=rimin=0:rimax={inv:.4f}"
                    f":gimin=0:gimax={inv:.4f}"
                    f":bimin=0:bimax={inv:.4f}"
                )
            else:
                img_scale += (
                    f",colorlevels=romin=0:romax={factor:.4f}"
                    f":gomin=0:gomax={factor:.4f}"
                    f":bomin=0:bomax={factor:.4f}"
                )

        img_eq_parts = []
        img_contrast = overlay.get("contrast", 0)
        if img_contrast != 0:
            img_eq_parts.append(f"contrast={1 + img_contrast / 100:.4f}")
        img_saturation = overlay.get("saturation", 0)
        if img_saturation != 0:
            img_eq_parts.append(f"saturation={1 + img_saturation / 100:.4f}")
        if img_eq_parts:
            img_scale += f",eq={':'.join(img_eq_parts)}"

        img_hue = overlay.get("hue_rotate", 0)
        if img_hue != 0:
            img_scale += f",hue=h={img_hue:.1f}"

        img_blur = overlay.get("blur", 0)
        if img_blur > 0:
            ffmpeg_blur = max(1, int(img_blur * 1.5))
            img_scale += f",boxblur={ffmpeg_blur}:{ffmpeg_blur}"

        img_sepia = overlay.get("sepia", 0)
        if img_sepia > 0:
            s = img_sepia / 100.0
            rr = 1 - s + s * 0.393
            rg = s * 0.769
            rb = s * 0.189
            gr = s * 0.349
            gg = 1 - s + s * 0.686
            gb = s * 0.168
            br = s * 0.272
            bg_ = s * 0.534
            bb = 1 - s + s * 0.131
            img_scale += (
                f",colorchannelmixer={rr:.3f}:{rg:.3f}:{rb:.3f}:0:"
                f"{gr:.3f}:{gg:.3f}:{gb:.3f}:0:"
                f"{br:.3f}:{bg_:.3f}:{bb:.3f}:0"
            )

        # Need format=rgba again after eq filter since eq outputs yuv
        if img_eq_parts or img_hue != 0:
            img_scale += ",format=rgba"

        # Rotation: apply after effects, before opacity/fade
        img_rotation = overlay.get("rotation", 0)
        if img_rotation != 0:
            rad = img_rotation * 3.14159265 / 180.0
            img_scale += (
                f",rotate={rad:.6f}:fillcolor=none"
                f":ow=rotw({rad:.6f}):oh=roth({rad:.6f})"
            )

        if opacity < 1.0:
            img_scale += f",colorchannelmixer=aa={opacity:.3f}"
        # Apply fade in/out on the image alpha channel
        safe_start = max(0, start_t)
        if fade_in > 0:
            img_scale += f",fade=t=in:st={safe_start:.3f}:d={fade_in:.3f}:alpha=1"
        if fade_out > 0 and end_t > start_t:
            fo_start = end_t - fade_out
            img_scale += f",fade=t=out:st={max(0, fo_start):.3f}:d={fade_out:.3f}:alpha=1"
        img_scale += f"[img{i}]"

        # Position: x_pct/y_pct are center coordinates, convert to top-left for overlay
        # main_w/main_h ARE valid inside the overlay filter
        x_expr = f"main_w*{x_pct:.4f}-overlay_w/2"
        y_expr = f"main_h*{y_pct:.4f}-overlay_h/2"

        overlay_filter = f"{in_label}[img{i}]overlay={x_expr}:{y_expr}"
        if end_t > start_t:
            overlay_filter += f":enable='between(t,{max(0, start_t):.3f},{end_t:.3f})'"
        overlay_filter += out_label

        fc_parts.append(img_scale)
        fc_parts.append(overlay_filter)

    return extra_args, ";".join(fc_parts), len(valid_overlays), warnings


def _build_single_drawtext(overlay: dict, clip_start: float, video_out_w: int = 1920, video_out_h: int = 1080) -> str | None:
    """Build a single drawtext filter string for one text overlay item.

    Returns the drawtext filter string (without stream labels) or None if skipped.

    Font size, outline width, background padding, and shadow offsets are scaled
    relative to the output resolution so the text appears the same proportional
    size as in the frontend preview container (~half the output resolution).
    """
    # Scale factor: preview container is roughly half the output frame size.
    _PREVIEW_REF = 960
    res_scale = max(video_out_w, video_out_h) / _PREVIEW_REF

    text = overlay.get("text", "")
    for ch in ('\\', "'", ':', '%', '{', '}', ';', '[', ']'):
        text = text.replace(ch, f'\\{ch}')
    if not text:
        return None

    x_pct = overlay.get("x", 50) / 100.0
    y_pct = overlay.get("y", 50) / 100.0
    # Scale font_size from preview-canvas pixels to output-video pixels
    font_size = max(8, int(round(overlay.get("font_size", 48) * res_scale)))
    font_color = overlay.get("font_color", "#FFFFFF")
    font_family = overlay.get("font_family", "sans-serif")
    opacity = overlay.get("opacity", 1.0)
    start_t = overlay.get("start_time", 0) - clip_start
    end_t = overlay.get("end_time", 0) - clip_start
    if end_t <= 0:
        return None
    font_weight = overlay.get("font_weight", 400)
    if isinstance(font_weight, str):
        font_weight = 700 if font_weight.lower() == "bold" else 400
    else:
        font_weight = int(round(font_weight))
    logger.info("_build_single_drawtext: font_family=%r, font_weight=%d, resolving font path...", font_family, font_weight)
    font_path = _resolve_font_path(font_family, font_weight=font_weight)
    logger.info("_build_single_drawtext: resolved font_path=%s", font_path)
    if not os.path.isfile(font_path):
        font_path = _DEFAULT_FONT_BOLD if font_weight >= 600 and os.path.isfile(_DEFAULT_FONT_BOLD) else _DEFAULT_FONT
        logger.warning("_build_single_drawtext: font not found, using fallback=%s", font_path)
        if not os.path.isfile(font_path):
            return None

    # CSS -webkit-text-stroke: Npx with paint-order:stroke fill shows N/2
    # visible per side (fill covers inner half).  FFmpeg borderw renders
    # the full value outward.  So: borderw = outlineWidth/2 * res_scale
    # to match the visible stroke thickness in the preview.
    outline_width = max(0, int(round(overlay.get("outline_width", 0) * res_scale / 2)))
    outline_color = overlay.get("outline_color", "#000000")
    fade_in = overlay.get("fade_in", 0)
    fade_out = overlay.get("fade_out", 0)

    safe_start = max(0, start_t)
    has_fade = (fade_in > 0 or fade_out > 0) and end_t > start_t
    if has_fade:
        alpha_parts = [f"{opacity:.2f}"]
        if fade_in > 0:
            alpha_parts.append(f"if(lt(t-{safe_start:.3f},{fade_in:.3f}),(t-{safe_start:.3f})/{fade_in:.3f},1)")
        if fade_out > 0:
            fo_start = end_t - fade_out
            alpha_parts.append(f"if(gt(t,{fo_start:.3f}),({end_t:.3f}-t)/{fade_out:.3f},1)")
        alpha_expr = "*".join(alpha_parts)
    else:
        alpha_expr = f"{opacity:.2f}"

    animation = overlay.get("animation", "")
    y_expr = f"h*{y_pct:.4f}-th/2"
    fontsize_expr = str(font_size)

    slide_offset = int(round(30 * res_scale))
    if animation == "slide-up" and end_t > start_t:
        anim_dur = 0.5
        y_expr = (
            f"h*{y_pct:.4f}-th/2"
            f"+if(lt(t-{safe_start:.3f},{anim_dur})"
            f",(1-(t-{safe_start:.3f})/{anim_dur})*{slide_offset},0)"
        )
    elif animation == "pop" and end_t > start_t:
        anim_dur = 0.3
        fontsize_expr = (
            f"if(lt(t-{safe_start:.3f},{anim_dur})"
            f",{font_size}*(0.5+0.5*(t-{safe_start:.3f})/{anim_dur})"
            f",{font_size})"
        )

    text_align = overlay.get("text_align", "center")
    if text_align == "left":
        x_expr = f"w*{x_pct:.4f}"
    elif text_align == "right":
        x_expr = f"w*{x_pct:.4f}-tw"
    else:
        x_expr = f"w*{x_pct:.4f}-tw/2"

    dt = (
        f"drawtext=text='{text}'"
        f":x={x_expr}"
        f":y={y_expr}"
        f":fontsize='{fontsize_expr}'"
        f":fontcolor={font_color}"
        f":fontfile={font_path}"
        f":alpha='{alpha_expr}'"
    )
    if outline_width > 0:
        dt += f":borderw={outline_width}:bordercolor={outline_color}"

    # Scale shadow offsets for output resolution
    shadow_x = int(round(overlay.get("shadow_offset_x", 0) * res_scale))
    shadow_y = int(round(overlay.get("shadow_offset_y", 0) * res_scale))
    shadow_color = overlay.get("shadow_color", "")
    if shadow_x or shadow_y or shadow_color:
        hex_shadow = "#000000"
        if shadow_color.startswith("rgba("):
            parts_c = shadow_color.replace("rgba(", "").replace(")", "").split(",")
            if len(parts_c) >= 3:
                try:
                    hex_shadow = "#{:02x}{:02x}{:02x}".format(
                        int(parts_c[0].strip()), int(parts_c[1].strip()), int(parts_c[2].strip())
                    )
                except ValueError:
                    pass
        elif shadow_color.startswith("#"):
            hex_shadow = shadow_color
        dt += f":shadowcolor={hex_shadow}:shadowx={shadow_x}:shadowy={shadow_y}"

    # NOTE: FFmpeg drawtext box=1 only supports rectangular backgrounds.
    # Frontend rounded corners (bgRadius) are approximated as sharp rectangles.
    # Shadow blur (shadowBlur) is also not supported — only offset + color.
    bg_color = overlay.get("background_color")
    bg_opacity_pct = overlay.get("bg_opacity", 0)
    # Scale bg_padding for output resolution
    bg_padding = int(round(overlay.get("bg_padding", 8) * res_scale))
    if bg_color and bg_opacity_pct > 0:
        bg_alpha = bg_opacity_pct / 100.0
        dt += f":box=1:boxcolor={bg_color}@{bg_alpha:.2f}:boxborderw={bg_padding}"
    elif bg_color:
        dt += f":box=1:boxcolor={bg_color}@0.5:boxborderw={bg_padding}"

    if end_t > start_t:
        dt += f":enable='between(t,{safe_start:.3f},{end_t:.3f})'"

    return dt


def _build_unified_overlay_chain(
    compositing_order: list,
    text_overlays: list,
    image_overlays: list,
    clip_start: float,
    base_input_idx: int,
    video_out_w: int,
    video_out_h: int,
    job_id: str,
) -> tuple[list[str], str, list[str]]:
    """Build a single FFmpeg filter chain that interleaves text and image overlays
    in the correct compositing order based on track position.

    Returns: (extra_input_args, filter_chain_suffix, warnings)

    The filter_chain_suffix expects the base video stream to be labeled ``[vbase]``
    and produces a final label ``[vcomp]``.
    """
    # Build lookup maps: item_id → overlay data
    text_by_id = {}
    for t in text_overlays:
        iid = t.get("item_id")
        if iid:
            text_by_id[iid] = t

    image_by_id = {}
    for im in image_overlays:
        iid = im.get("item_id")
        if iid:
            image_by_id[iid] = im

    logger.info(
        "Unified chain: %d compositing entries, text_by_id=%s, image_by_id=%s",
        len(compositing_order),
        list(text_by_id.keys())[:8],
        list(image_by_id.keys())[:8],
    )

    extra_args: list[str] = []
    fc_parts: list[str] = []
    warnings: list[str] = []
    current_label = "[vbase]"
    step = 0

    sorted_order = sorted(compositing_order, key=lambda e: e.get("compositing_priority", 0))

    for entry in sorted_order:
        item_type = entry.get("type")
        item_id = entry.get("id")

        if item_type == "text":
            if not _HAS_DRAWTEXT:
                logger.info("Unified chain: skipping text '%s' (drawtext unavailable, rendered via ASS)", str(item_id)[:20])
                continue
            if item_id not in text_by_id:
                logger.warning("Unified chain: text '%s' NOT in text_by_id — SKIPPED", item_id)
                warnings.append(f"text '{item_id}' not found")
                continue
            overlay = text_by_id[item_id]
            dt_filter = _build_single_drawtext(overlay, clip_start, video_out_w, video_out_h)
            if dt_filter:
                out_label = f"[vcmp{step}]"
                fc_parts.append(f"{current_label}{dt_filter}{out_label}")
                current_label = out_label
                logger.info("Unified chain step %d: text '%s' composited", step, str(item_id)[:12])
                step += 1

        elif item_type in ("image", "shape"):
            if item_id not in image_by_id:
                logger.warning(
                    "Unified chain: %s '%s' NOT in image_by_id — SKIPPED (keys: %s)",
                    item_type, item_id, list(image_by_id.keys())[:8],
                )
                warnings.append(f"{item_type} '{item_id}' not found")
                continue
            overlay = image_by_id[item_id]
            src = overlay.get("src", "")
            img_path = _resolve_media_path(src, job_id)
            if not img_path:
                warnings.append(f"Image/shape overlay '{item_id}': source not found — skipped")
                continue

            input_idx = base_input_idx + len(extra_args) // 2
            extra_args += ["-i", img_path]

            x_pct = overlay.get("x", 50) / 100.0
            y_pct = overlay.get("y", 50) / 100.0
            w_pct = overlay.get("width", 30) / 100.0
            h_pct = overlay.get("height", 30) / 100.0
            opacity = overlay.get("opacity", 1.0)
            start_t = overlay.get("start_time", 0) - clip_start
            end_t = overlay.get("end_time", 0) - clip_start
            fade_in = overlay.get("fade_in", 0)
            fade_out = overlay.get("fade_out", 0)

            scaled_w = max(2, int(video_out_w * w_pct) // 2 * 2)
            scaled_h = max(2, int(video_out_h * h_pct) // 2 * 2)
            img_scale = (
                f"[{input_idx}:v]"
                f"scale={scaled_w}:{scaled_h},"
                f"format=rgba"
            )

            # Apply image effects (brightness/contrast/saturation/blur/hue/sepia)
            u_brightness = overlay.get("brightness", 0)
            if u_brightness != 0:
                factor = 1 + u_brightness / 100
                if factor >= 1.0:
                    inv = 1.0 / factor
                    img_scale += (
                        f",colorlevels=rimin=0:rimax={inv:.4f}"
                        f":gimin=0:gimax={inv:.4f}"
                        f":bimin=0:bimax={inv:.4f}"
                    )
                else:
                    img_scale += (
                        f",colorlevels=romin=0:romax={factor:.4f}"
                        f":gomin=0:gomax={factor:.4f}"
                        f":bomin=0:bomax={factor:.4f}"
                    )

            u_eq_parts = []
            u_contrast = overlay.get("contrast", 0)
            if u_contrast != 0:
                u_eq_parts.append(f"contrast={1 + u_contrast / 100:.4f}")
            u_saturation = overlay.get("saturation", 0)
            if u_saturation != 0:
                u_eq_parts.append(f"saturation={1 + u_saturation / 100:.4f}")
            if u_eq_parts:
                img_scale += f",eq={':'.join(u_eq_parts)}"

            u_hue = overlay.get("hue_rotate", 0)
            if u_hue != 0:
                img_scale += f",hue=h={u_hue:.1f}"

            u_blur = overlay.get("blur", 0)
            if u_blur > 0:
                ffmpeg_blur = max(1, int(u_blur * 1.5))
                img_scale += f",boxblur={ffmpeg_blur}:{ffmpeg_blur}"

            u_sepia = overlay.get("sepia", 0)
            if u_sepia > 0:
                s = u_sepia / 100.0
                rr = 1 - s + s * 0.393
                rg = s * 0.769
                rb = s * 0.189
                gr = s * 0.349
                gg = 1 - s + s * 0.686
                gb = s * 0.168
                br = s * 0.272
                bg_ = s * 0.534
                bb = 1 - s + s * 0.131
                img_scale += (
                    f",colorchannelmixer={rr:.3f}:{rg:.3f}:{rb:.3f}:0:"
                    f"{gr:.3f}:{gg:.3f}:{gb:.3f}:0:"
                    f"{br:.3f}:{bg_:.3f}:{bb:.3f}:0"
                )

            # Re-convert to rgba after eq/hue filters (output yuv)
            if u_eq_parts or u_hue != 0:
                img_scale += ",format=rgba"

            # Rotation
            u_rotation = overlay.get("rotation", 0)
            if u_rotation != 0:
                rad = u_rotation * 3.14159265 / 180.0
                img_scale += (
                    f",rotate={rad:.6f}:fillcolor=none"
                    f":ow=rotw({rad:.6f}):oh=roth({rad:.6f})"
                )

            if opacity < 1.0:
                img_scale += f",colorchannelmixer=aa={opacity:.3f}"
            safe_start = max(0, start_t)
            if fade_in > 0:
                img_scale += f",fade=t=in:st={safe_start:.3f}:d={fade_in:.3f}:alpha=1"
            if fade_out > 0 and end_t > start_t:
                fo_start = end_t - fade_out
                img_scale += f",fade=t=out:st={max(0, fo_start):.3f}:d={fade_out:.3f}:alpha=1"
            img_scale += f"[uimg{step}]"

            x_expr = f"main_w*{x_pct:.4f}-overlay_w/2"
            y_expr = f"main_h*{y_pct:.4f}-overlay_h/2"

            out_label = f"[vcmp{step}]"
            overlay_filter = f"{current_label}[uimg{step}]overlay={x_expr}:{y_expr}"
            if end_t > start_t:
                overlay_filter += f":enable='between(t,{max(0, start_t):.3f},{end_t:.3f})'"
            overlay_filter += out_label

            fc_parts.append(img_scale)
            fc_parts.append(overlay_filter)
            current_label = out_label
            logger.info("Unified chain step %d: %s '%s' composited", step, item_type, str(item_id)[:12])
            step += 1

        else:
            logger.warning("Unified chain: unknown type '%s' for '%s'", item_type, item_id)

    logger.info(
        "Unified chain result: %d/%d items processed, %d skipped",
        step, len(sorted_order), len(sorted_order) - step,
    )

    if step == 0:
        return [], "", warnings

    # Rename final label to [vcomp]
    fc_parts[-1] = fc_parts[-1].rsplit(current_label, 1)[0] + "[vcomp]"
    logger.info(
        "Unified compositing chain: %d steps, %d image inputs",
        step, len(extra_args) // 2,
    )

    return extra_args, ";".join(fc_parts), warnings


async def export_clip(
    job_id: str,
    video_path: str,
    start: float,
    end: float,
    clip_id: int,
    clip_title: str | None = None,
    aspect_ratio: str | None = None,
    subtitles_enabled: bool = False,
    subtitle_settings: dict | None = None,
    transcript: list | None = None,
    video_width: int = 1920,
    video_height: int = 1080,
    subject_x: int = 50,
    subject_scenes: list | None = None,
    scene_cut_timestamps: list[float] | None = None,
    progress_callback=None,
    cancel_event: "asyncio.Event | None" = None,
    export_quality: str = "1080p",
    volume: float = 1.0,
    speed: float = 1.0,
    segments: list | None = None,
    global_subtitles_enabled: bool | None = None,
    video_effects: dict | None = None,
    text_overlays: list | None = None,
    image_overlays: list | None = None,
    shape_overlays: list | None = None,
    audio_overlays: list | None = None,
    overlay_compositing_order: list | None = None,
    layout_mode: str = "auto",
    pip_position: str = "bottom_right",
    pip_size_pct: float = 25.0,
    face_registry_data: dict | None = None,
    layout_timeline_data: list | None = None,
    hook_text: str = "",
    frontend_subject_keyframes: list[dict] | None = None,
) -> str:
    """Export a clip from video using FFmpeg.

    When aspect_ratio or subtitles are specified, re-encodes with filters.
    Otherwise uses stream copy for speed.

    subject_scenes: overlapping SceneDescription objects (or dicts with
        timestamp + subject_x).  When provided, enables dynamic crop that
        follows the subject through the clip.  Falls back to static
        subject_x if None or insufficient data.

    volume: Audio gain (0.0 to 2.0, default 1.0). Applied via -af volume filter.
    speed: Playback speed (0.25 to 4.0, default 1.0). Applied via setpts + atempo.

    progress_callback: optional async callable(message: str) for status updates.
    """
    # Version sentinel — proves which code version the container is running.
    _EXPORT_VERSION = "z-order-fix-v3-2026-03-12"
    logger.info("=== EXPORT START clip %s — code version: %s ===", clip_id, _EXPORT_VERSION)

    async def _notify(msg: str):
        if progress_callback:
            try:
                await progress_callback(msg)
            except Exception:
                pass

    clip_dur = end - start
    output_dir = f"/data/outputs/{job_id}/clips"
    os.makedirs(output_dir, exist_ok=True)

    # Quality tag for filename
    quality_tag = export_quality.upper() if export_quality else "1080P"

    # Use clip title for filename if provided, otherwise fall back to ID-based naming
    if clip_title:
        # Sanitize title for filesystem: replace unsafe chars, collapse whitespace
        safe_title = re.sub(r'[<>:"/\\|?*]', '', clip_title)
        safe_title = re.sub(r'\s+', ' ', safe_title).strip()
        if not safe_title:
            safe_title = f"clip_{clip_id}"
        # Truncate to avoid excessively long filenames
        if len(safe_title) > 120:
            safe_title = safe_title[:120].rstrip()
        output_path = os.path.join(output_dir, f"[{quality_tag}] {safe_title}.mp4")
    else:
        output_path = os.path.join(
            output_dir, f"[{quality_tag}] clip_{clip_id}_{int(start)}_{int(end)}.mp4"
        )

    # Delete any pre-existing output file to ensure we never serve a stale
    # export from a previous run (same clip title → same filename).
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
            logger.info("Removed stale output file: %s", output_path)
        except OSError as e:
            logger.warning("Could not remove stale output: %s", e)

    def _check_cancel():
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError("Export cancelled by user")

    ass_path = None
    subtitle_force_style = ""
    # Determine if quality requires resolution scaling
    quality_target_h = QUALITY_MAX_HEIGHT.get(export_quality, 1080)
    needs_quality_scale = (video_height != quality_target_h)
    has_speed = abs(speed - 1.0) > 0.001
    has_volume = abs(volume - 1.0) > 0.001
    has_segments = bool(segments) and len(segments) > 0
    # Detect per-segment speed overrides (different from global speed)
    has_seg_speed = False
    if has_segments:
        for seg in segments:
            if abs(seg.get("speed", 1.0) - 1.0) > 0.001:
                has_seg_speed = True
                break
    has_video_effects = bool(video_effects) and any((
        video_effects.get("brightness", 0) != 0,
        video_effects.get("contrast", 0) != 0,
        video_effects.get("saturation", 0) != 0,
        video_effects.get("blur", 0) != 0,
        video_effects.get("hue_rotate", 0) != 0,
        video_effects.get("sepia", 0) != 0,
        video_effects.get("opacity", 1.0) != 1.0,
        video_effects.get("position_x", 50) != 50,
        video_effects.get("position_y", 50) != 50,
        video_effects.get("width", 100) != 100,
        video_effects.get("height", 100) != 100,
        video_effects.get("rotation", 0) != 0,
        video_effects.get("fade_in", 0) > 0,
        video_effects.get("fade_out", 0) > 0,
    ))
    has_text_overlays = bool(text_overlays) and len(text_overlays) > 0
    has_image_overlays = bool(image_overlays) and len(image_overlays) > 0
    has_shape_overlays = bool(shape_overlays) and len(shape_overlays) > 0
    has_audio_overlays = bool(audio_overlays) and len(audio_overlays) > 0

    # Render shape overlays as temporary PNGs and merge into image_overlays
    _shape_temp_files: list[str] = []
    _shape_warnings: list[str] = []
    if has_shape_overlays:
        if image_overlays is None:
            image_overlays = []
        shape_tmp_dir = os.path.join(output_dir, "_shapes")
        os.makedirs(shape_tmp_dir, exist_ok=True)
        for si, shape in enumerate(shape_overlays):
            png_path = await _render_shape_to_png(shape, video_width, video_height, shape_tmp_dir, si)
            if png_path:
                _shape_temp_files.append(png_path)
                # Convert shape to image overlay format for the image pipeline
                image_overlays.append({
                    "item_id": shape.get("item_id"),
                    "src": png_path,
                    "x": shape.get("x", 50),
                    "y": shape.get("y", 50),
                    "width": shape.get("width", 20),
                    "height": shape.get("height", 20),
                    "start_time": shape.get("start_time", 0),
                    "end_time": shape.get("end_time", 0),
                    "opacity": shape.get("opacity", 1.0),
                    "fade_in": shape.get("fade_in", 0),
                    "fade_out": shape.get("fade_out", 0),
                })
                logger.info("Shape %d (%s) rendered to PNG: %s", si, shape.get("shape_type"), png_path)
            else:
                logger.warning("Shape %d (%s) failed to render, skipping", si, shape.get("shape_type"))
                _shape_warnings.append(f"Shape {si+1} ({shape.get('shape_type', 'unknown')}): failed to render — skipped")
        # Recompute has_image_overlays after merging shapes
        has_image_overlays = bool(image_overlays) and len(image_overlays) > 0

    # ── CRITICAL: Re-sort image_overlays to match compositing order ──
    # Shape overlays were appended to the END of image_overlays above,
    # which always puts them on top regardless of track position or
    # creation order.  Re-sort so the array order matches the frontend's
    # overlay_compositing_order.  This fixes the legacy path AND serves
    # as a safety net if the unified path falls through.
    if image_overlays and overlay_compositing_order:
        _comp_order_ids = [e.get("id") for e in overlay_compositing_order]
        def _comp_sort_key(overlay_dict):
            iid = overlay_dict.get("item_id", "")
            try:
                return _comp_order_ids.index(iid)
            except ValueError:
                return 999999  # items not in compositing order go last
        image_overlays.sort(key=_comp_sort_key)
        logger.info(
            ">>> Z-ORDER FIX ACTIVE <<< Re-sorted %d image_overlays by compositing order: %s",
            len(image_overlays),
            [im.get("item_id", "?")[:20] for im in image_overlays],
        )
    elif image_overlays and len(image_overlays) > 1 and not overlay_compositing_order:
        # No compositing order available (exported from ViralClips/Analysis page).
        # Fall back to sorting by item_id numeric suffix (creation order).
        # Items with higher numbers were created later and should render on top.
        def _fallback_sort_key(ov):
            m = re.search(r'(\d+)$', ov.get("item_id", "") or "")
            return int(m.group(1)) if m else 0
        image_overlays.sort(key=_fallback_sort_key)
        logger.info(
            ">>> Z-ORDER FALLBACK SORT <<< No compositing order — sorted %d image_overlays by item_id: %s",
            len(image_overlays),
            [im.get("item_id", "?")[:20] for im in image_overlays],
        )

    needs_filters = bool(aspect_ratio) or subtitles_enabled or needs_quality_scale or has_speed or has_volume or has_segments or has_seg_speed or has_video_effects or has_text_overlays or has_image_overlays or has_audio_overlays

    # Diagnostic: log the needs_filters decision with all contributing flags
    logger.info(
        "EXPORT DECISION clip %s: needs_filters=%s — reasons: aspect=%s subs=%s "
        "quality_scale=%s speed=%s volume=%s segments=%s seg_speed=%s "
        "video_effects=%s text_overlays=%s image_overlays=%s shape_overlays=%s audio_overlays=%s",
        clip_id, needs_filters,
        bool(aspect_ratio), subtitles_enabled, needs_quality_scale,
        has_speed, has_volume, has_segments, has_seg_speed,
        has_video_effects, has_text_overlays, has_image_overlays,
        has_shape_overlays, has_audio_overlays,
    )

    filter_parts = []
    if aspect_ratio:
        filter_parts.append(f"crop to {aspect_ratio}")
    if subtitles_enabled:
        filter_parts.append("burn subtitles")
    if needs_quality_scale:
        filter_parts.append(f"scale to {export_quality}")
    if has_seg_speed:
        seg_speeds = set(seg.get("speed", 1.0) for seg in segments)
        filter_parts.append(f"per-segment speed ({len(seg_speeds)} unique)")
    elif has_speed:
        filter_parts.append(f"speed {speed}x")
    if has_volume:
        filter_parts.append(f"volume {volume:.0%}")
    if has_segments:
        filter_parts.append(f"{len(segments)} segment overrides")
    if has_video_effects:
        filter_parts.append("video effects (brightness/contrast/etc)")
    if has_text_overlays:
        filter_parts.append(f"{len(text_overlays)} text overlay(s)")
    if has_image_overlays:
        filter_parts.append(f"{len(image_overlays)} image overlay(s)")
    if has_shape_overlays:
        filter_parts.append(f"{len(shape_overlays)} shape overlay(s)")
    if has_audio_overlays:
        filter_parts.append(f"{len(audio_overlays)} audio overlay(s)")
    filter_desc = " + ".join(filter_parts) if filter_parts else "stream copy"

    await _notify(f"Preparing clip {clip_id} ({clip_dur:.1f}s) — {filter_desc}")

    try:
        # Log all export settings for debugging subtitle burn-in
        logger.info(
            "Export clip %s: start=%.1f end=%.1f aspect=%s subs=%s "
            "video=%dx%d subject_x=%d scenes=%d quality=%s",
            clip_id, start, end, aspect_ratio or "original",
            subtitles_enabled, video_width, video_height,
            subject_x, len(subject_scenes or []), export_quality,
        )
        if subtitles_enabled and subtitle_settings:
            logger.info(
                "Subtitle settings: font=%s size=%s weight=%s color=%s "
                "pos=%s bg=%s outline=%s/%s/%s speakers=%s labels=%s "
                "max_width=%s offset_v=%s max_words=%s active_word=%s",
                subtitle_settings.get("font"),
                subtitle_settings.get("size"),
                subtitle_settings.get("font_weight"),
                subtitle_settings.get("font_color"),
                subtitle_settings.get("position"),
                subtitle_settings.get("background_enabled"),
                subtitle_settings.get("outline_color"),
                subtitle_settings.get("outline_opacity"),
                subtitle_settings.get("outline_width"),
                subtitle_settings.get("use_speaker_colors"),
                subtitle_settings.get("show_speaker_labels"),
                subtitle_settings.get("max_width"),
                subtitle_settings.get("offset_v"),
                subtitle_settings.get("max_words"),
                subtitle_settings.get("active_word_enabled"),
            )

        # Log overlay track QA summary
        _overlay_summary = []
        if has_text_overlays:
            _overlay_summary.append(f"text={len(text_overlays)}")
            for ti, to in enumerate(text_overlays):
                logger.info("  Text overlay %d: text=%r, pos=(%.0f%%,%.0f%%), time=%.1f-%.1f, font_size=%s, font_weight=%s, font_family=%s",
                    ti, (to.get("text", ""))[:40], to.get("x", 50), to.get("y", 50),
                    to.get("start_time", 0), to.get("end_time", 0), to.get("font_size", 48),
                    to.get("font_weight", 400), to.get("font_family", "sans-serif"))
        if has_image_overlays:
            _overlay_summary.append(f"image={len(image_overlays)}")
            for ii, io_item in enumerate(image_overlays):
                logger.info("  Image overlay %d: src=%s, pos=(%.0f%%,%.0f%%), size=(%.0f%%x%.0f%%), time=%.1f-%.1f",
                    ii, os.path.basename(io_item.get("src", ""))[:60], io_item.get("x", 50), io_item.get("y", 50),
                    io_item.get("width", 30), io_item.get("height", 30),
                    io_item.get("start_time", 0), io_item.get("end_time", 0))
        if has_shape_overlays:
            _overlay_summary.append(f"shape={len(shape_overlays)}")
            for si_q, so in enumerate(shape_overlays):
                logger.info("  Shape overlay %d: type=%s, pos=(%.0f%%,%.0f%%), size=(%.0f%%x%.0f%%), time=%.1f-%.1f, fill=%s",
                    si_q, so.get("shape_type", "rectangle"), so.get("x", 50), so.get("y", 50),
                    so.get("width", 20), so.get("height", 20),
                    so.get("start_time", 0), so.get("end_time", 0), so.get("fill_color", "?"))
        if has_audio_overlays:
            _overlay_summary.append(f"audio={len(audio_overlays)}")
        if _overlay_summary:
            logger.info("Overlay QA for clip %s: %s", clip_id, ", ".join(_overlay_summary))
        else:
            logger.info("Overlay QA for clip %s: no overlay items", clip_id)

        # Validate text overlay parity
        if has_text_overlays:
            parity_warnings = _validate_text_overlay_parity(
                text_overlays, video_width, video_height
            )
            for pw in parity_warnings:
                logger.warning("TEXT OVERLAY PARITY: %s", pw)

        # Generate ASS subtitle file if subtitles are enabled
        if subtitles_enabled and transcript:
            settings = subtitle_settings or {}
            transcript_segments = [
                TranscriptSegment(**s) if isinstance(s, dict) else s
                for s in transcript
            ]

            # Determine output dimensions for subtitle positioning.
            # Must match the actual output resolution (quality-aware) so
            # subtitle font sizes and margins are correct in the final video.
            #
            # When the source has baked-in black bars (pillarboxing), use
            # the detected content dimensions instead of the container size.
            # Otherwise subtitles span the full frame including black bars.
            eff_w, eff_h = video_width, video_height
            crop_result = _detect_crop(video_path, start=start, duration=3.0)
            if crop_result:
                cw, ch, _, _ = crop_result
                w_rem = (video_width - cw) / video_width * 100 if video_width > 0 else 0
                h_rem = (video_height - ch) / video_height * 100 if video_height > 0 else 0
                if w_rem >= 4 or h_rem >= 4:
                    eff_w = cw - (cw % 2)
                    eff_h = ch - (ch % 2)

            dims_table = ASPECT_RATIO_DIMS_BY_QUALITY.get(export_quality, ASPECT_RATIO_DIMS)
            if aspect_ratio and aspect_ratio in dims_table:
                out_w, out_h = dims_table[aspect_ratio]
            elif not aspect_ratio:
                # No aspect ratio — use quality-scaled height, derive width
                quality_h = QUALITY_MAX_HEIGHT.get(export_quality, 1080)
                if eff_h != quality_h and eff_h > 0:
                    scale_factor = quality_h / eff_h
                    out_w = int(eff_w * scale_factor)
                    out_w = out_w - (out_w % 2)  # ensure even
                    out_h = quality_h
                else:
                    out_w, out_h = eff_w, eff_h
            else:
                out_w, out_h = eff_w, eff_h

            ass_content = generate_ass(
                segments=transcript_segments,
                start_time=start,
                end_time=end,
                font=settings.get("font", "DM Sans"),
                font_size=settings.get("size", "medium"),
                font_weight=settings.get("font_weight", "bold"),
                font_color=settings.get("font_color", "#FFFFFF"),
                position=settings.get("position", "bottom"),
                speaker_colors=settings.get("speaker_colors"),
                use_speaker_colors=settings.get("use_speaker_colors", True),
                video_width=out_w,
                video_height=out_h,
                background_enabled=settings.get("background_enabled", False),
                background_color=settings.get("background_color", "#000000"),
                background_opacity=settings.get("background_opacity", 75),
                background_radius=settings.get("background_radius", 0),
                outline_color=settings.get("outline_color", "#000000"),
                outline_opacity=settings.get("outline_opacity", 100),
                outline_width=settings.get("outline_width", 2),
                content_inset_v=0,
                content_inset_h=0,
                show_speaker_labels=settings.get("show_speaker_labels", False),
                max_width_pct=settings.get("max_width", 90),
                offset_v_pct=settings.get("offset_v", 4),
                max_words=settings.get("max_words", 0),
                active_word_enabled=settings.get("active_word_enabled", False),
                active_word_color=settings.get("active_word_color", "#FFD700"),
                active_word_outline_color=settings.get("active_word_outline_color", "#000000"),
                active_word_bg_color=settings.get("active_word_bg_color", "#000000"),
                active_word_bg_opacity=settings.get("active_word_bg_opacity", 0),
                active_word_bg_radius=settings.get("active_word_bg_radius", 4),
                hook_text=hook_text,
            )

            if ass_content:
                # Filter ASS dialogue lines based on per-segment subtitle overrides.
                # Two modes:
                # 1. Global subs ON: remove dialogue in segments with subtitles_enabled=false
                # 2. Global subs OFF (enabled by segment overrides): keep ONLY dialogue
                #    in segments with subtitles_enabled=true
                if has_segments:
                    # Use the original global toggle to determine base behavior.
                    # Falls back to subtitles_enabled if not provided (backwards compat).
                    global_subs = global_subtitles_enabled if global_subtitles_enabled is not None else subtitles_enabled
                    subs_on_segs = [seg for seg in segments if seg.get("subtitles_enabled", True)]
                    subs_off_segs = [seg for seg in segments if not seg.get("subtitles_enabled", True)]

                    if not global_subs and subs_on_segs:
                        # Global subs off but some segments have subs on —
                        # keep only dialogue within subs-on segment ranges
                        subs_on_ranges = [
                            (max(0, seg["start"] - start), min(end - start, seg["end"] - start))
                            for seg in subs_on_segs
                        ]
                        ass_content = _filter_ass_keep_only_ranges(ass_content, subs_on_ranges)
                    elif subs_off_segs:
                        # Global subs on — remove dialogue in subs-off segment ranges
                        subs_off_ranges = [
                            (seg["start"] - start, seg["end"] - start)
                            for seg in subs_off_segs
                        ]
                        ass_content = _filter_ass_by_segments(ass_content, subs_off_ranges)

                ass_path = os.path.join(output_dir, f"clip_{clip_id}_sub.ass")
                # Remove any stale ASS file from a previous export to ensure
                # the fresh content is always used (prevents caching issues).
                if os.path.exists(ass_path):
                    os.remove(ass_path)
                with open(ass_path, "w", encoding="utf-8") as f:
                    f.write(ass_content)
                logger.info("ASS file written: %s (%d bytes)", ass_path, len(ass_content))

                # ── Text overlays via ASS when drawtext unavailable ──
                if has_text_overlays and not _HAS_DRAWTEXT:
                    logger.info(
                        "drawtext unavailable — rendering %d text overlay(s) via ASS for clip %s",
                        len(text_overlays), clip_id,
                    )
                    from backend.services.ass_generator import append_text_overlays_to_ass
                    with open(ass_path, "r", encoding="utf-8") as f:
                        _ass = f.read()
                    _ass = append_text_overlays_to_ass(
                        _ass, text_overlays,
                        clip_start=start,
                        video_out_w=video_width,
                        video_out_h=video_height,
                    )
                    with open(ass_path, "w", encoding="utf-8") as f:
                        f.write(_ass)
                    has_text_overlays = False  # Prevent drawtext from being added to filter chain

                # ── Diagnostic: verify two-layer architecture ──
                # Log whether the ASS uses the two-layer approach for active
                # word mode so we can confirm the black-bar fix is active.
                if settings.get("active_word_enabled"):
                    layer0_count = ass_content.count("Dialogue: 0,")
                    layer1_count = ass_content.count("Dialogue: 1,")
                    has_nobord = "\\bord0\\shad0\\3a&HFF&" in ass_content
                    logger.info(
                        "ASS two-layer check for clip %s: Layer0=%d events, Layer1=%d events, "
                        "has_nobord_tag=%s (expected: both layers populated, nobord=True)",
                        clip_id, layer0_count, layer1_count, has_nobord,
                    )
                    if layer0_count == 0 or layer1_count == 0:
                        logger.warning(
                            "ASS two-layer architecture NOT active for clip %s — "
                            "black bars may still appear. Layer0=%d, Layer1=%d",
                            clip_id, layer0_count, layer1_count,
                        )

                # NOTE: force_style is intentionally NOT used.
                #
                # The old approach extracted outline/border settings from the
                # first ASS Style line and passed them as force_style to the
                # subtitles filter.  This caused two problems:
                #
                # 1. force_style applies GLOBALLY to all styles, overriding
                #    per-speaker PrimaryColour when use_speaker_colors=True
                #    (it extracts OutlineColour from only the first style).
                #
                # 2. force_style modifies the style objects via
                #    ass_process_force_style(), which can conflict with the
                #    per-event inline override tags (\bord, \3c, \shad) that
                #    are already prepended to every Dialogue event in
                #    ass_generator.py.
                #
                # The inline override tags are sufficient to guarantee outline
                # rendering in the exported video.  Removing force_style
                # ensures the ASS file's carefully constructed per-event and
                # per-speaker styling is preserved exactly as generated.
                subtitle_force_style = ""
                logger.info(
                    "Subtitle rendering for clip %s: using inline override tags (no force_style)",
                    clip_id,
                )

                # QA: validate ASS content matches the input settings so
                # the exported video will match the frontend preview.
                qa_warnings = _validate_ass_settings(
                    ass_content=ass_content,
                    settings=settings,
                    video_width=out_w,
                    video_height=out_h,
                )
                for w in qa_warnings:
                    logger.warning("Export QA (clip %s): %s", clip_id, w)

                # Surface ALL QA warnings to the user via progress callback
                if qa_warnings:
                    await _notify(f"QA check found {len(qa_warnings)} issue(s) for clip {clip_id}")
                    for w in qa_warnings:
                        await _notify(f"QA (clip {clip_id}): {w}")

                await _notify(f"Subtitle file generated for clip {clip_id}")
            else:
                logger.warning(
                    "No ASS content generated for clip %s — no transcript "
                    "segments overlap clip range %.1f-%.1f (%d total segments)",
                    clip_id, start, end, len(transcript),
                )

        # If subtitles are off but text overlays need ASS (drawtext unavailable),
        # create a minimal ASS file just for text overlays.
        if has_text_overlays and not _HAS_DRAWTEXT and not ass_path:
            logger.info(
                "Creating ASS file for %d text overlay(s) (subs disabled, drawtext unavailable) clip %s",
                len(text_overlays), clip_id,
            )
            from backend.services.ass_generator import append_text_overlays_to_ass
            _minimal_ass = (
                "[Script Info]\nScriptType: v4.00+\n"
                f"PlayResX: {video_width}\nPlayResY: {video_height}\n"
                "ScaledBorderAndShadow: yes\n\n"
                "[V4+ Styles]\n"
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
                "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                "Alignment, MarginL, MarginR, MarginV, Encoding\n"
                "Style: Default,Arial,48,&H00FFFFFF&,&H000000FF&,&H00000000&,"
                "&H00000000&,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1\n\n"
                "[Events]\n"
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            )
            _minimal_ass = append_text_overlays_to_ass(
                _minimal_ass, text_overlays,
                clip_start=start, video_out_w=video_width, video_out_h=video_height,
            )
            ass_path = os.path.join(output_dir, f"clip_{clip_id}_textoverlay.ass")
            with open(ass_path, "w", encoding="utf-8") as f:
                f.write(_minimal_ass)
            subtitles_enabled = True  # Enable subtitle filter to render text overlays
            has_text_overlays = False  # Don't add drawtext to filter chain

        _check_cancel()

        # ── RenderPlan-based export path ──
        # When USE_RENDER_PLAN is enabled and reframe segments are available,
        # build the RenderPlan and use the FFmpeg filter builder for the crop/reframe
        # portion. This ensures the export matches the Canvas preview exactly.
        _render_plan_used = False
        try:
            from backend.services.render_plan import USE_RENDER_PLAN
            if USE_RENDER_PLAN and subject_scenes and aspect_ratio:
                from backend.services.render_plan_builder import build_render_plan
                from backend.services.ffmpeg_filter_builder import build_ffmpeg_command as _build_rp_cmd, cleanup_filter_script
                _rp_segments = _extract_render_plan_segments(subject_scenes)
                if _rp_segments:
                    _rp_plan = build_render_plan(
                        segments=_rp_segments,
                        source_width=video_width,
                        source_height=video_height,
                        source_fps=30.0,
                        target_aspect=aspect_ratio,
                        clip_range=(start, end),
                    )
                    logger.info(
                        "[RenderPlan] Built plan for clip %s: %d ops, %.1fs duration",
                        clip_id, len(_rp_plan.ops), _rp_plan.total_duration_sec,
                    )
                    _render_plan_used = True
        except Exception as rp_exc:
            logger.warning(
                "[RenderPlan] Failed to build plan for clip %s, falling back to legacy: %s",
                clip_id, rp_exc,
            )
            _render_plan_used = False

        if needs_filters:
            # Build subject keyframes for dynamic crop tracking
            keyframes = None
            _avg_face_y = 50.0
            _avg_face_w = 0.0
            logger.info(
                "═══════════════════════════════════════════════════════════════════════════",
            )
            logger.info(
                "[SubjectTracking] ▶ EXPORT clip %s: start=%.1f end=%.1f (%.1fs), "
                "aspect=%s, src=%dx%d, subject_x=%d, scenes=%d",
                clip_id, start, end, end - start,
                aspect_ratio or "original", video_width, video_height,
                subject_x, len(subject_scenes or []),
            )
            # Compute aspect ratios for dynamic safe margin
            _src_ratio = video_width / video_height if video_height else 1
            _target_ratio = ASPECT_RATIO_VALUES.get(aspect_ratio, _src_ratio) if aspect_ratio else _src_ratio

            # If all segments have tracking disabled, skip dynamic tracking entirely
            all_tracking_off = (
                segments
                and all(not seg.get("subject_tracking_enabled", True) for seg in segments)
            )
            if all_tracking_off:
                logger.info(
                    "[SubjectTracking] clip %s: ALL segments have tracking disabled — using static center crop",
                    clip_id,
                )

            # ── Solver keyframe override: use camera solver's per-shot keyframes ──
            # When layout_timeline_data has segments with solver_mode in face_positions,
            # use those directly instead of the cluster-snap pipeline.
            _using_solver_keyframes = False
            if (layout_timeline_data and aspect_ratio and not all_tracking_off
                    and not frontend_subject_keyframes
                    and os.environ.get("CLIPAI_CAMERA_SOLVER", "on").lower() != "off"):
                try:
                    solver_kf = _extract_solver_keyframes(
                        layout_timeline_data, start, end,
                    )
                    if solver_kf:
                        _using_solver_keyframes = True
                        keyframes = solver_kf
                        logger.info(
                            "[SubjectTracking] clip %s: Using %d solver keyframes (bypassing cluster-snap)",
                            clip_id, len(keyframes),
                        )
                        unique_x = set(kf[1] for kf in keyframes)
                        if len(unique_x) <= 1:
                            subject_x = keyframes[0][1]
                            keyframes = None
                            logger.info(
                                "[SubjectTracking] clip %s: solver keyframes all sx=%d — static crop",
                                clip_id, subject_x,
                            )
                except Exception as _solver_err:
                    logger.warning(
                        "[SubjectTracking] clip %s: solver keyframe extraction failed (%s), falling back",
                        clip_id, _solver_err,
                    )

            # ── Frontend keyframe override: use preview player's exact keyframes ──
            _using_frontend_keyframes = False
            if frontend_subject_keyframes and aspect_ratio and not all_tracking_off:
                _using_frontend_keyframes = True
                keyframes = [
                    (round(kf.get("time", 0), 3), int(round(kf.get("x", 50))))
                    for kf in frontend_subject_keyframes
                ]
                keyframes.sort()

                # Compute face Y from scene data for vertical positioning
                # (matters for 1:1, 4:5 crops where vertical offset is needed)
                if subject_scenes:
                    _face_ys = []
                    for s in subject_scenes:
                        if hasattr(s, 'face_positions') and s.face_positions:
                            for fp in s.face_positions:
                                y = fp.get('y', 0)
                                if 10 < y < 90:
                                    _face_ys.append(y)
                        elif hasattr(s, 'precise_y') and s.precise_y and 10 < s.precise_y < 90:
                            _face_ys.append(s.precise_y)
                    if _face_ys:
                        _avg_face_y = sum(_face_ys) / len(_face_ys)

                logger.info(
                    "[SubjectTracking] clip %s: Using %d frontend keyframes for crop (preview-export parity)",
                    clip_id, len(keyframes),
                )
                for kf_t, kf_x in keyframes[:5]:
                    logger.info("  t=%.1fs x=%d%%", kf_t, kf_x)
                if len(keyframes) > 5:
                    logger.info("  ... (%d more)", len(keyframes) - 5)

                # Check if all keyframes converge to single value → use static crop
                unique_x = set(kf[1] for kf in keyframes)
                if len(unique_x) <= 1:
                    subject_x = keyframes[0][1]
                    keyframes = None
                    logger.info(
                        "[SubjectTracking] clip %s: frontend keyframes all sx=%d — static crop",
                        clip_id, subject_x,
                    )

            # ── Phase 10 follow-up: cached RenderPlan keyframes ──
            #
            # The preview player reads ``job.render_plan`` verbatim via
            # ``GET /api/jobs/{id}/render_plan``. That cached dict is
            # built from the full-fidelity ``ReframeSegment`` list at
            # pipeline time and carries ``motion_path`` for walking
            # vlog subjects, anime action pans, and any other Stage 10
            # L1-solver-smoothed camera moves. The legacy export path
            # re-derives its own RenderPlan from the stripped
            # ``SceneDescription`` list (see
            # ``_extract_render_plan_segments``) which hard-codes
            # ``motion_path=None`` and so every tracking_crop
            # degrades to a static crop in the MP4.
            #
            # This branch closes the preview/export parity gap for
            # VLOG content (``allow_motion_tracking=True``). It runs
            # only when none of the higher-priority keyframe sources
            # fired, so solver keyframes and the Canvas frontend
            # override both still win.
            _using_cached_render_plan_keyframes = False
            if (not _using_solver_keyframes
                    and not _using_frontend_keyframes
                    and aspect_ratio
                    and not all_tracking_off
                    and job_id
                    and keyframes is None):
                try:
                    _job_for_rp = await database.load_job(job_id)
                    _cached_rp_dict = (
                        getattr(_job_for_rp, "render_plan", None)
                        if _job_for_rp
                        else None
                    )
                    if _cached_rp_dict:
                        _rp_kfs = _keyframes_from_cached_render_plan(
                            _cached_rp_dict,
                            clip_start=start,
                            clip_end=end,
                        )
                        if _rp_kfs and len(_rp_kfs) >= 2:
                            keyframes = _rp_kfs
                            _using_cached_render_plan_keyframes = True
                            logger.info(
                                "[SubjectTracking] clip %s: Using %d keyframes "
                                "from cached RenderPlan (motion_path preserved — "
                                "preview/export parity)",
                                clip_id, len(keyframes),
                            )
                            for kf_t, kf_x in keyframes[:5]:
                                logger.info("  t=%.1fs x=%d%%", kf_t, kf_x)
                            if len(keyframes) > 5:
                                logger.info("  ... (%d more)", len(keyframes) - 5)

                            # Collapse to static when every keyframe has
                            # the same x, matching the other keyframe
                            # tiers' behaviour.
                            unique_x = set(kf[1] for kf in keyframes)
                            if len(unique_x) <= 1:
                                subject_x = keyframes[0][1]
                                keyframes = None
                                logger.info(
                                    "[SubjectTracking] clip %s: cached RenderPlan "
                                    "keyframes all sx=%d — static crop",
                                    clip_id, subject_x,
                                )
                except Exception as _rp_kf_err:
                    logger.warning(
                        "[SubjectTracking] clip %s: cached RenderPlan keyframe "
                        "extraction failed (%s), falling back to SceneDescription",
                        clip_id, _rp_kf_err,
                    )
                    _using_cached_render_plan_keyframes = False

            if (subject_scenes and aspect_ratio and not all_tracking_off
                    and not _using_solver_keyframes
                    and not _using_frontend_keyframes
                    and not _using_cached_render_plan_keyframes):
                # ── PHASE 0: Build raw keyframes ──
                raw_kf = _build_subject_keyframes(subject_scenes, start, end, src_ratio=_src_ratio, target_ratio=_target_ratio)

                # ── Dense clip-level face detection ──
                # Full-video analysis gives ~3 face samples per 30s clip.
                # Dense detection extracts frames at 0.5s intervals for the clip
                # and runs speaker-aware face detection, giving 60+ accurate positions.
                try:
                    from backend.services.face_detector import detect_faces_dense as _dense_detect
                    _dense_results = _dense_detect(
                        video_path, start, end,
                        sample_rate=0.5,
                        min_confidence=0.4,
                        extract_embeddings=False,
                    )
                    # Reconstruct face_registry for slot center snapping
                    _clip_face_registry = None
                    if face_registry_data:
                        from backend.services.face_registry import FaceRegistry, FaceSlot
                        _clip_slots = [
                            FaceSlot(slot_id=s["id"], x_center=s["x"], x_min=s["x"], x_max=s["x"],
                                     frame_count=s.get("frames", 0), avg_width=0, avg_height=0)
                            for s in face_registry_data.get("slots", [])
                        ]
                        _clip_face_registry = FaceRegistry(
                            slots=_clip_slots,
                            total_frames=face_registry_data.get("total_frames", 0),
                            frames_with_faces=face_registry_data.get("frames_with_faces", 0),
                        )
                    # Use speaker-aware keyframes: track who's SPEAKING, not who's BIGGEST
                    dense_kf, _face_ys, _face_widths = _speaker_aware_keyframes(
                        _dense_results, start,
                        transcript=transcript,
                        face_registry=_clip_face_registry,
                    )
                    # Compute face metadata for vertical tracking and zoom
                    if _face_ys:
                        _avg_face_y = sum(_face_ys) / len(_face_ys)
                    if _face_widths:
                        _avg_face_w = sum(_face_widths) / len(_face_widths)
                    logger.info(
                        "[SubjectTracking] clip %s: speaker-aware dense detection (0.5s) produced %d keyframes "
                        "(avg_face_y=%.1f%%, avg_face_w=%.1f%%)",
                        clip_id, len(dense_kf),
                        _avg_face_y if _face_ys else 50, _avg_face_w if _face_widths else 0,
                    )
                except Exception as _dense_err:
                    logger.warning(
                        "[SubjectTracking] clip %s: improved dense detection failed (%s), falling back",
                        clip_id, _dense_err,
                    )
                    dense_kf = _dense_face_detection_for_clip(video_path, start, end, sample_rate=2.0)
                if dense_kf:
                    # Merge dense face keyframes with AI-derived keyframes.
                    # Face positions are pixel-accurate; prefer them over AI estimates.
                    existing_times = {round(t, 1) for t, _ in raw_kf}
                    added = 0
                    for t, sx in dense_kf:
                        if round(t, 1) not in existing_times:
                            raw_kf.append((t, sx))
                            added += 1
                        else:
                            # Replace AI estimate with face detection position
                            for idx, (rt, _) in enumerate(raw_kf):
                                if abs(rt - t) < 1.0:
                                    raw_kf[idx] = (rt, sx)
                                    break
                    raw_kf.sort()
                    logger.info(
                        "[SubjectTracking] clip %s: dense face detection added %d keyframes (total %d)",
                        clip_id, added, len(raw_kf),
                    )

                logger.info(
                    "[SubjectTracking] clip %s: %d raw keyframes from %d scenes",
                    clip_id, len(raw_kf), len(subject_scenes),
                )

                # ── PHASE 1: Detect position clusters (N speakers from visual data) ──
                # Works WITHOUT audio diarization — catches multi-speaker scenarios
                # even when Whisper only detects 1 speaker.
                _cluster_used = False
                _is_dense = len(raw_kf) >= 100  # Per-second dense face detection
                clusters = _detect_position_clusters(raw_kf)
                if clusters and len(clusters) >= 2 and not _is_dense:
                    centers = [c["center"] for c in clusters]
                    logger.info(
                        "[SubjectTracking] clip %s: %d CLUSTERS detected — centers=%s",
                        clip_id, len(clusters), centers,
                    )
                    snapped = _snap_to_clusters(raw_kf, clusters)

                    # Remove consecutive duplicates (same speaker holding)
                    deduped = [snapped[0]]
                    for i in range(1, len(snapped)):
                        if snapped[i][1] != deduped[-1][1]:
                            deduped.append(snapped[i])
                        elif i == len(snapped) - 1:
                            deduped.append((snapped[i][0], deduped[-1][1]))

                    # ── Anti-jitter: minimum hold duration ──
                    MIN_HOLD = 2.0
                    if len(deduped) >= 3:
                        # Pass 1: remove brief blips where surrounding positions match
                        i = 1
                        while i < len(deduped) - 1:
                            hold = deduped[i + 1][0] - deduped[i][0]
                            if hold < MIN_HOLD and deduped[i - 1][1] == deduped[i + 1][1]:
                                deduped.pop(i)
                            else:
                                i += 1
                    if len(deduped) >= 3:
                        # Pass 2: extend dominant position over remaining short holds
                        i = 1
                        while i < len(deduped) - 1:
                            hold = deduped[i + 1][0] - deduped[i][0]
                            if hold < MIN_HOLD:
                                deduped[i] = (deduped[i][0], deduped[i - 1][1])
                                if deduped[i][1] == deduped[i - 1][1]:
                                    deduped.pop(i)
                                else:
                                    i += 1
                            else:
                                i += 1

                    # Ensure start/end coverage
                    clip_dur = end - start
                    if deduped[0][0] > 0:
                        deduped.insert(0, (0.0, deduped[0][1]))
                    if deduped[-1][0] < clip_dur:
                        deduped.append((clip_dur, deduped[-1][1]))

                    # Fix initial snap: don't start at center default
                    if deduped and 44 <= deduped[0][1] <= 56:
                        first_real = next((kf for kf in raw_kf if kf[1] < 44 or kf[1] > 56), None)
                        if first_real:
                            best_c = min(clusters, key=lambda c: abs(first_real[1] - c["center"]))
                            deduped[0] = (deduped[0][0], best_c["center"])

                    # handleSceneCuts inserts 1ms instant-jump transitions
                    after_cuts = _handle_scene_cuts(deduped)
                    after_cuts = _inject_shot_boundary_cuts(after_cuts, scene_cut_timestamps, start, end)

                    # Final bounds enforcement
                    safe_lo, safe_hi = _compute_safe_range(_src_ratio, _target_ratio)
                    keyframes = [
                        (t, max(safe_lo, min(safe_hi, round(sx))))
                        for t, sx in after_cuts
                    ]

                    # Fix leading center keyframes after bounds enforcement
                    first_real_kf = next((kf for kf in raw_kf if kf[1] < 44 or kf[1] > 56), None)
                    if first_real_kf and keyframes:
                        best_c = min(clusters, key=lambda c: abs(first_real_kf[1] - c["center"]))
                        safe_center = max(safe_lo, min(safe_hi, best_c["center"]))
                        for j in range(len(keyframes)):
                            if 44 <= keyframes[j][1] <= 56:
                                keyframes[j] = (keyframes[j][0], safe_center)
                            else:
                                break

                    # QA validation
                    keyframes = _validate_tracking(keyframes, clusters, end - start)
                    _cluster_used = True

                    logger.info(
                        "[SubjectTracking] clip %s: %d-POSITION tracking — %d keyframes, "
                        "centers=%s, keyframes=%s",
                        clip_id, len(clusters), len(keyframes), centers,
                        [(f"t={t:.2f}s,sx={sx}") for t, sx in keyframes[:20]],
                    )

                # ── PHASE 2: Try speaker-aware tracking (requires 2+ speakers) ──
                _speaker_kf_used = False
                if not _cluster_used and transcript:
                    _spk_map = _build_speaker_position_map(subject_scenes, transcript)
                    if len(_spk_map) >= 2:
                        _spk_kf = _build_speaker_keyframes(transcript, _spk_map, start, end, src_ratio=_src_ratio, target_ratio=_target_ratio)
                        if _spk_kf and len(_spk_kf) >= 2:
                            after_cuts = _handle_scene_cuts(_spk_kf)
                            after_cuts = _inject_shot_boundary_cuts(after_cuts, scene_cut_timestamps, start, end)
                            safe_lo, safe_hi = _compute_safe_range(_src_ratio, _target_ratio)
                            keyframes = [(t, max(safe_lo, min(safe_hi, round(sx)))) for t, sx in after_cuts]
                            kf_xs = [kf[1] for kf in keyframes]
                            if max(kf_xs) - min(kf_xs) >= 5:
                                _speaker_kf_used = True
                                logger.info(
                                    "[SubjectTracking] clip %s: SPEAKER-AWARE tracking — %d keyframes, %d speakers, map=%s",
                                    clip_id, len(keyframes), len(_spk_map), _spk_map,
                                )
                            else:
                                keyframes = None

                # ── PHASE 3: Single-subject tracking (original pipeline) ──
                if not _cluster_used and not _speaker_kf_used and len(raw_kf) > 1:
                    # Sparse data detection: relax thresholds when we have ≤4 keyframes
                    # Dense data (100+ keyframes = per-second tracking) gets special handling:
                    # - NO compression (positions are already speaker-accurate from face detection)
                    # - Minimal dead zone (just filter sub-pixel jitter)
                    # - Higher smooth speed (allow fast speaker transitions)
                    is_sparse = len(raw_kf) <= 4
                    dz_threshold = 3 if is_sparse else (2 if _is_dense else 5)
                    compress_max = 60 if is_sparse else (100 if _is_dense else 30)
                    smooth_speed = 30 if is_sparse else (80 if _is_dense else 22)
                    hold_tolerance = 2 if is_sparse else (2 if _is_dense else 3)

                    # Full pipeline: compress → deadzone → scene cuts → snap transitions → smooth → merge holds
                    # Matches frontend processKeyframes() pipeline exactly for preview-export parity.
                    # For dense data, skip compression — the face positions are pixel-accurate
                    # from backend speaker-aware detection. Compressing toward median pulls all
                    # positions toward the dominant speaker, causing off-center framing.
                    after_compress = raw_kf[:] if _is_dense else _compress_range(raw_kf, max_range=compress_max, src_ratio=_src_ratio, target_ratio=_target_ratio)
                    after_dead_zone = _apply_dead_zone(after_compress, threshold=dz_threshold, src_ratio=_src_ratio, target_ratio=_target_ratio)
                    after_cuts = _handle_scene_cuts(after_dead_zone)
                    after_cuts = _inject_shot_boundary_cuts(after_cuts, scene_cut_timestamps, start, end)
                    # Insert hold-then-snap transitions BEFORE smoothing (matches frontend ordering)
                    after_snaps = _insert_snap_transitions(after_cuts)
                    after_smooth = _smooth_keyframes_bidirectional(after_snaps, max_speed=smooth_speed, src_ratio=_src_ratio, target_ratio=_target_ratio)
                    after_holds = _merge_holds(after_smooth, tolerance=hold_tolerance)

                    # Final bounds enforcement — clamp every keyframe to safe range
                    # to prevent any pipeline stage from producing values that push
                    # the crop off-screen or cause black bars
                    safe_lo, safe_hi = _compute_safe_range(_src_ratio, _target_ratio)
                    keyframes = [
                        (t, max(safe_lo, min(safe_hi, round(sx))))
                        for t, sx in after_holds
                    ]

                    logger.info(
                        "[SubjectTracking] clip %s: pipeline stages — raw=%d → compress=%d → deadzone=%d → cuts=%d → snaps=%d → smooth=%d → holds=%d → clamped=%d (safe=[%d,%d])",
                        clip_id, len(raw_kf), len(after_compress), len(after_dead_zone), len(after_cuts),
                        len(after_snaps), len(after_smooth), len(after_holds), len(keyframes), safe_lo, safe_hi,
                    )

                    # If pipeline collapsed to single value, keep as-is (static)
                    unique = set(kf[1] for kf in keyframes)
                    if len(unique) <= 1:
                        # Use the converged keyframe value as the static subject_x
                        # so the crop still centers on the tracked subject position
                        converged_sx = keyframes[0][1]
                        logger.info(
                            "[SubjectTracking] clip %s: all keyframes converged to sx=%d — using STATIC crop centered on subject",
                            clip_id, converged_sx,
                        )
                        subject_x = converged_sx
                        keyframes = None
                    else:
                        # Check for near-convergence: collapse to static to avoid jitter.
                        # Scale threshold by aspect ratio and data density.
                        kf_min = min(kf[1] for kf in keyframes)
                        kf_max = max(kf[1] for kf in keyframes)
                        R_conv = _src_ratio / _target_ratio if _target_ratio > 0 else 1
                        convergence_threshold = max(2, round(5 / R_conv)) if R_conv > 1.5 else 5
                        if is_sparse:
                            convergence_threshold = max(1, round(convergence_threshold * 0.6))
                        if kf_max - kf_min < convergence_threshold:
                            # Use keyframe closest to clip midpoint — picks whichever
                            # speaker the AI detected at the midpoint rather than
                            # averaging (which would center on the gap between speakers).
                            if len(keyframes) <= 1:
                                static_sx = keyframes[0][1]
                            else:
                                clip_mid_t = keyframes[len(keyframes) // 2][0]
                                best_idx = min(range(len(keyframes)), key=lambda j: abs(keyframes[j][0] - clip_mid_t))
                                static_sx = keyframes[best_idx][1]
                            logger.info(
                                "[SubjectTracking] clip %s: keyframe range too small (%d-%d, Δ=%d) — "
                                "collapsing to static sx=%d (midpoint, sparse=%s) to avoid jitter",
                                clip_id, kf_min, kf_max, kf_max - kf_min, static_sx, is_sparse,
                            )
                            subject_x = static_sx
                            keyframes = None
                        else:
                            logger.info(
                                "[SubjectTracking] clip %s: DYNAMIC tracking — %d processed keyframes, "
                                "sx range [%d, %d], keyframes=%s",
                                clip_id, len(keyframes),
                                kf_min, kf_max,
                                [(f"t={t:.2f}s,sx={sx}") for t, sx in keyframes],
                            )
                elif not _cluster_used and not _speaker_kf_used and len(raw_kf) == 1:
                    # Single keyframe — use its tracked value directly as the
                    # static subject_x so the crop centers on the actual subject
                    subject_x = raw_kf[0][1]
                    logger.info(
                        "[SubjectTracking] clip %s: single keyframe at sx=%d — using STATIC crop centered on subject",
                        clip_id, subject_x,
                    )
            else:
                if not subject_scenes:
                    logger.info(
                        "[SubjectTracking] clip %s: NO SCENE DATA — using default subject_x=%d (center crop)",
                        clip_id, subject_x,
                    )
                elif not aspect_ratio:
                    logger.info(
                        "[SubjectTracking] clip %s: no aspect ratio change — subject tracking not needed "
                        "(have %d scenes but original aspect ratio preserved)",
                        clip_id, len(subject_scenes),
                    )
                else:
                    logger.info(
                        "[SubjectTracking] clip %s: STATIC (subject_x=%d, scenes=%d, aspect=%s)",
                        clip_id, subject_x,
                        len(subject_scenes) if subject_scenes else 0,
                        aspect_ratio or "original",
                    )

            # Apply per-segment subject tracking toggles (if segments provided)
            if keyframes and segments:
                keyframes = _filter_keyframes_by_segment_tracking(
                    keyframes, segments, start, end, static_sx=subject_x,
                )
                # Re-check if keyframes collapsed to single value after filtering
                if keyframes:
                    unique = set(kf[1] for kf in keyframes)
                    if len(unique) <= 1:
                        subject_x = keyframes[0][1]
                        keyframes = None
                        logger.info(
                            "[SubjectTracking] clip %s: keyframes collapsed to static sx=%d after segment filtering",
                            clip_id, subject_x,
                        )

            # ── Layout-aware filter chain selection ──
            # If a non-single layout mode is requested (or auto-detected),
            # use the layout compositing pipeline instead of simple crop+pan.
            _layout_vf = None
            _layout_used = False
            if layout_mode and layout_mode != "single":
                try:
                    _layout_tl = None
                    if layout_mode == "auto" and layout_timeline_data:
                        # Reconstruct LayoutTimeline from serialized data
                        from backend.services.layout_engine import LayoutTimeline, LayoutSegment
                        from backend.services.face_registry import FaceRegistry, FaceSlot
                        _reg = None
                        if face_registry_data:
                            _slots = [
                                FaceSlot(slot_id=s["id"], x_center=s["x"], x_min=s["x"], x_max=s["x"],
                                         frame_count=s.get("frames", 0), avg_width=0, avg_height=0)
                                for s in face_registry_data.get("slots", [])
                            ]
                            _reg = FaceRegistry(slots=_slots,
                                                total_frames=face_registry_data.get("total_frames", 0),
                                                frames_with_faces=face_registry_data.get("frames_with_faces", 0))
                        _segs = [LayoutSegment(**s) for s in layout_timeline_data]
                        _layout_tl = LayoutTimeline(
                            segments=_segs,
                            default_mode=_segs[0].layout_mode if _segs else "single",
                            face_registry=_reg,
                            total_layout_changes=max(0, len(_segs) - 1),
                        )
                    elif layout_mode in ("split", "triple", "pip", "screenshare") and face_registry_data:
                        # User forced a specific layout — build a single-segment timeline
                        from backend.services.layout_engine import LayoutTimeline, LayoutSegment
                        from backend.services.face_registry import FaceRegistry, FaceSlot
                        _slots = [
                            FaceSlot(slot_id=s["id"], x_center=s["x"], x_min=s["x"], x_max=s["x"],
                                     frame_count=s.get("frames", 0), avg_width=0, avg_height=0)
                            for s in face_registry_data.get("slots", [])
                        ]
                        _reg = FaceRegistry(slots=_slots,
                                            total_frames=face_registry_data.get("total_frames", 0),
                                            frames_with_faces=face_registry_data.get("frames_with_faces", 0))
                        _seg = LayoutSegment(
                            start=0, end=end - start,
                            layout_mode=layout_mode,
                            pip_position=pip_position,
                            pip_size_pct=pip_size_pct,
                        )
                        if layout_mode == "split" and len(_slots) >= 2:
                            sorted_s = sorted(_slots, key=lambda s: s.x_center)
                            _seg.left_face_slot = sorted_s[0].slot_id
                            _seg.right_face_slot = sorted_s[1].slot_id
                        _layout_tl = LayoutTimeline(
                            segments=[_seg],
                            default_mode=layout_mode,
                            face_registry=_reg,
                            total_layout_changes=0,
                        )

                    if _layout_tl:
                        _layout_vf, _layout_complex, _layout_sub = _build_layout_filter_chain(
                            _layout_tl, video_width, video_height,
                            target_aspect=aspect_ratio or "16:9",
                            export_quality=export_quality,
                            clip_start=start, clip_end=end,
                            ass_path=ass_path,
                            subtitle_force_style=subtitle_force_style,
                            video_effects=video_effects if has_video_effects else None,
                        )
                        if _layout_vf:
                            _layout_used = True
                            logger.info("[Layout] Using layout filter chain for clip %s (mode=%s)", clip_id, layout_mode)
                except Exception as e:
                    logger.warning("[Layout] Layout filter chain failed (non-fatal), falling back to single: %s", e)

            # ── Gameplay composite ──
            # If tracking_mode is gameplay and we're doing a vertical crop,
            # use the HUD composite layout instead of simple center crop.
            _gameplay_used = False
            if not _layout_used and aspect_ratio in ("9:16", "4:5"):
                try:
                    # Check if this job is gameplay by reading tracking_mode from scenes
                    _job_for_gp = await database.load_job(job_id) if job_id else None
                    _gp_tracking = getattr(_job_for_gp, "tracking_mode", "") if _job_for_gp else ""
                    _gp_game_type = getattr(_job_for_gp, "game_type", "") if _job_for_gp else ""
                    if _gp_tracking == "gameplay":
                        from backend.services.game_layouts import get_hud_layout
                        _gp_layout = get_hud_layout(_gp_game_type or "generic_fps")
                        dims_t = ASPECT_RATIO_DIMS_BY_QUALITY.get(export_quality, ASPECT_RATIO_DIMS)
                        _gp_out_w, _gp_out_h = dims_t.get(aspect_ratio, (1080, 1920))
                        _gp_out_w = _gp_out_w - (_gp_out_w % 2)
                        _gp_out_h = _gp_out_h - (_gp_out_h % 2)
                        vf = _build_gameplay_composite_filter(
                            video_width, video_height,
                            _gp_out_w, _gp_out_h,
                            _gp_layout,
                        )
                        is_complex = True
                        _subtitle_vf = ""
                        _gameplay_used = True
                        logger.info(
                            "[Gameplay] Using composite filter for clip %s (game=%s, output=%dx%d)",
                            clip_id, _gp_game_type or "generic_fps", _gp_out_w, _gp_out_h,
                        )
                except Exception as e:
                    logger.warning("[Gameplay] Composite filter failed (non-fatal), falling back: %s", e)

            # Build filter chain and re-encode
            if _layout_used:
                vf = _layout_vf
                is_complex = True
                _subtitle_vf = _layout_sub
            elif _gameplay_used:
                pass  # vf already set above
            else:
                vf, is_complex, _subtitle_vf = _build_filter_chain(
                    aspect_ratio, video_width, video_height, ass_path,
                    subject_x=subject_x,
                    subject_keyframes=keyframes,
                    export_quality=export_quality,
                    subtitle_force_style=subtitle_force_style,
                    video_path=video_path,
                    start_time=start,
                    video_effects=video_effects if has_video_effects else None,
                    clip_duration=end - start,
                    face_y_center=_avg_face_y,
                    face_width_pct=_avg_face_w,
                    use_step_interpolation=_using_frontend_keyframes,
                )

            # Append text overlay drawtext filters.
            # For per-segment speed paths, text overlays must be applied AFTER
            # the concat (not per-segment) because the per-segment PTS
            # manipulation makes drawtext enable='between(t,...)' timing wrong.
            # When overlay_compositing_order is provided, text overlays are also
            # deferred so they can be interleaved with images in the unified chain.
            _overlay_warnings: list[str] = list(_shape_warnings)
            _text_vf_for_post_concat = ""
            _use_unified_compositing = bool(overlay_compositing_order) and (has_text_overlays or has_image_overlays)
            if has_text_overlays and _HAS_DRAWTEXT and text_overlays and not _use_unified_compositing:
                text_vf, _tw = _build_text_overlay_filters(text_overlays, clip_start=start, video_out_w=video_width, video_out_h=video_height)
                _overlay_warnings.extend(_tw)
                if text_vf:
                    if has_seg_speed:
                        # Defer: apply after concat in the per-segment path
                        _text_vf_for_post_concat = text_vf
                        logger.info("Text overlay filters for clip %s (deferred for post-concat): %s", clip_id, text_vf)
                    else:
                        vf = f"{vf},{text_vf}" if vf else text_vf
                        logger.info("Text overlay filters for clip %s: %s", clip_id, text_vf)

            # QA: validate subject tracking is correctly applied in filter chain
            st_warnings = _validate_subject_tracking(
                filter_chain=vf,
                aspect_ratio=aspect_ratio,
                src_w=video_width,
                src_h=video_height,
                subject_x=subject_x,
                subject_keyframes=keyframes,
            )
            for w in st_warnings:
                logger.warning("[SubjectTracking] QA ISSUE (clip %s): %s", clip_id, w)
            if st_warnings:
                await _notify(f"Subject tracking QA found {len(st_warnings)} issue(s) for clip {clip_id}")
                for w in st_warnings:
                    await _notify(f"QA (clip {clip_id}): {w}")
            else:
                logger.info(
                    "[SubjectTracking] ✓ QA PASSED for clip %s — subject centering validated",
                    clip_id,
                )

            # Run centering verification: compute the expected center offset for
            # each keyframe and log the centering precision
            if aspect_ratio and aspect_ratio in ASPECT_RATIO_VALUES:
                _verify_centering_math(
                    clip_id=clip_id,
                    aspect_ratio=aspect_ratio,
                    src_w=video_width,
                    src_h=video_height,
                    subject_x=subject_x,
                    subject_keyframes=keyframes,
                )

            # Resolve encoding params from quality preset (fallback to config)
            qp = QUALITY_PRESETS.get(export_quality, QUALITY_PRESETS["1080p"])
            enc_crf = qp["crf"]
            enc_preset = qp["preset"]

            # ── Per-segment speed via multi-input→setpts/atempo→concat ───
            # The previous split→trim→concat approach caused FFmpeg to
            # buffer the entire decoded video for each split output.  For
            # long videos (e.g. 722s at 1080p) this exhausts memory and
            # crashes the process.  The multi-input approach opens the
            # source file once per timeline segment with its own -ss seek,
            # so each input only decodes its own frames.
            if has_seg_speed:
                clip_dur_local = end - start
                timeline = _build_speed_timeline(segments, clip_dur_local, speed, volume, start)
                n = len(timeline)
                logger.info(
                    "Per-segment speed for clip %s: %d timeline entries from %d segments — %s",
                    clip_id, n, len(segments),
                    [(f"[{tl['start']:.1f}-{tl['end']:.1f}@{tl['speed']}x]") for tl in timeline],
                )

                # Probe whether the input has an audio stream
                _probe_cmd = [
                    "ffprobe", "-v", "quiet", "-select_streams", "a",
                    "-show_entries", "stream=index", "-of", "csv=p=0",
                    video_path,
                ]
                _probe_proc = await asyncio.create_subprocess_exec(
                    *_probe_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _probe_out, _ = await _probe_proc.communicate()
                _has_audio = bool(_probe_out.strip())

                # --- Build multi-input args: one -ss/-t/-i per segment ---
                # HW decode args go before each -i so the decoder is
                # initialised per-input (required for multi-input).
                _hw_dec = _gpu_decode_args_for_filter()
                input_args: list[str] = []
                for i, tl in enumerate(timeline):
                    abs_start = start + tl["start"]
                    seg_dur = tl["end"] - tl["start"]
                    # Add 1s buffer for keyframe alignment; trim in filter
                    # ensures exact segment duration
                    input_args += _hw_dec + [
                        "-ss", f"{abs_start:.3f}",
                        "-t", f"{seg_dur + 1.0:.3f}",
                        "-i", video_path,
                    ]

                # --- Video filter chain per input ---
                fc_lines: list[str] = []
                for i, tl in enumerate(timeline):
                    seg_dur = tl["end"] - tl["start"]
                    pts_offset = tl["start"]
                    pts_factor = 1.0 / tl["speed"]

                    # 1. trim to exact segment duration (safety for -ss buffer)
                    # 2. normalize PTS to 0 then shift to clip-relative time
                    #    so subtitle burn-in and dynamic crop keyframes align
                    # 3. apply visual filters (crop, scale, effects)
                    # 4. apply subtitles (needs PTS-aligned timing before speed change)
                    # 5. normalize PTS back to 0
                    # 6. apply speed scaling if needed
                    chain = f"[{i}:v]trim=duration={seg_dur:.3f},setpts=PTS-STARTPTS"
                    if pts_offset > 0.001:
                        chain += f"+{pts_offset:.3f}/TB"
                    if vf:
                        chain += f",{vf}"
                    # In the per-segment path, subtitles must be burned in here
                    # (before speed scaling) because the ASS timestamps are in
                    # the original time domain.  Overlays are applied post-concat
                    # and will render ON TOP of these subtitles.  For the common
                    # non-per-segment path, subtitles are applied after overlays.
                    if _subtitle_vf:
                        chain += f",{_subtitle_vf}"
                    chain += ",setpts=PTS-STARTPTS"
                    if abs(tl["speed"] - 1.0) > 0.001:
                        chain += f",setpts={pts_factor:.6f}*PTS"
                    chain += f"[vo{i}]"
                    fc_lines.append(chain)

                # --- Audio chain (only if audio exists) ---
                if _has_audio:
                    for i, tl in enumerate(timeline):
                        seg_dur = tl["end"] - tl["start"]
                        chain = (
                            f"[{i}:a]atrim=duration={seg_dur:.3f}"
                            f",asetpts=PTS-STARTPTS"
                        )
                        if abs(tl["speed"] - 1.0) > 0.001:
                            chain += f",{_atempo_chain(tl['speed'])}"
                        seg_vol = 0.0 if tl["muted"] else tl["volume"]
                        if abs(seg_vol - 1.0) > 0.001 or tl["muted"]:
                            chain += f",volume={seg_vol:.4f}"
                        chain += f"[ao{i}]"
                        fc_lines.append(chain)

                    # Concat video + audio
                    concat_inputs = "".join(f"[vo{i}][ao{i}]" for i in range(n))
                    fc_lines.append(f"{concat_inputs}concat=n={n}:v=1:a=1[finalv][finala]")
                    map_args = ["-map", "[finalv]", "-map", "[finala]"]
                else:
                    # No audio — video-only concat
                    concat_inputs = "".join(f"[vo{i}]" for i in range(n))
                    fc_lines.append(f"{concat_inputs}concat=n={n}:v=1:a=0[finalv]")
                    map_args = ["-map", "[finalv]"]

                # --- Overlay filters (applied after concat) ---
                # When overlay_compositing_order is provided, use unified chain
                # that interleaves text and image overlays in correct track order.
                # Otherwise fall back to the legacy separate pipeline.
                _vid_out_w, _vid_out_h = _compute_video_out_dims(
                    video_width, video_height, aspect_ratio, export_quality,
                )
                _img_base_idx = n  # n segment inputs → indices 0..n-1

                _per_seg_unified_applied = False
                if overlay_compositing_order and (has_text_overlays or has_image_overlays):
                    u_extra, u_fc, u_warns = _build_unified_overlay_chain(
                        compositing_order=overlay_compositing_order,
                        text_overlays=text_overlays or [],
                        image_overlays=image_overlays or [],
                        clip_start=start,
                        base_input_idx=_img_base_idx,
                        video_out_w=_vid_out_w,
                        video_out_h=_vid_out_h,
                        job_id=job_id,
                    )
                    _overlay_warnings.extend(u_warns)
                    if u_fc:
                        _per_seg_unified_applied = True
                        input_args += u_extra
                        fc_lines[-1] = fc_lines[-1].replace("[finalv]", "[vbase]", 1)
                        fc_lines.append(u_fc)
                        map_args = [m.replace("[finalv]", "[vcomp]") for m in map_args]
                        logger.info(
                            "Unified compositing chain for clip %s (per-seg): applied after concat (%d steps)",
                            clip_id, u_fc.count(";") + 1,
                        )
                    else:
                        logger.warning(
                            "Unified compositing chain for clip %s (per-seg): EMPTY result — "
                            "falling back to legacy pipeline. "
                            "compositing_order=%d entries, image_overlays=%d, text_overlays=%d",
                            clip_id,
                            len(overlay_compositing_order),
                            len(image_overlays or []),
                            len(text_overlays or []),
                        )

                # Legacy fallback (also runs when unified was never enabled or produced empty)
                if not _per_seg_unified_applied:
                    if has_image_overlays and image_overlays:
                        img_extra_args, img_overlay_fc, img_overlay_count, _iw = _build_image_overlay_data(
                            image_overlays, job_id, clip_start=start, base_input_idx=_img_base_idx,
                            video_out_w=_vid_out_w, video_out_h=_vid_out_h,
                            overlay_compositing_order=overlay_compositing_order,
                        )
                        _overlay_warnings.extend(_iw)
                        if img_overlay_count > 0:
                            input_args += img_extra_args
                            fc_lines[-1] = fc_lines[-1].replace("[finalv]", "[vbase]", 1)
                            fc_lines.append(img_overlay_fc)
                            map_args = [m.replace("[finalv]", "[vimg]") for m in map_args]
                            logger.info(
                                "Legacy image overlays for clip %s (per-seg): %d images, order=[%s]",
                                clip_id, img_overlay_count,
                                ", ".join(im.get("item_id", "?")[:12] for im in image_overlays),
                            )

                    if _text_vf_for_post_concat:
                        if any("[vimg]" in m for m in map_args):
                            fc_lines[-1] = fc_lines[-1].replace("[vimg]", "[vtxt_in]", 1)
                            fc_lines.append(f"[vtxt_in]{_text_vf_for_post_concat}[vtxt_out]")
                            map_args = [m.replace("[vimg]", "[vtxt_out]") for m in map_args]
                        else:
                            fc_lines[-1] = fc_lines[-1].replace("[finalv]", "[vtxt_in]", 1)
                            fc_lines.append(f"[vtxt_in]{_text_vf_for_post_concat}[finalv]")
                        logger.info("Text overlays applied post-concat for clip %s", clip_id)

                full_fc = ";".join(fc_lines)
                logger.info(
                    "Per-segment speed filter_complex for clip %s:\n%s",
                    clip_id, full_fc,
                )

                cmd = [
                    "ffmpeg", "-y",
                ] + input_args + [
                    "-filter_complex", full_fc,
                ] + map_args + [
                    *_gpu_encode_args(qp, export_quality),
                    "-threads", str(app_settings.FFMPEG_THREADS),
                ]
                if _has_audio:
                    cmd += ["-c:a", "aac"]
                cmd += ["-avoid_negative_ts", "make_zero"]
                if app_settings.FFMPEG_FASTSTART:
                    cmd += ["-movflags", "+faststart"]
                cmd += ["-progress", "pipe:1"]
                cmd.append(output_path)

            else:
                # ── Global speed + volume (original path) ─────────────
                # --- Speed filter: append setpts to video chain ---
                if has_speed and vf:
                    vf = f"{vf},setpts={1.0/speed}*PTS"
                elif has_speed:
                    vf = f"setpts={1.0/speed}*PTS"

                # --- Audio filter chain: atempo + volume ---
                af_parts: list[str] = []
                if has_speed:
                    atempo = _atempo_chain(speed)
                    if atempo:
                        af_parts.append(atempo)
                if has_volume:
                    if not has_segments:
                        af_parts.append(f"volume={volume:.2f}")
                # Per-segment volume overrides using FFmpeg volume expression.
                # When segments exist, we build a single volume filter with an
                # if(between()) expression that picks the right gain for each
                # time range, falling back to the global volume for gaps.
                if has_segments:
                    clip_dur_local = end - start
                    # Build expression: if(between(t,s1,e1),vol1,if(between(t,s2,e2),vol2,...,global))
                    expr = f"{volume:.4f}"  # fallback = global volume
                    for seg in reversed(segments):  # reversed so first segment is outermost if()
                        seg_start = max(0, seg["start"] - start)
                        seg_end = min(clip_dur_local, seg["end"] - start)
                        seg_vol = 0.0 if seg.get("muted", False) else seg.get("volume", 1.0)
                        expr = f"if(between(t\\,{seg_start:.3f}\\,{seg_end:.3f})\\,{seg_vol:.4f}\\,{expr})"
                    af_parts.append(f"volume='{expr}':eval=frame")
                af = ",".join(af_parts) if af_parts else None

                # --- Overlay integration for global path ---
                _use_complex_for_overlays = False
                _overlay_input_args: list[str] = []
                _overlay_fc = ""
                _overlay_out_label = "[vimg]"  # default for legacy path
                _vid_out_w, _vid_out_h = _compute_video_out_dims(
                    video_width, video_height, aspect_ratio, export_quality,
                )

                if _use_unified_compositing:
                    # Unified compositing: interleave text + images in track order
                    u_extra, u_fc, u_warns = _build_unified_overlay_chain(
                        compositing_order=overlay_compositing_order,
                        text_overlays=text_overlays or [],
                        image_overlays=image_overlays or [],
                        clip_start=start,
                        base_input_idx=1,
                        video_out_w=_vid_out_w,
                        video_out_h=_vid_out_h,
                        job_id=job_id,
                    )
                    _overlay_warnings.extend(u_warns)
                    if u_fc:
                        _use_complex_for_overlays = True
                        _overlay_input_args = u_extra
                        _overlay_fc = u_fc
                        _overlay_out_label = "[vcomp]"
                        logger.info(
                            "Unified compositing chain for clip %s (global): applied (%d steps)",
                            clip_id, u_fc.count(";") + 1,
                        )
                    else:
                        logger.warning(
                            "Unified compositing chain for clip %s: EMPTY result — "
                            "falling back to legacy pipeline. "
                            "compositing_order=%d entries, image_overlays=%d, text_overlays=%d",
                            clip_id,
                            len(overlay_compositing_order),
                            len(image_overlays or []),
                            len(text_overlays or []),
                        )
                        # Allow fallback to legacy path below
                        _use_unified_compositing = False

                # Legacy fallback (also runs when unified was never enabled)
                if not _use_complex_for_overlays and has_image_overlays and image_overlays:
                    img_extra_args, img_overlay_fc, img_overlay_count, _iw = _build_image_overlay_data(
                        image_overlays, job_id, clip_start=start, base_input_idx=1,
                        video_out_w=_vid_out_w, video_out_h=_vid_out_h,
                        overlay_compositing_order=overlay_compositing_order,
                    )
                    _overlay_warnings.extend(_iw)
                    if img_overlay_count > 0:
                        _use_complex_for_overlays = True
                        _overlay_input_args = img_extra_args
                        _overlay_fc = img_overlay_fc
                        _overlay_out_label = "[vimg]"
                        logger.info(
                            "Legacy image overlay chain for clip %s (global): %d overlays, order=[%s]",
                            clip_id, img_overlay_count,
                            ", ".join(im.get("item_id", "?")[:12] for im in image_overlays),
                        )

                # -t must be an OUTPUT option to correctly cap the output
                # duration.  FFmpeg applies options BETWEEN two -i flags to
                # the NEXT input, not the output.  When image/audio overlay
                # inputs follow the video input, placing -t before them
                # makes it an input option for the overlay file (useless for
                # images), and the video input loses its duration limit —
                # causing FFmpeg to encode from -ss to the END of the source.
                #
                # Fix: build the command without -t here, then append -t as
                # one of the last OUTPUT options (before the output file).
                output_dur = (end - start) / speed if has_speed else (end - start)
                cmd = [
                    "ffmpeg", "-y",
                    *_gpu_decode_args_for_filter(),
                    "-ss", str(start),
                    "-t", str(end - start),      # INPUT -t: raw source duration
                    "-i", video_path,
                ]
                # Add overlay inputs after the main video input
                if _overlay_input_args:
                    cmd += _overlay_input_args

                # Audio overlay inputs (background music, SFX)
                _audio_overlay_input_args: list[str] = []
                _audio_overlay_fc_parts: list[str] = []
                if has_audio_overlays and audio_overlays:
                    # Determine next input index after video (0) and image overlays
                    _ao_base_idx = 1 + (len(_overlay_input_args) // 2 if _overlay_input_args else 0)
                    _ao_labels: list[str] = []
                    for ao_i, ao in enumerate(audio_overlays):
                        ao_src_raw = ao.get("src", "")
                        ao_src = _resolve_media_path(ao_src_raw, job_id) if ao_src_raw else None
                        if not ao_src:
                            logger.warning("Audio overlay not found: src=%s, job_id=%s", ao_src_raw, job_id)
                            continue
                        ao_start = max(0, ao.get("start_time", 0) - start)
                        ao_end = ao.get("end_time", 0) - start
                        ao_vol = ao.get("volume", 1.0)
                        ao_speed = ao.get("speed", 1.0)
                        ao_fade_in = ao.get("fade_in", 0)
                        ao_fade_out = ao.get("fade_out", 0)
                        ao_idx = _ao_base_idx + len(_ao_labels)
                        _audio_overlay_input_args += ["-i", ao_src]
                        # Build per-overlay audio filter: speed, trim, delay, volume, fade
                        ao_chain = f"[{ao_idx}:a]"
                        ao_filters = []
                        if ao_speed and abs(ao_speed - 1.0) > 0.001:
                            atempo = _atempo_chain(ao_speed)
                            if atempo:
                                ao_filters.append(atempo)
                        if ao_vol != 1.0:
                            ao_filters.append(f"volume={ao_vol:.3f}")
                        if ao_fade_in > 0:
                            ao_filters.append(f"afade=t=in:st=0:d={ao_fade_in:.3f}")
                        if ao_fade_out > 0 and ao_end > ao_start:
                            ao_fo_start = max(0, ao_end - ao_start - ao_fade_out)
                            ao_filters.append(f"afade=t=out:st={ao_fo_start:.3f}:d={ao_fade_out:.3f}")
                        if ao_start > 0:
                            ao_filters.append(f"adelay={int(ao_start * 1000)}|{int(ao_start * 1000)}")
                        ao_label = f"[ao{ao_i}]"
                        if ao_filters:
                            ao_chain += ",".join(ao_filters) + ao_label
                        else:
                            ao_chain += f"anull{ao_label}"
                        _audio_overlay_fc_parts.append(ao_chain)
                        _ao_labels.append(ao_label)
                    if _ao_labels:
                        # Mix main audio with overlay audio tracks
                        main_audio_label = "[0:a]"
                        if af:
                            main_audio_label = "[amain]"
                            _audio_overlay_fc_parts.insert(0, f"[0:a]{af}[amain]")
                            af = None  # consumed into filter_complex
                        mix_inputs = main_audio_label + "".join(_ao_labels)
                        _audio_overlay_fc_parts.append(
                            f"{mix_inputs}amix=inputs={1 + len(_ao_labels)}:duration=first:dropout_transition=0[aout]"
                        )
                        cmd += _audio_overlay_input_args
                        logger.info(
                            "Audio overlays for clip %s: %d tracks",
                            clip_id, len(_ao_labels),
                        )

                if _use_complex_for_overlays:
                    # Build complex filter: [0:v]<existing_vf>[vbase]; <overlay_chain>
                    base_chain = f"[0:v]{vf}[vbase]" if vf else "[0:v]null[vbase]"
                    full_fc = f"{base_chain};{_overlay_fc}"
                    final_video_label = _overlay_out_label

                    # Burn subtitles AFTER overlay compositing so they render
                    # on top of shapes/images/text — matching the track stacking
                    # order where the subtitle track sits above overlay tracks.
                    if _subtitle_vf:
                        sub_in = final_video_label
                        sub_out = "[vsub]"
                        full_fc += f";{sub_in}{_subtitle_vf}{sub_out}"
                        final_video_label = sub_out
                        logger.info("Subtitle filter applied AFTER overlay compositing for clip %s", clip_id)

                    if _audio_overlay_fc_parts:
                        full_fc += ";" + ";".join(_audio_overlay_fc_parts)
                        cmd += ["-filter_complex", full_fc, "-map", final_video_label, "-map", "[aout]"]
                    else:
                        cmd += ["-filter_complex", full_fc, "-map", final_video_label, "-map", "0:a?"]
                elif _audio_overlay_fc_parts:
                    # No image overlays but have audio overlays — need filter_complex for audio mixing
                    # Append subtitle filter to the VF chain (no overlays to go on top of)
                    _vf_with_sub = f"{vf},{_subtitle_vf}" if vf and _subtitle_vf else (vf or _subtitle_vf or "")
                    fc_parts_list = []
                    if _vf_with_sub:
                        if is_complex:
                            fc_parts_list.append(_vf_with_sub)
                        else:
                            fc_parts_list.append(f"[0:v]{_vf_with_sub}[vout]")
                    fc_parts_list.extend(_audio_overlay_fc_parts)
                    full_fc = ";".join(fc_parts_list)
                    cmd += ["-filter_complex", full_fc]
                    if _vf_with_sub:
                        cmd += ["-map", "[vout]" if not is_complex else "[out]"]
                    else:
                        cmd += ["-map", "0:v"]
                    cmd += ["-map", "[aout]"]
                else:
                    # No overlays — append subtitle filter to VF chain directly
                    _vf_with_sub = f"{vf},{_subtitle_vf}" if vf and _subtitle_vf else (vf or _subtitle_vf or "")
                    if _vf_with_sub:
                        if is_complex:
                            cmd += ["-filter_complex", _vf_with_sub, "-map", "[out]", "-map", "0:a?"]
                        else:
                            cmd += ["-vf", _vf_with_sub]
                    elif vf:
                        if is_complex:
                            cmd += ["-filter_complex", vf, "-map", "[out]", "-map", "0:a?"]
                        else:
                            cmd += ["-vf", vf]
                if af:
                    cmd += ["-af", af]
                # When active word highlighting is enabled, ensure at least
                # 30fps output so subtitle color transitions appear smooth.
                # Low-fps source videos (e.g. 24fps) show visible lag because
                # subtitle updates only render at video frame boundaries.
                _aw_enabled = subtitle_settings.get("active_word_enabled", False) if subtitle_settings else False
                if _aw_enabled and subtitles_enabled:
                    cmd += ["-r", "30"]
                cmd += [
                    *_gpu_encode_args(qp, export_quality),
                    "-threads", str(app_settings.FFMPEG_THREADS),
                    "-c:a", "aac",
                    "-avoid_negative_ts", "make_zero",
                ]
                if app_settings.FFMPEG_FASTSTART:
                    cmd += ["-movflags", "+faststart"]
                # OUTPUT -t: limit output duration (speed-adjusted).
                # This is critical when speed != 1.0 (e.g. 0.5x makes
                # output 2x longer than input).
                if has_speed and abs(speed - 1.0) > 0.001:
                    cmd += ["-t", str(output_dur)]
                cmd += ["-progress", "pipe:1"]
                cmd.append(output_path)

            # Surface overlay skip warnings to the user before encoding
            if _overlay_warnings:
                await _notify(f"WARNING: {len(_overlay_warnings)} overlay(s) skipped during export")
                for _ow in _overlay_warnings:
                    await _notify(f"  ⚠ {_ow}")
                    logger.warning("Overlay skip (clip %s): %s", clip_id, _ow)

            # ── Z-ORDER VERIFICATION: Log the exact overlay rendering stack ──
            # This log line PROVES whether the fix is active in the running container.
            # If this log is absent, the Docker container has NOT been rebuilt.
            _z_order_dump = []
            if image_overlays:
                for _zi, _zov in enumerate(image_overlays):
                    _z_order_dump.append(
                        f"  layer {_zi} ({'BELOW' if _zi < len(image_overlays) - 1 else 'TOP'}): "
                        f"item_id={_zov.get('item_id', '?')}, "
                        f"src={os.path.basename(_zov.get('src', '?'))[:30]}"
                    )
                logger.info(
                    "Z-ORDER VERIFICATION for clip %s — %d overlay layers (first=bottom, last=top):\n%s",
                    clip_id, len(image_overlays), "\n".join(_z_order_dump),
                )
            else:
                logger.info("Z-ORDER VERIFICATION for clip %s — no image overlays", clip_id)
            # Also log filter_complex if present
            if "-filter_complex" in cmd:
                _fc_idx = cmd.index("-filter_complex") + 1
                if _fc_idx < len(cmd):
                    logger.info("FILTER_COMPLEX for clip %s:\n%s", clip_id, cmd[_fc_idx].replace(";", ";\n"))

            logger.info("FFmpeg export command for clip %s: %s", clip_id, " ".join(cmd))
            logger.info("FFmpeg filter chain for clip %s: %s", clip_id, vf or "(none)")
            _enc_label = get_encoder_label()
            await _notify(f"Encoding clip {clip_id} via {_enc_label} — filters ({filter_desc})...")

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Track encoding progress with ETA via FFmpeg -progress output
            import time as _time
            _enc_start = _time.monotonic()
            if has_seg_speed:
                # Compute expected output duration from per-segment speeds
                _clip_dur = sum((tl["end"] - tl["start"]) / tl["speed"] for tl in timeline)
            elif has_speed:
                _clip_dur = (end - start) / speed
            else:
                _clip_dur = (end - start)
            _last_notify_time = _enc_start
            _current_out_time = 0.0
            _stderr_chunks: list[bytes] = []

            async def _drain_stderr():
                while True:
                    chunk = await proc.stderr.read(8192)
                    if not chunk:
                        break
                    _stderr_chunks.append(chunk)

            _stderr_task = asyncio.create_task(_drain_stderr())

            # Read FFmpeg progress from stdout line by line
            async for _line in proc.stdout:
                _line_str = _line.decode("utf-8", errors="replace").strip()
                if _line_str.startswith("out_time_us="):
                    try:
                        _us = int(_line_str.split("=", 1)[1])
                        if _us > 0:
                            _current_out_time = _us / 1_000_000
                    except (ValueError, IndexError):
                        pass

                _now = _time.monotonic()
                if _now - _last_notify_time >= 2.0:
                    _last_notify_time = _now

                    if cancel_event and cancel_event.is_set():
                        proc.kill()
                        await proc.wait()
                        if os.path.exists(output_path):
                            try:
                                os.unlink(output_path)
                            except OSError:
                                pass
                        raise asyncio.CancelledError("Export cancelled by user")

                    # Safety timeout: if encoding takes 60x the clip
                    # duration (or at least 10 minutes), something is
                    # likely wrong. Kill FFmpeg to unblock the pipeline.
                    _elapsed_s = _now - _enc_start
                    _max_encode_s = max(600, _clip_dur * 60)
                    if _elapsed_s > _max_encode_s:
                        logger.error(
                            "Encoding timeout for clip %s: %.0fs elapsed (limit=%.0fs, clip=%.1fs)",
                            clip_id, _elapsed_s, _max_encode_s, _clip_dur,
                        )
                        proc.kill()
                        await proc.wait()
                        raise RuntimeError(
                            f"Encoding timed out after {int(_elapsed_s)}s "
                            f"(expected ~{int(_clip_dur)}s of output)"
                        )

                    _elapsed = int(_elapsed_s)
                    if _current_out_time > 0.5 and _elapsed_s > 2 and _clip_dur > 0:
                        _speed = _current_out_time / _elapsed_s
                        _remaining = max(0, _clip_dur - _current_out_time)
                        _eta_s = int(_remaining / _speed) if _speed > 0 else 0
                        _pct = min(99, int(_current_out_time / _clip_dur * 100))
                        if _eta_s >= 60:
                            _eta_str = f"{_eta_s // 60}m {_eta_s % 60}s"
                        else:
                            _eta_str = f"{_eta_s}s"
                        await _notify(
                            f"Encoding clip {clip_id} [{_enc_label}]... {_pct}% ({_elapsed}s elapsed, ~{_eta_str} remaining)"
                        )
                    else:
                        await _notify(f"Encoding clip {clip_id} [{_enc_label}]... ({_elapsed}s elapsed)")

            # Encoding frames done — FFmpeg may still be finalizing
            # (writing moov atom for faststart, flushing encoder).
            _enc_elapsed = int(_time.monotonic() - _enc_start)
            await _notify(f"Encoding clip {clip_id} [{_enc_label}]... 100% — finalizing ({_enc_elapsed}s)")

            await proc.wait()
            await _stderr_task
            stderr = b"".join(_stderr_chunks)

            if proc.returncode != 0:
                _stderr_text = stderr.decode(errors="replace")
                # Extract actual error lines (skip the FFmpeg version banner)
                _error_lines = [
                    line.strip() for line in _stderr_text.split("\n")
                    if line.strip() and any(kw in line.lower() for kw in [
                        "error", "no such filter", "filter not found", "failed",
                        "invalid", "no space", "permission denied", "cannot",
                    ])
                ]
                _error_msg = "\n".join(_error_lines[-5:]) if _error_lines else _stderr_text[-1500:]

                # ── GPU→CPU fallback: if CUDA/NVENC failed, retry with software encoding ──
                _cuda_errors = ["cuda_error", "cuinit", "nvenc", "device creation failed", "hwdevice"]
                _is_gpu_error = any(e in _stderr_text.lower() for e in _cuda_errors)
                _using_gpu = any(
                    arg in cmd for arg in ["h264_nvenc", "hevc_nvenc", "-hwaccel", "cuda"]
                )
                if _is_gpu_error and _using_gpu:
                    logger.warning(
                        "GPU encoding failed for clip %s — retrying with CPU (libx264). Error: %s",
                        clip_id, _error_msg[:200],
                    )
                    await _notify(f"GPU encoding failed — retrying clip {clip_id} with CPU encoding...")
                    # Rebuild command: strip GPU args, use CPU encoder
                    cpu_cmd = []
                    skip_next = False
                    for ci, arg in enumerate(cmd):
                        if skip_next:
                            skip_next = False
                            continue
                        # Remove GPU decode args
                        if arg in ("-hwaccel", "-hwaccel_output_format"):
                            skip_next = True
                            continue
                        if arg in ("cuda", "cuvid"):
                            continue
                        # Replace GPU encoder with CPU
                        if arg in ("h264_nvenc", "hevc_nvenc"):
                            cpu_cmd.append("libx264")
                            continue
                        # Remove NVENC-specific args
                        if arg in ("-gpu", "-rc", "-rc:v", "-spatial_aq", "-temporal_aq",
                                   "-b_ref_mode", "-weighted_pred", "-cq", "-tag:v"):
                            skip_next = True
                            continue
                        # -preset is legitimate for libx264, BUT NVENC uses
                        # p1..p7 preset names (fastest..slowest) which libx264
                        # rejects as "invalid preset 'p5'". Strip any NVENC-
                        # style preset so the CPU default gets inserted below.
                        if arg == "-preset" and ci + 1 < len(cmd):
                            next_val = cmd[ci + 1]
                            if (len(next_val) == 2 and next_val[0] == "p"
                                    and next_val[1].isdigit()):
                                skip_next = True
                                continue
                        if arg in ("1", "middle", "vbr", "vbr_hq"):
                            # Could be a value for a skipped arg; but if we're not
                            # skipping, keep it. This is handled by skip_next above.
                            pass
                        cpu_cmd.append(arg)
                    # Ensure CPU encoder settings
                    if "-pix_fmt" not in cpu_cmd:
                        # Insert before output path
                        cpu_cmd.insert(-1, "-pix_fmt")
                        cpu_cmd.insert(-1, "yuv420p")
                    if "-crf" not in cpu_cmd:
                        cpu_cmd.insert(-1, "-crf")
                        cpu_cmd.insert(-1, str(qp.get("crf", 23)))
                    if "-preset" not in cpu_cmd:
                        cpu_cmd.insert(-1, "-preset")
                        cpu_cmd.insert(-1, qp.get("preset", "medium"))
                    # Remove any leftover NVENC quality args
                    cpu_cmd = [a for a in cpu_cmd if a not in ("-qp", "-qmin", "-qmax")]

                    logger.info("FFmpeg CPU retry command for clip %s: %s", clip_id, " ".join(cpu_cmd))

                    # Delete failed output
                    if os.path.exists(output_path):
                        try:
                            os.unlink(output_path)
                        except OSError:
                            pass

                    proc2 = await asyncio.create_subprocess_exec(
                        *cpu_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    _enc_start2 = _time.monotonic()
                    _stderr_chunks2: list[bytes] = []

                    async def _drain_stderr2():
                        while True:
                            chunk = await proc2.stderr.read(8192)
                            if not chunk:
                                break
                            _stderr_chunks2.append(chunk)

                    _stderr_task2 = asyncio.create_task(_drain_stderr2())
                    async for _line2 in proc2.stdout:
                        _line2_str = _line2.decode("utf-8", errors="replace").strip()
                        if _line2_str.startswith("out_time_us="):
                            try:
                                _us2 = int(_line2_str.split("=", 1)[1])
                                if _us2 > 0:
                                    _current_out_time = _us2 / 1_000_000
                            except (ValueError, IndexError):
                                pass
                        _now2 = _time.monotonic()
                        if _now2 - _last_notify_time >= 2.0:
                            _last_notify_time = _now2
                            if cancel_event and cancel_event.is_set():
                                proc2.kill()
                                await proc2.wait()
                                raise asyncio.CancelledError("Export cancelled by user")
                            if _current_out_time > 0.5 and _clip_dur > 0:
                                _pct2 = min(99, int(_current_out_time / _clip_dur * 100))
                                _elapsed2 = int(_now2 - _enc_start2)
                                await _notify(f"Encoding clip {clip_id} [CPU libx264]... {_pct2}% ({_elapsed2}s elapsed)")

                    await proc2.wait()
                    await _stderr_task2
                    stderr2 = b"".join(_stderr_chunks2)
                    if proc2.returncode != 0:
                        _stderr2_text = stderr2.decode(errors="replace")[-1500:]
                        raise RuntimeError(f"Clip export failed (CPU retry):\n{_stderr2_text}")
                    logger.info("CPU retry succeeded for clip %s", clip_id)
                else:
                    raise RuntimeError(f"Clip export failed:\n{_error_msg}")
        else:
            # No filters — use stream copy for speed
            await _notify(f"Exporting clip {clip_id} (stream copy — fast mode)")
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", video_path,
                "-t", str(end - start),
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
            ]
            if app_settings.FFMPEG_FASTSTART:
                cmd += ["-movflags", "+faststart"]
            cmd.append(output_path)
            logger.info("FFmpeg stream-copy command for clip %s: %s", clip_id, " ".join(cmd))

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()

            if proc.returncode != 0:
                # Retry with re-encoding if stream copy fails
                await _notify(f"Stream copy failed for clip {clip_id}, re-encoding...")
                logger.warning(f"Stream copy failed for clip {clip_id}, re-encoding...")
                fb_qp = QUALITY_PRESETS.get(export_quality, QUALITY_PRESETS["1080p"])
                cmd = [
                    "ffmpeg", "-y",
                    *_gpu_decode_args_for_filter(),
                    "-ss", str(start),
                    "-i", video_path,
                    "-t", str(end - start),
                ]
                # Apply quality scale if source height differs from target.
                # Always normalize SAR to 1:1 to prevent black bars from
                # non-square pixel aspect ratios in the source video.
                fb_target_h = QUALITY_MAX_HEIGHT.get(export_quality, 1080)
                if video_height != fb_target_h:
                    cmd += ["-vf", f"setsar=1,scale=-2:{fb_target_h}"]
                else:
                    cmd += ["-vf", "setsar=1"]
                cmd += [
                    *_gpu_encode_args(fb_qp, export_quality),
                    "-threads", str(app_settings.FFMPEG_THREADS),
                    "-c:a", "aac",
                    "-avoid_negative_ts", "make_zero",
                ]
                if app_settings.FFMPEG_FASTSTART:
                    cmd += ["-movflags", "+faststart"]
                cmd.append(output_path)
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await proc.communicate()
                if proc.returncode != 0:
                    stderr_tail = stderr.decode(errors="replace")[-2000:] if stderr else "unknown error"
                    raise RuntimeError(f"Clip export failed:\n{stderr_tail}")

        # QA validation: verify the exported file is valid
        await _notify(f"Validating export for clip {clip_id}...")
        # Use speed-adjusted duration so the check matches actual output
        if has_seg_speed:
            qa_expected_dur = sum((tl["end"] - tl["start"]) / tl["speed"] for tl in timeline)
        elif has_speed:
            qa_expected_dur = clip_dur / speed
        else:
            qa_expected_dur = clip_dur
        await _validate_export(
            output_path=output_path,
            expected_duration=qa_expected_dur,
            aspect_ratio=aspect_ratio,
            subtitles_enabled=subtitles_enabled,
            export_quality=export_quality,
        )

        logger.info(f"Exported clip {clip_id} to {output_path}")
        return output_path

    finally:
        # Clean up temp ASS file
        if ass_path and os.path.exists(ass_path):
            try:
                os.unlink(ass_path)
            except OSError:
                pass
        # Clean up shape temp files
        for _sf in _shape_temp_files:
            try:
                if os.path.isfile(_sf):
                    os.unlink(_sf)
            except OSError:
                pass
        # Clean up shape temp directory
        shape_tmp_dir = os.path.join(output_dir, "_shapes")
        try:
            if os.path.isdir(shape_tmp_dir):
                os.rmdir(shape_tmp_dir)  # only removes if empty
        except OSError:
            pass
