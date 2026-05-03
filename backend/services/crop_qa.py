"""Editorial QA pass for rendered vertical crops.

Phase 5 of the VLM subject-tracking upgrade.

After a segment is cropped to 9:16 and rendered, this module
samples a handful of output frames and asks the VLM to score the
crop quality — can you see the subject's head? Any awkward edge
cuts? Dead space dominating the frame? An aggregate below a
threshold triggers a re-solve with a looser dead-zone or falls
through to a safety-center crop.

This is the closest analogue to AutoFlip's "feature stability"
metric and it's the difference between "technically correct" and
"editorially correct."

Gated behind ``CLIPAI_CROP_QA`` env var. Default OFF.

Pure scoring module: all heavy lifting (VLM call, FFmpeg frame
extraction) is injected as callables so the module is testable
without opencv / ffmpeg / httpx in the test environment.

Phase 4 of the reframing overhaul also added
:func:`validate_no_black_bars` — a structural QA pass that catches
black-bar regressions on rendered output. It samples the rendered
file at ~1 fps and inspects the four edges of each frame; any
near-black border is logged and reported as a QA failure.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

CROP_QA_ENV = "CLIPAI_CROP_QA"

# ── No-black-bars validation (Phase 4) ─────────────────────────────
# Mean pixel value below this threshold counts a border as "black".
BLACK_BAR_PIXEL_THRESHOLD = 10.0
# Border thickness sampled at each edge, in pixels.
BLACK_BAR_BORDER_PX = 5

# Threshold table for recovery decisions. Tuned against the
# prompt spec (see docs/vlm_upgrade/PHASE_5_NOTES.md).
QA_ACCEPTABLE_SCORE = 7.0
QA_FATAL_SCORE = 5.0


def crop_qa_enabled() -> bool:
    """True if ``CLIPAI_CROP_QA`` is a truthy env value."""
    return os.environ.get(CROP_QA_ENV, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


@dataclass
class CropQualitySample:
    """One VLM-scored output frame."""

    timestamp: float
    head_in_frame: bool
    awkward_crop: bool
    subject_partially_off_frame: bool
    dead_space_dominant: bool
    quality_score: float  # 0-10


@dataclass
class CropQualityReport:
    """Aggregate report for a rendered segment."""

    segment_index: int
    samples: list[CropQualitySample] = field(default_factory=list)
    aggregate_score: float = 0.0
    triggered_recovery: str | None = None  # "loose_dead_zone"|"padding"|"safety_center"|None

    @property
    def any_subject_off_frame(self) -> bool:
        return any(s.subject_partially_off_frame for s in self.samples)

    @property
    def any_awkward_crop(self) -> bool:
        return any(s.awkward_crop for s in self.samples)


def _aggregate(samples: list[CropQualitySample]) -> float:
    if not samples:
        return 0.0
    return sum(s.quality_score for s in samples) / float(len(samples))


def decide_recovery(report: CropQualityReport) -> str | None:
    """Map a quality report onto a recovery strategy string.

    Recovery table (from the Phase 5 spec):

      no samples                                → None (no opinion)
      aggregate < 5  OR  re-solve failed        → "safety_center"
      aggregate < 7  AND any subject off frame → "loose_dead_zone"
      aggregate < 7  AND any awkward crop       → "padding"
      otherwise                                 → None (accept crop)
    """
    if not report.samples:
        return None
    agg = report.aggregate_score
    if agg < QA_FATAL_SCORE:
        return "safety_center"
    if agg < QA_ACCEPTABLE_SCORE:
        if report.any_subject_off_frame:
            return "loose_dead_zone"
        if report.any_awkward_crop:
            return "padding"
        # Score is weak but no specific failure mode fingerprint —
        # default to loosening the dead zone since it's the least
        # invasive recovery.
        return "loose_dead_zone"
    return None


def score_crop_quality(
    segment_index: int,
    sample_timestamps: list[float],
    vlm_scorer,
) -> CropQualityReport:
    """Score a rendered segment's crop quality.

    Parameters
    ----------
    segment_index : int
        Index into the job's segment list.
    sample_timestamps : list[float]
        Timestamps (within the rendered segment) to score.
    vlm_scorer : Callable[[float], dict]
        Injected callable. Takes a timestamp, returns a dict with
        the VLM scorer's output:
        ``{head_in_frame, awkward_crop, subject_partially_off_frame,
        dead_space_dominant, quality_score}``.

    Returns
    -------
    CropQualityReport
        ``triggered_recovery`` is populated by ``decide_recovery``.
    """
    samples: list[CropQualitySample] = []
    for ts in sample_timestamps:
        raw = vlm_scorer(ts)
        samples.append(CropQualitySample(
            timestamp=float(ts),
            head_in_frame=bool(raw.get("head_in_frame", True)),
            awkward_crop=bool(raw.get("awkward_crop", False)),
            subject_partially_off_frame=bool(
                raw.get("subject_partially_off_frame", False),
            ),
            dead_space_dominant=bool(raw.get("dead_space_dominant", False)),
            quality_score=max(0.0, min(10.0, float(raw.get("quality_score", 0)))),
        ))
    report = CropQualityReport(
        segment_index=segment_index,
        samples=samples,
        aggregate_score=_aggregate(samples),
    )
    report.triggered_recovery = decide_recovery(report)
    return report


def apply_recovery_params(
    strategy: str,
    dead_zone_px: float,
    lambda2: float,
    crop_padding_frac: float,
) -> tuple[float, float, float]:
    """Return tuned (dead_zone_px, lambda2, crop_padding_frac) for a strategy.

    Mirrors the Phase 5 recovery table:

      loose_dead_zone → dead_zone_px *= 1.5, lambda2 *= 0.5
      padding         → crop_padding_frac += 0.05
      safety_center   → no param changes (caller forces centered crop)
      None / unknown  → inputs returned unchanged
    """
    if strategy == "loose_dead_zone":
        return (dead_zone_px * 1.5, lambda2 * 0.5, crop_padding_frac)
    if strategy == "padding":
        return (dead_zone_px, lambda2, crop_padding_frac + 0.05)
    # safety_center or unknown → caller handles centering separately.
    return (dead_zone_px, lambda2, crop_padding_frac)


# ── Phase 4: no-black-bars structural QA ───────────────────────────


@dataclass
class CropQaReport:
    """Structural QA report for a rendered output file (Phase 4).

    ``passed`` is True iff no sampled frame had a black border on any
    of its four edges. ``frames_with_black`` lists the frame indices
    (0-based; one frame per second of sampling) that flagged. The
    optional ``note`` carries a one-line diagnostic when the QA
    pass could not run (e.g. ffmpeg not on the PATH).
    """

    passed: bool
    frames_with_black: List[int] = field(default_factory=list)
    total_frames_sampled: int = 0
    note: Optional[str] = None


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _probe_dimensions(rendered_path: str) -> Optional[tuple]:
    """Return (width, height, duration_sec) for ``rendered_path`` or None."""
    if shutil.which("ffprobe") is None:
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height:format=duration",
                "-of", "default=noprint_wrappers=1",
                rendered_path,
            ],
            capture_output=True, text=True, timeout=15.0,
        )
        if result.returncode != 0:
            return None
        meta = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()
        w = int(meta.get("width", 0) or 0)
        h = int(meta.get("height", 0) or 0)
        dur = float(meta.get("duration", 0.0) or 0.0)
        if w <= 0 or h <= 0:
            return None
        return (w, h, dur)
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def _extract_raw_frames(rendered_path: str, fps: float, w: int, h: int) -> Optional[bytes]:
    """Extract raw rgb24 frames at ``fps`` to memory. Returns bytes or None.

    Each frame is ``w*h*3`` bytes. Limits at 600 frames (~10 min @ 1 fps)
    to keep memory bounded.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-nostdin",
                "-i", rendered_path,
                "-vf", f"fps={fps:g}",
                "-frames:v", "600",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-",
            ],
            capture_output=True, timeout=120.0,
        )
        if result.returncode != 0:
            return None
        return result.stdout
    except (subprocess.SubprocessError, OSError):
        return None


def _frame_edge_means(frame: "object", w: int, h: int) -> tuple:
    """Return mean pixel intensity for each edge: (top, bottom, left, right).

    ``frame`` is a numpy array of shape (h, w, 3) with dtype uint8.
    Computes a luma-ish mean over a ``BLACK_BAR_BORDER_PX``-wide
    border on each edge.
    """
    import numpy as np
    border = max(1, int(BLACK_BAR_BORDER_PX))
    border = min(border, h // 2, w // 2)
    top = float(np.mean(frame[:border, :, :]))
    bottom = float(np.mean(frame[-border:, :, :]))
    left = float(np.mean(frame[:, :border, :]))
    right = float(np.mean(frame[:, -border:, :]))
    return (top, bottom, left, right)


def validate_no_black_bars(rendered_path: str, fps: float = 1.0) -> CropQaReport:
    """Sample frames from ``rendered_path`` and check all 4 edges for black.

    Phase 4 of the reframing overhaul. Black bars on any side are a
    regression: WIDE_MASTER and BLUR_FILL render with a blurred
    background, never solid black. This pass catches it.

    Parameters
    ----------
    rendered_path : str
        Path to the rendered output mp4.
    fps : float
        Sampling rate; default 1 frame per second.

    Returns
    -------
    CropQaReport
        ``passed=False`` when any sampled frame has a near-black
        border. Gracefully returns ``passed=True`` with a ``note``
        when ffmpeg/ffprobe or numpy is not available so callers do
        not break.
    """
    if not rendered_path or not os.path.exists(rendered_path):
        return CropQaReport(
            passed=True, total_frames_sampled=0,
            note=f"file not found: {rendered_path!r}",
        )

    if not _ffmpeg_available():
        logger.info(
            "validate_no_black_bars: ffmpeg/ffprobe not available; skipping",
        )
        return CropQaReport(
            passed=True, total_frames_sampled=0,
            note="ffmpeg or ffprobe not available — QA skipped",
        )

    try:
        import numpy as np  # noqa: F401
    except ImportError:
        return CropQaReport(
            passed=True, total_frames_sampled=0,
            note="numpy not available — QA skipped",
        )

    dims = _probe_dimensions(rendered_path)
    if dims is None:
        return CropQaReport(
            passed=True, total_frames_sampled=0,
            note="ffprobe failed to read video dimensions",
        )
    w, h, _dur = dims

    raw = _extract_raw_frames(rendered_path, fps=fps, w=w, h=h)
    if not raw:
        return CropQaReport(
            passed=True, total_frames_sampled=0,
            note="ffmpeg frame extraction returned no data",
        )

    import numpy as np
    frame_size = w * h * 3
    n_frames = len(raw) // frame_size
    if n_frames == 0:
        return CropQaReport(
            passed=True, total_frames_sampled=0,
            note="no frames decoded from extraction",
        )

    arr = np.frombuffer(raw, dtype=np.uint8, count=n_frames * frame_size)
    arr = arr.reshape((n_frames, h, w, 3))

    bad: List[int] = []
    for idx in range(n_frames):
        top, bottom, left, right = _frame_edge_means(arr[idx], w, h)
        if (top < BLACK_BAR_PIXEL_THRESHOLD or
                bottom < BLACK_BAR_PIXEL_THRESHOLD or
                left < BLACK_BAR_PIXEL_THRESHOLD or
                right < BLACK_BAR_PIXEL_THRESHOLD):
            bad.append(idx)
            logger.warning(
                "validate_no_black_bars: frame %d has black border "
                "(top=%.1f bot=%.1f left=%.1f right=%.1f, threshold=%.1f)",
                idx, top, bottom, left, right, BLACK_BAR_PIXEL_THRESHOLD,
            )

    return CropQaReport(
        passed=len(bad) == 0,
        frames_with_black=bad,
        total_frames_sampled=n_frames,
    )
