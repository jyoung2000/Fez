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
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

CROP_QA_ENV = "CLIPAI_CROP_QA"

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
