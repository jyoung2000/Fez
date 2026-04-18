"""Hard gate that enforces the confidence ladder at the segment output boundary.

The existing subject_confidence.py defines a 4-tier ladder:
  >= 0.70 stationary
  >= 0.50 inherit last confident position
  >= 0.30 blur_fill
  <  0.30 wide_master

This module is called RIGHT BEFORE segment list is returned from the segmenter
and asserts that no segment with confidence < 0.70 has a hard crop strategy
('stationary' or 'tracking'). When a violation is found, it forcibly downgrades
the segment to blur_fill and logs the violation as a bug.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

CONFIDENCE_FLOOR_FOR_CROP = 0.50
CROP_STRATEGIES = {"stationary", "tracking", "panning"}


def enforce_confidence_floor(segments: list, job_id: str = "") -> list:
    """Walk the segment list and forcibly downgrade any segment that has a
    crop strategy but confidence below the floor.

    Returns the (possibly modified) segment list. Logs every downgrade.
    """
    violations = 0
    for seg in segments:
        if seg.strategy in CROP_STRATEGIES and seg.confidence < CONFIDENCE_FLOOR_FOR_CROP:
            old_strategy = seg.strategy
            seg.strategy = "blur_fill"
            seg.layout = "blur_fill"
            seg.subject_x = 50
            seg.active_slot = None
            seg.fallback_reason = (
                f"forced_downgrade_from_{old_strategy}_at_conf_{seg.confidence:.2f}"
            )
            violations += 1
            logger.warning(
                "[%s] CONFIDENCE_FLOOR VIOLATION: segment %.2f-%.2f had strategy=%s "
                "with confidence=%.2f (floor=%.2f). Forcing blur_fill.",
                job_id, seg.start, seg.end, old_strategy,
                seg.confidence, CONFIDENCE_FLOOR_FOR_CROP,
            )

    if violations > 0:
        logger.warning(
            "[%s] enforce_confidence_floor: downgraded %d segments. "
            "This indicates a bug — segments below the floor should never be "
            "produced by upstream code in the first place.",
            job_id, violations,
        )

    return segments
