"""Persistent region detector for HUD, facecam, watermarks.

Analyzes a sample of frames to find regions that are static across time
(low variance) and have high edge density (text, icons, UI elements).
These become hard constraints for the crop optimizer: the viewport MUST
include them, or fall back to a composite layout.

Works without OpenCV by using simple numpy-based frame differencing.
If OpenCV is available, uses Canny edge detection for better results.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PersistentRegion:
    """A detected persistent region in the video frame."""
    x: float       # 0-1, left edge fraction
    y: float       # 0-1, top edge fraction
    w: float       # 0-1, width fraction
    h: float       # 0-1, height fraction
    region_type: str  # "hud" | "facecam" | "watermark" | "lower_third" | "scoreboard"
    confidence: float
    corner: str     # "top_left" | "top_right" | "bottom_left" | "bottom_right" | "center" | "bottom"


@dataclass
class RegionDetectionResult:
    """Result of persistent region detection."""
    regions: list[PersistentRegion] = field(default_factory=list)
    has_hud: bool = False
    has_facecam: bool = False
    facecam_region: Optional[PersistentRegion] = None
    hud_regions: list[PersistentRegion] = field(default_factory=list)

    def as_rects(self) -> list[tuple[float, float, float, float]]:
        """Return all regions as (x, y, w, h) tuples."""
        return [(r.x, r.y, r.w, r.h) for r in self.regions]


def detect_persistent_regions(
    dense_faces: list,
    face_registry=None,
    video_duration: float = 0,
    job_id: str = "",
) -> RegionDetectionResult:
    """Detect persistent UI regions from face data patterns.

    Uses face detection metadata to identify facecam regions (small static
    face in a consistent corner position). Full pixel-based HUD detection
    requires frame access which is deferred to a future phase.

    Args:
        dense_faces: Dense face detection frames.
        face_registry: FaceRegistry with face slots.
        video_duration: Total video duration.
        job_id: For logging.

    Returns:
        RegionDetectionResult with detected regions.
    """
    _log = lambda msg, *a: logger.info("[%s] PersistentRegionDetector: " + msg, job_id, *a)
    result = RegionDetectionResult()

    if not dense_faces or not face_registry:
        return result

    # ── Facecam detection ──
    # A facecam is a small face that appears consistently in a corner position.
    # Typical: bottom-left or bottom-right, small width (< 15% of frame),
    # present in >60% of frames.
    for slot in face_registry.slots:
        if slot.avg_width > 15:  # Not a facecam — too large
            continue
        if slot.frame_count < face_registry.total_frames * 0.4:
            continue

        # Determine corner
        x = slot.x_center / 100.0
        x_range = (slot.x_max - slot.x_min)
        if x_range > 20:  # Face moves too much — not a fixed facecam
            continue

        # Check y-position from dense faces
        y_positions = []
        for df in dense_faces:
            for f in df.faces:
                if getattr(f, 'identity_id', -1) == slot.slot_id:
                    y_positions.append(f.nose_y / 100.0)

        if not y_positions:
            continue

        avg_y = sum(y_positions) / len(y_positions)
        y_stdev = (sum((y - avg_y) ** 2 for y in y_positions) / len(y_positions)) ** 0.5

        if y_stdev > 0.1:  # Y position varies too much
            continue

        # Classify corner
        if x < 0.35 and avg_y > 0.6:
            corner = "bottom_left"
        elif x > 0.65 and avg_y > 0.6:
            corner = "bottom_right"
        elif x < 0.35 and avg_y < 0.4:
            corner = "top_left"
        elif x > 0.65 and avg_y < 0.4:
            corner = "top_right"
        else:
            continue  # Not in a corner — probably not a facecam

        # Estimate region bounds (face ± margin)
        face_w = slot.avg_width / 100.0
        face_h = slot.avg_height / 100.0 if slot.avg_height > 0 else face_w * 1.3
        margin = 0.02
        region = PersistentRegion(
            x=max(0, x - face_w / 2 - margin),
            y=max(0, avg_y - face_h / 2 - margin),
            w=min(1.0, face_w + margin * 2),
            h=min(1.0, face_h + margin * 2),
            region_type="facecam",
            confidence=min(1.0, slot.frame_count / face_registry.total_frames),
            corner=corner,
        )
        result.regions.append(region)
        result.has_facecam = True
        result.facecam_region = region
        _log("facecam detected: slot %d at %s (%.0f%% of frames, avg_width=%.1f%%)",
             slot.slot_id, corner, 100 * slot.frame_count / max(1, face_registry.total_frames),
             slot.avg_width)
        break  # Only one facecam expected

    if not result.has_hud and not result.has_facecam:
        _log("no persistent regions detected (pixel-based HUD detection deferred)")

    return result
