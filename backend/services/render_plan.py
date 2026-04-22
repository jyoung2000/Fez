"""RenderPlan: the single intermediate representation for preview/export parity.

Both the frontend Canvas preview renderer and the backend FFmpeg filter builder
consume this same structure. All coordinates are normalized floats (0.0-1.0)
relative to source video dimensions. Pixel rounding happens at exactly one
place in each renderer.

Time references are seconds relative to the RenderPlan's timeline, which
always starts at 0.0 regardless of whether it's a clip or full video.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)

USE_RENDER_PLAN = os.environ.get("USE_RENDER_PLAN", "true").lower() in ("true", "1", "yes")


class RenderOpKind(str, Enum):
    CROP = "crop"                          # static rectangle crop (STATIONARY)
    TRACKING_CROP = "tracking_crop"        # animated crop via motion_path
    WIDE_MASTER = "wide_master"            # letterbox: source centered, black bars
    BLUR_FILL = "blur_fill"               # source centered, blurred dup as bg
    SPLIT_SCREEN = "split_screen"         # 2 crops stacked vertically, each 50%
    STACKED_GAMEPLAY = "stacked_gameplay"  # gameplay top 60%, facecam bottom 40%
    GRID_2X2 = "grid_2x2"                 # 4 tiles in 2x2
    # Phase 5: motivated zoom. Time-varying crop w/h (not zoompan) that
    # produces a smooth push-in or pull-out between ``primary_rect`` and
    # a target crop in ``motion_path[-1].rect``. Compatible with
    # TRACKING_CROP in that ``motion_path`` carries a 2-keypoint ramp;
    # the FFmpeg builder interpolates w/h linearly along with x/y.
    MOTIVATED_PUSH_IN = "motivated_push_in"
    MOTIVATED_PULL_OUT = "motivated_pull_out"
    # Blueprint v2 Phase 0: slow-push cinematography for scenes with
    # no tracked subject (landscape / B-roll / establishing). Total
    # travel capped by ``reframe_config.ken_burns_max_travel_frac``;
    # zoom capped by ``ken_burns_max_zoom``. Two keypoints in
    # ``motion_path`` (start rect, end rect) interpolated with a
    # cosine ease so the push eases in and out.
    KEN_BURNS = "ken_burns"


@dataclass
class Rect:
    """Normalized 0.0-1.0, relative to source video dimensions."""
    x: float
    y: float
    w: float
    h: float

    def to_pixels(self, source_w: int, source_h: int) -> tuple:
        """Convert normalized rect to pixel values with even-dimension enforcement.

        Returns (px_x, px_y, px_w, px_h) with even w/h.
        """
        pw = int(round(self.w * source_w))
        ph = int(round(self.h * source_h))
        # Enforce even dimensions for FFmpeg
        pw = pw - (pw % 2)
        ph = ph - (ph % 2)
        px = int(round(self.x * source_w))
        py = int(round(self.y * source_h))
        # Clamp to source bounds
        px = max(0, min(px, source_w - pw))
        py = max(0, min(py, source_h - ph))
        return (px, py, pw, ph)


@dataclass
class MotionKeypoint:
    """A keypoint in a motion path for TRACKING_CROP ops."""
    t: float          # seconds relative to the RenderOp's start_sec
    rect: Rect


@dataclass
class RenderOp:
    """A single reframing operation over a time window.

    primary_rect is always populated. Others depend on kind:
      CROP, TRACKING_CROP, WIDE_MASTER, BLUR_FILL -> primary only
      SPLIT_SCREEN -> primary (top) + secondary (bottom)
      STACKED_GAMEPLAY -> primary (gameplay) + secondary (facecam)
      GRID_2X2 -> primary, secondary, tertiary, quaternary
    """
    kind: RenderOpKind
    start_sec: float
    end_sec: float
    primary_rect: Rect
    secondary_rect: Optional[Rect] = None
    tertiary_rect: Optional[Rect] = None
    quaternary_rect: Optional[Rect] = None
    # TRACKING_CROP populates this; ignored otherwise
    motion_path: List[MotionKeypoint] = field(default_factory=list)
    # Transition duration INTO this op, from the previous one.
    # 0 = hard cut (use when crossing a shot boundary).
    # 400-600 for within-shot eases.
    ease_in_ms: int = 0
    # For debugging / logs only, not rendering
    strategy_label: str = ""
    content_type: str = "unknown"
    # Phase 5 (gaming): per-segment gaming layout mode populated
    # by the chooser in reframe_segmenter for gameplay clips. One
    # of "fullscreen" | "blurfill" | "composite" | "wide_zoom" or
    # ``None`` for non-gaming ops. The frontend Canvas renderer
    # and the FFmpeg filter graph builder both consume this to
    # pick the right draw path.
    gaming_layout_mode: Optional[str] = None
    speaker_slot: Optional[int] = None   # slot_id driving this crop segment, None if unknown
    speaker_label: Optional[str] = None  # resolved display name ("Alice" if renamed, else None)


@dataclass
class RenderPlan:
    """The complete render plan for a video or clip export.

    Gap-free and contiguous: op N's end_sec equals op N+1's start_sec exactly,
    covering every moment of the timeline.
    """
    source_width: int           # original video dimensions
    source_height: int
    target_width: int           # output dimensions, e.g. 1080
    target_height: int          # e.g. 1920
    total_duration_sec: float
    fps: float
    ops: List[RenderOp]
    # For clip exports: offset into source where this plan's timeline begins.
    # The timeline itself still starts at 0.0; this is metadata for the exporter
    # to know where to seek in the source file.
    source_offset_sec: float = 0.0

    def to_json(self) -> str:
        """Serialize to JSON with enum-aware encoding."""
        return json.dumps(asdict(self), cls=_EnumEncoder, separators=(",", ":"))

    def to_dict(self) -> dict:
        """Serialize to dict with enum-aware conversion."""
        return json.loads(self.to_json())

    def validate(self) -> List[str]:
        """Check invariants. Returns list of violation descriptions (empty = valid)."""
        violations = []

        if not self.ops:
            violations.append("RenderPlan has no ops")
            return violations

        # First op must start at 0.0
        if abs(self.ops[0].start_sec) > 0.001:
            violations.append(
                f"First op starts at {self.ops[0].start_sec:.4f}, expected 0.0"
            )

        # Last op must end at total_duration_sec
        if abs(self.ops[-1].end_sec - self.total_duration_sec) > 0.05:
            violations.append(
                f"Last op ends at {self.ops[-1].end_sec:.4f}, "
                f"expected {self.total_duration_sec:.4f}"
            )

        for i, op in enumerate(self.ops):
            prefix = f"Op[{i}] ({op.kind.value} {op.start_sec:.2f}-{op.end_sec:.2f})"

            # Duration must be positive
            if op.end_sec <= op.start_sec:
                violations.append(f"{prefix}: end_sec <= start_sec")

            # Primary rect always required
            if op.primary_rect is None:
                violations.append(f"{prefix}: missing primary_rect")
            else:
                _check_rect(violations, f"{prefix}.primary_rect", op.primary_rect)

            # Kind-specific rect requirements
            if op.kind in (RenderOpKind.SPLIT_SCREEN, RenderOpKind.STACKED_GAMEPLAY):
                if op.secondary_rect is None:
                    violations.append(f"{prefix}: {op.kind.value} requires secondary_rect")
                else:
                    _check_rect(violations, f"{prefix}.secondary_rect", op.secondary_rect)

            if op.kind == RenderOpKind.GRID_2X2:
                for label, rect in [
                    ("secondary_rect", op.secondary_rect),
                    ("tertiary_rect", op.tertiary_rect),
                    ("quaternary_rect", op.quaternary_rect),
                ]:
                    if rect is None:
                        violations.append(f"{prefix}: grid_2x2 requires {label}")
                    else:
                        _check_rect(violations, f"{prefix}.{label}", rect)

            # TRACKING_CROP / motivated-zoom / Ken Burns ops must have motion_path
            if op.kind in (
                RenderOpKind.TRACKING_CROP,
                RenderOpKind.MOTIVATED_PUSH_IN,
                RenderOpKind.MOTIVATED_PULL_OUT,
                RenderOpKind.KEN_BURNS,
            ):
                if not op.motion_path:
                    violations.append(
                        f"{prefix}: {op.kind.value} requires motion_path",
                    )
                else:
                    for j, kp in enumerate(op.motion_path):
                        _check_rect(
                            violations,
                            f"{prefix}.motion_path[{j}].rect",
                            kp.rect,
                        )

            # Contiguity: this op's end must equal next op's start
            if i < len(self.ops) - 1:
                gap = abs(self.ops[i + 1].start_sec - op.end_sec)
                if gap > 0.001:
                    violations.append(
                        f"Gap between Op[{i}] end ({op.end_sec:.4f}) and "
                        f"Op[{i+1}] start ({self.ops[i+1].start_sec:.4f}): "
                        f"{gap:.4f}s"
                    )

        return violations


def _check_rect(violations: list, label: str, rect: Rect):
    """Validate a normalized Rect's coordinates are in [0, 1]."""
    for attr in ("x", "y", "w", "h"):
        val = getattr(rect, attr)
        if val < -0.01 or val > 1.01:
            violations.append(f"{label}.{attr} = {val:.4f} outside [0, 1]")


class _EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)
