"""AutoFlip-style focus model: required vs non-required features and scene focus regions."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FeatureKind(str, Enum):
    FACE = "face"
    FACE_LIKE = "face_like"    # promoted from saliency, hard constraint but uncertain source
    TEXT = "text"
    HUD = "hud"
    LOGO = "logo"
    SALIENCY = "saliency"
    OBJECT = "object"


@dataclass
class RequiredFeature:
    """A bounding box that the camera should keep in frame.

    Coordinates are percentages of the source frame (0-100).
    `must_be_in_frame=True` -> hard constraint, the crop MUST contain this bbox.
    `must_be_in_frame=False` -> soft preference weighted by `weight`.
    """
    t_start: float
    t_end: float
    x: float       # center x, 0-100
    y: float       # center y, 0-100
    w: float       # width as % of source, 0-100
    h: float       # height as % of source, 0-100
    kind: FeatureKind
    weight: float              # 0.0-1.0
    must_be_in_frame: bool
    identity: Optional[int] = None  # face slot id or object class id
    excluded_from_centroid: bool = False  # True for text/HUD: must stay in frame but doesn't pull the crop center

    @property
    def left(self) -> float:
        return self.x - self.w / 2

    @property
    def right(self) -> float:
        return self.x + self.w / 2

    @property
    def top(self) -> float:
        return self.y - self.h / 2

    @property
    def bottom(self) -> float:
        return self.y + self.h / 2

    def to_dict(self) -> dict:
        def _s(v):
            return v.item() if hasattr(v, 'item') else v
        return {k: _s(getattr(self, k)) for k in
                ('t_start', 't_end', 'x', 'y', 'w', 'h', 'weight', 'must_be_in_frame', 'identity')} | {'kind': self.kind.value}


@dataclass
class SceneFocusRegion:
    """The aggregated focus geometry for one continuous shot."""
    shot_start: float
    shot_end: float
    required: list = field(default_factory=list)   # list[RequiredFeature] where must_be_in_frame=True
    optional: list = field(default_factory=list)    # list[RequiredFeature] where must_be_in_frame=False
    # Minimum bounding rect covering all `required` features across the whole shot
    min_bounding_rect: tuple = (0, 0, 0, 0)        # (x, y, w, h) in % -- center + size
    fits_target_aspect: bool = True                 # True if bounding rect fits inside target aspect window
    # Optimal crop center if it fits; otherwise the centroid of required features
    optimal_crop_center: tuple = (50, 50)           # (cx, cy) in %
    # Per-frame required-feature centers for trajectory planning
    per_frame_target: list = field(default_factory=list)  # list[(t, target_x, target_y)]


@dataclass
class SubjectTrack:
    """A unified subject track from any detection source.

    Sources: face_confirmed (from face_detector + FaceRegistry), or
    face_like_promoted (from saliency that passed the promotion gate).
    """
    track_id: int
    source: str                    # "face_confirmed" | "face_like_promoted"
    confidence: float              # 0.0-1.0
    face_slot_id: Optional[int] = None  # set when source="face_confirmed"

    # Bounding box trajectory, in % of source frame
    # list[(t, x, y, w, h)] where x,y are center coordinates
    # When source="face_like_promoted", this is EMA-smoothed. Raw values
    # are in raw_trajectory for debugging.
    bbox_trajectory: list = field(default_factory=list)
    raw_trajectory: Optional[list] = None  # unsmoothed, for debugging

    # HSV histogram appearance signature for cross-shot identity
    # Flattened numpy array, None if not yet computed
    appearance_signature: Optional[object] = None

    # Cross-shot persistent identity (resolved by SubjectRegistry)
    persistent_id: int = -1

    # Promotion gate diagnostics (only populated when source="face_like_promoted")
    gate_persistence_frames: int = 0
    gate_aspect_ratio: float = 0.0
    gate_area_ratio: float = 0.0
    gate_face_mesh_passed: bool = False

    @property
    def t_start(self) -> float:
        return self.bbox_trajectory[0][0] if self.bbox_trajectory else 0.0

    @property
    def t_end(self) -> float:
        return self.bbox_trajectory[-1][0] if self.bbox_trajectory else 0.0

    def bbox_at(self, t: float) -> Optional[tuple]:
        """Return (x, y, w, h) at time t via nearest-neighbor lookup,
        or None if t is outside the trajectory range."""
        if not self.bbox_trajectory:
            return None
        if t < self.bbox_trajectory[0][0] or t > self.bbox_trajectory[-1][0]:
            return None
        best = min(self.bbox_trajectory, key=lambda bt: abs(bt[0] - t))
        return (best[1], best[2], best[3], best[4])

    def to_dict(self) -> dict:
        def _s(v):
            return v.item() if hasattr(v, 'item') else v
        return {
            "track_id": int(self.track_id),
            "source": str(self.source),
            "confidence": _s(self.confidence),
            "face_slot_id": self.face_slot_id,
            "persistent_id": int(self.persistent_id),
            "bbox_count": len(self.bbox_trajectory),
            "t_start": _s(self.t_start),
            "t_end": _s(self.t_end),
            "gate_persistence_frames": int(self.gate_persistence_frames),
            "gate_aspect_ratio": _s(self.gate_aspect_ratio),
            "gate_area_ratio": _s(self.gate_area_ratio),
            "gate_face_mesh_passed": bool(self.gate_face_mesh_passed),
        }
