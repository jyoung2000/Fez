"""Per-source-frame subject position timeline.

Built by dense_propagator from sparse detector anchors plus OpenCV tracker
propagation. Consumed by intent_tracker, scene_focus, autoflip_segmenter,
and ffmpeg_filter_builder as a single shared source of per-frame positions.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FrameSample:
    """One source frame's worth of subject positions."""
    timestamp: float
    fps: float
    # {slot_id: (cx, cy, w, h)} in % of source frame coordinates
    bboxes: dict = field(default_factory=dict)
    # Per-slot confidence in [0, 1] — high near detection anchors, decays during propagation
    confidences: dict = field(default_factory=dict)
    # True if this frame is a real detector output, False if propagated
    is_anchor: bool = False
    # True if any tracker was reset on this frame (drift correction)
    had_reset: bool = False


@dataclass
class InterpolatedFaceTimeline:
    """Complete per-source-frame timeline of subject positions."""
    samples: list = field(default_factory=list)  # list[FrameSample]
    source_fps: float = 30.0
    source_width: int = 1920
    source_height: int = 1080
    # Quick lookup index: round(t, 3) -> samples index
    _index: dict = field(default_factory=dict)

    def build_index(self):
        self._index = {round(s.timestamp, 3): i for i, s in enumerate(self.samples)}

    def at(self, timestamp: float) -> Optional[FrameSample]:
        """Nearest sample to a timestamp via index lookup."""
        if not self.samples:
            return None
        if not self._index:
            self.build_index()
        key = round(timestamp, 3)
        if key in self._index:
            return self.samples[self._index[key]]
        # Fall back to nearest
        best = min(self.samples, key=lambda s: abs(s.timestamp - timestamp))
        return best

    def slot_positions_in_range(self, slot_id: int, t_start: float, t_end: float) -> list:
        """Return [(t, cx, cy, w, h, confidence), ...] for one slot in a time range."""
        out = []
        for s in self.samples:
            if s.timestamp < t_start or s.timestamp > t_end:
                continue
            if slot_id in s.bboxes:
                cx, cy, w, h = s.bboxes[slot_id]
                conf = s.confidences.get(slot_id, 0.5)
                out.append((s.timestamp, cx, cy, w, h, conf))
        return out

    def to_dict_summary(self) -> dict:
        n_anchors = sum(1 for s in self.samples if s.is_anchor)
        n_resets = sum(1 for s in self.samples if s.had_reset)
        return {
            "n_samples": len(self.samples),
            "n_anchors": n_anchors,
            "n_resets": n_resets,
            "source_fps": float(self.source_fps),
            "duration": float(self.samples[-1].timestamp) if self.samples else 0.0,
        }
