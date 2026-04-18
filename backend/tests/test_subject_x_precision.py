"""Phase 1 — Failing test: pixel-precise subject_x.

The current implementation snaps subject_x to an int 0-100, giving ~19.2px
buckets on a 1920px source.  A face centered at pixel 963 should produce
a crop center at 963 +/- 1 pixel, not snapped to a coarse bucket.

This test will require the Phase 3 refactor (subject_x as float in source
pixel units) to pass.
"""

from dataclasses import dataclass, field
from typing import Optional

import pytest

from backend.services.reframe_segmenter import build_reframe_segments


# ── Lightweight stubs ──

@dataclass
class _FaceSlot:
    slot_id: int
    x_center: float
    x_min: float = 0.0
    x_max: float = 100.0
    frame_count: int = 100
    avg_width: float = 10.0
    avg_height: float = 12.0


@dataclass
class _FaceRegistry:
    slots: list[_FaceSlot] = field(default_factory=list)
    total_frames: int = 100
    frames_with_faces: int = 100

    @property
    def multi_speaker(self) -> bool:
        return len(self.slots) >= 2

    def slot_by_id(self, slot_id: int):
        for s in self.slots:
            if s.slot_id == slot_id:
                return s
        return None

    def nearest_slot(self, x: float):
        if not self.slots:
            return None
        return min(self.slots, key=lambda s: abs(s.x_center - x))


@dataclass
class _FaceInfo:
    identity_id: int
    nose_x: float
    nose_y: float = 50.0
    width: float = 10.0
    height: float = 12.0
    lip_aperture: float = 0.0
    is_speaking: bool = False
    confidence: float = 0.9


@dataclass
class _FrameFaces:
    timestamp: float
    frame_path: str = ""
    faces: list[_FaceInfo] = field(default_factory=list)
    primary_face_idx: int = -1


@dataclass
class _TranscriptSeg:
    start: float
    end: float
    text: str = ""
    speaker: str = "Speaker 1"
    confidence: Optional[float] = 0.9
    words: Optional[list] = None
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None


def _make_dense(slot_id, x, start, end, step=0.5):
    frames = []
    t = start
    while t < end:
        frames.append(_FrameFaces(
            timestamp=round(t, 4),
            faces=[_FaceInfo(identity_id=slot_id, nose_x=x)],
        ))
        t += step
    return frames


class TestSubjectXPixelPrecision:
    """subject_x must be in source pixel units with sub-bucket precision."""

    def test_face_at_pixel_963_produces_precise_crop_center(self):
        """Face centered at pixel 963 in 1920px source must give subject_x
        that resolves to within 1px of 963, not a 19.2px bucket."""
        source_width = 1920
        # x_center in 0-100 space: 963/1920 * 100 = 50.15625
        face_x_pct = 963.0 / source_width * 100.0  # 50.15625

        registry = _FaceRegistry(slots=[
            _FaceSlot(slot_id=0, x_center=face_x_pct),
        ])
        dense = _make_dense(0, face_x_pct, 0, 10, step=0.5)
        transcript = [_TranscriptSeg(start=0, end=10, speaker="Speaker 1")]
        speaker_to_slot = {"Speaker 1": 0}

        segments = build_reframe_segments(
            shot_cuts=[],
            face_registry=registry,
            active_speaker_events=[],
            dense_faces=dense,
            transcript_segments=transcript,
            speaker_to_slot=speaker_to_slot,
            video_duration=10.0,
            source_width=source_width,
        )

        assert len(segments) >= 1
        seg = segments[0]

        # After Phase 3: subject_x should be in pixel units (float ~963.0)
        # The resolved pixel center should be within 1px of the actual face center
        if isinstance(seg.subject_x, float) and seg.subject_x > 100:
            # Phase 3 pixel-precise: subject_x is in pixel units
            assert abs(seg.subject_x - 963.0) <= 1.0, (
                f"subject_x={seg.subject_x} is not within 1px of 963"
            )
        else:
            # Current int 0-100: will be 50 (snapped), which maps to pixel 960
            # 960 is 3 pixels off from 963 — fails the precision requirement
            resolved_pixel = seg.subject_x / 100.0 * source_width
            assert abs(resolved_pixel - 963.0) <= 1.0, (
                f"subject_x={seg.subject_x} (0-100 int) resolves to pixel "
                f"{resolved_pixel}, which is {abs(resolved_pixel - 963.0):.1f}px "
                f"from face center at 963. Coarse quantisation detected."
            )

    def test_face_at_pixel_137_precise(self):
        """Face at pixel 137 — the 0-100 bucket would give 7 (=134.4px), 3px off."""
        source_width = 1920
        face_x_pct = 137.0 / source_width * 100.0  # 7.135416...

        registry = _FaceRegistry(slots=[
            _FaceSlot(slot_id=0, x_center=face_x_pct),
        ])
        dense = _make_dense(0, face_x_pct, 0, 10, step=0.5)
        transcript = [_TranscriptSeg(start=0, end=10, speaker="Speaker 1")]
        speaker_to_slot = {"Speaker 1": 0}

        segments = build_reframe_segments(
            shot_cuts=[],
            face_registry=registry,
            active_speaker_events=[],
            dense_faces=dense,
            transcript_segments=transcript,
            speaker_to_slot=speaker_to_slot,
            video_duration=10.0,
            source_width=source_width,
        )

        assert len(segments) >= 1
        seg = segments[0]

        if isinstance(seg.subject_x, float) and seg.subject_x > 100:
            assert abs(seg.subject_x - 137.0) <= 1.0, (
                f"subject_x={seg.subject_x} is not within 1px of 137"
            )
        else:
            resolved_pixel = seg.subject_x / 100.0 * source_width
            assert abs(resolved_pixel - 137.0) <= 1.0, (
                f"subject_x={seg.subject_x} resolves to {resolved_pixel:.1f}px, "
                f"expected 137 +/-1"
            )
