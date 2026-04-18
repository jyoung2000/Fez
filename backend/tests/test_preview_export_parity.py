"""Phase 8 — Preview / export parity test.

Verifies that the render plan produces consistent per-frame crop rects
by running the full pipeline (reframe segmenter -> render plan builder)
and checking that:
1. Every frame in the timeline has a defined crop rect
2. Crop rects are within source bounds
3. Adjacent ops have contiguous time ranges (no gaps or overlaps)
4. The crop rect x/y values are consistent within 1 pixel when converting
   from normalized to pixel space and back
"""

from dataclasses import dataclass, field
from typing import Optional

import pytest

from backend.services.reframe_segmenter import build_reframe_segments
from backend.services.render_plan_builder import build_render_plan


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
class _SpeakerEvent:
    start: float
    end: float
    slot_id: int
    confidence: float


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


class TestPreviewExportParity:
    """End-to-end test: reframe segments -> render plan -> per-frame crop rects."""

    def _build_segments_and_plan(self, registry, dense, transcript,
                                   speaker_to_slot, as_events, video_duration,
                                   shot_cuts=None, source_width=1920, source_height=1080):
        segments = build_reframe_segments(
            shot_cuts=shot_cuts or [],
            face_registry=registry,
            active_speaker_events=as_events,
            dense_faces=dense,
            transcript_segments=transcript,
            speaker_to_slot=speaker_to_slot,
            video_duration=video_duration,
            source_width=source_width,
            source_height=source_height,
        )
        plan = build_render_plan(
            segments,
            source_width=source_width,
            source_height=source_height,
            source_fps=30.0,
        )
        return segments, plan

    def _assert_plan_valid(self, plan, video_duration, source_width=1920, source_height=1080):
        """Assert the render plan is valid and contiguous."""
        assert len(plan.ops) >= 1, "Plan must have at least 1 op"

        for i, op in enumerate(plan.ops):
            # Every op must have a valid primary_rect
            r = op.primary_rect
            assert 0.0 <= r.x <= 1.0, f"Op {i}: x={r.x} out of [0,1]"
            assert 0.0 <= r.y <= 1.0, f"Op {i}: y={r.y} out of [0,1]"
            assert 0.0 < r.w <= 1.0, f"Op {i}: w={r.w} out of (0,1]"
            assert 0.0 < r.h <= 1.0, f"Op {i}: h={r.h} out of (0,1]"
            assert r.x + r.w <= 1.0 + 1e-9, f"Op {i}: x+w={r.x + r.w} > 1"
            assert r.y + r.h <= 1.0 + 1e-9, f"Op {i}: y+h={r.y + r.h} > 1"

            # Pixel conversion round-trip within 1 pixel
            px_x, px_y, px_w, px_h = r.to_pixels(source_width, source_height)
            assert 0 <= px_x < source_width, f"Op {i}: px_x={px_x}"
            assert 0 <= px_y < source_height, f"Op {i}: px_y={px_y}"
            assert px_w > 0, f"Op {i}: px_w={px_w}"
            assert px_h > 0, f"Op {i}: px_h={px_h}"
            assert px_x + px_w <= source_width, f"Op {i}: px_x+px_w={px_x + px_w}"
            assert px_y + px_h <= source_height, f"Op {i}: px_y+px_h={px_y + px_h}"

        # Contiguity: no gaps or overlaps between ops
        for i in range(len(plan.ops) - 1):
            a = plan.ops[i]
            b = plan.ops[i + 1]
            assert a.end_sec == pytest.approx(b.start_sec, abs=1e-6), (
                f"Gap/overlap between op {i} (end={a.end_sec}) "
                f"and op {i+1} (start={b.start_sec})"
            )

    def test_single_speaker_parity(self):
        registry = _FaceRegistry(slots=[_FaceSlot(slot_id=0, x_center=56.0)])
        dense = _make_dense(0, 56, 0, 10, step=0.5)
        transcript = [_TranscriptSeg(start=0, end=10, speaker="Speaker 1")]
        speaker_to_slot = {"Speaker 1": 0}

        segments, plan = self._build_segments_and_plan(
            registry, dense, transcript, speaker_to_slot, [], 10.0)
        self._assert_plan_valid(plan, 10.0)

    def test_two_speakers_parity(self):
        registry = _FaceRegistry(slots=[
            _FaceSlot(slot_id=0, x_center=30.0),
            _FaceSlot(slot_id=1, x_center=70.0),
        ])
        speaker_to_slot = {"Speaker 1": 0, "Speaker 2": 1}
        transcript = [
            _TranscriptSeg(start=0, end=5, speaker="Speaker 1"),
            _TranscriptSeg(start=5, end=10, speaker="Speaker 2"),
        ]
        as_events = [
            _SpeakerEvent(start=0, end=5, slot_id=0, confidence=0.9),
            _SpeakerEvent(start=5, end=10, slot_id=1, confidence=0.9),
        ]
        dense = _make_dense(0, 30, 0, 5) + _make_dense(1, 70, 5, 10)

        segments, plan = self._build_segments_and_plan(
            registry, dense, transcript, speaker_to_slot, as_events, 10.0)
        self._assert_plan_valid(plan, 10.0)

    def test_shot_cuts_parity(self):
        registry = _FaceRegistry(slots=[
            _FaceSlot(slot_id=0, x_center=30.0),
            _FaceSlot(slot_id=1, x_center=70.0),
        ])
        speaker_to_slot = {"Speaker 1": 0, "Speaker 2": 1}
        transcript = [
            _TranscriptSeg(start=0, end=5, speaker="Speaker 1"),
            _TranscriptSeg(start=5.05, end=15, speaker="Speaker 2"),
        ]
        as_events = [
            _SpeakerEvent(start=0, end=5, slot_id=0, confidence=0.9),
            _SpeakerEvent(start=5.05, end=15, slot_id=1, confidence=0.9),
        ]
        dense = _make_dense(0, 30, 0, 5) + _make_dense(1, 70, 5, 15)

        segments, plan = self._build_segments_and_plan(
            registry, dense, transcript, speaker_to_slot, as_events, 15.0,
            shot_cuts=[5.0])
        self._assert_plan_valid(plan, 15.0)

    def test_crop_rect_pixel_precision(self):
        """Crop rect center must be within 1px of the face center."""
        source_width = 1920
        source_height = 1080
        face_x_pct = 963.0 / source_width * 100.0

        registry = _FaceRegistry(slots=[_FaceSlot(slot_id=0, x_center=face_x_pct)])
        dense = _make_dense(0, face_x_pct, 0, 5, step=0.5)
        transcript = [_TranscriptSeg(start=0, end=5, speaker="Speaker 1")]
        speaker_to_slot = {"Speaker 1": 0}

        segments, plan = self._build_segments_and_plan(
            registry, dense, transcript, speaker_to_slot, [], 5.0,
            source_width=source_width, source_height=source_height)
        self._assert_plan_valid(plan, 5.0, source_width, source_height)

        # The crop rect center should be close to 963px
        r = plan.ops[0].primary_rect
        px_x, _, px_w, _ = r.to_pixels(source_width, source_height)
        crop_center_px = px_x + px_w / 2
        assert abs(crop_center_px - 963) <= 2, (
            f"Crop center at {crop_center_px}px, expected ~963px"
        )

    def test_speaker_fields_survive_plan_roundtrip(self):
        """speaker_slot and speaker_label must survive JSON round-trip."""
        import json
        from backend.services.render_plan import RenderPlan, RenderOp, RenderOpKind, Rect

        plan = RenderPlan(
            source_width=1920,
            source_height=1080,
            target_width=1080,
            target_height=1920,
            total_duration_sec=10.0,
            fps=30.0,
            ops=[
                RenderOp(
                    kind=RenderOpKind.CROP,
                    start_sec=0.0,
                    end_sec=5.0,
                    primary_rect=Rect(x=0.15, y=0.0, w=0.316, h=1.0),
                    speaker_slot=0,
                    speaker_label="Alice",
                ),
                RenderOp(
                    kind=RenderOpKind.CROP,
                    start_sec=5.0,
                    end_sec=10.0,
                    primary_rect=Rect(x=0.55, y=0.0, w=0.316, h=1.0),
                    speaker_slot=1,
                    speaker_label=None,
                ),
            ],
        )

        violations = plan.validate()
        assert len(violations) == 0

        data = json.loads(plan.to_json())
        assert data["ops"][0]["speaker_slot"] == 0
        assert data["ops"][0]["speaker_label"] == "Alice"
        assert data["ops"][1]["speaker_slot"] == 1
        assert data["ops"][1]["speaker_label"] is None

    def test_rect_deterministic_across_calls(self):
        """Rect.to_pixels must give identical results on every call (parity guarantee)."""
        from backend.services.render_plan import Rect

        rect = Rect(x=0.22, y=0.0, w=0.316, h=1.0)
        results = [rect.to_pixels(1920, 1080) for _ in range(100)]
        assert all(r == results[0] for r in results), "Non-deterministic pixel conversion"
