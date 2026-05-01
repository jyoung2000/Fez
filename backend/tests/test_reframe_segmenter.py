"""Tests for the ReframeSegmenter.

Covers stability, turn-taking, rapid switch suppression, shot-cut snap,
in-shot speaker turn ease, wide master on crowd, and anticipation offset.
"""
import random
from dataclasses import dataclass, field
from typing import Optional

import pytest

from backend.services.reframe_segmenter import (
    MIN_HOLD_SECONDS,
    ANTICIPATION_MS,
    EASE_SHOT_CUT_MS,
    EASE_SPEAKER_TURN_MS,
    WIDE_MASTER_X,
    ReframeSegment,
    _resolve_slot_for_interval,
    _slot_visible_in_interval,
    build_reframe_segments,
)


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


# ── Helpers ──

def _make_dense_frames(slot_id, x, start, end, step=0.5, noise=0):
    """Generate dense face frames for one speaker."""
    frames = []
    t = start
    while t < end:
        nx = x + (random.uniform(-noise, noise) if noise else 0)
        frames.append(_FrameFaces(
            timestamp=t,
            faces=[_FaceInfo(identity_id=slot_id, nose_x=nx)],
        ))
        t += step
    return frames


def _make_registry_1(x=56.0):
    return _FaceRegistry(slots=[_FaceSlot(slot_id=0, x_center=x)])


def _make_registry_2(x0=30.0, x1=70.0):
    return _FaceRegistry(slots=[
        _FaceSlot(slot_id=0, x_center=x0),
        _FaceSlot(slot_id=1, x_center=x1),
    ])


def _make_registry_4():
    return _FaceRegistry(slots=[
        _FaceSlot(slot_id=i, x_center=20 + i * 20)
        for i in range(4)
    ])


# ── Tests ──

class TestStability:
    """1. Single-speaker 30s clip with noisy dense data → 1 segment, stable subject_x."""

    def test_single_speaker_no_jitter(self):
        random.seed(42)
        registry = _make_registry_1(x=56.0)
        dense = _make_dense_frames(0, 56, 0, 30, step=0.5, noise=5)
        transcript = [_TranscriptSeg(start=0, end=30, speaker="Speaker 1")]
        speaker_to_slot = {"Speaker 1": 0}

        segments = build_reframe_segments(
            shot_cuts=[],
            face_registry=registry,
            active_speaker_events=[],
            dense_faces=dense,
            transcript_segments=transcript,
            speaker_to_slot=speaker_to_slot,
            video_duration=30.0,
        )

        assert len(segments) == 1
        # subject_x is now in pixel space: 56/100 * 1920 = 1075.2
        expected_px = 56.0 / 100.0 * 1920
        assert abs(segments[0].subject_x - expected_px) < 1.0


class TestTurnTaking:
    """2. Two speakers alternating every 3s for 30s → 10 segments, alternating slots."""

    def test_alternating_speakers(self):
        registry = _make_registry_2(x0=30, x1=70)
        transcript = []
        dense = []
        as_events = []
        speaker_to_slot = {"Speaker 1": 0, "Speaker 2": 1}

        for i in range(10):
            t_start = i * 3.0
            t_end = (i + 1) * 3.0
            spk = "Speaker 1" if i % 2 == 0 else "Speaker 2"
            slot = 0 if i % 2 == 0 else 1
            x = 30 if slot == 0 else 70

            transcript.append(_TranscriptSeg(
                start=t_start, end=t_end, speaker=spk, confidence=0.9,
            ))
            as_events.append(_SpeakerEvent(
                start=t_start, end=t_end, slot_id=slot, confidence=0.9,
            ))
            dense.extend(_make_dense_frames(slot, x, t_start, t_end, step=0.5))

        segments = build_reframe_segments(
            shot_cuts=[],
            face_registry=registry,
            active_speaker_events=as_events,
            dense_faces=dense,
            transcript_segments=transcript,
            speaker_to_slot=speaker_to_slot,
            video_duration=30.0,
        )

        assert len(segments) == 10
        for i, seg in enumerate(segments):
            expected_slot = 0 if i % 2 == 0 else 1
            assert seg.active_slot == expected_slot, f"Segment {i}: expected slot {expected_slot}, got {seg.active_slot}"


class TestRapidSwitchSuppression:
    """3. Speaker A for 10s, B for 0.5s, A for 10s.

    With MIN_HOLD=0.12, a 0.5s high-confidence switch survives — this is
    correct human-editor behavior.  A *low-confidence* sub-0.12s blip would
    be merged, but 0.5s at confidence 0.9 is a genuine speaker turn.
    """

    def test_brief_high_confidence_interjection_survives(self):
        registry = _make_registry_2(x0=30, x1=70)
        speaker_to_slot = {"Speaker 1": 0, "Speaker 2": 1}

        transcript = [
            _TranscriptSeg(start=0, end=10, speaker="Speaker 1", confidence=0.9),
            _TranscriptSeg(start=10, end=10.5, speaker="Speaker 2", confidence=0.9),
            _TranscriptSeg(start=10.5, end=20.5, speaker="Speaker 1", confidence=0.9),
        ]
        as_events = [
            _SpeakerEvent(start=0, end=10, slot_id=0, confidence=0.9),
            _SpeakerEvent(start=10, end=10.5, slot_id=1, confidence=0.9),
            _SpeakerEvent(start=10.5, end=20.5, slot_id=0, confidence=0.9),
        ]
        dense = (
            _make_dense_frames(0, 30, 0, 10, step=0.5)
            + _make_dense_frames(1, 70, 10, 10.5, step=0.5)
            + _make_dense_frames(0, 30, 10.5, 20.5, step=0.5)
        )

        segments = build_reframe_segments(
            shot_cuts=[],
            face_registry=registry,
            active_speaker_events=as_events,
            dense_faces=dense,
            transcript_segments=transcript,
            speaker_to_slot=speaker_to_slot,
            video_duration=20.5,
        )

        # 0.5s B at confidence 0.9 should survive (A, B, A = 3 segments)
        assert len(segments) == 3
        assert segments[0].active_slot == 0
        assert segments[1].active_slot == 1
        assert segments[2].active_slot == 0


class TestShotCutSnap:
    """4. Shot cut at t=5.0s with speaker change at t=5.05s → ease_in_ms must be 0."""

    def test_shot_cut_zero_ease(self):
        registry = _make_registry_2(x0=30, x1=70)
        speaker_to_slot = {"Speaker 1": 0, "Speaker 2": 1}

        transcript = [
            _TranscriptSeg(start=0, end=5, speaker="Speaker 1", confidence=0.9),
            _TranscriptSeg(start=5.05, end=15, speaker="Speaker 2", confidence=0.9),
        ]
        as_events = [
            _SpeakerEvent(start=0, end=5, slot_id=0, confidence=0.9),
            _SpeakerEvent(start=5.05, end=15, slot_id=1, confidence=0.9),
        ]
        dense = (
            _make_dense_frames(0, 30, 0, 5, step=0.5)
            + _make_dense_frames(1, 70, 5, 15, step=0.5)
        )

        segments = build_reframe_segments(
            shot_cuts=[5.0],
            face_registry=registry,
            active_speaker_events=as_events,
            dense_faces=dense,
            transcript_segments=transcript,
            speaker_to_slot=speaker_to_slot,
            video_duration=15.0,
        )

        # Find the segment that starts near the shot cut
        cut_seg = next(
            (s for s in segments if abs(s.start - 5.0) < 0.3),
            None,
        )
        assert cut_seg is not None, "Should have segment near shot cut at 5.0s"
        assert cut_seg.ease_in_ms == 0, f"Shot cut should snap (ease=0), got {cut_seg.ease_in_ms}"


class TestInShotSpeakerTurn:
    """5. Speaker change at t=10s, no shot cut → ease_in_ms must be 500."""

    def test_speaker_turn_eases(self):
        registry = _make_registry_2(x0=30, x1=70)
        speaker_to_slot = {"Speaker 1": 0, "Speaker 2": 1}

        transcript = [
            _TranscriptSeg(start=0, end=10, speaker="Speaker 1", confidence=0.9),
            _TranscriptSeg(start=10, end=20, speaker="Speaker 2", confidence=0.9),
        ]
        as_events = [
            _SpeakerEvent(start=0, end=10, slot_id=0, confidence=0.9),
            _SpeakerEvent(start=10, end=20, slot_id=1, confidence=0.9),
        ]
        dense = (
            _make_dense_frames(0, 30, 0, 10, step=0.5)
            + _make_dense_frames(1, 70, 10, 20, step=0.5)
        )

        segments = build_reframe_segments(
            shot_cuts=[],
            face_registry=registry,
            active_speaker_events=as_events,
            dense_faces=dense,
            transcript_segments=transcript,
            speaker_to_slot=speaker_to_slot,
            video_duration=20.0,
        )

        # Find the segment starting at the speaker change
        turn_seg = next(
            (s for s in segments if s.active_slot == 1),
            None,
        )
        assert turn_seg is not None, "Should have segment for Speaker 2"
        assert turn_seg.ease_in_ms == EASE_SPEAKER_TURN_MS, (
            f"In-shot speaker turn should ease ({EASE_SPEAKER_TURN_MS}ms), got {turn_seg.ease_in_ms}"
        )


class TestWideMasterOnCrowd:
    """6. 4 face slots all active in a 5s window → WIDE_MASTER."""

    def test_four_faces_wide(self):
        registry = _make_registry_4()
        speaker_to_slot = {}

        # All 4 faces visible in every dense frame for 5 seconds
        dense = []
        t = 0.0
        while t < 5.0:
            dense.append(_FrameFaces(
                timestamp=t,
                faces=[
                    _FaceInfo(identity_id=0, nose_x=20),
                    _FaceInfo(identity_id=1, nose_x=40),
                    _FaceInfo(identity_id=2, nose_x=60),
                    _FaceInfo(identity_id=3, nose_x=80),
                ],
            ))
            t += 0.5

        segments = build_reframe_segments(
            shot_cuts=[],
            face_registry=registry,
            active_speaker_events=[],
            dense_faces=dense,
            transcript_segments=[],
            speaker_to_slot=speaker_to_slot,
            video_duration=5.0,
        )

        # Should have wide master for this segment
        wide_segs = [s for s in segments if s.layout == "wide_master"]
        assert len(wide_segs) >= 1, "Should have at least one WIDE_MASTER segment"
        # subject_x is now in pixel space: center = 1920/2 = 960
        assert abs(wide_segs[0].subject_x - 960.0) < 1.0


class TestAnticipation:
    """7. Speaker-turn segment at t=10.0s → output segment.start must be 9.8s."""

    def test_anticipation_offset(self):
        registry = _make_registry_2(x0=30, x1=70)
        speaker_to_slot = {"Speaker 1": 0, "Speaker 2": 1}

        transcript = [
            _TranscriptSeg(start=0, end=10, speaker="Speaker 1", confidence=0.9),
            _TranscriptSeg(start=10, end=20, speaker="Speaker 2", confidence=0.9),
        ]
        as_events = [
            _SpeakerEvent(start=0, end=10, slot_id=0, confidence=0.9),
            _SpeakerEvent(start=10, end=20, slot_id=1, confidence=0.9),
        ]
        dense = (
            _make_dense_frames(0, 30, 0, 10, step=0.5)
            + _make_dense_frames(1, 70, 10, 20, step=0.5)
        )

        segments = build_reframe_segments(
            shot_cuts=[],
            face_registry=registry,
            active_speaker_events=as_events,
            dense_faces=dense,
            transcript_segments=transcript,
            speaker_to_slot=speaker_to_slot,
            video_duration=20.0,
        )

        # Find the segment for Speaker 2 — should start at 9.8s (200ms earlier)
        turn_seg = next(
            (s for s in segments if s.active_slot == 1),
            None,
        )
        assert turn_seg is not None, "Should have segment for Speaker 2"
        expected_start = 10.0 - (ANTICIPATION_MS / 1000.0)
        assert abs(turn_seg.start - expected_start) < 0.01, (
            f"Anticipated start should be {expected_start}, got {turn_seg.start}"
        )


class TestMinimumHoldEnforcement:
    """No segment shorter than MIN_HOLD_SECONDS in final output."""

    def test_no_short_segments(self):
        registry = _make_registry_2(x0=30, x1=70)
        speaker_to_slot = {"Speaker 1": 0, "Speaker 2": 1}

        # Create many rapid switches
        transcript = []
        as_events = []
        for i in range(20):
            t = i * 1.0
            spk = "Speaker 1" if i % 2 == 0 else "Speaker 2"
            slot = 0 if i % 2 == 0 else 1
            transcript.append(_TranscriptSeg(start=t, end=t + 1.0, speaker=spk, confidence=0.9))
            as_events.append(_SpeakerEvent(start=t, end=t + 1.0, slot_id=slot, confidence=0.9))

        dense = (
            _make_dense_frames(0, 30, 0, 20, step=0.5)
        )

        segments = build_reframe_segments(
            shot_cuts=[],
            face_registry=registry,
            active_speaker_events=as_events,
            dense_faces=dense,
            transcript_segments=transcript,
            speaker_to_slot=speaker_to_slot,
            video_duration=20.0,
        )

        for seg in segments:
            dur = seg.end - seg.start
            # Allow tiny float tolerance
            assert dur >= MIN_HOLD_SECONDS - 0.01, (
                f"Segment {seg.start:.2f}-{seg.end:.2f} ({dur:.2f}s) shorter than MIN_HOLD ({MIN_HOLD_SECONDS}s)"
            )


# ── Off-screen-active-speaker visibility gate ────────────────────


class TestOffscreenSpeakerVisibilityGate:
    """The audio-attributed slot (transcript Priority 1, active-speaker
    majority Priority 2) should NOT win when the dense face track shows
    that slot has no face on camera in the interval. The fall-through
    gives Priority 3/4 a chance to pick the visible speaker instead."""

    def test_transcript_priority_skips_offscreen_speaker(self):
        # Speaker 1 → slot 2 covers 100% of [0, 5], but dense_faces only
        # contains slot 4 (the visible reaction shot). Expect the
        # function to NOT return active_speaker_slot.
        registry = _FaceRegistry(slots=[
            _FaceSlot(slot_id=2, x_center=50.0),
            _FaceSlot(slot_id=4, x_center=80.0),
        ])
        transcript = [_TranscriptSeg(start=0.0, end=5.0, speaker="Speaker 1")]
        dense = _make_dense_frames(slot_id=4, x=80.0, start=0.0, end=5.0, step=0.25)
        slot, conf, layout, source = _resolve_slot_for_interval(
            start=0.0, end=5.0,
            transcript_segments=transcript,
            speaker_to_slot={"Speaker 1": 2},
            active_speaker_events=[],
            dense_faces=dense,
            face_registry=registry,
        )
        # The audio-attributed slot 2 has no faces in the interval; the
        # off-screen-speaker guard must drop it. Either Priority 3
        # (dense dominance) picks slot 4, or we fall through to wide.
        assert source != "active_speaker_slot", (
            "transcript priority must not crown an off-camera speaker"
        )

    def test_transcript_priority_keeps_visible_speaker(self):
        # Sanity: when the audio-attributed slot IS visible, behavior
        # is unchanged (Priority 1 still fires).
        registry = _FaceRegistry(slots=[
            _FaceSlot(slot_id=2, x_center=50.0),
        ])
        transcript = [_TranscriptSeg(start=0.0, end=5.0, speaker="Speaker 1")]
        dense = _make_dense_frames(slot_id=2, x=50.0, start=0.0, end=5.0, step=0.25)
        slot, conf, layout, source = _resolve_slot_for_interval(
            start=0.0, end=5.0,
            transcript_segments=transcript,
            speaker_to_slot={"Speaker 1": 2},
            active_speaker_events=[],
            dense_faces=dense,
            face_registry=registry,
        )
        assert slot == 2
        assert source == "active_speaker_slot"

    def test_active_speaker_majority_skips_offscreen_speaker(self):
        # No transcript path; active-speaker events say slot 0 is the
        # majority but no slot-0 faces are visible. Don't crown slot 0.
        registry = _FaceRegistry(slots=[
            _FaceSlot(slot_id=0, x_center=20.0),
            _FaceSlot(slot_id=1, x_center=80.0),
        ])
        as_events = [_SpeakerEvent(start=0.0, end=5.0, slot_id=0, confidence=0.9)]
        dense = _make_dense_frames(slot_id=1, x=80.0, start=0.0, end=5.0, step=0.25)
        slot, conf, layout, source = _resolve_slot_for_interval(
            start=0.0, end=5.0,
            transcript_segments=[],
            speaker_to_slot={},
            active_speaker_events=as_events,
            dense_faces=dense,
            face_registry=registry,
        )
        assert source != "active_speaker_slot"

    def test_slot_visible_helper_threshold(self):
        # ≥30% of frames must contain the slot to pass.
        dense_30 = (
            _make_dense_frames(slot_id=4, x=80.0, start=0.0, end=1.0, step=0.1)
            + _make_dense_frames(slot_id=2, x=50.0, start=1.0, end=3.0, step=0.1)
        )
        # First 10/30 frames carry slot 4 → ratio 33%, just over.
        assert _slot_visible_in_interval(4, 0.0, 3.0, dense_30)
        # Slot 99 isn't in any frame.
        assert not _slot_visible_in_interval(99, 0.0, 3.0, dense_30)
        # No frames in the interval → False.
        assert not _slot_visible_in_interval(4, 5.0, 6.0, dense_30)


class TestPerFrameSlotVisibleRate:
    """Fix 2: stricter per-segment visibility gate. The original
    per_frame_in_crop_pass_rate accepts any tracked slot whose face
    center sits inside the candidate crop, so a slot-3 face at x=68%
    can spuriously satisfy a slot-2 9:16 crop. The new helper checks
    the specific slot and rejects boundary leaks."""

    def test_slot_strict_rejects_boundary_neighbor(self):
        # Build a SubjectConfidenceEstimator with two slots: slot 2 at
        # the center of the frame, slot 3 at the right edge. The dense
        # face track only contains slot 3. A slot-2 visibility query
        # must return 0.0 even though slot 3 is in frame.
        from backend.services.subject_confidence import (
            SubjectConfidenceEstimator,
        )
        registry = _FaceRegistry(slots=[
            _FaceSlot(slot_id=2, x_center=50.0),
            _FaceSlot(slot_id=3, x_center=68.0),
        ])
        dense = _make_dense_frames(slot_id=3, x=68.0, start=0.0, end=2.0, step=0.1)
        est = SubjectConfidenceEstimator(
            face_registry=registry,
            dense_faces=dense,
            active_speaker_events=[],
            transcript_segments=[],
            speaker_to_slot={},
            source_width=1920, source_height=1080,
        )
        slot2_rate = est.per_frame_slot_visible_rate(0.0, 2.0, 2)
        slot3_rate = est.per_frame_slot_visible_rate(0.0, 2.0, 3)
        assert slot2_rate == 0.0, (
            "slot-strict gate must NOT count adjacent slot 3 toward slot 2"
        )
        assert slot3_rate == 1.0
