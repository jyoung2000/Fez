"""Content-aware tests for the ReframeSegmenter.

Tests that content-type-specific editorial conventions are applied correctly.
Keeps the original test_reframe_segmenter.py intact (those test the base behavior).
"""
import random
from dataclasses import dataclass, field
from typing import Optional

import pytest

from backend.services.reframe_segmenter import (
    ReframeSegment,
    build_reframe_segments,
    USE_CONTENT_AWARE_REFRAME,
)
from backend.services.content_classifier import ContentProfile


# ── Lightweight stubs (same as test_reframe_segmenter.py) ──

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
    def multi_speaker(self):
        return len(self.slots) >= 2

    def slot_by_id(self, slot_id):
        for s in self.slots:
            if s.slot_id == slot_id:
                return s
        return None

    def nearest_slot(self, x):
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
    x_center: float = 50.0


@dataclass
class _FrameFaces:
    timestamp: float
    frame_path: str = ""
    faces: list[_FaceInfo] = field(default_factory=list)


@dataclass
class _TranscriptSeg:
    start: float
    end: float
    text: str = ""
    speaker: str = "Speaker 1"
    confidence: Optional[float] = 0.9


def _make_dense_frames(slot_id, x, start, end, step=0.5):
    frames = []
    t = start
    while t < end:
        frames.append(_FrameFaces(timestamp=t, faces=[_FaceInfo(identity_id=slot_id, nose_x=x, x_center=x)]))
        t += step
    return frames


def _make_registry_2(x0=30.0, x1=70.0):
    return _FaceRegistry(slots=[
        _FaceSlot(slot_id=0, x_center=x0),
        _FaceSlot(slot_id=1, x_center=x1),
    ])


# ── Narrative tests ──

class TestNarrativeActionSequenceWidens:
    """For narrative content, clusters of rapid shot cuts → WIDE_MASTER."""

    def test_rapid_cuts_force_wide_master(self):
        """8 shot cuts in 6 seconds → segments should be WIDE_MASTER."""
        import os
        os.environ["USE_CONTENT_AWARE_REFRAME"] = "true"
        # Reload to pick up env change
        import importlib
        import backend.services.reframe_segmenter as rsm
        importlib.reload(rsm)

        profile = ContentProfile(content_type="narrative", confidence=0.9)
        registry = _make_registry_2()
        # 8 cuts in a 10s window centered at 5s
        shot_cuts = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
        dense = _make_dense_frames(0, 30, 0, 10, step=0.5)
        transcript = [_TranscriptSeg(start=0, end=10, speaker="Speaker 1")]

        segments = rsm.build_reframe_segments(
            shot_cuts=shot_cuts,
            face_registry=registry,
            active_speaker_events=[],
            dense_faces=dense,
            transcript_segments=transcript,
            speaker_to_slot={"Speaker 1": 0},
            video_duration=10.0,
            content_profile=profile,
        )

        # At least some segments in the high-cut-rate area should be WIDE_MASTER
        wide_segs = [s for s in segments if s.layout == "wide_master"]
        # Restore env
        os.environ["USE_CONTENT_AWARE_REFRAME"] = "false"
        importlib.reload(rsm)

        assert len(wide_segs) >= 1, f"Expected WIDE_MASTER segments for action sequence, got layouts: {[s.layout for s in segments]}"


class TestNarrativeShotCutSnap:
    """Shot cut within 100ms of segment transition → ease_in_ms=0."""

    def test_shot_cut_snaps(self):
        registry = _make_registry_2()
        transcript = [
            _TranscriptSeg(start=0, end=5, speaker="Speaker 1"),
            _TranscriptSeg(start=5.05, end=15, speaker="Speaker 2"),
        ]
        as_events = [
            _SpeakerEvent(0, 5, 0, 0.9),
            _SpeakerEvent(5.05, 15, 1, 0.9),
        ]
        dense = _make_dense_frames(0, 30, 0, 5) + _make_dense_frames(1, 70, 5, 15)
        profile = ContentProfile(content_type="narrative", confidence=0.9)

        segments = build_reframe_segments(
            shot_cuts=[5.0],
            face_registry=registry,
            active_speaker_events=as_events,
            dense_faces=dense,
            transcript_segments=transcript,
            speaker_to_slot={"Speaker 1": 0, "Speaker 2": 1},
            video_duration=15.0,
            content_profile=profile,
        )

        cut_seg = next((s for s in segments if abs(s.start - 5.0) < 0.3), None)
        assert cut_seg is not None
        assert cut_seg.ease_in_ms == 0


# ── Podcast tests ──

class TestPodcastMinHold:
    """Podcast content should have 1.8s minimum hold."""

    def test_podcast_min_hold(self):
        import os
        os.environ["USE_CONTENT_AWARE_REFRAME"] = "true"
        import importlib
        import backend.services.reframe_segmenter as rsm
        importlib.reload(rsm)

        profile = ContentProfile(content_type="podcast", confidence=0.9)
        registry = _make_registry_2()
        transcript = []
        as_events = []
        # Rapid 1s alternating speakers
        for i in range(20):
            t = i * 1.0
            spk = "Speaker 1" if i % 2 == 0 else "Speaker 2"
            slot = 0 if i % 2 == 0 else 1
            transcript.append(_TranscriptSeg(start=t, end=t + 1.0, speaker=spk))
            as_events.append(_SpeakerEvent(t, t + 1.0, slot, 0.9))
        dense = _make_dense_frames(0, 30, 0, 20)

        segments = rsm.build_reframe_segments(
            shot_cuts=[],
            face_registry=registry,
            active_speaker_events=as_events,
            dense_faces=dense,
            transcript_segments=transcript,
            speaker_to_slot={"Speaker 1": 0, "Speaker 2": 1},
            video_duration=20.0,
            content_profile=profile,
        )

        os.environ["USE_CONTENT_AWARE_REFRAME"] = "false"
        importlib.reload(rsm)

        # All segments should be >= 1.8s (podcast min hold)
        for seg in segments:
            dur = seg.end - seg.start
            assert dur >= 1.7, f"Podcast segment {seg.start:.1f}-{seg.end:.1f} ({dur:.1f}s) < 1.8s min hold"


class TestPodcastAnticipation:
    """Podcast speaker turns should be anticipated by 250ms."""

    def test_podcast_anticipation(self):
        import os
        os.environ["USE_CONTENT_AWARE_REFRAME"] = "true"
        import importlib
        import backend.services.reframe_segmenter as rsm
        importlib.reload(rsm)

        profile = ContentProfile(content_type="podcast", confidence=0.9)
        registry = _make_registry_2()
        transcript = [
            _TranscriptSeg(start=0, end=10, speaker="Speaker 1"),
            _TranscriptSeg(start=10, end=20, speaker="Speaker 2"),
        ]
        as_events = [
            _SpeakerEvent(0, 10, 0, 0.9),
            _SpeakerEvent(10, 20, 1, 0.9),
        ]
        dense = _make_dense_frames(0, 30, 0, 10) + _make_dense_frames(1, 70, 10, 20)

        segments = rsm.build_reframe_segments(
            shot_cuts=[],
            face_registry=registry,
            active_speaker_events=as_events,
            dense_faces=dense,
            transcript_segments=transcript,
            speaker_to_slot={"Speaker 1": 0, "Speaker 2": 1},
            video_duration=20.0,
            content_profile=profile,
        )

        os.environ["USE_CONTENT_AWARE_REFRAME"] = "false"
        importlib.reload(rsm)

        turn_seg = next((s for s in segments if s.active_slot == 1), None)
        assert turn_seg is not None
        # Podcast anticipation is 250ms
        expected_start = 10.0 - 0.25
        assert abs(turn_seg.start - expected_start) < 0.05, \
            f"Expected start ~{expected_start}, got {turn_seg.start}"


# ── Gaming tests ──

class TestGamingStackedLayout:
    """Gaming content with facecam → STACKED_GAMEPLAY strategy."""

    def test_stacked_with_facecam(self):
        import os
        os.environ["USE_CONTENT_AWARE_REFRAME"] = "true"
        import importlib
        import backend.services.reframe_segmenter as rsm
        importlib.reload(rsm)

        from backend.services.persistent_region_detector import RegionDetectionResult, PersistentRegion
        profile = ContentProfile(content_type="gaming", confidence=0.9, has_facecam=True)
        regions = RegionDetectionResult(
            has_facecam=True,
            facecam_region=PersistentRegion(
                x=0.8, y=0.7, w=0.15, h=0.2,
                region_type="facecam", confidence=0.9, corner="bottom_right",
            ),
            regions=[PersistentRegion(
                x=0.8, y=0.7, w=0.15, h=0.2,
                region_type="facecam", confidence=0.9, corner="bottom_right",
            )],
        )
        registry = _FaceRegistry(slots=[_FaceSlot(0, 85, 83, 87, 180, 8.0)])
        dense = _make_dense_frames(0, 85, 0, 10)
        transcript = [_TranscriptSeg(start=0, end=10, speaker="Speaker 1")]

        segments = rsm.build_reframe_segments(
            shot_cuts=[],
            face_registry=registry,
            active_speaker_events=[],
            dense_faces=dense,
            transcript_segments=transcript,
            speaker_to_slot={"Speaker 1": 0},
            video_duration=10.0,
            content_profile=profile,
            persistent_regions=regions,
        )

        os.environ["USE_CONTENT_AWARE_REFRAME"] = "false"
        importlib.reload(rsm)

        stacked = [s for s in segments if s.strategy == "stacked_gameplay"]
        assert len(stacked) > 0, f"Expected STACKED_GAMEPLAY segments, got strategies: {[s.strategy for s in segments]}"


# ── Content type field test ──

class TestContentTypeFieldSet:
    """Every segment should have the content_type field set."""

    def test_content_type_propagated(self):
        import os
        os.environ["USE_CONTENT_AWARE_REFRAME"] = "true"
        import importlib
        import backend.services.reframe_segmenter as rsm
        importlib.reload(rsm)

        profile = ContentProfile(content_type="vlog", confidence=0.8)
        registry = _FaceRegistry(slots=[_FaceSlot(0, 50)])
        dense = _make_dense_frames(0, 50, 0, 10)
        transcript = [_TranscriptSeg(start=0, end=10, speaker="Speaker 1")]

        segments = rsm.build_reframe_segments(
            shot_cuts=[],
            face_registry=registry,
            active_speaker_events=[],
            dense_faces=dense,
            transcript_segments=transcript,
            speaker_to_slot={"Speaker 1": 0},
            video_duration=10.0,
            content_profile=profile,
        )

        os.environ["USE_CONTENT_AWARE_REFRAME"] = "false"
        importlib.reload(rsm)

        for seg in segments:
            assert seg.content_type == "vlog", f"Expected content_type='vlog', got '{seg.content_type}'"
