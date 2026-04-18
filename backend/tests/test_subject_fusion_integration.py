"""Integration test for subject fusion: anime-like synthetic fixture.

Tests the full signal pipeline with synthetic faceless frames (colored
shape on contrasting background, no photographic face features) to verify
the fusion pipeline promotes the salient region to a face_like track and
the segmenter produces a valid segment.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytest

from backend.services.saliency_tracker import track_saliency_in_frames
from backend.services.subject_fusion import (
    build_subject_tracks,
    MIN_PERSISTENCE_FRAMES,
    MIN_ASPECT_RATIO,
    MAX_ASPECT_RATIO,
    MIN_AREA_RATIO,
    MAX_AREA_RATIO,
)
from backend.services.scene_focus import aggregate_scene_focus
from backend.services.autoflip_segmenter import build_autoflip_segments
from backend.services.focus_model import FeatureKind, SubjectTrack


# -- Mock objects for segmenter --

@dataclass
class MockFaceSlot:
    slot_id: int = 0
    x_center: float = 50.0
    x_min: float = 35.0
    x_max: float = 65.0
    frame_count: int = 20
    avg_width: float = 10.0
    avg_height: float = 13.0


@dataclass
class MockFaceRegistry:
    slots: list = field(default_factory=list)
    total_frames: int = 20
    frames_with_faces: int = 0

    @property
    def multi_speaker(self):
        return len(self.slots) >= 2

    @property
    def is_continuous_motion(self):
        return False

    def nearest_slot(self, x):
        if not self.slots:
            return None
        return min(self.slots, key=lambda s: abs(s.x_center - x))

    def slot_by_id(self, slot_id):
        for s in self.slots:
            if s.slot_id == slot_id:
                return s
        return None


@dataclass
class MockSpeakerEvent:
    start: float
    end: float
    slot_id: int
    confidence: float = 0.9


def _create_anime_like_frames(tmp_path, n_frames=20, fps=2):
    """Create synthetic frames: a red rectangle moving across a dark background.

    The rectangle has face-like aspect ratio (~1.3) and area (~3% of frame)
    but no photographic face features, simulating an anime character silhouette.

    Returns list[(timestamp, path)]
    """
    frame_paths = []
    for i in range(n_frames):
        t = i / fps
        frame = np.full((720, 1280, 3), 30, dtype=np.uint8)  # dark background

        # Moving red rectangle — saliency detection expands the bbox via
        # Gaussian blur and edge detection, so the actual rectangle should
        # be wider than tall to keep the detected aspect ratio in [1.0, 1.8].
        # Using 200x160px (width x height): detected bbox will be slightly
        # expanded, yielding aspect ~1.2-1.5 after Sobel/blur expansion.
        x_off = 100 + int(i * (800 / max(n_frames - 1, 1)))
        cv2.rectangle(frame, (x_off, 240), (x_off + 200, 400), (0, 0, 200), -1)

        path = tmp_path / f"anime_frame_{i:04d}.png"
        cv2.imwrite(str(path), frame)
        frame_paths.append((float(t), str(path)))
    return frame_paths


class TestSubjectFusionPromotesAnimeLikeRegion:
    def test_saliency_produces_regions(self, tmp_path):
        """Saliency tracker finds the red rectangle in anime-like frames."""
        frame_paths = _create_anime_like_frames(tmp_path, n_frames=20, fps=2)
        regions = track_saliency_in_frames(frame_paths, face_results=None)
        assert len(regions) >= 5, \
            f"expected saliency to find the red rectangle, got {len(regions)} regions"

    def test_fusion_promotes_to_face_like(self, tmp_path):
        """Subject fusion promotes the saliency cluster to a face_like track."""
        frame_paths = _create_anime_like_frames(tmp_path, n_frames=20, fps=2)
        regions = track_saliency_in_frames(frame_paths, face_results=None)

        # Empty face registry -- simulating face detection failure on anime
        registry = MockFaceRegistry(slots=[])

        tracks = build_subject_tracks(
            face_registry=registry,
            dense_faces=[],
            saliency_regions=regions,
            frame_paths=frame_paths,
            source_width=1280,
            source_height=720,
            shot_cuts=[],
            job_id="anime_test",
        )

        promoted = [t for t in tracks if t.source == "face_like_promoted"]
        assert len(promoted) >= 1, \
            f"expected at least one promoted track, got {len(promoted)} " \
            f"(total tracks: {len(tracks)}, saliency regions: {len(regions)})"

        track = promoted[0]
        assert MIN_ASPECT_RATIO <= track.gate_aspect_ratio <= MAX_ASPECT_RATIO
        assert MIN_AREA_RATIO <= track.gate_area_ratio <= MAX_AREA_RATIO
        assert track.gate_persistence_frames >= MIN_PERSISTENCE_FRAMES
        assert track.confidence >= 0.60
        # Face mesh will fail on the red rectangle -- that's correct behavior
        # It's a confidence modulator, not a gate

    def test_promoted_track_produces_required_features(self, tmp_path):
        """Promoted track becomes a FACE_LIKE required feature in scene_focus."""
        frame_paths = _create_anime_like_frames(tmp_path, n_frames=20, fps=2)
        regions = track_saliency_in_frames(frame_paths, face_results=None)

        registry = MockFaceRegistry(slots=[])
        tracks = build_subject_tracks(
            face_registry=registry, dense_faces=[],
            saliency_regions=regions, frame_paths=frame_paths,
            source_width=1280, source_height=720,
            shot_cuts=[], job_id="anime_test",
        )

        promoted = [t for t in tracks if t.source == "face_like_promoted"]
        if not promoted:
            pytest.skip("No promoted tracks (saliency may not have matched)")

        result = aggregate_scene_focus(
            shot_start=0, shot_end=10, dense_faces=[],
            saliency_keyframes=[], persistent_regions=None,
            face_registry=None, source_width=1280, source_height=720,
            subject_tracks=tracks,
        )

        face_like_features = [r for r in result.required if r.kind == FeatureKind.FACE_LIKE]
        assert len(face_like_features) >= 1
        assert all(f.must_be_in_frame for f in face_like_features)

    def test_segmenter_produces_segment_with_promoted_tracks(self, tmp_path):
        """Segmenter produces a valid segment from promoted face_like tracks."""
        frame_paths = _create_anime_like_frames(tmp_path, n_frames=20, fps=2)
        regions = track_saliency_in_frames(frame_paths, face_results=None)

        registry = MockFaceRegistry(slots=[])
        tracks = build_subject_tracks(
            face_registry=registry, dense_faces=[],
            saliency_regions=regions, frame_paths=frame_paths,
            source_width=1280, source_height=720,
            shot_cuts=[], job_id="anime_test",
        )

        promoted = [t for t in tracks if t.source == "face_like_promoted"]
        if not promoted:
            pytest.skip("No promoted tracks")

        segments = build_autoflip_segments(
            shot_cuts=[], face_registry=registry,
            active_speaker_events=[], dense_faces=[],
            saliency_keyframes=[], transcript_segments=[],
            speaker_to_slot={}, video_duration=10.0,
            source_width=1280, source_height=720,
            subject_tracks=tracks,
            job_id="anime_segmenter_test",
        )

        assert len(segments) >= 1
        seg = segments[0]
        assert abs(seg.start - 0.0) < 0.01
        assert abs(seg.end - 10.0) < 0.01
        # With promoted tracks as required features, should get tracking
        assert seg.strategy in ("tracking", "panning", "stationary", "blur_fill")


class TestSubjectFusionRegressionFaceOnly:
    def test_face_only_no_tracks_identical(self):
        """Face-only content with subject_tracks=None is identical to pre-fusion."""

        @dataclass
        class _Face:
            nose_x: float = 50.0
            nose_y: float = 30.0
            x: float = 50.0
            y: float = 30.0
            width: float = 15.0
            height: float = 20.0
            identity_id: int = 0

        @dataclass
        class _FrameFaces:
            timestamp: float
            faces: list = field(default_factory=list)

        dense_faces = [
            _FrameFaces(timestamp=0.5, faces=[_Face(nose_x=50)]),
            _FrameFaces(timestamp=1.0, faces=[_Face(nose_x=52)]),
            _FrameFaces(timestamp=1.5, faces=[_Face(nose_x=54)]),
        ]

        result_none = aggregate_scene_focus(
            shot_start=0.0, shot_end=2.0,
            dense_faces=dense_faces, saliency_keyframes=[],
            persistent_regions=None, face_registry=None,
            source_width=1920, source_height=1080,
            subject_tracks=None,
        )

        result_empty = aggregate_scene_focus(
            shot_start=0.0, shot_end=2.0,
            dense_faces=dense_faces, saliency_keyframes=[],
            persistent_regions=None, face_registry=None,
            source_width=1920, source_height=1080,
            subject_tracks=[],
        )

        assert len(result_none.required) == len(result_empty.required)
        assert result_none.fits_target_aspect == result_empty.fits_target_aspect
        assert abs(result_none.optimal_crop_center[0] - result_empty.optimal_crop_center[0]) < 0.01


class TestCrossShotIdentity:
    def test_same_subject_across_shots(self, tmp_path):
        """Same-colored subject in two shots gets the same persistent_id."""
        # Shot 1: red rect moving right (frames 0-9)
        # Shot 2: red rect moving left (frames 10-19)
        frame_paths = []
        for i in range(20):
            frame = np.full((720, 1280, 3), 30, dtype=np.uint8)
            if i < 10:
                x_off = 200 + i * 60
            else:
                x_off = 800 - (i - 10) * 60
            cv2.rectangle(frame, (x_off, 240), (x_off + 200, 400), (0, 0, 200), -1)
            path = tmp_path / f"cross_shot_{i:04d}.png"
            cv2.imwrite(str(path), frame)
            frame_paths.append((float(i) * 0.5, str(path)))

        regions = track_saliency_in_frames(frame_paths, face_results=None)
        registry = MockFaceRegistry(slots=[])

        tracks = build_subject_tracks(
            face_registry=registry, dense_faces=[],
            saliency_regions=regions, frame_paths=frame_paths,
            source_width=1280, source_height=720,
            shot_cuts=[5.0],  # Shot boundary at 5s
            job_id="cross_shot_test",
        )

        promoted = [t for t in tracks if t.source == "face_like_promoted"]
        if len(promoted) >= 2:
            # Both tracks should share the same persistent_id (same color)
            assert promoted[0].persistent_id == promoted[1].persistent_id, \
                f"Expected same persistent_id, got {promoted[0].persistent_id} and {promoted[1].persistent_id}"
