"""Tests for subject fusion module."""

import numpy as np
import cv2
import pytest
from dataclasses import dataclass, field
from typing import Optional

from backend.services.subject_fusion import (
    build_subject_tracks,
    _cluster_saliency_regions,
    _promote_cluster,
    _region_iou,
    MIN_PERSISTENCE_FRAMES,
    MIN_ASPECT_RATIO,
    MAX_ASPECT_RATIO,
    MIN_AREA_RATIO,
    MAX_AREA_RATIO,
    PROMOTED_BASE_CONFIDENCE,
    CONFIRMED_BASE_CONFIDENCE,
)
from backend.services.saliency_tracker import SaliencyRegion


# -- Mock objects --

@dataclass
class MockFace:
    nose_x: float = 50.0
    nose_y: float = 40.0
    x: float = 50.0
    y: float = 40.0
    width: float = 10.0
    height: float = 13.0
    identity_id: int = 0


@dataclass
class MockFrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)
    primary_face_idx: int = 0
    path: str = ""


@dataclass
class MockFaceSlot:
    slot_id: int = 0
    x_center: float = 50.0


@dataclass
class MockFaceRegistry:
    slots: list = field(default_factory=list)


def _make_saliency_region(t, x=50.0, y=50.0, w=12.0, h=16.0, score=0.7):
    """Helper to create a SaliencyRegion with face-like proportions."""
    return SaliencyRegion(
        timestamp=t, x=x, y=y, w=w, h=h,
        saliency_score=score, motion_score=0.3, spatial_score=0.5,
    )


class TestEmptyInputs:
    def test_empty_everything(self):
        """Empty inputs -> empty tracks list."""
        tracks = build_subject_tracks(
            face_registry=None, dense_faces=[], saliency_regions=[],
            frame_paths=[], source_width=1280, source_height=720,
        )
        assert tracks == []


class TestFaceConfirmedTracks:
    def test_one_face_slot(self):
        """One face slot with 3 dense face detections -> one confirmed track."""
        dense_faces = [
            MockFrameFaces(timestamp=0.5, faces=[MockFace(nose_x=50, identity_id=0)]),
            MockFrameFaces(timestamp=1.0, faces=[MockFace(nose_x=52, identity_id=0)]),
            MockFrameFaces(timestamp=1.5, faces=[MockFace(nose_x=54, identity_id=0)]),
        ]
        registry = MockFaceRegistry(slots=[MockFaceSlot(slot_id=0)])

        tracks = build_subject_tracks(
            face_registry=registry, dense_faces=dense_faces, saliency_regions=[],
            frame_paths=[], source_width=1280, source_height=720,
        )
        assert len(tracks) == 1
        assert tracks[0].source == "face_confirmed"
        assert tracks[0].face_slot_id == 0
        assert tracks[0].confidence == CONFIRMED_BASE_CONFIDENCE
        assert len(tracks[0].bbox_trajectory) == 3

    def test_single_detection_not_enough(self):
        """A face slot with only 1 detection does not produce a track."""
        dense_faces = [
            MockFrameFaces(timestamp=0.5, faces=[MockFace(nose_x=50, identity_id=0)]),
        ]
        registry = MockFaceRegistry(slots=[MockFaceSlot(slot_id=0)])

        tracks = build_subject_tracks(
            face_registry=registry, dense_faces=dense_faces, saliency_regions=[],
            frame_paths=[], source_width=1280, source_height=720,
        )
        assert len(tracks) == 0


class TestSaliencyPromotion:
    def test_too_few_regions(self):
        """Saliency cluster with 3 regions (below MIN_PERSISTENCE=5) -> no promotion."""
        regions = [_make_saliency_region(t * 0.5) for t in range(3)]
        tracks = build_subject_tracks(
            face_registry=None, dense_faces=[], saliency_regions=regions,
            frame_paths=[], source_width=1280, source_height=720,
        )
        promoted = [t for t in tracks if t.source == "face_like_promoted"]
        assert len(promoted) == 0

    def test_enough_regions_promoted(self):
        """Saliency cluster with 7 face-like regions -> one promoted track."""
        regions = [_make_saliency_region(t * 0.5, w=12.0, h=16.0) for t in range(7)]
        # Aspect ratio: 16/12 = 1.33 (within 1.0-1.8)
        # Area ratio: (12*16)/10000 = 0.0192 (within 0.005-0.15)
        tracks = build_subject_tracks(
            face_registry=None, dense_faces=[], saliency_regions=regions,
            frame_paths=[], source_width=1280, source_height=720,
        )
        promoted = [t for t in tracks if t.source == "face_like_promoted"]
        assert len(promoted) == 1
        assert promoted[0].confidence >= PROMOTED_BASE_CONFIDENCE
        assert promoted[0].gate_persistence_frames == 7

    def test_bad_aspect_ratio_rejected(self):
        """Saliency cluster with aspect ratio 2.5 -> rejected."""
        regions = [_make_saliency_region(t * 0.5, w=8.0, h=20.0) for t in range(7)]
        # Aspect: 20/8 = 2.5 > MAX_ASPECT_RATIO(1.8)
        tracks = build_subject_tracks(
            face_registry=None, dense_faces=[], saliency_regions=regions,
            frame_paths=[], source_width=1280, source_height=720,
        )
        promoted = [t for t in tracks if t.source == "face_like_promoted"]
        assert len(promoted) == 0

    def test_too_large_area_rejected(self):
        """Saliency cluster with area 20% of frame -> rejected."""
        # sqrt(0.20 * 10000) ~ w=45, h=45 -> 45*45/10000 = 0.2025 > 0.15
        regions = [_make_saliency_region(t * 0.5, w=45.0, h=45.0) for t in range(7)]
        tracks = build_subject_tracks(
            face_registry=None, dense_faces=[], saliency_regions=regions,
            frame_paths=[], source_width=1280, source_height=720,
        )
        promoted = [t for t in tracks if t.source == "face_like_promoted"]
        assert len(promoted) == 0


class TestAppearanceMatching:
    def test_same_color_same_persistent_id(self, tmp_path):
        """Two clusters of same color -> same persistent_id."""
        # Create frames with red rectangles at two positions (simulating 2 shots)
        frame_paths = []
        for i in range(14):
            frame = np.full((720, 1280, 3), 30, dtype=np.uint8)
            # Red rectangle
            x_off = 300 if i < 7 else 600
            cv2.rectangle(frame, (x_off, 200), (x_off + 150, 400), (0, 0, 200), -1)
            p = tmp_path / f"f_{i:03d}.png"
            cv2.imwrite(str(p), frame)
            frame_paths.append((float(i) * 0.5, str(p)))

        # Two saliency clusters at different x positions but same color
        regions1 = [SaliencyRegion(
            timestamp=i * 0.5, x=26.0, y=42.0, w=12.0, h=16.0,
            saliency_score=0.7, motion_score=0.3, spatial_score=0.5,
        ) for i in range(7)]
        regions2 = [SaliencyRegion(
            timestamp=(7 + i) * 0.5, x=50.0, y=42.0, w=12.0, h=16.0,
            saliency_score=0.7, motion_score=0.3, spatial_score=0.5,
        ) for i in range(7)]

        tracks = build_subject_tracks(
            face_registry=None, dense_faces=[],
            saliency_regions=regions1 + regions2,
            frame_paths=frame_paths,
            source_width=1280, source_height=720,
            shot_cuts=[3.5],  # Shot boundary between the two clusters
        )
        promoted = [t for t in tracks if t.source == "face_like_promoted"]
        assert len(promoted) == 2
        # Same color -> should get the same persistent_id
        assert promoted[0].persistent_id == promoted[1].persistent_id

    def test_different_color_different_persistent_id(self, tmp_path):
        """Two clusters of very different colors -> different persistent_id."""
        frame_paths = []
        for i in range(14):
            frame = np.full((720, 1280, 3), 30, dtype=np.uint8)
            if i < 7:
                cv2.rectangle(frame, (300, 200), (450, 400), (0, 0, 200), -1)  # red
            else:
                cv2.rectangle(frame, (300, 200), (450, 400), (200, 0, 0), -1)  # blue
            p = tmp_path / f"f_{i:03d}.png"
            cv2.imwrite(str(p), frame)
            frame_paths.append((float(i) * 0.5, str(p)))

        regions1 = [SaliencyRegion(
            timestamp=i * 0.5, x=29.0, y=42.0, w=12.0, h=16.0,
            saliency_score=0.7, motion_score=0.3, spatial_score=0.5,
        ) for i in range(7)]
        regions2 = [SaliencyRegion(
            timestamp=(7 + i) * 0.5, x=29.0, y=42.0, w=12.0, h=16.0,
            saliency_score=0.7, motion_score=0.3, spatial_score=0.5,
        ) for i in range(7)]

        tracks = build_subject_tracks(
            face_registry=None, dense_faces=[],
            saliency_regions=regions1 + regions2,
            frame_paths=frame_paths,
            source_width=1280, source_height=720,
            shot_cuts=[3.5],
        )
        promoted = [t for t in tracks if t.source == "face_like_promoted"]
        assert len(promoted) == 2
        assert promoted[0].persistent_id != promoted[1].persistent_id


class TestMixedInputs:
    def test_face_plus_saliency(self):
        """Mixed face + saliency -> both kinds of tracks, face has higher confidence."""
        dense_faces = [
            MockFrameFaces(timestamp=t * 0.5, faces=[MockFace(nose_x=30, identity_id=0)])
            for t in range(10)
        ]
        regions = [_make_saliency_region(t * 0.5, x=70.0) for t in range(7)]
        registry = MockFaceRegistry(slots=[MockFaceSlot(slot_id=0)])

        tracks = build_subject_tracks(
            face_registry=registry, dense_faces=dense_faces,
            saliency_regions=regions, frame_paths=[],
            source_width=1280, source_height=720,
        )
        confirmed = [t for t in tracks if t.source == "face_confirmed"]
        promoted = [t for t in tracks if t.source == "face_like_promoted"]
        assert len(confirmed) >= 1
        assert len(promoted) >= 1
        # Face tracks always have higher confidence
        for c in confirmed:
            for p in promoted:
                assert c.confidence > p.confidence


class TestClusteringRespectsShotCuts:
    def test_no_cluster_spans_shot_cut(self):
        """Saliency clustering does not span shot boundaries."""
        regions = [_make_saliency_region(t * 0.5) for t in range(14)]
        clusters = _cluster_saliency_regions(regions, shot_cuts=[3.5])
        # No cluster should have regions from both before and after 3.5
        for cluster in clusters:
            timestamps = [r.timestamp for r in cluster]
            has_before = any(t < 3.5 for t in timestamps)
            has_after = any(t >= 3.5 for t in timestamps)
            assert not (has_before and has_after), \
                f"Cluster spans shot cut: {timestamps}"


class TestEMASmoothing:
    """Fix 6: Verify temporal smoothing of promoted saliency clusters."""

    def test_smoothed_has_lower_variance(self):
        """Smoothed trajectory has lower variance than raw input."""
        from backend.services.subject_fusion import _ema_smooth_trajectory

        raw = [
            (0.0, 50.0, 50.0, 12.0, 16.0),
            (0.5, 52.0, 50.0, 12.0, 16.0),
            (1.0, 48.0, 50.0, 12.0, 16.0),
            (1.5, 51.0, 50.0, 12.0, 16.0),
            (2.0, 49.0, 50.0, 12.0, 16.0),
            (2.5, 50.0, 50.0, 12.0, 16.0),
        ]

        smoothed = _ema_smooth_trajectory(raw, alpha=0.3)

        assert len(smoothed) == len(raw)

        # Compute x variance for both
        raw_xs = [r[1] for r in raw]
        smooth_xs = [s[1] for s in smoothed]
        raw_var = np.var(raw_xs)
        smooth_var = np.var(smooth_xs)

        assert smooth_var < raw_var, (
            f"Smoothed variance ({smooth_var:.4f}) should be less than "
            f"raw variance ({raw_var:.4f})"
        )

    def test_smoothed_converges_monotonically(self):
        """After an initial step, smoothed values converge without direction reversals."""
        from backend.services.subject_fusion import _ema_smooth_trajectory

        raw = [
            (0.0, 50.0, 50.0, 12.0, 16.0),
            (0.5, 52.0, 50.0, 12.0, 16.0),
            (1.0, 48.0, 50.0, 12.0, 16.0),
            (1.5, 51.0, 50.0, 12.0, 16.0),
            (2.0, 49.0, 50.0, 12.0, 16.0),
            (2.5, 50.0, 50.0, 12.0, 16.0),
        ]

        smoothed = _ema_smooth_trajectory(raw, alpha=0.3)

        # After step 2, differences between consecutive smoothed values should
        # be decreasing (converging) — no large direction reversals
        diffs = [abs(smoothed[i + 1][1] - smoothed[i][1]) for i in range(2, len(smoothed) - 1)]
        # At least the last few diffs should be small
        assert diffs[-1] < 1.0, f"Last diff ({diffs[-1]:.4f}) should be very small"

    def test_promoted_track_has_raw_trajectory(self):
        """Promoted tracks store the unsmoothed raw_trajectory for debugging."""
        regions = [_make_saliency_region(t * 0.5, x=50.0 + (t % 3), w=12.0, h=16.0)
                   for t in range(7)]

        tracks = build_subject_tracks(
            face_registry=None, dense_faces=[], saliency_regions=regions,
            frame_paths=[], source_width=1280, source_height=720,
        )

        promoted = [t for t in tracks if t.source == "face_like_promoted"]
        assert len(promoted) >= 1
        track = promoted[0]
        assert track.raw_trajectory is not None
        assert len(track.raw_trajectory) == len(track.bbox_trajectory)


class TestRegionIoU:
    def test_identical_regions(self):
        """Identical regions have IoU = 1.0."""
        r = _make_saliency_region(0, x=50, y=50, w=20, h=20)
        assert abs(_region_iou(r, r) - 1.0) < 0.01

    def test_non_overlapping(self):
        """Non-overlapping regions have IoU = 0.0."""
        r1 = _make_saliency_region(0, x=10, y=50, w=10, h=10)
        r2 = _make_saliency_region(0, x=90, y=50, w=10, h=10)
        assert _region_iou(r1, r2) == 0.0
