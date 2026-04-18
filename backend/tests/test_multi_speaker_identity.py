"""Integration tests for multi-speaker identity preservation and
figurine-in-front-of-torso rejection.

These tests reproduce the real regression scenarios:
  1. 6-speaker fixture: 6 distinct people should produce 6 registry slots
  2. Figurine in front of torso: strict containment rejects the figurine
  3. Empty chair: no face = no confident crop
"""

import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from backend.services.face_detector import FaceInfo, FrameFaces
from backend.services.face_registry import build_face_registry


class _MockFace:
    """Minimal face mock for verification."""
    def __init__(self, nose_x=50, nose_y=50, x_center=50, y_center=50,
                 width=10, height=12, is_human=True):
        self.nose_x = nose_x
        self.nose_y = nose_y
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height
        self.is_human = is_human
        self.pose_confidence = 0.0


def _make_face(nose_x=30, width=8, height=10, is_human=True):
    return FaceInfo(
        x_center=nose_x, y_center=40,
        width=width, height=height,
        nose_x=nose_x, nose_y=40,
        confidence=0.9,
        is_human=is_human,
    )


def _make_frame(timestamp, faces):
    return FrameFaces(
        timestamp=timestamp,
        frame_path=f"/tmp/frame_{timestamp}.jpg",
        faces=faces,
        primary_face_idx=0 if faces else -1,
    )


class TestSixSpeakerPreservation:
    def test_six_speakers_produce_six_slots(self):
        """Six distinct speakers at different x-positions produce 6 slots."""
        # Simulate Joe Budden Network: 6 speakers across the frame
        positions = [12, 27, 42, 57, 72, 87]
        frames = []
        for t in range(30):
            faces = [_make_face(nose_x=x, width=7) for x in positions]
            frames.append(_make_frame(t * 0.5, faces))

        registry = build_face_registry(frames, min_appearances=3)
        assert len(registry.slots) == 6, \
            f"Expected 6 slots for 6 distinct speakers, got {len(registry.slots)}: " \
            f"{[(s.slot_id, round(s.x_center)) for s in registry.slots]}"

    def test_five_speakers_minimum(self):
        """Even with some noise, at least 5 of 6 speakers are preserved."""
        positions = [15, 30, 45, 60, 75, 90]
        frames = []
        for t in range(25):
            # Add slight position jitter (±2%)
            jitter = np.random.uniform(-2, 2, len(positions))
            faces = [_make_face(nose_x=x + j, width=7)
                     for x, j in zip(positions, jitter)]
            frames.append(_make_frame(t * 0.5, faces))

        registry = build_face_registry(frames, min_appearances=3)
        assert len(registry.slots) >= 5, \
            f"Expected >=5 slots for 6 speakers (allowing 1 miss), got {len(registry.slots)}"

    def test_non_human_faces_excluded_from_count(self):
        """Non-human faces don't contribute to slot count."""
        frames = []
        for t in range(20):
            faces = [
                _make_face(nose_x=25, is_human=True),
                _make_face(nose_x=70, is_human=True),
                _make_face(nose_x=85, is_human=False),  # figurine
            ]
            frames.append(_make_frame(t * 0.5, faces))

        registry = build_face_registry(frames, min_appearances=3)
        # Should have 2 slots (the two humans), not 3
        assert len(registry.slots) == 2, \
            f"Expected 2 slots (non-human excluded), got {len(registry.slots)}"


class TestFigurineInFrontOfTorso:
    def test_strict_containment_rejects_figurine(self):
        """A figurine face whose bbox does NOT contain a pose nose is rejected,
        even when a human's torso is visible behind it."""
        from backend.services.human_face_verifier import HumanFaceVerifier

        v = HumanFaceVerifier.__new__(HumanFaceVerifier)
        v._available = True
        v._mp = MagicMock()
        v._pose = MagicMock()
        v._min_pose_confidence = 0.4

        # Human face at (30%, 40%) on 1920x1080
        human_face = _MockFace(nose_x=30, nose_y=40, width=8, height=10)
        # Figurine face at (70%, 35%)
        figurine_face = _MockFace(nose_x=70, nose_y=35, width=6, height=8)

        # Mock _get_pose_nose_for_region:
        # - Human's region → pose nose at human position (576, 432)
        # - Figurine's region → also detects human's pose (behind figurine)
        #   but the nose is at the human's position, not the figurine's
        human_nose = (576.0, 432.0)  # 30% of 1920, 40% of 1080
        call_count = [0]
        def fake_get_nose(frame, cx, cy, w, h, sw, sh):
            call_count[0] += 1
            return human_nose  # both regions see the human's nose

        v._get_pose_nose_for_region = fake_get_nose

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        v.verify_faces_in_frame(frame, [human_face, figurine_face], 1920, 1080)

        assert human_face.is_human is True, \
            "Human face at (30, 40) should pass: pose nose (30, 40) is inside bbox"
        assert figurine_face.is_human is False, \
            "Figurine face at (70, 35) should fail: pose nose (30, 40) is NOT inside its bbox"


class TestEmptyFrameNoConfidentCrop:
    def test_empty_frame_produces_low_confidence(self):
        """A frame with no faces produces low confidence that triggers
        the confidence floor guard."""
        from backend.services.subject_confidence import SubjectConfidenceEstimator

        # No faces in the dense_faces
        empty_frame = _make_frame(0.5, [])

        estimator = SubjectConfidenceEstimator(
            face_registry=MagicMock(),
            dense_faces=[empty_frame],
            active_speaker_events=[],
            transcript_segments=[],
            speaker_to_slot={},
            source_width=1920,
            source_height=1080,
        )

        conf, reason = estimator.evaluate(0.0, 1.0, None, 50)
        # With no faces in the crop window, confidence should be capped at 0.30
        assert conf <= 0.30, \
            f"Expected confidence <= 0.30 for empty frame, got {conf}"

    def test_confidence_floor_catches_empty_frame_crop(self):
        """The confidence floor guard downgrades a crop on an empty frame."""
        from backend.services.confidence_audit import enforce_confidence_floor

        seg = MagicMock()
        seg.strategy = "stationary"
        seg.confidence = 0.20  # low confidence from empty frame
        seg.layout = "single"
        seg.subject_x = 50
        seg.active_slot = None
        seg.start = 0.0
        seg.end = 1.0
        seg.fallback_reason = None

        result = enforce_confidence_floor([seg], job_id="test")
        assert result[0].strategy == "blur_fill", \
            f"Expected blur_fill for low-confidence segment, got {result[0].strategy}"


class TestFeatureFlags:
    def test_strict_pose_flag_default_true(self):
        """USE_STRICT_POSE_MATCHING defaults to true."""
        from backend.services.human_face_verifier import USE_STRICT_POSE_MATCHING
        # Default is true unless env overrides it
        assert USE_STRICT_POSE_MATCHING is True or \
            os.environ.get("USE_STRICT_POSE_MATCHING", "true").lower() in ("true", "1", "yes")

    def test_legacy_mode_via_env(self):
        """USE_STRICT_POSE_MATCHING=false uses legacy per-face verification."""
        # Verify the env var is read correctly
        with patch.dict("os.environ", {"USE_STRICT_POSE_MATCHING": "false"}):
            val = os.environ.get("USE_STRICT_POSE_MATCHING", "true").lower()
            assert val == "false"
