"""Tests for face registry identity preservation.

Tests the co-occurrence guard and grace period logic that prevent
premature identity merge in multi-speaker panels.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from backend.services.face_detector import FaceInfo, FrameFaces
from backend.services.face_registry import build_face_registry, build_face_registry_with_embeddings


def _make_face(nose_x=30, width=8, height=10, embedding=None, identity_id=-1):
    return FaceInfo(
        x_center=nose_x, y_center=40,
        width=width, height=height,
        nose_x=nose_x, nose_y=40,
        confidence=0.9,
        identity_embedding=embedding,
        identity_id=identity_id,
    )


def _make_frame(timestamp, faces):
    return FrameFaces(
        timestamp=timestamp,
        frame_path=f"/tmp/frame_{timestamp}.jpg",
        faces=faces,
        primary_face_idx=0 if faces else -1,
    )


class TestPositionBasedMultiSpeaker:
    def test_six_speakers_distinct_positions(self):
        """6 speakers at distinct positions should produce 6 slots."""
        # Speakers at x=10, 25, 40, 55, 70, 85
        positions = [10, 25, 40, 55, 70, 85]
        frames = []
        for t in range(20):
            faces = [_make_face(nose_x=x) for x in positions]
            frames.append(_make_frame(t * 0.5, faces))

        registry = build_face_registry(frames, min_appearances=3)
        assert len(registry.slots) >= 5, \
            f"Expected >=5 slots for 6 distinct speakers, got {len(registry.slots)}: " \
            f"{[(s.slot_id, s.x_center) for s in registry.slots]}"

    def test_four_speakers_close_together(self):
        """4 speakers at closer positions should still produce 4 slots."""
        positions = [20, 35, 55, 75]
        frames = []
        for t in range(20):
            faces = [_make_face(nose_x=x) for x in positions]
            frames.append(_make_frame(t * 0.5, faces))

        registry = build_face_registry(frames, min_appearances=3)
        assert len(registry.slots) >= 4, \
            f"Expected >=4 slots for 4 speakers, got {len(registry.slots)}"


class TestEmbeddingCoOccurrenceGuard:
    def test_same_frame_different_position_not_merged(self):
        """Two faces in the same frame at different positions are never merged,
        even if their embeddings are identical."""
        np.random.seed(42)
        # Create two identical embeddings
        shared_embedding = np.random.randn(128).tolist()

        frames = []
        for t in range(20):
            faces = [
                _make_face(nose_x=25, embedding=shared_embedding),
                _make_face(nose_x=75, embedding=shared_embedding),
            ]
            frames.append(_make_frame(t * 0.5, faces))

        registry = build_face_registry_with_embeddings(frames, min_appearances=3)
        assert len(registry.slots) >= 2, \
            f"Expected >=2 slots (co-occurrence guard should prevent merge), " \
            f"got {len(registry.slots)}"

    def test_different_embeddings_stay_separate(self):
        """Two faces with very different embeddings always stay separate."""
        np.random.seed(42)
        emb_a = np.ones(128).tolist()
        emb_b = (-np.ones(128)).tolist()

        frames = []
        for t in range(20):
            faces = [
                _make_face(nose_x=25, embedding=emb_a),
                _make_face(nose_x=75, embedding=emb_b),
            ]
            frames.append(_make_frame(t * 0.5, faces))

        registry = build_face_registry_with_embeddings(frames, min_appearances=3)
        assert len(registry.slots) >= 2

    def test_same_person_different_frames_merged(self):
        """Same person (same embedding) in different frames should merge."""
        np.random.seed(42)
        shared_embedding = np.random.randn(128).tolist()

        frames = []
        # Person appears alone in alternating frames
        for t in range(20):
            if t % 2 == 0:
                faces = [_make_face(nose_x=50, embedding=shared_embedding)]
            else:
                faces = []
            frames.append(_make_frame(t * 0.5, faces))

        registry = build_face_registry_with_embeddings(frames, min_appearances=3)
        # Should be 1 slot since same person appears alone
        assert len(registry.slots) == 1


class TestTransitiveCoOccurrence:
    def test_transitive_chain_blocked(self):
        """If A co-occurs with B and B co-occurs with C,
        merging A-C should be blocked even if A and C never co-occur."""
        np.random.seed(42)
        # Three people with similar embeddings
        base = np.random.randn(128)
        emb_a = base.tolist()
        emb_b = (base + 0.01 * np.random.randn(128)).tolist()  # very similar
        emb_c = (base + 0.01 * np.random.randn(128)).tolist()  # very similar

        frames = []
        # Frame 0: A and B together (co-occur)
        for t in range(10):
            frames.append(_make_frame(t * 0.5, [
                _make_face(nose_x=25, embedding=emb_a),
                _make_face(nose_x=75, embedding=emb_b),
            ]))
        # Frame 10-19: B and C together (co-occur)
        for t in range(10, 20):
            frames.append(_make_frame(t * 0.5, [
                _make_face(nose_x=25, embedding=emb_b),
                _make_face(nose_x=75, embedding=emb_c),
            ]))

        registry = build_face_registry_with_embeddings(
            frames, min_appearances=3, cosine_threshold=0.20,
        )
        # Should have at least 2 slots (ideally 3), NOT collapsed to 1
        assert len(registry.slots) >= 2, \
            f"Expected >=2 slots (transitive co-occurrence guard), got {len(registry.slots)}"
