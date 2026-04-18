"""Phase 1 — Red test: wide-span embedding clusters should NOT be rejected.

Bug A: build_face_registry_with_embeddings rejects clusters with
(x_max - x_min) > 40, but on stage/panel formats a single person
legitimately spans >40% of frame width due to camera pans and movement.
The span check is a false-negative detector.

This test creates 6 identities with distinct embeddings but wide x-span
(simulating stage movement). The embedding clusterer should keep all 6,
not fall back to position-based clustering.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import pytest

from backend.services.face_registry import build_face_registry_with_embeddings


@dataclass
class _FaceInfo:
    x_center: float
    y_center: float = 50.0
    width: float = 8.0
    height: float = 10.0
    nose_x: float = 50.0
    nose_y: float = 50.0
    confidence: float = 0.9
    lip_aperture: float = 0.0
    identity_embedding: Optional[list] = None
    identity_id: int = -1
    is_speaking: bool = False
    y_bottom: float = 0.0
    is_human: bool = True
    pose_confidence: float = 0.9


@dataclass
class _FrameFaces:
    timestamp: float
    frame_path: str = ""
    faces: list = field(default_factory=list)
    primary_face_idx: int = -1


def _make_embedding(seed: int, dim: int = 128) -> list:
    """Generate a deterministic unit-norm embedding for a given identity seed."""
    rng = np.random.RandomState(seed)
    emb = rng.randn(dim).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return emb.tolist()


def _make_frame_with_faces(timestamp: float, face_specs: list[tuple]) -> _FrameFaces:
    """Create a frame with multiple faces.

    face_specs: [(identity_seed, nose_x), ...]
    """
    faces = []
    for seed, nx in face_specs:
        faces.append(_FaceInfo(
            x_center=nx,
            nose_x=nx,
            identity_embedding=_make_embedding(seed),
        ))
    return _FrameFaces(timestamp=timestamp, faces=faces)


class TestWideSpanEmbeddingClusters:
    """6 identities with wide x-span must survive embedding clustering."""

    def test_six_speakers_with_stage_movement_preserved(self):
        """6 distinct embeddings, each drifting across ~50% of frame width.

        The embedding clusterer correctly groups them by embedding similarity.
        The span check at line 602 incorrectly rejects them because
        (x_max - x_min) > 40 for each cluster. This test asserts the
        embedding result is kept, not the position-based fallback.
        """
        # 6 identities with seeds 100-105
        # Each identity appears in multiple frames at varying x-positions
        # (simulating stage movement: walking across ~50% of frame)
        frames = []
        t = 0.0

        # Identity 0 (seed=100): moves from x=10 to x=60 across frames
        # Identity 1 (seed=101): moves from x=15 to x=65
        # Identity 2 (seed=102): moves from x=20 to x=70
        # Identity 3 (seed=103): moves from x=25 to x=75
        # Identity 4 (seed=104): moves from x=30 to x=80
        # Identity 5 (seed=105): moves from x=35 to x=85
        identity_ranges = [
            (100, 10, 60),   # seed, x_start, x_end
            (101, 15, 65),
            (102, 20, 70),
            (103, 25, 75),
            (104, 30, 80),
            (105, 35, 85),
        ]

        # Generate 30 frames with 2-3 identities visible per frame
        # (not all 6 visible at once — simulating a stage show)
        for i in range(30):
            t = i * 1.0
            # Choose which 2-3 identities are visible this frame
            visible = []
            for idx, (seed, x_start, x_end) in enumerate(identity_ranges):
                # Each identity visible in ~50% of frames, staggered
                if (i + idx * 3) % 5 < 3:
                    # x-position drifts across the range
                    progress = i / 29.0
                    x = x_start + (x_end - x_start) * progress
                    visible.append((seed, x))

            if visible:
                frames.append(_make_frame_with_faces(t, visible))

        registry = build_face_registry_with_embeddings(frames, min_appearances=3)

        # The embedding clusterer should find 6 distinct identities.
        # Current code rejects them because each cluster spans > 40% width
        # and falls back to position-based which finds ~4.
        assert len(registry.slots) >= 6, (
            f"Expected >= 6 slots from 6 distinct embeddings, got {len(registry.slots)}. "
            f"The span gate likely rejected valid wide-span clusters."
        )

    def test_two_speakers_crossing_paths_not_merged(self):
        """Two speakers who walk past each other should remain two slots,
        even though their combined x-range overlaps."""
        frames = []

        # Speaker A (seed=200): moves left to right (x=20 -> 80)
        # Speaker B (seed=201): moves right to left (x=80 -> 20)
        # Both visible in every frame
        for i in range(20):
            t = i * 0.5
            progress = i / 19.0
            xa = 20 + 60 * progress      # 20 -> 80
            xb = 80 - 60 * progress      # 80 -> 20
            frames.append(_make_frame_with_faces(t, [
                (200, xa),
                (201, xb),
            ]))

        registry = build_face_registry_with_embeddings(frames, min_appearances=3)

        assert len(registry.slots) >= 2, (
            f"Expected 2 slots for 2 speakers crossing paths, got {len(registry.slots)}"
        )
