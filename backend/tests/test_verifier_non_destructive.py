"""Phase 1 — Red test: is_human=False should weight down, not gate.

Bug C: face_registry.py hard-filters faces with is_human=False,
completely discarding them from clustering. HumanFaceVerifier marks
~15% of real faces as non-human on profile views and stage lighting.
Those faces should still contribute to clustering evidence with
reduced weight, not be deleted entirely.

This test creates 100 faces where half are flagged is_human=False.
The registry should still reflect all 100 faces' positions (weighted).
"""

from dataclasses import dataclass, field
from typing import Optional

import pytest

from backend.services.face_registry import build_face_registry


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
    pose_confidence: float = 0.0


@dataclass
class _FrameFaces:
    timestamp: float
    frame_path: str = ""
    faces: list = field(default_factory=list)
    primary_face_idx: int = -1


class TestVerifierNonDestructive:
    """is_human=False faces should be weighted down, not deleted."""

    def test_non_human_faces_still_contribute_to_slot_count(self):
        """Two speakers: speaker A has 50 is_human=True faces,
        speaker B has 50 is_human=False faces (profile views, bad lighting).

        Both speakers should appear as slots in the registry. Currently
        speaker B is completely invisible because all their faces are
        hard-filtered at line 137.
        """
        frames = []
        # Speaker A at x=25, always is_human=True (50 frames)
        for i in range(50):
            frames.append(_FrameFaces(
                timestamp=i * 0.5,
                faces=[_FaceInfo(
                    x_center=25, nose_x=25,
                    is_human=True,
                )],
            ))
        # Speaker B at x=75, always is_human=False (50 frames)
        for i in range(50):
            frames.append(_FrameFaces(
                timestamp=25 + i * 0.5,
                faces=[_FaceInfo(
                    x_center=75, nose_x=75,
                    is_human=False,
                )],
            ))

        registry = build_face_registry(frames, min_appearances=3)

        # Both speakers should be detected
        assert len(registry.slots) >= 2, (
            f"Expected 2 slots (speaker A + speaker B), got {len(registry.slots)}. "
            f"is_human=False faces were likely hard-filtered instead of weighted."
        )

    def test_mixed_human_nonhuman_same_speaker(self):
        """One speaker with 70% is_human=True and 30% is_human=False.

        The speaker should appear with their full frame count, not with
        30% of their evidence deleted.
        """
        frames = []
        for i in range(100):
            is_hum = i % 10 < 7  # 70% True, 30% False
            frames.append(_FrameFaces(
                timestamp=i * 0.5,
                faces=[_FaceInfo(
                    x_center=40, nose_x=40,
                    is_human=is_hum,
                )],
            ))

        registry = build_face_registry(frames, min_appearances=3)

        assert len(registry.slots) >= 1, "Expected at least 1 slot"
        # The slot should reflect ALL appearances, not just the 70% human ones
        slot = registry.slots[0]
        # With hard filter: 70 frames. With weight: should count all 100 frames
        assert slot.frame_count >= 90, (
            f"Expected frame_count >= 90 (weighted evidence from all 100 frames), "
            f"got {slot.frame_count}. is_human=False faces were likely hard-filtered."
        )
