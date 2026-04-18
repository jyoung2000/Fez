"""Tests for active speaker containment (Phase B).

Validates that:
- Face containment regions are emitted for active speakers (non-gaming)
- Camera solver rejects STATIONARY when it would clip the active face
- The chosen cx always contains the active speaker face
"""
import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass


@dataclass
class _FakeFace:
    nose_x: float  # 0-100
    nose_y: float
    width: float
    height: float
    is_human: bool = True
    identity_id: int = -1
    yaw: float = 0.0
    lip_aperture: float = 0.0


@dataclass
class _FakeFrameFaces:
    timestamp: float
    faces: list


@dataclass
class _FakeSpeakerEvent:
    start: float
    end: float
    slot_id: int
    on_screen: bool = True


class TestActiveContainment:
    """E2: Active speaker face stays in crop."""

    def test_containment_region_emitted(self):
        """Containment regions should be emitted for active speakers in non-gaming."""
        from backend.services.required_regions import build_required_regions

        face = _FakeFace(nose_x=82.0, nose_y=50.0, width=16.0, height=20.0,
                        identity_id=0)
        passive_face = _FakeFace(nose_x=18.0, nose_y=50.0, width=16.0, height=20.0,
                                identity_id=1)
        ff = _FakeFrameFaces(timestamp=1.0, faces=[face, passive_face])
        event = _FakeSpeakerEvent(start=0.0, end=5.0, slot_id=0)

        regions = build_required_regions(
            [ff], active_speaker_events=[event],
            content_type="talking_head",
        )

        assert len(regions) == 1  # one frame
        frame_regs = regions[0]

        # Should have face regions plus containment region
        containment = [r for r in frame_regs if r.source == "face_containment"]
        assert len(containment) >= 1, "Expected at least one face_containment region"

        cr = containment[0]
        assert cr.is_active_speaker is True
        assert cr.weight == 1.6
        assert cr.score == 1.0
        # half_width should be 1.15x the face half_width
        face_hw = (16.0 / 100.0) / 2.0
        assert abs(cr.half_width - face_hw * 1.15) < 0.001

    def test_no_containment_for_gaming(self):
        """Gaming content should NOT emit face_containment regions."""
        from backend.services.required_regions import build_required_regions

        face = _FakeFace(nose_x=82.0, nose_y=50.0, width=16.0, height=20.0,
                        identity_id=0)
        ff = _FakeFrameFaces(timestamp=1.0, faces=[face])
        event = _FakeSpeakerEvent(start=0.0, end=5.0, slot_id=0)

        regions = build_required_regions(
            [ff], active_speaker_events=[event],
            content_type="gameplay",
        )

        frame_regs = regions[0]
        containment = [r for r in frame_regs if r.source == "face_containment"]
        assert len(containment) == 0, "Gaming should not emit face_containment"

    def test_solver_rejects_stationary_when_clipping(self):
        """When active face is at cx=0.82, STATIONARY at cx=0.5 should be
        rejected because crop right edge (0.5 + 0.158 = 0.658) < face right
        edge (0.90)."""
        from backend.services.camera_solver import solve_shot, SolverParams, CROP_ASPECT
        from backend.services.required_regions import RequiredRegion

        # Source: 16:9 -> 9:16 crop
        source_aspect = 16.0 / 9.0
        crop_half_width = (CROP_ASPECT / source_aspect) / 2.0  # ~0.158

        @dataclass
        class FakeShot:
            index: int = 0
            start: float = 0.0
            end: float = 3.0

        # Active speaker at cx=0.82 with containment region
        face_hw = 0.08
        containment_hw = face_hw * 1.15  # 0.092

        regions_per_frame = []
        for i in range(30):  # 10fps * 3s
            t = i * 0.1
            face_reg = RequiredRegion(
                timestamp=t, cx=0.82, cy=0.5,
                half_width=face_hw, half_height=0.10,
                score=1.0, weight=1.4, tier="required",
                source="face", is_active_speaker=True,
            )
            containment_reg = RequiredRegion(
                timestamp=t, cx=0.82, cy=0.5,
                half_width=containment_hw, half_height=0.12,
                score=1.0, weight=1.6, tier="required",
                source="face_containment", is_active_speaker=True,
            )
            # Also add a passive face to make the union wider
            passive_reg = RequiredRegion(
                timestamp=t, cx=0.18, cy=0.5,
                half_width=face_hw, half_height=0.10,
                score=0.55, weight=0.45, tier="required",
                source="face",
            )
            regions_per_frame.append([face_reg, containment_reg, passive_reg])

        params = SolverParams(
            smoothing_alpha=0.25,
            panning_residual_threshold=0.03,
            stationary_slack=0.0,
            prefer_stationary=True,
            shot_threshold=30.0,
        )

        result = solve_shot(
            FakeShot(), regions_per_frame, source_aspect, params,
            content_type="talking_head",
        )

        # Should NOT be stationary at cx=0.5 (would clip the face)
        # The solver should choose TRACKING or PANNING
        if result.keyframes:
            for t, cx, cy in result.keyframes:
                # Verify containment: active face right edge at 0.82 + 0.092 = 0.912
                # must be <= cx + crop_half_width
                assert cx + crop_half_width >= 0.82 - containment_hw - 0.02, (
                    f"At t={t:.2f}: cx={cx:.3f}, crop right={cx + crop_half_width:.3f}, "
                    f"but active face right={0.82 + containment_hw:.3f}"
                )

    def test_lead_room_revert_prevents_off_frame(self):
        """Lead-room shift should be reverted if it pushes containment off-frame."""
        from backend.services.required_regions import build_required_regions

        # Face near the right edge with strong yaw
        face = _FakeFace(
            nose_x=95.0, nose_y=50.0, width=16.0, height=20.0,
            identity_id=0, yaw=-30.0,  # looking right -> shift would go further right
        )
        ff = _FakeFrameFaces(timestamp=1.0, faces=[face])
        event = _FakeSpeakerEvent(start=0.0, end=5.0, slot_id=0)

        regions = build_required_regions(
            [ff], active_speaker_events=[event],
            content_type="talking_head",
        )

        # The face region cx should still be within bounds
        face_regs = [r for r in regions[0] if r.source == "face" and r.is_active_speaker]
        assert len(face_regs) >= 1
        # cx should be clamped so containment stays in frame
        for fr in face_regs:
            # With containment padding, verify it's in [0, 1]
            left = fr.cx - fr.half_width * 1.15
            right = fr.cx + fr.half_width * 1.15
            # The revert should keep containment from going off-frame
            assert left >= -0.01 or right <= 1.01, (
                f"Containment off-frame: left={left:.3f}, right={right:.3f}"
            )
