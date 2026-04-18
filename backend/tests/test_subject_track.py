"""Tests for SubjectTrack dataclass and FACE_LIKE FeatureKind."""

import numpy as np

from backend.services.focus_model import FeatureKind, SubjectTrack


class TestFeatureKindFaceLike:
    def test_face_like_accessible(self):
        """FACE_LIKE kind is accessible."""
        assert FeatureKind.FACE_LIKE == "face_like"
        assert FeatureKind.FACE_LIKE.value == "face_like"

    def test_face_like_distinct_from_face(self):
        """FACE_LIKE is distinct from FACE."""
        assert FeatureKind.FACE_LIKE != FeatureKind.FACE


class TestSubjectTrack:
    def test_empty_trajectory(self):
        """Empty bbox_trajectory returns t_start=0, t_end=0."""
        track = SubjectTrack(track_id=0, source="face_confirmed", confidence=0.9)
        assert track.t_start == 0.0
        assert track.t_end == 0.0

    def test_bbox_at_outside_range(self):
        """bbox_at(t) returns None outside trajectory range."""
        track = SubjectTrack(
            track_id=0, source="face_confirmed", confidence=0.9,
            bbox_trajectory=[(1.0, 50.0, 40.0, 10.0, 13.0), (2.0, 55.0, 40.0, 10.0, 13.0)],
        )
        assert track.bbox_at(0.5) is None
        assert track.bbox_at(3.0) is None

    def test_bbox_at_empty(self):
        """bbox_at on empty trajectory returns None."""
        track = SubjectTrack(track_id=0, source="face_confirmed", confidence=0.9)
        assert track.bbox_at(1.0) is None

    def test_bbox_at_inside_range(self):
        """bbox_at(t) returns nearest sample inside range."""
        track = SubjectTrack(
            track_id=0, source="face_confirmed", confidence=0.9,
            bbox_trajectory=[
                (1.0, 50.0, 40.0, 10.0, 13.0),
                (2.0, 55.0, 40.0, 10.0, 13.0),
                (3.0, 60.0, 40.0, 10.0, 13.0),
            ],
        )
        bbox = track.bbox_at(1.8)
        assert bbox is not None
        assert bbox == (55.0, 40.0, 10.0, 13.0)  # nearest to t=2.0

    def test_t_start_and_t_end(self):
        """t_start and t_end match first/last trajectory entry."""
        track = SubjectTrack(
            track_id=0, source="face_like_promoted", confidence=0.6,
            bbox_trajectory=[(0.5, 50.0, 40.0, 10.0, 13.0), (5.0, 60.0, 40.0, 10.0, 13.0)],
        )
        assert track.t_start == 0.5
        assert track.t_end == 5.0

    def test_to_dict_sanitizes(self):
        """to_dict() sanitizes all numeric fields."""
        track = SubjectTrack(
            track_id=0, source="face_like_promoted",
            confidence=np.float64(0.85),
            face_slot_id=None,
            bbox_trajectory=[(np.float64(1.0), 50.0, 40.0, 10.0, 13.0)],
            persistent_id=np.int64(2),
            gate_persistence_frames=7,
            gate_aspect_ratio=np.float32(1.3),
            gate_area_ratio=np.float64(0.03),
            gate_face_mesh_passed=True,
        )
        d = track.to_dict()
        assert isinstance(d["track_id"], int)
        assert isinstance(d["source"], str)
        assert isinstance(d["confidence"], (int, float))
        assert not hasattr(d["confidence"], 'item')
        assert isinstance(d["persistent_id"], int)
        assert isinstance(d["t_start"], (int, float))
        assert not hasattr(d["t_start"], 'item')
        assert isinstance(d["gate_aspect_ratio"], (int, float))
        assert not hasattr(d["gate_aspect_ratio"], 'item')
        assert isinstance(d["gate_face_mesh_passed"], bool)
        assert d["bbox_count"] == 1
