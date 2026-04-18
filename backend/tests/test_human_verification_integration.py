"""Integration tests for human face verification in the detection pipeline.

Tests that non-human faces (figurines, posters) are filtered from:
  - FaceRegistry slot creation
  - SceneFocus required feature collection
  - SubjectFusion track ingestion

And that the verifier fails open when unavailable.
"""

import pytest
from unittest.mock import MagicMock, patch

from backend.services.face_detector import FaceInfo, FrameFaces
from backend.services.face_registry import build_face_registry, FaceRegistry
from backend.services.scene_focus import aggregate_scene_focus
from backend.services.focus_model import FeatureKind


def _make_face(nose_x=30, nose_y=40, width=8, height=10, identity_id=-1,
               is_human=True, confidence=0.9):
    return FaceInfo(
        x_center=nose_x, y_center=nose_y,
        width=width, height=height,
        nose_x=nose_x, nose_y=nose_y,
        confidence=confidence,
        identity_id=identity_id,
        is_human=is_human,
    )


def _make_frame_faces(timestamp, faces):
    return FrameFaces(
        timestamp=timestamp,
        frame_path=f"/tmp/frame_{timestamp}.jpg",
        faces=faces,
        primary_face_idx=0 if faces else -1,
    )


class TestRegistryFiltering:
    def test_figurine_not_in_registry(self):
        """A frame with only a figurine: registry has zero slots."""
        frames = []
        for t in range(10):
            frames.append(_make_frame_faces(
                t * 0.5,
                [_make_face(nose_x=70, is_human=False)],
            ))

        registry = build_face_registry(frames, min_appearances=3)
        assert len(registry.slots) == 0

    def test_human_and_figurine_only_human_slot(self):
        """Frame with one human and one figurine: only human becomes a slot."""
        frames = []
        for t in range(10):
            frames.append(_make_frame_faces(
                t * 0.5,
                [
                    _make_face(nose_x=25, is_human=True),
                    _make_face(nose_x=75, is_human=False),
                ],
            ))

        registry = build_face_registry(frames, min_appearances=3)
        assert len(registry.slots) == 1
        assert abs(registry.slots[0].x_center - 25) < 10

    def test_fail_open_when_no_is_human_attr(self):
        """Faces without is_human attribute are treated as human (fail-open)."""
        face = MagicMock()
        face.nose_x = 30
        face.x_center = 30
        face.width = 8
        face.height = 10
        # Deliberately no is_human attribute — getattr should return True
        del face.is_human

        frames = []
        for t in range(10):
            frames.append(_make_frame_faces(t * 0.5, [face]))

        registry = build_face_registry(frames, min_appearances=3)
        # Should have a slot since face is accepted by default
        assert len(registry.slots) >= 1


class TestSceneFocusFiltering:
    def test_non_human_faces_excluded_from_required(self):
        """Non-human faces are excluded from required features in scene_focus."""
        dense_faces = []
        for t in [0.1, 0.2, 0.3]:
            dense_faces.append(_make_frame_faces(
                t,
                [
                    _make_face(nose_x=30, is_human=True, identity_id=0),
                    _make_face(nose_x=70, is_human=False, identity_id=1),
                ],
            ))

        focus = aggregate_scene_focus(
            shot_start=0.0,
            shot_end=0.5,
            dense_faces=dense_faces,
            saliency_keyframes=[],
            persistent_regions=None,
            face_registry=MagicMock(),
            source_width=1920,
            source_height=1080,
        )

        # Only human faces should be in required features
        face_features = [rf for rf in focus.required if rf.kind == FeatureKind.FACE]
        for ff in face_features:
            # All face features should come from the human face at x=30
            assert abs(ff.x - 30) < 5

    def test_all_non_human_means_no_required_faces(self):
        """When all faces are non-human, no face features are required."""
        dense_faces = []
        for t in [0.1, 0.2]:
            dense_faces.append(_make_frame_faces(
                t,
                [_make_face(nose_x=50, is_human=False, identity_id=0)],
            ))

        focus = aggregate_scene_focus(
            shot_start=0.0,
            shot_end=0.5,
            dense_faces=dense_faces,
            saliency_keyframes=[],
            persistent_regions=None,
            face_registry=MagicMock(),
            source_width=1920,
            source_height=1080,
        )

        face_features = [rf for rf in focus.required if rf.kind == FeatureKind.FACE]
        assert len(face_features) == 0


class TestVerificationFeatureFlag:
    def test_use_human_verification_false_accepts_all(self):
        """When USE_HUMAN_VERIFICATION=false, verification is skipped."""
        from backend.services.face_detector import _verify_faces_in_results

        frames = [
            _make_frame_faces(0.5, [_make_face(nose_x=50, is_human=True)]),
        ]

        with patch.dict("os.environ", {"USE_HUMAN_VERIFICATION": "false"}):
            result = _verify_faces_in_results(frames, [(0.5, "/tmp/frame.jpg")])

        # Faces should be unchanged (not re-verified)
        assert result[0].faces[0].is_human is True

    def test_verifier_unavailable_accepts_all(self):
        """When the verifier can't load, all faces are accepted."""
        from backend.services.face_detector import _verify_faces_in_results

        frames = [
            _make_frame_faces(0.5, [_make_face(nose_x=50, is_human=True)]),
        ]

        with patch("backend.services.face_detector.os.environ.get",
                    side_effect=lambda k, d="": "true" if k == "USE_HUMAN_VERIFICATION" else d):
            with patch("backend.services.human_face_verifier.get_verifier") as mock_get:
                mock_verifier = MagicMock()
                mock_verifier.available = False
                mock_get.return_value = mock_verifier
                result = _verify_faces_in_results(frames, [(0.5, "/tmp/frame.jpg")])

        assert result[0].faces[0].is_human is True
