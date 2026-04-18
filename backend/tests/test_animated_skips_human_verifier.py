"""Test that animated content (high human-verifier rejection rate) causes
every face to be reset to is_human=True so the face registry doesn't
penalize anime speakers down to weight=0.35.

Bug 2 from the Bungo Stray Dogs run: 1436 of 1952 faces (73.6%) were
rejected by MediaPipe Pose, and the registry kept them at weight=0.35,
leaving anime speakers below the saliency ceiling and letting the crop
drift onto background motion.
"""

from dataclasses import dataclass, field
from unittest.mock import patch, MagicMock


@dataclass
class _Face:
    nose_x: float = 50.0
    nose_y: float = 50.0
    width: float = 10.0
    height: float = 12.0
    confidence: float = 0.9
    lip_aperture: float = 0.0
    identity_id: int = -1
    is_human: bool = True  # set to False by verifier initially
    pose_confidence: float = 0.0
    y_bottom: float = 0.0


@dataclass
class _FrameFaces:
    timestamp: float
    frame_path: str = "/tmp/frame.jpg"
    faces: list = field(default_factory=list)


class _FakeVerifier:
    """Fake verifier that rejects 80% of faces as non-human — the typical
    signature of an anime episode being processed by MediaPipe Pose."""

    available = True
    _counter = {"n": 0}

    def verify_faces_in_frame(self, frame_bgr, faces, w, h):
        for face in faces:
            self._counter["n"] += 1
            # Reject 4 out of every 5 faces
            face.is_human = (self._counter["n"] % 5 == 0)
            face.pose_confidence = 0.9 if face.is_human else 0.1


def test_high_rejection_rate_restores_is_human():
    """When >50% of faces fail verification, all faces should be reset
    to is_human=True with pose_confidence=1.0 and the `[AnimeMode]` log
    line should fire."""
    from backend.services import face_detector

    frames = [
        _FrameFaces(
            timestamp=float(i),
            faces=[_Face(nose_x=30 + i % 10, identity_id=0),
                   _Face(nose_x=70 - i % 10, identity_id=1)],
        )
        for i in range(20)
    ]

    with patch.object(
        face_detector, "get_verifier",
        return_value=_FakeVerifier(),
        create=True,
    ) if hasattr(face_detector, "get_verifier") else patch(
        "backend.services.human_face_verifier.get_verifier",
        return_value=_FakeVerifier(),
    ):
        with patch("cv2.imread") as mock_imread:
            mock_frame = MagicMock()
            mock_frame.shape = (720, 1280, 3)
            mock_imread.return_value = mock_frame

            frame_paths = [(f.timestamp, "/tmp/frame.jpg") for f in frames]
            result = face_detector._verify_faces_in_results(frames, frame_paths)

    # After the auto-detect override, every face should be is_human=True.
    total = 0
    non_human = 0
    for fr in result:
        for face in fr.faces:
            total += 1
            if not face.is_human:
                non_human += 1
    assert total == 40
    assert non_human == 0, (
        f"Expected all faces restored to is_human=True in anime mode, "
        f"but {non_human} remain as non-human"
    )


def test_low_rejection_rate_leaves_faces_alone():
    """When <50% of faces are rejected, the auto-detect must NOT fire —
    live-action behavior is preserved.
    """
    from backend.services import face_detector

    class _MostlyHumanVerifier:
        available = True
        _counter = {"n": 0}

        def verify_faces_in_frame(self, frame_bgr, faces, w, h):
            for face in faces:
                self._counter["n"] += 1
                # Reject only 1 out of every 20 faces
                face.is_human = (self._counter["n"] % 20 != 0)
                face.pose_confidence = 0.9 if face.is_human else 0.1

    frames = [
        _FrameFaces(
            timestamp=float(i),
            faces=[_Face(nose_x=50, identity_id=0)],
        )
        for i in range(40)
    ]

    with patch(
        "backend.services.human_face_verifier.get_verifier",
        return_value=_MostlyHumanVerifier(),
    ):
        with patch("cv2.imread") as mock_imread:
            mock_frame = MagicMock()
            mock_frame.shape = (720, 1280, 3)
            mock_imread.return_value = mock_frame

            frame_paths = [(f.timestamp, "/tmp/frame.jpg") for f in frames]
            result = face_detector._verify_faces_in_results(frames, frame_paths)

    # About 2 of 40 should still be non-human — anime override NOT fired.
    non_human = sum(1 for fr in result for f in fr.faces if not f.is_human)
    assert 1 <= non_human <= 4, (
        f"Live-action path should keep the ~5% rejected faces "
        f"as non-human; got {non_human}"
    )
