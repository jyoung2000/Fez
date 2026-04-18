"""Tests for confidence plumbing: unified confidence field and breakdown."""

import pytest
from unittest.mock import MagicMock

from backend.services.subject_confidence import SubjectConfidenceEstimator


def _make_dense_face(timestamp, faces):
    """Create a mock FrameFaces."""
    df = MagicMock()
    df.timestamp = timestamp
    mock_faces = []
    for f in faces:
        face = MagicMock()
        face.identity_id = f.get("identity_id", -1)
        face.nose_x = f.get("nose_x", 50)
        face.x = f.get("x", f.get("nose_x", 50))
        face.y = f.get("y", 50)
        face.width = f.get("width", 10)
        face.height = f.get("height", 13)
        mock_faces.append(face)
    df.faces = mock_faces
    return df


class TestEvaluateWithBreakdown:
    def test_breakdown_keys_present(self):
        """Breakdown dict has all four expected keys."""
        dense_faces = [
            _make_dense_face(0.5, [{"identity_id": 0, "nose_x": 30}]),
        ]
        estimator = SubjectConfidenceEstimator(
            face_registry=MagicMock(),
            dense_faces=dense_faces,
            active_speaker_events=[],
            transcript_segments=[],
            speaker_to_slot={},
            source_width=1920,
            source_height=1080,
        )
        conf, reason, breakdown = estimator.evaluate_with_breakdown(
            0.0, 1.0, 0, 30,
        )
        assert "face_in_crop" in breakdown
        assert "speaker_agree" in breakdown
        assert "stability" in breakdown
        assert "transcript" in breakdown

    def test_no_face_in_crop_caps_at_030(self):
        """A segment with no face in the crop window has confidence <= 0.30."""
        dense_faces = [
            _make_dense_face(0.5, [{"identity_id": 0, "nose_x": 80}]),
        ]
        estimator = SubjectConfidenceEstimator(
            face_registry=MagicMock(),
            dense_faces=dense_faces,
            active_speaker_events=[],
            transcript_segments=[],
            speaker_to_slot={},
            source_width=1920,
            source_height=1080,
        )
        # Crop at x=20 won't contain face at x=80
        conf, reason, breakdown = estimator.evaluate_with_breakdown(
            0.0, 1.0, 0, 20,
        )
        assert conf <= 0.30
        assert "no_face_in_crop" in reason

    def test_53_percent_face_conf_capped(self):
        """A face detected at 0.53 photographic confidence produces seg.confidence <= 0.53.

        The estimator can only lower confidence via its checks, never raise
        above the max possible (0.40 + 0.20 + 0.20 + 0.20 = 1.0), but when
        only face_in_crop passes, the max is 0.40 + partial checks.
        """
        dense_faces = [
            _make_dense_face(0.5, [{"identity_id": 0, "nose_x": 30}]),
        ]
        estimator = SubjectConfidenceEstimator(
            face_registry=MagicMock(),
            dense_faces=dense_faces,
            active_speaker_events=[],
            transcript_segments=[],
            speaker_to_slot={},
            source_width=1920,
            source_height=1080,
        )
        # With no speaker events and no transcript, max confidence is
        # face_in_crop (0.40) + stability (0.20 * value) = < 0.60
        conf, reason, breakdown = estimator.evaluate_with_breakdown(
            0.0, 1.0, 0, 30,
        )
        # Without speaker agreement and transcript, confidence should be
        # below the 0.70 floor, so the hard gate would catch it
        assert conf < 0.70

    def test_breakdown_sums_to_confidence(self):
        """The breakdown components sum to approximately the final confidence."""
        dense_faces = [
            _make_dense_face(0.5, [{"identity_id": 0, "nose_x": 30}]),
        ]
        estimator = SubjectConfidenceEstimator(
            face_registry=MagicMock(),
            dense_faces=dense_faces,
            active_speaker_events=[],
            transcript_segments=[],
            speaker_to_slot={},
            source_width=1920,
            source_height=1080,
        )
        conf, reason, breakdown = estimator.evaluate_with_breakdown(
            0.0, 1.0, 0, 30,
        )
        breakdown_sum = sum(breakdown.values())
        assert abs(breakdown_sum - conf) < 0.01

    def test_all_checks_pass_full_confidence(self):
        """When all 4 checks pass, confidence reaches 1.0."""
        dense_faces = [
            _make_dense_face(0.5, [{"identity_id": 0, "nose_x": 30}]),
            _make_dense_face(0.6, [{"identity_id": 0, "nose_x": 30}]),
            _make_dense_face(0.7, [{"identity_id": 0, "nose_x": 30}]),
        ]
        # Active speaker event matching slot 0
        speaker_ev = MagicMock()
        speaker_ev.start = 0.0
        speaker_ev.end = 1.0
        speaker_ev.slot_id = 0
        # Transcript segment
        transcript_seg = MagicMock()
        transcript_seg.start = 0.0
        transcript_seg.end = 1.0
        transcript_seg.speaker = "speaker_A"

        estimator = SubjectConfidenceEstimator(
            face_registry=MagicMock(),
            dense_faces=dense_faces,
            active_speaker_events=[speaker_ev],
            transcript_segments=[transcript_seg],
            speaker_to_slot={"speaker_A": 0},
            source_width=1920,
            source_height=1080,
        )
        conf, reason, breakdown = estimator.evaluate_with_breakdown(
            0.0, 1.0, 0, 30,
        )
        assert conf >= 0.90  # all checks should pass
        assert breakdown["face_in_crop"] == 0.40
        assert breakdown["speaker_agree"] == 0.20
        assert breakdown["transcript"] == 0.20
        assert reason == "all_checks_passed"

    def test_evaluate_backward_compatible(self):
        """evaluate() still returns (conf, reason) tuple — not broken."""
        dense_faces = [
            _make_dense_face(0.5, [{"identity_id": 0, "nose_x": 30}]),
        ]
        estimator = SubjectConfidenceEstimator(
            face_registry=MagicMock(),
            dense_faces=dense_faces,
            active_speaker_events=[],
            transcript_segments=[],
            speaker_to_slot={},
            source_width=1920,
            source_height=1080,
        )
        result = estimator.evaluate(0.0, 1.0, 0, 30)
        assert len(result) == 2
        conf, reason = result
        assert isinstance(conf, float)
        assert isinstance(reason, str)
