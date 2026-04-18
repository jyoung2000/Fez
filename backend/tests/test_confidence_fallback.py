"""Tests for the confidence-gated fallback ladder.

Covers:
  5. Empty couch regression (no face in center crop → not center crop)
  6. Confident speaker (face + speaker agrees → high confidence)
  7. Last-known inheritance
  8. Cascading fallback (narrative → wide_master)
  9. Cascading fallback (podcast → blur_fill)
"""

import pytest

from backend.services.subject_confidence import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    SubjectConfidenceEstimator,
    get_fallback_strategy,
)


class _FakeFace:
    def __init__(self, x, y=40, identity_id=0):
        self.x = x
        self.y = y
        self.identity_id = identity_id


class _FakeDenseFace:
    def __init__(self, timestamp, faces):
        self.timestamp = timestamp
        self.faces = faces


class _FakeEvent:
    def __init__(self, slot_id, start, end):
        self.slot_id = slot_id
        self.start = start
        self.end = end


class _FakeSlot:
    def __init__(self, slot_id, x_center):
        self.slot_id = slot_id
        self.x_center = x_center
        self.x_min = x_center - 5
        self.x_max = x_center + 5
        self.frame_count = 100


class _FakeRegistry:
    def __init__(self, slots):
        self.slots = slots
        self.total_frames = 100

    def slot_by_id(self, slot_id):
        for s in self.slots:
            if s.slot_id == slot_id:
                return s
        return None


class TestEmptyCouchRegression:
    """Bug A: 2 faces at x=20 and x=80, candidate_x=50 (center).
    No face in the center crop window → confidence < 0.3 → fallback.
    """

    def test_no_face_in_center_crop(self):
        dense_faces = []
        for t in range(10):
            dense_faces.append(_FakeDenseFace(t, [
                _FakeFace(20, identity_id=0),
                _FakeFace(80, identity_id=1),
            ]))

        registry = _FakeRegistry([_FakeSlot(0, 20), _FakeSlot(1, 80)])
        estimator = SubjectConfidenceEstimator(
            face_registry=registry,
            dense_faces=dense_faces,
            active_speaker_events=[],
            transcript_segments=[],
            speaker_to_slot={},
            source_width=1920,
            source_height=1080,
        )

        # Candidate at center (x=50) for 9:16 crop
        conf, reason = estimator.evaluate(0, 10, None, 50, target_aspect_ratio=9/16)

        # Confidence should be very low — no face in center window
        assert conf <= 0.30, f"Expected conf <= 0.30, got {conf} (reason: {reason})"
        assert "no_face" in reason.lower()

    def test_center_crop_never_fallback(self):
        """The fallback should NOT be a center crop (STATIONARY at x=50)."""
        fallback = get_fallback_strategy(0.25, "podcast")
        assert fallback is not None
        strategy, layout, subject_x, active_slot, reason = fallback
        assert layout != "single", "Low confidence should not produce a single/crop layout"
        assert strategy in ("blur_fill", "wide_master")


class TestConfidentSpeaker:
    """Face at x=56, active speaker agrees, transcript covers."""

    def test_high_confidence(self):
        dense_faces = []
        for t in range(10):
            dense_faces.append(_FakeDenseFace(t, [_FakeFace(56, identity_id=0)]))

        registry = _FakeRegistry([_FakeSlot(0, 56)])

        class FakeTranscriptSeg:
            def __init__(self, start, end, speaker):
                self.start = start
                self.end = end
                self.speaker = speaker
                self.confidence = 0.9

        estimator = SubjectConfidenceEstimator(
            face_registry=registry,
            dense_faces=dense_faces,
            active_speaker_events=[_FakeEvent(0, 0, 10)],
            transcript_segments=[FakeTranscriptSeg(0, 10, "speaker_0")],
            speaker_to_slot={"speaker_0": 0},
            source_width=1920,
            source_height=1080,
        )

        conf, reason = estimator.evaluate(0, 10, 0, 56, target_aspect_ratio=9/16)
        assert conf > 0.85, f"Expected conf > 0.85, got {conf} (reason: {reason})"


class TestLastKnownInheritance:
    """Confident at t=10, brief low at t=12, resume at t=13."""

    def test_medium_confidence_uses_current_position(self):
        """Medium confidence uses the CURRENT candidate position, not stale last-known."""
        fallback = get_fallback_strategy(
            0.55, "podcast",
            last_confident_x=30,
            last_confident_slot=0,
            candidate_x=51,
            candidate_slot=2,
        )
        assert fallback is not None
        strategy, layout, subject_x, active_slot, reason = fallback
        assert subject_x == 51, "Should use current candidate position, not stale last_confident_x"
        assert active_slot == 2
        assert "current" in reason

    def test_medium_confidence_falls_back_to_last_when_no_candidate(self):
        """When no candidate position, medium confidence uses last_confident_x."""
        fallback = get_fallback_strategy(
            0.55, "podcast",
            last_confident_x=30,
            last_confident_slot=0,
            candidate_x=None,
            candidate_slot=None,
        )
        assert fallback is not None
        strategy, layout, subject_x, active_slot, reason = fallback
        assert subject_x == 30, "Should fall back to last confident when no candidate"

    def test_medium_confidence_no_history_falls_through(self):
        fallback = get_fallback_strategy(
            0.55, "podcast",
            last_confident_x=None,
            last_confident_slot=None,
        )
        assert fallback is not None
        strategy, layout, _, _, _ = fallback
        assert strategy in ("blur_fill", "wide_master")


class TestCascadingFallback:
    """Content-type determines fallback preference."""

    def test_narrative_prefers_wide_master(self):
        fallback = get_fallback_strategy(0.45, "narrative")
        assert fallback is not None
        strategy, layout, _, _, _ = fallback
        assert strategy == "wide_master"
        assert layout == "wide_master"

    def test_podcast_prefers_blur_fill(self):
        fallback = get_fallback_strategy(0.45, "podcast")
        assert fallback is not None
        strategy, layout, _, _, _ = fallback
        assert strategy == "blur_fill"
        assert layout == "blur_fill"

    def test_gaming_prefers_blur_fill(self):
        fallback = get_fallback_strategy(0.35, "gaming")
        assert fallback is not None
        strategy, _, _, _, _ = fallback
        assert strategy == "blur_fill"

    def test_very_low_always_wide_master(self):
        # Accuracy-fix Fix 5 raised the wide_master confidence floor
        # from 0.30 to 0.15 so transient detection jitter can't trigger
        # letterbox. Confidence strictly below 0.15 (e.g. 0.10) still
        # ends up at wide_master when no cascade candidate is provided.
        # The old 0.15 → wide_master assertion was locking in the
        # pre-fix behavior.
        fallback = get_fallback_strategy(
            0.10, "podcast",
            candidate_x=None, candidate_slot=None,
        )
        assert fallback is not None
        strategy, layout, _, _, _ = fallback
        assert strategy == "wide_master"

    def test_high_confidence_no_fallback(self):
        fallback = get_fallback_strategy(0.80, "podcast")
        assert fallback is None


class TestFallbackLadderInvariant:
    """THE KEY INVARIANT: confidence < 0.50 NEVER produces a CROP at center."""

    @pytest.mark.parametrize("conf", [0.0, 0.1, 0.2, 0.3, 0.4, 0.49])
    @pytest.mark.parametrize("content_type", [
        "narrative", "podcast", "gaming", "vlog", "sports",
        "music_video", "anime", "unknown",
    ])
    def test_low_confidence_never_center_crop(self, conf, content_type):
        fallback = get_fallback_strategy(conf, content_type)
        assert fallback is not None, f"Expected fallback for conf={conf}, type={content_type}"
        strategy, layout, subject_x, active_slot, reason = fallback
        # Must NOT be a center crop
        assert layout != "single" or subject_x != 50, (
            f"conf={conf}, type={content_type}: got single layout at center (x={subject_x})"
        )
        assert strategy in ("blur_fill", "wide_master", "stationary"), (
            f"conf={conf}, type={content_type}: unexpected strategy={strategy}"
        )
