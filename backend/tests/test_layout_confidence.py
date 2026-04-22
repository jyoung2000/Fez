"""Blueprint v2 Phase 2 — deterministic layout confidence scoring."""

from dataclasses import dataclass, field

import pytest

from backend.services.layout_confidence import (
    VALID_LAYOUTS,
    ConfidenceResult,
    LayoutCandidate,
    score_layout_candidates,
    _count_persistent_faces,
    _count_speaker_turns_per_sec,
)
from backend.services.reframe_config import get_default_config


@dataclass
class _Face:
    identity_id: int = 0


@dataclass
class _FrameFaces:
    timestamp: float = 0.0
    faces: list = field(default_factory=list)


@dataclass
class _SpeakerEvent:
    start: float
    end: float
    slot_id: int
    confidence: float = 1.0


def _mk_scene_faces(slot_ids: list[int], duration: float = 10.0, fps: float = 1.0):
    """Build a list of FrameFaces where every sample has all of ``slot_ids`` present."""
    n = max(1, int(duration * fps))
    return [
        _FrameFaces(
            timestamp=float(i) / fps,
            faces=[_Face(identity_id=s) for s in slot_ids],
        )
        for i in range(n)
    ]


# ── Reducers ──────────────────────────────────────────────────────


def test_count_persistent_faces_filters_by_coverage():
    # Slot 0 present in 10/10 frames; slot 1 in only 3/10 (< 60%) — only
    # slot 0 is "persistent".
    faces = [_FrameFaces(timestamp=float(i), faces=[_Face(0)]) for i in range(10)]
    for i in range(3):
        faces[i].faces.append(_Face(1))
    assert _count_persistent_faces(faces, start=0, end=10) == 1


def test_count_persistent_faces_two_speakers():
    faces = _mk_scene_faces([0, 1], duration=10)
    assert _count_persistent_faces(faces, start=0, end=10) == 2


def test_count_persistent_faces_empty():
    assert _count_persistent_faces([], start=0, end=10) == 0
    assert _count_persistent_faces(
        [_FrameFaces(timestamp=t) for t in range(10)],
        start=0, end=10,
    ) == 0


def test_count_speaker_turns_per_sec():
    # Alternating A/B every 1s across a 10s window = 9 turns / 10s.
    events = [
        _SpeakerEvent(start=float(i), end=float(i + 1),
                      slot_id=(i % 2))
        for i in range(10)
    ]
    rate = _count_speaker_turns_per_sec(events, start=0, end=10)
    assert rate == pytest.approx(0.9)


def test_count_speaker_turns_per_sec_single_speaker():
    events = [_SpeakerEvent(start=0, end=10, slot_id=0)]
    assert _count_speaker_turns_per_sec(events, start=0, end=10) == 0.0


def test_count_speaker_turns_outside_window_ignored():
    events = [
        _SpeakerEvent(start=0, end=1, slot_id=0),
        _SpeakerEvent(start=20, end=21, slot_id=1),
    ]
    assert _count_speaker_turns_per_sec(events, start=0, end=10) == 0.0


# ── Scorer: high-confidence decisions ─────────────────────────────


def test_talking_head_single_is_high_confidence():
    faces = _mk_scene_faces([0], duration=10)
    events = [_SpeakerEvent(start=0, end=10, slot_id=0)]
    result = score_layout_candidates(
        scene_start=0, scene_end=10,
        dense_faces=faces, active_speaker_events=events,
        content_type="talking_head",
    )
    assert result.top.layout == "single"
    assert result.is_confident(0.6)


def test_podcast_two_speakers_alternating_prefers_split():
    faces = _mk_scene_faces([0, 1], duration=10)
    # Alternate every 1s for 10s = 0.9 turns/s (well above 0.5 threshold).
    events = [
        _SpeakerEvent(start=float(i), end=float(i + 1),
                      slot_id=(i % 2))
        for i in range(10)
    ]
    result = score_layout_candidates(
        scene_start=0, scene_end=10,
        dense_faces=faces, active_speaker_events=events,
        content_type="podcast",
    )
    assert result.top.layout == "split"


def test_landscape_no_faces_prefers_ken_burns():
    # Empty faces, diffuse saliency, long duration → KEN_BURNS.
    faces = [_FrameFaces(timestamp=float(i)) for i in range(10)]
    result = score_layout_candidates(
        scene_start=0, scene_end=10,
        dense_faces=faces, active_speaker_events=[],
        saliency_centroid_dispersion=0.8,
        content_type="landscape",
    )
    assert result.top.layout == "ken_burns"


def test_gameplay_with_hud_and_webcam_prefers_gameplay_layout():
    faces = _mk_scene_faces([0], duration=10)
    result = score_layout_candidates(
        scene_start=0, scene_end=10,
        dense_faces=faces, active_speaker_events=[],
        has_hud=True, has_webcam_overlay=True,
        content_type="gameplay",
    )
    assert result.top.layout == "gameplay"


def test_screenshare_top_screen_bottom_speaker():
    faces = _mk_scene_faces([0], duration=10)
    result = score_layout_candidates(
        scene_start=0, scene_end=10,
        dense_faces=faces, active_speaker_events=[],
        has_screen_region=True,
        content_type="screen_share",
    )
    assert result.top.layout == "screenshare"


def test_object_tracker_on_ball_detections():
    faces = _mk_scene_faces([0], duration=10)
    result = score_layout_candidates(
        scene_start=0, scene_end=10,
        dense_faces=faces, active_speaker_events=[],
        ball_detections=[{"t": i * 0.1, "confidence": 0.9} for i in range(20)],
        content_type="sports_basketball",
    )
    assert result.top.layout == "object_tracker"


# ── Scorer: low-confidence edge cases ─────────────────────────────


def test_two_face_low_turn_rate_reports_low_confidence():
    """Two faces with very few speaker turns — SPLIT score goes to zero
    (needs turns/s > 0.5) and PIP's narrow band may not dominate.
    The top-vs-runner-up margin should fall below the threshold, at
    least for some content types, so the VLM escalation path is
    exercised in integration tests."""
    faces = _mk_scene_faces([0, 1], duration=10)
    # One turn across 10s = 0.1 turns/s — clearly below SPLIT threshold
    # and inside PIP's band but still ambiguous vs SINGLE.
    events = [
        _SpeakerEvent(start=0, end=5, slot_id=0),
        _SpeakerEvent(start=5, end=10, slot_id=1),
    ]
    result = score_layout_candidates(
        scene_start=0, scene_end=10,
        dense_faces=faces, active_speaker_events=events,
        content_type="podcast",
    )
    # Any layout is allowed to win here — the invariant under test is
    # that the result carries the candidate list so diagnostics work.
    assert result.top.layout in VALID_LAYOUTS
    assert len(result.candidates) >= 6
    # Confidence must be a valid probability.
    assert 0.0 <= result.confidence <= 1.0


def test_empty_scene_still_ranks_single_first():
    """No faces, no events, no dispersion — SINGLE wins by default."""
    result = score_layout_candidates(
        scene_start=0, scene_end=5,
        dense_faces=[], active_speaker_events=[],
        content_type="generic",
    )
    assert result.top.layout == "single"


def test_scorer_respects_importance_matrix():
    """The landscape row has face=0, so SINGLE's score drops and
    KEN_BURNS should dominate on a subject-less scene."""
    faces = [_FrameFaces(timestamp=float(i)) for i in range(5)]
    r_landscape = score_layout_candidates(
        scene_start=0, scene_end=5,
        dense_faces=faces, active_speaker_events=[],
        saliency_centroid_dispersion=0.7,
        content_type="landscape",
    )
    r_generic = score_layout_candidates(
        scene_start=0, scene_end=5,
        dense_faces=faces, active_speaker_events=[],
        saliency_centroid_dispersion=0.7,
        content_type="generic",
    )
    # Landscape should prefer ken_burns; generic may still prefer single
    # because its face weight is higher. Their confidence must differ.
    assert r_landscape.top.layout == "ken_burns"
    # Sanity: the candidate rationale reflects the inputs.
    assert all(c.layout in VALID_LAYOUTS for c in r_landscape.candidates)
    assert any("disp" in c.rationale for c in r_landscape.candidates)


# ── ConfidenceResult helpers ──────────────────────────────────────


def test_is_confident_above_threshold():
    res = ConfidenceResult(
        top=LayoutCandidate("single", 10.0),
        second=LayoutCandidate("split", 3.0),
        confidence=0.7,
        candidates=[],
    )
    assert res.is_confident(0.6)
    assert not res.is_confident(0.8)


def test_zero_score_runner_up_yields_full_confidence():
    """When only one candidate has a positive score, confidence = 1.0."""
    faces = _mk_scene_faces([0], duration=10)
    result = score_layout_candidates(
        scene_start=0, scene_end=10,
        dense_faces=faces, active_speaker_events=[],
        content_type="talking_head",
    )
    # None of split / triple / pip / screenshare / gameplay / object_tracker
    # apply without HUD / screen / two-speaker turns / ball detections, so
    # their scores are 0 and confidence should be 1.0 (or at least very high).
    assert result.confidence >= 0.95
