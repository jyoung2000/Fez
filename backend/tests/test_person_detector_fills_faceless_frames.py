"""v4 Change 1 — PersonDetector + AttentionAnchor person_body fallback.

Mocks PersonRegion + faceless FrameFaces and asserts that the
build_attention_anchors priority chain emits source="person_body" with
the head-bias cy applied for frames that have a person bbox but no face.
"""

from dataclasses import dataclass, field

from backend.services.attention_anchor import (
    build_attention_anchors,
    PERSON_BODY_CONFIDENCE,
    PERSON_HEAD_CY_BIAS,
)


@dataclass
class _Face:
    nose_x: float = 50.0
    nose_y: float = 50.0
    width: float = 10.0
    height: float = 12.0
    identity_id: int = 0
    lip_aperture: float = 0.0
    is_human: bool = True


@dataclass
class _FrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)


@dataclass
class _PersonRegion:
    timestamp: float
    cx: float
    cy: float
    width: float
    height: float
    confidence: float = 0.85
    has_face: bool = False


def test_person_body_anchor_emitted_for_faceless_frames():
    # 5 faceless frames at evenly spaced timestamps. Each has a person
    # bbox at a known location. Expect person_body anchors with cy
    # biased upward by PERSON_HEAD_CY_BIAS × height.
    frames = [
        _FrameFaces(timestamp=0.0, faces=[]),
        _FrameFaces(timestamp=0.5, faces=[]),
        _FrameFaces(timestamp=1.0, faces=[]),
        _FrameFaces(timestamp=1.5, faces=[]),
        _FrameFaces(timestamp=2.0, faces=[]),
    ]
    persons = [
        _PersonRegion(timestamp=0.0, cx=0.3, cy=0.6, width=0.20, height=0.50),
        _PersonRegion(timestamp=0.5, cx=0.4, cy=0.6, width=0.20, height=0.50),
        _PersonRegion(timestamp=1.0, cx=0.5, cy=0.6, width=0.20, height=0.50),
        _PersonRegion(timestamp=1.5, cx=0.6, cy=0.6, width=0.20, height=0.50),
        _PersonRegion(timestamp=2.0, cx=0.7, cy=0.6, width=0.20, height=0.50),
    ]

    anchors = build_attention_anchors(
        frame_faces=frames,
        active_speaker_events=[],
        frame_saliency=None,
        shot_cuts=[],
        frame_persons=persons,
    )

    assert len(anchors) == 5
    person_body_count = sum(1 for a in anchors if a.source == "person_body")
    assert person_body_count == 5, (
        f"expected 5 person_body anchors, got {person_body_count}"
    )

    # cx should match the person bbox center.
    expected_cx = [0.3, 0.4, 0.5, 0.6, 0.7]
    for i, a in enumerate(anchors):
        # Smoothing is allowed to nudge the values slightly.
        assert abs(a.cx - expected_cx[i]) < 0.05, (
            f"anchor[{i}] cx={a.cx}, expected ~{expected_cx[i]}"
        )
        # Each anchor should carry the person_body confidence (smoother
        # leaves confidence untouched).
        assert abs(a.confidence - PERSON_BODY_CONFIDENCE) < 1e-6

    # cy should be biased upward inside the bbox: cy_bias = 0.6 - 0.35 * 0.50 = 0.425
    expected_cy = 0.6 - PERSON_HEAD_CY_BIAS * 0.50
    for a in anchors:
        # Smoother averages cy too, but with all frames at the same cy
        # the smoother is a no-op.
        assert abs(a.cy - expected_cy) < 1e-6, (
            f"cy={a.cy} expected {expected_cy} (biased upward by 0.35*height)"
        )


def test_face_anchor_wins_over_person_body_when_face_present():
    # When a face IS detected, the face anchor should win regardless of
    # what person_detector reported. Person body shouldn't double-anchor.
    frames = [_FrameFaces(timestamp=0.0, faces=[_Face(nose_x=40.0, nose_y=30.0)])]
    persons = [
        _PersonRegion(
            timestamp=0.0, cx=0.3, cy=0.6, width=0.20, height=0.50,
            has_face=True,
        ),
    ]
    anchors = build_attention_anchors(
        frame_faces=frames, frame_persons=persons,
    )
    assert len(anchors) == 1
    assert anchors[0].source in ("face", "active_face")


def test_person_with_face_inside_is_skipped():
    # When has_face=True, the person bbox should NOT generate an anchor
    # even on a faceless frame (the upstream face for that bbox lives
    # in a different frame slot, not relevant here — the flag is the
    # contract).
    frames = [_FrameFaces(timestamp=0.0, faces=[])]
    persons = [
        _PersonRegion(
            timestamp=0.0, cx=0.3, cy=0.6, width=0.20, height=0.50,
            has_face=True,  # ineligible
        ),
    ]
    anchors = build_attention_anchors(frame_faces=frames, frame_persons=persons)
    assert len(anchors) == 1
    # Falls all the way down the chain to motion_centroid (no face,
    # no eligible person, no saliency).
    assert anchors[0].source == "motion_centroid"
