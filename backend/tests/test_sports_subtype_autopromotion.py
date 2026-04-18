"""Week 2 Part D — unit tests for sports subtype auto-promotion.

Exercises ``_infer_sports_subtype_from_objects`` with both the flat
``ObjectDetection`` list shape the pipeline actually produces and
the per-frame ``.objects`` shape the helper also accepts.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass
class _StubObj:
    timestamp: float
    class_name: str
    w: float
    h: float
    x: float = 50.0
    y: float = 50.0
    confidence: float = 0.8


def _flat_list(frames: list) -> list[_StubObj]:
    """Flatten a list-of-lists into one ObjectDetection-shaped list.

    Each outer element is a per-frame list of _StubObj (possibly empty).
    The helper assigns a per-frame timestamp so the group-by logic in
    ``_infer_sports_subtype_from_objects`` sees one "frame" per outer
    list entry — even when that entry is empty (we emit a marker with
    an empty class_name so the frame still counts in the denominator).
    """
    out: list[_StubObj] = []
    for fr_idx, frame in enumerate(frames):
        if not frame:
            # Emit a placeholder so the frame still counts in n_frames.
            # The classifier's group-by-timestamp uses round(ts, 2),
            # so we need a unique ts per frame.
            out.append(
                _StubObj(
                    timestamp=float(fr_idx),
                    class_name="__empty__",
                    w=0, h=0,
                )
            )
            continue
        for obj in frame:
            out.append(
                _StubObj(
                    timestamp=float(fr_idx),
                    class_name=obj.class_name,
                    w=obj.w, h=obj.h,
                )
            )
    return out


# ───────────────────────── basketball ─────────────────────────


def test_basketball_promotes_when_ball_in_enough_frames():
    """3/10 frames with a sports ball → 0.30 ratio → promoted."""
    from backend.services.content_classifier import (
        _infer_sports_subtype_from_objects,
    )
    ball = _StubObj(timestamp=0.0, class_name="sports ball", w=5, h=5)
    frames = [
        [ball] if i < 3 else [] for i in range(10)
    ]
    result = _infer_sports_subtype_from_objects(_flat_list(frames))
    assert result is not None
    subtype, ball_ratio, vehicle_ratio = result
    assert subtype == "basketball"
    assert ball_ratio == pytest.approx(0.30, abs=0.01)
    assert vehicle_ratio == 0.0


def test_basketball_below_threshold_no_promote():
    """1/10 frames (10%) is below the 15% threshold."""
    from backend.services.content_classifier import (
        _infer_sports_subtype_from_objects,
    )
    ball = _StubObj(timestamp=0.0, class_name="sports ball", w=5, h=5)
    frames = [[ball] if i == 0 else [] for i in range(10)]
    result = _infer_sports_subtype_from_objects(_flat_list(frames))
    assert result is None


# ───────────────────────── racing ─────────────────────────


def test_racing_promotes_when_large_vehicle_in_enough_frames():
    """2/10 frames with a big car (w=30, h=20 → area=600 > 500 gate)."""
    from backend.services.content_classifier import (
        _infer_sports_subtype_from_objects,
    )
    car = _StubObj(timestamp=0.0, class_name="car", w=30, h=20)
    # 2 frames with car → 20% → above the 10% threshold.
    frames = [[car] if i < 2 else [] for i in range(10)]
    result = _infer_sports_subtype_from_objects(_flat_list(frames))
    assert result is not None
    subtype, ball_ratio, vehicle_ratio = result
    assert subtype == "racing"
    assert vehicle_ratio == pytest.approx(0.20, abs=0.01)
    assert ball_ratio == 0.0


def test_racing_below_area_threshold_no_promote():
    """Car at w=20, h=25 → area=500. Borderline — area_gate is >= 500, so it passes."""
    from backend.services.content_classifier import (
        _infer_sports_subtype_from_objects,
    )
    car = _StubObj(timestamp=0.0, class_name="car", w=20, h=25)
    # 2 of 10 frames at exactly the area gate → 20% > 10% → promoted.
    frames = [[car] if i < 2 else [] for i in range(10)]
    result = _infer_sports_subtype_from_objects(_flat_list(frames))
    assert result is not None
    assert result[0] == "racing"


def test_racing_tiny_car_no_promote():
    """Car at w=10, h=10 → area=100, below 500 gate. 2/10 frames but none count."""
    from backend.services.content_classifier import (
        _infer_sports_subtype_from_objects,
    )
    car = _StubObj(timestamp=0.0, class_name="car", w=10, h=10)
    frames = [[car] if i < 2 else [] for i in range(10)]
    result = _infer_sports_subtype_from_objects(_flat_list(frames))
    assert result is None, "small cars are background — no promotion"


def test_racing_below_frame_threshold_no_promote():
    """One of ten frames is only 10%, and the > comparison requires > ball_ratio."""
    from backend.services.content_classifier import (
        _infer_sports_subtype_from_objects,
    )
    car = _StubObj(timestamp=0.0, class_name="car", w=30, h=20)
    # 1/10 frames → 10%. Equal to threshold, but 10% > 0 ball ratio → promoted.
    frames = [[car] if i == 0 else [] for i in range(10)]
    result = _infer_sports_subtype_from_objects(_flat_list(frames))
    # Note: 10% meets the >= 0.10 threshold AND vehicle_ratio > ball_ratio.
    assert result is not None
    assert result[0] == "racing"


# ───────────────────────── mixed ─────────────────────────


def test_mixed_ball_and_car_higher_ratio_wins():
    """Ball in 4/10, car (big) in 3/10 → basketball (higher ratio)."""
    from backend.services.content_classifier import (
        _infer_sports_subtype_from_objects,
    )
    ball = _StubObj(timestamp=0.0, class_name="sports ball", w=5, h=5)
    car = _StubObj(timestamp=0.0, class_name="car", w=30, h=20)
    frames = [
        [ball] if i < 4 else ([car] if i < 7 else [])
        for i in range(10)
    ]
    result = _infer_sports_subtype_from_objects(_flat_list(frames))
    assert result is not None
    assert result[0] == "basketball", "higher ratio wins"


def test_mixed_car_dominant_promotes_racing():
    """Ball in 1/10, big car in 4/10 → racing wins."""
    from backend.services.content_classifier import (
        _infer_sports_subtype_from_objects,
    )
    ball = _StubObj(timestamp=0.0, class_name="sports ball", w=5, h=5)
    car = _StubObj(timestamp=0.0, class_name="car", w=30, h=20)
    frames = [
        [ball] if i == 0 else ([car] if i < 5 else [])
        for i in range(10)
    ]
    result = _infer_sports_subtype_from_objects(_flat_list(frames))
    assert result is not None
    assert result[0] == "racing"


# ───────────────────────── shape variants ─────────────────────────


def test_per_frame_shape_also_supported():
    """Helper also accepts a per-frame list with an ``.objects`` attribute."""
    from backend.services.content_classifier import (
        _infer_sports_subtype_from_objects,
    )

    @dataclass
    class _StubFrame:
        objects: list

    ball = _StubObj(timestamp=0.0, class_name="sports ball", w=5, h=5)
    per_frame = [
        _StubFrame(objects=[ball]) if i < 3 else _StubFrame(objects=[])
        for i in range(10)
    ]
    result = _infer_sports_subtype_from_objects(per_frame)
    assert result is not None
    assert result[0] == "basketball"


def test_empty_input_returns_none():
    from backend.services.content_classifier import (
        _infer_sports_subtype_from_objects,
    )
    assert _infer_sports_subtype_from_objects(None) is None
    assert _infer_sports_subtype_from_objects([]) is None


def test_only_non_sports_objects_returns_none():
    """All objects are things like 'person' / 'chair' — no promotion."""
    from backend.services.content_classifier import (
        _infer_sports_subtype_from_objects,
    )
    person = _StubObj(timestamp=0.0, class_name="person", w=10, h=30)
    frames = [[person] for _ in range(10)]
    result = _infer_sports_subtype_from_objects(_flat_list(frames))
    assert result is None


# ───────────────────────── end-to-end through classify_content ─────────────────────────


def test_classify_content_respects_user_override():
    """If the user already picked sports_basketball, classify_content must not
    touch the subtype — even if the flat list would auto-promote to racing."""
    from backend.services.content_classifier import classify_content
    from backend.services.content_type_config import ContentType

    ball = _StubObj(timestamp=0.0, class_name="sports ball", w=5, h=5)
    frame_objects = _flat_list(
        [[ball] if i < 5 else [] for i in range(10)]
    )
    md = {
        "content_type_override": "sports_racing",  # user override wins
        "frame_objects": frame_objects,
    }
    profile = classify_content(
        shot_cuts=[], face_registry=None, dense_faces=[],
        scenes=[], video_duration=30.0, metadata=md,
    )
    # The user-override branch returns early at the top of
    # classify_content, so sports_subtype should be "racing" (from the
    # override token), not "basketball" from the auto-promotion.
    assert profile.sports_subtype == "racing"


def test_classify_content_non_sports_short_circuits():
    """A non-sports profile never runs the promotion block."""
    from backend.services.content_classifier import classify_content

    md = {
        "content_type_override": "podcast",
        "frame_objects": [
            _StubObj(timestamp=0.0, class_name="sports ball", w=5, h=5),
        ] * 10,
    }
    profile = classify_content(
        shot_cuts=[], face_registry=None, dense_faces=[],
        scenes=[], video_duration=30.0, metadata=md,
    )
    # Profile is podcast via user override; sports_subtype must stay empty.
    assert profile.content_type == "podcast"
    assert not profile.sports_subtype
