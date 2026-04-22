"""Blueprint v2 Phase 1 — parity guard for the importance-matrix migration.

The required_regions.py and genre_refinements.py callsites now read
their weights from ``ReframeConfig.importance`` instead of magic
numbers. This test pins the combined numerical output to match the
pre-Phase-1 hardcoded values so the migration cannot silently shift
the solver's data term.

These tests are cheap — they don't run the full render pipeline.
They exercise the two migration points directly.
"""

from dataclasses import dataclass, field

import pytest

from backend.services.genre_refinements import (
    BASKETBALL_BALL_BOOST,
    RACING_CAR_BOOST,
    _basketball_refinements,
    _racing_refinements,
)
from backend.services.reframe_config import get_default_config


# ── Genre refinements: ball / car boost parity ────────────────────


def test_basketball_ball_boost_preserves_legacy_1_1_weight():
    """Phase 1 moved the hardcoded ``weight=1.1`` to a ratio over
    ``config.importance.object``. Basketball has object=1.0 in the
    matrix, so the effective weight must still be exactly 1.1."""
    cfg = get_default_config().for_content("sports_basketball")
    dets = [{"t": i * 0.1, "confidence": 0.8} for i in range(10)]
    result = _basketball_refinements(
        ball_detections=dets, shot_boundaries=[], config=cfg,
    )
    assert result.required_boosts[0].weight == pytest.approx(1.1)
    # Sanity: BASKETBALL_BALL_BOOST * importance.object must equal legacy.
    assert BASKETBALL_BALL_BOOST * cfg.importance.object == pytest.approx(1.1)


def test_racing_car_boost_preserves_legacy_1_2_weight():
    cfg = get_default_config().for_content("sports_racing")
    dets = [{"t": 0, "bbox": [0.3, 0.4, 0.1, 0.1]}]
    result = _racing_refinements(car_detections=dets, config=cfg)
    assert result.required_boosts[0].weight == pytest.approx(1.2)
    assert RACING_CAR_BOOST * cfg.importance.object == pytest.approx(1.2)


def test_basketball_empty_detections_short_circuits():
    """Guards against regressions in the empty-list fast path."""
    cfg = get_default_config().for_content("sports_basketball")
    result = _basketball_refinements(
        ball_detections=[], shot_boundaries=[], config=cfg,
    )
    assert result.required_boosts == []


# ── required_regions passive-face score parity ────────────────────


@dataclass
class _Face:
    nose_x: float = 50.0
    nose_y: float = 50.0
    width: float = 10.0
    height: float = 14.0
    identity_id: int = 0
    lip_aperture: float = 0.0
    is_human: bool = True
    is_speaking: bool = False
    yaw: float = 0.0


@dataclass
class _FrameFaces:
    timestamp: float = 0.0
    faces: list = field(default_factory=list)


@dataclass
class _SpeakerEvent:
    start: float
    end: float
    slot_id: int
    on_screen: bool = True


def test_passive_face_score_defaults_to_legacy_0_55():
    """Default generic config must preserve the pre-Phase-1 score."""
    cfg = get_default_config()
    assert cfg.importance.passive_face_weight == pytest.approx(0.55)


def test_passive_face_score_used_in_required_regions():
    """When slot 1 is the active speaker and slot 0 is in-frame, slot 0
    is passive-same-shot and must receive ``score == passive_face_weight``."""
    from backend.services.required_regions import build_required_regions

    frame = _FrameFaces(
        timestamp=0.0,
        faces=[
            _Face(nose_x=30.0, identity_id=0),
            _Face(nose_x=70.0, identity_id=1),
        ],
    )
    events = [_SpeakerEvent(start=-1.0, end=10.0, slot_id=1)]

    regs = build_required_regions(
        frame_faces=[frame], active_speaker_events=events,
        content_type="talking_head",
    )
    assert regs and regs[0]

    # Find slot 0's face region (not containment).
    passive = [
        r for r in regs[0]
        if r.source == "face" and r.face_slot == 0
    ]
    assert passive, "expected a passive face region for slot 0"
    assert passive[0].score == pytest.approx(0.55)

    # Active speaker gets score 1.0.
    active = [
        r for r in regs[0]
        if r.source == "face" and r.face_slot == 1 and r.is_active_speaker
    ]
    assert active
    assert active[0].score == pytest.approx(1.0)


def test_passive_face_score_honors_custom_config():
    """When the content-override table contains a custom
    ``importance`` row, that row's ``passive_face_weight`` flows into
    every passive-same-shot face region."""
    from backend.services.reframe_config import ImportanceWeights
    from backend.services.required_regions import build_required_regions

    frame = _FrameFaces(
        timestamp=0.0,
        faces=[
            _Face(nose_x=30.0, identity_id=0),
            _Face(nose_x=70.0, identity_id=1),
        ],
    )
    events = [_SpeakerEvent(start=-1.0, end=10.0, slot_id=1)]

    base = get_default_config()
    overrides = dict(base.content_overrides)
    overrides["generic"] = {
        **(overrides.get("generic") or {}),
        "importance": ImportanceWeights(
            face=1.0, saliency=0.4, motion=0.2,
            passive_face_weight=0.30,
        ),
    }
    custom_cfg = base.override(content_overrides=overrides)

    regs = build_required_regions(
        frame_faces=[frame], active_speaker_events=events,
        content_type="generic",
        config=custom_cfg,
    )
    passive = [
        r for r in regs[0]
        if r.source == "face" and r.face_slot == 0
    ]
    assert passive[0].score == pytest.approx(0.30)


def test_no_active_speaker_falls_back_to_0_8():
    """Frames with no active speaker still use the legacy 0.8 score
    (Phase 1 didn't migrate this constant yet — parity guard so we
    don't accidentally break it later)."""
    from backend.services.required_regions import build_required_regions

    frame = _FrameFaces(
        timestamp=0.0,
        faces=[_Face(nose_x=50.0, identity_id=0)],
    )
    regs = build_required_regions(
        frame_faces=[frame], active_speaker_events=[],
        content_type="generic",
    )
    faces = [r for r in regs[0] if r.source == "face"]
    assert faces[0].score == pytest.approx(0.8)
