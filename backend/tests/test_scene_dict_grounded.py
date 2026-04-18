"""Phase 1 acceptance tests — grounding-output parse_scene_dict helper.

Back-compat contract checked end to end:
  - grounded bbox → derives subject_x / precise_x
  - legacy subject_x → synthesizes a centered strip + 0.5 confidence
  - no_subject_reason pathway
  - out-of-range box gets clamped (not rejected)
  - secondary_subjects capped at top-3
"""

import logging

from backend.models import SceneDescription
from backend.services.providers.base import (
    fill_box_from_legacy,
    parse_scene_dict,
)


def _make_scene(parsed: dict, thumbnail_path: str = "/tmp/t.jpg") -> SceneDescription:
    """Helper: merge parsed dict into a SceneDescription with a thumbnail."""
    return SceneDescription(thumbnail_path=thumbnail_path, **parsed)


def test_parse_scene_dict_grounded_box():
    """A grounded subject_box derives subject_x from its center."""
    d = {
        "timestamp": 1.0,
        "description": "woman at left-third of frame",
        "importance_score": 7,
        "subject_box": [0.25, 0.15, 0.35, 0.60],
        "subject_confidence": 0.85,
    }
    parsed = parse_scene_dict(d)
    assert parsed["subject_box"] == [0.25, 0.15, 0.35, 0.60]
    assert parsed["vlm_confidence"] == 0.85
    # Center at x=(0.25+0.35)/2 = 0.30 → 30%.
    assert abs(parsed["precise_x"] - 30.0) <= 1
    assert abs(parsed["subject_x"] - 30) <= 1
    assert parsed["no_subject_reason"] is None
    # SceneDescription construction should round-trip cleanly.
    scene = _make_scene(parsed)
    assert scene.subject_box == [0.25, 0.15, 0.35, 0.60]
    assert scene.vlm_confidence == 0.85
    assert 29 <= scene.subject_x <= 31


def test_parse_scene_dict_legacy_subject_x():
    """Only legacy subject_x → synthesized strip, confidence 0.5."""
    d = {
        "timestamp": 2.0,
        "description": "test",
        "importance_score": 5,
        "subject_x": 67,
    }
    parsed = parse_scene_dict(d)
    assert parsed["subject_x"] == 67
    assert abs(parsed["precise_x"] - 67.0) <= 1
    assert parsed["vlm_confidence"] == 0.5
    box = parsed["subject_box"]
    assert box is not None
    cx = (box[0] + box[2]) / 2.0
    assert abs(cx - 0.67) <= 0.01
    # 10% wide total (5% each side).
    assert abs((box[2] - box[0]) - 0.10) <= 0.001


def test_parse_scene_dict_no_subject_reason():
    """null box + no_subject_reason → centered default, zero VLM confidence."""
    d = {
        "timestamp": 3.0,
        "description": "abstract geometry",
        "importance_score": 4,
        "subject_box": None,
        "no_subject_reason": "abstract",
    }
    parsed = parse_scene_dict(d)
    assert parsed["subject_x"] == 50
    assert parsed["subject_box"] is None
    assert parsed["vlm_confidence"] == 0.0
    assert parsed["no_subject_reason"] == "abstract"
    scene = _make_scene(parsed)
    assert scene.no_subject_reason == "abstract"
    assert scene.vlm_confidence == 0.0


def test_parse_scene_dict_invalid_box_bounds(caplog):
    """Out-of-range box gets clamped and a warning is emitted."""
    d = {
        "timestamp": 4.0,
        "description": "edge case",
        "importance_score": 6,
        "subject_box": [-0.1, 0.5, 1.2, 0.8],
        "subject_confidence": 0.7,
    }
    with caplog.at_level(logging.WARNING, logger="backend.services.providers.base"):
        parsed = parse_scene_dict(d)
    assert parsed["subject_box"] == [0.0, 0.5, 1.0, 0.8]
    assert parsed["vlm_confidence"] == 0.7
    # Center of clamped x range = 0.5 → 50%.
    assert abs(parsed["subject_x"] - 50) <= 1
    assert any("clamped" in rec.message.lower() for rec in caplog.records)


def test_parse_scene_dict_secondary_subjects_capped():
    """5 secondary subjects → keep the top 3 by confidence, in order."""
    d = {
        "timestamp": 5.0,
        "description": "crowd",
        "importance_score": 7,
        "subject_box": [0.4, 0.2, 0.6, 0.8],
        "subject_confidence": 0.9,
        "secondary_subjects": [
            {"box": [0.1, 0.2, 0.2, 0.8], "confidence": 0.2, "label": "person"},
            {"box": [0.7, 0.2, 0.8, 0.8], "confidence": 0.9, "label": "person"},
            {"box": [0.2, 0.2, 0.3, 0.8], "confidence": 0.5, "label": "person"},
            {"box": [0.8, 0.2, 0.9, 0.8], "confidence": 0.75, "label": "object"},
            {"box": [0.3, 0.2, 0.4, 0.8], "confidence": 0.4, "label": "text"},
        ],
    }
    parsed = parse_scene_dict(d)
    secs = parsed["secondary_subjects"]
    assert len(secs) == 3
    confs = [s["confidence"] for s in secs]
    # Must be sorted descending and consist of the top three (0.9, 0.75, 0.5).
    assert confs == sorted(confs, reverse=True)
    assert confs == [0.9, 0.75, 0.5]


def test_parse_scene_dict_box_swapped_coords():
    """x1>x2 / y1>y2 gets swapped rather than rejected."""
    d = {
        "subject_box": [0.8, 0.9, 0.2, 0.3],
        "subject_confidence": 0.6,
    }
    parsed = parse_scene_dict(d, frame_timestamp=1.0)
    assert parsed["subject_box"] == [0.2, 0.3, 0.8, 0.9]


def test_parse_scene_dict_missing_box_and_legacy():
    """No subject_box and no subject_x → defaults to 50 and empty_frame reason."""
    d = {
        "timestamp": 6.0,
        "description": "",
        "importance_score": 5,
    }
    parsed = parse_scene_dict(d)
    assert parsed["subject_x"] == 50
    assert parsed["subject_box"] is None
    assert parsed["vlm_confidence"] == 0.0
    assert parsed["no_subject_reason"] == "empty_frame"


def test_fill_box_from_legacy_centered():
    """Legacy 50 → ~[0.45, 0.0, 0.55, 1.0]."""
    box = fill_box_from_legacy(50)
    assert abs(box[0] - 0.45) <= 0.001
    assert abs(box[2] - 0.55) <= 0.001
    assert box[1] == 0.0
    assert box[3] == 1.0


def test_fill_box_from_legacy_edge_clamped():
    """Legacy edge values clamp the strip at the frame boundary."""
    left = fill_box_from_legacy(2)
    assert left[0] == 0.0
    assert left[2] <= 0.08  # 0.02 + 0.05 = 0.07, minus floating slack
    right = fill_box_from_legacy(98)
    assert right[2] == 1.0
    assert right[0] >= 0.92


def test_scene_description_construction_back_compat():
    """SceneDescription still accepts a pure-legacy kwargs set."""
    scene = SceneDescription(
        timestamp=1.0,
        description="x",
        importance_score=5,
        thumbnail_path="/tmp/t.jpg",
        subject_x=42,
    )
    # New fields all default without breaking construction.
    assert scene.subject_box is None
    assert scene.vlm_confidence == 0.0
    assert scene.subject_confidence == 0.0
    assert scene.face_confidence == 0.0
    assert scene.secondary_subjects == []
    assert scene.no_subject_reason is None
    assert scene.fusion_source is None
    assert scene.subject_x == 42
