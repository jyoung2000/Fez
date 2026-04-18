"""Phase B — pose-derived head anchor for PersonRegion.

Tests the keypoint→anchor decision logic in person_detector and the
end-to-end consumption in attention_anchor._person_body_anchor.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

# Pre-mock heavy native deps.
for _mod in (
    "google.generativeai", "google.generativeai.types", "google.ai",
    "google.ai.generativelanguage_v1beta", "anthropic", "groq", "httpx",
    "openai", "ctranslate2", "faster_whisper", "mediapipe",
):
    sys.modules.setdefault(_mod, MagicMock())


# ── Keypoint priority logic ──

def _kpts_with(nose_conf, ls_conf, rs_conf):
    """Build a 17-keypoint list with the three relevant slots set."""
    kp = [(0.0, 0.0, 0.0)] * 17
    kp[0] = (0.5, 0.2, nose_conf)   # nose
    kp[5] = (0.4, 0.4, ls_conf)     # left shoulder
    kp[6] = (0.6, 0.4, rs_conf)     # right shoulder
    return kp


def test_anchor_from_keypoints_prefers_nose():
    from backend.services.person_detector import _anchor_from_keypoints
    kpts = _kpts_with(nose_conf=0.9, ls_conf=0.1, rs_conf=0.1)
    bbox = (0.3, 0.1, 0.7, 0.9)
    ax, ay, src = _anchor_from_keypoints(kpts, bbox)
    assert src == "nose"
    assert ax == pytest.approx(0.5)
    assert ay == pytest.approx(0.2)


def test_anchor_from_keypoints_falls_back_to_shoulders():
    from backend.services.person_detector import _anchor_from_keypoints
    kpts = _kpts_with(nose_conf=0.1, ls_conf=0.8, rs_conf=0.85)
    bbox = (0.3, 0.1, 0.7, 0.9)
    ax, ay, src = _anchor_from_keypoints(kpts, bbox)
    assert src == "shoulders"
    assert ax == pytest.approx(0.5)
    assert ay == pytest.approx(0.4)


def test_anchor_from_keypoints_back_turned():
    from backend.services.person_detector import _anchor_from_keypoints
    # All three keypoints are low confidence — back-turned subject.
    kpts = _kpts_with(nose_conf=0.1, ls_conf=0.2, rs_conf=0.1)
    bbox = (0.3, 0.1, 0.7, 0.9)
    ax, ay, src = _anchor_from_keypoints(kpts, bbox)
    assert src == "bbox_top"
    # x = bbox center
    assert ax == pytest.approx(0.5)
    # y in upper third of the bbox
    upper_third_top = 0.1
    upper_third_bottom = 0.1 + (0.9 - 0.1) / 3
    assert upper_third_top <= ay <= upper_third_bottom


# ── attention_anchor consumes the pose anchor ──

def test_attention_anchor_uses_pose_anchor():
    """When PersonRegion has anchor_x/anchor_y populated, the resulting
    AttentionAnchor's cx/cy match the pose anchor — NOT the bbox center."""
    from backend.services.person_detector import PersonRegion
    from backend.services.attention_anchor import _person_body_anchor

    person = PersonRegion(
        timestamp=1.0,
        cx=0.5, cy=0.6,           # bbox center near middle
        width=0.3, height=0.7,
        confidence=0.9,
        has_face=False,
        anchor_x=0.42,
        anchor_y=0.18,
        anchor_source="nose",
    )
    out = _person_body_anchor([person], timestamp=1.0)
    assert out is not None
    assert out.cx == pytest.approx(0.42)
    assert out.cy == pytest.approx(0.18)


def test_attention_anchor_falls_back_to_bbox_center_without_pose():
    """When anchor_x/anchor_y are None (default), the consumer falls
    back to the head-biased bbox center — preserves pre-Phase-B
    behaviour for the CLIPAI_PERSON_USE_POSE=false default."""
    from backend.services.person_detector import PersonRegion
    from backend.services.attention_anchor import (
        _person_body_anchor, PERSON_HEAD_CY_BIAS,
    )

    person = PersonRegion(
        timestamp=1.0,
        cx=0.5, cy=0.6,
        width=0.3, height=0.7,
        confidence=0.9,
        has_face=False,
    )
    out = _person_body_anchor([person], timestamp=1.0)
    assert out is not None
    assert out.cx == pytest.approx(0.5)
    expected_cy = max(0.0, 0.6 - PERSON_HEAD_CY_BIAS * 0.7)
    assert out.cy == pytest.approx(expected_cy)
