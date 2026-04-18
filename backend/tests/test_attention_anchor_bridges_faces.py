"""Test that AttentionAnchor linearly bridges between two face anchors
when no face is detected in the intervening frames.

Synthetic: face at t=0 (cx=0.3), face at t=2.0 (cx=0.7), no face in
between. Expect anchors at t=0.5, 1.0, 1.5 to exist with monotonically
changing cx and source="last_face_decay".
"""

from dataclasses import dataclass, field

from backend.services.attention_anchor import build_attention_anchors


@dataclass
class _Face:
    nose_x: float
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


def test_face_to_face_bridge_is_linear():
    frames = [
        _FrameFaces(timestamp=0.0, faces=[_Face(nose_x=30.0)]),
        _FrameFaces(timestamp=0.5, faces=[]),
        _FrameFaces(timestamp=1.0, faces=[]),
        _FrameFaces(timestamp=1.5, faces=[]),
        _FrameFaces(timestamp=2.0, faces=[_Face(nose_x=70.0)]),
    ]

    anchors = build_attention_anchors(frame_faces=frames, shot_cuts=[])
    assert len(anchors) == 5

    # Endpoints come from real faces (source=face or active_face)
    assert anchors[0].source in ("face", "active_face")
    assert abs(anchors[0].cx - 0.30) < 0.02
    assert anchors[-1].source in ("face", "active_face")
    assert abs(anchors[-1].cx - 0.70) < 0.02

    # Middle frames bridged.
    for i in (1, 2, 3):
        a = anchors[i]
        assert a.source == "last_face_decay", (
            f"anchor[{i}] should be bridged, got source={a.source}"
        )
        assert a.confidence > 0.3

    # Monotonically increasing cx (0.3 → 0.4 → 0.5 → 0.6 → 0.7 after smoothing).
    cxs = [a.cx for a in anchors]
    for i in range(1, len(cxs)):
        assert cxs[i] >= cxs[i - 1] - 1e-6, (
            f"cx should be monotone non-decreasing, got {cxs}"
        )
