"""Test that the AttentionAnchor boxcar smoother breaks at shot cuts.

Synthetic: shot 1 has anchors at cx=0.2 (ending at t=1.0), shot 2 has
anchors at cx=0.8 (starting at t=1.01 with a shot cut at t=1.005).
Post-smoother, the cx at the end of shot 1 should remain ~0.2 — NOT
get pulled toward 0.8 by the smoother averaging across the cut.
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


def test_smoother_breaks_at_shot_cut():
    # Shot 1: faces at cx=0.2 (nose_x=20) at t=0.8, 0.9, 1.0
    # Shot cut: t=1.005
    # Shot 2: faces at cx=0.8 (nose_x=80) at t=1.01, 1.1, 1.2
    frames = [
        _FrameFaces(timestamp=0.8, faces=[_Face(nose_x=20.0)]),
        _FrameFaces(timestamp=0.9, faces=[_Face(nose_x=20.0)]),
        _FrameFaces(timestamp=1.0, faces=[_Face(nose_x=20.0)]),
        _FrameFaces(timestamp=1.01, faces=[_Face(nose_x=80.0)]),
        _FrameFaces(timestamp=1.1, faces=[_Face(nose_x=80.0)]),
        _FrameFaces(timestamp=1.2, faces=[_Face(nose_x=80.0)]),
    ]

    anchors = build_attention_anchors(
        frame_faces=frames,
        shot_cuts=[1.005],
    )
    assert len(anchors) == 6

    # Last anchor of shot 1 (t=1.0) must stay near 0.2, not be dragged
    # to the midpoint 0.5 by the smoother.
    shot1_end = next(a for a in anchors if abs(a.timestamp - 1.0) < 1e-6)
    assert shot1_end.cx < 0.30, (
        f"shot1 end anchor should not be smoothed across cut; got "
        f"cx={shot1_end.cx:.3f}"
    )

    # First anchor of shot 2 (t=1.01) must stay near 0.8.
    shot2_start = next(a for a in anchors if abs(a.timestamp - 1.01) < 1e-6)
    assert shot2_start.cx > 0.70, (
        f"shot2 start anchor should not be smoothed across cut; got "
        f"cx={shot2_start.cx:.3f}"
    )


def test_smoother_smooths_within_shot():
    """Control: within a single shot (no cuts), the smoother should
    actually smooth — 5 cx values close together stay close together,
    not get wildly rearranged."""
    frames = [
        _FrameFaces(timestamp=0.1, faces=[_Face(nose_x=30.0)]),
        _FrameFaces(timestamp=0.2, faces=[_Face(nose_x=32.0)]),
        _FrameFaces(timestamp=0.3, faces=[_Face(nose_x=28.0)]),
        _FrameFaces(timestamp=0.4, faces=[_Face(nose_x=31.0)]),
        _FrameFaces(timestamp=0.5, faces=[_Face(nose_x=29.0)]),
    ]
    anchors = build_attention_anchors(frame_faces=frames, shot_cuts=[])
    assert len(anchors) == 5
    for a in anchors:
        # Smoothed values should all sit in [0.28, 0.32] — the local mean
        assert 0.27 <= a.cx <= 0.33, f"expected smoothed cx near 0.30, got {a.cx}"
