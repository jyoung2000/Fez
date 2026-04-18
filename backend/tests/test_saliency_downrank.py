"""Tests for the content-aware saliency downrank in required_regions.py.

In CINEMATIC_DIALOGUE mode, saliency regions whose center lies outside
the union of face bboxes (expanded by 1.5x) must be dropped. In GAMEPLAY
mode, the same saliency region must be kept.
"""

from dataclasses import dataclass, field

from backend.services.content_classifier import ClipContentType
from backend.services.required_regions import build_required_regions


@dataclass
class _Face:
    nose_x: float
    nose_y: float
    width: float
    height: float
    lip_aperture: float = 0.0
    identity_id: int = 0
    is_human: bool = True
    is_speaking: bool = False


@dataclass
class _FrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)


@dataclass
class _FrameSaliency:
    """Frame-level saliency object: has blobs + mean_score, matching what
    required_regions.build_required_regions expects."""
    timestamp: float
    mean_score: float
    blobs: list = field(default_factory=list)  # list of (x, y, w, h) in [0,1]


def _make_test_frames():
    """One face at center + one saliency blob at the right edge."""
    # Face at frame center: nose at (50, 50), width=20, height=25.
    # After /100: center (0.5, 0.5), half-width 0.1, half-height 0.125.
    # Expanded 1.5x => [0.35..0.65] x [0.3125..0.6875].
    ff = _FrameFaces(timestamp=1.0, faces=[
        _Face(nose_x=50, nose_y=50, width=20, height=25, identity_id=0),
    ])
    # Saliency blob at the right edge: center at (0.85, 0.5) — well outside
    # the 1.5x face expansion.
    sal = _FrameSaliency(
        timestamp=1.0,
        mean_score=0.8,
        blobs=[(0.80, 0.40, 0.10, 0.20)],  # area=0.02, min_area=0.04 CINEMATIC
    )
    # Use a larger blob so it passes the min-area gate in all modes.
    sal.blobs = [(0.78, 0.38, 0.14, 0.24)]  # area = 0.0336 — fails CINEMATIC gate (0.04)
    sal.blobs = [(0.70, 0.30, 0.22, 0.30)]  # area = 0.066 — passes both gates
    return [ff], [sal]


class TestSaliencyDownrankCinematicDialogue:
    def test_edge_saliency_dropped_in_cinematic_dialogue(self):
        frames, saliency = _make_test_frames()
        regions_per_frame = build_required_regions(
            frame_faces=frames,
            active_speaker_events=[],
            frame_saliency=saliency,
            content_type=ClipContentType.CINEMATIC_DIALOGUE,
        )
        assert len(regions_per_frame) == 1
        regs = regions_per_frame[0]
        # The face must still be present.
        assert any(r.source == "face" for r in regs)
        # The edge-of-frame saliency region must have been dropped.
        sal_regions = [r for r in regs if r.source == "saliency"]
        assert sal_regions == [], (
            f"Edge saliency should be dropped in dialogue mode, got: {sal_regions}"
        )

    def test_edge_saliency_kept_in_gameplay(self):
        frames, saliency = _make_test_frames()
        regions_per_frame = build_required_regions(
            frame_faces=frames,
            active_speaker_events=[],
            frame_saliency=saliency,
            content_type=ClipContentType.GAMEPLAY,
        )
        assert len(regions_per_frame) == 1
        regs = regions_per_frame[0]
        sal_regions = [r for r in regs if r.source == "saliency"]
        assert sal_regions, (
            "Edge saliency should survive in GAMEPLAY where saliency is "
            "load-bearing, but got: " + repr(regs)
        )
