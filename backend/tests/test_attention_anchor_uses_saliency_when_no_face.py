"""Test that AttentionAnchor falls back to the saliency peak when no
face is detected anywhere in the timeline.

Synthetic: no faces, a single saliency region at cx≈0.8 with score 0.6.
The resulting anchor for that frame should have source="saliency_peak",
cx≈0.8, confidence≈0.3 + 0.5·0.6 = 0.6.
"""

from dataclasses import dataclass, field

from backend.services.attention_anchor import build_attention_anchors


@dataclass
class _FrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)


@dataclass
class _FrameSaliency:
    timestamp: float
    mean_score: float
    blobs: list = field(default_factory=list)


def test_saliency_peak_picked_when_no_faces():
    frames = [_FrameFaces(timestamp=1.0, faces=[])]
    # Blob at (0.7, 0.4, 0.2, 0.2) → center (0.8, 0.5), area 0.04
    fs = [_FrameSaliency(
        timestamp=1.0,
        mean_score=0.6,
        blobs=[(0.70, 0.40, 0.20, 0.20)],
    )]

    anchors = build_attention_anchors(
        frame_faces=frames,
        frame_saliency=fs,
        shot_cuts=[],
    )
    assert len(anchors) == 1
    a = anchors[0]
    assert a.source == "saliency_peak", f"expected saliency_peak, got {a.source}"
    assert abs(a.cx - 0.80) < 1e-6
    assert abs(a.confidence - 0.60) < 1e-6


def test_saliency_peak_ignored_when_score_below_threshold():
    """A blob with score 0.1 is below SALIENCY_PEAK_MIN_SCORE (0.25) and
    must be rejected; the anchor falls through to motion_centroid."""
    frames = [_FrameFaces(timestamp=1.0, faces=[])]
    fs = [_FrameSaliency(
        timestamp=1.0,
        mean_score=0.1,
        blobs=[(0.70, 0.40, 0.20, 0.20)],
    )]

    anchors = build_attention_anchors(
        frame_faces=frames,
        frame_saliency=fs,
        shot_cuts=[],
    )
    assert len(anchors) == 1
    assert anchors[0].source == "motion_centroid"


def test_saliency_peak_picks_largest_score_area_product():
    """When multiple blobs exist, the peak is picked by score·area, not
    score alone. A big blob with modest score beats a tiny corner blob
    with a higher score."""
    frames = [_FrameFaces(timestamp=1.0, faces=[])]
    # Big blob — area 0.2·0.2 = 0.04, score 0.6 → rank 0.024
    # Tiny blob — area 0.03·0.03 = 0.0009, score 0.9 → rank ~0.0008
    fs = [_FrameSaliency(
        timestamp=1.0,
        mean_score=0.6,
        blobs=[
            (0.60, 0.40, 0.20, 0.20),  # big, center (0.7, 0.5)
            (0.05, 0.05, 0.03, 0.03),  # tiny corner, center ~(0.065, 0.065)
        ],
    )]

    anchors = build_attention_anchors(
        frame_faces=frames,
        frame_saliency=fs,
        shot_cuts=[],
    )
    a = anchors[0]
    assert a.source == "saliency_peak"
    # Expect the big blob to win — cx near 0.7, not 0.065
    assert abs(a.cx - 0.70) < 0.02, (
        f"expected cx≈0.70 from the big blob, got {a.cx}"
    )
