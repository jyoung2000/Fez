"""Test that feeding a flat list of SaliencyRegion objects into
build_required_regions (instead of FrameSaliency-with-blobs) is
auto-converted and produces non-zero saliency regions.

Regression for the K S01E12 run where the log showed
`RequiredRegions: 3017 frames, 1110 with data, 1795 total (face=1795 obj=0 sal=0)` —
saliency_tracker produced 766 regions but none of them survived the
per-frame loop because `getattr(sal_frame, 'blobs', [])` returned [].
"""

from dataclasses import dataclass, field

from backend.services.content_classifier import ClipContentType
from backend.services.required_regions import build_required_regions
from backend.services.saliency_tracker import SaliencyRegion


@dataclass
class _Face:
    nose_x: float = 50.0
    nose_y: float = 50.0
    width: float = 10.0
    height: float = 12.0
    identity_id: int = 0
    lip_aperture: float = 0.0
    is_human: bool = True
    is_speaking: bool = False


@dataclass
class _FrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)


def test_saliency_region_flat_list_converts_and_produces_regions():
    # Frames without faces — we want the saliency to fire.
    frames = [_FrameFaces(timestamp=float(t), faces=[]) for t in range(5)]

    # Saliency blobs at the same timestamps, center at 70% width,
    # width 20%, height 25%, score 0.6 — well above the GAMEPLAY
    # min_area of 0.02 (0.2 × 0.25 = 0.05).
    sal_regions = [
        SaliencyRegion(
            timestamp=float(t),
            x=70.0, y=50.0, w=20.0, h=25.0,
            saliency_score=0.6,
            motion_score=0.3,
            spatial_score=0.3,
        )
        for t in range(5)
    ]

    regs_per_frame = build_required_regions(
        frame_faces=frames,
        frame_saliency=sal_regions,
        content_type=ClipContentType.GAMEPLAY,
    )

    assert len(regs_per_frame) == 5
    sal_count = sum(1 for frs in regs_per_frame for r in frs if r.source == "saliency")
    assert sal_count >= 1, (
        f"Expected at least one saliency region to be generated after "
        f"the SaliencyRegion→FrameSaliency conversion fallback; got 0 "
        f"(regions_per_frame={regs_per_frame})"
    )


def test_frame_saliency_with_blobs_still_works():
    """Regression guard: the legacy path (FrameSaliency-like object with
    .blobs already present) must continue to work unchanged."""

    @dataclass
    class _FrameSal:
        timestamp: float
        mean_score: float
        blobs: list = field(default_factory=list)

    frames = [_FrameFaces(timestamp=1.0, faces=[])]
    fs = [_FrameSal(
        timestamp=1.0, mean_score=0.6,
        blobs=[(0.60, 0.30, 0.25, 0.30)],  # area 0.075
    )]
    regs_per_frame = build_required_regions(
        frame_faces=frames,
        frame_saliency=fs,
        content_type=ClipContentType.GAMEPLAY,
    )
    sal_count = sum(1 for frs in regs_per_frame for r in frs if r.source == "saliency")
    assert sal_count >= 1
