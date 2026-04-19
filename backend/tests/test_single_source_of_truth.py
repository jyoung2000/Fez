"""Fix 3.10: per-content knobs live in reframe_config._CONTENT_OVERRIDES.

Changing a fusion_aspect_max (or a saliency weight) in that one
table must propagate to every consumer — saliency_tracker,
subject_fusion — without touching those files.

Spec acceptance: mutating ``_CONTENT_OVERRIDES["animation"]["fusion_
aspect_max"]`` from 2.2 → 3.0 makes build_subject_tracks accept a
cluster with aspect=2.5 on an animation shot, with no other edits.
"""
import pytest

pytest.importorskip("cv2")

from dataclasses import dataclass, field

from backend.services import reframe_config as rc
from backend.services.subject_fusion import (
    _bounds_for_content,
    _promote_cluster,
)


@dataclass
class _SaliencyRegion:
    timestamp: float
    x: float
    y: float
    w: float
    h: float
    saliency_score: float = 0.5
    motion_score: float = 0.3
    spatial_score: float = 0.3


def _cluster(area_frac: float, aspect: float, n_frames: int, t0: float = 0.0):
    total_pct = (area_frac * 10000.0) ** 0.5
    w = total_pct / (aspect ** 0.5)
    h = aspect * w
    return [
        _SaliencyRegion(
            timestamp=t0 + i * 0.15,
            x=50.0, y=50.0, w=w, h=h,
        )
        for i in range(n_frames)
    ]


@pytest.fixture
def clean_config_cache():
    """Reset the DEFAULT_CONFIG singleton so _CONTENT_OVERRIDES edits
    propagate. Saves + restores per-test state."""
    saved = rc.DEFAULT_CONFIG
    saved_overrides = {
        k: dict(v) for k, v in rc._CONTENT_OVERRIDES.items()
    }
    rc.DEFAULT_CONFIG = None
    try:
        yield
    finally:
        rc.DEFAULT_CONFIG = saved
        rc._CONTENT_OVERRIDES.clear()
        rc._CONTENT_OVERRIDES.update(saved_overrides)


class TestSingleSourceOfTruth:
    def test_mutating_override_propagates_to_fusion_bounds(
        self, clean_config_cache,
    ):
        """Bump animation fusion_aspect_max from 2.2 → 3.0 in the
        config table. _bounds_for_content must reflect the change."""
        # Baseline: aspect 2.5 is REJECTED on animation.
        rc.DEFAULT_CONFIG = None
        ar_min, ar_max, _, _, _ = _bounds_for_content("animation")
        assert ar_max == 2.2

        # Mutate the single table.
        rc._CONTENT_OVERRIDES["animation"]["fusion_aspect_max"] = 3.0
        rc.DEFAULT_CONFIG = None  # force reload

        ar_min2, ar_max2, _, _, _ = _bounds_for_content("animation")
        assert ar_max2 == 3.0

    def test_mutating_override_propagates_to_promotion_gate(
        self, clean_config_cache,
    ):
        """End-to-end: cluster with aspect=2.5 on an animation shot
        is rejected by default (max=2.2) and accepted when max=3.0.
        No other file is touched."""
        cluster = _cluster(area_frac=0.05, aspect=2.5, n_frames=5)

        # Default: aspect_max=2.2, cluster rejected.
        rc.DEFAULT_CONFIG = None
        track = _promote_cluster(
            cluster, frame_lookup={}, source_width=1920, source_height=1080,
            track_id=0, job_id="", content_type="animation",
        )
        assert track is None, "baseline: 2.5 > 2.2, expected reject"

        # Mutate → cluster accepted.
        rc._CONTENT_OVERRIDES["animation"]["fusion_aspect_max"] = 3.0
        rc.DEFAULT_CONFIG = None
        track2 = _promote_cluster(
            cluster, frame_lookup={}, source_width=1920, source_height=1080,
            track_id=0, job_id="", content_type="animation",
        )
        assert track2 is not None, (
            "after bumping fusion_aspect_max to 3.0, "
            "aspect=2.5 cluster should be accepted"
        )

    def test_saliency_weights_sourced_from_config(self, clean_config_cache):
        """Confirm saliency_tracker reads weights from the config
        instead of its own local table."""
        from backend.services.saliency_tracker import _get_fusion_weights

        spatial, temporal, color = _get_fusion_weights("talking_head")
        # Spec defaults for talking_head: (0.45, 0.30, 0.25)
        assert abs(spatial - 0.45) < 1e-6
        assert abs(temporal - 0.30) < 1e-6
        assert abs(color - 0.25) < 1e-6

        # Mutate: reduce spatial weight.
        rc._CONTENT_OVERRIDES["talking_head"]["saliency_spatial_weight"] = 0.20
        rc.DEFAULT_CONFIG = None

        spatial2, _, _ = _get_fusion_weights("talking_head")
        assert abs(spatial2 - 0.20) < 1e-6
