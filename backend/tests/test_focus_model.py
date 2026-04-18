"""Tests for RequiredFeature and SceneFocusRegion dataclasses."""

import numpy as np

from backend.services.focus_model import (
    FeatureKind,
    RequiredFeature,
    SceneFocusRegion,
)


class TestRequiredFeature:
    def test_left_right_top_bottom(self):
        rf = RequiredFeature(
            t_start=0, t_end=1, x=50, y=40, w=20, h=30,
            kind=FeatureKind.FACE, weight=1.0, must_be_in_frame=True,
        )
        assert rf.left == 40.0
        assert rf.right == 60.0
        assert rf.top == 25.0
        assert rf.bottom == 55.0

    def test_to_dict_numpy_sanitization(self):
        rf = RequiredFeature(
            t_start=np.float32(0.0),
            t_end=np.float32(1.0),
            x=np.float64(50.0),
            y=np.float64(40.0),
            w=np.float32(20.0),
            h=np.float32(30.0),
            kind=FeatureKind.FACE,
            weight=np.float32(0.8),
            must_be_in_frame=True,
            identity=np.int64(2),
        )
        d = rf.to_dict()
        # All values should be plain Python types, not numpy scalars
        for key in ('t_start', 't_end', 'x', 'y', 'w', 'h', 'weight', 'identity'):
            val = d[key]
            assert not hasattr(val, 'item'), f"{key} is still a numpy scalar: {type(val)}"
        assert d['kind'] == 'face'
        assert d['must_be_in_frame'] is True

    def test_edge_feature_properties(self):
        """Feature at top-left corner."""
        rf = RequiredFeature(
            t_start=0, t_end=1, x=5, y=5, w=10, h=10,
            kind=FeatureKind.HUD, weight=0.5, must_be_in_frame=False,
        )
        assert rf.left == 0.0
        assert rf.top == 0.0
        assert rf.right == 10.0
        assert rf.bottom == 10.0


class TestSceneFocusRegion:
    def test_empty_required_list(self):
        sfr = SceneFocusRegion(shot_start=0, shot_end=10)
        assert sfr.min_bounding_rect == (0, 0, 0, 0)
        assert sfr.fits_target_aspect is True
        assert sfr.required == []
        assert sfr.optional == []

    def test_defaults(self):
        sfr = SceneFocusRegion(shot_start=0, shot_end=5)
        assert sfr.optimal_crop_center == (50, 50)
        assert sfr.per_frame_target == []
