"""Tests for check_required_features_fit in SubjectConfidenceEstimator."""

from backend.services.subject_confidence import SubjectConfidenceEstimator
from backend.services.focus_model import FeatureKind, RequiredFeature


def _make_estimator():
    """Create an estimator with minimal mock data."""
    return SubjectConfidenceEstimator(
        face_registry=None,
        dense_faces=[],
        active_speaker_events=[],
        transcript_segments=[],
        speaker_to_slot={},
        source_width=1920,
        source_height=1080,
    )


def _make_rf(x, y, w, h, identity=0, kind=FeatureKind.FACE):
    return RequiredFeature(
        t_start=0.0, t_end=1.0,
        x=x, y=y, w=w, h=h,
        kind=kind, weight=1.0, must_be_in_frame=True,
        identity=identity,
    )


class TestRequiredFeaturesFit:
    def test_face_center_inside_but_edge_outside(self):
        """Face center inside crop but right edge outside -> False."""
        estimator = _make_estimator()
        # Face centered at x=68 with width=10 -> right edge at 73
        # Crop centered at 50 with width=30 -> right edge at 65
        rf = _make_rf(x=68, y=40, w=10, h=13)
        fits, detail = estimator.check_required_features_fit(
            seg_start=0, seg_end=1,
            crop_center_x=50, crop_center_y=50,
            crop_width_pct=30, crop_height_pct=100,
            required_features=[rf],
        )
        assert fits is False
        assert "clipped" in detail

    def test_face_fully_inside(self):
        """Face fully inside crop -> True."""
        estimator = _make_estimator()
        rf = _make_rf(x=50, y=40, w=10, h=13)
        fits, detail = estimator.check_required_features_fit(
            seg_start=0, seg_end=1,
            crop_center_x=50, crop_center_y=50,
            crop_width_pct=30, crop_height_pct=100,
            required_features=[rf],
        )
        assert fits is True
        assert detail == "all_required_in_frame"

    def test_two_required_one_clipped(self):
        """Two required features, one clipped -> False with detail identifying which."""
        estimator = _make_estimator()
        rf_ok = _make_rf(x=50, y=40, w=10, h=13, identity=0)
        rf_clipped = _make_rf(x=80, y=40, w=10, h=13, identity=1)
        fits, detail = estimator.check_required_features_fit(
            seg_start=0, seg_end=1,
            crop_center_x=50, crop_center_y=50,
            crop_width_pct=30, crop_height_pct=100,
            required_features=[rf_ok, rf_clipped],
        )
        assert fits is False
        assert "1" in detail  # identity of clipped feature

    def test_zero_required_features(self):
        """Zero required features -> True."""
        estimator = _make_estimator()
        fits, detail = estimator.check_required_features_fit(
            seg_start=0, seg_end=1,
            crop_center_x=50, crop_center_y=50,
            crop_width_pct=30, crop_height_pct=100,
            required_features=[],
        )
        assert fits is True

    def test_non_fhd_source(self):
        """Non-FHD source dims used correctly."""
        estimator = SubjectConfidenceEstimator(
            face_registry=None,
            dense_faces=[],
            active_speaker_events=[],
            transcript_segments=[],
            speaker_to_slot={},
            source_width=1280,
            source_height=720,
        )
        # Source dims are stored on the estimator
        assert estimator.source_width == 1280
        assert estimator.source_height == 720
        # The check uses passed crop dimensions, not source dims
        rf = _make_rf(x=50, y=50, w=5, h=5)
        fits, _ = estimator.check_required_features_fit(
            seg_start=0, seg_end=1,
            crop_center_x=50, crop_center_y=50,
            crop_width_pct=30, crop_height_pct=100,
            required_features=[rf],
        )
        assert fits is True

    def test_non_required_feature_ignored(self):
        """Non-required features (must_be_in_frame=False) are ignored."""
        estimator = _make_estimator()
        rf = RequiredFeature(
            t_start=0.0, t_end=1.0,
            x=95, y=50, w=10, h=10,
            kind=FeatureKind.SALIENCY, weight=0.5, must_be_in_frame=False,
        )
        fits, detail = estimator.check_required_features_fit(
            seg_start=0, seg_end=1,
            crop_center_x=50, crop_center_y=50,
            crop_width_pct=30, crop_height_pct=100,
            required_features=[rf],
        )
        assert fits is True

    def test_feature_outside_time_range(self):
        """Feature outside segment time range is ignored."""
        estimator = _make_estimator()
        rf = _make_rf(x=95, y=50, w=10, h=10)
        rf.t_start = 5.0
        rf.t_end = 6.0
        fits, _ = estimator.check_required_features_fit(
            seg_start=0, seg_end=1,
            crop_center_x=50, crop_center_y=50,
            crop_width_pct=30, crop_height_pct=100,
            required_features=[rf],
        )
        assert fits is True
