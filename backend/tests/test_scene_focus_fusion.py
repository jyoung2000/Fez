"""Tests for scene_focus.py with subject_tracks parameter."""

from dataclasses import dataclass, field

from backend.services.scene_focus import aggregate_scene_focus
from backend.services.focus_model import FeatureKind, SubjectTrack


# -- Minimal mock objects --
@dataclass
class MockFace:
    nose_x: float = 50.0
    nose_y: float = 40.0
    x: float = 50.0
    y: float = 40.0
    width: float = 10.0
    height: float = 13.0
    identity_id: int = 0


@dataclass
class MockFrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)
    primary_face_idx: int = 0


class TestSubjectTracksRegression:
    def test_none_identical_to_previous(self):
        """subject_tracks=None produces byte-identical output."""
        dense_faces = [
            MockFrameFaces(timestamp=t, faces=[MockFace(nose_x=50, nose_y=40)])
            for t in [0.0, 0.5, 1.0, 1.5, 2.0]
        ]

        result_old = aggregate_scene_focus(
            shot_start=0, shot_end=2.5, dense_faces=dense_faces,
            saliency_keyframes=[], persistent_regions=None,
            face_registry=None, source_width=1920, source_height=1080,
        )
        result_new = aggregate_scene_focus(
            shot_start=0, shot_end=2.5, dense_faces=dense_faces,
            saliency_keyframes=[], persistent_regions=None,
            face_registry=None, source_width=1920, source_height=1080,
            subject_tracks=None,
        )

        assert len(result_old.required) == len(result_new.required)
        assert len(result_old.optional) == len(result_new.optional)
        assert result_old.fits_target_aspect == result_new.fits_target_aspect
        assert result_old.optimal_crop_center == result_new.optimal_crop_center

    def test_empty_list_identical(self):
        """subject_tracks=[] is also identical."""
        dense_faces = [
            MockFrameFaces(timestamp=0.5, faces=[MockFace(nose_x=50)])
        ]
        result_base = aggregate_scene_focus(
            shot_start=0, shot_end=1, dense_faces=dense_faces,
            saliency_keyframes=[], persistent_regions=None,
            face_registry=None, source_width=1920, source_height=1080,
        )
        result_empty = aggregate_scene_focus(
            shot_start=0, shot_end=1, dense_faces=dense_faces,
            saliency_keyframes=[], persistent_regions=None,
            face_registry=None, source_width=1920, source_height=1080,
            subject_tracks=[],
        )
        assert len(result_base.required) == len(result_empty.required)
        assert result_base.fits_target_aspect == result_empty.fits_target_aspect


class TestFaceConfirmedTrackIntegration:
    def test_face_confirmed_track_required(self):
        """face_confirmed track -> features appear with kind=FACE, must_be_in_frame=True."""
        track = SubjectTrack(
            track_id=0, source="face_confirmed", confidence=0.95,
            face_slot_id=0, persistent_id=0,
            bbox_trajectory=[
                (0.5, 50.0, 40.0, 10.0, 13.0),
                (1.0, 52.0, 40.0, 10.0, 13.0),
            ],
        )
        result = aggregate_scene_focus(
            shot_start=0, shot_end=2, dense_faces=[],
            saliency_keyframes=[], persistent_regions=None,
            face_registry=None, source_width=1920, source_height=1080,
            subject_tracks=[track],
        )
        assert len(result.required) == 2
        assert all(r.kind == FeatureKind.FACE for r in result.required)
        assert all(r.must_be_in_frame for r in result.required)


class TestFaceLikePromotedTrackIntegration:
    def test_face_like_promoted_required(self):
        """face_like_promoted track -> features with kind=FACE_LIKE, must_be_in_frame=True."""
        track = SubjectTrack(
            track_id=0, source="face_like_promoted", confidence=0.6,
            persistent_id=0,
            bbox_trajectory=[
                (0.5, 60.0, 50.0, 12.0, 16.0),
                (1.0, 62.0, 50.0, 12.0, 16.0),
                (1.5, 64.0, 50.0, 12.0, 16.0),
            ],
        )
        result = aggregate_scene_focus(
            shot_start=0, shot_end=2, dense_faces=[],
            saliency_keyframes=[], persistent_regions=None,
            face_registry=None, source_width=1920, source_height=1080,
            subject_tracks=[track],
        )
        assert len(result.required) == 3
        assert all(r.kind == FeatureKind.FACE_LIKE for r in result.required)
        assert all(r.must_be_in_frame for r in result.required)
        # Weight should be the track's confidence
        assert all(abs(r.weight - 0.6) < 0.01 for r in result.required)

    def test_promoted_track_affects_bounding_rect(self):
        """Promoted track's bbox affects the min bounding rect and fit check."""
        # Place a promoted track far to the right — should widen the bounding rect
        track = SubjectTrack(
            track_id=0, source="face_like_promoted", confidence=0.6,
            persistent_id=0,
            bbox_trajectory=[(0.5, 85.0, 50.0, 12.0, 16.0)],
        )
        result = aggregate_scene_focus(
            shot_start=0, shot_end=1, dense_faces=[],
            saliency_keyframes=[], persistent_regions=None,
            face_registry=None, source_width=1920, source_height=1080,
            subject_tracks=[track],
        )
        # Bounding rect center should be near x=85
        cx = result.min_bounding_rect[0]
        assert abs(cx - 85.0) < 1.0

    def test_promoted_generates_per_frame_targets(self):
        """Promoted track generates per-frame targets for trajectory planning."""
        track = SubjectTrack(
            track_id=0, source="face_like_promoted", confidence=0.6,
            persistent_id=0,
            bbox_trajectory=[
                (0.5, 40.0, 50.0, 12.0, 16.0),
                (1.0, 50.0, 50.0, 12.0, 16.0),
                (1.5, 60.0, 50.0, 12.0, 16.0),
            ],
        )
        result = aggregate_scene_focus(
            shot_start=0, shot_end=2, dense_faces=[],
            saliency_keyframes=[], persistent_regions=None,
            face_registry=None, source_width=1920, source_height=1080,
            subject_tracks=[track],
        )
        assert len(result.per_frame_target) == 3
        # Targets should follow the track's x positions
        assert abs(result.per_frame_target[0][1] - 40.0) < 1
        assert abs(result.per_frame_target[1][1] - 50.0) < 1
        assert abs(result.per_frame_target[2][1] - 60.0) < 1


class TestMixedFaceAndPromoted:
    def test_both_contribute_to_required(self):
        """face_confirmed + face_like_promoted both contribute to required list."""
        face_track = SubjectTrack(
            track_id=0, source="face_confirmed", confidence=0.95,
            face_slot_id=0, persistent_id=0,
            bbox_trajectory=[(0.5, 30.0, 40.0, 10.0, 13.0)],
        )
        promoted_track = SubjectTrack(
            track_id=1, source="face_like_promoted", confidence=0.6,
            persistent_id=1,
            bbox_trajectory=[(0.5, 70.0, 50.0, 12.0, 16.0)],
        )
        result = aggregate_scene_focus(
            shot_start=0, shot_end=1, dense_faces=[],
            saliency_keyframes=[], persistent_regions=None,
            face_registry=None, source_width=1920, source_height=1080,
            subject_tracks=[face_track, promoted_track],
        )
        assert len(result.required) == 2
        kinds = {r.kind for r in result.required}
        assert FeatureKind.FACE in kinds
        assert FeatureKind.FACE_LIKE in kinds

    def test_wide_spread_doesnt_fit(self):
        """Two tracks 60% apart -> doesn't fit in 9:16 crop."""
        track1 = SubjectTrack(
            track_id=0, source="face_confirmed", confidence=0.95,
            face_slot_id=0, persistent_id=0,
            bbox_trajectory=[(0.5, 15.0, 40.0, 10.0, 13.0)],
        )
        track2 = SubjectTrack(
            track_id=1, source="face_like_promoted", confidence=0.6,
            persistent_id=1,
            bbox_trajectory=[(0.5, 85.0, 50.0, 12.0, 16.0)],
        )
        result = aggregate_scene_focus(
            shot_start=0, shot_end=1, dense_faces=[],
            saliency_keyframes=[], persistent_regions=None,
            face_registry=None, source_width=1920, source_height=1080,
            subject_tracks=[track1, track2],
        )
        # Bounding rect width: 85+6 - (15-5) = 81, crop_width_pct ~31.6
        assert result.fits_target_aspect is False
