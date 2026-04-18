"""Tests for text overlay demotion from centroid contributors to constraints.

Text/HUD overlays must stay in frame (must_be_in_frame=True) but must NOT
pull the crop center toward them (excluded_from_centroid=True). The crop
center is derived only from human subjects.
"""

import pytest
from unittest.mock import MagicMock

from backend.services.face_detector import FaceInfo, FrameFaces
from backend.services.scene_focus import aggregate_scene_focus
from backend.services.focus_model import FeatureKind, RequiredFeature


def _make_face(nose_x=30, nose_y=40, width=8, height=10, identity_id=0):
    return FaceInfo(
        x_center=nose_x, y_center=nose_y,
        width=width, height=height,
        nose_x=nose_x, nose_y=nose_y,
        confidence=0.9,
        identity_id=identity_id,
    )


def _make_frame_faces(timestamp, faces):
    return FrameFaces(
        timestamp=timestamp,
        frame_path=f"/tmp/frame_{timestamp}.jpg",
        faces=faces,
        primary_face_idx=0 if faces else -1,
    )


def _make_hud_persistent_regions(hud_x=0.7, hud_y=0.1, hud_w=0.2, hud_h=0.05):
    """Create a mock persistent_regions with a HUD region."""
    region = MagicMock()
    region.x = hud_x
    region.y = hud_y
    region.w = hud_w
    region.h = hud_h

    pr = MagicMock()
    pr.has_facecam = False
    pr.facecam_region = None
    pr.has_hud = True
    pr.hud_regions = [region]
    pr.regions = []
    return pr


class TestTextDemotion:
    def test_face_and_hud_crop_centered_on_face(self):
        """Frame with one human face and one HUD → crop centered on face, not HUD."""
        dense_faces = []
        for t in [0.1, 0.2, 0.3]:
            dense_faces.append(_make_frame_faces(
                t, [_make_face(nose_x=25, nose_y=40)],
            ))

        # HUD at x=80 (far right)
        persistent = _make_hud_persistent_regions(hud_x=0.7, hud_y=0.05)

        focus = aggregate_scene_focus(
            shot_start=0.0,
            shot_end=0.5,
            dense_faces=dense_faces,
            saliency_keyframes=[],
            persistent_regions=persistent,
            face_registry=MagicMock(),
            source_width=1920,
            source_height=1080,
        )

        cx, cy = focus.optimal_crop_center
        # Crop center should be near the face at x=25, NOT pulled toward HUD at x=80
        assert cx < 40, f"Expected crop center near face (x=25), got cx={cx}"

    def test_text_only_frame_centers_at_50(self):
        """Frame with only a text overlay (no faces) → crop centered at (50, 50)."""
        # No dense faces
        persistent = _make_hud_persistent_regions(hud_x=0.8, hud_y=0.1)

        focus = aggregate_scene_focus(
            shot_start=0.0,
            shot_end=0.5,
            dense_faces=[],
            saliency_keyframes=[],
            persistent_regions=persistent,
            face_registry=MagicMock(),
            source_width=1920,
            source_height=1080,
        )

        cx, cy = focus.optimal_crop_center
        # With only excluded features, should fall back to (50, 50)
        assert cx == 50, f"Expected crop center at x=50, got cx={cx}"
        assert cy == 50, f"Expected crop center at y=50, got cy={cy}"

    def test_hud_still_in_required_features(self):
        """HUD features are still in the required list (must be in frame)."""
        persistent = _make_hud_persistent_regions()

        focus = aggregate_scene_focus(
            shot_start=0.0,
            shot_end=0.5,
            dense_faces=[],
            saliency_keyframes=[],
            persistent_regions=persistent,
            face_registry=MagicMock(),
            source_width=1920,
            source_height=1080,
        )

        hud_features = [rf for rf in focus.required if rf.kind == FeatureKind.HUD]
        assert len(hud_features) >= 1
        assert hud_features[0].must_be_in_frame is True
        assert hud_features[0].excluded_from_centroid is True

    def test_bounding_rect_includes_all_required(self):
        """The min bounding rect uses ALL required features (HUD included)."""
        dense_faces = [
            _make_frame_faces(0.1, [_make_face(nose_x=20, nose_y=40)]),
        ]
        # HUD at right edge
        persistent = _make_hud_persistent_regions(hud_x=0.8, hud_y=0.05, hud_w=0.15, hud_h=0.05)

        focus = aggregate_scene_focus(
            shot_start=0.0,
            shot_end=0.5,
            dense_faces=dense_faces,
            saliency_keyframes=[],
            persistent_regions=persistent,
            face_registry=MagicMock(),
            source_width=1920,
            source_height=1080,
        )

        # Bounding rect should span from face at x=20 to HUD at x=87.5
        rect_cx, rect_cy, rect_w, rect_h = focus.min_bounding_rect
        # The rect should be wide enough to cover both
        assert rect_w > 50, f"Expected wide bounding rect, got w={rect_w}"

    def test_excluded_from_centroid_default_false(self):
        """RequiredFeature.excluded_from_centroid defaults to False."""
        rf = RequiredFeature(
            t_start=0, t_end=1, x=50, y=50, w=10, h=10,
            kind=FeatureKind.FACE, weight=1.0, must_be_in_frame=True,
        )
        assert rf.excluded_from_centroid is False

    def test_multiple_faces_one_hud_centroid_from_faces(self):
        """Two faces + one HUD: centroid is the average of the two faces."""
        dense_faces = []
        for t in [0.1, 0.2, 0.3]:
            dense_faces.append(_make_frame_faces(
                t,
                [
                    _make_face(nose_x=20, nose_y=40, identity_id=0),
                    _make_face(nose_x=60, nose_y=40, identity_id=1),
                ],
            ))

        # HUD at x=90 (far right, should not pull centroid)
        persistent = _make_hud_persistent_regions(hud_x=0.85, hud_y=0.05)

        focus = aggregate_scene_focus(
            shot_start=0.0,
            shot_end=0.5,
            dense_faces=dense_faces,
            saliency_keyframes=[],
            persistent_regions=persistent,
            face_registry=MagicMock(),
            source_width=1920,
            source_height=1080,
        )

        cx, cy = focus.optimal_crop_center
        # Without HUD influence, centroid should be near (20+60)/2 = 40
        # The bounding rect center might be slightly different but should be
        # < 50, not pulled toward HUD at 90
        assert cx < 55, f"Expected crop center near faces (~40), got cx={cx}"
