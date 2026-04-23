"""Regression test for bench fixture geometric feasibility.

A bench fixture is useless if no solver choice satisfies the scoring
rubric. This test asserts each built-in fixture has at least ONE crop
placement that scores within spec, so a correctly-implemented solver
would actually pass."""

import pytest

from backend.scripts.measure_human_reframe_quality import (
    _fx_gappy_faces,
    _fx_mixed_genre,
    _fx_talking_head,
)
from backend.services.reframe_config import get_default_config


@pytest.mark.parametrize("fx_fn", [_fx_talking_head, _fx_mixed_genre, _fx_gappy_faces])
def test_fixture_headroom_is_reachable(fx_fn):
    """Assert each fixture has at least one face whose source-frame
    position allows a solver cy to place the face_top within the
    headroom window."""
    cfg = get_default_config()
    fx = fx_fn()
    crop_h_frac = 1.0  # 9:16 out of 16:9 source → full height
    reachable = 0
    total = 0
    for frame in fx["dense_faces"]:
        for face in frame.faces:
            total += 1
            face_top_src = (face.nose_y - face.height * 0.5) / 100.0
            # Solver can put crop anywhere in [crop_h/2, 1 - crop_h/2].
            # If source is 16:9 and output 9:16, there is NO freedom:
            # crop_h_frac == 1.0 and cy must be 0.5.
            # Target: face_top_in_crop in [headroom_min, headroom_max].
            # => face_top_src in cy + [h_min - 0.5, h_max - 0.5] * crop_h
            # For crop_h = 1.0, cy = 0.5, so face_top_src in [h_min, h_max].
            if cfg.headroom_min <= face_top_src <= cfg.headroom_max:
                reachable += 1
    assert reachable > 0, (
        f"{fx_fn.__name__}: zero faces reachable by any solver. "
        f"Fixture tests the scoring code, not the solver."
    )
    # Also require a reasonable fraction — at least 50% of faces should
    # be reachable so the fixture has discriminative power.
    assert reachable / total >= 0.5, (
        f"{fx_fn.__name__}: only {reachable}/{total} faces reachable."
    )
