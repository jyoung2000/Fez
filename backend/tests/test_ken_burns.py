"""Phase 0 Task 0.3 — Ken Burns layout.

``should_use_ken_burns`` is the detection gate; ``build_ken_burns_op``
turns a detected scene into a ``RenderOpKind.KEN_BURNS`` op with a
two-keypoint motion path. The FFmpeg filter builder renders that via
``_filter_ken_burns``.
"""

import pytest

from backend.services.layout_kenburns import (
    SaliencyCentroid,
    build_ken_burns_op,
    should_use_ken_burns,
)
from backend.services.reframe_config import ReframeConfig
from backend.services.render_plan import RenderOpKind


def test_skip_when_face_is_present():
    assert should_use_ken_burns(
        scene_start=0.0, scene_end=5.0,
        has_face_in_scene=True, saliency_spread=0.5,
        content_type="landscape",
    ) is False


def test_skip_when_saliency_peak_is_sharp():
    # Sharp peak (spread < 0.30) = there's a dominant subject, so
    # Ken Burns should stay off; a tracker should handle it.
    assert should_use_ken_burns(
        scene_start=0.0, scene_end=5.0,
        has_face_in_scene=False, saliency_spread=0.12,
        content_type="landscape",
    ) is False


def test_skip_when_scene_too_short():
    assert should_use_ken_burns(
        scene_start=0.0, scene_end=1.0,  # 1.0 < default 2.0
        has_face_in_scene=False, saliency_spread=0.40,
        content_type="landscape",
    ) is False


def test_skip_when_content_type_not_allowed():
    # talking_head is not in the Ken Burns content-type allowlist.
    assert should_use_ken_burns(
        scene_start=0.0, scene_end=5.0,
        has_face_in_scene=False, saliency_spread=0.40,
        content_type="talking_head",
    ) is False


def test_fires_on_landscape_broll():
    assert should_use_ken_burns(
        scene_start=0.0, scene_end=5.0,
        has_face_in_scene=False, saliency_spread=0.40,
        content_type="landscape",
    ) is True


def test_disabled_when_zoom_cap_is_1():
    cfg = ReframeConfig(ken_burns_max_zoom=1.0)
    assert should_use_ken_burns(
        scene_start=0.0, scene_end=5.0,
        has_face_in_scene=False, saliency_spread=0.40,
        content_type="landscape", config=cfg,
    ) is False


def test_op_has_two_keypoints_and_correct_kind():
    op = build_ken_burns_op(
        scene_start=0.0, scene_end=5.0,
        centroid=SaliencyCentroid(cx=0.66, cy=0.40, spread=0.40),
        source_width=1920, source_height=1080,
    )
    assert op.kind == RenderOpKind.KEN_BURNS
    assert op.start_sec == 0.0
    assert op.end_sec == 5.0
    assert len(op.motion_path) == 2


def test_op_respects_travel_cap():
    """Centroid in far corner should NOT produce travel > travel_cap."""
    cfg = ReframeConfig(ken_burns_max_travel_frac=0.05, ken_burns_max_zoom=1.08)
    op = build_ken_burns_op(
        scene_start=0.0, scene_end=5.0,
        centroid=SaliencyCentroid(cx=0.95, cy=0.95, spread=0.40),
        source_width=1920, source_height=1080, config=cfg,
    )
    start_kp = op.motion_path[0]
    end_kp = op.motion_path[-1]
    start_cx = start_kp.rect.x + start_kp.rect.w / 2
    start_cy = start_kp.rect.y + start_kp.rect.h / 2
    end_cx = end_kp.rect.x + end_kp.rect.w / 2
    end_cy = end_kp.rect.y + end_kp.rect.h / 2
    travel = abs(end_cx - start_cx) + abs(end_cy - start_cy)
    # Allow tiny float slack above the cap from clamping.
    assert travel <= cfg.ken_burns_max_travel_frac + 1e-3


def test_op_end_rect_is_smaller_than_start():
    """Push IN means the end crop is the zoomed-in one, i.e. smaller."""
    op = build_ken_burns_op(
        scene_start=0.0, scene_end=5.0,
        centroid=SaliencyCentroid(cx=0.50, cy=0.50, spread=0.40),
        source_width=1920, source_height=1080,
    )
    start_rect = op.motion_path[0].rect
    end_rect = op.motion_path[-1].rect
    assert end_rect.w < start_rect.w
    assert end_rect.h < start_rect.h


def test_ffmpeg_filter_has_cosine_ease():
    """The emitted filter string uses the cosine ease on t01 so the
    push eases in and out of static endpoints."""
    from backend.services.ffmpeg_filter_builder import _filter_ken_burns

    op = build_ken_burns_op(
        scene_start=0.0, scene_end=5.0,
        centroid=SaliencyCentroid(cx=0.66, cy=0.40, spread=0.40),
        source_width=1920, source_height=1080,
    )
    out = _filter_ken_burns(op, "v0", 1920, 1080, 1080, 1920, 0.0, 5.0)
    assert "cos" in out
    # No enable=between(...) — Phase 0 filter uses trim+setpts.
    assert "enable=between" not in out
    assert out.endswith("[v0]")
