"""Blueprint v2 Phase 5 — Tier 3 generative outpaint fill."""

import asyncio
import os
import tempfile
from dataclasses import dataclass, field

import pytest

from backend.services import outpaint_fill as op_mod
from backend.services.outpaint_fill import (
    SAFE_CONTENT_TYPES,
    UNSAFE_CONTENT_TYPES,
    OutpaintResult,
    _parse_provider_response,
    analyze_scene_outpaint_safety,
    get_telemetry_snapshot,
    maybe_promote_to_outpaint,
    run_outpaint,
    should_outpaint_scene,
)
from backend.services.render_plan import (
    Rect,
    RenderOp,
    RenderOpKind,
    RenderPlan,
)


# ── Safety gate ───────────────────────────────────────────────────


def test_safe_and_unsafe_sets_disjoint():
    assert SAFE_CONTENT_TYPES.isdisjoint(UNSAFE_CONTENT_TYPES)


def test_gate_rejects_unsafe_content_types():
    allow, reason = should_outpaint_scene(
        content_type="anime",
        scene_has_face_near_edge=False,
        scene_has_fast_motion_at_edge=False,
        scene_has_hud_at_edge=False,
        is_hero_clip=True,
        user_opt_in=True,
    )
    assert not allow
    assert "unsafe" in reason


def test_gate_rejects_content_types_outside_safe_set():
    """Generic / unknown content types should also be rejected —
    outpainting is opt-in per genre."""
    allow, _ = should_outpaint_scene(
        content_type="generic",
        scene_has_face_near_edge=False,
        scene_has_fast_motion_at_edge=False,
        scene_has_hud_at_edge=False,
        is_hero_clip=True,
        user_opt_in=True,
    )
    assert not allow


def test_gate_rejects_face_at_edge():
    allow, reason = should_outpaint_scene(
        content_type="documentary",
        scene_has_face_near_edge=True,
        scene_has_fast_motion_at_edge=False,
        scene_has_hud_at_edge=False,
        is_hero_clip=True,
        user_opt_in=True,
    )
    assert not allow
    assert "face near crop edge" in reason


def test_gate_rejects_fast_motion_at_edge():
    allow, reason = should_outpaint_scene(
        content_type="landscape",
        scene_has_face_near_edge=False,
        scene_has_fast_motion_at_edge=True,
        scene_has_hud_at_edge=False,
        is_hero_clip=True,
        user_opt_in=True,
    )
    assert not allow
    assert "fast motion" in reason


def test_gate_rejects_hud_at_edge():
    allow, reason = should_outpaint_scene(
        content_type="broll",
        scene_has_face_near_edge=False,
        scene_has_fast_motion_at_edge=False,
        scene_has_hud_at_edge=True,
        is_hero_clip=True,
        user_opt_in=True,
    )
    assert not allow
    assert "HUD" in reason


def test_gate_allows_hero_landscape():
    allow, reason = should_outpaint_scene(
        content_type="landscape",
        scene_has_face_near_edge=False,
        scene_has_fast_motion_at_edge=False,
        scene_has_hud_at_edge=False,
        is_hero_clip=True,
        user_opt_in=False,
    )
    assert allow
    assert reason == "ok"


def test_gate_allows_user_opt_in_even_on_non_hero():
    allow, _ = should_outpaint_scene(
        content_type="vlog",
        scene_has_face_near_edge=False,
        scene_has_fast_motion_at_edge=False,
        scene_has_hud_at_edge=False,
        is_hero_clip=False,
        user_opt_in=True,
    )
    assert allow


def test_gate_rejects_when_no_trigger():
    """Safe content + all-clear safety but neither hero nor opt-in →
    stay on BLUR_FILL so the user doesn't pay per-call without asking."""
    allow, reason = should_outpaint_scene(
        content_type="documentary",
        scene_has_face_near_edge=False,
        scene_has_fast_motion_at_edge=False,
        scene_has_hud_at_edge=False,
        is_hero_clip=False,
        user_opt_in=False,
    )
    assert not allow
    assert "not hero" in reason


# ── analyze_scene_outpaint_safety ─────────────────────────────────


@dataclass
class _Face:
    x_center: float = 50.0  # percent
    width: float = 10.0


@dataclass
class _FrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)


def test_safety_detects_face_near_left_edge():
    # Source 1920×1080, crop (0, 0, 1080, 1080) — left edge at x=0.
    # Face centered at x_center=5% → 96 px → inside 8% band (154 px)
    # of the left edge.
    frames = [_FrameFaces(timestamp=1.0, faces=[_Face(x_center=5.0)])]
    safety = analyze_scene_outpaint_safety(
        scene_start=0.0, scene_end=2.0,
        primary_crop_rect_px=(0, 0, 1080, 1080),
        source_width=1920,
        dense_faces=frames,
    )
    assert safety["face_near_edge"] is True


def test_safety_ignores_face_in_middle():
    frames = [_FrameFaces(timestamp=1.0, faces=[_Face(x_center=50.0)])]
    safety = analyze_scene_outpaint_safety(
        scene_start=0.0, scene_end=2.0,
        primary_crop_rect_px=(420, 0, 1080, 1080),
        source_width=1920,
        dense_faces=frames,
    )
    assert safety["face_near_edge"] is False


def test_safety_flags_fast_motion_at_edge():
    flow = [{"timestamp": 1.0, "magnitude_at_edges": 0.25}]
    safety = analyze_scene_outpaint_safety(
        scene_start=0.0, scene_end=2.0,
        primary_crop_rect_px=(420, 0, 1080, 1080),
        source_width=1920,
        dense_faces=[],
        optical_flow_frames=flow,
    )
    assert safety["fast_motion_at_edge"] is True


def test_safety_ignores_slow_motion_at_edge():
    flow = [{"timestamp": 1.0, "magnitude_at_edges": 0.05}]
    safety = analyze_scene_outpaint_safety(
        scene_start=0.0, scene_end=2.0,
        primary_crop_rect_px=(420, 0, 1080, 1080),
        source_width=1920,
        dense_faces=[],
        optical_flow_frames=flow,
    )
    assert safety["fast_motion_at_edge"] is False


def test_safety_flags_hud_at_edge_percent_units():
    # HUD region at 2% center width → well inside the 8% left-edge band.
    huds = [{"x": 0.0, "w": 4.0, "_units": "pct"}]
    safety = analyze_scene_outpaint_safety(
        scene_start=0.0, scene_end=2.0,
        primary_crop_rect_px=(0, 0, 1080, 1080),
        source_width=1920,
        dense_faces=[],
        hud_regions=huds,
    )
    assert safety["hud_at_edge"] is True


# ── _parse_provider_response ─────────────────────────────────────


def test_parse_success_response():
    r = _parse_provider_response(
        {
            "status": "done",
            "output_url": "/tmp/out.mp4",
            "cost_usd": 0.42,
            "latency_sec": 12.3,
        },
        provider="luma",
    )
    assert r.success
    assert r.provider == "luma"
    assert r.cost_usd == pytest.approx(0.42)


def test_parse_rejects_non_done_status():
    r = _parse_provider_response(
        {"status": "running", "output_url": "x"}, provider="luma",
    )
    assert not r.success


def test_parse_rejects_missing_output():
    r = _parse_provider_response(
        {"status": "done"}, provider="luma",
    )
    assert not r.success


def test_parse_rejects_non_dict():
    r = _parse_provider_response("garbage", provider="luma")
    assert not r.success


# ── run_outpaint dispatcher ──────────────────────────────────────


def test_run_outpaint_disabled_by_master_flag(monkeypatch):
    monkeypatch.delenv("CLIPAI_OUTPAINT_ENABLED", raising=False)
    r = asyncio.run(run_outpaint(
        source_segment_path="/tmp/x.mp4",
        primary_crop_rect={"x": 0, "y": 0, "w": 1, "h": 1},
        duration_sec=5.0,
    ))
    assert not r.success
    assert "CLIPAI_OUTPAINT_ENABLED" in r.failure_reason


def test_run_outpaint_unknown_provider(monkeypatch):
    monkeypatch.setenv("CLIPAI_OUTPAINT_ENABLED", "1")
    monkeypatch.setenv("CLIPAI_OUTPAINT_PROVIDER", "made_up")
    r = asyncio.run(run_outpaint(
        source_segment_path="/tmp/x.mp4",
        primary_crop_rect={"x": 0, "y": 0, "w": 1, "h": 1},
        duration_sec=5.0,
    ))
    assert not r.success
    assert "unknown provider" in r.failure_reason


def test_run_outpaint_success_path_records_telemetry(monkeypatch):
    monkeypatch.setenv("CLIPAI_OUTPAINT_ENABLED", "1")
    monkeypatch.setenv("CLIPAI_OUTPAINT_PROVIDER", "luma")

    async def _stub(**kwargs):
        return OutpaintResult(
            success=True, output_path="/tmp/fake.mp4",
            provider="luma", cost_usd=0.5,
        )

    monkeypatch.setattr(op_mod, "_call_luma", _stub)
    before = len(get_telemetry_snapshot(limit=256))
    r = asyncio.run(run_outpaint(
        source_segment_path="/tmp/x.mp4",
        primary_crop_rect={"x": 0, "y": 0, "w": 1, "h": 1},
        duration_sec=5.0,
    ))
    assert r.success
    after = get_telemetry_snapshot(limit=256)
    assert len(after) == before + 1
    assert after[-1]["ok"] is True
    assert after[-1]["cost_usd"] == pytest.approx(0.5)


def test_run_outpaint_failure_path_records_telemetry(monkeypatch):
    monkeypatch.setenv("CLIPAI_OUTPAINT_ENABLED", "1")
    monkeypatch.setenv("CLIPAI_OUTPAINT_PROVIDER", "luma")

    async def _stub(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(op_mod, "_call_luma", _stub)
    before = len(get_telemetry_snapshot(limit=256))
    r = asyncio.run(run_outpaint(
        source_segment_path="/tmp/x.mp4",
        primary_crop_rect={"x": 0, "y": 0, "w": 1, "h": 1},
        duration_sec=5.0,
    ))
    assert not r.success
    assert "boom" in r.failure_reason
    after = get_telemetry_snapshot(limit=256)
    assert len(after) == before + 1
    assert after[-1]["ok"] is False


# ── maybe_promote_to_outpaint ────────────────────────────────────


def _rect() -> Rect:
    return Rect(x=0.2, y=0.0, w=0.56, h=1.0)


def _mk_plan(ops: list, total: float) -> RenderPlan:
    return RenderPlan(
        source_width=1920, source_height=1080,
        target_width=1080, target_height=1920,
        total_duration_sec=total,
        fps=30.0,
        ops=ops,
    )


def _mk_blur_op(start: float, end: float) -> RenderOp:
    return RenderOp(
        kind=RenderOpKind.BLUR_FILL,
        start_sec=start,
        end_sec=end,
        primary_rect=_rect(),
    )


def test_promote_disabled_by_master_flag(monkeypatch):
    monkeypatch.delenv("CLIPAI_OUTPAINT_ENABLED", raising=False)
    plan = _mk_plan([_mk_blur_op(0, 5)], total=5)
    result = asyncio.run(maybe_promote_to_outpaint(
        plan,
        content_type="landscape",
        virality_score=9.0,
        source_width=1920,
    ))
    assert result is plan


def test_promote_non_hero_no_opt_in_returns_unchanged(monkeypatch):
    monkeypatch.setenv("CLIPAI_OUTPAINT_ENABLED", "1")
    plan = _mk_plan([_mk_blur_op(0, 5)], total=5)
    result = asyncio.run(maybe_promote_to_outpaint(
        plan,
        content_type="landscape",
        virality_score=0.0,
        source_width=1920,
    ))
    assert result is plan


def test_promote_unsafe_content_type_unchanged(monkeypatch):
    monkeypatch.setenv("CLIPAI_OUTPAINT_ENABLED", "1")
    plan = _mk_plan([_mk_blur_op(0, 5)], total=5)
    result = asyncio.run(maybe_promote_to_outpaint(
        plan,
        content_type="anime",
        virality_score=9.5,
        user_opt_in=True,
        source_width=1920,
    ))
    # Same BLUR_FILL ops survived — the unsafe-content-type gate fired.
    assert all(op.kind == RenderOpKind.BLUR_FILL for op in result.ops)


def test_promote_caps_at_max_per_clip(monkeypatch):
    """10 eligible BLUR_FILL ops with CLIPAI_OUTPAINT_MAX_PER_CLIP=3
    → exactly 3 get promoted."""
    monkeypatch.setenv("CLIPAI_OUTPAINT_ENABLED", "1")
    monkeypatch.setenv("CLIPAI_OUTPAINT_MAX_PER_CLIP", "3")
    monkeypatch.setenv("CLIPAI_OUTPAINT_PROVIDER", "luma")

    async def _stub(**kwargs):
        return OutpaintResult(
            success=True,
            output_path=__file__,  # any existing file so os.path.exists passes
            provider="luma",
            cost_usd=0.1,
        )

    monkeypatch.setattr(op_mod, "_call_luma", _stub)

    ops = [_mk_blur_op(float(i), float(i + 1)) for i in range(10)]
    plan = _mk_plan(ops, total=10.0)
    result = asyncio.run(maybe_promote_to_outpaint(
        plan,
        content_type="landscape",
        virality_score=9.5,
        source_width=1920,
    ))
    promoted = [op for op in result.ops if op.kind == RenderOpKind.OUTPAINT_FILL]
    remaining_blur = [op for op in result.ops if op.kind == RenderOpKind.BLUR_FILL]
    assert len(promoted) == 3
    assert len(remaining_blur) == 7
    # Provenance is stashed on the promoted op.
    for op in promoted:
        assert op.outpaint_provider == "luma"
        assert op.fallback_op_kind == "blur_fill"
    # Aggregate cost is exposed on the plan for DB write.
    assert getattr(result, "_outpaint_cost_usd", 0.0) == pytest.approx(0.3)


def test_promote_provider_failure_keeps_blur(monkeypatch):
    monkeypatch.setenv("CLIPAI_OUTPAINT_ENABLED", "1")
    monkeypatch.setenv("CLIPAI_OUTPAINT_PROVIDER", "luma")

    async def _stub(**kwargs):
        return OutpaintResult(
            success=False, provider="luma",
            failure_reason="http 500",
        )

    monkeypatch.setattr(op_mod, "_call_luma", _stub)

    plan = _mk_plan([_mk_blur_op(0, 5)], total=5)
    result = asyncio.run(maybe_promote_to_outpaint(
        plan,
        content_type="landscape",
        virality_score=9.5,
        source_width=1920,
    ))
    assert all(op.kind == RenderOpKind.BLUR_FILL for op in result.ops)


def test_promote_missing_output_file_keeps_blur(monkeypatch):
    """Provider says success=True but the output file doesn't exist
    — never promote, since the renderer would then silently fall to
    BLUR_FILL anyway. Don't paint success onto a broken state."""
    monkeypatch.setenv("CLIPAI_OUTPAINT_ENABLED", "1")
    monkeypatch.setenv("CLIPAI_OUTPAINT_PROVIDER", "luma")

    async def _stub(**kwargs):
        return OutpaintResult(
            success=True,
            output_path="/tmp/nonexistent-12345-does-not-exist.mp4",
            provider="luma", cost_usd=0.1,
        )

    monkeypatch.setattr(op_mod, "_call_luma", _stub)

    plan = _mk_plan([_mk_blur_op(0, 5)], total=5)
    result = asyncio.run(maybe_promote_to_outpaint(
        plan,
        content_type="landscape",
        virality_score=9.5,
        source_width=1920,
    ))
    assert all(op.kind == RenderOpKind.BLUR_FILL for op in result.ops)


def test_promote_non_blur_ops_untouched(monkeypatch):
    """Only BLUR_FILL ops are eligible. CROP / TRACKING_CROP / etc.
    flow through unchanged."""
    monkeypatch.setenv("CLIPAI_OUTPAINT_ENABLED", "1")
    monkeypatch.setenv("CLIPAI_OUTPAINT_PROVIDER", "luma")

    async def _stub(**kwargs):
        return OutpaintResult(
            success=True, output_path=__file__,
            provider="luma", cost_usd=0.1,
        )

    monkeypatch.setattr(op_mod, "_call_luma", _stub)

    crop_op = RenderOp(
        kind=RenderOpKind.CROP, start_sec=0.0, end_sec=5.0,
        primary_rect=_rect(),
    )
    plan = _mk_plan([crop_op, _mk_blur_op(5.0, 10.0)], total=10.0)
    result = asyncio.run(maybe_promote_to_outpaint(
        plan,
        content_type="landscape",
        virality_score=9.5,
        source_width=1920,
    ))
    kinds = [op.kind for op in result.ops]
    assert kinds[0] == RenderOpKind.CROP
    assert kinds[1] == RenderOpKind.OUTPAINT_FILL


# ── FFmpeg filter ────────────────────────────────────────────────


def test_outpaint_filter_falls_back_when_media_missing():
    from backend.services.ffmpeg_filter_builder import _filter_outpaint_fill

    op = RenderOp(
        kind=RenderOpKind.OUTPAINT_FILL,
        start_sec=0.0,
        end_sec=5.0,
        primary_rect=_rect(),
        outpainted_media_path="/tmp/does-not-exist-99999.mp4",
        fallback_op_kind="blur_fill",
    )
    # Missing file → filter collapses to BLUR_FILL expression.
    out = _filter_outpaint_fill(op, "v0", 1920, 1080, 1080, 1920, 0.0, 5.0)
    assert "gblur" in out  # blur_fill signature


def test_outpaint_filter_uses_media_when_present():
    from backend.services.ffmpeg_filter_builder import _filter_outpaint_fill

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"stub")
        media_path = f.name
    try:
        op = RenderOp(
            kind=RenderOpKind.OUTPAINT_FILL,
            start_sec=0.0,
            end_sec=5.0,
            primary_rect=_rect(),
            outpainted_media_path=media_path,
            fallback_op_kind="blur_fill",
        )
        op.outpaint_input_index = 1  # type: ignore[attr-defined]
        out = _filter_outpaint_fill(
            op, "v0", 1920, 1080, 1080, 1920, 0.0, 5.0,
        )
        # Uses the extra input stream instead of [0:v].
        assert "[1:v]" in out
        assert "scale=1080:1920" in out
    finally:
        os.unlink(media_path)
