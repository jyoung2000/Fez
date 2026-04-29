"""Unit coverage for the SOTA-preview/metric parity patch.

Three contracts are covered:

1. ``compare_autoflip_vs_clipai._serialize_reframe_segments``
   persists ``motion_path`` and ``hard_constraints`` (and emits
   ``None`` when the source segment has no path).

2. ``export_autoflip_compatible.reframe_segments_to_events`` derives
   per-frame ``crop_cx`` from the camera-path keypoint stream — so
   tracking shots show smooth motion and the legacy step path stays
   bit-compatible with cached ``segments.json`` blobs.

3. ``sota_render_preview._build_x_expression`` emits a flat sum of
   piecewise-linear lerp terms gated by half-open
   ``gte(t,t0)*lt(t,t1)`` intervals, with a tail hold past the last
   keypoint.

Pure Python — no ffmpeg, no numpy, no torch. Designed to run on the
sandbox the rest of the QA harness uses.
"""
from __future__ import annotations

import re

import pytest

from backend.scripts.compare_autoflip_vs_clipai import (
    _serialize_reframe_segments,
)
from backend.scripts.export_autoflip_compatible import (
    reframe_segments_to_events,
)
from backend.scripts.sota_render_preview import _build_x_expression


# ───────────────────────── helpers ─────────────────────────


class _Seg:
    """Tiny stand-in for ``ReframeSegment`` with attribute access."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _basic_seg(**overrides):
    base = dict(
        start=0.0, end=2.0, subject_x=500.0, subject_y=400.0,
        layout="single", active_slot=0, confidence=0.9,
        reason="speaker_turn", ease_in_ms=0, strategy="stationary",
        content_type="multi_speaker_panel",
    )
    base.update(overrides)
    return _Seg(**base)


# ───────────────────────── (1) serializer ─────────────────────────


class TestSerializerPersistsMotionPath:
    def test_persists_motion_path_pixel_pairs(self):
        seg = _basic_seg(
            strategy="tracking",
            motion_path=[(0.0, 500.0), (1.0, 600.0), (2.0, 700.0)],
        )
        out = _serialize_reframe_segments([seg])
        assert out[0]["motion_path"] == [
            [0.0, 500.0], [1.0, 600.0], [2.0, 700.0],
        ]

    def test_preserves_y_when_present(self):
        seg = _basic_seg(
            motion_path=[(0.0, 500.0, 400.0), (2.0, 700.0, 420.0)],
        )
        out = _serialize_reframe_segments([seg])
        assert out[0]["motion_path"] == [
            [0.0, 500.0, 400.0], [2.0, 700.0, 420.0],
        ]

    def test_emits_null_when_source_has_no_path(self):
        seg = _basic_seg()  # no motion_path
        out = _serialize_reframe_segments([seg])
        assert "motion_path" in out[0]
        assert out[0]["motion_path"] is None

    def test_persists_hard_constraints(self):
        seg = _basic_seg(
            hard_constraints=[(100.0, 200.0, 300.0, 400.0)],
        )
        out = _serialize_reframe_segments([seg])
        assert out[0]["hard_constraints"] == [
            [100.0, 200.0, 300.0, 400.0],
        ]

    def test_drops_garbage_motion_entries(self):
        # Mix valid + invalid; only valid survives.
        seg = _basic_seg(
            motion_path=[
                (0.0, 500.0),
                ("bad", "data"),  # ValueError on float()
                (1.0,),  # too short
                (2.0, 700.0),
            ],
        )
        out = _serialize_reframe_segments([seg])
        assert out[0]["motion_path"] == [[0.0, 500.0], [2.0, 700.0]]


# ───────────────────────── (2) events translator ─────────────────────────


class TestReframeSegmentsToEvents:
    def test_legacy_step_back_compat_holds_subject_x(self):
        # Cached segments.json without motion_path → held subject_x.
        segs = [
            {"start": 0.0, "end": 2.0, "subject_x": 100.0},
            {"start": 2.0, "end": 4.0, "subject_x": 1000.0},
        ]
        events = reframe_segments_to_events(segs, 1920, 1080, 30.0)
        # Every frame in segment 1 holds at 100/1920.
        seg1 = [e for e in events if e["t"] < 2.0]
        assert all(
            abs(e["crop_cx"] - 100.0 / 1920.0) < 1e-9 for e in seg1
        )
        # Every frame in segment 2 holds at 1000/1920.
        seg2 = [e for e in events if e["t"] >= 2.0]
        assert all(
            abs(e["crop_cx"] - 1000.0 / 1920.0) < 1e-9 for e in seg2
        )

    def test_motion_path_produces_smooth_motion(self):
        segs = [{
            "start": 0.0, "end": 1.0, "subject_x": 500.0,
            "motion_path": [(0.0, 500.0), (1.0, 1000.0)],
        }]
        events = reframe_segments_to_events(segs, 1920, 1080, 30.0)
        # First event ~ 500/1920, last event ~ 1000/1920 (just under).
        assert abs(events[0]["crop_cx"] - 500.0 / 1920.0) < 1e-6
        assert events[-1]["crop_cx"] > events[0]["crop_cx"] + 0.2
        # Motion should be MONOTONICALLY INCREASING (no jitter).
        xs = [e["crop_cx"] for e in events]
        assert xs == sorted(xs)

    def test_ease_smooths_a_motivated_cut(self):
        segs = [
            {"start": 0.0, "end": 1.0, "subject_x": 200.0},
            {
                "start": 1.0, "end": 2.0, "subject_x": 1700.0,
                "ease_in_ms": 200,
            },
        ]
        events = reframe_segments_to_events(segs, 1920, 1080, 60.0)
        # Sample x within the ease window — should be strictly between
        # the two endpoint values.
        ease_ev = [
            e for e in events
            if 1.01 <= e["t"] <= 1.19
        ]
        for e in ease_ev:
            x = e["crop_cx"] * 1920.0
            assert 200.0 < x < 1700.0, x
        # By the end of the ease (~t=1.20) the camera is at the
        # destination.
        post = next(e for e in events if e["t"] >= 1.21)
        assert abs(post["crop_cx"] * 1920.0 - 1700.0) < 5.0

    def test_scene_change_only_fires_at_segment_boundaries(self):
        segs = [
            {"start": 0.0, "end": 1.0, "subject_x": 200.0},
            {"start": 1.0, "end": 2.0, "subject_x": 1000.0},
        ]
        events = reframe_segments_to_events(segs, 1920, 1080, 30.0)
        scene_changes = [e for e in events if e.get("scene_change")]
        # Exactly two boundaries: t=0.0 and t=1.0.
        assert len(scene_changes) == 2
        assert scene_changes[0]["t"] == pytest.approx(0.0)
        assert scene_changes[1]["t"] == pytest.approx(1.0)


# ───────────────────────── (3) ffmpeg expression builder ─────────────


class TestBuildXExpression:
    @staticmethod
    def _src_w_crop_w():
        # 1920x1080 source → 9:16 crop_w = 1080 * 9/16 = 607.5 → 608.
        return 1920, 608

    def test_empty_segments_returns_centre_crop(self):
        src_w, crop_w = self._src_w_crop_w()
        out = _build_x_expression([], src_w, crop_w)
        # Centre = (src_w - crop_w) // 2 = (1920-608)//2 = 656.
        assert out == "656"

    def test_legacy_step_emits_held_x_per_segment(self):
        src_w, crop_w = self._src_w_crop_w()
        segs = [
            {"start": 0.0, "end": 2.0, "subject_x": 100.0},
            {"start": 2.0, "end": 4.0, "subject_x": 1500.0},
        ]
        out = _build_x_expression(segs, src_w, crop_w)
        # Each held interval should be a constant-x term, NOT a lerp.
        # Look for "<x>*gte(t,...)*lt(t,...)" patterns.
        held_pattern = re.compile(r"\d+\*gte\(t,[\d.]+\)\*lt\(t,[\d.]+\)")
        held_terms = held_pattern.findall(out)
        # Three intervals: seg1 hold, zero-width step, seg2 hold.
        assert len(held_terms) >= 2
        # NO lerp terms when both endpoints have the same x.
        assert "(0+(0)*" not in out

    def test_tracking_emits_piecewise_linear_lerp(self):
        src_w, crop_w = self._src_w_crop_w()
        segs = [{
            "start": 0.0, "end": 2.0, "subject_x": 500.0,
            "motion_path": [(0.0, 500.0), (1.0, 1000.0), (2.0, 1500.0)],
        }]
        out = _build_x_expression(segs, src_w, crop_w)
        # Two lerp intervals: [0,1) and [1,2).
        lerp_pattern = re.compile(
            r"\(\d+\+\(\d+\)\*\(t-[\d.]+\)/\([\d.]+\)\)"
            r"\*gte\(t,[\d.]+\)\*lt\(t,[\d.]+\)"
        )
        assert len(lerp_pattern.findall(out)) == 2

    def test_uses_half_open_gte_lt_intervals(self):
        src_w, crop_w = self._src_w_crop_w()
        segs = [{
            "start": 0.0, "end": 2.0, "subject_x": 500.0,
            "motion_path": [(0.0, 500.0), (1.0, 700.0)],
        }]
        out = _build_x_expression(segs, src_w, crop_w)
        assert "gte(t," in out
        assert "lt(t," in out
        # Ensure the legacy `between(t, …)` syntax is gone.
        assert "between(" not in out

    def test_emits_tail_hold_past_last_keypoint(self):
        src_w, crop_w = self._src_w_crop_w()
        segs = [{
            "start": 0.0, "end": 2.0, "subject_x": 500.0,
            "motion_path": [(0.0, 500.0), (2.0, 1000.0)],
        }]
        out = _build_x_expression(segs, src_w, crop_w)
        # Last term: <x_last>*gte(t,2.000) with no `lt` clause.
        tail = out.split("+")[-1]
        assert tail.endswith("*gte(t,2.000)")
        assert "*lt(" not in tail

    def test_clamps_crop_x_inside_source_frame(self):
        src_w, crop_w = self._src_w_crop_w()  # src=1920, crop=608, half=304
        # Subject far off-screen left → crop_x clamps to 0.
        segs = [{
            "start": 0.0, "end": 1.0, "subject_x": -500.0,
            "motion_path": [(0.0, -500.0), (1.0, -400.0)],
        }]
        out = _build_x_expression(segs, src_w, crop_w)
        # Both endpoints clamp to 0 → produces a held 0 term.
        assert "0*gte(t,0.000)*lt(t,1.000)" in out

    def test_clamps_crop_x_to_max_x(self):
        src_w, crop_w = self._src_w_crop_w()
        # Subject far past frame right → crop_x clamps to max_x.
        max_x = src_w - crop_w  # 1312
        segs = [{
            "start": 0.0, "end": 1.0, "subject_x": 9999.0,
            "motion_path": [(0.0, 9999.0), (1.0, 9999.0)],
        }]
        out = _build_x_expression(segs, src_w, crop_w)
        assert f"{max_x}*gte(t,0.000)*lt(t,1.000)" in out

    def test_no_recursion_depth_blowup_on_many_segments(self):
        # Confirm the flat-sum scheme handles thousands of keypoints
        # without nesting (the original `between(...)` chain blew
        # ffmpeg's parser at ~96 levels).
        src_w, crop_w = self._src_w_crop_w()
        segs = []
        for i in range(500):
            segs.append({
                "start": float(i), "end": float(i + 1),
                "subject_x": 600.0 + (i % 3) * 100.0,
            })
        out = _build_x_expression(segs, src_w, crop_w)
        # No nested `if(`, parens balance.
        assert "if(" not in out
        assert out.count("(") == out.count(")")

    def test_single_keypoint_degenerates_to_hold(self):
        src_w, crop_w = self._src_w_crop_w()
        # An empty-but-non-None motion_path drops to (start, subject_x)
        # / (end, subject_x); a single-keypoint stream only shows up
        # when every segment has end <= start. Force that here.
        segs = [{"start": 5.0, "end": 5.0, "subject_x": 800.0}]
        out = _build_x_expression(segs, src_w, crop_w)
        # No usable keypoints → centre crop fallback.
        assert out == str((src_w - crop_w) // 2)

    def test_smoothstep_ease_appears_as_multi_segment_ramp(self):
        src_w, crop_w = self._src_w_crop_w()
        segs = [
            {"start": 0.0, "end": 1.0, "subject_x": 200.0},
            {
                "start": 1.0, "end": 2.0, "subject_x": 1700.0,
                "ease_in_ms": 200,
            },
        ]
        out = _build_x_expression(segs, src_w, crop_w)
        # The ease window contributes >1 lerp segment between t=1.000
        # and t=1.200 — count distinct gte/lt intervals starting in
        # [1.0, 1.2).
        intervals = re.findall(r"gte\(t,(1\.\d{3})\)\*lt\(t,", out)
        in_ease = [t for t in intervals if 1.0 <= float(t) < 1.2]
        assert len(in_ease) >= 3  # smoothstep sampled at ~30 Hz
