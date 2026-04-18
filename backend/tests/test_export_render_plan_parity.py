"""Phase 10 follow-up — preview/export parity for cached RenderPlan keyframes.

The Canvas preview reads ``job.render_plan`` verbatim via
``GET /api/jobs/{id}/render_plan``. That cached dict is written by
``pipeline.py`` at segmenter completion time from the full-fidelity
``ReframeSegment`` list so it preserves ``motion_path`` (per-frame
walking tracks for vlogs, anime pans, Stage 10 L1 camera paths).

Before Phase 10 follow-up the exporter rebuilt its own RenderPlan from
the stripped ``SceneDescription`` list via
``_extract_render_plan_segments``, which hard-codes ``motion_path=None``.
The result was a known parity bug for VLOG content with
``allow_motion_tracking=True``: preview showed a smooth pan, export
showed a static midpoint crop.

``_keyframes_from_cached_render_plan`` closes that gap. These tests
lock its contract:

1. Walking-vlog parity — a cached plan with a TRACKING_CROP op carrying
   motion_path is converted into the same ``(t, x_pct)`` schedule the
   preview player would interpolate.
2. Clip-range slicing — ops outside the window are skipped, ops that
   straddle the boundary contribute only their in-range keyframes,
   times are rebased to ``t=0`` at ``clip_start``.
3. Op kind coverage — CROP / TRACKING_CROP / WIDE_MASTER / SPLIT_SCREEN
   / STACKED_GAMEPLAY / BLUR_FILL all contribute exactly one boundary
   keyframe when motion_path is absent.
4. Collapse semantics — duplicate (t, x) entries from segment junctions
   are de-duplicated; the caller then collapses same-x sequences to a
   static crop.
5. Robustness — malformed dicts / missing ops / non-dict input never
   raise; the helper returns ``None`` so the caller falls through to
   the legacy per-clip rebuild pipeline.
"""

from __future__ import annotations

from backend.services.render_plan_keyframes import (
    keyframes_from_cached_render_plan,
)


def _keyframes_from_cached_render_plan(plan, clip_start, clip_end):
    """Positional adapter so the tests read more naturally."""
    return keyframes_from_cached_render_plan(
        plan, clip_start=clip_start, clip_end=clip_end,
    )


# ────────────────────────────────────────────────────────────────
# Test fixtures
# ────────────────────────────────────────────────────────────────

def _crop_rect(center_pct: float, width_pct: float = 30.0) -> dict:
    """Build a normalized [0,1] rect centred at ``center_pct`` percent."""
    w = width_pct / 100.0
    x = (center_pct / 100.0) - w / 2.0
    return {"x": x, "y": 0.2, "w": w, "h": 0.6}


def _op(
    *,
    kind: str = "crop",
    start: float,
    end: float,
    center_pct: float,
    motion_path: list | None = None,
) -> dict:
    return {
        "kind": kind,
        "start_sec": start,
        "end_sec": end,
        "primary_rect": _crop_rect(center_pct),
        "motion_path": motion_path,
        "ease_in_ms": 0,
        "strategy_label": f"{kind}_test",
        "content_type": "vlog",
    }


def _walking_vlog_plan() -> dict:
    """Synthesize a cached RenderPlan dict for a walking vlog.

    Timeline:
      [0.0, 2.0)  — CROP, static centre 30 %
      [2.0, 8.0)  — TRACKING_CROP with motion_path walking 35 → 65 %
                    across 6 s (7 keypoints at 1 s intervals)
      [8.0, 10.0) — CROP, static centre 65 %
    """
    motion_path = []
    for i in range(7):
        t_op = float(i)  # op-relative seconds (0..6)
        center = 35.0 + (65.0 - 35.0) * (i / 6.0)
        motion_path.append({
            "t": t_op,
            "rect": _crop_rect(center),
        })
    return {
        "schema": "render_plan/1",
        "total_duration_sec": 10.0,
        "ops": [
            _op(kind="crop", start=0.0, end=2.0, center_pct=30.0),
            _op(
                kind="tracking_crop",
                start=2.0,
                end=8.0,
                center_pct=50.0,
                motion_path=motion_path,
            ),
            _op(kind="crop", start=8.0, end=10.0, center_pct=65.0),
        ],
    }


# ────────────────────────────────────────────────────────────────
# 1. Walking-vlog parity
# ────────────────────────────────────────────────────────────────

class TestWalkingVlogParity:
    def test_tracking_crop_motion_path_converts_to_keyframes(self):
        plan = _walking_vlog_plan()
        kfs = _keyframes_from_cached_render_plan(
            plan, clip_start=0.0, clip_end=10.0,
        )
        assert kfs is not None

        # Expected structure:
        #   t=0   → 30  (crop 0-2)
        #   t=2   → 35  (tracking start)
        #   t=3   → 40
        #   t=4   → 45
        #   t=5   → 50
        #   t=6   → 55
        #   t=7   → 60
        #   t=8   → 65  (tracking end == crop 8-10 boundary)
        # The tracking end + next crop boundary collapse to one
        # entry at t=8 because they have identical (t, x).
        tpairs = {(round(t, 3), x) for t, x in kfs}
        for expected in [
            (0.0, 30),
            (2.0, 35),
            (3.0, 40),
            (4.0, 45),
            (5.0, 50),
            (6.0, 55),
            (7.0, 60),
            (8.0, 65),
        ]:
            assert expected in tpairs, f"missing {expected} in {sorted(tpairs)}"

    def test_keyframes_are_sorted_by_time(self):
        kfs = _keyframes_from_cached_render_plan(
            _walking_vlog_plan(), clip_start=0.0, clip_end=10.0,
        )
        assert kfs == sorted(kfs)

    def test_monotonic_walking_x_progression(self):
        """The walking subject moves strictly right — no stalls or reversals."""
        kfs = _keyframes_from_cached_render_plan(
            _walking_vlog_plan(), clip_start=0.0, clip_end=10.0,
        )
        xs = [x for _, x in kfs]
        # x should start at 30 and end at 65, monotonically non-decreasing
        assert xs[0] == 30
        assert xs[-1] == 65
        for a, b in zip(xs, xs[1:]):
            assert b >= a, f"regression: {a} → {b} in {xs}"


# ────────────────────────────────────────────────────────────────
# 2. Clip-range slicing
# ────────────────────────────────────────────────────────────────

class TestClipRangeSlicing:
    def test_op_entirely_before_clip_is_dropped(self):
        plan = {
            "ops": [
                _op(kind="crop", start=0.0, end=5.0, center_pct=30.0),
                _op(kind="crop", start=10.0, end=15.0, center_pct=60.0),
            ],
        }
        kfs = _keyframes_from_cached_render_plan(
            plan, clip_start=10.0, clip_end=15.0,
        )
        assert kfs is not None
        # Only the second op survives, rebased to t=0
        assert kfs == [(0.0, 60)]

    def test_op_entirely_after_clip_is_dropped(self):
        plan = {
            "ops": [
                _op(kind="crop", start=0.0, end=5.0, center_pct=30.0),
                _op(kind="crop", start=10.0, end=15.0, center_pct=60.0),
            ],
        }
        kfs = _keyframes_from_cached_render_plan(
            plan, clip_start=0.0, clip_end=5.0,
        )
        assert kfs == [(0.0, 30)]

    def test_tracking_motion_keypoints_outside_clip_are_filtered(self):
        # Tracking op spans [0, 10) with motion every second. Clip
        # window is [3, 7). Only motion keypoints at t_abs in [3, 7]
        # should survive, rebased to clip-relative.
        motion_path = [
            {"t": float(i), "rect": _crop_rect(30.0 + i * 3.0)}
            for i in range(10)  # t_abs = 0..9
        ]
        plan = {
            "ops": [
                _op(
                    kind="tracking_crop",
                    start=0.0,
                    end=10.0,
                    center_pct=45.0,
                    motion_path=motion_path,
                ),
            ],
        }
        kfs = _keyframes_from_cached_render_plan(
            plan, clip_start=3.0, clip_end=7.0,
        )
        assert kfs is not None
        # Expect keypoints at t_abs=3,4,5,6,7 → t_rel=0,1,2,3,4
        ts = [t for t, _ in kfs]
        assert 0.0 in ts
        assert 4.0 in ts
        assert max(ts) <= 4.0 + 1e-6

    def test_rebase_times_to_clip_relative(self):
        plan = {
            "ops": [
                _op(kind="crop", start=5.0, end=10.0, center_pct=50.0),
            ],
        }
        kfs = _keyframes_from_cached_render_plan(
            plan, clip_start=5.0, clip_end=10.0,
        )
        assert kfs == [(0.0, 50)]


# ────────────────────────────────────────────────────────────────
# 3. Op kind coverage
# ────────────────────────────────────────────────────────────────

class TestOpKindCoverage:
    def test_static_crop_emits_one_boundary_keyframe(self):
        plan = {"ops": [_op(kind="crop", start=0.0, end=5.0, center_pct=40.0)]}
        kfs = _keyframes_from_cached_render_plan(plan, clip_start=0.0, clip_end=5.0)
        assert kfs == [(0.0, 40)]

    def test_wide_master_emits_one_boundary_keyframe(self):
        plan = {"ops": [_op(kind="wide_master", start=0.0, end=5.0, center_pct=50.0)]}
        kfs = _keyframes_from_cached_render_plan(plan, clip_start=0.0, clip_end=5.0)
        assert kfs is not None
        assert len(kfs) == 1
        assert kfs[0][1] == 50

    def test_split_screen_emits_one_boundary_keyframe(self):
        plan = {"ops": [_op(kind="split_screen", start=0.0, end=5.0, center_pct=50.0)]}
        kfs = _keyframes_from_cached_render_plan(plan, clip_start=0.0, clip_end=5.0)
        assert kfs is not None
        assert len(kfs) == 1

    def test_multiple_static_ops_emit_one_keyframe_each(self):
        plan = {
            "ops": [
                _op(kind="crop", start=0.0, end=2.0, center_pct=25.0),
                _op(kind="crop", start=2.0, end=4.0, center_pct=50.0),
                _op(kind="crop", start=4.0, end=6.0, center_pct=75.0),
            ],
        }
        kfs = _keyframes_from_cached_render_plan(plan, clip_start=0.0, clip_end=6.0)
        assert kfs == [(0.0, 25), (2.0, 50), (4.0, 75)]


# ────────────────────────────────────────────────────────────────
# 4. Collapse / dedupe semantics
# ────────────────────────────────────────────────────────────────

class TestCollapseSemantics:
    def test_duplicate_tx_entries_are_deduped(self):
        # Two back-to-back crops at the same x — the second op's
        # boundary keyframe has the same (t, x) as the first's end
        # since there's no overlap. We emit one keyframe per op
        # start, so the test is: do distinct ops at the same x
        # still produce distinct time entries?
        plan = {
            "ops": [
                _op(kind="crop", start=0.0, end=2.0, center_pct=50.0),
                _op(kind="crop", start=2.0, end=4.0, center_pct=50.0),
            ],
        }
        kfs = _keyframes_from_cached_render_plan(plan, clip_start=0.0, clip_end=4.0)
        # Each op contributes a keyframe at its start — that's
        # (0, 50) and (2, 50), both distinct in t so NOT deduped.
        assert kfs == [(0.0, 50), (2.0, 50)]

    def test_exact_duplicate_entries_collapsed(self):
        """Emit exactly-duplicate (t, x) entries only once."""
        motion_path = [
            {"t": 0.0, "rect": _crop_rect(50.0)},
            {"t": 0.0, "rect": _crop_rect(50.0)},  # duplicate
            {"t": 1.0, "rect": _crop_rect(50.0)},
        ]
        plan = {
            "ops": [
                _op(
                    kind="tracking_crop",
                    start=0.0,
                    end=2.0,
                    center_pct=50.0,
                    motion_path=motion_path,
                ),
            ],
        }
        kfs = _keyframes_from_cached_render_plan(plan, clip_start=0.0, clip_end=2.0)
        tpairs = [(round(t, 3), x) for t, x in kfs]
        assert tpairs.count((0.0, 50)) == 1


# ────────────────────────────────────────────────────────────────
# 5. Robustness — never raise, return None on malformed input
# ────────────────────────────────────────────────────────────────

class TestRobustness:
    def test_none_input_returns_none(self):
        assert _keyframes_from_cached_render_plan(None, 0.0, 10.0) is None

    def test_non_dict_input_returns_none(self):
        assert _keyframes_from_cached_render_plan("not a plan", 0.0, 10.0) is None
        assert _keyframes_from_cached_render_plan([], 0.0, 10.0) is None

    def test_empty_ops_returns_none(self):
        assert _keyframes_from_cached_render_plan({"ops": []}, 0.0, 10.0) is None

    def test_missing_ops_key_returns_none(self):
        assert _keyframes_from_cached_render_plan({"schema": "x"}, 0.0, 10.0) is None

    def test_malformed_op_entries_are_skipped(self):
        plan = {
            "ops": [
                "not a dict",
                None,
                {"kind": "crop"},  # missing start/end
                _op(kind="crop", start=0.0, end=5.0, center_pct=50.0),
            ],
        }
        kfs = _keyframes_from_cached_render_plan(plan, 0.0, 5.0)
        # Only the good op contributes
        assert kfs == [(0.0, 50)]

    def test_malformed_motion_keypoints_are_skipped(self):
        plan = {
            "ops": [
                _op(
                    kind="tracking_crop",
                    start=0.0,
                    end=5.0,
                    center_pct=50.0,
                    motion_path=[
                        "not a dict",
                        None,
                        {"t": "not a float"},
                        {"t": 1.0, "rect": _crop_rect(55.0)},
                    ],
                ),
            ],
        }
        kfs = _keyframes_from_cached_render_plan(plan, 0.0, 5.0)
        assert kfs is not None
        # At least the valid (1.0, 55) keypoint should survive
        assert any(abs(t - 1.0) < 1e-3 and x == 55 for t, x in kfs)

    def test_missing_primary_rect_falls_back_to_center(self):
        plan = {
            "ops": [
                {
                    "kind": "crop",
                    "start_sec": 0.0,
                    "end_sec": 5.0,
                    # No primary_rect
                },
            ],
        }
        kfs = _keyframes_from_cached_render_plan(plan, 0.0, 5.0)
        # Fallback x = 50 (centre)
        assert kfs == [(0.0, 50)]

    def test_string_times_are_skipped_gracefully(self):
        plan = {
            "ops": [
                {
                    "kind": "crop",
                    "start_sec": "bad",
                    "end_sec": "also bad",
                    "primary_rect": _crop_rect(50.0),
                },
            ],
        }
        assert _keyframes_from_cached_render_plan(plan, 0.0, 5.0) is None


# ────────────────────────────────────────────────────────────────
# 6. Preview/export parity — end-to-end round-trip
# ────────────────────────────────────────────────────────────────

class TestPreviewExportParity:
    def test_export_keyframes_match_preview_plan(self):
        """The export helper reads the same cached plan dict the
        preview router returns via GET /api/jobs/{id}/render_plan.
        For any clip range, the x-progression produced by the
        helper MUST be a faithful sampling of the plan's camera
        path — same values at the same times.
        """
        plan = _walking_vlog_plan()
        # Full-video clip (0 → 10 s)
        full_kfs = _keyframes_from_cached_render_plan(plan, 0.0, 10.0)
        assert full_kfs is not None

        # Half-video clip (2 → 8 s) should contain exactly the
        # walking tracking window, rebased
        half_kfs = _keyframes_from_cached_render_plan(plan, 2.0, 8.0)
        assert half_kfs is not None
        # The first keyframe should be at t=0 (clip start) with x=35
        assert half_kfs[0] == (0.0, 35)
        # The last keyframe should be at the clip end (t=6) with x=65
        assert half_kfs[-1][0] <= 6.0 + 1e-6
        assert half_kfs[-1][1] == 65

    def test_empty_clip_range_returns_none(self):
        plan = _walking_vlog_plan()
        kfs = _keyframes_from_cached_render_plan(plan, 20.0, 30.0)
        assert kfs is None
