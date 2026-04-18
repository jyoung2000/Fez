"""Phase 9 — unit tests for the AutoFlip parity metric library.

The metric module is intentionally numpy-free so it can be tested in
a minimal sandbox. These tests cover every metric with positive,
negative, boundary, and edge-case inputs, plus the
``score_fixture`` aggregator and the ``extract_switch_times`` helper.

If a metric definition drifts (e.g. someone tightens the tolerance,
swaps the percentage / pixel coordinate convention, or renames a
field on the dict outputs) one of these tests catches it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from backend.services.autoflip_parity_metrics import (
    ALL_METRICS,
    downbeat_snap_error,
    extract_switch_times,
    face_centroid_in_thirds_rate,
    hud_preservation_rate,
    max_acceleration,
    max_jerk,
    overlap_count,
    required_region_miss_rate,
    score_fixture,
    sub_second_switch_recall,
)


# ─────────────────── Metric 1: sub_second_switch_recall ───────────────────

class TestSubSecondSwitchRecall:
    def test_perfect_match(self):
        assert sub_second_switch_recall([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0

    def test_perfect_match_with_jitter_within_tolerance(self):
        assert sub_second_switch_recall([1.0, 2.0, 3.0], [1.4, 1.6, 3.4]) == 1.0

    def test_partial_match(self):
        # 2 of 3 expected switches matched (3.0 has nothing within 0.5 s).
        assert sub_second_switch_recall([1.0, 2.0, 3.0], [1.0, 2.0, 5.0]) == 2.0 / 3.0

    def test_no_actual_switches(self):
        assert sub_second_switch_recall([1.0, 2.0], []) == 0.0

    def test_no_expected_switches_returns_one(self):
        # Vacuous-true: nothing to miss. Important for the vlog fixture
        # which has zero speaker changes.
        assert sub_second_switch_recall([], [1.0, 2.0]) == 1.0

    def test_both_empty(self):
        assert sub_second_switch_recall([], []) == 1.0

    def test_tolerance_just_outside(self):
        # 0.51 s lag with default 0.5 s tolerance → miss.
        assert sub_second_switch_recall([1.0], [1.51]) == 0.0

    def test_tolerance_just_inside(self):
        assert sub_second_switch_recall([1.0], [1.5]) == 1.0

    def test_custom_tolerance(self):
        assert sub_second_switch_recall([1.0], [2.0], tolerance_sec=2.0) == 1.0
        assert sub_second_switch_recall([1.0], [2.0], tolerance_sec=0.1) == 0.0


# ─────────────────── Metric 2: overlap_count ───────────────────

class TestOverlapCount:
    def test_clean_contiguous(self):
        assert overlap_count([(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]) == 0

    def test_one_overlap(self):
        assert overlap_count([(0.0, 1.5), (1.0, 2.0)]) == 1

    def test_multiple_overlaps(self):
        assert overlap_count([(0, 2), (1, 3), (2, 4)]) == 2

    def test_empty(self):
        assert overlap_count([]) == 0

    def test_single(self):
        assert overlap_count([(0.0, 1.0)]) == 0

    def test_attr_objects_supported(self):
        @dataclass
        class S:
            start: float
            end: float

        segments = [S(0, 1), S(1, 2), S(1.5, 3)]  # second overlaps third
        assert overlap_count(segments) == 1

    def test_floating_point_epsilon_tolerance(self):
        # 0.5 + 0.5 → 1.0 exactly is NOT an overlap.
        assert overlap_count([(0.0, 0.5 + 0.5), (1.0, 2.0)]) == 0


# ─────────────────── Metric 3: max_acceleration ───────────────────

class TestMaxAcceleration:
    def test_constant_signal(self):
        assert max_acceleration([5.0, 5.0, 5.0, 5.0]) == 0.0

    def test_linear_ramp(self):
        # x[i+1] - 2*x[i] + x[i-1] = 0 for any linear sequence.
        assert max_acceleration([0, 1, 2, 3, 4, 5]) == 0.0

    def test_step_change(self):
        # 0,0,0 then jump to 10
        # at i=1: |0 - 2*0 + 0| = 0
        # at i=2: |10 - 2*0 + 0| = 10
        # at i=3: |10 - 2*10 + 0| = 10
        assert max_acceleration([0, 0, 0, 10, 10, 10]) == 10.0

    def test_short_input(self):
        assert max_acceleration([]) == 0.0
        assert max_acceleration([1.0]) == 0.0
        assert max_acceleration([1.0, 2.0]) == 0.0

    def test_negative_values_handled(self):
        # n=5 → iterations at i=1, 2, 3:
        #   i=1: |0 - 2*0  + 10|  = 10
        #   i=2: |-10 - 2*0 + 0|  = 10
        #   i=3: |0 - 2*(-10) + 0| = 20
        # max = 20
        assert max_acceleration([10, 0, -10, 0, 10]) == 20.0


# ─────────────────── Metric 4: max_jerk ───────────────────

class TestMaxJerk:
    def test_constant_signal(self):
        assert max_jerk([5.0] * 6) == 0.0

    def test_linear_ramp(self):
        assert max_jerk([0, 1, 2, 3, 4, 5]) == 0.0

    def test_quadratic_ramp(self):
        # x = i² → constant 2nd diff, zero jerk.
        xs = [i * i for i in range(8)]
        assert max_jerk(xs) == 0.0

    def test_short_input(self):
        assert max_jerk([]) == 0.0
        assert max_jerk([1.0, 2.0, 3.0]) == 0.0

    def test_step_signal(self):
        # max_jerk runs i in range(2, n-1). For [0,0,0,10,10] (n=5):
        #   i=2: |x[3] - 3*x[2] + 3*x[1] - x[0]| = |10 - 0 + 0 - 0| = 10
        #   i=3: |x[4] - 3*x[3] + 3*x[2] - x[1]| = |10 - 30 + 0 - 0| = 20
        # max = 20.
        assert max_jerk([0, 0, 0, 10, 10]) == 20.0


# ─────────────────── Metric 5: required_region_miss_rate ───────────────────

class TestRequiredRegionMissRate:
    def test_all_regions_inside(self):
        # crop centered at 50, width 30 → window [35, 65]
        regions = [[(40.0, 60.0)] for _ in range(3)]
        assert required_region_miss_rate([50, 50, 50], regions, 30.0) == 0.0

    def test_all_regions_outside(self):
        # Region at (10, 30) — entirely outside crop window [35, 65]
        regions = [[(10.0, 30.0)] for _ in range(3)]
        assert required_region_miss_rate([50, 50, 50], regions, 30.0) == 1.0

    def test_partial_misses(self):
        # 3 frames, 1 missing
        regions = [
            [(40.0, 60.0)],   # in
            [(10.0, 30.0)],   # out
            [(40.0, 60.0)],   # in
        ]
        assert required_region_miss_rate([50, 50, 50], regions, 30.0) == 1.0 / 3.0

    def test_empty_inputs(self):
        assert required_region_miss_rate([], [], 30.0) == 0.0
        assert required_region_miss_rate([50], [], 30.0) == 0.0
        # Frames with no required region are not counted.
        assert required_region_miss_rate([50, 50], [[], []], 30.0) == 0.0

    def test_multiple_regions_one_miss(self):
        # First region in-frame, second region out-of-frame → miss
        regions = [[(45.0, 55.0), (10.0, 20.0)]]
        assert required_region_miss_rate([50], regions, 30.0) == 1.0

    def test_uses_crop_centered_window(self):
        # crop centered at 80, width 30 → window [65, 95]
        regions = [[(70.0, 90.0)]]  # inside
        assert required_region_miss_rate([80], regions, 30.0) == 0.0


# ─────────────────── Metric 6: downbeat_snap_error ───────────────────

class TestDownbeatSnapError:
    def test_perfect_snap(self):
        result = downbeat_snap_error([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert result["mean_error_ms"] == 0.0
        assert result["max_error_ms"] == 0.0
        assert result["snap_rate"] == 1.0
        assert result["snapped"] == 3
        assert result["count"] == 3

    def test_off_by_100ms(self):
        result = downbeat_snap_error([1.1, 2.1], [1.0, 2.0])
        assert result["max_error_ms"] == pytest.approx(100.0, abs=0.01)
        assert result["snap_rate"] == 1.0  # 100 ms < 200 ms tolerance

    def test_off_by_300ms(self):
        result = downbeat_snap_error([1.3], [1.0, 2.0])
        assert result["max_error_ms"] == pytest.approx(300.0, abs=0.01)
        assert result["snap_rate"] == 0.0

    def test_partial_snap(self):
        result = downbeat_snap_error([1.0, 1.4], [1.0, 2.0])
        # 1.0 → 0 ms, 1.4 → 400 ms (nearest beat is 1.0). 1 of 2 snapped.
        assert result["snap_rate"] == 0.5
        assert result["snapped"] == 1
        assert result["count"] == 2

    def test_empty_switches(self):
        result = downbeat_snap_error([], [1.0])
        assert result["mean_error_ms"] is None
        assert result["max_error_ms"] is None
        assert result["snap_rate"] == 0.0
        assert result["count"] == 0

    def test_empty_beat_grid(self):
        result = downbeat_snap_error([1.0], [])
        assert result["mean_error_ms"] is None
        assert result["snap_rate"] == 0.0

    def test_custom_tolerance(self):
        # 300 ms gap with 500 ms tolerance → snapped
        result = downbeat_snap_error([1.3], [1.0], snap_tolerance_ms=500.0)
        assert result["snap_rate"] == 1.0


# ─────────────────── Metric 7: hud_preservation_rate ───────────────────

class TestHudPreservationRate:
    def test_hud_inside_crop(self):
        # crop centered at 50, width 90 → window [5, 95]
        # HUD at (40, 80, 20, 10) → spans x=40..60 → fully inside
        rate = hud_preservation_rate([50, 50], 90.0, [(40.0, 80.0, 20.0, 10.0)])
        assert rate == 1.0

    def test_hud_outside_crop(self):
        # crop window [5, 35] (centered at 20, width 30)
        # HUD at x=70..90 → entirely outside
        rate = hud_preservation_rate([20, 20], 30.0, [(70.0, 5.0, 20.0, 10.0)])
        assert rate == 0.0

    def test_hud_partial_visible_below_threshold(self):
        # crop window [5, 35]
        # HUD at x=20..50 (width=30); visible portion = [20, 35] = 15 of 30 = 50 %
        # Below 80 % → not preserved
        rate = hud_preservation_rate([20, 20], 30.0, [(20.0, 5.0, 30.0, 10.0)])
        assert rate == 0.0

    def test_hud_partial_visible_above_threshold(self):
        # crop window [10, 90] (centered at 50, width 80)
        # HUD at x=20..40 → 100 % visible
        rate = hud_preservation_rate([50, 50], 80.0, [(20.0, 5.0, 20.0, 10.0)])
        assert rate == 1.0

    def test_no_hud_returns_one(self):
        # Vacuous-true: no HUD constraints to fail.
        assert hud_preservation_rate([50, 50], 30.0, []) == 1.0

    def test_no_frames_returns_zero(self):
        assert hud_preservation_rate([], 30.0, [(0.0, 0.0, 10.0, 10.0)]) == 0.0

    def test_partial_frames_pass(self):
        # 2 crops at 50, 2 crops at 0 (HUD at x=70..90 only visible for the 50 crops if window is wide enough)
        # crop window centered at 50 with width 90 → [5, 95] → visible
        # crop window centered at 0 with width 90 → [-45, 45] → HUD outside
        rate = hud_preservation_rate([50, 50, 0, 0], 90.0, [(70.0, 5.0, 20.0, 10.0)])
        assert rate == 0.5

    def test_custom_min_visible_fraction(self):
        # Same as test_hud_partial_visible_below_threshold (50 %) but
        # with a 0.4 threshold → preserved.
        rate = hud_preservation_rate(
            [20], 30.0, [(20.0, 5.0, 30.0, 10.0)], min_visible_fraction=0.4,
        )
        assert rate == 1.0


# ─────────────────── Metric 8: face_centroid_in_thirds_rate ───────────────────

class TestFaceCentroidInThirdsRate:
    def test_all_in_thirds(self):
        rate = face_centroid_in_thirds_rate([1 / 3, 0.32, 0.34, 0.30])
        assert rate == 1.0

    def test_all_out_of_thirds(self):
        rate = face_centroid_in_thirds_rate([0.7, 0.8, 0.5])
        assert rate == 0.0

    def test_partial(self):
        rate = face_centroid_in_thirds_rate([1 / 3, 0.7])
        assert rate == 0.5

    def test_empty(self):
        assert face_centroid_in_thirds_rate([]) == 0.0

    def test_custom_tolerance(self):
        # 0.5 with default 0.10 tolerance → out (|0.5 - 0.333| = 0.167 > 0.10)
        assert face_centroid_in_thirds_rate([0.5]) == 0.0
        # With 0.2 tolerance → in
        assert face_centroid_in_thirds_rate([0.5], tolerance=0.2) == 1.0

    def test_custom_target(self):
        # Use the lower-third line (y = 2/3) instead of upper-third.
        rate = face_centroid_in_thirds_rate(
            [2 / 3, 0.65, 0.70], target_thirds=2 / 3,
        )
        assert rate == 1.0


# ─────────────────── score_fixture aggregator ───────────────────

class TestScoreFixture:
    def test_runs_only_requested_metrics(self):
        result = score_fixture(
            metrics_to_run=["overlap_count"],
            segment_spans=[(0, 1), (1, 2)],
        )
        assert set(result.keys()) == {"overlap_count"}
        assert result["overlap_count"] == 0

    def test_unknown_metric_returns_error_dict(self):
        result = score_fixture(
            metrics_to_run=["bogus_metric"],
        )
        assert "bogus_metric" in result
        assert isinstance(result["bogus_metric"], dict)
        assert "error" in result["bogus_metric"]

    def test_all_metrics_one_call(self):
        result = score_fixture(
            metrics_to_run=list(ALL_METRICS),
            expected_switches=[1.0],
            actual_switches=[1.0],
            segment_spans=[(0, 1), (1, 2)],
            crop_centers=[50, 50, 50, 50],
            required_regions_per_frame=[[(40, 60)]] * 4,
            crop_width_pct=30.0,
            beat_grid=[1.0, 2.0],
            hud_zones=[(45, 5, 10, 10)],
            face_y_in_crop_normalized=[1 / 3, 1 / 3],
        )
        for m in ALL_METRICS:
            assert m in result, f"missing metric {m}"

    def test_required_region_with_no_crop_width_returns_none(self):
        result = score_fixture(
            metrics_to_run=["required_region_miss_rate"],
            crop_centers=[50],
            required_regions_per_frame=[[(40, 60)]],
        )
        assert result["required_region_miss_rate"] is None

    def test_hud_with_no_crop_width_returns_none(self):
        result = score_fixture(
            metrics_to_run=["hud_preservation_rate"],
            crop_centers=[50],
            hud_zones=[(0, 0, 10, 10)],
        )
        assert result["hud_preservation_rate"] is None


# ─────────────────── extract_switch_times helper ───────────────────

class TestExtractSwitchTimes:
    def _seg(self, slot, start):
        return type("S", (), {"active_slot": slot, "start": start})()

    def test_no_switches(self):
        segments = [self._seg(0, 0.0), self._seg(0, 1.0), self._seg(0, 2.0)]
        assert extract_switch_times(segments) == []

    def test_one_switch(self):
        segments = [self._seg(0, 0.0), self._seg(1, 2.0)]
        assert extract_switch_times(segments) == [2.0]

    def test_multiple_switches(self):
        segments = [
            self._seg(0, 0.0),
            self._seg(0, 1.0),
            self._seg(1, 2.0),
            self._seg(0, 3.0),
            self._seg(0, 4.0),
            self._seg(1, 5.0),
        ]
        assert extract_switch_times(segments) == [2.0, 3.0, 5.0]

    def test_none_slot_does_not_count(self):
        # First segment has no active_slot → other segments count
        # against None on the second iteration, generating one switch.
        segments = [self._seg(None, 0.0), self._seg(0, 1.0)]
        # The first iteration has last_slot=sentinel, second has last=None
        # which differs from 0 → switches at t=1.
        assert extract_switch_times(segments) == [1.0]

    def test_empty(self):
        assert extract_switch_times([]) == []
