"""Task 2 — score_timeline pipes the four missing real-content metrics.

Validates that ``backend.scripts.compare_autoflip_vs_clipai.score_timeline``:
  * accepts the four optional kwargs (face / saliency / text / identity);
  * returns ``None`` for a metric whose input was ``None`` (so the
    verdict logic doesn't grade missing data as a free pass);
  * back-compat: callers passing no kwargs still work.

Run:  pytest tests/qa/test_score_timeline_extended.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _events(n=20, crop_cx=0.5):
    return [
        {"t": i * 0.1, "crop_cx": crop_cx, "crop_cy": 0.5,
         "crop_w": 0.5625, "crop_h": 1.0, "scene_change": False}
        for i in range(n)
    ]


def test_score_timeline_back_compat_no_kwargs():
    """Legacy callers (no Task 2 kwargs) still work."""
    from backend.scripts.compare_autoflip_vs_clipai import score_timeline
    out = score_timeline(_events())
    assert out["face_clipping_rate"] is None
    assert out["saliency_in_crop_fraction"] is None
    assert out["text_region_clipping_rate"] is None
    assert out["identity_switch_count"] is None
    # Back-compat keys still present.
    assert "max_acceleration" in out and "max_jerk" in out


def test_score_timeline_returns_none_when_data_missing():
    """Explicit None inputs should produce explicit None outputs."""
    from backend.scripts.compare_autoflip_vs_clipai import score_timeline
    out = score_timeline(
        _events(),
        face_bboxes_per_frame=None,
        saliency_peaks_per_second=None,
        text_regions_per_frame=None,
        identity_timeline=None,
    )
    assert out["face_clipping_rate"] is None
    assert out["saliency_in_crop_fraction"] is None
    assert out["text_region_clipping_rate"] is None
    assert out["identity_switch_count"] is None


def test_score_timeline_distinguishes_zero_from_none():
    """Empty-list inputs measure zero; None inputs return None.

    These two cases must not collapse — silent missing data masquerading
    as ``0.0`` is the failure mode this task fixes.
    """
    from backend.scripts.compare_autoflip_vs_clipai import score_timeline
    n = 20
    out_zero = score_timeline(
        _events(n=n),
        face_bboxes_per_frame=[[] for _ in range(n)],
        text_regions_per_frame=[[] for _ in range(n)],
    )
    out_none = score_timeline(
        _events(n=n),
        face_bboxes_per_frame=None,
        text_regions_per_frame=None,
    )
    assert out_zero["face_clipping_rate"] == 0.0
    assert out_zero["text_region_clipping_rate"] == 0.0
    assert out_none["face_clipping_rate"] is None
    assert out_none["text_region_clipping_rate"] is None


def test_score_timeline_face_clipping_pipes_through():
    """A face whose bbox extends well past the crop edge counts as clipped."""
    from backend.scripts.compare_autoflip_vs_clipai import score_timeline
    # crop centered at 50%, crop width 56.25% → crop = [21.875, 78.125].
    # A face anchored at x=10 with width=20 → extends to x=30; its left
    # edge at x=10 sits 11.875% outside the crop → clipped.
    n = 10
    faces = [[(10.0, 40.0, 20.0, 20.0)] for _ in range(n)]
    out = score_timeline(
        _events(n=n),
        face_bboxes_per_frame=faces,
    )
    assert out["face_clipping_rate"] is not None
    assert out["face_clipping_rate"] > 0.0


def test_score_timeline_saliency_in_crop():
    """Peaks inside the crop window register; peaks outside do not."""
    from backend.scripts.compare_autoflip_vs_clipai import score_timeline
    n = 5
    # All peaks at x=50 (= crop center) → 100 % in-crop.
    in_peaks = [[(50.0, 50.0, 1.0)] for _ in range(n)]
    # All peaks at x=5 → way outside the crop → 0 % in-crop.
    out_peaks = [[(5.0, 50.0, 1.0)] for _ in range(n)]
    in_score = score_timeline(
        _events(n=n), saliency_peaks_per_second=in_peaks,
    )["saliency_in_crop_fraction"]
    out_score = score_timeline(
        _events(n=n), saliency_peaks_per_second=out_peaks,
    )["saliency_in_crop_fraction"]
    assert in_score == pytest.approx(1.0)
    assert out_score == pytest.approx(0.0)


def test_score_timeline_identity_switches_counted():
    """A switch fires when a slot's bbox swaps to a non-overlapping bbox."""
    from backend.scripts.compare_autoflip_vs_clipai import score_timeline
    timeline = [
        # Slot 0 holds steady.
        {"t": 0.0, "slots": {0: (10, 10, 20, 20)}},
        {"t": 0.5, "slots": {0: (10, 10, 20, 20)}},
        # Slot 0 jumps far away → 1 switch.
        {"t": 1.0, "slots": {0: (200, 200, 20, 20)}},
        # Holds at the new position.
        {"t": 1.5, "slots": {0: (200, 200, 20, 20)}},
        # Jumps back → another switch (count = 2).
        {"t": 2.0, "slots": {0: (10, 10, 20, 20)}},
    ]
    out = score_timeline(_events(), identity_timeline=timeline)
    assert out["identity_switch_count"] == 2


def test_score_timeline_empty_events_still_has_keys():
    """An empty event list still emits the four keys with None."""
    from backend.scripts.compare_autoflip_vs_clipai import score_timeline
    out = score_timeline([])
    for k in (
        "face_clipping_rate", "saliency_in_crop_fraction",
        "text_region_clipping_rate", "identity_switch_count",
    ):
        assert k in out and out[k] is None


def test_extract_per_frame_data_from_payload_missing_keys():
    """When the cache payload predates Task 2, all four are None."""
    from backend.scripts.compare_autoflip_vs_clipai import (
        extract_per_frame_data_from_payload,
    )
    payload = {"metadata": {}, "segments": []}
    out = extract_per_frame_data_from_payload(payload)
    assert out == {
        "face_bboxes_per_frame": None,
        "saliency_peaks_per_second": None,
        "text_regions_per_frame": None,
        "identity_timeline": None,
    }


# ──────────────────────────────────────────────────────────────────
# Task 4 — default zone + verdict downgrades on new metrics
# ──────────────────────────────────────────────────────────────────


class TestTargetZoneFallback:
    def test_target_zone_unknown_falls_back_to_default(self):
        from backend.scripts.compare_autoflip_vs_clipai import (
            TARGET_ZONES, target_zone_for,
        )
        zone = target_zone_for({"target_clipcontenttype": "some_made_up_type"})
        assert zone is TARGET_ZONES["default"]

    def test_target_zone_known_returns_specific(self):
        from backend.scripts.compare_autoflip_vs_clipai import (
            TARGET_ZONES, target_zone_for,
        )
        zone = target_zone_for({"target_clipcontenttype": "multi_speaker_panel"})
        assert zone is TARGET_ZONES["multi_speaker_panel"]


class TestVerdictDowngrades:
    def _good_hold(self):
        return {
            "n_segments": 10, "median_hold_sec": 4.0,
            "segments_under_1s_rate": 0.02,
        }

    def test_pass_with_all_metrics_in_zone(self):
        from backend.scripts.compare_autoflip_vs_clipai import (
            TARGET_ZONES, verdict_for,
        )
        m = self._good_hold() | {
            "face_clipping_rate": 0.01, "saliency_in_crop_fraction": 0.95,
            "text_region_clipping_rate": 0.0, "identity_switch_count": 0,
        }
        assert verdict_for(m, TARGET_ZONES["multi_speaker_panel"]) == "PASS"

    def test_downgrades_on_face_clipping(self):
        from backend.scripts.compare_autoflip_vs_clipai import (
            TARGET_ZONES, verdict_for,
        )
        m = self._good_hold() | {
            "face_clipping_rate": 0.15, "saliency_in_crop_fraction": 0.95,
            "text_region_clipping_rate": 0.0, "identity_switch_count": 0,
        }
        assert verdict_for(m, TARGET_ZONES["multi_speaker_panel"]) == "MARGINAL"

    def test_downgrades_on_low_saliency(self):
        from backend.scripts.compare_autoflip_vs_clipai import (
            TARGET_ZONES, verdict_for,
        )
        m = self._good_hold() | {
            "face_clipping_rate": 0.0, "saliency_in_crop_fraction": 0.50,
            "text_region_clipping_rate": 0.0, "identity_switch_count": 0,
        }
        assert verdict_for(m, TARGET_ZONES["multi_speaker_panel"]) == "MARGINAL"

    def test_unknown_only_when_no_events(self):
        from backend.scripts.compare_autoflip_vs_clipai import (
            TARGET_ZONES, verdict_for,
        )
        # n_segments=0 → UNKNOWN.
        assert verdict_for(
            {"n_segments": 0}, TARGET_ZONES["default"],
        ) == "UNKNOWN"

    def test_default_zone_pass_for_unknown_content_type(self):
        """Pre-Task-4 this clip would have been UNKNOWN. Now the
        default zone catches it."""
        from backend.scripts.compare_autoflip_vs_clipai import (
            target_zone_for, verdict_for,
        )
        clip = {"target_clipcontenttype": "made_up_genre"}
        zone = target_zone_for(clip)
        m = {
            "n_segments": 8, "median_hold_sec": 3.0,
            "segments_under_1s_rate": 0.05,
            "face_clipping_rate": 0.02, "saliency_in_crop_fraction": 0.85,
            "text_region_clipping_rate": 0.01, "identity_switch_count": 1,
        }
        assert verdict_for(m, zone) == "PASS"


def test_extract_per_frame_data_from_payload_present_keys():
    """When the cache payload includes the four fields, they pass through."""
    from backend.scripts.compare_autoflip_vs_clipai import (
        extract_per_frame_data_from_payload,
    )
    payload = {
        "metadata": {}, "segments": [],
        "face_bboxes_per_frame": [[(10.0, 10.0, 20.0, 20.0)]],
        "saliency_peaks_per_second": [[(50.0, 50.0, 1.0)]],
        "text_regions_per_frame": [[]],
        "identity_timeline": [{"t": 0.0, "slots": {0: (10, 10, 20, 20)}}],
    }
    out = extract_per_frame_data_from_payload(payload)
    assert out["face_bboxes_per_frame"] == [[(10.0, 10.0, 20.0, 20.0)]]
    assert len(out["saliency_peaks_per_second"]) == 1
    assert len(out["text_regions_per_frame"]) == 1
    assert out["identity_timeline"][0]["t"] == 0.0
