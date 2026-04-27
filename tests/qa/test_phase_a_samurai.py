"""Phase A QA — SAMURAI tracker, dispatcher, OpenCV fallback, identity metric.

This suite is the gate Claude Code Opus 4.7 runs locally to decide
whether Phase A is shippable. It deliberately does NOT require an
actual GPU, the upstream ``sam2`` wheel, or model weights — everything
that touches the GPU is mocked.

The homelab-side bench (``compare_autoflip_vs_clipai``) is the OTHER
half of the gate and runs separately. This suite covers:

  1. The OpenCV fallback path keeps working.
  2. The dispatcher routes correctly per env var.
  3. ``SamuraiTracker`` raises ``RuntimeError`` when GPU is absent.
  4. The motion-aware memory layer behaves like the paper describes.
  5. The synthetic-occlusion fixture: a mocked SAMURAI tracker with a
     realistic occlusion-recovery profile keeps identity through a
     visual disappearance the OpenCV path would lose.
  6. The new ``identity_switch_count`` metric gives the right answer
     on canned timelines.

Run:  pytest tests/qa/test_phase_a_samurai.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

# Make ``backend`` importable when running from the repo root.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ──────────────────────────────────────────────────────────────────
# 1. OpenCV fallback path is unchanged
# ──────────────────────────────────────────────────────────────────


class TestOpenCvFallbackUnchanged:
    """The legacy callers (``dense_propagator`` etc.) still get the
    legacy-shape ``SlotTracker`` and the legacy ``update`` signature."""

    def test_slot_tracker_imports_from_tracker_wrapper(self):
        from backend.services.tracker_wrapper import SlotTracker
        assert SlotTracker is not None

    def test_slot_tracker_legacy_update_returns_optional_tuple(self):
        from backend.services.tracker_wrapper import SlotTracker
        tracker = SlotTracker(slot_id=0, backend="KCF")
        # Construct a synthetic 100x100 BGR frame with a bright square.
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[40:60, 40:60] = 255
        ok = tracker.init(frame, (40, 40, 20, 20))
        if not ok:  # KCF unavailable in this OpenCV — skip
            pytest.skip("KCF not available in this opencv build")
        # OpenCV trackers return either (x, y, w, h) on success or None.
        result = tracker.update(frame)
        assert result is None or len(result) == 4

    def test_create_tracker_instance_still_exported(self):
        from backend.services.tracker_wrapper import _create_tracker_instance
        assert callable(_create_tracker_instance)


# ──────────────────────────────────────────────────────────────────
# 2. Dispatcher routing rules
# ──────────────────────────────────────────────────────────────────


class TestDispatcherRouting:
    """``CLIPAI_TRACKER_BACKEND`` must drive the right code path."""

    def setup_method(self):
        # Reset the one-shot "logged the chosen backend" flag so each
        # test sees a fresh routing decision.
        import backend.services.tracker_wrapper as tw
        tw._BACKEND_LOGGED = False

    @mock.patch("backend.services.tracker_wrapper._gpu_visible", return_value=False)
    def test_auto_falls_back_to_opencv_when_no_gpu(self, _gpu):
        from backend.services.tracker_wrapper import (
            create_tracker, get_active_backend,
        )
        assert get_active_backend("auto") == "opencv"
        tracker = create_tracker(slot_id=0, backend="auto")
        assert type(tracker).__name__ == "OpenCvSlotTracker"

    @mock.patch("backend.services.tracker_wrapper._gpu_visible", return_value=True)
    def test_samurai_explicit_when_gpu_visible(self, _gpu):
        from backend.services.tracker_wrapper import create_tracker
        # Mock the upstream sam2 import path so the SamuraiTracker
        # can be constructed without the wheel installed.
        with mock.patch(
            "backend.services.samurai_tracker._is_gpu_visible",
            return_value=True,
        ):
            tracker = create_tracker(slot_id=0, backend="samurai")
        assert type(tracker).__name__ == "SamuraiTracker"

    @mock.patch("backend.services.tracker_wrapper._gpu_visible", return_value=False)
    def test_samurai_explicit_raises_when_no_gpu(self, _gpu):
        from backend.services.tracker_wrapper import create_tracker
        with pytest.raises(RuntimeError, match="GPU"):
            create_tracker(slot_id=0, backend="samurai")

    def test_opencv_explicit_always_returns_opencv(self):
        from backend.services.tracker_wrapper import create_tracker
        tracker = create_tracker(slot_id=0, backend="opencv")
        assert type(tracker).__name__ == "OpenCvSlotTracker"

    def test_invalid_backend_falls_back_to_auto(self, monkeypatch):
        # An unknown backend value should not crash — it warns and
        # falls back to ``auto``.
        monkeypatch.setenv("CLIPAI_TRACKER_BACKEND", "lolwhat")
        with mock.patch(
            "backend.services.tracker_wrapper._gpu_visible",
            return_value=False,
        ):
            from backend.services.tracker_wrapper import create_tracker
            tracker = create_tracker(slot_id=0, backend=None)
        assert type(tracker).__name__ == "OpenCvSlotTracker"

    def test_env_var_drives_default(self, monkeypatch):
        monkeypatch.setenv("CLIPAI_TRACKER_BACKEND", "opencv")
        from backend.services.tracker_wrapper import (
            create_tracker, get_active_backend,
        )
        assert get_active_backend() == "opencv"
        tracker = create_tracker(slot_id=0)
        assert type(tracker).__name__ == "OpenCvSlotTracker"


# ──────────────────────────────────────────────────────────────────
# 3. SamuraiTracker GPU gate
# ──────────────────────────────────────────────────────────────────


class TestSamuraiGpuGate:
    """Construction must hard-fail when GPU is absent."""

    @mock.patch("backend.services.samurai_tracker._is_gpu_visible", return_value=False)
    def test_init_raises_runtime_error_without_gpu(self, _gpu):
        from backend.services.samurai_tracker import SamuraiTracker
        with pytest.raises(RuntimeError, match="GPU not visible"):
            SamuraiTracker(slot_id=0, device="cuda")

    @mock.patch("backend.services.samurai_tracker._is_gpu_visible", return_value=True)
    def test_init_succeeds_with_gpu(self, _gpu):
        from backend.services.samurai_tracker import SamuraiTracker
        # Construction does NOT load the model — should be cheap and
        # succeed even without sam2 wheel installed.
        tracker = SamuraiTracker(slot_id=0, device="cuda")
        assert tracker.slot_id == 0
        assert tracker._initialized is False


# ──────────────────────────────────────────────────────────────────
# 4. Motion-aware memory layer
# ──────────────────────────────────────────────────────────────────


class TestMotionMemory:
    """The motion-aware memory IS the paper's contribution. Test it."""

    def test_predict_none_before_observation(self):
        from backend.services.samurai_tracker import MotionMemory
        m = MotionMemory()
        assert m.predict() is None
        assert m.is_initialized is False

    def test_observation_initializes_state(self):
        from backend.services.samurai_tracker import MotionMemory
        m = MotionMemory()
        m.observe(100.0, 50.0)
        assert m.is_initialized
        # First observation: predicted = current (no velocity yet).
        pred = m.predict()
        assert pred is not None
        assert abs(pred[0] - 100.0) < 1e-6
        assert abs(pred[1] - 50.0) < 1e-6

    def test_constant_velocity_extrapolation(self):
        """A subject moving 10 px/frame should be extrapolated forward."""
        from backend.services.samurai_tracker import MotionMemory
        m = MotionMemory(alpha=1.0)  # alpha=1 = pure last-step velocity
        for i in range(5):
            m.observe(10.0 * i, 0.0)
        pred = m.predict()
        assert pred is not None
        # alpha=1 EWMA velocity = last delta = 10. Last centroid = 40.
        # Predicted = 40 + 10 = 50.
        assert abs(pred[0] - 50.0) < 1e-3
        assert abs(pred[1]) < 1e-3

    def test_memory_length_caps_history(self):
        from backend.services.samurai_tracker import MotionMemory
        m = MotionMemory(length=4)
        for i in range(20):
            m.observe(float(i), 0.0)
        # History capped at 4 entries.
        assert len(m.centroids) == 4
        assert len(m.velocities) == 4

    def test_reset_clears_state(self):
        from backend.services.samurai_tracker import MotionMemory
        m = MotionMemory()
        for i in range(5):
            m.observe(float(i), float(i))
        assert m.is_initialized
        m.reset()
        assert not m.is_initialized
        assert m.predict() is None


# ──────────────────────────────────────────────────────────────────
# 5. Synthetic-occlusion fixture (the SAMURAI value-prop test)
# ──────────────────────────────────────────────────────────────────


class _FakeSam2Adapter:
    """Mock sam2 video predictor.

    Models the SAMURAI behaviour we care about: returns a mask centred
    near the motion-memory prediction when the subject is "visible",
    None otherwise. Caller controls visibility per frame via
    ``visible_per_frame``.
    """

    def __init__(self, model_size="tiny", device="cuda"):
        self.model_size = model_size
        self.device = device
        self._loaded = False
        self._init_state_called = False
        self._last_box = None
        self._last_point = None
        self._calls = 0
        # Caller injects these for the per-test scenario.
        self.visible_per_frame: list = []
        self.subject_centroid_per_frame: list = []  # (cx, cy) when visible

    def load(self):
        self._loaded = True

    def init_state(self, frame):
        self._init_state_called = True

    def add_box_prompt(self, frame_idx, obj_id, bbox):
        self._last_box = (frame_idx, obj_id, bbox)

    def add_point_prompt(self, frame_idx, obj_id, point, label=1):
        self._last_point = (frame_idx, obj_id, point, label)

    def propagate_one(self, frame, frame_idx):
        idx = self._calls
        self._calls += 1
        if idx >= len(self.visible_per_frame):
            return None
        if not self.visible_per_frame[idx]:
            return None
        # Synthesize a 30x30 mask centred on the configured centroid.
        mask = np.zeros(frame.shape[:2], dtype=bool)
        cx, cy = self.subject_centroid_per_frame[idx]
        cx_i, cy_i = int(cx), int(cy)
        h, w = frame.shape[:2]
        x0 = max(0, cx_i - 15)
        x1 = min(w, cx_i + 15)
        y0 = max(0, cy_i - 15)
        y1 = min(h, cy_i + 15)
        mask[y0:y1, x0:x1] = True
        return mask


class TestSamuraiOcclusionFixture:
    """End-to-end: SAMURAI tracker with a mocked predictor reproduces
    the "ID held through occlusion" behaviour."""

    def test_holds_identity_through_occlusion(self):
        from backend.services.samurai_tracker import SamuraiTracker

        with mock.patch(
            "backend.services.samurai_tracker._is_gpu_visible",
            return_value=True,
        ):
            tracker = SamuraiTracker(
                slot_id=0,
                _adapter_cls=_FakeSam2Adapter,
            )
        # Subject moves left-to-right 5 px/frame, occluded for 10 frames
        # in the middle.
        n = 30
        adapter = tracker._adapter
        adapter.visible_per_frame = (
            [True] * 10 + [False] * 10 + [True] * 10
        )
        adapter.subject_centroid_per_frame = [
            (100.0 + i * 5.0, 200.0) for i in range(n)
        ]

        frame0 = np.zeros((400, 800, 3), dtype=np.uint8)
        ok = tracker.init(frame0, (85.0, 185.0, 30.0, 30.0))
        assert ok

        n_visible = 0
        n_lost = 0
        last_seen_centroid = None
        for i in range(1, n):
            frame = np.zeros((400, 800, 3), dtype=np.uint8)
            ok, bbox, mask = tracker.update(frame)
            if ok:
                n_visible += 1
                last_seen_centroid = (bbox[0] + bbox[2] / 2,
                                      bbox[1] + bbox[3] / 2)
            else:
                n_lost += 1
        # 9 visible frames before occlusion + 10 after = 19; we count
        # up to 19. Occlusion frames count as "lost" but the tracker
        # stays initialized so it can recover.
        assert n_visible >= 18
        assert n_lost >= 8  # the 10 occlusion frames
        # After recovery the tracker is back on the subject.
        assert last_seen_centroid is not None
        assert abs(last_seen_centroid[0] - (100.0 + 29 * 5.0)) < 25.0

    def test_motion_memory_seeded_predictor_with_point_prompt(self):
        """Confirms the motion memory injects a point prompt every step."""
        from backend.services.samurai_tracker import SamuraiTracker

        with mock.patch(
            "backend.services.samurai_tracker._is_gpu_visible",
            return_value=True,
        ):
            tracker = SamuraiTracker(
                slot_id=7, _adapter_cls=_FakeSam2Adapter,
            )
        adapter = tracker._adapter
        adapter.visible_per_frame = [True, True, True]
        adapter.subject_centroid_per_frame = [
            (100.0, 200.0),
            (110.0, 200.0),
            (120.0, 200.0),
        ]

        frame = np.zeros((400, 800, 3), dtype=np.uint8)
        tracker.init(frame, (85.0, 185.0, 30.0, 30.0))
        tracker.update(frame)
        # After at least one update, a point prompt should have been
        # injected with slot_id 7 at a centroid near the latest one.
        assert adapter._last_point is not None
        _frame_idx, obj_id, point, label = adapter._last_point
        assert obj_id == 7
        assert label == 1
        # The predicted centroid should be in roughly the right area.
        assert 90.0 <= point[0] <= 140.0


# ──────────────────────────────────────────────────────────────────
# 6. New identity_switch_count metric
# ──────────────────────────────────────────────────────────────────


class TestIdentitySwitchMetric:
    def test_no_switches_on_stable_track(self):
        from backend.services.autoflip_parity_metrics import identity_switch_count
        timeline = [
            {"t": 0.0, "slots": {0: (100, 100, 50, 50)}},
            {"t": 0.5, "slots": {0: (102, 100, 50, 50)}},
            {"t": 1.0, "slots": {0: (104, 100, 50, 50)}},
        ]
        assert identity_switch_count(timeline) == 0

    def test_one_switch_when_bbox_jumps(self):
        from backend.services.autoflip_parity_metrics import identity_switch_count
        timeline = [
            {"t": 0.0, "slots": {0: (100, 100, 50, 50)}},
            {"t": 0.5, "slots": {0: (500, 500, 50, 50)}},  # massive jump = ID swap
            {"t": 1.0, "slots": {0: (502, 500, 50, 50)}},
        ]
        assert identity_switch_count(timeline) == 1

    def test_empty_timeline_returns_zero(self):
        from backend.services.autoflip_parity_metrics import identity_switch_count
        assert identity_switch_count([]) == 0

    def test_multi_slot_independent_switches(self):
        from backend.services.autoflip_parity_metrics import identity_switch_count
        timeline = [
            {"t": 0.0, "slots": {0: (100, 100, 50, 50), 1: (200, 200, 50, 50)}},
            {"t": 0.5, "slots": {0: (101, 100, 50, 50), 1: (700, 200, 50, 50)}},  # slot 1 swaps
            {"t": 1.0, "slots": {0: (102, 100, 50, 50), 1: (701, 200, 50, 50)}},
        ]
        assert identity_switch_count(timeline) == 1

    def test_score_fixture_runs_identity_metric(self):
        from backend.services.autoflip_parity_metrics import score_fixture
        timeline = [
            {"t": 0.0, "slots": {0: (100, 100, 50, 50)}},
            {"t": 0.5, "slots": {0: (500, 500, 50, 50)}},
        ]
        out = score_fixture(
            metrics_to_run=["identity_switch_count"],
            identity_timeline=timeline,
        )
        assert out["identity_switch_count"] == 1


# ──────────────────────────────────────────────────────────────────
# 7. Config flag exists with the right default
# ──────────────────────────────────────────────────────────────────


class TestConfigFlag:
    def test_tracker_backend_default_is_auto(self):
        try:
            from backend.config import Settings
        except ModuleNotFoundError as exc:
            pytest.skip(f"backend.config requires deps not installed in sandbox: {exc}")
        s = Settings()
        assert s.CLIPAI_TRACKER_BACKEND == "auto"

    def test_phase_flags_present_in_config_source(self):
        """Static check: config.py declares all five SOTA phase flags."""
        config_src = Path(_REPO / "backend" / "config.py").read_text()
        for flag in (
            "CLIPAI_TRACKER_BACKEND",
            "CLIPAI_DENSE_POINT_TRACKING",
            "CLIPAI_SALIENCY_ENABLED",
            "CLIPAI_COMPOSITION_HEAD",
            "CLIPAI_EDITORIAL_PLANNER",
        ):
            assert flag in config_src, f"{flag} missing from backend/config.py"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
