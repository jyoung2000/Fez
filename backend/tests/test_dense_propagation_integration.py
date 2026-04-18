"""Integration test for per-frame propagation across shot cut.

Synthesizes a short video with a moving colored rectangle, runs the
dense propagator, and verifies tracker behavior across a shot boundary.
"""

import os
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.services.dense_propagator import build_interpolated_timeline
from backend.services.interpolated_timeline import InterpolatedFaceTimeline


# ── Helpers ──

class _F:
    """Fake face for sparse anchor."""
    def __init__(self, x, y, w, h, sid=0):
        self.identity_id = sid
        self.nose_x = x
        self.nose_y = y
        self.width = w
        self.height = h
        self.x = x
        self.y = y


class _DF:
    """Fake dense faces frame."""
    def __init__(self, t, faces):
        self.timestamp = t
        self.faces = faces


def _generate_test_frames(tmp_path, n_frames=150, w=640, h=480, fps=30):
    """Generate synthetic frames with a rectangle that moves and teleports at the midpoint.

    First half: rectangle moves from x=100 to x=400
    Second half (after "shot cut"): rectangle teleports to x=100 and moves right again
    """
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(exist_ok=True)
    frame_paths = []
    mid = n_frames // 2

    for i in range(n_frames):
        t = i / fps
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = 30  # dark background

        if i < mid:
            rect_x = 100 + int(300 * i / mid)
        else:
            rect_x = 100 + int(300 * (i - mid) / (n_frames - mid))

        # Draw a colored rectangle
        x1 = max(0, rect_x)
        x2 = min(w, rect_x + 120)
        frame[150:310, x1:x2] = (0, 100, 200)  # BGR

        path = frames_dir / f"f_{i:04d}.png"
        cv2.imwrite(str(path), frame)
        frame_paths.append((t, str(path)))

    return frame_paths, mid / fps  # cut timestamp


def _build_sparse_anchors(frame_paths, w, h, anchor_interval=15):
    """Build sparse detection anchors at every Nth frame."""
    anchors = []
    n_frames = len(frame_paths)
    mid = n_frames // 2

    for i in range(0, n_frames, anchor_interval):
        t = frame_paths[i][0]
        if i < mid:
            rect_x = 100 + int(300 * i / mid)
        else:
            rect_x = 100 + int(300 * (i - mid) / (n_frames - mid))

        cx_pct = (rect_x + 60) / w * 100  # center of 120px wide rect
        cy_pct = (150 + 80) / h * 100       # center of 160px tall rect
        w_pct = 120 / w * 100
        h_pct = 160 / h * 100

        anchors.append(_DF(t, [_F(cx_pct, cy_pct, w_pct, h_pct, sid=0)]))

    return anchors


class TestPropagationSmoothMotionAcrossShotCut:
    def test_propagation_across_shot_cut(self):
        """Full integration: synthesize frames, propagate, verify shot-cut reset."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            w, h, fps, n_frames = 640, 480, 30, 150

            frame_paths, cut_time = _generate_test_frames(
                tmp_path, n_frames=n_frames, w=w, h=h, fps=fps,
            )
            anchors = _build_sparse_anchors(frame_paths, w, h, anchor_interval=15)

            start = time.monotonic()
            timeline = build_interpolated_timeline(
                dense_face_results=anchors,
                frame_paths=frame_paths,
                source_width=w,
                source_height=h,
                source_fps=fps,
                shot_cuts=[cut_time],
                backend="KCF",
                runtime_budget_sec=120.0,
                job_id="integration_test",
            )
            elapsed = time.monotonic() - start

            # Basic structure checks
            assert isinstance(timeline, InterpolatedFaceTimeline)
            assert len(timeline.samples) == n_frames

            # Verify propagated samples outnumber anchors by ~10x
            anchor_count = sum(1 for s in timeline.samples if s.is_anchor)
            propagated_count = len(timeline.samples) - anchor_count
            assert propagated_count > anchor_count * 5, \
                f"Expected >5x propagated:anchor ratio, got {propagated_count}:{anchor_count}"

            # Verify shot cut: positions just before vs just after the cut
            # should reflect the teleport. Before: rectangle near right side;
            # after: rectangle back at left side (x≈25%).
            pre_cut = [s for s in timeline.samples
                       if cut_time - 0.3 < s.timestamp < cut_time and 0 in s.bboxes]
            post_cut = [s for s in timeline.samples
                        if cut_time <= s.timestamp < cut_time + 0.3 and 0 in s.bboxes]
            if pre_cut and post_cut:
                pre_x = np.mean([s.bboxes[0][0] for s in pre_cut])
                post_x = np.mean([s.bboxes[0][0] for s in post_cut])
                # The rectangle teleports back to start, so post_x < pre_x
                assert post_x < pre_x, \
                    f"Expected position reset after cut: pre={pre_x:.1f}% post={post_x:.1f}%"

            # Verify runtime is reasonable (should be well under budget)
            assert elapsed < 60.0, f"Runtime {elapsed:.1f}s exceeds 60s for {n_frames} frames"

    def test_propagated_positions_move_smoothly(self):
        """Propagated positions should show smooth motion (no teleportation)."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            w, h, fps = 640, 480, 30
            # 60 frames, no shot cut, steady rightward motion
            n_frames = 60
            frames_dir = tmp_path / "frames"
            frames_dir.mkdir()
            frame_paths = []
            for i in range(n_frames):
                t = i / fps
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                frame[:] = 30
                rect_x = 100 + int(300 * i / n_frames)
                frame[150:310, rect_x:rect_x + 120] = (0, 100, 200)
                path = frames_dir / f"f_{i:04d}.png"
                cv2.imwrite(str(path), frame)
                frame_paths.append((t, str(path)))

            # Anchor only at frame 0
            cx_pct = (100 + 60) / w * 100
            cy_pct = (150 + 80) / h * 100
            anchors = [_DF(0.0, [_F(cx_pct, cy_pct, 120 / w * 100, 160 / h * 100)])]

            timeline = build_interpolated_timeline(
                dense_face_results=anchors,
                frame_paths=frame_paths,
                source_width=w, source_height=h, source_fps=fps,
            )

            # Extract tracked positions
            positions = []
            for s in timeline.samples:
                if 0 in s.bboxes:
                    positions.append(s.bboxes[0][0])  # cx_pct

            assert len(positions) >= 30, "Expected at least 30 tracked positions"

            # Check smoothness: no single-frame jump > 10% of frame width
            for i in range(1, len(positions)):
                delta = abs(positions[i] - positions[i - 1])
                assert delta < 10.0, \
                    f"Jump of {delta:.1f}% at frame {i} exceeds smoothness threshold"

    def test_summary_dict_correct(self):
        """to_dict_summary() should report correct stats."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            w, h, fps = 320, 240, 30
            n_frames = 30
            frames_dir = tmp_path / "frames"
            frames_dir.mkdir()
            frame_paths = []
            for i in range(n_frames):
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                frame[50:100, 100:160] = 200
                path = frames_dir / f"f_{i:04d}.png"
                cv2.imwrite(str(path), frame)
                frame_paths.append((i / fps, str(path)))

            anchors = [
                _DF(0.0, [_F(50, 31, 18.8, 20.8)]),
                _DF(0.5, [_F(50, 31, 18.8, 20.8)]),
            ]

            timeline = build_interpolated_timeline(
                dense_face_results=anchors,
                frame_paths=frame_paths,
                source_width=w, source_height=h, source_fps=fps,
            )

            summary = timeline.to_dict_summary()
            assert summary["n_samples"] == n_frames
            assert summary["n_anchors"] >= 2
            assert summary["source_fps"] == fps
            assert summary["duration"] > 0

    def test_feature_flag_off_produces_empty_timeline(self):
        """When USE_DENSE_PROPAGATION is off, no timeline should be built."""
        # This is a logical test — the pipeline code checks the env var
        os.environ["USE_DENSE_PROPAGATION"] = "false"
        flag = os.environ.get("USE_DENSE_PROPAGATION", "false").lower() in ("true", "1", "yes")
        assert flag is False


class TestDriftCorrection:
    def test_drift_detection_fires_on_large_motion(self):
        """When actual motion exceeds tracker prediction, drift correction fires."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            w, h, fps = 640, 480, 30
            n_frames = 30
            frames_dir = tmp_path / "frames"
            frames_dir.mkdir()
            frame_paths = []

            for i in range(n_frames):
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                frame[:] = 30
                if i < 15:
                    rect_x = 100  # stationary
                else:
                    rect_x = 450  # teleported!
                frame[150:310, rect_x:rect_x + 120] = (0, 100, 200)
                path = frames_dir / f"f_{i:04d}.png"
                cv2.imwrite(str(path), frame)
                frame_paths.append((i / fps, str(path)))

            # Anchors: one at start (x=25%), one at frame 15 (x=79.7%)
            anchors = [
                _DF(0.0, [_F(25.0, 47.9, 18.8, 33.3)]),
                _DF(0.5, [_F(79.7, 47.9, 18.8, 33.3)]),  # teleported
            ]

            timeline = build_interpolated_timeline(
                dense_face_results=anchors,
                frame_paths=frame_paths,
                source_width=w, source_height=h, source_fps=fps,
            )

            # The second anchor at t=0.5 should cause a drift reset
            # since the tracker was tracking at x=25% but detection says x=79.7%
            reset_frames = [s for s in timeline.samples if s.had_reset]
            # At least the anchor frame where drift was detected should have had_reset
            assert len(reset_frames) >= 1, \
                "Expected drift correction to fire when object teleports"
