"""Tests for dense propagator with KCF + LK validation."""

import os
import tempfile
import pytest
import numpy as np
import cv2

from backend.services.dense_propagator import (
    build_interpolated_timeline,
    ANCHOR_CONFIDENCE,
    DRIFT_THRESHOLD_PCT,
)
from backend.services.interpolated_timeline import InterpolatedFaceTimeline


# ── Helpers ──

class FakeFace:
    def __init__(self, identity_id, nose_x, nose_y, width=14, height=18):
        self.identity_id = identity_id
        self.nose_x = nose_x
        self.nose_y = nose_y
        self.x = nose_x
        self.y = nose_y
        self.width = width
        self.height = height


class FakeFrameFaces:
    def __init__(self, timestamp, faces):
        self.timestamp = timestamp
        self.faces = faces


def _make_frame_with_rect(w, h, rect_x, rect_y, rect_w, rect_h, color=200):
    """Create a frame with a colored rectangle on dark background."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = 30  # dark gray background
    x1 = max(0, int(rect_x))
    y1 = max(0, int(rect_y))
    x2 = min(w, int(rect_x + rect_w))
    y2 = min(h, int(rect_y + rect_h))
    frame[y1:y2, x1:x2] = color
    return frame


def _save_frame(frame, path):
    cv2.imwrite(str(path), frame)
    return path


class TestBuildInterpolatedTimeline:
    def test_empty_inputs(self):
        tl = build_interpolated_timeline(
            dense_face_results=[], frame_paths=[],
            source_width=1280, source_height=720, source_fps=30.0,
        )
        assert isinstance(tl, InterpolatedFaceTimeline)
        assert len(tl.samples) == 0

    def test_single_anchor_propagates_forward(self):
        """One detection anchor should seed a tracker that propagates forward."""
        with tempfile.TemporaryDirectory() as tmp:
            w, h = 640, 480
            frame_paths = []
            # Generate 10 frames with a rectangle moving slowly right
            for i in range(10):
                rect_x = 200 + i * 3
                frame = _make_frame_with_rect(w, h, rect_x, 150, 100, 100)
                path = os.path.join(tmp, f"f_{i:04d}.png")
                _save_frame(frame, path)
                frame_paths.append((i / 30.0, path))

            # Anchor at frame 0 with face at center of rectangle
            anchor = FakeFrameFaces(0.0, [
                FakeFace(identity_id=0, nose_x=(250 / w * 100), nose_y=(200 / h * 100),
                         width=(100 / w * 100), height=(100 / h * 100)),
            ])

            tl = build_interpolated_timeline(
                dense_face_results=[anchor],
                frame_paths=frame_paths,
                source_width=w, source_height=h, source_fps=30.0,
            )

            assert len(tl.samples) == 10
            # First frame should be anchor
            assert tl.samples[0].is_anchor is True
            # Subsequent frames should have propagated bboxes
            propagated_with_slot0 = [
                s for s in tl.samples[1:] if 0 in s.bboxes
            ]
            assert len(propagated_with_slot0) >= 5, \
                f"Expected at least 5 propagated frames, got {len(propagated_with_slot0)}"

    def test_anchor_confidence_higher_than_propagated(self):
        """Confidence at anchor frames should be higher than propagated frames."""
        with tempfile.TemporaryDirectory() as tmp:
            w, h = 640, 480
            frame_paths = []
            for i in range(10):
                frame = _make_frame_with_rect(w, h, 200, 150, 100, 100)
                path = os.path.join(tmp, f"f_{i:04d}.png")
                _save_frame(frame, path)
                frame_paths.append((i / 30.0, path))

            anchor = FakeFrameFaces(0.0, [
                FakeFace(identity_id=0, nose_x=39.0, nose_y=41.7, width=15.6, height=20.8),
            ])

            tl = build_interpolated_timeline(
                dense_face_results=[anchor],
                frame_paths=frame_paths,
                source_width=w, source_height=h, source_fps=30.0,
            )

            anchor_conf = tl.samples[0].confidences.get(0, 0)
            assert anchor_conf == ANCHOR_CONFIDENCE

            # Find a propagated frame with the slot
            for s in tl.samples[2:]:
                if 0 in s.confidences:
                    assert s.confidences[0] < ANCHOR_CONFIDENCE, \
                        "Propagated confidence should be less than anchor confidence"
                    break

    def test_shot_cut_resets_trackers(self):
        """A shot cut between anchors should reset all trackers."""
        with tempfile.TemporaryDirectory() as tmp:
            w, h = 640, 480
            frame_paths = []
            for i in range(20):
                t = i / 30.0
                if t < 0.3:
                    frame = _make_frame_with_rect(w, h, 200, 150, 100, 100)
                else:
                    frame = _make_frame_with_rect(w, h, 400, 200, 80, 80, color=150)
                path = os.path.join(tmp, f"f_{i:04d}.png")
                _save_frame(frame, path)
                frame_paths.append((t, path))

            # Anchors before and after the cut
            anchors = [
                FakeFrameFaces(0.0, [
                    FakeFace(identity_id=0, nose_x=39.0, nose_y=41.7, width=15.6, height=20.8),
                ]),
                FakeFrameFaces(0.5, [
                    FakeFace(identity_id=0, nose_x=68.8, nose_y=50.0, width=12.5, height=16.7),
                ]),
            ]

            tl = build_interpolated_timeline(
                dense_face_results=anchors,
                frame_paths=frame_paths,
                source_width=w, source_height=h, source_fps=30.0,
                shot_cuts=[0.3],  # Cut at 0.3s
            )

            # Frames right after the cut (0.3-0.5s) should have no tracked bboxes
            # until the second anchor at 0.5s re-initializes
            post_cut_pre_anchor = [
                s for s in tl.samples
                if 0.3 <= s.timestamp < 0.5 and not s.is_anchor
            ]
            for s in post_cut_pre_anchor:
                assert 0 not in s.bboxes or not s.bboxes, \
                    f"Expected no tracked slot at t={s.timestamp:.3f} after shot cut"

    def test_two_anchors_with_small_drift_no_reset(self):
        """If tracker prediction matches detection within threshold, no reset."""
        with tempfile.TemporaryDirectory() as tmp:
            w, h = 640, 480
            frame_paths = []
            # Stationary rectangle — minimal drift
            for i in range(30):
                frame = _make_frame_with_rect(w, h, 200, 150, 100, 100)
                path = os.path.join(tmp, f"f_{i:04d}.png")
                _save_frame(frame, path)
                frame_paths.append((i / 30.0, path))

            cx_pct = (250 / w) * 100
            cy_pct = (200 / h) * 100
            anchors = [
                FakeFrameFaces(0.0, [
                    FakeFace(identity_id=0, nose_x=cx_pct, nose_y=cy_pct,
                             width=(100 / w * 100), height=(100 / h * 100)),
                ]),
                FakeFrameFaces(0.5, [
                    FakeFace(identity_id=0, nose_x=cx_pct, nose_y=cy_pct,
                             width=(100 / w * 100), height=(100 / h * 100)),
                ]),
            ]

            tl = build_interpolated_timeline(
                dense_face_results=anchors,
                frame_paths=frame_paths,
                source_width=w, source_height=h, source_fps=30.0,
            )

            # With a stationary target, no frames should have had_reset=True
            resets = [s for s in tl.samples if s.had_reset]
            assert len(resets) == 0, f"Expected no resets for stationary target, got {len(resets)}"

    def test_synthetic_30_frames_smooth_motion(self):
        """30 frames with smooth rectangle motion should produce 30 samples."""
        with tempfile.TemporaryDirectory() as tmp:
            w, h = 640, 480
            frame_paths = []
            for i in range(30):
                rect_x = 100 + i * 10  # moving right
                frame = _make_frame_with_rect(w, h, rect_x, 150, 80, 100)
                path = os.path.join(tmp, f"f_{i:04d}.png")
                _save_frame(frame, path)
                frame_paths.append((i / 30.0, path))

            anchor = FakeFrameFaces(0.0, [
                FakeFace(identity_id=0,
                         nose_x=(140 / w * 100), nose_y=(200 / h * 100),
                         width=(80 / w * 100), height=(100 / h * 100)),
            ])

            tl = build_interpolated_timeline(
                dense_face_results=[anchor],
                frame_paths=frame_paths,
                source_width=w, source_height=h, source_fps=30.0,
            )

            assert len(tl.samples) == 30

            # Check that bbox centers generally move rightward
            positions = []
            for s in tl.samples:
                if 0 in s.bboxes:
                    positions.append(s.bboxes[0][0])
            if len(positions) >= 10:
                # First half should be left of second half
                first_half_mean = np.mean(positions[:len(positions) // 2])
                second_half_mean = np.mean(positions[len(positions) // 2:])
                assert second_half_mean >= first_half_mean - 5, \
                    "Expected rightward motion in tracked positions"

    def test_runtime_budget_abort(self):
        """When runtime budget is exceeded, return partial timeline without crash."""
        with tempfile.TemporaryDirectory() as tmp:
            w, h = 320, 240
            frame_paths = []
            # Just 5 frames, but set budget to 0 to trigger immediate abort
            for i in range(5):
                frame = _make_frame_with_rect(w, h, 100, 50, 60, 60)
                path = os.path.join(tmp, f"f_{i:04d}.png")
                _save_frame(frame, path)
                frame_paths.append((i / 30.0, path))

            anchor = FakeFrameFaces(0.0, [
                FakeFace(identity_id=0, nose_x=50, nose_y=33, width=18.8, height=25),
            ])

            # Budget of 0 — should abort immediately after first frame
            tl = build_interpolated_timeline(
                dense_face_results=[anchor],
                frame_paths=frame_paths,
                source_width=w, source_height=h, source_fps=30.0,
                runtime_budget_sec=0.001,
            )

            # Should not crash, may have 0 or partial samples
            assert isinstance(tl, InterpolatedFaceTimeline)
            assert len(tl.samples) < len(frame_paths)

    def test_empty_anchor_faces_handled(self):
        """Anchors with no valid faces should not crash."""
        with tempfile.TemporaryDirectory() as tmp:
            w, h = 320, 240
            frame_paths = []
            for i in range(5):
                frame = _make_frame_with_rect(w, h, 100, 50, 60, 60)
                path = os.path.join(tmp, f"f_{i:04d}.png")
                _save_frame(frame, path)
                frame_paths.append((i / 30.0, path))

            # Anchor with face that has identity_id=-1 (unassigned)
            anchor = FakeFrameFaces(0.0, [
                FakeFace(identity_id=-1, nose_x=50, nose_y=33),
            ])

            tl = build_interpolated_timeline(
                dense_face_results=[anchor],
                frame_paths=frame_paths,
                source_width=w, source_height=h, source_fps=30.0,
            )

            assert len(tl.samples) == 5
            # No tracked bboxes since identity_id=-1 is skipped
            for s in tl.samples:
                assert len(s.bboxes) == 0


class TestAbsenceDetection:
    """Tests for tracker killing on subject absence."""

    def test_subject_absent_two_anchors_killed(self):
        """Subject present at t=0.0 and absent at t=0.5 and t=1.0 → tracker killed."""
        with tempfile.TemporaryDirectory() as tmp:
            w, h = 640, 480
            # Generate 30 frames (1 second at 30fps)
            frame_paths = []
            for i in range(30):
                frame = _make_frame_with_rect(w, h, 200, 150, 100, 100)
                path = os.path.join(tmp, f"f_{i:04d}.png")
                _save_frame(frame, path)
                frame_paths.append((i / 30.0, path))

            cx_pct = (250 / w) * 100
            cy_pct = (200 / h) * 100
            anchors = [
                # Subject present at t=0.0
                FakeFrameFaces(0.0, [
                    FakeFace(identity_id=0, nose_x=cx_pct, nose_y=cy_pct,
                             width=15.6, height=20.8),
                ]),
                # Subject ABSENT at t=0.5 (no faces)
                FakeFrameFaces(0.5, []),
                # Subject ABSENT at t=1.0 → should trigger kill
                FakeFrameFaces(round(29 / 30.0, 3), []),
            ]

            tl = build_interpolated_timeline(
                dense_face_results=anchors,
                frame_paths=frame_paths,
                source_width=w, source_height=h, source_fps=30.0,
            )

            # After second absent anchor, slot 0 should be killed
            last_sample = tl.samples[-1]
            assert 0 not in last_sample.bboxes, \
                "Slot 0 should be killed after 2 consecutive absent anchors"

    def test_brief_absence_then_reappear_not_killed(self):
        """Subject absent for 1 anchor then reappears → NOT killed."""
        with tempfile.TemporaryDirectory() as tmp:
            w, h = 640, 480
            frame_paths = []
            for i in range(45):
                frame = _make_frame_with_rect(w, h, 200, 150, 100, 100)
                path = os.path.join(tmp, f"f_{i:04d}.png")
                _save_frame(frame, path)
                frame_paths.append((i / 30.0, path))

            cx_pct = (250 / w) * 100
            cy_pct = (200 / h) * 100
            anchors = [
                # Present at t=0.0
                FakeFrameFaces(0.0, [
                    FakeFace(identity_id=0, nose_x=cx_pct, nose_y=cy_pct,
                             width=15.6, height=20.8),
                ]),
                # Absent at t=0.5 (1 anchor)
                FakeFrameFaces(0.5, []),
                # Reappears at t=1.0 → counter should reset
                FakeFrameFaces(1.0, [
                    FakeFace(identity_id=0, nose_x=cx_pct, nose_y=cy_pct,
                             width=15.6, height=20.8),
                ]),
            ]

            tl = build_interpolated_timeline(
                dense_face_results=anchors,
                frame_paths=frame_paths,
                source_width=w, source_height=h, source_fps=30.0,
            )

            # After reappearing, slot 0 should still be tracked
            last_anchor_sample = None
            for s in tl.samples:
                if s.is_anchor and s.timestamp >= 1.0:
                    last_anchor_sample = s
                    break
            assert last_anchor_sample is not None
            assert 0 in last_anchor_sample.bboxes, \
                "Slot 0 should survive brief absence when subject reappears"

    def test_killed_slot_has_no_bbox(self):
        """After a tracker is killed, subsequent samples have no bbox for that slot."""
        with tempfile.TemporaryDirectory() as tmp:
            w, h = 640, 480
            frame_paths = []
            for i in range(45):
                frame = _make_frame_with_rect(w, h, 200, 150, 100, 100)
                path = os.path.join(tmp, f"f_{i:04d}.png")
                _save_frame(frame, path)
                frame_paths.append((i / 30.0, path))

            cx_pct = (250 / w) * 100
            cy_pct = (200 / h) * 100
            anchors = [
                FakeFrameFaces(0.0, [
                    FakeFace(identity_id=0, nose_x=cx_pct, nose_y=cy_pct,
                             width=15.6, height=20.8),
                ]),
                # 2 consecutive absent anchors → kill
                FakeFrameFaces(0.5, []),
                FakeFrameFaces(1.0, []),
            ]

            tl = build_interpolated_timeline(
                dense_face_results=anchors,
                frame_paths=frame_paths,
                source_width=w, source_height=h, source_fps=30.0,
            )

            # Find samples after the kill point (t >= 1.0)
            post_kill = [s for s in tl.samples if s.timestamp >= 1.0]
            for s in post_kill:
                assert 0 not in s.bboxes, \
                    f"Killed slot 0 should not appear at t={s.timestamp:.3f}"
