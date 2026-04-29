"""Phase E QA — editorial planner x-axis subject targeting (Task 1).

Verifies that the editorial planner's ``ab_cut`` / ``reaction_at`` /
``framing`` / ``subject`` decisions actually drive per-frame LP camera
targets via the new subject-position lookup, and that the LP solver
respects ``discontinuity_marks`` so it does not smooth across motivated
cuts.

Run:  pytest tests/qa/test_phase_e_editorial_x_targeting.py -v
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ──────────────────────────────────────────────────────────────────
# Synthetic fixtures
# ──────────────────────────────────────────────────────────────────


@dataclass
class _FakeFace:
    nose_x: float           # 0-100 percent
    nose_y: float
    identity_id: int
    is_speaking: bool = False


@dataclass
class _FakeFrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)


@dataclass
class _FakeSlot:
    slot_id: int
    x_center: float         # 0-100 percent


class _FakeRegistry:
    def __init__(self, slots):
        self.slots = slots

    def slot_by_id(self, slot_id):
        for s in self.slots:
            if s.slot_id == slot_id:
                return s
        return None


def _build_two_speaker_dense(width=1920):
    """speaker_1 at x=25%, speaker_2 at x=75%, sampled @ 5fps for 5s."""
    frames = []
    for i in range(26):
        t = i / 5.0
        frames.append(_FakeFrameFaces(
            timestamp=t,
            faces=[
                _FakeFace(nose_x=25.0, nose_y=50.0, identity_id=0,
                          is_speaking=(t < 2.5)),
                _FakeFace(nose_x=75.0, nose_y=50.0, identity_id=1,
                          is_speaking=(t >= 2.5)),
            ],
        ))
    return frames


def _build_three_speaker_dense():
    """speaker_1 @ 20%, speaker_2 @ 50%, speaker_3 @ 80%."""
    frames = []
    for i in range(31):
        t = i / 5.0
        frames.append(_FakeFrameFaces(
            timestamp=t,
            faces=[
                _FakeFace(nose_x=20.0, nose_y=50.0, identity_id=0,
                          is_speaking=(t < 2.0)),
                _FakeFace(nose_x=50.0, nose_y=50.0, identity_id=1,
                          is_speaking=(2.0 <= t < 4.0)),
                _FakeFace(nose_x=80.0, nose_y=50.0, identity_id=2,
                          is_speaking=(t >= 4.0)),
            ],
        ))
    return frames


def _basic_registry():
    return _FakeRegistry([
        _FakeSlot(slot_id=0, x_center=25.0),
        _FakeSlot(slot_id=1, x_center=75.0),
    ])


def _three_slot_registry():
    return _FakeRegistry([
        _FakeSlot(slot_id=0, x_center=20.0),
        _FakeSlot(slot_id=1, x_center=50.0),
        _FakeSlot(slot_id=2, x_center=80.0),
    ])


# ──────────────────────────────────────────────────────────────────
# 1. AB-cut hard override
# ──────────────────────────────────────────────────────────────────


class TestAbCut:
    def test_ab_cut_overrides_lp_target(self):
        """Frames in the ab_cut lock-in window snap to the cut subject's
        x position regardless of the existing tx."""
        from backend.services.editorial_planner import (
            EditorialPlan, EditorialShot, apply_plan_to_lp_targets,
        )
        from backend.services.editorial_subject_lookup import (
            build_subject_position_lookup,
        )

        dense = _build_two_speaker_dense()
        registry = _basic_registry()
        lookup = build_subject_position_lookup(
            face_registry=registry,
            dense_faces=dense,
            speaker_to_slot={"speaker_1": 0, "speaker_2": 1},
            source_width=1920,
        )

        plan = EditorialPlan(shots=[
            EditorialShot(start=0.0, end=2.0, subject="speaker_1",
                          framing="medium"),
            EditorialShot(start=2.0, end=4.0, subject="speaker_2",
                          ab_cut=True, ab_target="speaker_2",
                          framing="medium"),
        ])
        timestamps = [i * 0.1 for i in range(40)]
        tx = [0.5] * len(timestamps)
        ty = [0.5] * len(timestamps)

        tx_new, _, _, marks = apply_plan_to_lp_targets(
            plan, timestamps, tx, ty,
            subject_position_lookup=lookup, source_width=1920,
        )

        # speaker_2 x = 75% of 1920 → 0.75 source-fraction.
        target_x = 0.75
        # Lock-in window 0.4 s starting at 2.0 → frames 20..23.
        for i in range(20, 24):
            assert abs(tx_new[i] - target_x) < 0.05, (
                f"frame {i} (t={timestamps[i]:.2f}s) tx={tx_new[i]:.3f} "
                f"not within ±5% of speaker_2 target {target_x}"
            )
        assert pytest.approx(2.0, abs=0.001) in marks


# ──────────────────────────────────────────────────────────────────
# 2. Reaction-window hard override
# ──────────────────────────────────────────────────────────────────


class TestReactionWindow:
    def test_reaction_window_hard_override(self):
        from backend.services.editorial_planner import (
            EditorialPlan, EditorialShot, apply_plan_to_lp_targets,
        )
        from backend.services.editorial_subject_lookup import (
            build_subject_position_lookup,
        )

        dense = _build_three_speaker_dense()
        registry = _three_slot_registry()
        lookup = build_subject_position_lookup(
            face_registry=registry,
            dense_faces=dense,
            speaker_to_slot={"speaker_1": 0, "speaker_2": 1, "speaker_3": 2},
            source_width=1920,
        )

        plan = EditorialPlan(shots=[
            EditorialShot(start=0.0, end=5.0, subject="speaker_3",
                          intent="reaction", reaction_at=2.5,
                          framing="medium"),
        ])
        timestamps = [i * 0.05 for i in range(100)]
        tx = [0.20] * len(timestamps)   # cold tracker pointed elsewhere
        ty = [0.5] * len(timestamps)

        tx_new, _, _, marks = apply_plan_to_lp_targets(
            plan, timestamps, tx, ty,
            subject_position_lookup=lookup, source_width=1920,
        )

        target = 0.80   # speaker_3 at 80% → 0.80 source-fraction
        # Window: [2.1, 2.9] s → frames 42..58.
        for i, t in enumerate(timestamps):
            if 2.1 <= t <= 2.9:
                assert abs(tx_new[i] - target) < 0.02, (
                    f"reaction window frame t={t:.2f}s tx={tx_new[i]:.3f}"
                )
        assert pytest.approx(2.5, abs=0.001) in marks


# ──────────────────────────────────────────────────────────────────
# 3. Lead-room offset under tight framing
# ──────────────────────────────────────────────────────────────────


class TestLeadRoom:
    def test_lead_room_offsets_target(self):
        """tight framing + subject in left third → camera target shifts
        right by 5% of source width."""
        from backend.services.editorial_planner import (
            EditorialPlan, EditorialShot, apply_plan_to_lp_targets,
        )
        from backend.services.editorial_subject_lookup import (
            build_subject_position_lookup,
        )

        dense = _build_two_speaker_dense()
        registry = _basic_registry()
        lookup = build_subject_position_lookup(
            face_registry=registry,
            dense_faces=dense,
            speaker_to_slot={"speaker_1": 0, "speaker_2": 1},
            source_width=1920,
        )

        plan = EditorialPlan(shots=[
            EditorialShot(start=0.0, end=2.0, subject="speaker_1",
                          ab_cut=True, ab_target="speaker_1",
                          framing="tight"),
        ])
        timestamps = [i * 0.05 for i in range(20)]
        tx = [0.5] * len(timestamps)
        ty = [0.5] * len(timestamps)

        tx_new, _, _, _ = apply_plan_to_lp_targets(
            plan, timestamps, tx, ty,
            subject_position_lookup=lookup, source_width=1920,
        )

        # speaker_1 x = 0.25 (left third) → +0.05 lead-room → 0.30.
        for i in range(0, 8):  # within lock-in
            assert abs(tx_new[i] - 0.30) < 0.01, (
                f"frame {i} tx={tx_new[i]:.3f} expected ~0.30 "
                f"(0.25 base + 0.05 lead-room)"
            )


# ──────────────────────────────────────────────────────────────────
# 4. Discontinuity marks split LP into sub-problems
# ──────────────────────────────────────────────────────────────────


class TestDiscontinuityMarks:
    def test_discontinuity_marks_split_lp(self):
        """When ``discontinuity_marks`` is non-empty, the inner LP
        solver is invoked once per sub-problem with marks=None."""
        from backend.services import l1_camera_path

        face_positions = [(i * 0.1, 100.0 + i * 5) for i in range(40)]
        marks = [1.0, 2.0, 3.0, 1.02]   # last one within 50ms of 1.0 → dedupe

        with mock.patch.object(
            l1_camera_path, "solve_camera_path",
            wraps=l1_camera_path.solve_camera_path,
        ) as wrapped:
            l1_camera_path._solve_camera_path_with_discontinuities(
                face_positions=face_positions,
                source_width=1920,
                lam=l1_camera_path.TV_LAMBDA,
                hard_features=None,
                source_height=1080,
                crop_aspect=9 / 16,
                job_id="test",
                seg_idx=0,
                weights=None,
                discontinuity_marks=marks,
            )

        # 3 deduped marks → 4 sub-problems → 4 calls without marks kwarg.
        sub_calls = [
            c for c in wrapped.call_args_list
            if c.kwargs.get("discontinuity_marks") in (None, [], False)
        ]
        assert len(sub_calls) == 4, (
            f"expected 4 sub-problem solves, got {len(sub_calls)}"
        )

    def test_concurrent_marks_dedupe(self):
        """Two ab_cuts within 50ms collapse to one mark."""
        from backend.services.editorial_planner import (
            EditorialPlan, EditorialShot, apply_plan_to_lp_targets,
        )
        from backend.services.editorial_subject_lookup import (
            build_subject_position_lookup,
        )

        dense = _build_two_speaker_dense()
        lookup = build_subject_position_lookup(
            face_registry=_basic_registry(),
            dense_faces=dense,
            speaker_to_slot={"speaker_1": 0, "speaker_2": 1},
            source_width=1920,
        )

        plan = EditorialPlan(shots=[
            EditorialShot(start=2.000, end=2.500, subject="speaker_1",
                          ab_cut=True, ab_target="speaker_1"),
            EditorialShot(start=2.030, end=3.000, subject="speaker_2",
                          ab_cut=True, ab_target="speaker_2"),
        ])
        timestamps = [i * 0.05 for i in range(80)]
        tx = [0.5] * len(timestamps)
        ty = [0.5] * len(timestamps)

        _, _, _, marks = apply_plan_to_lp_targets(
            plan, timestamps, tx, ty,
            subject_position_lookup=lookup, source_width=1920,
        )
        assert len(marks) == 1, f"expected 1 mark after dedupe, got {marks}"


# ──────────────────────────────────────────────────────────────────
# 5. Feature flag preserves legacy behavior
# ──────────────────────────────────────────────────────────────────


class TestFlag:
    def test_disabled_flag_preserves_legacy_behavior(self, monkeypatch):
        """With CLIPAI_USE_EDITORIAL_X_TARGETING=0, tx unchanged even
        when a lookup is passed."""
        # Reload module to pick up env var.
        monkeypatch.setenv("CLIPAI_USE_EDITORIAL_X_TARGETING", "0")
        import importlib
        from backend.services import editorial_planner
        importlib.reload(editorial_planner)

        from backend.services.editorial_subject_lookup import (
            build_subject_position_lookup,
        )

        dense = _build_two_speaker_dense()
        lookup = build_subject_position_lookup(
            face_registry=_basic_registry(),
            dense_faces=dense,
            speaker_to_slot={"speaker_1": 0, "speaker_2": 1},
            source_width=1920,
        )
        plan = editorial_planner.EditorialPlan(shots=[
            editorial_planner.EditorialShot(
                start=0.0, end=4.0, subject="speaker_2",
                ab_cut=True, ab_target="speaker_2", framing="tight",
            ),
        ])
        timestamps = [i * 0.1 for i in range(40)]
        tx_orig = [0.5] * len(timestamps)
        ty = [0.5] * len(timestamps)

        tx_new, _, zoom, marks = editorial_planner.apply_plan_to_lp_targets(
            plan, timestamps, tx_orig, ty,
            subject_position_lookup=lookup, source_width=1920,
        )

        assert tx_new == tx_orig, "flag OFF should not modify tx"
        assert marks == [], "flag OFF should not produce marks"
        # Zoom hints still reflect framing (legacy behavior).
        assert all(abs(z - 0.70) < 1e-6 for z in zoom)

        # Restore flag for any subsequent tests.
        monkeypatch.setenv("CLIPAI_USE_EDITORIAL_X_TARGETING", "1")
        importlib.reload(editorial_planner)


# ──────────────────────────────────────────────────────────────────
# 6. Subject lookup behaviors
# ──────────────────────────────────────────────────────────────────


class TestSubjectLookup:
    def test_fresh_dense_face_resolution(self):
        """Lookup at t with a dense face within freshness_window returns
        that face's center, not the slot's persistent center."""
        from backend.services.editorial_subject_lookup import (
            build_subject_position_lookup,
        )

        # Slot says 25%, but the most recent dense face says 30%.
        dense = [_FakeFrameFaces(
            timestamp=1.0,
            faces=[_FakeFace(nose_x=30.0, nose_y=50.0, identity_id=0)],
        )]
        registry = _FakeRegistry([_FakeSlot(slot_id=0, x_center=25.0)])

        lookup = build_subject_position_lookup(
            face_registry=registry,
            dense_faces=dense,
            speaker_to_slot={"speaker_1": 0},
            source_width=1920,
        )
        x, _y = lookup("speaker_1", 1.05)
        assert abs(x - 0.30 * 1920) < 1.0   # dense, not slot

    def test_falls_back_to_slot_center(self):
        """Lookup at t with no fresh dense face falls back to slot
        x_center."""
        from backend.services.editorial_subject_lookup import (
            build_subject_position_lookup,
        )

        dense = [_FakeFrameFaces(
            timestamp=0.5,
            faces=[_FakeFace(nose_x=30.0, nose_y=50.0, identity_id=0)],
        )]
        registry = _FakeRegistry([_FakeSlot(slot_id=0, x_center=25.0)])
        lookup = build_subject_position_lookup(
            face_registry=registry,
            dense_faces=dense,
            speaker_to_slot={"speaker_1": 0},
            source_width=1920,
            freshness_window_s=0.2,
        )
        # t = 5.0 is far outside the freshness window.
        x, _y = lookup("speaker_1", 5.0)
        assert abs(x - 0.25 * 1920) < 1.0   # slot fallback

    def test_returns_none_for_unknown_subject(self):
        from backend.services.editorial_subject_lookup import (
            build_subject_position_lookup,
        )

        dense = _build_two_speaker_dense()
        lookup = build_subject_position_lookup(
            face_registry=_basic_registry(),
            dense_faces=dense,
            speaker_to_slot={"speaker_1": 0, "speaker_2": 1},
            source_width=1920,
        )
        assert lookup("speaker_99", 1.0) is None


# ──────────────────────────────────────────────────────────────────
# 7. Unknown subject in plan = no-op (not a crash)
# ──────────────────────────────────────────────────────────────────


class TestUnknownSubjectNoOp:
    def test_unknown_subject_no_op(self):
        from backend.services.editorial_planner import (
            EditorialPlan, EditorialShot, apply_plan_to_lp_targets,
        )
        from backend.services.editorial_subject_lookup import (
            build_subject_position_lookup,
        )

        dense = _build_two_speaker_dense()
        lookup = build_subject_position_lookup(
            face_registry=_basic_registry(),
            dense_faces=dense,
            speaker_to_slot={"speaker_1": 0, "speaker_2": 1},
            source_width=1920,
        )
        plan = EditorialPlan(shots=[
            EditorialShot(start=0.0, end=2.0, subject="speaker_99",
                          ab_cut=True, ab_target="speaker_99"),
        ])
        timestamps = [i * 0.1 for i in range(20)]
        tx = [0.42] * len(timestamps)
        ty = [0.5] * len(timestamps)

        tx_new, _, _, marks = apply_plan_to_lp_targets(
            plan, timestamps, tx, ty,
            subject_position_lookup=lookup, source_width=1920,
        )
        assert tx_new == tx, "unknown subject should leave tx untouched"
        assert marks == []   # no override, no mark


# ──────────────────────────────────────────────────────────────────
# 8. Full panel-clip smoke test
# ──────────────────────────────────────────────────────────────────


class TestFullPanelSmoke:
    def test_full_panel_clip_smoke(self):
        """End-to-end: plan + lookup + apply_plan_to_lp_targets +
        solve_camera_path with discontinuity_marks. Verifies no
        smoothing across the cut."""
        from backend.services.editorial_planner import (
            EditorialPlan, EditorialShot, apply_plan_to_lp_targets,
        )
        from backend.services.editorial_subject_lookup import (
            build_subject_position_lookup,
        )
        from backend.services.l1_camera_path import solve_camera_path

        dense = _build_two_speaker_dense()
        lookup = build_subject_position_lookup(
            face_registry=_basic_registry(),
            dense_faces=dense,
            speaker_to_slot={"speaker_1": 0, "speaker_2": 1},
            source_width=1920,
        )

        plan = EditorialPlan(shots=[
            EditorialShot(start=0.0, end=2.0, subject="speaker_1",
                          framing="medium"),
            EditorialShot(start=2.0, end=5.0, subject="speaker_2",
                          ab_cut=True, ab_target="speaker_2",
                          framing="medium"),
        ])
        timestamps = [i * 0.1 for i in range(50)]
        tx = [0.5] * len(timestamps)
        ty = [0.5] * len(timestamps)

        tx_new, _, _, marks = apply_plan_to_lp_targets(
            plan, timestamps, tx, ty,
            subject_position_lookup=lookup, source_width=1920,
        )
        assert marks == pytest.approx([2.0])

        # Build pixel-space face_positions for the LP solver from tx_new.
        face_positions = [(t, x * 1920.0) for t, x in zip(timestamps, tx_new)]
        result = solve_camera_path(
            face_positions=face_positions,
            source_width=1920,
            discontinuity_marks=marks,
        )
        # The solver returns a path. Critically: the boundary frames
        # immediately before and after the discontinuity should differ
        # by far more than the LP would normally allow — that's the
        # whole point of splitting at marks.
        path = result.get("path") or []
        assert path, "expected non-empty path from discontinuity solver"

        # Find the path entries straddling t=2.0.
        before_pts = [(t, x) for t, x in path if t < 2.0]
        after_pts = [(t, x) for t, x in path if t >= 2.0]
        assert before_pts and after_pts
        last_before = before_pts[-1][1]
        first_after = after_pts[0][1]
        # Lock-in target post-cut is speaker_2's pixel position (1440).
        # Pre-cut, with soft pull on speaker_1 + framing_weight=0.5 the
        # tracker landed near 0.375 * 1920 = 720. The boundary jump
        # therefore should be ≥ 600 px — a smoothness-bounded LP would
        # spread that move across many frames.
        assert first_after - last_before > 600, (
            f"expected hard cut at t=2.0; boundary delta only "
            f"{first_after - last_before:.0f}px "
            f"(last_before={last_before:.0f}, first_after={first_after:.0f})"
        )

        # Panel-target sanity: after applying overrides ≤ 5% of frames
        # should fall in the segments_under_1s zone. Synthetic clip is
        # one ab_cut (≥ 1s of hold each side), so the rate is 0.
        n_seg_under_1s = 0
        held_run_t0 = path[0][0]
        for i in range(1, len(path)):
            if abs(path[i][1] - path[i - 1][1]) > 30:  # ≥ 30px ≈ position change
                run = path[i][0] - held_run_t0
                if run < 1.0:
                    n_seg_under_1s += 1
                held_run_t0 = path[i][0]
        assert n_seg_under_1s / max(1, len(path)) <= 0.05
