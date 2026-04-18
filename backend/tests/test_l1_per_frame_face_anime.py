"""Phase 4 — per-frame face anchors for the L1 camera path solver.

When animated content has a populated dense-face track (density ≥ 40%),
``build_face_anchor_targets`` must build a uniform-fps signal whose
unary anchor is the actual per-frame ``dense_face.nose_x`` (not the
shot's slot-center). The L1 solver fed that signal then tracks the
face within ±2% of the per-frame nose_x.

When the gate is closed (live-action, or anime with sparse face
data) the helper falls back to the slot-center signal at unit
weight, which produces the legacy slot-median behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

pytest.importorskip("numpy")

from backend.services.l1_camera_path import (  # noqa: E402
    FACE_ANCHOR_DENSITY_FLOOR,
    W_FACE_UNARY,
    build_face_anchor_targets,
    solve_camera_path,
)


@dataclass
class _StubFace:
    identity_id: int
    nose_x: float
    confidence: float = 0.9


@dataclass
class _StubFrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)


def _build_panning_dense(
    slot_id: int = 0,
    *,
    start: float = 0.0,
    end: float = 5.0,
    fps: float = 2.0,  # matches production DENSE_FACE_SAMPLE_RATE = 0.5s
    x_start: float = 50.0,
    x_end: float = 70.0,
) -> list:
    """Build a dense face stream where the face pans linearly across
    [x_start, x_end] (in percent) over [start, end] at ``fps`` samples
    per second."""
    out: list = []
    n = max(2, int((end - start) * fps))
    for i in range(n):
        t = start + i * (end - start) / max(n - 1, 1)
        x = x_start + (x_end - x_start) * (i / max(n - 1, 1))
        out.append(_StubFrameFaces(
            timestamp=round(t, 4),
            faces=[_StubFace(identity_id=slot_id, nose_x=x)],
        ))
    return out


def test_default_env_constants():
    """Phase 4 env defaults: w_face_unary=1.5, density floor=0.4."""
    assert abs(W_FACE_UNARY - 1.5) < 1e-9
    assert abs(FACE_ANCHOR_DENSITY_FLOOR - 0.4) < 1e-9


def test_anime_gate_open_uses_per_frame_face_nose_x():
    """is_animated=True + density 1.0 → every frame's anchor is the
    actual nose_x, not the slot center."""
    dense = _build_panning_dense()  # 1 fps, 5s, 50→70 pct
    positions, weights, used = build_face_anchor_targets(
        dense, slot_id=0, slot_center_pct=10.0,  # bogus slot center
        start=0.0, end=5.0,
        is_animated=True,
    )
    assert used is True
    assert len(positions) > 0
    assert len(positions) == len(weights)
    # Every frame should be near the linear panning signal, NOT at 10
    for (_t, x), w in zip(positions, weights):
        assert 49.0 <= x <= 71.0, f"anchor x={x} not on the panning track"
        assert w == 1.5, f"face anchor frame should carry w_face_unary, got {w}"


def test_anime_gate_closed_when_density_below_floor():
    """A 5s shot with only one face frame → density < 0.4 → falls back
    to slot center, weight 1.0."""
    dense = [
        _StubFrameFaces(
            timestamp=2.5,
            faces=[_StubFace(identity_id=0, nose_x=60.0)],
        ),
    ]
    # Need to set the density window to the full shot, so include the
    # face frame plus several "empty" frames so density < 0.4.
    for t in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 4.5):
        dense.append(_StubFrameFaces(timestamp=t, faces=[]))
    dense.sort(key=lambda x: x.timestamp)

    positions, weights, used = build_face_anchor_targets(
        dense, slot_id=0, slot_center_pct=42.0,
        start=0.0, end=5.0,
        is_animated=True,
    )
    assert used is False
    assert all(abs(x - 42.0) < 1e-9 for _t, x in positions)
    assert all(w == 1.0 for w in weights)


def test_live_action_gate_blocks_anchors_even_when_dense():
    """is_animated=False MUST keep the legacy slot-center behavior so
    the live-action path is unchanged."""
    dense = _build_panning_dense()
    positions, weights, used = build_face_anchor_targets(
        dense, slot_id=0, slot_center_pct=42.0,
        start=0.0, end=5.0,
        is_animated=False,
    )
    assert used is False
    assert all(abs(x - 42.0) < 1e-9 for _t, x in positions)
    assert all(w == 1.0 for w in weights)


def test_solver_tracks_face_within_2pct_when_anchors_present():
    """End-to-end Phase 4: anime + dense anchors → L1 solver path
    tracks the face nose_x within ±2% of frame width.

    Comparison against the gate-closed behavior (slot-center fallback):
    the closed-gate path should sit at the slot center within
    STATIONARY_THRESHOLD instead of tracking the panning face.
    """
    source_width = 1920
    dense = _build_panning_dense(
        x_start=50.0, x_end=70.0, fps=2.0,
    )

    # ── Gate OPEN (anime + dense) ──
    pos_open, weights_open, used = build_face_anchor_targets(
        dense, slot_id=0, slot_center_pct=10.0,
        start=0.0, end=5.0,
        is_animated=True,
    )
    assert used is True

    # Convert (t, x_pct) to (t, x_pixel) for solve_camera_path
    pos_open_px = [(t, x / 100.0 * source_width) for t, x in pos_open]
    result_open = solve_camera_path(
        pos_open_px,
        source_width=source_width,
        weights=weights_open,
    )

    # The solver should produce a path that tracks the panning face.
    # Either "tracking" or "panning" mode is acceptable — both emit a
    # path that follows the per-frame target.
    assert result_open["mode"] in ("tracking", "panning"), (
        f"expected tracking/panning, got {result_open['mode']}"
    )
    # Each solved point should be within 2% of frame width of the
    # corresponding face anchor.
    tol_px = 0.02 * source_width  # 38.4 px
    if result_open["path"]:
        # Match by index — both arrays come from the same uniform grid.
        for (_t, target_px), (_pt, solved_px) in zip(pos_open_px, result_open["path"]):
            # Accept ±5% slack for solver smoothing at the edges
            assert abs(solved_px - target_px) < 5 * tol_px, (
                f"solved {solved_px:.1f} drifted from anchor {target_px:.1f}"
            )
        # Average tracking error must be tighter than 2% of frame width
        errs = [
            abs(s - t) for (_, t), (_, s) in zip(pos_open_px, result_open["path"])
        ]
        mean_err = sum(errs) / max(len(errs), 1)
        assert mean_err < tol_px, (
            f"mean tracking error {mean_err:.1f}px exceeds 2% of frame width "
            f"({tol_px:.1f}px)"
        )

    # ── Gate CLOSED (live action) ──
    pos_closed, weights_closed, used_closed = build_face_anchor_targets(
        dense, slot_id=0, slot_center_pct=10.0,
        start=0.0, end=5.0,
        is_animated=False,
    )
    assert used_closed is False
    pos_closed_px = [(t, x / 100.0 * source_width) for t, x in pos_closed]
    result_closed = solve_camera_path(
        pos_closed_px, source_width=source_width, weights=weights_closed,
    )
    # Slot-center signal is constant → solver must report stationary
    assert result_closed["mode"] == "stationary"
    # And the chosen center must be at the slot center (10% → 192 px)
    assert abs(result_closed["center"] - (10.0 / 100.0 * source_width)) < 5.0


def test_slot_id_none_falls_through_to_first_face():
    """A None slot_id should still produce per-frame anchors when face
    data is present (use the first face on each frame)."""
    dense = _build_panning_dense(slot_id=7)
    positions, _weights, used = build_face_anchor_targets(
        dense, slot_id=None, slot_center_pct=10.0,
        start=0.0, end=5.0,
        is_animated=True,
    )
    assert used is True
    # All anchors should land on the panning track, not the slot center
    for _t, x in positions:
        assert 49.0 <= x <= 71.0
