"""Fix 1 — face_registry teleport-merge + panel-aware arbitration.

Tests exercise the new `_merge_teleporting_embedding_slots` helper and
the `_detect_panel_mode` signal. We don't invoke build_face_registry_
with_embeddings end-to-end because it depends on cv2 / pose verifier —
the helpers here are the reusable pieces we need to lock in.
"""

from dataclasses import dataclass, field

import numpy as np

from backend.services.face_registry import (
    FaceSlot,
    _detect_panel_mode,
    _merge_teleporting_embedding_slots,
)


@dataclass
class _FakeFace:
    nose_x: float
    nose_y: float = 50.0
    width: float = 12.0
    height: float = 12.0
    identity_embedding: list = None
    identity_id: int = -1
    is_human: bool = True


@dataclass
class _FakeFR:
    timestamp: float
    faces: list = field(default_factory=list)


def _mk_slot(sid, cx, xmin, xmax, frames=40):
    return FaceSlot(
        slot_id=sid, x_center=cx, x_min=xmin, x_max=xmax,
        frame_count=frames, avg_width=12.0, avg_height=12.0,
    )


def test_detect_panel_mode_fires_for_seated_interview():
    # 30 frames, 40% multi-face (12 of 30 frames have 2+ faces),
    # avg_faces≈1.6 — panel territory.
    faces_l = _FakeFace(nose_x=23.0)
    faces_m = _FakeFace(nose_x=56.0)
    faces_r = _FakeFace(nose_x=80.0)
    frames = []
    for i in range(30):
        if i < 12:
            frames.append(_FakeFR(i * 0.5, faces=[faces_l, faces_m, faces_r]))
        else:
            frames.append(_FakeFR(i * 0.5, faces=[faces_l]))
    assert _detect_panel_mode(frames) is True


def test_detect_panel_mode_rejects_single_speaker():
    # Every frame has exactly one face at 50 → not a panel.
    frames = [_FakeFR(i * 0.5, faces=[_FakeFace(nose_x=50.0)]) for i in range(30)]
    assert _detect_panel_mode(frames) is False


def test_merge_collapses_same_identity_different_poses():
    # Two "slots" at nearly the same x_center (within 8%) whose
    # embeddings are cosine-similar above 0.70. They should merge to one.
    slots = [
        _mk_slot(0, cx=50.0, xmin=14.0, xmax=97.0, frames=50),
        _mk_slot(1, cx=52.0, xmin=15.0, xmax=98.0, frames=40),
    ]
    # Slot 0 owns faces 0-2 (identity A). Slot 1 owns faces 3-5
    # (identity A-prime — same identity under different lighting).
    emb_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    emb_a_prime = emb_a + np.array([0.05, 0.01, 0.0, 0.0], dtype=np.float32)
    embs = np.stack([emb_a, emb_a, emb_a, emb_a_prime, emb_a_prime, emb_a_prime])
    embs_norm = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    slot_face_indices = {0: [0, 1, 2], 1: [3, 4, 5]}
    merged = _merge_teleporting_embedding_slots(slots, slot_face_indices, embs_norm)
    assert len(merged) == 1, f"expected 1 merged slot, got {len(merged)}"
    # Combined frame count = 50 + 40 = 90.
    assert merged[0].frame_count == 90
    # x_min should pick up the lower of the two, x_max the higher.
    assert merged[0].x_min == 14.0
    assert merged[0].x_max == 98.0


def test_merge_keeps_distinct_identities():
    # Two slots with near-identical x_center but orthogonal embeddings
    # (cosine similarity below 0.70) should stay separate.
    slots = [
        _mk_slot(0, cx=50.0, xmin=45.0, xmax=55.0, frames=40),
        _mk_slot(1, cx=52.0, xmin=47.0, xmax=57.0, frames=40),
    ]
    emb_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    emb_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    embs = np.stack([emb_a, emb_a, emb_b, emb_b])
    embs_norm = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    slot_face_indices = {0: [0, 1], 1: [2, 3]}
    merged = _merge_teleporting_embedding_slots(slots, slot_face_indices, embs_norm)
    # Orthogonal embeddings → cos_sim = 0 → no merge.
    assert len(merged) == 2


def test_merge_keeps_far_apart_slots():
    # Two slots with very different x_center (>8%) should stay separate
    # even if embeddings match perfectly.
    slots = [
        _mk_slot(0, cx=23.0, xmin=20.0, xmax=26.0),
        _mk_slot(1, cx=80.0, xmin=76.0, xmax=84.0),
    ]
    emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    embs = np.stack([emb, emb, emb, emb])
    embs_norm = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    slot_face_indices = {0: [0, 1], 1: [2, 3]}
    merged = _merge_teleporting_embedding_slots(slots, slot_face_indices, embs_norm)
    assert len(merged) == 2


def test_merge_does_not_touch_stage_speakers_with_orthogonal_embeddings():
    # Regression for test_six_speakers_with_stage_movement_preserved:
    # 6 speakers moving across the stage have x_centers that are
    # close together (within 8%) BUT their embeddings are orthogonal.
    # My first draft of the merge helper re-assigned faces by nearest
    # x-center, which destroyed the true embedding groups and caused
    # false merges. This test locks in the correct behavior.
    slots = [
        _mk_slot(0, cx=35.0, xmin=10.0, xmax=60.0, frames=10),
        _mk_slot(1, cx=40.0, xmin=15.0, xmax=65.0, frames=10),
        _mk_slot(2, cx=45.0, xmin=20.0, xmax=70.0, frames=10),
        _mk_slot(3, cx=50.0, xmin=25.0, xmax=75.0, frames=10),
        _mk_slot(4, cx=55.0, xmin=30.0, xmax=80.0, frames=10),
        _mk_slot(5, cx=60.0, xmin=35.0, xmax=85.0, frames=10),
    ]
    # 6 distinct orthogonal embeddings, 2 faces per slot.
    rng = np.random.RandomState(42)
    embs = []
    slot_face_indices = {}
    face_idx = 0
    for sid in range(6):
        e = rng.randn(128).astype(np.float32)
        e /= np.linalg.norm(e)
        idxs = []
        for _ in range(2):
            embs.append(e)
            idxs.append(face_idx)
            face_idx += 1
        slot_face_indices[sid] = idxs
    embs = np.stack(embs)
    embs_norm = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    merged = _merge_teleporting_embedding_slots(slots, slot_face_indices, embs_norm)
    # All 6 orthogonal identities must survive even though their
    # x_centers overlap.
    assert len(merged) == 6
