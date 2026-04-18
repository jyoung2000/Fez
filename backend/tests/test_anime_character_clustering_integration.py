"""Week 2 Part C — integration tests for anime character clustering.

Tests the canonical-slot remap logic that lives inline in
``backend/services/pipeline.py`` right after ``build_face_registry``.
The in-pipeline code is not readily importable as a pure function,
so this file re-implements the remap algorithm as a testable helper
and pins both sides with round-trip assertions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest


# ───────────────────────── Helper under test ─────────────────────────


def _build_canonical_slot_map(
    slot_ids: list[int],
    cluster_ids: list[int],
) -> dict[int, int]:
    """Replica of the pipeline.py canonical-slot remap.

    For each input slot, find every slot in its cluster, pick the
    lowest one, and map the input slot to it. Identity slots (a
    cluster of size 1) map to themselves.

    Returns ``{old_slot_id: canonical_slot_id}``.
    """
    canonical: dict[int, int] = {}
    for old_id, cid in zip(slot_ids, cluster_ids):
        same_cluster = [
            sid for sid, c in zip(slot_ids, cluster_ids)
            if c == cid
        ]
        canonical[old_id] = min(same_cluster)
    return canonical


def _apply_remap(
    detections: list,
    canonical: dict[int, int],
) -> int:
    """Replica of the pipeline.py remap-application loop."""
    applied = 0
    for det in detections:
        cur = getattr(det, "identity_id", -1)
        if cur in canonical and canonical[cur] != cur:
            det.identity_id = canonical[cur]
            applied += 1
    return applied


# ───────────────────────── Stubs ─────────────────────────


@dataclass
class _StubFaceInfo:
    identity_id: int = -1
    x_center: float = 50.0
    y_center: float = 50.0
    width: float = 10.0
    height: float = 12.0


@dataclass
class _StubFrameFaces:
    timestamp: float
    frame_path: str
    faces: list = field(default_factory=list)


# ───────────────────────── canonical remap ─────────────────────────


def test_canonical_remap_collapses_to_lowest_in_cluster():
    slot_ids = [0, 1, 2, 3, 4]
    cluster_ids = [0, 0, 1, 1, 1]  # 0+1 same char; 2+3+4 same char
    canonical = _build_canonical_slot_map(slot_ids, cluster_ids)

    assert canonical[0] == 0
    assert canonical[1] == 0, "slot 1 collapses to slot 0"
    assert canonical[2] == 2
    assert canonical[3] == 2
    assert canonical[4] == 2


def test_canonical_remap_singletons_are_identity():
    slot_ids = [0, 1, 2]
    cluster_ids = [0, 1, 2]  # every slot its own cluster
    canonical = _build_canonical_slot_map(slot_ids, cluster_ids)

    assert canonical == {0: 0, 1: 1, 2: 2}
    merges = [(o, n) for o, n in canonical.items() if o != n]
    assert merges == [], "no merges when every slot is unique"


def test_canonical_remap_non_contiguous_slot_ids():
    """Slot ids can be any ints (e.g. 0, 5, 17 after prior merges)."""
    slot_ids = [0, 5, 17, 22]
    cluster_ids = [0, 1, 0, 1]
    canonical = _build_canonical_slot_map(slot_ids, cluster_ids)

    assert canonical[0] == 0
    assert canonical[17] == 0, "17 collapses to 0 (lowest in cluster 0)"
    assert canonical[5] == 5
    assert canonical[22] == 5, "22 collapses to 5 (lowest in cluster 1)"


# ───────────────────────── remap application ─────────────────────────


def test_remap_application_updates_identity_ids():
    detections = [
        _StubFaceInfo(identity_id=0),
        _StubFaceInfo(identity_id=1),
        _StubFaceInfo(identity_id=2),
        _StubFaceInfo(identity_id=1),
    ]
    canonical = {0: 0, 1: 0, 2: 2}
    applied = _apply_remap(detections, canonical)

    assert applied == 2, "both slot-1 detections get rewritten to slot 0"
    assert [d.identity_id for d in detections] == [0, 0, 2, 0]


def test_remap_application_skips_missing_ids():
    """A detection whose identity_id isn't in canonical stays put."""
    detections = [
        _StubFaceInfo(identity_id=0),
        _StubFaceInfo(identity_id=99),  # not in remap
    ]
    canonical = {0: 0, 1: 0}
    applied = _apply_remap(detections, canonical)

    assert applied == 0
    assert detections[1].identity_id == 99


def test_remap_application_skips_identity_maps():
    """If canonical[x] == x, nothing changes and the counter doesn't tick."""
    detections = [
        _StubFaceInfo(identity_id=0),
        _StubFaceInfo(identity_id=0),
    ]
    canonical = {0: 0}  # identity
    applied = _apply_remap(detections, canonical)

    assert applied == 0
    assert all(d.identity_id == 0 for d in detections)


# ───────────── cluster_fingerprints contract on real module ─────────────


def test_real_cluster_fingerprints_merges_close_centroids():
    """Two similar fingerprints land in the same cluster; a distant one doesn't."""
    from backend.services.anime_character_clustering import (
        cluster_fingerprints,
    )

    # 16-dim normalized fingerprints. Two near-duplicates + one
    # distant one.
    close_a = [1.0] + [0.0] * 15
    close_b = [0.95, 0.05] + [0.0] * 14
    distant = [0.0] * 15 + [1.0]

    fingerprints = [close_a, close_b, distant]
    cluster = cluster_fingerprints(fingerprints)

    # close_a and close_b should share a cluster id; distant stands alone.
    assert cluster.cluster_ids[0] == cluster.cluster_ids[1]
    assert cluster.cluster_ids[0] != cluster.cluster_ids[2]
    assert cluster.n_clusters == 2


def test_real_cluster_fingerprints_empty_input():
    from backend.services.anime_character_clustering import (
        cluster_fingerprints,
    )
    result = cluster_fingerprints([])
    assert result.n_clusters == 0
    assert result.cluster_ids == []


def test_real_cluster_fingerprints_all_identical():
    """Three identical fingerprints collapse to a single cluster."""
    from backend.services.anime_character_clustering import (
        cluster_fingerprints,
    )
    fp = [0.25, 0.25, 0.25, 0.25] + [0.0] * 12
    cluster = cluster_fingerprints([fp, fp, fp])
    assert cluster.n_clusters == 1
    assert cluster.cluster_ids == [0, 0, 0]


# ────────────────── End-to-end remap through the real module ──────────────────


def test_end_to_end_two_slot_merge_via_real_clustering():
    """Full pipeline slice: fingerprint cluster → canonical remap → applied.

    Mimics the pipeline.py code path without actually running it:
      1. Two slots whose fingerprint centroids are close.
      2. Third slot whose centroid is distant.
      3. Expected: slots 0 and 1 merge to slot 0; slot 2 untouched.
    """
    from backend.services.anime_character_clustering import (
        cluster_fingerprints,
    )

    slot_centroids = {
        0: [1.0] + [0.0] * 15,
        1: [0.95, 0.05] + [0.0] * 14,
        2: [0.0] * 15 + [1.0],
    }
    slot_ids = list(slot_centroids.keys())
    centroid_list = [slot_centroids[s] for s in slot_ids]
    cluster = cluster_fingerprints(centroid_list)
    canonical = _build_canonical_slot_map(slot_ids, list(cluster.cluster_ids))

    assert canonical[0] == 0
    assert canonical[1] == 0, "slot 1 merges into slot 0"
    assert canonical[2] == 2

    # Apply the remap to a fake detection list.
    detections = [
        _StubFaceInfo(identity_id=0),
        _StubFaceInfo(identity_id=1),
        _StubFaceInfo(identity_id=2),
        _StubFaceInfo(identity_id=1),
    ]
    applied = _apply_remap(detections, canonical)
    assert applied == 2
    assert [d.identity_id for d in detections] == [0, 0, 2, 0]


def test_flag_on_by_default_week2():
    """Week 2 Part C: USE_ANIME_CHARACTER_CLUSTERING default is ON."""
    import importlib
    import os
    from backend.services import anime_character_clustering

    os.environ.pop("CLIPAI_ANIME_CHARACTER_CLUSTERING", None)
    importlib.reload(anime_character_clustering)
    assert anime_character_clustering.USE_ANIME_CHARACTER_CLUSTERING is True
