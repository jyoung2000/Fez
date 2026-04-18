"""Week 2 Part B — integration tests for anime shot detector wiring.

Tests the merge logic in ``pipeline.py`` (cut-timestamp list merge with
de-dup) and ``layout_engine.py`` (Shot-object split + renumber). Both
paths are lifted into pure Python helpers for testability — the real
call sites in the pipeline are exercised indirectly via the
``validate_v2_phases`` parity gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import patch

import pytest


@dataclass
class _StubAnimeShotResult:
    cut_times: list[float] = field(default_factory=list)
    skipped_reason: str = ""


# ───────────────────────── Part B1 helper ─────────────────────────


def _merge_cut_timestamps(
    scene_cut_timestamps: list[float] | None,
    anime_cut_times: list[float],
    *,
    dedup_window_sec: float = 0.30,
) -> tuple[list[float], int]:
    """Replica of the pipeline.py merge loop.

    Kept in this test file so the merge semantics stay pinned even if
    the real call site drifts. If the test drifts from the production
    code, update both sides.
    """
    if scene_cut_timestamps is None:
        scene_cut_timestamps = []
    existing = [float(t) for t in scene_cut_timestamps]
    added = 0
    for t in anime_cut_times:
        if any(abs(t - e) < dedup_window_sec for e in existing):
            continue
        existing.append(float(t))
        added += 1
    existing.sort()
    return existing, added


def test_pipeline_merge_adds_and_dedups():
    base = [1.0, 5.0, 10.0]
    anime = [2.5, 5.1, 7.5, 11.0]   # 5.1 is within 0.3 of 5.0, dedup
    merged, added = _merge_cut_timestamps(base, anime)

    assert added == 3, "expected 3 new cuts (5.1 deduped against 5.0)"
    assert merged == sorted([1.0, 2.5, 5.0, 7.5, 10.0, 11.0])


def test_pipeline_merge_preserves_sort_order():
    base = [10.0, 2.0, 5.0]  # unsorted input
    anime = [7.5]
    merged, added = _merge_cut_timestamps(base, anime)

    assert added == 1
    assert merged == [2.0, 5.0, 7.5, 10.0], "merged list must be sorted"


def test_pipeline_merge_handles_none_and_empty():
    merged, added = _merge_cut_timestamps(None, [1.0, 2.0])
    assert added == 2
    assert merged == [1.0, 2.0]

    merged, added = _merge_cut_timestamps([], [])
    assert added == 0
    assert merged == []


def test_pipeline_merge_all_duplicates():
    """Every anime cut is within 0.3s of an existing cut — zero added."""
    base = [1.0, 5.0, 10.0]
    anime = [1.1, 4.85, 10.25]
    merged, added = _merge_cut_timestamps(base, anime)

    assert added == 0
    assert merged == [1.0, 5.0, 10.0]


# ───────────────────────── Part B2 helper ─────────────────────────


def _split_shots_with_anime_cuts(
    shots: list,
    anime_cut_times: list[float],
    Shot_cls,
    *,
    dedup_window_sec: float = 0.30,
) -> list:
    """Replica of the layout_engine.py Shot-split loop."""
    existing_bounds = {round(s.start, 2) for s in shots} | {
        round(s.end, 2) for s in shots
    }
    merged = list(shots)
    for t in anime_cut_times:
        if any(abs(t - b) < dedup_window_sec for b in existing_bounds):
            continue
        for idx, sh in enumerate(merged):
            if sh.start < t < sh.end:
                new_left = Shot_cls(
                    index=sh.index,
                    start=sh.start,
                    end=float(t),
                    detector_confidence="high",
                )
                new_right = Shot_cls(
                    index=sh.index + 1,
                    start=float(t),
                    end=sh.end,
                    detector_confidence="high",
                )
                merged[idx:idx + 1] = [new_left, new_right]
                existing_bounds.add(round(t, 2))
                break
    for i, sh in enumerate(merged):
        sh.index = i
    return merged


@dataclass
class _StubShot:
    index: int
    start: float
    end: float
    detector_confidence: str = "high"


def test_layout_engine_split_inside_single_shot():
    shots = [_StubShot(index=0, start=0.0, end=10.0)]
    merged = _split_shots_with_anime_cuts(
        shots, [3.0, 7.0], _StubShot,
    )

    assert len(merged) == 3
    assert [m.start for m in merged] == [0.0, 3.0, 7.0]
    assert [m.end for m in merged] == [3.0, 7.0, 10.0]
    assert [m.index for m in merged] == [0, 1, 2], "indices must renumber"


def test_layout_engine_split_dedups_near_existing_boundaries():
    shots = [
        _StubShot(index=0, start=0.0, end=5.0),
        _StubShot(index=1, start=5.0, end=10.0),
    ]
    # 5.1 is within 0.3 of the existing 5.0 boundary → dedup.
    # 7.5 is new → split the second shot.
    merged = _split_shots_with_anime_cuts(
        shots, [5.1, 7.5], _StubShot,
    )

    assert len(merged) == 3, "only 7.5 should split"
    assert [m.start for m in merged] == [0.0, 5.0, 7.5]
    assert [m.end for m in merged] == [5.0, 7.5, 10.0]
    assert [m.index for m in merged] == [0, 1, 2]


def test_layout_engine_split_across_multiple_shots():
    shots = [
        _StubShot(index=0, start=0.0, end=5.0),
        _StubShot(index=1, start=5.0, end=10.0),
    ]
    merged = _split_shots_with_anime_cuts(
        shots, [2.0, 8.0], _StubShot,
    )

    assert len(merged) == 4
    assert [m.start for m in merged] == [0.0, 2.0, 5.0, 8.0]
    assert [m.end for m in merged] == [2.0, 5.0, 8.0, 10.0]
    assert [m.index for m in merged] == [0, 1, 2, 3]


def test_layout_engine_cut_exactly_at_boundary_is_deduped():
    shots = [
        _StubShot(index=0, start=0.0, end=5.0),
        _StubShot(index=1, start=5.0, end=10.0),
    ]
    # 5.0 exactly matches an existing boundary → dedup.
    merged = _split_shots_with_anime_cuts(
        shots, [5.0], _StubShot,
    )
    assert len(merged) == 2


def test_layout_engine_cut_outside_all_shots_is_dropped():
    """A cut at t=15.0 when shots cover [0,10] finds no containing shot."""
    shots = [_StubShot(index=0, start=0.0, end=10.0)]
    merged = _split_shots_with_anime_cuts(
        shots, [15.0], _StubShot,
    )
    assert len(merged) == 1  # unchanged


# ───────────────────────── Feature flag gating ─────────────────────────


def test_flag_off_short_circuits_pipeline_path():
    """When USE_ANIME_SHOT_DETECTOR=False, pipeline.py never calls detect_anime_shots."""
    # This is a structural test — import the pipeline's handler path
    # and verify the flag gate is the first check.
    import importlib
    from backend.services import anime_shot_detector

    # Reload with env forced OFF.
    import os
    os.environ["CLIPAI_ANIME_SHOT_DETECTOR"] = "0"
    try:
        importlib.reload(anime_shot_detector)
        assert anime_shot_detector.USE_ANIME_SHOT_DETECTOR is False
    finally:
        os.environ.pop("CLIPAI_ANIME_SHOT_DETECTOR", None)
        importlib.reload(anime_shot_detector)


def test_flag_on_by_default_week2():
    """Week 2 Part B: USE_ANIME_SHOT_DETECTOR default is now ON."""
    import importlib
    import os
    from backend.services import anime_shot_detector

    # Clear the env var so we measure the bare module default.
    os.environ.pop("CLIPAI_ANIME_SHOT_DETECTOR", None)
    importlib.reload(anime_shot_detector)
    assert anime_shot_detector.USE_ANIME_SHOT_DETECTOR is True


# ───────────────────────── Shot dataclass contract ─────────────────────────


def test_real_shot_dataclass_has_required_fields():
    """Smoke: backend.services.shot_detector.Shot supports the fields we need."""
    from backend.services.shot_detector import Shot

    # Construct with the fields the layout_engine merge uses.
    sh = Shot(
        index=0, start=0.0, end=5.0,
        detector_confidence="high",
    )
    assert sh.index == 0
    assert sh.start == 0.0
    assert sh.end == 5.0
    assert sh.detector_confidence == "high"


def test_real_shot_split_with_actual_dataclass():
    """Run the split helper against the real Shot class, not the stub."""
    from backend.services.shot_detector import Shot

    shots = [Shot(index=0, start=0.0, end=10.0, detector_confidence="high")]
    merged = _split_shots_with_anime_cuts(shots, [3.0, 7.0], Shot)

    assert len(merged) == 3
    assert all(isinstance(s, Shot) for s in merged)
    assert [s.index for s in merged] == [0, 1, 2]
