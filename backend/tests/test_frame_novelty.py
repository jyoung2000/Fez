import os
import numpy as np
import cv2
import pytest

from backend.services.frame_novelty import (
    cluster_by_novelty, select_representatives, rank_frames_by_novelty,
)


class _FakeFrame:
    def __init__(self, path, timestamp):
        self.path = path
        self.timestamp = timestamp


def _write_solid(path, bgr_color):
    img = np.full((64, 64, 3), bgr_color, dtype=np.uint8)
    cv2.imwrite(path, img)


def test_clusters_identical_frames(tmp_path):
    paths = []
    for i in range(4):
        p = os.path.join(tmp_path, f"a{i}.jpg")
        _write_solid(p, (30, 40, 200))  # red-ish
        paths.append(p)
    novel = os.path.join(tmp_path, "b.jpg")
    _write_solid(novel, (200, 80, 30))  # blue-ish
    paths.append(novel)
    frames = [_FakeFrame(p, float(i)) for i, p in enumerate(paths)]
    clusters = cluster_by_novelty(frames, distance_threshold=0.35)
    assert len(clusters) == 2
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 4]


def test_select_representatives_picks_median(tmp_path):
    frames = [_FakeFrame("missing.jpg", float(i)) for i in range(5)]
    clusters = [[0, 1, 2, 3, 4]]
    reps = select_representatives(clusters)
    assert reps == [2]


def test_rank_promotes_cuts_and_novel(tmp_path):
    paths = []
    colors = [(30, 40, 200)] * 5 + [(200, 80, 30)]
    for i, c in enumerate(colors):
        p = os.path.join(tmp_path, f"f{i}.jpg")
        _write_solid(p, c)
        paths.append(p)
    frames = [_FakeFrame(p, float(i)) for i, p in enumerate(paths)]
    ranked = rank_frames_by_novelty(frames, scene_cuts=[5.0], distance_threshold=0.35)
    # The novel frame (index 5) is both a cluster representative AND at a cut —
    # it must be in the top 2.
    assert 5 in ranked[:2]
