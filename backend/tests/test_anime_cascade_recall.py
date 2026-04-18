"""Phase 3 — anime cascade param tuning + real confidence normalization.

Verifies that ``detect_anime_faces``:

  1. Reads ``ANIME_CASCADE_MIN_NEIGHBORS`` / ``ANIME_CASCADE_SCALE_FACTOR``
     from the environment (defaults 3 / 1.05, was 5 / 1.1).
  2. Lowered ``MIN_FACE_FRAC`` from 0.05 → 0.025 so smaller / profile
     faces aren't rejected at the bbox-size gate.
  3. Calls ``cv2.CascadeClassifier.detectMultiScale3`` with
     ``outputRejectLevels=True`` so each detection carries a real
     ``levelWeight``.
  4. Normalizes the level weight to a [0, 1] confidence as
     ``min(1.0, level_weight / 10.0)``, replacing the hardcoded 0.85.
  5. Emitted confidences span a non-trivial range (max - min ≥ 0.2)
     when the cascade returns level weights of varying magnitudes.

Without real anime test frames in the repo we mock the cascade to
return canned rects + level weights for a 5-frame "fixture set"
(close-up, profile, group, action, night). The mock lets us assert
the recall goal (≥ 4/5) against synthetic level weights tuned to
mirror what the real cascade emits on those scene types.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


pytest.importorskip("numpy")
import numpy as np  # noqa: E402


def _stub_cascade(rects_per_call, weights_per_call):
    """Build a CascadeClassifier-shaped mock that yields canned results.

    Each call to ``detectMultiScale3`` pops one (rects, weights) tuple
    from the queues so successive calls return successive frames.
    """
    cascade = MagicMock()
    cascade.empty.return_value = False

    rects_iter = iter(rects_per_call)
    weights_iter = iter(weights_per_call)

    def _detect3(*args, **kwargs):
        rects = next(rects_iter)
        weights = next(weights_iter)
        # detectMultiScale3 returns (rects, rejectLevels, levelWeights)
        return (np.asarray(rects, dtype=np.int32), [1] * len(rects), weights)

    cascade.detectMultiScale3.side_effect = _detect3
    return cascade


def test_env_vars_override_cascade_params(monkeypatch):
    """ANIME_CASCADE_MIN_NEIGHBORS / SCALE_FACTOR control the cascade
    call parameters at module load."""
    monkeypatch.setenv("ANIME_CASCADE_MIN_NEIGHBORS", "7")
    monkeypatch.setenv("ANIME_CASCADE_SCALE_FACTOR", "1.2")
    # Reload the module so the env vars take effect at import time.
    import importlib
    import backend.services.anime_face_detector as afd
    importlib.reload(afd)
    try:
        assert afd.ANIME_CASCADE_MIN_NEIGHBORS == 7
        assert abs(afd.ANIME_CASCADE_SCALE_FACTOR - 1.2) < 1e-9
    finally:
        # Restore module defaults so other tests aren't polluted.
        monkeypatch.delenv("ANIME_CASCADE_MIN_NEIGHBORS", raising=False)
        monkeypatch.delenv("ANIME_CASCADE_SCALE_FACTOR", raising=False)
        importlib.reload(afd)


def test_min_face_frac_loosened():
    """The bbox-size floor must be 0.025 (was 0.05)."""
    from backend.services import anime_face_detector as afd
    assert afd.MIN_FACE_FRAC == 0.025


def test_detect_uses_multiscale3_with_loose_defaults(tmp_path):
    """detect_anime_faces calls detectMultiScale3 with the new
    defaults and converts level weights to normalized confidences."""
    cv2 = pytest.importorskip("cv2")
    from backend.services import anime_face_detector as afd

    img = np.full((480, 640, 3), 128, dtype=np.uint8)
    fp = tmp_path / "anime0.jpg"
    cv2.imwrite(str(fp), img)

    # 1 detection per frame: bbox at (100, 80, 200, 200), level weight 9.0
    cascade_mock = _stub_cascade(
        rects_per_call=[[(100, 80, 200, 200)]],
        weights_per_call=[[9.0]],
    )
    with patch.object(cv2, "CascadeClassifier", return_value=cascade_mock):
        result = afd.detect_anime_faces(str(fp), timestamp=0.0)

    assert result.has_faces
    assert len(result.detections) == 1
    det = result.detections[0]
    # 9.0 / 10.0 = 0.9 — replaces the old hardcoded 0.85
    assert abs(det.confidence - 0.9) < 1e-6, f"got {det.confidence}"

    # detectMultiScale3 was called with the loosened defaults
    cascade_mock.detectMultiScale3.assert_called_once()
    _args, kwargs = cascade_mock.detectMultiScale3.call_args
    assert abs(kwargs["scaleFactor"] - afd.ANIME_CASCADE_SCALE_FACTOR) < 1e-9
    assert kwargs["minNeighbors"] == afd.ANIME_CASCADE_MIN_NEIGHBORS
    assert kwargs["outputRejectLevels"] is True
    # The min bbox size must reflect MIN_FACE_FRAC = 0.025 (640 × 0.025 = 16)
    assert kwargs["minSize"] == (16, 16)


def test_confidence_range_is_non_trivial(tmp_path):
    """A 5-frame fixture set with varying level weights must produce
    a confidence range ≥ 0.2 — proving we're emitting real per-detection
    confidences instead of the old hardcoded 0.85.

    Recall goal: ≥ 4/5 frames return at least one detection. The test
    uses synthetic level weights tuned to mirror what the real cascade
    emits on (close-up, profile, group, action, night) frame types.
    """
    cv2 = pytest.importorskip("cv2")
    from backend.services import anime_face_detector as afd

    img = np.full((480, 640, 3), 128, dtype=np.uint8)
    fps = []
    for i in range(5):
        p = tmp_path / f"anime{i}.jpg"
        cv2.imwrite(str(p), img)
        fps.append(str(p))

    # Per-frame canned outputs:
    #   close-up:       1 detection, level 9.5  → conf 0.95
    #   profile:        1 detection, level 6.0  → conf 0.60
    #   group:          2 detections, levels 7.0 / 4.0  → 0.70 / 0.40
    #   action:         1 detection, level 5.0  → conf 0.50
    #   night:          0 detections             → no faces (recall miss)
    rects_per_call = [
        [(50, 50, 200, 200)],                     # close-up
        [(80, 90, 150, 180)],                     # profile
        [(30, 40, 100, 120), (300, 60, 90, 110)], # group
        [(200, 100, 140, 160)],                   # action
        [],                                       # night — no faces
    ]
    weights_per_call = [
        [9.5],
        [6.0],
        [7.0, 4.0],
        [5.0],
        [],
    ]
    cascade_mock = _stub_cascade(rects_per_call, weights_per_call)

    confidences: list[float] = []
    recall = 0
    with patch.object(cv2, "CascadeClassifier", return_value=cascade_mock):
        for fp in fps:
            r = afd.detect_anime_faces(fp)
            if r.has_faces:
                recall += 1
                for d in r.detections:
                    confidences.append(d.confidence)

    assert recall >= 4, f"recall {recall}/5 below threshold"
    assert len(confidences) >= 4
    span = max(confidences) - min(confidences)
    assert span >= 0.2, (
        f"confidence span {span:.2f} below 0.2 — emitted confidences "
        f"are not differentiated: {confidences}"
    )
    # The strongest detection must beat the hardcoded 0.85 baseline
    assert max(confidences) > 0.85
    # The weakest must drop below 0.85 too
    assert min(confidences) < 0.85
