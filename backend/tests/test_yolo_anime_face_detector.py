"""Phase D — YOLOv8-anime-face ONNX backend for anime_face_detector.

Validates the CLIPAI_ANIME_FACE_BACKEND=yolo_anime flag plumbing
without requiring the real onnxruntime / weights:
  - graceful fallback to lbpcascade on session init failure
  - bbox conversion to AnimeFaceDetection 0-100 percent coordinates
  - flag-routed end-to-end call site
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Pre-mock heavy native deps.
for _mod in (
    "google.generativeai", "google.generativeai.types", "google.ai",
    "google.ai.generativelanguage_v1beta", "anthropic", "groq", "httpx",
    "openai", "ctranslate2", "faster_whisper", "mediapipe",
    "cv2", "onnxruntime",
):
    sys.modules.setdefault(_mod, MagicMock())


def test_yolo_anime_falls_back_to_lbpcascade_on_init_failure(monkeypatch, tmp_path):
    """If the session init returns None, the public entry point must
    still hit the existing lbpcascade tier (which itself returns a
    skipped_reason here because no XML is available)."""
    import backend.services.anime_face_detector as af
    monkeypatch.setattr(af, "_YOLO_ANIME_SESSION", None, raising=False)
    monkeypatch.setattr(af, "_get_yolo_anime_session", lambda: None)
    monkeypatch.setenv("CLIPAI_ANIME_FACE_BACKEND", "yolo_anime")

    frame = tmp_path / "f.png"
    frame.write_bytes(b"fakecontent")

    res = af.detect_anime_faces(str(frame), cascade_path="/tmp/__nope__.xml")
    # The yolo path returned no faces (session=None) and the cascade
    # tier failed because the XML is missing — the public entry point
    # must surface a skipped_reason rather than crash.
    assert res is not None
    assert isinstance(res.detections, list)
    assert len(res.detections) == 0
    assert "cascade not found" in res.skipped_reason


def test_yolo_anime_returns_detections_with_correct_bbox_format(monkeypatch, tmp_path):
    """With a mocked ORT session, _detect_with_yolo_anime should
    produce AnimeFaceDetection objects in 0-100 percent coordinates."""
    import backend.services.anime_face_detector as af

    fake_session = MagicMock()
    fake_input = MagicMock()
    fake_input.name = "images"
    fake_session.get_inputs.return_value = [fake_input]
    # Synthetic prediction at 640x640 input: one box in the upper-left
    # quadrant, one rejected low-conf box.
    preds = np.array([
        [100.0, 100.0, 300.0, 300.0, 0.92],
        [400.0, 400.0, 500.0, 500.0, 0.10],
    ], dtype=np.float32)
    fake_session.run.return_value = [preds]
    monkeypatch.setattr(af, "_YOLO_ANIME_SESSION", fake_session, raising=False)
    monkeypatch.setattr(af, "_get_yolo_anime_session", lambda: fake_session)

    # Stub cv2 so imread returns a 640x640 BGR ndarray.
    fake_cv2 = MagicMock()
    fake_cv2.imread.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
    fake_cv2.resize.side_effect = lambda img, sz: np.zeros(
        (sz[1], sz[0], 3), dtype=np.uint8
    )
    fake_cv2.cvtColor.side_effect = lambda img, code: img
    fake_cv2.COLOR_BGR2RGB = 4
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    res = af._detect_with_yolo_anime(
        "/tmp/__nope__.png", timestamp=2.5, min_confidence=0.4,
    )
    assert res.has_faces
    assert len(res.detections) == 1
    d = res.detections[0]
    # Coords must be 0-100 percent of frame.
    assert 0 <= d.x_center <= 100
    assert 0 <= d.y_center <= 100
    assert 0 < d.width <= 100
    assert 0 < d.height <= 100
    # Bbox center should land near (200/640, 200/640) = (~31%, ~31%).
    assert d.x_center == pytest.approx(31.25, abs=0.5)
    assert d.y_center == pytest.approx(31.25, abs=0.5)
    assert d.source == "yolo_anime"
    assert d.confidence == pytest.approx(0.92)


def test_default_backend_keeps_lbpcascade(monkeypatch):
    """With the env var unset / set to lbpcascade, the public entry point
    must NOT call the yolo_anime helper."""
    import backend.services.anime_face_detector as af
    called = []
    monkeypatch.delenv("CLIPAI_ANIME_FACE_BACKEND", raising=False)
    monkeypatch.setattr(
        af, "_detect_with_yolo_anime",
        lambda *a, **k: called.append("nope") or
        af.AnimeDetectionResult(timestamp=0.0),
    )
    af.detect_anime_faces("/tmp/__nope__.png", cascade_path="/tmp/__nope__.xml")
    assert called == []
