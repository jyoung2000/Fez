"""Phase B — YOLO11n backend selection in object_detector.

Verifies the CLIPAI_OBJECT_DETECTOR=yolo11n flag selects the new
weights and that any failure cleanly degrades to yolov8n.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# ── Pre-mock heavy native deps so the import survives the sandbox ──
for _mod in (
    "google.generativeai", "google.generativeai.types", "google.ai",
    "google.ai.generativelanguage_v1beta", "anthropic", "groq", "httpx",
    "openai", "ctranslate2", "faster_whisper", "mediapipe",
    "cv2",
):
    sys.modules.setdefault(_mod, MagicMock())


def _install_fake_ultralytics(monkeypatch, captured: list, fail: bool = False):
    """Stub ultralytics.YOLO so we don't need the real weights."""
    fake_mod = MagicMock()

    class _FakeYOLO:
        def __init__(self, model_path):
            captured.append(str(model_path))
            if fail:
                raise RuntimeError("forced load failure")
            self.names = {0: "person"}

        def __call__(self, *args, **kwargs):
            return []

    fake_mod.YOLO = _FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_mod)


def test_yolo11n_selected_by_flag(monkeypatch):
    captured: list = []
    _install_fake_ultralytics(monkeypatch, captured, fail=False)
    monkeypatch.setenv("CLIPAI_OBJECT_DETECTOR", "yolo11n")

    import backend.services.object_detector as od_mod
    od_mod.reset_detector()
    det = od_mod.ObjectDetector(model_dir="/tmp/__nope__")

    # The first weight path requested should be yolo11n.pt
    assert any("yolo11n.pt" in p for p in captured), captured
    assert det.backend_name == "ultralytics-yolo11n"


def test_yolo11n_falls_back_to_yolov8n_on_error(monkeypatch):
    """When yolo11n init raises, the constructor falls through to the
    yolov8n branch instead of crashing — the existing chain still wins.
    """
    captured: list = []

    fake_mod = MagicMock()

    class _FakeYOLO:
        def __init__(self, model_path):
            captured.append(str(model_path))
            if "yolo11n" in str(model_path):
                raise RuntimeError("forced yolo11n failure")
            self.names = {0: "person"}

        def __call__(self, *args, **kwargs):
            return []

    fake_mod.YOLO = _FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_mod)
    monkeypatch.setenv("CLIPAI_OBJECT_DETECTOR", "yolo11n")

    import backend.services.object_detector as od_mod
    od_mod.reset_detector()
    det = od_mod.ObjectDetector(model_dir="/tmp/__nope__")

    # We should have tried yolo11n first, then fallen back to yolov8n.
    assert any("yolo11n.pt" in p for p in captured), captured
    assert any("yolov8n.pt" in p for p in captured), captured
    assert det.backend_name == "ultralytics-yolov8n"


def test_default_flag_keeps_yolov8n(monkeypatch):
    captured: list = []
    _install_fake_ultralytics(monkeypatch, captured, fail=False)
    monkeypatch.delenv("CLIPAI_OBJECT_DETECTOR", raising=False)

    import backend.services.object_detector as od_mod
    od_mod.reset_detector()
    det = od_mod.ObjectDetector(model_dir="/tmp/__nope__")

    # Default behaviour: only yolov8n.pt should have been requested.
    assert all("yolo11n" not in p for p in captured), captured
    assert det.backend_name == "ultralytics-yolov8n"
