"""Phase A — ArcFace (InsightFace buffalo_s) embedding backend tests.

Validates the CLIPAI_FACE_EMBEDDING flag plumbing without requiring the
real insightface / onnxruntime installation:
  - graceful fallback to SFace when insightface isn't importable
  - 512-d embedding shape with a mocked recognition model
  - face_registry threshold constants pivot on the env var
  - mixed-dimension registry input falls back instead of crashing

Heavy native deps (cv2 / mediapipe / google.generativeai) are pre-mocked
so the test imports cleanly in the sandbox CI environment that doesn't
have any of them installed.
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

# ── Pre-mock heavy native deps to avoid import failures in CI ──
for _mod in (
    "google.generativeai", "google.generativeai.types", "google.ai",
    "google.ai.generativelanguage_v1beta", "anthropic", "groq", "httpx",
    "openai", "ctranslate2", "faster_whisper", "mediapipe",
    "mediapipe.python", "mediapipe.python.solutions",
    "mediapipe.python.solutions.face_mesh",
    "mediapipe.python.solutions.face_detection",
    "ultralytics",
):
    sys.modules.setdefault(_mod, MagicMock())


# ── Test 1: graceful fallback when insightface is missing ──

def test_arcface_falls_back_when_insightface_missing(monkeypatch):
    """If `insightface` isn't installed, _get_arcface_app returns None and
    _extract_face_embeddings_arcface returns the input list untouched so
    the caller can fall back to SFace."""
    import backend.services.face_detector as fd

    # Reset the module-level singleton.
    monkeypatch.setattr(fd, "_ARCFACE_APP", None, raising=False)

    # Force the import to fail.
    fake_modules = {"insightface": None, "insightface.app": None}
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "insightface" or name.startswith("insightface."):
            raise ImportError("forced — insightface not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)

    app = fd._get_arcface_app()
    assert app is None

    @dataclass
    class _F:
        x_center: float = 50.0
        y_center: float = 50.0
        width: float = 20.0
        height: float = 25.0
        nose_x: float = 50.0
        nose_y: float = 50.0
        identity_embedding: list = None
        y_bottom: float = 0.0

    faces = [_F()]
    # frame_img is unused when the app is None.
    out = fd._extract_face_embeddings_arcface(None, faces)
    assert out is faces
    assert out[0].identity_embedding is None  # nothing populated, no crash


# ── Test 2: 512-d embedding shape with mocked InsightFace ──

def test_arcface_embedding_dimension(monkeypatch):
    """With a mocked recognition model returning a 512-d vector, the
    extractor populates face.identity_embedding with len == 512."""
    import numpy as np
    import backend.services.face_detector as fd

    monkeypatch.setattr(fd, "_ARCFACE_APP", None, raising=False)

    # Build a fake recognition model whose .get(img, face) returns 512 zeros + 1.
    fake_feat = np.zeros((512,), dtype=np.float32)
    fake_feat[0] = 1.0
    fake_rec_model = MagicMock()
    fake_rec_model.get.return_value = fake_feat

    fake_app = MagicMock()
    fake_app.models = {"recognition": fake_rec_model}

    monkeypatch.setattr(fd, "_get_arcface_app", lambda: fake_app)

    # Stub the IFace import inside the extractor.
    fake_iface_cls = MagicMock()
    fake_common_mod = MagicMock()
    fake_common_mod.Face = fake_iface_cls
    monkeypatch.setitem(sys.modules, "insightface", MagicMock())
    monkeypatch.setitem(sys.modules, "insightface.app", MagicMock())
    monkeypatch.setitem(sys.modules, "insightface.app.common", fake_common_mod)

    @dataclass
    class _F:
        x_center: float = 50.0
        y_center: float = 50.0
        width: float = 30.0
        height: float = 30.0
        nose_x: float = 50.0
        nose_y: float = 50.0
        identity_embedding: list = None
        y_bottom: float = 0.0

    # Frame: 200 x 200 BGR image.
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    faces = [_F()]
    out = fd._extract_face_embeddings_arcface(frame, faces)
    assert out[0].identity_embedding is not None
    assert len(out[0].identity_embedding) == 512
    assert out[0].identity_embedding[0] == pytest.approx(1.0)


# ── Test 3: face_registry threshold constants pivot on the env var ──

def test_face_registry_threshold_selection(monkeypatch):
    """_SAME_ID_COSINE_THRESHOLD and _DIFF_ID_COSINE_THRESHOLD must
    differ between sface and arcface backends, picked up at import time."""
    monkeypatch.setenv("CLIPAI_FACE_EMBEDDING", "arcface")
    import backend.services.face_registry as fr_mod
    importlib.reload(fr_mod)
    same_arc = fr_mod._SAME_ID_COSINE_THRESHOLD
    diff_arc = fr_mod._DIFF_ID_COSINE_THRESHOLD
    teleport_arc = fr_mod._TELEPORT_MERGE_COSINE_SIM

    monkeypatch.setenv("CLIPAI_FACE_EMBEDDING", "sface")
    importlib.reload(fr_mod)
    same_sface = fr_mod._SAME_ID_COSINE_THRESHOLD
    diff_sface = fr_mod._DIFF_ID_COSINE_THRESHOLD
    teleport_sface = fr_mod._TELEPORT_MERGE_COSINE_SIM

    assert same_arc != same_sface
    assert diff_arc != diff_sface
    assert teleport_arc != teleport_sface
    assert same_arc == 0.45 and same_sface == 0.40
    assert diff_arc == 0.28 and diff_sface == 0.35


# ── Test 4: mixed-dimension registry falls back instead of crashing ──

@dataclass
class _FakeFace:
    nose_x: float = 50.0
    nose_y: float = 50.0
    width: float = 12.0
    height: float = 12.0
    identity_embedding: list = None
    identity_id: int = -1
    is_human: bool = True


@dataclass
class _FakeFR:
    timestamp: float = 0.0
    faces: list = field(default_factory=list)
    frame_path: str = ""


def test_mixed_dimension_registry(monkeypatch):
    """A FrameFaces sequence with mixed 128-d and 512-d embeddings must
    NOT crash build_face_registry_with_embeddings — the function must
    detect the heterogeneity and fall back to position-based clustering.
    """
    monkeypatch.setenv("CLIPAI_FACE_EMBEDDING", "sface")
    import backend.services.face_registry as fr_mod
    importlib.reload(fr_mod)

    # Build 6 frames — half 128-d, half 512-d — at two distinct positions.
    frames = []
    for i in range(6):
        f = _FakeFace(nose_x=25.0 if i % 2 == 0 else 75.0)
        dim = 128 if i < 3 else 512
        f.identity_embedding = [1.0 / dim] * dim
        frames.append(_FakeFR(timestamp=float(i), faces=[f]))

    # Should NOT crash on the np.stack — should fall back to position.
    registry = fr_mod.build_face_registry_with_embeddings(
        frames, min_appearances=2,
    )
    # The position-based fallback should produce at least one slot.
    assert registry is not None
    assert isinstance(registry.slots, list)
