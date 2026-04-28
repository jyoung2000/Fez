"""Phase D QA — CLIP composition head + training script extension.

Validates the Phase-D contributions WITHOUT requiring open_clip /
torch / a GPU. The CLIP encoder + torch head are mocked so the
heuristic-blend fallback path is exercised on real data.

Run:  pytest tests/qa/test_phase_d_clip_head.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ──────────────────────────────────────────────────────────────────
# 1. CompositionHints vector
# ──────────────────────────────────────────────────────────────────


class TestCompositionHints:
    def test_to_vector_returns_correct_shape(self):
        from backend.services.composition_head_clip import (
            CompositionHints, SCALAR_FEATURE_DIM,
        )
        v = CompositionHints().to_vector()
        assert len(v) == SCALAR_FEATURE_DIM

    def test_content_type_one_hot(self):
        from backend.services.composition_head_clip import CompositionHints
        v_th = CompositionHints(content_type="talking_head").to_vector()
        v_panel = CompositionHints(content_type="multi_speaker_panel").to_vector()
        # Vectors should differ on the content-type slots.
        assert v_th != v_panel

    def test_face_position_is_passed_through(self):
        from backend.services.composition_head_clip import CompositionHints
        h = CompositionHints(face_cx=0.7, face_cy=0.4, face_w=0.2)
        v = h.to_vector()
        # First entries should reflect the inputs (regardless of layout
        # because the legacy fallback also uses face_cx/cy/w).
        assert any(abs(x - 0.7) < 1e-6 for x in v[:5])
        assert any(abs(x - 0.4) < 1e-6 for x in v[:5])


# ──────────────────────────────────────────────────────────────────
# 2. Mocked CLIP head construction
# ──────────────────────────────────────────────────────────────────


class _FakeClipAdapter:
    def __init__(self, *, device="cuda"):
        self.device = device

    def load(self):
        pass

    def encode(self, frame):
        # Deterministic 512-d unit vector per (mean) pixel.
        rng = np.random.default_rng(int(frame.mean()))
        v = rng.normal(0, 1, 512).astype(np.float32)
        return v / max(np.linalg.norm(v), 1e-9)


class _FakeHead:
    def __init__(self, *, device="cuda"):
        self.device = device
        self._loaded = False

    def load(self, ckpt_path):
        return False  # no-op: forces heuristic fallback

    def predict(self, clip_feat, scalar):
        from backend.services.composition_head_clip import CompositionPrediction
        return CompositionPrediction(cx=0.6, cy=0.4, zoom=1.05, confidence=0.9)


class TestClipCompositionHead:
    def test_construction_falls_to_cpu_when_gpu_absent(self):
        from backend.services.composition_head_clip import ClipCompositionHead
        with mock.patch(
            "backend.services.composition_head_clip._is_gpu_visible",
            return_value=False,
        ):
            h = ClipCompositionHead(
                _adapter_cls=_FakeClipAdapter, _head_cls=_FakeHead,
            )
            assert h.device == "cpu"

    def test_predict_returns_valid_prediction_with_heuristic_fallback(self):
        from backend.services.composition_head_clip import (
            ClipCompositionHead, CompositionHints,
        )
        with mock.patch(
            "backend.services.composition_head_clip._is_gpu_visible",
            return_value=True,
        ):
            h = ClipCompositionHead(
                _adapter_cls=_FakeClipAdapter, _head_cls=_FakeHead,
            )
        frame = np.full((720, 1280, 3), 64, dtype=np.uint8)
        hints = CompositionHints(
            face_cx=0.55, face_cy=0.4, face_w=0.18, face_h=0.30,
            content_type="talking_head", speaker_dwell_sec=2.5,
        )
        pred = h.predict(frame, hints)
        # Should be in the valid range.
        assert 0.0 <= pred.cx <= 1.0
        assert 0.0 <= pred.cy <= 1.0
        assert 0.5 <= pred.zoom <= 2.0
        assert 0.0 <= pred.confidence <= 1.0

    def test_predict_handles_missing_face(self):
        from backend.services.composition_head_clip import (
            ClipCompositionHead, CompositionHints,
        )
        with mock.patch(
            "backend.services.composition_head_clip._is_gpu_visible",
            return_value=True,
        ):
            h = ClipCompositionHead(
                _adapter_cls=_FakeClipAdapter, _head_cls=_FakeHead,
            )
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Default hints = no face.
        pred = h.predict(frame)
        assert 0.0 <= pred.cx <= 1.0
        assert 0.0 <= pred.cy <= 1.0


# ──────────────────────────────────────────────────────────────────
# 3. Heuristic-blend fallback path (no torch checkpoint)
# ──────────────────────────────────────────────────────────────────


class TestHeuristicBlendFallback:
    def test_face_position_drives_cx(self):
        from backend.services.composition_head_clip import (
            _heuristic_blend, CompositionHints,
        )
        # Face on the right side.
        hints = CompositionHints(face_cx=0.7, face_cy=0.4, content_type="vlog")
        clip_feat = np.zeros(512, dtype=np.float32)
        scalar = hints.to_vector()
        pred = _heuristic_blend(clip_feat, scalar, hints)
        # Predicted cx should be on the right side of frame.
        assert pred.cx > 0.5

    def test_no_face_returns_centre(self):
        from backend.services.composition_head_clip import (
            _heuristic_blend, CompositionHints,
        )
        hints = CompositionHints()  # face_cx = -1 → no face
        pred = _heuristic_blend(
            np.zeros(512, dtype=np.float32), hints.to_vector(), hints,
        )
        # The legacy heuristic snaps no-face to centre / nearest third.
        assert 0.3 <= pred.cx <= 0.7


# ──────────────────────────────────────────────────────────────────
# 4. Parity: face_clipping_rate (Phase D's primary metric)
# ──────────────────────────────────────────────────────────────────


class TestFaceClippingRate:
    def test_face_inside_crop(self):
        from backend.services.autoflip_parity_metrics import face_clipping_rate
        # Crop centred at 50, width 40 → covers [30, 70]
        crops = [50.0, 50.0]
        faces = [[(35.0, 50.0, 30.0, 30.0)], [(40.0, 50.0, 25.0, 30.0)]]
        # First face: x=35 to 65 — inside [30,70]. Second: 40 to 65 — inside.
        rate = face_clipping_rate(crops, 40.0, faces)
        assert rate == pytest.approx(0.0)

    def test_face_clipped_left_or_right(self):
        from backend.services.autoflip_parity_metrics import face_clipping_rate
        # Crop covers [30, 70]; face at x=10..30 is clipped on left.
        crops = [50.0]
        faces = [[(10.0, 50.0, 20.0, 30.0)]]
        rate = face_clipping_rate(crops, 40.0, faces)
        assert rate == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────
# 5. Training script: --clip-head flag wiring
# ──────────────────────────────────────────────────────────────────


class TestTrainingScript:
    def test_clip_head_flag_present(self):
        # Static check: the script accepts the new flag.
        src = Path(_REPO / "backend" / "scripts" / "train_composition_head.py").read_text()
        assert "--clip-head" in src
        assert "_train_clip_head" in src
        assert "ViT-B-32" in src

    def test_clip_head_outputs_phase_d_checkpoint(self):
        src = Path(_REPO / "backend" / "scripts" / "train_composition_head.py").read_text()
        assert "composition_head_clip_v1.pt" in src

    def test_smooth_l1_loss_is_used(self):
        src = Path(_REPO / "backend" / "scripts" / "train_composition_head.py").read_text()
        assert "smooth_l1" in src.lower()


# ──────────────────────────────────────────────────────────────────
# 6. Config flag
# ──────────────────────────────────────────────────────────────────


class TestConfigFlag:
    def test_composition_head_default_clip_after_phase_e(self):
        """Phase D shipped CLIPAI_COMPOSITION_HEAD default 'legacy'.
        Phase E flipped it to 'clip'; this test checks the shipped default.
        Set CLIPAI_COMPOSITION_HEAD=legacy to opt out per-deploy.
        """
        try:
            from backend.config import Settings
        except ModuleNotFoundError as exc:
            pytest.skip(f"backend.config deps missing: {exc}")
        s = Settings()
        assert s.CLIPAI_COMPOSITION_HEAD == "clip"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
