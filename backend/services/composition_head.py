"""Learned composition head.

Given per-frame detection features, predicts the ``(cx, cy, zoom)`` a
human editor would pick for a 9:16 crop. Trained with MSE vs human-
recovered trajectories (see
``backend/scripts/extract_human_trajectories.py``).

Integration: the 2-D LP solver in :mod:`backend.services.camera_path_2d`
optionally accepts this head's predictions as an additional per-frame
data term. With a small weight (``lam_comp = 0.3``) the head biases the
solver toward human-like compositions without overriding the rigorous
headroom / chin-clip constraints.

Feature layout (19 floats / frame):

  [0:2]   dominant face nose_x / nose_y (0..1; -1 when absent)
  [2:4]   dominant face width / height (0..1; 0 when absent)
  [4]     dominant face yaw (-1..1; 0 = forward / unknown)
  [5]     speaker dwell time in current turn (sec; clipped to 5.0)
  [6:8]   Kalman velocity vx / vy (normalized / sec; clipped to ±1)
  [8]     n_faces_in_frame  (clipped to 4)
  [9]     motion energy (0..1)
  [10]    shot age (sec into current shot; clipped to 10.0)
  [11:19] one-hot content type (talking_head, multi_speaker_panel,
          sports, gaming, music, animation, vlog, other)

When weights are missing the head falls back to a hand-tuned linear
combination of the features that roughly matches the cinematography
rules (face on thirds + eye-line + velocity-weighted lead-room).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MODEL_PATH = Path("data/models/composition_head.pt")
_FEATURE_DIM = 19

_CACHED_MODEL = None


def _load_model():
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL
    if not _MODEL_PATH.exists():
        return None
    try:
        import torch
        import torch.nn as nn

        class CompositionHead(nn.Module):
            def __init__(self, in_dim: int = _FEATURE_DIM):
                super().__init__()
                self.trunk = nn.Sequential(
                    nn.Linear(in_dim, 128),
                    nn.GELU(),
                    nn.Linear(128, 64),
                    nn.GELU(),
                )
                self.cx = nn.Linear(64, 1)
                self.cy = nn.Linear(64, 1)
                self.zoom = nn.Linear(64, 1)

            def forward(self, x):
                h = self.trunk(x)
                return self.cx(h), self.cy(h), self.zoom(h)

        model = CompositionHead()
        state = torch.load(_MODEL_PATH, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        _CACHED_MODEL = model
        return model
    except Exception as e:
        logger.warning("composition_head model load failed: %s", e)
        return None


# ── Feature builder ─────────────────────────────────────────────


_CONTENT_SLOTS = (
    "talking_head", "multi_speaker_panel", "sports",
    "gameplay", "music_video", "animation", "vlog", "other",
)


@dataclass
class CompositionFeatures:
    face_cx: float = -1.0
    face_cy: float = -1.0
    face_w: float = 0.0
    face_h: float = 0.0
    face_yaw: float = 0.0
    speaker_dwell_sec: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    n_faces: int = 0
    motion_energy: float = 0.0
    shot_age_sec: float = 0.0
    content_type: str = "other"

    def to_vector(self) -> list[float]:
        one_hot = [0.0] * len(_CONTENT_SLOTS)
        ct = self.content_type if self.content_type in _CONTENT_SLOTS else "other"
        one_hot[_CONTENT_SLOTS.index(ct)] = 1.0
        return [
            self.face_cx, self.face_cy,
            self.face_w, self.face_h,
            max(-1.0, min(1.0, self.face_yaw)),
            max(0.0, min(5.0, self.speaker_dwell_sec)),
            max(-1.0, min(1.0, self.vx)),
            max(-1.0, min(1.0, self.vy)),
            float(max(0, min(4, self.n_faces))),
            max(0.0, min(1.0, self.motion_energy)),
            max(0.0, min(10.0, self.shot_age_sec)),
            *one_hot,
        ]


# ── Fallback heuristic ─────────────────────────────────────────


def _heuristic_predict(feat: CompositionFeatures) -> tuple[float, float, float]:
    """Hand-tuned linear combination as a stand-in when weights missing.

    Approximates human practice: face on nearest third, eye-line ~0.35,
    mild lead-room from yaw, zoom tightens on close-up + talking head.
    """
    fcx = feat.face_cx if feat.face_cx >= 0 else 0.5
    fcy = feat.face_cy if feat.face_cy >= 0 else 0.5
    # Pull toward nearest third
    near_third = 1.0 / 3.0 if abs(fcx - 1.0 / 3.0) < abs(fcx - 2.0 / 3.0) else 2.0 / 3.0
    tx = 0.5 * fcx + 0.5 * near_third
    # Eye-line bias: keep face's eye line (0.25*h above nose) near 0.35
    ty = 0.5 * fcy + 0.5 * (fcy - (0.35 - 0.5))
    # Lead-room
    tx -= feat.face_yaw * 0.08
    # Zoom: push in more for close-ups with strong speaker dwell
    zoom = 1.0
    if feat.content_type in ("talking_head", "vlog") and feat.speaker_dwell_sec > 2.0:
        zoom = min(1.18, 1.0 + (feat.speaker_dwell_sec - 2.0) * 0.04)
    if feat.content_type in ("gameplay", "sports") and feat.motion_energy > 0.7:
        zoom = max(0.85, 1.0 - (feat.motion_energy - 0.7) * 0.5)
    return (
        float(max(0.0, min(1.0, tx))),
        float(max(0.0, min(1.0, ty))),
        float(max(0.5, min(2.0, zoom))),
    )


# ── Public API ─────────────────────────────────────────────────


def predict_composition(feat: CompositionFeatures) -> tuple[float, float, float]:
    """Return ``(cx, cy, zoom)`` in source-frame fractions / scale units."""
    model = _load_model()
    if model is None:
        return _heuristic_predict(feat)
    try:
        import torch
        x = torch.tensor([feat.to_vector()], dtype=torch.float32)
        with torch.no_grad():
            cx, cy, zoom = model(x)
        return (
            float(max(0.0, min(1.0, cx.item()))),
            float(max(0.0, min(1.0, cy.item()))),
            float(max(0.5, min(2.0, zoom.item()))),
        )
    except Exception as e:
        logger.info("composition_head inference failed (%s); falling back", e)
        return _heuristic_predict(feat)
