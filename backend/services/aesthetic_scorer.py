"""Learned aesthetic scorer.

Small MLP head on top of a CLIP (or Qwen-VL vision-encoder) embedding
that predicts a 0-1 "how likely a human editor produced this crop"
score. Trained with pairwise margin loss on (human_crop, perturbed_crop)
pairs at the same timestamp.

Loading strategy:

  * Inference: if ``data/models/aesthetic_scorer.pt`` exists, use it.
    Otherwise fall back to a deterministic geometric heuristic (face in
    thirds + headroom + no chin clip) so the critic loop still produces
    meaningful scores when weights haven't been trained yet.
  * Training: see ``backend/scripts/train_aesthetic_scorer.py``. Not
    auto-invoked — the human-trajectory extractor
    (``extract_human_trajectories.py``) has to run first.

Public entry point: ``score_frame(frame_path) -> float`` in [0, 1].
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MODEL_PATH = Path("data/models/aesthetic_scorer.pt")
_EMBED_DIM = 512


# ── Lazy model loading ───────────────────────────────────────────

_CACHED_MODEL = None
_CACHED_CLIP = None


def _load_model():
    """Load the trained PyTorch model if available; else return None."""
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL
    if not _MODEL_PATH.exists():
        return None
    try:
        import torch
        import torch.nn as nn

        class AestheticHead(nn.Module):
            def __init__(self, in_dim: int = _EMBED_DIM):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim + 4, 256),   # +4 for crop rect
                    nn.GELU(),
                    nn.Linear(256, 128),
                    nn.GELU(),
                    nn.Linear(128, 1),
                    nn.Sigmoid(),
                )

            def forward(self, emb, crop):
                return self.net(torch.cat([emb, crop], dim=-1))

        model = AestheticHead()
        state = torch.load(_MODEL_PATH, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        _CACHED_MODEL = model
        return model
    except Exception as e:
        logger.warning("aesthetic_scorer model load failed: %s", e)
        return None


def _load_clip():
    """Lazy-load CLIP image encoder from ``open_clip`` if available."""
    global _CACHED_CLIP
    if _CACHED_CLIP is not None:
        return _CACHED_CLIP
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k",
        )
        model.eval()
        _CACHED_CLIP = (model, preprocess)
        return _CACHED_CLIP
    except Exception:
        return None


# ── Heuristic fallback ───────────────────────────────────────────


def _heuristic_score_from_path(frame_path: str) -> float:
    """Cheap geometric aesthetic proxy when no model is loaded.

    Detects the largest face with a Haar cascade and rewards:
      * face top in [0.05, 0.2] of frame (headroom)
      * face center on a thirds line (x ≈ 1/3 or 2/3, y ≈ 0.35)
      * face bottom <= 0.95 (no chin clip)

    When OpenCV is unavailable the function still returns a sensible
    neutral score so the critic never fails; it does NOT require
    network or any model files to produce a value.
    """
    try:
        import cv2
    except Exception:
        # No CV → conservative positive score. The critic treats this
        # as "no evidence of a bad crop" rather than "needs re-solve".
        return 0.6
    try:
        frame = cv2.imread(frame_path)
    except Exception:
        return 0.5
    if frame is None:
        return 0.5
    H, W = frame.shape[:2]
    try:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        clf = cv2.CascadeClassifier(path)
        if clf.empty():
            return 0.5
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = clf.detectMultiScale(gray, 1.2, 4)
        if len(boxes) == 0:
            return 0.4
        x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
        top = y / H
        bot = (y + h) / H
        cx = (x + w * 0.5) / W
        cy_eye = (y + h * 0.35) / H

        headroom_bonus = 1.0 - min(1.0, abs(top - 0.12) / 0.15)
        chin_bonus = 1.0 if bot < 0.95 else 0.3
        thirds_dist = min(abs(cx - 1 / 3), abs(cx - 2 / 3))
        thirds_bonus = 1.0 - min(1.0, thirds_dist / 0.20)
        eye_line_bonus = 1.0 - min(1.0, abs(cy_eye - 0.35) / 0.25)
        score = (headroom_bonus + chin_bonus + thirds_bonus + eye_line_bonus) / 4.0
        return float(max(0.0, min(1.0, score)))
    except Exception:
        return 0.5


# ── Public API ───────────────────────────────────────────────────


def score_frame(frame_path: Optional[str], crop_rect: Optional[dict] = None) -> float:
    """Score a single frame (0 = bad, 1 = great)."""
    if not frame_path or not os.path.exists(frame_path):
        return 0.5

    model = _load_model()
    clip = _load_clip()
    if model is None or clip is None:
        return _heuristic_score_from_path(frame_path)

    try:
        import torch
        from PIL import Image
        clip_model, preprocess = clip
        img = preprocess(Image.open(frame_path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            emb = clip_model.encode_image(img)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            rect = crop_rect or {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
            crop_t = torch.tensor([[rect["x"], rect["y"], rect["w"], rect["h"]]],
                                  dtype=emb.dtype)
            score = model(emb, crop_t).item()
        return float(max(0.0, min(1.0, score)))
    except Exception as e:
        logger.info("aesthetic_scorer inference failed (%s); falling back", e)
        return _heuristic_score_from_path(frame_path)
