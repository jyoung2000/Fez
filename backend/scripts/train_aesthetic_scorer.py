"""Train the aesthetic scorer head.

Pairwise margin loss over (human_crop, perturbed_crop) frame pairs:

    L = max(0, margin - score(human) + score(perturbed))

Data source: every clip in ``tests/real_content/manifest.json`` with a
``ground_truth_vertical_url`` has an extracted human trajectory at
``data/human_trajectories/<slug>.jsonl``. For each frame we:

  1. Decode the source frame.
  2. Crop it using the human-recovered rect → positive example.
  3. Crop it using 4 perturbed rects (shift ±0.1, scale ±15 %) →
     negatives.
  4. Embed all crops with CLIP ViT-B/32.
  5. Train the head.

Output: ``data/models/aesthetic_scorer.pt`` plus a short metric report
at ``data/models/aesthetic_scorer.training.json``.

Requires ``torch`` and ``open_clip_torch``. Runs on CPU in minutes for
the ~13-clip bench; CUDA if available.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_TRAJ_DIR = Path("data/human_trajectories")
_MODEL_OUT = Path("data/models/aesthetic_scorer.pt")
_REPORT_OUT = Path("data/models/aesthetic_scorer.training.json")

_PERTURB_SHIFTS = [(-0.1, 0.0), (0.1, 0.0), (0.0, -0.1), (0.0, 0.1)]
_PERTURB_SCALES = [0.85, 1.15]


def _perturb_rect(r: dict, shift: tuple[float, float], scale: float) -> dict:
    dx, dy = shift
    w = max(0.05, min(1.0, r["w"] * scale))
    h = max(0.05, min(1.0, r["h"] * scale))
    cx = r["x"] + r["w"] * 0.5 + dx
    cy = r["y"] + r["h"] * 0.5 + dy
    x = max(0.0, min(1.0 - w, cx - w * 0.5))
    y = max(0.0, min(1.0 - h, cy - h * 0.5))
    return {"x": x, "y": y, "w": w, "h": h}


def _load_pairs(cache_dir: Path, manifest_path: Path):
    import cv2
    manifest = json.loads(manifest_path.read_text())
    pairs = []
    for clip in manifest.get("clips", []):
        slug = clip["slug"]
        traj_path = _TRAJ_DIR / f"{slug}.jsonl"
        src_path = cache_dir / f"{slug}.{clip.get('ext', 'mp4')}"
        if not traj_path.exists() or not src_path.exists():
            continue
        cap = cv2.VideoCapture(str(src_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        for line in traj_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            t = float(rec["t"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            pos_rect = {
                "x": rec["cx"] - rec["w"] * 0.5,
                "y": rec["cy"] - rec["h"] * 0.5,
                "w": rec["w"],
                "h": rec["h"],
            }
            for shift in _PERTURB_SHIFTS:
                for scale in _PERTURB_SCALES:
                    neg = _perturb_rect(pos_rect, shift, scale)
                    pairs.append((frame, pos_rect, neg))
        cap.release()
    return pairs


def _crop_px(frame, rect: dict):
    import cv2
    import numpy as np
    h, w = frame.shape[:2]
    x0 = int(max(0, rect["x"] * w))
    y0 = int(max(0, rect["y"] * h))
    x1 = int(min(w, x0 + rect["w"] * w))
    y1 = int(min(h, y0 + rect["h"] * h))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    return cv2.resize(frame[y0:y1, x0:x1], (224, 224))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="tests/real_content/manifest.json")
    parser.add_argument("--cache-dir", default=os.environ.get(
        "CLIPAI_REAL_CONTENT_CACHE", "/var/cache/clipai/real_content"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        import open_clip
        from PIL import Image
    except Exception as e:
        print(f"required packages missing: {e}")
        return 2

    pairs = _load_pairs(Path(args.cache_dir), Path(args.manifest))
    if not pairs:
        print("no pairs found; populate manifest + trajectories first")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k",
    )
    clip_model = clip_model.to(device).eval()

    class AestheticHead(nn.Module):
        def __init__(self, in_dim: int = 512):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim + 4, 256), nn.GELU(),
                nn.Linear(256, 128), nn.GELU(),
                nn.Linear(128, 1), nn.Sigmoid(),
            )

        def forward(self, emb, crop):
            return self.net(torch.cat([emb, crop], dim=-1))

    head = AestheticHead().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)

    def embed(frame_bgr, rect):
        arr = _crop_px(frame_bgr, rect)
        img = Image.fromarray(arr[..., ::-1])
        x = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            e = clip_model.encode_image(x)
            e = e / e.norm(dim=-1, keepdim=True)
        crop_t = torch.tensor([[rect["x"], rect["y"], rect["w"], rect["h"]]],
                              dtype=e.dtype, device=device)
        return e, crop_t

    losses: list[float] = []
    for ep in range(args.epochs):
        random.shuffle(pairs)
        ep_loss = 0.0
        for frame, pos, neg in pairs:
            ep, cp = embed(frame, pos)
            en, cn = embed(frame, neg)
            sp = head(ep, cp)
            sn = head(en, cn)
            loss = torch.clamp(args.margin - sp + sn, min=0.0).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
        logger.info("epoch %d loss=%.4f", ep, ep_loss / max(len(pairs), 1))
        losses.append(ep_loss / max(len(pairs), 1))

    _MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    import torch as _t
    _t.save(head.state_dict(), _MODEL_OUT)
    _REPORT_OUT.write_text(json.dumps({
        "pairs": len(pairs),
        "epochs": args.epochs,
        "lr": args.lr,
        "loss_per_epoch": losses,
    }, indent=2))
    print(f"wrote {_MODEL_OUT}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
