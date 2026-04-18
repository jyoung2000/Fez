"""Train the composition head on recovered human trajectories.

MSE loss on the three targets ``cx``, ``cy``, ``zoom``:

    zoom_h = human_crop_w / content_type_default_crop_w

Features come from:
  * the largest face in the source frame (nose_x/y, width, height, yaw),
  * the speaker dwell + Kalman velocities (precomputed cache),
  * the content-type tag from the manifest.

Output: ``data/models/composition_head.pt``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_TRAJ_DIR = Path("data/human_trajectories")
_MODEL_OUT = Path("data/models/composition_head.pt")
_REPORT_OUT = Path("data/models/composition_head.training.json")


_CONTENT_SLOTS = (
    "talking_head", "multi_speaker_panel", "sports",
    "gameplay", "music_video", "animation", "vlog", "other",
)


def _infer_content_family(s: str) -> str:
    v = (s or "").lower()
    if v.startswith("talking"):
        return "talking_head"
    if "panel" in v:
        return "multi_speaker_panel"
    if v.startswith("sports"):
        return "sports"
    if v.startswith("gameplay") or v == "gaming":
        return "gameplay"
    if "music" in v:
        return "music_video"
    if v.startswith("animation") or v == "anime":
        return "animation"
    if "vlog" in v:
        return "vlog"
    return "other"


def _load_dataset(cache_dir: Path, manifest: Path):
    import cv2
    man = json.loads(manifest.read_text())
    X: list[list[float]] = []
    Y: list[list[float]] = []
    for clip in man.get("clips", []):
        slug = clip["slug"]
        traj = _TRAJ_DIR / f"{slug}.jsonl"
        src = cache_dir / f"{slug}.{clip.get('ext', 'mp4')}"
        if not traj.exists() or not src.exists():
            continue
        fam = _infer_content_family(clip.get("target_clipcontenttype", ""))
        one_hot = [0.0] * len(_CONTENT_SLOTS)
        one_hot[_CONTENT_SLOTS.index(fam)] = 1.0
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        clf = cv2.CascadeClassifier(haar_path)
        for line in traj.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(rec["t"] * fps)))
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes = clf.detectMultiScale(gray, 1.2, 4)
            if len(boxes) == 0:
                continue
            x, y, fw, fh = max(boxes, key=lambda b: b[2] * b[3])
            face_cx = (x + fw * 0.5) / w
            face_cy = (y + fh * 0.5) / h
            face_w = fw / w
            face_h = fh / h
            feat = [
                face_cx, face_cy, face_w, face_h,
                0.0,           # yaw (unknown from haar); keep 0
                1.5,           # speaker dwell default
                0.0, 0.0,      # vx, vy unknown here
                float(min(4, len(boxes))),
                0.0,           # motion energy unknown
                2.0,           # shot age default
                *one_hot,
            ]
            X.append(feat)
            zoom_target = rec["w"] / 0.56   # 16:9 9:16 default crop_w_frac ≈ 0.56
            Y.append([rec["cx"], rec["cy"], zoom_target])
        cap.release()
    return X, Y


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="tests/real_content/manifest.json")
    parser.add_argument("--cache-dir", default=os.environ.get(
        "CLIPAI_REAL_CONTENT_CACHE", "/var/cache/clipai/real_content"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        print(f"torch missing: {e}")
        return 2

    X, Y = _load_dataset(Path(args.cache_dir), Path(args.manifest))
    if not X:
        print("empty dataset — populate trajectories first")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y, dtype=torch.float32, device=device)

    class CompositionHead(nn.Module):
        def __init__(self, in_dim: int = 19):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(in_dim, 128), nn.GELU(),
                nn.Linear(128, 64), nn.GELU(),
            )
            self.cx = nn.Linear(64, 1)
            self.cy = nn.Linear(64, 1)
            self.zoom = nn.Linear(64, 1)

        def forward(self, x):
            h = self.trunk(x)
            return self.cx(h), self.cy(h), self.zoom(h)

    model = CompositionHead().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    losses = []
    for ep in range(args.epochs):
        cx, cy, zoom = model(Xt)
        pred = torch.cat([cx, cy, zoom], dim=-1)
        loss = torch.nn.functional.mse_loss(pred, Yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if ep % 5 == 0:
            logger.info("epoch %d loss=%.5f", ep, loss.item())

    _MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), _MODEL_OUT)
    _REPORT_OUT.write_text(json.dumps({
        "samples": len(X),
        "epochs": args.epochs,
        "lr": args.lr,
        "loss_per_epoch": losses,
    }, indent=2))
    print(f"wrote {_MODEL_OUT}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
