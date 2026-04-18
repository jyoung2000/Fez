"""Cut-timing head.

Small binary classifier that predicts "would a human cut here" on
sliding windows around candidate cut points (shot boundaries, speaker
turns, intent switches). Trained on the ms offset between human cuts
and detected speaker / shot events across the real-content manifest.

Why we need this separate from the saccade scheduler:

  The saccade scheduler (Phase 4) knows *that* an event fires; what it
  doesn't know is *when exactly* a human editor would cut around that
  event. Human cuts are typically 40-120 ms **before** the audio turn
  onset (J-cut); action cuts are often 60-100 ms **after** a hit frame
  (L-cut). Pretending they're all zero-ms cuts makes the output feel
  un-edited.

Feature layout (11 floats):

  [0]    time-to-next-shot-boundary (sec, clipped to 3.0)
  [1]    time-since-last-shot-boundary (sec, clipped to 3.0)
  [2]    audio-energy delta in 100ms window (RMS log ratio, -3..3)
  [3]    speaker-turn-offset (sec, -0.5..0.5; negative = J-cut territory)
  [4]    reaction-flag (0/1, from audio_analyzer laughter/gasp)
  [5]    motion-beat (0..1)
  [6]    intent-switch (0..1)
  [7:11] one-hot content family (dialogue, action, music, generic)

Output: a scalar offset in seconds to add to the candidate cut time
(clipped to ±0.25 sec) and a confidence score 0..1.

Fallback (no trained weights): a deterministic table keyed off
content family + trigger type. e.g. dialogue speaker-turn → −0.08 s;
action shot-cut + motion-beat → +0.08 s; music beat-lock → 0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MODEL_PATH = Path("data/models/cut_timing_head.pt")
_FEATURE_DIM = 11
_CONTENT_SLOTS = ("dialogue", "action", "music", "generic")

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

        class CutTimingHead(nn.Module):
            def __init__(self, in_dim: int = _FEATURE_DIM):
                super().__init__()
                self.trunk = nn.Sequential(
                    nn.Linear(in_dim, 64),
                    nn.GELU(),
                    nn.Linear(64, 32),
                    nn.GELU(),
                )
                self.offset = nn.Linear(32, 1)
                self.conf = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())

            def forward(self, x):
                h = self.trunk(x)
                return self.offset(h), self.conf(h)

        model = CutTimingHead()
        state = torch.load(_MODEL_PATH, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        _CACHED_MODEL = model
        return model
    except Exception as e:
        logger.warning("cut_timing_head model load failed: %s", e)
        return None


@dataclass
class CutFeatures:
    time_to_next_shot: float = 3.0
    time_since_last_shot: float = 3.0
    audio_energy_delta: float = 0.0
    speaker_turn_offset: float = 0.0
    reaction_flag: float = 0.0
    motion_beat: float = 0.0
    intent_switch: float = 0.0
    content_family: str = "generic"
    trigger_kind: str = "shot_cut"

    def to_vector(self) -> list[float]:
        one_hot = [0.0] * len(_CONTENT_SLOTS)
        fam = self.content_family if self.content_family in _CONTENT_SLOTS else "generic"
        one_hot[_CONTENT_SLOTS.index(fam)] = 1.0
        return [
            max(0.0, min(3.0, self.time_to_next_shot)),
            max(0.0, min(3.0, self.time_since_last_shot)),
            max(-3.0, min(3.0, self.audio_energy_delta)),
            max(-0.5, min(0.5, self.speaker_turn_offset)),
            1.0 if self.reaction_flag else 0.0,
            max(0.0, min(1.0, self.motion_beat)),
            max(0.0, min(1.0, self.intent_switch)),
            *one_hot,
        ]


# ── Fallback table ──────────────────────────────────────────────


_FALLBACK_OFFSETS = {
    # (content_family, trigger_kind) → (offset_sec, confidence)
    ("dialogue", "speaker_turn"):  (-0.08, 0.6),   # J-cut before turn
    ("dialogue", "reaction"):      (+0.05, 0.6),   # land on the react
    ("dialogue", "shot_cut"):      (-0.02, 0.4),   # slight lead
    ("action",   "motion_beat"):   (+0.08, 0.7),   # land after the hit
    ("action",   "shot_cut"):      (+0.00, 0.5),
    ("music",    "beat"):          (+0.00, 0.9),   # lock to beat
    ("music",    "shot_cut"):      (+0.00, 0.6),
    ("generic",  "shot_cut"):      (+0.00, 0.5),
}


def _fallback(feat: CutFeatures) -> tuple[float, float]:
    key = (feat.content_family, feat.trigger_kind)
    if key in _FALLBACK_OFFSETS:
        return _FALLBACK_OFFSETS[key]
    # Partial match by trigger
    for (fam, trig), val in _FALLBACK_OFFSETS.items():
        if trig == feat.trigger_kind:
            return val
    return (0.0, 0.3)


# ── Public API ─────────────────────────────────────────────────


def predict_cut_offset(feat: CutFeatures) -> tuple[float, float]:
    """Return ``(offset_sec, confidence)`` for a candidate cut."""
    model = _load_model()
    if model is None:
        return _fallback(feat)
    try:
        import torch
        x = torch.tensor([feat.to_vector()], dtype=torch.float32)
        with torch.no_grad():
            off, conf = model(x)
        return (
            float(max(-0.25, min(0.25, off.item()))),
            float(max(0.0, min(1.0, conf.item()))),
        )
    except Exception as e:
        logger.info("cut_timing_head inference failed (%s); falling back", e)
        return _fallback(feat)
