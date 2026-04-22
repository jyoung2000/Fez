"""Blueprint v2 Phase 3 — reframe-confidence virality multiplier.

Exposes two pure helpers (no pydantic / ClipCandidate deps) so tests
and downstream callers can scale a virality score by a clip's reframe
confidence without importing the full clip_scoring graph.

A clip that scores 90 on content but can't be cropped to 9:16 without
losing the subject is strictly worse for the user than a clip that
scores 70 and reframes cleanly. These multipliers surface the well-
reframed clip first.

Overridable via ``CLIPAI_VIRALITY_CONFIDENCE_<HIGH|MEDIUM|LOW>`` env
vars so QA can sweep ratios without a code change.
"""

from __future__ import annotations

import os
from typing import Optional


CLIP_CONFIDENCE_VIRALITY_MULTIPLIER: dict[str, float] = {
    "high": 1.0,
    "medium": 0.95,
    "low": 0.6,
}


def _confidence_multiplier(confidence: Optional[str]) -> float:
    """Return the virality multiplier for a given reframe confidence.

    Unknown / ``None`` confidence → 1.0 (no change). Env override
    per-level takes precedence over the defaults and is clamped to
    ``[0.0, 1.0]``.
    """
    if not confidence:
        return 1.0
    key = str(confidence).lower()
    env = os.environ.get(f"CLIPAI_VIRALITY_CONFIDENCE_{key.upper()}")
    if env is not None:
        try:
            return max(0.0, min(1.0, float(env)))
        except ValueError:
            pass
    return CLIP_CONFIDENCE_VIRALITY_MULTIPLIER.get(key, 1.0)


def apply_clip_confidence(
    viral_score: int,
    confidence: Optional[str],
) -> int:
    """Scale a virality score by its reframe confidence.

    Clamped to ``[1, 100]`` so the composite always stays in range.
    """
    try:
        raw = int(round(float(viral_score) * _confidence_multiplier(confidence)))
    except (TypeError, ValueError):
        return int(viral_score or 0)
    return max(1, min(100, raw))
