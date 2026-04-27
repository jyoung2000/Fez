"""OCR text-region detector — PaddleOCR-lite.

Phase C of the 2026 SOTA reframing rollout.

What this module does:
    * Samples one frame per scene (cheap — ~5 frames per minute on a
      typical clip), runs PaddleOCR-lite to detect text-region
      bounding boxes, and emits them as **HUD-style exclusion zones**
      consumed by the LP solver. Stops "FINAL SCORE: 89-86" from
      being mid-cropped to "INAL SCO" or "FINAL SCO" at the seams.

Why PaddleOCR-lite:
    * 50 MB Python wheel; runs CPU at ~5 FPS on 1080p input.
    * No GPU dependency — runs on the no-GPU fallback machines too.
    * Mature library, pip-installable, cross-platform.

Output per frame:
    list[(x_pct, y_pct, w_pct, h_pct)] — text bboxes in % of source
    frame width / height (matching the rest of the parity bench
    convention).

Public API:

    detector = OcrRegionDetector()
    regions = detector.detect(frame_bgr)   # one frame
    regions_seq = detector.detect_scenes(frames, scene_starts)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────

DEFAULT_MIN_CONFIDENCE = 0.6     # PaddleOCR confidence threshold
DEFAULT_MIN_AREA_PCT = 0.05      # ignore text smaller than this (likely noise)
PADDLEOCR_LANG = "en"


# ── Dataclasses ──────────────────────────────────────────────────────


@dataclass
class TextRegion:
    """One detected text-region bbox + confidence + raw text."""

    x_pct: float            # 0..100
    y_pct: float            # 0..100
    w_pct: float
    h_pct: float
    text: str = ""
    confidence: float = 1.0


# ── PaddleOCR adapter ────────────────────────────────────────────────


class _PaddleOcrAdapter:
    """Lazy wrapper around the PaddleOCR Python wheel.

    Isolated for unit-test mocking. The real implementation downloads
    PP-OCRv4 detector + recognizer weights on first use.
    """

    def __init__(
        self,
        *,
        lang: str = PADDLEOCR_LANG,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ):
        self.lang = lang
        self.min_confidence = min_confidence
        self._ocr: Optional[Any] = None

    def load(self) -> None:
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "paddleocr not installed; pin in requirements.txt and rebuild"
            ) from exc
        self._ocr = PaddleOCR(use_angle_cls=False, lang=self.lang, show_log=False)

    def detect(self, frame_bgr: np.ndarray) -> list:
        """Return raw PaddleOCR result for one frame.

        Each entry is ``[bbox_4_corners, (text, score)]``. Caller
        post-processes to ``(x_pct, y_pct, w_pct, h_pct)``.
        """
        if self._ocr is None:
            self.load()
        try:
            return self._ocr.ocr(frame_bgr, cls=False) or []  # type: ignore
        except Exception as exc:
            logger.warning("paddleocr.ocr failed: %s", exc)
            return []


# ── Helpers (testable without paddle) ────────────────────────────────


def _bbox_from_quadrilateral(quad) -> tuple:
    """Convert PaddleOCR's 4-point quadrilateral to ``(x, y, w, h)``."""
    if not quad or len(quad) < 4:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [float(p[0]) for p in quad]
    ys = [float(p[1]) for p in quad]
    x0 = min(xs)
    x1 = max(xs)
    y0 = min(ys)
    y1 = max(ys)
    return (x0, y0, x1 - x0, y1 - y0)


def _to_pct(bbox_px: tuple, frame_h: int, frame_w: int) -> tuple:
    """Convert pixel-space bbox to percentage of source frame width/height."""
    x, y, w, h = bbox_px
    return (
        x / max(frame_w, 1) * 100.0,
        y / max(frame_h, 1) * 100.0,
        w / max(frame_w, 1) * 100.0,
        h / max(frame_h, 1) * 100.0,
    )


def filter_regions(
    regions: list,
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_area_pct: float = DEFAULT_MIN_AREA_PCT,
) -> list:
    """Drop low-confidence and very small regions."""
    out = []
    for r in regions:
        if r.confidence < min_confidence:
            continue
        if (r.w_pct * r.h_pct) < min_area_pct:
            continue
        out.append(r)
    return out


# ── Public class ─────────────────────────────────────────────────────


@dataclass
class OcrRegionDetector:
    lang: str = PADDLEOCR_LANG
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    min_area_pct: float = DEFAULT_MIN_AREA_PCT
    _adapter_cls: type = field(default_factory=lambda: _PaddleOcrAdapter, repr=False)

    def __post_init__(self):
        self._adapter = self._adapter_cls(
            lang=self.lang, min_confidence=self.min_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> list:
        """Return ``list[TextRegion]`` for one frame."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        h, w = frame_bgr.shape[:2]
        raw = self._adapter.detect(frame_bgr)
        regions = []
        # PaddleOCR returns ``[[ [bbox, (text, score)], ... ]]``
        for item in (raw[0] if raw and len(raw) > 0 else []):
            try:
                quad, (text, score) = item
            except (ValueError, TypeError):
                continue
            bbox_px = _bbox_from_quadrilateral(quad)
            if bbox_px[2] <= 0 or bbox_px[3] <= 0:
                continue
            x_pct, y_pct, w_pct, h_pct = _to_pct(bbox_px, h, w)
            regions.append(TextRegion(
                x_pct=x_pct, y_pct=y_pct,
                w_pct=w_pct, h_pct=h_pct,
                text=str(text or ""),
                confidence=float(score),
            ))
        return filter_regions(
            regions,
            min_confidence=self.min_confidence,
            min_area_pct=self.min_area_pct,
        )

    def detect_scenes(
        self,
        frames: list,
        scene_indices: Optional[list] = None,
    ) -> list:
        """Run OCR on one frame per scene; return ``list[list[TextRegion]]``.

        ``scene_indices`` is the frame index to sample for each scene.
        When None, samples every frame (caller should down-sample).
        """
        if scene_indices is None:
            scene_indices = list(range(len(frames)))
        out = []
        for idx in scene_indices:
            if 0 <= idx < len(frames):
                out.append(self.detect(frames[idx]))
            else:
                out.append([])
        return out
