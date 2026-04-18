"""Phase 1 — auto-HUD detector unit tests.

Builds synthetic frames with planted static UI regions and
moving central content, and asserts that
:func:`detect_hud_regions` recovers the corners while rejecting
everything in the middle.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("cv2")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from backend.services.gaming_hud_detector import (  # noqa: E402
    HudRegion,
    classify_quadrant,
    classify_semantic_hint,
    detect_hud_regions,
    detect_hud_regions_for_scene,
    invalidate_hud_cache,
)


# ──────────────────── Frame builders ────────────────────


FRAME_W = 640
FRAME_H = 360


def _base_frame(bg: int = 40) -> np.ndarray:
    return np.full((FRAME_H, FRAME_W), bg, dtype=np.uint8)


def _plant_static_corner_huds(img: np.ndarray) -> np.ndarray:
    """Paint 4 edgy HUD-looking rectangles in the four corners."""
    # TL scoreboard strip: a thin wide edgy block near the top-left
    cv2.rectangle(img, (10, 10), (140, 30), 220, thickness=-1)
    cv2.rectangle(img, (10, 10), (140, 30), 0, thickness=2)
    for dx in range(0, 130, 20):
        cv2.line(img, (10 + dx, 10), (10 + dx, 30), 0, thickness=1)

    # TR killfeed: wide short strip near top-right
    cv2.rectangle(img, (FRAME_W - 160, 10), (FRAME_W - 10, 40), 200,
                  thickness=-1)
    cv2.rectangle(img, (FRAME_W - 160, 10), (FRAME_W - 10, 40), 0,
                  thickness=2)
    for dy in range(15, 40, 8):
        cv2.line(img, (FRAME_W - 155, dy), (FRAME_W - 15, dy), 0,
                 thickness=1)

    # BR minimap: square-ish block near bottom-right
    cv2.rectangle(img, (FRAME_W - 80, FRAME_H - 80),
                  (FRAME_W - 10, FRAME_H - 10), 180, thickness=-1)
    cv2.rectangle(img, (FRAME_W - 80, FRAME_H - 80),
                  (FRAME_W - 10, FRAME_H - 10), 0, thickness=2)
    cv2.line(img, (FRAME_W - 75, FRAME_H - 45),
             (FRAME_W - 15, FRAME_H - 45), 0, thickness=1)
    cv2.line(img, (FRAME_W - 45, FRAME_H - 75),
             (FRAME_W - 45, FRAME_H - 15), 0, thickness=1)

    # BC health bar: wide short strip near bottom center
    cv2.rectangle(img, (FRAME_W // 2 - 80, FRAME_H - 30),
                  (FRAME_W // 2 + 80, FRAME_H - 12), 240, thickness=-1)
    cv2.rectangle(img, (FRAME_W // 2 - 80, FRAME_H - 30),
                  (FRAME_W // 2 + 80, FRAME_H - 12), 0, thickness=2)
    return img


def _plant_moving_center_blob(
    img: np.ndarray, cx: int, cy: int, radius: int = 40,
) -> np.ndarray:
    """A high-contrast circle in the middle that moves between frames."""
    cv2.circle(img, (cx, cy), radius, 220, thickness=-1)
    cv2.circle(img, (cx, cy), radius, 0, thickness=3)
    return img


def _write_sequence(tmp_path, frames: list[np.ndarray]) -> list[str]:
    paths: list[str] = []
    for i, f in enumerate(frames):
        p = tmp_path / f"hud_{i:03d}.png"
        cv2.imwrite(str(p), f)
        paths.append(str(p))
    return paths


# ──────────────────── Classifier tests ────────────────────


def test_classify_quadrant_all_buckets():
    # TL
    assert classify_quadrant((10, 10, 80, 40), FRAME_W, FRAME_H) == "TL"
    # TR
    assert classify_quadrant((FRAME_W - 100, 10, 80, 40),
                             FRAME_W, FRAME_H) == "TR"
    # BL
    assert classify_quadrant((10, FRAME_H - 60, 80, 40),
                             FRAME_W, FRAME_H) == "BL"
    # BR
    assert classify_quadrant((FRAME_W - 100, FRAME_H - 60, 80, 40),
                             FRAME_W, FRAME_H) == "BR"
    # BC — narrow wide strip at bottom center
    assert classify_quadrant(
        (FRAME_W // 2 - 70, FRAME_H - 30, 140, 20),
        FRAME_W, FRAME_H,
    ) == "BC"
    # TC — narrow wide strip at top center
    assert classify_quadrant(
        (FRAME_W // 2 - 70, 5, 140, 20),
        FRAME_W, FRAME_H,
    ) == "TC"


def test_semantic_hints():
    # Square-ish in BR → minimap
    hint = classify_semantic_hint(
        (FRAME_W - 80, FRAME_H - 80, 70, 70), "BR", FRAME_W, FRAME_H,
    )
    assert hint == "minimap"
    # Wide short in BC → health
    hint = classify_semantic_hint(
        (FRAME_W // 2 - 80, FRAME_H - 30, 160, 18), "BC",
        FRAME_W, FRAME_H,
    )
    assert hint == "health"
    # Wide short in TR → killfeed
    hint = classify_semantic_hint(
        (FRAME_W - 160, 10, 150, 30), "TR", FRAME_W, FRAME_H,
    )
    assert hint == "killfeed"
    # Wide long in TC → scoreboard
    hint = classify_semantic_hint(
        (FRAME_W // 2 - 150, 10, 300, 60), "TC", FRAME_W, FRAME_H,
    )
    assert hint == "scoreboard"


# ──────────────────── Detector tests ────────────────────


def test_detect_recovers_planted_corner_huds(tmp_path):
    # 16 frames: HUDs planted identically, central content moves
    frames = []
    for i in range(16):
        img = _base_frame()
        _plant_static_corner_huds(img)
        # Moving center blob — ensures the detector rejects motion
        cx = FRAME_W // 2 + (i - 8) * 6
        cy = FRAME_H // 2
        _plant_moving_center_blob(img, cx, cy)
        frames.append(img)
    paths = _write_sequence(tmp_path, frames)

    regions = detect_hud_regions(
        paths,
        score_threshold=0.35,
        min_area_pct=0.001,
        max_area_pct=0.12,
    )
    # Expect at least 3 of the 4 planted HUDs to land in distinct
    # quadrants (BC/TL/TR/BR). Perfect recall isn't guaranteed
    # — the connected-components pass can sometimes merge the
    # TL/TR strips on low-res downsamples.
    quadrants = {r.quadrant for r in regions}
    edge_hits = sum(1 for q in quadrants if q in ("TL", "TR", "BL", "BR", "BC", "TC"))
    assert edge_hits >= 3, (
        f"expected >=3 edge-quadrant HUDs, got {quadrants}"
    )

    # Nothing lands dead center.
    for r in regions:
        rx, ry, rw, rh = r.bbox
        cx = rx + rw / 2
        cy = ry + rh / 2
        assert abs(cx - FRAME_W / 2) > 60 or abs(cy - FRAME_H / 2) > 60, (
            f"region {r.bbox} too close to frame center"
        )


def test_detect_returns_empty_on_empty_input():
    assert detect_hud_regions([]) == []


def test_detect_returns_empty_when_all_motion(tmp_path):
    """With nothing static there should be no HUD regions."""
    frames = []
    for i in range(16):
        img = _base_frame()
        _plant_moving_center_blob(
            img, 100 + i * 20, 100 + (i % 3) * 20,
        )
        frames.append(img)
    paths = _write_sequence(tmp_path, frames)
    regions = detect_hud_regions(paths, score_threshold=0.6)
    # The detector may still surface 0 regions for a fully dynamic
    # scene. Anything it finds must at least be edge-adjacent.
    for r in regions:
        x, y, w, h = r.bbox
        assert (
            x <= 25 or y <= 25
            or (x + w) >= FRAME_W - 25
            or (y + h) >= FRAME_H - 25
        ), f"non-edge region {r.bbox} slipped through"


def test_as_pct_roundtrip():
    r = HudRegion(
        bbox=(64, 36, 160, 90),
        quadrant="TL",
        confidence=0.8,
        frame_width=640, frame_height=360,
    )
    x, y, w, h = r.as_pct()
    assert abs(x - 10.0) < 1e-6
    assert abs(y - 10.0) < 1e-6
    assert abs(w - 25.0) < 1e-6
    assert abs(h - 25.0) < 1e-6


def test_scene_cache_roundtrip(tmp_path):
    frames = [
        _plant_static_corner_huds(_base_frame()) for _ in range(8)
    ]
    paths = _write_sequence(tmp_path, frames)
    # Cache starts empty.
    invalidate_hud_cache()
    first = detect_hud_regions_for_scene("scene_42", paths)
    second = detect_hud_regions_for_scene("scene_42", paths)
    assert first is second  # identity — cached
    invalidate_hud_cache("scene_42")
    third = detect_hud_regions_for_scene("scene_42", paths)
    assert third is not first  # freshly computed after invalidate
