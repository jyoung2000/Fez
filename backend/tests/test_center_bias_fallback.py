"""Phase 1 center-bias fallback tests.

Three scenarios the spec calls out as must-pass:

1. **Crosshair removed entirely.** A synthetic FPS sequence where
   no frame contains a detectable crosshair → every emitted
   ``CrosshairFrame`` is ``(t, 50.0, 50.0, 0.0)`` and every
   downstream per-segment ``subject_x`` lands at 50 ± 1 %.

2. **Half-and-half confidence.** Crosshair visible on the first
   50 % of frames, missing on the second 50 % → the first half
   tracks the crosshair, the second half is exactly 50 (no stale
   offset leaking forward from the last confident detection).

3. **Bright cartoon character + faint center crosshair.** A
   cartoon model at x=20 with a weak (conf < 0.6) center
   crosshair → the segmenter/solver resolver must return 50, not
   20. This is the TF2 / Marvel Rivals regression that the hard
   center fallback exists to prevent.

These scenarios exercise the crosshair tracker's always-emit
fallback, the pipeline's confidence-gated scene override, and
the L1 solver's :func:`build_gaming_center_biased_targets`
resolver. The segmenter's ``CENTER_BIAS_GENRES`` guard is also
covered indirectly — any low-confidence frame forces target_x
to 50 via the crosshair resolver.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

pytest.importorskip("numpy")
pytest.importorskip("cv2")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from backend.services.crosshair_tracker import (  # noqa: E402
    track_crosshair_path,
)
from backend.services.l1_camera_path import (  # noqa: E402
    CENTER_BIAS_GENRES,
    GAMING_CENTER_WEIGHT_DEFAULT,
    GAMING_CROSSHAIR_WEIGHT,
    build_gaming_center_biased_targets,
)


# ────────────────────── synthetic frame helpers ──────────────────────


def _blank_frame(width=1280, height=720, bg=128) -> np.ndarray:
    return np.full((height, width), bg, dtype=np.uint8)


def _with_crosshair(img: np.ndarray, x_pct: float, y_pct: float,
                    size: int = 12, thickness: int = 2) -> np.ndarray:
    h, w = img.shape[:2]
    cx = int(w * x_pct / 100.0)
    cy = int(h * y_pct / 100.0)
    half = size // 2
    cv2.line(img, (cx - half, cy), (cx + half, cy), (0, 0, 0),
             thickness=thickness)
    cv2.line(img, (cx, cy - half), (cx, cy + half), (0, 0, 0),
             thickness=thickness)
    return img


def _with_cartoon_character(img: np.ndarray, x_pct: float) -> np.ndarray:
    """Paint a bright high-contrast blob on the left third — the
    kind of feature that would win saliency / motion / face-cluster
    votes on a TF2-style cartoon FPS frame."""
    h, w = img.shape[:2]
    cx = int(w * x_pct / 100.0)
    cy = int(h * 0.55)
    # Solid ellipse — high-contrast, off-center blob
    cv2.ellipse(
        img, (cx, cy), (60, 90), 0, 0, 360,
        color=(20,), thickness=-1,
    )
    return img


def _write_sequence(tmp_path, frames: list, prefix: str = "f") -> list:
    paths = []
    for i, img in enumerate(frames):
        fp = tmp_path / f"{prefix}{i:03d}.png"
        cv2.imwrite(str(fp), img)
        paths.append((i / 30.0, str(fp)))
    return paths


# ────────────────────── tests ──────────────────────


def test_center_bias_genres_spec_membership():
    """Sanity — the spec lists five subtypes for center bias."""
    expected = {"fps", "gameplay_fps", "hero_shooter", "sandbox",
                "gameplay"}
    assert expected.issubset(CENTER_BIAS_GENRES)
    # Non-center-bias genres stay out.
    for g in ("moba", "tps", "racing", "rts"):
        assert g not in CENTER_BIAS_GENRES


def test_fps_clip_with_no_crosshair_emits_all_center_fallbacks(tmp_path):
    """Scenario 1: completely blank frames, no crosshair anywhere.
    Every emitted entry is the hard-center breadcrumb."""
    frames = [_blank_frame() for _ in range(30)]
    paths = _write_sequence(tmp_path, frames, "blank")

    tracked = track_crosshair_path(paths)
    assert len(tracked) == len(frames), "always-emit broken"
    for cf in tracked:
        assert cf.x_pct == 50.0, f"fallback x={cf.x_pct}"
        assert cf.y_pct == 50.0, f"fallback y={cf.y_pct}"
        assert cf.confidence == 0.0

    # Now resolve per-segment subject_x via the solver helper.
    positions, weights = build_gaming_center_biased_targets(
        crosshair_path=tracked,
        events=[],
        start=0.0,
        end=1.0,
    )
    assert positions
    for _t, x in positions:
        assert x == 50.0, f"solver target drifted to {x}"
    for w in weights:
        assert w == GAMING_CENTER_WEIGHT_DEFAULT


def test_half_confident_half_missing_no_stale_offset(tmp_path):
    """Scenario 2: crosshair visible on frames 0-29, missing on
    frames 30-59. The first half tracks the crosshair; the second
    half snaps to exactly 50 — no stale offset is inherited from
    the last confident detection.

    Crosshair placed off-center (x=35) on the confident half so
    the test can distinguish "anchored to last position" (wrong)
    from "hard fallback to 50" (right).
    """
    confident_x = 35.0
    frames = [
        _with_crosshair(_blank_frame(), confident_x, 50.0)
        for _ in range(30)
    ] + [
        _blank_frame() for _ in range(30)
    ]
    paths = _write_sequence(tmp_path, frames, "half")

    tracked = track_crosshair_path(paths)
    assert len(tracked) == 60

    # First half: at least one confident detection near x=35.
    first = [cf for cf in tracked[:30] if cf.confidence >= 0.4]
    assert first, "confident half produced no real detections"
    for cf in first:
        assert abs(cf.x_pct - confident_x) < 3.0, (
            f"confident-half x drifted: {cf.x_pct}"
        )

    # Second half: every entry is the hard-center breadcrumb.
    # The tracker re-seeds its search prior back to (50, 50)
    # after FALLBACK_RESEED_AFTER_DROPS consecutive drops, so any
    # late detections should have recovered center as the prior.
    second_half = tracked[30:]
    fallback_count = sum(
        1 for cf in second_half
        if cf.x_pct == 50.0 and cf.y_pct == 50.0 and cf.confidence == 0.0
    )
    # All 30 frames should be fallbacks (blank frame = no detection).
    assert fallback_count == len(second_half), (
        f"second half had {len(second_half) - fallback_count} "
        f"non-fallback entries — stale offset leak"
    )

    # Resolver: second-half targets are all 50.
    positions, weights = build_gaming_center_biased_targets(
        crosshair_path=tracked,
        events=[],
        start=1.0,  # second half window
        end=2.0,
    )
    assert positions
    for _t, x in positions:
        assert x == 50.0, (
            f"resolver target in second half = {x}, expected 50 "
            "(hard center fallback)"
        )
    for w in weights:
        assert w == GAMING_CENTER_WEIGHT_DEFAULT


def test_cartoon_character_does_not_override_hard_center(tmp_path):
    """Scenario 3 — the TF2 regression fixture.

    A bright cartoon character at x=20 plus a weak center crosshair.
    The tracker either picks up the center crosshair (confident,
    small x near 50) or falls back to center (x=50). In both cases
    the resolver must return x=50 at the segment level — NEVER the
    character's x=20. This proves the center-bias resolver
    suppresses face-cluster / saliency / motion anchors for FPS
    genres.
    """
    frames = []
    for _ in range(30):
        img = _blank_frame()
        img = _with_cartoon_character(img, 20.0)
        # Very faint center crosshair (thin lines, low contrast)
        img = _with_crosshair(img, 50.0, 50.0, size=10, thickness=1)
        frames.append(img)
    paths = _write_sequence(tmp_path, frames, "tf2")

    tracked = track_crosshair_path(paths)
    assert len(tracked) == 30

    # Whatever the tracker does, downstream the resolver must never
    # return x=20. For each frame, simulate what the pipeline would
    # pass to the solver — a fake "confident" detection at the
    # character's position to exercise the center-bias guard.
    from dataclasses import dataclass as _dc

    @_dc
    class _HallucinatedTrack:
        timestamp: float
        x_pct: float
        y_pct: float
        confidence: float

    # Low-confidence detection at the character position
    # (simulating saliency / face-cluster votes that a broken
    # pipeline might produce).
    hallucinated = [
        _HallucinatedTrack(i / 30.0, 20.0, 55.0, 0.5)  # conf < 0.6
        for i in range(30)
    ]
    positions, weights = build_gaming_center_biased_targets(
        crosshair_path=hallucinated,
        events=[],
        start=0.0,
        end=1.0,
    )
    assert positions, "solver produced no targets"
    for _t, x in positions:
        assert x == 50.0, (
            f"low-confidence cartoon-character position leaked "
            f"into target stream: x={x}"
        )
    for w in weights:
        assert w == GAMING_CENTER_WEIGHT_DEFAULT

    # And confirm: if the SAME x=20 detection had confidence ≥ 0.6,
    # THEN the resolver would track it — this is the intentional
    # escape hatch for legit off-center crosshairs.
    legit = [
        _HallucinatedTrack(i / 30.0, 20.0, 55.0, 0.9)
        for i in range(30)
    ]
    positions2, weights2 = build_gaming_center_biased_targets(
        crosshair_path=legit,
        events=[],
        start=0.0,
        end=1.0,
    )
    for _t, x in positions2:
        assert abs(x - 20.0) < 0.5
    for w in weights2:
        assert w == GAMING_CROSSHAIR_WEIGHT
