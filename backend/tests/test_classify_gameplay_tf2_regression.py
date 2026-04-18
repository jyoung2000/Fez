"""TF2 cartoon-contamination regression for classify_gameplay_content.

Reproduces the exact input signature the TF2 "Washed Up Tuber"
clip produced in production:

  * Sparse FaceMesh pass: 0/30 frames with faces
  * Dense pass (after YuNet augmentation): 213/301 frames with
    "faces" but 140 of the 291 detected faces are flagged
    non-human by :class:`HumanFaceVerifier`

Before this regression fix, the classifier's first check
(``face_ratio > 0.30``) fired on the raw dense ratio and
returned ``"not_gameplay"``. After the fix:

  * Filename hint on ``TF2： Washed Up Tuber.mp4`` matches
    immediately → ``"gameplay"``
  * Even without the filename hint, the
    cartoon-contamination signal
    (raw≥0.30 AND verified<0.15) promotes to ``"gameplay"``
  * Even without dense data, the sparse face_ratio<5% + HUD
    signal path fires

Each branch is exercised independently so one passing test
doesn't mask a regression in the others.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

pytest.importorskip("numpy")

from backend.services.face_detector import classify_gameplay_content  # noqa: E402


# ──────────────────── Stub face data ────────────────────


@dataclass
class _StubFace:
    x: float = 50.0
    y: float = 50.0
    is_human_verified: bool = True


@dataclass
class _StubFrameFaces:
    timestamp: float = 0.0
    faces: list = None  # type: ignore[assignment]


def _make_dense(
    n_total: int, n_with_faces: int, n_verified: int,
) -> list[_StubFrameFaces]:
    """Build dense face data that mimics the TF2 production signature."""
    out = []
    # Verified human faces
    for i in range(n_verified):
        out.append(_StubFrameFaces(
            timestamp=i * 0.5,
            faces=[_StubFace(is_human_verified=True)],
        ))
    # Cartoon / rejected faces
    for i in range(n_with_faces - n_verified):
        out.append(_StubFrameFaces(
            timestamp=(n_verified + i) * 0.5,
            faces=[_StubFace(is_human_verified=False)],
        ))
    # Faceless frames
    for i in range(n_total - n_with_faces):
        out.append(_StubFrameFaces(
            timestamp=(n_with_faces + i) * 0.5,
            faces=[],
        ))
    return out


def _make_sparse(
    n_total: int, n_with_faces: int,
) -> list[_StubFrameFaces]:
    """Sparse face stream. Sparse uses FaceMesh only so cartoon
    characters almost never register — the TF2 clip had 0/30."""
    out = []
    for i in range(n_with_faces):
        out.append(_StubFrameFaces(
            timestamp=i * 1.0,
            faces=[_StubFace(is_human_verified=True)],
        ))
    for i in range(n_total - n_with_faces):
        out.append(_StubFrameFaces(
            timestamp=(n_with_faces + i) * 1.0,
            faces=[],
        ))
    return out


# ──────────────────── Tests ────────────────────


def test_tf2_filename_hint_short_circuits_to_gameplay():
    """Filename hint fires before any face-based signal —
    returns ``gameplay`` even when the dense pass claims 213/301
    'faces'."""
    dense = _make_dense(n_total=301, n_with_faces=213, n_verified=8)
    sparse = _make_sparse(n_total=30, n_with_faces=0)

    result = classify_gameplay_content(
        dense, len(dense),
        sample_frame_paths=[],
        sparse_face_data=sparse,
        filename="TF2： Washed Up Tuber.mp4",
    )
    assert result == "gameplay"


def test_cartoon_contamination_promotes_without_filename():
    """Without a filename hint, the raw≥0.30 + verified<0.15
    divergence still promotes the clip to gameplay."""
    # 213/301 raw = 0.71, verified 8/301 = 0.027
    dense = _make_dense(n_total=301, n_with_faces=213, n_verified=8)

    result = classify_gameplay_content(
        dense, len(dense),
        sample_frame_paths=[],
        # Sparse data NOT passed — pure dense path.
    )
    assert result == "gameplay"


def test_sparse_face_rarity_needs_hud_signal_to_fire(tmp_path):
    """Without the filename hint or dense contamination, the
    sparse-rarity path requires an HUD / crosshair signal from
    sample frames to promote to ``gameplay``. Without that
    signal the classifier returns ``unknown`` — the legacy
    "skip the gameplay fast-path but don't mislabel as
    talking-head" middle-ground."""
    import cv2
    import numpy as np

    # Build a sample frame with zero edge / flat gray content
    img = np.full((360, 640), 128, dtype=np.uint8)
    fp = tmp_path / "flat.png"
    cv2.imwrite(str(fp), img)

    sparse = _make_sparse(n_total=30, n_with_faces=0)
    result = classify_gameplay_content(
        [], 0,
        sample_frame_paths=[str(fp)] * 4,
        sparse_face_data=sparse,
    )
    assert result == "unknown"


def test_sparse_face_rarity_plus_hud_signal_returns_gameplay(tmp_path):
    """Sparse face_ratio<5% AND a bright corner HUD → ``gameplay``."""
    import cv2
    import numpy as np

    # Frame with bright high-saturation corners (HUD-like).
    img = np.full((360, 640, 3), 60, dtype=np.uint8)
    # Colorful top-right corner (killfeed-style)
    img[:36, -64:] = (0, 255, 255)
    # Colorful bottom-center strip (health-style)
    img[-30:, 270:370] = (255, 200, 0)
    # Bottom-right minimap-style square
    img[-70:, -70:] = (200, 255, 50)
    fp = tmp_path / "huddy.png"
    cv2.imwrite(str(fp), img)

    sparse = _make_sparse(n_total=30, n_with_faces=0)
    result = classify_gameplay_content(
        [], 0,
        sample_frame_paths=[str(fp)] * 8,
        sparse_face_data=sparse,
    )
    assert result == "gameplay"


def test_healthy_talking_head_stays_not_gameplay():
    """A normal talking-head clip (70 % face rate, all verified
    human) must still return ``not_gameplay``."""
    dense = _make_dense(n_total=100, n_with_faces=70, n_verified=70)
    sparse = _make_sparse(n_total=30, n_with_faces=22)

    result = classify_gameplay_content(
        dense, len(dense),
        sample_frame_paths=[],
        sparse_face_data=sparse,
        filename="rogan_podcast_ep_420.mp4",
    )
    assert result == "not_gameplay"


def test_healthy_talking_head_ignores_cartoon_path():
    """70 % raw AND 68 % verified — verified_ratio is above the
    0.15 floor so the cartoon-promotion path does NOT fire."""
    dense = _make_dense(n_total=100, n_with_faces=70, n_verified=68)
    result = classify_gameplay_content(
        dense, len(dense),
        sample_frame_paths=[],
    )
    assert result == "not_gameplay"


def test_empty_inputs_return_not_gameplay():
    result = classify_gameplay_content(
        [], 0, sample_frame_paths=[],
        sparse_face_data=[],
    )
    # Nothing to go on → legacy "not_gameplay" fallback.
    assert result in ("not_gameplay", "unknown")


def test_filename_hint_overrides_talking_head_ratio():
    """Edge case: someone uploads a Valorant clip with a
    facecam overlay (high face ratio). The filename hint still
    short-circuits to ``gameplay`` so the gameplay path runs."""
    dense = _make_dense(n_total=100, n_with_faces=65, n_verified=60)
    result = classify_gameplay_content(
        dense, len(dense),
        sample_frame_paths=[],
        filename="Valorant pentakill ACE.mp4",
    )
    assert result == "gameplay"
