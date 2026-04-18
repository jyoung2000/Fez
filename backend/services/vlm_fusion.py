"""VLM ↔ face-detector confidence-weighted fusion.

Phase 3 of the VLM subject-tracking upgrade (see
``docs/vlm_upgrade/PHASE_3_NOTES.md``). This module replaces the
legacy "VLM produces, face detector overrides" hard switch with a
confidence-weighted blend so the pipeline recovers gracefully from
face-detection misses (mascots, profile shots, motion blur, anime
side characters, people facing away) without regressing the cases
where face detection is solid.

The fusion is gated behind the ``CLIPAI_VLM_FUSION`` env var:

  - ``hard`` (default) — legacy behavior. Face detection wins
    whenever any face is present; VLM subject_x is discarded.
  - ``weighted`` — apply ``fuse_vlm_and_face`` per scene with the
    rule table below.

Fusion rule (matches ``docs/vlm_upgrade/PHASE_3_NOTES.md`` exactly):

  face_conf >= 0.80 → "face_high_conf"   (trust face detector)
  0.40 <= face_conf < 0.80 → "blended"   (linear ramp on face_conf)
  face_conf < 0.40, vlm_conf >= 0.30 → "vlm_only"
  otherwise → "fallback_center" (50, confidence 0)

The module is pure: no pipeline state, no side effects, no logging
beyond the fusion-source field on the result. Callers are responsible
for attaching the result to ``SceneDescription.subject_x /
precise_x / subject_confidence / fusion_source / vlm_confidence /
face_confidence``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Env flag — default "hard" preserves legacy behavior end-to-end.
VLM_FUSION_ENV = "CLIPAI_VLM_FUSION"
VLM_FUSION_DEFAULT = "hard"


def fusion_mode() -> str:
    """Return the active fusion mode string.

    Reads from ``os.environ`` at call time so tests can flip it with
    ``monkeypatch.setenv``. Accepts any case; returns lowercase.
    """
    return os.environ.get(VLM_FUSION_ENV, VLM_FUSION_DEFAULT).strip().lower()


def fusion_enabled() -> bool:
    """True if ``CLIPAI_VLM_FUSION=weighted``; False otherwise."""
    return fusion_mode() == "weighted"


@dataclass
class FusionResult:
    """Output of ``fuse_vlm_and_face``.

    ``subject_x`` is a float in [0, 100] — callers can round to int
    for legacy consumers that require it.
    """

    subject_x: float
    subject_confidence: float
    fusion_source: str   # "face_high_conf" | "blended" | "vlm_only" | "fallback_center"


def fuse_vlm_and_face(
    face_x: float | None,
    face_conf: float,
    vlm_x: float | None,
    vlm_conf: float,
) -> FusionResult:
    """Confidence-weighted blend of VLM and face-detector subject positions.

    Parameters
    ----------
    face_x : float | None
        Horizontal subject position from face detection, 0-100.
        ``None`` is treated as ``0.0`` with ``face_conf`` forced to 0.
    face_conf : float
        Face-detector confidence, 0.0-1.0. Typically the max
        confidence across faces in the current frame.
    vlm_x : float | None
        Horizontal position from the VLM (derived from
        ``subject_box`` center in Phase 1). ``None`` is treated as
        ``50.0`` (safety center) with ``vlm_conf`` forced to 0.
    vlm_conf : float
        VLM confidence, 0.0-1.0.

    Returns
    -------
    FusionResult
        ``subject_x`` in [0, 100], ``subject_confidence`` in [0, 1],
        ``fusion_source`` tagging which branch fired.
    """
    # Normalize inputs.
    if face_x is None:
        face_x = 0.0
        face_conf = 0.0
    if vlm_x is None:
        vlm_x = 50.0
        vlm_conf = 0.0
    face_conf = max(0.0, min(1.0, float(face_conf)))
    vlm_conf = max(0.0, min(1.0, float(vlm_conf)))
    face_x = max(0.0, min(100.0, float(face_x)))
    vlm_x = max(0.0, min(100.0, float(vlm_x)))

    if face_conf >= 0.80:
        return FusionResult(
            subject_x=face_x,
            subject_confidence=face_conf,
            fusion_source="face_high_conf",
        )

    if face_conf >= 0.40:
        # Linear ramp on face confidence: w_face goes 0 → 1 as
        # face_conf goes 0.40 → 0.80. VLM contribution is scaled
        # by (1 - w_face) so at face_conf == 0.80 the blend
        # collapses to face_x (matching the boundary of the high-conf
        # branch above).
        w_face = (face_conf - 0.40) / 0.40
        w_vlm = vlm_conf * (1.0 - w_face)
        total = w_face + w_vlm
        if total < 1e-6:
            # VLM has zero confidence AND face is right at 0.40.
            # Face signal is still the best we have.
            return FusionResult(
                subject_x=face_x,
                subject_confidence=face_conf,
                fusion_source="face_high_conf",
            )
        blended_x = (face_x * w_face + vlm_x * w_vlm) / total
        return FusionResult(
            subject_x=blended_x,
            subject_confidence=max(face_conf, vlm_conf),
            fusion_source="blended",
        )

    if vlm_conf >= 0.30:
        return FusionResult(
            subject_x=vlm_x,
            subject_confidence=vlm_conf,
            fusion_source="vlm_only",
        )

    return FusionResult(
        subject_x=50.0,
        subject_confidence=0.0,
        fusion_source="fallback_center",
    )


def apply_fusion_to_scene(scene, face_x, face_conf, vlm_x, vlm_conf) -> None:
    """Fuse inputs and write the result back onto a SceneDescription.

    Writes:
      - ``subject_x`` (rounded int, for legacy consumers)
      - ``precise_x`` (float, for the L1 solver and preview)
      - ``subject_confidence``
      - ``vlm_confidence``, ``face_confidence`` (diagnostics)
      - ``fusion_source``
    """
    result = fuse_vlm_and_face(face_x, face_conf, vlm_x, vlm_conf)
    scene.subject_x = int(round(result.subject_x))
    scene.precise_x = result.subject_x
    scene.subject_confidence = result.subject_confidence
    scene.vlm_confidence = float(vlm_conf)
    scene.face_confidence = float(face_conf)
    scene.fusion_source = result.fusion_source
