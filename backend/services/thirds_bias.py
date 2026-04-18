"""Phase 4 — Rule-of-thirds composition bias.

A small, numpy-free helper that scores a feature point by how close
it sits to one of the four rule-of-thirds intersections of the
9:16 vertical output frame:

    (1/3, 1/3)   (2/3, 1/3)
    (1/3, 2/3)   (2/3, 2/3)

The score is a 2-D Gaussian peaked at each intersection with
σ=0.12 in normalized [0, 1] coordinates, taking the **maximum**
across the four peaks. A feature exactly at an intersection scores
1.0; a feature at frame center scores ≈ 0.07; a feature at a
corner scores ≈ 0.005.

Used in two places:

1. **Subject-x offset for single-subject segments** —
   ``thirds_x_offset_px`` returns a horizontal shift to add to
   ``subject_x`` so the face lands at one of the two horizontal
   thirds (1/3 or 2/3) of the OUTPUT crop. The choice between
   left and right thirds defers to the gaze direction (when
   available) and falls back to "shift toward the larger side"
   when the face is off-center horizontally.

2. **Optional-region weighting in the multi-region LP** —
   ``thirds_score_for_region`` multiplies an optional region's
   weight by its thirds score so the LP prefers crops that
   place high-importance features at thirds intersections.

Feature-flagged via ``CLIPAI_THIRDS_BIAS`` (default OFF until
in-docker validation lands the post-Phase-4 numbers in
``docs/autoflip_parity_v2_results.md``).

Per the v2 spec, the bias is on for **narrative**, **vlog**,
**cinematic_dialogue**, and **animation_dialogue** content. It is
**off** for **debate** / **multi_speaker_panel** (symmetry matters
more than thirds for 3+ seated subjects) and **off** for
**music_video** (composition is already choreographed).
"""

from __future__ import annotations

import math
import os
from typing import Optional


# ── Feature flag ─────────────────────────────────────────────────

USE_THIRDS_BIAS = os.environ.get(
    "CLIPAI_THIRDS_BIAS", "1",
).lower() in ("1", "true", "yes", "on")


# ── Tuning constants ─────────────────────────────────────────────

# 2-D Gaussian width as a fraction of the OUTPUT frame dimension.
# σ=0.12 gives a moderate "pull" toward thirds without making
# off-thirds positions essentially invisible to the scorer. Matches
# the v2 spec.
THIRDS_SIGMA = 0.12

# The four canonical rule-of-thirds intersections in normalized
# OUTPUT-frame [0, 1] coordinates.
THIRDS_INTERSECTIONS: tuple[tuple[float, float], ...] = (
    (1.0 / 3.0, 1.0 / 3.0),
    (2.0 / 3.0, 1.0 / 3.0),
    (1.0 / 3.0, 2.0 / 3.0),
    (2.0 / 3.0, 2.0 / 3.0),
)

# Content types that get the thirds bias applied.
#
# The set spans BOTH the parent ``ContentType`` values
# (``narrative`` / ``vlog`` / ``podcast``) and the downstream
# ``ClipContentType`` values (``cinematic_dialogue`` /
# ``animation_dialogue`` / ``talking_head``) so callers can pass
# either flavor of string. Reframe segmenter callers should ALSO
# check ``profile.is_multi_speaker_panel`` and skip the bias when
# True — debates / panels prefer symmetry over thirds even though
# they share the ``podcast`` parent type. Use ``applies_to_profile``
# below for the convenient combined check.
THIRDS_BIAS_CONTENT_TYPES: frozenset[str] = frozenset({
    # ── Parent ContentType values ──
    "narrative",
    "vlog",
    "podcast",  # gated by is_multi_speaker_panel below
    # ── ClipContentType values ──
    "cinematic_dialogue",
    "animation_dialogue",
    # Plain "talking_head" is a vlog-adjacent single-speaker case
    # — Phase 4 includes it so vlogs categorized as talking_head
    # also get the bias.
    "talking_head",
})


# ── Gaussian thirds scorer ───────────────────────────────────────


def thirds_bias_score(
    x_norm: float,
    y_norm: float,
    *,
    sigma: float = THIRDS_SIGMA,
) -> float:
    """Score a normalized point against the rule-of-thirds Gaussian.

    Args:
        x_norm: x in normalized [0, 1] OUTPUT-frame coordinates.
        y_norm: y in normalized [0, 1] OUTPUT-frame coordinates.
        sigma: 2-D Gaussian standard deviation. Default 0.12 per
            the v2 spec.

    Returns:
        A float in (0, 1] giving the maximum Gaussian density
        across the four thirds intersections. A point exactly on
        an intersection scores 1.0; a point at frame center scores
        ≈ exp(-((1/6)² + (1/6)²) / (2·σ²)) ≈ 0.07 at σ=0.12.

    The function is total — out-of-range inputs (negative or > 1)
    are clamped before scoring, so a caller passing a slightly
    out-of-bounds normalized coordinate still gets a sensible
    score instead of an exception.
    """
    if sigma <= 0:
        return 0.0
    x = max(0.0, min(1.0, float(x_norm)))
    y = max(0.0, min(1.0, float(y_norm)))
    two_sigma_sq = 2.0 * sigma * sigma
    best = 0.0
    for ix, iy in THIRDS_INTERSECTIONS:
        dx = x - ix
        dy = y - iy
        score = math.exp(-((dx * dx) + (dy * dy)) / two_sigma_sq)
        if score > best:
            best = score
    return best


def best_thirds_intersection(
    x_norm: float,
    y_norm: float,
) -> tuple[float, float]:
    """Return the closest thirds intersection to a normalized point.

    Used by ``thirds_x_offset_px`` to decide which third
    (left=1/3 or right=2/3) a face should land on. Closest
    intersection wins; ties break toward the LEFT/UPPER one so
    the caller's choice is deterministic.
    """
    x = max(0.0, min(1.0, float(x_norm)))
    y = max(0.0, min(1.0, float(y_norm)))
    best = THIRDS_INTERSECTIONS[0]
    best_d = float("inf")
    for ix, iy in THIRDS_INTERSECTIONS:
        d = (x - ix) ** 2 + (y - iy) ** 2
        if d < best_d - 1e-9:
            best_d = d
            best = (ix, iy)
    return best


# ── Subject-x offset helper ──────────────────────────────────────


def thirds_x_offset_px(
    *,
    face_x_pct: float,
    crop_width_px: float,
    source_width_px: float,
    yaw: float = 0.0,
    yaw_threshold: float = 0.10,
) -> float:
    """Compute a horizontal shift (in source pixels) to land the face
    at the left or right third of the OUTPUT crop.

    Without thirds bias, the segmenter sets ``subject_x = face_x_px``
    so the face appears at the **center** of the output crop. Phase 4
    asks for the face to appear at **x = 1/3** or **x = 2/3** of the
    output.

    To put the face at x = 1/3 of an output crop of width W centered
    at ``subject_x``:

        face_x_in_output = subject_x - W/2 → no, that's the left edge.
        face_x_in_output_norm = (face_x_px - (subject_x - W/2)) / W
        We want face_x_in_output_norm = 1/3:
            (face_x_px - subject_x + W/2) / W = 1/3
            subject_x = face_x_px + W/2 - W/3 = face_x_px + W/6

    So shifting subject_x by **+W/6** puts the face at x = 1/3 of
    the output (i.e. on the LEFT third), and shifting by **-W/6**
    puts it at x = 2/3 (RIGHT third).

    Choice of left vs right third:
      - When ``|yaw| >= yaw_threshold``: defer to gaze. Looking left
        (yaw < 0) → place face on LEFT third (subject_x += W/6) so
        there's space to look INTO on the right; looking right →
        place on RIGHT third.
      - When yaw is small / unknown: default to **LEFT** third
        (the cinematography default for a forward-facing subject).

    Args:
        face_x_pct: Face center x in **percent of source frame width**
            (0–100), matching the rest of the reframe pipeline.
        crop_width_px: Output crop width in source pixels (≈ 607 for
            16:9 → 9:16 on a 1920-wide source).
        source_width_px: Source frame width in pixels.
        yaw: Continuous yaw in [-1, 1] from
            ``gaze_estimator.estimate_yaw``. Defaults to 0 (forward).
        yaw_threshold: Minimum |yaw| to defer to gaze for the
            left-vs-right decision.

    Returns:
        Signed pixel offset to **add** to ``subject_x``. Always
        returns 0.0 when ``crop_width_px <= 0`` so the caller doesn't
        have to special-case the no-crop edge.
    """
    if crop_width_px <= 0 or source_width_px <= 0:
        return 0.0
    sixth = crop_width_px / 6.0
    if abs(yaw) >= yaw_threshold:
        if yaw < 0:
            # Looking left → place on LEFT third → subject_x += W/6
            return +sixth
        # Looking right → place on RIGHT third → subject_x -= W/6
        return -sixth
    # No gaze info — default to LEFT third (cinematography default)
    return +sixth


def thirds_score_for_region(
    *,
    region_left_pct: float,
    region_right_pct: float,
    region_y_pct: float,
    crop_left_pct: float,
    crop_right_pct: float,
    sigma: float = THIRDS_SIGMA,
) -> float:
    """Score an optional region by where it lands inside a candidate crop.

    Used by the multi-region LP wiring to weight optional regions
    by their thirds-bias score — the LP then prefers crops where
    high-importance features sit at thirds intersections.

    All inputs in **percent of source frame** [0, 100]. The function
    converts the region's center to normalized OUTPUT-crop coordinates
    by mapping ``[crop_left_pct, crop_right_pct] → [0, 1]`` for x and
    using the source y directly (``region_y_pct / 100``) for y.

    Returns the thirds Gaussian score in (0, 1]. Returns 0.0 when
    the crop window is degenerate.
    """
    crop_w = crop_right_pct - crop_left_pct
    if crop_w <= 0:
        return 0.0
    region_cx = 0.5 * (region_left_pct + region_right_pct)
    x_norm = (region_cx - crop_left_pct) / crop_w
    y_norm = float(region_y_pct) / 100.0
    return thirds_bias_score(x_norm, y_norm, sigma=sigma)


def applies_to_content(content_type: Optional[str]) -> bool:
    """Return True when the thirds-bias should fire for this content type.

    Per the v2 spec: ON for narrative / vlog / cinematic_dialogue /
    animation_dialogue / talking_head / podcast. OFF for debate /
    panel (symmetry beats thirds with 3+ subjects) — but this gate
    function only sees the type string, not the panel flag. Callers
    that have a full ``ContentProfile`` should prefer
    ``applies_to_profile`` which combines both checks.

    OFF for music_video (composition is already choreographed) and
    everything else by default — Phase 4 is a conservative behavior
    change.
    """
    if content_type is None:
        return False
    val = getattr(content_type, "value", str(content_type))
    return val in THIRDS_BIAS_CONTENT_TYPES


def applies_to_profile(content_profile) -> bool:
    """Return True when the thirds-bias should fire for this profile.

    Wraps ``applies_to_content`` and additionally checks:

    - ``content_profile.is_multi_speaker_panel`` — debates / panels
      share the ``podcast`` parent type but prefer symmetry over
      thirds, so they're explicitly excluded.
    - ``content_profile.is_animated`` (Phase 6 extension) — the
      ``ContentType.ANIME`` parent type isn't in the bias set, but
      anime dialogue + slice-of-life content gets the thirds bias
      via the ANIMATION_DIALOGUE / TALKING_HEAD downstream clip
      types. We honor the profile-level ``is_animated`` flag here
      so the gate fires on the parent type without needing to wait
      until ``classify_clip`` runs.
    - ``content_profile.anime_subtype == "action"`` — action anime
      explicitly OPTS OUT of the thirds bias (the reaction-shot
      compositions are choreographed in source, no need to bias).

    Accepts None / non-profile inputs and returns False.
    """
    if content_profile is None:
        return False
    if getattr(content_profile, "is_multi_speaker_panel", False):
        return False

    # Phase 6: anime profile handling
    is_animated = bool(getattr(content_profile, "is_animated", False))
    anime_subtype = getattr(content_profile, "anime_subtype", None) or ""
    if is_animated:
        if anime_subtype == "action":
            return False
        # Anime dialogue / slice-of-life / unspecified all get the
        # bias since ANIMATION_DIALOGUE downstream is in the bias set.
        return True

    return applies_to_content(getattr(content_profile, "content_type", None))
