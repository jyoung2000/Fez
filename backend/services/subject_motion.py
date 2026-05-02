"""Subject motion heuristics extracted from pipeline.py.

Pure helpers that operate on raw subject_x distributions. Kept in
their own module so unit tests don't need to import the whole pipeline
(which pulls in fastapi / PIL / cv2 / torch / httpx).
"""

from __future__ import annotations


def subject_xs_look_like_motion(subject_xs: list) -> tuple:
    """Replace the coarse is_continuous_motion registry heuristic with a
    histogram-based bimodality test.

    Raw subject_x values on a seated 3-speaker panel legitimately bounce
    across the frame — left speaker ~23, center ~56, right ~80. The
    old FaceRegistry.is_continuous_motion property flagged this as
    motion because the x-range was > 15. But the distribution is
    trimodal (one cluster per seat), not a smooth sweep. Treating it
    as motion disabled slot-snapping and the crops drifted.

    Args:
        subject_xs: list of 0-100 integer subject_x values from scenes.

    Returns:
        (looks_like_motion, peak_centers) — a bool and a list of
        histogram-bin centers where the peaks live. Callers can use
        peak_centers directly as snap targets when looks_like_motion
        is False and there are 2-4 peaks.
    """
    import numpy as _np

    if not subject_xs or len(subject_xs) < 10:
        return False, []

    # 10% bins across [0, 100].
    hist, edges = _np.histogram(subject_xs, bins=10, range=(0, 100))
    peak_threshold = max(3, int(0.1 * len(subject_xs)))
    peaks_mask = hist > peak_threshold
    n_peaks = int(peaks_mask.sum())

    # 2-4 peaks = seated panel; 1 peak = single subject; 5+ = actual motion.
    # For <=4 peaks, also require that frame-to-frame deltas spend >60% of
    # time above an 8-point threshold before we call it motion.
    peak_centers: list = []
    if n_peaks >= 1:
        for bi in range(len(hist)):
            if hist[bi] > peak_threshold:
                peak_centers.append(float((edges[bi] + edges[bi + 1]) / 2))

    if n_peaks <= 4:
        return False, peak_centers

    deltas = [abs(b - a) for a, b in zip(subject_xs, subject_xs[1:])]
    if not deltas:
        return False, peak_centers
    motion_frac = sum(1 for d in deltas if d > 8) / len(deltas)
    return (motion_frac > 0.6), peak_centers


def choose_snap_centers(
    registry_centers: list,
    peak_centers: list,
) -> tuple:
    """Pick snap targets between face_registry slot centers and bimodality
    peak centers.

    The bimodality peaks are derived from AI-attributed ``subject_x`` values,
    which are unreliable when VLM scene analysis fell back to synthesized
    scenes (rate-limited, error-fallback, etc.). A clean N-slot registry
    built from dense-face evidence should win over a degraded peak histogram.

    Bimodality peaks only override the registry when the registry is
    genuinely degenerate:
      * empty (no slots), OR
      * over-fragmented embedding-style: at least 2x the peak count AND
        contains at least one pair of slots within 8% of each other (a
        sign of cross-shot embedding split).

    Returns ``(slot_centers, used_peaks)``.
    """
    plausible_peaks = bool(peak_centers) and 2 <= len(peak_centers) <= 4

    if not registry_centers:
        if plausible_peaks:
            return list(peak_centers), True
        return [], False

    registry_overfragmented = False
    if plausible_peaks:
        sorted_centers = sorted(registry_centers)
        close_pairs = sum(
            1 for i in range(len(sorted_centers) - 1)
            if sorted_centers[i + 1] - sorted_centers[i] < 8.0
        )
        registry_overfragmented = (
            len(registry_centers) >= 2 * len(peak_centers)
            and close_pairs >= 1
        )

    if plausible_peaks and registry_overfragmented:
        return list(peak_centers), True
    return list(registry_centers), False
