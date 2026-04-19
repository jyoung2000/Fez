"""Per-shot content classification (Fix 3.2).

``classify_content`` in :mod:`backend.services.content_classifier` runs
once per clip. A single 60-second YouTube video can cut from a
dialogue shot → a montage → an action sequence; applying one solver
config across all three produces the wrong framing for at least two
thirds of the runtime.

This module classifies EACH shot independently from the signals local
to that shot's time window: face count, face-size variance, audio RMS,
optical-flow motion energy. The resulting ``list[ShotProfile]`` is
consumed by :mod:`backend.services.human_reframe` so the 2-D solver
and event scheduler run per-shot with the right tunables.

Note: cut-rate doesn't apply inside a single shot (we're between two
cuts by definition), so we use it only as an across-shots priors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from backend.services.content_classifier import ContentProfile

logger = logging.getLogger(__name__)


@dataclass
class ShotProfile:
    """Per-shot content classification.

    The solver and scheduler read ``content_type`` to pick the right
    ``ReframeConfig`` override. ``motion_class`` drives saccade-snap
    thresholds and headroom relaxation. ``dominant_subject_slot``, when
    non-negative, tells the A/B cut scheduler which face registry slot
    to hold on for this shot.
    """
    shot_idx: int
    start: float
    end: float
    content_type: str
    confidence: float
    motion_class: str   # "static" | "pan" | "dynamic" | "chaotic"
    face_coverage: float   # fraction of shot frames with >=1 face
    dominant_subject_slot: int = -1
    signals: dict = field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────────


def _frames_in_window(items: list, start: float, end: float) -> list:
    """Return items whose ``timestamp`` lies in [start, end). Items
    missing ``timestamp`` are skipped."""
    out = []
    for it in items or []:
        t = getattr(it, "timestamp", None)
        if t is None:
            continue
        if start <= t < end:
            out.append(it)
    return out


def _rms_in_window(audio_energy: list, start: float, end: float) -> float:
    """``audio_energy`` is a list of ``(timestamp, rms)`` pairs.
    Average the rms over the window; 0 when empty."""
    if not audio_energy:
        return 0.0
    samples = [rms for (t, rms) in audio_energy if start <= t < end]
    if not samples:
        return 0.0
    return sum(samples) / len(samples)


def _motion_class_from_faces(
    window_dense_faces: list,
    duration: float,
    audio_rms: float,
) -> str:
    """Classify shot motion from face-x variance + duration + audio."""
    if not window_dense_faces:
        return "static" if audio_rms < 0.02 else "pan"
    xs: list[float] = []
    for fr in window_dense_faces:
        if fr.faces:
            xs.append(sum(f.nose_x for f in fr.faces) / len(fr.faces))
    if len(xs) < 3:
        return "static"
    mean_x = sum(xs) / len(xs)
    variance = sum((x - mean_x) ** 2 for x in xs) / len(xs)
    stdev = variance ** 0.5
    # stdev is in percent (0-100); thresholds mirror content_classifier.
    if stdev < 4.0:
        return "static"
    if stdev < 10.0:
        return "pan"
    if stdev < 20.0:
        return "dynamic"
    return "chaotic"


def _dominant_slot_in_window(window_dense_faces: list) -> int:
    """Return the identity_id that appears in the most frames of the
    window, or -1 when no identified face dominates."""
    counts: dict[int, int] = {}
    for fr in window_dense_faces:
        seen = set()
        for f in fr.faces:
            sid = getattr(f, "identity_id", -1)
            if sid >= 0 and sid not in seen:
                counts[sid] = counts.get(sid, 0) + 1
                seen.add(sid)
    if not counts:
        return -1
    return max(counts, key=counts.get)


def _content_type_for_shot(
    *,
    face_coverage: float,
    avg_faces: float,
    motion_class: str,
    shot_duration: float,
    clip_profile: Optional[ContentProfile],
    has_strong_color_motion: bool = False,
) -> tuple[str, float]:
    """Pick the content_type and a confidence for one shot.

    The clip-level ``clip_profile`` biases the outcome by +0.8 on its
    type (weaker than the per-clip +2.0 hint in classify_content, since
    clip-level guesses shouldn't dominate real shot signals).
    """
    scores: dict[str, float] = {
        "talking_head": 0.0,
        "narrative": 0.0,
        "vlog": 0.0,
        "podcast": 0.0,
        "sports": 0.0,
        "music_video": 0.0,
        "anime": 0.0,
        "gaming": 0.0,
    }

    # Face-coverage signals. The talking-head bonus only fires when
    # motion is ALSO static/pan — a high-motion shot with a face
    # still in it (action scene, moving subject) is not a talking
    # head just because the face is visible.
    face_dependent_stationary = (
        face_coverage >= 0.75 and motion_class in ("static", "pan")
    )
    if face_dependent_stationary:
        scores["talking_head"] += 2.0
        if avg_faces >= 2.0:
            scores["podcast"] += 2.5
        else:
            scores["vlog"] += 1.0
    elif face_coverage >= 0.30 and motion_class in ("static", "pan"):
        scores["narrative"] += 1.5
        scores["vlog"] += 0.5
    elif face_coverage >= 0.10:
        # Sparse faces → likely montage / action.
        scores["sports"] += 1.5
        scores["music_video"] += 1.0
    elif face_coverage < 0.10:
        # Essentially no faces → montage / b-roll / action.
        scores["sports"] += 2.5
        scores["music_video"] += 2.0

    # Motion-class signals. Face-dependent talking-head / podcast
    # bonuses only fire when the shot actually has faces AND is
    # visually calm.
    if motion_class == "static" and face_coverage >= 0.30:
        scores["talking_head"] += 1.5
        scores["podcast"] += 1.0
    elif motion_class == "pan":
        scores["narrative"] += 1.0
        if face_coverage >= 0.30:
            scores["vlog"] += 1.0
    elif motion_class == "dynamic":
        scores["sports"] += 2.0
        scores["music_video"] += 1.5
    elif motion_class == "chaotic":
        # Chaotic motion dominates: force sports/music_video to win
        # even on high face-coverage shots.
        scores["sports"] += 3.0
        scores["music_video"] += 2.5

    # Short shots (<= 1.5s) are montage-ish. Stretched from 1.0 so
    # 1.0s montage shots register.
    if shot_duration <= 1.5:
        scores["music_video"] += 1.0

    # Clip-level prior as a soft bias.
    if clip_profile is not None:
        ct = clip_profile.content_type
        if ct in scores:
            scores[ct] += 0.8
        # A clip flagged animated biases shots toward anime.
        if getattr(clip_profile, "is_animated", False):
            scores["anime"] += 1.0

    winner = max(scores, key=scores.get)
    total = sum(scores.values()) or 1.0
    return winner, min(1.0, scores[winner] / total)


# ── Public API ────────────────────────────────────────────────────


def classify_shots(
    shot_boundaries: list[float],
    dense_faces: list,
    *,
    saliency_regions: Optional[list] = None,
    audio_energy: Optional[list] = None,
    object_detections: Optional[list] = None,
    duration_sec: float,
    clip_profile: Optional[ContentProfile] = None,
    job_id: str = "",
) -> list[ShotProfile]:
    """Per-shot content classification.

    Args:
        shot_boundaries: timestamps of shot CUTS (not shot starts).
            N cuts → N+1 shots. ``[]`` means a single shot covering
            the full duration.
        dense_faces: per-frame face detections (``list[FrameFaces]``).
        saliency_regions: optional, currently unused — reserved for
            future refinement (per-shot saliency clustering).
        audio_energy: optional ``list[(t, rms)]`` pairs.
        object_detections: optional, currently unused.
        duration_sec: clip duration. The last shot ends at this.
        clip_profile: optional clip-level ``ContentProfile`` used as a
            soft prior (+0.8) on the matching content_type.
        job_id: for logging.

    Returns:
        One ``ShotProfile`` per shot; ``len(result) == len(boundaries) + 1``.
    """
    if duration_sec <= 0:
        return []

    # Build list of (start, end) pairs. 0 boundaries → 1 shot.
    starts = [0.0] + sorted(shot_boundaries or [])
    ends = sorted(shot_boundaries or []) + [float(duration_sec)]
    shots = list(zip(starts, ends))

    profiles: list[ShotProfile] = []
    for idx, (s, e) in enumerate(shots):
        # Collapse degenerate zero-length shots (common when a boundary
        # coincides with the clip end).
        if e - s < 0.05:
            continue

        window_faces = _frames_in_window(dense_faces, s, e)
        total_window_frames = len(window_faces)
        face_coverage = (
            sum(1 for fr in window_faces if fr.faces) / total_window_frames
            if total_window_frames > 0 else 0.0
        )
        counts = [len(fr.faces) for fr in window_faces if fr.faces]
        avg_faces = (sum(counts) / len(counts)) if counts else 0.0
        audio_rms = _rms_in_window(audio_energy or [], s, e)
        motion_class = _motion_class_from_faces(
            window_faces, e - s, audio_rms,
        )
        dominant_slot = _dominant_slot_in_window(window_faces)

        content_type, conf = _content_type_for_shot(
            face_coverage=face_coverage,
            avg_faces=avg_faces,
            motion_class=motion_class,
            shot_duration=e - s,
            clip_profile=clip_profile,
        )

        signals = {
            "face_coverage": round(face_coverage, 2),
            "avg_faces": round(avg_faces, 2),
            "audio_rms": round(audio_rms, 3),
            "motion_class": motion_class,
            "n_frames_in_window": total_window_frames,
        }

        profiles.append(ShotProfile(
            shot_idx=idx,
            start=s,
            end=e,
            content_type=content_type,
            confidence=conf,
            motion_class=motion_class,
            face_coverage=face_coverage,
            dominant_subject_slot=dominant_slot,
            signals=signals,
        ))

    if job_id:
        for p in profiles:
            logger.info(
                "[%s] ShotClassifier: shot#%d [%.2f, %.2f] → %s (motion=%s, "
                "coverage=%.2f, conf=%.2f, slot=%d)",
                job_id, p.shot_idx, p.start, p.end, p.content_type,
                p.motion_class, p.face_coverage, p.confidence,
                p.dominant_subject_slot,
            )

    return profiles
