"""Heuristic key-scene generator.

When the VLM scene-analysis path returns 0 real descriptions (vision
provider failure, OOM, wrong model id, exhausted API key, etc.) the
Analysis page would otherwise show an empty Key Scenes tab. This
module synthesizes a useful set of "key scenes" from data the pipeline
already has — scene cuts, transcript timestamps, face tracks, and
audio energy — so the Key Scenes tab is never blank when the pipeline
has run to completion.

The generated scenes are clearly labelled as "Key moment …" so users
can tell them apart from real VLM-generated descriptions, and they
inherit the same ``SceneDescription`` shape so every downstream
consumer (clip detector, render plan, share view) works without
modification.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from backend.models import SceneDescription

logger = logging.getLogger(__name__)


def _round_ts(t: float) -> float:
    return round(float(t), 2)


def _excerpt_transcript(
    transcript: list,
    t_start: float,
    t_end: float,
    max_chars: int = 140,
) -> str:
    """Pull a short transcript excerpt covering [t_start, t_end].

    Returns "" when the window has no transcript overlap.
    """
    if not transcript:
        return ""
    pieces: list[str] = []
    total = 0
    for seg in transcript:
        seg_start = float(getattr(seg, "start", 0) or 0)
        seg_end = float(getattr(seg, "end", 0) or 0)
        if seg_end < t_start:
            continue
        if seg_start > t_end:
            break
        text = (getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        pieces.append(text)
        total += len(text) + 1
        if total >= max_chars:
            break
    if not pieces:
        return ""
    excerpt = " ".join(pieces).strip()
    if len(excerpt) > max_chars:
        excerpt = excerpt[: max_chars - 1].rstrip() + "\u2026"
    return excerpt


def _pick_face_subject_x(
    face_results: Optional[list],
    t: float,
    tolerance: float = 1.5,
) -> int:
    """Pick the subject_x for a synthesized scene at time ``t``.

    Uses the nearest face-detection frame within ``tolerance`` seconds
    when available; otherwise returns 50 (center).
    """
    if not face_results:
        return 50
    nearest = None
    nearest_dist = float("inf")
    for fr in face_results:
        ts = float(getattr(fr, "timestamp", 0) or 0)
        dist = abs(ts - t)
        if dist < nearest_dist and dist <= tolerance:
            nearest_dist = dist
            nearest = fr
    if not nearest:
        return 50
    faces = getattr(nearest, "faces", None) or []
    if not faces:
        return 50
    primary_idx = getattr(nearest, "primary_face_idx", -1)
    if 0 <= primary_idx < len(faces):
        face = faces[primary_idx]
    else:
        face = faces[0]
    x_pct = getattr(face, "x_center", None) or getattr(face, "nose_x", None)
    if x_pct is None:
        return 50
    try:
        return max(0, min(100, int(round(float(x_pct)))))
    except (TypeError, ValueError):
        return 50


def synthesize_key_scenes(
    *,
    duration: float,
    scene_cuts: Optional[list[float]] = None,
    transcript: Optional[list] = None,
    face_results: Optional[list] = None,
    frames: Optional[list] = None,
    target_count: int = 12,
) -> list[SceneDescription]:
    """Build a list of key-moment ``SceneDescription`` objects.

    Strategy (in priority order):

    1. **Scene cuts** — every detected shot boundary becomes a key
       moment, capped at ``target_count``. The transcript text inside
       each shot is used as the description.
    2. **Transcript-driven** — if no scene cuts, sample evenly across
       the transcript (every Nth segment).
    3. **Uniform sampling** — final fallback: ``target_count`` evenly
       spaced points across the video duration.

    Returns ``[]`` when there is genuinely nothing to work with
    (duration <= 0 and all signals empty).
    """
    if duration <= 0:
        # Rare — protect against div-by-zero downstream.
        return []

    targets: list[float] = []

    # ── 1. Scene cuts ────────────────────────────────────────────
    if scene_cuts:
        cuts = sorted(set(float(t) for t in scene_cuts if 0 < float(t) < duration))
        if cuts:
            # Down-sample to target_count by even stride so we don't
            # spam the Key Scenes tab with 200 cut markers on long
            # videos.
            if len(cuts) > target_count:
                stride = len(cuts) / target_count
                targets = [cuts[int(i * stride)] for i in range(target_count)]
            else:
                targets = cuts

    # ── 2. Transcript fallback ───────────────────────────────────
    if not targets and transcript:
        segs = list(transcript)
        if segs:
            stride = max(1, len(segs) // target_count)
            for i in range(0, len(segs), stride):
                seg = segs[i]
                t = float(getattr(seg, "start", 0) or 0)
                if 0 < t < duration:
                    targets.append(t)
                if len(targets) >= target_count:
                    break

    # ── 3. Uniform sampling ──────────────────────────────────────
    if not targets:
        # Even spacing — first sample at ``duration / (n + 1)`` so we
        # don't pin a key moment to t=0 (which is usually black).
        n = max(1, target_count)
        targets = [duration * (i + 1) / (n + 1) for i in range(n)]

    # Build SceneDescription objects.
    out: list[SceneDescription] = []
    sorted_targets = sorted(set(_round_ts(t) for t in targets if t > 0))
    for idx, ts in enumerate(sorted_targets, start=1):
        # Window the transcript excerpt around this moment.
        excerpt = _excerpt_transcript(
            transcript or [], max(0.0, ts - 4.0), ts + 4.0,
        )
        desc = (
            f"Key moment {idx}: {excerpt}"
            if excerpt
            else f"Key moment {idx} at {ts:.1f}s"
        )
        # Find a frame thumbnail near this timestamp when available.
        thumb = ""
        if frames:
            best = None
            best_dist = float("inf")
            for fr in frames:
                ft = float(getattr(fr, "timestamp", 0) or 0)
                dist = abs(ft - ts)
                if dist < best_dist:
                    best_dist = dist
                    best = fr
            if best is not None and best_dist <= 5.0:
                thumb = getattr(best, "path", "") or ""

        out.append(
            SceneDescription(
                timestamp=ts,
                description=desc,
                importance_score=6,    # neutral middle score
                thumbnail_path=thumb,
                subject_x=_pick_face_subject_x(face_results, ts),
            )
        )
    return out
