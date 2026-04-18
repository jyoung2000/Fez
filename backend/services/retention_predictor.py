"""Audience retention predictor for clip candidates.

Predicts viewer drop-off per clip using multiple engagement signals:
- Hook strength (first 3s analysis)
- Pacing variance (words per minute)
- Visual change frequency
- Emotional arc (sentiment trajectory)
- Standalone comprehension

Generates a retention curve that can be displayed in the frontend.
"""

import logging
from typing import Optional

from backend.models import TranscriptSegment, SceneDescription, ClipCandidate

logger = logging.getLogger(__name__)


def _calculate_hook_strength(
    clip: ClipCandidate,
    transcript: list[TranscriptSegment],
) -> float:
    """Score the hook (first 3 seconds) of a clip. Returns 0.0-1.0."""
    hook_end = clip.start_time + 3.0

    # Find transcript segments in the hook window
    hook_segments = [
        seg for seg in transcript
        if seg.start < hook_end and seg.end > clip.start_time
    ]

    score = 0.5  # Baseline

    if not hook_segments:
        return 0.3  # No dialogue in hook — weaker

    hook_text = " ".join(seg.text for seg in hook_segments)
    hook_lower = hook_text.lower()

    # Question hooks are strong
    if "?" in hook_text:
        score += 0.15

    # Exclamation = energy
    if "!" in hook_text:
        score += 0.1

    # Short, punchy hooks are better
    word_count = len(hook_text.split())
    if 3 <= word_count <= 15:
        score += 0.1

    # Hooks with strong emotional/curiosity words
    curiosity_words = ["secret", "never", "always", "worst", "best", "crazy", "insane",
                       "shocking", "truth", "actually", "honestly", "literally", "imagine"]
    if any(w in hook_lower for w in curiosity_words):
        score += 0.15

    return min(1.0, score)


def _calculate_pacing_variance(
    clip: ClipCandidate,
    transcript: list[TranscriptSegment],
) -> float:
    """Score pacing variety. Returns 0.0-1.0 (higher = more varied = more engaging)."""
    clip_segments = [
        seg for seg in transcript
        if seg.start >= clip.start_time and seg.end <= clip.end_time
    ]

    if len(clip_segments) < 3:
        return 0.5  # Not enough data

    # Calculate WPM per segment
    wpms = []
    for seg in clip_segments:
        duration = seg.end - seg.start
        if duration > 0:
            wpm = len(seg.text.split()) / (duration / 60)
            wpms.append(wpm)

    if not wpms:
        return 0.5

    avg_wpm = sum(wpms) / len(wpms)
    if avg_wpm == 0:
        return 0.3

    # Calculate coefficient of variation (higher = more varied pacing)
    variance = sum((w - avg_wpm) ** 2 for w in wpms) / len(wpms)
    std_dev = variance ** 0.5
    cv = std_dev / avg_wpm

    # Optimal CV is 0.2-0.5 (some variety but not chaotic)
    if 0.2 <= cv <= 0.5:
        return 0.8
    elif cv < 0.1:
        return 0.4  # Monotone
    elif cv > 0.8:
        return 0.5  # Too chaotic
    else:
        return 0.6


def _calculate_visual_change_frequency(
    clip: ClipCandidate,
    scenes: list[SceneDescription],
) -> float:
    """Score visual dynamism. Returns 0.0-1.0."""
    clip_scenes = [
        s for s in scenes
        if s.timestamp >= clip.start_time and s.timestamp <= clip.end_time
    ]

    if not clip_scenes:
        return 0.5

    # Calculate scene changes per minute
    scene_count = len(clip_scenes)
    duration_min = clip.duration / 60

    if duration_min == 0:
        return 0.5

    changes_per_min = scene_count / duration_min
    high_importance = sum(1 for s in clip_scenes if s.importance_score >= 7)

    # Base score from pacing
    if 3 <= changes_per_min <= 8:
        base = 0.8
    elif changes_per_min < 1:
        base = 0.3
    elif changes_per_min > 15:
        base = 0.5
    else:
        base = 0.6

    # Bonus for high-importance scenes
    if high_importance > 0:
        importance_bonus = min(0.15, high_importance * 0.05)
        return min(1.0, base + importance_bonus)

    return base


def _calculate_emotional_arc(
    clip: ClipCandidate,
    transcript: list[TranscriptSegment],
) -> float:
    """Score emotional trajectory. Returns 0.0-1.0 (higher = stronger arc)."""
    clip_segments = [
        seg for seg in transcript
        if seg.start >= clip.start_time and seg.end <= clip.end_time
    ]

    if len(clip_segments) < 2:
        return 0.5

    # Simple sentiment approximation using text features
    def _segment_energy(seg: TranscriptSegment) -> float:
        text = seg.text
        energy = 0.5
        if "!" in text:
            energy += 0.2
        if "?" in text:
            energy += 0.1
        if any(w == w.upper() and len(w) > 2 and w.isalpha() for w in text.split()):
            energy += 0.15
        lower = text.lower()
        if any(w in lower for w in ["wow", "oh my", "amazing", "incredible", "unbelievable"]):
            energy += 0.2
        if any(w in lower for w in ["sad", "unfortunately", "terrible", "awful"]):
            energy += 0.1  # Emotional = engaging
        return min(1.0, energy)

    energies = [_segment_energy(seg) for seg in clip_segments]

    # Best arcs have increasing energy or a build-drop-build pattern
    # Check if energy generally increases toward the end
    mid = len(energies) // 2
    first_half_avg = sum(energies[:mid]) / max(len(energies[:mid]), 1)
    second_half_avg = sum(energies[mid:]) / max(len(energies[mid:]), 1)

    if second_half_avg > first_half_avg + 0.1:
        return 0.8  # Building energy — strong arc
    elif abs(second_half_avg - first_half_avg) < 0.05:
        return 0.4  # Flat — weak arc
    else:
        return 0.6  # Some variation


def _calculate_standalone_comprehension(
    clip: ClipCandidate,
    transcript: list[TranscriptSegment],
) -> float:
    """Score how well the clip works without prior context. Returns 0.0-1.0."""
    clip_segments = [
        seg for seg in transcript
        if seg.start >= clip.start_time and seg.end <= clip.end_time
    ]

    if not clip_segments:
        return 0.5

    first_text = clip_segments[0].text.lower() if clip_segments else ""

    score = 0.6  # Baseline

    # Clips starting with pronouns without antecedent are harder to follow
    if first_text.startswith(("he ", "she ", "they ", "it ", "that ", "this ")):
        score -= 0.15

    # Clips starting with conjunctions suggest mid-thought
    if first_text.startswith(("and ", "but ", "so ", "because ", "then ")):
        score -= 0.1

    # Multiple speakers provide more context
    speakers = set(seg.speaker for seg in clip_segments)
    if len(speakers) >= 2:
        score += 0.1

    # Longer clips are generally more self-contained
    if clip.duration >= 60:
        score += 0.1

    return max(0.0, min(1.0, score))


def predict_retention(
    clip: ClipCandidate,
    transcript: list[TranscriptSegment],
    scenes: list[SceneDescription],
    num_points: int = 20,
) -> dict:
    """Predict audience retention curve for a clip.

    Returns a dict with:
    - retention_curve: list of {time_pct, retention_pct} points
    - overall_score: 0-100 predicted retention score
    - factors: breakdown of contributing factors
    """
    # Calculate individual factors
    hook = _calculate_hook_strength(clip, transcript)
    pacing = _calculate_pacing_variance(clip, transcript)
    visual = _calculate_visual_change_frequency(clip, scenes)
    arc = _calculate_emotional_arc(clip, transcript)
    standalone = _calculate_standalone_comprehension(clip, transcript)

    # Weighted overall score
    overall = (
        hook * 0.25 +
        pacing * 0.15 +
        visual * 0.15 +
        arc * 0.25 +
        standalone * 0.20
    )

    # Generate retention curve
    # Model: starts at 100%, drops sharply in first 10%, then gradual decline
    # Modified by hook strength (less initial drop) and arc (less late drop)
    curve = []
    for i in range(num_points):
        t_pct = (i / (num_points - 1)) * 100

        if t_pct <= 5:
            # First 5%: everyone watches the start
            retention = 100.0
        elif t_pct <= 15:
            # 5-15%: hook zone — sharp drop modulated by hook strength
            hook_drop = (1.0 - hook) * 25  # Better hook = less drop (0-25%)
            progress = (t_pct - 5) / 10
            retention = 100.0 - (hook_drop * progress)
        elif t_pct <= 80:
            # 15-80%: middle — gradual decline modulated by pacing + visual
            start_retention = 100.0 - (1.0 - hook) * 25
            middle_factor = (pacing + visual) / 2
            middle_drop = (1.0 - middle_factor) * 20  # 0-20% total middle drop
            progress = (t_pct - 15) / 65
            retention = start_retention - (middle_drop * progress)
        else:
            # 80-100%: ending — modulated by arc (strong arc = retention bump)
            mid_retention = 100.0 - (1.0 - hook) * 25 - (1.0 - (pacing + visual) / 2) * 20
            if arc > 0.6:
                # Strong arc: slight retention increase at the end
                end_bump = (arc - 0.6) * 10
                progress = (t_pct - 80) / 20
                retention = mid_retention + (end_bump * progress)
            else:
                # Weak arc: continued decline
                end_drop = (1.0 - arc) * 10
                progress = (t_pct - 80) / 20
                retention = mid_retention - (end_drop * progress)

        retention = max(20.0, min(100.0, retention))
        curve.append({
            "time_pct": round(t_pct, 1),
            "retention_pct": round(retention, 1),
        })

    return {
        "retention_curve": curve,
        "overall_score": round(overall * 100),
        "factors": {
            "hook_strength": round(hook * 100),
            "pacing_variance": round(pacing * 100),
            "visual_dynamism": round(visual * 100),
            "emotional_arc": round(arc * 100),
            "standalone_comprehension": round(standalone * 100),
        },
    }


def predict_retention_for_clips(
    clips: list[ClipCandidate],
    transcript: list[TranscriptSegment],
    scenes: list[SceneDescription],
) -> dict[int, dict]:
    """Predict retention for all clips. Returns {clip_id: retention_data}."""
    results = {}
    for clip in clips:
        try:
            results[clip.id] = predict_retention(clip, transcript, scenes)
        except Exception as e:
            logger.warning("Retention prediction failed for clip %d: %s", clip.id, e)
            results[clip.id] = {
                "retention_curve": [],
                "overall_score": 50,
                "factors": {},
            }

    scored = [(cid, data["overall_score"]) for cid, data in results.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    logger.info(
        "Retention predictions: %s",
        ", ".join(f"clip {cid}={score}" for cid, score in scored[:5]),
    )
    return results
