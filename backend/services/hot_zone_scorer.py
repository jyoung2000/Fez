"""Hot zone scoring — fast heuristic pre-analysis for clip detection.

Combines audio energy, transcript energy, scene importance, and speaker
dynamics into a composite score per time window.  Produces a ranked list
of "hot zones" that the AI should prioritize for clip detection.

Runs in <2 seconds with no AI calls — purely local computation.
"""

import logging
from dataclasses import dataclass, field
from backend.models import TranscriptSegment, SceneDescription

logger = logging.getLogger(__name__)

WINDOW_SIZE = 30.0    # seconds per scoring window
WINDOW_OVERLAP = 10.0 # seconds of overlap between windows


@dataclass
class HotZone:
    start: float
    end: float
    composite_score: float  # 0-100
    audio_score: float
    transcript_score: float
    scene_score: float
    speaker_score: float
    signals: list[str] = field(default_factory=list)


def score_hot_zones(
    transcript: list[TranscriptSegment],
    scenes: list[SceneDescription],
    audio_moments: list[dict],
    video_duration: float,
    window_size: float = WINDOW_SIZE,
    overlap: float = WINDOW_OVERLAP,
    filler_events: list[dict] | None = None,
    trend_matcher=None,
) -> list[HotZone]:
    """Score every time window in the video for viral potential.

    Returns zones sorted by composite_score descending.

    If ``trend_matcher`` (a ``TrendMatcher`` instance from
    ``backend.services.trend_matcher``) is supplied, an additional
    trend axis with weight 0.15 is folded into the composite. When
    omitted, the legacy 4-signal weights are preserved bit-for-bit so
    the no-flag default has no behavior change.
    """
    if video_duration <= 0:
        return []

    zones = []
    window_start = 0.0

    while window_start < video_duration:
        window_end = min(window_start + window_size, video_duration)

        audio_score, audio_signals = _score_audio(audio_moments, window_start, window_end)
        transcript_score, transcript_signals = _score_transcript(transcript, window_start, window_end)
        scene_score, scene_signals = _score_scenes(scenes, window_start, window_end)
        speaker_score, speaker_signals = _score_speakers(transcript, window_start, window_end)

        # Filler penalty: high filler density = low engagement potential
        filler_penalty = 0.0
        if filler_events:
            from backend.services.transcript_utils import compute_filler_density
            density = compute_filler_density(filler_events, window_start, window_end)
            if density > 6:
                filler_penalty = 15
                audio_signals.append("high filler density")
            elif density > 3:
                filler_penalty = 8

        trend_signals: list[str] = []
        if trend_matcher is not None:
            window_text = " ".join(
                s.text for s in transcript
                if s.start >= window_start and s.end <= window_end
            )
            try:
                trend_score_val, trend_reason = trend_matcher.score_clip(window_text)
            except Exception:
                trend_score_val, trend_reason = 50, ""
            if trend_reason and trend_score_val >= 60:
                trend_signals.append(f"trend match: {trend_reason}")
            composite = (
                audio_score * 0.20
                + transcript_score * 0.30
                + scene_score * 0.20
                + speaker_score * 0.15
                + trend_score_val * 0.15
            ) - filler_penalty
        else:
            # Legacy 4-signal weights — unchanged.
            composite = (
                audio_score * 0.25 +
                transcript_score * 0.35 +
                scene_score * 0.25 +
                speaker_score * 0.15
            ) - filler_penalty

        signals = audio_signals + transcript_signals + scene_signals + speaker_signals + trend_signals

        zones.append(HotZone(
            start=window_start,
            end=window_end,
            composite_score=round(composite, 1),
            audio_score=round(audio_score, 1),
            transcript_score=round(transcript_score, 1),
            scene_score=round(scene_score, 1),
            speaker_score=round(speaker_score, 1),
            signals=signals,
        ))

        window_start += window_size - overlap

    zones.sort(key=lambda z: z.composite_score, reverse=True)
    return zones


def _score_audio(moments: list[dict], start: float, end: float) -> tuple[float, list[str]]:
    """Score audio energy in a time window.

    Phase 5 (OpusClip parity gap) — when audio moments carry a
    ``sentiment`` tag (laughter / cheering / shouting / applause)
    those are treated as first-class signals on top of raw loudness.
    """
    window_moments = [m for m in moments if start <= m.get("timestamp", 0) <= end]
    if not window_moments:
        return 0.0, []

    signals = []
    score = 0.0

    for m in window_moments:
        mtype = m.get("type", "")
        ts = m.get("timestamp", 0)
        sentiment = m.get("sentiment", "")
        if mtype == "extreme_spike":
            score += 30
            signals.append(f"extreme audio spike at {ts:.0f}s")
        elif mtype == "silence_to_loud":
            score += 25
            signals.append(f"silence->loud at {ts:.0f}s")
        elif mtype == "volume_spike":
            score += 15

        # Sentiment-tag bonus / penalty — Phase 5
        if sentiment in ("laughter", "cheering"):
            score += 15
            signals.append(f"{sentiment} at {ts:.0f}s")
        elif sentiment == "applause":
            score += 10
        elif sentiment == "shouting":
            score += 8

    return min(100, score), signals[:3]


def _score_transcript(transcript: list[TranscriptSegment], start: float, end: float) -> tuple[float, list[str]]:
    """Score transcript energy in a time window."""
    segments = [s for s in transcript if s.start >= start and s.end <= end]
    if not segments:
        return 0.0, []

    score = 0.0
    signals = []

    for seg in segments:
        text = seg.text.strip()
        lower = text.lower()

        if "!" in text:
            score += 8
        if "?" in text:
            score += 6

        # Raised voice (all-caps words)
        caps_words = [w for w in text.split() if w == w.upper() and len(w) > 2 and w.isalpha()]
        if caps_words:
            score += 10
            if "raised voice" not in signals:
                signals.append("raised voice")

        # Emotional/reaction markers
        reaction_words = [
            "haha", "lol", "wow", "oh my", "oh no", "omg", "damn", "holy",
            "crazy", "insane", "unbelievable", "incredible", "amazing",
        ]
        if any(w in lower for w in reaction_words):
            score += 12
            if "emotional reaction" not in signals:
                signals.append("emotional reaction")

        # Quick statements = fast-paced content
        word_count = len(text.split())
        if word_count > 3 and (seg.end - seg.start) < 3:
            score += 5

    # Segment density bonus
    density = len(segments) / max(1, (end - start) / 10)
    if density > 2:
        score += 10
        signals.append("dense dialogue")

    return min(100, score), signals[:3]


def _score_scenes(scenes: list[SceneDescription], start: float, end: float) -> tuple[float, list[str]]:
    """Score scene visual importance in a time window."""
    window_scenes = [s for s in scenes if start <= s.timestamp <= end]
    if not window_scenes:
        return 0.0, []

    signals = []
    max_importance = max(s.importance_score for s in window_scenes)
    avg_importance = sum(s.importance_score for s in window_scenes) / len(window_scenes)

    score = max_importance * 6 + avg_importance * 4

    if max_importance >= 8:
        signals.append(f"visual peak ({max_importance}/10)")
    if max_importance >= 9:
        score += 20

    return min(100, score), signals[:2]


def _score_speakers(transcript: list[TranscriptSegment], start: float, end: float) -> tuple[float, list[str]]:
    """Score speaker dynamics in a time window."""
    segments = [s for s in transcript if s.start >= start and s.end <= end]
    if not segments:
        return 0.0, []

    signals = []
    score = 0.0

    speakers = set(s.speaker for s in segments)
    speaker_changes = 0
    for i in range(1, len(segments)):
        if segments[i].speaker != segments[i - 1].speaker:
            speaker_changes += 1

    if len(speakers) >= 2:
        score += 20
        signals.append(f"{len(speakers)} speakers")

    if speaker_changes >= 3:
        score += 25
        signals.append("rapid exchange")
    elif speaker_changes >= 1:
        score += 10

    return min(100, score), signals[:2]


def score_hot_zones_transcript_only(
    transcript: list[TranscriptSegment],
    video_duration: float,
    window_size: float = WINDOW_SIZE,
    overlap: float = WINDOW_OVERLAP,
) -> list[HotZone]:
    """Pre-score hot zones using only transcript data (no scenes or audio needed).

    Used for frame triage BEFORE scene analysis runs. Produces rough hot zone
    estimates good enough for deciding which frames to analyze in detail.
    Re-scored with full data (scenes + audio) later in the pipeline.
    """
    if video_duration <= 0:
        return []

    zones = []
    window_start = 0.0

    while window_start < video_duration:
        window_end = min(window_start + window_size, video_duration)

        transcript_score, transcript_signals = _score_transcript(transcript, window_start, window_end)
        speaker_score, speaker_signals = _score_speakers(transcript, window_start, window_end)

        # Without audio/scene data, weight transcript and speakers more heavily
        composite = transcript_score * 0.65 + speaker_score * 0.35
        signals = transcript_signals + speaker_signals

        zones.append(HotZone(
            start=window_start,
            end=window_end,
            composite_score=round(composite, 1),
            audio_score=0.0,
            transcript_score=round(transcript_score, 1),
            scene_score=0.0,
            speaker_score=round(speaker_score, 1),
            signals=signals,
        ))

        window_start += window_size - overlap

    zones.sort(key=lambda z: z.composite_score, reverse=True)
    return zones


def format_hot_zones_for_prompt(zones: list[HotZone], top_n: int = 15) -> str:
    """Format top hot zones for injection into AI clip detection prompt."""
    if not zones:
        return ""

    top = zones[:top_n]
    lines = []
    for z in top:
        signal_str = ", ".join(z.signals) if z.signals else "general engagement"
        lines.append(
            f"[{z.start:.0f}-{z.end:.0f}s] score={z.composite_score:.0f} "
            f"(audio={z.audio_score:.0f}, transcript={z.transcript_score:.0f}, "
            f"visual={z.scene_score:.0f}) — {signal_str}"
        )

    return (
        "\n\nHOT ZONES (pre-scored regions ranked by viral potential — "
        "PRIORITIZE clips that overlap with high-scoring zones):\n"
        + "\n".join(lines)
    )


def get_window_hot_zones(zones: list[HotZone], window_start: float, window_end: float, top_n: int = 8) -> list[HotZone]:
    """Get the top hot zones within a specific time window."""
    window_zones = [z for z in zones if z.start >= window_start and z.end <= window_end]
    window_zones.sort(key=lambda z: z.composite_score, reverse=True)
    return window_zones[:top_n]


def get_coverage_gaps(
    zones: list[HotZone],
    found_clips: list,  # list of ClipCandidate
    video_duration: float,
    min_gap_duration: float = 60.0,
    max_gaps: int = 4,
) -> list[tuple[float, float]]:
    """Find time regions with no clips that still have decent hot zone scores.

    Used for multi-pass clip detection — identifies where to do a second scan.
    Returns list of (start, end) tuples for under-covered regions.
    """
    if not found_clips or video_duration <= 0:
        return [(0, video_duration)]

    # Build coverage bitmap (1-second resolution)
    coverage = [False] * int(video_duration + 1)
    for clip in found_clips:
        clip_start = int(getattr(clip, "start_time", 0))
        clip_end = int(getattr(clip, "end_time", 0))
        for t in range(clip_start, min(clip_end + 1, len(coverage))):
            coverage[t] = True

    # Find uncovered gaps
    gaps = []
    gap_start = None
    for t in range(len(coverage)):
        if not coverage[t]:
            if gap_start is None:
                gap_start = t
        else:
            if gap_start is not None:
                gap_duration = t - gap_start
                if gap_duration >= min_gap_duration:
                    gaps.append((float(gap_start), float(t)))
                gap_start = None

    # Handle trailing gap
    if gap_start is not None:
        gap_duration = len(coverage) - gap_start
        if gap_duration >= min_gap_duration:
            gaps.append((float(gap_start), float(len(coverage))))

    # Filter: only return gaps that overlap with zones scoring >20
    if zones:
        decent_zones = [z for z in zones if z.composite_score > 20]
        if decent_zones:
            filtered_gaps = []
            for gs, ge in gaps:
                has_potential = any(
                    z.start < ge and z.end > gs
                    for z in decent_zones
                )
                if has_potential:
                    filtered_gaps.append((gs, ge))
            return filtered_gaps[:max_gaps]

    return gaps[:max_gaps]
