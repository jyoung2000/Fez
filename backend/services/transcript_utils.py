"""Shared transcript analysis utilities for clip detection.

Provides pre-processing functions that enrich the AI's input with energy
maps, natural boundary detection, and audio-visual correlation data.
Used by all providers to give the LLM richer signals for clip selection.
"""

import logging
from backend.models import TranscriptSegment, SceneDescription

logger = logging.getLogger(__name__)


def analyze_transcript_energy(
    transcript: list[TranscriptSegment], max_moments: int = 30
) -> str:
    """Generate an energy map identifying high-engagement transcript moments.

    Scans the transcript for signals like exclamations, questions, rapid
    exchanges, speaker changes, extended speeches, and reaction markers.
    Returns a formatted string ready to inject into LLM prompts.
    """
    if not transcript:
        return ""

    energy_moments: list[str] = []

    for i, seg in enumerate(transcript):
        text = seg.text.strip()
        if not text:
            continue
        energy_signals: list[str] = []

        # Exclamation / emphasis detection
        if text.count("!") >= 1:
            energy_signals.append("emphasis")
        if text.count("?") >= 1:
            energy_signals.append("question")

        # Raised-voice detection (all-caps words longer than 2 chars)
        if any(w == w.upper() and len(w) > 2 and w.isalpha() for w in text.split()):
            energy_signals.append("raised_voice")

        # Rapid exchange detection (short segments in quick succession)
        if (
            i > 0
            and (seg.start - transcript[i - 1].end) < 0.5
            and len(text.split()) < 15
        ):
            energy_signals.append("rapid_exchange")

        # Speaker change detection
        if i > 0 and seg.speaker != transcript[i - 1].speaker:
            energy_signals.append("speaker_change")

        # Long monologue detection (potential story / rant)
        word_count = len(text.split())
        if word_count > 50:
            energy_signals.append("extended_speech")

        # Laughter / reaction markers common in transcripts
        lower_text = text.lower()
        if any(
            marker in lower_text
            for marker in [
                "haha", "lol", "laugh", "[laughter]", "wow", "oh my",
                "oh no", "omg", "damn", "holy",
            ]
        ):
            energy_signals.append("reaction")

        # Disagreement / debate detection
        disagreement_markers = [
            "i disagree", "that's not true", "no way", "absolutely not",
            "you're wrong", "that's not right", "i don't think so",
            "but actually", "hold on", "wait a minute",
        ]
        if any(m in lower_text for m in disagreement_markers):
            energy_signals.append("disagreement")

        # Personal story / vulnerability detection
        story_markers = [
            "when i was", "i remember", "true story", "this happened to me",
            "i've never told", "for the first time", "i was so", "i couldn't believe",
        ]
        if any(m in lower_text for m in story_markers):
            energy_signals.append("personal_story")

        # Reveal / surprise detection
        reveal_markers = [
            "the truth is", "turns out", "plot twist", "nobody knows",
            "secret", "finally", "breaking", "just found out",
            "guess what", "you won't believe",
        ]
        if any(m in lower_text for m in reveal_markers):
            energy_signals.append("reveal")

        # List / ranking energy
        if any(f"number {n}" in lower_text or f"#{n}" in lower_text
               for n in range(1, 11)):
            energy_signals.append("ranking")

        # Strong opinion / hot take
        opinion_markers = [
            "the best", "the worst", "overrated", "underrated",
            "goat", "mid", "trash", "fire", "elite", "nobody can",
        ]
        if any(m in lower_text for m in opinion_markers):
            energy_signals.append("hot_take")

        if hasattr(seg, 'confidence') and seg.confidence is not None and seg.confidence < 0.4:
            energy_signals.append("low_confidence")

        if energy_signals:
            energy_moments.append(
                f"[{seg.start:.0f}s] {', '.join(energy_signals)}"
            )

    if not energy_moments:
        return ""

    capped = energy_moments[:max_moments]
    return (
        "\n\nTRANSCRIPT ENERGY MAP (high-engagement moments detected automatically):\n"
        + "\n".join(capped)
    )


def find_natural_boundaries(transcript: list[TranscriptSegment]) -> list[float]:
    """Identify natural clip boundary timestamps from speaker changes and pauses."""
    boundaries: list[float] = []
    for i in range(1, len(transcript)):
        prev, curr = transcript[i - 1], transcript[i]
        gap = curr.start - prev.end

        # Speaker change
        if curr.speaker != prev.speaker:
            boundaries.append(curr.start)
        # Significant pause (>1.5 seconds)
        elif gap > 1.5:
            boundaries.append(curr.start)

    return boundaries


def correlate_scenes_with_transcript(
    transcript: list[TranscriptSegment],
    scenes: list[SceneDescription],
    max_correlations: int = 20,
) -> str:
    """Find moments where high-importance visuals coincide with transcript energy.

    Returns a formatted string highlighting audio-visual peaks where both
    strong dialogue and strong visuals occur simultaneously.
    """
    hot_scenes = [s for s in scenes if s.importance_score >= 7]
    if not hot_scenes or not transcript:
        return ""

    correlations: list[str] = []
    for scene in hot_scenes:
        # Find transcript segments near this visual peak (within 10 seconds)
        nearby = [
            seg
            for seg in transcript
            if abs(seg.start - scene.timestamp) < 10
        ]
        if nearby:
            speakers = set(s.speaker for s in nearby)
            text_preview = " ".join(s.text[:50] for s in nearby[:2])
            correlations.append(
                f"[{scene.timestamp:.0f}s] Visual peak ({scene.importance_score}/10) + "
                f"dialogue by {', '.join(speakers)}: \"{text_preview}...\""
            )

    if not correlations:
        return ""

    capped = correlations[:max_correlations]
    return (
        "\n\nAUDIO-VISUAL PEAKS (strong dialogue coinciding with strong visuals — prioritize clips containing these):\n"
        + "\n".join(capped)
    )


def derive_content_guidance(video_summary: str | None) -> str:
    """Analyze video summary text to produce content-type-specific guidance.

    Returns a formatted guidance string that tells the AI what to prioritize
    and avoid based on the detected content type.
    """
    if not video_summary:
        return (
            "CONTENT TYPE: General\n"
            "PRIORITIZE: Emotional peaks, visual spectacle, quotable statements, "
            "complete micro-stories, reaction-worthy moments.\n\n"
        )

    summary_lower = video_summary.lower()

    if any(
        w in summary_lower
        for w in ["podcast", "interview", "conversation", "discussion"]
    ):
        return (
            "CONTENT TYPE: Conversational/Interview\n"
            "PRIORITIZE: Hot takes, disagreements, surprising admissions, emotional vulnerability, "
            "quotable one-liners, 'mic drop' moments, rapid-fire exchanges, audience-relatable stories. "
            "AVOID: Slow introductions, context-setting preambles, administrative talk.\n\n"
        )
    if any(
        w in summary_lower
        for w in ["tutorial", "how-to", "guide", "lesson", "teach"]
    ):
        return (
            "CONTENT TYPE: Tutorial/Educational\n"
            "PRIORITIZE: Complete mini-lessons with clear takeaway, surprising tips, "
            "'I never knew that' moments, before/after reveals, step-by-step segments that stand alone. "
            "AVOID: Mid-explanation cuts, clips that need previous context to understand.\n\n"
        )
    if any(
        w in summary_lower
        for w in ["game", "gaming", "stream", "gameplay", "play"]
    ):
        return (
            "CONTENT TYPE: Gaming/Stream\n"
            "PRIORITIZE: Clutch plays, epic fails, genuine reactions (rage, joy, shock), "
            "funny interactions, impressive skill displays, unexpected events, chat-worthy moments. "
            "AVOID: Routine gameplay, menu navigation, loading screens.\n\n"
        )
    if any(
        w in summary_lower for w in ["vlog", "travel", "daily", "routine", "life"]
    ):
        return (
            "CONTENT TYPE: Vlog/Lifestyle\n"
            "PRIORITIZE: Authentic emotional moments, beautiful visuals, funny mishaps, "
            "relatable experiences, satisfying reveals, unexpected turns in the narrative. "
            "AVOID: Mundane transitions, repetitive activities.\n\n"
        )
    if any(
        w in summary_lower
        for w in ["review", "unbox", "product", "comparison"]
    ):
        return (
            "CONTENT TYPE: Review/Product\n"
            "PRIORITIZE: First impressions, verdict moments, surprising findings, "
            "dramatic comparisons, deal-breaker reveals, 'worth it or not' conclusions. "
            "AVOID: Spec readings, unboxing filler, sponsor segments.\n\n"
        )
    if any(
        w in summary_lower for w in ["music", "song", "concert", "perform"]
    ):
        return (
            "CONTENT TYPE: Music/Performance\n"
            "PRIORITIZE: Best vocal/instrumental moments, crowd reactions, emotional peaks, "
            "surprising key changes, dance highlights, audience sing-alongs. "
            "AVOID: Sound check, between-song chatter, tuning.\n\n"
        )
    if any(
        w in summary_lower for w in ["comedy", "standup", "joke", "funny", "humor"]
    ):
        return (
            "CONTENT TYPE: Comedy/Entertainment\n"
            "PRIORITIZE: Punchlines, crowd reactions, callback jokes, physical comedy, "
            "relatable observations, unexpected twists. "
            "AVOID: Setup-only segments without payoff, dead air.\n\n"
        )
    if any(
        w in summary_lower for w in ["news", "report", "breaking", "politics"]
    ):
        return (
            "CONTENT TYPE: News/Commentary\n"
            "PRIORITIZE: Key revelations, strong opinions, heated exchanges, "
            "surprising statistics, quotable sound bites, emotional testimony. "
            "AVOID: Procedural reporting, reading teleprompter, transition filler.\n\n"
        )

    # Default for unrecognized content
    return (
        "CONTENT TYPE: General\n"
        "PRIORITIZE: Emotional peaks, visual spectacle, quotable statements, "
        "complete micro-stories, reaction-worthy moments.\n\n"
    )


# ── Filler Word and Dead Air Detection (Part 2) ──────────────────────────

FILLER_PATTERNS: dict[str, list[str]] = {
    "en": ["um", "uh", "uh huh", "you know", "i mean", "like", "basically",
           "literally", "actually", "right", "so yeah", "kind of", "sort of"],
}


def detect_filler_words(
    transcript: list[TranscriptSegment],
    language: str = "en",
) -> list[dict]:
    """Detect filler words and dead air in the transcript.

    Returns list of {start, end, type, text} dicts:
    - type="filler": filler word/phrase detected
    - type="dead_air": pause >1.5s between speech segments
    - type="repetition": word/phrase repeated immediately
    """
    fillers = FILLER_PATTERNS.get(language, FILLER_PATTERNS["en"])
    results: list[dict] = []

    for seg in transcript:
        text_lower = seg.text.lower().strip()
        words = text_lower.split()

        # Full-segment filler
        if text_lower in fillers:
            results.append({
                "start": seg.start, "end": seg.end,
                "type": "filler", "text": seg.text.strip(),
            })
            continue

        # Word-level filler detection using per-word timestamps
        if seg.words:
            for w in seg.words:
                word_lower = w.word.strip().lower().rstrip(".,!?")
                if word_lower in {"um", "uh", "hmm", "huh"}:
                    results.append({
                        "start": w.start, "end": w.end,
                        "type": "filler", "text": w.word.strip(),
                    })

        # Repetition detection
        for i in range(len(words) - 1):
            if words[i] == words[i + 1] and len(words[i]) > 1:
                results.append({
                    "start": seg.start, "end": seg.end,
                    "type": "repetition", "text": f"{words[i]} {words[i+1]}",
                })
                break

    # Dead air detection
    for i in range(1, len(transcript)):
        gap = transcript[i].start - transcript[i - 1].end
        if gap > 1.5:
            results.append({
                "start": transcript[i - 1].end,
                "end": transcript[i].start,
                "type": "dead_air",
                "text": f"{gap:.1f}s silence",
            })

    return results


def compute_filler_density(
    filler_events: list[dict],
    start: float,
    end: float,
) -> float:
    """Compute filler density (fillers per minute) for a time window."""
    window_events = [e for e in filler_events if e["start"] >= start and e["end"] <= end]
    duration_min = max((end - start) / 60.0, 0.01)
    return len(window_events) / duration_min


# ── Keyword Emphasis Detection (Part 9) ──────────────────────────────────

def detect_emphasis_words(
    transcript: list[TranscriptSegment],
    video_summary: str = "",
    max_keywords: int = 20,
) -> set[str]:
    """Identify words that should be visually emphasized in captions.

    Returns a set of lowercase words that deserve highlighting:
    - Key topics from the video summary
    - Proper nouns (capitalized in middle of sentences)
    - Numbers and statistics
    - Words spoken with emphasis (all-caps in transcript)
    """
    import re as _re

    keywords: set[str] = set()

    if video_summary:
        caps_words = _re.findall(r'\b[A-Z][a-z]{2,}\b', video_summary)
        for w in caps_words[:10]:
            keywords.add(w.lower())

    for seg in transcript:
        words = seg.text.split()
        for i, word in enumerate(words):
            clean = word.strip(".,!?\"'()[]")
            if not clean:
                continue

            # All-caps words (shouted/emphasized)
            if clean == clean.upper() and len(clean) > 2 and clean.isalpha():
                keywords.add(clean.lower())

            # Numbers and statistics
            if any(c.isdigit() for c in clean) and len(clean) <= 10:
                keywords.add(clean.lower())

            # Proper nouns (capitalized mid-sentence)
            if i > 0 and clean[0:1].isupper() and len(clean) > 2 and clean.isalpha():
                keywords.add(clean.lower())

    # Common emphasis words present in transcript
    emphasis_always = {"never", "always", "best", "worst", "only", "first", "last",
                       "million", "billion", "percent", "secret", "truth", "real"}
    for seg in transcript:
        seg_lower = seg.text.lower()
        for w in emphasis_always:
            if w in seg_lower:
                keywords.add(w)

    return set(list(keywords)[:max_keywords])
