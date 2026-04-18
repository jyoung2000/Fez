"""Chapter segmentation — Phase 6 of the OpusClip parity gap.

OpusClip's pitch is "understands the video → segments into chapters →
selects viral parts within each". Our pipeline used to scan fixed 30-s
windows, which clusters all clips at the front of long videos. This
module adds a simple topic-segmentation pass before clip detection.

Design constraints from the gap-close brief:

* No new ML models — TF-IDF cosine + scene-cut + speaker-change is
  enough to find a defensible chapter boundary.
* Skip on short videos (<3 min) — keep the legacy windowed scan.
* Title generation is keyword-based by default; an LLM call is
  optional and not used here (the orchestrator can synthesize titles
  later if it wants).
* Feature-flagged via ``USE_CHAPTER_SEGMENTATION`` — defaults off
  until the first end-to-end test passes per the brief.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

from backend.models import TranscriptSegment, SceneDescription

logger = logging.getLogger(__name__)


# Stop words to ignore when computing TF-IDF / extracting keywords.
# Kept compact — the goal is "topic shift detected", not perfect NLP.
_STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be been being but by can could did do does doing
    down for from had has have having he her here hers him his how i if
    in into is it its just like me my no nor not of off on once only or
    other our ours out over own same she should so some such than that
    the their them then there these they this those through to too
    under until up very was we were what when where which while who
    whom why will with would you your yours yeah okay ok um uh well
    really actually maybe right kind sort think going get got
    """.split()
)


@dataclass
class Chapter:
    start: float
    end: float
    title: str
    topic_keywords: list[str] = field(default_factory=list)
    speaker_turns: int = 0
    scene_cuts: int = 0

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "title": self.title,
            "topic_keywords": list(self.topic_keywords),
            "speaker_turns": self.speaker_turns,
            "scene_cuts": self.scene_cuts,
        }


def chapter_segmentation_enabled() -> bool:
    """Phase 6 feature flag — defaults off per the gap-close brief."""
    raw = os.environ.get("USE_CHAPTER_SEGMENTATION")
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z][a-z0-9'-]{1,}", text)
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]


def _bucket_text(
    transcript: list[TranscriptSegment],
    bucket_start: float,
    bucket_end: float,
) -> str:
    parts = [
        s.text for s in transcript
        if s.end > bucket_start and s.start < bucket_end
    ]
    return " ".join(parts)


def _cosine_similarity(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _topic_boundaries(
    transcript: list[TranscriptSegment],
    video_end: float,
    bucket_size: float = 60.0,
    threshold: float = 0.30,
) -> list[float]:
    """Find timestamps where the topic shifts.

    Walks the video in 60-second buckets and emits a boundary whenever
    the cosine similarity between the previous and current bucket's
    bag-of-words drops below ``threshold``.
    """
    boundaries: list[float] = []
    if not transcript or video_end <= bucket_size:
        return boundaries

    prev_counter: Counter | None = None
    t = 0.0
    while t < video_end:
        text = _bucket_text(transcript, t, t + bucket_size)
        tokens = _tokenize(text)
        counter: Counter = Counter(tokens)
        if prev_counter is not None and counter:
            sim = _cosine_similarity(prev_counter, counter)
            if sim < threshold:
                boundaries.append(round(t, 1))
        prev_counter = counter or prev_counter
        t += bucket_size
    return boundaries


def _scene_cut_boundaries(scenes: list[SceneDescription], min_gap: float = 30.0) -> list[float]:
    """Return scene timestamps spaced at least ``min_gap`` seconds apart."""
    if not scenes:
        return []
    sorted_scenes = sorted(scenes, key=lambda s: s.timestamp)
    out: list[float] = []
    last = -math.inf
    for s in sorted_scenes:
        if s.timestamp - last >= min_gap and s.importance_score >= 6:
            out.append(round(float(s.timestamp), 1))
            last = s.timestamp
    return out


def _speaker_change_boundaries(
    transcript: list[TranscriptSegment],
    min_hold: float = 20.0,
) -> list[float]:
    """Boundaries where one speaker holds the floor for ``min_hold`` seconds.

    A long uninterrupted monologue often marks a topic shift in
    podcasts and panels.
    """
    if not transcript:
        return []
    out: list[float] = []
    run_speaker: Optional[str] = None
    run_start = 0.0
    for seg in transcript:
        if seg.speaker != run_speaker:
            if run_speaker is not None and seg.start - run_start >= min_hold:
                out.append(round(seg.start, 1))
            run_speaker = seg.speaker
            run_start = seg.start
    return out


def _merge_close(boundaries: Iterable[float], min_gap: float) -> list[float]:
    sorted_b = sorted(set(round(float(b), 1) for b in boundaries))
    if not sorted_b:
        return []
    merged: list[float] = [sorted_b[0]]
    for b in sorted_b[1:]:
        if b - merged[-1] >= min_gap:
            merged.append(b)
    return merged


def _generate_title(
    transcript: list[TranscriptSegment],
    start: float,
    end: float,
    max_words: int = 4,
) -> tuple[str, list[str]]:
    """Pick the top-N TF-style keywords from the chapter's transcript.

    Returns (title, keyword_list).
    """
    text = _bucket_text(transcript, start, end)
    tokens = _tokenize(text)
    if not tokens:
        return f"Section {int(start // 60)}:{int(start % 60):02d}", []
    counter = Counter(tokens)
    keywords = [w for w, _ in counter.most_common(max_words)]
    title = " ".join(w.capitalize() for w in keywords)
    return title, keywords


def segment_chapters(
    transcript: list[TranscriptSegment],
    scenes: list[SceneDescription],
    audio_moments: list[dict] | None = None,
    min_len: float = 45.0,
    max_len: float = 300.0,
) -> list[Chapter]:
    """Segment a video into topic chapters.

    Combines three boundary signals:
      1. TF-IDF-style cosine drop between adjacent 60s windows.
      2. Scene cuts with importance_score >= 6.
      3. Long speaker holds (>20s of the same speaker).

    Boundaries are merged so no two are closer than ``min_len``, then
    capped to ``max_len`` (long stretches are split).

    Empty transcripts return a single chapter spanning the whole video.
    """
    if not transcript and not scenes:
        return []

    video_end = 0.0
    if transcript:
        video_end = max(s.end for s in transcript)
    if scenes:
        video_end = max(video_end, max(s.timestamp for s in scenes))
    if video_end <= 0:
        return []

    if video_end <= min_len * 2:
        # Too short to segment — return the whole thing as one chapter.
        title, kws = _generate_title(transcript, 0.0, video_end)
        return [Chapter(0.0, video_end, title, kws)]

    topic_b = _topic_boundaries(transcript, video_end)
    scene_b = _scene_cut_boundaries(scenes)
    speaker_b = _speaker_change_boundaries(transcript)

    # Combine and enforce min_len
    raw_boundaries = [b for b in (topic_b + scene_b + speaker_b) if 0 < b < video_end]
    boundaries = _merge_close(raw_boundaries, min_gap=min_len)

    # Build chapter ranges (insert 0 and video_end)
    cuts = [0.0] + boundaries + [video_end]
    chapters: list[Chapter] = []
    for i in range(len(cuts) - 1):
        cs = cuts[i]
        ce = cuts[i + 1]
        # Split chunks that exceed max_len by inserting evenly-spaced cuts
        while ce - cs > max_len:
            split = cs + max_len
            title, kws = _generate_title(transcript, cs, split)
            chapters.append(_build_chapter(cs, split, title, kws, transcript, scenes))
            cs = split
        title, kws = _generate_title(transcript, cs, ce)
        chapters.append(_build_chapter(cs, ce, title, kws, transcript, scenes))

    logger.info(
        "Chapter segmentation: %d chapters from %d topic + %d scene + %d speaker boundaries",
        len(chapters), len(topic_b), len(scene_b), len(speaker_b),
    )
    return chapters


def _build_chapter(
    start: float,
    end: float,
    title: str,
    keywords: list[str],
    transcript: list[TranscriptSegment],
    scenes: list[SceneDescription],
) -> Chapter:
    speaker_turns = 0
    prev_speaker: Optional[str] = None
    for s in transcript:
        if s.start < start or s.end > end:
            continue
        if prev_speaker is not None and s.speaker != prev_speaker:
            speaker_turns += 1
        prev_speaker = s.speaker
    scene_cuts = sum(
        1 for s in scenes
        if start <= s.timestamp <= end and s.importance_score >= 6
    )
    return Chapter(
        start=round(start, 1),
        end=round(end, 1),
        title=title,
        topic_keywords=keywords,
        speaker_turns=speaker_turns,
        scene_cuts=scene_cuts,
    )
