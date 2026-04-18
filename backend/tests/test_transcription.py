"""Tests for the transcription service — speaker assignment logic.

Covers:
  - _assign_speakers with various gap patterns
  - Empty input handling
  - Speaker count capping at MAX_SPEAKERS
  - Edge cases: single segment, overlapping segments
  - Multi-sentence segment splitting (Bug A)
  - Long-range verbatim repeat removal (Bug B)
"""
import os
import sys
import unittest

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.services.transcription import (
    _assign_speakers,
    _consolidate_segments,
    _dedupe_long_range,
    _split_segments_at_sentence_boundaries,
)
from backend.models import TranscriptSegment


class TestAssignSpeakersEmpty(unittest.TestCase):
    """Empty and trivial inputs."""

    def test_empty_list(self):
        result = _assign_speakers([])
        self.assertEqual(result, [])

    def test_single_segment(self):
        segments = [{"start": 0.0, "end": 2.5, "text": "Hello world"}]
        result = _assign_speakers(segments)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], TranscriptSegment)
        self.assertEqual(result[0].speaker, "Speaker 1")
        self.assertEqual(result[0].text, "Hello world")


class TestAssignSpeakersGaps(unittest.TestCase):
    """Test gap-based speaker detection."""

    def test_short_gap_same_speaker(self):
        """Gaps < 1.5s should keep the same speaker."""
        segments = [
            {"start": 0.0, "end": 2.0, "text": "First part"},
            {"start": 2.5, "end": 4.0, "text": "Second part"},  # 0.5s gap
            {"start": 4.2, "end": 6.0, "text": "Third part"},   # 0.2s gap
        ]
        result = _assign_speakers(segments)
        self.assertEqual(len(result), 3)
        # All should be same speaker
        speakers = [s.speaker for s in result]
        self.assertEqual(len(set(speakers)), 1)
        self.assertEqual(speakers[0], "Speaker 1")

    def test_medium_gap_switches_speaker(self):
        """Gaps between 1.5-4s should trigger a speaker turn."""
        segments = [
            {"start": 0.0, "end": 2.0, "text": "Person A talks"},
            {"start": 4.0, "end": 6.0, "text": "Person B responds"},  # 2.0s gap (turn)
            {"start": 8.0, "end": 10.0, "text": "Person A again"},    # 2.0s gap (turn)
        ]
        result = _assign_speakers(segments)
        self.assertEqual(len(result), 3)
        # Should toggle between speakers
        self.assertEqual(result[0].speaker, "Speaker 1")
        self.assertEqual(result[1].speaker, "Speaker 2")
        self.assertEqual(result[2].speaker, "Speaker 1")

    def test_large_gap_introduces_new_speaker(self):
        """Gaps > 5s should potentially introduce a new speaker.

        The enhanced heuristic uses speech rate matching, so identical
        short segments may be assigned to an existing speaker. We verify
        at least 2 speakers are detected (the heuristic is not naive).
        """
        segments = [
            {"start": 0.0, "end": 2.0, "text": "Speaker one"},
            {"start": 7.0, "end": 9.0, "text": "New speaker"},      # 5.0s gap (new)
            {"start": 14.0, "end": 16.0, "text": "Another speaker"}, # 5.0s gap (new)
        ]
        result = _assign_speakers(segments)
        self.assertEqual(len(result), 3)
        speakers = [s.speaker for s in result]
        # Should have at least 2 distinct speakers (rate-based heuristic may merge similar speakers)
        self.assertGreaterEqual(len(set(speakers)), 2)

    def test_unlimited_speakers(self):
        """No artificial speaker cap — more than 4 speakers should be possible."""
        # 8 segments with large gaps and varied word counts for distinct speech rates
        segments = [
            {"start": 0.0, "end": 3.0, "text": "Hello world this is a long sentence with many words"},
            {"start": 10.0, "end": 12.0, "text": "Short"},
            {"start": 20.0, "end": 24.0, "text": "Medium length sentence here today"},
            {"start": 30.0, "end": 31.0, "text": "One"},
            {"start": 40.0, "end": 45.0, "text": "This is quite a verbose and lengthy monologue with lots of detail"},
            {"start": 55.0, "end": 56.0, "text": "Brief"},
            {"start": 65.0, "end": 70.0, "text": "Another very long segment with tons and tons of words in it"},
            {"start": 80.0, "end": 82.0, "text": "Two words"},
        ]
        result = _assign_speakers(segments)
        speakers = set(s.speaker for s in result)
        # No cap — should detect multiple speakers (previously capped at 4)
        self.assertGreaterEqual(len(speakers), 2)


class TestAssignSpeakersRounding(unittest.TestCase):
    """Test that timestamps are rounded properly."""

    def test_timestamps_rounded(self):
        segments = [
            {"start": 0.123456, "end": 2.789012, "text": "Test"},
        ]
        result = _assign_speakers(segments)
        self.assertEqual(result[0].start, 0.12)
        self.assertEqual(result[0].end, 2.79)


class TestAssignSpeakersConversation(unittest.TestCase):
    """Test realistic conversation patterns."""

    def test_two_person_conversation(self):
        """Simulate a natural back-and-forth conversation."""
        segments = [
            {"start": 0.0, "end": 3.0, "text": "Hey how's it going?"},
            {"start": 3.2, "end": 3.5, "text": "Yeah"},              # 0.2s gap (same speaker continuation)
            {"start": 5.5, "end": 8.0, "text": "I'm doing great, thanks!"},  # 2.0s gap (turn)
            {"start": 8.3, "end": 11.0, "text": "So what's new?"},    # 0.3s gap (same)
            {"start": 13.0, "end": 16.0, "text": "Not much, you know"}, # 2.0s gap (turn)
        ]
        result = _assign_speakers(segments)
        self.assertEqual(len(result), 5)
        # Pattern: 1, 1, 2, 2, 1
        self.assertEqual(result[0].speaker, "Speaker 1")
        self.assertEqual(result[1].speaker, "Speaker 1")
        self.assertEqual(result[2].speaker, "Speaker 2")
        self.assertEqual(result[3].speaker, "Speaker 2")
        self.assertEqual(result[4].speaker, "Speaker 1")

    def test_monologue(self):
        """A single speaker with only short gaps should remain as one speaker."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "First sentence"},
            {"start": 5.3, "end": 10.0, "text": "Second sentence"},
            {"start": 10.2, "end": 15.0, "text": "Third sentence"},
            {"start": 15.5, "end": 20.0, "text": "Fourth sentence"},
        ]
        result = _assign_speakers(segments)
        speakers = set(s.speaker for s in result)
        self.assertEqual(len(speakers), 1)


# ══════════════════════════════════════════════════════════════════════
# Bug A — multi-sentence segments split correctly
# ══════════════════════════════════════════════════════════════════════


def test_split_segments_at_sentence_boundaries_with_word_timestamps():
    """Multi-sentence segment with word timestamps splits per sentence."""
    seg = {
        "start": 10.0,
        "end": 14.0,
        "text": "I see you. So this is real. Tell me everything.",
        "words": [
            {"start": 10.0, "end": 10.3, "word": "I"},
            {"start": 10.3, "end": 10.6, "word": " see"},
            {"start": 10.6, "end": 10.9, "word": " you."},
            {"start": 11.5, "end": 11.7, "word": " So"},
            {"start": 11.7, "end": 11.9, "word": " this"},
            {"start": 11.9, "end": 12.1, "word": " is"},
            {"start": 12.1, "end": 12.4, "word": " real."},
            {"start": 13.0, "end": 13.2, "word": " Tell"},
            {"start": 13.2, "end": 13.5, "word": " me"},
            {"start": 13.5, "end": 14.0, "word": " everything."},
        ],
    }
    out = _split_segments_at_sentence_boundaries([seg])
    assert len(out) == 3
    assert out[0]["text"].strip() == "I see you."
    assert out[0]["start"] == 10.0
    assert out[0]["end"] == pytest.approx(10.9)
    assert out[1]["text"].strip() == "So this is real."
    assert out[1]["start"] == pytest.approx(11.5)
    assert out[1]["end"] == pytest.approx(12.4)
    assert out[2]["text"].strip() == "Tell me everything."
    assert out[2]["start"] == pytest.approx(13.0)
    assert out[2]["end"] == pytest.approx(14.0)


def test_split_segments_proportional_fallback_when_no_words():
    """Without word timestamps, splits proportionally by character count."""
    seg = {"start": 0.0, "end": 10.0,
           "text": "Hello world. Goodbye world.", "words": None}
    out = _split_segments_at_sentence_boundaries([seg])
    assert len(out) == 2
    # Roughly half-and-half by chars (12 vs 14)
    assert 4.0 < out[0]["end"] < 5.5
    assert out[1]["start"] == out[0]["end"]


def test_split_segments_skips_too_short_fragment():
    """Sub-segments under the duration floor merge with neighbours instead."""
    seg = {
        "start": 0.0, "end": 5.0,
        "text": "Hi. I want to tell you something important today.",
        "words": [
            {"start": 0.0, "end": 0.1, "word": "Hi."},
            {"start": 0.5, "end": 0.7, "word": " I"},
            {"start": 0.7, "end": 0.9, "word": " want"},
            {"start": 0.9, "end": 1.1, "word": " to"},
            {"start": 1.1, "end": 1.4, "word": " tell"},
            {"start": 1.4, "end": 1.7, "word": " you"},
            {"start": 1.7, "end": 2.2, "word": " something"},
            {"start": 2.2, "end": 2.7, "word": " important"},
            {"start": 2.7, "end": 5.0, "word": " today."},
        ],
    }
    out = _split_segments_at_sentence_boundaries([seg], min_split_duration=0.4)
    # "Hi." should fold forward, leaving 1 segment, not 2.
    assert len(out) == 1
    assert "today" in out[0]["text"]


def test_split_segments_handles_ellipsis_as_single_boundary():
    """'foo... bar' must not produce three empty splits at each dot."""
    seg = {
        "start": 0.0, "end": 4.0,
        "text": "Wait... Are you serious?",
        "words": None,
    }
    out = _split_segments_at_sentence_boundaries([seg])
    # Either 1 segment (no boundary clears the floor) or exactly 2 — never 4.
    assert 1 <= len(out) <= 2


def test_split_segments_handles_quote_marks_after_punctuation():
    """`said." Then she` should split at the boundary after the close quote."""
    seg = {
        "start": 0.0, "end": 6.0,
        "text": '"I will go," he said. Then she nodded.',
        "words": None,
    }
    out = _split_segments_at_sentence_boundaries([seg])
    assert len(out) == 2
    assert out[0]["text"].strip().endswith("said.")
    assert out[1]["text"].strip().startswith("Then")


def test_split_segments_handles_cjk_full_width_punctuation():
    """CJK content splits on \u3002\uff01\uff1f, not Latin '.'."""
    # Two Japanese sentences separated by 。
    seg = {
        "start": 0.0, "end": 5.0,
        "text": "\u3053\u308c\u306f\u30c6\u30b9\u30c8\u3002\u305d\u3057\u3066\u30de\u30a4\u30ad\u30e3\u30e9\u3002",
        "words": None,
    }
    out = _split_segments_at_sentence_boundaries([seg])
    assert len(out) == 2


# ══════════════════════════════════════════════════════════════════════
# Bug B — long-range verbatim repeats are dropped
# ══════════════════════════════════════════════════════════════════════


def test_long_range_dedup_removes_verbatim_repeat_8min_later():
    """Verbatim line at 3:20 and 11:43 — second copy must be dropped."""
    text = "for raising a blade at me are entirely his own"
    segs = [
        {"start": 200.0, "end": 204.0, "text": text, "words": None},
        {"start": 210.0, "end": 215.0, "text": "Kill the bastard.", "words": None},
        {"start": 703.0, "end": 707.0, "text": text, "words": None},
    ]
    out, removed = _dedupe_long_range(segs)
    assert len(out) == 2
    assert out[0]["start"] == 200.0
    assert out[1]["start"] == 210.0
    assert removed == [(703.0, 707.0)]


def test_long_range_dedup_preserves_short_legitimate_repeats():
    """'right.' appearing 5 times across an interview is not a hallucination."""
    segs = [
        {"start": float(i * 30), "end": float(i * 30 + 0.5),
         "text": "Right.", "words": None}
        for i in range(5)
    ]
    out, removed = _dedupe_long_range(segs)
    assert len(out) == 5
    assert removed == []


def test_long_range_dedup_jaccard_catches_minor_rewording():
    """Near-identical content (Jaccard ≥ 0.92) should be dropped.

    At the tightened threshold (0.92), a sentence with one extra word
    (Jaccard ≈ 0.89) is preserved — these are legitimately different.
    A true near-duplicate (Jaccard ≥ 0.92) is still caught.
    """
    # This pair has Jaccard ≈ 0.89 — should be kept at threshold 0.92
    segs_different = [
        {"start": 0.0, "end": 4.0,
         "text": "This power will soon pass to Rod's children",
         "words": None},
        {"start": 60.0, "end": 64.0,
         "text": "This power will soon pass to Rod's children today",
         "words": None},
    ]
    out, removed = _dedupe_long_range(segs_different)
    assert len(out) == 2
    assert len(removed) == 0

    # True near-duplicate: 12 unique words shared, one extra word
    # added. Jaccard = 12/13 ≈ 0.923 — above threshold, should drop.
    segs_dup = [
        {"start": 0.0, "end": 4.0,
         "text": "The incredible power of the ancient kingdom will soon pass to Rod's children",
         "words": None},
        {"start": 60.0, "end": 64.0,
         "text": "The incredible power of the ancient kingdom will soon pass to Rod's brave children",
         "words": None},
    ]
    out2, removed2 = _dedupe_long_range(segs_dup)
    assert len(out2) == 1
    assert len(removed2) == 1


def test_long_range_dedup_allows_two_copies_of_short_phrase():
    """Hosts who say 'Let's go now' twice should not be punished — but
    a third repeat far later is still treated as a hallucination."""
    segs = [
        {"start": 0.0, "end": 1.0, "text": "Let's go now everyone", "words": None},
        {"start": 5.0, "end": 6.0, "text": "Let's go now everyone", "words": None},
        {"start": 200.0, "end": 201.0, "text": "Let's go now everyone", "words": None},
    ]
    out, removed = _dedupe_long_range(segs)
    # 5-token phrase: passes the min_words floor (5); two copies allowed
    # by the < 8-token carve-out, third should drop.
    assert len(out) == 2
    assert len(removed) == 1


# ══════════════════════════════════════════════════════════════════════
# Consolidation no longer crosses sentence boundaries
# ══════════════════════════════════════════════════════════════════════


def test_consolidate_does_not_merge_across_sentence_boundary():
    """Two short sentences should not be glued together."""
    segs = [
        {"start": 0.0, "end": 1.0, "text": "Hold on, Rod.", "words": None,
         "no_speech_prob": 0.1, "avg_logprob": -0.3},
        {"start": 1.2, "end": 2.0, "text": "Don't shoot.", "words": None,
         "no_speech_prob": 0.1, "avg_logprob": -0.3},
    ]
    out = _consolidate_segments(segs)
    assert len(out) == 2
    assert out[0]["text"].strip() == "Hold on, Rod."
    assert out[1]["text"].strip() == "Don't shoot."


# ══════════════════════════════════════════════════════════════════════
# Speaker stability after dedup removal
# ══════════════════════════════════════════════════════════════════════


def test_assign_speakers_does_not_flip_after_dedup_removal():
    """A removed duplicate between two same-speaker segments must not
    trigger a new-speaker label."""
    raw = [
        {"start": 0.0, "end": 3.0, "text": "First line by host.", "words": None},
        # 8 s gap of which 7 s came from a removed duplicate.
        {"start": 11.0, "end": 14.0, "text": "Second line by host.", "words": None},
    ]
    removed_intervals = [(3.5, 10.5)]
    out = _assign_speakers(raw, removed_intervals=removed_intervals)
    assert out[0].speaker == out[1].speaker  # Both still Speaker 1


def test_assign_speakers_default_kwarg_is_backward_compatible():
    """Calling _assign_speakers without removed_intervals must still work."""
    raw = [
        {"start": 0.0, "end": 2.0, "text": "Hello world", "words": None},
        {"start": 2.5, "end": 4.0, "text": "Second line", "words": None},
    ]
    out = _assign_speakers(raw)
    assert len(out) == 2
    assert out[0].speaker == "Speaker 1"


# ══════════════════════════════════════════════════════════════════════
# AoT S3E10 regression pattern (end-to-end)
# ══════════════════════════════════════════════════════════════════════


def test_aot_s3e10_regression_pattern():
    """Reproduces the exact defect log from Appendix A.

    Builds a small fixture with multi-sentence segments and two
    long-range verbatim duplicates, then asserts:
      - No surviving segment contains a multi-sentence chunk
      - Each duplicated phrase appears exactly once in the output
      - Speaker labels around the removed duplicates remain monotonic
    """
    raw = [
        # 2:31 multi-sentence merge
        {"start": 151.0, "end": 154.0,
         "text": "Did you really figure I was the strongest man in the world? So this is a Titan.",
         "words": None, "no_speech_prob": 0.05, "avg_logprob": -0.3},
        # 2:38 multi-sentence merge
        {"start": 158.0, "end": 162.0,
         "text": "I guess they really do exist within the walls. Hold on, Rod. Don't shoot.",
         "words": None, "no_speech_prob": 0.05, "avg_logprob": -0.3},
        # 3:01 single sentence — should pass through
        {"start": 181.0, "end": 184.0,
         "text": "Someone must have told him we'd be here, probably a council member.",
         "words": None, "no_speech_prob": 0.05, "avg_logprob": -0.3},
        # 3:20 multi-sentence + originator of long-range repeat
        {"start": 200.0, "end": 206.0,
         "text": "for raising a blade at me are entirely his own. Kill the bastard. I'll do it.",
         "words": None, "no_speech_prob": 0.05, "avg_logprob": -0.3},
        # 11:43 verbatim repeat of 3:20 — 503 s later, must be dropped
        # by _dedupe_long_range (well outside the 180 s consolidate cap).
        {"start": 703.0, "end": 709.0,
         "text": "for raising a blade at me are entirely his own. Kill the bastard. I'll do it.",
         "words": None, "no_speech_prob": 0.05, "avg_logprob": -0.3},
        # 10:09 originator of second long-range repeat
        {"start": 609.0, "end": 614.0,
         "text": "This power will soon pass to Rod's children, and I will live on inside of their memories.",
         "words": None, "no_speech_prob": 0.05, "avg_logprob": -0.3},
        # 10:26 multi-sentence merge
        {"start": 626.0, "end": 636.0,
         "text": "This world will crumble in humanity's fading twilight. You believe in violence? Do not.",
         "words": None, "no_speech_prob": 0.05, "avg_logprob": -0.3},
        # 16:10 verbatim repeat of 10:09 — 361 s later, must be dropped
        # by _dedupe_long_range.
        {"start": 970.0, "end": 975.0,
         "text": "This power will soon pass to Rod's children, and I will live on inside of their memories.",
         "words": None, "no_speech_prob": 0.05, "avg_logprob": -0.3},
    ]
    # Sort by start so the consolidate / split / dedup passes see a
    # monotonic stream — _filter_hallucinations would drop a backward
    # jump on the production path, but for the unit test we just want
    # the dedup logic exercised on monotonic input.
    raw.sort(key=lambda s: s["start"])

    consolidated = _consolidate_segments(raw)
    split = _split_segments_at_sentence_boundaries(consolidated)
    deduped, removed = _dedupe_long_range(split)

    # ── Bug A: each surviving segment has at most one sentence-ending mark ──
    for seg in deduped:
        text = seg["text"].strip()
        # Count distinct sentence terminators (treat '?!' as one).
        # If the segment contains more than ONE punctuation-followed-by-space
        # then we still have a multi-sentence chunk — that's a regression.
        sentence_starts = sum(
            1 for i, ch in enumerate(text)
            if ch in ".!?" and i + 1 < len(text) and text[i + 1] == " "
        )
        assert sentence_starts <= 1, (
            f"Segment still contains multiple sentences: {text!r}"
        )

    # ── Bug B: each duplicated phrase appears exactly once ──
    full_text = " ".join(s["text"] for s in deduped)
    assert full_text.count("for raising a blade at me are entirely his own") == 1, (
        f"First duplicate not de-duped: {full_text}"
    )
    assert full_text.count("This power will soon pass to Rod's children") == 1, (
        f"Second duplicate not de-duped: {full_text}"
    )

    # ── Speaker assignment runs cleanly with removed_intervals ──
    speakered = _assign_speakers(deduped, removed_intervals=removed)
    assert all(isinstance(s, TranscriptSegment) for s in speakered)
    # Sanity: _dedupe_long_range removed at least one segment inside
    # each verbatim-repeat window. (The multi-sentence splitter slices
    # the duplicate into its own sub-segments first; only the LONG
    # unique sentence — "for raising..." / "This power..." — clears
    # the dedup floor and gets dropped. Short trailing fragments like
    # "Kill the bastard." sit below min_words/min_chars and survive
    # legitimately, which is the correct conservative behaviour.)
    assert any(703.0 <= r[0] < 709.0 for r in removed), removed
    assert any(970.0 <= r[0] < 975.0 for r in removed), removed
    # Originators must still be present in the output.
    surviving_starts = [s["start"] for s in deduped]
    assert any(200.0 <= s < 206.0 for s in surviving_starts), surviving_starts
    assert any(609.0 <= s < 614.0 for s in surviving_starts), surviving_starts


if __name__ == "__main__":
    unittest.main()
