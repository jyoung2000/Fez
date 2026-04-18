"""Tests for summary placeholder detection and transcript-based fallback.

Covers:
  - _is_placeholder detection of various placeholder strings
  - has_real_summary_content validation of parsed summary data
  - build_summary_from_transcript deterministic fallback generation
  - build_fallback_summary handling of LLM responses with placeholder values
  - extract_json + validation integration
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.models import TranscriptSegment, SceneDescription, VideoSummary
from backend.services.providers.base import (
    _is_placeholder,
    has_real_summary_content,
    build_summary_from_transcript,
    build_fallback_summary,
    extract_json,
)


class TestIsPlaceholder(unittest.TestCase):
    """Test _is_placeholder detects common placeholder patterns."""

    def test_ellipsis_variants(self):
        self.assertTrue(_is_placeholder("..."))
        self.assertTrue(_is_placeholder("…"))
        self.assertTrue(_is_placeholder(".."))
        self.assertTrue(_is_placeholder("....."))

    def test_angle_bracket_placeholders(self):
        self.assertTrue(_is_placeholder("<paragraph>"))
        self.assertTrue(_is_placeholder("<topic1>"))
        self.assertTrue(_is_placeholder("<tone>"))
        self.assertTrue(_is_placeholder("<audience>"))
        self.assertTrue(_is_placeholder("<category>"))

    def test_empty_and_short_strings(self):
        self.assertTrue(_is_placeholder(""))
        self.assertTrue(_is_placeholder("  "))
        self.assertTrue(_is_placeholder("ab"))
        self.assertTrue(_is_placeholder("n/a"))

    def test_special_words(self):
        self.assertTrue(_is_placeholder("null"))
        self.assertTrue(_is_placeholder("undefined"))
        self.assertTrue(_is_placeholder("placeholder"))
        self.assertTrue(_is_placeholder("none"))
        self.assertTrue(_is_placeholder("N/A"))

    def test_real_content_not_placeholder(self):
        self.assertFalse(_is_placeholder("This is a real sentence about a video"))
        self.assertFalse(_is_placeholder("funny and casual"))
        self.assertFalse(_is_placeholder("fast food taste test"))
        self.assertFalse(_is_placeholder("foodies and fast food fans"))


class TestHasRealSummaryContent(unittest.TestCase):
    """Test has_real_summary_content validates summary dicts."""

    def test_good_summary_passes(self):
        data = {
            "overview": "Two friends taste-test fast food burgers and get into a heated debate.",
            "key_topics": ["fast food taste test", "In-N-Out vs Five Guys", "sauce disaster"],
            "tone": "funny and casual",
            "estimated_audience": "foodies",
            "content_category": "food review",
        }
        self.assertTrue(has_real_summary_content(data))

    def test_placeholder_overview_fails(self):
        data = {
            "overview": "...",
            "key_topics": ["topic1", "topic2"],
            "tone": "funny",
            "estimated_audience": "general",
            "content_category": "entertainment",
        }
        self.assertFalse(has_real_summary_content(data))

    def test_angle_bracket_overview_fails(self):
        data = {
            "overview": "<paragraph>",
            "key_topics": ["<topic1>", "<topic2>"],
            "tone": "<tone>",
            "estimated_audience": "<audience>",
            "content_category": "<category>",
        }
        self.assertFalse(has_real_summary_content(data))

    def test_empty_key_topics_fails(self):
        data = {
            "overview": "This is a real overview about a cooking video with great content.",
            "key_topics": [],
            "tone": "educational",
            "estimated_audience": "home cooks",
            "content_category": "cooking tutorial",
        }
        self.assertFalse(has_real_summary_content(data))

    def test_placeholder_key_topics_fails(self):
        data = {
            "overview": "This is a real overview about a cooking video with great content.",
            "key_topics": ["...", "<topic1>"],
            "tone": "educational",
            "estimated_audience": "home cooks",
            "content_category": "cooking tutorial",
        }
        self.assertFalse(has_real_summary_content(data))

    def test_short_overview_fails(self):
        data = {
            "overview": "Short text.",
            "key_topics": ["topic"],
            "tone": "casual",
            "estimated_audience": "general",
            "content_category": "video",
        }
        self.assertFalse(has_real_summary_content(data))

    def test_missing_overview_fails(self):
        data = {
            "key_topics": ["topic1"],
            "tone": "casual",
        }
        self.assertFalse(has_real_summary_content(data))


class TestBuildSummaryFromTranscript(unittest.TestCase):
    """Test the deterministic transcript-based fallback."""

    def _make_segments(self, texts, speaker="Speaker 1"):
        segments = []
        t = 0.0
        for text in texts:
            segments.append(TranscriptSegment(
                start=t, end=t + 5.0, text=text, speaker=speaker,
            ))
            t += 5.0
        return segments

    def _make_scenes(self, descs, scores=None):
        scenes = []
        for i, desc in enumerate(descs):
            score = scores[i] if scores else 5
            scenes.append(SceneDescription(
                timestamp=float(i * 10),
                description=desc,
                importance_score=score,
                thumbnail_path=f"/tmp/frame_{i}.jpg",
            ))
        return scenes

    def test_basic_summary_from_transcript(self):
        segments = self._make_segments([
            "Welcome to the show today we're going to talk about cooking.",
            "First let's look at the ingredients we need for this recipe.",
            "You'll need flour, eggs, butter, and some fresh herbs.",
        ])
        scenes = self._make_scenes([
            "Host standing in a kitchen with ingredients on counter",
            "Close-up of fresh ingredients being prepared",
        ], scores=[7, 6])

        result = build_summary_from_transcript(segments, scenes)

        self.assertIn("overview", result)
        self.assertIn("key_topics", result)
        self.assertIn("tone", result)
        self.assertIn("estimated_audience", result)
        self.assertIn("content_category", result)

        # Overview should be non-trivial
        self.assertGreater(len(result["overview"]), 20)
        self.assertNotIn("...", result["overview"])

        # Key topics should have real content
        self.assertGreater(len(result["key_topics"]), 0)
        for topic in result["key_topics"]:
            self.assertFalse(_is_placeholder(topic))

        # All fields should be non-placeholder
        self.assertFalse(_is_placeholder(result["tone"]))
        self.assertFalse(_is_placeholder(result["estimated_audience"]))
        self.assertFalse(_is_placeholder(result["content_category"]))

    def test_summary_with_multiple_speakers(self):
        segments = [
            TranscriptSegment(start=0, end=5, text="Hey what do you think about this new phone?", speaker="Speaker 1"),
            TranscriptSegment(start=5, end=10, text="I think it's amazing, the camera is incredible", speaker="Speaker 2"),
            TranscriptSegment(start=10, end=15, text="Let me show you some photos I took with it", speaker="Speaker 1"),
        ]
        scenes = self._make_scenes(["Two people examining a phone"], scores=[6])

        result = build_summary_from_transcript(segments, scenes)
        self.assertIn("2 speakers", result["overview"])

    def test_summary_with_empty_transcript(self):
        result = build_summary_from_transcript([], [])
        self.assertIn("overview", result)
        self.assertGreater(len(result["overview"]), 10)
        self.assertGreater(len(result["key_topics"]), 0)

    def test_result_creates_valid_video_summary(self):
        """Ensure the dict can be used to create a valid VideoSummary."""
        segments = self._make_segments(["This is a test video about software development."])
        scenes = self._make_scenes(["Screen showing code editor"], scores=[5])

        result = build_summary_from_transcript(segments, scenes)
        summary = VideoSummary(**result)

        self.assertIsInstance(summary, VideoSummary)
        self.assertGreater(len(summary.overview), 0)
        self.assertIsInstance(summary.key_topics, list)

    def test_topics_from_high_importance_scenes(self):
        """High-importance scenes should be used preferentially for topics."""
        segments = self._make_segments(["Just talking"])
        scenes = self._make_scenes([
            "Boring establishing shot of building exterior",
            "Dramatic reveal of the new product on stage with audience reaction",
            "Speaker standing at podium giving introduction",
        ], scores=[3, 9, 5])

        result = build_summary_from_transcript(segments, scenes)
        # The high-importance scene should be first in topics
        self.assertTrue(any("product" in t.lower() or "reveal" in t.lower() or "dramatic" in t.lower()
                           for t in result["key_topics"]))


class TestBuildFallbackSummaryWithPlaceholders(unittest.TestCase):
    """Test that build_fallback_summary handles LLM responses with placeholders."""

    def test_raw_json_with_ellipsis_values(self):
        raw = '{"overview": "...", "key_topics": [], "tone": "...", "estimated_audience": "...", "content_category": "..."}'
        result = build_fallback_summary(raw)
        # The function extracts fields from partial JSON - it should get "..." values
        # The has_real_summary_content check happens in the provider
        self.assertIn("overview", result)

    def test_empty_raw_gives_default(self):
        result = build_fallback_summary("")
        self.assertEqual(result["overview"], "Could not generate a summary for this video.")

    def test_good_json_gives_good_result(self):
        raw = '{"overview": "A fun cooking video where the chef makes pasta.", "key_topics": ["pasta", "cooking tips"], "tone": "casual", "estimated_audience": "home cooks", "content_category": "cooking"}'
        result = build_fallback_summary(raw)
        # build_fallback_summary tries to extract overview from raw
        self.assertIn("overview", result)


class TestExtractJsonWithValidation(unittest.TestCase):
    """Integration test: extract_json + has_real_summary_content."""

    def test_good_json_passes_validation(self):
        raw = '{"overview": "A great video about technology trends in 2024.", "key_topics": ["AI", "robotics", "sustainability"], "tone": "informative", "estimated_audience": "tech enthusiasts", "content_category": "technology"}'
        data = extract_json(raw)
        self.assertTrue(has_real_summary_content(data))

    def test_placeholder_json_fails_validation(self):
        raw = '{"overview": "...", "key_topics": ["...", "..."], "tone": "...", "estimated_audience": "...", "content_category": "..."}'
        data = extract_json(raw)
        self.assertFalse(has_real_summary_content(data))

    def test_angle_bracket_json_fails_validation(self):
        raw = '{"overview": "<paragraph>", "key_topics": ["<topic1>", "<topic2>"], "tone": "<tone>", "estimated_audience": "<audience>", "content_category": "<category>"}'
        data = extract_json(raw)
        self.assertFalse(has_real_summary_content(data))

    def test_json_in_markdown_code_block(self):
        raw = '```json\n{"overview": "A video about dogs playing in the park with their owners and having fun.", "key_topics": ["dogs", "outdoor activities", "pet care"], "tone": "wholesome", "estimated_audience": "pet owners", "content_category": "pets"}\n```'
        data = extract_json(raw)
        self.assertTrue(has_real_summary_content(data))

    def test_json_with_thinking_tags(self):
        raw = '<think>Let me analyze this video...</think>\n{"overview": "Friends compete in a cooking challenge that gets hilariously out of control.", "key_topics": ["cooking challenge", "competition", "funny moments"], "tone": "chaotic energy", "estimated_audience": "food lovers", "content_category": "entertainment"}'
        data = extract_json(raw)
        self.assertTrue(has_real_summary_content(data))


if __name__ == "__main__":
    unittest.main()
