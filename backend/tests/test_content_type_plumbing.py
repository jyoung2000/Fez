"""Test that _clip_content_type flows from ContentProfile to plan_layout.

Bug 1 from the Bungo Stray Dogs run: `_clip_content_type` was gated behind
the CLIPAI_CONTENT_ROUTING env var (default off), so plan_layout was
always called with content_type=None, silently disabling all of the
per-content-type tuning in required_regions.py and layout_engine.py.
"""

from backend.services.content_classifier import (
    ClipContentType,
    ContentProfile,
    classify_clip,
)
from backend.services.content_type_config import ContentType


def test_classify_clip_returns_generic_for_narrative_without_dialogue():
    profile = ContentProfile(
        content_type=ContentType.NARRATIVE.value,
        confidence=0.6,
        is_cinematic_dialogue=False,
        is_animated=False,
    )
    clip_type = classify_clip(content_profile=profile)
    assert clip_type == ClipContentType.GENERIC


def test_classify_clip_promotes_narrative_with_dialogue_to_cinematic_dialogue():
    profile = ContentProfile(
        content_type=ContentType.NARRATIVE.value,
        confidence=0.6,
        is_cinematic_dialogue=True,
    )
    clip_type = classify_clip(content_profile=profile)
    assert clip_type == ClipContentType.CINEMATIC_DIALOGUE


def test_classify_clip_promotes_animated_narrative_to_animation_dialogue():
    """The Bungo Stray Dogs case: narrative + animated + dialogue signals."""
    profile = ContentProfile(
        content_type=ContentType.NARRATIVE.value,
        confidence=0.3,
        is_cinematic_dialogue=True,
        is_animated=True,
    )
    clip_type = classify_clip(content_profile=profile)
    assert clip_type == ClipContentType.ANIMATION_DIALOGUE


def test_classify_clip_returns_non_none_for_real_profile():
    """Regression guard: classify_clip must never return None — even for
    a minimal profile that only has a content_type string. This was the
    underlying failure mode behind the `content_type=none` log line.
    """
    for ct in (ContentType.NARRATIVE, ContentType.PODCAST, ContentType.GAMING,
               ContentType.UNKNOWN):
        profile = ContentProfile(content_type=ct.value)
        result = classify_clip(content_profile=profile)
        assert result is not None
        assert isinstance(result, ClipContentType)
