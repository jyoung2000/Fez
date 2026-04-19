import pytest
from backend.services.scene_description_validator import (
    sanitize_description, is_valid_description, fallback_description_for_scene,
)


def test_rejects_dict_repr():
    ok, reason = is_valid_description("{'raw_response': 'hello'}")
    assert not ok and reason == "junk_prefix"


def test_rejects_json_leak():
    ok, _ = is_valid_description('{"description": "x"}')
    assert not ok


def test_rejects_error_string():
    assert is_valid_description("Error: connection refused")[0] is False
    assert is_valid_description("Traceback (most recent call last):")[0] is False
    assert is_valid_description("Failed to analyze")[0] is False


def test_rejects_refusal():
    assert is_valid_description("I cannot analyze this image.")[0] is False
    assert is_valid_description("I'm unable to see the image.")[0] is False
    assert is_valid_description("As an AI language model, I ...")[0] is False
    assert is_valid_description("[image unavailable]")[0] is False


def test_accepts_real_description():
    ok, _ = is_valid_description("Two people sitting at a desk, one gesturing with their hands as they speak.")
    assert ok


def test_accepts_synthetic_markers():
    ok, reason = is_valid_description("Key moment 5: the speaker introduces the topic")
    assert ok and reason == "synthetic_ok"
    ok, _ = is_valid_description("Key moment 3 at 42.1s")
    assert ok


def test_sanitize_strips_code_fences():
    assert sanitize_description("```json\n{\"foo\": 1}\n```") == ""
    assert sanitize_description("```\nA dog on grass.\n```") == "A dog on grass."


def test_sanitize_collapses_whitespace_and_control_chars():
    assert sanitize_description("A\x00dog\n\n\nruns") == "A dog runs"


def test_sanitize_strips_description_prefix():
    assert sanitize_description("Description: A room with furniture.") == "A room with furniture."


def test_sanitize_truncates_long():
    long = "word " * 200
    out = sanitize_description(long)
    assert len(out) <= 400


def test_sanitize_then_validate_catches_junk_after_cleaning():
    cleaned = sanitize_description('{"description": "hi"}')
    ok, _ = is_valid_description(cleaned)
    assert not ok


def test_fallback_uses_transcript_when_nearby():
    class Seg:
        def __init__(self, start, end, text):
            self.start, self.end, self.text = start, end, text
    segs = [Seg(10, 12, "Hello everyone"), Seg(12.5, 14, "welcome back")]
    out = fallback_description_for_scene(timestamp=11.5, transcript_segments=segs, scene_index=3)
    assert out.startswith("Key moment 3: ")
    assert "Hello everyone" in out


def test_fallback_timestamp_only_when_no_transcript():
    out = fallback_description_for_scene(timestamp=42.1, transcript_segments=None, scene_index=2)
    assert out == "Key moment 2 at 42.1s"


def test_high_symbol_ratio_rejected():
    ok, _ = is_valid_description("{}{}[][]:\":\":")
    assert not ok
