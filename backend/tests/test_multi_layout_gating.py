"""Phase 0 Task 0.1 — multi-layout gating by content type.

Regression test: ``ALLOW_MULTI_LAYOUT`` is no longer a global kill
switch. Instead, ``multi_layout_allowed_for(content_type, config)``
consults the per-content allowlist on ``ReframeConfig``. Env var
``ALLOW_MULTI_LAYOUT`` still overrides per-run.
"""

import os

import pytest

from backend.services.layout_engine import (
    DEFAULT_MULTI_LAYOUT_TYPES,
    multi_layout_allowed_for,
)
from backend.services.reframe_config import ReframeConfig


def test_default_allowlist_matches_blueprint_v2_phase0():
    # Blueprint v2 Phase 0: these are the content types where split /
    # triple / pip / screenshare / gameplay layouts are reachable.
    expected = {
        "podcast", "interview", "multi_speaker_panel",
        "stream", "gameplay", "gameplay_fps", "gameplay_moba",
        "gameplay_tps", "gameplay_racing",
        "tutorial", "screen_share",
    }
    assert DEFAULT_MULTI_LAYOUT_TYPES == expected


def test_multi_layout_allowed_for_podcast_by_default(monkeypatch):
    monkeypatch.delenv("ALLOW_MULTI_LAYOUT", raising=False)
    assert multi_layout_allowed_for("podcast") is True


def test_multi_layout_blocked_for_cinematic_dialogue_by_default(monkeypatch):
    monkeypatch.delenv("ALLOW_MULTI_LAYOUT", raising=False)
    # Cinematic dialogue should never get a split screen — that's the
    # whole reason we moved off the global flag.
    assert multi_layout_allowed_for("cinematic_dialogue") is False


def test_env_var_overrides_allowlist(monkeypatch):
    # Env-var forces ON regardless of allowlist.
    monkeypatch.setenv("ALLOW_MULTI_LAYOUT", "1")
    assert multi_layout_allowed_for("cinematic_dialogue") is True
    # And OFF regardless of allowlist.
    monkeypatch.setenv("ALLOW_MULTI_LAYOUT", "0")
    assert multi_layout_allowed_for("podcast") is False


def test_config_allowlist_is_respected(monkeypatch):
    monkeypatch.delenv("ALLOW_MULTI_LAYOUT", raising=False)
    cfg = ReframeConfig(multi_layout_content_types=frozenset({"podcast"}))
    assert multi_layout_allowed_for("podcast", cfg) is True
    assert multi_layout_allowed_for("gameplay", cfg) is False


def test_none_content_type_returns_false(monkeypatch):
    monkeypatch.delenv("ALLOW_MULTI_LAYOUT", raising=False)
    assert multi_layout_allowed_for(None) is False
    assert multi_layout_allowed_for("") is False


def test_enum_like_content_type_is_unwrapped(monkeypatch):
    """Accept ClipContentType-style enums via ``.value``."""
    monkeypatch.delenv("ALLOW_MULTI_LAYOUT", raising=False)

    class _FakeCT:
        def __init__(self, v):
            self.value = v

    assert multi_layout_allowed_for(_FakeCT("podcast")) is True
    assert multi_layout_allowed_for(_FakeCT("cinematic_dialogue")) is False


@pytest.mark.parametrize("content_type", sorted(DEFAULT_MULTI_LAYOUT_TYPES))
def test_every_default_type_is_allowed(monkeypatch, content_type):
    monkeypatch.delenv("ALLOW_MULTI_LAYOUT", raising=False)
    assert multi_layout_allowed_for(content_type) is True
