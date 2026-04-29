"""Task 6 — apply_visual_verification wires clip_verifier with a flag.

The verifier itself is in clip_verifier.py; the new wrapper in
clip_scoring.py is feature-flagged and exception-safe so it can be
called from any export path without risk of breaking the export.

Run:  pytest tests/qa/test_clip_verifier_wiring.py -v
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest import mock

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _make_clip(viral_score=70, idx=1):
    from backend.models import ClipCandidate
    return ClipCandidate(
        id=idx,
        title="Test clip",
        start_time=0.0,
        end_time=10.0,
        duration=10.0,
        viral_score=viral_score,
        viral_score_reasoning="test",
        clip_type="moment",
        platform="tiktok",
        suggested_caption="test",
        hook_text="test",
        why_this_works="test",
        hook_score=50, flow_score=50, value_score=50, trend_score=50,
    )


def test_verifier_disabled_flag_skips_call(monkeypatch):
    """CLIPAI_USE_VISUAL_CLIP_VERIFIER=0 → no vision API call."""
    monkeypatch.setenv("CLIPAI_USE_VISUAL_CLIP_VERIFIER", "0")
    from backend.services import clip_scoring

    sentinel_provider = mock.AsyncMock()
    clips = [_make_clip()]
    out = asyncio.run(clip_scoring.apply_visual_verification(
        clips, frames=[], vision_provider=sentinel_provider,
    ))
    assert out is clips
    sentinel_provider.analyze_frames.assert_not_called()


def test_verifier_enabled_invokes_underlying_verify(monkeypatch):
    """Flag ON + provider non-None → verify_clips_visually called."""
    monkeypatch.setenv("CLIPAI_USE_VISUAL_CLIP_VERIFIER", "1")
    from backend.services import clip_scoring

    clips = [_make_clip()]
    fake_frames = ["frame_data_placeholder"]
    fake_provider = mock.AsyncMock()

    async def fake_verify(*args, **kwargs):
        return clips
    with mock.patch(
        "backend.services.clip_verifier.verify_clips_visually",
        side_effect=fake_verify,
    ) as mocked:
        asyncio.run(clip_scoring.apply_visual_verification(
            clips, frames=fake_frames, vision_provider=fake_provider,
        ))
    mocked.assert_called_once()
    # max_clips_to_verify default is 8 per the spec.
    assert mocked.call_args.kwargs.get("max_clips_to_verify") == 8


def test_verifier_caps_at_max_clips(monkeypatch):
    """20 clips in → max_clips_to_verify=8 forwarded."""
    monkeypatch.setenv("CLIPAI_USE_VISUAL_CLIP_VERIFIER", "1")
    from backend.services import clip_scoring

    clips = [_make_clip() for _ in range(20)]
    fake_provider = mock.AsyncMock()

    async def fake_verify(passed_clips, *args, **kwargs):
        return passed_clips
    with mock.patch(
        "backend.services.clip_verifier.verify_clips_visually",
        side_effect=fake_verify,
    ) as mocked:
        asyncio.run(clip_scoring.apply_visual_verification(
            clips, frames=[], vision_provider=fake_provider,
            max_clips_to_verify=8,
        ))
    assert mocked.call_args.kwargs.get("max_clips_to_verify") == 8


def test_verifier_handles_provider_failure(monkeypatch):
    """Verifier raising → original clips returned, logged warning."""
    monkeypatch.setenv("CLIPAI_USE_VISUAL_CLIP_VERIFIER", "1")
    from backend.services import clip_scoring

    clips = [_make_clip()]
    with mock.patch(
        "backend.services.clip_verifier.verify_clips_visually",
        side_effect=RuntimeError("synthetic vision provider failure"),
    ):
        out = asyncio.run(clip_scoring.apply_visual_verification(
            clips, frames=[], vision_provider=mock.AsyncMock(),
        ))
    assert out is clips


def test_verifier_no_op_with_empty_clips(monkeypatch):
    monkeypatch.setenv("CLIPAI_USE_VISUAL_CLIP_VERIFIER", "1")
    from backend.services import clip_scoring
    out = asyncio.run(clip_scoring.apply_visual_verification(
        [], frames=[], vision_provider=mock.AsyncMock(),
    ))
    assert out == []


def test_verifier_no_op_with_no_provider(monkeypatch):
    """No provider → return clips unchanged (matches verify_clips_visually
    no-op behavior for missing providers)."""
    monkeypatch.setenv("CLIPAI_USE_VISUAL_CLIP_VERIFIER", "1")
    from backend.services import clip_scoring
    clips = [_make_clip()]
    out = asyncio.run(clip_scoring.apply_visual_verification(
        clips, frames=[], vision_provider=None,
    ))
    assert out is clips


def test_visual_clip_verifier_enabled_default_true(monkeypatch):
    """When the env var isn't set, the flag defaults to ON."""
    monkeypatch.delenv("CLIPAI_USE_VISUAL_CLIP_VERIFIER", raising=False)
    from backend.services.clip_scoring import visual_clip_verifier_enabled
    assert visual_clip_verifier_enabled() is True


def test_retention_predictor_documented():
    """The clip_scoring docstring documents that retention_predictor is
    surfaced via the GET endpoint, not folded into composite_score."""
    from backend.services import clip_scoring
    assert "retention_predictor" in clip_scoring.__doc__
    assert "GET" in clip_scoring.__doc__
