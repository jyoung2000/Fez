"""Blueprint v2 Phase 2 — layout VLM fallback cache + parsing + budget.

Mocks the orchestrator.vlm_critique method so we can exercise the
cache path, JSON parsing, and budget-cap contract without a real VLM.
"""

import asyncio
import json
import os
import tempfile
from dataclasses import dataclass, field

import pytest

from backend.services.layout_confidence import LayoutCandidate
from backend.services.layout_engine import decide_layout_for_scene
from backend.services.layout_vlm_fallback import (
    VLMLayoutDecision,
    _cache_key,
    _cache_read,
    _cache_write,
    _parse_decision,
    request_layout_from_vlm,
)
from backend.services.reframe_config import get_default_config


# ── _parse_decision ──────────────────────────────────────────────


def test_parse_valid_json():
    raw = '{"layout": "split", "subject": null, "reason": "2 speakers"}'
    d = _parse_decision(raw)
    assert d and d.layout == "split"
    assert d.subject is None
    assert d.reason == "2 speakers"


def test_parse_strips_markdown_fences():
    raw = "```json\n{\"layout\": \"single\", \"subject\": \"left speaker\", \"reason\": \"x\"}\n```"
    d = _parse_decision(raw)
    assert d and d.layout == "single"
    assert d.subject == "left speaker"


def test_parse_extracts_from_prose():
    raw = 'Sure! Here you go: {"layout": "ken_burns", "subject": null, "reason": "landscape"}'
    d = _parse_decision(raw)
    assert d and d.layout == "ken_burns"


def test_parse_rejects_invalid_layout():
    raw = '{"layout": "pokedex", "subject": null, "reason": "x"}'
    assert _parse_decision(raw) is None


def test_parse_rejects_garbage():
    assert _parse_decision("") is None
    assert _parse_decision("not json") is None


def test_parse_uppercase_layout_is_normalized():
    raw = '{"layout": "SPLIT", "subject": null, "reason": "x"}'
    d = _parse_decision(raw)
    assert d and d.layout == "split"


# ── Cache ─────────────────────────────────────────────────────────


def test_cache_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        key = _cache_key("scene_1.00_5.00", "abc123")
        decision = VLMLayoutDecision(
            layout="split", subject=None, reason="x", raw="raw",
        )
        assert _cache_read(key, d) is None
        _cache_write(key, d, decision)
        out = _cache_read(key, d)
        assert out is not None
        assert out.layout == "split"
        assert out.reason == "x"


def test_cache_key_is_stable():
    k1 = _cache_key("scene_1.00_5.00", "abc")
    k2 = _cache_key("scene_1.00_5.00", "abc")
    k3 = _cache_key("scene_1.00_5.00", "def")
    assert k1 == k2
    assert k1 != k3


# ── Integration via decide_layout_for_scene ──────────────────────


@dataclass
class _FakeScene:
    start: float = 0.0
    end: float = 10.0


@dataclass
class _FrameFaces:
    timestamp: float = 0.0
    faces: list = field(default_factory=list)


@dataclass
class _Face:
    identity_id: int = 0


@dataclass
class _SpeakerEvent:
    start: float
    end: float
    slot_id: int
    confidence: float = 1.0


class _StubOrchestrator:
    """Returns a canned VLM JSON response, counts calls."""
    def __init__(self, response: str):
        self._response = response
        self.calls = 0

    async def vlm_critique(self, *, prompt, images, max_tokens):
        self.calls += 1
        return self._response


def _write_tiny_png(path):
    png = bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
        0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
        0x54, 0x08, 0x99, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
        0x00, 0x00, 0x03, 0x00, 0x01, 0x5B, 0x34, 0x3F,
        0x9A, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
        0x44, 0xAE, 0x42, 0x60, 0x82,
    ])
    with open(path, "wb") as f:
        f.write(png)


def _stub_extract_factory(tmpdir):
    """Build an extractor that returns one FrameSample per fake sample."""
    from backend.services.critic_loop import FrameSample

    pngs = []
    for i in range(4):
        p = os.path.join(tmpdir, f"s{i}.png")
        _write_tiny_png(p)
        pngs.append(p)

    def _fake(path, *, duration_sec, interval_sec, out_dir=None):
        # Return 4 timestamps evenly spread across the scene.
        step = max(duration_sec / 4, 0.5)
        return [
            FrameSample(t=i * step, frame_path=pngs[i])
            for i in range(4)
        ]
    return _fake


def test_vlm_disabled_stays_deterministic(monkeypatch):
    """With the master flag off, no VLM call is made."""
    monkeypatch.delenv("CLIPAI_LAYOUT_VLM_ENABLED", raising=False)
    orch = _StubOrchestrator('{"layout": "split", "subject": null, "reason": "x"}')
    scene = _FakeScene(start=0, end=10)
    faces = [
        _FrameFaces(timestamp=float(i), faces=[_Face(0)])
        for i in range(10)
    ]
    layout, reason, result = asyncio.run(decide_layout_for_scene(
        scene=scene,
        dense_faces=faces,
        active_speaker_events=[_SpeakerEvent(0, 10, 0)],
        content_type="talking_head",
        orchestrator=orch,
    ))
    assert orch.calls == 0
    assert layout == "single"
    assert reason.startswith("deterministic:")


def test_vlm_enabled_high_confidence_still_deterministic(monkeypatch):
    """High-confidence scene shouldn't escalate even with flag on."""
    monkeypatch.setenv("CLIPAI_LAYOUT_VLM_ENABLED", "1")
    orch = _StubOrchestrator('{"layout": "split", "subject": null, "reason": "x"}')
    scene = _FakeScene(start=0, end=10)
    faces = [
        _FrameFaces(timestamp=float(i), faces=[_Face(0)])
        for i in range(10)
    ]
    layout, reason, _ = asyncio.run(decide_layout_for_scene(
        scene=scene,
        dense_faces=faces,
        active_speaker_events=[_SpeakerEvent(0, 10, 0)],
        content_type="talking_head",
        orchestrator=orch,
    ))
    assert orch.calls == 0
    assert reason.startswith("deterministic:")
    assert layout == "single"


def test_vlm_enabled_low_confidence_escalates_and_caches(monkeypatch):
    """Low-confidence scene triggers exactly one VLM call; rerun hits cache."""
    monkeypatch.setenv("CLIPAI_LAYOUT_VLM_ENABLED", "1")
    from backend.services import critic_loop as cl_mod
    from backend.services import layout_vlm_fallback as vlm_mod

    with tempfile.TemporaryDirectory() as sample_dir, tempfile.TemporaryDirectory() as cache_dir:
        # Stub the frame extractor so we don't need ffmpeg.
        monkeypatch.setattr(
            cl_mod, "extract_frames_for_critic",
            _stub_extract_factory(sample_dir),
        )
        # Point the cache at a tmp dir.
        monkeypatch.setattr(
            vlm_mod, "_CACHE_DIR_DEFAULT", cache_dir,
        )
        # Force low-confidence scoring via config threshold = 1.0.
        cfg = get_default_config().override(
            layout_confidence_threshold=1.01,  # no scene can clear this
            layout_vlm_cache_dir=cache_dir,
        )
        # Need an existing video_path for os.path.exists check.
        fake_video = os.path.join(sample_dir, "clip.mp4")
        with open(fake_video, "wb") as f:
            f.write(b"stub")

        orch = _StubOrchestrator(
            '{"layout": "pip", "subject": "left speaker", "reason": "canned"}'
        )
        scene = _FakeScene(start=0, end=10)
        faces = [
            _FrameFaces(timestamp=float(i), faces=[_Face(0)])
            for i in range(10)
        ]
        budget = [2]

        # First call — VLM is consulted, budget decrements to 1.
        layout, reason, _ = asyncio.run(decide_layout_for_scene(
            scene=scene,
            dense_faces=faces,
            active_speaker_events=[_SpeakerEvent(0, 10, 0)],
            content_type="talking_head",
            config=cfg,
            video_path=fake_video,
            source_sha="abc123",
            orchestrator=orch,
            budget_remaining=budget,
        ))
        assert layout == "pip"
        assert reason.startswith("vlm:")
        assert orch.calls == 1
        assert budget[0] == 1

        # Second call (same scene) — cache hit, no additional VLM call,
        # budget still decrements on the code path but the VLM isn't hit.
        layout2, reason2, _ = asyncio.run(decide_layout_for_scene(
            scene=scene,
            dense_faces=faces,
            active_speaker_events=[_SpeakerEvent(0, 10, 0)],
            content_type="talking_head",
            config=cfg,
            video_path=fake_video,
            source_sha="abc123",
            orchestrator=orch,
            budget_remaining=budget,
        ))
        assert layout2 == "pip"
        # The stub's ``calls`` counter only increments on a real call.
        assert orch.calls == 1, "cache hit must not invoke the VLM"


def test_vlm_budget_exhausted_stops_further_calls(monkeypatch):
    monkeypatch.setenv("CLIPAI_LAYOUT_VLM_ENABLED", "1")
    from backend.services import critic_loop as cl_mod
    from backend.services import layout_vlm_fallback as vlm_mod

    with tempfile.TemporaryDirectory() as sample_dir, tempfile.TemporaryDirectory() as cache_dir:
        monkeypatch.setattr(
            cl_mod, "extract_frames_for_critic",
            _stub_extract_factory(sample_dir),
        )
        monkeypatch.setattr(vlm_mod, "_CACHE_DIR_DEFAULT", cache_dir)

        cfg = get_default_config().override(
            layout_confidence_threshold=1.01,
            layout_vlm_cache_dir=cache_dir,
        )
        fake_video = os.path.join(sample_dir, "clip.mp4")
        with open(fake_video, "wb") as f:
            f.write(b"stub")
        orch = _StubOrchestrator(
            '{"layout": "pip", "subject": null, "reason": "x"}'
        )
        # Budget already exhausted before we even call.
        budget = [0]
        scene = _FakeScene(start=0, end=10)
        faces = [
            _FrameFaces(timestamp=float(i), faces=[_Face(0)])
            for i in range(10)
        ]
        layout, reason, _ = asyncio.run(decide_layout_for_scene(
            scene=scene,
            dense_faces=faces,
            active_speaker_events=[],
            content_type="talking_head",
            config=cfg,
            video_path=fake_video,
            source_sha="abc123",
            orchestrator=orch,
            budget_remaining=budget,
        ))
        assert orch.calls == 0
        assert reason.startswith("vlm-budget-exhausted:")


def test_vlm_returns_garbage_falls_back_to_top(monkeypatch):
    """If the VLM returns unparseable text, use the deterministic top."""
    monkeypatch.setenv("CLIPAI_LAYOUT_VLM_ENABLED", "1")
    from backend.services import critic_loop as cl_mod
    from backend.services import layout_vlm_fallback as vlm_mod

    with tempfile.TemporaryDirectory() as sample_dir, tempfile.TemporaryDirectory() as cache_dir:
        monkeypatch.setattr(
            cl_mod, "extract_frames_for_critic",
            _stub_extract_factory(sample_dir),
        )
        monkeypatch.setattr(vlm_mod, "_CACHE_DIR_DEFAULT", cache_dir)
        cfg = get_default_config().override(
            layout_confidence_threshold=1.01,
            layout_vlm_cache_dir=cache_dir,
        )
        fake_video = os.path.join(sample_dir, "clip.mp4")
        with open(fake_video, "wb") as f:
            f.write(b"stub")
        orch = _StubOrchestrator("I cannot respond in JSON, sorry!")
        scene = _FakeScene(start=0, end=10)
        faces = [
            _FrameFaces(timestamp=float(i), faces=[_Face(0)])
            for i in range(10)
        ]
        layout, reason, _ = asyncio.run(decide_layout_for_scene(
            scene=scene,
            dense_faces=faces,
            active_speaker_events=[_SpeakerEvent(0, 10, 0)],
            content_type="talking_head",
            config=cfg,
            video_path=fake_video,
            source_sha="abc123",
            orchestrator=orch,
            budget_remaining=[8],
        ))
        # Fell back to deterministic top (single).
        assert layout == "single"
        assert reason.startswith("vlm-failed:")
