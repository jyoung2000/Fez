"""Tests guaranteeing reframing works with local AI / no-AI modes.

Contracts this file defends:

  1. The default ``critic_mode`` is ``learned`` so every clip gets an
     aesthetic score without any cloud call.
  2. With ``critic_mode=vlm`` and ``critic_vlm_backend=ollama``, no
     network call to OpenRouter is ever attempted. If Ollama isn't
     reachable, the critic silently falls back to the local heuristic.
  3. ``aesthetic_scorer.score_frame`` always returns a finite value in
     ``[0, 1]`` even when the frame file doesn't exist, cv2 isn't
     installed, or CLIP weights are missing.
  4. ``composition_head`` and ``cut_timing_head`` fall back to
     deterministic heuristics when no trained weights exist.
  5. End-to-end: with every network-calling provider patched out, the
     human-reframe bridge still produces a valid, coverage-clean plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest

from backend.services import (
    aesthetic_scorer,
    composition_head,
    critic_loop,
    cut_timing_head,
    human_reframe,
    human_reframe_bridge,
    human_render_plan_adapter,
    reframe_config,
)


# ── Synthetic inputs ─────────────────────────────────────────────


@dataclass
class _Face:
    identity_id: int = 0
    is_human: bool = True
    nose_x: float = 50.0
    nose_y: float = 40.0
    width: float = 15.0
    height: float = 20.0
    lip_aperture: float = 0.0
    yaw: float = 0.0


@dataclass
class _FrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)


@dataclass
class _SpeakerEvent:
    start: float
    end: float
    slot_id: int
    on_screen: bool = True


def _dense(duration: float = 4.0, fps: float = 6.0):
    frames = []
    n = int(duration * fps)
    for i in range(n):
        frames.append(_FrameFaces(timestamp=i / fps, faces=[_Face()]))
    return frames


# ── Default mode is local ────────────────────────────────────────


def test_default_config_is_local_only():
    cfg = reframe_config.load_default_config()
    assert cfg.human_reframe_enabled is True
    assert cfg.critic_mode == "learned"
    assert cfg.critic_vlm_backend == "auto"


def test_env_can_turn_off_human_reframe(monkeypatch):
    monkeypatch.setenv("CLIPAI_HUMAN_REFRAME", "0")
    assert reframe_config.load_default_config().human_reframe_enabled is False


def test_env_can_force_local_vlm(monkeypatch):
    monkeypatch.setenv("CLIPAI_CRITIC_VLM_BACKEND", "ollama")
    assert reframe_config.load_default_config().critic_vlm_backend == "ollama"


# ── Critic loop: no network on ``learned`` default ──────────────


def test_learned_critic_makes_no_network_calls(monkeypatch, tmp_path):
    """The default critic mode must never import / touch any network
    provider. We detect this by monkeypatching urllib to raise.
    """
    import urllib.request

    def _block(*_a, **_kw):
        raise AssertionError("critic attempted a network call in learned mode")
    monkeypatch.setattr(urllib.request, "urlopen", _block)

    # Fake frame file so heuristic path has something to read. Empty
    # file is fine — the heuristic returns a neutral score on decode
    # failure.
    frame = tmp_path / "f.png"
    frame.write_bytes(b"")

    cfg = reframe_config.load_default_config().override(critic_mode="learned")
    rep = critic_loop.score_plan(
        samples=[critic_loop.FrameSample(t=0.0, frame_path=str(frame))],
        config=cfg,
    )
    assert rep.mode == "learned"
    assert 0.0 <= rep.mean_score <= 10.0
    # Learned mode must never use the VLM budget.
    assert rep.budget_used == 0


def test_vlm_ollama_fallback_to_learned_when_unreachable(monkeypatch, tmp_path):
    """When Ollama isn't reachable AND backend is ``ollama`` (forced
    local), the critic must fall through to the learned heuristic
    rather than trying OpenRouter."""
    # Force the reachability check to say no.
    monkeypatch.setattr(critic_loop, "_ollama_reachable", lambda: False)

    # If the code ever touched OpenRouter we'd know: block that import
    # with a stub that raises on attribute access.
    import sys, types
    fake = types.ModuleType("backend.services.providers.openrouter_provider")
    def _blow(*_a, **_kw):
        raise AssertionError("ollama-forced critic reached openrouter")
    fake.analyze_frames = _blow
    monkeypatch.setitem(
        sys.modules,
        "backend.services.providers.openrouter_provider",
        fake,
    )

    frame = tmp_path / "f.png"
    frame.write_bytes(b"")
    cfg = reframe_config.load_default_config().override(
        critic_mode="vlm", critic_vlm_backend="ollama",
    )
    rep = critic_loop.score_plan(
        samples=[critic_loop.FrameSample(t=0.0, frame_path=str(frame))],
        config=cfg,
    )
    assert rep.mode == "vlm"
    # Score came from the learned fallback so it's a valid 0-10.
    assert 0.0 <= rep.mean_score <= 10.0


def test_aesthetic_scorer_always_returns_valid(tmp_path):
    # Nonexistent file.
    s = aesthetic_scorer.score_frame(None)
    assert 0.0 <= s <= 1.0
    s = aesthetic_scorer.score_frame("/does/not/exist.png")
    assert 0.0 <= s <= 1.0
    # Zero-byte PNG: cv2 will fail to decode but heuristic must still
    # return a neutral value.
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"")
    s = aesthetic_scorer.score_frame(str(bad))
    assert 0.0 <= s <= 1.0


# ── Head fallbacks are deterministic ────────────────────────────


def test_composition_head_fallback_deterministic():
    feat = composition_head.CompositionFeatures(
        face_cx=0.5, face_cy=0.4,
        face_w=0.12, face_h=0.20,
        face_yaw=-0.5,
        content_type="talking_head",
        speaker_dwell_sec=3.5,
    )
    a = composition_head.predict_composition(feat)
    b = composition_head.predict_composition(feat)
    assert a == b
    # Lead-room: negative yaw (looking left) should push crop center
    # to the right of face.
    assert a[0] >= 0.5


def test_cut_timing_head_fallback_j_cuts_dialogue():
    feat = cut_timing_head.CutFeatures(
        content_family="dialogue",
        trigger_kind="speaker_turn",
    )
    off, conf = cut_timing_head.predict_cut_offset(feat)
    assert off < 0  # J-cut lead
    assert 0.0 <= conf <= 1.0


# ── End-to-end: full pipeline with no network ──────────────────


def test_end_to_end_no_network_still_produces_valid_plan(monkeypatch):
    """Human-reframe must produce a coverage-clean plan even when every
    network-capable provider is dead.
    """
    import urllib.request

    def _block(*_a, **_kw):
        raise AssertionError("pipeline attempted a network call")
    monkeypatch.setattr(urllib.request, "urlopen", _block)

    cfg = reframe_config.load_default_config().override(critic_mode="learned")
    dense = _dense(duration=5.0)
    events = [_SpeakerEvent(0.0, 5.0, slot_id=0)]

    inputs = human_reframe.HumanReframeInputs(
        duration_sec=5.0,
        source_w=1920, source_h=1080,
        content_type="talking_head",
        dense_faces=dense,
        active_speaker_events=events,
    )
    plan = human_reframe.run_human_reframe(inputs, config=cfg)
    rp = human_render_plan_adapter.render_plan_from_human_plan(
        plan,
        source_width=1920, source_height=1080, source_fps=30.0,
        config=cfg, content_type="talking_head",
    )
    cov = human_render_plan_adapter.verify_frame_coverage(rp)
    assert cov.ok


def test_bridge_hook_works_with_no_network(monkeypatch):
    from backend.services.render_plan import RenderPlan, RenderOp, Rect, RenderOpKind

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **kw: (_ for _ in ()).throw(
                            AssertionError("no network allowed")))

    legacy = RenderPlan(
        source_width=1920, source_height=1080,
        target_width=1080, target_height=1920,
        total_duration_sec=5.0, fps=30.0,
        ops=[RenderOp(
            kind=RenderOpKind.CROP,
            start_sec=0.0, end_sec=5.0,
            primary_rect=Rect(x=0.25, y=0.0, w=0.5, h=1.0),
        )],
    )
    dense = _dense(duration=5.0)
    result = human_reframe_bridge.maybe_override_render_plan(
        legacy,
        dense_faces=dense,
        active_speaker_events=[_SpeakerEvent(0.0, 5.0, slot_id=0)],
        shot_boundaries=[],
        content_type="talking_head",
        duration_sec=5.0,
        source_width=1920, source_height=1080,
        source_fps=30.0,
    )
    # Human-reframe path produced a valid plan without any network.
    assert result is not legacy
    cov = human_render_plan_adapter.verify_frame_coverage(result)
    assert cov.ok


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
