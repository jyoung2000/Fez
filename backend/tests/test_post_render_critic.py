"""Phase 0 Task 0.2 — post-render VLM quality gate.

Tests the pure-logic parts of ``post_render_critic.py`` without
calling a real VLM: JSON parsing robustness + severity-based ``ok``
flag. The VLM call itself is exercised via a stub orchestrator.
"""

import asyncio
import os
import tempfile

import pytest

from backend.services.post_render_critic import (
    PostRenderIssue,
    PostRenderReport,
    _parse_response,
    run_post_render_critic,
)


# ── _parse_response ──────────────────────────────────────────────


def test_parse_empty_array_means_no_issues():
    assert _parse_response("[]") == []


def test_parse_single_issue():
    raw = '[{"t": 3.2, "issue": "face off-frame", "severity": "high"}]'
    issues = _parse_response(raw)
    assert len(issues) == 1
    assert issues[0].t == 3.2
    assert issues[0].issue == "face off-frame"
    assert issues[0].severity == "high"


def test_parse_strips_markdown_fences():
    raw = "```json\n[{\"t\": 1.0, \"issue\": \"x\"}]\n```"
    issues = _parse_response(raw)
    assert len(issues) == 1


def test_parse_extracts_json_from_surrounding_prose():
    """Some VLMs wrap output in 'Here is the JSON: [...]' — we strip."""
    raw = 'Sure! Here is my analysis: [{"t": 2.5, "issue": "edge crop"}] — hope that helps.'
    issues = _parse_response(raw)
    assert len(issues) == 1
    assert issues[0].t == 2.5


def test_parse_unknown_severity_coerces_to_medium():
    raw = '[{"t": 0, "issue": "x", "severity": "catastrophic"}]'
    issues = _parse_response(raw)
    assert issues[0].severity == "medium"


def test_parse_garbage_returns_empty():
    assert _parse_response("not json") == []
    assert _parse_response("") == []
    assert _parse_response("{}") == []


def test_parse_drops_non_dict_entries():
    raw = '[{"t": 1, "issue": "ok"}, "garbage", 42]'
    issues = _parse_response(raw)
    assert len(issues) == 1


# ── Severity gating ─────────────────────────────────────────────


class _StubOrchestrator:
    def __init__(self, response: str):
        self._response = response
        self.calls = 0

    async def vlm_critique(self, *, prompt, images, max_tokens):
        self.calls += 1
        return self._response


def _write_tiny_png(path):
    """Write a 1x1 PNG so ffmpeg frame extraction sees a readable file."""
    # Minimal PNG signature + IHDR + IDAT + IEND for a 1x1 white pixel.
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


def test_report_not_ok_when_two_high_severity_issues(monkeypatch):
    """>= 2 high-severity issues trip the banner."""
    raw = (
        '[{"t": 1, "issue": "a", "severity": "high"},'
        '{"t": 3, "issue": "b", "severity": "high"}]'
    )
    # Stub out frame extraction so we don't need a real rendered clip.
    # post_render_critic imports extract_frames_for_critic lazily
    # inside the function, so we patch the source module.
    from backend.services import critic_loop as cl_mod
    from backend.services.critic_loop import FrameSample

    with tempfile.TemporaryDirectory() as d:
        stub_png = os.path.join(d, "s0.png")
        _write_tiny_png(stub_png)

        def _fake_extract(path, *, duration_sec, interval_sec, out_dir=None):
            return [FrameSample(t=0.0, frame_path=stub_png)]

        monkeypatch.setattr(
            cl_mod, "extract_frames_for_critic", _fake_extract,
        )
        orch = _StubOrchestrator(raw)
        # ``run_post_render_critic`` needs an existing ``rendered_path``
        # to pass the os.path.exists check.
        rendered = os.path.join(d, "clip.mp4")
        with open(rendered, "wb") as f:
            f.write(b"x")
        report = asyncio.run(run_post_render_critic(
            rendered_path=rendered,
            duration_sec=4.0,
            orchestrator=orch,
            max_frames=1,
        ))
        assert len(report.issues) == 2
        assert report.ok is False
        assert orch.calls == 1


def test_report_ok_when_single_high_severity_issue(monkeypatch):
    """One high issue alone does NOT flip ``ok``."""
    raw = '[{"t": 1, "issue": "a", "severity": "high"}]'
    # post_render_critic imports extract_frames_for_critic lazily
    # inside the function, so we patch the source module.
    from backend.services import critic_loop as cl_mod
    from backend.services.critic_loop import FrameSample

    with tempfile.TemporaryDirectory() as d:
        stub_png = os.path.join(d, "s0.png")
        _write_tiny_png(stub_png)

        def _fake_extract(path, *, duration_sec, interval_sec, out_dir=None):
            return [FrameSample(t=0.0, frame_path=stub_png)]

        monkeypatch.setattr(
            cl_mod, "extract_frames_for_critic", _fake_extract,
        )
        orch = _StubOrchestrator(raw)
        rendered = os.path.join(d, "clip.mp4")
        with open(rendered, "wb") as f:
            f.write(b"x")
        report = asyncio.run(run_post_render_critic(
            rendered_path=rendered,
            duration_sec=4.0,
            orchestrator=orch,
            max_frames=1,
        ))
        assert report.ok is True
        assert len(report.issues) == 1


def test_missing_rendered_path_returns_ok_report():
    orch = _StubOrchestrator("[]")
    report = asyncio.run(run_post_render_critic(
        rendered_path="/nonexistent/clip.mp4",
        duration_sec=4.0,
        orchestrator=orch,
    ))
    assert report.ok is True
    assert orch.calls == 0


def test_to_dict_is_json_safe():
    r = PostRenderReport(
        ok=False,
        issues=[PostRenderIssue(t=1.0, issue="x", severity="high")],
        sampled_frames=3,
        vlm_latency_sec=2.5,
    )
    d = r.to_dict()
    # All primitive types so save_job's JSON serializer accepts it.
    import json
    json.dumps(d)
    assert d["ok"] is False
    assert d["issues"][0]["severity"] == "high"
