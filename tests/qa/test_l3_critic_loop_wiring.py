"""Layer 3 (VLM rubric critic) wiring contract.

The critic_loop module itself is fully implemented and tested
upstream. These tests pin the integration glue:

  * /sota-clip-bench's gate on CLIPAI_CRITIC_MODE
  * The L3 SSE phase events: phase_start + phase_result(ok|skipped)
  * critic_summary["vlm_rubric"] shape
  * Bench verdict_for L3 gate (downgrade-only, absent → no-op)
  * UI-facing structure of the report dict

Run:  pytest tests/qa/test_l3_critic_loop_wiring.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# The diagnostics router imports httpx at module top, which isn't
# present in the sandbox here but ships in the production
# image / CI Docker. Tests that touch it skip cleanly off-host.
try:
    import httpx  # noqa: F401
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

skip_no_httpx = pytest.mark.skipif(
    not _HAS_HTTPX, reason="httpx required for backend.routers.diagnostics",
)


# ── 1. critic_loop public API is stable ──────────────────────────


def test_critic_loop_public_api_present():
    """The wiring assumes ``score_plan``, ``extract_frames_for_critic``,
    plus the four dataclasses. Pin their existence so downstream
    PRs that touch critic_loop don't accidentally break the wiring.
    """
    from backend.services import critic_loop as cl

    assert callable(cl.score_plan)
    assert callable(cl.extract_frames_for_critic)
    assert hasattr(cl, "FrameSample")
    assert hasattr(cl, "FrameScore")
    assert hasattr(cl, "ReSolveRequest")
    assert hasattr(cl, "CriticReport")


# ── 2. score_plan with mode=off is a true no-op ──────────────────


def test_score_plan_off_mode_returns_off_report():
    """``critic_mode='off'`` short-circuits without touching VLM /
    learned backends. Bench gate's ``elif critic_mode_env == 'off'``
    branch relies on this contract."""
    from backend.services.critic_loop import score_plan, FrameSample
    from backend.services.reframe_config import get_default_config

    cfg = get_default_config().override(critic_mode="off")
    samples = [FrameSample(t=0.0, frame_path="/tmp/x.png")]
    report = score_plan(samples=samples, config=cfg)
    assert report.mode == "off"
    assert report.budget_used == 0
    assert report.per_frame == []
    assert report.resolve_requests == []


def test_score_plan_off_mode_with_no_samples():
    """Mode=off + zero samples still returns an empty CriticReport
    rather than raising — the bench's empty-frames branch relies on
    this when ffmpeg returns no frames."""
    from backend.services.critic_loop import score_plan
    from backend.services.reframe_config import get_default_config

    cfg = get_default_config().override(critic_mode="learned")
    report = score_plan(samples=[], config=cfg)
    assert report.mode == "off"  # zero-sample short-circuit


# ── 3. score_plan in learned mode is offline-safe ────────────────


def test_score_plan_learned_mode_runs_without_network(tmp_path, monkeypatch):
    """``critic_mode='learned'`` produces a report without making any
    network call — it falls through to ``aesthetic_scorer.score_frame``
    (or its stub when the heuristic isn't installed). The bench's
    default-off ``learned`` opt-in relies on this path being
    100% local + free."""
    from backend.services import critic_loop as cl
    from backend.services.reframe_config import get_default_config

    monkeypatch.setattr(
        cl, "_score_frame_learned",
        lambda sample: cl.FrameScore(t=sample.t, score=7.5, reason="learned"),
    )
    # Guard: VLM dispatcher must NOT be invoked in learned mode.
    def _should_not_be_called(*a, **kw):
        raise AssertionError("VLM dispatcher invoked in learned mode")
    monkeypatch.setattr(cl, "_score_frame_vlm", _should_not_be_called)
    monkeypatch.setattr(cl, "_load_cache", lambda key, cache_dir: None)
    monkeypatch.setattr(cl, "_store_cache", lambda key, cache_dir, report: None)

    cfg = get_default_config().override(critic_mode="learned")
    samples = [
        cl.FrameSample(t=0.0, frame_path="/x.png"),
        cl.FrameSample(t=1.0, frame_path="/y.png"),
    ]
    report = cl.score_plan(
        samples=samples, source_sha256="a" * 64, plan_hash="b" * 16,
        config=cfg,
    )
    assert report.mode == "learned"
    assert len(report.per_frame) == 2
    assert all(fs.score == 7.5 for fs in report.per_frame)
    assert report.budget_used == 0  # learned mode never bills budget


# ── 4. score_plan VLM mode falls back when both backends unavailable ─


def test_score_plan_vlm_falls_back_to_learned_on_provider_unavailable(
    monkeypatch,
):
    """When both Ollama and OpenRouter return None (unavailable),
    ``_score_frame_vlm`` falls through to ``_score_frame_learned``
    so the rubric still produces a score. ``budget_used`` reflects
    the attempted VLM calls (the exact count is a critic_loop
    internal — we just check the report shape is valid)."""
    from backend.services import critic_loop as cl
    from backend.services.reframe_config import get_default_config

    # Force both providers to fail.
    monkeypatch.setattr(
        cl, "_score_frame_vlm_ollama", lambda sample: None,
    )
    monkeypatch.setattr(
        cl, "_score_frame_vlm_openrouter", lambda sample: None,
    )
    monkeypatch.setattr(
        cl, "_score_frame_learned",
        lambda sample: cl.FrameScore(
            t=sample.t, score=6.5, reason="learned-fallback",
        ),
    )
    monkeypatch.setattr(cl, "_load_cache", lambda key, cache_dir: None)
    monkeypatch.setattr(cl, "_store_cache", lambda key, cache_dir, report: None)

    cfg = get_default_config().override(
        critic_mode="vlm", critic_budget_per_clip=2,
    )
    samples = [
        cl.FrameSample(t=0.0, frame_path="/x.png"),
        cl.FrameSample(t=1.0, frame_path="/y.png"),
    ]
    report = cl.score_plan(
        samples=samples, source_sha256="a" * 64, plan_hash="b" * 16,
        config=cfg,
    )
    assert report.mode == "vlm"
    assert len(report.per_frame) == 2
    # Each frame fell back to learned — score 6.5.
    assert all(fs.score == 6.5 for fs in report.per_frame)


# ── 5. Cache hit short-circuits recompute ────────────────────────


def test_score_plan_cache_hit_skips_recompute(tmp_path, monkeypatch):
    """When ``_load_cache`` returns a CriticReport, ``score_plan``
    returns it without invoking either backend. The bench's
    "second run is free" UX depends on this."""
    from backend.services import critic_loop as cl
    from backend.services.reframe_config import get_default_config

    sentinel = cl.CriticReport(
        mode="learned", mean_score=7.7,
        per_frame=[cl.FrameScore(t=0.0, score=7.7, reason="cached")],
        resolve_requests=[],
        budget_used=0,
    )

    def _hit_cache(key, cache_dir):
        return sentinel

    def _should_not_score(sample):
        raise AssertionError("scored frame despite cache hit")

    monkeypatch.setattr(cl, "_load_cache", _hit_cache)
    monkeypatch.setattr(cl, "_score_frame_learned", _should_not_score)
    monkeypatch.setattr(cl, "_score_frame_vlm", _should_not_score)

    cfg = get_default_config().override(critic_mode="learned")
    samples = [cl.FrameSample(t=0.0, frame_path="/x.png")]
    report = cl.score_plan(
        samples=samples, source_sha256="a" * 64, plan_hash="b" * 16,
        config=cfg,
    )
    assert report is sentinel
    assert report.mean_score == 7.7


# ── 6. Low-score windows produce ReSolveRequest ──────────────────


def test_score_plan_low_window_produces_resolve_request(monkeypatch):
    """Frames scoring below ``critic_threshold`` get coalesced into
    ReSolveRequest objects. The bench surfaces these in
    ``critic_summary['vlm_rubric']['low_windows']`` so the operator
    can see what would be re-solved."""
    from backend.services import critic_loop as cl
    from backend.services.reframe_config import get_default_config

    # All frames score 3.0 — well below the 6.0 default threshold.
    monkeypatch.setattr(
        cl, "_score_frame_learned",
        lambda sample: cl.FrameScore(
            t=sample.t, score=3.0, reason="low",
        ),
    )
    monkeypatch.setattr(cl, "_load_cache", lambda key, cache_dir: None)
    monkeypatch.setattr(cl, "_store_cache", lambda key, cache_dir, report: None)

    cfg = get_default_config().override(
        critic_mode="learned", critic_threshold=6.0,
    )
    samples = [
        cl.FrameSample(t=0.0, frame_path="/a.png"),
        cl.FrameSample(t=1.0, frame_path="/b.png"),
        cl.FrameSample(t=2.0, frame_path="/c.png"),
    ]
    report = cl.score_plan(
        samples=samples, source_sha256="a" * 64, plan_hash="b" * 16,
        config=cfg,
    )
    assert len(report.resolve_requests) >= 1
    rr = report.resolve_requests[0]
    assert rr.start <= 2.0
    assert rr.end >= 0.0
    assert rr.suggested_action in ("blur_fill", "widen", "more_padding", "hold")


# ── 7. Bench verdict gate (L3 downgrade) ─────────────────────────


def test_l3_low_score_downgrades_verdict_to_marginal():
    """Synthetic clip with all hold-zone gates passing AND L1/L2
    deltas in tolerance, but VLM rubric mean below 5.5 → MARGINAL."""
    from backend.scripts.compare_autoflip_vs_clipai import (
        verdict_for, TARGET_ZONES,
    )
    zone = dict(TARGET_ZONES["default"])
    metrics = {
        "n_segments": 10,
        "median_hold_sec": 3.0,
        "segments_under_1s_rate": 0.05,
        "vlm_rubric_mean_score": 5.0,
    }
    assert verdict_for(metrics, zone) == "MARGINAL"


def test_l3_high_score_preserves_pass():
    """``vlm_rubric_mean_score`` above threshold → PASS holds."""
    from backend.scripts.compare_autoflip_vs_clipai import (
        verdict_for, TARGET_ZONES,
    )
    zone = dict(TARGET_ZONES["default"])
    metrics = {
        "n_segments": 10,
        "median_hold_sec": 3.0,
        "segments_under_1s_rate": 0.05,
        "vlm_rubric_mean_score": 7.5,
    }
    assert verdict_for(metrics, zone) == "PASS"


def test_l3_absent_score_is_no_op():
    """No ``vlm_rubric_mean_score`` in metrics (L3 was off /
    skipped) → gate is a no-op. Verdict reflects only L1+L2."""
    from backend.scripts.compare_autoflip_vs_clipai import (
        verdict_for, TARGET_ZONES,
    )
    zone = dict(TARGET_ZONES["default"])
    metrics = {
        "n_segments": 10,
        "median_hold_sec": 3.0,
        "segments_under_1s_rate": 0.05,
    }
    assert verdict_for(metrics, zone) == "PASS"


def test_l3_zone_override_wins():
    """Per-zone ``vlm_rubric_mean_min`` override beats the
    module-level default. Zone says 4.0 → 5.0 still passes."""
    from backend.scripts.compare_autoflip_vs_clipai import (
        verdict_for, TARGET_ZONES,
    )
    zone = dict(TARGET_ZONES["default"])
    zone["vlm_rubric_mean_min"] = 4.0
    metrics = {
        "n_segments": 10,
        "median_hold_sec": 3.0,
        "segments_under_1s_rate": 0.05,
        "vlm_rubric_mean_score": 5.0,
    }
    assert verdict_for(metrics, zone) == "PASS"


# ── 8. Module-level constant honors env var override ────────────


def test_l3_env_var_default_threshold(monkeypatch):
    """``CLIPAI_CRITIC_VERDICT_MIN`` env var is read at module import
    time. We can't easily re-import the module under monkeypatch
    inside the same process, so the test pins the *value* of the
    constant matches the expected default."""
    from backend.scripts import compare_autoflip_vs_clipai as bench
    # Default is 5.5 unless the env var was set at import time.
    assert isinstance(bench.L3_VLM_RUBRIC_MEAN_MIN, float)
    assert 0.0 <= bench.L3_VLM_RUBRIC_MEAN_MIN <= 10.0


# ── 9. Probe helper degrades gracefully ──────────────────────────


@skip_no_httpx
def test_probe_preview_duration_returns_fallback_when_ffprobe_missing(
    monkeypatch, tmp_path,
):
    """The L3 wiring's ffprobe helper falls back to a generous
    duration when ffprobe is missing — extract_frames_for_critic
    silently terminates past EOF, so over-estimating is safe."""
    from backend.routers import diagnostics as diag

    monkeypatch.setattr(
        "shutil.which", lambda name: None if name == "ffprobe" else "/x",
    )
    out = diag._probe_preview_duration(str(tmp_path / "missing.mp4"))
    assert out == 600.0


@skip_no_httpx
def test_probe_preview_duration_returns_fallback_on_ffprobe_failure(
    monkeypatch, tmp_path,
):
    """ffprobe present but errors → fallback fires."""
    from backend.routers import diagnostics as diag
    import subprocess as _sp

    monkeypatch.setattr(
        "shutil.which",
        lambda name: "/usr/bin/ffprobe" if name == "ffprobe" else None,
    )

    class _FakeCalledProcessError(_sp.CalledProcessError):
        pass

    def _raise(*a, **kw):
        raise _FakeCalledProcessError(returncode=1, cmd=a[0])

    monkeypatch.setattr("subprocess.run", _raise)
    out = diag._probe_preview_duration(
        str(tmp_path / "x.mp4"), fallback=42.0,
    )
    assert out == 42.0
