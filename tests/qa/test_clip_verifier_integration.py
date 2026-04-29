"""Task 6 — end-to-end wiring of apply_visual_verification.

Validates that ``ai_orchestrator.detect_viral_clips`` actually invokes
the verifier on every return path (primary, partial-timeout,
all-providers-failed) and that the orchestrator records the
verification outcome in ``score_diagnostics`` so the UI badge has the
data it needs.

The bare orchestrator is heavy to construct (probes Ollama, walks the
provider chain). These tests bypass ``__init__`` via
``object.__new__`` and wire only the attrs the methods under test
actually read.

Run:  pytest tests/qa/test_clip_verifier_integration.py -v
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

# AIOrchestrator imports the full provider stack (anthropic, google,
# openai, ...) at module load time. The sandboxed CI runner ships a
# subset of those deps; the production / homelab containers ship them
# all. Skip the suite cleanly in the sandbox so the harness stays
# green there, and run it for real inside docker compose where the
# imports succeed. This matches the skip pattern the config-suite
# uses for missing pydantic_settings.
try:
    from backend.services.ai_orchestrator import AIOrchestrator
except BaseException as _imp_err:  # pragma: no cover — sandbox-only
    # BaseException catches pyo3_runtime.PanicException raised by the
    # cryptography rust shim that google.auth pulls in transitively.
    AIOrchestrator = None
    _SKIP_REASON = f"ai_orchestrator import failed in sandbox: {_imp_err!r}"
else:
    _SKIP_REASON = None

# Per-test marker so static source-level tests still run in the
# sandbox (where AIOrchestrator's import chain panics on missing
# crypto deps) while integration tests skip cleanly.
needs_orchestrator = pytest.mark.skipif(
    AIOrchestrator is None, reason=_SKIP_REASON or "",
)


def _make_clip(
    viral_score=70, idx=1,
    hook_score=50, flow_score=50, value_score=50, trend_score=50,
):
    from backend.models import ClipCandidate
    return ClipCandidate(
        id=idx,
        title=f"Clip {idx}",
        start_time=float(idx),
        end_time=float(idx) + 10.0,
        duration=10.0,
        viral_score=viral_score,
        viral_score_reasoning="test",
        clip_type="moment",
        platform="tiktok",
        suggested_caption="cap",
        hook_text="hook",
        why_this_works="why",
        hook_score=hook_score, flow_score=flow_score,
        value_score=value_score, trend_score=trend_score,
    )


def _make_orchestrator(active_chain):
    """Build a minimal AIOrchestrator skipping the heavy __init__.

    ``active_chain`` is the list of providers ``_get_active_chain``
    will return.
    """
    orch = object.__new__(AIOrchestrator)
    orch._providers = {p.provider_name: p for p in active_chain}
    orch._unreachable = set()
    orch._cancel_check = None
    orch._ws_broadcast = None
    orch._custom_prompts = None
    orch._consecutive_ollama_failures = 0
    orch._current_model_override = None
    orch._ollama_text_fallback_only = False

    class _NopBreaker:
        def is_degraded(self, _name): return False
        def record_success(self, _name): pass
        def record_failure(self, _name): pass

    orch._circuit_breaker = _NopBreaker()
    orch._get_active_chain = lambda: list(active_chain)
    orch._notify_attempt = mock.AsyncMock()
    orch._notify_fallback = mock.AsyncMock()
    orch._get_task_model = lambda p, _task: f"{p.provider_name}:test-model"
    return orch


def _fake_provider(
    *,
    name="fake",
    supports_vision=True,
    detect_result=None,
    detect_raises=None,
    detect_timeout=False,
    populate_partial=None,
):
    """A duck-typed provider stand-in.

    ``detect_result`` — list of ClipCandidate to return.
    ``detect_raises`` — exception instance to raise instead.
    ``detect_timeout`` — sleep forever so asyncio.wait_for fires.
    ``populate_partial`` — list to write into _partial_results before
        raising / timing out (mirrors the real partial-results path).
    """
    p = mock.MagicMock()
    p.provider_name = name
    p.supports_vision = supports_vision
    p.text_model_name = "fake-model"
    p.is_thinking_model = False

    async def _detect(*_args, **kwargs):
        if populate_partial is not None:
            kwargs.get("_partial_results", []).extend(populate_partial)
        if detect_timeout:
            await asyncio.sleep(60)
        if detect_raises is not None:
            raise detect_raises
        return list(detect_result or [])

    p.detect_viral_clips = _detect
    p._deduplicate_clips = lambda clips: list(clips)
    return p


# ── 1. orchestrator invokes the verifier on the nominal path ──────────


@needs_orchestrator
def test_orchestrator_invokes_verifier_after_finalize(monkeypatch):
    """detect_viral_clips → finalize_clip_scores → apply_visual_verification."""
    monkeypatch.setenv("CLIPAI_USE_VISUAL_CLIP_VERIFIER", "1")
    clips_returned = [_make_clip(idx=1), _make_clip(idx=2)]
    provider = _fake_provider(detect_result=clips_returned)
    orch = _make_orchestrator([provider])

    call_order: list[str] = []

    def _spy_finalize(clips, *_a, **_kw):
        call_order.append("finalize")

    async def _spy_apply(clips, frames, vp, *, cancel_check=None):
        call_order.append("verify")
        return clips

    with mock.patch(
        "backend.services.clip_scoring.finalize_clip_scores",
        side_effect=_spy_finalize,
    ), mock.patch(
        "backend.services.clip_scoring.apply_visual_verification",
        side_effect=_spy_apply,
    ) as mocked_verify:
        result, _ = asyncio.run(orch.detect_viral_clips(
            transcript=[], scenes=[], video_duration=60.0,
            job_id="t1",
            frames=["frame1"], cancel_check=lambda: None,
        ))
    assert result == clips_returned
    assert call_order == ["finalize", "verify"]
    mocked_verify.assert_called_once()


@needs_orchestrator
def test_orchestrator_skips_verifier_with_no_frames(monkeypatch):
    """frames=None → wrapper still called but verifier no-ops; the
    closure records a 'skipped' note and the score_diagnostics block
    is left empty so the UI doesn't render a stale badge."""
    monkeypatch.setenv("CLIPAI_USE_VISUAL_CLIP_VERIFIER", "1")
    clip = _make_clip()
    provider = _fake_provider(detect_result=[clip])
    orch = _make_orchestrator([provider])

    progress_msgs: list[str] = []

    result, _ = asyncio.run(orch.detect_viral_clips(
        transcript=[], scenes=[], video_duration=60.0,
        job_id="t1",
        frames=None,
        progress_callback=lambda m: progress_msgs.append(m),
    ))
    assert result == [clip]
    diag = result[0].score_diagnostics or {}
    vv = diag.get("visual_verification") or {}
    assert vv.get("note") == "skipped"
    assert any("skipped" in m for m in progress_msgs)


@needs_orchestrator
def test_orchestrator_skips_verifier_with_no_vision_provider(monkeypatch):
    """Provider chain has no vision-capable provider → vision_provider
    is None and the closure tags the diagnostics as 'skipped'."""
    monkeypatch.setenv("CLIPAI_USE_VISUAL_CLIP_VERIFIER", "1")
    clip = _make_clip()
    provider = _fake_provider(
        supports_vision=False, detect_result=[clip],
    )
    orch = _make_orchestrator([provider])

    progress_msgs: list[str] = []
    result, _ = asyncio.run(orch.detect_viral_clips(
        transcript=[], scenes=[], video_duration=60.0,
        job_id="t1",
        frames=["frame1"],
        progress_callback=lambda m: progress_msgs.append(m),
    ))
    assert result[0].score_diagnostics["visual_verification"]["note"] == "skipped"
    assert any("no vision provider" in m for m in progress_msgs)


@needs_orchestrator
def test_orchestrator_skips_verifier_when_flag_disabled(monkeypatch):
    """CLIPAI_USE_VISUAL_CLIP_VERIFIER=0 → verify_clips_visually never
    fires no matter how the inputs look."""
    monkeypatch.setenv("CLIPAI_USE_VISUAL_CLIP_VERIFIER", "0")
    clip = _make_clip()
    provider = _fake_provider(detect_result=[clip])
    orch = _make_orchestrator([provider])

    with mock.patch(
        "backend.services.clip_verifier.verify_clips_visually",
        new_callable=mock.AsyncMock,
    ) as inner:
        asyncio.run(orch.detect_viral_clips(
            transcript=[], scenes=[], video_duration=60.0,
            job_id="t1",
            frames=["frame"],
        ))
    inner.assert_not_called()


# ── 2. pipeline → orchestrator wiring ─────────────────────────────────


def test_pipeline_passes_frames_through_to_orchestrator():
    """pipeline.py:6196 builds the detect_viral_clips call with
    frames=frames and cancel_check=cancel_check. We assert the
    keyword is in the call signature by reading the source."""
    pipe_src = (_REPO / "backend" / "services" / "pipeline.py").read_text()
    # Both call sites (primary + retry) must pass frames=frames and
    # cancel_check=cancel_check.
    occurrences = pipe_src.count("frames=frames,")
    assert occurrences >= 2, (
        f"expected >=2 'frames=frames,' wiring sites, found {occurrences}"
    )
    cancel_occ = pipe_src.count("cancel_check=cancel_check,")
    assert cancel_occ >= 2


# ── 3. partial-results paths must also run the verifier ───────────────


@needs_orchestrator
def test_partial_results_path_also_runs_verifier(monkeypatch):
    """Provider times out but produces partial clips → verifier
    runs against the deduped partials."""
    monkeypatch.setenv("CLIPAI_USE_VISUAL_CLIP_VERIFIER", "1")
    partial_clip = _make_clip(idx=99)
    provider = _fake_provider(
        detect_timeout=True,
        populate_partial=[partial_clip],
    )
    orch = _make_orchestrator([provider])

    seen: list = []

    async def _spy_apply(clips, frames, vp, *, cancel_check=None):
        seen.append(list(clips))
        return clips

    # Force the wait_for timeout to fire fast.
    real_wait = asyncio.wait_for

    async def _fast_wait(coro, timeout):
        return await real_wait(coro, 0.05)

    with mock.patch.object(
        asyncio, "wait_for", side_effect=_fast_wait,
    ), mock.patch(
        "backend.services.clip_scoring.apply_visual_verification",
        side_effect=_spy_apply,
    ) as mocked_verify:
        result, label = asyncio.run(orch.detect_viral_clips(
            transcript=[], scenes=[], video_duration=60.0,
            job_id="t1",
            frames=["f"],
        ))
    assert result == [partial_clip]
    assert "(partial)" in label
    assert mocked_verify.call_count == 1
    assert seen[0] == [partial_clip]


@needs_orchestrator
def test_all_providers_failed_path_runs_verifier_on_partial(monkeypatch):
    """Provider raises but partial clips were captured → catastrophic
    path runs the verifier before returning."""
    from backend.services.providers.base import ProviderError
    monkeypatch.setenv("CLIPAI_USE_VISUAL_CLIP_VERIFIER", "1")
    partial_clip = _make_clip(idx=42)
    provider = _fake_provider(
        detect_raises=ProviderError("synthetic failure"),
        populate_partial=[partial_clip],
    )
    orch = _make_orchestrator([provider])

    async def _spy_apply(clips, frames, vp, *, cancel_check=None):
        return clips

    with mock.patch(
        "backend.services.clip_scoring.apply_visual_verification",
        side_effect=_spy_apply,
    ) as mocked_verify:
        result, label = asyncio.run(orch.detect_viral_clips(
            transcript=[], scenes=[], video_duration=60.0,
            job_id="t1",
            frames=["f"],
        ))
    assert result == [partial_clip]
    assert label == "partial"
    assert mocked_verify.call_count == 1


# ── 4. exception safety: a verifier crash never aborts an export ──────


@needs_orchestrator
def test_verifier_failure_does_not_abort_export(monkeypatch):
    """If apply_visual_verification raises, detect_viral_clips still
    returns clips. (apply_visual_verification itself swallows
    exceptions; we double-check the orchestrator's defensive layer
    by injecting a crash at the inner verify_clips_visually.)"""
    monkeypatch.setenv("CLIPAI_USE_VISUAL_CLIP_VERIFIER", "1")
    clip = _make_clip()
    provider = _fake_provider(detect_result=[clip])
    orch = _make_orchestrator([provider])

    with mock.patch(
        "backend.services.clip_verifier.verify_clips_visually",
        side_effect=RuntimeError("vision provider exploded"),
    ):
        result, _ = asyncio.run(orch.detect_viral_clips(
            transcript=[], scenes=[], video_duration=60.0,
            job_id="t1",
            frames=["f"],
        ))
    assert result == [clip]


# ── 5. score_diagnostics records the verification outcome ─────────────


@needs_orchestrator
def test_score_diagnostics_records_verification_outcome(monkeypatch):
    """After a successful verification, score_diagnostics carries
    pre/post scores and a note label."""
    monkeypatch.setenv("CLIPAI_USE_VISUAL_CLIP_VERIFIER", "1")
    # Per-axis scores must match viral_score so finalize_clip_scores'
    # composite stays at 70 (composite = sum(axis * weight); weights
    # sum to 1, so axis=70 across the board → composite=70). Without
    # this, the composite rewrites viral_score down to ~50 BEFORE the
    # verifier runs, and the pre_score assertion below fails. The
    # orchestrator runs finalize_clip_scores → verifier in that order.
    clip = _make_clip(
        viral_score=70,
        hook_score=70, flow_score=70, value_score=70, trend_score=70,
    )
    provider = _fake_provider(detect_result=[clip])
    orch = _make_orchestrator([provider])

    async def _bump(clips, frames, vp, *, cancel_check=None):
        for c in clips:
            c.viral_score = min(100, c.viral_score + 5)
        return clips

    with mock.patch(
        "backend.services.clip_scoring.apply_visual_verification",
        side_effect=_bump,
    ):
        result, _ = asyncio.run(orch.detect_viral_clips(
            transcript=[], scenes=[], video_duration=60.0,
            job_id="t1",
            frames=["f"],
        ))
    vv = result[0].score_diagnostics["visual_verification"]
    assert vv["pre_score"] == 70
    assert vv["post_score"] == 75
    assert vv["delta"] == 5
    assert vv["note"] == "boosted"


@needs_orchestrator
def test_score_diagnostics_lowered_note_on_score_drop(monkeypatch):
    monkeypatch.setenv("CLIPAI_USE_VISUAL_CLIP_VERIFIER", "1")
    clip = _make_clip(viral_score=70)
    provider = _fake_provider(detect_result=[clip])
    orch = _make_orchestrator([provider])

    async def _drop(clips, frames, vp, *, cancel_check=None):
        for c in clips:
            c.viral_score = max(1, c.viral_score - 8)
        return clips

    with mock.patch(
        "backend.services.clip_scoring.apply_visual_verification",
        side_effect=_drop,
    ):
        result, _ = asyncio.run(orch.detect_viral_clips(
            transcript=[], scenes=[], video_duration=60.0,
            job_id="t1",
            frames=["f"],
        ))
    vv = result[0].score_diagnostics["visual_verification"]
    assert vv["delta"] == -8
    assert vv["note"] == "lowered"


# ── 6. progress callback announces the per-run change count ────────────


@needs_orchestrator
def test_progress_callback_announces_verification_count(monkeypatch):
    """The progress message includes both the count of changed clips
    and the total clips evaluated."""
    monkeypatch.setenv("CLIPAI_USE_VISUAL_CLIP_VERIFIER", "1")
    clips = [_make_clip(viral_score=60, idx=1),
             _make_clip(viral_score=70, idx=2),
             _make_clip(viral_score=80, idx=3)]
    provider = _fake_provider(detect_result=clips)
    orch = _make_orchestrator([provider])

    async def _bump_one(_clips, frames, vp, *, cancel_check=None):
        _clips[0].viral_score = 65
        return _clips

    msgs: list[str] = []
    with mock.patch(
        "backend.services.clip_scoring.apply_visual_verification",
        side_effect=_bump_one,
    ):
        asyncio.run(orch.detect_viral_clips(
            transcript=[], scenes=[], video_duration=60.0,
            job_id="t1",
            frames=["f"],
            progress_callback=lambda m: msgs.append(m),
        ))
    summary = next((m for m in msgs if "Visual verification adjusted" in m), None)
    assert summary is not None, msgs
    assert "1 of 3" in summary


# ── 7. vision_provider property selects the first vision-capable ──────


@needs_orchestrator
def test_vision_provider_property_picks_first_vision_capable():
    """AIOrchestrator.vision_provider walks the active chain in order
    and returns the first provider whose supports_vision is True."""
    text_only = _fake_provider(name="text_only", supports_vision=False)
    vision_a = _fake_provider(name="vision_a", supports_vision=True)
    vision_b = _fake_provider(name="vision_b", supports_vision=True)
    orch = _make_orchestrator([text_only, vision_a, vision_b])
    assert orch.vision_provider is vision_a


@needs_orchestrator
def test_vision_provider_returns_none_when_no_vision():
    text_only = _fake_provider(name="text_only", supports_vision=False)
    orch = _make_orchestrator([text_only])
    assert orch.vision_provider is None
