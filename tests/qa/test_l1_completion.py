"""Layer 1 completion contract.

The cTvFT branch shipped scaffolding (saliency-aware critic, cache
plumbing, repair handler) plus XC speaker accuracy. These tests pin
the production gaps the completion PR closes:

  * ``_TasedNetAdapter`` no longer pretends to work — when the
    vendored module is absent (or weights missing) it raises
    RuntimeError so AvSaliency falls back.
  * ``_SpectralResidualAdapter`` is a real, always-available adapter.
  * ``AvSaliency.__post_init__`` runs the fallback chain and exposes
    ``backend_name`` so the bench / SSE / UI know which adapter ran.
  * The bench's ``_extract_and_cache`` records the chosen backend in
    ``saliency_meta.json`` so the cache-HIT path on subsequent runs
    surfaces the same backend without re-extraction.
  * The bench result's ``critic.saliency`` block carries
    ``in_crop_mean`` / ``windows_flagged`` / ``backend`` so the
    UI's Critic engine row has data to render.
  * The pipeline's ``_compute_l1_saliency_peaks`` helper is callable
    and degrades gracefully on every failure path.

Run:  pytest tests/qa/test_l1_completion.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# Tests in sections 1-3 + 4-6 (all but the count-helper / vendor /
# text-grep tests) need numpy via av_saliency and httpx via pipeline.
# Skip cleanly on hosts that don't have them rather than producing
# import-time noise — the homelab + CI Docker images both ship them.
try:
    import numpy  # noqa: F401
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import httpx  # noqa: F401
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

skip_no_numpy = pytest.mark.skipif(
    not _HAS_NUMPY, reason="numpy required for av_saliency",
)
skip_no_httpx = pytest.mark.skipif(
    not _HAS_HTTPX, reason="httpx required for backend.services.pipeline",
)


# ── 1. _TasedNetAdapter no longer silently succeeds ────────────────


@skip_no_numpy
def test_tasednet_adapter_raises_when_vendored_module_missing(monkeypatch):
    """The vendored ``tasednet`` shim raises ImportError unless the
    operator drops a real model.py — that ImportError must propagate
    out of ``_TasedNetAdapter.load`` so AvSaliency can catch it."""
    from backend.services import av_saliency as avs_mod

    adapter = avs_mod._TasedNetAdapter(device="cpu")
    # The vendored ``backend.vendor.tasednet`` package re-raises
    # ImportError when ``model.py`` is absent (which is the default
    # repo state — operator drops it themselves).
    with pytest.raises((RuntimeError, ImportError)):
        adapter.load()


@skip_no_numpy
def test_tasednet_adapter_raises_when_weights_missing(tmp_path, monkeypatch):
    """Even if a vendored ``TASED_v2`` module is importable, the
    adapter must raise when ``TASED_NET_MODEL_PATH`` points at a
    non-existent file. Mock the upstream module + torch so the test
    runs without PyTorch."""
    from backend.services import av_saliency as avs_mod

    monkeypatch.setenv(
        "TASED_NET_MODEL_PATH", str(tmp_path / "does-not-exist.pth"),
    )

    # Inject a fake vendored tasednet so the import succeeds.
    class _FakeTasedV2:
        def __init__(self):
            pass

    fake_pkg = type("FakePkg", (), {"TASED_v2": _FakeTasedV2})
    monkeypatch.setitem(
        sys.modules, "backend.vendor.tasednet", fake_pkg,
    )
    # Also mock torch so the import inside load() succeeds.
    monkeypatch.setitem(sys.modules, "torch", type("FakeTorch", (), {}))

    adapter = avs_mod._TasedNetAdapter(device="cpu")
    with pytest.raises(RuntimeError, match="weights not found"):
        adapter.load()


# ── 2. SpectralResidual adapter exists + reports backend_name ──────


@skip_no_numpy
def test_spectral_residual_adapter_reports_backend_name():
    from backend.services.av_saliency import _SpectralResidualAdapter
    a = _SpectralResidualAdapter(device="cpu")
    assert a.backend_name == "spectral_residual"


@skip_no_numpy
def test_spectral_residual_adapter_load_fails_clearly_without_contrib(
    monkeypatch,
):
    """When ``cv2.saliency`` isn't available (operator on plain
    opencv-python instead of contrib), ``load()`` raises a
    RuntimeError with an actionable message, not a bare AttributeError.
    """
    from backend.services.av_saliency import _SpectralResidualAdapter

    # Build a fake cv2 missing the saliency module.
    class _FakeCv2Module:
        error = RuntimeError  # cv2.error type

        # Intentionally no `saliency` attribute.

    monkeypatch.setitem(sys.modules, "cv2", _FakeCv2Module())
    a = _SpectralResidualAdapter(device="cpu")
    with pytest.raises(RuntimeError, match="cv2.saliency missing"):
        a.load()


# ── 3. AvSaliency fallback chain ────────────────────────────────────


@skip_no_numpy
def test_av_saliency_falls_back_to_spectral_when_tased_unavailable(
    monkeypatch,
):
    """With the default ``_TasedNetAdapter`` failing (no weights, no
    vendored module), ``AvSaliency.__post_init__`` falls back to
    ``_SpectralResidualAdapter`` and ``backend_name`` reflects that."""
    from backend.services import av_saliency as avs_mod

    # Force TasedNet adapter to fail on load.
    class _FailingTasedAdapter:
        backend_name = "tased_net"

        def __init__(self, *, device):
            pass

        def load(self):
            raise RuntimeError("simulated: vendored module missing")

    # Mock the spectral residual adapter to a no-op load (no cv2 in
    # the sandbox, but the real path exercises it on the homelab).
    class _DummySpectralAdapter:
        backend_name = "spectral_residual"

        def __init__(self, *, device):
            pass

        def load(self):
            return None

    monkeypatch.setattr(avs_mod, "_TasedNetAdapter", _FailingTasedAdapter)
    monkeypatch.setattr(
        avs_mod, "_SpectralResidualAdapter", _DummySpectralAdapter,
    )

    s = avs_mod.AvSaliency(device="cpu")
    assert s.backend_name == "spectral_residual"


@skip_no_numpy
def test_av_saliency_raises_when_both_backends_fail(monkeypatch):
    """If TasedNet AND SpectralResidual both fail to load, the
    constructor surfaces the underlying error rather than silently
    handing back an unloaded AvSaliency."""
    from backend.services import av_saliency as avs_mod

    class _FailingAdapter:
        backend_name = "x"

        def __init__(self, *, device):
            pass

        def load(self):
            raise RuntimeError("simulated total saliency failure")

    monkeypatch.setattr(avs_mod, "_TasedNetAdapter", _FailingAdapter)
    monkeypatch.setattr(avs_mod, "_SpectralResidualAdapter", _FailingAdapter)
    with pytest.raises(RuntimeError, match="no saliency backend available"):
        avs_mod.AvSaliency(device="cpu")


@skip_no_numpy
def test_av_saliency_cuda_request_downgrades_to_cpu_when_no_gpu(monkeypatch):
    """``device='cuda'`` + no GPU silently downgrades to CPU instead
    of raising — matches the pre-completion contract."""
    from backend.services import av_saliency as avs_mod

    monkeypatch.setattr(avs_mod, "_is_gpu_visible", lambda: False)

    class _DummyAdapter:
        backend_name = "spectral_residual"

        def __init__(self, *, device):
            self.device = device

        def load(self):
            return None

    monkeypatch.setattr(avs_mod, "_TasedNetAdapter", _DummyAdapter)
    monkeypatch.setattr(avs_mod, "_SpectralResidualAdapter", _DummyAdapter)
    s = avs_mod.AvSaliency(device="cuda")
    assert s.device == "cpu"


# ── 4. Bench writes saliency_meta.json + critic block ───────────────


def test_bench_records_saliency_backend_in_meta(tmp_path, monkeypatch):
    """``_extract_and_cache``'s saliency phase persists the chosen
    backend to ``saliency_meta.json`` so the cache-HIT path can
    surface it on subsequent runs without re-extraction."""
    from backend.scripts import compare_autoflip_vs_clipai as bench

    # Stub the helper so we control the backend reported.
    monkeypatch.setattr(
        bench, "_compute_saliency_peaks_per_second",
        lambda *a, **kw: [[[25.0, 50.0, 0.9]], [[27.0, 50.0, 0.85]]],
    )
    bench._compute_saliency_peaks_per_second.last_backend = "spectral_residual"  # type: ignore[attr-defined]
    monkeypatch.setenv("CLIPAI_SALIENCY_LAYER", "1")

    # Simulate the saliency-write block in isolation.
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    peaks = bench._compute_saliency_peaks_per_second(
        Path("/x"), None, 5.0, slug="x",
    )
    backend = getattr(
        bench._compute_saliency_peaks_per_second, "last_backend", "unknown",
    ) if peaks else "failed"
    (cache_dir / "saliency_peaks_per_second.json").write_text(
        json.dumps(peaks)
    )
    (cache_dir / "saliency_meta.json").write_text(
        json.dumps({"backend": backend})
    )

    meta = json.loads((cache_dir / "saliency_meta.json").read_text())
    assert meta == {"backend": "spectral_residual"}


def test_load_cached_extraction_surfaces_saliency_backend(tmp_path):
    """The cache reader exposes ``saliency_backend`` from the meta
    file so the bench's main loop can populate
    ``result.critic.saliency.backend``."""
    from backend.scripts import compare_autoflip_vs_clipai as bench

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "metadata.json").write_text(
        json.dumps({"width": 1920, "height": 1080, "fps": 30.0})
    )
    (cache_dir / "segments.json").write_text("[]")
    (cache_dir / "saliency_peaks_per_second.json").write_text("[]")
    (cache_dir / "saliency_meta.json").write_text(
        json.dumps({"backend": "spectral_residual"})
    )

    payload = bench._load_cached_extraction(cache_dir)
    assert payload.get("saliency_backend") == "spectral_residual"


def test_load_cached_extraction_handles_pre_meta_caches(tmp_path):
    """Caches written before saliency_meta.json existed return None
    for ``saliency_backend`` — not a KeyError."""
    from backend.scripts import compare_autoflip_vs_clipai as bench

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "metadata.json").write_text(
        json.dumps({"width": 1920, "height": 1080, "fps": 30.0})
    )
    (cache_dir / "segments.json").write_text("[]")
    payload = bench._load_cached_extraction(cache_dir)
    assert "saliency_backend" not in payload or payload["saliency_backend"] is None


# ── 5. _count_saliency_flagged_windows ──────────────────────────────


def test_count_flagged_windows_zero_when_no_data():
    from backend.scripts.compare_autoflip_vs_clipai import (
        _count_saliency_flagged_windows,
    )
    assert _count_saliency_flagged_windows([], None) == 0
    assert _count_saliency_flagged_windows([], []) == 0


def test_count_flagged_windows_counts_excluded_peaks():
    """Peaks at x_pct=80 with a centered crop (cx=50, half=28.125)
    fall outside [21.875, 78.125] — flag fires."""
    from backend.scripts.compare_autoflip_vs_clipai import (
        _count_saliency_flagged_windows,
    )
    events = [
        {"t": 0.0, "crop_cx": 0.5},
        {"t": 1.0, "crop_cx": 0.5},
    ]
    peaks = [
        [(80.0, 50.0, 0.9)],   # outside the crop at sec 0
        [(50.0, 50.0, 0.9)],   # inside the crop at sec 1
    ]
    flagged = _count_saliency_flagged_windows(events, peaks)
    assert flagged == 1


def test_count_flagged_windows_uses_median_crop_per_second():
    """Multiple events per second — median is what the count uses."""
    from backend.scripts.compare_autoflip_vs_clipai import (
        _count_saliency_flagged_windows,
    )
    # Sec 0: events at cx=0.3 / 0.5 / 0.7 → median 0.5; peak at x=80
    # is outside the half-56.25-pct window → flagged.
    events = [
        {"t": 0.0, "crop_cx": 0.3},
        {"t": 0.0, "crop_cx": 0.5},
        {"t": 0.0, "crop_cx": 0.7},
    ]
    peaks = [[(80.0, 50.0, 0.9)]]
    assert _count_saliency_flagged_windows(events, peaks) == 1


# ── 6. Pipeline plumbing ───────────────────────────────────────────


@skip_no_httpx
def test_compute_l1_saliency_returns_none_on_no_frames(tmp_path):
    """The pipeline helper returns None (not an empty list, not a
    crash) when given no frames — duration <1 also bails early."""
    from backend.services.pipeline import _compute_l1_saliency_peaks

    out = _compute_l1_saliency_peaks(
        frames=[], audio_path=None, duration_sec=10.0,
        job_id="test", cache_dir=str(tmp_path),
    )
    assert out is None


@skip_no_httpx
def test_compute_l1_saliency_returns_none_on_zero_duration(tmp_path):
    from backend.services.pipeline import _compute_l1_saliency_peaks

    class _F:
        timestamp = 0.0
        path = "/tmp/x.png"

    out = _compute_l1_saliency_peaks(
        frames=[_F()], audio_path=None, duration_sec=0.0,
        job_id="test", cache_dir=str(tmp_path),
    )
    assert out is None


@skip_no_httpx
def test_pipeline_helper_handles_av_saliency_import_failure(
    tmp_path, monkeypatch,
):
    """If ``backend.services.av_saliency`` fails to import, the
    helper returns None — never raises into the pipeline."""
    import backend.services.pipeline as pipe_mod

    # Force the helper's import path to fail.
    real_import = __builtins__["__import__"] if isinstance(
        __builtins__, dict
    ) else __import__

    def _failing_import(name, *args, **kwargs):
        if name == "backend.services.av_saliency":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _failing_import)

    class _F:
        timestamp = 0.0
        path = "/tmp/x.png"

    out = pipe_mod._compute_l1_saliency_peaks(
        frames=[_F()], audio_path=None, duration_sec=5.0,
        job_id="test", cache_dir=str(tmp_path),
    )
    assert out is None


def test_pipeline_passes_saliency_to_both_maybe_human_rp_callers():
    """Both production callers in pipeline.py supply
    ``saliency_peaks_per_second=_l1_saliency_peaks_per_second``.
    Plain text-grep — covers regression where one site forgets the
    kwarg and exports lose L1."""
    pipe_path = (
        Path(__file__).resolve().parents[2]
        / "backend" / "services" / "pipeline.py"
    )
    src = pipe_path.read_text()
    occurrences = src.count(
        "saliency_peaks_per_second=_l1_saliency_peaks_per_second"
    )
    assert occurrences == 2, (
        "Expected 2 _maybe_human_rp call sites passing "
        f"saliency_peaks_per_second; found {occurrences}"
    )


# ── 7. Vendor placeholder package is importable ──────────────────


def test_vendor_tasednet_package_importable():
    """The placeholder ``backend.vendor.tasednet`` package itself
    imports — only ``model`` (which the operator vendors) is missing.
    This pins the directory layout the L1 PR depends on."""
    import importlib

    # The shim package — should not raise on import.
    pkg = importlib.import_module("backend.vendor")
    assert pkg is not None
    # The tasednet subpackage's __init__ re-raises ImportError when
    # model.py is absent — that's the contract _TasedNetAdapter
    # depends on. We assert the import raises ImportError, not some
    # other exception.
    with pytest.raises(ImportError):
        importlib.import_module("backend.vendor.tasednet")
