"""Layer 2 (DOVER-Mobile) contract.

Pins the wrapper's behavior on every failure path so the runtime
gracefully reports "L2 unavailable" instead of raising. Real DOVER
inference isn't exercised here — that runs on the homelab against a
baked ONNX. The math, cache, SHA verification, and verdict gate all
live in pure-Python paths the sandbox can run.

Run:  pytest tests/qa/test_critic_l2_dover.py -v
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# Skip imports that need numpy / cv2 cleanly when those aren't
# available (sandbox misses numpy; CI / homelab Docker have it).
try:
    import numpy  # noqa: F401
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

skip_no_numpy = pytest.mark.skipif(
    not _HAS_NUMPY, reason="numpy required for dover_quality wrapper",
)


# ── 1. Model-missing path ────────────────────────────────────────


def test_dover_returns_none_when_model_missing(tmp_path, monkeypatch):
    """``score_video`` returns None — never raises — when
    ``DOVER_MOBILE_MODEL_PATH`` points at a non-existent file."""
    monkeypatch.setenv(
        "DOVER_MOBILE_MODEL_PATH", str(tmp_path / "absent.onnx"),
    )
    from backend.services import dover_quality as dq
    monkeypatch.setattr(dq, "DOVER_MODEL_PATH", str(tmp_path / "absent.onnx"))
    dq._reset_session_for_tests()
    assert dq._get_session() is None
    assert dq.score_video(str(tmp_path / "anything.mp4")) is None


def test_dover_returns_none_when_onnxruntime_missing(monkeypatch):
    """When ``import onnxruntime`` fails, ``_get_session`` returns
    None and ``score_video`` propagates that as None."""
    from backend.services import dover_quality as dq

    real_import = __builtins__["__import__"] if isinstance(
        __builtins__, dict
    ) else __import__

    def _failing_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    dq._reset_session_for_tests()
    monkeypatch.setattr("builtins.__import__", _failing_import)
    assert dq._get_session() is None


@skip_no_numpy
def test_dover_returns_none_on_unreadable_video(tmp_path, monkeypatch):
    """Pass a path that isn't a video — ``score_video`` returns
    None gracefully."""
    from backend.services import dover_quality as dq

    # Fake a loaded session so we exercise the video-read branch.
    class _FakeSession:
        def get_inputs(self):
            class _I:
                name = "x"
            return [_I()]

        def run(self, *a, **kw):
            import numpy as np
            return [np.array([0.5, 0.5])]

    dq._reset_session_for_tests()
    monkeypatch.setattr(dq, "_get_session", lambda: _FakeSession())
    not_a_video = tmp_path / "not_a_video.txt"
    not_a_video.write_text("hello")
    # _sample_frames returns None for non-video paths.
    assert dq.score_video(str(not_a_video)) is None


@skip_no_numpy
def test_dover_returns_none_on_zero_frame_video(monkeypatch):
    """``_sample_frames`` returning None propagates as None from
    ``score_video`` — not an exception."""
    from backend.services import dover_quality as dq

    class _FakeSession:
        def get_inputs(self):
            class _I:
                name = "x"
            return [_I()]

        def run(self, *a, **kw):
            import numpy as np
            return [np.array([0.5, 0.5])]

    dq._reset_session_for_tests()
    monkeypatch.setattr(dq, "_get_session", lambda: _FakeSession())
    monkeypatch.setattr(dq, "_sample_frames", lambda *a, **kw: None)
    # Need to skip the os.path.isfile check; pretend the file exists.
    monkeypatch.setattr(dq.os.path, "isfile", lambda p: True)
    assert dq.score_video("/tmp/x.mp4") is None


# ── 2. Inference returns clamped scalars ────────────────────────


@skip_no_numpy
def test_dover_score_video_returns_two_axes_in_range(monkeypatch):
    """Mock onnxruntime to return synthetic logits; verify
    aesthetic and technical fall in [0, 1] and overall is the
    mean of the two."""
    from backend.services import dover_quality as dq
    import numpy as np

    class _FakeSession:
        def get_inputs(self):
            class _I:
                name = "x"
            return [_I()]

        def run(self, *a, **kw):
            return [np.array([0.78, 0.86])]

    dq._reset_session_for_tests()
    monkeypatch.setattr(dq, "_get_session", lambda: _FakeSession())
    monkeypatch.setattr(
        dq, "_sample_frames",
        lambda *a, **kw: np.zeros((32, 224, 224, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(dq.os.path, "isfile", lambda p: True)

    out = dq.score_video("/tmp/x.mp4")
    assert out is not None
    assert 0.0 <= out.aesthetic <= 1.0
    assert 0.0 <= out.technical <= 1.0
    assert out.aesthetic == pytest.approx(0.78)
    assert out.technical == pytest.approx(0.86)
    assert out.overall == pytest.approx((0.78 + 0.86) / 2.0)


@skip_no_numpy
def test_dover_clamps_out_of_range_logits(monkeypatch):
    """Sometimes the export produces logits, not sigmoid'd
    probabilities. The wrapper clamps to [0, 1] so the verdict
    gate never sees a -3.4 value that would pass spuriously."""
    from backend.services import dover_quality as dq
    import numpy as np

    class _FakeSession:
        def get_inputs(self):
            class _I:
                name = "x"
            return [_I()]

        def run(self, *a, **kw):
            return [np.array([-2.5, 1.6])]

    dq._reset_session_for_tests()
    monkeypatch.setattr(dq, "_get_session", lambda: _FakeSession())
    monkeypatch.setattr(
        dq, "_sample_frames",
        lambda *a, **kw: np.zeros((32, 224, 224, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(dq.os.path, "isfile", lambda p: True)

    out = dq.score_video("/tmp/x.mp4")
    assert out is not None
    assert out.aesthetic == 0.0  # clamped
    assert out.technical == 1.0  # clamped


# ── 3. Preprocessing math ────────────────────────────────────────


@skip_no_numpy
def test_dover_normalization_is_imagenet_in_rgb_order():
    """The wrapper's ImageNet normalization must:
      * convert BGR → RGB (channel reverse)
      * scale 0..255 → 0..1
      * subtract [0.485, 0.456, 0.406] (ImageNet mean)
      * divide by [0.229, 0.224, 0.225] (ImageNet std)
      * end up in (1, 3, T, 224, 224) layout

    We don't have a real DOVER fixture in the sandbox to compare
    absolute values against, so the test pins the math itself —
    each piece is verified analytically. Diverging from this
    produces score-looking nonsense (see the prompt's
    "If you get stuck" notes).
    """
    from backend.services.dover_quality import _preprocess_for_dover
    import numpy as np

    # Single frame, single channel each, distinct values per axis.
    # Frame is (T=2, H=2, W=2, C=3) BGR. Set all-zero, then poke a
    # known pattern so we can read it back at known indices.
    frames = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    # BGR pixel = [B=0, G=128, R=255]; after BGR→RGB it's [255, 128, 0].
    frames[0, 0, 0] = [0, 128, 255]
    out = _preprocess_for_dover(frames)
    assert out is not None
    # Output shape: (1, 3, T=2, H=2, W=2)
    assert out.shape == (1, 3, 2, 2, 2)
    # Read back the (R, G, B) values at (t=0, h=0, w=0):
    r = out[0, 0, 0, 0, 0]
    g = out[0, 1, 0, 0, 0]
    b = out[0, 2, 0, 0, 0]
    # Expected: each value ÷ 255, then mean-subtracted, std-divided.
    expected_r = (255 / 255.0 - 0.485) / 0.229
    expected_g = (128 / 255.0 - 0.456) / 0.224
    expected_b = (0 / 255.0 - 0.406) / 0.225
    assert r == pytest.approx(expected_r, abs=1e-5)
    assert g == pytest.approx(expected_g, abs=1e-5)
    assert b == pytest.approx(expected_b, abs=1e-5)


def test_dover_preprocess_returns_none_on_none_input():
    from backend.services.dover_quality import _preprocess_for_dover
    assert _preprocess_for_dover(None) is None


# ── 4. score_delta ───────────────────────────────────────────────


def test_dover_delta_combines_source_and_reframed(monkeypatch, tmp_path):
    """Given two synthetic ``DoverScores``, ``score_delta`` returns
    correct math: reframed - source per axis."""
    from backend.services import dover_quality as dq

    src = dq.DoverScores(
        aesthetic=0.80, technical=0.85, overall=0.825,
        inference_seconds=0.0, model_path="x",
    )
    ref = dq.DoverScores(
        aesthetic=0.76, technical=0.82, overall=0.79,
        inference_seconds=0.0, model_path="x",
    )

    # Mock score_video to return src then ref in order.
    calls = {"i": 0}

    def _score(path):
        v = src if calls["i"] == 0 else ref
        calls["i"] += 1
        return v

    monkeypatch.setattr(dq, "score_video", _score)
    monkeypatch.setattr(dq, "DOVER_CACHE_DIR", str(tmp_path / "cache"))

    out = dq.score_delta("/tmp/source.mp4", "/tmp/reframed.mp4")
    assert out is not None
    assert out["aesthetic_delta"] == pytest.approx(-0.04)
    assert out["technical_delta"] == pytest.approx(-0.03)
    assert out["overall_delta"] == pytest.approx(-0.035)
    assert out["cache_hit"] is False


def test_dover_delta_handles_missing_source(monkeypatch, tmp_path):
    """Source returns None → ``score_delta`` returns None even if
    reframed scoring succeeds."""
    from backend.services import dover_quality as dq

    calls = {"i": 0}

    def _score(path):
        # First call (source) fails; second never runs.
        if calls["i"] == 0:
            calls["i"] += 1
            return None
        calls["i"] += 1
        return dq.DoverScores(
            aesthetic=0.5, technical=0.5, overall=0.5,
            inference_seconds=0.0, model_path="x",
        )

    monkeypatch.setattr(dq, "score_video", _score)
    monkeypatch.setattr(dq, "DOVER_CACHE_DIR", str(tmp_path / "cache"))
    assert dq.score_delta("/tmp/x.mp4", "/tmp/y.mp4") is None


# ── 5. Cache (file-keyed) ───────────────────────────────────────


def test_dover_cache_hit_skips_recompute(monkeypatch, tmp_path):
    """Second call with same (source_sha, plan_hash) reads cached
    JSON; cache_hit=True; ``score_video`` not invoked the second time.
    """
    from backend.services import dover_quality as dq

    monkeypatch.setattr(dq, "DOVER_CACHE_DIR", str(tmp_path / "cache"))

    src = dq.DoverScores(
        aesthetic=0.80, technical=0.85, overall=0.825,
        inference_seconds=0.0, model_path="x",
    )
    ref = dq.DoverScores(
        aesthetic=0.76, technical=0.82, overall=0.79,
        inference_seconds=0.0, model_path="x",
    )
    score_calls = {"n": 0}

    def _score(path):
        score_calls["n"] += 1
        if score_calls["n"] % 2 == 1:
            return src
        return ref

    monkeypatch.setattr(dq, "score_video", _score)

    sha = "a" * 64
    plan = "b" * 32
    first = dq.score_delta(
        "/x", "/y", source_sha256=sha, plan_hash=plan,
    )
    assert first is not None
    assert first["cache_hit"] is False
    assert score_calls["n"] == 2

    second = dq.score_delta(
        "/x", "/y", source_sha256=sha, plan_hash=plan,
    )
    assert second is not None
    assert second["cache_hit"] is True
    assert score_calls["n"] == 2  # no additional inference


def test_dover_cache_corrupt_recomputes(monkeypatch, tmp_path):
    """A cache file that exists but isn't valid JSON triggers
    silent recompute, not a crash."""
    from backend.services import dover_quality as dq

    monkeypatch.setattr(dq, "DOVER_CACHE_DIR", str(tmp_path / "cache"))
    sha = "a" * 64
    plan = "b" * 32
    cache_file = dq._cache_path(sha, plan)
    cache_file.write_text("{not_valid_json")

    src = dq.DoverScores(
        aesthetic=0.5, technical=0.6, overall=0.55,
        inference_seconds=0.0, model_path="x",
    )
    ref = dq.DoverScores(
        aesthetic=0.5, technical=0.6, overall=0.55,
        inference_seconds=0.0, model_path="x",
    )
    seq = [src, ref]
    monkeypatch.setattr(dq, "score_video", lambda p: seq.pop(0))

    out = dq.score_delta("/x", "/y", source_sha256=sha, plan_hash=plan)
    assert out is not None
    assert out["cache_hit"] is False


# ── 6. SHA verification ─────────────────────────────────────────


def test_dover_sha_mismatch_disables_layer(tmp_path, monkeypatch):
    """``DOVER_MOBILE_SHA256`` set + actual file SHA differs →
    ``_get_session`` returns None and logs error."""
    from backend.services import dover_quality as dq

    fake_onnx = tmp_path / "fake.onnx"
    fake_onnx.write_bytes(b"hello world")  # short bytes, easy SHA
    real_sha = hashlib.sha256(b"hello world").hexdigest()
    wrong_sha = "0" * 64
    assert wrong_sha != real_sha

    monkeypatch.setattr(dq, "DOVER_MODEL_PATH", str(fake_onnx))
    monkeypatch.setenv("DOVER_MOBILE_SHA256", wrong_sha)
    dq._reset_session_for_tests()

    # Fake onnxruntime so the import succeeds — the mismatch should
    # short-circuit the session load before InferenceSession is called.
    class _FakeOrt:
        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

        class InferenceSession:
            def __init__(self, *a, **kw):
                raise RuntimeError(
                    "InferenceSession should not be called when SHA mismatches"
                )

    monkeypatch.setitem(sys.modules, "onnxruntime", _FakeOrt())
    assert dq._get_session() is None


def test_dover_sha_skipped_when_env_unset(tmp_path, monkeypatch):
    """No env-pinned SHA → ``_verify_model_sha`` returns silently.
    Operators who haven't pinned a hash still get a working layer."""
    from backend.services import dover_quality as dq

    fake_onnx = tmp_path / "fake.onnx"
    fake_onnx.write_bytes(b"hello world")
    monkeypatch.setattr(dq, "DOVER_MODEL_PATH", str(fake_onnx))
    monkeypatch.delenv("DOVER_MOBILE_SHA256", raising=False)

    # Should not raise.
    dq._verify_model_sha()


def test_dover_sha_match_passes(tmp_path, monkeypatch):
    """Correct SHA passes verification."""
    from backend.services import dover_quality as dq

    fake_onnx = tmp_path / "fake.onnx"
    fake_onnx.write_bytes(b"hello world")
    real_sha = hashlib.sha256(b"hello world").hexdigest()

    monkeypatch.setattr(dq, "DOVER_MODEL_PATH", str(fake_onnx))
    monkeypatch.setenv("DOVER_MOBILE_SHA256", real_sha)

    # Should not raise.
    dq._verify_model_sha()


# ── 7. Bench verdict gate ───────────────────────────────────────


def test_quality_delta_downgrades_to_marginal_aesthetic():
    """Aesthetic delta below threshold downgrades PASS → MARGINAL."""
    from backend.scripts.compare_autoflip_vs_clipai import (
        verdict_for, TARGET_ZONES,
    )
    zone = dict(TARGET_ZONES["default"])
    metrics = {
        "n_segments": 10,
        "median_hold_sec": 3.0,
        "segments_under_1s_rate": 0.05,
        "quality_delta_aesthetic": -0.15,
        "quality_delta_technical": -0.02,
    }
    assert verdict_for(metrics, zone) == "MARGINAL"


def test_quality_delta_downgrades_to_marginal_technical():
    """Technical delta below threshold downgrades PASS → MARGINAL."""
    from backend.scripts.compare_autoflip_vs_clipai import (
        verdict_for, TARGET_ZONES,
    )
    zone = dict(TARGET_ZONES["default"])
    metrics = {
        "n_segments": 10,
        "median_hold_sec": 3.0,
        "segments_under_1s_rate": 0.05,
        "quality_delta_aesthetic": -0.02,
        "quality_delta_technical": -0.15,
    }
    assert verdict_for(metrics, zone) == "MARGINAL"


def test_quality_delta_within_threshold_preserves_pass():
    """Both deltas above threshold → PASS holds."""
    from backend.scripts.compare_autoflip_vs_clipai import (
        verdict_for, TARGET_ZONES,
    )
    zone = dict(TARGET_ZONES["default"])
    metrics = {
        "n_segments": 10,
        "median_hold_sec": 3.0,
        "segments_under_1s_rate": 0.05,
        "quality_delta_aesthetic": -0.05,
        "quality_delta_technical": -0.03,
    }
    assert verdict_for(metrics, zone) == "PASS"


def test_quality_delta_absent_is_no_op():
    """Missing quality_delta_* keys → gate is a no-op (PASS holds)."""
    from backend.scripts.compare_autoflip_vs_clipai import (
        verdict_for, TARGET_ZONES,
    )
    zone = dict(TARGET_ZONES["default"])
    metrics = {
        "n_segments": 10,
        "median_hold_sec": 3.0,
        "segments_under_1s_rate": 0.05,
    }
    # No quality_delta_* keys; verdict stays PASS.
    assert verdict_for(metrics, zone) == "PASS"


def test_quality_delta_zone_override_wins():
    """A per-zone ``quality_delta_aesthetic_min`` override beats the
    module-level default. Zone says -0.20 → -0.15 still passes."""
    from backend.scripts.compare_autoflip_vs_clipai import (
        verdict_for, TARGET_ZONES,
    )
    zone = dict(TARGET_ZONES["default"])
    zone["quality_delta_aesthetic_min"] = -0.20
    metrics = {
        "n_segments": 10,
        "median_hold_sec": 3.0,
        "segments_under_1s_rate": 0.05,
        "quality_delta_aesthetic": -0.15,
        "quality_delta_technical": -0.02,
    }
    assert verdict_for(metrics, zone) == "PASS"
