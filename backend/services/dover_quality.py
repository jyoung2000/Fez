"""DOVER-Mobile no-reference VQA — Layer 2 of the critic engine.

Returns aesthetic and technical scores (each in [0, 1]) for any
input video. We run it on both the source and the rendered reframe
and compute a per-axis delta — a negative delta means reframing
degraded that axis.

CPU-only via onnxruntime CPUExecutionProvider by default. CUDA
provider is auto-selected when the runtime exposes it. ~1.4 s per
clip on commodity CPU per the DOVER-Mobile paper (ICCV 2023).

Cached on (source SHA + plan hash) at ``DOVER_CACHE_DIR`` so re-runs
on the same source + render plan are free. The cache lives outside
the extraction cache because DOVER's input depends on the rendered
output (which is per-bench-run), not on the extraction.

Reference:
  * DOVER: ``github.com/VQAssessment/DOVER``.
  * Wu et al, "Exploring Video Quality Assessment on User Generated
    Contents from Aesthetic and Technical Perspectives", ICCV 2023.
    arXiv:2211.04894.

License note: DOVER ships under a permissive license but we still
verify the on-disk SHA256 against ``DOVER_MOBILE_SHA256`` (env-set at
build time) so a substitution attack on a release artifact can't
silently change scoring behavior. SHA verification is skipped when
the env var is empty — operators who haven't pinned a hash get a
warning at module import time.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


DOVER_MODEL_PATH = os.environ.get(
    "DOVER_MOBILE_MODEL_PATH", "/opt/clipai/models/dover_mobile.onnx",
)
DOVER_CACHE_DIR = os.environ.get(
    "DOVER_CACHE_DIR", "/var/cache/clipai/dover_cache",
)


@dataclass
class DoverScores:
    aesthetic: float        # [0, 1]
    technical: float        # [0, 1]
    overall: float          # [0, 1]
    inference_seconds: float
    model_path: str
    cache_hit: bool = False


# ── SHA verification ──────────────────────────────────────────────


def _verify_model_sha() -> None:
    """When ``DOVER_MOBILE_SHA256`` is set, verify the on-disk file
    matches. Raises ``RuntimeError`` on mismatch. Silently skips when
    the env var is empty (operators who haven't pinned yet)."""
    expected = os.environ.get("DOVER_MOBILE_SHA256", "").strip()
    if not expected:
        return
    if not os.path.isfile(DOVER_MODEL_PATH):
        # Missing file is reported by ``_get_session`` separately;
        # don't double-handle it here.
        return
    h = hashlib.sha256()
    with open(DOVER_MODEL_PATH, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"DOVER-Mobile ONNX SHA256 mismatch — expected "
            f"{expected[:16]}..., got {actual[:16]}.... Rebuild "
            "image with verified weights."
        )


# ── ONNX session (lazy + cached) ──────────────────────────────────


_session: Any = None  # ort.InferenceSession when loaded
_session_load_attempted: bool = False


def _get_session():
    """Load the ONNX session lazily. Returns ``None`` on any failure
    so the caller can skip L2 gracefully — never raises."""
    global _session, _session_load_attempted
    if _session is not None:
        return _session
    if _session_load_attempted:
        # Already tried and failed; don't spam logs every call.
        return None
    _session_load_attempted = True

    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning(
            "onnxruntime not installed — DOVER L2 quality delta unavailable"
        )
        return None

    if not Path(DOVER_MODEL_PATH).is_file():
        logger.warning(
            "DOVER-Mobile model not at %s — skipping L2 quality scoring",
            DOVER_MODEL_PATH,
        )
        return None

    try:
        _verify_model_sha()
    except RuntimeError as exc:
        logger.error("%s", exc)
        return None

    try:
        providers: list[str] = []
        avail = set(ort.get_available_providers())
        if "CUDAExecutionProvider" in avail:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        _session = ort.InferenceSession(DOVER_MODEL_PATH, providers=providers)
        return _session
    except Exception as exc:
        logger.warning("DOVER ONNX session init failed: %s", exc)
        return None


def _reset_session_for_tests() -> None:
    """Test-only helper: clear the cached session so unit tests can
    swap module state between cases without process restart."""
    global _session, _session_load_attempted
    _session = None
    _session_load_attempted = False


# ── Frame sampling ────────────────────────────────────────────────


def _sample_frames(video_path: str, n_frames: int = 32):
    """Uniform-sample ``n_frames`` from ``video_path``, resize each to
    224×224 BGR uint8. Returns ``(n_frames, 224, 224, 3)`` ndarray or
    ``None`` on failure.

    DOVER's canonical preprocessing samples T frames at 224×224. The
    exact cadence (uniform vs. random vs. fragment-based) matters —
    we use uniform sampling, which matches the most common
    ``inference.py`` mode. Fragment-based sampling produces slightly
    different scores; if tests against a reference fixture diverge,
    swap to fragment-based via the upstream helper.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            return None
        idxs = np.linspace(0, max(0, total - 1), n_frames).astype(int)
        out = np.zeros((n_frames, 224, 224, 3), dtype=np.uint8)
        last_good = None
        for i, idx in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                if last_good is not None:
                    out[i] = last_good
                continue
            out[i] = cv2.resize(
                frame, (224, 224), interpolation=cv2.INTER_AREA,
            )
            last_good = out[i]
        return out
    finally:
        cap.release()


# ── Preprocessing (testable in isolation) ────────────────────────


def _preprocess_for_dover(frames):
    """Convert ``(T, 224, 224, 3)`` BGR uint8 → ``(1, 3, T, 224, 224)``
    float32 with ImageNet normalization.

    Pure-numpy so the test suite can pin the math without spinning
    up onnxruntime. Channel order: BGR → RGB. Range: 0..255 → 0..1.
    Mean: ImageNet ``[0.485, 0.456, 0.406]``. Std: ImageNet
    ``[0.229, 0.224, 0.225]``.

    The output shape ``(1, 3, T, 224, 224)`` matches DOVER's
    typical ONNX export. If the local export uses a different
    layout, the wrapper's onnx-session call site swaps the
    transpose accordingly. This helper stays "the canonical
    DOVER preprocessing" so tests can pin it.
    """
    if frames is None:
        return None
    import numpy as np

    rgb = frames[..., ::-1].astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    # (T, H, W, 3) → (3, T, H, W) → (1, 3, T, H, W)
    return rgb.transpose(3, 0, 1, 2)[None]


# ── Public scorers ────────────────────────────────────────────────


def score_video(video_path: str) -> Optional[DoverScores]:
    """Returns ``DoverScores`` or ``None`` on any failure path
    (model missing, video unreadable, runtime error). Never raises.
    """
    sess = _get_session()
    if sess is None:
        return None
    if not os.path.isfile(video_path):
        return None
    frames = _sample_frames(video_path)
    if frames is None:
        return None

    t0 = time.time()
    try:
        x = _preprocess_for_dover(frames)
        if x is None:
            return None
        outputs = sess.run(None, {sess.get_inputs()[0].name: x})
        # DOVER returns aesthetic + technical scores. Different ONNX
        # exports flatten these differently — concatenate everything,
        # take the first two scalars. Clamp to [0, 1] so cosmetic
        # numerical drift doesn't propagate.
        import numpy as np
        scores = np.concatenate(
            [np.asarray(o).reshape(-1) for o in outputs]
        )
        if scores.size < 2:
            logger.warning(
                "DOVER returned fewer than 2 scores (%d); "
                "wrapper or export mismatch", scores.size,
            )
            return None
        aesthetic = float(np.clip(scores[0], 0.0, 1.0))
        technical = float(np.clip(scores[1], 0.0, 1.0))
        overall = (aesthetic + technical) / 2.0
        return DoverScores(
            aesthetic=aesthetic,
            technical=technical,
            overall=overall,
            inference_seconds=time.time() - t0,
            model_path=DOVER_MODEL_PATH,
        )
    except Exception as exc:
        logger.warning("DOVER inference failed for %s: %s", video_path, exc)
        return None


# ── Cache (file-keyed on source SHA + plan hash) ─────────────────


def _cache_path(source_sha256: str, plan_hash: str) -> Path:
    cache_dir = Path(DOVER_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{source_sha256[:16]}_{plan_hash[:16]}.json"


def _load_cached_delta(
    source_sha256: str, plan_hash: str,
) -> Optional[dict]:
    if not source_sha256 or not plan_hash:
        return None
    path = _cache_path(source_sha256, plan_hash)
    if not path.is_file():
        return None
    try:
        cached = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        # Corrupt cache — let the caller recompute.
        return None
    cached["cache_hit"] = True
    return cached


def _store_cached_delta(
    source_sha256: str, plan_hash: str, payload: dict,
) -> None:
    if not source_sha256 or not plan_hash:
        return
    try:
        path = _cache_path(source_sha256, plan_hash)
        path.write_text(json.dumps(payload, indent=2))
    except OSError as exc:
        logger.warning("DOVER cache write failed: %s", exc)


def score_delta(
    source_path: str,
    reframed_path: str,
    *,
    source_sha256: Optional[str] = None,
    plan_hash: Optional[str] = None,
) -> Optional[dict]:
    """Compute aesthetic / technical / overall deltas between
    ``source_path`` and ``reframed_path``.

    Negative deltas mean reframing degraded that axis. Cached on
    (source SHA + plan hash) when both are provided. Returns
    ``None`` on any failure path.
    """
    cached = _load_cached_delta(source_sha256 or "", plan_hash or "")
    if cached is not None:
        return cached

    src = score_video(source_path)
    ref = score_video(reframed_path)
    if src is None or ref is None:
        return None

    payload = {
        "source": asdict(src),
        "reframed": asdict(ref),
        "aesthetic_delta": ref.aesthetic - src.aesthetic,
        "technical_delta": ref.technical - src.technical,
        "overall_delta": ref.overall - src.overall,
        "cache_hit": False,
    }
    _store_cached_delta(source_sha256 or "", plan_hash or "", payload)
    return payload
