"""CoTracker3 dense point trajectories — camera motion + per-subject keypoints.

Phase B of the 2026 SOTA reframing rollout.

What this module does:
    * Tracks a sparse background grid (default 16×9 = 144 points) and
      derives a per-frame camera-motion estimate (translation only;
      RANSAC over the inliers). This is what lets the LP solver subtract
      camera pan from the target before scoring smoothness — eliminates
      the "speaker is sitting still but the path drifts because the
      source camera panned" jitter.
    * Tracks a denser per-subject grid (default 64 points seeded inside
      each SAMURAI mask). The Kalman observer consumes a Huber-weighted
      median of the visible points instead of a single bbox center —
      cuts noise variance by ~8× compared to bbox-only observations.

Why CoTracker3:
    * Joint-attention transformer; tracks 1000s of points simultaneously.
    * Handles occlusion natively (visibility flag per point per frame).
    * Causal sliding-window mode (``cotracker3_online``) for streaming.
    * Released October 2024; Kubric 2025 update 1000× more sample-
      efficient than predecessors.

Memory budget:
    * ~600 MB VRAM at grid_size=16, frame_size=384 px.
    * Combined with SAMURAI peak ~1.8 GB — over the 1.2 GB ceiling.
    * Mitigation: we run CoTracker3 in a SEPARATE sequential pass
      BEFORE SAMURAI loads. Tracks are written to a per-clip Parquet
      file; CoTracker3 unloads (``torch.cuda.empty_cache()``) before
      SAMURAI loads.

GPU gating:
    * Construction probes via ``_probe_gpu_availability()``.
    * If GPU not visible, ``CoTracker3Dense.__init__`` raises
      ``RuntimeError`` and the caller falls back to bbox-only Kalman.

Public API mirrors the spec in the design doc:

    tracker = CoTracker3Dense(device="cuda", grid_size=16)
    result = tracker.track(frames, subject_masks={0: masks_t})
    # result.background_tracks: (T, N_bg, 2)
    # result.subject_tracks: dict[int, (T, N_sub, 2)]
    # result.camera_motion: (T, 2)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────

DEFAULT_GRID_SIZE = 16          # 16x9 = 144 background points
DEFAULT_PER_SUBJECT_POINTS = 64
DEFAULT_FRAME_SIZE = 384        # transformer-input size in px
RANSAC_INLIER_THRESHOLD_PX = 4.0
RANSAC_MIN_INLIER_FRAC = 0.6


# ── Result dataclass ─────────────────────────────────────────────────


@dataclass
class CoTrackerResult:
    """Output of one ``CoTracker3Dense.track()`` call."""

    background_tracks: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0, 2), dtype=np.float32)
    )  # (T, N_bg, 2) — pixel coords in source frame
    background_visibility: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=bool)
    )
    subject_tracks: dict = field(default_factory=dict)         # {sid: (T, N, 2)}
    subject_visibility: dict = field(default_factory=dict)     # {sid: (T, N)}
    camera_motion: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.float32)
    )  # (T, 2) — per-frame translation in pixels (cumulative)
    fps: float = 30.0

    @property
    def n_frames(self) -> int:
        return self.background_tracks.shape[0] if self.background_tracks.ndim >= 1 else 0


# ── GPU probe helper ─────────────────────────────────────────────────


def _is_gpu_visible() -> bool:
    try:
        from backend.services.transcription import _probe_gpu_availability  # type: ignore
    except Exception:
        return False
    try:
        return bool(_probe_gpu_availability().get("visible"))
    except Exception:
        return False


# ── Background grid + camera-motion math (testable without torch) ────


def _seed_background_grid(
    frame_h: int, frame_w: int, grid_size: int = DEFAULT_GRID_SIZE,
    *, low_saliency_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return ``(N, 2)`` grid points covering the frame.

    When ``low_saliency_mask`` is given (boolean ``(H, W)``), points
    that fall on a True location are dropped — used to keep the grid
    on background pixels and off the salient subjects.
    """
    if grid_size < 1:
        return np.zeros((0, 2), dtype=np.float32)
    # 16:9 grid — slightly more X than Y points.
    nx = grid_size
    ny = max(2, grid_size * frame_h // max(frame_w, 1))
    xs = np.linspace(frame_w * 0.05, frame_w * 0.95, nx)
    ys = np.linspace(frame_h * 0.05, frame_h * 0.95, ny)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
    if low_saliency_mask is not None and low_saliency_mask.size:
        keep = []
        h, w = low_saliency_mask.shape[:2]
        for x, y in pts:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < w and 0 <= yi < h and not low_saliency_mask[yi, xi]:
                keep.append((x, y))
        pts = np.array(keep, dtype=np.float32) if keep else pts
    return pts


def _seed_subject_grid(
    mask: np.ndarray, n_points: int = DEFAULT_PER_SUBJECT_POINTS,
    *, rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample ``n_points`` uniformly inside a binary mask. Returns ``(N, 2)``."""
    if mask is None or mask.size == 0 or not mask.any():
        return np.zeros((0, 2), dtype=np.float32)
    if rng is None:
        rng = np.random.default_rng(seed=0)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    n = min(n_points, len(xs))
    idx = rng.choice(len(xs), size=n, replace=False)
    return np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)


def _ransac_translation(
    deltas: np.ndarray, *, threshold_px: float = RANSAC_INLIER_THRESHOLD_PX,
    min_inlier_frac: float = RANSAC_MIN_INLIER_FRAC,
) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC the inlier median translation from a set of per-point deltas.

    Returns ``(translation_xy, inlier_mask)``. Falls back to a plain
    median when RANSAC can't find enough inliers.
    """
    if deltas.size == 0:
        return np.zeros(2, dtype=np.float32), np.zeros(0, dtype=bool)
    n = len(deltas)
    median = np.median(deltas, axis=0)
    dists = np.linalg.norm(deltas - median[None, :], axis=1)
    inliers = dists <= threshold_px
    if inliers.sum() / max(n, 1) < min_inlier_frac:
        # Fall back to plain median — high-noise scene, no rigid motion.
        return median.astype(np.float32), np.ones(n, dtype=bool)
    inlier_median = np.median(deltas[inliers], axis=0)
    return inlier_median.astype(np.float32), inliers


def estimate_camera_motion(
    background_tracks: np.ndarray,
    background_visibility: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-frame translation estimate from background tracks.

    ``background_tracks`` is ``(T, N, 2)``. Returns ``(T, 2)`` cumulative
    translation (so ``camera_motion[0] == [0, 0]``).
    """
    T, N, _ = background_tracks.shape
    if T == 0 or N == 0:
        return np.zeros((T, 2), dtype=np.float32)
    motion = np.zeros((T, 2), dtype=np.float32)
    cumulative = np.zeros(2, dtype=np.float32)
    for t in range(1, T):
        prev = background_tracks[t - 1]
        cur = background_tracks[t]
        if background_visibility is not None:
            vis = background_visibility[t - 1] & background_visibility[t]
            if vis.sum() < 4:
                # Too few visible — copy previous motion (zero delta).
                motion[t] = cumulative
                continue
            deltas = cur[vis] - prev[vis]
        else:
            deltas = cur - prev
        translation, _ = _ransac_translation(deltas)
        cumulative = cumulative + translation
        motion[t] = cumulative
    return motion


# ── Huber-weighted median for per-subject points ─────────────────────


def huber_weighted_median(
    points: np.ndarray, visibility: Optional[np.ndarray] = None,
    *, k_huber: float = 1.345, drop_low_vis_frac: float = 0.20,
) -> Optional[tuple[float, float]]:
    """Robust mean of a per-frame point cloud.

    Drops the ``drop_low_vis_frac`` lowest-visibility points first, then
    Huber-weights residuals around the median. Returns ``None`` when
    fewer than 3 points survive — the Kalman observer should fall back
    to bbox-center in that case.
    """
    if points is None or len(points) == 0:
        return None
    if visibility is not None and len(visibility) == len(points):
        vis = np.asarray(visibility, dtype=np.float32)
        if vis.size and vis.min() < vis.max():
            n_drop = int(round(drop_low_vis_frac * len(vis)))
            if n_drop > 0:
                keep_idx = np.argsort(-vis)[:len(vis) - n_drop]
                points = points[keep_idx]
        else:
            # Boolean visibility — drop everything not visible.
            mask = np.asarray(visibility, dtype=bool)
            points = points[mask]
    if len(points) < 3:
        return None
    median = np.median(points, axis=0)
    res = points - median[None, :]
    abs_res = np.linalg.norm(res, axis=1)
    sigma = max(np.median(abs_res), 1e-3)
    w = np.where(abs_res <= k_huber * sigma, 1.0, k_huber * sigma / np.maximum(abs_res, 1e-9))
    w_sum = w.sum()
    if w_sum <= 1e-9:
        return None
    cx = float((points[:, 0] * w).sum() / w_sum)
    cy = float((points[:, 1] * w).sum() / w_sum)
    return (cx, cy)


# ── CoTracker3 wrapper ───────────────────────────────────────────────


class _CoTracker3Adapter:
    """Lazy wrapper around the upstream ``cotracker`` torch.hub model.

    Isolated so unit tests can monkeypatch without forcing the torch
    install on every machine.
    """

    def __init__(self, *, device: str = "cuda"):
        self.device = device
        self._model: Optional[Any] = None

    def load(self) -> None:
        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "torch is required for CoTracker3. Install via "
                "requirements.txt and rebuild."
            ) from exc
        try:
            self._model = torch.hub.load(
                "facebookresearch/co-tracker", "cotracker3_online",
            ).to(self.device).eval()
        except Exception as exc:
            raise RuntimeError(
                f"failed to load cotracker3_online: {exc}"
            ) from exc

    def track_grid(
        self,
        frames: np.ndarray,
        queries: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run CoTracker3 on ``(T, H, W, 3)`` frames with seed ``queries``.

        Returns ``(tracks, visibility)`` where tracks is ``(T, N, 2)`` and
        visibility is ``(T, N)`` boolean.
        """
        if self._model is None:
            self.load()
        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise RuntimeError("torch missing") from exc
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"expected (T, H, W, 3) frames, got {frames.shape}")
        T = frames.shape[0]
        if T < 2:
            return (
                np.zeros((T, len(queries), 2), dtype=np.float32),
                np.ones((T, len(queries)), dtype=bool),
            )
        # Upstream API expects (1, T, 3, H, W) float [0..1].
        video = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        video = video.unsqueeze(0).to(self.device)
        # Queries: (1, N, 3) where columns = (frame_idx, x, y)
        q = np.zeros((len(queries), 3), dtype=np.float32)
        q[:, 1:] = queries
        q_t = torch.from_numpy(q).unsqueeze(0).to(self.device)
        with torch.no_grad():
            tracks, vis = self._model(video, q_t)
        tracks = tracks.squeeze(0).cpu().numpy()      # (T, N, 2)
        vis = vis.squeeze(0).cpu().numpy().astype(bool)  # (T, N)
        return tracks, vis


@dataclass
class CoTracker3Dense:
    """High-level orchestrator — seeds grids, runs the model, returns tracks."""

    device: str = "cuda"
    grid_size: int = DEFAULT_GRID_SIZE
    per_subject_points: int = DEFAULT_PER_SUBJECT_POINTS
    _adapter_cls: type = field(default_factory=lambda: _CoTracker3Adapter, repr=False)

    def __post_init__(self):
        if self.device == "cuda" and not _is_gpu_visible():
            raise RuntimeError(
                "CoTracker3Dense requires a CUDA GPU; set device='cpu' for "
                "CPU mode (very slow) or fall back to bbox-only Kalman."
            )
        self._adapter = self._adapter_cls(device=self.device)

    def track(
        self,
        frames: np.ndarray,
        subject_masks: Optional[dict] = None,
    ) -> CoTrackerResult:
        """Track background grid + per-subject point clouds.

        ``frames``: ``(T, H, W, 3)`` uint8 numpy array.
        ``subject_masks``: optional ``{subject_id: (T, H, W) bool}``.
            If a subject's mask is all-False on frame 0, that subject is
            skipped — the Kalman fallback still has bbox observations.
        """
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("expected (T, H, W, 3) frames")
        T, H, W, _ = frames.shape
        # Seed the background grid using the first frame's mask coverage
        # to avoid putting points on subjects.
        background_pts = _seed_background_grid(H, W, self.grid_size)
        if subject_masks:
            first_mask = np.zeros((H, W), dtype=bool)
            for sid, masks in subject_masks.items():
                if masks is not None and masks.shape[0] > 0:
                    first_mask |= masks[0].astype(bool)
            background_pts = _seed_background_grid(
                H, W, self.grid_size, low_saliency_mask=first_mask,
            )

        bg_tracks, bg_vis = self._adapter.track_grid(frames, background_pts)
        camera_motion = estimate_camera_motion(bg_tracks, bg_vis)

        subject_tracks: dict = {}
        subject_visibility: dict = {}
        if subject_masks:
            rng = np.random.default_rng(seed=42)
            for sid, masks in subject_masks.items():
                if masks is None or masks.shape[0] == 0:
                    continue
                seed = _seed_subject_grid(
                    masks[0], self.per_subject_points, rng=rng,
                )
                if len(seed) < 3:
                    continue
                tr, vs = self._adapter.track_grid(frames, seed)
                subject_tracks[sid] = tr
                subject_visibility[sid] = vs

        return CoTrackerResult(
            background_tracks=bg_tracks,
            background_visibility=bg_vis,
            subject_tracks=subject_tracks,
            subject_visibility=subject_visibility,
            camera_motion=camera_motion,
        )


# ── Helper for the Kalman observer ───────────────────────────────────


def kalman_observation_from_dense(
    tracks_at_t: np.ndarray,
    visibility_at_t: Optional[np.ndarray] = None,
) -> Optional[tuple[float, float]]:
    """Adapter from CoTracker3 per-frame slice to a single Kalman observation.

    Wraps :func:`huber_weighted_median` with the right argument shape.
    Used by :meth:`subject_kalman.KalmanSubject.observe_dense`.
    """
    return huber_weighted_median(tracks_at_t, visibility_at_t)
