"""Phase 2 — three saliency signal streams for gaming-mode reframing.

The existing reframe pipeline emits saliency regions via
:class:`backend.services.saliency_tracker.SaliencyRegion` — a
per-frame percentage-coordinate dataclass with separate motion
and spatial score fields. Gaming mode builds a trio of streams
on top of that existing infrastructure:

  * **Stream A — Center anchor.** An always-on synthetic region
    covering the middle 20 % × 40 % of the frame at a fixed
    weight (default 0.4). Its purpose is purely mathematical:
    the L1 solver sees a persistent pull toward center so the
    "stay centered" cost is always in play.

  * **Stream B — Action saliency.** A thin wrapper over the
    caller's existing detectors (motion + generic object +
    character). The wrapper rescales weights by
    ``gaming_action_scale`` so operators can tune the
    action-vs-center balance without patching every detector.

  * **Stream C — HUD glance.** Stateful — holds one weight per
    detected :class:`HudRegion` and updates it each frame based
    on the incoming :class:`GlanceTrigger` list. Weights are
    held at peak for ``hud_glance_hold_s``, then decay
    exponentially with time constant ``hud_glance_decay_tau_s``.
    Regions with weight below ``DECAY_FLOOR`` emit no output
    (reduces downstream fusion cost).

Each stream's output is a list of :class:`GamingSaliencyRegion`
— a gaming-local dataclass shaped exactly like the spec
(``bbox``, ``weight``, ``is_required``) so the SignalFusing
integration layer can consume it directly.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

logger = logging.getLogger(__name__)


# ──────────────────── Shared types ────────────────────


StreamKind = Literal["center_anchor", "action", "hud_glance"]


@dataclass
class GamingSaliencyRegion:
    """Per-frame saliency region emitted by a gaming stream.

    Shaped to match the task spec rather than the existing
    percentage-coordinate ``SaliencyRegion`` — callers that
    need to plug into the legacy fusion path can use
    :meth:`to_legacy_pct` to convert.
    """

    bbox: tuple[int, int, int, int]  # pixel coords in the source frame
    weight: float
    is_required: bool = False
    kind: StreamKind = "action"
    frame_width: int = 0
    frame_height: int = 0
    label: str = ""

    def to_legacy_pct(self, timestamp: float = 0.0) -> dict:
        """Convert to a dict matching the legacy SaliencyRegion shape."""
        if self.frame_width <= 0 or self.frame_height <= 0:
            return {}
        x, y, w, h = self.bbox
        return {
            "timestamp": timestamp,
            "x": (x + w / 2.0) / self.frame_width * 100.0,
            "y": (y + h / 2.0) / self.frame_height * 100.0,
            "w": w / self.frame_width * 100.0,
            "h": h / self.frame_height * 100.0,
            "saliency_score": float(self.weight),
            "motion_score": 0.0,
            "spatial_score": float(self.weight),
        }


# ──────────────────── Stream A — Center anchor ────────────────────


def center_anchor_stream(
    frame_w: int,
    frame_h: int,
    *,
    weight: float = 0.4,
    box_w_frac: float = 0.20,
    box_h_frac: float = 0.40,
) -> GamingSaliencyRegion:
    """Return the center-anchor saliency region for a frame.

    The region is a fixed 20 % × 40 % rectangle centered on the
    frame, emitted every frame at a constant weight. It's
    deliberately narrow horizontally (20 %) and tall vertically
    (40 %) so it biases the horizontal pan toward center without
    also clamping vertical freedom — gaming mode's center
    penalty is x-only.
    """
    box_w = max(1, int(round(frame_w * box_w_frac)))
    box_h = max(1, int(round(frame_h * box_h_frac)))
    x = (frame_w - box_w) // 2
    y = (frame_h - box_h) // 2
    return GamingSaliencyRegion(
        bbox=(x, y, box_w, box_h),
        weight=float(weight),
        is_required=False,
        kind="center_anchor",
        frame_width=frame_w,
        frame_height=frame_h,
        label="center_anchor",
    )


# ──────────────────── Stream B — Action saliency ────────────────────


def action_saliency_stream(
    existing_regions: Iterable,
    *,
    frame_w: int,
    frame_h: int,
    gaming_action_scale: float = 1.0,
) -> list[GamingSaliencyRegion]:
    """Pass-through wrapper over the caller's action detectors.

    Accepts any iterable of region-like objects that expose:

      - ``bbox: (x, y, w, h)`` OR ``(x_pct, y_pct, w_pct, h_pct)``
      - ``weight`` / ``saliency_score`` / ``score`` (first found)

    and optionally ``is_required: bool``. Weights are multiplied
    by ``gaming_action_scale`` so operators can rebalance
    action-vs-center without touching upstream detector code.
    """
    out: list[GamingSaliencyRegion] = []
    for r in existing_regions or []:
        bbox = _extract_bbox(r, frame_w, frame_h)
        if bbox is None:
            continue
        w = _extract_weight(r)
        if w is None:
            continue
        out.append(GamingSaliencyRegion(
            bbox=bbox,
            weight=float(w) * float(gaming_action_scale),
            is_required=bool(getattr(r, "is_required", False)),
            kind="action",
            frame_width=frame_w,
            frame_height=frame_h,
            label=getattr(r, "label", "action") or "action",
        ))
    return out


def _extract_bbox(
    r, frame_w: int, frame_h: int,
) -> Optional[tuple[int, int, int, int]]:
    """Pull a pixel-bbox out of any supported region shape."""
    bbox = getattr(r, "bbox", None)
    if bbox is not None:
        try:
            x, y, w, h = bbox
            return (int(x), int(y), int(w), int(h))
        except (TypeError, ValueError):
            pass

    # Legacy SaliencyRegion path — percentage coordinates with
    # (x, y) as centers.
    x_pct = getattr(r, "x", None)
    y_pct = getattr(r, "y", None)
    w_pct = getattr(r, "w", None)
    h_pct = getattr(r, "h", None)
    if None in (x_pct, y_pct, w_pct, h_pct):
        return None
    w_px = max(1, int(round(float(w_pct) / 100.0 * frame_w)))
    h_px = max(1, int(round(float(h_pct) / 100.0 * frame_h)))
    cx_px = float(x_pct) / 100.0 * frame_w
    cy_px = float(y_pct) / 100.0 * frame_h
    x_px = max(0, int(round(cx_px - w_px / 2.0)))
    y_px = max(0, int(round(cy_px - h_px / 2.0)))
    return (x_px, y_px, w_px, h_px)


def _extract_weight(r) -> Optional[float]:
    for attr in ("weight", "saliency_score", "score"):
        val = getattr(r, attr, None)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


# ──────────────────── Stream C — HUD glance ────────────────────


# Weights below this floor are dropped from the output entirely
# — keeps SignalFusing's input size bounded during long idle
# stretches.
DECAY_FLOOR = 0.05


class HudGlanceStream:
    """Stateful per-region glance-weight tracker.

    One instance per scene. On each frame the caller calls
    :meth:`update` with the current frame index and any new
    triggers; the stream applies peak weights, decays older
    weights, and returns the list of regions whose current
    weight is above :data:`DECAY_FLOOR`. Regions below the
    floor emit nothing.

    Hold + decay envelope for a region with a trigger at frame
    ``t_trigger`` (peak weight W):

      * ``[t_trigger, t_trigger + hold_frames]``  → weight = W
      * ``[t_trigger + hold_frames, ∞)``          → weight =
        W · exp(-Δt / τ_frames)

    A new trigger on the same region before the previous decay
    drops below :data:`DECAY_FLOOR` resets the envelope from the
    current weight (takes the max so overlapping triggers
    aren't clipped down).
    """

    def __init__(
        self,
        hud_regions: list,
        *,
        frame_rate: float = 30.0,
        peak_weight: float = 1.2,
        hold_s: float = 0.3,
        decay_tau_s: float = 1.0,
    ):
        self.regions = list(hud_regions)
        self.frame_rate = max(frame_rate, 1.0)
        self.peak_weight = float(peak_weight)
        self.hold_frames = max(1, int(round(hold_s * self.frame_rate)))
        self.tau_frames = max(1.0, float(decay_tau_s * self.frame_rate))
        self._weights: dict[int, float] = {
            id(r): 0.0 for r in self.regions
        }
        self._last_trigger_frame: dict[int, int] = {
            id(r): -10_000_000 for r in self.regions
        }
        self._current_frame: int = -1

    # ── Diagnostics / tests ──

    def weight_for(self, region) -> float:
        return self._weights.get(id(region), 0.0)

    def frames_since_last_trigger(self, region) -> int:
        last = self._last_trigger_frame.get(id(region), -10_000_000)
        if last < 0:
            return 10**9
        return max(0, self._current_frame - last)

    # ── Update loop ──

    def update(
        self,
        frame_idx: int,
        triggers: Optional[list] = None,
    ) -> list[GamingSaliencyRegion]:
        """Advance the stream by one frame.

        Args:
            frame_idx: Current frame index.
            triggers: Zero or more :class:`GlanceTrigger` objects
                hitting THIS frame. Each trigger's
                ``region_id`` must match ``id(region)`` for one
                of the regions this stream was constructed with.

        Returns:
            List of :class:`GamingSaliencyRegion` for every
            region whose current weight is above
            :data:`DECAY_FLOOR`, in decreasing weight order.
        """
        self._current_frame = int(frame_idx)
        incoming = triggers or []

        # Step 1: apply triggers — set weight to the MAX of
        # the current weight and the trigger's peak. Update the
        # last-trigger-frame bookkeeping so hold timing is
        # correct.
        for trig in incoming:
            rid = getattr(trig, "region_id", None)
            if rid is None or rid not in self._weights:
                continue
            peak = float(getattr(trig, "peak_weight", self.peak_weight))
            self._weights[rid] = max(self._weights[rid], peak)
            self._last_trigger_frame[rid] = self._current_frame

        # Step 2: decay each weight. Hold period first, then
        # exponential. Overlapping triggers fall through the
        # max() branch above so an early re-trigger preserves
        # the weight.
        output: list[tuple[float, GamingSaliencyRegion]] = []
        for r in self.regions:
            rid = id(r)
            w = self._weights[rid]
            if w <= 0.0:
                continue
            last = self._last_trigger_frame[rid]
            dt = self._current_frame - last
            if dt < 0:
                continue  # shouldn't happen but be safe
            if dt > self.hold_frames:
                # Decay from (t_trigger + hold) forward. Each
                # frame past the hold divides by exp(1/tau).
                steps_past = dt - self.hold_frames
                w *= math.exp(-steps_past / self.tau_frames)

            self._weights[rid] = w
            if w >= DECAY_FLOOR:
                bbox = tuple(getattr(r, "bbox", (0, 0, 0, 0)))
                fw = int(getattr(r, "frame_width", 0))
                fh = int(getattr(r, "frame_height", 0))
                label = (
                    getattr(r, "semantic_hint", None)
                    or getattr(r, "label", None)
                    or "hud_glance"
                )
                output.append((w, GamingSaliencyRegion(
                    bbox=bbox,  # type: ignore[arg-type]
                    weight=float(w),
                    is_required=False,
                    kind="hud_glance",
                    frame_width=fw,
                    frame_height=fh,
                    label=str(label),
                )))

        output.sort(key=lambda p: p[0], reverse=True)
        return [r for _w, r in output]

    def reset(self) -> None:
        """Drop all per-region weights (e.g. on shot boundary)."""
        for rid in self._weights:
            self._weights[rid] = 0.0
            self._last_trigger_frame[rid] = -10_000_000
        self._current_frame = -1


# ──────────────────── Stream fusion helper ────────────────────


@dataclass
class GamingStreamFrame:
    """Convenience snapshot of all three streams for one frame."""

    timestamp: float
    center: GamingSaliencyRegion
    action: list[GamingSaliencyRegion] = field(default_factory=list)
    hud_glance: list[GamingSaliencyRegion] = field(default_factory=list)

    def all_regions(self) -> list[GamingSaliencyRegion]:
        out = [self.center]
        out.extend(self.action)
        out.extend(self.hud_glance)
        return out
