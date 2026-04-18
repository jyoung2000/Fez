"""Unified configuration for the gaming-aware reframing mode.

Wraps the previously-added per-feature env vars (center weight,
pan weight, crosshair weight, etc.) behind a single
:class:`GamingReframeConfig` dataclass and a new
``USE_GAMING_REFRAME`` feature flag. Gaming mode composes with
the existing AutoFlip / L1 reframing path — it does not replace
it — so it requires ``USE_AUTOFLIP_REFRAME`` to be enabled as
well. :func:`load_gaming_reframe_config` emits a warning when
called with gaming enabled but AutoFlip off.

This module is deliberately free of heavyweight imports so it's
safe to load from config/settings paths without dragging in the
cv2 / scipy stack.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────── Feature flag ────────────────────


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


USE_GAMING_REFRAME = _env_bool("USE_GAMING_REFRAME", default=False)
USE_AUTOFLIP_REFRAME = _env_bool("USE_AUTOFLIP_REFRAME", default=True)


# ──────────────────── Config dataclass ────────────────────


@dataclass
class GamingReframeConfig:
    """All tunables for the gaming reframe mode.

    Every field has a production-safe default matching the
    task-spec values (Phase 5). Env-var overrides are read in
    :func:`from_env` so operators can tweak without a redeploy.
    """

    # Master toggle — mirrors ``USE_GAMING_REFRAME`` so callers
    # can pass a single config object around.
    enabled: bool = False

    # ── Phase 2 signal-stream weights ──
    center_anchor_weight: float = 0.4
    gaming_action_scale: float = 1.0
    hud_glance_peak_weight: float = 1.2
    hud_glance_decay_tau_s: float = 1.0
    hud_glance_hold_s: float = 0.3

    # ── Phase 3 trigger tunables ──
    pixel_delta_threshold: float = 8.0  # 8-bit MAD inside HUD bbox
    pixel_delta_audio_boost: float = 2.0
    audio_peak_sigma: float = 2.5
    audio_peak_window_ms: float = 200.0
    scheduled_glance_interval_s: float = 10.0
    glance_refractory_s: float = 2.0  # min gap between consecutive
                                      # glances at the same region

    # ── Phase 4 L1 solver knobs ──
    lambda_center: float = 0.4
    max_glance_deviation_pct: float = 0.65  # clamp as fraction of
                                            # the available pan range
    motion_stabilization_threshold_percent: float = 0.65

    # ── Phase 1 HUD-detection knobs ──
    hud_detection_score_threshold: float = 0.55
    hud_detection_sample_count: int = 24
    hud_min_area_pct: float = 0.003   # 0.3 %
    hud_max_area_pct: float = 0.08    # 8 %
    hud_edge_margin_pct: float = 0.15  # HUD must touch or be within
                                       # 15 % of a frame edge
    hud_downsample_width: int = 320

    @classmethod
    def from_env(cls) -> "GamingReframeConfig":
        """Build a config from the environment, falling back to
        the dataclass defaults for any unset variable."""

        def _f(name: str, default: float) -> float:
            raw = os.environ.get(name)
            if raw is None:
                return default
            try:
                return float(raw)
            except ValueError:
                logger.warning(
                    "GamingReframeConfig: env var %s=%r not a float "
                    "— falling back to default %s",
                    name, raw, default,
                )
                return default

        def _i(name: str, default: int) -> int:
            raw = os.environ.get(name)
            if raw is None:
                return default
            try:
                return int(raw)
            except ValueError:
                logger.warning(
                    "GamingReframeConfig: env var %s=%r not an int "
                    "— falling back to default %s",
                    name, raw, default,
                )
                return default

        return cls(
            enabled=USE_GAMING_REFRAME,
            center_anchor_weight=_f("GAMING_CENTER_ANCHOR_WEIGHT", 0.4),
            gaming_action_scale=_f("GAMING_ACTION_SCALE", 1.0),
            hud_glance_peak_weight=_f("GAMING_HUD_GLANCE_PEAK_WEIGHT", 1.2),
            hud_glance_decay_tau_s=_f("GAMING_HUD_GLANCE_DECAY_TAU_S", 1.0),
            hud_glance_hold_s=_f("GAMING_HUD_GLANCE_HOLD_S", 0.3),
            pixel_delta_threshold=_f("GAMING_PIXEL_DELTA_THRESHOLD", 8.0),
            pixel_delta_audio_boost=_f("GAMING_PIXEL_DELTA_AUDIO_BOOST", 2.0),
            audio_peak_sigma=_f("GAMING_AUDIO_PEAK_SIGMA", 2.5),
            audio_peak_window_ms=_f("GAMING_AUDIO_PEAK_WINDOW_MS", 200.0),
            scheduled_glance_interval_s=_f("GAMING_SCHEDULED_GLANCE_INTERVAL_S", 10.0),
            glance_refractory_s=_f("GAMING_GLANCE_REFRACTORY_S", 2.0),
            lambda_center=_f("GAMING_LAMBDA_CENTER", 0.4),
            max_glance_deviation_pct=_f("GAMING_MAX_GLANCE_DEVIATION_PCT", 0.65),
            motion_stabilization_threshold_percent=_f(
                "GAMING_MOTION_STABILIZATION_THRESHOLD_PCT", 0.65,
            ),
            hud_detection_score_threshold=_f(
                "GAMING_HUD_DETECTION_SCORE_THRESHOLD", 0.55,
            ),
            hud_detection_sample_count=_i(
                "GAMING_HUD_DETECTION_SAMPLE_COUNT", 24,
            ),
            hud_min_area_pct=_f("GAMING_HUD_MIN_AREA_PCT", 0.003),
            hud_max_area_pct=_f("GAMING_HUD_MAX_AREA_PCT", 0.08),
            hud_edge_margin_pct=_f("GAMING_HUD_EDGE_MARGIN_PCT", 0.15),
            hud_downsample_width=_i("GAMING_HUD_DOWNSAMPLE_WIDTH", 320),
        )

    def validate(self) -> list[str]:
        """Return a list of warning strings for suspicious values."""
        warnings: list[str] = []
        if self.center_anchor_weight < 0:
            warnings.append("center_anchor_weight must be >= 0")
        if not 0.0 <= self.max_glance_deviation_pct <= 1.0:
            warnings.append(
                "max_glance_deviation_pct must be in [0, 1]; got "
                f"{self.max_glance_deviation_pct}"
            )
        if self.hud_min_area_pct >= self.hud_max_area_pct:
            warnings.append(
                "hud_min_area_pct must be < hud_max_area_pct"
            )
        if self.hud_glance_decay_tau_s <= 0:
            warnings.append("hud_glance_decay_tau_s must be > 0")
        if self.hud_detection_sample_count < 4:
            warnings.append(
                "hud_detection_sample_count < 4 produces noisy masks"
            )
        return warnings


def load_gaming_reframe_config(
    *,
    autoflip_enabled: Optional[bool] = None,
) -> GamingReframeConfig:
    """Build and validate the active gaming reframe config.

    Args:
        autoflip_enabled: Override the ``USE_AUTOFLIP_REFRAME`` env
            read (useful in tests). When None, the module-level
            env flag is used.

    Returns:
        The constructed :class:`GamingReframeConfig`. Logs a
        warning and disables the gaming mode entirely when
        ``USE_GAMING_REFRAME`` is on but AutoFlip mode is off
        (gaming mode builds on top of the AutoFlip L1 solver).
    """
    cfg = GamingReframeConfig.from_env()

    af_on = (
        USE_AUTOFLIP_REFRAME if autoflip_enabled is None
        else bool(autoflip_enabled)
    )
    if cfg.enabled and not af_on:
        logger.warning(
            "USE_GAMING_REFRAME=1 but USE_AUTOFLIP_REFRAME=0 — "
            "gaming mode requires the AutoFlip L1 solver. "
            "Disabling gaming mode for this run.",
        )
        cfg.enabled = False

    for msg in cfg.validate():
        logger.warning("GamingReframeConfig: %s", msg)

    return cfg
