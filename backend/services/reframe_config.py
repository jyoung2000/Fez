"""Single source of truth for reframing tunables.

Historically every knob that affects reframing — L1 LP weights, ease
durations, dead-zones, thirds-bias sigmas, lead-room offsets, motivated-
zoom amplitudes, saccade durations — lived as a module-level constant
in whichever file first needed it. This created three classes of bug:

  * "Dual-path drift": the AutoFlip segmenter used ``lambda_smooth=0.3``
    while the legacy solver used ``LP_LAMBDA_V=20``. Env A/B tests set
    one but not the other, and outputs silently diverged.
  * Convention drift: ``subject_x`` is pixels in one path, 0-100 percent
    in another, 0-1 normalized in a third.
  * No single config object to pass through the pipeline — every module
    re-reads ``os.environ`` on import, so feature-flag overrides applied
    after import have no effect.

This module consolidates all reframing tunables into a single frozen
``ReframeConfig`` dataclass, plus per-``ClipContentType`` overrides.
Existing modules keep their module-level constants as **fallbacks** for
back-compat; the new 2-D solver, Kalman, event scheduler, motivated
zoom, A/B cut scheduler, and critic loop all accept a ``ReframeConfig``
and respect its values in preference to the legacy constants.

The single coordinate convention for all new code is:

  ``(t, x_frac, y_frac, zoom)`` — ``x_frac``, ``y_frac`` in [0, 1]
  relative to source frame; ``zoom`` a multiplicative scale on the
  content-type default crop width. Pixel conversion happens at the
  FFmpeg / RenderPlan boundary only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Optional


def _env_float(key: str, default: float) -> float:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


# ── Blueprint v2 Phase 1 — Stage 3 importance matrix ───────────────
#
# Single source of truth for how the reframing stack mixes Stage 3
# signals (face, saliency, motion, depth, object) to build the
# per-pixel importance map. Before Phase 1, these weights were
# scattered across ``required_regions.py``, ``genre_refinements.py``,
# ``content_type_config.py``, and several per-genre files.
#
# ``required_region_gain`` is the hard-constraint multiplier
# (blueprint's H = 1000); it must dominate the sum of all soft
# weights by at least 100x so the L1 solver treats required regions
# as hard constraints. ``passive_face_weight`` is the fraction of
# active-speaker weight applied to non-speaking faces in the same
# shot — previously hardcoded at 0.55 in ``required_regions.py``.
@dataclass(frozen=True)
class ImportanceWeights:
    """Stage-3 signal weights per genre."""

    face: float = 1.0
    saliency: float = 0.4
    motion: float = 0.2
    depth: float = 0.0
    object: float = 0.0
    # Hard-constraint gain for required regions (active speakers, text,
    # HUDs). Must dominate the sum of all soft weights by at least 100x.
    required_region_gain: float = 1000.0
    # Passive-face downweight: non-speaking faces get this fraction of
    # the active-speaker weight. Previously hardcoded in required_regions.py.
    passive_face_weight: float = 0.55


# Blueprint v2 Stage 3 matrix. One row per content type. Missing
# entries fall back to ``_IMPORTANCE_MATRIX["generic"]`` via
# ``importance_for``.
_IMPORTANCE_MATRIX: dict[str, "ImportanceWeights"] = {
    # Talking head / single speaker
    "talking_head":        ImportanceWeights(face=1.0, saliency=0.3, motion=0.1, depth=0.0, object=0.0),
    "vlog":                ImportanceWeights(face=0.9, saliency=0.4, motion=0.2, depth=0.0, object=0.0),
    "podcast":             ImportanceWeights(face=1.0, saliency=0.3, motion=0.1),
    "interview":           ImportanceWeights(face=1.0, saliency=0.4, motion=0.2),
    "multi_speaker_panel": ImportanceWeights(face=1.0, saliency=0.4, motion=0.2),

    # Cinematic / narrative
    "cinematic_dialogue":  ImportanceWeights(face=0.8, saliency=0.6, motion=0.2, depth=0.5),
    "narrative":           ImportanceWeights(face=0.8, saliency=0.6, motion=0.2, depth=0.5),

    # Sports
    "sports":              ImportanceWeights(face=0.4, saliency=0.5, motion=0.4, object=1.0),
    "sports_basketball":   ImportanceWeights(face=0.4, saliency=0.5, motion=0.4, object=1.0),
    "sports_racing":       ImportanceWeights(face=0.2, saliency=0.4, motion=0.6, object=1.0),

    # Gameplay
    "gameplay":            ImportanceWeights(face=0.3, saliency=0.7, motion=0.5, object=0.9),
    "gameplay_fps":        ImportanceWeights(face=0.3, saliency=0.6, motion=0.5, object=0.9),
    "gameplay_tps":        ImportanceWeights(face=0.3, saliency=0.7, motion=0.5, object=0.9),
    "gameplay_moba":       ImportanceWeights(face=0.2, saliency=0.8, motion=0.4, object=0.9),
    "gameplay_racing":     ImportanceWeights(face=0.2, saliency=0.6, motion=0.6, object=0.9),

    # Stream (gameplay + webcam)
    "stream":              ImportanceWeights(face=0.6, saliency=0.6, motion=0.4, object=0.7),

    # Animation
    "animation":           ImportanceWeights(face=1.0, saliency=0.8, motion=0.3, object=0.3),
    "anime":               ImportanceWeights(face=1.0, saliency=0.8, motion=0.3, object=0.3),
    "animation_dialogue":  ImportanceWeights(face=1.0, saliency=0.7, motion=0.2),

    # Music
    "music_video":         ImportanceWeights(face=0.6, saliency=0.5, motion=0.4, object=0.2),
    "music_performance":   ImportanceWeights(face=0.6, saliency=0.5, motion=0.4, object=0.2),

    # B-roll / documentary
    "documentary":         ImportanceWeights(face=0.2, saliency=0.9, motion=0.3, depth=0.3),
    "landscape":           ImportanceWeights(face=0.0, saliency=1.0, motion=0.2, depth=0.3),
    "broll":               ImportanceWeights(face=0.3, saliency=0.8, motion=0.3, depth=0.2),

    # Screen-share / tutorial
    "tutorial":            ImportanceWeights(face=0.9, saliency=0.4, motion=0.2),
    "screen_share":        ImportanceWeights(face=0.5, saliency=0.3, motion=0.1, object=0.8),

    # Fallback
    "generic":             ImportanceWeights(),
    "unknown":             ImportanceWeights(),
}


def importance_for(content_type) -> "ImportanceWeights":
    """Return the Stage-3 importance weights for ``content_type``.

    Accepts enum, string, or ``None``. Missing keys fall back to
    ``_IMPORTANCE_MATRIX["generic"]`` — which matches the legacy
    defaults so unknown content types keep their pre-Phase-1 behavior.
    """
    key = getattr(content_type, "value", content_type)
    if key is None:
        return _IMPORTANCE_MATRIX["generic"]
    return _IMPORTANCE_MATRIX.get(str(key), _IMPORTANCE_MATRIX["generic"])


def importance_matrix_as_dict() -> dict[str, dict[str, float]]:
    """Return the full matrix as a plain dict for JSON serialization
    (used by the diagnostics endpoint)."""
    return {
        key: {
            "face": w.face,
            "saliency": w.saliency,
            "motion": w.motion,
            "depth": w.depth,
            "object": w.object,
            "required_region_gain": w.required_region_gain,
            "passive_face_weight": w.passive_face_weight,
        }
        for key, w in _IMPORTANCE_MATRIX.items()
    }


@dataclass(frozen=True)
class ReframeConfig:
    """Frozen bundle of every tunable the reframing stack respects."""

    # ── Master feature flag ─────────────────────────────────────────
    #
    # When enabled, the pipeline prefers the human-reframing path:
    # 2-D solver + Kalman prediction + event-scheduled saccades +
    # motivated zoom + A/B cuts. When disabled, the legacy 1-D path
    # runs unchanged so existing AutoFlip fixtures stay bit-identical.
    #
    # Default ON: every clip should get the best framing possible out
    # of the box. The bridge in ``human_reframe_bridge.py`` is fully
    # safety-netted — if the new path produces a plan that fails
    # coverage verification it silently falls back to the legacy
    # output, so flipping this default cannot make any existing run
    # worse than the legacy baseline.
    human_reframe_enabled: bool = True

    # Phase A wiring flag. Distinct from ``human_reframe_enabled``:
    # that flag drives the post-hoc RenderPlan override hook
    # (``human_reframe_bridge.maybe_override_render_plan``). This flag
    # drives a *primary* pipeline branch that runs ``run_human_reframe``
    # before the legacy segmenter so the SOTA 2026 stack (2-D LP solver
    # + Kalman + event state machine + A/B scheduler + motivated zoom)
    # produces the :class:`ReframeSegment` list that the rest of the
    # pipeline consumes. Default OFF; promote to ON only after the
    # human-parity bench measurement lands (see docs/human_parity_bench.md).
    human_reframe_pipeline_enabled: bool = False

    # ── L1 / LP solver weights (Grundmann et al. 2011 Sec 4.2) ─────
    lp_lambda_data: float = 1.0
    lp_lambda_v: float = 20.0
    lp_lambda_a: float = 100.0
    lp_lambda_j: float = 100.0

    # Independent y-axis weights. Humans tolerate slower y motion, so
    # lambda_v_y is intentionally higher — prefer to hold y still.
    lp_lambda_v_y: float = 40.0
    lp_lambda_a_y: float = 200.0
    lp_lambda_j_y: float = 200.0

    # Max frames before Condat fallback (LP is O(n^3) worst case).
    lp_max_frames: int = 900

    # ── Dead-zone + stationary thresholds (fractions of source width)
    deadzone_frac: float = 0.018
    stationary_threshold: float = 0.08

    # ── Shot-cut / saccade easing ──────────────────────────────────
    # All durations in ms. 0 = hard cut.
    ease_shot_cut_ms: int = 0
    ease_speaker_turn_ms: int = 200
    ease_subject_walk_ms: int = 250

    # Saccade (new subject / speaker / intent switch) ease duration.
    # Human saccadic eye movements are ~100-300 ms; cosine ease keeps
    # it from feeling like a hard cut while staying sharp.
    saccade_ease_ms: int = 150

    # Match-cut threshold: if the next shot's subject is within this
    # fraction of the output frame of the current camera, the camera
    # holds framing across the cut instead of recentering.
    match_cut_threshold: float = 0.10

    # ── Composition ────────────────────────────────────────────────
    # Headroom in fractions of OUTPUT frame height. face_top >=
    # headroom_min and <= headroom_max at all times.
    headroom_min: float = 0.05
    headroom_max: float = 0.15

    # Eye-line third target (y fraction of output frame). 0.35 places
    # the eye mid-point on the top third, matching standard practice.
    eye_line_y: float = 0.35

    # Lead-room gain: output-frame offset per unit of gaze yaw. A face
    # looking 45° to one side is shifted by lead_room_gain * sign(yaw)
    # on the opposite third. Clamped so the face still fits.
    lead_room_gain: float = 0.08

    # Thirds-bias Gaussian sigma (fraction of output frame). Higher =
    # softer pull; 0.12 is the existing default.
    thirds_sigma: float = 0.12

    # ── Motivated zoom ────────────────────────────────────────────
    zoom_push_in_max_scale: float = 1.20    # 20% push-in ceiling
    zoom_pull_out_max_scale: float = 0.85   # pull-out floor
    zoom_min_duration_sec: float = 1.5
    zoom_max_duration_sec: float = 4.0
    zoom_min_gap_sec: float = 4.0

    # ── A/B cut scheduler ─────────────────────────────────────────
    ab_min_turn_sec: float = 0.8
    ab_max_cuts_per_sec: float = 2.0
    ab_overlap_sec: float = 0.2

    # ── Kalman filter ─────────────────────────────────────────────
    # Process noise multiplier. Higher = more responsive / less stable.
    kalman_process_noise: float = 0.04
    # Measurement noise multiplier. Higher = more smoothing.
    kalman_measurement_noise: float = 0.16
    # Prediction horizon in ms for the solver data term.
    kalman_prediction_ms: float = 350.0

    # ── Critic loop ────────────────────────────────────────────────
    # Modes:
    #   off       — no critic pass.
    #   learned   — local heuristic (face/composition geometry) plus
    #               optional CLIP embedding head. NO network calls.
    #               Default: every reframe gets a sanity score and
    #               low-score windows can re-solve.
    #   vlm       — call a vision LLM. Backend selected by
    #               ``critic_vlm_backend`` (auto / ollama / openrouter).
    #   both      — run both, average the score.
    #
    # Default is "learned" so reframing has aesthetic feedback out of
    # the box even on machines without any network or cloud API key.
    critic_mode: str = "learned"
    # Backend for the ``vlm`` mode. ``auto`` prefers Ollama if the
    # local server is reachable (keeps the run 100% local), then
    # falls back to OpenRouter if configured.
    critic_vlm_backend: str = "auto"
    critic_sample_interval_sec: float = 1.0
    critic_threshold: float = 6.0
    critic_budget_per_clip: int = 20
    critic_cache_dir: str = "/tmp/clipai_critic_cache"

    # ── Saliency fusion weights (Fix 3.10) ────────────────────────
    # (spatial, temporal, color) weights for the 3-channel saliency
    # fusion. Must sum to 1.0. Per-content overrides live in the
    # _CONTENT_OVERRIDES table below; saliency_tracker.py reads
    # config.saliency_* instead of keeping its own duplicate table.
    saliency_spatial_weight: float = 0.30
    saliency_temporal_weight: float = 0.50
    saliency_color_weight: float = 0.20

    # ── Subject-fusion promotion bounds (Fix 3.9 + 3.10) ──────────
    # Per-content bounds for the saliency→SubjectTrack promotion gate.
    # subject_fusion.py reads from config when available instead of
    # its own tables.
    fusion_aspect_min: float = 0.8
    fusion_aspect_max: float = 1.8
    fusion_area_min: float = 0.005
    fusion_area_max: float = 0.20
    fusion_min_persistence_frames: int = 5

    # ── Multi-layout gating (Blueprint v2 Phase 0) ────────────────
    # Content types where SPLIT / TRIPLE / PIP / SCREENSHARE /
    # GAMEPLAY modes are reachable. Any type outside this set stays
    # SINGLE-only (pure reframing). The ``ALLOW_MULTI_LAYOUT`` env
    # var still overrides per-run for fixture / QA use.
    multi_layout_content_types: frozenset = field(
        default_factory=lambda: frozenset({
            "podcast", "interview", "multi_speaker_panel",
            "stream", "gameplay", "gameplay_fps", "gameplay_moba",
            "gameplay_tps", "gameplay_racing",
            "tutorial", "screen_share",
        })
    )

    # ── Static-hold deadband (Blueprint v2 Phase 0) ───────────────
    # How far the target can drift before the camera commits to
    # tracking it. Distinct from ``deadzone_frac`` (which rejects
    # sub-pixel noise in the face-keypoint stream). Genre-dependent:
    # sports / vlog want small (fast response), panels want larger
    # (committed holds). Units: fraction of source width.
    deadband_frac: float = 0.10

    # ── Ken Burns layout (Blueprint v2 Phase 0) ───────────────────
    # Slow-push fallback for scenes with no tracked subject
    # (landscape / B-roll / establishing). Total travel is capped at
    # 8% of source. Set ``ken_burns_min_duration_sec`` to a huge
    # value (or zoom cap to 1.0) to disable.
    ken_burns_max_travel_frac: float = 0.08
    ken_burns_max_zoom: float = 1.08
    ken_burns_min_duration_sec: float = 2.0

    # ── Blueprint v2 Phase 1 — Stage 3 importance matrix ──────────
    # Weights for mixing face / saliency / motion / depth / object
    # signals when building the per-pixel importance map.
    # ``for_content(ct)`` overlays the per-genre row from
    # ``_IMPORTANCE_MATRIX`` automatically; callers rarely set this
    # directly.
    importance: ImportanceWeights = field(default_factory=ImportanceWeights)

    # ── Blueprint v2 Phase 2 — Hybrid layout selector ─────────────
    # When the deterministic scorer's top candidate beats the
    # runner-up by less than this fraction of the top score, the
    # layout engine escalates the decision to a VLM call. 0.6 = the
    # top must win by >= 60% margin.
    layout_confidence_threshold: float = 0.6
    # Hard cap on VLM layout calls per clip. Prevents pathologically
    # ambiguous clips from spamming the VLM.
    layout_vlm_budget_per_clip: int = 8
    # Backend selector, same convention as ``critic_vlm_backend``.
    layout_vlm_backend: str = "auto"
    # Cache dir for per-scene VLM layout decisions.
    layout_vlm_cache_dir: str = "/tmp/clipai_layout_vlm_cache"

    # Extra knobs — per-content overrides stored as a dict so the
    # per-type table can live in one place.
    content_overrides: dict = field(default_factory=dict)

    # ── Helpers ────────────────────────────────────────────────────
    def override(self, **kwargs) -> "ReframeConfig":
        """Return a new frozen config with the given fields replaced."""
        return replace(self, **kwargs)

    def for_content(self, content_type) -> "ReframeConfig":
        """Apply the per-content override table to this config.

        Blueprint v2 Phase 1: the Stage-3 ``importance`` matrix row is
        overlaid automatically from ``_IMPORTANCE_MATRIX`` so callers
        never have to hand-wire weights. An explicit override in
        ``content_overrides[key]["importance"]`` still wins.
        """
        key = getattr(content_type, "value", content_type)
        if key is None:
            return self
        over = dict(self.content_overrides.get(key) or {})
        if "importance" not in over:
            matrix_row = _IMPORTANCE_MATRIX.get(str(key))
            if matrix_row is not None:
                over["importance"] = matrix_row
        if not over:
            return self
        return replace(self, **over)


# ── Content-type override table ────────────────────────────────────
#
# The rules here encode the "Phase 9" genre-specific refinements. They
# are intentionally data — any new content type just needs an entry.

_CONTENT_OVERRIDES = {
    "talking_head": {
        "headroom_min": 0.08,
        "headroom_max": 0.14,
        "zoom_push_in_max_scale": 1.10,
        # Saliency weights tuned for face-dominant scenes
        "saliency_spatial_weight": 0.45,
        "saliency_temporal_weight": 0.30,
        "saliency_color_weight": 0.25,
        "deadband_frac": 0.08,
    },
    "cinematic_dialogue": {
        "headroom_min": 0.08,
        "headroom_max": 0.14,
        "saliency_spatial_weight": 0.45,
        "saliency_temporal_weight": 0.30,
        "saliency_color_weight": 0.25,
        "deadband_frac": 0.12,
    },
    "multi_speaker_panel": {
        "zoom_push_in_max_scale": 1.00,
        "zoom_pull_out_max_scale": 1.00,
        "eye_line_y": 0.38,
        "thirds_sigma": 0.16,
        "saliency_spatial_weight": 0.45,
        "saliency_temporal_weight": 0.30,
        "saliency_color_weight": 0.25,
        "deadband_frac": 0.12,
    },
    "podcast": {
        "deadband_frac": 0.10,
    },
    "interview": {
        "deadband_frac": 0.10,
    },
    "animation_dialogue": {
        "headroom_max": 0.20,
        "saliency_spatial_weight": 0.45,
        "saliency_temporal_weight": 0.30,
        "saliency_color_weight": 0.25,
        "fusion_aspect_min": 0.8,
        "fusion_aspect_max": 1.8,
        "deadband_frac": 0.10,
    },
    "animation": {
        "headroom_max": 0.22,
        "saccade_ease_ms": 100,
        "saliency_spatial_weight": 0.30,
        "saliency_temporal_weight": 0.50,
        "saliency_color_weight": 0.20,
        "fusion_aspect_min": 0.6,
        "fusion_aspect_max": 2.2,
        "fusion_area_min": 0.003,
        "fusion_area_max": 0.30,
        "fusion_min_persistence_frames": 4,
        "deadband_frac": 0.10,
    },
    "music_video": {
        "lp_lambda_v": 10.0,
        "zoom_push_in_max_scale": 1.00,
        "saliency_spatial_weight": 0.30,
        "saliency_temporal_weight": 0.45,
        "saliency_color_weight": 0.25,
        "fusion_aspect_min": 0.6,
        "fusion_aspect_max": 2.5,
        "fusion_min_persistence_frames": 4,
        "deadband_frac": 0.12,
    },
    "sports": {
        "lp_lambda_v_y": 60.0,
        "lead_room_gain": 0.12,
        "zoom_push_in_max_scale": 1.15,
        "saliency_spatial_weight": 0.25,
        "saliency_temporal_weight": 0.55,
        "saliency_color_weight": 0.20,
        "deadband_frac": 0.08,
    },
    "sports_basketball": {
        "lead_room_gain": 0.14,
        "eye_line_y": 0.40,
        "saliency_spatial_weight": 0.25,
        "saliency_temporal_weight": 0.55,
        "saliency_color_weight": 0.20,
        "fusion_aspect_min": 0.8,
        "fusion_aspect_max": 1.2,
        "fusion_area_min": 0.0005,
        "fusion_area_max": 0.05,
        "fusion_min_persistence_frames": 3,
    },
    "sports_racing": {
        "eye_line_y": 0.55,
        "lp_lambda_v_y": 80.0,
        "saliency_spatial_weight": 0.25,
        "saliency_temporal_weight": 0.55,
        "saliency_color_weight": 0.20,
        "fusion_aspect_min": 1.5,
        "fusion_aspect_max": 4.0,
        "fusion_area_min": 0.01,
        "fusion_area_max": 0.35,
        "fusion_min_persistence_frames": 3,
        "deadband_frac": 0.06,
    },
    "gameplay": {
        "zoom_push_in_max_scale": 1.00,
        "deadzone_frac": 0.025,
        "deadband_frac": 0.10,
    },
    "gameplay_moba": {
        "zoom_push_in_max_scale": 1.00,
        "deadzone_frac": 0.030,
        "deadband_frac": 0.10,
    },
    "gameplay_tps": {
        "zoom_push_in_max_scale": 1.00,
        "deadzone_frac": 0.020,
        "deadband_frac": 0.10,
    },
    "gameplay_racing": {
        "zoom_push_in_max_scale": 1.00,
        "deadzone_frac": 0.030,
        "deadband_frac": 0.10,
    },
    "documentary": {
        "deadband_frac": 0.15,
    },
    "landscape": {
        "deadband_frac": 0.15,
    },
    "stream": {
        "saliency_spatial_weight": 0.40,
        "saliency_temporal_weight": 0.40,
        "saliency_color_weight": 0.20,
    },
    "vlog": {
        "headroom_min": 0.08,
        "headroom_max": 0.14,
        "zoom_push_in_max_scale": 1.12,
        "deadband_frac": 0.08,
    },
    "narrative": {
        "zoom_push_in_max_scale": 1.15,
    },
    "generic": {
        "saliency_spatial_weight": 0.30,
        "saliency_temporal_weight": 0.50,
        "saliency_color_weight": 0.20,
    },
}


def load_default_config() -> ReframeConfig:
    """Load the default config, applying env overrides on top of
    defaults and merging in the per-content override table."""
    return ReframeConfig(
        human_reframe_enabled=_env_bool("CLIPAI_HUMAN_REFRAME", True),
        human_reframe_pipeline_enabled=_env_bool(
            "CLIPAI_HUMAN_REFRAME_PIPELINE", False,
        ),

        lp_lambda_v=_env_float("CLIPAI_LP_LAMBDA_V", 20.0),
        lp_lambda_a=_env_float("CLIPAI_LP_LAMBDA_A", 100.0),
        lp_lambda_j=_env_float("CLIPAI_LP_LAMBDA_J", 100.0),

        lp_lambda_v_y=_env_float("CLIPAI_LP_LAMBDA_V_Y", 40.0),
        lp_lambda_a_y=_env_float("CLIPAI_LP_LAMBDA_A_Y", 200.0),
        lp_lambda_j_y=_env_float("CLIPAI_LP_LAMBDA_J_Y", 200.0),

        lp_max_frames=_env_int("CLIPAI_LP_MAX_FRAMES", 900),

        deadzone_frac=_env_float("CLIPAI_DEADZONE_FRAC", 0.018),
        stationary_threshold=_env_float("CLIPAI_STATIONARY_THRESHOLD", 0.08),
        deadband_frac=_env_float("CLIPAI_DEADBAND_FRAC", 0.10),

        ken_burns_max_travel_frac=_env_float("CLIPAI_KEN_BURNS_TRAVEL", 0.08),
        ken_burns_max_zoom=_env_float("CLIPAI_KEN_BURNS_ZOOM", 1.08),
        ken_burns_min_duration_sec=_env_float("CLIPAI_KEN_BURNS_MIN_DUR", 2.0),

        ease_shot_cut_ms=_env_int("CLIPAI_EASE_SHOT_CUT_MS", 0),
        ease_speaker_turn_ms=_env_int("CLIPAI_EASE_SPEAKER_TURN_MS", 200),
        ease_subject_walk_ms=_env_int("CLIPAI_EASE_SUBJECT_WALK_MS", 250),
        saccade_ease_ms=_env_int("CLIPAI_SACCADE_EASE_MS", 150),
        match_cut_threshold=_env_float("CLIPAI_MATCH_CUT_THRESHOLD", 0.10),

        headroom_min=_env_float("CLIPAI_HEADROOM_MIN", 0.05),
        headroom_max=_env_float("CLIPAI_HEADROOM_MAX", 0.15),
        eye_line_y=_env_float("CLIPAI_EYE_LINE_Y", 0.35),
        lead_room_gain=_env_float("CLIPAI_LEAD_ROOM_GAIN", 0.08),
        thirds_sigma=_env_float("CLIPAI_THIRDS_SIGMA", 0.12),

        zoom_push_in_max_scale=_env_float("CLIPAI_ZOOM_PUSH_IN", 1.20),
        zoom_pull_out_max_scale=_env_float("CLIPAI_ZOOM_PULL_OUT", 0.85),
        zoom_min_duration_sec=_env_float("CLIPAI_ZOOM_MIN_DUR", 1.5),
        zoom_max_duration_sec=_env_float("CLIPAI_ZOOM_MAX_DUR", 4.0),
        zoom_min_gap_sec=_env_float("CLIPAI_ZOOM_MIN_GAP", 4.0),

        ab_min_turn_sec=_env_float("CLIPAI_AB_MIN_TURN", 0.8),
        ab_max_cuts_per_sec=_env_float("CLIPAI_AB_MAX_CUTS", 2.0),
        ab_overlap_sec=_env_float("CLIPAI_AB_OVERLAP", 0.2),

        kalman_process_noise=_env_float("CLIPAI_KALMAN_PROCESS", 0.04),
        kalman_measurement_noise=_env_float("CLIPAI_KALMAN_MEAS", 0.16),
        kalman_prediction_ms=_env_float("CLIPAI_KALMAN_PRED_MS", 350.0),

        layout_confidence_threshold=_env_float(
            "CLIPAI_LAYOUT_CONFIDENCE_THRESHOLD", 0.6,
        ),
        layout_vlm_budget_per_clip=_env_int(
            "CLIPAI_LAYOUT_VLM_BUDGET", 8,
        ),
        layout_vlm_backend=os.environ.get(
            "CLIPAI_LAYOUT_VLM_BACKEND", "auto",
        ),
        layout_vlm_cache_dir=os.environ.get(
            "CLIPAI_LAYOUT_VLM_CACHE",
            "/tmp/clipai_layout_vlm_cache",
        ),

        critic_mode=os.environ.get("CLIPAI_CRITIC_MODE", "learned"),
        critic_vlm_backend=os.environ.get("CLIPAI_CRITIC_VLM_BACKEND", "auto"),
        critic_sample_interval_sec=_env_float("CLIPAI_CRITIC_DT", 1.0),
        critic_threshold=_env_float("CLIPAI_CRITIC_THRESHOLD", 6.0),
        critic_budget_per_clip=_env_int("CLIPAI_CRITIC_BUDGET", 20),
        critic_cache_dir=os.environ.get("CLIPAI_CRITIC_CACHE",
                                        "/tmp/clipai_critic_cache"),

        content_overrides=dict(_CONTENT_OVERRIDES),
    )


# Module-level singleton for callers that don't want to thread a config.
# Immutable; callers that need to tweak should call ``.override()`` on
# a copy.
DEFAULT_CONFIG: Optional[ReframeConfig] = None


def get_default_config() -> ReframeConfig:
    global DEFAULT_CONFIG
    if DEFAULT_CONFIG is None:
        DEFAULT_CONFIG = load_default_config()
    return DEFAULT_CONFIG
