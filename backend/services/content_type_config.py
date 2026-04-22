"""Per-content-type configuration for the ReframeSegmenter.

Each content type has different editorial conventions for how a human
editor would reframe 16:9 content for vertical (9:16). These configs
encode those conventions as tunable parameters.

IMPORTANT: Timing constants (min_hold_seconds, anticipation_ms) are NO
LONGER in this config. They are derived from LocalPacingEstimator at
runtime based on measured video signals. This config only contains
style/layout preferences and detection thresholds.
"""

from enum import Enum


class ContentType(str, Enum):
    NARRATIVE = "narrative"
    PODCAST = "podcast"
    GAMING = "gaming"
    VLOG = "vlog"
    SPORTS = "sports"
    SPORTS_BASKETBALL = "sports_basketball"
    SPORTS_RACING = "sports_racing"
    MUSIC_VIDEO = "music_video"
    ANIME = "anime"
    ANIMATION_DIALOGUE = "animation_dialogue"
    # v2 Phase 11: distinct preset for seated 2-5 person panels (Verzuz,
    # debate show, interview show). Panels need tighter holds and snap-
    # on-speaker-turn pacing — not the vlog/podcast tracking presets.
    MULTI_SPEAKER_PANEL = "multi_speaker_panel"
    UNKNOWN = "unknown"


class ReframeStrategy(str, Enum):
    STATIONARY = "stationary"          # fixed viewport, hold at slot position
    TRACKING = "tracking"              # smooth follow with EMA
    PANNING = "panning"                # constant-velocity move (narrative walks)
    WIDE_MASTER = "wide_master"        # letterbox + blur, full original frame visible
    SPLIT_SCREEN = "split_screen"      # 2 tiles stacked (2 speakers active)
    GRID = "grid"                      # 3-4 tiles
    STACKED_GAMEPLAY = "stacked_gameplay"  # gameplay top, facecam bottom
    BLUR_FILL = "blur_fill"            # centered original + blurred bg fill


CONTENT_TYPE_CONFIG = {
    ContentType.NARRATIVE: {
        # Style/layout preferences (NOT timing — timing is from pacing estimator)
        "apply_lead_room": True,
        "wide_master_on_multi_face": True,
        "allow_tracking": False,
        "allow_motion_tracking": True,   # high-motion scenes use tracking
        "use_split_screen_on_overlap": False,
        "fallback_preference": "wide_master",  # honor DP's composition
        # Ease durations (style, not timing)
        "ease_speaker_turn_ms": 500,
        "ease_shot_cut_ms": 0,
        "ease_subject_walk_ms": 500,
        # Detection thresholds
        "speaker_confidence_threshold": 0.6,
        "speaker_coverage_threshold": 0.70,
        "dense_dominance_threshold": 0.70,
        "multi_speaker_threshold": 0.20,
        "action_cut_rate_threshold": 6.0,  # cuts per 10s window → WIDE_MASTER
        "action_window_seconds": 10.0,
        # Intent tracking
        "intent_ema_alpha": 0.40,
        "intent_switch_margin": 0.15,
    },
    ContentType.PODCAST: {
        "apply_lead_room": False,
        "wide_master_on_multi_face": False,
        "allow_tracking": False,
        "allow_motion_tracking": False,
        "use_split_screen_on_overlap": True,
        "overlap_threshold_seconds": 1.0,
        "laughter_widen_ms": 1500,
        "fallback_preference": "blur_fill",  # show the couch, not a letterbox
        "ease_speaker_turn_ms": 400,
        "ease_shot_cut_ms": 0,
        "ease_subject_walk_ms": 500,
        "speaker_confidence_threshold": 0.5,
        "speaker_coverage_threshold": 0.60,
        "dense_dominance_threshold": 0.70,
        "multi_speaker_threshold": 0.20,
        # Intent tracking — slower, ambiguous overlaps common
        "intent_ema_alpha": 0.25,
        "intent_switch_margin": 0.20,
    },
    ContentType.GAMING: {
        "apply_lead_room": False,
        "wide_master_on_multi_face": False,
        "allow_tracking": True,
        "allow_motion_tracking": True,
        "use_split_screen_on_overlap": False,
        "preserve_hud": True,
        "prefer_stacked_gameplay": True,
        "tracking_stabilization_threshold": 0.30,
        "fallback_preference": "blur_fill",  # preserve HUD
        "ease_speaker_turn_ms": 0,
        "ease_shot_cut_ms": 0,
        "ease_subject_walk_ms": 0,
        "speaker_confidence_threshold": 0.6,
        "speaker_coverage_threshold": 0.60,
        "dense_dominance_threshold": 0.70,
        "multi_speaker_threshold": 0.20,
        # Intent tracking — heavily resist switching, center-bias content
        "intent_ema_alpha": 0.20,
        "intent_switch_margin": 0.30,
    },
    ContentType.VLOG: {
        "apply_lead_room": True,
        "wide_master_on_multi_face": True,
        "allow_tracking": True,
        "allow_motion_tracking": True,
        "use_split_screen_on_overlap": False,
        "tracking_tau": 0.5,
        "tracking_max_velocity_pct_per_sec": 15.0,
        "fallback_preference": "blur_fill",
        "ease_speaker_turn_ms": 400,
        "ease_shot_cut_ms": 0,
        "ease_subject_walk_ms": 600,
        "speaker_confidence_threshold": 0.6,
        "speaker_coverage_threshold": 0.60,
        "dense_dominance_threshold": 0.70,
        "multi_speaker_threshold": 0.20,
        # Intent tracking
        "intent_ema_alpha": 0.30,
        "intent_switch_margin": 0.18,
    },
    ContentType.SPORTS: {
        "apply_lead_room": False,
        "wide_master_on_multi_face": True,
        "allow_tracking": True,
        "allow_motion_tracking": True,
        "use_split_screen_on_overlap": False,
        "prefer_wide_master": True,
        "preserve_scoreboard": True,
        "tracking_max_velocity_pct_per_sec": 25.0,
        "fallback_preference": "blur_fill",
        "ease_speaker_turn_ms": 0,
        "ease_shot_cut_ms": 0,
        "ease_subject_walk_ms": 0,
        "speaker_confidence_threshold": 0.6,
        "speaker_coverage_threshold": 0.60,
        "dense_dominance_threshold": 0.70,
        "multi_speaker_threshold": 0.20,
        # Intent tracking — snappy, fast cuts demand fast response
        "intent_ema_alpha": 0.55,
        "intent_switch_margin": 0.10,
    },
    ContentType.SPORTS_BASKETBALL: {
        "apply_lead_room": False,
        "wide_master_on_multi_face": True,
        "allow_tracking": True,
        "allow_motion_tracking": True,
        "use_split_screen_on_overlap": False,
        "prefer_wide_master": True,
        "preserve_scoreboard": True,
        "tracking_max_velocity_pct_per_sec": 35.0,
        "fallback_preference": "blur_fill",
        "ease_speaker_turn_ms": 0,
        "ease_shot_cut_ms": 0,
        "ease_subject_walk_ms": 0,
        "speaker_confidence_threshold": 0.6,
        "speaker_coverage_threshold": 0.60,
        "dense_dominance_threshold": 0.70,
        "multi_speaker_threshold": 0.20,
        "intent_ema_alpha": 0.60,
        "intent_switch_margin": 0.08,
    },
    ContentType.SPORTS_RACING: {
        "apply_lead_room": False,
        "wide_master_on_multi_face": True,
        "allow_tracking": True,
        "allow_motion_tracking": True,
        "use_split_screen_on_overlap": False,
        "prefer_wide_master": True,
        "preserve_scoreboard": True,
        "prefer_lower_third": True,
        "tracking_max_velocity_pct_per_sec": 30.0,
        "fallback_preference": "blur_fill",
        "ease_speaker_turn_ms": 0,
        "ease_shot_cut_ms": 0,
        "ease_subject_walk_ms": 0,
        "speaker_confidence_threshold": 0.6,
        "speaker_coverage_threshold": 0.60,
        "dense_dominance_threshold": 0.70,
        "multi_speaker_threshold": 0.20,
        "intent_ema_alpha": 0.45,
        "intent_switch_margin": 0.12,
    },
    ContentType.MUSIC_VIDEO: {
        "apply_lead_room": False,
        "wide_master_on_multi_face": True,
        "allow_tracking": True,
        "allow_motion_tracking": True,
        "use_split_screen_on_overlap": False,
        "fallback_preference": "blur_fill",  # motion reads through blur
        "ease_speaker_turn_ms": 0,
        "ease_shot_cut_ms": 0,
        "ease_subject_walk_ms": 0,
        "speaker_confidence_threshold": 0.6,
        "speaker_coverage_threshold": 0.60,
        "dense_dominance_threshold": 0.70,
        "multi_speaker_threshold": 0.20,
        # Intent tracking — snappy for fast cuts
        "intent_ema_alpha": 0.50,
        "intent_switch_margin": 0.12,
    },
    ContentType.ANIME: {
        "apply_lead_room": True,
        "wide_master_on_multi_face": True,
        "allow_tracking": True,
        "allow_motion_tracking": True,
        "use_split_screen_on_overlap": False,
        "fallback_preference": "blur_fill",  # motion reads through blur
        "ease_speaker_turn_ms": 300,
        "ease_shot_cut_ms": 0,
        "ease_subject_walk_ms": 400,
        "speaker_confidence_threshold": 0.6,
        "speaker_coverage_threshold": 0.60,
        "dense_dominance_threshold": 0.70,
        "multi_speaker_threshold": 0.20,
        # Intent tracking — snappiest, fast cuts demand fast response
        "intent_ema_alpha": 0.55,
        "intent_switch_margin": 0.10,
    },
    ContentType.ANIMATION_DIALOGUE: {
        # Anime dialogue: character-driven scenes with 1-2 speakers,
        # clear gaze direction, and slow-to-medium cut rates. Human
        # editors hold tight on the speaking character, apply lead room
        # in the speaking direction, and use blur_fill when composition
        # would otherwise go wide.
        "apply_lead_room": True,
        "wide_master_on_multi_face": False,   # NEVER go wide on anime dialogue —
                                               # always track the active speaker
        "allow_tracking": True,
        "allow_motion_tracking": True,
        "use_split_screen_on_overlap": False,
        "fallback_preference": "blur_fill",
        # Ease: snap on shot cuts (anime editors cut on beats), short
        # ease on speaker turn (200ms = one-frame anticipation at 24fps)
        "ease_speaker_turn_ms": 200,
        "ease_shot_cut_ms": 0,
        "ease_subject_walk_ms": 400,
        # Detection thresholds: slightly looser than anime because
        # character designs are consistent (lower false-positive rate)
        "speaker_confidence_threshold": 0.55,
        "speaker_coverage_threshold": 0.55,
        "dense_dominance_threshold": 0.65,
        "multi_speaker_threshold": 0.20,
        # Intent tracking: fast alpha for quick speaker-switch response,
        # tight margin for decisive snaps with no hover
        "intent_ema_alpha": 0.60,
        "intent_switch_margin": 0.08,
    },
    # v2 Phase 11 — multi-speaker panel (Verzuz, debate show, interview panel).
    # Mirrors PODCAST but with tighter per-speaker holds and no in-shot
    # tracking: seats are fixed so the camera should snap on speaker turn,
    # not pan. Short-shot override is gated separately on shot-detector
    # confidence in layout_engine.
    ContentType.MULTI_SPEAKER_PANEL: {
        "apply_lead_room": False,
        "wide_master_on_multi_face": False,
        "allow_tracking": False,
        "allow_tracking_within_shot": False,
        "allow_motion_tracking": False,
        "use_split_screen_on_overlap": True,
        "overlap_threshold_seconds": 1.0,
        "laughter_widen_ms": 1500,
        "fallback_preference": "blur_fill",
        "ease_speaker_turn_ms": 300,
        "ease_shot_cut_ms": 0,
        "ease_subject_walk_ms": 400,
        "speaker_confidence_threshold": 0.5,
        "speaker_coverage_threshold": 0.60,
        "dense_dominance_threshold": 0.70,
        "multi_speaker_threshold": 0.20,
        # Tighter hold floor than podcast (0.9s vs 1.2s default pacing)
        # so speaker turns feel immediate.
        "min_hold_seconds_panel": 0.9,
        "anticipation_lead_seconds": 0.20,
        "intent_ema_alpha": 0.35,
        "intent_switch_margin": 0.15,
    },
    ContentType.UNKNOWN: {
        "apply_lead_room": False,
        "wide_master_on_multi_face": True,
        "allow_tracking": False,
        "allow_motion_tracking": False,
        "use_split_screen_on_overlap": False,
        "fallback_preference": "blur_fill",
        "ease_speaker_turn_ms": 500,
        "ease_shot_cut_ms": 0,
        "ease_subject_walk_ms": 600,
        "speaker_confidence_threshold": 0.6,
        "speaker_coverage_threshold": 0.60,
        "dense_dominance_threshold": 0.70,
        "multi_speaker_threshold": 0.20,
    },
}


def get_config(content_type: str) -> dict:
    """Get the config dict for a content type, falling back to UNKNOWN."""
    try:
        ct = ContentType(content_type)
    except ValueError:
        ct = ContentType.UNKNOWN
    return CONTENT_TYPE_CONFIG.get(ct, CONTENT_TYPE_CONFIG[ContentType.UNKNOWN])


# ── Blueprint v2 Phase 1 — importance-matrix shims ─────────────────
#
# The face / saliency / motion / depth / object weights that used to
# be hardcoded across required_regions.py + genre_refinements.py now
# live on ``ReframeConfig.importance`` (single source of truth).
# These helpers let callers fetch one weight at a time by content
# type name without importing the whole ReframeConfig plumbing.

def get_face_weight(content_type) -> float:
    """Blueprint v2: Stage-3 face weight for a content type.

    ``content_type`` may be a string, a ``ClipContentType`` enum, or
    ``None`` (→ generic defaults).
    """
    from backend.services.reframe_config import get_default_config
    return float(get_default_config().for_content(content_type).importance.face)


def get_saliency_weight(content_type) -> float:
    from backend.services.reframe_config import get_default_config
    return float(get_default_config().for_content(content_type).importance.saliency)


def get_motion_weight(content_type) -> float:
    from backend.services.reframe_config import get_default_config
    return float(get_default_config().for_content(content_type).importance.motion)


def get_object_weight(content_type) -> float:
    from backend.services.reframe_config import get_default_config
    return float(get_default_config().for_content(content_type).importance.object)


def get_depth_weight(content_type) -> float:
    from backend.services.reframe_config import get_default_config
    return float(get_default_config().for_content(content_type).importance.depth)


# ──────────────────── Gap 5c — per-content-type vote split ────────────
#
# When a diarization pass ran in the pipeline (Gap 5a) AND at least
# one cluster mapped to a face slot, the ``SubjectConfidenceEstimator``
# treats the audio-side vote as an independent Check 2b alongside the
# lip-motion vote. The combined ``(lip_weight, diar_weight)`` pair is
# the total "speaker_agree" budget — the two values sum to 0.25,
# which is the same budget the legacy 0.20 lip slice occupied plus
# the 0.05 reallocated from ``transcript`` in the Phase B wiring.
#
# Signal-reliability rationale (per the Gap 5c tuning table in the
# task prompt):
#
#   * Multi-speaker panel — crosstalk is the failure mode for lip
#     motion; diarization gets a slight edge.
#   * Anime / animation dialogue — stylized mouth flaps produce
#     constant low-confidence lip motion; diarization (voice actors
#     have distinctive timbre) gets the edge.
#   * Vlog — close-up single face makes lip motion near-perfect;
#     B-roll audio contaminates the diarization side.
#   * Music video — studio-track lipsync is clean but the audio is
#     effectively a single vocal cluster, so lip dominates.
#   * Narrative — ADR / foley contaminates audio, lip is reliable
#     on professional close-ups.
#   * Sports / gaming / unknown — no specific signal; keep the
#     generic 0.15 / 0.10 default.
ACTIVE_SPEAKER_VOTE_WEIGHTS = {
    ContentType.MULTI_SPEAKER_PANEL: (0.12, 0.13),
    ContentType.PODCAST:             (0.14, 0.11),
    ContentType.NARRATIVE:           (0.17, 0.08),
    ContentType.VLOG:                (0.18, 0.07),
    ContentType.ANIME:               (0.12, 0.13),
    ContentType.ANIMATION_DIALOGUE:  (0.12, 0.13),
    ContentType.GAMING:              (0.15, 0.10),
    ContentType.MUSIC_VIDEO:         (0.18, 0.07),
    ContentType.SPORTS:              (0.15, 0.10),
    ContentType.SPORTS_BASKETBALL:   (0.15, 0.10),
    ContentType.SPORTS_RACING:       (0.15, 0.10),
    ContentType.UNKNOWN:             (0.15, 0.10),
}


def get_vote_weights(content_type: str, has_diarization: bool) -> tuple:
    """Return ``(lip_weight, diar_weight)`` for the given content type.

    When ``has_diarization`` is False, returns ``(0.20, 0.0)`` — the
    legacy lip-only budget, preserving bit-identical pre-Gap-5
    behavior. When True, looks up
    ``ACTIVE_SPEAKER_VOTE_WEIGHTS[content_type]``, falling back to
    the ``ContentType.UNKNOWN`` row when the content_type string
    isn't in the enum.
    """
    if not has_diarization:
        return (0.20, 0.0)
    try:
        ct = ContentType(content_type)
    except ValueError:
        ct = ContentType.UNKNOWN
    return ACTIVE_SPEAKER_VOTE_WEIGHTS.get(
        ct, ACTIVE_SPEAKER_VOTE_WEIGHTS[ContentType.UNKNOWN],
    )


class TuningConfig:
    """Structured access to per-content-type tuning parameters.

    Wraps the raw config dict and provides typed attributes with defaults
    for intent tracking fields.
    """

    # Intent tracking defaults
    _DEFAULTS = {
        "intent_ema_alpha": 0.4,
        "intent_switch_margin": 0.15,
        "intent_min_switch_confidence": 0.35,
        "intent_min_hold_fallback": 1.5,
    }

    def __init__(self, cfg: dict):
        self._cfg = cfg

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._DEFAULTS:
            return self._cfg.get(name, self._DEFAULTS[name])
        if name in self._cfg:
            return self._cfg[name]
        raise AttributeError(f"TuningConfig has no attribute {name!r}")


def get_tuning_from_profile(content_profile) -> TuningConfig:
    """Build a TuningConfig from a ContentProfile (or None).

    Accepts a ContentProfile object, a content_type string, or None.
    """
    ct = "unknown"
    if content_profile is not None:
        ct = getattr(content_profile, "content_type", content_profile)
        if not isinstance(ct, str):
            ct = "unknown"
    return TuningConfig(get_config(ct))
