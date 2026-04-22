"""Phase 0 Task 0.4 — per-genre deadband_frac threaded into L1 solver.

``ReframeConfig.deadband_frac`` is the static-hold deadband (how far
a target can drift before the camera commits to tracking). Distinct
from ``deadzone_frac`` (which rejects sub-pixel noise). Larger =
stickier holds. Per-genre entries let sports respond faster than
panels.
"""

from backend.services.reframe_config import ReframeConfig, get_default_config


def test_default_deadband_is_set():
    cfg = ReframeConfig()
    assert cfg.deadband_frac == 0.10


def test_env_override(monkeypatch):
    from backend.services import reframe_config as rc_mod
    monkeypatch.setenv("CLIPAI_DEADBAND_FRAC", "0.20")
    rc_mod.DEFAULT_CONFIG = None  # force reload
    cfg = rc_mod.load_default_config()
    assert cfg.deadband_frac == 0.20


def test_sports_racing_overrides_to_smaller_deadband():
    cfg = get_default_config().for_content("sports_racing")
    assert cfg.deadband_frac == 0.06


def test_cinematic_dialogue_overrides_to_larger_deadband():
    cfg = get_default_config().for_content("cinematic_dialogue")
    assert cfg.deadband_frac == 0.12


def test_documentary_matches_landscape_for_ken_burns_fallback():
    doc = get_default_config().for_content("documentary")
    lan = get_default_config().for_content("landscape")
    assert doc.deadband_frac == 0.15
    assert lan.deadband_frac == 0.15


def test_solver_respects_deadband_frac_kwarg():
    """A jittery held face with no trend should resolve to stationary
    when the deadband is generous and to tracking when it is tight.

    Uses non-linear jitter so the pre-solve panning detector (which
    fires on clean linear trends) stays out of the way — the deadband
    is specifically the gate for the stationary/tracking split.
    """
    from backend.services.l1_camera_path import solve_camera_path

    source_width = 1920
    # Alternating jitter at the same magnitude, no linear trend, so
    # R^2 ~= 0 and the panning detector skips. path_range == 5% of
    # source, which straddles the 2% and 10% deadbands.
    jitter_frac = 0.025
    positions = [
        (i * (2.0 / 60),
         source_width * (0.5 + (jitter_frac if i % 2 else -jitter_frac)))
        for i in range(60)
    ]

    tight = solve_camera_path(
        positions, source_width=source_width, deadband_frac=0.02,
    )
    generous = solve_camera_path(
        positions, source_width=source_width, deadband_frac=0.10,
    )
    assert generous["mode"] == "stationary"
    assert tight["mode"] != "stationary"


def test_multi_layout_allowlist_field_is_hashable():
    """frozenset default so the dataclass stays hashable."""
    cfg = ReframeConfig()
    assert isinstance(cfg.multi_layout_content_types, frozenset)
