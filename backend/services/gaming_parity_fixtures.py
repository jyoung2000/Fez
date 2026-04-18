"""Phase 6 — synthetic gaming reframe fixtures.

Six fixtures matching the spec test set:

  - ``valorant_clip``     — FPS, fixed crosshair, 1 kill event
  - ``apex_clip``         — FPS, fast aim swing, 2 kill events
  - ``lol_clip``          — MOBA, lane fight (long segment)
  - ``elden_ring_clip``   — TPS, action-center anchor at (50, 45)
  - ``rocket_league_clip``— Racing, fast horizontal motion
  - ``tf2_clip``          — FPS **center-bias regression fixture**:
    cartoon-style Team Fortress 2 content with no detectable
    crosshair. Before the Phase 1 hard-center fallback branch
    landed, the face-pipeline cluster centers would pull the
    crop to x≈20-32 on this clip. After the fix every frame
    resolves to x=50 with 0-conf breadcrumbs.

Each fixture exposes:

  - ``name`` / ``genre`` / ``game_key``
  - ``video_duration``
  - ``crosshair_truth``  : list of ``(t, x_pct, y_pct)`` ground
    truth at 1 fps. None for non-FPS clips.
  - ``event_truth``      : list of ``(t, kind)`` labeled critical
    moments. None when not labeled.
  - ``layout_target``    : dict mapping layout_mode → minimum
    expected fraction of segments. e.g. ``{"fullscreen": 0.8}``.

The fixtures are pure-Python dataclass stubs — no ffmpeg or
binary frames required. The measurement script
``backend/scripts/measure_gaming_reframe.py`` runs the chooser,
crosshair tracker (mocked), and event detector (mocked)
against them and checks the SLAs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class GamingFixture:
    """One gaming reframe fixture."""

    name: str
    genre: str
    game_key: str
    video_duration: float
    description: str
    crosshair_truth: Optional[list] = None
    event_truth: Optional[list] = None
    layout_target: dict = field(default_factory=dict)
    motion_profile: float = 5.0  # synthetic motion magnitude
    # Optional builder callable that produces extra inputs the
    # measurement script needs (e.g. mock frame-paths list).
    build: Optional[Callable[[], dict]] = None
    # Phase 1 center-bias regression fixtures (e.g. TF2) have no
    # crosshair truth — instead they assert that the tracker
    # returned the hard (50, 50) fallback on every frame. When
    # True, the measurement harness substitutes an all-fallback
    # synthetic tracked path and scores the ``center_lock_pct``
    # metric against a strict 100 % target instead of the usual
    # crosshair-error SLA.
    center_bias_regression: bool = False


# ────────────────────────────────────────────────────────


def _valorant_clip_build() -> dict:
    """Static crosshair at (50, 50) for the duration."""
    return {
        "frame_paths": [(t / 30.0, "/tmp/v.png") for t in range(60)],
        "audio_envelope": [
            (t / 10.0, 0.1) for t in range(60)
            if t not in (12, 13, 14)
        ] + [(1.2, 5.0), (1.3, 5.5), (1.4, 5.0)],
    }


def _apex_clip_build() -> dict:
    return {
        "frame_paths": [(t / 30.0, "/tmp/a.png") for t in range(120)],
        "audio_envelope": [
            (t / 10.0, 0.1) for t in range(120)
            if t not in (8, 9, 70, 71)
        ] + [(0.8, 6.0), (0.9, 6.5), (7.0, 5.0), (7.1, 5.5)],
    }


def _lol_clip_build() -> dict:
    return {
        "frame_paths": [(t / 30.0, "/tmp/l.png") for t in range(180)],
        "audio_envelope": [(t / 10.0, 0.4) for t in range(180)],
    }


def _elden_ring_build() -> dict:
    return {
        "frame_paths": [(t / 30.0, "/tmp/e.png") for t in range(150)],
        "audio_envelope": [(t / 10.0, 0.2) for t in range(150)],
    }


def _rocket_league_build() -> dict:
    return {
        "frame_paths": [(t / 30.0, "/tmp/r.png") for t in range(120)],
        "audio_envelope": [(t / 10.0, 0.3) for t in range(120)],
    }


def _tf2_clip_build() -> dict:
    """Washed-Up-Tuber TF2 clip — cartoon FPS, no detectable
    crosshair. Synthetic stand-in: 3 s worth of frame stubs
    with a flat (low-energy) audio envelope so no gaming events
    fire. The important property is the absence of a crosshair
    path — the fixture's ``crosshair_truth`` is None and the
    harness substitutes an all-fallback (50, 50, 0.0) tracker
    output to verify the center-lock guarantee."""
    return {
        "frame_paths": [(t / 30.0, "/tmp/tf2.png") for t in range(90)],
        "audio_envelope": [(t / 10.0, 0.2) for t in range(90)],
    }


# ────────────────────────────────────────────────────────


GAMING_FIXTURES: dict[str, GamingFixture] = {
    "valorant_clip": GamingFixture(
        name="valorant_clip",
        genre="fps",
        game_key="valorant",
        video_duration=2.0,
        description="Static crosshair, 1 kill event around t=1.3",
        crosshair_truth=[
            (t * 0.5, 50.0, 50.0) for t in range(5)
        ],
        event_truth=[(1.3, "kill")],
        layout_target={"fullscreen": 0.80},
        motion_profile=5.0,
        build=_valorant_clip_build,
    ),
    "apex_clip": GamingFixture(
        name="apex_clip",
        genre="fps",
        game_key="apex_legends",
        video_duration=4.0,
        description="Aim swing (45 → 60), 2 kills at t=0.85, 7.05",
        crosshair_truth=[
            (t * 0.5, 45.0 + 3.75 * t, 50.0) for t in range(8)
        ],
        event_truth=[(0.85, "kill"), (7.05, "kill")],
        layout_target={"fullscreen": 0.80},
        motion_profile=8.0,
        build=_apex_clip_build,
    ),
    "lol_clip": GamingFixture(
        name="lol_clip",
        genre="moba",
        game_key="league_of_legends",
        video_duration=6.0,
        description="6s lane fight, no individual events",
        crosshair_truth=None,
        event_truth=[],
        layout_target={"blurfill": 0.50},
        motion_profile=15.0,
        build=_lol_clip_build,
    ),
    "elden_ring_clip": GamingFixture(
        name="elden_ring_clip",
        genre="tps",
        game_key="elden_ring",
        video_duration=5.0,
        description="Player at (50, 45) anchor, no HUD events",
        crosshair_truth=None,
        event_truth=[],
        layout_target={"fullscreen": 0.60},
        motion_profile=10.0,
        build=_elden_ring_build,
    ),
    "rocket_league_clip": GamingFixture(
        name="rocket_league_clip",
        genre="racing",
        game_key="rocket_league",
        video_duration=4.0,
        description="Racing clip — high motion, fullscreen anchor",
        crosshair_truth=None,
        event_truth=[],
        layout_target={"fullscreen": 0.50},
        # Sit between racing fullscreen threshold (20) and wide_zoom
        # threshold (30) so the chooser picks fullscreen.
        motion_profile=25.0,
        build=_rocket_league_build,
    ),
    "tf2_clip": GamingFixture(
        name="tf2_clip",
        genre="fps",
        game_key="team_fortress_2",
        video_duration=3.0,
        description=(
            "Washed-Up-Tuber TF2 regression: cartoon FPS, no "
            "detectable crosshair, bright character models "
            "off-axis. Asserts center_lock_pct == 100%."
        ),
        # No crosshair truth — fixture is a center-lock regression.
        crosshair_truth=None,
        # No labeled events — a flat audio envelope produces none.
        event_truth=[],
        layout_target={"fullscreen": 0.80},
        motion_profile=5.0,
        build=_tf2_clip_build,
        center_bias_regression=True,
    ),
}


def list_gaming_fixtures() -> list[str]:
    return list(GAMING_FIXTURES.keys())


def get_gaming_fixture(name: str) -> GamingFixture:
    if name not in GAMING_FIXTURES:
        raise KeyError(
            f"Unknown gaming fixture {name!r}. Available: "
            f"{', '.join(sorted(GAMING_FIXTURES))}"
        )
    return GAMING_FIXTURES[name]
