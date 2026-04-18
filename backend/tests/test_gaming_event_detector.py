"""Phase 2 — gaming event detector tests.

The detector fuses HUD pixel-diff signals + audio envelope into
``GamingEvent`` markers. These tests stub the frame patch reader
so we can fire spikes on demand without writing real images.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

pytest.importorskip("numpy")

import numpy as np  # noqa: E402

from backend.services.gaming_event_detector import (  # noqa: E402
    GamingEvent,
    KILLFEED_DIFF_THRESHOLD,
    detect_crosshair_lock_events,
    detect_gaming_events,
    detect_hud_diff_spikes,
    merge_close_events,
)


@dataclass
class _StubCrosshair:
    timestamp: float
    x_pct: float
    y_pct: float
    confidence: float = 0.8


def _hud_layout() -> dict:
    """Minimal Valorant-shaped HUD layout."""
    return {
        "name": "test",
        "killfeed": {"x_pct": 75, "y_pct": 5, "w_pct": 24, "h_pct": 25},
        "ultimate": {"x_pct": 45, "y_pct": 80, "w_pct": 10, "h_pct": 10},
        "abilities": {"x_pct": 35, "y_pct": 88, "w_pct": 30, "h_pct": 10},
        "minimap": {"x_pct": 0, "y_pct": 0, "w_pct": 18, "h_pct": 25},
    }


def _make_frames(n: int = 30, fps: float = 30.0) -> list:
    return [(i / fps, f"/tmp/f{i:03d}.png") for i in range(n)]


def _calm_then_spike_reader(spike_t_min: float, spike_t_max: float):
    """Return a patch reader that emits a calm steady patch outside
    ``[spike_t_min, spike_t_max]`` and a flickering high-diff patch
    inside (real killfeed text flickers as it animates in)."""
    def _read(path: str, bbox):
        try:
            idx = int(path.replace("/tmp/f", "").replace(".png", ""))
        except ValueError:
            return None
        ts = idx / 30.0
        if spike_t_min <= ts <= spike_t_max:
            # Vary by sample so consecutive diffs are large
            base = 80 + (idx * 47) % 175
            return np.full((20, 20), base, dtype=np.uint8)
        return np.full((20, 20), 50, dtype=np.uint8)
    return _read


def test_detect_hud_diff_spikes_returns_spike_window():
    frames = _make_frames(60)
    reader = _calm_then_spike_reader(1.0, 1.5)
    spikes = detect_hud_diff_spikes(
        frames,
        (75.0, 5.0, 24.0, 25.0),
        threshold=KILLFEED_DIFF_THRESHOLD,
        min_duration=0.3,
        read_patch=reader,
    )
    assert len(spikes) == 1
    s_start, s_end = spikes[0]
    # Spike center should land within the [1.0, 1.5] window
    assert 0.9 <= s_start <= 1.6
    assert 1.0 <= s_end <= 1.7


def test_detect_hud_diff_spikes_ignores_short_blip():
    """A spike shorter than min_duration must NOT register."""
    frames = _make_frames(30)
    # Single-frame spike at t=0.5
    def _read(path, bbox):
        idx = int(path.replace("/tmp/f", "").replace(".png", ""))
        if idx == 15:
            return np.full((20, 20), 250, dtype=np.uint8)
        return np.full((20, 20), 50, dtype=np.uint8)
    spikes = detect_hud_diff_spikes(
        frames,
        (75.0, 5.0, 24.0, 25.0),
        threshold=KILLFEED_DIFF_THRESHOLD,
        min_duration=0.3,
        read_patch=_read,
    )
    assert spikes == []


def test_kill_event_emitted_with_target_region():
    frames = _make_frames(60)
    reader = _calm_then_spike_reader(1.0, 1.5)
    events = detect_gaming_events(
        frame_paths=frames,
        audio_envelope=None,
        hud_layout=_hud_layout(),
        crosshair_path=None,
        gameplay_subtype="fps",
        read_patch=reader,
    )
    kills = [e for e in events if e.kind == "kill"]
    assert kills, f"no kill events emitted, got: {[e.kind for e in events]}"
    ev = kills[0]
    assert ev.target_region is not None
    # Killfeed bbox is at (75, 5, 24, 25) → center (87, 17.5)
    assert abs(ev.target_region[0] - 75.0) < 1e-6
    assert abs(ev.target_region[1] - 5.0) < 1e-6


def test_audio_confirmation_lifts_confidence():
    frames = _make_frames(60)
    reader = _calm_then_spike_reader(1.0, 1.5)
    audio_envelope = (
        # Local mean ~0.1, then a peak at 1.2 well above 2σ
        [(t / 30.0, 0.1) for t in range(60) if not (35 <= t <= 38)]
        + [(t / 30.0, 5.0) for t in range(35, 39)]
    )
    audio_envelope.sort(key=lambda p: p[0])
    events = detect_gaming_events(
        frame_paths=frames,
        audio_envelope=audio_envelope,
        hud_layout=_hud_layout(),
        crosshair_path=None,
        gameplay_subtype="fps",
        read_patch=reader,
    )
    kills = [e for e in events if e.kind == "kill"]
    assert kills
    # With audio confirmation the confidence is 0.75; without it 0.55.
    assert kills[0].confidence >= 0.7
    assert "audio" in kills[0].source


def test_ult_requires_audio_confirmation():
    """Ability/ult flashes alone (no audio peak) must not emit events
    — they trip on every cooldown flash otherwise."""
    frames = _make_frames(60)

    def _read(path, bbox):
        idx = int(path.replace("/tmp/f", "").replace(".png", ""))
        ts = idx / 30.0
        # Spike the ult bbox during 1.0–1.5s
        if bbox[0] >= 35.0 and bbox[1] >= 80.0 and 1.0 <= ts <= 1.5:
            return np.full((20, 20), 250, dtype=np.uint8)
        return np.full((20, 20), 50, dtype=np.uint8)

    # No audio envelope — visual spike alone
    events_no_audio = detect_gaming_events(
        frame_paths=frames,
        audio_envelope=None,
        hud_layout=_hud_layout(),
        crosshair_path=None,
        gameplay_subtype="fps",
        read_patch=_read,
    )
    # When audio_envelope is None, audio_ok defaults to True, so the
    # ult event fires. When the envelope exists but lacks a spike,
    # the ult is suppressed.
    audio_envelope_quiet = [(t / 30.0, 0.1) for t in range(60)]
    events_quiet = detect_gaming_events(
        frame_paths=frames,
        audio_envelope=audio_envelope_quiet,
        hud_layout=_hud_layout(),
        crosshair_path=None,
        gameplay_subtype="fps",
        read_patch=_read,
    )
    ult_evs_quiet = [e for e in events_quiet if e.kind == "ult"]
    assert ult_evs_quiet == [], (
        f"ult events fired without audio confirmation: {ult_evs_quiet}"
    )


def test_merge_close_events_collapses_to_chaos():
    e1 = GamingEvent(timestamp=2.0, duration=0.8, kind="kill",
                     target_region=(75, 5, 24, 25), confidence=0.7)
    e2 = GamingEvent(timestamp=2.5, duration=0.8, kind="ult",
                     target_region=(35, 80, 30, 12), confidence=0.7)
    merged = merge_close_events([e1, e2])
    assert len(merged) == 1
    assert merged[0].kind == "chaos"
    assert merged[0].target_region is None


def test_merge_close_events_keeps_distant_events():
    e1 = GamingEvent(timestamp=1.0, duration=0.8, kind="kill")
    e2 = GamingEvent(timestamp=10.0, duration=0.8, kind="kill")
    merged = merge_close_events([e1, e2])
    assert len(merged) == 2


def test_crosshair_lock_event_requires_audio_spike():
    """Stillness alone is not enough — needs an audio spike."""
    # Crosshair stationary at (50, 50) for 2 seconds
    path = [
        _StubCrosshair(timestamp=t / 10.0, x_pct=50.0, y_pct=50.0)
        for t in range(20)
    ]
    # No audio
    events_no_audio = detect_crosshair_lock_events(path, None)
    assert events_no_audio == []

    # With audio spike at t=1.0
    audio = [(t / 10.0, 0.1) for t in range(20) if t != 10]
    audio.append((1.0, 5.0))
    audio.sort(key=lambda p: p[0])
    events = detect_crosshair_lock_events(path, audio)
    assert any(e.kind == "locked_on" for e in events)


def test_aim_swing_suppresses_kill_event():
    """A kill event that lands during a >20%/s aim swing must be
    suppressed so the camera doesn't fight the player."""
    frames = _make_frames(60)
    reader = _calm_then_spike_reader(1.0, 1.5)

    # Build a crosshair path that's swinging fast around t=1.2
    path = []
    for t in range(20):
        ts = t / 10.0
        # Fast pan from x=20 to x=80 over 2s = 30%/s
        x = 20.0 + 30.0 * ts
        path.append(_StubCrosshair(timestamp=ts, x_pct=x, y_pct=50.0))

    events = detect_gaming_events(
        frame_paths=frames,
        audio_envelope=None,
        hud_layout=_hud_layout(),
        crosshair_path=path,
        gameplay_subtype="fps",
        read_patch=reader,
    )
    kills = [e for e in events if e.kind == "kill"]
    assert kills == [], (
        f"aim-swing kill not suppressed: {[(e.kind, e.timestamp) for e in events]}"
    )


def test_no_events_when_no_hud_layout():
    """Without a HUD layout the detector returns no HUD-driven events."""
    frames = _make_frames(60)
    reader = _calm_then_spike_reader(1.0, 1.5)
    events = detect_gaming_events(
        frame_paths=frames,
        audio_envelope=None,
        hud_layout=None,
        crosshair_path=None,
        gameplay_subtype="fps",
        read_patch=reader,
    )
    assert events == []


def test_empty_frames_returns_empty():
    assert detect_gaming_events(
        frame_paths=[],
        hud_layout=_hud_layout(),
    ) == []
