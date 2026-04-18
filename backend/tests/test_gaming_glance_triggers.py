"""Phase 3 — glance trigger unit tests.

Covers pixel-delta firing, audio peak detection, scheduled
fallback after the idle interval, refractory behavior, and the
composed :class:`GlanceTriggerBus`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

pytest.importorskip("numpy")

import numpy as np  # noqa: E402

from backend.services.gaming_glance_triggers import (  # noqa: E402
    AudioPeakTrigger,
    GlanceTrigger,
    GlanceTriggerBus,
    PixelDeltaTrigger,
    ScheduledGlanceFallback,
)


@dataclass
class _StubHud:
    bbox: tuple[int, int, int, int]
    semantic_hint: str = "unknown"
    frame_width: int = 640
    frame_height: int = 360
    confidence: float = 0.8


def _flat_frame(value: int = 50) -> np.ndarray:
    return np.full((360, 640), value, dtype=np.uint8)


def _frame_with_hud_change(
    base_value: int, bbox: tuple[int, int, int, int], hud_value: int,
) -> np.ndarray:
    img = _flat_frame(base_value)
    x, y, w, h = bbox
    img[y:y + h, x:x + w] = hud_value
    return img


# ──────────────────── Pixel delta ────────────────────


def test_pixel_delta_fires_on_hud_change():
    bbox = (540, 30, 80, 30)  # TR killfeed-ish
    region = _StubHud(bbox=bbox, semantic_hint="killfeed")
    det = PixelDeltaTrigger([region], threshold=8.0)

    # Seed with a few steady frames — running mean warms up.
    for i in range(4):
        det.process(_frame_with_hud_change(50, bbox, 100), i)

    # No fire on the steady frames (after the first).
    out = det.process(_frame_with_hud_change(50, bbox, 100), 5)
    assert out == []

    # Big change inside the HUD bbox — should fire.
    out = det.process(_frame_with_hud_change(50, bbox, 210), 6)
    assert len(out) == 1
    assert out[0].region_id == id(region)
    assert out[0].source == "pixel_delta"
    assert out[0].label == "killfeed"


def test_pixel_delta_no_fire_on_steady():
    bbox = (10, 10, 80, 40)
    region = _StubHud(bbox=bbox)
    det = PixelDeltaTrigger([region], threshold=8.0)
    for i in range(12):
        out = det.process(_frame_with_hud_change(80, bbox, 100), i)
        assert out == [] or i <= 1  # first-frame warm-up tolerated


def test_pixel_delta_boost_lowers_threshold():
    bbox = (10, 10, 40, 20)
    region = _StubHud(bbox=bbox)
    det = PixelDeltaTrigger([region], threshold=20.0)
    for i in range(4):
        det.process(_frame_with_hud_change(100, bbox, 120), i)

    # Without boost: a small delta (~10) shouldn't fire.
    out = det.process(_frame_with_hud_change(100, bbox, 130), 5)
    assert out == []

    # With boost (multiplier 0.4 lowers effective threshold to 8).
    out = det.process(
        _frame_with_hud_change(100, bbox, 140), 6,
        region_threshold_boost={id(region): 0.4},
    )
    assert len(out) == 1


# ──────────────────── Audio peaks ────────────────────


def test_audio_peaks_detected_above_sigma():
    # Flat envelope with one clear spike.
    env = [(i * 0.05, 0.1) for i in range(40)]
    env.append((1.5, 10.0))
    det = AudioPeakTrigger(sigma=2.5, window_s=2.0)
    peaks = det.compute_peaks(env)
    assert 1.5 in peaks


def test_audio_is_near_peak_window():
    env = [(i * 0.05, 0.1) for i in range(20)]
    env.append((1.0, 5.0))
    det = AudioPeakTrigger(sigma=2.5)
    det.compute_peaks(env)
    assert det.is_near_peak(1.05, window_ms=200)
    assert not det.is_near_peak(1.5, window_ms=200)


def test_audio_no_envelope_is_noop():
    det = AudioPeakTrigger()
    assert det.compute_peaks([]) == []
    assert det.peak_times == []
    assert det.is_near_peak(0.0) is False


# ──────────────────── Scheduled fallback ────────────────────


def test_scheduled_fires_after_idle_interval():
    a = _StubHud(bbox=(0, 0, 10, 10), confidence=0.9)
    b = _StubHud(bbox=(100, 0, 10, 10), confidence=0.7)
    sched = ScheduledGlanceFallback(
        [a, b], interval_s=1.0, frame_rate=30.0,
    )
    # No triggers for 35 frames (> 1.0s).
    out = sched.process(frame_idx=35)
    assert len(out) == 1
    # Highest-confidence region wins.
    assert out[0].region_id == id(a)


def test_scheduled_no_fire_inside_interval():
    r = _StubHud(bbox=(0, 0, 10, 10), confidence=0.9)
    sched = ScheduledGlanceFallback([r], interval_s=1.0, frame_rate=30.0)
    sched.note_trigger(id(r), frame_idx=0)
    out = sched.process(frame_idx=10)
    assert out == []
    out = sched.process(frame_idx=35)
    # After the interval (>30 frames = 1 s) it should fire again,
    # but only if the region is beyond its per-region refractory.
    # interval * 0.5 = 15 frames; 35 > 15 → eligible.
    assert len(out) == 1


# ──────────────────── Bus composition ────────────────────


def test_bus_runs_pixel_then_scheduled_fallback():
    r = _StubHud(bbox=(540, 30, 80, 30), semantic_hint="killfeed")
    bus = GlanceTriggerBus(
        [r],
        frame_rate=30.0,
        pixel_delta_threshold=8.0,
        scheduled_interval_s=0.5,
        refractory_s=0.3,
    )
    # Warm up the running mean.
    for i in range(3):
        bus.process_frame(
            _frame_with_hud_change(50, r.bbox, 100),
            frame_idx=i, timestamp=i / 30.0,
        )

    # No change → scheduled should NOT fire yet (need 0.5s idle).
    out = bus.process_frame(
        _frame_with_hud_change(50, r.bbox, 100),
        frame_idx=4, timestamp=4 / 30.0,
    )
    assert out == []

    # Wait past the scheduled interval — fallback fires.
    out = bus.process_frame(
        _frame_with_hud_change(50, r.bbox, 100),
        frame_idx=30, timestamp=1.0,
    )
    assert len(out) == 1
    assert out[0].source == "scheduled"


def test_bus_refractory_merges_close_triggers():
    r = _StubHud(bbox=(540, 30, 80, 30), semantic_hint="killfeed")
    bus = GlanceTriggerBus(
        [r],
        frame_rate=30.0,
        pixel_delta_threshold=8.0,
        scheduled_interval_s=100.0,
        refractory_s=0.3,  # 9 frames
    )
    # Warm up
    for i in range(4):
        bus.process_frame(
            _frame_with_hud_change(50, r.bbox, 100),
            frame_idx=i, timestamp=i / 30.0,
        )

    # Big change — fires.
    out1 = bus.process_frame(
        _frame_with_hud_change(50, r.bbox, 200),
        frame_idx=5, timestamp=5 / 30.0,
    )
    assert len(out1) == 1

    # Another big change immediately after — refractory.
    out2 = bus.process_frame(
        _frame_with_hud_change(50, r.bbox, 250),
        frame_idx=6, timestamp=6 / 30.0,
    )
    assert out2 == []

    # Hold the steady state long enough to fully release
    # refractory AND let the EMA adapt, then inject a fresh
    # large change — should fire again.
    for i in range(7, 40):
        bus.process_frame(
            _frame_with_hud_change(50, r.bbox, 100),
            frame_idx=i, timestamp=i / 30.0,
        )
    out3 = bus.process_frame(
        _frame_with_hud_change(50, r.bbox, 240),
        frame_idx=50, timestamp=50 / 30.0,
    )
    assert len(out3) >= 1


def test_bus_with_audio_envelope_boosts_health_killfeed():
    r_hf = _StubHud(bbox=(540, 30, 80, 30), semantic_hint="killfeed")
    r_mm = _StubHud(bbox=(540, 280, 80, 60), semantic_hint="minimap")
    bus = GlanceTriggerBus(
        [r_hf, r_mm],
        frame_rate=30.0,
        pixel_delta_threshold=20.0,
        scheduled_interval_s=100.0,
        audio_boost=4.0,
    )
    env = [(i * 0.05, 0.1) for i in range(40)]
    env.append((0.2, 10.0))  # peak at t=0.2s
    bus.set_audio_envelope(env)

    # Warm up.
    for i in range(3):
        bus.process_frame(
            _flat_frame(60),
            frame_idx=i, timestamp=i / 30.0,
        )

    # Small delta (~8) inside killfeed, no delta inside minimap.
    img = _flat_frame(60)
    img[30:60, 540:620] = 72  # killfeed HUD +12
    # Timestamp at 0.2s → inside audio peak window, killfeed
    # threshold is lowered by audio_boost=4.0 (effective 5 instead
    # of 20). Minimap is NOT in the boosted set.
    out = bus.process_frame(
        img, frame_idx=6, timestamp=0.2,
    )
    assert any(t.region_id == id(r_hf) for t in out)
    assert not any(t.region_id == id(r_mm) for t in out)
