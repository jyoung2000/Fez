"""Phase 2 — signal stream unit tests.

Covers:

  * :func:`center_anchor_stream` always-on synthetic region
  * :func:`action_saliency_stream` weight-pass-through +
    scaling
  * :class:`HudGlanceStream` peak → hold → decay envelope
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.services.gaming_glance_triggers import GlanceTrigger
from backend.services.gaming_signal_streams import (
    DECAY_FLOOR,
    GamingSaliencyRegion,
    HudGlanceStream,
    action_saliency_stream,
    center_anchor_stream,
)


# ──────────────────── Stand-in region class ────────────────────


@dataclass
class _StubHud:
    bbox: tuple[int, int, int, int]
    semantic_hint: str = "unknown"
    frame_width: int = 1920
    frame_height: int = 1080
    confidence: float = 0.8


# ──────────────────── Stream A — center anchor ────────────────────


def test_center_anchor_dimensions():
    r = center_anchor_stream(1920, 1080)
    assert r.kind == "center_anchor"
    assert r.weight == 0.4
    x, y, w, h = r.bbox
    # 20% × 40% centered
    assert w == int(round(1920 * 0.20))
    assert h == int(round(1080 * 0.40))
    # Centered
    assert abs((x + w / 2) - 1920 / 2) < 1
    assert abs((y + h / 2) - 1080 / 2) < 1


def test_center_anchor_tunable_weight():
    r = center_anchor_stream(1280, 720, weight=0.7)
    assert r.weight == 0.7


# ──────────────────── Stream B — action ────────────────────


@dataclass
class _StubAction:
    bbox: tuple[int, int, int, int]
    weight: float
    is_required: bool = False
    label: str = "motion"


def test_action_stream_passthrough():
    regs = [
        _StubAction(bbox=(100, 200, 50, 60), weight=0.7),
        _StubAction(bbox=(500, 400, 80, 40), weight=0.5, is_required=True),
    ]
    out = action_saliency_stream(
        regs, frame_w=1920, frame_h=1080,
    )
    assert len(out) == 2
    assert out[0].weight == 0.7
    assert out[0].is_required is False
    assert out[1].weight == 0.5
    assert out[1].is_required is True
    assert all(r.kind == "action" for r in out)


def test_action_stream_scales_weights():
    regs = [_StubAction(bbox=(0, 0, 10, 10), weight=0.6)]
    out = action_saliency_stream(
        regs, frame_w=1920, frame_h=1080, gaming_action_scale=1.5,
    )
    assert out[0].weight == 0.6 * 1.5


def test_action_stream_accepts_legacy_pct_shape():
    """Legacy ``SaliencyRegion`` uses percentage center-coords
    + ``saliency_score``; the pass-through should adapt."""
    @dataclass
    class _Legacy:
        x: float  # center pct
        y: float
        w: float  # w pct
        h: float
        saliency_score: float

    leg = _Legacy(x=50.0, y=40.0, w=20.0, h=30.0, saliency_score=0.8)
    out = action_saliency_stream([leg], frame_w=1000, frame_h=500)
    assert len(out) == 1
    r = out[0]
    assert r.weight == 0.8
    # center at 50%/40% of 1000×500 = (500, 200); w=20% of 1000 = 200
    assert r.bbox[0] == 400  # center - w/2
    assert r.bbox[1] == 125  # center - h/2
    assert r.bbox[2] == 200
    assert r.bbox[3] == 150


# ──────────────────── Stream C — HUD glance ────────────────────


def test_glance_stream_empty_when_never_triggered():
    regions = [_StubHud(bbox=(10, 10, 100, 40))]
    stream = HudGlanceStream(regions, frame_rate=30.0)
    out = stream.update(frame_idx=0, triggers=[])
    assert out == []
    out = stream.update(frame_idx=30, triggers=[])
    assert out == []


def test_glance_stream_peak_and_decay():
    region = _StubHud(
        bbox=(1700, 20, 200, 50),
        semantic_hint="killfeed",
    )
    stream = HudGlanceStream(
        [region],
        frame_rate=30.0,
        peak_weight=1.2,
        hold_s=0.3,       # 9 frames
        decay_tau_s=1.0,  # 30 frames
    )
    trig = GlanceTrigger(
        region_id=id(region),
        peak_weight=1.2,
        source="pixel_delta",
        frame_idx=0,
    )

    # Trigger frame — full peak weight.
    out = stream.update(frame_idx=0, triggers=[trig])
    assert len(out) == 1
    assert out[0].weight == 1.2
    assert out[0].kind == "hud_glance"

    # Still in hold window — full weight.
    out = stream.update(frame_idx=5, triggers=[])
    assert len(out) == 1
    assert out[0].weight == 1.2

    # Just past hold — decay has begun.
    out = stream.update(frame_idx=20, triggers=[])
    assert len(out) == 1
    # Decayed strictly below 1.2 but still well above floor.
    assert out[0].weight < 1.2
    assert out[0].weight > DECAY_FLOOR

    # Far past decay — should drop out entirely.
    out = stream.update(frame_idx=300, triggers=[])
    assert out == []


def test_glance_stream_retrigger_refreshes():
    region = _StubHud(bbox=(10, 10, 100, 40), semantic_hint="health")
    stream = HudGlanceStream(
        [region],
        frame_rate=30.0,
        peak_weight=1.2,
        hold_s=0.3,
        decay_tau_s=1.0,
    )
    trig = GlanceTrigger(
        region_id=id(region),
        peak_weight=1.2,
        source="pixel_delta",
        frame_idx=0,
    )
    stream.update(frame_idx=0, triggers=[trig])
    # Let it decay some.
    stream.update(frame_idx=40, triggers=[])
    decayed_weight = stream.weight_for(region)
    assert 0.0 < decayed_weight < 1.2

    # Re-trigger — weight should jump back to at least peak.
    retrig = GlanceTrigger(
        region_id=id(region),
        peak_weight=1.2,
        source="pixel_delta",
        frame_idx=45,
    )
    out = stream.update(frame_idx=45, triggers=[retrig])
    assert len(out) == 1
    assert out[0].weight >= 1.2


def test_glance_stream_reset_clears_all():
    region = _StubHud(bbox=(10, 10, 100, 40))
    stream = HudGlanceStream([region])
    trig = GlanceTrigger(
        region_id=id(region), peak_weight=1.2,
        source="pixel_delta", frame_idx=0,
    )
    stream.update(0, [trig])
    assert stream.weight_for(region) > 0
    stream.reset()
    assert stream.weight_for(region) == 0.0
    # No output after reset.
    out = stream.update(frame_idx=1, triggers=[])
    assert out == []


def test_glance_output_sorted_by_weight():
    r_a = _StubHud(bbox=(0, 0, 10, 10), semantic_hint="health")
    r_b = _StubHud(bbox=(100, 0, 10, 10), semantic_hint="killfeed")
    stream = HudGlanceStream([r_a, r_b], frame_rate=30.0)
    stream.update(
        frame_idx=0,
        triggers=[
            GlanceTrigger(region_id=id(r_a), peak_weight=0.6,
                          source="pixel_delta", frame_idx=0),
            GlanceTrigger(region_id=id(r_b), peak_weight=1.2,
                          source="pixel_delta", frame_idx=0),
        ],
    )
    out = stream.update(frame_idx=1, triggers=[])
    # Both hold at peak; sorted descending by weight.
    assert [o.weight for o in out] == [1.2, 0.6]
