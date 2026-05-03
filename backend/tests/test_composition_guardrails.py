"""Phase 2 — tests for backend.services.composition_guardrails.

Covers the spec's 15 required scenarios. All inputs are synthetic; no
fixtures from disk and no model calls.
"""

from __future__ import annotations

import logging
import math

import pytest

from backend.services.composition_guardrails import (
    CropFrame,
    FrameAnalysis,
    GuardrailReport,
    SubjectBBox,
    TextBox,
    enforce_guardrails,
    guardrails_enabled,
)
from backend.services.reframe_config import get_default_config


SOURCE_W = 1920
SOURCE_H = 1920  # use a square-ish source so crop_h < source_h, giving the
                  # headroom + edge rules vertical freedom to operate.
CROP_W = 1080.0 * 9.0 / 16.0  # 607.5
CROP_H = 1080.0
FPS = 30.0


def _config(**overrides):
    cfg = get_default_config()
    if overrides:
        cfg = cfg.override(**overrides)
    return cfg


def _crop(t, x, y=0.0, w=CROP_W, h=CROP_H):
    return CropFrame(t=t, x=x, y=y, w=w, h=h)


def _analysis(t, subject=None, gaze_yaw=0.0, text=None):
    return FrameAnalysis(
        timestamp=t,
        subject=subject,
        gaze_yaw=gaze_yaw,
        text_regions=list(text or []),
    )


# ── 1. Headroom too little ─────────────────────────────────────────


def test_headroom_too_little():
    """Face top close to crop top → crop top should shift down so the
    head-top sits inside the crop with [5%, 15%] of crop height of
    headroom."""
    cfg = _config()
    # Crop initially placed with its top 200px below the source top.
    # Face top sits 10 px below the crop top → tiny headroom (~1%).
    crop_y = 200.0
    head_top = crop_y + 10.0
    subject = SubjectBBox(
        x=900, y=head_top, w=120, h=160, head_top_y=head_top,
    )
    crops = [_crop(0.0, x=600, y=crop_y)]
    analyses = [_analysis(0.0, subject=subject)]

    out, report = enforce_guardrails(crops, analyses, SOURCE_W, SOURCE_H, FPS, cfg)
    gap_frac = (head_top - out[0].y) / out[0].h
    assert cfg.headroom_min - 1e-6 <= gap_frac <= cfg.headroom_max + 1e-6, (
        f"gap_frac={gap_frac} not in range"
    )
    assert report.per_rule_counts["headroom"] == 1


# ── 2. Headroom too much ───────────────────────────────────────────


def test_headroom_too_much():
    """Face top at y=0.25 of source → crop has too much headroom;
    rule should shift the crop UP to <=15% headroom."""
    cfg = _config()
    head_top = 0.25 * SOURCE_H  # 270 px
    subject = SubjectBBox(
        x=900, y=head_top, w=120, h=160, head_top_y=head_top,
    )
    crops = [_crop(0.0, x=600, y=0.0)]
    analyses = [_analysis(0.0, subject=subject)]

    out, report = enforce_guardrails(crops, analyses, SOURCE_W, SOURCE_H, FPS, cfg)
    gap_frac = (head_top - out[0].y) / out[0].h
    assert gap_frac <= cfg.headroom_max + 1e-6
    assert report.per_rule_counts["headroom"] == 1


# ── 3. Face at source top → clamped ────────────────────────────────


def test_headroom_at_source_top():
    """Face at y=0 → can't add headroom because we're already at the
    source top; the y must remain at 0 (no_black_space hard rule)."""
    cfg = _config()
    head_top = 0.0
    subject = SubjectBBox(
        x=900, y=head_top, w=120, h=160, head_top_y=head_top,
    )
    crops = [_crop(0.0, x=600, y=0.0)]
    analyses = [_analysis(0.0, subject=subject)]

    out, _ = enforce_guardrails(crops, analyses, SOURCE_W, SOURCE_H, FPS, cfg)
    assert out[0].y == 0.0


# ── 4. Edge avoidance — subject at right edge of crop ──────────────


def test_edge_avoidance_right():
    """Subject right edge at 97% of crop → crop shifts LEFT to keep
    a 5% margin from the right edge."""
    cfg = _config()
    crop_x = 600.0
    crop_right = crop_x + CROP_W
    # Place subject so right edge is 97% of crop width inside.
    sub_right = crop_x + 0.97 * CROP_W
    sub_w = 80.0
    sub_x = sub_right - sub_w
    subject = SubjectBBox(
        x=sub_x, y=400, w=sub_w, h=120,
        head_top_y=400, body_top_y=400, body_bottom_y=520,
    )
    crops = [_crop(0.0, x=crop_x, y=0.0)]
    analyses = [_analysis(0.0, subject=subject)]

    out, report = enforce_guardrails(crops, analyses, SOURCE_W, SOURCE_H, FPS, cfg)
    # New right gap >= 5% of crop width.
    new_right_gap = (out[0].x + out[0].w) - (sub_x + sub_w)
    assert new_right_gap >= 0.05 * CROP_W - 1e-6
    assert report.per_rule_counts["edge_avoidance"] >= 1


# ── 5. Crop at natural joint ───────────────────────────────────────


def test_crop_at_natural_joint():
    """Subject body extends below the crop → crop bottom must NOT
    land in a forbidden joint band (ankle, knee, wrist, neck). It
    should snap to a mid-thigh / mid-torso allowed band.

    Setup: body_top placed so that 10% headroom puts the crop top at
    ~100 px above body_top. We pick a body height tall enough that the
    crop must end somewhere through the body (not below the feet). The
    initial bottom_frac sits in the knee forbidden band 0.69-0.74; the
    rule must snap to the 0.50 (mid-torso) target.
    """
    cfg = _config()
    # Choose values so the headroom-corrected crop still has its
    # bottom in a forbidden band. With headroom_max=0.15, crop_y
    # min = head_top - 0.15*1080 = head_top - 162.
    head_top = 250.0
    body_top = head_top
    body_h = 1500.0
    body_bottom = body_top + body_h

    # Aim for initial bottom_frac = 0.71 (knee band).
    crop_bottom_target = body_top + 0.71 * body_h
    crop_y = crop_bottom_target - CROP_H

    subject = SubjectBBox(
        x=900, y=body_top, w=120, h=body_h,
        head_top_y=head_top,
        body_top_y=body_top, body_bottom_y=body_bottom,
    )
    crops = [_crop(0.0, x=600, y=crop_y)]
    analyses = [_analysis(0.0, subject=subject)]

    out, report = enforce_guardrails(crops, analyses, SOURCE_W, SOURCE_H, FPS, cfg)
    new_bottom_frac = (out[0].y + out[0].h - body_top) / body_h
    forbidden = ((0.13, 0.22), (0.58, 0.66), (0.69, 0.74), (0.94, 1.00))
    for lo, hi in forbidden:
        assert not (lo <= new_bottom_frac <= hi), (
            f"new_bottom_frac={new_bottom_frac} landed in forbidden band {lo}-{hi}"
        )


# ── 6. Look space — subject looking right ──────────────────────────


def test_look_space_right():
    """Gaze yaw +30° (≈ +0.5 normalized) → subject should be on the
    LEFT of the crop center (looking INTO the right half)."""
    cfg = _config()
    sub_cx_initial = 1000.0
    sub_w = 100.0
    subject = SubjectBBox(
        x=sub_cx_initial - sub_w / 2.0, y=400, w=sub_w, h=120,
        head_top_y=400,
    )
    # Crop initially centered on the subject.
    crops = [_crop(0.0, x=sub_cx_initial - CROP_W / 2.0, y=0.0)]
    analyses = [_analysis(0.0, subject=subject, gaze_yaw=0.5)]

    out, _ = enforce_guardrails(crops, analyses, SOURCE_W, SOURCE_H, FPS, cfg)
    sub_cx = subject.x + subject.w / 2.0
    crop_cx = out[0].x + out[0].w / 2.0
    # Subject left of crop center.
    assert sub_cx < crop_cx, f"sub_cx={sub_cx} should be < crop_cx={crop_cx}"


# ── 7. Look space — subject looking left ───────────────────────────


def test_look_space_left():
    """Gaze yaw −30° → subject on RIGHT of crop center."""
    cfg = _config()
    sub_cx_initial = 1000.0
    sub_w = 100.0
    subject = SubjectBBox(
        x=sub_cx_initial - sub_w / 2.0, y=400, w=sub_w, h=120,
        head_top_y=400,
    )
    crops = [_crop(0.0, x=sub_cx_initial - CROP_W / 2.0, y=0.0)]
    analyses = [_analysis(0.0, subject=subject, gaze_yaw=-0.5)]

    out, _ = enforce_guardrails(crops, analyses, SOURCE_W, SOURCE_H, FPS, cfg)
    sub_cx = subject.x + subject.w / 2.0
    crop_cx = out[0].x + out[0].w / 2.0
    assert sub_cx > crop_cx


# ── 8. Text protection — both fit ──────────────────────────────────


def test_text_protection():
    """Essential text bottom-left, subject center-right → both should
    end up included in the crop (text fits in crop dimensions)."""
    cfg = _config()
    # Text and subject span ~360 px horizontally; crop_w=607.5 so they
    # both fit comfortably with 3% padding.
    text = TextBox(x=900, y=900, w=80, h=60, essential=True)
    sub_x = 1180.0
    subject = SubjectBBox(
        x=sub_x, y=400, w=80, h=160, head_top_y=400,
        body_top_y=400, body_bottom_y=560,
    )
    # Crop initially excludes text (positioned far right).
    crops = [_crop(0.0, x=1100, y=0.0)]
    analyses = [_analysis(0.0, subject=subject, text=[text])]

    out, report = enforce_guardrails(crops, analyses, SOURCE_W, SOURCE_H, FPS, cfg)
    # Text now inside.
    assert out[0].x <= text.x and out[0].x + out[0].w >= text.x + text.w
    assert out[0].y <= text.y and out[0].y + out[0].h >= text.y + text.h
    assert report.per_rule_counts["text_protection"] >= 1


# ── 9. Text vs subject conflict — subject wins ─────────────────────


def test_text_vs_subject_conflict(caplog):
    """Text + subject can't both fit (text wider than crop minus
    subject) → subject framing kept, warning logged, text rule
    counted as found-but-not-fixed."""
    cfg = _config()
    # Make the text region nearly as wide as the crop so it can't fit
    # alongside the subject horizontally without exceeding crop_w.
    text = TextBox(x=200, y=900, w=int(CROP_W) + 200, h=60, essential=True)
    sub_x = 1500.0
    subject = SubjectBBox(
        x=sub_x, y=400, w=100, h=160, head_top_y=400,
        body_top_y=400, body_bottom_y=560,
    )
    crops = [_crop(0.0, x=1200, y=0.0)]
    analyses = [_analysis(0.0, subject=subject, text=[text])]

    with caplog.at_level(logging.WARNING):
        out, report = enforce_guardrails(crops, analyses, SOURCE_W, SOURCE_H, FPS, cfg)

    # Subject still in crop.
    assert out[0].x <= subject.x
    assert out[0].x + out[0].w >= subject.x + subject.w
    # Warning logged.
    assert any("text" in r.message.lower() and "subject" in r.message.lower()
               for r in caplog.records)
    # Rule counted as found.
    assert report.per_rule_counts["text_protection"] >= 1


# ── 10. No-black-space — left clamp ────────────────────────────────


def test_no_black_space_left_clamp():
    """Crop x = -50 → clamped to 0."""
    cfg = _config()
    crops = [_crop(0.0, x=-50.0, y=0.0)]
    analyses = [_analysis(0.0)]

    out, report = enforce_guardrails(crops, analyses, SOURCE_W, SOURCE_H, FPS, cfg)
    assert out[0].x == 0.0
    assert report.per_rule_counts["no_black_space"] >= 1


# ── 11. No-black-space — right clamp ───────────────────────────────


def test_no_black_space_right_clamp():
    """Crop right beyond source_width → clamped."""
    cfg = _config()
    # x set so x + crop_w > SOURCE_W.
    bad_x = float(SOURCE_W) - CROP_W + 50.0
    crops = [_crop(0.0, x=bad_x, y=0.0)]
    analyses = [_analysis(0.0)]

    out, report = enforce_guardrails(crops, analyses, SOURCE_W, SOURCE_H, FPS, cfg)
    assert out[0].x + out[0].w <= SOURCE_W + 1e-6
    assert report.per_rule_counts["no_black_space"] >= 1


# ── 12. Pan speed limit ────────────────────────────────────────────


def test_pan_speed_limit():
    """Crop center moves 60% of crop width in 1 second → rule clamps
    the velocity to 40% per second."""
    cfg = _config()
    # 30 frames at 30 fps = 1 second. Move from cx=900 to cx=900 + 0.6*CROP_W.
    n = 31
    frames = []
    analyses = []
    start_cx = 900.0
    end_cx = start_cx + 0.6 * CROP_W
    for i in range(n):
        alpha = i / (n - 1)
        cx = start_cx + alpha * (end_cx - start_cx)
        frames.append(_crop(t=i / FPS, x=cx - CROP_W / 2.0, y=0.0))
        analyses.append(_analysis(i / FPS))

    out, report = enforce_guardrails(frames, analyses, SOURCE_W, SOURCE_H, FPS, cfg)
    # After clamping, total cx travel over 1 s must be <= 40% of crop_w.
    travel = out[-1].cx - out[0].cx
    assert travel <= cfg.pan_speed_max_frac_per_sec * CROP_W + 1e-6
    assert report.per_rule_counts["pan_speed"] > 0


# ── 13. Min hold time — merge close repositions ────────────────────


def test_min_hold_time():
    """Two repositions 0.8s apart (less than min_hold_sec=1.5) →
    merged. The output should not contain both repositions as
    distinct sub-1.5s events.

    pan_speed is bumped very high here so it doesn't smooth the jumps
    into a continuous ramp before the min-hold rule sees them.
    """
    cfg = _config(pan_speed_max_frac_per_sec=999.0)
    # 60 frames = 2 seconds. Reposition at frame 12 (0.4 s) and at
    # frame 36 (1.2 s) — gap = 0.8 s.
    n = 60
    frames = []
    analyses = []
    cx = 900.0
    for i in range(n):
        if i == 12:
            cx = 1100.0  # reposition 1
        elif i == 36:
            cx = 700.0   # reposition 2 — 0.8s after first
        frames.append(_crop(t=i / FPS, x=cx - CROP_W / 2.0, y=0.0))
        analyses.append(_analysis(i / FPS))

    out, report = enforce_guardrails(frames, analyses, SOURCE_W, SOURCE_H, FPS, cfg)
    assert report.per_rule_counts["min_hold"] >= 1
    # After merge, the cx series should have at most ONE jump > 3% of
    # crop width in any 1.5 s window.
    cxs = [c.cx for c in out]
    big_jumps = sum(
        1 for i in range(1, len(cxs))
        if abs(cxs[i] - cxs[i - 1]) > 0.03 * CROP_W
    )
    # Original had 2 jumps; merged should have 1 (or fewer, since the
    # blend can also dampen further).
    assert big_jumps <= 1


# ── 14. Priority ordering — no_black_space overrides look_space ───


def test_priority_ordering():
    """When look-space would push the crop out of bounds, no_black_space
    wins. Subject near source right edge looking right would normally
    shift the crop further right; clamp keeps it inside."""
    cfg = _config()
    # Subject at x=1800 (near right edge of 1920-wide source).
    sub_x = 1800.0
    sub_w = 100.0
    subject = SubjectBBox(
        x=sub_x, y=400, w=sub_w, h=160, head_top_y=400,
    )
    # Place crop at the right edge already.
    crops = [_crop(0.0, x=float(SOURCE_W) - CROP_W, y=0.0)]
    # Subject looking right → look-space wants subject on left of crop
    # which would require shifting the crop right (out of bounds).
    analyses = [_analysis(0.0, subject=subject, gaze_yaw=0.7)]

    out, report = enforce_guardrails(crops, analyses, SOURCE_W, SOURCE_H, FPS, cfg)
    assert out[0].x + out[0].w <= SOURCE_W + 1e-6
    assert out[0].x >= -1e-6


# ── 15. Report counts match actual violations ─────────────────────


def test_guardrail_report_counts():
    """The report should tally violations across all rules — at least
    one violation each from headroom + no_black_space + pan_speed —
    and the per-rule + total counts must agree."""
    cfg = _config()
    # Frame 0: headroom violation (face just below crop top) AND
    #          no_black_space violation (crop x = -100).
    crop_y = 200.0
    head_top = crop_y + 10.0
    sub_a = SubjectBBox(
        x=200, y=head_top, w=120, h=160, head_top_y=head_top,
    )
    sub_b = SubjectBBox(
        x=1500, y=head_top, w=120, h=160, head_top_y=head_top,
    )
    crops = [
        _crop(0.0, x=-100, y=crop_y),
        # Frame 2 has a different subject ~1300px to the right; edge
        # avoidance pulls the crop way over → pan_speed must clamp.
        _crop(1.0 / FPS, x=-100, y=crop_y),
    ]
    analyses = [
        _analysis(0.0, subject=sub_a),
        _analysis(1.0 / FPS, subject=sub_b),
    ]

    out, report = enforce_guardrails(crops, analyses, SOURCE_W, SOURCE_H, FPS, cfg)
    # Sum of per-rule counts >= total violations_found (some rules
    # like the entry/exit clamp can be double-counted intentionally).
    rule_sum = sum(report.per_rule_counts.values())
    assert rule_sum == report.violations_found
    assert report.violations_fixed + report.violations_unfixable == report.violations_found
    assert report.total_frames == 2
    assert report.per_rule_counts["no_black_space"] >= 1
    assert report.per_rule_counts["headroom"] >= 1
    assert report.per_rule_counts["pan_speed"] >= 1
    assert report.worst_frame is not None


# ── Bonus: env flag ────────────────────────────────────────────────


def test_env_flag(monkeypatch):
    """guardrails_enabled honors CLIPAI_COMPOSITION_GUARDRAILS."""
    monkeypatch.delenv("CLIPAI_COMPOSITION_GUARDRAILS", raising=False)
    assert guardrails_enabled() is True
    monkeypatch.setenv("CLIPAI_COMPOSITION_GUARDRAILS", "0")
    assert guardrails_enabled() is False
    monkeypatch.setenv("CLIPAI_COMPOSITION_GUARDRAILS", "false")
    assert guardrails_enabled() is False
    monkeypatch.setenv("CLIPAI_COMPOSITION_GUARDRAILS", "1")
    assert guardrails_enabled() is True
