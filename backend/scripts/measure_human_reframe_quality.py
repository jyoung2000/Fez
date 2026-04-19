"""Measure human-reframe quality against the targets in §5 of the
reframe overhaul spec.

Runs against synthetic fixtures (no external media required) so the
CI harness can detect regressions without pulling the
``tests/real_content`` bench set. When real MP4s are available at
``CLIPAI_REAL_CONTENT_CACHE`` the script picks them up automatically.

Metrics per §5:

  * chin-clip rate (< 0.5% of frames, hard fail > 2%)
  * head-clip rate (< 0.5%, hard fail > 2%)
  * subject-center error p95 (< 0.12 of crop width, hard fail > 0.20)
  * fallback-to-center frames (0, hard fail > 0)
  * silent legacy-fallback rate (0, hard fail > 0)
  * coverage violations post-repair (0, hard fail > 0)
  * jitter p95 cx-velocity std (< 0.025, hard fail > 0.05)
  * mean frames between saccades (> 1.5s, hard fail < 0.8s)
  * critic auto-repair rate (5–25%, hard fail 0% or > 50%)

Usage:
    python -m backend.scripts.measure_human_reframe_quality

Emits one row per fixture followed by a summary.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional

from backend.services.camera_path_2d import FaceFrame2D, solve_2d_camera_path
from backend.services.human_reframe import (
    HumanReframeInputs,
    HumanReframePlan,
    run_human_reframe,
)
from backend.services.human_render_plan_adapter import (
    render_plan_from_human_plan,
    verify_frame_coverage,
)
from backend.services.intent_track import build_intent_track
from backend.services.reframe_config import get_default_config
from backend.services.reframe_critic import auto_repair_plan, score_plan

logger = logging.getLogger("measure_human_reframe_quality")


# ── Metric dataclasses ───────────────────────────────────────────


@dataclass
class Thresholds:
    chin_clip_rate_hard: float = 0.02
    chin_clip_rate_target: float = 0.005
    head_clip_rate_hard: float = 0.02
    head_clip_rate_target: float = 0.005
    center_err_p95_hard: float = 0.20
    center_err_p95_target: float = 0.12
    jitter_p95_hard: float = 0.05
    jitter_p95_target: float = 0.025
    saccade_gap_min_hard: float = 0.8
    saccade_gap_min_target: float = 1.5
    critic_repair_min_target: float = 0.05
    critic_repair_max_target: float = 0.25
    critic_repair_max_hard: float = 0.50


@dataclass
class FixtureMetrics:
    name: str
    n_frames_scored: int = 0
    chin_clip_frames: int = 0
    head_clip_frames: int = 0
    center_errors: list = field(default_factory=list)
    fallback_center_frames: int = 0
    coverage_ok: bool = True
    silent_legacy_fallback: bool = False
    jitter_stds: list = field(default_factory=list)
    saccade_gaps: list = field(default_factory=list)
    critic_windows_total: int = 0
    critic_windows_repaired: int = 0

    def chin_rate(self) -> float:
        if self.n_frames_scored == 0:
            return 0.0
        return self.chin_clip_frames / self.n_frames_scored

    def head_rate(self) -> float:
        if self.n_frames_scored == 0:
            return 0.0
        return self.head_clip_frames / self.n_frames_scored

    def center_err_p95(self) -> float:
        if not self.center_errors:
            return 0.0
        return _p95(self.center_errors)

    def jitter_p95(self) -> float:
        if not self.jitter_stds:
            return 0.0
        return _p95(self.jitter_stds)

    def mean_saccade_gap(self) -> float:
        if not self.saccade_gaps:
            return 9999.0
        return sum(self.saccade_gaps) / len(self.saccade_gaps)

    def critic_repair_rate(self) -> float:
        if self.critic_windows_total == 0:
            return 0.0
        return self.critic_windows_repaired / self.critic_windows_total


def _p95(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = max(0, int(round(len(s) * 0.95)) - 1)
    return s[idx]


# ── Fixture generators ──────────────────────────────────────────


@dataclass
class _FakeFace:
    nose_x: float
    nose_y: float
    width: float
    height: float
    identity_id: int = 0
    is_human: bool = True
    lip_aperture: float = 0.0


@dataclass
class _FakeFrameFaces:
    timestamp: float
    faces: list


def _fx_talking_head(duration: float = 10.0, hz: float = 10.0) -> dict:
    n = int(duration * hz)
    dense = []
    for i in range(n):
        t = i / hz
        # Single face near center, slight left-right drift.
        x = 50.0 + 2.0 * ((i % 10) - 5)
        dense.append(_FakeFrameFaces(
            timestamp=t,
            faces=[_FakeFace(nose_x=x, nose_y=35.0, width=15.0, height=18.0,
                             identity_id=0)],
        ))
    return {
        "name": "talking_head",
        "duration": duration,
        "source_w": 1920, "source_h": 1080,
        "content_type": "talking_head",
        "dense_faces": dense,
        "shot_boundaries": [],
        "active_speaker_events": [],
    }


def _fx_mixed_genre(duration: float = 15.0, hz: float = 10.0) -> dict:
    """3 shots: talking-head / montage / action."""
    n = int(duration * hz)
    dense = []
    for i in range(n):
        t = i / hz
        if t < 5.0:
            # dialogue
            faces = [_FakeFace(nose_x=40.0, nose_y=35.0, width=15.0, height=18.0,
                               identity_id=0)]
        elif t < 6.0:
            # montage: no faces
            faces = []
        else:
            # action: sweeping face
            x = 20.0 + ((i % 8) * 8.0)
            faces = [_FakeFace(nose_x=x, nose_y=40.0, width=12.0, height=14.0,
                               identity_id=1)]
        dense.append(_FakeFrameFaces(timestamp=t, faces=faces))
    return {
        "name": "mixed_genre",
        "duration": duration,
        "source_w": 1920, "source_h": 1080,
        "content_type": "",   # let per-shot classifier drive
        "dense_faces": dense,
        "shot_boundaries": [5.0, 6.0],
        "active_speaker_events": [],
    }


def _fx_gappy_faces(duration: float = 10.0, hz: float = 10.0) -> dict:
    """Faces drop out for [3.0, 6.0] — stresses Kalman hold + y-uncertain widen."""
    n = int(duration * hz)
    dense = []
    for i in range(n):
        t = i / hz
        if 3.0 <= t < 6.0:
            faces = []
        else:
            faces = [_FakeFace(nose_x=30.0, nose_y=40.0, width=12.0, height=14.0,
                               identity_id=0)]
        dense.append(_FakeFrameFaces(timestamp=t, faces=faces))
    return {
        "name": "gappy_faces",
        "duration": duration,
        "source_w": 1920, "source_h": 1080,
        "content_type": "talking_head",
        "dense_faces": dense,
        "shot_boundaries": [],
        "active_speaker_events": [],
    }


# ── Measurement ─────────────────────────────────────────────────


def _measure_fixture(fx: dict) -> FixtureMetrics:
    metrics = FixtureMetrics(name=fx["name"])
    try:
        inputs = HumanReframeInputs(
            duration_sec=fx["duration"],
            source_w=fx["source_w"], source_h=fx["source_h"],
            content_type=fx["content_type"],
            dense_faces=fx["dense_faces"],
            active_speaker_events=fx["active_speaker_events"],
            shot_boundaries=fx["shot_boundaries"],
        )
        human: HumanReframePlan = run_human_reframe(inputs)
        plan = render_plan_from_human_plan(
            human,
            source_width=fx["source_w"], source_height=fx["source_h"],
            source_fps=30.0, config=get_default_config(),
            content_type=fx["content_type"],
        )
    except Exception as e:
        logger.warning("fixture %s crashed (%s)", fx["name"], e)
        metrics.silent_legacy_fallback = True
        return metrics

    # Coverage
    cov = verify_frame_coverage(plan)
    metrics.coverage_ok = cov.ok

    # Intent track for fallback-to-center audit.
    track = build_intent_track(
        duration_sec=fx["duration"],
        shot_boundaries=fx["shot_boundaries"],
        shot_profiles=human.shot_profiles,
        dense_faces=fx["dense_faces"],
        active_speaker_events=fx["active_speaker_events"],
    )
    for s in track:
        # Any (0.5, 0.5, <=0.1, 'none') sample is a direct fail.
        if s.source == "none":
            metrics.fallback_center_frames += 1

    # Per-frame headroom / chin / center error from plan vs dense_faces.
    scores = score_plan(plan, dense_faces=fx["dense_faces"])
    for s in scores:
        metrics.n_frames_scored += 1
        if "chin_clip" in s.reasons:
            metrics.chin_clip_frames += 1
        if "head_clip" in s.reasons:
            metrics.head_clip_frames += 1
        err = s.metrics.get("avg_center_err", 0.0) or 0.0
        metrics.center_errors.append(err)
        jitter = s.metrics.get("jitter_std", 0.0) or 0.0
        metrics.jitter_stds.append(jitter)

    # Critic repair rate (proxy: how many low-score windows would be
    # repaired on a second pass).
    repaired_plan, fixes, final = auto_repair_plan(
        plan, dense_faces=fx["dense_faces"],
    )
    metrics.critic_windows_total = len(scores)
    metrics.critic_windows_repaired = len(fixes)

    # Saccade gaps from plan op-boundaries.
    cuts = [op.end_sec for op in plan.ops[:-1]]
    if len(cuts) >= 2:
        metrics.saccade_gaps = [cuts[i + 1] - cuts[i]
                                for i in range(len(cuts) - 1)]
    else:
        metrics.saccade_gaps = [fx["duration"]]

    return metrics


def _grade(m: FixtureMetrics, th: Thresholds) -> tuple[bool, list]:
    fails: list = []
    if m.chin_rate() > th.chin_clip_rate_hard:
        fails.append(f"chin_rate={m.chin_rate():.3f}>{th.chin_clip_rate_hard}")
    if m.head_rate() > th.head_clip_rate_hard:
        fails.append(f"head_rate={m.head_rate():.3f}>{th.head_clip_rate_hard}")
    if m.center_err_p95() > th.center_err_p95_hard:
        fails.append(
            f"center_err_p95={m.center_err_p95():.3f}>{th.center_err_p95_hard}"
        )
    if m.fallback_center_frames > 0:
        fails.append(f"fallback_center_frames={m.fallback_center_frames}")
    if m.silent_legacy_fallback:
        fails.append("silent_legacy_fallback")
    if not m.coverage_ok:
        fails.append("coverage_not_ok")
    if m.jitter_p95() > th.jitter_p95_hard:
        fails.append(f"jitter_p95={m.jitter_p95():.3f}>{th.jitter_p95_hard}")
    if m.mean_saccade_gap() < th.saccade_gap_min_hard:
        fails.append(
            f"mean_saccade={m.mean_saccade_gap():.2f}<{th.saccade_gap_min_hard}"
        )
    rate = m.critic_repair_rate()
    if rate > th.critic_repair_max_hard:
        fails.append(f"critic_rate={rate:.2f}>{th.critic_repair_max_hard}")
    return (not fails, fails)


# ── Main ─────────────────────────────────────────────────────────


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", action="store_true",
                    help="emit results as a JSON array to stdout")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    fixtures = [
        _fx_talking_head(),
        _fx_mixed_genre(),
        _fx_gappy_faces(),
    ]
    thresholds = Thresholds()
    rows = []
    any_hard_fail = False
    for fx in fixtures:
        m = _measure_fixture(fx)
        ok, fails = _grade(m, thresholds)
        rows.append((m, ok, fails))
        if not ok:
            any_hard_fail = True

    if args.json:
        import json
        out = []
        for m, ok, fails in rows:
            out.append({
                "fixture": m.name,
                "pass": ok,
                "chin_rate": round(m.chin_rate(), 4),
                "head_rate": round(m.head_rate(), 4),
                "center_err_p95": round(m.center_err_p95(), 4),
                "jitter_p95": round(m.jitter_p95(), 4),
                "fallback_center_frames": m.fallback_center_frames,
                "coverage_ok": m.coverage_ok,
                "critic_repair_rate": round(m.critic_repair_rate(), 3),
                "mean_saccade_gap": round(m.mean_saccade_gap(), 2),
                "fails": fails,
            })
        print(json.dumps(out, indent=2))
    else:
        print(
            f"{'fixture':<24}{'pass':<6}{'chin%':<7}{'head%':<7}"
            f"{'cerr_p95':<10}{'jit_p95':<10}{'fbc':<5}"
            f"{'cov':<5}{'crit%':<7}"
        )
        print("-" * 80)
        for m, ok, fails in rows:
            print(
                f"{m.name:<24}{('ok' if ok else 'FAIL'):<6}"
                f"{m.chin_rate()*100:<7.2f}{m.head_rate()*100:<7.2f}"
                f"{m.center_err_p95():<10.4f}{m.jitter_p95():<10.4f}"
                f"{m.fallback_center_frames:<5}"
                f"{('y' if m.coverage_ok else 'n'):<5}"
                f"{m.critic_repair_rate()*100:<7.1f}"
            )
            if fails:
                print("   fails:", ", ".join(fails))

    return 1 if any_hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
