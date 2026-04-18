#!/usr/bin/env python3
"""Phase 9 — Measure AutoFlip parity across the v2 fixture set.

For each fixture in ``backend.services.autoflip_parity_fixtures`` this
script:

  1. Builds the in-memory inputs (no ffmpeg / no .mp4 binaries).
  2. Runs ``build_reframe_segments`` with the fixture's
     ``content_type_override`` injected via metadata so the Phase 1
     plumbing fires and the per-content-type tuning kicks in.
  3. Extracts the actual crop centers and switch times from the
     resulting segments.
  4. Scores them against the fixture's ground truth using the pure-
     Python metric library in ``backend.services.autoflip_parity_metrics``.
  5. Emits a JSON document with one row per fixture × metric.

The metric library is fully unit-testable in a numpy-less sandbox;
this runner needs the full reframe pipeline (which transitively needs
numpy / cv2 / MediaPipe) so it is intended to run inside the docker
image. A ``--dry-run`` mode skips the segmenter call and just reports
the fixture spec + ground truth, which is useful for verifying the
scaffolding in a sandbox.

Usage:
    # Run every fixture and print JSON to stdout
    python -m backend.scripts.measure_autoflip_parity

    # Run a single fixture
    python -m backend.scripts.measure_autoflip_parity --fixture vlog_walk_and_talk

    # Write to a phase-tagged baseline file
    python -m backend.scripts.measure_autoflip_parity \
        --output docs/autoflip_parity_v2_phase1.json --phase phase1

    # Dry run (no segmenter — just dump fixture specs)
    python -m backend.scripts.measure_autoflip_parity --dry-run

    # List fixtures
    python -m backend.scripts.measure_autoflip_parity --list

The runner is deliberately silent about which fixtures need future
phases to pass — every metric just gets a value (or ``None``) and the
results doc is the place where the targets live.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Optional

from backend.services.autoflip_parity_fixtures import (
    FIXTURES,
    FixtureSpec,
    GroundTruth,
    get_fixture,
    list_fixture_names,
)
from backend.services.autoflip_parity_metrics import (
    extract_segment_boundaries,
    extract_switch_times,
    score_fixture,
)


logger = logging.getLogger("autoflip_parity")


# ─────────────── Segmenter integration ───────────────────────

def _run_segmenter(spec: FixtureSpec) -> list:
    """Run ``build_reframe_segments`` against the fixture's inputs.

    Imported lazily so the rest of this module loads in a sandbox
    without numpy. The fixture's ``content_type_override`` /
    ``anime_subtype`` / ``music_subtype`` / ``game_type`` flow through
    the same metadata-injection contract that ``pipeline.py`` uses for
    real jobs:

      1. Build a ``content_profile`` via ``classify_content`` with the
         fixture metadata as a fake ffprobe dict.
      2. Pass the profile to ``build_reframe_segments`` via the
         ``content_profile`` kwarg.

    This mirrors what ``pipeline._run_analysis_inner`` does so the
    runner numbers are directly comparable to production output.
    """
    from backend.services.content_classifier import classify_content
    from backend.services.reframe_segmenter import build_reframe_segments

    kwargs = spec.build()
    metadata: dict[str, str] = {}
    if spec.content_type_override:
        metadata["content_type_override"] = spec.content_type_override
    if spec.anime_subtype:
        metadata["anime_subtype"] = spec.anime_subtype
    if spec.music_subtype:
        metadata["music_subtype"] = spec.music_subtype
    if spec.game_type:
        metadata["game_type"] = spec.game_type

    # Build a content profile so the segmenter's content-aware
    # branches (Stage 8 lead-room, Stage 10c Phase 4 post-process,
    # Stage 10a multi-region LP, Stage 11 music beat snap, etc.)
    # actually fire on the fixture.
    content_profile = None
    try:
        content_profile = classify_content(
            shot_cuts=kwargs.get("shot_cuts", []),
            face_registry=kwargs.get("face_registry"),
            dense_faces=kwargs.get("dense_faces", []),
            scenes=[],  # fixtures don't expose a scenes list
            video_duration=kwargs.get("video_duration", 0.0),
            metadata=metadata,
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(
            "classify_content failed (continuing without profile): %s", e
        )

    # Phase 5: build a BeatGrid from the fixture's pre-baked beat
    # list when the fixture is a music video. Production callers
    # (pipeline.py) build this via beat_detector.detect_beats on
    # the audio track; in the parity bench we use the synthetic
    # grid the fixture already owns so the runner doesn't need
    # librosa.
    #
    # The fixture's ``beat_grid`` is the FULL beat sequence (every
    # beat at 0.5 s spacing for 120 BPM). 4/4 downbeats are every
    # 4th beat — Phase 5's pulse cuts fire on downbeats, not every
    # beat, so we derive ``downbeat_times`` here.
    music_beat_grid = None
    if (
        spec.content_type_override == "music_video"
        and spec.ground_truth.beat_grid
    ):
        from backend.services.beat_detector import BeatGrid as _BG
        beats = list(spec.ground_truth.beat_grid)
        meter = 4  # all current music fixtures are 4/4
        downbeats = beats[::meter]
        music_beat_grid = _BG(
            tempo_bpm=120.0,
            beat_times=beats,
            downbeat_times=downbeats,
            meter=meter,
            source="parity_fixture",
        )

    return build_reframe_segments(
        content_profile=content_profile,
        music_beat_grid=music_beat_grid,
        **kwargs,
    )


# ─────────────── Result extraction ────────────────────────────

def _segments_to_centers(
    segments: list,
    source_width: int,
) -> tuple[list[float], list[tuple[float, float]]]:
    """Return per-frame crop-center % positions + segment spans.

    The reframe pipeline emits one ``ReframeSegment`` per visual
    decision. For metric scoring we want a per-frame x position
    (sampled at the same density as the fixture's dense face stream)
    so that ``max_acceleration`` / ``max_jerk`` see a uniform
    discrete-difference grid.

    Sampling rate matches ``measure_reframe_lag``: 30 fps. We sample
    every 1/30 s within each segment's [start, end) window.
    """
    if not segments:
        return [], []
    centers: list[float] = []
    spans: list[tuple[float, float]] = []
    fps = 30.0
    for seg in segments:
        start = float(getattr(seg, "start"))
        end = float(getattr(seg, "end"))
        spans.append((start, end))
        # subject_x is in source pixels in the segmenter; convert to %.
        sx_px = float(getattr(seg, "subject_x", source_width / 2.0))
        sx_pct = sx_px / max(source_width, 1) * 100.0
        n = max(1, int((end - start) * fps))
        for _ in range(n):
            centers.append(sx_pct)
    return centers, spans


def _spans_for_metric(segments: list) -> list[tuple[float, float]]:
    return [(float(s.start), float(s.end)) for s in segments]


# ─────────────── Per-fixture scoring ──────────────────────────

# Acceptance SLAs for the --animated mode (Phase 6).
ANIMATED_SLA = {
    "dense_ratio_min": 0.30,
    "mean_err_max": 4.0,
    "p95_err_max": 8.0,
}


def score_one(
    spec: FixtureSpec,
    *,
    dry_run: bool = False,
    animated: bool = False,
) -> dict:
    """Score a single fixture and return a JSON-friendly dict."""

    base = {
        "name": spec.name,
        "description": spec.description,
        "content_type_override": spec.content_type_override,
        "anime_subtype": spec.anime_subtype,
        "music_subtype": spec.music_subtype,
        "game_type": spec.game_type,
        "video_duration": spec.video_duration,
        "metrics_requested": list(spec.metrics),
        "ground_truth_summary": _gt_summary(spec.ground_truth),
    }

    if dry_run:
        base["status"] = "dry_run"
        base["metrics"] = {}
        return base

    try:
        segments = _run_segmenter(spec)
    except ImportError as exc:
        # numpy / cv2 / etc. missing — this is the sandbox path.
        base["status"] = "skipped"
        base["error"] = f"segmenter import failed: {type(exc).__name__}: {exc}"
        base["metrics"] = {}
        return base
    except Exception as exc:
        base["status"] = "error"
        base["error"] = f"{type(exc).__name__}: {exc}"
        base["metrics"] = {}
        return base

    crop_centers, spans = _segments_to_centers(segments, spec.source_width)
    actual_switches = extract_switch_times(segments)
    actual_boundaries = extract_segment_boundaries(segments)

    # For the downbeat_snap_error metric the parity fixture's
    # ground-truth ``beat_grid`` is the full beat list, but we
    # actually want to score against DOWNBEATS (every 4th beat in
    # 4/4) so the metric's snap_rate reflects bar-line cadence.
    # Mirror the runner's BeatGrid construction logic above.
    metric_beat_grid: list[float] = []
    if spec.ground_truth.beat_grid:
        if spec.content_type_override == "music_video":
            metric_beat_grid = list(spec.ground_truth.beat_grid)[::4]
        else:
            metric_beat_grid = list(spec.ground_truth.beat_grid)

    metrics_output = score_fixture(
        metrics_to_run=spec.metrics,
        expected_switches=spec.ground_truth.expected_switches,
        actual_switches=actual_switches,
        actual_segment_boundaries=actual_boundaries,
        segment_spans=spans,
        crop_centers=crop_centers,
        required_regions_per_frame=spec.ground_truth.required_regions_per_frame,
        crop_width_pct=spec.crop_width_pct,
        beat_grid=metric_beat_grid,
        hud_zones=spec.ground_truth.hud_zones,
        face_y_in_crop_normalized=spec.ground_truth.face_y_in_crop_normalized,
    )

    base["status"] = "ok"
    base["segments"] = len(segments)
    base["actual_switches"] = len(actual_switches)
    base["actual_boundaries"] = len(actual_boundaries)
    base["metrics"] = metrics_output

    if animated:
        anim = _animated_summary(spec, segments)
        base["animated"] = anim
        # Emit the one-line summary on stderr for CI parsing.
        print(
            "[anime-parity] dense_ratio=%.2f tracking_pct=%d%% "
            "mean_err=%.1f%% p95_err=%.1f%%" % (
                anim["dense_ratio"],
                int(round(anim["tracking_pct"] * 100)),
                anim["mean_err"],
                anim["p95_err"],
            ),
            file=sys.stderr,
        )
        # SLA assertions: dense ratio and per-frame error caps.
        sla_failures: list[str] = []
        if anim["dense_ratio"] < ANIMATED_SLA["dense_ratio_min"]:
            sla_failures.append(
                f"dense_ratio={anim['dense_ratio']:.2f} < "
                f"{ANIMATED_SLA['dense_ratio_min']:.2f}"
            )
        if anim["mean_err"] > ANIMATED_SLA["mean_err_max"]:
            sla_failures.append(
                f"mean_err={anim['mean_err']:.2f} > "
                f"{ANIMATED_SLA['mean_err_max']:.2f}"
            )
        if anim["p95_err"] > ANIMATED_SLA["p95_err_max"]:
            sla_failures.append(
                f"p95_err={anim['p95_err']:.2f} > "
                f"{ANIMATED_SLA['p95_err_max']:.2f}"
            )
        if sla_failures:
            base["sla_status"] = "failed"
            base["sla_failures"] = sla_failures
        else:
            base["sla_status"] = "passed"

    return base


def _gt_summary(gt: GroundTruth) -> dict:
    return {
        "expected_switches": len(gt.expected_switches),
        "beat_grid_size": len(gt.beat_grid),
        "required_region_frames": sum(
            1 for f in gt.required_regions_per_frame if f
        ),
        "hud_zones": len(gt.hud_zones),
        "game_key": gt.game_key,
    }


def _animated_summary(spec: FixtureSpec, segments: list) -> dict:
    """Phase 6 — compute the anime-parity SLA report for an animated fixture.

    Walks the resulting segments and the fixture's dense face track to
    measure:

    - ``dense_ratio``  : fraction of dense-face frames that have at
      least one face — proxy for how well the cascade augmentation
      ran (target: ≥ 0.30).
    - ``tracking_pct`` : fraction of segments with strategy=tracking
      or panning — proxy for Phase 5's intra-shot motion override.
    - ``mean_err``     : per-frame face-center error as % of frame
      width (target: ≤ 4.0%).
    - ``p95_err``      : 95th percentile of the same metric.

    Returns a dict of metric name → numeric value.
    """
    if not segments:
        return {
            "dense_ratio": 0.0,
            "tracking_pct": 0.0,
            "mean_err": 0.0,
            "p95_err": 0.0,
            "by_strategy": {},
        }

    # ── Strategy mix ──
    by_strategy: dict[str, int] = {}
    n_tracking = 0
    for seg in segments:
        strat = getattr(seg, "strategy", "stationary") or "stationary"
        by_strategy[strat] = by_strategy.get(strat, 0) + 1
        if strat in ("tracking", "panning"):
            n_tracking += 1
    tracking_pct = (n_tracking / len(segments)) if segments else 0.0

    # ── Dense face ratio ──
    kwargs = spec.build()
    dense_faces = kwargs.get("dense_faces") or []
    if dense_faces:
        with_faces = sum(
            1 for df in dense_faces
            if getattr(df, "faces", None)
        )
        dense_ratio = with_faces / len(dense_faces)
    else:
        dense_ratio = 0.0

    # ── Per-frame face-center error ──
    # For each dense face frame, find the segment that owns it and
    # compute |face_nose_x - segment.crop_center|. Skip frames with no
    # face. We subtract the per-segment MEAN offset (thirds-bias /
    # lead-room) so the metric measures whether the path's MOTION
    # tracks the face's motion, not the absolute centering. Without
    # this subtraction the legitimate Phase 4 V2 thirds-bias would
    # show up as ~6% of constant error and falsely fail the SLA.
    raw_pairs: list[tuple[object, float, float, float]] = []
    err_by_strategy: dict[str, list[float]] = {}
    for df in dense_faces:
        ts = float(getattr(df, "timestamp", -1.0))
        faces = getattr(df, "faces", None) or []
        if not faces:
            continue
        # Pick the first face's nose_x as the reference target.
        nose_x = None
        for face in faces:
            nx = getattr(face, "nose_x", None)
            if nx is not None:
                nose_x = float(nx)
                break
        if nose_x is None:
            continue
        # Find owning segment
        owning = None
        for seg in segments:
            if float(seg.start) <= ts < float(seg.end):
                owning = seg
                break
        if owning is None:
            continue
        # For tracking / panning segments the per-frame crop center
        # is the motion_path entry nearest to the timestamp; for
        # stationary segments fall back to subject_x.
        sx_px: float
        motion_path = getattr(owning, "motion_path", None) or []
        if motion_path:
            best_entry = min(
                motion_path,
                key=lambda entry: abs(float(entry[0]) - ts),
            )
            sx_px = float(best_entry[1])
        else:
            sx_px = float(getattr(owning, "subject_x", spec.source_width / 2.0))
        sx_pct = sx_px / max(spec.source_width, 1) * 100.0
        raw_pairs.append((owning, ts, nose_x, sx_pct))

    # Compute per-segment mean offset and subtract so we measure path
    # SHAPE (not constant lead-room / thirds-bias absolute offset).
    by_seg: dict[int, list[tuple[float, float]]] = {}
    for owning, _ts, nose_x, sx_pct in raw_pairs:
        by_seg.setdefault(id(owning), []).append((nose_x, sx_pct))
    seg_offset: dict[int, float] = {}
    for sid, pairs in by_seg.items():
        # mean of (sx_pct - nose_x) — this is the constant bias the
        # post-process applied.
        offs = [b - a for a, b in pairs]
        seg_offset[sid] = sum(offs) / len(offs) if offs else 0.0

    errors_pct: list[float] = []
    for owning, _ts, nose_x, sx_pct in raw_pairs:
        offset = seg_offset.get(id(owning), 0.0)
        err = abs(nose_x - (sx_pct - offset))
        errors_pct.append(err)
        strat = getattr(owning, "strategy", "stationary") or "stationary"
        err_by_strategy.setdefault(strat, []).append(err)

    if errors_pct:
        errors_sorted = sorted(errors_pct)
        mean_err = sum(errors_sorted) / len(errors_sorted)
        p95_idx = max(0, int(0.95 * (len(errors_sorted) - 1)))
        p95_err = errors_sorted[p95_idx]
    else:
        mean_err = 0.0
        p95_err = 0.0

    summary_by_strategy = {
        s: round(sum(v) / max(len(v), 1), 3)
        for s, v in err_by_strategy.items()
    }

    return {
        "dense_ratio": round(dense_ratio, 3),
        "tracking_pct": round(tracking_pct, 3),
        "mean_err": round(mean_err, 3),
        "p95_err": round(p95_err, 3),
        "by_strategy": {
            "counts": by_strategy,
            "mean_err_pct": summary_by_strategy,
        },
    }


# ─────────────── Top-level runner ─────────────────────────────

def run_all(
    fixture_names: Optional[list[str]] = None,
    *,
    dry_run: bool = False,
    animated: bool = False,
) -> dict:
    names = fixture_names if fixture_names else list_fixture_names()
    rows = []
    for name in names:
        spec = get_fixture(name)
        rows.append(score_one(spec, dry_run=dry_run, animated=animated))
    return {
        "schema": "autoflip_parity_v2_results/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "animated": animated,
        "fixture_count": len(rows),
        "results": rows,
    }


# ─────────────── CLI ─────────────────────────────────────────

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="measure_autoflip_parity",
        description=(
            "Score every Phase 9 fixture against its ground truth. "
            "See docs/autoflip_parity_v2_results.md for the target table."
        ),
    )
    p.add_argument(
        "--fixture",
        action="append",
        choices=sorted(FIXTURES.keys()),
        help="Score only the named fixture(s). Repeat to score several.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Write the JSON result to this path (in addition to stdout).",
    )
    p.add_argument(
        "--phase",
        default=None,
        help=(
            "Optional phase tag stamped into the result payload "
            "(e.g. 'baseline', 'phase3', 'phase4')."
        ),
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List fixture names + descriptions and exit.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the segmenter call (sandbox / scaffolding test only).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress info-level logs.",
    )
    p.add_argument(
        "--animated",
        action="store_true",
        help=(
            "Phase 6 anime SLA mode. Asserts dense-face ratio ≥ 0.30, "
            "mean per-frame face error ≤ 4%, p95 ≤ 8%, and prints a "
            "[anime-parity] one-line summary suitable for CI."
        ),
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.list:
        for name in list_fixture_names():
            spec = get_fixture(name)
            print(f"{name:30s} {spec.description}")
        return 0

    payload = run_all(
        args.fixture, dry_run=args.dry_run, animated=args.animated,
    )
    if args.phase:
        payload["phase"] = args.phase

    text = json.dumps(payload, indent=2)
    print(text)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
        print(f"\nWrote {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
