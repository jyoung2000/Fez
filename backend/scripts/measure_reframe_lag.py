#!/usr/bin/env python3
"""Phase 1 — Measure reframe lag at annotated speaker-change timestamps.

For each annotated speaker-change timestamp in a synthetic fixture, measures:
  lag = crop_switch_time - ground_truth_speaker_change_time  (in frames @ 30fps)

Outputs JSON to stdout and optionally writes to a baseline file.

Usage:
    python -m backend.scripts.measure_reframe_lag [--output path/to/lag_baseline.json]
"""

import json
import sys
from dataclasses import dataclass, field
from typing import Optional

from backend.services.reframe_segmenter import build_reframe_segments


# ── Stubs ──

@dataclass
class _FaceSlot:
    slot_id: int
    x_center: float
    x_min: float = 0.0
    x_max: float = 100.0
    frame_count: int = 100
    avg_width: float = 10.0
    avg_height: float = 12.0


@dataclass
class _FaceRegistry:
    slots: list[_FaceSlot] = field(default_factory=list)
    total_frames: int = 100
    frames_with_faces: int = 100

    @property
    def multi_speaker(self) -> bool:
        return len(self.slots) >= 2

    def slot_by_id(self, slot_id: int):
        for s in self.slots:
            if s.slot_id == slot_id:
                return s
        return None

    def nearest_slot(self, x: float):
        if not self.slots:
            return None
        return min(self.slots, key=lambda s: abs(s.x_center - x))


@dataclass
class _SpeakerEvent:
    start: float
    end: float
    slot_id: int
    confidence: float


@dataclass
class _FaceInfo:
    identity_id: int
    nose_x: float
    nose_y: float = 50.0
    width: float = 10.0
    height: float = 12.0
    lip_aperture: float = 0.05
    is_speaking: bool = True
    confidence: float = 0.95


@dataclass
class _FrameFaces:
    timestamp: float
    frame_path: str = ""
    faces: list[_FaceInfo] = field(default_factory=list)
    primary_face_idx: int = -1


@dataclass
class _TranscriptSeg:
    start: float
    end: float
    text: str = ""
    speaker: str = "Speaker 1"
    confidence: Optional[float] = 0.95
    words: Optional[list] = None
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None


def _make_dense(slot_id, x, start, end, step=0.1):
    frames = []
    t = start
    while t < end:
        frames.append(_FrameFaces(
            timestamp=round(t, 4),
            faces=[_FaceInfo(identity_id=slot_id, nose_x=x)],
        ))
        t += step
    return frames


FPS = 30.0
FRAME_DUR = 1.0 / FPS


def _build_fixture():
    """Build a synthetic fixture with known speaker-change ground truths.

    Returns (segments_kwargs, ground_truth_changes)
    where ground_truth_changes is [(time, from_slot, to_slot), ...]
    """
    registry = _FaceRegistry(slots=[
        _FaceSlot(slot_id=0, x_center=25.0),
        _FaceSlot(slot_id=1, x_center=75.0),
    ])
    speaker_to_slot = {"Speaker 1": 0, "Speaker 2": 1}

    # Ground truth: speakers alternate every 2 seconds for 20s
    changes = []
    transcript = []
    as_events = []
    dense = []

    for i in range(10):
        t_start = i * 2.0
        t_end = (i + 1) * 2.0
        spk = "Speaker 1" if i % 2 == 0 else "Speaker 2"
        slot = 0 if i % 2 == 0 else 1
        x = 25 if slot == 0 else 75

        transcript.append(_TranscriptSeg(start=t_start, end=t_end, speaker=spk))
        as_events.append(_SpeakerEvent(start=t_start, end=t_end, slot_id=slot, confidence=0.95))
        dense.extend(_make_dense(slot, x, t_start, t_end, step=0.1))

        if i > 0:
            prev_slot = 0 if (i - 1) % 2 == 0 else 1
            changes.append((t_start, prev_slot, slot))

    kwargs = dict(
        shot_cuts=[],
        face_registry=registry,
        active_speaker_events=as_events,
        dense_faces=dense,
        transcript_segments=transcript,
        speaker_to_slot=speaker_to_slot,
        video_duration=20.0,
    )

    return kwargs, changes


def _find_crop_switch_time(segments, gt_time, from_slot, to_slot):
    """Find the segment boundary closest to gt_time that represents
    a switch from from_slot to to_slot. Returns the boundary time or None."""
    for i in range(len(segments) - 1):
        a = segments[i]
        b = segments[i + 1]
        if a.active_slot == from_slot and b.active_slot == to_slot:
            boundary = b.start
            if abs(boundary - gt_time) < 2.0:  # within 2s search window
                return boundary
    return None


def measure():
    """Run the fixture and measure lag."""
    kwargs, gt_changes = _build_fixture()
    segments = build_reframe_segments(**kwargs)

    results = {
        "fps": FPS,
        "ground_truth_changes": len(gt_changes),
        "detected_changes": 0,
        "missed_changes": 0,
        "per_change": [],
    }

    for gt_time, from_slot, to_slot in gt_changes:
        crop_time = _find_crop_switch_time(segments, gt_time, from_slot, to_slot)
        if crop_time is None:
            results["missed_changes"] += 1
            results["per_change"].append({
                "ground_truth_time": gt_time,
                "from_slot": from_slot,
                "to_slot": to_slot,
                "crop_switch_time": None,
                "lag_seconds": None,
                "lag_frames": None,
                "status": "missed",
            })
        else:
            lag_s = crop_time - gt_time
            lag_frames = round(lag_s * FPS)
            results["detected_changes"] += 1
            results["per_change"].append({
                "ground_truth_time": gt_time,
                "from_slot": from_slot,
                "to_slot": to_slot,
                "crop_switch_time": round(crop_time, 4),
                "lag_seconds": round(lag_s, 4),
                "lag_frames": lag_frames,
                "status": "detected",
            })

    # Sub-second switch recall
    sub_second_kwargs, sub_second_changes = _build_sub_second_fixture()
    sub_segs = build_reframe_segments(**sub_second_kwargs)
    sub_slot_seq = [s.active_slot for s in sub_segs]
    sub_switch_count = sum(
        1 for i in range(1, len(sub_slot_seq))
        if sub_slot_seq[i] != sub_slot_seq[i - 1]
    )
    results["sub_second_switch_recall"] = {
        "expected_switches": len(sub_second_changes),
        "detected_switches": sub_switch_count,
        "recall_pct": round(
            sub_switch_count / max(len(sub_second_changes), 1) * 100, 1
        ),
    }

    # Overlap count
    overlap_count = 0
    for i in range(len(segments) - 1):
        if segments[i].end > segments[i + 1].start + 1e-9:
            overlap_count += 1
    results["overlap_count"] = overlap_count

    # Summary stats
    detected = [r for r in results["per_change"] if r["status"] == "detected"]
    if detected:
        lags = [abs(r["lag_frames"]) for r in detected]
        results["summary"] = {
            "avg_lag_frames": round(sum(lags) / len(lags), 2),
            "max_lag_frames": max(lags),
            "min_lag_frames": min(lags),
        }
    else:
        results["summary"] = {
            "avg_lag_frames": None,
            "max_lag_frames": None,
            "min_lag_frames": None,
        }

    return results


def _build_sub_second_fixture():
    """Sub-second switches: 5 alternating 400ms segments."""
    registry = _FaceRegistry(slots=[
        _FaceSlot(slot_id=0, x_center=25.0),
        _FaceSlot(slot_id=1, x_center=75.0),
    ])
    speaker_to_slot = {"Speaker 1": 0, "Speaker 2": 1}

    changes = []
    transcript = []
    as_events = []
    dense = []

    for i in range(5):
        t_start = i * 0.4
        t_end = (i + 1) * 0.4
        spk = "Speaker 1" if i % 2 == 0 else "Speaker 2"
        slot = 0 if i % 2 == 0 else 1
        x = 25 if slot == 0 else 75

        transcript.append(_TranscriptSeg(start=t_start, end=t_end, speaker=spk))
        as_events.append(_SpeakerEvent(start=t_start, end=t_end, slot_id=slot, confidence=0.95))
        dense.extend(_make_dense(slot, x, t_start, t_end, step=0.05))

        if i > 0:
            prev_slot = 0 if (i - 1) % 2 == 0 else 1
            changes.append((t_start, prev_slot, slot))

    kwargs = dict(
        shot_cuts=[],
        face_registry=registry,
        active_speaker_events=as_events,
        dense_faces=dense,
        transcript_segments=transcript,
        speaker_to_slot=speaker_to_slot,
        video_duration=2.0,
    )
    return kwargs, changes


if __name__ == "__main__":
    results = measure()
    output = json.dumps(results, indent=2)
    print(output)

    # Write to baseline file if --output specified
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            path = sys.argv[idx + 1]
            with open(path, "w") as f:
                f.write(output)
            print(f"\nBaseline written to {path}", file=sys.stderr)
