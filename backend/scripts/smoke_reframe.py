"""Minimal smoke test for the reframe pipeline.

Usage:
    USE_AUTOFLIP_REFRAME=false python -m backend.scripts.smoke_reframe [video_path]
    USE_AUTOFLIP_REFRAME=true  python -m backend.scripts.smoke_reframe [video_path]

If no video path is provided, generates a synthetic 10s fixture.
"""

import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MockFace:
    nose_x: float = 50.0
    nose_y: float = 40.0
    x: float = 50.0
    y: float = 40.0
    width: float = 10.0
    height: float = 13.0
    identity_id: int = 0
    lip_aperture: float = 0.03
    is_speaking: bool = False
    identity_embedding: Optional[list] = None


@dataclass
class MockFrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)
    primary_face_idx: int = 0
    path: str = ""


@dataclass
class MockFaceSlot:
    slot_id: int = 0
    x_center: float = 50.0
    x_min: float = 35.0
    x_max: float = 65.0
    frame_count: int = 20
    avg_width: float = 10.0
    avg_height: float = 13.0


@dataclass
class MockFaceRegistry:
    slots: list = field(default_factory=list)
    total_frames: int = 20
    frames_with_faces: int = 20

    @property
    def multi_speaker(self):
        return len(self.slots) >= 2

    @property
    def is_continuous_motion(self):
        return False

    def nearest_slot(self, x):
        if not self.slots:
            return None
        return min(self.slots, key=lambda s: abs(s.x_center - x))

    def slot_by_id(self, slot_id):
        for s in self.slots:
            if s.slot_id == slot_id:
                return s
        return None


@dataclass
class MockSpeakerEvent:
    start: float
    end: float
    slot_id: int
    confidence: float = 0.9


@dataclass
class MockTranscriptSeg:
    start: float
    end: float
    speaker: str = "SPEAKER_00"
    confidence: float = 0.95
    words: list = field(default_factory=list)


def main():
    use_autoflip = os.environ.get("USE_AUTOFLIP_REFRAME", "false").lower() in ("true", "1", "yes")
    mode = "AUTOFLIP" if use_autoflip else "SEGMENTER"
    print(f"Smoke reframe test — mode: {mode}")

    duration = 10.0
    fps = 2
    n_frames = int(duration * fps)

    # Build synthetic data: face walks from x=30 to x=70
    dense_faces = []
    for i in range(n_frames):
        t = i / fps
        x = 30 + 40 * (i / max(n_frames - 1, 1))
        dense_faces.append(MockFrameFaces(
            timestamp=t,
            faces=[MockFace(nose_x=x, nose_y=40)],
        ))

    registry = MockFaceRegistry(slots=[MockFaceSlot(slot_id=0, x_center=50)])
    speaker_events = [MockSpeakerEvent(start=0, end=duration, slot_id=0)]
    transcript = [MockTranscriptSeg(start=0, end=duration)]

    if use_autoflip:
        from backend.services.autoflip_segmenter import build_autoflip_segments
        segments = build_autoflip_segments(
            shot_cuts=[],
            face_registry=registry,
            active_speaker_events=speaker_events,
            dense_faces=dense_faces,
            saliency_keyframes=[],
            transcript_segments=transcript,
            speaker_to_slot={"SPEAKER_00": 0},
            video_duration=duration,
            source_width=1280, source_height=720,
            job_id="smoke",
        )
    else:
        from backend.services.reframe_segmenter import build_reframe_segments
        segments = build_reframe_segments(
            shot_cuts=[],
            face_registry=registry,
            active_speaker_events=speaker_events,
            dense_faces=dense_faces,
            transcript_segments=transcript,
            speaker_to_slot={"SPEAKER_00": 0},
            video_duration=duration,
            source_width=1280, source_height=720,
            job_id="smoke",
        )

    print(f"\nSegments ({len(segments)}):")
    for s in segments:
        print(f"  t={s.start:.2f}-{s.end:.2f}  strategy={s.strategy}  "
              f"subject_x={s.subject_x}  confidence={s.confidence:.2f}  "
              f"motion_path={'yes' if s.motion_path else 'no'}")

    # Build render plan
    from backend.services.render_plan_builder import build_render_plan
    plan = build_render_plan(
        segments=segments,
        source_width=1280, source_height=720,
        source_fps=30.0, target_aspect="9:16",
    )
    violations = plan.validate()
    if violations:
        print(f"\nRenderPlan INVALID: {violations}")
        sys.exit(1)
    else:
        print(f"\nRenderPlan OK: {len(plan.ops)} ops, {plan.total_duration_sec:.1f}s")

    print("\nSmoke test PASSED")


if __name__ == "__main__":
    main()
