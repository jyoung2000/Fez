"""Phase C — Light-ASD audio-visual active speaker tests.

Validates the v3 timeline builder and the Light-ASD wrapper without
needing the real onnxruntime / model. Coverage:
  - graceful fallback when onnxruntime is missing
  - empty face_results returns []
  - v3 timeline picks the face with high p_speaking
  - v3 represents overlapping speakers (a v2-impossible case)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

# Pre-mock heavy native deps.
for _mod in (
    "google.generativeai", "google.generativeai.types", "google.ai",
    "google.ai.generativelanguage_v1beta", "anthropic", "groq", "httpx",
    "openai", "ctranslate2", "faster_whisper", "mediapipe",
    "cv2", "librosa",
):
    sys.modules.setdefault(_mod, MagicMock())


@dataclass
class _Face:
    x_center: float = 50.0
    y_center: float = 50.0
    width: float = 12.0
    height: float = 14.0
    nose_x: float = 50.0
    nose_y: float = 50.0
    identity_id: int = 0
    is_speaking: bool = False
    lip_aperture: float = 0.0


@dataclass
class _FF:
    timestamp: float
    frame_path: str = ""
    faces: list = field(default_factory=list)


@dataclass
class _Seg:
    start: float
    end: float


def test_light_asd_falls_back_when_onnxruntime_missing(monkeypatch):
    """If onnxruntime can't be imported, _get_session returns None and
    score_faces_for_clip returns an empty list (the pipeline.py wiring
    then falls back to the v2 heuristic)."""
    import backend.services.light_asd as la
    monkeypatch.setattr(la, "_ASD_SESSION", None, raising=False)

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("forced — onnxruntime not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)

    sess = la._get_session()
    assert sess is None
    out = la.score_faces_for_clip(
        video_path="/tmp/_v.mp4",
        face_results=[_FF(timestamp=0.0, faces=[_Face()])],
        audio_wav_path="/tmp/_a.wav",
    )
    assert out == []


def test_light_asd_handles_empty_face_results(monkeypatch):
    import backend.services.light_asd as la
    monkeypatch.setattr(la, "_ASD_SESSION", MagicMock())
    out = la.score_faces_for_clip(
        video_path="/tmp/_v.mp4",
        face_results=[],
        audio_wav_path="/tmp/_a.wav",
    )
    assert out == []


def test_active_speaker_v3_uses_p_speaking(monkeypatch):
    """When face_idx=0 has p_speaking=0.9 and face_idx=1 has 0.1, the
    timeline must attribute the speech to slot 0."""
    import backend.services.active_speaker as asm
    from backend.services.light_asd import ASDResult

    # Stub out the v2 helpers so we don't pull cv2 / pyannote.
    monkeypatch.setattr(asm, "_collapse_short_runs",
                        lambda evs: (evs, 0))
    monkeypatch.setattr(asm, "_apply_shot_reverse_tolerance",
                        lambda evs, *a, **k: (evs, 0))

    f0 = _Face(identity_id=0, x_center=25.0, nose_x=25.0)
    f1 = _Face(identity_id=1, x_center=75.0, nose_x=75.0)
    fr_a = _FF(timestamp=1.0, frame_path="/tmp/a.png", faces=[f0, f1])
    fr_b = _FF(timestamp=1.5, frame_path="/tmp/b.png", faces=[f0, f1])

    asd = [
        ASDResult(timestamp=1.0, face_idx=0, p_speaking=0.9),
        ASDResult(timestamp=1.0, face_idx=1, p_speaking=0.1),
        ASDResult(timestamp=1.5, face_idx=0, p_speaking=0.85),
        ASDResult(timestamp=1.5, face_idx=1, p_speaking=0.15),
    ]
    transcript = [_Seg(start=1.0, end=1.5)]

    events = asm.build_active_speaker_timeline_v3(
        face_results=[fr_a, fr_b],
        transcript_segments=transcript,
        asd_scores=asd,
    )
    assert events
    # Only one identity above threshold → exactly one event for slot 0.
    assert all(e.slot_id == 0 for e in events)
    assert events[0].confidence >= 0.5


def test_active_speaker_v3_handles_overlap(monkeypatch):
    """Two faces both above threshold simultaneously must emit two
    co-active SpeakerEvents — v2 cannot do this."""
    import backend.services.active_speaker as asm
    from backend.services.light_asd import ASDResult

    monkeypatch.setattr(asm, "_collapse_short_runs",
                        lambda evs: (evs, 0))
    monkeypatch.setattr(asm, "_apply_shot_reverse_tolerance",
                        lambda evs, *a, **k: (evs, 0))

    f0 = _Face(identity_id=0, x_center=25.0, nose_x=25.0)
    f1 = _Face(identity_id=1, x_center=75.0, nose_x=75.0)
    fr = _FF(timestamp=2.0, frame_path="/tmp/c.png", faces=[f0, f1])

    asd = [
        ASDResult(timestamp=2.0, face_idx=0, p_speaking=0.78),
        ASDResult(timestamp=2.0, face_idx=1, p_speaking=0.72),
    ]
    transcript = [_Seg(start=2.0, end=2.0)]

    events = asm.build_active_speaker_timeline_v3(
        face_results=[fr],
        transcript_segments=transcript,
        asd_scores=asd,
    )
    assert events
    slot_ids = sorted(e.slot_id for e in events)
    assert slot_ids == [0, 1], (
        f"v3 must emit two co-active SpeakerEvents, got {slot_ids}"
    )


def test_active_speaker_v3_returns_empty_without_asd_scores(monkeypatch):
    import backend.services.active_speaker as asm
    f0 = _Face(identity_id=0)
    fr = _FF(timestamp=1.0, faces=[f0])
    out = asm.build_active_speaker_timeline_v3(
        face_results=[fr],
        transcript_segments=[_Seg(0.5, 1.5)],
        asd_scores=[],
    )
    assert out == []
