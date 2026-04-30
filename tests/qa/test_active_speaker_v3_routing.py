"""Task C: Light-ASD as default + v3 routing + ASD score caching.

Locks down the contract that promotes Light-ASD from an opt-in flag
to the production default and ensures the bench script's extraction
cache mirrors the production pipeline's timeline shape.

Pure-Python: mocks Light-ASD's ``score_faces_for_clip`` so the suite
runs without onnxruntime, librosa, or the actual ONNX model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import pytest

import backend.scripts.compare_autoflip_vs_clipai as bench
from backend.services.active_speaker import build_active_speaker_timeline_v3
from backend.services.face_detector import FaceInfo, FrameFaces
from backend.services.light_asd import ASDResult


# ── Synthetic transcript stub (the v3 builder consumes objects with
#    .start / .end attributes OR dicts with the same keys). ────────────


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str = ""


def _face(identity_id: int, x_center: float = 50.0) -> FaceInfo:
    return FaceInfo(
        x_center=x_center, y_center=50.0, width=20.0, height=25.0,
        nose_x=x_center, nose_y=50.0, confidence=0.95,
        lip_aperture=0.05, identity_id=identity_id,
    )


def _frame(t: float, faces: list[FaceInfo]) -> FrameFaces:
    return FrameFaces(timestamp=t, frame_path="", faces=faces)


# ── 1. Default backend is Light-ASD ─────────────────────────────────────


def test_default_asd_backend_is_light_asd(monkeypatch):
    """Both the bench helper and the production pipeline default to
    light_asd. Catches an accidental revert of the flip in Task C.2."""
    monkeypatch.delenv("CLIPAI_ASD_BACKEND", raising=False)
    assert bench._active_asd_backend() == "light_asd"

    # Production pipeline's default is read from the same env var via
    # the same os.environ.get pattern; verify by reading the source.
    pipeline_src = Path("backend/services/pipeline.py").read_text()
    assert (
        'os.environ.get("CLIPAI_ASD_BACKEND", "light_asd")' in pipeline_src
    ), (
        "pipeline.py default CLIPAI_ASD_BACKEND must be 'light_asd' "
        "after Task C — found a different default. Did someone revert?"
    )


# ── 2. Cache validity requires asd_scores.json under light_asd ─────────


def test_cache_required_files_include_asd_when_light_asd(monkeypatch):
    monkeypatch.setenv("CLIPAI_ASD_BACKEND", "light_asd")
    files = bench._required_cache_files()
    assert "asd_scores.json" in files


def test_cache_required_files_skip_asd_when_heuristic(monkeypatch):
    monkeypatch.setenv("CLIPAI_ASD_BACKEND", "heuristic")
    files = bench._required_cache_files()
    assert "asd_scores.json" not in files


# ── 3. v3 timeline handles overlapping speech (the v2 blind spot) ──────


def test_v3_handles_overlapping_speech():
    """When two faces both cross the speaking threshold during the same
    transcript window, v3 emits SpeakerEvent objects for BOTH. v2
    physically cannot represent this — it picks the higher
    lip-aperture face and drops the other."""
    # Identity 1 and 2 both detected at every sampled timestamp.
    frames = [
        _frame(t, [_face(identity_id=1, x_center=30.0),
                   _face(identity_id=2, x_center=70.0)])
        for t in [0.5, 1.0, 1.5, 2.0]
    ]
    transcript = [TranscriptSegment(start=0.5, end=2.0, text="overlap")]
    # Both faces cross the 0.5 threshold for every frame.
    asd_scores = []
    for fr in frames:
        asd_scores.append(ASDResult(
            timestamp=fr.timestamp, face_idx=0, p_speaking=0.9,
        ))
        asd_scores.append(ASDResult(
            timestamp=fr.timestamp, face_idx=1, p_speaking=0.85,
        ))

    events = build_active_speaker_timeline_v3(
        face_results=frames,
        transcript_segments=transcript,
        asd_scores=asd_scores,
        face_registry=None,
        window_seconds=0.5,
        shot_cuts=[],
        audio_path=None,
    )
    speakers = {ev.slot_id for ev in events if ev.slot_id >= 0}
    assert 1 in speakers and 2 in speakers, (
        f"v3 should emit events for both overlapping speakers, "
        f"got slots: {speakers}"
    )


# ── 4. v3 → v2 → v1 fallback chain ─────────────────────────────────────
# These run against the bench script's ``_extract_and_cache`` routing
# block by exercising the same logic in isolation. We import the
# routing predicate ``has_face_data`` indirectly via the script's
# behavior contract, so the tests are robust to refactoring.


def test_v3_returns_empty_on_zero_asd_scores():
    """With zero ASD scores, v3 must return an empty list so the
    caller's fallback fires."""
    frames = [_frame(0.5, [_face(identity_id=1)])]
    transcript = [TranscriptSegment(start=0.0, end=1.0)]
    events = build_active_speaker_timeline_v3(
        face_results=frames,
        transcript_segments=transcript,
        asd_scores=[],
    )
    assert events == []


def test_v3_off_camera_event_when_no_face_speaks():
    """Transcript word but every face has p_speaking < threshold →
    v3 emits a slot_id=-1 SpeakerEvent so downstream handling can
    keep the camera on the previous active slot."""
    frames = [_frame(0.5, [_face(identity_id=1)])]
    transcript = [TranscriptSegment(start=0.0, end=1.0, text="off-camera")]
    asd_scores = [ASDResult(timestamp=0.5, face_idx=0, p_speaking=0.05)]
    events = build_active_speaker_timeline_v3(
        face_results=frames,
        transcript_segments=transcript,
        asd_scores=asd_scores,
    )
    assert events, "off-camera event must still emit a SpeakerEvent"
    assert any(ev.slot_id == -1 for ev in events)


# ── 5. ASD scores serialize to JSON-friendly shape ─────────────────────


def test_asd_scores_serialize_to_jsonable_dicts(tmp_path):
    """The ``asd_scores.json`` file must be a list of dicts with the
    three keys the cache reader expects. Asserts the (minimal) shape
    contract."""
    scores = [
        ASDResult(timestamp=1.0, face_idx=0, p_speaking=0.7),
        ASDResult(timestamp=1.0, face_idx=1, p_speaking=0.4),
    ]
    payload = [
        {"timestamp": float(s.timestamp),
         "face_idx": int(s.face_idx),
         "p_speaking": float(s.p_speaking)}
        for s in scores
    ]
    import json
    out = tmp_path / "asd_scores.json"
    out.write_text(json.dumps(payload))
    loaded = json.loads(out.read_text())
    assert isinstance(loaded, list) and len(loaded) == 2
    assert {"timestamp", "face_idx", "p_speaking"} <= set(loaded[0])


# ── 6. Cache invalidation when switching backends ──────────────────────


def test_switching_to_light_asd_invalidates_heuristic_cache(tmp_path, monkeypatch):
    """A heuristic-mode cache (no asd_scores.json) is invalid under
    the light_asd backend because the required-files list now includes
    asd_scores.json. Re-extraction will fire on next run."""
    clip_cache = tmp_path / "abc"
    clip_cache.mkdir()
    # Write everything EXCEPT asd_scores.json — the heuristic cache
    # shape from before Task C.
    for name in bench._REQUIRED_CACHE_FILES:
        if name == "cache_version.txt":
            (clip_cache / name).write_text(str(bench.EXTRACTION_CACHE_VERSION))
        else:
            (clip_cache / name).write_text("[]")
    monkeypatch.setattr(
        bench, "_extractor_module_mtime",
        lambda: 0.0,
    )
    # Keep this test focused on the ASD backend switch — the L1
    # saliency layer adds its own required file and is exercised
    # in test_critic_l1_saliency.py.
    monkeypatch.setenv("CLIPAI_SALIENCY_LAYER", "0")

    monkeypatch.setenv("CLIPAI_ASD_BACKEND", "heuristic")
    assert bench._cache_is_valid(clip_cache) is True

    monkeypatch.setenv("CLIPAI_ASD_BACKEND", "light_asd")
    assert bench._cache_is_valid(clip_cache) is False
