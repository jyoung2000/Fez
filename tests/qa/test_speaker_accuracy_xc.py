"""XC QA — speaker-accuracy hardening (burn-in + confidence floor).

Locks down the contract that:
  * the face-registry burn-in helper drops the first N seconds of
    detections from registry input but never empties the list out
    when the clip is short or the post-burn-in window is too sparse
  * v3 active-speaker emits ``slot_id=-1`` when the top two
    candidates are within the confidence-floor margin

Pure-Python tests — no torch, no ffmpeg, no whisper.

Run:  pytest tests/qa/test_speaker_accuracy_xc.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ── Fixtures ─────────────────────────────────────────────────────────


def _frame_with_faces(timestamp: float, n_faces: int = 1):
    """Build a FrameFaces with ``n_faces`` faces at distinct x positions."""
    from backend.services.face_detector import FaceInfo, FrameFaces
    faces = []
    for i in range(n_faces):
        x = (i + 1) * 100.0 / (n_faces + 1)
        faces.append(FaceInfo(
            x_center=x, y_center=50.0,
            width=15.0, height=20.0,
            nose_x=x, nose_y=50.0,
            confidence=0.9,
            identity_id=i,
        ))
    return FrameFaces(timestamp=timestamp, frame_path="", faces=faces)


# ── 1. Burn-in window ────────────────────────────────────────────────


def test_burn_in_filters_early_frames(monkeypatch):
    """Default burn-in (5 s) drops detections at t<5 from registry input."""
    monkeypatch.delenv("CLIPAI_FACE_REGISTRY_BURN_IN_S", raising=False)
    from backend.services.face_registry import apply_face_registry_burn_in

    frames = [
        _frame_with_faces(t, n_faces=2)
        for t in (0.5, 1.5, 2.5, 3.5, 4.5, 6.0, 7.0, 8.0, 9.0, 10.0)
    ]
    out = apply_face_registry_burn_in(frames)
    out_times = [f.timestamp for f in out]
    assert all(t >= 5.0 for t in out_times)
    assert len(out) == 5


def test_burn_in_disabled_via_env(monkeypatch):
    """``CLIPAI_FACE_REGISTRY_BURN_IN_S=0`` returns the input unchanged."""
    monkeypatch.setenv("CLIPAI_FACE_REGISTRY_BURN_IN_S", "0")
    from backend.services.face_registry import apply_face_registry_burn_in

    frames = [_frame_with_faces(t) for t in (0.5, 1.0, 1.5)]
    out = apply_face_registry_burn_in(frames)
    assert [f.timestamp for f in out] == [0.5, 1.0, 1.5]


def test_burn_in_short_clip_returns_unchanged(monkeypatch):
    """Clip shorter than burn-in window keeps every frame."""
    monkeypatch.setenv("CLIPAI_FACE_REGISTRY_BURN_IN_S", "5.0")
    from backend.services.face_registry import apply_face_registry_burn_in

    # Clip spans 0..3 s — well under the 5 s burn-in.
    frames = [_frame_with_faces(t) for t in (0.0, 1.0, 2.0, 3.0)]
    out = apply_face_registry_burn_in(frames)
    assert len(out) == len(frames)


def test_burn_in_falls_back_when_post_window_sparse(monkeypatch):
    """If <3 frames have faces post-burn-in, fall back to unfiltered."""
    monkeypatch.setenv("CLIPAI_FACE_REGISTRY_BURN_IN_S", "5.0")
    from backend.services.face_registry import apply_face_registry_burn_in

    # 5 frames before burn-in (with faces), 2 after (still with faces).
    pre = [_frame_with_faces(t, n_faces=2) for t in (0.5, 1.5, 2.5, 3.5, 4.5)]
    post = [_frame_with_faces(t, n_faces=2) for t in (6.0, 7.0)]
    out = apply_face_registry_burn_in(pre + post)
    # Falls back to unfiltered list.
    assert len(out) == len(pre) + len(post)


# ── 2. Confidence floor in v3 timeline ──────────────────────────────


def test_confidence_floor_emits_unsure_when_top_two_close(monkeypatch):
    """One face crosses threshold, runner-up is within the margin →
    slot_id=-1. (Both crossing threshold is overlap and is preserved.)"""
    monkeypatch.delenv("CLIPAI_ASD_CONFIDENCE_MARGIN", raising=False)
    from backend.services.active_speaker import (
        SpeakerEvent, build_active_speaker_timeline_v3,
    )
    from backend.services.face_detector import FaceInfo, FrameFaces

    # One face above threshold (p=0.55 > 0.5), runner-up at 0.45 —
    # only one identity crosses the threshold but the gap (0.10) is
    # below the default margin (0.15). Floor fires → slot=-1.
    faces = [
        FaceInfo(
            x_center=25.0, y_center=50.0, width=15.0, height=20.0,
            nose_x=25.0, nose_y=50.0, confidence=0.9, identity_id=0,
        ),
        FaceInfo(
            x_center=75.0, y_center=50.0, width=15.0, height=20.0,
            nose_x=75.0, nose_y=50.0, confidence=0.9, identity_id=1,
        ),
    ]
    fr = FrameFaces(timestamp=1.0, frame_path="", faces=faces)

    class _ASD:
        def __init__(self, ts, idx, p):
            self.timestamp = ts
            self.face_idx = idx
            self.p_speaking = p
    asd = [_ASD(1.0, 0, 0.55), _ASD(1.0, 1, 0.45)]

    transcript = [{"start": 0.5, "end": 1.5, "text": "hi"}]
    events = build_active_speaker_timeline_v3(
        face_results=[fr],
        transcript_segments=transcript,
        asd_scores=asd,
        face_registry=None,
        window_seconds=0.6,
    )
    assert events
    # Within the segment span, expect slot=-1 (unsure) rather than two
    # competing co-active SpeakerEvents.
    seg_events = [e for e in events if e.start <= 1.0 <= e.end]
    assert seg_events, "expected at least one event covering t=1.0"
    assert any(e.slot_id == -1 for e in seg_events), (
        "confidence floor should emit slot=-1 for the close-call segment, "
        f"got {[ (e.slot_id, e.confidence) for e in seg_events ]}"
    )


def test_confidence_floor_lets_clear_winner_through(monkeypatch):
    """Top p_speaking >> runner-up → slot_id is the top identity, not -1."""
    monkeypatch.delenv("CLIPAI_ASD_CONFIDENCE_MARGIN", raising=False)
    from backend.services.active_speaker import (
        build_active_speaker_timeline_v3,
    )
    from backend.services.face_detector import FaceInfo, FrameFaces

    faces = [
        FaceInfo(
            x_center=25.0, y_center=50.0, width=15.0, height=20.0,
            nose_x=25.0, nose_y=50.0, confidence=0.9, identity_id=0,
        ),
        FaceInfo(
            x_center=75.0, y_center=50.0, width=15.0, height=20.0,
            nose_x=75.0, nose_y=50.0, confidence=0.9, identity_id=1,
        ),
    ]
    fr = FrameFaces(timestamp=1.0, frame_path="", faces=faces)

    class _ASD:
        def __init__(self, ts, idx, p):
            self.timestamp = ts
            self.face_idx = idx
            self.p_speaking = p
    asd = [_ASD(1.0, 0, 0.85), _ASD(1.0, 1, 0.20)]

    transcript = [{"start": 0.5, "end": 1.5, "text": "hi"}]
    events = build_active_speaker_timeline_v3(
        face_results=[fr],
        transcript_segments=transcript,
        asd_scores=asd,
        face_registry=None,
        window_seconds=0.6,
    )
    seg_events = [e for e in events if e.start <= 1.0 <= e.end]
    # The clear winner emits slot_id=0 (not -1).
    slot_ids = [e.slot_id for e in seg_events]
    assert 0 in slot_ids, (
        f"clear winner (id=0, p=0.85) should be emitted, got {slot_ids}"
    )


def test_confidence_floor_disabled_via_zero_margin(monkeypatch):
    """``CLIPAI_ASD_CONFIDENCE_MARGIN=0`` disables the floor — close
    calls produce a SpeakerEvent for the winning slot instead of -1."""
    monkeypatch.setenv("CLIPAI_ASD_CONFIDENCE_MARGIN", "0")
    from backend.services.active_speaker import (
        build_active_speaker_timeline_v3,
    )
    from backend.services.face_detector import FaceInfo, FrameFaces

    faces = [
        FaceInfo(
            x_center=25.0, y_center=50.0, width=15.0, height=20.0,
            nose_x=25.0, nose_y=50.0, confidence=0.9, identity_id=0,
        ),
        FaceInfo(
            x_center=75.0, y_center=50.0, width=15.0, height=20.0,
            nose_x=75.0, nose_y=50.0, confidence=0.9, identity_id=1,
        ),
    ]
    fr = FrameFaces(timestamp=1.0, frame_path="", faces=faces)

    class _ASD:
        def __init__(self, ts, idx, p):
            self.timestamp = ts
            self.face_idx = idx
            self.p_speaking = p
    asd = [_ASD(1.0, 0, 0.55), _ASD(1.0, 1, 0.45)]

    events = build_active_speaker_timeline_v3(
        face_results=[fr],
        transcript_segments=[{"start": 0.5, "end": 1.5}],
        asd_scores=asd,
        face_registry=None,
        window_seconds=0.6,
    )
    seg_events = [e for e in events if e.start <= 1.0 <= e.end]
    # Floor disabled → the winning slot (id=0) is emitted, not -1.
    slot_ids = [e.slot_id for e in seg_events]
    assert 0 in slot_ids and -1 not in slot_ids
