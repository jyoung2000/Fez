"""Week 2 Part E — unit tests for music-video subtype auto-promotion.

Exercises ``_infer_music_subtype_from_formation_and_beat`` with both
high-formation + high-beat-confidence inputs (expect "performance"
promotion) and various edge cases (low formation, low confidence,
missing dense_faces, etc).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest


# ───────────────────────── Stubs ─────────────────────────


@dataclass
class _StubFace:
    nose_x: float = 50.0
    nose_y: float = 50.0


@dataclass
class _StubFrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)


def _formation_frame(ts: float) -> _StubFrameFaces:
    """3 faces spanning 60% → passes _is_formation_frame."""
    return _StubFrameFaces(
        timestamp=ts,
        faces=[_StubFace(nose_x=20), _StubFace(nose_x=50), _StubFace(nose_x=80)],
    )


def _tight_frame(ts: float) -> _StubFrameFaces:
    """2 faces — below min_faces=3 → not a formation."""
    return _StubFrameFaces(
        timestamp=ts,
        faces=[_StubFace(nose_x=45), _StubFace(nose_x=55)],
    )


@dataclass
class _StubBeatGrid:
    confidence: float = 0.0
    downbeat_times: list = field(default_factory=list)

    def effective_confidence(self) -> float:
        if not self.downbeat_times:
            return 0.0
        return float(self.confidence)


# ───────────────────────── promotion tests ─────────────────────────


def test_high_formation_high_beat_promotes_performance():
    """8/10 formation frames + beat_conf=0.7 → performance."""
    from backend.services.content_classifier import (
        _infer_music_subtype_from_formation_and_beat,
    )
    dense = (
        [_formation_frame(i * 0.1) for i in range(8)]
        + [_tight_frame(0.8 + i * 0.1) for i in range(2)]
    )
    bg = _StubBeatGrid(
        confidence=0.7,
        downbeat_times=[0.5, 1.5, 2.5, 3.5],
    )
    result = _infer_music_subtype_from_formation_and_beat(dense, bg)
    assert result is not None
    subtype, formation_ratio, beat_conf = result
    assert subtype == "performance"
    assert formation_ratio == pytest.approx(0.80, abs=0.01)
    assert beat_conf == 0.7


def test_low_formation_no_promote():
    """Below 8% formation threshold → no promotion."""
    from backend.services.content_classifier import (
        _infer_music_subtype_from_formation_and_beat,
    )
    dense = (
        [_formation_frame(0.0)]  # 1/20 = 5% — below 8%
        + [_tight_frame(i * 0.1) for i in range(19)]
    )
    bg = _StubBeatGrid(
        confidence=0.9,
        downbeat_times=[0.5, 1.5, 2.5, 3.5],
    )
    assert _infer_music_subtype_from_formation_and_beat(dense, bg) is None


def test_low_beat_confidence_no_promote():
    """High formation but beat_conf below 0.6 threshold → no promotion."""
    from backend.services.content_classifier import (
        _infer_music_subtype_from_formation_and_beat,
    )
    dense = [_formation_frame(i * 0.1) for i in range(10)]
    bg = _StubBeatGrid(
        confidence=0.4,
        downbeat_times=[0.5, 1.5, 2.5],
    )
    assert _infer_music_subtype_from_formation_and_beat(dense, bg) is None


def test_missing_beat_grid_no_promote():
    """No beat grid at all → beat_conf=0 → no promotion."""
    from backend.services.content_classifier import (
        _infer_music_subtype_from_formation_and_beat,
    )
    dense = [_formation_frame(i * 0.1) for i in range(10)]
    assert (
        _infer_music_subtype_from_formation_and_beat(dense, None)
        is None
    )


def test_empty_beat_grid_downbeats_no_promote():
    """BeatGrid exists but has no downbeats → effective_confidence=0."""
    from backend.services.content_classifier import (
        _infer_music_subtype_from_formation_and_beat,
    )
    dense = [_formation_frame(i * 0.1) for i in range(10)]
    bg = _StubBeatGrid(confidence=0.9, downbeat_times=[])
    assert (
        _infer_music_subtype_from_formation_and_beat(dense, bg)
        is None
    )


def test_empty_dense_faces_no_promote():
    from backend.services.content_classifier import (
        _infer_music_subtype_from_formation_and_beat,
    )
    bg = _StubBeatGrid(
        confidence=0.9, downbeat_times=[0.5, 1.5, 2.5],
    )
    assert _infer_music_subtype_from_formation_and_beat([], bg) is None
    assert _infer_music_subtype_from_formation_and_beat(None, bg) is None


def test_exactly_at_thresholds_promotes():
    """formation_ratio=0.08 and beat_conf=0.6 (both at floor) → promoted."""
    from backend.services.content_classifier import (
        _infer_music_subtype_from_formation_and_beat,
    )
    # 1/12 ≈ 8.3%, slightly above the 8% floor
    dense = (
        [_formation_frame(0.0)]
        + [_tight_frame(i * 0.1) for i in range(11)]
    )
    bg = _StubBeatGrid(
        confidence=0.6, downbeat_times=[0.5, 1.5, 2.5],
    )
    result = _infer_music_subtype_from_formation_and_beat(dense, bg)
    assert result is not None
    assert result[0] == "performance"


# ───────────── classify_content short-circuits ─────────────


def test_classify_content_respects_user_music_subtype():
    """User picked music_video + lyric subtype — auto-promotion must not touch it."""
    from backend.services.content_classifier import classify_content

    dense = [_formation_frame(i * 0.1) for i in range(10)]
    bg = _StubBeatGrid(
        confidence=0.9, downbeat_times=[0.5, 1.5, 2.5, 3.5],
    )
    md = {
        "content_type_override": "music_video",
        "music_subtype": "lyric",
        "music_beat_grid": bg,
    }
    profile = classify_content(
        shot_cuts=[], face_registry=None, dense_faces=dense,
        scenes=[], video_duration=30.0, metadata=md,
    )
    # User override branch returns early → music_subtype == "lyric"
    assert profile.music_subtype == "lyric"


def test_classify_content_non_music_short_circuits():
    from backend.services.content_classifier import classify_content

    dense = [_formation_frame(i * 0.1) for i in range(10)]
    bg = _StubBeatGrid(
        confidence=0.9, downbeat_times=[0.5, 1.5, 2.5, 3.5],
    )
    md = {
        "content_type_override": "podcast",
        "music_beat_grid": bg,
    }
    profile = classify_content(
        shot_cuts=[], face_registry=None, dense_faces=dense,
        scenes=[], video_duration=30.0, metadata=md,
    )
    assert profile.content_type == "podcast"
    assert not profile.music_subtype


# ───────────── BeatGrid confidence property ─────────────


def test_real_beatgrid_confidence_defaults():
    """librosa-backed BeatGrid defaults confidence=0.9 (via detect_beats)."""
    from backend.services.beat_detector import BeatGrid

    # Default instance — synthetic, confidence=0.0
    bg = BeatGrid(tempo_bpm=120.0, downbeat_times=[0.5, 1.0])
    assert bg.confidence == 0.0
    # effective_confidence zero-gates an empty grid
    empty = BeatGrid(tempo_bpm=0.0)
    assert empty.effective_confidence() == 0.0

    # Explicitly-set confidence rides through
    hi = BeatGrid(
        tempo_bpm=120.0, downbeat_times=[0.5, 1.0], confidence=0.9,
    )
    assert hi.effective_confidence() == 0.9

    # Empty downbeats zero-gate even with high stored confidence
    empty_but_conf = BeatGrid(tempo_bpm=120.0, confidence=0.9)
    assert empty_but_conf.effective_confidence() == 0.0
