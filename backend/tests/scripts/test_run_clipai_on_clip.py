"""Unit tests for
``backend.scripts.compare_autoflip_vs_clipai.run_clipai_on_clip``.

The real invocation path wires a long extraction stack (shots →
dense faces → registry → transcript → active speaker → content
profile → anime anchors → reframe segments) together. This suite
pins each branch without needing GPU, ffmpeg, or MediaPipe — every
external dependency is monkeypatched.

The harness's scoring layer is covered separately by
``test_compare_autoflip_vs_clipai.py``; this file only targets the
invocation helper.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from backend.scripts import compare_autoflip_vs_clipai as harness
from backend.scripts.compare_autoflip_vs_clipai import (
    EXTRACTION_CACHE_VERSION,
    _REQUIRED_CACHE_FILES,
    run_clipai_on_clip,
)
from backend.services.content_classifier import ContentProfile
from backend.services.face_detector import FaceInfo, FrameFaces
from backend.services.face_registry import FaceRegistry, FaceSlot
from backend.services.reframe_segmenter import ReframeSegment
from backend.services.shot_detector import Shot


# ─────────────────── Shared fake video fixture ───────────────────


@pytest.fixture
def fake_video(tmp_path: Path) -> Path:
    """Write a tiny non-empty file that stands in for an .mp4.

    The real invocation path only touches the file for its sha256,
    so the contents don't need to be a valid video — every ffmpeg /
    ffprobe call is monkeypatched.
    """
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"FAKE-MP4-PAYLOAD-FOR-SHA256")
    return path


@pytest.fixture
def fake_cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "extractions"


def _clip_dict(
    slug: str = "test_panel",
    content_type: str = "podcast",
    target: str = "multi_speaker_panel",
    subtype=None,
    duration_sec: float = 2.0,
) -> dict:
    return {
        "slug": slug,
        "content_type": content_type,
        "subtype": subtype,
        "target_clipcontenttype": target,
        "duration_sec": duration_sec,
        "ext": "mp4",
    }


# ─────────────────── Extractor stubs ───────────────────
#
# These helpers are installed as monkeypatches on the source modules
# (not on the harness). ``run_clipai_on_clip`` does ``from
# backend.services.X import Y`` lazily, which re-reads the (patched)
# module attribute and picks up the stub.


def _stub_probe_metadata(video_path: Path) -> dict:
    return {"width": 1920, "height": 1080, "fps": 30.0, "duration": 2.0}


def _stub_extract_audio(video_path: Path, cache_dir: Path) -> Path:
    out = cache_dir / "audio.wav"
    out.write_bytes(b"FAKE-WAV")
    return out


def _stub_transcribe(audio_path: Path) -> list:
    return []


def _stub_detect_shots(video_path, threshold=27.0, video_duration=None):
    return [Shot(index=0, start=0.0, end=float(video_duration or 2.0))]


def _stub_detect_faces_dense(
    video_path, start=0.0, end=0.0, sample_rate=0.5,
    min_confidence=0.4, extract_embeddings=True, progress_callback=None,
    is_animated=False,
):
    """Return a minimal two-frame dense face list with one face each."""
    return [
        FrameFaces(
            timestamp=0.5,
            frame_path="",
            faces=[
                FaceInfo(
                    x_center=50.0, y_center=50.0, width=10.0, height=10.0,
                    nose_x=50.0, nose_y=50.0, confidence=0.9,
                    lip_aperture=0.01, identity_id=0,
                )
            ],
        ),
        FrameFaces(
            timestamp=1.5,
            frame_path="",
            faces=[
                FaceInfo(
                    x_center=55.0, y_center=50.0, width=10.0, height=10.0,
                    nose_x=55.0, nose_y=50.0, confidence=0.9,
                    lip_aperture=0.01, identity_id=0,
                )
            ],
        ),
    ]


def _stub_build_face_registry_with_embeddings(
    face_results, min_appearances=3, cosine_threshold=0.25,
) -> FaceRegistry:
    return FaceRegistry(
        slots=[
            FaceSlot(
                slot_id=0, x_center=50.0, x_min=45.0, x_max=55.0,
                frame_count=len(face_results or []), avg_width=10.0,
                avg_height=10.0,
            )
        ],
        total_frames=len(face_results or []),
        frames_with_faces=len(face_results or []),
    )


def _stub_classify_content_panel(**kwargs) -> ContentProfile:
    """Returns a multi-speaker-panel profile so the pipeline's v2 gate
    does NOT fire (we have no transcript, so panel detection is just
    for the classifier branch)."""
    return ContentProfile(
        content_type="podcast",
        confidence=0.9,
        is_multi_speaker_panel=True,
    )


def _stub_classify_content_talking_head(**kwargs) -> ContentProfile:
    """Returns a talking-head-ish profile so `classify_clip` diverges
    from a manifest target of ``multi_speaker_panel``."""
    return ContentProfile(
        content_type="podcast",
        confidence=0.9,
        is_multi_speaker_panel=False,
    )


def _stub_build_reframe_segments(**kwargs) -> list:
    """Return a single full-clip ReframeSegment so the cache has
    something to serialize and the translator has something to
    convert into events."""
    duration = float(kwargs.get("video_duration") or 2.0)
    return [
        ReframeSegment(
            start=0.0, end=duration,
            subject_x=960.0, subject_y=540.0,  # center of 1920x1080
            layout="single", active_slot=0, confidence=0.9,
            reason="hold", ease_in_ms=0,
        ),
    ]


def _install_extraction_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the full happy-path stub chain. Every external call
    the real invocation path makes is redirected to an in-memory
    stub, so the unit suite runs in the sandbox."""
    from backend.services import (
        active_speaker as _as,
        content_classifier as _cc,
        face_detector as _fd,
        face_registry as _fr,
        reframe_segmenter as _rs,
        shot_detector as _sd,
    )

    monkeypatch.setattr(harness, "_probe_source_metadata", _stub_probe_metadata)
    monkeypatch.setattr(harness, "_extract_audio_wav", _stub_extract_audio)
    monkeypatch.setattr(harness, "_transcribe_sync", _stub_transcribe)

    monkeypatch.setattr(_sd, "detect_shots", _stub_detect_shots)
    monkeypatch.setattr(_fd, "detect_faces_dense", _stub_detect_faces_dense)
    monkeypatch.setattr(
        _fr, "build_face_registry_with_embeddings",
        _stub_build_face_registry_with_embeddings,
    )
    monkeypatch.setattr(
        _as, "build_active_speaker_timeline_v2", lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        _as, "build_active_speaker_timeline", lambda *a, **kw: [],
    )
    monkeypatch.setattr(_cc, "classify_content", _stub_classify_content_panel)
    monkeypatch.setattr(_rs, "build_reframe_segments", _stub_build_reframe_segments)


# ─────────────────── Test 1: stub / dry_run mode ───────────────────


def test_dry_run_returns_synthetic_timeline():
    """``dry_run=True`` keeps the legacy N-frame synthetic output so
    the scoring layer iteration path stays working. This is the same
    assertion the existing harness tests rely on."""
    clip = _clip_dict(duration_sec=10)
    events = run_clipai_on_clip(clip, video_path=None, dry_run=True)
    assert events is not None
    # 30 fps * 10 sec = 300 frames
    assert len(events) == 300
    assert events[0]["scene_change"] is True
    assert events[0]["crop_w"] == pytest.approx(0.3164)
    assert events[0]["crop_h"] == 1.0
    # Interior frames have scene_change=False
    assert events[10]["scene_change"] is False


def test_video_path_none_also_returns_synthetic():
    """When ``video_path is None`` (not just dry_run), the synthetic
    path still fires so the harness's ``--dry-run`` default flow
    stays backwards compatible."""
    events = run_clipai_on_clip(_clip_dict(), video_path=None)
    assert events is not None
    assert len(events) > 0


# ─────────────────── Test 2: missing video path ───────────────────


def test_missing_video_path_returns_none_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
):
    """A supplied video path that doesn't resolve on disk returns
    ``None`` (so the harness writes a ``clipai_skipped`` row) and
    logs a WARNING so the operator notices."""
    missing = tmp_path / "does_not_exist.mp4"
    clip = _clip_dict()
    caplog.set_level(logging.WARNING, logger="compare_autoflip_vs_clipai")
    events = run_clipai_on_clip(clip, video_path=missing)
    assert events is None
    assert any(
        "video not found" in rec.message
        for rec in caplog.records
    )


# ─────────────────── Test 3: cache-hit path ───────────────────


def _write_valid_cache(clip_cache: Path, *, duration: float = 2.0) -> None:
    """Pre-populate every required file under ``clip_cache`` so the
    cache-hit path finds a valid cache and skips all extractors."""
    clip_cache.mkdir(parents=True, exist_ok=True)
    metadata = {
        "slug": "test_panel",
        "sha256": "deadbeef",
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "duration": duration,
    }
    (clip_cache / "metadata.json").write_text(json.dumps(metadata))
    (clip_cache / "shots.json").write_text(json.dumps(
        [{"index": 0, "start": 0.0, "end": duration}],
    ))
    (clip_cache / "dense_faces.json").write_text("[]")
    (clip_cache / "face_registry.json").write_text(json.dumps(
        {"slots": [], "total_frames": 0, "frames_with_faces": 0},
    ))
    (clip_cache / "transcript.json").write_text("[]")
    (clip_cache / "speaker_events.json").write_text("[]")
    (clip_cache / "content_profile.json").write_text(json.dumps(
        {
            "content_type": "podcast",
            "confidence": 0.9,
            "is_multi_speaker_panel": True,
            "is_animated": False,
            "anime_subtype": None,
            "music_subtype": None,
            "gameplay_subtype": None,
            "game_type": "",
        },
    ))
    (clip_cache / "anime_anchors.json").write_text("[]")
    (clip_cache / "segments.json").write_text(json.dumps([
        {
            "start": 0.0, "end": duration,
            "subject_x": 960.0, "subject_y": 540.0,
            "layout": "single", "active_slot": 0,
            "confidence": 0.9, "reason": "hold", "ease_in_ms": 0,
            "strategy": "stationary", "content_type": "podcast",
        },
    ]))
    (clip_cache / "cache_version.txt").write_text(
        str(EXTRACTION_CACHE_VERSION),
    )


def test_cache_hit_skips_extraction(
    fake_video: Path,
    fake_cache_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """When the cache is valid, ``detect_faces_dense`` and
    ``transcribe_audio`` must NOT fire. We install raising stubs so
    any call would immediately explode — then we call the function
    and verify it returns a well-formed event list."""
    # Compute the sha so we know where the cache should live
    import hashlib
    sha = hashlib.sha256(fake_video.read_bytes()).hexdigest()
    clip_cache = fake_cache_dir / sha
    _write_valid_cache(clip_cache)

    # Poison every extractor: calling one should explode.
    def _poison(*a, **kw):
        raise AssertionError("extractor fired on cache hit")

    from backend.services import face_detector as _fd
    from backend.services import shot_detector as _sd
    from backend.services import reframe_segmenter as _rs
    monkeypatch.setattr(_fd, "detect_faces_dense", _poison)
    monkeypatch.setattr(_sd, "detect_shots", _poison)
    monkeypatch.setattr(_rs, "build_reframe_segments", _poison)
    monkeypatch.setattr(harness, "_probe_source_metadata", _poison)
    monkeypatch.setattr(harness, "_extract_audio_wav", _poison)
    monkeypatch.setattr(harness, "_transcribe_sync", _poison)

    events = run_clipai_on_clip(
        _clip_dict(), video_path=fake_video, cache_dir=fake_cache_dir,
    )
    assert events is not None
    assert len(events) > 0
    # 9:16 crop from 1920x1080
    assert events[0]["crop_w"] == pytest.approx(0.3164, abs=1e-3)
    assert events[0]["crop_h"] == pytest.approx(1.0)
    assert events[0]["scene_change"] is True


def test_cache_hit_uses_counter_wrapped_detector(
    fake_video: Path, fake_cache_dir: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Acceptance checklist item: wrap ``detect_faces_dense`` in a
    counter and verify the second call against the same clip keeps
    that counter at ZERO (i.e. the cache hit bypasses it).

    We wrap ALL stubs in a counter, pre-populate a valid cache for
    the first call, and assert the wrapped counter stays at zero.
    """
    import hashlib
    sha = hashlib.sha256(fake_video.read_bytes()).hexdigest()
    clip_cache = fake_cache_dir / sha
    _write_valid_cache(clip_cache)

    call_count = {"detect_faces_dense": 0}

    def _counted_detect_faces_dense(*a, **kw):
        call_count["detect_faces_dense"] += 1
        return _stub_detect_faces_dense(*a, **kw)

    from backend.services import face_detector as _fd
    monkeypatch.setattr(_fd, "detect_faces_dense", _counted_detect_faces_dense)

    events = run_clipai_on_clip(
        _clip_dict(), video_path=fake_video, cache_dir=fake_cache_dir,
    )
    assert events is not None
    assert call_count["detect_faces_dense"] == 0


# ─────────────────── Test 4: cache-miss path ───────────────────


def test_cache_miss_populates_all_required_files(
    fake_video: Path,
    fake_cache_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Cold cache: every extractor is monkeypatched to a small
    canned result, the function runs the full stack, and the
    resulting cache directory contains every required file exactly
    matching the extraction contract."""
    _install_extraction_stubs(monkeypatch)

    events = run_clipai_on_clip(
        _clip_dict(), video_path=fake_video, cache_dir=fake_cache_dir,
    )
    assert events is not None
    assert len(events) > 0
    # 9:16 crop from 1920x1080: crop_w ≈ 0.3164
    assert events[0]["crop_w"] == pytest.approx(0.3164, abs=1e-3)
    assert events[0]["crop_h"] == pytest.approx(1.0)

    # Find the cache dir (sha256-indexed subdir of cache_dir).
    import hashlib
    sha = hashlib.sha256(fake_video.read_bytes()).hexdigest()
    clip_cache = fake_cache_dir / sha
    assert clip_cache.is_dir()
    # Every file the contract requires must exist.
    for fname in _REQUIRED_CACHE_FILES:
        assert (clip_cache / fname).is_file(), f"missing {fname}"
    # cache_version.txt contains the current version
    assert (
        (clip_cache / "cache_version.txt").read_text().strip()
        == str(EXTRACTION_CACHE_VERSION)
    )
    # segments.json is non-empty
    segs = json.loads((clip_cache / "segments.json").read_text())
    assert len(segs) == 1
    assert segs[0]["subject_x"] == 960.0


# ─────────────────── Test 5: cache version mismatch ───────────────


def test_cache_version_mismatch_forces_rebuild(
    fake_video: Path,
    fake_cache_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A cache with an outdated ``cache_version.txt`` must be wiped
    and rebuilt from scratch."""
    import hashlib
    sha = hashlib.sha256(fake_video.read_bytes()).hexdigest()
    clip_cache = fake_cache_dir / sha
    _write_valid_cache(clip_cache)
    # Downgrade the version: this should trigger a full rebuild.
    (clip_cache / "cache_version.txt").write_text("0")
    # Leave a sentinel file that survives if the rebuild fails to
    # wipe the directory.
    (clip_cache / "stale_sentinel.txt").write_text("should be wiped")

    _install_extraction_stubs(monkeypatch)

    events = run_clipai_on_clip(
        _clip_dict(), video_path=fake_video, cache_dir=fake_cache_dir,
    )
    assert events is not None
    # Version got rewritten to the current contract
    assert (
        (clip_cache / "cache_version.txt").read_text().strip()
        == str(EXTRACTION_CACHE_VERSION)
    )
    # Stale sentinel got nuked by the wipe-and-rebuild step
    assert not (clip_cache / "stale_sentinel.txt").exists()


# ─────────────────── Test 6: content-type mismatch warning ──────


def test_content_type_mismatch_logs_warning_but_still_returns_events(
    fake_video: Path,
    fake_cache_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """When the classifier routes the clip to a ClipContentType
    different from the manifest's ``target_clipcontenttype``, the
    function must log a WARNING (for the rollup to capture the
    divergence) but still return a valid event list."""
    _install_extraction_stubs(monkeypatch)
    # Override classify_content to produce a non-panel profile.
    from backend.services import content_classifier as _cc
    monkeypatch.setattr(
        _cc, "classify_content", _stub_classify_content_talking_head,
    )

    caplog.set_level(logging.WARNING, logger="compare_autoflip_vs_clipai")

    clip = _clip_dict(target="multi_speaker_panel")
    events = run_clipai_on_clip(
        clip, video_path=fake_video, cache_dir=fake_cache_dir,
        force_reextract=True,
    )
    # Still returns events
    assert events is not None
    assert len(events) > 0
    # Divergence is logged
    divergence_logs = [
        rec for rec in caplog.records
        if "divergence" in rec.message
    ]
    assert divergence_logs, "expected classify_clip divergence warning"
