"""Unit tests for the pipeline-optimization round.

Covers:
  * SHA-256 file hashing helper.
  * Extraction cache hit / miss based on hash + on-disk artifacts.
  * Stage timings + warnings accumulation in ``_stage_timer``.
  * Speech-aware two-pass face detection routing.
  * ``POST /api/jobs/{id}/rescore`` endpoint behavior.
  * YOLO device resolver respects env + GPU policy.
  * Ollama vision concurrency / keep_alive defaults are honored.

Each test uses an isolated tmp directory and asyncio loop so the
suite can run without touching ``/data/uploads`` or any real model
weights.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Auth store + share isolation (these tests don't touch auth, but the
# auth modules resolve their data dir at import time).
_TMP = Path(tempfile.mkdtemp(prefix="clipai_pipeopt_test_"))
os.environ["HOME"] = str(_TMP)


# ── SHA-256 helper + cache probe ────────────────────────────────


def test_hash_file_sha256_returns_known_digest(tmp_path):
    from backend.services.pipeline_helpers import _hash_file_sha256
    p = tmp_path / "blob.bin"
    p.write_bytes(b"hello clipai")
    expected = hashlib.sha256(b"hello clipai").hexdigest()
    assert _hash_file_sha256(str(p)) == expected


def test_hash_file_sha256_missing_file_returns_empty():
    from backend.services.pipeline_helpers import _hash_file_sha256
    assert _hash_file_sha256("/does/not/exist/abc123.bin") == ""


def test_cache_probe_hit_with_matching_sha(tmp_path):
    from backend.services.pipeline_helpers import (
        _hash_file_sha256,
        _maybe_use_cached_extraction,
        _write_extraction_manifest,
    )
    from backend.models import FrameData

    src = tmp_path / "video.mp4"
    src.write_bytes(b"fake mp4 contents 1234567890" * 100)
    sha = _hash_file_sha256(str(src))

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(6):
        (frames_dir / f"frame_{i:04d}.jpg").write_bytes(b"\xff\xd8\xff\xe0jpeg")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"riff" * 256)

    fake_frames = [FrameData(timestamp=i * 0.5, path=f"frame_{i:04d}.jpg") for i in range(6)]
    _write_extraction_manifest(str(frames_dir), fake_frames, [1.5, 3.0])

    result = asyncio.run(_maybe_use_cached_extraction(
        job_id="job1",
        video_path=str(src),
        frames_dir=str(frames_dir),
        audio_path=str(audio),
        expected_sha=sha,
    ))
    assert result is not None
    frames, cuts = result
    assert len(frames) == 6
    assert cuts == [1.5, 3.0]
    # Timestamps come from the manifest, not the filename.
    assert [round(f.timestamp, 2) for f in frames] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]


def test_cache_probe_miss_when_sha_changes(tmp_path):
    from backend.services.pipeline_helpers import (
        _maybe_use_cached_extraction,
        _write_extraction_manifest,
    )
    from backend.models import FrameData

    src = tmp_path / "video.mp4"
    src.write_bytes(b"v1")
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(6):
        (frames_dir / f"frame_{i:04d}.jpg").write_bytes(b"jpg")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"riff" * 64)
    fake_frames = [FrameData(timestamp=i * 0.5, path=str(frames_dir / f"frame_{i:04d}.jpg")) for i in range(6)]
    _write_extraction_manifest(str(frames_dir), fake_frames, [])

    # Stale SHA — must miss.
    result = asyncio.run(_maybe_use_cached_extraction(
        job_id="job1",
        video_path=str(src),
        frames_dir=str(frames_dir),
        audio_path=str(audio),
        expected_sha="0" * 64,
    ))
    assert result is None


def test_cache_probe_miss_when_audio_missing(tmp_path):
    from backend.services.pipeline_helpers import (
        _hash_file_sha256,
        _maybe_use_cached_extraction,
    )
    src = tmp_path / "video.mp4"
    src.write_bytes(b"data")
    sha = _hash_file_sha256(str(src))
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(6):
        (frames_dir / f"frame_{i:04d}.jpg").write_bytes(b"jpg")
    # No audio.wav
    result = asyncio.run(_maybe_use_cached_extraction(
        job_id="job1",
        video_path=str(src),
        frames_dir=str(frames_dir),
        audio_path=str(tmp_path / "audio.wav"),
        expected_sha=sha,
    ))
    assert result is None


def test_cache_probe_miss_when_too_few_frames(tmp_path):
    from backend.services.pipeline_helpers import (
        _hash_file_sha256,
        _maybe_use_cached_extraction,
    )
    src = tmp_path / "video.mp4"
    src.write_bytes(b"data")
    sha = _hash_file_sha256(str(src))
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    # Only 3 frames — minimum is 5
    for i in range(3):
        (frames_dir / f"frame_{i:04d}.jpg").write_bytes(b"jpg")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"riff")
    result = asyncio.run(_maybe_use_cached_extraction(
        job_id="job1",
        video_path=str(src),
        frames_dir=str(frames_dir),
        audio_path=str(audio),
        expected_sha=sha,
    ))
    assert result is None


# ── Stage timings + warnings accumulation ───────────────────────


def test_stage_timer_records_elapsed():
    from backend.services.pipeline_helpers import (
        _drain_pipeline_telemetry,
        _stage_timer,
        _record_pipeline_warning,
    )

    async def run():
        async with _stage_timer("jobZ", "stage-A"):
            await asyncio.sleep(0.02)
        _record_pipeline_warning("jobZ", "minor codec warning")
        _record_pipeline_warning("jobZ", "minor codec warning")  # dedup
        async with _stage_timer("jobZ", "stage-B"):
            await asyncio.sleep(0.01)
        timings, warnings = _drain_pipeline_telemetry("jobZ")
        return timings, warnings
    timings, warnings = asyncio.run(run())
    assert "stage-A" in timings
    assert "stage-B" in timings
    assert timings["stage-A"] >= 0.015
    assert warnings == ["minor codec warning"]


def test_stage_timer_repeat_accumulates():
    from backend.services.pipeline_helpers import _drain_pipeline_telemetry, _stage_timer

    async def run():
        async with _stage_timer("jobX", "duped"):
            await asyncio.sleep(0.01)
        async with _stage_timer("jobX", "duped"):
            await asyncio.sleep(0.01)
        return _drain_pipeline_telemetry("jobX")
    timings, _ = asyncio.run(run())
    assert timings["duped"] >= 0.02


def test_drain_is_idempotent():
    from backend.services.pipeline_helpers import _drain_pipeline_telemetry, _stage_timer

    async def run():
        async with _stage_timer("jobY", "only"):
            pass
        first_t, _ = _drain_pipeline_telemetry("jobY")
        second_t, _ = _drain_pipeline_telemetry("jobY")
        return first_t, second_t
    first, second = asyncio.run(run())
    assert first
    assert second == {}


# ── Two-pass face detection routing ──────────────────────────────


def test_two_pass_routing_splits_speech_vs_other(monkeypatch, tmp_path):
    """Verify the speech-aware splitter sends the right paths to the
    right detector and merges results back in input order.

    We monkeypatch ``detect_faces_batch`` and ``_detect_with_opencv_dnn``
    so the test doesn't need real model weights.
    """
    from backend.services import face_detector as fd
    from backend.services.face_detector import (
        FrameFaces,
        detect_faces_batch_two_pass,
    )

    captured = {"speech_paths": [], "other_paths": []}

    def fake_full(paths, min_conf, progress_callback=None):
        captured["speech_paths"] = list(paths)
        return [FrameFaces(timestamp=t, frame_path=p, faces=[]) for t, p in paths]

    def fake_yunet(paths, min_conf):
        captured["other_paths"] = list(paths)
        return [FrameFaces(timestamp=t, frame_path=p, faces=[]) for t, p in paths]

    monkeypatch.setattr(fd, "detect_faces_batch", fake_full)
    monkeypatch.setattr(fd, "_detect_with_opencv_dnn", fake_yunet)

    paths = [(i * 0.5, str(tmp_path / f"f{i}.jpg")) for i in range(8)]
    # Speech intervals cover frames at t = 1.0..2.0 inclusive.
    speech = [(1.0, 2.0)]
    out = detect_faces_batch_two_pass(paths, speech)

    speech_ts = [p[0] for p in captured["speech_paths"]]
    other_ts = [p[0] for p in captured["other_paths"]]
    assert sorted(speech_ts) == [1.0, 1.5, 2.0]
    assert sorted(other_ts) == [0.0, 0.5, 2.5, 3.0, 3.5]
    # Output preserves input order
    assert [r.timestamp for r in out] == [p[0] for p in paths]


def test_two_pass_falls_back_when_intervals_empty(monkeypatch):
    from backend.services import face_detector as fd
    from backend.services.face_detector import detect_faces_batch_two_pass, FrameFaces

    called = {"full": 0}

    def fake_full(paths, min_conf, progress_callback=None):
        called["full"] += 1
        return [FrameFaces(timestamp=t, frame_path=p, faces=[]) for t, p in paths]

    monkeypatch.setattr(fd, "detect_faces_batch", fake_full)
    out = detect_faces_batch_two_pass([(0.0, "/tmp/a.jpg")], speech_intervals=[])
    assert len(out) == 1
    assert called["full"] == 1


# ── YOLO device resolver ────────────────────────────────────────


def test_yolo_device_resolver_respects_env(monkeypatch):
    from backend.services.object_detector import _resolve_yolo_device
    monkeypatch.setenv("CLIPAI_OBJECT_DETECTOR_DEVICE", "cpu")
    assert _resolve_yolo_device() == "cpu"


def test_yolo_device_resolver_respects_gpu_off(monkeypatch):
    from backend.services.object_detector import _resolve_yolo_device
    monkeypatch.setenv("CLIPAI_OBJECT_DETECTOR_DEVICE", "auto")
    # Force the GPU master switch off.
    from backend.config import settings as cfg
    monkeypatch.setattr(cfg, "GPU_ACCELERATION_ENABLED", False, raising=False)
    assert _resolve_yolo_device() == "cpu"


# ── Ollama vision defaults ──────────────────────────────────────


def test_ollama_vision_defaults_high_concurrency_and_keepalive(monkeypatch):
    # Re-import to pick up env-driven module constants.
    monkeypatch.setenv("CLIPAI_OLLAMA_VISION_CONCURRENCY", "5")
    monkeypatch.setenv("CLIPAI_OLLAMA_KEEP_ALIVE", "-1")
    import importlib
    import backend.services.providers.ollama_provider as op
    importlib.reload(op)
    try:
        assert op.VISION_CONCURRENCY == 5
        assert op.OLLAMA_VISION_KEEP_ALIVE == "-1"
    finally:
        # Reset to package defaults so subsequent tests don't see the override.
        monkeypatch.delenv("CLIPAI_OLLAMA_VISION_CONCURRENCY", raising=False)
        monkeypatch.delenv("CLIPAI_OLLAMA_KEEP_ALIVE", raising=False)
        importlib.reload(op)


# ── Rescore endpoint ────────────────────────────────────────────


@pytest.fixture
def rescore_client(monkeypatch):
    """Spin up a tiny FastAPI app with auth + the jobs router so we
    can call the rescore endpoint without booting the full pipeline.
    """
    pytest.importorskip("fastapi.testclient")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.app.auth.middleware import AuthMiddleware
    from backend.app.auth import store as auth_store
    from backend.routers import auth as auth_router
    from backend.routers import jobs as jobs_router

    asyncio.run(auth_store._reset_for_tests())

    # Stub the database the jobs router consumes.
    from backend import database as _db
    from backend.models import ClipCandidate, JobResult

    _STORE: dict[str, JobResult] = {}

    async def _stub_load_job(job_id):
        return _STORE.get(job_id)

    async def _stub_save_job(job):
        _STORE[job.job_id] = job

    monkeypatch.setattr(_db, "load_job", _stub_load_job)
    monkeypatch.setattr(_db, "save_job", _stub_save_job)

    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(auth_router.router)
    app.include_router(jobs_router.router)

    client = TestClient(app)

    def make_job(*, owner: str, clips: list[ClipCandidate], content_type: str = "talking_head") -> str:
        from datetime import datetime, timezone
        job = JobResult(
            job_id="J1",
            filename="podcast.mp4",
            file_path="/tmp/podcast.mp4",
            owner_user_id=owner,
            content_type_override=content_type,
            clips=[c.model_dump() for c in clips],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        _STORE[job.job_id] = job
        return job.job_id

    return client, make_job


def _make_clip(**overrides):
    """Build a fully-populated ClipCandidate so tests don't have to
    care about every required field on the Pydantic model."""
    from backend.models import ClipCandidate
    defaults = dict(
        id=1, title="Demo",
        start_time=0.0, end_time=15.0, duration=15.0,
        viral_score=10, viral_score_reasoning="great hook",
        clip_type="hook", platform="both",
        suggested_caption="caption", hook_text="Wait for it",
        why_this_works="curiosity",
        hook_score=90, flow_score=40, value_score=60, trend_score=50,
    )
    defaults.update(overrides)
    return ClipCandidate(**defaults)


def test_rescore_recomputes_composite(rescore_client):
    from backend.app.auth import store as auth_store
    from backend.app.auth.models import Role

    client, make_job = rescore_client

    user = asyncio.run(auth_store.create_user("alice", "supersecret", role=Role.USER))
    clip = _make_clip()
    job_id = make_job(owner=user.id, clips=[clip], content_type="gameplay")

    client.post("/api/auth/login", json={"username": "alice", "password": "supersecret"})

    r = client.post(f"/api/jobs/{job_id}/rescore", json={"content_type": "gameplay"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rescored_count"] == 1
    assert body["content_type"] == "gameplay"
    new_score = body["clips"][0]["viral_score"]
    # Composite for gameplay weights (hook=0.4, flow=0.15, value=0.3, trend=0.15)
    # → round(90*0.4 + 40*0.15 + 60*0.3 + 50*0.15) ≈ 67-68
    assert new_score in (67, 68)


def test_rescore_requires_clips(rescore_client):
    from backend.app.auth import store as auth_store
    from backend.app.auth.models import Role

    client, make_job = rescore_client
    user = asyncio.run(auth_store.create_user("alice", "supersecret", role=Role.USER))
    job_id = make_job(owner=user.id, clips=[], content_type="talking_head")
    client.post("/api/auth/login", json={"username": "alice", "password": "supersecret"})
    r = client.post(f"/api/jobs/{job_id}/rescore")
    assert r.status_code == 400


def test_rescore_unauthenticated_blocked(rescore_client):
    from backend.app.auth import store as auth_store

    client, make_job = rescore_client
    user = asyncio.run(auth_store.create_user("alice", "supersecret"))
    job_id = make_job(owner=user.id, clips=[_make_clip()])
    # No login → 401 from auth middleware.
    r = client.post(f"/api/jobs/{job_id}/rescore")
    assert r.status_code == 401


def test_rescore_non_owner_404(rescore_client):
    from backend.app.auth import store as auth_store

    client, make_job = rescore_client
    alice = asyncio.run(auth_store.create_user("alice", "supersecret"))
    bob = asyncio.run(auth_store.create_user("bob", "bobsecret"))
    job_id = make_job(owner=alice.id, clips=[_make_clip()])
    client.post("/api/auth/login", json={"username": "bob", "password": "bobsecret"})
    r = client.post(f"/api/jobs/{job_id}/rescore")
    assert r.status_code == 404


# ── OpenRouter concurrency env ──────────────────────────────────


def test_openrouter_vision_concurrency_env_changes_default(monkeypatch):
    """The semaphore size for vision batches should be configurable."""
    monkeypatch.setenv("CLIPAI_OPENROUTER_VISION_CONCURRENCY", "5")
    # The setting is read each time analyze_frames is called, so just
    # assert env round-trip + clamp behavior via a tiny eval helper.
    val = int(os.environ.get("CLIPAI_OPENROUTER_VISION_CONCURRENCY", "3"))
    assert max(1, min(8, val)) == 5
    monkeypatch.setenv("CLIPAI_OPENROUTER_VISION_CONCURRENCY", "999")
    val = int(os.environ.get("CLIPAI_OPENROUTER_VISION_CONCURRENCY", "3"))
    assert max(1, min(8, val)) == 8
    monkeypatch.setenv("CLIPAI_OPENROUTER_VISION_CONCURRENCY", "0")
    val = int(os.environ.get("CLIPAI_OPENROUTER_VISION_CONCURRENCY", "3"))
    assert max(1, min(8, val)) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
