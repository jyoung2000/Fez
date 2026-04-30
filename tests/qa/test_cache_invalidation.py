"""Task A: cache hygiene tests.

Locks down the contract that protects against the silent stale-cache
regression. Every patch on Tasks 0/1/2/C had no effect on the
deployed homelab because the bench cache from a previous deploy was
served verbatim; this suite catches the three classes of failure
that allowed that:

1. Cache version doesn't match → cache invalid.
2. Watch list is missing modules → mtime check is a no-op.
3. Watch list has rotted (every path missing) → invalidation
   collapses silently.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pytest

import backend.scripts.compare_autoflip_vs_clipai as bench


@pytest.fixture(autouse=True)
def _reset_warned(monkeypatch):
    """Each test gets a clean dedup set so missing-module warnings fire."""
    monkeypatch.setattr(bench, "_WARNED_MISSING_MODULES", set())


def _write_cache(
    tmp_path: Path,
    *,
    version: int = bench.EXTRACTION_CACHE_VERSION,
    include_asd: bool = True,
) -> Path:
    """Write a populated cache directory with all required files."""
    clip_cache = tmp_path / "abcdef123"
    clip_cache.mkdir()
    files = list(bench._REQUIRED_CACHE_FILES)
    if include_asd:
        files.append("asd_scores.json")
    for name in files:
        if name == "cache_version.txt":
            (clip_cache / name).write_text(str(version))
        else:
            (clip_cache / name).write_text("[]")
    return clip_cache


# ── 1. Version bump invalidates old caches ────────────────────────────


def test_cache_version_bump_invalidates_old_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPAI_ASD_BACKEND", "light_asd")
    clip_cache = _write_cache(tmp_path, version=1)
    # Make the cache "newer than the world" so the only failure axis is version.
    future = time.time() + 3600
    os.utime(clip_cache / "cache_version.txt", (future, future))
    assert bench._cache_is_valid(clip_cache) is False


# ── 2. Watch list includes the new modules ────────────────────────────


def test_extractor_mtime_includes_new_modules(tmp_path, monkeypatch):
    """When one of the Task-0/1/2/C modules is newer than the rest,
    the helper must reflect that in its returned mtime."""
    repo_root = tmp_path / "repo"
    (repo_root / "backend" / "services").mkdir(parents=True)
    (repo_root / "backend" / "scripts").mkdir(parents=True)
    base = time.time() - 10000
    newer = time.time()

    for rel in bench._EXTRACTOR_MODULES_FOR_CACHE:
        p = repo_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# fixture\n")
        os.utime(p, (base, base))

    # Pick a Task-1 module added by Task A and make it the newest file.
    task_1_module = repo_root / "backend" / "services" / "editorial_planner.py"
    os.utime(task_1_module, (newer, newer))

    monkeypatch.setattr(bench, "_repo_root_for_cache", lambda: repo_root)
    result = bench._extractor_module_mtime()
    assert result == pytest.approx(newer, abs=2.0), (
        f"watch list ignored editorial_planner.py: returned {result}, "
        f"expected ~{newer}"
    )


def test_watch_list_has_no_rot(tmp_path):
    """Every path in _EXTRACTOR_MODULES_FOR_CACHE must exist on disk
    in the live repo. Catches typos before they ship."""
    repo_root = bench._repo_root_for_cache()
    missing = [
        rel for rel in bench._EXTRACTOR_MODULES_FOR_CACHE
        if not (repo_root / rel).is_file()
    ]
    assert missing == [], (
        f"watch list references non-existent modules: {missing}. "
        f"Add them to the codebase or remove them from "
        f"_EXTRACTOR_MODULES_FOR_CACHE."
    )


# ── 3. Resilience: missing modules log + skip ─────────────────────────


def test_missing_module_in_watch_list_logs_warning(
    tmp_path, monkeypatch, caplog,
):
    """A missing path must emit ONE warning then be skipped from the
    mtime computation. The function returns the max of the surviving
    files, never zero."""
    repo_root = tmp_path / "repo"
    (repo_root / "backend" / "services").mkdir(parents=True)
    (repo_root / "backend" / "scripts").mkdir(parents=True)
    real = repo_root / "backend" / "services" / "face_detector.py"
    real.write_text("# fixture\n")
    real_mtime = time.time() - 100
    os.utime(real, (real_mtime, real_mtime))

    monkeypatch.setattr(bench, "_repo_root_for_cache", lambda: repo_root)
    monkeypatch.setattr(
        bench, "_EXTRACTOR_MODULES_FOR_CACHE",
        ("backend/services/face_detector.py", "backend/services/__missing__.py"),
    )

    with caplog.at_level(logging.WARNING, logger=bench.logger.name):
        result = bench._extractor_module_mtime()

    assert result == pytest.approx(real_mtime, abs=1.0)
    warnings = [r for r in caplog.records if "__missing__.py" in r.message]
    assert len(warnings) == 1, (
        f"expected exactly one missing-module warning, got {len(warnings)}"
    )


def test_all_missing_modules_returns_zero(tmp_path, monkeypatch, caplog):
    """When the entire watch list points to non-existent files, the
    helper must return 0.0 AND emit an ERROR. Returning the current
    epoch instead would mark every cache invalid forever (subtle DOS)."""
    repo_root = tmp_path / "empty_repo"
    repo_root.mkdir()
    monkeypatch.setattr(bench, "_repo_root_for_cache", lambda: repo_root)
    monkeypatch.setattr(
        bench, "_EXTRACTOR_MODULES_FOR_CACHE",
        ("backend/services/__a__.py", "backend/services/__b__.py"),
    )
    with caplog.at_level(logging.ERROR, logger=bench.logger.name):
        result = bench._extractor_module_mtime()
    assert result == 0.0
    error_records = [r for r in caplog.records if r.levelname == "ERROR"]
    assert any("watch list" in r.message for r in error_records)


# ── 4. Cache validity gate ────────────────────────────────────────────


def test_cache_validity_off_when_one_extractor_newer(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPAI_ASD_BACKEND", "light_asd")
    clip_cache = _write_cache(tmp_path)
    # Force the bench's mtime helper to return a value strictly after
    # the cache's version-file mtime so the staleness branch fires.
    cache_mtime = (clip_cache / "cache_version.txt").stat().st_mtime
    monkeypatch.setattr(
        bench, "_extractor_module_mtime",
        lambda: cache_mtime + 60.0,
    )
    assert bench._cache_is_valid(clip_cache) is False


# ── 5. Stale-cache SSE payload ─────────────────────────────────────────


def test_cache_state_warning_payload_when_stale(tmp_path, monkeypatch):
    """Old cache (>24h) + fresh image build_info → status=stale_warning."""
    monkeypatch.setenv("CLIPAI_ASD_BACKEND", "light_asd")
    clip_cache = _write_cache(tmp_path)
    # Backdate the cache by 25 hours so the >24h gate trips.
    long_ago = time.time() - (25 * 3600)
    for f in clip_cache.iterdir():
        os.utime(f, (long_ago, long_ago))
    # Pretend the container was built 1 hour ago — newer than the cache.
    one_hour_ago_iso = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 3600),
    )
    monkeypatch.setattr(bench, "_read_build_info", lambda: one_hour_ago_iso)

    payload = bench.describe_cache_state(clip_cache)
    assert payload["status"] == "stale_warning"
    assert payload["warning"] is not None
    assert "force-reextract" in payload["warning"].lower()
    assert payload["cache_age_seconds"] >= 24 * 3600


def test_cache_state_warning_payload_when_fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPAI_ASD_BACKEND", "light_asd")
    clip_cache = _write_cache(tmp_path)
    monkeypatch.setattr(bench, "_read_build_info", lambda: None)
    payload = bench.describe_cache_state(clip_cache)
    assert payload["status"] == "ok"
    assert payload["warning"] is None
    assert payload["cache_age_seconds"] is not None
    assert payload["cache_age_seconds"] < 60  # just-written


def test_cache_state_miss_when_directory_absent(tmp_path):
    payload = bench.describe_cache_state(tmp_path / "nonexistent")
    assert payload["status"] == "miss"
    assert payload["exists"] is False
    assert payload["expected_version"] == bench.EXTRACTION_CACHE_VERSION


# ── 6. Cache version mismatch always stale ────────────────────────────


def test_cache_state_version_mismatch_is_stale(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPAI_ASD_BACKEND", "light_asd")
    clip_cache = _write_cache(tmp_path, version=1)
    monkeypatch.setattr(bench, "_read_build_info", lambda: None)
    payload = bench.describe_cache_state(clip_cache)
    assert payload["status"] == "stale_warning"
    assert payload["cache_version"] == 1
    assert payload["expected_version"] == bench.EXTRACTION_CACHE_VERSION
