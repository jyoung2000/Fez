"""Regression tests for the browser-preview transcode path.

Covers the three fixes from the preview-loading debug pass:

  1. Atomic rename — FFmpeg writes to ``<target>.tmp`` and only lands
     at ``<target>`` after a successful exit. A concurrent
     ``ensure_browser_preview_status`` caller during an in-flight
     encode must NOT see the half-written tmp file as the final
     preview. It must either return the completed target OR signal
     ``ready=False`` so the HTTP handler returns 503 + Retry-After.

  2. No leftover tmp file on ffmpeg failure.

  3. ``_probe`` cache — the second call with the same source + mtime
     does not re-fork ffprobe.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.services import browser_preview  # noqa: E402


def _reset_caches() -> None:
    browser_preview._probe_cache.clear()
    browser_preview._probe_cache_expiry.clear()


def test_ffmpeg_output_path_is_tmp_not_target():
    """The ffmpeg cmd writes through the scratch path — never the
    final target — so ``isfile(target)`` is false during encoding
    and the fast path can't return a half-written file."""
    probe = browser_preview._ProbeResult(
        video_codec="h264", audio_codec="aac",
        has_audio=True, width=1920, height=1080,
        fps=30.0, bitrate_kbps=5000,
    )
    cmd = browser_preview._build_ffmpeg_cmd(
        "/tmp/src.mp4", "/tmp/out.mp4", probe,
    )
    # Last positional arg is the output — must be the .tmp path.
    assert cmd[-1] == "/tmp/out.mp4.tmp", cmd
    # Target path must not appear anywhere else as an output either.
    assert "/tmp/out.mp4" not in cmd[:-1] or all(
        a != "/tmp/out.mp4" for a in cmd[:-1]
    )


def test_run_ffmpeg_renames_tmp_to_target_on_success():
    with tempfile.TemporaryDirectory() as td:
        target = os.path.join(td, "preview.mp4")
        tmp = browser_preview._tmp_path_for(target)

        # Simulate a successful ffmpeg that wrote the tmp file.
        def fake_run(cmd, **kwargs):
            with open(tmp, "wb") as f:
                f.write(b"fake mp4 bytes")
            result = mock.Mock()
            result.returncode = 0
            result.stderr = ""
            return result

        with mock.patch.object(browser_preview.subprocess, "run",
                               side_effect=fake_run):
            ok = browser_preview._run_ffmpeg(["ffmpeg", "fake"], target)

        assert ok is True
        assert os.path.isfile(target), "target should exist after rename"
        assert not os.path.exists(tmp), "tmp should be gone after rename"
        with open(target, "rb") as f:
            assert f.read() == b"fake mp4 bytes"


def test_run_ffmpeg_cleans_tmp_on_failure():
    with tempfile.TemporaryDirectory() as td:
        target = os.path.join(td, "preview.mp4")
        tmp = browser_preview._tmp_path_for(target)

        # Simulate ffmpeg writing a partial file then exiting nonzero.
        def fake_run(cmd, **kwargs):
            with open(tmp, "wb") as f:
                f.write(b"partial bytes")
            result = mock.Mock()
            result.returncode = 1
            result.stderr = "encoding failed"
            return result

        with mock.patch.object(browser_preview.subprocess, "run",
                               side_effect=fake_run):
            ok = browser_preview._run_ffmpeg(["ffmpeg", "fake"], target)

        assert ok is False
        assert not os.path.isfile(target), "target must NOT exist on failure"
        assert not os.path.exists(tmp), "tmp must be cleaned on failure"


def test_concurrent_reader_does_not_see_partial_file():
    """The core race this patch closes: during an in-flight encode,
    a concurrent HTTP handler calling ensure_browser_preview_status
    must not return the half-written tmp file as the final target.
    With the atomic-rename fix, the target simply doesn't exist until
    the rename completes, so the fast path correctly misses."""
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "source.mkv")
        with open(src, "wb") as f:
            f.write(b"fake source bytes")
        # Bump source mtime back a second so the test isn't dependent
        # on filesystem timestamp granularity.
        past = time.time() - 5
        os.utime(src, (past, past))

        target = browser_preview._preview_path_for(src)
        tmp = browser_preview._tmp_path_for(target)

        # Simulate a peer mid-encode: tmp file exists, target does not.
        with open(tmp, "wb") as f:
            f.write(b"half-written mp4")
        # And the peer holds the lock.
        lock_path = browser_preview._lock_path_for(target)
        assert browser_preview._acquire_lock(lock_path), \
            "test setup: lock should be free"

        try:
            # Stub the probe so we don't need a real mkv.
            fake_probe = browser_preview._ProbeResult(
                video_codec="h264", audio_codec="ac3",
                has_audio=True, width=1920, height=1080,
                fps=24.0, bitrate_kbps=8000,
            )
            _reset_caches()
            with mock.patch.object(browser_preview, "_probe",
                                   return_value=fake_probe):
                path, ready = browser_preview.ensure_browser_preview_status(
                    src, wait_for_peer=False,
                )

            # The function must NOT have returned the tmp file, and
            # must NOT have returned (target, True) because target
            # doesn't exist. It should signal "not ready" so the HTTP
            # handler emits 503 + Retry-After.
            assert path == src, f"expected src fallback, got {path}"
            assert ready is False, "must signal not-ready during peer encode"
            assert not os.path.isfile(target), \
                "target must not exist during peer encode"
        finally:
            browser_preview._release_lock(lock_path)
            if os.path.exists(tmp):
                os.remove(tmp)


def test_probe_cache_skips_subprocess_on_second_call():
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "x.mp4")
        with open(src, "wb") as f:
            f.write(b"bytes")

        fake_stdout = (
            '{"streams":[{"codec_type":"video","codec_name":"h264",'
            '"width":1920,"height":1080,"avg_frame_rate":"30/1",'
            '"bit_rate":"5000000"},'
            '{"codec_type":"audio","codec_name":"aac"}]}'
        )

        call_count = {"n": 0}

        def fake_run(cmd, **kwargs):
            call_count["n"] += 1
            result = mock.Mock()
            result.returncode = 0
            result.stdout = fake_stdout
            result.stderr = ""
            return result

        _reset_caches()
        with mock.patch.object(browser_preview.subprocess, "run",
                               side_effect=fake_run):
            r1 = browser_preview._probe(src)
            r2 = browser_preview._probe(src)
            r3 = browser_preview._probe(src)

        assert r1 is not None and r1.video_codec == "h264"
        assert r2 is r1, "cache should return the identical object"
        assert r3 is r1
        assert call_count["n"] == 1, (
            f"ffprobe should fork once, forked {call_count['n']} times"
        )


def test_probe_cache_invalidates_on_mtime_change():
    """If the source file is replaced, the cache key (which includes
    mtime) changes, so the next probe call correctly re-forks ffprobe."""
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "x.mp4")
        with open(src, "wb") as f:
            f.write(b"v1")
        os.utime(src, (1000.0, 1000.0))

        fake_stdout = (
            '{"streams":[{"codec_type":"video","codec_name":"h264",'
            '"width":1920,"height":1080,"avg_frame_rate":"30/1"}]}'
        )
        call_count = {"n": 0}

        def fake_run(cmd, **kwargs):
            call_count["n"] += 1
            result = mock.Mock()
            result.returncode = 0
            result.stdout = fake_stdout
            result.stderr = ""
            return result

        _reset_caches()
        with mock.patch.object(browser_preview.subprocess, "run",
                               side_effect=fake_run):
            browser_preview._probe(src)
            # "Replace" the source — bump mtime forward.
            os.utime(src, (2000.0, 2000.0))
            browser_preview._probe(src)

        assert call_count["n"] == 2, (
            "mtime change must invalidate cache; got "
            f"{call_count['n']} probe calls"
        )


if __name__ == "__main__":
    # Minimal runner so this file works without pytest installed.
    import inspect
    failed = 0
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                fn()
                print(f"PASS {name}")
                passed += 1
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
                failed += 1
            except Exception as e:
                print(f"ERROR {name}: {type(e).__name__}: {e}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
