"""Tests for the signed media URL helpers in ``backend.app.auth.security``.

Covers:
  * Round-trip: a freshly signed URL verifies.
  * Mismatch in user_id / job_id / path fails.
  * Expired signature fails.
  * Tampered signature fails.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

# Point the auth store at a throwaway dir before importing anything that
# resolves AUTH_DIR at import time (matches ``tests/test_auth.py``).
_TMP = Path(tempfile.mkdtemp(prefix="clipai_media_signing_test_"))
os.environ["HOME"] = str(_TMP)

from backend.app.auth import security  # noqa: E402


def _fresh_key():
    """Reset the module-level signing-key cache so each test starts
    from a deterministic on-disk state."""
    security._reset_signing_key_cache_for_tests()


def test_sign_and_verify_roundtrip():
    _fresh_key()
    expiry, sig = security.sign_media_url("user-a", "job-1", "video.mp4")
    assert isinstance(expiry, int) and expiry > int(time.time())
    assert isinstance(sig, str) and len(sig) == 64  # sha256 hex digest
    assert security.verify_media_signature(
        "user-a", "job-1", "video.mp4", expiry, sig,
    )


def test_verify_rejects_wrong_user_id():
    _fresh_key()
    expiry, sig = security.sign_media_url("user-a", "job-1", "video.mp4")
    assert not security.verify_media_signature(
        "user-b", "job-1", "video.mp4", expiry, sig,
    )


def test_verify_rejects_wrong_job_id():
    _fresh_key()
    expiry, sig = security.sign_media_url("user-a", "job-1", "video.mp4")
    assert not security.verify_media_signature(
        "user-a", "job-2", "video.mp4", expiry, sig,
    )


def test_verify_rejects_wrong_path():
    _fresh_key()
    expiry, sig = security.sign_media_url("user-a", "job-1", "video.mp4")
    assert not security.verify_media_signature(
        "user-a", "job-1", "other.mp4", expiry, sig,
    )


def test_verify_rejects_expired_signature():
    _fresh_key()
    # Back-date the expiry so it's already in the past.
    past_time = time.time() - 1_000_000
    expiry, sig = security.sign_media_url(
        "user-a", "job-1", "video.mp4",
        ttl=10,
        now=past_time,
    )
    assert expiry < int(time.time())
    assert not security.verify_media_signature(
        "user-a", "job-1", "video.mp4", expiry, sig,
    )


def test_verify_rejects_tampered_signature():
    _fresh_key()
    expiry, sig = security.sign_media_url("user-a", "job-1", "video.mp4")
    # Flip one hex nibble.
    flipped = ("0" if sig[0] != "0" else "1") + sig[1:]
    assert flipped != sig
    assert not security.verify_media_signature(
        "user-a", "job-1", "video.mp4", expiry, flipped,
    )


def test_verify_rejects_malformed_inputs():
    _fresh_key()
    # Empty sig / non-int expiry are explicit no-ops.
    assert not security.verify_media_signature("u", "j", "p", 9999999999, "")
    assert not security.verify_media_signature("u", "j", "p", "not-int", "abc")  # type: ignore[arg-type]


def test_signatures_differ_per_key_restart(tmp_path, monkeypatch):
    """A wiped-and-regenerated key must not verify old signatures."""
    _fresh_key()
    # Force the store to use a throwaway directory so we can wipe it.
    from backend.app.auth import store as auth_store
    monkeypatch.setattr(auth_store, "AUTH_DIR", str(tmp_path))
    _fresh_key()
    expiry, sig = security.sign_media_url("u", "j", "p")
    key_path = os.path.join(str(tmp_path), "signing_key")
    assert os.path.isfile(key_path)
    os.unlink(key_path)
    _fresh_key()
    # New key → old signature fails.
    assert not security.verify_media_signature("u", "j", "p", expiry, sig)
