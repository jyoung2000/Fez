"""Tests for Fernet token encryption and the log-redaction filter.

These tests are deliberately hermetic — no network, no filesystem, no
FastAPI app. The goal is to prove that:

1. Tokens round-trip through :func:`encrypt` / :func:`decrypt` with a
   stable key set in the environment.
2. A missing key triggers a one-shot warning and still produces usable
   (but ephemeral) encryption.
3. The log-redaction filter actually scrubs JSON, urlencoded, and
   Authorization-header token payloads from emitted log records.
"""
from __future__ import annotations

import logging
import os
import sys

import pytest

# Ensure project root is on sys.path so ``backend.*`` imports work when
# pytest is invoked from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from cryptography.fernet import Fernet

from backend.app.cloud import crypto
from backend.app.cloud.logging_filter import TokenRedactionFilter


@pytest.fixture(autouse=True)
def _reset_crypto():
    crypto.reset_for_tests()
    yield
    crypto.reset_for_tests()


def test_round_trip_with_env_key(monkeypatch):
    key = Fernet.generate_key().decode()
    monkeypatch.setenv(crypto.ENV_VAR, key)
    crypto.reset_for_tests()

    blob = crypto.encrypt("super-secret-token")
    assert blob is not None
    assert b"super-secret-token" not in blob

    assert crypto.decrypt(blob) == "super-secret-token"


def test_none_passes_through(monkeypatch):
    monkeypatch.setenv(crypto.ENV_VAR, Fernet.generate_key().decode())
    crypto.reset_for_tests()

    assert crypto.encrypt(None) is None
    assert crypto.decrypt(None) is None


def test_invalid_env_key_raises(monkeypatch):
    monkeypatch.setenv(crypto.ENV_VAR, "not-a-valid-fernet-key")
    crypto.reset_for_tests()
    with pytest.raises(crypto.TokenCryptoError):
        crypto.encrypt("hello")


def test_missing_key_generates_ephemeral(monkeypatch, caplog):
    monkeypatch.delenv(crypto.ENV_VAR, raising=False)
    crypto.reset_for_tests()
    with caplog.at_level(logging.WARNING, logger="backend.app.cloud.crypto"):
        blob = crypto.encrypt("value")
    assert blob is not None
    assert crypto.decrypt(blob) == "value"
    # The warning is only emitted once per process.
    assert any(crypto.ENV_VAR in record.message for record in caplog.records)


def test_wrong_key_fails_to_decrypt(monkeypatch):
    key_a = Fernet.generate_key().decode()
    monkeypatch.setenv(crypto.ENV_VAR, key_a)
    crypto.reset_for_tests()
    blob = crypto.encrypt("secret")

    key_b = Fernet.generate_key().decode()
    monkeypatch.setenv(crypto.ENV_VAR, key_b)
    crypto.reset_for_tests()

    with pytest.raises(crypto.TokenCryptoError):
        crypto.decrypt(blob)


def test_redaction_filter_scrubs_json_tokens():
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='{"access_token": "abc123", "user": "alice"}',
        args=(),
        exc_info=None,
    )
    TokenRedactionFilter().filter(record)
    assert "abc123" not in record.msg
    assert "<redacted>" in record.msg
    assert "alice" in record.msg


def test_redaction_filter_scrubs_urlencoded_tokens():
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="body=refresh_token=ya29.xyz&client_id=abc",
        args=(),
        exc_info=None,
    )
    TokenRedactionFilter().filter(record)
    assert "ya29.xyz" not in record.msg
    assert "client_id=abc" in record.msg


def test_redaction_filter_scrubs_bearer_header():
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='Authorization: Bearer my-real-token-value-12345',
        args=(),
        exc_info=None,
    )
    TokenRedactionFilter().filter(record)
    assert "my-real-token-value-12345" not in record.msg
