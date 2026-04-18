"""Fernet-based encryption wrapper for OAuth tokens at rest.

Tokens are never logged — the logging filter in
:mod:`backend.app.cloud.logging_filter` scrubs anything that slips through,
and this module deliberately keeps ``__repr__`` of its inputs out of any
exception messages.

The key is pulled from the ``CLIPAI_TOKEN_ENC_KEY`` environment variable.
If it is missing, we generate one on first use and log a loud warning
telling the operator to persist it in the Unraid template — rotating the
key invalidates all stored cloud sessions, so silent rotation would drop
every user's cloud connections the next time the container restarts.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Final

try:
    from cryptography.fernet import Fernet, InvalidToken
except ImportError as exc:  # pragma: no cover - dependency is declared in requirements.txt
    raise ImportError(
        "The 'cryptography' package is required for cloud storage token "
        "encryption. Install it via `pip install cryptography` or ensure "
        "backend/requirements.txt is honored in the container build."
    ) from exc


logger = logging.getLogger(__name__)


ENV_VAR: Final[str] = "CLIPAI_TOKEN_ENC_KEY"

_warning_emitted = False
_fernet: Fernet | None = None
_lock = threading.Lock()


class TokenCryptoError(Exception):
    """Raised when a token cannot be encrypted or decrypted."""


def _load_or_generate_key() -> bytes:
    """Return the Fernet key bytes, generating a new one if necessary."""
    global _warning_emitted
    raw = os.environ.get(ENV_VAR, "").strip()
    if raw:
        try:
            # Validate shape — Fernet raises if this isn't 32 url-safe bytes.
            Fernet(raw.encode("ascii"))
            return raw.encode("ascii")
        except Exception as exc:
            raise TokenCryptoError(
                f"{ENV_VAR} is set but not a valid Fernet key (expected 32 "
                "url-safe base64-encoded bytes)."
            ) from exc

    generated = Fernet.generate_key()
    if not _warning_emitted:
        logger.warning(
            "\n%s\n"
            "%s is not set — generated an ephemeral token-encryption key.\n"
            "Any cloud-storage sessions you authorize will be lost the next "
            "time this container restarts. Set %s in your Unraid template "
            "(or .env) to a persistent Fernet key to keep them.\n"
            "Generate one with: python -c "
            "'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'\n"
            "%s",
            "=" * 72,
            ENV_VAR,
            ENV_VAR,
            "=" * 72,
        )
        _warning_emitted = True
    return generated


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        with _lock:
            if _fernet is None:
                _fernet = Fernet(_load_or_generate_key())
    return _fernet


def reset_for_tests() -> None:
    """Drop the cached Fernet instance so tests can swap the env var."""
    global _fernet, _warning_emitted
    with _lock:
        _fernet = None
        _warning_emitted = False


def encrypt(plaintext: str | None) -> bytes | None:
    """Encrypt *plaintext*. ``None`` passes through unchanged."""
    if plaintext is None:
        return None
    if not isinstance(plaintext, str):
        raise TokenCryptoError("encrypt() expects a str")
    try:
        return _get_fernet().encrypt(plaintext.encode("utf-8"))
    except Exception as exc:  # pragma: no cover - Fernet.encrypt is deterministic after construction
        raise TokenCryptoError("token encryption failed") from exc


def decrypt(ciphertext: bytes | None) -> str | None:
    """Decrypt *ciphertext*. ``None`` passes through unchanged."""
    if ciphertext is None:
        return None
    if isinstance(ciphertext, str):
        ciphertext = ciphertext.encode("ascii")
    try:
        return _get_fernet().decrypt(ciphertext).decode("utf-8")
    except InvalidToken as exc:
        # Do not echo the ciphertext — it may hint at the wrapped payload.
        raise TokenCryptoError(
            "token decryption failed: wrong key or corrupted payload"
        ) from exc
