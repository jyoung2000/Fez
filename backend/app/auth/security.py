"""Password hashing + session fingerprinting.

Uses stdlib (``hashlib``, ``hmac``, ``secrets``) to avoid adding a
dependency on bcrypt / argon2 / passlib. PBKDF2-HMAC-SHA256 with
200 000 iterations is OWASP-recommended and runs in ~120 ms on a
laptop — acceptable for the few logins a self-hosted instance sees.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_PBKDF2_ITERATIONS = 200_000
_SALT_BYTES = 16
_HASH_BYTES = 32
_TOKEN_BYTES = 32

# Password requirements
MIN_PASSWORD_LEN = 6


def new_salt() -> str:
    """Return a URL-safe base64 salt."""
    return base64.urlsafe_b64encode(secrets.token_bytes(_SALT_BYTES)).decode("ascii")


def hash_password(password: str, salt: str) -> str:
    """Return a PBKDF2-HMAC-SHA256 hash encoded as URL-safe base64.

    The salt must be URL-safe base64. The caller owns both the salt
    and the returned hash.
    """
    if not password or len(password) < MIN_PASSWORD_LEN:
        raise ValueError(f"password must be at least {MIN_PASSWORD_LEN} characters")
    raw = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        base64.urlsafe_b64decode(salt),
        _PBKDF2_ITERATIONS,
        dklen=_HASH_BYTES,
    )
    return base64.urlsafe_b64encode(raw).decode("ascii")


def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    """Constant-time password check."""
    if not password:
        return False
    try:
        candidate = hash_password(password, salt)
    except Exception:
        return False
    return hmac.compare_digest(candidate, expected_hash)


def new_token() -> str:
    """Return a cryptographically secure opaque session token."""
    return secrets.token_urlsafe(_TOKEN_BYTES)


def normalize_ip(ip_raw: Optional[str]) -> str:
    """Collapse an IP header into a canonical form for fingerprinting.

    Truncates aggressively so users on rotating IPs (residential ISPs,
    mobile carriers, dual-stack v4/v6 hops, VPN servers) don't get
    signed out the moment their network changes:

      * IPv4 → first three octets only (``/24`` block). House → cafe
        across the same provider keeps the session alive when both
        share a CGNAT / ASN-level prefix.
      * IPv6 → first three groups only (``/48`` block). Carriers
        commonly re-issue the lower 80 bits.

    An IP we can't parse returns an empty string — the fingerprint
    then falls through to the user-agent alone, which is a deliberate
    relaxation so brand-new networks still keep the user signed in
    (the cookie itself is the credential; the fingerprint is a
    defense-in-depth signal).
    """
    if not ip_raw:
        return ""
    # X-Forwarded-For can carry multiple addresses; take the leftmost
    # (the original client) and strip whitespace.
    ip = (ip_raw.split(",") or [""])[0].strip()
    if ":" in ip and "." not in ip:
        # IPv6: keep first 3 groups (~/48 prefix)
        parts = ip.split(":")
        ip = ":".join(parts[:3]) + "::"
    elif "." in ip:
        # IPv4: keep first 3 octets (/24 block)
        parts = ip.split(".")
        if len(parts) == 4:
            ip = ".".join(parts[:3]) + ".0"
    return ip


def compute_fingerprint(ip: str, user_agent: str) -> str:
    """Return a short hex digest binding the session to IP + UA.

    Stable across requests from the same browser AND the same /24
    (IPv4) or /48 (IPv6) network, so a tab reload after the ISP
    re-leased the dynamic IP doesn't invalidate the session. Set the
    env var ``CLIPAI_BIND_SESSION_TO_IP=false`` to disable the IP
    component entirely (recommended for installs behind a load
    balancer that rewrites the client IP per request).
    """
    bind_ip = os.environ.get("CLIPAI_BIND_SESSION_TO_IP", "true").lower() not in (
        "0", "false", "no", "off",
    )
    norm_ip = normalize_ip(ip) if bind_ip else ""
    norm_ua = (user_agent or "").strip()
    h = hashlib.sha256()
    h.update(norm_ip.encode("utf-8"))
    h.update(b"|")
    h.update(norm_ua.encode("utf-8"))
    return h.hexdigest()[:32]


# ── Server signing secret + signed media URLs ───────────────────
#
# We bind media URLs for ``<video>`` / ``<img>`` elements to a
# short-lived HMAC signature so the browser doesn't have to send
# cookies with range requests. That uncouples streaming from the
# session cookie, which is important because:
#
#   * HTML5 ``<video>`` fires many range requests from different
#     request contexts; any 401 aborts playback.
#   * Behind a reverse proxy, a single mis-rewritten cookie header
#     would otherwise kill the whole preview session.
#
# The signature covers ``(user_id, job_id, path, expiry)`` under a
# persistent server secret stored at ``/data/auth/signing_key``. The
# secret never leaves the server; only the resulting hex digest goes
# into the URL.

MEDIA_URL_TTL_SECONDS = 3600  # 1h — long enough for a full-length preview

_SIGNING_KEY_FILENAME = "signing_key"
_signing_key_cache: Optional[bytes] = None


def _signing_key_path() -> str:
    # Deferred import so this module doesn't pull in aiofiles /
    # store.py at import time for tests that stub out the store.
    from backend.app.auth import store as _store
    return os.path.join(_store.AUTH_DIR, _SIGNING_KEY_FILENAME)


def _load_or_create_signing_key() -> bytes:
    """Return the persistent HMAC key, creating it on first call.

    Stored as 32 raw bytes (not base64) alongside ``sessions.json``.
    Re-created only if the file is missing or unreadable; existing
    signatures survive restarts.
    """
    global _signing_key_cache
    if _signing_key_cache is not None:
        return _signing_key_cache
    path = _signing_key_path()
    try:
        with open(path, "rb") as f:
            data = f.read()
        if len(data) >= 32:
            _signing_key_cache = data[:32]
            return _signing_key_cache
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning("failed to read signing key %s: %s", path, e)
    # Generate a fresh key and persist it with restrictive perms.
    key = secrets.token_bytes(32)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, key)
        finally:
            os.close(fd)
    except Exception as e:
        logger.warning("failed to persist signing key %s: %s", path, e)
    _signing_key_cache = key
    return key


def _reset_signing_key_cache_for_tests() -> None:
    """Test-only hook — drop the in-process key cache."""
    global _signing_key_cache
    _signing_key_cache = None


def _media_sign_payload(user_id: str, job_id: str, path: str, expiry: int) -> bytes:
    # Newline-separated fields so no single field can smuggle the
    # delimiter into another field's slot.
    return (
        "media-v1\n"
        f"{user_id}\n{job_id}\n{path}\n{expiry}"
    ).encode("utf-8")


def sign_media_url(
    user_id: str,
    job_id: str,
    path: str,
    *,
    ttl: int = MEDIA_URL_TTL_SECONDS,
    now: Optional[float] = None,
) -> Tuple[int, str]:
    """Return ``(expiry_epoch, signature_hex)`` for a media URL.

    Callers embed ``?exp={expiry}&sig={signature}`` in the URL. The
    signature binds all four inputs; tampering with any of them (or
    presenting the URL after ``expiry``) fails verification.
    """
    key = _load_or_create_signing_key()
    ts = int((now if now is not None else time.time())) + int(ttl)
    payload = _media_sign_payload(user_id, job_id, path, ts)
    sig = hmac.new(key, payload, hashlib.sha256).hexdigest()
    return ts, sig


def verify_media_signature(
    user_id: str,
    job_id: str,
    path: str,
    expiry: int,
    signature: str,
    *,
    now: Optional[float] = None,
) -> bool:
    """Constant-time signature + expiry check.

    Returns ``False`` for expired, tampered, or malformed inputs —
    never raises, so callers can pass raw user-supplied query values.
    """
    if not signature or not isinstance(expiry, int):
        return False
    current = now if now is not None else time.time()
    if expiry < current:
        return False
    key = _load_or_create_signing_key()
    payload = _media_sign_payload(user_id, job_id, path, expiry)
    expected = hmac.new(key, payload, hashlib.sha256).hexdigest()
    try:
        return hmac.compare_digest(expected, signature)
    except Exception:
        return False
