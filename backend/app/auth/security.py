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
import os
import secrets
from typing import Optional

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
