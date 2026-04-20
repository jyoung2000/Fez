"""Auth dataclasses: Role, User, Session."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Optional


class Role(str, Enum):
    """Permission roles.

    Only two levels by design: ``ADMIN`` can manage users; ``USER`` has
    access to their own data only.
    """

    ADMIN = "admin"
    USER = "user"


@dataclass
class User:
    """Persisted user record.

    The password is stored as ``pbkdf2_sha256`` with a per-user random
    salt. ``head_admin`` marks the seeded account (``Jadmin``) so it
    can never be deleted or demoted even by another admin.
    """

    id: str
    username: str
    password_hash: str
    salt: str
    role: Role
    created_at: str
    head_admin: bool = False
    active: bool = True

    def to_public(self) -> dict:
        """Return a dict safe to send to the frontend (no hash/salt)."""
        return {
            "id": self.id,
            "username": self.username,
            "role": self.role.value,
            "head_admin": self.head_admin,
            "active": self.active,
            "created_at": self.created_at,
        }

    def to_storage(self) -> dict:
        d = asdict(self)
        d["role"] = self.role.value
        return d

    @classmethod
    def from_storage(cls, data: dict) -> "User":
        return cls(
            id=data["id"],
            username=data["username"],
            password_hash=data["password_hash"],
            salt=data["salt"],
            role=Role(data.get("role", "user")),
            created_at=data["created_at"],
            head_admin=bool(data.get("head_admin", False)),
            active=bool(data.get("active", True)),
        )


@dataclass
class Session:
    """Persisted session record.

    ``fingerprint`` is a sha256 of ``ip_hint + user_agent_hint``.
    Requests arriving with a different fingerprint are rejected with
    401 — the frontend then redirects to the login page, so a new IP
    or browser always forces a re-login.

    Sessions expire after ``expires_at``; logout deletes the record.
    """

    token: str
    user_id: str
    fingerprint: str
    ip: str
    user_agent: str
    created_at: str
    last_seen: str
    expires_at: str
    # ``remember`` controls cookie persistence on the BROWSER side:
    #   * True  → cookie has Max-Age, survives browser restarts,
    #             user stays signed in for the rolling TTL window.
    #   * False → session-scoped cookie, dropped when the browser
    #             quits — every new launch requires sign-in.
    # Defaults to True so older session JSONs without this field
    # behave the way they always did.
    remember: bool = True
    # ``fp_v2`` marks sessions whose stored ``fingerprint`` is the
    # UA-only V2 digest. Legacy sessions default to False and are
    # migrated to V2 on their first fingerprint check; after that the
    # flag is True and the middleware treats a mismatch as a real
    # rejection rather than a stale-format quirk.
    fp_v2: bool = False

    def to_storage(self) -> dict:
        return asdict(self)

    @classmethod
    def from_storage(cls, data: dict) -> "Session":
        # Drop unknown fields defensively so adding new ones doesn't
        # crash on existing on-disk records.
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
