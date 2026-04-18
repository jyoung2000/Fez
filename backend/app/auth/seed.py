"""Head-admin seeding.

On every startup, ensure the ``Jadmin`` account exists with the
hard-coded default password and admin role. The head-admin flag
prevents accidental deletion or demotion even by another admin. The
default password is only set on first creation — if the head admin
later changes their password, the seeder will NOT overwrite it.
"""

from __future__ import annotations

import logging

from backend.app.auth.models import Role
from backend.app.auth.store import (
    create_user,
    get_user_by_username,
    update_user,
)

logger = logging.getLogger(__name__)

HEAD_ADMIN_USERNAME = "Jadmin"
HEAD_ADMIN_PASSWORD = "Ilovesnoop5994!"


async def ensure_head_admin() -> None:
    """Create the head admin if missing; no-op otherwise."""
    existing = await get_user_by_username(HEAD_ADMIN_USERNAME)
    if existing is None:
        await create_user(
            HEAD_ADMIN_USERNAME,
            HEAD_ADMIN_PASSWORD,
            role=Role.ADMIN,
            head_admin=True,
        )
        logger.info("seeded head admin user %r", HEAD_ADMIN_USERNAME)
        return
    # Guarantee role + head_admin flag on every boot, in case the file
    # was manually edited.
    if existing.role != Role.ADMIN or not existing.head_admin or not existing.active:
        # update_user only accepts role / active; head_admin is set on
        # creation and cannot be mutated. If the flag is off on an
        # existing row, re-seed by direct update.
        if not existing.head_admin:
            from backend.app.auth.store import (
                USERS_PATH,
                _atomic_write_json,
                _read_json,
                _users_lock,
            )
            async with _users_lock:
                data = await _read_json(USERS_PATH, {"users": []})
                for u in data.get("users", []):
                    if u["username"].lower() == HEAD_ADMIN_USERNAME.lower():
                        u["head_admin"] = True
                        u["role"] = Role.ADMIN.value
                        u["active"] = True
                await _atomic_write_json(USERS_PATH, data)
        else:
            await update_user(existing.id, role=Role.ADMIN, active=True)
        logger.info("normalized head admin user %r", HEAD_ADMIN_USERNAME)
