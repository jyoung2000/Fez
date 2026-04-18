"""User identity helper for cloud-account scoping.

ClipAI on Unraid is single-user, so there is no real user model. Every
cloud account is keyed on the literal string ``"local"``. Concentrating
that assumption in one function means the day we add real authentication
only this file changes.
"""

from __future__ import annotations

from fastapi import Request

LOCAL_USER_ID = "local"


def current_user_id(request: Request | None = None) -> str:  # noqa: ARG001 - kept for future auth
    """Return the current user's id.

    *request* is accepted (and ignored) so call sites can already depend
    on it. When real auth lands, this function can inspect cookies /
    headers without any call-site changes.
    """
    return LOCAL_USER_ID
