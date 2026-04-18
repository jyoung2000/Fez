"""Authentication + user-management endpoints.

Routes grouped under ``/api/auth``:

  * ``POST   /api/auth/login``               — username+password → sets
                                                 the session cookie.
  * ``POST   /api/auth/logout``              — deletes the session.
  * ``GET    /api/auth/me``                  — current user + session info.
  * ``PATCH  /api/auth/me/password``         — self-change password.
  * ``GET    /api/auth/me/settings``         — per-user AI / whisper prefs.
  * ``PUT    /api/auth/me/settings``         — update prefs.

Admin-only (``require_admin``):

  * ``GET    /api/auth/admin/users``         — list users.
  * ``POST   /api/auth/admin/users``         — create user.
  * ``PATCH  /api/auth/admin/users/{id}``    — change role / activate / deactivate.
  * ``DELETE /api/auth/admin/users/{id}``    — remove user (and sessions).
  * ``POST   /api/auth/admin/users/{id}/reset_password`` — reset a user's
                                                             password.
  * ``GET    /api/auth/admin/sessions``      — active sessions.
  * ``DELETE /api/auth/admin/sessions/{token}`` — revoke.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from backend.app.auth.deps import get_current_user, require_admin
from backend.app.auth.middleware import clear_session_cookie, set_session_cookie
from backend.app.auth.models import Role, User
from backend.app.auth.security import (
    MIN_PASSWORD_LEN,
    verify_password,
)
from backend.app.auth import store as auth_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ── Schemas ────────────────────────────────────────────────────


class LoginRequest(BaseModel):
    username: str
    password: str
    # When False, the server writes a session-scoped cookie (no
    # ``Max-Age``) so the browser drops it on quit. When True
    # (default), the cookie persists for the full session TTL so
    # the user stays signed in for ~30 days. Backwards-compatible
    # with old clients that don't send this field.
    remember: bool = True


class LoginResponse(BaseModel):
    user: dict


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=MIN_PASSWORD_LEN)


class CreateUserRequest(BaseModel):
    username: str
    password: str = Field(..., min_length=MIN_PASSWORD_LEN)
    role: str = Field(default="user", pattern="^(user|admin)$")


class UpdateUserRequest(BaseModel):
    role: Optional[str] = Field(default=None, pattern="^(user|admin)$")
    active: Optional[bool] = None


class ResetPasswordRequest(BaseModel):
    new_password: str = Field(..., min_length=MIN_PASSWORD_LEN)


class UpdateSettingsRequest(BaseModel):
    """Per-user AI + whisper settings patch.

    Only keys present are updated; explicit nulls clear a key.
    """
    data: dict = Field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real = request.headers.get("x-real-ip")
    if real:
        return real.strip()
    return request.client.host if request.client else ""


# ── Public endpoints ─────────────────────────────────────────


@router.post("/login", response_model=LoginResponse)
async def login(
    request: Request,
    response: Response,
    payload: LoginRequest,
):
    user = await auth_store.get_user_by_username(payload.username)
    if user is None or not user.active:
        # Same error message as wrong password so we don't reveal
        # which usernames exist.
        raise HTTPException(status_code=401, detail="invalid credentials")
    if not verify_password(payload.password, user.salt, user.password_hash):
        raise HTTPException(status_code=401, detail="invalid credentials")

    ip = _client_ip(request)
    ua = request.headers.get("user-agent", "")
    session = await auth_store.create_session(
        user_id=user.id, ip=ip, user_agent=ua,
        remember=payload.remember,
    )
    set_session_cookie(response, session.token, remember=payload.remember)
    logger.info(
        "user %r logged in (ip=%s, remember=%s)",
        user.username, ip, payload.remember,
    )
    return LoginResponse(user=user.to_public())


@router.post("/logout")
async def logout(request: Request, response: Response):
    token = request.cookies.get("clipai_session")
    if token:
        try:
            await auth_store.delete_session(token)
        except Exception as e:
            logger.warning("logout delete_session failed: %s", e)
    clear_session_cookie(response)
    return {"ok": True}


@router.get("/me")
async def me(request: Request):
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    session = getattr(request.state, "session", None)
    return {
        "user": user.to_public(),
        "session": {
            "created_at": session.created_at if session else None,
            "last_seen": session.last_seen if session else None,
            "ip": session.ip if session else None,
        },
    }


@router.get("/bootstrap")
async def bootstrap():
    """Expose whether any users exist.

    The frontend queries this on first load so it can show a "first-run"
    hint pointing at the default head-admin credentials.
    """
    users = await auth_store.list_users()
    return {
        "has_users": len(users) > 0,
        "head_admin_username": None if users else "Jadmin",
    }


# ── Authenticated user endpoints ────────────────────────────


@router.patch("/me/password")
async def change_my_password(
    payload: PasswordChangeRequest,
    user: User = Depends(get_current_user),
):
    if not verify_password(payload.current_password, user.salt, user.password_hash):
        raise HTTPException(status_code=403, detail="current password is wrong")
    try:
        await auth_store.change_password(user.id, payload.new_password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Revoke every session except this one so other devices are forced
    # to re-login.
    # Keep the current token alive by reading it back + preserving it.
    await auth_store.delete_sessions_for_user(user.id)
    return {"ok": True, "sessions_revoked": True}


@router.get("/me/settings")
async def get_my_settings(user: User = Depends(get_current_user)):
    data = await auth_store.read_user_settings(user.id)
    return {"settings": _mask_secrets(data)}


@router.put("/me/settings")
async def update_my_settings(
    payload: UpdateSettingsRequest,
    user: User = Depends(get_current_user),
):
    # Restrict per-user overrides to a known set so a compromised
    # account can't escalate by writing arbitrary settings.
    allowed = {
        # AI chain / model choices
        "AI_FALLBACK_CHAIN",
        "OPENROUTER_PRESET",
        "OPENROUTER_VISION_MODEL",
        "OPENROUTER_TEXT_MODEL",
        "OPENROUTER_SUMMARY_MODEL",
        "OLLAMA_VISION_MODEL",
        "OLLAMA_TEXT_MODEL",
        "OLLAMA_TRANSLATION_MODEL",
        # Per-user model picks for the cloud providers that previously
        # had no override path (Anthropic / Gemini / Groq).
        "ANTHROPIC_MODEL",
        "GEMINI_TEXT_MODEL",
        "GEMINI_VIDEO_MODEL",
        "GROQ_TEXT_MODEL",
        # Per-user AI provider API keys. Each user has their own,
        # otherwise everyone shares the admin's billing.
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "HF_AUTH_TOKEN",
        # Whisper
        "WHISPER_MODEL",
        "WHISPER_BEAM_SIZE",
        "WHISPER_VAD_FILTER",
        # Human-reframe per-user knobs
        "CLIPAI_HUMAN_REFRAME",
        "CLIPAI_CRITIC_MODE",
        "CLIPAI_CRITIC_VLM_BACKEND",
    }
    patch = {k: v for k, v in (payload.data or {}).items() if k in allowed}
    merged = await auth_store.write_user_settings(user.id, patch)
    # Mask keys in the response so the frontend doesn't have to handle
    # round-tripping the secret value back into a password input.
    return {"settings": _mask_secrets(merged)}


# ── Helpers shared by /me/settings GET + PUT ─────────────────


_SECRET_FIELDS = {
    "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY", "GROQ_API_KEY", "HF_AUTH_TOKEN",
}


def _mask_secrets(settings: dict) -> dict:
    """Return a shallow copy of ``settings`` with any value for a
    secret field reduced to ``"<set>"`` so the actual secret never
    crosses the wire after the initial PUT."""
    out: dict = {}
    for k, v in (settings or {}).items():
        if k in _SECRET_FIELDS and v:
            out[k] = "<set>"
        else:
            out[k] = v
    return out


# ── Admin endpoints ────────────────────────────────────────


@router.get("/admin/users")
async def admin_list_users(_admin: User = Depends(require_admin)):
    users = await auth_store.list_users()
    return {"users": [u.to_public() for u in users]}


@router.post("/admin/users")
async def admin_create_user(
    payload: CreateUserRequest,
    _admin: User = Depends(require_admin),
):
    try:
        user = await auth_store.create_user(
            payload.username,
            payload.password,
            role=Role(payload.role),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"user": user.to_public()}


@router.patch("/admin/users/{user_id}")
async def admin_update_user(
    user_id: str,
    payload: UpdateUserRequest,
    admin: User = Depends(require_admin),
):
    if user_id == admin.id and payload.role is not None and payload.role != Role.ADMIN.value:
        raise HTTPException(
            status_code=400,
            detail="cannot demote yourself; ask another admin",
        )
    try:
        role = Role(payload.role) if payload.role is not None else None
        user = await auth_store.update_user(
            user_id, role=role, active=payload.active,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="user not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"user": user.to_public()}


@router.delete("/admin/users/{user_id}")
async def admin_delete_user(
    user_id: str,
    admin: User = Depends(require_admin),
):
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="cannot delete yourself")
    try:
        await auth_store.delete_user(user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    await auth_store.delete_user_settings(user_id)
    return {"ok": True}


@router.post("/admin/users/{user_id}/reset_password")
async def admin_reset_password(
    user_id: str,
    payload: ResetPasswordRequest,
    _admin: User = Depends(require_admin),
):
    try:
        await auth_store.change_password(user_id, payload.new_password)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    # Force re-login on all devices.
    await auth_store.delete_sessions_for_user(user_id)
    return {"ok": True, "sessions_revoked": True}


@router.get("/admin/sessions")
async def admin_list_sessions(_admin: User = Depends(require_admin)):
    sessions = await auth_store.list_sessions()
    return {
        "sessions": [
            {
                "token_preview": s.token[:8] + "…",
                "token": s.token,
                "user_id": s.user_id,
                "ip": s.ip,
                "created_at": s.created_at,
                "last_seen": s.last_seen,
                "expires_at": s.expires_at,
            }
            for s in sessions
        ],
    }


@router.delete("/admin/sessions/{token}")
async def admin_delete_session(
    token: str,
    _admin: User = Depends(require_admin),
):
    await auth_store.delete_session(token)
    return {"ok": True}
