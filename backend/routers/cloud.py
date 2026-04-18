"""HTTP endpoints under ``/api/cloud``.

The router is intentionally thin: it does input validation and delegates
every non-trivial operation to :mod:`backend.app.cloud.service` or a
concrete :class:`~backend.app.cloud.base.CloudProvider`.

Flow map:

- ``GET    /api/cloud/providers``                — list status per provider
- ``GET    /api/cloud/{provider}/authorize``     — returns {authorize_url, state}
- ``GET    /api/cloud/{provider}/callback``      — OAuth redirect target
- ``DELETE /api/cloud/accounts/{account_id}``    — disconnect
- ``GET    /api/cloud/{provider}/browse``        — folder listing
- ``GET    /api/cloud/{provider}/search``        — name search
- ``POST   /api/cloud/{provider}/import``        — streamed import → job_id
"""

from __future__ import annotations

import logging
import secrets
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from backend.app.cloud import accounts
from backend.app.cloud.auth import current_user_id
from backend.app.cloud.providers import cloud_storage_enabled, get_provider, all_providers
from backend.app.cloud.service import (
    CloudServiceError,
    call_with_refresh,
    import_cloud_file,
)
from backend.config import settings
from backend.services.ingest import IngestMetadata
from backend.services.pipeline import broadcast_ws

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cloud", tags=["cloud-storage"])


# ── helpers ─────────────────────────────────────────────────────────────────


def _ensure_enabled() -> None:
    if not cloud_storage_enabled():
        raise HTTPException(
            status_code=503,
            detail="Cloud storage is disabled. Set CLIPAI_CLOUD_STORAGE_ENABLED=true.",
        )


def _raise_service_error(exc: CloudServiceError) -> None:
    raise HTTPException(status_code=exc.status_code, detail=str(exc))


def _provider_or_404(name: str):
    _ensure_enabled()
    provider = get_provider(name)
    if provider is None:
        raise HTTPException(status_code=404, detail=f"Unknown cloud provider: {name}")
    return provider


# ── schemas ─────────────────────────────────────────────────────────────────


class ProviderStatus(BaseModel):
    name: str
    configured: bool
    connected: bool
    account: Optional[dict] = None


class ProvidersResponse(BaseModel):
    enabled: bool
    providers: list[ProviderStatus]


class AuthorizeResponse(BaseModel):
    authorize_url: str
    state: str


class FolderPageResponse(BaseModel):
    folders: list[dict]
    files: list[dict]
    next_page_token: Optional[str]


class ImportRequest(BaseModel):
    file_id: str = Field(..., description="Provider-specific file id to import")
    language: str = ""
    subtitle_language: str = ""
    content_type_override: str = ""
    game_type: str = ""
    anime_subtype: str = ""
    music_subtype: str = ""
    sports_subtype: str = ""


class ImportResponse(BaseModel):
    job_id: str
    status: str
    filename: str


# ── providers listing ──────────────────────────────────────────────────────


@router.get("/providers", response_model=ProvidersResponse)
async def list_providers() -> ProvidersResponse:
    """Return every known provider and whether the current user is connected."""
    if not cloud_storage_enabled():
        return ProvidersResponse(enabled=False, providers=[])

    user_id = current_user_id()
    user_accounts = {a.provider: a for a in await accounts.list_for_user(user_id)}

    result: list[ProviderStatus] = []
    for name, provider in all_providers().items():
        account = user_accounts.get(name)
        result.append(
            ProviderStatus(
                name=name,
                configured=provider.is_configured(),
                connected=account is not None,
                account=account.to_public() if account else None,
            )
        )
    return ProvidersResponse(enabled=True, providers=result)


# ── OAuth ──────────────────────────────────────────────────────────────────


@router.get("/{provider}/authorize", response_model=AuthorizeResponse)
async def authorize(provider: str) -> AuthorizeResponse:
    prov = _provider_or_404(provider)
    if not prov.is_configured():
        raise HTTPException(
            status_code=503,
            detail=f"{provider} is not configured on the server. See docs/cloud-storage/SETUP.md.",
        )
    state = secrets.token_urlsafe(32)
    await accounts.remember_state(state, provider)
    return AuthorizeResponse(authorize_url=prov.build_authorize_url(state), state=state)


@router.get("/{provider}/callback")
async def callback(
    provider: str,
    code: str = Query(..., description="OAuth authorization code"),
    state: str = Query(..., description="CSRF token returned by /authorize"),
):
    prov = _provider_or_404(provider)
    expected = await accounts.consume_state(state)
    if expected != provider:
        # Don't leak which half was wrong (timing / enumeration safety).
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state")

    try:
        bundle = await prov.exchange_code(code)
        info = await prov.get_account_info(bundle.access_token)
    except Exception as exc:
        logger.exception("OAuth callback failed for %s", provider)
        raise HTTPException(status_code=502, detail=f"OAuth exchange failed: {exc}")

    await accounts.upsert_for_provider(
        user_id=current_user_id(),
        provider=provider,
        provider_user_id=info.provider_user_id,
        display_name=info.display_name,
        access_token=bundle.access_token,
        refresh_token=bundle.refresh_token,
        token_expires_at=bundle.expires_at.isoformat() if bundle.expires_at else None,
        scopes=bundle.scopes,
    )

    # Redirect back into the frontend Settings page, which polls
    # /providers on mount to refresh its state.
    return RedirectResponse(url=f"/settings?connected={provider}", status_code=302)


@router.delete("/accounts/{account_id}")
async def disconnect(account_id: str):
    _ensure_enabled()
    account = await accounts.get(account_id)
    if not account or account.user_id != current_user_id():
        raise HTTPException(status_code=404, detail="Account not found")
    provider = get_provider(account.provider)
    if provider is not None:
        try:
            await provider.revoke(account.access_token)
        except Exception as exc:
            logger.info("Provider revoke failed (non-fatal): %s", exc)
    removed = await accounts.delete(account_id)
    return {"removed": removed}


# ── browse / search ────────────────────────────────────────────────────────


async def _account_for(provider_name: str):
    _ensure_enabled()
    account = await accounts.get_for_provider(current_user_id(), provider_name)
    if account is None:
        raise HTTPException(
            status_code=401,
            detail=f"{provider_name} is not connected. Connect it in Settings first.",
        )
    return account


@router.get("/{provider}/browse", response_model=FolderPageResponse)
async def browse(
    provider: str,
    folder_id: Optional[str] = Query(None),
    page_token: Optional[str] = Query(None),
):
    prov = _provider_or_404(provider)
    account = await _account_for(provider)
    try:
        page = await call_with_refresh(
            account,
            lambda token: prov.list_folder(token, folder_id, page_token),
        )
    except CloudServiceError as exc:
        _raise_service_error(exc)
    except Exception as exc:
        logger.exception("browse failed for %s folder=%s", provider, folder_id)
        raise HTTPException(status_code=502, detail=f"browse failed: {exc}")

    return FolderPageResponse(
        folders=[f.__dict__ for f in page.folders],
        files=[f.__dict__ for f in page.files],
        next_page_token=page.next_page_token,
    )


@router.get("/{provider}/search", response_model=FolderPageResponse)
async def search(
    provider: str,
    q: str = Query(..., min_length=1),
    page_token: Optional[str] = Query(None),
):
    prov = _provider_or_404(provider)
    account = await _account_for(provider)
    try:
        page = await call_with_refresh(
            account,
            lambda token: prov.search(token, q, page_token),
        )
    except CloudServiceError as exc:
        _raise_service_error(exc)
    except Exception as exc:
        logger.exception("search failed for %s q=%r", provider, q)
        raise HTTPException(status_code=502, detail=f"search failed: {exc}")

    return FolderPageResponse(
        folders=[],
        files=[f.__dict__ for f in page.files],
        next_page_token=page.next_page_token,
    )


# ── import ─────────────────────────────────────────────────────────────────


@router.post("/{provider}/import", response_model=ImportResponse)
async def import_file(
    provider: str,
    payload: ImportRequest,
    background_tasks: BackgroundTasks,
):
    """Start a server-side import → returns the job_id immediately.

    The actual download happens in a background task so the HTTP response
    comes back before the (potentially large) file is on disk. The
    frontend navigates to the usual processing view and the existing
    WebSocket on ``/ws/jobs/{job_id}`` picks up the progress channel —
    we emit ``downloading_from_cloud`` status updates from the download
    loop so the existing progress UI renders without changes.
    """
    prov = _provider_or_404(provider)
    account = await _account_for(provider)

    # Resolve the filename up front so we can include it in the response
    # and have something reasonable for the pending job row.
    try:
        meta = await call_with_refresh(
            account,
            lambda token: prov.get_file_metadata(token, payload.file_id),
        )
    except CloudServiceError as exc:
        _raise_service_error(exc)
    except Exception as exc:
        logger.exception("import metadata fetch failed for %s file=%s", provider, payload.file_id)
        raise HTTPException(status_code=502, detail=f"metadata fetch failed: {exc}")

    metadata = IngestMetadata.from_form(
        language=payload.language,
        subtitle_language=payload.subtitle_language,
        content_type_override=payload.content_type_override,
        game_type=payload.game_type,
        anime_subtype=payload.anime_subtype,
        music_subtype=payload.music_subtype,
        sports_subtype=payload.sports_subtype,
        source=provider,
    )

    # We need to expose a job_id in the response BEFORE running the
    # import (the frontend immediately navigates to the processing
    # screen and opens a ws on /ws/jobs/{job_id}). import_cloud_file
    # generates its own job_id internally, so we have to wrap it in a
    # background task that uses a pre-allocated id and threads it
    # through. To keep the public surface small we pass the id by
    # closing over it here.
    import uuid

    pending_job_id = str(uuid.uuid4())

    last_percent = -1

    async def _on_progress(downloaded: int, total: Optional[int]) -> None:
        nonlocal last_percent
        percent = int(downloaded / total * 100) if total and total > 0 else 0
        if percent == last_percent:
            return
        last_percent = percent
        await broadcast_ws(
            pending_job_id,
            {
                "type": "status",
                "status": "downloading_from_cloud",
                "progress": percent,
                "message": (
                    f"Downloading from {provider}: {percent}%"
                    if total
                    else f"Downloading from {provider}..."
                ),
            },
        )

    async def _run() -> None:
        try:
            await broadcast_ws(
                pending_job_id,
                {
                    "type": "status",
                    "status": "downloading_from_cloud",
                    "progress": 0,
                    "message": f"Starting download from {provider}...",
                },
            )
            await import_cloud_file(
                account,
                payload.file_id,
                metadata=metadata,
                progress=_on_progress,
                pending_job_id=pending_job_id,
            )
        except Exception as exc:
            logger.exception(
                "Cloud import failed (job_id=%s provider=%s)",
                pending_job_id,
                provider,
            )
            await broadcast_ws(
                pending_job_id,
                {
                    "type": "status",
                    "status": "failed",
                    "progress": 0,
                    "message": f"Cloud import failed: {exc}",
                },
            )

    background_tasks.add_task(_run)

    return ImportResponse(
        job_id=pending_job_id,
        status="downloading_from_cloud",
        filename=meta.name,
    )


# ── Credentials management ────────────────────────────────────────────────
#
# The Unraid / env-var pathway still works: if the operator sets
# ``GOOGLE_DRIVE_CLIENT_ID`` etc. via the container template, nothing here
# changes. But the out-of-the-box experience for users who just pulled
# the image and want to click "Connect" is broken when env vars aren't
# set, so this endpoint pair lets the user paste their OAuth client
# credentials directly in the ClipAI Settings page. Saved credentials
# are persisted to ``user_settings.json`` on the data volume, so they
# survive container restarts without touching the ``.env`` file.


class CloudProviderCredentials(BaseModel):
    client_id: str = ""
    client_secret: str = ""
    redirect_uri: str = ""


class CloudCredentialsResponse(BaseModel):
    """One row per provider. Secret values are masked on read — the
    frontend never sees the raw client secret after it's persisted.
    """
    enabled: bool
    providers: dict[str, dict]


class SaveCredentialsRequest(BaseModel):
    provider: str
    client_id: str = ""
    client_secret: str = ""
    redirect_uri: str = ""


_CRED_FIELDS: dict[str, dict[str, str]] = {
    "google_drive": {
        "client_id": "GOOGLE_DRIVE_CLIENT_ID",
        "client_secret": "GOOGLE_DRIVE_CLIENT_SECRET",
        "redirect_uri": "GOOGLE_DRIVE_REDIRECT_URI",
    },
    "box": {
        "client_id": "BOX_CLIENT_ID",
        "client_secret": "BOX_CLIENT_SECRET",
        "redirect_uri": "BOX_REDIRECT_URI",
    },
}


def _mask_secret(value: str) -> str:
    """Return a masked version of ``value`` safe for the frontend.

    Preserves the last 4 characters so the user can recognise which key
    they pasted, but never sends the full secret back.
    """
    if not value:
        return ""
    if len(value) <= 4:
        return "••••"
    return "••••" + value[-4:]


@router.get("/config", response_model=CloudCredentialsResponse)
async def get_cloud_config() -> CloudCredentialsResponse:
    """Return the current cloud OAuth client credentials (secrets masked).

    The frontend uses this to pre-fill the credentials form so a user
    who has already entered their Google Drive client id can see it
    without being forced to re-paste it.
    """
    out: dict[str, dict] = {}
    for provider, mapping in _CRED_FIELDS.items():
        out[provider] = {
            "client_id": str(getattr(settings, mapping["client_id"], "") or ""),
            # Never ship the raw secret back — mask it.
            "client_secret_masked": _mask_secret(
                str(getattr(settings, mapping["client_secret"], "") or "")
            ),
            "client_secret_set": bool(
                str(getattr(settings, mapping["client_secret"], "") or "").strip()
            ),
            "redirect_uri": str(getattr(settings, mapping["redirect_uri"], "") or ""),
        }
    return CloudCredentialsResponse(
        enabled=cloud_storage_enabled(),
        providers=out,
    )


@router.post("/config")
async def save_cloud_config(req: SaveCredentialsRequest) -> dict:
    """Persist OAuth client credentials for one provider.

    Blank secret fields are treated as "leave the existing secret
    alone" — that way a user can update just the redirect URI without
    being forced to re-paste the client secret (which the frontend
    only ever saw as masked).
    """
    mapping = _CRED_FIELDS.get(req.provider)
    if not mapping:
        raise HTTPException(
            status_code=404, detail=f"Unknown cloud provider: {req.provider}"
        )

    # Apply the fields to the in-memory settings object. Everything here
    # is optional so we can partial-update.
    if req.client_id.strip():
        setattr(settings, mapping["client_id"], req.client_id.strip())
    if req.client_secret.strip():
        setattr(settings, mapping["client_secret"], req.client_secret.strip())
    if req.redirect_uri.strip():
        setattr(settings, mapping["redirect_uri"], req.redirect_uri.strip())

    # Also mirror into the .env file (best effort) and persist to the
    # Docker volume JSON for restart survival. We deliberately import
    # inside the function so the cloud router has no import-time
    # dependency on the settings router.
    try:
        from backend.routers.settings import (
            _find_env_file, _upsert_env_var, _persist_user_settings,
        )
        env_path = _find_env_file()
        if env_path:
            for field_key, env_key in mapping.items():
                value = getattr(settings, env_key, "")
                if value:
                    _upsert_env_var(env_path, env_key, str(value))
        _persist_user_settings()
    except Exception as exc:
        logger.warning("Cloud credential persistence failed (non-fatal): %s", exc)

    # Re-check configured state so the response tells the frontend whether
    # it should flip the card to "Connect" mode.
    provider = get_provider(req.provider)
    configured = bool(provider and provider.is_configured())
    return {
        "status": "saved",
        "provider": req.provider,
        "configured": configured,
    }


@router.post("/config/test/{provider}")
async def test_cloud_config(provider: str) -> dict:
    """Sanity-check the saved credentials for one provider.

    This deliberately does **not** try to run a full OAuth flow — that
    requires a browser redirect and a user consent. Instead it checks:

    1. The provider is recognised.
    2. Client id / secret / redirect URI are all present on the server.
    3. The redirect URI parses as a valid http(s) URL.
    4. The provider-specific authorize URL builds without raising.
    5. (Google Drive only) the token endpoint responds to a cheap
       probe request — a real client id must be known to Google.

    Returns ``{status: "ok" | "error", checks: [...]}`` so the UI can
    show which step passed and which failed.
    """
    prov = _provider_or_404(provider)
    mapping = _CRED_FIELDS[provider]
    checks: list[dict] = []

    client_id = str(getattr(settings, mapping["client_id"], "") or "").strip()
    client_secret = str(getattr(settings, mapping["client_secret"], "") or "").strip()
    redirect_uri = str(getattr(settings, mapping["redirect_uri"], "") or "").strip()

    def _add(label: str, ok: bool, detail: str = "") -> None:
        checks.append({"label": label, "ok": ok, "detail": detail})

    _add("client_id set", bool(client_id), "" if client_id else "Field is empty")
    _add("client_secret set", bool(client_secret), "" if client_secret else "Field is empty")
    _add("redirect_uri set", bool(redirect_uri), "" if redirect_uri else "Field is empty")

    if redirect_uri:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(redirect_uri)
            valid_scheme = parsed.scheme in ("http", "https")
            valid_host = bool(parsed.netloc)
            valid_path = parsed.path.endswith(f"/api/cloud/{provider}/callback")
            _add(
                "redirect_uri scheme",
                valid_scheme,
                "" if valid_scheme else f"Expected http/https, got {parsed.scheme!r}",
            )
            _add(
                "redirect_uri host",
                valid_host,
                "" if valid_host else "Host/port is empty",
            )
            _add(
                "redirect_uri path",
                valid_path,
                (
                    ""
                    if valid_path
                    else f"Must end with /api/cloud/{provider}/callback"
                ),
            )
        except Exception as exc:
            _add("redirect_uri parseable", False, str(exc))

    # Provider-specific spot checks
    if provider == "google_drive":
        if client_id and not client_id.endswith(".apps.googleusercontent.com"):
            _add(
                "client_id format",
                False,
                "Google client IDs typically end with .apps.googleusercontent.com",
            )
        else:
            _add("client_id format", True, "")
    elif provider == "box":
        _add("client_id format", True, "")

    # Build the authorize URL as a smoke test.
    configured_now = prov.is_configured()
    if configured_now:
        try:
            url = prov.build_authorize_url("test-state-token")
            _add(
                "authorize URL builds",
                bool(url and url.startswith(("http://", "https://"))),
                "" if url else "Provider returned empty URL",
            )
        except Exception as exc:
            _add("authorize URL builds", False, f"{type(exc).__name__}: {exc}")
    else:
        _add(
            "authorize URL builds",
            False,
            "Provider reports is_configured()==False; set all three fields first.",
        )

    # Cheap live probe for Google Drive: hit the token endpoint with an
    # obviously invalid grant and confirm Google answers with a proper
    # JSON error (which means the client_id is recognised).
    if provider == "google_drive" and configured_now:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(
                    "https://oauth2.googleapis.com/token",
                    data={
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "grant_type": "refresh_token",
                        "refresh_token": "clipai-credential-probe-invalid",
                    },
                )
            # Google answers 400 + json for bogus grants when the
            # client is recognised; 401 when it isn't.
            if resp.status_code == 400:
                _add("google token endpoint reachable", True, "")
            elif resp.status_code == 401:
                body = resp.json() if resp.content else {}
                _add(
                    "google token endpoint reachable",
                    False,
                    f"Google rejected the client: {body.get('error', 'unauthorized')}",
                )
            else:
                _add(
                    "google token endpoint reachable",
                    True,
                    f"HTTP {resp.status_code} (accepted as live)",
                )
        except Exception as exc:
            _add(
                "google token endpoint reachable",
                False,
                f"Network error: {exc}",
            )

    overall_ok = all(c["ok"] for c in checks)
    return {
        "status": "ok" if overall_ok else "error",
        "configured": configured_now,
        "checks": checks,
    }
