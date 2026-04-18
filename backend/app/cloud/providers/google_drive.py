"""Google Drive cloud provider.

Implemented directly against the Drive v3 REST API with ``httpx`` — we
deliberately avoid ``google-api-python-client`` because it pulls in a
large dependency tree and its transport is sync-first. The surface we
need (OAuth dance, list/search/get, streaming download) is eight
endpoints and the extra code is worth the leaner container image.

Scope: ``https://www.googleapis.com/auth/drive.readonly`` — read-only,
which Google presents to users as a less alarming consent screen than
``drive`` (full access). We never write to the user's drive.
"""

from __future__ import annotations

import logging
import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, Optional

import httpx

from backend.config import settings

from ..base import (
    AccountInfo,
    CloudProvider,
    FileMeta,
    FolderMeta,
    FolderPage,
    TokenBundle,
)
from ..mime import is_video_mime
from ..retry import CloudProviderError, retry_request

logger = logging.getLogger(__name__)


AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
API_BASE = "https://www.googleapis.com/drive/v3"

SCOPE = "https://www.googleapis.com/auth/drive.readonly openid email"

GDRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"

_FILE_FIELDS = (
    "id,name,mimeType,size,modifiedTime,thumbnailLink,parents"
)

# The list_folder query filters to folders + any video MIME, so the user
# never sees docs/images while browsing.
_LIST_Q_TEMPLATE = (
    "'{folder_id}' in parents and trashed=false and "
    "(mimeType='" + GDRIVE_FOLDER_MIME + "' or mimeType contains 'video/')"
)
_LIST_Q_ROOT = (
    "'root' in parents and trashed=false and "
    "(mimeType='" + GDRIVE_FOLDER_MIME + "' or mimeType contains 'video/')"
)

_SEARCH_Q_TEMPLATE = (
    "name contains '{q}' and mimeType contains 'video/' and trashed=false"
)


def _escape_drive_query(value: str) -> str:
    """Escape single quotes and backslashes for a Drive ``q`` parameter."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


class GoogleDriveProvider(CloudProvider):
    name = "google_drive"

    def is_configured(self) -> bool:
        return bool(settings.GOOGLE_DRIVE_CLIENT_ID and settings.GOOGLE_DRIVE_CLIENT_SECRET)

    # ── OAuth ────────────────────────────────────────────────────────────

    def build_authorize_url(self, state: str) -> str:
        params = {
            "client_id": settings.GOOGLE_DRIVE_CLIENT_ID,
            "redirect_uri": settings.GOOGLE_DRIVE_REDIRECT_URI,
            "response_type": "code",
            "scope": SCOPE,
            "access_type": "offline",
            "prompt": "consent",  # force refresh_token on every connect
            "state": state,
            "include_granted_scopes": "true",
        }
        return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"

    async def exchange_code(self, code: str) -> TokenBundle:
        data = {
            "code": code,
            "client_id": settings.GOOGLE_DRIVE_CLIENT_ID,
            "client_secret": settings.GOOGLE_DRIVE_CLIENT_SECRET,
            "redirect_uri": settings.GOOGLE_DRIVE_REDIRECT_URI,
            "grant_type": "authorization_code",
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await retry_request(
                lambda: client.post(TOKEN_URL, data=data),
                description="google_drive exchange_code",
            )
            if response.status_code != 200:
                raise CloudProviderError(
                    f"google_drive token exchange failed: HTTP {response.status_code}",
                    status_code=response.status_code,
                )
            body = response.json()
        return self._bundle_from_response(body)

    async def refresh(self, refresh_token: str) -> TokenBundle:
        data = {
            "refresh_token": refresh_token,
            "client_id": settings.GOOGLE_DRIVE_CLIENT_ID,
            "client_secret": settings.GOOGLE_DRIVE_CLIENT_SECRET,
            "grant_type": "refresh_token",
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await retry_request(
                lambda: client.post(TOKEN_URL, data=data),
                description="google_drive refresh",
            )
            if response.status_code != 200:
                raise CloudProviderError(
                    f"google_drive refresh failed: HTTP {response.status_code}",
                    status_code=response.status_code,
                )
            body = response.json()
        # Google refresh responses do not re-emit the refresh token, so we
        # preserve the one we already have.
        bundle = self._bundle_from_response(body)
        if not bundle.refresh_token:
            bundle.refresh_token = refresh_token
        return bundle

    async def revoke(self, token: str) -> None:
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                await client.post(REVOKE_URL, params={"token": token})
            except httpx.HTTPError as exc:
                # Best-effort only — the account row is already being deleted.
                logger.info("google_drive revoke failed (non-fatal): %s", exc)

    async def get_account_info(self, access_token: str) -> AccountInfo:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await retry_request(
                lambda: client.get(
                    USERINFO_URL,
                    headers={"Authorization": f"Bearer {access_token}"},
                ),
                description="google_drive userinfo",
            )
            if response.status_code != 200:
                raise CloudProviderError(
                    f"google_drive userinfo failed: HTTP {response.status_code}",
                    status_code=response.status_code,
                )
            body = response.json()
        return AccountInfo(
            provider_user_id=str(body.get("sub") or body.get("email") or "unknown"),
            display_name=str(body.get("email") or body.get("name") or "Google Drive"),
        )

    # ── Browsing ─────────────────────────────────────────────────────────

    async def list_folder(
        self,
        access_token: str,
        folder_id: Optional[str],
        page_token: Optional[str],
    ) -> FolderPage:
        if folder_id and folder_id != "root":
            q = _LIST_Q_TEMPLATE.format(folder_id=_escape_drive_query(folder_id))
        else:
            q = _LIST_Q_ROOT
        params = {
            "q": q,
            "fields": f"nextPageToken, files({_FILE_FIELDS})",
            "pageSize": 50,
            "orderBy": "folder,name",
        }
        if page_token:
            params["pageToken"] = page_token
        body = await self._api_get("files", access_token, params=params, description="list_folder")
        return self._page_from_body(body)

    async def search(
        self,
        access_token: str,
        query: str,
        page_token: Optional[str],
    ) -> FolderPage:
        q = _SEARCH_Q_TEMPLATE.format(q=_escape_drive_query(query))
        params = {
            "q": q,
            "fields": f"nextPageToken, files({_FILE_FIELDS})",
            "pageSize": 50,
        }
        if page_token:
            params["pageToken"] = page_token
        body = await self._api_get("files", access_token, params=params, description="search")
        return self._page_from_body(body, folders_allowed=False)

    async def get_file_metadata(
        self,
        access_token: str,
        file_id: str,
    ) -> FileMeta:
        params = {"fields": _FILE_FIELDS}
        body = await self._api_get(
            f"files/{file_id}", access_token, params=params, description="file_metadata"
        )
        return self._file_meta_from_body(body)

    async def open_stream(
        self,
        access_token: str,
        file_id: str,
    ) -> AsyncIterator[bytes]:
        """Yield the file bytes in chunks. Refreshes the token on 401 once.

        This is an async generator. Callers stream it directly into the
        ingest path; we do not buffer the whole file in memory.
        """
        # Generator so we can re-issue the request with a new token if
        # we hit a 401 while already inside the streaming loop.
        return self._stream(access_token, file_id)

    async def _stream(
        self,
        access_token: str,
        file_id: str,
    ) -> AsyncIterator[bytes]:
        url = f"{API_BASE}/files/{file_id}?alt=media"
        headers = {"Authorization": f"Bearer {access_token}"}
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", url, headers=headers, follow_redirects=True) as response:
                if response.status_code == 401:
                    raise CloudProviderError(
                        "google_drive download: unauthorized",
                        status_code=401,
                    )
                if response.status_code != 200:
                    raise CloudProviderError(
                        f"google_drive download failed: HTTP {response.status_code}",
                        status_code=response.status_code,
                    )
                async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                    if chunk:
                        yield chunk

    # ── internal helpers ─────────────────────────────────────────────────

    async def _api_get(
        self,
        path: str,
        access_token: str,
        *,
        params: dict,
        description: str,
    ) -> dict:
        url = f"{API_BASE}/{path}"
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await retry_request(
                lambda: client.get(
                    url,
                    headers={"Authorization": f"Bearer {access_token}"},
                    params=params,
                ),
                description=f"google_drive {description}",
            )
            if response.status_code == 401:
                raise CloudProviderError(
                    "google_drive: unauthorized (token expired?)",
                    status_code=401,
                )
            if response.status_code != 200:
                raise CloudProviderError(
                    f"google_drive {description} failed: HTTP {response.status_code}",
                    status_code=response.status_code,
                )
            return response.json()

    def _bundle_from_response(self, body: dict) -> TokenBundle:
        expires_in = body.get("expires_in")
        expires_at: Optional[datetime] = None
        if expires_in:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))
        scope_str = body.get("scope", "")
        scopes = [s for s in scope_str.split() if s]
        return TokenBundle(
            access_token=body.get("access_token", ""),
            refresh_token=body.get("refresh_token"),
            expires_at=expires_at,
            scopes=scopes,
        )

    def _page_from_body(self, body: dict, *, folders_allowed: bool = True) -> FolderPage:
        folders: list[FolderMeta] = []
        files: list[FileMeta] = []
        for item in body.get("files", []):
            if item.get("mimeType") == GDRIVE_FOLDER_MIME:
                if folders_allowed:
                    folders.append(
                        FolderMeta(
                            id=item["id"],
                            name=item.get("name", ""),
                            modified_at=item.get("modifiedTime"),
                        )
                    )
                continue
            if not is_video_mime(item.get("mimeType", "")):
                continue
            files.append(self._file_meta_from_body(item))
        return FolderPage(
            folders=folders,
            files=files,
            next_page_token=body.get("nextPageToken"),
        )

    def _file_meta_from_body(self, item: dict) -> FileMeta:
        size_raw = item.get("size")
        return FileMeta(
            id=item["id"],
            name=item.get("name", "unnamed"),
            size=int(size_raw) if size_raw is not None else None,
            mime_type=item.get("mimeType", "application/octet-stream"),
            modified_at=item.get("modifiedTime"),
            thumbnail_url=item.get("thumbnailLink"),
        )
