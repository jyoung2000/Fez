"""Box cloud provider.

Implemented against the Box v2.0 REST API with ``httpx``. The Python SDK
(``boxsdk``) has an opinionated credential store that fights our
encrypted-JSON persistence, so we drive OAuth directly.

Box quirks this module handles explicitly:

1. **Refresh-token rotation.** Box invalidates the old refresh token
   every time you exchange one. If we don't persist the freshly minted
   refresh token returned from ``/oauth2/token`` we lock the user out
   after the first refresh. See :meth:`BoxProvider.refresh`.
2. **Download redirect.** ``GET /files/{id}/content`` returns a 302 to a
   short-lived download URL. ``httpx`` follows redirects for us but the
   ``Authorization`` header is stripped on the redirected host, which is
   actually what we want — the signed URL is pre-authorized.
3. **Folder id 0.** The root folder in Box is literally ``"0"``, not
   ``"root"``.
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
from ..mime import BOX_SEARCH_EXTENSIONS, is_video_extension, is_video_mime
from ..retry import CloudProviderError, retry_request

logger = logging.getLogger(__name__)


AUTHORIZE_URL = "https://account.box.com/api/oauth2/authorize"
TOKEN_URL = "https://api.box.com/oauth2/token"
REVOKE_URL = "https://api.box.com/oauth2/revoke"
API_BASE = "https://api.box.com/2.0"

SCOPE = "root_readwrite"  # Box's standard scope; the app-level permission
                           # matrix in the developer console decides what the
                           # user actually sees. For a read-only ClipAI
                           # experience leave all write checkboxes off there.

ROOT_FOLDER_ID = "0"

_ITEM_FIELDS = "id,name,type,size,modified_at,extension"


class BoxProvider(CloudProvider):
    name = "box"

    def is_configured(self) -> bool:
        return bool(settings.BOX_CLIENT_ID and settings.BOX_CLIENT_SECRET)

    # ── OAuth ────────────────────────────────────────────────────────────

    def build_authorize_url(self, state: str) -> str:
        params = {
            "response_type": "code",
            "client_id": settings.BOX_CLIENT_ID,
            "redirect_uri": settings.BOX_REDIRECT_URI,
            "state": state,
            "scope": SCOPE,
        }
        return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"

    async def exchange_code(self, code: str) -> TokenBundle:
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": settings.BOX_CLIENT_ID,
            "client_secret": settings.BOX_CLIENT_SECRET,
            "redirect_uri": settings.BOX_REDIRECT_URI,
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await retry_request(
                lambda: client.post(TOKEN_URL, data=data),
                description="box exchange_code",
            )
            if response.status_code != 200:
                raise CloudProviderError(
                    f"box token exchange failed: HTTP {response.status_code}",
                    status_code=response.status_code,
                )
            body = response.json()
        return self._bundle_from_response(body)

    async def refresh(self, refresh_token: str) -> TokenBundle:
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": settings.BOX_CLIENT_ID,
            "client_secret": settings.BOX_CLIENT_SECRET,
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await retry_request(
                lambda: client.post(TOKEN_URL, data=data),
                description="box refresh",
            )
            if response.status_code != 200:
                raise CloudProviderError(
                    f"box refresh failed: HTTP {response.status_code}",
                    status_code=response.status_code,
                )
            body = response.json()
        bundle = self._bundle_from_response(body)
        # Box rotates refresh tokens on every refresh; the response is
        # guaranteed to contain a new one, and the old one is now dead.
        # If the provider ever omits it, fall back to the old value so
        # we don't end up with a row that has no way to refresh again.
        if not bundle.refresh_token:
            logger.warning(
                "box refresh response did not include a new refresh_token — "
                "keeping the previous one (this should not happen)."
            )
            bundle.refresh_token = refresh_token
        return bundle

    async def revoke(self, token: str) -> None:
        data = {
            "client_id": settings.BOX_CLIENT_ID,
            "client_secret": settings.BOX_CLIENT_SECRET,
            "token": token,
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                await client.post(REVOKE_URL, data=data)
            except httpx.HTTPError as exc:
                logger.info("box revoke failed (non-fatal): %s", exc)

    async def get_account_info(self, access_token: str) -> AccountInfo:
        body = await self._api_get(
            "users/me",
            access_token,
            params={"fields": "id,login,name"},
            description="users_me",
        )
        return AccountInfo(
            provider_user_id=str(body.get("id") or "unknown"),
            display_name=str(body.get("login") or body.get("name") or "Box"),
        )

    # ── Browsing ─────────────────────────────────────────────────────────

    async def list_folder(
        self,
        access_token: str,
        folder_id: Optional[str],
        page_token: Optional[str],
    ) -> FolderPage:
        folder = folder_id if folder_id else ROOT_FOLDER_ID
        params: dict[str, object] = {
            "fields": _ITEM_FIELDS,
            "limit": 100,
        }
        if page_token:
            try:
                params["offset"] = int(page_token)
            except ValueError:
                params["offset"] = 0
        body = await self._api_get(
            f"folders/{folder}/items",
            access_token,
            params=params,
            description="list_folder",
        )
        folders: list[FolderMeta] = []
        files: list[FileMeta] = []
        for item in body.get("entries", []):
            if item.get("type") == "folder":
                folders.append(
                    FolderMeta(
                        id=str(item["id"]),
                        name=item.get("name", ""),
                        modified_at=item.get("modified_at"),
                    )
                )
            elif item.get("type") == "file":
                if not _looks_like_video(item):
                    continue
                files.append(self._file_meta_from_entry(item))
        next_token = self._compute_next_offset(body)
        return FolderPage(folders=folders, files=files, next_page_token=next_token)

    async def search(
        self,
        access_token: str,
        query: str,
        page_token: Optional[str],
    ) -> FolderPage:
        params: dict[str, object] = {
            "query": query,
            "type": "file",
            "file_extensions": ",".join(BOX_SEARCH_EXTENSIONS),
            "fields": _ITEM_FIELDS,
            "limit": 50,
        }
        if page_token:
            try:
                params["offset"] = int(page_token)
            except ValueError:
                params["offset"] = 0
        body = await self._api_get(
            "search",
            access_token,
            params=params,
            description="search",
        )
        files: list[FileMeta] = []
        for item in body.get("entries", []):
            if item.get("type") != "file":
                continue
            if not _looks_like_video(item):
                continue
            files.append(self._file_meta_from_entry(item))
        next_token = self._compute_next_offset(body)
        return FolderPage(folders=[], files=files, next_page_token=next_token)

    async def get_file_metadata(
        self,
        access_token: str,
        file_id: str,
    ) -> FileMeta:
        body = await self._api_get(
            f"files/{file_id}",
            access_token,
            params={"fields": _ITEM_FIELDS},
            description="file_metadata",
        )
        return self._file_meta_from_entry(body)

    async def open_stream(
        self,
        access_token: str,
        file_id: str,
    ) -> AsyncIterator[bytes]:
        return self._stream(access_token, file_id)

    async def _stream(
        self,
        access_token: str,
        file_id: str,
    ) -> AsyncIterator[bytes]:
        url = f"{API_BASE}/files/{file_id}/content"
        headers = {"Authorization": f"Bearer {access_token}"}
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", url, headers=headers, follow_redirects=True) as response:
                if response.status_code == 401:
                    raise CloudProviderError("box download: unauthorized", status_code=401)
                if response.status_code != 200:
                    raise CloudProviderError(
                        f"box download failed: HTTP {response.status_code}",
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
                description=f"box {description}",
            )
            if response.status_code == 401:
                raise CloudProviderError(
                    "box: unauthorized (token expired?)",
                    status_code=401,
                )
            if response.status_code != 200:
                raise CloudProviderError(
                    f"box {description} failed: HTTP {response.status_code}",
                    status_code=response.status_code,
                )
            return response.json()

    def _bundle_from_response(self, body: dict) -> TokenBundle:
        expires_in = body.get("expires_in")
        expires_at: Optional[datetime] = None
        if expires_in:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))
        # Box returns ``restricted_to`` rather than ``scope``; we store an
        # empty list and rely on the developer-console permission matrix.
        return TokenBundle(
            access_token=body.get("access_token", ""),
            refresh_token=body.get("refresh_token"),
            expires_at=expires_at,
            scopes=[SCOPE],
        )

    def _file_meta_from_entry(self, item: dict) -> FileMeta:
        name = item.get("name", "unnamed")
        ext = (item.get("extension") or "").lstrip(".").lower()
        mime = _mime_for_box_extension(ext)
        size = item.get("size")
        return FileMeta(
            id=str(item["id"]),
            name=name,
            size=int(size) if size is not None else None,
            mime_type=mime,
            modified_at=item.get("modified_at"),
            thumbnail_url=None,  # Box thumbnails require a separate signed call; skip for now.
        )

    def _compute_next_offset(self, body: dict) -> Optional[str]:
        total = body.get("total_count")
        offset = body.get("offset", 0)
        limit = body.get("limit", 0)
        if total is None:
            return None
        try:
            next_offset = int(offset) + int(limit)
            if next_offset < int(total):
                return str(next_offset)
        except (TypeError, ValueError):
            return None
        return None


# ── module helpers ──────────────────────────────────────────────────────────

def _looks_like_video(item: dict) -> bool:
    ext = (item.get("extension") or "").lstrip(".").lower()
    if ext in BOX_SEARCH_EXTENSIONS:
        return True
    # Fall back on the name, just in case Box omits the extension field.
    return is_video_extension(item.get("name", ""))


_BOX_EXT_TO_MIME = {
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "mkv": "video/x-matroska",
    "webm": "video/webm",
    "avi": "video/x-msvideo",
    "m4v": "video/x-m4v",
    "3gp": "video/3gpp",
    "mpeg": "video/mpeg",
    "mpg": "video/mpeg",
}


def _mime_for_box_extension(ext: str) -> str:
    return _BOX_EXT_TO_MIME.get(ext, "video/mp4")
