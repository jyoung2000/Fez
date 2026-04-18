"""Provider-agnostic data types and the :class:`CloudProvider` Protocol.

Each concrete provider (Google Drive, Box) implements this interface so
the router and frontend remain provider-oblivious. The frontend sees a
uniform shape for folder listings and file metadata regardless of which
cloud it's talking to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator, Literal, Optional, Protocol, runtime_checkable


ProviderName = Literal["google_drive", "box"]


@dataclass
class TokenBundle:
    """Result of an OAuth token exchange / refresh."""

    access_token: str
    refresh_token: Optional[str]
    expires_at: Optional[datetime]
    scopes: list[str] = field(default_factory=list)


@dataclass
class AccountInfo:
    """Information about the cloud account behind an access token."""

    provider_user_id: str
    display_name: str  # Email or name shown in the UI


@dataclass
class FileMeta:
    """A file entry inside a folder listing or search result."""

    id: str
    name: str
    size: Optional[int]
    mime_type: str
    modified_at: Optional[str]  # ISO-8601 string
    thumbnail_url: Optional[str] = None  # Short-lived signed URL (may be None)
    path: Optional[str] = None


@dataclass
class FolderMeta:
    """A folder entry (only shown in browse responses, never in search)."""

    id: str
    name: str
    modified_at: Optional[str] = None


@dataclass
class FolderPage:
    """One page of a folder listing or search response."""

    folders: list[FolderMeta]
    files: list[FileMeta]
    next_page_token: Optional[str]


@runtime_checkable
class CloudProvider(Protocol):
    """Provider-agnostic interface every cloud implementation satisfies."""

    name: ProviderName

    # ── configuration / health ───────────────────────────────────────────
    def is_configured(self) -> bool:
        """Return True when the operator has supplied client credentials."""
        ...

    # ── OAuth ────────────────────────────────────────────────────────────
    def build_authorize_url(self, state: str) -> str: ...

    async def exchange_code(self, code: str) -> TokenBundle: ...

    async def refresh(self, refresh_token: str) -> TokenBundle: ...

    async def get_account_info(self, access_token: str) -> AccountInfo: ...

    # Optional best-effort revoke during disconnect.
    async def revoke(self, token: str) -> None: ...

    # ── browsing ─────────────────────────────────────────────────────────
    async def list_folder(
        self,
        access_token: str,
        folder_id: Optional[str],
        page_token: Optional[str],
    ) -> FolderPage: ...

    async def search(
        self,
        access_token: str,
        query: str,
        page_token: Optional[str],
    ) -> FolderPage: ...

    async def get_file_metadata(
        self,
        access_token: str,
        file_id: str,
    ) -> FileMeta: ...

    async def open_stream(
        self,
        access_token: str,
        file_id: str,
    ) -> AsyncIterator[bytes]: ...
