"""Concrete cloud-storage provider implementations."""

from __future__ import annotations

from typing import Optional

from backend.config import settings

from ..base import CloudProvider, ProviderName
from .google_drive import GoogleDriveProvider
from .box import BoxProvider

__all__ = [
    "CloudProvider",
    "GoogleDriveProvider",
    "BoxProvider",
    "get_provider",
    "all_providers",
]


_PROVIDERS: dict[ProviderName, CloudProvider] = {
    "google_drive": GoogleDriveProvider(),
    "box": BoxProvider(),
}


def get_provider(name: str) -> Optional[CloudProvider]:
    """Return the provider with the given name, or ``None`` if unknown."""
    return _PROVIDERS.get(name)  # type: ignore[arg-type]


def all_providers() -> dict[ProviderName, CloudProvider]:
    return dict(_PROVIDERS)


def cloud_storage_enabled() -> bool:
    """Global feature flag — flipping this kills all cloud storage features."""
    return bool(getattr(settings, "CLIPAI_CLOUD_STORAGE_ENABLED", True))
