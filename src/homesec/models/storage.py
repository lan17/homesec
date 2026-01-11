"""Storage-related data models."""

from __future__ import annotations

from pydantic import BaseModel


class StorageUploadResult(BaseModel):
    """Result of a storage upload."""

    storage_uri: str
    view_url: str | None = None
