"""Dropbox storage backend plugin."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

dropbox: Any

try:
    import dropbox as _dropbox  # type: ignore[import-untyped]
except Exception:
    dropbox = None
else:
    dropbox = _dropbox


from homesec.interfaces import StorageBackend
from homesec.models.config import DropboxStorageConfig
from homesec.models.storage import StorageUploadResult
from homesec.plugins.registry import PluginType, plugin

logger = logging.getLogger(__name__)

CHUNK_SIZE = 4 * 1024 * 1024


@plugin(plugin_type=PluginType.STORAGE, name="dropbox")
class DropboxStorage(StorageBackend):
    """Dropbox storage backend.

    Uses dropbox SDK for file operations.
    Implements idempotent uploads with overwrite mode.

    Supports two auth modes:
    1. Simple token: Set DROPBOX_TOKEN env var
    2. Refresh token flow: Set DROPBOX_APP_KEY, DROPBOX_APP_SECRET, DROPBOX_REFRESH_TOKEN
    """

    config_cls = DropboxStorageConfig

    @classmethod
    def create(cls, config: DropboxStorageConfig) -> StorageBackend:
        return cls(config)

    def __init__(self, config: DropboxStorageConfig) -> None:
        """Initialize Dropbox storage with config validation.

        Required config:
            root: Root path in Dropbox (e.g., /homecam)

        Optional config:
            token_env: Env var name for simple token auth (default: DROPBOX_TOKEN)
            app_key_env: Env var name for app key (default: DROPBOX_APP_KEY)
            app_secret_env: Env var name for app secret (default: DROPBOX_APP_SECRET)
            refresh_token_env: Env var name for refresh token (default: DROPBOX_REFRESH_TOKEN)
            web_url_prefix: URL prefix for view links (default: https://www.dropbox.com/home)
        """
        self.root = str(config.root).rstrip("/")
        self.web_url_prefix = str(config.web_url_prefix)

        # Initialize Dropbox client using env vars
        if dropbox is None:
            raise RuntimeError("Missing dependency: dropbox. Install with: uv pip install dropbox")
        self.client = self._create_client(config)
        self._shutdown_called = False

        logger.info("DropboxStorage initialized: root=%s", self.root)

    def _create_client(self, config: DropboxStorageConfig) -> dropbox.Dropbox:
        """Create Dropbox client from env vars.

        Tries simple token first, then falls back to refresh token flow.
        """
        if dropbox is None:
            raise RuntimeError("Missing dependency: dropbox. Install with: uv pip install dropbox")
        dbx = dropbox

        # Try simple token auth first
        token_var = str(config.token_env)
        token = os.getenv(token_var)
        if token:
            logger.info("Using Dropbox simple token auth")
            return dbx.Dropbox(token)

        # Try refresh token flow
        app_key_var = str(config.app_key_env)
        app_secret_var = str(config.app_secret_env)
        refresh_token_var = str(config.refresh_token_env)

        app_key = os.getenv(app_key_var)
        app_secret = os.getenv(app_secret_var)
        refresh_token = os.getenv(refresh_token_var)

        if app_key and app_secret and refresh_token:
            logger.info("Using Dropbox refresh token auth")
            return dbx.Dropbox(
                app_key=app_key,
                app_secret=app_secret,
                oauth2_refresh_token=refresh_token,
            )

        raise ValueError(
            f"Missing Dropbox credentials. Set {token_var} or "
            f"({app_key_var}, {app_secret_var}, {refresh_token_var})."
        )

    async def put_file(self, local_path: Path, dest_path: str) -> StorageUploadResult:
        """Upload file to Dropbox."""
        self._ensure_open()

        dest_path = self._full_dest_path(dest_path)

        # Run blocking upload in executor
        await asyncio.to_thread(self._upload_file, local_path, dest_path)

        storage_uri = f"dropbox:{dest_path}"
        view_url = await self.get_view_url(storage_uri)
        return StorageUploadResult(storage_uri=storage_uri, view_url=view_url)

    async def get_view_url(self, storage_uri: str) -> str | None:
        """Compute a web-accessible URL for a Dropbox storage URI."""
        self._ensure_open()
        if not storage_uri.startswith("dropbox:"):
            return None
        path = storage_uri[8:]
        prefix = self.web_url_prefix.rstrip("/")
        return f"{prefix}{path}"

    def _upload_file(self, local_path: Path, dest_path: str) -> None:
        """Upload file (blocking operation)."""
        if dropbox is None:
            raise RuntimeError("Missing dependency: dropbox. Install with: uv pip install dropbox")
        dbx = dropbox
        file_size = local_path.stat().st_size
        with open(local_path, "rb") as f:
            if file_size <= CHUNK_SIZE:
                self.client.files_upload(
                    f.read(),
                    dest_path,
                    mode=dbx.files.WriteMode.overwrite,
                )
            else:
                self._upload_file_chunked(f, dest_path, file_size)
        logger.debug("Uploaded to Dropbox: %s", dest_path)

    def _upload_file_chunked(self, file_handle: BinaryIO, dest_path: str, file_size: int) -> None:
        """Upload file in chunks using a Dropbox upload session."""
        chunk = file_handle.read(CHUNK_SIZE)
        if not chunk:
            raise ValueError("Cannot upload empty file")

        session = self.client.files_upload_session_start(chunk)
        if dropbox is None:
            raise RuntimeError("Missing dependency: dropbox. Install with: uv pip install dropbox")
        dbx = dropbox
        cursor = dbx.files.UploadSessionCursor(
            session_id=session.session_id,
            offset=file_handle.tell(),
        )
        commit = dbx.files.CommitInfo(
            path=dest_path,
            mode=dbx.files.WriteMode.overwrite,
        )

        while file_handle.tell() < file_size:
            chunk = file_handle.read(CHUNK_SIZE)
            if not chunk:
                raise RuntimeError("Unexpected end of file during chunked upload")
            if file_handle.tell() >= file_size:
                self.client.files_upload_session_finish(chunk, cursor, commit)
                return
            self.client.files_upload_session_append_v2(chunk, cursor)
            cursor.offset = file_handle.tell()

    async def get(self, storage_uri: str, local_path: Path) -> None:
        """Download file from Dropbox."""
        self._ensure_open()

        # Parse storage_uri
        if not storage_uri.startswith("dropbox:"):
            raise ValueError(f"Invalid storage_uri: {storage_uri}")

        remote_path = storage_uri[8:]  # Strip "dropbox:"

        # Run blocking download in executor
        await asyncio.to_thread(self._download_file, remote_path, local_path)

    def _download_file(self, remote_path: str, local_path: Path) -> None:
        """Download file (blocking operation)."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.files_download_to_file(str(local_path), remote_path)
        logger.debug("Downloaded from Dropbox: %s", remote_path)

    async def exists(self, storage_uri: str) -> bool:
        """Check if file exists in Dropbox."""
        self._ensure_open()

        # Parse storage_uri
        if not storage_uri.startswith("dropbox:"):
            return False

        remote_path = storage_uri[8:]

        # Run blocking check in executor
        return await asyncio.to_thread(self._check_exists, remote_path)

    async def delete(self, storage_uri: str) -> None:
        """Delete file from Dropbox.

        Idempotent: missing files are treated as success.
        """
        self._ensure_open()

        if not storage_uri.startswith("dropbox:"):
            raise ValueError(f"Invalid storage_uri: {storage_uri}")

        remote_path = storage_uri[8:]

        try:
            await asyncio.to_thread(self.client.files_delete_v2, remote_path)
        except dropbox.exceptions.ApiError as exc:
            # Treat missing file as success
            err = getattr(exc, "error", None)
            try:
                if err is not None and getattr(err, "is_path_lookup", lambda: False)():
                    lookup = err.get_path_lookup()
                    if getattr(lookup, "is_not_found", lambda: False)():
                        return
            except Exception:
                pass
            raise

    def _check_exists(self, remote_path: str) -> bool:
        """Check if file exists (blocking operation)."""
        try:
            self.client.files_get_metadata(remote_path)
            return True
        except dropbox.exceptions.ApiError:
            return False

    async def ping(self) -> bool:
        """Health check - verify Dropbox connection."""
        try:
            await asyncio.to_thread(self.client.users_get_current_account)
            return True
        except Exception as e:
            logger.warning("Dropbox ping failed: %s", e, exc_info=True)
            return False

    async def shutdown(self, timeout: float | None = None) -> None:
        """Cleanup resources."""
        _ = timeout
        if self._shutdown_called:
            return

        self._shutdown_called = True
        logger.info("DropboxStorage closed")

    def _ensure_open(self) -> None:
        if self._shutdown_called:
            raise RuntimeError("Storage has been shut down")

    def _full_dest_path(self, dest_path: str) -> str:
        cleaned = str(dest_path).lstrip("/")
        if not cleaned or "\\" in cleaned:
            raise ValueError(f"Invalid dest_path: {dest_path}")
        path = PurePosixPath(cleaned)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Invalid dest_path: {dest_path}")
        return f"{self.root}/{path}"
