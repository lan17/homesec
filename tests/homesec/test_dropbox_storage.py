"""Tests for Dropbox storage plugin."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from homesec.plugins.storage.dropbox import DropboxStorageConfig
from homesec.plugins.storage.dropbox import DropboxStorage


class _FakeApiError(Exception):
    def __init__(self, message: str = "", *, error: object | None = None) -> None:
        super().__init__(message)
        self.error = error


class _FakePathLookupError:
    def __init__(self, *, not_found: bool) -> None:
        self._not_found = not_found

    def is_not_found(self) -> bool:
        return self._not_found


class _FakeDeleteError:
    def __init__(self, *, not_found: bool) -> None:
        self._not_found = not_found

    def is_path_lookup(self) -> bool:
        return True

    def get_path_lookup(self) -> _FakePathLookupError:
        return _FakePathLookupError(not_found=self._not_found)


class _FakeWriteMode:
    overwrite = "overwrite"


class _FakeSession:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id


class _FakeUploadSessionCursor:
    def __init__(self, session_id: str, offset: int) -> None:
        self.session_id = session_id
        self.offset = offset


class _FakeCommitInfo:
    def __init__(self, path: str, mode: object) -> None:
        self.path = path
        self.mode = mode


class _FakeDropboxClient:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}
        self.uploaded: list[tuple[str, bytes]] = []
        self.deleted: list[str] = []
        # Track API calls made (for contract testing)
        self.api_calls: list[str] = []
        self.metadata: set[str] = set()

    def files_upload(self, data: bytes, dest_path: str, mode: object) -> None:
        self.api_calls.append("files_upload")
        self.uploaded.append((dest_path, data))
        self.metadata.add(dest_path)

    def files_upload_session_start(self, chunk: bytes) -> _FakeSession:
        self.api_calls.append("files_upload_session_start")
        return _FakeSession(session_id="session_1")

    def files_upload_session_append_v2(
        self, chunk: bytes, cursor: _FakeUploadSessionCursor
    ) -> None:
        self.api_calls.append("files_upload_session_append_v2")
        cursor.offset += len(chunk)

    def files_upload_session_finish(
        self,
        chunk: bytes,
        cursor: _FakeUploadSessionCursor,
        commit: _FakeCommitInfo,
    ) -> None:
        self.api_calls.append("files_upload_session_finish")
        self.metadata.add(commit.path)

    def files_get_metadata(self, path: str) -> object:
        if path not in self.metadata:
            raise _FakeApiError("missing")
        return {"path": path}

    def files_download_to_file(self, local: str, remote: str) -> None:
        Path(local).write_bytes(b"downloaded")

    def files_delete_v2(self, path: str) -> None:
        if path not in self.metadata:
            raise _FakeApiError("missing", error=_FakeDeleteError(not_found=True))
        self.metadata.remove(path)
        self.deleted.append(path)

    def users_get_current_account(self) -> object:
        return {"account_id": "test"}


def _fake_dropbox_module(client: _FakeDropboxClient) -> SimpleNamespace:
    def _dropbox_ctor(*args: object, **kwargs: object) -> _FakeDropboxClient:
        if args and "token" not in kwargs:
            kwargs["token"] = args[0]
        client.kwargs = dict(kwargs)
        return client

    return SimpleNamespace(
        Dropbox=_dropbox_ctor,
        files=SimpleNamespace(
            WriteMode=_FakeWriteMode,
            UploadSessionCursor=_FakeUploadSessionCursor,
            CommitInfo=_FakeCommitInfo,
        ),
        exceptions=SimpleNamespace(ApiError=_FakeApiError),
    )


@pytest.mark.asyncio
async def test_dropbox_storage_uses_token_auth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Dropbox storage should use token auth when available."""
    # Given a fake Dropbox client and token env var
    client = _FakeDropboxClient()
    monkeypatch.setattr(
        "homesec.plugins.storage.dropbox.dropbox",
        _fake_dropbox_module(client),
    )
    monkeypatch.setenv("DROPBOX_TOKEN", "token")
    config = DropboxStorageConfig(root="/homecam", token_env="DROPBOX_TOKEN")
    storage = DropboxStorage(config)

    # When uploading a small file
    file_path = tmp_path / "clip.mp4"
    file_path.write_bytes(b"video")
    result = await storage.put_file(file_path, "front/clip.mp4")

    # Then the token auth path is used and upload succeeds
    assert client.kwargs == {"token": "token"}
    assert result.storage_uri == "dropbox:/homecam/front/clip.mp4"
    assert any(path.endswith("/homecam/front/clip.mp4") for path, _ in client.uploaded)


@pytest.mark.asyncio
async def test_dropbox_storage_uses_refresh_token_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropbox storage should use refresh token auth when token missing."""
    # Given refresh token env vars
    client = _FakeDropboxClient()
    monkeypatch.setattr(
        "homesec.plugins.storage.dropbox.dropbox",
        _fake_dropbox_module(client),
    )
    monkeypatch.delenv("DROPBOX_TOKEN", raising=False)
    monkeypatch.setenv("DROPBOX_APP_KEY", "app_key")
    monkeypatch.setenv("DROPBOX_APP_SECRET", "app_secret")
    monkeypatch.setenv("DROPBOX_REFRESH_TOKEN", "refresh_token")
    config = DropboxStorageConfig(root="/homecam")

    # When initializing storage
    DropboxStorage(config)

    # Then refresh token auth is used
    assert client.kwargs == {
        "app_key": "app_key",
        "app_secret": "app_secret",
        "oauth2_refresh_token": "refresh_token",
    }


@pytest.mark.asyncio
async def test_dropbox_storage_chunked_upload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Dropbox storage should use chunked uploads for large files."""
    # Given a fake Dropbox client and small chunk size
    client = _FakeDropboxClient()
    monkeypatch.setattr(
        "homesec.plugins.storage.dropbox.dropbox",
        _fake_dropbox_module(client),
    )
    monkeypatch.setenv("DROPBOX_TOKEN", "token")
    monkeypatch.setattr("homesec.plugins.storage.dropbox.CHUNK_SIZE", 2)
    storage = DropboxStorage(DropboxStorageConfig(root="/homecam"))

    # When uploading a file larger than CHUNK_SIZE
    file_path = tmp_path / "big.mp4"
    file_path.write_bytes(b"12345")
    await storage.put_file(file_path, "front/big.mp4")

    # Then chunked upload API calls are made
    assert "files_upload_session_start" in client.api_calls
    assert "files_upload_session_finish" in client.api_calls


@pytest.mark.asyncio
async def test_dropbox_storage_exists_handles_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropbox storage should return False for missing objects."""
    # Given a fake Dropbox client with no metadata
    client = _FakeDropboxClient()
    monkeypatch.setattr(
        "homesec.plugins.storage.dropbox.dropbox",
        _fake_dropbox_module(client),
    )
    monkeypatch.setenv("DROPBOX_TOKEN", "token")
    storage = DropboxStorage(DropboxStorageConfig(root="/homecam"))

    # When checking for a missing object
    exists = await storage.exists("dropbox:/homecam/missing.mp4")

    # Then exists returns False
    assert exists is False


@pytest.mark.asyncio
async def test_dropbox_storage_get_view_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropbox storage should build view URLs for Dropbox URIs."""
    # Given a storage with a custom view URL prefix
    client = _FakeDropboxClient()
    monkeypatch.setattr(
        "homesec.plugins.storage.dropbox.dropbox",
        _fake_dropbox_module(client),
    )
    monkeypatch.setenv("DROPBOX_TOKEN", "token")
    storage = DropboxStorage(
        DropboxStorageConfig(root="/homecam", web_url_prefix="https://example.com/view")
    )

    # When computing a view URL
    view_url = await storage.get_view_url("dropbox:/homecam/front/clip.mp4")

    # Then it uses the configured prefix and path
    assert view_url == "https://example.com/view/homecam/front/clip.mp4"
    assert await storage.get_view_url("s3://bucket/key") is None


@pytest.mark.asyncio
async def test_dropbox_storage_delete_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropbox storage delete should treat missing objects as success."""
    # Given a fake Dropbox client and storage
    client = _FakeDropboxClient()
    monkeypatch.setattr(
        "homesec.plugins.storage.dropbox.dropbox",
        _fake_dropbox_module(client),
    )
    monkeypatch.setenv("DROPBOX_TOKEN", "token")
    storage = DropboxStorage(DropboxStorageConfig(root="/homecam"))

    # When deleting a missing object
    await storage.delete("dropbox:/homecam/missing.mp4")

    # Then no exception is raised
    assert client.deleted == []


@pytest.mark.asyncio
async def test_dropbox_storage_delete_removes_existing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropbox storage delete should call files_delete_v2 for existing objects."""
    # Given a fake Dropbox client with an existing path
    client = _FakeDropboxClient()
    client.metadata.add("/homecam/front/clip.mp4")
    monkeypatch.setattr(
        "homesec.plugins.storage.dropbox.dropbox",
        _fake_dropbox_module(client),
    )
    monkeypatch.setenv("DROPBOX_TOKEN", "token")
    storage = DropboxStorage(DropboxStorageConfig(root="/homecam"))

    # When deleting that object
    await storage.delete("dropbox:/homecam/front/clip.mp4")

    # Then it is recorded as deleted
    assert client.deleted == ["/homecam/front/clip.mp4"]


class TestDropboxStoragePing:
    """Tests for ping method."""

    @pytest.mark.asyncio
    async def test_ping_returns_true_when_connected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns True when Dropbox API responds successfully."""
        # Given: A working Dropbox client
        client = _FakeDropboxClient()
        monkeypatch.setattr(
            "homesec.plugins.storage.dropbox.dropbox",
            _fake_dropbox_module(client),
        )
        monkeypatch.setenv("DROPBOX_TOKEN", "token")
        storage = DropboxStorage(DropboxStorageConfig(root="/homecam"))

        # When: Calling ping
        result = await storage.ping()

        # Then: Returns True
        assert result is True

    @pytest.mark.asyncio
    async def test_ping_returns_false_on_api_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns False when Dropbox API fails."""
        # Given: A client that raises on users_get_current_account

        class _FakeDropboxClientPingFails(_FakeDropboxClient):
            def users_get_current_account(self) -> object:
                raise _FakeApiError("Auth failed")

        client = _FakeDropboxClientPingFails()
        monkeypatch.setattr(
            "homesec.plugins.storage.dropbox.dropbox",
            _fake_dropbox_module(client),
        )
        monkeypatch.setenv("DROPBOX_TOKEN", "token")
        storage = DropboxStorage(DropboxStorageConfig(root="/homecam"))

        # When: Calling ping
        result = await storage.ping()

        # Then: Returns False
        assert result is False


class TestDropboxStorageShutdown:
    """Tests for shutdown method."""

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Shutdown can be called multiple times safely."""
        # Given: A Dropbox storage
        client = _FakeDropboxClient()
        monkeypatch.setattr(
            "homesec.plugins.storage.dropbox.dropbox",
            _fake_dropbox_module(client),
        )
        monkeypatch.setenv("DROPBOX_TOKEN", "token")
        storage = DropboxStorage(DropboxStorageConfig(root="/homecam"))

        # When: Calling shutdown multiple times
        await storage.shutdown()
        await storage.shutdown()
        await storage.shutdown()

        # Then: No exception raised (idempotent behavior verified by no exception)

    @pytest.mark.asyncio
    async def test_operations_fail_after_shutdown(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Operations raise RuntimeError after shutdown."""
        # Given: A storage that has been shut down
        client = _FakeDropboxClient()
        monkeypatch.setattr(
            "homesec.plugins.storage.dropbox.dropbox",
            _fake_dropbox_module(client),
        )
        monkeypatch.setenv("DROPBOX_TOKEN", "token")
        storage = DropboxStorage(DropboxStorageConfig(root="/homecam"))
        await storage.shutdown()

        file_path = tmp_path / "clip.mp4"
        file_path.write_bytes(b"video")

        # When/Then: Operations raise RuntimeError
        with pytest.raises(RuntimeError, match="shut down"):
            await storage.put_file(file_path, "clip.mp4")

        with pytest.raises(RuntimeError, match="shut down"):
            await storage.get("dropbox:/homecam/clip.mp4", file_path)

        with pytest.raises(RuntimeError, match="shut down"):
            await storage.exists("dropbox:/homecam/clip.mp4")

        with pytest.raises(RuntimeError, match="shut down"):
            await storage.delete("dropbox:/homecam/clip.mp4")

        with pytest.raises(RuntimeError, match="shut down"):
            await storage.get_view_url("dropbox:/homecam/clip.mp4")


class TestDropboxStorageGet:
    """Tests for get (download) method."""

    @pytest.mark.asyncio
    async def test_get_downloads_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Downloads file from Dropbox to local path."""
        # Given: A Dropbox storage with a file
        client = _FakeDropboxClient()
        monkeypatch.setattr(
            "homesec.plugins.storage.dropbox.dropbox",
            _fake_dropbox_module(client),
        )
        monkeypatch.setenv("DROPBOX_TOKEN", "token")
        storage = DropboxStorage(DropboxStorageConfig(root="/homecam"))

        # When: Downloading a file
        local_path = tmp_path / "downloaded.mp4"
        await storage.get("dropbox:/homecam/front/clip.mp4", local_path)

        # Then: File is downloaded
        assert local_path.exists()
        assert local_path.read_bytes() == b"downloaded"

    @pytest.mark.asyncio
    async def test_get_creates_parent_directories(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Creates parent directories for local path if needed."""
        # Given: A Dropbox storage
        client = _FakeDropboxClient()
        monkeypatch.setattr(
            "homesec.plugins.storage.dropbox.dropbox",
            _fake_dropbox_module(client),
        )
        monkeypatch.setenv("DROPBOX_TOKEN", "token")
        storage = DropboxStorage(DropboxStorageConfig(root="/homecam"))

        # When: Downloading to a nested path that doesn't exist
        local_path = tmp_path / "nested" / "dir" / "downloaded.mp4"
        await storage.get("dropbox:/homecam/clip.mp4", local_path)

        # Then: Parent dirs created and file downloaded
        assert local_path.exists()

    @pytest.mark.asyncio
    async def test_get_rejects_invalid_uri(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Raises ValueError for non-dropbox storage URIs."""
        # Given: A Dropbox storage
        client = _FakeDropboxClient()
        monkeypatch.setattr(
            "homesec.plugins.storage.dropbox.dropbox",
            _fake_dropbox_module(client),
        )
        monkeypatch.setenv("DROPBOX_TOKEN", "token")
        storage = DropboxStorage(DropboxStorageConfig(root="/homecam"))

        # When/Then: Get with invalid URI raises ValueError
        with pytest.raises(ValueError, match="Invalid storage_uri"):
            await storage.get("s3://bucket/key", tmp_path / "file.mp4")


class TestDropboxStorageExists:
    """Tests for exists method."""

    @pytest.mark.asyncio
    async def test_exists_returns_true_for_existing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns True when file exists in Dropbox."""
        # Given: A storage with an existing file
        client = _FakeDropboxClient()
        client.metadata.add("/homecam/front/clip.mp4")
        monkeypatch.setattr(
            "homesec.plugins.storage.dropbox.dropbox",
            _fake_dropbox_module(client),
        )
        monkeypatch.setenv("DROPBOX_TOKEN", "token")
        storage = DropboxStorage(DropboxStorageConfig(root="/homecam"))

        # When: Checking if it exists
        result = await storage.exists("dropbox:/homecam/front/clip.mp4")

        # Then: Returns True
        assert result is True

    @pytest.mark.asyncio
    async def test_exists_returns_false_for_non_dropbox_uri(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returns False for non-dropbox storage URIs."""
        # Given: A Dropbox storage
        client = _FakeDropboxClient()
        monkeypatch.setattr(
            "homesec.plugins.storage.dropbox.dropbox",
            _fake_dropbox_module(client),
        )
        monkeypatch.setenv("DROPBOX_TOKEN", "token")
        storage = DropboxStorage(DropboxStorageConfig(root="/homecam"))

        # When: Checking non-dropbox URI
        result = await storage.exists("s3://bucket/key")

        # Then: Returns False
        assert result is False


class TestDropboxStoragePathValidation:
    """Tests for path validation."""

    @pytest.mark.asyncio
    async def test_rejects_path_traversal(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Rejects paths with .. traversal."""
        # Given: A Dropbox storage
        client = _FakeDropboxClient()
        monkeypatch.setattr(
            "homesec.plugins.storage.dropbox.dropbox",
            _fake_dropbox_module(client),
        )
        monkeypatch.setenv("DROPBOX_TOKEN", "token")
        storage = DropboxStorage(DropboxStorageConfig(root="/homecam"))

        file_path = tmp_path / "clip.mp4"
        file_path.write_bytes(b"video")

        # When/Then: Path traversal is rejected
        with pytest.raises(ValueError, match="Invalid dest_path"):
            await storage.put_file(file_path, "../../../etc/passwd")

    @pytest.mark.asyncio
    async def test_rejects_backslash_paths(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Rejects paths with backslashes."""
        # Given: A Dropbox storage
        client = _FakeDropboxClient()
        monkeypatch.setattr(
            "homesec.plugins.storage.dropbox.dropbox",
            _fake_dropbox_module(client),
        )
        monkeypatch.setenv("DROPBOX_TOKEN", "token")
        storage = DropboxStorage(DropboxStorageConfig(root="/homecam"))

        file_path = tmp_path / "clip.mp4"
        file_path.write_bytes(b"video")

        # When/Then: Backslash paths are rejected
        with pytest.raises(ValueError, match="Invalid dest_path"):
            await storage.put_file(file_path, "front\\clip.mp4")

    @pytest.mark.asyncio
    async def test_rejects_empty_dest_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Rejects empty destination path."""
        # Given: A Dropbox storage
        client = _FakeDropboxClient()
        monkeypatch.setattr(
            "homesec.plugins.storage.dropbox.dropbox",
            _fake_dropbox_module(client),
        )
        monkeypatch.setenv("DROPBOX_TOKEN", "token")
        storage = DropboxStorage(DropboxStorageConfig(root="/homecam"))

        file_path = tmp_path / "clip.mp4"
        file_path.write_bytes(b"video")

        # When/Then: Empty path is rejected
        with pytest.raises(ValueError, match="Invalid dest_path"):
            await storage.put_file(file_path, "")


class TestDropboxStorageCredentials:
    """Tests for credential handling."""

    @pytest.mark.asyncio
    async def test_raises_when_no_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises ValueError when no credentials are available."""
        # Given: No Dropbox credentials in env
        client = _FakeDropboxClient()
        monkeypatch.setattr(
            "homesec.plugins.storage.dropbox.dropbox",
            _fake_dropbox_module(client),
        )
        monkeypatch.delenv("DROPBOX_TOKEN", raising=False)
        monkeypatch.delenv("DROPBOX_APP_KEY", raising=False)
        monkeypatch.delenv("DROPBOX_APP_SECRET", raising=False)
        monkeypatch.delenv("DROPBOX_REFRESH_TOKEN", raising=False)

        config = DropboxStorageConfig(root="/homecam")

        # When/Then: Creating storage raises ValueError
        with pytest.raises(ValueError, match="Missing Dropbox credentials"):
            DropboxStorage(config)

    @pytest.mark.asyncio
    async def test_raises_when_partial_refresh_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Raises ValueError when only some refresh credentials are set."""
        # Given: Only app_key is set (missing app_secret and refresh_token)
        client = _FakeDropboxClient()
        monkeypatch.setattr(
            "homesec.plugins.storage.dropbox.dropbox",
            _fake_dropbox_module(client),
        )
        monkeypatch.delenv("DROPBOX_TOKEN", raising=False)
        monkeypatch.setenv("DROPBOX_APP_KEY", "app_key")
        monkeypatch.delenv("DROPBOX_APP_SECRET", raising=False)
        monkeypatch.delenv("DROPBOX_REFRESH_TOKEN", raising=False)

        config = DropboxStorageConfig(root="/homecam")

        # When/Then: Creating storage raises ValueError
        with pytest.raises(ValueError, match="Missing Dropbox credentials"):
            DropboxStorage(config)


class TestDropboxStorageDelete:
    """Additional tests for delete method."""

    @pytest.mark.asyncio
    async def test_delete_rejects_invalid_uri(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises ValueError for non-dropbox storage URIs."""
        # Given: A Dropbox storage
        client = _FakeDropboxClient()
        monkeypatch.setattr(
            "homesec.plugins.storage.dropbox.dropbox",
            _fake_dropbox_module(client),
        )
        monkeypatch.setenv("DROPBOX_TOKEN", "token")
        storage = DropboxStorage(DropboxStorageConfig(root="/homecam"))

        # When/Then: Delete with invalid URI raises ValueError
        with pytest.raises(ValueError, match="Invalid storage_uri"):
            await storage.delete("s3://bucket/key")


class TestDropboxStorageChunkedUpload:
    """Additional tests for chunked upload edge cases."""

    @pytest.mark.asyncio
    async def test_chunked_upload_with_append(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Chunked upload uses append for middle chunks."""
        # Given: A fake Dropbox client with very small chunk size
        client = _FakeDropboxClient()
        monkeypatch.setattr(
            "homesec.plugins.storage.dropbox.dropbox",
            _fake_dropbox_module(client),
        )
        monkeypatch.setenv("DROPBOX_TOKEN", "token")
        # Set chunk size to 2 bytes to force multiple chunks
        monkeypatch.setattr("homesec.plugins.storage.dropbox.CHUNK_SIZE", 2)
        storage = DropboxStorage(DropboxStorageConfig(root="/homecam"))

        # When: Uploading a file that needs multiple chunks (7 bytes = 4 chunks with size 2)
        file_path = tmp_path / "big.mp4"
        file_path.write_bytes(b"1234567")
        await storage.put_file(file_path, "front/big.mp4")

        # Then: Session start, appends, and finish API calls are all made
        assert "files_upload_session_start" in client.api_calls
        assert "files_upload_session_append_v2" in client.api_calls
        assert "files_upload_session_finish" in client.api_calls
