"""Tests for Dropbox storage plugin."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from homesec.models.config import DropboxStorageConfig
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
        self.session_started = False
        self.session_finished = False
        self.session_appends = 0
        self.metadata: set[str] = set()

    def files_upload(self, data: bytes, dest_path: str, mode: object) -> None:
        self.uploaded.append((dest_path, data))
        self.metadata.add(dest_path)

    def files_upload_session_start(self, chunk: bytes) -> _FakeSession:
        self.session_started = True
        return _FakeSession(session_id="session_1")

    def files_upload_session_append_v2(self, chunk: bytes, cursor: _FakeUploadSessionCursor) -> None:
        self.session_appends += 1
        cursor.offset += len(chunk)

    def files_upload_session_finish(
        self,
        chunk: bytes,
        cursor: _FakeUploadSessionCursor,
        commit: _FakeCommitInfo,
    ) -> None:
        self.session_finished = True
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

    # Then chunked upload paths are used
    assert client.session_started is True
    assert client.session_finished is True


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
