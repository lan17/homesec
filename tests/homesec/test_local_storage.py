"""Tests for LocalStorage backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from homesec.plugins.storage.local import LocalStorage, LocalStorageConfig


def _make_storage(tmp_path: Path) -> LocalStorage:
    """Create a LocalStorage instance with tmp_path as root."""
    config = LocalStorageConfig(root=str(tmp_path / "storage"))
    return LocalStorage(config)


class TestLocalStorageHappyPath:
    """Integration tests for complete file lifecycle."""

    @pytest.mark.asyncio
    async def test_put_get_delete_roundtrip(self, tmp_path: Path) -> None:
        """Full lifecycle: put -> exists -> get_view_url -> get -> delete."""
        # Given: A LocalStorage instance and a source file
        storage = _make_storage(tmp_path)
        source_file = tmp_path / "source.mp4"
        source_file.write_bytes(b"video content here")

        # When: Uploading the file
        result = await storage.put_file(source_file, "clips/test.mp4")

        # Then: File is stored and accessible
        assert result.storage_uri.startswith("local:")
        assert "test.mp4" in result.storage_uri
        assert result.view_url is not None
        assert result.view_url.startswith("file://")

        # When: Checking if file exists
        exists = await storage.exists(result.storage_uri)

        # Then: File exists
        assert exists is True

        # When: Getting view URL
        view_url = await storage.get_view_url(result.storage_uri)

        # Then: View URL is returned
        assert view_url is not None
        assert view_url.startswith("file://")

        # When: Downloading the file
        download_path = tmp_path / "downloaded.mp4"
        await storage.get(result.storage_uri, download_path)

        # Then: Content matches original
        assert download_path.read_bytes() == b"video content here"

        # When: Deleting the file
        await storage.delete(result.storage_uri)

        # Then: File no longer exists
        exists_after = await storage.exists(result.storage_uri)
        assert exists_after is False

        await storage.shutdown()

    @pytest.mark.asyncio
    async def test_put_creates_parent_directories(self, tmp_path: Path) -> None:
        """Nested dest paths create parent directories automatically."""
        # Given: A LocalStorage instance and a source file
        storage = _make_storage(tmp_path)
        source_file = tmp_path / "source.mp4"
        source_file.write_bytes(b"nested content")

        # When: Uploading to a deeply nested path
        result = await storage.put_file(source_file, "a/b/c/d/nested.mp4")

        # Then: File is stored successfully
        assert await storage.exists(result.storage_uri)

        # Then: Parent directories were created
        expected_path = tmp_path / "storage" / "a" / "b" / "c" / "d" / "nested.mp4"
        assert expected_path.exists()

        await storage.shutdown()

    @pytest.mark.asyncio
    async def test_get_view_url_returns_file_uri(self, tmp_path: Path) -> None:
        """get_view_url returns a file:// URI for local storage."""
        # Given: A LocalStorage instance with a stored file
        storage = _make_storage(tmp_path)
        source_file = tmp_path / "source.mp4"
        source_file.write_bytes(b"content")
        result = await storage.put_file(source_file, "video.mp4")

        # When: Getting view URL
        view_url = await storage.get_view_url(result.storage_uri)

        # Then: URL is a file:// URI
        assert view_url is not None
        assert view_url.startswith("file://")
        assert "video.mp4" in view_url

        await storage.shutdown()

    @pytest.mark.asyncio
    async def test_get_view_url_returns_none_for_non_local_uri(self, tmp_path: Path) -> None:
        """get_view_url returns None for non-local storage URIs."""
        # Given: A LocalStorage instance
        storage = _make_storage(tmp_path)

        # When: Getting view URL for a non-local URI
        view_url = await storage.get_view_url("s3://bucket/key")

        # Then: None is returned
        assert view_url is None

        await storage.shutdown()

    @pytest.mark.asyncio
    async def test_ping_returns_true_when_root_exists(self, tmp_path: Path) -> None:
        """ping returns True when root directory exists."""
        # Given: A LocalStorage instance (root created on init)
        storage = _make_storage(tmp_path)

        # When: Pinging storage
        result = await storage.ping()

        # Then: ping returns True
        assert result is True

        await storage.shutdown()


class TestLocalStoragePathValidation:
    """Tests for path security validation."""

    @pytest.mark.asyncio
    async def test_rejects_path_traversal_attempts(self, tmp_path: Path) -> None:
        """Path traversal attempts are rejected."""
        # Given: A LocalStorage instance
        storage = _make_storage(tmp_path)
        source_file = tmp_path / "source.mp4"
        source_file.write_bytes(b"malicious")

        # When/Then: Path traversal is rejected
        with pytest.raises(ValueError, match="Invalid dest_path"):
            await storage.put_file(source_file, "../../../etc/passwd")

        with pytest.raises(ValueError, match="Invalid dest_path"):
            await storage.put_file(source_file, "foo/../../../etc/passwd")

        await storage.shutdown()

    @pytest.mark.asyncio
    async def test_rejects_backslash_paths(self, tmp_path: Path) -> None:
        """Windows-style backslash paths are rejected."""
        # Given: A LocalStorage instance
        storage = _make_storage(tmp_path)
        source_file = tmp_path / "source.mp4"
        source_file.write_bytes(b"content")

        # When/Then: Backslash path is rejected
        with pytest.raises(ValueError, match="Invalid dest_path"):
            await storage.put_file(source_file, "foo\\bar\\baz.mp4")

        await storage.shutdown()

    @pytest.mark.asyncio
    async def test_rejects_empty_dest_path(self, tmp_path: Path) -> None:
        """Empty destination paths are rejected."""
        # Given: A LocalStorage instance
        storage = _make_storage(tmp_path)
        source_file = tmp_path / "source.mp4"
        source_file.write_bytes(b"content")

        # When/Then: Empty path is rejected
        with pytest.raises(ValueError, match="Invalid dest_path"):
            await storage.put_file(source_file, "")

        # When/Then: Root-only path (just slashes) is rejected
        with pytest.raises(ValueError, match="Invalid dest_path"):
            await storage.put_file(source_file, "/")

        await storage.shutdown()

    @pytest.mark.asyncio
    async def test_handles_unicode_filenames(self, tmp_path: Path) -> None:
        """Unicode filenames are handled correctly."""
        # Given: A LocalStorage instance and a source file
        storage = _make_storage(tmp_path)
        source_file = tmp_path / "source.mp4"
        source_file.write_bytes(b"unicode content")

        # When: Uploading with unicode filename (Cyrillic and Japanese)
        result = await storage.put_file(source_file, "clips/камера_передняя.mp4")

        # Then: File is stored successfully
        assert await storage.exists(result.storage_uri)

        # When: Another unicode filename
        result2 = await storage.put_file(source_file, "clips/玄関カメラ.mp4")

        # Then: File is stored successfully
        assert await storage.exists(result2.storage_uri)

        await storage.shutdown()


class TestLocalStorageShutdown:
    """Tests for shutdown behavior."""

    @pytest.mark.asyncio
    async def test_operations_fail_after_shutdown(self, tmp_path: Path) -> None:
        """Storage operations raise RuntimeError after shutdown."""
        # Given: A LocalStorage instance that has been shut down
        storage = _make_storage(tmp_path)
        source_file = tmp_path / "source.mp4"
        source_file.write_bytes(b"content")
        await storage.shutdown()

        # When/Then: Operations raise RuntimeError
        with pytest.raises(RuntimeError, match="shut down"):
            await storage.put_file(source_file, "test.mp4")

        with pytest.raises(RuntimeError, match="shut down"):
            await storage.get("local:/some/path", tmp_path / "out.mp4")

        with pytest.raises(RuntimeError, match="shut down"):
            await storage.exists("local:/some/path")

        with pytest.raises(RuntimeError, match="shut down"):
            await storage.delete("local:/some/path")

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(self, tmp_path: Path) -> None:
        """shutdown() can be called multiple times safely."""
        # Given: A LocalStorage instance
        storage = _make_storage(tmp_path)

        # When: Calling shutdown multiple times
        await storage.shutdown()
        await storage.shutdown()
        await storage.shutdown()

        # Then: No exception is raised (idempotent)

    @pytest.mark.asyncio
    async def test_ping_works_after_shutdown(self, tmp_path: Path) -> None:
        """ping() works even after shutdown (read-only operation)."""
        # Given: A LocalStorage instance that has been shut down
        storage = _make_storage(tmp_path)
        await storage.shutdown()

        # When: Pinging storage
        result = await storage.ping()

        # Then: ping still works (doesn't check shutdown state)
        assert result is True


class TestLocalStorageEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_exists_returns_false_for_invalid_uri(self, tmp_path: Path) -> None:
        """exists() returns False for invalid storage URIs."""
        # Given: A LocalStorage instance
        storage = _make_storage(tmp_path)

        # When: Checking existence of invalid URI
        result = await storage.exists("s3://bucket/key")

        # Then: Returns False (not a local URI)
        assert result is False

        await storage.shutdown()

    @pytest.mark.asyncio
    async def test_delete_is_idempotent(self, tmp_path: Path) -> None:
        """Deleting a non-existent file is a no-op."""
        # Given: A LocalStorage instance
        storage = _make_storage(tmp_path)

        # When: Deleting a file that doesn't exist
        await storage.delete("local:" + str(tmp_path / "storage" / "nonexistent.mp4"))

        # Then: No exception is raised (idempotent)

        await storage.shutdown()

    @pytest.mark.asyncio
    async def test_get_missing_file_raises(self, tmp_path: Path) -> None:
        """get() raises when source file doesn't exist."""
        # Given: A LocalStorage instance
        storage = _make_storage(tmp_path)

        # When/Then: Getting non-existent file raises
        with pytest.raises(FileNotFoundError):
            await storage.get(
                "local:" + str(tmp_path / "storage" / "missing.mp4"),
                tmp_path / "out.mp4",
            )

        await storage.shutdown()

    @pytest.mark.asyncio
    async def test_parse_storage_uri_rejects_invalid_prefix(self, tmp_path: Path) -> None:
        """_parse_storage_uri rejects URIs without local: prefix."""
        # Given: A LocalStorage instance
        storage = _make_storage(tmp_path)

        # When/Then: Invalid URI raises ValueError
        with pytest.raises(ValueError, match="Invalid storage_uri"):
            await storage.get("s3://bucket/key", tmp_path / "out.mp4")

        await storage.shutdown()
