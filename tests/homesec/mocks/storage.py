"""Mock storage backend for testing."""

from __future__ import annotations

import asyncio
from pathlib import Path

from homesec.errors import UploadError
from homesec.models.storage import StorageUploadResult


class MockStorage:
    """Mock implementation of StorageBackend interface for testing.
    
    Stores files in memory dict for test assertions.
    Supports configurable failure injection and delays.
    """

    def __init__(
        self,
        simulate_failure: bool = False,
        delay_s: float = 0.0,
    ) -> None:
        """Initialize mock storage.
        
        Args:
            simulate_failure: If True, put/get operations raise UploadError
            delay_s: Artificial delay before returning
        """
        self.simulate_failure = simulate_failure
        self.delay_s = delay_s
        self.files: dict[str, bytes] = {}  # storage_uri -> file content
        self.put_count = 0
        self.get_count = 0
        self.shutdown_called = False

    async def put_file(self, local_path: Path, dest_path: str) -> StorageUploadResult:
        """Upload file to storage (mock implementation)."""
        self.put_count += 1

        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)

        if self.simulate_failure:
            raise UploadError(
                clip_id=local_path.stem,
                storage_uri=None,
                cause=RuntimeError("Simulated storage upload failure"),
            )

        storage_uri = f"mock://{dest_path}"
        # Simulate reading file content (in tests, file may not exist)
        try:
            self.files[storage_uri] = local_path.read_bytes()
        except FileNotFoundError:
            # In tests, files may not exist - store placeholder
            self.files[storage_uri] = b"mock_content"

        return StorageUploadResult(storage_uri=storage_uri, view_url=storage_uri)

    async def get_view_url(self, storage_uri: str) -> str | None:
        """Return a view URL for the stored object (mock implementation)."""
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)
        return storage_uri

    async def get(self, storage_uri: str, local_path: Path) -> None:
        """Download file from storage to local path (mock implementation)."""
        self.get_count += 1

        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)

        if self.simulate_failure:
            raise RuntimeError("Simulated storage download failure")

        if storage_uri not in self.files:
            raise FileNotFoundError(f"Storage URI not found: {storage_uri}")

        # Write mock content to local path
        local_path.write_bytes(self.files[storage_uri])

    async def exists(self, storage_uri: str) -> bool:
        """Check if file exists in storage (mock implementation)."""
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)

        return storage_uri in self.files

    async def delete(self, storage_uri: str) -> None:
        """Delete file from storage (mock implementation).

        Idempotent: deleting a missing object is a no-op.
        """
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)

        self.files.pop(storage_uri, None)

    async def ping(self) -> bool:
        """Health check (mock implementation)."""
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)

        return not self.simulate_failure

    async def shutdown(self, timeout: float | None = None) -> None:
        """Cleanup resources (no-op for mock)."""
        _ = timeout
        self.shutdown_called = True
