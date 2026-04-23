"""Runtime controller abstraction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from homesec.models.config import Config
    from homesec.runtime.models import (
        CameraPreviewStartRefusal,
        CameraPreviewStatus,
        CameraPreviewStopResult,
        ManagedRuntime,
    )


class RuntimeController(Protocol):
    """Build/start/shutdown contract for a runtime implementation."""

    async def build_candidate(self, config: Config, generation: int) -> ManagedRuntime:
        """Build a candidate runtime bundle."""
        ...

    async def start_runtime(self, runtime: ManagedRuntime) -> None:
        """Start a runtime bundle and run startup preflight."""
        ...

    async def shutdown_runtime(self, runtime: ManagedRuntime) -> None:
        """Gracefully stop and clean up a runtime bundle."""
        ...

    async def shutdown_all(self) -> None:
        """Best-effort cleanup for any active or in-flight runtimes."""
        ...

    async def get_preview_status(
        self,
        runtime: ManagedRuntime,
        camera_name: str,
    ) -> CameraPreviewStatus:
        """Return the current preview status for a camera."""
        ...

    async def ensure_preview_active(
        self,
        runtime: ManagedRuntime,
        camera_name: str,
    ) -> CameraPreviewStatus | CameraPreviewStartRefusal:
        """Ensure preview is attachable for a camera."""
        ...

    async def force_stop_preview(
        self,
        runtime: ManagedRuntime,
        camera_name: str,
    ) -> CameraPreviewStopResult:
        """Force-stop preview for a camera."""
        ...
