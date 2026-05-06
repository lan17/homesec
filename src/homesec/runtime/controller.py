"""Runtime controller abstraction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from homesec.models.config import Config
    from homesec.models.talk import CameraTalkStatus, TalkInputFormat
    from homesec.runtime.models import (
        CameraPreviewStartRefusal,
        CameraPreviewStatus,
        CameraPreviewStopResult,
        CameraTalkSessionPrepared,
        CameraTalkStartRefusal,
        CameraTalkStopResult,
        ManagedRuntime,
        RuntimeTalkStream,
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

    async def note_preview_viewer_activity(
        self,
        runtime: ManagedRuntime,
        camera_name: str,
        *,
        viewer_id: str | None = None,
    ) -> None:
        """Record successful preview playback activity for a camera."""
        ...

    async def get_talk_status(
        self,
        runtime: ManagedRuntime,
        camera_name: str,
    ) -> CameraTalkStatus:
        """Return the current talk status for a camera."""
        ...

    async def prepare_talk_session(
        self,
        runtime: ManagedRuntime,
        camera_name: str,
        *,
        session_id: str,
        input_format: TalkInputFormat,
    ) -> CameraTalkSessionPrepared | CameraTalkStartRefusal:
        """Reserve a talk session slot for browser stream attachment."""
        ...

    async def open_talk_stream(
        self,
        runtime: ManagedRuntime,
        camera_name: str,
        *,
        session_id: str,
        input_format: TalkInputFormat,
    ) -> RuntimeTalkStream:
        """Open a binary IPC talk stream to the runtime worker."""
        ...

    async def stop_talk_session(
        self,
        runtime: ManagedRuntime,
        camera_name: str,
        *,
        session_id: str,
    ) -> CameraTalkStopResult:
        """Stop a talk session for a camera."""
        ...
