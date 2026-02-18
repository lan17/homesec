"""Runtime controller abstraction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from homesec.models.config import Config
    from homesec.runtime.models import ManagedRuntime


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
