"""Runtime controller abstraction and in-process implementation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
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


@dataclass(slots=True)
class InProcessRuntimeController(RuntimeController):
    """In-process runtime controller backed by callback functions."""

    build_candidate_fn: Callable[[Config, int], Awaitable[ManagedRuntime]]
    start_runtime_fn: Callable[[ManagedRuntime], Awaitable[None]]
    shutdown_runtime_fn: Callable[[ManagedRuntime], Awaitable[None]]

    async def build_candidate(self, config: Config, generation: int) -> ManagedRuntime:
        return await self.build_candidate_fn(config, generation)

    async def start_runtime(self, runtime: ManagedRuntime) -> None:
        await self.start_runtime_fn(runtime)

    async def shutdown_runtime(self, runtime: ManagedRuntime) -> None:
        await self.shutdown_runtime_fn(runtime)

    async def shutdown_all(self) -> None:
        # In-process controller does not keep additional global runtime state.
        return None
