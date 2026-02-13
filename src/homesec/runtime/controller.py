"""Runtime controller abstraction and in-process implementation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from homesec.models.config import Config
    from homesec.runtime.models import RuntimeBundle


class RuntimeController(Protocol):
    """Build/start/shutdown contract for a runtime implementation."""

    async def build_candidate(self, config: Config, generation: int) -> RuntimeBundle:
        """Build a candidate runtime bundle."""
        ...

    async def start_runtime(self, runtime: RuntimeBundle) -> None:
        """Start a runtime bundle and run startup preflight."""
        ...

    async def shutdown_runtime(self, runtime: RuntimeBundle) -> None:
        """Gracefully stop and clean up a runtime bundle."""
        ...


@dataclass(slots=True)
class InProcessRuntimeController(RuntimeController):
    """In-process runtime controller backed by callback functions."""

    build_candidate_fn: Callable[[Config, int], Awaitable[RuntimeBundle]]
    start_runtime_fn: Callable[[RuntimeBundle], Awaitable[None]]
    shutdown_runtime_fn: Callable[[RuntimeBundle], Awaitable[None]]

    async def build_candidate(self, config: Config, generation: int) -> RuntimeBundle:
        return await self.build_candidate_fn(config, generation)

    async def start_runtime(self, runtime: RuntimeBundle) -> None:
        await self.start_runtime_fn(runtime)

    async def shutdown_runtime(self, runtime: RuntimeBundle) -> None:
        await self.shutdown_runtime_fn(runtime)
