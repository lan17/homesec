"""Shared runtime bootstrap helpers for storage and persistence wiring."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from homesec.interfaces import EventStore
from homesec.plugins.storage import load_storage_plugin
from homesec.repository import ClipRepository
from homesec.state import NoopEventStore, PostgresStateStore

if TYPE_CHECKING:
    from homesec.interfaces import StateStore, StorageBackend
    from homesec.models.config import Config, StateStoreConfig, StorageConfig

logger = logging.getLogger(__name__)


class _InitializableStateStore(Protocol):
    """State store contract needed during runtime bootstrap."""

    async def initialize(self) -> bool:
        """Initialize backing resources."""

    def create_event_store(self) -> EventStore:
        """Create associated event store."""


@dataclass(frozen=True, slots=True)
class RuntimePersistenceStack:
    """Shared storage and persistence components for runtime wiring."""

    storage: StorageBackend
    state_store: StateStore
    event_store: EventStore
    repository: ClipRepository


def _create_storage_backend(
    config: Config,
    *,
    loader: Callable[[StorageConfig], StorageBackend],
) -> StorageBackend:
    """Create the configured storage backend."""
    return loader(config.storage)


async def _create_state_store(
    config: StateStoreConfig,
    *,
    resolve_env: Callable[[str], str | None],
    state_store_factory: Callable[[str], _InitializableStateStore],
    missing_dsn_message: str,
) -> StateStore:
    """Create and initialize the configured state store."""
    dsn = config.dsn
    if config.dsn_env:
        dsn = resolve_env(config.dsn_env)
    if not dsn:
        raise RuntimeError(missing_dsn_message)

    store = state_store_factory(dsn)
    await store.initialize()
    return cast("StateStore", store)


def _create_event_store(
    state_store: StateStore,
    *,
    unavailable_warning: str,
) -> EventStore:
    """Create event store from state store and warn when events are unavailable."""
    event_store = state_store.create_event_store()
    if isinstance(event_store, NoopEventStore):
        logger.warning(unavailable_warning)
    return event_store


def _create_repository(
    *,
    state_store: StateStore,
    event_store: EventStore,
    config: Config,
) -> ClipRepository:
    """Create repository over the configured state and event stores."""
    return ClipRepository(
        state_store,
        event_store,
        retry=config.retry,
    )


async def build_runtime_persistence_stack(
    config: Config,
    *,
    resolve_env: Callable[[str], str | None],
    missing_dsn_message: str,
    event_store_unavailable_warning: str,
    storage_loader: Callable[[StorageConfig], StorageBackend] | None = None,
    state_store_factory: Callable[[str], _InitializableStateStore] | None = None,
) -> RuntimePersistenceStack:
    """Build shared storage and persistence components for a runtime."""
    loader = load_storage_plugin if storage_loader is None else storage_loader
    factory = PostgresStateStore if state_store_factory is None else state_store_factory

    storage = _create_storage_backend(config, loader=loader)
    state_store: StateStore | None = None
    try:
        state_store = await _create_state_store(
            config.state_store,
            resolve_env=resolve_env,
            state_store_factory=factory,
            missing_dsn_message=missing_dsn_message,
        )
        event_store = _create_event_store(
            state_store,
            unavailable_warning=event_store_unavailable_warning,
        )
        repository = _create_repository(
            state_store=state_store,
            event_store=event_store,
            config=config,
        )
        return RuntimePersistenceStack(
            storage=storage,
            state_store=state_store,
            event_store=event_store,
            repository=repository,
        )
    except Exception:
        if state_store is not None:
            await _safe_shutdown(state_store, label="state store")
        await _safe_shutdown(storage, label="storage backend")
        raise


async def _safe_shutdown(component: object, *, label: str) -> None:
    """Best-effort shutdown for partially built bootstrap components."""
    shutdown = getattr(component, "shutdown", None)
    if shutdown is None:
        return
    try:
        await shutdown()
    except Exception as exc:
        logger.warning("Failed to shut down partial %s during bootstrap cleanup: %s", label, exc)
