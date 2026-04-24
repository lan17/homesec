"""Tests for shared runtime bootstrap helpers."""

from __future__ import annotations

import pytest

from homesec.models.config import (
    AlertPolicyConfig,
    CameraConfig,
    CameraSourceConfig,
    Config,
    StateStoreConfig,
    StorageConfig,
)
from homesec.models.enums import VLMRunMode
from homesec.models.filter import FilterConfig
from homesec.models.vlm import VLMConfig
from homesec.runtime.bootstrap import build_runtime_persistence_stack
from homesec.state.postgres import NoopEventStore


def _make_config() -> Config:
    return Config(
        cameras=[
            CameraConfig(
                name="front",
                source=CameraSourceConfig(backend="local_folder", config={}),
            )
        ],
        storage=StorageConfig(backend="local", config={}),
        state_store=StateStoreConfig(dsn="postgresql://user:pass@localhost/homesec"),
        notifiers=[],
        filter=FilterConfig(backend="yolo", config={}),
        vlm=VLMConfig(backend="openai", run_mode=VLMRunMode.TRIGGER_ONLY, config={}),
        alert_policy=AlertPolicyConfig(backend="default", config={}),
    )


class _RecordingStorage:
    def __init__(self) -> None:
        self.shutdown_called = False

    async def shutdown(self, timeout: float | None = None) -> None:
        # Given: Bootstrap cleanup calls storage shutdown without depending on timeout
        _ = timeout
        self.shutdown_called = True


def _noop_event_store_factory(_store: object) -> NoopEventStore:
    return NoopEventStore()


@pytest.mark.asyncio
async def test_build_runtime_persistence_stack_requires_event_factory_with_custom_state() -> None:
    # Given: Custom state-store wiring without the matching event-store factory
    config = _make_config()
    storage_loaded = False

    class _CustomStateStore:
        async def initialize(self) -> bool:
            return True

    def _load_storage(_cfg: StorageConfig) -> _RecordingStorage:
        nonlocal storage_loaded
        storage_loaded = True
        return _RecordingStorage()

    # When/Then: Bootstrap rejects the partial override before acquiring resources
    with pytest.raises(RuntimeError, match="event_store_factory"):
        await build_runtime_persistence_stack(
            config,
            resolve_env=lambda env: env,
            missing_dsn_message="missing dsn",
            event_store_unavailable_warning="events unavailable",
            storage_loader=_load_storage,
            state_store_factory=lambda _dsn: _CustomStateStore(),
        )

    assert storage_loaded is False


@pytest.mark.asyncio
async def test_build_runtime_persistence_stack_shuts_down_storage_when_state_init_fails() -> None:
    # Given: A storage backend created before state-store initialization fails
    config = _make_config()
    storage = _RecordingStorage()

    class _FailingStateStore:
        def __init__(self, dsn: str) -> None:
            self.dsn = dsn

        async def initialize(self) -> bool:
            raise RuntimeError("boom")

        async def shutdown(self, timeout: float | None = None) -> None:
            _ = timeout

    # When: Building the shared runtime persistence stack
    with pytest.raises(RuntimeError, match="boom"):
        await build_runtime_persistence_stack(
            config,
            resolve_env=lambda env: env,
            missing_dsn_message="missing dsn",
            event_store_unavailable_warning="events unavailable",
            storage_loader=lambda _cfg: storage,
            state_store_factory=_FailingStateStore,
            event_store_factory=_noop_event_store_factory,
        )

    # Then: Storage is cleaned up even though the worker never captured it
    assert storage.shutdown_called is True


@pytest.mark.asyncio
async def test_build_runtime_persistence_stack_shuts_down_partial_state_when_event_store_fails() -> (
    None
):
    # Given: A state store initialized successfully but event-store creation raises
    config = _make_config()
    storage = _RecordingStorage()

    class _StateStoreWithFailingEvents:
        def __init__(self, dsn: str) -> None:
            self.dsn = dsn
            self.shutdown_called = False

        async def initialize(self) -> bool:
            return True

        async def shutdown(self, timeout: float | None = None) -> None:
            _ = timeout
            self.shutdown_called = True

    store = _StateStoreWithFailingEvents("unused")

    def _raise_event_store_failure(_store: object) -> NoopEventStore:
        raise RuntimeError("event store failed")

    # When: Building the shared runtime persistence stack
    with pytest.raises(RuntimeError, match="event store failed"):
        await build_runtime_persistence_stack(
            config,
            resolve_env=lambda env: env,
            missing_dsn_message="missing dsn",
            event_store_unavailable_warning="events unavailable",
            storage_loader=lambda _cfg: storage,
            state_store_factory=lambda _dsn: store,
            event_store_factory=_raise_event_store_failure,
        )

    # Then: Both initialized resources are shut down on partial-build failure
    assert store.shutdown_called is True
    assert storage.shutdown_called is True


@pytest.mark.asyncio
async def test_build_runtime_persistence_stack_preserves_primary_failure_when_cleanup_raises() -> (
    None
):
    # Given: Partial-bootstrap cleanup that fails while handling the primary bootstrap error
    config = _make_config()

    class _FailingStorage:
        async def shutdown(self, timeout: float | None = None) -> None:
            _ = timeout
            raise RuntimeError("cleanup failed")

    class _FailingStateStore:
        def __init__(self, dsn: str) -> None:
            self.dsn = dsn

        async def initialize(self) -> bool:
            raise RuntimeError("primary failure")

        async def shutdown(self, timeout: float | None = None) -> None:
            _ = timeout

    # When: Building the shared runtime persistence stack
    with pytest.raises(RuntimeError, match="primary failure"):
        await build_runtime_persistence_stack(
            config,
            resolve_env=lambda env: env,
            missing_dsn_message="missing dsn",
            event_store_unavailable_warning="events unavailable",
            storage_loader=lambda _cfg: _FailingStorage(),
            state_store_factory=_FailingStateStore,
            event_store_factory=_noop_event_store_factory,
        )

    # Then: Cleanup errors do not replace the original bootstrap failure


@pytest.mark.asyncio
async def test_build_runtime_persistence_stack_uses_noop_events_when_state_init_returns_false() -> (
    None
):
    # Given: Postgres initialization degrades without raising
    config = _make_config()
    storage = _RecordingStorage()

    class _UnavailableStateStore:
        def __init__(self, dsn: str) -> None:
            self.dsn = dsn

        async def initialize(self) -> bool:
            return False

        async def shutdown(self, timeout: float | None = None) -> None:
            _ = timeout

    store = _UnavailableStateStore(config.state_store.dsn or "")

    # When: Building the shared runtime persistence stack
    persistence = await build_runtime_persistence_stack(
        config,
        resolve_env=lambda env: env,
        missing_dsn_message="missing dsn",
        event_store_unavailable_warning="events unavailable",
        storage_loader=lambda _cfg: storage,
        state_store_factory=lambda _dsn: store,
        event_store_factory=_noop_event_store_factory,
    )

    # Then: Runtime can continue with best-effort/no-op event persistence
    assert persistence.state_store is store
    assert isinstance(persistence.event_store, NoopEventStore)
    assert storage.shutdown_called is False
