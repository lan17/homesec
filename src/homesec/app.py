"""Main application that wires all components together."""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from homesec.api import APIServer, create_app
from homesec.config import load_config, resolve_env_var, validate_config, validate_plugin_names
from homesec.config.loader import ConfigError, ConfigErrorCode
from homesec.config.manager import ConfigManager
from homesec.health import HealthServer
from homesec.interfaces import EventStore
from homesec.plugins.registry import PluginType, get_plugin_names
from homesec.plugins.storage import load_storage_plugin
from homesec.repository import ClipRepository
from homesec.runtime.controller import RuntimeController
from homesec.runtime.errors import RuntimeReloadConfigError
from homesec.runtime.manager import RuntimeManager
from homesec.runtime.models import (
    ManagedRuntime,
    RuntimeCameraStatus,
    RuntimeReloadRequest,
    RuntimeReloadResult,
    RuntimeState,
    RuntimeStatusSnapshot,
)
from homesec.runtime.subprocess_controller import (
    SubprocessRuntimeController,
    SubprocessRuntimeHandle,
)
from homesec.state import NoopEventStore, NoopStateStore, PostgresStateStore

if TYPE_CHECKING:
    from homesec.interfaces import (
        StateStore,
        StorageBackend,
    )
    from homesec.models.config import Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _CameraSourceSnapshot:
    """Minimal source health view exposed to API routes."""

    healthy: bool
    heartbeat: float | None

    def is_healthy(self) -> bool:
        return self.healthy

    def last_heartbeat(self) -> float | None:
        return self.heartbeat


class Application:
    """Main application that orchestrates all components.

    Handles component creation, lifecycle, and graceful shutdown.
    """

    def __init__(self, config_path: Path) -> None:
        """Initialize application with config file path.

        Args:
            config_path: Path to YAML config file
        """
        self._config_path = config_path
        self._config: Config | None = None

        # Components (created in _create_components)
        self._storage: StorageBackend | None = None
        self._state_store: StateStore = NoopStateStore()
        self._event_store: EventStore = NoopEventStore()
        self._repository: ClipRepository | None = None
        self._health_server: HealthServer | None = None
        self._api_server: APIServer | None = None
        self._runtime_manager: RuntimeManager | None = None
        self._runtime_heartbeat_stale_s = 10.0
        self._config_manager = ConfigManager(config_path)
        self._start_time: float | None = None

        # Shutdown state
        self._shutdown_event = asyncio.Event()
        self._shutdown_started = False

    async def run(self) -> None:
        """Run the application.

        Loads config, creates components, and runs until shutdown signal.
        """
        logger.info("Starting HomeSec application...")

        # Load config
        self._config = load_config(self._config_path)
        logger.info("Config loaded from %s", self._config_path)

        # Create components
        await self._create_components()

        # Set up signal handlers
        self._setup_signal_handlers()

        # Start health server
        if self._health_server:
            await self._health_server.start()

        if self._api_server:
            await self._api_server.start()

        self._start_time = time.time()
        logger.info("Application started. Waiting for clips...")

        # Wait for shutdown signal
        await self._shutdown_event.wait()

        # Graceful shutdown
        await self.shutdown()

    async def _create_components(self) -> None:
        """Create all components based on config."""
        config = self._require_config()

        # Discover plugins before validation
        from homesec.plugins import discover_all_plugins

        discover_all_plugins()

        # Validate config references and plugin names before instantiating components
        self._validate_config(config)

        # Create shell components that remain stable across runtime reloads.
        self._storage = self._create_storage(config)
        self._state_store = await self._create_state_store(config)
        self._event_store = self._create_event_store(self._state_store)
        self._repository = ClipRepository(
            self._state_store,
            self._event_store,
            retry=config.retry,
        )
        assert self._storage is not None

        health_cfg = config.health
        self._health_server = HealthServer(
            host=health_cfg.host,
            port=health_cfg.port,
        )
        self._health_server.set_components(
            state_store=self._state_store,
            storage=self._storage,
            notifier=None,
            sources=[],
            mqtt_is_critical=health_cfg.mqtt_is_critical,
        )

        # Create runtime manager and start the initial runtime.
        self._runtime_manager = RuntimeManager(
            self._create_runtime_controller(),
            on_runtime_activated=self._bind_active_runtime,
            on_runtime_cleared=self._clear_active_runtime,
        )
        await self._runtime_manager.start_initial_runtime(config)

        server_cfg = config.server
        if server_cfg.enabled:
            api_app = create_app(self)
            self._api_server = APIServer(
                app=api_app,
                host=server_cfg.host,
                port=server_cfg.port,
            )
        else:
            self._api_server = None

        logger.info("All components created")

    def _create_storage(self, config: Config) -> StorageBackend:
        """Create storage backend based on config."""
        return load_storage_plugin(config.storage)

    async def _create_state_store(self, config: Config) -> StateStore:
        """Create state store based on config."""
        state_cfg = config.state_store
        dsn = state_cfg.dsn
        if state_cfg.dsn_env:
            dsn = resolve_env_var(state_cfg.dsn_env)
        if not dsn:
            raise RuntimeError("Postgres DSN is required for state_store backend")
        store = PostgresStateStore(dsn)
        await store.initialize()
        return store

    def _create_event_store(self, state_store: StateStore) -> EventStore:
        """Create event store based on state store backend."""
        event_store = state_store.create_event_store()
        if isinstance(event_store, NoopEventStore):
            logger.warning(
                "Event store unavailable (NoopEventStore returned); events will be dropped"
            )
        return event_store

    def _create_runtime_controller(self) -> RuntimeController:
        controller = SubprocessRuntimeController()
        self._runtime_heartbeat_stale_s = controller.heartbeat_stale_s
        return controller

    def _bind_active_runtime(self, runtime: ManagedRuntime) -> None:
        """Bind activated runtime metadata to application state."""
        self._config = runtime.config

        if self._health_server is not None and self._storage is not None:
            self._health_server.set_components(
                state_store=self._state_store,
                storage=self._storage,
                notifier=None,
                sources=[],
                mqtt_is_critical=runtime.config.health.mqtt_is_critical,
            )

    def _clear_active_runtime(self) -> None:
        """Clear active runtime references from application accessors."""
        if self._health_server is not None and self._storage is not None:
            mqtt_is_critical = self._config.health.mqtt_is_critical if self._config else False
            self._health_server.set_components(
                state_store=self._state_store,
                storage=self._storage,
                notifier=None,
                sources=[],
                mqtt_is_critical=mqtt_is_critical,
            )

    def _require_runtime_manager(self) -> RuntimeManager:
        if self._runtime_manager is None:
            raise RuntimeError("Runtime manager not initialized")
        return self._runtime_manager

    def _active_subprocess_runtime(self) -> SubprocessRuntimeHandle | None:
        manager = self._runtime_manager
        if manager is None:
            return None
        runtime = manager.active_runtime
        if isinstance(runtime, SubprocessRuntimeHandle):
            return runtime
        return None

    def _camera_statuses(self) -> dict[str, RuntimeCameraStatus]:
        runtime = self._active_subprocess_runtime()
        if runtime is None:
            return {}
        statuses = dict(runtime.camera_statuses)
        if runtime.heartbeat_is_fresh(max_age_s=self._runtime_heartbeat_stale_s):
            return statuses

        return {
            camera: RuntimeCameraStatus(
                healthy=False,
                last_heartbeat=status.last_heartbeat,
            )
            for camera, status in statuses.items()
        }

    @staticmethod
    def _classify_reload_config_error(exc: ConfigError) -> tuple[int, str]:
        unprocessable_codes = {
            ConfigErrorCode.VALIDATION_FAILED,
            ConfigErrorCode.CAMERA_REFERENCES_INVALID,
            ConfigErrorCode.PLUGIN_NAMES_INVALID,
            ConfigErrorCode.PLUGIN_CONFIG_INVALID,
        }
        if exc.code in unprocessable_codes:
            return 422, exc.code.value
        return 400, exc.code.value

    def get_runtime_status(self) -> RuntimeStatusSnapshot:
        """Return current runtime-manager status."""
        snapshot = self._require_runtime_manager().get_status()
        if snapshot.state == RuntimeState.RELOADING:
            return snapshot

        runtime = self._active_subprocess_runtime()
        if runtime is None:
            return snapshot

        if not runtime.heartbeat_is_fresh(max_age_s=self._runtime_heartbeat_stale_s):
            snapshot.state = RuntimeState.FAILED
            if runtime.last_error is not None:
                snapshot.last_reload_error = runtime.last_error
            elif runtime.worker_exit_code is not None:
                snapshot.last_reload_error = (
                    f"runtime worker exited with code {runtime.worker_exit_code}"
                )
            else:
                snapshot.last_reload_error = "runtime worker heartbeat timed out"
        return snapshot

    async def request_runtime_reload(self) -> RuntimeReloadRequest:
        """Request a runtime reload using the latest persisted config."""
        try:
            config = await asyncio.to_thread(self._config_manager.get_config)
            self._validate_config(config)
        except ConfigError as exc:
            status_code, error_code = self._classify_reload_config_error(exc)
            raise RuntimeReloadConfigError(
                str(exc),
                status_code=status_code,
                error_code=error_code,
            ) from exc
        return self._require_runtime_manager().request_reload(config)

    async def wait_for_runtime_reload(self) -> RuntimeReloadResult | None:
        """Wait for the in-flight runtime reload (if any)."""
        return await self._require_runtime_manager().wait_for_reload()

    def _require_config(self) -> Config:
        if self._config is None:
            raise RuntimeError("Config not loaded")
        return self._config

    def _validate_config(self, config: Config) -> None:
        validate_plugin_names(
            config,
            valid_filters=get_plugin_names(PluginType.FILTER),
            valid_vlms=get_plugin_names(PluginType.ANALYZER),
            valid_storage=get_plugin_names(PluginType.STORAGE),
            valid_notifiers=get_plugin_names(PluginType.NOTIFIER),
            valid_alert_policies=get_plugin_names(PluginType.ALERT_POLICY),
            valid_sources=get_plugin_names(PluginType.SOURCE),
        )
        validate_config(config)

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_signal, sig)

    def _handle_signal(self, sig: signal.Signals) -> None:
        """Handle shutdown signal."""
        if self._shutdown_started:
            logger.warning("Shutdown already in progress, ignoring signal")
            return

        logger.info("Received signal %s, initiating shutdown...", sig.name)
        self._shutdown_started = True
        self._shutdown_event.set()

    async def shutdown(self) -> None:
        """Graceful shutdown of all components."""
        logger.info("Shutting down application...")

        # Stop API server first to prevent new requests during shutdown.
        if self._api_server:
            await self._api_server.stop()

        # Stop runtime manager (sources/pipeline/plugins).
        if self._runtime_manager:
            await self._runtime_manager.shutdown()

        # Stop health server
        if self._health_server:
            await self._health_server.stop()

        # Close state store
        if self._state_store:
            await self._state_store.shutdown()

        # Close storage
        if self._storage:
            await self._storage.shutdown()

        logger.info("Application shutdown complete")

    @property
    def config(self) -> Config:
        return self._require_config()

    @property
    def bootstrap_mode(self) -> bool:
        return False

    @property
    def repository(self) -> ClipRepository:
        if self._repository is None:
            raise RuntimeError("Repository not initialized")
        return self._repository

    @property
    def storage(self) -> StorageBackend:
        if self._storage is None:
            raise RuntimeError("Storage not initialized")
        return self._storage

    @property
    def sources(self) -> list[_CameraSourceSnapshot]:
        return [
            _CameraSourceSnapshot(
                healthy=status.healthy,
                heartbeat=status.last_heartbeat,
            )
            for status in self._camera_statuses().values()
        ]

    @property
    def config_manager(self) -> ConfigManager:
        return self._config_manager

    @property
    def pipeline_running(self) -> bool:
        if self._shutdown_event.is_set():
            return False
        runtime = self._active_subprocess_runtime()
        if runtime is None:
            return False
        return runtime.heartbeat_is_fresh(max_age_s=self._runtime_heartbeat_stale_s)

    @property
    def uptime_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def get_source(self, camera_name: str) -> _CameraSourceSnapshot | None:
        status = self._camera_statuses().get(camera_name)
        if status is None:
            return None
        return _CameraSourceSnapshot(
            healthy=status.healthy,
            heartbeat=status.last_heartbeat,
        )
