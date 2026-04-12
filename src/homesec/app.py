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
from homesec.models.config import FastAPIServerConfig
from homesec.plugins.registry import PluginType, get_plugin_names
from homesec.runtime.bootstrap import (
    RuntimePersistenceStack,
    build_runtime_persistence_stack,
)
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
from homesec.state import NoopEventStore, NoopStateStore

if TYPE_CHECKING:
    from homesec.interfaces import EventStore, StateStore, StorageBackend
    from homesec.models.config import Config
    from homesec.repository import ClipRepository

logger = logging.getLogger(__name__)
RESTART_EXIT_CODE = 42


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

    def __init__(
        self,
        config_path: Path,
        *,
        bootstrap_host: str = "0.0.0.0",
        bootstrap_port: int = 8081,
    ) -> None:
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
        self._api_server: APIServer | None = None
        self._runtime_manager: RuntimeManager | None = None
        self._runtime_heartbeat_stale_s = 10.0
        self._config_manager = ConfigManager(config_path)
        self._start_time: float | None = None
        self._bootstrap = False
        self._bootstrap_server_config = FastAPIServerConfig(
            host=bootstrap_host,
            port=bootstrap_port,
            enabled=True,
        )

        # Shutdown state
        self._shutdown_event = asyncio.Event()
        self._shutdown_started = False
        self._restart_requested = False
        self._setup_test_connection_lock = asyncio.Lock()

    async def run(self) -> None:
        """Run the application.

        Loads config, creates components, and runs until shutdown signal.
        """
        logger.info("Starting HomeSec application...")

        if not self._config_path.exists():
            await self._run_bootstrap()
            return

        # Load config
        self._config = load_config(self._config_path)
        logger.info("Config loaded from %s", self._config_path)

        # Create components
        await self._create_components()

        # Set up signal handlers
        self._setup_signal_handlers()

        if self._api_server:
            await self._api_server.start()

        self._start_time = time.time()
        logger.info("Application started. Waiting for clips...")

        # Wait for shutdown signal
        await self._shutdown_event.wait()

        # Graceful shutdown
        await self.shutdown()

    async def _run_bootstrap(self) -> None:
        """Start in bootstrap mode with API/UI only."""
        logger.info(
            "Config file not found at %s. Starting in bootstrap mode.",
            self._config_path,
        )
        self._bootstrap = True

        # Keep plugin metadata routes available during onboarding.
        from homesec.plugins import discover_all_plugins

        try:
            discover_all_plugins()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Plugin discovery failed in bootstrap mode; continuing with setup APIs only: %s",
                exc,
                exc_info=True,
            )

        server_cfg = self.server_config
        self._api_server = APIServer(
            app=create_app(self),
            host=server_cfg.host,
            port=server_cfg.port,
        )

        self._setup_signal_handlers()
        try:
            await self._api_server.start()

            self._start_time = time.time()
            logger.info(
                "Bootstrap mode active at http://%s:%d",
                server_cfg.host,
                server_cfg.port,
            )

            await self._shutdown_event.wait()
        finally:
            await self.shutdown()

    async def _create_components(self, *, configure_api_server: bool = True) -> None:
        """Create all components based on config."""
        config = self._require_config()

        # Discover plugins before validation
        from homesec.plugins import discover_all_plugins

        discover_all_plugins()

        # Validate config references and plugin names before instantiating components
        self._validate_config(config)

        # Build components locally first so partial failures do not leak mutable app state.
        persistence = await self._build_runtime_persistence_stack(config)
        runtime_manager = RuntimeManager(
            self._create_runtime_controller(),
            on_runtime_activated=self._bind_active_runtime,
        )
        try:
            await runtime_manager.start_initial_runtime(config)
        except Exception:
            await runtime_manager.shutdown()
            await persistence.state_store.shutdown()
            await persistence.storage.shutdown()
            raise

        api_server = self._api_server
        if configure_api_server:
            server_cfg = config.server
            if server_cfg.enabled:
                api_app = create_app(self)
                api_server = APIServer(
                    app=api_app,
                    host=server_cfg.host,
                    port=server_cfg.port,
                )
            else:
                api_server = None

        self._storage = persistence.storage
        self._state_store = persistence.state_store
        self._event_store = persistence.event_store
        self._repository = persistence.repository
        self._runtime_manager = runtime_manager
        self._api_server = api_server

        logger.info("All components created")

    async def activate_setup_config(self, config: Config) -> None:
        """Apply setup-finalized config and start runtime without restarting FastAPI."""
        if not self._bootstrap:
            raise RuntimeError("Setup activation is only supported in bootstrap mode")

        self._config = config
        try:
            await self._create_components(configure_api_server=False)
        except Exception:
            self._config = None
            raise

        self._bootstrap = False
        logger.info("Setup finalized; runtime activated in-process without API restart")

    async def _build_runtime_persistence_stack(self, config: Config) -> RuntimePersistenceStack:
        """Build shared storage and persistence components for the app runtime."""
        return await build_runtime_persistence_stack(
            config,
            resolve_env=resolve_env_var,
            missing_dsn_message="Postgres DSN is required for state_store backend",
            event_store_unavailable_warning=(
                "Event store unavailable (NoopEventStore returned); events will be dropped"
            ),
        )

    def _create_runtime_controller(self) -> RuntimeController:
        controller = SubprocessRuntimeController()
        self._runtime_heartbeat_stale_s = controller.heartbeat_stale_s
        return controller

    def _bind_active_runtime(self, runtime: ManagedRuntime) -> None:
        """Bind activated runtime metadata to application state."""
        self._config = runtime.config

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

    def request_restart(self) -> None:
        """Request graceful shutdown so an external supervisor can restart the process."""
        if self._shutdown_started:
            return
        if self._restart_requested:
            return
        logger.info("Restart requested by setup finalize endpoint")
        self._restart_requested = True
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
        return self._bootstrap

    @property
    def server_config(self) -> FastAPIServerConfig:
        if self._config is not None:
            return self._config.server
        return self._bootstrap_server_config

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

    @property
    def restart_requested(self) -> bool:
        return self._restart_requested

    @property
    def restart_exit_code(self) -> int:
        return RESTART_EXIT_CODE

    @property
    def setup_test_connection_lock(self) -> asyncio.Lock:
        return self._setup_test_connection_lock

    def get_source(self, camera_name: str) -> _CameraSourceSnapshot | None:
        status = self._camera_statuses().get(camera_name)
        if status is None:
            return None
        return _CameraSourceSnapshot(
            healthy=status.healthy,
            heartbeat=status.last_heartbeat,
        )
