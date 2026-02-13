"""Main application that wires all components together."""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from pathlib import Path
from typing import TYPE_CHECKING

from homesec.api import APIServer, create_app
from homesec.config import load_config, resolve_env_var, validate_config, validate_plugin_names
from homesec.config.loader import ConfigError, ConfigErrorCode
from homesec.config.manager import ConfigManager
from homesec.health import HealthServer
from homesec.interfaces import EventStore
from homesec.notifiers.multiplex import MultiplexNotifier, NotifierEntry
from homesec.pipeline import ClipPipeline
from homesec.plugins.alert_policies import load_alert_policy
from homesec.plugins.notifiers import load_notifier_plugin
from homesec.plugins.registry import PluginType, get_plugin_names
from homesec.plugins.sources import load_source_plugin
from homesec.plugins.storage import load_storage_plugin
from homesec.repository import ClipRepository
from homesec.runtime.assembly import RuntimeAssembler
from homesec.runtime.controller import InProcessRuntimeController
from homesec.runtime.errors import RuntimeReloadConfigError
from homesec.runtime.manager import RuntimeManager
from homesec.runtime.models import (
    RuntimeBundle,
    RuntimeReloadRequest,
    RuntimeReloadResult,
    RuntimeStatusSnapshot,
)
from homesec.state import NoopEventStore, NoopStateStore, PostgresStateStore

if TYPE_CHECKING:
    from homesec.interfaces import (
        AlertPolicy,
        ClipSource,
        Notifier,
        ObjectFilter,
        StateStore,
        StorageBackend,
        VLMAnalyzer,
    )
    from homesec.models.config import Config

logger = logging.getLogger(__name__)


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
        self._notifier: Notifier | None = None
        self._notifier_entries: list[NotifierEntry] = []
        self._filter: ObjectFilter | None = None
        self._vlm: VLMAnalyzer | None = None
        self._sources: list[ClipSource] = []
        self._sources_by_camera: dict[str, ClipSource] = {}
        self._pipeline: ClipPipeline | None = None
        self._health_server: HealthServer | None = None
        self._api_server: APIServer | None = None
        self._runtime_assembler: RuntimeAssembler | None = None
        self._runtime_manager: RuntimeManager | None = None
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
        assert self._repository is not None
        self._runtime_assembler = RuntimeAssembler(
            storage=self._storage,
            repository=self._repository,
            notifier_factory=self._create_notifier,
            notifier_health_logger=self._log_notifier_health,
            alert_policy_factory=self._create_alert_policy,
            source_factory=self._create_sources,
        )

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
            InProcessRuntimeController(
                build_candidate_fn=self._build_runtime_bundle,
                start_runtime_fn=self._start_runtime_bundle,
                shutdown_runtime_fn=self._shutdown_runtime_bundle,
            ),
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

    def _create_notifier(self, config: Config) -> tuple[Notifier, list[NotifierEntry]]:
        """Create notifier(s) based on config using plugin registry."""
        entries: list[NotifierEntry] = []
        for index, notifier_cfg in enumerate(config.notifiers):
            if not notifier_cfg.enabled:
                continue
            notifier = load_notifier_plugin(notifier_cfg.backend, notifier_cfg.config)
            name = f"{notifier_cfg.backend}[{index}]"
            entries.append(NotifierEntry(name=name, notifier=notifier))

        if not entries:
            raise RuntimeError("No enabled notifiers configured")
        if len(entries) == 1:
            return entries[0].notifier, entries
        return MultiplexNotifier(entries), entries

    async def _log_notifier_health(self, entries: list[NotifierEntry]) -> None:
        if not entries:
            return

        tasks = [asyncio.create_task(entry.notifier.ping()) for entry in entries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for entry, result in zip(entries, results, strict=True):
            match result:
                case bool() as ok:
                    if ok:
                        logger.info("Notifier reachable at startup: %s", entry.name)
                    else:
                        logger.error("Notifier unreachable at startup: %s", entry.name)
                case BaseException() as err:
                    logger.error(
                        "Notifier ping failed at startup: %s error=%s",
                        entry.name,
                        err,
                        exc_info=err,
                    )

    def _create_alert_policy(self, config: Config) -> AlertPolicy:
        """Create alert policy using the plugin registry."""
        return load_alert_policy(
            config.alert_policy,
            trigger_classes=config.vlm.trigger_classes,
        )

    def _create_sources(self, config: Config) -> tuple[list[ClipSource], dict[str, ClipSource]]:
        """Create clip sources based on config using plugin registry."""
        sources: list[ClipSource] = []
        sources_by_camera: dict[str, ClipSource] = {}

        for camera in config.cameras:
            if not camera.enabled:
                continue
            source_cfg = camera.source
            source = load_source_plugin(
                source_backend=source_cfg.backend,
                config=source_cfg.config,
                camera_name=camera.name,
            )
            sources.append(source)
            sources_by_camera[camera.name] = source

        return sources, sources_by_camera

    async def _build_runtime_bundle(self, config: Config, generation: int) -> RuntimeBundle:
        """Build restartable runtime components for a generation."""
        return await self._require_runtime_assembler().build_bundle(config, generation)

    async def _start_runtime_bundle(self, runtime: RuntimeBundle) -> None:
        """Start runtime sources and run startup preflight."""
        await self._require_runtime_assembler().start_bundle(runtime)

    async def _shutdown_runtime_bundle(self, runtime: RuntimeBundle) -> None:
        """Gracefully stop a runtime bundle."""
        await self._require_runtime_assembler().shutdown_bundle(runtime)

    def _bind_active_runtime(self, runtime: RuntimeBundle) -> None:
        """Bind an activated runtime bundle to application accessors."""
        self._config = runtime.config
        self._notifier = runtime.notifier
        self._notifier_entries = list(runtime.notifier_entries)
        self._filter = runtime.filter_plugin
        self._vlm = runtime.vlm_plugin
        self._sources = list(runtime.sources)
        self._sources_by_camera = dict(runtime.sources_by_camera)
        self._pipeline = runtime.pipeline

        if self._health_server is not None and self._storage is not None:
            self._health_server.set_components(
                state_store=self._state_store,
                storage=self._storage,
                notifier=runtime.notifier,
                sources=self._sources,
                mqtt_is_critical=runtime.config.health.mqtt_is_critical,
            )

    def _clear_active_runtime(self) -> None:
        """Clear active runtime references from application accessors."""
        self._notifier = None
        self._notifier_entries = []
        self._filter = None
        self._vlm = None
        self._sources = []
        self._sources_by_camera = {}
        self._pipeline = None

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

    def _require_runtime_assembler(self) -> RuntimeAssembler:
        if self._runtime_assembler is None:
            raise RuntimeError("Runtime assembler not initialized")
        return self._runtime_assembler

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
        return self._require_runtime_manager().get_status()

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
    def sources(self) -> list[ClipSource]:
        return list(self._sources)

    @property
    def config_manager(self) -> ConfigManager:
        return self._config_manager

    @property
    def pipeline_running(self) -> bool:
        return self._pipeline is not None and not self._shutdown_event.is_set()

    @property
    def uptime_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def get_source(self, camera_name: str) -> ClipSource | None:
        return self._sources_by_camera.get(camera_name)
