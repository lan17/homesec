"""Main application that wires all components together."""

from __future__ import annotations

import asyncio
import logging
import signal
from pathlib import Path
from typing import TYPE_CHECKING

from homesec.config import (
    load_config,
    resolve_env_var,
    validate_camera_references,
    validate_plugin_names,
    validate_plugin_configs,
)
from homesec.health import HealthServer
from homesec.interfaces import EventStore
from homesec.notifiers.multiplex import MultiplexNotifier, NotifierEntry
from homesec.pipeline import ClipPipeline
from homesec.plugins.alert_policies import load_alert_policy
from homesec.plugins.analyzers import load_analyzer
from homesec.plugins.filters import load_filter
from homesec.plugins.notifiers import load_notifier_plugin
from homesec.plugins.registry import PluginType, get_plugin_names
from homesec.plugins.sources import load_source_plugin
from homesec.plugins.storage import load_storage_plugin
from homesec.repository import ClipRepository
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
        self._pipeline: ClipPipeline | None = None
        self._health_server: HealthServer | None = None

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

        # Start sources
        for source in self._sources:
            await source.start()

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

        # Create storage backend
        self._storage = self._create_storage(config)

        # Create state store
        self._state_store = await self._create_state_store(config)
        self._event_store = self._create_event_store(self._state_store)
        self._repository = ClipRepository(
            self._state_store,
            self._event_store,
            retry=config.retry,
        )
        assert self._storage is not None
        assert self._repository is not None

        # Create notifier
        self._notifier = self._create_notifier(config)
        assert self._notifier is not None
        await self._log_notifier_health()

        # Create filter, VLM, and alert policy plugins
        filter_plugin = load_filter(config.filter)
        vlm_plugin = load_analyzer(config.vlm)
        self._filter = filter_plugin
        self._vlm = vlm_plugin
        alert_policy = self._create_alert_policy(config)

        # Create pipeline
        self._pipeline = ClipPipeline(
            config=config,
            storage=self._storage,
            repository=self._repository,
            filter_plugin=filter_plugin,
            vlm_plugin=vlm_plugin,
            notifier=self._notifier,
            alert_policy=alert_policy,
            notifier_entries=self._notifier_entries,
        )
        # Set event loop for thread-safe callbacks from sources
        self._pipeline.set_event_loop(asyncio.get_running_loop())

        # Create sources and register callback
        self._sources = self._create_sources(config)
        for source in self._sources:
            source.register_callback(self._pipeline.on_new_clip)

        # Create health server
        health_cfg = config.health
        self._health_server = HealthServer(
            host=health_cfg.host,
            port=health_cfg.port,
        )
        self._health_server.set_components(
            state_store=self._state_store,
            storage=self._storage,
            notifier=self._notifier,
            sources=self._sources,
            mqtt_is_critical=health_cfg.mqtt_is_critical,
        )

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

    def _create_notifier(self, config: Config) -> Notifier:
        """Create notifier(s) based on config using plugin registry."""
        entries: list[NotifierEntry] = []
        for index, notifier_cfg in enumerate(config.notifiers):
            if not notifier_cfg.enabled:
                continue
            notifier = load_notifier_plugin(notifier_cfg.backend, notifier_cfg.config)
            name = f"{notifier_cfg.backend}[{index}]"
            entries.append(NotifierEntry(name=name, notifier=notifier))

        self._notifier_entries = entries
        if not entries:
            raise RuntimeError("No enabled notifiers configured")
        if len(entries) == 1:
            return entries[0].notifier
        return MultiplexNotifier(entries)

    async def _log_notifier_health(self) -> None:
        if not self._notifier_entries:
            return

        tasks = [asyncio.create_task(entry.notifier.ping()) for entry in self._notifier_entries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for entry, result in zip(self._notifier_entries, results, strict=True):
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

    def _create_sources(self, config: Config) -> list[ClipSource]:
        """Create clip sources based on config using plugin registry."""
        sources: list[ClipSource] = []

        for camera in config.cameras:
            source_cfg = camera.source
            source = load_source_plugin(
                source_backend=source_cfg.backend,
                config=source_cfg.config,
                camera_name=camera.name,
            )
            sources.append(source)

        return sources

    def _require_config(self) -> Config:
        if self._config is None:
            raise RuntimeError("Config not loaded")
        return self._config

    def _validate_config(self, config: Config) -> None:
        validate_camera_references(config)
        validate_plugin_names(
            config,
            valid_filters=get_plugin_names(PluginType.FILTER),
            valid_vlms=get_plugin_names(PluginType.ANALYZER),
            valid_storage=get_plugin_names(PluginType.STORAGE),
            valid_notifiers=get_plugin_names(PluginType.NOTIFIER),
            valid_alert_policies=get_plugin_names(PluginType.ALERT_POLICY),
            valid_sources=get_plugin_names(PluginType.SOURCE),
        )
        validate_plugin_configs(config)

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

        # Stop sources first
        if self._sources:
            await asyncio.gather(
                *(source.shutdown() for source in self._sources),
                return_exceptions=True,
            )

        # Shutdown pipeline (waits for in-flight clips)
        if self._pipeline:
            await self._pipeline.shutdown()

        # Stop health server
        if self._health_server:
            await self._health_server.stop()

        # Close filter and VLM plugins
        if self._filter:
            await self._filter.shutdown()
        if self._vlm:
            await self._vlm.shutdown()

        # Close state store
        if self._state_store:
            await self._state_store.shutdown()

        # Close storage
        if self._storage:
            await self._storage.shutdown()

        # Close notifier
        if self._notifier:
            await self._notifier.shutdown()

        logger.info("Application shutdown complete")
