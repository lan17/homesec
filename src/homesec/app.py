"""Main application that wires all components together."""

from __future__ import annotations

import asyncio
import logging
import signal
from pathlib import Path
from typing import TYPE_CHECKING, Callable, cast

from pydantic import BaseModel

from homesec.config import (
    load_config,
    resolve_env_var,
    validate_camera_references,
    validate_plugin_names,
)
from homesec.health import HealthServer
from homesec.pipeline import ClipPipeline
from homesec.plugins.analyzers import VLM_REGISTRY, load_vlm_plugin
from homesec.plugins.alert_policies import ALERT_POLICY_REGISTRY
from homesec.plugins.filters import FILTER_REGISTRY, load_filter_plugin
from homesec.plugins.notifiers import (
    NOTIFIER_REGISTRY,
    MultiplexNotifier,
    NotifierEntry,
    NotifierPlugin,
)
from homesec.plugins.storage import STORAGE_REGISTRY, create_storage
from homesec.sources import FtpSource, LocalFolderSource, RTSPSource
from homesec.models.config import NotifierConfig
from homesec.models.source import FtpSourceConfig, LocalFolderSourceConfig, RTSPSourceConfig
from homesec.interfaces import EventStore
from homesec.repository import ClipRepository
from homesec.state import NoopEventStore, NoopStateStore, PostgresStateStore

if TYPE_CHECKING:
    from homesec.models.config import Config
    from homesec.interfaces import (
        AlertPolicy,
        ClipSource,
        Notifier,
        ObjectFilter,
        StateStore,
        StorageBackend,
        VLMAnalyzer,
    )

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
        filter_plugin = load_filter_plugin(config.filter)
        vlm_plugin = load_vlm_plugin(config.vlm)
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
        return create_storage(config.storage)

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
        create_event_store = getattr(state_store, "create_event_store", None)
        if callable(create_event_store):
            event_store = cast(Callable[[], EventStore], create_event_store)()
            if isinstance(event_store, NoopEventStore):
                logger.warning("Event store unavailable; events will be dropped")
            return event_store
        logger.warning("Unsupported state store for events; events will be dropped")
        return NoopEventStore()

    def _create_notifier(self, config: Config) -> Notifier:
        """Create notifier(s) based on config."""
        entries: list[NotifierEntry] = []
        for index, notifier_cfg in enumerate(config.notifiers):
            plugin, validated_cfg = self._validate_notifier_config(notifier_cfg)
            if not notifier_cfg.enabled:
                continue
            notifier = plugin.factory(validated_cfg)
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

        tasks = [
            asyncio.create_task(entry.notifier.ping())
            for entry in self._notifier_entries
        ]
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

    def _validate_notifier_config(
        self, notifier_cfg: NotifierConfig
    ) -> tuple[NotifierPlugin, BaseModel]:
        """Validate notifier config and return the plugin and validated config."""
        plugin = NOTIFIER_REGISTRY.get(notifier_cfg.backend)
        if plugin is None:
            raise RuntimeError(f"Unsupported notifier backend: {notifier_cfg.backend}")
        validated_cfg = plugin.config_model.model_validate(notifier_cfg.config)
        return plugin, validated_cfg

    def _create_alert_policy(self, config: Config) -> AlertPolicy:
        policy_cfg = config.alert_policy

        # Use noop backend when alert policy is disabled
        backend = "noop" if not policy_cfg.enabled else policy_cfg.backend

        plugin = ALERT_POLICY_REGISTRY.get(backend)
        if plugin is None:
            raise RuntimeError(f"Unsupported alert policy backend: {backend}")

        # Always validate to ensure proper BaseModel contract
        if policy_cfg.enabled:
            settings = plugin.config_model.model_validate(policy_cfg.config)
        else:
            # Noop uses empty BaseModel, validate empty dict to get BaseModel instance
            settings = plugin.config_model.model_validate({})

        return plugin.factory(settings, config.per_camera_alert, config.vlm.trigger_classes)

    def _create_sources(self, config: Config) -> list[ClipSource]:
        """Create clip sources based on config."""
        sources: list[ClipSource] = []

        for camera in config.cameras:
            source_cfg = camera.source
            match source_cfg.type:
                case "local_folder":
                    local_cfg = source_cfg.config
                    assert isinstance(local_cfg, LocalFolderSourceConfig)
                    sources.append(
                        LocalFolderSource(
                            local_cfg,
                            camera_name=camera.name,
                            state_store=self._state_store,
                        )
                    )

                case "rtsp":
                    rtsp_cfg = source_cfg.config
                    assert isinstance(rtsp_cfg, RTSPSourceConfig)
                    sources.append(RTSPSource(rtsp_cfg, camera_name=camera.name))

                case "ftp":
                    ftp_cfg = source_cfg.config
                    assert isinstance(ftp_cfg, FtpSourceConfig)
                    sources.append(FtpSource(ftp_cfg, camera_name=camera.name))

                case _:
                    raise RuntimeError(f"Unsupported source type: {source_cfg.type}")

        return sources

    def _require_config(self) -> Config:
        if self._config is None:
            raise RuntimeError("Config not loaded")
        return self._config

    def _validate_config(self, config: Config) -> None:
        validate_camera_references(config)
        validate_plugin_names(
            config,
            valid_filters=sorted(FILTER_REGISTRY.keys()),
            valid_vlms=sorted(VLM_REGISTRY.keys()),
            valid_storage=sorted(STORAGE_REGISTRY.keys()),
            valid_notifiers=sorted(NOTIFIER_REGISTRY.keys()),
            valid_alert_policies=sorted(ALERT_POLICY_REGISTRY.keys()),
        )

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
