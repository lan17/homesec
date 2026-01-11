"""HTTP health check endpoint."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from aiohttp import web

if TYPE_CHECKING:
    from homesec.interfaces import ClipSource, Notifier, StateStore, StorageBackend

logger = logging.getLogger(__name__)


class HealthServer:
    """HTTP server for health checks.
    
    Provides /health endpoint returning component status.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
    ) -> None:
        """Initialize health server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
        """
        self.host = host
        self.port = port
        
        # Components to check (set via set_components)
        self._state_store: StateStore | None = None
        self._storage: StorageBackend | None = None
        self._notifier: Notifier | None = None
        self._sources: list[ClipSource] = []
        self._mqtt_is_critical = False
        
        # Server state
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        
        logger.info("HealthServer initialized: %s:%d", host, port)

    def set_components(
        self,
        *,
        state_store: StateStore | None = None,
        storage: StorageBackend | None = None,
        notifier: Notifier | None = None,
        sources: list[ClipSource] | None = None,
        mqtt_is_critical: bool = False,
    ) -> None:
        """Set components to check.
        
        Args:
            state_store: State store to ping
            storage: Storage backend to ping
            notifier: Notifier to ping
            sources: List of clip sources to check
            mqtt_is_critical: Whether MQTT failure is critical (unhealthy vs degraded)
        """
        self._state_store = state_store
        self._storage = storage
        self._notifier = notifier
        self._sources = sources or []
        self._mqtt_is_critical = mqtt_is_critical

    async def start(self) -> None:
        """Start HTTP server."""
        self._app = web.Application()
        self._app.router.add_get("/health", self._health_handler)
        
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        
        logger.info("HealthServer started: http://%s:%d/health", self.host, self.port)

    async def stop(self) -> None:
        """Stop HTTP server."""
        if self._runner:
            await self._runner.cleanup()
        
        self._app = None
        self._runner = None
        self._site = None
        
        logger.info("HealthServer stopped")

    async def _health_handler(self, request: web.Request) -> web.Response:
        """Handle GET /health request."""
        health_data = await self.compute_health()
        return web.json_response(health_data)

    async def compute_health(self) -> dict[str, Any]:
        """Compute health status and return JSON data.

        Returns:
            Health data dict with status, checks, warnings
        """
        sources = self._get_source_health()
        # Run component checks
        checks = {
            "db": await self._check_db(),
            "storage": await self._check_storage(),
            "mqtt": await self._check_mqtt(),
            "sources": self._check_sources(),
        }

        # Compute overall status
        status = self._compute_status(checks)

        # Generate warnings
        warnings = self._compute_warnings()

        return {
            "status": status,
            "checks": checks,
            "warnings": warnings,
            "sources": sources,
        }

    async def _check_db(self) -> bool:
        """Check if state store is healthy."""
        return await self._check_component(self._state_store, "State store")

    async def _check_storage(self) -> bool:
        """Check if storage backend is healthy."""
        return await self._check_component(self._storage, "Storage")

    async def _check_mqtt(self) -> bool:
        """Check if notifier is healthy."""
        return await self._check_component(self._notifier, "Notifier")

    def _check_sources(self) -> bool:
        """Check if all clip sources are healthy."""
        if not self._sources:
            return True  # No sources configured
        
        # All sources must be healthy
        return all(source.is_healthy() for source in self._sources)

    def _get_source_health(self) -> list[dict[str, object]]:
        """Return per-source health detail."""
        if not self._sources:
            return []

        current_time = time.monotonic()
        details: list[dict[str, object]] = []
        for source in self._sources:
            camera_name = getattr(source, "camera_name", "unknown")
            last_heartbeat = source.last_heartbeat()
            details.append({
                "name": camera_name,
                "healthy": source.is_healthy(),
                "last_heartbeat": last_heartbeat,
                "last_heartbeat_age_s": round(current_time - last_heartbeat, 3),
            })
        return details

    async def _check_component(
        self,
        component: StateStore | StorageBackend | Notifier | None,
        label: str,
    ) -> bool:
        if component is None:
            return True

        try:
            return await component.ping()
        except Exception as e:
            logger.warning("%s health check failed: %s", label, e, exc_info=True)
            return False

    def _compute_status(self, checks: dict[str, bool]) -> str:
        """Compute overall health status.
        
        Args:
            checks: Dict of component check results
            
        Returns:
            "healthy", "degraded", or "unhealthy"
        """
        # Critical checks (unhealthy if fail)
        if not checks["sources"]:
            return "unhealthy"
        
        if not checks["storage"]:
            return "unhealthy"
        
        # MQTT can be critical (configurable)
        if not checks["mqtt"] and self._mqtt_is_critical:
            return "unhealthy"
        
        # Non-critical checks (degraded if fail)
        if not checks["db"] or not checks["mqtt"]:
            return "degraded"
        
        return "healthy"

    def _compute_warnings(self) -> list[str]:
        """Generate warnings for stale heartbeats.
        
        Returns:
            List of warning messages
        """
        warnings: list[str] = []
        
        # Check source heartbeats (warn if > 2 minutes)
        current_time = time.monotonic()
        for source in self._sources:
            heartbeat_age = current_time - source.last_heartbeat()
            if heartbeat_age > 120:  # 2 minutes
                camera_name = getattr(source, "camera_name", "unknown")
                warnings.append(f"source_{camera_name}_heartbeat_stale")
        
        return warnings
