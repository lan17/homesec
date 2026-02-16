"""In-process runtime bundle assembly and lifecycle helpers."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol

from homesec.notifiers.multiplex import NotifierEntry
from homesec.pipeline import ClipPipeline
from homesec.plugins.analyzers import load_analyzer
from homesec.plugins.filters import load_filter
from homesec.repository import ClipRepository
from homesec.runtime.models import RuntimeBundle, config_signature

if TYPE_CHECKING:
    from homesec.interfaces import (
        AlertPolicy,
        ClipSource,
        Notifier,
        ObjectFilter,
        StorageBackend,
        VLMAnalyzer,
    )
    from homesec.models.config import Config

logger = logging.getLogger(__name__)


class _AsyncShutdown(Protocol):
    async def shutdown(self, timeout: float = 30.0) -> None:
        """Release resources."""


class RuntimeAssembler:
    """Builds and manages the lifecycle of in-process runtime bundles."""

    def __init__(
        self,
        *,
        storage: StorageBackend,
        repository: ClipRepository,
        notifier_factory: Callable[[Config], tuple[Notifier, list[NotifierEntry]]],
        notifier_health_logger: Callable[[list[NotifierEntry]], Awaitable[None]],
        alert_policy_factory: Callable[[Config], AlertPolicy],
        source_factory: Callable[[Config], tuple[list[ClipSource], dict[str, ClipSource]]],
        source_start_timeout_s: float = 10.0,
        source_shutdown_timeout_s: float = 30.0,
        component_shutdown_timeout_s: float = 30.0,
        shutdown_wait_grace_s: float = 2.0,
    ) -> None:
        self._storage = storage
        self._repository = repository
        self._notifier_factory = notifier_factory
        self._notifier_health_logger = notifier_health_logger
        self._alert_policy_factory = alert_policy_factory
        self._source_factory = source_factory
        self._source_start_timeout_s = source_start_timeout_s
        self._source_shutdown_timeout_s = source_shutdown_timeout_s
        self._component_shutdown_timeout_s = component_shutdown_timeout_s
        self._shutdown_wait_grace_s = shutdown_wait_grace_s

    async def build_bundle(self, config: Config, generation: int) -> RuntimeBundle:
        """Build a runtime bundle and clean up partial state on failure."""
        notifier: Notifier | None = None
        notifier_entries: list[NotifierEntry] = []
        filter_plugin: ObjectFilter | None = None
        vlm_plugin: VLMAnalyzer | None = None
        pipeline: ClipPipeline | None = None
        sources: list[ClipSource] = []

        try:
            notifier, notifier_entries = self._notifier_factory(config)
            await self._notifier_health_logger(notifier_entries)

            filter_plugin = load_filter(config.filter)
            vlm_plugin = load_analyzer(config.vlm)
            alert_policy = self._alert_policy_factory(config)

            pipeline = ClipPipeline(
                config=config,
                storage=self._storage,
                repository=self._repository,
                filter_plugin=filter_plugin,
                vlm_plugin=vlm_plugin,
                notifier=notifier,
                alert_policy=alert_policy,
                notifier_entries=notifier_entries,
            )
            pipeline.set_event_loop(asyncio.get_running_loop())

            sources, sources_by_camera = self._source_factory(config)
            for source in sources:
                source.register_callback(pipeline.on_new_clip)

            return RuntimeBundle(
                generation=generation,
                config=config,
                config_signature=config_signature(config),
                notifier=notifier,
                notifier_entries=notifier_entries,
                filter_plugin=filter_plugin,
                vlm_plugin=vlm_plugin,
                alert_policy=alert_policy,
                pipeline=pipeline,
                sources=sources,
                sources_by_camera=sources_by_camera,
            )
        except Exception:
            await self._cleanup_partial_build(
                sources=sources,
                pipeline=pipeline,
                filter_plugin=filter_plugin,
                vlm_plugin=vlm_plugin,
                notifier=notifier,
            )
            raise

    async def start_bundle(self, runtime: RuntimeBundle) -> None:
        """Start runtime sources and fail fast on startup preflight errors."""
        started_sources: list[ClipSource] = []
        startup_errors: list[tuple[str, Exception]] = []

        for source in runtime.sources:
            camera_name = getattr(source, "camera_name", source.__class__.__name__)
            try:
                await asyncio.wait_for(source.start(), timeout=self._source_start_timeout_s)
            except asyncio.TimeoutError as exc:
                startup_errors.append(
                    (
                        str(camera_name),
                        RuntimeError(
                            f"source start timed out after {self._source_start_timeout_s:.1f}s"
                        ),
                    )
                )
                logger.error(
                    "Source startup timed out: camera=%s timeout_s=%.1f",
                    camera_name,
                    self._source_start_timeout_s,
                    exc_info=exc,
                )
            except Exception as exc:
                startup_errors.append((str(camera_name), exc))
            else:
                started_sources.append(source)

        if not startup_errors:
            return

        await self._safe_shutdown_sources(started_sources)

        for camera_name, error in startup_errors:
            logger.error(
                "Source startup failed: camera=%s error=%s",
                camera_name,
                error,
                exc_info=error,
            )
        summary = "; ".join(f"{camera_name}: {error}" for camera_name, error in startup_errors)
        raise RuntimeError(f"Source startup preflight failed: {summary}")

    async def shutdown_bundle(self, runtime: RuntimeBundle) -> None:
        """Gracefully stop a runtime bundle with best-effort cleanup."""
        await self._shutdown_runtime_parts(
            sources=runtime.sources,
            pipeline=runtime.pipeline,
            filter_plugin=runtime.filter_plugin,
            vlm_plugin=runtime.vlm_plugin,
            notifier=runtime.notifier,
        )

    async def _cleanup_partial_build(
        self,
        *,
        sources: list[ClipSource],
        pipeline: ClipPipeline | None,
        filter_plugin: ObjectFilter | None,
        vlm_plugin: VLMAnalyzer | None,
        notifier: Notifier | None,
    ) -> None:
        await self._shutdown_runtime_parts(
            sources=sources,
            pipeline=pipeline,
            filter_plugin=filter_plugin,
            vlm_plugin=vlm_plugin,
            notifier=notifier,
        )

    async def _shutdown_runtime_parts(
        self,
        *,
        sources: list[ClipSource],
        pipeline: ClipPipeline | None,
        filter_plugin: ObjectFilter | None,
        vlm_plugin: VLMAnalyzer | None,
        notifier: Notifier | None,
    ) -> None:
        await self._safe_shutdown_sources(sources)
        await self._safe_shutdown_component(pipeline, label="Pipeline")
        await self._safe_shutdown_component(filter_plugin, label="Filter")
        await self._safe_shutdown_component(vlm_plugin, label="VLM")
        await self._safe_shutdown_component(notifier, label="Notifier")

    async def _safe_shutdown_sources(self, sources: list[ClipSource]) -> None:
        if not sources:
            return

        async def _shutdown_source(source: ClipSource) -> Exception | None:
            try:
                await asyncio.wait_for(
                    source.shutdown(timeout=self._source_shutdown_timeout_s),
                    timeout=self._source_shutdown_timeout_s + self._shutdown_wait_grace_s,
                )
                return None
            except Exception as exc:
                return exc

        results = await asyncio.gather(
            *(_shutdown_source(source) for source in sources),
        )
        for source, result in zip(sources, results, strict=True):
            if result is not None:
                camera_name = getattr(source, "camera_name", source.__class__.__name__)
                logger.error(
                    "Source shutdown failed: camera=%s error=%s",
                    camera_name,
                    result,
                    exc_info=result,
                )

    async def _safe_shutdown_component(
        self,
        component: _AsyncShutdown | None,
        *,
        label: str,
    ) -> None:
        if component is None:
            return
        try:
            await asyncio.wait_for(
                component.shutdown(timeout=self._component_shutdown_timeout_s),
                timeout=self._component_shutdown_timeout_s + self._shutdown_wait_grace_s,
            )
        except Exception as exc:
            logger.error("%s shutdown failed: %s", label, exc, exc_info=exc)
