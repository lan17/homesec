"""Interface definitions for HomeSec pipeline components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from homesec.models.alert import Alert, AlertDecision
    from homesec.models.clip import Clip, ClipStateData
    from homesec.models.events import ClipLifecycleEvent
    from homesec.models.filter import FilterOverrides, FilterResult
    from homesec.models.storage import StorageUploadResult
    from homesec.models.vlm import AnalysisResult, VLMConfig


class Shutdownable(ABC):
    """Async shutdown interface for managed components."""

    @abstractmethod
    async def shutdown(self, timeout: float | None = None) -> None:
        """Release resources and stop background work."""
        raise NotImplementedError


class ClipSource(Shutdownable, ABC):
    """Produces finalized clips and notifies pipeline via callback."""

    @abstractmethod
    def register_callback(self, callback: Callable[[Clip], None]) -> None:
        """Register callback to be invoked when a new clip is finalized."""
        raise NotImplementedError

    @abstractmethod
    async def start(self) -> None:
        """Start producing clips (long-running, blocks or runs in background)."""
        raise NotImplementedError

    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if source is actively able to receive clips.

        Implementation should check:
        - Process/thread is alive
        - Receiving data recently (e.g., RTSP frames within timeout)
        - NOT dependent on motion/clip activity

        Examples:
        - RTSP: frame pipeline running + receiving frames recently
        - FTP: server thread alive + accepting connections

        Returns False if source has failed and needs restart.
        """
        raise NotImplementedError

    @abstractmethod
    def last_heartbeat(self) -> float:
        """Return timestamp (monotonic) of last successful operation.

        Examples:
        - RTSP: timestamp of last frame received
        - FTP: timestamp of last connection check

        Updated continuously (every ~60s), independent of motion/clips.
        Used for observability, not health status.
        """
        raise NotImplementedError

    @abstractmethod
    async def ping(self) -> bool:
        """Health check for the clip source.

        Returns True if the source is operational:
        - Connection/watcher is alive
        - Can receive new clips

        This is similar to is_healthy() but async and follows the standard
        ping() pattern used by other interfaces.
        """
        raise NotImplementedError


class StorageBackend(Shutdownable, ABC):
    """Stores raw clips and derived artifacts."""

    @abstractmethod
    async def put_file(self, local_path: Path, dest_path: str) -> StorageUploadResult:
        """Upload file to storage. Returns storage result."""
        raise NotImplementedError

    @abstractmethod
    async def get_view_url(self, storage_uri: str) -> str | None:
        """Return a web-accessible view URL for the given storage URI."""
        raise NotImplementedError

    @abstractmethod
    async def get(self, storage_uri: str, local_path: Path) -> None:
        """Download file from storage to local path."""
        raise NotImplementedError

    @abstractmethod
    async def exists(self, storage_uri: str) -> bool:
        """Check if file exists in storage."""
        raise NotImplementedError

    @abstractmethod
    async def delete(self, storage_uri: str) -> None:
        """Delete an object from storage.

        Must be idempotent: deleting a missing object should succeed.
        """
        raise NotImplementedError

    @abstractmethod
    async def ping(self) -> bool:
        """Health check. Returns True if storage is reachable."""
        raise NotImplementedError


class StateStore(Shutdownable, ABC):
    """Manages clip workflow state in Postgres."""

    @abstractmethod
    async def upsert(self, clip_id: str, data: ClipStateData) -> None:
        """Insert or update clip state."""
        raise NotImplementedError

    @abstractmethod
    async def get(self, clip_id: str) -> ClipStateData | None:
        """Retrieve clip state. Returns None if not found."""
        raise NotImplementedError

    @abstractmethod
    async def list_candidate_clips_for_cleanup(
        self,
        *,
        older_than_days: int | None,
        camera_name: str | None,
        batch_size: int,
        cursor: tuple[datetime, str] | None = None,
    ) -> list[tuple[str, ClipStateData, datetime]]:
        """List clip states for cleanup scanning."""
        raise NotImplementedError

    @abstractmethod
    async def ping(self) -> bool:
        """Health check. Returns True if database is reachable."""
        raise NotImplementedError

    @abstractmethod
    def create_event_store(self) -> EventStore:
        """Create an event store associated with this state store.

        Returns NoopEventStore if not supported.
        """
        from homesec.state import NoopEventStore

        return NoopEventStore()


class EventStore(Shutdownable, ABC):
    """Manages clip lifecycle events in Postgres."""

    @abstractmethod
    async def append(self, event: ClipLifecycleEvent) -> None:
        """Append a lifecycle event.

        Events are appended with database-assigned ids.
        Raises on database errors (should be retried by caller).
        """
        raise NotImplementedError

    @abstractmethod
    async def get_events(
        self,
        clip_id: str,
        after_id: int | None = None,
    ) -> list[ClipLifecycleEvent]:
        """Get all events for a clip, optionally after an event id.

        Returns events ordered by id. Returns empty list on error.
        """
        raise NotImplementedError

    @abstractmethod
    async def ping(self) -> bool:
        """Health check. Returns True if event store is reachable."""
        raise NotImplementedError


class Notifier(Shutdownable, ABC):
    """Sends notifications (e.g., MQTT, email, SMS)."""

    @abstractmethod
    async def send(self, alert: Alert) -> None:
        """Send notification. Raises on failure."""
        raise NotImplementedError

    @abstractmethod
    async def ping(self) -> bool:
        """Health check. Returns True if notifier is reachable."""
        raise NotImplementedError


class AlertPolicy(ABC):
    """Decides whether to notify based on analysis results."""

    @abstractmethod
    def should_notify(
        self,
        camera_name: str,
        filter_result: FilterResult | None,
        analysis: AnalysisResult | None,
    ) -> tuple[bool, str]:
        """Determine if notification should be sent.

        Returns:
            (notify, reason) tuple where reason explains the decision.
        """
        raise NotImplementedError

    def make_decision(
        self,
        camera_name: str,
        filter_result: FilterResult | None,
        analysis: AnalysisResult | None,
    ) -> AlertDecision:
        """Build an AlertDecision from should_notify output."""
        from homesec.models.alert import AlertDecision

        notify, reason = self.should_notify(camera_name, filter_result, analysis)
        return AlertDecision(notify=notify, notify_reason=reason)


class ObjectFilter(Shutdownable, ABC):
    """Plugin interface for object detection in video clips."""

    @abstractmethod
    async def detect(
        self, video_path: Path, overrides: FilterOverrides | None = None
    ) -> FilterResult:
        """Detect objects in video clip.

        Implementation notes:
        - MUST be async (use asyncio.to_thread or run_in_executor for blocking code)
        - CPU/GPU-bound plugins should manage their own ProcessPoolExecutor internally
        - I/O-bound plugins can use async HTTP clients directly
        - If managing a worker pool, use concurrency settings from the plugin's config model
        - Should support early exit on first detection for efficiency
        - overrides apply per-call (model path cannot be overridden)

        Returns:
            FilterResult with detected_classes, confidence, sampled_frames, model name
        """
        raise NotImplementedError

    @abstractmethod
    async def ping(self) -> bool:
        """Health check for the filter.

        Returns True if the filter is operational:
        - Model is loaded
        - Executor pool is alive (if applicable)
        - Ready to process detection requests
        """
        raise NotImplementedError


class VLMAnalyzer(Shutdownable, ABC):
    """Plugin interface for VLM-based clip analysis."""

    @abstractmethod
    async def analyze(
        self, video_path: Path, filter_result: FilterResult, config: VLMConfig
    ) -> AnalysisResult:
        """Analyze clip and produce structured assessment.

        Implementation notes:
        - MUST be async (use asyncio.to_thread or run_in_executor for blocking code)
        - Local models: manage ProcessPoolExecutor internally
        - API-based: use async HTTP clients (aiohttp, httpx)
        - If managing a worker pool, use concurrency settings from the plugin's config model
        - Should use filter_result to focus analysis (e.g., detected person at timestamp X)

        Returns:
            AnalysisResult with risk_level, activity_type, summary, etc.
        """
        raise NotImplementedError

    @abstractmethod
    async def ping(self) -> bool:
        """Health check for the analyzer.

        Returns True if the analyzer is operational:
        - API endpoint reachable (for API-based analyzers)
        - Model loaded (for local analyzers)
        - HTTP session alive
        """
        raise NotImplementedError
