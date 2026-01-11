"""Local folder clip source for production and development."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from anyio import Path as AsyncPath

from homesec.models.clip import Clip
from homesec.models.source import LocalFolderSourceConfig
from homesec.sources.base import AsyncClipSource

if TYPE_CHECKING:
    from homesec.interfaces import StateStore

logger = logging.getLogger(__name__)


class LocalFolderSource(AsyncClipSource):
    """Watches a local folder for new video clips.

    Production-ready async clip source that monitors a directory for new .mp4 files.
    Uses anyio for non-blocking filesystem operations (glob, stat).
    Suitable for both testing and production use with local camera storage.
    """

    def __init__(
        self,
        config: LocalFolderSourceConfig,
        camera_name: str = "local",
        state_store: StateStore | None = None,
    ) -> None:
        """Initialize folder watcher.

        Args:
            config: LocalFolder source configuration
            camera_name: Name of the camera (used in Clip objects)
            state_store: Optional state store for deduplication via clip_states table.
                If None, falls back to in-memory cache only (may reprocess files after restart).
        """
        super().__init__()
        self.watch_dir = Path(config.watch_dir)
        self.camera_name = camera_name
        self.poll_interval = float(config.poll_interval)
        self.stability_threshold_s = float(config.stability_threshold_s)
        self._state_store = state_store

        # Ensure watch dir exists
        self.watch_dir.mkdir(parents=True, exist_ok=True)

        # Bounded in-memory cache for performance (avoid DB query on every scan)
        # 10,000 files ≈ 100 days for 1 camera @ 100 clips/day (≈500 KB memory)
        # When limit exceeded, oldest half are removed (FIFO eviction)
        # This is just an optimization - clip_states table is source of truth
        self._seen_files: OrderedDict[str, None] = OrderedDict()
        self._max_seen_files = 10000

        logger.info(
            "LocalFolderSource initialized: watch_dir=%s, has_state_store=%s",
            self.watch_dir,
            state_store is not None,
        )

    def register_callback(self, callback: Callable[[Clip], None]) -> None:
        """Register callback to be invoked when new clip is ready."""
        super().register_callback(callback)
        logger.debug("Callback registered for %s", self.camera_name)

    def is_healthy(self) -> bool:
        """Check if source is healthy.

        Returns True if:
        - Watch directory exists and is readable
        - Watch task is alive (if started)
        """
        if not self.watch_dir.exists():
            return False

        if not self.watch_dir.is_dir():
            return False

        # If started, task should be running
        if not self._task_is_healthy():
            return False

        return True

    def _on_start(self) -> None:
        logger.info("Starting LocalFolderSource: %s", self.watch_dir)

    def _on_stop(self) -> None:
        logger.info("Stopping LocalFolderSource...")

    def _on_stopped(self) -> None:
        logger.info("LocalFolderSource stopped")

    async def _run(self) -> None:
        """Background task that polls for new files.

        Uses anyio.Path for non-blocking filesystem operations to avoid
        stalling the event loop on slow/network filesystems.
        """
        logger.info("Watch loop started")

        # Create async path wrapper for watch directory
        async_watch_dir = AsyncPath(self.watch_dir)

        while not self._stop_event.is_set():
            try:
                # Update heartbeat
                self._touch_heartbeat()

                # Scan for new .mp4 files (async to avoid blocking event loop)
                async for async_file_path in async_watch_dir.glob("*.mp4"):
                    file_path = Path(async_file_path)  # Convert to regular Path for Clip
                    file_id = str(file_path)
                    clip_id = file_path.stem

                    # Check in-memory cache first (fast path)
                    if file_id in self._seen_files:
                        continue

                    # Check file stability (avoid processing while still being written)
                    # Use async stat to avoid blocking event loop
                    try:
                        stat_info = await async_file_path.stat()
                        mtime = stat_info.st_mtime
                        age_s = time.time() - mtime
                        if age_s < self.stability_threshold_s:
                            logger.debug(
                                "Skipping unstable file: %s (modified %.1fs ago)",
                                file_path.name,
                                age_s,
                            )
                            continue
                    except OSError as e:
                        logger.warning("Failed to stat file %s: %s", file_path, e, exc_info=True)
                        continue

                    # Check clip_states table (source of truth for deduplication)
                    # This prevents reprocessing files even after cache eviction or restart
                    if await self._has_clip_state(clip_id):
                        # File was already processed - add to cache and skip
                        self._add_to_cache(file_id)
                        logger.debug("Skipping already-processed file: %s", file_path.name)
                        continue

                    # Mark as seen in cache
                    self._add_to_cache(file_id)

                    # Create Clip object (reuse mtime from async stat to avoid blocking)
                    clip = self._create_clip(file_path, mtime=mtime)

                    # Invoke callback
                    logger.info("New clip detected: %s", file_path.name)
                    self._emit_clip(clip)

            except Exception as e:
                logger.error("Error in watch loop: %s", e, exc_info=True)

            # Sleep before next poll
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.poll_interval)
            except asyncio.TimeoutError:
                pass  # Normal - just means poll_interval elapsed

        logger.info("Watch loop exited")

    def _add_to_cache(self, file_id: str) -> None:
        """Add file to in-memory cache with FIFO eviction."""
        self._seen_files[file_id] = None
        if len(self._seen_files) > self._max_seen_files:
            # Remove oldest half (FIFO eviction)
            evict_count = self._max_seen_files // 2
            for _ in range(evict_count):
                self._seen_files.popitem(last=False)
            logger.debug("Evicted %d old entries from seen files cache", evict_count)

    async def _has_clip_state(self, clip_id: str) -> bool:
        """Check if clip_id exists in clip_states table.

        Returns:
            True if clip_state exists (any status including 'deleted'), False otherwise
        """
        if self._state_store is None:
            return False

        try:
            # Direct async DB access - no threading complexity!
            state = await asyncio.wait_for(
                self._state_store.get(clip_id),
                timeout=5.0,
            )
            return state is not None
        except asyncio.TimeoutError:
            logger.warning(
                "DB query timeout for clip_states check: %s (assuming not seen)",
                clip_id,
            )
            return False
        except Exception as e:
            logger.warning(
                "Error checking clip_states for %s: %s (assuming not seen)",
                clip_id,
                e,
                exc_info=True,
            )
            return False

    def _create_clip(self, file_path: Path, mtime: float) -> Clip:
        """Create Clip object from file path.

        Args:
            file_path: Path to the video file
            mtime: File modification timestamp (from async stat)

        Estimates timestamps based on file modification time.
        """
        mtime_dt = datetime.fromtimestamp(mtime)

        # Estimate clip duration (assume 10s if we can't determine)
        # In production, would parse from filename or video metadata
        duration_s = 10.0

        return Clip(
            clip_id=file_path.stem,
            camera_name=self.camera_name,
            local_path=file_path,
            start_ts=mtime_dt - timedelta(seconds=duration_s),
            end_ts=mtime_dt,
            duration_s=duration_s,
            source_type="local_folder",
        )
