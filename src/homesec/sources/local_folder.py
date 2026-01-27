"""Local folder clip source for production and development."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

from anyio import Path as AsyncPath

from homesec.models.clip import Clip
from homesec.models.source.local_folder import LocalFolderSourceConfig
from homesec.sources.base import AsyncClipSource

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
        camera_name: str | None = None,
    ) -> None:
        """Initialize folder watcher.

        Args:
            config: LocalFolder source configuration
            camera_name: Name of the camera (overrides config.camera_name).
        """
        super().__init__()
        self.watch_dir = Path(config.watch_dir)
        # Use config's camera_name if not explicitly passed, else default to "local"
        self.camera_name = camera_name or config.camera_name or "local"
        self.poll_interval = float(config.poll_interval)
        self.stability_threshold_s = float(config.stability_threshold_s)

        # Ensure watch dir exists
        self.watch_dir.mkdir(parents=True, exist_ok=True)

        # Local State Manifest (replaces StateStore dependency)
        # This file tracks which clips have been processed to avoid re-emitting on restart.
        self._state_file = self.watch_dir / ".homesec_state.json"
        self._processed_files: set[str] = set()
        self._load_local_state()

        # Bounded in-memory cache for performance (avoid checking set on every scan)
        self._seen_files: OrderedDict[str, None] = OrderedDict()
        self._max_seen_files = 10000

        logger.info(
            "LocalFolderSource initialized: watch_dir=%s, camera_name=%s",
            self.watch_dir,
            self.camera_name,
        )

    def register_callback(self, callback: Callable[[Clip], None]) -> None:
        """Register callback to be invoked when new clip is ready."""
        super().register_callback(callback)
        logger.debug("Callback registered for %s", self.camera_name)

    def is_healthy(self) -> bool:
        """Check if source is healthy."""
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

    def _load_local_state(self) -> None:
        """Load processed files from local JSON manifest."""
        if not self._state_file.exists():
            return
        try:
            with open(self._state_file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    self._processed_files = set(data)
                elif isinstance(data, dict) and "processed_files" in data:
                    self._processed_files = set(data["processed_files"])
            logger.info("Loaded %d processed files from local state", len(self._processed_files))
        except Exception as e:
            logger.warning("Failed to load local state file: %s", e)

    def _save_local_state(self) -> None:
        """Save processed files to local JSON manifest."""
        # Simple atomic write
        try:
            temp_file = self._state_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump({"processed_files": list(self._processed_files)}, f)
            temp_file.replace(self._state_file)
        except Exception as e:
            logger.warning("Failed to save local state file: %s", e)

    async def _run(self) -> None:
        """Background task that polls for new files."""
        logger.info("Watch loop started")

        # Create async path wrapper for watch directory
        async_watch_dir = AsyncPath(self.watch_dir)

        while not self._stop_event.is_set():
            try:
                # Update heartbeat
                self._touch_heartbeat()

                new_files_processed = False

                # Scan for new .mp4 files (async to avoid blocking event loop)
                async for async_file_path in async_watch_dir.glob("*.mp4"):
                    file_path = Path(async_file_path)  # Convert to regular Path for Clip
                    file_id = str(file_path)
                    clip_id = file_path.stem

                    # Check if already processed (Session cache OR Persistent Manifest)
                    if clip_id in self._processed_files:
                        continue

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

                    # Mark as seen
                    self._processed_files.add(clip_id)
                    self._add_to_cache(file_id)
                    new_files_processed = True

                    # Create Clip object (reuse mtime from async stat to avoid blocking)
                    clip = self._create_clip(file_path, mtime=mtime)

                    # Invoke callback
                    logger.info("New clip detected: %s", file_path.name)
                    self._emit_clip(clip)

                # Check for removed files to clean up manifest (OPTIONAL, maybe too expensive?)
                # For now, let's just save if we added anything.
                if new_files_processed:
                    await asyncio.to_thread(self._save_local_state)

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

    def _create_clip(self, file_path: Path, mtime: float) -> Clip:
        """Create Clip object from file path."""
        mtime_dt = datetime.fromtimestamp(mtime)

        # Estimate clip duration (assume 10s if we can't determine)
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
