"""ClipPipeline orchestrator - core processing logic."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, TypeVar

from homesec.errors import FilterError, NotifyError, UploadError, VLMError
from homesec.models.alert import Alert, AlertDecision
from homesec.models.clip import Clip
from homesec.models.config import Config
from homesec.models.filter import FilterResult
from homesec.models.vlm import AnalysisResult
from homesec.notifiers.multiplex import NotifierEntry
from homesec.repository import ClipRepository
from homesec.storage_paths import build_clip_path

if TYPE_CHECKING:
    from homesec.interfaces import (
        AlertPolicy,
        Notifier,
        ObjectFilter,
        StorageBackend,
        VLMAnalyzer,
    )

logger = logging.getLogger(__name__)

TResult = TypeVar("TResult")


@dataclass(frozen=True)
class UploadOutcome:
    storage_uri: str
    view_url: str | None


class ClipPipeline:
    """Orchestrates clip processing through all pipeline stages.

    Implements error-as-value pattern: stage methods return Result | Error
    instead of raising. This enables partial failures (e.g., upload fails
    but filter+notify still run).
    """

    def __init__(
        self,
        config: Config,
        storage: StorageBackend,
        repository: ClipRepository,
        filter_plugin: ObjectFilter,
        vlm_plugin: VLMAnalyzer,
        notifier: Notifier,
        alert_policy: AlertPolicy,
        notifier_entries: list[NotifierEntry] | None = None,
    ) -> None:
        """Initialize pipeline with all dependencies."""
        self._config = config
        self._storage = storage
        self._repository = repository
        self._filter = filter_plugin
        self._vlm = vlm_plugin
        self._notifier = notifier
        self._notifier_entries = self._resolve_notifier_entries(notifier, notifier_entries)
        self._alert_policy = alert_policy

        # Track in-flight processing
        self._tasks: set[asyncio.Task[None]] = set()

        # Concurrency limits
        self._sem_global = asyncio.Semaphore(config.concurrency.max_clips_in_flight)
        self._sem_upload = asyncio.Semaphore(config.concurrency.upload_workers)
        self._sem_filter = asyncio.Semaphore(config.concurrency.filter_workers)
        self._sem_vlm = asyncio.Semaphore(config.concurrency.vlm_workers)

        # Event loop for thread-safe callback handling
        self._loop: asyncio.AbstractEventLoop | None = None

    @staticmethod
    def _resolve_notifier_entries(
        notifier: Notifier,
        notifier_entries: list[NotifierEntry] | None,
    ) -> list[NotifierEntry]:
        if notifier_entries:
            return list(notifier_entries)
        name = getattr(notifier, "name", type(notifier).__name__)
        return [NotifierEntry(name=name, notifier=notifier)]

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set event loop for thread-safe callback handling.

        Must be called before registering with ClipSource if source
        runs in a different thread.
        """
        self._loop = loop

    def _create_task(self, loop: asyncio.AbstractEventLoop, clip: Clip) -> None:
        """Create and track a processing task in the given loop."""
        task = loop.create_task(self._process_clip(clip))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(self._log_task_exception)

    def _log_task_exception(self, task: asyncio.Task[None]) -> None:
        """Log unexpected task exceptions."""
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            logger.error("Clip processing failed: %s", exc, exc_info=exc)

    def on_new_clip(self, clip: Clip) -> None:
        """Callback for ClipSource when new clip is ready.

        Thread-safe: can be called from any thread. Uses stored event loop
        if available, otherwise tries to get current loop.
        """
        # Try to get current running loop (works if called from async context)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            self._create_task(loop, clip)
            return

        # Use stored loop for thread-safe scheduling
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._create_task, self._loop, clip)
            return

        logger.error(
            "Cannot process clip %s: no event loop available. "
            "Call set_event_loop() before registering with ClipSource.",
            clip.clip_id,
        )

    async def _process_clip(self, clip: Clip) -> None:
        """Process a single clip through all stages.

        Flow:
        1. Parallel: upload + filter
        2. Conditional: VLM (if filter detects trigger classes)
        3. Alert decision
        4. Conditional: Notify (if alert decision is True)
        """
        async with self._sem_global:
            logger.info("Processing clip: %s", clip.clip_id)

            # Initialize state + record clip arrival
            await self._repository.initialize_clip(clip)

            storage_uri: str | None = None
            view_url: str | None = None
            upload_failed = False

            # Stage 1 & 2: Upload and Filter in parallel
            upload_task = asyncio.create_task(self._upload_stage(clip))
            filter_task = asyncio.create_task(self._filter_stage(clip))

            filter_result = await filter_task

            # Handle filter result (critical - cannot proceed without it)
            match filter_result:
                case FilterError() as filter_err:
                    logger.error(
                        "Filter failed for %s: %s",
                        clip.clip_id,
                        filter_err.cause,
                        exc_info=filter_err.cause,
                    )
                    upload_result = await upload_task
                    await self._apply_upload_result(clip, upload_result)
                    return
                case FilterResult() as filter_res:
                    pass
                case _:
                    raise TypeError(
                        f"Unexpected filter result type: {type(filter_result).__name__}"
                    )
            logger.info(
                "Filter complete for %s: detected %s",
                clip.clip_id,
                filter_res.detected_classes,
            )

            # Stage 3: VLM (conditional)
            analysis_result: AnalysisResult | None = None
            vlm_failed = False
            if self._should_run_vlm(filter_res):
                vlm_result = await self._vlm_stage(clip, filter_res)
                match vlm_result:
                    case VLMError() as vlm_err:
                        logger.warning(
                            "VLM failed for %s (continuing): %s",
                            clip.clip_id,
                            vlm_err.cause,
                        )
                        vlm_failed = True
                    case AnalysisResult() as analysis_result:
                        logger.info(
                            "VLM complete for %s: risk=%s, activity=%s",
                            clip.clip_id,
                            analysis_result.risk_level,
                            analysis_result.activity_type,
                        )
                    case _:
                        raise TypeError(f"Unexpected VLM result type: {type(vlm_result).__name__}")
            else:
                await self._repository.record_vlm_skipped(
                    clip.clip_id,
                    reason="no_trigger_classes",
                )
                logger.info("VLM skipped for %s: no trigger classes", clip.clip_id)

            # Await upload after filter/VLM to maximize overlap
            upload_result = await upload_task

            # Handle upload result (non-critical - can proceed without URL)
            storage_uri, view_url, upload_failed = await self._apply_upload_result(
                clip, upload_result
            )

            # Stage 4: Alert decision
            alert_decision = self._alert_policy.make_decision(
                clip.camera_name, filter_res, analysis_result
            )
            logger.info(
                "Alert decision for %s: notify=%s, reason=%s",
                clip.clip_id,
                alert_decision.notify,
                alert_decision.notify_reason,
            )
            await self._repository.record_alert_decision(
                clip.clip_id,
                alert_decision,
                detected_classes=filter_res.detected_classes,
                vlm_risk=analysis_result.risk_level if analysis_result else None,
            )

            # Stage 5: Notify (conditional)
            if alert_decision.notify:
                notify_result = await self._notify_stage(
                    clip,
                    alert_decision,
                    analysis_result,
                    filter_res.detected_classes,
                    storage_uri,
                    view_url,
                    upload_failed,
                    vlm_failed,
                )
                match notify_result:
                    case NotifyError() as notify_err:
                        logger.error(
                            "Notify failed for %s: %s",
                            clip.clip_id,
                            notify_err.cause,
                            exc_info=notify_err.cause,
                        )
                    case None:
                        logger.info("Notification sent for %s", clip.clip_id)
                    case _:
                        raise TypeError(
                            f"Unexpected notify result type: {type(notify_result).__name__}"
                        )

            await self._repository.mark_done(clip.clip_id)
            logger.info("Clip processing complete: %s", clip.clip_id)

    async def _run_stage_with_retries(
        self,
        *,
        stage: str,
        clip_id: str,
        op: Callable[[], Awaitable[TResult]],
        on_attempt_start: Callable[[int], Awaitable[None]] | None = None,
        on_attempt_success: Callable[[TResult, int, int], Awaitable[None]] | None = None,
        on_attempt_failure: Callable[[Exception, int, bool, int], Awaitable[None]] | None = None,
    ) -> TResult:
        """Run stage with retry logic and event emission."""
        max_attempts = max(1, int(self._config.retry.max_attempts))
        backoff_s = max(0.0, float(self._config.retry.backoff_s))
        attempts = 1

        while True:
            if on_attempt_start is not None:
                await on_attempt_start(attempts)
            started = time.monotonic()
            try:
                result = await op()
            except Exception as exc:
                duration_ms = int((time.monotonic() - started) * 1000)
                will_retry = attempts < max_attempts
                if on_attempt_failure is not None:
                    await on_attempt_failure(exc, attempts, will_retry, duration_ms)
                if attempts >= max_attempts:
                    raise
                logger.warning(
                    "Stage %s failed for %s (attempt %d/%d): %s",
                    stage,
                    clip_id,
                    attempts,
                    max_attempts,
                    exc,
                    exc_info=True,
                )
                delay = backoff_s * (2 ** (attempts - 1))
                if delay > 0:
                    await asyncio.sleep(delay)
                attempts += 1
            else:
                duration_ms = int((time.monotonic() - started) * 1000)
                if on_attempt_success is not None:
                    await on_attempt_success(result, attempts, duration_ms)
                return result

    async def _upload_stage(self, clip: Clip) -> UploadOutcome | UploadError:
        """Upload clip to storage. Returns UploadOutcome or UploadError."""
        dest_path = build_clip_path(clip, self._config.storage.paths)

        async def attempt() -> UploadOutcome:
            async with self._sem_upload:
                storage_result = await self._storage.put_file(
                    clip.local_path,
                    dest_path,
                )
                return UploadOutcome(
                    storage_uri=storage_result.storage_uri,
                    view_url=storage_result.view_url,
                )

        async def on_attempt_start(attempt_num: int) -> None:
            await self._repository.record_upload_started(
                clip.clip_id,
                dest_key=dest_path,
                attempt=attempt_num,
            )

        async def on_attempt_success(
            result: UploadOutcome, attempt_num: int, duration_ms: int
        ) -> None:
            await self._repository.record_upload_completed(
                clip.clip_id,
                result.storage_uri,
                result.view_url,
                duration_ms,
                attempt=attempt_num,
            )

        async def on_attempt_failure(
            exc: Exception, attempt_num: int, will_retry: bool, _duration_ms: int
        ) -> None:
            await self._repository.record_upload_failed(
                clip.clip_id,
                error_message=self._format_error_message(exc),
                error_type=self._format_error_type(exc),
                attempt=attempt_num,
                will_retry=will_retry,
            )

        try:
            return await self._run_stage_with_retries(
                stage="upload",
                clip_id=clip.clip_id,
                op=attempt,
                on_attempt_start=on_attempt_start,
                on_attempt_success=on_attempt_success,
                on_attempt_failure=on_attempt_failure,
            )
        except Exception as e:
            return UploadError(clip.clip_id, storage_uri=None, cause=e)

    async def _filter_stage(self, clip: Clip) -> FilterResult | FilterError:
        """Run object detection filter. Returns FilterResult or FilterError."""

        async def attempt() -> FilterResult:
            async with self._sem_filter:
                return await self._filter.detect(clip.local_path)

        async def on_attempt_start(attempt_num: int) -> None:
            await self._repository.record_filter_started(clip.clip_id, attempt=attempt_num)

        async def on_attempt_success(
            result: FilterResult, attempt_num: int, duration_ms: int
        ) -> None:
            await self._repository.record_filter_completed(
                clip.clip_id,
                result,
                duration_ms,
                attempt=attempt_num,
            )

        async def on_attempt_failure(
            exc: Exception, attempt_num: int, will_retry: bool, _duration_ms: int
        ) -> None:
            await self._repository.record_filter_failed(
                clip.clip_id,
                error_message=self._format_error_message(exc),
                error_type=self._format_error_type(exc),
                attempt=attempt_num,
                will_retry=will_retry,
            )

        try:
            return await self._run_stage_with_retries(
                stage="filter",
                clip_id=clip.clip_id,
                op=attempt,
                on_attempt_start=on_attempt_start,
                on_attempt_success=on_attempt_success,
                on_attempt_failure=on_attempt_failure,
            )
        except Exception as e:
            return FilterError(clip.clip_id, plugin_name=self._config.filter.backend, cause=e)

    async def _vlm_stage(
        self, clip: Clip, filter_result: FilterResult
    ) -> AnalysisResult | VLMError:
        """Run VLM analysis. Returns AnalysisResult or VLMError."""

        async def attempt() -> AnalysisResult:
            async with self._sem_vlm:
                return await self._vlm.analyze(clip.local_path, filter_result, self._config.vlm)

        async def on_attempt_start(attempt_num: int) -> None:
            await self._repository.record_vlm_started(clip.clip_id, attempt=attempt_num)

        async def on_attempt_success(
            result: AnalysisResult, attempt_num: int, duration_ms: int
        ) -> None:
            await self._repository.record_vlm_completed(
                clip.clip_id,
                result,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                duration_ms=duration_ms,
                attempt=attempt_num,
            )

        async def on_attempt_failure(
            exc: Exception, attempt_num: int, will_retry: bool, _duration_ms: int
        ) -> None:
            await self._repository.record_vlm_failed(
                clip.clip_id,
                error_message=self._format_error_message(exc),
                error_type=self._format_error_type(exc),
                attempt=attempt_num,
                will_retry=will_retry,
            )

        try:
            return await self._run_stage_with_retries(
                stage="vlm",
                clip_id=clip.clip_id,
                op=attempt,
                on_attempt_start=on_attempt_start,
                on_attempt_success=on_attempt_success,
                on_attempt_failure=on_attempt_failure,
            )
        except Exception as e:
            return VLMError(clip.clip_id, plugin_name=self._config.vlm.backend, cause=e)

    async def _notify_stage(
        self,
        clip: Clip,
        decision: AlertDecision,
        analysis_result: AnalysisResult | None,
        detected_classes: list[str],
        storage_uri: str | None,
        view_url: str | None,
        upload_failed: bool,
        vlm_failed: bool,
    ) -> None | NotifyError:
        """Send notification. Returns None on success or NotifyError."""
        alert = Alert(
            clip_id=clip.clip_id,
            camera_name=clip.camera_name,
            storage_uri=storage_uri,
            view_url=view_url,
            risk_level=analysis_result.risk_level if analysis_result else None,
            activity_type=analysis_result.activity_type if analysis_result else None,
            notify_reason=decision.notify_reason,
            summary=analysis_result.summary if analysis_result else None,
            analysis=analysis_result.analysis if analysis_result else None,
            detected_classes=detected_classes,
            ts=datetime.now(),
            dedupe_key=clip.clip_id,
            upload_failed=upload_failed,
            vlm_failed=vlm_failed,
        )

        tasks = [self._notify_with_entry(entry, alert) for entry in self._notifier_entries]
        results = await asyncio.gather(*tasks)

        errors: list[NotifyError] = []
        for result in results:
            match result:
                case NotifyError() as err:
                    errors.append(err)
                case None:
                    continue
                case _:
                    raise TypeError(f"Unexpected notify result type: {type(result).__name__}")

        if errors:
            return errors[0]
        return None

    async def _notify_with_entry(
        self,
        entry: NotifierEntry,
        alert: Alert,
    ) -> None | NotifyError:
        notifier_name = entry.name

        async def on_attempt_success(_result: object, attempt_num: int, _duration_ms: int) -> None:
            await self._repository.record_notification_sent(
                alert.clip_id,
                notifier_name=notifier_name,
                dedupe_key=alert.dedupe_key,
                attempt=attempt_num,
            )

        async def on_attempt_failure(
            exc: Exception, attempt_num: int, will_retry: bool, _duration_ms: int
        ) -> None:
            await self._repository.record_notification_failed(
                alert.clip_id,
                notifier_name=notifier_name,
                error_message=self._format_error_message(exc),
                error_type=self._format_error_type(exc),
                attempt=attempt_num,
                will_retry=will_retry,
            )

        try:
            await self._run_stage_with_retries(
                stage=f"notify:{notifier_name}",
                clip_id=alert.clip_id,
                op=lambda: entry.notifier.send(alert),
                on_attempt_success=on_attempt_success,
                on_attempt_failure=on_attempt_failure,
            )
            return None
        except Exception as exc:
            return NotifyError(alert.clip_id, notifier_name=notifier_name, cause=exc)

    @staticmethod
    def _format_error_message(exc: Exception) -> str:
        if isinstance(exc, (UploadError, FilterError, VLMError, NotifyError)):
            if exc.cause is not None:
                return str(exc.cause)
        return str(exc)

    @staticmethod
    def _format_error_type(exc: Exception) -> str:
        if isinstance(exc, (UploadError, FilterError, VLMError, NotifyError)):
            if exc.cause is not None:
                return type(exc.cause).__name__
        return type(exc).__name__

    def _should_run_vlm(self, filter_result: FilterResult) -> bool:
        """Check if VLM should run based on detected classes and config."""
        run_mode = self._config.vlm.run_mode
        if run_mode == "never":
            return False
        if run_mode == "always":
            return True

        detected = set(filter_result.detected_classes)
        trigger = set(self._config.vlm.trigger_classes)
        return bool(detected & trigger)

    async def _apply_upload_result(
        self,
        clip: Clip,
        upload_result: UploadOutcome | UploadError,
    ) -> tuple[str | None, str | None, bool]:
        """Return upload metadata for downstream stages."""
        match upload_result:
            case UploadError() as upload_err:
                logger.warning(
                    "Upload failed for %s (continuing): %s",
                    clip.clip_id,
                    upload_err.cause,
                )
                return None, None, True
            case UploadOutcome() as outcome:
                storage_uri = outcome.storage_uri
                view_url = outcome.view_url
            case _:
                raise TypeError(f"Unexpected upload result type: {type(upload_result).__name__}")
        logger.info("Upload complete for %s: %s", clip.clip_id, storage_uri)
        return storage_uri, view_url, False

    async def shutdown(self, timeout: float = 30.0) -> None:
        """Graceful shutdown of pipeline.

        Waits for in-flight tasks to complete. The app owns plugin shutdown.
        """
        logger.info("Shutting down pipeline...")

        # Wait for in-flight tasks
        if self._tasks:
            logger.info("Waiting for %d in-flight clips...", len(self._tasks))
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._tasks, return_exceptions=True),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for tasks, cancelling...")
                for task in self._tasks:
                    task.cancel()

        logger.info("Pipeline shutdown complete")
