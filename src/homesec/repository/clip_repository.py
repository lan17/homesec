"""ClipRepository for coordinating state + event persistence."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Awaitable, Callable, TypeVar

from homesec.models.clip import Clip, ClipStateData
from homesec.models.config import RetryConfig
from homesec.models.events import (
    AlertDecisionMadeEvent,
    ClipDeletedEvent,
    ClipRecheckedEvent,
    ClipLifecycleEvent,
    ClipRecordedEvent,
    FilterCompletedEvent,
    FilterFailedEvent,
    FilterStartedEvent,
    NotificationFailedEvent,
    NotificationSentEvent,
    UploadCompletedEvent,
    UploadFailedEvent,
    UploadStartedEvent,
    VLMCompletedEvent,
    VLMFailedEvent,
    VLMStartedEvent,
    VLMSkippedEvent,
)
from homesec.state.postgres import is_retryable_pg_error

if TYPE_CHECKING:
    from homesec.models.alert import AlertDecision
    from homesec.models.filter import FilterResult
    from homesec.models.vlm import AnalysisResult
    from homesec.interfaces import EventStore, StateStore

logger = logging.getLogger(__name__)

TResult = TypeVar("TResult")


class ClipRepository:
    """Coordinates state + event writes with best-effort retries."""

    def __init__(
        self,
        state_store: StateStore,
        event_store: EventStore,
        retry: RetryConfig | None = None,
        should_retry: Callable[[Exception], bool] | None = None,
    ) -> None:
        self._state = state_store
        self._events = event_store
        self._retry = retry or RetryConfig()
        self._should_retry = should_retry or is_retryable_pg_error
        self._max_attempts = max(1, int(self._retry.max_attempts))
        self._backoff_s = max(0.0, float(self._retry.backoff_s))

    async def initialize_clip(self, clip: Clip) -> ClipStateData:
        """Create initial state + record clip received event."""
        state = ClipStateData(
            camera_name=clip.camera_name,
            status="queued_local",
            local_path=str(clip.local_path),
        )

        event = ClipRecordedEvent(
            clip_id=clip.clip_id,
            timestamp=datetime.now(),
            camera_name=clip.camera_name,
            duration_s=clip.duration_s,
            source_type=clip.source_type,
        )

        await self._safe_upsert(clip.clip_id, state)
        await self._safe_append(event)
        return state

    async def record_upload_started(self, clip_id: str, dest_key: str, attempt: int) -> None:
        """Record upload start event."""
        await self._safe_append(
            UploadStartedEvent(
                clip_id=clip_id,
                timestamp=datetime.now(),
                dest_key=dest_key,
                attempt=attempt,
            )
        )

    async def record_upload_completed(
        self,
        clip_id: str,
        storage_uri: str,
        view_url: str | None,
        duration_ms: int,
        attempt: int = 1,
    ) -> ClipStateData | None:
        """Record upload completion + update state."""
        state = await self._load_state(clip_id, action="upload")
        if state is None:
            return None

        state.storage_uri = storage_uri
        state.view_url = view_url
        if state.status not in ("analyzed", "done", "error", "deleted"):
            state.status = "uploaded"

        event = UploadCompletedEvent(
            clip_id=clip_id,
            timestamp=datetime.now(),
            storage_uri=storage_uri,
            view_url=view_url,
            attempt=attempt,
            duration_ms=duration_ms,
        )

        await self._safe_upsert(clip_id, state)
        await self._safe_append(event)
        return state

    async def record_upload_failed(
        self,
        clip_id: str,
        error_message: str,
        error_type: str,
        *,
        attempt: int = 1,
        will_retry: bool = False,
    ) -> None:
        """Record upload failure event."""
        await self._safe_append(
            UploadFailedEvent(
                clip_id=clip_id,
                timestamp=datetime.now(),
                attempt=attempt,
                error_message=error_message,
                error_type=error_type,
                will_retry=will_retry,
            )
        )

    async def record_filter_started(self, clip_id: str, attempt: int) -> None:
        """Record filter start event."""
        await self._safe_append(
            FilterStartedEvent(
                clip_id=clip_id,
                timestamp=datetime.now(),
                attempt=attempt,
            )
        )

    async def record_filter_completed(
        self,
        clip_id: str,
        result: FilterResult,
        duration_ms: int,
        attempt: int = 1,
    ) -> ClipStateData | None:
        """Record filter completion + update state."""
        state = await self._load_state(clip_id, action="filter")
        if state is None:
            return None

        state.filter_result = result

        event = FilterCompletedEvent(
            clip_id=clip_id,
            timestamp=datetime.now(),
            detected_classes=result.detected_classes,
            confidence=result.confidence,
            model=result.model,
            sampled_frames=result.sampled_frames,
            attempt=attempt,
            duration_ms=duration_ms,
        )

        await self._safe_upsert(clip_id, state)
        await self._safe_append(event)
        return state

    async def record_filter_failed(
        self,
        clip_id: str,
        error_message: str,
        error_type: str,
        *,
        attempt: int = 1,
        will_retry: bool = False,
    ) -> ClipStateData | None:
        """Record filter failure + mark state as error."""
        event = FilterFailedEvent(
            clip_id=clip_id,
            timestamp=datetime.now(),
            attempt=attempt,
            error_message=error_message,
            error_type=error_type,
            will_retry=will_retry,
        )

        await self._safe_append(event)
        if will_retry:
            return None

        state = await self._load_state(clip_id, action="filter failure")
        if state is None:
            return None

        if state.status != "deleted":
            state.status = "error"
        await self._safe_upsert(clip_id, state)
        return state

    async def record_vlm_started(self, clip_id: str, attempt: int) -> None:
        """Record VLM start event."""
        await self._safe_append(
            VLMStartedEvent(
                clip_id=clip_id,
                timestamp=datetime.now(),
                attempt=attempt,
            )
        )

    async def record_vlm_completed(
        self,
        clip_id: str,
        result: AnalysisResult,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        duration_ms: int,
        attempt: int = 1,
    ) -> ClipStateData | None:
        """Record VLM completion + update state."""
        state = await self._load_state(clip_id, action="VLM")
        if state is None:
            return None

        state.analysis_result = result
        if state.status != "deleted":
            state.status = "analyzed"

        event = VLMCompletedEvent(
            clip_id=clip_id,
            timestamp=datetime.now(),
            risk_level=result.risk_level,
            activity_type=result.activity_type,
            summary=result.summary,
            analysis=result.analysis.model_dump(mode="json") if result.analysis else {},
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            attempt=attempt,
            duration_ms=duration_ms,
        )

        await self._safe_upsert(clip_id, state)
        await self._safe_append(event)
        return state

    async def record_vlm_failed(
        self,
        clip_id: str,
        error_message: str,
        error_type: str,
        *,
        attempt: int = 1,
        will_retry: bool = False,
    ) -> None:
        """Record VLM failure event."""
        await self._safe_append(
            VLMFailedEvent(
                clip_id=clip_id,
                timestamp=datetime.now(),
                attempt=attempt,
                error_message=error_message,
                error_type=error_type,
                will_retry=will_retry,
            )
        )

    async def record_vlm_skipped(self, clip_id: str, reason: str) -> None:
        """Record VLM skipped event."""
        await self._safe_append(
            VLMSkippedEvent(
                clip_id=clip_id,
                timestamp=datetime.now(),
                reason=reason,
            )
        )

    async def record_alert_decision(
        self,
        clip_id: str,
        decision: AlertDecision,
        detected_classes: list[str] | None,
        vlm_risk: str | None,
    ) -> ClipStateData | None:
        """Record alert decision + update state."""
        state = await self._load_state(clip_id, action="alert decision")
        if state is None:
            return None

        state.alert_decision = decision

        event = AlertDecisionMadeEvent(
            clip_id=clip_id,
            timestamp=datetime.now(),
            should_notify=decision.notify,
            reason=decision.notify_reason,
            detected_classes=detected_classes,
            vlm_risk=vlm_risk,
        )

        await self._safe_upsert(clip_id, state)
        await self._safe_append(event)
        return state

    async def record_notification_sent(
        self,
        clip_id: str,
        notifier_name: str,
        dedupe_key: str,
        attempt: int = 1,
    ) -> ClipStateData | None:
        """Record notification sent + mark state as done."""
        state = await self._load_state(clip_id, action="notification")
        if state is None:
            return None

        if state.status != "deleted":
            state.status = "done"

        event = NotificationSentEvent(
            clip_id=clip_id,
            timestamp=datetime.now(),
            notifier_name=notifier_name,
            dedupe_key=dedupe_key,
            attempt=attempt,
        )

        await self._safe_upsert(clip_id, state)
        await self._safe_append(event)
        return state

    async def record_notification_failed(
        self,
        clip_id: str,
        notifier_name: str,
        error_message: str,
        error_type: str,
        *,
        attempt: int = 1,
        will_retry: bool = False,
    ) -> None:
        """Record notification failure event."""
        await self._safe_append(
            NotificationFailedEvent(
                clip_id=clip_id,
                timestamp=datetime.now(),
                notifier_name=notifier_name,
                error_message=error_message,
                error_type=error_type,
                attempt=attempt,
                will_retry=will_retry,
            )
        )

    async def record_clip_deleted(
        self,
        clip_id: str,
        *,
        reason: str,
        run_id: str,
        deleted_local: bool,
        deleted_storage: bool,
    ) -> ClipStateData | None:
        """Mark clip as deleted and append a clip_deleted event."""
        state = await self._load_state(clip_id, action="clip delete")
        if state is None:
            return None

        state.status = "deleted"

        event = ClipDeletedEvent(
            clip_id=clip_id,
            timestamp=datetime.now(),
            camera_name=state.camera_name,
            reason=reason,
            run_id=run_id,
            local_path=state.local_path,
            storage_uri=state.storage_uri,
            deleted_local=deleted_local,
            deleted_storage=deleted_storage,
        )

        await self._safe_upsert(clip_id, state)
        await self._safe_append(event)
        return state

    async def record_clip_rechecked(
        self,
        clip_id: str,
        *,
        result: FilterResult,
        prior_filter: FilterResult | None,
        reason: str,
        run_id: str,
    ) -> ClipStateData | None:
        """Record a recheck result and update the clip state."""
        state = await self._load_state(clip_id, action="clip recheck")
        if state is None:
            return None
        if state.status == "deleted":
            return state

        state.filter_result = result

        event = ClipRecheckedEvent(
            clip_id=clip_id,
            timestamp=datetime.now(),
            camera_name=state.camera_name,
            reason=reason,
            run_id=run_id,
            prior_filter=prior_filter,
            recheck_filter=result,
        )

        await self._safe_upsert(clip_id, state)
        await self._safe_append(event)
        return state

    async def list_candidate_clips_for_cleanup(
        self,
        *,
        older_than_days: int | None,
        camera_name: str | None,
        batch_size: int,
        cursor: tuple[datetime, str] | None = None,
    ) -> list[tuple[str, ClipStateData, datetime]]:
        """List clip states eligible for cleanup."""
        try:
            return await self._run_with_retries(
                label="State store cleanup list",
                clip_id="cleanup",
                op=lambda: self._state.list_candidate_clips_for_cleanup(
                    older_than_days=older_than_days,
                    camera_name=camera_name,
                    batch_size=batch_size,
                    cursor=cursor,
                ),
            )
        except Exception as exc:
            logger.error(
                "State store cleanup list failed after retries: %s",
                exc,
                exc_info=exc,
            )
            return []

    async def mark_done(self, clip_id: str) -> ClipStateData | None:
        """Mark processing as done (no event)."""
        state = await self._load_state(clip_id, action="completion")
        if state is None:
            return None

        if state.status in ("done", "deleted"):
            return state

        state.status = "done"
        await self._safe_upsert(clip_id, state)
        return state

    async def _load_state(self, clip_id: str, *, action: str) -> ClipStateData | None:
        state = await self._safe_get(clip_id)
        if state is None:
            logger.error("Cannot update %s: clip %s not found", action, clip_id)
        return state

    async def _safe_get(self, clip_id: str) -> ClipStateData | None:
        try:
            return await self._run_with_retries(
                label="State store get",
                clip_id=clip_id,
                op=lambda: self._state.get(clip_id),
            )
        except Exception as exc:
            logger.error(
                "State store get failed for %s after retries: %s",
                clip_id,
                exc,
                exc_info=exc,
            )
            return None

    async def _safe_upsert(self, clip_id: str, state: ClipStateData) -> None:
        try:
            await self._run_with_retries(
                label="State store upsert",
                clip_id=clip_id,
                op=lambda: self._state.upsert(clip_id, state),
            )
        except Exception as exc:
            logger.error(
                "State store upsert failed for %s after retries: %s",
                clip_id,
                exc,
                exc_info=exc,
            )

    async def _safe_append(self, event: ClipLifecycleEvent) -> None:
        clip_id = event.clip_id
        try:
            await self._run_with_retries(
                label="Event store append",
                clip_id=clip_id,
                op=lambda: self._events.append(event),
            )
        except Exception as exc:
            logger.error(
                "Event store append failed for %s after retries: %s",
                clip_id,
                exc,
                exc_info=exc,
            )

    async def _run_with_retries(
        self,
        *,
        label: str,
        clip_id: str,
        op: Callable[[], Awaitable[TResult]],
    ) -> TResult:
        attempt = 1

        while True:
            try:
                return await op()
            except Exception as exc:
                if not self._should_retry(exc) or attempt >= self._max_attempts:
                    raise
                logger.warning(
                    "%s failed for %s (attempt %d/%d): %s",
                    label,
                    clip_id,
                    attempt,
                    self._max_attempts,
                    exc,
                )
                delay = self._backoff_s * (2 ** (attempt - 1))
                if delay > 0:
                    await asyncio.sleep(delay)
                attempt += 1
