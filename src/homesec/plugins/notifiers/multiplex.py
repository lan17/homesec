"""Fan-out notifier that sends alerts to multiple notifiers in parallel."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from homesec.interfaces import Notifier
from homesec.models.alert import Alert

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NotifierEntry:
    """Notifier entry with a name for logging."""

    name: str
    notifier: Notifier


class MultiplexNotifier(Notifier):
    """Send notifications to multiple notifiers in parallel."""

    def __init__(self, entries: list[NotifierEntry]) -> None:
        if not entries:
            raise ValueError("MultiplexNotifier requires at least one notifier")
        self._entries = list(entries)
        self._shutdown_called = False

    async def send(self, alert: Alert) -> None:
        """Send alert notification via all notifiers in parallel."""
        if self._shutdown_called:
            raise RuntimeError("Notifier has been shut down")

        results = await self._call_all(lambda notifier: notifier.send(alert))

        failures: list[tuple[str, BaseException]] = []
        for entry, result in results:
            match result:
                case BaseException() as err:
                    failures.append((entry.name, err))
                    logger.error(
                        "Notifier send failed: notifier=%s error=%s",
                        entry.name,
                        err,
                        exc_info=err,
                    )

        if failures:
            detail = "; ".join(
                f"{name}={type(error).__name__}: {error}" for name, error in failures
            )
            raise RuntimeError(f"Notifier failures: {detail}") from failures[0][1]

    async def ping(self) -> bool:
        """Health check - verify all notifiers are reachable."""
        if self._shutdown_called:
            return False

        results = await self._call_all(lambda notifier: notifier.ping())

        healthy = True
        for entry, result in results:
            match result:
                case bool() as ok:
                    if not ok:
                        healthy = False
                        logger.warning("Notifier ping failed: notifier=%s", entry.name)
                case BaseException() as err:
                    healthy = False
                    logger.warning(
                        "Notifier ping failed: notifier=%s error=%s",
                        entry.name,
                        err,
                    )

        return healthy

    async def shutdown(self, timeout: float | None = None) -> None:
        """Cleanup resources for all notifiers."""
        _ = timeout
        if self._shutdown_called:
            return
        self._shutdown_called = True

        results = await self._call_all(lambda notifier: notifier.shutdown())
        for entry, result in results:
            match result:
                case BaseException() as err:
                    logger.warning(
                        "Notifier close failed: notifier=%s error=%s",
                        entry.name,
                        err,
                    )

    async def _call_all(
        self, func: Callable[[Notifier], Coroutine[Any, Any, object]]
    ) -> list[tuple[NotifierEntry, object | BaseException]]:
        tasks: list[asyncio.Task[object]] = [
            asyncio.create_task(func(entry.notifier)) for entry in self._entries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return list(zip(self._entries, results, strict=True))
