"""Mock notifier for testing."""

from __future__ import annotations

import asyncio

from homesec.errors import NotifyError
from homesec.models.alert import Alert


class MockNotifier:
    """Mock implementation of Notifier interface for testing.
    
    Tracks all sent alerts in a list for test assertions.
    Supports configurable failure injection and delays.
    """

    def __init__(
        self,
        simulate_failure: bool = False,
        delay_s: float = 0.0,
    ) -> None:
        """Initialize mock notifier.
        
        Args:
            simulate_failure: If True, send() raises NotifyError
            delay_s: Artificial delay before returning
        """
        self.simulate_failure = simulate_failure
        self.delay_s = delay_s
        self.sent_alerts: list[Alert] = []
        self.shutdown_called = False

    async def send(self, alert: Alert) -> None:
        """Send notification (mock implementation - stores in list)."""
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)

        if self.simulate_failure:
            raise NotifyError(
                clip_id=alert.clip_id,
                notifier_name="mock_notifier",
                cause=RuntimeError("Simulated notifier failure"),
            )

        self.sent_alerts.append(alert)

    async def ping(self) -> bool:
        """Health check (mock implementation)."""
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)

        return not self.simulate_failure

    async def shutdown(self, timeout: float | None = None) -> None:
        """Cleanup resources (no-op for mock)."""
        _ = timeout
        self.shutdown_called = True
