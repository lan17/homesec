"""No-op alert policy."""

from __future__ import annotations

from pydantic import BaseModel

from homesec.interfaces import AlertPolicy
from homesec.models.alert import AlertDecision
from homesec.models.config import AlertPolicyOverrides
from homesec.models.filter import FilterResult
from homesec.models.vlm import AnalysisResult


class NoopAlertPolicy(AlertPolicy):
    """Alert policy that never notifies."""

    def should_notify(
        self,
        camera_name: str,
        filter_result: FilterResult | None,
        analysis: AnalysisResult | None,
    ) -> tuple[bool, str]:
        return False, "alert_policy_disabled"

    def make_decision(
        self,
        camera_name: str,
        filter_result: FilterResult | None,
        analysis: AnalysisResult | None,
    ) -> AlertDecision:
        return AlertDecision(notify=False, notify_reason="alert_policy_disabled")


# Plugin registration
from homesec.interfaces import AlertPolicy
from homesec.plugins.alert_policies import AlertPolicyPlugin, alert_policy_plugin


@alert_policy_plugin(name="noop")
def noop_alert_policy_plugin() -> AlertPolicyPlugin:
    """Noop alert policy plugin that never sends alerts.

    Returns:
        AlertPolicyPlugin for no-op alert policy
    """

    def factory(
        cfg: BaseModel,
        overrides: dict[str, AlertPolicyOverrides],
        trigger_classes: list[str],
    ) -> AlertPolicy:
        # NoopAlertPolicy doesn't use any config
        return NoopAlertPolicy()

    return AlertPolicyPlugin(
        name="noop",
        config_model=BaseModel,  # No config needed
        factory=factory,
    )
