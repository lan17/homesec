"""No-op alert policy."""

from __future__ import annotations

from pydantic import BaseModel

from homesec.interfaces import AlertPolicy
from homesec.models.alert import AlertDecision
from homesec.models.filter import FilterResult
from homesec.models.vlm import AnalysisResult
from homesec.plugins.registry import PluginType, plugin


class NoopAlertPolicySettings(BaseModel):
    """Settings for no-op alert policy (empty - no configuration needed)."""

    model_config = {"extra": "forbid"}


@plugin(plugin_type=PluginType.ALERT_POLICY, name="noop")
class NoopAlertPolicy(AlertPolicy):
    """Alert policy that never notifies."""

    config_cls = NoopAlertPolicySettings

    @classmethod
    def create(cls, config: NoopAlertPolicySettings) -> AlertPolicy:
        return cls()

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
