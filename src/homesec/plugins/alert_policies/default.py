"""Default alert policy plugin."""

from __future__ import annotations

from homesec.interfaces import AlertPolicy
from homesec.models.alert import AlertDecision
from homesec.models.config import AlertPolicyOverrides, DefaultAlertPolicySettings
from homesec.models.filter import FilterResult
from homesec.models.vlm import AnalysisResult

# Risk level ordering for comparison
RISK_LEVELS = {"low": 0, "medium": 1, "high": 2, "critical": 3}


class DefaultAlertPolicy(AlertPolicy):
    """Default alert policy implementation."""

    def __init__(
        self,
        settings: DefaultAlertPolicySettings,
        overrides: dict[str, AlertPolicyOverrides],
        trigger_classes: list[str],
    ) -> None:
        self._settings = settings
        self._overrides = overrides
        self._trigger_classes = list(trigger_classes)

    def should_notify(
        self,
        camera_name: str,
        filter_result: FilterResult | None,
        analysis: AnalysisResult | None,
    ) -> tuple[bool, str]:
        policy = self._policy_for(camera_name)

        # Check notify_on_motion first (bypasses VLM)
        if policy.notify_on_motion:
            return True, "notify_on_motion=true"

        # If no analysis, check if filter detected trigger classes (VLM failure fallback)
        if analysis is None:
            if filter_result and self._filter_detected_trigger_classes(filter_result):
                return True, "filter_detected_trigger_vlm_failed"
            return False, "no_analysis"

        # Check risk level threshold
        if self._risk_meets_threshold(analysis.risk_level, policy.min_risk_level):
            return True, f"risk_level={analysis.risk_level}"

        # Check activity type list
        if analysis.activity_type in policy.notify_on_activity_types:
            return True, f"activity_type={analysis.activity_type} (per-camera)"

        return False, "below_threshold"

    def make_decision(
        self,
        camera_name: str,
        filter_result: FilterResult | None,
        analysis: AnalysisResult | None,
    ) -> AlertDecision:
        notify, reason = self.should_notify(camera_name, filter_result, analysis)
        return AlertDecision(notify=notify, notify_reason=reason)

    def _policy_for(self, camera_name: str) -> DefaultAlertPolicySettings:
        overrides = self._overrides.get(camera_name)
        if overrides is None:
            return self._settings
        merged = {
            **self._settings.model_dump(),
            **overrides.model_dump(exclude_none=True),
        }
        return DefaultAlertPolicySettings.model_validate(merged)

    def _risk_meets_threshold(self, actual: str, threshold: str) -> bool:
        return RISK_LEVELS.get(actual, 0) >= RISK_LEVELS.get(threshold, 0)

    def _filter_detected_trigger_classes(self, filter_result: FilterResult) -> bool:
        detected = set(filter_result.detected_classes)
        trigger = set(self._trigger_classes)
        return bool(detected & trigger)


# Plugin registration
from pydantic import BaseModel
from homesec.plugins.alert_policies import AlertPolicyPlugin, alert_policy_plugin
from homesec.interfaces import AlertPolicy


@alert_policy_plugin(name="default")
def default_alert_policy_plugin() -> AlertPolicyPlugin:
    """Default alert policy plugin factory.

    Returns:
        AlertPolicyPlugin for default risk-based alert policy
    """
    from homesec.models.config import AlertPolicyOverrides, DefaultAlertPolicySettings

    def factory(
        cfg: BaseModel,
        overrides: dict[str, AlertPolicyOverrides],
        trigger_classes: list[str],
    ) -> AlertPolicy:
        settings = DefaultAlertPolicySettings.model_validate(cfg)
        return DefaultAlertPolicy(settings, overrides, trigger_classes)

    return AlertPolicyPlugin(
        name="default",
        config_model=DefaultAlertPolicySettings,
        factory=factory,
    )
