"""Default alert policy plugin."""

from __future__ import annotations

from homesec.interfaces import AlertPolicy
from homesec.models.alert import AlertDecision
from homesec.models.config import DefaultAlertPolicySettings
from homesec.models.enums import RiskLevel
from homesec.models.filter import FilterResult
from homesec.models.vlm import AnalysisResult
from homesec.plugins.registry import PluginType, plugin


@plugin(plugin_type=PluginType.ALERT_POLICY, name="default")
class DefaultAlertPolicy(AlertPolicy):
    """Default alert policy implementation."""

    config_cls = DefaultAlertPolicySettings

    @classmethod
    def create(cls, config: DefaultAlertPolicySettings) -> AlertPolicy:
        return cls(config)

    def __init__(self, settings: DefaultAlertPolicySettings) -> None:
        self._settings = settings
        self._overrides = settings.overrides
        self._trigger_classes = list(settings.trigger_classes)

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

    def _risk_meets_threshold(self, actual: RiskLevel, threshold: RiskLevel) -> bool:
        """Check if actual risk level meets or exceeds threshold.

        Uses IntEnum comparison for natural ordering:
            RiskLevel.HIGH >= RiskLevel.MEDIUM  # True
        """
        return actual >= threshold

    def _filter_detected_trigger_classes(self, filter_result: FilterResult) -> bool:
        detected = set(filter_result.detected_classes)
        trigger = set(self._trigger_classes)
        return bool(detected & trigger)
