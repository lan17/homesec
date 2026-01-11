"""Tests for alert policy decisions."""

from __future__ import annotations

from homesec.models.config import DefaultAlertPolicySettings
from homesec.models.filter import FilterResult
from homesec.models.vlm import AnalysisResult
from homesec.plugins.alert_policies.default import DefaultAlertPolicy


def test_notify_on_motion_always_notifies() -> None:
    """notify_on_motion should always notify even without analysis."""
    # Given notify_on_motion is enabled
    policy = DefaultAlertPolicy(
        DefaultAlertPolicySettings(min_risk_level="high", notify_on_motion=True),
        overrides={},
        trigger_classes=["person"],
    )

    # When analysis is missing and filter results are empty
    notify, reason = policy.should_notify(
        "front_door",
        FilterResult(detected_classes=[], confidence=0.0, model="mock", sampled_frames=1),
        analysis=None,
    )

    # Then notify is True with notify_on_motion reason
    assert notify is True
    assert reason == "notify_on_motion=true"


def test_risk_threshold_blocks_when_below() -> None:
    """Risk threshold should block notifications below minimum."""
    # Given a high risk threshold
    policy = DefaultAlertPolicy(
        DefaultAlertPolicySettings(min_risk_level="high", notify_on_motion=False),
        overrides={},
        trigger_classes=["person"],
    )
    analysis = AnalysisResult(
        risk_level="medium",
        activity_type="person_walking",
        summary="Person walking past.",
    )

    # When risk is below threshold
    notify, reason = policy.should_notify(
        "front_door",
        FilterResult(detected_classes=["person"], confidence=0.8, model="mock", sampled_frames=1),
        analysis=analysis,
    )

    # Then notify is False with below_threshold reason
    assert notify is False
    assert reason == "below_threshold"


def test_risk_threshold_allows_when_above() -> None:
    """Risk threshold should allow notifications at or above minimum."""
    # Given a medium risk threshold
    policy = DefaultAlertPolicy(
        DefaultAlertPolicySettings(min_risk_level="medium", notify_on_motion=False),
        overrides={},
        trigger_classes=["person"],
    )
    analysis = AnalysisResult(
        risk_level="high",
        activity_type="person_loitering",
        summary="Person loitering suspiciously.",
    )

    # When risk is above threshold
    notify, reason = policy.should_notify(
        "front_door",
        FilterResult(detected_classes=["person"], confidence=0.9, model="mock", sampled_frames=1),
        analysis=analysis,
    )

    # Then notify is True with risk_level reason
    assert notify is True
    assert reason == "risk_level=high"


def test_activity_type_override_notifies() -> None:
    """Activity type allow-list should trigger notifications."""
    # Given an activity type override list
    policy = DefaultAlertPolicy(
        DefaultAlertPolicySettings(
            min_risk_level="high",
            notify_on_motion=False,
            notify_on_activity_types=["delivery"],
        ),
        overrides={},
        trigger_classes=["person"],
    )
    analysis = AnalysisResult(
        risk_level="low",
        activity_type="delivery",
        summary="Package delivery.",
    )

    # When activity type matches the override list
    notify, reason = policy.should_notify(
        "front_door",
        FilterResult(detected_classes=["person"], confidence=0.8, model="mock", sampled_frames=1),
        analysis=analysis,
    )

    # Then notify is True with activity_type reason
    assert notify is True
    assert reason == "activity_type=delivery (per-camera)"


def test_vlm_failure_falls_back_to_filter_triggers() -> None:
    """Missing analysis should still notify if filter hits trigger classes."""
    # Given no analysis and filter detects trigger class
    policy = DefaultAlertPolicy(
        DefaultAlertPolicySettings(min_risk_level="high", notify_on_motion=False),
        overrides={},
        trigger_classes=["person"],
    )
    filter_result = FilterResult(
        detected_classes=["person"],
        confidence=0.9,
        model="mock",
        sampled_frames=1,
    )

    # When analysis is None
    notify, reason = policy.should_notify("front_door", filter_result, analysis=None)

    # Then notify falls back to filter trigger detection
    assert notify is True
    assert reason == "filter_detected_trigger_vlm_failed"


def test_no_analysis_no_triggers_skips_notification() -> None:
    """Missing analysis should skip when no trigger classes are detected."""
    # Given no analysis and no trigger classes detected
    policy = DefaultAlertPolicy(
        DefaultAlertPolicySettings(min_risk_level="low", notify_on_motion=False),
        overrides={},
        trigger_classes=["person"],
    )
    filter_result = FilterResult(
        detected_classes=["dog"],
        confidence=0.6,
        model="mock",
        sampled_frames=1,
    )

    # When analysis is None
    notify, reason = policy.should_notify("front_door", filter_result, analysis=None)

    # Then notify is False with no_analysis reason
    assert notify is False
    assert reason == "no_analysis"
