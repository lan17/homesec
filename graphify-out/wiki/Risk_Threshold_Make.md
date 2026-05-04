# Risk Threshold Make

> 40 nodes · cohesion 0.06

## Key Concepts

- **DefaultAlertPolicy** (31 connections) — `src/homesec/plugins/alert_policies/default.py`
- **NoopAlertPolicy** (10 connections) — `src/homesec/plugins/alert_policies/noop.py`
- **_make_policy()** (10 connections) — `tests/homesec/test_alert_policy.py`
- **NoopAlertPolicySettings** (9 connections) — `src/homesec/plugins/alert_policies/noop.py`
- **test_alert_policy.py** (8 connections) — `tests/homesec/test_alert_policy.py`
- **TestFtpSourceIntegration** (6 connections) — `tests/homesec/test_integration.py`
- **TestFullPipelineIntegration** (6 connections) — `tests/homesec/test_integration.py`
- **.should_notify()** (5 connections) — `src/homesec/plugins/alert_policies/default.py`
- **test_activity_type_override_notifies()** (5 connections) — `tests/homesec/test_alert_policy.py`
- **test_risk_threshold_allows_when_above()** (5 connections) — `tests/homesec/test_alert_policy.py`
- **test_risk_threshold_blocks_when_below()** (5 connections) — `tests/homesec/test_alert_policy.py`
- **default.py** (5 connections) — `src/homesec/plugins/alert_policies/default.py`
- **test_no_analysis_no_triggers_skips_notification()** (4 connections) — `tests/homesec/test_alert_policy.py`
- **test_notify_on_motion_always_notifies()** (4 connections) — `tests/homesec/test_alert_policy.py`
- **test_vlm_failure_falls_back_to_filter_triggers()** (4 connections) — `tests/homesec/test_alert_policy.py`
- **._filter_detected_trigger_classes()** (3 connections) — `src/homesec/plugins/alert_policies/default.py`
- **.make_decision()** (3 connections) — `src/homesec/plugins/alert_policies/default.py`
- **._risk_meets_threshold()** (3 connections) — `src/homesec/plugins/alert_policies/default.py`
- **noop.py** (3 connections) — `src/homesec/plugins/alert_policies/noop.py`
- **._policy_for()** (2 connections) — `src/homesec/plugins/alert_policies/default.py`
- **.make_decision()** (2 connections) — `src/homesec/plugins/alert_policies/noop.py`
- **AlertPolicy** (2 connections)
- **create()** (1 connections) — `src/homesec/plugins/alert_policies/default.py`
- **Default alert policy plugin.** (1 connections) — `src/homesec/plugins/alert_policies/default.py`
- **Default alert policy implementation.** (1 connections) — `src/homesec/plugins/alert_policies/default.py`
- *... and 15 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/plugins/alert_policies/default.py`
- `src/homesec/plugins/alert_policies/noop.py`
- `tests/homesec/test_alert_policy.py`
- `tests/homesec/test_integration.py`

## Audit Trail

- EXTRACTED: 98 (64%)
- INFERRED: 55 (36%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*