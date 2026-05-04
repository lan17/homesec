# Mqtt Notifier Alert

> 66 nodes · cohesion 0.07

## Key Concepts

- **MQTTNotifier** (37 connections) — `src/homesec/plugins/notifiers/mqtt.py`
- **MQTTConfig** (31 connections) — `src/homesec/plugins/notifiers/mqtt.py`
- **_FakeClient** (27 connections) — `tests/homesec/test_mqtt_notifier.py`
- **test_mqtt_notifier.py** (27 connections) — `tests/homesec/test_mqtt_notifier.py`
- **MQTTAuthConfig** (18 connections) — `src/homesec/plugins/notifiers/mqtt.py`
- **_FakeClientNoConnect** (9 connections) — `tests/homesec/test_mqtt_notifier.py`
- **_FakePublishResult** (8 connections) — `tests/homesec/test_mqtt_notifier.py`
- **_sample_alert()** (7 connections) — `tests/homesec/test_mqtt_notifier.py`
- **test_mqtt_notifier_publishes_alert()** (7 connections) — `tests/homesec/test_mqtt_notifier.py`
- **test_mqtt_notifier_raises_on_timeout()** (6 connections) — `tests/homesec/test_mqtt_notifier.py`
- **TestMQTTNotifierAuth** (6 connections) — `tests/homesec/test_mqtt_notifier.py`
- **TestMQTTNotifierConfig** (6 connections) — `tests/homesec/test_mqtt_notifier.py`
- **TestMQTTNotifierConnection** (6 connections) — `tests/homesec/test_mqtt_notifier.py`
- **TestMQTTNotifierPing** (6 connections) — `tests/homesec/test_mqtt_notifier.py`
- **TestMQTTNotifierShutdown** (6 connections) — `tests/homesec/test_mqtt_notifier.py`
- **test_credentials_not_set_when_partial()** (5 connections) — `tests/homesec/test_mqtt_notifier.py`
- **test_custom_qos_and_retain()** (5 connections) — `tests/homesec/test_mqtt_notifier.py`
- **test_send_fails_after_shutdown()** (5 connections) — `tests/homesec/test_mqtt_notifier.py`
- **test_topic_template_formatting()** (5 connections) — `tests/homesec/test_mqtt_notifier.py`
- **test_warning_when_password_env_missing()** (5 connections) — `tests/homesec/test_mqtt_notifier.py`
- **test_warning_when_username_env_missing()** (5 connections) — `tests/homesec/test_mqtt_notifier.py`
- **mqtt.py** (5 connections) — `src/homesec/plugins/notifiers/mqtt.py`
- **test_no_auth_when_not_configured()** (4 connections) — `tests/homesec/test_mqtt_notifier.py`
- **test_ping_returns_false_after_shutdown()** (4 connections) — `tests/homesec/test_mqtt_notifier.py`
- **test_ping_returns_true_when_connected()** (4 connections) — `tests/homesec/test_mqtt_notifier.py`
- *... and 41 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/plugins/notifiers/mqtt.py`
- `tests/homesec/test_mqtt_notifier.py`

## Audit Trail

- EXTRACTED: 177 (55%)
- INFERRED: 142 (45%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*