# Tasks

## 1. Specification and decisions

- [x] 1.1 Draft proposal, design, and `integration-contracts` spec delta from the
  existing MQTT notifier, plugin boundary, and Home Assistant MQTT discovery
  behavior.
- [ ] 1.2 Confirm the open questions in [design.md](design.md#open-questions) with
  Leonidas before treating the event taxonomy, discovery timing, and failure
  containment as settled.

## 2. Implementation

- [x] 2.1 Add `HomeAssistantConfig` and the discovery/event payload builders in
  `src/homesec/plugins/notifiers/mqtt_home_assistant.py`, validated with Pydantic.
- [x] 2.2 Extend `MQTTNotifier` with the optional `home_assistant` block: last
  will, availability, lazy per-camera discovery, event publication, reconnect
  republish, and offline-on-shutdown. Leave the alert publication unchanged.
- [x] 2.3 Add `integrations/home-assistant/homesec_alert_notification.yaml`.
- [x] 2.4 Add `docs/home-assistant.md`, link it from `docs/README.md` and the
  README, and add a commented example to `config/example.yaml`.

## 3. Verification

Each scenario maps to a test in
[test_mqtt_home_assistant.py](../../../tests/homesec/test_mqtt_home_assistant.py)
or [test_home_assistant_blueprint.py](../../../tests/homesec/test_home_assistant_blueprint.py).

- [x] 3.1 Disabled by default: `test_disabled_by_default_publishes_only_the_alert`.
- [x] 3.2 First alert, once-per-camera discovery, unchanged alert payload, event
  typing, and reconnect republish.
- [x] 3.3 Availability: last will, online on connect, offline on shutdown, and no
  availability publication when disabled.
- [x] 3.4 Topic and identity safety: config validation cases, slug safety and
  distinctness, wildcard-bearing camera name.
- [x] 3.5 Blueprint structure: declared inputs match references, risk options
  match `RiskLevel`, `unanalyzed` handled, mobile_app target with per-clip tag.
- [ ] 3.6 Blueprint behavior: the minimum-risk and `unanalyzed` conditions were
  checked only by a one-off render of the template across eight cases. Add a
  committed behavioral test, or exercise the blueprint on a live Home Assistant
  instance, before claiming these scenarios as verified.
- [ ] 3.7 Verify against a real Home Assistant and Mosquitto: entity creation
  after the first alert, event attributes, availability transitions, and the
  mobile_app device-action `data` fields. Record the Home Assistant version.
- [ ] 3.8 Measure whether `discovery_settle_seconds` default of 1.0 avoids losing
  the first event on a real instance.

## Validation and publication

Follow [AGENTS.md](../../../AGENTS.md): run `make typecheck` and `make check`
before publication. Database-writing checks need Leonidas's immediate approval
for that action, or an explicitly approved narrower scope. New tests follow
[TESTING.md](../../../TESTING.md) with Given/When/Then comments.

- [x] 4.1 `ruff check`, `ruff format --check`, and strict `mypy` pass.
- [x] 4.2 New and existing MQTT/notifier tests pass offline.
- [ ] 4.3 Full `make check`, including Postgres-backed tests, with approval.

## Evidence and archive

- [x] 5.1 Add the new scenarios to `docs/behavior-evidence.md` with accurate
  evidence levels and boundaries.
- [ ] 5.2 After review and verification, sync the delta into
  `openspec/specs/integration-contracts` and archive this change.
