# Proposal

## Why

HomeSec's MQTT notifier publishes one alert JSON message per camera. Home
Assistant users can react to it only with hand-written MQTT triggers and
templates, and the payload has no `event_type`, so it cannot back a Home
Assistant `event` entity. There is no availability signal and no packaged way
to send companion-app push notifications.

## What Changes

- Add an opt-in `home_assistant` block to the MQTT notifier configuration. When
  absent, behavior is unchanged.
- When present, publish retained MQTT discovery configs that create an
  **Alerts** `event` entity per camera, a non-retained event message per alert
  (`event_type` is the risk level or `unanalyzed`), and retained
  online/offline availability with a last will.
- Ship a Home Assistant blueprint that sends a companion-app push notification
  from that entity, filtered by minimum risk.
- Document setup, limits, and tests in `docs/home-assistant.md`.

The existing alert topic and JSON payload, retry behavior, and plugin
boundaries do not change.

## Affected Capabilities

- `integration-contracts`: ADDED requirements for the Home Assistant MQTT
  integration, topic safety, and the blueprint. The existing "MQTT alert
  publication" requirement is unchanged.

## Unresolved Choices

These need Leonidas's confirmation before they are treated as settled; see
[design.md](design.md#open-questions).

- Event taxonomy: `event_type` is the risk level plus `unanalyzed`.
- Discovery is lazy (first alert per camera), not at startup.
- Whether to also subscribe to Home Assistant's birth message.
- Whether a Home Assistant failure should fail the whole MQTT send.
