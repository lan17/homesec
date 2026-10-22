# Spec Delta

## ADDED Requirements

### Requirement: Home Assistant MQTT discovery

When the MQTT notifier is configured with a `home_assistant` block, HomeSec
SHALL publish a retained discovery configuration for each camera's alert event
entity before publishing that camera's first event on a connection, publish
each alert as a non-retained event message whose `event_type` is the alert's
risk level or `unanalyzed` when the alert has no risk level, and retain the
existing alert publication unchanged. Without the block, HomeSec SHALL publish
only the existing alert message.

#### Scenario: Disabled by default

- **GIVEN** an MQTT notifier without a `home_assistant` block
- **WHEN** it sends an alert
- **THEN** only the existing alert topic receives a message
- **AND** no last will, availability, discovery, or event message is published

#### Scenario: First alert for a camera announces the entity

- **GIVEN** a connected MQTT notifier with `home_assistant` configured
- **WHEN** it sends the first alert for a camera
- **THEN** a retained discovery configuration for that camera is published
  before its event message
- **AND** the event message is not retained
- **AND** the alert topic still carries the unchanged alert JSON

#### Scenario: Later alerts do not repeat discovery

- **GIVEN** a camera whose discovery was already published on the current
  connection
- **WHEN** another alert is sent for that camera
- **THEN** only the alert and event messages are published

#### Scenario: Event type reflects analysis

- **GIVEN** alerts with risk levels `low`, `medium`, `high`, or `critical`, and
  an alert with no risk level
- **WHEN** event messages are built
- **THEN** `event_type` equals the risk level name, or `unanalyzed` when absent
- **AND** an absent risk level is not replaced with an invented one

#### Scenario: Reconnect republishes discovery

- **GIVEN** a camera already announced before the client reconnects
- **WHEN** the next alert for that camera is sent after reconnecting
- **THEN** its discovery configuration is published again

#### Scenario: Reconnect interrupts discovery

- **GIVEN** discovery was published on a connection that changes before its
  settle delay completes
- **WHEN** the notifier prepares that camera's event
- **THEN** it republishes discovery on the new connection before the event
- **AND** after three interrupted attempts it reports failure rather than
  caching an old-connection announcement or retrying indefinitely

#### Scenario: Home Assistant publication is not acknowledged

- **GIVEN** the original alert has been delivered but HA discovery or event
  publication is not acknowledged within ten seconds
- **WHEN** that publication wait expires
- **THEN** the send reports failure under the existing retry policy
- **AND** an unacknowledged discovery is not cached as successful
- **AND** a caller retry can republish the original alert

### Requirement: Home Assistant availability

When the `home_assistant` block is configured, HomeSec SHALL register a retained
`offline` last will on the configured availability topic, publish a retained
`online` message when it connects, and publish a retained `offline` message
during clean shutdown without waiting indefinitely for broker acknowledgement.

#### Scenario: Connect announces availability

- **GIVEN** an MQTT notifier with `home_assistant` configured
- **WHEN** it connects to the broker
- **THEN** `offline` is registered as the retained last will
- **AND** `online` is published retained to the availability topic

#### Scenario: Clean shutdown announces offline

- **GIVEN** a connected MQTT notifier with `home_assistant` configured
- **WHEN** it shuts down
- **THEN** a retained `offline` message is published to the availability topic

#### Scenario: Offline acknowledgement does not arrive

- **GIVEN** the broker does not acknowledge clean-shutdown availability
- **WHEN** the two-second publication wait expires
- **THEN** HomeSec logs the unacknowledged result and continues transport cleanup

### Requirement: Home Assistant topic and identity safety

Home Assistant configuration SHALL reject topic settings that contain MQTT
wildcards or NUL, that begin or end with `/`, or whose event topic template
lacks the `{camera_slug}` placeholder or uses unknown placeholders. Camera
identities used in discovery, entity, device, and event topics SHALL contain
only lowercase letters, digits, and underscores, and SHALL remain distinct for
camera names that differ only by case or punctuation.

#### Scenario: Unpublishable topic setting

- **GIVEN** a `home_assistant` block with a wildcard in a topic setting
- **WHEN** the configuration is validated
- **THEN** validation fails

#### Scenario: Escaped placeholder does not separate cameras

- **GIVEN** an event topic template uses only an escaped `{{camera_slug}}`
- **WHEN** the configuration is validated
- **THEN** validation fails because different slugs would produce the same topic

#### Scenario: Names that normalize alike stay distinct

- **GIVEN** cameras named `Front Door` and `front-door`
- **WHEN** their discovery identities are built
- **THEN** their entity IDs, device identifiers, and topics differ

#### Scenario: Unsafe camera name

- **GIVEN** a camera name containing `/`, `+`, or `#`
- **WHEN** an alert for it is sent with `home_assistant` configured
- **THEN** the discovery and event topics contain no wildcard characters

### Requirement: Home Assistant event time

HA event timestamps SHALL preserve the alert instant and include a timezone
offset, interpreting naive alert times in HomeSec's process-local timezone.
The legacy alert JSON SHALL remain unchanged. Blueprint freshness SHALL use
payload `ts` rather than Home Assistant receipt time.

#### Scenario: Pipeline timestamp has no timezone

- **GIVEN** a pipeline alert with a naive process-local timestamp
- **WHEN** its HA event payload is built
- **THEN** `ts` includes the local timezone offset and preserves the same instant
- **AND** the original alert payload is unchanged

### Requirement: Home Assistant push-notification blueprint

HomeSec SHALL ship a Home Assistant automation blueprint that notifies a
companion-app device from a camera's alert event entity. It SHALL ignore
unavailable or unknown states and events older than five minutes, notify only
for risk at or above a configurable minimum, treat `unanalyzed` alerts
according to a configurable option, and use a per-clip notification tag.

#### Scenario: Risk below the minimum is ignored

- **GIVEN** a minimum risk of `medium`
- **WHEN** a `low` event arrives
- **THEN** no notification is sent

#### Scenario: Risk at or above the minimum notifies

- **GIVEN** a minimum risk of `medium`
- **WHEN** a `medium`, `high`, or `critical` event arrives
- **THEN** a notification is sent with that clip's tag

#### Scenario: Unanalyzed alerts follow the option

- **GIVEN** an `unanalyzed` event
- **WHEN** notification of unanalyzed alerts is enabled or disabled
- **THEN** a notification is sent only when it is enabled

#### Scenario: Delayed or malformed payload time

- **GIVEN** an event was just received but its payload `ts` is at least five
  minutes old, future-dated, absent, invalid or lacks a timezone offset
- **WHEN** the blueprint evaluates its conditions
- **THEN** no notification is sent

#### Scenario: Availability recovery re-exposes an old clip

- **GIVEN** an entity was unavailable and recovers with the same clip ID
- **WHEN** its previous event timestamp becomes visible again
- **THEN** the availability transition does not send another notification

#### Scenario: A genuine new clip arrives after an outage

- **GIVEN** an entity was unavailable and now exposes a different clip ID
- **WHEN** the fresh event meets the risk or unanalyzed policy
- **THEN** the new clip may notify

#### Scenario: Restored state or attribute-only updates

- **GIVEN** a state is restored without a prior state, or attributes change
  without changing the event timestamp
- **WHEN** the blueprint's state trigger fires
- **THEN** no notification is sent

#### Scenario: First event after unknown state

- **GIVEN** a new event entity has unknown state
- **WHEN** a fresh event arrives and meets the notification policy
- **THEN** it is eligible for notification
