# Design

## Context

The [MQTT notifier](../../../src/homesec/plugins/notifiers/mqtt.py) connects
to a broker, publishes `Alert.model_dump_json()` to `topic_template`, and exposes
`ping` and `shutdown`. The pipeline retries a notifier's `send` on failure and
fans out per entry (see the maintained
[analysis-alerts](../../specs/analysis-alerts/spec.md) and
[integration-contracts](../../specs/integration-contracts/spec.md) specs).
Home Assistant's MQTT `event` entity requires a JSON state payload with an
`event_type` from a declared `event_types` list, discards retained event
payloads, and subscribes to a state topic only after it processes discovery.

## Decisions

- **Extend the existing notifier rather than add a plugin.** The integration is
  MQTT transport and shares the connection, credentials, and QoS. A separate
  plugin would open a second broker connection and split configuration. The
  payload builders live in
  [mqtt_home_assistant.py](../../../src/homesec/plugins/notifiers/mqtt_home_assistant.py),
  which imports only core models, so the plugin boundary in
  [DESIGN.md](../../../DESIGN.md) is respected and core does not change.
- **Opt-in through an optional config block.** `home_assistant: null` keeps
  today's behavior exactly; the new fields are validated by Pydantic at the
  config boundary, including topic safety (no wildcards, no leading or
  trailing `/`).
- **A new event message instead of reshaping the alert payload.** Adding
  `event_type` to the alert JSON would change a published contract. The alert
  message stays byte-identical; the event message is separate and typed.
- **`event_type` is the risk level or `unanalyzed`.** Skipped or failed analysis
  produces no risk, and inventing one would be misleading; the alert already
  distinguishes it through `risk_level is None`.
- **Camera identity is a slug plus a hash.** Slugging alone maps "Front Door"
  and "front-door" to one entity. A short SHA-256 suffix keeps distinct names
  distinct, and the slug contains only `[a-z0-9_]`, which satisfies Home
  Assistant's ID rules and keeps wildcards out of topics.
- **Lazy per-camera discovery.** The notifier does not know the camera set at
  construction, so it announces a camera on its first alert, waits
  `discovery_settle_seconds`, then publishes the event. Entities therefore
  appear after the first alert. A reconnect increments a connection epoch,
  tracked per announced camera so the next alert republishes discovery; this
  covers a broker that lost retained data. Per-camera locks serialize a camera's
  discovery while allowing different cameras to settle concurrently. Discovery
  interrupted by reconnect retries for at most three connection generations.
- **Availability through last will plus an explicit offline.** A clean
  disconnect does not trigger the last will, so shutdown publishes a retained
  `offline` with a bounded wait. `online` is published from the connect callback
  without waiting, because that callback runs on the client's network thread.
- **Event messages are not retained.** Home Assistant discards retained event
  payloads, and retention would replay old alerts as new.
- **Unambiguous HA timestamps.** Event payloads attach the process-local timezone
  to naive pipeline timestamps without changing the legacy alert JSON. The
  blueprint checks payload time, rejects malformed/ambiguous/future times and
  suppresses attribute-only/restored-state/old-clip availability replays.
- **Acknowledgements.** Added HA discovery/event publishes wait at most ten
  seconds each and verify paho's acknowledgement flag. Clean offline waits at
  most two seconds and logs an unacknowledged result before cleanup. The legacy
  alert wait remains unchanged; a failed HA publish still fails the whole send.
- **Blueprint for push.** A blueprint reuses Home Assistant's companion-app
  notification path, needs no HomeSec runtime change, and keeps notification
  policy (minimum risk, unanalyzed handling) in Home Assistant.

## Risks / Trade-offs

- **First event can be lost.** The settle delay is a heuristic, not an
  acknowledgement. Documented as a limit.
- **Retry can duplicate.** If the event publish fails after the alert publish
  succeeded, the pipeline retry publishes both again. Consumers of the alert
  topic deduplicate on `dedupe_key`. This favors not losing a push over
  avoiding a duplicate.
- **Stale entities.** Retained discovery configs outlive removed cameras.
  Cleanup is manual and documented.
- **Verification boundaries.** Repository tests render the shipped blueprint
  and cover notifier transport failures. The [manual record](../../../docs/home-assistant-validation.md)
  verifies the original PR commit on real HA/Mosquitto/iPhone. Readiness fixes
  still need a live rerun; first-event reliability is not established by one success.
- **Availability ownership.** A prior random-client-ID process can publish a
  delayed offline will after its replacement is online. Stable client identity
  and multi-publisher ownership need a decision before claiming reliable recovery.
- **Settings UI.** The MQTT form preserves unknown keys when editing other
  fields but has no controls for `home_assistant`.

## Open Questions

For Leonidas, per [config.yaml](../../config.yaml):

1. Is `event_type` = risk level plus `unanalyzed` the right taxonomy, or should
   it express the alert reason?
2. Should discovery also run at startup from configured cameras, or on Home
   Assistant's birth message (`homeassistant/status`), instead of lazily?
3. Should a Home Assistant publish failure fail the whole send, as now, or be
   contained so the original alert stays delivered?
4. Is `integrations/home-assistant/` the right home for the blueprint, and
   should it also be offered through HACS or the Home Assistant blueprint
   gallery?
5. Should this add a settings-UI form for the block in a follow-up?

## Migration Plan

No migration. Existing configurations are unaffected because the block is
optional. Enabling it creates retained discovery topics at the broker; removing
the block stops new events but leaves those topics until cleared.
