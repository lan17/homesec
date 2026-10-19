# Home Assistant integration

HomeSec's MQTT notifier can expose each camera to Home Assistant as an **Alerts**
event entity and drive companion-app push notifications through a blueprint.
It is off unless the notifier config has a `home_assistant` block, and it never
changes the existing alert message on `topic_template`.

## What it publishes

With `home_assistant` set, the MQTT notifier adds:

| Topic | Content | Retained |
| --- | --- | --- |
| `<discovery_prefix>/event/homesec_<camera>/config` | Home Assistant MQTT discovery config for the camera's event entity | Yes |
| `event_topic_template` (default `homecam/events/<camera>`) | One event per alert: `event_type` plus clip context | No |
| `availability_topic` (default `homecam/status`) | `online` / `offline`; `offline` is also the broker last will | Yes |

`<camera>` is a lowercase slug of the camera name plus a short hash, so camera
names that differ only in case or punctuation stay distinct and wildcard or `/`
characters never reach a topic.

`event_type` is the alert's risk level (`low`, `medium`, `high`, `critical`), or
`unanalyzed` when analysis was skipped or failed. HomeSec does not invent a risk
level in that case. Other fields (`clip_id`, `camera_name`, `summary`,
`activity_type`, `view_url`, `notify_reason`, `ts`, `upload_failed`,
`vlm_failed`) become entity attributes.

## Configure

Add the block to the MQTT notifier; every field has a default:

```yaml
notifiers:
  - backend: mqtt
    config:
      host: "homeassistant.local"
      port: 1883
      auth:
        username_env: MQTT_USERNAME
        password_env: MQTT_PASSWORD
      home_assistant:
        discovery_prefix: homeassistant    # must match Home Assistant's MQTT discovery prefix
        event_topic_template: "homecam/events/{camera_slug}"
        availability_topic: homecam/status
        discovery_settle_seconds: 1.0
```

Prerequisites in Home Assistant: the MQTT integration with a broker (for
example the Mosquitto app) and a broker login for HomeSec. The settings UI
preserves the `home_assistant` block when you edit other MQTT fields, but it
has no fields for it; edit the YAML.

Camera entities appear after the first alert for that camera. HomeSec announces
a camera once per broker connection and again after a reconnect, so a broker
that loses retained data recovers on the next alert.

## Push notifications

Import [the blueprint](../integrations/home-assistant/homesec_alert_notification.yaml)
in Home Assistant under **Settings > Automations & scenes > Blueprints**, then
create an automation from it:

1. Choose the camera's **Alerts** event entity.
2. Choose the phone (a device with the companion app).
3. Set the minimum risk level and whether to notify for `unanalyzed` alerts.

The notification title shows the camera and risk, the message is the VLM
summary, tapping it opens `view_url` when present, and `high`/`critical` alerts
use the iOS `time-sensitive` interruption level. Notifications use a per-clip
`tag`, so repeats replace instead of stacking. Create one automation per
camera.

## Limits

- **First-alert timing.** Home Assistant subscribes to the event topic only
  after it processes discovery. HomeSec waits `discovery_settle_seconds`
  after publishing discovery; a slow Home Assistant can still miss the very
  first event for a camera. Event messages are never retained because Home
  Assistant discards retained event payloads.
- **Delivery is best-effort.** A failed publish fails that notifier's send, like
  any other MQTT alert. If the event publish fails after the alert publish
  succeeded, the pipeline's retry publishes both again, so consumers of the
  original topic should deduplicate on `dedupe_key`. Alerts older than five
  minutes are ignored by the blueprint, which prevents stale replays after a
  Home Assistant restart.
- **Renamed or removed cameras.** Discovery configs are retained at the broker
  and are not cleaned up automatically. Remove stale entities in Home Assistant
  or clear the retained discovery topic.
- **Push delivery.** Notifications go through the Home Assistant companion app
  and its cloud relay or your own remote access, not through HomeSec.
- **Blueprint verification.** Repository tests check the blueprint's structure
  and its alignment with HomeSec's risk levels. It has not been exercised
  against a live Home Assistant instance by the automated suite.

## Test from the command line

With `mosquitto_sub` installed:

```sh
mosquitto_sub -h <broker> -u <user> -P <password> -v -t 'homeassistant/event/#' -t 'homecam/#'
```

Trigger an alert (or publish a sample event to the camera's event topic) and
confirm the discovery config, then the event, appear.
