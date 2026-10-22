# Home Assistant validation record

## WIP checkpoint and pickup checklist — October 21, 2026

Work is saved on local branch `codex/pr126-readiness`, starting from PR #126
head `dcd36ea`. The PR's original feature checkout has not been changed, and
this checkpoint has not been pushed. The user requested a local WIP commit
while the remaining work waits.

Completed readiness checks: **116 focused tests passed**, strict Python typing,
repository-wide Python lint/format, lock consistency, strict OpenSpec validation
(10 items), and a clean `git diff --check`. Jinja2 is now a declared direct test
dependency. The regenerated lockfile changes no package versions.

Resume in this order:

1. Make Cargo/the repository's Rust toolchain available. `make check` currently
   stops at `cargo fmt`; Rust, UI and the full Python suite have not passed here.
2. Obtain Leonidas's explicit approval immediately before database-writing
   validation, then run `make check` against an appropriate disposable database.
   The WIP checkpoint is not approval to write to any database.
3. Reimport the revised blueprint into the test Home Assistant and rerun the
   changed freshness/recovery guards and notifier behavior. Existing phone
   evidence applies to the original PR commit, not these subsequent fixes.
4. Measure first-event behavior with multiple fresh camera identities and the
   default settle delay. Seed discovery and create the automation before relying
   on a camera's first real alert.
5. Confirm the open discovery/failure-policy questions in the change design.
   Also resolve stable MQTT client identity/availability ownership before claiming
   robust replacement-process recovery. Preserve current policies until decided.
6. When Dropbox access returns, test a real clip viewing URL, sharing/login
   permissions and phone playback. Public-video playback already passed; it
   does not establish Dropbox access. This is a separate provider check, not a
   reason to block the offline fixes.
7. Review the complete diff, integrate it into the PR branch, update the PR's
   verification notes, and only then publish. Sync/archive OpenSpec after the
   required review and verification, not as part of this WIP checkpoint.

The test publisher is stopped and the Home Assistant test automation is disabled.
No credentials are saved in this document or the test source.

## Manual integration check — October 21, 2026

Tested source: PR #126 commit
`dcd36ea525ec7d69cedb8bca3f4cc0ed3297bb35`.

Environment: Home Assistant OS 18.3, Core 2026.10.2, Supervisor 2026.10.2,
Frontend 20260930.4; real Mosquitto app and an existing iPhone companion-app
device. Credentials stayed outside source/config/result files.

The real MQTT notifier was selected through `load_notifier_plugin`, then its
public `send()` method received labeled synthetic Pydantic alerts. Original
alerts, HA events and availability used dedicated test topics. The blueprint
was imported from the exact tested Git commit, with minimum risk `medium` and
unanalyzed notifications enabled. No camera recording, VLM provider, upload,
database or full HomeSec runtime was involved.

| Check | Observation |
| --- | --- |
| Discovery and first event | A new camera device and Alerts event entity appeared. The first high event arrived with the default one-second settle delay in this one attempt. |
| Event attributes | Matching clip ID, camera name, risk, summary, timestamp, URL and failure flags were visible in the HA entity/trace. |
| High risk | HA's trace executed the phone action and the operator confirmed receipt. |
| Summary and urgency | An iPhone screenshot showed the summary and Time Sensitive presentation. Long messages were truncated in the collapsed notification. |
| Below threshold | A low event failed the risk condition; the phone action was not executed. |
| Unanalyzed enabled | Conditions passed and the phone action executed. Phone receipt was not independently confirmed for this case. |
| Link and playback | An alert with Mozilla's public flower MP4 as `view_url` executed the phone action; the operator confirmed tapping opened and played the video. |
| Clean shutdown | Stopping the publisher made the event entity Unavailable. |

The publisher was stopped and the dedicated test automation disabled after
testing. The test device/discovery and imported blueprint were retained for
repeat tests. No credentials or private clip URLs are recorded here.

## Automated readiness regressions

Run with the locked development environment:

```sh
uv run --locked pytest tests/homesec/test_mqtt_home_assistant.py \
  tests/homesec/test_home_assistant_blueprint.py \
  tests/homesec/test_mqtt_notifier.py
```

The notifier tests use the actual plugin and payload/config models, replacing
the paho transport boundary. They cover concurrent first alerts, reconnects
during discovery, bounded connection churn, missing acknowledgements, failure
followed by retry, topic validation, unchanged legacy alert payloads, and
cleanup. The fake rejects MQTT wildcard publish topics; it does not model all
broker, network or paho behavior.

Blueprint tests render the YAML's real Jinja templates with a fixed clock and
small HA state/conversion fixtures. Cases cover all risk/threshold pairs, both
unanalyzed options, stale/invalid/future timestamps, timezone offsets,
availability recovery, state restoration, attribute-only updates, and the
rendered summary, link, tag, urgency and missing-analysis fallbacks. These are
template-boundary tests, not the full HA automation engine.

The manual results above apply to the original PR commit. The subsequent
readiness fixes to freshness/recovery guards, timezone encoding, discovery and
acknowledgements are covered by repository regressions; they have not yet been
reimported and revalidated on live HA.

## Remaining checks and decisions

- Dropbox sharing/authentication and actual clip playback were not available.
  Public-video playback verifies notification navigation, not Dropbox access.
- A single first-event success does not establish a reliable discovery delay
  under load. Repeat with several new camera identities and record loss rates
  before treating the one-second default as reliable.
- Broker restart with lost retained state, unclean last-will delivery, stalled
  real broker acknowledgements and overlapping publisher replacement remain
  unverified on real infrastructure.
- Lazy discovery requires a seeded entity and configured automation before
  the first real alert can notify a phone. Startup/birth-message discovery and
  stable MQTT client identity remain design decisions.
- The current policy propagates HA publish failure after an original alert may
  already have been delivered, so pipeline retries can duplicate the original
  message. Failure containment remains an explicit open design question.
- Full repository validation, including Postgres-backed tests, is a separate
  publication gate. This record does not certify `make check`.
