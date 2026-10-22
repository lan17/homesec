# Scenario evidence register

This register describes the corrected PR118 existing-behavior baseline. Production source was pinned to `99270859ecb8bd8e6d67d3568271fca461996fbf`; corrections change documentation, not that implementation. The [archived design](../openspec/changes/archive/2026-10-10-baseline-existing-behavior/design.md) retains original requirement-level source and representative test evidence. It does not certify every scenario or supersede corrected specs.

Each scenario is listed separately under its requirement, with implementation leads and an evidence level. Leads inherited from the earlier 77-requirement audit are navigation/provenance, not new line-by-line proof or a claim that a particular test covers the scenario. **No scenario is labelled production-boundary-test verified solely because CI is green or an excerpt check passed.** Existing test coverage is not denied; its scenario-specific attribution remains to be checked.

## Evidence levels

| Label | Meaning and execution boundary |
| --- | --- |
| S | Source-supported or retained source review; specific repository test attribution not verified here |
| X-A | 12 copied media/preview authorization-guard cases with real request objects and fake application/token-validation boundaries |
| X-E | 7 minimal-ASGI validation/auth cases using the exact error module, copied request models/guard, and fake application |
| X-M | 4 media/staging cases using exact media/error modules in minimal ASGI or direct handler cancellation, fake repository/storage, and disposable files |
| X-D | 7 copied detector/upload-stage cases with a byte-verified path builder and fake decoder/model/retry boundary |
| X-B | 2 copied backup run/retry cases with fake dump/upload and temporary manifests/files |
| X-C | 8 copied merge/recheck/retention cases with minimal models and temporary files |
| B | Reserved for a specifically identified and executed repository production-boundary test; none newly claimed by this correction |
| Q | Reserved for documented physical camera/browser/codec/restore QA; none executed by this correction |

The six X groups total **40 isolated cases**, not 40 production conformance tests. A group can support a branch or control in several scenarios; rows do not count independent test executions. Reproductions were rerun during correction with Python 3.13.5, FastAPI 0.128.2, Starlette 0.50.0, Pydantic 2.13.5, and httpx 0.28.1 in the review container, not a freshly installed repository lockfile. Full mounted routing, real database/provider writes, native/device QA, and real dump/restore were not executed. Local review artifacts retain scripts and results; this register intentionally does not claim they are committed application tests.

The following labels retain those limits until a follow-up supplies a test path/name, tested commit, command/result, and exact boundary. Corrected limitation scenarios describe current defects, not a policy requiring future implementations to retain them.


## Local correction review on 2026-10-18

Two additional actual-code characterizations checked validation-only setup through the real service with a fake application and temporary missing config, and compared a fresh OpenAPI export with JSON-safe validation failure through the actual mounted contract route. They reproduced the unknown-backend validation gap and declared/actual 422 mismatch. These standalone scripts are not committed repository regression tests and do not complete the follow-up tasks. Their rows remain S with the boundary noted explicitly. The native admission-cap test named below was inspected, not executed during this correction.

## analysis-alerts

[Maintained spec](../openspec/specs/analysis-alerts/spec.md). S rows retain the source/test-attribution limit in the legend.


### Clip processing stage ordering

Implementation leads: [src/homesec/pipeline/core.py](../src/homesec/pipeline/core.py); [src/homesec/storage_paths.py](../src/homesec/storage_paths.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Analysis overlaps a slow upload | S | — |
| Filter failure stops analysis and notification | S | — |
| Upload preparation aborts the later join | S | Stage escape reproduced by X-D; later join source-traced |

### Supported object-filter results

Implementation leads: [src/homesec/plugins/filters/yolo.py](../src/homesec/plugins/filters/yolo.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Accepted target detection is returned | S | — |
| Small detections are excluded | S | — |
| Zero decoded frames return an empty detector result | X-D | Zero-frame worker cases; not real decoder/cleanup |
| A later failed read returns accumulated detection | X-D | Partial-read worker case; not real decoder |

### Analysis run modes and skip reasons

Implementation leads: [src/homesec/pipeline/core.py](../src/homesec/pipeline/core.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Trigger detection permits analysis | S | — |
| Non-trigger detection skips analysis | S | — |
| Always mode ignores trigger eligibility | S | — |
| Never mode disables analysis requests | S | — |

### OpenAI-compatible analysis validation

Implementation leads: [src/homesec/plugins/analyzers/openai.py](../src/homesec/plugins/analyzers/openai.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Valid structured response supplies analysis | S | — |
| Preprocessing bounds submitted images | S | — |
| Invalid provider response is rejected | S | — |
| No readable frames prevent a provider request | S | — |
| Schema-valid output has inconsistent temporal metadata | S | — |

### Default alert policy decision precedence

Implementation leads: [src/homesec/plugins/alert_policies/default.py](../src/homesec/plugins/alert_policies/default.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Camera motion override permits notification without analysis | S | — |
| Risk threshold permits notification | S | — |
| Activity list permits a low-risk notification | S | — |
| Below-threshold activity is suppressed | S | — |

### Disabled alert policy suppresses notification

Implementation leads: [src/homesec/plugins/alert_policies/noop.py](../src/homesec/plugins/alert_policies/noop.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| High-risk clip with alert policy disabled | S | — |

### Failed-analysis fallback and missing-analysis suppression

Implementation leads: [src/homesec/plugins/alert_policies/default.py](../src/homesec/plugins/alert_policies/default.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Failed analysis falls back to a trigger detection | S | — |
| Motion notification survives analysis failure | S | — |
| No analysis and no trigger suppress notification | S | — |

### Bounded processing-stage retries

Implementation leads: [src/homesec/pipeline/core.py](../src/homesec/pipeline/core.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Transient stage failure recovers | S | — |
| Persistent stage failure reaches its limit | S | — |
| An operation never settles within the retry runner | S | — |

### Notification eligibility and clip completion

Implementation leads: [src/homesec/pipeline/core.py](../src/homesec/pipeline/core.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Policy suppresses delivery | S | — |
| No enabled notifier skips delivery | S | — |
| Notification failure does not prevent completion | S | — |

### Best-effort workflow persistence during processing

Implementation leads: [src/homesec/repository/clip_repository.py](../src/homesec/repository/clip_repository.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Workflow-state write failure does not stop delivery | S | — |

## capture-recording

[Maintained spec](../openspec/specs/capture-recording/spec.md). S rows retain the source/test-attribution limit in the legend.


### Local folder ready-file ingestion

Implementation leads: [src/homesec/sources/local_folder.py](../src/homesec/sources/local_folder.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Recently modified file waits | S | — |
| Eligible local clip is handed off | S | — |
| Other files stay outside local ingestion | S | — |
| Uppercase suffix is excluded with case-sensitive matching | S | — |
| Local-folder timestamps are estimates | S | — |

### Local folder repeat suppression

Implementation leads: [src/homesec/sources/local_folder.py](../src/homesec/sources/local_folder.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Repeated scans do not re-emit a clip | S | — |
| Existing manifest suppresses an accepted clip | S | — |

### FTP completed-upload ingestion

Implementation leads: [src/homesec/sources/ftp.py](../src/homesec/sources/ftp.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Allowed upload becomes a clip | S | — |
| Empty allowlist accepts another extension | S | — |
| Disallowed upload is excluded | S | — |
| FTP timestamps use the configured duration estimate | S | — |

### FTP incomplete-upload handling

Implementation leads: [src/homesec/sources/ftp.py](../src/homesec/sources/ftp.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Interrupted upload is discarded or retained by configuration | S | — |
| Incomplete upload cannot be unlinked | S | — |

### Motion-triggered recording boundaries

Implementation leads: [src/homesec/sources/rtsp/core.py](../src/homesec/sources/rtsp/core.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| New motion starts a recording | S | — |
| Motion refreshes the stop delay | S | — |
| Recording uses its configured keepalive sensitivity | S | — |
| Ongoing motion rotates the clip | S | — |

### RTSP camera unavailability and reconnection

Implementation leads: [src/homesec/sources/rtsp/core.py](../src/homesec/sources/rtsp/core.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Offline camera does not reject source start | S | — |
| Camera becomes available after failed checks | S | — |
| Finite reconnect attempts are exhausted | S | — |
| Finite startup attempts are exhausted | S | — |
| Zero attempt limit permits later recovery | S | — |

### RTSP clip handoff eligibility

Implementation leads: [src/homesec/sources/rtsp/core.py](../src/homesec/sources/rtsp/core.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Completed recording reaches processing | S | — |
| Failed finalization does not deliver unfinished bytes | S | — |
| Missing or empty output is not a clip | S | — |

### Native recording publication and compatibility

Implementation leads: [src/homesec/sources/rtsp/shared_media.py](../src/homesec/sources/rtsp/shared_media.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Native writer dies before publication | S | — |
| Unsupported profile uses compatible recording | S | — |

### Ingestion callback failure isolation

Implementation leads: [src/homesec/sources/base.py](../src/homesec/sources/base.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Later local clips still reach the callback | S | — |

## events-playback

[Maintained spec](../openspec/specs/events-playback/spec.md). S rows retain the source/test-attribution limit in the legend.


### Event queries support filtering and cursor pagination

Implementation leads: [src/homesec/state/postgres.py](../src/homesec/state/postgres.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Continue to the next result page | S | — |
| Reject invalid time bounds or a malformed cursor | S | — |
| Exclude deleted events from the default list | S | — |
| Query deleted records explicitly | S | — |

### Event browsing preserves filter context in the URL

Implementation leads: [ui/src/features/clips/queryParams.ts](../ui/src/features/clips/queryParams.ts).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Reopen an event search | S | — |
| Change filters after paging | S | — |
| Include events without detections | S | — |

### Event views present available summaries and inline playback

Implementation leads: [ui/src/features/clips/ClipsPage.tsx](../ui/src/features/clips/ClipsPage.tsx).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Browse an event with playable media | S | — |
| Expand event details | S | — |
| No events match the search | S | — |

### Event detail lookup distinguishes known and missing events

Implementation leads: [src/homesec/api/routes/clips.py](../src/homesec/api/routes/clips.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Retrieve a known event | S | — |
| Open details without a result window | S | — |
| Retrieve an unknown event | S | — |

### Media URLs respect authentication and clip availability

Implementation leads: [src/homesec/api/dependencies.py](../src/homesec/api/dependencies.py); [src/homesec/api/routes/__init__.py](../src/homesec/api/routes/__init__.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Request media without authentication enabled | S | — |
| Request authorized media access | S | — |
| Attempt to reuse authorization for another clip | X-A | Guard negative case; token boundary faked |
| Attempt to mint a token with a playback token | S | — |
| API key takes precedence over an invalid media query token | X-A | Copied authorization function; not mounted router |
| Disabled authentication ignores a media query token | X-A | Copied authorization function; not mounted router |

### Media delivery exposes availability and retrieval failures

Implementation leads: [src/homesec/api/routes/media.py](../src/homesec/api/routes/media.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Play stored media | X-M | ASCII minimal-ASGI control; normal gates assumed |
| Request media before storage is available | S | — |
| Storage retrieval fails | X-M | Minimal-ASGI storage-error control |
| A retrieved clip filename fails response-header encoding | X-M | Exact modules in minimal ASGI; guards assumed |

### Playback stages media before HTTP delivery

Implementation leads: [src/homesec/api/routes/media.py](../src/homesec/api/routes/media.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Media retrieval completes before response construction | X-M | Minimal ASGI success control; not mounted full router |
| Cancellation bypasses retrieval cleanup | X-M | Direct handler cancellation with fake storage |

### Tokenized playback supports bounded source refresh

Implementation leads: [ui/src/api/hooks/useClipMediaUrl.ts](../ui/src/api/hooks/useClipMediaUrl.ts).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| A media token approaches expiry | S | — |
| Playback repeatedly fails on an unchanged tokenized URL | S | — |
| Direct playback fails | S | — |

## integration-contracts

[Maintained spec](../openspec/specs/integration-contracts/spec.md). S rows retain the source/test-attribution limit in the legend.


### API key authentication

Implementation leads: [src/homesec/api/dependencies.py](../src/homesec/api/dependencies.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Protected request lacks valid credentials | X-E | ASCII/missing-key minimal-ASGI controls |
| Valid bearer credentials | S | — |
| Enabled authentication has no effective key | S | — |
| Public health with authentication enabled | S | — |
| Authentication disabled | S | — |
| Non-ASCII bearer comparison fails before route execution | X-E | Minimal ASGI with copied guard and exact error module |

### Browser session authentication

Implementation leads: [ui/src/api/client.ts](../ui/src/api/client.ts).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Reuse the supplied key within the browser session | S | — |
| Clear stored authentication | S | — |
| Override the stored key for one request | S | — |

### Legacy browser route compatibility

Implementation leads: [ui/src](../ui/src).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Open a legacy event search or detail link | S | — |
| Open former navigation destinations | S | — |

### Canonical API exception responses

Implementation leads: [src/homesec/api/errors.py](../src/homesec/api/errors.py); [src/homesec/models/talk.py](../src/homesec/models/talk.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Route reports a conflict with additional context | S | — |
| Invalid request parameters | X-E | JSON-safe validation control only |
| Unexpected route exception | S | — |
| A model-validator exception cannot be serialized | X-E | Minimal ASGI; copied request models |

### Generated API contract

Implementation leads: [src/homesec/api/openapi_export.py](../src/homesec/api/openapi_export.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Generate the contract without application startup | S | — |
| Generated validation-error schema differs from the HTTP response | S | Fresh schema and actual mounted contract-route characterization with fake app; no committed regression test |

### Notification payload meaning

Implementation leads: [src/homesec/pipeline/core.py](../src/homesec/pipeline/core.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Notification includes available analysis and media references | S | — |
| Notification without successful upload or analysis | S | — |
| Delayed processing does not preserve capture time as notification time | S | — |

### MQTT alert publication

Implementation leads: [src/homesec/plugins/notifiers/mqtt.py](../src/homesec/plugins/notifiers/mqtt.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Publish to a camera-specific topic | S | — |
| Broker connection does not become available | S | — |

### Home Assistant MQTT discovery

Implementation leads: [src/homesec/plugins/notifiers/mqtt.py](../src/homesec/plugins/notifiers/mqtt.py);
[src/homesec/plugins/notifiers/mqtt_home_assistant.py](../src/homesec/plugins/notifiers/mqtt_home_assistant.py).
Proposed in the active change [home-assistant-mqtt-discovery](../openspec/changes/home-assistant-mqtt-discovery/);
not yet synced into the maintained specs. B rows run the real notifier, config
validation, and payload builders against a fake `paho` client in
[test_mqtt_home_assistant.py](../tests/homesec/test_mqtt_home_assistant.py)
(`uv run --locked pytest tests/homesec/test_mqtt_home_assistant.py`); no broker
or Home Assistant instance is involved.

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Disabled by default | B | Fake client; `test_disabled_by_default_publishes_only_the_alert` |
| First alert for a camera announces the entity | B | Fake client; `test_first_alert_publishes_discovery_before_the_event`, `test_existing_alert_payload_is_unchanged` |
| Later alerts do not repeat discovery | B | Fake client; `test_discovery_is_published_once_per_camera` |
| Event type reflects analysis | B | `TestEventPayload`, `test_unanalyzed_alert_publishes_unanalyzed_event` |
| Reconnect republishes discovery | B | Fake client reconnect callback; `test_reconnect_republishes_discovery_and_availability`. A real broker that drops retained data was not exercised |
| Reconnect during discovery | B | `test_reconnect_during_discovery_reannounces_before_the_event`; repeats discovery on the new connection before the event |
| Repeated connection churn | B | `test_repeated_reconnects_fail_without_publishing_an_undiscovered_event`; finite attempt count |
| Concurrent first alerts | B | `test_concurrent_first_alerts_discover_each_camera_once`; includes simultaneous sends for the same camera and a nonzero settle delay |
| Missing publish acknowledgements | B | `test_event_acknowledgement_timeout_fails_after_original_alert_delivery`, `test_failed_discovery_is_retried_before_any_event`; real broker stalls not exercised |

### Home Assistant availability

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Connect announces availability | B | Fake client; `test_connect_sets_last_will_and_announces_online`. Last-will delivery on an unclean disconnect is broker behavior and was not exercised |
| Clean shutdown announces offline | B | Fake client; `test_shutdown_announces_offline`, `test_shutdown_without_home_assistant_publishes_nothing`, `test_shutdown_logs_unacknowledged_offline_and_still_disconnects`. Real stalled-broker cleanup remains unverified |

### Home Assistant topic and identity safety

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Unpublishable topic setting | B | `TestHomeAssistantConfig` |
| Names that normalize alike stay distinct | B | `TestCameraSlug` |
| Unsafe camera name | B | `test_camera_name_with_wildcards_yields_safe_home_assistant_topics` uses a publishable fixed legacy topic; `test_invalid_legacy_topic_fails_before_home_assistant_publication` covers raw-name legacy rejection. HA slug safety does not repair the original alert topic |
| Escaped event placeholder | B | `TestHomeAssistantConfig` rejects a literal placeholder that would send every camera to the same event topic |
| Timezone semantics | B | `test_event_timestamp_has_an_offset_and_preserves_the_instant`; original alert timestamp remains unchanged |

### Home Assistant push-notification blueprint

Implementation lead: [integrations/home-assistant/homesec_alert_notification.yaml](../integrations/home-assistant/homesec_alert_notification.yaml).
[test_home_assistant_blueprint.py](../tests/homesec/test_home_assistant_blueprint.py)
checks structure and renders the shipped conditions and action fields using
Jinja2, fixed time, and small state/conversion fixtures. It does not run HA's
full template sandbox or automation engine. The [manual validation record](home-assistant-validation.md)
documents real HA/Mosquitto/iPhone observations on the original PR commit;
readiness fixes still need a live rerun.

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Risk below the minimum is ignored | B | `test_risk_threshold_is_inclusive` renders all 16 risk/threshold pairs |
| Risk at or above the minimum notifies | B | Same test plus `test_rendered_push_contains_summary_clip_link_tag_and_urgency` |
| Unanalyzed alerts follow the option | B | `test_unanalyzed_respects_the_option` covers enabled and disabled |
| Stale or ambiguous event time | B | `test_freshness_uses_payload_time_and_fails_closed` covers exact five-minute boundary, future, missing, invalid, naive and offset timestamps |
| Availability recovery/restored state | B | `test_state_transitions_require_a_new_event` distinguishes old-clip recovery, genuine new clips, first events from unknown, and attribute-only changes |
| Missing summary or media URL | B | `test_push_has_readable_fallbacks_without_summary_or_link` |

### Email alert delivery

Implementation leads: [src/homesec/plugins/notifiers/sendgrid_email.py](../src/homesec/plugins/notifiers/sendgrid_email.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Send a templated alert email | S | — |
| Structured analysis contains HTML characters | S | — |
| SendGrid rejects a request | S | — |
| Credentials absent or integration shut down | S | — |

### Notification fan-out isolation

Implementation leads: [src/homesec/pipeline/core.py](../src/homesec/pipeline/core.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| One notifier fails while another succeeds | S | — |
| All configured notifiers disabled | S | — |

### Plugin registration and discovery

Implementation leads: [src/homesec/plugins/registry.py](../src/homesec/plugins/registry.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Select a registered extension | S | — |
| Duplicate or unknown backend | S | — |
| Discover an installed external extension | S | — |

### Plugin configuration and component contracts

Implementation leads: [src/homesec/plugins/registry.py](../src/homesec/plugins/registry.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Invalid plugin configuration | S | — |
| Validate without constructing a plugin | S | — |

## operations-maintenance

[Maintained spec](../openspec/specs/operations-maintenance/spec.md). S rows retain the source/test-attribution limit in the legend.


### CLI operation and configuration validation

Implementation leads: [src/homesec/config/loader.py](../src/homesec/config/loader.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Validate usable configuration | S | — |
| Configuration invalid or missing during validation | S | — |
| Runtime reports invalid configuration | S | — |

### Public health summaries

Implementation leads: [src/homesec/api/routes/health.py](../src/homesec/api/routes/health.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Database unavailable but pipeline running | S | — |
| Pipeline is not running | S | — |
| Pipeline and database healthy | S | — |
| Service awaiting initial setup | S | — |

### Component diagnostics

Implementation leads: [src/homesec/api/routes/health.py](../src/homesec/api/routes/health.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Storage probe raises an exception | S | — |
| Camera health affects diagnostics | S | — |
| Disabled camera has no active source | S | — |
| Pipeline stopped | S | — |

### Operational system statistics

Implementation leads: [src/homesec/api/routes/stats.py](../src/homesec/api/routes/stats.py); [src/homesec/state/postgres.py](../src/homesec/state/postgres.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Read current system statistics | S | — |
| Database availability check fails | S | — |
| Statistics requested before setup | S | — |
| Count query fails after the availability check | S | — |

### Shutdown ordering and runtime termination

Implementation leads: [src/homesec/app.py](../src/homesec/app.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Normal process shutdown | S | — |
| Worker ignores graceful termination | S | — |
| In-flight clips can finish within shutdown limits | S | — |

### Cleanup selection and dry run

Implementation leads: [src/homesec/maintenance/cleanup_clips.py](../src/homesec/maintenance/cleanup_clips.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Preview deletion of an empty clip | S | — |
| Limit the cleanup scope | S | — |

### Backup opt-in and prerequisites

Implementation leads: [src/homesec/maintenance/postgres_backup.py](../src/homesec/maintenance/postgres_backup.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Backups not enabled | S | — |
| Required dump executable unavailable | S | — |
| Invalid backup interval | S | — |

### Backup status and manual requests

Implementation leads: [src/homesec/maintenance/postgres_backup.py](../src/homesec/maintenance/postgres_backup.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Accept a manual backup | S | — |
| Manual request cannot run | S | — |

### Dump artifacts and credential isolation

Implementation leads: [src/homesec/maintenance/postgres_backup.py](../src/homesec/maintenance/postgres_backup.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Database credentials passed to a dump | S | — |
| Two successful dumps share a timestamp | S | — |

### Interrupted dump cleanup

Implementation leads: [src/homesec/maintenance/postgres_backup.py](../src/homesec/maintenance/postgres_backup.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Dump times out | S | — |
| Dump cancelled during shutdown | S | — |

### Backup manifest recovery and upload outcomes

Implementation leads: [src/homesec/maintenance/postgres_backup.py](../src/homesec/maintenance/postgres_backup.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Recover local backups after manifest corruption | S | — |
| Local dump succeeds but upload fails | X-B | Successful-housekeeping control; not real dump |
| Housekeeping failure reclassifies a completed dump | X-B | Run/retry excerpts with fake dump and temporary manifest |

### Backup retention and remote-delete recovery

Implementation leads: [src/homesec/maintenance/postgres_backup.py](../src/homesec/maintenance/postgres_backup.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Retention after successful upload | S | — |
| Repeated upload failures | S | — |
| Interrupted upload has uncertain remote outcome | S | — |
| Remote deletion fails or is interrupted | S | — |

## preview-talk

[Maintained spec](../openspec/specs/preview-talk/spec.md). S rows retain the source/test-attribution limit in the legend.


### Preview availability and transport are explicit

Implementation leads: [src/homesec/api/routes/preview.py](../src/homesec/api/routes/preview.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Start a WebRTC preview | S | — |
| Use existing HLS configuration | S | — |
| Preview cannot be served | S | — |

### Preview access is scoped and expires

Implementation leads: [src/homesec/api/dependencies.py](../src/homesec/api/dependencies.py); [src/homesec/api/routes/preview.py](../src/homesec/api/routes/preview.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Attempt access to another camera | X-A | Guard negative case; token boundary faked |
| Negotiate near token expiry | S | — |
| Stale HLS files remain after preview stops | S | — |
| API-key preview authorization ignores an invalid query token | X-A | Copied authorization function; not mounted router |
| Disabled preview authentication ignores tokens | X-A | Copied authorization function; not mounted router |

### Preview yields according to recording policy

Implementation leads: [src/homesec/sources/rtsp/live_publisher.py](../src/homesec/sources/rtsp/live_publisher.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Recording starts under the default policy | S | — |
| Concurrent preview is available | S | — |
| Camera session capacity prevents preview | S | — |

### Preview resources have bounded lifetimes

Implementation leads: [native/webrtc/src/engine.rs](../native/webrtc/src/engine.rs). Existing admission-cap/negotiation-timeout coverage: `malformed_offers_and_negotiation_timeout_release_admission_capacity` in [native media delivery tests](../native/webrtc/tests/media_delivery.rs), inspected but not executed during this correction.

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Preview loses all viewers | S | — |
| A WebRTC lease expires while status is polled | S | — |
| Close one of several WebRTC viewers | S | — |

### Browser preview handles visibility and changing authorization

Implementation leads: [ui/src/features/cameras/hooks/useCameraPreview.ts](../ui/src/features/cameras/hooks/useCameraPreview.ts).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Background and return to a WebRTC preview | S | — |
| Switch cameras while cleanup is pending | S | — |

### Talk capability and backend selection reflect camera support

Implementation leads: [src/homesec/sources/rtsp/core.py](../src/homesec/sources/rtsp/core.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| ONVIF is supported in automatic mode | S | — |
| Manual RTSP setup needs local Tapo discovery | S | — |
| Explicit Tapo configuration fails | S | — |

### Talk preparation defines the authorized audio stream

Implementation leads: [src/homesec/api/routes/talk.py](../src/homesec/api/routes/talk.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Prepare a supported talk session | S | — |
| Send an invalid audio frame | S | — |
| Use expired stream authorization | S | — |
| Talk bearer authorization supersedes an expired token | S | — |

### Talk sessions are exclusive and bounded per camera

Implementation leads: [src/homesec/sources/rtsp/talk/manager.py](../src/homesec/sources/rtsp/talk/manager.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Reserve talk while another session exists | S | — |
| A reservation or stream times out | S | — |
| Activation starts a new duration limit | S | — |
| Disconnect before streaming starts | S | — |
| Cancel opening or shut down an active session | S | — |

### Browser talk respects hold intent and microphone availability

Implementation leads: [ui/src/features/cameras/hooks/usePushToTalk.ts](../ui/src/features/cameras/hooks/usePushToTalk.ts).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Hold and release talk | S | — |
| Microphone permission is denied | S | — |
| Release before startup completes | S | — |

### Browser talk stops on excessive audio backpressure

Implementation leads: [ui/src/features/cameras/hooks/usePushToTalk.ts](../ui/src/features/cameras/hooks/usePushToTalk.ts).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Audio transmission cannot keep up | S | — |

## setup-configuration

[Maintained spec](../openspec/specs/setup-configuration/spec.md). S rows retain the source/test-attribution limit in the legend.


### Setup readiness reflects configuration and runtime state

Implementation leads: [src/homesec/services/setup.py](../src/homesec/services/setup.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| First-run setup is needed | S | — |
| Configuration exists before the pipeline runs | S | — |
| Configured cameras are running | S | — |

### Setup checks report readiness without saving configuration

Implementation leads: [src/homesec/services/setup.py](../src/homesec/services/setup.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| A preflight check times out | S | — |
| A proposed connection is tested | S | — |
| Generic connection test only establishes its ping result | S | — |

### Setup finalization validates before persistence and activation

Implementation leads: [src/homesec/services/setup.py](../src/homesec/services/setup.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Validate a valid setup without applying it | S | Core-model/prerequisite checks only; full candidate validation is skipped |
| Reject an invalid setup | S | Applied finalization; validation-only has a narrower path |
| Apply a valid setup | S | — |
| Reject finalization outside bootstrap | S | — |
| Validation-only accepts an unregistered source backend | S | Actual setup-service characterization with fake app/temporary path; committed mounted-route regression remains pending |

### ONVIF setup discovers cameras and reports probe outcomes

Implementation leads: [src/homesec/api/routes/onvif.py](../src/homesec/api/routes/onvif.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Discover available cameras | S | — |
| No cameras respond | S | — |
| Probe a reachable camera | S | — |
| Discovery or probing fails | S | — |
| A camera probe exceeds its timeout | S | — |

### Camera configuration mutations validate and preserve omitted fields

Implementation leads: [src/homesec/config/manager.py](../src/homesec/config/manager.py); [src/homesec/config/validation.py](../src/homesec/config/validation.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Update one source setting | S | — |
| Switch source backends | S | — |
| Reject an invalid camera mutation | S | — |
| Reject duplicate or missing camera names | S | — |
| Merge a nested object without clearing omitted keys | X-C | Merge helper only; persisted route coverage pending |
| Clear a source setting through null | S | Removal helper covered by X-C; default resolution source-traced |
| Replace a list rather than append to it | X-C | Merge helper only; persisted route coverage pending |
| Camera removal leaves an invalid policy reference | S | — |
| Leave source settings unchanged for a null top-level patch | S | Real manager source path; persisted-route attribution remains pending |
| Replace a non-object source value with an object | S | Merge helper source path; schema-valid replacement and persisted-route attribution remain pending |

### Component disabling does not bypass all configuration validation

Implementation leads: [src/homesec/config/validation.py](../src/homesec/config/validation.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Disabled source or notifier has invalid plugin configuration | S | — |
| Never analysis mode retains analyzer schema validation | S | — |

### Configuration reads protect sensitive values

Implementation leads: [src/homesec/api/redaction.py](../src/homesec/api/redaction.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Read a configuration containing credentials | S | — |
| Submit a displayed redaction placeholder | S | — |

### Saving camera configuration is distinct from activating it

Implementation leads: [src/homesec/config/manager.py](../src/homesec/config/manager.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Save camera settings without activation | S | — |
| Request reload of saved configuration | S | — |
| Reject a concurrent reload | S | — |
| Save a camera mutation before a reload request is rejected | S | — |
| Retry camera creation after its reload request failed | S | — |

### Runtime activation reports completion and preserves the prior runtime on candidate failure

Implementation leads: [src/homesec/runtime/manager.py](../src/homesec/runtime/manager.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Complete a successful reload | S | — |
| Candidate startup or activation fails | S | — |

## storage-retention

[Maintained spec](../openspec/specs/storage-retention/spec.md). S rows retain the source/test-attribution limit in the legend.


### Clip destination and upload outcome

Implementation leads: [src/homesec/storage_paths.py](../src/homesec/storage_paths.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Stored clip retains its camera grouping | S | — |
| Dot camera segment is normalized away | X-D | Path builder / upload-stage excerpt |
| Parent camera segment fails before an upload attempt | X-D | Stage escape; not full pipeline join |

### Built-in storage delivery and destination validation

Implementation leads: [src/homesec/plugins/storage/local.py](../src/homesec/plugins/storage/local.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Local storage round trip | S | — |
| Dropbox view URL follows the configured prefix | S | — |
| Unsafe destination is refused | S | — |

### Dropbox credential selection

Implementation leads: [src/homesec/plugins/storage/dropbox.py](../src/homesec/plugins/storage/dropbox.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Access token takes precedence | S | — |
| Complete refresh credentials are accepted | S | — |
| Incomplete credentials fail setup | S | — |

### Bounded upload retries and failure outcome

Implementation leads: [src/homesec/pipeline/core.py](../src/homesec/pipeline/core.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Transient upload failure recovers | S | — |
| Upload failure survives retry exhaustion | S | — |

### Local retention deletion eligibility

Implementation leads: [src/homesec/retention/pruner.py](../src/homesec/retention/pruner.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Uploaded completed clip is eligible | S | — |
| Unconfirmed local files are preserved | S | — |
| Selected pathname is replaced before unlink | X-C | Already-selected candidate; not FTP/Postgres concurrency |

### Oldest eligible clip pruning

Implementation leads: [src/homesec/retention/pruner.py](../src/homesec/retention/pruner.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Oldest eligible clip is removed first | S | — |
| Equal creation times have deterministic order | S | — |
| Size failure prevents an all-clear claim | S | — |
| Protected files can leave usage over the limit | S | — |

### Retention triggering and failure isolation

Implementation leads: [src/homesec/pipeline/core.py](../src/homesec/pipeline/core.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| New clip discovers a local retention directory | S | — |
| Overlapping prune request is dropped | S | — |
| Pruning fails independently of processing | S | — |

### Built-in storage deletion is idempotent

Implementation leads: [src/homesec/plugins/storage/local.py](../src/homesec/plugins/storage/local.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Stored file is deleted twice | S | — |

### API clip deletion reports partial failures

Implementation leads: [src/homesec/api/routes/clips.py](../src/homesec/api/routes/clips.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Delete a known uploaded clip | S | — |
| Delete a clip without a stored object | S | — |
| Delete a missing clip | S | — |
| Marking deletion fails | S | — |
| Storage deletion fails after marking | S | — |

### Cleanup rechecks before deleting empty clips

Implementation leads: [src/homesec/maintenance/cleanup_clips.py](../src/homesec/maintenance/cleanup_clips.py); [src/homesec/plugins/filters/yolo.py](../src/homesec/plugins/filters/yolo.py).

| Scenario | Evidence | Boundary / remaining verification |
| --- | --- | --- |
| Recheck rescues a previous empty detection | S | — |
| Empty clip is deleted after recheck | S | — |
| Dry run reports without applying deletion | S | — |
| Recheck or deletion failure is reported | S | — |
| Cleanup receives an empty zero-frame recheck | S | Zero-frame detector reproduced by X-D; applied cleanup source-traced |
| Cleanup cannot build an external filter configuration | X-C | Recheck-builder excerpt; not whole cleanup |

## Inventory

| Capability | Requirements | Scenarios |
| --- | ---: | ---: |
| analysis-alerts | 10 | 31 |
| capture-recording | 9 | 28 |
| events-playback | 8 | 28 |
| integration-contracts | 11 | 32 |
| operations-maintenance | 12 | 36 |
| preview-talk | 10 | 32 |
| setup-configuration | 9 | 34 |
| storage-retention | 10 | 33 |
| **Total** | **79** | **254** |
