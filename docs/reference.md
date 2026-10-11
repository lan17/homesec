# HomeSec public behavior reference

The maintained [OpenSpec capability specifications](../openspec/specs/) describe HomeSec's public behavior. Use them for requirements, lifecycle transitions, failure outcomes, and compatibility. Use the generated API contract for exact HTTP and payload shapes, and the existing guides for deployment procedures.

The initial baseline describes the implementation reviewed on **2026-10-10**. It covers the major public surfaces and critical scenarios, rather than every setting, device, timing edge case, or failure combination. Existing tests were inspected as evidence; local baseline validation did not execute application tests or physical-device checks. Structural and documentation validation do not certify runtime conformance.

## Capability index

| Capability | Maintained behavior | Related guides and references |
| --- | --- | --- |
| Setup and configuration | [Setup configuration spec](../openspec/specs/setup-configuration/spec.md) | [Configuration and commands](../README.md#configuration); [configuration models](../src/homesec/models/config.py); [configuration loader](../src/homesec/config/loader.py) |
| Capture and recording | [Capture recording spec](../openspec/specs/capture-recording/spec.md) | [Shared camera media](shared-rust-media.md); [recording configuration](../README.md#configuration) |
| Storage and retention | [Storage retention spec](../openspec/specs/storage-retention/spec.md) | [Storage configuration](../README.md#configuration); [cleanup command](../README.md#commands); [retention implementation](../src/homesec/retention/pruner.py) |
| Analysis and alerts | [Analysis alerts spec](../openspec/specs/analysis-alerts/spec.md) | [Pipeline overview](../README.md#pipeline-at-a-glance); [built-in plugins](../README.md#built-in-plugins) |
| Events and playback | [Events playback spec](../openspec/specs/events-playback/spec.md) | [Browser client](../ui/README.md); [generated API contract](../ui/src/api/generated/openapi.json) |
| Preview and talk | [Preview talk spec](../openspec/specs/preview-talk/spec.md) | [Preview deployment](preview-deployment.md); [WebRTC preview](webrtc-preview.md); [push-to-talk](push-to-talk.md) |
| Integration contracts | [Integration contracts spec](../openspec/specs/integration-contracts/spec.md) | [API schemas and commands](#api-schemas-and-commands); [plugin guide](../PLUGIN_DEVELOPMENT.md); [component interfaces](../src/homesec/interfaces.py) |
| Operations and maintenance | [Operations maintenance spec](../openspec/specs/operations-maintenance/spec.md) | [CLI commands](../README.md#commands); [observability](../README.md#observability); [Postgres backups](postgres-backups.md) |

## Known limitations and evidence limits

The following observations bound the baseline. They identify current defects, documentation disagreements, pending features, or incomplete verification; they do not establish promises to preserve those behaviors. Corrective behavior changes require their own reviewed spec changes. The capability headings below are durable links from the maintained specs.

### Setup configuration

- **Saved and active configuration can differ after failure.** Setup persists configuration before activation. Runtime rollback preserves the prior runtime, but does not establish rollback of the saved file. Camera create, update, and delete also save before their optional reload request: a busy or failed reload can return an error after a successful save. Retrying that camera creation then conflicts with the name already saved. Focused persisted-mutation/create-retry coverage was not identified; the existing route tests establish error responses. See the [setup service](../src/homesec/services/setup.py), [camera routes](../src/homesec/api/routes/cameras.py), and [runtime manager tests](../tests/homesec/test_runtime_manager.py).
- **First-run setup and camera editing are the supported editing surfaces.** Setup finalization is bootstrap-only. The settings page links several settings to setup, but this does not provide a general configuration editor or a general configuration PATCH endpoint. A broader editor remains a separate feature.

### Capture recording

- **Finite RTSP retry exhaustion stops that camera's monitoring.** The camera source worker ends while the runtime process remains available. Later camera availability does not resume monitoring until runtime reload or process restart creates another source worker. `max_attempts=0` retries indefinitely. The [RTSP source](../src/homesec/sources/rtsp/core.py) supports this consequence; a focused end-to-end exhaustion/recovery test was not identified.
- **Repeat suppression is not durable processing replay.** The local-folder source accepts an ID before callback completion, and the source boundary contains callback failures. It does not guarantee retry of failed processing or exactly-once delivery. Manifest-based restart suppression is source-supported without a focused restart test identified. The `*.mp4` glob follows platform case rules; uppercase `.MP4` is excluded where matching is case-sensitive, unlike FTP's case-insensitive extension allowlist. Focused uppercase local-ingestion coverage was not identified. See the [local-folder source](../src/homesec/sources/local_folder.py).
- **Publication conditions do not prove universal media integrity.** Native finalization and legacy recording use different checks; a nonempty file alone does not prove playable media. The baseline does not promise universal codec/hardware compatibility or gapless rotation. Failed native recordings can leave excluded `.partial` files; this does not mean every finalization failure produces a scanned MP4. See the [shared-media guide](shared-rust-media.md).
- **The README startup flag example has a compatibility limit.** `stream.ffmpeg_flags` is ignored by current startup preflight while automatic negotiation is enabled. Do not use that example as a guarantee of current startup control.

### Storage retention

- **API deletion can leave a separate original recording.** The route marks the clip `DELETED` and deletes its stored object, without separately unlinking the original recording. That original is then ineligible for retention, which requires `DONE`, and maintenance cleanup excludes deleted rows. If the stored object and original share a physical file, storage deletion can already remove it. A separate original can otherwise remain indefinitely and contribute to measured usage. All-copies deletion semantics and deletion recovery remain unresolved; no focused deletion/retention interaction test was identified. See the [clip routes](../src/homesec/api/routes/clips.py) and [retention pruner](../src/homesec/retention/pruner.py).
- **Successful upload alone does not make every local file reclaimable.** Exhausted filter failure leaves `ERROR` before completion, even if upload succeeded. Discovered MP4s without readable state are also ineligible. No automatic whole-clip processing replay or reclamation path was found for files while they remain ineligible; recovery or manual intervention can be necessary. Do not assume every orphan directory is discovered. See the [pipeline](../src/homesec/pipeline/core.py).
- **Retention is not a hard total-disk bound.** Pruning is triggered using eligible uploaded/completed bytes, while reported usage measures discovered local bytes. Protected files can leave usage above the configured limit even after pruning. The baseline preserves eligibility and oldest-first deletion when pruning occurs, without adopting the disputed trigger as an intended universal budget policy.
- **Local storage performs syntax checks, not symlink confinement.** Destinations are validated and constructed relative to the configured root, but existing symlinks can be followed during copy. See the [local storage backend](../src/homesec/plugins/storage/local.py).
- **Cleanup media deletion and workflow persistence are not atomic.** Repository writes are best-effort, so media deletion can succeed while state/event persistence fails. Default dry-run preservation is source-supported, but a focused test proving preservation of original media and workflow state/events was not identified. See the [cleanup implementation](../src/homesec/maintenance/cleanup_clips.py).
- **Deletion can report not-found after a lookup failure.** The shared lookup limitation is described under [events playback](#events-playback); HTTP 404 alone does not prove that the clip never existed.

### Analysis alerts

- **Detector setting names do not fully explain their current semantics.** YOLO `sample_fps` is a frame-index stride; `min_hits` counts distinct classes for early exit rather than suppressing below-threshold results. Its supported target mapping is narrower than some README examples. The baseline describes supported results and acceptance checks, without deciding revised settings or expanding detector promises. Existing detector tests exercise boundaries rather than real-model accuracy. See the [YOLO implementation](../src/homesec/plugins/filters/yolo.py).
- **An absent analysis can produce a misleading reason.** The default alert policy uses `filter_detected_trigger_vlm_failed` when analysis is absent and a trigger is detected, including intentional `never` mode. `never` skips analysis; it does not itself disable notification. The reason is not proof of a provider failure. See the [default policy](../src/homesec/plugins/alert_policies/default.py).
- **Some critical boundaries remain source-supported.** Dedicated disabled-policy suppression, risk-threshold equality, and event-append failure-containment coverage require further assessment. Best-effort workflow persistence does not establish universal outage resilience or durable replay.

### Events playback

- **Deleted records remain addressable by identity.** Default lists exclude deleted clips, while an explicit deleted-status filter can include them. Detail and media-token lookups do not reject `DELETED` records, whose metadata and storage URI can remain. A token/media URL can therefore still be returned. Fetching a removed stored object returns HTTP 502 `CLIP_MEDIA_FETCH_FAILED` if retrieval raises; playback can still succeed if deletion failed and the object remains retrievable. Status-aware lookup/playback and deletion retry semantics remain unresolved. See the [clip routes](../src/homesec/api/routes/clips.py), [media route](../src/homesec/api/routes/media.py), and [Postgres store](../src/homesec/state/postgres.py).
- **Lookup failures can look like missing clips.** The availability guard can pass before a query or hydration failure. The default store converts such failures to `None`, bypassing repository retries; exhausted repository lookup exceptions also become `None`. Detail, token, media, and delete can then return HTTP 404 `CLIP_NOT_FOUND`. Unknown-identity guarantees assume successful lookup; 404 alone is not proof of absence. See the [repository](../src/homesec/repository/clip_repository.py).
- **List failures can look like successful empty or incomplete searches.** After authentication and the database availability check succeed, a later list-query failure can still produce HTTP 200 with `clips=[]`, `next_cursor=null`, and `has_more=false`. The default Postgres store logs connection/query failures and returns that empty page, so repository retries are not triggered; exceptions that reach the repository produce the same empty page after retry exhaustion. Rows that fail hydration are logged and omitted, while pagination is calculated from the fetched rows before hydration. A short or empty page is therefore not proof that no other matching events exist, and HTTP 200 does not certify a complete result set. See the [Postgres store](../src/homesec/state/postgres.py), [repository](../src/homesec/repository/clip_repository.py), and [clip routes](../src/homesec/api/routes/clips.py). This is a source-supported limitation, not an intended failure policy; focused characterization coverage remains pending.
- **Playback timing coverage is incomplete.** Refresh calculations and retry helpers have tests, but a complete mounted-hook timer test was not identified. Event cards currently acquire media eagerly; activation-only loading remains pending work, and the baseline does not require a prefetch schedule.

### Preview talk

- **Native viewer caps have limited focused evidence.** The implementation enforces caps, but a focused cap test was not identified. Broader camera, codec, browser, microphone, and physical-device QA remains a follow-up; the deployment guides are not a universal compatibility certification.
- **Talk duration limits apply to separate phases.** The reservation timer starts at preparation and is replaced by an active-session timer when activation succeeds. The WebSocket first-message deadline starts after connection/authentication, not at HTTP preparation. These origins are source-supported; a focused total-lifetime timer test was not identified. See the [talk manager](../src/homesec/sources/rtsp/talk/manager.py) and [talk routes](../src/homesec/api/routes/talk.py).

### Integration contracts

- **Production notifier fan-out needs focused mixed-success coverage.** The pipeline dispatches per enabled entry and records individual outcomes. `MultiplexNotifier` helper tests exercise a different path, so they do not establish that one production-entry failure leaves other delivery intact. See the [pipeline](../src/homesec/pipeline/core.py).
- **Plugin examples can lag the current registry.** Current extension contracts use the unified class-based registry and external `homesec.plugins` entry points. Older AGENTS examples use per-type factory/context patterns, and some plugin-guide examples use `type` where the decorator accepts `plugin_type`. Use the [registry source](../src/homesec/plugins/registry.py), [discovery source](../src/homesec/plugins/__init__.py), and [interfaces](../src/homesec/interfaces.py) when reconciling those examples. The baseline does not promise automatic interface verification.
- **Installed plugin discovery and construction boundaries have limited tests.** External entry-point discovery, validate-before-factory behavior, and configuration-only paths that do not construct components are source-supported; dedicated end-to-end/boundary coverage was not identified for all those paths.

### Operations maintenance

- **Health summaries and diagnostics answer different questions.** Their degradation criteria differ. A successful summary is not proof that every camera and storage dependency is healthy; inspect component diagnostics and camera counts. Some architecture-document wording conflates the surfaces. See the [health routes](../src/homesec/api/routes/health.py).
- **Zero statistics can hide a count-query failure.** Individual repository counts can fall back to zero after an earlier availability check passed. Counts are UTC-day clip-level observations; a zero is not proof of no activity. See the [repository](../src/homesec/repository/clip_repository.py).
- **Exit code 42 is not a reachable public setup contract.** `request_restart()` has no production callers, setup activates in-process, and the CLI test forces the restart flag with a mock. The baseline excludes that branch; dead-code cleanup remains separate.
- **Backup recovery does not prove dump integrity or successful restore.** Recovery recognizes artifact names/metadata, and unit tests fake dump output. Missing-executable startup degradation and private artifact permissions are source-supported with limited focused coverage. No real dump/restore round trip was verified for this baseline. Persisting the local backup directory/manifest and performing restore procedures remain operator responsibilities. Retention is not a universal backup disk bound; uncertain interrupted uploads retain recovery handles. Follow the [backup guide](postgres-backups.md).

## API schemas and commands

The [OpenAPI exporter](../src/homesec/api/openapi_export.py) builds the schema from registered API routes and models. The [UI generation script](../ui/scripts/api_codegen.mjs) uses that exporter to produce the committed [OpenAPI JSON](../ui/src/api/generated/openapi.json), [TypeScript schema](../ui/src/api/generated/schema.ts), [types](../ui/src/api/generated/types.ts), and [client](../ui/src/api/generated/client.ts). Those generated files provide exact shapes; the capability specs provide behavior that schemas cannot express.

From the repository root, export a standalone schema:

```bash
uv run --locked python -m homesec.api.openapi_export --output /tmp/homesec-openapi.json
```

Use the existing UI commands to regenerate or check committed contracts:

```bash
pnpm --dir ui api:generate
pnpm --dir ui api:check
```

Configuration fields and validation come from the [core configuration models](../src/homesec/models/config.py), [loader](../src/homesec/config/loader.py), and each plugin's configuration model. There is no complete exported plugin-configuration reference today. Validate an operator's configuration and inspect current CLI options using the existing commands:

```bash
uv run --locked homesec validate --config config.yaml
uv run --locked homesec --help
uv run --locked homesec cleanup --help
```

For later public behavior changes, update the relevant OpenSpec delta, meaningful behavioral coverage, and affected reference documentation together. Internal refactors that preserve public behavior need no invented behavior delta. Review documentation corrections separately from intentional behavior changes.
