# operations-maintenance Specification

## Purpose

Describe HomeSec's existing operator contracts for starting, inspecting, stopping, and maintaining the service, including the distinction between database dump completion, upload completion, and recoverable maintenance state.

Applied clip cleanup and media deletion outcomes are specified by the `storage-retention` capability.

## Reference

See [reference notes and known limitations](../../../docs/reference.md#operations-maintenance) for this capability's operator guides and verification limits.

HTTP scenarios use the [shared API preconditions and gate order](../integration-contracts/spec.md#shared-api-preconditions) unless the scenario explicitly tests a failed prerequisite. CLI and direct component scenarios are not HTTP requests.

See the [scenario evidence register](../../../docs/behavior-evidence.md) for the verification level of each scenario.

## Requirements

### Requirement: CLI operation and configuration validation

The CLI SHALL support running HomeSec from a selected configuration path and validating configuration without starting the camera runtime. Validation SHALL report success with configured component information or configuration failure with exit code 1. A runtime configuration failure SHALL also exit with code 1.

#### Scenario: Validate usable configuration

- **GIVEN** an operator supplies a usable configuration file
- **WHEN** the operator invokes CLI validation
- **THEN** the CLI reports valid configuration and configured component information without running the camera pipeline

#### Scenario: Configuration invalid or missing during validation

- **GIVEN** an operator supplies an invalid or missing configuration file
- **WHEN** the operator invokes CLI validation
- **THEN** the CLI reports the configuration error and exits with code 1

#### Scenario: Runtime reports invalid configuration

- **GIVEN** a run encounters a configuration error
- **WHEN** the CLI handles the failure
- **THEN** it reports the configuration error and exits with code 1

### Requirement: Public health summaries

`/health` and `/api/v1/health` SHALL return equivalent public summaries. A running pipeline with reachable Postgres SHALL report `healthy`; unavailable Postgres with a running pipeline SHALL report `degraded` with HTTP 200. A stopped or stale pipeline SHALL report `unhealthy` with HTTP 503. Summaries SHALL include pipeline/database state and healthy-source count. Bootstrap mode SHALL report `setup_required` with unconfigured components and HTTP 200.

#### Scenario: Database unavailable but pipeline running

- **GIVEN** the pipeline is running and Postgres is unavailable
- **WHEN** an operator requests either public health endpoint
- **THEN** the endpoint returns HTTP 200 with `status` equal to `degraded`, pipeline `running`, and Postgres `unavailable`

#### Scenario: Pipeline is not running

- **GIVEN** the pipeline is stopped or its runtime heartbeat is stale
- **WHEN** an operator requests either public health endpoint
- **THEN** the endpoint returns HTTP 503 with `status` equal to `unhealthy` and pipeline `stopped`

#### Scenario: Pipeline and database healthy

- **GIVEN** the pipeline is running and Postgres is reachable
- **WHEN** an operator requests either public health endpoint
- **THEN** the endpoint returns HTTP 200 with `status` equal to `healthy` and the healthy-source count

#### Scenario: Service awaiting initial setup

- **GIVEN** HomeSec is in bootstrap mode
- **WHEN** an operator requests either public health endpoint
- **THEN** the endpoint returns HTTP 200 with `setup_required`, `bootstrap_mode` true, and no configured pipeline, database, or online cameras

### Requirement: Component diagnostics

`/api/v1/diagnostics` SHALL provide uptime, Postgres and storage probe results, and configured-camera health/enabled/heartbeat information, subject to API-key authentication and normal-mode availability. Diagnostics SHALL report `unhealthy` when the pipeline is stopped, otherwise `degraded` for failed component probes or unhealthy enabled cameras, otherwise `healthy`. Disabled cameras SHALL not degrade the overall status. Probe exceptions SHALL become component errors.

#### Scenario: Storage probe raises an exception

- **GIVEN** the pipeline is running and the storage probe raises an exception
- **WHEN** an authorized operator requests diagnostics
- **THEN** diagnostics identifies the storage component error and reports overall `degraded`

#### Scenario: Camera health affects diagnostics

- **GIVEN** the pipeline and component probes are healthy
- **WHEN** diagnostics includes an unhealthy enabled camera or an enabled camera with no active source
- **THEN** overall status is `degraded`

#### Scenario: Disabled camera has no active source

- **GIVEN** the pipeline and component probes are healthy and the only unhealthy camera is disabled
- **WHEN** diagnostics is requested
- **THEN** the disabled camera is reported but does not cause overall degradation

#### Scenario: Pipeline stopped

- **GIVEN** the pipeline is not running
- **WHEN** an authorized operator requests diagnostics
- **THEN** diagnostics reports overall `unhealthy`

### Requirement: Operational system statistics

`/api/v1/stats` SHALL report today's persisted clip creations and distinct clips with affirmative alert decisions since midnight UTC, plus configured/healthy-source camera counts and uptime. It SHALL require normal mode, API-key authentication when enabled, and a successful database availability check. The built-in store counts clips by database record creation time, but counts alerts by affirmative alert-decision event timestamps, distinct by clip ID. An older clip can therefore contribute to today's alert count, independently of current snapshot fields or notifier delivery. Count-query failures after that check MAY return zero without a separate count-error field, so these counters SHALL not guarantee complete activity or successful notification delivery.

#### Scenario: Read current system statistics

- **GIVEN** the database availability check and count queries succeed
- **WHEN** an authorized client requests system statistics
- **THEN** the response contains clip and clip-level alert-decision counts for the UTC day, camera totals/healthy-source count, and uptime
- **AND** alert counts do not count individual notifier deliveries

#### Scenario: Database availability check fails

- **GIVEN** the database availability check reports unavailable
- **WHEN** a client requests system statistics
- **THEN** the API returns HTTP 503 with `DB_UNAVAILABLE`

#### Scenario: Statistics requested before setup

- **GIVEN** HomeSec is in bootstrap mode
- **WHEN** a client requests system statistics
- **THEN** the API returns HTTP 503 with `SETUP_REQUIRED`

#### Scenario: Count query fails after the availability check

- **GIVEN** the database availability check passed but a later clip or alert count query fails
- **WHEN** the API returns system statistics
- **THEN** the corresponding count may be zero without a distinct count-error field
- **AND** that zero is not evidence that no activity occurred

### Requirement: Shutdown ordering and runtime termination

HomeSec SHALL respond to SIGINT and SIGTERM by initiating shutdown. Normal shutdown SHALL stop API request admission before the camera runtime, stop backup scheduling before shared storage closes, and release persistence resources. Runtime termination SHALL give workers a bounded graceful-exit opportunity and escalate to forced termination when they ignore the termination signal. Shutdown SHALL attempt to finish in-flight clip work within the runtime's shutdown limits.

#### Scenario: Normal process shutdown

- **GIVEN** the API, camera runtime, and backup scheduler are active
- **WHEN** the process begins normal shutdown
- **THEN** API shutdown precedes camera-runtime shutdown
- **AND** backup scheduling stops before shared storage is closed

#### Scenario: Worker ignores graceful termination

- **GIVEN** a camera-runtime worker ignores its termination signal
- **WHEN** the graceful-exit timeout expires
- **THEN** HomeSec escalates to forced worker termination

#### Scenario: In-flight clips can finish within shutdown limits

- **GIVEN** a clip is being processed and can finish within the allowed shutdown period
- **WHEN** its pipeline shuts down
- **THEN** shutdown waits for the in-flight clip before completing pipeline shutdown

### Requirement: Cleanup selection and dry run

When its implemented YOLO-shaped recheck configuration can be constructed, CLI cleanup SHALL re-analyze candidate clips whose prior filter result has no detected classes, with optional age and camera filters and configured batch/concurrency settings. Normal pipeline support for an external filter does not imply cleanup compatibility. Empty returned detection, including zero sampled frames, follows the empty-result path; a raised detection exception follows the error path. The recheck-builder and zero-frame safety limits are specified in [storage retention](../storage-retention/spec.md#requirement-cleanup-rechecks-before-deleting-empty-clips). Dry run SHALL be enabled by default and SHALL report proposed outcomes without deleting original local/stored media or changing clip workflow state/events. Cleanup MAY retrieve media and create temporary re-analysis files while evaluating candidates.

#### Scenario: Preview deletion of an empty clip

- **GIVEN** a candidate clip remains empty on re-analysis and cleanup is running with default dry-run behavior
- **WHEN** cleanup evaluates the clip
- **THEN** it reports that the clip would be deleted
- **AND** original media and clip workflow state/events remain unchanged

#### Scenario: Limit the cleanup scope

- **GIVEN** an operator supplies age or camera filters
- **WHEN** cleanup selects candidate clips
- **THEN** candidate selection uses those filters and the requested batch/concurrency settings

### Requirement: Backup opt-in and prerequisites

Postgres backups SHALL be disabled by default. When enabled, HomeSec SHALL schedule dumps using the configured interval, which MUST be at least 15 minutes. Backup availability SHALL depend on access to `pg_dump`; an unavailable backup subsystem SHALL expose its reason and SHALL not prevent application startup. The configured local directory SHALL be resolved relative to the process working directory when not absolute.

#### Scenario: Backups not enabled

- **GIVEN** default maintenance settings
- **WHEN** HomeSec starts
- **THEN** no periodic Postgres dump is scheduled and backup status reports disabled

#### Scenario: Required dump executable unavailable

- **GIVEN** backups are enabled but `pg_dump` cannot be found
- **WHEN** the backup subsystem starts
- **THEN** backup status reports unavailable with a reason while application startup continues

#### Scenario: Invalid backup interval

- **GIVEN** a backup interval shorter than 15 minutes
- **WHEN** maintenance configuration is validated
- **THEN** configuration validation rejects the interval

### Requirement: Backup status and manual requests

The maintenance API SHALL expose backup availability, running state, attempt/success/error information, artifact references, next scheduled run, and pending remote-delete count through its existing response schema. A manual request SHALL return HTTP 202 when accepted, HTTP 409 with `BACKUP_IN_PROGRESS` when another run is active, and HTTP 503 with `BACKUP_DISABLED` or `BACKUP_UNAVAILABLE` when appropriate. Acceptance SHALL mean scheduled work, not completed backup.

#### Scenario: Accept a manual backup

- **GIVEN** backups are enabled, available, and idle
- **WHEN** an authorized operator requests a manual backup
- **THEN** the API returns HTTP 202 with `accepted` true and schedules one backup
- **AND** another concurrent request cannot start a second run

#### Scenario: Manual request cannot run

- **GIVEN** backups are disabled, unavailable, or already running
- **WHEN** an authorized operator requests a manual backup
- **THEN** the response uses the corresponding disabled, unavailable, or in-progress status and error code

### Requirement: Dump artifacts and credential isolation

Backups SHALL produce uniquely named custom-format Postgres dump files in the configured local directory and make a final artifact available only after dump success. Dump files SHALL have private file permissions. The target database and credentials SHALL come from the configured state-store DSN through the dump process environment rather than command arguments; unrelated ambient Postgres connection variables SHALL not override that target. Backup error text SHALL redact DSN passwords.

#### Scenario: Database credentials passed to a dump

- **GIVEN** the configured DSN includes database credentials and ambient Postgres variables point elsewhere
- **WHEN** HomeSec prepares a dump
- **THEN** the dump targets the configured database using environment-based credentials
- **AND** its command arguments contain no DSN password

#### Scenario: Two successful dumps share a timestamp

- **GIVEN** multiple backups start at the same reported timestamp
- **WHEN** they complete successfully
- **THEN** their final artifact names differ and neither overwrites the other

### Requirement: Interrupted dump cleanup

A dump that exceeds its configured timeout or is cancelled SHALL have its process terminated and temporary output removed. Failed or cancelled dumps SHALL be reflected in backup error state and SHALL not expose a partial file as a successful final artifact. Backup scheduler shutdown SHALL wait for active work within its shutdown timeout and cancel remaining backup tasks when that timeout expires.

#### Scenario: Dump times out

- **GIVEN** a dump does not complete within its configured timeout
- **WHEN** the timeout expires
- **THEN** the dump process is terminated, temporary output is removed, and backup status records failure

#### Scenario: Dump cancelled during shutdown

- **GIVEN** a dump is writing temporary output
- **WHEN** its task is cancelled during shutdown
- **THEN** its process is terminated, temporary output is removed, and cancellation is recorded

### Requirement: Backup manifest recovery and upload outcomes

HomeSec SHALL persist backup records and pending storage work in a manifest beside local dumps. Startup SHALL reconcile interrupted records and recover recognized local dump artifacts when the manifest is absent or invalid. Local dump completion and optional upload completion SHALL be recorded separately on the normal bookkeeping path. A later housekeeping exception can still change the current new record from `success` to `failed` even after its final local dump exists. Upload retries require a retained, upload-enabled, not-yet-uploaded record with status `success`; a record reclassified as `failed` is skipped by that retry pass. A final file and a successful record are therefore distinct observations. Failed or interrupted upload attempts SHALL update their in-memory record with retry information while the local artifact is retained. Public backup status exposes manifest summary fields, not every pending record's error. Preserving changed retry information across restart requires a subsequent successful manifest save. Cancellation of startup storage-retry work can bypass that save, leaving older retry information on disk; see the [startup upload-retry cancellation limitation](../../../docs/reference.md#operations-maintenance).

#### Scenario: Recover local backups after manifest corruption

- **GIVEN** recognized local dump artifacts exist and the manifest is absent or invalid
- **WHEN** the backup subsystem initializes
- **THEN** it rebuilds local backup records and identifies the artifacts as completed local dumps
- **AND** recovered artifacts become pending uploads when backups and uploading are enabled

#### Scenario: Local dump succeeds but upload fails

- **GIVEN** backup uploads are enabled, subsequent bookkeeping/manifest saving succeeds, and the new artifact remains retained
- **WHEN** a dump completes and its upload fails
- **THEN** local dump success is recorded separately from the upload failure
- **AND** retained local output remains eligible for an upload retry

#### Scenario: Housekeeping failure reclassifies a completed dump

- **GIVEN** a new dump has been finalized, its upload failed, and pending-upload retention attempts to unlink an older artifact
- **WHEN** that older unlink raises and the new run's error bookkeeping and manifest save succeed
- **THEN** the new record is marked `failed` despite its completed final dump remaining on disk
- **AND** a later pending-upload retry pass skips that record because its status is not `success`

### Requirement: Backup retention and remote-delete recovery

Backup retention SHALL attempt pruning of the oldest eligible completed backups beyond `keep_count`, deleting local files and known uploaded objects when those deletion operations succeed. Local unlink or bookkeeping errors can interrupt the pass and affect the current run's classification as described above. A failed new upload SHALL preserve prior uploaded backups; ordinary pending local uploads SHALL have their own `keep_count` limit. Pending-upload pruning SHALL preserve retained local artifacts marked `pending_upload_unknown`. That protection survives restart only when the unknown-outcome marker is available in the recovered manifest; cancellation does not guarantee it was saved. Failed or interrupted remote deletion SHALL persist pending work for a later retry.

#### Scenario: Retention after successful upload

- **GIVEN** completed uploaded backups exceed `keep_count`
- **WHEN** a successful backup permits retention to run and the selected deletion operations succeed
- **THEN** the oldest eligible local artifact and known uploaded object are deleted

#### Scenario: Repeated upload failures

- **GIVEN** a prior uploaded backup exists and later uploads repeatedly fail
- **WHEN** ordinary pending local uploads exceed `keep_count` and their selected local deletions succeed
- **THEN** the oldest pending local artifacts are pruned while prior uploaded backups are preserved

#### Scenario: Interrupted upload has uncertain remote outcome

- **GIVEN** an upload was cancelled after a remote object may have been created, its retained local artifact is marked `pending_upload_unknown`, and that marker has been saved and recovered if a restart occurs
- **WHEN** later upload failures cause pending-upload pruning without resolving the unknown outcome
- **THEN** the marked local retry handle is preserved
- **AND** this does not promise protection after startup-retry cancellation leaves only an ordinary pending-upload marker on disk

#### Scenario: Remote deletion fails or is interrupted

- **GIVEN** retention has selected a known uploaded object for deletion
- **WHEN** remote deletion fails or is cancelled
- **THEN** pending remote-deletion information remains in the manifest for a later retry
