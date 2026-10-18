# storage-retention Specification

## Purpose

Describe clip storage outcomes, bounded upload retries, local retention eligibility and limits, and API and cleanup deletion behavior.

## Reference

See [reference notes and known limitations](../../../docs/reference.md#storage-retention) for this capability's operator guides and verification limits.

HTTP scenarios use the [shared API preconditions and gate order](../integration-contracts/spec.md#shared-api-preconditions) unless the scenario explicitly tests a failed prerequisite. CLI and direct component scenarios are not HTTP requests.

See the [scenario evidence register](../../../docs/behavior-evidence.md) for the verification level of each scenario.

## Requirements

### Requirement: Clip destination and upload outcome

HomeSec SHALL build paths under the configured clip directory from camera/filename segments: trim, replace separators with underscores, join whitespace-separated words with underscores, and use `unknown` when empty. `PurePosixPath` may remove dot segments; absolute paths and retained parent traversal SHALL be rejected. Collisions are not rejected. Preparation errors can escape before retries. Successful uploads SHALL supply URI/optional view URL without deleting local input.

#### Scenario: Stored clip retains its camera grouping

- **GIVEN** clip storage is configured with a `clips` directory and a clip belongs to camera `front door`
- **WHEN** a local file named `motion clip.mp4` is uploaded successfully
- **THEN** the storage destination is `clips/front_door/motion_clip.mp4` beneath the configured storage root
- **AND** the returned storage URI and any backend-provided view URL are available to downstream processing
- **AND** uploading does not itself remove the original local file

#### Scenario: Dot camera segment is normalized away

- **GIVEN** a clip for camera `.`, filename `motion.mp4`, and clip directory `clips`
- **WHEN** its destination is built
- **THEN** the destination is `clips/motion.mp4` without a distinct camera segment

#### Scenario: Parent camera segment fails before an upload attempt

- **GIVEN** a clip for camera `..`, filename `motion.mp4`, and clip directory `clips`
- **WHEN** its upload stage builds the destination
- **THEN** path validation raises before any storage attempt or retry, instead of returning `UploadError`

### Requirement: Built-in storage delivery and destination validation

The built-in local and Dropbox backends SHALL construct root-relative destinations and reject empty destinations, backslash-containing destinations, and parent-traversal segments. Local storage SHALL return a `local:` URI and file view URL; Dropbox SHALL return a `dropbox:` URI and a view URL derived from its configured web prefix and destination path.

#### Scenario: Local storage round trip

- **GIVEN** a local-storage root and a valid destination
- **WHEN** a file is stored and then retrieved using its returned URI
- **THEN** the retrieved bytes match the stored bytes
- **AND** the upload result identifies the stored local path and its file view URL

#### Scenario: Dropbox view URL follows the configured prefix

- **GIVEN** a Dropbox storage root and configured web URL prefix
- **WHEN** a clip upload succeeds
- **THEN** the result identifies the destination with a `dropbox:` URI
- **AND** its view URL combines the configured web prefix and Dropbox destination path without creating a public share link

#### Scenario: Unsafe destination is refused

- **GIVEN** either built-in storage backend
- **WHEN** a caller attempts an empty destination, a destination containing a backslash, or a parent-traversal segment
- **THEN** the upload is rejected rather than writing to that destination

### Requirement: Dropbox credential selection

The Dropbox backend SHALL prefer a present configured access-token environment variable. If it is absent, the backend SHALL accept a complete configured app-key, app-secret, and refresh-token environment-variable set. Missing or incomplete credentials SHALL reject backend construction.

#### Scenario: Access token takes precedence

- **GIVEN** the configured access-token environment variable is present
- **WHEN** Dropbox storage is constructed
- **THEN** the access token is used for its authentication

#### Scenario: Complete refresh credentials are accepted

- **GIVEN** the access token is absent and all three configured refresh-authentication values are present
- **WHEN** Dropbox storage is constructed
- **THEN** refresh-token authentication is used

#### Scenario: Incomplete credentials fail setup

- **GIVEN** neither an access token nor a complete refresh-authentication set is available
- **WHEN** Dropbox storage is constructed
- **THEN** construction fails with a missing-credentials error

### Requirement: Bounded upload retries and failure outcome

After successful destination preparation, HomeSec SHALL retry failures from protected upload attempts up to the configured maximum, using the configured exponential backoff between attempts. The shared runner bounds attempts, not elapsed time, and adds no universal operation deadline. Exhaustion SHALL yield an upload-failure outcome with no storage URI or view URL; it SHALL NOT by itself prevent filter, analysis, or alert processing from continuing under their own rules.

#### Scenario: Transient upload failure recovers

- **GIVEN** the first upload fails and another attempt is permitted
- **WHEN** the next upload attempt succeeds
- **THEN** processing uses that attempt's storage URI and any backend-provided view URL

#### Scenario: Upload failure survives retry exhaustion

- **GIVEN** storage fails for every permitted upload attempt and filtering succeeds
- **WHEN** the upload attempt limit is exhausted
- **THEN** downstream processing receives no storage URI or view URL and an upload-failure indication
- **AND** analysis and alert evaluation can still complete

### Requirement: Local retention deletion eligibility

Local retention SHALL select only discovered files whose selection-time snapshot has a matching recorded local path, storage URI, status `DONE`, and readable scanned size. It SHALL NOT delete through storage backends or mutate state/events. Unlink uses the selected pathname and captured size without revalidation; replacements or aliased stored copies can be removed. See the [selection limits](../../../docs/reference.md#storage-retention).

#### Scenario: Uploaded completed clip is eligible

- **GIVEN** a discovered local clip has matching recorded path, storage URI, status `DONE`, a readable size, and a physically separate stored copy, with no replacement or competing mutation
- **WHEN** retention selects it for deletion and unlink succeeds
- **THEN** its local file is removed
- **AND** its storage copy and clip state remain available

#### Scenario: Unconfirmed local files are preserved

- **GIVEN** discovered clips include a local-only clip, an unfinished clip, a clip without state, and a clip whose recorded local path differs
- **WHEN** retention evaluates deletion candidates
- **THEN** those files are excluded from deletion even if local storage is over its configured limit

#### Scenario: Selected pathname is replaced before unlink

- **GIVEN** retention selected a three-byte completed uploaded file and a different file then replaces it at the same pathname
- **WHEN** the deletion worker unlinks that selected pathname successfully
- **THEN** the replacement is removed without rechecking its identity or current workflow status
- **AND** the pass uses the captured three-byte size for its accounting

### Requirement: Oldest eligible clip pruning

When eligible bytes exceed the cap, retention SHALL attempt deletion oldest first by recorded creation time, then clip ID, until captured-size accounting is within the cap or candidates run out. Unlink errors can leave usage above the cap. Summaries SHALL distinguish incomplete measurement and remaining over-limit local usage from reclamation; scanned-size accounting is not an atomic post-deletion filesystem measurement.

#### Scenario: Oldest eligible clip is removed first

- **GIVEN** three eligible clips total 150 bytes with oldest, middle, and newest sizes of 60, 50, and 40 bytes, and a 90-byte limit
- **WHEN** a retention pass prunes those unchanged files and deletion succeeds
- **THEN** the oldest clip is removed and the two newer clips remain
- **AND** the summary reports 60 reclaimed bytes and 90 eligible bytes remaining

#### Scenario: Equal creation times have deterministic order

- **GIVEN** eligible clips have the same recorded creation time
- **WHEN** the retention pass must choose which to remove first
- **THEN** it orders those clips by clip ID

#### Scenario: Size failure prevents an all-clear claim

- **GIVEN** a discovered local file cannot be measured
- **WHEN** the retention pass produces its summary
- **THEN** it reports incomplete measurement and does not claim local usage is safely below the limit

#### Scenario: Protected files can leave usage over the limit

- **GIVEN** ineligible local clips remain after eligible clips have been pruned
- **WHEN** measured remaining local usage still exceeds the configured limit
- **THEN** the summary reports that usage remains blocked over the limit
- **AND** the protected files remain intact

### Requirement: Retention triggering and failure isolation

Clip arrivals SHALL register their local parent directories for retention scanning and request a background prune. While a prune is active, overlapping requests SHALL be dropped. Retention failures SHALL be reported without aborting clip processing.

#### Scenario: New clip discovers a local retention directory

- **GIVEN** a clip arrives from a previously unregistered local directory
- **WHEN** its processing is scheduled
- **THEN** its parent directory is registered and a retention pass is requested
- **AND** upload success is not required to request that pass

#### Scenario: Overlapping prune request is dropped

- **GIVEN** a retention pass is still active
- **WHEN** another clip arrival requests pruning
- **THEN** no overlapping pass starts for that request

#### Scenario: Pruning fails independently of processing

- **GIVEN** a retention pass raises an error
- **WHEN** clip processing continues
- **THEN** the retention error is reported and does not abort that clip's processing

### Requirement: Built-in storage deletion is idempotent

The local and Dropbox storage backends SHALL treat deletion of an already-missing file as successful. They SHALL surface other deletion failures rather than claiming the stored file was removed.

#### Scenario: Stored file is deleted twice

- **GIVEN** a valid storage URI identifies a file that is removed by its first deletion
- **WHEN** deletion is requested again through the same built-in backend
- **THEN** the repeated deletion succeeds without requiring the file to exist

### Requirement: API clip deletion reports partial failures

The API SHALL mark a known clip deleted before deleting its available storage object and return the clip response on success. Readable lookups establishing absence SHALL return 404; unexpected marking or storage-deletion failures SHALL return distinct 500 errors, without rolling back an earlier mark. Competing writes or ID reuse can overwrite `DELETED`; see the [deletion limits](../../../docs/reference.md#storage-retention).

#### Scenario: Delete a known uploaded clip

- **GIVEN** a known clip with a stored object, successful persistence and storage operations, and no concurrent state mutation or re-ingestion of its ID
- **WHEN** a client deletes that clip through the API
- **THEN** the clip is marked deleted and the stored object is deleted
- **AND** the response returns the clip data with HTTP 200

#### Scenario: Delete a clip without a stored object

- **GIVEN** a known clip has no storage URI and its deletion mark succeeds
- **WHEN** a client deletes it through the API
- **THEN** the API returns the deleted clip response without requesting storage deletion

#### Scenario: Delete a missing clip

- **GIVEN** a readable state lookup establishes that the requested clip does not exist
- **WHEN** a client deletes it through the API
- **THEN** the response is HTTP 404 with `CLIP_NOT_FOUND`

#### Scenario: Marking deletion fails

- **GIVEN** a known clip whose deletion mark raises an unexpected failure
- **WHEN** a client deletes it through the API
- **THEN** the response is HTTP 500 with `CLIP_DELETE_MARK_FAILED`
- **AND** storage deletion is not attempted

#### Scenario: Storage deletion fails after marking

- **GIVEN** a known uploaded clip is marked deleted but storage deletion fails
- **WHEN** the API returns the operation result
- **THEN** the response is HTTP 500 with `CLIP_STORAGE_DELETE_FAILED`
- **AND** the API does not roll back the earlier deletion mark; competing writes or ID reuse can still overwrite it

### Requirement: Cleanup rechecks before deleting empty clips

Cleanup SHALL recheck prior empty detections when its YOLO-shaped configuration can be built. Detections SHALL preserve media. For empty results, including zero frames, applied cleanup SHALL attempt local/stored deletion and then best-effort deletion recording only after deletion attempts succeed. Dry run SHALL preserve original media and workflow state. See the [configuration and safety limits](../../../docs/reference.md#storage-retention).

#### Scenario: Recheck rescues a previous empty detection

- **GIVEN** a cleanup candidate previously had no detected classes, workflow persistence is available, and no competing mutation or re-ingestion changes its state
- **WHEN** rechecking detects a class
- **THEN** the clip copies are retained
- **AND** a non-dry-run cleanup records the rechecked detection result rather than a deletion

#### Scenario: Empty clip is deleted after recheck

- **GIVEN** a cleanup candidate has available media, rechecking finds no detected classes, workflow persistence succeeds, and no competing mutation or re-ingestion changes its state
- **WHEN** non-dry-run cleanup successfully deletes its existing local and stored copies
- **THEN** it records the clip's deleted status and deletion event

#### Scenario: Dry run reports without applying deletion

- **GIVEN** a candidate still has no detected classes after rechecking
- **WHEN** cleanup runs in dry-run mode
- **THEN** it reports the planned deletion while retaining the original local and stored copies and their workflow state

#### Scenario: Recheck or deletion failure is reported

- **GIVEN** obtaining or rechecking media fails, or an attempted deletion fails
- **WHEN** cleanup handles the candidate
- **THEN** it reports the failure without claiming successful clip deletion
- **AND** a deletion failure does not roll back a copy already removed by another deletion attempt

#### Scenario: Cleanup receives an empty zero-frame recheck

- **GIVEN** a cleanup candidate's recheck returns empty detected classes with `sampled_frames: 0` rather than raising
- **WHEN** applied cleanup handles that returned result
- **THEN** it follows the empty-result deletion path without requiring positive-frame inspection
- **AND** dry-run cleanup instead retains original copies and workflow state; the applied behavior is a current safety limitation, not a recommended deletion policy

#### Scenario: Cleanup cannot build an external filter configuration

- **GIVEN** a normally supported filter whose configuration dictionary is invalid as `YoloFilterConfig`, or whose configuration is a different model type
- **WHEN** CLI cleanup builds its recheck configuration
- **THEN** construction fails before candidate re-analysis rather than providing generic plugin-compatible cleanup
