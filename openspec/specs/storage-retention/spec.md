# storage-retention Specification

## Purpose

Describe clip storage outcomes, bounded upload retries, local retention eligibility and limits, and API and cleanup deletion behavior.

## Reference

See [reference notes and known limitations](../../../docs/reference.md#storage-retention) for this capability's operator guides and verification limits.

## Requirements

### Requirement: Clip destination and upload outcome

HomeSec SHALL build clip destinations under the configured clip-storage directory using sanitized camera-name and local-filename segments. Each segment is trimmed, path separators are replaced with underscores, whitespace-separated words are joined with underscores, and an empty result becomes `unknown`. The mapping is not unique: distinct names may produce the same destination, and this builder does not reject collisions. Successful uploads SHALL supply the backend's storage URI and optional view URL to downstream processing. The upload operation SHALL NOT itself delete its local input.

#### Scenario: Stored clip retains its camera grouping

- **GIVEN** clip storage is configured with a `clips` directory and a clip belongs to camera `front door`
- **WHEN** a local file named `motion clip.mp4` is uploaded successfully
- **THEN** the storage destination is `clips/front_door/motion_clip.mp4` beneath the configured storage root
- **AND** the returned storage URI and any backend-provided view URL are available to downstream processing
- **AND** uploading does not itself remove the original local file

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

HomeSec SHALL retry failed upload attempts up to the configured maximum, using the configured exponential backoff between attempts. Exhaustion SHALL yield an upload-failure outcome with no storage URI or view URL; it SHALL NOT by itself prevent filter, analysis, or alert processing from continuing under their own rules.

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

Local retention SHALL delete only discovered local clip files with matching recorded local paths, a recorded storage URI, processing status `DONE`, and a readable size. It SHALL preserve files without those prerequisites. Retention deletion SHALL remove only the local file and SHALL NOT delete the stored copy or mark the clip as deleted from event history.

#### Scenario: Uploaded completed clip is eligible

- **GIVEN** a discovered local clip has matching recorded path, storage URI, status `DONE`, and a readable size
- **WHEN** retention selects it for deletion
- **THEN** its local file is removed
- **AND** its storage copy and clip state remain available

#### Scenario: Unconfirmed local files are preserved

- **GIVEN** discovered clips include a local-only clip, an unfinished clip, a clip without state, and a clip whose recorded local path differs
- **WHEN** retention evaluates deletion candidates
- **THEN** those files are excluded from deletion even if local storage is over its configured limit

### Requirement: Oldest eligible clip pruning

When a retention pass selects clips to reduce an eligible-byte total exceeding the configured limit, it SHALL delete eligible clips oldest first by recorded creation time, breaking ties by clip ID. It SHALL stop once that total is within the limit. Its summary SHALL distinguish incomplete measurement and remaining over-limit local usage from successful reclamation.

#### Scenario: Oldest eligible clip is removed first

- **GIVEN** three eligible clips total 150 bytes with oldest, middle, and newest sizes of 60, 50, and 40 bytes, and a 90-byte limit
- **WHEN** a retention pass prunes those clips
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

The clip-deletion API SHALL mark a known clip deleted before deleting its storage object when a URI is available, then return the clip response on success. Missing clips established by a readable state lookup SHALL return HTTP 404. Marking or storage-deletion failures SHALL return HTTP 500 with distinct reasons; the API SHALL NOT issue a rollback of the earlier deletion mark when storage deletion fails. This ordering is not a concurrency-safe terminal-state guarantee: competing snapshot writes or re-ingestion of the same clip ID can overwrite `DELETED`. The response describes the deletion operation's snapshot, not a guarantee that subsequent reads retain it; see the [deletion limitations](../../../docs/reference.md#storage-retention).

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

Cleanup SHALL recheck candidate clips with an existing empty detection result before deleting them. If rechecking finds detections, it SHALL retain the clip. If rechecking remains empty, non-dry-run cleanup SHALL attempt local and stored-copy deletion and attempt to record deletion only after those deletion attempts succeed. Dry-run cleanup SHALL NOT delete clip copies or update their workflow state. Workflow writes remain best-effort and subject to the same concurrent-snapshot and reused-ID limitations as processing and API deletion.

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
