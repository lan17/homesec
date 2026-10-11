# capture-recording Specification

## Purpose

Describe how HomeSec accepts local and FTP clips and produces motion-triggered RTSP recordings, including camera unavailability and incomplete recording handling.

## Reference

See [reference notes and known limitations](../../../docs/reference.md#capture-recording) for this capability's operator guides and verification limits.

## Clip identity scope

Within one HomeSec state store, clip IDs share a namespace across all cameras and sources; camera names and source directories do not namespace event identity. Independent clips need distinct IDs, including against previously ingested records. Local-folder and FTP sources use filename stems without a global collision check. Source-local repeat suppression is not a global uniqueness guarantee, and a handoff does not necessarily create a new event record. See the [clip-ID collision and re-ingestion limitations](../../../docs/reference.md#capture-recording) for overwrite, timestamp, and notification-key consequences. Those consequences are existing limitations, not a policy to preserve.

## Requirements

### Requirement: Local folder ready-file ingestion

The local-folder source SHALL accept top-level files matching its `*.mp4` glob after their last modification is at least the configured stability threshold old, identify clips by filename stem, and associate them with the configured camera. Glob matching SHALL follow the host platform's case rules; on Linux/POSIX with case-sensitive matching, uppercase `.MP4` files SHALL remain outside local ingestion.

#### Scenario: Recently modified file waits

- **GIVEN** a running local-folder source and a newly modified `.mp4` file in its watch directory
- **WHEN** the file's modification age is below the stability threshold
- **THEN** the source does not hand off that file for processing yet
- **AND** a later scan can hand it off once the threshold is satisfied

#### Scenario: Eligible local clip is handed off

- **GIVEN** an unprocessed top-level `.mp4` file old enough for acceptance
- **WHEN** the source scans its watch directory
- **THEN** it hands off a clip with that local path, filename stem as clip ID, configured camera name, and `local_folder` source backend

#### Scenario: Other files stay outside local ingestion

- **GIVEN** files ending in `.txt`, `.jpg`, or `.partial` in the watch directory
- **WHEN** the local-folder source scans
- **THEN** it does not hand off those files as clips

#### Scenario: Uppercase suffix is excluded with case-sensitive matching

- **GIVEN** a Linux/POSIX deployment using case-sensitive glob matching and an otherwise ready top-level file ending in `.MP4`
- **WHEN** the local-folder source scans its watch directory
- **THEN** the uppercase suffix does not match `*.mp4` and the file is not handed off

### Requirement: Local folder repeat suppression

The local-folder source SHALL suppress further handoffs for clip IDs it has already accepted during its running instance. It SHALL also suppress IDs loaded from a readable processed-file manifest in the watch directory.

#### Scenario: Repeated scans do not re-emit a clip

- **GIVEN** a local clip has been accepted by a running local-folder source
- **WHEN** later scans find the same clip ID
- **THEN** the source does not hand it off again

#### Scenario: Existing manifest suppresses an accepted clip

- **GIVEN** a readable local processed-file manifest contains a clip ID
- **WHEN** a new source instance scans a matching `.mp4` file
- **THEN** it does not hand off that clip ID again

### Requirement: FTP completed-upload ingestion

The FTP source SHALL hand off completed uploads whose extensions match the configured case-insensitive allowlist. An empty allowlist SHALL accept every extension. It SHALL exclude disallowed files from processing and delete them only when configured to do so.

#### Scenario: Allowed upload becomes a clip

- **GIVEN** an FTP source allows `.mp4` and a complete upload has a `.MP4` suffix
- **WHEN** the completed upload is received
- **THEN** it hands off a clip identified by filename stem, with its configured camera and `ftp` source backend

#### Scenario: Empty allowlist accepts another extension

- **GIVEN** an FTP source has an empty extension allowlist
- **WHEN** a completed upload with an `.avi` suffix is received
- **THEN** it hands off the upload as a clip

#### Scenario: Disallowed upload is excluded

- **GIVEN** an upload has a disallowed extension
- **WHEN** the completed upload is received
- **THEN** it is not handed off as a clip
- **AND** the local file is deleted when `delete_non_matching` is enabled and retained when it is disabled

### Requirement: FTP incomplete-upload handling

The FTP source SHALL exclude incomplete uploads from clip handoff and SHALL delete their local files when `delete_incomplete` is enabled.

#### Scenario: Interrupted upload is discarded or retained by configuration

- **GIVEN** an FTP upload ends before completion
- **WHEN** the source receives the incomplete-upload notification
- **THEN** it does not hand off a clip
- **AND** the partial file is deleted when `delete_incomplete` is enabled and retained when it is disabled

### Requirement: Motion-triggered recording boundaries

An operational RTSP source SHALL start recording on detected motion, refresh its motion stop-delay on subsequent motion, and stop after that delay elapses without motion. While motion remains recent, it SHALL rotate recordings after the configured maximum duration rather than intentionally keeping one indefinitely growing clip.

#### Scenario: New motion starts a recording

- **GIVEN** an RTSP source is monitoring frames, is not recording, and can start its recorder
- **WHEN** motion satisfies the configured idle threshold
- **THEN** a recording starts for that camera

#### Scenario: Motion refreshes the stop delay

- **GIVEN** a camera is recording with a configured stop delay
- **WHEN** further motion is detected before the delay expires
- **THEN** the no-motion countdown restarts from that motion
- **AND** the recording stops after the refreshed delay elapses without further motion

#### Scenario: Recording uses its configured keepalive sensitivity

- **GIVEN** the recording sensitivity factor lowers the motion threshold relative to idle detection
- **WHEN** frame changes exceed the recording threshold but not the idle threshold
- **THEN** they can keep an active recording alive without starting an idle recording

#### Scenario: Ongoing motion rotates the clip

- **GIVEN** a recording has reached its configured maximum duration and motion remains recent
- **WHEN** the next recording can start
- **THEN** a new clip starts and the preceding recording is stopped for handoff
- **AND** the new clip's duration starts when its recorder is ready

### Requirement: RTSP camera unavailability and reconnection

RTSP startup checks SHALL run in the background without camera unavailability itself raising from source start. Startup checks and stream reconnection SHALL follow the configured attempt policy until usable input arrives, shutdown is requested, or a finite limit is exhausted; `max_attempts=0` means no limit. Finite exhaustion SHALL stop that camera's monitoring until runtime reload or process restart, while the runtime process remains available.

#### Scenario: Offline camera does not reject source start

- **GIVEN** the camera is unavailable during startup checks
- **WHEN** the RTSP source starts
- **THEN** source start returns without raising for that camera unavailability
- **AND** the background worker retries according to the configured attempt policy

#### Scenario: Camera becomes available after failed checks

- **GIVEN** background startup checks initially fail and the attempt policy permits another try
- **WHEN** a later startup check succeeds
- **THEN** the source proceeds to motion monitoring

#### Scenario: Finite reconnect attempts are exhausted

- **GIVEN** a running camera has a finite maximum of two reconnect attempts and its input never becomes usable
- **WHEN** both reconnect attempts fail
- **THEN** that camera's source worker ends after cleanup and stops monitoring
- **AND** the runtime process remains available, but later camera availability does not restart monitoring until a runtime reload or process restart creates another source worker

#### Scenario: Finite startup attempts are exhausted

- **GIVEN** a camera remains unavailable through every permitted background startup attempt
- **WHEN** its finite startup attempt limit is exhausted
- **THEN** that camera's source worker ends without starting motion monitoring
- **AND** the runtime process remains available and another source worker must be created by runtime reload or process restart to resume startup checks

#### Scenario: Zero attempt limit permits later recovery

- **GIVEN** `max_attempts=0`, unsuccessful reconnect attempts, and no shutdown request
- **WHEN** a later attempt obtains usable camera input
- **THEN** the source resumes monitoring rather than exhausting a finite attempt budget

### Requirement: RTSP clip handoff eligibility

The RTSP source SHALL hand off a recording only when its local output exists, is nonempty, and recorder stop has not reported failure or raised. For compatibility, the legacy recorder's stop result without an explicit success value SHALL remain eligible for this existence-and-size check; it does not establish native-style container validation.

#### Scenario: Completed recording reaches processing

- **GIVEN** a recording has a nonempty local output and its recorder stops without reporting failure
- **WHEN** recording ends through normal stop, rotation, or recording-health failure
- **THEN** the output is handed off with the camera, local path, clip ID, timestamps, duration, and `rtsp` source backend

#### Scenario: Failed finalization does not deliver unfinished bytes

- **GIVEN** the recorder reports unsuccessful finalization or raises while stopping
- **WHEN** the source ends that recording
- **THEN** it does not hand off the output as a clip
- **AND** the ended recording does not prevent a subsequent recording attempt

#### Scenario: Missing or empty output is not a clip

- **GIVEN** recorder stop has completed but the local output is absent or empty
- **WHEN** the source considers clip handoff
- **THEN** it does not hand off a clip

### Requirement: Native recording publication and compatibility

Native recordings SHALL remain outside completed `.mp4` paths until successful finalization publishes them. An unfinished native recording SHALL NOT be handed off as complete. Recording profiles unsupported by the native path SHALL retain the existing compatible recording path.

#### Scenario: Native writer dies before publication

- **GIVEN** a native recording has only an unfinished `.partial` output
- **WHEN** its owning media process dies before finalization
- **THEN** no completed `.mp4` is published for that recording
- **AND** the unfinished output is not handed off as a clip

#### Scenario: Unsupported profile uses compatible recording

- **GIVEN** the negotiated recording profile is unsupported by native recording
- **WHEN** recording starts
- **THEN** the compatible recorder is used rather than claiming native support for that profile

### Requirement: Ingestion callback failure isolation

Local-folder and FTP sources SHALL contain exceptions from their clip-handoff callback so that one failed handoff does not itself terminate the source. This isolation SHALL NOT imply that the failed clip is automatically delivered again.

#### Scenario: Later local clips still reach the callback

- **GIVEN** a running local-folder source whose clip callback raises on a clip
- **WHEN** another eligible clip appears
- **THEN** the source invokes the callback for the later clip and remains operational
