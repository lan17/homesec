# Spec Delta

## ADDED Requirements

### Requirement: Camera inspection is independent of installed media tools

RTSP stream discovery, startup validation, detect-stream recovery inspection, and preview metadata inspection SHALL use HomeSec's bundled media implementation without launching `ffmpeg` or `ffprobe`. Inspection SHALL preserve ordered candidate deduplication, first-video/first-audio selection, codec names, dimensions, and available frame-rate values. Metadata discovery SHALL remain distinct from media execution eligibility.

#### Scenario: Discover streams without CLI tools

- **GIVEN** a reachable camera and no `ffmpeg` or `ffprobe` executable available
- **WHEN** HomeSec discovers its candidate streams
- **THEN** it returns the first video and optional first audio metadata for each candidate, preserving dimensions and available numeric and rational frame-rate values
- **AND** no media CLI is launched or required

#### Scenario: Preserve candidate order and partial results

- **GIVEN** repeated candidate URLs and a mixture of reachable and failed candidates
- **WHEN** discovery completes
- **THEN** each unique candidate is attempted in its original order and successful metadata is retained alongside per-candidate failures
- **AND** a missing or invalid frame-rate value is not invented

#### Scenario: Reachable media exceeds a recorder's capability

- **GIVEN** a camera exposes valid video metadata for a codec unsupported by a particular recording path
- **WHEN** HomeSec inspects that stream
- **THEN** it returns the observed metadata without requiring that recorder to accept the codec or receive a keyframe first
- **AND** any later unsupported-operation result remains distinguishable from connection failure

#### Scenario: Invalid video metadata does not become usable input

- **GIVEN** a responsive stream has no video or cannot provide the dimensions required for stream selection
- **WHEN** inspection completes
- **THEN** HomeSec reports the corresponding bounded inspection failure rather than treating that candidate as usable video

#### Scenario: Recovery and preview planning share native inspection

- **GIVEN** detect-stream recovery or HLS video/audio planning needs fresh camera metadata
- **WHEN** it inspects the camera with absent CLI tools or an older incompatible CLI earlier on `PATH`
- **THEN** inspection uses the bundled implementation and returns the metadata or bounded failure needed by the existing consumer
- **AND** HLS transport selection and recovery retry policy remain unchanged

### Requirement: Native preflight preserves recording negotiation

RTSP preflight SHALL retain existing stream-selection and recording-profile ordering while validating candidates without media CLI processes. A selected profile SHALL have successful bounded media validation and nonempty finalized trial output. Copy-audio, AAC conversion, silent fallback, and timestamp retry behavior SHALL remain supported; missing bundled capabilities SHALL NOT silently remove an otherwise supported candidate.

#### Scenario: Select motion and recording streams

- **GIVEN** multiple usable streams with different dimensions and frame rates
- **WHEN** preflight selects its motion and recording inputs
- **THEN** it retains the existing low-resolution motion and high-resolution recording ordering, including deterministic tie-breaking

#### Scenario: Preserve audio negotiation order

- **GIVEN** a video stream with audio eligible for MP4 copy
- **WHEN** recording validation negotiates a profile
- **THEN** it tries video/audio copy before AAC audio conversion and silent fallback
- **AND** for audio ineligible for copy it tries AAC conversion before silent fallback, while an audio-free stream uses the existing silent candidate

#### Scenario: Missing implementation support does not remove audio

- **GIVEN** an otherwise supported audio profile lacks a required component in an incomplete bundled build
- **WHEN** preflight attempts that profile
- **THEN** it reports a local capability failure rather than misclassifying it as a camera failure or silently selecting no audio because of that missing component
- **AND** that build does not satisfy the compatibility acceptance gate

#### Scenario: Timestamp retry keeps the more stable successful profile

- **GIVEN** a successful baseline trial reaches the existing timestamp-instability retry threshold
- **WHEN** preflight tries wallclock timestamps
- **THEN** it selects that mode only if validation succeeds and reduces the measured instability score
- **AND** a failed or no-better retry retains the successful baseline mode

#### Scenario: Failed or empty trial cannot pass

- **GIVEN** a trial fails to write media, produces an empty output, or fails finalization
- **WHEN** preflight evaluates the trial
- **THEN** it does not report successful recording validation
- **AND** trial output is removed and is never delivered to the clip pipeline

### Requirement: Media capability results reflect tested support

Capability inspection SHALL distinguish linked-build support, available host devices, and successful validation of a requested operation. Existing supported codec and hardware configurations SHALL retain native validation coverage. A timeout or device listing alone SHALL NOT establish successful decode. Capability results SHALL identify local limitations separately from camera connectivity and SHALL NOT claim that a different media executor was validated.

#### Scenario: Inspect the linked build rather than a host CLI

- **GIVEN** host media executables expose different features or are absent
- **WHEN** HomeSec reports native media capabilities
- **THEN** results reflect its bundled implementation and host resources, with no feature inferred from those executables

#### Scenario: Hardware must produce decode evidence

- **GIVEN** an available hardware backend and a selected camera stream
- **WHEN** the bounded hardware validation completes
- **THEN** success requires actual decoded frames through that backend
- **AND** unavailable devices, unsupported operations, failures, and timeouts do not become successful hardware validation; software decoding remains available as an explicit fallback

#### Scenario: Native hardware validation succeeds but compatible execution fails

- **GIVEN** native hardware validation succeeds but the remaining compatible motion executor rejects that backend or fails before usable frames arrive within its startup deadline
- **WHEN** motion input starts
- **THEN** it makes at most one software retry for that input attempt and remembers the hardware failure for that camera worker's lifetime
- **AND** a failed software retry follows the existing input retry/exhaustion policy without cycling between hardware and software or claiming that native validation certified the compatible executor

#### Scenario: Preserve compatible execution during staged migration

- **GIVEN** a negotiated profile uses the existing compatible recorder rather than native recording
- **WHEN** native preflight validates the profile
- **THEN** native validation does not remove that compatible execution path or certify the separate executor's capabilities
- **AND** remaining execution dependencies stay explicit until their migration is complete

### Requirement: Probe resources and diagnostics are bounded

Camera probes SHALL enforce connection, I/O, operation, and cleanup bounds. Completion, failure, timeout, cancellation, and runtime shutdown SHALL close local inputs, stop owned work, and remove trial artifacts. Stuck work SHALL be terminated under an outer lifetime bound. Cleanup failures SHALL be reported without claiming success. Probe diagnostics SHALL be bounded and exclude credentials, tokens, and raw media.

#### Scenario: Connect or media input stalls

- **GIVEN** connection, name resolution, or media input does not complete
- **WHEN** the applicable deadline expires
- **THEN** owned work is interrupted or terminated and cleaned up within the overall lifetime bound
- **AND** the result reports a timeout without blocking the control runtime indefinitely

#### Scenario: Cancel a probe or shut down its runtime

- **GIVEN** a probe owns camera input and a temporary trial artifact
- **WHEN** its caller cancels or runtime shutdown requests stop
- **THEN** the probe stops opening inputs and closes local sessions or terminates their owner before cleanup is reported complete
- **AND** temporary artifacts are removed after their writers stop; remote protocol teardown remains best-effort within the cleanup bound

#### Scenario: Native inspection crashes or returns invalid output

- **GIVEN** inspection exits unexpectedly or emits malformed or oversized control output
- **WHEN** its owner handles the failure
- **THEN** it reports a bounded local inspection failure, cleans up owned work and files, and does not treat the reply as successful camera validation

#### Scenario: Secrets stay outside diagnostics

- **GIVEN** a camera URL contains credentials or a media operation handles private frames
- **WHEN** a probe succeeds or fails
- **THEN** process arguments, logs, errors, and persisted diagnostics contain neither those credentials or tokens nor raw media

### Requirement: Preflight concurrency requires classified evidence

CLI-free preflight SHALL preserve the existing independent overlapping input checks and their session classification. Successful concurrency SHALL require usable media and successful bounded completion for the checked roles. Explicit camera session-limit evidence SHALL remain distinct from authentication, timeout, codec, and generic failures. Preview checks SHALL retain recording-first behavior without stopping an existing recording to run inspection.

#### Scenario: Equal URLs still represent independent checked roles

- **GIVEN** the existing preflight topology checks motion, recording, and optionally preview or preview metadata roles, with some equal URLs
- **WHEN** it tests overlap
- **THEN** it retains the independent opens for those roles rather than inferring shared-session capacity from URL equality

#### Scenario: A process being alive is insufficient evidence

- **GIVEN** checked roles remain open during the overlap interval but one later fails or never produces the required media
- **WHEN** the concurrency result is finalized
- **THEN** it does not report supported concurrency

#### Scenario: Capacity rejection preserves recording priority

- **GIVEN** an optional preview overlap check receives explicit session-limit evidence
- **WHEN** preflight classifies preview concurrency
- **THEN** it retains the recording-first downgrade and existing stable refusal reason
- **AND** it closes optional inspection inputs without stopping an active recording

#### Scenario: Generic failures do not prove a session limit

- **GIVEN** optional preview validation fails with a generic network, authentication, decode, or timeout error
- **WHEN** preflight classifies concurrency
- **THEN** it leaves preview concurrency unclassified and retains existing best-effort behavior rather than reporting proven session-budget exhaustion

#### Scenario: Failed startup preserves background retry behavior

- **GIVEN** native startup inspection fails for one camera
- **WHEN** its configured attempt policy permits another attempt or reaches finite exhaustion
- **THEN** the existing background retry and finite-exhaustion behavior applies while the control runtime remains available
- **AND** `max_attempts=0` continues to permit later recovery until shutdown
