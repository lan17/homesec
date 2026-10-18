# analysis-alerts Specification

## Purpose

Describe how completed clips are filtered and analyzed, how alert eligibility is decided, and how processing handles skipped stages and partial failures.

## Reference

See [reference notes and known limitations](../../../docs/reference.md#analysis-alerts) for this capability's operator guides and verification limits.

See the [scenario evidence register](../../../docs/behavior-evidence.md) for the verification level of each scenario.

## Requirements

### Requirement: Clip processing stage ordering

HomeSec SHALL start upload and filtering independently for accepted clips. After successful filtering, eligible analysis SHALL start without waiting for upload. Alert handling SHALL await the eventual upload outcome; filter failure SHALL collect that outcome and stop analysis and notification. These rules assume joined stages return handled outcomes; destination-preparation exceptions can escape before the retry boundary and abort the later join.

#### Scenario: Analysis overlaps a slow upload

- **GIVEN** an accepted clip whose filter succeeds quickly and whose upload remains in progress
- **WHEN** its analysis run mode permits analysis
- **THEN** analysis can run before upload finishes
- **AND** subsequent alert processing uses the completed upload outcome

#### Scenario: Filter failure stops analysis and notification

- **GIVEN** a clip's filtering fails after its permitted attempts
- **WHEN** processing handles the failure
- **THEN** the outstanding upload is allowed to finish and its outcome is handled
- **AND** analysis and notification are not attempted for that processing run
- **AND** processing attempts to record the filter failure and does not attempt successful processing completion for that run
- **AND** the recorded failure status is subject to the concurrent-write limitations below

#### Scenario: Upload preparation aborts the later join

- **GIVEN** filtering succeeds, the upload task raises during destination preparation, and any eligible analysis finishes
- **WHEN** processing awaits that upload task before alert evaluation
- **THEN** the exception propagates instead of becoming a returned noncritical upload-failure outcome
- **AND** that run does not reach alert evaluation, notification, or successful completion recording through this path

### Requirement: Supported object-filter results

The built-in object filter SHALL apply supported target classes, confidence threshold, and minimum box-height ratio to sampled frames. Results SHALL expose detected classes, confidence, model, and sampled-frame count, with no classes and zero confidence when no boxes are accepted. Empty or partial returned results SHALL NOT certify complete inspection or safety for deletion; see the [decoding limits](../../../docs/reference.md#analysis-alerts).

#### Scenario: Accepted target detection is returned

- **GIVEN** a sampled frame contains a configured supported target detection that passes confidence and box-height constraints
- **WHEN** the object filter evaluates the clip
- **THEN** the result contains the detected class and its accepted confidence with model and sampled-frame information

#### Scenario: Small detections are excluded

- **GIVEN** detections have boxes shorter than the configured minimum fraction of frame height
- **WHEN** the object filter evaluates the clip and finds no other accepted boxes
- **THEN** it returns no detected classes and zero confidence

#### Scenario: Zero decoded frames return an empty detector result

- **GIVEN** the detector dependencies and model are available and capture reports zero frames or its first read returns failure rather than raising
- **WHEN** the built-in detector finishes its sampling loop
- **THEN** it returns empty detected classes, zero confidence, and `sampled_frames` equal to zero rather than a decoding exception
- **AND** this returned result is a current decoding limitation, not evidence of safe-to-delete footage

#### Scenario: A later failed read returns accumulated detection

- **GIVEN** some frames have been sampled, the early-exit condition has not been reached, and a later read returns failure rather than raising
- **WHEN** the detector ends its loop
- **THEN** it returns the classes, confidence, and sampled-frame count accumulated so far without an incomplete-decoding flag

### Requirement: Analysis run modes and skip reasons

HomeSec SHALL run analysis after successful filtering in `always` mode, skip it in `never` mode, and in `trigger_only` mode run it only when a detected class matches a configured trigger class. Skipped analysis SHALL distinguish intentional disabling from absence of trigger classes in its available lifecycle history.

#### Scenario: Trigger detection permits analysis

- **GIVEN** analysis uses `trigger_only` mode with `person` as a trigger class
- **WHEN** successful filtering detects `person`
- **THEN** analysis is attempted for the clip

#### Scenario: Non-trigger detection skips analysis

- **GIVEN** analysis uses `trigger_only` mode and filtering detects no configured trigger class
- **WHEN** the analysis stage is considered
- **THEN** analysis is not attempted
- **AND** available lifecycle history records `no_trigger_classes` as its skip reason

#### Scenario: Always mode ignores trigger eligibility

- **GIVEN** analysis uses `always` mode and filtering succeeds without a configured trigger detection
- **WHEN** the analysis stage is considered
- **THEN** analysis is attempted

#### Scenario: Never mode disables analysis requests

- **GIVEN** analysis uses `never` mode even though filtering detects a trigger class
- **WHEN** the analysis stage is considered
- **THEN** no analysis request is made
- **AND** available lifecycle history records `run_mode_never` as its skip reason

### Requirement: OpenAI-compatible analysis validation

The OpenAI-compatible analyzer SHALL submit timestamped sampled images within configured preprocessing limits and accept only its structured sequence-analysis contract. Successful output SHALL provide risk, activity, summary, and detailed analysis. Provider errors, malformed output, and no readable frames SHALL fail analysis. Schema validation SHALL NOT certify factual or temporal accuracy.

#### Scenario: Valid structured response supplies analysis

- **GIVEN** a readable clip and a provider returning valid structured sequence analysis
- **WHEN** analysis succeeds
- **THEN** downstream processing receives the response's maximum risk, primary activity, sequence summary, and detailed analysis

#### Scenario: Preprocessing bounds submitted images

- **GIVEN** a readable clip exceeds the configured frame-count and image-size limits
- **WHEN** images are prepared for analysis
- **THEN** submitted images respect those limits and retain their sampled timestamps

#### Scenario: Invalid provider response is rejected

- **GIVEN** the provider returns a non-success status, malformed JSON content, or content that does not satisfy the analysis contract
- **WHEN** the analyzer handles the response
- **THEN** it reports analysis failure rather than manufacturing a successful risk assessment

#### Scenario: No readable frames prevent a provider request

- **GIVEN** a clip yields no readable frames
- **WHEN** analysis is attempted
- **THEN** the attempt fails before sending an image-analysis request

#### Scenario: Schema-valid output has inconsistent temporal metadata

- **GIVEN** provider content satisfies the structured model but contains a negative frame count or string timestamps inconsistent with submitted frames
- **WHEN** the analyzer parses that content
- **THEN** those inconsistencies alone do not reject the structured result or certify its factual correctness
- **AND** timestamp fields remain strings, `frame_count` is neither constrained to be nonnegative nor compared with submitted samples, and entity chronology is not checked against the video

### Requirement: Default alert policy decision precedence

The default alert policy SHALL apply non-null per-camera overrides to its base settings. It SHALL permit notification when `notify_on_motion` is enabled, otherwise when analyzed risk meets or exceeds the configured threshold, otherwise when analyzed activity matches the configured activity list. It SHALL suppress analyzed clips matching none of those conditions and provide a decision reason.

#### Scenario: Camera motion override permits notification without analysis

- **GIVEN** the base policy disables motion notification and the camera's override enables it
- **WHEN** that camera's successfully filtered clip has no analysis result
- **THEN** the decision permits notification with reason `notify_on_motion=true`

#### Scenario: Risk threshold permits notification

- **GIVEN** motion notification is disabled and analyzed risk is at or above the effective threshold
- **WHEN** the default policy evaluates the clip
- **THEN** it permits notification with a reason identifying the analyzed risk

#### Scenario: Activity list permits a low-risk notification

- **GIVEN** motion notification is disabled, analyzed risk is below the effective threshold, and activity matches the effective configured activity list
- **WHEN** the default policy evaluates the clip
- **THEN** it permits notification with a reason identifying the matching activity

#### Scenario: Below-threshold activity is suppressed

- **GIVEN** motion notification is disabled, analyzed risk is below threshold, and activity is not in the configured list
- **WHEN** the default policy evaluates the clip
- **THEN** it does not permit notification and reports `below_threshold`

### Requirement: Disabled alert policy suppresses notification

When the alert policy is disabled, HomeSec SHALL suppress notification with decision reason `alert_policy_disabled`, regardless of the clip's filter or analysis result.

#### Scenario: High-risk clip with alert policy disabled

- **GIVEN** the alert policy is disabled and an analyzed clip has high risk
- **WHEN** its alert decision is evaluated
- **THEN** notification is not permitted and the reason is `alert_policy_disabled`

### Requirement: Failed-analysis fallback and missing-analysis suppression

An exhausted analysis failure SHALL NOT itself abort alert evaluation. After such failure, the default alert policy SHALL first permit notification with reason `notify_on_motion=true` when motion notification is enabled. Otherwise, it SHALL permit notification for a successfully filtered trigger detection using reason `filter_detected_trigger_vlm_failed`. Without analysis, trigger detection, or motion notification, the policy SHALL suppress notification with reason `no_analysis`.

#### Scenario: Failed analysis falls back to a trigger detection

- **GIVEN** filtering detects a configured trigger class, motion notification is disabled, and analysis attempts fail
- **WHEN** the default alert policy evaluates the clip without analysis
- **THEN** it permits notification with reason `filter_detected_trigger_vlm_failed`
- **AND** any resulting notification identifies that analysis failed instead of supplying an invented risk or activity

#### Scenario: Motion notification survives analysis failure

- **GIVEN** motion notification is enabled and analysis fails
- **WHEN** the default alert policy evaluates the clip
- **THEN** it permits notification with reason `notify_on_motion=true`

#### Scenario: No analysis and no trigger suppress notification

- **GIVEN** no analysis is available, no trigger class is detected, and motion notification is disabled
- **WHEN** the default alert policy evaluates the clip
- **THEN** it suppresses notification with reason `no_analysis`

### Requirement: Bounded processing-stage retries

Filtering, analysis, and individual notifier operations within the retry runner SHALL retry raised failures up to `max(1, max_attempts)` attempts, with delays of `max(0, backoff_s) * 2**(attempt - 1)`. Success SHALL supply the successful outcome; exhaustion SHALL follow the stage's failure rules. This bounds attempts, not elapsed time, and adds no generic operation timeout; backend deadlines, shutdown, and cancellation are separate.

#### Scenario: Transient stage failure recovers

- **GIVEN** filtering, analysis, or notification fails once and another attempt is permitted
- **WHEN** the later attempt succeeds
- **THEN** processing uses the successful outcome
- **AND** available lifecycle history identifies the attempts and successful completion

#### Scenario: Persistent stage failure reaches its limit

- **GIVEN** a stage fails on every permitted attempt
- **WHEN** its configured maximum is reached
- **THEN** no further attempt is made in that processing run
- **AND** processing follows the critical filter-failure or noncritical analysis/notification-failure behavior as applicable

#### Scenario: An operation never settles within the retry runner

- **GIVEN** an attempt remains pending and no backend timeout, shutdown, or cancellation settles it
- **WHEN** time passes while the runner awaits the operation
- **THEN** the runner does not start another attempt merely because time has passed and does not impose a generic operation deadline

### Requirement: Notification eligibility and clip completion

HomeSec SHALL attempt notification only when the alert decision permits it and enabled notifiers are available. With successful filtering, completed alert evaluation, and all invoked stages having returned their handled outcomes, processing SHALL attempt to mark the clip completed even when upload, analysis, or notification has failed. Completion SHALL NOT imply that a notification was delivered or every processing stage succeeded.

#### Scenario: Policy suppresses delivery

- **GIVEN** a completed alert decision does not permit notification
- **WHEN** processing reaches notification handling
- **THEN** no notification is attempted
- **AND** processing can complete

#### Scenario: No enabled notifier skips delivery

- **GIVEN** the alert decision permits notification but no enabled notifier is configured
- **WHEN** processing reaches notification handling
- **THEN** no notification is attempted
- **AND** the decision remains available in workflow history when persistence succeeds
- **AND** processing can complete

#### Scenario: Notification failure does not prevent completion

- **GIVEN** filtering and alert evaluation complete but all permitted notification attempts fail
- **WHEN** processing finishes handling notification failure
- **THEN** it reports the failed notification and attempts to mark the clip completed

### Requirement: Best-effort workflow persistence during processing

Workflow-state and lifecycle-history write failures SHALL be reported and contained so in-process upload, filtering, analysis, and permitted notification can proceed. Successful work or writes SHALL NOT imply complete state/history, concurrency-safe snapshots, terminal deletion, or automatic restart replay; see the [concurrent-write limitations](../../../docs/reference.md#analysis-alerts).

#### Scenario: Workflow-state write failure does not stop delivery

- **GIVEN** a clip is accepted, its dependencies can process it, and workflow-state writes fail
- **WHEN** filtering and alert policy permit notification
- **THEN** in-process work and notification can proceed despite those state-write failures
- **AND** durable clip state is not guaranteed to contain those outcomes
