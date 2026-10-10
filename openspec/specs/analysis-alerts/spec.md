# analysis-alerts Specification

## Purpose

Describe how completed clips are filtered and analyzed, how alert eligibility is decided, and how processing handles skipped stages and partial failures.

## Reference

See [reference notes and known limitations](../../../docs/reference.md#analysis-alerts) for this capability's operator guides and verification limits.

## Requirements

### Requirement: Clip processing stage ordering

HomeSec SHALL start upload and filtering independently for an accepted clip. Eligible analysis SHALL begin after successful filtering without waiting for upload to finish. Alert evaluation and any notification SHALL use the eventual upload outcome. A filter failure SHALL end analysis and notification processing after the upload outcome is collected.

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
- **AND** available workflow state records the filter failure rather than successful processing completion

### Requirement: Supported object-filter results

The built-in object filter SHALL apply its configured supported target classes, confidence threshold, and minimum box-height ratio to sampled frames. Its result SHALL expose detected classes, confidence, model identity, and sampled-frame count. It SHALL report no detected classes and zero confidence when no accepted boxes are found.

#### Scenario: Accepted target detection is returned

- **GIVEN** a sampled frame contains a configured supported target detection that passes confidence and box-height constraints
- **WHEN** the object filter evaluates the clip
- **THEN** the result contains the detected class and its accepted confidence with model and sampled-frame information

#### Scenario: Small detections are excluded

- **GIVEN** detections have boxes shorter than the configured minimum fraction of frame height
- **WHEN** the object filter evaluates the clip and finds no other accepted boxes
- **THEN** it returns no detected classes and zero confidence

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

The OpenAI-compatible analyzer SHALL analyze timestamped sampled images within configured preprocessing limits and accept only responses that satisfy its structured sequence-analysis contract. Valid responses SHALL provide risk, activity, summary, and detailed analysis to downstream processing. Provider errors, malformed responses, and clips yielding no readable frames SHALL be treated as analysis failures.

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

- **GIVEN** analyzed risk is below the effective threshold but activity matches the effective configured activity list
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

An exhausted analysis failure SHALL NOT itself abort alert evaluation. After such failure, the default alert policy SHALL permit notification for a successfully filtered trigger detection using reason `filter_detected_trigger_vlm_failed`; `notify_on_motion` SHALL also permit notification. Without analysis, trigger detection, or motion notification, the policy SHALL suppress notification with reason `no_analysis`.

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

Failed filtering, analysis, and individual notification attempts SHALL use the configured maximum attempt count and exponential backoff. A successful retry SHALL supply that stage's successful outcome. Exhaustion SHALL follow the stage's failure behavior rather than claiming success or retrying indefinitely within that processing run.

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

### Requirement: Notification eligibility and clip completion

HomeSec SHALL attempt notification only when the alert decision permits it and enabled notifiers are available. With successful filtering and completed alert evaluation, processing SHALL attempt to mark the clip completed even when upload, analysis, or notification has failed. Completion SHALL NOT imply that a notification was delivered or every processing stage succeeded.

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

Failures while saving workflow state or appending lifecycle history SHALL be reported and contained so the accepted clip's in-process upload, filtering, analysis, and permitted notification can proceed. HomeSec SHALL NOT equate successful in-process work with guaranteed durable state, complete event history, or automatic replay after a process restart.

#### Scenario: Workflow-state write failure does not stop delivery

- **GIVEN** a clip is accepted, its dependencies can process it, and workflow-state writes fail
- **WHEN** filtering and alert policy permit notification
- **THEN** in-process work and notification can proceed despite those state-write failures
- **AND** durable clip state is not guaranteed to contain those outcomes
