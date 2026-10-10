## Purpose

Describe how operators discover setup readiness, validate and save configuration, manage cameras, and activate saved camera settings in the running service.

## ADDED Requirements

### Requirement: Setup readiness reflects configuration and runtime state
HomeSec SHALL report setup readiness as fresh, partial, or complete, together with camera presence, pipeline activity, and authentication configuration.

#### Scenario: First-run setup is needed
- **GIVEN** no active configuration, configured cameras, or running pipeline
- **WHEN** a client requests setup status
- **THEN** the response reports a fresh setup state and a stopped pipeline

#### Scenario: Configuration exists before the pipeline runs
- **GIVEN** a configuration with cameras and no running pipeline
- **WHEN** a client requests setup status
- **THEN** the response reports a partial setup state

#### Scenario: Configured cameras are running
- **GIVEN** configured cameras and a running pipeline
- **WHEN** a client requests setup status
- **THEN** the response reports a complete setup state

### Requirement: Setup checks report readiness without saving configuration
HomeSec SHALL expose preflight checks and connection tests independently of configuration persistence. Check responses SHALL identify success or failure; preflight responses SHALL include individual results and an aggregate result.

#### Scenario: A preflight check times out
- **GIVEN** a preflight dependency does not respond within its check timeout
- **WHEN** the preflight result is returned
- **THEN** the affected check reports failure and the aggregate result is not passed

#### Scenario: A proposed connection is tested
- **GIVEN** a supported camera, storage, notifier, or analyzer connection-test request
- **WHEN** the request is processed
- **THEN** HomeSec reports the connection-test result without saving that request as the application configuration

### Requirement: Setup finalization validates before persistence and activation
HomeSec SHALL restrict setup finalization to bootstrap mode, validate the assembled configuration, and support validation-only requests. Successful applied finalization SHALL save the configuration and activate the runtime without requesting an application restart.

#### Scenario: Validate a valid setup without applying it
- **GIVEN** bootstrap mode and a valid setup request with validation-only enabled
- **WHEN** setup finalization is requested
- **THEN** the response reports success without writing the configuration or activating the runtime

#### Scenario: Reject an invalid setup
- **GIVEN** bootstrap mode and a setup request that fails semantic configuration validation
- **WHEN** setup finalization is requested
- **THEN** HomeSec returns a validation error and does not save that invalid configuration

#### Scenario: Apply a valid setup
- **GIVEN** bootstrap mode and a valid setup request with at least one camera
- **WHEN** applied setup finalization succeeds
- **THEN** the saved configuration is activated and the response reports that no application restart was requested

#### Scenario: Reject finalization outside bootstrap
- **GIVEN** HomeSec is already operating outside bootstrap mode
- **WHEN** a client requests setup finalization
- **THEN** HomeSec rejects the request

### Requirement: ONVIF setup discovers cameras and reports probe outcomes
HomeSec SHALL expose ONVIF discovery results and credentialed probes for device information, media profiles, and available stream URIs. An empty discovery result SHALL be successful; discovery or probe failures SHALL be distinguishable from a probe timeout.

#### Scenario: Discover available cameras
- **GIVEN** ONVIF cameras respond to a discovery request
- **WHEN** discovery completes
- **THEN** HomeSec returns their network addresses and discovery metadata without creating configured cameras

#### Scenario: No cameras respond
- **GIVEN** discovery completes without finding an ONVIF camera
- **WHEN** HomeSec returns its discovery response
- **THEN** the response is a successful empty list

#### Scenario: Probe a reachable camera
- **GIVEN** valid connection details for a responsive ONVIF camera
- **WHEN** the client probes it
- **THEN** HomeSec returns device information and media profiles with their available stream URI or stream error

#### Scenario: Discovery or probing fails
- **GIVEN** the discovery operation or camera probe fails
- **WHEN** HomeSec returns the operation result
- **THEN** the response is HTTP 502 with the corresponding discovery-failed or probe-failed reason

#### Scenario: A camera probe exceeds its timeout
- **GIVEN** a camera probe does not complete within the requested timeout
- **WHEN** HomeSec ends the probe
- **THEN** the response is HTTP 504 with the probe-timeout reason

### Requirement: Camera configuration mutations validate and preserve omitted fields
HomeSec SHALL support creating, updating, and removing named cameras. A patch within the same source backend SHALL preserve omitted source settings. Invalid mutations SHALL fail without saving invalid configuration; duplicate creation and mutation of a missing camera SHALL return conflict and not-found responses respectively.

#### Scenario: Update one source setting
- **GIVEN** an existing camera with multiple source settings
- **WHEN** a valid patch supplies one setting without changing the source backend
- **THEN** HomeSec saves the supplied setting and preserves the omitted settings

#### Scenario: Switch source backends
- **GIVEN** an existing camera configured for one source backend
- **WHEN** a valid patch selects a different backend and supplies its settings
- **THEN** HomeSec validates and saves the new source configuration without carrying over the former backend's settings

#### Scenario: Reject an invalid camera mutation
- **GIVEN** a saved camera configuration
- **WHEN** a mutation fails configuration validation
- **THEN** HomeSec returns an error without saving the invalid mutation or requesting runtime reload

#### Scenario: Reject duplicate or missing camera names
- **GIVEN** a camera name is already configured, or a mutation targets an absent camera
- **WHEN** a client creates the duplicate name, or updates or deletes the absent camera
- **THEN** duplicate creation returns HTTP 409 and a missing-camera mutation returns HTTP 404

### Requirement: Configuration reads protect sensitive values
HomeSec SHALL redact sensitive values from configuration and camera API responses while preserving environment-variable references. Camera source patches SHALL reject redacted placeholders as replacement values.

#### Scenario: Read a configuration containing credentials
- **GIVEN** configuration contains sensitive values, credential-bearing URLs, and environment-variable references
- **WHEN** a client reads the configuration or camera source settings
- **THEN** the response hides sensitive values and URL credentials while retaining environment-variable names

#### Scenario: Submit a displayed redaction placeholder
- **GIVEN** a camera source field was displayed as a redacted placeholder
- **WHEN** a patch submits that placeholder as a source setting
- **THEN** HomeSec rejects the patch rather than replacing the saved value with the placeholder

### Requirement: Saving camera configuration is distinct from activating it
HomeSec SHALL save camera mutations without reloading by default and indicate that activation is required. A client SHALL be able to request activation with a mutation or through the runtime reload endpoint; accepted reload responses SHALL identify the target generation rather than claim completed activation.

#### Scenario: Save camera settings without activation
- **GIVEN** a valid camera mutation with no request to apply changes
- **WHEN** HomeSec saves the mutation
- **THEN** the response indicates restart-required configuration and no runtime reload is requested by that mutation

#### Scenario: Request reload of saved configuration
- **GIVEN** saved configuration that can be prepared for reload and no reload already in progress
- **WHEN** a client requests runtime reload
- **THEN** HomeSec returns HTTP 202 with acceptance and the target generation

#### Scenario: Reject a concurrent reload
- **GIVEN** a runtime reload is already in progress
- **WHEN** another reload is requested
- **THEN** HomeSec returns HTTP 409 with the reload-in-progress reason

### Requirement: Runtime activation reports completion and preserves the prior runtime on candidate failure
HomeSec SHALL expose runtime state, generation, active configuration version, reload activity, and the last reload error. Successful reload SHALL advance the active generation. If a candidate fails to start or activate while the prior runtime remains active, HomeSec SHALL retain the prior generation and report the failure.

#### Scenario: Complete a successful reload
- **GIVEN** an active runtime and an accepted reload
- **WHEN** the new runtime starts and activates successfully
- **THEN** status reports the new active generation, clears the reload error, and no longer reports a reload in progress

#### Scenario: Candidate startup or activation fails
- **GIVEN** an active runtime and an accepted reload
- **WHEN** the candidate fails before successful activation
- **THEN** the failed candidate is cleaned up, the prior generation remains active, and status exposes the reload error
