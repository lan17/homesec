# Spec Delta

## MODIFIED Requirements

### Requirement: Setup checks report readiness without saving configuration

HomeSec SHALL expose preflight and connection checks without saving configuration. Responses SHALL identify check success or failure; aggregate preflight SHALL pass only when all reported checks pass. Connection tests SHALL use registered probes or the implemented category fallback. The media check SHALL assess bundled inspection readiness independently of CLI availability. A successful check does not certify end-to-end processing or future connectivity.

#### Scenario: A preflight check times out

- **GIVEN** a preflight dependency does not respond within its check timeout
- **WHEN** the preflight result is returned
- **THEN** the affected check reports failure and the aggregate result is not passed
- **AND** the response reports the selected Postgres, ffmpeg, disk-space, network, configuration/environment, and pg_dump checks

#### Scenario: A proposed connection is tested

- **GIVEN** a supported camera, storage, notifier, or analyzer connection-test request
- **WHEN** the request is processed
- **THEN** HomeSec reports the connection-test result without saving that request as the application configuration
- **AND** testing uses the registered backend-specific probe when available or the implemented fallback for that category otherwise

#### Scenario: Generic connection test only establishes its ping result

- **GIVEN** a supported storage, notifier, or analyzer backend with no registered setup probe, valid configuration, successful construction, and a successful ping
- **WHEN** its generic connection test completes
- **THEN** HomeSec reports probe success and attempts plugin shutdown without saving the proposed configuration
- **AND** that result does not establish that an actual clip was uploaded, analyzed, or delivered

#### Scenario: Native inspection is ready without CLI tools

- **GIVEN** a compatible bundled inspection implementation with the required baseline features and absent or incompatible host media CLI tools
- **WHEN** setup checks media readiness
- **THEN** the existing `ffmpeg` check reports native inspection readiness based on a successful bounded capability exchange
- **AND** its message identifies that scope rather than claiming an executable was found or all media execution dependencies were removed

#### Scenario: An executable file alone does not establish readiness

- **GIVEN** the native inspector is missing, cannot start, has an incompatible control contract, or lacks a required baseline capability
- **WHEN** setup checks media readiness
- **THEN** the media check fails with a bounded actionable local-dependency reason and the aggregate result does not pass
- **AND** finding a host `ffmpeg` or `ffprobe` executable does not turn that failure into success

#### Scenario: Inspection readiness does not certify compatibility execution

- **GIVEN** native inspection is available but a configured recording or preview path retains a separate execution dependency during migration
- **WHEN** setup returns its media check
- **THEN** the result describes inspection readiness and does not certify that separate execution path
- **AND** remaining execution prerequisites are documented explicitly until their migration is complete

## ADDED Requirements

### Requirement: RTSP connection testing uses bounded native preflight

Registered RTSP Test Connection SHALL exercise the same CLI-independent inspection and profile-validation behavior as camera startup, within its advertised timeout and cleanup budget. It SHALL preserve existing success and failure detail fields without saving or activating configuration. Timeout and cancellation SHALL stop owned camera work and temporary writers rather than merely stop waiting for them.

#### Scenario: Test a supported RTSP configuration without CLI tools

- **GIVEN** an admitted valid RTSP connection-test request, a supported reachable camera, and no media CLI tools
- **WHEN** its native preflight succeeds
- **THEN** the result reports success with `session_mode`, `selected_recording_profile`, and `probed_streams` details
- **AND** no configuration is saved, runtime activated, production clip handed off, or media CLI launched by the test

#### Scenario: Preserve failure detail and distinguish local capability failures

- **GIVEN** RTSP preflight fails during discovery, selection, validation, or local capability inspection
- **WHEN** Test Connection returns the failure
- **THEN** existing `stage` and `camera_key` details remain available for preflight failures
- **AND** the bounded reason distinguishes local missing or unsupported capabilities from camera connectivity without exposing credentials or tokens

#### Scenario: Timeout or cancellation stops camera work

- **GIVEN** RTSP Test Connection is waiting for a stalled native operation
- **WHEN** the request times out or is cancelled
- **THEN** owned work is stopped and locally closed or terminated under the overall cleanup bound before cleanup is reported complete
- **AND** trial writers stop before their temporary files are removed, with no background thread continuing to open camera inputs after cleanup
