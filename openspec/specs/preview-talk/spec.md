# preview-talk Specification

## Purpose

Describe live camera viewing and push-to-talk behavior, including transport compatibility, authorization, recording priority, session limits, and browser controls.

## Reference

See [reference notes and known limitations](../../../docs/reference.md#preview-talk) for this capability's operator guides and verification limits.

## Requirements

### Requirement: Preview availability and transport are explicit

HomeSec SHALL expose camera preview availability, state, and refusal reasons. Enabled preview SHALL use the configured WebRTC or HLS transport; explicit HLS configuration and recognized legacy HLS settings SHALL remain supported when the default transport is WebRTC.

#### Scenario: Start a WebRTC preview

- **GIVEN** preview is enabled for a supported source using WebRTC
- **WHEN** an authorized client successfully starts preview
- **THEN** the response identifies WebRTC and supplies signaling information rather than an HLS playlist URL

#### Scenario: Use existing HLS configuration

- **GIVEN** preview is configured explicitly for HLS or uses recognized legacy HLS configuration fields without a backend
- **WHEN** HomeSec resolves the preview configuration
- **THEN** it retains HLS behavior rather than interpreting those settings as WebRTC

#### Scenario: Preview cannot be served

- **GIVEN** a preview request names an absent camera or the runtime is unavailable
- **WHEN** HomeSec handles the request
- **THEN** an absent camera returns HTTP 404 and unavailable runtime returns HTTP 503

### Requirement: Preview access is scoped and expires

With authentication enabled, HomeSec SHALL authorize preview playback and signaling with the API key or a valid camera-scoped preview token. WebRTC viewer leases SHALL not outlive the authorization used to negotiate or renew them. HLS playback SHALL refuse stale media when the corresponding preview is disabled or inactive.

#### Scenario: Attempt access to another camera

- **GIVEN** a preview token issued for one camera
- **WHEN** a client uses it for another camera's signaling
- **THEN** HomeSec rejects the request

#### Scenario: Negotiate near token expiry

- **GIVEN** a valid preview token has less remaining lifetime than a requested viewer lease
- **WHEN** a WebRTC viewer is negotiated or renewed
- **THEN** its lease is bounded by the token's remaining lifetime

#### Scenario: Stale HLS files remain after preview stops

- **GIVEN** HLS files exist but runtime status reports preview disabled or inactive
- **WHEN** a client requests playback
- **THEN** HomeSec refuses to serve those files as a current preview

### Requirement: Preview yields according to recording policy

With the default stop-on-recording policy, HomeSec SHALL stop active preview and refuse new preview while recording is active. Allow-during-recording mode SHALL permit concurrent preview on a best-effort basis and expose recording-priority, camera-session-budget, or temporary-unavailability refusals when concurrency cannot be provided.

#### Scenario: Recording starts under the default policy

- **GIVEN** preview is active with stop-on-recording configured
- **WHEN** the camera begins recording
- **THEN** HomeSec stops preview and refuses new preview with the recording-priority reason while recording remains active

#### Scenario: Concurrent preview is available

- **GIVEN** allow-during-recording is configured and the camera can support concurrent preview
- **WHEN** recording is active and a client starts preview
- **THEN** recording does not by itself cause the preview request to be refused

#### Scenario: Camera session capacity prevents preview

- **GIVEN** the runtime reports that the camera session budget prevents preview
- **WHEN** a client requests preview
- **THEN** the API returns HTTP 409 with the session-budget-exhausted reason

### Requirement: Preview resources have bounded lifetimes

HomeSec SHALL end idle preview after the configured idle timeout. WebRTC viewers SHALL be subject to configured viewer and session-duration limits, negotiation timeout, and renewable authorization leases. Stopping a viewer SHALL release that viewer; expired authorization SHALL not be revived by status polling.

#### Scenario: Preview loses all viewers

- **GIVEN** no active or pending authorized viewer remains
- **WHEN** the configured idle timeout expires
- **THEN** HomeSec stops the unused preview input

#### Scenario: A WebRTC lease expires while status is polled

- **GIVEN** a WebRTC viewer's lease has expired
- **WHEN** a client continues polling preview status without valid renewal
- **THEN** the polling does not keep that viewer's media session alive

#### Scenario: Close one of several WebRTC viewers

- **GIVEN** multiple viewers are attached to a camera
- **WHEN** one client closes its WebRTC viewer session
- **THEN** that session closes without requesting a camera-wide stop for the remaining viewers

### Requirement: Browser preview handles visibility and changing authorization

The browser SHALL refresh preview authorization while viewing and release its WebRTC peer when stopped, unmounted, or backgrounded. A viewer with continuing viewing intent SHALL reconnect with current authorization when foregrounded. Stale responses from a previous camera or session SHALL not replace the current viewer state.

#### Scenario: Background and return to a WebRTC preview

- **GIVEN** the user is viewing a WebRTC preview
- **WHEN** the page becomes hidden and later visible while viewing intent remains active
- **THEN** the browser closes the old peer and establishes a fresh authorized peer on return

#### Scenario: Switch cameras while cleanup is pending

- **GIVEN** a viewer is attached to one camera
- **WHEN** the user switches cameras
- **THEN** cleanup uses the former camera's session authorization and stale responses do not overwrite the new camera's state

### Requirement: Talk capability and backend selection reflect camera support

HomeSec SHALL expose talk policy, capability, selected backend, and stable refusal reasons. Automatic selection SHALL prefer a supported ONVIF backchannel before eligible Tapo local fallback. Explicit backend selection SHALL use that backend. Unsupported cameras or codecs SHALL be reported rather than treated as successfully connected.

#### Scenario: ONVIF is supported in automatic mode

- **GIVEN** automatic talk selection and a supported ONVIF audio backchannel
- **WHEN** HomeSec selects a talk backend
- **THEN** it selects ONVIF without initiating passive Tapo discovery

#### Scenario: Manual RTSP setup needs local Tapo discovery

- **GIVEN** automatic talk selection has not found a supported backend and no explicit Tapo configuration or existing discovery hint applies
- **WHEN** HomeSec performs passive local Tapo endpoint discovery
- **THEN** that discovery sends no credentials, setup request, or audio and only a recognized endpoint hint enables subsequent Tapo selection

#### Scenario: Explicit Tapo configuration fails

- **GIVEN** the operator explicitly selected the Tapo local backend
- **WHEN** that backend cannot establish supported talk
- **THEN** HomeSec reports the backend's refusal without silently switching to ONVIF

### Requirement: Talk preparation defines the authorized audio stream

HomeSec SHALL prepare a talk session before streaming, returning a session identity, WebSocket endpoint, and PCM input contract. The default input SHALL be mono signed 16-bit little-endian PCM at 16000 Hz in 20 ms frames. With authentication enabled, preparation SHALL require the API key and streaming SHALL require valid stream authorization.

#### Scenario: Prepare a supported talk session

- **GIVEN** an enabled, supported camera with no reserved talk session
- **WHEN** an authorized client prepares talk without overriding the configured input
- **THEN** the response supplies the configured input contract, session identity, and WebSocket endpoint

#### Scenario: Send an invalid audio frame

- **GIVEN** a prepared and opened talk stream
- **WHEN** the client sends a binary frame whose size does not match the agreed PCM contract
- **THEN** HomeSec rejects the frame and ends the invalid stream

#### Scenario: Use expired stream authorization

- **GIVEN** server authentication is enabled
- **WHEN** a client attempts to open a talk WebSocket with an expired stream token
- **THEN** HomeSec refuses the connection

### Requirement: Talk sessions are exclusive and bounded per camera

HomeSec SHALL allow at most one reserved or active talk session per camera. Reservation and active phases SHALL each have a separate `max_session_s` limit, restarted at activation. Active sessions SHALL also have an `idle_timeout_s` audio-idle limit. After WebSocket acceptance and authorization, the start-message wait SHALL use `idle_timeout_s`. Stop, disconnect, failed opening, cancellation, and runtime shutdown SHALL release the affected session and resources.

#### Scenario: Reserve talk while another session exists

- **GIVEN** a camera already has a reserved or active talk session
- **WHEN** another client prepares talk for that camera
- **THEN** HomeSec refuses it with the session-already-active reason

#### Scenario: A reservation or stream times out

- **GIVEN** a reservation reaches `max_session_s` since preparation or an active session reaches `max_session_s` since activation or its audio-idle limit
- **WHEN** the corresponding timeout is detected
- **THEN** HomeSec releases the reservation and closes any active camera talk session

#### Scenario: Activation starts a new duration limit

- **GIVEN** a reserved session becomes active before its reservation expires
- **WHEN** the original reservation deadline passes and no active-session limit or other stop condition has been reached
- **THEN** the active session continues under its own `max_session_s` limit measured from activation

#### Scenario: Disconnect before streaming starts

- **GIVEN** a reserved talk session with an accepted and authorized WebSocket awaiting the start message
- **WHEN** the browser disconnects or the `idle_timeout_s` wait for that message expires
- **THEN** HomeSec releases the reserved session

#### Scenario: Cancel opening or shut down an active session

- **GIVEN** talk is opening or active
- **WHEN** opening is cancelled, opening fails, an explicit stop arrives, or the runtime shuts down
- **THEN** HomeSec cleans up the affected session so it cannot continue transmitting after cleanup

### Requirement: Browser talk respects hold intent and microphone availability

The browser SHALL start talk from an explicit hold action, stop it on release, and keep preview audio muted while talk is active. Denied microphone permission SHALL prevent session preparation. Releasing before preparation or stream readiness SHALL cancel the pending start and clean up any prepared session.

#### Scenario: Hold and release talk

- **GIVEN** a supported camera and microphone permission
- **WHEN** the user holds the talk control and the stream becomes ready
- **THEN** microphone audio is transmitted using the prepared format and preview audio remains muted during talk
- **WHEN** the user releases the control
- **THEN** the browser stops microphone capture and the talk session

#### Scenario: Microphone permission is denied

- **GIVEN** the browser denies microphone access
- **WHEN** the user attempts to start talk
- **THEN** the UI reports the blocked microphone and does not prepare a server talk session

#### Scenario: Release before startup completes

- **GIVEN** microphone acquisition or talk preparation is pending
- **WHEN** the user releases the hold control before streaming is ready
- **THEN** the browser cancels the start and stops any session already prepared for that attempt

### Requirement: Browser talk stops on excessive audio backpressure

The browser SHALL stop microphone capture and close the talk session when buffered audio exceeds its supported backpressure threshold, rather than accumulating an unbounded audio backlog.

#### Scenario: Audio transmission cannot keep up

- **GIVEN** an active browser talk stream
- **WHEN** buffered audio exceeds the supported threshold
- **THEN** the browser closes the stream, stops the session, and exposes the failure to the user
