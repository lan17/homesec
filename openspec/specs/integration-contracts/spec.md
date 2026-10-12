# integration-contracts Specification

## Purpose

Describe HomeSec's existing contracts for API clients, notification recipients, and plugin authors so integrations can depend on observable behavior and the existing authoritative schemas.

## Reference

See [reference notes and known limitations](../../../docs/reference.md#integration-contracts) for this capability's operator guides and verification limits.

## Shared API preconditions

Unless an HTTP scenario explicitly tests a failed prerequisite, it assumes an initialized application, a request that can be parsed and validated, and admission by that route's configured authentication, mode, and availability gates. An explicit failure replaces only that prerequisite; earlier gates still apply. Endpoint-specific outcomes describe requests that reach that endpoint's relevant branch, not a promise that they take precedence over every malformed request or dependency failure. Browser and direct plugin scenarios are not endpoint admission tests.

The [route registration](../../../src/homesec/api/routes/__init__.py) and [dependencies](../../../src/homesec/api/dependencies.py) define these control paths before the endpoint body:

| Route family | Registered admission gates, in order |
| --- | --- |
| Public `/health` and `/api/v1/health` | No API-key or normal-mode gate; bootstrap has its own summary |
| Setup status/checks and ONVIF | API key when authentication is enabled; setup finalization additionally requires bootstrap mode |
| Configuration, cameras, runtime, maintenance, preview/talk HTTP controls, diagnostics | API key when enabled, then normal mode |
| Clip list/detail/token/deletion and statistics | API key when enabled, then normal mode, then database availability |
| Clip media | Normal mode, database availability, then media authorization |
| Preview playback/signaling | Normal mode, then preview authorization; camera/runtime/transport checks follow |

These are dependency gates, not a total ordering of every body-decoding/schema-validation failure. API-key failure normally produces 401; a missing effective configured key produces 500. Normal-mode rejection produces 503 `SETUP_REQUIRED`; bootstrap-only finalization outside bootstrap produces 409. A failed database guard produces 503 `DB_UNAVAILABLE`; a passed probe, which can be cached, does not guarantee subsequent queries or hydration succeed.

Media and preview authorization skip token validation when authentication is disabled. With authentication enabled, a matching bearer API key is accepted before examining the query token; otherwise the scoped token is validated. This assumes credential comparison itself completes; malformed non-ASCII comparisons are a separate limitation below. Permitted authorization is not proof of media/camera availability.

Talk streaming is a WebSocket lifecycle, not an HTTP error-envelope route. The [talk handler](../../../src/homesec/api/routes/talk.py) checks application/bootstrap availability, accepts the WebSocket, then authorizes it: disabled authentication or a matching bearer API key bypasses query-token validation; otherwise a valid camera-and-session-scoped stream token is required. Rejected token authorization closes the accepted socket before camera-stream opening. It does not use the database guard.

See the [scenario evidence register](../../../docs/behavior-evidence.md) for the verification level of each scenario.

## Requirements

### Requirement: API key authentication

API routes protected by API-key authentication SHALL accept the configured ASCII key in an `Authorization: Bearer <key>` header when authentication is enabled. Missing bearer credentials or an incorrect ASCII-comparable key SHALL return HTTP 401 with `UNAUTHORIZED`. Non-ASCII strings passed to the current constant-time string comparison can raise instead of returning a mismatch, producing a generic pre-response HTTP 500; see the [malformed-credential limitation](../../../docs/reference.md#integration-contracts). A missing configured key SHALL return HTTP 500 with `API_KEY_NOT_CONFIGURED`. Disabling authentication SHALL remove the API-key requirement. Public health summaries SHALL remain accessible without credentials.

#### Scenario: Protected request lacks valid credentials

- **GIVEN** API authentication is enabled and an ASCII key is configured
- **WHEN** a client requests camera configuration with no bearer header or an incorrect ASCII bearer key
- **THEN** the request returns HTTP 401 with `error_code` equal to `UNAUTHORIZED`

#### Scenario: Valid bearer credentials

- **GIVEN** API authentication is enabled and an ASCII key is configured
- **WHEN** a client supplies that key to an API-key-protected route
- **THEN** authentication permits the request to proceed to the route's other checks

#### Scenario: Enabled authentication has no effective key

- **GIVEN** API authentication is enabled but its configured environment variable has no key
- **WHEN** a client calls an API-key-protected route
- **THEN** the response is HTTP 500 with `error_code` equal to `API_KEY_NOT_CONFIGURED`

#### Scenario: Public health with authentication enabled

- **GIVEN** API authentication is enabled
- **WHEN** an operator calls `/health` or `/api/v1/health` without credentials
- **THEN** API-key authentication does not reject the probe

#### Scenario: Authentication disabled

- **GIVEN** API authentication is disabled
- **WHEN** a client calls an API-key-protected route without a bearer header
- **THEN** the request proceeds to the route's other checks

#### Scenario: Non-ASCII bearer comparison fails before route execution

- **GIVEN** authentication is enabled with an ASCII key and the parsed bearer value contains a non-ASCII character
- **WHEN** an API-key-protected HTTP request reaches the string comparison
- **THEN** the current comparison raises and the generic pre-response exception handler returns 500 rather than the ordinary 401 mismatch response
- **AND** the protected route body is not admitted; this is an error-handling limitation, not granted access

### Requirement: Browser session authentication

The browser UI SHALL retain a supplied API key in browser session storage and use it for API requests that rely on stored authentication. Clearing the stored key SHALL remove it from subsequent requests using that stored authentication. An explicitly supplied request key SHALL take precedence over the stored value.

#### Scenario: Reuse the supplied key within the browser session

- **GIVEN** the user has supplied an API key and it is stored for the browser session
- **WHEN** the browser makes an API request without an explicit request-key override
- **THEN** the request includes that stored key as a bearer credential

#### Scenario: Clear stored authentication

- **GIVEN** an API key is stored for the browser session
- **WHEN** the user clears the stored authentication and a later request relies on the stored key
- **THEN** that request does not include the cleared key

#### Scenario: Override the stored key for one request

- **GIVEN** an API key is stored for the browser session
- **WHEN** a request explicitly supplies a different key or no key
- **THEN** that explicit choice takes precedence over the stored key

### Requirement: Legacy browser route compatibility

The browser UI SHALL redirect `/` and `/home` to Live, `/dashboard` to System, `/cameras` to Settings camera setup, and legacy clip routes to Events. Dashboard, cameras, and clip redirects SHALL preserve query parameters; clip-detail redirects SHALL preserve the encoded clip identity.

#### Scenario: Open a legacy event search or detail link

- **GIVEN** a `/clips` or `/clips/{clipId}` link contains search context
- **WHEN** the browser opens the link
- **THEN** it redirects to the corresponding Events route with the query parameters preserved
- **AND** a detail redirect preserves the clip identity in encoded form

#### Scenario: Open former navigation destinations

- **GIVEN** a user opens `/dashboard`, `/cameras`, `/`, or `/home`
- **WHEN** the browser resolves the route
- **THEN** dashboard opens System, cameras opens Settings camera setup, and root or home opens Live
- **AND** dashboard and cameras preserve their query parameters

### Requirement: Canonical API exception responses

Handled API exceptions and request-validation failures with JSON-serializable error data SHALL return a JSON envelope containing textual `detail` and `error_code`, with applicable additional fields preserved. Request-schema failures SHALL return HTTP 422 with `REQUEST_VALIDATION_FAILED` and validation details when that error payload can be serialized. The validation handler inserts `exc.errors()` without converting exception objects in `ctx.error`; such payloads can fail serialization and become a generic HTTP 500 instead. Before a response has started, unexpected exceptions reaching the generic handler SHALL return HTTP 500 with generic detail and `INTERNAL_SERVER_ERROR` rather than expose the exception text. This is not a promise to replace an already-started streaming response or a WebSocket close with a new HTTP envelope.

#### Scenario: Route reports a conflict with additional context

- **GIVEN** a route rejects an operation with a specific conflict code and additional context
- **WHEN** the API returns that exception response
- **THEN** the HTTP conflict status, `detail`, `error_code`, and additional context are preserved

#### Scenario: Invalid request parameters

- **GIVEN** an API request reaches request-schema validation and its invalid parameters produce JSON-serializable validation errors
- **WHEN** the request is processed
- **THEN** it returns HTTP 422 with `REQUEST_VALIDATION_FAILED` and `validation_errors`

#### Scenario: Unexpected route exception

- **GIVEN** a route encounters an unexpected exception before its response has started
- **WHEN** the API produces its response
- **THEN** it returns HTTP 500 with `detail` equal to `Internal server error` and `error_code` equal to `INTERNAL_SERVER_ERROR`

#### Scenario: A model-validator exception cannot be serialized

- **GIVEN** an admitted talk-session preparation request contains input with `sample_rate: 11025` and `frame_ms: 20`, causing the integral-frame validator to raise `ValueError`
- **WHEN** the request-validation handler attempts to JSON-encode errors containing that exception in `ctx.error`
- **THEN** error-response construction fails and the generic handler returns 500 `INTERNAL_SERVER_ERROR`, rather than the intended 422 validation envelope
- **AND** the endpoint body is not executed; this characterizes a current serializer limitation

### Requirement: Generated API contract

HomeSec SHALL expose an OpenAPI description derived from its registered API routes and request/response models, and SHALL support exporting that description without starting cameras or the application runtime. Exact request and response shapes SHALL be obtained from the existing generated OpenAPI contract instead of a separate hand-maintained field inventory.

#### Scenario: Generate the contract without application startup

- **GIVEN** HomeSec is installed without a running camera pipeline
- **WHEN** a client developer exports the OpenAPI contract to a selected output path
- **THEN** the file contains valid JSON describing versioned API routes and their model schemas
- **AND** contract generation does not start the camera pipeline

### Requirement: Notification payload meaning

Notifications SHALL use the existing alert payload schema, identifying the clip and camera, decision reason, timestamp, and a deduplication key equal to the clip identifier. Storage/view links and analysis fields SHALL be nullable when unavailable. The payload SHALL indicate upload and analysis failures independently so recipients can distinguish missing enrichment from available clip identity. The key SHALL NOT imply exactly-once delivery or camera-scoped uniqueness. Reusing a clip ID reuses the deduplication key, including across cameras; see the [clip identity limitations](../../../docs/reference.md#capture-recording). The pipeline SHALL construct `ts` with `datetime.now()` when it builds the alert, as a naive process-local notification time, not the clip capture time or the database record's creation time. That same alert object is supplied to per-entry retries.

#### Scenario: Notification includes available analysis and media references

- **GIVEN** a notification is selected for a clip with analysis and an uploaded media reference
- **WHEN** the payload is delivered to a notifier
- **THEN** it contains the clip/camera identity, available analysis and references, decision reason, and a deduplication key matching the clip identifier

#### Scenario: Notification without successful upload or analysis

- **GIVEN** a notification is selected despite an upload or analysis failure
- **WHEN** the payload is delivered
- **THEN** the corresponding failure flag is true and unavailable links or analysis values are null
- **AND** the clip identity and decision reason remain present

#### Scenario: Delayed processing does not preserve capture time as notification time

- **GIVEN** a clip is processed substantially after its estimated source times
- **WHEN** the pipeline builds its notification payload
- **THEN** `ts` reflects alert construction time, while clip identity continues to identify the recorded event

### Requirement: MQTT alert publication

The MQTT integration SHALL publish the alert payload as JSON to the configured topic template expanded with the camera name, using the configured QoS and retain settings. A send that cannot establish a connection within its configured connection wait SHALL report failure rather than report a successful publication.

#### Scenario: Publish to a camera-specific topic

- **GIVEN** a connected MQTT integration with a camera-name topic template and configured QoS/retain values
- **WHEN** it sends an alert
- **THEN** the published topic contains the alert's camera name
- **AND** the message is the alert's JSON payload with the configured QoS and retain settings

#### Scenario: Broker connection does not become available

- **GIVEN** the MQTT integration has no connection
- **WHEN** its configured connection wait expires during a send
- **THEN** the send reports a connection failure

### Requirement: Email alert delivery

The SendGrid email integration SHALL render configured subject and text/HTML templates from the alert context and send to configured recipients and sender, including optional CC/BCC and sender name. Structured analysis inserted through `analysis_html` SHALL be HTML-escaped. Sending without an effective API key, after shutdown, or receiving an HTTP error from SendGrid SHALL report failure.

#### Scenario: Send a templated alert email

- **GIVEN** email credentials, recipients, sender, and content templates are configured
- **WHEN** an alert is sent
- **THEN** the SendGrid request contains the rendered subject and configured text/HTML content, sender, and recipient lists

#### Scenario: Structured analysis contains HTML characters

- **GIVEN** structured analysis contains HTML markup and the email template uses `analysis_html`
- **WHEN** the HTML body is rendered
- **THEN** the analysis markup is escaped rather than inserted as executable markup

#### Scenario: SendGrid rejects a request

- **GIVEN** a configured email integration
- **WHEN** SendGrid returns an HTTP status of 400 or higher
- **THEN** sending reports failure

#### Scenario: Credentials absent or integration shut down

- **GIVEN** the email integration has no effective API key or has completed shutdown
- **WHEN** it is asked to send an alert
- **THEN** sending reports failure without submitting a SendGrid request

### Requirement: Notification fan-out isolation

HomeSec SHALL send a selected alert to each enabled configured notifier. Disabled notifiers SHALL not receive alerts. A delivery failure from one notifier SHALL not prevent delivery attempts to the others, and the failed delivery SHALL remain observable as a failure. When no notifier is enabled, notification delivery SHALL have no transport side effects. Fan-out uses concurrent per-entry attempts and waits for all results; one indefinitely pending notifier can delay aggregate completion even when another succeeds. Failure isolation does not impose a shared deadline.

#### Scenario: One notifier fails while another succeeds

- **GIVEN** two enabled notifiers and one will fail during send
- **WHEN** HomeSec delivers an alert
- **THEN** both notifiers receive a delivery attempt
- **AND** the successful notifier receives the alert while the failed delivery is reported

#### Scenario: All configured notifiers disabled

- **GIVEN** no configured notifier is enabled
- **WHEN** a clip would otherwise produce a notification
- **THEN** no notifier transport receives a send request

### Requirement: Plugin registration and discovery

HomeSec SHALL support registered source, filter, analyzer, storage, notifier, and alert-policy extensions selected by backend name. Backend names SHALL be unique within their extension category; duplicate registration and loading an unknown backend SHALL fail explicitly. Discovery SHALL import built-in plugin modules and installed modules declared in the `homesec.plugins` entry-point group so their registrations become available.

#### Scenario: Select a registered extension

- **GIVEN** an extension is registered under a backend name and category
- **WHEN** that backend is selected with valid configuration
- **THEN** HomeSec creates the registered extension for that category

#### Scenario: Duplicate or unknown backend

- **GIVEN** a backend name is already registered or a requested backend has no registration
- **WHEN** another extension registers that name in the same category or the unknown backend is loaded
- **THEN** the operation fails with an explicit registration or unknown-backend error

#### Scenario: Discover an installed external extension

- **GIVEN** an installed package declares a `homesec.plugins` entry point whose module registers an extension
- **WHEN** HomeSec performs plugin discovery
- **THEN** the declared module is imported and its registered backend becomes available

### Requirement: Plugin configuration and component contracts

Extensions SHALL declare a configuration schema and creation factory and conform to the existing interface for their category, including typed inputs/results and applicable asynchronous health/shutdown operations. HomeSec SHALL resolve applicable runtime configuration before schema validation, validate configuration before creation, and support validation without constructing the extension. Invalid extension configuration SHALL fail before creation.

#### Scenario: Invalid plugin configuration

- **GIVEN** a registered extension with a configuration schema
- **WHEN** its supplied configuration fails that schema
- **THEN** validation fails before the extension's factory is invoked

#### Scenario: Validate without constructing a plugin

- **GIVEN** a registered extension and its resolved configuration
- **WHEN** HomeSec performs configuration-only validation
- **THEN** the configuration is validated without constructing or starting the extension
