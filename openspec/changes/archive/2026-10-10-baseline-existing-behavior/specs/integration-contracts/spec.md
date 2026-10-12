# Spec Delta

## Purpose

Describe HomeSec's existing contracts for API clients, notification recipients, and plugin authors so integrations can depend on observable behavior and the existing authoritative schemas.

## ADDED Requirements

### Requirement: API key authentication

API routes protected by API-key authentication SHALL accept the configured key in an `Authorization: Bearer <key>` header when authentication is enabled. Missing or incorrect bearer credentials SHALL return HTTP 401 with `UNAUTHORIZED`. A missing configured key SHALL return HTTP 500 with `API_KEY_NOT_CONFIGURED`. Disabling authentication SHALL remove the API-key requirement. Public health summaries SHALL remain accessible without credentials.

#### Scenario: Protected request lacks valid credentials

- **GIVEN** API authentication is enabled and a key is configured
- **WHEN** a client requests camera configuration with no bearer header or an incorrect key
- **THEN** the request returns HTTP 401 with `error_code` equal to `UNAUTHORIZED`

#### Scenario: Valid bearer credentials

- **GIVEN** API authentication is enabled and a key is configured
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

API exceptions and request-validation failures SHALL return a JSON envelope containing textual `detail` and `error_code`, with applicable additional fields preserved. Invalid request parameters SHALL return HTTP 422 with `REQUEST_VALIDATION_FAILED` and validation details. Unexpected route exceptions SHALL return HTTP 500 with generic detail and `INTERNAL_SERVER_ERROR` rather than expose the exception text.

#### Scenario: Route reports a conflict with additional context

- **GIVEN** a route rejects an operation with a specific conflict code and additional context
- **WHEN** the API returns that exception response
- **THEN** the HTTP conflict status, `detail`, `error_code`, and additional context are preserved

#### Scenario: Invalid request parameters

- **GIVEN** an API request contains parameters that fail request-schema validation
- **WHEN** the request is processed
- **THEN** it returns HTTP 422 with `REQUEST_VALIDATION_FAILED` and `validation_errors`

#### Scenario: Unexpected route exception

- **GIVEN** a route encounters an unexpected exception
- **WHEN** the API produces its response
- **THEN** it returns HTTP 500 with `detail` equal to `Internal server error` and `error_code` equal to `INTERNAL_SERVER_ERROR`

### Requirement: Generated API contract

HomeSec SHALL expose an OpenAPI description derived from its registered API routes and request/response models, and SHALL support exporting that description without starting cameras or the application runtime. Exact request and response shapes SHALL be obtained from the existing generated OpenAPI contract instead of a separate hand-maintained field inventory.

#### Scenario: Generate the contract without application startup

- **GIVEN** HomeSec is installed without a running camera pipeline
- **WHEN** a client developer exports the OpenAPI contract to a selected output path
- **THEN** the file contains valid JSON describing versioned API routes and their model schemas
- **AND** contract generation does not start the camera pipeline

### Requirement: Notification payload meaning

Notifications SHALL use the existing alert payload schema, identifying the clip and camera, decision reason, timestamp, and a deduplication key equal to the clip identifier. Storage/view links and analysis fields SHALL be nullable when unavailable. The payload SHALL indicate upload and analysis failures independently so recipients can distinguish missing enrichment from available clip identity. The key SHALL identify the alert; it SHALL NOT imply exactly-once delivery.

#### Scenario: Notification includes available analysis and media references

- **GIVEN** a notification is selected for a clip with analysis and an uploaded media reference
- **WHEN** the payload is delivered to a notifier
- **THEN** it contains the clip/camera identity, available analysis and references, decision reason, and a deduplication key matching the clip identifier

#### Scenario: Notification without successful upload or analysis

- **GIVEN** a notification is selected despite an upload or analysis failure
- **WHEN** the payload is delivered
- **THEN** the corresponding failure flag is true and unavailable links or analysis values are null
- **AND** the clip identity and decision reason remain present

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

HomeSec SHALL send a selected alert to each enabled configured notifier. Disabled notifiers SHALL not receive alerts. A delivery failure from one notifier SHALL not prevent delivery attempts to the others, and the failed delivery SHALL remain observable as a failure. When no notifier is enabled, notification delivery SHALL have no transport side effects.

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
