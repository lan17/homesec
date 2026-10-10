# Spec Delta

## Purpose

Describe how users browse recorded events, retain their browsing context, inspect event details, and access recorded video through HomeSec.

## Current limitations

Identity-based detail and media-token lookup do not exclude deleted records, whose metadata and storage URI can remain present. Such a record can still receive a media URL; playback returns HTTP 502 if storage retrieval fails, but can succeed if the stored object remains retrievable. A clip lookup error can also be treated as missing state and produce HTTP 404 after the earlier database availability check passes. These behaviors remain review gaps rather than permanent contracts; see [Deleted-record visibility and media access and Clip lookup failure masking in the design review gaps](../../design.md#review-gaps-and-documentation-disagreements).

## ADDED Requirements

### Requirement: Event queries support filtering and cursor pagination

HomeSec SHALL support clip queries filtered by camera, processing status, alert outcome, detection outcome, risk, activity, and creation-time bounds. Queries without an explicit status filter SHALL exclude deleted clips; an explicit deleted-status filter SHALL include matching deleted records. Results SHALL use newest-first cursor pagination, with clip identity breaking equal-time ties, and expose whether another page exists.

#### Scenario: Continue to the next result page

- **GIVEN** a valid query with more matching events than its page limit
- **WHEN** the client requests a page and then supplies its returned cursor
- **THEN** HomeSec returns the continuation page and a next cursor only when further results exist

#### Scenario: Reject invalid time bounds or a malformed cursor

- **GIVEN** a request contains a timezone-naive time bound, an inverted time interval, or a malformed cursor
- **WHEN** the client queries events
- **THEN** HomeSec returns an error identifying the invalid query rather than silently accepting it

#### Scenario: Exclude deleted events from the default list

- **GIVEN** readable clip state contains both deleted and non-deleted events that otherwise match the query
- **WHEN** a client lists events without an explicit status filter
- **THEN** the results exclude the deleted events

#### Scenario: Query deleted records explicitly

- **GIVEN** readable clip state contains deleted events matching the remaining query filters
- **WHEN** a client lists events with an explicit deleted-status filter
- **THEN** the matching deleted records are returned

### Requirement: Event browsing preserves filter context in the URL

The event UI SHALL represent applied filters in the URL, preserve filters when navigating result pages, and discard stale cursor history when the filter selection changes. The default detection filter SHALL show detected events, with an explicit selection available for any detection outcome.

#### Scenario: Reopen an event search

- **GIVEN** an event-list URL containing supported filter parameters
- **WHEN** the user opens or reloads that URL
- **THEN** the UI restores the corresponding search rather than relying only on prior in-memory form state

#### Scenario: Change filters after paging

- **GIVEN** the user has navigated beyond the first page
- **WHEN** the user applies a different filter selection
- **THEN** the next query uses the new filters without the former search's cursor history

#### Scenario: Include events without detections

- **GIVEN** the default event search filters for detected events
- **WHEN** the user selects any detection outcome
- **THEN** the URL retains that explicit choice and the API query omits the detection restriction

### Requirement: Event views present available summaries and inline playback

The event UI SHALL present camera, time, available activity summary, risk, detected objects, and alert outcome. Event cards SHALL offer inline video controls when a media URL is available and expandable technical details. Empty search results SHALL have an explicit no-results presentation.

#### Scenario: Browse an event with playable media

- **GIVEN** an event list contains an event with a resolved media URL
- **WHEN** its card is displayed
- **THEN** the card provides inline video controls alongside event information

#### Scenario: Expand event details

- **GIVEN** a displayed event card
- **WHEN** the user opens its details
- **THEN** the card exposes additional event information, including its identity, without requiring navigation away from the list

#### Scenario: No events match the search

- **GIVEN** a successful query with no matching events
- **WHEN** the event list is displayed
- **THEN** the UI explicitly reports that no results were found

### Requirement: Event detail lookup distinguishes known and missing events

When the clip-state lookup succeeds, HomeSec SHALL return available processing, analysis, and alert details for a known clip identity and return HTTP 404 for an unknown identity. The detail UI SHALL keep video and event summary accessible and limit previous/next navigation to an available client result window.

#### Scenario: Retrieve a known event

- **GIVEN** a stored clip with analysis and alert metadata
- **WHEN** a client retrieves its detail
- **THEN** the response includes the available analysis and alert fields associated with that clip

#### Scenario: Open details without a result window

- **GIVEN** the detail page has no loaded client result window
- **WHEN** the page is displayed
- **THEN** previous and next event controls are disabled

#### Scenario: Retrieve an unknown event

- **GIVEN** the clip-state lookup succeeds and the requested clip identity does not exist
- **WHEN** a client requests its detail
- **THEN** HomeSec returns HTTP 404 with the clip-not-found reason

### Requirement: Media URLs respect authentication and clip availability

HomeSec SHALL return a direct media URL when authentication is disabled and a clip-scoped, expiring URL when it is enabled. Token issuance SHALL require normal API authentication when enabled. Media playback SHALL reject invalid, expired, or wrong-clip tokens and permit valid media tokens or a valid API key.

#### Scenario: Request media without authentication enabled

- **GIVEN** an existing clip with a storage URI and server authentication disabled
- **WHEN** a client requests its media URL
- **THEN** HomeSec returns a direct URL with no token expiry

#### Scenario: Request authorized media access

- **GIVEN** authentication is enabled and an existing clip has a storage URI
- **WHEN** a client uses a valid API key to request its media URL
- **THEN** HomeSec returns a tokenized URL with an expiry for that clip

#### Scenario: Attempt to reuse authorization for another clip

- **GIVEN** a valid token issued for one clip
- **WHEN** it is used to request another clip's media
- **THEN** HomeSec rejects the request

#### Scenario: Attempt to mint a token with a playback token

- **GIVEN** authentication is enabled and the client supplies only a media playback token
- **WHEN** it requests a new media token
- **THEN** HomeSec rejects the token-issuance request

### Requirement: Media delivery exposes availability and retrieval failures

HomeSec SHALL serve available clip media with an inline content disposition. When the clip-state lookup succeeds, unknown clips SHALL return HTTP 404 and clips without a storage URI SHALL return HTTP 409. Storage retrieval failures SHALL return HTTP 502 with the media-fetch-failed reason.

#### Scenario: Play stored media

- **GIVEN** a known clip whose configured storage object can be retrieved
- **WHEN** an authorized client requests clip media
- **THEN** HomeSec returns the media for inline playback

#### Scenario: Request media before storage is available

- **GIVEN** a known clip without a storage URI
- **WHEN** a client requests its media or media URL
- **THEN** HomeSec returns HTTP 409 with the media-unavailable reason

#### Scenario: Storage retrieval fails

- **GIVEN** a known clip has a storage URI but retrieval fails
- **WHEN** an authorized client requests its media
- **THEN** HomeSec returns HTTP 502 with the media-fetch-failed reason

### Requirement: Tokenized playback supports bounded source refresh

The event UI SHALL schedule media-token refresh before expiry and allow one automatic token-source refresh after a playback error for an unchanged media source. Direct, non-token playback SHALL not trigger that token refresh path.

#### Scenario: A media token approaches expiry

- **GIVEN** a mounted event playback view with a tokenized URL and an expiry
- **WHEN** the scheduled refresh point before expiry is reached
- **THEN** the UI requests a refreshed media URL

#### Scenario: Playback repeatedly fails on an unchanged tokenized URL

- **GIVEN** a tokenized media source has already used its automatic error refresh
- **WHEN** playback fails again without the source changing
- **THEN** the UI does not repeatedly request automatic error-triggered refreshes for that unchanged source

#### Scenario: Direct playback fails

- **GIVEN** a direct media URL without a token expiry
- **WHEN** playback fails
- **THEN** the UI does not attempt the token-source error refresh path
