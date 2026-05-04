# Clipevent Clipstatedata Postgresstatestore

> 103 nodes · cohesion 0.06

## Key Concepts

- **ClipStateData** (95 connections) — `src/homesec/models/clip.py`
- **PostgresStateStore** (79 connections) — `src/homesec/state/postgres.py`
- **NoopEventStore** (46 connections) — `src/homesec/state/postgres.py`
- **NoopStateStore** (39 connections) — `src/homesec/state/postgres.py`
- **VLMSkipReason** (34 connections) — `src/homesec/models/enums.py`
- **PostgresEventStore** (33 connections) — `src/homesec/state/postgres.py`
- **EventType** (32 connections) — `src/homesec/models/enums.py`
- **ClipEvent** (29 connections) — `src/homesec/models/events.py`
- **_StatusTransition** (29 connections) — `src/homesec/repository/clip_repository.py`
- **Base** (28 connections) — `src/homesec/state/postgres.py`
- **ClipEvent** (28 connections) — `src/homesec/state/postgres.py`
- **ClipState** (28 connections) — `src/homesec/state/postgres.py`
- **AlertDecisionMadeEvent** (18 connections) — `src/homesec/models/events.py`
- **ClipRecordedEvent** (18 connections) — `src/homesec/models/events.py`
- **UploadCompletedEvent** (18 connections) — `src/homesec/models/events.py`
- **events.py** (18 connections) — `src/homesec/models/events.py`
- **ClipDeletedEvent** (17 connections) — `src/homesec/models/events.py`
- **FilterCompletedEvent** (17 connections) — `src/homesec/models/events.py`
- **NotificationSentEvent** (17 connections) — `src/homesec/models/events.py`
- **ClipRecheckedEvent** (16 connections) — `src/homesec/models/events.py`
- **FilterFailedEvent** (16 connections) — `src/homesec/models/events.py`
- **FilterStartedEvent** (16 connections) — `src/homesec/models/events.py`
- **NotificationFailedEvent** (16 connections) — `src/homesec/models/events.py`
- **UploadFailedEvent** (16 connections) — `src/homesec/models/events.py`
- **UploadStartedEvent** (16 connections) — `src/homesec/models/events.py`
- *... and 78 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/models/clip.py`
- `src/homesec/models/enums.py`
- `src/homesec/models/events.py`
- `src/homesec/repository/clip_repository.py`
- `src/homesec/state/postgres.py`
- `tests/homesec/conftest.py`
- `tests/homesec/test_event_store.py`
- `tests/homesec/test_state_store.py`

## Audit Trail

- EXTRACTED: 250 (27%)
- INFERRED: 666 (73%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*