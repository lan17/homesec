# Backup Postgres Run

> 26 nodes · cohesion 0.13

## Key Concepts

- **test_api_maintenance_routes.py** (13 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **_StubBackupManager** (11 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **PostgresBackupRunRequest** (10 connections) — `src/homesec/maintenance/postgres_backup.py`
- **PostgresBackupStatus** (9 connections) — `src/homesec/maintenance/postgres_backup.py`
- **_client()** (7 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **_status()** (6 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **_StubMaintenanceApp** (6 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **test_postgres_backup_run_endpoint_rejects_busy_manager()** (6 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **test_postgres_backup_run_endpoint_accepts_manual_backup()** (5 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **test_postgres_backup_run_endpoint_reports_disabled_or_unavailable()** (5 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **test_postgres_backup_status_endpoint_returns_manager_status()** (5 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **.__init__()** (3 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **.__init__()** (2 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **bootstrap_mode()** (1 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **config()** (1 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **postgres_backup_manager()** (1 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **Tests for maintenance API routes.** (1 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **POST backup run should return 202 when single-flight accepts the request.** (1 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **POST backup run should return 409 when another backup is in flight.** (1 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **POST backup run should return clear API errors for disabled/unavailable backups.** (1 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **GET backup status should expose manager status fields.** (1 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **server_config()** (1 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **.request_backup_now()** (1 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **.status()** (1 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **Current postgres backup subsystem status.** (1 connections) — `src/homesec/maintenance/postgres_backup.py`
- *... and 1 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/maintenance/postgres_backup.py`
- `tests/homesec/test_api_maintenance_routes.py`

## Audit Trail

- EXTRACTED: 76 (75%)
- INFERRED: 25 (25%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*