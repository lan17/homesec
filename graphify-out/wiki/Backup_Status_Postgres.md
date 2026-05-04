# Backup Status Postgres

> 19 nodes · cohesion 0.15

## Key Concepts

- **_StubBackupManager** (10 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **_StubMaintenanceApp** (8 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **_client** (6 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **_status** (4 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **test_postgres_backup_run_endpoint_accepts_manual_backup** (3 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **test_postgres_backup_run_endpoint_rejects_busy_manager** (3 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **test_postgres_backup_status_endpoint_returns_manager_status** (3 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **postgres_backup_manager** (2 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **PostgresBackupRunRequest** (2 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **PostgresBackupStatus** (2 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **bootstrap_mode** (1 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **config** (1 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **create_app** (1 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **ensure_stub_ui_dist** (1 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **FastAPIServerConfig** (1 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **request_backup_now** (1 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **server_config** (1 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **status** (1 connections) — `tests/homesec/test_api_maintenance_routes.py`
- **TestClient** (1 connections) — `tests/homesec/test_api_maintenance_routes.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `tests/homesec/test_api_maintenance_routes.py`

## Audit Trail

- EXTRACTED: 46 (88%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 6 (12%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*