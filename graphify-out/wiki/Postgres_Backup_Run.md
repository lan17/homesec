# Postgres Backup Run

> 8 nodes · cohesion 0.25

## Key Concepts

- **run_postgres_backup_now** (4 connections) — `src/homesec/api/routes/maintenance.py`
- **get_homesec_app** (3 connections) — `src/homesec/api/routes/maintenance.py`
- **get_postgres_backup_status** (2 connections) — `src/homesec/api/routes/maintenance.py`
- **APIError** (1 connections) — `src/homesec/api/routes/maintenance.py`
- **APIErrorCode** (1 connections) — `src/homesec/api/routes/maintenance.py`
- **src/homesec/api/routes/maintenance.py** (1 connections) — `src/homesec/api/routes/maintenance.py`
- **PostgresBackupRunResponse** (1 connections) — `src/homesec/api/routes/maintenance.py`
- **PostgresBackupStatusResponse** (1 connections) — `src/homesec/api/routes/maintenance.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/api/routes/maintenance.py`

## Audit Trail

- EXTRACTED: 14 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*