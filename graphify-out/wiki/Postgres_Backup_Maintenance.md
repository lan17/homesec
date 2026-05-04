# Postgres Backup Maintenance

> 8 nodes · cohesion 0.32

## Key Concepts

- **PostgresBackupRunResponse** (6 connections) — `src/homesec/api/routes/maintenance.py`
- **PostgresBackupStatusResponse** (6 connections) — `src/homesec/api/routes/maintenance.py`
- **maintenance.py** (5 connections) — `src/homesec/api/routes/maintenance.py`
- **get_postgres_backup_status()** (3 connections) — `src/homesec/api/routes/maintenance.py`
- **run_postgres_backup_now()** (3 connections) — `src/homesec/api/routes/maintenance.py`
- **Maintenance control-plane endpoints.** (1 connections) — `src/homesec/api/routes/maintenance.py`
- **Return current Postgres backup subsystem status.** (1 connections) — `src/homesec/api/routes/maintenance.py`
- **Trigger a manual Postgres backup.** (1 connections) — `src/homesec/api/routes/maintenance.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/api/routes/maintenance.py`

## Audit Trail

- EXTRACTED: 20 (77%)
- INFERRED: 6 (23%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*