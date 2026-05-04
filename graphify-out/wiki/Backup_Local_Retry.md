# Backup Local Retry

> 43 nodes · cohesion 0.11

## Key Concepts

- **PostgresBackupManager** (51 connections) — `src/homesec/maintenance/postgres_backup.py`
- **._run_backup_locked()** (15 connections) — `src/homesec/maintenance/postgres_backup.py`
- **._save_manifest()** (10 connections) — `src/homesec/maintenance/postgres_backup.py`
- **.start()** (9 connections) — `src/homesec/maintenance/postgres_backup.py`
- **redact_backup_text()** (9 connections) — `src/homesec/maintenance/postgres_backup.py`
- **._initialize_local_state()** (7 connections) — `src/homesec/maintenance/postgres_backup.py`
- **._retry_pending_storage_work_safely()** (7 connections) — `src/homesec/maintenance/postgres_backup.py`
- **._dump_database()** (6 connections) — `src/homesec/maintenance/postgres_backup.py`
- **._local_dir()** (6 connections) — `src/homesec/maintenance/postgres_backup.py`
- **._refresh_manifest_summary()** (5 connections) — `src/homesec/maintenance/postgres_backup.py`
- **.request_backup_now()** (5 connections) — `src/homesec/maintenance/postgres_backup.py`
- **._retry_pending_remote_deletes()** (5 connections) — `src/homesec/maintenance/postgres_backup.py`
- **._retry_pending_uploads()** (5 connections) — `src/homesec/maintenance/postgres_backup.py`
- **._run_backup()** (5 connections) — `src/homesec/maintenance/postgres_backup.py`
- **._scan_local_backups()** (5 connections) — `src/homesec/maintenance/postgres_backup.py`
- **._schedule_loop()** (5 connections) — `src/homesec/maintenance/postgres_backup.py`
- **.status()** (5 connections) — `src/homesec/maintenance/postgres_backup.py`
- **._upload_record()** (5 connections) — `src/homesec/maintenance/postgres_backup.py`
- **._load_manifest()** (4 connections) — `src/homesec/maintenance/postgres_backup.py`
- **._manifest_path()** (4 connections) — `src/homesec/maintenance/postgres_backup.py`
- **._prune_record()** (4 connections) — `src/homesec/maintenance/postgres_backup.py`
- **._prune_retention()** (4 connections) — `src/homesec/maintenance/postgres_backup.py`
- **._run_marked_backup()** (4 connections) — `src/homesec/maintenance/postgres_backup.py`
- **._build_artifact_name()** (3 connections) — `src/homesec/maintenance/postgres_backup.py`
- **._last_success_record()** (3 connections) — `src/homesec/maintenance/postgres_backup.py`
- *... and 18 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/maintenance/postgres_backup.py`

## Audit Trail

- EXTRACTED: 205 (92%)
- INFERRED: 17 (8%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*