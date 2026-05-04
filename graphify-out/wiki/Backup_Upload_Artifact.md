# Backup Upload Artifact

> 54 nodes · cohesion 0.08

## Key Concepts

- **test_postgres_backup.py** (35 connections) — `tests/homesec/test_postgres_backup.py`
- **_Clock** (23 connections) — `tests/homesec/test_postgres_backup.py`
- **_config()** (21 connections) — `tests/homesec/test_postgres_backup.py`
- **_WritingBackupManager** (16 connections) — `tests/homesec/test_postgres_backup.py`
- **MockStorage** (10 connections)
- **PostgresBackupManager** (10 connections)
- **_FixedClock** (9 connections) — `tests/homesec/test_postgres_backup.py`
- **_WaitingBackupManager** (9 connections) — `tests/homesec/test_postgres_backup.py`
- **test_backup_cancellation_terminates_pg_dump_and_removes_temp_artifact()** (7 connections) — `tests/homesec/test_postgres_backup.py`
- **test_backup_timeout_removes_temp_artifact()** (7 connections) — `tests/homesec/test_postgres_backup.py`
- **test_cancelled_remote_delete_preserves_retry_tombstone()** (6 connections) — `tests/homesec/test_postgres_backup.py`
- **test_cancelled_upload_is_retried_before_retention_prunes_record()** (6 connections) — `tests/homesec/test_postgres_backup.py`
- **test_cancelled_upload_record_is_not_pruned_during_later_upload_outage()** (6 connections) — `tests/homesec/test_postgres_backup.py`
- **test_manifest_recovery_marks_local_backups_pending_upload_when_enabled()** (6 connections) — `tests/homesec/test_postgres_backup.py`
- **test_manifest_recovery_reconciles_running_record_with_local_artifact()** (6 connections) — `tests/homesec/test_postgres_backup.py`
- **test_manifest_recovery_scans_local_backups()** (6 connections) — `tests/homesec/test_postgres_backup.py`
- **test_manual_backup_requests_are_single_flight_before_task_starts()** (6 connections) — `tests/homesec/test_postgres_backup.py`
- **test_pending_remote_delete_retries_on_successful_backup()** (6 connections) — `tests/homesec/test_postgres_backup.py`
- **test_scheduled_backup_backs_off_when_manual_backup_is_active()** (6 connections) — `tests/homesec/test_postgres_backup.py`
- **test_scheduler_rechecks_next_run_after_sleep_deadline()** (6 connections) — `tests/homesec/test_postgres_backup.py`
- **test_startup_remote_delete_retry_does_not_block_manager_start()** (6 connections) — `tests/homesec/test_postgres_backup.py`
- **test_startup_storage_retry_serializes_with_manual_backup()** (6 connections) — `tests/homesec/test_postgres_backup.py`
- **test_successful_backup_clears_previous_upload_failure_status()** (6 connections) — `tests/homesec/test_postgres_backup.py`
- **test_successful_backups_do_not_overwrite_same_timestamp_artifacts()** (6 connections) — `tests/homesec/test_postgres_backup.py`
- **test_successful_backups_upload_and_prune_oldest()** (6 connections) — `tests/homesec/test_postgres_backup.py`
- *... and 29 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `tests/homesec/test_postgres_backup.py`

## Audit Trail

- EXTRACTED: 260 (94%)
- INFERRED: 18 (6%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*