# Path Build Backup

> 20 nodes · cohesion 0.12

## Key Concepts

- **build_backup_path** (4 connections) — `src/homesec/storage_paths.py`
- **build_clip_path** (4 connections) — `src/homesec/storage_paths.py`
- **_ManifestRecord** (3 connections) — `src/homesec/maintenance/postgres_backup.py`
- **maintenance.postgres_backup config** (3 connections) — `docs/postgres-backups.md`
- **docs/postgres-backups.md** (3 connections) — `docs/postgres-backups.md`
- **_normalize_dest_path** (3 connections) — `src/homesec/storage_paths.py`
- **_sanitize_segment** (3 connections) — `src/homesec/storage_paths.py`
- **src/homesec/storage_paths.py** (3 connections) — `src/homesec/storage_paths.py`
- **_BackupManifest** (2 connections) — `src/homesec/maintenance/postgres_backup.py`
- **build_backup_path** (2 connections) — `src/homesec/maintenance/postgres_backup.py`
- **paths.backups_dir** (2 connections) — `docs/postgres-backups.md`
- **build_artifact_path** (2 connections) — `src/homesec/storage_paths.py`
- **Clip** (2 connections) — `src/homesec/storage_paths.py`
- **StoragePathsConfig** (2 connections) — `src/homesec/storage_paths.py`
- **manifest.json** (1 connections) — `src/homesec/maintenance/postgres_backup.py`
- **redact_backup_text** (1 connections) — `src/homesec/maintenance/postgres_backup.py`
- **manifest.json** (1 connections) — `docs/postgres-backups.md`
- **pg_dump custom-format backups** (1 connections) — `docs/postgres-backups.md`
- **pg_restore** (1 connections) — `docs/postgres-backups.md`
- **PurePosixPath** (1 connections) — `src/homesec/storage_paths.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `docs/postgres-backups.md`
- `src/homesec/maintenance/postgres_backup.py`
- `src/homesec/storage_paths.py`

## Audit Trail

- EXTRACTED: 36 (82%)
- INFERRED: 8 (18%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*