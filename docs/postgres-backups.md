# Postgres Backups

HomeSec can create periodic custom-format dumps of its configured state-store
database. Backups are disabled by default. To opt in, add:

```yaml
maintenance:
  postgres_backup:
    enabled: true
    interval: 24h
    keep_count: 5
    local_dir: ./backups/postgres
    upload:
      enabled: true
      storage_backend: primary
```

`interval` accepts a number followed by `s`, `m`, `h`, or `d` and must be at least
15 minutes. `keep_count` must be at least 1. Other defaults are a 1,800-second
dump timeout and compression level 6 (range 0–9). Uploading is enabled by default
when backups are enabled; set `upload.enabled: false` for local-only dumps.
`primary` selects the application's existing storage backend. Uploaded paths use
`storage.paths.backups_dir` (default `backups`); they need not be cloud storage.

## Prerequisites and local storage

The HomeSec process needs `pg_dump` in `PATH`, access to the configured state-store
database, and a writable local backup directory. It uses `state_store.dsn_env`
when configured, otherwise `state_store.dsn`; credentials are supplied to
`pg_dump` through its environment. The Docker image includes Postgres 16 client
tools for the bundled Postgres 16 service. For another server version, select a
compatible client. A missing executable makes backups unavailable without
preventing HomeSec startup; executable availability does not verify connectivity
or sufficient database permissions.

`local_dir` defaults to `./backups/postgres` and is resolved from the HomeSec
process working directory. In the Docker image that is `/app`, whose default
backup directory is not persisted by the bundled Compose mounts. For that setup,
set `local_dir: /data/storage/backups/postgres`, which maps to the host's
`./storage/backups/postgres`, or mount another durable directory writable by the
`homesec` container user.

The manager sets the backup directory's permissions to `0700` and new dump and
manifest files to `0600`. Final artifacts are named
`homesec-postgres-YYYYMMDD-HHMMSS-ffffff.dump` using UTC timestamps, with a numeric
suffix on collisions. Failed/timed-out dumps do not become final artifacts;
shutdown cancellation terminates an active dump and removes its temporary file.

Persist the whole backup directory, including `manifest.json`. The manifest
tracks local artifacts, upload results, and pending remote deletes. Startup can
rebuild records by scanning recognized dump filenames when the manifest is
missing or invalid, but that scan cannot recover unknown remote-object history
or verify dump contents. These database dumps do not back up application
configuration/credential files or recorded media.

## Scheduling, status, and retention

An enabled manager schedules an initial or overdue dump after a short startup
jitter, then normally schedules the next run from dump completion. Failed runs
normally retry after 15 minutes. Only one backup run is admitted at a time.

The maintenance API, subject to API-key authentication when enabled, exposes
`GET /api/v1/maintenance/postgres-backups/status` and
`POST /api/v1/maintenance/postgres-backups/run` in normal mode. A manual request's
HTTP 202 means accepted work; inspect status for completion, upload errors, next
run time, and pending remote deletes. `available` means the subsystem can attempt
a backup. `last_success_at` is not a restore verification or upload guarantee.

Retention prunes oldest eligible completed backups beyond `keep_count`, including
known uploaded objects. After ordinary upload failures, pending local dumps have
a separate `keep_count` cap while prior uploaded backups are preserved. A failed
remote delete remains pending for retry. Retention does not bound all backup
disk usage or guarantee an off-site copy: cancellation with an uncertain upload
outcome can preserve extra local files, and manifest-saving or housekeeping
failures have [documented limitations](reference.md#operations-maintenance).

## Restore

Use `pg_restore` for these custom-format dumps. The example below creates and
writes a separate recovery database; configure normal libpq authentication and
connection settings first, and use an actual retained artifact path:

```bash
createdb homesec_restore
pg_restore --exit-on-error --dbname=homesec_restore /path/to/homesec-postgres-YYYYMMDD-HHMMSS-ffffff.dump
```

The restore account needs permission to create the target database and restore
its objects; source owner/privilege roles must exist or be deliberately remapped
for the recovery environment. Check restore errors and inspect the restored data
before selecting it as a HomeSec database. A recorded dump success, recovered
filename, or uploaded URI does not establish a successful restore. Do not paste
DSNs with passwords into tickets or logs.

See the [maintained operations contract](../openspec/specs/operations-maintenance/spec.md#requirement-backup-opt-in-and-prerequisites)
and [reference limitations](reference.md#operations-maintenance) for the current
behavior and verification boundaries.
