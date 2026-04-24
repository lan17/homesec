# Postgres Backups

HomeSec can create periodic Postgres backups with `pg_dump --format=custom`.
Backups are disabled by default. To opt in, add:

```yaml
maintenance:
  postgres_backup:
    enabled: true
    interval: 24h
    keep_count: 5
    local_dir: /data/storage/backups/postgres
    upload:
      enabled: true
      storage_backend: primary
```

The backup manager writes `manifest.json` next to local dump files and uses the
configured storage `paths.backups_dir` for uploads. The manifest is the recovery
source for retention and pending remote deletes; retention state is not stored
only in Postgres.

The bundled Docker Compose setup persists `/data/storage`, so Docker users
should keep `local_dir` under `/data/storage` or mount another durable directory.
For non-Docker installs, `./backups/postgres` is fine if that path is on durable
storage.

Restore example:

```bash
createdb homesec_restore
pg_restore --dbname=homesec_restore /data/storage/backups/postgres/homesec-postgres-YYYYMMDD-HHMMSS-ffffff.dump
```

Use the same Postgres credentials and network access you would use for normal
database administration. Do not paste DSNs with passwords into tickets or logs.
